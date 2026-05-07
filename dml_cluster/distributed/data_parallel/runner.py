from __future__ import annotations

import queue
import socket
import threading
import time
from datetime import timedelta
from math import ceil
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from ..common.power import PowerSampler
from ..common.network import configure_gloo_socket_ifname
from ..datasets import load_dataset, shard_dataset
from ..models import CifarCnn, build_model
from .compression import TopKCompressor

DEFAULT_DIST_TIMEOUT_SECONDS = 60.0
DATASET_NAME = "cifar10"
MODEL_NAME = "cifar_cnn"
IMAGE_SIZE = 32
CLASSES = 10
LR = 1e-3
MOMENTUM = 0.0
WEIGHT_DECAY = 0.0


def choose_device(accelerator: str) -> torch.device:
    requested = accelerator.lower()
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
    if requested == "mps" and mps_backend is not None and mps_backend.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        mps = getattr(torch, "mps", None)
        if mps is not None and hasattr(mps, "synchronize"):
            torch.mps.synchronize()


def _allreduce_grads(model: CifarCnn, world_size: int, device: torch.device) -> None:
    params = [param for param in model.parameters() if param.grad is not None]
    if not params:
        return

    flat = torch.cat([param.grad.detach().cpu().view(-1) for param in params])
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    flat /= world_size

    offset = 0
    for param in params:
        size = param.grad.numel()
        param.grad.copy_(flat[offset : offset + size].view_as(param.grad).to(device))
        offset += size


def _topk_allreduce_grads(
    model: torch.nn.Module,
    world_size: int,
    device: torch.device,
    compressor: TopKCompressor,
) -> dict[str, float]:
    raw_numel = 0
    compressed_numel = 0
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        payload = compressor.compress(name, param.grad)
        raw_numel += payload.original_numel
        compressed_numel += payload.compressed_numel

        local_indices = payload.indices.to(dtype=torch.int64).contiguous()
        local_values = payload.values.to(dtype=torch.float32).contiguous()
        gathered_indices = [torch.empty_like(local_indices) for _ in range(world_size)]
        gathered_values = [torch.empty_like(local_values) for _ in range(world_size)]
        dist.all_gather(gathered_indices, local_indices)
        dist.all_gather(gathered_values, local_values)

        reduced = torch.zeros(payload.original_numel, dtype=torch.float32)
        for indices, values in zip(gathered_indices, gathered_values):
            reduced[indices] += values
        reduced /= world_size
        param.grad.copy_(reduced.reshape(payload.shape).to(device=device, dtype=param.grad.dtype))

    return {
        "raw_gradient_numel": float(raw_numel),
        "compressed_gradient_numel": float(compressed_numel),
        "compression_ratio": float(raw_numel / max(1, compressed_numel)),
    }


def _build_loader(
    shard: tuple[Any, Any],
    batch_size: int,
    batches_per_epoch: int,
) -> DataLoader:
    images_np, labels_np = shard
    images = torch.from_numpy(images_np.copy()).float().permute(0, 3, 1, 2) / 255.0
    labels = torch.from_numpy(labels_np.copy()).long()
    usable = min(len(images), max(1, batches_per_epoch) * max(1, batch_size))
    return DataLoader(
        TensorDataset(images[:usable], labels[:usable]),
        batch_size=max(1, batch_size),
        shuffle=True,
        drop_last=True,
    )


def _build_local_dataset_loader(
    shard_config: dict[str, Any],
    batch_size: int,
    batches_per_epoch: int,
) -> DataLoader:
    from pathlib import Path

    dataset_samples = int(shard_config.get("dataset_samples") or 0)
    spec = load_dataset(
        name=str(shard_config["dataset"]),
        project_dir=Path(str(shard_config["project_dir"])),
        download=bool(shard_config.get("download")),
        max_samples=dataset_samples,
        image_size=int(shard_config.get("image_size") or IMAGE_SIZE),
    )
    start = int(shard_config.get("start") or 0)
    stop = int(shard_config.get("stop") or len(spec.train))
    subset = shard_dataset(spec.train, start, stop)
    usable = min(len(subset), max(1, batches_per_epoch) * max(1, batch_size))
    if usable < len(subset):
        subset = shard_dataset(subset, 0, usable)
    return DataLoader(
        subset,
        batch_size=max(1, batch_size),
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )


def _build_eval_loader(
    shard_config: dict[str, Any],
    batch_size: int,
    eval_batches: int,
) -> DataLoader | None:
    if eval_batches <= 0:
        return None
    from pathlib import Path

    spec = load_dataset(
        name=str(shard_config["dataset"]),
        project_dir=Path(str(shard_config["project_dir"])),
        download=bool(shard_config.get("download")),
        max_samples=0,
        image_size=int(shard_config.get("image_size") or IMAGE_SIZE),
    )
    if spec.val is None:
        return None
    usable = min(len(spec.val), max(1, eval_batches) * max(1, batch_size))
    subset = shard_dataset(spec.val, 0, usable)
    return DataLoader(
        subset,
        batch_size=max(1, batch_size),
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )


def _evaluate(
    model: torch.nn.Module,
    loader: DataLoader | None,
    device: torch.device,
) -> dict[str, float | int]:
    if loader is None:
        return {"val_loss": 0.0, "val_acc": 0.0, "val_top5_acc": 0.0, "val_samples": 0}

    model.eval()
    total_loss = 0.0
    total_samples = 0
    correct_top1 = 0
    correct_top5 = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            batch_samples = int(images.shape[0])
            total_loss += float(loss.detach().cpu()) * batch_samples
            total_samples += batch_samples
            correct_top1 += int((outputs.argmax(dim=1) == labels).sum().detach().cpu())
            topk = min(5, outputs.shape[1])
            correct_top5 += int(
                (outputs.topk(topk, dim=1).indices == labels.view(-1, 1)).any(dim=1).sum().detach().cpu()
            )
    model.train()
    return {
        "val_loss": total_loss / max(1, total_samples),
        "val_acc": correct_top1 / max(1, total_samples),
        "val_top5_acc": correct_top5 / max(1, total_samples),
        "val_samples": total_samples,
    }


def _broadcast_model(model: CifarCnn, device: torch.device) -> None:
    for parameter in model.parameters():
        buffer = parameter.data.detach().cpu().clone().contiguous()
        dist.broadcast(buffer, src=0)
        parameter.data.copy_(buffer.to(device))


def _run_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    world_size: int,
    device: torch.device,
    batches_per_epoch: int,
    stop_event: threading.Event,
    compression: str,
    compressor: TopKCompressor | None,
    rank: int,
    straggler_rank: int,
    straggler_delay_seconds: float,
) -> dict[str, Any]:
    model.train()
    total_loss = 0.0
    batch_count = 0
    sample_count = 0
    final_batch_loss = 0.0
    raw_gradient_numel = 0.0
    compressed_gradient_numel = 0.0
    compression_ratio = 1.0
    straggler_delay_total = 0.0

    for images_batch, labels_batch in loader:
        if batch_count >= batches_per_epoch or stop_event.is_set():
            break
        images_batch = images_batch.to(device)
        labels_batch = labels_batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(images_batch), labels_batch)
        if straggler_rank == rank and straggler_delay_seconds > 0:
            time.sleep(straggler_delay_seconds)
            straggler_delay_total += straggler_delay_seconds
        loss.backward()
        if compression == "topk":
            if compressor is None:
                raise RuntimeError("topk compression selected without compressor")
            compression_stats = _topk_allreduce_grads(model, world_size, device, compressor)
            raw_gradient_numel += compression_stats["raw_gradient_numel"]
            compressed_gradient_numel += compression_stats["compressed_gradient_numel"]
            compression_ratio = compression_stats["compression_ratio"]
        else:
            _allreduce_grads(model, world_size, device)
        optimizer.step()

        batch_loss = float(loss.detach().cpu())
        batch_samples = int(images_batch.shape[0])
        total_loss += batch_loss * batch_samples
        sample_count += batch_samples
        final_batch_loss = batch_loss
        batch_count += 1

    local_loss = total_loss / max(1, sample_count)
    loss_tensor = torch.tensor([local_loss], dtype=torch.float32)
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
    samples_tensor = torch.tensor([sample_count], dtype=torch.float32)
    dist.all_reduce(samples_tensor, op=dist.ReduceOp.SUM)
    return {
        "loss": float((loss_tensor / world_size).item()),
        "batches": batch_count,
        "samples": sample_count,
        "global_samples": int(samples_tensor.item()),
        "final_batch_loss": final_batch_loss,
        "device": str(device),
        "raw_gradient_numel": raw_gradient_numel,
        "compressed_gradient_numel": compressed_gradient_numel,
        "compression_ratio": compression_ratio if compression == "topk" else 1.0,
        "straggler_delay_total_seconds": straggler_delay_total,
    }


def run_training(
    rank: int,
    world_size: int,
    master_addr: str,
    dist_port: int,
    shard: tuple[Any, Any],
    accelerator: str,
    batch_size: int,
    batches_per_epoch: int,
    epochs: int,
    result_q: queue.SimpleQueue[dict[str, Any]],
    stop_event: threading.Event | None = None,
    timeout_seconds: float = DEFAULT_DIST_TIMEOUT_SECONDS,
) -> None:
    stop = stop_event or threading.Event()
    device = choose_device(accelerator)
    timeout = timedelta(seconds=max(1.0, timeout_seconds))
    current_epoch = 0
    if isinstance(shard, dict):
        model_name = str(shard.get("model") or MODEL_NAME)
        dataset_name = str(shard.get("dataset") or DATASET_NAME)
        classes = int(shard.get("classes") or CLASSES)
        image_size = int(shard.get("image_size") or IMAGE_SIZE)
        dataset_samples = int(shard.get("dataset_samples") or 0)
        amp = bool(shard.get("amp"))
        optimizations = str(shard.get("optimizations") or "none")
        compression = str(shard.get("compression") or "none")
        compress_ratio = float(shard.get("compress_ratio") or 0.01)
        straggler_rank = int(shard.get("straggler_rank") or -1)
        straggler_delay_seconds = float(shard.get("straggler_delay_seconds") or 0.0)
    else:
        model_name = MODEL_NAME
        dataset_name = DATASET_NAME
        classes = CLASSES
        image_size = IMAGE_SIZE
        dataset_samples = int(shard[0].shape[0]) if shard and hasattr(shard[0], "shape") else 0
        amp = False
        optimizations = "none"
        compression = "none"
        compress_ratio = 0.01
        straggler_rank = -1
        straggler_delay_seconds = 0.0

    gloo_ifname = configure_gloo_socket_ifname(master_addr)
    if gloo_ifname:
        print(f"[distributed] rank {rank} using GLOO_SOCKET_IFNAME={gloo_ifname}", flush=True)

    try:
        print(
            f"[distributed] rank {rank} initializing process group "
            f"{master_addr}:{dist_port} world_size={world_size}",
            flush=True,
        )
        dist.init_process_group(
            backend="gloo",
            init_method=f"tcp://{master_addr}:{dist_port}",
            world_size=world_size,
            rank=rank,
            timeout=timeout,
        )
        print(f"[distributed] rank {rank} process group ready", flush=True)
    except Exception as exc:
        result_q.put(
            {
                "type": "distributed_error",
                "epoch": 0,
                "rank": rank,
                "error": f"dist init: {exc}",
            }
        )
        return

    try:
        torch.manual_seed(42)
        model = build_model(model_name, classes, image_size).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        loader = (
            _build_local_dataset_loader(shard, batch_size, batches_per_epoch)
            if isinstance(shard, dict)
            else _build_loader(shard, batch_size, batches_per_epoch)
        )
        eval_loader = (
            _build_eval_loader(shard, batch_size, int(shard.get("eval_batches") or 0))
            if isinstance(shard, dict) and rank == 0
            else None
        )

        print(f"[distributed] rank {rank} broadcasting model", flush=True)
        _broadcast_model(model, device)
        print(f"[distributed] rank {rank} model broadcast complete", flush=True)
        dist.barrier()

        run_started = time.monotonic()
        total_samples = 0
        total_batches = 0
        total_loss_weighted = 0.0
        final_batch_loss = 0.0
        total_straggler_delay = 0.0
        power = PowerSampler(device, interval_seconds=2.0)
        compressor = TopKCompressor(compress_ratio) if compression == "topk" else None
        if rank == 0:
            power.start()

        for epoch in range(1, epochs + 1):
            current_epoch = epoch
            if stop.is_set():
                result_q.put(
                    {
                        "type": "distributed_error",
                        "epoch": epoch,
                        "rank": rank,
                        "error": "training stopped",
                    }
                )
                return

            synchronize_device(device)
            epoch_started = time.monotonic()
            result = _run_epoch(
                model=model,
                optimizer=optimizer,
                loader=loader,
                world_size=world_size,
                device=device,
                batches_per_epoch=batches_per_epoch,
                stop_event=stop,
                compression=compression,
                compressor=compressor,
                rank=rank,
                straggler_rank=straggler_rank,
                straggler_delay_seconds=straggler_delay_seconds,
            )
            dist.barrier()
            synchronize_device(device)
            eval_metrics = _evaluate(model, eval_loader, device) if rank == 0 else {}
            duration_seconds = time.monotonic() - epoch_started
            epoch_samples = int(result["global_samples"])
            total_samples += epoch_samples
            total_batches += int(result["batches"])
            total_loss_weighted += float(result["loss"]) * max(1, epoch_samples)
            final_batch_loss = float(result["final_batch_loss"])
            total_straggler_delay += float(result.get("straggler_delay_total_seconds") or 0.0)
            cumulative_seconds = max(time.monotonic() - run_started, 1e-9)
            throughput = total_samples / cumulative_seconds
            result_q.put(
                {
                    "type": "distributed_epoch",
                    "epoch": epoch,
                    "rank": rank,
                    "hostname": socket.gethostname(),
                    "world_size": world_size,
                    "participant_count": world_size,
                    "model": model_name,
                    "dataset": dataset_name,
                    "classes": classes,
                    "image_size": image_size,
                    "batch_size": batch_size,
                    "epochs": epochs,
                    "batches_per_epoch": batches_per_epoch,
                    "lr": LR,
                    "momentum": MOMENTUM,
                    "weight_decay": WEIGHT_DECAY,
                    "dataset_samples": dataset_samples,
                    "measured_batches": total_batches,
                    "warmup_batches": 0,
                    "samples": total_samples,
                    "loss": float(result["loss"]),
                    "cumulative_loss": total_loss_weighted / max(1, total_samples),
                    "final_batch_loss": final_batch_loss,
                    "val_loss": float(eval_metrics.get("val_loss", 0.0)),
                    "val_acc": float(eval_metrics.get("val_acc", 0.0)),
                    "val_top5_acc": float(eval_metrics.get("val_top5_acc", 0.0)),
                    "val_samples": int(eval_metrics.get("val_samples", 0)),
                    "batches": int(result["batches"]),
                    "device": str(result["device"]),
                    "duration_seconds": duration_seconds,
                    "total_seconds": cumulative_seconds,
                    "samples_per_second": throughput,
                    "throughput": throughput,
                    "seconds_per_batch": duration_seconds / max(1, int(result["batches"])),
                    "speedup": 1.0,
                    "efficiency": 1.0 / max(1, world_size),
                    "worker_score": throughput,
                    "estimated_epoch_seconds": dataset_samples / max(throughput, 1e-9),
                    "estimated_100_epoch_hours": (dataset_samples / max(throughput, 1e-9)) * 100 / 3600,
                    "amp": amp,
                    "optimizations": optimizations,
                    "compression": compression,
                    "compress_ratio": compress_ratio,
                    "raw_gradient_numel": float(result.get("raw_gradient_numel") or 0.0),
                    "compressed_gradient_numel": float(result.get("compressed_gradient_numel") or 0.0),
                    "compression_ratio": float(result.get("compression_ratio") or 1.0),
                    "straggler_rank": straggler_rank,
                    "straggler_delay_seconds": straggler_delay_seconds,
                    "straggler_delay_total_seconds": total_straggler_delay,
                    "metric_accuracy_source": "validation_rank0" if rank == 0 else "not_evaluated",
                }
            )

        dist.barrier()
        total_seconds = max(time.monotonic() - run_started, 1e-9)
        avg_power, max_power, energy, power_source = power.stats(total_seconds)
        if rank == 0:
            power.stop()
        result_q.put(
            {
                "type": "distributed_complete",
                "rank": rank,
                "epochs": epochs,
                "hostname": socket.gethostname(),
                "world_size": world_size,
                "participant_count": world_size,
                "model": model_name,
                "dataset": dataset_name,
                "classes": classes,
                "image_size": image_size,
                "batch_size": batch_size,
                "batches_per_epoch": batches_per_epoch,
                "lr": LR,
                "momentum": MOMENTUM,
                "weight_decay": WEIGHT_DECAY,
                "dataset_samples": dataset_samples,
                "measured_batches": total_batches,
                "warmup_batches": 0,
                "samples": total_samples,
                "seconds": total_seconds,
                "samples_per_second": total_samples / total_seconds,
                "throughput": total_samples / total_seconds,
                "seconds_per_batch": total_seconds / max(1, total_batches),
                "speedup": 1.0,
                "efficiency": 1.0 / max(1, world_size),
                "worker_score": total_samples / total_seconds,
                "worker_scores": "{}",
                "estimated_epoch_seconds": dataset_samples / max(total_samples / total_seconds, 1e-9),
                "estimated_100_epoch_hours": (dataset_samples / max(total_samples / total_seconds, 1e-9)) * 100 / 3600,
                "loss": total_loss_weighted / max(1, total_samples),
                "final_batch_loss": final_batch_loss,
                "avg_power_watts": avg_power,
                "max_power_watts": max_power,
                "energy_joules": energy,
                "power_source": power_source,
                "amp": amp,
                "optimizations": optimizations,
                "compression": compression,
                "compress_ratio": compress_ratio,
                "straggler_rank": straggler_rank,
                "straggler_delay_seconds": straggler_delay_seconds,
                "straggler_delay_total_seconds": total_straggler_delay,
                "metric_accuracy_source": "validation_rank0" if rank == 0 else "not_evaluated",
            }
        )
    except Exception as exc:
        result_q.put(
            {
                "type": "distributed_error",
                "epoch": current_epoch,
                "rank": rank,
                "error": str(exc),
            }
        )
    finally:
        try:
            dist.destroy_process_group()
        except Exception:
            pass
