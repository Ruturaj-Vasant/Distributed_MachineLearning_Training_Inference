from __future__ import annotations

import queue
import socket
import threading
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from ..common.power import PowerSampler
from ..common.network import configure_gloo_socket_ifname, create_gloo_pg_options
from ..data_parallel.runner import DEFAULT_DIST_TIMEOUT_SECONDS, choose_device, synchronize_device
from ..datasets import load_dataset, shard_dataset
from ..models import build_model
from .splits import PipelineSplit, get_pipeline_split

LR = 1e-3
MOMENTUM = 0.0
WEIGHT_DECAY = 0.0
IMAGE_SIZE = 224

_DTYPE_TO_CODE = {
    torch.float32: 0,
    torch.int64: 1,
}
_CODE_TO_DTYPE = {
    0: torch.float32,
    1: torch.int64,
}


class PipelineParallelNotImplementedError(NotImplementedError):
    """Raised when the selected pipeline layout is outside the implemented runner."""


@dataclass(frozen=True)
class PipelineRunConfig:
    model: str
    dataset: str
    stages: int
    microbatch_size: int

    @property
    def split(self) -> PipelineSplit:
        return get_pipeline_split(self.model, self.stages)


class ResNetStage0(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class ResNetStage1(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = model.avgpool
        self.fc = model.fc

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.layer3(inputs)
        outputs = self.layer4(outputs)
        outputs = self.avgpool(outputs)
        outputs = torch.flatten(outputs, 1)
        return self.fc(outputs)


class VitStage0(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.conv_proj = model.conv_proj
        self.class_token = model.class_token
        self.pos_embedding = model.encoder.pos_embedding
        self.dropout = model.encoder.dropout
        layers = list(model.encoder.layers.children())
        midpoint = len(layers) // 2
        self.layers = nn.Sequential(*layers[:midpoint])
        self.image_size = int(model.image_size)
        self.patch_size = int(model.patch_size)
        self.hidden_dim = int(model.hidden_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, _channels, height, width = inputs.shape
        if height != self.image_size or width != self.image_size:
            raise ValueError(
                f"ViT pipeline expected {self.image_size}x{self.image_size}, got {height}x{width}"
            )
        patches_h = height // self.patch_size
        patches_w = width // self.patch_size
        outputs = self.conv_proj(inputs)
        outputs = outputs.reshape(batch_size, self.hidden_dim, patches_h * patches_w)
        outputs = outputs.permute(0, 2, 1)
        class_token = self.class_token.expand(batch_size, -1, -1)
        outputs = torch.cat([class_token, outputs], dim=1)
        outputs = outputs + self.pos_embedding
        outputs = self.dropout(outputs)
        return self.layers(outputs)


class VitStage1(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        layers = list(model.encoder.layers.children())
        midpoint = len(layers) // 2
        self.layers = nn.Sequential(*layers[midpoint:])
        self.ln = model.encoder.ln
        self.heads = model.heads

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.layers(inputs)
        outputs = self.ln(outputs)
        outputs = outputs[:, 0]
        return self.heads(outputs)


def validate_pipeline_config(config: PipelineRunConfig) -> PipelineSplit:
    if config.microbatch_size <= 0:
        raise ValueError("microbatch_size must be greater than zero")
    if config.stages != 2:
        raise PipelineParallelNotImplementedError("only 2-stage pipeline execution is implemented")
    if config.model.lower() not in {"resnet50", "resnet101", "vit_b_16"}:
        raise PipelineParallelNotImplementedError(
            "real pipeline execution currently supports resnet50, resnet101, and vit_b_16 only"
        )
    split = config.split
    return split


def _build_stage(model_name: str, classes: int, image_size: int, rank: int) -> nn.Module:
    torch.manual_seed(42)
    model = build_model(model_name, classes, image_size)
    if model_name.lower() in {"resnet50", "resnet101"}:
        return ResNetStage0(model) if rank == 0 else ResNetStage1(model)
    if model_name.lower() == "vit_b_16":
        return VitStage0(model) if rank == 0 else VitStage1(model)
    else:
        raise PipelineParallelNotImplementedError(
            f"pipeline stage construction is not implemented for {model_name}"
        )


def _build_train_loader(config: dict[str, Any], batch_size: int, batches_per_epoch: int) -> DataLoader:
    dataset_samples = int(config.get("dataset_samples") or 0)
    optimizations = str(config.get("optimizations") or "none")
    spec = load_dataset(
        name=str(config["dataset"]),
        project_dir=Path(str(config["project_dir"])),
        download=bool(config.get("download")),
        max_samples=dataset_samples,
        image_size=int(config.get("image_size") or IMAGE_SIZE),
    )
    usable = min(len(spec.train), max(1, batch_size) * max(1, batches_per_epoch))
    if usable < 1:
        raise ValueError("pipeline training needs at least one usable training sample")
    subset = shard_dataset(spec.train, 0, usable)
    return DataLoader(
        subset,
        batch_size=max(1, batch_size),
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )


def _send_ints(values: list[int], dst: int) -> None:
    dist.send(torch.tensor(values, dtype=torch.int64), dst=dst)


def _recv_ints(count: int, src: int) -> list[int]:
    tensor = torch.empty(count, dtype=torch.int64)
    dist.recv(tensor, src=src)
    return [int(value) for value in tensor.tolist()]


def _send_tensor(tensor: torch.Tensor, dst: int) -> None:
    payload = tensor.detach().cpu().contiguous()
    dtype_code = _DTYPE_TO_CODE.get(payload.dtype)
    if dtype_code is None:
        if payload.is_floating_point():
            payload = payload.to(torch.float32)
            dtype_code = _DTYPE_TO_CODE[torch.float32]
        else:
            payload = payload.to(torch.int64)
            dtype_code = _DTYPE_TO_CODE[torch.int64]
    _send_ints([payload.dim(), dtype_code], dst)
    _send_ints(list(payload.shape), dst)
    dist.send(payload, dst=dst)


def _recv_tensor(src: int) -> torch.Tensor:
    ndim, dtype_code = _recv_ints(2, src)
    shape = _recv_ints(ndim, src)
    dtype = _CODE_TO_DTYPE.get(dtype_code)
    if dtype is None:
        raise RuntimeError(f"unsupported pipeline tensor dtype code: {dtype_code}")
    tensor = torch.empty(tuple(shape), dtype=dtype)
    dist.recv(tensor, src=src)
    return tensor


def _topk_correct(outputs: torch.Tensor, labels: torch.Tensor, k: int) -> int:
    topk = min(k, outputs.shape[1])
    return int(
        (outputs.topk(topk, dim=1).indices == labels.view(-1, 1)).any(dim=1).sum().detach().cpu()
    )


def _run_stage0_epoch(
    stage: nn.Module,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    device: torch.device,
    batches_per_epoch: int,
    microbatch_size: int,
    stop_event: threading.Event,
) -> dict[str, int]:
    stage.train()
    batch_count = 0
    sample_count = 0

    for images_batch, labels_batch in loader:
        if batch_count >= batches_per_epoch or stop_event.is_set():
            break
        optimizer.zero_grad(set_to_none=True)
        micro_images = torch.split(images_batch, max(1, microbatch_size))
        micro_labels = torch.split(labels_batch, max(1, microbatch_size))
        _send_ints([len(micro_images)], dst=1)

        for images_micro, labels_micro in zip(micro_images, micro_labels):
            activations = stage(images_micro.to(device))
            _send_tensor(activations, dst=1)
            _send_tensor(labels_micro.to(torch.int64), dst=1)
            activation_grad = _recv_tensor(src=1).to(device)
            activations.backward(activation_grad)
            sample_count += int(images_micro.shape[0])

        optimizer.step()
        batch_count += 1

    return {"batches": batch_count, "samples": sample_count}


def _run_stage1_epoch(
    stage: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    batches_per_epoch: int,
    stop_event: threading.Event,
) -> dict[str, float | int]:
    stage.train()
    total_loss = 0.0
    batch_count = 0
    sample_count = 0
    correct_top1 = 0
    correct_top5 = 0
    final_batch_loss = 0.0

    for _ in range(batches_per_epoch):
        if stop_event.is_set():
            break
        microbatch_count = _recv_ints(1, src=0)[0]
        optimizer.zero_grad(set_to_none=True)

        for _microbatch_index in range(microbatch_count):
            activations_cpu = _recv_tensor(src=0)
            labels_cpu = _recv_tensor(src=0).long()
            activations = activations_cpu.to(device).requires_grad_(True)
            labels = labels_cpu.to(device)

            outputs = stage(activations)
            unscaled_loss = F.cross_entropy(outputs, labels)
            loss = unscaled_loss / max(1, microbatch_count)
            loss.backward()
            _send_tensor(activations.grad, dst=0)

            batch_samples = int(labels.shape[0])
            batch_loss = float(unscaled_loss.detach().cpu())
            total_loss += batch_loss * batch_samples
            sample_count += batch_samples
            correct_top1 += int((outputs.argmax(dim=1) == labels).sum().detach().cpu())
            correct_top5 += _topk_correct(outputs, labels, 5)
            final_batch_loss = batch_loss

        optimizer.step()
        batch_count += 1

    return {
        "loss": total_loss / max(1, sample_count),
        "batches": batch_count,
        "samples": sample_count,
        "correct_top1": correct_top1,
        "correct_top5": correct_top5,
        "final_batch_loss": final_batch_loss,
    }


def _send_epoch_metrics(metrics: dict[str, float | int], dst: int) -> None:
    tensor = torch.tensor(
        [
            float(metrics.get("loss") or 0.0),
            float(metrics.get("samples") or 0),
            float(metrics.get("correct_top1") or 0),
            float(metrics.get("correct_top5") or 0),
            float(metrics.get("batches") or 0),
            float(metrics.get("final_batch_loss") or 0.0),
        ],
        dtype=torch.float32,
    )
    dist.send(tensor, dst=dst)


def _recv_epoch_metrics(src: int) -> dict[str, float | int]:
    tensor = torch.empty(6, dtype=torch.float32)
    dist.recv(tensor, src=src)
    values = tensor.tolist()
    samples = int(values[1])
    return {
        "loss": float(values[0]),
        "samples": samples,
        "correct_top1": int(values[2]),
        "correct_top5": int(values[3]),
        "batches": int(values[4]),
        "final_batch_loss": float(values[5]),
        "val_acc": float(values[2]) / max(1, samples),
        "val_top5_acc": float(values[3]) / max(1, samples),
    }


def run_training(
    rank: int,
    world_size: int,
    master_addr: str,
    dist_port: int,
    config: dict[str, Any],
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
    model_name = str(config.get("model") or "resnet101")
    dataset_name = str(config.get("dataset") or "cifar10")
    classes = int(config.get("classes") or 10)
    image_size = int(config.get("image_size") or IMAGE_SIZE)
    dataset_samples = int(config.get("dataset_samples") or 0)
    microbatch_size = max(1, min(int(config.get("microbatch_size") or batch_size), batch_size))
    current_epoch = 0

    if world_size != 2:
        result_q.put(
            {
                "type": "distributed_error",
                "epoch": 0,
                "rank": rank,
                "error": "pipeline execution currently requires exactly 2 ranks",
            }
        )
        return

    pg_options, gloo_hostname = create_gloo_pg_options(master_addr, dist, timeout)
    if pg_options is not None:
        print(f"[pipeline] rank {rank} Gloo explicit device hostname={gloo_hostname}", flush=True)
    else:
        gloo_ifname = configure_gloo_socket_ifname(master_addr)
        if gloo_ifname:
            print(f"[pipeline] rank {rank} using GLOO_SOCKET_IFNAME={gloo_ifname}", flush=True)
        elif gloo_hostname:
            print(
                f"[pipeline] rank {rank} Tailscale IP={gloo_hostname} "
                f"(no interface mapped, Gloo will use system default)",
                flush=True,
            )

    try:
        dist.init_process_group(
            backend="gloo",
            init_method=f"tcp://{master_addr}:{dist_port}",
            world_size=world_size,
            rank=rank,
            timeout=timeout,
            pg_options=pg_options,
        )
    except Exception as exc:
        result_q.put(
            {
                "type": "distributed_error",
                "epoch": 0,
                "rank": rank,
                "error": f"pipeline dist init: {exc}",
            }
        )
        return

    power = PowerSampler(device, interval_seconds=2.0)
    try:
        stage = _build_stage(model_name, classes, image_size, rank).to(device)
        optimizer = torch.optim.Adam(stage.parameters(), lr=LR)
        loader = _build_train_loader(config, batch_size, batches_per_epoch) if rank == 0 else None
        dist.barrier()

        run_started = time.monotonic()
        total_samples = 0
        total_batches = 0
        total_loss_weighted = 0.0
        final_batch_loss = 0.0
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
                        "error": "pipeline training stopped",
                    }
                )
                return

            synchronize_device(device)
            epoch_started = time.monotonic()
            if rank == 0:
                if loader is None:
                    raise RuntimeError("rank 0 pipeline loader was not initialized")
                local_result = _run_stage0_epoch(
                    stage,
                    optimizer,
                    loader,
                    device,
                    batches_per_epoch,
                    microbatch_size,
                    stop,
                )
                metrics = _recv_epoch_metrics(src=1)
                metrics["leader_batches"] = int(local_result["batches"])
                metrics["leader_samples"] = int(local_result["samples"])
            else:
                metrics = _run_stage1_epoch(stage, optimizer, device, batches_per_epoch, stop)
                _send_epoch_metrics(metrics, dst=0)

            dist.barrier()
            synchronize_device(device)
            duration_seconds = time.monotonic() - epoch_started
            epoch_samples = int(metrics.get("samples") or 0)
            total_samples += epoch_samples
            total_batches += int(metrics.get("batches") or 0)
            total_loss_weighted += float(metrics.get("loss") or 0.0) * max(1, epoch_samples)
            final_batch_loss = float(metrics.get("final_batch_loss") or final_batch_loss)
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
                    "parallelism": "pipeline",
                    "pipeline_stage": rank,
                    "pipeline_stages": world_size,
                    "microbatch_size": microbatch_size,
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
                    "loss": float(metrics.get("loss") or 0.0),
                    "cumulative_loss": total_loss_weighted / max(1, total_samples),
                    "final_batch_loss": final_batch_loss,
                    "val_loss": 0.0,
                    "val_acc": float(metrics.get("val_acc") or 0.0),
                    "val_top5_acc": float(metrics.get("val_top5_acc") or 0.0),
                    "val_samples": epoch_samples,
                    "batches": int(metrics.get("batches") or 0),
                    "device": str(device),
                    "duration_seconds": duration_seconds,
                    "total_seconds": cumulative_seconds,
                    "seconds": cumulative_seconds,
                    "samples_per_second": throughput,
                    "throughput": throughput,
                    "seconds_per_batch": duration_seconds / max(1, int(metrics.get("batches") or 0)),
                    "speedup": 1.0,
                    "efficiency": 1.0 / max(1, world_size),
                    "worker_score": throughput,
                    "estimated_epoch_seconds": dataset_samples / max(throughput, 1e-9),
                    "estimated_100_epoch_hours": (dataset_samples / max(throughput, 1e-9)) * 100 / 3600,
                    "amp": False,
                    "optimizations": optimizations,
                    "compression": "none",
                    "compress_ratio": 1.0,
                    "raw_gradient_numel": 0.0,
                    "compressed_gradient_numel": 0.0,
                    "compression_ratio": 1.0,
                    "metric_accuracy_source": "pipeline_train_stage1",
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
                "samples": total_samples,
                "seconds": total_seconds,
                "duration_seconds": total_seconds,
                "total_seconds": total_seconds,
                "samples_per_second": total_samples / total_seconds,
                "throughput": total_samples / total_seconds,
                "avg_power_watts": avg_power if rank == 0 else 0.0,
                "max_power_watts": max_power if rank == 0 else 0.0,
                "energy_joules": energy if rank == 0 else 0.0,
                "power_source": power_source if rank == 0 else "not_sampled",
                "optimizations": optimizations,
                "metric_accuracy_source": "pipeline_train_stage1",
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
        if rank == 0:
            power.stop()
        if dist.is_initialized():
            dist.destroy_process_group()
