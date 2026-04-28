from __future__ import annotations

import queue
import threading
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .models import CifarCnn

DEFAULT_DIST_TIMEOUT_SECONDS = 60.0


def choose_device(accelerator: str) -> torch.device:
    requested = accelerator.lower()
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
    if requested == "mps" and mps_backend is not None and mps_backend.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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


def run_one_epoch(
    rank: int,
    world_size: int,
    master_addr: str,
    dist_port: int,
    shard: tuple[Any, Any],
    accelerator: str,
    batch_size: int,
    batches_per_epoch: int,
    model_state: dict[str, torch.Tensor] | None,
    timeout_seconds: float,
    result_q: queue.SimpleQueue[dict[str, Any]],
) -> None:
    device = choose_device(accelerator)
    timeout = timedelta(seconds=max(1.0, timeout_seconds))

    try:
        dist.init_process_group(
            backend="gloo",
            init_method=f"tcp://{master_addr}:{dist_port}",
            world_size=world_size,
            rank=rank,
            timeout=timeout,
        )
    except Exception as exc:
        result_q.put({"error": f"dist init: {exc}"})
        return

    try:
        torch.manual_seed(42)
        model = CifarCnn().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        if rank == 0 and model_state is not None:
            model.load_state_dict({key: value.to(device) for key, value in model_state.items()})

        for parameter in model.parameters():
            buffer = parameter.data.detach().cpu().clone().contiguous()
            dist.broadcast(buffer, src=0)
            parameter.data.copy_(buffer.to(device))

        images_np, labels_np = shard
        images = torch.from_numpy(images_np.copy()).float().permute(0, 3, 1, 2) / 255.0
        labels = torch.from_numpy(labels_np.copy()).long()
        usable = min(len(images), max(1, batches_per_epoch) * max(1, batch_size))
        loader = DataLoader(
            TensorDataset(images[:usable], labels[:usable]),
            batch_size=max(1, batch_size),
            shuffle=True,
            drop_last=True,
        )

        model.train()
        total_loss = 0.0
        batch_count = 0
        for images_batch, labels_batch in loader:
            if batch_count >= batches_per_epoch:
                break
            images_batch = images_batch.to(device)
            labels_batch = labels_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(images_batch), labels_batch)
            loss.backward()
            _allreduce_grads(model, world_size, device)
            optimizer.step()

            total_loss += float(loss.detach().cpu())
            batch_count += 1

        local_loss = total_loss / max(1, batch_count)
        loss_tensor = torch.tensor([local_loss], dtype=torch.float32)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        global_loss = float((loss_tensor / world_size).item())

        result: dict[str, Any] = {
            "loss": global_loss,
            "batches": batch_count,
            "device": str(device),
        }
        if rank == 0:
            result["model_state"] = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        result_q.put(result)
    except Exception as exc:
        result_q.put({"error": str(exc)})
    finally:
        try:
            dist.destroy_process_group()
        except Exception:
            pass


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
    model_state: dict[str, torch.Tensor] | None = None
    stop = stop_event or threading.Event()

    for epoch in range(1, epochs + 1):
        if stop.is_set():
            result_q.put({"type": "distributed_error", "epoch": epoch, "error": "training stopped"})
            return

        epoch_q: queue.SimpleQueue[dict[str, Any]] = queue.SimpleQueue()
        run_one_epoch(
            rank=rank,
            world_size=world_size,
            master_addr=master_addr,
            dist_port=dist_port,
            shard=shard,
            accelerator=accelerator,
            batch_size=batch_size,
            batches_per_epoch=batches_per_epoch,
            model_state=model_state,
            timeout_seconds=timeout_seconds,
            result_q=epoch_q,
        )
        result = epoch_q.get()
        if "error" in result:
            result_q.put(
                {
                    "type": "distributed_error",
                    "epoch": epoch,
                    "rank": rank,
                    "error": result["error"],
                }
            )
            return

        if rank == 0:
            model_state = result.get("model_state")
        result_q.put(
            {
                "type": "distributed_epoch",
                "epoch": epoch,
                "rank": rank,
                "loss": float(result["loss"]),
                "batches": int(result["batches"]),
                "device": str(result["device"]),
            }
        )

    result_q.put({"type": "distributed_complete", "rank": rank, "epochs": epochs})
