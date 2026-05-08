from __future__ import annotations

import time
from typing import Any

import torch
import torch.nn.functional as F

from ...models import build_model
from ..all_reduce.runner import (
    LR,
    _build_eval_loader,
    _build_local_dataset_loader,
    _evaluate,
    choose_device,
    synchronize_device,
)
from .serialization import StateDict


def train_local_state(
    rank: int,
    shard_config: dict[str, Any],
    initial_state: StateDict,
    accelerator: str,
    batch_size: int,
    batches_per_epoch: int,
) -> dict[str, Any]:
    device = choose_device(accelerator)
    model_name = str(shard_config.get("model") or "cifar_cnn")
    dataset_name = str(shard_config.get("dataset") or "cifar10")
    classes = int(shard_config.get("classes") or 10)
    image_size = int(shard_config.get("image_size") or 32)
    eval_batches = int(shard_config.get("eval_batches") or 0)

    torch.manual_seed(42 + rank)
    model = build_model(model_name, classes, image_size).to(device)
    model.load_state_dict(initial_state, strict=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loader = _build_local_dataset_loader(shard_config, batch_size, batches_per_epoch)
    eval_loader = _build_eval_loader(shard_config, batch_size, eval_batches) if rank == 0 else None

    model.train()
    started = time.monotonic()
    total_loss = 0.0
    total_samples = 0
    batch_count = 0
    final_batch_loss = 0.0
    for images, labels in loader:
        if batch_count >= batches_per_epoch:
            break
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_samples = int(images.shape[0])
        batch_loss = float(loss.detach().cpu())
        total_loss += batch_loss * batch_samples
        total_samples += batch_samples
        final_batch_loss = batch_loss
        batch_count += 1

    synchronize_device(device)
    duration_seconds = time.monotonic() - started
    eval_metrics = _evaluate(model, eval_loader, device) if rank == 0 else {}
    updated_state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    return {
        "state": updated_state,
        "loss": total_loss / max(1, total_samples),
        "batches": batch_count,
        "samples": total_samples,
        "final_batch_loss": final_batch_loss,
        "duration_seconds": duration_seconds,
        "device": str(device),
        "model": model_name,
        "dataset": dataset_name,
        "classes": classes,
        "image_size": image_size,
        "val_loss": float(eval_metrics.get("val_loss") or 0.0),
        "val_acc": float(eval_metrics.get("val_acc") or 0.0),
        "val_top5_acc": float(eval_metrics.get("val_top5_acc") or 0.0),
        "val_samples": int(eval_metrics.get("val_samples") or 0),
    }
