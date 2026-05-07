from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

DATASET_NAME = "mnist"


class MnistNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model() -> MnistNet:
    return MnistNet()


def state_dict_to_payload(state_dict: dict[str, torch.Tensor]) -> str:
    buffer = io.BytesIO()
    torch.save({key: value.detach().cpu() for key, value in state_dict.items()}, buffer)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def payload_to_state_dict(payload: str) -> dict[str, torch.Tensor]:
    raw = base64.b64decode(payload.encode("ascii"))
    buffer = io.BytesIO(raw)
    return torch.load(buffer, map_location="cpu", weights_only=True)


def initial_model_payload(seed: int = 1337) -> str:
    torch.manual_seed(seed)
    model = build_model()
    return state_dict_to_payload(model.state_dict())


def average_state_payloads(results: list[dict[str, Any]]) -> str:
    weighted = [result for result in results if int(result.get("samples") or 0) > 0]
    if not weighted:
        raise ValueError("cannot average empty training results")

    total_samples = sum(int(result["samples"]) for result in weighted)
    states = [payload_to_state_dict(str(result["model_state"])) for result in weighted]
    averaged: dict[str, torch.Tensor] = {}

    for key in states[0]:
        first = states[0][key]
        if first.is_floating_point():
            value = torch.zeros_like(first, dtype=torch.float32)
            for state, result in zip(states, weighted):
                weight = int(result["samples"]) / total_samples
                value += state[key].to(dtype=torch.float32) * weight
            averaged[key] = value.to(dtype=first.dtype)
        else:
            averaged[key] = first.clone()

    return state_dict_to_payload(averaged)


def _mnist_dataset(project_dir: Path, train: bool) -> datasets.MNIST:
    return datasets.MNIST(
        root=str(project_dir / ".data"),
        train=train,
        download=True,
        transform=transforms.ToTensor(),
    )


def _contiguous_batch_subset(
    dataset: datasets.MNIST,
    start_batch: int,
    num_batches: int,
    batch_size: int,
) -> Subset:
    start = max(0, start_batch) * batch_size
    count = max(0, num_batches) * batch_size
    stop = min(len(dataset), start + count)
    return Subset(dataset, range(start, stop))


def train_assignment(task: dict[str, Any], project_dir: Path) -> dict[str, Any]:
    if task.get("dataset") != DATASET_NAME:
        raise ValueError(f"unsupported dataset: {task.get('dataset')}")

    device = choose_device()
    model = build_model()
    model.load_state_dict(payload_to_state_dict(str(task["model_state"])))
    model.to(device)
    model.train()

    batch_size = int(task["batch_size"])
    dataset = _mnist_dataset(project_dir, train=True)
    subset = _contiguous_batch_subset(
        dataset=dataset,
        start_batch=int(task["start_batch"]),
        num_batches=int(task["num_batches"]),
        batch_size=batch_size,
    )
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)

    optimizer = torch.optim.SGD(model.parameters(), lr=float(task["lr"]), momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_samples = 0
    batches = 0
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        samples = int(inputs.shape[0])
        total_loss += float(loss.detach().cpu()) * samples
        total_samples += samples
        batches += 1

    avg_loss = total_loss / max(1, total_samples)
    return {
        "type": "train_result",
        "run_id": task["run_id"],
        "epoch": int(task["epoch"]),
        "assignment_id": task["assignment_id"],
        "samples": total_samples,
        "batches": batches,
        "loss": avg_loss,
        "device": str(device),
        "model_state": state_dict_to_payload(model.state_dict()),
    }


def evaluate_payload(
    model_state: str,
    project_dir: Path,
    batch_size: int,
    max_batches: int,
) -> dict[str, float | int]:
    device = choose_device()
    model = build_model()
    model.load_state_dict(payload_to_state_dict(model_state))
    model.to(device)
    model.eval()

    dataset = _mnist_dataset(project_dir, train=False)
    if max_batches > 0:
        dataset = Subset(dataset, range(0, min(len(dataset), max_batches * batch_size)))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    samples = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            batch_samples = int(inputs.shape[0])
            total_loss += float(loss.detach().cpu()) * batch_samples
            correct += int((outputs.argmax(dim=1) == targets).sum().detach().cpu())
            samples += batch_samples

    return {
        "loss": total_loss / max(1, samples),
        "accuracy": correct / max(1, samples),
        "samples": samples,
    }
