from __future__ import annotations

from pathlib import Path
from typing import Any


def load_cifar10_arrays(project_dir: Path, max_samples: int = 0) -> tuple[Any, Any]:
    """Load CIFAR-10 training data as numpy arrays.

    Imports stay local so the existing federated MNIST mode does not require
    numpy unless the distributed mode is used.
    """

    import numpy as np
    from torchvision.datasets import CIFAR10

    dataset = CIFAR10(root=str(project_dir / ".data"), train=True, download=True)
    images = dataset.data
    labels = np.asarray(dataset.targets, dtype=np.int64)
    if max_samples > 0:
        images = images[:max_samples]
        labels = labels[:max_samples]
    return images, labels


def compute_weighted_slices(
    participant_scores: dict[str, float],
    total_samples: int,
) -> dict[str, tuple[int, int]]:
    """Split sample indices proportionally by participant score."""

    if not participant_scores:
        raise ValueError("participant_scores cannot be empty")
    if total_samples <= 0:
        raise ValueError("total_samples must be positive")

    participants = list(participant_scores)
    weights = [max(1.0, float(participant_scores[participant])) for participant in participants]
    weight_total = sum(weights)
    raw_counts = [weight / weight_total * total_samples for weight in weights]
    counts = [int(count) for count in raw_counts]
    remainder = total_samples - sum(counts)

    order = sorted(
        range(len(participants)),
        key=lambda index: raw_counts[index] - counts[index],
        reverse=True,
    )
    for index in order[:remainder]:
        counts[index] += 1

    if total_samples >= len(participants):
        for index, count in enumerate(counts):
            if count == 0:
                donor = max(range(len(counts)), key=lambda item: counts[item])
                if counts[donor] > 1:
                    counts[donor] -= 1
                    counts[index] = 1

    slices: dict[str, tuple[int, int]] = {}
    cursor = 0
    for participant, count in zip(participants, counts):
        stop = min(total_samples, cursor + max(0, count))
        slices[participant] = (cursor, stop)
        cursor = stop
    if slices:
        last = participants[-1]
        slices[last] = (slices[last][0], total_samples)
    return slices


def score_batch_multipliers(participant_scores: dict[str, float]) -> dict[str, int]:
    """Convert benchmark scores into bounded batch-size multipliers."""

    if not participant_scores:
        return {}
    baseline = min(max(1.0, float(score)) for score in participant_scores.values())
    multipliers: dict[str, int] = {}
    for participant, score in participant_scores.items():
        ratio = max(1.0, float(score)) / baseline
        multipliers[participant] = max(1, min(8, round(ratio)))
    return multipliers
