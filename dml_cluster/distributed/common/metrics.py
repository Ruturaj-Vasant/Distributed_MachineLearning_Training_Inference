from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkResult:
    run_id: str
    hostname: str
    os: str
    machine: str
    device: str
    world_size: int
    participant_count: int
    model: str
    dataset: str
    classes: int
    image_size: int
    batch_size: int
    epochs: int
    batches_per_epoch: int
    lr: float
    momentum: float
    weight_decay: float
    dataset_samples: int
    measured_batches: int
    warmup_batches: int
    samples: int
    seconds: float
    samples_per_second: float
    throughput: float
    seconds_per_batch: float
    speedup: float
    efficiency: float
    worker_score: float
    worker_scores: str
    estimated_epoch_seconds: float
    estimated_100_epoch_hours: float
    loss: float
    final_batch_loss: float
    avg_power_watts: float | None
    max_power_watts: float | None
    energy_joules: float | None
    power_source: str
    amp: bool


@dataclass(frozen=True)
class ProgressRow:
    run_id: str
    hostname: str
    device: str
    world_size: int
    model: str
    dataset: str
    batch_size: int
    epoch: int
    epoch_batch: int
    measured_batch: int
    samples: int
    interval_seconds: float
    total_seconds: float
    interval_samples_per_second: float
    cumulative_samples_per_second: float
    interval_loss: float
    cumulative_loss: float
    avg_power_watts: float | None
    max_power_watts: float | None
    energy_joules: float | None
    power_source: str
