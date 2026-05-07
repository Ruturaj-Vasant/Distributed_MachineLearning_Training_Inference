from __future__ import annotations

from dataclasses import dataclass


class HybridParallelNotImplementedError(NotImplementedError):
    """Raised when hybrid parallelism is selected before implementation."""


@dataclass(frozen=True)
class HybridRunConfig:
    pipeline_stages: int
    data_replicas: int

    @property
    def world_size(self) -> int:
        return self.pipeline_stages * self.data_replicas


def validate_hybrid_config(config: HybridRunConfig, available_ranks: int) -> None:
    if config.pipeline_stages < 2:
        raise ValueError("hybrid mode requires at least two pipeline stages")
    if config.data_replicas < 2:
        raise ValueError("hybrid mode requires at least two data replicas")
    if available_ranks != config.world_size:
        raise ValueError(
            "hybrid mode requires world_size == pipeline_stages * data_replicas; "
            f"got world_size={available_ranks}, expected={config.world_size}"
        )


def run_training(*args, **kwargs) -> None:
    raise HybridParallelNotImplementedError(
        "Hybrid parallelism is intentionally deferred; use data or pipeline mode for now."
    )
