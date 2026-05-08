from __future__ import annotations

from dataclasses import dataclass

OPTIMIZATION_CHOICES = ("none", "topk", "straggler", "topk-straggler", "fp16")


@dataclass(frozen=True)
class DistributedOptimizationConfig:
    label: str
    compression: str
    straggler_rank: int
    straggler_delay_seconds: float


def infer_optimization_label(
    compression: str,
    straggler_rank: int,
    straggler_delay_seconds: float,
) -> str:
    has_topk = compression == "topk"
    has_straggler = straggler_rank >= 0 and straggler_delay_seconds > 0
    if has_topk and has_straggler:
        return "topk-straggler"
    if has_topk:
        return "topk"
    if has_straggler:
        return "straggler"
    return "none"


def resolve_optimizations(
    selected: str | None,
    parallelism: str,
    compression: str,
    straggler_rank: int,
    straggler_delay_seconds: float,
) -> DistributedOptimizationConfig:
    label = selected or infer_optimization_label(
        compression,
        straggler_rank,
        straggler_delay_seconds,
    )
    if label not in OPTIMIZATION_CHOICES:
        raise ValueError(f"unknown optimization mode: {label}")

    resolved_compression = compression
    resolved_straggler_rank = straggler_rank
    resolved_straggler_delay = straggler_delay_seconds
    if selected is not None:
        resolved_compression = "topk" if "topk" in label else "none"
        if "straggler" not in label:
            resolved_straggler_rank = -1
            resolved_straggler_delay = 0.0

    if parallelism == "data" and label == "fp16":
        raise ValueError("fp16 optimization is only supported with --distributed-parallel parameter-server")
    if resolved_compression not in {"none", "topk"}:
        raise ValueError(f"unsupported compression mode: {resolved_compression}")
    if resolved_straggler_rank < 0:
        resolved_straggler_delay = 0.0
    if resolved_straggler_delay < 0:
        raise ValueError("straggler delay must be non-negative")
    if "straggler" in label and (resolved_straggler_rank < 0 or resolved_straggler_delay <= 0):
        raise ValueError(
            "straggler optimization needs --distributed-straggler-rank >= 0 "
            "and --distributed-straggler-delay > 0"
        )

    if parallelism == "parameter-server":
        if label not in {"none", "fp16"}:
            raise ValueError("parameter-server supports --distributed-optimizations none or fp16")
        return DistributedOptimizationConfig(
            label=label,
            compression="none",
            straggler_rank=-1,
            straggler_delay_seconds=0.0,
        )

    if parallelism != "data":
        return DistributedOptimizationConfig(
            label="none",
            compression="none",
            straggler_rank=-1,
            straggler_delay_seconds=0.0,
        )

    return DistributedOptimizationConfig(
        label=infer_optimization_label(
            resolved_compression,
            resolved_straggler_rank,
            resolved_straggler_delay,
        ),
        compression=resolved_compression,
        straggler_rank=resolved_straggler_rank,
        straggler_delay_seconds=resolved_straggler_delay,
    )
