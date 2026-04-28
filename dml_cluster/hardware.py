"""Hardware detection and a small bootstrap benchmark.

This first milestone intentionally avoids importing PyTorch. The score is a
rough CPU-side readiness signal for the leader; the training milestone will
replace it with a torch benchmark on MPS/CUDA/CPU.
"""

from __future__ import annotations

import json
import platform
import shutil
import socket
import subprocess
import time
import contextlib
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class HardwareInfo:
    hostname: str
    os: str
    machine: str
    accelerator: str
    accelerator_name: str
    benchmark_score: float


def _run_text(command: list[str], timeout: float = 3.0) -> str:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _nvidia_gpu_name() -> str:
    if not shutil.which("nvidia-smi"):
        return ""
    output = _run_text(
        [
            "nvidia-smi",
            "--query-gpu=name",
            "--format=csv,noheader",
        ]
    )
    return output.splitlines()[0].strip() if output else ""


def _apple_cpu_name() -> str:
    output = _run_text(["sysctl", "-n", "machdep.cpu.brand_string"])
    return output or "Apple Silicon"


def benchmark_cpu(seconds: float = 0.75) -> float:
    values = [((idx % 97) + 1) / 97.0 for idx in range(4096)]
    loops = 0
    checksum = 0.0
    started = time.perf_counter()
    deadline = started + seconds
    while time.perf_counter() < deadline:
        subtotal = 0.0
        for left, right in zip(values, reversed(values)):
            subtotal += left * right
        checksum += subtotal
        loops += 1
    elapsed = max(time.perf_counter() - started, 1e-9)
    # Keep checksum alive so the interpreter cannot discard the loop body.
    if checksum < 0:
        return 0.0
    return round((loops * len(values)) / elapsed, 2)


def _benchmark_torch(torch: object, device: str, seconds: float = 1.0) -> float:
    size = 512
    try:
        matrix_a = torch.randn((size, size), device=device)
        matrix_b = torch.randn((size, size), device=device)
        deadline = time.perf_counter() + seconds
        loops = 0
        started = time.perf_counter()
        while time.perf_counter() < deadline:
            _ = matrix_a @ matrix_b
            loops += 1
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            with contextlib.suppress(Exception):
                torch.mps.synchronize()
        elapsed = max(time.perf_counter() - started, 1e-9)
        return round((loops * 2 * (size**3)) / elapsed, 2)
    except Exception:
        return benchmark_cpu()


def _detect_with_torch() -> tuple[str, str, float] | None:
    try:
        import torch
    except Exception:
        return None

    if torch.cuda.is_available():
        return "cuda", torch.cuda.get_device_name(0), _benchmark_torch(torch, "cuda")

    mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps", _apple_cpu_name(), _benchmark_torch(torch, "mps")

    return "cpu", "CPU", _benchmark_torch(torch, "cpu")


def detect_hardware() -> HardwareInfo:
    system = platform.system() or "Unknown"
    machine = platform.machine() or "unknown"
    accelerator = "cpu"
    accelerator_name = "CPU"
    benchmark_score = benchmark_cpu()

    torch_detection = _detect_with_torch()
    if torch_detection is not None:
        accelerator, accelerator_name, benchmark_score = torch_detection
        return HardwareInfo(
            hostname=socket.gethostname(),
            os=system,
            machine=machine,
            accelerator=accelerator,
            accelerator_name=accelerator_name,
            benchmark_score=benchmark_score,
        )

    gpu_name = _nvidia_gpu_name()
    if gpu_name:
        accelerator = "cuda"
        accelerator_name = gpu_name
    elif system == "Darwin" and machine.lower() in {"arm64", "aarch64"}:
        accelerator = "mps"
        accelerator_name = _apple_cpu_name()

    return HardwareInfo(
        hostname=socket.gethostname(),
        os=system,
        machine=machine,
        accelerator=accelerator,
        accelerator_name=accelerator_name,
        benchmark_score=benchmark_score,
    )


def main() -> None:
    print(json.dumps(asdict(detect_hardware()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
