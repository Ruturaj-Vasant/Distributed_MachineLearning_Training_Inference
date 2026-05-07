from __future__ import annotations

import shutil
import subprocess
import threading

import torch


class PowerSampler:
    def __init__(self, device: torch.device, interval_seconds: float) -> None:
        self.device = device
        self.interval_seconds = max(0.25, interval_seconds)
        self.values: list[float] = []
        self.source = "not_available"
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self.device.type == "cuda" and shutil.which("nvidia-smi"):
            self.source = "nvidia-smi"
            self._thread = threading.Thread(target=self._run_nvidia, daemon=True)
            self._thread.start()
        elif self.device.type == "mps":
            self.source = "not_available_mps"
        elif self.device.type == "cpu":
            self.source = "not_available_cpu"

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def stats(self, seconds: float) -> tuple[float | None, float | None, float | None, str]:
        if not self.values:
            return None, None, None, self.source
        avg_power = sum(self.values) / len(self.values)
        return avg_power, max(self.values), avg_power * max(0.0, seconds), self.source

    def _run_nvidia(self) -> None:
        while not self._stop.is_set():
            value = self._read_nvidia_power()
            if value is not None:
                self.values.append(value)
            self._stop.wait(self.interval_seconds)

    @staticmethod
    def _read_nvidia_power() -> float | None:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=power.draw",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=3,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if result.returncode != 0:
            return None
        line = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
        try:
            return float(line.strip())
        except ValueError:
            return None
