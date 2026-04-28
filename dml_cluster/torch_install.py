"""Select and install the PyTorch wheel set for this worker."""

from __future__ import annotations

import argparse
import platform
import re
import shutil
import subprocess
import sys

TORCH_VERSION = "2.10.0"
TORCHVISION_VERSION = "0.25.0"
PACKAGES = [f"torch=={TORCH_VERSION}", f"torchvision=={TORCHVISION_VERSION}"]


def _run_text(command: list[str]) -> str:
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    if result.returncode != 0:
        return ""
    return (result.stdout + "\n" + result.stderr).strip()


def _nvidia_cuda_version() -> tuple[int, int] | None:
    if not shutil.which("nvidia-smi"):
        return None
    output = _run_text(["nvidia-smi"])
    match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", output)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _cuda_index_url(cuda_version: tuple[int, int] | None) -> str | None:
    if cuda_version is None:
        return None
    major, minor = cuda_version
    if (major, minor) >= (13, 0):
        return "https://download.pytorch.org/whl/cu130"
    if (major, minor) >= (12, 8):
        return "https://download.pytorch.org/whl/cu128"
    if (major, minor) >= (12, 6):
        return "https://download.pytorch.org/whl/cu126"
    return None


def selected_install_command() -> tuple[list[str], str]:
    system = platform.system()
    command = [sys.executable, "-m", "pip", "install", *PACKAGES]

    if system == "Darwin":
        return command, "macOS default PyPI wheels with MPS support"

    cuda_url = _cuda_index_url(_nvidia_cuda_version())
    if cuda_url:
        return [*command, "--index-url", cuda_url], f"CUDA wheels from {cuda_url}"

    if system == "Windows":
        return [
            *command,
            "--index-url",
            "https://download.pytorch.org/whl/cpu",
        ], "Windows CPU wheels"

    return command, "default PyPI wheels"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Install the selected PyTorch wheel set.")
    parser.add_argument("--install", action="store_true", help="Run the selected pip install command.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    command, reason = selected_install_command()
    print(f"[torch-install] selected: {reason}")
    print("[torch-install] command:", " ".join(command))
    if args.install:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
