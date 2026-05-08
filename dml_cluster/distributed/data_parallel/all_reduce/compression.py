from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CompressedTensor:
    indices: torch.Tensor
    values: torch.Tensor
    shape: tuple[int, ...]
    original_numel: int

    @property
    def compressed_numel(self) -> int:
        return int(self.values.numel())

    @property
    def compression_ratio(self) -> float:
        return self.original_numel / max(1, self.compressed_numel)


class TopKCompressor:
    def __init__(self, compress_ratio: float = 0.01) -> None:
        if not 0 < compress_ratio <= 1:
            raise ValueError("compress_ratio must be in (0, 1]")
        self.compress_ratio = compress_ratio
        self.residuals: dict[str, torch.Tensor] = {}

    def compress(self, name: str, tensor: torch.Tensor) -> CompressedTensor:
        source = tensor.detach().cpu()
        residual = self.residuals.get(name)
        if residual is not None:
            source = source + residual

        flat = source.flatten()
        k = max(1, int(flat.numel() * self.compress_ratio))
        _, indices = torch.topk(flat.abs(), k)
        values = flat[indices].clone()

        reconstructed = torch.zeros_like(flat)
        reconstructed[indices] = values
        self.residuals[name] = (flat - reconstructed).reshape_as(source)

        return CompressedTensor(
            indices=indices.to(dtype=torch.int64),
            values=values,
            shape=tuple(source.shape),
            original_numel=flat.numel(),
        )

    @staticmethod
    def decompress(payload: CompressedTensor) -> torch.Tensor:
        flat = torch.zeros(payload.original_numel, dtype=payload.values.dtype)
        flat[payload.indices] = payload.values
        return flat.reshape(payload.shape)
