from __future__ import annotations

import io
from collections.abc import Mapping
from typing import Any

import torch


StateDict = dict[str, torch.Tensor]


def serialize_state_dict(state: Mapping[str, torch.Tensor], optimization: str = "none") -> bytes:
    payload: StateDict = {}
    use_fp16 = optimization == "fp16"
    for name, tensor in state.items():
        detached = tensor.detach().cpu()
        if use_fp16 and torch.is_floating_point(detached):
            detached = detached.to(dtype=torch.float16)
        payload[name] = detached.contiguous()

    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return buffer.getvalue()


def deserialize_state_dict(payload: bytes, force_float32: bool = True) -> StateDict:
    buffer = io.BytesIO(payload)
    state = torch.load(buffer, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError("state payload did not contain a state_dict")
    restored: StateDict = {}
    for name, value in state.items():
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"state entry is not a tensor: {name}")
        if force_float32 and torch.is_floating_point(value):
            value = value.to(dtype=torch.float32)
        restored[str(name)] = value.cpu()
    return restored


def raw_state_nbytes(state: Mapping[str, torch.Tensor]) -> int:
    return int(sum(tensor.numel() * tensor.element_size() for tensor in state.values()))


def aggregate_state_dicts(
    base_state: Mapping[str, torch.Tensor],
    weighted_states: list[tuple[Mapping[str, torch.Tensor], int]],
) -> StateDict:
    total_samples = sum(max(0, int(samples)) for _state, samples in weighted_states)
    if total_samples <= 0:
        return {name: tensor.detach().cpu().clone() for name, tensor in base_state.items()}

    aggregated: StateDict = {}
    for name, base_tensor in base_state.items():
        if torch.is_floating_point(base_tensor):
            output = torch.zeros_like(base_tensor.detach().cpu(), dtype=torch.float32)
            for state, samples in weighted_states:
                weight = max(0, int(samples)) / total_samples
                output += state[name].detach().cpu().to(dtype=torch.float32) * weight
            aggregated[name] = output
        else:
            aggregated[name] = base_tensor.detach().cpu().clone()
    return aggregated


def payload_size_mb(byte_count: int) -> float:
    return float(byte_count) / 1024.0 / 1024.0
