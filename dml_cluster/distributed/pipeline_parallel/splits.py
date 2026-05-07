from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineSplit:
    model: str
    stages: int
    stage_descriptions: tuple[str, ...]


def get_pipeline_split(model: str, stages: int) -> PipelineSplit:
    model_name = model.lower()
    if stages != 2:
        raise ValueError("only 2-stage pipeline splits are defined right now")
    if model_name in {"resnet50", "resnet101"}:
        return PipelineSplit(
            model=model_name,
            stages=2,
            stage_descriptions=(
                "conv1, bn1, relu, maxpool, layer1, layer2",
                "layer3, layer4, avgpool, fc",
            ),
        )
    if model_name == "vit_b_16":
        return PipelineSplit(
            model=model_name,
            stages=2,
            stage_descriptions=(
                "patch embedding and first half of encoder blocks",
                "second half of encoder blocks and classification head",
            ),
        )
    raise ValueError(f"no pipeline split is defined for model: {model}")
