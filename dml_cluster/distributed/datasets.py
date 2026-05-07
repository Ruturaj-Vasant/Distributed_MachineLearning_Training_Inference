from __future__ import annotations

import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms

TINY_IMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
IMAGE_SIZE = 224
DATASET_CHOICES = ("cifar10", "cifar100", "tiny-imagenet-200")


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    classes: int
    image_size: int
    train: Dataset
    val: Dataset | None


def train_transform(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


def eval_transform(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


def maybe_download_tiny_imagenet(data_dir: Path) -> Path:
    root = data_dir / "tiny-imagenet-200"
    train_dir = root / "train"
    if train_dir.exists():
        return root

    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / "tiny-imagenet-200.zip"
    if not archive.exists():
        print(f"[data] downloading Tiny ImageNet: {TINY_IMAGENET_URL}")
        with urllib.request.urlopen(TINY_IMAGENET_URL, timeout=60) as response:
            with archive.open("wb") as handle:
                shutil.copyfileobj(response, handle)

    print(f"[data] extracting {archive}")
    with zipfile.ZipFile(archive) as zipped:
        zipped.extractall(data_dir)
    if not train_dir.exists():
        raise FileNotFoundError(f"Tiny ImageNet train directory not found: {train_dir}")
    prepare_tiny_imagenet_val(root)
    return root


def prepare_tiny_imagenet_val(root: Path) -> Path | None:
    """Convert TinyImageNet's raw val layout into ImageFolder-compatible dirs."""

    val_dir = root / "val"
    images_dir = val_dir / "images"
    annotations = val_dir / "val_annotations.txt"
    classed_dir = root / "val_classed"
    if classed_dir.exists() and any(classed_dir.iterdir()):
        return classed_dir
    if not images_dir.exists() or not annotations.exists():
        return None

    classed_dir.mkdir(parents=True, exist_ok=True)
    with annotations.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            image_name, class_name = parts[0], parts[1]
            source = images_dir / image_name
            if not source.exists():
                continue
            target_dir = classed_dir / class_name
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / image_name
            if not target.exists():
                shutil.copy2(source, target)
    return classed_dir if any(classed_dir.iterdir()) else None


def load_dataset(
    name: str,
    project_dir: Path,
    download: bool,
    max_samples: int = 0,
    image_size: int = IMAGE_SIZE,
) -> DatasetSpec:
    data_dir = project_dir / ".data"
    dataset_name = name.lower()
    if dataset_name == "cifar10":
        train = datasets.CIFAR10(
            root=str(data_dir),
            train=True,
            download=download,
            transform=train_transform(image_size),
        )
        val = datasets.CIFAR10(
            root=str(data_dir),
            train=False,
            download=download,
            transform=eval_transform(image_size),
        )
        classes = 10
    elif dataset_name == "cifar100":
        train = datasets.CIFAR100(
            root=str(data_dir),
            train=True,
            download=download,
            transform=train_transform(image_size),
        )
        val = datasets.CIFAR100(
            root=str(data_dir),
            train=False,
            download=download,
            transform=eval_transform(image_size),
        )
        classes = 100
    elif dataset_name == "tiny-imagenet-200":
        root = maybe_download_tiny_imagenet(data_dir) if download else data_dir / "tiny-imagenet-200"
        train_dir = root / "train"
        val_dir = prepare_tiny_imagenet_val(root)
        if not train_dir.exists():
            raise FileNotFoundError(
                f"Tiny ImageNet is missing at {train_dir}. "
                "Run once with dataset download enabled or place tiny-imagenet-200 under .data."
            )
        train = datasets.ImageFolder(str(train_dir), transform=train_transform(image_size))
        val = datasets.ImageFolder(str(val_dir), transform=eval_transform(image_size)) if val_dir else None
        classes = 200
    else:
        raise ValueError(f"unsupported dataset: {name}")

    if max_samples > 0:
        train = Subset(train, range(min(max_samples, len(train))))
    return DatasetSpec(
        name=dataset_name,
        classes=classes,
        image_size=image_size,
        train=train,
        val=val,
    )


def shard_dataset(dataset: Dataset, start: int, stop: int) -> Subset:
    bounded_start = max(0, min(len(dataset), start))
    bounded_stop = max(bounded_start, min(len(dataset), stop))
    return Subset(dataset, range(bounded_start, bounded_stop))


def dataset_to_cifar_arrays(dataset: Dataset) -> tuple[torch.Tensor, torch.Tensor]:
    raise NotImplementedError("array conversion is intentionally not used for heavy datasets")
