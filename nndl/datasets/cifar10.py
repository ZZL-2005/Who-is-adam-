import random
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms


@dataclass
class DataConfig:
    data_root: str = "./data"
    batch_size: int = 128
    num_workers: int = 4
    val_split: int = 5000
    seed: int = 42
    download: bool = True


def _build_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """Create train/test transform pipelines."""
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])

    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_transform = transforms.Compose([transforms.ToTensor(), normalize])
    return train_transform, eval_transform


def _split_dataset(
    dataset: Dataset,
    val_split: int,
    seed: int,
) -> Tuple[Dataset, Dataset]:
    """Split dataset into training and validation subsets."""
    if val_split <= 0 or val_split >= len(dataset):
        return dataset, None

    generator = torch.Generator().manual_seed(seed)
    train_len = len(dataset) - val_split
    train_dataset, val_dataset = random_split(dataset, [train_len, val_split], generator=generator)
    return train_dataset, val_dataset


def create_cifar10_dataloaders(config: DataConfig) -> Dict[str, DataLoader]:
    """Create CIFAR-10 dataloaders for train/val/test splits."""
    train_transform, eval_transform = _build_transforms()
    train_dataset = datasets.CIFAR10(
        root=config.data_root,
        train=True,
        download=config.download,
        transform=train_transform,
    )
    train_subset, val_subset = _split_dataset(train_dataset, config.val_split, config.seed)

    test_dataset = datasets.CIFAR10(
        root=config.data_root,
        train=False,
        download=config.download,
        transform=eval_transform,
    )

    loaders = {
        "train": DataLoader(
            train_subset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )
    }

    if val_subset is not None:
        loaders["val"] = DataLoader(
            val_subset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )

    loaders["test"] = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    return loaders


def set_global_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
