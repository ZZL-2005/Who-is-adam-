from typing import Callable, Dict

import torch.nn as nn

from .ResNet import build_resnet18
from .VGG import build_vgg11

MODEL_REGISTRY: Dict[str, Callable[[], nn.Module]] = {
    "ResNet18": build_resnet18,
    "VGG11": build_vgg11,
}


def build_model(name: str, num_classes: int = 10, pretrained: bool = False) -> nn.Module:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY)}")

    builder = MODEL_REGISTRY[name]
    weights = "DEFAULT" if pretrained else None
    return builder(num_classes=num_classes, weights=weights)
