from typing import Literal, Optional

import torch.nn as nn
import torch.nn.init as init
from torchvision.models import ResNet18_Weights, resnet18

__all__ = ["build_resnet18"]


def _configure_stem(model: nn.Module, use_pretrained: bool) -> None:
    """Adapt the ResNet stem for CIFAR-10 sized inputs."""
    if isinstance(model.conv1, nn.Conv2d):
        if use_pretrained:
            model.conv1.stride = (1, 1)
            model.conv1.padding = (3, 3)
        else:
            new_conv = nn.Conv2d(
                in_channels=model.conv1.in_channels,
                out_channels=model.conv1.out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
            init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
            model.conv1 = new_conv
    if hasattr(model, "maxpool"):
        model.maxpool = nn.Identity()


def build_resnet18(num_classes: int = 10, weights: Optional[Literal["DEFAULT"]] = None) -> nn.Module:
    """
    Create a ResNet-18 classifier adapted for CIFAR-10.

    Parameters
    ----------
    num_classes: number of output classes.
    weights: pass "DEFAULT" to load torchvision pretrained weights, otherwise random init.
    """
    pretrained_weights = ResNet18_Weights.DEFAULT if weights == "DEFAULT" else None
    model = resnet18(weights=pretrained_weights)
    _configure_stem(model, use_pretrained=pretrained_weights is not None)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
