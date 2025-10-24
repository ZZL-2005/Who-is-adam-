from typing import Literal, Optional

import torch.nn as nn
from torchvision.models import VGG11_BN_Weights, vgg11_bn


def build_vgg11(num_classes: int = 10, weights: Optional[Literal["DEFAULT"]] = None) -> nn.Module:
    """
    Create a VGG11-BN model for CIFAR-10 classification.

    Parameters
    ----------
    num_classes: output dimension.
    weights: pass "DEFAULT" to load torchvision pretrained weights, otherwise initialized from scratch.
    """
    pretrained = VGG11_BN_Weights.DEFAULT if weights == "DEFAULT" else None
    model = vgg11_bn(weights=pretrained)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model
