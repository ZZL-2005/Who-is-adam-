# Make local 'datasets' a proper package to avoid conflicts with third-party 'datasets'.
# This ensures imports like `from datasets.cifar10 import ...` resolve to this project.

__all__ = [
    "cifar10",
]
