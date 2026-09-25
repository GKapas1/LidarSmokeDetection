from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class LocalMLP(nn.Module):
    def __init__(self, input_features: int, hidden_widths: Sequence[int] = (64, 32)):
        super().__init__()
        if input_features <= 0 or not hidden_widths or any(width <= 0 for width in hidden_widths):
            raise ValueError("model widths must be positive")
        layers: list[nn.Module] = []
        width = input_features
        for hidden in hidden_widths:
            layers.extend((nn.Linear(width, hidden), nn.ReLU()))
            width = hidden
        layers.append(nn.Linear(width, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def create_model(name: str, input_features: int, parameters: dict) -> nn.Module:
    if name == "local_mlp":
        return LocalMLP(input_features, tuple(int(v) for v in parameters["hidden_widths"]))
    raise ValueError(f"unknown model: {name}")

