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


def _positive_widths(name: str, widths: Sequence[int]) -> tuple[int, ...]:
    parsed = tuple(int(width) for width in widths)
    if not parsed or any(width <= 0 for width in parsed):
        raise ValueError(f"{name} must contain positive widths")
    return parsed


def _mlp(input_width: int, widths: Sequence[int], *, final_width: int | None = None) -> nn.Sequential:
    layers: list[nn.Module] = []
    width = input_width
    for hidden in widths:
        layers.extend((nn.Linear(width, hidden), nn.ReLU()))
        width = hidden
    if final_width is not None:
        layers.append(nn.Linear(width, final_width))
    return nn.Sequential(*layers)


class InputTNet(nn.Module):
    """Small PointNet-style input transform, initialized to the identity matrix."""

    def __init__(self, widths: Sequence[int] = (32, 64)):
        super().__init__()
        encoder_widths = _positive_widths("alignment_widths", widths)
        self.encoder = _mlp(3, encoder_widths)
        self.head = _mlp(encoder_widths[-1], (32,), final_width=9)
        final = self.head[-1]
        if not isinstance(final, nn.Linear):
            raise AssertionError("input transform head must end with a linear layer")
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError("input alignment expects an N x 3 XYZ tensor")
        if not len(xyz):
            return torch.eye(3, dtype=xyz.dtype, device=xyz.device)
        encoded = self.encoder(xyz)
        summary = encoded.max(dim=0).values
        correction = self.head(summary).reshape(3, 3)
        return correction + torch.eye(3, dtype=xyz.dtype, device=xyz.device)


class GlobalPointEncoder(nn.Module):
    """Shared PointNet-style scan encoder used by G1 and GT."""

    def __init__(
        self,
        input_features: int,
        point_widths: Sequence[int] = (64, 128),
        *,
        input_alignment: str = "none",
        alignment_widths: Sequence[int] = (32, 64),
    ):
        super().__init__()
        if input_features < 3:
            raise ValueError("global PointNet requires XYZ as its first three features")
        point_widths = _positive_widths("point_widths", point_widths)
        if input_alignment not in {"none", "tnet"}:
            raise ValueError("input_alignment must be 'none' or 'tnet'")
        self.input_features = input_features
        self.embedding_width = point_widths[-1]
        self.input_alignment = input_alignment
        self.alignment = InputTNet(alignment_widths) if input_alignment == "tnet" else None
        self.encoder = _mlp(input_features, point_widths)
        self._last_transform: torch.Tensor | None = None

    def encode_scan(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if features.ndim != 2 or features.shape[1] != self.input_features:
            raise ValueError(
                f"global PointNet expects N x {self.input_features} feature tensors"
            )
        if not len(features):
            self._last_transform = None
            return (
                features.new_empty((0, self.embedding_width)),
                features.new_zeros(self.embedding_width),
                None,
            )
        model_features = features
        transform = None
        if self.alignment is not None:
            transform = self.alignment(features[:, :3])
            aligned_xyz = features[:, :3] @ transform.T
            model_features = torch.cat((aligned_xyz, features[:, 3:]), dim=1)
            self._last_transform = transform
        else:
            self._last_transform = None
        point_embedding = self.encoder(model_features)
        scan_summary = point_embedding.max(dim=0).values
        return point_embedding, scan_summary, transform

    @staticmethod
    def transform_penalty(transform: torch.Tensor) -> torch.Tensor:
        identity = torch.eye(3, dtype=transform.dtype, device=transform.device)
        gram = transform @ transform.T
        return torch.mean(torch.square(gram - identity))

    def regularization_loss(self) -> torch.Tensor:
        """Encourage the learned XYZ transform to remain close to orthogonal."""

        if self._last_transform is None:
            return next(self.parameters()).new_zeros(())
        return self.transform_penalty(self._last_transform)


class GlobalPointNet(GlobalPointEncoder):
    """Compact PointNet-style semantic segmentation with optional XYZ alignment."""

    def __init__(
        self,
        input_features: int,
        point_widths: Sequence[int] = (64, 128),
        head_widths: Sequence[int] = (64,),
        *,
        input_alignment: str = "none",
        alignment_widths: Sequence[int] = (32, 64),
    ):
        super().__init__(
            input_features,
            point_widths,
            input_alignment=input_alignment,
            alignment_widths=alignment_widths,
        )
        head_widths = _positive_widths("head_widths", head_widths)
        self.head = _mlp(self.embedding_width * 2, head_widths, final_width=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        point_embedding, scan_summary, _ = self.encode_scan(features)
        if not len(point_embedding):
            return features.new_empty(0)
        combined = torch.cat(
            (point_embedding, scan_summary.unsqueeze(0).expand_as(point_embedding)), dim=1
        )
        return self.head(combined).squeeze(-1)


class VoxelPointNet(nn.Module):
    """PointNet classifier with independent max pooling for each occupied voxel."""

    def __init__(self, input_features: int, point_widths: Sequence[int], head_widths: Sequence[int]):
        super().__init__()
        if input_features != 5:
            raise ValueError("voxel PointNet requires local XYZ, intensity, and a voxel index")
        widths = _positive_widths("point_widths", point_widths)
        self.encoder = _mlp(4, widths)
        self.head = _mlp(widths[-1], _positive_widths("head_widths", head_widths), final_width=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != 5:
            raise ValueError("voxel PointNet expects N x 5 feature tensors")
        if not len(features):
            return features.new_empty(0)
        indices = features[:, 4].long()
        _, inverse = torch.unique(indices, sorted=True, return_inverse=True)
        embeddings = self.encoder(features[:, :4])
        pooled = embeddings.new_full((int(inverse.max().item()) + 1, embeddings.shape[1]), -torch.inf)
        pooled = pooled.scatter_reduce(0, inverse[:, None].expand_as(embeddings), embeddings, reduce="amax", include_self=True)
        return self.head(pooled).squeeze(-1)[inverse]


class GlobalTemporalPointNet(nn.Module):
    """Five-scan global context model that predicts points in the current scan."""

    def __init__(
        self,
        input_features: int,
        point_widths: Sequence[int] = (64, 128),
        head_widths: Sequence[int] = (64,),
        *,
        history_length: int = 5,
        gru_hidden_width: int = 32,
        input_alignment: str = "none",
        alignment_widths: Sequence[int] = (32, 64),
    ):
        super().__init__()
        if history_length <= 0:
            raise ValueError("history_length must be positive")
        if gru_hidden_width <= 0:
            raise ValueError("gru_hidden_width must be positive")
        head_widths = _positive_widths("head_widths", head_widths)
        self.history_length = int(history_length)
        self.gru_hidden_width = int(gru_hidden_width)
        self.scan_encoder = GlobalPointEncoder(
            input_features,
            point_widths,
            input_alignment=input_alignment,
            alignment_widths=alignment_widths,
        )
        self.gru = nn.GRU(
            input_size=self.scan_encoder.embedding_width,
            hidden_size=self.gru_hidden_width,
            batch_first=True,
        )
        self.head = _mlp(
            self.scan_encoder.embedding_width * 2 + self.gru_hidden_width,
            head_widths,
            final_width=1,
        )
        self._last_transforms: tuple[torch.Tensor, ...] = ()

    def encode_scan(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        return self.scan_encoder.encode_scan(features)

    def predict_from_encoded(
        self,
        current_embedding: torch.Tensor,
        current_summary: torch.Tensor,
        history_summaries: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        if not history_summaries or len(history_summaries) > self.history_length:
            raise ValueError("history summaries must contain 1 through history_length scans")
        if current_summary.shape != (self.scan_encoder.embedding_width,):
            raise ValueError("current scan summary has the wrong shape")
        summaries = torch.stack(tuple(history_summaries), dim=0)
        if summaries.ndim != 2 or summaries.shape[1] != self.scan_encoder.embedding_width:
            raise ValueError("history summaries have the wrong shape")
        _, hidden = self.gru(summaries.unsqueeze(0))
        temporal_summary = hidden[-1, 0]
        if not len(current_embedding):
            return current_embedding.new_empty(0)
        combined = torch.cat(
            (
                current_embedding,
                current_summary.unsqueeze(0).expand_as(current_embedding),
                temporal_summary.unsqueeze(0).expand(len(current_embedding), -1),
            ),
            dim=1,
        )
        return self.head(combined).squeeze(-1)

    def forward(self, scan_features: Sequence[torch.Tensor]) -> torch.Tensor:
        if not scan_features or len(scan_features) > self.history_length:
            raise ValueError("GT requires 1 through history_length scans")
        encodings = [self.encode_scan(features) for features in scan_features]
        current_embedding, current_summary, _ = encodings[-1]
        self._last_transforms = tuple(
            transform for _, _, transform in encodings if transform is not None
        )
        return self.predict_from_encoded(
            current_embedding,
            current_summary,
            tuple(summary for _, summary, _ in encodings),
        )

    def regularization_loss(self) -> torch.Tensor:
        if not self._last_transforms:
            return next(self.parameters()).new_zeros(())
        return torch.stack(
            [self.scan_encoder.transform_penalty(transform) for transform in self._last_transforms]
        ).mean()


def create_model(name: str, input_features: int, parameters: dict) -> nn.Module:
    if name == "local_mlp":
        return LocalMLP(input_features, tuple(int(v) for v in parameters["hidden_widths"]))
    if name == "voxel_pointnet":
        return VoxelPointNet(input_features, parameters["point_widths"], parameters["head_widths"])
    if name == "global_pointnet":
        return GlobalPointNet(
            input_features,
            tuple(int(v) for v in parameters["point_widths"]),
            tuple(int(v) for v in parameters["head_widths"]),
            input_alignment=str(parameters.get("input_alignment", "none")),
            alignment_widths=tuple(int(v) for v in parameters.get("alignment_widths", (32, 64))),
        )
    if name == "global_temporal_pointnet":
        return GlobalTemporalPointNet(
            input_features,
            tuple(int(v) for v in parameters["point_widths"]),
            tuple(int(v) for v in parameters["head_widths"]),
            history_length=int(parameters.get("history_length", 5)),
            gru_hidden_width=int(parameters.get("gru_hidden_width", 32)),
            input_alignment=str(parameters.get("input_alignment", "none")),
            alignment_widths=tuple(int(v) for v in parameters.get("alignment_widths", (32, 64))),
        )
    raise ValueError(f"unknown model: {name}")
