"""Five-model Gchg/RT neural-network inference."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from torch import nn

from .constants import DEFAULT_MODELS
from .exceptions import ModelAssetError
from .utils import sha256_file


class MLPBranch(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values)


class TemperatureBranch(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values)


class ChargeFusionNet(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 768,
        smiles_hidden_dim: int = 512,
        smiles_proj_dim: int = 384,
        geometry_hidden_dim: int = 64,
        geometry_proj_dim: int = 32,
        temperature_hidden_dim: int = 64,
        temperature_proj_dim: int = 32,
        fusion_hidden_dim: int = 512,
        fusion_bottleneck_dim: int = 256,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.smiles_branch = MLPBranch(embedding_dim, smiles_hidden_dim, smiles_proj_dim, dropout)
        self.geometry_branch = MLPBranch(3, geometry_hidden_dim, geometry_proj_dim, dropout)
        self.temperature_branch = TemperatureBranch(2, temperature_hidden_dim, temperature_proj_dim, dropout)
        fusion_input_dim = smiles_proj_dim + geometry_proj_dim + temperature_proj_dim
        self.fusion_norm = nn.LayerNorm(fusion_input_dim)
        self.fusion_body = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, fusion_bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fusion_skip = nn.Linear(fusion_input_dim, fusion_bottleneck_dim)
        self.head = nn.Sequential(
            nn.LayerNorm(fusion_bottleneck_dim),
            nn.Linear(fusion_bottleneck_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        embedding: torch.Tensor,
        geometry: torch.Tensor,
        temperature_features: torch.Tensor,
    ) -> torch.Tensor:
        smiles_features = self.smiles_branch(embedding)
        geometry_features = self.geometry_branch(geometry)
        temperature_features = self.temperature_branch(temperature_features)
        fused = torch.cat([smiles_features, geometry_features, temperature_features], dim=-1)
        fused = self.fusion_norm(fused)
        fused = self.fusion_body(fused) + self.fusion_skip(fused)
        return self.head(fused)


@dataclass(frozen=True)
class LoadedModel:
    model_id: str
    network: ChargeFusionNet
    metadata: dict[str, object]


def model_directory() -> Path:
    return Path(str(files("physaft_thermo").joinpath("models")))


def load_manifest() -> dict[str, object]:
    path = model_directory() / "manifest.json"
    if not path.is_file():
        raise ModelAssetError(f"Missing model manifest: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_model(model_id: str, device: str = "cpu") -> LoadedModel:
    model_id = str(model_id)
    if model_id not in DEFAULT_MODELS:
        raise ModelAssetError(f"Unsupported model {model_id!r}; available models are {DEFAULT_MODELS}.")
    root = model_directory()
    metadata_path = root / f"{model_id}.json"
    weights_path = root / f"{model_id}.safetensors"
    if not metadata_path.is_file() or not weights_path.is_file():
        raise ModelAssetError(f"Missing converted assets for model {model_id!r} in {root}.")
    metadata_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = str(metadata_payload["safetensors_sha256"])
    observed = sha256_file(weights_path)
    if observed != expected:
        raise ModelAssetError(
            f"Model SHA-256 mismatch for {model_id!r}: expected {expected}, observed {observed}."
        )
    architecture = dict(metadata_payload["architecture"])
    network = ChargeFusionNet(**architecture).to(torch.device(device))
    state = load_file(str(weights_path), device="cpu")
    network.load_state_dict(state, strict=True)
    network.eval()
    return LoadedModel(model_id=model_id, network=network, metadata=metadata_payload)


def predict_gchg_rt(
    loaded: LoadedModel,
    embedding: np.ndarray,
    volume_a3: float,
    area_a2: float,
    temperatures_k: np.ndarray,
    device: str = "cpu",
) -> np.ndarray:
    metadata_payload = loaded.metadata
    geometry_scaler = dict(metadata_payload["geometry_scaler"])
    temperature_scaler = dict(metadata_payload["temperature_scaler"])
    target_scaler = dict(metadata_payload["target_scaler"])
    geometry_raw = np.asarray([volume_a3, area_a2, volume_a3 / area_a2], dtype=np.float32)
    geometry = (
        geometry_raw - np.asarray(geometry_scaler["mean"], dtype=np.float32)
    ) / np.asarray(geometry_scaler["std"], dtype=np.float32)
    temperatures = np.asarray(temperatures_k, dtype=np.float32)
    temperature_raw = np.column_stack([temperatures, 1.0 / temperatures]).astype(np.float32)
    temperature_features = (
        temperature_raw - np.asarray(temperature_scaler["mean"], dtype=np.float32)
    ) / np.asarray(temperature_scaler["std"], dtype=np.float32)
    count = len(temperatures)
    torch_device = torch.device(device)
    embedding_tensor = torch.as_tensor(embedding, dtype=torch.float32, device=torch_device).reshape(1, -1).repeat(count, 1)
    geometry_tensor = torch.as_tensor(geometry, dtype=torch.float32, device=torch_device).reshape(1, -1).repeat(count, 1)
    temperature_tensor = torch.as_tensor(temperature_features, dtype=torch.float32, device=torch_device)
    with torch.inference_mode():
        scaled = loaded.network(embedding_tensor, geometry_tensor, temperature_tensor).squeeze(-1)
    prediction = scaled.detach().cpu().numpy().astype(np.float64)
    prediction = prediction * float(target_scaler["std"]) + float(target_scaler["mean"])
    if prediction.shape != temperatures.shape or not np.all(np.isfinite(prediction)):
        raise ModelAssetError(f"Model {loaded.model_id!r} produced invalid Gchg/RT predictions.")
    return prediction


def predict_gchg_rt_derivative(
    loaded: LoadedModel,
    embedding: np.ndarray,
    volume_a3: float,
    area_a2: float,
    temperatures_k: np.ndarray,
    device: str = "cpu",
) -> np.ndarray:
    """Network derivative d(Gchg/RT)/dT using the trained feature scaling."""
    metadata = loaded.metadata
    geometry_scaler = metadata["geometry_scaler"]
    temperature_scaler = metadata["temperature_scaler"]
    raw = np.asarray([volume_a3, area_a2, volume_a3 / area_a2], dtype=np.float32)
    geometry = (raw - np.asarray(geometry_scaler["mean"], dtype=np.float32)) / np.asarray(
        geometry_scaler["std"], dtype=np.float32
    )
    count = len(temperatures_k)
    target = torch.device(device)
    embedding_tensor = torch.as_tensor(embedding, dtype=torch.float32, device=target).reshape(1, -1).repeat(count, 1)
    geometry_tensor = torch.as_tensor(geometry, dtype=torch.float32, device=target).reshape(1, -1).repeat(count, 1)
    with torch.enable_grad():
        temperature = torch.tensor(np.asarray(temperatures_k, dtype=np.float32), device=target, requires_grad=True)
        features = torch.stack([temperature, 1.0 / temperature], dim=1)
        mean = torch.as_tensor(temperature_scaler["mean"], dtype=torch.float32, device=target)
        std = torch.as_tensor(temperature_scaler["std"], dtype=torch.float32, device=target)
        prediction = loaded.network(embedding_tensor, geometry_tensor, (features - mean) / std).squeeze(-1)
        prediction = prediction * float(metadata["target_scaler"]["std"]) + float(metadata["target_scaler"]["mean"])
        gradient = torch.autograd.grad(prediction.sum(), temperature)[0]
    values = gradient.detach().cpu().numpy().astype(np.float64)
    if not np.all(np.isfinite(values)):
        raise ModelAssetError(f"Model {loaded.model_id!r} produced non-finite Gchg/RT derivatives.")
    return values


__all__ = [
    "ChargeFusionNet",
    "LoadedModel",
    "load_manifest",
    "load_model",
    "predict_gchg_rt",
    "predict_gchg_rt_derivative",
]
