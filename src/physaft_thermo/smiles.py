"""SMILES input validation and canonicalization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rdkit import Chem

from .exceptions import InputValidationError


@dataclass(frozen=True)
class CanonicalSmiles:
    input_smiles: str
    geometry_smiles: str
    embedding_smiles: str


def canonicalize_smiles(smiles: str) -> CanonicalSmiles:
    if not isinstance(smiles, str) or not smiles.strip():
        raise InputValidationError("SMILES must be a non-empty string.")
    raw = smiles.strip()
    mol = Chem.MolFromSmiles(raw)
    if mol is None:
        raise InputValidationError(f"Invalid SMILES: {raw!r}")
    if len(Chem.GetMolFrags(mol)) != 1:
        raise InputValidationError("Only single-fragment, pure-component SMILES are supported; '.' is not allowed.")
    geometry = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    embedding = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    return CanonicalSmiles(raw, geometry, embedding)


def validate_temperatures(temperatures_k: object) -> np.ndarray:
    try:
        values = np.asarray(temperatures_k, dtype=float)
    except (TypeError, ValueError) as exc:
        raise InputValidationError("temperatures_k must be a one-dimensional numeric sequence.") from exc
    if values.ndim != 1:
        raise InputValidationError("temperatures_k must be one-dimensional.")
    if len(values) < 5:
        raise InputValidationError("At least five distinct temperatures are required for epsilon(T) regression.")
    if not np.all(np.isfinite(values)):
        raise InputValidationError("All temperatures must be finite.")
    if np.any(values <= 0.0):
        raise InputValidationError("All temperatures must be greater than 0 K.")
    if len(np.unique(values)) != len(values):
        raise InputValidationError("All temperatures must be distinct.")
    return values.astype(np.float64, copy=False)
