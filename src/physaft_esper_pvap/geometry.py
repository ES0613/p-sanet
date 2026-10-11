"""Deterministic molecular geometry used by the trained models."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem, rdMolDescriptors

from .constants import MAX_EMBED_ATTEMPTS, N_CONFORMERS, RADIUS_SCALE, UFF_MAX_ITERS
from .exceptions import GeometryError
from .smiles import CanonicalSmiles

MAX_RANDOM_SEED = 2_147_483_647
SEED0 = 2026


@dataclass(frozen=True)
class GeometryResult:
    input_smiles: str
    canonical_isomeric_smiles: str
    canonical_nonisomeric_smiles: str
    vdw_surface_area_a2: float
    vdw_volume_a3: float
    pcsaft_m: float
    pcsaft_sigma_a: float
    radius_scale: float
    conformers_requested: int
    conformers_embedded: int
    uff_converged: int
    uff_not_converged: int
    uff_energy_min: float
    uff_energy_mean: float
    uff_energy_max: float
    atoms_with_hydrogen: int
    area_volume_method: str
    force_field: str
    rdkit_version: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def smiles_hash_seed(smiles: str, offset: int = 0) -> int:
    digest = hashlib.sha256(smiles.strip().encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return int((SEED0 + value + int(offset)) % (MAX_RANDOM_SEED - 1) + 1)


def _embed_100(mol: Chem.Mol, smiles: str, num_threads: int) -> list[int]:
    conformer_ids: list[int] = []
    for attempt in range(MAX_EMBED_ATTEMPTS):
        missing = N_CONFORMERS - len(conformer_ids)
        if missing <= 0:
            break
        params = AllChem.ETKDGv3()
        params.randomSeed = smiles_hash_seed(smiles, offset=attempt * 10_000)
        params.pruneRmsThresh = -1.0
        params.numThreads = int(num_threads)
        params.clearConfs = attempt == 0
        params.ignoreSmoothingFailures = True
        params.useRandomCoords = attempt >= 25
        known = set(conformer_ids)
        new_ids = AllChem.EmbedMultipleConfs(mol, numConfs=missing, params=params)
        conformer_ids.extend(int(conf_id) for conf_id in new_ids if int(conf_id) not in known)
    conformer_ids = conformer_ids[:N_CONFORMERS]
    if len(conformer_ids) != N_CONFORMERS:
        raise GeometryError(
            f"Required {N_CONFORMERS} conformers, but RDKit embedded only {len(conformer_ids)}."
        )
    return conformer_ids


def _pcsaft_shape_parameters(area: float, volume: float) -> tuple[float, float]:
    if not (math.isfinite(area) and math.isfinite(volume) and area > 0.0 and volume > 0.0):
        raise GeometryError("Non-finite or non-positive mean VDW area/volume.")
    sigma = 6.0 * volume / area
    m = area**3 / (36.0 * math.pi * volume**2)
    return float(m), float(sigma)


def calculate_geometry(canonical: CanonicalSmiles, num_threads: int = 1) -> GeometryResult:
    """Calculate the exact 100-conformer descriptors used during training."""
    base = Chem.MolFromSmiles(canonical.geometry_smiles)
    if base is None:  # defensive; canonicalization has already validated it
        raise GeometryError("RDKit could not reconstruct the canonical SMILES.")
    mol = Chem.AddHs(base)
    conformer_ids = _embed_100(mol, canonical.geometry_smiles, max(1, int(num_threads)))
    optimization = AllChem.UFFOptimizeMoleculeConfs(
        mol,
        numThreads=max(1, int(num_threads)),
        maxIters=UFF_MAX_ITERS,
    )
    if len(optimization) != N_CONFORMERS:
        raise GeometryError(
            f"Expected {N_CONFORMERS} UFF results, received {len(optimization)}."
        )

    periodic_table = Chem.GetPeriodicTable()
    scaled_radii = [
        float(periodic_table.GetRvdw(atom.GetAtomicNum()) * RADIUS_SCALE)
        for atom in mol.GetAtoms()
    ]
    areas: list[float] = []
    volumes: list[float] = []
    for conformer_id in conformer_ids:
        lattice = rdMolDescriptors.DoubleCubicLatticeVolume(
            mol,
            scaled_radii,
            False,
            True,
            0.0,
            int(conformer_id),
        )
        areas.append(float(lattice.GetSurfaceArea()))
        volumes.append(float(lattice.GetVDWVolume()))

    mean_area = float(np.mean(np.asarray(areas, dtype=np.float64)))
    mean_volume = float(np.mean(np.asarray(volumes, dtype=np.float64)))
    m_value, sigma_value = _pcsaft_shape_parameters(mean_area, mean_volume)
    flags = np.asarray([int(item[0]) for item in optimization], dtype=np.int64)
    energies = np.asarray([float(item[1]) for item in optimization], dtype=np.float64)
    return GeometryResult(
        input_smiles=canonical.input_smiles,
        canonical_isomeric_smiles=canonical.geometry_smiles,
        canonical_nonisomeric_smiles=canonical.embedding_smiles,
        vdw_surface_area_a2=mean_area,
        vdw_volume_a3=mean_volume,
        pcsaft_m=m_value,
        pcsaft_sigma_a=sigma_value,
        radius_scale=RADIUS_SCALE,
        conformers_requested=N_CONFORMERS,
        conformers_embedded=len(conformer_ids),
        uff_converged=int(np.sum(flags == 0)),
        uff_not_converged=int(np.sum(flags != 0)),
        uff_energy_min=float(np.min(energies)),
        uff_energy_mean=float(np.mean(energies)),
        uff_energy_max=float(np.max(energies)),
        atoms_with_hydrogen=mol.GetNumAtoms(),
        area_volume_method="RDKit DoubleCubicLatticeVolume; probe radius 0 A",
        force_field=f"UFF; maxIters={UFF_MAX_ITERS}",
        rdkit_version=rdBase.rdkitVersion,
    )


__all__ = ["GeometryResult", "calculate_geometry", "smiles_hash_seed"]
