import math

import pytest

from physaft_thermo.geometry import calculate_geometry
from physaft_thermo.smiles import canonicalize_smiles


def test_100_conformer_geometry_matches_public_fixture() -> None:
    result = calculate_geometry(canonicalize_smiles("CC1CCCC1C"))
    assert result.conformers_embedded == 100
    assert result.radius_scale == 0.856
    assert result.vdw_surface_area_a2 == pytest.approx(144.54318617734353, abs=1e-6)
    assert result.vdw_volume_a3 == pytest.approx(89.88961738286953, abs=1e-6)
    assert result.pcsaft_m == pytest.approx(3.3046192918322, abs=1e-8)
    assert result.pcsaft_sigma_a == pytest.approx(3.731325692762098, abs=1e-8)
    assert result.pcsaft_sigma_a == pytest.approx(
        6.0 * result.vdw_volume_a3 / result.vdw_surface_area_a2
    )
    assert result.pcsaft_m == pytest.approx(
        result.vdw_surface_area_a2**3
        / (36.0 * math.pi * result.vdw_volume_a3**2)
    )
