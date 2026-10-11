import numpy as np
import pytest

from physaft_esper_pvap.exceptions import InputValidationError
from physaft_esper_pvap.smiles import canonicalize_smiles, validate_temperatures


def test_canonicalization_keeps_geometry_stereo_but_not_embedding_stereo() -> None:
    result = canonicalize_smiles("C/C=C/C=O")
    assert "/" in result.geometry_smiles or "\\" in result.geometry_smiles
    assert "/" not in result.embedding_smiles
    assert "\\" not in result.embedding_smiles


@pytest.mark.parametrize("smiles", ["", "not-a-smiles", "CCO.O"])
def test_invalid_or_multifragment_smiles_are_rejected(smiles: str) -> None:
    with pytest.raises(InputValidationError):
        canonicalize_smiles(smiles)


def test_temperature_validation() -> None:
    values = validate_temperatures([280, 290, 300, 310, 320])
    assert values.dtype == np.float64
    for invalid in (
        [280, 290, 300, 310],
        [280, 290, 300, 310, 310],
        [280, 290, 300, 310, np.nan],
        [0, 10, 20, 30, 40],
    ):
        with pytest.raises(InputValidationError):
            validate_temperatures(invalid)
