import numpy as np

from p_sanet.constants import DEFAULT_MODELS
from p_sanet.model import load_manifest, load_model, predict_gchg_rt


def test_manifest_and_all_model_hashes_load() -> None:
    manifest = load_manifest()
    assert tuple(item["model_id"] for item in manifest["models"]) == DEFAULT_MODELS
    for model_id in DEFAULT_MODELS:
        loaded = load_model(model_id)
        assert loaded.model_id == model_id
        assert loaded.metadata["architecture"]["embedding_dim"] == 768


def test_model_1_fixed_embedding_reference() -> None:
    prediction = predict_gchg_rt(
        load_model("model_1"),
        np.zeros(768, dtype=np.float32),
        89.88961738286953,
        144.54318617734353,
        np.asarray([250, 300, 350, 400, 450], dtype=float),
    )
    expected = np.asarray([-42.71954659, -33.57545838, -27.47155989, -23.12203068, -19.85174868])
    np.testing.assert_allclose(prediction, expected, atol=2e-5, rtol=0.0)
