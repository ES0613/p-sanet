"""Property selection and output contracts without external encoder downloads."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from p_sanet import pipeline
from p_sanet.cli import main
from p_sanet.critical_properties import CriticalRun


@pytest.fixture
def mocked_inference(monkeypatch):
    values = dict(canonical_isomeric_smiles="CCO", canonical_nonisomeric_smiles="CCO",
                  vdw_volume_a3=100., vdw_surface_area_a2=150., pcsaft_m=3., pcsaft_sigma_a=3.5)
    geometry = SimpleNamespace(**values, to_dict=lambda: values.copy())
    monkeypatch.setattr(pipeline, "calculate_geometry", lambda _: geometry)
    monkeypatch.setattr(pipeline, "verify_smi_ted", lambda _: {"mock": "verified"})
    monkeypatch.setattr(pipeline, "load_encoder", lambda _: object())
    monkeypatch.setattr(pipeline, "encode_smiles", lambda *_: np.zeros(768))
    monkeypatch.setattr(pipeline, "load_model", lambda *_, **__: object())
    monkeypatch.setattr(pipeline, "predict_gchg_rt", lambda _l, _e, _v, _a, ts, **__: -np.ones_like(ts))
    fit = SimpleNamespace(valid=True, coefficients=np.asarray([250., 500., .01]),
        to_dict=lambda: {"epsilon0_k": 250., "epsilon1_k2": 500., "epsilon2_dimensionless": .01},
        epsilon=lambda t: 250 + 500 / t + .01 * t)
    monkeypatch.setattr(pipeline, "fit_epsilon_curve", lambda *_: fit)
    return geometry


def test_vaporization_reuses_vle_and_writes_densities(mocked_inference, tmp_path):
    result = pipeline.predict_range("CCO", 280, 320, n_points=5, models=["model_1"],
                                    properties=["heat_of_vaporization"])
    assert not result.has_failures
    assert result.critical_properties_by_model.empty
    assert result.thermodynamics_by_model.Hvap_kJ_mol.gt(0).all()
    assert result.pvap_by_model.rho_liq_kg_m3.gt(result.pvap_by_model.rho_vap_kg_m3).all()
    result.write(tmp_path)
    assert (tmp_path / "thermodynamics_summary.csv").is_file()
    assert not (tmp_path / "critical_properties_by_model.csv").exists()


@pytest.mark.parametrize("property_name,expected_columns", [
    ("enthalpy", {"Hres_liq_J_mol", "Hres_vap_J_mol"}),
    ("entropy", {"Sres_liq_J_mol_K", "Sres_vap_J_mol_K"}),
    ("heat_of_vaporization", {"Hvap_J_mol", "Hvap_kJ_mol"}),
    ("entropy_of_vaporization", {"Svap_J_mol_K"}),
])
def test_caloric_selection_and_output(mocked_inference, tmp_path, property_name, expected_columns):
    result = pipeline.predict_range("CCO", 280, 320, n_points=5, models=["model_1"],
                                    properties=[property_name])
    assert not result.has_failures
    all_columns = set().union(*map(set, pipeline.CALORIC_VALUES.values()))
    assert expected_columns <= set(result.thermodynamics_by_model)
    assert set(result.thermodynamics_by_model) & (all_columns - expected_columns) == set()
    assert {f"{column}_mean" for column in expected_columns} <= set(result.thermodynamics_summary)
    result.write(tmp_path)
    saved = pd.read_csv(tmp_path / "thermodynamics_by_model.csv")
    assert expected_columns <= set(saved)
    assert (tmp_path / "pvap_by_model.csv").is_file()


def test_critical_only_skips_vle_grid(mocked_inference, monkeypatch, tmp_path):
    def forbidden(*_, **__):
        pytest.fail("Critical-only must skip the requested VLE grid")
    monkeypatch.setattr(pipeline, "predict_gchg_rt", forbidden)
    monkeypatch.setattr(pipeline, "fit_epsilon_curve", forbidden)
    monkeypatch.setattr(pipeline, "solve_vapor_pressure", forbidden)
    frame = pd.DataFrame([{"model_id": "model_1", "Tc_K": 500., "Pc_Pa": 1e6,
        "status": "closest_point_no_intersection", "message": "approximate",
        "is_approximate": True, "intersection_found": False}])
    curves = pd.DataFrame([{"model_id": "model_1", "T_K": t, "Gchg_over_RT": -18.,
        "critical_target_gchg_rt": -20.} for t in (11.5002, 3728.)])
    monkeypatch.setattr(pipeline, "_critical_run", lambda *_: CriticalRun(frame, curves,
        pd.DataFrame(), pd.DataFrame(), pd.DataFrame()))
    result = pipeline.predict_range("CCO", 280, 320, n_points=5, models=["model_1"],
                                    properties=["critical_properties"])
    assert not result.has_failures
    assert result.pvap_by_model.empty and result.epsilon_fits.empty
    result.write(tmp_path)
    assert (tmp_path / "critical_gchg_diagnostics.svg").is_file()
    assert not (tmp_path / "pvap_by_model.csv").exists()


@pytest.mark.parametrize("properties", [[], ["unknown"], ["liquid_density", "liquid_density"]])
def test_invalid_property_selection_is_rejected(properties):
    with pytest.raises(ValueError):
        pipeline.predict_range("CCO", 280, 320, n_points=5, properties=properties)


def test_cli_property_selection(mocked_inference, tmp_path):
    output = tmp_path / "caloric"
    assert main(["predict", "--smiles", "CCO", "--t-min", "280", "--t-max", "320",
                 "--n-points", "5", "--models", "model_1", "--properties",
                 "enthalpy,entropy,heat_of_vaporization,entropy_of_vaporization",
                 "--output", str(output), "--strict"]) == 0
    assert (output / "thermodynamics_by_model.csv").is_file()
    assert set().union(*map(set, pipeline.CALORIC_VALUES.values())) <= set(
        pd.read_csv(output / "thermodynamics_by_model.csv"))
