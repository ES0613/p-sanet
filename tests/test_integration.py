import os

import pytest

from physaft_thermo import predict, predict_range


@pytest.mark.integration
def test_public_end_to_end_fixture(tmp_path) -> None:
    if not os.environ.get("SMI_TED_HOME"):
        pytest.skip("Set SMI_TED_HOME to the separately downloaded SMI-TED-Light directory.")
    result = predict("CCO", [280.0, 290.0, 300.0, 310.0, 320.0])
    assert not result.has_failures
    assert len(result.gchg_rt) == 25
    assert len(result.epsilon_fits) == 5
    assert len(result.pvap_by_model) == 25
    assert result.ensemble_summary["successful_model_count"].eq(5).all()
    assert result.pvap_by_model["fugacity_residual_abs"].le(1e-5).all()
    output = result.write(tmp_path / "prediction")
    assert sorted(path.name for path in output.iterdir()) == [
        "ensemble_summary.csv", "epsilon_fits.csv", "gchg_rt.csv",
        "geometry.json", "pvap_by_model.csv", "run_metadata.json",
    ]


@pytest.mark.integration
def test_vaporization_and_critical_end_to_end(tmp_path) -> None:
    if not os.environ.get("SMI_TED_HOME"):
        pytest.skip("Set SMI_TED_HOME to the separately downloaded SMI-TED-Light directory.")
    result = predict_range("CCO", 280.0, 320.0, n_points=5, models=["model_1"],
        properties=["vapor_pressure", "liquid_density", "vapor_density",
                    "enthalpy", "entropy", "heat_of_vaporization",
                    "entropy_of_vaporization", "critical_properties"])
    assert not result.has_failures
    assert result.pvap_by_model.rho_liq_kg_m3.gt(result.pvap_by_model.rho_vap_kg_m3).all()
    assert result.thermodynamics_by_model.Hvap_J_mol.gt(0).all()
    caloric = result.thermodynamics_by_model
    assert (caloric.Hres_vap_J_mol - caloric.Hres_liq_J_mol - caloric.Hvap_J_mol).abs().lt(1e-6).all()
    assert (caloric.Sres_vap_J_mol_K - caloric.Sres_liq_J_mol_K - caloric.Svap_J_mol_K).abs().lt(1e-8).all()
    assert result.critical_properties_by_model.status.isin({"ok", "closest_point_no_intersection"}).all()
    output = result.write(tmp_path / "properties")
    for name in ("thermodynamics_by_model.csv", "thermodynamics_summary.csv",
                 "critical_properties_by_model.csv", "critical_properties_summary.csv",
                 "critical_gchg_diagnostics.svg"):
        assert (output / name).is_file()
