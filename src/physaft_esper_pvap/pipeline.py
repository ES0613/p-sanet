"""Public API for selected saturation, vaporization, and critical properties."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

from .constants import DEFAULT_MODELS, FRAMEWORK_NAME
from .critical_properties import CRITICAL_ACCEPTED, CriticalRun, solve_critical_properties
from .diagnostics import property_summary, write_critical_diagnostics
from .epsilon import fit_epsilon_curve
from .geometry import GeometryResult, calculate_geometry
from .model import load_manifest, load_model, predict_gchg_rt, predict_gchg_rt_derivative
from .smi_ted import default_smi_ted_dir, encode_smiles, load_encoder, verify_smi_ted
from .smiles import canonicalize_smiles, validate_temperatures
from .solver import solve_vapor_pressure
from .thermodynamics import molar_density, saturation_caloric_properties
from .utils import runtime_metadata, write_json

PROPERTIES = (
    "vapor_pressure", "liquid_density", "vapor_density",
    "heat_of_vaporization", "critical_properties",
)
DEFAULT_PROPERTIES = ("vapor_pressure", "liquid_density", "vapor_density")
CRITICAL_VALUES = ["Tc_K", "Pc_Pa", "Pc_kPa", "rho_c_mol_m3", "rho_c_kg_m3", "Zc", "critical_molar_volume_m3_mol"]
CRITICAL_COLUMNS = ["model_id", *CRITICAL_VALUES, "status", "message", "is_approximate", "intersection_found"]


def _normalize_properties(properties: Iterable[str] | None) -> tuple[str, ...]:
    if properties is None:
        return DEFAULT_PROPERTIES
    if isinstance(properties, str):
        properties = properties.split(",")
    selected = tuple(str(value).strip() for value in properties)
    if not selected or len(set(selected)) != len(selected) or set(selected) - set(PROPERTIES):
        raise ValueError(f"Choose distinct properties from {PROPERTIES}.")
    return selected


@dataclass
class PredictionResult:
    geometry: GeometryResult
    gchg_rt: pd.DataFrame
    epsilon_fits: pd.DataFrame
    pvap_by_model: pd.DataFrame
    ensemble_summary: pd.DataFrame
    metadata: dict[str, object]
    thermodynamics_by_model: pd.DataFrame = field(default_factory=pd.DataFrame)
    thermodynamics_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    critical_properties_by_model: pd.DataFrame = field(default_factory=pd.DataFrame)
    critical_properties_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    critical_gchg_curves: pd.DataFrame = field(default_factory=pd.DataFrame)
    critical_candidates: pd.DataFrame = field(default_factory=pd.DataFrame)
    critical_vle_validation: pd.DataFrame = field(default_factory=pd.DataFrame)
    critical_branch_choices: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def has_failures(self) -> bool:
        stages = self.metadata["stages"]
        if stages["vle"]:
            if self.epsilon_fits.empty or not self.epsilon_fits["status"].eq("ok").all():
                return True
            if self.pvap_by_model.empty or not self.pvap_by_model["converged"].fillna(False).astype(bool).all():
                return True
        if stages["vaporization"]:
            if self.thermodynamics_by_model.empty or not self.thermodynamics_by_model["status"].eq("ok").all():
                return True
        if stages["critical"]:
            if self.critical_properties_by_model.empty or not self.critical_properties_by_model["status"].isin(CRITICAL_ACCEPTED).all():
                return True
        return False

    def write(self, output_directory: str | Path) -> Path:
        output = Path(output_directory).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "geometry.json", self.geometry.to_dict())
        if self.metadata["stages"]["vle"]:
            for name in ("gchg_rt", "epsilon_fits", "pvap_by_model", "ensemble_summary"):
                getattr(self, name).to_csv(output / f"{name}.csv", index=False)
        if self.metadata["stages"]["vaporization"]:
            self.thermodynamics_by_model.to_csv(output / "thermodynamics_by_model.csv", index=False)
            self.thermodynamics_summary.to_csv(output / "thermodynamics_summary.csv", index=False)
        if self.metadata["stages"]["critical"]:
            for name in ("critical_properties_by_model", "critical_properties_summary", "critical_gchg_curves",
                         "critical_candidates", "critical_vle_validation", "critical_branch_choices"):
                getattr(self, name).to_csv(output / f"{name}.csv", index=False)
            write_critical_diagnostics(output, self.critical_gchg_curves, self.critical_properties_by_model)
        write_json(output / "run_metadata.json", self.metadata)
        return output


def _normalize_models(models: Iterable[str]) -> tuple[str, ...]:
    normalized = tuple(str(model_id) for model_id in models)
    if not normalized or len(set(normalized)) != len(normalized) or set(normalized) - set(DEFAULT_MODELS):
        raise ValueError(f"Choose distinct models from {DEFAULT_MODELS}.")
    return normalized


def _failed_pvap_rows(model_id: str, temperatures: np.ndarray, message: str) -> list[dict[str, object]]:
    return [{"model_id": model_id, "input_index": i, "T_K": float(t), "epsilon_K": math.nan,
             "P_Pa": math.nan, "lnP_Pa": math.nan, "eq_liq_reduced_density_eta": math.nan,
             "eq_vap_reduced_density_eta": math.nan, "rho_liq_mol_m3": math.nan, "rho_vap_mol_m3": math.nan,
             "rho_liq_kg_m3": math.nan, "rho_vap_kg_m3": math.nan,
             "fugacity_residual_abs": math.nan, "phase_gap_eta": math.nan,
             "converged": False, "status": "epsilon_fit_failed", "solver_method": "none", "message": message}
            for i, t in enumerate(temperatures)]


def _ensemble_summary(pvap: pd.DataFrame, temperatures: np.ndarray) -> pd.DataFrame:
    rows = []
    for i, temperature in enumerate(temperatures):
        group = pvap.loc[pvap["input_index"].eq(i) & pvap["converged"].fillna(False).astype(bool)]
        pressures = group["P_Pa"].to_numpy(float)
        logs = group["lnP_Pa"].to_numpy(float)
        count = len(group)
        row = {"input_index": i, "T_K": float(temperature), "successful_model_count": count,
               "requested_model_count": int(pvap["model_id"].nunique()),
               "P_mean_Pa": float(pressures.mean()) if count else math.nan,
               "P_std_Pa": float(pressures.std(ddof=1)) if count > 1 else (0.0 if count else math.nan),
               "lnP_mean": float(logs.mean()) if count else math.nan,
               "lnP_std": float(logs.std(ddof=1)) if count > 1 else (0.0 if count else math.nan),
               "P_geometric_mean_Pa": float(math.exp(logs.mean())) if count else math.nan,
               "uncertainty_note": "Uncalibrated between-model ensemble spread; not a prediction interval."}
        for phase in ("liq", "vap"):
            for unit in ("mol_m3", "kg_m3"):
                column = f"rho_{phase}_{unit}"
                values = group[column].dropna().to_numpy(float)
                row[column + "_mean"] = float(values.mean()) if len(values) else math.nan
                row[column + "_std"] = float(values.std(ddof=1)) if len(values) > 1 else (0.0 if len(values) else math.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def _critical_run(geometry: GeometryResult, embedding: np.ndarray, selected_models: tuple[str, ...],
                  device: str, loaded_models: dict) -> CriticalRun:
    predictors, derivatives, failed = {}, {}, []
    for model_id in selected_models:
        try:
            loaded = loaded_models.get(model_id)
            if loaded is None:
                loaded = load_model(model_id, device=device)
            args = (embedding, geometry.vdw_volume_a3, geometry.vdw_surface_area_a2)
            predictors[model_id] = lambda ts, loaded=loaded: predict_gchg_rt(loaded, *args, ts, device=device)
            derivatives[model_id] = lambda ts, loaded=loaded: predict_gchg_rt_derivative(loaded, *args, ts, device=device)
        except Exception as exc:
            failed.append({"model_id": model_id, "status": "model_load_failed", "message": str(exc),
                           "Tc_K": math.nan, "is_approximate": False, "intersection_found": False})
    mass = Descriptors.MolWt(Chem.MolFromSmiles(geometry.canonical_isomeric_smiles))
    run = solve_critical_properties(geometry.pcsaft_m, geometry.pcsaft_sigma_a, mass, predictors, derivatives)
    if failed:
        run.properties = pd.concat([run.properties, pd.DataFrame(failed)], ignore_index=True)
    for frame in (run.candidates, run.vle_validation, run.branch_choices):
        if "canonical_smiles_nonisomeric" in frame:
            frame["canonical_smiles_nonisomeric"] = geometry.canonical_nonisomeric_smiles
    return run


def predict(smiles: str, temperatures_k: object, models: Iterable[str] = DEFAULT_MODELS,
            device: str = "cpu", properties: Iterable[str] | None = None) -> PredictionResult:
    """Predict selected properties; vaporization uses internally solved saturation states."""
    requested = _normalize_properties(properties)
    selected_models = _normalize_models(models)
    temperatures = validate_temperatures(temperatures_k)
    need_vle = bool(set(requested) - {"critical_properties"})
    need_vaporization = "heat_of_vaporization" in requested
    need_critical = "critical_properties" in requested
    canonical = canonicalize_smiles(smiles)
    geometry = calculate_geometry(canonical)
    mass = Descriptors.MolWt(Chem.MolFromSmiles(canonical.geometry_smiles))
    smi_ted_dir = default_smi_ted_dir()
    smi_ted_hashes = verify_smi_ted(smi_ted_dir)
    embedding = encode_smiles(load_encoder(smi_ted_dir), canonical.embedding_smiles)
    gchg_rows, epsilon_rows, pvap_rows, caloric_rows = [], [], [], []
    loaded_models = {}
    if need_vle:
        for model_id in selected_models:
            try:
                loaded = load_model(model_id, device=device)
                loaded_models[model_id] = loaded
                gchg = predict_gchg_rt(loaded, embedding, geometry.vdw_volume_a3,
                                       geometry.vdw_surface_area_a2, temperatures, device=device)
                gchg_rows.extend({"model_id": model_id, "input_index": i, "T_K": float(t),
                                  "Gchg_over_RT": float(g), "canonical_nonisomeric_smiles": canonical.embedding_smiles}
                                 for i, (t, g) in enumerate(zip(temperatures, gchg)))
                fit = fit_epsilon_curve(temperatures, gchg, geometry.pcsaft_m, geometry.pcsaft_sigma_a)
                epsilon_rows.append({"model_id": model_id, "status": "ok" if fit.valid else "invalid_bounds", **fit.to_dict()})
                if not fit.valid:
                    raise ValueError("epsilon(T) violates release bounds.")
            except Exception as exc:
                if not any(row["model_id"] == model_id for row in epsilon_rows):
                    epsilon_rows.append({"model_id": model_id, "status": "failed", "message": str(exc)})
                pvap_rows.extend(_failed_pvap_rows(model_id, temperatures, str(exc)))
                if need_vaporization:
                    caloric_rows.extend({"model_id": model_id, "input_index": i, "T_K": float(t),
                                         "status": "epsilon_fit_failed", "message": str(exc)}
                                        for i, t in enumerate(temperatures))
                continue
            pressure_hint = None
            model_rows = {}
            for index in np.argsort(temperatures)[::-1]:
                temperature = float(temperatures[index])
                epsilon_k = float(fit.epsilon(temperature))
                solved = solve_vapor_pressure(temperature, geometry.pcsaft_m, geometry.pcsaft_sigma_a,
                                              epsilon_k, pressure_hint_pa=pressure_hint)
                if solved.converged:
                    pressure_hint = solved.pressure_pa
                row = {"model_id": model_id, "input_index": int(index), "T_K": temperature,
                       "epsilon_K": epsilon_k, "P_Pa": solved.pressure_pa, "lnP_Pa": solved.ln_pressure_pa,
                       "eq_liq_reduced_density_eta": solved.eta_liq, "eq_vap_reduced_density_eta": solved.eta_vap,
                       "fugacity_residual_abs": solved.fugacity_residual, "phase_gap_eta": solved.phase_gap,
                       "converged": solved.converged, "status": solved.status,
                       "solver_method": solved.method, "message": solved.message}
                for phase, eta in (("liq", solved.eta_liq), ("vap", solved.eta_vap)):
                    rho = molar_density(eta, temperature, geometry.pcsaft_m, geometry.pcsaft_sigma_a, epsilon_k) if solved.converged else math.nan
                    row[f"rho_{phase}_mol_m3"] = rho
                    row[f"rho_{phase}_kg_m3"] = rho * mass / 1000.0
                model_rows[int(index)] = row
                if need_vaporization:
                    caloric = {"model_id": model_id, "input_index": int(index), "T_K": temperature,
                               "P_Pa": solved.pressure_pa, "status": "vle_failed", "message": solved.message}
                    if solved.converged:
                        try:
                            values = saturation_caloric_properties(temperature, solved.eta_liq, solved.eta_vap,
                                geometry.pcsaft_m, geometry.pcsaft_sigma_a, tuple(fit.coefficients),
                                float(temperatures.min()), float(temperatures.max()))
                            caloric.update({name: values[name] for name in ("Hvap_J_mol", "Hvap_kJ_mol",
                                "thermo_identity_abs_J_mol_K", "epsilon_derivative", "epsilon_region")})
                            caloric.update(status="ok", message="")
                        except Exception as exc:
                            caloric.update(status="vaporization_failed", message=str(exc))
                    caloric_rows.append(caloric)
            pvap_rows.extend(model_rows[i] for i in range(len(temperatures)))
    gchg_frame = pd.DataFrame(gchg_rows, columns=["model_id", "input_index", "T_K", "Gchg_over_RT", "canonical_nonisomeric_smiles"])
    epsilon_frame = pd.DataFrame(epsilon_rows, columns=["model_id", "status", "coefficients", "message"] if not epsilon_rows else None)
    pvap_frame = pd.DataFrame(pvap_rows, columns=["model_id", "input_index", "T_K", "P_Pa", "lnP_Pa", "converged", "status"] if not pvap_rows else None)
    summary = _ensemble_summary(pvap_frame, temperatures) if need_vle else pd.DataFrame(columns=["input_index", "T_K", "successful_model_count"])
    critical = _critical_run(geometry, embedding, selected_models, device, loaded_models) if need_critical else CriticalRun(
        pd.DataFrame(columns=CRITICAL_COLUMNS), *[pd.DataFrame() for _ in range(4)])
    caloric_frame = pd.DataFrame(caloric_rows, columns=["model_id", "input_index", "T_K", "P_Pa", "Hvap_J_mol", "Hvap_kJ_mol",
        "thermo_identity_abs_J_mol_K", "epsilon_derivative", "epsilon_region", "status", "message"])
    if not caloric_frame.empty:
        caloric_frame = caloric_frame.sort_values(["model_id", "input_index"]).reset_index(drop=True)
    metadata = {"schema_version": 2, "framework_name": FRAMEWORK_NAME,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline": "SMILES -> geometry -> ML Gchg/RT -> selected PC-SAFT properties",
        "input": {"smiles": smiles, "canonical_isomeric_smiles": canonical.geometry_smiles,
                  "canonical_nonisomeric_smiles": canonical.embedding_smiles,
                  "temperatures_k": temperatures.tolist(), "models": list(selected_models), "device": device,
                  "properties": list(requested)},
        "stages": {"vle": need_vle, "vaporization": need_vaporization, "critical": need_critical},
        "runtime": runtime_metadata(), "model_manifest": load_manifest(),
        "smi_ted": {"asset_location": "SMI_TED_HOME or the documented default user cache", "verified_sha256": smi_ted_hashes},
        "thermodynamics": {"vaporization_definition": "Difference of residual enthalpies at coexistence",
                           "derivative": "analytic fixed-number-density Helmholtz derivative including epsilon(T) chain rule",
                           "molar_mass_g_mol": mass},
        "critical": {"search_T_min_K": 11.5002, "search_T_max_K": 3728.0,
                     "closest_point_is_approximate": True, "diagnostic_plot": "critical_gchg_diagnostics.svg"},
        "experimental_data_used_at_inference": False}
    return PredictionResult(geometry, gchg_frame, epsilon_frame, pvap_frame, summary, metadata,
        caloric_frame, property_summary(caloric_frame, ["input_index", "T_K"], ["Hvap_J_mol", "Hvap_kJ_mol"]),
        critical.properties, property_summary(critical.properties, [], CRITICAL_VALUES),
        critical.curves, critical.candidates, critical.vle_validation, critical.branch_choices)


def predict_range(smiles: str, t_min_k: float, t_max_k: float, n_points: int = 125,
                  models: Iterable[str] = DEFAULT_MODELS, device: str = "cpu",
                  properties: Iterable[str] | None = None) -> PredictionResult:
    if int(n_points) != n_points or int(n_points) < 5:
        raise ValueError("n_points must be an integer of at least 5.")
    if not (math.isfinite(float(t_min_k)) and math.isfinite(float(t_max_k))):
        raise ValueError("Temperature bounds must be finite.")
    if not (float(t_min_k) > 0.0 and float(t_max_k) > float(t_min_k)):
        raise ValueError("Require 0 < t_min_k < t_max_k.")
    temperatures = np.linspace(float(t_min_k), float(t_max_k), int(n_points), dtype=float)
    return predict(smiles, temperatures, models=models, device=device, properties=properties)
