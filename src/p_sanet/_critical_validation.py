"""Portable selection and VLE gates extracted from the monotonic-root benchmark."""
from __future__ import annotations
import itertools
import json
import math
from typing import Any, Sequence
import numpy as np
import pandas as pd

# Public ensemble indices, not training initialization identifiers.
SEEDS = (1, 2, 3, 4, 5)
SMILES_COL = "canonical_smiles_nonisomeric"
FUGACITY_TOLERANCE = 1.0e-5
STRONG_R2_MIN = 0.95
STRONG_TC_EXTRAPOLATION_ERROR = 0.02
STRONG_PC_EXTRAPOLATION_ERROR = 0.05
SUPPORTIVE_R2_MIN = 0.85
SUPPORTIVE_TC_EXTRAPOLATION_ERROR = 0.05
SUPPORTIVE_PC_EXTRAPOLATION_ERROR = 0.10


def linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    denominator = float(np.sum((y - np.mean(y)) ** 2))
    numerator = float(np.sum((y - fitted) ** 2))
    r_squared = math.nan if denominator <= 0.0 else 1.0 - numerator / denominator
    return float(slope), float(intercept), float(r_squared)


def summarize_candidate(group: pd.DataFrame) -> dict[str, Any]:
    group = group.sort_values("temperature_ratio")
    first = group.iloc[0]
    epsilon_all = bool(group["epsilon_status"].eq("ok").all())
    q_values = group["q_over_q_critical"].to_numpy(dtype=float)
    subcritical_all = bool(epsilon_all and np.all(q_values > 1.0))
    vle_all = bool(group["vle_status"].eq("ok").all())
    result: dict[str, Any] = {
        "candidate_id": first["candidate_id"],
        "seed": int(first["seed"]),
        SMILES_COL: first[SMILES_COL],
        "root_index": int(first["root_index"]),
        "temperature_root_K": float(first["temperature_root_K"]),
        "stability_class": first["stability_class"],
        "fixed_point_selected": bool(first["fixed_point_selected"]),
        "epsilon_all_five_unique": epsilon_all,
        "subcritical_side_all_five": subcritical_all,
        "vle_all_five_converged": vle_all,
        "epsilon_failure_count": int((~group["epsilon_status"].eq("ok")).sum()),
        "vle_failure_count": int((~group["vle_status"].eq("ok")).sum()),
        "q_over_qc_min": float(np.nanmin(q_values)) if np.isfinite(q_values).any() else math.nan,
        "q_over_qc_at_0p99": float(group.iloc[-1]["q_over_q_critical"]),
        "candidate_pc_kPa": float(first["candidate_pc_kPa"]),
        "candidate_rho_c_mol_m3": float(first["candidate_rho_c_mol_m3"]),
        "candidate_z_c": float(first["candidate_z_c"]),
        "gap_monotone_decreasing": False,
        "psat_monotone_increasing": False,
        "eta_c_bracketed_near_critical": False,
        "gap_squared_fit_slope": math.nan,
        "gap_squared_fit_intercept": math.nan,
        "gap_squared_fit_R2": math.nan,
        "density_extrapolated_zero_T_over_root_T": math.nan,
        "density_Tc_extrapolation_relative_error": math.nan,
        "psat_fit_slope": math.nan,
        "psat_fit_intercept": math.nan,
        "psat_fit_R2": math.nan,
        "psat_extrapolated_at_root_over_pc": math.nan,
        "psat_extrapolation_relative_error": math.nan,
        "max_fugacity_residual": math.nan,
        "validation_class": "",
    }
    if not epsilon_all:
        result["validation_class"] = "inconclusive_epsilon_or_model_domain"
        return result
    if not subcritical_all:
        result["validation_class"] = "inconsistent_subcritical_direction"
        return result
    if not vle_all:
        result["validation_class"] = "inconclusive_vle_solver"
        return result

    gaps = group["delta_rho_over_candidate_rho_c"].to_numpy(dtype=float)
    pressures = group["psat_over_candidate_pc"].to_numpy(dtype=float)
    ratios = group["temperature_ratio"].to_numpy(dtype=float)
    eta_liq = group["eta_liq"].to_numpy(dtype=float)
    eta_vap = group["eta_vap"].to_numpy(dtype=float)
    eta_c = float(first["eta_c"])
    monotone_tolerance_gap = 1.0e-8 * max(1.0, float(np.nanmax(np.abs(gaps))))
    monotone_tolerance_pressure = 1.0e-8 * max(1.0, float(np.nanmax(np.abs(pressures))))
    gap_monotone = bool(np.all(np.diff(gaps) <= monotone_tolerance_gap))
    pressure_monotone = bool(np.all(np.diff(pressures) >= -monotone_tolerance_pressure))
    bracketed = bool(np.all((eta_vap[-3:] < eta_c) & (eta_c < eta_liq[-3:])))
    gap_slope, gap_intercept, gap_r2 = linear_fit(ratios[-3:], gaps[-3:] ** 2)
    zero_ratio = -gap_intercept / gap_slope if gap_slope != 0.0 else math.nan
    tc_error = abs(zero_ratio - 1.0) if np.isfinite(zero_ratio) else math.inf
    pressure_slope, pressure_intercept, pressure_r2 = linear_fit(ratios[-3:], pressures[-3:])
    pressure_at_root = pressure_slope + pressure_intercept
    pressure_error = abs(pressure_at_root - 1.0)
    result.update(
        gap_monotone_decreasing=gap_monotone,
        psat_monotone_increasing=pressure_monotone,
        eta_c_bracketed_near_critical=bracketed,
        gap_squared_fit_slope=gap_slope,
        gap_squared_fit_intercept=gap_intercept,
        gap_squared_fit_R2=gap_r2,
        density_extrapolated_zero_T_over_root_T=zero_ratio,
        density_Tc_extrapolation_relative_error=tc_error,
        psat_fit_slope=pressure_slope,
        psat_fit_intercept=pressure_intercept,
        psat_fit_R2=pressure_r2,
        psat_extrapolated_at_root_over_pc=pressure_at_root,
        psat_extrapolation_relative_error=pressure_error,
        max_fugacity_residual=float(group["fugacity_residual"].max()),
    )
    common = (
        gap_monotone
        and pressure_monotone
        and bracketed
        and gap_slope < 0.0
        and pressure_slope > 0.0
        and float(group["fugacity_residual"].max()) <= FUGACITY_TOLERANCE
    )
    strong = (
        common
        and gap_r2 >= STRONG_R2_MIN
        and pressure_r2 >= STRONG_R2_MIN
        and tc_error <= STRONG_TC_EXTRAPOLATION_ERROR
        and pressure_error <= STRONG_PC_EXTRAPOLATION_ERROR
    )
    supportive = (
        common
        and gap_r2 >= SUPPORTIVE_R2_MIN
        and pressure_r2 >= SUPPORTIVE_R2_MIN
        and tc_error <= SUPPORTIVE_TC_EXTRAPOLATION_ERROR
        and pressure_error <= SUPPORTIVE_PC_EXTRAPOLATION_ERROR
    )
    if strong:
        result["validation_class"] = "vle_validated_strong"
    elif supportive:
        result["validation_class"] = "vle_supportive"
    elif common:
        # A five-point grid ending at 0.99 Tc may be too coarse for very long
        # chains, whose Psat curve remains strongly nonlinear there.  When all
        # direct physical checks pass, failure of the deliberately tight
        # extrapolation gate is lack of resolution, not contrary evidence.
        result["validation_class"] = "vle_termination_grid_unresolved"
    else:
        result["validation_class"] = "inconsistent_vle_termination"
    return result


def dispersion_scores(values: Sequence[float]) -> tuple[float, float]:
    logs = np.log(np.asarray(values, dtype=float))
    if len(logs) <= 1:
        return 0.0, 0.0
    full = float(np.std(logs, ddof=1))
    if len(logs) >= 4:
        trimmed = min(
            float(np.std(logs[list(indices)], ddof=1))
            for indices in itertools.combinations(range(len(logs)), len(logs) - 1)
        )
    else:
        trimmed = full
    return trimmed, full


def select_cross_seed_branches(
    candidates: pd.DataFrame,
    selections: pd.DataFrame,
    label: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = candidates.copy()
    candidates["branch_selected"] = False
    candidate_groups = {
        (int(seed), str(smiles)): group.copy()
        for (seed, smiles), group in candidates.groupby(["seed", SMILES_COL], sort=False)
    }
    selection_rows: list[dict[str, Any]] = []
    choice_rows: list[dict[str, Any]] = []
    for smiles, seed_base in selections.groupby(SMILES_COL, sort=True):
        options: list[list[dict[str, Any]]] = []
        option_seeds: list[int] = []
        for seed in SEEDS:
            group = candidate_groups.get((seed, str(smiles)))
            if group is None or group.empty:
                continue
            options.append(group.to_dict(orient="records"))
            option_seeds.append(seed)
        combinations: list[dict[str, Any]] = []
        for combination in itertools.product(*options) if options else []:
            temperatures = [float(item["temperature_root_K"]) for item in combination]
            trimmed, full = dispersion_scores(temperatures)
            direct_count = sum(item["candidate_source"] == "direct_monotonic_root" for item in combination)
            combinations.append({
                "items": combination,
                "seed_count": len(combination),
                "direct_count": direct_count,
                "trimmed_logTc_sd": trimmed,
                "full_logTc_sd": full,
                "candidate_ids": tuple(str(item["candidate_id"]) for item in combination),
            })
        combinations.sort(key=lambda item: (
            -item["seed_count"], -item["direct_count"],
            item["trimmed_logTc_sd"], item["full_logTc_sd"], item["candidate_ids"],
        ))
        resolved = False
        reason = "no_eligible_candidate"
        best: dict[str, Any] | None = combinations[0] if combinations else None
        if best is not None:
            peers = [
                item for item in combinations[1:]
                if item["seed_count"] == best["seed_count"] and item["direct_count"] == best["direct_count"]
            ]
            if not peers:
                resolved = True
                reason = "unique_branch_combination"
            else:
                second = peers[0]
                trimmed_clear = best["trimmed_logTc_sd"] <= 0.90 * second["trimmed_logTc_sd"]
                trimmed_tied = math.isclose(
                    best["trimmed_logTc_sd"], second["trimmed_logTc_sd"], rel_tol=1.0e-9, abs_tol=1.0e-12
                )
                full_clear = best["full_logTc_sd"] <= 0.90 * second["full_logTc_sd"]
                resolved = bool(trimmed_clear or (trimmed_tied and full_clear))
                reason = "cross_seed_dispersion_clear" if resolved else "cross_seed_assignment_unresolved"
        if resolved and best is not None:
            selected_ids = set(best["candidate_ids"])
            candidates.loc[candidates["candidate_id"].isin(selected_ids), "branch_selected"] = True
        else:
            selected_ids = set()
        for rank, combination in enumerate(combinations):
            choice_rows.append({
                SMILES_COL: smiles,
                "selection_scope": label,
                "combination_rank": rank + 1,
                "seed_count": combination["seed_count"],
                "direct_count": combination["direct_count"],
                "trimmed_logTc_sd": combination["trimmed_logTc_sd"],
                "full_logTc_sd": combination["full_logTc_sd"],
                "candidate_ids_json": json.dumps(combination["candidate_ids"]),
                "selected_combination": resolved and rank == 0,
            })
        for seed in SEEDS:
            group = candidate_groups.get((seed, str(smiles)))
            chosen = None
            if group is not None and selected_ids:
                hit = group.loc[group["candidate_id"].isin(selected_ids)]
                if len(hit) == 1:
                    chosen = hit.iloc[0]
            selection_rows.append({
                "seed": seed,
                SMILES_COL: smiles,
                "selection_scope": label,
                "status": "ok" if chosen is not None else reason,
                "selection_reason": reason,
                "eligible_candidate_count": 0 if group is None else len(group),
                "selected_candidate_id": None if chosen is None else chosen["candidate_id"],
                "selected_candidate_source": None if chosen is None else chosen["candidate_source"],
                "selected_root_index": math.nan if chosen is None else chosen["root_index"],
                "TC_pred_K": math.nan if chosen is None else chosen["temperature_root_K"],
                "validation_class": None if chosen is None else chosen["validation_class"],
                "best_trimmed_logTc_sd": math.nan if best is None else best["trimmed_logTc_sd"],
                "best_full_logTc_sd": math.nan if best is None else best["full_logTc_sd"],
            })
    return pd.DataFrame(selection_rows), pd.DataFrame(choice_rows)
