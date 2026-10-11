"""Pure-component, non-associating PC-SAFT critical-property solver.

This module contains only equations and numerical solvers.  It is extracted
from the research implementation without workbook readers, experimental
critical properties, or absolute local paths.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict, field

import numpy as np
from scipy.optimize import least_squares, brentq, minimize_scalar
from typing import Callable, Sequence

import pandas as pd

from .pcsaft import gchg_over_rt
from .thermodynamics import molar_density, R_GAS
from ._critical_validation import select_cross_seed_branches, summarize_candidate

DEFAULT_DENSE = 0.4
DEFAULT_EPSILON_MIN_K = 1.0e-8
DEFAULT_EPSILON_MAX_K = 5000.0
DEFAULT_EPSILON_BRACKET_POINTS = 65
TEMPERATURE_MIN_K = 11.5002
TEMPERATURE_MAX_K = 3728.0
SCAN_POINTS = 513
FD_RELATIVE_STEPS = (0.001, 0.003, 0.01)
CRITICAL_ACCEPTED = {"ok", "closest_point_no_intersection"}
ModelPredictor = Callable[[np.ndarray], np.ndarray]



A_COEFF = np.array(
    [
        [0.91056314451539, -0.30840169182720, -0.09061483509767],
        [0.63612814494991, 0.18605311591713, 0.45278428063920],
        [2.68613478913903, -2.50300472586548, 0.59627007280101],
        [-26.5473624914884, 21.4197936296668, -1.72418291311787],
        [97.7592087835073, -65.2558853303492, -4.13021125311661],
        [-159.591540865600, 83.3186804808856, 13.7766318697211],
        [91.2977740839123, -33.7469229297323, -8.67284703679646],
    ],
    dtype=float,
)
B_COEFF = np.array(
    [
        [0.72409469413165, -0.57554980753450, 0.09768831158356],
        [1.11913959304690 * 2.0, 0.34975477607218 * 2.0, -0.12787874908050 * 2.0],
        [-1.33419498282114 * 3.0, 1.29752244631769 * 3.0, -3.05195205099107 * 3.0],
        [-5.25089420371162 * 4.0, -4.30386791194303 * 4.0, 5.16051899359931 * 4.0],
        [5.37112827253230 * 5.0, 38.5344528930499 * 5.0, -7.76088601041257 * 5.0],
        [34.4252230677698 * 6.0, -26.9710769414608 * 6.0, 15.6044623461691 * 6.0],
        [-50.8003365888685 * 7.0, -23.6010990650801 * 7.0, -4.23812936930675 * 7.0],
    ],
    dtype=float,
)


@dataclass(frozen=True)
class CriticalSolution:
    m: float
    t_tilde_c: float
    eta_c: float
    residual_d1: float
    residual_d2: float
    root_count: int




def _chain_coefficients(m: float) -> tuple[np.ndarray, np.ndarray]:
    w1 = 1.0 - 1.0 / m
    w2 = w1 * (1.0 - 2.0 / m)
    return (
        A_COEFF[:, 0] + w1 * A_COEFF[:, 1] + w2 * A_COEFF[:, 2],
        B_COEFF[:, 0] + w1 * B_COEFF[:, 1] + w2 * B_COEFF[:, 2],
    )


def _z_and_dz_deta(
    eta: complex | float,
    temperature: float,
    m: float,
    epsilon_over_k: float,
) -> tuple[complex | float, complex | float, float]:
    if temperature <= 0.0 or m <= 0.0 or epsilon_over_k <= 0.0:
        raise ValueError("temperature, m, and epsilon_over_k must be positive")
    a, b = _chain_coefficients(m)
    q = epsilon_over_k / temperature
    d_ratio = 1.0 - 0.12 * math.exp(-3.0 * q)
    z3_star = (math.pi / 6.0) * m * d_ratio**3
    rho_sigma3 = eta / z3_star
    one_minus_eta = 1.0 - eta

    g = (
        1.0 / one_minus_eta
        + 1.5 * eta / one_minus_eta**2
        + 0.5 * eta**2 / one_minus_eta**3
    )
    dg = (
        1.0 / one_minus_eta**2
        + 1.5 * (1.0 + eta) / one_minus_eta**3
        + eta * (1.0 + 0.5 * eta) / one_minus_eta**4
    )
    d2g = (
        2.0 / one_minus_eta**3
        + 3.0 * (2.0 + eta) / one_minus_eta**4
        + (1.0 + 4.0 * eta + eta**2) / one_minus_eta**5
    )
    z_hs = m * (
        eta / one_minus_eta
        + 3.0 * eta / one_minus_eta**2
        + eta**2 * (3.0 - eta) / one_minus_eta**3
    )
    dz_hs = m * (
        1.0 / one_minus_eta**2
        + 3.0 * (1.0 + eta) / one_minus_eta**3
        + 6.0 * eta / one_minus_eta**4
    )
    z_hc = (1.0 - m) * eta * dg / g
    dz_hc = (1.0 - m) * (-eta * (dg / g) ** 2 + dg / g + eta * d2g / g)

    powers = np.array([eta**i for i in range(7)])
    i2 = np.dot(b, powers)
    eta_i1_prime = np.dot(a * np.arange(1, 8), powers)
    eta_i2_prime = np.dot(b * np.arange(1, 8), powers)
    eta_i1_second = sum(a[i] * (i + 1) * i * eta ** (i - 1) for i in range(1, 7))
    eta_i2_second = sum(b[i] * (i + 1) * i * eta ** (i - 1) for i in range(1, 7))

    c1 = 1.0 / (
        1.0
        + m * (8.0 * eta - 2.0 * eta**2) / one_minus_eta**4
        + (1.0 - m)
        * (20.0 * eta - 27.0 * eta**2 + 12.0 * eta**3 - 2.0 * eta**4)
        / (one_minus_eta * (2.0 - eta)) ** 2
    )
    c2 = -(c1**2) * (
        m * (-4.0 * eta**2 + 20.0 * eta + 8.0) / one_minus_eta**5
        + (1.0 - m)
        * (2.0 * eta**3 + 12.0 * eta**2 - 48.0 * eta + 40.0)
        / (one_minus_eta * (2.0 - eta)) ** 3
    )
    c3 = 2.0 * c2**2 / c1 - c1**2 * (
        m * (-12.0 * eta**2 + 72.0 * eta + 60.0) / one_minus_eta**6
        + (1.0 - m)
        * (-6.0 * eta**4 - 48.0 * eta**3 + 288.0 * eta**2 - 480.0 * eta + 264.0)
        / (one_minus_eta * (2.0 - eta)) ** 4
    )
    order1_scaled = m**2 * q
    order2_scaled = m**2 * q**2
    z_disp = (
        -2.0 * math.pi * rho_sigma3 * eta_i1_prime * order1_scaled
        - math.pi * rho_sigma3 * order2_scaled * m * (c2 * i2 * eta + c1 * eta_i2_prime)
    )
    dz_disp = (
        z_disp / eta
        - 2.0 * math.pi * rho_sigma3 * eta_i1_second * order1_scaled
        - math.pi
        * rho_sigma3
        * order2_scaled
        * m
        * (c3 * i2 * eta + 2.0 * c2 * eta_i2_prime + c1 * eta_i2_second)
    )
    return 1.0 + z_hs + z_hc + z_disp, dz_hs + dz_hc + dz_disp, d_ratio


def reduced_pressure(
    eta: complex | float, temperature: float, m: float, epsilon_over_k: float
) -> complex | float:
    z, _, d_ratio = _z_and_dz_deta(eta, temperature, m, epsilon_over_k)
    return z * eta / ((math.pi / 6.0) * m * d_ratio**3)


def dpressure_deta(
    eta: complex | float, temperature: float, m: float, epsilon_over_k: float
) -> complex | float:
    z, dz, d_ratio = _z_and_dz_deta(eta, temperature, m, epsilon_over_k)
    return (z + eta * dz) / ((math.pi / 6.0) * m * d_ratio**3)


def d2pressure_deta2(
    eta: float,
    temperature: float,
    m: float,
    epsilon_over_k: float,
    complex_step: float = 1.0e-20,
) -> float:
    value = dpressure_deta(complex(eta, complex_step), temperature, m, epsilon_over_k)
    return float(np.imag(value) / complex_step)


def _critical_residual(values: np.ndarray, m: float) -> np.ndarray:
    t_tilde, eta = values
    try:
        d1 = float(np.real(dpressure_deta(eta, t_tilde, m, 1.0)))
        d2 = d2pressure_deta2(eta, t_tilde, m, 1.0)
    except (FloatingPointError, OverflowError, ValueError, ZeroDivisionError):
        return np.array([1.0e6, 1.0e6])
    if not (np.isfinite(d1) and np.isfinite(d2)):
        return np.array([1.0e6, 1.0e6])
    return np.array([d1, d2])


def solve_reduced_critical_point(m: float) -> CriticalSolution:
    """Return the highest-temperature physically admissible critical root."""
    if not np.isfinite(m) or m <= 0.0:
        raise ValueError(f"Invalid segment number m={m!r}")
    t_guesses = np.array([1.1, 1.3, 1.6, 2.0, 2.5, 3.0, 3.6, 4.3, 5.0, 5.7])
    eta_guesses = np.array([0.005, 0.012, 0.025, 0.05, 0.09, 0.14, 0.21, 0.30])
    roots: list[tuple[float, float, float]] = []
    for t0 in t_guesses:
        for eta0 in eta_guesses:
            result = least_squares(
                _critical_residual,
                x0=np.array([t0, eta0]),
                args=(m,),
                bounds=(np.array([0.45, 1.0e-4]), np.array([7.0, 0.55])),
                xtol=1.0e-13,
                ftol=1.0e-13,
                gtol=1.0e-13,
                max_nfev=1200,
                x_scale=np.array([2.0, 0.1]),
            )
            residual = _critical_residual(result.x, m)
            norm = float(np.linalg.norm(residual, ord=np.inf))
            if result.success and norm < 5.0e-8:
                root = (float(result.x[0]), float(result.x[1]), norm)
                if not any(
                    abs(root[0] - old[0]) < 2.0e-7 and abs(root[1] - old[1]) < 2.0e-7
                    for old in roots
                ):
                    roots.append(root)
    if not roots:
        raise RuntimeError(f"No critical root converged for m={m:.12g}")
    ordinary = max(roots, key=lambda item: item[0])
    residual = _critical_residual(np.array(ordinary[:2]), m)
    return CriticalSolution(
        m=float(m),
        t_tilde_c=ordinary[0],
        eta_c=ordinary[1],
        residual_d1=float(residual[0]),
        residual_d2=float(residual[1]),
        root_count=len(roots),
    )



@dataclass(frozen=True)
class EpsilonInversionResult:
    status: str
    converged: bool
    message: str
    temperature_k: float
    target_gchg_rt: float
    epsilon_k: float = math.nan
    reconstructed_gchg_rt: float = math.nan
    residual_gchg_rt: float = math.nan
    root_count: int = 0
    roots_k: tuple[float, ...] = ()
    brackets_k: tuple[tuple[float, float], ...] = ()
    diagnostics: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _deduplicate(values: list[float], relative_tolerance: float = 1.0e-8) -> list[float]:
    unique: list[float] = []
    for value in sorted(values):
        if not unique or abs(value - unique[-1]) > relative_tolerance * max(1.0, abs(value)):
            unique.append(value)
    return unique


def invert_epsilon_at_temperature(
    temperature_k: float,
    target_gchg_rt: float,
    m: float,
    sigma_a: float,
    *,
    dense: float = DEFAULT_DENSE,
    epsilon_min_k: float = DEFAULT_EPSILON_MIN_K,
    epsilon_max_k: float = DEFAULT_EPSILON_MAX_K,
    bracket_points: int = DEFAULT_EPSILON_BRACKET_POINTS,
) -> EpsilonInversionResult:
    """Invert one ML Gchg/RT value to epsilon with scan-plus-Brent."""
    inputs = [temperature_k, target_gchg_rt, m, sigma_a, dense, epsilon_min_k, epsilon_max_k]
    if (
        not all(np.isfinite(inputs))
        or temperature_k <= 0.0
        or m <= 0.0
        or sigma_a <= 0.0
        or not 0.0 < dense < 1.0
        or epsilon_min_k <= 0.0
        or epsilon_max_k <= epsilon_min_k
        or int(bracket_points) != bracket_points
        or int(bracket_points) < 3
    ):
        return EpsilonInversionResult(
            status="invalid_input",
            converged=False,
            message="Invalid epsilon-inversion input or bounds.",
            temperature_k=float(temperature_k),
            target_gchg_rt=float(target_gchg_rt),
        )

    epsilon_grid = np.geomspace(epsilon_min_k, epsilon_max_k, int(bracket_points))

    def residual(epsilon: float) -> float:
        return gchg_over_rt(temperature_k, m, sigma_a, epsilon, dense) - target_gchg_rt

    try:
        residuals = np.asarray([residual(float(value)) for value in epsilon_grid], dtype=float)
    except Exception as exc:
        return EpsilonInversionResult(
            status="epsilon_scan_failed",
            converged=False,
            message=str(exc),
            temperature_k=float(temperature_k),
            target_gchg_rt=float(target_gchg_rt),
            diagnostics={"bracket_points": int(bracket_points)},
        )
    finite = np.isfinite(residuals)
    if not finite.all():
        return EpsilonInversionResult(
            status="epsilon_scan_failed",
            converged=False,
            message="Non-finite residual encountered during epsilon bracket scan.",
            temperature_k=float(temperature_k),
            target_gchg_rt=float(target_gchg_rt),
            diagnostics={"finite_scan_count": int(finite.sum()), "bracket_points": int(bracket_points)},
        )

    exact_tolerance = 1.0e-10 * max(1.0, abs(target_gchg_rt))
    candidates = [
        float(epsilon_grid[index])
        for index, value in enumerate(residuals)
        if abs(float(value)) <= exact_tolerance
    ]
    brackets: list[tuple[float, float]] = []
    brent_failures: list[str] = []
    for index in range(len(epsilon_grid) - 1):
        left_value = float(residuals[index])
        right_value = float(residuals[index + 1])
        if left_value * right_value < 0.0:
            left = float(epsilon_grid[index])
            right = float(epsilon_grid[index + 1])
            brackets.append((left, right))
            try:
                candidates.append(
                    float(brentq(residual, left, right, xtol=1.0e-8, rtol=1.0e-12, maxiter=200))
                )
            except Exception as exc:
                brent_failures.append(str(exc))
    roots = _deduplicate(candidates)
    diagnostics = {
        "bracket_points": int(bracket_points),
        "epsilon_min_k": float(epsilon_min_k),
        "epsilon_max_k": float(epsilon_max_k),
        "scan_residual_min": float(np.min(residuals)),
        "scan_residual_max": float(np.max(residuals)),
        "brent_failure_count": len(brent_failures),
        "brent_failure_messages": tuple(brent_failures),
    }
    if not roots:
        return EpsilonInversionResult(
            status="no_epsilon_root",
            converged=False,
            message="No epsilon root was bracketed inside the accepted epsilon interval.",
            temperature_k=float(temperature_k),
            target_gchg_rt=float(target_gchg_rt),
            brackets_k=tuple(brackets),
            diagnostics=diagnostics,
        )
    if len(roots) > 1:
        return EpsilonInversionResult(
            status="multiple_epsilon_roots",
            converged=False,
            message="Multiple epsilon roots were found; inversion is ambiguous.",
            temperature_k=float(temperature_k),
            target_gchg_rt=float(target_gchg_rt),
            root_count=len(roots),
            roots_k=tuple(roots),
            brackets_k=tuple(brackets),
            diagnostics=diagnostics,
        )
    epsilon = roots[0]
    reconstructed = gchg_over_rt(temperature_k, m, sigma_a, epsilon, dense)
    final_residual = reconstructed - target_gchg_rt
    if not math.isfinite(final_residual) or abs(final_residual) > 1.0e-8 * max(1.0, abs(target_gchg_rt)):
        return EpsilonInversionResult(
            status="invalid_epsilon_root",
            converged=False,
            message="The bracketed epsilon root failed residual validation.",
            temperature_k=float(temperature_k),
            target_gchg_rt=float(target_gchg_rt),
            epsilon_k=float(epsilon),
            reconstructed_gchg_rt=float(reconstructed),
            residual_gchg_rt=float(final_residual),
            root_count=1,
            roots_k=(float(epsilon),),
            brackets_k=tuple(brackets),
            diagnostics=diagnostics,
        )
    return EpsilonInversionResult(
        status="ok",
        converged=True,
        message="",
        temperature_k=float(temperature_k),
        target_gchg_rt=float(target_gchg_rt),
        epsilon_k=float(epsilon),
        reconstructed_gchg_rt=float(reconstructed),
        residual_gchg_rt=float(final_residual),
        root_count=1,
        roots_k=(float(epsilon),),
        brackets_k=tuple(brackets),
        diagnostics=diagnostics,
    )


@dataclass
class CriticalRun:
    """Selected states and auditable curves/candidates/continuation/branch choices."""

    properties: pd.DataFrame
    curves: pd.DataFrame
    candidates: pd.DataFrame
    vle_validation: pd.DataFrame
    branch_choices: pd.DataFrame

    def __post_init__(self) -> None:
        schemas = {
            "properties": ["model_id", "Tc_K", "status", "message", "is_approximate", "intersection_found"],
            "curves": ["model_id", "T_K", "Gchg_over_RT", "dgchg_dT", "critical_target_gchg_rt", "signed_gap_gchg_rt"],
            "candidates": ["model_id", "root_index", "temperature_root_K", "validation_class", "direct_monotonic_candidate"],
            "vle_validation": ["model_id", "root_index", "temperature_K", "temperature_ratio", "epsilon_status", "vle_status"],
            "branch_choices": ["selection_scope", "combination_rank", "selected_combination"],
        }
        for name, columns in schemas.items():
            if len(getattr(self, name).columns) == 0:
                setattr(self, name, pd.DataFrame(columns=columns))


def _values(predictor: ModelPredictor, temperatures: np.ndarray) -> np.ndarray:
    values = np.asarray(predictor(np.asarray(temperatures, dtype=float)), dtype=float)
    if values.shape != temperatures.shape or not np.all(np.isfinite(values)):
        raise FloatingPointError("Critical predictor must return finite values with the input shape.")
    return values


def critical_physical_state(
    reduced: CriticalSolution, tc_k: float, sigma_a: float, molar_mass_g_mol: float,
) -> dict[str, float]:
    """Convert the reduced critical construction; epsilon_required = Tc/T_tilde_c."""
    if not all(math.isfinite(v) and v > 0 for v in (tc_k, sigma_a, molar_mass_g_mol)):
        raise ValueError("Tc, sigma and molar mass must be positive and finite.")
    epsilon = tc_k / reduced.t_tilde_c
    z, _, diameter_ratio = _z_and_dz_deta(reduced.eta_c, tc_k, reduced.m, epsilon)
    rho = molar_density(reduced.eta_c, tc_k, reduced.m, sigma_a, epsilon)
    pressure = float(z) * rho * R_GAS * tc_k
    if not math.isfinite(pressure) or pressure <= 0:
        raise FloatingPointError("Invalid constructed critical pressure.")
    return {
        "Tc_K": tc_k, "epsilon_required_at_Tc_K": epsilon,
        "Pc_Pa": pressure, "Pc_kPa": pressure / 1000.0,
        "rho_c_mol_m3": rho, "rho_c_kg_m3": rho * molar_mass_g_mol / 1000.0,
        "Zc": float(z), "critical_molar_volume_m3_mol": 1.0 / rho,
        "segment_diameter_A": sigma_a * diameter_ratio,
    }


def closest_temperature(
    predictor: ModelPredictor, target: float, grid: np.ndarray, values: np.ndarray,
) -> tuple[float, float]:
    """Refine every sampled |gap| minimum in log T, including both boundaries."""
    gaps = np.abs(values - target)
    choices = [(float(grid[i]), float(values[i])) for i in (0, len(grid) - 1, int(np.argmin(gaps)))]
    for i in range(1, len(grid) - 1):
        if gaps[i] <= gaps[i - 1] and gaps[i] <= gaps[i + 1]:
            # Flat regions need only their lower endpoint, which is already in choices.
            if gaps[i - 1] == gaps[i] == gaps[i + 1]:
                continue
            result = minimize_scalar(
                lambda log_t: abs(float(_values(predictor, np.asarray([math.exp(log_t)]))[0]) - target),
                bounds=(math.log(grid[i - 1]), math.log(grid[i + 1])), method="bounded",
                options={"xatol": 1.0e-12},
            )
            if not result.success or not math.isfinite(result.fun):
                raise RuntimeError("Closest-point minimization failed.")
            temperature = math.exp(result.x)
            choices.extend([(float(grid[i]), float(values[i])),
                            (temperature, float(_values(predictor, np.asarray([temperature]))[0]))])
    best_gap = min(abs(value - target) for _, value in choices)
    # Resolve exact/effectively exact ties deterministically, without changing the objective.
    tied = [(t, v) for t, v in choices if math.isclose(abs(v - target), best_gap, rel_tol=1e-12, abs_tol=1e-12)]
    return min(tied, key=lambda item: item[0])


def _validate_vle(
    candidate: dict[str, object], predictor: ModelPredictor, reduced: CriticalSolution,
    sigma_a: float,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    from .pcsaft import make_eos
    from .solver import _find_isotherm_branches, _solve_bracketed

    tc = float(candidate["temperature_root_K"])
    physical = critical_physical_state(reduced, tc, sigma_a, 1.0)
    rows = []
    hint = None
    for ratio in (0.80, 0.85, 0.90, 0.95, 0.99):
        temperature = ratio * tc
        row = {
            **candidate, "temperature_ratio": ratio, "temperature_K": temperature,
            "eta_c": reduced.eta_c, "candidate_pc_kPa": physical["Pc_kPa"],
            "candidate_rho_c_mol_m3": physical["rho_c_mol_m3"], "candidate_z_c": physical["Zc"],
            "epsilon_status": "temperature_outside_model_domain", "epsilon_K": math.nan,
            "q_over_q_critical": math.nan, "vle_status": "not_attempted",
            "eta_liq": math.nan, "eta_vap": math.nan, "psat_pa": math.nan,
            "psat_over_candidate_pc": math.nan, "delta_rho_over_candidate_rho_c": math.nan,
            "fugacity_residual": math.nan, "message": "",
        }
        if TEMPERATURE_MIN_K <= temperature <= TEMPERATURE_MAX_K:
            g = float(_values(predictor, np.asarray([temperature]))[0])
            inversion = invert_epsilon_at_temperature(temperature, g, reduced.m, sigma_a)
            row.update(epsilon_status=inversion.status, epsilon_K=inversion.epsilon_k,
                       epsilon_root_count=inversion.root_count, epsilon_message=inversion.message)
            if inversion.converged:
                epsilon = inversion.epsilon_k
                row["q_over_q_critical"] = epsilon / temperature * reduced.t_tilde_c
                try:
                    eos = make_eos(temperature, reduced.m, sigma_a, epsilon)
                    result = _solve_bracketed(_find_isotherm_branches(eos), hint)
                    if not result.converged:
                        raise RuntimeError(f"{result.status}: {result.message}")
                    hint = result.pressure_pa
                    rho_l = molar_density(result.eta_liq, temperature, reduced.m, sigma_a, epsilon)
                    rho_v = molar_density(result.eta_vap, temperature, reduced.m, sigma_a, epsilon)
                    row.update(vle_status="ok", eta_liq=result.eta_liq, eta_vap=result.eta_vap,
                               psat_pa=result.pressure_pa, psat_over_candidate_pc=result.pressure_pa / physical["Pc_Pa"],
                               delta_rho_over_candidate_rho_c=(rho_l - rho_v) / physical["rho_c_mol_m3"],
                               fugacity_residual=result.fugacity_residual)
                except Exception as exc:
                    row.update(vle_status="vle_solver_failed", message=str(exc))
        rows.append(row)
    return summarize_candidate(pd.DataFrame(rows)), rows


def solve_critical_properties(
    m: float, sigma_a: float, molar_mass_g_mol: float,
    predictors: dict[str, ModelPredictor], derivatives: dict[str, ModelPredictor],
) -> CriticalRun:
    """Full monotonic ML-root/VLE/ensemble algorithm, with explicit no-root fallback.

    Both callback dictionaries use public model IDs. Real networks must supply
    their autograd derivative, rather than a fitted epsilon(T) derivative.
    """
    grid = np.geomspace(TEMPERATURE_MIN_K, TEMPERATURE_MAX_K, SCAN_POINTS)
    rows, curve_rows, candidate_rows, continuation_rows = [], [], [], []
    try:
        if not predictors or set(predictors) != set(derivatives):
            raise ValueError("Predictor and derivative model IDs must match and be nonempty.")
        reduced = solve_reduced_critical_point(m)
        target = gchg_over_rt(1.0, m, sigma_a, 1.0 / reduced.t_tilde_c, DEFAULT_DENSE)
    except Exception as exc:
        return CriticalRun(pd.DataFrame([
            {"model_id": key, "Tc_K": math.nan, "status": "reduced_critical_failed", "message": str(exc),
             "is_approximate": False, "intersection_found": False}
            for key in predictors
        ]), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    for model_id, predictor in predictors.items():
        base = {
            "model_id": model_id, "m": m, "sigma_A": sigma_a, "molar_mass_g_mol": molar_mass_g_mol,
            "t_tilde_c": reduced.t_tilde_c, "eta_c": reduced.eta_c,
            "critical_residual_d1": reduced.residual_d1, "critical_residual_d2": reduced.residual_d2,
            "critical_root_count": reduced.root_count, "critical_target_gchg_rt": target,
            "Tc_K": math.nan, "Pc_Pa": math.nan, "Pc_kPa": math.nan,
            "rho_c_mol_m3": math.nan, "rho_c_kg_m3": math.nan, "Zc": math.nan,
            "critical_molar_volume_m3_mol": math.nan, "epsilon_required_at_Tc_K": math.nan,
            "ml_gchg_rt_at_selected_T": math.nan, "signed_gap_gchg_rt": math.nan,
            "absolute_gap_gchg_rt": math.nan, "search_T_min_K": TEMPERATURE_MIN_K,
            "search_T_max_K": TEMPERATURE_MAX_K, "is_approximate": False,
            "intersection_found": False, "temperature_root_count": 0,
            "status": "failed", "message": "", "diagnostic_plot": "critical_gchg_diagnostics.svg",
            "epsilon_definition": "Tc / T_tilde_c; required by the critical target construction",
        }
        try:
            seed = int(model_id.removeprefix("model_"))
            if model_id != f"model_{seed}" or seed not in range(1, 6):
                raise ValueError("Critical ensemble supports model_1 through model_5.")
            values = _values(predictor, grid)
            gradients = _values(derivatives[model_id], grid)
            residuals = values - target
            curve_rows.extend({"model_id": model_id, "T_K": t, "Gchg_over_RT": v,
                               "dgchg_dT": d, "critical_target_gchg_rt": target,
                               "signed_gap_gchg_rt": v - target}
                              for t, v, d in zip(grid, values, gradients))
            roots = [float(grid[i]) for i in np.flatnonzero(np.abs(residuals) <= 1e-11 * max(1.0, abs(target)))]
            brackets = []
            for i in range(len(grid) - 1):
                if residuals[i] * residuals[i + 1] < 0:
                    root = brentq(lambda t: float(_values(predictor, np.asarray([t]))[0]) - target,
                                  grid[i], grid[i + 1], xtol=1e-7, rtol=1e-12, maxiter=100)
                    roots.append(float(root))
                    brackets.append((grid[i], grid[i + 1], bool(gradients[i] > 0 and gradients[i + 1] > 0)))
            roots = _deduplicate(roots, 1e-7)
            if not roots:
                t, value = closest_temperature(predictor, target, grid, values)
                # A tangency is an intersection, but not an admissible positive-slope critical root.
                if abs(value - target) <= 1e-11 * max(1.0, abs(target)):
                    roots = [t]
                else:
                    base.update(critical_physical_state(reduced, t, sigma_a, molar_mass_g_mol))
                    base.update(status="closest_point_no_intersection", is_approximate=True,
                                ml_gchg_rt_at_selected_T=value, signed_gap_gchg_rt=value - target,
                                absolute_gap_gchg_rt=abs(value - target),
                                message="No ML/critical-target intersection in the fixed domain; closest-point constructed critical properties are approximate.")
            if roots:
                base.update(intersection_found=True, temperature_root_count=len(roots), status="no_eligible_candidate")
                for index, root in enumerate(roots):
                    value = float(_values(predictor, np.asarray([root]))[0])
                    slopes = []
                    for step in FD_RELATIVE_STEPS:
                        left, right = root * (1 - step), root * (1 + step)
                        slope = math.nan
                        if TEMPERATURE_MIN_K < left and right < TEMPERATURE_MAX_K:
                            pair = _values(predictor, np.asarray([left, right]))
                            slope = float((pair[1] - pair[0]) / (2 * root * step))
                        slopes.append(slope)
                    monotonic_bracket = any(a <= root <= b and positive for a, b, positive in brackets)
                    # Exact grid roots can be admitted when their surrounding interval is positive.
                    grid_index = int(np.argmin(abs(grid - root)))
                    if grid[grid_index] == root and 0 < grid_index < len(grid) - 1:
                        monotonic_bracket |= bool(np.all(gradients[grid_index - 1:grid_index + 2] > 0))
                    positive = monotonic_bracket and all(math.isfinite(s) and s > 0 for s in slopes)
                    candidate = {
                        "seed": seed, "model_id": model_id, "canonical_smiles_nonisomeric": "",
                        "root_index": index, "temperature_root_K": root, "stability_class": "positive" if positive else "rejected",
                        "fixed_point_selected": False, "candidate_source": "direct_monotonic_root",
                        "candidate_id": f"{model_id}:{index:06d}", "direct_monotonic_candidate": positive,
                        "critical_target_gchg_rt": target, "root_residual_gchg_rt": value - target,
                        "ml_gchg_derivative_relstep_0.001": slopes[0],
                        "ml_gchg_derivative_relstep_0.003": slopes[1],
                        "ml_gchg_derivative_relstep_0.01": slopes[2],
                        "validation_class": "not_attempted_nonmonotonic_root",
                    }
                    if positive and abs(value - target) <= 2e-5:
                        try:
                            validation, continuation = _validate_vle(candidate, predictor, reduced, sigma_a)
                            candidate.update(validation)
                            continuation_rows.extend(continuation)
                        except Exception as exc:
                            candidate.update(validation_class="inconclusive_vle_evaluation", message=str(exc))
                    elif abs(value - target) > 2e-5:
                        candidate["validation_class"] = "root_residual_failed"
                    candidate_rows.append(candidate)
        except Exception as exc:
            base.update(status="critical_evaluation_failed", message=f"{type(exc).__name__}: {exc}")
        rows.append(base)
    candidates = pd.DataFrame(candidate_rows)
    choices = pd.DataFrame()
    if not candidates.empty:
        eligible = candidates.loc[candidates["direct_monotonic_candidate"] & candidates["validation_class"].isin(
            {"vle_validated_strong", "vle_supportive"})]
        selections, choices = select_cross_seed_branches(
            eligible, candidates[["seed", "canonical_smiles_nonisomeric"]].drop_duplicates(), "main_strong_plus_supportive"
        )
        for base in rows:
            if not base["intersection_found"] or base["status"] == "critical_evaluation_failed":
                continue
            selected = selections.loc[selections["seed"].eq(int(base["model_id"].removeprefix("model_")))]
            if not selected.empty:
                selection = selected.iloc[0]
                base.update(status=selection["status"], message=selection["selection_reason"])
                if selection["status"] == "ok":
                    t = float(selection["TC_pred_K"])
                    value = float(_values(predictors[base["model_id"]], np.asarray([t]))[0])
                    base.update(critical_physical_state(reduced, t, sigma_a, molar_mass_g_mol))
                    base.update(ml_gchg_rt_at_selected_T=value, signed_gap_gchg_rt=value - target,
                                absolute_gap_gchg_rt=abs(value - target), selected_root_index=int(selection["selected_root_index"]),
                                validation_class=selection["validation_class"])
    return CriticalRun(pd.DataFrame(rows), pd.DataFrame(curve_rows), candidates,
                       pd.DataFrame(continuation_rows), choices)
