"""Data-free pure-component PC-SAFT vapor-pressure solver."""

from __future__ import annotations

import contextlib
import io
import math
import warnings
from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from .constants import FUGACITY_RESIDUAL_TOL, PHASE_GAP_TOL, PVAP_MAX_PA, PVAP_MIN_PA
from .pcsaft import EOS, PhaseState, lnphi_at_eta, make_eos, pressure_at_eta


@dataclass(frozen=True)
class VaporPressureResult:
    pressure_pa: float
    ln_pressure_pa: float
    eta_liq: float
    eta_vap: float
    fugacity_residual: float
    phase_gap: float
    status: str
    method: str
    converged: bool
    message: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class _IsothermBranches:
    eos: EOS
    eta_vap_spinodal: float
    eta_liq_spinodal: float
    eta_min: float
    eta_max: float
    pressure_low_pa: float
    pressure_high_pa: float


def _failed(status: str, message: str) -> VaporPressureResult:
    return VaporPressureResult(
        pressure_pa=math.nan,
        ln_pressure_pa=math.nan,
        eta_liq=math.nan,
        eta_vap=math.nan,
        fugacity_residual=math.nan,
        phase_gap=math.nan,
        status=status,
        method="none",
        converged=False,
        message=message,
    )


def _find_isotherm_branches(eos: EOS) -> _IsothermBranches:
    eta_min = 1.0e-30
    eta_max = 0.895
    eta_grid = np.unique(
        np.concatenate(
            [
                np.logspace(-30.0, -3.0, 260),
                np.linspace(1.0e-3, eta_max, 1200),
            ]
        )
    )
    pressures = np.asarray([pressure_at_eta(eos, eta) for eta in eta_grid], dtype=float)
    finite = np.isfinite(pressures)
    if int(finite.sum()) < 10:
        raise RuntimeError("Too few finite points on the packing-fraction isotherm.")

    maxima: list[int] = []
    minima: list[int] = []
    for index in range(1, len(eta_grid) - 1):
        if not (finite[index - 1] and finite[index] and finite[index + 1]):
            continue
        if pressures[index] >= pressures[index - 1] and pressures[index] > pressures[index + 1]:
            maxima.append(index)
        if pressures[index] <= pressures[index - 1] and pressures[index] < pressures[index + 1]:
            minima.append(index)

    pairs: list[tuple[float, int, int]] = []
    for maximum in maxima:
        for minimum in minima:
            if minimum <= maximum:
                continue
            lower = max(float(pressures[minimum]), PVAP_MIN_PA)
            upper = min(float(pressures[maximum]), PVAP_MAX_PA)
            if upper > lower:
                score = math.log(upper / lower)
                pairs.append((score, maximum, minimum))
    if not pairs:
        raise RuntimeError("No three-density-root pressure interval was found; the temperature may be supercritical.")
    _, maximum, minimum = max(pairs, key=lambda item: item[0])

    maximum_fit = minimize_scalar(
        lambda eta: -pressure_at_eta(eos, eta),
        bounds=(float(eta_grid[maximum - 1]), float(eta_grid[maximum + 1])),
        method="bounded",
    )
    minimum_fit = minimize_scalar(
        lambda eta: pressure_at_eta(eos, eta),
        bounds=(float(eta_grid[minimum - 1]), float(eta_grid[minimum + 1])),
        method="bounded",
    )
    eta_vap_spinodal = float(maximum_fit.x if maximum_fit.success else eta_grid[maximum])
    eta_liq_spinodal = float(minimum_fit.x if minimum_fit.success else eta_grid[minimum])
    p_high = min(pressure_at_eta(eos, eta_vap_spinodal), PVAP_MAX_PA)
    p_low = max(pressure_at_eta(eos, eta_liq_spinodal), PVAP_MIN_PA)
    if not (math.isfinite(p_low) and math.isfinite(p_high) and p_high > p_low):
        raise RuntimeError("Invalid spinodal pressure interval.")
    return _IsothermBranches(
        eos=eos,
        eta_vap_spinodal=eta_vap_spinodal,
        eta_liq_spinodal=eta_liq_spinodal,
        eta_min=eta_min,
        eta_max=eta_max,
        pressure_low_pa=p_low,
        pressure_high_pa=p_high,
    )


def _eta_root(eos: EOS, pressure_pa: float, low: float, high: float) -> float:
    function = lambda eta: pressure_at_eta(eos, eta) - pressure_pa
    f_low = function(low)
    f_high = function(high)
    if not (math.isfinite(f_low) and math.isfinite(f_high)) or f_low * f_high > 0.0:
        raise RuntimeError("Density branch did not bracket the requested pressure.")
    # An absolute tolerance suitable for liquid eta is far too loose for a
    # low-pressure vapor root (eta can be ~1e-11 or smaller).  Use a nearly
    # pure relative criterion so log-pressure accuracy is not lost there.
    return float(brentq(function, low, high, xtol=1.0e-30, rtol=1.0e-14, maxiter=160))


def _phase_state(branches: _IsothermBranches, pressure_pa: float) -> PhaseState:
    eos = branches.eos
    eta_vap = _eta_root(eos, pressure_pa, branches.eta_min, branches.eta_vap_spinodal)
    eta_middle = _eta_root(eos, pressure_pa, branches.eta_vap_spinodal, branches.eta_liq_spinodal)
    eta_liq = _eta_root(eos, pressure_pa, branches.eta_liq_spinodal, branches.eta_max)
    residual = lnphi_at_eta(eos, eta_liq) - lnphi_at_eta(eos, eta_vap)
    if not math.isfinite(residual):
        raise RuntimeError("Non-finite fugacity residual.")
    return PhaseState(float(pressure_pa), float(residual), eta_liq, eta_vap, eta_middle)


def _accepted(state: PhaseState, method: str, message: str = "") -> VaporPressureResult:
    phase_gap = state.phase_gap
    accepted = abs(state.residual_lnphi) <= FUGACITY_RESIDUAL_TOL and phase_gap > PHASE_GAP_TOL
    return VaporPressureResult(
        pressure_pa=state.pressure_pa if accepted else math.nan,
        ln_pressure_pa=math.log(state.pressure_pa) if accepted else math.nan,
        eta_liq=state.eta_liq if accepted else math.nan,
        eta_vap=state.eta_vap if accepted else math.nan,
        fugacity_residual=abs(state.residual_lnphi) if accepted else math.nan,
        phase_gap=phase_gap if accepted else math.nan,
        status="ok" if accepted else "rejected_solution",
        method=method,
        converged=accepted,
        message=message,
    )


def _solve_bracketed(branches: _IsothermBranches, pressure_hint_pa: float | None) -> VaporPressureResult:
    margin = 1.0e-8
    low = branches.pressure_low_pa * (1.0 + margin)
    high = branches.pressure_high_pa * (1.0 - margin)
    if not high > low:
        raise RuntimeError("Three-root pressure window collapsed after endpoint protection.")

    pressures = list(np.geomspace(low, high, 56))
    fixed = np.geomspace(PVAP_MIN_PA, PVAP_MAX_PA, 22)
    pressures.extend(float(value) for value in fixed if low < value < high)
    if pressure_hint_pa is not None and math.isfinite(pressure_hint_pa) and low < pressure_hint_pa < high:
        pressures.extend(
            float(value)
            for value in pressure_hint_pa * np.exp(np.linspace(-2.0, 2.0, 17))
            if low < value < high
        )
    pressures = sorted(set(pressures))

    states: list[PhaseState] = []
    for pressure in pressures:
        try:
            states.append(_phase_state(branches, pressure))
        except Exception:
            continue
    if not states:
        raise RuntimeError("No valid three-root fugacity states were found.")
    best = min(states, key=lambda state: abs(state.residual_lnphi))

    for left, right in zip(states[:-1], states[1:]):
        if left.residual_lnphi * right.residual_lnphi > 0.0:
            continue
        state_cache: dict[float, PhaseState] = {}

        def residual(log_pressure: float) -> float:
            key = round(float(log_pressure), 14)
            if key not in state_cache:
                state_cache[key] = _phase_state(branches, math.exp(log_pressure))
            return state_cache[key].residual_lnphi

        log_pressure = brentq(
            residual,
            math.log(left.pressure_pa),
            math.log(right.pressure_pa),
            xtol=1.0e-10,
            rtol=1.0e-12,
            maxiter=100,
        )
        state = _phase_state(branches, math.exp(log_pressure))

        # Very dilute vapor fugacity can contain a small numerical ripple.
        # Minimize |residual| inside the same sign-change bracket and retain
        # whichever state has the smaller thermodynamic residual.
        refined_minimum = minimize_scalar(
            lambda value: abs(residual(float(value))),
            bounds=(math.log(left.pressure_pa), math.log(right.pressure_pa)),
            method="bounded",
            options={"xatol": 1.0e-12, "maxiter": 120},
        )
        if refined_minimum.success:
            refined_state = _phase_state(branches, math.exp(float(refined_minimum.x)))
            if abs(refined_state.residual_lnphi) < abs(state.residual_lnphi):
                state = refined_state
        result = _accepted(state, "isotherm_log_pressure_brent")
        if result.converged:
            return result

    # At extremely low pressure, finite-precision evaluation of the vapor
    # density can make the residual touch zero without a numerically visible
    # sign change. Refine the neighborhood of the best scan point by
    # minimizing |residual|, while retaining the same strict acceptance gate.
    best_index = states.index(best)
    neighborhood_low = states[max(0, best_index - 1)].pressure_pa
    neighborhood_high = states[min(len(states) - 1, best_index + 1)].pressure_pa
    if neighborhood_high > neighborhood_low:
        state_cache: dict[float, PhaseState] = {}

        def absolute_residual(log_pressure: float) -> float:
            key = round(float(log_pressure), 14)
            try:
                if key not in state_cache:
                    state_cache[key] = _phase_state(branches, math.exp(log_pressure))
                return abs(state_cache[key].residual_lnphi)
            except Exception:
                return 1.0e30

        minimum = minimize_scalar(
            absolute_residual,
            bounds=(math.log(neighborhood_low), math.log(neighborhood_high)),
            method="bounded",
            options={"xatol": 1.0e-10, "maxiter": 120},
        )
        if minimum.success:
            refined = _phase_state(branches, math.exp(float(minimum.x)))
            result = _accepted(refined, "isotherm_min_abs_fugacity_residual")
            if result.converged:
                return result
    if abs(best.residual_lnphi) <= FUGACITY_RESIDUAL_TOL:
        return _accepted(best, "isotherm_scan")
    raise RuntimeError(
        f"Three-root states were found, but no fugacity bracket converged; best residual={abs(best.residual_lnphi):.3e}."
    )


def _direct_multistart(
    temperature_k: float,
    m: float,
    sigma_a: float,
    epsilon_k: float,
    pressure_hint_pa: float | None,
    branches: _IsothermBranches | None,
) -> VaporPressureResult:
    guesses: list[float] = []
    if pressure_hint_pa is not None and math.isfinite(pressure_hint_pa) and pressure_hint_pa > 0.0:
        guesses.append(float(np.clip(pressure_hint_pa, PVAP_MIN_PA, PVAP_MAX_PA)))
    guesses.extend(float(value) for value in np.geomspace(PVAP_MIN_PA, PVAP_MAX_PA, 8))
    seen: set[float] = set()
    for guess in guesses:
        key = round(math.log10(guess), 10)
        if key in seen:
            continue
        seen.add(key)
        eos = make_eos(temperature_k, m, sigma_a, epsilon_k)
        try:
            with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
                warnings.simplefilter("ignore")
                pressure = float(eos.cal_Pvap_PCSAFT(guess, max_iter=160, tol=1.0e-8))
            if not (math.isfinite(pressure) and PVAP_MIN_PA <= pressure <= PVAP_MAX_PA):
                continue
            active_branches = branches or _find_isotherm_branches(eos)
            state = _phase_state(active_branches, pressure)
            result = _accepted(state, "deterministic_log_multistart", f"initial_guess_pa={guess:.8e}")
            if result.converged:
                return result
        except Exception:
            continue
    return _failed("solver_failed", "Both the isotherm bracket and deterministic multistart failed.")


def solve_vapor_pressure(
    temperature_k: float,
    m: float,
    sigma_a: float,
    epsilon_k: float,
    pressure_hint_pa: float | None = None,
) -> VaporPressureResult:
    """Solve vapor pressure without experimental pressure guesses or coefficients."""
    values = (temperature_k, m, sigma_a, epsilon_k)
    if not all(math.isfinite(float(value)) and float(value) > 0.0 for value in values):
        return _failed("invalid_input", "T, m, sigma, and epsilon must all be finite and positive.")
    eos = make_eos(temperature_k, m, sigma_a, epsilon_k)
    branches: _IsothermBranches | None = None
    bracket_error = ""
    try:
        branches = _find_isotherm_branches(eos)
        return _solve_bracketed(branches, pressure_hint_pa)
    except Exception as exc:
        bracket_error = str(exc)
    fallback = _direct_multistart(
        temperature_k,
        m,
        sigma_a,
        epsilon_k,
        pressure_hint_pa,
        branches,
    )
    if not fallback.converged and bracket_error:
        return VaporPressureResult(**{**fallback.to_dict(), "message": f"{bracket_error} {fallback.message}".strip()})
    return fallback


__all__ = ["VaporPressureResult", "solve_vapor_pressure"]
