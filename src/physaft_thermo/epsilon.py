"""Regression of temperature-dependent PC-SAFT epsilon from ML Gchg/RT."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import minimize

from .constants import (
    EPSILON_CONSTANT_STARTS_K,
    EPSILON_DENSE,
    EPSILON_MAX_K,
    EPSILON_MIN_K,
)
from .pcsaft import gchg_over_rt


@dataclass(frozen=True)
class EpsilonFit:
    epsilon0_k: float
    epsilon1_k2: float
    epsilon2_dimensionless: float
    mae_gchg_rt: float
    rmse_gchg_rt: float
    max_abs_error_gchg_rt: float
    selected_initial_constant_k: float
    selected_solution_source: str
    optimizer_success: bool
    optimizer_status: int
    optimizer_message: str
    optimizer_nfev: int
    optimizer_nit: int
    epsilon_min_over_grid_k: float
    epsilon_max_over_grid_k: float
    valid: bool

    @property
    def coefficients(self) -> np.ndarray:
        return np.asarray(
            [self.epsilon0_k, self.epsilon1_k2, self.epsilon2_dimensionless],
            dtype=float,
        )

    def epsilon(self, temperature_k: float | np.ndarray) -> np.ndarray:
        temperatures = np.asarray(temperature_k, dtype=float)
        return self.epsilon0_k + self.epsilon1_k2 / temperatures + self.epsilon2_dimensionless * temperatures

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class _Evaluation:
    coefficients: np.ndarray
    epsilon_values: np.ndarray
    fitted: np.ndarray
    mae: float
    rmse: float
    max_abs_error: float


class _Objective:
    def __init__(self, temperatures: np.ndarray, targets: np.ndarray, m: float, sigma_a: float) -> None:
        self.temperatures = temperatures
        self.targets = targets
        self.m = float(m)
        self.sigma_a = float(sigma_a)
        self.cache: dict[tuple[float, float, float], _Evaluation] = {}

    def evaluate(self, coefficients: np.ndarray) -> _Evaluation:
        coefficients = np.asarray(coefficients, dtype=float)
        if coefficients.shape != (3,):
            raise ValueError("epsilon(T) requires exactly three coefficients.")
        key = tuple(float(value) for value in np.round(coefficients, 12))
        if key in self.cache:
            return self.cache[key]
        epsilon_values = coefficients[0] + coefficients[1] / self.temperatures + coefficients[2] * self.temperatures
        if (
            not np.all(np.isfinite(epsilon_values))
            or np.any(epsilon_values <= EPSILON_MIN_K)
            or np.any(epsilon_values > EPSILON_MAX_K)
        ):
            raise ValueError("epsilon(T) is outside the accepted interval over the requested grid.")
        fitted = np.asarray(
            [
                gchg_over_rt(t, self.m, self.sigma_a, epsilon, EPSILON_DENSE)
                for t, epsilon in zip(self.temperatures, epsilon_values)
            ],
            dtype=float,
        )
        errors = fitted - self.targets
        result = _Evaluation(
            coefficients=coefficients.copy(),
            epsilon_values=epsilon_values,
            fitted=fitted,
            mae=float(np.mean(np.abs(errors))),
            rmse=float(np.sqrt(np.mean(errors**2))),
            max_abs_error=float(np.max(np.abs(errors))),
        )
        self.cache[key] = result
        return result

    def __call__(self, coefficients: np.ndarray) -> float:
        try:
            return self.evaluate(coefficients).mae
        except Exception:
            return 1.0e30


def fit_epsilon_curve(
    temperatures_k: np.ndarray,
    predicted_gchg_rt: np.ndarray,
    m: float,
    sigma_a: float,
) -> EpsilonFit:
    temperatures = np.asarray(temperatures_k, dtype=float)
    targets = np.asarray(predicted_gchg_rt, dtype=float)
    if temperatures.shape != targets.shape or temperatures.ndim != 1:
        raise ValueError("temperatures and Gchg/RT targets must be one-dimensional arrays of equal length.")
    if not np.all(np.isfinite(targets)):
        raise ValueError("Gchg/RT targets contain non-finite values.")

    objective = _Objective(temperatures, targets, m, sigma_a)
    constant_evaluations: list[tuple[float, _Evaluation]] = []
    for constant in EPSILON_CONSTANT_STARTS_K:
        constant_evaluations.append(
            (constant, objective.evaluate(np.asarray([constant, 0.0, 0.0], dtype=float)))
        )
    initial_constant, initial = min(constant_evaluations, key=lambda item: item[1].mae)
    optimization = minimize(
        objective,
        x0=np.asarray([initial_constant, 0.0, 0.0], dtype=float),
        method="Powell",
        options={"maxiter": 40, "xtol": 1.0e-3, "ftol": 1.0e-4, "disp": False},
    )
    candidates = [("best_constant_multistart", initial)]
    try:
        candidates.append(("powell_optimizer", objective.evaluate(np.asarray(optimization.x, dtype=float))))
    except Exception:
        pass
    source, selected = min(candidates, key=lambda item: item[1].mae)
    epsilon_min = float(np.min(selected.epsilon_values))
    epsilon_max = float(np.max(selected.epsilon_values))
    valid = (
        math.isfinite(epsilon_min)
        and math.isfinite(epsilon_max)
        and epsilon_min > EPSILON_MIN_K
        and epsilon_max <= EPSILON_MAX_K
    )
    return EpsilonFit(
        epsilon0_k=float(selected.coefficients[0]),
        epsilon1_k2=float(selected.coefficients[1]),
        epsilon2_dimensionless=float(selected.coefficients[2]),
        mae_gchg_rt=selected.mae,
        rmse_gchg_rt=selected.rmse,
        max_abs_error_gchg_rt=selected.max_abs_error,
        selected_initial_constant_k=float(initial_constant),
        selected_solution_source=source,
        optimizer_success=bool(optimization.success),
        optimizer_status=int(getattr(optimization, "status", 0)),
        optimizer_message=str(getattr(optimization, "message", "")),
        optimizer_nfev=int(getattr(optimization, "nfev", len(objective.cache))),
        optimizer_nit=int(getattr(optimization, "nit", 0) or 0),
        epsilon_min_over_grid_k=epsilon_min,
        epsilon_max_over_grid_k=epsilon_max,
        valid=valid,
    )


__all__ = ["EpsilonFit", "fit_epsilon_curve"]
