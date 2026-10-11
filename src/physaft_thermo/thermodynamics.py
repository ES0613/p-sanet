"""Saturation caloric properties for temperature-dependent, non-associating PC-SAFT.

Hres and Sres are departures from the ideal gas at the same T and P.
The ideal-gas terms cancel between coexisting pure-component phases.
"""

from __future__ import annotations

import contextlib
import io
import math
import warnings

import numpy as np

from .pcsaft import Component, EOS

N_AVOGADRO = 6.0221367e23
R_GAS = 8.314


def molar_density(eta: float, temperature_k: float, m: float, sigma_a: float, epsilon_k: float) -> float:
    """Convert packing fraction to mol/m³ using the EOS's original constants."""
    values = np.asarray([eta, temperature_k, m, sigma_a, epsilon_k], dtype=float)
    if not np.all(np.isfinite(values)) or eta <= 0 or np.any(values[1:] <= 0) or eta >= 1:
        raise ValueError("Density conversion requires 0 < eta < 1 and positive finite parameters.")
    diameter = sigma_a * (1.0 - 0.12 * math.exp(-3.0 * epsilon_k / temperature_k))
    return eta / ((math.pi / 6.0) * m * diameter**3) * 1.0e30 / N_AVOGADRO


def residual_properties(
    temperature_k: float,
    eta: float,
    m: float,
    sigma_a: float,
    epsilon_coefficients: tuple[float, float, float],
    t_min_k: float,
    t_max_k: float,
) -> dict[str, object]:
    """Evaluate analytic fixed-number-density Helmholtz derivatives at a state."""
    if not math.isfinite(temperature_k) or temperature_k <= 0 or not 0 < eta < 1:
        raise ValueError("Require positive finite temperature and 0 < eta < 1.")
    if not all(math.isfinite(v) and v > 0 for v in (m, sigma_a)):
        raise ValueError("m and sigma must be positive and finite.")
    component = Component(
        m, sigma_a, PCSAFTeps_coeffs=epsilon_coefficients,
        PCSAFTeps_Tmin=t_min_k, PCSAFTeps_Tmax=t_max_k,
    )
    eos = EOS(1, temperature_k, [1.0], [component], np.zeros((1, 1)))
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        eos.PCSAFT_parameter()
        eos.dense = eta
        eos.PCSAFT_P()
        eos.PCSAFT_Ares()
        eos.PCSAFT_Hres()
    values = [eos.Hres, eos.Sres, eos.Gresd, eos.dAresdT, eos.pges]
    if not all(math.isfinite(float(v)) for v in values):
        raise FloatingPointError("Non-finite PC-SAFT residual thermodynamic state.")
    return {
        "Hres_J_mol": float(eos.Hres),
        "Sres_J_mol_K": float(eos.Sres),
        "Gres_J_mol": float(eos.Gresd),
        "dAres_dT_per_K": float(eos.dAresdT),
        "P_Pa": float(eos.pges),
        "epsilon_K": float(component.PCSAFTeps),
        "epsilon_derivative": float(component.PCSAFTepsdT),
        "epsilon_region": component.PCSAFTeps_region,
    }


def saturation_caloric_properties(
    temperature_k: float, eta_liq: float, eta_vap: float, m: float, sigma_a: float,
    epsilon_coefficients: tuple[float, float, float], t_min_k: float, t_max_k: float,
) -> dict[str, object]:
    """Return two phase departures and vaporization differences at accepted VLE."""
    if not eta_liq > eta_vap > 0:
        raise ValueError("Require distinct liquid and vapor packing fractions.")
    args = (m, sigma_a, epsilon_coefficients, t_min_k, t_max_k)
    liquid = residual_properties(temperature_k, eta_liq, *args)
    vapor = residual_properties(temperature_k, eta_vap, *args)
    hvap = float(vapor["Hres_J_mol"]) - float(liquid["Hres_J_mol"])
    svap = float(vapor["Sres_J_mol_K"]) - float(liquid["Sres_J_mol_K"])
    if not (math.isfinite(hvap) and math.isfinite(svap) and hvap > 0 and svap > 0):
        raise FloatingPointError("Non-positive or non-finite vaporization enthalpy/entropy.")
    return {
        "Hres_liq_J_mol": liquid["Hres_J_mol"],
        "Hres_vap_J_mol": vapor["Hres_J_mol"],
        "Sres_liq_J_mol_K": liquid["Sres_J_mol_K"],
        "Sres_vap_J_mol_K": vapor["Sres_J_mol_K"],
        "Hvap_J_mol": hvap,
        "Hvap_kJ_mol": hvap / 1000.0,
        "Svap_J_mol_K": svap,
        "thermo_identity_abs_J_mol_K": abs(hvap / temperature_k - svap),
        "epsilon_derivative": liquid["epsilon_derivative"],
        "epsilon_region": liquid["epsilon_region"],
    }
