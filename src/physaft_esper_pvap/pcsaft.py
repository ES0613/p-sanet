"""Validated pure-component, non-associating PC-SAFT interface."""

from __future__ import annotations

import contextlib
import io
import math
import warnings
from dataclasses import dataclass

import numpy as np

from ._pcsaft_core import Component, EOS


@dataclass(frozen=True)
class PhaseState:
    pressure_pa: float
    residual_lnphi: float
    eta_liq: float
    eta_vap: float
    eta_middle: float

    @property
    def phase_gap(self) -> float:
        return abs(self.eta_liq - self.eta_vap)


def make_eos(temperature_k: float, m: float, sigma_a: float, epsilon_k: float) -> EOS:
    component = Component(
        PCSAFTm=float(m),
        PCSAFTsig=float(sigma_a),
        PCSAFTeps=float(epsilon_k),
    )
    eos = EOS(1, float(temperature_k), [1.0], [component], np.zeros((1, 1)))
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        eos.PCSAFT_parameter()
    return eos


def gchg_over_rt(
    temperature_k: float,
    m: float,
    sigma_a: float,
    epsilon_k: float,
    dense: float = 0.4,
) -> float:
    """Return ``zdsp + Adsp`` at the specified reduced density."""
    eos = make_eos(temperature_k, m, sigma_a, epsilon_k)
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        eos.dense = float(dense)
        eos.PCSAFT_P()
        eos.PCSAFT_Ares()
    value = float(eos.zdsp + eos.Adsp)
    if not math.isfinite(value):
        raise FloatingPointError("PC-SAFT returned non-finite Gchg/RT.")
    return value


def pressure_at_eta(eos: EOS, eta: float) -> float:
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        return float(eos._PCSAFT_pressure_at_eta(float(eta)))


def lnphi_at_eta(eos: EOS, eta: float) -> float:
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        return float(eos._PCSAFT_lnphi_at_eta(float(eta)))


__all__ = [
    "Component",
    "EOS",
    "PhaseState",
    "gchg_over_rt",
    "lnphi_at_eta",
    "make_eos",
    "pressure_at_eta",
]
