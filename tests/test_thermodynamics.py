import contextlib
import io
import math

import numpy as np
import pytest

from physaft_esper_pvap.pcsaft import Component, EOS, make_eos
from physaft_esper_pvap.solver import solve_vapor_pressure
from physaft_esper_pvap.thermodynamics import molar_density, residual_properties, saturation_caloric_properties


COEFFICIENTS = (250.0, 500.0, 0.01)


def state(t, eta, coefficients=COEFFICIENTS):
    eos = EOS(1, t, [1.0], [Component(3, 3.5, PCSAFTeps_coeffs=coefficients,
              PCSAFTeps_Tmin=200, PCSAFTeps_Tmax=450)], np.zeros((1, 1)))
    with contextlib.redirect_stdout(io.StringIO()):
        eos.PCSAFT_parameter()
        eos.dense = eta
        eos.PCSAFT_P()
        eos.PCSAFT_Ares()
        eos.PCSAFT_Hres()
    return eos


@pytest.mark.parametrize('temperature,eta', [(250, .001), (300, .45), (420, .001)])
def test_analytic_helmholtz_derivative_at_fixed_number_density(temperature, eta):
    center = state(temperature, eta)
    rho = center.rho
    def helmholtz(t):
        epsilon = COEFFICIENTS[0] + COEFFICIENTS[1] / t + COEFFICIENTS[2] * t
        d = 3.5 * (1 - .12 * math.exp(-3 * epsilon / t))
        test_eta = rho * math.pi / 6 * 3 * d**3
        return state(t, test_eta).Ares
    h = .02
    derivative = (helmholtz(temperature - 2*h) - 8*helmholtz(temperature - h)
                  + 8*helmholtz(temperature + h) - helmholtz(temperature + 2*h)) / (12*h)
    assert abs(center.dAresdT - derivative) <= 1e-9 + 5e-6 * abs(derivative)
    assert center.Gresd == pytest.approx(center.Hres - temperature * center.Sres, abs=1e-10)
    assert center.Gresd == pytest.approx(center.R * temperature * (center.Ares + center.Zres - math.log(center.Zges)), abs=1e-9)


@pytest.mark.parametrize('temperature,derivative,region', [
    (190, 0, 'below_Tmin_clamped'), (200, -.0025, 'at_Tmin_inward_derivative'),
    (300, .0044444444444444444, 'within_range'),
    (450, .01-500/450**2, 'at_Tmax_inward_derivative'), (460, 0, 'above_Tmax_clamped'),
])
def test_clamp_values_and_inward_endpoint_derivative(temperature, derivative, region):
    result = residual_properties(temperature, .0001, 3, 3.5, COEFFICIENTS, 200, 450)
    assert result['epsilon_derivative'] == pytest.approx(derivative)
    assert result['epsilon_region'] == region
    evaluation_t = min(max(temperature, 200), 450)
    assert result['epsilon_K'] == pytest.approx(250 + 500/evaluation_t + .01*evaluation_t)


def test_constant_epsilon_matches_scalar_interface_and_low_density_limit():
    scalar = make_eos(300, 3, 3.5, 250)
    scalar.dense = .001
    scalar.PCSAFT_P()
    scalar.PCSAFT_Ares()
    scalar.PCSAFT_Hres()
    coefficient = state(300, .001, (250, 0, 0))
    for name in ('Ares', 'dAresdT', 'Hres', 'Sres', 'Gresd'):
        assert getattr(coefficient, name) == pytest.approx(getattr(scalar, name), abs=1e-12, rel=1e-12)
    low = state(300, 1e-9)
    assert abs(low.Hres) < .01
    assert abs(low.Sres) < 1e-5


def test_saturation_vaporization_identity_and_clapeyron():
    temperature = 300.
    def saturation(t):
        epsilon = 250 + 500/t + .01*t
        return solve_vapor_pressure(t, 3, 3.5, epsilon)
    center = saturation(temperature)
    assert center.converged
    result = saturation_caloric_properties(temperature, center.eta_liq, center.eta_vap,
                                            3, 3.5, COEFFICIENTS, 200, 450)
    assert result['Hvap_J_mol'] > 0
    assert result['Svap_J_mol_K'] == pytest.approx(result['Hvap_J_mol']/temperature, abs=1e-7)
    step = .1
    p = {i: saturation(temperature + i*step).pressure_pa for i in (-2, -1, 1, 2)}
    numeric = (p[-2] - 8*p[-1] + 8*p[1] - p[2])/(12*step)
    epsilon = 250 + 500/temperature + .01*temperature
    vl = 1/molar_density(center.eta_liq, temperature, 3, 3.5, epsilon)
    vv = 1/molar_density(center.eta_vap, temperature, 3, 3.5, epsilon)
    clapeyron = result['Hvap_J_mol']/(temperature*(vv-vl))
    assert clapeyron == pytest.approx(numeric, rel=.005)


def test_association_is_explicitly_unsupported():
    eos = EOS(1, 300, [1.0], [Component(3, 3.5, 250, .02, 1200)], np.zeros((1, 1)))
    with pytest.raises(NotImplementedError):
        eos.PCSAFT_Hres()
