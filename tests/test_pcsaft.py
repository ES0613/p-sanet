import math

import pytest

from physaft_esper_pvap.pcsaft import gchg_over_rt
from physaft_esper_pvap.solver import solve_vapor_pressure


@pytest.mark.parametrize(
    ("temperature", "m", "sigma", "epsilon", "expected"),
    [
        (300.0, 3.0, 3.5, 250.0, -27.76057828779513),
        (500.0, 2.0, 3.8, 300.0, -13.614037982229839),
    ],
)
def test_gchg_reference_values(temperature, m, sigma, epsilon, expected) -> None:
    assert gchg_over_rt(temperature, m, sigma, epsilon, 0.4) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize(
    ("temperature", "epsilon", "reference_pressure"),
    [
        (226.06, 244.15199972791459, 53.64869332924947),
        (367.3475, 248.10460025450152, 89007.98219605371),
        (508.635, 249.21776812071317, 1706351.2743922565),
    ],
)
def test_data_free_solver_reproduces_reference_pcsaft_output(
    temperature: float, epsilon: float, reference_pressure: float
) -> None:
    result = solve_vapor_pressure(
        temperature,
        m=3.3046192918322,
        sigma_a=3.731325692762098,
        epsilon_k=epsilon,
    )
    assert result.converged
    assert result.fugacity_residual <= 1e-5
    assert result.phase_gap > 1e-12
    assert abs(math.log(result.pressure_pa) - math.log(reference_pressure)) <= 1e-4
