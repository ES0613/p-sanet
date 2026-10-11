import numpy as np

from physaft_thermo.epsilon import fit_epsilon_curve
from physaft_thermo.pcsaft import gchg_over_rt


def test_epsilon_fit_stays_inside_bounds_and_reconstructs_curve() -> None:
    temperatures = np.asarray([260.0, 300.0, 340.0, 380.0, 420.0])
    expected_epsilon = 240.0 + 500.0 / temperatures + 0.01 * temperatures
    targets = np.asarray(
        [gchg_over_rt(t, 3.0, 3.6, epsilon, 0.4) for t, epsilon in zip(temperatures, expected_epsilon)]
    )
    result = fit_epsilon_curve(temperatures, targets, m=3.0, sigma_a=3.6)
    assert result.valid
    assert result.epsilon_min_over_grid_k > 1e-8
    assert result.epsilon_max_over_grid_k <= 5000.0
    assert result.mae_gchg_rt < 1e-2
