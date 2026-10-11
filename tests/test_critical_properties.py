"""Critical-state equations and explicit approximation statuses."""

import math

import numpy as np
import pytest

from physaft_esper_pvap.critical_properties import (
    TEMPERATURE_MAX_K, TEMPERATURE_MIN_K, critical_physical_state,
    solve_critical_properties, solve_reduced_critical_point,
)
from physaft_esper_pvap.pcsaft import gchg_over_rt


def test_reduced_critical_reference_and_unit_conversions():
    reduced = solve_reduced_critical_point(5.0)
    assert reduced.t_tilde_c == pytest.approx(2.648680568587752, abs=2e-10)
    assert abs(reduced.residual_d1) < 1e-10
    assert abs(reduced.residual_d2) < 1e-10
    state = critical_physical_state(reduced, 500, 3.5, 44)
    assert state["Pc_Pa"] == pytest.approx(state["Zc"] * state["rho_c_mol_m3"] * 8.314 * 500)
    assert state["rho_c_kg_m3"] == pytest.approx(state["rho_c_mol_m3"] * .044)
    assert state["critical_molar_volume_m3_mol"] == pytest.approx(1 / state["rho_c_mol_m3"])


def test_no_intersection_reports_approximate_closest_boundary(tmp_path):
    reduced = solve_reduced_critical_point(3.0)
    target = gchg_over_rt(1, 3, 3.5, 1 / reduced.t_tilde_c)
    predictor = lambda ts: target + 3.0 + .001 * ts
    derivative = lambda ts: np.full_like(ts, .001)
    result = solve_critical_properties(3, 3.5, 44, {"model_1": predictor}, {"model_1": derivative})
    row = result.properties.iloc[0]
    assert row.status == "closest_point_no_intersection"
    assert row.Tc_K == pytest.approx(TEMPERATURE_MIN_K)
    assert row.is_approximate and not row.intersection_found
    assert row.absolute_gap_gchg_rt == pytest.approx(3 + .001 * TEMPERATURE_MIN_K)


def test_negative_slope_intersection_is_rejected():
    reduced = solve_reduced_critical_point(3.0)
    target = gchg_over_rt(1, 3, 3.5, 1 / reduced.t_tilde_c)
    predictor = lambda ts: target - (ts - 500) * .01
    derivative = lambda ts: np.full_like(ts, -.01)
    result = solve_critical_properties(3, 3.5, 44, {"model_1": predictor}, {"model_1": derivative})
    row = result.properties.iloc[0]
    assert row.intersection_found and not row.is_approximate
    assert row.status == "no_eligible_candidate"
    assert math.isnan(row.Tc_K)
