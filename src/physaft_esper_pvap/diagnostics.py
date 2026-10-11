"""Dependency-free SVG diagnostics for the critical ML curves and target gap."""

from __future__ import annotations

import html
import math
from pathlib import Path

import numpy as np
import pandas as pd


def write_critical_diagnostics(output: Path, curves: pd.DataFrame, properties: pd.DataFrame) -> None:
    if curves.empty:
        return
    colors = ["#2563eb", "#dc2626", "#059669", "#9333ea", "#d97706"]
    ids = properties["model_id"].tolist()
    width, panel_height = 860, 280
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{len(ids) * panel_height + 45}" viewBox="0 0 {width} {len(ids) * panel_height + 45}">',
           '<rect width="100%" height="100%" fill="white"/>',
           '<g font-family="Arial,sans-serif" font-size="12" fill="#1f2937">',
           '<text x="45" y="25" font-size="17">ML Gchg/RT versus PC-SAFT critical target (eta_ref = 0.4)</text>']
    for i, model_id in enumerate(ids):
        group = curves.loc[curves["model_id"].eq(model_id)]
        row = properties.loc[properties["model_id"].eq(model_id)].iloc[0]
        top = 45 + i * panel_height
        status = html.escape(str(row["status"]))
        svg.append(f'<text x="70" y="{top + 16}" font-weight="bold">{html.escape(model_id)} — {status}</text>')
        if group.empty:
            svg.append(f'<text x="70" y="{top + 55}">{html.escape(str(row.get("message", "Curve unavailable")))}</text>')
            continue
        t = group["T_K"].to_numpy(float)
        g = group["Gchg_over_RT"].to_numpy(float)
        target = float(group.iloc[0]["critical_target_gchg_rt"])
        low, high = min(float(g.min()), target), max(float(g.max()), target)
        margin = max((high - low) * 0.05, 0.01)
        low, high = low - margin, high + margin
        x = lambda value: 75 + (math.log(value) - math.log(t[0])) / math.log(t[-1] / t[0]) * 735
        y = lambda value: top + 200 - (value - low) / (high - low) * 150
        points = " ".join(f"{x(a):.2f},{y(b):.2f}" for a, b in zip(t, g))
        svg.extend([f'<path d="M75 {top + 45} V{top + 200} H810" fill="none" stroke="#9ca3af"/>',
                    f'<polyline points="{points}" fill="none" stroke="{colors[i % 5]}" stroke-width="1.8"/>',
                    f'<path d="M75 {y(target):.2f} H810" stroke="#111827" stroke-dasharray="5 4"/>',
                    f'<text x="78" y="{top + 40}">Gchg/RT; target = {target:.7g}</text>'])
        for tick in (low, (low + high) / 2.0, high):
            svg.append(f'<text x="68" y="{y(tick) + 4:.2f}" text-anchor="end">{tick:.4g}</text>')
        for tick in (t[0], 30, 100, 300, 1000, t[-1]):
            if t[0] <= tick <= t[-1]:
                svg.append(f'<text x="{x(tick):.2f}" y="{top + 218}" text-anchor="middle">{tick:.6g}</text>')
        svg.append(f'<text x="440" y="{top + 237}" text-anchor="middle">Temperature (K, logarithmic axis)</text>')
        tc = float(row.get("Tc_K", math.nan))
        value = float(row.get("ml_gchg_rt_at_selected_T", math.nan))
        if math.isfinite(tc) and math.isfinite(value):
            svg.extend([f'<path d="M{x(tc):.2f} {y(value):.2f} V{y(target):.2f}" stroke="#be123c" stroke-width="3"/>',
                        f'<circle cx="{x(tc):.2f}" cy="{y(value):.2f}" r="4" fill="#be123c"/>',
                        f'<text x="75" y="{top + 259}">Selected T = {tc:.7g} K; ML = {value:.7g}; signed gap = {value-target:.7g}; |gap| = {abs(value-target):.7g}</text>'])
    svg.append('</g></svg>')
    (output / "critical_gchg_diagnostics.svg").write_text("\n".join(svg) + "\n", encoding="utf-8")


def property_summary(frame: pd.DataFrame, group_columns: list[str], value_columns: list[str]) -> pd.DataFrame:
    """Uncalibrated ensemble summaries; failed rows never contribute numeric values."""
    if frame.empty:
        return pd.DataFrame(columns=[*group_columns, "successful_model_count", "requested_model_count"])
    rows = []
    groups = frame.groupby(group_columns, sort=False) if group_columns else [((), frame)]
    for key, group in groups:
        if not isinstance(key, tuple):
            key = (key,)
        valid = group.loc[group["status"].isin({"ok", "closest_point_no_intersection"})]
        row = dict(zip(group_columns, key))
        row.update(successful_model_count=len(valid), requested_model_count=len(group),
                   uncertainty_note="Uncalibrated between-model sample spread; not an error band.")
        for column in value_columns:
            values = valid[column].dropna().to_numpy(float) if column in valid else np.asarray([])
            row[column + "_mean"] = float(values.mean()) if len(values) else math.nan
            row[column + "_std"] = float(values.std(ddof=1)) if len(values) > 1 else (0.0 if len(values) == 1 else math.nan)
        if "is_approximate" in group:
            row["uses_approximate_Tc"] = bool(valid["is_approximate"].any())
        rows.append(row)
    return pd.DataFrame(rows)
