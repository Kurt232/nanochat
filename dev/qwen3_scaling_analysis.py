"""Fit and plot the Qwen3 fixed-token/parameter scaling miniseries.

The checked-in CSV is the source of truth. This script deliberately fits a
simple log-linear power law over the measured range; five co-scaled runs are
not enough to estimate a Chinchilla-style compute-optimal frontier.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from xml.sax.saxutils import escape


HERE = Path(__file__).resolve().parent


def load_results(path: Path) -> list[dict[str, float | str]]:
    numeric = {
        "parameters",
        "tokens",
        "flops_per_token",
        "val_bpb",
        "training_seconds",
        "hidden_size",
        "intermediate_size",
        "layers",
        "q_heads",
        "kv_heads",
    }
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in numeric:
            row[key] = float(row[key])
        row["training_flops"] = row["tokens"] * row["flops_per_token"]
    return rows


def fit_power_law(x: list[float], y: list[float]) -> tuple[float, float, float]:
    """Fit y = coefficient * x**exponent and return coefficient/exponent/R2."""
    log_x = [math.log(value) for value in x]
    log_y = [math.log(value) for value in y]
    mean_x = sum(log_x) / len(log_x)
    mean_y = sum(log_y) / len(log_y)
    exponent = sum((a - mean_x) * (b - mean_y) for a, b in zip(log_x, log_y)) / sum(
        (value - mean_x) ** 2 for value in log_x
    )
    log_coefficient = mean_y - exponent * mean_x
    prediction = [log_coefficient + exponent * value for value in log_x]
    residual = sum((actual - predicted) ** 2 for actual, predicted in zip(log_y, prediction))
    total = sum((value - mean_y) ** 2 for value in log_y)
    r_squared = 1.0 - residual / total
    return math.exp(log_coefficient), exponent, r_squared


def plot(rows: list[dict[str, float | str]], output: Path) -> None:
    params = [float(row["parameters"]) for row in rows]
    compute = [float(row["training_flops"]) for row in rows]
    bpb = [float(row["val_bpb"]) for row in rows]
    names = [str(row["model"]) for row in rows]

    param_coefficient, param_exponent, param_r2 = fit_power_law(params, bpb)
    compute_coefficient, compute_exponent, compute_r2 = fit_power_law(compute, bpb)
    matched = [
        row for row in rows if 49.8 <= float(row["tokens"]) / float(row["parameters"]) <= 50.2
    ]
    matched_compute_coefficient, matched_compute_exponent, matched_compute_r2 = fit_power_law(
        [float(row["training_flops"]) for row in matched],
        [float(row["val_bpb"]) for row in matched],
    )

    width, height = 1200, 500
    panel_width, panel_height = 510, 350
    panel_top = 90
    specifications = [
        (70, params, param_coefficient, param_exponent, param_r2, "Parameters", "params"),
        (650, compute, compute_coefficient, compute_exponent, compute_r2, "Training FLOPs", "FLOPs"),
    ]
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:system-ui,sans-serif;fill:#172033}.title{font-size:20px;font-weight:600}.axis{font-size:13px}.note{font-size:12px}.grid{stroke:#d6dbe3;stroke-width:1}.frame{stroke:#748094;stroke-width:1.2;fill:none}</style>',
        '<text x="600" y="35" text-anchor="middle" class="title">Qwen3 architecture + nanochat training stack: scaling miniseries</text>',
    ]
    y_min = min(bpb) - 0.04
    y_max = max(bpb) + 0.04

    for left, x, coefficient, exponent, r_squared, label, unit in specifications:
        log_min, log_max = math.log10(min(x)), math.log10(max(x))

        def sx(value: float) -> float:
            return left + (math.log10(value) - log_min) / (log_max - log_min) * panel_width

        def sy(value: float) -> float:
            return panel_top + (y_max - value) / (y_max - y_min) * panel_height

        for tick in range(5):
            value = y_min + tick * (y_max - y_min) / 4
            y_coord = sy(value)
            elements.append(f'<line x1="{left}" y1="{y_coord:.1f}" x2="{left + panel_width}" y2="{y_coord:.1f}" class="grid"/>')
            elements.append(f'<text x="{left - 8}" y="{y_coord + 4:.1f}" text-anchor="end" class="axis">{value:.2f}</text>')
        elements.append(f'<rect x="{left}" y="{panel_top}" width="{panel_width}" height="{panel_height}" class="frame"/>')

        line_points = []
        for index in range(101):
            log_value = log_min + index * (log_max - log_min) / 100
            value = 10**log_value
            line_points.append(f"{sx(value):.1f},{sy(coefficient * value**exponent):.1f}")
        elements.append(f'<polyline points="{" ".join(line_points)}" fill="none" stroke="#f97316" stroke-width="2.5"/>')

        for name, x_value, y_value in zip(names, x, bpb):
            x_coord, y_coord = sx(x_value), sy(y_value)
            elements.append(f'<circle cx="{x_coord:.1f}" cy="{y_coord:.1f}" r="5" fill="#2563eb"/>')
            elements.append(f'<text x="{x_coord + 7:.1f}" y="{y_coord - 7:.1f}" class="note">{escape(name)}</text>')

        elements.append(f'<text x="{left + panel_width / 2}" y="{panel_top + panel_height + 35}" text-anchor="middle" class="axis">{escape(label)} (log scale)</text>')
        elements.append(f'<text x="{left + panel_width / 2}" y="{panel_top - 18}" text-anchor="middle" class="axis">BPB ∝ {unit}^{exponent:.4f} (log-space R²={r_squared:.4f})</text>')

    elements.append('</svg>')
    output.write_text("\n".join(elements) + "\n")
    print(f"parameters: BPB = {param_coefficient:.6g} * P^{param_exponent:.6f}; R2={param_r2:.6f}")
    print(f"compute:    BPB = {compute_coefficient:.6g} * C^{compute_exponent:.6f}; R2={compute_r2:.6f}")
    print(
        "matched:    "
        f"BPB = {matched_compute_coefficient:.6g} * C^{matched_compute_exponent:.6f}; "
        f"R2={matched_compute_r2:.6f} ({len(matched)} points)"
    )
    print(f"wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=HERE / "qwen3_scaling_results.csv")
    parser.add_argument("--output", type=Path, default=HERE / "qwen3_scaling.svg")
    args = parser.parse_args()
    plot(load_results(args.input), args.output)


if __name__ == "__main__":
    main()
