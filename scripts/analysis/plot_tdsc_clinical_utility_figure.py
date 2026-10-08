#!/usr/bin/env python3
"""Create the paper-ready TDSC decision-curve figure from saved summaries."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


KEY_THRESHOLDS = np.asarray([0.10, 0.20, 0.30, 0.40, 0.50])
STRATEGIES = {
    "equal_logit": {
        "label": "R3D-TriMIL",
        "color": "#000000",
        "linewidth": 2.2,
        "linestyle": "-",
        "zorder": 5,
    },
    "r3d": {
        "label": "R3D-18",
        "color": "#0072B2",
        "linewidth": 1.7,
        "linestyle": "--",
        "zorder": 4,
    },
    "triplanar": {
        "label": "TriMIL",
        "color": "#D55E00",
        "linewidth": 1.7,
        "linestyle": "-.",
        "zorder": 4,
    },
    "treat_all": {
        "label": "Escalate all",
        "color": "#7A7A7A",
        "linewidth": 1.4,
        "linestyle": (0, (4, 2)),
        "zorder": 2,
    },
    "treat_none": {
        "label": "Escalate none",
        "color": "#4D4D4D",
        "linewidth": 1.2,
        "linestyle": ":",
        "zorder": 1,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curve-csv", type=Path, required=True)
    parser.add_argument("--contrast-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--stem", default="figure4_tdsc_decision_curve", help="Output filename stem."
    )
    return parser.parse_args()


def validate(curves: pd.DataFrame, contrasts: pd.DataFrame) -> None:
    curve_columns = {
        "strategy",
        "threshold",
        "observed_mean_net_benefit",
        "ci95_lower",
        "ci95_upper",
        "bootstrap_replicates",
    }
    contrast_columns = {
        "contrast",
        "candidate",
        "reference",
        "threshold",
        "observed_delta_net_benefit",
        "ci95_lower",
        "ci95_upper",
        "bootstrap_replicates",
        "interval_scope",
    }
    if missing := curve_columns.difference(curves.columns):
        raise ValueError(f"Curve CSV is missing columns: {sorted(missing)}")
    if missing := contrast_columns.difference(contrasts.columns):
        raise ValueError(f"Contrast CSV is missing columns: {sorted(missing)}")

    found = set(curves["strategy"])
    expected = set(STRATEGIES)
    if found != expected:
        raise ValueError(f"Expected strategies {sorted(expected)}, found {sorted(found)}")
    thresholds = np.sort(curves["threshold"].unique())
    if not (
        np.isclose(thresholds[0], 0.05)
        and np.isclose(thresholds[-1], 0.80)
        and np.allclose(np.diff(thresholds), 0.01)
    ):
        raise ValueError("Decision curves must span 0.05--0.80 in 0.01 steps")
    if not set(curves["bootstrap_replicates"].unique()) == {10_000}:
        raise ValueError("Expected 10,000 bootstrap replicates")
    if not set(contrasts["bootstrap_replicates"].unique()) == {10_000}:
        raise ValueError("Expected 10,000 bootstrap replicates for contrasts")

    needed = {"equal_logit_minus_r3d", "equal_logit_minus_triplanar"}
    if not needed.issubset(set(contrasts["contrast"])):
        raise ValueError("Required fusion-versus-expert contrasts are missing")


def select_key_thresholds(frame: pd.DataFrame, contrast: str) -> pd.DataFrame:
    rows = frame[frame["contrast"].eq(contrast)].copy()
    rows = rows[
        rows["threshold"].map(
            lambda value: bool(np.any(np.isclose(value, KEY_THRESHOLDS)))
        )
    ].sort_values("threshold")
    if len(rows) != len(KEY_THRESHOLDS) or not np.allclose(
        rows["threshold"], KEY_THRESHOLDS
    ):
        raise ValueError(f"Missing key thresholds for {contrast}")
    return rows


def configure_matplotlib() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8.0,
            "axes.labelsize": 8.5,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def make_figure(curves: pd.DataFrame, contrasts: pd.DataFrame):
    configure_matplotlib()
    import matplotlib.pyplot as plt

    figure, (axis_curve, axis_delta) = plt.subplots(
        1,
        2,
        figsize=(7.25, 3.15),
        gridspec_kw={"width_ratios": [1.24, 1.0], "wspace": 0.32},
    )

    for strategy, style in STRATEGIES.items():
        rows = curves[curves["strategy"].eq(strategy)].sort_values("threshold")
        axis_curve.plot(
            rows["threshold"],
            rows["observed_mean_net_benefit"],
            label=style["label"],
            color=style["color"],
            linewidth=style["linewidth"],
            linestyle=style["linestyle"],
            zorder=style["zorder"],
        )
    axis_curve.set_xlim(0.05, 0.80)
    axis_curve.set_ylim(-0.10, 0.62)
    axis_curve.set_xticks(np.arange(0.1, 0.81, 0.1))
    axis_curve.set_yticks(np.arange(-0.1, 0.61, 0.1))
    axis_curve.set_xlabel("Hypothetical escalation threshold")
    axis_curve.set_ylabel("Net benefit")
    axis_curve.grid(color="#D9D9D9", linewidth=0.55, alpha=0.75)
    axis_curve.set_axisbelow(True)
    axis_curve.legend(
        loc="lower left",
        ncol=2,
        frameon=False,
        handlelength=2.4,
        columnspacing=1.1,
        borderaxespad=0.25,
    )
    axis_curve.text(
        0.03,
        0.96,
        "A",
        transform=axis_curve.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
    )
    axis_curve.text(
        0.79,
        -0.087,
        "Escalate-all continues below the axis",
        ha="right",
        va="bottom",
        fontsize=6.4,
        color="#666666",
    )

    r3d = select_key_thresholds(contrasts, "equal_logit_minus_r3d")
    trimil = select_key_thresholds(contrasts, "equal_logit_minus_triplanar")
    y = np.arange(len(KEY_THRESHOLDS), dtype=float)
    offset = 0.11

    def add_forest(rows: pd.DataFrame, y_values: np.ndarray, label: str, color: str, marker: str):
        estimate = rows["observed_delta_net_benefit"].to_numpy(float)
        lower = rows["ci95_lower"].to_numpy(float)
        upper = rows["ci95_upper"].to_numpy(float)
        axis_delta.errorbar(
            estimate,
            y_values,
            xerr=np.vstack((estimate - lower, upper - estimate)),
            fmt=marker,
            markersize=4.2,
            markerfacecolor="white",
            markeredgewidth=1.1,
            color=color,
            ecolor=color,
            elinewidth=1.0,
            capsize=2.1,
            label=label,
            zorder=3,
        )

    add_forest(r3d, y + offset, "vs R3D-18", "#0072B2", "s")
    add_forest(trimil, y - offset, "vs TriMIL", "#D55E00", "o")
    axis_delta.axvline(0.0, color="#4D4D4D", linewidth=0.9, linestyle=":", zorder=1)
    axis_delta.set_yticks(y)
    axis_delta.set_yticklabels([f"{threshold:.1f}" for threshold in KEY_THRESHOLDS])
    axis_delta.set_ylim(-0.5, len(KEY_THRESHOLDS) - 0.5)
    axis_delta.set_xlim(-0.045, 0.12)
    axis_delta.set_xlabel("Net-benefit difference\n(R3D-TriMIL $-$ reference)")
    axis_delta.set_ylabel("Escalation threshold")
    axis_delta.grid(axis="x", color="#D9D9D9", linewidth=0.55, alpha=0.75)
    axis_delta.set_axisbelow(True)
    axis_delta.legend(loc="lower right", frameon=False, handletextpad=0.45)
    axis_delta.text(
        0.03,
        0.96,
        "B",
        transform=axis_delta.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
    )

    figure.subplots_adjust(left=0.075, right=0.992, bottom=0.22, top=0.98)
    return figure


def main() -> None:
    args = parse_args()
    curves = pd.read_csv(args.curve_csv)
    contrasts = pd.read_csv(args.contrast_csv)
    validate(curves, contrasts)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure = make_figure(curves, contrasts)
    outputs = {
        "pdf": args.output_dir / f"{args.stem}.pdf",
        "png": args.output_dir / f"{args.stem}.png",
        "svg": args.output_dir / f"{args.stem}.svg",
    }
    figure.savefig(outputs["pdf"], bbox_inches="tight")
    figure.savefig(outputs["png"], dpi=600, bbox_inches="tight")
    figure.savefig(outputs["svg"], bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(figure)

    manifest = {
        "figure": "TDSC exploratory decision-curve analysis",
        "panels": {
            "A": "mean net benefit over the complete threshold range 0.05--0.80",
            "B": "fusion-minus-expert net-benefit differences with pointwise 95% bootstrap confidence intervals at selected illustrative thresholds",
        },
        "bootstrap": "10,000 hierarchical replicates resampling split seeds and unique cases",
        "interval_scope": "pointwise; not simultaneous and not multiplicity-adjusted",
        "inputs": {
            str(args.curve_csv): sha256(args.curve_csv),
            str(args.contrast_csv): sha256(args.contrast_csv),
        },
        "outputs": {name: str(path) for name, path in outputs.items()},
    }
    manifest_path = args.output_dir / f"{args.stem}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
