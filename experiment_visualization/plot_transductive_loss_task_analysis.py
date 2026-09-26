"""Reproduce Supplementary Figure 1 from transductive loss caches.

If a required cache is absent, the original per-graph optimization procedure
is called for only that dynamics/network/removal/task combination.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEVICE = "cuda"
FORCE_REGENERATE = False
DYNAMICS = ("SIS", "Neuronal")
SCENARIOS = (
    ("SF", "random"),
    ("SF", "degree"),
    ("ER", "random"),
    ("ER", "degree"),
)
TASKS = ("multitask", "prediction", "reconstruction", "untrained")
TASK_LABELS = {
    "multitask": "Multitask",
    "prediction": "Prediction only",
    "reconstruction": "Reconstruction only",
    "untrained": "Untrained",
}
PANEL_LABELS = tuple("abcdefgh")
OUTPUT_DIR = PROJECT_ROOT / "experiment_results" / "figures"
PREDICTION_COLOR = "#B64342"
RECONSTRUCTION_COLOR = "#0F4D92"
STATE_COLOR = "#7884B4"
REFERENCE_COLOR = "#767676"

matplotlib.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "pdf.fonttype": 42,
        "font.size": 5.5,
        "axes.linewidth": 0.65,
        "axes.spines.top": False,
        "legend.frameon": False,
        "xtick.major.width": 0.55,
        "ytick.major.width": 0.55,
        "xtick.major.size": 2,
        "ytick.major.size": 2,
    }
)


def cache_path(dynamic: str, task: str, net_struct: str, remove_type: str) -> Path:
    return (
        PROJECT_ROOT
        / "experiment_results"
        / dynamic
        / "loss_transductive_analysis"
        / f"{net_struct}_{remove_type}_record_{task}.json"
    )


def required_requests() -> list[tuple[str, str, str, str]]:
    return [
        (dynamic, task, net_struct, remove_type)
        for dynamic in DYNAMICS
        for net_struct, remove_type in SCENARIOS
        for task in TASKS
    ]


def ensure_required_caches() -> None:
    requests = required_requests()
    missing = [request for request in requests if not cache_path(*request).is_file()]
    if requests:
        from experiment_process.generate_transductive_analysis_cache import (
            ensure_transductive_caches,
        )

        ensure_transductive_caches(
            requests,
            force=FORCE_REGENERATE,
            device=DEVICE,
        )


def load_cache(path: Path, task: str) -> dict[str, Any]:
    from experiment_process.generate_transductive_analysis_cache import (
        validate_transductive_record,
    )

    with path.open("r", encoding="utf-8") as stream:
        record = json.load(stream)
    validate_transductive_record(
        record,
        str(path),
        allow_legacy_untrained_iterations=task == "untrained",
    )
    return record


def tipping_point(
    time: list[float] | list[int], state: list[float], dynamic: str
) -> float:
    time_values = np.asarray(time)
    state_values = np.asarray(state)
    window_size = 10
    if dynamic == "Neuronal":
        if len(state_values) <= window_size:
            raise ValueError("Neuronal stable-state sequence is too short")
        change_index = int(
            np.argmax(
                np.abs(
                    state_values[window_size:] - state_values[:-window_size]
                )
                / window_size
            )
        )
    elif dynamic == "SIS":
        candidates = np.flatnonzero(state_values < 1e-2)
        if not len(candidates):
            raise ValueError("SIS stable state never falls below 1e-2")
        change_index = int(candidates[0])
    else:
        raise ValueError(f"Unsupported dynamics: {dynamic}")
    return float(time_values[change_index])


def _same_coordinates(
    reference: dict[str, Any], candidate: dict[str, Any], context: str
) -> None:
    for key in (
        "remove_nodes_record",
        "remove_nodes_scatter_record",
        "xlast_system_scatter_record",
    ):
        if reference[key] != candidate[key]:
            raise ValueError(f"{context}: inconsistent {key} across task caches")


def draw_panel(
    fig: plt.Figure,
    outer_cell: Any,
    records: dict[str, dict[str, Any]],
    dynamic: str,
    net_struct: str,
    remove_type: str,
    panel_label: str,
    column_index: int,
) -> None:
    grid = outer_cell.subgridspec(5, 1, hspace=0.04)
    axes: list[plt.Axes] = []
    for row in range(5):
        axes.append(fig.add_subplot(grid[row, 0], sharex=axes[0] if axes else None))

    reference = records["multitask"]
    for task in TASKS[1:]:
        _same_coordinates(reference, records[task], f"{dynamic}/{net_struct}/{remove_type}")
    line_x = reference["remove_nodes_record"]
    scatter_x = reference["remove_nodes_scatter_record"]
    scatter_y = reference["xlast_system_scatter_record"]
    critical_x = tipping_point(scatter_x, scatter_y, dynamic)

    axes[0].scatter(
        scatter_x,
        scatter_y,
        s=7,
        alpha=0.7,
        color=STATE_COLOR,
        edgecolors="black",
        linewidths=0.15,
    )
    if column_index == 0:
        axes[0].set_ylabel(r"$\langle X\rangle$", fontsize=5.5, labelpad=2)
    topology = "SF" if net_struct == "SF" else "ER"
    attack = (
        "Targeted perturbation"
        if remove_type == "degree"
        else "Random perturbation"
    )
    if dynamic == "SIS":
        axes[0].set_title(
            f"{topology}\n{attack}", fontsize=6.5, fontweight="bold", pad=4
        )
    axes[0].text(
        -0.29,
        1.20,
        panel_label,
        transform=axes[0].transAxes,
        fontsize=8,
        fontweight="bold",
        va="top",
    )

    for row, task in enumerate(TASKS, start=1):
        performance = records[task]["perform_record"]
        left = axes[row]
        right = left.twinx()
        prediction_line = left.plot(
            line_x,
            performance["ts_err"],
            color=PREDICTION_COLOR,
            linewidth=0.85,
            solid_capstyle="round",
            alpha=0.96,
            zorder=3,
            label="Prediction loss",
        )[0]
        reconstruction_line = right.plot(
            line_x,
            performance["graph_err"],
            color=RECONSTRUCTION_COLOR,
            linewidth=0.85,
            solid_capstyle="round",
            alpha=0.96,
            zorder=3,
            label="Reconstruction loss",
        )[0]
        if column_index == 0:
            left.set_ylabel(
                TASK_LABELS[task],
                color=PREDICTION_COLOR,
                fontsize=4.8,
                labelpad=1.5,
            )
        left.tick_params(axis="y", labelcolor=PREDICTION_COLOR)
        right.tick_params(axis="y", labelcolor=RECONSTRUCTION_COLOR)
        right.spines["right"].set_color(RECONSTRUCTION_COLOR)
        right.spines["right"].set_linewidth(0.65)
        for curve_axis in (left, right):
            curve_axis.relim()
            curve_axis.autoscale_view(scalex=False, scaley=True)
            curve_axis.margins(y=0.08)
            curve_axis.yaxis.set_major_locator(MaxNLocator(nbins=3))

    for row, axis in enumerate(axes):
        axis.axvline(
            critical_x,
            color=REFERENCE_COLOR,
            linestyle="--",
            linewidth=0.65,
            alpha=0.85,
        )
        axis.tick_params(axis="both", labelsize=4.4, pad=1.5)
        axis.yaxis.set_major_locator(MaxNLocator(nbins=3))
        axis.spines["left"].set_color("#4D4D4D")
        axis.spines["bottom"].set_color("#4D4D4D")
        if row < 4:
            axis.tick_params(axis="x", labelbottom=False)
    axes[-1].set_xlabel("Removed nodes", fontsize=5.2, labelpad=2)


def main() -> None:
    ensure_required_caches()
    fig = plt.figure(figsize=(7.2, 6.7))
    master = fig.add_gridspec(
        3,
        1,
        left=0.075,
        right=0.925,
        bottom=0.055,
        top=0.985,
        height_ratios=(0.055, 1.0, 1.0),
        hspace=0.11,
    )
    legend_handles = [
        Line2D([0], [0], color=PREDICTION_COLOR, lw=1.2),
        Line2D([0], [0], color=RECONSTRUCTION_COLOR, lw=1.2),
        Line2D([0], [0], color=REFERENCE_COLOR, lw=0.8, ls="--"),
    ]
    legend_axis = fig.add_subplot(master[0, 0])
    legend_axis.set_axis_off()
    legend_axis.legend(
        legend_handles,
        ["Prediction loss", "Reconstruction loss", "Critical point"],
        loc="center",
        ncol=3,
        fontsize=5.5,
        handlelength=1.8,
        columnspacing=1.6,
    )
    panel_rows = (
        master[1, 0].subgridspec(1, 4, wspace=0.62),
        master[2, 0].subgridspec(1, 4, wspace=0.62),
    )
    fig.text(-0.01, 0.705, "SIS", rotation=90, fontsize=7, fontweight="bold")
    fig.text(
        -0.01, 0.255, "Neuronal", rotation=90, fontsize=7, fontweight="bold"
    )
    panel_index = 0
    for row, dynamic in enumerate(DYNAMICS):
        for column, (net_struct, remove_type) in enumerate(SCENARIOS):
            records = {
                task: load_cache(
                    cache_path(dynamic, task, net_struct, remove_type), task
                )
                for task in TASKS
            }
            draw_panel(
                fig,
                panel_rows[row][0, column],
                records,
                dynamic,
                net_struct,
                remove_type,
                PANEL_LABELS[panel_index],
                column,
            )
            panel_index += 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_DIR / "loss_task_analysis_transductive.png"
    pdf_path = OUTPUT_DIR / "loss_task_analysis_transductive.pdf"
    from experiment_process.generate_resilience_inference_cache import archive
    archive(png_path)
    archive(pdf_path)
    fig.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    DEVICE, FORCE_REGENERATE, OUTPUT_DIR = args.device, args.force_cache, args.output_dir
    main()
