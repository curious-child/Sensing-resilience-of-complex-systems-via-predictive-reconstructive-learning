"""Reproduce main Figure 3 from trained multitask representation caches.

The script reads latent-representation metrics already stored in the legacy
inductive JSON files. A missing multitask cache is generated automatically.
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
PANEL_LABELS = tuple("abcdefgh")
OUTPUT_DIR = PROJECT_ROOT / "experiment_results" / "figures"
PRIMARY_COLOR = "#B64342"
SECONDARY_COLOR = "#0F4D92"
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


def cache_path(dynamic: str, net_struct: str, remove_type: str) -> Path:
    return (
        PROJECT_ROOT
        / "experiment_results"
        / dynamic
        / "loss_inductive_analysis"
        / f"multitask_{net_struct}_{remove_type}_record_train_True.json"
    )


def required_requests() -> list[tuple[str, str, str, str]]:
    return [
        (dynamic, "multitask", net_struct, remove_type)
        for dynamic in DYNAMICS
        for net_struct, remove_type in SCENARIOS
    ]


def ensure_required_caches() -> None:
    requests = required_requests()
    missing = [
        request
        for request in requests
        if not cache_path(request[0], request[2], request[3]).is_file()
    ]
    if requests:
        from experiment_process.generate_inductive_analysis_cache import (
            ensure_inductive_caches,
        )

        ensure_inductive_caches(
            requests,
            force=FORCE_REGENERATE,
            device=DEVICE,
        )


def load_cache(path: Path) -> dict[str, Any]:
    from experiment_process.generate_inductive_analysis_cache import (
        validate_inductive_record,
    )

    with path.open("r", encoding="utf-8") as stream:
        record = json.load(stream)
    validate_inductive_record(
        record, str(path), allow_legacy_missing_distance_g_var=True
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


def rolling_slope(data: list[float], window_size: int) -> np.ndarray:
    """Copy the rolling least-squares slope used in the original Figure 3."""

    values = np.asarray(data)
    count = len(values)
    if window_size > count or window_size < 2:
        raise ValueError("rolling window must be between 2 and data length")
    x = np.arange(window_size)
    sum_x = np.sum(x)
    sum_x2 = np.sum(x**2)
    denominator = window_size * sum_x2 - sum_x**2
    windows = np.lib.stride_tricks.sliding_window_view(values, window_size)
    sum_y = np.sum(windows, axis=1)
    sum_xy = np.dot(windows, x)
    slopes = (window_size * sum_xy - sum_x * sum_y) / denominator
    result = np.full(count, np.nan)
    result[window_size - 1 :] = slopes
    return result


def _paired_lines(
    left: plt.Axes,
    x: list[float] | list[int],
    first: list[float] | np.ndarray,
    second: list[float] | np.ndarray,
    first_label: str,
    second_label: str,
    show_left_label: bool,
    show_right_label: bool,
) -> plt.Axes:
    right = left.twinx()
    line1 = left.plot(
        x,
        first,
        color=PRIMARY_COLOR,
        linewidth=0.85,
        solid_capstyle="round",
        alpha=0.96,
        zorder=3,
        label=first_label,
    )[0]
    line2 = right.plot(
        x,
        second,
        color=SECONDARY_COLOR,
        linewidth=0.85,
        solid_capstyle="round",
        alpha=0.96,
        zorder=3,
        label=second_label,
    )[0]
    left.tick_params(axis="y", labelcolor=PRIMARY_COLOR)
    right.tick_params(axis="y", labelcolor=SECONDARY_COLOR)
    right.spines["right"].set_color(SECONDARY_COLOR)
    right.spines["right"].set_linewidth(0.65)
    # Independent autoscaling keeps one metric from compressing the other.
    # The raw sequences are shown without smoothing or normalization.
    for curve_axis in (left, right):
        curve_axis.relim()
        curve_axis.autoscale_view(scalex=False, scaley=True)
        curve_axis.margins(y=0.08)
        curve_axis.yaxis.set_major_locator(MaxNLocator(nbins=3))
    if show_left_label:
        left.set_ylabel(
            first_label, color=PRIMARY_COLOR, fontsize=4.8, labelpad=1.5
        )
    if show_right_label:
        right.set_ylabel(
            second_label, color=SECONDARY_COLOR, fontsize=4.8, labelpad=1.5
        )
    return right


def draw_panel(
    fig: plt.Figure,
    outer_cell: Any,
    record: dict[str, Any],
    dynamic: str,
    net_struct: str,
    remove_type: str,
    panel_label: str,
    column_index: int,
) -> None:
    grid = outer_cell.subgridspec(4, 1, hspace=0.04)
    axes: list[plt.Axes] = []
    for row in range(4):
        axes.append(fig.add_subplot(grid[row, 0], sharex=axes[0] if axes else None))

    line_x = record["remove_nodes_record"]
    scatter_x = record["remove_nodes_scatter_record"]
    scatter_y = record["xlast_system_scatter_record"]
    latent = record["perform_record"]["Zlatent"]
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

    _paired_lines(
        axes[1],
        line_x,
        latent["Z_mean"],
        latent["Z_var"],
        "Distribution mean",
        "Distribution var.",
        column_index == 0,
        column_index == 3,
    )
    _paired_lines(
        axes[2],
        line_x,
        latent["distance_mean"],
        latent["distance_var"],
        "Node-dist. mean",
        "Node-dist. var.",
        column_index == 0,
        column_index == 3,
    )
    window_size = max(10, int(0.05 * len(latent["distance_G"])))
    slope = rolling_slope(latent["distance_G"], window_size)
    _paired_lines(
        axes[3],
        line_x,
        latent["distance_G"],
        slope,
        "Graph distance",
        "Distance slope",
        column_index == 0,
        column_index == 3,
    )

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
        if row < 3:
            axis.tick_params(axis="x", labelbottom=False)
    axes[-1].set_xlabel("Removed nodes", fontsize=5.2, labelpad=2)


def main() -> None:
    ensure_required_caches()
    fig = plt.figure(figsize=(7.2, 5.45))
    outer = fig.add_gridspec(
        2,
        4,
        left=0.10,
        right=0.90,
        bottom=0.065,
        top=0.94,
        wspace=0.72,
        hspace=0.14,
    )
    fig.text(0.018, 0.705, "SIS", rotation=90, fontsize=7, fontweight="bold")
    fig.text(
        0.018, 0.255, "Neuronal", rotation=90, fontsize=7, fontweight="bold"
    )
    panel_index = 0
    for row, dynamic in enumerate(DYNAMICS):
        for column, (net_struct, remove_type) in enumerate(SCENARIOS):
            record = load_cache(cache_path(dynamic, net_struct, remove_type))
            draw_panel(
                fig,
                outer[row, column],
                record,
                dynamic,
                net_struct,
                remove_type,
                PANEL_LABELS[panel_index],
                column,
            )
            panel_index += 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_DIR / "latent_warning.png"
    pdf_path = OUTPUT_DIR / "latent_warning.pdf"
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
