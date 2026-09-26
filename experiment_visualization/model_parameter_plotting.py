"""Shared Nature-style plotting helpers for SIS parameter sensitivity."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Mapping

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESS_ROOT = PROJECT_ROOT / "experiment_process"
if str(PROCESS_ROOT) not in sys.path:
    sys.path.insert(0, str(PROCESS_ROOT))

from generate_model_parameter_analysis_cache import (  # noqa: E402
    PARAMETER_GROUPS,
    PARAMETER_DISPLAY_NAMES,
    ensure_parameter_groups,
    group_records,
)


PALETTE = ("#B64342", "#0F4D92", "#2D7D46", "#7A3E9D")
NETWORK_CONDITIONS = (
    ("SF", "random"), ("SF", "degree"), ("ER", "random"), ("ER", "degree"),
)
STATE_COLOR = "#7884B4"
REFERENCE_COLOR = "#767676"
NETWORK_TITLES = {
    ("ER", "degree"): "ER\nTargeted perturbation",
    ("ER", "random"): "ER\nRandom perturbation",
    ("SF", "degree"): "SF\nTargeted perturbation",
    ("SF", "random"): "SF\nRandom perturbation",
}
METRICS = (
    ("Learn_error", "ts_err", "Pred. loss", True),
    ("Learn_error", "graph_err", "Recon. loss", True),
    ("Zlatent", "Z_var", "Distribution var.", False),
    ("Zlatent", "distance_var", "Node-dist. var.", False),
)


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 5.5,
            "axes.labelsize": 5.5,
            "axes.titlesize": 6.5,
            "axes.linewidth": 0.65,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 4.4,
            "ytick.labelsize": 4.4,
            "xtick.major.size": 2,
            "ytick.major.size": 2,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "legend.fontsize": 5.5,
            "legend.frameon": False,
            "lines.linewidth": 0.85,
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def _normalize(values) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.zeros_like(array)
    lower, upper = finite.min(), finite.max()
    if np.isclose(lower, upper):
        return np.zeros_like(array)
    return (array - lower) / (upper - lower)


def _tipping_point(record: dict) -> float:
    x = np.asarray(record["remove_nodes_scatter_record"], dtype=float)
    y = np.asarray(record["xlast_system_scatter_record"], dtype=float)
    indices = np.flatnonzero(y < 1e-2)
    return float(x[indices[0]]) if indices.size else float(x[np.argmin(y)])


def _style_axis(ax: mpl.axes.Axes) -> None:
    ax.tick_params(pad=1.5)
    ax.margins(x=0.015, y=0.08)
    ax.spines["left"].set_color("#4D4D4D")
    ax.spines["bottom"].set_color("#4D4D4D")


def plot_parameter_panel(
    figure: mpl.figure.Figure,
    outer_spec,
    records: Mapping[str, dict],
    *,
    title: str,
    show_ylabels: bool,
    show_xlabel: bool,
    show_title: bool,
) -> list[mpl.axes.Axes]:
    inner = outer_spec.subgridspec(5, 1, hspace=0.04)
    axes = [figure.add_subplot(inner[index, 0]) for index in range(5)]
    first_record = next(iter(records.values()))
    tipping = _tipping_point(first_record)

    axes[0].scatter(
        first_record["remove_nodes_scatter_record"],
        first_record["xlast_system_scatter_record"],
        s=7,
        color=STATE_COLOR,
        alpha=0.7,
        edgecolors="black",
        linewidths=0.15,
        rasterized=True,
    )
    if show_title:
        axes[0].set_title(title, fontsize=6.5, pad=4, fontweight="bold")
    if show_ylabels:
        axes[0].set_ylabel(r"$\langle X\rangle$", fontsize=5.5, labelpad=2)
    _style_axis(axes[0])

    for metric_index, (section, key, ylabel, normalize) in enumerate(
        METRICS, start=1
    ):
        axis = axes[metric_index]
        for curve_index, (label, record) in enumerate(records.items()):
            values = record["perform_record"][section][key]
            if normalize:
                values = _normalize(values)
            x = np.asarray(record["remove_nodes_record"], dtype=float)
            axis.plot(
                x,
                values,
                color=PALETTE[curve_index],
                linewidth=0.85,
                solid_capstyle="round",
                alpha=0.96,
                label=label,
                zorder=3 + curve_index,
            )
        if show_ylabels:
            axis.set_ylabel(ylabel)
        _style_axis(axis)
    for axis in axes:
        axis.axvline(
            tipping,
            color=REFERENCE_COLOR,
            linestyle=(0, (3, 2)),
            linewidth=0.65,
            alpha=0.85,
            zorder=2,
        )
    for axis in axes[:-1]:
        axis.tick_params(labelbottom=False)
    if show_xlabel:
        axes[-1].set_xlabel("Removed nodes", fontsize=5.2, labelpad=2)
    else:
        axes[-1].tick_params(labelbottom=False)
    return axes


def create_parameter_figure(
    top_group: str,
    bottom_group: str,
    *,
    top_label: str | None = None,
    bottom_label: str | None = None,
    panel_letters: str = "abcdefgh",
    device: str = "cuda",
    force_cache: bool = False,
    force_train: bool = False,
) -> mpl.figure.Figure:
    configure_style()
    ensure_parameter_groups(
        (top_group, bottom_group),
        device=device,
        force_cache=force_cache,
        force_train=force_train,
    )
    figure = plt.figure(figsize=(7.2, 7.0), constrained_layout=False)
    layout = figure.add_gridspec(
        4,
        1,
        left=0.09,
        right=0.925,
        bottom=0.055,
        top=0.985,
        height_ratios=(0.10, 1.0, 0.10, 1.0),
        hspace=0.10,
    )
    groups = (
        (
            top_group,
            top_label or PARAMETER_DISPLAY_NAMES[top_group],
        ),
        (
            bottom_group,
            bottom_label or PARAMETER_DISPLAY_NAMES[bottom_group],
        ),
    )
    for row, (group, row_label) in enumerate(groups):
        legend_labels = [
            f"{row_label} = {label}"
            for label, _ in PARAMETER_GROUPS[group]
        ]
        legend_handles = [
            Line2D([0], [0], color=PALETTE[index], linewidth=1.2)
            for index in range(len(legend_labels))
        ]
        legend_axis = figure.add_subplot(layout[row * 2, 0])
        legend_axis.set_axis_off()
        legend_axis.legend(
            legend_handles,
            legend_labels,
            loc="center",
            ncol=len(legend_labels),
            handlelength=1.8,
            columnspacing=1.5,
            fontsize=5.5,
        )
        panel_row = layout[row * 2 + 1, 0].subgridspec(
            1, 4, wspace=0.62
        )
        for column, condition in enumerate(NETWORK_CONDITIONS):
            records = group_records(
                group,
                *condition,
                device=device,
                force_cache=False,
                force_train=False,
            )
            axes = plot_parameter_panel(
                figure,
                panel_row[0, column],
                records,
                title=NETWORK_TITLES[condition],
                show_ylabels=column == 0,
                show_xlabel=row == 1,
                show_title=True,
            )
            axes[0].text(
                -0.29,
                1.14,
                panel_letters[row * 4 + column],
                transform=axes[0].transAxes,
                fontsize=8,
                fontweight="bold",
                va="top",
                ha="left",
            )
    return figure


def save_figure(figure: mpl.figure.Figure, stem: str) -> tuple[Path, Path]:
    output = OUTPUT_DIR
    output.mkdir(parents=True, exist_ok=True)
    png = output / f"{stem}.png"
    pdf = output / f"{stem}.pdf"
    from experiment_process.generate_resilience_inference_cache import archive
    archive(png)
    archive(pdf)
    figure.savefig(png, dpi=600, bbox_inches="tight", pad_inches=0.03)
    figure.savefig(pdf, bbox_inches="tight", pad_inches=0.03)
    return png, pdf


OUTPUT_DIR = PROJECT_ROOT / "experiment_results" / "figures"
