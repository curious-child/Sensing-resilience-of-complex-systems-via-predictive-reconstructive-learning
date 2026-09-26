"""Plot manuscript Figure 5 from cached PRISM embeddings and t-SNE projections."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiment_process.generate_figure5_tsne_cache import (  # noqa: E402
    TASKS,
    ensure_figure5_caches,
    validate_cache,
)


OUTPUT_DIR = PROJECT_ROOT / "experiment_results" / "figures"
SIS_LABELS = {0: "Non-resilient", 1: "Resilient"}
NEURONAL_LABELS = {
    0: "Non-resilient (unactivated state)",
    1: "Non-resilient (alternative states)",
    2: "Resilient (activated state)",
}
COLORS = {0: "#969696", 1: "#3569A8", 2: "#C43E4F"}
SIS_COLORS = {0: COLORS[1], 1: COLORS[2]}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 6.0,
            "axes.titlesize": 7.0,
            "axes.labelsize": 6.2,
            "axes.linewidth": 0.65,
            "xtick.labelsize": 5.2,
            "ytick.labelsize": 5.2,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "legend.fontsize": 5.2,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def _load(task_key: str) -> tuple[dict[str, np.ndarray], dict]:
    task = TASKS[task_key]
    validate_cache(task, require_current=False)
    with np.load(task.cache_path, allow_pickle=False) as cached:
        arrays = {key: cached[key].copy() for key in cached.files}
    with task.metadata_path.open("r", encoding="utf-8") as stream:
        metadata = json.load(stream)
    return arrays, metadata


def _scatter_order(labels: np.ndarray, seed: int = 42) -> np.ndarray:
    order = np.arange(len(labels))
    np.random.default_rng(seed).shuffle(order)
    return order


def _style_2d(axis) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(direction="out", pad=1.5)
    axis.grid(False)
    axis.set_xlabel("t-SNE dimension 1", labelpad=2)
    axis.set_ylabel("t-SNE dimension 2", labelpad=2)
    axis.margins(0.04)


def _draw_2d(
    axis,
    projection: np.ndarray,
    labels: np.ndarray,
    label_names: dict[int, str],
    colors: dict[int, str],
) -> None:
    order = _scatter_order(labels)
    for label, name in label_names.items():
        selected = order[labels[order] == label]
        axis.scatter(
            projection[selected, 0],
            projection[selected, 1],
            s=7.0,
            color=colors[label],
            alpha=0.62,
            edgecolors="white",
            linewidths=0.18,
            rasterized=True,
            label=name,
        )
    _style_2d(axis)


def _style_3d(axis) -> None:
    axis.set_xlabel("t-SNE dimension 1", labelpad=-1)
    axis.set_ylabel("t-SNE dimension 2", labelpad=-1)
    axis.set_zlabel("t-SNE dimension 3", labelpad=-1)
    axis.tick_params(axis="both", which="major", labelsize=4.5, pad=-1)
    axis.grid(True, alpha=0.22, linewidth=0.4)
    axis.set_box_aspect((1.0, 1.0, 0.85))
    axis.set_proj_type("ortho")
    for pane_axis in (axis.xaxis, axis.yaxis, axis.zaxis):
        pane_axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        pane_axis.pane.set_edgecolor("#B8B8B8")
        pane_axis._axinfo["grid"].update(color=(0.75, 0.75, 0.75, 0.45), linewidth=0.4)


def _draw_3d(
    axis,
    projection: np.ndarray,
    labels: np.ndarray,
    *,
    emphasis: str,
) -> None:
    order = _scatter_order(labels)
    if emphasis == "red_blue":
        draw_order = (0, 1, 2)
        alpha = {0: 0.25, 1: 0.66, 2: 0.66}
    elif emphasis == "grey":
        draw_order = (1, 2, 0)
        alpha = {0: 0.67, 1: 0.36, 2: 0.36}
    else:
        raise ValueError(f"Unknown 3D emphasis: {emphasis}")
    for label in draw_order:
        selected = order[labels[order] == label]
        axis.scatter(
            projection[selected, 0],
            projection[selected, 1],
            projection[selected, 2],
            s=5.0,
            color=COLORS[label],
            alpha=alpha[label],
            edgecolors="none",
            depthshade=False,
            rasterized=True,
        )
    _style_3d(axis)


def _panel_label(axis, label: str, *, x: float = -0.12, y: float = 1.04) -> None:
    text_method = axis.text2D if hasattr(axis, "text2D") else axis.text
    text_method(
        x,
        y,
        label,
        transform=axis.transAxes,
        fontsize=8,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def create_figure() -> mpl.figure.Figure:
    configure_style()
    sis, sis_metadata = _load("sis_binary")
    neuronal, neuronal_metadata = _load("neuronal_three_class")

    figure = plt.figure(figsize=(7.2, 5.80), constrained_layout=False)
    layout = figure.add_gridspec(
        3,
        2,
        left=0.065,
        right=0.975,
        bottom=0.055,
        top=0.99,
        width_ratios=(1.0, 1.0),
        height_ratios=(0.10, 0.88, 1.05),
        wspace=0.18,
        hspace=0.18,
    )
    legend_axis = figure.add_subplot(layout[0, :])
    legend_axis.set_axis_off()
    axis_a = figure.add_subplot(layout[1, 0])
    axis_b = figure.add_subplot(layout[1, 1])
    axis_c = figure.add_subplot(layout[2, 0], projection="3d")
    axis_d = figure.add_subplot(layout[2, 1], projection="3d")

    _draw_2d(axis_a, sis["tsne_2d"], sis["labels"], SIS_LABELS, SIS_COLORS)
    _draw_2d(
        axis_b,
        neuronal["tsne_2d"],
        neuronal["labels"],
        NEURONAL_LABELS,
        COLORS,
    )
    _draw_3d(
        axis_c,
        neuronal["tsne_3d"],
        neuronal["labels"],
        emphasis="red_blue",
    )
    _draw_3d(
        axis_d,
        neuronal["tsne_3d"],
        neuronal["labels"],
        emphasis="grey",
    )

    axis_a.set_title("SIS dynamics", fontweight="bold", pad=4)
    axis_b.set_title("Neuronal dynamics (2D)", fontweight="bold", pad=4)
    axis_c.text2D(
        0.5,
        0.955,
        "Neuronal dynamics (3D, lateral view)",
        transform=axis_c.transAxes,
        ha="center",
        va="top",
        fontsize=7.0,
        fontweight="bold",
    )
    axis_d.text2D(
        0.5,
        0.955,
        "Neuronal dynamics (3D, vertical view)",
        transform=axis_d.transAxes,
        ha="center",
        va="top",
        fontsize=7.0,
        fontweight="bold",
    )
    # The two views use the same cached coordinates.  The lateral view was
    # selected to preserve the red-blue division in the screen plane, whereas
    # the vertical view exposes the grey lower domain relative to both upper
    # state families.
    axis_c.view_init(elev=10, azim=255)
    axis_d.view_init(elev=17, azim=142)

    sis_handles = [
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker="o",
            markersize=3.8,
            markerfacecolor=SIS_COLORS[label],
            markeredgecolor="none",
            alpha=0.75,
            label=name,
        )
        for label, name in SIS_LABELS.items()
    ]
    neuronal_handles = [
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker="o",
            markersize=3.8,
            markerfacecolor=COLORS[label],
            markeredgecolor="none",
            alpha=0.75,
            label={0: "Unactivated", 1: "Alternative", 2: "Activated"}[label],
        )
        for label, name in NEURONAL_LABELS.items()
    ]
    sis_legend = legend_axis.legend(
        handles=sis_handles,
        title="SIS",
        title_fontsize=5.5,
        loc="center",
        bbox_to_anchor=(0.23, 0.50),
        ncol=2,
        handletextpad=0.35,
        columnspacing=1.0,
        fontsize=5.2,
    )
    legend_axis.add_artist(sis_legend)
    legend_axis.legend(
        handles=neuronal_handles,
        title="Neuronal",
        title_fontsize=5.5,
        loc="center",
        bbox_to_anchor=(0.73, 0.50),
        ncol=3,
        handletextpad=0.35,
        columnspacing=1.0,
        fontsize=5.2,
    )

    for axis, label in zip((axis_a, axis_b, axis_c, axis_d), "abcd"):
        _panel_label(
            axis,
            label,
            x=-0.12 if label in "ab" else 0.005,
            y=1.025 if label in "ab" else 0.955,
        )

    figure._figure5_metadata = {  # type: ignore[attr-defined]
        "sis": sis_metadata,
        "neuronal": neuronal_metadata,
    }
    return figure


def save_figure(figure: mpl.figure.Figure) -> tuple[Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_DIR / "emb_visual.png"
    pdf_path = OUTPUT_DIR / "emb_visual.pdf"
    from experiment_process.generate_resilience_inference_cache import archive
    archive(png_path)
    archive(pdf_path)
    figure.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0.03)
    figure.savefig(pdf_path, bbox_inches="tight", pad_inches=0.03)
    return png_path, pdf_path


def main() -> None:
    global OUTPUT_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", help="used only if cache recovery is needed")
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    OUTPUT_DIR = args.output_dir
    ensure_figure5_caches(device=args.device, force=args.force_cache)
    figure = create_figure()
    png_path, pdf_path = save_figure(figure)
    plt.close(figure)
    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
