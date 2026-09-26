"""Shared plotting primitives for resilience-inference manuscript figures."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Mapping

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESS_DIR = PROJECT_ROOT / "experiment_process"
FIGURE_DIR = PROJECT_ROOT / "experiment_results" / "figures"
if str(PROCESS_DIR) not in sys.path:
    sys.path.insert(0, str(PROCESS_DIR))

from generate_resilience_inference_cache import ensure_caches, archive, selection


COLORS = {
    "mlp": "#C44E52",
    "knn": "#6A51A3",
    "gbb": "#6F6F6F",
    "resinf": "#D99B00",
    "positive": "#4C956C",
    "negative": "#C45B5B",
}
MARKERS = {"mlp": "o", "knn": "^", "gbb": "D", "resinf": "v"}
METHOD_ZORDER = {"gbb": 2, "resinf": 3, "knn": 5, "mlp": 6}
METHOD_LABELS = {
    "mlp": "MLP",
    "knn": "KNN",
    "gbb": "GBB",
    "resinf": "ResInf",
}
PERCENT_LABELS = ("0.2", "0.5", "1", "2", "5", "10", "20", "40", "80", "100")


def set_nature_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.5,
            "axes.labelsize": 8,
            "axes.titlesize": 8.5,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "legend.fontsize": 7,
            "legend.frameon": False,
            "lines.linewidth": 1.2,
            "lines.markersize": 3.3,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def load_task(task: str, methods=("mlp", "knn", "resinf"), *, ratios=None, trial_ids=None,
              device="cuda", force=False) -> dict[str, pd.DataFrame]:
    ratios, trials = selection(ratios,trial_ids)
    paths = ensure_caches(task, methods, ratios=ratios, trial_ids=trials, device=device, force=force)
    result={}
    for method,path in paths.items():
        frame=pd.read_csv(path)
        frame=frame[frame.data_ratio.isin(ratios)].reset_index(drop=True)
        historical = method in ("mlp", "resinf")
        frame.attrs.update(source=str(path), historical=historical,
                           trial_ids=None if historical else list(trials))
        result[method]=frame
    return result


def plot_options():
    global FIGURE_DIR
    import argparse
    parser=argparse.ArgumentParser(description="Cache first; missing requested MLP/ResInf weights are automatically trained.")
    parser.add_argument("--ratios",type=float,nargs="+")
    parser.add_argument("--trials",type=int,nargs="+")
    parser.add_argument("--device",default="cuda")
    parser.add_argument("--force", "--force-cache",action="store_true",help="Refresh inference, not existing weights")
    parser.add_argument("--output-dir",type=Path,default=FIGURE_DIR)
    args=parser.parse_args()
    FIGURE_DIR=args.output_dir
    return dict(ratios=args.ratios,trial_ids=args.trials,device=args.device,force=args.force)


def _bounds(frame: pd.DataFrame, stem: str) -> np.ndarray:
    mean = frame[f"{stem}_mean_f1"].to_numpy(float)
    lower = np.maximum(mean - frame[f"{stem}_min_f1"].to_numpy(float), 0)
    upper = np.maximum(frame[f"{stem}_max_f1"].to_numpy(float) - mean, 0)
    return np.vstack((lower, upper))


def _finish_ratio_axis(ax: mpl.axes.Axes, xlabel: bool = True) -> None:
    ax.set_xscale("log")
    ratios = np.asarray((0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.0))
    ax.set_xticks(ratios)
    ax.set_xticklabels(PERCENT_LABELS, rotation=55, ha="right", fontsize=6.5)
    if xlabel:
        ax.set_xlabel("Labeled training data (%)")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.45, alpha=0.65)
    ax.margins(x=0.025, y=0.1)


def plot_all_methods(
    ax: mpl.axes.Axes,
    frames: Mapping[str, pd.DataFrame],
    title: str,
    *,
    show_legend: bool = True,
) -> None:
    mlp, knn, resinf = frames["mlp"], frames["knn"], frames["resinf"]
    ratios = mlp["data_ratio"].to_numpy(float)
    series = (
        ("mlp", mlp, "frozen"),
        ("knn", knn, "knn"),
        ("resinf", resinf, "Resinf"),
    )
    for method, frame, stem in series:
        ax.errorbar(
            ratios,
            frame[f"{stem}_mean_f1"],
            yerr=_bounds(frame, stem),
            color=COLORS[method],
            marker=MARKERS[method],
            markersize=3.4,
            markeredgecolor="white",
            markeredgewidth=0.45,
            linewidth=1.2,
            alpha=0.95,
            capsize=2.8,
            capthick=0.9,
            elinewidth=0.9,
            barsabove=True,
            label=METHOD_LABELS[method],
            zorder=METHOD_ZORDER[method],
        )
    ax.plot(
        ratios,
        knn["baser_mean_f1"],
        color=COLORS["gbb"],
        marker=MARKERS["gbb"],
        markersize=3.2,
        markeredgecolor="white",
        markeredgewidth=0.35,
        linestyle=(0, (3, 2)),
        linewidth=1.1,
        alpha=0.82,
        label=METHOD_LABELS["gbb"],
        zorder=METHOD_ZORDER["gbb"],
    )
    ax.set_title(title, loc="left", fontweight="semibold", pad=5)
    ax.set_ylabel("F1-score")
    _finish_ratio_axis(ax)
    if show_legend:
        ax.legend(ncol=4, loc="lower right", handlelength=1.7, columnspacing=1.1)


def plot_knn_vs_gbb(
    ax: mpl.axes.Axes,
    frame: pd.DataFrame,
    title: str,
    *,
    show_legend: bool = True,
) -> None:
    ratios = frame["data_ratio"].to_numpy(float)
    ax.errorbar(
        ratios,
        frame["knn_mean_f1"],
        yerr=_bounds(frame, "knn"),
        color=COLORS["knn"],
        marker=MARKERS["knn"],
        markersize=3.4,
        markeredgecolor="white",
        markeredgewidth=0.45,
        linewidth=1.2,
        alpha=0.95,
        capsize=2.8,
        capthick=0.9,
        elinewidth=0.9,
        barsabove=True,
        label="KNN",
        zorder=METHOD_ZORDER["knn"],
    )
    ax.plot(
        ratios,
        frame["baser_mean_f1"],
        color=COLORS["gbb"],
        marker=MARKERS["gbb"],
        markersize=3.2,
        markeredgecolor="white",
        markeredgewidth=0.35,
        linestyle=(0, (3, 2)),
        linewidth=1.1,
        alpha=0.82,
        label="GBB",
        zorder=METHOD_ZORDER["gbb"],
    )
    ax.set_title(title, loc="left", fontweight="semibold", pad=5)
    ax.set_ylabel("F1-score")
    _finish_ratio_axis(ax)
    if show_legend:
        ax.legend(loc="lower right", ncol=2, handlelength=1.7)


def plot_improvement(
    ax: mpl.axes.Axes,
    frame: pd.DataFrame,
    title: str = "Performance difference (KNN - GBB)",
) -> None:
    ratios = frame["data_ratio"].to_numpy(float)
    changes = frame["improvement_mean"].to_numpy(float)
    colors = np.where(changes >= 0, COLORS["positive"], COLORS["negative"])
    positions = np.arange(len(frame))
    ax.bar(
        positions,
        changes,
        color=colors,
        width=0.7,
        alpha=0.88,
        edgecolor="white",
        linewidth=0.3,
    )
    ax.axhline(0, color="#555555", linewidth=0.7)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{r*100:g}" for r in ratios], rotation=38, ha="right")
    ax.set_xlabel("Labeled training data (%)")
    ax.set_ylabel(r"$\Delta$ F1-score")
    ax.set_title(title, loc="left", fontweight="semibold", pad=5)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.45, alpha=0.65)
    ax.margins(y=0.12)


def add_panel_label(
    ax: mpl.axes.Axes, label: str, *, x: float = -0.115, y: float = 1.045
) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="left",
    )


def save_figure(fig: mpl.figure.Figure, stem: str, *, output_dir=None, test_only=False, options=None) -> tuple[Path, ...]:
    directory=Path(output_dir) if output_dir else FIGURE_DIR
    if options and (options.get("ratios") is not None or options.get("trial_ids") is not None):
        import hashlib
        key=selection(options.get("ratios"),options.get("trial_ids"))
        stem+="_selection_"+hashlib.sha256(repr(key).encode()).hexdigest()[:12]
        fig.suptitle(f"Selected experiment: {len(key[1])} repeats",fontsize=8)
    if test_only:
        fig.suptitle("TEST ONLY — not manuscript results",fontsize=9)
    directory.mkdir(parents=True, exist_ok=True)
    outputs=tuple(directory/f"{stem}.{ext}" for ext in ("svg","pdf","png"))
    for path in outputs:
        archive(path)
        fig.savefig(path,dpi=600,bbox_inches="tight",pad_inches=.04)
    notes=directory/f"{stem}_notes.txt"
    archive(notes)
    notes.write_text("Weighted F1; PRISM repeats as selected (default 10). Center: arithmetic mean; error bars: min-max, not SD.\n"
                     "Cached SD uses ddof=0. All methods use identical test indices within each task: all valid N>20 graphs for SIS and neuronal three-class, and one shared balanced subset of valid graphs for neuronal binary.\n"
                     "Audited historical MLP/ResInf summaries are frozen into versioned common-support caches; KNN records retain sample IDs and per-trial predictions.\n"
                     "ResInf also reuses historical caches first; missing caches use saved weights or automatically train missing/empty weights.\n",encoding="utf-8")
    plt.close(fig)
    return outputs
