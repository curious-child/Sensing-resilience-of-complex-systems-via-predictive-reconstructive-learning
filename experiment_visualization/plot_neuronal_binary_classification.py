"""Reproduce Supplementary Figure 5 for binary Neuronal classification."""

from __future__ import annotations

import matplotlib.pyplot as plt

from resilience_inference_plotting import (
    add_panel_label,
    load_task,
    plot_options,
    plot_all_methods,
    plot_improvement,
    plot_knn_vs_gbb,
    save_figure,
    set_nature_style,
)


def main(**options):
    set_nature_style()
    frames = load_task("neuronal_binary", **options)
    fig = plt.figure(figsize=(7.2, 5.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=(1.05, 1))
    top = fig.add_subplot(grid[0, :])
    lower_left = fig.add_subplot(grid[1, 0])
    lower_right = fig.add_subplot(grid[1, 1])
    plot_all_methods(
        top,
        frames,
        "Neuronal dynamics",
    )
    plot_knn_vs_gbb(lower_left, frames["knn"], "KNN vs GBB")
    plot_improvement(lower_right, frames["knn"])
    add_panel_label(top, "a")
    add_panel_label(lower_left, "b", y=1.105)
    return save_figure(fig, "supervised_performance_neuronal_2d", options=options)


if __name__ == "__main__":
    for output in main(**plot_options()):
        print(output)
