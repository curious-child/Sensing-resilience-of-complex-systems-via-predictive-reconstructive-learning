"""Reproduce Supplementary Figure 4 (KNN vs GBB)."""

from __future__ import annotations

import matplotlib.pyplot as plt

from resilience_inference_plotting import (
    add_panel_label,
    load_task,
    plot_options,
    plot_improvement,
    plot_knn_vs_gbb,
    save_figure,
    set_nature_style,
)


def main(**options):
    set_nature_style()
    sis = load_task("sis_binary", methods=("knn",), **options)["knn"]
    neuronal = load_task("neuronal_three_class", methods=("knn",), **options)["knn"]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.25), constrained_layout=True)
    plot_knn_vs_gbb(axes[0, 0], sis, "SIS dynamics")
    plot_improvement(axes[0, 1], sis)
    plot_knn_vs_gbb(axes[1, 0], neuronal, "Neuronal dynamics")
    plot_improvement(axes[1, 1], neuronal)
    add_panel_label(axes[0, 0], "a")
    add_panel_label(axes[1, 0], "b")
    return save_figure(fig, "supervised_performance_knn", options=options)


if __name__ == "__main__":
    for output in main(**plot_options()):
        print(output)
