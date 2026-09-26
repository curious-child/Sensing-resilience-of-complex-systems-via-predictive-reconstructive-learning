"""Reproduce main-text Figure 4 from canonical resilience caches."""

from __future__ import annotations

import matplotlib.pyplot as plt

from resilience_inference_plotting import (
    add_panel_label,
    load_task,
    plot_options,
    plot_all_methods,
    save_figure,
    set_nature_style,
)


def main(**options):
    set_nature_style()
    sis = load_task("sis_binary", **options)
    neuronal = load_task("neuronal_three_class", **options)
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.35), constrained_layout=True)
    plot_all_methods(axes[0], sis, "SIS dynamics")
    plot_all_methods(axes[1], neuronal, "Neuronal dynamics")
    add_panel_label(axes[0], "a")
    add_panel_label(axes[1], "b")
    return save_figure(fig, "supervised_performance_all", options=options)


if __name__ == "__main__":
    for output in main(**plot_options()):
        print(output)
