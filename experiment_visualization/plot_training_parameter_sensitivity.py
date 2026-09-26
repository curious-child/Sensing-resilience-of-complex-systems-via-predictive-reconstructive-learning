"""Reproduce Supplementary Figure 3: training-parameter sensitivity."""

from __future__ import annotations

import matplotlib.pyplot as plt

from model_parameter_plotting import create_parameter_figure, save_figure


DEVICE = "cuda"
FORCE_REGENERATE_CACHE = False
FORCE_RETRAIN_MODEL = False


def main() -> None:
    figure = create_parameter_figure(
        "learning_rate",
        "dropout",
        device=DEVICE,
        force_cache=FORCE_REGENERATE_CACHE,
        force_train=FORCE_RETRAIN_MODEL,
    )
    png, pdf = save_figure(
        figure, "param_train_sensitivity"
    )
    plt.close(figure)
    print(png)
    print(pdf)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    if args.output_dir:
        from pathlib import Path
        import model_parameter_plotting
        model_parameter_plotting.OUTPUT_DIR = Path(args.output_dir)
    DEVICE, FORCE_REGENERATE_CACHE = args.device, args.force_cache
    main()
