# Sensing resilience of complex systems via predictive-reconstructive learning

PRISM learns representations of resilience changes through joint prediction of
node trajectories and network reconstruction. It supports tracking resilience
changes without resilience labels and classification using limited labeled data.
The experiments cover SIS and neuronal dynamics on ER and SF networks.

Authors: Jun Fu, Peng Zhang, Ruisheng Gu, and Lucas Böttcher.

## Release contents

- **This repository:** code, configurations, documentation and release manifests.
- **Model weights:** [v1.0.0 Release](https://github.com/curious-child/Sensing-resilience-of-complex-systems-via-predictive-reconstructive-learning/releases/tag/v1.0.0),
  in three ZIP assets (base PRISM models, SIS classifiers, neuronal classifiers).
- **Datasets and experiment caches:** Zenodo publication is being prepared;
  the DOI and download URLs will be added after publication. No DOI is claimed yet.

`release/manifest.json` lists each archive and its individual files, sizes,
SHA-256 checksums, purpose and original restoration path. Model weights and large
data are not stored in Git history. See `release/VALIDATION.md` for verification
results and limitations, and `release/excluded-files.json` for omitted files.

## Installation

The original environment uses Python 3.10.11, PyTorch 2.0.1 and PyG 2.5.3.
Create a separate environment; run commands from the repository root.

```bash
python -m venv .venv
# Windows PowerShell: .venv/Scripts/Activate.ps1
# Linux/macOS shell: source .venv/bin/activate
python -m pip install -r requirements.txt torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.1+cpu.html
python tools/fix_windows_dependency.py
python -m pip check
```

The command above installs CPU wheels for validation and cache plotting. For
GPU training/inference, first install PyTorch 2.0.1 with the appropriate CUDA
build and matching PyG extension wheels; the original setup used CUDA 11.8
and `https://data.pyg.org/whl/torch-2.0.1+cu118.html`.
Do not mix CPU and CUDA extension wheels.
The Windows helper removes an unused POSIX-only import in torch-timeseries
0.1.10, matching the existing environment's compatibility adjustment. It changes
no model layer or computation.

## Download and restore

```bash
python tools/download_assets.py --category caches
python tools/download_assets.py --category models
python tools/download_assets.py --category data
# Or select --category all. Archives are retained in .downloads/.
python tools/download_assets.py --category all --verify-only
```

The downloader checks archive and file hashes, restores original relative paths,
skips identical files and refuses to overwrite different existing content.
For archives downloaded manually, use `--archive-dir /path/to/archives`.
The data commands require the Zenodo URLs to have been published in the manifest.

The source was developed on Windows. On a case-sensitive filesystem, two loaders
expect `dataset/neuronal_supervised` while the historical folder is
`dataset/Neuronal_supervised`. After restoring data, create this alias if absent:

```bash
ln -s Neuronal_supervised dataset/neuronal_supervised
```

## Reproduce figures from supplied caches

Restore the **caches** package first. All ten experimental figure entrypoints
were exercised using the supplied caches with model/data loading disabled.
Fig. 1 is a conceptual illustration and has no numerical reproduction script.

| Figure | Script in `experiment_visualization/` | Cache directory under `experiment_results/` |
|---|---|---|
| Main Fig. 2 | `plot_inductive_loss_task_analysis.py` | each dynamics: `loss_inductive_analysis/` |
| Main Fig. 3 | `plot_inductive_representation_early_warning.py` | each dynamics: `loss_inductive_analysis/` |
| Main Fig. 4 | `plot_resilience_inference_performance.py` | each dynamics: `resilience_inference_analysis/` |
| Main Fig. 5 | `plot_latent_embedding_tsne.py` | each dynamics: `resilience_inference_analysis/figure5_tsne/` |
| Supplementary Fig. 1 | `plot_transductive_loss_task_analysis.py` | each dynamics: `loss_transductive_analysis/` |
| Supplementary Fig. 2 | `plot_loss_parameter_sensitivity.py` | `SIS/model_param_analysis/` |
| Supplementary Fig. 3 | `plot_training_parameter_sensitivity.py` | `SIS/model_param_analysis/` |
| Supplementary Fig. 4 | `plot_knn_gbb_comparison.py` | each dynamics: `resilience_inference_analysis/` |
| Supplementary Fig. 5 | `plot_neuronal_binary_classification.py` | `Neuronal/resilience_inference_analysis/` |
| Supplementary Fig. 6 | `plot_sis_scaling.py` | `SIS/scaling/` |

For example:

```bash
python experiment_visualization/plot_inductive_loss_task_analysis.py
python experiment_visualization/plot_resilience_inference_performance.py
python experiment_visualization/plot_sis_scaling.py
```

Figures are written under `experiment_results/figures/`. Classification plots use
weighted F1 means and observed min–max ranges; scaling plots use their recorded
standard deviations. Cache values, class definitions and scientific plotting
logic have not been altered for publication.

**Use the complete supplied caches for historical figures.** Existing entrypoints
can generate missing caches and, in some cases, train missing models. Do not use
`--force`/`--force-cache`, remove caches, or request partial trial selections when
the intention is only to redraw the reported results.

## Inference and new training

These are distinct from cache-based reproduction:

1. **Existing models:** restore models and the corresponding test/train reference
   data. The `experiment_process/` scripts implement inference and cache creation;
   inspect their `--help` before using a force option. All 19 base models and 437
   nonempty classifier checkpoints passed strict loading in their corresponding
   architectures. This is a compatibility check, not a claim that fresh inference
   was run for every reported experiment.
2. **New data and training:** use the SIS and neuronal generators and their
   documented configurations in `dataset/sis_dataset_generation/` and
   `dataset/neuronal_dataset_generation/`. Generate into a separate directory.
   Newly generated data are not the missing historical original data.
   The pretraining entrypoint is:

   ```bash
   python multitask_model_train.py --cfg configs/grid_search/AE_recon_pred_model.yaml --train_mode grid
   ```

   Configure dataset paths and hyperparameters before running: set
   `loss.loss_type: ["multitask"]` for joint learning (the supplied grid currently
   lists separate prediction/reconstruction runs), and change `net.device`
   from its historical `cuda:2` value to an available device. Original raw
   pretraining folders `dataset/SIS/` and `dataset/Neuronal/` are empty in the
   supplied project. The historical `dataset/SIS_scaling/` files were not found.
   Scaling caches can be plotted, but those original large-network inputs cannot
   be restored from this release.

Sixty empty neuronal classifier files are excluded. Some historical MLP/ResInf
results are supplied as verified aggregate summaries rather than complete
per-sample predictions. In particular, neuronal-binary MLP checkpoints for the
historical comparison were not saved. These limitations prevent claiming full
end-to-end regeneration of every historical result. No models were retrained to
fill gaps for this release.

## Code structure

`models/`: PRISM and baseline architectures; `loss_functions/`: training losses;
`train/`, `optimizers/`, `configs/`: training and configuration;
`dataset/*_dataset_generation/`: simulation code;
`experiment_process/`: evaluation and cache generation;
`experiment_visualization/`: figure entrypoints; `utils/`: shared utilities;
`tools/`: release download and verification.

## Citation and license

See `CITATION.cff` for authorship and the software title. A manuscript DOI has not
been added because none has been verified. The authors' code and released
self-generated data/caches use MIT; third-party notices and source references are
listed in `THIRD_PARTY_NOTICES.md` and `licenses/`.
