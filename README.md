# Sensing resilience of complex systems via predictive-reconstructive learning

Official implementation of **Sensing resilience of complex systems via predictive-reconstructive learning** by Jun Fu, Peng Zhang, Ruisheng Gu and Lucas Böttcher.

PRISM learns resilience-related representations through joint prediction of node trajectories and network reconstruction. The representations support label-free tracking of resilience changes and classification with limited labeled samples. The experiments cover SIS epidemic and Wilson–Cowan neuronal dynamics on Erdős–Rényi and scale-free networks. Representation learning uses no resilience labels, and inference does not require network structure.

## Installation

The reference environment uses Python 3.10.11, PyTorch 2.0.1 with CUDA 11.8 and PyTorch Geometric 2.5.3. The following commands target Linux with a compatible NVIDIA driver:

```bash
conda create -n prism python=3.10.11 -y
conda activate prism
python -m pip install --upgrade pip
python -m pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
python -m pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.1+cu118.html
python -m pip install -r requirements.txt
```

The project was developed on Windows; the Linux installation above has not been independently validated. For CPU execution or another CUDA version, select matching PyTorch and PyG builds. On Windows, `torch-timeseries==0.1.10` also needs its unused POSIX `resource` imports removed from its dataset loaders.

## Data and checkpoints

Trained PRISM models and classifiers are available in the [v1.0.0 model Release](https://github.com/curious-child/Sensing-resilience-of-complex-systems-via-predictive-reconstructive-learning/releases/tag/v1.0.0). Training and test datasets, simulated trajectories and figure caches are being uploaded to [Zenodo](https://doi.org/10.5281/zenodo.22969883). **The Zenodo record is not yet public.**

Download the required archives and extract them into the repository root, preserving their directory structure:

```text
dataset/                  Training and test trajectories
experiment_results/       Model checkpoints, configurations and analysis caches
```

With the downloaded files in the repository root:

```bash
unzip -n prism-v1.0.0-models-base.zip
unzip -n prism-v1.0.0-models-sis.zip
unzip -n prism-v1.0.0-models-neuronal.zip
```

The SIS and neuronal data ZIPs are split into numbered parts. Once available, download all parts of each required dataset and join them in order before extraction:

```bash
cat prism-v1.0.0-data-sis.zip.part* > prism-v1.0.0-data-sis.zip
cat prism-v1.0.0-data-neuronal.zip.part* > prism-v1.0.0-data-neuronal.zip
unzip -n prism-v1.0.0-data-sis.zip
unzip -n prism-v1.0.0-data-neuronal.zip
```

Extract the separate figure-cache ZIP in the same way. The Release includes `SHA256SUMS.txt` for checking downloaded files. On case-sensitive filesystems, create the alias required by two historical neuronal loaders if it is absent:

```bash
ln -s Neuronal_supervised dataset/neuronal_supervised
```

### Pretraining datasets

The complete neuronal and SIS pretraining inputs are being added to the same Zenodo record. Download every `prism-v1.0.0-pretraining-*.pkl.part*` file and `PRETRAINING_SHA256SUMS.txt` after the record becomes public. Keep each numbered filename unchanged. Each sequence contains raw pickle bytes, not a ZIP archive.

In a clean project checkout containing the downloaded parts, reconstruct the four files and check their SHA-256 values:

```bash
mkdir -p dataset/Neuronal dataset/SIS
(
  set -C  # Refuse to overwrite existing files.
  cat prism-v1.0.0-pretraining-neuronal-As.pkl.part* > dataset/Neuronal/As.pkl || exit 1
  cat prism-v1.0.0-pretraining-neuronal-numes.pkl.part* > dataset/Neuronal/numes.pkl || exit 1
  cat prism-v1.0.0-pretraining-sis-As.pkl.part* > dataset/SIS/As.pkl || exit 1
  cat prism-v1.0.0-pretraining-sis-numes.pkl.part* > dataset/SIS/numes.pkl || exit 1
)
sha256sum -c PRETRAINING_SHA256SUMS.txt
```

Only use parts listed in `manifest.json`; the manifest also records each part's size and checksum. The restored files occupy approximately 60.26 GB in total. Retain the parts until all four checksum checks pass. These pretraining inputs supplement the supervised and test data ZIPs above.

## Simulation and training

The data-generation configurations select training, perturbation and classification experiments. Generate new datasets in a separate directory with:

```bash
python dataset/sis_dataset_generation/generate_sis_dataset.py --config dataset/sis_dataset_generation/config.yaml --output generated_data
python dataset/neuronal_dataset_generation/generate_neuronal_dataset.py --config dataset/neuronal_dataset_generation/config.yaml --output generated_data
```

For PRISM pretraining, edit `configs/grid_search/AE_recon_pred_model.yaml`: set `dataset.spdata_file_path` to the training directory, `loss.loss_type` to `["multitask"]`, and `net.device` to an available device, such as `["cuda:0"]`. Then run:

```bash
python multitask_model_train.py --cfg configs/grid_search/AE_recon_pred_model.yaml --train_mode grid
```

The supplied configuration otherwise selects separate prediction and reconstruction runs. Experiment-specific settings are stored alongside the checkpoints in `experiment_results/`.

## Inference and figure reproduction

Inference and cache-generation routines are in `experiment_process/`. After restoring the corresponding models and data, this command computes or reuses SIS inductive-analysis caches:

```bash
python experiment_process/generate_inductive_analysis_cache.py --dynamic SIS --task multitask --device cuda:0
```

To redraw the reported figures, restore the supplied figure caches and run the relevant scripts from the repository root:

```bash
python experiment_visualization/plot_inductive_loss_task_analysis.py
python experiment_visualization/plot_resilience_inference_performance.py
```

| Figure | Script in `experiment_visualization/` |
|---|---|
| Figure 2 | `plot_inductive_loss_task_analysis.py` |
| Figure 3 | `plot_inductive_representation_early_warning.py` |
| Figure 4 | `plot_resilience_inference_performance.py` |
| Figure 5 | `plot_latent_embedding_tsne.py` |
| Supplementary Figure 1 | `plot_transductive_loss_task_analysis.py` |
| Supplementary Figure 2 | `plot_loss_parameter_sensitivity.py` |
| Supplementary Figure 3 | `plot_training_parameter_sensitivity.py` |
| Supplementary Figure 4 | `plot_knn_gbb_comparison.py` |
| Supplementary Figure 5 | `plot_neuronal_binary_classification.py` |
| Supplementary Figure 6 | `plot_sis_scaling.py` |

Outputs are saved under `experiment_results/figures/`. Figure 1 is a conceptual illustration. Cache-based plotting does not require retraining; missing caches can trigger inference or training, so retain the supplied caches and avoid force options when redrawing historical results.

The supplied caches support reproduction of the manuscript figures; running the experiments from scratch requires the corresponding datasets and configurations.

## Repository structure

```text
configs/                   Model and training configurations
dataset/                   Simulation and data-generation code
models/                    PRISM and baseline architectures
loss_functions/            Prediction and reconstruction objectives
train/                     Training loops
optimizers/                Optimizer utilities
experiment_process/        Inference and analysis-cache generation
experiment_visualization/  Manuscript figure scripts
experiment_results/        Experiment-specific configurations
utils/                     Shared data and model utilities
```

## Citation

If this repository contributes to your work, please cite the associated manuscript:

Jun Fu, Peng Zhang, Ruisheng Gu and Lucas Böttcher. *Sensing resilience of complex systems via predictive-reconstructive learning*.

## Acknowledgements

The implementation uses or adapts components from [ResInf](https://github.com/tsinghua-fib-lab/ResInf), [Non-stationary Transformers](https://github.com/thuml/Nonstationary_Transformers) and [geoopt](https://github.com/geoopt/geoopt). The GBB baseline follows Gao, Barzel and Barabási, [Universal resilience patterns in complex networks](https://doi.org/10.1038/nature16948). The legacy geoopt optimizer is not used by the configured Adam training path.

## License

The authors' code is released under the MIT License. Third-party components retain their original licenses and copyright notices, reproduced in [LICENSE](LICENSE).
