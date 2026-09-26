# Third-party methods and software

The repository's MIT license applies to the authors' contributions. It does not
replace third-party copyright notices or licenses.

- **ResInf**: the baseline implementation in `models/pred_model/resinf.py`
  follows the ResInf architecture, with project-specific classification support.
  Source: https://github.com/tsinghua-fib-lab/ResInf
  Paper: Liu et al., *Deep learning resilience inference for complex networked systems*,
  Nature Communications (2024), https://doi.org/10.1038/s41467-024-53303-4.
  Upstream MIT notice: `licenses/resinf-MIT.txt`.
- **Non-stationary Transformers**: `models/pred_model/NS_Transformer.py` adapts
  this architecture and imports components from `torch-timeseries`.
  Source: https://github.com/thuml/Nonstationary_Transformers
  Paper: Liu et al., *Non-stationary Transformers: Exploring the Stationarity in
  Time Series Forecasting*, NeurIPS (2022).
  Upstream MIT notice: `licenses/transformer-MIT.txt`.
- **GBB**: analytical baseline based on Gao, Barzel and Barabási,
  *Universal resilience patterns in complex networks*, Nature (2016),
  https://doi.org/10.1038/nature16948. See the implementation and manuscript
  for the dynamical assumptions and task-specific mappings.

External Python packages are installed separately and retain their own licenses.
- **geoopt legacy optimizer:** `optimizers/radam.py` identifies geoopt as its
  source and contains a project-specific manifold import. Its original Apache
  2.0 notice is retained in `licenses/geoopt-LICENSE.txt`.
  Source: https://github.com/geoopt/geoopt. This optional legacy module is not
  used by the configured Adam optimizer; its manifold dependency is absent.

Their source distributions are not bundled in the release archives.
The released SIS and neuronal files are simulation data supplied with this project;
the release does not incorporate the real-world datasets from the ResInf repository.
