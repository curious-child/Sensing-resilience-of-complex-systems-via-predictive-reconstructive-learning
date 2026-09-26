# PRISM v1.0.0 local release verification

Validation date: 2026-09-26. The source project was kept separate from the
publishing copy. No model training, simulation regeneration, weight changes or
scientific computation changes were made for publication.

## Model and archive integrity

- 19 PRISM base checkpoints and 437 nonempty classifier checkpoints passed
  strict architecture/state-dictionary loading on CPU, including checks for
  nonfinite floating-point parameters.
- 60 empty neuronal classifier files were excluded and recorded in
  `excluded-files.json`.
- Three model ZIPs are each below 2 GiB. Two data ZIPs contain 57 data files;
  one cache ZIP contains 161 records, summaries, arrays and provenance files.
- Archives were extracted to independent locations. Every restored file was
  compared against its source SHA-256. Relative paths and file bytes are preserved.
- Downloader tests passed for checksum rejection, path traversal rejection,
  identical-file skipping and refusal to overwrite different existing files.

## Dependencies and cache reproduction

- A fresh Python 3.10 virtual environment was installed from `requirements.txt`
  plus the official PyG CPU wheels. `pip check` passed.
- Installation exposed missing `torchvision`/`torchaudio` import dependencies
  and unused POSIX `resource` imports in torch-timeseries 0.1.10 on Windows.
  These were addressed in the release requirements and explicit compatibility
  helper. The existing environment already had those unused imports disabled.
  The installed torch-timeseries neural-layer source files match the original
  environment after normalizing line endings.
- All ten experimental figure entrypoints (main Figs. 2–5 and supplementary
  Figs. 1–6) completed in the original environment and the fresh CPU environment.
  During these checks, `torch.load` and `pickle.load` were blocked so an absent
  cache could not silently trigger model inference, data loading or training.
- All 161 selected cache file hashes remained unchanged after plotting.
- Classification checks covered nine task/method cache groups: six historical
  aggregate groups had matching stored summary hashes and valid min–mean–max
  bounds; 300 KNN trial weighted F1 scores and their GBB scores were recomputed
  from stored predictions and labels.
- Scaling caches contained all 372 expected method/trial/topology/size jobs.
  The existing pooled ER/SF summary routine produced 24 size/method rows from
  the recorded predictions, timings and memory measurements. No new timings
  were substituted for historical measurements.

`requirements-validated-windows-cpu.txt` records the complete fresh-environment
package versions. CPU cache reproduction does not require a GPU. Fresh training
and GPU inference were not benchmarked during this packaging task.

## Material limits

- Historical raw pretraining folders `dataset/SIS/` and `dataset/Neuronal/`
  were empty. Historical `dataset/SIS_scaling/` inputs were not found in the
  supplied project or workspace archives. These are not included.
- The cache package contains historical summaries where full predictions were
  not retained. Historical neuronal-binary MLP checkpoints were not saved.
  Other available models do not replace those missing historical checkpoints.
- Fresh simulation or training is a new experiment; it is not a byte-for-byte
  reconstruction of missing historical data. Random seeds alone do not remove
  differences between hardware, libraries or training runs.
- Existing scientific scripts can train missing models when asked to regenerate
  caches. Use the complete cache download and default plotting commands to
  reproduce existing figures without that behavior.
- Windows was tested. Linux/macOS were not tested; see the README for the
  historical neuronal directory case alias required on case-sensitive filesystems.
- The optional legacy `optimizers/radam.py` refers to an absent
  `models.manifolds` module. It is not imported by the configured Adam training
  path, whose command-line entrypoint was checked. This missing optional
  dependency was documented rather than replaced or silently repaired.

These checks establish the documented cache reproduction and checkpoint-loading
scope, not complete regeneration of every published experiment from raw inputs.
