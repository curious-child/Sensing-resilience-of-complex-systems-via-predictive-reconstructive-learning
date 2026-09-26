"""Train/evaluate SIS parameter-sensitivity models and maintain their caches.

The implementation is standalone with respect to AE_params_analysis.py and the
legacy experiment entrypoints.  It uses the project's model definitions, but
copies the dataset preparation, hold-out training, checkpointing, inference,
and cache validation logic needed by Supplementary Figures 2 and 3.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import pickle
import random
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ANALYSIS_ROOT = (
    PROJECT_ROOT / "experiment_results" / "SIS" / "model_param_analysis"
)
TRAIN_DATA_ROOT = PROJECT_ROOT / "dataset" / "SIS_supervised"
TEST_DATA_ROOT = PROJECT_ROOT / "dataset" / "SIS_test"

RANDOM_SEED = 123
INFERENCE_SEED = 1029
OBSERVATIONS = 5
WINDOWS = 10
NETWORK_CONDITIONS = (
    ("ER", "degree"),
    ("ER", "random"),
    ("SF", "degree"),
    ("SF", "random"),
)
ZLATENT_KEYS = (
    "Z_mean",
    "Z_var",
    "distance_mean",
    "distance_var",
    "distance_G",
)


@dataclass(frozen=True)
class RunSpec:
    parameter_group: str
    value_directory: str

    @property
    def key(self) -> str:
        return f"{self.parameter_group}/{self.value_directory}"

    @property
    def directory(self) -> Path:
        return ANALYSIS_ROOT / self.parameter_group / self.value_directory

    @property
    def config_path(self) -> Path:
        return self.directory / "config.yaml"

    @property
    def checkpoint_path(self) -> Path:
        return self.directory / "model_trained"

    @property
    def training_history_path(self) -> Path:
        return self.directory / "record_scores.json"

    def cache_path(self, net_struct: str, remove_type: str) -> Path:
        return self.directory / (
            f"multitask_{net_struct}_{remove_type}_record_train_True.json"
        )


PARAMETER_GROUPS: dict[str, tuple[tuple[str, RunSpec], ...]] = {
    "negative_sampling_ratio": (
        ("1", RunSpec("negative_sampling_ratio", "value_1")),
        ("3", RunSpec("negative_sampling_ratio", "value_3")),
    ),
    "degree_loss_weight": (
        ("10", RunSpec("degree_loss_weight", "value_10")),
        ("1", RunSpec("degree_loss_weight", "value_1")),
        ("0", RunSpec("degree_loss_weight", "value_0")),
    ),
    "learning_rate": (
        ("1e-3", RunSpec("learning_rate", "value_1e-3")),
        ("1e-4", RunSpec("learning_rate", "value_1e-4")),
    ),
    "dropout": (
        ("0.05", RunSpec("dropout", "value_0.05")),
        ("0.10", RunSpec("dropout", "value_0.10")),
        ("0.20", RunSpec("dropout", "value_0.20")),
        ("0.50", RunSpec("dropout", "value_0.50")),
    ),
}

FIGURE_GROUPS = {
    "sfigure2": ("negative_sampling_ratio", "degree_loss_weight"),
    "sfigure3": ("learning_rate", "dropout"),
}

PARAMETER_DISPLAY_NAMES = {
    "negative_sampling_ratio": "Negative sampling ratio",
    "degree_loss_weight": "Degree-loss weight",
    "learning_rate": "Learning rate",
    "dropout": "Dropout",
}


def _seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _atomic_json(record: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _atomic_torch_save(record: Any, destination: Path) -> None:
    import torch

    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    try:
        torch.save(record, temporary)
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _load_config(spec: RunSpec) -> dict[str, Any]:
    if not spec.config_path.is_file():
        raise FileNotFoundError(
            f"Missing model configuration for {spec.key}: {spec.config_path}"
        )
    with spec.config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    required = {"dataset", "train", "net", "loss", "optimizer"}
    missing = required - set(config or {})
    if missing:
        raise ValueError(
            f"{spec.config_path} is missing sections: {sorted(missing)}"
        )
    expected_value = float(spec.value_directory.removeprefix("value_"))
    actual_values = {
        "negative_sampling_ratio": config["net"]["neg_samples_ratio"],
        "degree_loss_weight": config["loss"]["degree_loss_weight"],
        "learning_rate": config["optimizer"]["lr"],
        "dropout": config["net"]["Dy_transformer"]["dropout"],
    }
    actual_value = actual_values[spec.parameter_group]
    if not np.isclose(float(actual_value), expected_value):
        raise ValueError(
            f"{spec.config_path}: {spec.parameter_group} is {actual_value!r}, "
            f"but directory {spec.value_directory!r} declares {expected_value!r}"
        )
    return config


def _load_pickle(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("rb") as stream:
        return pickle.load(stream)


def _prepare_training_dataset(config: dict[str, Any], source_root: Path | None = None):
    """Copy of the original trajectory-to-PyG conversion for train+val data."""

    import torch
    import torch_geometric.utils as tg_utils
    from torch_geometric.data import Data

    dataset: list[Any] = []
    maximum_degree = 0.0
    observations = int(config["dataset"]["observations"])
    windows = int(config["dataset"]["windows"])
    maximum_samples = int(config["dataset"].get("max_sample_num", 1))

    adjacency_list: list[Any] = []
    trajectory_list: list[Any] = []
    missing: list[Path] = []
    roots = (source_root,) if source_root is not None else tuple(
        TRAIN_DATA_ROOT / split for split in ("train_dataset", "val_dataset"))
    for root in roots:
        adjacency_path = root / "As.pkl"
        trajectory_path = root / "numes.pkl"
        for path in (adjacency_path, trajectory_path):
            if not path.is_file():
                missing.append(path)
        if not missing:
            adjacency_list.extend(_load_pickle(adjacency_path))
            trajectory_list.extend(_load_pickle(trajectory_path))
    if missing:
        listing = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            f"Cannot train PRISM; missing original training inputs:\n{listing}"
        )
    if len(adjacency_list) != len(trajectory_list):
        raise ValueError("SIS training adjacency and trajectory counts differ")

    for adjacency, trajectory in zip(adjacency_list, trajectory_list):
        adjacency_tensor = torch.as_tensor(adjacency)
        trajectory_tensor = torch.as_tensor(trajectory)
        if adjacency_tensor.shape[0] <= 20:
            continue
        if trajectory_tensor.shape[2] != adjacency_tensor.shape[0]:
            raise ValueError("Training trajectory and adjacency node counts differ")
        if trajectory_tensor.shape[1] < 2 * windows:
            raise ValueError("Training trajectory is shorter than two windows")
        if trajectory_tensor.shape[0] < observations:
            raise ValueError("Training trajectory has too few observations")

        edge_index = tg_utils.dense_to_sparse(adjacency_tensor)[0]
        trajectory_tensor = (
            trajectory_tensor.permute(2, 0, 1).unsqueeze(-1).float()
        )
        number_observations = trajectory_tensor.shape[1]
        total_combinations = math.comb(number_observations, observations)
        if total_combinations <= maximum_samples:
            import itertools

            combinations = list(
                itertools.combinations(range(number_observations), observations)
            )
        else:
            selected: set[tuple[int, ...]] = set()
            while len(selected) < maximum_samples:
                selected.add(
                    tuple(
                        sorted(
                            random.sample(
                                range(number_observations), observations
                            )
                        )
                    )
                )
            combinations = list(selected)

        for indices in combinations[:maximum_samples]:
            sampled = trajectory_tensor[:, indices, :, :]
            dataset.append(
                Data(
                    x=sampled[:, :, :windows, :],
                    edge_index=edge_index,
                    y=sampled[:, :, windows : 2 * windows, :],
                )
            )
        maximum_degree = max(
            maximum_degree,
            float(tg_utils.degree(edge_index[0]).max().item()),
        )
    if not dataset:
        raise ValueError("The prepared SIS training dataset is empty")
    return dataset, maximum_degree


def _build_net_parameters(
    config: dict[str, Any], maximum_degree: float, device: str
) -> dict[str, Any]:
    parameters = copy.deepcopy(config["net"])
    parameters.update(
        {
            "model_task": "AE_recon",
            "device": device,
            "max_degree_dataset": maximum_degree,
            "degree_loss_weight": config["loss"]["degree_loss_weight"],
            "log_var": config["loss"].get("log_var", 0.0),
            "lr": config["optimizer"]["lr"],
        }
    )
    parameters["Dy_transformer"]["windows"] = config["dataset"]["windows"]
    parameters["Dy_transformer"]["dataset_nf"] = 1
    return parameters


def _train_model(spec: RunSpec, device: str, *, config=None, source_root=None,
                 task="multitask") -> Path:
    """Train one missing checkpoint using copied hold-out training behavior."""

    import torch
    from sklearn.model_selection import train_test_split
    from torch_geometric.loader import DataLoader

    from models.models import Resilience_model

    config = _load_config(spec) if config is None else config
    _seed_everything(RANDOM_SEED)
    dataset, maximum_degree = _prepare_training_dataset(config, source_root)
    train_set, validation_set = train_test_split(
        dataset,
        train_size=float(config["train"]["traindata_size"]),
        random_state=RANDOM_SEED,
        shuffle=True,
    )
    net_parameters = _build_net_parameters(config, maximum_degree, device)
    model = Resilience_model(net_parameters).to(device)
    optimizer_name = str(config["optimizer"]["optimizer_name"]).lower()
    optimizer_class = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
    }.get(optimizer_name)
    if optimizer_class is None:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    optimizer = optimizer_class(
        model.parameters(),
        lr=float(config["optimizer"]["lr"]),
        weight_decay=float(config["optimizer"].get("weight_decay", 0.0)),
    )
    scheduler = None
    if config["optimizer"].get("scheduler_set", False):
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=config["optimizer"]["MstepLR_milestones"],
            gamma=float(config["optimizer"]["MstepLR_gamma"]),
        )

    train_loader = DataLoader(
        train_set,
        batch_size=int(config["train"]["train_batch_size"]),
        shuffle=True,
        drop_last=False,
    )
    validation_loader = DataLoader(
        validation_set,
        batch_size=int(config["train"]["val_batch_size"]),
        shuffle=False,
        drop_last=False,
    )
    history = {
        "epoch": [],
        "predictability": {"train_scores": [], "val_scores": []},
        "reconstructability": {"train_scores": [], "val_scores": []},
        "multitask": {"train_scores": [], "val_scores": []},
    }
    if source_root is not None:
        positions = {id(graph): index for index, graph in enumerate(dataset)}
        history["training_source"] = dict(seed=RANDOM_SEED, task=task, config=config,
            data_root=str(source_root), train_positions=[positions[id(g)] for g in train_set],
            validation_positions=[positions[id(g)] for g in validation_set],
            checkpoint_selection="final_epoch", dataset_kind="original_unlabelled")
    task_condition = torch.tensor([{"prediction": 0, "reconstruction": 1,
                                    "multitask": 2}[task]], device=device)
    epochs = int(config["train"]["train_epochs"])
    for epoch in range(epochs):
        model.train()
        train_totals = np.zeros(3, dtype=float)
        train_batches = 0
        for graph in train_loader:
            graph = graph.to(device)
            optimizer.zero_grad()
            losses, selected_task = model.training_step(
                graph, task_cond=task_condition
            )
            loss = losses[selected_task]
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_totals += np.asarray(
                [float(item.detach().cpu()) for item in losses]
            )
            train_batches += 1
        if not train_batches:
            raise RuntimeError(
                f"No finite training batches for {spec.key} at epoch {epoch}"
            )
        if scheduler is not None:
            scheduler.step()

        model.eval()
        validation_totals = np.zeros(3, dtype=float)
        validation_batches = 0
        with torch.no_grad():
            for graph in validation_loader:
                graph = graph.to(device)
                losses, _ = model.training_step(
                    graph, task_cond=task_condition
                )
                if not all(torch.isfinite(item) for item in losses):
                    continue
                validation_totals += np.asarray(
                    [float(item.detach().cpu()) for item in losses]
                )
                validation_batches += 1
        if not validation_batches:
            raise RuntimeError(
                f"No finite validation batches for {spec.key} at epoch {epoch}"
            )
        history["epoch"].append(epoch)
        names = ("predictability", "reconstructability", "multitask")
        for index, name in enumerate(names):
            history[name]["train_scores"].append(
                float(train_totals[index] / train_batches)
            )
            history[name]["val_scores"].append(
                float(validation_totals[index] / validation_batches)
            )

    _atomic_torch_save(
        {"net_param": net_parameters, "state_dict": model.state_dict()},
        spec.checkpoint_path,
    )
    _atomic_json(history, spec.training_history_path)
    return spec.checkpoint_path


def ensure_prism_checkpoint(checkpoint: Path, device: str, *, task="multitask") -> Path:
    """Restore only an absent base model, using its original unlabelled data."""
    from types import SimpleNamespace
    checkpoint = Path(checkpoint)
    if checkpoint.is_file():
        return checkpoint  # A corrupt existing model must fail strict loading.
    config_path = checkpoint.with_suffix(".yaml")
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    source = Path(config["dataset"]["spdata_file_path"])
    if not source.is_absolute():
        source = PROJECT_ROOT / source
    spec = SimpleNamespace(key=str(checkpoint), checkpoint_path=checkpoint,
        training_history_path=checkpoint.parent / "record_scores.json")
    print(f"Training missing PRISM {task} model: {checkpoint}", flush=True)
    return _train_model(spec, device, config=config, source_root=source, task=task)


def _load_model(spec: RunSpec, device: str):
    import torch

    from models.models import Resilience_model

    checkpoint = torch.load(spec.checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or not {
        "net_param",
        "state_dict",
    }.issubset(checkpoint):
        raise ValueError(f"Invalid model checkpoint: {spec.checkpoint_path}")
    net_parameters = copy.deepcopy(checkpoint["net_param"])
    net_parameters.update({"model_task": "AE_recon", "device": device})
    model = Resilience_model(net_parameters).to(device)
    state = {
        key.replace("module.", ""): value
        for key, value in checkpoint["state_dict"].items()
    }
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def _load_test_data(net_struct: str, remove_type: str):
    paths = (
        TEST_DATA_ROOT / f"As_{net_struct}_{remove_type}.pkl",
        TEST_DATA_ROOT / f"numes_{net_struct}_{remove_type}.pkl",
        TEST_DATA_ROOT / f"xlast_{net_struct}_{remove_type}.pkl",
    )
    missing = [path for path in paths if not path.is_file()]
    if missing:
        listing = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing SIS parameter-analysis test data:\n{listing}")
    values = tuple(_load_pickle(path) for path in paths)
    if len({len(value) for value in values}) != 1:
        raise ValueError(
            f"SIS test-data lengths differ for {net_struct}/{remove_type}"
        )
    return values


def _evaluate_model(
    model: Any,
    adjacency_list: Sequence[Any],
    trajectory_list: Sequence[Any],
    stable_state_list: Sequence[Any],
) -> dict[str, Any]:
    """Copied parameter-analysis inference and latent-statistics calculation."""

    import torch
    import torch_geometric.utils as tg_utils
    from torch_geometric.data import Data

    performance = {
        "Learn_error": {"ts_err": [], "graph_err": []},
        "Zlatent": {key: [] for key in (*ZLATENT_KEYS, "distance_G_var")},
    }
    scatter_x: list[int] = []
    line_x: list[int] = []
    scatter_y: list[float] = []
    original_nodes = int(np.asarray(adjacency_list[0]).shape[0])
    original_graph_embedding = None

    for adjacency, trajectory, stable_state in zip(
        adjacency_list, trajectory_list, stable_state_list
    ):
        adjacency_tensor = torch.as_tensor(adjacency)
        edge_index = tg_utils.dense_to_sparse(adjacency_tensor)[0]
        if edge_index.shape[1] <= 0:
            break
        current_nodes = int(adjacency_tensor.shape[0])
        removed_nodes = original_nodes - current_nodes
        line_x.append(removed_nodes)
        for state in torch.as_tensor(stable_state):
            if state.shape[0] != current_nodes:
                raise ValueError("Stable-state and adjacency node counts differ")
            scatter_x.append(removed_nodes)
            scatter_y.append(float(state.mean()))

        trajectory_tensor = (
            torch.as_tensor(trajectory).permute(2, 0, 1).unsqueeze(-1).float()
        )
        selected = random.sample(
            range(trajectory_tensor.shape[1]), OBSERVATIONS
        )
        sampled = trajectory_tensor[:, selected, :, :]
        graph = Data(
            x=sampled[:, :, :WINDOWS, :],
            edge_index=edge_index,
            y=sampled[:, :, WINDOWS : 2 * WINDOWS, :],
        ).to(model.device)
        prediction_error, reconstruction_error = model.eval_pred_error(graph)
        z_mean, z_variance, node_embedding = model.eval_Zlatent(graph)
        pairwise_distance = torch.pdist(node_embedding, p=2)
        current_graph_embedding = node_embedding.mean(dim=0)
        if original_graph_embedding is None:
            original_graph_embedding = current_graph_embedding
            graph_distance = 0.0
        else:
            graph_distance = float(
                torch.norm(
                    current_graph_embedding - original_graph_embedding, p=2
                )
                .detach()
                .cpu()
            )
        performance["Learn_error"]["ts_err"].append(
            float(prediction_error.detach().cpu())
        )
        performance["Learn_error"]["graph_err"].append(
            float(reconstruction_error.detach().cpu())
        )
        performance["Zlatent"]["Z_mean"].append(float(z_mean.detach().cpu()))
        performance["Zlatent"]["Z_var"].append(
            float(z_variance.detach().cpu())
        )
        performance["Zlatent"]["distance_mean"].append(
            float(pairwise_distance.mean().detach().cpu())
        )
        performance["Zlatent"]["distance_var"].append(
            float(pairwise_distance.var().detach().cpu())
        )
        performance["Zlatent"]["distance_G"].append(graph_distance)
        performance["Zlatent"]["distance_G_var"].append(
            float(current_graph_embedding.var().detach().cpu())
        )
    return {
        "perform_record": performance,
        "remove_nodes_scatter_record": scatter_x,
        "remove_nodes_record": line_x,
        "xlast_system_scatter_record": scatter_y,
    }


def validate_cache(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as stream:
        record = json.load(stream)
    required_top = {
        "perform_record",
        "remove_nodes_scatter_record",
        "remove_nodes_record",
        "xlast_system_scatter_record",
    }
    missing = required_top - set(record)
    if missing:
        raise ValueError(f"{path} is missing fields: {sorted(missing)}")
    line_count = len(record["remove_nodes_record"])
    if line_count == 0:
        raise ValueError(f"{path} has no analysis samples")
    if len(record["remove_nodes_scatter_record"]) != len(
        record["xlast_system_scatter_record"]
    ):
        raise ValueError(f"{path} has inconsistent system-state coordinates")
    performance = record["perform_record"]
    for section, keys in (
        ("Learn_error", ("ts_err", "graph_err")),
        ("Zlatent", ZLATENT_KEYS),
    ):
        if section not in performance:
            raise ValueError(f"{path} is missing perform_record.{section}")
        for key in keys:
            values = performance[section].get(key)
            if not isinstance(values, list) or len(values) != line_count:
                raise ValueError(
                    f"{path}: {section}.{key} must contain {line_count} values"
                )
            finite = np.isfinite(np.asarray(values, dtype=float))
            if not finite.all():
                invalid = np.flatnonzero(~finite).tolist()
                legacy_sparse_graph_nan = (
                    section == "Learn_error"
                    and key == "graph_err"
                    and finite.mean() >= 0.95
                )
                if not legacy_sparse_graph_nan:
                    raise ValueError(
                        f"{path}: {section}.{key} has non-finite values "
                        f"at indices {invalid}"
                    )
    return record


def ensure_run(
    spec: RunSpec,
    conditions: Iterable[tuple[str, str]] = NETWORK_CONDITIONS,
    *,
    device: str = "cuda",
    force_cache: bool = False,
    force_train: bool = False,
) -> list[Path]:
    """Ensure one model and the requested inference caches exist and are valid."""

    requested = tuple(dict.fromkeys(conditions))
    for condition in requested:
        if condition not in NETWORK_CONDITIONS:
            raise ValueError(f"Unsupported SIS test condition: {condition}")

    pending: list[tuple[str, str]] = []
    resolved: list[Path] = []
    for net_struct, remove_type in requested:
        path = spec.cache_path(net_struct, remove_type)
        from experiment_process.generate_resilience_inference_cache import cache_matches_dataset
        data_root = Path(__file__).resolve().parents[1] / "dataset/SIS_test"
        if path.is_file() and not force_cache and cache_matches_dataset(path, data_root):
            try:
                validate_cache(path)
                resolved.append(path)
                continue
            except (ValueError, KeyError, TypeError):
                pass
        pending.append((net_struct, remove_type))

    if not pending and not force_train:
        return resolved
    _load_config(spec)
    if force_train or not spec.checkpoint_path.is_file():
        _train_model(spec, device)
        pending = list(requested)
        resolved = []
    elif not pending:
        return resolved

    if pending:
        import torch

        model = _load_model(spec, device)
        for net_struct, remove_type in pending:
            _seed_everything(INFERENCE_SEED)
            data = _load_test_data(net_struct, remove_type)
            with torch.no_grad():
                record = _evaluate_model(model, *data)
            destination = spec.cache_path(net_struct, remove_type)
            from experiment_process.generate_resilience_inference_cache import archive
            archive(destination)
            _atomic_json(record, destination)
            from experiment_process.generate_resilience_inference_cache import stamp_dataset
            stamp_dataset(destination, data_root)
            validate_cache(destination)
            resolved.append(destination)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return resolved


def group_records(
    group: str,
    net_struct: str,
    remove_type: str,
    *,
    device: str = "cuda",
    force_cache: bool = False,
    force_train: bool = False,
) -> dict[str, dict[str, Any]]:
    if group not in PARAMETER_GROUPS:
        raise ValueError(f"Unknown parameter group: {group}")
    records: dict[str, dict[str, Any]] = {}
    for label, spec in PARAMETER_GROUPS[group]:
        ensure_run(
            spec,
            ((net_struct, remove_type),),
            device=device,
            force_cache=force_cache,
            force_train=force_train,
        )
        records[label] = validate_cache(
            spec.cache_path(net_struct, remove_type)
        )
    return records


def ensure_parameter_groups(
    groups: Iterable[str],
    *,
    device: str = "cuda",
    force_cache: bool = False,
    force_train: bool = False,
) -> None:
    requested_groups = tuple(dict.fromkeys(groups))
    unknown = set(requested_groups) - set(PARAMETER_GROUPS)
    if unknown:
        raise ValueError(f"Unknown parameter groups: {sorted(unknown)}")
    runs = {
        spec
        for group in requested_groups
        for _, spec in PARAMETER_GROUPS[group]
    }
    for spec in sorted(runs, key=lambda item: item.key):
        ensure_run(
            spec,
            device=device,
            force_cache=force_cache,
            force_train=force_train,
        )


def ensure_figure(
    figure: str,
    *,
    device: str = "cuda",
    force_cache: bool = False,
    force_train: bool = False,
) -> None:
    if figure not in FIGURE_GROUPS:
        raise ValueError(f"Unknown figure: {figure}")
    ensure_parameter_groups(
        FIGURE_GROUPS[figure],
        device=device,
        force_cache=force_cache,
        force_train=force_train,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", choices=tuple(FIGURE_GROUPS), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--force-train", action="store_true")
    args = parser.parse_args()
    ensure_figure(
        args.figure,
        device=args.device,
        force_cache=args.force_cache,
        force_train=args.force_train,
    )


if __name__ == "__main__":
    main()
