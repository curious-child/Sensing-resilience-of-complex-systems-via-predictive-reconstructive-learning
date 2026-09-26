"""Generate the inductive caches used by main Figures 2 and 3.

This module keeps the JSON schema and filenames used by the original project.
The analysis routine is copied here as a standalone implementation, so the
original experiment entrypoint is not imported. Existing checkpoints are only
evaluated; missing checkpoints are trained from the original configured data.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEVICE = "cuda"
RANDOM_SEED = 1029
OBSERVATIONS = 5
WINDOWS = 10

DYNAMICS = ("SIS", "Neuronal")
NETWORK_STRUCTURES = ("ER", "SF")
REMOVE_TYPES = ("degree", "random")
TRAINED_TASKS = ("multitask", "prediction", "reconstruction")
FIGURE2_TASKS = (*TRAINED_TASKS, "untrained")
ZLATENT_KEYS = (
    "Z_mean",
    "Z_var",
    "distance_mean",
    "distance_var",
    "distance_G",
    "distance_G_var",
)

Request = tuple[str, str, str, str]


def cache_path(dynamic: str, task: str, net_struct: str, remove_type: str) -> Path:
    """Return the canonical top-level inductive cache path."""

    analysis_dir = (
        PROJECT_ROOT / "experiment_results" / dynamic / "loss_inductive_analysis"
    )
    if task == "untrained":
        filename = f"multitask_{net_struct}_{remove_type}_record_train_False.json"
    else:
        filename = f"{task}_{net_struct}_{remove_type}_record_train_True.json"
    return analysis_dir / filename


def model_path(dynamic: str, task: str) -> Path:
    """Return the checkpoint used for a trained task or untrained architecture."""

    model_task = "multitask" if task == "untrained" else task
    return (
        PROJECT_ROOT
        / "experiment_results"
        / dynamic
        / "loss_inductive_analysis"
        / model_task
        / "model_trained"
    )


def dataset_paths(
    dynamic: str, net_struct: str, remove_type: str
) -> dict[str, Path]:
    base = PROJECT_ROOT / "dataset" / f"{dynamic}_test"
    return {
        "adjacency": base / f"As_{net_struct}_{remove_type}.pkl",
        "trajectory": base / f"numes_{net_struct}_{remove_type}.pkl",
        "stable_state": base / f"xlast_{net_struct}_{remove_type}.pkl",
    }


def validate_request(request: Request) -> None:
    dynamic, task, net_struct, remove_type = request
    if dynamic not in DYNAMICS:
        raise ValueError(f"Unsupported dynamics: {dynamic!r}")
    if task not in FIGURE2_TASKS:
        raise ValueError(f"Unsupported task: {task!r}")
    if net_struct not in NETWORK_STRUCTURES:
        raise ValueError(f"Unsupported network structure: {net_struct!r}")
    if remove_type not in REMOVE_TYPES:
        raise ValueError(f"Unsupported removal type: {remove_type!r}")


def _require_list(record: dict[str, Any], key: str, context: str) -> list[Any]:
    value = record.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{context}: {key!r} must be a list")
    return value


def validate_inductive_record(
    record: dict[str, Any],
    context: str = "cache",
    *,
    allow_legacy_missing_distance_g_var: bool = False,
) -> None:
    """Validate the legacy JSON schema and all plot-relevant sequence lengths."""

    for key in (
        "perform_record",
        "remove_nodes_scatter_record",
        "remove_nodes_record",
        "xlast_system_scatter_record",
    ):
        if key not in record:
            raise ValueError(f"{context}: missing top-level key {key!r}")

    line_x = _require_list(record, "remove_nodes_record", context)
    scatter_x = _require_list(record, "remove_nodes_scatter_record", context)
    scatter_y = _require_list(record, "xlast_system_scatter_record", context)
    if not line_x:
        raise ValueError(f"{context}: remove_nodes_record is empty")
    if len(scatter_x) != len(scatter_y):
        raise ValueError(
            f"{context}: scatter coordinates differ in length "
            f"({len(scatter_x)} != {len(scatter_y)})"
        )

    performance = record["perform_record"]
    if not isinstance(performance, dict):
        raise ValueError(f"{context}: perform_record must be an object")
    learn_error = performance.get("Learn_error")
    zlatent = performance.get("Zlatent")
    if not isinstance(learn_error, dict) or not isinstance(zlatent, dict):
        raise ValueError(f"{context}: Learn_error and Zlatent must be objects")

    expected = len(line_x)
    for key in ("ts_err", "graph_err"):
        values = _require_list(learn_error, key, f"{context}.Learn_error")
        if len(values) != expected:
            raise ValueError(
                f"{context}: Learn_error.{key} has {len(values)} values; "
                f"expected {expected}"
            )
    for key in ZLATENT_KEYS:
        if (
            key == "distance_G_var"
            and key not in zlatent
            and allow_legacy_missing_distance_g_var
        ):
            warnings.warn(
                f"{context}: accepting legacy cache without distance_G_var; "
                "this field is not used by Figures 2 or 3",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        values = _require_list(zlatent, key, f"{context}.Zlatent")
        if len(values) != expected:
            raise ValueError(
                f"{context}: Zlatent.{key} has {len(values)} values; "
                f"expected {expected}"
            )


def load_inductive_cache(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        record = json.load(stream)
    validate_inductive_record(
        record, str(path), allow_legacy_missing_distance_g_var=True
    )
    return record


def _preflight(requests: Iterable[Request], device: str) -> None:
    errors: list[str] = []
    if device.startswith("cuda"):
        try:
            import torch

            if not torch.cuda.is_available():
                errors.append(
                    f"DEVICE={device!r}, but CUDA is unavailable; set DEVICE='cpu'"
                )
        except ImportError as exc:
            errors.append(f"PyTorch cannot be imported: {exc}")

    checked: set[Path] = set()
    for request in requests:
        validate_request(request)
        dynamic, task, net_struct, remove_type = request
        required = [
            model_path(dynamic, task).with_suffix(".yaml"),
            *dataset_paths(dynamic, net_struct, remove_type).values(),
        ]
        for path in required:
            if path in checked:
                continue
            checked.add(path)
            if not path.is_file():
                errors.append(f"Missing required input: {path}")

    if errors:
        details = "\n".join(f"  - {error}" for error in errors)
        raise FileNotFoundError(
            "Inductive cache generation cannot start:\n" + details
        )


def _load_dataset(
    dynamic: str, net_struct: str, remove_type: str
) -> tuple[list[Any], list[Any], list[Any]]:
    paths = dataset_paths(dynamic, net_struct, remove_type)
    with paths["adjacency"].open("rb") as stream:
        adjacency = pickle.load(stream)
    with paths["trajectory"].open("rb") as stream:
        trajectories = pickle.load(stream)
    with paths["stable_state"].open("rb") as stream:
        stable_states = pickle.load(stream)
    if not adjacency:
        raise ValueError(f"Empty dataset: {dynamic}/{net_struct}/{remove_type}")
    if len({len(adjacency), len(trajectories), len(stable_states)}) != 1:
        raise ValueError(
            f"Dataset length mismatch: {dynamic}/{net_struct}/{remove_type}"
        )
    return adjacency, trajectories, stable_states


def _set_random_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _atomic_json_dump(record: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(record, stream, indent=4, separators=(",", ":"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _model_data_analysis(
    resilience_model: Any,
    adjacency_list: list[Any],
    trajectory_list: list[Any],
    stable_state_list: list[Any],
    *,
    observations: int,
    windows: int,
    original_node_count: int,
) -> tuple[dict[str, Any], list[int], list[int], list[float]]:
    """Standalone copy of the original inductive evaluation calculation."""

    import torch
    import torch_geometric.utils as tg_utils
    from torch_geometric.data import Data

    performance = {
        "Learn_error": {"ts_err": [], "graph_err": []},
        "Zlatent": {
            "Z_mean": [],
            "Z_var": [],
            "distance_mean": [],
            "distance_var": [],
            "distance_G": [],
            "distance_G_var": [],
        },
    }
    remove_nodes_scatter_record: list[int] = []
    remove_nodes_record: list[int] = []
    xlast_system_scatter_record: list[float] = []
    original_graph_embedding = None

    for adjacency, trajectory, stable_state in zip(
        adjacency_list, trajectory_list, stable_state_list
    ):
        adjacency_tensor = torch.tensor(adjacency)
        edge_index = tg_utils.dense_to_sparse(adjacency_tensor)[0]
        if edge_index.shape[1] <= 0:
            break

        current_nodes = adjacency_tensor.shape[0]
        removed_nodes = original_node_count - current_nodes
        remove_nodes_record.append(removed_nodes)

        trajectory_tensor = torch.tensor(trajectory)
        stable_state_tensor = torch.tensor(stable_state)
        for observation_state in stable_state_tensor:
            if observation_state.shape[0] != current_nodes:
                raise ValueError(
                    "Stable-state node count does not match adjacency size"
                )
            xlast_system_scatter_record.append(
                observation_state.mean().item()
            )
            remove_nodes_scatter_record.append(removed_nodes)

        trajectory_tensor = (
            trajectory_tensor.permute(2, 0, 1).unsqueeze(-1).float()
        )
        observation_count = trajectory_tensor.shape[1]
        if observation_count < observations:
            raise ValueError(
                f"Requested {observations} observations, but only "
                f"{observation_count} are available"
            )
        if trajectory_tensor.shape[2] < 2 * windows:
            raise ValueError(
                f"Trajectory length {trajectory_tensor.shape[2]} is shorter "
                f"than the required {2 * windows}"
            )
        sampled = random.sample(range(observation_count), observations)
        sampled_trajectory = trajectory_tensor[:, sampled, :, :]
        graph_data = Data(
            x=sampled_trajectory[:, :, :windows, :],
            edge_index=edge_index,
            y=sampled_trajectory[:, :, windows : 2 * windows, :],
        ).to(resilience_model.device)

        prediction_error, reconstruction_error = (
            resilience_model.eval_pred_error(gdata=graph_data)
        )
        z_mean, z_variance, node_embedding = (
            resilience_model.eval_Zlatent(gdata=graph_data)
        )
        pairwise_distance = torch.pdist(node_embedding, p=2)

        if original_graph_embedding is None:
            original_graph_embedding = node_embedding.mean(dim=0)
            graph_distance = 0
            graph_embedding_variance = (
                original_graph_embedding.var().cpu().detach().item()
            )
        else:
            current_graph_embedding = node_embedding.mean(dim=0)
            graph_distance = (
                torch.norm(
                    current_graph_embedding - original_graph_embedding,
                    p=2,
                    dim=-1,
                )
                .cpu()
                .detach()
                .item()
            )
            graph_embedding_variance = (
                current_graph_embedding.var().cpu().detach().item()
            )

        performance["Learn_error"]["ts_err"].append(
            prediction_error.cpu().detach().item()
        )
        performance["Learn_error"]["graph_err"].append(
            reconstruction_error.cpu().detach().item()
        )
        performance["Zlatent"]["Z_mean"].append(
            z_mean.cpu().detach().item()
        )
        performance["Zlatent"]["Z_var"].append(
            z_variance.cpu().detach().item()
        )
        performance["Zlatent"]["distance_mean"].append(
            pairwise_distance.mean().cpu().detach().item()
        )
        performance["Zlatent"]["distance_var"].append(
            pairwise_distance.var().cpu().detach().item()
        )
        performance["Zlatent"]["distance_G"].append(graph_distance)
        performance["Zlatent"]["distance_G_var"].append(
            graph_embedding_variance
        )

    return (
        performance,
        remove_nodes_scatter_record,
        remove_nodes_record,
        xlast_system_scatter_record,
    )


def _generate_one(request: Request, device: str) -> Path:
    import torch

    from utils.utils import load_Resilience_model

    dynamic, task, net_struct, remove_type = request
    adjacency, trajectories, stable_states = _load_dataset(
        dynamic, net_struct, remove_type
    )
    _set_random_seed(RANDOM_SEED)
    trained = task != "untrained"
    if not trained and (not model_path(dynamic, task).is_file()
                        or model_path(dynamic, task).stat().st_size == 0):
        import yaml
        from models.models import Resilience_model
        from experiment_process.generate_model_parameter_analysis_cache import _build_net_parameters
        with model_path(dynamic, task).with_suffix(".yaml").open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        model = Resilience_model(_build_net_parameters(config, 0, device)).to(device)
    else:
        model, _ = load_Resilience_model(
            str(model_path(dynamic, task)), device, load_state=trained,
            infer_para={"model_task": "AE_recon"})
    model.eval()
    with torch.no_grad():
        performance, scatter_x, line_x, scatter_y = _model_data_analysis(
            model,
            adjacency,
            trajectories,
            stable_states,
            observations=OBSERVATIONS,
            windows=WINDOWS,
            original_node_count=adjacency[0].shape[0],
        )
    record = {
        "perform_record": performance,
        "remove_nodes_scatter_record": scatter_x,
        "remove_nodes_record": line_x,
        "xlast_system_scatter_record": scatter_y,
    }
    destination = cache_path(dynamic, task, net_struct, remove_type)
    validate_inductive_record(record, str(destination))
    _atomic_json_dump(record, destination)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"Generated: {destination}")
    return destination


def ensure_inductive_caches(
    requests: Iterable[Request],
    *,
    force: bool = False,
    device: str = DEVICE,
) -> list[Path]:
    """Validate existing caches and generate only absent/forced requests."""

    unique_requests = list(dict.fromkeys(requests))
    pending: list[Request] = []
    resolved: list[Path] = []
    for request in unique_requests:
        validate_request(request)
        path = cache_path(*request)
        from experiment_process.generate_resilience_inference_cache import cache_matches_dataset
        data_root = Path(__file__).resolve().parents[1] / "dataset" / f"{request[0]}_test"
        if path.is_file() and not force and cache_matches_dataset(path, data_root):
            try:
                load_inductive_cache(path)
                resolved.append(path)
                continue
            except (ValueError, KeyError, TypeError):
                pass
        pending.append(request)

    if pending:
        _preflight(pending, device)
        from experiment_process.generate_model_parameter_analysis_cache import ensure_prism_checkpoint
        for dynamic, task in dict.fromkeys((r[0], r[1]) for r in pending):
            if task != "untrained":
                ensure_prism_checkpoint(model_path(dynamic, task), device, task=task)
        for request in pending:
            from experiment_process.generate_resilience_inference_cache import archive
            archive(cache_path(*request))
            generated = _generate_one(request, device)
            from experiment_process.generate_resilience_inference_cache import stamp_dataset
            stamp_dataset(generated, Path(__file__).resolve().parents[1] / "dataset" / f"{request[0]}_test")
            resolved.append(generated)
    return resolved


def all_requests(tasks: Iterable[str] = FIGURE2_TASKS) -> list[Request]:
    return [
        (dynamic, task, net_struct, remove_type)
        for dynamic in DYNAMICS
        for net_struct in NETWORK_STRUCTURES
        for remove_type in REMOVE_TYPES
        for task in tasks
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dynamic", choices=(*DYNAMICS, "all"), default="all")
    parser.add_argument("--task", choices=(*FIGURE2_TASKS, "all"), default="all")
    parser.add_argument(
        "--network", choices=(*NETWORK_STRUCTURES, "all"), default="all"
    )
    parser.add_argument("--remove", choices=(*REMOVE_TYPES, "all"), default="all")
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    dynamics = DYNAMICS if args.dynamic == "all" else (args.dynamic,)
    tasks = FIGURE2_TASKS if args.task == "all" else (args.task,)
    networks = (
        NETWORK_STRUCTURES if args.network == "all" else (args.network,)
    )
    remove_types = REMOVE_TYPES if args.remove == "all" else (args.remove,)
    requests = [
        (dynamic, task, network, remove_type)
        for dynamic in dynamics
        for network in networks
        for remove_type in remove_types
        for task in tasks
    ]
    paths = ensure_inductive_caches(
        requests, force=args.force, device=args.device
    )
    print(f"Ready: {len(paths)} inductive cache file(s)")


if __name__ == "__main__":
    main()
