"""Generate the transductive caches used by Supplementary Figure 1.

The per-graph analysis and optimization routines are copied here as standalone
implementations. The original experiment entrypoint is not imported. Every task
starts from the same deterministic random weights.
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
TASKS = ("multitask", "prediction", "reconstruction", "untrained")
TASK_CONDITIONS = {
    "prediction": 0,
    "reconstruction": 1,
    "multitask": 2,
    "untrained": None,
}

Request = tuple[str, str, str, str]


def cache_path(dynamic: str, task: str, net_struct: str, remove_type: str) -> Path:
    return (
        PROJECT_ROOT
        / "experiment_results"
        / dynamic
        / "loss_transductive_analysis"
        / f"{net_struct}_{remove_type}_record_{task}.json"
    )


def config_path(dynamic: str) -> Path:
    return (
        PROJECT_ROOT
        / "experiment_results"
        / dynamic
        / "loss_transductive_analysis"
        / "model_trained.yaml"
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
    if task not in TASKS:
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


def validate_transductive_record(
    record: dict[str, Any],
    context: str = "cache",
    *,
    allow_legacy_untrained_iterations: bool = False,
) -> None:
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
    expected = len(line_x)
    for key in ("ts_err", "graph_err"):
        values = _require_list(performance, key, f"{context}.perform_record")
        if len(values) != expected:
            raise ValueError(
                f"{context}: {key} has {len(values)} values; expected {expected}"
            )
    iterations = performance.get("num_iter")
    if not isinstance(iterations, list):
        if allow_legacy_untrained_iterations and (
            iterations is None or isinstance(iterations, (int, float))
        ):
            warnings.warn(
                f"{context}: accepting legacy scalar/null num_iter; this field "
                "is not used for Supplementary Figure 1",
                RuntimeWarning,
                stacklevel=2,
            )
            return
        raise ValueError(f"{context}: num_iter must be a list")
    if len(iterations) != expected:
        if allow_legacy_untrained_iterations and len(iterations) == 1:
            warnings.warn(
                f"{context}: accepting legacy singleton num_iter list; this "
                "field is not used for Supplementary Figure 1",
                RuntimeWarning,
                stacklevel=2,
            )
            return
        raise ValueError(
            f"{context}: num_iter has {len(iterations)} values; expected {expected}"
        )


def load_transductive_cache(
    path: Path, *, task: str | None = None
) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        record = json.load(stream)
    validate_transductive_record(
        record,
        str(path),
        allow_legacy_untrained_iterations=task == "untrained",
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
        dynamic, _, net_struct, remove_type = request
        required = [
            config_path(dynamic),
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
            "Transductive cache generation cannot start:\n" + details
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


def _optimize_single_sample(
    initial_states: dict[str, Any],
    model: Any,
    train_data: Any,
    task_condition: Any,
    *,
    max_iter: int = 100,
    convergence_window: int = 10,
    patience: int = 5,
) -> tuple[float, float, int]:
    """Standalone copy of the original single-graph optimization routine."""

    import numpy as np
    import torch

    model.load_state_dict(initial_states)
    train_data = train_data.to(model.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=model.lr)
    loss_history: list[float] = []
    prediction_errors: list[float] = []
    reconstruction_errors: list[float] = []
    patience_counter = 0

    for iteration in range(max_iter):
        converged = False
        model.train()
        loss_list, task_train = model.training_step_onegraph(
            gdata=train_data, task_cond=task_condition
        )
        prediction_errors.append(loss_list[0].cpu().detach().item())
        reconstruction_errors.append(loss_list[1].cpu().detach().item())
        loss = loss_list[task_train]
        if torch.isnan(loss).any():
            print("loss is NaN; skipping this optimization step")
            continue

        current_loss = loss.cpu().detach().item()
        loss_history.append(current_loss)
        if iteration >= convergence_window:
            recent_losses = loss_history[-convergence_window:]
            loss_std = np.std(recent_losses)
            loss_mean = np.mean(recent_losses)
            if loss_mean < 1e-4:
                converged = loss_std < 1e-4
            else:
                converged = loss_std / loss_mean < 0.01

        if converged:
            patience_counter += 1
        else:
            patience_counter = 0
        if patience_counter > patience:
            break

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if not prediction_errors or not reconstruction_errors:
        raise RuntimeError("No valid optimization errors were produced")
    prediction_mean = float(
        np.mean(prediction_errors[-convergence_window:])
    )
    reconstruction_mean = float(
        np.mean(reconstruction_errors[-convergence_window:])
    )
    iterations = min(len(loss_history), max_iter)
    return prediction_mean, reconstruction_mean, iterations


def _model_data_analysis(
    task_condition: Any,
    resilience_model: Any,
    adjacency_list: list[Any],
    trajectory_list: list[Any],
    stable_state_list: list[Any],
    *,
    observations: int,
    windows: int,
    original_node_count: int,
) -> tuple[dict[str, Any], list[int], list[int], list[float]]:
    """Standalone copy of the original transductive evaluation calculation."""

    import copy

    import torch
    import torch_geometric.utils as tg_utils
    from torch_geometric.data import Data

    performance = {"ts_err": [], "graph_err": [], "num_iter": []}
    remove_nodes_scatter_record: list[int] = []
    remove_nodes_record: list[int] = []
    xlast_system_scatter_record: list[float] = []
    initial_states = copy.deepcopy(resilience_model.state_dict())

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

        if task_condition is not None:
            prediction_error, reconstruction_error, iterations = (
                _optimize_single_sample(
                    initial_states,
                    resilience_model,
                    graph_data,
                    task_condition,
                )
            )
        else:
            prediction_tensor, reconstruction_tensor = (
                resilience_model.eval_pred_error(gdata=graph_data)
            )
            prediction_error = prediction_tensor.cpu().detach().item()
            reconstruction_error = (
                reconstruction_tensor.cpu().detach().item()
            )
            iterations = 0

        performance["ts_err"].append(prediction_error)
        performance["graph_err"].append(reconstruction_error)
        performance["num_iter"].append(iterations)

    return (
        performance,
        remove_nodes_scatter_record,
        remove_nodes_record,
        xlast_system_scatter_record,
    )


def _generate_one(request: Request, device: str) -> Path:
    import torch
    import yaml

    from models.pred_model.AE_recon_model import AE_recon_model

    dynamic, task, net_struct, remove_type = request
    adjacency, trajectories, stable_states = _load_dataset(
        dynamic, net_struct, remove_type
    )
    with config_path(dynamic).open("r", encoding="utf-8") as stream:
        net_params = yaml.safe_load(stream)
    if not isinstance(net_params, dict):
        raise ValueError(f"Invalid model configuration: {config_path(dynamic)}")
    net_params = dict(net_params)
    net_params["device"] = device

    # Re-seeding for every task makes all four task models start identically.
    # The local analysis routine restores that state before every graph snapshot.
    _set_random_seed(RANDOM_SEED)
    model = AE_recon_model(net_params).to(device)
    condition_value = TASK_CONDITIONS[task]
    task_condition = (
        None
        if condition_value is None
        else torch.tensor([condition_value], device=model.device)
    )
    performance, scatter_x, line_x, scatter_y = _model_data_analysis(
        task_condition,
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
    validate_transductive_record(record, str(destination))
    _atomic_json_dump(record, destination)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"Generated: {destination}")
    return destination


def ensure_transductive_caches(
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
        dynamic, task, net_struct, remove_type = request
        path = cache_path(dynamic, task, net_struct, remove_type)
        from experiment_process.generate_resilience_inference_cache import cache_matches_dataset
        data_root = Path(__file__).resolve().parents[1] / "dataset" / f"{request[0]}_test"
        if path.is_file() and not force and cache_matches_dataset(path, data_root):
            try:
                load_transductive_cache(path, task=task)
                resolved.append(path)
                continue
            except (ValueError, KeyError, TypeError):
                pass
        pending.append(request)

    if pending:
        _preflight(pending, device)
        for request in pending:
            from experiment_process.generate_resilience_inference_cache import archive
            archive(cache_path(*request))
            generated = _generate_one(request, device)
            from experiment_process.generate_resilience_inference_cache import stamp_dataset
            stamp_dataset(generated, Path(__file__).resolve().parents[1] / "dataset" / f"{request[0]}_test")
            resolved.append(generated)
    return resolved


def all_requests() -> list[Request]:
    return [
        (dynamic, task, net_struct, remove_type)
        for dynamic in DYNAMICS
        for net_struct in NETWORK_STRUCTURES
        for remove_type in REMOVE_TYPES
        for task in TASKS
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dynamic", choices=(*DYNAMICS, "all"), default="all")
    parser.add_argument("--task", choices=(*TASKS, "all"), default="all")
    parser.add_argument(
        "--network", choices=(*NETWORK_STRUCTURES, "all"), default="all"
    )
    parser.add_argument("--remove", choices=(*REMOVE_TYPES, "all"), default="all")
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    dynamics = DYNAMICS if args.dynamic == "all" else (args.dynamic,)
    tasks = TASKS if args.task == "all" else (args.task,)
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
    paths = ensure_transductive_caches(
        requests, force=args.force, device=args.device
    )
    print(f"Ready: {len(paths)} transductive cache file(s)")


if __name__ == "__main__":
    main()
