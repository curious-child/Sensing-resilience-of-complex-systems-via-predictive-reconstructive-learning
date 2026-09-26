"""Generate the latent-representation caches used by manuscript Figure 5.

The implementation is self-contained with respect to the two legacy experiment
entry points.  It copies their graph-level statistical pooling and t-SNE logic,
loads the existing frozen PRISM checkpoints, and understands the updated
``three_class_*.pkl``/``binary_*.pkl`` Neuronal label convention.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import pickle
import random
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "experiment_results"
DATASET_ROOT = PROJECT_ROOT / "dataset"
SCHEMA_VERSION = 1
RANDOM_SEED = 42
OBSERVATIONS = 5
WINDOWS = 10
MAX_SAMPLE_NUM = 1
BATCH_SIZE = 8
TSNE_PERPLEXITY = 30.0
TSNE_ITERATIONS = 1000
POOLING_STATS = ("mean", "std")


@dataclass(frozen=True)
class Figure5Task:
    key: str
    dynamics: str
    dataset_dir: Path
    label_name: str
    classes: int
    components: tuple[int, ...]

    @property
    def analysis_dir(self) -> Path:
        return RESULTS_ROOT / self.dynamics / "resilience_inference_analysis"

    @property
    def checkpoint(self) -> Path:
        return self.analysis_dir / "model_trained"

    @property
    def cache_dir(self) -> Path:
        return self.analysis_dir / "figure5_tsne"

    @property
    def cache_path(self) -> Path:
        return self.cache_dir / f"{self.key}_tsne.npz"

    @property
    def metadata_path(self) -> Path:
        return self.cache_dir / f"{self.key}_tsne.json"

    @property
    def split_dir(self) -> Path:
        return self.dataset_dir / "train_dataset"


TASKS = {
    "sis_binary": Figure5Task(
        key="sis_binary",
        dynamics="SIS",
        dataset_dir=DATASET_ROOT / "SIS_supervised",
        label_name="rs.pkl",
        classes=2,
        components=(2,),
    ),
    "neuronal_three_class": Figure5Task(
        key="neuronal_three_class",
        dynamics="Neuronal",
        dataset_dir=DATASET_ROOT / "neuronal_supervised",
        label_name="three_class_rs.pkl",
        classes=3,
        components=(2, 3),
    ),
}


def _source_files(task: Figure5Task) -> tuple[Path, ...]:
    return (
        task.split_dir / "As.pkl",
        task.split_dir / "numes.pkl",
        task.split_dir / task.label_name,
        task.checkpoint,
    )


def _file_signature(path: Path) -> dict[str, int | str]:
    stat = path.stat()
    return {
        "path": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _source_signatures(task: Figure5Task) -> list[dict[str, int | str]]:
    return [_file_signature(path) for path in _source_files(task)]


def missing_inputs(task: Figure5Task) -> list[Path]:
    return [path for path in _source_files(task) if not path.is_file()]


def _atomic_json(payload: dict, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.stem}_", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False)
        os.replace(temporary_name, target)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _atomic_npz(arrays: dict[str, np.ndarray], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.stem}_", suffix=".npz", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temporary_name, target)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _resolve_device(requested: str) -> str:
    import torch

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
    return requested


def _load_prism(task: Figure5Task, device: str):
    """Copy the checkpoint-loading logic used by the legacy experiments."""
    import torch

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from models.models import Resilience_model

    state = torch.load(task.checkpoint, map_location="cpu")
    net_param = dict(state["net_param"])
    net_param.update({"model_task": "AE_recon", "device": device})
    model = Resilience_model(net_param=net_param).to(device)
    model_keys = set(model.state_dict())
    weights = state["state_dict"]
    if weights and not set(weights).issubset(model_keys):
        stripped = {key.removeprefix("module."): value for key, value in weights.items()}
        if set(stripped).issubset(model_keys):
            weights = stripped
    model.load_state_dict(weights, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def _load_supervised_graphs(task: Figure5Task):
    """Copy ``pred_DataSet_supervised`` with an explicit label filename."""
    import torch
    import torch_geometric.utils as pyg_utils
    from torch_geometric.data import Data

    with (task.split_dir / "As.pkl").open("rb") as stream:
        adjacency_list = pickle.load(stream)
    with (task.split_dir / "numes.pkl").open("rb") as stream:
        trajectory_list = pickle.load(stream)
    with (task.split_dir / task.label_name).open("rb") as stream:
        labels = pickle.load(stream)

    lengths = (len(adjacency_list), len(trajectory_list), len(labels))
    if len(set(lengths)) != 1:
        raise ValueError(
            f"{task.key}: adjacency/trajectory/label lengths differ: {lengths}"
        )

    rng = random.Random(RANDOM_SEED)
    graphs = []
    skipped_small = 0
    for adjacency, trajectory, label in zip(
        adjacency_list, trajectory_list, labels
    ):
        adjacency_tensor = torch.as_tensor(adjacency)
        if adjacency_tensor.shape[0] <= 20:
            skipped_small += 1
            continue
        trajectory_tensor = torch.as_tensor(trajectory)
        if trajectory_tensor.shape[2] != adjacency_tensor.shape[0]:
            raise ValueError("Trajectory node dimension does not match adjacency")
        if trajectory_tensor.shape[1] < 2 * WINDOWS:
            raise ValueError("Trajectory window is shorter than 2 * WINDOWS")
        if trajectory_tensor.shape[0] < OBSERVATIONS:
            raise ValueError("Not enough observations in trajectory sample")

        label_int = int(np.asarray(label).reshape(-1)[0])
        if not 0 <= label_int < task.classes:
            raise ValueError(
                f"{task.key}: label {label_int} is outside 0..{task.classes - 1}"
            )
        edge_index = pyg_utils.dense_to_sparse(adjacency_tensor)[0]
        trajectory_tensor = (
            trajectory_tensor.permute(2, 0, 1).unsqueeze(-1).float()
        )
        number_observations = trajectory_tensor.shape[1]
        total = math.comb(number_observations, OBSERVATIONS)
        if total <= MAX_SAMPLE_NUM:
            observation_sets: Iterable[tuple[int, ...]] = itertools.combinations(
                range(number_observations), OBSERVATIONS
            )
        else:
            observation_sets = [
                tuple(sorted(rng.sample(range(number_observations), OBSERVATIONS)))
            ]
        for observation_indices in itertools.islice(
            observation_sets, MAX_SAMPLE_NUM
        ):
            sampled = trajectory_tensor[:, observation_indices, :, :]
            graphs.append(
                Data(
                    x=sampled[:, :, :WINDOWS, :],
                    edge_index=edge_index,
                    y=torch.tensor([label_int], dtype=torch.long),
                    num_nodes=int(adjacency_tensor.shape[0]),
                )
            )
    if not graphs:
        raise ValueError(f"{task.key}: no valid graphs were loaded")
    return graphs, skipped_small


def _statistical_pool(node_features, batch_index):
    """Graph mean and unbiased standard deviation copied from KNN scripts."""
    import torch
    import torch_geometric.nn as pyg_nn

    mean = pyg_nn.global_mean_pool(node_features, batch_index)
    expanded = mean[batch_index]
    variance = pyg_nn.global_mean_pool(
        (node_features - expanded).pow(2), batch_index
    )
    graph_count = int(batch_index.max().item()) + 1
    for graph_index in range(graph_count):
        count = int((batch_index == graph_index).sum().item())
        if count > 1:
            variance[graph_index] = variance[graph_index] * count / (count - 1)
    standard_deviation = torch.sqrt(variance + 1e-8)
    return torch.cat((mean, standard_deviation), dim=1)


def _extract_embeddings(task: Figure5Task, device: str):
    import torch
    from torch_geometric.loader import DataLoader

    graphs, skipped_small = _load_supervised_graphs(task)
    model = _load_prism(task, device)
    loader = DataLoader(graphs, batch_size=BATCH_SIZE, shuffle=False)
    embeddings: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            node_features = model.get_zlatent(batch)
            pooled = _statistical_pool(node_features, batch.batch)
            embeddings.append(pooled.detach().cpu().numpy())
            labels.append(batch.y.detach().cpu().numpy().reshape(-1))
    matrix = np.concatenate(embeddings, axis=0).astype(np.float32, copy=False)
    label_array = np.concatenate(labels, axis=0).astype(np.int64, copy=False)
    if not np.isfinite(matrix).all():
        raise ValueError(f"{task.key}: extracted embeddings contain non-finite values")
    return matrix, label_array, skipped_small


def _run_tsne(embeddings: np.ndarray, components: int) -> np.ndarray:
    from sklearn.manifold import TSNE

    if len(embeddings) <= TSNE_PERPLEXITY:
        raise ValueError(
            f"t-SNE requires more than {TSNE_PERPLEXITY:g} samples; got {len(embeddings)}"
        )
    projector = TSNE(
        n_components=components,
        perplexity=TSNE_PERPLEXITY,
        n_iter=TSNE_ITERATIONS,
        random_state=RANDOM_SEED,
        init="pca",
        learning_rate="auto",
    )
    return projector.fit_transform(embeddings).astype(np.float32, copy=False)


def validate_cache(task: Figure5Task, *, require_current: bool = True) -> Path:
    if not task.cache_path.is_file() or not task.metadata_path.is_file():
        raise FileNotFoundError(task.cache_path)
    with task.metadata_path.open("r", encoding="utf-8") as stream:
        metadata = json.load(stream)
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported cache schema: {task.metadata_path}")
    if require_current and metadata.get("source_files") != _source_signatures(task):
        raise ValueError(f"Source files changed after cache creation: {task.cache_path}")
    with np.load(task.cache_path, allow_pickle=False) as cached:
        required = {"graph_embeddings", "labels"} | {
            f"tsne_{components}d" for components in task.components
        }
        missing = required - set(cached.files)
        if missing:
            raise ValueError(f"{task.cache_path} is missing arrays: {sorted(missing)}")
        labels = cached["labels"]
        sample_count = len(labels)
        if sample_count == 0:
            raise ValueError(f"{task.cache_path} contains no samples")
        for key in required - {"labels"}:
            values = cached[key]
            if len(values) != sample_count or not np.isfinite(values).all():
                raise ValueError(f"Invalid {key} in {task.cache_path}")
        observed = sorted(np.unique(labels).astype(int).tolist())
        if observed != list(range(task.classes)):
            raise ValueError(
                f"{task.cache_path} labels are {observed}; expected 0..{task.classes - 1}"
            )
    return task.cache_path


def generate_cache(task: Figure5Task, device: str) -> Path:
    missing = missing_inputs(task)
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing Figure 5 inputs for {task.key}:\n{formatted}")
    print(f"[{task.key}] loading graphs and extracting PRISM embeddings...")
    embeddings, labels, skipped_small = _extract_embeddings(task, device)
    arrays = {"graph_embeddings": embeddings, "labels": labels}
    for components in task.components:
        print(f"[{task.key}] fitting {components}D t-SNE...")
        arrays[f"tsne_{components}d"] = _run_tsne(embeddings, components)
    counts = {
        str(label): int((labels == label).sum()) for label in range(task.classes)
    }
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "figure": "Figure 5",
        "task": task.key,
        "dynamics": task.dynamics,
        "dataset_split": "train_dataset",
        "label_file": task.label_name,
        "class_count": task.classes,
        "class_samples": counts,
        "sample_count": int(len(labels)),
        "skipped_graphs_with_at_most_20_nodes": int(skipped_small),
        "embedding_dim": int(embeddings.shape[1]),
        "pooling_stats": list(POOLING_STATS),
        "random_seed": RANDOM_SEED,
        "observations": OBSERVATIONS,
        "windows": WINDOWS,
        "max_sample_num": MAX_SAMPLE_NUM,
        "tsne": {
            "components": list(task.components),
            "perplexity": TSNE_PERPLEXITY,
            "n_iter": TSNE_ITERATIONS,
            "init": "pca",
            "learning_rate": "auto",
        },
        "source_files": _source_signatures(task),
    }
    from experiment_process.generate_resilience_inference_cache import archive
    archive(task.cache_path)
    archive(task.metadata_path)
    _atomic_npz(arrays, task.cache_path)
    _atomic_json(metadata, task.metadata_path)
    validate_cache(task)
    print(f"[{task.key}] saved {task.cache_path}")
    return task.cache_path


def ensure_figure5_caches(
    *, device: str = "auto", force: bool = False
) -> dict[str, Path]:
    resolved: dict[str, Path] = {}
    for key, task in TASKS.items():
        regenerate = force
        if not regenerate:
            try:
                from experiment_process.generate_resilience_inference_cache import dataset_versions
                resolved[key] = validate_cache(task, require_current=bool(dataset_versions(task.split_dir)))
                continue
            except (FileNotFoundError, ValueError) as error:
                print(f"[{key}] cache unavailable or stale: {error}")
                regenerate = True
        if regenerate:
            resolved_device = _resolve_device(device)
            from experiment_process.generate_model_parameter_analysis_cache import ensure_prism_checkpoint
            ensure_prism_checkpoint(task.checkpoint, resolved_device)
            resolved[key] = generate_cache(task, resolved_device)
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--force", action="store_true", help="regenerate valid caches")
    parser.add_argument(
        "--task",
        choices=("all", *TASKS),
        default="all",
        help="generate one cache or both",
    )
    args = parser.parse_args()
    device = _resolve_device(args.device)
    selected = TASKS.values() if args.task == "all" else (TASKS[args.task],)
    for task in selected:
        if args.force:
            generate_cache(task, device)
        else:
            try:
                path = validate_cache(task)
                print(f"[{task.key}] valid cache: {path}")
            except (FileNotFoundError, ValueError) as error:
                print(f"[{task.key}] cache unavailable or stale: {error}")
                generate_cache(task, device)


if __name__ == "__main__":
    main()
