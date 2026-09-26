"""Paper inference caches: reuse results/weights and train missing MLP or ResInf models."""
from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
import os
import pickle
import random
import shutil

import torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool, global_max_pool
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "experiment_results"
DATASET_ROOT = PROJECT_ROOT / "dataset"
RATIOS = (0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.0)
NUM_TRIALS = 10
BATCH_SIZE = int(os.environ.get("PRISM_BATCH_SIZE", "8"))
TRAIN_EPOCHS = 50
if BATCH_SIZE < 1:
    raise ValueError("PRISM_BATCH_SIZE must be positive")


@dataclass(frozen=True)
class TaskSpec:
    key: str
    dynamics: str
    classes: int
    dataset_name: str
    pooling_stats: tuple[str, ...]
    seed: int
    prefix: str
    label_file: str
    baseline_file: str

    @property
    def analysis_dir(self) -> Path:
        return RESULTS_ROOT / self.dynamics / "resilience_inference_analysis"

    @property
    def dataset_dir(self) -> Path:
        return DATASET_ROOT / self.dataset_name

    @property
    def prism_checkpoint(self) -> Path:
        return self.analysis_dir / "model_trained"

    @property
    def prism_config(self) -> Path:
        return self.analysis_dir / "model_trained.yaml"


TASKS: Mapping[str, TaskSpec] = {
    "sis_binary": TaskSpec(
        "sis_binary",
        "SIS",
        2,
        "SIS_supervised",
        ("std",),
        42,
        "binary",
        "rs.pkl",
        "basers.pkl",
    ),
    "neuronal_three_class": TaskSpec(
        "neuronal_three_class",
        "Neuronal",
        3,
        "neuronal_supervised",
        ("max", "min", "mean", "std"),
        233,
        "three_class",
        "three_class_rs.pkl",
        "three_class_basers.pkl",
    ),
    "neuronal_binary": TaskSpec(
        "neuronal_binary",
        "Neuronal",
        2,
        "neuronal_supervised",
        ("max", "min", "mean", "std"),
        42,
        "binary",
        "binary_rs.pkl",
        "binary_basers.pkl",
    ),
}

METHODS = ("mlp", "knn", "resinf")


def canonical_csv(spec: TaskSpec, method: str) -> Path:
    suffix = {
        "mlp": "mlp_multisample.csv",
        "knn": "knn_gbb_k4_distance_multisample.csv",
        "resinf": "resinf_multisample.csv",
    }[method]
    return spec.analysis_dir / f"{spec.prefix}_{suffix}"


def canonical_trials(spec: TaskSpec, method: str) -> Path:
    return spec.analysis_dir / f"{spec.prefix}_{method}_trials.json"


def classifier_model_root(spec: TaskSpec, method: str) -> Path:
    """Return the single canonical checkpoint root for a task and classifier."""
    return spec.analysis_dir / "classifier_models" / spec.prefix / method


def model_dir(spec: TaskSpec, method: str, ratio: float, trial: int) -> Path:
    return classifier_model_root(spec, method) / (
        f"ratio_{ratio:.3f}_sample_{trial:02d}"
    )


def trained_model_path(spec: TaskSpec, method: str, ratio: float, trial: int) -> Path:
    if method not in ("mlp", "resinf"):
        raise ValueError("KNN has no model checkpoint")
    return model_dir(spec, method, ratio, trial) / "model.pth"


LEGACY_CSV: Mapping[tuple[str, str], Path] = {
    ("sis_binary", "mlp"): TASKS["sis_binary"].analysis_dir
    / "model_comparison_multisample.csv",
    ("sis_binary", "knn"): TASKS["sis_binary"].analysis_dir
    / "KNN_classifier_multisample_std"
    / "KNN_n_neighbors_4_weights_distance.csv",
    ("sis_binary", "resinf"): TASKS["sis_binary"].analysis_dir / "model_Resinf.csv",
    ("neuronal_three_class", "mlp"): TASKS[
        "neuronal_three_class"
    ].analysis_dir
    / "3d_model_comparison_multisample.csv",
    ("neuronal_three_class", "knn"): TASKS[
        "neuronal_three_class"
    ].analysis_dir
    / "3d_KNN_n_neighbors_4_weights_distance.csv",
    ("neuronal_three_class", "resinf"): TASKS[
        "neuronal_three_class"
    ].analysis_dir
    / "3d_model_Resinf.csv",
    ("neuronal_binary", "mlp"): TASKS["neuronal_binary"].analysis_dir
    / "2d_model_comparison_multisample.csv",
    ("neuronal_binary", "knn"): TASKS["neuronal_binary"].analysis_dir
    / "2d_KNN_n_neighbors_4_weights_distance.csv",
    ("neuronal_binary", "resinf"): TASKS["neuronal_binary"].analysis_dir
    / "2d_model_Resinf.csv",
}

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _balanced_indices(dataset, classes: int, count_per_class: int, seed: int) -> list[int]:
    groups: dict[int, list[int]] = {label: [] for label in range(classes)}
    for index in range(len(dataset)):
        label = int(dataset[index].y.item())
        if label not in groups:
            raise ValueError(f"Unexpected class {label}; expected 0..{classes - 1}")
        groups[label].append(index)
    # Preserve the old helpers' class order, RandomState permutations and
    # progressive multiclass quota handling. Isolate their global shuffle.
    from utils.utils import select_balanced_samples, select_balanced_samples_multi
    state = random.getstate()
    random.seed(seed)
    try:
        if classes == 2:
            selected, _, _ = select_balanced_samples(groups[1], groups[0],
                sample_seed=seed, num_per_class=count_per_class)
        else:
            selected = select_balanced_samples_multi(groups,
                sample_seed=seed, num_per_class=count_per_class)
    finally:
        random.setstate(state)
    return selected


def _load_supervised_split(
    split_dir: Path,
    *,
    label_file: str,
    windows: int,
    observations: int,
    max_sample_num: int,
    seed: int,
    classes: int,
):
    """Load one supervised split with a task-specific label filename.

    This copies the graph construction performed by
    ``pred_DataSet_supervised`` while removing its hard-coded ``rs.pkl``
    dependency.  The local RNG makes observation subsampling reproducible and
    does not alter global training randomness.
    """
    import torch
    import torch_geometric.utils as pyg_utils
    from torch_geometric.data import Data

    with (split_dir / "As.pkl").open("rb") as stream:
        adjacency_list = pickle.load(stream)
    with (split_dir / "numes.pkl").open("rb") as stream:
        trajectory_list = pickle.load(stream)
    with (split_dir / label_file).open("rb") as stream:
        label_list = pickle.load(stream)

    lengths = (len(adjacency_list), len(trajectory_list), len(label_list))
    if len(set(lengths)) != 1:
        raise ValueError(
            f"{split_dir}: adjacency/trajectory/label lengths differ: {lengths}"
        )

    rng = random.Random(seed)
    dataset = []
    for source_index, (adjacency, trajectory, label) in enumerate(zip(
        adjacency_list, trajectory_list, label_list
    )):
        adjacency_tensor = torch.as_tensor(adjacency)
        if adjacency_tensor.shape[0] <= 20:
            continue
        trajectory_tensor = torch.as_tensor(trajectory)
        if trajectory_tensor.shape[2] != adjacency_tensor.shape[0]:
            raise ValueError(
                f"{split_dir}: trajectory node dimension does not match adjacency"
            )
        if trajectory_tensor.shape[1] < 2 * windows:
            raise ValueError(f"{split_dir}: trajectory window is too short")
        if trajectory_tensor.shape[0] < observations:
            raise ValueError(f"{split_dir}: not enough observations")

        edge_index = pyg_utils.dense_to_sparse(adjacency_tensor)[0]
        trajectory_tensor = (
            trajectory_tensor.permute(2, 0, 1).unsqueeze(-1).float()
        )
        observation_count = int(trajectory_tensor.shape[1])
        total_combinations = math.comb(observation_count, observations)
        if total_combinations <= max_sample_num:
            observation_sets = itertools.combinations(
                range(observation_count), observations
            )
        else:
            observation_sets = [
                tuple(sorted(rng.sample(range(observation_count), observations)))
                for _ in range(max_sample_num)
            ]

        label_value = int(np.asarray(label).reshape(-1)[0])
        if not 0 <= label_value < classes:
            raise ValueError(
                f"{split_dir}: label {label_value} is outside 0..{classes - 1}"
            )
        for observation_indices in itertools.islice(
            observation_sets, max_sample_num
        ):
            sampled = trajectory_tensor[:, observation_indices, :, :]
            dataset.append(
                Data(
                    x=sampled[:, :, :windows, :],
                    edge_index=edge_index,
                    y=torch.tensor([label_value], dtype=torch.long),
                    num_nodes=int(adjacency_tensor.shape[0]),
                    source_index=source_index,
                    observation_indices=torch.tensor([observation_indices], dtype=torch.long),
                )
            )
    if not dataset:
        raise ValueError(f"{split_dir}: no valid graphs were loaded")
    return dataset


def _build_prism(spec: TaskSpec, device: str):
    from experiment_process.generate_model_parameter_analysis_cache import ensure_prism_checkpoint
    ensure_prism_checkpoint(spec.prism_checkpoint, device)
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from utils.utils import load_Resilience_model

    model, _ = load_Resilience_model(
        str(spec.prism_checkpoint),
        device,
        infer_para={"model_task": "AE_recon"},
    )
    return model



SCHEMA_VERSION = 5
POLICY = "common_test_support_v1_new_training_best_validation"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))



def selection(ratios=None, trial_ids=None):
    ratios = tuple(RATIOS if ratios is None else ratios)
    trials = tuple(range(NUM_TRIALS) if trial_ids is None else trial_ids)
    if not ratios or len(set(ratios)) != len(ratios) or any(r not in RATIOS for r in ratios):
        raise ValueError("ratios must be a nonempty unique subset of the configured ratios")
    if not trials or len(set(trials)) != len(trials) or any(type(t) is not int or not 0<=t<NUM_TRIALS for t in trials):
        raise ValueError("trial_ids must be a nonempty unique subset of 0..9")
    return tuple(sorted(ratios)), tuple(sorted(trials))


def cache_paths(spec, method, ratios, trials):
    csv, records = canonical_csv(spec,method), canonical_trials(spec,method)
    csv, records = (p.with_name(p.stem+"_common_test_support_v1"+p.suffix) for p in (csv, records))
    if ratios == RATIOS and trials == tuple(range(NUM_TRIALS)):
        return csv, records
    token=hashlib.sha256(repr((ratios,trials)).encode()).hexdigest()[:12]
    return tuple(p.with_name(p.stem+"_selection_"+token+p.suffix) for p in (csv,records))

def archive(path: Path) -> None:
    """Preserve a previous result before replacing it; never archive model weights."""
    if path.exists():
        digest = _sha256(path)[:12]
        destination = path.parent / "archive" / f"{path.stem}_{digest}{path.suffix}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copy2(path, destination)


def _atomic_json(payload, target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=target.name, suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)


def available_model_path(spec, method, ratio, trial):
    if method not in ("mlp", "resinf"):
        raise ValueError("KNN has no model checkpoint")
    root = classifier_model_root(spec, method)
    candidates = list(dict.fromkeys(
        root / f"ratio_{ratio:.3f}_sample_{name}" / "model.pth"
        for name in (str(trial), f"{trial:02d}")
    ))
    found = [p for p in candidates if p.is_file()]
    if len(found) > 1 and len({_sha256(p) for p in found}) > 1:
        raise ValueError(f"Conflicting checkpoint aliases: {found}")
    return found[0] if found else None


def checkpoint_record(path):
    return dict(path=str(path), sha256=_sha256(path))


def save_new_checkpoint(payload, target):
    """Publish a fully written checkpoint without replacing an existing file."""
    target.parent.mkdir(parents=True,exist_ok=True)
    fd,temporary=tempfile.mkstemp(prefix="checkpoint_",suffix=".tmp",dir=target.parent)
    try:
        with os.fdopen(fd,"wb") as stream:
            torch.save(payload,stream)
        if os.name=="nt":
            os.rename(temporary,target)  # Windows refuses an existing destination.
        else:
            os.link(temporary,target)  # Atomic exclusive publication on POSIX.
    finally:
        Path(temporary).unlink(missing_ok=True)


def remap_state(state):
    substitutions = (
        ("seq_emb_agg.", "pool."), ("classifier.mlp_layers.", "body."),
        ("classifier.k_output.", "output."),
    )
    result = {}
    for name, tensor in state.items():
        name = name.removeprefix("module.")
        name = {"classifier.w": "scale", "classifier.b": "bias"}.get(name, name)
        for previous, current in substitutions:
            if name.startswith(previous):
                name = current + name[len(previous):]
                break
        if name in result:
            raise ValueError(f"Duplicate mapped parameter: {name}")
        result[name] = tensor
    return result


class AttentionPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attention = nn.Linear(dim, 1)

    def forward(self, values, batch=None):
        # Preserve the supplied training script, including its dim=1 softmax.
        weighted = F.softmax(self.attention(values), dim=1) * values
        return global_mean_pool(weighted, batch) if batch is not None else weighted.mean(0, keepdim=True)


class Classifier(nn.Module):
    def __init__(self, encoder, classes):
        super().__init__()
        self.encoder, self.classes = encoder, classes
        self.encoder.requires_grad_(False)
        dim = encoder.get_representation_dim()
        self.pool = AttentionPool(dim)
        layers = []
        for hidden in (128, 64, 32):
            layers.extend([nn.Linear(dim, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(.1)])
            dim = hidden
        self.body = nn.Sequential(*layers)
        self.output = nn.Linear(dim, 1 if classes == 2 else classes)
        self.scale = nn.Parameter(torch.randn(1) * .01)
        self.bias = nn.Parameter(torch.zeros(1))

    def classify(self, z, batch=None):
        logits = self.output(self.body(self.pool(z, batch)))
        return torch.sigmoid(self.scale * logits + self.bias) if self.classes == 2 else logits

    def forward(self, graph):
        with torch.no_grad():
            z = self.encoder.get_zlatent(graph)
        return self.classify(z, getattr(graph, "batch", None))


def load_mlp(spec, ratio, trial, device="cuda", encoder=None):
    path = available_model_path(spec, "mlp", ratio, trial)
    if path is None:
        raise FileNotFoundError(f"{spec.key} MLP ratio={ratio}, trial={trial}: model preparation must train this missing checkpoint first")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not np.isclose(checkpoint.get("ratio", ratio), ratio) or checkpoint.get("sample_idx", trial) != trial:
        raise ValueError(f"Checkpoint ratio/trial mismatch: {path}")
    model = Classifier(encoder if encoder is not None else _build_prism(spec, "cpu"), spec.classes)
    model.load_state_dict(remap_state(checkpoint["model_state_dict"]), strict=True)
    return model.to(device).eval(), checkpoint, path


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_split(spec, split):
    offset = {"train_dataset": 0, "val_dataset": 1, "test_dataset": 2}[split]
    return _load_supervised_split(spec.dataset_dir / split, label_file=spec.label_file,
        windows=10, observations=5, max_sample_num=1, seed=spec.seed+offset, classes=spec.classes)


def sample_ids(dataset):
    return [int(g.source_index) for g in dataset]


def task_seed(spec, method):
    return 42 if method in ("mlp", "resinf") and spec.classes == 2 else 233


def reference_indices(spec, train, ratio, trial, method="knn"):
    return _balanced_indices(train, spec.classes, math.ceil(ratio*len(train)/spec.classes),
                             task_seed(spec, method) + trial*1000)


def evaluation_split(spec, split, method):
    data = load_split(spec, split)
    # Keep one evaluation support for every method within a task.  The
    # neuronal binary benchmark retains the balanced support used by its
    # historical MLP and ResInf results; the other tasks use all valid graphs.
    if spec.key == "neuronal_binary":
        counts = [sum(y == label for y in labels(data)) for label in range(2)]
        indices = _balanced_indices(data, 2, min(counts), 233)
        return [data[i] for i in indices]
    return data


def learning_rate_factor(epoch, epochs=TRAIN_EPOCHS):
    warmup = int(epochs * .1)
    if epoch < warmup:
        return epoch / max(1, warmup)
    return .5 * (1 + math.cos(math.pi * (epoch-warmup) / max(1, epochs-warmup)))


@torch.inference_mode()
def extract_features(encoder, dataset, stats, device):
    vectors = []
    encoder.eval()
    for graph in DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False):
        graph = graph.to(device)
        z = encoder.get_zlatent(graph)
        parts = []
        for name in stats:
            mean = global_mean_pool(z, graph.batch)
            if name == "std":
                variance = global_mean_pool((z-mean[graph.batch])**2, graph.batch)
                counts = torch.bincount(graph.batch).to(z.dtype)
                variance *= torch.where(counts>1, counts/(counts-1), 1).unsqueeze(1)
                parts.append(torch.sqrt(variance+1e-8))
            elif name == "mean":
                parts.append(mean)
            elif name == "max":
                parts.append(global_max_pool(z, graph.batch))
            elif name == "min":
                parts.append(-global_max_pool(-z, graph.batch))
            else:
                raise ValueError(name)
        vectors.append(torch.cat(parts, 1).cpu().numpy())
    return np.concatenate(vectors)


@torch.inference_mode()
def predict(model, dataset, device):
    model.eval()
    predictions = []
    for graph in DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False):
        out = model(graph.to(device))
        predicted = (out.reshape(-1)>.5).long() if model.classes==2 else out.argmax(-1)
        predictions.extend(predicted.cpu().tolist())
    return predictions


def labels(dataset):
    return [int(g.y.item()) for g in dataset]


def gbb_predictions(spec, dataset):
    path = spec.dataset_dir / "test_dataset" / spec.baseline_file
    with path.open("rb") as stream:
        values = pickle.load(stream)
    with (path.parent / spec.label_file).open("rb") as stream:
        original_labels = pickle.load(stream)
    if len(values) != len(original_labels):
        raise ValueError("GBB/label source lengths differ")
    result = [values[i] for i in sample_ids(dataset)]
    if any(v is None or np.asarray(v).size != 1 or float(v) != int(v) or not 0<=int(v)<spec.classes for v in result):
        raise ValueError("Invalid GBB prediction on the selected test set")
    return list(map(int,result))


def historical_gbb(spec):
    """Old KNN comparison scores GBB on the raw label/prediction files."""
    folder = spec.dataset_dir / "test_dataset"
    with (folder/spec.label_file).open("rb") as stream:
        truth = pickle.load(stream)
    with (folder/spec.baseline_file).open("rb") as stream:
        predictions = pickle.load(stream)
    # Do not silently misalign samples if the old independent None filtering
    # would make the two sequences incompatible.
    if len(truth) != len(predictions) or any(y is None or p is None for y,p in zip(truth,predictions)):
        raise ValueError("Historical GBB requires aligned non-missing raw labels/predictions")
    return float(f1_score(truth, predictions, average="weighted")), len(truth)


def preflight(task, methods=("mlp","knn"), *, ratios=None, trial_ids=None):
    ratios, trials = selection(ratios,trial_ids)
    spec = TASKS[task]
    missing = []
    models = []
    if "mlp" in methods:
        for ratio in ratios:
            for trial in trials:
                path = available_model_path(spec,"mlp",ratio,trial)
                if path is None:
                    missing.append(dict(ratio=ratio, trial=trial))
                else:
                    models.append(checkpoint_record(path))
    return dict(task=task, missing_mlp=missing, available_models=models,
                note="Missing requested MLP weights are trained automatically when generating caches; KNN needs no weights")


def fingerprint(spec, method, ratios, trials):
    sources = {}
    splits = ["test_dataset"] + (["train_dataset"] if method=="knn" else [])
    for split in splits:
        for name in ("As.pkl","numes.pkl",spec.label_file):
            path = spec.dataset_dir / split / name
            sources[str(path)] = _sha256(path)
    for path in (spec.prism_checkpoint, spec.prism_config,
                 spec.dataset_dir/"test_dataset"/spec.baseline_file):
        sources[str(path)] = _sha256(path)
    if method=="mlp":
        report = preflight(spec.key, ("mlp",), ratios=ratios, trial_ids=trials)
        if report["missing_mlp"]:
            raise FileNotFoundError(f"{spec.key}: model preparation did not finish: {report['missing_mlp']}")
        sources.update({r["path"]:r["sha256"] for r in report["available_models"]})
    code_files = [Path(__file__), PROJECT_ROOT/"models/pred_model/AE_recon_model.py",
                  PROJECT_ROOT/"utils/utils.py"]
    sources = {os.path.relpath(p, PROJECT_ROOT): value for p,value in sources.items()}
    return dict(schema_version=SCHEMA_VERSION, policy=POLICY, sources=sources,
                dataset_versions=dataset_versions(spec.dataset_dir),
                code_sha256={p.name:_sha256(p) for p in code_files}, seed=spec.seed,
                ratios=list(ratios), trials=list(trials), pooling=list(spec.pooling_stats),
                model_definition="supplied generate_resilience_inference_cache.Classifier")


def _f1_summary(samples, train_size=None, trial_ids=None):
    expected=list(range(NUM_TRIALS)) if trial_ids is None else sorted(trial_ids)
    if len(samples)!=len(expected) or sorted(s["sample_idx"] for s in samples)!=expected:
        raise ValueError("A ratio requires all requested unique trials")
    scores = np.asarray([s["test_f1"] for s in samples])
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite F1")
    return dict(samples=samples, mean_test_f1=float(scores.mean()), std_test_f1=float(scores.std(ddof=0)),
                max_test_f1=float(scores.max()), min_test_f1=float(scores.min()),
                train_size=train_size, num_samples=len(expected))


def valid_payload(payload, identity):
    if not isinstance(payload,dict) or payload.get("fingerprint")!=identity or not payload.get("complete"):
        return False
    try:
        for ratio in identity["ratios"]:
            summary = payload["results"][str(ratio)]
            expected = _f1_summary(summary["samples"], summary["train_size"], identity["trials"])
            for k in ("mean_test_f1","std_test_f1","max_test_f1","min_test_f1"):
                if not np.isclose(summary[k],expected[k],atol=1e-12,rtol=0):
                    return False
            for row in summary["samples"]:
                if len(row["predictions"]) != len(payload["test_labels"]):
                    return False
                if not np.isclose(row["test_f1"],f1_score(payload["test_labels"],row["predictions"],average="weighted"),atol=1e-12,rtol=0):
                    return False
        return True
    except (KeyError, ValueError, TypeError, IndexError):
        return False


def rows_from_payload(method, payload):
    rows = []
    for ratio in payload["fingerprint"]["ratios"]:
        result = payload["results"][str(ratio)]
        stem = {"mlp": "frozen", "knn": "knn", "resinf": "Resinf"}[method]
        row = dict(data_ratio=ratio, train_size=result["train_size"], num_trials=result["num_samples"])
        row.update({f"{stem}_{key}_f1":result[f"{key}_test_f1"] for key in ("mean","std","max","min")})
        if method=="knn":
            base = payload["gbb_f1"]
            improvement = result["mean_test_f1"]-base
            row.update(baser_mean_f1=base, improvement_mean=improvement,
                       relative_improvement_pct=improvement/base*100 if base else None)
        rows.append(row)
    return pd.DataFrame(rows)


def write_summary(spec, method, payload):
    target, _ = cache_paths(spec,method,tuple(payload["fingerprint"]["ratios"]),tuple(payload["fingerprint"]["trials"]))
    frame = rows_from_payload(method,payload)
    if target.exists():
        try:
            pd.testing.assert_frame_equal(pd.read_csv(target),frame,check_dtype=False,atol=1e-12,rtol=0)
            return target
        except (AssertionError, ValueError):
            archive(target)
    target.parent.mkdir(parents=True,exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=target.parent, suffix=".csv")
    os.close(fd)
    try:
        frame.to_csv(temporary,index=False)
        os.replace(temporary,target)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return target


def dataset_versions(*roots):
    versions = {}
    for root in roots:
        for path in Path(root).rglob("generation*.json"):
            value = json.loads(path.read_text(encoding="utf-8")).get("dataset_version")
            if value:
                versions[str(path.resolve())] = value
    return versions


def cache_matches_dataset(path, *roots):
    versions = dataset_versions(*roots)
    if not versions:
        return True  # Original datasets retain their legacy-cache protocol.
    try:
        return json.loads(Path(path).read_text(encoding="utf-8")).get("_dataset_versions") == versions
    except (OSError, ValueError):
        return False


def stamp_dataset(path, *roots):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    value["_dataset_versions"] = dataset_versions(*roots)
    _atomic_json(value, Path(path))


def historical_resinf(spec):
    frozen = frozen_historical_summary(spec, "resinf")
    if frozen is not None:
        return frozen
    if dataset_versions(spec.dataset_dir):
        raise FileNotFoundError("Legacy ResInf cache predates the published dataset version")
    path = canonical_csv(spec,"resinf")
    if not path.is_file():
        path = LEGACY_CSV[(spec.key,"resinf")]
    if not path.is_file():
        raise FileNotFoundError(f"Historical ResInf cache missing: {path}")
    frame = pd.read_csv(path)
    columns = [f"Resinf_{k}_f1" for k in ("mean","std","max","min")]
    if not set(columns+["data_ratio"]).issubset(frame.columns) or len(frame)!=len(RATIOS):
        raise ValueError(f"Invalid historical ResInf cache: {path}")
    if not np.allclose(frame.data_ratio,RATIOS) or not np.isfinite(frame[columns].to_numpy()).all():
        raise ValueError(f"Invalid historical ResInf values: {path}")
    return materialize_historical_summary(spec, "resinf", path)


def frozen_historical_summary(spec, method):
    """Return a versioned, immutable plotting snapshot when its manifest matches."""
    target, manifest = cache_paths(spec, method, RATIOS, tuple(range(NUM_TRIALS)))
    if not target.is_file() or not manifest.is_file():
        return None
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if (payload.get("cache_kind") != "verified_historical_summary"
                or payload.get("schema_version") != SCHEMA_VERSION
                or payload.get("protocol") != POLICY
                or payload.get("task") != spec.key
                or payload.get("classifier") != method
                or payload.get("summary_sha256") != _sha256(target)):
            return None
        frame = pd.read_csv(target)
        stem = "frozen" if method == "mlp" else "Resinf"
        columns = [f"{stem}_{key}_f1" for key in ("mean","std","min","max")]
        if (len(frame) != len(RATIOS) or not set(columns+["data_ratio"]).issubset(frame)
                or not np.allclose(frame.data_ratio,RATIOS)
                or not np.isfinite(frame[columns].to_numpy()).all()):
            return None
        return target
    except (OSError, ValueError, KeyError, TypeError):
        return None


def materialize_historical_summary(spec, method, source):
    """Freeze an audited legacy summary with the exact shared test support."""
    existing = frozen_historical_summary(spec, method)
    if existing is not None:
        return existing
    source = Path(source)
    frame = pd.read_csv(source)
    target, manifest = cache_paths(spec, method, RATIOS, tuple(range(NUM_TRIALS)))
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=target.parent, suffix=".csv")
    os.close(fd)
    try:
        frame.to_csv(temporary, index=False)
        if target.exists():
            archive(target)
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)
    test = evaluation_split(spec, "test_dataset", method)
    truth, ids = labels(test), sample_ids(test)
    counts = {str(label): truth.count(label) for label in sorted(set(truth))}
    support_sources = {}
    for name in ("As.pkl", spec.label_file):
        path = spec.dataset_dir / "test_dataset" / name
        support_sources[os.path.relpath(path, PROJECT_ROOT)] = _sha256(path)
    payload = dict(
        cache_kind="verified_historical_summary",
        schema_version=SCHEMA_VERSION,
        protocol=POLICY,
        task=spec.key,
        classifier=method,
        complete=True,
        evaluation_scope="balanced_valid" if spec.key=="neuronal_binary" else "all_valid",
        node_filter="N > 20",
        test_indices=ids,
        test_labels=truth,
        test_sample_count=len(ids),
        class_counts=counts,
        summary_sha256=_sha256(target),
        source_summary=dict(path=os.path.relpath(source, PROJECT_ROOT), sha256=_sha256(source)),
        support_sources=support_sources,
        note="Audited historical aggregate copied without changing values; the manifest fixes the common test support used by the task.",
    )
    archive(manifest)
    _atomic_json(payload, manifest)
    return target


def historical_cache(spec, method):
    """Only explicitly identified legacy CSVs, never full-test v3 outputs."""
    # Historical KNN CSVs contain a GBB score computed on raw, unfiltered
    # files; neuronal-binary KNN also used a different test support from the
    # other classifiers.  Rebuild KNN so both curves use the task's shared
    # evaluation indices.
    if method == "knn":
        return None
    frozen = frozen_historical_summary(spec, method)
    if frozen is not None:
        return frozen
    if dataset_versions(spec.dataset_dir):
        return None
    path = LEGACY_CSV.get((spec.key,method))
    old_records = None
    if path is None or not path.is_file():
        path = canonical_csv(spec,method)
        if not path.is_file():
            return None
        records = canonical_trials(spec,method)
        if records.is_file():
            old_records = json.loads(records.read_text(encoding="utf-8"))
            if ("fingerprint" in old_records or old_records.get("num_samples")!=NUM_TRIALS
                    or old_records.get("data_ratios")!=list(RATIOS)):
                return None
        elif not (method=="mlp" and "finetuned_mean_f1" in pd.read_csv(path,nrows=0).columns):
            return None  # No evidence that this canonical CSV came from old code.
    frame = pd.read_csv(path)
    stem = "frozen" if method == "mlp" else "knn"
    columns = [f"{stem}_{k}_f1" for k in ("mean","std","min","max")]
    if method == "knn":
        columns += ["baser_mean_f1", "improvement_mean"]
    if not set(columns+["data_ratio"]).issubset(frame) or len(frame)!=len(RATIOS):
        raise ValueError(f"Malformed legacy cache: {path}")
    if not np.allclose(frame.data_ratio, RATIOS) or not np.isfinite(frame[columns].to_numpy()).all():
        raise ValueError(f"Invalid legacy cache values: {path}")
    lo, mean, hi = (frame[f"{stem}_{k}_f1"] for k in ("min","mean","max"))
    if not ((lo<=mean+1e-12)&(mean<=hi+1e-12)&(lo>=0)&(hi<=1)).all():
        raise ValueError(f"Invalid legacy F1 bounds: {path}")
    # Old JSON filenames omitted KNN parameters and could be overwritten across
    # runs. Their schema identifies the legacy format, not a paired run. Keep
    # the validated plotting CSV authoritative; never rebuild it from that JSON.
    return materialize_historical_summary(spec, method, path)


def ensure_caches(task, methods=METHODS, *, device="cuda", force=False, ratios=None, trial_ids=None):
    ratios, trials = selection(ratios,trial_ids)
    spec = TASKS[task]
    if set(methods)-set(METHODS):
        raise ValueError("Unknown method")
    # Only missing requested weights are trained.
    result = {}
    for method in methods:
        if method=="resinf":
            from experiment_process.resinf_inference import ensure_resinf_cache
            result[method] = ensure_resinf_cache(spec, ratios, trials, device, force, api=sys.modules[__name__])
            continue
        _, pending_path = cache_paths(spec,method,ratios,trials)
        interrupted = False
        if pending_path.is_file() and not force:
            try:
                pending = json.loads(pending_path.read_text(encoding="utf-8"))
                interrupted = isinstance(pending,dict) and pending.get("protocol")==POLICY and pending.get("complete") is False
            except (ValueError,OSError):
                pass
        if not force and not interrupted and trials == tuple(range(NUM_TRIALS)):
            try:
                old = historical_cache(spec, method)
            except (ValueError, KeyError, TypeError):
                old = None
            if old is not None:
                result[method] = old
                continue
        if not force and pending_path.is_file():
            try:
                cached = json.loads(pending_path.read_text(encoding="utf-8"))
                saved_identity = cached["fingerprint"]
                if (saved_identity.get("dataset_versions", {}) == dataset_versions(spec.dataset_dir)
                        and saved_identity.get("policy") == POLICY
                        and saved_identity.get("ratios") == list(ratios)
                        and saved_identity.get("trials") == list(trials)
                        and valid_payload(cached, saved_identity)):
                    result[method] = write_summary(spec, method, cached)
                    continue
            except (ValueError, KeyError, TypeError):
                pass
        from experiment_process.generate_model_parameter_analysis_cache import ensure_prism_checkpoint
        ensure_prism_checkpoint(spec.prism_checkpoint, device)
        if method=="mlp":
            report = preflight(task,("mlp",),ratios=ratios,trial_ids=trials)
            if report["missing_mlp"]:
                train_missing(task,device,ratios=ratios,trial_ids=trials)
        identity = fingerprint(spec,method,ratios,trials)
        _, path = cache_paths(spec,method,ratios,trials)
        payload = None
        candidate = None
        if path.exists() and not force:
            try:
                candidate = json.loads(path.read_text(encoding="utf-8"))
                if valid_payload(candidate,identity):
                    payload = candidate
            except (ValueError, OSError):
                pass
        if payload is None:
            archive(path)
            previous = candidate if not force else None
            payload = evaluate_task(spec,method,identity,device, previous=previous, checkpoint_path=path)
            _atomic_json(payload,path)
        result[method] = write_summary(spec,method,payload)
    return result


def evaluate_task(spec, method, identity, device, previous=None, checkpoint_path=None):
    _seed_everything(spec.seed)
    torch.set_num_threads(1)
    test = evaluation_split(spec,"test_dataset",method)
    truth, ids = labels(test), sample_ids(test)
    basers = gbb_predictions(spec,test)
    payload = dict(fingerprint=identity, task=spec.key, classifier=method, complete=False,
        test_indices=ids, test_labels=truth,
        observation_indices=[g.observation_indices.reshape(-1).tolist() for g in test],
        gbb_predictions=basers, gbb_f1=float(f1_score(truth,basers,average="weighted")),
        results={}, environment=dict(torch=torch.__version__, device=device,
           gpu=torch.cuda.get_device_name(0) if str(device).startswith("cuda") else None))
    payload.update(protocol=POLICY, evaluation_scope="balanced_valid" if spec.key=="neuronal_binary" else "all_valid",
        historical_random_indices_recovered=False)
    if method == "knn":
        payload["gbb_sample_count"] = len(truth)
        payload["gbb_scope"] = "same_test_indices"
    if (isinstance(previous,dict) and previous.get("fingerprint")==identity
            and previous.get("test_indices")==ids and previous.get("test_labels")==truth):
        payload["results"] = previous.get("results",{})
    encoder = _build_prism(spec,device).eval()
    if method=="knn":
        train = load_split(spec,"train_dataset")
        train_labels = np.asarray(labels(train))
        feature_identity={k:v for k,v in identity.items() if k not in ("ratios","trials")}
        feature_path=spec.analysis_dir/f"{spec.prefix}_knn_features_common_test_support_v1.json"
        saved=None
        if feature_path.exists():
            try:
                candidate=json.loads(feature_path.read_text(encoding="utf-8"))
                if (candidate.get("fingerprint")==feature_identity and candidate.get("train_indices")==sample_ids(train)
                        and candidate.get("test_indices")==ids and candidate.get("labels")==train_labels.tolist()):
                    a,b=np.asarray(candidate["train_features"],dtype=np.float32),np.asarray(candidate["test_features"],dtype=np.float32)
                    dim=encoder.get_representation_dim()*len(spec.pooling_stats)
                    if a.shape==(len(train),dim) and b.shape==(len(test),dim) and np.isfinite(a).all() and np.isfinite(b).all():
                        saved=(a,b)
            except (ValueError,KeyError,TypeError):
                pass
        if saved is None:
            train_features = extract_features(encoder,train,spec.pooling_stats,device)
            test_features = extract_features(encoder,test,spec.pooling_stats,device)
            archive(feature_path)
            _atomic_json(dict(fingerprint=feature_identity,train_indices=sample_ids(train),test_indices=ids,
                labels=train_labels.tolist(),train_features=train_features.tolist(),test_features=test_features.tolist()),feature_path)
        else:
            train_features,test_features=saved
        payload["reference_pool"] = dict(source_split="train_dataset", indices=sample_ids(train),
            features=train_features.tolist(), labels=train_labels.tolist(),
            observation_indices=[g.observation_indices.reshape(-1).tolist() for g in train])
    for ratio in identity["ratios"]:
        trials = []
        old_rows = payload["results"].get(str(ratio),{}).get("samples",[])
        for trial in identity["trials"]:
            old = [r for r in old_rows if r.get("sample_idx")==trial]
            if len(old)==1 and len(old[0].get("predictions",[]))==len(truth):
                if np.isclose(old[0].get("test_f1",np.nan),f1_score(truth,old[0]["predictions"],average="weighted")):
                    trials.append(old[0])
                    continue
            seed = task_seed(spec,method) + trial*1000
            _seed_everything(seed)
            row = dict(sample_idx=trial,seed=seed)
            if method=="mlp":
                model, checkpoint, path = load_mlp(spec,ratio,trial,device,encoder)
                prediction = predict(model,test,device)
                row.update(model=checkpoint_record(path), train_size=checkpoint.get("train_size"),
                    training_protocol=checkpoint.get("config",{}).get("policy","historical_unspecified"),
                    checkpoint_selection=checkpoint.get("config",{}).get("checkpoint_selection","historical_unspecified"),
                    historical_test_f1=checkpoint.get("test_f1",checkpoint.get("results",{}).get("test_f1")),
                    historical_train_indices=checkpoint.get("train_indices"))
                del model
            else:
                selected = reference_indices(spec,train,ratio,trial)
                if len(selected)<4:
                    raise ValueError(f"{spec.key}/{ratio}: fewer than 4 references; k is not silently changed")
                knn = KNeighborsClassifier(n_neighbors=4,weights="distance",metric="euclidean",
                                          n_jobs=1).fit(train_features[selected],train_labels[selected])
                prediction = knn.predict(test_features).tolist()
                row.update(reference_positions=selected,
                    reference_indices=[int(train[i].source_index) for i in selected],train_size=len(selected))
            row.update(predictions=prediction,test_f1=float(f1_score(truth,prediction,average="weighted")))
            if row.get("historical_test_f1") is not None:
                row["historical_f1_difference"]=row["test_f1"]-row["historical_test_f1"]
            trials.append(row)
            payload["results"][str(ratio)] = dict(samples=trials)
            if checkpoint_path is not None:
                _atomic_json(payload,checkpoint_path)
            print(f"{spec.key}/{method} ratio={ratio} trial={trial}: F1={row['test_f1']:.6f}",flush=True)
        sizes = [row["train_size"] for row in trials]
        payload["results"][str(ratio)] = _f1_summary(trials,sizes[0] if len(set(sizes))==1 else None,identity["trials"])
    payload["complete"]=True
    return payload


def train_missing(task,device="cuda", *, ratios=None, trial_ids=None):
    """Train requested missing heads with the legacy recipe; never overwrite."""
    spec = TASKS[task]
    report = preflight(task,("mlp",),ratios=ratios,trial_ids=trial_ids)
    if not report["missing_mlp"]:
        return report
    print(f"{task}: automatically training {len(report['missing_mlp'])} missing MLP models: {report['missing_mlp']}",flush=True)
    torch.set_num_threads(1)
    train = load_split(spec,"train_dataset")
    val = evaluation_split(spec,"val_dataset","mlp")
    validation_labels = labels(val)
    encoder = _build_prism(spec,device)
    validation_loader = DataLoader(val,batch_size=BATCH_SIZE,shuffle=False)
    sources = {str(spec.dataset_dir/split/name):_sha256(spec.dataset_dir/split/name)
        for split in ("train_dataset","val_dataset")
        for name in ("As.pkl","numes.pkl",spec.label_file)}
    lr = .001 if spec.classes == 2 else .0005
    for item in report["missing_mlp"]:
        ratio,trial=item["ratio"],item["trial"]
        seed=task_seed(spec,"mlp")+trial*1000
        _seed_everything(seed)
        selected=reference_indices(spec,train,ratio,trial,method="mlp")
        if not selected:
            raise ValueError("No training references available")
        model=Classifier(copy.deepcopy(encoder),spec.classes).to(device)
        optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=lr,weight_decay=1e-4)
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda epoch:learning_rate_factor(epoch,TRAIN_EPOCHS))
        loader=DataLoader([train[i] for i in selected],batch_size=BATCH_SIZE,shuffle=True,
                          generator=torch.Generator().manual_seed(seed))
        best=-1.
        history=[]
        for epoch in range(TRAIN_EPOCHS):
            model.train()  # Legacy: frozen weights, but encoder dropout remains active.
            epoch_lr=optimizer.param_groups[0]["lr"]
            for graph in loader:
                graph=graph.to(device)
                out=model(graph)
                target=graph.y
                loss=F.binary_cross_entropy(out.reshape(-1),target.float()) if spec.classes==2 else F.cross_entropy(out,target.long())
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"{task}/{ratio}/{trial}: nonfinite training loss")
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            scheduler.step()
            model.eval()
            predictions=[]
            with torch.no_grad():
                for graph in validation_loader:
                    out=model(graph.to(device))
                    if not torch.isfinite(out).all():
                        raise FloatingPointError("Nonfinite validation output")
                    predictions.extend(((out.reshape(-1)>.5).long() if spec.classes==2 else out.argmax(-1)).cpu().tolist())
            score=float(f1_score(validation_labels,predictions,average="weighted"))
            history.append(dict(epoch=epoch,lr=epoch_lr,val_f1=score))
            if epoch==0 or (epoch+1)%10==0:
                print(f"{task}/{ratio}/trial={trial}: epoch {epoch+1}/{TRAIN_EPOCHS}, validation F1={score:.6f}",flush=True)
            if score>best:
                best=score
                best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                best_epoch=epoch
        target=trained_model_path(spec,"mlp",ratio,trial)
        if available_model_path(spec,"mlp",ratio,trial) is not None:
            raise FileExistsError(target)
        target.parent.mkdir(parents=True,exist_ok=True)
        save_new_checkpoint(dict(model_state_dict=best_state,ratio=ratio,sample_idx=trial,seed=seed,
            train_size=len(selected),train_indices=[int(train[i].source_index) for i in selected],
            val_indices=sample_ids(val),best_val_f1=best,best_epoch=best_epoch,history=history,
            observation_indices={split:[g.observation_indices.reshape(-1).tolist() for g in data]
                                 for split,data in (("train",train),("validation",val))},
            seeds=dict(initialization=seed,reference=seed,batches=seed,observations=spec.seed,validation_observations=spec.seed+1),
            config=dict(epochs=TRAIN_EPOCHS,lr=lr,weight_decay=1e-4,batch_size=BATCH_SIZE,
                encoder_eval=False,encoder_frozen=True,scheduler="linear_warmup_10pct_cosine",
                checkpoint_selection="best_validation_clone",policy=POLICY,
                historical_difference="Old state_dict references tracked final weights; new checkpoints clone the best validation epoch"),
            data_sources=sources),target)
        del model, optimizer, scheduler, loader, best_state
        print(f"Saved missing {task} ratio={ratio} trial={trial}, validation F1={best:.6f}",flush=True)
    return preflight(task,("mlp",),ratios=ratios,trial_ids=trial_ids)


def generate_task(task,methods=METHODS,*,device="cuda",force=False,ratios=None,trial_ids=None):
    return ensure_caches(task,tuple(methods),device=device,force=force,ratios=ratios,trial_ids=trial_ids)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    action=parser.add_mutually_exclusive_group()
    action.add_argument("--preflight",action="store_true")
    action.add_argument("--generate",action="store_true")
    action.add_argument("--train-missing",action="store_true")
    parser.add_argument("--task",choices=tuple(TASKS),action="append")
    parser.add_argument("--method",choices=METHODS,action="append")
    parser.add_argument("--device",default="cuda")
    parser.add_argument("--ratios",type=float,nargs="+")
    parser.add_argument("--trials",type=int,nargs="+")
    parser.add_argument("--smoke-test",action="store_true",help="Run isolated temporary tests only")
    parser.add_argument("--force",action="store_true",help="Recompute inference caches, NEVER retrain")
    args=parser.parse_args()
    if args.device.startswith("cuda") and os.environ.get("PRISM_GPU_MEMORY_FRACTION"):
        torch.cuda.set_per_process_memory_fraction(float(os.environ["PRISM_GPU_MEMORY_FRACTION"]), device="cuda:0" if args.device == "cuda" else args.device)
    if args.smoke_test:
        import subprocess
        subprocess.run([sys.executable,str(PROJECT_ROOT/"experiment_process/test_resilience_inference_smoke.py")],check=True)
        return
    if args.train_missing and (args.force or (args.method and args.method!=["mlp"])):
        parser.error("--train-missing is MLP-only and cannot be combined with --force")
    for task in args.task or list(TASKS):
        if args.train_missing:
            result=train_missing(task,args.device,ratios=args.ratios,trial_ids=args.trials)
        elif args.generate:
            result=generate_task(task,args.method or METHODS,device=args.device,force=args.force,ratios=args.ratios,trial_ids=args.trials)
        else:
            result=preflight(task,args.method or ("mlp","knn"),ratios=args.ratios,trial_ids=args.trials)
        print(json.dumps(result,default=str,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
