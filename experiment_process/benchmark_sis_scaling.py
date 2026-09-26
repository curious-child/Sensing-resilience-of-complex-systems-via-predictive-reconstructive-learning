"""SIS scaling: cached evaluation, original-data model preparation and serial measurement."""
from __future__ import annotations

import argparse
import csv
import gc
import importlib.metadata
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import sys
import threading
import time
from collections.abc import Sequence

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
from scipy import sparse
from dataset.sis_dataset_generation.generate_sis_dataset import IndexedPickleList, resource_guard

PROJECT = Path(__file__).resolve().parents[1]
METHODS = ("knn", "mlp", "resinf", "gbb")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_config(path):
    import yaml
    with Path(path).open(encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    if not isinstance(cfg, dict):
        raise ValueError("Configuration must be a mapping")
    defaults = dict(methods=list(METHODS), training_ratio=0.1, trial_ids=list(range(10)),
                    observations=5, windows=10, seed=42, device="cuda:0", cpu_threads=1,
                    warmup=10, repeats=30, timing_samples=20, timeout_seconds=120,
                    startup_timeout_seconds=180, memory_poll_seconds=0.001,
                    allow_missing_methods=False,
                    test_only=False)
    cfg = {**defaults, **cfg}
    if not cfg.get("datasets"):
        raise ValueError("datasets must contain at least one data group")
    if not cfg["methods"] or len(set(cfg["methods"])) != len(cfg["methods"]) or set(cfg["methods"]) - set(METHODS):
        raise ValueError(f"methods must be a unique subset of {METHODS}")
    if not 0 < cfg["training_ratio"] <= 1:
        raise ValueError("training_ratio must be in (0, 1]")
    if not cfg["trial_ids"] or len(set(cfg["trial_ids"])) != len(cfg["trial_ids"]) or any(type(t) is not int or t < 0 for t in cfg["trial_ids"]):
        raise ValueError("trial_ids must contain unique nonnegative integers")
    for key in ("observations", "windows", "cpu_threads", "repeats", "timing_samples"):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if type(cfg["warmup"]) is not int or cfg["warmup"] < 0:
        raise ValueError("warmup must be a nonnegative integer")
    for key in ("timeout_seconds", "startup_timeout_seconds", "memory_poll_seconds"):
        if not np.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if "activity_threshold" in cfg:
        raise ValueError("GBB uses infection_rate*beta_eff>delta; remove obsolete activity_threshold")
    if not str(cfg["device"]).startswith("cuda:") and not (cfg["test_only"] and cfg["device"] == "cpu"):
        raise ValueError("Neural device must be cuda:<index>; CPU override requires test_only: true")
    for key, default in (
        ("checkpoint_root", "experiment_results/SIS/resilience_inference_analysis/classifier_models/binary"),
        ("prism_checkpoint", "experiment_results/SIS/resilience_inference_analysis/model_trained"),
    ):
        cfg[key] = str((PROJECT / cfg.get(key, default)).resolve())
    seen, combinations = set(), set()
    for group in cfg["datasets"]:
        if group.get("topology") not in ("ER", "SF"):
            raise ValueError("topology must be ER or SF")
        if type(group.get("n")) is not int or group["n"] < 1:
            raise ValueError("Each group needs an exact positive integer n")
        group.setdefault("id", f"{group['topology']}_N{group['n']}")
        if group["id"] in seen:
            raise ValueError("Duplicate group id")
        seen.add(group["id"])
        combination = (group["topology"], group["n"])
        if combination in combinations:
            raise ValueError("Use one data group per topology/N; concatenate independent test samples in that group")
        combinations.add(combination)
        group["path"] = str((PROJECT / group["path"]).resolve())
    return cfg


def checkpoint_path(cfg, method, trial):
    from experiment_process import generate_resilience_inference_cache as api
    spec = api.TASKS["sis_binary"]
    return api.available_model_path(spec, method, cfg["training_ratio"], trial) or api.trained_model_path(spec, method, cfg["training_ratio"], trial)


def _load(path):
    # Only load local, trusted project data/checkpoints: pickle is executable.
    with Path(path).open("rb") as stream:
        return pickle.load(stream)


def validate_params(value):
    if not isinstance(value, dict) or "delta" not in value:
        raise ValueError("Explicit SIS delta is required (never inferred from labels)")
    delta, rate = float(value["delta"]), float(value.get("infection_rate", 1))
    if not np.isfinite([delta, rate]).all() or delta <= 0 or rate < 0:
        raise ValueError("SIS delta must be positive; infection_rate must be nonnegative")
    return dict(delta=delta, infection_rate=rate)


def load_group(group, cfg):
    """Load indexed sources lazily; each retrieved sample is validated, never filtered."""
    root = Path(group["path"])
    metadata = root / "generation.json"
    index = json.loads(metadata.read_text(encoding="utf-8")).get("pickle_index", {}) if metadata.exists() else {}
    def source(name):
        return IndexedPickleList(root / name, index.get(name))
    matrices, trajectories, labels = [source(name) for name in ("As.pkl", "numes.pkl", "rs.pkl")]
    if not len(labels) or not len(matrices) == len(trajectories) == len(labels):
        raise ValueError("Adjacency/trajectory/label lists must be nonempty and have equal lengths")
    params = group.get("params")
    if "params_file" in group:
        with (root / group["params_file"]).open(encoding="utf-8") as stream:
            params = json.load(stream)
    if isinstance(params, dict):
        params = [params] * len(labels)
    if not isinstance(params, list) or len(params) != len(labels):
        raise ValueError("Provide params mapping or params_file with one SIS parameter mapping per sample")
    baseline = source("basers.pkl")
    if len(baseline) != len(labels):
        raise ValueError("basers.pkl length mismatch")
    return ScalingSamples(group, cfg, matrices, trajectories, labels, params, baseline)


class ScalingSamples(Sequence):
    def __init__(self, group, cfg, matrices, trajectories, labels, params, baseline):
        self.group, self.cfg = group, cfg
        self.sources = matrices, trajectories, labels, params, baseline

    def __len__(self):
        return len(self.sources[0])

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self[j] for j in range(*i.indices(len(self)))]
        matrix, trajectory, label, param, old = (source[i] for source in self.sources)
        group, cfg = self.group, self.cfg
        A = sparse.csr_matrix(matrix, dtype=np.float64)
        A.sum_duplicates()
        A.eliminate_zeros()
        n = group["n"]
        if A.shape != (n, n):
            raise ValueError(f"sample {i}: actual adjacency shape {A.shape} differs from declared n={n}")
        if not np.isfinite(A.data).all() or np.any(A.data != 1) or np.any(A.diagonal() != 0) or (A != A.T).nnz:
            raise ValueError(f"sample {i}: expected a finite, undirected, unweighted, loop-free adjacency")
        y = np.asarray(label)
        if y.size != 1 or y.item() not in (0, 1):
            raise ValueError(f"sample {i}: label must be 0 or 1")
        if old is not None and (np.asarray(old).size != 1 or np.asarray(old).item() not in (0, 1)):
            raise ValueError(f"sample {i}: cached GBB prediction must be 0 or 1")
        x = np.asarray(trajectory)
        if x.ndim != 3 or x.shape[2] != n or x.shape[0] < cfg["observations"] or x.shape[1] < cfg["windows"]:
            raise ValueError(f"sample {i}: expected trajectories [M,T,{n}] with M/T >= configured observations/windows")
        if not np.isfinite(x).all():
            raise ValueError(f"sample {i}: non-finite trajectories")
        seed = int.from_bytes(hashlib.sha256(f"{cfg['seed']}:{group['id']}:{i}".encode()).digest()[:8], "little")
        chosen = np.sort(np.random.default_rng(seed).choice(x.shape[0], cfg["observations"], replace=False))
        past = np.ascontiguousarray(x[chosen, :cfg["windows"], :].transpose(2, 0, 1)[..., None], dtype=np.float32)
        if not np.isfinite(past).all():
            raise ValueError(f"sample {i}: trajectories overflow float32")
        return dict(id=f"{group['id']}:{i}", n=n, A=A, x=past, y=int(y.item()),
                            observation_indices=chosen.tolist(), params=validate_params(param),
                            cached_gbb=None if old is None else int(np.asarray(old).item()))


def gbb_predict(A, delta, infection_rate=1):
    """GBB SIS: beta_eff=<k_out*k_in>/<k>; x*=max(0,1-delta/(c*beta_eff))."""
    params = validate_params(dict(delta=delta, infection_rate=infection_rate))
    A = sparse.csr_matrix(A, dtype=np.float64)
    kin = np.asarray(A.sum(axis=1)).ravel()
    kout = kin
    total = float(kin.sum())
    beta = float(kout @ kin / total) if total > 0 else 0.0
    effective_rate = params["infection_rate"] * beta
    equilibrium = max(0.0, 1 - params["delta"] / effective_rate) if effective_rate > 0 else 0.0
    return dict(prediction=int(effective_rate > delta),
                score=equilibrium, beta_eff=beta, spectral_margin=None)


def normalized_resinf_inputs(x, A):
    """Exactly the existing virtual-node convention, without dense diagonal products."""
    n = A.shape[0]
    resource_guard((n + 1) ** 2 * 20)  # ResInf itself still requires dense topology.
    expanded = np.zeros((n + 1, n + 1), dtype=np.float32)
    expanded[:n, :n] = A.toarray() if sparse.issparse(A) else A
    expanded[n, :n] = 1
    row, col = expanded.sum(1), expanded.sum(0)
    rinv, cinv = np.zeros_like(row), np.zeros_like(col)
    np.power(row, -0.5, out=rinv, where=row != 0)
    np.power(col, -0.5, out=cinv, where=col != 0)
    normalized = rinv[:, None] * expanded * cinv[None, :]
    numerical = x[..., 0].transpose(1, 0, 2)
    numerical = np.concatenate([numerical, numerical.mean(1, keepdims=True)], axis=1)
    return normalized, numerical

import torch
from torch import nn

class SharedEncoder(nn.Module):
    """Reuse trained modules only; neither decoder is retained or invoked."""
    def __init__(self, full):
        super().__init__()
        ns = full.cond_pred_model
        for name in ("enc_embedding", "encoder", "tau_learner", "delta_learner", "node_enc_reduc"):
            setattr(self, name, getattr(ns, name))
        self.traj_reduc = full.traj_reduc
        self.hidden_dim = full.hidden_dim
        self.windows = full.windows

    def get_zlatent(self, graph):
        return self.forward(graph.x)

    def forward(self, x):
        n, m, w, f = x.shape
        if w != self.windows or f != 1:
            raise ValueError("History shape does not match the trained encoder")
        raw = x.reshape(n * m, w, f)
        mean = raw.mean(1, keepdim=True)
        centered = raw - mean
        std = torch.sqrt(centered.var(dim=1, keepdim=True, unbiased=False) + 1e-5)
        tau = self.tau_learner(raw, std).exp()
        delta = self.delta_learner(raw, mean)
        embedded = self.enc_embedding(x=centered / std, x_mark=None)
        encoded, _ = self.encoder(embedded, attn_mask=None, tau=tau, delta=delta)
        nodes = self.node_enc_reduc(encoded).reshape(n, m, self.hidden_dim)
        return self.traj_reduc(nodes)


class NeuralAdapter:
    def __init__(self, cfg, method, trial):
        self.method, self.device = method, torch.device(cfg["device"])
        if self.device.type == "cuda" and (not torch.cuda.is_available() or (self.device.index or 0) >= torch.cuda.device_count()):
            raise RuntimeError(f"Requested GPU is unavailable: {self.device}")
        torch.set_num_threads(cfg["cpu_threads"])
        from experiment_process import generate_resilience_inference_cache as api
        from experiment_process import resinf_inference as ri
        spec = api.TASKS["sis_binary"]
        if method == "resinf":
            checkpoint = torch.load(checkpoint_path(cfg, method, trial), map_location="cpu", weights_only=False)
            self.model = ri.build_model(2, checkpoint)
        elif method == "mlp":
            self.model, _, _ = api.load_mlp(spec, cfg["training_ratio"], trial, "cpu")
            self.model.encoder = SharedEncoder(self.model.encoder)
        else:
            from sklearn.neighbors import KNeighborsClassifier
            self.model = SharedEncoder(api._build_prism(spec, "cpu"))
            reference = cfg["_references"][str(trial)]
            self.classifier = KNeighborsClassifier(n_neighbors=4, weights="distance", metric="euclidean")
            self.classifier.fit(np.asarray(reference["features"]), reference["labels"])
            self.stats = spec.pooling_stats
        self.model.to(self.device).eval()
        self.model.requires_grad_(False)

    def sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def pool_knn(self, z):
        parts = []
        for name in self.stats:
            if name == "std":
                variance = ((z - z.mean(0)) ** 2).mean(0)
                if len(z) > 1:
                    variance = variance * len(z) / (len(z) - 1)
                value = torch.sqrt(variance + 1e-8)
            elif name == "mean":
                value = z.mean(0)
            elif name == "max":
                value = z.max(0).values
            else:
                value = z.min(0).values
            parts.append(value)
        return torch.cat(parts).unsqueeze(0).cpu().numpy()

    @torch.inference_mode()
    def predict(self, payload, stages=False):
        times = {}
        self.sync()
        start = time.perf_counter_ns()
        if self.method == "resinf":
            adj, x = normalized_resinf_inputs(payload["x"], payload["A"])
            x, adj = torch.as_tensor(x, device=self.device), torch.as_tensor(adj, device=self.device)
            if stages:
                self.sync()
                times["preprocess_ms"] = (time.perf_counter_ns() - start) / 1e6
            score = float(self.model(x, adj).reshape(-1)[0].cpu())
            result = dict(prediction=int(score > 0.5), score=score)
        else:
            x = torch.as_tensor(payload["x"], device=self.device)
            encoder = self.model.encoder if self.method == "mlp" else self.model
            z = encoder(x)
            if stages:
                self.sync()
                times["representation_ms"] = (time.perf_counter_ns() - start) / 1e6
            classify_start = time.perf_counter_ns()
            if self.method == "mlp":
                score = float(self.model.classify(z).reshape(-1)[0].cpu())
                result = dict(prediction=int(score > 0.5), score=score)
            else:
                features = self.pool_knn(z)
                prediction = int(self.classifier.predict(features)[0])
                score = None
                result = dict(prediction=prediction, score=score)
            if stages:
                self.sync()
                times["classification_ms"] = (time.perf_counter_ns() - classify_start) / 1e6
        self.sync()
        return {**result, **times}


def _status(error):
    text = str(error).lower()
    return "oom" if "out of memory" in text or isinstance(error, MemoryError) else "error"


def _memory_call(predict, adapter, payload, interval):
    import psutil
    gc.collect()
    gpu = adapter is not None and adapter.device.type == "cuda"
    if gpu:
        import torch
        adapter.sync()
        torch.cuda.empty_cache()
        base_alloc = torch.cuda.memory_allocated(adapter.device)
        torch.cuda.reset_peak_memory_stats(adapter.device)
    process = psutil.Process()
    baseline = process.memory_info().rss
    peaks = [baseline]
    stop = threading.Event()

    def sample():
        while not stop.wait(interval):
            peaks[0] = max(peaks[0], process.memory_info().rss)

    monitor = threading.Thread(target=sample, daemon=True)
    monitor.start()
    try:
        predict(payload)
        peaks[0] = max(peaks[0], process.memory_info().rss)
    finally:
        stop.set()
        monitor.join()
    result = dict(cpu_baseline_rss_bytes=baseline, cpu_peak_rss_sampled_bytes=peaks[0],
                  cpu_increment_rss_sampled_bytes=max(0, peaks[0] - baseline),
                  memory_poll_seconds=interval)
    if gpu:
        result.update(gpu_baseline_allocated_bytes=base_alloc,
                      gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(adapter.device),
                      gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(adapter.device),
                      gpu_increment_allocated_bytes=torch.cuda.max_memory_allocated(adapter.device) - base_alloc)
    return result


def _worker(connection, cfg, method, trial):
    """Isolate model/input memory from the controller's complete test dataset."""
    try:
        from threadpoolctl import threadpool_limits
        limit = threadpool_limits(limits=cfg["cpu_threads"])
        adapter = None
        if method == "gbb":
            def predict(p, stages=False):
                return gbb_predict(p["A"], **p["params"])
        else:
            adapter = NeuralAdapter(cfg, method, trial)
            predict = adapter.predict
        connection.send(dict(status="ready"))
    except Exception as error:
        connection.send(dict(status=_status(error), error=str(error)))
        connection.close()
        return
    while True:
        try:
            message = connection.recv()
        except EOFError:
            break
        if message is None:
            break
        command, payload = message
        try:
            if command == "memory":
                response = _memory_call(predict, adapter, payload, cfg["memory_poll_seconds"])
            elif command == "timing":
                rows = []
                for repeat in range(cfg["repeats"]):
                    if adapter:
                        adapter.sync()
                    start = time.perf_counter_ns()
                    result = predict(payload)
                    if adapter:
                        adapter.sync()
                    rows.append(dict(repeat=repeat, total_ms=(time.perf_counter_ns() - start) / 1e6,
                                     prediction=result["prediction"]))
                response = dict(rows=rows)
            elif command == "warmup":
                for _ in range(cfg["warmup"]):
                    predict(payload)
                response = {}
            else:
                response = predict(payload, stages=command == "stages")
            connection.send(dict(status="ok", **response))
        except Exception as error:
            connection.send(dict(status=_status(error), error=str(error)))
        finally:
            del payload, message
    connection.close()
    limit.restore_original_limits()


class Worker:
    def __init__(self, cfg, method, trial):
        ctx = mp.get_context("spawn")
        self.pipe, child = ctx.Pipe()
        self.process = ctx.Process(target=_worker, args=(child, cfg, method, trial))
        self.process.start()
        child.close()
        self.timeout = cfg["timeout_seconds"]
        self.ready = self._receive(cfg["startup_timeout_seconds"])

    def _receive(self, timeout):
        if not self.pipe.poll(timeout):
            self.process.terminate()
            self.process.join(5)
            return dict(status="timeout", error=f"Worker exceeded {timeout}s; terminated")
        try:
            return self.pipe.recv()
        except (EOFError, OSError) as error:
            return dict(status="error", error=f"Worker exited: {error}")

    def call(self, command, payload):
        if not self.process.is_alive():
            return dict(status="worker_unavailable", error="Worker stopped after an earlier failure")
        try:
            self.pipe.send((command, payload))
        except (OSError, EOFError) as error:
            return dict(status="error", error=str(error))
        return self._receive(self.timeout)

    def close(self):
        if self.process.is_alive():
            try:
                self.pipe.send(None)
            except OSError:
                pass
            self.process.join(5)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(5)
        self.pipe.close()


def preflight(cfg):
    issues, jobs, groups, sources = [], [], [], {}
    for group in cfg["datasets"]:
        try:
            samples = load_group(group, cfg)
            labels, degrees = [], []
            for sample in samples:
                labels.append(sample["y"])
                degrees.append(sample["A"].nnz / sample["n"])
            groups.append(dict(id=group["id"], topology=group["topology"], n=group["n"],
                               samples=len(samples), class_0=labels.count(0),
                               class_1=labels.count(1),
                               mean_degree_min=min(degrees), mean_degree_max=max(degrees)))
            names = ["As.pkl", "numes.pkl", "rs.pkl", "basers.pkl", "generation.json"]
            if group.get("params_file"):
                names.append(group["params_file"])
            for name in names:
                path = Path(group["path"]) / name
                if path.is_file():
                    sources[str(path)] = file_hash(path)
        except Exception as error:
            issues.append(dict(kind="data", group=group["id"], detail=str(error), fatal=True))
    neural_ok = True
    if set(cfg["methods"]) - {"gbb"}:
        try:
            import torch
            device = torch.device(cfg["device"])
            if device.type == "cuda" and (not torch.cuda.is_available() or (device.index or 0) >= torch.cuda.device_count()):
                raise RuntimeError(f"GPU unavailable: {device}")
        except Exception as error:
            neural_ok = False
            issues.append(dict(kind="device", detail=str(error), fatal=not cfg["allow_missing_methods"]))
    for method in cfg["methods"]:
        trials = [None] if method == "gbb" else cfg["trial_ids"]
        for trial in trials:
            required = [] if method in ("gbb", "knn") else [checkpoint_path(cfg, method, trial)]
            if method in ("mlp", "knn"):
                required.append(Path(cfg["prism_checkpoint"]))
            missing = [str(p) for p in required if not p.is_file()]
            if missing:
                issues.append(dict(kind="checkpoint", method=method, trial=trial, missing=missing,
                                   fatal=False))
            if method == "gbb" or neural_ok:
                jobs.append(dict(method=method, trial=trial))
                for path in required:
                    if path.is_file() and str(path) not in sources:
                        sources[str(path)] = file_hash(path)
    if not jobs:
        issues.append(dict(kind="empty", detail="No runnable method", fatal=True))
    return dict(ready=not any(i["fatal"] for i in issues), partial=any(i["kind"] != "checkpoint" for i in issues),
                issues=issues, jobs=jobs, groups=groups, source_sha256=sources)


def _dump(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def _csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row)) or ["status"]
    with Path(path).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(predictions):
    from sklearn.metrics import confusion_matrix, f1_score
    keys = sorted({(r["group"], r["method"], str(r["trial"])) for r in predictions})
    summary = []
    for group, method, trial in keys:
        rows = [r for r in predictions if (r["group"], r["method"], str(r["trial"])) == (group, method, trial)]
        ok = [r for r in rows if r["status"] == "ok"]
        result = {k: rows[0][k] for k in ("group", "topology", "n", "method", "trial", "device", "training_ratio")}
        result.update(expected=len(rows), succeeded=len(ok), coverage=len(ok) / len(rows), status="ok" if len(ok) == len(rows) else "incomplete")
        # Never silently calculate the headline F1 on only successful samples.
        if len(ok) == len(rows):
            y, p = [r["truth"] for r in ok], [r["prediction"] for r in ok]
            for average in ("weighted", "macro", "binary"):
                result[f"f1_{average}"] = float(f1_score(y, p, average=average, labels=[0, 1] if average != "binary" else None, zero_division=0))
            tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
            result.update(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp), class_0=y.count(0), class_1=y.count(1))
        summary.append(result)
    return summary


def source_identity(cfg):
    paths = [Path(__file__), PROJECT / "dataset/sis_dataset_generation/generate_sis_dataset.py"]
    for group in cfg["datasets"]:
        paths += [Path(group["path"]) / n for n in ("As.pkl", "numes.pkl", "rs.pkl", "basers.pkl")]
        metadata = Path(group["path"]) / "generation.json"
        if metadata.exists():
            paths.append(metadata)
        if group.get("params_file"):
            paths.append(Path(group["path"]) / group["params_file"])
    if set(cfg["methods"]) - {"gbb"}:
        from experiment_process import generate_resilience_inference_cache as api
        spec = api.TASKS["sis_binary"]
        paths += [Path(api.__file__), PROJECT / "experiment_process/resinf_inference.py"]
        paths += list(spec.dataset_dir.rglob("*.pkl"))
        paths += list(spec.dataset_dir.rglob("generation*.json"))
        if set(cfg["methods"]) & {"knn", "mlp"}:
            paths.append(spec.prism_checkpoint)
        for method in set(cfg["methods"]) & {"mlp", "resinf"}:
            paths += [checkpoint_path(cfg, method, t) for t in cfg["trial_ids"]]
    return dict(config={k:v for k,v in cfg.items() if not k.startswith("_")},
                sources={str(p): file_hash(p) if p.is_file() else None for p in sorted(set(paths))})


def prepare_models(cfg, references):
    if not set(cfg["methods"]) - {"gbb"}:
        return {}
    from experiment_process import generate_resilience_inference_cache as api
    from experiment_process import resinf_inference as ri
    spec, device = api.TASKS["sis_binary"], cfg["device"]
    default_root = api.classifier_model_root(spec, "mlp").parent
    if Path(cfg["checkpoint_root"]) != default_root or Path(cfg["prism_checkpoint"]) != spec.prism_checkpoint:
        raise ValueError("Scaling reuses the original supervised model paths; custom checkpoint roots are unsupported")
    if "mlp" in cfg["methods"]:
        api.train_missing("sis_binary", device, ratios=[cfg["training_ratio"]], trial_ids=cfg["trial_ids"])
    if "resinf" in cfg["methods"]:
        for trial in cfg["trial_ids"]:
            path = checkpoint_path(cfg, "resinf", trial)
            if not path.exists() or path.stat().st_size == 0:
                print(f"Preparing missing ResInf trial {trial} on original train/validation data", flush=True)
                ri.train_one(api, spec, cfg["training_ratio"], trial, device,
                             api.load_split(spec, "train_dataset"), api.evaluation_split(spec, "val_dataset", "resinf"), path,
                             data_sources={str(p): file_hash(p) for p in spec.dataset_dir.rglob("*.pkl")})
            else:
                model = ri.build_model(2, torch.load(path, map_location="cpu", weights_only=False))
                del model
    if "knn" not in cfg["methods"]:
        return {}
    missing = [t for t in cfg["trial_ids"] if str(t) not in references]
    if missing:
        train = api.load_split(spec, "train_dataset")
        encoder = SharedEncoder(api._build_prism(spec, device)).to(device).eval()
        features = api.extract_features(encoder, train, spec.pooling_stats, device)
        labels = api.labels(train)
        for trial in missing:
            indices = api.reference_indices(spec, train, cfg["training_ratio"], trial, "knn")
            if len(indices) < 4:
                raise ValueError("KNN requires at least four reference samples")
            references[str(trial)] = dict(indices=[int(i) for i in indices],
                source_indices=[int(train[i].source_index) for i in indices],
                features=features[indices].tolist(), labels=[labels[i] for i in indices])
        del encoder, features, train
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return references


def run(cfg, output, force=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    record_path = output / "records.json"
    identity = source_identity(cfg)
    previous = json.loads(record_path.read_text(encoding="utf-8")) if record_path.exists() else {}
    saved = previous.get("identity", {})
    compatible = saved.get("config") == identity["config"] and all(
        digest is None or saved.get("sources", {}).get(path) == digest
        for path, digest in identity["sources"].items())
    replace_cache = bool(previous) and (force or not compatible)
    if previous and compatible and not force:
        if previous.get("complete"):
            if previous.get("metrics") != summarize(previous.get("predictions", [])):
                raise ValueError("Cached metrics disagree with raw predictions")
            aggregate(output)[0].to_csv(output / "summary.csv", index=False)
            return True
    else:
        previous = {}
    report = preflight(cfg)
    if not report["ready"]:
        raise ValueError(json.dumps(report["issues"], ensure_ascii=False, indent=2))
    preparation = {**cfg, "methods": list(dict.fromkeys(j["method"] for j in report["jobs"]))}
    references = prepare_models(preparation, previous.get("references", {}))
    identity = source_identity(cfg)  # Newly saved models belong to the final provenance.
    if replace_cache:
        archive = output / "archive" / str(time.time_ns())
        archive.mkdir(parents=True)
        record_path.rename(archive / "records.json")
        if (output / "summary.csv").exists():
            (output / "summary.csv").rename(archive / "summary.csv")
    cfg = {**cfg, "_references": references}
    run_record = {**previous, "identity": identity, "config": identity["config"], "references": references,
               "environment": dict(python=sys.version, platform=platform.platform(), torch=torch.__version__,
                                   device=cfg["device"], cpu_threads=cfg["cpu_threads"]),
               "complete": False, "test_only": cfg["test_only"]}
    predictions, timings, memories, stages, failures = [previous.get(k, []) for k in
        ("predictions", "timings", "memory", "stages", "failures")]
    done = set(previous.get("completed_jobs", []))
    def save():
        run_record.update(predictions=predictions, timings=timings, memory=memories, stages=stages,
                       failures=failures, metrics=summarize(predictions), completed_jobs=sorted(done))
        _dump(record_path, run_record)
    save()
    for group in cfg["datasets"]:
        samples = load_group(group, cfg)
        rng = np.random.default_rng(cfg["seed"])
        selected = set(rng.choice(len(samples), min(cfg["timing_samples"], len(samples)), replace=False).tolist())
        for job in report["jobs"]:
            method, trial = job["method"], job["trial"]
            job_key = f"{group['id']}:{method}:{trial}"
            if job_key in done:
                continue
            print(f"{group['id']} / {method} / trial={trial}", flush=True)
            worker = Worker(cfg, method, trial)
            base = dict(group=group["id"], topology=group["topology"], n=group["n"], method=method,
                        trial=trial, training_ratio=None if method == "gbb" else cfg["training_ratio"],
                        device="cpu" if method == "gbb" else cfg["device"] + ("+cpu(knn)" if method == "knn" else ""))
            try:
                warmed = False
                for index, sample in enumerate(samples):
                    identity = dict(**base, sample_id=sample["id"])
                    if method in ("mlp", "knn"):
                        payload = dict(x=sample["x"])
                    elif method == "resinf":
                        payload = dict(x=sample["x"], A=sample["A"])
                    else:
                        payload = dict(A=sample["A"], params=sample["params"])
                    response = worker.call("predict", payload) if worker.ready["status"] == "ready" else worker.ready
                    record = dict(**identity, truth=sample["y"], observation_indices=json.dumps(sample["observation_indices"]), **response)
                    if method == "gbb":
                        record["cached_prediction"] = sample["cached_gbb"]
                        record["cached_agrees"] = sample["cached_gbb"] == response.get("prediction") if sample["cached_gbb"] is not None else None
                    predictions.append(record)
                    if response["status"] != "ok":
                        failures.append(dict(**identity, phase="predict", **response))
                        continue
                    if index not in selected:
                        continue
                    if not warmed:
                        warm = worker.call("warmup", payload)
                        if warm["status"] != "ok":
                            failures.append(dict(**identity, phase="warmup", **warm))
                            continue
                        warmed = True
                    for command, destination in (("timing", timings), ("stages", stages), ("memory", memories)):
                        measured = worker.call(command, payload)
                        if measured["status"] != "ok":
                            failures.append(dict(**identity, phase=command, **measured))
                            destination.append(dict(**identity, **measured))
                        elif command == "timing":
                            destination.extend(dict(**identity, status="ok", **row) for row in measured["rows"])
                        else:
                            destination.append(dict(**identity, **measured))
            finally:
                worker.close()
            done.add(job_key)
            save()
    run_record["complete"] = not report["partial"] and not failures
    save()
    summary, _ = aggregate(output)
    summary.to_csv(output / "summary.csv", index=False)
    return run_record["complete"]


def aggregate(folder):
    folder = Path(folder)
    import pandas as pd
    payload = json.loads((folder / "records.json").read_text(encoding="utf-8"))
    frames = {name: pd.DataFrame(payload[name]) for name in ("metrics", "timings", "memory", "stages")}
    keys = ["topology", "n", "method"]
    records = {}

    def record(key):
        return records.setdefault(key, dict(zip(keys, key)))

    metrics = frames["metrics"]
    for key, rows in metrics[metrics.status == "ok"].groupby(keys):
        expected = 1 if key[2] == "gbb" else len(payload["config"]["trial_ids"])
        if len(rows) != expected:
            continue
        target = record(key)
        target["completed_trials"] = len(rows)
        for metric in ("f1_weighted", "f1_macro", "f1_binary"):
            target[metric + "_mean"] = rows[metric].mean()
            target[metric + "_std"] = rows[metric].std(ddof=0) if len(rows) > 0 else np.nan
    times = frames["timings"]
    if "total_ms" in times:
        # Repetitions are not independent checkpoints: aggregate them first.
        sample = times[times.status == "ok"].groupby(keys + ["trial", "sample_id"], dropna=False).total_ms.median().reset_index()
        trials = sample.groupby(keys + ["trial"], dropna=False).total_ms.median().reset_index()
        for key, rows in trials.groupby(keys):
            target = record(key)
            target.update(time_ms_mean=rows.total_ms.mean(),
                          time_ms_std=rows.total_ms.std(ddof=0) if len(rows) > 0 else np.nan,
                          time_trials=len(rows))
    memory = frames["memory"]
    if "cpu_peak_rss_sampled_bytes" in memory:
        for key, rows in memory[memory.status == "ok"].groupby(keys):
            for metric in ("cpu_peak_rss_sampled_bytes", "gpu_peak_allocated_bytes", "gpu_peak_reserved_bytes"):
                if metric in rows:
                    target = record(key)
                    peaks = rows.groupby("trial", dropna=False)[metric].max().dropna()
                    target[metric] = rows[metric].max()
                    target[metric + "_mean"] = peaks.mean()
                    target[metric + "_std"] = peaks.std(ddof=0) if len(peaks) > 0 else np.nan
    stages = frames["stages"]
    stage_records = []
    if "representation_ms" in stages:
        valid = stages[(stages.status == "ok") & stages.method.isin(["knn", "mlp"])]
        for key, rows in valid.groupby(keys + ["trial"], dropna=False):
            result = dict(zip(keys + ["trial"], key))
            for metric in ("representation_ms", "classification_ms"):
                result[metric + "_median"] = rows[metric].median()
            result["samples"] = len(rows)
            stage_records.append(result)
    return pd.DataFrame(records.values()), pd.DataFrame(stage_records)


def smoke_test():
    """Small isolated integration test; never writes official data or models."""
    import tempfile
    from unittest.mock import patch
    import yaml
    from torch_geometric.data import Data
    from experiment_process import generate_resilience_inference_cache as api
    from experiment_process import resinf_inference as ri
    from experiment_visualization.plot_sis_scaling import plot
    root = Path(tempfile.mkdtemp(prefix="prism_scaling_TEST_ONLY_"))
    config = dict(methods=["gbb"], device="cpu", test_only=True, trial_ids=[0, 1],
                  warmup=1, repeats=2, timing_samples=2, datasets=[])
    for n in (4, 6):
        folder = root / f"N{n}"
        folder.mkdir()
        matrices = [np.zeros((n,n), dtype=np.float32), np.ones((n,n), dtype=np.float32)-np.eye(n,dtype=np.float32)]
        # Artificial records validate contracts only; not dynamical experiment results.
        values = dict(As=matrices, numes=[np.random.default_rng(n+i).random((16,20,n)) for i in range(2)],
                      rs=[0,1], basers=[0,1])
        for name, value in values.items():
            with (folder / (name+".pkl")).open("wb") as stream:
                pickle.dump(value, stream)
        config["datasets"].append(dict(topology="ER", n=n, path=str(folder), params=dict(delta=2)))
    path = root/"config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    cfg = read_config(path)
    cache = root/"cache"
    assert run(cfg, cache)
    signature = file_hash(cache/"records.json")
    (cache/"summary.csv").unlink()
    with patch(__name__+".Worker", side_effect=AssertionError("cache must be reused")):
        assert run(cfg, cache)
    assert signature == file_hash(cache/"records.json") and (cache/"summary.csv").is_file()
    records = json.loads((cache/"records.json").read_text())
    assert len(records["metrics"]) == 2  # GBB is not replicated by checkpoint count.
    assert all(r["f1_weighted"] == 1 for r in records["metrics"])
    for delta, expected in ((3.,0), (3.-1e-9,1), (3.+1e-9,0)):
        assert gbb_predict(sparse.csr_matrix(np.ones((4,4))-np.eye(4)),delta)["prediction"] == expected
    plot(cache, root/"figures")
    assert all((root/f"figures/SIS_scaling_ER.{ext}").is_file() for ext in ("svg","pdf","png"))
    # Restore a missing job from raw records, without reevaluating the other N.
    records["complete"] = False
    records["completed_jobs"] = ["ER_N4:gbb:None"]
    for key in ("predictions","timings","memory","stages","failures","metrics"):
        records[key] = [r for r in records[key] if r["group"] == "ER_N4"]
    _dump(cache/"records.json", records)
    assert run(cfg, cache)
    assert len(json.loads((cache/"records.json").read_text())["metrics"]) == 2
    spec = api.TASKS["sis_binary"]
    checkpoint = api.available_model_path(spec,"mlp",.1,0)
    if checkpoint is None:
        raise FileNotFoundError("Smoke comparison requires existing SIS MLP 10%, trial 0; will not train it")
    digest = file_hash(checkpoint)
    model, _, _ = api.load_mlp(spec,.1,0,"cpu")
    x = torch.from_numpy(load_group(cfg["datasets"][0],cfg)[0]["x"])
    graph = Data(x=x, edge_index=torch.empty((2,0),dtype=torch.long))
    with torch.inference_mode():
        expected = model(graph)
        z = SharedEncoder(model.encoder).eval()(x)
        torch.testing.assert_close(expected, model.classify(z), rtol=1e-5, atol=1e-6)
    assert file_hash(checkpoint) == digest
    adjacency = sparse.csr_matrix(np.ones((len(x),len(x)))-np.eye(len(x)))
    norm, numerical = normalized_resinf_inputs(x.numpy(), adjacency)
    expanded = np.zeros((len(x)+1,len(x)+1),np.float32)
    expanded[:-1,:-1] = adjacency.toarray()
    expanded[-1,:-1] = 1
    row, col = expanded.sum(1), expanded.sum(0)
    a, b = np.zeros_like(row),np.zeros_like(col)
    np.power(row,-.5,out=a,where=row!=0); np.power(col,-.5,out=b,where=col!=0)
    np.testing.assert_allclose(norm,np.diag(a)@expanded@np.diag(b),rtol=1e-6)
    respath = api.available_model_path(spec,"resinf",.1,0)
    res = ri.build_model(2,torch.load(respath,map_location="cpu",weights_only=False)).eval()
    with torch.inference_mode():
        torch.testing.assert_close(res(torch.from_numpy(numerical),torch.from_numpy(norm)),
            res(torch.from_numpy(numerical),torch.from_numpy(np.diag(a)@expanded@np.diag(b))))
    del model, res
    train = [Data(x=x.clone(), y=torch.tensor([i%2]), source_index=i) for i in range(6)]
    kcfg = {**cfg,"methods":["knn"],"training_ratio":1.,"trial_ids":[0]}
    with patch.object(api,"load_split",return_value=train):
        first = prepare_models(kcfg,{})
        second = prepare_models(kcfg,{})
    assert first == second
    kcfg["_references"] = first
    adapter = NeuralAdapter(kcfg,"knn",0)
    result = adapter.predict(dict(x=x.numpy()))
    assert result == adapter.predict(dict(x=x.numpy()))
    assert first == kcfg["_references"]  # Inference at different N never changes references.
    adapter.predict(dict(x=np.tile(x.numpy(),(2,1,1,1))))
    assert first == kcfg["_references"]
    print(f"PASS: scaling contracts, cache/resume, strict model outputs, KNN, three exports. TEST ONLY: {root}")
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT / "experiment_process/sis_scaling.example.yaml"))
    parser.add_argument("--mode", choices=("preflight", "run"), default="preflight")
    parser.add_argument("--output", help="New output directory; existing directories are never overwritten")
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    try:
        if args.smoke_test:
            smoke_test()
            return 0
        cfg = read_config(args.config)
        if args.mode == "preflight":
            report = preflight(cfg)
            print(json.dumps(report, indent=2, ensure_ascii=False))
            return 0 if report["ready"] else 2
        if not args.output:
            parser.error("--output is required for run")
        return 0 if run(cfg, args.output, args.force_cache) else 3
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
