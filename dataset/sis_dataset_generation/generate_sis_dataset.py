"""SIS 数据生成：原命令及 config.yaml 可继续使用，只需替换本文件。

生成过程不导入 experiment_process、可视化或其他项目脚本。新增安全参数
均有内置默认值，无须给旧配置补充字段；N>=2000 默认保存 CSR。
规模数据按 ER/SF + N 独立发布；重启时校验并跳过完整组，支持恢复旧暂存完整组。
尚未凑齐样本的组会重新生成；不要同时启动多个进程写入同一输出根目录。
"""
from __future__ import annotations
import argparse
from collections.abc import Sequence
from contextlib import ExitStack
import hashlib
import json
import pickle
import platform
import shutil
import time
from pathlib import Path

import networkx as nx
import numpy as np
import yaml
from scipy import sparse
from scipy.integrate import solve_ivp
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent


class TopologyRejected(ValueError):
    """A sampled mother exceeded a topology safety bound; never an ODE failure."""


def resource_guard(estimated_bytes=0, *, max_working_mb=2048):
    # psutil is optional: replacing this script must not require requirements edits.
    try:
        import psutil
        available = psutil.virtual_memory().available
    except ImportError:
        if platform.system() == "Linux":
            fields = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
            available = int(fields["MemAvailable"].split()[0]) * 1024
        elif platform.system() == "Windows":
            import ctypes
            class MemoryStatus(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                    (name, ctypes.c_ulonglong) for name in
                    ("total_physical", "available_physical", "total_page", "available_page",
                     "total_virtual", "available_virtual", "available_extended")]
            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                raise OSError("Cannot query available memory; generation stopped safely")
            available = status.available_physical
        else:
            raise RuntimeError("Memory checks require psutil on platforms other than Windows/Linux")
    limit = min(max_working_mb * 2**20, available * .6)
    if estimated_bytes > limit:
        raise MemoryError(f"Operation needs about {estimated_bytes / 2**20:.1f} MiB; safe allowance {limit / 2**20:.1f} MiB")


def disk_guard(path, additional_bytes):
    if shutil.disk_usage(path).free < additional_bytes + 2**30:
        raise OSError("Insufficient free disk space (including 1 GiB safety reserve); staging is retained")


class PickleListWriter:
    """Stream independent items into a standard pickle list, without retaining its memo.

    Protocol 3 uses explicit BINPUT indices: each independent item may overwrite
    the preceding memo. The outer list is deliberately not memoized. Replace the
    item's STOP by APPEND; ordinary pickle.load still returns a Python list.
    Offsets permit bounded-memory reads without unpickling the entire list.
    """
    def __init__(self, path):
        self.stream = Path(path).open("xb")
        self.stream.write(b"\x80\x03]")
        self.index = []

    def append(self, value):
        payload = pickle.dumps(value, protocol=3)
        start = self.stream.tell()
        self.stream.write(memoryview(payload)[2:-1])
        self.stream.write(b"a")
        self.index.append([start, len(payload) - 2])

    def close(self):
        self.stream.write(b".")
        self.stream.close()


class IndexedPickleList(Sequence):
    """Read one indexed item at a time; old files fall back to a guarded full read."""
    def __init__(self, path, index=None):
        self.path, self.index, self.legacy = Path(path), index, None
        size = self.path.stat().st_size
        if index is None:
            resource_guard(size * 3)
            with self.path.open("rb") as stream:
                self.legacy = pickle.load(stream)
            if type(self.legacy) is not list:
                raise ValueError(f"Expected pickle list: {path}")
        else:
            previous = 3
            for offset, length in index:
                if type(offset) is not int or type(length) is not int or offset != previous or length < 1 or offset + length > size - 1:
                    raise ValueError(f"Invalid pickle index: {path}")
                previous = offset + length
            if previous != size - 1:
                raise ValueError(f"Incomplete pickle index: {path}")
            with self.path.open("rb") as stream:
                header = stream.read(3)
                stream.seek(-1, 2)
                if header != b"\x80\x03]" or stream.read(1) != b".":
                    raise ValueError(f"Invalid streamed-list envelope: {path}")

    def __len__(self):
        return len(self.legacy) if self.legacy is not None else len(self.index)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return [self[i] for i in range(*item.indices(len(self)))]
        if self.legacy is not None:
            return self.legacy[item]
        offset, length = self.index[item]
        resource_guard(length * 3)
        with self.path.open("rb") as stream:
            stream.seek(offset)
            payload = stream.read(length)
        if len(payload) != length or payload[-1:] != b"a":
            raise ValueError(f"Truncated or corrupt indexed pickle item: {self.path}")
        return pickle.loads(payload[:-1] + b".")


class SampleRows(Sequence):
    """Accepted samples are spooled to staging, not accumulated as arrays in RAM."""
    def __init__(self, path):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=False)
        self.paths = []
        self.bytes = 0

    def append(self, row):
        size = row["trajectory"].nbytes + (row["end"].nbytes if row["end"] is not None else 0)
        size += row["A"].data.nbytes + row["A"].indices.nbytes + row["A"].indptr.nbytes
        disk_guard(self.path, 2 * size)
        path = self.path / f"sample_{len(self.paths):08d}.pkl"
        with path.open("xb") as stream:
            pickle.dump(row, stream, protocol=pickle.HIGHEST_PROTOCOL)
        self.paths.append(path)
        self.bytes += path.stat().st_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with self.paths[index].open("rb") as stream:
            return pickle.load(stream)

    def shuffle(self, rng):
        rng.shuffle(self.paths)


def seed_for(seed, *parts):
    return int.from_bytes(hashlib.sha256(str((seed, parts)).encode()).digest()[:8], "little")


def gcc(A):
    if A.shape[0] == 0:
        raise TopologyRejected("Empty mother graph")
    _, labels = sparse.csgraph.connected_components(A, directed=False)
    nodes = np.flatnonzero(labels == np.bincount(labels).argmax())
    return A[nodes][:, nodes].tocsr()


def build_graph(n, topology, rng, fixed=False, max_stubs=10_000_000):
    """原 ER/SF 规则；显式保留节点，合并重边、去除自环。"""
    if not isinstance(n, (int, np.integer)) or n < 2 or topology not in ("ER", "SF"):
        raise ValueError("Require n >= 2 and ER/SF topology")
    resource_guard(int(n) * 2048)
    if topology == "ER":
        k = float(np.log(n) + 1 if fixed else rng.uniform(1, 3))
        graph = nx.fast_gnp_random_graph(n, min(1., k / (n - 1)), seed=int(rng.integers(2**32)))
        params = dict(expected_degree=k)
    else:
        gamma = 2.5 if fixed else float(rng.uniform(3, 4) if rng.random() < .3 else rng.uniform(2.5, 3))
        k0 = 2. if fixed else float(rng.uniform(1.5, 3))
        sampled = np.round(k0 * np.maximum(rng.random(n), np.finfo(float).tiny) ** (1 / (1 - gamma)))
        if not np.isfinite(sampled).all() or sampled.sum() > max_stubs:
            raise TopologyRejected("SF stub count exceeds safety limit")
        degrees = sampled.astype(np.int64)
        resource_guard(int(degrees.sum()) * 80 + n * 64)
        stubs = np.repeat(np.arange(n), degrees)
        rng.shuffle(stubs)
        stubs = stubs[:len(stubs) // 2 * 2]
        u, v = stubs[::2], stubs[1::2]
        keep = u != v
        u, v = u[keep], v[keep]
        A = sparse.csr_matrix((np.ones(2 * len(u)), (np.r_[u, v], np.r_[v, u])), shape=(n, n))
        A.data[:] = 1  # Same simple graph as merging edges in networkx.Graph.
        params = dict(gamma=gamma, k0=k0)
        return A, params
    A = sparse.csr_matrix(nx.to_scipy_sparse_array(graph, nodelist=range(n), dtype=np.float64))
    return A, params


def topology_family(A, rng, target_n=None, *, max_edge_attempts=None, diagnostics=None):
    """训练和分类唯一共用的候选来源；每个增广都从母网络独立产生。"""
    A = gcc(A)
    diagnostics = diagnostics if diagnostics is not None else {}
    if target_n is not None and A.shape[0] < target_n:
        diagnostics["subtarget_mothers"] = diagnostics.get("subtarget_mothers", 0) + 1
        return
    yield A, "original_gcc"
    n = A.shape[0]
    if n < 20:
        return
    removals = range(1, n - 10, 2)
    if target_n is not None and max_edge_attempts is not None:
        removals = [n - target_n] if n - target_n in removals else []
    for removed in removals:
        for _ in range(100):
            keep = np.sort(rng.choice(n, n - removed, replace=False))
            B = A[keep][:, keep].tocsr()
            if B.nnz:
                yield B, f"node_random_{removed}"
                break
        else:
            diagnostics["empty_node_augmentations"] = diagnostics.get("empty_node_augmentations", 0) + 1
            continue  # 当前删除比例未获得含边候选，跳过；不终止整批生成。
    edges = np.column_stack(sparse.triu(A, k=1).nonzero())
    for attempt, removed in enumerate(range(1, len(edges) - 15, 2)):
        if max_edge_attempts is not None and attempt >= max_edge_attempts:
            diagnostics["edge_search_limits"] = diagnostics.get("edge_search_limits", 0) + 1
            return
        keep = np.ones(len(edges), dtype=bool)
        keep[rng.choice(len(edges), removed, replace=False)] = False
        u, v = edges[keep].T
        B = sparse.csr_matrix((np.ones(2 * len(u)), (np.r_[u, v], np.r_[v, u])), shape=A.shape)
        yield gcc(B), f"edge_random_{removed}"


def sis_rhs(t, x, A, delta, infection_rate=1.):
    return -delta * x + infection_rate * (1 - x) * (A @ x)


def gbb(A, delta, infection_rate=1.):
    degree = np.asarray(A.sum(1)).ravel()
    total = degree.sum()
    beta = float(degree @ degree / total) if total else 0.
    return int(infection_rate * beta > delta)


def simulate(A, delta, cfg, rng, terminal):
    """短时采样与终态检查分离；延长终态不改变模型观测。"""
    times = np.linspace(0, cfg["T"], cfg["time_points"])[:cfg["saved_points"]]
    shape = (cfg["trajectories"], len(times), A.shape[0])
    resource_guard(int(np.prod(shape)) * 16 + A.shape[0] * 1024)
    trajectories = np.empty(shape, dtype=np.float64)
    ends = np.empty((shape[0], shape[2]), dtype=np.float64) if terminal else None
    end_times = []
    rate = cfg["infection_rate"]
    method = cfg.get("method", "auto")
    large = A.shape[0] >= cfg.get("explicit_solver_threshold", 2000)
    method = ("RK45" if large else "LSODA") if method == "auto" else method
    if large and method != "RK45":
        raise ValueError("Large SIS systems require RK45 to avoid a dense implicit Jacobian")
    deadline = 0.

    def integrate(x, start, stop, evaluation):
        def rhs(t, state):
            if time.monotonic() > deadline:
                raise TimeoutError("SIS trajectory exceeded trajectory_timeout_seconds; no label assigned")
            return sis_rhs(t, state, A, delta, rate)
        # Tighten explicit integration enough for the requested absolute residual.
        rtol, atol = cfg["rtol"], cfg["atol"]
        if method == "RK45" and terminal:
            tolerance = cfg["residual_tol"] * .01 / max(1., delta + rate * np.asarray(A.sum(1)).max())
            rtol, atol = min(rtol, tolerance), min(atol, tolerance)
        sol = solve_ivp(rhs, (start, stop), x,
                        method=method, rtol=rtol, atol=atol, t_eval=evaluation)
        if not sol.success or sol.y.shape[1] != len(evaluation) or not np.isfinite(sol.y).all():
            raise RuntimeError(f"SIS solver failed: {sol.message}")
        if sol.y.min() < -1e-7 or sol.y.max() > 1 + 1e-7:
            raise RuntimeError("SIS trajectory outside [0,1] beyond numerical tolerance")
        return sol.y

    for trajectory in range(cfg["trajectories"]):
        if large:
            print(f"SIS N={A.shape[0]}: trajectory {trajectory + 1}/{cfg['trajectories']}, solver={method}", flush=True)
        deadline = time.monotonic() + cfg.get("trajectory_timeout_seconds", 1800)
        initial = rng.uniform(0, 1, A.shape[0])
        short = integrate(initial, 0., float(times[-1]), times)
        trajectories[trajectory] = short.T
        if not terminal:
            continue
        state, start, stop = short[:, -1], float(times[-1]), float(cfg["T"])
        while True:
            evaluation = [stop - 1, stop]
            if evaluation[0] <= start:
                raise ValueError("Terminal check requires saved trajectory to end before T-1")
            tail = integrate(state, start, stop, evaluation)
            state = tail[:, -1]
            change = np.max(np.abs(state - tail[:, -2]))
            residual = np.max(np.abs(sis_rhs(stop, state, A, delta, rate)))
            if change <= cfg["change_tol"] and residual <= cfg["residual_tol"]:
                break
            if stop >= cfg["max_T"]:
                raise RuntimeError(f"SIS did not converge by t={stop}; delta={delta}, N={A.shape[0]}")
            next_stop = min(2 * stop, cfg["max_T"])
            if next_stop - stop <= 1:
                state, start = tail[:, 0], stop - 1
            else:
                start = stop
            stop = next_stop
        ends[trajectory] = state
        end_times.append(stop)
    return trajectories, ends, end_times


def sample(A, source, delta, cfg, seed, terminal):
    trajectory, end, end_times = simulate(A, delta, cfg["simulation"], np.random.default_rng(seed), terminal)
    return dict(A=A, trajectory=trajectory, end=end,
                label=int(end.mean() >= cfg["label_threshold"]) if end is not None else None,
                baser=gbb(A, delta, cfg["simulation"]["infection_rate"]),
                params=dict(delta=float(delta), infection_rate=cfg["simulation"]["infection_rate"]),
                source={**source, "actual_n": A.shape[0], "simulation_seed": seed, "end_times": end_times,
                        "solver": ("RK45" if A.shape[0] >= cfg["simulation"].get("explicit_solver_threshold", 2000) else "LSODA")
                        if cfg["simulation"].get("method", "auto") == "auto" else cfg["simulation"]["method"]})


def graph_hash(A):
    # Exact ordered adjacency duplicates, not a graph-isomorphism claim.
    A = A.copy(); A.sort_indices()
    return hashlib.sha256(str(A.shape).encode() + A.indptr.astype("int64").tobytes() + A.indices.astype("int64").tobytes()).hexdigest()


def save_group(path, rows, cfg, suffix="", supervised=False, terminal=False, stats=None):
    path.mkdir(parents=True, exist_ok=True)
    names = ["As", "numes", "basers"]
    if supervised:
        names += ["rs", "x_last"]
    elif terminal:
        names += ["xlast"]
    if cfg["save_gbb_times"]:
        names += ["gbb_times"]
    parameters, sources, storage, counts = [], [], [], [0, 0]
    with ExitStack() as stack:
        writers = {}
        for name in names:
            writer = PickleListWriter(path / f"{name}{suffix}.pkl")
            stack.callback(writer.close)
            writers[name] = writer
        for row in rows:
            a = row["A"].astype(np.float32)
            use_csr = a.shape[0] >= cfg.get("large_graph_threshold", 2000)
            if not use_csr:
                resource_guard(a.shape[0] ** 2 * 12)
                a = a.toarray()
            disk_guard(path, row["trajectory"].nbytes * 2 + (a.data.nbytes if use_csr else a.nbytes) * 2)
            values = dict(As=a, numes=row["trajectory"], basers=row["baser"])
            if supervised:
                values.update(rs=row["label"], x_last=row["end"])
                counts[row["label"]] += 1
            elif terminal:
                values["xlast"] = row["end"]
            if cfg["save_gbb_times"]:
                for _ in range(cfg["gbb_warmup"]):
                    gbb(row["A"], **row["params"])
                durations = []
                for _ in range(cfg["gbb_repeats"]):
                    start = time.perf_counter_ns()
                    gbb(row["A"], **row["params"])
                    durations.append((time.perf_counter_ns() - start) / 1e9)
                values["gbb_times"] = durations
            for name, value in values.items():
                writers[name].append(value)
            parameters.append(row["params"])
            sources.append(row["source"])
            storage.append("csr" if use_csr else "dense")
    dump(path / f"sis_params{suffix}.json", parameters)
    dump(path / f"generation{suffix}.json", dict(samples=len(rows), sources=sources,
        pickle_index={f"{name}{suffix}.pkl": writer.index for name, writer in writers.items()},
        adjacency_storage=storage,
        class_counts={str(y): counts[y] for y in (0, 1)} if supervised else None,
        stats=stats, gbb_timing=dict(unit="seconds", warmup=cfg["gbb_warmup"], repeats=cfg["gbb_repeats"],
        cpu_threads=1, processor=platform.processor(), platform=platform.platform(),
        scope="CSR degree calculation through binary classification; serial CPU") if cfg["save_gbb_times"] else None))


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def generate_candidates(cfg, namespace, size_range, topology):
    diagnostics = cfg.get("_diagnostics", {})
    target_n = cfg.get("_target_n")
    for mother in range(cfg["max_mothers"]):
        seed = seed_for(cfg["seed"], namespace, mother)
        rng = np.random.default_rng(seed)
        n = int(rng.integers(size_range[0], size_range[1] + 1))
        kind = topology or ("ER" if rng.random() < .5 else "SF")
        try:
            A, params = build_graph(n, kind, rng, max_stubs=cfg.get("max_sf_stubs", 10_000_000))
        except TopologyRejected as error:
            diagnostics["rejected_mothers"] = diagnostics.get("rejected_mothers", 0) + 1
            print(f"{namespace}: skip mother={mother}: {error}", flush=True)
            continue
        source = dict(mother_id=f"{namespace}:{mother}", topology=kind, topology_seed=seed,
                      initial_n=n, topology_parameters=params)
        limit = cfg.get("max_edge_augmentations", 128) if target_n is not None and target_n >= cfg.get("large_graph_threshold", 2000) else None
        yield mother, source, topology_family(A, rng, target_n, max_edge_attempts=limit, diagnostics=diagnostics)


def classification(cfg, namespace, count, seen, n=None, topology=None, spool_dir=None):
    if count % 2:
        raise ValueError("Balanced classification count must be even")
    rows, counts, attempts = SampleRows(spool_dir) if spool_dir is not None else [], [0, 0], 0
    diagnostics = {}
    candidate_cfg = {**cfg, "_target_n": n, "_diagnostics": diagnostics}
    size_range = [n, 2 * n] if n else cfg["initial_size_range"]
    for mother, source, family in generate_candidates(candidate_cfg, namespace, size_range, topology):
        for index, (A, augmentation) in enumerate(family):
            if n is None and A.shape[0] <= 20:
                continue  # Original supervised loaders discard these graphs.
            if n is not None and A.shape[0] != n:
                continue
            fingerprint = graph_hash(A)
            if fingerprint in seen:
                continue
            attempts += 1
            seed = seed_for(cfg["seed"], namespace, mother, index, "simulation")
            rng = np.random.default_rng(seed)
            bounds = cfg["classification_delta"]
            delta = int(rng.integers(bounds[0], bounds[1] + 1))
            print(f"{namespace}: simulate candidate={attempts}, mother={mother}, N={A.shape[0]}, delta={delta}, accepted={counts}", flush=True)
            try:
                row = sample(A, {**source, "augmentation": augmentation}, delta, cfg, seed, True)
            except Exception as error:
                raise RuntimeError(f"{namespace}: candidate={attempts}, mother={mother}, seed={seed}, accepted={counts}: {error}") from error
            label = row["label"]
            if counts[label] < count // 2:
                rows.append(row); counts[label] += 1; seen.add(fingerprint)
                print(f"{namespace}: {counts}, candidates={attempts}", flush=True)
            if len(rows) == count:
                rng = np.random.default_rng(seed_for(cfg["seed"], namespace, "shuffle"))
                rows.shuffle(rng) if isinstance(rows, SampleRows) else rng.shuffle(rows)
                return rows, dict(candidates=attempts, mothers=mother + 1, topology_diagnostics=diagnostics)
            if attempts >= cfg["max_candidates"]:
                raise RuntimeError(f"{namespace}: candidate limit reached, counts={counts}")
            if n is not None:
                break  # 每个母网络在同一规模组最多贡献一个候选。
    raise RuntimeError(f"{namespace}: mother limit reached, counts={counts}")


def perturbations(A, strategy, minimum, rng):
    ids = np.arange(A.shape[0]); removed = 0
    while True:
        yield A, ids.tolist(), removed
        n = A.shape[0]
        if n <= minimum or not A.nnz:
            break
        number = min(n - minimum, int(np.ceil(.01 * n)) if strategy == "random" else 1)
        if strategy == "random":
            delete = rng.choice(n, number, replace=False)
        else:
            degree = np.asarray(A.sum(1)).ravel()
            delete = np.argsort(-degree, kind="stable")[:number]
        keep = np.ones(n, dtype=bool); keep[delete] = False
        A = A[keep][:, keep].tocsr(); ids = ids[keep]; removed += number


def run(cfg, output, seen=None):
    validate_config(cfg)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    dump(output / "config.json", cfg)
    seen, evaluation = set() if seen is None else seen, []
    try:
        with threadpool_limits(limits=1):
            if "train" in cfg["modes"]:
                rows = SampleRows(output / ".samples" / "train")
                for mother, source, family in generate_candidates(cfg, "train", cfg["initial_size_range"], None):
                    if mother >= cfg["train_mothers"]:
                        break
                    for index, (A, augmentation) in enumerate(family):
                        rows.append(sample(A, {**source, "augmentation": augmentation}, cfg["delta"], cfg,
                                           seed_for(cfg["seed"], "train", mother, index), False))
                        seen.add(graph_hash(A))
                save_group(output / "SIS", rows, cfg)
            if "perturbation" in cfg["modes"]:
                prior_hashes = seen.copy()
                for kind in ("ER", "SF"):
                    rng = np.random.default_rng(seed_for(cfg["seed"], "perturbation", kind))
                    A, params = build_graph(cfg["perturbation_n"], kind, rng, fixed=True)
                    for strategy in ("random", "degree"):
                        rows = SampleRows(output / ".samples" / f"{kind}_{strategy}")
                        prng = np.random.default_rng(seed_for(cfg["seed"], kind, strategy))
                        for index, (B, ids, removed) in enumerate(perturbations(A, strategy, cfg["min_nodes"], prng)):
                            if graph_hash(B) in prior_hashes:
                                raise RuntimeError("Perturbation sample duplicates a training adjacency; use another seed")
                            source = dict(mother_id=f"perturbation:{kind}", topology_parameters=params,
                                          topology_seed=seed_for(cfg["seed"], "perturbation", kind),
                                          original_node_ids=ids, removed=removed)
                            rows.append(sample(B, source, cfg["delta"], cfg,
                                               seed_for(cfg["seed"], "perturbation", kind, strategy, index), True))
                            seen.add(graph_hash(B))
                        save_group(output / "SIS_test", rows, cfg, f"_{kind}_{strategy}", terminal=True)
            if "classification" in cfg["modes"]:
                for split, count in cfg["classification_splits"].items():
                    if count:
                        rows, stats = classification(cfg, f"classification:{split}", count, seen,
                                                     spool_dir=output / ".samples" / split)
                        save_group(output / "SIS_supervised" / split, rows, cfg, supervised=True, stats=stats)
                for kind in cfg["scaling_topologies"] if cfg["scaling_count"] else []:
                    for n in cfg["scaling_sizes"]:
                        rows, stats = classification(cfg, f"scaling:{kind}:{n}", cfg["scaling_count"], seen, n, kind,
                                                     spool_dir=output / ".samples" / f"{kind}_N{n}")
                        path = output / f"SIS_scaling/{kind}/N{n}"
                        save_group(path, rows, cfg, supervised=True, stats=stats)
                        evaluation.append(dict(topology=kind, n=n, path=str(path), params_file="sis_params.json"))
        if evaluation:
            with (output / "scaling_evaluation.yaml").open("w", encoding="utf-8") as stream:
                yaml.safe_dump(dict(datasets=evaluation, methods=["knn", "mlp", "resinf", "gbb"],
                                    training_ratio=.1, trial_ids=list(range(10))), stream)
        dump(output / "status.json", dict(status="complete"))
    except BaseException as error:
        dump(output / "status.json", dict(status="failed", error=str(error)))
        raise


def read_config(path):
    with Path(path).open(encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    validate_config(cfg)
    return cfg


def validate_config(cfg):
    if not set(cfg["modes"]) <= {"train", "perturbation", "classification"}:
        raise ValueError("Unknown mode")
    if not cfg["modes"] or len(set(cfg["modes"])) != len(cfg["modes"]):
        raise ValueError("Select unique, nonempty modes")
    if set(cfg["classification_splits"]) - {"train_dataset", "val_dataset", "test_dataset"}:
        raise ValueError("Use original supervised split names")
    if set(cfg["scaling_topologies"]) - {"ER", "SF"}:
        raise ValueError("Only ER/SF supported")
    for count in [*cfg["classification_splits"].values(), cfg["scaling_count"]]:
        if type(count) is not int or count < 0 or count % 2:
            raise ValueError("Balanced group counts must be nonnegative even integers")
    for key in ("max_mothers", "max_candidates", "gbb_repeats"):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if "train" in cfg["modes"] and not 1 <= cfg["train_mothers"] <= cfg["max_mothers"]:
        raise ValueError("max_mothers must cover train_mothers")
    for key in ("scaling_sizes", "scaling_topologies"):
        if len(set(cfg[key])) != len(cfg[key]):
            raise ValueError(f"Duplicate {key}")
    if any(type(n) is not int or n <= 20 for n in cfg["scaling_sizes"]):
        raise ValueError("Scaling sizes must be integers greater than 20")
    for key in ("classification_delta", "initial_size_range"):
        bounds = cfg[key]
        if len(bounds) != 2 or any(type(v) is not int or v < 1 for v in bounds) or bounds[0] > bounds[1]:
            raise ValueError(f"Invalid inclusive integer range: {key}")
    if cfg["initial_size_range"][0] < 2:
        raise ValueError("Initial networks require at least two nodes")
    for key, default in (("large_graph_threshold", 2000), ("max_edge_augmentations", 128), ("max_sf_stubs", 10_000_000)):
        if type(cfg.get(key, default)) is not int or cfg.get(key, default) < 1:
            raise ValueError(f"{key} must be a positive integer")
    if type(cfg["gbb_warmup"]) is not int or cfg["gbb_warmup"] < 0:
        raise ValueError("gbb_warmup must be a nonnegative integer")
    if "perturbation" in cfg["modes"] and not 2 <= cfg["min_nodes"] <= cfg["perturbation_n"]:
        raise ValueError("Require 2 <= min_nodes <= perturbation_n")
    s = cfg["simulation"]
    for key in ("saved_points", "time_points", "trajectories"):
        if type(s[key]) is not int:
            raise ValueError(f"{key} must be an integer")
    for key in ("T", "max_T", "rtol", "atol", "change_tol", "residual_tol"):
        if not np.isfinite(s[key]) or s[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if s.get("method", "auto") not in ("auto", "RK45", "LSODA"):
        raise ValueError("SIS solver method must be auto, RK45 or LSODA")
    if not np.isfinite(s.get("trajectory_timeout_seconds", 1800)) or s.get("trajectory_timeout_seconds", 1800) <= 0:
        raise ValueError("trajectory_timeout_seconds must be finite and positive")
    if type(s.get("explicit_solver_threshold", 2000)) is not int or s.get("explicit_solver_threshold", 2000) < 1:
        raise ValueError("explicit_solver_threshold must be a positive integer")
    if not 2 <= s["saved_points"] <= s["time_points"] or s["trajectories"] < 1:
        raise ValueError("Invalid trajectory shape")
    if s["max_T"] < s["T"] or s["T"] <= 1:
        raise ValueError("Invalid simulation duration")
    if np.linspace(0, s["T"], s["time_points"])[s["saved_points"] - 1] >= s["T"] - 1:
        raise ValueError("Saved observation horizon must precede terminal check at T-1")
    if (not np.isfinite(s["infection_rate"]) or s["infection_rate"] < 0 or
            not np.isfinite(cfg["delta"]) or cfg["delta"] <= 0 or
            not np.isfinite(cfg["label_threshold"]) or not 0 < cfg["label_threshold"] < 1):
        raise ValueError("Invalid SIS parameters")
    return cfg


def validate_group(folder):
    """Validate synchronized legacy pickle groups before any publication."""
    for adjacency in folder.glob("As*.pkl"):
        suffix = adjacency.stem[2:]
        metadata = json.loads((folder / f"generation{suffix}.json").read_text(encoding="utf-8"))
        index = metadata.get("pickle_index", {})
        matrices = IndexedPickleList(adjacency, index.get(adjacency.name))
        if not len(matrices):
            raise ValueError(f"Empty group: {adjacency}")
        arrays = {}
        for file in folder.glob("*.pkl"):
            if file.stem.endswith(suffix) if suffix else "_" not in file.stem or file.stem in (
                    "x_last", "binary_rs", "three_class_rs", "binary_basers", "three_class_basers", "gbb_times"):
                values = matrices if file == adjacency else IndexedPickleList(file, index.get(file.name))
                if len(values) != len(matrices):
                    raise ValueError(f"Unsynchronized file: {file}")
                arrays[file.stem.removesuffix(suffix) if suffix else file.stem] = values
        neuronal = "binary_basers" in arrays
        supervised = "rs" in arrays or "three_class_rs" in arrays
        required = {"As", "numes"} | ({"binary_basers", "three_class_basers"} if neuronal else {"basers"})
        if supervised:
            required |= {"binary_rs", "three_class_rs"} if neuronal else {"rs", "x_last"}
        if suffix:
            required.add("xlast")
        if not required <= arrays.keys():
            raise ValueError(f"Missing required files: {required - arrays.keys()}")
        information = folder / f"generation{suffix}.json"
        metadata = json.loads(information.read_text(encoding="utf-8"))
        if metadata["samples"] != len(matrices) or len(metadata["sources"]) != len(matrices):
            raise ValueError("Source metadata/sample correspondence mismatch")
        for i, a in enumerate(matrices):
            x = arrays["numes"][i]
            dtype = np.float64 if neuronal and supervised else np.float32
            if sparse.issparse(a):
                valid = sparse.isspmatrix_csr(a) and a.shape[0] == a.shape[1] and not (a != a.T).nnz and np.isin(a.data, [0, 1]).all() and not a.diagonal().any()
            else:
                valid = isinstance(a, np.ndarray) and a.ndim == 2 and a.shape[0] == a.shape[1] and np.array_equal(a, a.T) and np.isin(a, [0, 1]).all() and not np.diag(a).any()
            if not valid or a.dtype != dtype:
                raise ValueError(f"Invalid adjacency: {adjacency}, sample {i}")
            if x.dtype != np.float64 or x.ndim != 3 or x.shape[2] != a.shape[0] or not np.isfinite(x).all():
                raise ValueError("Invalid trajectory dtype/axes/values")
            for name in ("xlast", "x_last"):
                if name in arrays:
                    end = arrays[name][i]
                    if end.dtype != np.float64 or end.shape != (len(x), a.shape[0]) or not np.isfinite(end).all():
                        raise ValueError("Invalid terminal data")
            for name in ("rs", "basers", "binary_rs", "binary_basers", "three_class_rs", "three_class_basers"):
                if name in arrays:
                    value = arrays[name][i]
                    if type(value) is not int or value not in (range(3) if name.startswith("three_class") else range(2)):
                        raise ValueError("Invalid classification value")
            for ending in ("rs", "basers"):
                if "three_class_" + ending in arrays:
                    if arrays["binary_" + ending][i] != int(arrays["three_class_" + ending][i] == 2):
                        raise ValueError("Binary/three-class mapping mismatch")
        if supervised:
            labels = arrays["three_class_rs"] if neuronal else arrays["rs"]
            counts = np.bincount(labels, minlength=3 if neuronal else 2)
            if len(set(counts)) != 1:
                raise ValueError(f"Unfulfilled class quotas: {counts}")


def scaling_signature(cfg):
    """Generation semantics only: changing requested sizes/retry limits is not new data."""
    return dict(seed=cfg["seed"], classification_delta=cfg["classification_delta"],
                label_threshold=cfg["label_threshold"],
                simulation={**dict(method="auto", explicit_solver_threshold=2000,
                                   trajectory_timeout_seconds=1800), **cfg["simulation"]},
                large_graph_threshold=cfg.get("large_graph_threshold", 2000),
                max_edge_augmentations=cfg.get("max_edge_augmentations", 128),
                max_sf_stubs=cfg.get("max_sf_stubs", 10_000_000))


def reusable_scaling(folder, cfg, kind, n, seen):
    """Return verified adjacency fingerprints, or a reason to regenerate. No ODE rerun."""
    required = ("As.pkl", "numes.pkl", "rs.pkl", "basers.pkl", "x_last.pkl",
                "sis_params.json", "generation.json")
    if not all((folder / name).is_file() for name in required):
        return None, "missing required files"
    try:
        metadata = json.loads((folder / "generation.json").read_text(encoding="utf-8"))
        previous = metadata.get("configuration")
        if previous is None:
            # Older completed staging groups stored configuration at the run root.
            previous = json.loads((folder.parents[2] / "config.json").read_text(encoding="utf-8"))
        if scaling_signature(previous) != scaling_signature(cfg):
            return None, "generation parameters differ"
        if metadata["samples"] != cfg["scaling_count"]:
            return None, "sample count differs"
        validate_group(folder)
        indices = metadata.get("pickle_index", {})
        columns = {name: IndexedPickleList(folder / (name + ".pkl"), indices.get(name + ".pkl"))
                   for name in ("As", "numes", "rs", "basers", "x_last")}
        params = json.loads((folder / "sis_params.json").read_text(encoding="utf-8"))
        if len(params) != metadata["samples"]:
            return None, "parameter count differs"
        hashes, mothers = set(), set()
        s = cfg["simulation"]
        for i, source in enumerate(metadata["sources"]):
            a = sparse.csr_matrix(columns["As"][i])
            x, end, p = columns["numes"][i], columns["x_last"][i], params[i]
            if a.shape != (n, n) or x.shape != (s["trajectories"], s["saved_points"], n):
                return None, "node count or trajectory dimensions differ"
            if source.get("topology") != kind or not source.get("mother_id", "").startswith(f"scaling:{kind}:{n}:"):
                return None, "topology/source metadata do not match"
            if source["mother_id"] in mothers:
                return None, "repeated mother network"
            mothers.add(source["mother_id"])
            low, high = cfg["classification_delta"]
            if not low <= p["delta"] <= high or int(p["delta"]) != p["delta"] or p["infection_rate"] != s["infection_rate"]:
                return None, "SIS parameters differ"
            if columns["rs"][i] != int(end.mean() >= cfg["label_threshold"]) or columns["basers"][i] != gbb(a, **p):
                return None, "labels or GBB predictions disagree"
            if max(np.abs(sis_rhs(0, state, a, **p)).max() for state in end) > s["residual_tol"]:
                return None, "terminal residual does not satisfy tolerance"
            digest = graph_hash(a)
            if digest in seen or digest in hashes:
                return None, "duplicate adjacency across requested samples"
            hashes.add(digest)
        if cfg["save_gbb_times"]:
            timing = metadata.get("gbb_timing") or {}
            times = IndexedPickleList(folder / "gbb_times.pkl", indices.get("gbb_times.pkl"))
            if timing.get("warmup") != cfg["gbb_warmup"] or timing.get("repeats") != cfg["gbb_repeats"] or len(times) != len(params):
                return None, "GBB timing configuration differs"
            if any(len(t) != cfg["gbb_repeats"] or not np.isfinite(t).all() or np.any(np.asarray(t) <= 0) for t in times):
                return None, "invalid GBB timings"
        return hashes, "complete and compatible"
    except (OSError, ValueError, KeyError, TypeError, IndexError, EOFError, pickle.UnpicklingError) as error:
        return None, str(error)


def publish_scaling(cfg, output):
    """Commit each scale independently; completed scales survive a later failure."""
    import uuid
    validate_config(cfg)
    root = Path(output).resolve()
    stage = root / ".staging" / (time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8])
    stage.mkdir(parents=True)
    dump(stage / "config.json", cfg)
    seen, completed, evaluation = set(), [], []
    def record(status, error=None):
        dump(stage / "status.json", dict(status=status, completed_groups=completed, error=error))
        (stage / "scaling_evaluation.yaml").write_text(yaml.safe_dump(dict(
            datasets=evaluation, methods=["knn", "mlp", "resinf", "gbb"],
            training_ratio=.1, trial_ids=list(range(10)))), encoding="utf-8")
    record("running")
    try:
        # Non-scaling modes retain their original grouped publication protocol.
        other = {**cfg, "scaling_count": 0}
        if set(cfg["modes"]) - {"classification"} or any(cfg["classification_splits"].values()):
            publish_run(run, other, root, _incremental=False)
            for name in ("SIS", "SIS_test", "SIS_supervised"):
                for file in (root / name).rglob("As*.pkl"):
                    info = file.with_name("generation" + file.stem[2:] + ".json")
                    meta = json.loads(info.read_text(encoding="utf-8"))
                    seen.update(graph_hash(sparse.csr_matrix(a)) for a in IndexedPickleList(file, meta.get("pickle_index", {}).get(file.name)))
        for kind in cfg["scaling_topologies"]:
            for n in cfg["scaling_sizes"]:
                relative = Path("SIS_scaling") / kind / f"N{n}"
                target = root / relative
                print(f"VERIFY {kind}/N{n}: checking existing data", flush=True)
                hashes, reason = reusable_scaling(target, cfg, kind, n, seen)
                if hashes is not None:
                    print(f"SKIP {kind}/N{n}: verified {cfg['scaling_count']} samples", flush=True)
                    seen.update(hashes)
                else:
                    print(f"CHECK {kind}/N{n}: {reason}", flush=True)
                    recovered = None
                    for candidate in sorted((root / ".staging").glob(f"*/SIS_scaling/{kind}/N{n}"), reverse=True):
                        hashes, _ = reusable_scaling(candidate, cfg, kind, n, seen)
                        if hashes is not None:
                            recovered = candidate
                            break
                    one = {**cfg, "modes": ["classification"], "classification_splits": {},
                           "scaling_topologies": [kind], "scaling_sizes": [n]}
                    def generate_one(config, destination):
                        if recovered is None:
                            run(config, destination, seen=seen)
                        else:
                            destination.mkdir(parents=True)
                            dump(destination / "config.json", config)
                            disk_guard(root, sum(p.stat().st_size for p in recovered.iterdir() if p.is_file()))
                            shutil.copytree(recovered, destination / relative)
                            info = destination / relative / "generation.json"
                            metadata = json.loads(info.read_text(encoding="utf-8"))
                            metadata["recovered_from"] = str(recovered)
                            metadata["original_generator_sha256"] = metadata.get("generator_sha256")
                            dump(info, metadata)
                    publish_run(generate_one, one, root, _incremental=False)
                    if recovered is not None:
                        seen.update(hashes)
                        print(f"RECOVER {kind}/N{n}: {recovered}", flush=True)
                completed.append(f"{kind}/N{n}")
                evaluation.append(dict(topology=kind, n=n, path=str(target), params_file="sis_params.json"))
                record("running")
        record("complete")
    except BaseException as error:
        record("failed", str(error))
        raise
    print(f"All requested scale groups ready; run metadata: {stage}", flush=True)
    return stage


def publish_run(generator, cfg, output, *, _incremental=True):
    """Stage and validate all requested groups; rollback publication on failure.

    Only resolved descendants of the explicitly supplied dataset root are renamed.
    Failed staging is retained for diagnosis; existing data are untouched.
    """
    if _incremental and generator is run and "classification" in cfg["modes"] and cfg["scaling_count"] and cfg["scaling_sizes"]:
        return publish_scaling(cfg, output)
    import uuid
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    stage = root / ".staging" / run_id
    generator(cfg, stage)
    groups = sorted({p.parent for p in stage.rglob("As*.pkl")})
    if not groups:
        raise ValueError("No requested nonempty data groups were generated")
    for folder in groups:
        validate_group(folder)
        relative = folder.relative_to(stage)
        target = (root / relative).resolve()
        target.relative_to(root)
        if target == root or relative.parts[0] not in {
                "SIS", "SIS_test", "SIS_supervised", "SIS_scaling",
                "Neuronal", "Neuronal_test", "Neuronal_supervised"}:
            raise ValueError(f"Unsafe publication target: {target}")
        for info in folder.glob("generation*.json"):
            metadata = json.loads(info.read_text(encoding="utf-8"))
            metadata.update(dataset_version=run_id, configuration=cfg,
                            generator_sha256=file_digest(Path(generator.__code__.co_filename)))
            dump(info, metadata)
    # Prepare final paths before the first destructive rename.
    evaluation = stage / "scaling_evaluation.yaml"
    if evaluation.exists():
        config = yaml.safe_load(evaluation.read_text(encoding="utf-8"))
        for group in config["datasets"]:
            group["path"] = str(root / Path(group["path"]).relative_to(stage))
        evaluation.write_text(yaml.safe_dump(config), encoding="utf-8")
    moved, archived = [], []
    try:
        for folder in groups:
            relative = folder.relative_to(stage)
            target = (root / relative).resolve()
            backup = (root / "archive" / run_id / relative).resolve()
            backup.relative_to(root / "archive")
            if target.exists():
                backup.parent.mkdir(parents=True, exist_ok=True)
                target.rename(backup)
                archived.append((backup, target))
            target.parent.mkdir(parents=True, exist_ok=True)
            folder.rename(target)
            moved.append((target, folder))
    except BaseException:
        for target, folder in reversed(moved):
            folder.parent.mkdir(parents=True, exist_ok=True)
            target.rename(folder)
        for backup, target in reversed(archived):
            backup.rename(target)
        raise
    # Only discard this successful run's temporary sample copies, never failed runs.
    sample_root = (stage / ".samples").resolve()
    sample_root.relative_to(stage.resolve())
    if sample_root.exists():
        for file in sample_root.rglob("sample_*.pkl"):
            file.resolve().relative_to(sample_root)
            try:
                file.unlink()
            except OSError as error:
                print(f"Data published; temporary copy retained: {file}: {error}", flush=True)
    print(f"Published {len(groups)} groups; run metadata: {stage}", flush=True)
    return stage


def file_digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--output", default=str(ROOT.parent), help="Dataset root; existing requested groups are archived after validation")
    args = parser.parse_args()
    publish_run(run, read_config(args.config), args.output)
