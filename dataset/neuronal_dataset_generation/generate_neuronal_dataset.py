"""Neuronal 三类数据生成；沿用原项目的 pickle 文件和参数定义。"""
from __future__ import annotations

import argparse
import csv
from collections import deque
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import pickle
import platform
import sys
import time

import numpy as np
from scipy import sparse
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.special import expit
from threadpoolctl import threadpool_limits
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
# 只导入纯函数，不运行或修改 SIS 的生成入口。
from sis_dataset_generation.generate_sis_dataset import (
    build_graph, generate_candidates, graph_hash, perturbations, seed_for, publish_run,
)

REFERENCE = [(3.5, 2., 6.61), (3., 2., 4.23), (2.7, 1.5, 4.37),
             (3.2, 2., 5.03), (3., 1.5, 5.64), (3.6, 1.8, 8.06),
             (3.65, 2.12, 7.16), (3.65, 1.78, 8.53), (3.85, 2.1, 8.72)]


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


@lru_cache(maxsize=256)
def critical_values(mu, d):
    """返回 (下临界值, 上临界值)，参数对应 exp(mu-d*x)。"""
    if not np.isfinite([mu, d]).all() or mu <= 2 or d <= 0:
        raise ValueError("Two neuronal folds require finite mu>2 and d>0")
    # z=y-1>0；在 log(z) 中求根避免大mu下y1过于接近1的消减误差。
    def equation(log_z):
        return log_z + mu - 1 - np.exp(log_z)
    left = brentq(equation, -mu - 2, 0, xtol=1e-13)
    right_end = np.log(mu + 2)
    while equation(right_end) > 0:
        right_end += 1
    right = brentq(equation, 0, right_end, xtol=1e-13)
    def beta(log_z):
        return float((np.exp(log_z) + 2 + np.exp(-log_z)) / d)
    lower, upper = beta(right), beta(left)
    if not np.isfinite([lower, upper]).all() or not lower < upper:
        raise ValueError("Critical values numerically unresolved for these parameters")
    return lower, upper


def classify_beta(beta, lower, upper):
    at_critical = (abs(beta - lower) <= 1e-10 * abs(lower)
                   or abs(beta - upper) <= 1e-10 * abs(upper))
    label = 1 if at_critical else (0 if beta < lower else (2 if beta > upper else 1))
    return int(label), bool(at_critical)


def gbb(A, thresholds):
    degrees = np.asarray(A.sum(axis=1)).ravel()
    total = float(degrees.sum())
    beta = float(degrees @ degrees / total) if total else 0.
    label, boundary = classify_beta(beta, *thresholds)
    return dict(three_class=int(label), binary=int(label == 2), beta_eff=beta, at_critical=boundary)


def neuronal_rhs(t, x, A, mu, d):
    return -x + A @ expit(d * x - mu)


def initial_conditions(n, rng, random_trajectories=14):
    values = np.r_[0., 10. ** rng.uniform(-2, 1, random_trajectories), 5.]
    return np.repeat(values[:, None], n, axis=1)


def sampling_grid(cfg, mode):
    points = cfg["sampling"][mode]
    return np.linspace(0., cfg["simulation"]["T"], points["time_points"])[:points["saved_points"]]


def true_labels(end):
    means = end.mean(axis=1)
    label = 1 if np.ptp(means) > 3 else (2 if means.min() > 3.5 else 0)
    return int(label), int(label == 2)


def simulate(A, params, cfg, mode, seed):
    """保持原采样网格；每条轨迹独立检查收敛，不混淆终态和观测末点。"""
    s = cfg["simulation"]
    times = sampling_grid(cfg, mode)
    starts = initial_conditions(A.shape[0], np.random.default_rng(seed), s["random_trajectories"])
    trajectories, ends, records = [], [], []
    terminal = mode != "train"
    degree_max = float(np.asarray(A.sum(axis=1)).max())
    for x0 in starts:
        upper_bound = max(float(x0.max()), degree_max)
        def integrate(state, start, stop, evaluation, precision=False):
            # 检查前的精化积分收紧误差，避免容差平台掩盖残差收敛条件。
            rtol = min(s["rtol"], s["residual_tol"] * .01 / max(1., upper_bound)) if precision else s["rtol"]
            atol = min(s["atol"], s["residual_tol"] * .01) if precision else s["atol"]
            sol = solve_ivp(neuronal_rhs, (start, stop), state, args=(A, params["mu"], params["d"]),
                            method="RK45", rtol=rtol, atol=atol, t_eval=evaluation)
            if not sol.success or sol.y.shape != (A.shape[0], len(evaluation)) or not np.isfinite(sol.y).all():
                raise RuntimeError(f"RK45 failed: {sol.message}")
            if sol.y.min() < -1e-7 or sol.y.max() > upper_bound + 1e-6:
                raise RuntimeError("Neuronal trajectory outside invariant nonnegative activity bounds")
            return sol.y

        if not terminal:
            trajectories.append(integrate(x0, 0., times[-1], times).T)
            continue
        stop, previous = float(s["T"]), 0.
        state = x0
        trajectory = np.empty((len(times), A.shape[0]), dtype=np.float64)
        first = True
        while True:
            check = stop - 1.
            if check <= previous:
                raise ValueError("max_T leaves insufficient time for the final one-unit convergence check")
            refine = max(previous, stop - 30.)
            requested = times[times <= check] if first else np.array([])
            early = requested[requested <= refine]
            early_times = np.unique(np.r_[early, refine])
            if refine > previous:
                coarse = integrate(state, previous, refine, early_times)
                refined_state = coarse[:, -1]
            else:
                coarse = state[:, None]
                refined_state = state
            late = requested[requested > refine]
            late_times = np.unique(np.r_[late, check])
            before = integrate(refined_state, refine, check, late_times, precision=True)
            if first:
                trajectory[:len(early)] = coarse[:, np.searchsorted(early_times, early)].T
                trajectory[len(early):len(requested)] = before[:, np.searchsorted(late_times, late)].T
            check_state = before[:, -1]
            remaining = times[times > check] if first else np.array([])
            tail_times = np.unique(np.r_[remaining, stop])
            tail = integrate(check_state, check, stop, tail_times, precision=True)
            if first and len(remaining):
                trajectory[len(requested):] = tail[:, np.searchsorted(tail_times, remaining)].T
            state = tail[:, -1]
            change = float(np.max(np.abs(state - check_state)))
            residual = float(np.max(np.abs(neuronal_rhs(stop, state, A, **params))))
            if change <= s["change_tol"] and residual <= s["residual_tol"]:
                break
            if stop >= s["max_T"]:
                raise RuntimeError(f"No convergence: N={A.shape[0]}, params={params}, T={stop}, change={change}, residual={residual}")
            previous, stop, first = stop, min(2 * stop, s["max_T"]), False
        trajectories.append(trajectory)
        ends.append(state)
        records.append(dict(time=stop, change=change, residual=residual))
    return np.asarray(trajectories, dtype=np.float64), (np.asarray(ends) if terminal else None), records, starts[:, 0].tolist()


def choose_params(cfg, mode, seed):
    options = cfg["parameters"][mode]
    index = int(np.random.default_rng(seed).integers(len(options)))
    return dict(zip(("mu", "d"), map(float, options[index]))), index


def make_sample(A, source, cfg, mode, seed, params=None):
    params, param_index = choose_params(cfg, mode, seed_for(seed, "parameters")) if params is None else (params, None)
    trace, end, convergence, initials = simulate(A, params, cfg, mode, seed)
    result = gbb(A, critical_values(**params))
    labels = true_labels(end) if mode == "classification" else (None, None)
    return dict(A=A, trajectory=trace, end=end, three_class=labels[0], binary=labels[1],
                params=params, gbb=result, source={**source, "actual_n": A.shape[0],
                    "simulation_seed": seed, "parameter_index": param_index, "initial_values": initials,
                    "convergence": convergence, "adjacency_sha256": graph_hash(A)})


class Isolation:
    """一个运行内的有序邻接去重；母网络跨运行另以run_id/seed记录。"""
    def __init__(self):
        self.owners = {}

    def duplicate(self, A, owner):
        return graph_hash(A) in self.owners and self.owners[graph_hash(A)] != owner

    def add(self, A, owner):
        self.owners.setdefault(graph_hash(A), owner)


def save_group(path, rows, cfg, mode, suffix="", stats=None):
    path.mkdir(parents=True, exist_ok=True)
    supervised = mode == "classification"
    data = dict(As=[r["A"].toarray().astype(np.float64 if supervised else np.float32) for r in rows],
                numes=[r["trajectory"] for r in rows],
                binary_basers=[r["gbb"]["binary"] for r in rows],
                three_class_basers=[r["gbb"]["three_class"] for r in rows])
    if supervised:
        data.update(binary_rs=[r["binary"] for r in rows], three_class_rs=[r["three_class"] for r in rows])
        if cfg["save_supervised_terminal"]:
            data["x_last"] = [r["end"] for r in rows]
    elif mode == "perturbation":
        data["xlast"] = [r["end"] for r in rows]
    if cfg["save_gbb_times"]:
        durations = []
        for r in rows:
            thresholds = critical_values(**r["params"])
            for _ in range(cfg["gbb_warmup"]):
                gbb(r["A"], thresholds)
            seconds = []
            for _ in range(cfg["gbb_repeats"]):
                start = time.perf_counter_ns()
                gbb(r["A"], thresholds)
                seconds.append((time.perf_counter_ns() - start) / 1e9)
            durations.append(seconds)
        data["gbb_times"] = durations
    for name, values in data.items():
        with (path / f"{name}{suffix}.pkl").open("xb") as stream:
            pickle.dump(values, stream, protocol=pickle.HIGHEST_PROTOCOL)
    dump(path / f"neuronal_params{suffix}.json", [r["params"] for r in rows])
    dump(path / f"generation{suffix}.json", dict(samples=len(rows), mode=mode, stats=stats,
        three_class_counts=[sum(r["three_class"] == k for r in rows) for k in range(3)] if supervised else None,
        binary_counts=[sum(r["binary"] == k for r in rows) for k in range(2)] if supervised else None,
        sources=[r["source"] for r in rows], gbb=[r["gbb"] for r in rows],
        gbb_timing=dict(units="seconds", cpu=platform.processor(), cpu_threads=1,
            warmup=cfg["gbb_warmup"], repeats=cfg["gbb_repeats"],
            scope="CSR degree calculation through both predictions; cached folds excluded") if cfg["save_gbb_times"] else None))


def quotas(count):
    if count % 3:
        raise ValueError("Three-state balanced count must be divisible by 3")
    return np.full(3, count // 3, dtype=int)


def classification_candidates(cfg, namespace):
    """轮流遍历母网络；不改变每个母网络的参数分布或增广序列。"""
    mothers = iter(generate_candidates(cfg, namespace, cfg["initial_size_range"], None))
    active = deque()
    exhausted = False
    while active or not exhausted:
        while len(active) < cfg["classification_mother_window"] and not exhausted:
            try:
                mother, source, family = next(mothers)
                active.append((mother, source, enumerate(family)))
            except StopIteration:
                exhausted = True
        if not active:
            break
        mother, source, family = active.popleft()
        try:
            index, (A, augmentation) = next(family)
        except StopIteration:
            continue
        active.append((mother, source, family))
        yield mother, source, index, A, augmentation


def collect_classification(cfg, namespace, count, isolation):
    targets = quotas(count)
    counts = np.zeros(3, dtype=int)
    rows, attempts, duplicates, small = [], 0, 0, 0
    mothers_seen = set()
    for mother, source, index, A, augmentation in classification_candidates(cfg, namespace):
        mothers_seen.add(mother)
        # 原训练/分类加载器会跳过N<=20；此处明确拒收，避免保存后配额丢失。
        if A.shape[0] <= cfg["min_usable_nodes"]:
            small += 1
            continue
        if isolation.duplicate(A, namespace):
            duplicates += 1
            continue
        seed = seed_for(cfg["seed"], namespace, mother, index, "simulation")
        try:
            row = make_sample(A, {**source, "augmentation": augmentation}, cfg, "classification", seed)
        except Exception as error:
            raise RuntimeError(f"{namespace}, mother={mother}, candidate={index}: {error}") from error
        attempts += 1
        label = row["three_class"]
        if attempts % 100 == 0:
            print(f"{namespace}: checked {attempts} candidates from {len(mothers_seen)} mothers", flush=True)
        if counts[label] < targets[label]:
            rows.append(row); counts[label] += 1
            isolation.add(A, namespace)
            print(f"{namespace}: counts={counts.tolist()}/{targets.tolist()}, candidates={attempts}", flush=True)
        if len(rows) == count:
            np.random.default_rng(seed_for(cfg["seed"], namespace, "shuffle")).shuffle(rows)
            return rows, dict(candidates=attempts, mothers=len(mothers_seen), cross_group_duplicates=duplicates,
                              small_graphs=small, quotas=targets.tolist())
        if attempts >= cfg["max_candidates"]:
            raise RuntimeError(f"{namespace}: candidate limit reached; counts={counts.tolist()}, targets={targets.tolist()}")
    raise RuntimeError(f"{namespace}: mother limit reached; counts={counts.tolist()}, targets={targets.tolist()}")


def write_threshold_table(path, cfg):
    rows = []
    for i, (mu, d, reference) in enumerate(REFERENCE, 1):
        lower, upper = critical_values(mu, d)
        rows.append(dict(source="ResInf Supplementary Table 4", index=i, mu=mu, d=d,
                         lower=lower, upper=upper, reference_upper=reference, difference=upper-reference))
    known = {(r["mu"], r["d"]) for r in rows}
    for mu, d in [(3.3, 1.8), (3.2, 2.1)] + [tuple(p) for opts in cfg["parameters"].values() for p in opts]:
        if (mu, d) not in known:
            lower, upper = critical_values(mu, d)
            rows.append(dict(source="local/config", index="", mu=mu, d=d, lower=lower, upper=upper,
                             reference_upper="", difference=""))
            known.add((mu, d))
    with path.open("x", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def run(cfg, output):
    validate_config(cfg)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    dump(output / "config.json", cfg)
    dump(output / "environment.json", dict(python=sys.version, numpy=np.__version__,
         platform=platform.platform(), processor=platform.processor(), cpu_threads=1,
         shared_topology_sha256=hashlib.sha256((ROOT.parent / "sis_dataset_generation/generate_sis_dataset.py").read_bytes()).hexdigest()))
    write_threshold_table(output / "gbb_critical_values.csv", cfg)
    isolation, completed = Isolation(), []
    try:
        with threadpool_limits(limits=1):
            if "train" in cfg["modes"]:
                rows = []
                for mother, source, family in generate_candidates(cfg, "neuronal:train", cfg["initial_size_range"], None):
                    if mother >= cfg["train_mothers"]:
                        break
                    for index, (A, augmentation) in enumerate(family):
                        rows.append(make_sample(A, {**source, "augmentation": augmentation}, cfg, "train",
                                    seed_for(cfg["seed"], "neuronal:train", mother, index)))
                        isolation.add(A, "neuronal:train")
                save_group(output / "Neuronal", rows, cfg, "train")
                completed.append("Neuronal")
            if "perturbation" in cfg["modes"]:
                for kind in ("ER", "SF"):
                    owner = f"neuronal:perturbation:{kind}"
                    topology_seed = seed_for(cfg["seed"], owner)
                    A, topology_params = build_graph(cfg["perturbation_n"], kind, np.random.default_rng(topology_seed), fixed=True)
                    params, _ = choose_params(cfg, "perturbation", seed_for(topology_seed, "parameters"))
                    for strategy in ("random", "degree"):
                        rows = []
                        rng = np.random.default_rng(seed_for(cfg["seed"], owner, strategy))
                        for index, (B, ids, removed) in enumerate(perturbations(A, strategy, cfg["min_nodes"], rng)):
                            if isolation.duplicate(B, owner):
                                raise RuntimeError(f"{owner}: adjacency duplicates another dataset; select a different seed")
                            source = dict(mother_id=owner, topology_seed=topology_seed, topology=kind,
                                          topology_parameters=topology_params, original_node_ids=ids, removed=removed)
                            rows.append(make_sample(B, source, cfg, "perturbation",
                                        seed_for(cfg["seed"], owner, strategy, index), params=params))
                            isolation.add(B, owner)
                        suffix = f"_{kind}_{strategy}"
                        save_group(output / "Neuronal_test", rows, cfg, "perturbation", suffix)
                        completed.append(f"Neuronal_test/*{suffix}")
            if "classification" in cfg["modes"]:
                for split, count in cfg["classification_splits"].items():
                    if not count:
                        continue
                    namespace = f"neuronal:classification:{split}"
                    rows, stats = collect_classification(cfg, namespace, count, isolation)
                    relative = f"Neuronal_supervised/{split}"
                    save_group(output / relative, rows, cfg, "classification", stats=stats)
                    completed.append(relative)
            dump(output / "status.json", dict(status="complete", completed=completed))
    except Exception as error:
        dump(output / "status.json", dict(status="failed", error=str(error), completed=completed))
        raise


def validate_config(cfg):
    modes = cfg["modes"]
    if not modes or len(set(modes)) != len(modes) or set(modes) - {"train", "perturbation", "classification"}:
        raise ValueError("Select unique modes: train, perturbation, classification")
    if "balance_schemes" in cfg:
        raise ValueError("Use one shared three-state-balanced dataset; remove balance_schemes")
    if set(cfg["classification_splits"]) - {"train_dataset", "val_dataset", "test_dataset"}:
        raise ValueError("Use original split names")
    for count in cfg["classification_splits"].values():
        if type(count) is not int or count < 0:
            raise ValueError("Split counts must be nonnegative integers")
        quotas(count)
    for mode in ("train", "perturbation", "classification"):
        if not cfg["parameters"][mode]:
            raise ValueError("At least one parameter pair required per mode")
        for mu, d in cfg["parameters"][mode]:
            critical_values(float(mu), float(d))
        points = cfg["sampling"][mode]
        if not 2 <= points["saved_points"] <= points["time_points"]:
            raise ValueError("Invalid saved sample count")
    s = cfg["simulation"]
    for key in ("rtol", "atol", "change_tol", "residual_tol"):
        if not np.isfinite(s[key]) or s[key] <= 0:
            raise ValueError(f"Invalid {key}")
    if (not np.isfinite([s["T"], s["max_T"]]).all() or s["T"] <= 1 or s["max_T"] < s["T"]
            or type(s["random_trajectories"]) is not int or s["random_trajectories"] < 0):
        raise ValueError("Invalid simulation settings")
    lo, hi = cfg["initial_size_range"]
    if (any(type(v) is not int for v in (lo, hi, cfg["min_nodes"], cfg["perturbation_n"]))
            or not 2 <= lo <= hi or not 2 <= cfg["min_nodes"] <= cfg["perturbation_n"]):
        raise ValueError("Invalid network sizes")
    if type(cfg["min_usable_nodes"]) is not int or cfg["min_usable_nodes"] < 20:
        raise ValueError("Original supervised loader requires N>20")
    if type(cfg["seed"]) is not int:
        raise ValueError("seed must be an integer")
    for name in ("train_mothers", "max_mothers", "max_candidates", "gbb_repeats", "classification_mother_window"):
        if type(cfg[name]) is not int or cfg[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    if cfg["train_mothers"] > cfg["max_mothers"] or type(cfg["gbb_warmup"]) is not int or cfg["gbb_warmup"] < 0:
        raise ValueError("Invalid generation/timing limits")


def read_config(path):
    with Path(path).open(encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    validate_config(cfg)
    return cfg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--output", default=str(ROOT.parent), help="Dataset root; stage, validate and archive before publication")
    args = parser.parse_args()
    publish_run(run, read_config(args.config), args.output)
