"""Plot measured SIS scaling results only; never interpolate missing results."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

COLORS = dict(knn="#2166AC", mlp="#67A9CF", resinf="#D6604D", gbb="#555555")
LABELS = dict(knn="PRISM-KNN (GPU + CPU)", mlp="PRISM-MLP (GPU)",
              resinf="ResInf (GPU)", gbb="GBB (CPU)")


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))


def selected_groups(config, max_n=None):
    groups = [g for g in config["datasets"] if max_n is None or g["n"] <= max_n]
    if not groups:
        raise ValueError("No cached/configured network sizes match --max-n")
    return groups


def complete_jobs(status, groups):
    """Check cached jobs without touching datasets, checkpoints or CUDA."""
    config = status["config"]
    expected = {(g["id"], method, trial) for g in groups for method in config["methods"]
                for trial in ([None] if method == "gbb" else config["trial_ids"])}
    def key(row):
        return row["group"], row["method"], row["trial"]
    valid = {key(r) for r in status.get("metrics", [])
             if r.get("status") == "ok" and r.get("expected", 0) > 0
             and r.get("succeeded") == r.get("expected")}
    # Stages are not plotted. Historical failures do not invalidate later successes.
    for section in ("timings", "memory"):
        valid &= {key(r) for r in status.get(section, []) if r.get("status") == "ok"}
    return expected, valid & expected


def ensure_results(folder, config_path, max_n=None, force=False):
    folder = Path(folder).resolve()
    if (folder / "records.json").is_file() and not force:
        print(f"Reading cache: {folder / 'records.json'}; no dataset/model loading.", flush=True)
        return folder
    from experiment_process.benchmark_sis_scaling import read_config, run
    cfg = read_config(config_path)
    cfg["datasets"] = selected_groups(cfg, max_n)
    print(f"Generating requested results in: {folder}", flush=True)
    run(cfg, folder, force=force)
    return folder


def merged_summary(status, groups):
    """Pool ER/SF samples within each trial before calculating statistics."""
    from sklearn.metrics import f1_score
    config = status["config"]
    frames = {k: pd.DataFrame(status.get(k, [])) for k in ("predictions", "timings", "memory")}
    results = []
    for n in sorted({g["n"] for g in groups}):
        selected = [g for g in groups if g["n"] == n]
        if {g["topology"] for g in selected} != {"ER", "SF"}:
            raise ValueError(f"N={n}: merged statistics require both ER and SF groups")
        group_ids = {g["id"] for g in selected}
        for method in config["methods"]:
            trials = [None] if method == "gbb" else config["trial_ids"]
            values = {}
            for trial in trials:
                records = {}
                for key, frame in frames.items():
                    if frame.empty:
                        records[key] = frame
                        continue
                    trial_mask = frame.trial.isna() if trial is None else frame.trial == trial
                    records[key] = frame[frame.group.isin(group_ids) & (frame.method == method) & trial_mask]
                pred = records["predictions"]
                if not pred.empty and set(pred.group) == group_ids and (pred.status == "ok").all():
                    if pred.sample_id.duplicated().any():
                        raise ValueError(f"N={n}, {method}, trial={trial}: duplicate sample predictions")
                    values.setdefault("f1_weighted", []).append(f1_score(pred.truth, pred.prediction, average="weighted", zero_division=0))
                timing = records["timings"]
                if not timing.empty and "total_ms" in timing:
                    timing = timing[(timing.status == "ok") & timing.total_ms.notna()]
                    if set(timing.group) == group_ids:
                        values.setdefault("time_ms", []).append(timing.groupby("sample_id").total_ms.median().median())
                memory = records["memory"]
                for field in ("cpu_peak_rss_sampled_bytes", "gpu_peak_allocated_bytes"):
                    if not memory.empty and field in memory:
                        measured = memory[(memory.status == "ok") & memory[field].notna()]
                        if set(measured.group) == group_ids:
                            values.setdefault(field, []).append(measured[field].max())
            row = dict(n=n, method=method)
            for field, samples in values.items():
                # Never substitute a smaller set of model trials for the requested ensemble.
                if len(samples) == len(trials):
                    row[field + "_mean"] = float(np.mean(samples))
                    row[field + "_std"] = float(np.std(samples, ddof=0))
            if len(row) > 2:
                results.append(row)
    return pd.DataFrame(results)


def plot(folder, output, allow_partial=False, max_n=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder, output = Path(folder), Path(output)
    status = json.loads((folder / "records.json").read_text(encoding="utf-8"))
    groups = selected_groups(status["config"], max_n)
    for topology in ("ER", "SF"):
        sizes = sorted({g["n"] for g in groups if g["topology"] == topology})
        print(f"{topology}: selected cached sizes (inclusive upper bound): {sizes}", flush=True)
        if max_n is not None and max_n not in sizes:
            print(f"{topology}: N={max_n} is absent from this cache's dataset list.", flush=True)
    expected, valid = complete_jobs(status, groups)
    complete = expected == valid
    if not complete and not allow_partial:
        missing = sorted(f"{g} / {m} / trial={t}" for g, m, t in expected - valid)
        reasons = [f"{r['group']} / {r['method']} / trial={r['trial']} / "
                   f"{r.get('phase', '?')}: {r.get('error', r.get('status', '?'))}"
                   for r in status.get("failures", [])
                   if (r['group'], r['method'], r['trial']) in expected - valid]
        raise ValueError(f"Cache: {folder / 'records.json'}\n"
                         "Missing/incomplete F1, timing or memory records for:\n" + "\n".join(missing) +
                         ("\nFirst recorded failures:\n" + "\n".join(reasons[:5]) if reasons else "") +
                         "\nNo evaluation was started. Use --allow-partial to plot available results.")
    output.mkdir(parents=True, exist_ok=True)
    summary = merged_summary(status, groups)
    if summary.empty:
        raise ValueError("No valid measurements to plot")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7,
                         "axes.labelsize": 7, "axes.titlesize": 8, "legend.fontsize": 6,
                         "svg.fonttype": "none", "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    config = status["config"]
    ratio = config.get("training_ratio")
    print(f"Cached labelled training ratio: {ratio}; trial IDs: {config['trial_ids']}", flush=True)
    for topology in ("ER_SF",):
        rows = summary
        if rows.empty:
            continue
        sizes = sorted({g["n"] for g in groups})
        fig = plt.figure(figsize=(7.20, 3.25), layout="constrained")
        grid = fig.add_gridspec(2, 3, width_ratios=[1, 1.12, 1])
        a, cpu, gpu, c = fig.add_subplot(grid[:, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, 1]), fig.add_subplot(grid[:, 2])
        handles = []
        for method in COLORS:
            data = rows[rows.method == method]
            if data.empty:
                continue
            data = data.set_index("n").reindex(sizes)
            style = dict(color=COLORS[method], marker="o", markersize=3, linewidth=1)
            label = LABELS[method]
            if config["device"] == "cpu":
                label = label.replace("GPU + CPU", "CPU test").replace("GPU", "CPU test")
            handle, = a.plot([], [], label=label, **style)
            handles.append(handle)
            for ax, mean, spread in ((a, "time_ms_mean", "time_ms_std"), (c, "f1_weighted_mean", "f1_weighted_std")):
                if mean in data and data[mean].notna().any():
                    ax.errorbar(sizes, data[mean], yerr=data.get(spread, pd.Series(0., index=data.index)).fillna(0), capsize=2, **style)
            for ax, field in ((cpu, "cpu_peak_rss_sampled_bytes"), (gpu, "gpu_peak_allocated_bytes")):
                if field + "_mean" in data and data[field + "_mean"].notna().any():
                    ax.errorbar(sizes, data[field + "_mean"] / 2**20,
                                yerr=data[field + "_std"].fillna(0) / 2**20, capsize=2, **style)
        a.set(xscale="log", yscale="log", xlabel="Number of nodes, N", ylabel="Full inference time (ms)")
        cpu.set(xscale="log", ylabel="CPU RSS peak (MiB)", ylim=(0, None))
        gpu.set(xscale="log", xlabel="Number of nodes, N", ylabel="GPU allocated peak (MiB)", ylim=(0, None))
        c.set(xscale="log", xlabel="Number of nodes, N", ylabel="F1-score", ylim=(-0.02, 1.04))
        for ax, letter in ((a, "a"), (cpu, "b"), (c, "c")):
            ax.text(-0.14, 1.06, letter, transform=ax.transAxes, fontweight="bold", fontsize=10)
        for ax in (a, cpu, gpu, c):
            ax.grid(alpha=0.15, linewidth=0.5)
            if len(sizes) <= 8:
                ax.set_xticks(sizes, labels=[str(n) for n in sizes])
                ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        for ax in (cpu, gpu):
            ax.ticklabel_format(axis="y", style="plain", useOffset=False)
            ax.set_ylim(0, ax.get_ylim()[1] * 1.08)
        if "gpu_peak_allocated_bytes_mean" not in rows or rows.gpu_peak_allocated_bytes_mean.isna().all():
            gpu.text(.5, .5, "No GPU measurement", ha="center", transform=gpu.transAxes)
        title = "SIS dynamics"
        if ratio is not None:
            title += f" | {ratio:.1%} labeled training data"
        if status.get("test_only"):
            title += " | TEST ONLY — not manuscript results"
        if not complete:
            title += " | PARTIAL RUN"
        fig.suptitle(title, fontsize=9)
        fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False)
        for extension in ("svg", "pdf", "png"):
            fig.savefig(output / f"SIS_scaling_{topology}.{extension}", dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT / "experiment_process/sis_scaling.example.yaml"))
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--results", default=str(PROJECT / "experiment_results/SIS/scaling"))
    parser.add_argument("--output", default=str(PROJECT / "experiment_results/figures/sis_scaling"))
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--max-n", type=int, help="Use cached network sizes up to this value")
    args = parser.parse_args()
    if args.max_n is not None and args.max_n < 1:
        parser.error("--max-n must be positive")
    results = ensure_results(args.results, args.config, args.max_n, args.force_cache)
    plot(results, args.output, args.allow_partial, args.max_n)
