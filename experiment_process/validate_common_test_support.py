"""Validate and index the versioned supervised-inference caches."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from experiment_process import generate_resilience_inference_cache as api


def _support_digest(indices, labels):
    encoded = json.dumps(
        {"indices": indices, "labels": labels},
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_task(task, device):
    spec = api.TASKS[task]
    paths = api.ensure_caches(task, device=device)
    method_records = {}
    supports = []
    for method, csv_path in paths.items():
        expected_csv, records_path = api.cache_paths(
            spec, method, api.RATIOS, tuple(range(api.NUM_TRIALS))
        )
        if Path(csv_path) != expected_csv or not records_path.is_file():
            raise ValueError(f"{task}/{method}: versioned cache is missing")
        payload = json.loads(records_path.read_text(encoding="utf-8"))
        if payload.get("protocol", payload.get("fingerprint", {}).get("policy")) != api.POLICY:
            raise ValueError(f"{task}/{method}: protocol mismatch")
        indices, truth = payload.get("test_indices"), payload.get("test_labels")
        if not isinstance(indices, list) or not isinstance(truth, list) or len(indices) != len(truth):
            raise ValueError(f"{task}/{method}: invalid test support")
        supports.append((indices, truth))
        frame = pd.read_csv(csv_path)
        if not np.isfinite(frame.select_dtypes(include=[np.number]).to_numpy()).all():
            raise ValueError(f"{task}/{method}: non-finite summary")
        if method == "knn":
            predictions = payload.get("gbb_predictions", [])
            if len(predictions) != len(truth):
                raise ValueError(f"{task}: GBB predictions are not aligned")
            score = float(f1_score(truth, predictions, average="weighted"))
            if not np.isclose(score, payload.get("gbb_f1"), atol=1e-12, rtol=0):
                raise ValueError(f"{task}: GBB F1 does not match cached predictions")
            if not np.allclose(frame["baser_mean_f1"], score, atol=1e-12, rtol=0):
                raise ValueError(f"{task}: plotted GBB F1 does not match the shared support")
        method_records[method] = {
            "summary": os.path.relpath(csv_path, api.PROJECT_ROOT),
            "summary_sha256": api._sha256(csv_path),
            "records": os.path.relpath(records_path, api.PROJECT_ROOT),
            "records_sha256": api._sha256(records_path),
        }
    if any(item != supports[0] for item in supports[1:]):
        raise ValueError(f"{task}: methods do not share identical ordered test IDs and labels")
    indices, truth = supports[0]
    return {
        "evaluation_scope": "balanced_valid" if task == "neuronal_binary" else "all_valid",
        "node_filter": "N > 20",
        "test_sample_count": len(indices),
        "class_counts": {str(k): v for k, v in sorted(Counter(truth).items())},
        "test_support_sha256": _support_digest(indices, truth),
        "methods": method_records,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output",
        type=Path,
        default=api.RESULTS_ROOT / "common_test_support_v1_manifest.json",
    )
    args = parser.parse_args()
    payload = {
        "schema_version": api.SCHEMA_VERSION,
        "protocol": api.POLICY,
        "complete": True,
        "tasks": {task: validate_task(task, args.device) for task in api.TASKS},
    }
    api._atomic_json(payload, args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
