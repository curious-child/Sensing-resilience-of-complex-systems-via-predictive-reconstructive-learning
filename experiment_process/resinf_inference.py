"""ResInf recovery for paper figures; invoked by the supervised cache entrypoint."""
from pathlib import Path
import json
import os
import tempfile

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch_geometric.utils import to_dense_adj

EPOCHS = 100
PARAMETERS = dict(input_plane=5, seq_len=10, trans_layers=1, gcn_layers=3,
    hidden_layers=1, gcn_emb_size=8, trans_emb_size=32, pool_type="virtual", n_heads=4)


def build_model(classes, checkpoint=None):
    from models.pred_model.resinf import ResInf
    parameters = dict(PARAMETERS)
    if checkpoint is not None:
        parameters.update(checkpoint.get("model_parameters", {}))
    parameters["num_class"] = classes
    model = ResInf(**parameters)
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model


def inputs(graph, device):
    """Original virtual-node preprocessing, with explicit zero-degree handling."""
    adjacency = to_dense_adj(graph.edge_index.cpu(), max_num_nodes=graph.num_nodes)[0].numpy()
    adjacency = np.concatenate((adjacency, np.ones((1, len(adjacency)))), axis=0)
    adjacency = np.concatenate((adjacency, np.zeros((len(adjacency), 1))), axis=1)
    out_degree = adjacency.sum(1).astype(np.float32)
    in_degree = adjacency.sum(0).astype(np.float32)
    left, right = np.zeros_like(out_degree), np.zeros_like(in_degree)
    np.power(out_degree, -.5, out=left, where=out_degree != 0)
    np.power(in_degree, -.5, out=right, where=in_degree != 0)
    adjacency = left[:, None] * adjacency * right[None, :]
    x = graph.x.detach().cpu().squeeze(-1).permute(1, 0, 2)
    x = torch.cat((x, x.mean(1, keepdim=True)), dim=1)
    return x.to(device), torch.as_tensor(adjacency, dtype=torch.float32, device=device)


def output(model, graph, classes, device):
    x, adjacency = inputs(graph, device)
    if classes == 2:
        scores = model(x, adjacency).reshape(1)
        prediction = int(scores.detach().item() > .5)
    else:
        scores, probability = model.mclassifier(x, adjacency)
        scores = scores.reshape(1, classes)
        prediction = int(probability.detach().argmax().item())
    if not torch.isfinite(scores).all():
        raise ValueError("Non-finite ResInf output; no substitute predictions are generated")
    return scores, prediction


@torch.no_grad()
def predict(model, data, classes, device):
    model.eval()
    return [output(model, graph, classes, device)[1] for graph in data]


def train_one(api, spec, ratio, trial, device, train, validation, target, data_sources=None):
    seed = api.task_seed(spec, "resinf") + trial * 1000
    api._seed_everything(seed)
    chosen = api.reference_indices(spec, train, ratio, trial, method="resinf")
    model = build_model(spec.classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    criterion = torch.nn.BCELoss() if spec.classes == 2 else torch.nn.CrossEntropyLoss()
    best, best_epoch, state, history = -1., None, None, []
    for epoch in range(EPOCHS):
        model.train()
        for position in torch.randperm(len(chosen), generator=generator).tolist():
            graph = train[chosen[position]]
            scores, _ = output(model, graph, spec.classes, device)
            truth = graph.y.reshape(1).to(device)
            loss = criterion(scores, truth.float() if spec.classes == 2 else truth.long())
            if not torch.isfinite(loss):
                raise ValueError("Non-finite ResInf training loss")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        predictions = predict(model, validation, spec.classes, device)
        score = float(f1_score(api.labels(validation), predictions, average="weighted"))
        history.append(dict(epoch=epoch, val_f1=score))
        if score > best:
            best, best_epoch = score, epoch
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"{spec.key}/resinf/{ratio}/{trial}: {epoch+1}/{EPOCHS}, val F1={score:.6f}", flush=True)
    payload = dict(model_state_dict=state, model_parameters={**PARAMETERS, "num_class": spec.classes},
        ratio=ratio, sample_idx=trial, seed=seed, train_size=len(chosen),
        train_indices=[int(train[i].source_index) for i in chosen], val_indices=api.sample_ids(validation),
        observation_indices={name: [g.observation_indices.reshape(-1).tolist() for g in data]
            for name, data in (("train", train), ("validation", validation))},
        best_epoch=best_epoch, best_val_f1=best, history=history, data_sources=data_sources,
        seeds=dict(initialization=seed, reference=seed, batches=seed, observations=spec.seed),
        config=dict(epochs=EPOCHS, batch_size=1, lr=.001, weight_decay=1e-4,
                    optimizer="AdamW", checkpoint_selection="best_validation_clone", policy=api.POLICY))
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.stat().st_size != 0:
            raise FileExistsError(target)
        fd, temporary = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as stream:
                torch.save(payload, stream)
            if target.stat().st_size != 0:
                raise FileExistsError(target)
            os.replace(temporary, target)  # Explicitly authorized empty-file recovery.
        finally:
            Path(temporary).unlink(missing_ok=True)
    else:
        api.save_new_checkpoint(payload, target)
    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def ensure_resinf_cache(spec, ratios, trials, device, force=False, *, api):
    if not force and trials == tuple(range(api.NUM_TRIALS)):
        try:
            return api.historical_resinf(spec)
        except (FileNotFoundError, ValueError):
            pass
    _, path = api.cache_paths(spec, "resinf", ratios, trials)
    code = {str(p.relative_to(api.PROJECT_ROOT)): api._sha256(p) for p in
            (Path(__file__), api.PROJECT_ROOT / "models/pred_model/resinf.py")}
    previous = None
    if path.is_file() and not force:
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
            identity = previous["fingerprint"]
            if (identity.get("dataset_versions", {}) == api.dataset_versions(spec.dataset_dir)
                    and identity.get("code_sha256") == code and identity.get("ratios") == list(ratios)
                    and identity.get("trials") == list(trials) and api.valid_payload(previous, identity)):
                return api.write_summary(spec, "resinf", previous)
        except (ValueError, KeyError, TypeError):
            previous = None
    missing = []
    for ratio in ratios:
        for trial in trials:
            target = api.available_model_path(spec, "resinf", ratio, trial)
            if target is None or target.stat().st_size == 0:
                missing.append((ratio, trial, target or api.trained_model_path(spec, "resinf", ratio, trial)))
            else:
                checkpoint = torch.load(target, map_location="cpu", weights_only=False)
                if checkpoint.get("sample_idx", trial) != trial or not np.isclose(checkpoint.get("ratio", ratio), ratio):
                    raise ValueError(f"ResInf checkpoint identity mismatch: {target}")
                build_model(spec.classes, checkpoint)  # Preflight strictly before any training.
    if missing:
        print(f"{spec.key}: automatically training {len(missing)} missing/empty ResInf models", flush=True)
        train = api.load_split(spec, "train_dataset")
        validation = api.evaluation_split(spec, "val_dataset", "resinf")
        data_sources = {str(spec.dataset_dir / split / name): api._sha256(spec.dataset_dir / split / name)
            for split in ("train_dataset", "val_dataset") for name in ("As.pkl", "numes.pkl", spec.label_file)}
        for ratio, trial, target in missing:
            train_one(api, spec, ratio, trial, device, train, validation, target, data_sources)
        del train, validation
    test = api.evaluation_split(spec, "test_dataset", "resinf")
    truth = api.labels(test)
    sources = {str(spec.dataset_dir / "test_dataset" / name):
               api._sha256(spec.dataset_dir / "test_dataset" / name)
               for name in ("As.pkl", "numes.pkl", spec.label_file)}
    for ratio in ratios:
        for trial in trials:
            target = api.available_model_path(spec, "resinf", ratio, trial)
            sources[str(target)] = api._sha256(target)
    identity = dict(policy=api.POLICY, schema_version=api.SCHEMA_VERSION, sources=sources,
        dataset_versions=api.dataset_versions(spec.dataset_dir),
        code_sha256=code, ratios=list(ratios), trials=list(trials))
    payload = dict(fingerprint=identity, protocol=api.POLICY, task=spec.key, classifier="resinf",
        complete=False, test_indices=api.sample_ids(test), test_labels=truth,
        observation_indices=[g.observation_indices.reshape(-1).tolist() for g in test], results={})
    if previous and previous.get("fingerprint") == identity:
        payload["results"] = previous.get("results", {})
    api.archive(path)
    for ratio in ratios:
        rows = payload["results"].get(str(ratio), {}).get("samples", [])
        rows = [r for r in rows if r.get("sample_idx") in trials
            and len(r.get("predictions", [])) == len(truth)
            and np.isclose(r.get("test_f1", np.nan), f1_score(truth, r["predictions"], average="weighted"))]
        if len({r["sample_idx"] for r in rows}) != len(rows):
            raise ValueError("Duplicate ResInf repeats in interrupted cache")
        for trial in trials:
            if any(r.get("sample_idx") == trial for r in rows):
                continue
            target = api.available_model_path(spec, "resinf", ratio, trial)
            checkpoint = torch.load(target, map_location="cpu", weights_only=False)
            if checkpoint.get("sample_idx", trial) != trial or not np.isclose(checkpoint.get("ratio", ratio), ratio):
                raise ValueError(f"ResInf checkpoint identity mismatch: {target}")
            model = build_model(spec.classes, checkpoint).to(device)
            predictions = predict(model, test, spec.classes, device)
            rows.append(dict(sample_idx=trial, predictions=predictions,
                test_f1=float(f1_score(truth, predictions, average="weighted")),
                train_size=checkpoint.get("train_size"), model=api.checkpoint_record(target)))
            payload["results"][str(ratio)] = dict(samples=rows)
            api._atomic_json(payload, path)
            del model
        sizes = [r["train_size"] for r in rows]
        payload["results"][str(ratio)] = api._f1_summary(rows, sizes[0] if len(set(sizes)) == 1 else None, trials)
    payload["complete"] = True
    api._atomic_json(payload, path)
    return api.write_summary(spec, "resinf", payload)
