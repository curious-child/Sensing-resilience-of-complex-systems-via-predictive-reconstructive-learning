"""兼容性检查及临时微型测试；不是论文数据生成入口。"""
import copy
import pickle
import tempfile
from pathlib import Path
import numpy as np
from scipy import sparse
from generate_sis_dataset import ROOT, read_config, run, build_graph, topology_family, sis_rhs, gbb


def large_smoke():
    """Real 30,000-node CPU simulation/I/O test; temporary, not paper data."""
    import gc
    import json
    import time
    import threading
    import psutil
    import sys
    import generate_sis_dataset as gen
    sys.path.insert(0, str(ROOT.parents[1]))
    from experiment_process.benchmark_sis_scaling import load_group
    cfg = read_config(ROOT / "config.yaml")
    cfg["save_gbb_times"] = False
    process = psutil.Process()
    peak = [process.memory_info().rss]
    stopped = threading.Event()
    def monitor():
        while not stopped.wait(.02):
            peak[0] = max(peak[0], process.memory_info().rss)
    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    try:
        with tempfile.TemporaryDirectory(prefix="sis_30000_TEST_ONLY_") as temp:
            for kind in ("ER", "SF"):
                started = time.monotonic()
                a, params = build_graph(30000, kind, np.random.default_rng(123))
                row = gen.sample(a, dict(mother_id=f"TEST_ONLY:{kind}", topology_parameters=params),
                                 6, cfg, 42, True)
                assert row["trajectory"].shape == (16, 20, 30000)
                assert row["end"].shape == (16, 30000)
                for state in row["end"]:
                    assert np.abs(sis_rhs(0, state, a, 6)).max() <= cfg["simulation"]["residual_tol"]
                folder = Path(temp) / kind
                rows = gen.SampleRows(Path(temp) / (kind + "_spool"))
                rows.append(row)
                # A single validation sample does not claim balanced class quotas.
                gen.save_group(folder, rows, cfg, supervised=True)
                metadata = json.loads((folder / "generation.json").read_text())
                indexed = gen.IndexedPickleList(folder / "As.pkl", metadata["pickle_index"]["As.pkl"])
                assert sparse.isspmatrix_csr(indexed[0]) and (indexed[0] != a).nnz == 0
                with (folder / "As.pkl").open("rb") as stream:
                    ordinary = pickle.load(stream)
                assert type(ordinary) is list and sparse.isspmatrix_csr(ordinary[0])
                loaded = load_group(dict(path=folder, n=30000, id=kind, params_file="sis_params.json"),
                                    dict(seed=42, observations=5, windows=10))
                assert len(loaded) == 1 and loaded[0]["x"].shape == (30000, 5, 10, 1)
                print(json.dumps(dict(test="TEST_ONLY", topology=kind, n=30000, trajectories=16,
                    label=row["label"], end_time_max=max(row["source"]["end_times"]),
                    elapsed_seconds=round(time.monotonic()-started, 2),
                    adjacency_mib=round((folder / "As.pkl").stat().st_size/2**20, 3),
                    peak_process_rss_mib=round(peak[0]/2**20, 1))), flush=True)
                del row, a, indexed, ordinary, loaded
                gc.collect()
    finally:
        stopped.set()
        watcher.join()


def safety_micro():
    import json
    import networkx as nx
    import generate_sis_dataset as gen
    from unittest.mock import patch
    cfg = read_config(ROOT / "config.yaml")
    cfg["max_mothers"] = 2  # Unused train_mothers must not invalidate classification.
    gen.validate_config(cfg)
    for changes in (dict(classification_delta=[8,2]), dict(scaling_sizes=[30,30]),
                    dict(scaling_sizes=[0]), dict(max_edge_augmentations=0)):
        try:
            gen.validate_config({**cfg, **changes})
        except ValueError:
            pass
        else:
            raise AssertionError(changes)
    star = sparse.csr_matrix(nx.to_scipy_sparse_array(nx.star_graph(999), dtype=float))
    assert sum(1 for _ in topology_family(star, np.random.default_rng(0))) > 0
    assert list(topology_family(star, np.random.default_rng(0), target_n=1001)) == []
    diagnostics = {}
    list(topology_family(star, np.random.default_rng(0), target_n=998,
                         max_edge_attempts=2, diagnostics=diagnostics))
    assert diagnostics["edge_search_limits"] == 1
    # Same simple SF graph as the original networkx stub-merging implementation.
    for seed in (0, 3, 123):
        rng = np.random.default_rng(seed)
        gamma = float(rng.uniform(3,4) if rng.random() < .3 else rng.uniform(2.5,3))
        k0 = float(rng.uniform(1.5,3))
        degrees = np.round(k0 * np.maximum(rng.random(100),np.finfo(float).tiny)**(1/(1-gamma))).astype(int)
        stubs = np.repeat(np.arange(100),degrees)
        rng.shuffle(stubs)
        stubs = stubs[:len(stubs)//2*2]
        graph = nx.Graph()
        graph.add_nodes_from(range(100))
        graph.add_edges_from(zip(stubs[::2],stubs[1::2]))
        graph.remove_edges_from(nx.selfloop_edges(graph))
        expected = sparse.csr_matrix(nx.to_scipy_sparse_array(graph,nodelist=range(100),dtype=float))
        actual, _ = build_graph(100,"SF",np.random.default_rng(seed))
        assert (actual != expected).nnz == 0
    a = sparse.csr_matrix(nx.to_scipy_sparse_array(nx.cycle_graph(30),dtype=float))
    simulation = {**cfg["simulation"], "trajectories":2}
    old = gen.simulate(a,6,{**simulation,"method":"LSODA"},np.random.default_rng(42),True)
    explicit = gen.simulate(a,6,{**simulation,"method":"RK45"},np.random.default_rng(42),True)
    np.testing.assert_allclose(old[0],explicit[0],rtol=2e-5,atol=2e-8)
    assert int(old[1].mean() >= .001) == int(explicit[1].mean() >= .001)
    try:
        gen.simulate(a,2,{**simulation,"max_T":20.5},np.random.default_rng(42),True)
    except RuntimeError as error:
        assert "did not converge" in str(error)  # No misleading time-window error.
    try:
        gen.simulate(a,6,{**simulation,"trajectory_timeout_seconds":1e-12},np.random.default_rng(42),True)
    except TimeoutError:
        pass
    else:
        raise AssertionError("Timeout was not enforced")
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "items.pkl"
        values = [np.arange(12).reshape(3,4), sparse.eye(4,format="csr"), 1,
                  {"repeat":["abc"]*4}, np.ones((2,3))]
        writer = gen.PickleListWriter(path)
        for value in values:
            writer.append(value)
        writer.close()
        with path.open("rb") as stream:
            whole = pickle.load(stream)
        indexed = gen.IndexedPickleList(path, writer.index)
        assert type(whole) is list and len(whole) == len(indexed) == len(values)
        for i, value in enumerate(values):
            if sparse.issparse(value):
                assert (indexed[i] != value).nnz == (whole[i] != value).nnz == 0
            elif isinstance(value,np.ndarray):
                np.testing.assert_array_equal(indexed[i],value)
                np.testing.assert_array_equal(whole[i],value)
            else:
                assert indexed[i] == whole[i] == value
        rows = [dict(A=a, trajectory=np.full((16,20,30),float(label)),
                     end=np.full((16,30),float(label)), label=label, baser=label,
                     params=dict(delta=6,infection_rate=1), source=dict(mother_id=f"TEST_ONLY:{label}"))
                for label in (0,1)]
        group = Path(temp) / "csr"
        gen.save_group(group,rows,{**cfg,"large_graph_threshold":1},supervised=True)
        gen.validate_group(group)
        # A late evaluation-YAML failure must happen before official targets move.
        target = Path(temp) / "publication" / "SIS_scaling" / "ER" / "N30"
        target.mkdir(parents=True)
        sentinel = target / "unchanged.txt"
        sentinel.write_text("old data")
        def generator(config,stage):
            folder=stage / "SIS_scaling/ER/N30"
            gen.save_group(folder,rows,config,supervised=True)
            (stage / "scaling_evaluation.yaml").write_text("datasets:\n- path: '"+str(folder)+"'\n")
        with patch.object(gen.yaml,"safe_dump",side_effect=OSError("TEST_ONLY YAML failure")):
            try:
                gen.publish_run(generator,cfg,target.parents[2])
            except OSError:
                pass
            else:
                raise AssertionError("Expected publication preparation failure")
        assert sentinel.read_text()=="old data"
    print("PASS: topology equivalence/rejection bounds, solver equivalence/timeouts, config checks, streamed/standard pickle compatibility", flush=True)


def scaling_cache_micro():
    import json
    import sys
    import yaml
    import generate_sis_dataset as gen
    sys.path.insert(0, str(ROOT.parents[1]))
    from experiment_process import benchmark_sis_scaling as benchmark
    cfg = read_config(ROOT / "config.yaml")
    with tempfile.TemporaryDirectory(prefix="sis_csr_cache_TEST_ONLY_") as temp:
        root = Path(temp)
        rows = []
        for y in (0,1):
            a = sparse.csr_matrix((30,30)) if not y else sparse.csr_matrix(np.ones((30,30))-np.eye(30))
            rows.append(dict(A=a,trajectory=np.full((16,20,30),float(y)),end=np.full((16,30),float(y)),
                label=y,baser=y,params=dict(delta=6,infection_rate=1.),source=dict(mother_id=f"TEST_ONLY:{y}")))
        folder = root / "ER_N30"
        gen.save_group(folder,rows,{**cfg,"large_graph_threshold":1,"save_gbb_times":False},supervised=True)
        gen.validate_group(folder)
        config = dict(methods=["gbb"],device="cpu",test_only=True,trial_ids=[0,1],warmup=1,repeats=2,timing_samples=2,
            datasets=[dict(topology="ER",n=30,path=str(folder),params_file="sis_params.json")])
        config_path = root / "config.yaml"
        config_path.write_text(yaml.safe_dump(config),encoding="utf-8")
        settings = benchmark.read_config(config_path)
        assert benchmark.run(settings,root / "cache")
        record = root / "cache/records.json"
        digest = benchmark.file_hash(record)
        (root / "cache/summary.csv").unlink()
        assert benchmark.run(settings,root / "cache")
        assert benchmark.file_hash(record) == digest
        records = json.loads(record.read_text())
        assert len(records["metrics"]) == 1 and records["metrics"][0]["f1_weighted"] == 1
        print("PASS: CSR -> serial GBB evaluation -> cache reuse/summary recovery; TEST ONLY",flush=True)


def classification_micro(cfg, root):
    import networkx as nx
    from unittest.mock import patch
    import generate_sis_dataset as gen
    import sys
    sys.path.insert(0,str(ROOT.parents[1]))
    from experiment_process.benchmark_sis_scaling import load_group
    from experiment_process.generate_resilience_inference_cache import _load_supervised_split
    toy = copy.deepcopy(cfg)
    toy.update(modes=["classification"], classification_delta=[6,6],
               classification_splits=dict(train_dataset=2,val_dataset=0,test_dataset=0),
               scaling_topologies=["ER","SF"],scaling_sizes=[30],scaling_count=2,save_gbb_times=False)
    def candidates(config,namespace,size_range,topology):
        for i,degree in enumerate((2,8)):
            n = 30 if topology else 22
            seed = gen.seed_for(config["seed"],namespace,i) % (2**32)
            graph = nx.random_regular_graph(degree,n,seed=seed)
            a = sparse.csr_matrix(nx.to_scipy_sparse_array(graph,dtype=float))
            yield i,dict(mother_id=f"{namespace}:{i}",topology=topology or "TEST_ONLY"),iter([(a,"TEST_ONLY_regular")])
    with patch.object(gen,"generate_candidates",side_effect=candidates):
        stage = gen.publish_run(gen.run,toy,root)
    check_folder(root)
    for folder in root.rglob("As.pkl"):
        gen.validate_group(folder.parent)
        data = _load_supervised_split(folder.parent,label_file="rs.pkl",windows=10,observations=5,max_sample_num=1,seed=42,classes=2)
        assert len(data)==2
    for kind in ("ER","SF"):
        group=dict(path=root/f"SIS_scaling/{kind}/N30",n=30,id=kind,params_file="sis_params.json")
        assert not (group["path"]/"test_dataset").exists()
        loaded=load_group(group,dict(seed=42,windows=10,observations=5))
        assert len(loaded)==2 and sorted(s["y"] for s in loaded)==[0,1]
    import yaml
    evaluation=yaml.safe_load((stage/"scaling_evaluation.yaml").read_text(encoding="utf-8"))
    assert evaluation["trial_ids"]==list(range(10))
    assert all(Path(g["path"]).is_dir() for g in evaluation["datasets"])


def check_folder(path):
    for apath in path.rglob("As*.pkl"):
        suffix = apath.stem[2:]
        with apath.open("rb") as stream:
            matrices = pickle.load(stream)
        with apath.with_name(f"numes{suffix}.pkl").open("rb") as stream:
            trajectories = pickle.load(stream)
        with apath.with_name(f"basers{suffix}.pkl").open("rb") as stream:
            basers = pickle.load(stream)
        assert isinstance(matrices, list) and len(matrices) == len(trajectories) == len(basers)
        for A, x, b in zip(matrices, trajectories, basers):
            assert A.dtype == np.float32 and np.array_equal(A, A.T)
            assert x.dtype == np.float64 and x.shape == (16, 20, len(A))
            assert type(b) is int and b in (0, 1)
        label = apath.with_name("rs.pkl")
        if not suffix and label.exists():
            with label.open("rb") as stream:
                labels = pickle.load(stream)
            assert labels.count(0) == labels.count(1)
            with apath.with_name("x_last.pkl").open("rb") as stream:
                ends = pickle.load(stream)
            for A, end, y in zip(matrices, ends, labels):
                assert end.shape == (16, len(A)) and end.dtype == np.float64
                assert y == int(end.mean() >= .001)
        timepath = apath.with_name(f"gbb_times{suffix}.pkl")
        if timepath.exists():
            with timepath.open("rb") as stream:
                times = pickle.load(stream)
            assert len(times) == len(matrices) and all(np.all(np.asarray(t) > 0) for t in times)


if __name__ == "__main__":
    import sys
    if "--large-smoke-test" in sys.argv:
        large_smoke()
        raise SystemExit(0)
    if "--scaling-cache-test" in sys.argv:
        scaling_cache_micro()
        raise SystemExit(0)
    safety_micro()
    cfg = read_config(ROOT / "config.yaml")
    A, _ = build_graph(30, "SF", np.random.default_rng(3))
    again, _ = build_graph(30, "SF", np.random.default_rng(3))
    assert (A != again).nnz == 0
    x = np.random.default_rng(2).random(30)
    np.testing.assert_allclose(sis_rhs(0, x, A, 6), -6*x+(1-x)*(A.toarray() @ x))
    assert gbb(sparse.csr_matrix((3, 3)), 1) == 0
    toy = copy.deepcopy(cfg)
    toy.update(modes=["train", "perturbation"], train_mothers=1, initial_size_range=[22, 22],
               perturbation_n=22, min_nodes=20, save_gbb_times=False)
    with tempfile.TemporaryDirectory() as temp:
        output = Path(temp) / "TEST_ONLY"
        run(toy, output)
        check_folder(output)
        # 原训练加载接口：只读生成文件，不修改训练程序。
        import sys
        project = ROOT.parents[1]
        sys.path.insert(0, str(project))
        from utils.utils import pre_DataSet_spdata
        data, _ = pre_DataSet_spdata(str(output / "SIS"), windows=10, observations=5, max_sample_num=1)
        assert data and data[0].x.shape[1:] == (5, 10, 1)
        assert data[0].y.shape[1:] == (5, 10, 1)
        classification_micro(cfg,Path(temp)/"classification_TEST_ONLY")
    print("PASS: reproducibility, SIS RHS, GBB, train/perturbation formats, original training loader")
