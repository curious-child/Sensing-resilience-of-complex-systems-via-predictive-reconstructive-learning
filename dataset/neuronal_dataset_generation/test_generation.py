"""Micro tests only: temporary data are never manuscript results."""
import copy
import importlib
import itertools
import json
from pathlib import Path
import pickle
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy import sparse
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.special import lambertw, expit

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_neuronal_dataset as gen

PROJECT = gen.ROOT.parents[1]
sys.path.insert(0, str(PROJECT))


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.cfg = gen.read_config(gen.ROOT / 'config.yaml')

    def test_folds_and_stability(self):
        for mu, d, reference in gen.REFERENCE:
            low, high = gen.critical_values(mu, d)
            roots = [1 - lambertw(-np.exp(1-mu), branch).real for branch in (0, -1)]
            np.testing.assert_allclose([high, low], [y*y/(d*(y-1)) for y in roots], rtol=1e-12)
            for beta, y in zip((high, low), roots):
                x = y/d
                self.assertAlmostEqual(-x + beta*expit(d*x-mu), 0, places=11)
                self.assertAlmostEqual(-1 + beta*d*expit(d*x-mu)*(1-expit(d*x-mu)), 0, places=11)
            for beta, expected in ((low*.9, 1), ((low+high)/2, 3), (high*1.1, 1)):
                f = lambda x: -x + beta*expit(d*x-mu)
                grid = np.linspace(0, beta+1, 20000)
                indices = np.flatnonzero(f(grid[:-1])*f(grid[1:]) < 0)
                eq = [brentq(f, grid[i], grid[i+1]) for i in indices]
                self.assertEqual(len(eq), expected)
                stable = [-1+beta*d*expit(d*x-mu)*(1-expit(d*x-mu)) < 0 for x in eq]
                self.assertEqual(stable, [True] if expected == 1 else [True, False, True])

    def test_labels_boundaries_and_gbb(self):
        low, high = gen.critical_values(3.5, 2)
        for b in (low, high, low*(1-5e-11), high*(1+5e-11)):
            self.assertEqual(gen.classify_beta(b, low, high), (1, True))
        self.assertEqual(gen.classify_beta(low*.9, low, high), (0, False))
        self.assertEqual(gen.classify_beta(high*1.1, low, high), (2, False))
        for mu, d in ((2, 2), (3, 0), (np.nan, 2)):
            with self.assertRaises(ValueError): gen.critical_values(mu, d)
        self.assertEqual(gen.gbb(sparse.csr_matrix((5, 5)), (low, high))['three_class'], 0)
        a, _ = gen.build_graph(40, 'SF', np.random.default_rng(123))
        dense = a.toarray()
        self.assertAlmostEqual(gen.gbb(a, (low, high))['beta_eff'], (dense@dense).sum()/dense.sum())
        for values, expected in (([0, 0], (0, 0)), ([0, 5], (1, 0)), ([4, 4], (2, 1)), ([.5, 3.5], (0, 0))):
            self.assertEqual(gen.true_labels(np.repeat(np.array(values)[:, None], 3, axis=1)), expected)
        np.testing.assert_array_equal(gen.quotas(24), [8, 8, 8])

    def test_dynamics_grid_and_initials(self):
        a, _ = gen.build_graph(30, 'ER', np.random.default_rng(42))
        init = gen.initial_conditions(30, np.random.default_rng(52))
        self.assertEqual(init.shape, (16, 30))
        np.testing.assert_array_equal(init[0], 0)
        np.testing.assert_array_equal(init[-1], 5)
        self.assertTrue(np.all(init == init[:, :1]))
        np.testing.assert_array_equal(init, gen.initial_conditions(30, np.random.default_rng(52)))
        np.testing.assert_allclose(gen.neuronal_rhs(0, init[1], a, 3, 1.5), -init[1]+a.toarray()@(1/(1+np.exp(3-1.5*init[1]))))
        for mode, length in [('train', 20), ('perturbation', 20), ('classification', 100)]:
            p = {'mu': 3.5, 'd': 2.}
            x, end, records, starts = gen.simulate(a, p, self.cfg, mode, 52)
            self.assertEqual(x.shape, (16, length, 30))
            np.testing.assert_array_equal(x[:, 0], init)
            grid = gen.sampling_grid(self.cfg, mode)
            # Independent dense RHS with original equation, matched tolerances.
            ref = solve_ivp(lambda t,v: -v+a.toarray()@(1/(1+np.exp(p['mu']-p['d']*v))),
                            (0, grid[-1]), init[-1], t_eval=grid, rtol=1e-6, atol=1e-9)
            np.testing.assert_allclose(x[-1], ref.y.T, atol=2e-5, rtol=2e-5)
            if mode != 'train':
                self.assertEqual(end.shape, (16, 30))
                self.assertTrue(all(v['change']<=1e-6 and v['residual']<=1e-7 for v in records))
        fail = copy.deepcopy(self.cfg)
        fail['simulation'].update(T=2., max_T=2.)
        with self.assertRaisesRegex(RuntimeError, 'No convergence'):
            gen.simulate(a, p, fail, 'classification', 52)
        fail['simulation']['max_T'] = 12800
        short, end, records, _ = gen.simulate(a, p, fail, 'classification', 52)
        self.assertTrue(any(r['time']>2 for r in records))
        self.assertFalse(np.allclose(short[:, -1], end))
        self.assertEqual(gen.sampling_grid(fail, 'classification')[-1], 2)

    def test_isolation_sampling_and_limits(self):
        a, _ = gen.build_graph(30, 'SF', np.random.default_rng(5))
        isolation = gen.Isolation()
        isolation.add(a, 'train')
        self.assertTrue(isolation.duplicate(a.copy(), 'test'))
        self.assertFalse(isolation.duplicate(a, 'train'))
        def fingerprints():
            return [(m,i,gen.graph_hash(a)) for m,s,i,a,aug in itertools.islice(gen.classification_candidates(self.cfg,'micro'),70)]
        self.assertEqual(fingerprints(), fingerprints())
        cfg = copy.deepcopy(self.cfg)
        cfg['max_candidates'] = 1
        candidate = (0, {}, 0, a, 'original_gcc')
        with patch.object(gen, 'classification_candidates', return_value=iter([candidate])), patch.object(gen, 'make_sample', return_value={'three_class':0}):
            with self.assertRaisesRegex(RuntimeError, 'candidate limit reached'):
                gen.collect_classification(cfg,'limited',12,gen.Isolation())

    def test_temporary_modes_and_original_loaders(self):
        cfg = copy.deepcopy(self.cfg)
        cfg.update(modes=['train', 'perturbation'], initial_size_range=[30, 30], train_mothers=1,
                   perturbation_n=23, min_nodes=22, save_gbb_times=True, gbb_warmup=1, gbb_repeats=2)
        with tempfile.TemporaryDirectory(prefix='neuronal_micro_') as tmp:
            out = Path(tmp)/'run'
            gen.run(cfg, out)
            from utils.utils import pre_DataSet_spdata
            ds, _ = pre_DataSet_spdata(str(out/'Neuronal'), windows=10, observations=5, max_sample_num=1)
            self.assertGreater(len(ds), 0)
            module = importlib.import_module('experiment_process.generate_inductive_analysis_cache')
            for kind in ('ER', 'SF'):
                for strategy in ('degree', 'random'):
                    suffix = f'_{kind}_{strategy}.pkl'
                    base = out/'Neuronal_test'
                    paths = dict(adjacency=base/f'As{suffix}', trajectory=base/f'numes{suffix}', stable_state=base/f'xlast{suffix}')
                    with patch.object(module, 'dataset_paths', return_value=paths):
                        aa, xx, ee = module._load_dataset('Neuronal', kind, strategy)
                    self.assertTrue(len(aa)>0)
                    for a,x,e in zip(aa,xx,ee):
                        self.assertEqual(a.dtype, np.float32)
                        self.assertEqual(x.shape, (16,20,len(a)))
                        self.assertEqual(e.shape, (16,len(a)))
                    with (base/f'gbb_times{suffix}').open('rb') as f:
                        self.assertTrue(all(len(t)==2 and min(t)>0 for t in pickle.load(f)))
            with self.assertRaises(FileExistsError): gen.run(cfg, out)

    def test_shared_labels_and_safe_publication(self):
        import networkx as nx
        from sis_dataset_generation import generate_sis_dataset as sis
        from experiment_process.generate_resilience_inference_cache import _load_supervised_split, cache_matches_dataset, stamp_dataset
        cfg = copy.deepcopy(self.cfg)
        cfg["classification_splits"] = dict(train_dataset=3,val_dataset=3,test_dataset=3)
        def candidates(config, namespace):
            split = list(config["classification_splits"]).index(namespace.split(":")[-1])
            for i, degree in enumerate((2,5,8)):
                n = 22+split*6+i*2
                a = sparse.csr_matrix(nx.to_scipy_sparse_array(nx.random_regular_graph(degree,n,seed=10+split),dtype=float))
                yield i,dict(mother_id=f"{namespace}:{i}"),0,a,"TEST_ONLY_regular"
        with tempfile.TemporaryDirectory(prefix="neuronal_publish_TEST_ONLY_") as tmp:
            root = Path(tmp)
            unrelated = root/"SIS"
            unrelated.mkdir()
            (unrelated/"sentinel").write_text("unchanged")
            with patch.object(gen,"classification_candidates",side_effect=candidates):
                first = gen.publish_run(gen.run,cfg,root)
            for split in cfg["classification_splits"]:
                folder = root/"Neuronal_supervised"/split
                gen_file = json.loads((folder/"generation.json").read_text(encoding="utf-8"))
                self.assertEqual(gen_file["three_class_counts"],[1,1,1])
                self.assertEqual(gen_file["binary_counts"],[2,1])
                sis.validate_group(folder)
                for name, classes in (("binary_rs.pkl",2),("three_class_rs.pkl",3)):
                    data = _load_supervised_split(folder,label_file=name,windows=10,observations=5,max_sample_num=1,seed=42,classes=classes)
                    self.assertEqual(len(data),3)
            self.assertFalse((root/"binary_balanced").exists())
            original = (root/"Neuronal_supervised/train_dataset/As.pkl").read_bytes()
            version = (root/"Neuronal_supervised/train_dataset/generation.json").read_bytes()
            only_train = copy.deepcopy(cfg)
            only_train["classification_splits"] = dict(train_dataset=3)
            cache = root/"cache.json"
            gen.dump(cache,{})
            self.assertFalse(cache_matches_dataset(cache, root/"Neuronal_supervised"))
            stamp_dataset(cache,root/"Neuronal_supervised")
            self.assertTrue(cache_matches_dataset(cache,root/"Neuronal_supervised"))
            def failed(config, stage):
                stage.mkdir(parents=True)
                raise RuntimeError("TEST_ONLY generation failure")
            with self.assertRaisesRegex(RuntimeError,"generation failure"):
                sis.publish_run(failed,cfg,root)
            self.assertEqual(version,(root/"Neuronal_supervised/train_dataset/generation.json").read_bytes())
            rename = Path.rename
            def interrupted_publish(source, target):
                if ".staging" in source.parts and source.name == "val_dataset":
                    raise OSError("TEST_ONLY publication interruption")
                return rename(source,target)
            with patch.object(gen,"classification_candidates",side_effect=candidates), patch.object(Path,"rename",interrupted_publish):
                with self.assertRaisesRegex(OSError,"publication interruption"):
                    gen.publish_run(gen.run,cfg,root)
            self.assertEqual(version,(root/"Neuronal_supervised/train_dataset/generation.json").read_bytes())
            val_version = (root/"Neuronal_supervised/val_dataset/generation.json").read_bytes()
            with patch.object(gen,"classification_candidates",side_effect=candidates):
                second = gen.publish_run(gen.run,only_train,root)
            self.assertEqual(original,(root/"archive"/second.name/"Neuronal_supervised/train_dataset/As.pkl").read_bytes())
            self.assertEqual(val_version,(root/"Neuronal_supervised/val_dataset/generation.json").read_bytes())
            self.assertEqual((unrelated/"sentinel").read_text(),"unchanged")
            self.assertFalse(cache_matches_dataset(cache,root/"Neuronal_supervised"))


def validate_delivery(output):
    """Validate all six real splits, using the existing classification loader."""
    from experiment_process.generate_resilience_inference_cache import _load_supervised_split
    owners, mothers, total, summary = {}, {}, 0, []
    for folder in sorted(Path(output).glob('Neuronal_supervised/*_dataset')):
        files = {p.stem: pickle.loads(p.read_bytes()) for p in folder.glob('*.pkl')}
        n = len(files['As'])
        assert all(type(v) is list and len(v)==n for v in files.values())
        metadata = json.loads((folder/'generation.json').read_text(encoding='utf-8'))
        params = json.loads((folder/'neuronal_params.json').read_text(encoding='utf-8'))
        np.testing.assert_array_equal(np.bincount(files['three_class_rs'], minlength=3), gen.quotas(n))
        for i,a in enumerate(files['As']):
            x,e = files['numes'][i],files['x_last'][i]
            assert a.dtype==np.float64 and x.dtype==np.float64 and e.dtype==np.float64
            assert a.shape==(len(a),len(a)) and x.shape==(16,100,len(a)) and e.shape==(16,len(a))
            assert np.isfinite(x).all() and np.isfinite(e).all() and np.array_equal(a,a.T)
            assert np.isin(a,[0,1]).all() and np.diag(a).sum()==0
            for name in ('binary_rs','three_class_rs','binary_basers','three_class_basers'):
                assert type(files[name][i]) is int
            assert gen.true_labels(e)==(files['three_class_rs'][i],files['binary_rs'][i])
            prediction=gen.gbb(sparse.csr_matrix(a),gen.critical_values(**params[i]))
            assert (prediction['three_class'],prediction['binary'])==(files['three_class_basers'][i],files['binary_basers'][i])
            source=metadata['sources'][i]
            np.testing.assert_array_equal(x[:,0], np.repeat(np.asarray(source['initial_values'])[:,None],len(a),axis=1))
            assert source['actual_n']==len(a)
            assert all(r['time']>=200 for r in source['convergence'])
            residual=np.max(np.abs(-e+expit(params[i]['d']*e-params[i]['mu'])@a.T),axis=1)
            assert np.all(residual<=1e-7)
            key=gen.graph_hash(sparse.csr_matrix(a))
            assert key==source['adjacency_sha256']
            assert owners.setdefault(key,str(folder))==str(folder)
            assert mothers.setdefault(source['mother_id'],str(folder))==str(folder)
            assert all(r['change']<=1e-6 and r['residual']<=1e-7 for r in source['convergence'])
        for label,classes in [('binary_rs.pkl',2),('three_class_rs.pkl',3)]:
            data = _load_supervised_split(folder,label_file=label,windows=10,observations=5,max_sample_num=1,seed=42,classes=classes)
            assert len(data)==n
        total+=n
        summary.append(dict(folder=str(folder.relative_to(output)),samples=n,counts=metadata['three_class_counts']))
    assert total==48 and len(summary)==3, (total,summary)
    gen.dump(Path(output)/'validation.json',dict(status='passed',total=total,splits=summary,
        checks='Original binary/three-class loaders, dtypes/shapes, labels/GBB, convergence, mother and exact adjacency separation'))
    print(f'Validated {total} samples with original loaders.')


if __name__=='__main__':
    if len(sys.argv)==3 and sys.argv[1]=='--validate-output':
        validate_delivery(Path(sys.argv[2]).resolve())
    else:
        unittest.main()
