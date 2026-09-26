"""Isolated tests: no formal checkpoints, caches or manuscript figures are written."""
from pathlib import Path
import ast
import copy
import json
import pickle
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch_geometric.data import Data, Batch

PROJECT=Path(__file__).resolve().parents[1]
KEEP_PREVIEW='--keep-preview' in sys.argv
if KEEP_PREVIEW:
    sys.argv.remove('--keep-preview')
sys.path.insert(0,str(PROJECT))
from experiment_process import generate_resilience_inference_cache as api


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear=nn.Linear(1,4)
        with torch.no_grad():
            self.linear.weight.fill_(.2)
            self.linear.bias.fill_(.1)

    def get_representation_dim(self):
        return 4

    def get_zlatent(self,graph):
        return self.linear(graph.x.mean((1,2)))


def fixture(root,classes):
    spec=api.TaskSpec('micro','Micro',classes,'micro',('std',),42,
                      'binary' if classes==2 else 'three_class','rs.pkl','basers.pkl')
    for split in ('train_dataset','val_dataset','test_dataset'):
        folder=root/'dataset/micro'/split
        folder.mkdir(parents=True)
        rng=np.random.default_rng({'train_dataset':1,'val_dataset':2,'test_dataset':3}[split])
        n=21
        arrays=[np.zeros((n,n),dtype=np.float32) for _ in range(12)]
        trajectories=[rng.uniform(0,1,(16,20,n)).astype(np.float64) for _ in arrays]
        labels=[i%classes for i in range(12)]
        for name,values in dict(As=arrays,numes=trajectories,rs=labels,basers=labels).items():
            with (folder/f'{name}.pkl').open('wb') as stream:
                pickle.dump(values,stream)
    base=root/'results/Micro/resilience_inference_analysis'
    base.mkdir(parents=True)
    (base/'model_trained').write_bytes(b'test only encoder fixture')
    (base/'model_trained.yaml').write_text('test_only: true\n')
    return spec


class SmokeTests(unittest.TestCase):
    def test_base_model_and_parameter_cache_first(self):
        from experiment_process import generate_model_parameter_analysis_cache as base
        from experiment_process import generate_figure5_tsne_cache as tsne
        import yaml
        with tempfile.TemporaryDirectory(prefix='prism_base_') as tmp:
            root=Path(tmp); target=root/'model_trained'
            config=yaml.safe_load(api.TASKS['sis_binary'].prism_config.read_text())
            config['dataset']['spdata_file_path']=str(root/'missing_original_data')
            target.with_suffix('.yaml').write_text(yaml.safe_dump(config))
            with self.assertRaisesRegex(FileNotFoundError,'original training inputs'):
                base.ensure_prism_checkpoint(target,'cpu')
            self.assertFalse(target.exists())
            config['train'].update(train_epochs=1,train_batch_size=2,val_batch_size=2)
            target.with_suffix('.yaml').write_text(yaml.safe_dump(config))
            n=21
            edges=torch.stack((torch.arange(n),torch.arange(n).roll(1)))
            edges=torch.cat((edges,edges.flip(0)),dim=1)
            graphs=[Data(x=torch.rand(n,5,10,1),y=torch.rand(n,5,10,1),edge_index=edges) for _ in range(4)]
            with patch.object(base,'_prepare_training_dataset',return_value=(graphs,2.)):
                base.ensure_prism_checkpoint(target,'cpu',task='prediction')
            self.assertTrue(target.is_file())
            checkpoint=torch.load(target,map_location='cpu',weights_only=False)
            from models.models import Resilience_model
            model=Resilience_model(checkpoint['net_param'])
            model.load_state_dict(checkpoint['state_dict'],strict=True)
            with patch.object(base,'_train_model',side_effect=AssertionError('existing model')):
                self.assertEqual(base.ensure_prism_checkpoint(target,'cpu'),target)
        # Complete caches are usable even when configuration/model sources are absent.
        with patch.object(base,'_load_config',side_effect=AssertionError('cache needs no config')), \
             patch.object(base,'_train_model',side_effect=AssertionError('cache needs no training')):
            spec=base.PARAMETER_GROUPS['learning_rate'][0][1]
            self.assertEqual(len(base.ensure_run(spec)),4)
        with patch.object(tsne,'missing_inputs',side_effect=AssertionError('cache needs no sources')), \
             patch.object(tsne,'_resolve_device',side_effect=AssertionError('cache needs no GPU')):
            self.assertEqual(len(tsne.ensure_figure5_caches()),2)

    def test_resinf_recovery_and_cache_first(self):
        from experiment_process import resinf_inference as resinf
        torch.set_num_threads(1)
        for classes in (2, 3):
            with self.subTest(classes=classes), tempfile.TemporaryDirectory(prefix='prism_resinf_') as tmp:
                root=Path(tmp); spec=fixture(root,classes)
                with patch.object(api,'RESULTS_ROOT',root/'results'), patch.object(api,'DATASET_ROOT',root/'dataset'), \
                     patch.dict(api.TASKS,{'micro':spec}), patch.object(resinf,'EPOCHS',1):
                    target=api.trained_model_path(spec,'resinf',1.,0)
                    target.parent.mkdir(parents=True)
                    target.touch()  # Authorized empty-checkpoint recovery.
                    options=dict(ratios=[1.],trial_ids=[0],device='cpu')
                    paths=api.ensure_caches('micro',('resinf',),**options)
                    self.assertGreater(target.stat().st_size,0)
                    before=api._sha256(target)
                    with patch.object(resinf,'train_one',side_effect=AssertionError('must not train')), \
                         patch.object(api,'load_split',side_effect=AssertionError('cache requires no data')):
                        self.assertEqual(api.ensure_caches('micro',('resinf',),**options),paths)
                    paths['resinf'].unlink()
                    with patch.object(resinf,'train_one',side_effect=AssertionError('summary only')):
                        self.assertTrue(api.ensure_caches('micro',('resinf',),**options)['resinf'].is_file())
                    with patch.object(resinf,'train_one',side_effect=AssertionError('existing model')):
                        api.ensure_caches('micro',('resinf',),force=True,**options)
                    self.assertEqual(before,api._sha256(target))
                    target.write_bytes(b'nonempty corrupt checkpoint')
                    with patch.object(resinf,'train_one') as trainer:
                        with self.assertRaises(Exception):
                            api.ensure_caches('micro',('resinf',),force=True,**options)
                        trainer.assert_not_called()
                    self.assertEqual(target.read_bytes(),b'nonempty corrupt checkpoint')

    def test_auto_training_cache_knn_and_isolation(self):
        for classes in (2,3):
            with self.subTest(classes=classes), tempfile.TemporaryDirectory(prefix='prism_inference_smoke_') as tmp:
                root=Path(tmp)
                spec=fixture(root,classes)
                with patch.object(api,'RESULTS_ROOT',root/'results'), patch.object(api,'DATASET_ROOT',root/'dataset'), \
                     patch.dict(api.TASKS,{'micro':spec}), patch.object(api,'TRAIN_EPOCHS',2), \
                     patch.object(api,'_build_prism',side_effect=lambda s,d:TinyEncoder().to(d)):
                    options=dict(ratios=[1.],trial_ids=[0,2],device='cpu')
                    paths=api.ensure_caches('micro',('mlp','knn'),**options)
                    models=list((root/'results').rglob('model.pth'))
                    self.assertEqual(len(models),2)
                    self.assertFalse(list((root/'results').rglob('model.pkl')))
                    hashes={str(p):api._sha256(p) for p in models}
                    with patch.object(api,'train_missing',side_effect=AssertionError('unexpected training')), \
                         patch.object(api,'evaluate_task',side_effect=AssertionError('unexpected inference')):
                        self.assertEqual(paths,api.ensure_caches('micro',('mlp','knn'),**options))
                        paths['mlp'].unlink()  # Owned temporary summary, not a formal result.
                        api.ensure_caches('micro',('mlp',),**options)
                    self.assertEqual(hashes,{str(p):api._sha256(p) for p in models})
                    csv,records=api.cache_paths(spec,'knn',(1.,),(0,2))
                    payload=json.loads(records.read_text())
                    self.assertEqual(payload['test_indices'],list(range(12)))
                    self.assertEqual(payload['gbb_predictions'],payload['test_labels'])
                    first_predictions=payload['results']['1.0']['samples'][0]['predictions']
                    with patch.object(api,'extract_features',side_effect=AssertionError('features should be reused')):
                        api.ensure_caches('micro',('knn',),force=True,**options)
                    again=json.loads(records.read_text())
                    self.assertEqual(first_predictions,again['results']['1.0']['samples'][0]['predictions'])
                    train=api.load_split(spec,'train_dataset')
                    for row in again['results']['1.0']['samples']:
                        expected=api.reference_indices(spec,train,1.,row['sample_idx'])
                        self.assertEqual(row['reference_positions'],expected)
                    # An incomplete set is never accepted as the requested two repeats.
                    again['results']['1.0']['samples'].pop()
                    self.assertFalse(api.valid_payload(again,again['fingerprint']))
                    with self.assertRaises(ValueError):
                        api._f1_summary([dict(sample_idx=0,test_f1=.5)],trial_ids=[0,2])
                    # Data/config changes invalidate cache but do not retrain existing weights.
                    spec.prism_config.write_text('test_only: true\nchanged: true\n')
                    with patch.object(api,'train_missing',side_effect=AssertionError('existing weights must remain')):
                        api.ensure_caches('micro',('mlp',),**options)
                    # Corruption is a load error, never an excuse to retrain.
                    path=api.available_model_path(spec,'mlp',1.,0)
                    path.write_bytes(b'corrupt temporary checkpoint')
                    with patch.object(api,'train_missing') as forbidden_training:
                        with self.assertRaises(Exception):
                            api.ensure_caches('micro',('mlp',),ratios=[1.],trial_ids=[0],device='cpu')
                        forbidden_training.assert_not_called()

    def test_mapping_statistics_and_conflict(self):
        canonical=api.Classifier(TinyEncoder(),3).state_dict()
        reverse={}
        for k,v in canonical.items():
            key=k.replace('pool.','seq_emb_agg.').replace('body.','classifier.mlp_layers.').replace('output.','classifier.k_output.')
            key={'scale':'classifier.w','bias':'classifier.b'}.get(key,key)
            reverse[key]=v
        mapped=api.remap_state(reverse)
        self.assertEqual(set(mapped),set(canonical))
        for k in mapped: self.assertTrue(torch.equal(mapped[k],canonical[k]))
        model=api.Classifier(TinyEncoder(),3)
        model.load_state_dict(mapped,strict=True)
        model.train()
        self.assertTrue(model.encoder.training)
        self.assertTrue(all(not p.requires_grad for p in model.encoder.parameters()))
        scores=[dict(sample_idx=i,test_f1=float(i)/9) for i in range(10)]
        summary=api._f1_summary(scores)
        self.assertAlmostEqual(summary['std_test_f1'],np.std(np.arange(10)/9,ddof=0))
        for ratios,trials in (([],[0]),([1.],[0,0]),([1.],[10])):
            with self.assertRaises(ValueError): api.selection(ratios,trials)
        with tempfile.TemporaryDirectory(prefix='prism_alias_smoke_') as tmp:
            spec=fixture(Path(tmp),2)
            with patch.object(api,'RESULTS_ROOT',Path(tmp)/'results'):
                for name,content in [('0',b'a'),('00',b'b')]:
                    p=api.classifier_model_root(spec,'mlp')/f'ratio_1.000_sample_{name}/model.pth'
                    p.parent.mkdir(parents=True); p.write_bytes(content)
                with self.assertRaisesRegex(ValueError,'Conflicting'):
                    api.available_model_path(spec,'mlp',1.,0)

    def test_real_checkpoints_read_only(self):
        torch.set_num_threads(1)
        for task in ('sis_binary','neuronal_three_class'):
            spec=api.TASKS[task]
            path=api.available_model_path(spec,'mlp',.1,0)
            self.assertIsNotNone(path)
            before=api._sha256(path)
            model,checkpoint,_=api.load_mlp(spec,.1,0,device='cpu')
            graph=Data(x=torch.linspace(0,1,21*5*10).reshape(21,5,10,1),
                       edge_index=torch.empty((2,0),dtype=torch.long),num_nodes=21)
            with torch.inference_mode():
                actual=model(graph)
                z=model.encoder.get_zlatent(graph)
                pooled=(torch.softmax(model.pool.attention(z),dim=1)*z).mean(0,keepdim=True)
                expected=model.output(model.body(pooled))
                if spec.classes==2: expected=torch.sigmoid(model.scale*expected+model.bias)
            torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-7)
            # Compare batched inference to independent graph calls without
            # importing or executing any historical experiment source.
            batch=Batch.from_data_list([graph,graph.clone()])
            with torch.inference_mode():
                expected=torch.cat([model(graph),model(graph.clone())],dim=0)
                torch.testing.assert_close(model(batch),expected,rtol=1e-6,atol=1e-7)
            for key,tensor in api.remap_state(checkpoint['model_state_dict']).items():
                self.assertTrue(torch.equal(model.state_dict()[key].cpu(),tensor.cpu()))
            self.assertEqual(before,api._sha256(path))

    def test_legacy_protocol_and_resume(self):
        from utils.utils import select_balanced_samples, select_balanced_samples_multi
        import random
        for classes in (2,3):
            data=[Data(y=torch.tensor([i%classes])) for i in range(12)]
            groups={c:[i for i,g in enumerate(data) if int(g.y)==c] for c in range(classes)}
            random.seed(233)
            expected=(select_balanced_samples(groups[1],groups[0],sample_seed=233,num_per_class=3)[0]
                      if classes==2 else select_balanced_samples_multi(groups,sample_seed=233,num_per_class=3))
            self.assertEqual(api._balanced_indices(data,classes,3,233),expected)
        for epoch in range(51):
            expected=epoch/5 if epoch<5 else .5*(1+np.cos(np.pi*(epoch-5)/45))
            self.assertAlmostEqual(api.learning_rate_factor(epoch),expected)
        with tempfile.TemporaryDirectory(prefix='prism_resume_') as tmp:
            root=Path(tmp); spec=fixture(root,2)
            with patch.object(api,'RESULTS_ROOT',root/'results'),patch.object(api,'DATASET_ROOT',root/'dataset'), \
                 patch.dict(api.TASKS,{'micro':spec}),patch.object(api,'_build_prism',side_effect=lambda s,d:TinyEncoder()):
                options=dict(ratios=[1.],trial_ids=[0,2],device='cpu')
                original=api.reference_indices
                def interrupted(s,data,ratio,trial,**kw):
                    if trial==2: raise RuntimeError('simulated interruption')
                    return original(s,data,ratio,trial,**kw)
                with patch.object(api,'reference_indices',side_effect=interrupted):
                    with self.assertRaisesRegex(RuntimeError,'simulated'):
                        api.ensure_caches('micro',('knn',),**options)
                _,path=api.cache_paths(spec,'knn',(1.,),(0,2))
                partial=json.loads(path.read_text())
                self.assertFalse(partial['complete'])
                self.assertEqual(len(partial['results']['1.0']['samples']),1)
                with patch.object(api,'reference_indices',wraps=original) as resumed:
                    api.ensure_caches('micro',('knn',),**options)
                    self.assertEqual([c.args[3] for c in resumed.call_args_list],[2])
                self.assertTrue(json.loads(path.read_text())['complete'])
                self.assertIn('common_test_support_v1',path.name)
                # KNN legacy summaries are never reused because their GBB
                # support was not aligned with the learned classifier.
                csv=root/'old.csv'
                import pandas as pd
                frame=pd.DataFrame(dict(data_ratio=api.RATIOS,knn_mean_f1=.5,knn_std_f1=0.,knn_min_f1=.5,knn_max_f1=.5,baser_mean_f1=.4,improvement_mean=.1))
                frame.to_csv(csv,index=False)
                with patch.dict(api.LEGACY_CSV,{('micro','knn'):csv}):
                    self.assertIsNone(api.historical_cache(spec,'knn'))
                canonical=api.canonical_csv(spec,'knn')
                frame.to_csv(canonical,index=False)
                api.canonical_trials(spec,'knn').write_text(json.dumps({'fingerprint':{'policy':'full_test_N_gt_20'}}))
                self.assertIsNone(api.historical_cache(spec,'knn'))
                # Legacy JSON is not run-linked to the plotting CSV: old code
                # reused one JSON filename across KNN configurations/reruns.
                records=api.canonical_trials(spec,'knn')
                records.write_text(json.dumps({'num_samples':api.NUM_TRIALS,
                    'data_ratios':list(api.RATIOS),'results':{
                        str(r):{'samples':[{'test_f1':.2} for _ in range(api.NUM_TRIALS)]}
                        for r in api.RATIOS}}))
                self.assertIsNone(api.historical_cache(spec,'knn'))

    def test_best_validation_snapshot(self):
        with tempfile.TemporaryDirectory(prefix='prism_best_') as tmp:
            root=Path(tmp); spec=fixture(root,2)
            instances=[]; snapshots=[]
            classifier=api.Classifier
            def build(*args,**kwargs):
                model=classifier(*args,**kwargs); instances.append(model); return model
            def score(*args,**kwargs):
                snapshots.append({k:v.detach().cpu().clone() for k,v in instances[-1].state_dict().items()})
                return .9 if len(snapshots)==1 else .1
            with patch.object(api,'RESULTS_ROOT',root/'results'),patch.object(api,'DATASET_ROOT',root/'dataset'), \
                 patch.dict(api.TASKS,{'micro':spec}),patch.object(api,'TRAIN_EPOCHS',2), \
                 patch.object(api,'_build_prism',side_effect=lambda s,d:TinyEncoder()), \
                 patch.object(api,'Classifier',side_effect=build),patch.object(api,'f1_score',side_effect=score):
                api.train_missing('micro',device='cpu',ratios=[1.],trial_ids=[0])
                saved=torch.load(api.available_model_path(spec,'mlp',1.,0),weights_only=False)
                self.assertEqual(saved['best_epoch'],0)
                self.assertEqual(saved['config']['checkpoint_selection'],'best_validation_clone')
                self.assertEqual(saved['config']['batch_size'],api.BATCH_SIZE)
                for k,v in snapshots[0].items():
                    self.assertTrue(torch.equal(v,saved['model_state_dict'][k]))
                self.assertTrue(any(not torch.equal(v,snapshots[1][k]) for k,v in snapshots[0].items()))

    def test_existing_plot_cache_read_only(self):
        import warnings
        import pandas as pd
        sys.path.insert(0,str(PROJECT/'experiment_visualization'))
        import resilience_inference_plotting as plotting
        with patch.object(api,'fingerprint',side_effect=AssertionError('No recomputation')), \
             patch.object(api,'train_missing',side_effect=AssertionError('No training')), \
             patch.object(plotting,'ensure_caches',side_effect=api.ensure_caches),warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            for task in api.TASKS:
                frames=plotting.load_task(task)
                for method,frame in frames.items():
                    source=Path(frame.attrs['source']); before=api._sha256(source)
                    pd.testing.assert_frame_equal(frame,pd.read_csv(source))
                    stem={'mlp':'frozen','knn':'knn','resinf':'Resinf'}[method]
                    expected=np.vstack((frame[f'{stem}_mean_f1']-frame[f'{stem}_min_f1'],frame[f'{stem}_max_f1']-frame[f'{stem}_mean_f1']))
                    np.testing.assert_allclose(plotting._bounds(frame,stem),np.maximum(expected,0))
                    self.assertEqual(before,api._sha256(source))

    def test_visualization_wiring_and_exports(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import pandas as pd
        sys.path.insert(0,str(PROJECT/'experiment_visualization'))
        import resilience_inference_plotting as plotting
        from plot_knn_gbb_comparison import main
        frame=pd.DataFrame(dict(data_ratio=[.1,1.],train_size=[4,12],
            knn_mean_f1=[.5,.6],knn_std_f1=[.1,.1],knn_min_f1=[.4,.5],knn_max_f1=[.6,.7],
            baser_mean_f1=[.5,.5],improvement_mean=[0,.1]))
        with tempfile.TemporaryDirectory(prefix='prism_plot_smoke_') as tmp:
            tmp=Path(tmp)
            source=tmp/'knn.csv'; frame.to_csv(source,index=False)
            options=dict(ratios=[.1,1.],trial_ids=[0,2],device='cpu')
            with patch.object(plotting,'ensure_caches',return_value={'knn':source}) as ensure:
                loaded=plotting.load_task('sis_binary',methods=('knn',),**options)
                self.assertEqual(len(loaded['knn']),2)
                self.assertEqual(ensure.call_args.kwargs['trial_ids'],(0,2))
            original_save=plotting.save_figure
            with patch('plot_knn_gbb_comparison.load_task',return_value={'knn':frame}), \
                 patch('plot_knn_gbb_comparison.save_figure',side_effect=lambda fig,stem,**kw:original_save(fig,stem,output_dir=tmp,test_only=True,**kw)):
                paths=main(**options)
            self.assertEqual({p.suffix for p in paths},{'.svg','.pdf','.png'})
            self.assertTrue(all(p.stat().st_size>1000 for p in paths))
            svg=next(p for p in paths if p.suffix=='.svg').read_text(encoding='utf-8')
            self.assertIn('TEST ONLY',svg)
            self.assertIn('<text',svg)
            if KEEP_PREVIEW:
                import shutil
                target=PROJECT/'tests_artifacts/resilience_inference_smoke'
                target.mkdir(parents=True,exist_ok=True)
                for p in paths:
                    api.archive(target/p.name)
                    shutil.copy2(p,target/p.name)


if __name__=='__main__':
    unittest.main(verbosity=2)
