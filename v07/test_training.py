"""CPU checks for continuous state, document boundaries, masking and exact recovery."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from .common import Console, save
from .data import ContinuousStream, Corpus, ROOT, prepare
from .model import Config, Network, initialize
from .run import (build, cases, checkpoint, load_config, move_state, restore_rng,
                  train, train_update, validate_config)


def fixture():
    documents = [np.array([1, *[4+(i+j)%55 for j in range(length-2)], 2], dtype=np.int64)
                 for i, length in enumerate((43, 7, 31, 54))]
    offsets = np.r_[0, np.cumsum([len(doc) for doc in documents])]
    tokens = np.concatenate(documents)
    corpus = SimpleNamespace(offsets={split: offsets for split in ('train', 'validation', 'test')},
                             tokens={split: tokens for split in ('train', 'validation', 'test')},
                             manifest={'fixture': 'synthetic-boundary-test', 'vocab_size': 64})
    corpus.sample = lambda *args: Corpus.sample(corpus, *args)
    return corpus


def assert_tree(test, a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    elif isinstance(a, dict):
        test.assertEqual(a.keys(), b.keys())
        for key in a:
            assert_tree(test, a[key], b[key])
    elif isinstance(a, (tuple, list)):
        test.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            assert_tree(test, x, y)
    else:
        test.assertEqual(a, b)


class TrainingProtocol(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def config(self, **extra):
        return load_config() | dict(vocab_size=64, matrix_size=4, width=16, modes=2, nodes=[32],
                                   seeds=[11], variants=['full'], train_steps=3, batch_size=2,
                                   sequence_tokens=8, bptt_tokens=4, internal_steps=4,
                                   validation_windows=1, test_windows=1, prefix_tokens=4, target_tokens=4,
                                   eval_every=3, checkpoint_every=1, cpu_threads=1, device='cpu', session_tokens=64, intrusion_tokens=4, intrusion_interval=16) | extra

    def model(self, variant='full'):
        cfg = Config(vocab_size=64, matrix_size=4, width=16, modes=2, nodes=32, internal_steps=4,
                     variant=variant, transformer_width=16, transformer_ff=32, context_limit=256)
        model = build(cfg); initialize(model, 41)
        return model

    def test_context_stream_sources_masks_no_oracle_resets_and_exact_resume(self):
        corpus = fixture()
        options = dict(session_tokens=64, intrusion_tokens=4, intrusion_interval=16, marked_probability=.5)
        stream = ContinuousStream(corpus, 2, 8, 7, 11, **options)
        previous = None; saw_switch_without_reset = False; max_age = 0
        for step in range(40):
            values, mask, reset, info = stream.next()
            self.assertEqual(reset.shape, mask.shape)
            for lane in range(2):
                records = [None]*9
                for span in info['sources'][lane]:
                    for index in range(span['begin'], span['end']):
                        doc, offset = span['document'], span['offset']+index-span['begin']
                        records[index] = (doc, offset)
                        if doc >= 0:
                            self.assertEqual(values[lane,index], corpus.tokens['train'][corpus.offsets['train'][doc]+offset])
                for t in range(8):
                    left, right = records[t:t+2]
                    real_adjacent = left[0] >= 0 and left[0] == right[0] and right[1] == left[1]+1
                    self.assertEqual(bool(mask[lane,t]), real_adjacent)
                self.assertEqual(int(mask[lane].sum()), info['valid_tokens'][lane])
                self.assertEqual(info['age_start'][lane], (step%8)*8)
                self.assertEqual(bool(reset[lane,0]), step%8 == 0)
                self.assertFalse(bool(reset[lane,1:].any()))
                if previous is not None:
                    self.assertEqual(values[lane,0], previous[lane,-1])
                saw_switch_without_reset |= not bool(reset[lane].any()) and not bool(mask[lane].all())
                max_age = max(max_age, info['age_end'][lane])
            previous = values
        self.assertTrue(saw_switch_without_reset); self.assertEqual(max_age, 64)
        self.assertGreater(stream.switches['intrusion'], 0); self.assertGreater(stream.switches['return_to_A'], 0)
        self.assertGreater(stream.switches['marked'], 0); self.assertGreater(stream.switches['unmarked'], 0)
        saved = stream.state_dict()
        self.assertTrue(all(lane['pending'] is not None for lane in saved['lanes']))
        clone = ContinuousStream(corpus, 2, 8, 7, 11, **options); clone.load_state_dict(deepcopy(saved))
        for _ in range(12):
            expected, actual = stream.next(), clone.next()
            for a, b in zip(expected[:3], actual[:3]):
                self.assertTrue(np.array_equal(a,b))
            self.assertEqual(expected[3], actual[3])
        with self.assertRaisesRegex(ValueError, 'protocol'):
            ContinuousStream(corpus, 1, 8, 7, 11, **options).load_state_dict(saved)

    def test_document_reset_is_only_an_explicit_control(self):
        for marked in (0.,1.):
            options = dict(session_tokens=64, intrusion_tokens=4, intrusion_interval=16, marked_probability=marked)
            persistent = ContinuousStream(fixture(),2,8,7,11,reset_policy='session',**options)
            control = ContinuousStream(fixture(),2,8,7,11,reset_policy='document',**options)
            resets = [0,0]
            for _ in range(20):
                a,b=persistent.next(),control.next()
                self.assertTrue(np.array_equal(a[0],b[0])); self.assertTrue(np.array_equal(a[1],b[1]))
                self.assertTrue(bool((~a[2]|b[2]).all()))
                resets[0] += int(a[2].sum()); resets[1] += int(b[2].sum())
            self.assertGreater(resets[1],resets[0])

    def test_unmarked_independent_intrusion_and_return_seams_have_no_cached_specials(self):
        stream=ContinuousStream(fixture(),2,8,7,11,session_tokens=64,
                                intrusion_tokens=4,intrusion_interval=16,marked_probability=0.)
        masked=0
        for _ in range(60):
            values,mask,reset,info=stream.next()
            self.assertFalse(bool(np.isin(values,[1,2]).any()))
            self.assertFalse(bool(reset[:,1:].any()))
            masked+=int((~mask).sum())
            self.assertTrue(all(span['document']>=0 for lane in info['sources'] for span in lane))
        self.assertGreater(masked,0)
        for kind in ('independent','intrusion','return_to_A'):
            self.assertGreater(stream.switches[kind],0)
        self.assertEqual(stream.switches['marked'],0)

    def test_marked_real_doc_boundaries_preserve_only_real_boundary_supervision(self):
        stream=ContinuousStream(fixture(),2,8,7,11,session_tokens=64,
                                intrusion_tokens=4,intrusion_interval=16,marked_probability=1.)
        eos_labels=bos_inputs=artificial=0
        for _ in range(60):
            values,mask,_,info=stream.next()
            for lane,spans in enumerate(info['sources']):
                records=[None]*9
                for span in spans:
                    for t in range(span['begin'],span['end']):
                        records[t]=dict(document=span['document'],kind=span['kind'])
                        if values[lane,t] in (1,2):
                            self.assertTrue(span['kind'].endswith(('_BOS','_EOS')))
                        if span['document']==-1:
                            self.assertTrue(span['kind'].startswith(('intrusion_','return_to_A_')))
                for t in range(8):
                    left,right=records[t:t+2]
                    if right['kind']=='independent_EOS':
                        self.assertTrue(mask[lane,t]);eos_labels+=1
                    if left['kind']=='independent_BOS':
                        self.assertTrue(mask[lane,t]);bos_inputs+=1
                    if left['document']==-1 or right['document']==-1:
                        self.assertFalse(mask[lane,t]);artificial+=1
                    if left['kind']=='independent_EOS' and right['kind']=='independent_BOS':
                        self.assertFalse(mask[lane,t])
        self.assertGreater(eos_labels,0);self.assertGreater(bos_inputs,0);self.assertGreater(artificial,0)
        self.assertEqual(stream.switches['unmarked'],0)

    def test_content_bounds_trim_only_real_edge_specials_and_exclude_empty_docs(self):
        docs=[np.array([],dtype=np.int64),np.array([1,2]),np.array([8,9,10]),np.array([1,11,12,2])]
        offsets=np.r_[0,np.cumsum([len(doc) for doc in docs])]
        corpus=SimpleNamespace(tokens={'train':np.concatenate(docs)},offsets={'train':offsets})
        stream=ContinuousStream(corpus,1,8,7,11,session_tokens=64,
                                intrusion_tokens=0,intrusion_interval=0,marked_probability=0.)
        self.assertEqual(stream.eligible.tolist(),[2,3])
        self.assertEqual(stream.content_bounds,{2:(0,3),3:(1,3)})
        seen=set()
        for _ in range(4):
            values,_,_,info=stream.next();seen.update(values.flatten().tolist())
            for span in info['sources'][0]:
                lower,upper=stream.content_bounds[span['document']]
                self.assertGreaterEqual(span['offset'],lower)
                self.assertLess(span['offset']+span['end']-span['begin']-1,upper)
        self.assertEqual(seen,{8,9,10,11,12})

    def test_change_proxy_annotations_do_not_change_stream_or_resets(self):
        annotated=fixture(); plain=fixture()
        annotated.change_proxy_spans=lambda split: [dict(document=0,offset=20,cue='fixture-proxy')]
        options=dict(session_tokens=64,intrusion_tokens=4,intrusion_interval=16)
        a=ContinuousStream(annotated,2,8,7,11,**options)
        b=ContinuousStream(plain,2,8,7,11,**options)
        for _ in range(20):
            first,second=a.next(),b.next()
            for left,right in zip(first[:3],second[:3]):
                self.assertTrue(np.array_equal(left,right))
            self.assertEqual(first[3]['content_sha256'],second[3]['content_sha256'])
        self.assertGreater(a.change_proxy_hits,0)
        self.assertEqual(b.change_proxy_hits,0)

    def test_return_uses_unseen_A_continuation_without_replaying_prefix(self):
        stream = ContinuousStream(fixture(),1,8,7,11,session_tokens=64,
                                  intrusion_tokens=4,intrusion_interval=16,marked_probability=.5)
        last_base = None; interrupted = False; returns = 0
        for _ in range(80):
            _,_,_,info=stream.next()
            for span in info['sources'][0]:
                for index in range(span['begin'],min(span['end'],8)):
                    record=(span['document'],span['offset']+index-span['begin'])
                    if span['kind']=='intrusion':
                        interrupted=True
                    elif span['kind']=='base':
                        if interrupted:
                            self.assertEqual(record,(last_base[0],last_base[1]+1))
                            returns+=1; interrupted=False
                        last_base=record
        self.assertGreater(returns,0)

    def test_hidden_loss_mask_does_not_suppress_feedback_or_change_runtime(self):
        cfg=self.config(); values=np.array([[1,4,5,6,7,8,9,10,11]])
        models=[self.model(),self.model()]
        masks=[np.ones((1,8),dtype=bool),np.array([[False,False,True,False,True,True,False,True]])]
        states=[]
        for model,mask in zip(models,masks):
            optimizer=torch.optim.AdamW(model.parameters(),lr=0.)
            _,state=train_update(model,optimizer,values,cfg,torch.device('cpu'),0,loss_mask=mask)
            states.append(state)
        assert_tree(self,states[0],states[1])
        self.assertGreater(float(states[1]['slow'].abs().sum()),0.)

    def test_training_diagnostics_weight_valid_tokens_across_bptt_and_lane_resets(self):
        class DiagnosticModel(nn.Module):
            def __init__(self):
                super().__init__(); self.weight=nn.Parameter(torch.tensor(1.))
            def forward(self,tokens,state=None,reset_mask=None,diagnostic_mask=None):
                state={'seen':torch.zeros(len(tokens))} if state is None else state
                seen=torch.where(reset_mask,torch.zeros_like(state['seen']),state['seen'])+tokens.float().sum(1)
                chosen=tokens.float()[diagnostic_mask]
                diag={} if not chosen.numel() else dict(marker=chosen.mean(),
                        covariance_min_diagonal=chosen.min(),covariance_max_diagonal=chosen.max())
                logits=(self.weight*tokens.float())[...,None].expand(-1,-1,64)
                return logits,{'seen':seen},diag
        cfg=self.config(); values=np.arange(1,19).reshape(2,9)
        mask=np.array([[True,False,True,False,True,True,False,True],
                       [False,True,False,True,True,False,True,True]])
        resets=np.zeros_like(mask); resets[:,0]=True; resets[0,2]=True; resets[1,6]=True
        model=DiagnosticModel(); optimizer=torch.optim.AdamW(model.parameters(),lr=0.)
        metrics,_=train_update(model,optimizer,values,cfg,torch.device('cpu'),0,
                               loss_mask=mask,reset_mask=resets)
        chosen=values[:,:-1][mask]
        self.assertAlmostEqual(metrics['marker'],float(chosen.mean()),places=6)
        self.assertEqual(metrics['covariance_min_diagonal'],float(chosen.min()))
        self.assertEqual(metrics['covariance_max_diagonal'],float(chosen.max()))
        self.assertEqual(metrics['diagnostic_samples'],int(mask.sum()))

    def test_masked_padding_cannot_change_loss_or_parameter_update(self):
        cfg = self.config(); values = np.array([[1, 4, 5, 2, 0, 0, 0, 0, 0]])
        mask = np.array([[True, True, True, False, False, False, False, False]])
        changed = values.copy(); changed[:, 4:] = 61
        a, b = self.model(), self.model()
        optimizers = [torch.optim.AdamW(model.parameters(), lr=.001) for model in (a, b)]
        first, _ = train_update(a, optimizers[0], values, cfg, torch.device('cpu'), 0, loss_mask=mask)
        second, _ = train_update(b, optimizers[1], changed, cfg, torch.device('cpu'), 0, loss_mask=mask)
        self.assertEqual(first['valid_targets'], 3); self.assertEqual(first['padding_targets'], 5)
        self.assertEqual(first['loss'], second['loss'])
        assert_tree(self, a.state_dict(), b.state_dict())

    def test_runtime_and_stream_checkpoint_resume_exact_all_model_families(self):
        cfg = self.config()
        for variant in ('full', 'transformer', 'stateless'):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as directory:
                corpus = fixture(); stream = ContinuousStream(corpus, 2, 8, 7, 11)
                model = self.model(variant); optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
                values, mask, reset, _ = stream.next()
                _, runtime = train_update(model, optimizer, values, cfg, torch.device('cpu'), 0,
                                          reset_mask=reset, loss_mask=mask)
                if variant == 'full':
                    self.assertFalse(bool(runtime['feedback_valid'].any()))
                    self.assertGreater(float(runtime['slow'].abs().sum()), 0.)
                path = Path(directory)/'last.pt'
                checkpoint(path, model, optimizer, {}, 1, float('inf'), 0., torch.device('cpu'), runtime, stream)
                cp = torch.load(path, weights_only=True)
                self.assertIn('runtime_state', cp); self.assertIn('stream', cp)
                restored = self.model(variant); other = torch.optim.AdamW(restored.parameters(), lr=.001)
                restored.load_state_dict(cp['model']); other.load_state_dict(cp['optimizer'])
                recovered = ContinuousStream(corpus, 2, 8, 7, 11); recovered.load_state_dict(cp['stream'])
                restored_runtime = move_state(cp['runtime_state'], torch.device('cpu'))
                restore_rng(cp['rng'], torch.device('cpu'))
                for step in range(1, 5):
                    expected, actual = stream.next(), recovered.next()
                    self.assertEqual(expected[3], actual[3])
                    values, mask, reset, _ = expected
                    metrics, runtime = train_update(model, optimizer, values, cfg, torch.device('cpu'), step,
                                                    state=runtime, reset_mask=reset, loss_mask=mask)
                    values, mask, reset, _ = actual
                    other_metrics, restored_runtime = train_update(restored, other, values, cfg, torch.device('cpu'), step,
                                                                  state=restored_runtime, reset_mask=reset, loss_mask=mask)
                    self.assertEqual(metrics, other_metrics)
                    assert_tree(self, runtime, restored_runtime)
                    assert_tree(self, model.state_dict(), restored.state_dict())
                    assert_tree(self, optimizer.state_dict(), other.state_dict())

    def test_runner_resume_matches_uninterrupted_with_state_and_cursor(self):
        cfg = self.config(); case = cases(cfg)[0]; cpu = torch.device('cpu')
        with tempfile.TemporaryDirectory() as directory:
            continuous, resumed = Path(directory)/'continuous', Path(directory)/'resumed'
            train(cfg, case, fixture(), continuous, cpu, Console(enabled=False), {'requested': False})
            stop = {'requested': False}
            def stop_after_update(*args, **kwargs):
                result = train_update(*args, **kwargs); stop['requested'] = True
                return result
            with patch('v07.run.train_update', side_effect=stop_after_update):
                self.assertIsNone(train(cfg, case, fixture(), resumed, cpu, Console(enabled=False), stop))
            partial = torch.load(resumed/'last.pt', weights_only=True)
            self.assertEqual(partial['step'], 1)
            self.assertIsNotNone(partial['runtime_state'])
            self.assertEqual(partial['stream']['inputs'], 16)
            self.assertGreater(partial['stream']['targets'], 0)
            result = train(cfg, case, fixture(), resumed, cpu, Console(enabled=False), {'requested': False})
            self.assertEqual(result['status'], 'completed')
            first = torch.load(continuous/'last.pt', weights_only=True)
            second = torch.load(resumed/'last.pt', weights_only=True)
            for key in ('model', 'optimizer', 'runtime_state', 'stream', 'rng', 'best'):
                assert_tree(self, first[key], second[key])
            records = [json.loads(line) for line in (resumed/'metrics.jsonl').read_text().splitlines()]
            self.assertEqual([row['step'] for row in records], [1, 2, 3])
            self.assertGreater(max(row['history_age_max'] for row in records), cfg['sequence_tokens'])
            with self.assertRaisesRegex(ValueError, '源码/配置/数据/设备'):
                train(cfg | {'learning_rate': .002}, case, fixture(), resumed, cpu, Console(enabled=False), {'requested': False})

    def test_final_update_stop_resumes_validation_without_an_extra_update(self):
        cfg = self.config(train_steps=1, eval_every=1); stop = {'requested': False}
        def stop_after_update(*args, **kwargs):
            result = train_update(*args, **kwargs); stop['requested'] = True
            return result
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory); case = cases(cfg)[0]
            with patch('v07.run.train_update', side_effect=stop_after_update):
                self.assertIsNone(train(cfg, case, fixture(), folder, torch.device('cpu'), Console(enabled=False), stop))
            self.assertFalse((folder/'best.pt').exists())
            result = train(cfg, case, fixture(), folder, torch.device('cpu'), Console(enabled=False), {'requested': False})
            self.assertEqual(result['status'], 'completed')
            self.assertEqual(len((folder/'metrics.jsonl').read_text().splitlines()), 1)

    def test_failure_before_first_checkpoint_archives_uncommitted_logs_and_best(self):
        cfg = self.config(variants=['stateless'], checkpoint_every=50, eval_every=3)
        calls = {'count': 0}
        def fail_second_update(*args, **kwargs):
            calls['count'] += 1
            if calls['count'] == 2:
                raise RuntimeError('injected failure before first checkpoint')
            return train_update(*args, **kwargs)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory); case = cases(cfg)[0]
            with patch('v07.run.train_update', side_effect=fail_second_update):
                with self.assertRaisesRegex(RuntimeError, 'injected failure'):
                    train(cfg, case, fixture(), folder, torch.device('cpu'), Console(enabled=False), {'requested': False})
            self.assertFalse((folder/'last.pt').exists())
            (folder/'validation.jsonl').write_text('{"step":1,"uncommitted":true}\n')
            save(folder/'best.pt', dict(model={}, step=999, validation_nll=0.))
            result = train(cfg, case, fixture(), folder, torch.device('cpu'), Console(enabled=False), {'requested': False})
            self.assertEqual(result['status'], 'completed')
            rows = [json.loads(line) for line in (folder/'metrics.jsonl').read_text().splitlines()]
            self.assertEqual([row['step'] for row in rows], [1, 2, 3])
            self.assertEqual(len(list(folder.glob('metrics.jsonl.uncommitted-*'))), 1)
            self.assertEqual(len(list(folder.glob('validation.jsonl.uncommitted-*'))), 1)
            self.assertEqual(len(list(folder.glob('best-uncommitted-*'))), 1)

    def test_final_validation_log_is_uncommitted_until_checkpoint_and_best_are_saved(self):
        cfg = self.config(variants=['stateless'], train_steps=1, eval_every=1, checkpoint_every=1)
        stop = {'requested': False}
        def stop_after_update(*args, **kwargs):
            result = train_update(*args, **kwargs); stop['requested'] = True
            return result
        def fail_best_save(path, value):
            if Path(path).name == 'best.pt':
                raise RuntimeError('injected failure after validation append')
            return save(path, value)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory); case = cases(cfg)[0]
            with patch('v07.run.train_update', side_effect=stop_after_update):
                self.assertIsNone(train(cfg, case, fixture(), folder, torch.device('cpu'), Console(enabled=False), stop))
            first = torch.load(folder/'last.pt', weights_only=True)
            self.assertEqual(first['step'], 1); self.assertEqual(first['validation_step'], 0)
            with patch('v07.run.save', side_effect=fail_best_save):
                with self.assertRaisesRegex(RuntimeError, 'injected failure'):
                    train(cfg, case, fixture(), folder, torch.device('cpu'), Console(enabled=False), {'requested': False})
            self.assertTrue((folder/'validation.jsonl').exists())
            self.assertFalse((folder/'best.pt').exists())
            result = train(cfg, case, fixture(), folder, torch.device('cpu'), Console(enabled=False), {'requested': False})
            self.assertEqual(result['status'], 'completed')
            final = torch.load(folder/'last.pt', weights_only=True)
            self.assertEqual(final['step'], 1); self.assertEqual(final['validation_step'], 1)
            for key in ('model', 'optimizer', 'runtime_state', 'stream'):
                assert_tree(self, first[key], final[key])
            self.assertEqual(len((folder/'metrics.jsonl').read_text().splitlines()), 1)
            self.assertEqual(len((folder/'validation.jsonl').read_text().splitlines()), 1)
            self.assertEqual(len(list(folder.glob('validation.jsonl.uncommitted-*'))), 1)

    def test_budget_and_pair_plan(self):
        cfg = load_config(); validate_config(cfg)
        core = cases(cfg)
        self.assertEqual(len(core), 45)
        self.assertEqual([case['variant'] for case in core[:5]], cfg['variants'])
        self.assertEqual([case['nodes'] for case in core[::5]][:3], [32, 128, 512])
        self.assertGreaterEqual(cfg['sequence_tokens'], 128)
        self.assertGreaterEqual(cfg['context_limit'], 2*cfg['prefix_tokens']+cfg['target_tokens'])
        mechanism = cases(cfg | {'stage': 'mechanism'})
        self.assertTrue({'bypass', 'no_local', 'no_slow'} <= {case['variant'] for case in mechanism})
        credit = cases(cfg | {'stage': 'credit'})
        self.assertEqual({(case['variant'], case['bptt_tokens']) for case in credit},
                         {(variant, length) for variant in ('full', 'no_local') for length in (16, 64, 128)})

    @unittest.skipUnless((ROOT/'manifest.json').exists(), 'Frozen v0.6 data unavailable')
    def test_default_data_reuses_frozen_artifacts_without_download_or_fit(self):
        expected = (ROOT/'manifest.json').read_bytes()
        with patch('v07.data.urllib.request.urlopen', side_effect=AssertionError('network forbidden')):
            manifest = prepare()
        self.assertEqual((ROOT/'manifest.json').read_bytes(), expected)
        self.assertEqual(manifest['version'], 'v06-wikitext2-doc-bpe-1')
        corpus=Corpus()
        corpus.tokens={'train':corpus.tokens['train']};corpus.offsets={'train':corpus.offsets['train']}
        proxies=corpus.change_proxy_spans('train')
        self.assertGreater(len(proxies),0)
        for row in proxies:
            doc=row['document'];start=int(corpus.offsets['train'][doc])+row['offset']
            self.assertLessEqual(start+len(row['token_ids']),corpus.offsets['train'][doc+1])
            self.assertEqual(corpus.tokens['train'][start:start+len(row['token_ids'])].tolist(),row['token_ids'])


if __name__ == '__main__':
    unittest.main()
