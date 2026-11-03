"""Causal block contracts, reference statistics, gradients and bounded node updates."""
import copy
from dataclasses import replace
import unittest

import torch
from torch.nn import functional as F

from .model import Config, Network, PhaseMemory


def config(**changes):
    return Config(vocab_size=32,width=16,heads=2,layers=2,ff=32,segment_tokens=4,
                  context_limit=8,nodes=4,phase_bands=3,**changes)


def tree_close(a,b,atol=3e-6):
    if isinstance(a,dict):
        assert a.keys()==b.keys()
        for name in a:tree_close(a[name],b[name],atol)
    elif isinstance(a,(tuple,list)):
        for x,y in zip(a,b):tree_close(x,y,atol)
    elif a is None:assert b is None
    else:torch.testing.assert_close(a,b,atol=atol,rtol=3e-5)


class ModelChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def setUp(self):torch.manual_seed(73)

    def test_future_tokens_cannot_change_earlier_predictions(self):
        model=Network(config()).eval()
        x=torch.randint(0,32,(2,13));y=x.clone();y[:,6:]=torch.randint(0,32,(2,7))
        a=model(x)[0];b=model(y)[0]
        torch.testing.assert_close(a[:,:6],b[:,:6],atol=1e-7,rtol=1e-6)

    def test_arbitrary_python_splits_preserve_blocks_and_state(self):
        model=Network(config()).double().eval();x=torch.randint(0,32,(2,19))
        a,state,_=model(x);parts=[];other=None
        for start,end in ((0,1),(1,3),(3,8),(8,9),(9,16),(16,19)):
            logits,other,_=model(x[:,start:end],other);parts.append(logits)
        torch.testing.assert_close(a,torch.cat(parts,1),atol=2e-12,rtol=2e-10)
        tree_close(state,other,atol=2e-12)

    def test_lane_reset_and_mixed_block_phases(self):
        model=Network(config()).eval()
        before=torch.randint(0,32,(2,6));after=torch.randint(0,32,(2,7))
        _,state,_=model(before)
        got,_,_=model(after,state,reset_mask=torch.tensor([True,False]))
        fresh=model(after[:1])[0]
        continuous=model(torch.cat((before[1:],after[1:]),1))[0][:,6:]
        torch.testing.assert_close(got[:1],fresh,atol=2e-6,rtol=2e-5)
        torch.testing.assert_close(got[1:],continuous,atol=2e-6,rtol=2e-5)

    def test_suffix_loss_reaches_previous_content_and_address_updates(self):
        model=Network(config())
        _,state,_=model(torch.randint(0,32,(2,4)))
        content=state['memory'][0]['content'];content.retain_grad()
        logits,_,_=model(torch.randint(0,32,(2,8)),state)
        F.cross_entropy(logits[:,-1],torch.tensor([5,9])).backward()
        self.assertGreater(float(content.grad.abs().sum()),0)
        for memory in model.memories:
            for p in (memory.address.weight,memory.phase_projection,memory.anchor_angles,memory.raw_address_rate,
                      memory.raw_write_rate,memory.output.weight):
                self.assertIsNotNone(p.grad);self.assertTrue(bool(p.grad.isfinite().all()))
                self.assertGreater(float(p.grad.abs().sum()),0)

    def test_batch_statistics_match_explicit_scalar_reference(self):
        cfg=config();module=PhaseMemory(cfg).double()
        b,t,h,n,d=2,3,cfg.heads,cfg.nodes,cfg.width//cfg.heads
        key=F.normalize(torch.randn(b,t,h,d,dtype=torch.double),dim=-1)
        value=torch.randn_like(key);gate=torch.randn(b,t,h,n,dtype=torch.double).softmax(-1)
        phase=torch.randn(b,t,h,cfg.phase_bands,2,dtype=torch.double)
        controls=dict(key=key,value=value,error=value,gates=gate,phase=phase)
        actual=module.accumulate(module.empty(b),controls)
        expected=module.empty(b)
        for at in range(t):
            for node in range(n):
                weight=gate[:,at,:,node]
                expected['cross'][:,node]+=weight[...,None,None]*value[:,at,:, :,None]*key[:,at,:,None,:]
                expected['gram'][:,node]+=weight[...,None,None]*key[:,at,:, :,None]*key[:,at,:,None,:]
                expected['phase'][:,node]+=weight[...,None,None]*phase[:,at]
                expected['mass'][:,node]+=weight
            expected['count']+=1
        tree_close(actual,expected,atol=1e-12)

    def test_node_permutation_preserves_joint_output(self):
        model=Network(config()).double().eval();other=copy.deepcopy(model);perm=torch.tensor([2,0,3,1])
        with torch.no_grad():
            for memory in other.memories:
                memory.anchor_angles.copy_(memory.anchor_angles[perm].clone())
                memory.node_shape.copy_(memory.node_shape[perm].clone())
        x=torch.randint(0,32,(2,16))
        torch.testing.assert_close(model(x)[0],other(x)[0],atol=2e-12,rtol=1e-10)

    def test_content_and_addresses_commit_once_at_boundary(self):
        model=Network(config()).eval();initial=model.initial_state(2)
        x=torch.randint(0,32,(2,4));_,s,_=model(x[:,:3],initial)
        tree_close(s['memory'],initial['memory'],atol=0)
        _,s,_=model(x[:,3:],s)
        self.assertEqual(s['memory'][0]['blocks'].tolist(),[1,1])
        self.assertGreater(float(s['memory'][0]['content'].detach().abs().sum()),0)
        self.assertGreater(float((s['memory'][0]['addresses']-initial['memory'][0]['addresses']).detach().abs().sum()),0)

    def test_static_address_control_preserves_runtime_layout(self):
        cfg=replace(config(),dynamic_addresses=False);model=Network(cfg)
        initial=model.initial_state(2);_,s,_=model(torch.randint(0,32,(2,20)),initial)
        for old,new in zip(initial['memory'],s['memory']):
            torch.testing.assert_close(old['addresses'],new['addresses'],atol=0,rtol=0)
            self.assertGreater(float(new['content'].detach().abs().sum()),0)

    def test_empty_commit_and_forward_never_modify_shared_state_or_parameters(self):
        model=Network(config());state=model.initial_state(2)
        saved=copy.deepcopy({n:p.detach() for n,p in model.state_dict().items()})
        with torch.no_grad():clone=model.initial_state(2)
        for old,module in zip(state['memory'],model.memories):
            tree_close(old,module.commit(module.empty(2),old,torch.ones(2,dtype=torch.bool)),atol=0)
        # Independent initial state has identical values; forward uses functional updates.
        model(torch.randint(0,32,(2,12)),state)
        tree_close(state,clone,atol=0)
        for n,p in model.state_dict().items():torch.testing.assert_close(p,saved[n],atol=0,rtol=0)

    def test_covariance_phase_bounds_and_fixed_state_capacity(self):
        from v07.common import state_bytes
        model=Network(config()).eval()
        with torch.no_grad():
            _,a,_=model(torch.randint(0,32,(2,16)))
            _,b,diag=model(torch.randint(0,32,(2,128)))
        self.assertEqual(state_bytes(a),state_bytes(b))
        self.assertLess(float(diag['phase_error']),1e-6)
        for state in b['memory']:
            self.assertGreater(float(torch.linalg.eigvalsh(state['free']).min()),-1e-6)
            self.assertLessEqual(float(torch.linalg.vector_norm(state['content'],dim=(-1,-2)).max()),4.00001)

    def test_batch_write_reduces_observed_associative_error(self):
        module=PhaseMemory(config());old=module.initial_state(2)
        controls=module.controls(torch.randn(2,4,16),old)
        pending=module.accumulate(module.empty(2),controls)
        new=module.commit(pending,old,torch.ones(2,dtype=torch.bool))
        def loss(state):
            predictions=torch.einsum('bnhde,bthe->btnhd',state['content'],controls['key'])
            errors=(predictions-controls['value'][:,:,None]).square().sum(-1)
            return (errors*controls['gates'].transpose(2,3)).sum()
        self.assertLess(float(loss(new).detach()),float(loss(old).detach()))

    def test_transformer_single_pass_matches_streaming_and_read_off(self):
        model=Network(replace(config(),memory_enabled=False)).double().eval()
        x=torch.randint(0,32,(2,19));a=model(x)[0];parts=[];state=None
        for at in range(0,19,3):
            v,state,_=model(x[:,at:at+3],state);parts.append(v)
        torch.testing.assert_close(a,torch.cat(parts,1),atol=1e-12,rtol=1e-10)
        memory=Network(config()).double();memory.load_state_dict(model.state_dict(),strict=False)
        off=memory(x,memory_read=False,memory_write=False)[0]
        torch.testing.assert_close(off,a,atol=1e-12,rtol=1e-10)

    def test_state_and_config_reject_malformed_values(self):
        with self.assertRaises(ValueError):Config(width=15,heads=2)
        model=Network(config());s=model.initial_state(2)
        s['pending'][0]['gram']=s['pending'][0]['gram'][:,:1]
        with self.assertRaises(ValueError):model(torch.ones(2,1,dtype=torch.long),s)

    def test_cooperative_write_statistics_match_joint_prediction_gradient(self):
        module=PhaseMemory(config()).double()
        state=module.initial_state(2)
        content=torch.randn_like(state['content'],requires_grad=True)
        state=state | {'content':content}
        controls=module.controls(torch.randn(2,4,16,dtype=torch.double),state)
        g=controls['gates'].detach();k=controls['key'].detach();v=controls['value'].detach()
        predictions=torch.einsum('bthn,bnhde,bthe->bthd',g,content,k)
        gradient=torch.autograd.grad(.5*(predictions-v).square().sum(),content)[0]
        statistics=module.accumulate(module.empty(2),controls)
        torch.testing.assert_close(statistics['cross'],-gradient,atol=1e-12,rtol=1e-10)

    def test_two_queries_select_distinct_overlapping_phase_groups(self):
        module=PhaseMemory(replace(config(),nodes=5))
        x=torch.randn(1,1,16)
        a=module.controls(x,module.initial_state(1))['phase'][0,0]
        b=module.controls(-x,module.initial_state(1))['phase'][0,0]
        middle=F.normalize(a+b,dim=-1)
        phases=torch.stack((a,middle,b,a,b))
        with torch.no_grad():module.anchor_angles.copy_(torch.atan2(phases[...,1],phases[...,0]))
        state=module.initial_state(1)
        left=module.controls(x,state)['gates'][0,0].topk(3,-1).indices
        right=module.controls(-x,state)['gates'][0,0].topk(3,-1).indices
        for head in range(module.cfg.heads):
            self.assertEqual(set(left[head].tolist()),{0,1,3})
            self.assertEqual(set(right[head].tolist()),{1,2,4})


if __name__=='__main__':unittest.main()
