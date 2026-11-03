"""Causality, decomposition, prefix reference, local credit and bounded vector state."""
import copy
from dataclasses import replace
import unittest

import torch
from torch.nn import functional as F

from v07.common import state_bytes
from .model import Config,Network,VectorNodes,unit


def config(**changes):
    return Config(**(dict(vocab_size=32,width=16,heads=2,layers=2,ff=32,nodes=4,
                         phase_bands=3,segment_tokens=4,context_limit=4)|changes))


def tree_close(a,b,atol=2e-11):
    if isinstance(a,dict):
        assert a.keys()==b.keys()
        for name in a: tree_close(a[name],b[name],atol)
    elif isinstance(a,list):
        assert len(a)==len(b)
        for x,y in zip(a,b):tree_close(x,y,atol)
    else:torch.testing.assert_close(a,b,atol=atol,rtol=1e-9)


class ModelChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def setUp(self):torch.manual_seed(81)

    def test_distributed_read_matches_explicit_rank_one_matrices(self):
        q=torch.randn(2,7,8,dtype=torch.double)
        k,v=torch.randn(5,8,dtype=torch.double),torch.randn(5,8,dtype=torch.double)
        g=torch.randn(2,7,5,dtype=torch.double).softmax(-1)
        vector=((q@k.T)*g)@v
        matrices=v[:,:,None]*k[:,None,:]
        dense=torch.einsum('btn,nde,bte->btd',g,matrices,q)
        torch.testing.assert_close(vector,dense,atol=1e-12,rtol=1e-12)

    def test_dual_read_matches_explicit_serial_vector_writes_and_gradients(self):
        module=VectorNodes(config()).double();other=copy.deepcopy(module)
        hidden=torch.randn(2,4,16,dtype=torch.double,requires_grad=True)
        source=hidden.detach().clone().requires_grad_()
        state=module.initial(2);expected=other.initial(2)
        actual,end,_=module(hidden,state,state)
        x=other.norm(source);qkv=other.qkv(x).reshape(2,4,3,2,8)
        q,k=unit(qkv[:,:,:2]).unbind(2);v=qkv[:,:,2].tanh()
        qp,kp=other.phase(torch.cat((q,k),1)).chunk(2,1)
        terms=other.write_terms(k,v,kp,expected)
        value=expected['value'];outputs=[]
        alpha=torch.exp(-1/other.log_tau.exp())[None,:,None,None]
        rate=.25*other.raw_write_rate.sigmoid()[None,:,None,None]
        for at in range(4):
            view={name:z[:,None] for name,z in expected.items()}
            view['value']=value[:,None]
            prediction=other.route(q[:,at:at+1],qp[:,at:at+1],view)[0]
            gate=other.output_gate(x[:,at:at+1]).reshape_as(prediction).sigmoid()
            outputs.append(source[:,at:at+1]+other.output((prediction*gate).flatten(-2)))
            delta=terms['value_factor'][:,at,...,None]*terms['error'][:,at,:,None,:]
            value=alpha*value+(1-alpha)*expected['value']+rate*delta
        reference=torch.cat(outputs,1)
        torch.testing.assert_close(actual,reference,atol=1e-12,rtol=1e-11)
        torch.testing.assert_close(end['value'],value,atol=1e-12,rtol=1e-11)
        actual.square().mean().backward();reference.square().mean().backward()
        torch.testing.assert_close(hidden.grad,source.grad,atol=1e-12,rtol=1e-10)
        for (name,p),(othername,r) in zip(module.named_parameters(),other.named_parameters()):
            self.assertEqual(name,othername)
            if p.grad is not None:torch.testing.assert_close(p.grad,r.grad,atol=1e-12,rtol=1e-9,msg=name)

    def test_future_tokens_do_not_change_earlier_predictions(self):
        model=Network(config()).double()
        a=torch.randint(0,32,(2,17));b=a.clone();b[:,6:]=torch.randint(0,32,(2,11))
        torch.testing.assert_close(model(a)[0][:,:6],model(b)[0][:,:6],atol=1e-12,rtol=1e-12)

    def test_arbitrary_splits_and_single_tokens_match_batched_training(self):
        model=Network(config()).double()
        x=torch.randint(0,32,(2,19));expected,state,_=model(x)
        for lengths in ((1,)*19,(3,5,1,7,3)):
            parts=[];other=None;offset=0
            for length in lengths:
                y,other,_=model(x[:,offset:offset+length],other);parts.append(y);offset+=length
            torch.testing.assert_close(expected,torch.cat(parts,1),atol=2e-11,rtol=1e-9)
            tree_close(state,other)

    def test_batch_backward_matches_token_reference(self):
        a=Network(config()).double();b=copy.deepcopy(a);x=torch.randint(0,32,(2,11))
        y=a(x)[0];parts=[];s=None
        for at in range(x.shape[1]):
            out,s,_=b(x[:,at:at+1],s);parts.append(out)
        y.square().mean().backward();torch.cat(parts,1).square().mean().backward()
        for (name,p),(other,q) in zip(a.named_parameters(),b.named_parameters()):
            self.assertEqual(name,other)
            torch.testing.assert_close(p.grad,q.grad,atol=3e-11,rtol=1e-8,msg=name)

    def test_error_directions_match_joint_loss_gradient(self):
        module=VectorNodes(config()).double()
        state={name:x.detach().requires_grad_() for name,x in module.initial(2).items()}
        key=unit(torch.randn(2,1,2,8,dtype=torch.double));value=torch.randn_like(key)*.1
        phase=module.phase(key)
        prediction=module.route(key,phase,{name:x[:,None] for name,x in state.items()})[0]
        grads=torch.autograd.grad(.5*(prediction-value).square().sum(),list(state.values()))
        terms=module.write_terms(key,value,phase,state)
        gradients=module.joint_gradients(terms,state,torch.ones(1,1,2,1,dtype=torch.double))
        for (name,_),gradient in zip(state.items(),grads):
            torch.testing.assert_close(gradients[name],-gradient,atol=1e-12,rtol=1e-10,msg=name)

    def test_same_block_history_changes_later_outputs_and_has_credit(self):
        model=Network(config()).double()
        a=torch.tensor([[1,2,3]]);b=torch.tensor([[9,2,3]])
        self.assertGreater(float((model(a)[0][:,1:]-model(b)[0][:,1:]).detach().abs().max()),1e-8)
        hidden=model.embedding(a);hidden.retain_grad()
        nodes=model.layers[0].nodes;initial=nodes.initial(1)
        out,_,_=nodes(hidden,initial,initial)
        out[:,-1].square().sum().backward()
        self.assertGreater(float(hidden.grad[:,0].abs().sum()),0)

    def test_suffix_loss_reaches_earlier_vectors_and_phase_updates(self):
        model=Network(config()).double();_,state,_=model(torch.randint(0,32,(2,4)))
        for x in state['memory'][0].values():x.retain_grad()
        logits,_,_=model(torch.randint(0,32,(2,5)),state)
        F.cross_entropy(logits[:,-1],torch.tensor([7,11])).backward()
        for x in state['memory'][0].values():self.assertGreater(float(x.grad.abs().sum()),0)
        nodes=model.layers[0].nodes
        for name in ('qkv.weight','phase_projection','anchor_angles','raw_address_rate','raw_write_rate'):
            grad=dict(nodes.named_parameters())[name].grad
            self.assertIsNotNone(grad);self.assertTrue(bool(grad.isfinite().all()))
            self.assertGreater(float(grad.abs().sum()),0,msg=name)

    def test_lane_reset_and_mixed_boundaries(self):
        model=Network(config()).double();before=torch.randint(0,32,(2,6));after=torch.randint(0,32,(2,7))
        _,state,_=model(before)
        result=model(after,state,reset_mask=torch.tensor([True,False]))[0]
        torch.testing.assert_close(result[:1],model(after[:1])[0],atol=2e-11,rtol=1e-9)
        continuous=model(torch.cat((before[1:],after[1:]),1))[0][:,6:]
        torch.testing.assert_close(result[1:],continuous,atol=2e-11,rtol=1e-9)

    def test_node_permutation_preserves_output(self):
        model=Network(config()).double();other=copy.deepcopy(model);perm=torch.tensor([2,0,3,1])
        with torch.no_grad():
            for layer in other.layers:
                for name in ('key_anchor','value_anchor','anchor_angles'):
                    p=getattr(layer.nodes,name);p.copy_(p[:,perm].clone())
        x=torch.randint(0,32,(2,13))
        torch.testing.assert_close(model(x)[0],other(x)[0],atol=1e-12,rtol=1e-10)

    def test_static_freezes_keys_and_phase_but_writes_values(self):
        model=Network(config(dynamic_addresses=False)).double();initial=model.initial_state(2)
        _,state,_=model(torch.randint(0,32,(2,20)))
        for old,new in zip(initial['memory'],state['memory']):
            for name in ('key','phase'):torch.testing.assert_close(old[name],new[name],atol=2e-15,rtol=1e-15)
            self.assertGreater(float((old['value']-new['value']).detach().abs().sum()),0)

    def test_fixed_capacity_bounds_and_absence_of_internal_matrices_or_attention(self):
        model=Network(config()).eval()
        with torch.no_grad():
            _,a,_=model(torch.randint(0,32,(2,16)))
            _,b,diag=model(torch.randint(0,32,(2,128)))
        self.assertEqual(state_bytes(a),state_bytes(b));self.assertLess(float(diag['phase_error']),1e-6)
        self.assertNotIn('kv',b)
        for group in ('memory','pending'):
            for row in b[group]:
                self.assertEqual(set(row),{'key','value','phase'})
                self.assertTrue(all(x.ndim==4 for x in row.values()))
                self.assertLessEqual(float(torch.linalg.vector_norm(row['value'],dim=-1).max()),4.00001)
        self.assertFalse(any('attn' in name or 'attention' in name for name,_ in model.named_modules()))

    def test_forward_does_not_mutate_state_or_parameters_and_read_off_is_pointwise(self):
        model=Network(config()).double();state=model.initial_state(2)
        saved={name:([{k:v.detach().clone() for k,v in row.items()} for row in value]
                     if isinstance(value,list) else value.clone()) for name,value in state.items()}
        weights=copy.deepcopy(model.state_dict());a=torch.randint(0,32,(2,11));b=a.clone();b[:,:6]=1
        model(a,state);tree_close(state,saved,atol=0)
        for name,value in weights.items():torch.testing.assert_close(value,model.state_dict()[name],atol=0,rtol=0)
        left=model(a,memory_read=False,memory_write=False)[0]
        right=model(b,memory_read=False,memory_write=False)[0]
        torch.testing.assert_close(left[:,6:],right[:,6:],atol=1e-12,rtol=1e-12)

    def test_invalid_config_and_state_rejected(self):
        for args in (dict(nodes=0),dict(width=15),dict(dynamic_addresses=1),dict(segment_tokens=4096)):
            with self.assertRaises(ValueError):config(**args)
        model=Network(config());state=model.initial_state(2);state['memory'][0]['key']=state['memory'][0]['key'][:,:,:1]
        with self.assertRaises(ValueError):model(torch.ones(2,1,dtype=torch.long),state)


if __name__=='__main__':unittest.main()
