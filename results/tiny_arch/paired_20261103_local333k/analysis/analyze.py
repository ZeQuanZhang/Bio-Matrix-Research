"""Read paired results; optional fixed seed11 free generation after training ends."""
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT.parents[2]
sys.path.insert(0,str(PROJECT))

import numpy as np
import torch
from tokenizers import Tokenizer

from v07.common import file_hash,write
from v1.train import load_corpus
from v11.data import prepare_qa
from v3.train import build_models,validation_data
from scale.engine import qa_score

OLD=PROJECT/'results/v3/paired_20261103T094249_111075'
NAMES=('trans_local','trans_full','v3_static','v3_dynamic','qwen35_tiny','deepseek4_tiny')
LABELS=('TF128','TF512','Bio static','Bio dynamic','Qwen3.5 tiny','DSV4 tiny')
SEEDS=(11,23,37)


def result(name,seed):
    path=(OLD if name in NAMES[:4] else ROOT)/f'{name}_s{seed}'/'result.json'
    row=json.loads(path.read_text());assert row['status']=='completed';return row


def interval(rows,baseline,groups,task):
    unique=sorted(set(groups));table=[];counts=[]
    for seed in SEEDS:
        a=rows[seed]['validation'][task]['conditions']['warm'];b=baseline[seed]['validation'][task]['conditions']['warm']
        target=np.array(a['per_window_targets']);assert list(target)==b['per_window_targets']
        loss=(np.array(a['per_window_nll'])-np.array(b['per_window_nll']))*target
        table.append([loss[np.array(groups)==g].sum() for g in unique]);counts.append([target[np.array(groups)==g].sum() for g in unique])
    table=np.array(table);counts=np.array(counts);rng=np.random.default_rng(31107)
    seeds=rng.integers(0,3,(20000,3));docs=rng.integers(0,len(unique),(20000,len(unique)))
    draws=table[seeds[:,:,None],docs[:,None,:]].sum((1,2))/counts[seeds[:,:,None],docs[:,None,:]].sum((1,2))
    return dict(nll_difference=float(table.sum()/counts.sum()),crossed_seed_document_bootstrap_95ci=np.quantile(draws,[.025,.975]).tolist(),
                repetitions=20000,seeds=3,documents=len(unique))


def summarize():
    registry=json.loads((ROOT/'registry.json').read_text());assert registry['status']=='completed'
    protocol=json.loads((ROOT/'protocol.json').read_text());old_protocol=json.loads((OLD/'protocol.json').read_text())
    assert all(file_hash(PROJECT/p)==sha for p,sha in protocol['sources'].items())
    allrows={n:{s:result(n,s) for s in SEEDS} for n in NAMES};out={}
    paired=0
    for name in NAMES:
        rows=allrows[name];stats=dict(parameters=rows[11]['total_parameters'],seeds=[])
        for seed,row in rows.items():
            lm=row['validation']['lm']['conditions']['warm'];qa=row['validation']['qa']['conditions']['warm']
            stats['seeds'].append(dict(seed=seed,lm_nll=lm['nll'],lm_ppl=lm['ppl'],qa_nll=qa['nll'],
                first_answer_nll=qa['first_answer_nll'],first_answer_accuracy=qa['first_answer_accuracy'],train_seconds=row['train_seconds']))
            if name in NAMES[4:]:
                saved=[json.loads(x) for x in (ROOT/f'{name}_s{seed}'/'metrics.jsonl').read_text().splitlines()]
                reference=[json.loads(x) for x in (OLD/f'trans_full_s{seed}'/'metrics.jsonl').read_text().splitlines()]
                assert len(saved)==len(reference)==1024
                assert [x['sample_sha256'] for x in saved]==[x['sample_sha256'] for x in reference]
                assert row['answer_targets']==reference[-1]['answer_targets']
                assert row['cpt_tokens']==1_048_576 and row['pt_updates']==512
                paired+=1024
        for key in stats['seeds'][0]:
            if key!='seed':stats[key]=float(np.mean([r[key] for r in stats['seeds']]))
        out[name]=stats
    comparisons={}
    groups_qa=protocol['evaluation']['qa']['documents'];groups_lm=protocol['evaluation']['lm']['documents']
    for name in NAMES[4:]:
        comparisons[name+'-minus-bio_dynamic']={task:interval(allrows[name],allrows['v3_dynamic'],groups,task)
            for task,groups in (('qa',groups_qa),('lm',groups_lm))}
    write(ROOT/'analysis/statistics.json',dict(conditions=out,comparisons=comparisons,
        scoring='Same original warm-only 8192 LM and 475 answer targets; three matched seeds. Teacher-forced diagnostics.',
        first_token_accuracy='Original first-answer mask, same 64 questions.',
        timing='Core forward/backward/clip/optimizer; public reference kernels on MPS, historical runs at another wall-clock time. No flagship speed claims.'))
    write(ROOT/'analysis/audit.json',dict(status='completed',new_conditions=6,paired_update_hashes=paired,
        original_model_sources_unchanged=58,all_sources_unchanged=len(protocol['sources']),
        original_checkpoints_unchanged='Not modified',training_data='Same frozen WikiText2 and SQuAD preparations',
        pretrained_weights_used=False,large_scale_queue='unchanged, not_executed'))
    plot(out)
    print(json.dumps(dict(conditions=out,comparisons=comparisons),ensure_ascii=False))


def plot(stats):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4.8),layout='constrained')
    colors=['#989fa4','#697680','#589389','#2d756a','#c78b36','#b45a51']
    for ax,key,title in zip(axes,('lm_ppl','qa_nll','train_seconds'),('Text perplexity (lower is better)','Answer NLL (lower is better)','Core training seconds / seed')):
        means=[stats[n][key] for n in NAMES];std=[np.std([s[key] for s in stats[n]['seeds']],ddof=1) for n in NAMES]
        ax.bar(range(6),means,color=colors,yerr=std,capsize=3)
        ax.set_xticks(range(6),LABELS,rotation=45,ha='right');ax.set_title(title);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Local 333K prototypes | same 1,048,576 text targets + 512 QA updates | 3 seeds, error bars: seed SD')
    fig.savefig(ROOT/'analysis/comparison.png',dpi=170);fig.savefig(ROOT/'analysis/comparison.svg');plt.close(fig)


def generation():
    from transformers.utils import logging
    from evaluations.tiny_arch.model import Model
    logging.set_verbosity_error();torch.set_num_threads(4);device=torch.device('mps')
    registry=json.loads((ROOT/'registry.json').read_text());assert registry['status']=='completed'
    old_protocol=json.loads((OLD/'protocol.json').read_text());c=old_protocol['training']
    corpus=load_corpus(c['data_root']);qa=prepare_qa(c['qa_root'],corpus,length=512,prefix=384,context_limit=128,backbone_layers=2,allow_download=False)
    (_,values,mask,first),meta=validation_data(corpus,qa,c);refs=meta['qa']['rows']
    tokenizer=Tokenizer.from_file(str(Path(c['data_root'])/'tokenizer.json'));summary={}
    for name in NAMES:
        destination=ROOT/'analysis'/f'generation_{name}.private.json'
        if destination.exists():rows=json.loads(destination.read_text())
        else:
            if name in NAMES[:4]:model=build_models(4096,c,11)[0][name]
            else:model=Model(name,11,c)
            path=(OLD if name in NAMES[:4] else ROOT)/f'{name}_s11/last.pt';before=file_hash(path)
            payload=torch.load(path,map_location='cpu',weights_only=True);model.load_state_dict(payload['model']);del payload
            model.to(device).eval();rows=[];buckets=defaultdict(list)
            for i,row in enumerate(refs):buckets[row['answer_target_start']+1].append(i)
            with torch.inference_mode():
                for length,indices in sorted(buckets.items()):
                    allowed=min(50,513-length)
                    assert allowed>0
                    for start in range(0,len(indices),8):
                        ids=indices[start:start+8];tokens=torch.as_tensor(values[ids,:length].copy(),device=device)
                        logits,state,_=model(tokens);generated=[[] for _ in ids];finished=[False]*len(ids)
                        for step in range(allowed):
                            output=logits[:,-1].argmax(-1).cpu().tolist()
                            for j,token in enumerate(output):
                                if not finished[j]:generated[j].append(token)
                                if token==2:finished[j]=True
                            if all(finished) or step==allowed-1:break
                            feed=torch.tensor([[2 if done else token] for token,done in zip(output,finished)],device=device)
                            logits,state,_=model(feed,state)
                        for i,g in zip(ids,generated):
                            prediction=tokenizer.decode(g,skip_special_tokens=True).strip();references=refs[i]['reference_answers']
                            em,f1=qa_score(prediction,references)
                            rows.append(dict(index=i,question_id=refs[i]['question_id'],article_id=refs[i]['article_id'],
                                prediction=prediction,reference_answers=references,generated_ids=g,em=em,f1=f1,
                                generated_tokens=len(g),allowed_new_tokens=allowed,hit_limit=len(g)==allowed and g[-1]!=2))
            assert before==file_hash(path);write(destination,sorted(rows,key=lambda r:r['index']))
            model.to('cpu');del model;torch.mps.empty_cache()
        assert len(rows)==64
        common=Counter(r['prediction'] for r in rows).most_common(1)[0]
        summary[name]=dict(questions=64,em=float(np.mean([r['em'] for r in rows])),f1=float(np.mean([r['f1'] for r in rows])),
            distinct_predictions=len({r['prediction'] for r in rows}),token_limit_fraction=float(np.mean([r['hit_limit'] for r in rows])),
            most_common_prediction_count=common[1],most_common_prediction_sha256=hashlib.sha256(common[0].encode()).hexdigest(),
            most_common_prediction_words=Counter(common[0].split()).most_common(3))
        write(ROOT/'analysis/generation_summary.json',dict(seed=11,max_new_tokens=50,input_context_limit=512,
            decode='greedy, EOS stop; per-question allowance min(50, 513-prefix_length), identical for all six models',conditions=summary))
        print('generation',name,summary[name],flush=True)


if __name__=='__main__':
    if '--generation' in sys.argv:generation()
    else:summarize()
