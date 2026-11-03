"""Pinned official LoCoMo, text-only histories, and separate annotated-evidence controls."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import urllib.request

from v07.common import PROJECT,write,file_hash,digest

REVISION='3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376'
FILES={'locomo10.json':('data/locomo10.json','d95b872480b413d935821fdc3c84f8a8f5f29e73',2805274),
       'evaluation.py':('task_eval/evaluation.py','8f597dd687e66832da1f6f04a169e05622049576',9441),
       'LICENSE.txt':('LICENSE.txt','fe463e0f7888bbf8b82e42d55f5743508ddafb7e',19347)}


def prepare(root,progress=lambda text:None):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);sources={}
    for name,(relative,expected,size) in FILES.items():
        path=root/name;url=f'https://raw.githubusercontent.com/snap-research/locomo/{REVISION}/{relative}'
        if not path.exists():
            temporary=path.with_suffix(path.suffix+'.part')
            with urllib.request.urlopen(url,timeout=60) as response,temporary.open('wb') as handle:
                done=0
                while block:=response.read(128*1024):
                    handle.write(block);done+=len(block);progress(f'下载 {name} {done}/{size} 字节')
            raw=temporary.read_bytes()
            if len(raw)!=size or hashlib.sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest()!=expected:
                raise ValueError(f'Official source hash mismatch: {name}; .part retained')
            temporary.replace(path)
        raw=path.read_bytes()
        if len(raw)!=size or hashlib.sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest()!=expected:
            raise ValueError(f'Official artifact changed: {path}')
        sources[name]=dict(url=url,bytes=size,git_blob_sha1=expected,sha256=file_hash(path))
    samples=json.loads((root/'locomo10.json').read_text())
    validate(samples)
    manifest=dict(revision=REVISION,files=sources,license='CC-BY-NC-4.0',conversations=len(samples),
                  category_counts=dict(Counter(q['category'] for s in samples for q in s['qa'])))
    write(root/'manifest.json',manifest)
    return samples,manifest


def validate(samples):
    if not isinstance(samples,list) or not samples:raise ValueError('Need conversation list')
    seen=set()
    for sample in samples:
        if not isinstance(sample.get('sample_id'),str) or sample['sample_id'] in seen:raise ValueError('Duplicate/bad sample ID')
        seen.add(sample['sample_id']);conv=sample['conversation'];turns=list(ordered_turns(conv))
        ids=[turn['dia_id'] for _,turn in turns]
        if not turns or len(set(ids))!=len(ids):raise ValueError('Missing or duplicate turns')
        for qa in sample['qa']:
            if qa['category'] not in range(1,6) or not isinstance(qa['question'],str) or not qa['question'].strip():
                raise ValueError('Invalid question')
            if qa['category']!=5 and not isinstance(qa.get('answer'),(str,int)):raise ValueError('Missing answer')
            if not isinstance(qa.get('evidence',[]),list):raise ValueError('Bad evidence')


def ordered_turns(conv):
    numbers=sorted(int(k.split('_')[1]) for k,v in conv.items() if k.startswith('session_') and isinstance(v,list))
    for number in numbers:
        date=conv[f'session_{number}_date_time']
        for turn in conv[f'session_{number}']:
            if not all(isinstance(turn.get(k),str) for k in ('speaker','text','dia_id')):raise ValueError('Bad turn')
            yield date,turn


def history_text(sample,evidence=None):
    """Only conversation text/captions; QA, observations and summaries never enter."""
    lines=[];last_date=None;found=set()
    for date,turn in ordered_turns(sample['conversation']):
        if evidence is not None and turn['dia_id'] not in evidence:continue
        if date!=last_date:lines.append(f'Date: {date}');last_date=date
        caption=(' [Image caption: '+turn['blip_caption'].strip()+']') if turn.get('blip_caption') else ''
        lines.append(f'[{turn["dia_id"]}] {turn["speaker"]}: {turn["text"].strip()}{caption}')
        found.add(turn['dia_id'])
    if evidence is not None and set(evidence)!=found:raise ValueError('Evidence references missing turn IDs')
    return '\n'.join(lines).rstrip()


def encode_question(tokenizer,qa):
    prompt='\nQuestion: '+qa['question'].strip()+'\nAnswer:'
    query=tokenizer.encode(prompt).ids
    answer=[]
    if qa['category']!=5:
        target=str(qa['answer'])
        if qa['category']==3:target=target.split(';')[0].strip()
        combined=tokenizer.encode(prompt+' '+target)
        if combined.ids[:len(query)]!=query:raise ValueError('Answer tokenization changes prompt boundary')
        answer=combined.ids[len(query):]
        if not answer or tokenizer.decode(answer).strip()!=target.strip():raise ValueError('Answer alignment failed')
    return query,answer


def encoded_samples(samples,tokenizer,progress=lambda text:None):
    output=[]
    for at,sample in enumerate(samples):
        history=history_text(sample)
        tokens=[1]+tokenizer.encode(history).ids
        cases=[]
        for index,qa in enumerate(sample['qa']):
            query,answer=encode_question(tokenizer,qa)
            evidence=qa.get('evidence',[]);oracle=None;reason=None
            if qa['category']==5:reason='adversarial_no_gold_evidence_control'
            elif not evidence:reason='no_evidence_annotation'
            else:
                try:oracle=[1]+tokenizer.encode(history_text(sample,evidence)).ids+query
                except ValueError:reason='unknown_evidence_id'
            cases.append(dict(index=index,id=f'{sample["sample_id"]}:{index}',qa=qa,query=query,answer_ids=answer,
                oracle=oracle,oracle_reason=reason,question_sha256=digest(qa['question'])))
        # No cross-boundary merge is permitted in this segmented input protocol.
        for case in cases[:1]:
            prompt='\nQuestion: '+case['qa']['question'].strip()+'\nAnswer:'
            if tokens[1:]+case['query']!=tokenizer.encode(history+prompt).ids:
                raise ValueError('History/query token boundary differs from whole-string encoding')
        output.append(dict(id=sample['sample_id'],tokens=tokens,cases=cases))
        progress(f'编码对话 {at+1}/{len(samples)} · {len(tokens)} token · {len(cases)}题')
    return output
