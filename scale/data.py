"""Pinned public shards, bounded downloads, exact token packs and isolated SQuAD."""
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import urllib.request

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer

from v07.common import digest,file_hash,write
from v11.data import REVISION as SQUAD_REVISION,SOURCES as SQUAD_SOURCES,article_id,article_split

REPO='HuggingFaceFW/fineweb-edu'
REVISION='87f09149ef4734204d70ed1d046ddc9ca3f2b8f9'
TOKENIZER_REPO='openai-community/gpt2'
TOKENIZER_REVISION='607a30d783dfa663caf39e06633721c8d4cfcd7e'
TOKENIZER_SHA256='8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6'
EOS=50256
RESERVE_BYTES=16*2**30


def get_json(url):
    with urllib.request.urlopen(url,timeout=30) as response:return json.load(response)


def download(url,target,expected_sha=None,expected_size=None,stop=None):
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        if expected_size is not None and target.stat().st_size!=expected_size:raise ValueError(f'Existing size mismatch: {target.name}')
        if expected_sha is not None and file_hash(target)!=expected_sha:raise ValueError(f'Existing hash mismatch: {target.name}')
        return
    part=target.with_suffix(target.suffix+'.part');offset=part.stat().st_size if part.exists() else 0
    if expected_size is not None and offset==expected_size:
        if expected_sha is not None and file_hash(part)!=expected_sha:raise ValueError('Complete partial file has a bad hash')
        part.replace(target);return
    if expected_size is not None and offset>expected_size:raise ValueError('Partial file exceeds pinned size')
    request=urllib.request.Request(url,headers={'Range':f'bytes={offset}-'} if offset else {})
    with urllib.request.urlopen(request,timeout=45) as response:
        append=offset>0 and response.status==206
        if append and not response.headers.get('Content-Range','').startswith(f'bytes {offset}-'):raise ValueError('Invalid resumed range')
        with part.open('ab' if append else 'wb') as handle:
            while block:=response.read(1024*1024):
                if stop is not None and stop.requested:raise KeyboardInterrupt
                handle.write(block)
                if expected_size is not None and handle.tell()>expected_size:raise ValueError('Download exceeds pinned length')
    if expected_size is not None and part.stat().st_size!=expected_size:raise ValueError(f'Download length mismatch: {target.name}')
    if expected_sha is not None and file_hash(part)!=expected_sha:raise ValueError(f'Download checksum mismatch: {target.name}')
    part.replace(target)


def sources(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);path=root/'sources.json'
    if path.exists():
        saved=json.loads(path.read_text())
        if saved['dataset_revision']!=REVISION or saved['tokenizer_revision']!=TOKENIZER_REVISION:raise ValueError('Source revision changed')
        return saved
    url=f'https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}/sample/100BT?limit=1000'
    with urllib.request.urlopen(url,timeout=30) as response:
        rows=json.load(response)
        if response.headers.get('Link'):raise ValueError('Unexpected paginated shard inventory; do not silently truncate')
    shards=[]
    for row in rows:
        if row['type']!='file' or not row['path'].endswith('.parquet'):continue
        if not row['path'].startswith('sample/100BT/') or '..' in Path(row['path']).parts:raise ValueError('Untrusted shard path')
        sha=row.get('lfs',{}).get('oid')
        if not sha or len(sha)!=64:raise ValueError('Missing pinned LFS SHA256')
        shards.append(dict(path=row['path'],bytes=row['size'],sha256=sha))
    if not shards:raise ValueError('Empty source inventory')
    download(f'https://huggingface.co/{TOKENIZER_REPO}/resolve/{TOKENIZER_REVISION}/tokenizer.json',root/'tokenizer.json',TOKENIZER_SHA256)
    tokenizer=Tokenizer.from_file(str(root/'tokenizer.json'))
    if tokenizer.get_vocab_size()!=50257 or tokenizer.token_to_id('<|endoftext|>')!=EOS:raise ValueError('Unexpected GPT-2 tokenizer')
    saved=dict(dataset=REPO,dataset_revision=REVISION,configuration='sample-100BT',license='ODC-By-1.0',
        tokenizer=TOKENIZER_REPO,tokenizer_revision=TOKENIZER_REVISION,tokenizer_sha256=file_hash(root/'tokenizer.json'),
        official_token_count='approximately 100B GPT-2 tokens',shards=sorted(shards,key=lambda x:x['path']))
    write(path,saved);return saved


def document_split(text):
    number=int(hashlib.sha256(text.encode()).hexdigest()[:16],16)%1000
    return 'validation' if number==0 else 'test' if number==1 else 'train'


def pack_shard(source,out,index,tokenizer,remaining,holdout_remaining,stop=None):
    names=('train','validation','test');paths={k:out/f'{k}_{index:03d}.bin' for k in names}
    handles={k:p.with_suffix('.bin.part').open('wb') for k,p in paths.items()}
    counts={k:0 for k in names};docs={k:0 for k in names};scanned=0
    try:
        for batch in pq.ParquetFile(source).iter_batches(batch_size=32,columns=['text']):
            if stop is not None and stop.requested:raise KeyboardInterrupt
            texts=[str(x or '') for x in batch.column(0).to_pylist()]
            encoded=tokenizer.encode_batch(texts,add_special_tokens=False)
            for text,tokens in zip(texts,encoded):
                scanned+=1
                if not text.strip():continue
                split=document_split(text)
                limit=remaining if split=='train' else holdout_remaining[split]
                keep=max(0,limit-counts[split]);ids=(tokens.ids+[EOS])[:keep]
                if ids:
                    if max(ids)>=65536:raise ValueError('Tokenizer does not fit uint16')
                    handles[split].write(np.asarray(ids,dtype='<u2').tobytes());counts[split]+=len(ids);docs[split]+=1
                if counts['train']>=remaining:break
            if counts['train']>=remaining:break
    finally:
        for handle in handles.values():handle.close()
    files=[]
    for split,path in paths.items():
        part=path.with_suffix('.bin.part')
        if counts[split]:
            part.replace(path);files.append(dict(path=path.name,split=split,tokens=counts[split],sha256=file_hash(path)))
        else:part.unlink()
    return dict(files=files,counts=counts,documents=docs,source_rows_scanned=scanned)


def prepare(root,tokens,sequence,stage=lambda s:None,stop=None):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    # Extra blocks cover partial updates at the three exact milestone boundaries.
    required=tokens+1_048_576;settings=dict(tokens=tokens,required_tokens=required,sequence=sequence,order_seed=31001,holdout_cap=2_097_152)
    manifest_path=root/'manifest.json'
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        if manifest['preparation']!=settings:raise ValueError('Preparation budget changed; choose a new data root')
        if manifest['status']=='completed':return validate(root)
    else:manifest=dict(status='preparing',preparation=settings,files=[],completed_shards=[],totals=dict(train=0,validation=0,test=0))
    remaining=max(0,required-manifest['totals']['train'])
    needed=remaining*2+RESERVE_BYTES+3*2**30
    free=shutil.disk_usage(root).free
    if free<needed:raise OSError(f'数据准备需要约 {needed/2**30:.1f} GiB 可用空间，当前 {free/2**30:.1f} GiB；请在GPU机器的数据盘准备。')
    stage('核对固定公开语料与分词器');source=sources(root)
    if file_hash(root/'tokenizer.json')!=source['tokenizer_sha256']:raise ValueError('Tokenizer hash mismatch')
    manifest['source_manifest_sha256']=digest(source);write(manifest_path,manifest)
    tokenizer=Tokenizer.from_file(str(root/'tokenizer.json'));shards=list(source['shards']);random.Random(settings['order_seed']).shuffle(shards)
    cache=root/'download';cache.mkdir(exist_ok=True)
    for index,shard in enumerate(shards):
        if index<len(manifest['completed_shards']):
            if manifest['completed_shards'][index]['source']!=shard:raise ValueError('Shard order mismatch')
            continue
        remaining=required-manifest['totals']['train']
        if remaining<=0:break
        stage(f'下载/编码分片 {index+1}/{len(shards)} · 已准备 {manifest["totals"]["train"]:,}/{required:,} token')
        path=cache/Path(shard['path']).name
        download(f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{shard["path"]}',path,shard['sha256'],shard['bytes'],stop)
        packed=pack_shard(path,root,index,tokenizer,remaining,
            {k:max(0,settings['holdout_cap']-manifest['totals'][k]) for k in ('validation','test')},stop)
        manifest['files'].extend(packed['files']);manifest['completed_shards'].append(dict(source=shard,**packed))
        for key,value in packed['counts'].items():manifest['totals'][key]+=value
        write(manifest_path,manifest);path.unlink()
    if manifest['totals']['train']<required:raise ValueError('Pinned source exhausted before requested token budget')
    if min(manifest['totals'][k] for k in ('validation','test'))<sequence+1:raise ValueError('Insufficient isolated holdout data')
    stage('准备官方SQuAD问答监督与EOS');prepare_qa(root,tokenizer,sequence,stage,stop)
    manifest['qa_manifest_sha256']=file_hash(root/'qa/manifest.json');manifest['status']='completed';write(manifest_path,manifest)
    return validate(root)


def validate(root):
    root=Path(root);manifest=json.loads((root/'manifest.json').read_text());source=json.loads((root/'sources.json').read_text())
    if manifest['status']!='completed' or digest(source)!=manifest['source_manifest_sha256']:raise ValueError('Dataset incomplete or source changed')
    if file_hash(root/'tokenizer.json')!=source['tokenizer_sha256']:raise ValueError('Tokenizer changed')
    for row in manifest['files']:
        if Path(row['path']).name!=row['path']:raise ValueError('Token pack path must be a local basename')
        path=root/row['path']
        if path.stat().st_size!=row['tokens']*2 or file_hash(path)!=row['sha256']:raise ValueError(f'Frozen token pack changed: {path.name}')
    if file_hash(root/'qa/manifest.json')!=manifest['qa_manifest_sha256']:raise ValueError('QA manifest changed')
    qa=json.loads((root/'qa/manifest.json').read_text())
    if qa['sequence']!=manifest['preparation']['sequence']:raise ValueError('QA sequence differs from token preparation')
    if qa['tokenizer_fingerprint']!=digest(json.loads(Tokenizer.from_file(str(root/'tokenizer.json')).to_str())):raise ValueError('QA tokenizer changed')
    for name,sha in qa['files'].items():
        if Path(name).name!=name:raise ValueError('QA path must be a local basename')
        if file_hash(root/'qa'/name)!=sha:raise ValueError(f'QA data changed: {name}')
    return manifest


class Tokens:
    def __init__(self,root,manifest,split):
        self.parts=[np.memmap(Path(root)/r['path'],dtype='<u2',mode='r') for r in manifest['files'] if r['split']==split]
        self.ends=np.cumsum([len(x) for x in self.parts]);self.total=int(self.ends[-1]) if len(self.ends) else 0

    def read(self,start,length):
        if start<0 or start+length>self.total:raise IndexError('Token read exceeds frozen split')
        chunks=[]
        while length:
            index=int(np.searchsorted(self.ends,start,side='right'));before=0 if index==0 else int(self.ends[index-1])
            count=min(length,int(self.ends[index])-start);chunks.append(self.parts[index][start-before:start-before+count])
            start+=count;length-=count
        return np.concatenate(chunks).astype(np.int64)


def qa_example(paragraph,qa,tokenizer,sequence):
    answer=qa['answers'][0];text=paragraph['context'];start=answer['answer_start'];gold=answer['text']
    if text[start:start+len(gold)]!=gold:raise ValueError('Official answer character alignment mismatch')
    encoded=tokenizer.encode(text,add_special_tokens=False);query=tokenizer.encode('\nQuestion: '+qa['question'].strip()+'\nAnswer:',add_special_tokens=False).ids
    labels=tokenizer.encode(' '+gold,add_special_tokens=False).ids
    budget=sequence-len(query)-len(labels)-2
    if budget<8 or not labels:return None
    overlap=[i for i,(a,b) in enumerate(encoded.offsets) if b>start and a<start+len(gold)]
    if not overlap:return None
    if int(hashlib.sha256(qa['id'].encode()).hexdigest()[:8],16)%2:budget=min(budget,384)
    if overlap[-1]-overlap[0]+1>budget:return None
    left=max(0,min(overlap[0]-budget//2,len(encoded.ids)-budget));right=min(len(encoded.ids),left+budget)
    if not left<=overlap[0]<=overlap[-1]<right:raise ValueError('Evidence crop lost the answer')
    prefix=[EOS]+encoded.ids[left:right]+query;values=prefix+labels+[EOS]
    if len(values)>sequence+1:raise ValueError('QA pack exceeds declared sequence')
    return dict(values=values,start=len(prefix)-1,end=len(prefix)+len(labels),prompt_tokens=len(prefix),
        question_id=qa['id'],reference_answers=[a['text'] for a in qa['answers']],evidence_tokens=right-left)


def prepare_qa(root,tokenizer,sequence,stage=lambda s:None,stop=None):
    root=Path(root)/'qa';root.mkdir(exist_ok=True)
    fingerprint=digest(json.loads(tokenizer.to_str()))
    if (root/'manifest.json').exists():
        previous=json.loads((root/'manifest.json').read_text())
        if (previous.get('sequence')!=sequence or previous.get('tokenizer_fingerprint')!=fingerprint
                or previous.get('source_revision')!=SQUAD_REVISION):raise ValueError('Existing QA preparation differs; choose a fresh data root')
        return
    name='train-v1.1.json';expected=SQUAD_SOURCES[name]
    download(f'https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/{SQUAD_REVISION}/dataset/{name}',root/name,
             expected['sha256'],expected['bytes'],stop)
    raw=json.loads((root/name).read_text());rows=dict(train=[],validation=[]);skipped=0
    for article in raw['data']:
        if stop is not None and stop.requested:raise KeyboardInterrupt
        identity=article_id(article['title']);split=article_split(identity)
        for paragraph in article['paragraphs']:
            for question in paragraph['qas']:
                row=qa_example(paragraph,question,tokenizer,sequence)
                if row is None:skipped+=1;continue
                rows[split].append(row|dict(article_id=identity))
        stage(f'SQuAD编码 · train {len(rows["train"]):,} / validation {len(rows["validation"]):,}')
    files={}
    for split,items in rows.items():
        if not items:raise ValueError('Empty SQuAD split')
        items.sort(key=lambda x:hashlib.sha256(x['question_id'].encode()).digest())
        array=np.lib.format.open_memmap(root/f'{split}.npy.part',mode='w+',dtype=np.uint16,shape=(len(items),sequence+1))
        array[:]=EOS
        with (root/f'{split}.jsonl.part').open('w') as handle:
            for i,row in enumerate(items):
                values=row.pop('values');array[i,:len(values)]=values;handle.write(json.dumps(row,ensure_ascii=False)+'\n')
        array.flush();del array
        for name in (f'{split}.npy',f'{split}.jsonl'):
            (root/(name+'.part')).replace(root/name);files[name]=file_hash(root/name)
    write(root/'manifest.json',dict(source_revision=SQUAD_REVISION,source_sha256=expected['sha256'],sequence=sequence,tokenizer_fingerprint=fingerprint,
        counts={k:len(v) for k,v in rows.items()},skipped=skipped,files=files,
        supervision='Answer tokens and EOS only; evidence is retained. Mixed <=384 and available full-context evidence. Article-disjoint validation; official dev untouched.'))


class QA:
    def __init__(self,root,split):
        root=Path(root)/'qa';self.values=np.load(root/f'{split}.npy',mmap_mode='r')
        self.rows=[json.loads(x) for x in (root/f'{split}.jsonl').read_text().splitlines()]
        if len(self.values)!=len(self.rows):raise ValueError('QA row count mismatch')

    def sample(self,indices):
        values=np.asarray(self.values[indices],dtype=np.int64);mask=np.zeros((len(indices),values.shape[1]-1),dtype=bool)
        for i,index in enumerate(indices):
            row=self.rows[index]
            if not 0<=row['start']<row['end']<=mask.shape[1]:raise ValueError('Invalid QA mask')
            mask[i,row['start']:row['end']]=True
        return values,mask
