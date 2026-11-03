"""Height-aware progress frames, separate from the frozen experiment sources."""
import math
import os
from pathlib import Path
import re
import resource
import subprocess
import sys
import threading
import time
import unicodedata


def width(text):
    return sum(0 if unicodedata.combining(c) else 2 if unicodedata.east_asian_width(c) in ('W','F') else 1 for c in text)


def fit(text, columns):
    result=[];used=0
    for c in str(text).replace('\n',' ').replace('\r',' '):
        size=0 if unicodedata.combining(c) else 2 if unicodedata.east_asian_width(c) in ('W','F') else 1
        if used+size>columns:break
        result.append(c);used+=size
    return ''.join(result)


class Bar:
    def __init__(self,total,desc):self.total=total;self.n=0;self.desc=desc;self.postfix=''
    def update(self,n=1):self.n+=n
    def set_description_str(self,value,refresh=True):self.desc=value
    def set_postfix_str(self,value,refresh=True):self.postfix=value
    def reset(self,total=None):
        self.n=0
        if total is not None:self.total=total


class Progress:
    label='v2'
    def __init__(self,records,updates,enabled=True,stream=None,pid=None):
        self.stream=stream or sys.stderr;self.enabled=enabled;self.pid=pid or os.getpid()
        self.total=Bar(len(records),self.label);self.details={r['id']:Bar(updates,r['id']) for r in records}
        self.message='准备';self.device=None;self.stop=threading.Event();self.closed=False
        self.terminal=enabled and self.stream.isatty()
        self.thread=None
        if enabled:
            if self.terminal:self.stream.write('\x1b[?1049h\x1b[?25l');self.stream.flush()
            self.draw()
            self.thread=threading.Thread(target=self.refresh,daemon=True);self.thread.start()

    def stage(self,message):self.message=message

    def size(self):
        try:columns,rows=os.get_terminal_size(self.stream.fileno())
        except (OSError,ValueError,AttributeError):columns,rows=80,8
        return columns or 80,rows or 8

    def resources(self):
        try:
            result=subprocess.run(['ps','-p',str(self.pid),'-o','%cpu=,rss='],capture_output=True,text=True,timeout=1)
            cpu,rss=result.stdout.split()[:2]
            text=f'PID {self.pid} | CPU {cpu}% | RSS {int(rss)/1024**2:.2f} GiB'
        except (OSError,ValueError,subprocess.TimeoutExpired):text=f'PID {self.pid}'
        if self.device is not None and self.device.type=='mps':
            import torch
            text+=f' | Metal {torch.mps.current_allocated_memory()/2**30:.2f} GiB'
        return text

    def frame(self,columns,rows,resources=''):
        columns=max(1,columns-2);available=max(1,rows-1)
        headers=[f'Bio-Matrix {self.label} | 完成 {int(self.total.n)}/{self.total.total} | {time.strftime("%H:%M:%S")}',
                 str(self.message),resources]
        header_count=min(3,max(1,available-2))
        capacity=max(1,available-header_count)
        cols=max(1,math.ceil(len(self.details)/capacity))
        cell=(columns-3*(cols-1))//cols
        if cell<9:
            return [fit(headers[0],columns),fit('请增高或加宽控制台以显示全部进度',columns)][:available]
        cells=[]
        aliases={'trans_local':'L','trans_full':'F','v2_static':'S','v2_dynamic':'D','v3_static':'S','v3_dynamic':'D'}
        for identity,bar in list(self.details.items()):
            ratio=min(1.,max(0.,bar.n/max(1,bar.total)))
            label=identity
            if cell<48:
                condition,_,seed=identity.rpartition('_s')
                label=aliases.get(condition,condition)+seed
            ce=re.search(r'(CE|QA)=([\d.eE+\-]+)',bar.postfix)
            suffix=f' {ce.group(1)}={ce.group(2)}' if ce and cell>=35 else ''
            bar_width=max(1,min(20,cell-width(label)-len(suffix)-9))
            fill=int(bar_width*ratio)
            text=f'{label} [{"#"*fill}{"-"*(bar_width-fill)}] {ratio:3.0%}{suffix}' if cell>=15 else f'{label} {ratio:.0%}'
            cells.append(fit(text,cell))
        lines=[fit(x,columns) for x in headers[:header_count]]
        for at in range(0,len(cells),cols):
            group=cells[at:at+cols]
            lines.append('   '.join(c+' '*(cell-width(c)) for c in group).rstrip())
        return lines

    def draw(self):
        if not self.enabled:return
        columns,rows=self.size()
        lines=self.frame(columns,rows,self.resources())
        if self.terminal:
            self.stream.write('\x1b[H'+'\r\n'.join(line+'\x1b[K' for line in lines)+'\x1b[J')
        else:
            self.stream.write('\r'+fit(f'{self.label} {self.total.n}/{self.total.total} {self.message}',max(1,columns-2)))
        self.stream.flush()

    def refresh(self):
        while not self.stop.wait(1.):self.draw()

    def close(self):
        if self.closed:return
        self.closed=True;self.stop.set()
        if self.thread:self.thread.join(timeout=2)
        if self.enabled:
            if self.terminal:
                columns,rows=self.size();lines=self.frame(columns,rows,self.resources())
                self.stream.write('\x1b[?25h\x1b[?1049l'+'\n'.join(lines)+'\n')
            else:self.stream.write('\n')
            self.stream.flush()


def follow(root,pid=None):
    import json
    root=Path(root)
    protocol=json.loads((root/'protocol.json').read_text());c=protocol['training']
    registry=json.loads((root/'registry.json').read_text())
    records=[dict(id=name) for name in registry['conditions']]
    work=c['cpt_tokens']+c['pt_updates']*c['batch']*c['segments']*c['segment_tokens']
    ui=Progress(records,work,pid=pid)
    try:
        while True:
            registry=json.loads((root/'registry.json').read_text())
            rows={name:dict(value) for name,value in registry['conditions'].items()}
            live_path=root/'live.json';live=json.loads(live_path.read_text()) if live_path.exists() else {}
            identity=live.get('condition')
            if identity in rows and rows[identity]['status']=='running':rows[identity].update(live)
            ui.total.n=sum(row['status']=='completed' for row in rows.values())
            for name,row in rows.items():
                bar=ui.details[name];bar.n=row.get('completed_work',0)
                if row.get('ce') is not None:bar.postfix=f'CE={row["ce"]:.4f}'
                elif 'validation' in row:bar.postfix=f'QA={row["validation"]["qa"]["warm_nll"]:.4f}'
            current=[(name,row) for name,row in rows.items() if row['status']=='running']
            ui.stage(' | '.join(f'{name} {row.get("stage","验证/准备")} {row.get("updates",0)}步' for name,row in current)
                     or f'{root.name} · {registry["status"]}')
            if registry['status'] in ('completed','aborted','failed'):break
            if pid is not None:
                try:os.kill(pid,0)
                except ProcessLookupError:ui.stage('训练进程已结束，保留最后保存的进度');break
            if ui.stop.wait(1.):break
    except KeyboardInterrupt:pass
    finally:ui.close()
