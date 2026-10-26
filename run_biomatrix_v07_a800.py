"""PyCharm controller for the immutable A800 queue; no local model training."""
import argparse
import base64
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time


PROJECT = Path(__file__).resolve().parent
MODE = 'follow'                  # start / status / follow / pause / resume / retry_failed / fetch
CONNECTION_FILE = PROJECT/'results/v07/remote_a800/connection.local.json'
POLL_SECONDS = 1.
RESUME_ABORTED = False          # Explicit opt-in; MODE='resume' also opts in.
RETRY_FAILED = False            # Explicit opt-in; MODE='retry_failed' also opts in.

# This adapter only reads records, starts the frozen scheduler or requests its safe stop.
# It never edits the release, model/config/data, worker arguments or checkpoint files.
REMOTE = r'''
import base64, fcntl, json, os, pathlib, signal, stat, subprocess, sys, tarfile, time
c=json.loads(base64.b64decode(sys.argv[1])); operation=c.pop('operation')
w=pathlib.Path(c['workspace']).resolve(); release=pathlib.Path(c['release']).resolve()
scheduler=pathlib.Path(c['scheduler']).resolve(); w.mkdir(parents=True,exist_ok=True)
marker=w/'scheduler.pid.json'; log=w/'scheduler.log'
def read(path,default=None):
    try:return json.loads(pathlib.Path(path).read_text())
    except FileNotFoundError:return default
def write(path,value):
    temporary=path.with_name(path.name+'.tmp');temporary.write_text(json.dumps(value,indent=2));temporary.replace(path)
def proc(pid):
    try:
        values=pathlib.Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        if values[0]=='Z':return None
        argv=pathlib.Path(f'/proc/{pid}/cmdline').read_bytes().decode(errors='replace').strip('\0').split('\0')
        return dict(pid=int(pid),process_start=values[19],argv=argv)
    except (OSError,IndexError):return None
def argument(argv,name):
    for i,value in enumerate(argv):
        if value==name and i+1<len(argv):return argv[i+1]
        if value.startswith(name+'='):return value.split('=',1)[1]
def expected(row):
    if not row:return False
    try:
        return any(value.startswith('/') and pathlib.Path(value).resolve()==scheduler for value in row['argv']) and pathlib.Path(argument(row['argv'],'--workspace')).resolve()==w and pathlib.Path(argument(row['argv'],'--release')).resolve()==release
    except (TypeError,OSError):return False
def schedulers():
    rows=[]
    for path in pathlib.Path('/proc').iterdir():
        if path.name.isdigit():
            row=proc(path.name)
            if expected(row):rows.append(row)
    return rows
def last_record(path):
    try:
        with path.open('rb') as f:
            f.seek(0,2);f.seek(max(0,f.tell()-1048576));lines=f.read().splitlines()
        for line in reversed(lines):
            try:return json.loads(line)
            except (ValueError,UnicodeError):pass
    except FileNotFoundError:pass
    return {}
def gpu_telemetry():
    try:
        result=subprocess.run(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used,memory.total,power.draw,power.limit,temperature.gpu',
                               '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=3)
        if result.returncode:return []
        rows=[]
        for line in result.stdout.splitlines():
            values=[value.strip() for value in line.split(',')]
            if len(values)!=7:continue
            row=dict(name=values[0])
            for key,value in zip(('utilization_percent','memory_used_mib','memory_total_mib','power_watts','power_limit_watts','temperature_c'),values[1:]):
                try:row[key]=float(value)
                except ValueError:row[key]=None
            rows.append(row)
        return rows
    except (OSError,subprocess.TimeoutExpired):return []
def snapshot(action=None):
    path=pathlib.Path(c.get('status_file',str(w/'status.json')))
    state=read(path)
    if state is None:state=read(w/'scheduler.status.json',dict(status='not_executed',conditions=[]))
    active=schedulers();rows=[];orphans=[]
    for row in state.get('conditions',[]):
        case=row['case'];folder=w/'raw'/case['id']/case['id'];metric=last_record(folder/'metrics.jsonl')
        event=last_record(w/'raw'/case['id']/'events.jsonl');attempt=row.get('attempts',[])
        acceleration=read(folder/'acceleration.json',{}); acceleration_status=acceleration.get('status')
        stage={'compiling':'首次编译','failed':'编译失败'}.get(acceleration_status,event.get('stage',''))
        if not active and row['status'] in ('running','starting') and attempt:
            child=read(attempt[-1].get('pid_marker',''),attempt[-1]) if attempt[-1].get('pid_marker') else attempt[-1]
            found=proc(child.get('pid',0))
            if found and found['process_start']==child.get('process_start'):
                orphans.append(dict(id=case['id'],pid=found['pid']))
        rows.append(dict(id=case['id'],case=case,status=row['status'],worker=attempt[-1].get('worker') if attempt else None,
                         step=metric.get('step',0),loss=metric.get('loss'),tokens_per_second=metric.get('tokens_per_second'),
                         stage=stage,acceleration_status=acceleration_status,reason=row.get('reason',''),
                         attempt=len(attempt),last_metric=metric.get('time')))
    tail=[]
    if not active and log.exists():
        with log.open('rb') as f:f.seek(0,2);f.seek(max(0,f.tell()-4096));tail=f.read().decode(errors='replace').splitlines()[-8:]
    return dict(status=state.get('status','unknown'),updated_at=state.get('updated_at'),conditions=rows,
                workers=state.get('workers',c['workers']),scheduler_alive=bool(active),
                scheduler_processes=[dict(pid=r['pid'],process_start=r['process_start']) for r in active],
                workspace=str(w),action=action,log_tail=tail,orphan_children=orphans,
                sampled_at=time.time(),gpu_telemetry=gpu_telemetry())
if operation=='fetch':
    with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as archive:
        def add_entry(parent,name,relative):
            if name.endswith('.tmp'):return
            try:
                entry=os.stat(name,dir_fd=parent,follow_symlinks=False)
                if stat.S_ISDIR(entry.st_mode):
                    descriptor=os.open(name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=parent)
                    try:
                        current=os.fstat(descriptor);header=tarfile.TarInfo(relative)
                        header.type=tarfile.DIRTYPE;header.mode=stat.S_IMODE(current.st_mode)
                        header.uid=current.st_uid;header.gid=current.st_gid;header.mtime=current.st_mtime
                        archive.addfile(header)
                        for child in sorted(os.listdir(descriptor)):add_entry(descriptor,child,relative+'/'+child)
                    finally:os.close(descriptor)
                elif stat.S_ISREG(entry.st_mode):
                    descriptor=os.open(name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=parent)
                    with os.fdopen(descriptor,'rb') as handle:
                        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):raise OSError('Snapshot file changed kind')
                        header=archive.gettarinfo(fileobj=handle,arcname=relative)
                        archive.addfile(header,fileobj=handle)
            except FileNotFoundError:
                pass
        directory=os.open(w,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        try:
            for name in ('manifest.json','status.json','scheduler.status.json','scheduler.pid.json','scheduler.log','configs','raw','attempts',
                         'scheduling_migrations','live_adoptions','backfill_before.json'):
                add_entry(directory,name,name)
        finally:os.close(directory)
    sys.exit(0)
with (w/'.pycharm-launch.lock').open('a+') as lease:
    fcntl.flock(lease,fcntl.LOCK_EX)
    active=schedulers()
    if len(active)>1:raise RuntimeError('Multiple matching schedulers; refusing to launch or signal any process')
    if operation=='pause':
        action='already_stopped'
        if active:
            row=active[0];current=proc(row['pid'])
            if not expected(current) or current['process_start']!=row['process_start']:raise RuntimeError('Scheduler identity changed; signal refused')
            os.kill(row['pid'],signal.SIGTERM);action='pause_requested'
    elif operation=='start':
        action='attached' if active else 'started'
        if active:
            argv=active[0]['argv']; actual_compile='--compile-step' in argv
            actual_threads=int(argument(argv,'--cpu-threads') or '1')
            actual_dispatch=argument(argv,'--dispatch-policy') or 'barrier'
            actual_reserve=int(argument(argv,'--gpu-reserve-mib') or '12288')
            identity=read(w/'manifest.json',{})
            if (actual_compile!=bool(c.get('compile_step',False)) or actual_threads!=c['cpu_threads']
                    or identity.get('acceleration',{}).get('compile_step',actual_compile)!=bool(c.get('compile_step',False))
                    or identity.get('effective_config',{}).get('cpu_threads',actual_threads)!=c['cpu_threads']
                    or actual_dispatch!=c.get('dispatch_policy','barrier')
                    or actual_reserve!=c.get('gpu_reserve_mib',12288)
                    or identity.get('dispatch_policy',actual_dispatch)!=c.get('dispatch_policy','barrier')
                    or identity.get('gpu_reserve_mib',actual_reserve)!=c.get('gpu_reserve_mib',12288)):
                raise RuntimeError('Running scheduler acceleration/thread/scheduling mode differs; attach with follow/status or use a separate verified workspace')
        if not active:
            if snapshot().get('orphan_children'):raise RuntimeError('Old child processes still run without their scheduler; refusing a second launch')
            argv=['nohup',c['remote_python'],str(scheduler),'--release',str(release),'--workspace',str(w),
                  '--workers',str(c['workers']),'--cpu-threads',str(c['cpu_threads']),'--python',c['remote_python'],
                  '--dispatch-policy',c.get('dispatch_policy','barrier'),'--gpu-reserve-mib',str(c.get('gpu_reserve_mib',12288))]
            if c.get('resume_aborted'):argv.append('--resume-aborted')
            if c.get('retry_failed'):argv.append('--retry-failed')
            if c.get('compile_step'):argv.append('--compile-step')
            env=os.environ.copy();env.update(c.get('environment',{}));env['PYTHONUNBUFFERED']='1'
            with log.open('ab',buffering=0) as output:
                child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,
                                       cwd=release,env=env,start_new_session=True)
            row=proc(child.pid)
            write(marker,dict(pid=child.pid,process_start=row['process_start'] if row else None,argv=argv,started_at=time.time()))
    elif operation=='status':action=None
    else:raise ValueError('Unknown controller operation')
print(json.dumps(snapshot(action),ensure_ascii=False))
'''


def load_connection(path):
    path = Path(path).expanduser().resolve()
    if os.name != 'nt' and path.stat().st_mode & 0o077:
        raise PermissionError('连接配置含已保存密码，请先将 connection.local.json 权限设为 600。')
    cfg = json.loads(path.read_text())
    required = ('host', 'port', 'username', 'password', 'known_hosts', 'remote_python', 'release', 'workspace', 'scheduler')
    if any(key not in cfg for key in required):
        raise ValueError('连接配置缺少必要字段。')
    if not re.fullmatch(r'[A-Za-z0-9_.:-]+', cfg['host']) or cfg['host'].startswith('-'):
        raise ValueError('Invalid SSH host')
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', cfg['username']) or cfg['username'].startswith('-'):
        raise ValueError('Invalid SSH username')
    if not isinstance(cfg['port'], int) or not 1 <= cfg['port'] <= 65535:
        raise ValueError('Invalid SSH port')
    if not isinstance(cfg['password'], str) or not cfg['password'] or any(ch in cfg['password'] for ch in '\r\n'):
        raise ValueError('Invalid local password configuration')
    cfg['workers'] = cfg.get('workers', 5); cfg['cpu_threads'] = cfg.get('cpu_threads', 1)
    cfg['dispatch_policy'] = cfg.get('dispatch_policy', 'barrier')
    cfg['gpu_reserve_mib'] = cfg.get('gpu_reserve_mib', 12288)
    if not isinstance(cfg.get('compile_step', False), bool):
        raise ValueError('compile_step must be a JSON boolean')
    if not isinstance(cfg['workers'], int) or not 1 <= cfg['workers'] <= 32 or not isinstance(cfg['cpu_threads'], int) or cfg['cpu_threads'] < 1:
        raise ValueError('workers must be 1..32 and cpu_threads positive')
    if cfg['dispatch_policy'] not in ('barrier', 'backfill'):
        raise ValueError('dispatch_policy must be barrier or backfill')
    if type(cfg['gpu_reserve_mib']) is not int or cfg['gpu_reserve_mib'] < 0:
        raise ValueError('gpu_reserve_mib must be a nonnegative integer')
    cfg['known_hosts'] = str(Path(cfg['known_hosts']).expanduser().resolve())
    if not Path(cfg['known_hosts']).is_file() or not Path(cfg['known_hosts']).stat().st_size:
        raise FileNotFoundError('缺少已核验主机公钥文件；请使用本次连接固定的 known_hosts。')
    environment = cfg.get('environment', {})
    if not isinstance(environment, dict) or any(not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', key) or not isinstance(value, str)
                                                for key, value in environment.items()):
        raise ValueError('Remote environment must contain string names and values')
    return cfg


class SSH:
    def __init__(self, cfg, config_path, wrapper):
        self.cfg = cfg
        self.env = os.environ.copy()
        self.env.update(SSH_ASKPASS=str(wrapper), SSH_ASKPASS_REQUIRE='force',
                        BIOMATRIX_SSH_CONFIG=str(Path(config_path).resolve()))
        self.env.setdefault('DISPLAY', 'biomatrix-askpass')

    def command(self, operation, **flags):
        public = {key: self.cfg[key] for key in ('remote_python', 'release', 'workspace', 'scheduler', 'workers', 'cpu_threads',
                                               'dispatch_policy', 'gpu_reserve_mib')}
        for key in ('environment', 'status_file'):
            if key in self.cfg:
                public[key] = self.cfg[key]
        public['compile_step'] = bool(self.cfg.get('compile_step', False))
        public.update(operation=operation, **flags)
        payload = base64.b64encode(json.dumps(public).encode()).decode()
        program = base64.b64encode(REMOTE.encode()).decode()
        remote = shlex.join([self.cfg['remote_python'], '-c', f'import base64;exec(base64.b64decode({program!r}))', payload])
        args = ['ssh', '-T', '-p', str(self.cfg['port']), '-l', self.cfg['username'],
                '-o', 'StrictHostKeyChecking=yes', '-o', 'UserKnownHostsFile='+self.cfg['known_hosts'],
                '-o', 'GlobalKnownHostsFile=/dev/null', '-o', 'BatchMode=no', '-o', 'NumberOfPasswordPrompts=1',
                '-o', 'ConnectTimeout=15', '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=3',
                '-o', 'LogLevel=ERROR']
        if self.cfg.get('control_path'):
            args += ['-o', 'ControlMaster=auto', '-o', 'ControlPersist=120', '-o', 'ControlPath='+self.cfg['control_path']]
        return args+[self.cfg['host'], remote]

    def redact(self, value):
        return value.replace(self.cfg['password'], '[redacted]')

    def request(self, operation, **flags):
        result = subprocess.run(self.command(operation, **flags), env=self.env, stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60)
        if result.returncode:
            raise RuntimeError(self.redact(result.stderr.strip() or f'SSH exit {result.returncode}'))
        return json.loads(result.stdout)

    def fetch(self, destination):
        from tqdm import tqdm
        destination = Path(destination); destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_suffix(destination.suffix+'.part')
        if destination.exists():
            raise FileExistsError('Preserving the existing downloaded archive')
        with tempfile.TemporaryFile() as errors:
            process = subprocess.Popen(self.command('fetch'), env=self.env, stdin=subprocess.DEVNULL,
                                       stdout=subprocess.PIPE, stderr=errors)
            try:
                with partial.open('xb') as output, tqdm(desc='取回原始结果', unit='B', unit_scale=True, dynamic_ncols=True) as bar:
                    while block := process.stdout.read(256*1024):
                        output.write(block); bar.update(len(block))
                code = process.wait()
                if code:
                    errors.seek(0)
                    raise RuntimeError(self.redact(errors.read().decode(errors='replace').strip()) or f'SSH fetch exit {code}; partial archive retained')
                partial.replace(destination)
            except BaseException:
                if process.poll() is None:
                    process.terminate(); process.wait()
                raise
            finally:
                process.stdout.close()
        return destination


@contextmanager
def connection(path):
    cfg = load_connection(path)
    with tempfile.TemporaryDirectory(prefix='biomatrix-askpass-') as directory:
        wrapper = Path(directory)/'askpass'
        helper = PROJECT/'scripts/v07_ssh_askpass.py'
        wrapper.write_text('#!/bin/sh\nexec '+shlex.join([sys.executable, str(helper)])+' "$@"\n')
        wrapper.chmod(0o700)
        yield SSH(cfg, path, wrapper)


class Progress:
    """Local-style tqdm bars at fixed positions, with safe PyCharm PTY sizing."""
    def __init__(self, workers):
        from tqdm import tqdm
        self.tqdm = tqdm; self.workers = workers; self.refresh_count = 0; self.states = None
        self.worker_bars = []
        self.bar_format = '{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} {postfix}'
        self.bar = self.new_bar(45*24000, '连接 A800', 0)
        self.resource_bar = self.new_bar(0, '连接远端资源', 1, bar_format='{desc}')
        self.ensure_workers(workers)

    @staticmethod
    def terminal_shape(stream):
        try:
            columns, rows = os.get_terminal_size(stream.fileno())
        except (OSError, ValueError, TypeError, AttributeError):
            columns, rows = 0, 0
        return (max(1, columns-1) if columns > 0 else 120,
                max(2, rows-1) if rows > 0 else 24)

    def bar_shape(self, stream):
        columns, rows = self.terminal_shape(stream)
        return columns, max(rows, self.workers+3)

    def new_bar(self, total, description, position, bar_format=None):
        columns, rows = self.bar_shape(sys.stdout)
        bar = self.tqdm(total=total, desc=description, unit='步', position=position,
                        leave=False, dynamic_ncols=False, mininterval=1., maxinterval=5.,
                        disable=False, gui=False, delay=0., nrows=rows,
                        ncols=columns, file=sys.stdout, bar_format=bar_format or self.bar_format)
        # tqdm 4.70 maps a transient 0×0 PyCharm PTY to negative dimensions,
        # hiding bars or looping forever in Chinese string trimming.
        bar.dynamic_ncols = self.bar_shape
        return bar

    def ensure_workers(self, workers):
        self.workers = min(32, max(self.workers, workers))
        for slot in range(len(self.worker_bars), self.workers):
            self.worker_bars.append(self.new_bar(24000, f'{slot+1:02} 待命', slot+2))

    def connection(self, message):
        self.bar.set_description_str('连接 A800', refresh=False)
        self.bar.set_postfix_str(message, refresh=True)

    def update(self, snapshot):
        self.refresh_count += 1
        rows = snapshot['conditions']; active = {row['worker']: row for row in rows if row['status'] in ('running', 'starting')}
        done = sum(row['status'] == 'completed' for row in rows)
        failed = sum(row['status'] == 'failed' for row in rows); aborted = sum(row['status'] == 'aborted' for row in rows)
        states = {row.get('id', row['case'].get('id', str(index))): row['status'] for index, row in enumerate(rows)}
        if self.states is None:
            self.tqdm.write(f'[{time.strftime("%H:%M:%S")}] 已连接 A800；完成 {done}/{len(rows) or 45}，并行 {len(active)} 组', file=sys.stdout)
        else:
            changed = [(name, status) for name, status in states.items()
                       if self.states.get(name) != status and status in ('running', 'completed', 'failed', 'aborted')]
            if changed:
                names = dict(running='开始', completed='完成', failed='失败', aborted='中止')
                self.tqdm.write(f'[{time.strftime("%H:%M:%S")}] '+ '；'.join(f'{names[status]} {name}' for name, status in changed), file=sys.stdout)
        self.states = states
        occupied = [slot+1 for slot in active if isinstance(slot, int)]
        self.ensure_workers(max(snapshot.get('workers', self.workers), max(occupied, default=0)))
        self.bar.total = (len(rows) or 45)*24000
        self.bar.n = sum(24000 if row['status'] == 'completed' else max(0, min(24000, row.get('step', 0))) for row in rows)
        self.bar.set_description_str('A800 总训练', refresh=False)
        self.bar.set_postfix_str(f'完成{done}/{len(rows) or 45} 活跃{len(active)}/{self.workers} 失败{failed} 中止{aborted} {time.strftime("%H:%M:%S")}', refresh=False)
        self.bar.refresh()
        speed = sum(row.get('tokens_per_second') or 0. for row in active.values())
        resources = [f'活跃{len(active)}/{self.workers}', f'{speed:.0f} target/s']
        for gpu in snapshot.get('gpu_telemetry', []):
            values = []
            if gpu.get('utilization_percent') is not None:values.append(f'GPU{gpu["utilization_percent"]:.0f}%')
            if gpu.get('memory_used_mib') is not None and gpu.get('memory_total_mib') is not None:
                values.append(f'{gpu["memory_used_mib"]/1024:.1f}/{gpu["memory_total_mib"]/1024:g}G')
            if gpu.get('power_watts') is not None:values.append(f'{gpu["power_watts"]:.0f}W')
            if gpu.get('temperature_c') is not None:values.append(f'{gpu["temperature_c"]:.0f}℃')
            resources.append(' '.join(values))
        if not snapshot.get('gpu_telemetry'):resources.append('GPU 遥测暂不可用')
        self.resource_bar.set_description_str(' | '.join(resources), refresh=True)
        for slot, bar in enumerate(self.worker_bars):
            row = active.get(slot)
            if row is None:
                bar.n = 0; bar.set_description_str(f'{slot+1:02} 待命', refresh=False); bar.set_postfix_str('', refresh=False)
            else:
                case = row['case']; bar.n = min(24000, row['step'])
                bar.set_description_str(f'{slot+1:02} {case["variant"]} N{case["nodes"]} s{case["seed"]}', refresh=False)
                loss = '—' if row.get('loss') is None else f'{row["loss"]:.4f}'
                rate = '—' if row.get('tokens_per_second') is None else f'{row["tokens_per_second"]:.1f}'
                bar.set_postfix_str(f'loss={loss} {rate} target/s {row.get("stage") or "准备"}', refresh=False)
            bar.refresh()

    def close(self):
        for bar in reversed(self.worker_bars):bar.close()
        self.resource_bar.close(); self.bar.close()


def follow(ssh, once=False, json_output=False, first=None, poll_seconds=POLL_SECONDS):
    progress = None; snapshot = first
    try:
        if not json_output:
            progress = Progress(ssh.cfg['workers'])
            progress.connection('正在连接 SSH 并读取训练进度…')
        while True:
            cycle_started = time.monotonic()
            if snapshot is None:
                try:
                    snapshot = ssh.request('status')
                except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
                    if once or progress is None:
                        raise
                    detail = 'SSH 请求超时' if isinstance(error, subprocess.TimeoutExpired) else str(error).replace('\n', ' ')[:120]
                    progress.connection(f'{detail}；连接暂未成功，正在重连…')
                    time.sleep(min(30., poll_seconds*2)); continue
            if json_output:
                print(json.dumps(snapshot, ensure_ascii=False, indent=2)); return snapshot
            if progress is None:
                progress = Progress(max(ssh.cfg['workers'], snapshot.get('workers', 0)))
            progress.update(snapshot)
            if once or not snapshot['scheduler_alive']:
                return snapshot
            time.sleep(max(0., poll_seconds-(time.monotonic()-cycle_started))); snapshot = None
    finally:
        if progress is not None:
            progress.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('start', 'status', 'follow', 'pause', 'resume', 'retry_failed', 'fetch'), default=MODE)
    parser.add_argument('--connection', type=Path, default=CONNECTION_FILE)
    parser.add_argument('--once', action='store_true', help='One status snapshot; start still starts the detached scheduler')
    parser.add_argument('--json', action='store_true', help='One public JSON snapshot, without progress bars')
    parser.add_argument('--resume-aborted', action='store_true', default=RESUME_ABORTED)
    parser.add_argument('--retry-failed', action='store_true', default=RETRY_FAILED)
    parser.add_argument('--poll-seconds', type=float, default=POLL_SECONDS)
    args = parser.parse_args(argv)
    if args.poll_seconds <= 0 or not shutil.which('ssh'):
        parser.error('Need native ssh and positive poll seconds')
    if not args.json:
        print(f'[A800] v0.7 控制台已启动 · {args.mode} · 每 {args.poll_seconds:g} 秒刷新', flush=True)
        print('正在连接 A800；首轮 SSH 请求最长等待 60 秒。停止本地查看后，远端训练继续。', flush=True)
    with connection(args.connection) as ssh:
        if args.mode == 'fetch':
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            target = args.connection.parent/'downloads'/f'a800-raw-{stamp}.tar.gz'
            print(ssh.fetch(target)); return 0
        first = None
        if args.mode in ('start', 'resume', 'retry_failed'):
            first = ssh.request('start', resume_aborted=args.resume_aborted or args.mode == 'resume',
                                retry_failed=args.retry_failed or args.mode == 'retry_failed')
        elif args.mode == 'pause':
            first = ssh.request('pause')
        result = follow(ssh, once=args.once or args.mode == 'status', json_output=args.json,
                        first=first, poll_seconds=args.poll_seconds)
        if result.get('orphan_children'):
            raise RuntimeError('调度器已停止，但仍有旧 child 存活；未向不明进程发信号，恢复前请先核对远端状态。')
        if not result['scheduler_alive'] and result.get('log_tail') and result['status'] not in ('completed', 'aborted'):
            print('\n远端调度器记录：\n'+ssh.redact('\n'.join(result['log_tail'])), file=sys.stderr)
        stopped_during_launch = (args.mode in ('start', 'resume', 'retry_failed') and not result['scheduler_alive']
                                 and result['status'] not in ('completed', 'aborted'))
        return 1 if result['status'] == 'failed' or stopped_during_launch else 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('\n已停止本地查看；远端训练继续。需要保存并暂停时显式使用 MODE="pause"。')
    except Exception as error:
        print(f'\n[A800 控制器] {error}', file=sys.stderr)
        raise SystemExit(1)
