"""Adaptive console for new/resumed v2 runs, or read-only viewing of a current run."""
import sys
from pathlib import Path

from scripts.v2_console import Progress,follow

PROJECT=Path(__file__).resolve().parent


def main():
    if '--follow' in sys.argv[1:]:
        at=sys.argv.index('--follow')
        if len(sys.argv)>at+1:root=Path(sys.argv[at+1]).resolve()
        else:
            candidates=[p for p in (PROJECT/'results/v2').glob('paired_*') if (p/'registry.json').is_file()]
            if not candidates:raise FileNotFoundError('尚无v2正式运行结果')
            root=max(candidates,key=lambda p:(p/'registry.json').stat().st_mtime)
        try:pid=int((PROJECT/'results/v2/.train.lock').read_text())
        except (OSError,ValueError):pid=None
        follow(root,pid);return
    import run_biomatrix_v2 as entry
    from v2 import train
    train.Progress=Progress
    arguments=['--mode',entry.MODE,'--device',entry.DEVICE]
    if entry.RESUME is not None:arguments+=['--resume',str(entry.RESUME)]
    elif '--resume' not in sys.argv[1:]:
        arguments+=['--cpt-tokens',str(entry.CPT_TARGETS),'--pt-updates',str(entry.PT_UPDATES),'--seeds',entry.SEEDS]
    train.main(arguments+sys.argv[1:])


if __name__=='__main__':main()
