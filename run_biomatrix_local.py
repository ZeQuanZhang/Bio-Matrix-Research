"""PyCharm: native Mac training, automatic resume, and pinned public evaluation."""
from importlib.metadata import PackageNotFoundError, version
import os
from pathlib import Path
import sys

MODE = 'train'  # train 自动准备、训练、评测并续跑；smoke 短验收；prepare 只准备数据
PROJECT = Path(__file__).resolve().parent
OUTPUT = PROJECT/'results/local_train/v3_31m_100m_s11'

if __name__ == '__main__':
    print('Bio-Matrix 本地训练启动 · MPS · 每秒原位刷新 · 正在载入环境…', flush=True)
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')
    os.environ.setdefault('PYTHONUNBUFFERED', '1')
    try:
        ready = version('lm_eval') == '0.4.13' and version('torch') == '2.15.0'
    except PackageNotFoundError:
        ready = False
    if not ready:
        runtime = PROJECT/'results/standard_eval/runtime'
        python = runtime/'bin/python'
        if not python.is_file() or Path(sys.prefix).resolve() == runtime.resolve():
            raise RuntimeError('请先安装 evaluations/standard/requirements.txt；详见 local_train/README.md。')
        os.execv(str(python), [str(python), str(Path(__file__).resolve()), *sys.argv[1:]])
    from local_train.run import main
    args = sys.argv[1:]
    if not args:
        args = ['--mode', MODE]
        if MODE != 'smoke':
            args += ['--output', str(OUTPUT)]
    main(args)
