"""PyCharm: run the corrected local comparison or follow its one-second progress frame."""
import importlib.metadata
import os
from pathlib import Path
import sys

MODE = 'run'  # run / follow / prepare
OUTPUT = Path(__file__).resolve().parent/'results/repaired/local333k_corrected_20261103'

if __name__ == '__main__':
    runtime = Path(__file__).resolve().parent/'results/tiny_arch/runtime/bin/python'
    try:
        compatible = importlib.metadata.version('transformers') == '5.22.1'
    except importlib.metadata.PackageNotFoundError:
        compatible = False
    if not compatible:
        if not runtime.is_file():
            raise RuntimeError('请先按 evaluations/tiny_arch/README.md 准备 Transformers 5.22.1 隔离运行环境。')
        os.execv(str(runtime), [str(runtime), str(Path(__file__).resolve()), *sys.argv[1:]])
    if '--mode' not in sys.argv:
        sys.argv += ['--mode', MODE]
    if '--output' not in sys.argv:
        sys.argv += ['--output', str(OUTPUT)]
    from evaluations.repaired.train import main
    main()
