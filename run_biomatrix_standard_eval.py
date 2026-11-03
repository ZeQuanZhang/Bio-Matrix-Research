"""PyCharm entry for the independent, pinned standard evaluation."""
from importlib.metadata import PackageNotFoundError, version
import os
from pathlib import Path
import sys

if __name__ == '__main__':
    root = Path(__file__).resolve().parent
    try:
        ready = version('lm_eval') == '0.4.13'
    except PackageNotFoundError:
        ready = False
    if not ready:
        runtime = root/'results/standard_eval/runtime'
        python = runtime/'bin/python'
        if not python.is_file() or Path(sys.prefix).resolve() == runtime.resolve():
            raise RuntimeError('请先安装 evaluations/standard/requirements.txt。')
        os.execv(str(python), [str(python), str(Path(__file__).resolve()), *sys.argv[1:]])
    from evaluations.standard.run import main
    main()
