"""Restore pinned public source prerequisites without overwriting historical data."""
from pathlib import Path
import shutil
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from v07.data import prepare, ROOT as WIKI_ROOT
from v11.data import SOURCES, REVISION
from scale.data import download


def main():
    if WIKI_ROOT.exists():
        prepare(WIKI_ROOT)
    else:
        staging = PROJECT/'data/repaired_public_sources/wikitext2'
        prepare(staging)
        WIKI_ROOT.parent.mkdir(parents=True, exist_ok=True)
        # Copy into an absent destination only; historical caches are never replaced.
        shutil.copytree(staging, WIKI_ROOT)
    target = PROJECT/'data/v11/squad_remote256'
    for name, expected in SOURCES.items():
        download(f'https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/{REVISION}/dataset/{name}',
                 target/name, expected['sha256'], expected['bytes'])
    print('固定公开来源已核验；现在可运行 run_biomatrix_repaired.py --mode prepare。')


if __name__ == '__main__':
    main()
