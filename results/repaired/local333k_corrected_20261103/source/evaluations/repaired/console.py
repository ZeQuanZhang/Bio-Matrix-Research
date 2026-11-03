"""Reuse the one-second frame with short, distinguishable labels for all jobs."""
from types import SimpleNamespace
from scripts.v2_console import Progress as BaseProgress


class Progress(BaseProgress):
    label = '修正版 · 同预算架构对照'

    def frame(self, columns, rows, resources=''):
        aliases = {'trans_local': 'TF128', 'trans_full': 'TF512', 'v3_static': 'BioS',
                   'v3_dynamic': 'BioD', 'qwen35_tiny': 'Qwen',
                   'deepseek4_tiny': 'DS4', 'deepseek4_balanced': 'DS2'}
        def short(identity):
            for name in sorted(aliases, key=len, reverse=True):
                identity = identity.replace(name, aliases[name])
            return identity.replace('overfit_', '拟合-').replace('cpt_search_', 'C-').replace('qa_search_', 'Q-').replace('final_', '').replace('_cpt', '-C').replace('_qa', '-Q').replace('_s', '-').replace('_r', '-r')
        proxy = SimpleNamespace(total=self.total, label=self.label, message=self.message,
                                details={short(k): v for k, v in self.details.items()})
        return BaseProgress.frame(proxy, columns, rows, resources)
