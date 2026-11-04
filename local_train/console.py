"""One renderer owns the terminal; zero-size IDE terminals use a usable fallback."""
import math
import os
import sys
import time

from scripts.v2_console import Progress as BaseProgress, fit, width

LABELS = {'data': '数据', 'transformer': 'TF', 'bio_static': 'BioS', 'bio_dynamic': 'BioD',
          'wikitext': 'WT', 'lambada_openai': 'LAMB', 'piqa': 'PIQA', 'arc_easy': 'ARC-E', 'hellaswag': 'HS'}


class IDEStream:
    def __init__(self, stream):
        self.stream = stream

    def __getattr__(self, name):
        return getattr(self.stream, name)

    def isatty(self):
        return self.stream.isatty() or os.environ.get('PYCHARM_HOSTED') == '1'


class Progress(BaseProgress):
    label = 'V3 · 本地31M'

    def __init__(self, enabled=True):
        super().__init__([dict(id=k) for k in LABELS], 1, enabled, stream=IDEStream(sys.stderr))

    def size(self):
        try:
            columns, rows = os.get_terminal_size(self.stream.fileno())
        except (OSError, ValueError, TypeError, AttributeError):
            columns, rows = 0, 0
        return columns if columns > 0 else 120, rows if rows > 0 else 24

    def resources(self):
        text = super().resources()
        if self.device is not None and self.device.type == 'mps':
            import torch
            text += f' | 驱动 {torch.mps.driver_allocated_memory()/2**30:.1f} GiB'
        return text

    def frame(self, columns, rows, resources=''):
        columns, available = max(1, columns-2), max(1, rows-1)
        if available < 4 or columns < 30:
            return [fit('请扩大控制台：至少 32 列、5 行', columns)]
        headers = [f'Bio-Matrix {self.label} | {time.strftime("%H:%M:%S")} | 完成 {self.total.n}/{self.total.total}',
                   self.message, resources]
        capacity = available-3
        cols = max(1, math.ceil(len(self.details)/capacity))
        cell = (columns-3*(cols-1))//cols
        if cell < 14:
            return [fit(headers[0], columns), fit('请扩大控制台以显示全部 9 条进度', columns)]
        cells = []
        for name, bar in self.details.items():
            ratio = min(1., max(0., bar.n/max(1, bar.total)))
            length = max(2, min(22, cell-16))
            filled = int(length*ratio)
            value = f'{LABELS[name]:5} [{"#"*filled}{"-"*(length-filled)}] {ratio:5.1%}'
            if cell >= 65:
                value += f' {bar.n:,.0f}/{bar.total:,.0f} {bar.postfix}'
            elif cell >= 40:
                value += ' '+bar.postfix
            cells.append(fit(value, cell))
        lines = [fit(h, columns) for h in headers]
        for start in range(0, len(cells), cols):
            lines.append('   '.join(v+' '*(cell-width(v)) for v in cells[start:start+cols]).rstrip())
        return lines
