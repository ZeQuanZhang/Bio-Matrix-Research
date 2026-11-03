"""Regression: 12 bars must fit a 13-row PyCharm terminal without wrapping."""
import io
import unittest

from scripts.v2_console import Progress,width


class ConsoleChecks(unittest.TestCase):
    def make(self):
        records=[dict(id=f'{n}_s{s}') for s in (11,23,37)
                 for n in ('trans_local','trans_full','v2_static','v2_dynamic')]
        p=Progress(records,100,False,stream=io.StringIO())
        for i,bar in enumerate(p.details.values()):
            bar.update(i*8);bar.set_postfix_str('CE=4.1234 输入/s=12345 答案目标=6789')
        p.stage('正在执行完整历史验证，长阶段名称也必须留在原行，不允许换行')
        return p

    def test_all_twelve_bars_fit_short_and_narrow_terminals(self):
        for cols,rows in ((162,13),(80,13),(80,8),(80,4),(40,13)):
            with self.subTest(columns=cols,rows=rows):
                p=self.make();lines=p.frame(cols,rows,'CPU 100% | RSS 2.1 GiB')
                self.assertLessEqual(len(lines),rows-1)
                self.assertTrue(all(width(line)<=cols-2 for line in lines))
                joined='\n'.join(lines)
                for s in (11,23,37):
                    for name,short in (('trans_local','L'),('trans_full','F'),('v2_static','S'),('v2_dynamic','D')):
                        self.assertTrue(f'{name}_s{s}' in joined or f'{short}{s}' in joined)

    def test_updates_do_not_print_individual_lines(self):
        p=self.make();stream=p.stream
        for _ in range(100):
            p.stage('训练');p.details['v2_dynamic_s11'].update(1)
        self.assertEqual(stream.getvalue(),'')

    def test_tiny_window_stays_bounded_and_close_is_idempotent(self):
        p=self.make()
        self.assertLessEqual(len(p.frame(20,2)),1)
        p.close();p.close()


if __name__=='__main__':unittest.main()
