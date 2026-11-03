import unittest
from pathlib import Path
import numpy as np
import torch
from tokenizers import Tokenizer

from . import data
from .model import build, metadata, NAMES
from .engine import intervention, generate


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def test_full_passage_answer_eos_and_rejection_without_crop(self):
        tok = Tokenizer.from_file(str(data.ROOT/'tokenizer.json'))
        text = 'Paris is the capital of France. It has many museums.'
        q = dict(id='test', question='What is the capital of France?', answers=[dict(text='Paris', answer_start=0)])
        row, reason = data.qa_example(tok, {'context': text}, q)
        self.assertIsNone(reason)
        self.assertEqual(tok.decode(row['values'][1:1+row['context_tokens']]), text)
        self.assertEqual(row['values'][row['supervised_end']], 2)
        self.assertEqual(tok.decode(row['values'][row['answer_start']+1:row['answer_end']+1]).strip(), 'Paris')
        row, reason = data.qa_example(tok, {'context': text*200}, q)
        self.assertIsNone(row); self.assertEqual(reason, 'full_passage_prompt_too_long')

    def test_coverage_has_each_target_once_even_for_short_documents(self):
        class Tiny:
            tokens = {'test': np.arange(1400)}
            offsets = {'test': np.asarray([0, 2, 1400])}
        seen = []
        for values, mask, _ in data.lm_windows(Tiny(), 'test'):
            seen.extend(values[1:][mask].tolist())
        self.assertEqual(seen, [1]+list(range(3, 1400)))

    def test_data_disjoint_and_objective_no_prompt_or_padding(self):
        groups = {s: data.QA(data.ROOT, s) for s in ('train', 'validation', 'test')}
        for key in ('article_id', 'context_id', 'question_id'):
            sets = [{r[key] for r in groups[s].rows} for s in groups]
            self.assertFalse(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
        q = groups['train']; v, m, rows = q.batch([0, 1, 2])
        for lane, r in enumerate(rows):
            self.assertFalse(m[lane, :r['answer_start']].any())
            self.assertTrue(m[lane, r['answer_start']:r['supervised_end']].all())
            self.assertFalse(m[lane, r['supervised_end']:].any())
            self.assertEqual(int(v[lane, r['supervised_end']]), 2)
        eligible = [i for i, r in enumerate(q.rows) if r['prefix_tokens'] <= 384][:2]
        v, _, rows = q.batch(eligible)
        changed, shifted = intervention(q, eligible, v, rows, 'independent_prior')
        for i, (r, other) in enumerate(zip(rows, shifted)):
            np.testing.assert_array_equal(v[i, 1:r['supervised_end']+1], changed[i, 65:other['supervised_end']+1])
            self.assertEqual(other['prefix_tokens'], r['prefix_tokens']+64)

    def test_all_models_causal_prefix_equivalence_and_parameter_budgets(self):
        x = torch.randint(3, 4096, (1, 512)); other = x.clone(); other[:, 384:] = 17
        for name in NAMES:
            model = build(name, 11).eval(); metadata(name, model)
            with torch.no_grad():
                expected = model(x)[0]; altered = model(other)[0]
                torch.testing.assert_close(expected[:, :384], altered[:, :384], atol=4e-5, rtol=4e-4)
                _, state, _ = model(x[:, :384]); suffix = model(x[:, 384:], state)[0]
                torch.testing.assert_close(expected[:, 384:], suffix, atol=4e-5, rtol=4e-4)

    def test_native_mps_backward_finite(self):
        if not torch.backends.mps.is_available():
            self.skipTest('MPS unavailable')
        for name in NAMES:
            model = build(name, 11).to('mps')
            logits, _, diag = model(torch.randint(3, 4096, (2, 32), device='mps'))
            loss = logits.square().mean()+.001*diag.get('native_auxiliary', 0.)
            loss.backward()
            self.assertTrue(bool(torch.isfinite(loss)))
            self.assertTrue(all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None))

    def test_batched_generation_never_reads_gold(self):
        qa = data.QA(data.ROOT, 'validation')
        tokenizer = Tokenizer.from_file(str(data.ROOT/'tokenizer.json'))
        captured = []
        class EosModel:
            def eval(self):
                return self
            def __call__(self, x):
                captured.append(x.clone())
                out = torch.zeros((*x.shape, 4096)); out[:, :, 2] = 1
                return out, None, {}
        result = generate(EosModel(), qa, [0, 1], tokenizer, 'cpu')
        self.assertEqual(result['cap_rate'], 0.)
        for i in range(2):
            prefix = qa.rows[i]['prefix_tokens']
            np.testing.assert_array_equal(captured[0][i, :prefix], qa.values[i, :prefix])
            self.assertFalse(bool(captured[0][i, prefix:].any()))


if __name__ == '__main__':
    unittest.main()
