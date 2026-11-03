import unittest

from lm_eval.api.instance import Instance
from lm_eval.models.huggingface import HFLM
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from .run import backend, protocol, reference_tokenizer


class AdapterTests(unittest.TestCase):
    def test_official_backend_parity(self):
        tokenizer, _ = reference_tokenizer()
        torch.manual_seed(11)
        network = GPT2LMHeadModel(GPT2Config(vocab_size=tokenizer.vocab_size,
            n_positions=64, n_embd=16, n_layer=1, n_head=2, bos_token_id=1, eos_token_id=2, pad_token_id=0))
        native = HFLM(pretrained=network, tokenizer=tokenizer, backend='causal', max_length=32,
            batch_size=2, add_bos_token=False, softmax_dtype='float32')
        wrapped = backend(network, tokenizer, 32, 2)
        pairs = [('A short test ', 'continues here.'), ('', 'An empty context.'), ('a prefix '*20, 'the ending')]
        requests = [Instance('loglikelihood', {}, pair, i) for i, pair in enumerate(pairs)]
        self.assertEqual(native.loglikelihood(requests), wrapped.loglikelihood(requests))
        rolling = [Instance('loglikelihood_rolling', {}, ('History and a rolling context. '*18,), 0)]
        self.assertEqual(native.loglikelihood_rolling(rolling), wrapped.loglikelihood_rolling(rolling))

    def test_bio_batch_padding(self):
        from evaluations.repaired.model import build
        tokenizer, _ = reference_tokenizer()
        network = build('v3_dynamic', 11)
        requests = [Instance('loglikelihood', {}, pair, i) for i, pair in enumerate(
            [('A test', ' with a short answer.'), ('A longer passage of text. '*30, ' This is the final answer.')])]
        one = backend(network, tokenizer, 512, 1).loglikelihood(requests)
        two = backend(network, tokenizer, 512, 2).loglikelihood(requests)
        for a, b in zip(one, two):
            self.assertAlmostEqual(a[0], b[0], delta=1e-4)
            self.assertEqual(a[1], b[1])

    def test_locked_definition(self):
        self.assertEqual(len(protocol()['tasks']), 5)
