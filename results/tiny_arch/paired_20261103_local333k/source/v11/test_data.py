"""Tiny schema fixtures test data contracts; formal QA uses pinned human data."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from .data import (SPLITS, _local_answer_leak, _query_answer, _validated_articles, article_id,
                   article_split, normalized, prepare_qa)


def title_for(prefix, split):
    for index in range(1000):
        title = prefix+str(index)
        if article_split(article_id(title)) == split:
            return title
    raise AssertionError('Could not construct a deterministic split fixture')


def article(title, answer, identifier, before=20):
    context = ' '.join(['filler']*before+[answer]+['filler']*130)
    return dict(title=title, paragraphs=[dict(context=context, qas=[dict(id=identifier,
                question='What gemstone was identified?',
                answers=[dict(text=answer, answer_start=context.index(answer))])])])


def fixture(root, duplicates=False):
    vocabulary = {'[PAD]': 0, '[BOS]': 1, '[EOS]': 2, '[UNK]': 3, 'filler': 4,
                  'Amberstone': 5, 'Berylstone': 6, '蓝晶石': 7, 'What': 8,
                  'gemstone': 9, 'was': 10, 'identified': 11, '?': 12,
                  'Question': 13, ':': 14, 'Answer': 15}
    tokenizer = Tokenizer(models.WordLevel(vocabulary, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    frozen = root/'frozen'; frozen.mkdir()
    tokenizer.save(str(frozen/'tokenizer.json'))
    corpus = SimpleNamespace(root=frozen, manifest=dict(vocab_size=len(vocabulary), fixture=True))
    official_train = dict(version='1.1', data=[article(title_for('train', 'train'), 'Amberstone', 'train-qa'),
                                             article(title_for('validation', 'validation'), 'Berylstone', 'val-qa')])
    official_dev = dict(version='1.1', data=[article('reserved_dev_article', '蓝晶石', 'test-qa')])
    if duplicates:
        duplicate = article(title_for('duplicate_train', 'train'), 'Amberstone', 'dup-train', before=23)
        second = deepcopy(duplicate)
        second['title'] = 'different_dev_article'
        second['paragraphs'][0]['qas'][0]['id'] = 'dup-test'
        official_train['data'].append(duplicate)
        official_dev['data'].append(second)
    destination = root/'qa'; destination.mkdir()
    pins = {}
    for name, value in [('train-v1.1.json', official_train), ('dev-v1.1.json', official_dev)]:
        raw = json.dumps(value, ensure_ascii=False).encode()
        (destination/name).write_bytes(raw)
        pins[name] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
                          git_blob_sha1=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest())
    return destination, corpus, tokenizer, pins


class DataChecks(unittest.TestCase):
    def prepare(self, root, duplicates=False):
        destination, corpus, tokenizer, pins = fixture(root, duplicates)
        with patch('v11.data.SOURCES', pins), patch('v11.data.tqdm', side_effect=lambda values, **_: values):
            qa = prepare_qa(destination, corpus, length=64, prefix=48, context_limit=8, allow_download=False)
        return qa, corpus, tokenizer, pins

    def test_original_span_schema_requires_exact_unicode_character_alignment(self):
        raw = dict(version='1.1', data=[article('unicode', '蓝晶石', 'unicode-qa')])
        self.assertEqual(len(_validated_articles(raw, 'fixture')), 1)
        changed = deepcopy(raw)
        changed['data'][0]['paragraphs'][0]['qas'][0]['answers'][0]['answer_start'] += 1
        with self.assertRaisesRegex(ValueError, 'answer_start/text mismatch'):
            _validated_articles(changed, 'fixture')
        changed = deepcopy(raw)
        changed['data'][0]['paragraphs'][0]['qas'][0]['answers'][0]['answer_start'] = True
        with self.assertRaises(ValueError):
            _validated_articles(changed, 'fixture')

    def test_masks_only_answer_targets_and_first_answer_with_remote_evidence(self):
        with TemporaryDirectory() as folder:
            qa, _, tokenizer, _ = self.prepare(Path(folder))
            for split in SPLITS:
                values, mask, metadata = qa.sample(split, np.random.default_rng(17), 2)
                self.assertEqual(values.shape, (2, 65))
                self.assertEqual(mask.shape, (2, 64))
                first = metadata['first_answer_mask']
                self.assertTrue(np.all(first.sum(1) == 1))
                self.assertTrue(np.all(first <= mask))
                for lane, row in enumerate(metadata['rows']):
                    labels = values[lane, 1:][mask[lane]].tolist()
                    self.assertEqual(labels, _query_answer(tokenizer, 'What gemstone was identified?', row['answer_text'])[1])
                    self.assertLess(row['context_answer_end'], 48-2*8)
                    self.assertGreater(row['answer_distance_tokens'], 2*8)
                    self.assertFalse(bool(mask[lane, row['answer_target_end']:].any()))
                    self.assertIsNone(_local_answer_leak(tokenizer, values[lane, :48].tolist(),
                                      values[lane, 48:row['answer_input_start']].tolist(), labels,
                                      row['answer_text'], 2*8))

    def test_same_seed_pairs_samples_and_test_arrays_are_opened_only_on_request(self):
        with TemporaryDirectory() as folder:
            qa, _, _, _ = self.prepare(Path(folder))
            left = qa.sample('train', np.random.default_rng(81), 3)
            right = qa.sample('train', np.random.default_rng(81), 3)
            np.testing.assert_array_equal(left[0], right[0])
            np.testing.assert_array_equal(left[1], right[1])
            np.testing.assert_array_equal(left[2]['first_answer_mask'], right[2]['first_answer_mask'])
            self.assertEqual(left[2]['record_ids'], right[2]['record_ids'])
            self.assertEqual(set(qa._sequences), {'train'})
            sets = {split: {row['article_id'] for row in qa.records[split]} for split in SPLITS}
            self.assertFalse(sets['train'] & sets['validation'])
            self.assertFalse((sets['train'] | sets['validation']) & sets['test'])

    def test_duplicate_official_context_is_excluded_from_both_splits(self):
        with TemporaryDirectory() as folder:
            qa, _, _, _ = self.prepare(Path(folder), duplicates=True)
            audit = qa.manifest['split_overlap_audit']
            self.assertEqual(audit['official_train_dev_context_overlap'], 1)
            self.assertEqual(audit['cross_split_context_ids_excluded'], 1)
            for split in ('train', 'test'):
                self.assertEqual(qa.manifest['filtering'][split]['duplicate_context_questions_excluded'], 1)
            self.assertEqual(sum(len(rows) for rows in qa.records.values()), 3)

    def test_local_token_and_decoded_text_leak_checks(self):
        with TemporaryDirectory() as folder:
            _, _, tokenizer, _ = fixture(Path(folder))
            answer = tokenizer.encode('Amberstone').ids
            self.assertEqual(_local_answer_leak(tokenizer, [4, 5, 4], [13, 14], answer, 'Amberstone', 8),
                             'first_answer_token_in_local_context')
            # Decode catches token-boundary variants even when the supplied full token sequence differs.
            self.assertEqual(_local_answer_leak(tokenizer, [4, 5, 4], [13, 14], [7, 7], 'Amberstone', 8),
                             'answer_text_in_decoded_local_context')
            self.assertIsNone(_local_answer_leak(tokenizer, [4]*8, [13, 14], answer, 'Amberstone', 8))

    def test_cached_preparation_rejects_raw_and_prepared_hash_drift(self):
        with TemporaryDirectory() as folder:
            qa, corpus, _, pins = self.prepare(Path(folder))
            with patch('v11.data.SOURCES', pins):
                recovered = prepare_qa(qa.root, corpus, length=64, prefix=48, context_limit=8, allow_download=False)
                self.assertEqual(recovered.manifest, qa.manifest)
                prepared = qa.root/'train_records.jsonl'
                prepared.write_text(prepared.read_text()+'\n')
                with self.assertRaisesRegex(ValueError, 'Prepared QA artifacts changed'):
                    prepare_qa(qa.root, corpus, length=64, prefix=48, context_limit=8, allow_download=False)
                raw = qa.root/'train-v1.1.json'
                raw.write_bytes(raw.read_bytes()+b' ')
                with self.assertRaisesRegex(ValueError, 'Pinned official SQuAD artifact changed'):
                    prepare_qa(qa.root, corpus, length=64, prefix=48, context_limit=8, allow_download=False)

    def test_missing_sources_never_download_when_disabled(self):
        with TemporaryDirectory() as folder:
            destination, corpus, _, pins = fixture(Path(folder))
            (destination/'dev-v1.1.json').unlink()
            with patch('v11.data.SOURCES', pins), patch('v11.data.urllib.request.urlopen') as network:
                with self.assertRaises(FileNotFoundError):
                    prepare_qa(destination, corpus, length=64, prefix=48, context_limit=8, allow_download=False)
                network.assert_not_called()

    def test_canonical_bytelevel_answer_boundary_and_multilayer_exclusion(self):
        tokenizer = Tokenizer(models.BPE(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.train_from_iterator(['Arthur Arthur Beatrice filler\nQuestion: Who ruled?\nAnswer: Arthur']*10,
                     trainers.BpeTrainer(vocab_size=300, special_tokens=['[PAD]', '[BOS]', '[EOS]', '[UNK]'],
                                        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False))
        query, answer = _query_answer(tokenizer, 'Who ruled?', 'Arthur')
        canonical = tokenizer.encode('\nQuestion: Who ruled?\nAnswer: Arthur').ids
        self.assertEqual(query+answer, canonical)
        self.assertEqual(tokenizer.decode(answer).strip(), 'Arthur')
        with TemporaryDirectory() as folder:
            qa, corpus, _, pins = self.prepare(Path(folder))
            self.assertEqual(qa.history_exclusion, 16)
            self.assertEqual(qa.manifest['protocol']['maximum_evidence_end_exclusive'], 32)
            with patch('v11.data.SOURCES', pins):
                with self.assertRaisesRegex(ValueError, 'protocol changed'):
                    prepare_qa(qa.root, corpus, length=64, prefix=48, context_limit=8,
                               backbone_layers=1, allow_download=False)


if __name__ == '__main__':
    unittest.main()
