"""LoCoMo F1 subset, adapted from snap-research/locomo at 3eb6f2c.

Original authors: Adyasha Maharana et al. (ACL 2024).
Source: task_eval/evaluation.py, git blob 8f597dd687e66832da1f6f04a169e05622049576.
CC BY-NC 4.0; see LICENSE.locomo.txt. Unrelated scorers/imports removed.
"""
from collections import Counter
import string
import regex
import numpy as np
from nltk.stem import PorterStemmer
ps = PorterStemmer()

def normalize_answer(s):

    s = s.replace(',', "")
    def remove_articles(text):
        # return regex.sub(r'\b(a|an|the)\b', ' ', text)
        return regex.sub(r'\b(a|an|the|and)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
    prediction_tokens = [ps.stem(w) for w in normalize_answer(prediction).split()]
    ground_truth_tokens = [ps.stem(w) for w in normalize_answer(ground_truth).split()]
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    # print('# F1 #', prediction, ' | ', ground_truth, ' #', precision, recall, f1)
    # return recall
    return f1

def f1(prediction, ground_truth):
    predictions = [p.strip() for p in prediction.split(',')]
    ground_truths = [g.strip() for g in ground_truth.split(',')]
    # print('# F1 [multi-answer]#', predictions, ' | ', ground_truths, ' #', np.mean([max([f1_score(prediction, gt) for prediction in predictions]) for gt in ground_truths]))
    return np.mean([max([f1_score(prediction, gt) for prediction in predictions]) for gt in ground_truths])

def score(prediction, qa):
    category=qa['category']
    if category==5:
        return float('no information available' in prediction.lower() or 'not mentioned' in prediction.lower())
    answer=str(qa['answer'])
    if category==3: answer=answer.split(';')[0].strip()
    if category==1: return float(f1(prediction,answer))
    if category in (2,3,4): return float(f1_score(prediction,answer))
    raise ValueError('Unknown LoCoMo category')
