"""Repeatable CPU checks against the explicit v0.7 design contract."""
import datetime as dt
import json
from pathlib import Path
import sys
import time
import unittest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

import torch

from v07.common import Console, file_hash, source_hashes, write


def flatten(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from flatten(item)
        else:
            yield item


def main():
    torch.set_num_threads(1)
    specification = PROJECT/'configs/v07/conformance.json'
    contract = json.loads(specification.read_text())
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = PROJECT/'results/v07/conformance/checks'/stamp
    output.mkdir(parents=True)
    ui = Console(output/'events.jsonl')
    files = [specification, Path(__file__).resolve()]
    files += [PROJECT/name for name in contract['sources']]
    files += [PROJECT/Path(*name.split('.')).with_suffix('.py') for name in contract['test_modules']]
    evidence_hashes = {str(path.relative_to(PROJECT)): file_hash(path) for path in files}
    runtime_hashes = source_hashes()
    suite = unittest.defaultTestLoader.loadTestsFromNames(contract['test_modules'])
    ids = {test.id() for test in flatten(suite)}
    missing = sorted({test for row in contract['requirements'] for test in row['tests'] if test not in ids})
    if missing:
        write(output/'report.json', dict(status='failed', missing_tests=missing))
        raise RuntimeError(f'Contract refers to missing checks: {missing}')
    ui.stage('v0.7 设计一致性检查', f'{len(contract["requirements"])} 项实现要求；CPU单线程；详细过程写入日志',
             total=suite.countTestCases(), unit='项')
    outcomes = {}

    class Result(unittest.TextTestResult):
        def addSuccess(self, test):
            super().addSuccess(test); outcomes[test.id()] = 'passed'

        def addFailure(self, test, error):
            super().addFailure(test, error); outcomes[test.id()] = 'failed'

        def addError(self, test, error):
            super().addError(test, error); outcomes[test.id()] = 'error'

        def addSkip(self, test, reason):
            super().addSkip(test, reason); outcomes[test.id()] = 'not_executed'

        def stopTest(self, test):
            super().stopTest(test); ui.update(1, 检查=test.id().rsplit('.', 1)[-1][:35])

    started = time.perf_counter()
    try:
        with (output/'unit_tests.log').open('w') as log:
            result = unittest.TextTestRunner(stream=log, verbosity=2, resultclass=Result).run(suite)
    finally:
        ui.close()
    stable = (source_hashes() == runtime_hashes and
              all(file_hash(PROJECT/name) == expected for name, expected in evidence_hashes.items()))
    requirements = [row | dict(status='passed' if all(outcomes.get(test) == 'passed' for test in row['tests'])
                               else 'unverified_or_failed', test_outcomes={test: outcomes.get(test) for test in row['tests']})
                    for row in contract['requirements']]
    success = result.wasSuccessful() and stable and all(row['status'] == 'passed' for row in requirements)
    report = dict(status='passed' if success else 'failed', scope='Implementation correspondence and CPU mechanism/engineering checks',
                  tests_run=result.testsRun, tests_skipped=len(result.skipped), seconds=time.perf_counter()-started,
                  sources_unchanged_during_check=stable, runtime_source_hashes=runtime_hashes,
                  evidence_hashes=evidence_hashes, requirements=requirements, test_outcomes=outcomes,
                  empirical_questions=contract['empirical_questions'], explicit_exclusions=contract['explicit_exclusions'],
                  gpu_training='not_executed', formal_training='not_executed', output=str(output))
    write(output/'report.json', report)
    write(PROJECT/'results/v07/conformance/latest.json', dict(status=report['status'], report=str(output/'report.json')))
    lines = ['# v0.7 代码与设计复核', '', f'状态：{report["status"]}；测试：{result.testsRun}；跳过：{len(result.skipped)}。', '',
             '| 要求 | 对照内容 | 验收 |', '|---|---|---|']
    lines += [f'| {row["id"]} | {row["requirement"]} | {row["status"]} |' for row in requirements]
    lines += ['', '这些检查验证实现与机制性质；科学能力问题需正式配对实验，见report.json中的empirical_questions。']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    ui.stage('设计复核完成' if success else '设计复核失败', f'{sum(row["status"] == "passed" for row in requirements)}/{len(requirements)} 要求通过；{output}/report.json')
    return 0 if success else 1


if __name__ == '__main__':
    raise SystemExit(main())
