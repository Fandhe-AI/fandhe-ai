#!/usr/bin/env python3
"""judge_bench_fandhe_symmetry_ab.py の回帰テスト（stdlib の unittest のみ。イシュー #3266）。

回帰: リポジトリ内の 0.11.0 正式系列の run1〜5 を old・new の両腕へ入れ、gen_0110_macmini.out の
相手比・区分（Mac mini 13 行・GB10 14 行）を再現できることを確認する（判定式を変えていない証拠）。
"""
import hashlib
import importlib.util
import io
import json
import re
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    'judge_bench_fandhe_symmetry_ab', Path(__file__).with_name('judge_bench_fandhe_symmetry_ab.py'))
J = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(J)

REPO = Path(__file__).resolve().parents[3]
LOGS = REPO / 'docs/perf/logs'
MAC = LOGS / 'framework-compare-0.11.0-macmini-remeasure'
MAC_B = MAC / 'macmini-series-b'
GB = LOGS / 'framework-compare-0.11.0-remeasure/gb10'
EXPECT = MAC / 'scoreboard/gen_0110_macmini.out'


def rows_of(files):
    out = []
    for f in files:
        out += [l for l in Path(f).read_text().splitlines() if l.strip()]
    return out


def make_series(dst, profile, mutate=None, nruns=5, gate=True):
    """0.11.0 の run ログから old=new の合成系列を作る。mutate(arm, run, rows)->rows で改変する。"""
    src, names = (MAC_B, ('results-macmini-0.11.0.jsonl', 'results-macmini-py-0.11.0.jsonl')) if profile == 'macmini' else \
        (GB, ('results-dgx-0.11.0.jsonl', 'results-dgx-0.11.0-extra.jsonl', 'results-dgx-py-0.11.0.jsonl'))
    for i in range(1, nruns + 1):
        rd = Path(dst) / f'run{i}'
        rd.mkdir(parents=True)
        main = rows_of([src / f'run{i}' / n for n in names[:-1]])
        py = rows_of([src / f'run{i}' / names[-1]])
        for arm in ('old', 'new'):
            rows = mutate(arm, i, list(main)) if mutate else list(main)
            (rd / f'results-{arm}.jsonl').write_text('\n'.join(rows) + '\n')
        (rd / 'results-opp.jsonl').write_text('\n'.join(main) + '\n')
        (rd / 'results-py.jsonl').write_text('\n'.join(py) + '\n')
    rule = Path(dst) / 'RULE.txt'
    rule.write_text('rule\n')
    env = Path(dst) / 'env_info.txt'
    env.write_text('rule_sha256=' + hashlib.sha256(rule.read_bytes()).hexdigest() + '\n')
    glog = Path(dst) / 'gate.log'
    glog.write_text(''.join(f'round{r} gate=pass\n' for r in range(1, 6)))
    return rule, env, glog


def run_judge(dst, profile, rule, env, glog, gate=True, ref=None):
    argv = ['--profile', profile, '--label', 't', '--series-dir', str(dst), '--env-info', str(env),
            '--rule', str(rule), '--gate-log', str(glog)]
    if gate:
        argv.append('--require-gate')
    if ref:
        argv += ['--ref-dir', str(ref)]
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = J.main(argv)
    return code, buf.getvalue()


class JudgeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_regression_macmini_matches_scoreboard(self):
        d = self.tmp / 's'
        rule, env, glog = make_series(d, 'macmini')
        code, text = run_judge(d, 'macmini', rule, env, glog)
        self.assertEqual(code, 0, text)
        # gen_0110_macmini.out: 「Mac mini（M4） gemm Metal N=256  2 位 vs candle 0.84x run別 [...]」
        n = 0
        for line in EXPECT.read_text().splitlines():
            m = re.match(r'Mac mini（M4） (gemm \w+ N=\d+|train \w+|infer \w+)\s+\S+ 位?\S*\s+vs \S+\s+([0-9.]+)x\s+run別 (\[[^\]]*\])', line)
            if not m:
                continue
            cell, ratio, runs = m.groups()
            cell = cell.replace('CPU', 'cpu').replace('Metal', 'metal').replace('CUDA', 'cuda')
            row = [l for l in text.splitlines() if l.startswith(f'| {cell} |')]
            self.assertEqual(len(row), 1, cell)
            self.assertIn(f'{float(ratio):.2f} ', row[0], cell)
            self.assertIn(runs, row[0], cell)
            n += 1
        self.assertEqual(n, 13)

    def test_regression_gb10_matches_scoreboard(self):
        d = self.tmp / 's'
        rule, env, glog = make_series(d, 'gb10')
        code, text = run_judge(d, 'gb10', rule, env, glog)
        self.assertEqual(code, 0, text)
        n = 0
        for line in EXPECT.read_text().splitlines():
            m = re.match(r'GB10\s+(gemm \w+ N=\d+|train \w+|infer \w+)\s+.+?\s+vs \S+\s+([0-9.]+)x\s+run別 (\[[^\]]*\])', line)
            if not m:
                continue
            cell, ratio, runs = m.groups()
            cell = cell.replace('CPU', 'cpu').replace('Metal', 'metal').replace('CUDA', 'cuda')
            row = [l for l in text.splitlines() if l.startswith(f'| {cell} |')]
            self.assertEqual(len(row), 1, cell)
            self.assertIn(f'{float(ratio):.2f} ', row[0], cell)
            self.assertIn(runs, row[0], cell)
            n += 1
        self.assertEqual(n, 14)

    def test_checksum_one_bit_mismatch_exit3(self):
        def mut(arm, run, rows):
            if arm == 'new' and run == 3:
                rows = [l.replace('"checksum":237.546660', '"checksum":237.546661') for l in rows]
            return rows
        d = self.tmp / 's'
        rule, env, glog = make_series(d, 'macmini', mut)
        code, text = run_judge(d, 'macmini', rule, env, glog)
        self.assertEqual(code, 3, text)
        self.assertIn('hard 条件の不一致', text)

    def test_parity_field_mismatch_exit3(self):
        def mut(arm, run, rows):
            if arm == 'old' and run == 1:
                rows = [l.replace('"parity_total":65536', '"parity_total":65537') for l in rows]
            return rows
        d = self.tmp / 's'
        rule, env, glog = make_series(d, 'macmini', mut)
        self.assertEqual(run_judge(d, 'macmini', rule, env, glog)[0], 3)

    def test_missing_run_exit2(self):
        d = self.tmp / 's'
        rule, env, glog = make_series(d, 'macmini')
        shutil.rmtree(d / 'run5')
        self.assertEqual(run_judge(d, 'macmini', rule, env, glog)[0], 2)

    def test_duplicate_row_exit2(self):
        def mut(arm, run, rows):
            return rows + [rows[0]] if arm == 'new' and run == 2 else rows
        d = self.tmp / 's'
        rule, env, glog = make_series(d, 'macmini', mut)
        code, text = run_judge(d, 'macmini', rule, env, glog)
        self.assertEqual(code, 2)
        self.assertIn('重複行', text)

    def test_gate_failure_exit2(self):
        d = self.tmp / 's'
        rule, env, glog = make_series(d, 'macmini')
        glog.write_text('round1 gate=pass\nGATE-TIMEOUT before run2\n')
        self.assertEqual(run_judge(d, 'macmini', rule, env, glog)[0], 2)
        # ゲートなし系列（系列 A）は gate.log を要求しない
        self.assertEqual(run_judge(d, 'macmini', rule, env, glog, gate=False)[0], 0)

    def test_rule_sha_mismatch_exit2(self):
        d = self.tmp / 's'
        rule, env, glog = make_series(d, 'macmini')
        rule.write_text('changed\n')
        code, text = run_judge(d, 'macmini', rule, env, glog)
        self.assertEqual(code, 2)
        self.assertIn('rule_sha256', text)

    def test_all_opponents_parity_fail_is_undetermined(self):
        def mut(arm, run, rows):
            return rows
        d = self.tmp / 's'
        rule, env, glog = make_series(d, 'macmini', mut)
        for i in range(1, 6):
            for name in ('results-opp.jsonl', 'results-py.jsonl'):
                p = d / f'run{i}' / name
                out = []
                for l in p.read_text().splitlines():
                    r = json.loads(l)
                    if r['task'] == 'train' and r['device'] == 'cpu' and r['framework'] != 'fandhe-ai':
                        r['parity_fail_count'] = 5
                    out.append(json.dumps(r))
                p.write_text('\n'.join(out) + '\n')
        code, text = run_judge(d, 'macmini', rule, env, glog)
        self.assertEqual(code, 2)
        self.assertIn('有効な比較相手がない train cpu', text)

    def test_arm_ratio_median_and_order(self):
        def mut(arm, run, rows):
            if arm != 'new':
                return rows
            out = []
            for l in rows:
                r = json.loads(l)
                if (r['framework'], r['task'], r['device'], r['size'], r['mode']) == ('fandhe-ai', 'gemm', 'cpu', 256, 'reuse'):
                    r['median_s'] = r['median_s'] * 0.5
                out.append(json.dumps(r))
            return out
        d = self.tmp / 's'
        rule, env, glog = make_series(d, 'macmini', mut)
        code, text = run_judge(d, 'macmini', rule, env, glog)
        row = [l for l in text.splitlines() if l.startswith('| gemm cpu N=256 |')][0]
        self.assertIn('0.500 [0.50 0.50 0.50 0.50 0.50]', row)

    def test_reference_column(self):
        d = self.tmp / 's'
        rule, env, glog = make_series(d, 'macmini')
        code, text = run_judge(d, 'macmini', rule, env, glog, ref=MAC_B)
        row = [l for l in text.splitlines() if l.startswith('| gemm cpu N=256 |')][0]
        self.assertNotIn('| — |', row.split('|')[2] + '|')

    def test_empty_py_results_is_undetermined(self):
        d = self.tmp / 's'
        rule, env, glog = make_series(d, 'macmini')
        (d / 'run2' / 'results-py.jsonl').write_text('')
        code, text = run_judge(d, 'macmini', rule, env, glog)
        self.assertEqual(code, 2, text)
        self.assertIn('比較相手の行が欠損', text)

    def test_hard_mismatch_wins_over_undetermined_cell(self):
        def mut(arm, run, rows):
            if arm == 'new' and run == 1:
                rows = [l.replace('"parity_total":65536', '"parity_total":65537') for l in rows]
            return rows
        d = self.tmp / 's'
        rule, env, glog = make_series(d, 'macmini', mut)
        (d / 'run2' / 'results-py.jsonl').write_text('')
        code, text = run_judge(d, 'macmini', rule, env, glog)
        self.assertEqual(code, 3, text)

    def test_hard_fields_removed_everywhere_is_undetermined(self):
        def mut(arm, run, rows):
            out = []
            for l in rows:
                r = json.loads(l)
                for f in J.HARD_FIELDS:
                    r.pop(f, None)
                out.append(json.dumps(r))
            return out
        d = self.tmp / 's'
        rule, env, glog = make_series(d, 'macmini', mut)
        code, text = run_judge(d, 'macmini', rule, env, glog)
        self.assertEqual(code, 2, text)
        self.assertIn('必須項目の欠落', text)

    def _drop_row(self, arm_name, run_no, mode):
        def mut(arm, run, rows):
            if arm == arm_name and run == run_no:
                rows = [l for l in rows if not (
                    '"framework":"fandhe-ai"' in l and f'"mode":"{mode}"' in l
                    and '"task":"gemm"' in l and '"device":"cpu"' in l and '"size":256' in l)]
            return rows
        d = self.tmp / 's'
        rule, env, glog = make_series(d, 'macmini', mut)
        return run_judge(d, 'macmini', rule, env, glog)

    def test_missing_fresh_row_in_one_arm_is_undetermined_not_hard(self):
        code, text = self._drop_row('old', 2, 'fresh')
        self.assertEqual(code, 2, text)
        self.assertIn('行欠損', text)
        self.assertNotIn('hard 条件の不一致', text)

    def test_missing_reuse_row_in_one_arm_is_undetermined_not_hard(self):
        code, text = self._drop_row('new', 4, 'reuse')
        self.assertEqual(code, 2, text)
        self.assertNotIn('hard 条件の不一致', text)


if __name__ == '__main__':
    unittest.main()
