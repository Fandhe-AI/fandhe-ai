#!/usr/bin/env python3
"""judge_prepost_0150_4168.py の単体テスト（イシュー #4168）。合成 fixture だけを使い、標準ライブラリのみ。

役割: 集計器の fail-closed（欠損・重複・run 不足・diag 行・版違い・RULE 不一致・親 parity 停止・hard 不一致）が
止まるべき入力で止まり、通るべき入力で期待どおりの比・終了コードになることを固定する。
実行: python3 -I judge_prepost_0150_4168_test.py
"""
import hashlib
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import judge_prepost_0150_4168 as J  # noqa: E402

PROFILE = 'macmini'
GATE_OK = ''.join(f'round{i} gate=pass threshold=3.0 waited_s=0 load1=0.5\n' for i in range(1, 6))


def fandhe_row(task, dev, size, mode, sec, fail=0, ver=J.EXPECT_VERSION, checksum=1.0, extra=None):
    r = {'framework': 'fandhe-ai', 'task': task, 'device': dev, 'size': size, 'mode': mode, 'median_s': sec,
         'version': ver, 'checksum': checksum}
    if task == 'gemm':
        r.update(parity_total=100, parity_fail_count=fail, parity_max_abs_err=0.0, parity_max_rel_err=0.0,
                 parity_scaled_abs_bound=0.0, parity_scaled_abs_rescued=0)
    elif fail:
        r['parity_fail_count'] = fail
    if extra:
        r.update(extra)
    return r


def opp_row(fw, task, dev, size, sec, fail=0):
    r = {'framework': fw, 'task': task, 'device': dev, 'size': size, 'mode': 'fresh', 'median_s': sec, 'checksum': 2.0}
    if task == 'gemm':
        r['parity_fail_count'] = fail
    return r


def write(path, rows):
    Path(path).write_text(''.join(json.dumps(r) + '\n' for r in rows))


def make_series(root, before_sec=1.0, after_sec=0.5, mutate=None, runs=5, extra_dir=None, pymutate=None):
    """全セルに fandhe（before/after 別の median_s）と candle・pytorch（2.0・3.0 秒）を置いた系列を作る。"""
    d = Path(root) / 'series'
    for i in range(1, runs + 1):
        rd = d / f'run{i}'
        rd.mkdir(parents=True)
        arms = {'before': [], 'after': []}
        opp = []
        py = []
        for task, dev, size in J.CELLS[PROFILE]:
            for arm, sec in (('before', before_sec), ('after', after_sec)):
                for mode in ('reuse', 'fresh'):
                    arms[arm].append(fandhe_row(task, dev, size, mode, sec))
            for fw in sorted(J.required_opponents(PROFILE, task, dev, size)):
                sec = {'candle': 2.0, 'pytorch': 3.0}.get(fw, 9.0)
                (opp if fw in ('candle', 'burn') else py).append(opp_row(fw, task, dev, size, sec))
        if mutate:
            mutate(i, arms, opp)
        if pymutate:
            pymutate(i, py)
        write(rd / 'results-before.jsonl', arms['before'])
        write(rd / 'results-after.jsonl', arms['after'])
        write(rd / 'results-opp.jsonl', opp)
        write(rd / 'results-py.jsonl', py)
    if extra_dir:
        (d / extra_dir).mkdir()
    return d


def make_meta(root, rule_text='RULE\n', env_sha=None, gate=GATE_OK):
    rule = Path(root) / 'RULE.txt'
    rule.write_text(rule_text)
    sha = hashlib.sha256(rule.read_bytes()).hexdigest() if env_sha is None else env_sha
    env = Path(root) / 'env_info.txt'
    env.write_text(f'rule_sha256={sha}\n')
    g = Path(root) / 'gate.log'
    g.write_text(gate)
    return rule, env, g


def run_main(root, series, rule, env, gate):
    argv = ['--profile', PROFILE, '--label', 't', '--series-dir', str(series), '--env-info', str(env),
            '--rule', str(rule), '--gate-log', str(gate), '--ratios-out', str(Path(root) / 'ratios.tsv')]
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = J.main(argv)
    return code, buf.getvalue()


class JudgeTest(unittest.TestCase):
    def go(self, **kw):
        meta_kw = {k: kw.pop(k) for k in ('rule_text', 'env_sha', 'gate') if k in kw}
        with tempfile.TemporaryDirectory() as t:
            s = make_series(t, **kw)
            rule, env, g = make_meta(t, **meta_kw)
            return run_main(t, s, rule, env, g)

    def test_clean_series_ratios(self):
        code, out = self.go()
        self.assertEqual(code, 0, out)
        # before 比 2.0（candle 2.0 / 1.0）、after 比 4.0（2.0 / 0.5）、腕比 0.5
        self.assertIn('M-CPU-G256', out)
        self.assertIn('2.00 1 位', out)
        self.assertIn('4.00 1 位', out)
        self.assertIn('0.500', out)
        self.assertIn('見込みで 1 位', out)

    def test_loss_classification(self):
        code, out = self.go(before_sec=5.0, after_sec=5.0)  # 比 0.40 は負け
        self.assertEqual(code, 0, out)
        self.assertIn('0.40 ', out)

    def test_missing_run_dir(self):
        with tempfile.TemporaryDirectory() as t:
            s = make_series(t, runs=4)
            rule, env, g = make_meta(t)
            code, out = run_main(t, s, rule, env, g)
            self.assertEqual(code, 2, out)
            self.assertIn('run 集合', out)

    def test_extra_round_dir(self):
        with tempfile.TemporaryDirectory() as t:
            s = make_series(t, extra_dir='run6')
            rule, env, g = make_meta(t)
            code, out = run_main(t, s, rule, env, g)
            self.assertEqual(code, 2, out)
            self.assertIn('追加', out)

    def test_duplicate_row(self):
        def mut(i, arms, opp):
            if i == 2:
                arms['after'].append(dict(arms['after'][0]))
        code, out = self.go(mutate=mut)
        self.assertEqual(code, 2, out)
        self.assertIn('重複行', out)

    def test_diag_marker_row(self):
        def mut(i, arms, opp):
            if i == 1:
                arms['before'][0]['diag_breakdown'] = {'x': 1}
        code, out = self.go(mutate=mut)
        self.assertEqual(code, 2, out)
        self.assertIn('diag', out)

    def test_version_mismatch(self):
        def mut(i, arms, opp):
            for r in arms['after']:
                r['version'] = '0.13.0'
        code, out = self.go(mutate=mut)
        self.assertEqual(code, 2, out)
        self.assertIn('version', out)

    def test_fandhe_parity_fail_stops(self):
        def mut(i, arms, opp):
            if i == 3:
                for r in arms['after']:
                    if r['task'] == 'gemm' and r['mode'] == 'fresh':
                        r['parity_fail_count'] = 2
                        break
        code, out = self.go(mutate=mut)
        self.assertEqual(code, 4, out)

    def test_opponent_parity_fail_excluded(self):
        def mut(i, arms, opp):
            for r in opp:
                if r['framework'] == 'candle' and r['task'] == 'gemm' and r['size'] == 256 and r['device'] == 'cpu':
                    r['parity_fail_count'] = 9  # candle を除外 → pytorch 3.0 が最速
        code, out = self.go(mutate=mut)
        self.assertEqual(code, 0, out)
        self.assertIn('6.00 1 位', out)  # after: 3.0 / 0.5

    def test_within_arm_checksum_mismatch_is_exit3(self):
        def mut(i, arms, opp):
            if i == 4:
                for r in arms['after']:
                    if r['task'] == 'train' and r['device'] == 'cpu':
                        r['checksum'] = 99.0
        code, out = self.go(mutate=mut)
        self.assertEqual(code, 3, out)

    def test_between_arm_checksum_diff_recorded_only(self):
        def mut(i, arms, opp):
            for r in arms['after']:
                r['checksum'] = 7.0  # 腕の中では一貫、腕間で差
        code, out = self.go(mutate=mut)
        self.assertEqual(code, 0, out)
        self.assertIn('差あり', out)

    def test_rule_sha_mismatch(self):
        code, out = self.go(env_sha='0' * 64)
        self.assertEqual(code, 2, out)
        self.assertIn('rule_sha256', out)

    def test_rule_placeholder(self):
        code, out = self.go(rule_text='after = <要記入>\n')
        self.assertEqual(code, 2, out)

    def test_gate_failures(self):
        code, out = self.go(gate=GATE_OK + 'round3 GATE-TIMEOUT load1=9\n')
        self.assertEqual(code, 2, out)
        code, out = self.go(gate='round1 gate=pass\n')
        self.assertEqual(code, 2, out)

    def test_missing_opponent_file(self):
        with tempfile.TemporaryDirectory() as t:
            s = make_series(t)
            (s / 'run2' / 'results-opp.jsonl').unlink()
            rule, env, g = make_meta(t)
            code, out = run_main(t, s, rule, env, g)
            self.assertEqual(code, 2, out)

    def test_empty_python_opponents_is_undetermined(self):
        code, out = self.go(pymutate=lambda i, py: py.clear())
        self.assertEqual(code, 2, out)
        self.assertIn('必須の比較相手が欠損', out)

    def test_known_burn_metal_gap_is_tolerated(self):
        def mut(i, arms, opp):
            opp[:] = [r for r in opp if not (r['framework'] == 'burn' and r['task'] == 'gemm' and r['device'] == 'metal' and r['size'] >= 512)]
        code, out = self.go(mutate=mut)
        self.assertEqual(code, 0, out)

    def test_invalid_median_s(self):
        for bad in (0, -1.0, None, True, float('nan')):
            def mut(i, arms, opp, bad=bad):
                if i == 1:
                    arms['after'][0]['median_s'] = bad
            code, out = self.go(mutate=mut)
            self.assertEqual(code, 2, (bad, out))
            self.assertIn('median_s', out)
        def mut2(i, arms, opp):
            if i == 1:
                del opp[0]['median_s']
        code, out = self.go(mutate=mut2)
        self.assertEqual(code, 2, out)

    def test_missing_or_invalid_parity_fail_count(self):
        for bad in ('missing', None, True, -1, 1.5, '0'):
            def mut(i, arms, opp, bad=bad):
                if i == 1:
                    row = next(r for r in opp if r['task'] == 'gemm')
                    if bad == 'missing':
                        del row['parity_fail_count']
                    else:
                        row['parity_fail_count'] = bad
            code, out = self.go(mutate=mut)
            self.assertEqual(code, 2, (bad, out))
            self.assertIn('parity_fail_count', out)

    def test_fill_draft_only_ratio_column(self):
        with tempfile.TemporaryDirectory() as t:
            tsv = Path(t) / 'r.tsv'
            tsv.write_text(''.join(f'{c}\t1.00\t0.77\t負け\n' for c in sorted(J.UNREACHED)))
            draft = Path(t) / 'd.tsv'
            rows = ['# c'] + [f'{c}\tm\t<要記入（#4168）>\t分類\t理由\t出典' for c in sorted(J.UNREACHED)]
            draft.write_text('\n'.join(rows) + '\n')
            out = Path(t) / 'o.tsv'
            with redirect_stdout(io.StringIO()):
                J.fill_draft(draft, [tsv], out)
            lines = [l.split('\t') for l in out.read_text().splitlines() if not l.startswith('#')]
            self.assertTrue(all(f[2] == '0.77' and f[3:] == ['分類', '理由', '出典'] for f in lines))
            tsv.write_text('M-CPU-G256\t1\t0.5\tx\n')
            with self.assertRaises(SystemExit):
                J.fill_draft(draft, [tsv], out)

    def test_verbatim_funcs_present(self):
        names = {n.name for n in J.ast.parse(Path(J.__file__).read_text()).body if isinstance(n, J.ast.FunctionDef)}
        self.assertTrue(set(J.VERBATIM_FUNCS) <= names)


if __name__ == '__main__':
    unittest.main()
