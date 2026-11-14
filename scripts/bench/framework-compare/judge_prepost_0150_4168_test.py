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
GATE_OK = ''.join(f'round{i} gate=pass threshold=3.0 waited_s=0 load1=0.5\nround{i} foreign-before=clean\nround{i} foreign-after=clean\n'
                  for i in range(1, 6))
CONSTS = ['crates/a/src/x.rs:FOO_ENABLED=true', 'crates/b/src/y.rs:BAR_ENABLED=cfg!(all(target_os="linux"))']


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


def make_tools(root):
    """一時ツリーに TOOL_FILES と同じ相対パスの道具を置き、RULE の [道具の sha256] ブロックの行を返す。"""
    lines = []
    for rel in J.TOOL_FILES:
        f = Path(root) / 'tools' / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        # 判定器だけは実体を写す（CLI 経路は --tool-root でも実行中の判定器自身を RULE の sha256 と照合するため）
        if rel == J.JUDGE_REL:
            f.write_bytes(Path(J.__file__).read_bytes())
        else:
            f.write_text(f'# dummy {rel}\n')
        lines.append(f'{hashlib.sha256(f.read_bytes()).hexdigest()}  {rel}')
    return lines


def make_meta(root, rule_text='RULE\n', env_sha=None, gate=GATE_OK, tool_lines=None, consts=None, env_extra=None, env_drop=()):
    tools = make_tools(root) if tool_lines is None else tool_lines
    cons = CONSTS if consts is None else consts
    body = rule_text + '# >>> expected-consts\n' + ''.join(c + '\n' for c in cons) + '# <<< expected-consts\n'
    body += '# >>> tool-sha256\n' + ''.join(t + '\n' for t in tools) + '# <<< tool-sha256\n'
    rule = Path(root) / 'RULE.txt'
    rule.write_text(body)
    sha = hashlib.sha256(rule.read_bytes()).hexdigest() if env_sha is None else env_sha
    csha = hashlib.sha256(('\n'.join(cons) + '\n').encode()).hexdigest()
    kv = {'rule_sha256': sha, 'tool_sha256_check': 'ok', 'expected_consts_sha256': csha,
          'fandhe_ai_rust_ai_env_vars': '0', 'rayon_num_threads': 'unset', 'smoke': '0'}
    kv.update(env_extra or {})
    env = Path(root) / 'env_info.txt'
    env.write_text(''.join(f'{k}={v}\n' for k, v in kv.items() if k not in env_drop))
    g = Path(root) / 'gate.log'
    g.write_text(gate)
    return rule, env, g


def run_main(root, series, rule, env, gate):
    argv = ['--profile', PROFILE, '--label', 't', '--series-dir', str(series), '--env-info', str(env),
            '--rule', str(rule), '--gate-log', str(gate), '--ratios-out', str(Path(root) / 'ratios.tsv'),
            '--tool-root', str(Path(root) / 'tools')]
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = J.main(argv)
    return code, buf.getvalue()


class JudgeTest(unittest.TestCase):
    def go(self, **kw):
        meta_kw = {k: kw.pop(k) for k in ('rule_text', 'env_sha', 'gate', 'tool_lines', 'consts', 'env_extra', 'env_drop') if k in kw}
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

    def test_foreign_gate_tokens_required_on_macmini(self):
        # load1 の gate=pass だけでは足りない（Mac mini は各ラウンドの前後の他ジョブゲートが CLEAN であること）
        plain = ''.join(f'round{i} gate=pass threshold=3.0 waited_s=0 load1=0.5\n' for i in range(1, 6))
        code, out = self.go(gate=plain)
        self.assertEqual(code, 2, out)
        self.assertIn('foreign-before=clean', out)
        code, out = self.go(gate=GATE_OK.replace('round4 foreign-after=clean\n', ''))
        self.assertEqual(code, 2, out)
        self.assertIn('round4 の foreign-after=clean', out)
        code, out = self.go(gate=GATE_OK + 'round2 GATE-INVALID reason=foreign-after\n')
        self.assertEqual(code, 2, out)

    def test_tool_sha256_mismatch_in_rule(self):
        with tempfile.TemporaryDirectory() as t:
            lines = make_tools(t)
            lines[3] = '0' * 64 + '  ' + J.TOOL_FILES[3]
            s = make_series(t)
            rule, env, g = make_meta(t, tool_lines=lines)
            code, out = run_main(t, s, rule, env, g)
        self.assertEqual(code, 2, out)
        self.assertIn('道具の sha256 が RULE と不一致', out)
        self.assertIn('foreign_job_gate.py', out)

    def test_tool_file_modified_after_rule(self):
        with tempfile.TemporaryDirectory() as t:
            lines = make_tools(t)
            (Path(t) / 'tools' / J.TOOL_FILES[2]).write_text('改変\n')
            s = make_series(t)
            rule, env, g = make_meta(t, tool_lines=lines)
            code, out = run_main(t, s, rule, env, g)
        self.assertEqual(code, 2, out)
        self.assertIn('orchestrate.sh', out)

    def test_tool_block_missing_or_wrong_set(self):
        code, out = self.go(tool_lines=[])
        self.assertEqual(code, 2, out)
        self.assertIn('[道具の sha256]', out)
        with tempfile.TemporaryDirectory() as t:
            lines = make_tools(t)[:-1]
            s = make_series(t)
            rule, env, g = make_meta(t, tool_lines=lines)
            code, out = run_main(t, s, rule, env, g)
        self.assertEqual(code, 2, out)
        self.assertIn('TOOL_FILES と一致しない', out)

    def test_judge_self_hash_checked(self):
        with tempfile.TemporaryDirectory() as t:
            lines = make_tools(t)
            f = Path(t) / 'judge_copy.py'
            f.write_text('# 別の判定器\n')
            block = '\n'.join(lines)
            self.assertIn('実行中の判定器', ' '.join(J.check_tool_sha256(lines, Path(t) / 'tools', f)))
            self.assertEqual(J.check_tool_sha256(lines, Path(t) / 'tools', Path(t) / 'tools' / J.JUDGE_REL), [], block)

    def test_env_info_env_var_records(self):
        code, out = self.go(env_extra={'fandhe_ai_rust_ai_env_vars': '1'})
        self.assertEqual(code, 2, out)
        self.assertIn('fandhe_ai_rust_ai_env_vars', out)
        code, out = self.go(env_extra={'rayon_num_threads': 'set'})
        self.assertEqual(code, 2, out)
        self.assertIn('rayon_num_threads', out)
        code, out = self.go(env_drop=('fandhe_ai_rust_ai_env_vars',))
        self.assertEqual(code, 2, out)

    def test_smoke_output_refused(self):
        # --smoke の出力は計測の代わりにならない: env_info の smoke=1・smoke の欠落・ディレクトリ名のどれでも拒否
        code, out = self.go(env_extra={'smoke': '1'})
        self.assertEqual(code, 2, out)
        self.assertIn('smoke', out)
        code, out = self.go(env_drop=('smoke',))
        self.assertEqual(code, 2, out)
        with tempfile.TemporaryDirectory() as t:
            s = make_series(t)
            s = s.rename(Path(t) / 'ab4168-smoke-x')
            rule, env, g = make_meta(t)
            code, out = run_main(t, s, rule, env, g)
        self.assertEqual(code, 2, out)
        self.assertIn('--smoke の出力ディレクトリ', out)

    def test_expected_consts_block_and_sha(self):
        code, out = self.go(consts=[])
        self.assertEqual(code, 2, out)
        self.assertIn('[期待するゲート定数]', out)
        code, out = self.go(env_extra={'expected_consts_sha256': '0' * 64})
        self.assertEqual(code, 2, out)
        self.assertIn('expected_consts_sha256', out)

    def test_blank_line_in_blocks_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            lines = make_tools(t) + ['']
            s = make_series(t)
            rule, env, g = make_meta(t, tool_lines=lines)
            code, out = run_main(t, s, rule, env, g)
        self.assertEqual(code, 2, out)
        self.assertIn('空行', out)
        code, out = self.go(consts=CONSTS + [''])
        self.assertEqual(code, 2, out)
        self.assertIn('空行', out)

    def test_runner_print_output_accepted_by_judge(self):
        # ランナーの --print-tool-sha256 の実出力（リポジトリの実ファイル）を、そのまま判定器が受け入れること
        import os
        import subprocess
        root = J.DEFAULT_ROOT
        env = {k: v for k, v in os.environ.items() if not k.startswith(('FANDHE_AI_', 'RUST_AI_'))}
        env.update(AB_BASE_ROOT=str(root), AB_RUNNER_ROOT=str(root))
        r = subprocess.run(['bash', str(root / 'scripts/bench/framework-compare/run_ab_prepost_0150_4168.sh'), '--print-tool-sha256'],
                           env=env, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        lines = r.stdout.rstrip('\n').split('\n')
        self.assertEqual(J.check_tool_sha256(lines, root, J.__file__), [])

    def test_tool_files_match_runner(self):
        # ランナーの TOOL_FILES（シェル）と判定器の TOOL_FILES が同じ集合・同じ順であること（ずれると実体化したブロックを片方が拒否する）
        src = (Path(__file__).resolve().parent / 'run_ab_prepost_0150_4168.sh').read_text()
        body = src.split('TOOL_FILES="', 1)[1].split('"\n', 1)[0]
        base = 'scripts/bench/framework-compare'
        logs = 'docs/perf/logs/prepost-0150-4168'
        got = [l.replace('$FC_REL', base).replace('$LOGS_REL', logs) for l in body.split('\n')]
        self.assertEqual(got, list(J.TOOL_FILES))

    def test_clean_series_with_all_meta_passes(self):
        code, out = self.go()
        self.assertEqual(code, 0, out)

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
