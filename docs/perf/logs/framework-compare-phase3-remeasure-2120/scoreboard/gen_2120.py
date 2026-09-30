#!/usr/bin/env python3
"""framework-compare スコアボード（HTML）を JSONL から生成する（イシュー #2120・Phase 3 前後再計測版）。

役割: docs/perf/logs/framework-compare-precision-class-remeasure-1988/scoreboard/gen_1988.py の派生。
  判定規則（順位・僅差・判定不能 `parity_fail_count > 0`）・表構造は不変。2 つのモードを持つ:

- 既定（#2120 正式モード）: `--m4-agg`／`--gb-agg`（aggregate.py の集計 JSON `<prefix>-aggregate.json`）が必須。
  - 「前比」表示・カード・`--tsv` の B→C 比は集計 JSON の `bc_med`（RULE.txt 判定 3: 同一 run 内比 C/B の 5 run 中央値。
    infer も時間比）をそのまま使い、本スクリプトでは再計算しない（RULE.txt 判定 3 追記。単一情報源）。
    カードの ms 値は判定 2 の採用行（集計 JSON の picked）で、その比は表示しない。
  - `--m4`／`--gb` は集計 JSON が記録した `--arm` 腕（既定 C。腕 B の倍率を得るときは B）の派生 JSONL の sha256 と
    一致しなければ停止する（集計後の差し替え・腕の取り違えを拒否）。集計 JSON の status が formal でなければ停止する。
  - `--gb-extra`／`--gb-py` は Python FW（PyTorch／TensorFlow／SciPy）行のみ受理する（後勝ちマージで fandhe-ai・candle・
    burn の集計値を上書きさせない）。行は aggregate.py と同じ厳格読み取り（NaN・重複キー・型）で検査する。
  - 計測条件注記へ集計 JSON の専有ゲート結果（M4 Max の不通過 run を含む）を自動で併記する。
  - `--m4-prev`／`--gb-prev`／`--prev-label` は使わない（指定すると停止）。
- `--legacy-1988`: gen_1988.py と同じ入力・出力（`--prev-label`・`--m4-prev`／`--gb-prev` で前比を自前計算）。
  `--body body_1988.html --prev-label 0.8.0` と #1988 の元入力で実行した HTML・標準出力は gen_1988.py の出力と
  byte 同一（非後退確認。`--self-test` が cmp で検査する）。#2120 の正式スコアボードには使わない。

共通の追加引数: `--main-label`（判定列の見出し）・`--body`・`--style`・`--tsv`（検証出力。正式モードでは B→C 比
・q1／q3・非後退・checksum 一致の列を集計 JSON から追加する）。
呼び出し元: aggregate.py の派生 JSONL と集計 JSON（README・docs/perf/framework-compare-phase3-remeasure.md §5）。
"""
import argparse
import json
import html
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# 集計の厳格読み取り・fixture・sha256 は aggregate.py（親ディレクトリ）を単一の実装として使う。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import aggregate as AGGMOD  # noqa: E402

REPO = Path('<repo>')
RAW = REPO / 'scripts/bench/framework-compare/results/raw'
DGX08 = REPO / 'docs/perf/logs/lowlayer-diagnosis-2026-09-12/dgx'

# docs/perf/logs ディレクトリ。本ファイルは docs/perf/logs/<2120 dir>/scoreboard/gen_2120.py にあるため
# parents[0]=scoreboard, parents[1]=2120 dir, parents[2]=logs。parent の段数に依らず名前で検証して誤読・移設を検出する。
LOGS_DIR = Path(__file__).resolve().parents[2]
if LOGS_DIR.name != 'logs' or LOGS_DIR.parent.name != 'perf':
    raise SystemExit(f'error: LOGS_DIR が docs/perf/logs ではありません（{LOGS_DIR}）。本スクリプトの配置を確認してください')
DEFAULT_STYLE = LOGS_DIR / 'framework-compare-0.9.0-remeasure' / 'scoreboard' / 'style.css'
DEFAULT_BODY = Path(__file__).resolve().with_name('body_2120.html')



LEGACY_GB_PREV = [
    str(DGX08 / 'results-dgx-0.8.0.jsonl'),
    str(DGX08 / 'results-dgx-0.8.0-extra.jsonl'),
    str(DGX08 / 'results-dgx-py-0.8.0.jsonl'),
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--m4', required=True, help='M4 Max の主入力 JSONL（正式モードは aggregate.py の <prefix>-<arm>-full.jsonl）')
    p.add_argument('--m4-prev', default=None, help='[--legacy-1988 のみ] 前比の比較元 M4 Max JSONL（既定 0.8.0）')
    p.add_argument('--gb', required=True, help='GB10 の主入力 JSONL（正式モードは aggregate.py の <prefix>-<arm>-full.jsonl）')
    p.add_argument('--gb-extra', default=None, help='GB10 追加 JSONL（--legacy-1988 では必須。正式モードは Python FW 行のみ・任意）')
    p.add_argument('--gb-py', required=True, help='GB10 Python 3 FW（PyTorch/TensorFlow/SciPy）JSONL')
    p.add_argument('--gb-prev', nargs='+', default=None, help='[--legacy-1988 のみ] 前比の比較元 GB10 JSONL 群（既定 0.8.0 の 3 本）')
    p.add_argument('--m4-agg', default=None, help='[正式モード必須] aggregate.py --machine m4max の <prefix>-aggregate.json')
    p.add_argument('--gb-agg', default=None, help='[正式モード必須] aggregate.py --machine gb10 の <prefix>-aggregate.json')
    p.add_argument('--arm', choices=('B', 'C'), default='C', help='[正式モード] --m4／--gb に渡した腕（既定 C。sha256 で照合）')
    p.add_argument('--legacy-1988', action='store_true', help='gen_1988.py 互換モード（非後退確認用。#2120 の正式スコアボードには使わない）')
    p.add_argument('--out', default='fandhe-ai-0.9.0-scoreboard.html', help='出力 HTML パス')
    p.add_argument('--m4-load', default='', help='M4 Max の負荷条件注記文（計測条件節に埋め込む）')
    p.add_argument('--gb-load', default='', help='GB10 の負荷条件注記文（計測条件節に埋め込む）')
    p.add_argument('--body', default=str(DEFAULT_BODY), help='本文テンプレート HTML')
    # 既定 CSS は LOGS_DIR（docs/perf/logs）直下の 0.9.0 版 CSS。
    p.add_argument('--style', default=str(DEFAULT_STYLE), help='CSS（0.9.0 版と同一ファイル）')
    p.add_argument('--prev-label', default=None, help='[--legacy-1988 のみ] 「前比」列の比較元ラベル（既定 0.8.0）')
    p.add_argument('--main-label', default='0.9.0', help='判定列の見出しに使う fandhe-ai 側ラベル（既定 0.9.0。#2120 では "Phase 3 後（HEAD）"）')
    p.add_argument('--tsv', default='', help='検証出力を TSV でも書くパス（任意）')
    p.add_argument('--self-test', action='store_true', help='既定の CSS・本文の存在と、集計 JSON との一致・拒否条件・legacy の byte 同一を検査して終了する（他の必須引数は不要）')
    return p.parse_args()


def _require_file(path, flag):
    if not Path(path).is_file():
        raise SystemExit(f'error: {flag} が存在しません: {path}')


PY_FWS = ('pytorch', 'tensorflow', 'scipy')  # --gb-extra／--gb-py で受理する FW（RULE.txt: Python FW 行は既存値を流用）
HERE = Path(__file__).resolve()


def _die(msg):
    raise SystemExit(f'error: {msg}')


def _self_test():
    """RULE.txt の表示・転記条項の self-test（表駆動）。aggregate.py の fixture から集計 JSON を作り、本スクリプトを
    サブプロセスで起動して、前比が bc_med と一致すること・前提を満たさない入力で停止すること・legacy の byte 同一を検査する。"""
    import contextlib
    import io
    for flag, path in (('--style（既定）', DEFAULT_STYLE), ('--body（既定）', DEFAULT_BODY)):
        _require_file(path, flag)
        print(f'OK {flag}: {path}')

    def run(args):
        return subprocess.run([sys.executable, str(HERE)] + args, capture_output=True, text=True)

    with tempfile.TemporaryDirectory() as td:
        def prep(tag, m4_mut=None, gb_mut=None, c_factors=(2.0, 1.0, 0.5, 1.0, 1.0)):
            # c_factors 既定: 同一 run 内比 C/B = (2, 1, 0.5, 1, 1) → 中央値 1.00。採用行（run 中央値）同士の比は
            # C=run1 or run2（2.0×base）÷ B=run3（3.0×base）= 0.67 となり、自前再計算すると必ず食い違う。
            base = os.path.join(td, tag)
            out = {}
            for machine, mut in (('m4max', m4_mut), ('gb10', gb_mut)):
                logs = os.path.join(base, machine)
                os.makedirs(logs)
                AGGMOD.make_fixture(logs, machine, c_factors=c_factors)
                if mut:
                    mut(logs)
                with contextlib.redirect_stderr(io.StringIO()) as err:
                    rc, res = AGGMOD.run_aggregate(logs, machine, '', os.path.join(logs, 'results'), quiet=True)
                assert rc == 0, f'{tag}: fixture の集計が失敗: {err.getvalue()}'
                out[machine] = (os.path.join(logs, 'results'), res)
            py = os.path.join(base, 'py.jsonl')
            with open(py, 'w') as w:
                w.write(json.dumps({'framework': 'pytorch', 'version': 'x', 'task': 'gemm', 'device': 'cuda', 'size': 256,
                                    'median_s': 0.5, 'mode': 'fresh', 'parity_total': 65536, 'parity_fail_count': 0}) + '\n')
            return out, py, base

        def args_for(out, py, base, arm='C', m4_file=None, swap_agg=False):
            m4p, gbp = out['m4max'][0], out['gb10'][0]
            m4a, gba = f'{m4p}-aggregate.json', f'{gbp}-aggregate.json'
            if swap_agg:
                m4a, gba = gba, m4a
            return ['--m4', m4_file or f'{m4p}-{arm}-full.jsonl', '--m4-agg', m4a, '--gb', f'{gbp}-{arm}-full.jsonl',
                    '--gb-agg', gba, '--gb-py', py, '--arm', arm, '--main-label', 'Phase 3 後（HEAD）',
                    '--out', os.path.join(base, f'out-{arm}.html'), '--tsv', os.path.join(base, f'out-{arm}.tsv')]

        def edit_json(path, fn):
            d = json.load(open(path))
            fn(d)
            with open(path, 'w') as w:
                json.dump(d, w, ensure_ascii=False)

        def check_tsv_matches(out, base, arm):
            rows = [l.split('\t') for l in open(os.path.join(base, f'out-{arm}.tsv')).read().splitlines()]
            aggs = {'M4 Max': out['m4max'][1], 'GB10': out['gb10'][1]}
            assert len(rows) == len(M4_ROWS_SPEC) + len(GB_ROWS_SPEC), len(rows)
            for row in rows:
                mach, t, d, n = row[0], row[6], row[7], int(row[8])
                c = [x for x in aggs[mach]['cells'] if (x['task'], x['device'], x['size'], x['mode']) == (t, d, n, 'reuse')][0]
                assert row[9] == f'{c["bc_med"]:.4f}' and row[10] == f'{c["bc_q1"]:.4f}' and row[11] == f'{c["bc_q3"]:.4f}', row
                assert row[12] == ('非後退' if c['nonreg'] else '後退'), row
                if not row[5].startswith('判定不能'):
                    assert f'{c["bc_med"]:.2f}×' in row[5], row
                assert abs(c['bc_med'] - 1.0) < 1e-12, c['bc_med']  # 採用行同士の比（0.67）ではない

        def c_parity_fail(logs):
            for r in range(1, AGGMOD.NRUN + 1):
                p = os.path.join(logs, f'run{r}', 'C.jsonl')
                ls = open(p).read().splitlines()
                ls = [l.replace('"parity_fail_count": 0', '"parity_fail_count": 5') if ('"metal"' in l and '"size": 256' in l and '"reuse"' in l and '"gemm"' in l) else l for l in ls]
                open(p, 'w').write('\n'.join(ls) + '\n')

        def m4_gate_fail(logs):
            p = os.path.join(logs, 'gate.log')
            ls = [l.replace('load1=7.50', 'load1=9.10').replace(' pass', ' fail') if l.startswith('run=3 ') else l
                  for l in open(p).read().splitlines()]
            open(p, 'w').write('\n'.join(ls) + '\n')

        ok = lambda r: r.returncode == 0  # noqa: E731
        cases = []

        def case(cid, fn):
            cases.append((cid, fn))

        # 判定 3・6（前比・22 セル転記の B→C 比は集計 JSON の bc_med。gen は再計算しない）
        def c_match():
            out, py, base = prep('match')
            r = run(args_for(out, py, base))
            assert ok(r), r.stderr
            check_tsv_matches(out, base, 'C')
            h = open(os.path.join(base, 'out-C.html')).read()
            assert 'B→C 比 1.00×' in h and '0.67' not in h, 'HTML に bc_med 以外の比が出ている'
        case('J3/J6-前比は bc_med（採用行比 0.67 ではない）', c_match)

        def c_arm_b():
            out, py, base = prep('armb')
            r = run(args_for(out, py, base, arm='B'))
            assert ok(r), r.stderr
            check_tsv_matches(out, base, 'B')
        case('J6-腕 B 実行でも B→C 比は同じ bc_med', c_arm_b)

        def c_sha():
            out, py, base = prep('sha')
            with open(f'{out["gb10"][0]}-C-full.jsonl', 'a') as w:
                w.write('\n')
            r = run(args_for(out, py, base))
            assert not ok(r) and 'sha256' in r.stderr, r.stderr
        case('SRC-集計後に派生 JSONL を改変 → 停止', c_sha)

        def c_arm_mix():
            out, py, base = prep('armmix')
            r = run(args_for(out, py, base, m4_file=f'{out["m4max"][0]}-B-full.jsonl'))
            assert not ok(r) and 'sha256' in r.stderr, r.stderr
        case('SRC-腕の取り違え（--arm C に腕 B の JSONL）→ 停止', c_arm_mix)

        def c_formal():
            out, py, base = prep('formal')
            edit_json(f'{out["gb10"][0]}-aggregate.json', lambda d: d.update(status='reference'))
            r = run(args_for(out, py, base))
            assert not ok(r) and 'formal' in r.stderr, r.stderr
        case('PRE-集計 JSON が formal でない → 停止', c_formal)

        def c_gate_json():
            out, py, base = prep('gatejson')
            edit_json(f'{out["gb10"][0]}-aggregate.json', lambda d: d['gate']['runs'][1].update(status='fail'))
            r = run(args_for(out, py, base))
            assert not ok(r) and '専有ゲート' in r.stderr, r.stderr
        case('PRE-GB10 の集計 JSON に不通過 run → 停止（前提条件）', c_gate_json)

        def c_swap():
            out, py, base = prep('swap')
            r = run(args_for(out, py, base, swap_agg=True))
            assert not ok(r) and 'machine' in r.stderr, r.stderr
        case('SRC-機体の取り違え（--m4-agg に GB10 の JSON）→ 停止', c_swap)

        def c_bc_nan():
            out, py, base = prep('bcnan')
            p = f'{out["m4max"][0]}-aggregate.json'
            txt = open(p).read()
            i = txt.index('"bc_med": ') + len('"bc_med": ')
            j = min(txt.index(',', i), txt.index('\n', i))
            open(p, 'w').write(txt[:i] + 'NaN' + txt[j:])
            r = run(args_for(out, py, base))
            assert not ok(r) and 'NaN' in r.stderr, r.stderr
        case('IN-集計 JSON の bc_med が NaN → 停止', c_bc_nan)

        def c_exc():
            out, py, base = prep('exc')
            edit_json(f'{out["m4max"][0]}-aggregate.json', lambda d: d.update(others_exceptions=[]))
            r = run(args_for(out, py, base))
            assert not ok(r) and '例外' in r.stderr, r.stderr
        case('EXC-明示例外が M4_SKIP と不一致 → 停止', c_exc)

        for tag, line, hint in (
            ('pymix', {'framework': 'fandhe-ai', 'task': 'gemm', 'device': 'cuda', 'size': 256, 'median_s': 0.1, 'mode': 'reuse', 'parity_fail_count': 0}, 'Python FW 以外'),
            ('pynan', '{"framework": "scipy", "task": "gemm", "device": "cpu", "size": 256, "median_s": NaN, "mode": "fresh", "parity_fail_count": 0}', 'NaN'),
            ('pydup', {'framework': 'pytorch', 'version': 'x', 'task': 'gemm', 'device': 'cuda', 'size': 256, 'median_s': 0.5, 'mode': 'fresh', 'parity_total': 65536, 'parity_fail_count': 0}, '重複'),
        ):
            def c_py(tag=tag, line=line, hint=hint):
                out, py, base = prep(tag)
                with open(py, 'a') as w:
                    w.write((line if isinstance(line, str) else json.dumps(line)) + '\n')
                r = run(args_for(out, py, base))
                assert not ok(r) and hint in r.stderr, r.stderr
            case(f'PY-{tag}（--gb-py に不正行）→ 停止', c_py)

        def c_need_agg():
            out, py, base = prep('needagg')
            a = args_for(out, py, base)
            i = a.index('--m4-agg')
            r = run(a[:i] + a[i + 2:])
            assert not ok(r) and '--m4-agg' in r.stderr, r.stderr
        case('MODE-正式モードで集計 JSON なし → 停止', c_need_agg)

        def c_prev():
            out, py, base = prep('prev')
            r = run(args_for(out, py, base) + ['--m4-prev', py])
            assert not ok(r) and '--m4-prev' in r.stderr, r.stderr
        case('MODE-正式モードで --m4-prev（自前の前比）→ 停止', c_prev)

        def c_m4_gate():
            out, py, base = prep('m4gate', m4_mut=m4_gate_fail)
            r = run(args_for(out, py, base))
            assert ok(r), r.stderr
            h = open(os.path.join(base, 'out-C.html')).read()
            assert '不通過 run3（load1=9.10）' in h, 'M4 Max の不通過 run が併記されていない'
            check_tsv_matches(out, base, 'C')
        case('GATE-M4 Max 不通過 run は採用し計測条件へ併記', c_m4_gate)

        def c_parity():
            out, py, base = prep('parity', m4_mut=c_parity_fail)
            r = run(args_for(out, py, base))
            assert ok(r), r.stderr
            rows = [l.split('\t') for l in open(os.path.join(base, 'out-C.tsv')).read().splitlines()]
            row = [x for x in rows if x[0] == 'M4 Max' and x[1] == 'gemm Metal N=256'][0]
            assert row[2] == '判定不能' and row[9] == '1.0000', row
        case('J5-fandhe-ai parity 不合格は判定不能（B→C 比は併記）', c_parity)

        def c_legacy():
            d88 = LOGS_DIR / 'framework-compare-precision-class-remeasure-1988'
            R, L = LOGS_DIR / 'framework-compare-0.9.0-remeasure', LOGS_DIR / 'lowlayer-diagnosis-2026-09-12' / 'dgx'
            raw = LOGS_DIR.parents[2] / 'scripts' / 'bench' / 'framework-compare' / 'results' / 'raw'
            common = ['--m4', str(R / 'm4max-series-b' / 'results-m4max-0.9.0-median5.jsonl'),
                      '--m4-prev', str(raw / 'results-m4max-0.8.0.jsonl'),
                      '--gb', str(d88 / 'results-dgx-0.9.0-precision-class.jsonl'),
                      '--gb-extra', str(R / 'gb10' / 'results-dgx-0.9.0-extra.jsonl'),
                      '--gb-py', str(d88 / 'results-dgx-py-precision-class.jsonl'),
                      '--gb-prev', str(L / 'results-dgx-0.8.0.jsonl'), str(L / 'results-dgx-0.8.0-extra.jsonl'),
                      str(L / 'results-dgx-py-0.8.0.jsonl')]
            base = os.path.join(td, 'legacy')
            os.makedirs(base)
            r88 = subprocess.run([sys.executable, str(d88 / 'scoreboard' / 'gen_1988.py')] + common
                                 + ['--out', os.path.join(base, 'a.html')], capture_output=True, text=True)
            r20 = run(common + ['--legacy-1988', '--body', str(d88 / 'scoreboard' / 'body_1988.html'), '--prev-label', '0.8.0',
                                '--out', os.path.join(base, 'b.html')])
            assert r88.returncode == 0 and r20.returncode == 0, (r88.stderr, r20.stderr)
            assert open(os.path.join(base, 'a.html'), 'rb').read() == open(os.path.join(base, 'b.html'), 'rb').read(), 'HTML が byte 同一でない'
            assert r88.stdout == r20.stdout, '標準出力が byte 同一でない'
        case('LEGACY-gen_1988.py と HTML・標準出力が byte 同一', c_legacy)

        for cid, fn in cases:
            fn()
            print(f'ok  {cid}')
    print(f'self-test ok（{len(cases)} ケース）')


if '--self-test' in sys.argv:
    # 行定義（M4_ROWS／GB_ROWS）はモジュール後半にあるため、self-test 用に同じ定義をここで参照できるようにする
    M4_ROWS_SPEC = [('gemm', 'metal', n) for n in (256, 512, 1024, 2048, 4096)] + [('gemm', 'cpu', n) for n in (256, 512, 1024, 2048)] + \
        [('train', 'metal', 64), ('infer', 'metal', 64), ('train', 'cpu', 64), ('infer', 'cpu', 64)]
    GB_ROWS_SPEC = [('gemm', 'cuda', n) for n in (256, 512, 1024, 2048, 4096)] + [('gemm', 'cpu', n) for n in (256, 512, 1024, 2048, 4096)] + \
        [('train', 'cuda', 64), ('infer', 'cuda', 64), ('train', 'cpu', 64), ('infer', 'cpu', 64)]
    _self_test()
    raise SystemExit(0)

ARGS = parse_args()
_require_file(ARGS.style, '--style')
_require_file(ARGS.body, '--body')
LEGACY = ARGS.legacy_1988
if LEGACY:
    if ARGS.m4_agg or ARGS.gb_agg:
        _die('--legacy-1988 では --m4-agg／--gb-agg を使わない')
    if ARGS.gb_extra is None:
        _die('--legacy-1988 では --gb-extra が必須')
    PREV_VER = ARGS.prev_label or '0.8.0'  # 比較元ラベル（この版との差分を「{PREV_VER} 比」として表示する）
else:
    if not (ARGS.m4_agg and ARGS.gb_agg):
        _die('正式モードは --m4-agg と --gb-agg（aggregate.py の集計 JSON）が必須（gen_1988 互換の出力は --legacy-1988）')
    for _flag, _v in (('--m4-prev', ARGS.m4_prev), ('--gb-prev', ARGS.gb_prev), ('--prev-label', ARGS.prev_label)):
        if _v is not None:
            _die(f'正式モードでは {_flag} を使わない（前比は集計 JSON の bc_med。RULE.txt 判定 3 追記）')
    PREV_VER = None


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def key(r):
    return (r['framework'], r['task'], r['device'], int(r['size']), r['mode'])


def index(rows):
    d = {}
    for r in rows:
        if r['task'] in ('train_phases', 'infer_phases'):
            continue
        d[key(r)] = r  # 後勝ち（追記分を優先）
    return d


def load_agg(path, machine, main_path, arm):
    """集計 JSON を読み、前提（formal・機体・セル集合・専有ゲート・派生 JSONL の sha256）を検査して {cell: dict} を返す。"""
    _require_file(path, f'集計 JSON（{machine}）')
    try:
        d = json.loads(open(path).read(), parse_constant=AGGMOD._reject_constant, object_pairs_hook=AGGMOD._no_dup_keys)
    except ValueError as e:
        _die(f'{path}: 集計 JSON を読めない: {e}')
    if d.get('schema') != AGGMOD.SCHEMA:
        _die(f'{path}: schema が {AGGMOD.SCHEMA} ではない')
    if d.get('machine') != machine:
        _die(f'{path}: machine={d.get("machine")!r}（期待 {machine}）。機体の取り違え')
    if d.get('status') != 'formal':
        _die(f'{path}: status={d.get("status")!r}。前提不成立の集計から正式スコアボードを作らない（status formal が必要）')
    g = d.get('gate') or {}
    runs = g.get('runs') or []
    if len(runs) != AGGMOD.NRUN or (AGGMOD.GATE_SPEC[machine]['precondition'] and any(x.get('status') != 'pass' for x in runs)):
        _die(f'{path}: 専有ゲートの記録が前提（{machine}）を満たさない')
    cells = {}
    for c in d.get('cells', []):
        k = (c.get('task'), c.get('device'), c.get('size'), c.get('mode'))
        if k in cells:
            _die(f'{path}: セル {k} が重複')
        for f in ('bc_med', 'bc_q1', 'bc_q3'):
            try:
                AGGMOD._positive_finite(c.get(f), f)
            except ValueError as e:
                _die(f'{path}: セル {k}: {e}')
        cells[k] = c
    if set(cells) != set(AGGMOD.judged_cells(machine)):
        _die(f'{path}: セル集合が RULE.txt のセル範囲と一致しない')
    exc = {tuple(x) for x in d.get('others_exceptions', [])}
    if exc != AGGMOD.others_exceptions(machine):
        _die(f'{path}: 明示例外（others_exceptions）が aggregate.py の定義と一致しない')
    dg = (d.get('derived') or {}).get(arm) or {}
    if AGGMOD.sha256_file(main_path) != dg.get('sha256'):
        _die(f'{main_path} が集計 JSON の腕 {arm} の派生 JSONL（sha256）と一致しない（集計後の改変・腕／機体の取り違え）')
    return d, cells


def load_py(paths):
    """--gb-py／--gb-extra: Python FW 行のみを aggregate.py と同じ厳格読み取りで受理する（後勝ちマージで集計値を上書きさせない）。"""
    seen, out = set(), []
    for p in paths:
        try:
            rows = AGGMOD.load_rows(p)
        except ValueError as e:
            _die(f'Python FW JSONL を読めない: {e}')
        for r, _ in rows:
            if r.get('framework') not in PY_FWS:
                _die(f'{p}: Python FW 以外の行（{r.get("framework")!r}）。fandhe-ai・candle・burn は集計の派生 JSONL のみから取る')
            if r.get('task') in AGGMOD.PHASE_TASKS or 'phase' in r:
                continue
            k = key(r)
            if k in seen:
                _die(f'{p}: セル {k} が重複')
            seen.add(k)
            out.append(r)
    return out


if LEGACY:
    AGG = None
    m4 = index(load(ARGS.m4))
    m4_prev = index(load(ARGS.m4_prev or str(RAW / 'results-m4max-0.8.0.jsonl')))
    gb = index(load(ARGS.gb) + load(ARGS.gb_extra) + load(ARGS.gb_py))
    gb_prev_rows = []
    for f in (ARGS.gb_prev or LEGACY_GB_PREV):
        gb_prev_rows += load(f)
    gb_prev = index(gb_prev_rows)
else:
    AGG_M4, M4_CELLS = load_agg(ARGS.m4_agg, 'm4max', ARGS.m4, ARGS.arm)
    AGG_GB, GB_CELLS = load_agg(ARGS.gb_agg, 'gb10', ARGS.gb, ARGS.arm)
    AGG = {'M4 Max': M4_CELLS, 'GB10': GB_CELLS}
    m4 = index(load(ARGS.m4))
    gb = index(load(ARGS.gb) + load_py([ARGS.gb_py] + ([ARGS.gb_extra] if ARGS.gb_extra else [])))
    m4_prev = gb_prev = None


def gemm_ms(n, gf):
    return 2.0 * n ** 3 / (gf * 1e9) * 1e3


# 2026-09-12 ページからの転記（M4 Max・Python 3 FW。元 JSONL はリポジトリ未収録）。
# 0.9.0 でも Python 3 FW（PyTorch/TensorFlow/SciPy）は再計測していないため、0.8.0 版と
# 同じ転記値をそのまま流用する（fandhe-ai 側のみが変わるフレームワーク横並び比較のため、
# 比較対象 FW 自体の数値は版が進んでも変わらない）。
# gemm は GFLOP/s から ms を逆算（表示 2 桁より精度が高い）、infer は µs から。
M4_PY = {
    'pytorch': {'version': '2.14.0', 'gemm': {'metal': {256: 73, 512: 315, 1024: 705, 2048: 1170, 4096: 2663},
                                              'cpu': {256: 207, 512: 414, 1024: 749, 2048: 1142}},
                'train': {'metal': 0.40, 'cpu': 0.20}, 'infer_us': {'metal': 295, 'cpu': 33}, 'resc': {}},
    'tensorflow': {'version': '2.16.2', 'gemm': {'metal': {256: 60, 512: 250, 1024: 533, 2048: 1080, 4096: 2530},
                                                 'cpu': {256: 129, 512: 282, 1024: 471, 2048: 699}},
                   'train': {'metal': 1.44, 'cpu': 0.96}, 'infer_us': {'metal': 665, 'cpu': 266},
                   'resc': {('gemm', 'cpu', 2048): 4}},
    'scipy': {'version': '1.18.1', 'gemm': {'cpu': {256: 121, 512: 227, 1024: 372, 2048: 470}},
              'train': {'cpu': 0.31}, 'infer_us': {'cpu': 164}, 'resc': {}},
}
for fw, d in M4_PY.items():
    for dev, sizes in d['gemm'].items():
        for n, gf in sizes.items():
            m4[(fw, 'gemm', dev, n, 'fresh')] = {'framework': fw, 'version': d['version'], 'task': 'gemm', 'device': dev, 'size': n, 'mode': 'fresh',
                                                 'median_s': gemm_ms(n, gf) / 1e3, 'gflops': gf, 'parity_fail_count': 0,
                                                 'parity_scaled_abs_rescued': d['resc'].get(('gemm', dev, n), 0), 'transcribed': True}
    for dev, ms in d['train'].items():
        m4[(fw, 'train', dev, 64, 'fresh')] = {'framework': fw, 'version': d['version'], 'task': 'train', 'device': dev, 'size': 64, 'mode': 'fresh', 'median_s': ms / 1e3, 'transcribed': True}
    for dev, us in d['infer_us'].items():
        m4[(fw, 'infer', dev, 64, 'fresh')] = {'framework': fw, 'version': d['version'], 'task': 'infer', 'device': dev, 'size': 64, 'mode': 'fresh', 'median_s': us / 1e6, 'transcribed': True}

# M4 Max burn Metal N>=512 は結果テンソル全ゼロで記録拒否（skipped-m4max-0.9.0.log。0.8.0 と同一の
# upstream 既知バグによる継続現象）
M4_SKIP = {('burn', 'gemm', 'metal', n): '結果テンソルが全ゼロ（upstream 既知バグ）のため記録拒否' for n in (512, 1024, 2048, 4096)}
# 正式モード: 表示上の記録拒否（—）と aggregate.py の判定 1 明示例外が同じ集合であること（片側だけの変更を検出）
if not LEGACY and {k[:4] for k in AGGMOD.others_exceptions('m4max')} != set(M4_SKIP):
    _die('M4_SKIP と aggregate.others_exceptions(m4max) が一致しない（明示例外の片側変更）')

FW_ORDER = ['candle', 'burn', 'pytorch', 'tensorflow', 'scipy']

def fmt_ms(ms):
    if ms >= 100: return f'{ms:,.2f}'
    if ms < 0.1: return f'{ms:.3f}'
    return f'{ms:.2f}'

def cell_value(r, task, size):
    """(表示値, 有効か, 注記) を返す。"""
    ms = r['median_s'] * 1e3
    fail = r.get('parity_fail_count') or 0
    resc = r.get('parity_scaled_abs_rescued') or 0
    valid = fail == 0
    if task == 'gemm':
        gf = r.get('gflops') or (2.0 * size ** 3 / r['median_s'] / 1e9)
        txt = f'{fmt_ms(ms)} <small>/ {gf:,.0f} GF</small>'
    elif task == 'train':
        txt = f'{fmt_ms(ms)} <small>ms</small>'
    else:
        tp = 1.0 / r['median_s']
        txt = f'{tp:,.0f} <small>/ {ms*1e3:,.0f} µs</small>'
    return txt, valid, fail, resc

def metric(r, task):
    """比較用スカラ。時間は小さいほど良い・推論はスループット大きいほど良い。"""
    return (1.0 / r['median_s']) if task == 'infer' else r['median_s']

def better(a, b, task):
    return a > b if task == 'infer' else a < b

# 正式モードの前比の注記（行の vprevs は TSV・標準出力にも出るため HTML 実体参照を含めない。カードは &lt;1 を付ける）
AGG_PREV_NOTE = '時間比 C/B・同一 run 内比の 5 run 中央値'


def agg_prev(machine, task, device, size):
    """正式モードの前比: 判定列（reuse）セルの bc_med と表示文字列。値は集計 JSON から転記するだけで再計算しない。"""
    c = AGG[machine][(task, device, size, 'reuse')]
    return c['bc_med'], f'B→C 比 {c["bc_med"]:.2f}×（{AGG_PREV_NOTE}）'


def build_row(data, data_prev, machine, task, device, size, skip):
    label = {'gemm': f'gemm {DEVNAME[device]} N={size}', 'train': f'train {DEVNAME[device]}', 'infer': f'infer {DEVNAME[device]}'}[task]
    desc = {'gemm': 'f32 正方 GEMM・reuse', 'train': '784→256→10 MLP・バッチ 64・1 step・reuse（デバイス常駐 SGD）', 'infer': '同 MLP forward・バッチ 64・reuse'}[task]
    me = data.get(('fandhe-ai', task, device, size, 'reuse'))
    me_fresh = data.get(('fandhe-ai', task, device, size, 'fresh'))
    assert me is not None, (machine, task, device, size)
    # fandhe-ai 自身の判定行（reuse）が parity 不合格なら他 FW と同じく判定不能。順位・倍率を出さない（RULE.txt 判定 5）。
    me_fail = me.get('parity_fail_count') or 0
    if me_fail > 0:
        me_txt = cell_value(me, task, size)[0]
        tr = (f'<tr class="v-undet"><th scope="row"><span class="ph">{label}</span><span class="ds">{desc}</span></th>'
              f'<td><span class="pill undet">判定不能</span><span class="vs">fandhe-ai parity 不合格 {me_fail:,} 要素</span></td>'
              f'<td class="ratio"><span class="rt">—</span></td>'
              f'<td class="num me inv">{me_txt} <small>reuse</small></td><td class="num me2"><span class="na">—</span></td>'
              + ''.join('<td class="num"><span class="na">—</span></td>' for _ in FW_ORDER) + '</tr>')
        return dict(tr=tr, cls='undet', verdict='判定不能', ratio=None, label=label, machine=machine, best=None, vprev=None,
                    vprevs='判定不能（fandhe-ai parity 不合格）', me=me, me_fresh=None, old=None, fresh_note='')
    mine = metric(me, task)
    comp = []  # (fw, r, valid)
    best = None
    cells = []
    for fw in FW_ORDER:
        r = data.get((fw, task, device, size, 'fresh'))
        if r is None:
            reason = skip.get((fw, task, device, size))
            title = f' title="{html.escape(reason)}"' if reason else ''
            cells.append((fw, f'<td class="num"><span class="na"{title}>—</span></td>', None))
            continue
        txt, valid, fail, resc = cell_value(r, task, size)
        if not valid:
            tot = r.get('parity_total')
            cells.append((fw, f'<td class="num inv" title="要素検証超過 {fail:,}/{tot:,}（判定不能・順位に含めない）">{txt}</td>', None))
            continue
        mark = ''
        if resc:
            if fw == 'burn' and r.get('tf32') is True:
                mark = f' <span class="resc" title="精度クラス TF32（u=2^-11）の第 3 救済項で救済 {resc:,} 要素（#1989 承認・#2046 実装・有効セル扱い）">△{resc:,}</span>'
            else:
                mark = f' <span class="resc" title="スケール付き絶対誤差で救済 {resc:,} 要素（#1241 承認契約・有効セル扱い）">△{resc:,}</span>'
        v = metric(r, task)
        comp.append((fw, v))
        if best is None or better(v, best[1], task):
            best = (fw, v)
        cells.append((fw, txt + mark, v))
    # 有効な比較対象が 1 つも無い（行欠損・全 FW が parity 不合格）セルは判定不能として表示し、他セルの生成を継続する
    if best is None:
        me_txt = cell_value(me, task, size)[0]
        cells_html = ''.join(c if v is None else f'<td class="num">{c}</td>' for fw, c, v in cells)
        tr = (f'<tr class="v-undet"><th scope="row"><span class="ph">{label}</span><span class="ds">{desc}</span></th>'
              f'<td><span class="pill undet">判定不能</span><span class="vs">有効な比較対象なし</span></td>'
              f'<td class="ratio"><span class="rt">—</span></td>'
              f'<td class="num me">{me_txt} <small>reuse</small></td><td class="num me2"><span class="na">—</span></td>'
              f'{cells_html}</tr>')
        return dict(tr=tr, cls='undet', verdict='判定不能', ratio=None, label=label, machine=machine, best=None, vprev=None,
                    vprevs='判定不能（有効な比較対象なし）', me=me, me_fresh=None, old=None, fresh_note='')
    # 順位
    n_better = sum(1 for fw, v in comp if better(v, mine, task))
    rank = 1 + n_better
    if task == 'infer':
        ratio = mine / best[1]
    else:
        ratio = best[1] / mine
    if rank == 1:
        verdict, cls = '1 位', 'win'
    elif ratio >= 0.90:
        verdict, cls = '僅差', 'near'
    else:
        verdict, cls = f'{rank} 位', 'loss'
    # 前比
    if AGG is not None:
        # 正式モード: 集計 JSON の bc_med（RULE.txt 判定 3。同一 run 内比 C/B の 5 run 中央値・infer も時間比）をそのまま使い、
        # 採用行同士の比は計算しない（判定 3 追記。単一情報源）
        vprev, vprevs = agg_prev(machine, task, device, size)
        old, fresh_note = None, ''
    else:
        # --legacy-1988: gen_1988.py と同一の {PREV_VER} 比（採用行同士の比。#2120 の正式表示には使わない）
        old = data_prev.get(('fandhe-ai', task, device, size, 'reuse'))
        fresh_note = ''
        if old is None:
            old = data_prev.get(('fandhe-ai', task, device, size, 'fresh'))
            base = me_fresh if me_fresh is not None else me
            fresh_note = '（fresh 同士）'
        else:
            base = me
        if old is not None:
            vprev = (metric(base, task) / metric(old, task)) if task == 'infer' else (base['median_s'] / old['median_s'])
            vprevs = f'{PREV_VER} 比 {vprev:.2f}×{fresh_note}'
        else:
            vprev, vprevs = None, f'{PREV_VER} 記録なし'
    # bar
    if ratio < 1:
        bar = f'<div class="bar"><i class="down" style="width:{ratio*100:.0f}%"></i></div>'
    else:
        w = min(100, max(10, (ratio - 1) * 100))
        bar = f'<div class="bar"><i class="up" style="width:{w:.0f}%"></i></div>'
    me_txt = cell_value(me, task, size)[0]
    fresh_txt = cell_value(me_fresh, task, size)[0] if me_fresh else '<span class="na">—</span>'
    cells_html = ''.join(
        (f'<td class="num{" best" if best and fw == best[0] else ""}">{c}</td>' if v is not None else c)
        for fw, c, v in cells)
    tr = (f'<tr class="v-{cls}"><th scope="row"><span class="ph">{label}</span><span class="ds">{desc}</span></th>'
          f'<td><span class="pill {cls}">{verdict}</span><span class="vs">vs {FWNAME[best[0]]}</span><span class="vs">{vprevs}</span></td>'
          f'<td class="ratio">{bar}<span class="rt">{ratio:.2f}×</span></td>'
          f'<td class="num me">{me_txt} <small>reuse</small></td>'
          f'<td class="num me2">{fresh_txt}{" <small>fresh</small>" if me_fresh else ""}</td>'
          f'{cells_html}</tr>')
    return dict(tr=tr, cls=cls, verdict=verdict, ratio=ratio, label=label, machine=machine, best=best[0], vprev=vprev, vprevs=vprevs,
                me=me, me_fresh=me_fresh, old=old, fresh_note=fresh_note)

DEVNAME = {'metal': 'Metal', 'cpu': 'CPU', 'cuda': 'CUDA'}
FWNAME = {'candle': 'candle', 'burn': 'burn', 'pytorch': 'PyTorch', 'tensorflow': 'TensorFlow', 'scipy': 'SciPy'}

M4_ROWS = [('gemm', 'metal', n) for n in (256, 512, 1024, 2048, 4096)] + [('gemm', 'cpu', n) for n in (256, 512, 1024, 2048)] + \
          [('train', 'metal', 64), ('infer', 'metal', 64), ('train', 'cpu', 64), ('infer', 'cpu', 64)]
GB_ROWS = [('gemm', 'cuda', n) for n in (256, 512, 1024, 2048, 4096)] + [('gemm', 'cpu', n) for n in (256, 512, 1024, 2048, 4096)] + \
          [('train', 'cuda', 64), ('infer', 'cuda', 64), ('train', 'cpu', 64), ('infer', 'cpu', 64)]

m4_rows = [build_row(m4, m4_prev, 'M4 Max', *r, M4_SKIP) for r in M4_ROWS]
gb_rows = [build_row(gb, gb_prev, 'GB10', *r, {}) for r in GB_ROWS]
allrows = m4_rows + gb_rows
wins = [r for r in allrows if r['cls'] == 'win']
nears = [r for r in allrows if r['cls'] == 'near']
losses = [r for r in allrows if r['cls'] == 'loss']
undets = [r for r in allrows if r['cls'] == 'undet']
assert len(wins) + len(nears) + len(losses) + len(undets) == len(allrows), 'tally が行数と分割にならない'
# 無効セル（判定不能）数
def count_invalid(data, rows):
    n = 0
    for task, dev, size in rows:
        for fw in FW_ORDER:
            r = data.get((fw, task, dev, size, 'fresh'))
            if r is not None and (r.get('parity_fail_count') or 0) > 0:
                n += 1
        me = data.get(('fandhe-ai', task, dev, size, 'reuse'))
        if me is not None and (me.get('parity_fail_count') or 0) > 0:
            n += 1
    return n
invalid_n = count_invalid(m4, M4_ROWS) + count_invalid(gb, GB_ROWS)

def head(cols):
    return f'<thead><tr><th>対象</th><th>勝敗</th><th>最速他 FW ÷ fandhe-ai</th><th>fandhe-ai {ARGS.main_label}（判定）</th><th>fandhe-ai 別モード（参考）</th>' + ''.join(f'<th>{c}</th>' for c in cols) + '</tr></thead>'

def lst(rows):
    return '、'.join(f'{r["machine"]} {r["label"]}（{r["ratio"]:.2f}×）' for r in rows if r['ratio'] is not None)

def msprev(data, data_prev, task, dev, size):
    """{PREV_VER}→0.9.0 の同一モード（reuse 優先）中央値ペアを返す。"""
    old = data_prev.get(('fandhe-ai', task, dev, size, 'reuse'))
    mode = 'reuse'
    if old is None:
        old = data_prev.get(('fandhe-ai', task, dev, size, 'fresh')); mode = 'fresh'
    new = data.get(('fandhe-ai', task, dev, size, mode))
    return old, new, mode

def card_gemm(data, data_prev, dev, sizes, title):
    parts = []; ratios = []
    for n in sizes:
        old, new, mode = msprev(data, data_prev, 'gemm', dev, n)
        if old is None or new is None: continue
        parts.append(f'N={n} {old["median_s"]*1e3:.2f}→{new["median_s"]*1e3:.2f} ms')
        ratios.append(new['median_s'] / old['median_s'])
    mode_lbl = mode
    return f'<div><h3>{title}（{mode_lbl}）</h3><span class="v">{PREV_VER} 比 {min(ratios):.2f}〜{max(ratios):.2f}（&lt;1 が高速化）</span><p>{"・".join(parts)}</p></div>'

def card_train(data, data_prev, devs, title):
    parts = []; ratios = []
    for dev in devs:
        old, new, mode = msprev(data, data_prev, 'train', dev, 64)
        parts.append(f'{DEVNAME[dev]} {old["median_s"]*1e3:.2f}→{new["median_s"]*1e3:.2f} ms（{mode}）')
        ratios.append(new['median_s'] / old['median_s'])
    return f'<div><h3>{title}</h3><span class="v">{PREV_VER} 比 {min(ratios):.2f}〜{max(ratios):.2f}（&lt;1 が高速化）</span><p>{"・".join(parts)}</p></div>'

def card_infer(data, data_prev, devs, title):
    parts = []; ratios = []
    for dev in devs:
        old, new, mode = msprev(data, data_prev, 'infer', dev, 64)
        parts.append(f'{DEVNAME[dev]} {1/old["median_s"]:,.0f}→{1/new["median_s"]:,.0f} /s（{mode}）')
        ratios.append(old['median_s'] / new['median_s'])
    return f'<div><h3>{title}</h3><span class="v">{PREV_VER} 比 {min(ratios):.2f}〜{max(ratios):.2f}（&gt;1 が改善）</span><p>{"・".join(parts)}</p></div>'

def _agg_card(machine, task, keys, title, fmt_pair, mode_lbl):
    """正式モードのカード: 比の範囲は集計 JSON の bc_med（時間比・infer も同じ向き）、値は判定 2 の採用行（参考表示・比は出さない）。"""
    parts, ratios = [], []
    for dev, size, lbl in keys:
        c = AGG[machine][(task, dev, size, 'reuse')]
        parts.append(f'{lbl} {fmt_pair(c["picked"]["B"]["median_s"], c["picked"]["C"]["median_s"])}')
        ratios.append(c['bc_med'])
    return (f'<div><h3>{title}{mode_lbl}</h3><span class="v">B→C 比 {min(ratios):.2f}〜{max(ratios):.2f}（{AGG_PREV_NOTE}。&lt;1 が高速化）</span>'
            f'<p>採用行 B→C（参考）: {"・".join(parts)}</p></div>')


def agg_cards(machine, gdev, gsizes, csizes, devs, prefix):
    ms = lambda b, c: f'{b*1e3:.2f}→{c*1e3:.2f} ms'  # noqa: E731
    tp = lambda b, c: f'{1/b:,.0f}→{1/c:,.0f} /s'  # noqa: E731
    return (_agg_card(machine, 'gemm', [(gdev, n, f'N={n}') for n in gsizes], f'{prefix} GEMM {DEVNAME[gdev]}', ms, '（reuse）')
            + _agg_card(machine, 'gemm', [('cpu', n, f'N={n}') for n in csizes], f'{prefix} GEMM CPU', ms, '（reuse）')
            + _agg_card(machine, 'train', [(d, 64, DEVNAME[d]) for d in devs], f'{prefix} MLP 学習 1 step', ms, '（reuse）')
            + _agg_card(machine, 'infer', [(d, 64, DEVNAME[d]) for d in devs], f'{prefix} 推論スループット', tp, '（reuse）'))


def gate_note(d):
    """集計 JSON の専有ゲート結果を計測条件注記へ併記する文（RULE.txt: M4 Max の不通過 run は除外せず併記）。"""
    runs = d['gate']['runs']
    ng = [x for x in runs if x['status'] != 'pass']
    s = f'gate.log: 通過 {len(runs) - len(ng)}/{len(runs)} run'
    if ng:
        s += '・不通過 ' + '・'.join(f'run{x["run"]}（load1={x["load1"]}）' for x in ng)
    return html.escape(s)


if AGG is not None:
    M4_LOAD = '；'.join(x for x in (ARGS.m4_load, gate_note(AGG_M4)) if x)
    GB_LOAD = '；'.join(x for x in (ARGS.gb_load, gate_note(AGG_GB)) if x)
    M4_CARDS = agg_cards('M4 Max', 'metal', (256, 512, 1024, 2048, 4096), (256, 512, 1024, 2048), ('metal', 'cpu'), 'M4 Max')
    GB_CARDS = agg_cards('GB10', 'cuda', (256, 512, 1024, 2048, 4096), (256, 512, 1024, 2048, 4096), ('cuda', 'cpu'), 'GB10')
else:
    M4_LOAD, GB_LOAD = ARGS.m4_load, ARGS.gb_load
    M4_CARDS = (card_gemm(m4, m4_prev, 'metal', (256, 512, 1024, 2048, 4096), 'M4 Max GEMM Metal') + card_gemm(m4, m4_prev, 'cpu', (256, 512, 1024, 2048), 'M4 Max GEMM CPU')
                + card_train(m4, m4_prev, ('metal', 'cpu'), 'M4 Max MLP 学習 1 step') + card_infer(m4, m4_prev, ('metal', 'cpu'), 'M4 Max 推論スループット'))
    GB_CARDS = (card_gemm(gb, gb_prev, 'cuda', (256, 512, 1024, 2048, 4096), 'GB10 GEMM CUDA') + card_gemm(gb, gb_prev, 'cpu', (256, 512, 1024, 2048, 4096), 'GB10 GEMM CPU')
                + card_train(gb, gb_prev, ('cuda', 'cpu'), 'GB10 MLP 学習 1 step') + card_infer(gb, gb_prev, ('cuda', 'cpu'), 'GB10 推論スループット'))

CSS = open(ARGS.style).read()
BODY = open(ARGS.body).read()

out = BODY.format(
    css=CSS,
    n_win=len(wins), n_near=len(nears), n_loss=len(losses), n_invalid=invalid_n, n_undet=len(undets), n_total=len(allrows),
    m4_head=head(['candle-core 0.11.0', 'burn 0.21.0', 'PyTorch 2.14.0（MPS）', 'TensorFlow 2.16.2（Metal プラグイン）', 'SciPy 1.18.1']),
    m4_rows=''.join(r['tr'] for r in m4_rows),
    gb_head=head(['candle-core 0.11.0', 'burn 0.21.0', 'PyTorch 2.14.0+cu130', 'TensorFlow 2.21.0（CPU のみ）', 'SciPy 1.18.1']),
    gb_rows=''.join(r['tr'] for r in gb_rows),
    win_list=lst(wins), near_list=lst(nears), loss_list=lst(losses),
    m4_load=M4_LOAD, gb_load=GB_LOAD,
    m4_cards=M4_CARDS, gb_cards=GB_CARDS,
)
Path(ARGS.out).write_text(out)

# 検証出力
for r in allrows:
    rt = 'n/a' if r['ratio'] is None else f"{r['ratio']:.2f}x"
    print(f"{r['machine']:7} {r['label']:22} {r['verdict']:4} vs {r['best'] or '-':10} {rt}  {r['vprevs']}")
if ARGS.tsv:
    def _tsv_tail(r, spec):
        # 正式モードは 22 セル転記用に、行のセル（task／device／size）と集計 JSON の B→C 比・q1／q3・非後退・checksum 一致を
        # そのまま追記する（値は集計 JSON からの転記のみ。判定 3・4・6）
        if AGG is None:
            return ''
        t, d, n = spec
        c = AGG[r['machine']][(t, d, n, 'reuse')]
        ck = '一致' if c['ck_within'] and c['ck_across'] else '不一致'
        return (f"\t{t}\t{d}\t{n}\t{c['bc_med']:.4f}\t{c['bc_q1']:.4f}\t{c['bc_q3']:.4f}"
                f"\t{'非後退' if c['nonreg'] else '後退'}\t{ck}")
    Path(ARGS.tsv).write_text(''.join(f"{r['machine']}\t{r['label']}\t{r['verdict']}\t{r['best'] or ''}\t{'' if r['ratio'] is None else format(r['ratio'], '.4f')}\t{r['vprevs']}{_tsv_tail(r, spec)}\n"
                                      for r, spec in zip(allrows, M4_ROWS + GB_ROWS)))
if LEGACY:
    # gen_1988.py と byte 同一の集計行（判定不能行は 0.9.0 系データでは生じない）
    print('tally', len(wins), len(nears), len(losses), 'invalid', invalid_n, 'total', len(allrows))
else:
    print('tally', len(wins), len(nears), len(losses), 'undet', len(undets), 'invalid', invalid_n, 'total', len(allrows))
