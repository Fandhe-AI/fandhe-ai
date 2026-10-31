#!/usr/bin/env python3
"""fandhe-ai 0.13.0 対戦成績スコアボード（HTML）を framework-compare JSONL から生成する（計測キット。未実測）。

役割: `gen_0120.py`（0.12.0）からの派生で、0.13.0 の 5 ラウンド中央値 JSONL（GB10 は 5 ラウンド、Mac mini は系列 B の 5 ラウンド）を
同じ判定規則で HTML にする。呼び出し元は `run_0130.sh`（検証は `check_0130.py`）。body は同ディレクトリの `body_0130.html`、
色・クラスは `style.css`（0.12.0 版の逐語コピー）。

0.12.0 版からの差分（表示・入力まわりだけ。判定経路は変えない）:
- VER／PREV_VER を 0.13.0／0.12.0 にした。前版比（「0.12.0 比」）は同一セル・同一モードの fandhe-ai 中央値同士の比で、
  0.12.0 の正式値（Mac mini は系列 B・GB10 は 5 ラウンド）を読む。日をまたぐため負荷の揺れを含む（ページにも書く）。
- 単腕にした: 旧条件腕（--m4-old／--gb-old・--oldarm-out・swap 経路・表の「旧条件 x」注記・GB10 CPU gemm の新旧比の注記）を削除した。
  bench-fandhe は 0.12.0 から変えずピンだけ上げるため、条件表は「0.13.0（正式値）」と「0.12.0」の 2 行になる。
- 対照系列を置かない: --m4-alt／--m4-py-alt／--m4-alt-runs を削除した。ノイズ帯（対照系列との区分反転）は算出できないので、
  tally 行は `noise na`、ページの条件表・まとめ節は「未算出（対照系列なし）」と表示する（0 と区別する）。
- fandhe-ai 行の `version` が --run-ver（既定 0.13.0）と一致しない入力は SystemExit（別版の値を 0.13.0 として載せない）。
- 実測依存の文字列（日付・負荷・skipped・公開日・コミット数・GB10 CPU gemm の所見）は引数で受け取り、`<要記入` を含む値が 1 つでも
  残っていれば SystemExit（推測値を埋めない。gen_args.txt の雛形は全て未記入）。GB10 の driver は run の env_info.txt から読む。
- 学習の最終 loss の注記・CUDA GEMM の reuse／fresh の注記・Python 3 FW の版は、入力 JSONL から生成する（元版は文言を直書きしていた）。
- --unreached-tsv: 1.0 以下のセルの理由表。セルの集合と比は入力から生成し直して照合する（TSV と違えば SystemExit）。
  分類・理由・出典が `<要記入` のまま、分類が 3 種（と表示専用の「分類保留」）以外、理由と出典が空のときも SystemExit。
  `--emit-unreached-skeleton PATH` は、判定を終えたあとに 1.0 以下のセルの雛形 TSV（分類・理由・出典は `<要記入>`）を書き出す。

判定規則（変えていない。0.12.0 版の関数群と字句で同じ）: 同一 run 内の対戦相手比の 5 run 中央値で区分する
（>1 で 1 位、0.90 以上で僅差、それ未満は負け。RULE.txt の主判定）。parity_fail_count>0 のセルは判定不能で順位に含めない。
判定側（fandhe-ai）の要素検証が不合格なら生成停止。fandhe-ai／candle の `tf32: true` 行は f32 判定から除外し、burn の `tf32: true` 行は残す
（burn は TF32 が既定）。欠損・不正値は SystemExit で fail-closed にする。

入力: --m4／--m4-py（Mac mini 系列 B の fandhe/candle/burn と Python 3 FW の 5 ラウンド中央値）、--gb／--gb-extra／--gb-py（GB10）、
--m4-runs／--gb-runs（run1〜run5 を含む親ディレクトリ）。
"""
import argparse
import re
import statistics
import json
import html
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]  # docs/perf/logs/<dir>/scoreboard/ から見たリポジトリルート
PREV_GB = REPO / 'docs/perf/logs/framework-compare-0.12.0-remeasure'  # GB10 の前版（0.12.0）の 5 ラウンド中央値の置き場
PREV_M4 = REPO / 'docs/perf/logs/framework-compare-0.12.0-macmini-remeasure'  # Mac mini の前版（同じ機体・正式系列 B）
M4_NAME = 'Mac mini（M4）'  # Apple 側の機体名（表の見出し・まとめ節の行ラベルなど生成器が出す全箇所で使う）

VER = '0.13.0'
PREV_VER = '0.12.0'  # 比較元バージョン（この版との差分を「{PREV_VER} 比」として表示する）
PLACEHOLDER = '<要記入'  # 実測前の雛形に置く未記入の印。引数・TSV に 1 つでも残っていれば生成を止める


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--m4', required=True, help='0.13.0 Apple 機 Mac mini 系列 B（fandhe/candle/burn。5 ラウンド中央値）JSONL')
    p.add_argument('--m4-py', required=True, help='0.13.0 Apple 機 Mac mini 系列 B の Python 3 FW（5 ラウンド中央値）JSONL')
    p.add_argument('--m4-runs', required=True, help='Apple 機（系列 B）run ディレクトリ群の親（run1〜run5 を含む。同一 run 内比の算出用）')
    p.add_argument('--gb-runs', required=True, help='GB10 run ディレクトリ群の親（run1〜run5 を含む。env_info.txt から driver も読む）')
    p.add_argument('--m4-prev', nargs='+', default=[
        str(PREV_M4 / 'macmini-series-b/results-macmini-0.12.0-median5.jsonl'),
        str(PREV_M4 / 'macmini-series-b/results-macmini-py-0.12.0-median5.jsonl'),
    ], help='Apple 機（Mac mini）の前版 JSONL 群（正式系列 B）。空を渡すと出さない（Apple 行は「前版比なし」）')
    p.add_argument('--gb', required=True, help='0.13.0 GB10（fandhe/candle/burn 本体）JSONL')
    p.add_argument('--gb-extra', required=True, help='0.13.0 GB10 追加計測（CPU GEMM reuse 等）JSONL')
    p.add_argument('--gb-py', required=True, help='0.13.0 GB10 Python 3 FW（PyTorch/TensorFlow/SciPy）JSONL')
    p.add_argument('--gb-prev', nargs='+', default=[
        str(PREV_GB / 'gb10/results-dgx-0.12.0-median5.jsonl'),
        str(PREV_GB / 'gb10/results-dgx-0.12.0-extra-median5.jsonl'),
    ], help='前版（0.12.0）GB10 JSONL 群（複数可。fandhe-ai 行だけを使うので Python 3 FW は不要）')
    p.add_argument('--out', default='fandhe-ai-0.13.0-scoreboard.html', help='出力 HTML パス')
    # 実測依存の文字列。全て必須（gen_args.txt の雛形は未記入＝<要記入>。未記入のままなら SystemExit）
    p.add_argument('--m4-load', required=True, help='Apple 機（Mac mini）の負荷条件注記文（各 run 開始時 load1・ゲート待機の実績。計測条件節に埋め込む）')
    p.add_argument('--gb-load', required=True, help='GB10 の負荷条件注記文（計測時刻・各ラウンド開始時 load1・GPU 使用率・compute プロセス。計測条件節に埋め込む）')
    p.add_argument('--m4-series', required=True, help='Apple 機（Mac mini）の系列の説明（系列 B のみ・対照系列なし）')
    p.add_argument('--m4-date', required=True, help='Apple 機（Mac mini）の計測日（eyebrow・見出しに埋め込む）')
    p.add_argument('--m4-skipped-note', required=True, help='Apple 機の計測失敗の一覧（各 run の skipped-macmini-0.13.0.log の要約。失敗なしなら「なし」）')
    p.add_argument('--updated', required=True, help='ページ eyebrow の更新日')
    p.add_argument('--gb-date', required=True, help='GB10 の計測日（eyebrow・見出しに埋め込む）')
    p.add_argument('--gb-skipped-note', required=True, help='GB10 の計測失敗の一覧（失敗なしなら「なし」）')
    p.add_argument('--release-date', required=True, help='crates.io への fandhe-ai 0.13.0 の公開日（lede に埋め込む）')
    p.add_argument('--commit-count', required=True, help='git rev-list --count v0.12.0..v0.13.0 の実測値（「何が変わったか」のカードに埋め込む）')
    p.add_argument('--gb-cpu-note', required=True, help='GB10 CPU gemm の reuse／fresh についての所見（0.13.0 の実測に合わせて書く。数値は生成器が前置する）')
    p.add_argument('--run-ver', default=VER, help='run ディレクトリ内 JSONL 名の版部分・fandhe-ai 行の version の期待値（既定は 0.13.0。0.12.0 データでのドライラン用）')
    p.add_argument('--prev-ver', default=PREV_VER, help='前版の表記（既定は 0.12.0）')
    p.add_argument('--unreached-tsv', default='', help='1.0 以下のセルの理由表 TSV（セルID・機体・比・分類・理由・出典。--emit-unreached-skeleton を使わないときは必須）')
    p.add_argument('--emit-unreached-skeleton', default='', help='判定後に 1.0 以下のセルの雛形 TSV を書き出して終了する（未記入チェックと HTML 生成は行わない）')
    p.add_argument('--prev-out', required=True, help='前版の判定一覧（0.12.0 の gen_0120.out）。tally を 2 条件表に併記する')
    return p.parse_args()


ARGS = parse_args()
PREV_VER = ARGS.prev_ver  # --prev-ver が既定と違うときだけ変わる
EMIT_SKELETON = bool(ARGS.emit_unreached_skeleton)
if not EMIT_SKELETON:
    # 未記入の雛形のまま本番生成しない（推測値を埋めない。skeleton 出力時は判定だけが目的なので緩める）
    _ph = [k for k, v in vars(ARGS).items() if isinstance(v, str) and PLACEHOLDER in v]
    if _ph:
        raise SystemExit('未記入（' + PLACEHOLDER + '）の引数が残っている: ' + ', '.join('--' + k.replace('_', '-') for k in _ph))
    if not ARGS.unreached_tsv:
        raise SystemExit('--unreached-tsv が必要（雛形は --emit-unreached-skeleton で作る）')


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def key(r):
    return (r['framework'], r['task'], r['device'], int(r['size']), r['mode'])


def index(rows):
    d = {}
    for r in rows:
        if r['task'] in ('train_phases', 'infer_phases'):
            continue
        # TF32 opt-in スイープ行（fandhe-ai／candle）は f32 判定から外す。burn は TF32 既定で
        # その行しか存在しないため残す（0.10.0 版と同じ扱い）
        if r.get('tf32') and r['framework'] != 'burn':
            continue
        k = key(r)
        assert k not in d, f'重複行: {k}'
        d[k] = r
    return d


def tf32_rows(rows):
    return {(r['framework'], int(r['size'])): r for r in rows
            if r.get('tf32') and r['task'] == 'gemm' and r['device'] == 'cuda' and r['mode'] == 'fresh'}


RV = ARGS.run_ver


def check_fandhe_version(name, idx):
    """fandhe-ai 行の version が RV（0.13.0）でなければ止める。別版の値を載せないための入口検査。"""
    vs = {r.get('version') for k, r in idx.items() if k[0] == 'fandhe-ai'}
    if vs != {RV}:
        raise SystemExit(f'{name}: fandhe-ai 行の version が {RV} でない: {sorted(map(str, vs))}')


m4 = index(load(ARGS.m4) + load(ARGS.m4_py))
M4_PREV_FILES = [f for f in ARGS.m4_prev if f]  # '' を渡すと「前版比なし」
m4_prev = index(sum((load(f) for f in M4_PREV_FILES), [])) if M4_PREV_FILES else None  # None＝Apple 側は前版比を出さない
gb_raw = load(ARGS.gb)
gb = index(gb_raw + load(ARGS.gb_extra) + load(ARGS.gb_py))
gb_tf32 = tf32_rows(gb_raw)
gb_prev_rows = []
for f in ARGS.gb_prev:
    gb_prev_rows += load(f)
gb_prev = index(gb_prev_rows)
check_fandhe_version('--m4', m4)
check_fandhe_version('--gb', gb)

NRUNS = 5
M4_RUN_FILES = (f'results-macmini-{RV}.jsonl', f'results-macmini-py-{RV}.jsonl')
GB_RUN_FILES = (f'results-dgx-{RV}.jsonl', f'results-dgx-{RV}-extra.jsonl', f'results-dgx-py-{RV}.jsonl')

def load_runs(parent, names):
    """run1〜run5 の索引を run 順に返す。ファイル欠損は open() が例外で止める。"""
    return [index(sum((load(Path(parent) / f'run{i}' / n) for n in names), [])) for i in range(1, NRUNS + 1)]

m4_runs = load_runs(ARGS.m4_runs, M4_RUN_FILES)
gb_runs = load_runs(ARGS.gb_runs, GB_RUN_FILES)
for _i, _d in enumerate(m4_runs, 1):
    check_fandhe_version(f'Mac mini run{_i}', _d)
for _i, _d in enumerate(gb_runs, 1):
    check_fandhe_version(f'GB10 run{_i}', _d)

def gb_env():
    """GB10 の run1〜run5 の env_info.txt 先頭行（`<GPU 名>, <driver>`）を読み、5 本が一致することを確かめて返す。"""
    seen = set()
    for i in range(1, NRUNS + 1):
        first = open(Path(ARGS.gb_runs) / f'run{i}' / 'env_info.txt', encoding='utf-8').readline().strip()
        if not re.fullmatch(r'[^,]+, \d+(\.\d+)*', first):
            raise SystemExit(f'run{i}/env_info.txt の先頭行が `<GPU 名>, <driver>` 形式でない: {first!r}')
        seen.add(first)
    if len(seen) != 1:
        raise SystemExit(f'GB10 の env_info.txt が run 間で一致しない: {sorted(seen)}')
    gpu, drv = seen.pop().split(', ')
    return gpu, drv

GB_GPU, GB_DRIVER = gb_env()

# Apple 機 burn Metal N>=512 は結果テンソル全ゼロで記録拒否（skipped-macmini-<ver>.log。0.8.0／0.9.0／0.10.0 と同一の
# upstream 既知バグによる継続現象）
M4_SKIP = {('burn', 'gemm', 'metal', n): '結果テンソルが全ゼロ（upstream 既知バグ）のため記録拒否' for n in (512, 1024, 2048, 4096)}

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

def paired_ratio(runs, task, device, size):
    """同一 run 内の対戦相手比の 5 run 中央値と run 別の値を返す（RULE.txt の主判定）。

    run ごとに「その run の有効な相手最速 median_s ÷ fandhe-ai reuse median_s」を求める。推論は
    スループット比（fandhe ÷ 相手最速）だが、スループット = 1/median_s のため同じ式になる。
    run 欠損・判定側の要素検証不合格・有効な相手の不在は生成を止める（fail-closed）。"""
    rs = []
    for i, d in enumerate(runs, 1):
        me = d.get(('fandhe-ai', task, device, size, 'reuse'))
        if me is None:
            raise SystemExit(f'run{i}: fandhe-ai reuse 欠損 {task} {device} N={size}')
        if (me.get('parity_fail_count') or 0) > 0:
            raise SystemExit(f'run{i}: fandhe-ai reuse の要素検証が不合格 {task} {device} N={size}')
        comp = [r['median_s'] for fw in FW_ORDER
                for r in [d.get((fw, task, device, size, 'fresh'))]
                if r is not None and (r.get('parity_fail_count') or 0) == 0]
        if not comp:
            raise SystemExit(f'run{i}: 有効な比較相手がない {task} {device} N={size}')
        rs.append(min(comp) / me['median_s'])
    return statistics.median(rs), rs

def judge(data, task, device, size, runs):
    """(区分 cls, 表示 verdict, ratio, best fw, rank, 中央値同士比, run 別比) を返す。

    区分と ratio は同一 run 内比の中央値（paired_ratio）で決める。順位・相手名は 5 ラウンド中央値
    （data）で数える。build_row と対照系列の反転判定で共用する。"""
    me = data.get(('fandhe-ai', task, device, size, 'reuse'))
    assert me is not None, (task, device, size)
    # 判定側（fandhe-ai）の要素検証が不合格なら順位を出さない。判定不能行を集計
    # （判定対象 = 勝ち＋僅差＋負け）へ黙って混ぜないよう、生成自体を止める（fail-closed）
    me_fail = me.get('parity_fail_count') or 0
    if me_fail > 0:
        raise SystemExit(f'fandhe-ai reuse の要素検証が不合格（{me_fail} 要素）: {task} {device} N={size}。判定不能のため勝敗を出さない')
    mine = metric(me, task)
    comp = []
    for fw in FW_ORDER:
        r = data.get((fw, task, device, size, 'fresh'))
        if r is None or (r.get('parity_fail_count') or 0) > 0:
            continue
        comp.append((fw, metric(r, task)))
    if not comp:
        raise SystemExit(f'有効な比較相手がない: {task} {device} N={size}')
    best = None
    for fw, v in comp:
        if best is None or better(v, best[1], task):
            best = (fw, v)
    rank = 1 + sum(1 for fw, v in comp if better(v, mine, task))
    mm_ratio = (mine / best[1]) if task == 'infer' else (best[1] / mine)
    ratio, runs_r = paired_ratio(runs, task, device, size)
    if ratio > 1:
        return 'win', '1 位', ratio, best[0], rank, mm_ratio, runs_r
    if ratio >= 0.90:
        return 'near', '僅差', ratio, best[0], rank, mm_ratio, runs_r
    if rank == 1:
        # 中央値では最速なのに同一 run 内比が 0.90 未満という表示不能な組み合わせ。黙って丸めない
        raise SystemExit(f'順位 1 位だが同一 run 内比 {ratio:.2f}: {task} {device} N={size}')
    return 'loss', f'{rank} 位', ratio, best[0], rank, mm_ratio, runs_r

CLS_JA = {'win': '1 位', 'near': '僅差', 'loss': '負け'}

def build_row(data, data_prev, machine, task, device, size, skip, runs):
    label = {'gemm': f'gemm {DEVNAME[device]} N={size}', 'train': f'train {DEVNAME[device]}', 'infer': f'infer {DEVNAME[device]}'}[task]
    desc = {'gemm': 'f32 正方 GEMM・reuse', 'train': '784→256→10 MLP・バッチ 64・1 step・reuse（デバイス常駐 SGD）', 'infer': '同 MLP forward・バッチ 64・reuse'}[task]
    me = data.get(('fandhe-ai', task, device, size, 'reuse'))
    me_fresh = data.get(('fandhe-ai', task, device, size, 'fresh'))
    cls, verdict, ratio, best_fw, rank, mm_ratio, runs_r = judge(data, task, device, size, runs)
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
        cells.append((fw, txt + mark, metric(r, task)))
    # {PREV_VER} 比
    old = None
    fresh_note = ''
    if data_prev is not None:
        old = data_prev.get(('fandhe-ai', task, device, size, 'reuse'))
        if old is None:
            old = data_prev.get(('fandhe-ai', task, device, size, 'fresh'))
            base = me_fresh if me_fresh is not None else me
            fresh_note = '（fresh 同士）'
        else:
            base = me
    if data_prev is None:
        vprev, vprevs = None, f'{PREV_VER} 比なし（機体変更）'
    elif old is not None:
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
        (f'<td class="num{" best" if fw == best_fw else ""}">{c}</td>' if v is not None else c)
        for fw, c, v in cells)
    tr = (f'<tr class="v-{cls}"><th scope="row"><span class="ph">{label}</span><span class="ds">{desc}</span></th>'
          f'<td><span class="pill {cls}">{verdict}</span><span class="vs">vs {FWNAME[best_fw]}</span><span class="vs">{vprevs}</span></td>'
          f'<td class="ratio">{bar}<span class="rt">{ratio:.2f}×</span></td>'
          f'<td class="num me">{me_txt} <small>reuse</small></td>'
          f'<td class="num me2">{fresh_txt}{" <small>fresh</small>" if me_fresh else ""}</td>'
          f'{cells_html}</tr>')
    return dict(tr=tr, cls=cls, verdict=verdict, ratio=ratio, mm_ratio=mm_ratio, runs_r=runs_r, label=label, machine=machine, best=best_fw, vprev=vprev, vprevs=vprevs,
                me=me, me_fresh=me_fresh, old=old, fresh_note=fresh_note)

DEVNAME = {'metal': 'Metal', 'cpu': 'CPU', 'cuda': 'CUDA'}
FWNAME = {'candle': 'candle', 'burn': 'burn', 'pytorch': 'PyTorch', 'tensorflow': 'TensorFlow', 'scipy': 'SciPy'}

M4_ROWS = [('gemm', 'metal', n) for n in (256, 512, 1024, 2048, 4096)] + [('gemm', 'cpu', n) for n in (256, 512, 1024, 2048)] + \
          [('train', 'metal', 64), ('infer', 'metal', 64), ('train', 'cpu', 64), ('infer', 'cpu', 64)]
GB_ROWS = [('gemm', 'cuda', n) for n in (256, 512, 1024, 2048, 4096)] + [('gemm', 'cpu', n) for n in (256, 512, 1024, 2048, 4096)] + \
          [('train', 'cuda', 64), ('infer', 'cuda', 64), ('train', 'cpu', 64), ('infer', 'cpu', 64)]

m4_rows = [build_row(m4, m4_prev, M4_NAME, *r, M4_SKIP, m4_runs) for r in M4_ROWS]
gb_rows = [build_row(gb, gb_prev, 'GB10', *r, {}, gb_runs) for r in GB_ROWS]
allrows = m4_rows + gb_rows
wins = [r for r in allrows if r['cls'] == 'win']
nears = [r for r in allrows if r['cls'] == 'near']
losses = [r for r in allrows if r['cls'] == 'loss']
# 無効セル（判定不能）数
def count_invalid(data, rows):
    n = 0
    for task, dev, size in rows:
        for fw in FW_ORDER:
            r = data.get((fw, task, dev, size, 'fresh'))
            if r is not None and (r.get('parity_fail_count') or 0) > 0:
                n += 1
    return n
invalid_n = count_invalid(m4, M4_ROWS) + count_invalid(gb, GB_ROWS)

def gb_inv_note():
    """burn CUDA GEMM（TF32 降格・本文で別記）以外の GB10 判定不能セルを注記文にする。"""
    parts = []
    for task, dev, size in GB_ROWS:
        for fw in FW_ORDER:
            r = gb.get((fw, task, dev, size, 'fresh'))
            if r is None or not (r.get('parity_fail_count') or 0):
                continue
            if fw == 'burn' and task == 'gemm' and dev == 'cuda':
                continue
            parts.append(f'{FWNAME[fw]} {DEVNAME[dev]} N={size} は {r["parity_fail_count"]:,} 要素が複合判定を外れ判定不能')
    return ('<b>' + '・'.join(parts) + '</b>（spec REQ-2 (b-1) 上の比較データ側の判定不能。PyTorch CPU N=4096 は #1985／#1989 で現状維持〈T1〉と承認済み）。') if parts else ''

def head(cols):
    return f'<thead><tr><th>対象</th><th>勝敗</th><th>最速他 FW ÷ fandhe-ai</th><th>fandhe-ai {VER}（判定）</th><th>fandhe-ai 別モード（参考）</th>' + ''.join(f'<th>{c}</th>' for c in cols) + '</tr></thead>'

def lst(rows):
    return '、'.join(f'{r["machine"]} {r["label"]}（{r["ratio"]:.2f}×）' for r in rows) or 'なし'

def msprev(data, data_prev, task, dev, size):
    """{PREV_VER}→{VER} の同一モード（reuse 優先）中央値ペアを返す。"""
    old = data_prev.get(('fandhe-ai', task, dev, size, 'reuse'))
    mode = 'reuse'
    if old is None:
        old = data_prev.get(('fandhe-ai', task, dev, size, 'fresh')); mode = 'fresh'
    new = data.get(('fandhe-ai', task, dev, size, mode))
    return old, new, mode

def card_m4_nocompare():
    """Apple 機は前版比を出さない旨の 1 枚カード（--m4-prev に空を渡したとき m4_cards の代わりに使う。機体が違う前版との比較用）。"""
    return (f'<div><h3>{M4_NAME} — 比較なし</h3><span class="v">{PREV_VER} 比は出さない（機体変更）</span>'
            f'<p>{PREV_VER} の Apple 側の記録は{M4_NAME}とは機体が異なる計測のため、同一セルの {PREV_VER} 比は出さない。'
            f'{M4_NAME}の {VER} は上の表（同一ラウンド内で同席計測した対戦相手との比）で判定する。</p></div>')

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

def card_tf32():
    """GB10 CUDA GEMM の TF32 opt-in スイープ（#1983。参考・判定外）を 1 枚のカードにする。"""
    parts = []
    for n in (256, 512, 1024, 2048, 4096):
        cells = []
        for fw, nm in (('fandhe-ai', 'fandhe-ai'), ('candle', 'candle'), ('burn', 'burn')):
            r = gb_tf32.get((fw, n))
            if r is None:
                continue
            gf = r.get('gflops') or (2.0 * n ** 3 / r['median_s'] / 1e9)
            cells.append(f'{nm} {gf:,.0f}')
        parts.append(f'N={n}: ' + '／'.join(cells))
    r4 = {fw: gb_tf32[(fw, 4096)] for fw in ('fandhe-ai', 'candle', 'burn') if (fw, 4096) in gb_tf32}
    gf = {fw: (r.get('gflops') or 2.0 * 4096 ** 3 / r['median_s'] / 1e9) for fw, r in r4.items()}
    best_other = max(v for fw, v in gf.items() if fw != 'fandhe-ai')
    return (f'<div><h3>GB10 CUDA GEMM の TF32 opt-in（参考・判定外）</h3><span class="v">N=4096 で fandhe-ai ÷ 他最速 {gf["fandhe-ai"]/best_other:.2f}×</span>'
            f'<p>GFLOP/s（fresh）。{"・".join(parts)}。fandhe-ai は <code>--tf32</code>（facade の TF32 opt-in・既定 OFF）、candle も同フラグの TF32 経路、burn は TF32 が既定で f32 表の burn 列と同じ行。'
            f'fandhe-ai／candle の TF32 行は精度クラス F32（u=2^-24）のまま判定されるため要素検証を超過する（TF32 精度クラスの適用は burn cuda 行に限る。<code>docs/candle-parity-precision-class-decision.md</code>）。f32 同士の比較ではないため上の表の判定には含めない。</p></div>')

def mm_note():
    """中央値同士の比で判定すると区分が変わる行の説明文（読み方の節）。データから生成する（元版は文言を直書きしていた）。"""
    diff = []
    for r in allrows:
        mm_cls = 'win' if r['mm_ratio'] > 1 else ('near' if r['mm_ratio'] >= 0.90 else 'loss')
        if mm_cls != r['cls']:
            diff.append(f'{r["machine"]} {r["label"]} だけが{CLS_JA[r["cls"]]}（同一ラウンド内比 {r["ratio"]:.2f}×）から{CLS_JA[mm_cls]}（{r["mm_ratio"]:.2f}×）になる')
    if not diff:
        return '中央値同士の比で判定しても区分が変わる行はない。'
    if len(diff) == 1:
        return f'中央値同士の比で判定すると、{diff[0]}。'
    return '中央値同士の比で判定すると、' + '。また'.join(d.replace(' だけが', ' が') for d in diff) + '。'

def cell_id(r):
    """セルの短い ID（M-MTL-G1024・G-CPU-INF など）。check_0130.py の cid() と同じ規則。"""
    mach = 'M' if r['machine'].startswith('Mac') else 'G'
    t = r['label'].split()
    dev = {'Metal': 'MTL', 'CPU': 'CPU', 'CUDA': 'CUDA'}[t[1]]
    sfx = {'gemm': 'G' + t[2][2:] if len(t) > 2 else '', 'train': 'TRN', 'infer': 'INF'}[t[0]]
    return f'{mach}-{dev}-{sfx}'

def print_verdicts():
    """判定一覧（gen_0130.out の中身）を標準出力へ。tally 行は末尾。ノイズ帯は対照系列がないので `noise na`（0 ではない）。"""
    for r in allrows:
        mm_cls = 'win' if r['mm_ratio'] > 1 else ('near' if r['mm_ratio'] >= 0.90 else 'loss')
        mm = f'  [中央値同士比 {r["mm_ratio"]:.2f}x で区分が {CLS_JA[mm_cls]}]' if mm_cls != r['cls'] else ''
        runs_s = ' '.join(f'{x:.2f}' for x in r['runs_r'])
        print(f"{r['machine']:7} {r['label']:22} {r['verdict']:4} vs {r['best']:10} {r['ratio']:.2f}x  run別 [{runs_s}]  {r['vprevs']}{mm}")
    print('tally', len(wins), len(nears), len(losses), 'invalid', invalid_n, 'noise na', 'total', len(allrows))

UNREACHED_CLASSES = ('ハードの上限', '計測窓の定義', '自分たちの制約')  # loss-attribution-matrix.md §10.5 の 3 分類
UNREACHED_HOLD_OWNER_DECIDED = frozenset({'G-CPU-INF'})  # 所有者判断（2026-10-27）を経た分類保留セル
UNREACHED_HOLD = '分類保留'  # 所有者判断（2026-10-27・0.12.0 の #3274）: 数字で 3 分類に割り当てられなかったセルの表示専用ラベル

def unreached_expected():
    """新条件で 1 位でない（比が 1.00 以下の）セルを (セルID, 機体, 比の文字列) の列で返す。TSV の集合・順序・比の照合基準。"""
    return [(cell_id(r), r['machine'], f'{r["ratio"]:.2f}') for r in allrows if r['cls'] != 'win']

def emit_unreached_skeleton(path):
    """1.0 以下のセルの雛形 TSV を書く。セルID・機体・比は入力から生成し、分類・理由・出典は未記入の印にする。"""
    lines = [
        f'# {VER} の 1.0 以下のセルの届かなかった理由（雛形。gen_0130.py --emit-unreached-skeleton の出力）。',
        '# 列: セルID・機体・比（実測。生成器が入力から再計算して照合する）・分類・理由（数字つき。「実測／推定」を明記）・出典。',
        f'# 分類は {"／".join(UNREACHED_CLASSES)} の 3 区分と、表示専用の「{UNREACHED_HOLD}」だけ。「確度低」は分類列に入れず理由列の先頭に書く。',
        f'# {PLACEHOLDER}> が 1 つでも残っていると gen_0130.py は HTML を生成しない。推測値で埋めない（実測・出典のある数字だけを書く）。',
    ]
    for c, m, ratio in unreached_expected():
        lines.append('\t'.join([c, m, ratio, f'{PLACEHOLDER}>', f'{PLACEHOLDER}>', f'{PLACEHOLDER}>']))
    Path(path).write_text('\n'.join(lines) + '\n', encoding='utf-8')

if EMIT_SKELETON:
    print_verdicts()
    emit_unreached_skeleton(ARGS.emit_unreached_skeleton)
    raise SystemExit(0)

CSS = open(Path(__file__).with_name('style.css')).read()
BODY = open(Path(__file__).with_name('body_0130.html')).read()

def unreached_table(path):
    """届かなかったセル（1.0 以下）の理由表を HTML にする。セル集合・機体・比が入力の判定と違う行、分類が 3 種と「分類保留」以外、
    未記入の印が残る行、理由・出典が空の行、列数不一致は SystemExit。理由文と出典は推定を含みうるため、数字つきで
    「実測／推定」を書いた文を受け取る（生成器は判定に使わない）。"""
    rows = []
    for n, line in enumerate(open(path, encoding='utf-8'), 1):
        line = line.rstrip('\n')
        if not line.strip() or line.startswith('#'):
            continue
        cols = line.split('\t')
        if len(cols) != 6:
            raise SystemExit(f'{path}:{n}: 列数が 6 でない（セルID・機体・比・分類・理由・出典）')
        cell, machine, ratio, cls, reason, src = cols
        if any(PLACEHOLDER in c for c in cols):
            raise SystemExit(f'{path}:{n}: 未記入（{PLACEHOLDER}）が残っている: {cell}')
        try:
            ratio_v = float(ratio)
        except ValueError:
            raise SystemExit(f'{path}:{n}: 比が数値でない: {ratio!r}')
        if cls not in UNREACHED_CLASSES and cls != UNREACHED_HOLD:
            raise SystemExit(f'{path}:{n}: 分類が 3 種（{"／".join(UNREACHED_CLASSES)}）と「{UNREACHED_HOLD}」以外: {cls!r}')
        if not reason.strip() or not src.strip():
            raise SystemExit(f'{path}:{n}: 理由と出典は必須')
        rows.append((cell, machine, ratio, ratio_v, cls, reason, src))
    got = [(c, m, r) for c, m, r, *_ in rows]
    exp = unreached_expected()
    if got != exp:
        raise SystemExit(f'{path}: セル・機体・比が入力の判定（1 位でないセル）と一致しない。TSV={got} 入力={exp}')
    if not rows:
        raise SystemExit(f'{path}: 行がない（全セル 1 位の場合は本節を別途書き換える）')
    e = html.escape
    def cls_html(k, cell):
        if k == UNREACHED_HOLD:
            # 所有者判断（2026-10-27）を経たのは 0.12.0 の G-CPU-INF だけ。0.13.0 で新たに生じた保留は所有者判断を経ていない
            src = '所有者判断 2026-10-27' if cell in UNREACHED_HOLD_OWNER_DECIDED else '0.13.0 で新たに生じた保留・所有者判断は未'
            return f'<span class="pill solo" title="数字で 3 分類に割り当てられなかった（{src}）">{e(k)}</span>'
        return e(k)
    body = ''.join(f'<tr class="ur"><th scope="row">{e(c)}</th><td>{e(m)}</td><td class="num">{rv:.2f}×</td><td>{cls_html(k, c)}</td><td class="why">{e(why)}</td><td class="why"><small>{e(s)}</small></td></tr>'
                   for c, m, _r, rv, k, why, s in rows)
    counts = '・'.join(f'{k} {sum(1 for x in rows if x[4] == k)}' for k in UNREACHED_CLASSES + (UNREACHED_HOLD,))
    return (f'<p class="note">対象 {len(rows)} セル（{VER} の比が 1.00 以下）: {counts}。</p>'
            '<div class="wrap"><table class="ur"><thead><tr><th>セル</th><th>機体</th><th>比（実測）</th><th>分類</th><th>理由（数字つき）</th><th>出典</th></tr></thead>'
            f'<tbody>{body}</tbody></table></div>')

def read_tally(path, label):
    """判定一覧（gen_*.out）の tally 行を (win, near, loss, invalid, noise, total) で返す。noise は数値または None（`na`＝未算出）。
    寛容にパースしない。"""
    lines = [l.rstrip('\n') for l in open(path, encoding='utf-8') if l.strip()]
    tl = [l for l in lines if l.startswith('tally ')]
    if len(tl) != 1 or lines[-1] != tl[0]:
        raise SystemExit(f'{path}: tally 行が末尾に 1 行だけではない（{label}）')
    m = re.fullmatch(r'tally (\d+) (\d+) (\d+) invalid (\d+) noise (\d+|na) total (\d+)', tl[0])
    if not m:
        raise SystemExit(f'{path}: tally 行の形式が違う（{label}）: {tl[0]!r}')
    g = m.groups()
    t = (int(g[0]), int(g[1]), int(g[2]), int(g[3]), (None if g[4] == 'na' else int(g[4])), int(g[5]))
    if t[0] + t[1] + t[2] != t[5] or len(lines) - 1 != t[5]:
        raise SystemExit(f'{path}: 勝ち＋僅差＋負け・判定行数が total と合わない（{label}）')
    return t

def gb_cpu_gemm_note():
    """GB10 CPU gemm の fandhe-ai reuse／fresh（5 ラウンド中央値）と、同じ比の前版値を文にする（数値のみ。所見は --gb-cpu-note）。
    表示のみ。判定には使わない。欠損は SystemExit。"""
    parts = []
    for n in (256, 512, 1024, 2048, 4096):
        vals = []
        for d, nm in ((gb, VER), (gb_prev, PREV_VER)):
            ru = d.get(('fandhe-ai', 'gemm', 'cpu', n, 'reuse'))
            fr = d.get(('fandhe-ai', 'gemm', 'cpu', n, 'fresh'))
            if ru is None or fr is None:
                raise SystemExit(f'GB10 CPU gemm N={n} の {nm} reuse／fresh が欠損')
            vals.append((ru['median_s'] * 1e3, fr['median_s'] * 1e3))
        (nr, nf), (pr, pf) = vals
        parts.append(f'N={n} {nr:.3f}／{nf:.3f} ms（{nr / nf:.2f}。{PREV_VER} は {pr:.3f}／{pf:.3f} ms〈{pr / pf:.2f}〉）')
    return '；'.join(parts)

def cuda_reuse_note():
    """GB10 CUDA GEMM の fandhe-ai reuse／fresh（5 ラウンド中央値）と、reuse が fresh より遅いサイズの一覧を文にする（入力から生成）。"""
    parts, slower = [], []
    for n in (256, 512, 1024, 2048, 4096):
        ru = gb.get(('fandhe-ai', 'gemm', 'cuda', n, 'reuse'))
        fr = gb.get(('fandhe-ai', 'gemm', 'cuda', n, 'fresh'))
        if ru is None or fr is None:
            raise SystemExit(f'GB10 CUDA gemm N={n} の reuse／fresh が欠損')
        parts.append(f'N={n} {ru["median_s"] * 1e3:.3f}／{fr["median_s"] * 1e3:.3f} ms')
        if ru['median_s'] > fr['median_s']:
            slower.append(n)
    verdict = ('reuse が fresh より遅いのは N=' + '・'.join(map(str, slower)) + '（reuse は matmul ごとにホスト実体化と checksum を含む計測窓のため。<code>docs/perf/cuda-gemm-reuse-phase-breakdown.md</code>）で、判定列は規則どおり <code>reuse</code> のまま'
               if slower else '全サイズで reuse は fresh 以下で、判定列は規則どおり <code>reuse</code>')
    return f'CUDA GEMM の fandhe-ai reuse／fresh（5 ラウンド中央値）: {"・".join(parts)}。{verdict}。'

def loss_note():
    """学習の最終 loss（train 行の checksum）をフレームワーク別に、両機体の全 run（run1〜run5）から min〜max（小数 6 桁）で文にする。"""
    vals = {}
    for runs in (m4_runs, gb_runs):
        for d in runs:
            for (fw, task, dev, size, mode), r in d.items():
                if task == 'train':
                    vals.setdefault(fw, set()).add(round(float(r['checksum']), 6))
    order = ['fandhe-ai'] + FW_ORDER
    miss = [fw for fw in order if fw not in vals]
    if miss:
        raise SystemExit(f'学習 checksum が run に無いフレームワーク: {miss}')
    parts = []
    for fw in order:
        lo, hi = min(vals[fw]), max(vals[fw])
        parts.append(f'{FWNAME.get(fw, fw)} {lo:.6f}' if lo == hi else f'{FWNAME.get(fw, fw)} {lo:.6f}〜{hi:.6f}')
    return '／'.join(parts)

def cond_tally_table():
    """0.13.0（正式値）と前版の tally を 1 枚の表にする。0.13.0 のノイズ帯は対照系列がないため「未算出」（0 と区別する）。"""
    rows = [(f'{VER}（Mac mini 系列 B のみ・GB10 5 ラウンド）<b>＝正式値</b>', (len(wins), len(nears), len(losses), invalid_n, None, len(allrows)), '本ページの表'),
            (f'{PREV_VER}（Mac mini 系列 B・GB10 5 ラウンド。bench-fandhe 是正後の正式値）', read_tally(ARGS.prev_out, PREV_VER), '<code>gen_0120.out</code>')]
    def noise_cell(v):
        return '未算出（対照系列なし）' if v is None else str(v)
    body = ''.join(f'<tr class="ct"><th scope="row">{name}</th><td class="num">{t[0]}</td><td class="num">{t[1]}</td><td class="num">{t[2]}</td>'
                   f'<td class="num">{t[3]}</td><td class="num">{noise_cell(t[4])}</td><td class="num">{t[5]}</td><td><small>{src}</small></td></tr>'
                   for name, t, src in rows)
    return ('<div class="wrap"><table class="ct"><thead><tr><th>条件</th><th>1 位</th><th>僅差</th><th>負け</th><th>判定不能</th><th>ノイズ帯</th><th>計</th><th>出典</th></tr></thead>'
            f'<tbody>{body}</tbody></table></div>')

def fwver(data, fw):
    vs = {r['version'] for k, r in data.items() if k[0] == fw}
    assert len(vs) == 1, (fw, vs)
    return vs.pop()

out = BODY.format(
    css=CSS,
    n_win=len(wins), n_near=len(nears), n_loss=len(losses), n_invalid=invalid_n, n_total=len(allrows),
    m4_head=head([f'candle-core {fwver(m4, "candle")}', f'burn {fwver(m4, "burn")}', f'PyTorch {fwver(m4, "pytorch")}（MPS）',
                  f'TensorFlow {fwver(m4, "tensorflow")}（Metal プラグイン）', f'SciPy {fwver(m4, "scipy")}']),
    m4_rows=''.join(r['tr'] for r in m4_rows),
    gb_head=head([f'candle-core {fwver(gb, "candle")}', f'burn {fwver(gb, "burn")}', f'PyTorch {fwver(gb, "pytorch")}',
                  f'TensorFlow {fwver(gb, "tensorflow")}（CPU のみ）', f'SciPy {fwver(gb, "scipy")}']),
    gb_rows=''.join(r['tr'] for r in gb_rows),
    win_list=lst(wins), near_list=lst(nears), loss_list=lst(losses),
    m4_load=html.escape(ARGS.m4_load), gb_load=html.escape(ARGS.gb_load), m4_series=html.escape(ARGS.m4_series), gb_inv_note=gb_inv_note(),
    m4_date=html.escape(ARGS.m4_date), m4_skipped=html.escape(ARGS.m4_skipped_note), updated=html.escape(ARGS.updated),
    candle_ver=fwver(m4, 'candle'), burn_ver=fwver(m4, 'burn'), scipy_ver=fwver(m4, 'scipy'),
    pt_ver_m4=fwver(m4, 'pytorch'), pt_ver_gb=fwver(gb, 'pytorch'),
    m4_tf_ver=fwver(m4, 'tensorflow'), gb_tf_ver=fwver(gb, 'tensorflow'),
    gb_gpu=html.escape(GB_GPU), gb_driver=html.escape(GB_DRIVER),
    m4_series_h='・' + html.escape(ARGS.m4_series),
    m4_cond='。'.join(x for x in (html.escape(ARGS.m4_series), html.escape(ARGS.m4_load)) if x),
    m4_prev_note=(f'<b>{M4_NAME}の行は {PREV_VER} 比を出さない</b>（比較欄は「{PREV_VER} 比なし（機体変更）」）。'
                  if m4_prev is None else f'{M4_NAME}の {PREV_VER} も同じ機体（Mac mini）の正式系列 B で計測したため、{PREV_VER} 比を出す。'),
    mm_note=mm_note(),
    m4_cards=(card_m4_nocompare() if m4_prev is None else
              card_gemm(m4, m4_prev, 'metal', (256, 512, 1024, 2048, 4096), f'{M4_NAME} GEMM Metal') + card_gemm(m4, m4_prev, 'cpu', (256, 512, 1024, 2048), f'{M4_NAME} GEMM CPU')
              + card_train(m4, m4_prev, ('metal', 'cpu'), f'{M4_NAME} MLP 学習 1 step') + card_infer(m4, m4_prev, ('metal', 'cpu'), f'{M4_NAME} 推論スループット')),
    gb_cards=card_gemm(gb, gb_prev, 'cuda', (256, 512, 1024, 2048, 4096), 'GB10 GEMM CUDA') + card_gemm(gb, gb_prev, 'cpu', (256, 512, 1024, 2048, 4096), 'GB10 GEMM CPU')
             + card_train(gb, gb_prev, ('cuda', 'cpu'), 'GB10 MLP 学習 1 step') + card_infer(gb, gb_prev, ('cuda', 'cpu'), 'GB10 推論スループット'),
    tf32_card=card_tf32(),
    unreached_table=unreached_table(ARGS.unreached_tsv),
    cond_tally=cond_tally_table(),
    gb_cpu_numbers=gb_cpu_gemm_note(), gb_cpu_note=html.escape(ARGS.gb_cpu_note),
    cuda_reuse_note=cuda_reuse_note(), loss_note=loss_note(),
    release_date=html.escape(ARGS.release_date), commit_count=html.escape(ARGS.commit_count),
    gb_date=html.escape(ARGS.gb_date), gb_skipped=html.escape(ARGS.gb_skipped_note),
    updated_h=html.escape(ARGS.updated),
    ver=VER, ver_nodot=VER.replace('.', ''), prev_ver=PREV_VER,
)
Path(ARGS.out).write_text(out, encoding='utf-8')

# 検証出力（判定一覧。run_0130.sh が gen_0130.out として保存し、check_0130.py が HTML と照合する）
print_verdicts()
