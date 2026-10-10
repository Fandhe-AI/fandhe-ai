#!/usr/bin/env python3
"""fandhe-ai 0.11.0 対戦成績スコアボード（HTML）を framework-compare JSONL から生成する。

gen_0100.py（0.10.0 版・0.9.0 比）の改変版。0.11.0 の結果を、直前版 0.10.0 と比較する。
判定ロジック（同一 run 内比・ノイズ帯・TF32 opt-in 行の除外・fail-closed）は 0.10.0 版と同じ。
0.10.0 版からの変更は版数・入力ファイル名・比較元（0.10.0 の 5 ラウンド中央値）・本文テンプレート名だけ。
以下の箇条は 0.10.0 版から引き継いだ説明（「0.9.0 比」は本版では「0.10.0 比」）。

- M4 Max: --m4（fandhe/candle/burn・5 ラウンド中央値）＋ --m4-py（PyTorch/TensorFlow/SciPy・
  同じ 5 ラウンドで同席計測した中央値）。0.9.0 版にあった Python 3 FW の転記値（M4_PY）は
  0.10.0 で実測へ置き換えたため撤去した。
- M4 Max の対照系列: --m4-alt／--m4-py-alt（正式でない側の系列の 5 ラウンド中央値）を渡すと、
  正式系列と勝敗区分（1 位／僅差／負け）が反転した行を「ノイズ帯」として印を付ける
  （事前登録規則 RULE.txt の「A/B 間で verdict が反転したセルをノイズ帯として明示する」）。
- △ の title は gen_1988.py（#1988）と同じく出し分ける: burn の `tf32: true` 行は精度クラス TF32
  （u=2^-11。#1989 承認・#2046 実装）による救済、それ以外は #1241 承認契約（u=2^-24）による救済。
- GB10: --gb ＋ --gb-extra ＋ --gb-py（fandhe-ai/candle/burn と Python 3 FW を同一セッションで計測）。
  run_all_cuda.sh の TF32 opt-in スイープ（#1983。fandhe-ai／candle の `"tf32": true` 行）は
  f32 同士の判定から除外する（burn は TF32 が既定でその行しかないため従来どおり残す）。
- 勝敗区分と「最速他 FW ÷ fandhe-ai」の比は、事前登録規則 RULE.txt の主判定「同一 run 内の対戦相手比」で
  出す（--m4-runs／--m4-alt-runs／--gb-runs の run1〜run5）。run ごとに「その run の有効な相手最速の
  median_s ÷ fandhe-ai reuse の median_s」（推論スループット比と同値）を求め、5 run の中央値を採る。
  表のセル値・順位（N 位）・「vs」の相手名は従来どおり framework ごとに選んだ 5 ラウンド中央値から出す。
  中央値同士の比で区分が変わる行は標準出力に「中央値同士比」として併記する（PR #2498 codex P2）。
- 0.10.0 比: --m4-prev（既定 0.10.0 系列 B の 5 ラウンド中央値）／--gb-prev（既定 0.10.0 の
  results-dgx-0.10.0{,-extra}-median5.jsonl。0.10.0 から GB10 も 5 ラウンド中央値）。
"""
import argparse
import statistics
import json
import html
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]  # docs/perf/logs/<dir>/scoreboard/ から見たリポジトリルート
PREV = REPO / 'docs/perf/logs/framework-compare-0.10.0-remeasure'

VER = '0.11.0'
PREV_VER = '0.10.0'  # 比較元バージョン（この版との差分を「{PREV_VER} 比」として表示する）


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--m4', required=True, help='0.11.0 M4 Max（fandhe/candle/burn。5 ラウンド中央値）JSONL')
    p.add_argument('--m4-py', required=True, help='0.11.0 M4 Max Python 3 FW（5 ラウンド中央値）JSONL')
    p.add_argument('--m4-alt', default='', help='対照系列の M4 Max（fandhe/candle/burn）JSONL（ノイズ帯判定用・任意）')
    p.add_argument('--m4-py-alt', default='', help='対照系列の M4 Max Python 3 FW JSONL（ノイズ帯判定用・任意）')
    p.add_argument('--m4-runs', required=True, help='正式系列の M4 Max run ディレクトリ群の親（run1〜run5 を含む。同一 run 内比の算出用）')
    p.add_argument('--m4-alt-runs', default='', help='対照系列の M4 Max run ディレクトリ群の親（ノイズ帯の同一 run 内比用。--m4-alt と併用）')
    p.add_argument('--gb-runs', required=True, help='GB10 run ディレクトリ群の親（run1〜run5 を含む）')
    p.add_argument('--m4-prev', default=str(PREV / 'm4max-series-b/results-m4max-0.10.0-median5.jsonl'), help='0.10.0 M4 Max JSONL')
    p.add_argument('--gb', required=True, help='0.11.0 GB10（fandhe/candle/burn 本体）JSONL')
    p.add_argument('--gb-extra', required=True, help='0.11.0 GB10 追加計測（CPU GEMM reuse 等）JSONL')
    p.add_argument('--gb-py', required=True, help='0.11.0 GB10 Python 3 FW（PyTorch/TensorFlow/SciPy）JSONL')
    p.add_argument('--gb-prev', nargs='+', default=[
        str(PREV / 'gb10/results-dgx-0.10.0-median5.jsonl'),
        str(PREV / 'gb10/results-dgx-0.10.0-extra-median5.jsonl'),
    ], help='0.10.0 GB10 JSONL 群（複数可）')
    p.add_argument('--out', default='fandhe-ai-0.11.0-scoreboard.html', help='出力 HTML パス')
    p.add_argument('--m4-load', default='', help='M4 Max の負荷条件注記文（計測条件節に埋め込む）')
    p.add_argument('--gb-load', default='', help='GB10 の負荷条件注記文（計測条件節に埋め込む）')
    p.add_argument('--m4-series', default='', help='M4 Max の正式系列の説明（例: 系列 B〈load1 < 8.0 ゲート付き〉を正式採用）')
    return p.parse_args()


ARGS = parse_args()


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
        # その行しか存在しないため残す（0.9.0 版と同じ扱い）
        if r.get('tf32') and r['framework'] != 'burn':
            continue
        k = key(r)
        assert k not in d, f'重複行: {k}'
        d[k] = r
    return d


def tf32_rows(rows):
    return {(r['framework'], int(r['size'])): r for r in rows
            if r.get('tf32') and r['task'] == 'gemm' and r['device'] == 'cuda' and r['mode'] == 'fresh'}


m4 = index(load(ARGS.m4) + load(ARGS.m4_py))
m4_alt = index(load(ARGS.m4_alt) + load(ARGS.m4_py_alt)) if ARGS.m4_alt else None
m4_prev = index(load(ARGS.m4_prev))
gb_raw = load(ARGS.gb)
gb = index(gb_raw + load(ARGS.gb_extra) + load(ARGS.gb_py))
gb_tf32 = tf32_rows(gb_raw)
gb_prev_rows = []
for f in ARGS.gb_prev:
    gb_prev_rows += load(f)
gb_prev = index(gb_prev_rows)

NRUNS = 5
M4_RUN_FILES = ('results-m4max-0.11.0.jsonl', 'results-m4max-py-0.11.0.jsonl')
GB_RUN_FILES = ('results-dgx-0.11.0.jsonl', 'results-dgx-0.11.0-extra.jsonl', 'results-dgx-py-0.11.0.jsonl')

def load_runs(parent, names):
    """run1〜run5 の索引を run 順に返す。ファイル欠損は open() が例外で止める。"""
    return [index(sum((load(Path(parent) / f'run{i}' / n) for n in names), [])) for i in range(1, NRUNS + 1)]

m4_runs = load_runs(ARGS.m4_runs, M4_RUN_FILES)
m4_alt_runs = load_runs(ARGS.m4_alt_runs, M4_RUN_FILES) if ARGS.m4_alt_runs else None
if (m4_alt is None) != (m4_alt_runs is None):
    raise SystemExit('--m4-alt と --m4-alt-runs は併用する')
gb_runs = load_runs(ARGS.gb_runs, GB_RUN_FILES)

# M4 Max burn Metal N>=512 は結果テンソル全ゼロで記録拒否（skipped-m4max-0.11.0.log。0.8.0〜0.10.0 と同一の
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

def build_row(data, data_prev, machine, task, device, size, skip, runs, alt=None, alt_runs=None):
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
    # 対照系列との区分反転（ノイズ帯）
    noise = None
    if alt is not None:
        acls, averdict, aratio, abest = judge(alt, task, device, size, alt_runs)[:4]
        if acls != cls:
            noise = dict(cls=acls, verdict=averdict, ratio=aratio, best=abest)
    # {PREV_VER} 比
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
        (f'<td class="num{" best" if fw == best_fw else ""}">{c}</td>' if v is not None else c)
        for fw, c, v in cells)
    noise_html = ''
    if noise:
        noise_html = (f'<span class="vs noise" title="対照系列では {CLS_JA[noise["cls"]]}（{noise["ratio"]:.2f}×・vs {FWNAME[noise["best"]]}）">'
                      f'ノイズ帯: 対照系列は{noise["verdict"]}</span>')
    tr = (f'<tr class="v-{cls}"><th scope="row"><span class="ph">{label}</span><span class="ds">{desc}</span></th>'
          f'<td><span class="pill {cls}">{verdict}</span><span class="vs">vs {FWNAME[best_fw]}</span><span class="vs">{vprevs}</span>{noise_html}</td>'
          f'<td class="ratio">{bar}<span class="rt">{ratio:.2f}×</span></td>'
          f'<td class="num me">{me_txt} <small>reuse</small></td>'
          f'<td class="num me2">{fresh_txt}{" <small>fresh</small>" if me_fresh else ""}</td>'
          f'{cells_html}</tr>')
    return dict(tr=tr, cls=cls, verdict=verdict, ratio=ratio, mm_ratio=mm_ratio, runs_r=runs_r, label=label, machine=machine, best=best_fw, vprev=vprev, vprevs=vprevs,
                me=me, me_fresh=me_fresh, old=old, fresh_note=fresh_note, noise=noise)

DEVNAME = {'metal': 'Metal', 'cpu': 'CPU', 'cuda': 'CUDA'}
FWNAME = {'candle': 'candle', 'burn': 'burn', 'pytorch': 'PyTorch', 'tensorflow': 'TensorFlow', 'scipy': 'SciPy'}

M4_ROWS = [('gemm', 'metal', n) for n in (256, 512, 1024, 2048, 4096)] + [('gemm', 'cpu', n) for n in (256, 512, 1024, 2048)] + \
          [('train', 'metal', 64), ('infer', 'metal', 64), ('train', 'cpu', 64), ('infer', 'cpu', 64)]
GB_ROWS = [('gemm', 'cuda', n) for n in (256, 512, 1024, 2048, 4096)] + [('gemm', 'cpu', n) for n in (256, 512, 1024, 2048, 4096)] + \
          [('train', 'cuda', 64), ('infer', 'cuda', 64), ('train', 'cpu', 64), ('infer', 'cpu', 64)]

m4_rows = [build_row(m4, m4_prev, 'M4 Max', *r, M4_SKIP, m4_runs, m4_alt, m4_alt_runs) for r in M4_ROWS]
gb_rows = [build_row(gb, gb_prev, 'GB10', *r, {}, gb_runs) for r in GB_ROWS]
allrows = m4_rows + gb_rows
wins = [r for r in allrows if r['cls'] == 'win']
nears = [r for r in allrows if r['cls'] == 'near']
losses = [r for r in allrows if r['cls'] == 'loss']
noisy = [r for r in allrows if r['noise']]
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

def noise_list(rows):
    if not rows:
        return '対照系列との間で勝敗区分が反転した行はない。'
    return '、'.join(f'{r["machine"]} {r["label"]}（正式 {r["verdict"]} {r["ratio"]:.2f}× ／ 対照 {r["noise"]["verdict"]} {r["noise"]["ratio"]:.2f}×）' for r in rows)

def msprev(data, data_prev, task, dev, size):
    """{PREV_VER}→{VER} の同一モード（reuse 優先）中央値ペアを返す。"""
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

CSS = open(Path(__file__).with_name('style.css')).read()
BODY = open(Path(__file__).with_name('body_0110.html')).read()

def fwver(data, fw):
    vs = {r['version'] for k, r in data.items() if k[0] == fw}
    assert len(vs) == 1, (fw, vs)
    return vs.pop()

out = BODY.format(
    css=CSS,
    n_win=len(wins), n_near=len(nears), n_loss=len(losses), n_invalid=invalid_n, n_total=len(allrows), n_noise=len(noisy),
    m4_head=head([f'candle-core {fwver(m4, "candle")}', f'burn {fwver(m4, "burn")}', f'PyTorch {fwver(m4, "pytorch")}（MPS）',
                  f'TensorFlow {fwver(m4, "tensorflow")}（Metal プラグイン）', f'SciPy {fwver(m4, "scipy")}']),
    m4_rows=''.join(r['tr'] for r in m4_rows),
    gb_head=head([f'candle-core {fwver(gb, "candle")}', f'burn {fwver(gb, "burn")}', f'PyTorch {fwver(gb, "pytorch")}',
                  f'TensorFlow {fwver(gb, "tensorflow")}（CPU のみ）', f'SciPy {fwver(gb, "scipy")}']),
    gb_rows=''.join(r['tr'] for r in gb_rows),
    win_list=lst(wins), near_list=lst(nears), loss_list=lst(losses), noise_list=noise_list(noisy),
    m4_load=ARGS.m4_load, gb_load=ARGS.gb_load, m4_series=ARGS.m4_series, gb_inv_note=gb_inv_note(),
    m4_cards=card_gemm(m4, m4_prev, 'metal', (256, 512, 1024, 2048, 4096), 'M4 Max GEMM Metal') + card_gemm(m4, m4_prev, 'cpu', (256, 512, 1024, 2048), 'M4 Max GEMM CPU')
             + card_train(m4, m4_prev, ('metal', 'cpu'), 'M4 Max MLP 学習 1 step') + card_infer(m4, m4_prev, ('metal', 'cpu'), 'M4 Max 推論スループット'),
    gb_cards=card_gemm(gb, gb_prev, 'cuda', (256, 512, 1024, 2048, 4096), 'GB10 GEMM CUDA') + card_gemm(gb, gb_prev, 'cpu', (256, 512, 1024, 2048, 4096), 'GB10 GEMM CPU')
             + card_train(gb, gb_prev, ('cuda', 'cpu'), 'GB10 MLP 学習 1 step') + card_infer(gb, gb_prev, ('cuda', 'cpu'), 'GB10 推論スループット'),
    tf32_card=card_tf32(),
)
Path(ARGS.out).write_text(out)

# 検証出力
for r in allrows:
    nz = f'  [ノイズ帯: 対照 {r["noise"]["verdict"]} {r["noise"]["ratio"]:.2f}x]' if r['noise'] else ''
    mm_cls = 'win' if r['mm_ratio'] > 1 else ('near' if r['mm_ratio'] >= 0.90 else 'loss')
    mm = f'  [中央値同士比 {r["mm_ratio"]:.2f}x で区分が {CLS_JA[mm_cls]}]' if mm_cls != r['cls'] else ''
    runs_s = ' '.join(f'{x:.2f}' for x in r['runs_r'])
    print(f"{r['machine']:7} {r['label']:22} {r['verdict']:4} vs {r['best']:10} {r['ratio']:.2f}x  run別 [{runs_s}]  {r['vprevs']}{nz}{mm}")
print('tally', len(wins), len(nears), len(losses), 'invalid', invalid_n, 'noise', len(noisy), 'total', len(allrows))
