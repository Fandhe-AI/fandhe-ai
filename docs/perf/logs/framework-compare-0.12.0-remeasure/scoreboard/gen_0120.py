#!/usr/bin/env python3
"""fandhe-ai 0.12.0 対戦成績スコアボード（HTML）を framework-compare JSONL から生成する（#3274）。

役割: `gen_0110_macmini.py`（0.11.0・Apple 側は Mac mini）からの派生で、0.12.0 の 5 ラウンド中央値 JSONL
（#3273 の計測成果物）を同じ判定規則で HTML にする。呼び出し元は `docs/perf/logs/framework-compare-0.12.0-remeasure/README.md`
の再現手順 4（実測後に実行する）。body は同ディレクトリの `body_0120.html`、色・クラスは `style.css`（0.11.0 版の逐語コピー）。

0.11.0 版からの差分（表示・入力まわりだけ。判定経路は変えない）:
- VER／PREV_VER を 0.12.0／0.11.0 にした。0.11.0 の Apple 側も Mac mini で計測したため、Apple 側も前版比を出せる
  （--m4-prev は nargs='+'。空で渡すと従来どおり「前版比なし」）。--prev-ver で前版の表記を差し替えられる（回帰確認用）。
- --run-ver: run ディレクトリ内の JSONL 名（results-*-<ver>.jsonl）の版部分だけを差し替える（0.11.0 データでの回帰確認用）。
- --m4-old／--gb-old: 旧条件腕（#3273 の旧条件の fandhe-ai 行・5 run 中央値）を渡すと、fandhe-ai の判定セルに「旧条件 x ms」を
  注記する（表示のみ。旧条件での勝敗は swap_fandhe_arm.py で入力を置き換えて同じ生成器をもう 1 回かける。判定経路は 1 本）。
- --unreached-tsv: 1.0 以下のセルの届かなかった理由（ハードの上限／計測窓の定義／自分たちの制約）の表。分類が 3 種以外・比が
  数値でない行は SystemExit（寛容にパースしない）。未指定なら「0.12.0 は未実測」と 0.11.0 の分類の出典だけを出す。

判定規則（変えていない。0.11.0 版の関数群と字句で同じ）: 同一 run 内の対戦相手比の 5 run 中央値で区分する
（>1 で 1 位、0.90 以上で僅差、それ未満は負け。RULE.txt の主判定）。parity_fail_count>0 のセルは判定不能で順位に含めない。
fandhe-ai／candle の `tf32: true` 行は f32 判定から除外し、burn の `tf32: true` 行は残す（burn は TF32 が既定）。
--m4-alt／--m4-py-alt（対照系列）と区分が反転した行は「ノイズ帯」。欠損・不正値は SystemExit で fail-closed にする。

入力: --m4／--m4-py（Mac mini の fandhe/candle/burn と Python 3 FW の 5 ラウンド中央値）、--gb／--gb-extra／--gb-py（GB10）、
--m4-runs／--m4-alt-runs／--gb-runs（run1〜run5 を含む親ディレクトリ）。
"""
import argparse
import statistics
import json
import html
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]  # docs/perf/logs/<dir>/scoreboard/ から見たリポジトリルート
PREV_GB = REPO / 'docs/perf/logs/framework-compare-0.11.0-remeasure'  # GB10 の前版（0.11.0）の 5 ラウンド中央値の置き場
PREV_M4 = REPO / 'docs/perf/logs/framework-compare-0.11.0-macmini-remeasure'  # Mac mini の前版（同じ機体）
M4_NAME = 'Mac mini（M4）'  # Apple 側の機体名（表の見出し・まとめ節の行ラベルなど生成器が出す全箇所で使う）

VER = '0.12.0'
PREV_VER = '0.11.0'  # 比較元バージョン（この版との差分を「{PREV_VER} 比」として表示する）


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--m4', required=True, help='0.12.0 Apple 機 Mac mini（fandhe/candle/burn。5 ラウンド中央値）JSONL')
    p.add_argument('--m4-py', required=True, help='0.12.0 Apple 機 Mac mini の Python 3 FW（5 ラウンド中央値）JSONL')
    p.add_argument('--m4-alt', default='', help='対照系列の Apple 機（fandhe/candle/burn）JSONL（ノイズ帯判定用・任意）')
    p.add_argument('--m4-py-alt', default='', help='対照系列の Apple 機 Python 3 FW JSONL（ノイズ帯判定用・任意）')
    p.add_argument('--m4-runs', required=True, help='正式系列の Apple 機 run ディレクトリ群の親（run1〜run5 を含む。同一 run 内比の算出用）')
    p.add_argument('--m4-alt-runs', default='', help='対照系列の Apple 機 run ディレクトリ群の親（ノイズ帯の同一 run 内比用。--m4-alt と併用）')
    p.add_argument('--gb-runs', required=True, help='GB10 run ディレクトリ群の親（run1〜run5 を含む）')
    p.add_argument('--m4-prev', nargs='+', default=[
        str(PREV_M4 / 'macmini-series-b/results-macmini-0.11.0-median5.jsonl'),
        str(PREV_M4 / 'macmini-series-b/results-macmini-py-0.11.0-median5.jsonl'),
    ], help='Apple 機（Mac mini）の前版 JSONL 群（正式系列 B）。空を渡すと出さない（Apple 行は「前版比なし」）')
    p.add_argument('--gb', required=True, help='0.12.0 GB10（fandhe/candle/burn 本体）JSONL')
    p.add_argument('--gb-extra', required=True, help='0.12.0 GB10 追加計測（CPU GEMM reuse 等）JSONL')
    p.add_argument('--gb-py', required=True, help='0.12.0 GB10 Python 3 FW（PyTorch/TensorFlow/SciPy）JSONL')
    p.add_argument('--gb-prev', nargs='+', default=[
        str(PREV_GB / 'gb10/results-dgx-0.11.0-median5.jsonl'),
        str(PREV_GB / 'gb10/results-dgx-0.11.0-extra-median5.jsonl'),
    ], help='前版（0.11.0）GB10 JSONL 群（複数可）')
    p.add_argument('--out', default='fandhe-ai-0.12.0-scoreboard.html', help='出力 HTML パス')
    p.add_argument('--m4-load', default='', help='Apple 機（Mac mini）の負荷条件注記文（計測条件節に埋め込む）')
    p.add_argument('--gb-load', default='', help='GB10 の負荷条件注記文（計測条件節に埋め込む）')
    p.add_argument('--m4-series', default='', help='Apple 機（Mac mini）の正式系列の説明（例: 系列 B〈ゲート付き〉を正式採用。空なら省略）')
    p.add_argument('--m4-date', required=True, help='Apple 機（Mac mini）の計測日（eyebrow・見出しに埋め込む。例: 2026-10-11、複数日なら 2026-10-11〜12）')
    p.add_argument('--m4-skipped-note', default='', help='Apple 機の計測失敗の一覧（各ラウンドの skipped-macmini-0.11.0.log の要約。計測条件節に埋め込む。失敗なしなら「なし」）')
    p.add_argument('--updated', default='', help='ページ eyebrow の更新日（実測後に指定。未指定は「未実測」表示）')
    p.add_argument('--gb-date', default='', help='GB10 の計測日（eyebrow・見出しに埋め込む。未指定は「未実測」）')
    p.add_argument('--gb-skipped-note', default='', help='GB10 の計測失敗の一覧（計測条件節に埋め込む。失敗なしなら「なし」）')
    p.add_argument('--run-ver', default=VER, help='run ディレクトリ内 JSONL 名の版部分（既定は 0.12.0。0.11.0 データでの回帰確認用）')
    p.add_argument('--prev-ver', default=PREV_VER, help='前版の表記（既定は 0.11.0。回帰確認で 0.10.0 データを渡すとき用）')
    p.add_argument('--m4-old', default='', help='旧条件腕の Apple 機 fandhe-ai 行（5 run 中央値）JSONL。判定セルに注記するだけ（任意）')
    p.add_argument('--gb-old', default='', help='旧条件腕の GB10 fandhe-ai 行（5 run 中央値）JSONL。判定セルに注記するだけ（任意）')
    p.add_argument('--unreached-tsv', default='', help='1.0 以下のセルの理由表 TSV（セルID・機体・比・分類・理由・出典。任意）')
    return p.parse_args()


ARGS = parse_args()
PREV_VER = ARGS.prev_ver  # --prev-ver が既定と違うときだけ変わる


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


m4 = index(load(ARGS.m4) + load(ARGS.m4_py))
m4_alt = index(load(ARGS.m4_alt) + load(ARGS.m4_py_alt)) if ARGS.m4_alt else None
M4_PREV_FILES = [f for f in ARGS.m4_prev if f]  # '' を渡すと「前版比なし」
m4_prev = index(sum((load(f) for f in M4_PREV_FILES), [])) if M4_PREV_FILES else None  # None＝Apple 側は前版比を出さない
gb_raw = load(ARGS.gb)
gb = index(gb_raw + load(ARGS.gb_extra) + load(ARGS.gb_py))
gb_tf32 = tf32_rows(gb_raw)
gb_prev_rows = []
for f in ARGS.gb_prev:
    gb_prev_rows += load(f)
gb_prev = index(gb_prev_rows)
# 旧条件腕（表示だけ。判定には使わない）。fandhe-ai 行のみの 5 run 中央値
m4_old = index(load(ARGS.m4_old)) if ARGS.m4_old else None
gb_old = index(load(ARGS.gb_old)) if ARGS.gb_old else None

NRUNS = 5
RV = ARGS.run_ver
M4_RUN_FILES = (f'results-macmini-{RV}.jsonl', f'results-macmini-py-{RV}.jsonl')
GB_RUN_FILES = (f'results-dgx-{RV}.jsonl', f'results-dgx-{RV}-extra.jsonl', f'results-dgx-py-{RV}.jsonl')

def load_runs(parent, names):
    """run1〜run5 の索引を run 順に返す。ファイル欠損は open() が例外で止める。"""
    return [index(sum((load(Path(parent) / f'run{i}' / n) for n in names), [])) for i in range(1, NRUNS + 1)]

m4_runs = load_runs(ARGS.m4_runs, M4_RUN_FILES)
m4_alt_runs = load_runs(ARGS.m4_alt_runs, M4_RUN_FILES) if ARGS.m4_alt_runs else None
if (m4_alt is None) != (m4_alt_runs is None):
    raise SystemExit('--m4-alt と --m4-alt-runs は併用する')
gb_runs = load_runs(ARGS.gb_runs, GB_RUN_FILES)

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

def build_row(data, data_prev, machine, task, device, size, skip, runs, alt=None, alt_runs=None, oldarm=None):
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
    if oldarm is not None:
        # 旧条件腕の注記（表示のみ。区分・比には使わない）。欠損は黙って省かず止める
        o = oldarm.get(('fandhe-ai', task, device, size, 'reuse'))
        if o is None:
            raise SystemExit(f'旧条件腕に行がない: {task} {device} N={size}')
        me_txt += f' <small class="oldarm" title="旧条件腕（#3273）の fandhe-ai reuse 中央値">旧条件 {cell_value(o, task, size)[0]}</small>'
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

m4_rows = [build_row(m4, m4_prev, M4_NAME, *r, M4_SKIP, m4_runs, m4_alt, m4_alt_runs, m4_old) for r in M4_ROWS]
gb_rows = [build_row(gb, gb_prev, 'GB10', *r, {}, gb_runs, None, None, gb_old) for r in GB_ROWS]
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

CSS = open(Path(__file__).with_name('style.css')).read()
BODY = open(Path(__file__).with_name('body_0120.html')).read()

UNREACHED_CLASSES = ('ハードの上限', '計測窓の定義', '自分たちの制約')  # loss-attribution-matrix.md §10.5 の 3 分類

def unreached_table(path):
    """届かなかったセル（1.0 以下）の理由表を HTML にする。分類が 3 種以外・比が数値でない行・列数不一致は SystemExit。
    理由文と出典は推定を含みうるため、数字つきで「実測／推定」を書いた文を受け取る（生成器は判定に使わない）。"""
    if not path:
        return ('<p class="note">0.12.0 は未実測のため、このページには 0.12.0 の値での理由表を載せていない（#3273 の実測待ち）。'
                '0.11.0 の 27 セルの分類は <code>docs/perf/loss-attribution-matrix.md</code> §10.5。</p>')
    rows = []
    for n, line in enumerate(open(path, encoding='utf-8'), 1):
        line = line.rstrip('\n')
        if not line.strip() or line.startswith('#'):
            continue
        cols = line.split('\t')
        if len(cols) != 6:
            raise SystemExit(f'{path}:{n}: 列数が 6 でない（セルID・機体・比・分類・理由・出典）')
        cell, machine, ratio, cls, reason, src = cols
        try:
            ratio_v = float(ratio)
        except ValueError:
            raise SystemExit(f'{path}:{n}: 比が数値でない: {ratio!r}')
        if cls not in UNREACHED_CLASSES:
            raise SystemExit(f'{path}:{n}: 分類が 3 種（{"／".join(UNREACHED_CLASSES)}）以外: {cls!r}')
        if not reason.strip() or not src.strip():
            raise SystemExit(f'{path}:{n}: 理由と出典は必須')
        rows.append((cell, machine, ratio_v, cls, reason, src))
    if not rows:
        raise SystemExit(f'{path}: 行がない')
    e = html.escape
    body = ''.join(f'<tr><th scope="row">{e(c)}</th><td>{e(m)}</td><td class="num">{r:.2f}×</td><td>{e(k)}</td><td>{e(why)}</td><td><small>{e(s)}</small></td></tr>'
                   for c, m, r, k, why, s in rows)
    return ('<div class="wrap"><table><thead><tr><th>セル</th><th>機体</th><th>比（実測）</th><th>分類</th><th>理由（数字つき）</th><th>出典</th></tr></thead>'
            f'<tbody>{body}</tbody></table></div>')

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
    win_list=lst(wins), near_list=lst(nears), loss_list=lst(losses),
    noise_list=(noise_list(noisy) if m4_alt is not None else '対照系列は置いていない（ノイズ帯の判定なし）。'),
    m4_load=ARGS.m4_load, gb_load=ARGS.gb_load, m4_series=ARGS.m4_series, gb_inv_note=gb_inv_note(),
    m4_date=ARGS.m4_date, m4_skipped=ARGS.m4_skipped_note, updated=ARGS.updated,
    m4_tf_ver=fwver(m4, 'tensorflow'),
    m4_series_h=('・' + ARGS.m4_series) if ARGS.m4_series else '',
    m4_cond='。'.join(x for x in (ARGS.m4_series, ARGS.m4_load) if x),
    m4_prev_note=(f'<b>{M4_NAME}の行は {PREV_VER} 比を出さない</b>（比較欄は「{PREV_VER} 比なし（機体変更）」）。'
                  if m4_prev is None else f'{M4_NAME}の {PREV_VER} も同じ機体（Mac mini）の正式系列で計測したため、{PREV_VER} 比を出す。'),
    m4_noise_sentence=('「ノイズ帯」の印は、もう一方の系列（ゲート有無の対照系列）で勝敗区分が反転した行（' + str(len(noisy)) + ' 行。下のまとめ節に一覧）。'
                       if m4_alt is not None else '対照系列は置いていないため「ノイズ帯」の印は付かない。'),
    mm_note=mm_note(),
    m4_cards=(card_m4_nocompare() if m4_prev is None else
              card_gemm(m4, m4_prev, 'metal', (256, 512, 1024, 2048, 4096), f'{M4_NAME} GEMM Metal') + card_gemm(m4, m4_prev, 'cpu', (256, 512, 1024, 2048), f'{M4_NAME} GEMM CPU')
              + card_train(m4, m4_prev, ('metal', 'cpu'), f'{M4_NAME} MLP 学習 1 step') + card_infer(m4, m4_prev, ('metal', 'cpu'), f'{M4_NAME} 推論スループット')),
    gb_cards=card_gemm(gb, gb_prev, 'cuda', (256, 512, 1024, 2048, 4096), 'GB10 GEMM CUDA') + card_gemm(gb, gb_prev, 'cpu', (256, 512, 1024, 2048, 4096), 'GB10 GEMM CPU')
             + card_train(gb, gb_prev, ('cuda', 'cpu'), 'GB10 MLP 学習 1 step') + card_infer(gb, gb_prev, ('cuda', 'cpu'), 'GB10 推論スループット'),
    tf32_card=card_tf32(),
    unreached_table=unreached_table(ARGS.unreached_tsv),
    gb_date=html.escape(ARGS.gb_date) if ARGS.gb_date else '未実測',
    gb_skipped=html.escape(ARGS.gb_skipped_note) if ARGS.gb_skipped_note else '記録なし',
    updated_h=html.escape(ARGS.updated) if ARGS.updated else '未実測',
    ver=VER, prev_ver=PREV_VER,
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
