#!/usr/bin/env python3
"""0.12.0 スコアボード HTML と判定一覧（gen_0120.out ほか）の機械照合（イシュー #3274）。

役割: run_0120.sh の最後に呼ばれ、生成した HTML の表の行数・各行の区分・比・相手・前版比、tally、3 条件の tally 表、
届かなかったセルの表（unreached_0120.tsv）が、判定一覧と一致することを確かめる。あわせて、未実測の placeholder・
実ホスト名・外部参照（フォントの stylesheet 以外）が HTML に残っていないことを確かめる。
寛容にパースしない: 形式が違う・件数が違う・値が違うときはすべて SystemExit（rc=1）。判定はしない（照合だけ）。
使い方: check_0120.py HTML NEW.out OLDARM.out PREV.out UNREACHED.tsv
"""
import html as H
import re
import sys

ROW_RE = re.compile(r'^(Mac mini（M4）|GB10)\s+(gemm \S+ N=\d+|train \S+|infer \S+)\s+(1 位|僅差|\d 位)\s+vs (\S+)\s+(\d+\.\d\d)x  run別 \[([\d. ]+)\]  (.+?)(  \[.*)?$')
TALLY_RE = re.compile(r'^tally (\d+) (\d+) (\d+) invalid (\d+) noise (\d+) total (\d+)$')
FW = {'candle': 'candle', 'burn': 'burn', 'pytorch': 'PyTorch', 'tensorflow': 'TensorFlow', 'scipy': 'SciPy'}
N_M4, N_GB = 13, 14


def fail(msg):
    raise SystemExit(f'CHECK-FAIL: {msg}')


def read_out(path):
    rows, tally = [], None
    for n, l in enumerate(open(path, encoding='utf-8'), 1):
        l = l.rstrip('\n')
        if not l.strip():
            continue
        m = TALLY_RE.match(l)
        if m:
            if tally is not None:
                fail(f'{path}: tally 行が 2 行以上')
            tally = tuple(int(x) for x in m.groups())
            continue
        if tally is not None:
            fail(f'{path}:{n}: tally 行の後に行がある')
        m = ROW_RE.match(l)
        if not m:
            fail(f'{path}:{n}: 判定行の形式が違う: {l!r}')
        mach, label, verdict, best, ratio, runs, vprev, _ = m.groups()
        if len(runs.split()) != 5:
            fail(f'{path}:{n}: run 別の比が 5 個でない')
        rows.append(dict(machine=mach, label=label, verdict=verdict, best=best, ratio=ratio, vprev=vprev.strip()))
    if tally is None:
        fail(f'{path}: tally 行がない')
    if len(rows) != tally[5] or tally[0] + tally[1] + tally[2] != tally[5]:
        fail(f'{path}: 判定行数 {len(rows)} と tally {tally} が合わない')
    return rows, tally


def cls_of(verdict):
    return 'win' if verdict == '1 位' else ('near' if verdict == '僅差' else 'loss')


def main(html_path, new_out, old_out, prev_out, tsv_path):
    doc = open(html_path, encoding='utf-8').read()
    new_rows, new_t = read_out(new_out)
    _, old_t = read_out(old_out)
    _, prev_t = read_out(prev_out)
    if len(new_rows) != N_M4 + N_GB:
        fail(f'新条件の判定行が {len(new_rows)} 行（期待 {N_M4 + N_GB}）')
    if [r['machine'] for r in new_rows] != ['Mac mini（M4）'] * N_M4 + ['GB10'] * N_GB:
        fail('新条件の機体の並びが Mac mini 13 行 → GB10 14 行でない')

    # 1. 判定表の行（順序・ラベル・区分・比・相手・前版比）
    trs = re.findall(r'<tr class="v-(win|near|loss)">(.*?)</tr>', doc, re.S)
    if len(trs) != len(new_rows):
        fail(f'HTML の判定行が {len(trs)} 行（判定一覧は {len(new_rows)} 行）')
    for i, ((cls, tr), r) in enumerate(zip(trs, new_rows), 1):
        label = re.search(r'<span class="ph">(.*?)</span>', tr).group(1)
        pill = re.search(r'<span class="pill (win|near|loss)">(.*?)</span>', tr)
        vs = re.findall(r'<span class="vs">(.*?)</span>', tr)
        rt = re.search(r'<span class="rt">(\d+\.\d\d)×</span>', tr).group(1)
        exp = (r['label'], cls_of(r['verdict']), r['verdict'], f'vs {FW[r["best"]]}', r['ratio'], r['vprev'])
        got = (label, cls, pill.group(2), vs[0] if vs else None, rt, vs[1] if len(vs) > 1 else None)
        if pill.group(1) != cls or got != exp:
            fail(f'判定行 {i}: HTML {got} ≠ 判定一覧 {exp}')

    # 2. tally（先頭の 4 枚）
    tally = re.search(r'<div class="tally"><div class="w"><b>(\d+)</b>.*?<div class="n"><b>(\d+)</b>.*?<div class="l"><b>(\d+)</b>.*?<div><b>(\d+)</b><span>[^<]*判定対象 (\d+) 行', doc, re.S)
    if not tally or tuple(int(x) for x in tally.groups()) != (new_t[0], new_t[1], new_t[2], new_t[3], new_t[5]):
        fail(f'HTML の tally {tally.groups() if tally else None} ≠ 新条件 {new_t}')

    # 3. 3 条件の tally 表（新条件・旧条件・前版の順）
    cts = re.findall(r'<tr class="ct">(.*?)</tr>', doc, re.S)
    got_ct = [tuple(int(x) for x in re.findall(r'<td class="num">(\d+)</td>', c)) for c in cts]
    if got_ct != [new_t, old_t, prev_t]:
        fail(f'3 条件の tally 表 {got_ct} ≠ {[new_t, old_t, prev_t]}')

    # 4. 届かなかったセル: TSV の集合と比 == 新条件で 1 位でない行。HTML の表は TSV と同じ行
    def cid(r):
        mach = 'M' if r['machine'].startswith('Mac') else 'G'
        t = r['label'].split()
        dev = {'Metal': 'MTL', 'CPU': 'CPU', 'CUDA': 'CUDA'}[t[1]]
        sfx = {'gemm': 'G' + t[2][2:] if len(t) > 2 else '', 'train': 'TRN', 'infer': 'INF'}[t[0]]
        return f'{mach}-{dev}-{sfx}'
    exp_ur = [(cid(r), r['ratio']) for r in new_rows if r['verdict'] != '1 位']
    tsv = []
    for n, l in enumerate(open(tsv_path, encoding='utf-8'), 1):
        l = l.rstrip('\n')
        if not l.strip() or l.startswith('#'):
            continue
        c = l.split('\t')
        if len(c) != 6:
            fail(f'{tsv_path}:{n}: 列数が 6 でない')
        tsv.append(c)
    if [(c[0], c[2]) for c in tsv] != exp_ur:
        fail(f'TSV のセルと比 {[(c[0], c[2]) for c in tsv]} ≠ 新条件の 1 位以外 {exp_ur}')
    hold = [c[0] for c in tsv if c[3] == '分類保留']
    if hold != ['G-CPU-INF']:
        fail(f'分類保留のセルが G-CPU-INF だけでない: {hold}')
    urs = re.findall(r'<tr class="ur"><th scope="row">(.*?)</th><td>.*?</td><td class="num">(\d+\.\d\d)×</td><td>(.*?)</td>', doc, re.S)
    got_ur = [(a, b, re.sub(r'<[^>]+>', '', k)) for a, b, k in urs]
    if got_ur != [(c[0], c[2], H.escape(c[3])) for c in tsv]:
        fail(f'HTML の届かなかったセルの表 {got_ur} ≠ TSV')

    # 5. GB10 CPU gemm の後退の注記に書いた旧条件・新条件の比が判定一覧と一致するか
    old_rows, _ = read_out(old_out)
    gcpu = [f'gemm CPU N={n}' for n in (256, 512, 1024, 2048, 4096)]
    pick = lambda rows: '／'.join(next(r['ratio'] for r in rows if r['machine'] == 'GB10' and r['label'] == g) for g in gcpu)
    want = f'旧条件の比 {pick(old_rows)} → 新条件 {pick(new_rows)}'
    if want not in doc:
        fail(f'GB10 CPU gemm の注記に「{want}」がない')
    if '#3646' not in doc:
        fail('GB10 CPU gemm の切り分け先 #3646 の記載がない')

    # 6. placeholder・実ホスト名・外部参照
    text = re.sub(r'<style>.*?</style>', '', doc, flags=re.S)
    for pat in ('（未実測）', '未実測のため', '実測後に', '計測後に記入', '実測待ち', '>未実測<', ' 未実測 '):
        if pat in text:
            fail(f'HTML に残してはいけない文字列: {pat!r}')
    if re.search(r'\{[a-z_0-9]+\}', text):
        fail(f'テンプレートの差し込み口が残っている: {re.findall(r"[{][a-z_0-9]+[}]", text)}')
    if re.search(r'\bspark-[0-9a-z]+|mac-mini|\.local\b|/Users/|/home/', doc, re.I):
        fail('HTML に実ホスト名・絶対パスらしき文字列がある')
    if re.search(r'<script|<img|<iframe', doc, re.I):
        fail('HTML に script／img／iframe がある（1 ファイルで完結させる）')
    links = re.findall(r'<link[^>]*href="([^"]+)"', doc)
    if any(not h.startswith('https://fonts.googleapis.com/') for h in links):
        fail(f'フォント以外の外部参照: {links}')
    if '<meta charset="utf-8">' not in doc:
        fail('meta charset がない')

    print(f'check ok: 判定行 {len(trs)}（Mac mini {N_M4}・GB10 {N_GB}）・tally 新 {new_t[:4]} 旧 {old_t[:4]} 前版 {prev_t[:4]}・'
          f'届かなかったセル {len(tsv)}（分類保留 {len(hold)}）')


if __name__ == '__main__':
    if len(sys.argv) != 6:
        sys.exit(__doc__)
    main(*sys.argv[1:])
