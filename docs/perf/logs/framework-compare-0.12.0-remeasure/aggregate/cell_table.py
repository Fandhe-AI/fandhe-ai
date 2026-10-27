#!/usr/bin/env python3
"""0.12.0 の 3 条件（新条件・旧条件・0.11.0）の判定一覧を 1 枚の Markdown 表に並べる（イシュー #3273 の README「結果」節の生成器）。

役割: `scoreboard/gen_0120.py` の標準出力（判定一覧）を 3 本受け取り、セル（機体・ラベル）ごとに区分・同一 run 内比の 5 run 中央値・
最速相手を横に並べる。判定はしない（区分・比は gen 系の出力をそのまま写す）。セルの集合・順序が 3 本で一致しなければ非零で終わる
（寛容にパースしない。欠損・重複・書式外の行は SystemExit）。
使い方: cell_table.py NEW.out OLD.out PREV.out > table.md
  NEW.out  = 新条件（0.12.0 bench-fandhe）の gen_0120.py 出力
  OLD.out  = 旧条件（swap_fandhe_arm.py で fandhe-ai 行を旧腕に置き換えた入力）の gen_0120.py 出力
  PREV.out = 0.11.0 の判定一覧（framework-compare-0.11.0-macmini-remeasure/scoreboard/gen_0110_macmini.out）
"""
import re
import sys

LINE = re.compile(r'^(?P<machine>Mac mini（M4）|GB10)\s+(?P<label>.+?)\s{2,}(?P<verdict>\d 位|僅差)\s+vs (?P<best>\S+)\s+(?P<ratio>\d+\.\d+)x\s+run別 \[(?P<runs>[0-9. ]+)\]')


def parse(path):
    cells, order, tally = {}, [], None
    for raw in open(path, encoding='utf-8'):
        line = raw.rstrip('\n')
        if not line.strip():
            continue
        if line.startswith('tally '):
            if tally is not None:
                raise SystemExit(f'{path}: tally 行が複数')
            tally = line
            continue
        m = LINE.match(line)
        if not m:
            raise SystemExit(f'{path}: 書式外の行: {line}')
        k = (m['machine'], m['label'].strip())
        if k in cells:
            raise SystemExit(f'{path}: 重複セル: {k}')
        cells[k] = m.groupdict()
        order.append(k)
    if tally is None:
        raise SystemExit(f'{path}: tally 行がない')
    return cells, order, tally


def fmt(c):
    return f"{c['verdict']} {c['ratio']}×（vs {c['best']}）"


def main():
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    new, order, t_new = parse(sys.argv[1])
    old, order_old, t_old = parse(sys.argv[2])
    prev, order_prev, t_prev = parse(sys.argv[3])
    if order != order_old or order != order_prev:
        raise SystemExit('3 本のセル集合・順序が一致しない')
    for machine in ('Mac mini（M4）', 'GB10'):
        print(f'\n#### {machine}\n')
        print('| セル | 新条件（0.12.0） | 旧条件（0.12.0・旧腕） | 0.11.0 | 新条件の run 別比 |')
        print('|---|---|---|---|---|')
        for k in order:
            if k[0] != machine:
                continue
            n = new[k]
            mark = '' if float(n['ratio']) > 1.0 else ' ★'
            print(f"| {k[1]}{mark} | {fmt(n)} | {fmt(old[k])} | {fmt(prev[k])} | {n['runs']} |")
    print(f'\n- 新条件: `{t_new}`\n- 旧条件: `{t_old}`\n- 0.11.0: `{t_prev}`\n- ★ = 新条件の比が 1.0 以下')


if __name__ == '__main__':
    main()
