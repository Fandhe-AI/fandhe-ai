#!/usr/bin/env python3
"""旧条件腕の判定用: 新腕 JSONL の bench-fandhe 行（framework=fandhe-ai）を、同じ run の旧腕 JSONL の行で置き換えた JSONL を作る。

呼び出し元: median_rounds.py の前段（run ごとに 1 回）。判定コード（gen 系）は 1 か所のまま、入力だけを差し替える（イシュー #3273）。
`median_rounds.py` の index は key で重複を除く（重複は例外）ため、追加ではなく置き換えにする。
旧腕に無いセルの新腕行は残す（旧腕が測らない行を欠かさない）。旧腕のキーが新腕に無ければ例外（取り違えの検知）。
使い方: swap_fandhe_arm.py NEW.jsonl OLD.jsonl OUT.jsonl  /  swap_fandhe_arm.py --self-test
"""
import json, sys, os, tempfile


def key(r):
    return (r['framework'], r['task'], r['device'], int(r['size']), r['mode'], bool(r.get('tf32')))


def load(p):
    rows = []
    for l in open(p):
        if l.strip():
            rows.append(json.loads(l))
    return rows


def swap(new_rows, old_rows):
    old = {}
    for r in old_rows:
        if r['framework'] != 'fandhe-ai':
            raise SystemExit(f"旧腕に fandhe-ai 以外の行: {r['framework']}")
        k = key(r)
        if k in old:
            raise SystemExit(f'旧腕に重複セル {k}')
        old[k] = r
    present = {key(r) for r in new_rows if r['framework'] == 'fandhe-ai'}
    missing = set(old) - present
    if missing:
        raise SystemExit(f'旧腕のセルが新腕に無い: {sorted(missing)}')
    return [old.get(key(r), r) if r['framework'] == 'fandhe-ai' else r for r in new_rows]


def self_test():
    a = {'framework': 'fandhe-ai', 'task': 'gemm', 'device': 'cpu', 'size': 256, 'mode': 'fresh', 'median_s': 1.0}
    b = dict(a, median_s=2.0)
    c = {'framework': 'candle', 'task': 'gemm', 'device': 'cpu', 'size': 256, 'mode': 'fresh', 'median_s': 3.0}
    assert swap([a, c], [b]) == [b, c]
    assert swap([a, c], [a]) == [a, c]  # 同じ行を渡せば不変
    for bad in ([dict(b, size=512)], [c]):
        try:
            swap([a, c], bad)
        except SystemExit:
            continue
        raise AssertionError('欠損・取り違えが例外にならない')
    print('self-test ok')


if __name__ == '__main__':
    if sys.argv[1:] == ['--self-test']:
        self_test(); sys.exit(0)
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    out = swap(load(sys.argv[1]), load(sys.argv[2]))
    with open(sys.argv[3], 'w') as w:
        for r in out:
            w.write(json.dumps(r) + '\n')
