#!/usr/bin/env python3
"""全 run の同一性検査（0.13.0 計測キット。0.12.0 版 check_identity.py の単腕版）。計測後に 1 回実行し、1 件でも違えば非零で終わる。

0.12.0 版からの変更点: 旧条件腕を置かない（bench-fandhe は 0.12.0 から変えずピンだけ上げる単腕）ため、
tree-old.txt・rev-stamp-old.txt の検査を外し、SHA は 1 つ（RULE.txt の期待 SHA）、registry ピンは v0.13.0 にした。
検査: 各 run ディレクトリの tree.txt が `fandhe-ai v0.13.0` で終わる行を持つこと、rev-stamp.txt が RULE の SHA と一致すること、
結果 JSONL の行数が期待値と一致すること、異なる run が 5 つあること。欠けているファイル・run の不足は例外扱い（黙って少ない run で通さない）。
使い方: check_identity.py SHA ROWS_SPEC RUN_DIR [RUN_DIR ...]
  ROWS_SPEC は `ファイル名=行数` をカンマ区切り
  （GB10 例: results-dgx-0.13.0.jsonl=112,results-dgx-0.13.0-extra.jsonl=8,results-dgx-py-0.13.0.jsonl=28
    Mac mini 例: results-macmini-0.13.0.jsonl=48,results-macmini-py-0.13.0.jsonl=32）
  --self-test で自己検査。
"""
import os, re, sys, tempfile


def check(sha, spec, runs, expect_runs=5):
    errs = []
    # realpath で正規化し、同じ run ディレクトリの重複指定で件数を満たすことを拒否する
    real = [os.path.realpath(d) for d in runs]
    if len(set(real)) != len(real):
        errs.append('run ディレクトリが重複している')
    if len(set(real)) != expect_runs:
        errs.append(f'異なる run 数 {len(set(real))}（期待 {expect_runs}）')
    for d in runs:
        p = os.path.join(d, 'rev-stamp.txt')
        if not os.path.isfile(p):
            errs.append(f'{p} なし')
        else:
            with open(p) as f:
                if f.read().strip() != sha:
                    errs.append(f'{p} が RULE と不一致')
        p = os.path.join(d, 'tree.txt')
        if not os.path.isfile(p):
            errs.append(f'{p} なし')
        else:
            with open(p) as f:
                if not re.search(r'fandhe-ai v0\.13\.0$', f.read(), re.M):
                    errs.append(f'{p} が fandhe-ai v0.13.0 でない')
        for fn, n in spec.items():
            p = os.path.join(d, fn)
            if not os.path.isfile(p):
                errs.append(f'{p} なし'); continue
            with open(p) as f:
                got = sum(1 for l in f if l.strip())
            if got != n:
                errs.append(f'{p} 行数 {got}（期待 {n}）')
    return errs


def self_test():
    with tempfile.TemporaryDirectory() as t:
        runs = []
        for i in range(5):
            d = os.path.join(t, f'run{i}'); os.makedirs(d); runs.append(d)
            open(os.path.join(d, 'rev-stamp.txt'), 'w').write('a' * 40 + '\n')
            open(os.path.join(d, 'tree.txt'), 'w').write('├── fandhe-ai v0.13.0\n')
            open(os.path.join(d, 'r.jsonl'), 'w').write('{}\n' * 3)
        assert check('a' * 40, {'r.jsonl': 3}, runs) == []
        assert check('c' * 40, {'r.jsonl': 3}, runs)          # SHA 不一致
        assert check('a' * 40, {'r.jsonl': 4}, runs)          # 行数不一致
        assert check('a' * 40, {'r.jsonl': 3}, runs[:4])      # run 不足
        assert check('a' * 40, {'r.jsonl': 3}, [runs[0]] * 5)  # 重複指定
        open(os.path.join(runs[1], 'tree.txt'), 'w').write('├── fandhe-ai v0.12.0\n')
        assert check('a' * 40, {'r.jsonl': 3}, runs)          # 旧ピン
        open(os.path.join(runs[1], 'tree.txt'), 'w').write('├── fandhe-ai v0.13.0\n')
        os.remove(os.path.join(runs[0], 'tree.txt'))
        assert check('a' * 40, {'r.jsonl': 3}, runs)          # ファイル欠損
    print('self-test ok')


if __name__ == '__main__':
    if sys.argv[1:] == ['--self-test']:
        self_test(); sys.exit(0)
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    spec = {kv.split('=')[0]: int(kv.split('=')[1]) for kv in sys.argv[2].split(',')}
    e = check(sys.argv[1], spec, sys.argv[3:])
    for x in e:
        print('NG:', x)
    sys.exit(1 if e else 0)
