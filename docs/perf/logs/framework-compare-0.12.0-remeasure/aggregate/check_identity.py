#!/usr/bin/env python3
"""全 run の同一性検査（イシュー #3273 の受け入れ条件 A1）。計測後に 1 回実行し、1 件でも違えば非零で終わる。

検査: 各 run ディレクトリの tree.txt・tree-old.txt が `fandhe-ai v0.12.0` で終わる行を持つこと、
rev-stamp.txt が RULE の新腕 SHA、rev-stamp-old.txt が旧腕 SHA と一致すること、結果 JSONL の行数が期待値と一致すること。
欠けているファイル・run の不足は例外（黙って少ない run で通さない）。
使い方: check_identity.py NEW_SHA OLD_SHA ROWS_SPEC RUN_DIR [RUN_DIR ...]
  ROWS_SPEC は `glob接頭辞=行数` をカンマ区切り（例: results-macmini-0.12.0.jsonl=48,results-macmini-old-0.12.0.jsonl=26）
  --self-test で自己検査。
"""
import os, re, sys, tempfile


def check(new_sha, old_sha, spec, runs, expect_runs=5):
    errs = []
    if len(runs) != expect_runs:
        errs.append(f'run 数 {len(runs)}（期待 {expect_runs}）')
    for d in runs:
        for name, sha in (('rev-stamp.txt', new_sha), ('rev-stamp-old.txt', old_sha)):
            p = os.path.join(d, name)
            if not os.path.isfile(p):
                errs.append(f'{p} なし'); continue
            with open(p) as f:
                got_sha = f.read().strip()
            if got_sha != sha:
                errs.append(f'{p} が RULE と不一致')
        for name in ('tree.txt', 'tree-old.txt'):
            p = os.path.join(d, name)
            if not os.path.isfile(p):
                errs.append(f'{p} なし'); continue
            with open(p) as f:
                tree_text = f.read()
            if not re.search(r'fandhe-ai v0\.12\.0$', tree_text, re.M):
                errs.append(f'{p} が fandhe-ai v0.12.0 でない')
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
            open(os.path.join(d, 'rev-stamp-old.txt'), 'w').write('b' * 40 + '\n')
            for n in ('tree.txt', 'tree-old.txt'):
                open(os.path.join(d, n), 'w').write('├── fandhe-ai v0.12.0\n')
            open(os.path.join(d, 'r.jsonl'), 'w').write('{}\n' * 3)
        assert check('a' * 40, 'b' * 40, {'r.jsonl': 3}, runs) == []
        assert check('c' * 40, 'b' * 40, {'r.jsonl': 3}, runs)
        assert check('a' * 40, 'b' * 40, {'r.jsonl': 4}, runs)
        assert check('a' * 40, 'b' * 40, {'r.jsonl': 3}, runs[:4])
        os.remove(os.path.join(runs[0], 'tree-old.txt'))
        assert check('a' * 40, 'b' * 40, {'r.jsonl': 3}, runs)
    print('self-test ok')


if __name__ == '__main__':
    if sys.argv[1:] == ['--self-test']:
        self_test(); sys.exit(0)
    if len(sys.argv) < 5:
        sys.exit(__doc__)
    spec = {kv.split('=')[0]: int(kv.split('=')[1]) for kv in sys.argv[3].split(',')}
    e = check(sys.argv[1], sys.argv[2], spec, sys.argv[4:])
    for x in e:
        print('NG:', x)
    sys.exit(1 if e else 0)
