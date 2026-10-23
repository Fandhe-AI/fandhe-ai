#!/usr/bin/env python3
"""ツリー指紋: <root> 配下の全ファイルを `相対パス<TAB>sha256` の昇順一覧で <out> に書く（イシュー #3215 RULE 項 1）。

役割: ノードへ転送したツリーと、Mac 側の `git archive <SHA>` 展開ツリーが同一であることを突合するための一覧を作る。
除外（RULE 項 1 に固定）: 本成果物ディレクトリ・target/・Cargo.lock（cargo が書き換える）・.git・.rev-stamp。
呼び出し元: Mac 側（git archive 展開ツリー）とノード側（orchestrate_gb10.sh 冒頭）の双方から同じ引数で呼ぶ。
標準ライブラリのみ。シンボリックリンクは辿らず、リンク先パス文字列をハッシュ対象にする。
"""
import hashlib
import os
import sys

EXCLUDE_DIR_NAMES = {".git", "target"}
EXCLUDE_FILE_NAMES = {"Cargo.lock", ".rev-stamp"}
EXCLUDE_PREFIX = "docs/perf/logs/cuda-tma-stage2-ab-gb10-3215/"


def main(argv):
    if len(argv) != 3:
        print("usage: tree_fingerprint.py <root> <out>", file=sys.stderr)
        return 2
    root, out = argv[1], argv[2]
    rows = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = sorted(d for d in dirnames if d not in EXCLUDE_DIR_NAMES)
        for name in sorted(filenames):
            if name in EXCLUDE_FILE_NAMES:
                continue
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            if rel.startswith(EXCLUDE_PREFIX):
                continue
            h = hashlib.sha256()
            if os.path.islink(full):
                h.update(os.readlink(full).encode())
            else:
                with open(full, "rb") as f:
                    for chunk in iter(lambda: f.read(1 << 20), b""):
                        h.update(chunk)
            rows.append(f"{rel}\t{h.hexdigest()}")
    rows.sort()
    with open(out, "w") as f:
        f.write("\n".join(rows) + "\n")
    print(f"files={len(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
