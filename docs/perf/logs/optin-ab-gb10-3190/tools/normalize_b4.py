#!/usr/bin/env python3
"""B4 参考集計用の正規化コピー作成。唯一の変換: Layer B ログの
`test ... DIAG2109 ...`（--nocapture の test 行末に最初の DIAG 行が融合）を改行で分離する。
使い方: normalize_b4.py <src_dir> <dst_dir>（dst は空であること）"""
import os
import re
import shutil
import sys

src, dst = sys.argv[1:3]
if os.path.exists(dst) and os.listdir(dst):
    sys.exit("dst must be empty")
os.makedirs(dst, exist_ok=True)
n = 0
for f in sorted(os.listdir(src)):
    s = os.path.join(src, f)
    d = os.path.join(dst, f)
    if f.startswith("layerB-run"):
        t = open(s, encoding="utf-8").read()
        t2 = re.sub(r"^(test [^\n]*? \.\.\. )(DIAG2109 )", r"\1\n\2", t, flags=re.M)
        if t2 != t:
            n += 1
        open(d, "w", encoding="utf-8").write(t2)
    else:
        shutil.copy(s, d)
print("normalized layerB logs:", n)
