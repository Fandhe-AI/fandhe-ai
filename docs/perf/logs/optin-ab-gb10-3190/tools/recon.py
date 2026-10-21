#!/usr/bin/env python3
"""stdout ログ (b3-ab.log / b6-ab.log) の `== ... arm=before|after ==` 直後の JSON 行から
before/after の JSONL を無加工で再構成する。使い方: recon.py <stdout.log> <before.jsonl> <after.jsonl>"""
import json
import re
import sys

src, outb, outa = sys.argv[1:4]
arm = None
rows = {"before": [], "after": []}
for line in open(src, encoding="utf-8"):
    m = re.match(r"^== .* arm=(before|after) ==\s*$", line)
    if m:
        arm = m.group(1)
        continue
    if arm and line.startswith("{"):
        json.loads(line)  # 壊れた行は例外で停止（fail-closed）
        rows[arm].append(line if line.endswith("\n") else line + "\n")
        arm = None
for name, path in (("before", outb), ("after", outa)):
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(rows[name])
print({k: len(v) for k, v in rows.items()})
