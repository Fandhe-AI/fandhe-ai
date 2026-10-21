#!/usr/bin/env python3
"""継承ゲートログから round 別 machine_gate_ok を判定する（イシュー #3184。補遺 RULE.txt 3）。

役割: 継承スクリプト（2104／2102／2113／2112／§15.1／2111／2114）が書く負荷ゲートログの
`round<N> ... load1=<x>`／`run<N> load1=<x>` 行から、round ごとの load1 を読み、全 5 round が
< 3.0 のとき machine_gate_ok=1 を出力する。欠損・非数値・round 数 != 5 は fail-closed（0）。
継承ログは編集しない。stdlib のみ。`python3 -I machine_gate_check.py <gate_log>...` で実行。
各 round の最後の記録を採用する（同一 round が複数行ある場合は後勝ち）。
round 番号を持たない `uptime` 形式のログ（30 秒間隔の定期サンプル）は round との対応を確認できないため
machine_gate_ok=undetermined（判定不能。1 と見なさない）とし、サンプルは参考値として列挙するだけにする。
出力: `<file>: round1=0|1(load1=..) ... machine_gate_ok=0|1`。終了コード 0 / 2（入力不正）。
"""
import math
import re
import sys

THRESHOLD = 3.0
LINE = re.compile(r"\b(?:round|run)\s*(\d+)\b.*?\bload1=([^\s,]+)")
# `uptime` 形式（`load averages: 5.82 5.68 5.60` / 区切りがカンマの場合も許容）。round 番号を持たないログ用
UPTIME = re.compile(r"load averages?:\s*([0-9.]+|nan|inf)")


def check_text(text):
    """text から (per_round dict, ok bool) を返す。per_round[n] = (load1|None, 0|1)。"""
    seen = {}
    for line in text.splitlines():
        m = LINE.search(line)
        if m:
            seen[int(m.group(1))] = m.group(2)
    if not seen:
        # round 番号付き行が無い: uptime は定期サンプルで round と対応しないため判定不能（None）を返す
        readings = UPTIME.findall(text)
        if readings:
            return {i + 1: (None, None) for i in range(len(readings))}, None
    per_round = {}
    for n in range(1, 6):
        raw = seen.get(n)
        try:
            v = float(raw)
            good = math.isfinite(v) and v < THRESHOLD
        except (TypeError, ValueError):
            v, good = None, False
        per_round[n] = (v, 1 if good else 0)
    ok = len(seen) == 5 and all(n in seen for n in range(1, 6)) and all(g for _, g in per_round.values())
    return per_round, ok


def render(name, per_round, ok):
    if ok is None:
        n = len(per_round)
        return f"{name}: round 対応なし（uptime 定期サンプル {n} 件）machine_gate_ok=undetermined"
    cells = " ".join(f"round{n}={g}(load1={v})" for n, (v, g) in per_round.items())
    return f"{name}: {cells} machine_gate_ok={1 if ok else 0}"


def self_test():
    good = "\n".join(f"round{n} gate=ok threshold=8.0 load1=1.{n}" for n in range(1, 6))
    assert check_text(good)[1] is True
    high = good.replace("load1=1.3", "load1=3.0")
    assert check_text(high)[1] is False
    assert check_text("\n".join(good.splitlines()[:4]))[1] is False
    assert check_text(good.replace("load1=1.2", "load1=nan"))[1] is False
    assert check_text(good.replace("load1=1.2", "load1="))[1] is False
    assert check_text("run1 load1=0.5 (record_only)\nrun2 load1=0.5\nrun3 load1=0.5\nrun4 load1=0.5\nrun5 load1=0.5")[1]
    assert check_text("")[1] is False
    up = "\n".join(f"17:2{n}  up 1 day, load averages: 1.{n} 2.0 2.0" for n in range(1, 6))
    assert check_text(up)[1] is None  # 定期サンプルは round と対応しないので 5 件でも成立させない
    assert check_text(up.replace("1.3", "3.4"))[1] is None
    assert check_text(up + "\n17:30  up, load averages: 1.0 1 1")[1] is None
    assert render("x", *check_text(up)).endswith("machine_gate_ok=undetermined")
    print("self-test OK")


def main(argv):
    if len(argv) == 2 and argv[1] == "--self-test":
        self_test()
        return 0
    if len(argv) < 2:
        print("usage: machine_gate_check.py <gate_log>... | --self-test", file=sys.stderr)
        return 2
    for path in argv[1:]:
        try:
            with open(path, encoding="utf-8") as f:
                per_round, ok = check_text(f.read())
        except OSError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        print(render(path.rsplit("/", 1)[-1], per_round, ok))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
