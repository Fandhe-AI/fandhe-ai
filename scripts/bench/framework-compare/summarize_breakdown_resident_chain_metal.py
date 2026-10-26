#!/usr/bin/env python3
"""イシュー #3612: Mac mini Metal 常駐チェーンの区間別内訳の集計（RULE.txt の hard 条件・突き合わせ・振り分け規則を fail-closed に適用する）。

入力: ランナー（run_breakdown_resident_chain_metal.sh）が出す `results-3612-<task>-<arm>-<label>.jsonl`（1 round 1 行・5 行）と
`load_gate-3612-<label>.log`（round ごとの gate=pass／fail）。A0（全体窓。#3257 と同じ bench-fandhe の step 中央値 µs・round 順）は
`--a0-train-us`／`--a0-infer-us`（カンマ区切り 5 値）で渡す。省略時は gap2・N は「未取得」で、結論は「判定不能」側へ倒す。
欠損・非数値・行数不足は例外（終了 2）。hard 条件違反・ゲート不通過は終了 1、成立は 0。分類であり判定ではなく、
tolerance・baseline・閾値は変えない。標準ライブラリのみ（python3 -I で実行）。
"""
import argparse
import json
import math
import os
import re
import statistics
import sys

ROUNDS = 5
TASKS = ("train", "infer")
SEGMENTS = {
    "train": ("prep_us", "encode_us", "gpu_busy_us", "host_fixed_us", "readout_us"),
    "infer": ("encode_us", "gpu_busy_us", "host_fixed_us", "readout_us"),
}
# 区間のタグ（RULE.txt 固定。事後に変えない）。
TAGS = {
    "host_fixed_us": "共通の計測窓",
    "readout_us": "共通の計測窓",
    "gpu_busy_us": "GPU 計算（ハード上限との比はピーク値の出典が無いため算出しない）",
    "prep_us": "自分たちの制約",
    "encode_us": "自分たちの制約",
}
# #3257 の参考系列（出典: docs/compat-resident-train-step-decision.md §17.3）。
REF_DELTA_US = {"train": 14.3, "infer": 6.9}
LABEL_RE = re.compile(r"^[A-Za-z0-9._-]+$")


class Invalid(Exception):
    pass


def num(v, what):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise Invalid(f"非数値・非有限: {what}={v!r}")
    return float(v)


def load_jsonl(path):
    if not os.path.isfile(path):
        raise Invalid(f"ファイルが無い: {path}")
    rows = []
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except ValueError as e:
                raise Invalid(f"{path}:{ln}: JSON 不正: {e}")
    if len(rows) != ROUNDS:
        raise Invalid(f"{path}: run 数が {ROUNDS} でない（{len(rows)}）")
    return rows


def parse_gate(path):
    if not os.path.isfile(path):
        raise Invalid(f"ゲートログが無い: {path}")
    passed = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = re.match(r"^round(\d+) .*gate=(pass|fail) ", line)
            if m and m.group(2) == "pass":
                passed.add(int(m.group(1)))
    return passed


def parse_list(s, what):
    if s is None:
        return None
    try:
        vals = [num(float(x), what) for x in s.split(",")]
    except ValueError as e:
        raise Invalid(f"{what}: 数値でない: {e}")
    if len(vals) != ROUNDS:
        raise Invalid(f"{what}: {ROUNDS} 値が必要（{len(vals)}）")
    return vals


def fmt_runs(vals):
    return ", ".join(f"r{i + 1}={v:.3f}" for i, v in enumerate(vals))


def analyse(dirpath, label, a0):
    violations = []
    out = []
    data = {}
    for task in TASKS:
        for arm in ("b0", "b1"):
            rows = load_jsonl(os.path.join(dirpath, f"results-3612-{task}-{arm}-{label}.jsonl"))
            for r in rows:
                if r.get("task") != task or r.get("arm") != arm:
                    raise Invalid(f"task/arm 不一致: {r.get('task')}/{r.get('arm')} (期待 {task}/{arm})")
                if num(r.get("wait_max"), "wait_max") != 1:
                    violations.append(f"{task}/{arm}: wait_max={r['wait_max']}（1 でない。フォールバックまたは追加同期）")
                if task == "infer" and arm == "b1" and r.get("bit_identical_to_b0") is not True:
                    violations.append("infer/b1: B0 と bit 一致でない")
                if "command_buffers_max" not in r:
                    violations.append(f"{task}/{arm}: command_buffers_max が欠落（RULE.txt hard 条件 3 の必須項目）")
                elif num(r["command_buffers_max"], "command_buffers_max") != 1:
                    out.append(f"注記: {task}/{arm} command_buffers_max={r['command_buffers_max']}（2 以上は auto-flush）")
            data[(task, arm)] = rows
    passed = parse_gate(os.path.join(dirpath, f"load_gate-3612-{label}.log"))
    if passed != set(range(1, ROUNDS + 1)):
        violations.append(f"負荷ゲート不通過の round あり（通過={sorted(passed)}）")

    for task in TASKS:
        b0, b1 = data[(task, "b0")], data[(task, "b1")]
        out.append(f"\n## {task}")
        out.append("| 区間 | 代表値 µs（run 別の中央値の中央値） | run 別 | min–max | タグ |")
        out.append("|---|---:|---|---|---|")
        seg_med = {}
        for seg in SEGMENTS[task]:
            vals = [num(r.get(seg), seg) for r in b1]
            seg_med[seg] = statistics.median(vals)
            out.append(f"| {seg} | {seg_med[seg]:.3f} | {fmt_runs(vals)} | {min(vals):.3f}–{max(vals):.3f} | {TAGS[seg]} |")
        tot1 = [num(r.get("total_us"), "total_us") for r in b1]
        tot0 = [num(r.get("total_us"), "total_us") for r in b0]
        t1, t0 = statistics.median(tot1), statistics.median(tot0)
        out.append(f"| total(B1) | {t1:.3f} | {fmt_runs(tot1)} | {min(tot1):.3f}–{max(tot1):.3f} | - |")
        out.append(f"| total(B0) | {t0:.3f} | {fmt_runs(tot0)} | {min(tot0):.3f}–{max(tot0):.3f} | - |")
        gap1 = t1 - sum(seg_med.values())
        out.append(f"\n- gap1（B1 total − Σ区間。計時のすき間）: {gap1:.3f} µs")
        out.append(f"- B1 − B0（明示同期・タイムスタンプ読み出しの負荷）: {t1 - t0:.3f} µs")
        a0v = a0.get(task)
        n_us = None
        if a0v is None:
            out.append("- gap2（B0 total − A0 窓）・N: 未取得（A0 未指定）")
        else:
            a0m = statistics.median(a0v)
            n_us = max(abs(v - a0m) for v in a0v)
            out.append(f"- A0 窓 中央値 {a0m:.3f} µs（{fmt_runs(a0v)}）")
            out.append(f"- gap2（B0 total − A0 窓。ハーネスの差）: {t0 - a0m:.3f} µs")
            out.append(f"- ノイズ幅 N（max|A0_r − median|）: {n_us:.3f} µs")
        delta = REF_DELTA_US[task]
        out.append(f"- Δ（#3257 参考系列 after − torch。同一セッションの torch は未取得）: +{delta} µs")
        if n_us is None:
            out.append("- 結論: 判定不能（N 未取得）。区間別の値のみ記録する。")
        elif abs(delta) <= n_us:
            out.append("- 結論: |Δ| ≤ N。揺れの範囲として記録（未説明残差は誤差範囲）。")
        else:
            out.append("- 結論: |Δ| > N。振り分け不能（相手の内訳が未測定）。自分側の量を Δ を説明しうる上限として並べる:")
            for seg in SEGMENTS[task]:
                out.append(f"  - {seg} = {seg_med[seg]:.3f} µs（{TAGS[seg]}）")
        out.append("- REQ-8 の境界検査の負荷: 未測定（分離できない）。制約の緩和は提案しない。")
    return violations, out


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--a0-train-us")
    ap.add_argument("--a0-infer-us")
    a = ap.parse_args(argv)
    if not LABEL_RE.match(a.label):
        print("label 不正", file=sys.stderr)
        return 2
    try:
        a0 = {"train": parse_list(a.a0_train_us, "a0-train"), "infer": parse_list(a.a0_infer_us, "a0-infer")}
        violations, out = analyse(a.dir, a.label, a0)
    except Invalid as e:
        print(f"INVALID: {e}", file=sys.stderr)
        return 2
    print(f"# #3612 Mac mini Metal 常駐チェーン内訳 集計（label={a.label}）")
    print("\n".join(out))
    if violations:
        print("\n## hard 条件違反（判定不能）")
        for v in violations:
            print(f"- {v}")
        return 1
    print("\nhard 条件: すべて通過")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
