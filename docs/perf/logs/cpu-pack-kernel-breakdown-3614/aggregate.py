#!/usr/bin/env python3
"""イシュー #3614 の fail-closed 集計。RULE.txt（計測前に固定）の指標・判定・停止条件を実装する。

役割: run_breakdown.sh が機体ごとに出す `<machine>/round{N}-{base,timed}-{native,t1}.log` を読み、
6 形状の A pack・B pack・kernel・その他の内訳、S_A の round 別値、PACK_MAIN／NOT_PACK_MAIN／
INDETERMINATE の判定を Markdown で標準出力へ出す。標準ライブラリのみ。

使い方: python3 -I aggregate.py <machine_dir> [<machine_dir> ...] > aggregate.md
停止条件（RULE §8）に当たると AggregateError を投げ、終了コード 1 で理由を標準エラーへ出す。
"""

import math
import os
import re
import statistics
import sys

ROUNDS = (1, 2, 3, 4, 5)
PROCS = (("base", "native"), ("base", "t1"), ("timed", "native"), ("timed", "t1"))
# 判定の閾値（RULE §10〜12。実装側が登録した値で、計測後に変えない）。
S_A_THRESHOLD = 0.50
PACK_MAIN_MIN_ROUNDS = 4
NOT_PACK_MAIN_MAX_ROUNDS = 1
RESOLUTION_TICKS = 10
SPLIT_RATIO = 1.5
SHAPES = ("l1_fwd", "l1_dw", "l1_dx", "l2_fwd", "l2_dw", "l2_dx")
REF_SHAPE = "ref_square_512"
ALL_SHAPES = SHAPES + (REF_SHAPE,)
# RULE §2 の形状定義: ラベル -> (trans, m, n, k)。ログ行の対応を検証する。
SHAPE_DEFS = {
    "l1_fwd": ("nn", 64, 256, 784),
    "l1_dw": ("tn", 784, 256, 64),
    "l1_dx": ("nt", 64, 784, 256),
    "l2_fwd": ("nn", 64, 10, 256),
    "l2_dw": ("tn", 256, 10, 64),
    "l2_dx": ("nt", 64, 256, 10),
    REF_SHAPE: ("nn", 512, 512, 512),
}
# A pack を取り除く (A) の対象（Nn／Nt）。決定記録 §4.1。
JUDGE_SHAPES = ("l1_fwd", "l1_dx", "l2_fwd", "l2_dx")
INTERVAL_KEYS = (
    ("a_pack", "a_pack_ns", "a_intervals"),
    ("b_pack", "b_pack_ns", "b_intervals"),
    ("kernel", "kernel_ns", "k_intervals"),
)

COMMON_KEYS = (
    "arm", "mode", "shape", "trans", "m", "n", "k", "rayon_threads", "calls",
    "wall_median_ns", "wall_mean_ns", "bits_hash",
)
TIMED_KEYS = (
    "a_pack_ns", "b_pack_ns", "kernel_ns", "job_ns", "jobs",
    "a_intervals", "b_intervals", "k_intervals",
)
INT_KEYS = (
    "m", "n", "k", "rayon_threads", "calls", "wall_median_ns",
) + TIMED_KEYS


FLOAT_KEYS = ("wall_mean_ns",)


class AggregateError(Exception):
    """RULE §8 の停止条件（fail-closed）。"""


def parse_line(text, where):
    """`key=value` を空白区切りで厳密に解析する。`=` を欠く語・重複キーは例外。"""
    out = {}
    for tok in text.split():
        if "=" not in tok:
            raise AggregateError(f"{where}: key=value 形式でない語 {tok!r}")
        key, val = tok.split("=", 1)
        if not key or key in out:
            raise AggregateError(f"{where}: キーが空または重複 {key!r}")
        out[key] = val
    return out


def parse_log(path, arm, mode):
    """1 プロセスのログを解析し `(calib, rows)` を返す。"""
    calib = None
    rows = {}
    with open(path, encoding="utf-8") as fh:
        for n, raw in enumerate(fh, 1):
            line = raw.strip()
            if not line:
                continue
            where = f"{path}:{n}"
            kv = parse_line(line, where)
            if "shape" not in kv:
                if "timer_tick_ns" not in kv or "timer_pair_ns" not in kv:
                    raise AggregateError(f"{where}: 校正行に timer_tick_ns/timer_pair_ns がない")
                if calib is not None:
                    raise AggregateError(f"{where}: 校正行が重複")
                try:
                    calib = {"tick": float(kv["timer_tick_ns"]), "pair": float(kv["timer_pair_ns"])}
                except ValueError as exc:
                    raise AggregateError(f"{where}: 校正値が数値でない") from exc
                for ck, cv in calib.items():
                    if not math.isfinite(cv) or cv <= 0:
                        raise AggregateError(f"{where}: 校正値 {ck} が有限の正値でない: {cv!r}")
                continue
            need = COMMON_KEYS + (TIMED_KEYS if arm == "timed" else ())
            for key in need:
                if key not in kv:
                    raise AggregateError(f"{where}: キー {key} がない")
            for key in INT_KEYS:
                if key in kv:
                    try:
                        kv[key] = int(kv[key])
                    except ValueError as exc:
                        raise AggregateError(f"{where}: {key} が整数でない: {kv[key]!r}") from exc
            for key in FLOAT_KEYS:
                try:
                    kv[key] = float(kv[key])
                except ValueError as exc:
                    raise AggregateError(f"{where}: {key} が数値でない: {kv[key]!r}") from exc
                if not math.isfinite(kv[key]) or kv[key] < 0:
                    raise AggregateError(f"{where}: {key} が有限の非負値でない: {kv[key]!r}")
            if not re.fullmatch(r"0x[0-9a-f]{16}", kv["bits_hash"]):
                raise AggregateError(f"{where}: bits_hash の形式が不正")
            if kv["arm"] != arm or kv["mode"] != mode:
                raise AggregateError(f"{where}: arm/mode がファイル名と食い違う ({kv['arm']}/{kv['mode']})")
            if kv["shape"] not in SHAPE_DEFS:
                raise AggregateError(f"{where}: 未知の shape {kv['shape']!r}")
            got = (kv["trans"], kv["m"], kv["n"], kv["k"])
            if got != SHAPE_DEFS[kv["shape"]]:
                raise AggregateError(
                    f"{where}: shape {kv['shape']} の trans/m/n/k {got} が RULE §2 の定義 {SHAPE_DEFS[kv['shape']]} と不一致")
            if kv["shape"] in rows:
                raise AggregateError(f"{where}: shape {kv['shape']} が重複")
            if kv["calls"] <= 0:
                raise AggregateError(f"{where}: calls が 0 以下")
            rows[kv["shape"]] = kv
    if calib is None:
        raise AggregateError(f"{path}: 校正行がない")
    for s in ALL_SHAPES:
        if s not in rows:
            raise AggregateError(f"{path}: shape {s} の行がない")
    return calib, rows


def load_machine(machine_dir):
    """全 (round × プロセス) を読み、RULE §8 の整合条件を検査して返す。"""
    data = {}
    for r in ROUNDS:
        for arm, mode in PROCS:
            path = os.path.join(machine_dir, f"round{r}-{arm}-{mode}.log")
            if os.path.exists(path + ".unavailable"):
                raise AggregateError(f"{path}: 負荷ゲート unavailable（参考系列。正式判定に使えない）")
            if not os.path.exists(path):
                raise AggregateError(f"{path}: 予定したログが欠けている")
            calib, rows = parse_log(path, arm, mode)
            for s, row in rows.items():
                if mode == "t1" and row["rayon_threads"] != 1:
                    raise AggregateError(f"{path}: t1 腕で rayon_threads={row['rayon_threads']}")
                if mode == "native" and row["rayon_threads"] == 1:
                    raise AggregateError(f"{path}: native 腕で rayon_threads=1")
                if arm == "timed" and row["jobs"] == 0:
                    raise AggregateError(f"{path}: shape {s} で jobs=0（経路が想定と違う）")
            data[(r, arm, mode)] = (calib, rows)
    for r in ROUNDS:
        for mode in ("native", "t1"):
            for s in ALL_SHAPES:
                hb = data[(r, "base", mode)][1][s]["bits_hash"]
                ht = data[(r, "timed", mode)][1][s]["bits_hash"]
                if hb != ht:
                    raise AggregateError(f"round{r} {mode} {s}: base と timed の bits_hash が不一致 ({hb} vs {ht})")
    for arm, mode in PROCS:
        for s in ALL_SHAPES:
            hashes = {data[(r, arm, mode)][1][s]["bits_hash"] for r in ROUNDS}
            if len(hashes) != 1:
                raise AggregateError(f"{arm}/{mode} {s}: round 間で bits_hash が不一致 {sorted(hashes)}")
    return data


def per_call(row, key):
    return row[key] / row["calls"]


def shape_metrics(row):
    """timed の 1 行から 1 回あたりの tA・tB・tK・tJob・tO を返す。"""
    ta = per_call(row, "a_pack_ns")
    tb = per_call(row, "b_pack_ns")
    tk = per_call(row, "kernel_ns")
    tj = per_call(row, "job_ns")
    return {"tA": ta, "tB": tb, "tK": tk, "tJ": tj, "tO": tj - ta - tb - tk}


def detect_split(values):
    """5 round の値が 1.5 倍以上離れた 2 群に割れていれば `(低群の中央値, 高群の中央値)`、なければ None。"""
    v = sorted(values)
    if v[0] <= 0 or v[-1] / v[0] < SPLIT_RATIO:
        return None
    gaps = [(v[i + 1] / v[i], i) for i in range(len(v) - 1)]
    ratio, idx = max(gaps)
    if ratio < SPLIT_RATIO:
        return None
    return statistics.median(v[: idx + 1]), statistics.median(v[idx + 1:])


def verdict_from(s_a):
    """S_A(r) の列から判定を返す（RULE §10）。"""
    med = statistics.median(s_a)
    hi = sum(1 for x in s_a if x >= S_A_THRESHOLD)
    if med >= S_A_THRESHOLD and hi >= PACK_MAIN_MIN_ROUNDS:
        return "PACK_MAIN"
    if med < S_A_THRESHOLD and hi <= NOT_PACK_MAIN_MAX_ROUNDS:
        return "NOT_PACK_MAIN"
    return "INDETERMINATE"


def analyze(data, mode="native"):
    """機体 1 つ・1 モードの指標と判定を計算する（timed の mode 腕の内訳）。"""
    per = {}  # shape -> round -> metrics
    for s in ALL_SHAPES:
        per[s] = {r: shape_metrics(data[(r, "timed", mode)][1][s]) for r in ROUNDS}
    s_a = []
    for r in ROUNDS:
        num = sum(per[s][r]["tA"] for s in JUDGE_SHAPES)
        den = sum(per[s][r]["tJ"] for s in JUDGE_SHAPES)
        s_a.append(num / den)
    spack_t = [
        sum(per[s][r]["tA"] + per[s][r]["tB"] for s in JUDGE_SHAPES)
        / sum(per[s][r]["tJ"] for s in JUDGE_SHAPES)
        for r in ROUNDS
    ]
    spack_all = [
        sum(per[s][r]["tA"] + per[s][r]["tB"] for s in SHAPES) / sum(per[s][r]["tJ"] for s in SHAPES)
        for r in ROUNDS
    ]
    ticks = [data[(r, "timed", mode)][0]["tick"] for r in ROUNDS]
    tick = statistics.median(ticks)
    res = {}
    for s in ALL_SHAPES:
        notes = []
        for name, nskey, ikey in INTERVAL_KEYS:
            rows = [data[(r, "timed", mode)][1][s] for r in ROUNDS]
            tot_ns = sum(x[nskey] for x in rows)
            tot_n = sum(x[ikey] for x in rows)
            avg = tot_ns / tot_n if tot_n else 0.0
            if avg < RESOLUTION_TICKS * tick:
                notes.append(f"{name} 区間 {avg:.0f} ns/回 < {RESOLUTION_TICKS}×tick({tick:.1f} ns)")
        res[s] = {"resolution_notes": notes}
    perturb = {
        s: statistics.median(
            data[(r, "timed", mode)][1][s]["wall_median_ns"] / data[(r, "base", mode)][1][s]["wall_median_ns"]
            for r in ROUNDS
        )
        for s in ALL_SHAPES
    }
    return {
        "per": per, "s_a": s_a, "verdict": verdict_from(s_a), "spack_t": spack_t,
        "spack_all": spack_all, "tick": tick, "res": res, "perturb": perturb,
    }


def med(per, s, key):
    return statistics.median(per[s][r][key] for r in ROUNDS)


def render_machine(name, data):
    out = [f"## {name}", ""]
    for mode in ("native", "t1"):
        a = analyze(data, mode)
        role = "判定に使う" if mode == "native" else "参照（判定に使わない）"
        rt = data[(1, "timed", mode)][1]["l1_fwd"]["rayon_threads"]
        out += [f"### {mode} 腕（{role}。rayon_threads={rt}）", ""]
        out += [
            f"timer_tick_ns 中央値 {a['tick']:.1f}。値は 1 回あたり ns の 5 round 中央値。",
            "",
            "| 形状 | tA | tB | tK | tO | tJob | sA | sB | sK | sO | s_pack | wall 摂動比 (timed/base) | 注記 |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for s in ALL_SHAPES:
            per = a["per"]
            tj = [per[s][r]["tJ"] for r in ROUNDS]
            share = lambda k: statistics.median(per[s][r][k] / per[s][r]["tJ"] for r in ROUNDS)
            spack = statistics.median((per[s][r]["tA"] + per[s][r]["tB"]) / per[s][r]["tJ"] for r in ROUNDS)
            notes = []
            if a["res"][s]["resolution_notes"]:
                notes.append("分解能不足: " + "; ".join(a["res"][s]["resolution_notes"]) + "。個別判定は判定不能（分解能）")
            sp = detect_split(tj)
            if sp:
                notes.append(f"2 水準: tJob 低群中央値 {sp[0]:.0f} ns・高群中央値 {sp[1]:.0f} ns")
            if s == REF_SHAPE:
                notes.append("参照（判定に使わない）")
            out.append(
                f"| {s} | {med(per, s, 'tA'):.0f} | {med(per, s, 'tB'):.0f} | {med(per, s, 'tK'):.0f} | "
                f"{med(per, s, 'tO'):.0f} | {med(per, s, 'tJ'):.0f} | {share('tA'):.3f} | {share('tB'):.3f} | "
                f"{share('tK'):.3f} | {share('tO'):.3f} | {spack:.3f} | {a['perturb'][s]:.3f} | {' / '.join(notes)} |"
            )
        out += ["", "run 番号付きの S_A（T = L1 fwd・L1 dX・L2 fwd・L2 dX の時間加重 tA/tJob）:", "",
                "| round | S_A | S_A >= 0.50 |", "|---|---|---|"]
        for r, v in zip(ROUNDS, a["s_a"]):
            out.append(f"| {r} | {v:.4f} | {'はい' if v >= S_A_THRESHOLD else 'いいえ'} |")
        out += [
            "",
            f"- S_A 中央値: {statistics.median(a['s_a']):.4f}（>= 0.50 の round: "
            f"{sum(1 for x in a['s_a'] if x >= S_A_THRESHOLD)}/5）",
            f"- 補助（判定に使わない）: T 全体の s_pack 中央値 {statistics.median(a['spack_t']):.4f}・"
            f"6 形状全体の s_pack 中央値 {statistics.median(a['spack_all']):.4f}",
        ]
        if mode == "native":
            out.append(f"- **判定: {a['verdict']}**")
        else:
            out.append(f"- 参照の判定式の当てはめ: {a['verdict']}（t1 は判定に使わない）")
        out.append("")
    return out


def main(argv):
    if len(argv) < 2:
        print("usage: aggregate.py <machine_dir> [...]", file=sys.stderr)
        return 2
    lines = ["# #3614 内訳計測の集計（aggregate.py の出力。手で直さない）", ""]
    try:
        for d in argv[1:]:
            name = os.path.basename(os.path.normpath(d))
            lines += render_machine(name, load_machine(d))
    except AggregateError as exc:
        print(f"fail-closed: {exc}", file=sys.stderr)
        return 1
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
