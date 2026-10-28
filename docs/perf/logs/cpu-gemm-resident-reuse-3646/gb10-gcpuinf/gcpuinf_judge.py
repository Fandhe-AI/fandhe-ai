#!/usr/bin/env python3
"""イシュー #3646: RULE.txt 末尾「G-CPU-INF の分類規則」の算出器（標準ライブラリのみ。fail-closed）。

役割: GB10 同一セッションの PyTorch 窓（bench_py の infer）・fandhe 窓（bench-fandhe の infer fresh／reuse）・
PyTorch 純カーネル（opp_kernel_torch_diag の l1_fwd／l2_fwd mm_pure）・fandhe 層別区間（#2105 診断の
fixedcost-2105/run*.jsonl）から D と成分を求め、bridge の閾値 50% に対する比を出す。分類（3 区分／保留）は
この出力と BLAS 経路の実行時確認（blas_path.log・blas_bindings.txt）を突き合わせて人が README／doc §9.4 に書く。
呼び出し元: 人手（`python3 gcpuinf_judge.py <gb10-gcpuinf ディレクトリ>`。結果は gcpuinf_judge.md）。
欠落・非有限値・run 不足・重複はすべて例外（exit 1）。
"""
import json
import math
import os
import statistics
import sys

RUNS = (1, 2, 3, 4, 5)


class JErr(Exception):
    pass


def rows(path):
    out = []
    try:
        with open(path, encoding="utf-8") as f:
            for i, ln in enumerate(f, 1):
                ln = ln.strip()
                if ln:
                    try:
                        out.append(json.loads(ln))
                    except ValueError as e:
                        raise JErr(f"{path}:{i}: {e}")
    except OSError as e:
        raise JErr(f"{path}: {e}")
    return out


def pos(x, w):
    v = float(x)
    if not math.isfinite(v) or v <= 0:
        raise JErr(f"{w}: 有限の正値でない {v}")
    return v


def single(d, name, **want):
    rs = rows(os.path.join(d, name))
    if len(rs) != 1:
        raise JErr(f"{name}: 行数 {len(rs)} != 1")
    for k, v in want.items():
        if rs[0].get(k) != v:
            raise JErr(f"{name}: {k}={rs[0].get(k)!r} != {v!r}")
    return pos(rs[0]["median_s"], name)


def check_load_gate(d):
    """負荷ゲートログ（RULE.txt [ゲート]）を検査する。未通過・通過行数の不一致は例外（参考扱い＝判定しない）。"""
    path = os.path.join(d, "..", "session-scripts", "load_gate_bench.log")
    try:
        with open(path, encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip().startswith("gcpuinf/")]
    except OSError as e:
        raise JErr(f"{path}: {e}")
    if any(ln.endswith("gate=unpassed") for ln in lines):
        raise JErr(f"{path}: 負荷ゲート未通過の系列は参考扱いで判定しない")
    got = sum(1 for ln in lines if ln.endswith("gate=pass"))
    want = 3 * len(RUNS)
    if got != want:
        raise JErr(f"{path}: gcpuinf の gate=pass 行数 {got} != {want}")


def unique_by(rs, key, name):
    out = {}
    for x in rs:
        k = x[key]
        if k in out:
            raise JErr(f"{name}: 重複 {key}={k!r}")
        out[k] = x
    return out


def med(vals):
    return statistics.median(vals)


def main(argv):
    if len(argv) != 2:
        print("usage: gcpuinf_judge.py <gb10-gcpuinf dir>", file=sys.stderr)
        return 2
    d = argv[1]
    check_load_gate(d)
    torch_win = [single(d, f"torch_infer_r{r}.jsonl", task="infer", framework="pytorch") for r in RUNS]
    f_fresh = [single(d, f"fandhe_infer_fresh_r{r}.jsonl", task="infer", mode="fresh") for r in RUNS]
    f_reuse = [single(d, f"fandhe_infer_reuse_r{r}.jsonl", task="infer", mode="reuse") for r in RUNS]
    f_phase = []
    for r in RUNS:
        got = unique_by(rows(os.path.join(d, f"fandhe_infer_phases_r{r}.jsonl")), "phase", f"fandhe_infer_phases_r{r}")
        if "iter_total" not in got:
            raise JErr(f"fandhe_infer_phases_r{r}: iter_total が無い")
        f_phase.append(pos(got["iter_total"]["median_s"], f"phases r{r}"))
    kern = {}
    for r in RUNS:
        seen = set()
        for x in rows(os.path.join(d, f"torch_kernel_r{r}.jsonl")):
            if x["phase"] == "mm_pure" and x["shape_id"] in ("l1_fwd", "l2_fwd"):
                if x["shape_id"] in seen:
                    raise JErr(f"torch_kernel_r{r}: 重複 {x['shape_id']}")
                seen.add(x["shape_id"])
                kern.setdefault(x["shape_id"], []).append(pos(x["median_s"], f"torch_kernel_r{r}"))
        if seen != {"l1_fwd", "l2_fwd"}:
            raise JErr(f"torch_kernel_r{r}: l1_fwd/l2_fwd が揃わない")
    fc = {}
    for r in RUNS:
        for x in rows(os.path.join(d, "fixedcost-2105", f"run{r}.jsonl")):
            ph, arm = x.get("phase"), x.get("arm")
            if (arm, ph) in (("reuse_decomposed", "l1_linear_forward_device"), ("reuse_decomposed", "l2_linear_forward_device"),
                             ("fresh_layers", "l1_linear_relu_fused"), ("fresh_layers", "l2_linear_gemm_add")):
                cell = fc.setdefault((arm, ph), {})
                if r in cell:
                    raise JErr(f"2105 run{r}: 重複 {(arm, ph)}")
                cell[r] = pos(x["median_s"], f"2105 run{r}")
    keys = [("reuse_decomposed", "l1_linear_forward_device"), ("reuse_decomposed", "l2_linear_forward_device"),
            ("fresh_layers", "l1_linear_relu_fused"), ("fresh_layers", "l2_linear_gemm_add")]
    for k in keys:
        if sorted(fc.get(k, {})) != list(RUNS):
            raise JErr(f"2105 {k}: run が揃わない")
    us = 1e6
    t_win = med(torch_win)
    tk = med(kern["l1_fwd"]) + med(kern["l2_fwd"])
    lines = ["# G-CPU-INF の算出（RULE.txt 末尾。#3646）", "",
             "単位 us。5 run の run 間中央値。D = fandhe 窓 - PyTorch 窓。",
             "純カーネル差 = fandhe 層カーネル合計 - PyTorch mm 合計（l1_fwd 64x256x784 + l2_fwd 64x10x256）。",
             "窓の構成差 = D - 純カーネル差（checksum・readout・fandhe 固有の固定費・bias／活性化など、窓に入るが mm 以外の部分の残差）。",
             "", f"- PyTorch 窓（bench_py infer）: {t_win*us:.1f}（run 別 " + ", ".join(f"{v*us:.1f}" for v in torch_win) + "）",
             f"- PyTorch mm_pure l1_fwd {med(kern['l1_fwd'])*us:.1f} + l2_fwd {med(kern['l2_fwd'])*us:.1f} = {tk*us:.1f}",
             "", "| fandhe の窓 | 窓 | run 別 | D | fandhe 層カーネル合計 | 純カーネル差 | 窓の構成差 | 純カーネル差/D | 構成差/D |",
             "|---|---|---|---|---|---|---|---|---|"]
    wins = [("fresh 通常行", f_fresh, ("fresh_layers", "l1_linear_relu_fused"), ("fresh_layers", "l2_linear_gemm_add")),
            ("reuse 通常行", f_reuse, ("reuse_decomposed", "l1_linear_forward_device"), ("reuse_decomposed", "l2_linear_forward_device")),
            ("reuse --phases iter_total", f_phase, ("reuse_decomposed", "l1_linear_forward_device"), ("reuse_decomposed", "l2_linear_forward_device"))]
    for name, vals, k1, k2 in wins:
        w = med(vals)
        D = w - t_win
        fk = med(list(fc[k1].values())) + med(list(fc[k2].values()))
        kd = fk - tk
        cd = D - kd
        lines.append(f"| {name} | {w*us:.1f} | " + ", ".join(f"{v*us:.1f}" for v in vals) +
                     f" | {D*us:+.1f} | {fk*us:.1f} | {kd*us:+.1f} | {cd*us:+.1f} | "
                     f"{(kd/D if D > 0 else float('nan')):.2f} | {(cd/D if D > 0 else float('nan')):.2f} |")
    lines += ["", "注: fandhe 層カーネル合計は #2105 診断（fixedcost-2105）の層別区間で、fresh 行には fresh_layers（L1 は linear+relu 融合・L2 は gemm+add）、",
              "reuse 行には reuse_decomposed の l1／l2 forward_device を当てた。層別区間は診断の別プロセスの値で窓の値とは同一プロセスではない。"]
    with open(os.path.join(d, "gcpuinf_judge.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except JErr as e:
        print(f"gcpuinf_judge: {e}", file=sys.stderr)
        sys.exit(1)
