#!/usr/bin/env python3
"""#4151 の round{i}/raw/ から values.tsv（quantity／value の 2 列）を作る厳格変換（README 手順 4 の変換手順）。

役割: run_round.sh が残した raw を、aggregate_4151.py が読む values.tsv に写す。値の計算は単位換算と
SME probe の最小二乗（切片）だけで、推定値で埋めない。欠損・重複・非有限・0／負・status≠ok はすべて例外で止まる。
  W_off／Wf_off   raw/rust.jsonl（off-base 向け bench-fandhe）の mode=reuse／fresh の median_s × 1000（ms）
  W_on／Wf_on     raw/rust-on.jsonl（on-base 向け bench-fandhe）の mode=reuse／fresh の median_s × 1000（ms）
  O               raw/py.jsonl（bench_py.py pytorch gemm 256）の mode=fresh の median_s × 1000（ms）
  K_opp           raw/measure_gemm256.jsonl（torch_cpu_mlp_diag.py）の item=mm_pure の median_s × 1000（ms）
  c_sme_kc0_ns    raw/sme-probe.txt の kernel=SME kc=0 の median_ns_per_call（ns。直接計時の主値）
  c_sme_intercept_ns  同 kernel=SME kc>=1（1,2,4,…,256）の median_ns_per_call を kc に対して最小二乗した切片（ns。
                  #3175 の aggregate.py lsq_line と同じ無重みの最小二乗。probe の docstring の「kc>=1 の最小二乗を補助値」）
  probe が無い・レコード未出力の round は c_sme_* を書かない（RULE 項 4: unavailable は values.tsv に書かない）。
使い方: python3 -I to_values_4151.py <round{i} ディレクトリ> [...]
標準ライブラリのみ。gate.txt が PASS 以外の round は変換しない。既存の values.tsv は上書きしない。
"""
import json
import math
import os
import re
import sys

PROBE_RE = re.compile(r"probe=microcall kernel=(SME|NEON) kc=(\d+) reps=(\d+) median_ns_per_call=([0-9.eE+-]+)$")
KCS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256]


class ConvError(Exception):
    pass


def pos(v, what):
    if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) or v <= 0:
        raise ConvError(f"{what}: 有限の正数でない（{v!r}）")
    return float(v)


def jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ConvError(f"{os.path.basename(path)}:{ln}: JSON 解釈不能") from e
    return rows


def bench_modes(path, expect_modes, framework):
    rows = jsonl(path)
    got = {}
    for r in rows:
        if r.get("framework") != framework or r.get("task") != "gemm" or r.get("device") != "cpu" or r.get("size") != 256:
            raise ConvError(f"{os.path.basename(path)}: 想定外の行（framework/task/device/size）")
        m = r.get("mode")
        if m in got:
            raise ConvError(f"{os.path.basename(path)}: mode={m} が重複")
        if r.get("parity_fail_count") != 0:
            raise ConvError(f"{os.path.basename(path)}: parity_fail_count が 0 でない")
        got[m] = pos(r.get("median_s"), f"{os.path.basename(path)} mode={m} median_s") * 1000.0
    if sorted(got) != sorted(expect_modes):
        raise ConvError(f"{os.path.basename(path)}: mode の集合が {sorted(got)}（期待 {sorted(expect_modes)}）")
    return got


def k_opp(path):
    hit = [r for r in jsonl(path) if r.get("item") == "mm_pure"]
    if len(hit) != 1:
        raise ConvError(f"measure_gemm256.jsonl: item=mm_pure が {len(hit)} 行")
    r = hit[0]
    if r.get("status") != "ok" or r.get("section") != "gemm256":
        raise ConvError("measure_gemm256.jsonl: mm_pure の status／section が想定外")
    return pos(r.get("median_s"), "mm_pure median_s") * 1000.0


def sme_probe(path):
    if not os.path.exists(path):
        return None
    vals = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            i = line.find("probe=microcall")
            if i < 0:
                continue
            m = PROBE_RE.match(line[i:].rstrip("\n"))
            if not m:
                raise ConvError("sme-probe.txt: probe 行の形式が想定外")
            if m.group(1) != "SME":
                continue
            kc = int(m.group(2))
            if kc in vals:
                raise ConvError(f"sme-probe.txt: SME kc={kc} が重複")
            vals[kc] = pos(float(m.group(4)), f"SME kc={kc}")
    if not vals:
        return None
    if sorted(vals) != KCS:
        raise ConvError(f"sme-probe.txt: SME の kc 集合が {sorted(vals)}")
    xs = [float(k) for k in KCS if k >= 1]
    ys = [vals[k] for k in KCS if k >= 1]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    icpt = my - slope * mx
    return vals[0], pos(icpt, "SME 切片")


def convert(rd):
    gate = open(os.path.join(rd, "gate.txt"), encoding="utf-8").read().split()
    if not gate:
        raise ConvError("gate.txt が空")
    if gate[0] != "PASS":
        return f"skip（gate={gate[0]}）"
    out = os.path.join(rd, "values.tsv")
    if os.path.exists(out):
        raise ConvError("values.tsv が既にある（上書きしない）")
    raw = os.path.join(rd, "raw")
    off = bench_modes(os.path.join(raw, "rust.jsonl"), ["fresh", "reuse"], "fandhe-ai")
    on = bench_modes(os.path.join(raw, "rust-on.jsonl"), ["fresh", "reuse"], "fandhe-ai")
    py = bench_modes(os.path.join(raw, "py.jsonl"), ["fresh"], "pytorch")
    q = [("W_off", off["reuse"]), ("W_on", on["reuse"]), ("Wf_off", off["fresh"]), ("Wf_on", on["fresh"]),
         ("O", py["fresh"]), ("K_opp", k_opp(os.path.join(raw, "measure_gemm256.jsonl")))]
    sme = sme_probe(os.path.join(raw, "sme-probe.txt"))
    if sme is not None:
        q += [("c_sme_kc0_ns", sme[0]), ("c_sme_intercept_ns", sme[1])]
    with open(out, "x", encoding="utf-8") as f:
        f.write("quantity\tvalue\n")
        for k, v in q:
            f.write(f"{k}\t{v:.9g}\n")
    return "values.tsv written" + ("" if sme is not None else "（c_sme は unavailable）")


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    for rd in argv:
        try:
            print(f"{os.path.basename(rd.rstrip('/'))}: {convert(rd)}")
        except (ConvError, OSError) as e:
            print(f"CONVERT-ERROR {os.path.basename(rd.rstrip('/'))}: {e}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
