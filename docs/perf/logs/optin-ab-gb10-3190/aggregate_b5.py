#!/usr/bin/env python3
"""イシュー #3190 B5（境界述語の事前計算。#537）の判定集計。python3 標準ライブラリのみ。

判定規則の正は `RULE.txt` 8 と `RULE-reopen.txt`（B5 節）。本ファイルは規則の機械実装であり、
規則を緩める判定を足さない。`python3 -I aggregate_b5.py <b5 ログディレクトリ>` で実行し、
`--self-test` は固定 fixture による自己検証（実機不要）。

入力（`<dir>/`）:
  - parity-base.log / parity-head.log … 前提ゲート (a)。`cargo test ... -- --ignored --test-threads=1`
    の出力（`--nocapture` を付けない。付けると `test X ... ok` が出力に融合して読めない）。
  - ptxas-base.log / ptxas-head.log … 前提ゲート (b)。`ptxas -arch=sm_121 -v` の出力
  - run{1..5}-{base,head}.log … `gemm_mma_bench` の標準出力（`size=N ... mma_f16_tflops=X ...` 行）
  - gate.log … 専有ゲート行 `run<i>-<arm> PASS|FAIL(参考扱い) load1=<f> gpu_util=<n>`（10 行）
前提ゲート (a)(b) のいずれかが不成立なら性能判定へ進まず undetermined（run ログは不要）。

fail-closed: 欠損・重複・round 数 != 5・非有限値・未マスクの `/home/<name>`・前提ゲート
の入力不整合は `LogIntegrityError`（非ゼロ終了。判定を出力しない）。
"""

from __future__ import annotations

import argparse
import math
import re
import statistics
import sys
from pathlib import Path

RUNS = 5
SIZES = (512, 1024, 2048, 4096)
ARMS = ("base", "head")
UNMASKED_RE = re.compile(r"/home/[A-Za-z0-9_.-]+")
TEST_LINE_RE = re.compile(r"^test (\S+) \.\.\. (ok|FAILED|ignored)$")
RUNNING_RE = re.compile(r"^\s*Running (.+?) \(")
BENCH_RE = re.compile(
    r"^size=(\d+) tiled_f32_tflops=(\S+) wmma_f16_tflops=(\S+) mma_f16_tflops=(\S+) "
    r"mma_over_tiled=(\S+) mma_over_wmma=(\S+)$"
)
GATE_RE = re.compile(r"^(run\d+-(?:base|head)) (PASS|FAIL\(参考扱い\)) load1=(\S+) gpu_util=(\S+)$")


class LogIntegrityError(Exception):
    """完全性検査の失敗（黙って除外せず非ゼロ終了する）。"""


def check_masked(text: str, name: str) -> None:
    m = UNMASKED_RE.search(text)
    if m:
        raise LogIntegrityError(f"{name}: 未マスクの絶対パス {m[0]!r}")


def parse_parity(text: str, name: str) -> dict:
    """cargo test の出力 → {(バイナリ, テスト名): ok|FAILED}。重複・融合行・0 件は拒否する。"""
    check_masked(text, name)
    res: dict = {}
    binary = None
    for raw in text.splitlines():
        m = RUNNING_RE.match(raw)
        if m:
            binary = m[1]
            continue
        line = raw.rstrip()
        if line.startswith("test ") and not line.startswith("test result:"):
            t = TEST_LINE_RE.match(line)
            if not t:
                raise LogIntegrityError(f"{name}: 結果を読めない test 行（出力の融合?）: {line!r}")
            if binary is None:
                raise LogIntegrityError(f"{name}: Running 行より前の test 行: {line!r}")
            key = (binary, t[1])
            if key in res:
                raise LogIntegrityError(f"{name}: test の重複 {key}")
            res[key] = t[2]
    if not res:
        raise LogIntegrityError(f"{name}: test 行が 1 件も無い")
    if not any(v == "ok" for v in res.values()) and not any(v == "FAILED" for v in res.values()):
        raise LogIntegrityError(f"{name}: ok/FAILED のテストが無い")
    return res


def judge_parity(base: dict, head: dict) -> tuple[bool, str]:
    """前提 (a): head で FAILED かつ base で ok のテストが 1 件も無ければ成立。"""
    if set(base) != set(head):
        raise LogIntegrityError(
            f"parity: base と head のテスト集合が一致しない（差 {sorted(set(base) ^ set(head))[:5]}）")
    regress = sorted(k for k in head if head[k] == "FAILED" and base[k] == "ok")
    both_fail = sorted(k for k in head if head[k] == "FAILED" and base[k] == "FAILED")
    n_ok_head = sum(1 for v in head.values() if v == "ok")
    detail = (f"テスト {len(head)} 件（head ok {n_ok_head}・head FAILED {len(head) - n_ok_head}）。"
              f"後退（base ok → head FAILED）{len(regress)} 件"
              + (f": {regress}" if regress else "")
              + f"。base でも FAILED（区別して記録）{len(both_fail)} 件"
              + (f": {both_fail}" if both_fail else ""))
    return (not regress), detail


PTXAS_ENTRY_RE = re.compile(r"Compiling entry function '([^']+)' for '([^']+)'")
PTXAS_USED_RE = re.compile(r"Used (\d+) registers")
PTXAS_SPILL_RE = re.compile(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads")
PTXAS_LMEM_RE = re.compile(r"(\d+) bytes lmem")


def parse_ptxas(text: str, name: str) -> dict:
    """ptxas -v 出力 → registers／stack／spill／lmem。エントリ関数がちょうど 1 つであること。"""
    check_masked(text, name)
    entries = PTXAS_ENTRY_RE.findall(text)
    used = PTXAS_USED_RE.findall(text)
    spill = PTXAS_SPILL_RE.findall(text)
    if len(entries) != 1 or len(used) != 1 or len(spill) != 1:
        raise LogIntegrityError(
            f"{name}: ptxas -v の出力が 1 エントリ分でない（entry={len(entries)} used={len(used)} spill={len(spill)}）")
    lmem = PTXAS_LMEM_RE.findall(text)
    if len(lmem) > 1:
        raise LogIntegrityError(f"{name}: lmem 行が複数ある")
    return {"entry": entries[0][0], "regs": int(used[0]), "stack": int(spill[0][0]),
            "spill_st": int(spill[0][1]), "spill_ld": int(spill[0][2]),
            "lmem": int(lmem[0]) if lmem else 0}


def judge_ptxas(base: dict, head: dict) -> tuple[bool, str]:
    """前提 (b): head の lmem／stack が base より増えず、head のスピルが 0 なら成立。"""
    ok = (head["lmem"] <= base["lmem"] and head["stack"] <= base["stack"]
          and head["spill_st"] == 0 and head["spill_ld"] == 0)
    fmt = lambda d: (f"registers {d['regs']}・stack {d['stack']} B・spill st/ld {d['spill_st']}/{d['spill_ld']} B"
                     f"・lmem {d['lmem']} B")
    return ok, f"base: {fmt(base)} ／ head: {fmt(head)}"


def parse_bench(text: str, name: str) -> dict:
    """gemm_mma_bench 標準出力 → {size: mma_f16_tflops}。4 形状が 1 回ずつ・有限正値であること。"""
    check_masked(text, name)
    got: dict = {}
    for raw in text.splitlines():
        m = BENCH_RE.match(raw.strip())
        if not m:
            continue
        size = int(m[1])
        if size in got:
            raise LogIntegrityError(f"{name}: size={size} の重複")
        try:
            v = float(m[4])
        except ValueError as exc:
            raise LogIntegrityError(f"{name}: mma_f16_tflops が数値でない: {m[4]!r}") from exc
        if not math.isfinite(v) or v <= 0.0:
            raise LogIntegrityError(f"{name}: mma_f16_tflops が非有限／非正: {v}")
        got[size] = v
    if set(got) != set(SIZES):
        raise LogIntegrityError(f"{name}: 形状集合が {SIZES} と一致しない: {sorted(got)}")
    return got


def parse_gate(text: str) -> bool:
    """gate.log を検証し、10 行すべて PASS なら True（FAIL があれば False＝参考扱い）。"""
    check_masked(text, "gate")
    expect = {f"run{i}-{a}" for i in range(1, RUNS + 1) for a in ARMS}
    seen: dict = {}
    ok = True
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = GATE_RE.match(line)
        if not m:
            raise LogIntegrityError(f"gate: 認識できない行: {line!r}")
        if m[1] in seen:
            raise LogIntegrityError(f"gate: ラベルの重複 {m[1]}")
        try:
            load1, util = float(m[3]), int(m[4])
        except ValueError as exc:
            raise LogIntegrityError(f"gate: 実測値が数値でない: {line!r}") from exc
        meets = load1 < 1.0 and util == 0
        if (m[2] == "PASS") != meets:
            raise LogIntegrityError(f"gate: 判定と実測値が矛盾: {line!r}")
        seen[m[1]] = meets
        ok = ok and meets
    if set(seen) != expect:
        raise LogIntegrityError(f"gate: ラベル集合が不一致（不足 {sorted(expect - set(seen))}）")
    return ok


def judge_perf(base_runs: list, head_runs: list) -> tuple[str, list]:
    """RULE 8 の判定式。返り値は (採用候補|不採用|undetermined, 形状別の行)。"""
    rows = []
    all_ge = True
    any_improve = False
    any_regress = False
    for s in SIZES:
        b = [r[s] for r in base_runs]
        h = [r[s] for r in head_runs]
        mb, mh = statistics.median(b), statistics.median(h)
        imp = all(hh > bb for bb, hh in zip(b, h))
        reg = all(hh < bb for bb, hh in zip(b, h))
        all_ge = all_ge and mh >= mb
        any_improve = any_improve or imp
        any_regress = any_regress or reg
        rows.append((s, b, h, mb, mh, mh / mb, imp, reg))
    if any_regress:
        return "不採用", rows
    if all_ge and any_improve:
        return "採用候補", rows
    return "undetermined", rows


def aggregate(parity_base: str, parity_head: str, ptxas_base: str, ptxas_head: str,
              gate: str | None, runs: dict | None) -> str:
    """前提ゲート → 性能判定 → Markdown。runs は {(r, arm): text}（前提不成立なら None 可）。"""
    pb, ph = parse_parity(parity_base, "parity-base"), parse_parity(parity_head, "parity-head")
    a_ok, a_detail = judge_parity(pb, ph)
    xb, xh = parse_ptxas(ptxas_base, "ptxas-base"), parse_ptxas(ptxas_head, "ptxas-head")
    if xb["entry"] != xh["entry"]:
        raise LogIntegrityError("ptxas: base と head のエントリ関数名が違う")
    b_ok, b_detail = judge_ptxas(xb, xh)
    out = ["## B5 判定（RULE.txt 8・RULE-reopen.txt）", "",
           f"- 前提 (a) parity 非後退: **{'成立' if a_ok else '不成立'}** — {a_detail}",
           f"- 前提 (b) レジスタ・local memory: **{'成立' if b_ok else '不成立'}** — {b_detail}", ""]
    if not (a_ok and b_ok):
        out.append("**判定: undetermined（前提ゲート不成立のため性能判定へ進まない）**")
        return "\n".join(out) + "\n"
    if gate is None or runs is None:
        raise LogIntegrityError("前提ゲートが成立しているのに gate／run ログが無い")
    gate_ok = parse_gate(gate)
    base_runs, head_runs = [], []
    for r in range(1, RUNS + 1):
        base_runs.append(parse_bench(runs[(r, "base")], f"run{r}-base"))
        head_runs.append(parse_bench(runs[(r, "head")], f"run{r}-head"))
    verdict, rows = judge_perf(base_runs, head_runs)
    if not gate_ok:
        verdict_txt = f"undetermined（専有ゲート FAIL の run があるため参考扱い。ゲートなしの機械判定は {verdict}）"
    else:
        verdict_txt = verdict
    out += ["| size | run 別 base（TFLOPS） | run 別 head（TFLOPS） | base 中央値 | head 中央値 | head/base | 5/5 改善 | 5/5 後退 |",
            "|---:|---|---|---:|---:|---:|:-:|:-:|"]
    for s, b, h, mb, mh, ratio, imp, reg in rows:
        out.append(f"| {s} | {' / '.join(f'{x:.4f}' for x in b)} | {' / '.join(f'{x:.4f}' for x in h)} "
                   f"| {mb:.4f} | {mh:.4f} | {ratio:.4f} | {'○' if imp else '-'} | {'○' if reg else '-'} |")
    out += ["", f"- 専有ゲート: {'全 10 run PASS' if gate_ok else 'FAIL あり（参考扱い）'}",
            f"- **判定: {verdict_txt}**"]
    return "\n".join(out) + "\n"


def load_dir(d: Path) -> str:
    def rd(n: str) -> str:
        p = d / n
        if not p.is_file():
            raise LogIntegrityError(f"ログが無い: {p}")
        return p.read_text(encoding="utf-8")
    pb, ph, xb, xh = (rd("parity-base.log"), rd("parity-head.log"),
                      rd("ptxas-base.log"), rd("ptxas-head.log"))
    # 前提が成立する場合だけ run ログを要求する（成立判定は aggregate が行う）。
    try:
        gate = rd("gate.log")
        runs = {(r, a): rd(f"run{r}-{a}.log") for r in range(1, RUNS + 1) for a in ARMS}
    except LogIntegrityError:
        gate, runs = None, None
    return aggregate(pb, ph, xb, xh, gate, runs)


# ---- self-test ----------------------------------------------------------

def _parity(fail: tuple = (), extra: tuple = ()) -> str:
    names = ["a_ok", "b_ok", "c_ok", *extra]
    lines = ["     Running tests/gemm_mma.rs (target/release/deps/gemm_mma-0)", "", "running 3 tests"]
    for n in names:
        lines.append(f"test {n} ... {'FAILED' if n in fail else 'ok'}")
    lines.append("test result: ok. 3 passed; 0 failed")
    return "\n".join(lines) + "\n"


def _ptxas(regs=120, stack=0, st=0, ld=0, lmem=None) -> str:
    s = ("ptxas info    : Compiling entry function 'gemm_mma_f16' for 'sm_121'\n"
         f"ptxas info    : Function properties for gemm_mma_f16\n"
         f"    {stack} bytes stack frame, {st} bytes spill stores, {ld} bytes spill loads\n"
         f"ptxas info    : Used {regs} registers, used 1 barriers"
         + (f", {lmem} bytes lmem" if lmem is not None else "") + ", 28160 bytes smem\n")
    return s


def _bench(vals: dict) -> str:
    return "".join(f"size={s} tiled_f32_tflops=1.0000 wmma_f16_tflops=2.0000 mma_f16_tflops={vals[s]:.4f} "
                   f"mma_over_tiled=1.0000 mma_over_wmma=1.0000\n" for s in SIZES)


def _gate(bad: str | None = None) -> str:
    out = []
    for i in range(1, RUNS + 1):
        for a in ARMS:
            lab = f"run{i}-{a}"
            out.append(f"{lab} FAIL(参考扱い) load1=2.00 gpu_util=0" if lab == bad
                       else f"{lab} PASS load1=0.10 gpu_util=0")
    return "\n".join(out) + "\n"


def _runs(head_scale) -> dict:
    base = {s: 10.0 + i for i, s in enumerate(SIZES)}
    res = {}
    for r in range(1, RUNS + 1):
        for a in ARMS:
            f = head_scale(r) if a == "head" else (lambda s: 1.0)
            res[(r, a)] = _bench({s: base[s] * f(s) for s in SIZES})
    return res


def self_test() -> None:
    def must_fail(fn, label):
        try:
            fn()
        except LogIntegrityError:
            return
        raise AssertionError(f"fail-closed でない: {label}")

    pb, ph, xb, xh = _parity(), _parity(), _ptxas(), _ptxas()
    # 採用候補: 全形状 head >= base、かつ 1 形状で 5/5 改善
    runs = _runs(lambda r: (lambda s: 1.02 if s == 1024 else 1.0))
    md = aggregate(pb, ph, xb, xh, _gate(), runs)
    assert "判定: 採用候補" in md, md
    # 不採用: 1 形状で 5/5 後退
    md = aggregate(pb, ph, xb, xh, _gate(), _runs(lambda r: (lambda s: 0.97 if s == 4096 else 1.01)))
    assert "判定: 不採用" in md, md
    # undetermined: 全形状同値（5/5 改善なし）
    md = aggregate(pb, ph, xb, xh, _gate(), _runs(lambda r: (lambda s: 1.0)))
    assert "判定: undetermined" in md, md
    # undetermined: 中央値が下回るが 5/5 後退ではない
    def mixed(r):
        return lambda s: (0.9 if (s == 512 and r <= 3) else 1.05 if s == 512 else 1.01)
    md = aggregate(pb, ph, xb, xh, _gate(), _runs(mixed))
    assert "判定: undetermined" in md, md
    # 専有ゲート FAIL は参考扱い
    md = aggregate(pb, ph, xb, xh, _gate("run2-head"), runs)
    assert "参考扱い" in md and "判定: undetermined" in md, md
    # 前提 (a) 後退 → undetermined（run ログ不要）
    md = aggregate(pb, _parity(fail=("b_ok",)), xb, xh, None, None)
    assert "前提 (a) parity 非後退: **不成立**" in md and "判定: undetermined" in md, md
    # base でも FAILED なら後退ではない
    md = aggregate(_parity(fail=("b_ok",)), _parity(fail=("b_ok",)), xb, xh, _gate(), runs)
    assert "前提 (a) parity 非後退: **成立**" in md and "base でも FAILED" in md, md
    # 前提 (b) lmem 増加／スピル → undetermined
    md = aggregate(pb, ph, xb, _ptxas(lmem=64), None, None)
    assert "前提 (b) レジスタ・local memory: **不成立**" in md, md
    md = aggregate(pb, ph, xb, _ptxas(st=16, ld=16), None, None)
    assert "前提 (b) レジスタ・local memory: **不成立**" in md, md
    # fail-closed 群
    must_fail(lambda: aggregate(pb, ph, xb, xh, None, None), "前提成立なのに run ログ無し")
    must_fail(lambda: aggregate(pb, _parity(extra=("d_ok",)), xb, xh, None, None), "テスト集合の不一致")
    must_fail(lambda: aggregate(pb + "test dup ... ok\ntest dup ... ok\n", ph, xb, xh, None, None), "test 重複")
    must_fail(lambda: aggregate(pb, "     Running tests/x.rs (a)\ntest a ... 12.3 ok\n", xb, xh, None, None), "融合行")
    must_fail(lambda: aggregate(pb, "no tests\n", xb, xh, None, None), "test 0 件")
    must_fail(lambda: aggregate(pb, ph, xb, xh + xh, None, None), "ptxas 2 エントリ")
    must_fail(lambda: aggregate(pb, ph, xb, "nothing", None, None), "ptxas 空")
    must_fail(lambda: aggregate(pb + "/home/someone/x\n", ph, xb, xh, None, None), "未マスク")
    r2 = dict(runs)
    r2[(3, "head")] = r2[(3, "head")].replace("size=2048", "size=2049")
    must_fail(lambda: aggregate(pb, ph, xb, xh, _gate(), r2), "形状欠落")
    r3 = dict(runs)
    r3[(1, "base")] = r3[(1, "base")].replace("mma_f16_tflops=10.0000", "mma_f16_tflops=nan")
    must_fail(lambda: aggregate(pb, ph, xb, xh, _gate(), r3), "非有限")
    r4 = dict(runs)
    r4[(1, "base")] = r4[(1, "base")] + r4[(1, "base")]
    must_fail(lambda: aggregate(pb, ph, xb, xh, _gate(), r4), "形状重複")
    must_fail(lambda: aggregate(pb, ph, xb, xh, _gate().replace("run5-head", "run6-head"), runs), "ゲートラベル不一致")
    must_fail(lambda: aggregate(pb, ph, xb, xh, _gate().replace("PASS load1=0.10", "PASS load1=3.00", 1), runs),
              "PASS だが実測が専有条件を満たさない")
    print("self-test OK")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log_dir", nargs="?")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    if not args.log_dir:
        ap.error("log_dir が必要")
    try:
        sys.stdout.write(load_dir(Path(args.log_dir)))
    except LogIntegrityError as exc:
        print(f"FAIL（fail-closed）: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
