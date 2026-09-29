#!/usr/bin/env python3
"""イシュー #2100: backward 診断ログ（DIAG_BACKWARD 行）の iteration 集計器。

役割: `orchestrate_backward_diag.sh` が収録した
`run{R}-{device}-{mode}-{instr,plain}.{err,jsonl}` を読み、

  1. 計装あり／計装なしの JSONL `checksum` の完全一致を fail-closed で検査する
     （計装が数値を変えないことの証明。不一致なら系列を無効として exit 1）
  2. 各プロセスの DIAG_BACKWARD 行がちょうど 100 行（train --phases の
     TRAIN_STEPS）で、seq が単調増加であることを検査する
  3. iteration ごとの CSV（iterations.csv）を出力する
  4. セル別（device × mode）の集計表（aggregate.md）を出力する
     （各 run で step 20..99 の中央値 → その run 中央値の中央値。
     step < 20 は warmup として CSV に is_warmup=1 で残すが集計から除外）

標準ライブラリのみ（追加依存なし）。ログ内容は厳格な正規表現で解析するだけで、
eval 等でコードとして評価しない（.claude/rules/security.md A03）。
`--self-test` は内蔵 fixture で解析・検査・集計を検証する
（`scripts/bench/framework-compare/parity_torch_truth.py` と同方式）。

判定種別は record_only（RULE.txt）。本スクリプトは ADOPT／REJECT を出力しない。
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import statistics
import sys
import tempfile
import unittest
from pathlib import Path

# train --phases は 1 プロセス 100 step・先頭 20 step が warmup
# （scripts/bench/framework-compare/bench-fandhe/src/main.rs の
# TRAIN_STEPS／TRAIN_WARMUP）。
TRAIN_STEPS = 100
TRAIN_WARMUP = 20

# DIAG_BACKWARD 行のキー順（crates/autodiff/src/diag.rs::report_backward）。
KEYS = [
    "seq",
    "total_ns",
    "vjp_ns",
    "gemm_ns",
    "fill_ns",
    "mask_ns",
    "ewise_ns",
    "transpose_ns",
    "materialize_ns",
    "accumulate_ns",
    "loss_ns",
    "vjp_calls",
    "gemm_calls",
    "mask_calls",
]
LINE_RE = re.compile(
    r"^DIAG_BACKWARD " + " ".join(rf"{k}=(\d+)" for k in KEYS) + r"$"
)
# gemm_ns 以外の「カテゴリ」（total との残差計算に使う。fill_ns は gemm_ns の内訳
# なので二重計上しない）。
CATEGORIES = [
    "gemm_ns",
    "mask_ns",
    "ewise_ns",
    "transpose_ns",
    "materialize_ns",
    "accumulate_ns",
    "loss_ns",
]
MODES = ["fresh", "reuse"]
CSV_COLUMNS = (
    ["machine", "device", "mode", "run", "step", "is_warmup"]
    + KEYS
    + ["nongemm_ns"]
)


class AggregateError(Exception):
    """fail-closed 検査の失敗（系列を無効とする）。"""


def parse_diag_lines(text: str, label: str) -> list[dict[str, int]]:
    """DIAG_BACKWARD 行だけを厳格に解析する。件数・seq 単調性を検査する。"""
    rows: list[dict[str, int]] = []
    for line in text.splitlines():
        if not line.startswith("DIAG_BACKWARD"):
            continue
        m = LINE_RE.match(line.strip())
        if m is None:
            raise AggregateError(f"{label}: 形式不正の DIAG_BACKWARD 行: {line[:120]!r}")
        rows.append({k: int(v) for k, v in zip(KEYS, m.groups())})
    if len(rows) != TRAIN_STEPS:
        raise AggregateError(
            f"{label}: DIAG_BACKWARD 行が {len(rows)} 件（期待 {TRAIN_STEPS} 件）"
        )
    seqs = [r["seq"] for r in rows]
    if any(b <= a for a, b in zip(seqs, seqs[1:])):
        raise AggregateError(f"{label}: seq が単調増加でない")
    return rows


REQUIRED_PHASES = ("backward", "step_total")


def read_checksums(text: str, label: str) -> list[tuple[str, float]]:
    """JSONL 全行の (phase, checksum) を出現順に返す。集合へ縮約しない。

    欠落・空・phase 重複・必須 phase（backward／step_total）の欠落は fail-closed。
    """
    rows: list[tuple[str, float]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AggregateError(f"{label}: JSONL 解析失敗: {exc}") from exc
        if "checksum" not in obj or not isinstance(obj.get("phase"), str):
            raise AggregateError(f"{label}: phase／checksum 欠落")
        rows.append((obj["phase"], float(obj["checksum"])))
    if not rows:
        raise AggregateError(f"{label}: JSONL が空")
    phases = [p for p, _ in rows]
    if len(set(phases)) != len(phases):
        raise AggregateError(f"{label}: phase 重複 {phases}")
    for req in REQUIRED_PHASES:
        if req not in phases:
            raise AggregateError(f"{label}: 必須 phase {req} なし")
    return rows


def compare_checksums(
    instr: list[tuple[str, float]], plain: list[tuple[str, float]], label: str
) -> None:
    """計装あり／なしで phase 列・行数・各行 checksum が完全一致することを検査する。"""
    if [p for p, _ in instr] != [p for p, _ in plain]:
        raise AggregateError(
            f"{label}: phase 列が不一致 instr={[p for p, _ in instr]} "
            f"plain={[p for p, _ in plain]}"
        )
    for (phase, ci), (_, cp) in zip(instr, plain):
        if not ci == cp:  # NaN は不一致扱い（fail-closed）
            raise AggregateError(
                f"{label}: checksum 不一致 phase={phase} instr={ci!r} plain={cp!r}"
                "（計装が数値を変えた。系列は無効）"
            )


def read_step_total_median(text: str) -> float | None:
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if obj.get("phase") == "step_total":
            return float(obj["median_s"])
    return None


def median(values: list[float]) -> float:
    return float(statistics.median(values))


def collect(machine: str, in_dir: Path, runs: int, devices: list[str]):
    """全セルを読み込み検査する。戻り値: (csv 行, セル別 run 中央値, オーバーヘッド)。"""
    csv_rows: list[list[object]] = []
    cells: dict[tuple[str, str], dict[str, list[float]]] = {}
    overhead: dict[tuple[str, str], list[float]] = {}
    for dev in devices:
        for mode in MODES:
            per_metric: dict[str, list[float]] = {}
            for run in range(1, runs + 1):
                base = in_dir / f"run{run}-{dev}-{mode}"
                label = f"{dev}/{mode}/run{run}"
                paths = {
                    "err": Path(f"{base}-instr.err"),
                    "ij": Path(f"{base}-instr.jsonl"),
                    "pj": Path(f"{base}-plain.jsonl"),
                }
                for k, p in paths.items():
                    if not p.is_file():
                        raise AggregateError(f"{label}: 入力ファイルなし: {p.name}")
                texts = {k: p.read_text(encoding="utf-8") for k, p in paths.items()}
                # 1. checksum 完全一致（fail-closed）
                si = read_checksums(texts["ij"], label + " instr")
                sp = read_checksums(texts["pj"], label + " plain")
                compare_checksums(si, sp, label)
                # 2. DIAG 行の件数・seq
                rows = parse_diag_lines(texts["err"], label)
                for step, r in enumerate(rows):
                    is_warm = 1 if step < TRAIN_WARMUP else 0
                    nongemm = r["total_ns"] - r["gemm_ns"]
                    csv_rows.append(
                        [machine, dev, mode, run, step, is_warm]
                        + [r[k] for k in KEYS]
                        + [nongemm]
                    )
                # 3. run 内中央値（step 20..99）
                measured = rows[TRAIN_WARMUP:]
                for k in KEYS[1:]:
                    per_metric.setdefault(k, []).append(median([r[k] for r in measured]))
                per_metric.setdefault("nongemm_ns", []).append(
                    median([r["total_ns"] - r["gemm_ns"] for r in measured])
                )
                # 4. 計装オーバーヘッド比（記録のみ・判定しない）
                mi = read_step_total_median(texts["ij"])
                mp = read_step_total_median(texts["pj"])
                if mi is not None and mp:
                    overhead.setdefault((dev, mode), []).append(mi / mp)
            cells[(dev, mode)] = per_metric
    return csv_rows, cells, overhead


def load_gate(
    in_dir: Path, runs: int, devices: list[str]
) -> dict[tuple[str, str], str]:
    """gate.tsv からセル別のゲート通過状況（全 run 通過なら 'pass'）を返す。

    RULE.txt「1 run でも不通過ならその系列は参考扱い」に合わせ、各 device × mode に
    予定した run 1..runs がちょうど 1 行ずつ（欠落・重複・範囲外・不正値なし）揃い、
    かつ全行が pass=1 のときだけ 'pass' とする。それ以外は通過扱いにしない。
    """
    p = in_dir / "gate.tsv"
    if not p.is_file():
        return {}
    seen: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for i, line in enumerate(p.read_text(encoding="utf-8").splitlines()):
        if i == 0 or not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) != 6:
            continue
        seen.setdefault((parts[1], parts[2]), []).append((parts[0], parts[5]))
    expected = [str(r) for r in range(1, runs + 1)]
    result: dict[tuple[str, str], str] = {}
    for dev in devices:
        for mode in MODES:
            recs = seen.get((dev, mode), [])
            run_ids = sorted(r for r, _ in recs)
            complete = run_ids == sorted(expected)
            if complete and all(v == "1" for _, v in recs):
                result[(dev, mode)] = "pass"
            elif not complete:
                result[(dev, mode)] = "参考扱い（gate.tsv の run 記録が欠落・重複・不正）"
            else:
                result[(dev, mode)] = "参考扱い（ゲート不通過あり）"
    return result


def render_md(machine, cells, overhead, gate, runs) -> str:
    out = io.StringIO()
    w = out.write
    w(f"# backward 非 GEMM 内訳の集計（イシュー #2100・machine={machine}）\n\n")
    w("判定種別: **record_only**（診断のみ。ADOPT／REJECT は判定しない）。\n\n")
    w(
        f"集計方法: 各 run で step {TRAIN_WARMUP}..{TRAIN_STEPS - 1} の中央値 → "
        f"その {runs} run の中央値。単位は µs（`total`／`gemm` 等の壁時計時間）。\n\n"
    )
    w(
        "注記: `gemm` は resident grad staging 書き込み（`fill`。bias 縮約を含みうる）を"
        "含む（旧診断 §4 と比較可能）。`fill` は `gemm` の内訳で二重計上しない。"
        "`非 GEMM` は step ごとに total − gemm を求めてから中央値を取る（表の total 行と "
        "gemm 行の中央値同士の差とは一致しない）。"
        "`残差` = total − Σ(gemm+mask+ewise+transpose+materialize+accumulate+loss)"
        "（走査ループ・grads clone・checkpoint 再解放・未計装 arm 等）。"
        "GPU デバイスは非同期処理が同期点のカテゴリへ計上される。\n\n"
    )
    for (dev, mode), m in sorted(cells.items()):
        med = {k: median(v) for k, v in m.items()}
        total = med["total_ns"]
        cat_sum = sum(med[c] for c in CATEGORIES)
        w(f"## device={dev} mode={mode}\n\n")
        g = gate.get((dev, mode), "gate.tsv なし")
        w(f"- 負荷ゲート: {g}\n")
        ov = overhead.get((dev, mode))
        if ov:
            w(f"- 計装オーバーヘッド比（記録のみ）: step_total 計装あり÷なし の run 中央値 = {median(ov):.3f}\n")
        w("\n| カテゴリ | µs | total 比 |\n|---|---:|---:|\n")
        rows = [("total", total)]
        rows += [(c.removesuffix("_ns"), med[c]) for c in CATEGORIES]
        rows += [("(fill ⊂ gemm)", med["fill_ns"]), ("vjp（参考）", med["vjp_ns"])]
        rows += [
            ("**非 GEMM（step ごとの total − gemm の中央値）**", med["nongemm_ns"]),
            ("残差（中央値同士の差）", total - cat_sum),
        ]
        for name, v in rows:
            pct = (v / total * 100.0) if total else float("nan")
            w(f"| {name} | {v / 1000.0:.2f} | {pct:.1f}% |\n")
        w(
            f"\n呼び出し回数（run 中央値）: vjp={med['vjp_calls']:.0f} "
            f"gemm={med['gemm_calls']:.0f} mask={med['mask_calls']:.0f}\n\n"
        )
    return out.getvalue()


def run(args: argparse.Namespace) -> int:
    in_dir = Path(args.in_dir)
    devices = args.devices.split()
    try:
        csv_rows, cells, overhead = collect(args.machine, in_dir, args.runs, devices)
    except AggregateError as exc:
        print(f"AGGREGATE_FAIL: {exc}", file=sys.stderr)
        return 1
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        wtr = csv.writer(f)
        wtr.writerow(CSV_COLUMNS)
        wtr.writerows(csv_rows)
    md = render_md(args.machine, cells, overhead, load_gate(in_dir, args.runs, devices), args.runs)
    Path(args.out_md).write_text(md, encoding="utf-8")
    print(f"wrote {args.out_csv} ({len(csv_rows)} rows), {args.out_md}")
    return 0


# ---------------------------------------------------------------- self-test


def _diag_line(seq: int, total: int = 1000, gemm: int = 600) -> str:
    vals = dict(
        seq=seq, total_ns=total, vjp_ns=900, gemm_ns=gemm, fill_ns=100, mask_ns=50,
        ewise_ns=80, transpose_ns=20, materialize_ns=30, accumulate_ns=40,
        loss_ns=60, vjp_calls=9, gemm_calls=7, mask_calls=2,
    )
    return "DIAG_BACKWARD " + " ".join(f"{k}={vals[k]}" for k in KEYS)


def _jsonl(checksum: float, step_total: float = 0.001) -> str:
    return (
        json.dumps({"phase": "backward", "median_s": 0.0005, "checksum": checksum})
        + "\n"
        + json.dumps({"phase": "step_total", "median_s": step_total, "checksum": checksum})
        + "\n"
    )


class SelfTest(unittest.TestCase):
    def _make(self, d: Path, *, n_lines=TRAIN_STEPS, sum_plain=0.5, sum_instr=0.5):
        for dev in ["cpu"]:
            for mode in MODES:
                for run_i in (1, 2):
                    base = d / f"run{run_i}-{dev}-{mode}"
                    lines = ["noise: hello"] + [_diag_line(s) for s in range(n_lines)]
                    Path(f"{base}-instr.err").write_text("\n".join(lines) + "\n")
                    Path(f"{base}-instr.jsonl").write_text(_jsonl(sum_instr, 0.0012))
                    Path(f"{base}-plain.jsonl").write_text(_jsonl(sum_plain, 0.0010))

    def test_parse_ok(self):
        rows = parse_diag_lines("\n".join(_diag_line(s) for s in range(TRAIN_STEPS)), "t")
        self.assertEqual(len(rows), TRAIN_STEPS)
        self.assertEqual(rows[3]["seq"], 3)

    def test_parse_rejects_bad_count(self):
        with self.assertRaises(AggregateError):
            parse_diag_lines("\n".join(_diag_line(s) for s in range(99)), "t")

    def test_parse_rejects_non_monotonic_seq(self):
        lines = [_diag_line(s) for s in range(TRAIN_STEPS)]
        lines[10] = _diag_line(3)
        with self.assertRaises(AggregateError):
            parse_diag_lines("\n".join(lines), "t")

    def test_parse_rejects_malformed(self):
        with self.assertRaises(AggregateError):
            parse_diag_lines("DIAG_BACKWARD seq=1 total_ns=abc", "t")

    def test_checksum_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            self._make(d, sum_instr=0.6)
            with self.assertRaises(AggregateError):
                collect("t", d, 2, ["cpu"])

    def test_checksum_per_row_and_phase_validation(self):
        ok = [("backward", 0.5), ("step_total", 0.5)]
        compare_checksums(ok, list(ok), "t")
        # 行ごとの不一致（集合に縮約すると見逃す形）
        with self.assertRaises(AggregateError):
            compare_checksums(ok, [("backward", 0.5), ("step_total", 0.6)], "t")
        # phase 列・行数の不一致
        with self.assertRaises(AggregateError):
            compare_checksums(ok, [("backward", 0.5)], "t")
        with self.assertRaises(AggregateError):
            compare_checksums(ok, [("step_total", 0.5), ("backward", 0.5)], "t")
        # 必須 phase 欠落・phase 重複
        with self.assertRaises(AggregateError):
            read_checksums(json.dumps({"phase": "backward", "checksum": 1.0}), "t")
        dup = "\n".join(json.dumps({"phase": "backward", "checksum": 1.0}) for _ in range(2))
        with self.assertRaises(AggregateError):
            read_checksums(dup, "t")

    def test_end_to_end(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            self._make(d)
            rows, cells, ov = collect("t", d, 2, ["cpu"])
            self.assertEqual(len(rows), 2 * 2 * TRAIN_STEPS)  # 2 mode × 2 run
            med = {k: median(v) for k, v in cells[("cpu", "fresh")].items()}
            self.assertEqual(med["nongemm_ns"], 400)
            self.assertAlmostEqual(median(ov[("cpu", "fresh")]), 1.2)
            md = render_md("t", cells, ov, {}, 2)
            self.assertIn("record_only", md)
            self.assertIn("非 GEMM", md)
            # warmup 列
            self.assertEqual(sum(r[5] for r in rows), 2 * 2 * TRAIN_WARMUP)

    def _gate(self, d: Path, rows: list[tuple[int, str, str, int]]) -> None:
        lines = ["run\tdevice\tmode\tload1\tgpu_util\tpass"]
        lines += [f"{r}\t{dv}\t{m}\t0.1\t0\t{p}" for r, dv, m, p in rows]
        (d / "gate.tsv").write_text("\n".join(lines) + "\n")

    def test_load_gate_requires_all_runs_unique(self):
        full = [(r, "cpu", m, 1) for m in MODES for r in (1, 2)]
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            self._gate(d, full)
            g = load_gate(d, 2, ["cpu"])
            self.assertEqual(g[("cpu", "fresh")], "pass")
            # 欠落（run2 の行なし）は pass にしない
            self._gate(d, [x for x in full if x[:3] != (2, "cpu", "fresh")])
            self.assertNotEqual(load_gate(d, 2, ["cpu"])[("cpu", "fresh")], "pass")
            # 重複（run1 が 2 行）で run2 欠落も pass にしない
            self._gate(d, full + [(1, "cpu", "reuse", 1)])
            self.assertNotEqual(load_gate(d, 2, ["cpu"])[("cpu", "reuse")], "pass")
            # 範囲外の run 番号
            self._gate(d, [(1, "cpu", "fresh", 1), (3, "cpu", "fresh", 1)])
            self.assertNotEqual(load_gate(d, 2, ["cpu"])[("cpu", "fresh")], "pass")
            # セル自体が無い
            self.assertNotEqual(load_gate(d, 2, ["cpu"])[("cpu", "reuse")], "pass")
            # 不通過あり
            self._gate(d, [(r, "cpu", m, int(r == 1)) for m in MODES for r in (1, 2)])
            self.assertNotEqual(load_gate(d, 2, ["cpu"])[("cpu", "fresh")], "pass")

    def test_missing_input_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(AggregateError):
                collect("t", Path(td), 1, ["cpu"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--machine")
    ap.add_argument("--in-dir")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--devices", default="cpu")
    ap.add_argument("--out-csv")
    ap.add_argument("--out-md")
    args = ap.parse_args()
    if args.self_test:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(SelfTest)
        res = unittest.TextTestRunner(verbosity=1).run(suite)
        return 0 if res.wasSuccessful() else 1
    for req in ("machine", "in_dir", "out_csv", "out_md"):
        if getattr(args, req) is None:
            ap.error(f"--{req.replace('_', '-')} is required")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
