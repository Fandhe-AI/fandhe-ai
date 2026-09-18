#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TMA ロード経路 Stage 1（イシュー #1975・#1976）のゲート C（純カーネル時間
5 回中央値比較）を集計し、`RULE.txt` の事前登録判定規則を機械適用して
`aggregate.md` 相当の Markdown を生成する。

python3 標準ライブラリのみを使う（`docs/perf/logs/metal-gemm-smem-swizzle-ab-1970/
aggregate.py` と `docs/perf/logs/train-infer-phases-0.9.0-1980-1981/aggregate.py`
の書式・`--self-test` を踏襲）。

# 入力

`gateC_run{1..5}.log`（`cargo run --example gemm_tiled_pipeline_bench` の
stdout）。1 size につき 1 行、`size=<N> key=value ... | key=value ...` の
形式（`|` は区切りのみでトークンとしては無視する）。値は有限小数または
`skipped`（example の実出力では `n/a`。両方を skipped として扱う）。行頭が `size=` でない行（cargo の Compiling／Running 等）は無視する。

対象キー（RULE.txt ゲート C）:
  - 分母（共通）: `pipeline3_gpu_only_tflops`・`pipeline128x64_gpu_only_tflops`
  - 各腕（`none`／`b64`）: `tma_<arm>_gpu_only_tflops`・
    `tma_<arm>_over_pipeline3_gpu_only`（主判定）・
    `tma_<arm>_over_pipeline128x64_gpu_only`（参考）

任意入力: `load_gate.log`（`run=N load1=X gpu_util=Y pass|fail` 形式。
専有ゲート通過状況を run ごとに表へ併記する。不在でも致命的エラーにしない）。

# 判定規則（RULE.txt ゲート C の逐語適用。事後緩和しない）

候補腕（`none`・`--b64-hypothesis ok` のときのみ `b64` も候補腕）ごとに
size∈{256,512,1024,2048,4096} の 5 run 中央値（主判定列
`tma_<arm>_over_pipeline3_gpu_only`）を用いて次の 3 分岐を適用する。

  - ADOPT 候補: size∈{1024,2048,4096} のいずれかで中央値 >= 1.05、かつ
    5 size 全てで中央値 >= 1.00。
  - REJECT: いずれかの size で中央値 < 1.00。
  - undetermined: 上記いずれでもない。または当該腕の主判定列に
    skipped／欠損／run 数不足（5 run 未満）が 1 件でもある場合。

`--b64-hypothesis gap`（ゲート A で B64 腕の仮説不成立）の場合、b64 腕は
判定式をそのまま計算はするが「参考（仮説不成立）」として候補腕から除外し、
総合判定サマリには含めない。

5 run 中央値と反対側（>=1.00 か <1.00 か）の run が 2 件以上ある size は、
判定自体は中央値で行ったうえで「5/5 一貫ではない」と併記する（規則を緩めない
記録用の注記）。

# 想定外入力の扱い（fail-soft。self-test で検証）

  - 行の欠損（あるサイズの行が特定 run に無い）: そのサイズ・その run の値を
    「欠損」として扱い、当該腕・当該 size の主判定列に欠損があれば
    undetermined 化する（skipped と同様の扱い）。
  - run ファイルが 5 件未満: 検出できた run 数を明示し、5 件未満の腕は
    「run 数不足」で undetermined とする。
  - size 不揃い（RULE.txt 既定の 256/512/1024/2048/4096 以外の size 行）:
    無視せず「想定外 size」として警告に記録するが、判定には使わない
    （RULE.txt が列挙する 5 size のみが判定対象のため）。
  - 数値変換に失敗した値（`skipped` でも数値でもない文字列）: 「不正値」として
    警告に記録し、欠損と同様に扱う（fail-closed。黙って読み捨てない）。
"""

from __future__ import annotations

import argparse
import re
import statistics
import sys
from pathlib import Path

SIZES = [256, 512, 1024, 2048, 4096]
ADOPT_TRIGGER_SIZES = {1024, 2048, 4096}
ARMS = ["none", "b64"]
REQUIRED_RUNS = 5

RUN_FILE_RE = re.compile(r"^gateC_run(?P<run>[1-5])\.log$")
LOAD_GATE_LINE_RE = re.compile(
    r"^run=(?P<run>\d+)\s+load1=(?P<load1>[0-9.]+)\s+gpu_util=(?P<gpu_util>\d+)\s+(?P<status>pass|fail)$"
)

# 値のトークン化に用いる sentinel（float ではなく明示的な文字列で区別する。
# NaN は比較演算が常に False になり fail-closed の判定に使いにくいため）。
SKIPPED = "skipped"
# example 側（`gemm_tiled_pipeline_bench.rs`）は既存列の慣例に合わせ skip を `n/a` と出力する。
# 両表記を skipped として扱う（いずれも undetermined 化の対象）。
SKIP_MARKERS = (SKIPPED, "n/a")
MISSING = "missing"
INVALID = "invalid"


def discover_runs(log_dir: Path) -> dict[int, Path]:
    """`log_dir` から run 番号（1〜5）-> `gateC_run<N>.log` の対応を作る。

    ファイル名完全一致のみを受理し、退避ログ・範囲外の run 番号を除外する
    （`metal-gemm-smem-swizzle-ab-1970/aggregate.py::discover_runs` と同方針）。
    """
    runs: dict[int, Path] = {}
    for p in sorted(log_dir.iterdir()):
        if not p.is_file():
            continue
        m = RUN_FILE_RE.match(p.name)
        if m:
            runs[int(m.group("run"))] = p
    return runs


def parse_gate_c_log(path: Path, warnings: list[str]) -> dict[int, dict[str, object]]:
    """1 run 分の `gateC_run<N>.log` から size -> {key: 値} を抽出する。

    値は float（数値）／`SKIPPED`／`INVALID`（数値でも skipped でもない文字列）
    のいずれかとして格納する。`size=` で始まらない行は無視する（cargo の
    ビルドログ等）。`|` はトークンとして現れるが `key=value` 形式に
    一致しないため自然に無視される。
    """
    result: dict[int, dict[str, object]] = {}
    text = path.read_text(encoding="utf-8", errors="replace")
    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line.startswith("size="):
            continue
        tokens = line.split()
        entry: dict[str, object] = {}
        size_val: int | None = None
        for tok in tokens:
            if tok == "|":
                continue
            if "=" not in tok:
                # `|` 以外の非 key=value トークンは想定外として警告する。
                warnings.append(f"{path.name}:{lineno}: 非 key=value トークン {tok!r} を無視")
                continue
            key, _, value = tok.partition("=")
            if key == "size":
                try:
                    size_val = int(value)
                except ValueError:
                    warnings.append(f"{path.name}:{lineno}: size 値 {value!r} を整数化できない行を無視")
                    size_val = None
                    break
                continue
            if value in SKIP_MARKERS:
                entry[key] = SKIPPED
                continue
            try:
                entry[key] = float(value)
            except ValueError:
                entry[key] = INVALID
                warnings.append(f"{path.name}:{lineno}: key={key} の値 {value!r} が数値でも skipped でもない")
        if size_val is None:
            continue
        if size_val in result:
            warnings.append(f"{path.name}:{lineno}: size={size_val} の重複行（後勝ちで上書き）")
        result[size_val] = entry
        if size_val not in SIZES:
            warnings.append(f"{path.name}:{lineno}: 想定外 size={size_val}（判定対象は {SIZES}）")
    return result


def parse_load_gate(path: Path) -> list[dict[str, object]]:
    """`load_gate.log`（`run=N load1=X gpu_util=Y pass|fail`）を解析する。"""
    rows: list[dict[str, object]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        m = LOAD_GATE_LINE_RE.match(line)
        if not m:
            continue
        rows.append(
            {
                "run": int(m.group("run")),
                "load1": float(m.group("load1")),
                "gpu_util": int(m.group("gpu_util")),
                "status": m.group("status"),
            }
        )
    return rows


def value_at(data: dict[int, dict[int, dict[str, object]]], run_no: int, size: int, key: str) -> object:
    """run_no・size・key の値を取り出す。行自体が欠損していれば MISSING を返す。"""
    per_size = data.get(run_no, {})
    entry = per_size.get(size)
    if entry is None:
        return MISSING
    return entry.get(key, MISSING)


def collect_series(
    data: dict[int, dict[int, dict[str, object]]], run_numbers: list[int], size: int, key: str
) -> tuple[list[float], int, int, int]:
    """(数値のみのリスト, skipped 件数, missing 件数, invalid 件数) を返す。"""
    numeric: list[float] = []
    n_skipped = 0
    n_missing = 0
    n_invalid = 0
    for run_no in run_numbers:
        v = value_at(data, run_no, size, key)
        if isinstance(v, float):
            numeric.append(v)
        elif v == SKIPPED:
            n_skipped += 1
        elif v == INVALID:
            n_invalid += 1
        else:
            n_missing += 1
    return numeric, n_skipped, n_missing, n_invalid


def sign_consistency(numeric: list[float]) -> str:
    """RULE.txt 補足: 中央値と反対側の run が 2 件以上あれば「5/5 一貫ではない」。

    5 件未満（欠損・skipped 混在）では判定不能として明示する。
    """
    if len(numeric) < REQUIRED_RUNS:
        return "N/A（欠損あり）"
    median = statistics.median(numeric)
    median_side_ge1 = median >= 1.00
    mismatches = sum(1 for v in numeric if (v >= 1.00) != median_side_ge1)
    if mismatches >= 2:
        return f"5/5 一貫ではない（不一致{mismatches}件）"
    if mismatches == 1:
        return "不一致1件（規則上は問題なし）"
    return "5/5 一貫"


def judge_arm(
    data: dict[int, dict[int, dict[str, object]]],
    run_numbers: list[int],
    arm: str,
) -> tuple[str, dict[int, tuple[list[float], int, int, int]]]:
    """候補腕 1 つの Gate C 判定（RULE.txt 逐語）を行う。

    戻り値は (判定文字列, size -> (numeric値リスト, skipped件数, missing件数, invalid件数))。
    """
    ratio_key = f"tma_{arm}_over_pipeline3_gpu_only"
    per_size: dict[int, tuple[list[float], int, int, int]] = {}
    for size in SIZES:
        per_size[size] = collect_series(data, run_numbers, size, ratio_key)

    if len(run_numbers) < REQUIRED_RUNS:
        return f"undetermined（run 数不足: {len(run_numbers)}/{REQUIRED_RUNS}）", per_size

    incomplete_sizes = [
        size
        for size, (numeric, n_skipped, n_missing, n_invalid) in per_size.items()
        if len(numeric) < REQUIRED_RUNS
    ]
    if incomplete_sizes:
        return (
            "undetermined（主判定列に skipped／欠損／不正値: "
            + ", ".join(f"size={s}" for s in sorted(incomplete_sizes))
            + "）",
            per_size,
        )

    medians = {size: statistics.median(per_size[size][0]) for size in SIZES}

    reject_sizes = sorted(size for size, med in medians.items() if med < 1.00)
    if reject_sizes:
        return "REJECT（後退 size: " + ", ".join(str(s) for s in reject_sizes) + "）", per_size

    adopt_trigger_sizes = sorted(
        size for size in ADOPT_TRIGGER_SIZES if medians[size] >= 1.05
    )
    if adopt_trigger_sizes and all(medians[size] >= 1.00 for size in SIZES):
        return (
            "ADOPT候補（>=1.05 達成 size: " + ", ".join(str(s) for s in adopt_trigger_sizes) + "）",
            per_size,
        )

    return "undetermined（全 size 1.00〜1.05 未満）", per_size


def format_value(v: object) -> str:
    if isinstance(v, float):
        return f"{v:.6f}"
    if v == SKIPPED:
        return "skipped"
    if v == MISSING:
        return "欠損"
    if v == INVALID:
        return "不正値"
    return str(v)


def build_report(
    log_dir: Path,
    b64_hypothesis: str,
    warnings: list[str],
) -> str:
    runs = discover_runs(log_dir)
    run_numbers = sorted(runs.keys())
    data: dict[int, dict[int, dict[str, object]]] = {}
    for run_no, path in runs.items():
        data[run_no] = parse_gate_c_log(path, warnings)

    lines: list[str] = []
    lines.append("# TMA Stage 1 ゲート C 集計（イシュー #1975／#1976）\n")

    if not runs:
        lines.append(
            "verdict=undetermined（計測未実施）: `gateC_run<N>.log`（N=1〜5）が"
            "見つかりません。RULE.txt の順序（ゲート A → ゲート B → ゲート C 5 回起動）"
            "に従って計測してください。\n"
        )
        lines.append(f"ゲート C 結果: none=undetermined（計測未実施）, b64=undetermined（計測未実施）\n")
        return "\n".join(lines)

    lines.append(
        f"対象 run: {len(run_numbers)} 件（run 番号 "
        f"{', '.join(str(r) for r in run_numbers)}。`gateC_run<N>.log` 完全一致のみを走査対象とした）\n"
    )

    # 専有ゲート（load_gate.log）の併記。任意入力のため不在でも致命的にしない。
    load_gate_rows = parse_load_gate(log_dir / "load_gate.log")
    if load_gate_rows:
        lines.append("## 専有ゲート（load_gate.log）\n")
        lines.append("| run | load1 | gpu_util | 判定 |")
        lines.append("|---|---:|---:|---|")
        for row in sorted(load_gate_rows, key=lambda r: r["run"]):
            lines.append(f"| {row['run']} | {row['load1']} | {row['gpu_util']} | {row['status']} |")
        lines.append("")
    else:
        lines.append("専有ゲート: `load_gate.log` が見つからないため記録なし（任意入力）。\n")

    verdicts: dict[str, str] = {}
    for arm in ARMS:
        verdict, per_size = judge_arm(data, run_numbers, arm)
        verdicts[arm] = verdict

        ratio_key = f"tma_{arm}_over_pipeline3_gpu_only"
        ref_ratio_key = f"tma_{arm}_over_pipeline128x64_gpu_only"
        abs_key = f"tma_{arm}_gpu_only_tflops"

        lines.append(f"## 腕: tma_{arm}\n")
        lines.append(
            "| size | " + " | ".join(f"run{r}" for r in run_numbers)
            + " | 中央値 | 符号一貫 | skipped | 欠損 | 不正値 | "
            + "参考中央値(対128x64) | tma_abs中央値 | pipeline3_abs中央値 | pipeline128x64_abs中央値 |"
        )
        lines.append(
            "|---" * (1 + len(run_numbers)) + "|---:|---|---:|---:|---:|---:|---:|---:|---:|"
        )
        for size in SIZES:
            numeric, n_skipped, n_missing, n_invalid = per_size[size]
            row_vals = [value_at(data, r, size, ratio_key) for r in run_numbers]
            median_str = f"{statistics.median(numeric):.6f}" if numeric else "-"
            consistency = sign_consistency(numeric)

            ref_numeric, _, _, _ = collect_series(data, run_numbers, size, ref_ratio_key)
            ref_median_str = f"{statistics.median(ref_numeric):.6f}" if ref_numeric else "-"

            abs_numeric, _, _, _ = collect_series(data, run_numbers, size, abs_key)
            abs_median_str = f"{statistics.median(abs_numeric):.4f}" if abs_numeric else "-"

            p3_numeric, _, _, _ = collect_series(data, run_numbers, size, "pipeline3_gpu_only_tflops")
            p3_median_str = f"{statistics.median(p3_numeric):.4f}" if p3_numeric else "-"

            p128_numeric, _, _, _ = collect_series(
                data, run_numbers, size, "pipeline128x64_gpu_only_tflops"
            )
            p128_median_str = f"{statistics.median(p128_numeric):.4f}" if p128_numeric else "-"

            lines.append(
                f"| {size} | "
                + " | ".join(format_value(v) for v in row_vals)
                + f" | {median_str} | {consistency} | {n_skipped} | {n_missing} | {n_invalid} | "
                + f"{ref_median_str} | {abs_median_str} | {p3_median_str} | {p128_median_str} |"
            )

        lines.append("")
        if arm == "b64" and b64_hypothesis == "gap":
            lines.append(
                f"腕 tma_b64 の Gate C 判定（機械算出。候補腕からは除外）: {verdict}\n"
                "**注記: ゲート A で B64 swizzle 仮説が不成立（`--b64-hypothesis gap`）のため、"
                "本腕は候補腕ではなく参考記録として扱う。**\n"
            )
        else:
            lines.append(f"腕 tma_{arm} の Gate C 判定: {verdict}\n")

    if warnings:
        lines.append("## 警告（想定外入力）\n")
        for w in warnings:
            lines.append(f"- {w}")
        lines.append("")

    none_verdict = verdicts["none"]
    if b64_hypothesis == "gap":
        b64_summary = f"参考（仮説不成立。機械算出値: {verdicts['b64']}）"
    else:
        b64_summary = verdicts["b64"]
    lines.append(f"ゲート C 結果: none={none_verdict}, b64={b64_summary}\n")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--log-dir",
        default=str(Path(__file__).resolve().parent),
        help="gateC_run<N>.log・load_gate.log を探すディレクトリ（既定: 本スクリプトと同じディレクトリ）",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="出力先ファイル（省略時は標準出力へ Markdown をそのまま出す。指定時はファイルへ書き込み『wrote <path>』のみ標準出力に出す）",
    )
    parser.add_argument(
        "--b64-hypothesis",
        choices=["ok", "gap"],
        default=None,
        help="ゲート A の B64 swizzle 仮説成立可否（RULE.txt 16〜18 行・36 行）。"
        "'ok' なら b64 も候補腕として扱い、'gap' なら参考記録に格下げする（必須引数）。",
    )
    parser.add_argument("--self-test", action="store_true", help="ユニットテストのみ実行して終了する")
    args = parser.parse_args()

    if args.self_test:
        return run_self_test()

    if args.b64_hypothesis is None:
        parser.error("--b64-hypothesis は必須です（'ok' または 'gap' を指定してください）")

    log_dir = Path(args.log_dir)
    warnings: list[str] = []
    report = build_report(log_dir, args.b64_hypothesis, warnings)

    if args.out:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = log_dir / out_path
        out_path.write_text(report, encoding="utf-8")
        print(f"wrote {out_path}")
    else:
        print(report)
    return 0


def run_self_test() -> int:
    """python3 標準ライブラリのみのユニットテスト（実機ログなしで実行可能）。"""
    import shutil
    import tempfile

    # --- discover_runs: 完全一致のみ・退避ログ／範囲外 run 番号を除外 ---
    work = Path(tempfile.mkdtemp(prefix="tma_stage1_discover_"))
    try:
        (work / "gateC_run1.log").write_text("", encoding="utf-8")
        (work / "gateC_run3.log").write_text("", encoding="utf-8")
        (work / "gateC_run1.log.bak").write_text("stale", encoding="utf-8")
        (work / "gateC_run9.log").write_text("", encoding="utf-8")
        runs = discover_runs(work)
        assert set(runs.keys()) == {1, 3}, runs
    finally:
        shutil.rmtree(work)

    # --- parse_gate_c_log: |無視・非 size 行無視・skipped/不正値/欠損 ---
    sample = (
        "   Compiling fandhe-ai-backend-cuda v0.0.0\n"
        "     Running `target/release/examples/gemm_tiled_pipeline_bench`\n"
        "size=256 tiled_f32_tflops=1.0 | pipeline3_gpu_only_tflops=2.0 "
        "pipeline128x64_gpu_only_tflops=2.5 | tma_none_gpu_only_tflops=2.2 "
        "tma_none_over_pipeline3_gpu_only=1.10 tma_none_over_pipeline128x64_gpu_only=0.88 | "
        "tma_b64_gpu_only_tflops=skipped tma_b64_over_pipeline3_gpu_only=skipped "
        "tma_b64_over_pipeline128x64_gpu_only=skipped\n"
        "size=512 pipeline3_gpu_only_tflops=garbage tma_none_over_pipeline3_gpu_only=1.02\n"
    )
    tmp_dir = Path(tempfile.mkdtemp(prefix="tma_stage1_parse_"))
    try:
        p = tmp_dir / "gateC_run1.log"
        p.write_text(sample, encoding="utf-8")
        w: list[str] = []
        parsed = parse_gate_c_log(p, w)
        assert parsed[256]["pipeline3_gpu_only_tflops"] == 2.0
        assert parsed[256]["tma_none_over_pipeline3_gpu_only"] == 1.10
        assert parsed[256]["tma_b64_over_pipeline3_gpu_only"] == SKIPPED
        assert parsed[512]["pipeline3_gpu_only_tflops"] == INVALID
        assert any("garbage" in msg for msg in w)
    finally:
        shutil.rmtree(tmp_dir)

    # --- judge_arm: ADOPT 候補（5 size 全て >=1.00・N>=1024 のいずれかで >=1.05） ---
    def make_run(size_ratios: dict[int, float]) -> dict[int, dict[str, object]]:
        return {
            size: {
                "pipeline3_gpu_only_tflops": 10.0,
                "pipeline128x64_gpu_only_tflops": 10.5,
                "tma_none_gpu_only_tflops": 10.0 * ratio,
                "tma_none_over_pipeline3_gpu_only": ratio,
                "tma_none_over_pipeline128x64_gpu_only": ratio * 0.95,
            }
            for size, ratio in size_ratios.items()
        }

    adopt_ratios = {256: 1.01, 512: 1.02, 1024: 1.06, 2048: 1.07, 4096: 1.08}
    data_adopt = {r: make_run(adopt_ratios) for r in range(1, 6)}
    verdict, per_size = judge_arm(data_adopt, [1, 2, 3, 4, 5], "none")
    assert verdict.startswith("ADOPT候補"), verdict

    # --- judge_arm: REJECT（いずれかの size で中央値 <1.00） ---
    reject_ratios = {256: 1.01, 512: 0.98, 1024: 1.06, 2048: 1.07, 4096: 1.08}
    data_reject = {r: make_run(reject_ratios) for r in range(1, 6)}
    verdict, _ = judge_arm(data_reject, [1, 2, 3, 4, 5], "none")
    assert verdict.startswith("REJECT"), verdict
    assert "512" in verdict, verdict

    # --- judge_arm: undetermined（全 size 1.00〜1.05 未満） ---
    undet_ratios = {256: 1.00, 512: 1.01, 1024: 1.02, 2048: 1.03, 4096: 1.04}
    data_undet = {r: make_run(undet_ratios) for r in range(1, 6)}
    verdict, _ = judge_arm(data_undet, [1, 2, 3, 4, 5], "none")
    assert verdict.startswith("undetermined（全 size"), verdict

    # --- judge_arm: run 数不足（5 未満） ---
    data_short = {r: make_run(adopt_ratios) for r in range(1, 4)}
    verdict, _ = judge_arm(data_short, [1, 2, 3], "none")
    assert verdict.startswith("undetermined（run 数不足"), verdict

    # --- judge_arm: skipped が 1 件でもあれば undetermined（ADOPT を満たす数値でも） ---
    data_skip = {r: make_run(adopt_ratios) for r in range(1, 6)}
    data_skip[3][2048]["tma_none_over_pipeline3_gpu_only"] = SKIPPED
    verdict, _ = judge_arm(data_skip, [1, 2, 3, 4, 5], "none")
    assert verdict.startswith("undetermined（主判定列に"), verdict
    assert "size=2048" in verdict, verdict

    # --- judge_arm: 欠損（run に size 行自体が無い）も skipped と同様に undetermined 化 ---
    data_missing = {r: make_run(adopt_ratios) for r in range(1, 6)}
    del data_missing[4][1024]
    verdict, _ = judge_arm(data_missing, [1, 2, 3, 4, 5], "none")
    assert verdict.startswith("undetermined（主判定列に"), verdict
    assert "size=1024" in verdict, verdict

    # --- sign_consistency: 2 件以上の不一致で「5/5 一貫ではない」 ---
    assert sign_consistency([1.10, 1.11, 0.99, 1.12, 0.98]) == "5/5 一貫ではない（不一致2件）"
    assert sign_consistency([1.10, 1.11, 0.99, 1.12, 1.13]) == "不一致1件（規則上は問題なし）"
    assert sign_consistency([1.10, 1.11, 1.09, 1.12, 1.13]) == "5/5 一貫"
    assert sign_consistency([1.10, 1.11]) == "N/A（欠損あり）"

    # --- build_report: --b64-hypothesis gap で b64 は参考記録に格下げされる ---
    work2 = Path(tempfile.mkdtemp(prefix="tma_stage1_report_"))
    try:
        for run_no in range(1, 6):
            body = []
            for size, ratio in adopt_ratios.items():
                body.append(
                    f"size={size} pipeline3_gpu_only_tflops=10.0 pipeline128x64_gpu_only_tflops=10.5 "
                    f"tma_none_gpu_only_tflops={10.0*ratio:.4f} "
                    f"tma_none_over_pipeline3_gpu_only={ratio} "
                    f"tma_none_over_pipeline128x64_gpu_only={ratio*0.95:.4f} "
                    f"tma_b64_gpu_only_tflops=skipped tma_b64_over_pipeline3_gpu_only=skipped "
                    f"tma_b64_over_pipeline128x64_gpu_only=skipped"
                )
            (work2 / f"gateC_run{run_no}.log").write_text("\n".join(body) + "\n", encoding="utf-8")
        warnings: list[str] = []
        report = build_report(work2, "gap", warnings)
        assert "ADOPT候補" in report
        assert "参考（仮説不成立" in report
        assert "ゲート C 結果: none=" in report
        assert "b64=参考（仮説不成立" in report
    finally:
        shutil.rmtree(work2)

    # --- main(): 未計測時（ログなし）は verdict=undetermined（計測未実施）を出す ---
    work3 = Path(tempfile.mkdtemp(prefix="tma_stage1_empty_"))
    try:
        report = build_report(work3, "ok", [])
        assert "計測未実施" in report
    finally:
        shutil.rmtree(work3)

    print("self-test OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
