#!/usr/bin/env python3
"""イシュー #1308: `gemm_splitk_shapes_bench` 実測結果（`run{1..5}.log`）の集計。

python3 標準ライブラリのみで完結する（追加依存なし。`docs/perf/logs/
cpu-gemm-kc-sweep-1315/` 等の先例と同型の集計方式）。

各 run ログから `target=(...) control=(...) target_tflops=... control_tflops=...
target_over_control=... spread_target=... spread_control=...
target_actual_groups=... control_actual_groups=...` 行を抽出し、12 組
（TARGET_MN × TARGET_K）それぞれについて:

- 5 run 生値（target_tflops・control_tflops・target_over_control・spread_*）
- run 別 `target_over_control` の中央値と 5/5 run が `< 0.7` かどうか（符号
  一貫性。`docs/perf/metal-gemm-splitk-shapes.md` §3.3 の事前登録判定規則）
- 両群中央値の比（target 5 run 中央値 ÷ control 5 run 中央値。参考指標。
  PR #1389/#1392 codex-review 指摘の踏襲で「run 内対応比」の中央値と明確に
  区別する）
- §5 判定基準（条件1・条件2）の該当有無

を Markdown 表として `aggregate.md` へ出力する。

**採否判定前の検証（codex-review 指摘対応）**: 各 run ログは、期待する 12 組
（`EXPECTED_KEYS` = `gemm_splitk_shapes_bench.rs::TARGET_MN` × `TARGET_K` の
ミラー）の測定行をそれぞれちょうど 1 件ずつ含んでいることを `parse_log` が
自己完結で検証する。run 間のキー順一致だけを見る旧チェックでは、全 run で
同じ形状が揃って欠落していたり、実測行が 1 件も無い場合（例: 非 macOS 環境
で `main` が解析値のみを出力するケース）でも `keys=[]` のまま集計ループが
黙って 0 走査で完了し「不採用（現状維持）」を誤って機械判定してしまう
（実測なしを不採用と取り違える）。欠落・重複を検出した時点で `ValueError`
を送出し、集計を中止する（fail-closed。どの run・どの形状が問題かをエラー
メッセージに含める）。

`--self-test` で行パース・中央値算出・上記の欠落／重複検出の回帰確認を行う
（実機ログなしで実行可能）。
"""

from __future__ import annotations

import re
import statistics
import sys
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent

LINE_RE = re.compile(
    r"target=\((?P<tm>\d+),(?P<tn>\d+),(?P<tk>\d+)\) "
    r"control=\((?P<cm>\d+),(?P<cn>\d+),(?P<ck>\d+)\) "
    r"target_tflops=(?P<target_tflops>[\d.]+) "
    r"control_tflops=(?P<control_tflops>[\d.]+) "
    r"target_over_control=(?P<target_over_control>[\d.]+) "
    r"spread_target=(?P<spread_target>[\d.]+) "
    r"spread_control=(?P<spread_control>[\d.]+) "
    r"target_actual_groups=(?P<target_actual_groups>\d+) "
    r"control_actual_groups=(?P<control_actual_groups>\d+)"
)

# §3.3 事前登録: (32,32,*)・(64,64,*)・(128,128,*) は analytics 側
# actual_groups（16／4／16）が実機コア数 40 未満のため条件 2 該当。
# (256,256,*) は analytics actual_groups=64 が 40 以上のため条件 2 非該当
# （target_actual_groups 列そのものから機械判定する。resolved 値ではなく
# analytics 値を使うのは §3.3 の事前登録が analytics::analyze 基準で書かれた
# ためであり、本集計はその定義をそのまま踏襲する）。
CONDITION2_GROUPS_THRESHOLD = 40
CONDITION1_RATIO_THRESHOLD = 0.7
STABILITY_SPREAD_GATE = 0.05

# `gemm_splitk_shapes_bench.rs::TARGET_MN` × `TARGET_K` のミラー（12 組）。
# import ではなく値の複製: 当該 example は Rust クレート（Cargo ビルド必須）
# であり、python3 標準ライブラリのみで完結する本集計スクリプトの設計
# （モジュール冒頭コメント）から直接 import する経路がないため。値が乖離
# した場合は `--self-test` の `expected_keys_matches_bench_rs_literal`相当の
# 固定値検査、または実 run ログの欠落検出（`parse_log` の ValueError）で
# 顕在化する。
EXPECTED_KEYS: frozenset[tuple[int, int, int]] = frozenset(
    (mn, mn, k) for mn in (32, 64, 128, 256) for k in (2048, 4096, 8192)
)


def parse_log(path: Path) -> dict[tuple[int, int, int], dict]:
    """1 run ログから 12 組の測定行を抽出する。

    期待する 12 組（`EXPECTED_KEYS`）がそれぞれちょうど 1 件ずつ存在する
    ことを検証し、欠落・重複があれば `ValueError` で集計を中止する
    （fail-closed。モジュール docstring「採否判定前の検証」節参照）。
    測定行が 1 件も無い場合（非 macOS 実行等）も「12 件すべて欠落」として
    同じ経路で検出され、黙って 0 走査のまま完了することはない。
    """
    rows: dict[tuple[int, int, int], dict] = {}
    duplicates: list[tuple[int, int, int]] = []
    unexpected: list[tuple[int, int, int]] = []
    text = path.read_text()
    for m in LINE_RE.finditer(text):
        key = (int(m["tm"]), int(m["tn"]), int(m["tk"]))
        if key in rows:
            duplicates.append(key)
        if key not in EXPECTED_KEYS:
            unexpected.append(key)
        rows[key] = {
            "control": (int(m["cm"]), int(m["cn"]), int(m["ck"])),
            "target_tflops": float(m["target_tflops"]),
            "control_tflops": float(m["control_tflops"]),
            "target_over_control": float(m["target_over_control"]),
            "spread_target": float(m["spread_target"]),
            "spread_control": float(m["spread_control"]),
            "target_actual_groups": int(m["target_actual_groups"]),
            "control_actual_groups": int(m["control_actual_groups"]),
        }

    missing = sorted(EXPECTED_KEYS - rows.keys())
    if missing or duplicates or unexpected:
        parts = [f"{path.name}: 期待する 12 形状の測定行検証に失敗した"]
        if missing:
            parts.append(f"欠落 {len(missing)} 件: {missing}")
        if duplicates:
            parts.append(f"重複 {len(duplicates)} 件: {duplicates}")
        if unexpected:
            parts.append(f"想定外の組 {len(unexpected)} 件: {unexpected}")
        raise ValueError("。".join(parts))

    return rows


def aggregate(run_logs: list[Path]) -> str:
    per_run = [parse_log(p) for p in run_logs]
    # 各 run は parse_log で EXPECTED_KEYS の 12 組ちょうど 1 件ずつを
    # 持つことが既に保証されているため、run 間で走査順序（辞書挿入順＝
    # ログ内出現順）が食い違っていないかのみを追加確認する。
    keys = list(per_run[0].keys())
    for i, rows in enumerate(per_run[1:], start=2):
        if list(rows.keys()) != keys:
            raise ValueError(f"run{i}.log の組順序が run1.log と一致しない")

    lines = []
    lines.append("| target (M,N,K) | control (S,S,S) | target_over_control (5 run) | median | all<0.7 | spread max(t/c) | actual_groups(t) | 条件1 | 条件2 | 判定対象 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")

    qualifying = 0
    candidate = 0
    for key in keys:
        target_tflops = [r[key]["target_tflops"] for r in per_run]
        control_tflops = [r[key]["control_tflops"] for r in per_run]
        ratios = [r[key]["target_over_control"] for r in per_run]
        spreads_t = [r[key]["spread_target"] for r in per_run]
        spreads_c = [r[key]["spread_control"] for r in per_run]
        control = per_run[0][key]["control"]
        target_groups = per_run[0][key]["target_actual_groups"]

        median_ratio = statistics.median(ratios)
        all_below = all(r < CONDITION1_RATIO_THRESHOLD for r in ratios)
        spread_ok = all(s <= STABILITY_SPREAD_GATE for s in spreads_t + spreads_c)
        # 条件1: 中央値 <0.7 かつ 5/5 run 一貫（run 間で閾値を跨がない）
        cond1 = median_ratio < CONDITION1_RATIO_THRESHOLD and all_below
        cond2 = target_groups < CONDITION2_GROUPS_THRESHOLD

        is_candidate = cond2
        if is_candidate:
            candidate += 1
        if is_candidate and cond1:
            qualifying += 1

        alt_ratio = statistics.median(target_tflops) / statistics.median(control_tflops)
        ratios_str = ",".join(f"{r:.4f}" for r in ratios)

        lines.append(
            f"| {key} | {control} | {ratios_str} | {median_ratio:.4f} (alt={alt_ratio:.4f}) "
            f"| {'yes' if all_below else 'no'} | {'ok' if spread_ok else 'EXCEEDED'} "
            f"| {target_groups} | {'○' if cond1 else '×'} | {'○' if cond2 else '×'} "
            f"| {'candidate' if is_candidate else '-'} |"
        )

    lines.append("")
    total_candidates = candidate
    lines.append(
        f"条件2 該当（並列度不足の解析裏付け）候補: {total_candidates} / 12 点。"
        f"うち条件1（劣化率 <0.7・5/5 run 一貫）も成立: {qualifying} / {total_candidates}"
        f"（{'≥7' if qualifying >= 7 else '<7'}）。"
    )
    verdict = "採用検討推奨" if qualifying >= 7 else "不採用（現状維持）"
    lines.append(f"§3.3 事前登録規則による機械判定: **{verdict}**")

    return "\n".join(lines)


def _line_for(m: int, n: int, k: int, control: tuple[int, int, int]) -> str:
    """self-test 用の 1 形状分の測定行を組み立てる（`LINE_RE` の形式に合わせる）。"""
    cm, cn, ck = control
    return (
        f"target=({m},{n},{k}) control=({cm},{cn},{ck}) target_tflops=0.1000 "
        "control_tflops=0.2000 target_over_control=0.5000 spread_target=0.0100 "
        "spread_control=0.0100 target_actual_groups=4 control_actual_groups=64\n"
    )


def _all_12_lines() -> str:
    return "".join(
        _line_for(mn, mn, k, (256, 256, 256))
        for mn in (32, 64, 128, 256)
        for k in (2048, 4096, 8192)
    )


def self_test() -> None:
    sample = (
        "target=(64,64,4096) control=(256,256,256) target_tflops=0.0909 "
        "control_tflops=0.1998 target_over_control=0.4552 spread_target=0.7399 "
        "spread_control=0.4255 target_actual_groups=4 control_actual_groups=64\n"
    )
    tmp = LOG_DIR / "_self_test_sample.log"

    def write_and_parse(text: str) -> dict[tuple[int, int, int], dict]:
        tmp.write_text(text)
        return parse_log(tmp)

    try:
        # 単一行のパース自体は従来どおり検証できる（12 組検証はこの後の
        # ケースで別途確認する）が、EXPECTED_KEYS の 12 組検証が新設された
        # ため単独では ValueError になる。行パース結果の中身の検証は
        # 12 組が揃った `_all_12_lines()` ベースの正常系ケースで行う。
        try:
            write_and_parse(sample)
        except ValueError:
            pass
        else:
            raise AssertionError("1 行のみのログは 12 組検証で ValueError になるはず")

        # 正常系: 12 組がちょうど 1 件ずつ揃っていればパースが通る。
        rows = write_and_parse(_all_12_lines())
        key = (64, 64, 4096)
        assert key in rows, "行パース失敗"
        assert rows[key]["control"] == (256, 256, 256)
        assert abs(rows[key]["target_tflops"] - 0.1000) < 1e-9
        assert rows[key]["target_actual_groups"] == 4
        assert rows[key]["control_actual_groups"] == 64
        assert len(rows) == 12, "12 組ちょうどのはず"

        # 異常系1: 測定行が 1 件も無い（非 macOS 環境で解析値のみ出力する
        # ケースの再現）。旧実装は keys=[] のまま黙って完走し「不採用」を
        # 誤って機械判定していた（codex-review 指摘対応の回帰確認）。
        try:
            write_and_parse("")
        except ValueError as e:
            assert "欠落 12 件" in str(e), f"空ログのエラーメッセージが想定外: {e}"
        else:
            raise AssertionError("空ログは ValueError になるはず")

        # 異常系2: 12 組中 1 組欠落（(256,256,8192) を除いた 11 行）。
        missing_one = "".join(
            _line_for(mn, mn, k, (256, 256, 256))
            for mn in (32, 64, 128, 256)
            for k in (2048, 4096, 8192)
            if not (mn == 256 and k == 8192)
        )
        try:
            write_and_parse(missing_one)
        except ValueError as e:
            assert "欠落 1 件" in str(e), f"欠落 1 件のエラーメッセージが想定外: {e}"
        else:
            raise AssertionError("11 組のログは ValueError になるはず")

        # 異常系3: 同一形状が重複（(32,32,2048) を 2 回）。
        duplicated = _line_for(32, 32, 2048, (256, 256, 256)) + _all_12_lines()
        try:
            write_and_parse(duplicated)
        except ValueError as e:
            assert "重複" in str(e), f"重複のエラーメッセージが想定外: {e}"
        else:
            raise AssertionError("重複を含むログは ValueError になるはず")

        print("self-test OK")
    finally:
        tmp.unlink(missing_ok=True)


def main() -> None:
    if "--self-test" in sys.argv:
        self_test()
        return

    run_logs = sorted(LOG_DIR.glob("run[1-5].log"))
    if len(run_logs) != 5:
        print(
            f"run1.log〜run5.log が揃っていない（見つかった数: {len(run_logs)}）",
            file=sys.stderr,
        )
        sys.exit(1)

    md = aggregate(run_logs)
    out = LOG_DIR / "aggregate.md"
    out.write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
