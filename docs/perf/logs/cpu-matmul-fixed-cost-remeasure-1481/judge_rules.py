#!/usr/bin/env python3
"""イシュー #1481: 出力並列ゼロ埋め（#1299）on/off 独立再計測の機械判定。

`docs/perf/cpu-gemm-candle-gate-remeasurement.md` §20.1（規則 1〜3・5）・
§20.1a（規則 4 改定版）を事前登録規則としてハードコードし、計測後は
一切変更しない（PR #1448 codex-review 指摘の是正がこのスクリプトの
存在理由）。入力は本ディレクトリの集計済み Markdown／JSONL から手で
拾った数値ではなく、`--layer-a-md`（`compare_gemm_ab.py --device cpu`
出力）・`--layer-b-md`（`aggregate_layer_b.py` 出力）・`--gate-md`
（`compare_gemm_gate.py --device cpu` 出力。off/on × 実機で計 4 本）
から正規表現で機械抽出する。

PR #1501 codex-review 指摘の是正（2026-09-09）: 初版は規則 1（checksum
完全一致）をテキスト出力のみに留め最終 verdict へ反映しておらず、
規則 5（M4 Max N=2048 後退）も表示のみで `all_ok` に一切寄与しなかった
（§20.1 の場合分け (a)〜(c) が実装されず、常に「全規則充足なら ADOPT・
不成立なら REJECT」の二値判定に単純化されていた）。本版は
`compare_gemm_ab.py` 出力の `checksum` 列を機械抽出して規則 1 を
判定の必須前提とし（不一致・欠落セルがあれば `judge()` は "UNDETERMINED"
を返し他規則の成否によらず判定不能とする）、規則 5 の発火有無を
§20.1 の場合分けへ折り込んで "ADOPT_UNCONDITIONAL"／
"ADOPT_LINUX_ONLY"／"REJECT"／"UNDETERMINED" の 4 値判定にした。

使い方:
  python3 judge_rules.py \
    --layer-a-md compare_gemm_ab-dgx.md --layer-a-md compare_gemm_ab-m4max.md \
    --layer-b-md aggregate-dgx.md --layer-b-md aggregate-m4max.md \
    --gate-md dgx:off:compare_gemm_gate-dgx-off.md \
    --gate-md dgx:on:compare_gemm_gate-dgx-on.md \
    --gate-md m4max:off:compare_gemm_gate-m4max-off.md \
    --gate-md m4max:on:compare_gemm_gate-m4max-on.md \
    > judge.md

自己検証: `python3 judge_rules.py --self-test`（固定サンプル文字列に対する
パーサ・判定ロジックの単体検証。標準ライブラリのみ）。
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass

# ---- 事前登録した数値閾値（計測前に固定。以後変更しない） ----
RULE234_THRESHOLD = 1.05  # 規則 2 の Layer A・規則 3・規則 5 の on/off 比上限
RULE2_OPS_GEMM_THRESHOLD = 1.00  # 規則 2 の ops_gemm on/off 比上限（厳密。緩めない）
RULE4_MIN_RATIO = 1.0 / 1.05  # 規則 4（改定版）: on/off 比の下限 (~0.952380...)
CHECKSUM_OK_VALUE = "完全一致"  # 規則 1 が要求する checksum 列の合格値

# N=512/1024/2048 の fresh/reuse 全セル。規則 1 の網羅性確認（欠落検出）に使う。
EXPECTED_LAYER_A_CELLS = tuple(
    (n, mode) for n in (512, 1024, 2048) for mode in ("fresh", "reuse")
)

GATE_ROW_RE = re.compile(
    r"^\|\s*(?P<n>\d+)\s*\|[^|]*\|[^|]*\|\s*(?P<ratio>[0-9.]+)\s*\|"
)
# size/mode | before median | after median | after/before | checksum | 判定
# の 6 列形式（`compare_gemm_ab.py` 実出力）。5 列目 checksum を規則 1 用に追加抽出する。
LAYER_A_ROW_RE = re.compile(
    r"^\|\s*(?P<n>\d+)/(?P<mode>fresh|reuse)\s*\|[^|]*\|[^|]*\|\s*(?P<ratio>[0-9.]+)\s*\|"
    r"\s*(?P<checksum>[^|]*?)\s*\|"
)
LAYER_B_SECTION_RE = re.compile(r"^##\s*N=(?P<n>\d+)\s*$")
LAYER_B_ROW_RE = re.compile(
    r"^\|\s*(?P<phase>[A-Za-z_]+)\s*\|[^|]*\|[^|]*\|[^|]*\|[^|]*\|\s*(?P<ratio>[0-9.nNaA]+)\s*\|"
)


@dataclass
class LayerACell:
    """`compare_gemm_ab.py` 出力 1 行分（1 セル）を保持する。

    `ratio` は規則 2〜5 の on/off 比判定に、`checksum` は規則 1 の
    完全一致判定に使う（別々の規則が同じ表の別列を参照するため、
    パース段階で両方保持し呼び出し側で規則ごとに参照する）。
    """

    ratio: float
    checksum: str


def parse_gate_md(text: str) -> dict[int, float]:
    """`compare_gemm_gate.py --device cpu` 出力 md から {N: candle/fandhe} を抽出する。"""
    out: dict[int, float] = {}
    for line in text.splitlines():
        m = GATE_ROW_RE.match(line)
        if m:
            n = int(m.group("n"))
            # ヘッダ行 `| N | ... |` を誤検出しないよう ratio が数値変換
            # できることを確認する（できなければヘッダ・区切り行として無視）。
            try:
                out[n] = float(m.group("ratio"))
            except ValueError:
                continue
    return out


def parse_layer_a_md(text: str) -> dict[tuple[int, str], LayerACell]:
    """`compare_gemm_ab.py --device cpu` 出力 md から {(N, mode): LayerACell} を抽出する。

    `ratio`（after/before 比。規則 2〜5 で使用）と `checksum`（規則 1 で
    使用）を同一行から同時に取り出す。
    """
    out: dict[tuple[int, str], LayerACell] = {}
    for line in text.splitlines():
        m = LAYER_A_ROW_RE.match(line)
        if m:
            try:
                ratio = float(m.group("ratio"))
            except ValueError:
                continue
            out[(int(m.group("n")), m.group("mode"))] = LayerACell(
                ratio=ratio, checksum=m.group("checksum").strip()
            )
    return out


def parse_layer_b_md(text: str) -> dict[tuple[int, str], float]:
    """`aggregate_layer_b.py` 出力 md から {(N, phase): on/off 比} を抽出する。"""
    out: dict[tuple[int, str], float] = {}
    cur_n: int | None = None
    for line in text.splitlines():
        m_sec = LAYER_B_SECTION_RE.match(line)
        if m_sec:
            cur_n = int(m_sec.group("n"))
            continue
        m = LAYER_B_ROW_RE.match(line)
        if m and cur_n is not None:
            phase = m.group("phase")
            raw = m.group("ratio")
            if raw.lower() == "nan":
                continue
            try:
                out[(cur_n, phase)] = float(raw)
            except ValueError:
                continue
    return out


def _check_rule1_checksum(
    layer_a: dict[str, dict[tuple[int, str], LayerACell]],
) -> tuple[bool, list[str]]:
    """規則 1: 両実機・全セル（N=512/1024/2048 × fresh/reuse）で checksum 完全一致。

    `compare_gemm_ab.py` の複合判定（`==` 列）が「完全一致」でない行、
    または期待セルが 1 つでも欠落していれば不成立とし、呼び出し元
    （`judge()`）はこれを他規則の成否によらない前提条件として扱う
    （§20.1「不一致なら判定不能」）。
    """
    lines: list[str] = []
    ok = True
    for node in ("dgx", "m4max"):
        cells = layer_a.get(node, {})
        for n, mode in EXPECTED_LAYER_A_CELLS:
            cell = cells.get((n, mode))
            if cell is None:
                lines.append(f"規則1 checksum {node} N={n}/{mode}: セル欠落 -> 不成立（判定不能）")
                ok = False
                continue
            cell_ok = cell.checksum == CHECKSUM_OK_VALUE
            lines.append(
                f"規則1 checksum {node} N={n}/{mode}: checksum={cell.checksum!r} "
                f"(== {CHECKSUM_OK_VALUE!r}) -> {'満たす' if cell_ok else '不成立（判定不能）'}"
            )
            ok = ok and cell_ok
    return ok, lines


def judge(
    layer_a: dict[str, dict[tuple[int, str], LayerACell]],
    layer_b: dict[str, dict[tuple[int, str], float]],
    gate: dict[tuple[str, str], dict[int, float]],
) -> tuple[list[str], str]:
    """規則 1〜5 を機械判定し、(明細行のリスト, verdict) を返す。

    verdict は次の 4 値（§20.1 の場合分け (a)〜(c) をそのまま実装する）:
    - "UNDETERMINED": 規則 1（checksum 完全一致）が不成立・入力欠落
      （判定不能。他規則の成否によらず優先する）
    - "ADOPT_UNCONDITIONAL": 両実機とも規則 1〜4 を満たし、かつ規則 5
      （M4 Max N=2048 後退）が発火しない（§20.1 場合分け (c)）
    - "ADOPT_LINUX_ONLY": DGX が規則 1〜4 を満たし、M4 Max が規則 5 で
      後退する（§20.1 場合分け (a)。Linux 限定 cfg gating）
    - "REJECT": 上記いずれにも該当しない（DGX が規則 2〜4 のいずれかを
      満たさない、または M4 Max が規則 5 で後退しないのに規則 3・4 の
      いずれかを満たさない等。§20.1 場合分け (b) およびそれ以外の
      未網羅ケースを安全側にすべて REJECT へ倒す）
    """
    lines: list[str] = []

    # 規則 1（前提条件）: 両実機・全セルで checksum 完全一致。
    checksum_ok, checksum_lines = _check_rule1_checksum(layer_a)
    lines.extend(checksum_lines)

    # 規則 2: DGX N=2048 決定セル（DGX 専用）。
    dgx_a = layer_a.get("dgx", {})
    dgx_b = layer_b.get("dgx", {})
    r2_layer_a_cell = dgx_a.get((2048, "reuse"))
    r2_layer_a = r2_layer_a_cell.ratio if r2_layer_a_cell is not None else None
    r2_alloc_c = dgx_b.get((2048, "alloc_c"))
    r2_ops_gemm = dgx_b.get((2048, "ops_gemm"))
    # 「alloc_c 中央値が削減され」= on/off 比 < 1.0（既存コメントの見落とし
    # を自己検証で発見。バグ修正: 当初実装は alloc_c 比を判定に使わず
    # ops_gemm・Layer A のみで r2_ok を決めていた）。
    r2_ok = (
        r2_layer_a is not None
        and r2_layer_a <= RULE234_THRESHOLD
        and r2_alloc_c is not None
        and r2_alloc_c < 1.0
        and r2_ops_gemm is not None
        and r2_ops_gemm <= RULE2_OPS_GEMM_THRESHOLD
    )
    lines.append(
        f"規則2 DGX N=2048決定セル: layer_a(reuse)={r2_layer_a} (<= {RULE234_THRESHOLD}) "
        f"alloc_c比={r2_alloc_c} ops_gemm比={r2_ops_gemm} (<= {RULE2_OPS_GEMM_THRESHOLD}) "
        f"-> {'満たす' if r2_ok else '不成立'}"
    )

    # 規則 3: 対照セル（両実機 N=512/1024 の fresh/reuse 全セル）。
    # §20.1 場合分けの (a) が「DGX が規則 1〜4 を満たす」ことのみを要求
    # するため、DGX 側・M4 Max 側を分けて判定を保持する。
    r3_dgx_ok = True
    r3_m4max_ok = True
    for node in ("dgx", "m4max"):
        a = layer_a.get(node, {})
        for n in (512, 1024):
            for mode in ("fresh", "reuse"):
                cell = a.get((n, mode))
                v = cell.ratio if cell is not None else None
                ok = v is not None and v <= RULE234_THRESHOLD
                lines.append(
                    f"規則3 対照セル {node} N={n}/{mode}: 比={v} (<= {RULE234_THRESHOLD}) "
                    f"-> {'満たす' if ok else '不成立'}"
                )
                if node == "dgx":
                    r3_dgx_ok = r3_dgx_ok and ok
                else:
                    r3_m4max_ok = r3_m4max_ok and ok

    # 規則 4（改定版）: 各セル（実機×N=512/1024/2048）の candle 比 on/off 比。
    # 規則 3 と同様に DGX 側・M4 Max 側を分けて保持する。
    r4_dgx_ok = True
    r4_m4max_ok = True
    for node in ("dgx", "m4max"):
        off = gate.get((node, "off"), {})
        on = gate.get((node, "on"), {})
        for n in (512, 1024, 2048):
            off_v = off.get(n)
            on_v = on.get(n)
            if off_v is None or on_v is None or off_v == 0:
                ok = False
                ratio = None
            else:
                ratio = on_v / off_v
                ok = ratio >= RULE4_MIN_RATIO
            lines.append(
                f"規則4改定版 {node} N={n}: candle比 off={off_v} on={on_v} "
                f"on/off比={ratio} (>= {RULE4_MIN_RATIO:.4f}) -> {'満たす' if ok else '不成立'}"
            )
            if node == "dgx":
                r4_dgx_ok = r4_dgx_ok and ok
            else:
                r4_m4max_ok = r4_m4max_ok and ok

    # 規則 5: M4 Max N=2048 の後退判定（Layer A on/off 比 > 1.05 なら
    # 規則発火＝§20.1 場合分け (a) の Linux 限定化条件）。本判定は
    # run 別 JSONL の符号一貫性までは見ず、集計後の中央値比のみで判定
    # する（5 run 中央値そのものが規則 2〜3 と同じ Layer A 表由来のため、
    # 符号一貫性は生ログ側で別途確認し env_info.txt に記録する）。
    m4_2048_reuse_cell = layer_a.get("m4max", {}).get((2048, "reuse"))
    m4_2048_reuse = m4_2048_reuse_cell.ratio if m4_2048_reuse_cell is not None else None
    r5_regression = m4_2048_reuse is not None and m4_2048_reuse > RULE234_THRESHOLD
    lines.append(
        f"規則5 M4Max N=2048: layer_a(reuse)={m4_2048_reuse} "
        f"-> {'後退あり（5run符号一貫は生ログで別途確認）' if r5_regression else '後退なし（発火せず）'}"
    )

    # ---- §20.1 場合分け (a)〜(c) を最終 verdict へ折り込む ----
    dgx_rules_1_4_ok = checksum_ok and r2_ok and r3_dgx_ok and r4_dgx_ok
    m4max_rules_1_4_ok = checksum_ok and r3_m4max_ok and r4_m4max_ok

    if not checksum_ok:
        # 規則 1 は前提条件。不成立なら他規則の成否によらず判定不能。
        verdict = "UNDETERMINED"
    elif dgx_rules_1_4_ok and m4max_rules_1_4_ok and not r5_regression:
        # (c) 両実機とも全規則を満たす → 無条件 ADOPT。
        verdict = "ADOPT_UNCONDITIONAL"
    elif dgx_rules_1_4_ok and r5_regression:
        # (a) DGX が規則 1〜4 を満たし M4 Max が規則 5 で後退
        #     → ADOPT（Linux 限定 cfg gating）。M4 Max は Linux 限定化に
        #     より当該分岐を通らないため、M4 Max 側の規則 3・4 の成否は
        #     この場合分けの成立条件に含めない（§20.1 原文どおり）。
        verdict = "ADOPT_LINUX_ONLY"
    else:
        # (b) DGX で削減されない・後退する、または (a)/(c) いずれの
        # 条件にも当てはまらない未網羅ケース（例: 規則 5 は発火しない
        # が M4 Max が規則 3・4 を満たさない）は安全側に REJECT とする。
        verdict = "REJECT"

    lines.append(
        f"折り込み判定: checksum_ok={checksum_ok} dgx_rules_1_4_ok={dgx_rules_1_4_ok} "
        f"m4max_rules_1_4_ok={m4max_rules_1_4_ok} r5_regression={r5_regression} "
        f"-> verdict={verdict}"
    )

    return lines, verdict


def _self_test() -> int:
    sample_gate_off = """
| N | fandhe-ai reuse median (min–max, n) | candle fresh median (n) | candle/fandhe | GFLOP/s | 判定 | fandhe-ai fresh median（参考。n） |
|---|---|---|---|---|---|---|
| 512 | 2.447 ms (2.316 ms–2.500 ms, n=5) | 1.720 ms (n=5) | 0.703 | 109.70 | 未達 | 2.522 ms (n=5) |
| 1024 | 6.919 ms | 5.386 ms | 0.778 | 310.36 | 未達 | 7.620 ms (n=5) |
| 2048 | 34.799 ms | 33.595 ms | 0.965 | 493.69 | 未達（candle 救済 2 要素） | 36.783 ms (n=5) |
"""
    parsed = parse_gate_md(sample_gate_off)
    assert parsed == {512: 0.703, 1024: 0.778, 2048: 0.965}, parsed

    # 実際の compare_gemm_ab.py 出力形式（1 実機分。size/mode | before median
    # | after median | after/before | checksum | 判定）。
    sample_layer_a = """
| size/mode | before median | after median | after/before | checksum | 判定 |
|---|---|---|---|---|---|
| 512/fresh | 2.522 ms (min 2.182 ms / max 2.571 ms) | 2.541 ms (min 2.300 ms / max 2.600 ms) | 1.0075 | 完全一致 | 非後退 |
| 512/reuse | 2.447 ms (min 2.316 ms / max 2.500 ms) | 2.176 ms (min 2.100 ms / max 2.300 ms) | 0.8892 | 完全一致 | 非後退 |
| 1024/fresh | 7.620 ms | 7.500 ms | 0.9843 | 完全一致 | 非後退 |
| 1024/reuse | 6.919 ms | 6.949 ms | 1.0043 | 完全一致 | 非後退 |
| 2048/fresh | 36.783 ms | 38.503 ms | 1.0468 | 完全一致 | 非後退 |
| 2048/reuse | 34.799 ms | 34.764 ms | 0.9990 | 完全一致 | 非後退 |
"""
    parsed_a = parse_layer_a_md(sample_layer_a)
    assert parsed_a[(2048, "reuse")].ratio == 0.9990, parsed_a
    assert parsed_a[(512, "fresh")].ratio == 1.0075, parsed_a
    assert parsed_a[(2048, "reuse")].checksum == "完全一致", parsed_a
    assert set(parsed_a.keys()) == set(EXPECTED_LAYER_A_CELLS), parsed_a

    # checksum 列が完全一致でない行を機械抽出できること（規則 1 の検出対象）。
    sample_layer_a_mismatch = """
| size/mode | before median | after median | after/before | checksum | 判定 |
|---|---|---|---|---|---|
| 512/fresh | 2.522 ms | 2.541 ms | 1.0075 | 不一致（fail=2） | 非後退 |
"""
    parsed_mismatch = parse_layer_a_md(sample_layer_a_mismatch)
    assert parsed_mismatch[(512, "fresh")].checksum == "不一致（fail=2）", parsed_mismatch

    sample_layer_b = """
## N=2048

| phase | off median (of 5 run medians, ms) | off n | on median (ms) | on n | on/off 比 |
|---|---|---|---|---|---|
| alloc_c | 3.2554 | 5 | 1.6817 | 5 | 0.5166 |
| ops_gemm | 26.8177 | 5 | 26.7420 | 5 | 0.9972 |

## N=1024

| phase | off median (of 5 run medians, ms) | off n | on median (ms) | on n | on/off 比 |
|---|---|---|---|---|---|
| alloc_c | 0.0306 | 5 | 0.0269 | 5 | 0.8791 |
| ops_gemm | 5.4276 | 5 | 5.3701 | 5 | 0.9894 |
"""
    parsed_b = parse_layer_b_md(sample_layer_b)
    assert parsed_b[(2048, "ops_gemm")] == 0.9972, parsed_b
    assert parsed_b[(2048, "alloc_c")] == 0.5166, parsed_b

    def _all_ok_layer_a() -> dict[str, dict[tuple[int, str], LayerACell]]:
        cell = LayerACell(ratio=1.00, checksum=CHECKSUM_OK_VALUE)
        return {
            "dgx": {k: cell for k in EXPECTED_LAYER_A_CELLS},
            "m4max": {k: cell for k in EXPECTED_LAYER_A_CELLS},
        }

    layer_b_ok = {"dgx": {(2048, "alloc_c"): 0.5, (2048, "ops_gemm"): 0.99}}
    gate_ok = {
        ("dgx", "off"): {512: 0.70, 1024: 0.78, 2048: 0.96},
        ("dgx", "on"): {512: 0.70, 1024: 0.78, 2048: 0.96},
        ("m4max", "off"): {512: 0.89, 1024: 0.76, 2048: 0.81},
        ("m4max", "on"): {512: 0.89, 1024: 0.76, 2048: 0.81},
    }

    # judge() の一貫性チェック 1: 全セル満たす合成サンプルで
    # ADOPT_UNCONDITIONAL（§20.1 場合分け (c)）を返すこと。
    _, verdict = judge(_all_ok_layer_a(), layer_b_ok, gate_ok)
    assert verdict == "ADOPT_UNCONDITIONAL", verdict

    # 規則 4 不成立ケース（on 腕 candle 比が off 腕比で 10% 悪化）→ REJECT。
    gate_bad = dict(gate_ok)
    gate_bad[("dgx", "on")] = {512: 0.63, 1024: 0.78, 2048: 0.96}
    _, verdict_bad = judge(_all_ok_layer_a(), layer_b_ok, gate_bad)
    assert verdict_bad == "REJECT", verdict_bad

    # checksum 不一致ケース（規則 1 不成立）→ 他が全て満たしても
    # UNDETERMINED（判定不能）を返すこと。ここが初版の欠陥（規則 1 が
    # verdict に一切反映されない）を再発させないための回帰テスト。
    layer_a_checksum_mismatch = _all_ok_layer_a()
    layer_a_checksum_mismatch["dgx"] = dict(layer_a_checksum_mismatch["dgx"])
    layer_a_checksum_mismatch["dgx"][(2048, "reuse")] = LayerACell(
        ratio=1.00, checksum="不一致（fail=3）"
    )
    _, verdict_checksum = judge(layer_a_checksum_mismatch, layer_b_ok, gate_ok)
    assert verdict_checksum == "UNDETERMINED", verdict_checksum

    # checksum セル欠落ケース（規則 1 不成立）→ UNDETERMINED。
    layer_a_missing_cell = _all_ok_layer_a()
    layer_a_missing_cell["m4max"] = {
        k: v for k, v in layer_a_missing_cell["m4max"].items() if k != (2048, "reuse")
    }
    _, verdict_missing = judge(layer_a_missing_cell, layer_b_ok, gate_ok)
    assert verdict_missing == "UNDETERMINED", verdict_missing

    # 規則 5 発火ケース（M4 Max N=2048 が 1.05 超で後退）だが DGX 側は
    # 規則 1〜4 を満たす → ADOPT_LINUX_ONLY（§20.1 場合分け (a)）。
    # これが初版のもう一つの欠陥（規則 5 が verdict に反映されず常に
    # 無条件 ADOPT/REJECT の二値になっていた）を再発させないための
    # 回帰テスト。
    layer_a_r5 = _all_ok_layer_a()
    layer_a_r5["m4max"] = dict(layer_a_r5["m4max"])
    layer_a_r5["m4max"][(2048, "reuse")] = LayerACell(
        ratio=1.20, checksum=CHECKSUM_OK_VALUE
    )
    _, verdict_r5 = judge(layer_a_r5, layer_b_ok, gate_ok)
    assert verdict_r5 == "ADOPT_LINUX_ONLY", verdict_r5

    # 規則 5 は発火しない（M4 Max N=2048 <= 1.05）が M4 Max の対照セル
    # （規則 3）が不成立 → (a)/(c) いずれにも該当しないため REJECT。
    layer_a_m4max_r3_fail = _all_ok_layer_a()
    layer_a_m4max_r3_fail["m4max"] = dict(layer_a_m4max_r3_fail["m4max"])
    layer_a_m4max_r3_fail["m4max"][(512, "fresh")] = LayerACell(
        ratio=1.20, checksum=CHECKSUM_OK_VALUE
    )
    _, verdict_m4max_r3_fail = judge(layer_a_m4max_r3_fail, layer_b_ok, gate_ok)
    assert verdict_m4max_r3_fail == "REJECT", verdict_m4max_r3_fail

    print("self-test: OK", file=sys.stderr)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer-a-md", action="append", default=[], help="node:path または path（node は '-dgx.md'/'−m4max.md' 等のファイル名から推定）")
    ap.add_argument("--layer-b-md", action="append", default=[])
    ap.add_argument("--gate-md", action="append", default=[], help="node:arm:path 形式（例 dgx:off:compare_gemm_gate-dgx-off.md）")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return _self_test()

    def infer_node(path: str) -> str:
        if "dgx" in path:
            return "dgx"
        if "m4max" in path:
            return "m4max"
        raise SystemExit(f"ERROR: node を推定できない path={path}（ファイル名に dgx/m4max を含めること）")

    layer_a: dict[str, dict[tuple[int, str], LayerACell]] = {}
    for p in args.layer_a_md:
        node = infer_node(p)
        with open(p, encoding="utf-8") as f:
            layer_a[node] = parse_layer_a_md(f.read())

    layer_b: dict[str, dict[tuple[int, str], float]] = {}
    for p in args.layer_b_md:
        node = infer_node(p)
        with open(p, encoding="utf-8") as f:
            layer_b[node] = parse_layer_b_md(f.read())

    gate: dict[tuple[str, str], dict[int, float]] = {}
    for spec in args.gate_md:
        parts = spec.split(":", 2)
        if len(parts) != 3:
            raise SystemExit(f"ERROR: --gate-md は node:arm:path 形式が必須（実際: {spec}）")
        node, arm, path = parts
        if node not in ("dgx", "m4max") or arm not in ("off", "on"):
            raise SystemExit(f"ERROR: --gate-md の node/arm が不正: {spec}")
        with open(path, encoding="utf-8") as f:
            gate[(node, arm)] = parse_gate_md(f.read())

    lines, verdict = judge(layer_a, layer_b, gate)
    print(f"# イシュー #1481 機械判定\n")
    print(f"事前登録閾値: RULE234_THRESHOLD={RULE234_THRESHOLD} "
          f"RULE2_OPS_GEMM_THRESHOLD={RULE2_OPS_GEMM_THRESHOLD} "
          f"RULE4_MIN_RATIO={RULE4_MIN_RATIO:.10f}\n")
    for line in lines:
        print(f"- {line}")
    print(f"\n**verdict={verdict}**")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
