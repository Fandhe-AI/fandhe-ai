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

# ---- 事前登録した数値閾値（計測前に固定。以後変更しない） ----
RULE234_THRESHOLD = 1.05  # 規則 2 の Layer A・規則 3・規則 5 の on/off 比上限
RULE2_OPS_GEMM_THRESHOLD = 1.00  # 規則 2 の ops_gemm on/off 比上限（厳密。緩めない）
RULE4_MIN_RATIO = 1.0 / 1.05  # 規則 4（改定版）: on/off 比の下限 (~0.952380...)

GATE_ROW_RE = re.compile(
    r"^\|\s*(?P<n>\d+)\s*\|[^|]*\|[^|]*\|\s*(?P<ratio>[0-9.]+)\s*\|"
)
LAYER_A_ROW_RE = re.compile(
    r"^\|\s*(?P<n>\d+)/(?P<mode>fresh|reuse)\s*\|[^|]*\|[^|]*\|\s*(?P<ratio>[0-9.]+)\s*\|"
)
LAYER_B_SECTION_RE = re.compile(r"^##\s*N=(?P<n>\d+)\s*$")
LAYER_B_ROW_RE = re.compile(
    r"^\|\s*(?P<phase>[A-Za-z_]+)\s*\|[^|]*\|[^|]*\|[^|]*\|[^|]*\|\s*(?P<ratio>[0-9.nNaA]+)\s*\|"
)


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


def parse_layer_a_md(text: str) -> dict[tuple[int, str], float]:
    """`compare_gemm_ab.py --device cpu` 出力 md から {(N, mode): on/off 比} を抽出する。"""
    out: dict[tuple[int, str], float] = {}
    for line in text.splitlines():
        m = LAYER_A_ROW_RE.match(line)
        if m:
            try:
                out[(int(m.group("n")), m.group("mode"))] = float(m.group("ratio"))
            except ValueError:
                continue
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


def judge(
    layer_a: dict[str, dict[tuple[int, str], float]],
    layer_b: dict[str, dict[tuple[int, str], float]],
    gate: dict[tuple[str, str], dict[int, float]],
) -> tuple[list[str], str]:
    """規則 1〜5 を機械判定し、(明細行のリスト, verdict) を返す。

    verdict は "ADOPT" | "REJECT"。規則を満たさないセルが 1 つでもあれば
    REJECT（§20.1 の場合分け (b) 相当。改定版規則 4 は全 6 セル必須の
    片側検査であり (a) の非対称扱いは行わない — 両実機とも同一プロトコル
    で計測している独立再計測では (a)/(c) を区別する意味がないため、本
    スクリプトでは「全規則充足なら ADOPT・いずれか不成立なら REJECT」の
    二値判定に単純化する）。
    """
    lines: list[str] = []
    all_ok = True

    # 規則 2: DGX N=2048 決定セル
    dgx_a = layer_a.get("dgx", {})
    dgx_b = layer_b.get("dgx", {})
    r2_layer_a = dgx_a.get((2048, "reuse"))
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
    all_ok = all_ok and r2_ok

    # 規則 3: 対照セル（両実機 N=512/1024 の fresh/reuse 全セル）
    r3_ok = True
    for node in ("dgx", "m4max"):
        a = layer_a.get(node, {})
        for n in (512, 1024):
            for mode in ("fresh", "reuse"):
                v = a.get((n, mode))
                ok = v is not None and v <= RULE234_THRESHOLD
                lines.append(
                    f"規則3 対照セル {node} N={n}/{mode}: 比={v} (<= {RULE234_THRESHOLD}) "
                    f"-> {'満たす' if ok else '不成立'}"
                )
                r3_ok = r3_ok and ok
    all_ok = all_ok and r3_ok

    # 規則 4（改定版）: 各セル（実機×N=512/1024/2048）の candle 比 on/off 比
    r4_ok = True
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
            r4_ok = r4_ok and ok
    all_ok = all_ok and r4_ok

    # 規則 1: checksum 完全一致は compare_gemm_ab.py 自体の複合判定 pass
    # （`==` 列）を前提とする。本スクリプトは md をパースするのみで
    # 独自の bit 一致検査は行わない（実データは compare_gemm_ab.py の
    # 判定に委ねる。ここでは「md が生成できた」= 複合判定が走ったことの
    # 記録として明示するに留める。実際の pass/fail は env_info.txt に
    # compare_gemm_ab.py の生出力（exit code・`==` 列）を転記して確認する）
    lines.append(
        "規則1 checksum: compare_gemm_ab.py の複合判定結果は生ログ "
        "(compare_gemm_ab-<node>.md 全文・exit code) を env_info.txt に転記して確認 "
        "(本スクリプトは数値抽出のみ)"
    )

    # 規則 5: M4 Max N=2048 の後退判定（Layer A on/off 比 > 1.05 かつ
    # 5 run 符号一貫なら Linux 限定化。本判定は run 別 JSONL の符号一貫性
    # までは見ず、集計後の中央値比のみで判定する（5 run 中央値そのものが
    # 規則 2〜3 と同じ Layer A 表由来のため、符号一貫性は生ログ側で別途
    # 確認し env_info.txt に記録する）。
    m4_2048_reuse = layer_a.get("m4max", {}).get((2048, "reuse"))
    r5_regression = m4_2048_reuse is not None and m4_2048_reuse > RULE234_THRESHOLD
    lines.append(
        f"規則5 M4Max N=2048: layer_a(reuse)={m4_2048_reuse} "
        f"-> {'後退あり（5run符号一貫は生ログで別途確認）' if r5_regression else '後退なし（発火せず）'}"
    )

    verdict = "ADOPT" if all_ok else "REJECT"
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
    assert parsed_a[(2048, "reuse")] == 0.9990, parsed_a
    assert parsed_a[(512, "fresh")] == 1.0075, parsed_a

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

    # judge() の一貫性チェック: 全セル満たす合成サンプルで ADOPT を返すこと
    layer_a = {
        "dgx": {
            (512, "fresh"): 1.00,
            (512, "reuse"): 1.00,
            (1024, "fresh"): 1.00,
            (1024, "reuse"): 1.00,
            (2048, "reuse"): 1.00,
        },
        "m4max": {
            (512, "fresh"): 1.00,
            (512, "reuse"): 1.00,
            (1024, "fresh"): 1.00,
            (1024, "reuse"): 1.00,
            (2048, "reuse"): 1.00,
        },
    }
    layer_b = {"dgx": {(2048, "alloc_c"): 0.5, (2048, "ops_gemm"): 0.99}}
    gate = {
        ("dgx", "off"): {512: 0.70, 1024: 0.78, 2048: 0.96},
        ("dgx", "on"): {512: 0.70, 1024: 0.78, 2048: 0.96},
        ("m4max", "off"): {512: 0.89, 1024: 0.76, 2048: 0.81},
        ("m4max", "on"): {512: 0.89, 1024: 0.76, 2048: 0.81},
    }
    _, verdict = judge(layer_a, layer_b, gate)
    assert verdict == "ADOPT", verdict

    # 規則 4 不成立ケース（on 腕 candle 比が off 腕比で 10% 悪化）→ REJECT
    gate_bad = dict(gate)
    gate_bad[("dgx", "on")] = {512: 0.63, 1024: 0.78, 2048: 0.96}
    _, verdict_bad = judge(layer_a, layer_b, gate_bad)
    assert verdict_bad == "REJECT", verdict_bad

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

    layer_a: dict[str, dict[tuple[int, str], float]] = {}
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
