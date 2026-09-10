#!/usr/bin/env python3
"""イシュー #1521: Metal GEMM candle 比ゲート（旧 #1037）の対照系列
（A: registry `fandhe-ai =0.8.0`）と参考系列（B: split-K 結線後 HEAD への
`crates/facade` path patch）の 2 本の JSONL から帰属表を機械生成する。

判定の正は `compare_gemm_gate.py --device metal` の出力のみ（本スクリプト
は集計・帰属分類のみを行い、判定式・tolerance・`checksums_match` を独自
実装しない。`compare_gemm_gate.load_rows`／`evaluate_size` をそのまま
import して使う。`docs/perf/metal-gemm-candle-gate-remeasurement.md`
§18.1 の事前登録判定規則）。

帰属分類（規則 4。N ごと・fandhe-ai reuse 中央値）:
    r = median_B / median_A
    |r - 1| <= BAND（既定 0.05） または median_B が A の 5 run min-max
    範囲内 -> 「負荷差（ノイズ帯）・構造分析と整合」
    それ以外 -> 「構造分析と矛盾・原因未確定」（コード差確定ではない）

使い方:
    python3 attribute.py <A.jsonl> <B.jsonl> [--band 0.05] [--device metal]
    python3 attribute.py --self-test

fail-closed 契約（A03/A08。codex-review 指摘）: `load_rows` が破損 JSON・
不正型の行を除外して返す読み込み警告が A・B いずれか一方でも 1 件でも
あれば、帰属分類は行わず「判定不能」を出力して非ゼロ終了する。外部入力
（JSONL）の一部行が不正であることは全体の信頼性を損なうため、正常な行
だけが残っても数値・分類を確定表示しない。
"""

from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_FC_DIR = os.path.normpath(
    os.path.join(_HERE, "..", "..", "..", "..", "scripts", "bench", "framework-compare")
)
if _FC_DIR not in sys.path:
    sys.path.insert(0, _FC_DIR)

import compare_gemm_gate as gate  # noqa: E402  (sys.path 設定後の import)

DEFAULT_BAND = 0.05

# `docs/perf/metal-gemm-candle-gate-remeasurement.md` §16.4 の実測値
# （正式系列 `0.8.0-1490`。ドリフト指標としてのみ使う。判定には使わない）。
_SEC16_REFERENCE = {
    1024: {"ratio": 0.743},
    2048: {"ratio": 1.002},
    4096: {"ratio": 0.634},
}


def classify(median_a, min_a, max_a, median_b, band):
    """N 1 個分の帰属分類を返す（規則 4）。

    `(classification, r)` を返す。`median_a<=0` 等の不正値は呼び出し側
    （`evaluate_size` の `status != "ok"`）で弾かれている前提とし、ここ
    では 0 除算のみ防御する。
    """
    if median_a <= 0:
        return "判定不能（A の中央値が非正）", None
    r = median_b / median_a
    in_band = abs(r - 1.0) <= band
    in_minmax = min_a <= median_b <= max_a
    if in_band or in_minmax:
        return "負荷差（ノイズ帯）・構造分析と整合", r
    return "構造分析と矛盾・原因未確定", r


class InputWarningError(RuntimeError):
    """`load_rows` の読み込み警告が 1 件でもある場合に送出する。

    帰属分類（`classify`）は正常な行だけを見て数値を確定表示するが、
    不正行を含む入力を外部から検証せず分類を続行すると、破損データが
    混入した状態でも「判定不能」ではなく確定した数値・分類が出力されて
    しまう（fail-closed 契約違反。A03/A08）。呼び出し側（`render`／
    `main`）はこの例外を受けて「判定不能」表示・非ゼロ終了へ倒す。
    """


def render(path_a, path_b, band, device="metal"):
    rows_a, warn_a = gate.load_rows(path_a)
    rows_b, warn_b = gate.load_rows(path_b)

    if warn_a or warn_b:
        lines = []
        lines.append(f"# 帰属表（A={path_a} / B={path_b}）")
        lines.append("")
        lines.append("## 判定不能（読み込み警告あり。fail-closed）")
        lines.append("")
        lines.append(
            "入力 JSONL に不正行（破損 JSON・不正型）が含まれるため、"
            "帰属分類を行わず判定不能として扱う。"
        )
        lines.append("")
        lines.append("## 読み込み警告")
        for w in warn_a:
            lines.append(f"- A: {w}")
        for w in warn_b:
            lines.append(f"- B: {w}")
        lines.append("")
        raise InputWarningError("\n".join(lines) + "\n")

    lines = []
    lines.append(f"# 帰属表（A={path_a} / B={path_b}）")
    lines.append("")
    lines.append(
        "| N | A中央値(min-max) | B中央値 | B/A | 分類 | candle/A | candle/B "
        "| §16参照値(candle/fandhe) | §16比 |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")

    for size in gate._SIZES_BY_DEVICE[device]:
        res_a = gate.evaluate_size(rows_a, size, device)
        res_b = gate.evaluate_size(rows_b, size, device)
        if res_a.get("status") != "ok" or res_b.get("status") != "ok":
            reason_a = res_a.get("reason", "")
            reason_b = res_b.get("reason", "")
            # ヘッダ・区切り行と同じ 9 列に揃える（Cursor Bugbot 指摘）。
            # A中央値/B中央値/B-A/分類/candle-A/candle-B/§16参照値 の 7 列を
            # 「-」で埋め、最終列（§16比）に理由をまとめて 1 列で収める。
            # 独立の列を追加すると `|` 区切り数がヘッダとずれ、fail-closed
            # の理由が表の外へはみ出す（markdown レンダリングが崩れる）。
            lines.append(
                f"| {size} | 判定不能 | 判定不能 | - | - | - | - | - "
                f"| （A: {reason_a}｜B: {reason_b}） |"
            )
            continue
        med_a = res_a["fandhe_median_s"]
        min_a = res_a["fandhe_min_s"]
        max_a = res_a["fandhe_max_s"]
        med_b = res_b["fandhe_median_s"]
        classification, r = classify(med_a, min_a, max_a, med_b, band)
        candle_over_a = res_a["ratio_candle_over_fandhe"]
        candle_over_b = res_b["ratio_candle_over_fandhe"]
        sec16 = _SEC16_REFERENCE.get(size, {}).get("ratio")
        sec16_ratio_b = (candle_over_b / sec16) if sec16 else None
        lines.append(
            f"| {size} "
            f"| {med_a * 1e3:.3f} ms ({min_a * 1e3:.3f}-{max_a * 1e3:.3f}) "
            f"| {med_b * 1e3:.3f} ms "
            f"| {r:.4f} "
            f"| {classification} "
            f"| {candle_over_a:.3f} "
            f"| {candle_over_b:.3f} "
            f"| {sec16 if sec16 is not None else '-'} "
            f"| {f'{sec16_ratio_b:.3f}' if sec16_ratio_b is not None else '-'} |"
        )
    return "\n".join(lines) + "\n"


def _self_test():
    import json
    import tempfile

    def make_row(framework, mode, size, median_s, checksum=1.0, fail_count=0):
        return {
            "task": "gemm",
            "framework": framework,
            "device": "metal",
            "size": size,
            "mode": mode,
            "median_s": median_s,
            "checksum": checksum,
            "parity_fail_count": fail_count,
            "parity_total": size * size,
            "parity_max_abs_err": 0.0,
            "parity_max_rel_err": 0.0,
        }

    def write_jsonl(rows):
        f = tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        )
        for r in rows:
            f.write(json.dumps(r) + "\n")
        f.close()
        return f.name

    size = 1024
    # ケース 1: ノイズ帯内（B/A ratio 内 5% 以内）。
    rows_a = [make_row("fandhe-ai", "reuse", size, 0.010) for _ in range(5)]
    rows_a += [make_row("candle", "fresh", size, 0.011) for _ in range(5)]
    rows_b_in_band = [make_row("fandhe-ai", "reuse", size, 0.0102) for _ in range(5)]
    rows_b_in_band += [make_row("candle", "fresh", size, 0.011) for _ in range(5)]

    path_a = write_jsonl(rows_a)
    path_b_in_band = write_jsonl(rows_b_in_band)
    try:
        rows_a_loaded, _ = gate.load_rows(path_a)
        rows_b_loaded, _ = gate.load_rows(path_b_in_band)
        res_a = gate.evaluate_size(rows_a_loaded, size, "metal")
        res_b = gate.evaluate_size(rows_b_loaded, size, "metal")
        assert res_a["status"] == "ok"
        assert res_b["status"] == "ok"
        cls, r = classify(
            res_a["fandhe_median_s"],
            res_a["fandhe_min_s"],
            res_a["fandhe_max_s"],
            res_b["fandhe_median_s"],
            DEFAULT_BAND,
        )
        assert "ノイズ帯" in cls, cls
    finally:
        os.unlink(path_a)
        os.unlink(path_b_in_band)

    # ケース 2: 帯域外だが A の min-max 範囲内で救済される（分散大の run 内）。
    rows_a_wide = [
        make_row("fandhe-ai", "reuse", size, v)
        for v in (0.008, 0.009, 0.010, 0.011, 0.015)
    ]
    rows_a_wide += [make_row("candle", "fresh", size, 0.011) for _ in range(5)]
    rows_b_minmax = [make_row("fandhe-ai", "reuse", size, 0.0145) for _ in range(5)]
    rows_b_minmax += [make_row("candle", "fresh", size, 0.011) for _ in range(5)]
    path_a_wide = write_jsonl(rows_a_wide)
    path_b_minmax = write_jsonl(rows_b_minmax)
    try:
        rows_a_loaded, _ = gate.load_rows(path_a_wide)
        rows_b_loaded, _ = gate.load_rows(path_b_minmax)
        res_a = gate.evaluate_size(rows_a_loaded, size, "metal")
        res_b = gate.evaluate_size(rows_b_loaded, size, "metal")
        cls, r = classify(
            res_a["fandhe_median_s"],
            res_a["fandhe_min_s"],
            res_a["fandhe_max_s"],
            res_b["fandhe_median_s"],
            DEFAULT_BAND,
        )
        # r = 0.0145/0.010 = 1.45 -> 帯域外だが A の min-max(0.008-0.015) 内。
        assert abs(r - 1.45) < 1e-6, r
        assert "ノイズ帯" in cls, cls
    finally:
        os.unlink(path_a_wide)
        os.unlink(path_b_minmax)

    # ケース 3: 帯域外・min-max 外（構造分析と矛盾）。
    rows_b_out = [make_row("fandhe-ai", "reuse", size, 0.030) for _ in range(5)]
    rows_b_out += [make_row("candle", "fresh", size, 0.011) for _ in range(5)]
    path_a2 = write_jsonl(rows_a)
    path_b_out = write_jsonl(rows_b_out)
    try:
        rows_a_loaded, _ = gate.load_rows(path_a2)
        rows_b_loaded, _ = gate.load_rows(path_b_out)
        res_a = gate.evaluate_size(rows_a_loaded, size, "metal")
        res_b = gate.evaluate_size(rows_b_loaded, size, "metal")
        cls, r = classify(
            res_a["fandhe_median_s"],
            res_a["fandhe_min_s"],
            res_a["fandhe_max_s"],
            res_b["fandhe_median_s"],
            DEFAULT_BAND,
        )
        assert "矛盾" in cls, cls
    finally:
        os.unlink(path_a2)
        os.unlink(path_b_out)

    # ケース 4: A 側 JSONL に不正行（破損 JSON）が 1 行混入。正常な行は
    # 各 5 件残るが、`render` は分類を行わず `InputWarningError`（判定
    # 不能・非ゼロ終了）を送出しなければならない（fail-closed 契約。
    # codex-review P0 指摘）。
    rows_a_broken = [make_row("fandhe-ai", "reuse", size, 0.010) for _ in range(5)]
    rows_a_broken += [make_row("candle", "fresh", size, 0.011) for _ in range(5)]
    path_a_broken = write_jsonl(rows_a_broken)
    with open(path_a_broken, "a", encoding="utf-8") as f:
        f.write("{not valid json\n")
    path_b_broken = write_jsonl(rows_b_in_band)
    try:
        try:
            render(path_a_broken, path_b_broken, DEFAULT_BAND, "metal")
            raise AssertionError(
                "render() は不正行混入時に InputWarningError を送出すべき"
            )
        except InputWarningError as e:
            assert "判定不能" in str(e), str(e)
        # main() 経由でも非ゼロ終了することを確認する。
        rc = main([path_a_broken, path_b_broken])
        assert rc == 1, rc
    finally:
        os.unlink(path_a_broken)
        os.unlink(path_b_broken)

    # ケース 5: 判定不能サイズ（一部 N のレコードが欠落）を含む出力の
    # 各行が、ヘッダ・区切り行と同じ `|` 区切り列数であることを機械検査
    # する（Cursor Bugbot 指摘。`res_a`/`res_b` の `status != "ok"` 行に
    # 独立列を追加すると表が崩れ、fail-closed の理由が表の外へ出てしまう
    # ため、テーブル整形自体を自己検証する）。size=1024 のみデータを持ち
    # size=2048/4096 は `evaluate_size` が自然に "undeterminable" を返す
    # （MIN_RECORDS 件数不一致）状態で `render` を呼び出す。
    rows_a_one_size = [make_row("fandhe-ai", "reuse", size, 0.010) for _ in range(5)]
    rows_a_one_size += [make_row("candle", "fresh", size, 0.011) for _ in range(5)]
    path_a_one_size = write_jsonl(rows_a_one_size)
    path_b_one_size = write_jsonl(rows_b_in_band)
    try:
        table = render(path_a_one_size, path_b_one_size, DEFAULT_BAND, "metal")
        table_lines = table.rstrip("\n").split("\n")
        header_line = next(ln for ln in table_lines if ln.startswith("| N |"))
        sep_line = next(ln for ln in table_lines if ln.startswith("|---"))
        expected_cols = header_line.count("|") - 1
        assert sep_line.count("|") - 1 == expected_cols, (sep_line, expected_cols)
        row_lines = [
            ln
            for ln in table_lines
            if ln.startswith("|") and ln not in (header_line, sep_line)
        ]
        assert len(row_lines) == len(gate._SIZES_BY_DEVICE["metal"]), row_lines
        for ln in row_lines:
            got_cols = ln.count("|") - 1
            assert got_cols == expected_cols, (ln, got_cols, expected_cols)
        # size=2048/4096 は判定不能行のはずで、理由が最終列に収まっている
        # こと（列がヘッダからはみ出していないこと）を明示確認する。
        undetermined_rows = [ln for ln in row_lines if "判定不能" in ln]
        assert len(undetermined_rows) == 2, undetermined_rows
    finally:
        os.unlink(path_a_one_size)
        os.unlink(path_b_one_size)

    print("attribute.py --self-test: all cases passed")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl_a", nargs="?", help="対照系列（A）の JSONL")
    parser.add_argument("jsonl_b", nargs="?", help="参考系列（B）の JSONL")
    parser.add_argument(
        "--band",
        type=float,
        default=DEFAULT_BAND,
        help=(
            "ノイズ帯の相対幅（既定 0.05）。事前登録済みの値であり計測後に"
            "変更しない（docs/perf/metal-gemm-candle-gate-remeasurement.md"
            " §18.1 規則 4）。"
        ),
    )
    parser.add_argument("--device", default="metal", choices=["cuda", "metal", "cpu"])
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)

    if args.self_test:
        _self_test()
        return 0

    if not args.jsonl_a or not args.jsonl_b:
        parser.error("jsonl_a と jsonl_b の両方が必要（--self-test 以外）")

    try:
        print(render(args.jsonl_a, args.jsonl_b, args.band, args.device))
    except InputWarningError as e:
        print(str(e))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
