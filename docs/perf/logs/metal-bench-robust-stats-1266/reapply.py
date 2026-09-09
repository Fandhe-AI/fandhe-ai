#!/usr/bin/env python3
"""イシュー #1266 向け: ロバスト統計候補を #1249 系（#1242/#1186/#1187/#1255）実測ログへ再適用する。

`docs/perf/metal-bench-noise-protocol.md` §8「ロバスト統計の設計提案」の表を
生成する固定スクリプト。**新規計測は行わない**（既存ログの再集計のみ）。
Python3 標準ライブラリのみで完結させ（`.claude/rules/security.md` A03: 外部
入力〈ログ本文〉を正規表現でパースするのみで `subprocess`／`eval`／ネット
ワークを使わない）、ホスト名・ユーザー名・絶対パスをハードコードしない
（`Path(__file__)` から相対解決する。`docs/real-hardware-verification-env.md`
方針）。

閾値（0.05）は `crates/bench-harness/src/ab.rs::STABILITY_SPREAD_GATE`
（単一真実源）をこのスクリプトでは再定義せず、判定には**ログ行から読み取った
実際の `gate=`**（`logged_gate`。セルごとに保持）を使う。#1255（正確値）系は
実ログの `gate=` 値をそのまま使い、`validate_gate_consistency()` で全 25 セル
の `logged_gate` が `GATE` 定数と一致することを検査する（不一致ならログ側の
実閾値が定数からドリフトしている証拠であり、目視で気づけないまま判定基準が
分散するのを防ぐため fail-closed で異常終了する）。1186/1187 系（参考値）は
`gate=` を出力しない旧フォーマットのため、`GATE` 定数を「ログが本来使って
いた値」として明示的に仮定した値を `logged_gate` に代入する（実ログ由来の
値ではないことをコード上も分離して残す。値そのものは変更しない）。

分位点の定義は `crates/bench-harness/src/stats.rs::median_q1_q3`
（median-of-halves 方式: `idx = round(p * (n-1))`）と同一のものをここでも
用いる。Rust `f64::round` は half-away-from-zero、Python 組み込み `round()`
は銀行家丸めのため、`floor(x + 0.5)` で Rust 側の丸めを明示的に再現する
（`--self-test` で #1255 の実測 `spread=` 値と 25/25 一致することを検査）。

出力は標準出力へ Markdown を書く（決定的。乱数なし）。
`docs/perf/metal-bench-noise-protocol.md` §8 の表はこの出力から転記する。
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

# ログディレクトリはこのスクリプト自身の位置からの相対パスで解決する
# （worktree の絶対パス・ユーザー名を埋め込まない）。
LOGS_ROOT = Path(__file__).resolve().parents[1] / "metal-gemm-transpose-route-ab-1242"

SIZES = (256, 512, 1024, 2048, 4096)
GATE = 0.05  # STABILITY_SPREAD_GATE（ab.rs）の値をここでは転記するのみ。再定義しない。


def rust_round(x: float) -> int:
    """Rust `f64::round`（half-away-from-zero）を再現する。

    Python 組み込み `round()` は銀行家丸め（round-half-to-even）のため
    n=10 等の偶数長サンプルで `median_q1_q3`（stats.rs）の idx 選択が
    ずれる（例: 4.5 → Python 4／Rust 5）。`floor(x + 0.5)` は非負の x
    （p*(n-1) は p∈[0,1]・n≥1 で常に非負）に対して half-away-from-zero と
    一致する。
    """
    return math.floor(x + 0.5)


def median_q1_q3(samples: list[float]) -> tuple[float, float, float]:
    """`stats.rs::median_q1_q3` と同一定義（median-of-halves 方式）。

    返り値は (median, q1, q3)。
    """
    s = sorted(samples)
    n = len(s)

    def pick(p: float) -> float:
        idx = rust_round(p * (n - 1))
        return s[min(idx, n - 1)]

    return pick(0.5), pick(0.25), pick(0.75)


def raw_spread(samples: list[float]) -> float:
    """`stats.rs::relative_spread` と同一定義: (max - min) / median。"""
    median, _, _ = median_q1_q3(samples)
    return (max(samples) - min(samples)) / median


def trimmed_range_spread(samples: list[float], k: int) -> float | None:
    """案 A/A′: 上下各 k 個を除いた残りのレンジ ÷ 全系列 median。

    n - 2k < 2 の場合（レンジが定義できない）は None を返す
    （本データセットは n=10 固定のため k∈{1,2} では到達しない防御的分岐）。
    """
    median, _, _ = median_q1_q3(samples)
    s = sorted(samples)
    n = len(s)
    if n - 2 * k < 2:
        return None
    trimmed = s[k : n - k]
    return (max(trimmed) - min(trimmed)) / median


def tukey_fence_spread(samples: list[float], c: float) -> tuple[float, int]:
    """案 B: Tukey フェンス（[Q1 - c*IQR, Q3 + c*IQR]）外を除去した残りの
    レンジ ÷ 全系列 median。全系列の Q1/Q3 を使う（除去が非対称になり
    うるため、トリム後の再計算ではなく元の分位点を固定して使う）。

    戻り値は (spread, 除去数)。全除去（残り 0）の場合は spread=0.0 とする
    （理論上 n=10・c>=1.5 では到達しない防御的分岐）。
    """
    median, q1, q3 = median_q1_q3(samples)
    iqr = q3 - q1
    lo, hi = q1 - c * iqr, q3 + c * iqr
    kept = [x for x in samples if lo <= x <= hi]
    removed = len(samples) - len(kept)
    if not kept:
        return 0.0, removed
    return (max(kept) - min(kept)) / median, removed


def iqr_over_median(samples: list[float]) -> float:
    """案 C: (Q3 - Q1) / median。"""
    median, q1, q3 = median_q1_q3(samples)
    return (q3 - q1) / median


def mad_over_median(samples: list[float]) -> float:
    """案 D: 2 * MAD / median。MAD は |x - median| の median-of-halves 中央値。"""
    median, _, _ = median_q1_q3(samples)
    deviations = [abs(x - median) for x in samples]
    mad, _, _ = median_q1_q3(deviations)
    return 2.0 * mad / median


# ---------------------------------------------------------------------------
# ログパース
# ---------------------------------------------------------------------------

_EXACT_LINE_RE = re.compile(
    r"phase1_round_stats size=(?P<size>\d+) rounds=(?P<rounds>\d+) "
    r"spread=(?P<spread>[0-9.eE+-]+) gate=(?P<gate>[0-9.eE+-]+) "
    r"within_gate=(?P<within_gate>true|false) median_secs=[0-9.eE+-]+ "
    r"min_secs=[0-9.eE+-]+ min_round_idx=\d+ max_secs=[0-9.eE+-]+ "
    r"max_round_idx=\d+ round_medians_secs=(?P<samples>[0-9.eE+,-]+)"
)

_REF_LINE_RE = re.compile(
    r"size=(?P<size>\d+) spread=(?P<spread>[0-9.]+) \((?P<verdict>OK|NG[^)]*)\) "
    r"round_tflops=\[(?P<samples>[0-9., ]+)\]"
)


def parse_exact_log(path: Path) -> dict[int, dict]:
    """#1255 の `phase1_round_stats` 行（正確値。秒単位）を読む。

    `round_medians_secs` は本番コード（`ab.rs::run_stability`）が実測した
    ラウンド中央値そのものであり、変換を要しない。
    """
    result: dict[int, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = _EXACT_LINE_RE.search(line)
        if not m:
            continue
        size = int(m.group("size"))
        samples = [float(x) for x in m.group("samples").split(",")]
        result[size] = {
            "samples_secs": samples,
            "logged_spread": float(m.group("spread")),
            "logged_gate": float(m.group("gate")),
        }
    return result


def parse_reference_log(path: Path) -> dict[int, dict]:
    """#1186/#1187 の `round_tflops` 行（参考値。TFLOPS 単位）を読む。

    TFLOPS は `time ∝ 1/tflops` の反比例量のため、`secs = 1/tflops` へ
    復元してから他候補と同じ秒領域の統計量を適用する（丸め誤差により
    ログの `spread=`〈4 桁〉と再計算値が数 % ずれうる。「参考」ラベル
    が必要な理由）。ログ自体の `spread=` は tflops 値ではなく `1/tflops`
    領域で計算されていることを `--self-test` は検証しない（近似のため）
    が、目視突合のためログ値も保持する。
    """
    result: dict[int, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = _REF_LINE_RE.search(line)
        if not m:
            continue
        size = int(m.group("size"))
        tflops = [float(x) for x in m.group("samples").split(",")]
        secs = [1.0 / t for t in tflops]
        result[size] = {
            "samples_secs": secs,
            "logged_spread": float(m.group("spread")),
            "logged_gate": GATE,
        }
    return result


def load_all_sessions() -> tuple[dict[str, dict[int, dict]], dict[str, dict[int, dict]]]:
    """(正確値セッション群, 参考値セッション群) を返す。

    キーはセッション識別子（例: "1255-run1"）。値は size→cell dict。
    """
    exact: dict[str, dict[int, dict]] = {}
    for i in range(1, 6):
        p = LOGS_ROOT / f"1255-phase1_run{i}.log"
        exact[f"1255-run{i}"] = parse_exact_log(p)

    reference: dict[str, dict[int, dict]] = {}
    ref_1186 = LOGS_ROOT.parent / "metal-gemm-transpose-route-ab-1186" / "route_ab_run1.log"
    reference["1186-run1"] = parse_reference_log(ref_1186)
    for i in range(1, 5):
        p = LOGS_ROOT.parent / "metal-gemm-transpose-route-ab-1187" / f"route_ab_run{i}.log"
        reference[f"1187-run{i}"] = parse_reference_log(p)

    return exact, reference


# ---------------------------------------------------------------------------
# 候補統計量の適用
# ---------------------------------------------------------------------------

CANDIDATES = ("raw", "A(k=1)", "A'(k=2)", "B(1.5)", "B(3.0)", "C", "D")


def compute_candidates(samples: list[float]) -> dict[str, float]:
    b15, _ = tukey_fence_spread(samples, 1.5)
    b30, _ = tukey_fence_spread(samples, 3.0)
    out = {
        "raw": raw_spread(samples),
        "A(k=1)": trimmed_range_spread(samples, 1),
        "A'(k=2)": trimmed_range_spread(samples, 2),
        "B(1.5)": b15,
        "B(3.0)": b30,
        "C": iqr_over_median(samples),
        "D": mad_over_median(samples),
    }
    return out


# ---------------------------------------------------------------------------
# self-test: #1255（正確値）の raw spread がログ `spread=` と一致するか
# ---------------------------------------------------------------------------


def self_test(exact: dict[str, dict[int, dict]]) -> bool:
    ok = True
    checked = 0
    for session, cells in exact.items():
        for size, cell in cells.items():
            recomputed = raw_spread(cell["samples_secs"])
            logged = cell["logged_spread"]
            checked += 1
            if logged == 0.0:
                rel_ok = abs(recomputed) < 1e-9
            else:
                rel_ok = abs(recomputed - logged) / abs(logged) < 1e-3
            if not rel_ok:
                ok = False
                print(
                    f"[self-test FAIL] {session} size={size}: "
                    f"recomputed={recomputed:.6e} logged={logged:.6e}",
                    file=sys.stderr,
                )
    print(f"[self-test] {checked} 件（#1255 正確値）を検査", file=sys.stderr)
    if checked != 25:
        print(f"[self-test FAIL] 期待 25 件、実際 {checked} 件", file=sys.stderr)
        ok = False
    return ok


def validate_gate_consistency(exact: dict[str, dict[int, dict]]) -> bool:
    """判定に使う `logged_gate`（#1255 正確値セッション。実ログの `gate=` 転記）が
    `GATE` 定数と全件一致することを検査する。

    判定処理自体は各セルの `logged_gate` を直接使う（共通閾値と信じて `GATE`
    定数を判定式に埋め込むのではなく、ログが実際に使っていた値を使う設計。
    P1 是正: `logged_gate` を保存するだけで判定には未使用だった構造を解消）。
    この検査は「共通閾値という前提」自体が壊れていないかを別途保証する
    フェイルセーフで、レポート生成時（`--self-test` なし）にも常に実行する。
    1186/1187 系（参考値）は `gate=` を出力しない旧フォーマットのため
    `logged_gate` 自体が `GATE` の代入値であり、本検査の対象外（トートロジー
    になるため）。
    """
    ok = True
    checked = 0
    for session, cells in exact.items():
        for size, cell in cells.items():
            checked += 1
            if cell["logged_gate"] != GATE:
                ok = False
                print(
                    f"[gate-consistency FAIL] {session} size={size}: "
                    f"logged_gate={cell['logged_gate']!r} != GATE={GATE!r}",
                    file=sys.stderr,
                )
    print(f"[gate-consistency] {checked} 件（#1255 正確値）の logged_gate を検査", file=sys.stderr)
    return ok


# ---------------------------------------------------------------------------
# レポート生成
# ---------------------------------------------------------------------------


def render_report(exact: dict[str, dict[int, dict]], reference: dict[str, dict[int, dict]]) -> str:
    lines: list[str] = []
    lines.append("<!-- 本ファイルは reapply.py の決定的出力。手動編集しない。 -->")
    lines.append("")
    lines.append(
        "gate（閾値。判定は各セルの `logged_gate`〈ログの `gate=` 転記。"
        f"#1255 系は `validate_gate_consistency()` で `GATE`={GATE} と全件一致を検証済み〉"
        "を使用。1186/1187 系は `gate=` 非出力の旧フォーマットのため GATE 定数を仮定値として転記）"
    )
    lines.append("")

    all_sessions: dict[str, dict[int, dict]] = {}
    all_sessions.update(exact)
    all_sessions.update(reference)

    # --- per-cell 表 ---
    lines.append("## per-cell 表（run × size × 候補。値 ≤ gate なら成立）")
    lines.append("")
    header = "| session | size | " + " | ".join(CANDIDATES) + " |"
    # per-cell 表（session・size の 2 列 + 候補列）用の区切り行。
    sep = "|---|---|" + "---|" * len(CANDIDATES)
    # サイズ別・セッション別の集計表（1 列 + 候補列）は列数が異なるため
    # 専用の区切り行を用いる（P2 是正: per-cell 用 sep の使い回しは
    # ヘッダーと列数が食い違い GitHub 上で表として描画されない）。
    sep_1col = "|---|" + "---|" * len(CANDIDATES)
    lines.append(header)
    lines.append(sep)

    cell_pass_count = {c: 0 for c in CANDIDATES}
    total_cells = 0
    session_order = list(exact.keys()) + list(reference.keys())
    per_session_pass_all: dict[str, dict[str, bool]] = {c: {} for c in CANDIDATES}

    for session in session_order:
        cells = all_sessions[session]
        for size in SIZES:
            if size not in cells:
                continue
            total_cells += 1
            cell_gate = cells[size]["logged_gate"]
            candidates = compute_candidates(cells[size]["samples_secs"])
            row = [session, str(size)]
            for c in CANDIDATES:
                v = candidates[c]
                passed = v is not None and v <= cell_gate
                if passed:
                    cell_pass_count[c] += 1
                mark = "✓" if passed else ""
                row.append(f"{v:.4f}{mark}" if v is not None else "n/a")
            lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append(f"合計セル数: {total_cells}")
    lines.append("")

    # --- サイズ別成立数（2 分母） ---
    lines.append("## サイズ別成立数")
    lines.append("")
    lines.append("### 分母: #1255 正確値のみ（5 run × 5 size = 25 セル）")
    lines.append("")
    header2 = "| size | " + " | ".join(CANDIDATES) + " |"
    lines.append(header2)
    lines.append(sep_1col)
    for size in SIZES:
        row = [str(size)]
        for c in CANDIDATES:
            n_pass = 0
            n_total = 0
            for session in exact.keys():
                cells = exact[session]
                if size not in cells:
                    continue
                n_total += 1
                v = compute_candidates(cells[size]["samples_secs"])[c]
                if v is not None and v <= cells[size]["logged_gate"]:
                    n_pass += 1
            row.append(f"{n_pass}/{n_total}")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append("### 分母: #1255 正確値 + #1186/#1187 参考値込み（10 run × 5 size = 50 セル）")
    lines.append("")
    lines.append(header2)
    lines.append(sep_1col)
    for size in SIZES:
        row = [str(size)]
        for c in CANDIDATES:
            n_pass = 0
            n_total = 0
            for session in session_order:
                cells = all_sessions[session]
                if size not in cells:
                    continue
                n_total += 1
                v = compute_candidates(cells[size]["samples_secs"])[c]
                if v is not None and v <= cells[size]["logged_gate"]:
                    n_pass += 1
            row.append(f"{n_pass}/{n_total}")
        lines.append("| " + " | ".join(row) + " |")

    # --- セッション単位（5 サイズ全成立） ---
    lines.append("")
    lines.append("## セッション単位の成立可否（5 サイズすべて成立して初めてゲート通過）")
    lines.append("")
    header3 = "| session | " + " | ".join(CANDIDATES) + " |"
    lines.append(header3)
    lines.append(sep_1col)
    session_pass_count = {c: 0 for c in CANDIDATES}
    for session in session_order:
        cells = all_sessions[session]
        row = [session]
        for c in CANDIDATES:
            all_pass = True
            for size in SIZES:
                if size not in cells:
                    all_pass = False
                    break
                v = compute_candidates(cells[size]["samples_secs"])[c]
                if v is None or v > cells[size]["logged_gate"]:
                    all_pass = False
                    break
            if all_pass:
                session_pass_count[c] += 1
            row.append("PASS" if all_pass else "fail")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append(f"セッション単位成立数（{len(session_order)} セッション中）: "
                 + ", ".join(f"{c}={session_pass_count[c]}" for c in CANDIDATES))
    lines.append("")
    lines.append(f"セル単位成立数（{total_cells} セル中）: "
                 + ", ".join(f"{c}={cell_pass_count[c]}" for c in CANDIDATES))

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="#1255 正確値 25 セルで raw spread の再計算値がログ spread= と一致するか検証する",
    )
    args = parser.parse_args()

    exact, reference = load_all_sessions()

    if args.self_test:
        ok = self_test(exact) and validate_gate_consistency(exact)
        return 0 if ok else 1

    # レポート生成時も「判定に使う logged_gate が GATE 定数から乖離していない
    # か」を毎回検査する（--self-test 限定にすると通常実行〈標準出力を
    # reapply.md へ転記する経路〉ではドリフトを検出できないため）。
    if not validate_gate_consistency(exact):
        return 1

    sys.stdout.write(render_report(exact, reference))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
