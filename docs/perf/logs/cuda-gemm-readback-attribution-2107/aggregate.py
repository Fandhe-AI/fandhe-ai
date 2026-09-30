#!/usr/bin/env python3
"""イシュー #2107 実測ログの集計スクリプト（python3 標準ライブラリのみ）。

`orchestrate.sh` が生成する次を読み、`RULE.txt`（実測前に固定）の規則で
5 run 中央値・帰属判定を Markdown（`aggregate.md`）として出力する。

- `<dir>/run{1..5}/n{N}_{arm}.jsonl`: `readback_attribution_diag_tests_2107.rs`
  の `DIAG_JSON` 1 行（1 プロセス 1 行）。単位は μs。
- `<dir>/layerA-phases-N{N}.log`: 同一セッションの
  `bench-fandhe --task gemm --device cuda --mode reuse --phases` の JSONL を
  5 run 分連結したもの（`matmul` 区間の `median_s`・`checksum` を使う）。
- `<dir>/env_info.txt`（任意）: `layerA_same_code: yes|no|unknown` 行（計測経路の同一性検査
  `check_layer_a_path_identity.py` の結果）と `host_kind: gb10|x86` 行。
- `<dir>/load_gate_status.txt`: `run<k> gate=pass|unpassed|record_only`。
  GB10 の通過済み系列（host_kind=gb10 かつ run1..5 の全記録が `gate=pass`）以外
  （ファイル欠落・run 記録の不足/重複・不明値・未通過・x86 smoke の record_only）は
  通常の帰属判定を出さず「参考扱い」と明示する（fail-closed。RULE.txt の専有ゲート契約・
  x86 smoke は GB10 実測の代替にしない）。

完全性・checksum 検査は fail-closed（不一致・欠落は `IntegrityError` を
送出し、集計値を一切出力せず exit 1）。`--self-test` は GPU なしで
合成 fixture（正常系・checksum 不一致・residual<=0・run 欠落）を検証する。
判定しきい値は RULE.txt と同一（0.80 / 0.50。事後に変更しない）。

帰属判定（residual・alloc_fill_share）は本番順序の 2 腕（`prod_order_fresh`／
`prod_order_reused`。起動後に同期しない）の readback 区間壁時計差のみを使う。
直列腕（`kernel_wait` で事前同期する 5 腕）は区間内訳の補助診断で、判定に使わない。
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import tempfile
from pathlib import Path

SIZES = (1024, 2048, 4096)
RUNS = 5
ARMS = (
    "clone_dtoh_legacy_to_vec",
    "clone_dtoh_borrowed_keep_alive",
    "clone_dtoh_borrowed_dummy_alloc_free",
    "pretouched_fresh_split",
    "pretouched_fresh_production",
    "prod_order_fresh",
    "prod_order_reused",
)
# 直列腕（起動直後に kernel_wait で同期してから readback を計る。区間内訳の補助診断用）と
# 本番順序腕（同期なし。帰属判定用）で前半区間が異なる。
FRONT_PHASES_SERIALIZED = ("h2d_a", "h2d_b", "alloc_c", "launch_issue", "kernel_wait")
FRONT_PHASES_PROD_ORDER = ("h2d_a", "h2d_b", "alloc_c", "launch_issue")
PROD_ORDER_ARMS = ("prod_order_fresh", "prod_order_reused")


def front_phases(arm: str) -> tuple[str, ...]:
    return FRONT_PHASES_PROD_ORDER if arm in PROD_ORDER_ARMS else FRONT_PHASES_SERIALIZED

# 腕別 readback サブフェーズ名（readback_attribution_diag_tests_2107.rs の
# `Arm::readback_parts` と一致させる）。
ARM_PARTS = {
    "clone_dtoh_legacy_to_vec": ("clone_dtoh", "d2h_sync"),
    "clone_dtoh_borrowed_keep_alive": ("clone_dtoh", "d2h_sync"),
    "clone_dtoh_borrowed_dummy_alloc_free": ("clone_dtoh", "d2h_sync"),
    "pretouched_fresh_split": ("dest_alloc", "pretouch_fill", "d2h_issue", "d2h_sync"),
    "pretouched_fresh_production": ("readback_total",),
    "prod_order_fresh": ("readback_total",),
    "prod_order_reused": ("d2h_issue", "d2h_sync"),
}
# JSONL の値は Rust 側で `{:.3}`（μs）に丸められた区間中央値の和のため、
# 区間数（最大 9）×0.0005 を上回る余裕を持たせた絶対許容（μs）。
SUM_RECHECK_TOL_US = 0.01
KEEP_ALIVE = "clone_dtoh_borrowed_keep_alive"
PROD_FRESH = "prod_order_fresh"
PROD_REUSED = "prod_order_reused"
SPLIT = "pretouched_fresh_split"
PRODUCTION = "pretouched_fresh_production"
LEGACY = "clone_dtoh_legacy_to_vec"
# RULE.txt 記載の Layer A 既知 checksum（§12.3）。
KNOWN_LAYER_A_CHECKSUM = {
    1024: -1855.597736,
    2048: -6016.774008,
    4096: -25768.747284,
}
SUPPORT_THRESHOLD = 0.80
REJECT_THRESHOLD = 0.50
HEALTH_TOLERANCE = 0.05


class IntegrityError(Exception):
    """完全性・checksum 検査の失敗（fail-closed。集計値を出力しない）。"""


def med(values: list[float]) -> float:
    return statistics.median(values)


def load_arm_runs(base: Path, n: int, arm: str) -> list[dict]:
    rows = []
    for k in range(1, RUNS + 1):
        p = base / f"run{k}" / f"n{n}_{arm}.jsonl"
        if not p.is_file():
            raise IntegrityError(f"欠落: {p}")
        lines = [ln for ln in p.read_text().splitlines() if ln.strip()]
        if len(lines) != 1:
            raise IntegrityError(f"{p}: JSONL は 1 行のはずだが {len(lines)} 行")
        try:
            row = json.loads(lines[0])
        except ValueError as e:
            raise IntegrityError(f"{p}: 壊れた JSON: {e}") from e
        if row.get("n") != n or row.get("arm") != arm:
            raise IntegrityError(f"{p}: n/arm がファイル名と不一致")
        recheck_sums(p, arm, row)
        rows.append(row)
    return rows


def recheck_sums(p: Path, arm: str, row: dict) -> None:
    """`sum_readback_us`／`sum_matmul_equiv_us` を phases の区間中央値から再計算して照合する。

    JSONL の合計値をそのまま信用せず（改竄・出力側バグの検出）、区間値の和と
    一致しなければ fail-closed（PR #2452 codex-review 指摘 P2）。
    """
    try:
        ph = row["phases"]
        readback = sum(float(ph[nm]["median_us"]) for nm in ARM_PARTS[arm])
        front = sum(float(ph[nm]["median_us"]) for nm in front_phases(arm))
        got_rb = float(row["sum_readback_us"])
        got_mm = float(row["sum_matmul_equiv_us"])
    except (KeyError, TypeError, ValueError) as e:
        raise IntegrityError(f"{p}: 合計値の再計算に必要な項目が欠落・不正: {e!r}") from e
    if abs(got_rb - readback) > SUM_RECHECK_TOL_US:
        raise IntegrityError(
            f"{p}: sum_readback_us={got_rb} が区間値からの再計算 {readback:.3f} と不一致"
        )
    if abs(got_mm - (front + readback)) > SUM_RECHECK_TOL_US:
        raise IntegrityError(
            f"{p}: sum_matmul_equiv_us={got_mm} が区間値からの再計算 {front + readback:.3f} と不一致"
        )


def phase_summary(rows: list[dict], phase: str) -> dict:
    """run ごとの中央値（μs）を集め、5 run 間の中央値・min・max を返す。"""
    try:
        vals = [r["phases"][phase]["median_us"] for r in rows]
    except KeyError as e:
        raise IntegrityError(f"phase {phase} が欠落: {e}") from e
    return {"median": med(vals), "min": min(vals), "max": max(vals)}


def parse_layer_a(path: Path, n: int) -> dict:
    if not path.is_file():
        raise IntegrityError(f"欠落: {path}")
    matmul: list[float] = []
    checks: list[float] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line.startswith("{"):
            continue
        try:
            row = json.loads(line)
        except ValueError as e:
            raise IntegrityError(f"{path}: 壊れた JSON: {e}") from e
        if row.get("task") != "gemm_phases" or row.get("phase") != "matmul":
            continue
        if row.get("size") != n:
            raise IntegrityError(f"{path}: size={row.get('size')} が N={n} と不一致")
        matmul.append(float(row["median_s"]) * 1e6)
        checks.append(float(row["checksum"]))
    if len(matmul) != RUNS:
        raise IntegrityError(f"{path}: matmul 行が {len(matmul)} 件（期待 {RUNS}）")
    known = KNOWN_LAYER_A_CHECKSUM[n]
    for c in checks:
        if abs(c - known) > 1e-6:
            raise IntegrityError(
                f"{path}: Layer A checksum {c} が既知値 {known} と不一致"
            )
    return {"median": med(matmul), "min": min(matmul), "max": max(matmul)}


def series_reliability(base: Path, host_kind: str) -> str:
    """GB10 通過済み系列でない理由を返す（空文字なら通過済み系列）。

    欠落・不明値・run 記録不足・未通過・GB10 以外は全て非空（fail-closed）。
    """
    if host_kind != "gb10":
        return (
            f"host_kind={host_kind or '不明'}: GB10 実測ではない"
            "（x86 smoke 等は GB10 実測の代替にしない）"
        )
    gate = base / "load_gate_status.txt"
    if not gate.is_file():
        return "load_gate_status.txt が欠落（専有ゲート状態を確認できない）"
    seen: dict[int, list[str]] = {}
    for raw in gate.read_text().splitlines():
        parts = raw.split()
        if not parts:
            continue
        m = re.fullmatch(r"run(\d+)", parts[0])
        g = parts[1].removeprefix("gate=") if len(parts) == 2 and parts[1].startswith("gate=") else ""
        if not m:
            return f"load_gate_status.txt に不明な行: {raw!r}"
        seen.setdefault(int(m.group(1)), []).append(g)
    expected = set(range(1, RUNS + 1))
    if set(seen) != expected or any(len(v) != 1 for v in seen.values()):
        return f"load_gate_status.txt の run 記録が run1..run{RUNS} を 1 件ずつ満たさない"
    bad = {k: v[0] for k, v in sorted(seen.items()) if v[0] != "pass"}
    if bad:
        return "専有ゲートが pass でない run あり: " + ", ".join(
            f"run{k}={v or '不明'}" for k, v in bad.items()
        )
    return ""


def verdict(residual: float, share: float | None) -> str:
    if residual <= 0:
        return "未説明分非再現（帰属は行わない）"
    assert share is not None
    if share >= SUPPORT_THRESHOLD:
        return "宛先確保・事前タッチへの帰属を支持"
    if share < REJECT_THRESHOLD:
        return "棄却（主因は別）"
    return "未確定"


def analyze(base: Path) -> tuple[str, dict]:
    same_code = "unknown"
    host_kind = ""
    env = base / "env_info.txt"
    if env.is_file():
        for ln in env.read_text().splitlines():
            if ln.startswith("layerA_same_code:"):
                same_code = ln.split(":", 1)[1].strip()
            elif ln.startswith("host_kind:"):
                host_kind = ln.split(":", 1)[1].strip()
    gate_note = series_reliability(base, host_kind)
    if gate_note:
        gate_note = f"GB10 の専有ゲート通過済み系列ではない（{gate_note}）。系列は参考扱い"

    out: list[str] = ["# #2107 集計結果（RULE.txt の規則。単位 μs・5 run 中央値）", ""]
    if same_code != "yes":
        out.append(
            f"> 注意: layerA_same_code={same_code}。Layer A と Layer B が同一コードと"
            "確認できないため、帰属判定は「無効（参考扱い）」。"
        )
        out.append("")
    if gate_note:
        out.append(f"> 注意: {gate_note}")
        out.append("")
    results: dict = {}
    for n in SIZES:
        arms_rows = {a: load_arm_runs(base, n, a) for a in ARMS}
        # checksum: (N, 腕) 内の全 run 一致・同一 N の全腕一致（fail-closed）。
        bits_all = set()
        for a, rows in arms_rows.items():
            bits = {r["checksum_bits"] for r in rows}
            if len(bits) != 1:
                raise IntegrityError(f"N={n} 腕 {a}: run 間で checksum_bits が不一致")
            bits_all |= bits
        if len(bits_all) != 1:
            raise IntegrityError(f"N={n}: 腕間で checksum_bits が不一致 {sorted(bits_all)}")
        # Layer B の checksum を Layer A 既知値（§12.3）と照合する（RULE.txt hard 条件。
        # 入力は Layer A と同一の SEED_A/SEED_B なので一致するはず。不一致は系列無効）。
        known = KNOWN_LAYER_A_CHECKSUM[n]
        for a, rows in arms_rows.items():
            for r in rows:
                c = float(r["checksum"])
                if not (abs(c - known) <= 1e-6):
                    raise IntegrityError(
                        f"N={n} 腕 {a}: Layer B checksum {c} が Layer A 既知値 {known} と不一致"
                    )
        layer_a = parse_layer_a(base / f"layerA-phases-N{n}.log", n)

        out += [f"## N={n}", "", "| 腕 | 区間 | median | min | max |", "|---|---|---|---|---|"]
        summ: dict = {}
        for a, rows in arms_rows.items():
            phases = list(rows[0]["phases"].keys())
            summ[a] = {p: phase_summary(rows, p) for p in phases}
            # 各 run の合計は load_arm_runs で区間値から再計算照合済み（不一致は
            # IntegrityError）。ここでは照合済みの値の 5 run 中央値を使う。
            summ[a]["_sum_matmul_equiv"] = {
                "median": med([r["sum_matmul_equiv_us"] for r in rows])
            }
            summ[a]["_sum_readback"] = {"median": med([r["sum_readback_us"] for r in rows])}
            for p in phases:
                s = summ[a][p]
                out.append(
                    f"| {a} | {p} | {s['median']:.1f} | {s['min']:.1f} | {s['max']:.1f} |"
                )
            out.append(
                f"| {a} | Σ_matmul_equiv | {summ[a]['_sum_matmul_equiv']['median']:.1f} | | |"
            )
        out += [
            f"| Layer A | matmul | {layer_a['median']:.1f} | {layer_a['min']:.1f} | "
            f"{layer_a['max']:.1f} |",
            "",
        ]

        # --- 帰属判定（本番順序の 2 腕のみ。直列腕は判定に使わない。RULE.txt「帰属の定義」）
        # exposed cost = 同じ本番順序（起動後に同期しない）での「新規宛先 readback」と
        # 「確保済み・タッチ済み宛先への D2H」の readback 区間壁時計差。直列区間の和は使わない。
        tail_fresh = summ[PROD_FRESH]["readback_total"]["median"]
        tail_reused = summ[PROD_REUSED]["d2h_issue"]["median"] + summ[PROD_REUSED]["d2h_sync"]["median"]
        exposed = tail_fresh - tail_reused
        sigma_fresh = summ[PROD_FRESH]["_sum_matmul_equiv"]["median"]
        sigma_reused = summ[PROD_REUSED]["_sum_matmul_equiv"]["median"]
        residual = layer_a["median"] - sigma_reused
        share = exposed / residual if residual > 0 else None
        v = verdict(residual, share)
        # --- 記録のみ（判定には使わない）
        calib = layer_a["median"] - sigma_fresh  # 本番順序の Layer B が Layer A を再現する度合い
        sigma_serial_prod = summ[PRODUCTION]["_sum_matmul_equiv"]["median"]
        overlap = sigma_serial_prod - sigma_fresh  # 同一 readback 関数の直列/本番順序の壁時計差
        serial_alloc_fill = (
            summ[SPLIT]["dest_alloc"]["median"] + summ[SPLIT]["pretouch_fill"]["median"]
        )
        old_residual = layer_a["median"] - summ[KEEP_ALIVE]["_sum_matmul_equiv"]["median"]
        split_d2h = summ[SPLIT]["d2h_issue"]["median"] + summ[SPLIT]["d2h_sync"]["median"]
        clone_minus_split = (
            summ[KEEP_ALIVE]["clone_dtoh"]["median"]
            + summ[KEEP_ALIVE]["d2h_sync"]["median"]
            - split_d2h
        )
        split_readback = serial_alloc_fill + split_d2h
        prod_total = summ[PRODUCTION]["readback_total"]["median"]
        distortion = abs(split_readback - prod_total) / prod_total if prod_total > 0 else float("inf")
        healthy = distortion <= HEALTH_TOLERANCE
        ratio = summ[SPLIT]["_sum_readback"]["median"] / summ[LEGACY]["_sum_readback"]["median"]
        if same_code != "yes":
            v_out = f"無効（参考扱い。素の判定: {v}）"
        else:
            v_out = v
        # 専有ゲート未通過の系列は参考扱い（RULE.txt「専有ゲート」）。各 N の判定にも明示する。
        if gate_note and not v_out.startswith("無効"):
            v_out = f"参考扱い（専有ゲート未通過。素の判定: {v_out}）"
        out += [
            f"- exposed_alloc_fill = prod_order_fresh.readback_total − prod_order_reused.(d2h_issue+d2h_sync)"
            f" = {tail_fresh:.1f} − {tail_reused:.1f} = {exposed:.1f} us",
            f"- residual = LayerA.matmul − Σ_matmul_equiv(prod_order_reused) = {residual:.1f} us",
            f"- alloc_fill_share = {'n/a' if share is None else f'{share:.3f}'}"
            "（exposed_alloc_fill / residual）",
            f"- **判定: {v_out}**",
            "- 判定が示す範囲: 本番順序で表に出た宛先確保・事前タッチ費用（exposed cost）が"
            "residual を占める割合のみ。重なりに隠れた費用・Layer B が再現しない Layer A 側の"
            "固定費は検出対象外",
            f"- 校正: LayerA.matmul − Σ_matmul_equiv(prod_order_fresh) = {calib:.1f} us"
            f"（{calib / layer_a['median'] * 100:.1f}% of LayerA。本番順序の Layer B が"
            "Layer A を再現する度合い。記録のみ）",
            f"- 重なり: Σ_matmul_equiv(直列 production) − Σ_matmul_equiv(prod_order_fresh) = {overlap:.1f} us"
            "（同一 readback 関数の直列/本番順序の壁時計差。記録のみ）",
            f"- 直列腕の dest_alloc+pretouch_fill = {serial_alloc_fill:.1f} us"
            f"（直列区間の値。exposed との差が重なりに隠れた分。判定には使わない・記録のみ）",
            f"- 旧 residual（§12.5 基準）= LayerA.matmul − Σ_matmul_equiv(keep_alive 直列) = {old_residual:.1f} us"
            "（連続性のための記録のみ）",
            f"- clone_dtoh 系 readback − split の (d2h_issue+d2h_sync) = {clone_minus_split:.1f} us"
            "（直列腕同士の比較。記録のみ）",
            f"- 直列分解の整合（補助）: |split 分解和 − production readback_total| / production = "
            f"{distortion:.3f}（許容 {HEALTH_TOLERANCE}。{'OK' if healthy else '超過'}。"
            "直列腕同士のみの整合で判定には使わない・記録のみ）",
            f"- 非後退 ratio = split.Σreadback / legacy.Σreadback = {ratio:.3f}（直列腕同士。記録のみ）",
            "",
        ]
        results[n] = {
            "residual": residual,
            "exposed": exposed,
            "overlap": overlap,
            "share": share,
            "verdict": v_out,
            "healthy": healthy,
        }
    return "\n".join(out) + "\n", results


# ---------------------------------------------------------------- self-test


def _phase(m: float) -> dict:
    return {"median_us": m, "q1_us": m, "q3_us": m, "min_us": m, "max_us": m}


def _write_fixture(
    base: Path,
    *,
    layer_a_us: float,
    checksum_bits: str = "c09cf4e2a0000000",
    bad_bits_arm: str | None = None,
    bad_value_arm: str | None = None,
    drop_run: bool = False,
    host_kind: str = "gb10",
    gate: str = "pass",
) -> None:
    parts = {
        "clone_dtoh_legacy_to_vec": {"clone_dtoh": 300, "d2h_sync": 50},
        "clone_dtoh_borrowed_keep_alive": {"clone_dtoh": 300, "d2h_sync": 50},
        "clone_dtoh_borrowed_dummy_alloc_free": {"clone_dtoh": 300, "d2h_sync": 50},
        "pretouched_fresh_split": {
            "dest_alloc": 10,
            "pretouch_fill": 240,
            "d2h_issue": 50,
            "d2h_sync": 10,
        },
        "pretouched_fresh_production": {"readback_total": 310},
        # 本番順序: fresh の readback 壁時計 520・reused の D2H 壁時計 250 → exposed=270。
        "prod_order_fresh": {"readback_total": 520},
        "prod_order_reused": {"d2h_issue": 50, "d2h_sync": 200},
    }
    for n in SIZES:
        for k in range(1, RUNS + 1):
            d = base / f"run{k}"
            d.mkdir(parents=True, exist_ok=True)
            for arm in ARMS:
                if drop_run and k == 3 and arm == SPLIT and n == 1024:
                    continue
                ph = {nm: _phase(100.0) for nm in front_phases(arm)}
                ph.update({nm: _phase(float(v)) for nm, v in parts[arm].items()})
                ph["host_read"] = _phase(1.0)
                s_back = float(sum(parts[arm].values()))
                row = {
                    "issue": 2107,
                    "n": n,
                    "arm": arm,
                    "warmup": 20,
                    "measured": 20,
                    "phases": ph,
                    "checksum": (
                        KNOWN_LAYER_A_CHECKSUM[n] + 1.0
                        if arm == bad_value_arm and k == 3
                        else KNOWN_LAYER_A_CHECKSUM[n]
                    ),
                    "checksum_bits": (
                        "deadbeefdeadbeef" if arm == bad_bits_arm and k == 2 else checksum_bits
                    ),
                    "sum_matmul_equiv_us": 100.0 * len(front_phases(arm)) + s_back,
                    "sum_readback_us": s_back,
                }
                (d / f"n{n}_{arm}.jsonl").write_text(json.dumps(row) + "\n")
        lines = []
        for k in range(1, RUNS + 1):
            lines.append(f"-- run {k}/N={n} --")
            lines.append(
                json.dumps(
                    {
                        "task": "gemm_phases",
                        "phase": "matmul",
                        "size": n,
                        "median_s": layer_a_us / 1e6,
                        "checksum": KNOWN_LAYER_A_CHECKSUM[n],
                    }
                )
            )
        (base / f"layerA-phases-N{n}.log").write_text("\n".join(lines) + "\n")
    (base / "env_info.txt").write_text(f"layerA_same_code: yes\nhost_kind: {host_kind}\n")
    (base / "load_gate_status.txt").write_text(
        "".join(f"run{k} gate={gate}\n" for k in range(1, RUNS + 1))
    )


def self_test() -> int:
    failures = 0

    def expect(cond: bool, msg: str) -> None:
        nonlocal failures
        if not cond:
            failures += 1
            print(f"FAIL: {msg}", file=sys.stderr)

    with tempfile.TemporaryDirectory() as t:
        # 正常系: Σ_matmul_equiv(prod_order_reused)=650・exposed=520−250=270。
        # layerA=1100 で residual=450 → share=0.6（未確定）。重なり=直列 production
        # (500+310=810) − prod_order_fresh(400+520=920) = −110。
        base = Path(t) / "ok"
        _write_fixture(base, layer_a_us=1100.0)
        text, res = analyze(base)
        expect(abs(res[1024]["residual"] - 450.0) < 1e-6, "residual 450")
        expect(abs(res[1024]["exposed"] - 270.0) < 1e-6, "exposed 270")
        expect(abs(res[1024]["overlap"] - (-110.0)) < 1e-6, "overlap")
        expect(abs(res[1024]["share"] - 270.0 / 450.0) < 1e-9, "share")
        expect("exposed_alloc_fill" in text and "重なり" in text, "exposed・重なりの出力")
        expect("検出対象外" in text, "判定範囲の明示")
        expect(res[1024]["verdict"] == "未確定", f"verdict {res[1024]['verdict']}")
        expect("## N=4096" in text, "N=4096 セクション")

        # 直列腕（split の dest_alloc/pretouch_fill）を変えても判定・share は不変
        # （帰属判定は本番順序 2 腕のみ。直列区間の和は判定に使わない）。
        base = Path(t) / "serial_indep"
        _write_fixture(base, layer_a_us=1100.0)
        for k in range(1, RUNS + 1):
            pth = base / f"run{k}" / f"n1024_{SPLIT}.jsonl"
            row = json.loads(pth.read_text())
            row["phases"]["pretouch_fill"]["median_us"] = 9999.0
            row["sum_readback_us"] += 9999.0 - 240.0
            row["sum_matmul_equiv_us"] += 9999.0 - 240.0
            pth.write_text(json.dumps(row) + "\n")
        _, res2 = analyze(base)
        expect(abs(res2[1024]["share"] - 270.0 / 450.0) < 1e-9, "直列腕は share に無関係")
        expect(res2[1024]["verdict"] == "未確定", "直列腕は判定に無関係")

        # 支持: residual=300（layerA=950）→ share=0.9。
        base = Path(t) / "support"
        _write_fixture(base, layer_a_us=950.0)
        _, res = analyze(base)
        expect("支持" in res[2048]["verdict"], f"support {res[2048]['verdict']}")

        # 棄却: residual=800（layerA=1450）→ share=0.3375。
        base = Path(t) / "reject"
        _write_fixture(base, layer_a_us=1450.0)
        _, res = analyze(base)
        expect("棄却" in res[2048]["verdict"], f"reject {res[2048]['verdict']}")

        # residual<=0: 帰属を行わない。
        base = Path(t) / "nores"
        _write_fixture(base, layer_a_us=600.0)
        _, res = analyze(base)
        expect(res[1024]["share"] is None, "residual<=0 で share None")
        expect("非再現" in res[1024]["verdict"], "非再現")

        # checksum 不一致: fail-closed。
        base = Path(t) / "bad"
        _write_fixture(base, layer_a_us=1300.0, bad_bits_arm=SPLIT)
        try:
            analyze(base)
            expect(False, "checksum 不一致で IntegrityError")
        except IntegrityError:
            pass

        # Layer B checksum が Layer A 既知値と不一致（bits は全腕一致）: fail-closed。
        base = Path(t) / "badvalue"
        _write_fixture(base, layer_a_us=1300.0, bad_value_arm=KEEP_ALIVE)
        try:
            analyze(base)
            expect(False, "Layer B checksum 既知値不一致で IntegrityError")
        except IntegrityError:
            pass

        # 合計値が区間値からの再計算と不一致（JSONL 改竄）: fail-closed。
        for key in ("sum_readback_us", "sum_matmul_equiv_us"):
            base = Path(t) / f"badsum_{key}"
            _write_fixture(base, layer_a_us=1300.0)
            p = base / "run2" / f"n1024_{KEEP_ALIVE}.jsonl"
            row = json.loads(p.read_text())
            row[key] += 5.0
            p.write_text(json.dumps(row) + "\n")
            try:
                analyze(base)
                expect(False, f"{key} 再計算不一致で IntegrityError")
            except IntegrityError:
                pass

        # run 欠落: fail-closed。
        base = Path(t) / "drop"
        _write_fixture(base, layer_a_us=1300.0, drop_run=True)
        try:
            analyze(base)
            expect(False, "run 欠落で IntegrityError")
        except IntegrityError:
            pass

        # Layer A checksum が既知値と不一致: fail-closed。
        base = Path(t) / "lacs"
        _write_fixture(base, layer_a_us=1300.0)
        p = base / "layerA-phases-N1024.log"
        p.write_text(p.read_text().replace("-1855.597736", "-1.0"))
        try:
            analyze(base)
            expect(False, "Layer A checksum 不一致で IntegrityError")
        except IntegrityError:
            pass

        # 専有ゲート未通過: 各 N の判定にも参考扱いを明示する。
        base = Path(t) / "gate"
        _write_fixture(base, layer_a_us=950.0)
        (base / "load_gate_status.txt").write_text("run1 gate=unpassed\n")
        _, res = analyze(base)
        expect(
            all(res[n]["verdict"].startswith("参考扱い") for n in SIZES),
            "ゲート未通過なら全 N の判定が参考扱い",
        )
        expect("支持" in res[2048]["verdict"], "素の判定は括弧内に保持")

        # 通過済み GB10 系列: 通常の判定（参考扱いにならない）。
        base = Path(t) / "gbok"
        _write_fixture(base, layer_a_us=950.0)
        _, res = analyze(base)
        expect(not res[2048]["verdict"].startswith("参考扱い"), "GB10 全 pass は通常判定")

        # load_gate_status.txt 欠落・run 不足・不明値・record_only・x86: 全て参考扱い。
        def reference_case(name: str, mutate) -> None:
            b = Path(t) / name
            _write_fixture(b, layer_a_us=1150.0)
            mutate(b)
            text, r = analyze(b)
            expect(
                all(r[n]["verdict"].startswith("参考扱い") for n in SIZES),
                f"{name}: 参考扱い",
            )
            expect("参考扱い" in text.split("## N=")[0], f"{name}: 冒頭に注意書き")

        reference_case("gate_missing", lambda b: (b / "load_gate_status.txt").unlink())
        reference_case(
            "gate_short",
            lambda b: (b / "load_gate_status.txt").write_text("run1 gate=pass\n"),
        )
        reference_case(
            "gate_unknown",
            lambda b: (b / "load_gate_status.txt").write_text(
                "".join(f"run{k} gate={'weird' if k == 3 else 'pass'}\n" for k in range(1, 6))
            ),
        )
        reference_case(
            "gate_dup",
            lambda b: (b / "load_gate_status.txt").write_text(
                "".join(f"run{k} gate=pass\n" for k in (1, 1, 2, 3, 4))
            ),
        )
        b = Path(t) / "x86"
        _write_fixture(b, layer_a_us=1150.0, host_kind="x86", gate="record_only")
        text, r = analyze(b)
        expect(
            all(r[n]["verdict"].startswith("参考扱い") for n in SIZES),
            "x86 smoke は参考扱い",
        )
        reference_case(
            "host_missing",
            lambda b: (b / "env_info.txt").write_text("layerA_same_code: yes\n"),
        )

        # 同一コード未確認: 判定は無効（参考扱い）。
        base = Path(t) / "unk"
        _write_fixture(base, layer_a_us=1150.0)
        (base / "env_info.txt").write_text("layerA_same_code: no\n")
        _, res = analyze(base)
        expect("無効" in res[1024]["verdict"], "同一コードでなければ無効")

    print("self-test:", "OK" if failures == 0 else f"{failures} failure(s)")
    return 0 if failures == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dir", nargs="?", help="orchestrate.sh の出力ディレクトリ")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if not args.dir:
        ap.error("dir が必要（または --self-test）")
    base = Path(args.dir)
    try:
        text, _ = analyze(base)
    except IntegrityError as e:
        print(f"完全性検査に失敗（系列は無効。集計値を出力しない）: {e}", file=sys.stderr)
        return 1
    (base / "aggregate.md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
