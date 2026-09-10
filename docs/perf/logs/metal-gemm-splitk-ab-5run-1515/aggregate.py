#!/usr/bin/env python3
"""split-K vs classic 経路の A/B 5 run 正式確定（イシュー #1515）ログを
集計する。

`crates/backend-metal/examples/gemm_splitk_ab_bench.rs` が各プロセス起動
（`runN.log`）へ出力する `splitk_ab` 行・`phase0` 行・`env_guard_*` 行を
解析し、`docs/perf/metal-gemm-splitk-ab.md` §10 へ転記する `aggregate.md`
を生成する。`docs/perf/logs/metal-gemm-splitk-ab-1475/aggregate.py`（旧
3 run 版）を元に、以下を追加・変更する（イシュー #1515・ルート #1509
のユーザー指示: 専有ゲートを受け入れ条件にしない）:

- `--gate-mode {gated,record_only}`（既定 `gated`。fail-closed）。
  `record_only` を指定した場合のみ各 run ログが `env_guard_mode=
  record_only` かつ `env_guard_load_avg one=<数値>` 行を持つことを要求し、
  `gated` ログ（`env_guard_mode=gated`）が紛れていれば違反とする。逆に
  `gated`（既定）では旧版と同じ `env_guard_mode=gated` ＋
  `env_guard_overall verdict=pass` を要求する。
- `phase0 kind=...` 行から run-to-run bit 同一（`classic_stable`／
  `target_tile_stable`／`splitk_stable`）・`a_vs_at_fail_count=0`・
  `checksum_a`／`checksum_at`／`checksum_b` の全 run 一致を検査する
  （受け入れ条件「checksum 一致」の機械化）。`a_vs_b_fail_count` は
  情報としてのみ表に残し判定へは使わない（split-K 経路自体の既知 fail
  が再現することが期待値のため。`docs/perf/metal-gemm-splitk-two-pass.md`
  §5.5）。
- `--monitor-logs a.log,b.log,...`（任意・カンマ区切り）: run ごとの
  バックグラウンド `uptime` サンプラーログから load1 の
  min／median／max・サンプル数を「負荷推移」表として出力する（情報のみ。
  verdict の入力にしない。共有負荷下であることの記録用）。

事前登録判定規則（`docs/perf/metal-gemm-splitk-ab.md` §10.2。#1475 の
規則を据え置き、計測後に変更しない）:
- ADOPT: 対象 9 形状すべてで (i) run 内比の 5 run 中央値 >= 1.5 かつ
  (ii) 5/5 run すべてで run 内比 > 1.0、かつ対照 3 形状すべてで
  中央値 >= 0.95。
- 上記いずれかが不成立なら REJECT（共有負荷下でも有効な REJECT）。
- undetermined: (i) 5 run の完全性（形状の欠落・重複なし）が崩れる、
  (ii) フェーズ 0 の run-to-run bit 同一・checksum 一致が崩れる、
  (iii) `--gate-mode` に応じた env_guard 記録が欠落・不一致、のいずれか。

python3 標準ライブラリのみ（許容依存外の追加なし）。

使い方:
    python3 aggregate.py --self-test
    python3 aggregate.py --gate-mode=record_only \
        --monitor-logs=run1_monitor.log,run2_monitor.log,... \
        run1.log run2.log run3.log run4.log run5.log > aggregate.md
"""

from __future__ import annotations

import statistics
import sys
from collections import Counter
from dataclasses import dataclass, field

ADOPT_TARGET_MIN_SPEEDUP = 1.5
ADOPT_CONTROL_MIN_SPEEDUP = 0.95

# 5 run 正式確定契約（イシュー #1515。旧 #1475 の 3 run は是正前バイナリ
# 〈seed_offset 不一致・B′ 未呼び出し〉による計測のため新規 5 run とし、
# 混在させない。`docs/perf/metal-gemm-splitk-ab.md` §10.2）。
MIN_FORMAL_RUNS = 5

# `crates/backend-metal/examples/gemm_splitk_ab_bench.rs` の
# `target_shapes`／`control_shapes` と同一の期待形状集合。
_TARGET_MN = (32, 64, 128)
_CONTROL_MN = 256
_K_LIST = (2048, 4096, 8192)
EXPECTED_TARGET_KEYS = frozenset(
    ("target", mn, mn, k) for mn in _TARGET_MN for k in _K_LIST
)
EXPECTED_CONTROL_KEYS = frozenset(
    ("control", _CONTROL_MN, _CONTROL_MN, k) for k in _K_LIST
)

GATE_MODES = ("gated", "record_only")


def check_run_shape_completeness(
    rows: list[dict[str, str]], run_index: int
) -> list[str]:
    """1 run 分の `splitk_ab` 行（target/control kind に限る）が、期待形状
    集合の各キーをちょうど 1 回ずつ含むことを検査する純関数
    （`metal-gemm-splitk-ab-1475/aggregate.py` と同型。イシュー #1499
    codex-review P2 指摘: run をまたいだ欠落・重複の相殺を検出するため
    run 単位で検査する）。
    """
    counts: Counter[tuple[str, int, int, int]] = Counter()
    for row in rows:
        kind = row.get("kind")
        if kind not in ("target", "control"):
            continue
        if "m" not in row or "n" not in row or "k" not in row:
            continue
        key = (kind, int(row["m"]), int(row["n"]), int(row["k"]))
        counts[key] += 1

    violations: list[str] = []
    expected = EXPECTED_TARGET_KEYS | EXPECTED_CONTROL_KEYS
    for key in sorted(expected):
        n = counts.get(key, 0)
        if n != 1:
            violations.append(
                f"run{run_index}: kind={key[0]} m={key[1]} n={key[2]} k={key[3]} "
                f"の行数が {n}（期待 1）"
            )
    unexpected = set(counts) - expected
    for key in sorted(unexpected):
        violations.append(
            f"run{run_index}: kind={key[0]} m={key[1]} n={key[2]} k={key[3]} "
            f"は期待形状集合外（n={counts[key]}）"
        )
    return violations


def check_env_guard(text: str, run_index: int, gate_mode: str) -> list[str]:
    """1 run 分のログテキストが `gate_mode` に応じた env_guard 記録を
    持つことを検査する純関数（イシュー #1515）。

    - `gate_mode == "gated"`: `env_guard_mode=gated` かつ
      `env_guard_overall verdict=pass` の両方を要求する（旧
      `metal-gemm-splitk-ab-1475/aggregate.py::check_env_guard_gated_pass`
      と同一の検査）。
    - `gate_mode == "record_only"`: `env_guard_mode=record_only` かつ
      `env_guard_load_avg one=<数値>` 行（load average が数値として
      記録されている）を要求する（イシュー #1515・ルート #1509 のユーザー
      指示により専有ゲートは受け入れ条件にしないため、判定は行わず記録の
      有無のみを検査する）。

    いずれのモードでも、指定モードと矛盾するモード宣言行
    （`gate_mode=record_only` 指定時に `env_guard_mode=gated` 行がある等）
    が見つかれば違反とする（fail-closed。ログの取り違え検出）。
    """
    violations: list[str] = []
    declared_modes: set[str] = set()
    has_overall_pass = False
    has_numeric_load_avg = False

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("env_guard_mode="):
            declared_modes.add(stripped.split("=", 1)[1])
        elif stripped.startswith("env_guard_overall ") and "verdict=pass" in stripped:
            has_overall_pass = True
        elif stripped.startswith("env_guard_load_avg "):
            for tok in stripped.split(" "):
                if tok.startswith("one="):
                    value = tok.split("=", 1)[1]
                    try:
                        float(value)
                        has_numeric_load_avg = True
                    except ValueError:
                        pass

    if gate_mode == "gated":
        if "gated" not in declared_modes:
            violations.append(
                f"run{run_index}: env_guard_mode=gated が見つからない"
                "（専有ゲート未実施・記録のみモードの可能性）"
            )
        if not has_overall_pass:
            violations.append(
                f"run{run_index}: env_guard_overall verdict=pass が見つからない"
                "（専有ゲート不成立、または env_guard 自体が未実行）"
            )
        if "record_only" in declared_modes:
            violations.append(
                f"run{run_index}: --gate-mode=gated 指定だが "
                "env_guard_mode=record_only 行が混在している（ログの取り違え）"
            )
    else:
        if "record_only" not in declared_modes:
            violations.append(
                f"run{run_index}: env_guard_mode=record_only が見つからない"
                "（--gate-mode=record_only 指定時は record_only 運用のログのみ許容する）"
            )
        if not has_numeric_load_avg:
            violations.append(
                f"run{run_index}: env_guard_load_avg の one= が数値として記録されていない"
                "（共有負荷下の記録が欠落している）"
            )
        if "gated" in declared_modes:
            violations.append(
                f"run{run_index}: --gate-mode=record_only 指定だが "
                "env_guard_mode=gated 行が混在している（ログの取り違え）"
            )
    return violations


@dataclass
class Phase0Target:
    m: int
    n: int
    k: int
    classic_stable: bool = False
    target_tile_stable: bool = False
    splitk_stable: bool = False
    a_vs_at_fail_count: int | None = None
    a_vs_b_fail_count: int | None = None
    checksum_a: str | None = None
    checksum_at: str | None = None
    checksum_b: str | None = None

    @property
    def stable(self) -> bool:
        return self.classic_stable and self.target_tile_stable and self.splitk_stable


def parse_phase0_line(line: str) -> dict[str, str] | None:
    """`phase0 kind=... ...` 行を key=value の dict へ分解する（純関数）。"""
    if not line.startswith("phase0 "):
        return None
    out: dict[str, str] = {}
    for tok in line.strip().split(" ")[1:]:
        if "=" not in tok:
            continue
        k, v = tok.split("=", 1)
        out[k] = v
    return out


def parse_bool(s: str | None) -> bool:
    return s == "true"


def check_phase0_consistency(
    per_run_phase0: list[list[dict[str, str]]],
) -> list[str]:
    """全 run の `phase0` 行から (a) run-to-run bit 同一（`classic_stable`
    等）・`a_vs_at_fail_count=0`、(b) target の checksum（`checksum_a`／
    `checksum_at`／`checksum_b`）が全 run で完全一致すること、を検査する
    純関数（受け入れ条件「checksum 一致」の機械化。イシュー #1515）。

    checksum は f64 の科学的記数法文字列（`format_ab_line` 等と同様に
    example 側が既に固定精度で出力するため、文字列一致で bit 完全一致相当
    を検査できる。浮動小数点の再パース・再比較による誤差混入を避ける）。
    """
    violations: list[str] = []
    n_runs = len(per_run_phase0)

    target_by_run: list[dict[tuple[int, int, int], dict[str, str]]] = []
    control_by_run: list[dict[tuple[int, int, int], dict[str, str]]] = []
    for run_index, rows in enumerate(per_run_phase0):
        t: dict[tuple[int, int, int], dict[str, str]] = {}
        c: dict[tuple[int, int, int], dict[str, str]] = {}
        for row in rows:
            kind = row.get("kind")
            if kind is None or "m" not in row or "n" not in row or "k" not in row:
                continue
            key = (int(row["m"]), int(row["n"]), int(row["k"]))
            if kind == "target":
                t[key] = row
            elif kind == "control":
                c[key] = row
        target_by_run.append(t)
        control_by_run.append(c)

        for key in sorted(EXPECTED_TARGET_KEYS):
            _, m, n, k = key
            row = t.get((m, n, k))
            if row is None:
                violations.append(
                    f"run{run_index}: phase0 kind=target m={m} n={n} k={k} が見つからない"
                )
                continue
            if not parse_bool(row.get("classic_stable")):
                violations.append(
                    f"run{run_index}: phase0 target m={m} n={n} k={k} classic_stable が true でない"
                )
            if not parse_bool(row.get("target_tile_stable")):
                violations.append(
                    f"run{run_index}: phase0 target m={m} n={n} k={k} target_tile_stable が true でない"
                )
            if not parse_bool(row.get("splitk_stable")):
                violations.append(
                    f"run{run_index}: phase0 target m={m} n={n} k={k} splitk_stable が true でない"
                )
            fail_count = row.get("a_vs_at_fail_count")
            if fail_count != "0":
                violations.append(
                    f"run{run_index}: phase0 target m={m} n={n} k={k} "
                    f"a_vs_at_fail_count={fail_count}（期待 0）"
                )

        for key in sorted(EXPECTED_CONTROL_KEYS):
            _, m, n, k = key
            row = c.get((m, n, k))
            if row is None:
                violations.append(
                    f"run{run_index}: phase0 kind=control m={m} n={n} k={k} が見つからない"
                )
                continue
            if not parse_bool(row.get("classic_stable")):
                violations.append(
                    f"run{run_index}: phase0 control m={m} n={n} k={k} classic_stable が true でない"
                )

    # checksum の全 run 一致（target のみ。control は checksum を出力しない）。
    if n_runs > 0:
        for key in sorted(EXPECTED_TARGET_KEYS):
            _, m, n, k = key
            for field_name in ("checksum_a", "checksum_at", "checksum_b"):
                values = {
                    target_by_run[i].get((m, n, k), {}).get(field_name)
                    for i in range(n_runs)
                    if (m, n, k) in target_by_run[i]
                }
                values.discard(None)
                if len(values) > 1:
                    violations.append(
                        f"target m={m} n={n} k={k} の {field_name} が run 間で不一致: "
                        f"{sorted(values)}"
                    )

    return violations


@dataclass
class ShapeSamples:
    kind: str
    m: int
    n: int
    k: int
    speedups: list[float] = field(default_factory=list)
    median_a_secs_list: list[float] = field(default_factory=list)

    @property
    def key(self) -> tuple[str, int, int, int]:
        return (self.kind, self.m, self.n, self.k)

    @property
    def median_speedup(self) -> float | None:
        if not self.speedups:
            return None
        return statistics.median(self.speedups)

    @property
    def median_a_secs(self) -> float | None:
        if not self.median_a_secs_list:
            return None
        return statistics.median(self.median_a_secs_list)

    @property
    def all_positive(self) -> bool:
        return all(s > 1.0 for s in self.speedups)


def parse_splitk_ab_line(line: str) -> dict[str, str] | None:
    """`splitk_ab ...` 行を key=value の dict へ分解する（純関数）。"""
    if not line.startswith("splitk_ab "):
        return None
    out: dict[str, str] = {}
    for tok in line.strip().split(" ")[1:]:
        if "=" not in tok:
            continue
        k, v = tok.split("=", 1)
        out[k] = v
    return out


def parse_log(text: str) -> tuple[list[dict[str, str]], list[dict[str, str]], bool]:
    """1 run 分のログを解析し、(splitk_ab 行の dict 列, phase0 行の dict 列,
    undetermined か) を返す。
    """
    rows: list[dict[str, str]] = []
    phase0_rows: list[dict[str, str]] = []
    undetermined = False
    for line in text.splitlines():
        if line.strip() == "verdict=undetermined":
            undetermined = True
        parsed = parse_splitk_ab_line(line)
        if parsed is not None:
            rows.append(parsed)
        p0 = parse_phase0_line(line)
        if p0 is not None:
            phase0_rows.append(p0)
    return rows, phase0_rows, undetermined


def aggregate(
    logs: list[str],
    gate_mode: str = "gated",
) -> tuple[dict[tuple[str, int, int, int], ShapeSamples], bool, list[str]]:
    """複数 run のログテキストを集約する。`gate_mode` は
    `check_env_guard` へそのまま渡す（イシュー #1515）。戻り値の構造は
    `metal-gemm-splitk-ab-1475/aggregate.py::aggregate` と同型。
    """
    if gate_mode not in GATE_MODES:
        raise ValueError(f"gate_mode は {GATE_MODES} のいずれかである必要がある: {gate_mode}")

    shapes: dict[tuple[str, int, int, int], ShapeSamples] = {}
    any_undetermined = False
    violations: list[str] = []
    per_run_phase0: list[list[dict[str, str]]] = []

    for run_index, text in enumerate(logs):
        rows, phase0_rows, undetermined = parse_log(text)
        per_run_phase0.append(phase0_rows)
        if undetermined:
            any_undetermined = True
        run_violations = check_run_shape_completeness(rows, run_index)
        if run_violations:
            violations.extend(run_violations)
        env_guard_violations = check_env_guard(text, run_index, gate_mode)
        if env_guard_violations:
            violations.extend(env_guard_violations)
        for row in rows:
            kind = row.get("kind")
            if kind is None or "m" not in row or "n" not in row or "k" not in row:
                continue
            m, n, k = int(row["m"]), int(row["n"]), int(row["k"])
            key = (kind, m, n, k)
            if key not in shapes:
                shapes[key] = ShapeSamples(kind, m, n, k)
            speedup = row.get("speedup")
            median_a = row.get("median_a_secs")
            median_b = row.get("median_b_secs")
            if (
                speedup is not None
                and speedup != "NA"
                and median_a is not None
                and median_a != "NA"
                and median_b is not None
                and median_b != "NA"
            ):
                shapes[key].speedups.append(float(median_a) / float(median_b))
            if median_a is not None and median_a != "NA":
                shapes[key].median_a_secs_list.append(float(median_a))

    phase0_violations = check_phase0_consistency(per_run_phase0)
    if phase0_violations:
        violations.extend(phase0_violations)

    if violations:
        any_undetermined = True
    return shapes, any_undetermined, violations


@dataclass
class MonitorSummary:
    path: str
    samples: int
    load1_min: float | None
    load1_median: float | None
    load1_max: float | None


def parse_monitor_log(text: str) -> list[float]:
    """`orchestrate.sh` のバックグラウンド `uptime` サンプラーログ
    （`<UTC時刻> load1=<x> load5=<y> load15=<z>` 形式）から load1 の値列を
    抽出する純関数。判定には使わない情報表示専用（イシュー #1515）。
    """
    values: list[float] = []
    for line in text.splitlines():
        for tok in line.strip().split(" "):
            if tok.startswith("load1="):
                try:
                    values.append(float(tok.split("=", 1)[1]))
                except ValueError:
                    pass
    return values


def summarize_monitor_logs(paths: list[str]) -> list[MonitorSummary]:
    out: list[MonitorSummary] = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            values = parse_monitor_log(f.read())
        if values:
            out.append(
                MonitorSummary(
                    path=p,
                    samples=len(values),
                    load1_min=min(values),
                    load1_median=statistics.median(values),
                    load1_max=max(values),
                )
            )
        else:
            out.append(MonitorSummary(p, 0, None, None, None))
    return out


def render_markdown(
    shapes: dict[tuple[str, int, int, int], ShapeSamples],
    any_undetermined: bool,
    n_runs: int,
    violations: list[str] | None = None,
    gate_mode: str = "gated",
    monitor_summaries: list[MonitorSummary] | None = None,
) -> str:
    violations = violations or []
    lines: list[str] = []
    lines.append(f"# split-K A/B 5 run 正式確定 集計（{n_runs} run・gate_mode={gate_mode}）\n")
    if gate_mode == "record_only":
        lines.append(
            "**共有負荷下（専有ゲートなし・record_only 運用）。イシュー #1515・"
            "ルート #1509 のユーザー指示により専有ゲートは受け入れ条件にしない。**\n"
        )
    if any_undetermined:
        lines.append(
            "**いずれかの run が `verdict=undetermined`（gated 運用時の専有ゲート不成立）、"
            "run 単位の期待形状集合検査での不一致（欠落・重複・想定外キー）、"
            "env_guard 記録の不一致・欠落、またはフェーズ 0 の run-to-run bit 同一・"
            "checksum 一致検査での不一致を検出した。以下は参考値であり判定には使わない。**\n"
        )
    if violations:
        lines.append("\n### 完全性・env_guard・フェーズ 0 の不一致\n")
        for v in violations:
            lines.append(f"- {v}")
        lines.append("")

    if monitor_summaries:
        lines.append("\n### 負荷推移（情報のみ。判定には使わない）\n")
        lines.append("| log | samples | load1_min | load1_median | load1_max |")
        lines.append("|-----|---------|-----------|--------------|-----------|")
        for s in monitor_summaries:
            def fmt(v: float | None) -> str:
                return f"{v:.2f}" if v is not None else "NA"

            lines.append(
                f"| {s.path} | {s.samples} | {fmt(s.load1_min)} | "
                f"{fmt(s.load1_median)} | {fmt(s.load1_max)} |"
            )
        lines.append("")

    for kind in ["target", "target_tile", "control", "control_forced", "floor"]:
        rows = [v for v in shapes.values() if v.kind == kind]
        if not rows:
            continue
        lines.append(f"\n## {kind}\n")
        lines.append("| m | n | k | n_runs | speedups | median_speedup | all_run_positive | median_a_secs |")
        lines.append("|---|---|---|--------|----------|-----------------|-------------------|----------------|")
        for s in sorted(rows, key=lambda r: (r.m, r.n, r.k)):
            speedups_str = ",".join(f"{v:.4f}" for v in s.speedups) or "NA"
            median = s.median_speedup
            median_str = f"{median:.4f}" if median is not None else "NA"
            median_a = s.median_a_secs
            median_a_str = f"{median_a:.6e}" if median_a is not None else "NA"
            lines.append(
                f"| {s.m} | {s.n} | {s.k} | {len(s.speedups) or len(s.median_a_secs_list)} | "
                f"{speedups_str} | {median_str} | {s.all_positive} | {median_a_str} |"
            )

    if not any_undetermined:
        target_rows = [v for v in shapes.values() if v.kind == "target"]
        control_rows = [v for v in shapes.values() if v.kind == "control"]
        enough_runs = n_runs >= MIN_FORMAL_RUNS
        target_complete = len(target_rows) == 9 and all(
            len(t.speedups) == n_runs for t in target_rows
        )
        control_complete = len(control_rows) == 3 and all(
            len(c.speedups) == n_runs for c in control_rows
        )
        if not (enough_runs and target_complete and control_complete):
            reason = (
                f"n_runs={n_runs} が MIN_FORMAL_RUNS={MIN_FORMAL_RUNS} 未満のため暫定値"
                if not enough_runs
                else "対象・対照形状のいずれかが全 run に揃っていない"
            )
            lines.append(
                "\n## verdict\n\n**undetermined**"
                f"（正式な ADOPT/REJECT は {MIN_FORMAL_RUNS} run 完了後にのみ出力する。"
                f"{reason}。n_runs={n_runs}・enough_runs={enough_runs}・"
                f"target_shapes={len(target_rows)}/9・target_complete={target_complete}・"
                f"control_shapes={len(control_rows)}/3・control_complete={control_complete}）\n"
            )
        else:
            target_ok = all(
                (t.median_speedup or 0.0) >= ADOPT_TARGET_MIN_SPEEDUP and t.all_positive
                for t in target_rows
            )
            control_ok = all(
                (c.median_speedup or 0.0) >= ADOPT_CONTROL_MIN_SPEEDUP for c in control_rows
            )
            verdict = "ADOPT" if (target_ok and control_ok) else "REJECT"
            lines.append(f"\n## verdict\n\n**{verdict}**"
                          f"（target_shapes={len(target_rows)}/9・target_ok={target_ok}・"
                          f"control_shapes={len(control_rows)}/3・control_ok={control_ok}）\n")
    else:
        lines.append("\n## verdict\n\n**undetermined**\n")

    return "\n".join(lines) + "\n"


def _make_phase0_lines(gated: bool = True) -> list[str]:
    """自己検証用の擬似 phase0 行（全形状 stable=true・checksum 固定値・
    `a_vs_at_fail_count=0`）を生成する（`self_test` からのみ呼ばれる）。
    """
    out: list[str] = []
    for mn in _TARGET_MN:
        for k in _K_LIST:
            out.append(
                f"phase0 kind=target m={mn} n={mn} k={k} classic_stable=true "
                "target_tile_stable=true splitk_stable=true "
                "checksum_a=1.000000e2 checksum_at=1.000000e2 checksum_b=1.100000e2 "
                "a_vs_b_fail_count=3 a_vs_at_fail_count=0"
            )
    for k in _K_LIST:
        out.append(
            f"phase0 kind=control m={_CONTROL_MN} n={_CONTROL_MN} k={k} classic_stable=true"
        )
    return out


def self_test() -> None:
    sample = (
        "splitk_ab kind=target m=32 n=32 k=2048 partitions=32 median_a_secs=2e-4 "
        "median_b_secs=1e-4 speedup=2.0000 spread_a=0.1 spread_b=0.1\n"
        "splitk_ab kind=control m=256 n=256 k=2048 partitions=NA median_a_secs=5e-4 "
        "median_b_secs=5e-4 speedup=1.0000 spread_a=0.1 spread_b=0.1\n"
        "splitk_ab kind=floor m=32 n=32 k=64 partitions=NA median_a_secs=8e-5 "
        "median_b_secs=NA speedup=NA spread_a=0.1 spread_b=NA\n"
    )
    rows, phase0_rows, undetermined = parse_log(sample)
    assert len(rows) == 3, rows
    assert not phase0_rows
    assert not undetermined

    # gate_mode 引数のバリデーション。
    try:
        aggregate([sample], gate_mode="bogus")
        raise AssertionError("bogus gate_mode で例外が出なかった")
    except ValueError:
        pass

    def make_run(
        target_mn: tuple[int, ...] = _TARGET_MN,
        control_mn: int = _CONTROL_MN,
        k_list: tuple[int, ...] = _K_LIST,
        drop: tuple[str, int, int, int] | None = None,
        dup: tuple[str, int, int, int] | None = None,
        env_mode: str = "record_only",
        load_avg_one: str = "1.23",
        include_phase0: bool = True,
        phase0_override: list[str] | None = None,
    ) -> str:
        """1 run 分の擬似ログを生成する（`gemm_splitk_ab_bench.rs::main` の
        実出力を模す）。`env_mode` は `record_only`／`gated`／`none`
        （env_guard 出力自体を省略。旧版フォーマット互換の異常系検証用）。
        """
        lines_: list[str] = []
        for mn in target_mn:
            for k in k_list:
                if drop == ("target", mn, mn, k):
                    continue
                line = (
                    f"splitk_ab kind=target m={mn} n={mn} k={k} partitions=4 "
                    "median_a_secs=2e-4 median_b_secs=1e-4 speedup=2.0000 "
                    "spread_a=0.1 spread_b=0.1"
                )
                lines_.append(line)
                if dup == ("target", mn, mn, k):
                    lines_.append(line)
        for k in k_list:
            if drop == ("control", control_mn, control_mn, k):
                continue
            line = (
                f"splitk_ab kind=control m={control_mn} n={control_mn} k={k} "
                "partitions=NA median_a_secs=5e-4 median_b_secs=5e-4 "
                "speedup=1.0000 spread_a=0.1 spread_b=0.1"
            )
            lines_.append(line)
            if dup == ("control", control_mn, control_mn, k):
                lines_.append(line)

        if include_phase0:
            lines_.extend(phase0_override if phase0_override is not None else _make_phase0_lines())

        if env_mode == "record_only":
            lines_.append("== env_guard(gemm_splitk_ab_bench) ==")
            lines_.append("env_guard_mode=record_only")
            lines_.append(
                f"env_guard_load_avg one={load_avg_one} five=1.00 fifteen=1.00 "
                "max_1min=NA verdict=undetermined"
            )
            lines_.append("env_guard_result=record_only")
        elif env_mode == "gated":
            lines_.append("== env_guard(gemm_splitk_ab_bench) ==")
            lines_.append("env_guard_mode=gated")
            lines_.append(
                "env_guard_overall verdict=pass attempts=1 max_attempts=10 total_wait_secs=0.00"
            )
            lines_.append("env_guard_result=pass")
        return "\n".join(lines_) + "\n"

    # 5 run 完全（record_only）→ ADOPT。
    complete_runs = [make_run() for _ in range(5)]
    shapes, any_undetermined, violations = aggregate(complete_runs, gate_mode="record_only")
    assert not any_undetermined, violations
    assert not violations
    md = render_markdown(shapes, any_undetermined, len(complete_runs), violations, "record_only")
    assert "**ADOPT**" in md, md

    # gated ログを record_only モードへ渡すと違反（ログの取り違え検出）。
    gated_runs = [make_run(env_mode="gated") for _ in range(5)]
    g_shapes, g_undetermined, g_violations = aggregate(gated_runs, gate_mode="record_only")
    assert g_undetermined
    assert any("record_only が見つからない" in v or "取り違え" in v for v in g_violations), g_violations

    # record_only ログを gated モードへ渡しても違反。
    ro_shapes, ro_undetermined, ro_violations = aggregate(complete_runs, gate_mode="gated")
    assert ro_undetermined
    assert any("gated が見つからない" in v or "取り違え" in v for v in ro_violations), ro_violations

    # env_guard_load_avg の one= が数値でない（欠落）場合は record_only でも違反。
    no_load_avg_run = make_run().replace(
        "env_guard_load_avg one=1.23 five=1.00 fifteen=1.00 max_1min=NA verdict=undetermined",
        "env_guard_load_avg one=NA five=NA fifteen=NA max_1min=NA verdict=undetermined",
    )
    na_runs = [no_load_avg_run] + [make_run() for _ in range(4)]
    na_shapes, na_undetermined, na_violations = aggregate(na_runs, gate_mode="record_only")
    assert na_undetermined
    assert any("load_avg" in v for v in na_violations), na_violations

    # フェーズ 0 の checksum が 1 run だけ異なる → undetermined。
    mismatched_phase0 = _make_phase0_lines()
    mismatched_phase0[0] = mismatched_phase0[0].replace(
        "checksum_a=1.000000e2", "checksum_a=9.999999e1"
    )
    mismatched_runs = [make_run() for _ in range(4)] + [
        make_run(phase0_override=mismatched_phase0)
    ]
    cs_shapes, cs_undetermined, cs_violations = aggregate(mismatched_runs, gate_mode="record_only")
    assert cs_undetermined
    assert any("checksum_a" in v and "不一致" in v for v in cs_violations), cs_violations

    # フェーズ 0 の stable=false → undetermined。
    unstable_phase0 = _make_phase0_lines()
    unstable_phase0[0] = unstable_phase0[0].replace("splitk_stable=true", "splitk_stable=false")
    unstable_runs = [make_run() for _ in range(4)] + [
        make_run(phase0_override=unstable_phase0)
    ]
    us_shapes, us_undetermined, us_violations = aggregate(unstable_runs, gate_mode="record_only")
    assert us_undetermined
    assert any("splitk_stable" in v for v in us_violations), us_violations

    # 3 run（完全でも 5 未満）→ undetermined（MIN_FORMAL_RUNS）。
    three_runs = [make_run() for _ in range(3)]
    t_shapes, t_undetermined, t_violations = aggregate(three_runs, gate_mode="record_only")
    assert not t_undetermined
    assert not t_violations
    t_md = render_markdown(t_shapes, t_undetermined, len(three_runs), t_violations, "record_only")
    assert "**undetermined**" in t_md, t_md
    assert "MIN_FORMAL_RUNS" in t_md, t_md

    # --monitor-logs の集計（load1 の min/median/max）。
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "run1_monitor.log")
        with open(p, "w", encoding="utf-8") as f:
            f.write(
                "2026-09-10T00:00:00Z load1=1.00 load5=1.00 load15=1.00\n"
                "2026-09-10T00:00:10Z load1=3.00 load5=1.50 load15=1.00\n"
                "2026-09-10T00:00:20Z load1=2.00 load5=1.50 load15=1.00\n"
            )
        summaries = summarize_monitor_logs([p])
        assert len(summaries) == 1
        assert summaries[0].samples == 3
        assert summaries[0].load1_min == 1.0
        assert summaries[0].load1_median == 2.0
        assert summaries[0].load1_max == 3.0
        monitor_md = render_markdown(
            shapes, any_undetermined, len(complete_runs), violations, "record_only", summaries
        )
        assert "負荷推移" in monitor_md
        assert "run1_monitor.log" in monitor_md

    print("self-test OK", file=sys.stderr)


def main() -> None:
    argv = sys.argv[1:]
    if "--self-test" in argv:
        self_test()
        return

    gate_mode = "gated"
    monitor_paths: list[str] = []
    paths: list[str] = []
    for arg in argv:
        if arg.startswith("--gate-mode="):
            gate_mode = arg.split("=", 1)[1]
        elif arg.startswith("--monitor-logs="):
            monitor_paths = [p for p in arg.split("=", 1)[1].split(",") if p]
        elif arg == "--self-test":
            continue
        else:
            paths.append(arg)

    if gate_mode not in GATE_MODES:
        print(f"--gate-mode は {GATE_MODES} のいずれかを指定する: {gate_mode}", file=sys.stderr)
        sys.exit(1)

    if not paths:
        print(
            "usage: aggregate.py [--gate-mode=gated|record_only] "
            "[--monitor-logs=a.log,b.log,...] run1.log run2.log ... [--self-test]",
            file=sys.stderr,
        )
        sys.exit(1)

    logs = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            logs.append(f.read())

    shapes, any_undetermined, violations = aggregate(logs, gate_mode=gate_mode)
    monitor_summaries = summarize_monitor_logs(monitor_paths) if monitor_paths else None
    print(
        render_markdown(
            shapes, any_undetermined, len(paths), violations, gate_mode, monitor_summaries
        )
    )


if __name__ == "__main__":
    main()
