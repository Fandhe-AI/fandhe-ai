#!/usr/bin/env python3
"""イシュー #1261: 分離計測ログ（`phase1_gpu_host_round`/`phase1_gpu_host_stats`/
`phase1_round_stats`）から、計画（`docs/perf/metal-gemm-transpose-tiled.md`
§5.6）で事前宣言した判定規則（§3.4）を機械的に適用し markdown 表を出力する。

Python3 標準ライブラリのみ（依存追加なし。deps-policy.md 対象外）。

入力: `1261-exclusive-run{1,2,3}.log`・`1261-warmup9s-run{1,2}.log`
（`1261-orchestrate.sh` が生成する）。

判定規則（計画 §3.4 の要約。詳細は本ファイルの各関数 docstring）:
- スパイク所在: r* = max_round_idx_wall のラウンドで、
  gpu_share = Δkernel_gpu / Δwall（Δ はラウンド別中央値との差）。
  gpu_share >= 0.5 なら「GPU 側」、それ未満なら「host 側」
  （副分類は Δ 最大の host 成分）。
- (b) ウォームアップ不足: E1〜E3 の 15 セル中 `max_round_idx_wall==0` が
  8 セル以上なら「先頭偏在あり」。W1〜W2 で偏在が消えれば支持。
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field

ROUND_RE = re.compile(r"^phase1_gpu_host_round\s+(.*)$")
# `resolved_cfg` の値は `TileConfig` の `{:?}`（Debug）フォーマットのため
# `TileConfig { bm: 64, bn: 64, ... }` のように**内部に空白を含む**——
# 単純な空白区切りトークナイズでは壊れるため、`resolved_cfg=` から末尾の
# ` valid=(true|false)` 直前までを 1 つの値として先に切り出す
# （`format_gpu_host_round_line` ドキュメンテーションコメント参照。
# `gemm_transpose_route_ab_bench.rs:596`）。
CFG_TAIL_RE = re.compile(r"resolved_cfg=(?P<cfg>.*) valid=(?P<valid>true|false)\s*$")


def parse_kv_line(rest: str) -> dict[str, str]:
    """`key=value` トークン列（`resolved_cfg`/`valid` を除き値に空白を
    含まない前提。空白区切り）を辞書へ変換する純関数。
    `format_gpu_host_round_line` の出力形式に対応する
    （`gemm_transpose_route_ab_bench.rs` 側のドキュメンテーション
    コメント参照）。"""
    kv: dict[str, str] = {}
    m_cfg = CFG_TAIL_RE.search(rest)
    if m_cfg:
        kv["resolved_cfg"] = m_cfg.group("cfg")
        kv["valid"] = m_cfg.group("valid")
        rest = rest[: m_cfg.start()]
    for token in rest.split():
        if "=" not in token:
            continue
        k, v = token.split("=", 1)
        kv[k] = v
    return kv


def parse_float_or_na(v: str) -> float | None:
    if v == "NA":
        return None
    try:
        return float(v)
    except ValueError:
        return None


@dataclass
class RoundSample:
    size: int
    round_idx: int
    valid: bool
    wall_median_secs: float
    kernel_gpu_median_secs: float | None
    commit_wait_median_secs: float
    upload_median_secs: float
    alloc_median_secs: float
    encode_median_secs: float
    readback_median_secs: float


@dataclass
class RunLog:
    label: str
    rounds: list[RoundSample] = field(default_factory=list)
    min_warmup_override_secs: int | None = None

    def rounds_for_size(self, size: int) -> list[RoundSample]:
        return sorted(
            (r for r in self.rounds if r.size == size), key=lambda r: r.round_idx
        )


def load_run_log(path: str) -> RunLog:
    label = path.split("/")[-1]
    run = RunLog(label=label)
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("phase1_min_warmup_override_secs="):
                run.min_warmup_override_secs = int(line.split("=", 1)[1])
                continue
            m = ROUND_RE.match(line)
            if not m:
                continue
            kv = parse_kv_line(m.group(1))
            try:
                run.rounds.append(
                    RoundSample(
                        size=int(kv["size"]),
                        round_idx=int(kv["round"]),
                        valid=(kv.get("valid") == "true"),
                        wall_median_secs=float(kv["wall_median_secs"]),
                        kernel_gpu_median_secs=parse_float_or_na(
                            kv.get("kernel_gpu_median_secs", "NA")
                        ),
                        commit_wait_median_secs=float(kv["commit_wait_median_secs"]),
                        upload_median_secs=float(kv["upload_median_secs"]),
                        alloc_median_secs=float(kv["alloc_median_secs"]),
                        encode_median_secs=float(kv["encode_median_secs"]),
                        readback_median_secs=float(kv["readback_median_secs"]),
                    )
                )
            except (KeyError, ValueError):
                # 不完全な行（途中打ち切りログ等）は集計対象から除外する
                # （fail-closed。判定は残りのラウンドのみで行う）。
                continue
    return run


def median(values: list[float]) -> float | None:
    if not values:
        return None
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2.0


@dataclass
class CellVerdict:
    size: int
    r_star: int
    gpu_share: float | None
    attribution: str  # "GPU側" | "host側" | "NA"
    host_subclass: str | None
    delta_wall: float
    delta_kernel_gpu: float | None
    kernel_gpu_occupancy: float | None  # kernel_gpu_median / wall_median


def classify_cell(rounds: list[RoundSample]) -> CellVerdict | None:
    """計画 §3.4 のスパイク所在判定を 1 サイズ分適用する。

    `valid=false` のラウンドが 1 つでもあれば NA を返す（fail-closed。
    計画 §3.4「valid=false ラウンドを含むサイズは NA で帰属しない」）。
    """
    if not rounds:
        return None
    size = rounds[0].size
    if any(not r.valid for r in rounds):
        return CellVerdict(
            size=size,
            r_star=-1,
            gpu_share=None,
            attribution="NA",
            host_subclass=None,
            delta_wall=0.0,
            delta_kernel_gpu=None,
            kernel_gpu_occupancy=None,
        )

    wall_vals = [r.wall_median_secs for r in rounds]
    r_star = max(range(len(rounds)), key=lambda i: wall_vals[i])
    r_star_round = rounds[r_star]

    wall_med = median(wall_vals)
    kgpu_vals = [r.kernel_gpu_median_secs for r in rounds]
    kgpu_med = median([v for v in kgpu_vals if v is not None])
    commit_vals = [r.commit_wait_median_secs for r in rounds]
    upload_vals = [r.upload_median_secs for r in rounds]
    alloc_vals = [r.alloc_median_secs for r in rounds]
    encode_vals = [r.encode_median_secs for r in rounds]
    readback_vals = [r.readback_median_secs for r in rounds]

    commit_med = median(commit_vals)
    upload_med = median(upload_vals)
    alloc_med = median(alloc_vals)
    encode_med = median(encode_vals)
    readback_med = median(readback_vals)

    delta_wall = r_star_round.wall_median_secs - wall_med if wall_med is not None else 0.0
    delta_kgpu = None
    if r_star_round.kernel_gpu_median_secs is not None and kgpu_med is not None:
        delta_kgpu = r_star_round.kernel_gpu_median_secs - kgpu_med

    delta_commit_minus_gpu = None
    if (
        r_star_round.kernel_gpu_median_secs is not None
        and kgpu_med is not None
        and commit_med is not None
    ):
        r_commit_minus_gpu = r_star_round.commit_wait_median_secs - r_star_round.kernel_gpu_median_secs
        med_commit_minus_gpu = commit_med - kgpu_med
        delta_commit_minus_gpu = r_commit_minus_gpu - med_commit_minus_gpu

    delta_upload = r_star_round.upload_median_secs - upload_med if upload_med is not None else 0.0
    delta_alloc = r_star_round.alloc_median_secs - alloc_med if alloc_med is not None else 0.0
    delta_encode = r_star_round.encode_median_secs - encode_med if encode_med is not None else 0.0
    delta_readback = (
        r_star_round.readback_median_secs - readback_med if readback_med is not None else 0.0
    )

    gpu_share = None
    attribution = "NA"
    if delta_wall > 0 and delta_kgpu is not None:
        gpu_share = delta_kgpu / delta_wall
        attribution = "GPU側" if gpu_share >= 0.5 else "host側"
    elif delta_wall <= 0:
        attribution = "NA"

    host_subclass = None
    if attribution == "host側":
        candidates = {
            "commit_wait_minus_gpu": delta_commit_minus_gpu
            if delta_commit_minus_gpu is not None
            else float("-inf"),
            "upload": delta_upload,
            "alloc": delta_alloc,
            "encode": delta_encode,
            "readback": delta_readback,
        }
        host_subclass = max(candidates, key=lambda k: candidates[k])

    kernel_gpu_occupancy = None
    if kgpu_med is not None and wall_med and wall_med > 0:
        kernel_gpu_occupancy = kgpu_med / wall_med

    return CellVerdict(
        size=size,
        r_star=r_star,
        gpu_share=gpu_share,
        attribution=attribution,
        host_subclass=host_subclass,
        delta_wall=delta_wall,
        delta_kernel_gpu=delta_kgpu,
        kernel_gpu_occupancy=kernel_gpu_occupancy,
    )


SIZES = [256, 512, 1024, 2048, 4096]


def render_run_table(run: RunLog) -> str:
    lines = [f"### {run.label}", ""]
    if run.min_warmup_override_secs is not None:
        lines.append(f"（`--min-warmup-secs={run.min_warmup_override_secs}` 指定）")
        lines.append("")
    lines.append(
        "| size | r* | gpu_share | attribution | host_subclass | delta_wall(s) | kernel_occupancy |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for size in SIZES:
        rounds = run.rounds_for_size(size)
        v = classify_cell(rounds)
        if v is None:
            lines.append(f"| {size} | - | - | NA(no data) | - | - | - |")
            continue
        gpu_share_str = f"{v.gpu_share:.3f}" if v.gpu_share is not None else "NA"
        occ_str = (
            f"{v.kernel_gpu_occupancy:.3f}" if v.kernel_gpu_occupancy is not None else "NA"
        )
        lines.append(
            f"| {size} | {v.r_star} | {gpu_share_str} | {v.attribution} | "
            f"{v.host_subclass or '-'} | {v.delta_wall:.4e} | {occ_str} |"
        )
    lines.append("")
    return "\n".join(lines)


def head_index_bias_summary(runs: list[RunLog]) -> str:
    """(b) ウォームアップ不足の判定材料: 各 (run, size) の
    max_round_idx_wall（= r*）が 0 に偏っているかを数える
    （計画 §3.4「先頭偏在あり」＝15 セル中 8 セル以上が r*==0）。"""
    lines = ["### 先頭偏在集計（r* == 0 のセル数）", ""]
    lines.append("| run | zero_count | total_cells |")
    lines.append("|---|---|---|")
    for run in runs:
        zero = 0
        total = 0
        for size in SIZES:
            rounds = run.rounds_for_size(size)
            v = classify_cell(rounds)
            if v is None or v.r_star < 0:
                continue
            total += 1
            if v.r_star == 0:
                zero += 1
        lines.append(f"| {run.label} | {zero} | {total} |")
    lines.append("")
    return "\n".join(lines)


def self_test() -> None:
    """合成ログでの自己テスト（計画 §4 ステップ 6）。"""
    sample_lines = []
    # size=1024: 5 ラウンド、ラウンド 2 だけ wall が突出しており
    # kernel_gpu も比例して増えている（GPU 側の想定ケース）。
    base = dict(
        size=1024,
        valid="true",
        kernel_gpu_median_secs="0.010",
        commit_wait_median_secs="0.011",
        upload_median_secs="0.001",
        alloc_median_secs="0.0005",
        encode_median_secs="0.0002",
        readback_median_secs="0.001",
        resolved_cfg="TileConfig { bm: 64, bn: 64, bk: 16, wm: 2, wn: 2, staged: false }",
    )
    for i, wall in enumerate([0.02, 0.02, 0.05, 0.02, 0.02]):
        kv = dict(base)
        kv["round"] = str(i)
        kv["iters"] = "20"
        kv["wall_median_secs"] = str(wall)
        if i == 2:
            kv["kernel_gpu_median_secs"] = "0.04"
            kv["commit_wait_median_secs"] = "0.041"
        line = "phase1_gpu_host_round " + " ".join(f"{k}={v}" for k, v in kv.items())
        sample_lines.append(line)

    import tempfile
    import os

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".log", delete=False, encoding="utf-8"
    ) as f:
        f.write("\n".join(sample_lines) + "\n")
        path = f.name
    try:
        run = load_run_log(path)
        assert len(run.rounds) == 5, f"5 ラウンド読めるはず: {len(run.rounds)}"
        v = classify_cell(run.rounds_for_size(1024))
        assert v is not None
        assert v.r_star == 2, f"r* はラウンド 2 のはず: {v.r_star}"
        assert v.attribution == "GPU側", f"kernel_gpu も比例して増えたので GPU 側のはず: {v.attribution}"
    finally:
        os.unlink(path)
    print("self-test: ok")


def main(argv: list[str]) -> int:
    if len(argv) >= 1 and argv[0] == "--self-test":
        self_test()
        return 0

    if not argv:
        print("usage: 1261-aggregate.py <log1> [<log2> ...] | --self-test", file=sys.stderr)
        return 2

    runs = [load_run_log(p) for p in argv]

    out = []
    out.append("# イシュー #1261 分離計測集計（自動生成）")
    out.append("")
    for run in runs:
        out.append(render_run_table(run))
    out.append(head_index_bias_summary(runs))
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
