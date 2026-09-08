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

# `phase1_gpu_host_stats size=<N> rounds=<M> ...`（1 サイズの計測が完了
# した時点で 1 回だけ出力される完了行。`format_gpu_host_size_line`・
# `gemm_transpose_route_ab_bench.rs:753`）。`size=`／`rounds=` は常に
# 先頭の空白区切りトークンとして現れる（後続の
# `kernel_gpu_round_medians_secs=`／`wall_round_medians_secs=` はカンマ
# 区切りリストのため単純な空白トークナイズでは壊れるが、先頭 2 トークン
# の抽出には影響しない）。
STATS_RE = re.compile(r"^phase1_gpu_host_stats\s+size=(?P<size>\d+)\s+rounds=(?P<rounds>\d+)\b")

# 1 サイズあたりの計画ラウンド数（`gemm_transpose_route_ab_bench.rs`
# `const ROUNDS: usize = 10`。フェーズ 1／2 共通のコンパイル時定数——
# codex-review 指摘・PR #1457: 収集できた `phase1_gpu_host_round` 行数
# から期待ラウンド数を逆算していたため、計測が先頭から連番のまま途中で
# 打ち切られたログ（例: round=0..2 の 3 行のみ）でも
# `round_idxs == range(len(rounds))` の連番整合性チェックを通過し
# 「完全な系列」として誤って確定的な帰属を出しうる不整合があった。
# 期待ラウンド数を外部定数として持ち、かつ当該サイズの完了行
# （`phase1_gpu_host_stats`）自体の有無を検証することで、系列途中
# 打ち切りを NA へ倒す）。
EXPECTED_ROUNDS = 10


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


def parse_required_float_or_na(v: str) -> float | None:
    """`kernel_gpu_median_secs`/`commit_wait_minus_gpu_median_secs` 等、
    Rust 側で `valid=false` の場合にのみ `NA` を出力する契約のフィールド
    用パーサ（`format_gpu_host_round_line`・`gemm_transpose_route_ab_
    bench.rs:596` 参照）。文字列 `"NA"` は正当な欠損（`None`）として
    受理するが、キー自体が無い場合は呼び出し元の `kv[...]`（`.get` では
    ない）で `KeyError` になり、数値にもならず `"NA"` でもない値は
    `ValueError` を送出する——いずれも他の必須フィールドと同じ
    `except (KeyError, ValueError)` で `incomplete_sizes` へ倒れる
    （codex-review 指摘・PR #1457: 従来の `parse_float_or_na` は
    パース不能値も無条件に `None` へ変換していたため、`valid=true` の
    行で本来 `incomplete` とすべき欠損・破損を静かに見逃していた）。
    """
    if v == "NA":
        return None
    return float(v)


@dataclass
class RoundSample:
    size: int
    round_idx: int
    valid: bool
    wall_median_secs: float
    kernel_gpu_median_secs: float | None
    commit_wait_median_secs: float
    # Rust 側（`aggregate_gpu_host_round`）が**ラウンド内の iters 個の
    # サンプルごとに** `commit_wait - kernel_gpu` を計算してから中央値化
    # した値（`commit_wait_median_secs - kernel_gpu_median_secs` とは
    # 一般に一致しない。中央値の非線形性）。`kernel_gpu_median_secs` と
    # 同じ理由で `valid=false` のときのみ `None`
    # （`gemm_transpose_route_ab_bench.rs:563` 周辺参照）。
    # host_subclass 判定はラウンド**間**の中央値化が必要なため、Python
    # 側で再計算せずこの記録済み値を使う（codex-review 指摘・PR #1457）。
    commit_wait_minus_gpu_median_secs: float | None
    upload_median_secs: float
    alloc_median_secs: float
    encode_median_secs: float
    readback_median_secs: float


@dataclass
class RunLog:
    label: str
    rounds: list[RoundSample] = field(default_factory=list)
    min_warmup_override_secs: int | None = None
    # 解析失敗（`KeyError`/`ValueError`）した行のうち `size` だけは救出
    # できたもの（codex-review 指摘・PR #1457: 欠落・解析失敗を含む
    # サイズを黙って除外せず NA として明示するための帰属先集合）。
    incomplete_sizes: set[int] = field(default_factory=set)
    # `phase1_gpu_host_stats`（完了行）が観測できたサイズ → その行が
    # 報告した `rounds=` 値（codex-review 指摘・PR #1457。完了行自体が
    # 存在しないサイズは「計測が完了する前にログが打ち切られた」ことの
    # 直接証拠であり、収集できた `phase1_gpu_host_round` 行数だけからは
    # 判別できない）。
    completed_size_rounds: dict[int, int] = field(default_factory=dict)

    def rounds_for_size(self, size: int) -> list[RoundSample]:
        return sorted(
            (r for r in self.rounds if r.size == size), key=lambda r: r.round_idx
        )

    def is_complete_size(self, size: int) -> bool:
        """このサイズの判定材料が「完全な系列」と扱えるかを検査する
        （codex-review 指摘・PR #1457）。以下すべてを満たす場合のみ True:
        - 解析失敗行から `size` のみ救出されたケースが無い
          （`incomplete_sizes` 非該当）
        - 完了行（`phase1_gpu_host_stats`）が観測できている
        - 収集できたラウンド数が計画ラウンド数（`EXPECTED_ROUNDS`）と
          一致する
        - 収集できたラウンド数が完了行の `rounds=` 値と一致する
          （完了行とラウンド行が異なるログ破損を検出する代理指標）
        """
        if size in self.incomplete_sizes:
            return False
        declared = self.completed_size_rounds.get(size)
        if declared is None:
            return False
        n = len(self.rounds_for_size(size))
        return n == EXPECTED_ROUNDS and n == declared


def load_run_log(path: str) -> RunLog:
    label = path.split("/")[-1]
    run = RunLog(label=label)
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("phase1_min_warmup_override_secs="):
                run.min_warmup_override_secs = int(line.split("=", 1)[1])
                continue
            m_stats = STATS_RE.match(line)
            if m_stats:
                run.completed_size_rounds[int(m_stats.group("size"))] = int(
                    m_stats.group("rounds")
                )
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
                        # `kv[...]`（`.get` ではない）でキー欠落を
                        # `KeyError` として検出し、`parse_required_float_
                        # or_na` で `"NA"` 以外のパース不能値を
                        # `ValueError` として検出する（codex-review
                        # 指摘・PR #1457）。いずれも下の except で
                        # `incomplete_sizes` へ倒れる。
                        kernel_gpu_median_secs=parse_required_float_or_na(
                            kv["kernel_gpu_median_secs"]
                        ),
                        commit_wait_median_secs=float(kv["commit_wait_median_secs"]),
                        commit_wait_minus_gpu_median_secs=parse_required_float_or_na(
                            kv["commit_wait_minus_gpu_median_secs"]
                        ),
                        upload_median_secs=float(kv["upload_median_secs"]),
                        alloc_median_secs=float(kv["alloc_median_secs"]),
                        encode_median_secs=float(kv["encode_median_secs"]),
                        readback_median_secs=float(kv["readback_median_secs"]),
                    )
                )
            except (KeyError, ValueError):
                # 不完全な行（途中打ち切りログ等）はラウンドとしては
                # 追加しないが、`size` だけは救出できる場合が多いため
                # `incomplete_sizes` へ記録し、当該サイズの判定を
                # NA へ倒す（codex-review 指摘・PR #1457: 以前は黙って
                # 除外していたため、欠落ラウンドを含むログからでも
                # 確定的な帰属〈GPU側/host側〉が出てしまっていた）。
                size_str = kv.get("size")
                if size_str is not None:
                    try:
                        run.incomplete_sizes.add(int(size_str))
                    except ValueError:
                        pass
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


def classify_cell(
    rounds: list[RoundSample], incomplete: bool = False
) -> CellVerdict | None:
    """計画 §3.4 のスパイク所在判定を 1 サイズ分適用する。

    NA（fail-closed）で帰属しない条件（計画 §3.4 に加え、
    codex-review 指摘・PR #1457 で追加）:
    - `valid=false` のラウンドが 1 つでもある
    - `incomplete=True`（解析失敗行から `size` のみ救出できたケースを
      含む。呼び出し元が `RunLog.incomplete_sizes` から渡す）
    - ラウンドの `round_idx` 列が 0 始まりの連番として揃っていない
      （欠落・重複を検出する。期待ラウンド数を外部から与えられない
      ため、収集できたラウンドの `round_idx` 集合が
      `{0, 1, ..., len(rounds)-1}` と一致することを整合性の代理指標
      とする）
    """
    if not rounds:
        return None
    size = rounds[0].size
    if incomplete or any(not r.valid for r in rounds):
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

    round_idxs = sorted(r.round_idx for r in rounds)
    if round_idxs != list(range(len(rounds))):
        # ラウンド欠落・重複あり（例: 先頭ラウンドが丸ごと欠けている
        # 場合、`round_idxs` は `{1, 2, ...}` のようになり
        # `range(len(rounds))` と一致しない）。以前はここを検査せず
        # 配列位置をそのまま `round_idx` として扱っていたため、先頭
        # ラウンド欠落時に誤って `r*=0`（実際には別ラウンド）と判定
        # しうる不整合があった。
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
    r_star_pos = max(range(len(rounds)), key=lambda i: wall_vals[i])
    r_star_round = rounds[r_star_pos]
    # 出力する r* は配列位置ではなく元の `round_idx`
    # （codex-review 指摘・PR #1457）。連番整合性チェックを通過した後
    # であれば両者は数値として一致するが、意味上は常に「元の
    # round_idx」を報告する契約とし、チェック方式が将来変わっても
    # 出力契約が揺らがないようにする。
    r_star = r_star_round.round_idx

    wall_med = median(wall_vals)
    kgpu_vals = [r.kernel_gpu_median_secs for r in rounds]
    kgpu_med = median([v for v in kgpu_vals if v is not None])
    upload_vals = [r.upload_median_secs for r in rounds]
    alloc_vals = [r.alloc_median_secs for r in rounds]
    encode_vals = [r.encode_median_secs for r in rounds]
    readback_vals = [r.readback_median_secs for r in rounds]

    upload_med = median(upload_vals)
    alloc_med = median(alloc_vals)
    encode_med = median(encode_vals)
    readback_med = median(readback_vals)

    delta_wall = r_star_round.wall_median_secs - wall_med if wall_med is not None else 0.0
    delta_kgpu = None
    if r_star_round.kernel_gpu_median_secs is not None and kgpu_med is not None:
        delta_kgpu = r_star_round.kernel_gpu_median_secs - kgpu_med

    # `commit_wait_minus_gpu` の基準値は、ログに既に記録済みの
    # `commit_wait_minus_gpu_median_secs`（Rust 側 `aggregate_gpu_host_
    # round` がラウンド内の iters サンプルごとに commit_wait -
    # kernel_gpu を計算してから中央値化した値。§5.5 契約）の
    # ラウンド**間**中央値であって、`r.commit_wait_median_secs -
    # r.kernel_gpu_median_secs`（ラウンド中央値どうしの差分。中央値の
    # 非線形性により記録済み値と一般に一致しない）を Python 側で
    # 再計算した値ではない（codex-review 指摘・PR #1457）。
    commit_minus_gpu_series = [
        r.commit_wait_minus_gpu_median_secs
        for r in rounds
        if r.commit_wait_minus_gpu_median_secs is not None
    ]
    commit_minus_gpu_med = median(commit_minus_gpu_series)
    delta_commit_minus_gpu = None
    if (
        r_star_round.commit_wait_minus_gpu_median_secs is not None
        and commit_minus_gpu_med is not None
    ):
        delta_commit_minus_gpu = (
            r_star_round.commit_wait_minus_gpu_median_secs - commit_minus_gpu_med
        )

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
        v = classify_cell(rounds, incomplete=not run.is_complete_size(size))
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
            # codex-review 指摘（PR #1457）・Cursor Bugbot 指摘（同 PR）:
            # 以前は `incomplete` を渡していなかったため、
            # `render_run_table` が NA にする欠損・途中打ち切りサイズが
            # ここでは有効セルとして数えられ、両表の間で矛盾していた。
            v = classify_cell(rounds, incomplete=not run.is_complete_size(size))
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
        # Rust 側が iters サンプルごとに計算してから中央値化した値
        # （commit_wait_median_secs - kernel_gpu_median_secs とは別の
        # 記録済みフィールド。ここでは合成データのため両者を一致させて
        # 単純化する）。
        commit_wait_minus_gpu_median_secs="0.001",
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
            kv["commit_wait_minus_gpu_median_secs"] = "0.001"
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

    # 追加テスト（codex-review 指摘・PR #1457）: 先頭ラウンド（round=0）
    # が丸ごと欠落したログ（round=1..4 の 4 ラウンドのみ）を渡すと、
    # 以前は配列位置 0 を r*=0 として誤って扱いうる不整合があった。
    # 現在は round_idx の連番整合性チェックにより NA を返すことを検証
    # する。
    missing_head_lines = []
    for i, wall in [(1, 0.02), (2, 0.05), (3, 0.02), (4, 0.02)]:
        kv = dict(base)
        kv["round"] = str(i)
        kv["iters"] = "20"
        kv["wall_median_secs"] = str(wall)
        missing_head_lines.append(
            "phase1_gpu_host_round " + " ".join(f"{k}={v}" for k, v in kv.items())
        )
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".log", delete=False, encoding="utf-8"
    ) as f:
        f.write("\n".join(missing_head_lines) + "\n")
        path2 = f.name
    try:
        run2 = load_run_log(path2)
        assert len(run2.rounds) == 4
        v2 = classify_cell(run2.rounds_for_size(1024))
        assert v2 is not None
        assert v2.attribution == "NA", (
            f"先頭ラウンド欠落時は round_idx 連番不整合により NA のはず: {v2.attribution}"
        )
    finally:
        os.unlink(path2)

    # 追加テスト（codex-review 指摘・PR #1457）: `commit_wait_minus_gpu`
    # の基準値は「ラウンドごとの (commit_wait - kernel_gpu) の中央値」
    # であって「commit_wait の中央値 - kernel_gpu の中央値」ではない
    # （一般に一致しない）ことを、両者が異なる値になるよう選んだ合成
    # ログで検証する。
    #
    # commit = [0.011, 0.020, 0.005]  kgpu = [0.010, 0.010, 0.001]
    # 各ラウンドの差分 diff_i = commit_i - kgpu_i = [0.001, 0.010, 0.004]
    #   median(diff_i) = 0.004  （系列の中央値。新実装）
    # median(commit) - median(kgpu) = 0.011 - 0.010 = 0.001
    #   （中央値同士の差分。旧実装。系列の中央値 0.004 と異なる）
    # r*（wall 最大）はラウンド 1: diff = 0.010
    #   新実装の delta = 0.010 - 0.004 = 0.006
    #   旧実装なら       delta = 0.010 - 0.001 = 0.009
    diff_lines = []
    diff_base = dict(base)
    # 4 要素目は Rust 側が記録済みの `commit_wait_minus_gpu_median_secs`
    # を模す（このテストでは合成 1 ラウンド 1 サンプル相当のため
    # commit_v - kgpu_v と一致させている。python 側が再計算するのでは
    # なくこの記録済み値をそのまま読む契約を検証する。codex-review
    # 指摘・PR #1457）。
    triples = [
        (0.02, "0.011", "0.010", "0.001"),
        (0.05, "0.020", "0.010", "0.010"),  # r*: wall 最大
        (0.02, "0.005", "0.001", "0.004"),
    ]
    for i, (wall_v, commit_v, kgpu_v, diff_v) in enumerate(triples):
        kv = dict(diff_base)
        kv["round"] = str(i)
        kv["iters"] = "20"
        kv["wall_median_secs"] = str(wall_v)
        kv["commit_wait_median_secs"] = commit_v
        kv["kernel_gpu_median_secs"] = kgpu_v
        kv["commit_wait_minus_gpu_median_secs"] = diff_v
        diff_lines.append(
            "phase1_gpu_host_round " + " ".join(f"{k}={v}" for k, v in kv.items())
        )
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".log", delete=False, encoding="utf-8"
    ) as f:
        f.write("\n".join(diff_lines) + "\n")
        path3 = f.name
    try:
        run3 = load_run_log(path3)
        rounds3 = run3.rounds_for_size(1024)
        v3 = classify_cell(rounds3)
        assert v3 is not None
        assert v3.r_star == 1, f"r* はラウンド 1 のはず: {v3.r_star}"
        assert v3.attribution == "host側", (
            f"kernel_gpu は横ばいなので host 側のはず: {v3.attribution}"
        )
        assert v3.host_subclass == "commit_wait_minus_gpu", (
            f"host_subclass は commit_wait_minus_gpu が最大のはず: {v3.host_subclass}"
        )
    finally:
        os.unlink(path3)

    # 追加テスト（codex-review 指摘・Cursor Bugbot 指摘・PR #1457）:
    # 計測が途中終了し、先頭から連番のまま欠落したログ（round=0..2 の
    # 3 行のみ。`round_idxs == range(3)` の連番整合性チェックは通過
    # するが `EXPECTED_ROUNDS=10` に満たず、かつ完了行
    # （`phase1_gpu_host_stats`）も存在しない）を渡すと、
    # `RunLog.is_complete_size` が False を返し、`render_run_table`
    # （NA 表示）・`head_index_bias_summary`（total から除外）の両方が
    # 一貫して当該サイズを除外することを検証する（以前は
    # `head_index_bias_summary` が `incomplete` を渡さず矛盾していた）。
    truncated_lines = []
    for i, wall in enumerate([0.02, 0.02, 0.05]):
        kv = dict(base)
        kv["round"] = str(i)
        kv["iters"] = "20"
        kv["wall_median_secs"] = str(wall)
        truncated_lines.append(
            "phase1_gpu_host_round " + " ".join(f"{k}={v}" for k, v in kv.items())
        )
    # 完了行（`phase1_gpu_host_stats`）は意図的に出力しない
    # （計測打ち切りを模擬する）。
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".log", delete=False, encoding="utf-8"
    ) as f:
        f.write("\n".join(truncated_lines) + "\n")
        path4 = f.name
    try:
        run4 = load_run_log(path4)
        assert len(run4.rounds_for_size(1024)) == 3
        assert not run4.is_complete_size(1024), (
            "完了行が無く EXPECTED_ROUNDS 未満のため incomplete のはず"
        )
        table_text = render_run_table(run4)
        row1024 = [
            line for line in table_text.splitlines() if line.strip().startswith("| 1024 |")
        ]
        assert len(row1024) == 1
        # 列: | size | r* | gpu_share | attribution | host_subclass | ... |
        attribution_col = row1024[0].split("|")[4].strip()
        assert attribution_col == "NA", (
            f"render_run_table は打ち切りサイズを NA 表示するはず: {row1024[0]}"
        )
        bias_text = head_index_bias_summary([run4])
        # total_cells 列（3 列目）が 0 であること
        # （`head_index_bias_summary` も同じ incomplete 判定で size を
        # 除外していることの確認。以前は `classify_cell` に `incomplete`
        # を渡さず `total` が 1 のまま数えられていた）。
        data_row = [
            line for line in bias_text.splitlines() if line.startswith(f"| {run4.label}")
        ]
        assert len(data_row) == 1
        assert data_row[0].split("|")[3].strip() == "0", (
            f"打ち切りサイズは total_cells から除外されるはず: {data_row[0]}"
        )
    finally:
        os.unlink(path4)

    # 完了行はあるが `rounds=` 値と実際の収集行数が食い違う（ログ破損の
    # 代理検出）場合も incomplete 扱いになることを検証する。
    mismatched_lines = list(truncated_lines)  # size=1024 の 3 ラウンド
    mismatched_lines.append("phase1_gpu_host_stats size=1024 rounds=10 valid=3")
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".log", delete=False, encoding="utf-8"
    ) as f:
        f.write("\n".join(mismatched_lines) + "\n")
        path5 = f.name
    try:
        run5 = load_run_log(path5)
        assert run5.completed_size_rounds.get(1024) == 10
        assert not run5.is_complete_size(1024), (
            "完了行の rounds= と実収集行数が食い違うため incomplete のはず"
        )
    finally:
        os.unlink(path5)

    # 追加テスト（codex-review 指摘・PR #1457）: `valid=true` の行で
    # `kernel_gpu_median_secs` キー自体が欠落した場合（ログ破損・
    # フォーマット不整合の想定）、以前は `parse_float_or_na` が黙って
    # `None` へ変換し `incomplete_sizes` に記録しなかったため、
    # 他ラウンドの GPU 中央値だけで確定的な帰属が出てしまっていた。
    # 10 ラウンド分揃った「完全な系列」（完了行の rounds= とも一致）
    # であっても、この欠落 1 行だけで当該サイズ全体が incomplete に
    # 倒れることを検証する。
    missing_key_lines = []
    for i in range(10):
        kv = dict(base)
        kv["round"] = str(i)
        kv["iters"] = "20"
        kv["wall_median_secs"] = "0.02"
        if i == 5:
            del kv["kernel_gpu_median_secs"]
        missing_key_lines.append(
            "phase1_gpu_host_round " + " ".join(f"{k}={v}" for k, v in kv.items())
        )
    missing_key_lines.append("phase1_gpu_host_stats size=1024 rounds=10 valid=9")
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".log", delete=False, encoding="utf-8"
    ) as f:
        f.write("\n".join(missing_key_lines) + "\n")
        path6 = f.name
    try:
        run6 = load_run_log(path6)
        assert len(run6.rounds_for_size(1024)) == 9, (
            f"kernel_gpu_median_secs 欠落行は救出されず 9 ラウンドのはず: "
            f"{len(run6.rounds_for_size(1024))}"
        )
        assert 1024 in run6.incomplete_sizes, (
            "kernel_gpu_median_secs 欠落行から size は救出され incomplete_sizes に記録されるはず"
        )
        assert not run6.is_complete_size(1024), (
            "kernel_gpu_median_secs 欠落により incomplete のはず"
        )
        v6 = classify_cell(
            run6.rounds_for_size(1024), incomplete=not run6.is_complete_size(1024)
        )
        assert v6 is not None
        assert v6.attribution == "NA", (
            f"kernel_gpu_median_secs 欠落サイズは NA のはず: {v6.attribution}"
        )
    finally:
        os.unlink(path6)

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
