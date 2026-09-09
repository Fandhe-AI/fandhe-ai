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

PR #1501 codex-review 指摘の追加是正（2026-09-09 その 2）: 上記初版の
規則 5 は集計済み Markdown（`compare_gemm_ab.py` 出力）の中央値比のみで
発火判定しており、5 run 個々の符号一貫性を検証していなかった（DGX 側が
合格でも M4 Max 側が規則 3／4 不成立のまま符号混在の計測から
ADOPT_LINUX_ONLY を返しうる契約不整合。保存済みデータの REJECT という
結論自体には影響しない）。本版は `--jsonl`（`bench-fandhe` の生 JSONL。
`node:arm:path` 形式・1 行 1 run）を追加入力とし、`parse_jsonl_medians`
で `(size, mode) -> [median_s, ...]`（run 出現順を保持）を抽出したうえで、
規則 5 は (i) run 別 on/off 比（`on_runs[i] / off_runs[i]`。生ログの run
順で対応付け）が全 run で 1.0 超（符号一貫）、かつ (ii) 生 JSONL から
算出した中央値比（`statistics.median(on) / statistics.median(off)`）が
閾値超、の両方を満たす場合のみ発火する（run 別データが未指定の場合は
Markdown 由来の中央値比のみで判定し、符号一貫性は「未確認」として
`lines` に明記する。判定不能側〈発火させない〉に倒す fail-safe）。
副次効果として、規則 2／3 の on/off 比も `--jsonl` 指定時は
Markdown 表示値（3〜4 桁丸め）ではなく生 JSONL 由来の中央値比
（丸めなし）を優先して使う（未指定時は従来どおり Markdown 値を使う）。

PR #1501 codex-review 指摘の追加是正（2026-09-09 その 4）: 規則 4
〈candle 比〉は当初、candle 側の生 median_s がこのディレクトリの成果物
として保存されていないことを理由に `compare_gemm_gate.py` 出力
Markdown の candle/fandhe 列（3 桁丸め）を on/off で再除算しており、
`RULE4_MIN_RATIO` 境界（~0.952380...）付近で表示丸めにより採否が逆転
しうる欠陥があった。`scripts/bench/framework-compare/results/raw/
results-*-1481-pzero-*.jsonl`（`framework` タグ付き生 JSONL。本イシュー
の成果物として保存済みで candle・fandhe-ai 両方の生 median_s を含む）
を `--candle-jsonl` で追加入力とし、`parse_candle_jsonl_medians`／
`_raw_candle_ratio` で丸めなしの candle 比 on/off 比を算出するよう是正
した（未指定・データ欠落時は従来どおり Markdown 丸め値へフォール
バックし、この既知の精度限界を判定ロジック中にコメントで明記する）。

使い方:
  python3 judge_rules.py \
    --layer-a-md compare_gemm_ab-dgx.md --layer-a-md compare_gemm_ab-m4max.md \
    --layer-b-md aggregate-dgx.md --layer-b-md aggregate-m4max.md \
    --gate-md dgx:off:compare_gemm_gate-dgx-off.md \
    --gate-md dgx:on:compare_gemm_gate-dgx-on.md \
    --gate-md m4max:off:compare_gemm_gate-m4max-off.md \
    --gate-md m4max:on:compare_gemm_gate-m4max-on.md \
    --jsonl dgx:off:results-dgx-...-off.fandhe-only.jsonl \
    --jsonl dgx:on:results-dgx-...-on.fandhe-only.jsonl \
    --jsonl m4max:off:results-m4max-...-off.fandhe-only.jsonl \
    --jsonl m4max:on:results-m4max-...-on.fandhe-only.jsonl \
    --candle-jsonl dgx:off:.../results/raw/results-dgx-...-off.jsonl \
    --candle-jsonl dgx:on:.../results/raw/results-dgx-...-on.jsonl \
    --candle-jsonl m4max:off:.../results/raw/results-m4max-...-off.jsonl \
    --candle-jsonl m4max:on:.../results/raw/results-m4max-...-on.jsonl \
    > judge.md

自己検証: `python3 judge_rules.py --self-test`（固定サンプル文字列に対する
パーサ・判定ロジックの単体検証。標準ライブラリのみ）。
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from dataclasses import dataclass

# ---- 事前登録した数値閾値（計測前に固定。以後変更しない） ----
RULE234_THRESHOLD = 1.05  # 規則 2 の Layer A・規則 3・規則 5 の on/off 比上限
RULE2_OPS_GEMM_THRESHOLD = 1.00  # 規則 2 の ops_gemm on/off 比上限（厳密。緩めない）
RULE4_MIN_RATIO = 1.0 / 1.05  # 規則 4（改定版）: on/off 比の下限 (~0.952380...)
CHECKSUM_OK_VALUE = "完全一致"  # 規則 1 が要求する checksum 列の合格値
# AGENTS.md の 5 回計測契約（bench-runner は 5 回計測中央値を採用する）。
# 規則 5 の run 別符号一貫性判定（PR #1501 codex-review P1 是正）は
# 両腕とも本数ちょうど 5 件の有効値を要求し、欠落・不足時は
# 「後退なし」ではなく判定不能（UNDETERMINED）に倒す。
REQUIRED_RUN_COUNT = 5

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
# `aggregate_layer_b.py`（PR #1501 codex-review P2 是正）が各行の直後に
# 出力する丸めなし raw コメント行。`repr(float)` の round-trip 表現
# （指数表記 `e±NN` を含みうる）を許容する。
LAYER_B_RAW_RE = re.compile(
    r"^<!--\s*raw\s+N=(?P<n>\d+)\s+phase=(?P<phase>[A-Za-z_]+)\s+"
    r"off_med=(?P<off>[-0-9.eE+]+)\s+on_med=(?P<on>[-0-9.eE+]+)\s+"
    r"ratio=(?P<ratio>[-0-9.eE+]+)\s*-->\s*$"
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
    """`aggregate_layer_b.py` 出力 md から {(N, phase): on/off 比} を抽出する。

    PR #1501 codex-review P2 是正: 人間可読テーブルの `on/off 比` 列は
    表示用に小数 4 桁へ丸められており、これを規則 2（`RULE2_OPS_GEMM_
    THRESHOLD=1.00` の厳密な `<=`／`<` 判定）へそのまま使うと丸め誤差で
    誤通過・誤棄却しうる。`aggregate_layer_b.py` が追加出力する
    `<!-- raw N=... phase=... ... ratio=... -->` コメント行（`repr(float)`
    による丸めなし round-trip 表現）が同じ (N, phase) に存在すれば
    そちらを優先し、無ければ従来どおりテーブル列の丸め値へフォール
    バックする（旧形式の md〈本ファイルの自己テストフィクスチャを
    含む〉との後方互換を保つため）。
    """
    out: dict[tuple[int, str], float] = {}
    raw_out: dict[tuple[int, str], float] = {}
    cur_n: int | None = None
    for line in text.splitlines():
        m_sec = LAYER_B_SECTION_RE.match(line)
        if m_sec:
            cur_n = int(m_sec.group("n"))
            continue
        m_raw = LAYER_B_RAW_RE.match(line)
        if m_raw:
            try:
                raw_out[(int(m_raw.group("n")), m_raw.group("phase"))] = float(
                    m_raw.group("ratio")
                )
            except ValueError:
                pass
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
    # raw コメント由来の丸めなし値があれば表示丸め値より優先する
    # （表示は表示専用・判定は生値、という P2 是正の核心）。
    out.update(raw_out)
    return out


def parse_jsonl_medians(text: str) -> dict[tuple[int, str], list[float]]:
    """`bench-fandhe` 生 JSONL（1 行 1 run）から {(size, mode): [median_s, ...]} を抽出する。

    行の出現順を保持する（本イシューのオーケストレーションスクリプトは
    5 回の独立プロセス起動を順に連結して 1 ファイルへ書き出すため、
    各 (size, mode) の要素は run 1〜5 の順で並ぶ。規則 5 の run 別
    符号一貫性判定〈`_check_rule5_regression`〉はこの並び順を
    run 番号の対応付けとしてそのまま使う）。パース不能な行・必要
    フィールド欠落行は静かに無視する（Layer A の `--layer-a-md` と
    同じ「壊れた行は無視してヘッダ等を誤検出しない」方針）。
    """
    out: dict[tuple[int, str], list[float]] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        try:
            size = int(d["size"])
            mode = str(d["mode"])
            median_s = float(d["median_s"])
        except (KeyError, TypeError, ValueError):
            continue
        out.setdefault((size, mode), []).append(median_s)
    return out


def parse_candle_jsonl_medians(text: str) -> dict[int, list[float]]:
    """フレームワーク横並び生 JSONL（`framework` フィールドを含む。
    `scripts/bench/framework-compare/results/raw/results-*.jsonl`）から
    candle 側（`framework == "candle"` かつ `mode == "fresh"`。
    `compare_gemm_gate.py` が「candle 比」として使う値と同じ定義）の
    `{size: [median_s, ...]}` を抽出する。

    PR #1501 codex-review P2 是正: 規則 4（改定版）は従来
    `compare_gemm_gate.py` 出力 Markdown の candle/fandhe 列（3 桁丸め）を
    on/off で再除算していたため、事前登録閾値の境界付近で表示丸めにより
    採否が逆転しうる欠陥があった（例: 真の比が閾値未満でも表示値の
    再除算では閾値を満たして見える）。本関数は同ディレクトリの成果物
    ではなく `scripts/bench/framework-compare/results/raw/` 配下の生
    JSONL（`framework` タグ付き。同一ファイルに fandhe-ai・candle 両方の
    行が混在するため `framework` で明示的に絞り込む必要がある——
    `parse_jsonl_medians` は `framework` を見ないため、この生 JSONL を
    そのまま渡すと fandhe-ai の `mode=="fresh"` 行と candle の
    `mode=="fresh"` 行が同じ (size, mode) キーへ混入してしまう）から
    candle 行のみを抽出し、`_raw_median_ratio` 系と同様に丸めなしの
    `median_s` を使えるようにする。パース不能な行・必要フィールド欠落
    行・candle 以外の行は静かに無視する（既存パーサ群と同じ方針）。
    """
    out: dict[int, list[float]] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("framework") != "candle" or d.get("mode") != "fresh":
            continue
        try:
            size = int(d["size"])
            median_s = float(d["median_s"])
        except (KeyError, TypeError, ValueError):
            continue
        out.setdefault(size, []).append(median_s)
    return out


def _raw_candle_ratio(
    off_fandhe: dict[tuple[int, str], list[float]],
    on_fandhe: dict[tuple[int, str], list[float]],
    off_candle: dict[int, list[float]],
    on_candle: dict[int, list[float]],
    n: int,
) -> float | None:
    """規則 4（改定版）: 丸めなしの生 median_s から candle 比の on/off 比を計算する。

    「candle 比」= `candle fresh median / fandhe-ai reuse median`
    （`compare_gemm_gate.py` の定義に一致）。off/on それぞれこの比を
    丸めなしで算出してから除算するため、Markdown 表示値（3 桁丸め）を
    再除算する経路（`RULE4_MIN_RATIO` 境界近傍で採否が逆転しうる）を
    経由しない。いずれかの生値が欠落、または 4 入力（fandhe-ai
    off/on・candle off/on）のいずれかがちょうど `REQUIRED_RUN_COUNT`
    （5）件の有効値を持たない場合は None を返す（呼び出し側は
    Markdown 由来の丸め値へフォールバックする）。

    PR #1501 codex-review P1 是正（2026-09-09 その 5）: 従来は 4 入力
    いずれも「非空であること」しか検証しておらず、例えば片方の腕が
    1 run しか収集できていなくても `statistics.median` は単一値を
    そのまま返すため、AGENTS.md の 5 回計測中央値契約に反した「1 run
    だけの中央値」を規則 4 の判定に使いうる契約不整合があった
    （`_raw_median_ratio` と同型の欠陥。`_raw_run_pair_ratios` は既に
    5 run 要求を実装済みだったが、本関数・`_raw_median_ratio` は対象外
    のまま残っていた）。本版は 4 入力すべてがちょうど
    `REQUIRED_RUN_COUNT` 件であることを要求し、いずれか 1 つでも
    不足・超過（欠落を含む）であれば None（判定不能。呼び出し側で
    Markdown フォールバックへ倒れる）を返す。
    """
    off_c = off_candle.get(n)
    on_c = on_candle.get(n)
    off_f = off_fandhe.get((n, "reuse"))
    on_f = on_fandhe.get((n, "reuse"))
    if not off_c or not on_c or not off_f or not on_f:
        return None
    if (
        len(off_c) != REQUIRED_RUN_COUNT
        or len(on_c) != REQUIRED_RUN_COUNT
        or len(off_f) != REQUIRED_RUN_COUNT
        or len(on_f) != REQUIRED_RUN_COUNT
    ):
        return None
    off_fandhe_median = statistics.median(off_f)
    on_fandhe_median = statistics.median(on_f)
    if off_fandhe_median == 0 or on_fandhe_median == 0:
        return None
    off_ratio = statistics.median(off_c) / off_fandhe_median
    on_ratio = statistics.median(on_c) / on_fandhe_median
    if off_ratio == 0:
        return None
    return on_ratio / off_ratio


def _raw_median_ratio(
    off_runs: dict[tuple[int, str], list[float]],
    on_runs: dict[tuple[int, str], list[float]],
    n: int,
    mode: str,
) -> float | None:
    """生 JSONL から丸めなしの on/off 中央値比を計算する（`--jsonl` 未指定時は None）。

    `statistics.median(on) / statistics.median(off)`。`compare_gemm_ab.py`
    が Markdown へ出力する `after/before` 列（3〜4 桁丸め）と同じ定義だが、
    本関数は生の `median_s` から直接計算するため丸め誤差を持ち込まない
    （PR #1501 codex-review P2: 表示用に丸めた値を再除算しない）。

    PR #1501 codex-review P1 是正（2026-09-09 その 5）: 本関数は規則
    2〜4 の判定（DGX N=2048 決定セル・対照セル・M4Max N=2048 の
    Layer A 比）に直接使われるにもかかわらず、従来は両腕とも
    「非空であること」しか検証していなかった。AGENTS.md の 5 回計測
    中央値契約は判定に使う全セルへ適用されるべきところ、`REQUIRED_
    RUN_COUNT`（5）件の完備要求は規則 5 専用の `_raw_run_pair_ratios`
    にしか実装されておらず、片腕が 1 run しか収集できていなくても
    その 1 値を「中央値」として採用し比を計算してしまう契約不整合
    があった（例: off が 5 run とも 1.0、on が本来 [1.2]×5 のところ
    後半 4 行が欠落し `[1.2]` のみ収集された場合、`statistics.
    median([1.2]) == 1.2` で比は正しく 1.2 になるが、逆に on の収集が
    `[1.0]` のみで真の 5 run 中央値が 1.2 だった場合は比が 1.0 に
    見え規則 3 を誤って通過させうる）。本版は両腕ちょうど
    `REQUIRED_RUN_COUNT` 件であることを要求し、不足・超過（欠落を
    含む）であれば None を返す（呼び出し側は Markdown 由来の丸め値へ
    フォールバックする。フォールバック自体は `--jsonl` 完全未指定時の
    既存挙動と同一で、生データが「一部だけ」揃っている場合に限り
    判定不能として弾く）。
    """
    off_list = off_runs.get((n, mode))
    on_list = on_runs.get((n, mode))
    if not off_list or not on_list:
        return None
    if len(off_list) != REQUIRED_RUN_COUNT or len(on_list) != REQUIRED_RUN_COUNT:
        return None
    return statistics.median(on_list) / statistics.median(off_list)


def _raw_run_pair_ratios(
    off_runs: dict[tuple[int, str], list[float]],
    on_runs: dict[tuple[int, str], list[float]],
    n: int,
    mode: str,
) -> list[float] | None:
    """run 番号を対応付けた on/off 比のリストを返す（規則 5 の符号一貫性判定用）。

    `parse_jsonl_medians` が保持する出現順（= run 1〜5 の順）で
    `off_runs[i]`・`on_runs[i]` を同じ run 番号の対として扱う。

    PR #1501 codex-review P1 是正: 従来は両ファイルの該当 (size, mode) の
    run 数が異なる場合に短い方（`min(len(off_list), len(on_list))`）へ
    切り詰めていたため、例えば片腕が 1 run しか無くても「5 run 符号一貫」
    相当の判定を通過しうる契約不整合があった（AGENTS.md の 5 回計測契約・
    `docs/perf/cpu-gemm-candle-gate-remeasurement.md` §20.1 の採否条件は
    両腕とも 5 run の対応関係を前提とする）。本版は両腕とも厳密に
    `REQUIRED_RUN_COUNT`（5）件の有効値を持つ場合にのみ対応付けを返し、
    どちらかが欠落・不足（解析失敗を含む）していれば None を返して
    呼び出し側に「判定不能」を伝える（「後退なし」とは区別する。
    呼び出し側 `judge()` はこの None を UNDETERMINED へ倒す）。
    """
    off_list = off_runs.get((n, mode))
    on_list = on_runs.get((n, mode))
    if not off_list or not on_list:
        return None
    if len(off_list) != REQUIRED_RUN_COUNT or len(on_list) != REQUIRED_RUN_COUNT:
        return None
    return [on_list[i] / off_list[i] for i in range(REQUIRED_RUN_COUNT)]


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
    jsonl: dict[tuple[str, str], dict[tuple[int, str], list[float]]] | None = None,
    candle_jsonl: dict[tuple[str, str], dict[int, list[float]]] | None = None,
) -> tuple[list[str], str]:
    """規則 1〜5 を機械判定し、(明細行のリスト, verdict) を返す。

    `jsonl` は `{(node, arm): {(size, mode): [median_s, ...]}}`（`--jsonl`
    引数から `parse_jsonl_medians` で構築。`arm` は "off"／"on"）。規則 2・3
    の on/off 比は `--jsonl` 未指定時のみ Markdown 由来の丸め済み比へ
    フォールバックする（`_self_test` の後方互換ケース用）が、規則 5 は
    両腕ちょうど `REQUIRED_RUN_COUNT`（5）run の対応関係を必須とし、
    `--jsonl` が未指定・不完全な場合は下記のとおり UNDETERMINED を返す
    （PR #1501 codex-review P1 是正: 5 run 未満のデータで ADOPT 系判定を
    返さない）。

    `candle_jsonl` は `{(node, arm): {size: [candle median_s, ...]}}`
    （`--candle-jsonl` 引数から `parse_candle_jsonl_medians` で構築。
    `scripts/bench/framework-compare/results/raw/results-*.jsonl` の
    `framework=="candle"` 行由来）。規則 4（改定版）は指定時、
    `compare_gemm_gate.py` 出力 Markdown の candle/fandhe 列（3 桁丸め）
    を on/off で再除算せず、生 median_s から丸めなしで candle 比・
    on/off 比を算出する（PR #1501 codex-review P2 是正: 事前登録閾値の
    境界での表示丸めによる採否逆転を避ける）。未指定・データ欠落時は
    従来どおり Markdown 由来の丸め済み比へフォールバックする。

    verdict は次の 4 値（§20.1 の場合分け (a)〜(c) をそのまま実装する）:
    - "UNDETERMINED": 規則 1（checksum 完全一致）が不成立・入力欠落、
      または規則 5 の入力（M4 Max N=2048 の両腕 `REQUIRED_RUN_COUNT`
      件の run 対応関係）が欠落・不完全（判定不能。他規則の成否に
      よらず優先する）
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
    jsonl = jsonl or {}
    candle_jsonl = candle_jsonl or {}

    # 規則 1（前提条件）: 両実機・全セルで checksum 完全一致。
    checksum_ok, checksum_lines = _check_rule1_checksum(layer_a)
    lines.extend(checksum_lines)

    # 規則 2: DGX N=2048 決定セル（DGX 専用）。
    # PR #1501 codex-review P2 是正: `--jsonl` 指定時は生 median_s から
    # 丸めなしで中央値比を再計算し、`compare_gemm_ab.py` の Markdown 表示
    # （3〜4 桁丸め）に依存しない（未指定時は従来どおり Markdown 値）。
    dgx_a = layer_a.get("dgx", {})
    dgx_b = layer_b.get("dgx", {})
    r2_layer_a_cell = dgx_a.get((2048, "reuse"))
    r2_layer_a_md = r2_layer_a_cell.ratio if r2_layer_a_cell is not None else None
    r2_layer_a_raw = _raw_median_ratio(
        jsonl.get(("dgx", "off"), {}), jsonl.get(("dgx", "on"), {}), 2048, "reuse"
    )
    r2_layer_a = r2_layer_a_raw if r2_layer_a_raw is not None else r2_layer_a_md
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
        f"規則2 DGX N=2048決定セル: layer_a(reuse)={r2_layer_a} "
        f"(md={r2_layer_a_md}, raw={r2_layer_a_raw}) (<= {RULE234_THRESHOLD}) "
        f"alloc_c比={r2_alloc_c} ops_gemm比={r2_ops_gemm} (<= {RULE2_OPS_GEMM_THRESHOLD}) "
        f"-> {'満たす' if r2_ok else '不成立'}"
    )

    # 規則 3: 対照セル（両実機 N=512/1024 の fresh/reuse 全セル）。
    # §20.1 場合分けの (a) が「DGX が規則 1〜4 を満たす」ことのみを要求
    # するため、DGX 側・M4 Max 側を分けて判定を保持する。規則 2 と同様、
    # `--jsonl` 指定時は生 median_s から丸めなしで比を再計算する。
    r3_dgx_ok = True
    r3_m4max_ok = True
    for node in ("dgx", "m4max"):
        a = layer_a.get(node, {})
        off_runs = jsonl.get((node, "off"), {})
        on_runs = jsonl.get((node, "on"), {})
        for n in (512, 1024):
            for mode in ("fresh", "reuse"):
                cell = a.get((n, mode))
                v_md = cell.ratio if cell is not None else None
                v_raw = _raw_median_ratio(off_runs, on_runs, n, mode)
                v = v_raw if v_raw is not None else v_md
                ok = v is not None and v <= RULE234_THRESHOLD
                lines.append(
                    f"規則3 対照セル {node} N={n}/{mode}: 比={v} "
                    f"(md={v_md}, raw={v_raw}) (<= {RULE234_THRESHOLD}) "
                    f"-> {'満たす' if ok else '不成立'}"
                )
                if node == "dgx":
                    r3_dgx_ok = r3_dgx_ok and ok
                else:
                    r3_m4max_ok = r3_m4max_ok and ok

    # 規則 4（改定版）: 各セル（実機×N=512/1024/2048）の candle 比 on/off 比。
    # 規則 3 と同様に DGX 側・M4 Max 側を分けて保持する。
    # PR #1501 codex-review P2 是正（2026-09-09 その 4）: 当初実装は
    # candle 側の生 median_s が本ディレクトリの成果物として保存されて
    # いない（`--jsonl` は fandhe-ai 自系列の生ログのみを持つ）ことを
    # 理由に、`compare_gemm_gate.py` 出力 Markdown の candle/fandhe 列
    # （3 桁丸め）を on/off で再除算していたが、これは表示用に丸めた比を
    # 再除算するため事前登録閾値 `RULE4_MIN_RATIO`（境界 ~0.952380...）
    # の境界付近で採否が逆転しうる欠陥だった（例: off=1.00049,
    # on=0.95251 は真の比が閾値未満でも表示値 1.000/0.953 の再除算では
    # 通過する）。`scripts/bench/framework-compare/results/raw/results-*.
    # jsonl`（`framework` タグ付き生 JSONL。本イシューの成果物として
    # 保存済み）には candle 側の生 median_s も含まれているため、
    # `--candle-jsonl` 指定時は `_raw_candle_ratio`（丸めなしの
    # candle fresh／fandhe-ai reuse 中央値から直接算出）を優先し、
    # Markdown 由来の丸め済み比は `--candle-jsonl` 未指定・データ欠落時
    # のみのフォールバックとする。
    r4_dgx_ok = True
    r4_m4max_ok = True
    for node in ("dgx", "m4max"):
        off = gate.get((node, "off"), {})
        on = gate.get((node, "on"), {})
        off_fandhe = jsonl.get((node, "off"), {})
        on_fandhe = jsonl.get((node, "on"), {})
        off_candle = candle_jsonl.get((node, "off"), {})
        on_candle = candle_jsonl.get((node, "on"), {})
        for n in (512, 1024, 2048):
            off_v_md = off.get(n)
            on_v_md = on.get(n)
            ratio_raw = _raw_candle_ratio(off_fandhe, on_fandhe, off_candle, on_candle, n)
            if ratio_raw is not None:
                ratio = ratio_raw
                ok = ratio >= RULE4_MIN_RATIO
                lines.append(
                    f"規則4改定版 {node} N={n}: candle比 on/off比（丸めなし raw）={ratio} "
                    f"(md参考: off={off_v_md} on={on_v_md}) "
                    f"(>= {RULE4_MIN_RATIO:.4f}) -> {'満たす' if ok else '不成立'}"
                )
            elif off_v_md is None or on_v_md is None or off_v_md == 0:
                ok = False
                ratio = None
                lines.append(
                    f"規則4改定版 {node} N={n}: candle比 off={off_v_md} on={on_v_md} "
                    f"on/off比={ratio} (>= {RULE4_MIN_RATIO:.4f}) -> {'不成立（データ欠落）'}"
                )
            else:
                # `--candle-jsonl` 未指定・データ欠落: Markdown 由来の
                # 丸め済み比へフォールバックする（既知の精度限界: 3 桁
                # 丸め由来の乖離は高々 5e-4 程度）。
                ratio = on_v_md / off_v_md
                ok = ratio >= RULE4_MIN_RATIO
                lines.append(
                    f"規則4改定版 {node} N={n}: candle比 off={off_v_md} on={on_v_md} "
                    f"on/off比（md フォールバック・丸めあり）={ratio} "
                    f"(>= {RULE4_MIN_RATIO:.4f}) -> {'満たす' if ok else '不成立'}"
                )
            if node == "dgx":
                r4_dgx_ok = r4_dgx_ok and ok
            else:
                r4_m4max_ok = r4_m4max_ok and ok

    # 規則 5: M4 Max N=2048 の後退判定（§20.1 場合分け (a) の Linux 限定化
    # 条件）。PR #1501 codex-review P1 是正: 当初実装は集計後の中央値比の
    # みで発火判定しており、5 run 個々の符号一貫性を検証していなかった
    # （DGX 合格・M4 Max 側規則 3/4 不成立でも符号混在の計測から
    # ADOPT_LINUX_ONLY を返しうる契約不整合）。本版は `--jsonl` 指定時、
    # (i) 生 median_s から丸めなしで算出した中央値比が閾値超、かつ
    # (ii) run 番号を対応付けた on/off 比（`_raw_run_pair_ratios`）が
    # 全 run で 1.0 超（符号一貫）、の両方を満たす場合のみ発火する。
    #
    # PR #1501 codex-review P1 追加是正（2026-09-09 その 3）: 上記までの
    # 版は `--jsonl` 未指定・データ欠落（`_raw_run_pair_ratios` が要求する
    # 両腕ちょうど `REQUIRED_RUN_COUNT`（5）件の対応関係を確認できない場合
    # を含む）を「後退なし（発火せず）」の fail-safe として扱っていたが、
    # これは規則 5 単体の判定を安全側へ倒しただけで、最終 verdict の
    # 決定ロジック（`dgx_rules_1_4_ok and m4max_rules_1_4_ok and not
    # r5_regression` → ADOPT_UNCONDITIONAL）へは反映されなかった。結果、
    # 規則 5 の入力が完全に欠落・不完全でも他規則さえ充足していれば
    # ADOPT_UNCONDITIONAL を返しうる契約不整合があった（AGENTS.md の
    # 5 回計測契約・§20.1 の採否条件は両腕 5 run の対応関係が前提）。
    # 本版は `r5_data_complete`（両腕ちょうど 5 run の対応関係を確認できた
    # か）を独立に保持し、「後退なし」（発火しない）と「判定不能」
    # （欠落・解析失敗）を明確に区別したうえで、判定不能な場合は
    # 他規則の成否によらず最終 verdict を UNDETERMINED に倒す
    # （下記「§20.1 場合分け」を参照）。
    m4_2048_reuse_cell = layer_a.get("m4max", {}).get((2048, "reuse"))
    m4_2048_reuse_md = m4_2048_reuse_cell.ratio if m4_2048_reuse_cell is not None else None
    m4_off_runs = jsonl.get(("m4max", "off"), {})
    m4_on_runs = jsonl.get(("m4max", "on"), {})
    m4_2048_reuse_raw = _raw_median_ratio(m4_off_runs, m4_on_runs, 2048, "reuse")
    run_pairs = _raw_run_pair_ratios(m4_off_runs, m4_on_runs, 2048, "reuse")

    if m4_2048_reuse_raw is not None and run_pairs is not None:
        r5_data_complete = True
        median_exceeds = m4_2048_reuse_raw > RULE234_THRESHOLD
        sign_consistent = all(r > 1.0 for r in run_pairs)
        r5_regression = median_exceeds and sign_consistent
        lines.append(
            f"規則5 M4Max N=2048: layer_a(reuse)={m4_2048_reuse_raw} "
            f"(md={m4_2048_reuse_md}) run別比={run_pairs} "
            f"中央値超過={median_exceeds} 符号一貫={sign_consistent} "
            f"-> {'後退あり（run別符号一貫を確認済み）' if r5_regression else '後退なし（発火せず）'}"
        )
    else:
        # `--jsonl` 未指定・データ欠落・両腕 5 run 対応の不足（解析失敗を
        # 含む）: 「後退なし」とは区別し判定不能として扱う。規則 5 単体の
        # 発火は安全側（させない）へ倒すが、`r5_data_complete=False` は
        # 最終 verdict を UNDETERMINED へ強制する（P1 是正の核心）。
        r5_regression = False
        r5_data_complete = False
        lines.append(
            f"規則5 M4Max N=2048: layer_a(reuse)={m4_2048_reuse_md} "
            f"(生 JSONL 未指定、または両腕ちょうど {REQUIRED_RUN_COUNT} run の対応関係が"
            f"確認できないため run別符号一貫性を確認できず、"
            f"中央値比のみでの発火判定は行わない) -> 判定不能（最終 verdict を UNDETERMINED へ）"
        )

    # ---- §20.1 場合分け (a)〜(c) を最終 verdict へ折り込む ----
    dgx_rules_1_4_ok = checksum_ok and r2_ok and r3_dgx_ok and r4_dgx_ok
    m4max_rules_1_4_ok = checksum_ok and r3_m4max_ok and r4_m4max_ok

    if not checksum_ok:
        # 規則 1 は前提条件。不成立なら他規則の成否によらず判定不能。
        verdict = "UNDETERMINED"
    elif not r5_data_complete:
        # 規則 5 の入力（両腕 5 run の対応関係）が欠落・不完全な場合も、
        # 規則 1 と同様に他規則の成否によらず判定不能とする（P1 是正）。
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
        f"r5_data_complete={r5_data_complete} -> verdict={verdict}"
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

    # PR #1501 codex-review P2 是正の回帰テスト: `<!-- raw ... -->`
    # コメント行があれば、テーブル列の丸め値（小数 4 桁）ではなく
    # 丸めなしの raw 比を優先して採用すること（境界値の誤通過・誤棄却
    # を防ぐ本修正の核心）。
    sample_layer_b_raw = """
## N=2048

| phase | off median (of 5 run medians, ms) | off n | on median (ms) | on n | on/off 比 |
|---|---|---|---|---|---|
| ops_gemm | 1.0 | 5 | 1.00004 | 5 | 1.0000 |
<!-- raw N=2048 phase=ops_gemm off_med=1.0 on_med=1.00004 ratio=1.00004 -->
"""
    parsed_b_raw = parse_layer_b_md(sample_layer_b_raw)
    # テーブル列だけを見れば 1.0000（<= 1.00 を満たすように見える）だが、
    # raw コメントの真の比 1.00004 は不成立（> 1.00）であるべき。
    assert parsed_b_raw[(2048, "ops_gemm")] == 1.00004, parsed_b_raw
    assert parsed_b_raw[(2048, "ops_gemm")] > RULE2_OPS_GEMM_THRESHOLD, parsed_b_raw

    # raw コメントが無い旧形式の md では従来どおりテーブル列の丸め値へ
    # フォールバックすること（後方互換）。
    assert parse_layer_b_md(sample_layer_b)[(2048, "ops_gemm")] == 0.9972

    # `parse_jsonl_medians`: bench-fandhe 生 JSONL（1 行 1 run）から
    # (size, mode) ごとの median_s リストを出現順で抽出できること。
    # 壊れた行（JSON 不正・必須フィールド欠落）は静かに無視すること。
    sample_jsonl = (
        '{"size":2048,"mode":"reuse","median_s":0.0210}\n'
        '{"size":2048,"mode":"fresh","median_s":0.0217}\n'
        "not json\n"
        '{"size":2048,"mode":"reuse"}\n'
        '{"size":2048,"mode":"reuse","median_s":0.0213}\n'
    )
    parsed_jsonl = parse_jsonl_medians(sample_jsonl)
    assert parsed_jsonl[(2048, "reuse")] == [0.0210, 0.0213], parsed_jsonl
    assert parsed_jsonl[(2048, "fresh")] == [0.0217], parsed_jsonl

    # `_raw_median_ratio`／`_raw_run_pair_ratios`: 生 median_s から
    # 丸めなしの中央値比・run 対応付き比リストを計算できること。
    off_runs_sample = {(2048, "reuse"): [1.0, 1.0, 1.0, 1.0, 1.0]}
    on_runs_sample = {(2048, "reuse"): [1.2, 1.2, 1.2, 1.2, 1.2]}
    assert _raw_median_ratio(off_runs_sample, on_runs_sample, 2048, "reuse") == 1.2
    assert _raw_run_pair_ratios(off_runs_sample, on_runs_sample, 2048, "reuse") == [
        1.2,
        1.2,
        1.2,
        1.2,
        1.2,
    ]
    # データ欠落セルは None を返すこと。
    assert _raw_median_ratio(off_runs_sample, on_runs_sample, 4096, "reuse") is None
    assert _raw_run_pair_ratios({}, on_runs_sample, 2048, "reuse") is None

    # PR #1501 codex-review P1 是正の回帰テスト: 片腕の run 数が
    # `REQUIRED_RUN_COUNT`（5）未満・超過の場合は「短い方へ切り詰めて」
    # 対応付けを返さず None（判定不能）を返すこと（旧実装は
    # `min(len(off), len(on))` で切り詰めていたため、片腕 1 run のみでも
    # 対応付けが得られてしまっていた）。
    off_runs_full = {(2048, "reuse"): [1.0, 1.0, 1.0, 1.0, 1.0]}
    on_runs_short = {(2048, "reuse"): [1.2]}
    assert _raw_run_pair_ratios(off_runs_full, on_runs_short, 2048, "reuse") is None
    on_runs_long = {(2048, "reuse"): [1.2, 1.2, 1.2, 1.2, 1.2, 1.2]}
    assert _raw_run_pair_ratios(off_runs_full, on_runs_long, 2048, "reuse") is None

    # PR #1501 codex-review P1 是正の回帰テスト（2026-09-09 その 5）:
    # `_raw_median_ratio`（規則 2〜4 の判定に直接使う）も片腕が
    # `REQUIRED_RUN_COUNT`（5）未満・超過の場合は None（判定不能。
    # Markdown フォールバックへ倒れる）を返すこと。旧実装は「両腕とも
    # 非空」しか検証せず、1 run だけの「中央値」を採用しうる欠陥が
    # あった（`_raw_run_pair_ratios` にしか 5 run 完備要求が実装
    # されておらず、規則 5 以外の判定〈規則 2・3〉が本欠陥の対象
    # だった）。
    assert _raw_median_ratio(off_runs_full, on_runs_short, 2048, "reuse") is None
    assert _raw_median_ratio(off_runs_full, on_runs_long, 2048, "reuse") is None
    # 片腕が完全に欠落している場合も引き続き None。
    assert _raw_median_ratio({}, off_runs_full, 2048, "reuse") is None
    # 両腕ちょうど 5 件なら従来どおり丸めなし中央値比を返す（非退行確認）。
    assert _raw_median_ratio(off_runs_full, off_runs_full, 2048, "reuse") == 1.0

    # `_raw_candle_ratio` も同型の 5 run 完備要求を持つこと（PR #1501
    # codex-review P1 是正）。4 入力（fandhe-ai off/on・candle off/on）
    # のいずれか 1 つでも run 数が `REQUIRED_RUN_COUNT` と異なれば None。
    candle_full = {2048: [1.0, 1.0, 1.0, 1.0, 1.0]}
    candle_short = {2048: [1.0]}
    assert (
        _raw_candle_ratio(off_runs_full, off_runs_full, candle_full, candle_short, 2048)
        is None
    )
    assert (
        _raw_candle_ratio(off_runs_full, on_runs_short, candle_full, candle_full, 2048)
        is None
    )

    # `parse_candle_jsonl_medians`: `framework` タグ付き生 JSONL から
    # candle（`mode=="fresh"`）行のみを size ごとに抽出し、fandhe-ai の
    # 行（同じ `mode=="fresh"` でも `framework` が異なる）を混入させない
    # こと。壊れた行・candle 以外の行は無視すること。
    sample_framework_jsonl = (
        '{"framework":"fandhe-ai","size":2048,"mode":"reuse","median_s":1.0}\n'
        '{"framework":"candle","size":2048,"mode":"fresh","median_s":1.00049}\n'
        '{"framework":"fandhe-ai","size":2048,"mode":"fresh","median_s":9.99}\n'
        "not json\n"
        '{"framework":"candle","size":2048,"mode":"reuse","median_s":8.88}\n'
    )
    parsed_candle = parse_candle_jsonl_medians(sample_framework_jsonl)
    assert parsed_candle == {2048: [1.00049]}, parsed_candle

    # PR #1501 codex-review P2 是正の核心テスト: 指摘に挙げられた具体例
    # （off=1.00049, on=0.95251）を丸めなしの生 median_s で再現する。
    # 真の on/off 比は `RULE4_MIN_RATIO`（~0.952380...）未満だが、
    # 3 桁丸め表示値（1.000／0.953）を再除算すると 0.953 となり
    # 閾値を満たして見える（採否逆転）。`_raw_candle_ratio` は生値から
    # 直接計算するため、この逆転を起こさず正しく不成立と判定できること。
    off_fandhe_sample = {(2048, "reuse"): [1.0, 1.0, 1.0, 1.0, 1.0]}
    on_fandhe_sample = {(2048, "reuse"): [1.0, 1.0, 1.0, 1.0, 1.0]}
    # 5 件とも同一値にして中央値が単一値ケースと一致するようにする
    # （PR #1501 codex-review P1 是正: `_raw_candle_ratio` も 4 入力
    # すべてちょうど `REQUIRED_RUN_COUNT` 件を要求するようになったため、
    # 1 件だけのフィクスチャでは None が返り本テストの意図〈丸めなし比の
    # 逆転再現〉を検証できなくなった）。
    off_candle_sample = {2048: [1.00049] * REQUIRED_RUN_COUNT}
    on_candle_sample = {2048: [0.95251] * REQUIRED_RUN_COUNT}
    raw_ratio = _raw_candle_ratio(
        off_fandhe_sample, on_fandhe_sample, off_candle_sample, on_candle_sample, 2048
    )
    assert raw_ratio is not None
    assert raw_ratio < RULE4_MIN_RATIO, raw_ratio  # 真の比は不成立
    md_reround_ratio = round(0.95251, 3) / round(1.00049, 3)
    assert md_reround_ratio >= RULE4_MIN_RATIO, md_reround_ratio  # 丸め値の再除算では成立して見える（欠陥の再現）
    assert raw_ratio != md_reround_ratio

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
    # 規則 5 の入力（M4 Max N=2048 の両腕 `REQUIRED_RUN_COUNT` 件の run
    # 対応関係）を「後退なし」（on/off 比 1.0・5 run とも符号一貫）で
    # 満たす jsonl フィクスチャ。PR #1501 codex-review P1 是正により
    # 規則 5 の入力が欠落・不完全な場合は他規則の成否によらず
    # verdict が UNDETERMINED に倒れるため、ADOPT_UNCONDITIONAL／REJECT
    # を検証する自己テストは本フィクスチャで規則 5 の入力を明示的に
    # 完備させる必要がある。
    jsonl_r5_no_regression = {
        ("m4max", "off"): {(2048, "reuse"): [1.0, 1.0, 1.0, 1.0, 1.0]},
        ("m4max", "on"): {(2048, "reuse"): [1.0, 1.0, 1.0, 1.0, 1.0]},
    }

    # judge() の一貫性チェック 1: 全セル満たす合成サンプルで
    # ADOPT_UNCONDITIONAL（§20.1 場合分け (c)）を返すこと。
    _, verdict = judge(_all_ok_layer_a(), layer_b_ok, gate_ok, jsonl_r5_no_regression)
    assert verdict == "ADOPT_UNCONDITIONAL", verdict

    # 規則 4 不成立ケース（on 腕 candle 比が off 腕比で 10% 悪化）→ REJECT。
    gate_bad = dict(gate_ok)
    gate_bad[("dgx", "on")] = {512: 0.63, 1024: 0.78, 2048: 0.96}
    _, verdict_bad = judge(_all_ok_layer_a(), layer_b_ok, gate_bad, jsonl_r5_no_regression)
    assert verdict_bad == "REJECT", verdict_bad

    # PR #1501 codex-review P2 是正の end-to-end 回帰テスト: 規則 4 の
    # DGX N=512 セルが Markdown 丸め値（1.000／0.953）では成立して見える
    # が、`--candle-jsonl` 経由の生 median_s（off=1.00049, on=0.95251）
    # では不成立となる境界ケースで、`judge()` が REJECT を返すこと
    # （Markdown 値のみで判定していれば ADOPT_UNCONDITIONAL になって
    # しまう欠陥の直接的な再発防止テスト）。
    gate_boundary = dict(gate_ok)
    gate_boundary[("dgx", "off")] = {512: round(1.00049, 3), 1024: 0.78, 2048: 0.96}
    gate_boundary[("dgx", "on")] = {512: round(0.95251, 3), 1024: 0.78, 2048: 0.96}
    # 5 件とも同一値にして中央値が単一値ケースと一致するようにする
    # （PR #1501 codex-review P1 是正: `_raw_candle_ratio` が 4 入力とも
    # ちょうど `REQUIRED_RUN_COUNT` 件を要求するようになったため、1 件
    # だけのフィクスチャでは None が返り Markdown フォールバックへ
    # 倒れてしまい、本テストが検証すべき「生値では正しく不成立」の
    # end-to-end 経路を検証できなくなる）。
    candle_jsonl_boundary = {
        ("dgx", "off"): {512: [1.00049] * REQUIRED_RUN_COUNT},
        ("dgx", "on"): {512: [0.95251] * REQUIRED_RUN_COUNT},
    }
    jsonl_dgx_512_fandhe = dict(jsonl_r5_no_regression)
    jsonl_dgx_512_fandhe[("dgx", "off")] = {(512, "reuse"): [1.0, 1.0, 1.0, 1.0, 1.0]}
    jsonl_dgx_512_fandhe[("dgx", "on")] = {(512, "reuse"): [1.0, 1.0, 1.0, 1.0, 1.0]}
    _, verdict_boundary_md_only = judge(_all_ok_layer_a(), layer_b_ok, gate_boundary, jsonl_r5_no_regression)
    assert verdict_boundary_md_only == "ADOPT_UNCONDITIONAL", verdict_boundary_md_only  # 丸め値のみでは欠陥どおり通ってしまう
    _, verdict_boundary_raw = judge(
        _all_ok_layer_a(), layer_b_ok, gate_boundary, jsonl_dgx_512_fandhe, candle_jsonl_boundary
    )
    assert verdict_boundary_raw == "REJECT", verdict_boundary_raw  # 生値では正しく不成立

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

    # 規則 5 発火ケース（M4 Max N=2048 が 1.05 超で後退・かつ run 別比が
    # 全 run 1.0 超で符号一貫）だが DGX 側は規則 1〜4 を満たす →
    # ADOPT_LINUX_ONLY（§20.1 場合分け (a)）。これが初版のもう一つの欠陥
    # （規則 5 が verdict に反映されず常に無条件 ADOPT/REJECT の二値に
    # なっていた）を再発させないための回帰テスト。PR #1501 codex-review
    # P1 是正後は `--jsonl` 経由の run 別データも与え、符号一貫性が
    # 確認された場合に限り発火することを検証する。
    layer_a_r5 = _all_ok_layer_a()
    layer_a_r5["m4max"] = dict(layer_a_r5["m4max"])
    layer_a_r5["m4max"][(2048, "reuse")] = LayerACell(
        ratio=1.20, checksum=CHECKSUM_OK_VALUE
    )
    jsonl_r5_consistent = {
        ("m4max", "off"): {(2048, "reuse"): [1.0, 1.0, 1.0, 1.0, 1.0]},
        ("m4max", "on"): {(2048, "reuse"): [1.2, 1.2, 1.2, 1.2, 1.2]},
    }
    _, verdict_r5 = judge(layer_a_r5, layer_b_ok, gate_ok, jsonl_r5_consistent)
    assert verdict_r5 == "ADOPT_LINUX_ONLY", verdict_r5

    # 規則 5 の P1 是正の核心テスト: Markdown 中央値比は 1.05 超（後退
    # ありに見える）だが run 別 JSONL の符号が混在（5 run 中 1 run は
    # on < off）している場合、規則 5 は「符号一貫性を確認できない」ため
    # 発火**しない**こと（中央値比のみで判定していた初版の欠陥の直接的な
    # 再発防止テスト）。他規則（1〜4）は両実機とも満たすため、規則 5 が
    # 発火しない結果として ADOPT_UNCONDITIONAL になる（規則 5 は
    # ADOPT_LINUX_ONLY への切替条件であり、未確定を REJECT 直結にはしない
    # 設計。規則 3・4 の対照セル・N=2048 決定セル自体は別途 §20.1a の
    # 独立した閾値判定で担保されている）。
    layer_a_r5_mixed = _all_ok_layer_a()
    layer_a_r5_mixed["m4max"] = dict(layer_a_r5_mixed["m4max"])
    layer_a_r5_mixed["m4max"][(2048, "reuse")] = LayerACell(
        ratio=1.20, checksum=CHECKSUM_OK_VALUE
    )
    jsonl_r5_mixed = {
        ("m4max", "off"): {(2048, "reuse"): [1.0, 1.0, 1.0, 1.0, 1.0]},
        ("m4max", "on"): {(2048, "reuse"): [1.5, 1.5, 1.5, 0.9, 1.5]},
    }
    _, verdict_r5_mixed = judge(layer_a_r5_mixed, layer_b_ok, gate_ok, jsonl_r5_mixed)
    assert verdict_r5_mixed == "ADOPT_UNCONDITIONAL", verdict_r5_mixed
    assert verdict_r5_mixed != "ADOPT_LINUX_ONLY", verdict_r5_mixed

    # 規則 5 の P1 追加是正の核心テスト（2026-09-09 その 3）: `--jsonl`
    # 未指定（None）の場合、規則 5 の入力（両腕 5 run の対応関係）を
    # 確認できないため、他規則（1〜4）が全て満たされていても
    # ADOPT_UNCONDITIONAL を返さず UNDETERMINED（判定不能）を返すこと。
    # 「後退なし」の fail-safe が最終 verdict の ADOPT 判定を誤って
    # 素通りさせていた契約不整合の直接的な再発防止テスト。
    _, verdict_r5_no_jsonl = judge(layer_a_r5, layer_b_ok, gate_ok, None)
    assert verdict_r5_no_jsonl == "UNDETERMINED", verdict_r5_no_jsonl

    # 規則 5 の入力が片腕のみ・run 数不足（4 run）の場合も同様に
    # UNDETERMINED を返すこと（「短い方へ切り詰めて判定を通す」旧実装の
    # 再発防止テスト。`_raw_run_pair_ratios` は両腕ちょうど 5 run を
    # 要求するため、4 run しかない側があれば None を返す）。
    jsonl_r5_incomplete = {
        ("m4max", "off"): {(2048, "reuse"): [1.0, 1.0, 1.0, 1.0, 1.0]},
        ("m4max", "on"): {(2048, "reuse"): [1.2, 1.2, 1.2, 1.2]},
    }
    _, verdict_r5_incomplete = judge(layer_a_r5, layer_b_ok, gate_ok, jsonl_r5_incomplete)
    assert verdict_r5_incomplete == "UNDETERMINED", verdict_r5_incomplete

    # 規則 5 は発火しない（M4 Max N=2048 <= 1.05）が M4 Max の対照セル
    # （規則 3）が不成立 → (a)/(c) いずれにも該当しないため REJECT。
    # 規則 5 の入力（両腕 5 run の対応関係）は `jsonl_r5_no_regression`
    # で明示的に完備させる（未完備だと UNDETERMINED になり本テストの
    # 意図〈規則 3 不成立 → REJECT〉を検証できないため）。
    layer_a_m4max_r3_fail = _all_ok_layer_a()
    layer_a_m4max_r3_fail["m4max"] = dict(layer_a_m4max_r3_fail["m4max"])
    layer_a_m4max_r3_fail["m4max"][(512, "fresh")] = LayerACell(
        ratio=1.20, checksum=CHECKSUM_OK_VALUE
    )
    _, verdict_m4max_r3_fail = judge(
        layer_a_m4max_r3_fail, layer_b_ok, gate_ok, jsonl_r5_no_regression
    )
    assert verdict_m4max_r3_fail == "REJECT", verdict_m4max_r3_fail

    print("self-test: OK", file=sys.stderr)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer-a-md", action="append", default=[], help="node:path または path（node は '-dgx.md'/'−m4max.md' 等のファイル名から推定）")
    ap.add_argument("--layer-b-md", action="append", default=[])
    ap.add_argument("--gate-md", action="append", default=[], help="node:arm:path 形式（例 dgx:off:compare_gemm_gate-dgx-off.md）")
    ap.add_argument(
        "--jsonl",
        action="append",
        default=[],
        help=(
            "node:arm:path 形式（例 m4max:off:results-m4max-....jsonl）。"
            "bench-fandhe 生 JSONL（1 行 1 run）を渡すと規則 2・3・5 が "
            "丸めなしの中央値比・run 別符号一貫性で判定される（PR #1501 "
            "codex-review P1/P2 是正）。未指定でも Markdown 由来の値で "
            "動作する（後方互換）。"
        ),
    )
    ap.add_argument(
        "--candle-jsonl",
        action="append",
        default=[],
        help=(
            "node:arm:path 形式（例 m4max:off:results-m4max-....jsonl）。"
            "`scripts/bench/framework-compare/results/raw/results-*.jsonl` "
            "の `framework` タグ付き生 JSONL（candle・fandhe-ai 両方の行を"
            "含む）を渡すと規則 4（改定版）が丸めなしの candle 比 on/off 比 "
            "で判定される（PR #1501 codex-review P2 是正）。未指定でも "
            "compare_gemm_gate.py 出力 Markdown の丸め済み値で動作する "
            "（後方互換フォールバック）。"
        ),
    )
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

    jsonl: dict[tuple[str, str], dict[tuple[int, str], list[float]]] = {}
    for spec in args.jsonl:
        parts = spec.split(":", 2)
        if len(parts) != 3:
            raise SystemExit(f"ERROR: --jsonl は node:arm:path 形式が必須（実際: {spec}）")
        node, arm, path = parts
        if node not in ("dgx", "m4max") or arm not in ("off", "on"):
            raise SystemExit(f"ERROR: --jsonl の node/arm が不正: {spec}")
        with open(path, encoding="utf-8") as f:
            jsonl[(node, arm)] = parse_jsonl_medians(f.read())

    candle_jsonl: dict[tuple[str, str], dict[int, list[float]]] = {}
    for spec in args.candle_jsonl:
        parts = spec.split(":", 2)
        if len(parts) != 3:
            raise SystemExit(f"ERROR: --candle-jsonl は node:arm:path 形式が必須（実際: {spec}）")
        node, arm, path = parts
        if node not in ("dgx", "m4max") or arm not in ("off", "on"):
            raise SystemExit(f"ERROR: --candle-jsonl の node/arm が不正: {spec}")
        with open(path, encoding="utf-8") as f:
            candle_jsonl[(node, arm)] = parse_candle_jsonl_medians(f.read())

    lines, verdict = judge(layer_a, layer_b, gate, jsonl, candle_jsonl)
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
