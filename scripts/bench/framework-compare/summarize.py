#!/usr/bin/env python3
"""JSONL 計測結果 → Markdown 表の集計ツール。

使い方:
    python3 summarize.py [JSONL ...] [--out FILE]

- 入力: results/raw/ の JSONL（省略時は results/raw/*.jsonl を全件）。
  環境ごとのファイル（例: results.jsonl = Apple M4 Max / macOS、
  results-dgx.jsonl = DGX Spark）をそれぞれ独立のセクションとして表化する。
  表化するデバイス列（cpu / metal / cuda）は各ファイルに実在する行から導出する
  （macOS 前提の固定デバイス集合をハードコードしない）。
- 出力: 既定は標準出力。`--out FILE` を明示した場合のみファイルへ書き込む。
  コミット済みの results/summary.md（複数環境を統合した一次データ。環境情報・
  備考は人間が追記済み）を既定動作で上書きしない。
- 環境情報（チップ・OS 等）は入力 JSONL からは分からないため出力に含めない
  （リモート環境の JSONL をローカルのホスト情報でラベル付けしない）。
  環境の正は results/summary.md・results/versions.txt・run_all*.log。
- mode（イシュー #925）: "fresh"（既定・毎回新規デバイス/tape）と "reuse"
  （デバイス/tape 使い回し。初期化コスト init_s を分離計測）を区別する。
  本フィールド追加前にコミットされた JSONL には mode キーが無いため、
  欠損は "fresh" として扱う（互換維持。get(row, "mode", "fresh")）。
  既存の GEMM 表（(a)）は fresh 行のみを集計し、reuse 行が存在するファイル
  にのみ (a') 節（初期化 init_s・中央値・fresh との並記）を追加する。
  train の reuse 行（イシュー #957/#958/#959。`DeviceParamStore` によるデバイス
  常駐パラメータ更新）も同様に (b') 節で集計し、最終 loss（checksum）を fresh と
  突合する（gemm と異なりフレームワーク間では突合しない: 重み初期化が異なる設計
  のため fandhe-ai と candle/Burn の最終 loss は一致しない。突合は同一フレーム
  ワーク内の fresh vs reuse のみ）。
- checksum 相互突合（イシュー #965）: GEMM は全フレームワーク・全 mode で
  同一入力（xorshift64* の同一シード・同一生成式）のため、同一 size の
  checksum は本体の数値一致契約（相対誤差 1e-3 未満 または 絶対誤差 1e-5
  未満。`.claude/rules/coding-rust.md`）内で一致するはずである。本ツールは
  各 size ごとに参照値を選び、外れる行を GEMM 表で「無効」表示する
  （`gemm_checksum_reference` / `gemm_checksum_mismatches`）。対象は gemm
  タスクのみ（train/infer は fandhe-ai の重み初期化が candle/Burn と異なる
  設計のため checksum が一致しない。突合しない）。既定では警告を stderr へ
  出すのみで終了コードは変えない（`--out` 契約と同様、既存の呼び出し元を
  壊さない）。`--strict` を付けると不一致 1 件以上で終了コード 2 を返す。
- 要素単位検証（イシュー #970）: checksum（全要素和）は要素の入れ替わり・
  正負誤差の相殺で偶然一致しうる破損を見逃す。GEMM バイナリ
  （`bench-fandhe`/`bench-candle`/`bench-burn`）は各反復で結果を FMA 契約の
  参照 GEMM（`bench-common::GemmReference`。本体 `backend-cpu::parity::
  matmul_reference_fma` と同じ契約）と要素単位で突合し、反復間 worst-case
  を `parity_total`/`parity_fail_count`/`parity_max_abs_err`/
  `parity_max_rel_err` として JSONL に記録する（閾値は本体の数値一致契約と
  同一の `PARITY_ABS_TOL`/`PARITY_REL_TOL`）。本ツールはこれを読み、
  `parity_fail_count > 0` または各フィールドの型・値が不正（`null` 含む）
  な行を GEMM 表で「無効（要素誤差超過）」表示し `--strict` の対象にする
  （`parity_status`）。本フィールド追加前の JSONL（キー欠損）は「無効」と
  区別して「未検証（旧形式）」と表示する（表示上は区別するが、要素単位検証
  を一度も受けていない点は数値正当性が未確認のままであり、`--strict` の
  対象にも含める。fail-closed だがデータの誤破棄はしない）。checksum
  突合（#965）とは独立に判定し、両方に該当する行は理由を併記する。
- 実行時失敗（skipped*.log）節は、集計対象として渡された各入力 JSONL と
  同一ディレクトリの skipped*.log のみを集める（入力省略時は従来どおり
  results/raw/ 配下が対象。articles#68 Bugbot 指摘・イシュー #971）。
- (c) のバッチ/秒は 10 未満を小数 1 桁で表示する（`:.0f` だと 1 未満の値が
  1 に丸まり実際の約 2 倍に見えるため。articles#68 Bugbot 指摘・イシュー #971）。
- (b') train reuse（イシュー #957/#958/#959）: reuse 行の checksum（最終
  loss）を同一フレームワークの fresh 行と突合し、不一致・突合不能（無効
  値）・時間値（median_s/q1_s/q3_s）の不正を表で「無効」表示するだけでなく
  `--strict` の失敗条件（終了コード 2）にも含める。fresh 行が存在しない
  （比較対象なしで突合不能）のみは値そのものの正当性を否定しないため
  「無効」扱いにせず `--strict` の対象にしない（イシュー #959 codex-review
  P1 指摘: 旧実装は表示のみで `section()` の戻り値に反映されず
  `--strict` でも終了コード 0 のままだった）。
"""

import argparse
import glob
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

FRAMEWORKS = ["fandhe-ai", "candle", "burn"]
DEVICE_ORDER = ["cpu", "metal", "cuda"]
DEVICE_LABEL = {"cpu": "CPU", "metal": "Metal", "cuda": "CUDA"}


def fmt_ms(s):
    if s >= 1.0:
        return f"{s:.3f} s"
    if s >= 1e-3:
        return f"{s * 1e3:.3f} ms"
    return f"{s * 1e6:.1f} µs"


def _safe_time_s(v):
    """外部 JSONL 由来の時間値（init_s / median_s / q1_s / q3_s）を表示・
    比率計算の前に検証する。`_is_plain_number` で bool・NaN・Infinity・
    非数値を弾いたうえで、時間として不正な非正値（0 以下）も無効とする
    （security.md「外部フォーマットのパース時検証（A03）」。イシュー #959
    codex-review P0 指摘: (b') 経路で検証していたのは比率計算の
    median_s のみで、init_s・q1_s・q3_s・fresh 側の値は未検証のまま
    `fmt_ms` へ渡され、文字列や負値・NaN が混入すると比較演算で
    TypeError になるか不正な表示を生じていた）。有効なら `float` を、
    無効なら `None` を返す（fail-closed。呼び出し側は `None` を
    「無効な値」表示に倒す）。
    """
    if not _is_plain_number(v):
        return None
    fv = float(v)
    return fv if fv > 0 else None


def _safe_finite_number(v):
    """外部 JSONL 由来の数値（checksum 等、正値制約のないもの）を使用前に
    検証する。`_is_plain_number` と同じ bool・NaN・Infinity 除外に加え、
    `float` へ正規化する（イシュー #959 codex-review P0 指摘。`checksums_match`
    への未検証な受け渡しを避ける）。有効なら `float` を、無効なら `None` を
    返す。
    """
    return float(v) if _is_plain_number(v) else None


def load_rows(path):
    # mode（イシュー #925）欠損は "fresh" 扱い（本フィールド追加前にコミット
    # 済みの JSONL との互換維持。モジュール docstring 参照）。
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    for r in rows:
        r.setdefault("mode", "fresh")
    return rows


def get(rows, fw, task, device, size=None, mode="fresh"):
    for r in rows:
        if (
            r["framework"] == fw
            and r["task"] == task
            and r["device"] == device
            and r["mode"] == mode
        ):
            if size is None or r["size"] == size:
                return r
    return None


def devices_in(rows, task, mode="fresh"):
    present = {r["device"] for r in rows if r["task"] == task and r["mode"] == mode}
    return [d for d in DEVICE_ORDER if d in present]


# 本体の数値一致契約と同一（`.claude/rules/coding-rust.md`「バックエンド構成」節）:
# 相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満。ここを緩めない。
#
# 正は `crates/backend-cpu/src/parity.rs`（RELATIVE_TOLERANCE /
# ABSOLUTE_RESCUE_THRESHOLD）。本ハーネスは独立 workspace（deps-policy.md
# 第 9 区分）で本体クレートを import できないため値をここへ再定義して
# いるが、本体側だけが変更されると静かに乖離しうる。乖離は
# `summarize_test.py::ToleranceDriftTests` が本体ソースを直接読んで
# 機械照合し fail-closed に検出する（イシュー #970 codex-review 指摘・
# PR #978 P1）。
CHECKSUM_ABS_TOL = 1e-5
CHECKSUM_REL_TOL = 1e-3

# 要素単位検証（イシュー #970）の閾値。CHECKSUM_* と同値（本体契約と揃える
# ための独立の名前。JSONL 側の生成は `bench-common::parity::{PARITY_ABS_TOL,
# PARITY_REL_TOL}` を正とし、判定はバイナリ側で完結する。本ツールは判定済み
# の parity_fail_count 等を読むだけで、ここでは閾値を再適用しない）。
PARITY_ABS_TOL = CHECKSUM_ABS_TOL
PARITY_REL_TOL = CHECKSUM_REL_TOL

# 参照値選択の優先順（イシュー #965）。GEMM の入力は全フレームワーク共通
# なので、最も検証済みの経路（CPU・fresh）から優先的に参照を取る。
_REFERENCE_PRIORITY = [
    ("fandhe-ai", "cpu"),
    ("candle", "cpu"),
    ("burn", "cpu"),
]


def _priority_rank(row):
    """`_REFERENCE_PRIORITY` 上の (framework, device) の優先順位を返す。

    該当が無ければ `_REFERENCE_PRIORITY` の長さ（最下位）を返す。
    """
    key = (row["framework"], row["device"])
    for i, cand in enumerate(_REFERENCE_PRIORITY):
        if cand == key:
            return i
    return len(_REFERENCE_PRIORITY)


def checksums_match(a, b):
    """本体の数値一致契約と同一の複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）。

    分母は `max(abs(a), abs(b), 1e-12)` とし、引数順序（a, b のどちらを参照値と
    するか）に依らず対称な判定にする（`composite_close` 相当。イシュー #965
    codex-review P1 指摘: 旧実装の `diff / abs(b)` は非対称で、境界付近では
    a/b の順序次第で判定結果が変わりうる問題があった）。
    """
    diff = abs(a - b)
    if diff < CHECKSUM_ABS_TOL:
        return True
    denom = max(abs(a), abs(b), 1e-12)
    return diff / denom < CHECKSUM_REL_TOL


def gemm_checksum_reference(rows):
    """size ごとの GEMM checksum 参照値を選ぶ。

    優先順は `_REFERENCE_PRIORITY`（fandhe-ai/cpu → candle/cpu → burn/cpu、
    いずれも fresh）。ただし優先経路の値も無条件では採用しない: 同一 size の
    候補群から相互一致クラスタ（`checksums_match` で相互一致する集合）の
    うち最大のものを先に求め、そのクラスタが 2 件以上（＝真の多数派が存在
    する）であるにもかかわらず優先経路の値がそのクラスタに属さない場合は
    孤立した誤値とみなして採用せず、次の優先経路 → 多数決クラスタの順に
    フォールバックする（イシュー #965 P2 指摘: 優先経路が無条件採用されると
    孤立した誤値でも参照として選ばれてしまう問題の修正）。全候補が互いに
    不一致（最大クラスタが 1 件）の場合は多数派による否定ができないため、
    優先経路の値をそのまま信頼する。クラスタサイズが同点の場合はファイル内
    出現順ではなく `_REFERENCE_PRIORITY` 上の最上位メンバーを含むクラスタを
    優先する（イシュー #965 Bugbot 指摘: 出現順依存だと GPU 行が先に現れる
    ケースで CPU 優先経路の checksum が不当に無効判定されうる問題の修正）。

    候補が 1 件も無い、または相互一致クラスタが 2 件未満で優先経路にも
    該当が無い場合は突合不能として None を返す。

    `candidate_count` は同一 size の fresh 行の総数。1 以下（この JSONL
    ファイル内に比較対象となる他フレームワーク／他デバイスの行がそもそも
    存在しない）の場合、`gemm_checksum_mismatches` はその size を「無効」と
    誤判定しない（クロスチェック不能と実データの不整合を区別するため。
    例: `results-rtx3060.jsonl` は fandhe-ai/cuda のみを計測しており、
    比較対象が無いだけで checksum 自体は正当）。呼び出し側（`section` の
    データ有効性節）も candidate_count<=1 の行を「一致」と誤表示せず
    「突合不能」として区別する（`gemm_checksum_unverifiable` 参照）。

    戻り値: {size: (ref_value, ref_source_label, candidate_count)}
    """
    sizes = sorted({r["size"] for r in rows if r["task"] == "gemm"})
    result = {}
    for size in sizes:
        candidates = [
            r
            for r in rows
            if r["task"] == "gemm" and r["size"] == size and r["mode"] == "fresh"
        ]

        # 相互一致するクラスタのうち最大のものを先に求める（多数派の把握）。
        # クラスタサイズが同点の場合、ファイル内での出現順（GPU 行が先に
        # 現れるかどうか）で多数派が決まってしまうと、CPU 優先経路
        # （`_REFERENCE_PRIORITY`）の checksum が不当に「孤立した誤値」
        # 判定されうる（イシュー #965 Bugbot 指摘）。そこで同点時は
        # `_REFERENCE_PRIORITY` 上の最上位メンバーを含むクラスタを優先する
        # （それも同点なら最初に見つかったクラスタを使う。決定的な順序）。
        best_cluster = []
        best_rank = len(_REFERENCE_PRIORITY)
        for r in candidates:
            cluster = [
                c for c in candidates if checksums_match(c["checksum"], r["checksum"])
            ]
            cluster_rank = min(
                (_priority_rank(c) for c in cluster), default=len(_REFERENCE_PRIORITY)
            )
            if len(cluster) > len(best_cluster) or (
                len(cluster) == len(best_cluster) and cluster_rank < best_rank
            ):
                best_cluster = cluster
                best_rank = cluster_rank

        ref = None
        ref_label = None
        for fw, device in _REFERENCE_PRIORITY:
            hit = next(
                (r for r in candidates if r["framework"] == fw and r["device"] == device),
                None,
            )
            if hit is None:
                continue
            if len(best_cluster) >= 2 and hit not in best_cluster:
                # 真の多数派クラスタが存在するのに優先経路の値がそこに
                # 属さない＝孤立した誤値。優先経路として採用しない。
                continue
            ref = hit["checksum"]
            ref_label = f"{fw}/{device}/fresh"
            break
        if ref is None and len(best_cluster) >= 2:
            # 多数決フォールバック（優先経路の該当なし、または孤立値として
            # 却下された場合）。
            rep = best_cluster[0]
            ref = rep["checksum"]
            ref_label = f"{rep['framework']}/{rep['device']}/fresh（多数決）"
        result[size] = (ref, ref_label, len(candidates))
    return result


def gemm_checksum_mismatches(rows):
    """GEMM 行のうち参照値と不一致なものを列挙する。

    reuse 行も同一 size の参照（fresh 由来）に対して突合する（fresh/reuse は
    同一入力のため）。train/infer は対象外（モジュール docstring 参照）。
    同一 size の fresh 行が 1 件以下（クロスチェックできる他行が無い）場合は
    「突合不能」を報告しない（`gemm_checksum_reference` docstring 参照）。

    戻り値: [(row, ref_value_or_None, ref_label_or_None), ...]
    """
    reference = gemm_checksum_reference(rows)
    mismatches = []
    for r in rows:
        if r["task"] != "gemm":
            continue
        ref, ref_label, candidate_count = reference.get(r["size"], (None, None, 0))
        if ref is None:
            if candidate_count >= 2:
                mismatches.append((r, None, None))
            continue
        if not checksums_match(r["checksum"], ref):
            mismatches.append((r, ref, ref_label))
    return mismatches


def gemm_checksum_unverifiable(rows):
    """突合不能（比較対象が候補 1 件以下）な GEMM 行を列挙する。

    イシュー #965 P2 指摘: `candidate_count<=1`（この JSONL 内に同一 size の
    他フレームワーク／他デバイス行が存在せず、そもそも相互突合できない）の
    行を、データ有効性節で「全 checksum 一致」と表示すると、実際には
    突合していないのに検証済みであるかのように誤認させる。本関数は
    そうした行を明示的に区別して報告するために使う（`gemm_checksum_reference`
    の `candidate_count` を参照。値の正当性自体は否定しない）。

    戻り値: [row, ...]
    """
    reference = gemm_checksum_reference(rows)
    unverifiable = []
    for r in rows:
        if r["task"] != "gemm":
            continue
        _, _, candidate_count = reference.get(r["size"], (None, None, 0))
        if candidate_count <= 1:
            unverifiable.append(r)
    return unverifiable


def _is_plain_number(v):
    """`bool` は `int` のサブクラスのため `isinstance(v, (int, float))` だけ
    では `True`/`False` を数値として通してしまう。JSON の `parity_*` 4
    フィールドは常に数値または `null`（非有限センチネル）であるべきで、
    誤って `bool` が混入した場合も無効として扱いたい（fail-closed。
    security.md A03 と同じ「外部入力の型を信頼しない」思想）。

    Python の `json` モジュールは既定で `NaN`/`Infinity`/`-Infinity` を
    パース可能にする（RFC 8259 非準拠の拡張）ため、型が `float` であっても
    `math.isfinite()` を別途要求する（イシュー #970 PR #978 codex-review P0
    指摘: 外部 JSONL の `NaN` が型検査だけでは弾けず "ok" 判定へ通ってしまう）。

    `int` は Python では任意精度で常に有限（`NaN`/`Infinity` になり得ない）
    ため `math.isfinite()` を適用しない。適用すると内部で `float` へ変換
    されるため、外部 JSONL の桁数の大きい `int`（例:
    `parity_total: 10**1000`）で `OverflowError: int too large to convert
    to float` が発生し、"fail" 判定を返す前に集計全体が例外終了してしまう
    （イシュー #970 PR #978 codex-review P0 指摘: 巨大整数で fail-closed
    契約が破られる）。`int` は有限として扱ったうえで、後続の値域・整数性・
    期待要素数検証（`parity_status`）で明示的に妥当性を判定する。
    """
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        return False
    if isinstance(v, int):
        return True
    return math.isfinite(v)


def _non_integral(v):
    """`v` が整数値でない `float` かどうかを判定する。

    `float(total) != int(total)` のような整数性検証は、`total` が桁数の
    大きい `int`（`_is_plain_number` により通過済み）の場合に `float()`
    変換で `OverflowError` を送出する（上記 `_is_plain_number` と同じ
    問題）。`int` は変換なしに自明に整数値であるため、`float` の場合の
    みを判定対象とする。
    """
    return isinstance(v, float) and not v.is_integer()


def _format_maybe_huge(v):
    """指数表記フォーマット（`f"{v:.3e}"`）は桁数の大きい `int` に対し
    内部で `float` 変換が発生し `OverflowError` になる（`_is_plain_number`
    と同根の問題）。報告用文字列の生成で集計全体を例外終了させないよう、
    フォーマット失敗時は `str()` へフォールバックする（fail-closed。
    イシュー #970 PR #978 codex-review P0 指摘）。
    """
    try:
        return f"{v:.3e}"
    except OverflowError:
        return str(v)


def parity_status(row):
    """GEMM 行の要素単位検証結果（イシュー #970）を判定する。

    戻り値: "unverified" | "fail" | "ok"

    - "unverified": `parity_fail_count`・`parity_total`・
      `parity_max_abs_err`・`parity_max_rel_err` の 4 キーが**すべて**
      存在しない（本フィールド追加前にコミットされた旧形式 JSONL）。
      `row.get(...)` だけでは「キー欠損（None）」と「値が JSON `null`
      （= Python None、非有限センチネル）」を区別できないため、まずキー
      の存在を検査する（欠損＝旧形式・未検証と、存在するが不正＝無効を
      混同しない）。4 キーのうち一部だけが欠けている場合は旧形式ではなく
      部分的に破損・改変された JSONL であるため "unverified" ではなく
      "fail" とする（fail-closed。イシュー #970 PR #978 codex-review P0
      指摘3: `parity_fail_count` のみが欠落し他 3 キーが存在する外部
      JSONL が "unverified" 扱いとなり `invalid_reasons()` の無効理由に
      含まれず GFLOP/s が有効値として表示され `--strict` も通過して
      しまっていた）。
    - "fail": 上記の「4 キー全欠損」に該当しないが、4 フィールドのいずれか
      の型が不正（キー欠損・数値でない・`null`・非有限）、`parity_total`/
      `parity_fail_count` が整数値でない、値域が不正（`parity_total` が
      0 以下、`parity_fail_count` が負または `parity_total` 超過、誤差
      2 項が負）、`parity_total` が GEMM の期待要素数（`size * size`）と
      不一致、または `parity_fail_count > 0`。壊れた入力を黙って「一致」
      扱いにしない（fail-closed。イシュー #970 PR #978 codex-review P0
      指摘1: 型検査のみでは `parity_fail_count=-1`・`parity_total=0`・
      負の誤差値を伴う外部 JSONL が "ok" 判定へ通ってしまっていた。P0
      指摘2: `parity_total` の値そのものは検査していなかったため、
      `parity_total=1, parity_fail_count=0` のように GEMM 結果のごく
      一部しか検証していない破損・改変 JSONL でも "ok" 判定になり
      GFLOP/s を有効値として表示してしまっていた。`size * size` との
      完全一致を要求することで、検証件数の水増し・過小を fail-closed に
      検出する。size 自体は本ツールが自ファイルから読んだ `row["size"]`
      であり JSONL の parity_* フィールドとは独立した信頼できる値のため、
      比較の基準として使える。Python の int は多倍長のためオーバーフロー
      しないが、`size` 側が数値型で非負整数であることも併せて検査する
      （`size` が不正な外部入力であれば期待要素数を算出できず、
      その場合も fail-closed で "fail" とする）。
    - "ok": 4 フィールドすべてが妥当な数値・整数値・値域で、`parity_total`
      が `size * size` と完全一致し、`parity_fail_count == 0`。
    """
    parity_keys = (
        "parity_fail_count",
        "parity_total",
        "parity_max_abs_err",
        "parity_max_rel_err",
    )
    if all(k not in row for k in parity_keys):
        return "unverified"
    fail_count = row.get("parity_fail_count")
    total = row.get("parity_total")
    max_abs = row.get("parity_max_abs_err")
    max_rel = row.get("parity_max_rel_err")
    if not _is_plain_number(fail_count) or not _is_plain_number(total):
        return "fail"
    if not _is_plain_number(max_abs) or not _is_plain_number(max_rel):
        return "fail"
    # 整数性検証（イシュー #970 PR #978 codex-review P0 指摘2）: total・
    # fail_count は要素数のカウントであり非整数値（例: 1.5）は不正入力。
    if _non_integral(total) or _non_integral(fail_count):
        return "fail"
    total = int(total)
    fail_count = int(fail_count)
    # 値域検証（イシュー #970 PR #978 codex-review P0 指摘1）: total は正、
    # fail_count は [0, total] の範囲、誤差 2 項は非負でなければならない。
    if total <= 0:
        return "fail"
    if fail_count < 0 or fail_count > total:
        return "fail"
    if max_abs < 0 or max_rel < 0:
        return "fail"
    # 期待要素数検証（イシュー #970 PR #978 codex-review P0 指摘2）: GEMM は
    # size×size 要素の正方行列であるため、parity_total は size*size と
    # 完全一致しなければならない。row["size"] は本ツールが自ファイルから
    # 読んだ信頼できる値（JSONL の parity_* とは独立の情報源）。
    size = row.get("size")
    if not _is_plain_number(size) or _non_integral(size) or int(size) < 0:
        return "fail"
    expected_total = int(size) * int(size)
    if total != expected_total:
        return "fail"
    if fail_count > 0:
        return "fail"
    return "ok"


def gemm_parity_failures(rows):
    """`parity_status(r) == "fail"` の gemm 行を列挙する。"""
    return [r for r in rows if r["task"] == "gemm" and parity_status(r) == "fail"]


def gemm_parity_unverified(rows):
    """`parity_status(r) == "unverified"`（旧形式 JSONL）の gemm 行を列挙する。"""
    return [r for r in rows if r["task"] == "gemm" and parity_status(r) == "unverified"]


def _parity_reason(row):
    """表・データ有効性節向けの「無効（要素誤差超過）」理由テキスト。

    `parity_status(row) != "fail"` の場合は `None` を返す。
    """
    if parity_status(row) != "fail":
        return None
    fail_count = row.get("parity_fail_count")
    total = row.get("parity_total")
    max_abs = row.get("parity_max_abs_err")
    max_rel = row.get("parity_max_rel_err")
    fail_str = (
        f"{fail_count}/{total}"
        if _is_plain_number(fail_count) and _is_plain_number(total)
        else "?"
    )
    abs_str = _format_maybe_huge(max_abs) if _is_plain_number(max_abs) else "null"
    rel_str = _format_maybe_huge(max_rel) if _is_plain_number(max_rel) else "null"
    return f"要素誤差超過 fail={fail_str}, max_abs={abs_str}, max_rel={rel_str}"


def _row_key(r):
    return (r["framework"], r["device"], r["size"], r["mode"])


def section(path, rows):
    lines = []
    rel = os.path.relpath(path, HERE)
    lines.append(f"## 集計対象: {rel}\n")

    # イシュー #965: GEMM checksum 相互突合。不一致行は表で「無効」表示し、
    # GFLOP/s 列を "-" にする（壊れた計算の実行時間を性能値として見せない）。
    mismatches = gemm_checksum_mismatches(rows)
    mismatch_by_key = {_row_key(r): (ref, ref_label) for r, ref, ref_label in mismatches}
    for r, ref, ref_label in mismatches:
        if ref is None:
            print(
                f"warning: {rel}: {r['framework']}/{r['device']}/size={r['size']}/{r['mode']} "
                "の gemm checksum は参照値を決定できず突合不能",
                file=sys.stderr,
            )
        else:
            print(
                f"warning: {rel}: {r['framework']}/{r['device']}/size={r['size']}/{r['mode']} "
                f"の gemm checksum {r['checksum']:.6f} が参照 {ref:.6f}（{ref_label}）と不一致 "
                "— 無効データとして表示",
                file=sys.stderr,
            )

    # イシュー #970: GEMM 要素単位検証。閾値超過（または parity フィールドの
    # 型・値が不正）は checksum 突合とは独立に「無効」表示・GFLOP/s "-" の
    # 対象にする。旧形式（キー欠損）は「未検証」として区別し無効扱いしない。
    parity_failures = gemm_parity_failures(rows)
    for r in parity_failures:
        print(
            f"warning: {rel}: {r['framework']}/{r['device']}/size={r['size']}/{r['mode']} "
            f"の gemm 要素単位検証が閾値超過 — {_parity_reason(r)} — 無効データとして表示",
            file=sys.stderr,
        )

    def invalid_reasons(r):
        """表のフレームワーク列に付記する「無効」理由のリスト（無ければ空）。"""
        reasons = []
        if mismatch_by_key.get(_row_key(r)) is not None:
            reasons.append("checksum 不一致")
        preason = _parity_reason(r)
        if preason is not None:
            reasons.append(preason)
        return reasons

    versions = {r["framework"]: r["version"] for r in rows}
    lines.append("| フレームワーク | バージョン |")
    lines.append("| --- | --- |")
    for fw in FRAMEWORKS:
        lines.append(f"| {fw} | {versions.get(fw, '?')} |")
    lines.append("")

    lines.append("### (a) GEMM（C = A×B、f32、正方行列）\n")
    for device in devices_in(rows, "gemm"):
        sizes = sorted(
            {r["size"] for r in rows if r["task"] == "gemm" and r["device"] == device}
        )
        lines.append(f"#### {DEVICE_LABEL[device]}\n")
        lines.append("| N | フレームワーク | 中央値 | Q1 | Q3 | GFLOP/s |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for n in sizes:
            for fw in FRAMEWORKS:
                r = get(rows, fw, "gemm", device, n)
                if r:
                    reasons = invalid_reasons(r)
                    if reasons:
                        fw_col = f"{fw}（無効: {'; '.join(reasons)}）"
                        lines.append(
                            f"| {n} | {fw_col} | {fmt_ms(r['median_s'])} | {fmt_ms(r['q1_s'])} | {fmt_ms(r['q3_s'])} | - |"
                        )
                    else:
                        lines.append(
                            f"| {n} | {fw} | {fmt_ms(r['median_s'])} | {fmt_ms(r['q1_s'])} | {fmt_ms(r['q3_s'])} | {r['gflops']:.1f} |"
                        )
                else:
                    lines.append(f"| {n} | {fw} | 計測不可 | - | - | - |")
        lines.append("")

    # (a') デバイス/tape 再利用モード（イシュー #925）。reuse 行が存在する
    # ファイルにのみ出力する（本フィールド追加前の JSONL では常にスキップ）。
    if any(r["task"] == "gemm" and r["mode"] == "reuse" for r in rows):
        lines.append(
            "### (a') GEMM（デバイス/tape 再利用モード。初期化コストとカーネル実行の分離。イシュー #925）\n"
        )
        for device in devices_in(rows, "gemm", mode="reuse"):
            sizes = sorted(
                {
                    r["size"]
                    for r in rows
                    if r["task"] == "gemm" and r["device"] == device and r["mode"] == "reuse"
                }
            )
            lines.append(f"#### {DEVICE_LABEL[device]}\n")
            lines.append(
                "| N | フレームワーク | 初期化(init_s) | 中央値 | Q1 | Q3 | GFLOP/s | fresh 中央値（参考） |"
            )
            lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
            for n in sizes:
                for fw in FRAMEWORKS:
                    r = get(rows, fw, "gemm", device, n, mode="reuse")
                    if not r:
                        continue
                    fresh = get(rows, fw, "gemm", device, n, mode="fresh")
                    fresh_col = fmt_ms(fresh["median_s"]) if fresh else "未計測"
                    init_col = fmt_ms(r["init_s"]) if r.get("init_s") is not None else "-"
                    reasons = invalid_reasons(r)
                    fw_col = f"{fw}（無効: {'; '.join(reasons)}）" if reasons else fw
                    gflops_col = "-" if reasons else f"{r['gflops']:.1f}"
                    lines.append(
                        f"| {n} | {fw_col} | {init_col} | {fmt_ms(r['median_s'])} | {fmt_ms(r['q1_s'])} | {fmt_ms(r['q3_s'])} | {gflops_col} | {fresh_col} |"
                    )
            lines.append("")

    lines.append(
        "### (b) MLP 学習（784→256→10、ReLU、バッチ 64、MSE、SGD lr=0.01、1 ステップあたり時間）\n"
    )
    lines.append("| デバイス | フレームワーク | 中央値 | Q1 | Q3 |")
    lines.append("| --- | --- | --- | --- | --- |")
    for device in devices_in(rows, "train"):
        for fw in FRAMEWORKS:
            r = get(rows, fw, "train", device)
            if r:
                lines.append(
                    f"| {device} | {fw} | {fmt_ms(r['median_s'])} | {fmt_ms(r['q1_s'])} | {fmt_ms(r['q3_s'])} |"
                )
            else:
                lines.append(f"| {device} | {fw} | 計測不可 | - | - |")
    lines.append("")

    # (b') MLP 学習 — デバイス常駐パラメータ更新モード（イシュー #957/#958/#959）。
    # reuse 行が存在するファイルにのみ出力する（(a') と同じく本フィールド追加前
    # の JSONL では常にスキップ）。gemm と異なり fandhe-ai/candle/Burn 間の
    # checksum（最終 loss）は重み初期化の違いにより一致しないため、フレーム
    # ワーク横断の参照値は選ばず、同一フレームワーク内の fresh 行とのみ突合する。
    # イシュー #959 codex-review P1 指摘: 上記 fw_col/match_col が「無効」
    # 判定する状態（checksum 不一致・checksum 突合不能・時間値の無効値）を
    # `section()` の戻り値へ反映していなかったため `--strict` が fail-open
    # のままだった（`summarize_test.py` の
    # `test_main_strict_exit_code_unaffected_by_train_reuse_rows` が固定して
    # いた挙動）。fresh 欠落のみ（`match_col == "突合不能"`、比較対象なし）
    # は gemm 側の「突合不能（検証対象外）」と同様に値そのものの正当性を
    # 否定しないため無効扱いにしない。
    has_train_reuse_invalid = False
    if any(r["task"] == "train" and r["mode"] == "reuse" for r in rows):
        lines.append(
            "### (b') MLP 学習（デバイス常駐パラメータ更新モード。ホスト経由 SGD との分離。イシュー #957/#958/#959）\n"
        )
        lines.append(
            "| デバイス | フレームワーク | 初期化(init_s) | 中央値 | Q1 | Q3 | fresh 中央値（参考） | fresh/reuse 比 | 最終 loss 突合（fresh） |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for device in devices_in(rows, "train", mode="reuse"):
            for fw in FRAMEWORKS:
                r = get(rows, fw, "train", device, mode="reuse")
                if not r:
                    continue
                fresh = get(rows, fw, "train", device, mode="fresh")
                # median_s・q1_s・q3_s・init_s（表示・比率計算に使う全時間値）
                # と checksum（最終 loss 突合）は外部 JSONL（bench-fandhe /
                # 他フレームワークの計測出力）由来であり、型・値域を検証せず
                # `fmt_ms` へ渡す・除数にする・`checksums_match` へ渡すと、
                # 文字列で TypeError、bool・NaN・Infinity・負値で不正な表示・
                # 判定を生じる（security.md「外部フォーマットのパース時検証
                # （A03）」。イシュー #959 codex-review P0 指摘: 旧実装は
                # 比率計算の median_s のみを検証しており、init_s・q1_s・q3_s・
                # fresh 側の値・checksum は未検証のまま使われていた）。
                # `_safe_time_s`（時間値: 有限かつ正のみ有効）・
                # `_safe_finite_number`（checksum: 有限数のみ有効、符号は
                # 制約しない）で使用前に検証し、不正な値は表示・計算に
                # 使わず fail-closed に「無効な値」扱いとする。
                r_median = _safe_time_s(r.get("median_s"))
                r_q1 = _safe_time_s(r.get("q1_s"))
                r_q3 = _safe_time_s(r.get("q3_s"))
                r_init = _safe_time_s(r.get("init_s")) if r.get("init_s") is not None else None
                r_checksum = _safe_finite_number(r.get("checksum"))

                init_col = fmt_ms(r_init) if r_init is not None else "-"
                median_col = fmt_ms(r_median) if r_median is not None else "無効な値"
                q1_col = fmt_ms(r_q1) if r_q1 is not None else "無効な値"
                q3_col = fmt_ms(r_q3) if r_q3 is not None else "無効な値"

                fresh_median = _safe_time_s(fresh.get("median_s")) if fresh else None
                fresh_checksum = _safe_finite_number(fresh.get("checksum")) if fresh else None

                # fresh_col は fresh 自身の有効性のみで決める（reuse 側の
                # median が無効でも fresh の計測値自体は隠さない。Bugbot
                # 指摘: reuse median_s 無効・fresh 有効時に「計測不正」で
                # 上書きされ有効な fresh 計測値が見えなくなっていた）。
                # ratio_col（fresh/reuse 比）は両者が揃って初めて計算できる
                # ため別条件で判定する。
                if fresh:
                    fresh_col = fmt_ms(fresh_median) if fresh_median is not None else "計測不正"
                    if fresh_median is not None and r_median is not None:
                        ratio_col = f"{fresh_median / r_median:.2f} 倍"
                    else:
                        ratio_col = "-"
                else:
                    fresh_col = "未計測"
                    ratio_col = "-"

                # 時間値（median_s/q1_s/q3_s）の無効値も「無効」判定の一部
                # として扱う（イシュー #959 codex-review P1 指摘）。
                row_invalid = r_median is None or r_q1 is None or r_q3 is None

                if fresh:
                    if r_checksum is None or fresh_checksum is None:
                        match_col = "突合不能（無効値）"
                        fw_col = f"{fw}（無効: checksum が不正な値）"
                        row_invalid = True
                        print(
                            f"warning: {rel}: {fw}/{device}/train/reuse の最終 loss "
                            "checksum が不正な値（非数値・NaN・Infinity 等）のため突合不能 "
                            "— 無効データとして表示",
                            file=sys.stderr,
                        )
                    elif checksums_match(r_checksum, fresh_checksum):
                        match_col = "一致"
                        fw_col = fw
                    else:
                        match_col = "不一致"
                        fw_col = f"{fw}（無効: fresh と最終 loss 不一致）"
                        row_invalid = True
                        print(
                            f"warning: {rel}: {fw}/{device}/train/reuse の最終 loss "
                            f"{r_checksum:.6f} が fresh {fresh_checksum:.6f} と不一致 "
                            "— 無効データとして表示",
                            file=sys.stderr,
                        )
                else:
                    match_col = "突合不能"
                    fw_col = fw

                if row_invalid:
                    has_train_reuse_invalid = True
                    if "（無効" not in fw_col:
                        fw_col = f"{fw}（無効: 時間値が不正な値）"

                lines.append(
                    f"| {device} | {fw_col} | {init_col} | {median_col} | "
                    f"{q1_col} | {q3_col} | {fresh_col} | {ratio_col} | {match_col} |"
                )
        lines.append("")

    lines.append(
        "### (c) 推論スループット（同 MLP forward のみ、バッチ 64。表のスループットはバッチ/秒 = 1/中央値。1 バッチ = 64 件）\n"
    )
    lines.append("| デバイス | フレームワーク | 中央値 | Q1 | Q3 | バッチ/秒 |")
    lines.append("| --- | --- | --- | --- | --- | --- |")

    def fmt_tps(v):
        # 10 バッチ/秒未満は小数 1 桁。`:.0f` だと 1 未満の値（fandhe-ai CUDA
        # 初回計測の約 0.55）が 1 に丸まり約 2 倍に見える（articles#68
        # Bugbot 指摘・イシュー #971）。
        if v < 10:
            return f"{v:.1f}"
        return f"{v:.0f}"

    for device in devices_in(rows, "infer"):
        for fw in FRAMEWORKS:
            r = get(rows, fw, "infer", device)
            if r:
                lines.append(
                    f"| {device} | {fw} | {fmt_ms(r['median_s'])} | {fmt_ms(r['q1_s'])} | {fmt_ms(r['q3_s'])} | {fmt_tps(r['throughput_per_s'])} |"
                )
            else:
                lines.append(f"| {device} | {fw} | 計測不可 | - | - | - |")
    lines.append("")

    lines.append("#### データ有効性（checksum 突合・要素単位検証。イシュー #965・#970）\n")
    # candidate_count<=1（この JSONL 内に比較対象となる他フレームワーク／
    # 他デバイス行が無い）の行は、値そのものは正当でも相互突合が原理的に
    # できていない。「一致」と混同されないよう区別して報告する
    # （イシュー #965 P2 指摘: candidate_count<=1 の行が誤って「全 checksum
    # 一致」表示に含まれ検証済みと誤認させていた問題の修正）。
    unverifiable_rows = gemm_checksum_unverifiable(rows)
    unverifiable_keys = {_row_key(r) for r in unverifiable_rows}
    verified_total = sum(
        1 for r in rows if r["task"] == "gemm" and _row_key(r) not in unverifiable_keys
    )
    if mismatches:
        for r, ref, ref_label in mismatches:
            if ref is None:
                lines.append(
                    f"- **無効（突合不能）**: {r['framework']}/{r['device']}/size={r['size']}/{r['mode']} "
                    f"— checksum {r['checksum']:.6f}、参照値を決定できず"
                )
            else:
                lines.append(
                    f"- **無効**: {r['framework']}/{r['device']}/size={r['size']}/{r['mode']} "
                    f"— checksum {r['checksum']:.6f} が参照 {ref:.6f}（{ref_label}）と不一致"
                )
    elif verified_total > 0:
        lines.append(f"- 不一致なし（相互突合できた {verified_total} 行の checksum が参照値と一致）")
    else:
        lines.append("- 相互突合できた行なし（全 gemm 行が比較対象なしで突合不能）")
    if unverifiable_rows:
        for r in unverifiable_rows:
            lines.append(
                f"- **突合不能（検証対象外）**: {r['framework']}/{r['device']}/size={r['size']}/{r['mode']} "
                f"— checksum {r['checksum']:.6f}、この JSONL 内に比較対象となる他フレームワーク／"
                "他デバイス行が無いため相互突合していない（値の正当性を否定するものではない）"
            )

    # イシュー #970: 要素単位検証の集計（checksum 突合とは独立の節）。
    unverified_rows = gemm_parity_unverified(rows)
    parity_verified_total = sum(
        1 for r in rows if r["task"] == "gemm" and parity_status(r) != "unverified"
    )
    if parity_failures:
        for r in parity_failures:
            lines.append(
                f"- **無効（要素誤差超過）**: {r['framework']}/{r['device']}/size={r['size']}/{r['mode']} "
                f"— {_parity_reason(r)}"
            )
    elif parity_verified_total > 0:
        lines.append(
            f"- 要素誤差超過なし（検証済み {parity_verified_total} 行が全て閾値内。"
            f"PARITY_ABS_TOL={PARITY_ABS_TOL:.0e}、PARITY_REL_TOL={PARITY_REL_TOL:.0e}）"
        )
    else:
        lines.append("- 要素単位検証済みの行なし（全 gemm 行が旧形式または対象外）")
    if unverified_rows:
        lines.append(
            f"- **未検証（旧形式）**: {len(unverified_rows)} 行（本フィールド追加〈イシュー #970〉前に"
            "コミットされた JSONL のため要素単位検証未実施。値の正当性を否定するものではない）"
        )
    lines.append("")
    return lines, bool(mismatches), bool(parity_failures), bool(unverified_rows), has_train_reuse_invalid


def main():
    parser = argparse.ArgumentParser(
        description="framework-compare の JSONL 計測結果を Markdown 表へ集計する"
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help="入力 JSONL（省略時は results/raw/*.jsonl を全件）",
    )
    parser.add_argument(
        "--out",
        help="出力先ファイル（省略時は標準出力。コミット済み summary.md を既定で上書きしない）",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "GEMM checksum の不一致（イシュー #965）・要素単位検証の閾値超過・"
            "要素単位検証を受けていない旧形式行（いずれもイシュー #970）・"
            "train reuse (b') の checksum 不一致／突合不能／時間値不正"
            "（イシュー #959）が1 件以上あれば終了コード 2 を返す"
            "（既定は 0 のまま警告のみ）"
        ),
    )
    args = parser.parse_args()

    inputs = args.inputs or sorted(glob.glob(os.path.join(HERE, "results/raw/*.jsonl")))
    if not inputs:
        print("error: 入力 JSONL がありません（results/raw/*.jsonl）", file=sys.stderr)
        return 1

    lines = ["# ベンチマーク集計（summarize.py 生成）\n"]
    any_checksum_mismatch = False
    any_parity_failure = False
    any_parity_unverified = False
    any_train_reuse_invalid = False
    for path in inputs:
        rows = load_rows(path)
        if not rows:
            lines.append(f"## 集計対象: {os.path.relpath(path, HERE)}\n")
            lines.append("（有効な行なし）\n")
            continue
        (
            section_lines,
            has_mismatch,
            has_parity_failure,
            has_unverified,
            has_train_reuse_invalid,
        ) = section(path, rows)
        lines.extend(section_lines)
        any_checksum_mismatch = any_checksum_mismatch or has_mismatch
        any_parity_failure = any_parity_failure or has_parity_failure
        any_parity_unverified = any_parity_unverified or has_unverified
        any_train_reuse_invalid = any_train_reuse_invalid or has_train_reuse_invalid

    # 入力 JSONL と同一ディレクトリからのみ skipped*.log を収集する。HERE
    # 固定 glob だと、別ディレクトリの JSONL を明示指定して集計した際に
    # 無関係な別ホスト・別ラウンドの skipped*.log が混ざる（articles#68
    # Bugbot 指摘・イシュー #971）。入力省略時は inputs が
    # HERE/results/raw/*.jsonl になるため、収集元は従来どおり
    # results/raw/ に一致する（後方互換）。
    input_dirs = sorted({os.path.dirname(os.path.abspath(p)) for p in inputs})
    skip_logs = sorted(
        log
        for d in input_dirs
        for log in glob.glob(os.path.join(d, "skipped*.log"))
    )
    lines.append("## 実行時失敗（skipped*.log）\n")
    any_skip = False
    for sl in skip_logs:
        for line in open(sl):
            line = line.strip()
            if line:
                any_skip = True
                lines.append(f"- **{os.path.basename(sl)}**: {line}")
    if not any_skip:
        lines.append("- なし（skipped*.log は空または不在）")
    lines.append("")

    text = "\n".join(lines) + "\n"
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        sys.stdout.write(text)

    if (
        any_checksum_mismatch
        or any_parity_failure
        or any_parity_unverified
        or any_train_reuse_invalid
    ) and args.strict:
        if any_checksum_mismatch:
            print(
                "error: --strict: 1 件以上の gemm checksum 不一致（イシュー #965）",
                file=sys.stderr,
            )
        if any_parity_failure:
            print(
                "error: --strict: 1 件以上の gemm 要素単位検証の閾値超過（イシュー #970）",
                file=sys.stderr,
            )
        if any_parity_unverified:
            print(
                "error: --strict: 1 件以上の gemm 行が要素単位検証を受けていない"
                "旧形式（イシュー #970）",
                file=sys.stderr,
            )
        if any_train_reuse_invalid:
            # イシュー #959 codex-review P1 指摘: train reuse (b') の
            # checksum 不一致・checksum 突合不能（無効値）・時間値の無効値
            # を fail-closed に --strict の失敗条件へ含める（旧実装は
            # 表示のみで section() の戻り値に反映されず fail-open だった）。
            print(
                "error: --strict: 1 件以上の train reuse 行が無効"
                "（checksum 不一致・突合不能・時間値の不正。イシュー #959）",
                file=sys.stderr,
            )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
