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
  区別して「未検証（旧形式）」と表示する（fail-closed だがデータの誤破棄を
  しない）。checksum 突合（#965）とは独立に判定し、両方に該当する行は
  理由を併記する。
- 実行時失敗（skipped*.log）節は、集計対象として渡された各入力 JSONL と
  同一ディレクトリの skipped*.log のみを集める（入力省略時は従来どおり
  results/raw/ 配下が対象。articles#68 Bugbot 指摘・イシュー #971）。
- (c) のバッチ/秒は 10 未満を小数 1 桁で表示する（`:.0f` だと 1 未満の値が
  1 に丸まり実際の約 2 倍に見えるため。articles#68 Bugbot 指摘・イシュー #971）。
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
    """
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        return False
    return math.isfinite(v)


def parity_status(row):
    """GEMM 行の要素単位検証結果（イシュー #970）を判定する。

    戻り値: "unverified" | "fail" | "ok"

    - "unverified": `parity_fail_count` キー自体が存在しない（本フィールド
      追加前にコミットされた旧形式 JSONL）。`row.get(...)` だけでは
      「キー欠損（None）」と「値が JSON `null`（= Python None、非有限
      センチネル）」を区別できないため、まずキーの存在を検査する
      （欠損＝旧形式・未検証と、存在するが不正＝無効を混同しない）。
    - "fail": `parity_fail_count` キーは存在するが、4 フィールドのいずれか
      の型が不正（数値でない・`null`・非有限）、値域が不正（`parity_total`
      が 0 以下、`parity_fail_count` が負または `parity_total` 超過、誤差
      2 項が負）、または `parity_fail_count > 0`。壊れた入力を黙って
      「一致」扱いにしない（fail-closed。イシュー #970 PR #978 codex-review
      P0 指摘: 型検査のみでは `parity_fail_count=-1`・`parity_total=0`・
      負の誤差値を伴う外部 JSONL が "ok" 判定へ通ってしまっていた）。
    - "ok": 4 フィールドすべてが妥当な数値・値域で `parity_fail_count == 0`。
    """
    if "parity_fail_count" not in row:
        return "unverified"
    fail_count = row.get("parity_fail_count")
    total = row.get("parity_total")
    max_abs = row.get("parity_max_abs_err")
    max_rel = row.get("parity_max_rel_err")
    if not _is_plain_number(fail_count) or not _is_plain_number(total):
        return "fail"
    if not _is_plain_number(max_abs) or not _is_plain_number(max_rel):
        return "fail"
    # 値域検証（イシュー #970 PR #978 codex-review P0）: total は正、
    # fail_count は [0, total] の範囲、誤差 2 項は非負でなければならない。
    if total <= 0:
        return "fail"
    if fail_count < 0 or fail_count > total:
        return "fail"
    if max_abs < 0 or max_rel < 0:
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
    abs_str = f"{max_abs:.3e}" if _is_plain_number(max_abs) else "null"
    rel_str = f"{max_rel:.3e}" if _is_plain_number(max_rel) else "null"
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
    return lines, bool(mismatches), bool(parity_failures)


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
            "GEMM checksum の不一致（イシュー #965）または要素単位検証の閾値超過"
            "（イシュー #970）が 1 件以上あれば終了コード 2 を返す（既定は 0 のまま警告のみ）"
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
    for path in inputs:
        rows = load_rows(path)
        if not rows:
            lines.append(f"## 集計対象: {os.path.relpath(path, HERE)}\n")
            lines.append("（有効な行なし）\n")
            continue
        section_lines, has_mismatch, has_parity_failure = section(path, rows)
        lines.extend(section_lines)
        any_checksum_mismatch = any_checksum_mismatch or has_mismatch
        any_parity_failure = any_parity_failure or has_parity_failure

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

    if (any_checksum_mismatch or any_parity_failure) and args.strict:
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
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
