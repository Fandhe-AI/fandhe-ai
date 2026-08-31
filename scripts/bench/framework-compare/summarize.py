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
- (b'') MLP 学習の 1 step フェーズ分解（イシュー #1009。`bench-fandhe`
  `--task train --phases` が出力する `task:"train_phases"` 行）。
  `(device, mode)` ごとに `phase_index` 昇順で表示し、`step_total` 比
  （= phase median / step_total median）を参考値として添える。`(b)`/`(b')`
  とは独立の節で、`task != "train_phases"` の既存関数（`get`/`devices_in`
  等）は本フィールドを一切読まないため既存表への影響はない。
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
- (b'') train phases（イシュー #1009）: `task:"train_phases"` 行の
  `step_total` phase の `median_s`/`q1_s`/`q3_s`・`init_s`（reuse 行のみ
  必須）は `_safe_time_s`（> 0 のみ有効）、`step_total` 以外の phase の
  `median_s`/`q1_s`/`q3_s` は `_safe_phase_time_s`（>= 0 有効。sub-ns
  区間の一部標本が計時クロックの分解能により 0 になりうるための専用
  検証。イシュー #1010・`_safe_phase_time_s` docstring 参照）、`phase` は
  非空 `str`、
  `phase_index` は非負整数として検証する。
  `step_total` phase の欠落・`phase`/`phase_index` の重複・`step_total`
  比が 100% を超える不整合（時間値の集計不整合を示す）に加え、mode ごとの
  必須 phase 名・順序・件数（`_TRAIN_PHASES_REQUIRED_PHASES`。fresh:
  `phase_index` 0..9・reuse: 0..7）との不一致（`backward` 等の必須 phase
  行が欠落していても `step_total` さえ残っていれば有効判定してしまって
  いた codex-review 指摘・PR #1055 への対処）は表で「無効」表示
  し `--strict` の失敗条件にも含める（`section()` の戻り値に
  `has_train_phases_invalid` を追加）。`task != "train_phases"` の既存
  節（(a)/(a')/(b)/(b')/(c)・checksum 突合・parity 判定・`devices_in`）は
  本フィールドを一切読まないため影響を受けない。
- 目標達成ゲート（イシュー #1051。`--target candle`/`--target burn`）:
  親 #1049「横並び再計測と目標達成ゲート」の完了判定を人間の目視に頼らず
  機械的に行うためのオプション。**同一入力 JSONL ファイル内**（1 ファイル
  = 1 環境。モジュール docstring 冒頭の方針と同じ）の `(task, device,
  size)` ごとに、fandhe-ai と `--target` 指定フレームワークの `median_s`
  を突合し、fandhe-ai が同等以上の性能（`fandhe_median_s <=
  target_median_s`）かを判定する。ファイルをまたいだ突合は環境混同になる
  ため行わない。fandhe-ai・target とも reuse 行があれば reuse を優先し
  無ければ fresh を使う（`_pick_row_for_gate`。infer には reuse 行が
  存在しないため常に fresh）。checksum 不一致・要素誤差超過・train reuse
  の checksum 不一致等（既存の無効判定と同じ規則）に該当する行は「達成」
  と判定せず「判定不能（無効データ）」に倒す（壊れた計算の実行時間で
  達成判定しない。A08）。`--target` 指定時、1 件でも未達／判定不能が
  あれば終了コード 3 を返す（`--strict` 由来の 2 と区別。両方の条件を
  満たす場合はデータ無効の解消を優先させるため 2 を返す）。判定式・許容
  マージンはユーザー承認なく緩めない（`.claude/rules/coding-rust.md`）。
"""

import argparse
import glob
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

FRAMEWORKS = ["fandhe-ai", "candle", "burn"]
DEVICE_ORDER = ["cpu", "metal", "cuda"]
DEVICE_LABEL = {"cpu": "CPU", "metal": "Metal", "cuda": "CUDA"}

# イシュー #1051: 目標達成ゲート（--target）の対象タスク・比較先 allowlist。
# `--target` は argparse の `choices=GATE_TARGET_CHOICES` で検証するため、
# CLI から任意の framework 文字列が Markdown 出力・辞書キーへ流れ込むことは
# ない（security.md A03）。fandhe-ai 自身を target にする比較は無意味な
# ため FRAMEWORKS から除外する。
GATE_TARGET_CHOICES = [fw for fw in FRAMEWORKS if fw != "fandhe-ai"]
GATE_TASKS = ("gemm", "train", "infer")

# codex-review P0 指摘（PR #1082 3 巡目）: `run_all.sh`/`run_all_cuda.sh` の
# `run()` 関数は実行時失敗を JSONL には一切書かず（"never fabricated"。
# 両スクリプト冒頭コメント参照）`skipped*.log` にのみ記録する。従来の
# `target_gate`/`_gate_devices` は成功して JSONL に残った fandhe-ai/target
# 行だけからデバイス集合・size 集合を導出していたため、あるデバイスの
# 全実行（または既知デバイス上の特定 size）が丸ごと失敗して
# skipped*.log にしか記録されなかった場合、その組合せが判定対象に一切
# 現れず「全達成」に混入する fail-open があった。`_parse_skip_failure`
# は両スクリプトが書き出す行形式（`<bin> task=<task> device=<device>
# size=<size> mode=<mode> extra=<extra> : <エラーメッセージ>`。両
# スクリプトの `run()` 定義で同一書式）を解析し、`main()` が
# `gate_records_all` へ判定不能レコードとして注入する
# （`_inject_skip_failures_into_gate` 参照）。
_SKIP_BIN_TO_FRAMEWORK = {
    "bench-fandhe": "fandhe-ai",
    "bench-candle": "candle",
    "bench-burn": "burn",
}

_SKIP_LINE_RE = re.compile(
    r"^(?P<bin>\S+)\s+task=(?P<task>\S+)\s+device=(?P<device>\S+)\s+"
    r"size=(?P<size>\S+)\s+mode=(?P<mode>\S+)\s+extra=(?P<extra>\S+)\s*:\s*(?P<err>.*)$"
)


def _parse_skip_failure(line):
    """`skipped*.log` の 1 行を解析し、ゲート判定に使える構造化情報を返す。

    `run_all.sh`/`run_all_cuda.sh` の `run()` 関数（両スクリプト同一書式）が
    書き出す `<bin> task=<task> device=<device> size=<size> mode=<mode>
    extra=<extra> : <エラーメッセージ>` を対象とする。`build()`
    （`run_all_cuda.sh` のビルド失敗記録）等、この書式に一致しない行は
    `task`/`device`/`size` を特定できないため全て `None` に倒す
    （呼び出し側 `_inject_skip_failures_into_gate` が device 単位の粗い
    判定不能レコードへ fail-closed に倒す）。

    `framework`/`task`/`device` は許可された値（`_SKIP_BIN_TO_FRAMEWORK`・
    `GATE_TASKS`・`DEVICE_ORDER`）のみを採用する（security.md A03。外部
    プロセスの stderr 出力を含む行を未検証のまま辞書キー・Markdown へ
    通さない）。`size` は `_valid_gate_size` で検証し、不正・パース不能
    なら `None` に倒す（`--size` に非整数が渡ることはないが、外部ログの
    内容は信頼しない）。

    `mode`（PR #1082 5 巡目 codex-review P0 指摘その1）: `--mode` の
    allowlist（`_TRAIN_PHASES_MODES`。producer 側 `bench_common::
    parse_cli_from` の `--mode` allowlist と同じ値域）で検証する。
    旧実装は `mode` を一切保持していなかったため、`_inject_skip_failures_
    into_gate` は「どの実行（framework の fresh/reuse どちらか）が
    失敗したか」を区別できず、失敗した実行とは異なるモードの成功データ
    のみで「達成」を出してしまう fail-open があった。

    戻り値: {"framework": str|None, "task": str|None, "device": str|None,
              "size": int|None, "mode": str|None, "raw": str}
    """
    m = _SKIP_LINE_RE.match(line)
    if not m:
        return {
            "framework": None,
            "task": None,
            "device": None,
            "size": None,
            "mode": None,
            "raw": line,
        }
    framework = _SKIP_BIN_TO_FRAMEWORK.get(m.group("bin"))
    task = m.group("task") if m.group("task") in GATE_TASKS else None
    device = m.group("device") if m.group("device") in DEVICE_ORDER else None
    mode = m.group("mode") if m.group("mode") in _TRAIN_PHASES_MODES else None
    size = None
    try:
        size_candidate = int(m.group("size"))
    except ValueError:
        size_candidate = None
    if size_candidate is not None and _valid_gate_size(size_candidate):
        size = size_candidate
    return {
        "framework": framework,
        "task": task,
        "device": device,
        "size": size,
        "mode": mode,
        "raw": line,
    }


def _skip_log_paths_for_input(path):
    """入力 JSONL ファイルに対応する skipped*.log の一覧を返す（環境スコープ）。

    codex P0・Bugbot High 指摘（PR #1082 4 巡目）: skip 失敗の注入を
    全入力ファイル横断で共有される `gate_records_all` へ行うと、環境 A の
    JSONL に `gemm/cuda/N=256` の達成行があるだけで、環境 B（別ディレクトリ
    ・skipped*.log にしか記録が残っていない）の同じ組の失敗が
    `existing_keys` により注入されず握りつぶされる。`target_gate` 自身が
    課している「ファイルをまたいだ突合は環境混同になるため行わない」契約
    （モジュール docstring 参照）に反していた。

    加えて、`run_all.sh`/`run_all_cuda.sh` は同一ディレクトリ
    （`results/raw/`）に複数環境の JSONL を書き出す運用（例:
    `results.jsonl`〈環境 1〉と `results-cuda.jsonl`〈環境 2〉が同居。
    README「計測結果」節参照）のため、単純にディレクトリ単位で
    skipped*.log を共有すると同一ディレクトリ内の別環境の失敗まで
    混入しうる。両スクリプトの命名規約（`results<suffix>.jsonl` <->
    `skipped<suffix>.log`。例: `results.jsonl`<->`skipped.log`、
    `results-cuda.jsonl`<->`skipped-cuda.log`）に厳密一致する
    skipped*.log が存在すればそれだけを対象にする（最も精度が高い環境
    識別）。一致するファイルが存在しない場合（命名規約に従わないファイル名
    ・アドホックな実行等で環境識別子が特定できない）は、同じディレクトリの
    skipped*.log 全件へフォールバックする（判定不能の握りつぶしを避ける
    fail-closed 側の選択。ディレクトリという境界自体は超えない）。

    戻り値: 一致した skipped*.log の絶対パスのリスト（重複なし・昇順）。
    """
    d = os.path.dirname(os.path.abspath(path))
    base = os.path.basename(path)
    if base.endswith(".jsonl"):
        stem = base[: -len(".jsonl")]
        if stem.startswith("results"):
            suffix = stem[len("results") :]
            expected = os.path.join(d, f"skipped{suffix}.log")
            if os.path.isfile(expected):
                return [expected]
    return sorted(glob.glob(os.path.join(d, "skipped*.log")))


def _skip_failures_for_paths(log_paths):
    """`log_paths`（`_skip_log_paths_for_input` が返す skipped*.log 群）を
    読み、`_parse_skip_failure` で構造化した失敗のリストを返す。
    """
    failures = []
    for log_path in log_paths:
        for line in open(log_path):
            line = line.strip()
            if line:
                failures.append(_parse_skip_failure(line))
    return failures


def _sanitize_skip_raw_for_display(raw, max_len=120):
    """外部プロセスの stderr 出力を含む skipped*.log の生行（`raw`）を
    Markdown 表セルへ安全に埋め込める表現へ変換する。

    codex-review P0 指摘（PR #1082 6 巡目・security.md A03）: `raw` は
    ベンチバイナリの stderr（外部プロセス出力）をそのまま含む未信頼
    文字列であり、`target_gate_section()` は `reason` を `|` 区切りの
    Markdown 表セル・箇条書きへ無加工で埋め込む。`raw` に `|` が含まれると
    表構造が壊れ、`<script>` 等の HTML/Markdown 構文を含む場合はそのまま
    出力ページへ混入しうる（生成物が GitHub 等でレンダリングされる場合の
    注入経路になる）。

    呼び出し元は `target_gate_section()` の `reason` 生成箇所に加え、
    `main()` 内の「実行時失敗（skipped*.log）」節（skipped*.log の生行を
    箇条書きへ埋め込む別コードパス）でも使う（イシュー #1085）。

    - 改行・連続空白は単一の半角スペースへ正規化する（表セル内で改行する
      とレンダリングが崩れるため）。
    - `&` を先に `&amp;` へ変換してから `<`/`>` を実体参照へ、`|` を
      Markdown のエスケープ形式 `\\|` へ変換する（`&` を後回しにすると
      置換で生成した実体参照自体を再エスケープしてしまう二重エスケープ
      を防ぐ）。
    - `max_len` 文字を超える場合は切り詰めて末尾に `…` を付す（表の
      横幅を制御し、巨大な stderr 出力の丸ごと埋め込みも防ぐ）。
    """
    normalized = " ".join(raw.split())
    escaped = (
        normalized.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("|", "\\|")
    )
    if len(escaped) > max_len:
        escaped = escaped[:max_len] + "…"
    return escaped


def _inject_skip_failures_into_gate(gate_records, skip_failures, target, rows):
    """skip 由来の失敗を `gate_records` へ判定不能レコードとして注入する。

    **呼び出し契約（codex P0・Bugbot High 指摘・PR #1082 4 巡目）**:
    `gate_records`・`skip_failures`・`rows` は必ず **単一の入力ファイル
    （＝単一環境）に属するものだけ** を渡すこと。全入力ファイル横断で
    集約した `gate_records_all` を渡すと、環境 A の JSONL に達成行がある
    `(task, device, size)` の組について、環境 B（別ファイル・skipped*.log
    にしか記録が残っていない）の同じ組の失敗が握りつぶされる
    （`target_gate` 自身の「ファイルをまたいだ突合は環境混同になるため
    行わない」契約への違反。呼び出し元 `main()` は空ファイル用
    プレースホルダには `rows=[]` を渡す。`_skip_log_paths_for_input`/
    `_skip_failures_for_paths` も参照）。

    **(task, device, size) 単位ではなく (framework, task, device, size,
    mode) 単位で判定する（codex-review P0 指摘その1・PR #1082 5 巡目）**:
    旧実装は `(task, device, size)` のみで重複を判定していたため、
    fandhe-ai の fresh 実行が失敗して skipped*.log に記録されていても、
    同じ組の fandhe-ai reuse 実行が成功し（`_pick_row_for_gate` は
    reuse を優先）target 側の fresh 実行も成功していれば「達成」の
    まま握りつぶされていた（train reuse は比較対象の fresh 行が
    存在しない場合を「値そのものの正当性を否定しない」= 有効値として
    扱う仕様〈`_train_reuse_row_invalid_reason` docstring 参照〉のため、
    fresh 実行が実際には試みられて失敗したという明確な証拠があっても
    checksum 突合不能な状態を見逃していた）。本関数は skip 失敗
    1 件ごとに、**その正確な (framework, task, device, size, mode) の
    実行が実際に成功して `rows`（同一環境の実データ）へ残っているか**
    を `get()` で確認する。残っていれば「再実行後に成功した古い失敗
    記録（stale）」とみなして無視する（重複抑止はこの場合のみ）。
    残っていなければ、その (task, device, size) の既存レコードが
    「達成」であっても判定不能へ格下げする（既に判定不能なら理由は
    上書きしない。1 件でも本物の失敗があれば「達成」を名乗らせない
    という fail-closed の原則）。既存レコードが無い場合は新規の判定
    不能レコードを追加する（従来どおり）。

    **framework が特定できない行も握りつぶさない（codex-review P0
    指摘その2・PR #1082 5 巡目）**: 正規表現自体は一致し `task`/`device`
    が妥当な値でも、binary 名が `_SKIP_BIN_TO_FRAMEWORK` の allowlist に
    無ければ `framework` は `None` になる。旧実装は `task`/`device` が
    `None` の行のみを「詳細不明」の粗い判定不能レコードへ倒し、
    `framework is None` はその後の `sf["framework"] not in (...)` 判定で
    `None not in (...)` が常に真になるため無条件で `continue`（無視）
    されていた（未知 binary 名の失敗が握りつぶされる fail-open）。
    `framework is None` も `task`/`device` が `None` の場合と同じ粗い
    「詳細不明」判定不能レコードへ倒す。

    `task`/`device`/`framework` のいずれかを特定できない行（ビルド失敗・
    未知 binary 名等）は、値を捏造せず device 単位より粗い「詳細不明」の
    判定不能レコード 1 件（重複行はまとめる）に倒す。

    戻り値: 新規追加または格下げされたレコードのリスト（Markdown 節への
    表示用。空なら `[]`）。`gate_records` は本関数の呼び出しにより
    直接変更される（呼び出し元が同じ list オブジェクトを
    `gate_records_all` へ extend する運用を前提に、in-place 追加のみ
    行い新しい list は作らない）。
    """
    existing_by_key = {(r["task"], r["device"], r["size"]): r for r in gate_records}
    injected = []
    seen_unparsed = set()
    for sf in skip_failures:
        if sf["task"] is None or sf["device"] is None or sf["framework"] is None:
            # codex-review P0 指摘その2（PR #1082 5 巡目）: task/device が
            # 妥当でも framework が allowlist 外で None になるケースも
            # ここへ倒す（docstring 参照）。
            if sf["raw"] in seen_unparsed:
                continue
            seen_unparsed.add(sf["raw"])
            # codex-review P0 指摘（PR #1082 6 巡目・security.md A03）:
            # `sf["raw"]`（ベンチバイナリの stderr を含む未信頼文字列）を
            # 無加工で `reason` に格納すると、`target_gate_section()` が
            # `|` 区切りの Markdown 表セル・箇条書きへそのまま埋め込むため
            # `|` で表構造を、`<script>` 等で出力ページの HTML/Markdown
            # 構文を改変できてしまう。理由文の主要部分は固定文にし、
            # 生内容は `_sanitize_skip_raw_for_display`（長さ制限・
            # Markdown/HTML エスケープ）を通した安全な表現のみを付記する。
            record = {
                "task": "-",
                "device": "-",
                "size": None,
                "fandhe_mode": None,
                "target_mode": None,
                "fandhe_median": None,
                "target_median": None,
                "ratio": None,
                "status": "undeterminable",
                "reason": (
                    "skipped ログに未解析の失敗記録あり: "
                    f"{_sanitize_skip_raw_for_display(sf['raw'])}"
                ),
                "note": None,
            }
            gate_records.append(record)
            injected.append(record)
            continue
        if sf["framework"] not in ("fandhe-ai", target):
            continue
        mode = sf.get("mode")
        # Bugbot Medium 指摘（PR #1082 6 巡目）: `sf["size"] is None`
        # （size をパースできなかった skip 行）のまま `get(..., size=None,
        # ...)` を呼ぶと、`get()` は `size=None` を「size で絞らない」と
        # 解釈するため、同一 framework/task/device/mode の**任意の** size
        # の成功行が存在するだけで stale 判定されてしまう（size が
        # 分からない失敗を、たまたま別 size の成功と混同して握りつぶす
        # fail-open）。size が特定できている場合のみ stale 抑止の対象に
        # する。
        if (
            mode is not None
            and sf["size"] is not None
            and get(
                rows, sf["framework"], sf["task"], sf["device"], sf["size"], mode=mode
            )
            is not None
        ):
            # codex-review P0 指摘その1（PR #1082 5 巡目）: この正確な
            # (framework, task, device, size, mode) の実行は実際には
            # 成功して `rows` に残っている＝skipped*.log は再実行前の
            # 古い失敗記録（stale）。判定不能化しない（docstring 参照）。
            continue
        key = (sf["task"], sf["device"], sf["size"])
        size_label = f"N={sf['size']}" if sf["size"] is not None else "size 不明"
        mode_label = mode if mode is not None else "mode 不明"
        reason = f"skipped*.log に実行時失敗（{sf['framework']}・{mode_label}・{size_label}）"
        existing = existing_by_key.get(key)
        if existing is not None:
            if existing["status"] != "undeterminable":
                existing["status"] = "undeterminable"
                existing["reason"] = reason
                injected.append(existing)
            continue
        record = {
            "task": sf["task"],
            "device": sf["device"],
            "size": sf["size"],
            "fandhe_mode": None,
            "target_mode": None,
            "fandhe_median": None,
            "target_median": None,
            "ratio": None,
            "status": "undeterminable",
            "reason": reason,
            "note": None,
        }
        existing_by_key[key] = record
        gate_records.append(record)
        injected.append(record)
    return injected

# `train_phases`（(b'') 節。イシュー #1009）専用の allowlist。producer 側の
# 契約（`bench_common::parse_cli_from` の `--mode` allowlist・`PHASE_*` 定数
# の値域）と同じ値域に固定する。security.md A03: 外部 JSONL 由来の値を
# 辞書キーへ使う前・Markdown へ出力する前に allowlist 検証する
# （codex-review 指摘。PR #1055）。
_TRAIN_PHASES_MODES = ("fresh", "reuse")
_PHASE_NAME_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789_")

# `bench-fandhe --task train --phases`（本 harness の
# `bench-fandhe/src/main.rs` の `PHASE_*` 定数群）が mode ごとに出力する
# phase 名の完全な集合・順序（`phase_index` 0 始まり連番）。codex-review
# 指摘（PR #1055）: `_train_phases_validate` が `step_total` の存在のみを
# 検証し、`backward` 等の必須 phase 行が欠落しても `step_total` さえ残って
# いれば有効判定してしまっていたため、mode ごとの必須 phase 名・順序・件数
# を突合する（集合・順序・件数のいずれかが不一致なら当該グループ全体を
# 無効とする）。producer 側で phase を追加・変更した場合は本定数も追従
# させる必要がある（`bench-fandhe/src/main.rs` 側の doc コメント参照）。
_TRAIN_PHASES_REQUIRED_PHASES = {
    "fresh": (
        "tape_build",
        "leaf_register",
        "forward",
        "loss_readout",
        "backward",
        "param_readout",
        "host_sgd",
        "apply_params",
        "tape_drop",
        "step_total",
    ),
    "reuse": (
        "tape_build",
        "leaf_register",
        "forward_resident",
        "loss_readout",
        "backward",
        "device_update",
        "tape_drop",
        "step_total",
    ),
}


def _valid_phase_name(value):
    """`phase` 文字列を producer 側（`bench_common::validate_phase_name`）と
    同じ `[a-z0-9_]+`（非空）allowlist で検証する。`str` 以外（配列・
    オブジェクト等の手組み JSON 混入）は無条件で拒否する。
    """
    return (
        isinstance(value, str)
        and len(value) > 0
        and all(c in _PHASE_NAME_CHARS for c in value)
    )


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

    `_is_plain_number` は任意精度の `int`（例: `10**1000`）を有限として
    許容するため、`float(v)` 自体が `OverflowError: int too large to
    convert to float` を送出しうる（イシュー #959 codex-review 2 巡目 P0
    指摘: 巨大整数 1 件の混入で集計スクリプト全体が例外終了していた）。
    変換不能な値も「無効な値」として `None` に倒す（fail-closed）。
    """
    if not _is_plain_number(v):
        return None
    try:
        fv = float(v)
    except OverflowError:
        return None
    return fv if fv > 0 else None


def _safe_phase_time_s(v):
    """`train_phases`（(b'') 節。イシュー #1010）の phase 行専用の時間値検証。

    `bench-fandhe --task train --phases` の producer 側は各区間の時間を
    9 桁固定小数の秒（ナノ秒単位）で JSONL へ書き出す（`bench-common` の
    JSONL シリアライズ契約）。`tape_build` のような sub-100 ns 区間は
    41 ns なら `0.000000041` としてそのまま表現でき、9 桁固定小数への
    シリアライズ自体は `0.000000000` へ丸めない（41/42 ns の実測値が
    生データにそのまま残っていることでも確認済み）。実際に median_s が
    0 になるのは、同一区間の 5 反復中の一部標本で `Instant::now()` の
    連続 2 回呼び出しが計時クロックの分解能（OS・プラットフォーム依存の
    タイマ粒度）未満の間隔しか空かずに同一時刻を返し、区間長が厳密に
    0 と計測されるためである（sub-ns の演算コストを計時できないタイマ
    分解能の限界であり、シリアライズの丸め誤差ではない）。0 は sub-ns
    区間の実測値としてあり得る妥当な下限であり、`_safe_time_s` が本来
    弾きたい不正値（負値・NaN・Infinity・bool・非数・巨大整数の変換不能
    値）とは性質が異なるため、phase 行（`step_total` を除く）の
    median_s/q1_s/q3_s に限り本関数で 0 を許容する（イシュー #1010
    実装時に `summarize.py --strict` が cpu fresh/reuse の sub-ns 区間で
    誤って exit 2 になっていたことへの対処）。fail-closed の対象（負値・
    NaN・Infinity・非数・巨大整数）は `_safe_time_s` と同じ判定基準の
    まま変えない。呼び出し側 `_train_phases_validate` は `step_total`
    行・`init_s`・比率計算の分母（`step_total_median`）には従来どおり
    `_safe_time_s`（> 0 のみ有効）を使い続ける（0 秒の step_total・
    init_s は実測として不合理であり、比率計算のゼロ除算も避ける）。
    """
    if not _is_plain_number(v):
        return None
    try:
        fv = float(v)
    except OverflowError:
        return None
    return fv if fv >= 0 else None


def _safe_finite_number(v):
    """外部 JSONL 由来の数値（checksum 等、正値制約のないもの）を使用前に
    検証する。`_is_plain_number` と同じ bool・NaN・Infinity 除外に加え、
    `float` へ正規化する（イシュー #959 codex-review P0 指摘。`checksums_match`
    への未検証な受け渡しを避ける）。有効なら `float` を、無効なら `None` を
    返す。

    `_safe_time_s` と同じ理由（巨大整数 `int` の `float` 変換）で
    `OverflowError` を捕捉し `None` に倒す（イシュー #959 codex-review
    2 巡目 P0 指摘）。
    """
    if not _is_plain_number(v):
        return None
    try:
        return float(v)
    except OverflowError:
        return None


def load_rows(path):
    # mode（イシュー #925）欠損は "fresh" 扱い（本フィールド追加前にコミット
    # 済みの JSONL との互換維持。モジュール docstring 参照）。
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    for r in rows:
        r.setdefault("mode", "fresh")
        # イシュー #1042 codex-review P0 指摘（PR #1091）: `tf32` は外部
        # JSONL 由来の値であり、`bool(r.get("tf32", False))` は文字列
        # `"false"`・空でない配列／オブジェクト等の非 bool 値も真として
        # 誤って受理してしまう fail-open の欠陥だった（本来 FP32 として
        # 通常ゲートに含めるべき行が、型検証なしに TF32 専用扱いへ
        # 誤って除外されうる）。ここでキー欠損（`False` 扱い。互換規約
        # ドキュメント `get()`／`devices_in()` docstring 参照）または
        # 厳密な `bool` 型であることを検証し、不正型はロード全体を
        # エラー終了させる（fail-closed。AGENTS.md「外部フォーマットの
        # パース検証」「fail-closed の維持」）。検証後は各利用箇所が
        # `r.get("tf32", False) is True` のように厳密な bool 一致で
        # 判定できる（`bool(...)` での再変換は行わない）。
        if "tf32" in r and not isinstance(r["tf32"], bool):
            raise ValueError(
                f"{path}: 不正な 'tf32' フィールド型（bool を期待）: "
                f"{r['tf32']!r}（行: {r!r}）"
            )
    return rows


def get(rows, fw, task, device, size=None, mode="fresh", tf32=False):
    """`(framework, task, device, size, mode, tf32)` に一致する最初の行を返す。

    `size=None` は size 条件を適用しない（呼び出し元がフレームワーク・
    デバイス・mode のみで絞り込みたい場合の既定動作）。`size` を指定する
    場合、行側の `size` は `r.get("size")` で取得し `_valid_gate_size` で
    検証したうえで比較する（Bugbot Medium 指摘・PR #1082 2 巡目: 直接
    `r["size"]` を読むと `size` キー欠損行で `KeyError`、`bool`（`True ==
    1`）混入行で意図しない一致が起こりうる。行の `size` が不正な場合は
    比較対象から除外する＝一致しないものとして扱う。fail-closed）。

    `tf32`（既定 `False`。イシュー #1042）: `r.get("tf32", False)` と一致
    する行のみを対象にする（キー欠損 = `False` の互換規約。`Record.tf32`
    ドキュメンテーションコメント参照）。既定を `False` にすることで、
    通常の呼び出し元（`section()` の (a) GEMM 節・`target_gate` 系）が
    明示指定なしに TF32 opt-in 行を誤って FP32 行として拾わないようにする
    （fail-open 防止）。
    """
    for r in rows:
        if (
            r["framework"] == fw
            and r["task"] == task
            and r["device"] == device
            and r["mode"] == mode
            and (r.get("tf32", False) is True) == tf32
        ):
            if size is None:
                return r
            row_size = r.get("size")
            if _valid_gate_size(row_size) and row_size == size:
                return r
    return None


def devices_in(rows, task, mode="fresh", tf32=False):
    """`task`／`mode` に一致するデバイス一覧（`DEVICE_ORDER` 順）を返す。

    `tf32`（既定 `False`。イシュー #1042）: `get()` と同じ
    `r.get("tf32", False)` 一致規約でデバイスを絞り込む。既定 `False` の
    呼び出し元（(a) GEMM 節・(a') reuse 節）が、TF32 opt-in 専用に
    計測されたデバイス（FP32 gemm 行を持たない）まで拾って「計測不可」
    プレースホルダ行を作らないようにするため（Cursor Bugbot Low
    指摘・PR #1091。TF32 専用行は別途 (a-tf32) 節が表示する）。
    """
    present = {
        r["device"]
        for r in rows
        if r["task"] == task and r["mode"] == mode and (r.get("tf32", False) is True) == tf32
    }
    return [d for d in DEVICE_ORDER if d in present]


def _get_train_infer_row(rows, fw, task, device, mode="fresh"):
    """train/infer 表示節（(b)/(c)）専用の行取得。既定の FP32
    （`tf32=False`）行を優先し、無ければ TF32 強制フレームワーク（burn
    CUDA 等）の行をフォールバックとして返す。戻り値は `(row, is_tf32)`
    （該当行が無ければ `(None, False)`）。

    イシュー #1042 Cursor Bugbot 指摘（PR #1091・Medium）: burn の CUDA
    train/infer は常時 TF32（FP32 厳密経路を持たない。
    `bench-burn/src/main.rs` の `tf32: cli.device == "cuda"`）で
    記録されるが、`get()` の既定 `tf32=False` のみで探索すると burn
    CUDA train/infer 行が「計測不可」表示になり、成功した計測が欠測
    として消える。GEMM の `--tf32` opt-in（同一フレームワークが
    FP32/TF32 両方の行を持ちうるため既定 FP32 探索から除外し、TF32 行は
    別途 (a-tf32) 節が表示する）とは異なり、train/infer では TF32 が
    当該フレームワーク・デバイスの唯一の実測値であるため、表示側では
    フォールバックとして拾う（精度条件が異なることは呼び出し元が
    `is_tf32` を見て列に注記する）。
    """
    row = get(rows, fw, task, device, mode=mode, tf32=False)
    if row is not None:
        return row, False
    row = get(rows, fw, task, device, mode=mode, tf32=True)
    if row is not None:
        return row, True
    return None, False


def _devices_in_train_infer(rows, task, mode="fresh"):
    """train/infer 表示節（(b)/(c)）専用のデバイス一覧（`DEVICE_ORDER`
    順）。`devices_in()` と異なり `tf32` の値を問わず union で集める
    （`_get_train_infer_row` と対になるヘルパー。理由は同関数の
    docstring 参照）。
    """
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

    `size` は `_valid_gate_size` で検証済みの値のみを対象にする（イシュー
    #1051 codex-review P0 指摘・PR #1082: 本関数は `target_gate` の起点
    〈`gemm_checksum_mismatches` 経由〉から呼ばれるため、外部 JSONL の
    `size` が配列・文字列混在等の不正値だと集合化・`sorted()` で例外終了
    しうる。不正な size を持つ行はここで黙って除外する〈本関数の役割は
    checksum 参照値の算出であり、size 自体の妥当性判定は `target_gate` 側
    の `_valid_gate_size` 検査が別途「判定不能」として明示する〉）。
    """
    # イシュー #1042: `tf32` 行はここでは除外する。TF32 は FP32 と異なる
    # 精度（reduced precision accumulation）で計算するため、同一 size で
    # FP32 行と混在させると多数決クラスタ・優先経路の判定が TF32 行の
    # checksum に引きずられ、正当な FP32 行を「孤立した誤値」と誤判定
    # しうる（fail-open のおそれ）。TF32 行自身の checksum 妥当性検証は
    # `section()` の TF32 専用節（`_tf32_gemm_reference`）が別途担う。
    sizes = sorted(
        {
            r["size"]
            for r in rows
            if r["task"] == "gemm" and _valid_gate_size(r.get("size")) and r.get("tf32", False) is not True
        }
    )
    result = {}
    for size in sizes:
        candidates = [
            r
            for r in rows
            if r["task"] == "gemm"
            and r["size"] == size
            and r["mode"] == "fresh"
            and r.get("tf32", False) is not True
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
        # イシュー #1042: TF32 行は FP32 参照との単純突合対象から除外する
        # （`gemm_checksum_reference` のコメント参照。専用の妥当性検証は
        # `section()` の TF32 節が別途担う）。
        if r.get("tf32", False) is True:
            continue
        # `reference` のキーは `_valid_gate_size` 検証済みの size のみ
        # （`gemm_checksum_reference` docstring 参照）。`r["size"]` が不正
        # （配列・オブジェクト等の unhashable 値）だと `dict.get()` 自体が
        # `TypeError: unhashable type` を送出し呼び出し元（`target_gate`
        # は本関数を起点で呼ぶ）が例外終了する（イシュー #1051
        # codex-review P0 指摘・PR #1082 2 巡目）。不正な size の行は
        # ここでは扱わず、判定不能レコード化は呼び出し元
        # （`target_gate` の `invalid_size_rows`）の責務とする。
        if not _valid_gate_size(r.get("size")):
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
        # イシュー #1042: `gemm_checksum_mismatches` と同じ理由で TF32 行を
        # 除外する。
        if r.get("tf32", False) is True:
            continue
        # `gemm_checksum_mismatches` と同じ理由（不正 size での
        # `dict.get()` 例外終了防止。イシュー #1051 codex-review P0
        # 指摘 2 巡目・PR #1082）で、不正な size の行はここでは扱わない。
        if not _valid_gate_size(r.get("size")):
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


def _valid_gate_size(v):
    """`size` フィールドを set 内包・`sorted()`・突合比較へ渡す前に検証する。

    producer 契約（`bench-common::parse_cli_from` の `--size`）上 `size` は
    常に正の整数だが、外部 JSONL の値は型・値域未検証のまま信頼できない
    （security.md「外部フォーマットのパース時検証（A03）」）。配列・
    オブジェクト等の unhashable な値は `{r["size"] for r in ...}` の集合化で
    `TypeError: unhashable type` を、文字列と整数の混在は `sorted()` の
    比較演算で `TypeError` を、それぞれ送出し集計全体を例外終了させうる
    （イシュー #1051 codex-review P0 指摘・PR #1082）。

    `bool` は Python では `int` のサブクラスのため、後続の `_is_plain_number`
    に処理を委ねる前にここで明示的に弾く（Bugbot Medium 指摘・PR #1082
    2 巡目: `True == 1` により `_pick_row_for_gate`/
    `_train_reuse_row_invalid_reason` の突合で `size: true` の行が
    `size=1` の行として誤って選ばれうる。`_is_plain_number` も内部で
    `isinstance(v, bool)` を弾いているため機能的には冗長だが、本関数の
    契約〈bool を size として許容しない〉を独立に読み取れるようにする）。
    """
    if isinstance(v, bool):
        return False
    return _is_plain_number(v) and not _non_integral(v) and int(v) > 0


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
    # イシュー #1042: `tf32` をキーへ含める。同一
    # `(framework, device, size, mode)` で FP32 行と TF32 opt-in 行が
    # 同一ファイルへ同居しうるため（`--tf32` は同じ size を使う想定。
    # `docs/cuda-tf32-optin-api-decision.md`）、`tf32` を含めないと
    # `mismatch_by_key`/`unverifiable_keys` 等の辞書で片方が上書きされ、
    # 無効判定（checksum 不一致・parity 失敗）の付記先を取り違える
    # （fail-open のおそれ）。
    return (r["framework"], r["device"], r["size"], r["mode"], r.get("tf32", False) is True)


# イシュー #1051: 目標達成ゲート（--target）専用のヘルパー群。既存の
# `section()`（Markdown 表生成）とは独立に、同じ無効判定規則（checksum
# 突合・要素単位検証・train reuse 突合）を再適用して「達成／未達／
# 判定不能」を機械的に判定する。`section()` の戻り値シグネチャ（6-tuple）
# を破壊しない実装方針（実装計画 §3）のため、無効判定ロジックは
# `gemm_checksum_mismatches`/`_parity_reason`/`_safe_time_s`/
# `_safe_finite_number`/`checksums_match` を再利用しつつ、train reuse の
# 判定のみ `_train_reuse_row_invalid_reason` として同等ロジックを本関数
# 群専用に複製する（`section()` 内の (b') ループは表示副作用〈stderr
# warning・Markdown 行〉を持つため直接共有すると呼び出しごとに warning が
# 重複出力される。ゲート判定は無音で理由文字列のみ返す）。


def _pick_row_for_gate(rows, fw, task, device, size):
    """ゲート判定に使う行を選ぶ: reuse 行があれば優先し、無ければ fresh。

    `size` は必須（唯一の呼び出し元 `target_gate` は `_valid_gate_size` で
    検証済みの整数を常に渡す）。`size=None`（size 条件を適用しない）を
    許容すると、複数 size を持つフレームワークで後述の重複検出が
    「異なる size の行が複数ある」を「重複キー」と誤検知するため、あえて
    既定値を持たせず必須引数にしている（codex-review 指摘・PR #1082）。

    戻り値は `(row, mode, dup_reason, used_tf32)`。該当行が無ければ
    `(None, None, None, False)`。gemm/train は fandhe-ai に reuse 行が
    存在しうる（イシュー #925/#957）。infer は reuse モード自体が
    存在しないため常に fresh へフォールバックする（`bench-fandhe` の
    `--mode` allowlist は gemm/train 用。モジュール docstring 参照）。

    `used_tf32`: 選ばれた行が TF32 フォールバック（後述）経由なら
    `True`。呼び出し元（`target_gate`）はこれを見て達成／未達判定に
    「TF32 強制計測」の注記を添える（精度条件が FP32 と異なることを
    黙って隠さないため）。

    同じ `(framework, task, device, size, mode)` に複数行が存在する場合は
    `dup_reason`（`None` 以外）を返し `row`/`mode` は `None` にする
    （codex-review P0 指摘・PR #1082: 旧実装は `get()` が返す最初に一致
    した行だけを採用し残りを検証しなかった。producer 側の重複バグ・
    JSONL の手組み改変で同一キーに正常行と壊れた行が混在した場合、
    出現順次第で遅い〈未達を示す〉fandhe-ai 行を握りつぶし「達成」判定を
    返してしまう fail-open があったため、重複自体を判定不能として扱う）。

    TF32 フォールバック（イシュー #1042 Cursor Bugbot 指摘・PR #1091・
    Medium）: gemm は `--tf32` opt-in（同一フレームワークが FP32/TF32
    両方の行を持ちうる）のため引き続き TF32 行を判定対象から除外する
    （既定 FP32 との混同・fail-open 防止）。一方 train/infer は burn の
    CUDA 実行が常時 TF32（FP32 厳密経路を持たない。
    `bench-burn/src/main.rs` の `tf32: cli.device == "cuda"`）で、TF32 行
    を除外し続けると成功した burn CUDA train/infer 計測が「未計測」判定
    に化ける（表示側の `_get_train_infer_row` と同じ根本原因）。
    train/infer では TF32 行を持つフレームワーク・デバイスにとって
    それが唯一の実測値であるため、FP32（`tf32=False`）行が 1 件も
    無い場合に限り TF32 行へフォールバックする。
    """
    tf32_candidates = (False,) if task == "gemm" else (False, True)
    for tf32_value in tf32_candidates:
        for mode in ("reuse", "fresh"):
            # `r.get("size")` を `_valid_gate_size` で検証してから比較する
            # （Bugbot Medium 指摘・PR #1082 2 巡目: 直接 `r["size"]` を
            # 読むと `size` キー欠損行で `KeyError`、`bool` 混入行で
            # `True == 1` により意図しない一致が起こりうる。不正な size
            # を持つ行は比較対象から除外する＝一致しないものとして扱う。
            # fail-closed）。
            matches = [
                r
                for r in rows
                if r["framework"] == fw
                and r["task"] == task
                and r["device"] == device
                and r["mode"] == mode
                and (r.get("tf32", False) is True) == tf32_value
                and _valid_gate_size(r.get("size"))
                and r.get("size") == size
            ]
            if len(matches) > 1:
                return (
                    None,
                    None,
                    (
                        f"重複キー: framework={fw}, task={task}, device={device}, "
                        f"size={size}, mode={mode}, tf32={tf32_value} の行が "
                        f"{len(matches)} 件あり一意に選べない"
                    ),
                    False,
                )
            if matches:
                return matches[0], mode, None, tf32_value
    return None, None, None, False


def _train_reuse_row_invalid_reason(rows, r):
    """train task・mode="reuse" 行の無効理由を返す（有効なら `None`）。

    `section()` の (b') ループと同一の判定規則（時間値の検証・同一
    フレームワーク内 fresh 行との checksum 突合）を、表示副作用なしで
    単一行に対して適用する。fresh 行が存在しない（比較対象なし）だけの
    場合は値そのものの正当性を否定しないため無効扱いにしない
    （`section()` の同節コメント参照）。

    fresh 側の検索は `r` と同一 `size` に限定する（Bugbot Medium 指摘・
    PR #1082: `size` を渡さず `get()` を呼ぶと `size=None`＝size 条件
    無視となり、複数 size のデータが混在する場合に別 size の fresh 行と
    誤って突合される。正常な複数 size データを判定不能に倒す、または
    壊れた reuse 行がたまたま別 size の先頭 fresh checksum と一致して
    無効判定をすり抜けるおそれがあったため、同一 size の fresh 行のみを
    比較対象にする）。

    同一 size の fresh 行が複数存在する場合は `_pick_row_for_gate` と
    同じ理由（advisor 指摘・PR #1082 3 巡目）で判定不能に倒す:
    `get()` は最初に一致した行だけを返し残りを検証しないため、複数の
    fresh 行のうち checksum が一致するものが先に現れると、他の
    不一致な fresh 行を握りつぶして「有効」判定してしまう fail-open が
    ありうる（`_pick_row_for_gate` の重複キー検出は reuse 側のみを見る
    ため、fresh 側 1 行を選ぶ本関数のこの箇所は独立に検証が必要）。
    """
    r_median = _safe_time_s(r.get("median_s"))
    r_q1 = _safe_time_s(r.get("q1_s"))
    r_q3 = _safe_time_s(r.get("q3_s"))
    r_init = _safe_time_s(r.get("init_s"))
    if r_median is None or r_q1 is None or r_q3 is None or r_init is None:
        return "時間値が不正な値"
    # `x.get("size")` を `_valid_gate_size` で検証してから比較する
    # （Bugbot Medium 指摘・PR #1082 2 巡目: 直接 `x["size"]` を読むと
    # `size` キー欠損行で `KeyError`、`bool` 混入行で `True == 1` に
    # より意図しない一致が起こりうる。不正な size を持つ行は比較対象から
    # 除外する＝一致しないものとして扱う。fail-closed。`r`（reuse 行）は
    # `_pick_row_for_gate` 経由で既に有効な size を持つことが保証されて
    # いるが、`r.get("size")` で取得し直接キーアクセスは避ける）。
    fresh_matches = [
        x
        for x in rows
        if x["framework"] == r["framework"]
        and x["task"] == "train"
        and x["device"] == r["device"]
        and x["mode"] == "fresh"
        and _valid_gate_size(x.get("size"))
        and x.get("size") == r.get("size")
    ]
    if len(fresh_matches) > 1:
        return f"同一 size の fresh 行が {len(fresh_matches)} 件あり突合先が一意に決まらない"
    if not fresh_matches:
        return None
    fresh = fresh_matches[0]
    fresh_median = _safe_time_s(fresh.get("median_s"))
    if fresh_median is None:
        return "fresh 側の時間値が不正な値"
    r_checksum = _safe_finite_number(r.get("checksum"))
    fresh_checksum = _safe_finite_number(fresh.get("checksum"))
    if r_checksum is None or fresh_checksum is None:
        return "checksum が不正な値"
    if not checksums_match(r_checksum, fresh_checksum):
        return "fresh と最終 loss 不一致"
    return None


def _gate_row_invalid_reason(rows, gemm_mismatch_map, row):
    """ゲート判定用の行無効理由（有効なら `None`）。

    task ごとに既存の無効判定規則を再適用する: gemm は checksum 不一致
    （`gemm_mismatch_map`。呼び出し側が `gemm_checksum_mismatches(rows)`
    から 1 回だけ構築し使い回す。行ごとに再計算すると size 数に対し
    O(n^2) になるため）と要素単位検証失敗（`_parity_reason`）、train の
    reuse 行は `_train_reuse_row_invalid_reason`。train の fresh 行・
    infer 行には（現状 `section()` 側にも）無効判定規則が無いため
    `None` を返す。
    """
    if row["task"] == "gemm":
        reasons = []
        if _row_key(row) in gemm_mismatch_map:
            reasons.append("checksum 不一致")
        preason = _parity_reason(row)
        if preason is not None:
            reasons.append(preason)
        return "; ".join(reasons) if reasons else None
    if row["task"] == "train" and row["mode"] == "reuse":
        return _train_reuse_row_invalid_reason(rows, row)
    return None


def _gate_devices(rows, target):
    """`target_gate` が列挙対象とするデバイス集合を返す。

    task 単位（`devices_in(rows, task, ...)`）ではなく **ファイル内の
    fandhe-ai/target 行全体**からデバイス集合を導出する。理由は 2 点
    （イシュー #1051 codex-review 指摘）:

    - P1: `devices_in` は framework を絞らないため、`--target candle`
      指定時に burn 専用デバイス（candle 側が存在しないデバイス）まで
      対象に入り、双方未計測でも「fandhe-ai 未計測」の判定不能レコード
      が生成されてしまう。framework を `("fandhe-ai", target)` に限定
      することで防ぐ。
    - P0: task 単位で集合を作ると、あるデバイスで特定 task（例: train）
      が fandhe-ai/target 双方とも 0 件（実行時失敗等で計測が丸ごと
      欠落）の場合、そのデバイスは当該 task の列挙対象から漏れ、
      `target_gate` がそのデバイス×task の組を一切生成しない
      （＝判定不能ではなく「そもそも存在しない」扱いになり、後段の
      `gate_records_all` 集計が「全達成」を誤って通す）。task をまたいで
      デバイス集合を 1 つに統一することで、あるデバイスが他 task で
      計測されている限りは当該デバイス×task の組が必ず列挙され、
      `fandhe_row`/`target_row` が `None` の場合の既存の「未計測」判定
      経路（本関数の呼び出し元 `target_gate` 内）で判定不能として
      捕捉される（run_all.sh/run_all_cuda.sh は 1 ファイル内で device
      ごとに gemm/train/infer を必ず揃って計測する構成のため、
      「そのデバイスは対象外」と「そのデバイスの当該 task だけ欠落」を
      混同しない）。
    """
    present = {
        r["device"]
        for r in rows
        if r.get("framework") in ("fandhe-ai", target) and r.get("device") in DEVICE_ORDER
    }
    return [d for d in DEVICE_ORDER if d in present]


def target_gate(rows, target):
    """fandhe-ai と `target`（candle/burn）の中央値を同一ファイル内で
    突合し、達成／未達／判定不能を判定する（イシュー #1051）。

    戻り値: dict のリスト。各要素のキーは
    `task`/`device`/`size`（fandhe-ai/target いずれの行にも当該
    task・device の実測が 1 件も無い場合のみ `None`。それ以外は
    task を問わず実データの size を列挙する。イシュー #1051
    codex-review 追加指摘: train/infer も gemm と同じ経路で size 集合を
    実測から都度導出するため、通常は現状の運用〈train/infer は単一
    size=64 のみ〉により単一値になるが、複数 size が存在する場合も
    取りこぼさず全て判定する）/`fandhe_mode`/
    `target_mode`/`fandhe_median`/`target_median`/`ratio`
    （= target_median / fandhe_median）/`status`
    （"achieved"|"unmet"|"undeterminable"）/`reason`（判定不能の理由。
    achieved/unmet では `None`）/`note`（旧形式 gemm 行への注記など）。
    """
    mismatches = gemm_checksum_mismatches(rows)
    gemm_mismatch_map = {_row_key(r): (ref, ref_label) for r, ref, ref_label in mismatches}

    records = []
    devices = _gate_devices(rows, target)
    for task in GATE_TASKS:
        for device in devices:
            # イシュー #1051 codex-review 追加指摘（P0）: 当初は
            # task == "gemm" のみサイズを列挙し、train/infer は
            # `sizes = [None]` 固定にしていた。`get()`/`_pick_row_for_gate`
            # は size=None を「size 条件を適用しない（最初に一致した行を
            # 返す）」ものとして扱うため、train/infer に複数 size の行が
            # 混在すると先頭の 1 行しか評価されず、他 size の未達・
            # target 側の未計測が黙って無視されて「全達成」側へ
            # fail-open してしまう。run_all.sh/run_all_cuda.sh は現状
            # train/infer を単一 size（64）でしか実行しないが、この
            # 判定不能検出（P0）と同じ理由で、実データが持つ size 集合を
            # 実測から都度導出し、gemm と同じ経路で 1 size ごとに突合
            # する（size が実際に 1 つしか無ければ従来と同じ挙動になり、
            # 複数あれば取りこぼさず全て判定する）。
            # イシュー #1042 codex-review P2 指摘（PR #1091）: gemm は
            # `_pick_row_for_gate` が tf32=False の行のみを候補にする
            # （`tf32_candidates = (False,) if task == "gemm" else ...`）
            # ため、`sizes`/`invalid_size_rows` の導出元である
            # `candidate_rows` に tf32=True の行を含めたままだと、
            # FP32 側に存在しない size を持つ TF32 専用行が混在した
            # 場合に `sizes` へその size が混入し、`_pick_row_for_gate`
            # は tf32=False で探すため両フレームワークとも
            # 「該当行なし」となって undeterminable が生成され、ゲートが
            # 誤って失敗しうる。gemm では tf32=True の行を候補集合から
            # 除外し、`_pick_row_for_gate` が実際に参照する母集団と
            # size 集合の導出元を一致させる（train/infer は
            # `_pick_row_for_gate` が tf32=True もフォールバック候補に
            # 含めるため除外しない）。
            candidate_rows = [
                r
                for r in rows
                if r["task"] == task
                and r["device"] == device
                and r["framework"] in ("fandhe-ai", target)
                and (task != "gemm" or r.get("tf32", False) is not True)
            ]
            # 外部 JSONL 由来の `size` を検証せず set 内包・`sorted()` へ
            # 渡すと、配列／オブジェクト混入で `unhashable type`、文字列と
            # 整数の混在で比較 `TypeError` となり集計全体が例外終了しうる
            # （codex-review P0 指摘・PR #1082）。producer 契約どおりの
            # 値（正の整数）のみを `sizes` に採用し、不正値を持つ行は
            # 例外にせず判定不能レコードへ倒す（security.md A03）。
            invalid_size_rows = [r for r in candidate_rows if not _valid_gate_size(r.get("size"))]
            sizes = sorted({int(r["size"]) for r in candidate_rows if _valid_gate_size(r.get("size"))})
            if invalid_size_rows:
                records.append(
                    {
                        "task": task,
                        "device": device,
                        "size": None,
                        "fandhe_mode": None,
                        "target_mode": None,
                        "fandhe_median": None,
                        "target_median": None,
                        "ratio": None,
                        "status": "undeterminable",
                        "reason": (
                            f"{task} の size が不正な値（正の整数以外）の行が "
                            f"{len(invalid_size_rows)} 件"
                        ),
                        "note": None,
                    }
                )
            if not sizes:
                # P0: このデバイスでは task が fandhe-ai/target
                # 双方とも 0 件（サイズが 1 つも存在しない）。
                # sizes が空のまま `for size in sizes` を素通りさせる
                # と、このデバイス×task の組が gate_records_all へ
                # 一切現れず「全達成」の誤判定に加担する。size=None
                # の判定不能レコードを明示的に積む。ただし不正 size 行が
                # 存在する場合は上記で既に判定不能レコードを積んでいる
                # ため、ここでの重複追加は避ける。
                if not invalid_size_rows:
                    records.append(
                        {
                            "task": task,
                            "device": device,
                            "size": None,
                            "fandhe_mode": None,
                            "target_mode": None,
                            "fandhe_median": None,
                            "target_median": None,
                            "ratio": None,
                            "status": "undeterminable",
                            "reason": f"{task} 未計測（fandhe-ai/target 双方 0 件）",
                            "note": None,
                        }
                    )
                continue
            for size in sizes:
                fandhe_row, fandhe_mode, fandhe_dup_reason, fandhe_used_tf32 = _pick_row_for_gate(
                    rows, "fandhe-ai", task, device, size
                )
                target_row, target_mode, target_dup_reason, target_used_tf32 = _pick_row_for_gate(
                    rows, target, task, device, size
                )
                record = {
                    "task": task,
                    "device": device,
                    "size": size,
                    "fandhe_mode": fandhe_mode,
                    "target_mode": target_mode,
                    "fandhe_median": None,
                    "target_median": None,
                    "ratio": None,
                    "status": "undeterminable",
                    "reason": None,
                    "note": None,
                }
                if fandhe_dup_reason is not None or target_dup_reason is not None:
                    parts = [r for r in (fandhe_dup_reason, target_dup_reason) if r is not None]
                    record["reason"] = "; ".join(parts)
                    records.append(record)
                    continue
                if fandhe_row is None:
                    record["reason"] = "fandhe-ai 未計測"
                    records.append(record)
                    continue
                if target_row is None:
                    record["reason"] = f"{target} 未計測"
                    records.append(record)
                    continue
                freason = _gate_row_invalid_reason(rows, gemm_mismatch_map, fandhe_row)
                treason = _gate_row_invalid_reason(rows, gemm_mismatch_map, target_row)
                if freason is not None or treason is not None:
                    parts = []
                    if freason is not None:
                        parts.append(f"fandhe-ai: {freason}")
                    if treason is not None:
                        parts.append(f"{target}: {treason}")
                    record["reason"] = "無効データ（" + "; ".join(parts) + "）"
                    records.append(record)
                    continue
                fandhe_median = _safe_time_s(fandhe_row.get("median_s"))
                target_median = _safe_time_s(target_row.get("median_s"))
                if fandhe_median is None or target_median is None:
                    record["reason"] = "時間値が不正"
                    records.append(record)
                    continue
                record["fandhe_median"] = fandhe_median
                record["target_median"] = target_median
                record["ratio"] = target_median / fandhe_median
                # 旧形式（要素単位検証キー欠損）gemm 行は判定は行うが、
                # 一度も要素単位検証を受けていない点を注記する（実装計画
                # §3「旧形式（parity 未検証）行」）。
                notes = []
                if (
                    fandhe_row["task"] == "gemm" and parity_status(fandhe_row) == "unverified"
                ) or (target_row["task"] == "gemm" and parity_status(target_row) == "unverified"):
                    notes.append("未検証（旧形式）")
                # イシュー #1042 Cursor Bugbot 指摘（PR #1091・Medium）:
                # train/infer の TF32 フォールバック（`_pick_row_for_gate`）で
                # 選ばれた行は burn CUDA 等の常時 TF32 実行であり FP32 と
                # 精度条件が異なる。時間比較そのものは有効だが、精度条件の
                # 違いを黙って隠さないよう注記する（判定を隠蔽しない・
                # security.md の fail-closed 方針と同じ透明性の趣旨）。
                tf32_parties = []
                if fandhe_used_tf32:
                    tf32_parties.append("fandhe-ai")
                if target_used_tf32:
                    tf32_parties.append(target)
                if tf32_parties:
                    notes.append(
                        "TF32 強制計測（" + "・".join(tf32_parties) + "。FP32 と精度条件が異なる）"
                    )
                if notes:
                    record["note"] = "; ".join(notes)
                if fandhe_median <= target_median:
                    record["status"] = "achieved"
                else:
                    record["status"] = "unmet"
                records.append(record)
    return records


def target_gate_section(rel, records, target):
    """target_gate() の結果を 1 ファイル分の Markdown 節へ整形する。

    `device`/`task`/`mode` は `DEVICE_ORDER`/`GATE_TASKS`/
    `_TRAIN_PHASES_MODES` allowlist 由来の値のみで構成される
    `records`（`target_gate` が生成）から来るため、JSONL の生文字列を
    未検証のまま Markdown へ流さない（既存 (b'') 節と同じ方針）。
    """
    lines = [f"### 集計対象: {rel}\n"]
    lines.append(
        f"| タスク | デバイス | N | fandhe-ai 中央値 | {target} 中央値 | "
        "比（target/fandhe） | 判定 | 備考 |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    unmet = []
    undeterminable = []
    for rec in records:
        device_label = DEVICE_LABEL.get(rec["device"], rec["device"])
        size_label = str(rec["size"]) if rec["size"] is not None else "-"
        key = f"{rec['task']}/{device_label}"
        if rec["size"] is not None:
            key += f"/N={rec['size']}"
        if rec["status"] == "undeterminable":
            fandhe_col = (
                f"{fmt_ms(rec['fandhe_median'])}（{rec['fandhe_mode']}）"
                if rec["fandhe_median"] is not None
                else "-"
            )
            target_col = (
                f"{fmt_ms(rec['target_median'])}（{rec['target_mode']}）"
                if rec["target_median"] is not None
                else "-"
            )
            ratio_col = "-"
            status_col = f"判定不能（{rec['reason']}）"
            undeterminable.append(f"{key}（{rec['reason']}）")
        else:
            fandhe_col = f"{fmt_ms(rec['fandhe_median'])}（{rec['fandhe_mode']}）"
            target_col = f"{fmt_ms(rec['target_median'])}（{rec['target_mode']}）"
            ratio_col = f"{rec['ratio']:.2f} 倍"
            if rec["status"] == "achieved":
                status_col = "達成"
            else:
                status_col = "**未達**"
                unmet.append(key)
        note_col = rec["note"] or ""
        lines.append(
            f"| {rec['task']} | {device_label} | {size_label} | {fandhe_col} | "
            f"{target_col} | {ratio_col} | {status_col} | {note_col} |"
        )
    lines.append("")
    lines.append("未達一覧:")
    lines.extend([f"- {u}" for u in unmet] if unmet else ["- なし"])
    lines.append("")
    lines.append("判定不能一覧:")
    lines.extend([f"- {u}" for u in undeterminable] if undeterminable else ["- なし"])
    lines.append("")
    return lines


# イシュー #1009: `bench-fandhe --task train --phases` が出力する
# `task:"train_phases"` 行の集計（(b'') 節）。既存の `get`/`devices_in`/
# gemm 系関数・(b')（`task == "train"` のみ読む）はこのタスク値を一切
# 読まないため、以下のヘルパーは他節から独立している。


def _train_phases_groups(rows):
    """`task == "train_phases"` 行を `(device, mode)` ごとにグループ化する
    （出現順を保持）。fandhe-ai 単独タスクのためフレームワーク横断の集約は
    行わない（bench-candle/bench-burn は `--phases` を実装しない。README
    「train --phases」節参照）。

    `device`/`mode` は外部 JSONL 由来のためグループ化キーへ使う前に型・
    allowlist 検証する（security.md A03。codex-review 指摘・PR #1055）。
    配列・オブジェクト等の unhashable な値がそのまま辞書キーになり
    `TypeError` で集計全体が例外終了するのを防ぐほか、想定外の device/mode
    文字列が (b'') 節に無検証で紛れ込むのも防ぐ。不正な行は集計対象から
    除外し `skipped` として返す（呼び出し元が警告を出す）。

    戻り値: (groups, skipped)
    groups: {(device, mode): [row, ...]}
    skipped: [row, ...]（device/mode が不正だった行）
    """
    groups = {}
    skipped = []
    for r in rows:
        if r.get("task") != "train_phases":
            continue
        device = r.get("device")
        mode = r.get("mode")
        if not (isinstance(device, str) and device in DEVICE_ORDER) or mode not in _TRAIN_PHASES_MODES:
            skipped.append(r)
            continue
        groups.setdefault((device, mode), []).append(r)
    return groups, skipped


def _train_phases_devices(groups):
    """`_train_phases_groups` が返した `groups` のキー `(device, mode)` を
    DEVICE_ORDER → mode（fresh → reuse）の順で返す。"""

    def rank(key):
        device, mode = key
        device_rank = DEVICE_ORDER.index(device) if device in DEVICE_ORDER else len(DEVICE_ORDER)
        mode_rank = 0 if mode == "fresh" else 1
        return (device_rank, mode_rank)

    return sorted(groups.keys(), key=rank)


def _train_phases_validate(group_rows, mode):
    """1 つの `(device, mode)` グループ内の `train_phases` 行を検証する。

    外部 JSONL 由来の `phase`/`phase_index`/時間値は使用前に検証する
    （security.md A03。他節と同じ `_safe_time_s`/`_is_plain_number` 方針）。

    - `phase` は producer 側（`bench_common::validate_phase_name`）と同じ
      `[a-z0-9_]+`（非空）allowlist、`phase_index` は非負整数でなければ
      ならない（不正・重複はその行を無効として個別に報告する）。非空文字列
      チェックのみでは改行・`|` 等を含む値が通過し Markdown 表へ無加工出力
      されうる（codex-review 指摘・PR #1055）。
    - `phase_index` の重複に加え、`phase` 名自体の重複（異なる
      `phase_index` に同じ `phase` 名を混入させる手組み JSON）も無効とする
      （分母となる `step_total` が複数存在する場合に最初の行だけを分母に
      使ってしまう不整合を防ぐ。同上指摘）。
    - `median_s`/`q1_s`/`q3_s`（reuse 行は `init_s` も）は `_safe_time_s`
      で検証する。
    - `step_total` phase が存在しない・複数存在する・`step_total` の
      `median_s` が不正な場合、本グループ全体を無効とする（比の分母が
      一意に決まらないため）。
    - 各 phase の中央値が `step_total` の中央値を上回る（計時区間の
      合計が全体を超過する不整合）場合もその行を無効とする。
    - 上記の個別行検証を通過しても、mode ごとの必須 phase 名の集合・
      `phase_index` 順序・件数（`_TRAIN_PHASES_REQUIRED_PHASES`。producer
      側 `bench-fandhe/src/main.rs` の `PHASE_*` 定数と同期）に完全一致
      しなければ本グループ全体を無効とする。`step_total` 行さえ残って
      いれば `backward` 等の必須 phase 行が欠落していても有効判定して
      しまっていた（codex-review 指摘・PR #1055）ことへの対処。

    戻り値: (entries, invalid, step_total_median, phase_set_reason)
    entries: [{"phase": str, "median": float|None, "q1": float|None,
               "q3": float|None, "reason": str|None}, ...]
             （有効な phase_index を持つ行は昇順、それ以外は末尾）
    invalid: bool（本グループに 1 件以上の無効行があるか）
    step_total_median: float|None（比の分母。一意に決まらなければ None）
    phase_set_reason: str|None（必須 phase 集合・順序・件数の不一致理由。
                       一致していれば None）
    """
    invalid = False
    keyed = {}
    unresolved = []
    for r in group_rows:
        phase = r.get("phase")
        phase_index = r.get("phase_index")
        phase_ok = _valid_phase_name(phase)
        index_ok = (
            _is_plain_number(phase_index)
            and not _non_integral(phase_index)
            and int(phase_index) >= 0
        )
        if not phase_ok or not index_ok:
            invalid = True
            unresolved.append((phase if phase_ok else "?", r, "phase/phase_index が不正な値"))
            continue
        pi = int(phase_index)
        if pi in keyed:
            invalid = True
            unresolved.append((phase, r, f"phase_index {pi} が重複"))
            continue
        keyed[pi] = (phase, r)

    name_to_pis = {}
    for pi, (phase, _r) in keyed.items():
        name_to_pis.setdefault(phase, []).append(pi)
    duplicate_names = {name for name, pis in name_to_pis.items() if len(pis) > 1}
    if duplicate_names:
        invalid = True

    step_total_pis = name_to_pis.get("step_total", [])
    if len(step_total_pis) != 1:
        invalid = True
        step_total_median = None
    else:
        step_total_median = _safe_time_s(keyed[step_total_pis[0]][1].get("median_s"))
        if step_total_median is None:
            invalid = True

    entries = []
    for pi in sorted(keyed):
        phase, r = keyed[pi]
        # `step_total`（比率の分母。ゼロ除算回避のため 0 秒を許容しない）
        # のみ従来の `_safe_time_s`（> 0）を使い、他の phase は sub-ns
        # 区間の 0 値を許容する `_safe_phase_time_s`（>= 0）を使う
        # （`_safe_phase_time_s` docstring 参照。イシュー #1010）。
        time_validator = _safe_time_s if phase == "step_total" else _safe_phase_time_s
        median = time_validator(r.get("median_s"))
        q1 = time_validator(r.get("q1_s"))
        q3 = time_validator(r.get("q3_s"))
        reason = None
        if median is None or q1 is None or q3 is None:
            reason = "時間値が不正な値"
        if reason is None and r.get("mode") == "reuse" and _safe_time_s(r.get("init_s")) is None:
            reason = "init_s が不正な値"
        if reason is None and phase in duplicate_names:
            reason = f"phase 名 '{phase}' が重複"
        if (
            reason is None
            and step_total_median is not None
            and median is not None
            and median > step_total_median * (1.0 + 1e-9)
        ):
            reason = "step_total 比が 100% を超過"
        if reason is not None:
            invalid = True
        entries.append({"phase": phase, "median": median, "q1": q1, "q3": q3, "reason": reason})
    for phase, r, reason in unresolved:
        entries.append({"phase": phase, "median": None, "q1": None, "q3": None, "reason": reason})

    # 必須 phase 名の集合・`phase_index` 順序・件数の突合（codex-review
    # 指摘・PR #1055）。`keyed` は個別行検証を通過した行のみを含むため、
    # 欠落・余剰・順序違いのいずれも `actual_order != required` として
    # 検出できる（同名 phase の重複は上記 `duplicate_names` 側で既に
    # invalid 化済みだが、`actual_order` にも重複が残るため二重に不一致
    # となり結果は変わらない）。
    phase_set_reason = None
    required = _TRAIN_PHASES_REQUIRED_PHASES.get(mode)
    if required is not None:
        actual_order = tuple(keyed[pi][0] for pi in sorted(keyed))
        if actual_order != required:
            invalid = True
            missing = [p for p in required if p not in name_to_pis]
            extra = [p for p in name_to_pis if p not in required]
            details = []
            if missing:
                details.append(f"欠落: {', '.join(missing)}")
            if extra:
                details.append(f"未知: {', '.join(extra)}")
            if not details:
                details.append("phase の並び順が想定と不一致")
            phase_set_reason = f"mode={mode!r} の必須 phase 集合と不一致（{'; '.join(details)}）"

    return entries, invalid, step_total_median, phase_set_reason


def _train_phases_section(rel, rows):
    """(b'') 節の Markdown 行を生成する。`train_phases` 行が無ければ
    `([], False)`（節自体を出力しない。既存 (a')/(b') と同じ方針）。
    """
    groups, skipped = _train_phases_groups(rows)
    if not groups and not skipped:
        return [], False

    lines = ["### (b'') MLP 学習 1 step のフェーズ分解（イシュー #1009）\n"]
    any_invalid = False
    for r in skipped:
        any_invalid = True
        print(
            f"warning: {rel}: train_phases 行の device={r.get('device')!r}/"
            f"mode={r.get('mode')!r} が不正な値 — 集計対象外",
            file=sys.stderr,
        )
    for device, mode in _train_phases_devices(groups):
        group_rows = groups[(device, mode)]
        entries, invalid, step_total_median, phase_set_reason = _train_phases_validate(group_rows, mode)
        any_invalid = any_invalid or invalid
        device_label = DEVICE_LABEL.get(device, device or "?")
        lines.append(f"#### {device_label} / {mode}\n")
        if invalid:
            for e in entries:
                if e["reason"] is not None:
                    print(
                        f"warning: {rel}: train_phases {device}/{mode}/{e['phase']} "
                        f"— {e['reason']} — 無効データとして表示",
                        file=sys.stderr,
                    )
            if step_total_median is None:
                print(
                    f"warning: {rel}: train_phases {device}/{mode} "
                    "の step_total 行が欠落または不正 — 比の算出不能",
                    file=sys.stderr,
                )
            if phase_set_reason is not None:
                print(
                    f"warning: {rel}: train_phases {device}/{mode} — {phase_set_reason}",
                    file=sys.stderr,
                )
        if mode == "reuse":
            init_val = next(
                (_safe_time_s(r.get("init_s")) for r in group_rows if r.get("init_s") is not None),
                None,
            )
            init_col = fmt_ms(init_val) if init_val is not None else "無効な値"
            lines.append(f"初期化(init_s): {init_col}\n")
        lines.append("| フェーズ | 中央値 | Q1 | Q3 | step_total 比 |")
        lines.append("| --- | --- | --- | --- | --- |")
        median_total = 0.0
        median_total_valid = bool(entries)
        for e in entries:
            phase_col = f"{e['phase']}（無効: {e['reason']}）" if e["reason"] else e["phase"]
            median_col = fmt_ms(e["median"]) if e["median"] is not None else "無効な値"
            q1_col = fmt_ms(e["q1"]) if e["q1"] is not None else "無効な値"
            q3_col = fmt_ms(e["q3"]) if e["q3"] is not None else "無効な値"
            if e["median"] is not None and step_total_median is not None:
                ratio_col = f"{e['median'] / step_total_median * 100:.1f}%"
            else:
                ratio_col = "-"
            lines.append(f"| {phase_col} | {median_col} | {q1_col} | {q3_col} | {ratio_col} |")
            if e["phase"] != "step_total":
                if e["median"] is None or e["reason"] is not None:
                    # 無効行（重複 phase 名等）は合計に含めない。含めると
                    # 例えば同名 phase の重複混入が「フェーズ合計」へ二重
                    # 計上され、無効データの影響が集計値へ紛れ込む。
                    median_total_valid = False
                else:
                    median_total += e["median"]
        if median_total_valid:
            lines.append(
                f"\n- フェーズ合計（中央値の和。参考値: 中央値は加法的でないため "
                f"step_total と一致しない場合がある）: {fmt_ms(median_total)}"
            )
        lines.append("")
    return lines, any_invalid


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
        # `_valid_gate_size` で不正な size（配列・文字列混在等）を除外して
        # から集合化・`sorted()` する（イシュー #1051 codex-review 指摘の
        # 防御的スイープ・PR #1082。target_gate 側の同一パターンと同じ
        # 理由: 外部 JSONL 由来の size を未検証のまま渡すと
        # `unhashable type`/比較 `TypeError` で `main()` 全体が
        # traceback 停止しうる）。`r.get("tf32", False) is not True` で TF32
        # opt-in 行を除外する（Cursor Bugbot Low 指摘・PR #1091: 含めると
        # `get()` が既定 tf32=False で見つけられない size が「計測不可」
        # と誤表示される。TF32 専用行は (a-tf32) 節が表示する）。
        sizes = sorted(
            {
                r["size"]
                for r in rows
                if r["task"] == "gemm"
                and r["device"] == device
                and r.get("tf32", False) is not True
                and _valid_gate_size(r.get("size"))
            }
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
            # (a) 節と同じ防御（`_valid_gate_size`。イシュー #1051
            # codex-review 指摘の防御的スイープ・PR #1082）。
            # (a) 節と同じ理由で TF32 opt-in 行を除外する（Cursor Bugbot
            # Low 指摘・PR #1091）。
            sizes = sorted(
                {
                    r["size"]
                    for r in rows
                    if r["task"] == "gemm"
                    and r["device"] == device
                    and r["mode"] == "reuse"
                    and r.get("tf32", False) is not True
                    and _valid_gate_size(r.get("size"))
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

    # (a-tf32) TF32 opt-in GEMM（イシュー #1042）。`--tf32` で計測された行が
    # 存在するファイルにのみ出力する。目標達成ゲート（(a) 節・
    # `target_gate`）からは既定で除外済み（`get()`/`_pick_row_for_gate`/
    # `gemm_checksum_reference` の `tf32` 除外フィルタ参照）のため、ここで
    # 独立に表示する。checksum 相互突合は（a) 節と異なり FP32 行を巻き込ま
    # ないため行わず、行自身の要素単位検証（`parity_status`）のみを「無効」
    # 判定に使う。
    tf32_rows = [r for r in rows if r["task"] == "gemm" and r.get("tf32", False) is True]
    if tf32_rows:
        for r in tf32_rows:
            preason = _parity_reason(r)
            if preason is not None:
                print(
                    f"warning: {rel}: {r['framework']}/{r['device']}/size={r['size']}/{r['mode']} "
                    f"(tf32) の gemm 要素単位検証が閾値超過 — {preason} — 無効データとして表示",
                    file=sys.stderr,
                )
        lines.append(
            "### (a-tf32) GEMM TF32（--tf32 opt-in。REQ-2 統一複合判定。CUDA Tensor Core reduced precision）\n"
        )
        tf32_devices = sorted(
            {r["device"] for r in tf32_rows if r["device"] in DEVICE_ORDER},
            key=DEVICE_ORDER.index,
        )
        for device in tf32_devices:
            sizes = sorted(
                {
                    r["size"]
                    for r in tf32_rows
                    if r["device"] == device and _valid_gate_size(r.get("size"))
                }
            )
            lines.append(f"#### {DEVICE_LABEL[device]}\n")
            lines.append("| N | フレームワーク | 中央値 | Q1 | Q3 | GFLOP/s |")
            lines.append("| --- | --- | --- | --- | --- | --- |")
            for n in sizes:
                for fw in FRAMEWORKS:
                    r = get(rows, fw, "gemm", device, n, tf32=True)
                    if r:
                        preason = _parity_reason(r)
                        if preason is not None:
                            fw_col = f"{fw}（無効: {preason}）"
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

    lines.append(
        "### (b) MLP 学習（784→256→10、ReLU、バッチ 64、MSE、SGD lr=0.01、1 ステップあたり時間）\n"
    )
    lines.append("| デバイス | フレームワーク | 中央値 | Q1 | Q3 |")
    lines.append("| --- | --- | --- | --- | --- |")
    for device in _devices_in_train_infer(rows, "train"):
        for fw in FRAMEWORKS:
            r, is_tf32 = _get_train_infer_row(rows, fw, "train", device)
            if r:
                # イシュー #1042 Cursor Bugbot 指摘（PR #1091・Medium）: burn
                # CUDA のように TF32 フォールバック行しか無い場合、FP32 と
                # 精度条件が異なることを黙って隠さないようフレームワーク列に
                # 注記する（(a-tf32) 節と同じ透明性の趣旨）。
                fw_col = f"{fw}（TF32）" if is_tf32 else fw
                lines.append(
                    f"| {device} | {fw_col} | {fmt_ms(r['median_s'])} | {fmt_ms(r['q1_s'])} | {fmt_ms(r['q3_s'])} |"
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
                # `_safe_time_s(None)` は `_is_plain_number(None)` が False
                # を返すため素通しで None になる（`r.get("init_s") is not
                # None` による事前分岐は不要）。
                r_init = _safe_time_s(r.get("init_s"))
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
                # reuse 行の init_s は本節（(b')）が計測する初期化コストの
                # 主対象であり必須フィールドのため、欠損・不正値も無効判定
                # に含める（イシュー #959 codex-review 2 巡目 P0 指摘:
                # 旧実装は init_s を表示列にのみ反映し `row_invalid` へ
                # 含めていなかったため、init_s が不正でも `--strict` が
                # exit 0 のまま fail-open していた）。
                # fresh 側 median_s（fresh/reuse 比の算出・突合に使う値）が
                # fresh 行自体は存在するのに不正（NaN・負値等）な場合も、
                # 比較に使う値そのものが信頼できないため無効判定に含める
                # （fresh 行が存在しない「突合不能」ケースとは区別する。
                # 同 P0 指摘）。
                row_invalid = (
                    r_median is None
                    or r_q1 is None
                    or r_q3 is None
                    or r_init is None
                    or (fresh is not None and fresh_median is None)
                )

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

    # (b'') MLP 学習 1 step のフェーズ分解（イシュー #1009。`bench-fandhe`
    # `--task train --phases` が出力する `task:"train_phases"` 行）。
    # `(b)`/`(b')` とは独立の節で、(b')/(b) は `task == "train"` のみを
    # 読むため本節の追加による既存表への影響はない（モジュール docstring
    # 参照）。
    train_phases_lines, has_train_phases_invalid = _train_phases_section(rel, rows)
    lines.extend(train_phases_lines)

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

    for device in _devices_in_train_infer(rows, "infer"):
        for fw in FRAMEWORKS:
            r, is_tf32 = _get_train_infer_row(rows, fw, "infer", device)
            if r:
                # (b) 節と同じ理由で TF32 フォールバック行を注記する
                # （イシュー #1042 Cursor Bugbot 指摘・PR #1091・Medium）。
                fw_col = f"{fw}（TF32）" if is_tf32 else fw
                lines.append(
                    f"| {device} | {fw_col} | {fmt_ms(r['median_s'])} | {fmt_ms(r['q1_s'])} | {fmt_ms(r['q3_s'])} | {fmt_tps(r['throughput_per_s'])} |"
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
    # `_row_key(r)` は `r["size"]` を含むタプルのため、size が不正
    # （配列等の unhashable 値）だとタプル自体が unhashable になり
    # `not in unverifiable_keys` のハッシュ計算で `TypeError` を送出する
    # （イシュー #1051 codex-review 指摘の防御的スイープ・PR #1082）。
    # `gemm_checksum_unverifiable`/`gemm_checksum_mismatches` 側で既に
    # 不正 size の行を除外しているのと同じ理由で、ここでも
    # `_valid_gate_size` で事前に弾く（不正 size の行は「検証済み」にも
    # 「突合不能」にも数えない）。
    verified_total = sum(
        1
        for r in rows
        if r["task"] == "gemm"
        # イシュー #1042: TF32 行は checksum 相互突合の対象から除外済み
        # （`gemm_checksum_unverifiable`/`gemm_checksum_reference` 参照）
        # のため `unverifiable_keys` にも決して現れない。ここで明示的に
        # 除かないと「相互突合できた」件数へ誤って計上されてしまう
        # （fail-open のおそれ）。
        and r.get("tf32", False) is not True
        and _valid_gate_size(r.get("size"))
        and _row_key(r) not in unverifiable_keys
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
    return (
        lines,
        bool(mismatches),
        bool(parity_failures),
        bool(unverified_rows),
        has_train_reuse_invalid,
        has_train_phases_invalid,
    )


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
            "（イシュー #959）・train_phases (b'') の phase/phase_index 不正・"
            "step_total 欠落／時間値不正（イシュー #1009）が1 件以上あれば"
            "終了コード 2 を返す（既定は 0 のまま警告のみ）"
        ),
    )
    parser.add_argument(
        "--target",
        choices=GATE_TARGET_CHOICES,
        default=None,
        help=(
            "指定フレームワーク（candle/burn）と fandhe-ai の GEMM/学習/推論"
            "中央値を同一ファイル内で突合し、目標達成ゲート節を出力する"
            "（イシュー #1051）。未達または判定不能が 1 件以上あれば"
            "終了コード 3 を返す（`--strict` の無効データ判定〈終了コード 2〉"
            "と両方該当する場合はデータ無効の解消を優先し 2 を返す）"
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
    any_train_phases_invalid = False
    gate_section_lines_by_file = []
    gate_records_all = []
    for path in inputs:
        rows = load_rows(path)
        if not rows:
            lines.append(f"## 集計対象: {os.path.relpath(path, HERE)}\n")
            lines.append("（有効な行なし）\n")
            if args.target:
                # P0 修正（codex-review 指摘・PR #1082 2 巡目）: 空ファイルを
                # 無条件に `continue` で読み飛ばすと、`--target` 指定時に
                # 複数入力のうち 1 ファイルが空でも他ファイルが全達成なら
                # `gate_records_all` が非空のまま空ファイル分の判定不能が
                # 一切計上されず exit 0 になる fail-open があった
                # （計測が丸ごと欠落したファイルを「対象外」と黙って扱う）。
                # ファイル単位の判定不能レコードを 1 件積み、後段の
                # `gate_achieved`/`gate_unmet`/`gate_undeterminable` 集計・
                # exit code 判定へ確実に反映させる。
                rel = os.path.relpath(path, HERE)
                empty_record = {
                    "task": "-",
                    "device": "-",
                    "size": None,
                    "fandhe_mode": None,
                    "target_mode": None,
                    "fandhe_median": None,
                    "target_median": None,
                    "ratio": None,
                    "status": "undeterminable",
                    "reason": "入力ファイルに有効な行が無い",
                    "note": None,
                }
                gate_records = [empty_record]
                # codex P0・Bugbot High 指摘（PR #1082 4 巡目）: skip 由来の
                # 失敗はこのファイル（環境）自身に対応する skipped*.log
                # のみを対象にする（`_inject_skip_failures_into_gate`
                # docstring「呼び出し契約」参照。全ファイル横断の
                # `gate_records_all` へ直接注入すると環境が混同される）。
                skip_paths = _skip_log_paths_for_input(path)
                skip_failures = _skip_failures_for_paths(skip_paths)
                _inject_skip_failures_into_gate(gate_records, skip_failures, args.target, rows)
                gate_section_lines_by_file.append(
                    target_gate_section(rel, gate_records, args.target)
                )
                gate_records_all.extend(gate_records)
            continue
        (
            section_lines,
            has_mismatch,
            has_parity_failure,
            has_unverified,
            has_train_reuse_invalid,
            has_train_phases_invalid,
        ) = section(path, rows)
        lines.extend(section_lines)
        any_checksum_mismatch = any_checksum_mismatch or has_mismatch
        any_parity_failure = any_parity_failure or has_parity_failure
        any_parity_unverified = any_parity_unverified or has_unverified
        any_train_reuse_invalid = any_train_reuse_invalid or has_train_reuse_invalid
        any_train_phases_invalid = any_train_phases_invalid or has_train_phases_invalid

        # イシュー #1051: 目標達成ゲート。--target 指定時のみ計算する
        # （既存の呼び出し元は --target を渡さないため非破壊。モジュール
        # docstring・実装計画 §3 参照）。
        if args.target:
            rel = os.path.relpath(path, HERE)
            gate_records = target_gate(rows, args.target)
            # codex P0・Bugbot High 指摘（PR #1082 4 巡目）: skip 由来の
            # 失敗はこのファイル（環境）自身に対応する skipped*.log の
            # みを対象にする（`_inject_skip_failures_into_gate` docstring
            # 「呼び出し契約」参照）。以前は全ファイル横断で集約した
            # `gate_records_all` を渡していたため、環境 A の JSONL に
            # 達成行がある組について、環境 B（別ファイル）の同じ組の
            # skipped*.log 失敗が `existing_keys` に紛れて注入されず
            # 「全達成」に混入する fail-open があった。
            skip_paths = _skip_log_paths_for_input(path)
            skip_failures = _skip_failures_for_paths(skip_paths)
            _inject_skip_failures_into_gate(gate_records, skip_failures, args.target, rows)
            gate_section_lines_by_file.append(target_gate_section(rel, gate_records, args.target))
            gate_records_all.extend(gate_records)

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
                lines.append(
                    f"- **{os.path.basename(sl)}**: "
                    f"{_sanitize_skip_raw_for_display(line)}"
                )
    if not any_skip:
        lines.append("- なし（skipped*.log は空または不在）")
    lines.append("")
    # 目標達成ゲートへの skip 失敗の組み込みは、上のファイルごとのループ
    # 内（`_skip_log_paths_for_input`/`_inject_skip_failures_into_gate`）
    # で環境スコープを保ったまま既に完了している（codex P0・Bugbot High
    # 指摘・PR #1082 4 巡目）。ここでの表示用ループはあくまで人間向けの
    # 生ログ一覧であり、ゲート判定には使わない。ただし `line` 自体は
    # ベンチバイナリの stderr を含む未信頼文字列のため、ゲート節と同じ
    # `_sanitize_skip_raw_for_display` で Markdown/HTML エスケープしてから
    # 埋め込む（イシュー #1085・security.md A03。ファイル名部分
    # `os.path.basename(sl)` はローカル glob 一致でありユーザー入力由来
    # ではないため対象外）。

    gate_unmet = 0
    gate_undeterminable = 0
    if args.target:
        lines.append(
            f"## 目標達成ゲート（--target {args.target}。イシュー #1051）\n"
        )
        for gate_section_lines in gate_section_lines_by_file:
            lines.extend(gate_section_lines)
        gate_records_empty = not gate_records_all
        if gate_records_empty:
            # P0（イシュー #1051 codex-review 指摘）: 入力 JSONL が全て
            # 「有効な行なし」だった、または全ファイルで fandhe-ai/target
            # いずれの行も存在しなかった等の理由で gate_records_all が
            # 1 件も生成されない場合、達成 0・未達 0・判定不能 0 のまま
            # 下の分岐を素通りし exit 0（「全達成」の誤判定）になって
            # しまう。計測対象が丸ごと欠落した入力を「全達成」として通す
            # fail-open を避けるため、判定不能 1 件として扱い後段の
            # 非ゼロ終了判定（`gate_unmet > 0 or gate_undeterminable > 0`）
            # に確実に載せる（`--strict` との優先順位はこの後の分岐に
            # そのまま委ねる。実装計画 §3 の優先順位を変えない）。
            lines.append(
                "**全体集計**: 目標達成ゲートの対象データが 0 件のため判定不能"
                "（入力 JSONL に有効な行がない、または fandhe-ai/target "
                "いずれの行も存在しない）\n"
            )
        gate_achieved = sum(1 for r in gate_records_all if r["status"] == "achieved")
        gate_unmet = sum(1 for r in gate_records_all if r["status"] == "unmet")
        gate_undeterminable = sum(
            1 for r in gate_records_all if r["status"] == "undeterminable"
        )
        if gate_records_empty:
            gate_undeterminable = 1
        else:
            lines.append(
                f"**全体集計**: 達成 {gate_achieved} / 未達 {gate_unmet} / "
                f"判定不能 {gate_undeterminable}\n"
            )

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
        or any_train_phases_invalid
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
        if any_train_phases_invalid:
            print(
                "error: --strict: 1 件以上の train_phases 行が無効"
                "（phase/phase_index の不正・重複・step_total 欠落・"
                "時間値の不正。イシュー #1009）",
                file=sys.stderr,
            )
        # イシュー #1051 実装計画 §3: --strict の無効データ判定（終了コード
        # 2）と目標達成ゲートの未達／判定不能（終了コード 3）が両方該当
        # する場合は、壊れたデータ上の達成判定は信用できないためデータ
        # 無効の解消を優先させる（2 を返す。ゲート結果自体は上で Markdown
        # へ出力済みのため情報は失われない）。
        return 2

    if args.target and (gate_unmet > 0 or gate_undeterminable > 0):
        print(
            f"error: --target {args.target}: 未達 {gate_unmet} 件 / "
            f"判定不能 {gate_undeterminable} 件",
            file=sys.stderr,
        )
        for r in gate_records_all:
            if r["status"] not in ("unmet", "undeterminable"):
                continue
            label = f"{r['task']}/{r['device']}"
            if r["size"] is not None:
                label += f"/N={r['size']}"
            if r["status"] == "unmet":
                print(f"  unmet: {label}", file=sys.stderr)
            else:
                print(f"  undeterminable: {label}（{r['reason']}）", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
