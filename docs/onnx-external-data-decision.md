# ONNX external data（外部 `.onnx.data` ファイル）の initializer 読み込み設計判断（イシュー #2347）

## 1. 背景・目的

PyTorch の既定 exporter（`torch.onnx.export(..., dynamo=True)`）は、
テンソルサイズに応じて initializer を external data
（`TensorProto.data_location = EXTERNAL` + companion `.onnx.data` ファイル
への `location`/`offset`/`length` 参照）として出力する。#2329 の実測では
432 バイトの `conv.weight` も external になり、閾値はごく小さい。

`crates/onnx-interop` の `TensorProto`（`src/onnx/proto.rs`）はこれまで
`data_location`（tag=14）・`external_data`（tag=13）を宣言していなかった
ため、prost は両フィールドを無言スキップし、`graph::decode_tensor` は
data フィールドが空のまま「raw_data・float_data とも空」分岐へ進み
`GraphError::RawDataByteLenMismatch`（期待バイト数 > 0・実バイト数 0）で
拒否していた。その結果 PyTorch の既定出力をそのまま import できず、
#2329（PR #2343）の fixture 生成では external data を本体へ再 inline 化
することで回避していた（`docs/perf/logs/onnx-cnn-ops-pytorch-fixture-2329/
README.md` R5 節）。

本イシューの目的は、外部ファイルを解決する基点ディレクトリを受け取る
import 入口を `onnx-interop` 内部に新設し、外部参照を fail-closed に
検証してから読み込むことである。**バイト列入力の既存入口の挙動は一切
変えない**（不変条件。3 節参照）。

## 2. API 形状（`onnx-interop` 内部限定。facade `OnnxModel::from_path` から利用）

新規モジュール `crates/onnx-interop/src/onnx/external_data.rs`（`onnx::mod`
から `pub mod external_data;`）:

- `pub struct ExternalDataOptions { pub max_total_bytes: u64, pub max_external_files: usize }`
  - `Default` の既定値は `DEFAULT_MAX_EXTERNAL_DATA_TOTAL_BYTES = 64 GiB`
    （2026-09-28 ユーザー承認。当初の 4 GiB 暫定値から改定済み。変更は
    定数 1 行の書き換えで済む。6 節参照）。
  - `max_external_files` の既定値は `DEFAULT_MAX_EXTERNAL_FILES = 4096`
    （2026-09-28 ユーザー承認。当初の暫定値 4096 を正式な既定値として確定）。distinct な external
    data ファイル実体（[`FileKey`] で畳み込んだ後の数）の上限で、
    `max_total_bytes` がバイト数のみを制限する隙間（サイズ 0 のテンソルを
    大量の異なるファイルへ分散させるとファイルハンドルだけが増え fd 上限
    に達しうる。A04 資源枯渇対策）を塞ぐ。超過は `ExternalDataError::
    TooManyExternalFiles` で拒否する（#2347 P0 是正・PR #2348 コード
    レビュー対応・PRRT_kwDOTuUCJc6mlxhy）。
- `pub fn resolve_external_data(model: &mut ModelProto, base_dir: &Path, options: &ExternalDataOptions) -> Result<(), GraphError>`
  — in-place で external なテンソルを `raw_data` へ inline 化する。
- `pub fn build_graph_with_external_data(model: &ModelProto, base_dir: &Path, options: &ExternalDataOptions) -> Result<Graph, GraphError>`
  — `model.clone()` → `resolve_external_data` → 変更していない
    `graph::build_graph` の順に呼ぶ新しい import 入口。
- `GraphError` に variant `ExternalData(ExternalDataError)` を 1 つだけ
  追加した（`crates/onnx-interop/src/onnx/graph.rs`）。
- `ExternalDataError`（`#[non_exhaustive]`・`Debug`/`Clone`/`PartialEq`/`Eq`）
  は `InvalidDataLocation`／`InconsistentDataFields`／`MissingLocationKey`／
  `DuplicateKey`／`UnknownKey`／`ChecksumUnsupported`／`InvalidLocation`
  （`LocationRejectReason` 付き）／`InvalidNumber`／`RangeOutOfFile`／
  `LengthMismatch`／`OverlappingRegion`／`TotalSizeLimitExceeded`／
  `TooManyExternalFiles`／`Io`／`FileChangedDuringLoad`／`InvalidBaseDir`／
  `DuplicateInitializerName`／`UnsupportedPlatformForSecureResolve`／
  `Internal` の variant を持つ。診断文字列（`tensor_name`／`key`）は
  `proto::cap_sparse_tensor_diag_name`（256 バイト上限）を再利用して
  切り詰める。ホストの絶対パス・canonicalize 後のパスはいずれの
  variant にも含めない（security.md A05）。

`element_count`（`graph::element_count`。dims の非負性・`checked_mul`）は
既存実装をそのまま再利用する（検証ロジックを二重実装しない）。

## 3. proto の変更・不変条件（A6）

`TensorProto` に `external_data: Vec<StringStringEntryProto>`（tag=13）・
`data_location: i32`（tag=14）を追加した。`StringStringEntryProto{ key,
value }` は新規メッセージ。フィールド番号の出典は `proto.rs` 冒頭コメント
と同じ `onnx==1.23.0` 同梱の `onnx/onnx.proto`。`data_location` 定数
モジュール（`DEFAULT = 0`・`EXTERNAL = 1`）も追加した。

**不変条件**: [`graph::decode_tensor`] は本モジュール追加後も 1 バイトも
変更していない。`onnx::proto::decode_model` → `graph::build_graph` の
バイト列入口は `data_location`／`external_data` を一切参照しないため、
external data を持つモデルをバイト列入口へ渡した場合の挙動（従来どおり
`RawDataByteLenMismatch` で拒否）は本モジュール導入前後で変わらない。
これは以下で回帰テスト化した:

- `crates/onnx-interop/tests/onnx_external_data.rs::
  bytes_entry_point_still_rejects_external_data_model`
- `crates/onnx-interop/tests/onnx_interp_pytorch_cnn_fixture.rs::
  external_data_fixture_bytes_entry_point_still_rejects`（実 fixture 版）
- `crates/facade/tests/interop_onnx_internal_parity.rs::
  facade_from_bytes_rejects_external_data_model_from_path_attempts_resolution`（facade 経由）

既存の export バイト一致・roundtrip テスト（`tests/onnx_export_roundtrip.rs`・
`crates/facade/tests/interop_onnx_internal_parity.rs`）もすべて無変更で
pass することを確認済み（prost は既定値のスカラーと空の repeated を
出力しないため、`encode_model` の出力バイト列は変わらない）。

`onnx-interop` の 36 箇所の `TensorProto` リテラルのうち、フィールドを
すべて列挙している箇所は `external_data: Vec::new()`／`data_location: 0`
を明示追加し、`..Default::default()` を使っている箇所は無変更（コンパイル
時に自動で新フィールドが既定値になる）。

## 4. 2 パス設計（検証と読み込みの分離。security.md A03／A04）

1. **パス 1（`plan`）**: ファイル内容を一切読まず、以下をすべて検証する。
   1 件でも失敗すれば `Err` を返しファイルは一切読まない（A04 資源枯渇
   対策）。
   - `data_location` は `DEFAULT`（0）／`EXTERNAL`（1）のみ許す。
   - `EXTERNAL` なのに `raw_data`/`float_data`/`int64_data` が非空、
     または `DEFAULT` なのに `external_data` が非空なら拒否する
     （`InconsistentDataFields`）。
   - `external_data` の許容キーは `location`（必須）・`offset`・
     `length` のみ。同一キーの重複・未知キーは拒否する。**`checksum`
     キーは非対応のため fail-closed に拒否する**（依存を追加できない
     ため SHA-1 検証は実装しない。no-silent-skip 契約。security.md A08）。
   - `offset`／`length` は空でない ASCII 数字列のみ許す（符号・空白・
     先頭 `+`・非数字はすべて拒否。checked 変換で `u64` の範囲外・
     `offset + length` のオーバーフローも拒否）。`length` 省略時は
     ONNX 仕様どおり EOF まで（`file_len - offset`。checked）と解釈する。
   - `location` は文字列段階（空・NUL・4096 バイト超過・`\`・ドライブ
     文字接頭辞）と `Path::components()`（`RootDir`/`Prefix`/`ParentDir`
     をすべて拒否）の両方で検証する。**unix 全般**（Linux／macOS／その他
     unix）では `base_dir` を 1 度だけディレクトリ fd として開き
     （`no_follow_open::open_base_dir`。全テンソル分を通して再利用する）、
     検証済みの `location` をその fd 起点で解決する
     （`crates/onnx-interop/src/onnx/external_data.rs::no_follow_open`。
     2026-09-28・イシュー #2347 是正版）。**Linux** では `openat2(2)`
     （`RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS | RESOLVE_NO_MAGICLINKS`。
     `open_chain_openat2`）で経路解決全体を 1 回のシステムコールとして
     カーネルへアトミックに封じ込めさせる。`openat2` が未対応
     （`ENOSYS`／`EPERM`。古いカーネル・seccomp 等）の場合のみ、成分
     ごと逐次 `openat(dirfd, name, O_NOFOLLOW)` で辿る方式
     （`open_chain_component_walk`）へフォールバックする。**macOS・その他
     unix**（`openat2` 非対応）では常にこの逐次方式を使う。いずれの方式
     も経路文字列を`canonicalize`／`File::open`／`symlink_metadata`で
     **再解決しない**（エラー分類用の診断も、既に開いた親ディレクトリ fd
     を起点にした単一コンポーネントの `fstatat(AT_SYMLINK_NOFOLLOW)` の
     みを使う）ため、検証と実際のオープン対象が fd レベルで一致すること
     が構造的に保証される。**非 unix**（Windows 等）は `openat`／
     `openat2` 相当の安全な経路解決手段を持たないため、`resolve_and_open`
     は常に `UnsupportedPlatformForSecureResolve` で拒否する
     （fail-closed。5 節参照）。
   - ファイルを開いてサイズを取り、`offset + length` がファイル長を
     超えないこと・`length` が dims/data_type から導出した期待バイト長
     （`element_count` × 要素サイズ。FLOAT=4／INT64=8／BOOL=1／
     FLOAT16=2）と一致することを検証する。
   - 同一ファイル内の読み込み区間の重複（同一区間の二重参照を含む）を
     拒否する。initializer 名の重複は I/O の前に拒否する。
   - 全テンソルの `length` を `checked_add` で積み上げ、
     `ExternalDataOptions::max_total_bytes` を超えた時点（または
     overflow した時点）で `TotalSizeLimitExceeded` とする。確保より
     前に検査するため、巨大な `length` でメモリを確保することはない。
   - distinct な external data ファイル実体（`FileKey`。dev/ino ベース）
     の数を `ExternalDataOptions::max_external_files` と比較し、`files`
     マップへ登録する前（＝ファイルを開いた直後）に超過を検査する。
     `max_total_bytes` はバイト数のみを制限するため、サイズ 0 の
     テンソルを大量の異なる空ファイルへ分散させる入力は合計サイズを
     常に 0 に保ったままファイルハンドルだけを増やしプロセスの fd
     上限に達しうる。この検査で `TooManyExternalFiles` として拒否する
     （2026-09-28・#2347 P0 是正・PR #2348 コードレビュー対応・
     PRRT_kwDOTuUCJc6mlxhy）。
2. **パス 2（`load`）**: パス 1 が全件成功した場合のみ、パス 1 で開いた
   ファイルハンドルを再利用して該当区間だけを `read_exact` する
   （`.data` ファイル全体は読まない）。読み込み直前に `metadata().len()`
   （Unix では dev/ino も）をパス 1 の記録と再照合し、不一致は
   `FileChangedDuringLoad` とする（TOCTOU の窓を縮める）。

読み込んだバイト列は `raw_data` へ書き戻し、`data_location = DEFAULT`・
`external_data` は空にする（パス 2 完了後にのみ書き戻す。検証・読み込み
が全件成功した場合限定）。

対象は `GraphProto.initializer` に加え、各 `NodeProto.attribute[].t`
（Constant の `value` 等）も含む（`enumerate_tensors`）。サブグラフ属性
（`g`/`graphs`）は `proto::AttributeProto` に未定義のため対象外。

## 5. 残るリスク（受容済み）

- **2026-09-28 更新（`libc` 導入・`openat2` 採用による是正。PR #2348 codex
  レビュー discussion_r4119392011・PRRT_kwDOTuUCJc6mkUsI・
  PRRT_kwDOTuUCJc6mk30J の P0 是正）**: 当初は「std に `O_NOFOLLOW`
  相当が無く、`libc` crate も deps-policy.md の許容依存区分に含まれない
  ため追加できない」ことを理由に、`symlink_metadata` 検証後
  `canonicalize` → `File::open` と経路文字列を再解決する実装、続いて
  手書き `extern "C"` 宣言＋手書き `openat` フラグ定数（`O_DIRECTORY`等）
  による実装を採用していた。後者は Linux では同じ定数でも CPU
  アーキテクチャごとに値が異なり（x86 は `O_DIRECTORY=0o200000`・
  aarch64 は `O_DIRECTORY=0o40000`）、x86 向けの値のまま aarch64 Linux
  （DGX Spark GB10）でビルドするとシンボリックリンク拒否が機能しない
  実装バグを生んでいた。また旧実装は診断用の分類に累積パス文字列への
  `symlink_metadata` 呼び出しを使っており、これ自体が「検証済みの経路を
  文字列として再解決する」構造で TOCTOU 窓の類型に該当した。
  2026-09-28 にユーザー承認を得て `libc =0.2.189`（`cfg(unix)` 限定。
  `.claude/rules/deps-policy.md`「OS FFI」区分）を導入し、次の設計へ
  是正した:
  - **Linux**: `openat2(2)`（`RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS |
    RESOLVE_NO_MAGICLINKS`）で経路解決全体を 1 回のシステムコールとして
    カーネルへアトミックに封じ込めさせる（`no_follow_open::
    open_chain_openat2`。`libc::syscall(libc::SYS_openat2, ...)` 経由。
    `libc` は `open_how`／`SYS_openat2`／`RESOLVE_*` 定数は提供するが
    `openat2()` 関数ラッパー自体は未提供のため）。`openat2` 未対応
    （`ENOSYS`／`EPERM`）の場合のみ次のフォールバックへ委譲する。
  - **フォールバック（macOS・その他 unix は常にこちら）**: 成分ごと逐次
    `openat(dirfd, name, O_NOFOLLOW)` で辿る（`open_chain_component_walk`）。
    `libc` の定数（`O_DIRECTORY`／`O_NOFOLLOW`／`O_CLOEXEC`／
    `O_NONBLOCK`／`ELOOP`／`ENOTDIR` 等）を使うことで、OS・アーキテク
    チャごとの値の違いは `libc` クレートが吸収する（自作の手書き定数を
    廃止）。
  - **診断（エラー種別の分類）**: 既に開いた親ディレクトリ fd を起点に
    した単一コンポーネントの `fstatat(AT_SYMLINK_NOFOLLOW)`
    （`is_symlink_component`）のみを使い、パス文字列の再解決を一切
    行わない。
  - **非 unix**（Windows 等）: 上記の安全な経路解決手段を持たないため、
    `resolve_and_open` は常に `UnsupportedPlatformForSecureResolve` で
    拒否する（fail-closed のまま変更なし）。Windows 対応はイシュー
    #2349 で追跡中。
- **O_NONBLOCK 未指定によるハングの是正（2026-09-28・Cursor Bugbot
  High 指摘 PRRT_kwDOTuUCJc6mk6-d）**: `no_follow_open::openat_no_follow`
  は `O_NOFOLLOW` のみを指定しており、`base_dir` 配下に FIFO（named
  pipe）等の特殊ファイルが置かれていた場合、`is_file()` による種別
  検証より前の `open`/`openat` 自体が対向の reader/writer 待ちで無期限
  にブロックし得た。中間ディレクトリ・最終ファイルいずれの `openat`
  呼び出しにも `O_NONBLOCK` を無条件で付与するよう是正した（通常
  ファイル・ディレクトリの open には副作用が無い POSIX の性質を利用）。
  `openat2` 経路（`open_how.flags`）にも同じ理由で無条件付与する。
- `base_dir` 自体の信頼は呼び出し元の責務とする（呼び出し元が与える
  信頼済み入力として扱い、location 側だけを fail-closed に検証する）。
- `checksum` の検証（SHA-1）は本 issue のスコープ外（依存を追加でき
  ないため）。拒否することで no-silent-skip 契約を守る。

## 6. facade 公開・合計上限の既定値

- **facade へのパス入力 import 入口の公開（2026-09-28 ユーザー承認・
  実施済み）**: 既存 API `crate::facade::interop::onnx::OnnxModel::
  from_path`（新規 API 名の追加ではない）を external data 対応へ拡張
  した。モデルファイルの親ディレクトリを `base_dir` とし、既定の
  `ExternalDataOptions::default()` を使う。`OnnxModel::from_bytes` は
  従来どおり external data を fail-closed 拒否する（挙動不変。回帰
  テストで固定）。`docs/facade-onnx-import-exposure-decision.md` §6.3・
  `docs/compat-api-scope.md` §5 の承認待ち記録を解消した。
- `ExternalDataOptions::max_total_bytes` の既定値は 2026-09-28
  ユーザー承認により 4 GiB から **64 GiB** へ改定した
  （`DEFAULT_MAX_EXTERNAL_DATA_TOTAL_BYTES`。`options` で変更可能な
  ままであることは不変）。
- `ExternalDataOptions::max_external_files` の既定値（4096）は 2026-09-28
  ユーザー承認により当初の暫定値をそのまま正式な既定値として確定した
  （`DEFAULT_MAX_EXTERNAL_FILES`。`options` で変更可能なままであることは不変）。

## 7. スコープ外の事項（`.claude/rules/out-of-scope-tracking.md`）

- external data での export（`onnx::export`）。常に inline（`raw_data`）
  で書き出す契約は不変。
- `checksum`（SHA-1）の検証。
- `max_external_files` の既定値（4096）の承認（6 節）。

自動運転中はユーザー承認を取れないため Issue は起票せず、本節と PR 本文に
起票候補として記録する。

## 8. テスト・実測

- 合成入力の網羅テスト: `crates/onnx-interop/tests/onnx_external_data.rs`
  （45 テスト。正常系〈FLOAT/INT64/BOOL/FLOAT16・offset 省略・length 省略・
  隣接区間・Constant 属性テンソル〉・異常系〈A2〜A5 のパス検証・数値検証・
  重複検証・キー検証・A6 回帰〉。base_dir 外へのシンボリックリンク脱出
  〈`symlink_escaping_base_dir_via_absolute_target_is_rejected`〉を含む）。
  `openat2`／逐次 `openat(O_NOFOLLOW)` フォールバックの両方式を直接検証
  する単体テストは `crates/onnx-interop/src/onnx/external_data.rs::tests`
  （2 テスト）。
- PyTorch 実生成 fixture: `crates/onnx-interop/tests/fixtures/
  pytorch-onnx-external-data/`・`tests/onnx_interp_pytorch_cnn_fixture.rs`
  の `external_data_fixture_*` 3 テスト＋
  `external_data_baselines_are_well_formed`（実測記録は `docs/perf/logs/
  onnx-external-data-pytorch-fixture-2347/README.md`・fixture 側の
  `README.md` を正とする）。`external_data_fixture_matches_self_contained_
  reference` は REQ-7 事前固定式（`fail_count == 0`）に加え、#2329 の
  `Req7BaselineNonRegression` と同じ仕組み（`EXTERNAL_DATA_BASELINES`。
  `total`／`max_abs_diff`／`max_rel_err`／`mean_abs_diff` の実測値
  そのものを ceiling とする fail-closed 非後退判定）を適用する
  （2026-09-28 ユーザー承認。コミット済みの固定 fixture は重みが変わら
  ないため baseline を張れる。`pytorch-onnx-cnn-ops` 側の
  `REDUCTION_BASELINES` とは別テーブル——両 fixture は独立に再生成
  されうるため）。`MaxPool`／`Flatten` 等のパラメータを持たない op は
  external data になる initializer 自体を持たないため、external data
  fixture のケース集合には含まれない（`EXTERNAL_DATA_CASE_NAMES` の
  doc コメント参照）。
- facade A6 回帰: `crates/facade/tests/interop_onnx_internal_parity.rs::
  facade_from_bytes_rejects_external_data_model_from_path_attempts_resolution`。

## 9. OWASP Top 10 観点

4 節の 2 パス設計・5 節の残るリスクを参照。要点は `security.md` A03
（外部フォーマットのパース検証を長さ・形状の検証が先行する）・A04
（確保の前に合計上限を検査する）・A08（`checksum` を黙って無視しない・
no-silent-skip 契約）。`unsafe` は `no_follow_open`（`cfg(unix)` 限定）の
`libc::openat`／`libc::fstatat`／`libc::syscall(SYS_openat2, ...)` FFI
呼び出しに限定して使用する（2026-09-28・#2347 是正で `libc =0.2.189`
〈`.claude/rules/deps-policy.md`「OS FFI」区分〉を導入。呼び出し箇所には
`coding-rust.md` 準拠の `// SAFETY:` コメントを付与済み。5 節参照）。
