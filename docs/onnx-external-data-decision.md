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

## 2. API 形状（`onnx-interop` 内部限定。facade へは公開しない）

新規モジュール `crates/onnx-interop/src/onnx/external_data.rs`（`onnx::mod`
から `pub mod external_data;`）:

- `pub struct ExternalDataOptions { pub max_total_bytes: u64 }`
  - `Default` の既定値は `DEFAULT_MAX_EXTERNAL_DATA_TOTAL_BYTES = 4 GiB`
    （**暫定値・ユーザー承認待ち**。変更は定数 1 行の書き換えで済む）。
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
  `LengthMismatch`／`OverlappingRegion`／`TotalSizeLimitExceeded`／`Io`／
  `FileChangedDuringLoad`／`InvalidBaseDir`／`DuplicateInitializerName` の
  variant を持つ。診断文字列（`tensor_name`／`key`）は
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
  facade_rejects_external_data_model_bytes_and_path`（facade 経由）

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
     をすべて拒否）の両方で検証する。`base_dir` を起点にコンポーネント
     を 1 つずつ連結しながら各段で `symlink_metadata` を取り、経路の
     途中を含めシンボリックリンクを拒否する。解決結果を
     `canonicalize` し、canonicalize 済み `base_dir` で `starts_with`
     しなければ拒否する（多層防御）。
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

- std では `O_NOFOLLOW` 相当を使えない（`libc` を追加できないため）。
  そのためモデルのディレクトリへ書き込める攻撃者による TOCTOU 競合は
  完全には排除できない。パス 2 直前のファイル長・dev/ino 再照合が
  この窓を縮める多層防御である。
- `base_dir` 自体の信頼は呼び出し元の責務とする（呼び出し元が与える
  信頼済み入力として扱い、location 側だけを fail-closed に検証する）。
- `checksum` の検証（SHA-1）は本 issue のスコープ外（依存を追加でき
  ないため）。拒否することで no-silent-skip 契約を守る。

## 6. facade 公開・合計上限の既定値（承認待ち）

- facade へのパス入力 import 入口の公開可否は未確定のまま保留した
  （`docs/compat-api-scope.md` §5・`docs/facade-onnx-import-exposure-
  decision.md` §15）。
- `ExternalDataOptions::max_total_bytes` の既定値（4 GiB）は暫定値で
  あり、ユーザー承認が必要（`DEFAULT_MAX_EXTERNAL_DATA_TOTAL_BYTES` の
  1 行変更で調整可能）。

## 7. スコープ外の事項（`.claude/rules/out-of-scope-tracking.md`）

- external data での export（`onnx::export`）。常に inline（`raw_data`）
  で書き出す契約は不変。
- `checksum`（SHA-1）の検証。
- facade へのパス入力 import 入口の公開（承認待ち。6 節）。

自動運転中はユーザー承認を取れないため Issue は起票せず、本節と PR 本文に
起票候補として記録する。

## 8. テスト・実測

- 合成入力の網羅テスト: `crates/onnx-interop/tests/onnx_external_data.rs`
  （39 テスト。正常系〈FLOAT/INT64/BOOL/FLOAT16・offset 省略・length 省略・
  隣接区間・Constant 属性テンソル〉・異常系〈A2〜A5 のパス検証・数値検証・
  重複検証・キー検証・A6 回帰〉）。
- PyTorch 実生成 fixture: `crates/onnx-interop/tests/fixtures/
  pytorch-onnx-external-data/`・`tests/onnx_interp_pytorch_cnn_fixture.rs`
  の `external_data_fixture_*` 3 テスト（実測記録は `docs/perf/logs/
  onnx-external-data-pytorch-fixture-2347/README.md`・fixture 側の
  `README.md` を正とする）。
- facade A6 回帰: `crates/facade/tests/interop_onnx_internal_parity.rs::
  facade_rejects_external_data_model_bytes_and_path`。

## 9. OWASP Top 10 観点

4 節の 2 パス設計・5 節の残るリスクを参照。要点は `security.md` A03
（外部フォーマットのパース検証を長さ・形状の検証が先行する）・A04
（確保の前に合計上限を検査する）・A08（`checksum` を黙って無視しない・
no-silent-skip 契約）。`unsafe` は使用していない。
