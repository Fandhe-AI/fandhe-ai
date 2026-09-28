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
    大量の異なるファイルへ分散させると open／`fstat` の回数だけが無制限に
    増えうる。A04 資源枯渇対策）を塞ぐ。超過は `ExternalDataError::
    TooManyExternalFiles` で拒否する（#2347 P0 是正・PR #2348 コード
    レビュー対応・PRRT_kwDOTuUCJc6mlxhy）。**本上限は同時保持 fd 数の
    上限ではない**: 4 節のハンドル非保持構成により同時に開く external
    data ファイルは常に 1 つで、4096 がプロセスの fd 上限（例: soft
    limit 1024）を上回っても `EMFILE` は生じない（PR #2348 codex P1 是正）。
- `pub fn resolve_external_data(model: &mut ModelProto, base_dir: &Path, options: &ExternalDataOptions) -> Result<(), GraphError>`
  — in-place で external なテンソルを `raw_data` へ inline 化する
    （読み込みバッファは失敗可能確保。グラフ構築まで行う場合は、復号も
    失敗可能確保になる `build_graph_with_external_data` を使う。4.3 節）。
- `pub fn build_graph_with_external_data(model: &ModelProto, base_dir: &Path, options: &ExternalDataOptions) -> Result<Graph, GraphError>`
  — `model.clone()` → `resolve_external_data` 相当の inline 化 → グラフ
    構築の順に呼ぶ新しい import 入口。external テンソルが 0 件なら
    `graph::build_graph` へそのまま渡し、1 件以上なら `build_graph` と
    検証ロジックを共有する所有権ベースの `graph::build_graph_owned`
    （external 由来 initializer を失敗可能確保で復号・ノード列を move）へ
    渡す（2026-09-28・PR #2348 codex P0 是正。4.3 節）。
- `GraphError` に variant `ExternalData(ExternalDataError)` を 1 つだけ
  追加した（`crates/onnx-interop/src/onnx/graph.rs`）。
- `ExternalDataError`（`#[non_exhaustive]`・`Debug`/`Clone`/`PartialEq`/`Eq`）
  は `InvalidDataLocation`／`InconsistentDataFields`／`MissingLocationKey`／
  `DuplicateKey`／`UnknownKey`／`ChecksumUnsupported`／`InvalidLocation`
  （`LocationRejectReason` 付き）／`InvalidNumber`／`RangeOutOfFile`／
  `LengthMismatch`／`OverlappingRegion`／`TotalSizeLimitExceeded`／
  `TooManyExternalFiles`／`Io`／`FileChangedDuringLoad`／`InvalidBaseDir`／
  `DuplicateInitializerName`／`UnsupportedPlatformForSecureResolve`／
  `AllocationFailed`（2026-09-28・PR #2348 codex P0 是正で追加。4.3 節）／
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
     拒否する（全テンソルの計画後にファイル単位のソート＋走査で
     O(n log n) に判定する。4.2 節）。initializer 名の重複は I/O の前に
     拒否する。
   - 全テンソルの `length` を `checked_add` で積み上げ、
     `ExternalDataOptions::max_total_bytes` を超えた時点（または
     overflow した時点）で `TotalSizeLimitExceeded` とする。確保より
     前に検査するため、巨大な `length` でメモリを確保することはない。
   - distinct な external data ファイル実体（`FileKey`。dev/ino ベース）
     の数を `ExternalDataOptions::max_external_files` と比較し、既知
     ファイル集合（`known_keys`）へ登録する前（＝ファイルを開いて
     `FileKey` を得た直後）に超過を検査する。`max_total_bytes` はバイト
     数のみを制限するため、サイズ 0 のテンソルを大量の異なる空ファイルへ
     分散させる入力は合計サイズを常に 0 に保ったまま open／`fstat` の
     回数だけを増やしうる。この検査で `TooManyExternalFiles` として拒否
     する（2026-09-28・#2347 P0 是正・PR #2348 コードレビュー対応・
     PRRT_kwDOTuUCJc6mlxhy）。
   - **パス 1 はファイルハンドルを保持しない**（2026-09-28・PR #2348
     codex P1 是正）: distinct な正規化済み location ごとに安全 open →
     ハンドル自身の `fstat` で `FileKey` と `FileSnapshot`（ファイル長・
     unix では dev/ino・ctime・mtime〈秒＋ナノ秒〉）だけを記録
     （`PlannedLocation`）→ 直ちに close する。同一 location を複数
     テンソルが参照する場合は location キャッシュで 2 件目以降の open を
     省略し、異なる location 名が同一実体（ハードリンク等）を指す場合は
     個別に開いた `FileKey` で畳み込んで重複区間検出を行う（意味論は旧
     構成と同一。長さ 0 の区間は重複検査の対象外）。
2. **パス 2（`load`）**: パス 1 が全件成功した場合のみ、正規化済み
   location ごとに 1 ファイルずつ「パス 1 と同じ安全 open（同じ
   `base_dir` fd 起点の `openat2`／逐次 `openat(O_NOFOLLOW)`）→ 開いた
   ハンドル自身の `FileKey`〈dev, ino〉・`FileSnapshot`（ファイル長・
   unix では dev/ino・ctime・mtime）をパス 1 の記録と完全一致で照合
   （`ensure_unchanged`）→ その location を参照する全テンソルの区間だけを
   読む（`read_region`。`.data` ファイル全体は読まない。4.3 節）→ close」
   を逐次に行う。
   各区間の読み込み直前にも同じハンドルへ `fstat` を取り直して同じ照合を
   行う。照合の不一致は `FileChangedDuringLoad`、再 open 自体の失敗
   （削除による `NotFound`・シンボリックリンクへの差し替え等）はパス 1 と
   同じ variant（`Io`／`InvalidLocation`）でいずれも fail-closed に拒否
   する。**ただし最後の照合を通過した直後〜読み込みの間に truncate
   された場合は、照合ではなく `read_region` の読み込み不足
   （`UnexpectedEof`）として
   `ExternalDataError::Io` で拒否される**（`FileChangedDuringLoad` とは
   別 variant だが fail-closed。2026-09-28・PR #2348 security-auditor
   P2-2。同じ窓での同長 in-place 書き換えの扱いは 5 節）。

### 4.1 ハンドル非保持の構成（2026-09-28・PR #2348 codex P1 是正）

- **指摘**: 旧構成はパス 1 で開いた distinct ファイルのハンドルをすべて
  保持したままパス 2 で再利用していた。`max_external_files` の既定値
  4096（ユーザー承認済み）は一般的なプロセスの fd 上限（例: soft limit
  1024）を上回るため、小さなファイルを多数参照するモデルでは
  `TooManyExternalFiles` に到達する前に `EMFILE` が発生し、同一プロセスの
  ほかの I/O にも影響し得た。
- **採用した構成**: 上限値（4096）は下げず、「検証パス（ハンドル非保持）
  と読み込みパス（location ごとに再 open → 照合 → 読込 → close）の分離」
  とした。同時に保持する fd は `base_dir` のディレクトリ fd と処理中の
  external data ファイル 1 つの高々 2 つ（`openat2` 非対応時の逐次方式
  では経路途中のディレクトリ fd が一時的に 1 つ加わる）で、ファイル数に
  依存しない。
- **不採用とした構成**: 「単一パスでファイルごとに検証・読み込み・close
  を進める」構成。`length` 省略時の読み込み長は `file_len - offset` で
  確定するためファイルを開かずに総量上限・ファイル数上限を判定できず、
  単一パスでは上限超過を途中まで読んでから検出することになる（「1 件でも
  検証に失敗すればファイルは一切読まない」という 1. の契約を崩す）。
- **TOCTOU の論拠**: 読み込みは常に「パス 2 で安全 open し、そのハンドル
  自身に対する `fstat` の `FileKey`〈dev, ino〉・ファイル長・ctime・mtime
  がパス 1 の記録と完全一致したハンドル」からのみ行い、経路文字列を
  再解決しない（再 open も `base_dir` fd 起点・シンボリックリンク拒否の
  同じ手段）。よって (1) `base_dir` 外・シンボリックリンク経由のファイルは
  読まない、(2) 読む実体はパス 1 で当該 location について検証した実体と
  同一（照合できる範囲。5 節の残存リスク参照）で長さも同一、(3) 読み
  込み量はパス 1 で上限検査済みの区間に有界、の 3 点を保証する。ハンドル
  非保持に伴い旧構成には無かった差し替えの窓（unlink → 同長の別ファイル
  作成 → inode 番号再利用）が生じるため、ctime・mtime を照合に加えて
  これを塞ぐ（2026-09-28・PR #2348 security-auditor P2-1。残存条件は
  5 節）。

### 4.2 区間重複検査の計算量（2026-09-28・PR #2348 codex P0 是正）

- **指摘**: 旧実装はテンソルを 1 件計画するたびに、同一ファイルの既存
  区間を全走査して重なりを調べていた。1 つの external data ファイルへ
  重ならない短い区間を多数指定した入力では検証が O(n²) になり、入力だけ
  で処理を停止させられる（A04 資源枯渇）。`max_external_files` は同一
  ファイルを 1 件としか数えず、`max_total_bytes` は短い区間の合計しか
  制限しないため、既存の上限では抑えられない。
- **採用した方式**: パス 1 のテンソルごとの処理では区間
  `(offset, end, entry_idx)`（`entry_idx` は external テンソルの入力順）を
  `FileKey` で畳み込んだファイル単位に収集するだけとし、全テンソルの
  計画後にファイルごとに `(offset, entry_idx)` の昇順へソートして 1 回
  走査する（`external_data.rs::find_overlap`）。走査中はそれまでの最大
  `end` とその持ち主を保持し、次の区間の `offset` が最大 `end` 未満なら
  持ち主の区間と重なっているとして `OverlappingRegion` を返す（直前の
  区間ではなく最大 `end` と比べるため包含関係も漏らさない。境界が
  接するだけ〈`end == 次の offset`〉は重なりではない）。計算量は
  ファイルあたり O(n log n)。長さ 0 の区間は従来どおり対象外。
- **エラーの意味論**: variant（`OverlappingRegion { tensor_name,
  other_tensor_name }`）と中身の意味（重なり合う 2 テンソルの名前）は
  不変。報告するペアの選び方のみ変わる: 旧実装は「後から計画された
  テンソル」と「それと重なる既存テンソルのうち最初のもの」、新方式は
  「ソート順で後側の区間」と「その時点で最大 `end` を持つ区間（同値なら
  先に現れた区間）」。ソートキーに入力順を含め、ファイルは初出順
  （`file_ids` が振る番号順）に検査するため、報告は入力に対して決定的。
- **判定順序**: 上限判定（合計サイズ・distinct ファイル数）は従来どおり
  各テンソルの計画時に行い、重複検査はその後（全テンソル分の上限判定を
  終えてから）・パス 2 の読み込みより前に行う。いずれも読み込み前の
  fail-closed 判定であることは変わらない。重複と別種の検証エラー
  （例: 後続テンソルの `LengthMismatch`）が同じモデルに併存する場合は、
  旧実装と異なり別種のエラーが先に報告されうる。
- **同類型の洗い出し**（未信頼入力で回数が決まるループの二次計算）:
  `external_data.rs` の `plan`／`load`／`resolve_external_data`、呼び出し
  側の `graph::build_graph`、facade `OnnxModel::from_path` を確認し、
  該当は上記の区間重複検査のみだった（initializer 名の重複・location
  キャッシュ・distinct ファイル集合・`build_graph` の名前解決はいずれも
  `HashSet`／`HashMap`、`load` の location 別振り分けは添字による
  `Vec` で線形。テンソルごとの `external_data` キー走査は重複・未知キー
  で即座に拒否するため高々数件）。
- **回帰テスト**: `external_data.rs::overlap_tests`（隣接〈接するだけ／
  1 バイト重なる〉・包含・長さ 0 の除外・同一 offset の決定的な報告・
  入力順と位置順の不一致・20 万区間）と、統合テスト
  `many_non_overlapping_unit_regions_in_one_file_resolve`（1 ファイルに
  長さ 1 の区間 5 万件を公開入口経由で解決。実時間の閾値判定はせず、
  通ること自体と値の正しさで確認）・
  `one_overlap_among_many_unit_regions_is_rejected_with_names`。参考実測
  （x86_64 Linux・debug ビルド）: 同じ 5 万区間の統合テストは旧実装で
  2.50 秒、新方式ではテストファイル全体（50 テスト）で 0.22 秒。

読み込んだバイト列は `raw_data` へ書き戻し、`data_location = DEFAULT`・
`external_data` は空にする（パス 2 完了後にのみ書き戻す。検証・読み込み
が全件成功した場合限定）。

対象は `GraphProto.initializer` に加え、各 `NodeProto.attribute[].t`
（Constant の `value` 等）も含む（`enumerate_tensors`）。サブグラフ属性
（`g`/`graphs`）は `proto::AttributeProto` に未定義のため対象外。

### 4.3 メモリ確保の失敗可能化と読み込み経路のメモリ予算（2026-09-28・PR #2348 codex P0 是正）

- **指摘**（HEAD 33b942a6 へのレビュー）: `load` は検証済みの
  `entry.length` に対して `vec![0u8; buf_len]` で一度に確保していた。
  `max_total_bytes` の既定値は 64 GiB であり、小さな `.onnx` と疎ファイル
  から数十 GiB の単一テンソルを宣言できるため、利用可能メモリが足りない
  環境では型付きエラーを返す前に確保失敗（`handle_alloc_error`）で
  プロセスが終了しうる（security.md A04）。
- **採用した方式**:
  1. `load` の区間バッファは `external_data.rs::read_region` で確保する:
     `alloc_region_buf`（`usize::try_from` ＋ `Vec::try_reserve_exact`）で
     容量ちょうどの空 Vec を用意し、`Read::take(length)` を挟んだ
     `read_to_end` で埋める（ゼロ初期化の二重書き込みを避け、未初期化
     メモリを `unsafe` で扱わない。容量ちょうどまで埋まると std は
     スタック上の探査読み込みで EOF を確かめて再確保しないことを単体
     テストで実測固定）。`File` に直接 `read_to_end` しない（`File` の
     特殊化は残りファイル長で追加確保しうる）。読み込み不足は旧実装の
     `read_exact` と同じ `Io { kind: UnexpectedEof }`（4 節 2.・5 節の truncate
     に関する記述は機構名を `read_region` へ更新済み）。確保は従来どおり
     `ensure_unchanged` の照合通過後に限る。
  2. 確保失敗は新 variant `ExternalDataError::AllocationFailed {
     tensor_name, bytes }` で返す。`usize` へ変換できない長さ（32bit
     ターゲット）も、旧実装の `Internal` ではなく本 variant とする
     （内部不変条件違反ではなく「このプロセスでは確保できないサイズ」の
     ため）。`isize::MAX` 超の要求は `try_reserve_exact` がアロケータを
     呼ばずに `CapacityOverflow` で失敗するため、確保を試みる前に拒否
     される。
  3. `build_graph_with_external_data` は、external テンソルを 1 件以上
     持つモデルを所有権ベースの新関数 `graph::build_graph_owned`
     （`pub(super)`）で構築する。`build_graph` の検証本体（sparse 拒否・
     initializer 名重複・トポロジ・グラフ出力）を `reject_sparse_
     initializers`／`insert_initializer`／`validate_topology` へ抽出して
     両者で共有し（検証ロジックを二重実装しない。検査順序・エラー
     variant は不変）、(a) external 由来の initializer は
     `try_decode_external_initializer`（`try_alloc_vec` で確保した Vec へ
     `decode_tensor` と同じリトルエンディアン変換。BOOL は非ゼロ→true）で
     復号し、`raw_data` を `mem::take` して復号直後に解放する、(b) inline
     の initializer は従来どおり `decode_tensor`（入力バイト列長で有界）、
     (c) ノード列は clone せず move する（external 由来の Constant 属性
     テンソルの `raw_data` を 2 重に持たない）。external テンソルが 0 件の
     モデルは従来どおり `build_graph` へそのまま渡す。
  4. facade `map_graph_error` は `AllocationFailed` を既存の
     `OnnxError::Io(ErrorKind::OutOfMemory)` へ写像する（新規 variant
     なし）。同じ `from_path` 内の `std::fs::read` が `.onnx` 本体の確保
     失敗を std の `try_reserve` 規約どおり `Io(OutOfMemory)` で返すため、
     どちらの確保失敗も利用者が `kind() == OutOfMemory` で一様に判別
     できる。`InvalidModel` へ畳み込むと資源不足を「モデル不正」と誤分類
     するため採らない。`tensor_name`・`bytes` は既存の `Io` 写像と同じく
     落ちる。あわせて `from_path` はデコード後に `.onnx` 本体のバイト列を
     解放してから external data の読み込みへ進む。
- **A6 との関係**: `graph::decode_tensor` は変更していない（3 節の
  不変条件は維持。失敗可能復号は external 入口専用の別関数）。
  `graph::build_graph` は検証ヘルパの抽出のみで挙動不変（既存テスト
  すべて無変更で pass）。所有権ベースの構築結果が旧経路（複製 →
  `resolve_external_data` → `build_graph`）と同一の `Graph` になることを
  `tests/onnx_external_data.rs::owned_build_matches_resolve_then_build_
  graph`（4 dtype の external initializer・inline〈`raw_data`／
  `float_data`〉・external Constant 属性を併せ持つモデル）で固定した。
- **同類型の洗い出し**（未信頼の宣言値〈`length`・dims の積〉から無条件
  確保している箇所。external data 読み込み結果がテンソルへ変換される
  までの全経路と facade `OnnxModel::from_path` を対象）:

  | 箇所 | 確保方式（是正前） | 判定 | 対応 |
  |------|------------------|------|------|
  | `external_data.rs::load` の区間バッファ | `vec![0u8; length]` | external 由来で巨大化しうる | `read_region`（`try_reserve_exact` ＋ `take().read_to_end`）へ |
  | `graph::build_graph` → `decode_tensor` の要素 Vec（external 由来 initializer） | `collect`（正確なサイズヒントによる無条件確保） | external 由来で巨大化しうる（raw と同時に存在し 2 倍） | `build_graph_owned` ＋ `try_decode_external_initializer`（失敗可能確保・raw を復号直後に解放） |
  | `graph::build_graph` の `g.node.clone()`（external 由来の Constant 属性テンソルの `raw_data`） | clone（無条件確保） | external 由来で巨大化しうる | `build_graph_owned` で move（確保自体を無くす） |
  | `build_graph_with_external_data` の `model.clone()` | clone | 複製時点で external テンソルの `raw_data` は空（`plan` が inline データ非空を拒否）。`.onnx` 本体の長さで有界 | 対象外 |
  | `plan` の各 Vec／HashMap・`load` の `by_location`／`out`・`resolve_external_data_slots` の slot 列・external initializer 判定の `vec![false; n]` | 無条件確保 | テンソル件数（`.onnx` 本体で有界）に比例し宣言長に依存しない | 対象外 |
  | `enumerate_tensors`・`decode_tensor` の `dims.clone()`／名前の複製 | 無条件確保 | `.onnx` 本体で有界 | 対象外 |
  | `decode_tensor` の inline 由来 `raw_data`／`float_data`／`int64_data`（バイト列入口・external 入口の inline 分） | `collect`／`clone` | 入力バイト列長で有界（dims 積は `raw_data` 長との照合**後**にのみ確保に使う。照合前確保なし） | 対象外（A6 により不変） |
  | facade `from_path` の `std::fs::read` | std 内部で `try_with_capacity` | 既に失敗可能（`Io(OutOfMemory)`） | 対象外（デコード後に解放する変更のみ） |
  | `interp::compute_constant`（`decode_tensor` ＋ `raw_to_value` の `data.clone()`）・`interp::run` が実行ごとに initializer を env へ clone | `collect`／clone | external 由来で巨大化しうるが**実行（`run`）経路**であり読み込み経路ではない | 本 P0 の範囲外。7 節の起票候補 |
  | `onnx::autograd` の Constant 属性復号（`decode_tensor`） | `collect` | 同上（学習実行経路） | 本 P0 の範囲外。7 節の起票候補 |
  | `resolve_external_data`（pub）を直接呼び、続けて `build_graph` を呼ぶ利用者 | 読み込みは失敗可能、復号は `decode_tensor` の `collect` | 復号側は無条件確保のまま | 推奨入口は `build_graph_with_external_data`（doc に明記）。`resolve_external_data` の契約（raw inline）は不変 |

- **ピークメモリ見積もり**（`N` = external data 合計（≤
  `max_total_bytes`）、`L_max` = 最大の external initializer、`S` =
  `.onnx` 本体の長さ）:
  - 是正前: 全 raw（`N`）が複製モデル内に残ったまま `build_graph` が全
    initializer を復号し（＋`N`）、さらにノード列の clone で Constant
    属性の raw を複製するため、最大でおよそ `2N`（＋Constant 分）＋
    `O(S)`。いずれも無条件確保。
  - 是正後: 読み込み完了時点で raw `N`、以後 initializer を 1 件ずつ復号
    して直後に raw を解放するため、ピークはおよそ `N + L_max + O(S)`
    （`O(S)` は デコード済み `ModelProto` と複製の 2 つ分）。最悪（単一の
    巨大テンソル）で `2 × max_total_bytes`。構築後の `Graph` の保持量は
    およそ `N`（復号済み initializer ＋ Constant 属性の raw）。
  - 実行時（範囲外。参考）: `interp::run` は実行ごとに initializer を
    env へ clone するため ＋`N`（Constant 属性は実行時に復号 ＋ clone で
    一時的に ＋`2 ×` 当該分）。
- **低メモリ環境での運用**: `onnx-interop` の `build_graph_with_external_
  data`／`resolve_external_data` を直接呼ぶ利用者は、`ExternalDataOptions
  { max_total_bytes, .. }` を利用可能メモリに合わせて下げて渡せる
  （既存オプション。上限既定値 64 GiB 自体は変更しない。疎ファイルで
  4 GiB の単一テンソルを宣言したモデルが下げた予算で確保前に拒否される
  ことを `sparse_file_huge_tensor_is_rejected_by_lowered_budget_before_
  allocation` で固定）。一方 facade `OnnxModel::from_path` は
  `ExternalDataOptions::default()` 固定で、facade 利用者が予算を下げる
  公開手段は無い（公開 API 面を変えないため本 P0 では追加しない。7 節の
  起票候補）。この場合も確保失敗は abort ではなく
  `OnnxError::Io(OutOfMemory)` になる。
- **回帰テスト**: 数十 GiB の実確保は CI で危険なため行わない。
  `external_data.rs::alloc_tests`（`try_alloc_vec`／`alloc_region_buf` の
  `usize::MAX`・`isize::MAX + 1`・`u64::MAX` 要求が `AllocationFailed`、
  `read_region` が容量ちょうどまで埋め再確保しないこと・読み込み不足が
  `UnexpectedEof`、`try_decode_external_initializer` が 4 型で
  `decode_tensor` と同一の `RawTensor` を返し raw を解放すること、余り
  バイトの `Internal` 拒否）、`external_data.rs::tests::load_rejects_
  unallocatable_length_as_allocation_failed`（`plan` 済み計画の
  `length` を `isize::MAX + 1` に書き換えた `load` が確保前に
  `AllocationFailed`）、統合テスト 4 件（上記の同一性・`owned_build_
  keeps_topology_validation`・下げた予算での疎ファイル拒否・
  `sparse_file_unallocatable_tensor_returns_allocation_failed_instead_of_
  abort`〈指摘の再現経路そのもの: 4 TiB の疎ファイル＋4 TiB の単一
  テンソル＋`max_total_bytes = u64::MAX` で `AllocationFailed` が返る。
  Linux・`vm.overcommit_memory` が 0／2 の環境限定。区間バッファの確保を
  `vec![0u8; n]` へ戻す変異でテストプロセスが SIGABRT で落ちることを
  確認済み〉）、facade `interop::onnx::
  map_graph_error_tests`（`AllocationFailed` → `Io(OutOfMemory)`）。

## 5. 残るリスク（受容済み）

- **確保成功後のページ実コミット時の OOM（2026-09-28・PR #2348 codex P0
  是正に伴い明記）**: 失敗可能確保（4.3 節）が型付きエラーにできるのは、
  アロケータが確保要求そのものを拒否した場合（アドレス空間不足・
  `isize::MAX` 超・overcommit ヒューリスティックによる拒否等）に限る。
  Linux の `vm.overcommit_memory=1` 等で確保要求が成功した場合、読み込み
  でページを実際に書き込む時点で OS の OOM killer がプロセスを終了させ
  うる。これはユーザー空間の確保 API では検知できないため受容し、緩和は
  呼び出し側が `max_total_bytes` を利用可能メモリに合わせて下げることで
  行う（4.3 節「低メモリ環境での運用」）。

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
- **パス間でハンドルを保持しないことによる inode 番号再利用の窓
  （2026-09-28・PR #2348 codex P1 是正に伴い発生。同日 security-auditor
  P2-1 で記述を訂正し ctime／mtime 照合を追加）**: パス 1 で close した
  ファイルを unlink し、同じ長さの別ファイルを同じ名前で作成すると、
  ファイルシステムによっては inode 番号が再利用され dev/ino・長さが
  一致しうる。当初の記述は「dev/ino・長さ照合を通過する差し替えは
  同一 inode の in-place 改変と同等の能力でしか起こせない」としていたが
  不正確で、実際には `base_dir` 配下の**ディレクトリ書き込み権
  （unlink／create）だけ**で起こしうる（旧構成はハンドルを保持していた
  ためこの窓自体が無かった）。是正として `PlannedLocation` に ctime・
  mtime（`st_ctime`＋`st_ctime_nsec`・`st_mtime`＋`st_mtime_nsec`。
  `std::os::unix::fs::MetadataExt`）を記録し、パス 2 の再 open 直後と
  各区間の読み込み直前の照合で完全一致を要求する（不一致は
  `FileChangedDuringLoad`）。新しく作られた inode の ctime は作成時刻に
  なり、同一 inode の in-place 改変（write・truncate・chmod・link 等）でも
  ctime は更新される。ctime はユーザー空間から任意の値へ設定できない
  （`utimensat` で mtime を書き戻す操作自体が ctime を更新する）ため照合の
  要は ctime で、mtime は補助である（atime は読み込みだけで更新されうる
  ため照合しない）。
  - **残存リスク（受容）**: 照合を通過する差し替えには「ディレクトリ
    書き込み権 ＋ inode 番号の再利用 ＋ 長さ一致 ＋ ctime 一致」が必要
    で、ctime 一致は**ファイルシステム／カーネルのタイムスタンプ粒度内
    での再作成**を意味する。秒単位の粒度しか持たないファイルシステム
    （一部の古い形式・ネットワークファイルシステム等）では同一秒内、
    ナノ秒表現を持つファイルシステムでもカーネルがタイムスタンプに
    粗いクロック（tick 刻み・数 ms）を使う場合はその刻み内の再作成で
    通過しうる。
  - **照合と読み込みの間の残存窓（旧構成と同じ）**: 各区間の直前の照合を
    通過した直後〜読み込みの間に truncate されると `read_region` の
    読み込み不足（`UnexpectedEof`）として `ExternalDataError::Io` で fail-closed に拒否
    される（4 節 2.）。同じ窓で同じ長さのまま in-place 書き換えされた
    場合は検出できないが、これは旧構成（ハンドル保持）でも同一で、
    `base_dir` 配下のファイル自体への書き込み権を前提とする。
  - (a) パス 1 内で別ファイルが同じ dev/ino を得ても `FileKey` の併合は
    重複区間検出を増やす方向（fail-closed 側）にしか働かず見逃しを
    生まない。
  - いずれも `base_dir` 配下・非シンボリックリンク・検査済み区間内の
    有界読み込みという保証は崩さないため、上記の残存リスクを受容する。
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
- （2026-09-28・PR #2348 codex P0 是正に伴う起票候補。4.3 節）facade
  `OnnxModel::from_path` から `max_total_bytes` を下げる公開手段（現状は
  `ExternalDataOptions::default()` 固定。公開 API 面の追加にはユーザー
  承認が要る）。
- （同上）実行経路の確保: `interp::run` が実行ごとに initializer を env
  へ clone する処理、`interp::compute_constant`／`onnx::autograd` の
  Constant 属性テンソル復号（`decode_tensor` の `collect`＋`raw_to_value`
  の clone）は、external data 由来で巨大化しうる無条件確保のまま
  （読み込み経路ではないため本 P0 の範囲外）。

自動運転中はユーザー承認を取れないため Issue は起票せず、本節と PR 本文に
起票候補として記録する。

## 8. テスト・実測

- 合成入力の網羅テスト: `crates/onnx-interop/tests/onnx_external_data.rs`
  （unix で 54 テスト〈うち Linux 限定 1 件〉＋非 unix 契約テスト 2 件。正常系〈FLOAT/INT64/BOOL/FLOAT16・offset 省略・length 省略・
  隣接区間・Constant 属性テンソル〉・異常系〈A2〜A5 のパス検証・数値検証・
  重複検証・キー検証・A6 回帰〉。base_dir 外へのシンボリックリンク脱出
  〈`symlink_escaping_base_dir_via_absolute_target_is_rejected`〉を含む）。
  ハンドル非保持構成（4.1 節）の回帰テストとして
  `many_small_external_files_load_with_bounded_open_handles`（512 本の
  別ファイルを参照するモデルの解決・値検査。Linux では解決前後の
  `/proc/self/fd` 件数の非増加も検査）と、それを fd soft limit 64 の
  子プロセス（`sh -c 'ulimit -S -n 64; exec …'`。テストプロセス自身の
  rlimit は変えない）で実行する
  `many_small_external_files_do_not_exhaust_fd_limit_in_child_process`
  を置く（`cfg(unix)`。旧構成では子プロセスが `Io { kind:
  TooManyOpenFiles }` で失敗することを実測確認済み）。子プロセスへ soft
  limit が実際に適用されていることの確認は `getrlimit(RLIMIT_NOFILE)`
  で unix 共通に行う（2026-09-28・PR #2348 security-auditor P2-3。旧実装は
  Linux 限定の `/proc/self/limits` 読み取りで、macOS 等では確認が
  空振りしていた）。
  単体テストは `crates/onnx-interop/src/onnx/external_data.rs::tests`
  （7 テスト。うち 1 件は 4.3 節の `load_rejects_unallocatable_length_
  as_allocation_failed`）: `openat2`／逐次 `openat(O_NOFOLLOW)` フォールバックの
  両方式の直接検証（2 テスト）と、パス 1／パス 2 間の差し替え検知
  （5 節。PR #2348 security-auditor P2-1）の 4 テスト——照合関数
  `ensure_unchanged` が長さ・dev・ino・ctime（秒／ナノ秒）・mtime（秒／
  ナノ秒）のいずれか 1 フィールドの差でも `FileChangedDuringLoad` に
  すること、`plan` → `load` の対照（差し替えなしで成功）、`plan` 後の
  unlink → 同長の別ファイル作成、`plan` 後の同一 inode・同長の in-place
  上書き（dev/ino・長さが不変で ctime／mtime だけが検出する経路。照合を
  dev/ino・長さのみに戻すと本テストが失敗することを変異確認済み）をそれ
  ぞれ `load` が拒否すること。区間重複検査（4.2 節。codex P0 是正）の
  回帰テストは統合テスト 2 件（`many_non_overlapping_unit_regions_in_
  one_file_resolve`・`one_overlap_among_many_unit_regions_is_rejected_
  with_names`。上記 54 テストに含む）と、全プラットフォームで実行する
  単体テスト `external_data.rs::overlap_tests`（7 テスト）。メモリ確保の
  失敗可能化（4.3 節）の回帰テストは全プラットフォームで実行する
  `external_data.rs::alloc_tests`（6 テスト）・統合テスト 4 件（上記 54
  テストに含む）・facade `interop::onnx::map_graph_error_tests`（3 テスト）。
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
