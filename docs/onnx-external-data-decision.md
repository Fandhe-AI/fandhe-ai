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
     が構造的に保証される。**Windows**（イシュー #2349）は `openat` 相当
     （ディレクトリハンドル起点の相対オープン）を std が持たないため、
     祖先ディレクトリのハンドル連鎖保持＋開いたハンドル自身の reparse
     point 属性検査による封じ込めオープン（`win_contained_open`）で TOCTOU
     を閉じる（4.6 節）。**それ以外**（wasm32 等）は上記いずれの安全な
     経路解決手段も持たないため、`resolve_and_open` は常に
     `UnsupportedPlatformForSecureResolve` で拒否する（fail-closed。
     5 節参照）。
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

### 4.3 メモリ確保の失敗可能化とメモリ予算（読み込み・実行・export 経路。2026-09-28・PR #2348 codex P0 是正 2 回）

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
  5. **実行経路・export 経路（2026-09-28・PR #2348 codex P0 是正 2 回目）**:
     指摘（`Constant` 属性テンソルが実行時に `interp::compute_constant` の
     `decode_tensor`〈`collect`〉で無条件に復号され、既定上限 64 GiB 内の
     小さな `.onnx` と巨大な外部ファイルでメモリ不足時に abort しうる）を
     受け、同類型を実行経路・export 経路まで洗い出して一括で是正した
     （下表）。確保ヘルパは読み込み・実行・export の 3 経路で共有する
     非公開モジュール `onnx::fallible_alloc` へ集約した（`try_alloc_vec`・
     `try_clone_slice`・`decode_le_into`・`try_decode_tensor`・
     `try_clone_nodes`。`external_data` の `try_alloc_vec`／`try_decode_le`
     はこれへ委譲する薄いラッパー）。
     - `Constant` 属性テンソルの復号（`interp::decode_constant_tensor`・
       `autograd` の `Constant` 腕）は `fallible_alloc::try_decode_tensor`
       で行い、得た `Vec` を `Tensor` へ **move** する（旧実装は
       `decode_tensor` の結果を `raw_to_value` でさらに複製しており、実行
       ごとに一時 2 倍を要した。是正後は 1 倍）。`try_decode_tensor` は
       「`raw_data` 非空 かつ対応 4 型」の場合だけ `decode_tensor` と同じ
       順序で検証（`element_count` → `checked_mul` → 長さ照合）して同じ
       変換を行い、それ以外（typed data・空 raw・未知 dtype）は
       `decode_tensor` へ委譲する（検証ロジックの並行実装を最小にする。
       Ok・Err〈`RawDataByteLenMismatch`・`NegativeDim`・`ElementCount
       Overflow`・`UnknownDataType` 等〉とも `decode_tensor` と同じ結果に
       なることを単体テストで固定）。`decode_tensor` 自体は A6 により不変。
     - `interp::run` が実行ごとに initializer を実行時値へ複製する処理
       （`raw_to_value`）・`autograd::BoundGraph::bind` の同じ処理は
       `try_clone_slice`（`try_reserve_exact` ＋ `extend_from_slice`）で
       行う。
     - export（`export::build_model_proto`）はノード列の複製を
       `try_clone_nodes`（属性テンソル本体だけ失敗可能確保。構造体
       リテラルの全フィールド列挙でフィールド追加も検出）、`encode_tensor`
       の `raw_data` を失敗可能確保で作り、モデル全体の encode は新関数
       `export::try_encode_model`（`encoded_len()` 分を `try_reserve_exact`
       → `Message::encode`。`encode_to_vec` と同じ手順のため出力バイト列は
       同一）で行う。facade `OnnxModel::to_bytes` は `proto::encode_model`
       の代わりにこれを使う（`encode_model` 自体は既存シグネチャのまま
       残す）。
     - 新 variant: `InterpError::AllocationFailed { tensor_name, bytes }`・
       `ExportError::AllocationFailed { tensor_name, bytes }`（いずれも
       `#[non_exhaustive]` の内部 enum。`autograd` は既存の
       `AutogradError::Interp` 経由で同じ variant を返す）。
       `InterpError::Graph(GraphError::ExternalData(AllocationFailed))` の
       再利用は、inline 由来テンソルにも external data のエラーを返す
       ことになるため採らない。facade は両者を読み込み時と同じ既存の
       `OnnxError::Io(ErrorKind::OutOfMemory)` へ写像する（公開 variant
       追加なし。`Execution { message }` へ畳み込むと資源不足を型で判別
       できず、読み込み時と実行時で判別方法が分かれるため採らない。
       `docs/facade-onnx-import-exposure-decision.md` §15）。
     - **読み込み時に復号済みテンソルを `Graph` へ保持する方式は採らない**:
       (a) `Graph` は pub フィールドの構造体で、`initializers` と同じく
       `RawTensor`（`Vec` 所有）でしか保持できない。`Tensor::new` は
       `Vec` を所有で受けるため、`Graph` を借用のまま複数回 `run` できる
       契約の下では実行時の `Tensor` 化で結局 1 回複製が要り、実行時
       ピークは本方式（実行時に失敗可能復号して move）と同じになる。
       (b) `export::build_model_proto` はノード列の `raw_data` をそのまま
       書き出す契約のため、復号済みを保持すると raw を解放できず常駐量が
       2 倍になる（raw を解放すると export の再エンコード経路が別途要る）。
       (c) `Graph` へのフィールド追加は構造体リテラルで構築している
       約 12 箇所（テスト・`export_nn`）の変更を伴う。initializer の
       実行ごとの複製の回避（`Arc` 共有）も同じ (a) の理由で
       `tensor-core` の `Tensor` 構築 API か `RawTensor` の表現を変える
       必要があり、本 P0 の範囲では失敗可能確保で是正する。
     - 数値不変: 置き換えたヘルパは確保方式だけを変え、書き込む値は
       `clone`／`decode_tensor`／旧 `encode_tensor`／`encode_to_vec` と
       同一。external data 由来の `Constant` 属性テンソル（4 dtype）と
       initializer を持つモデルの推論結果が同じ値を inline で持つモデルと
       bit 一致し、同じ `Graph` での 3 回の `run` で不変であること、export
       バイト列が inline モデルの export と同一であることを統合テストで
       固定した（8 節）。既存の PyTorch 実生成 fixture（`external_data_
       fixture_*`）も無変更で pass する。
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
  までの全経路と facade `OnnxModel::from_path` を対象とし、2 回目の是正
  〈5.〉で読み込み後の実行経路〈`interp::run`・`onnx::autograd`〉と
  export 経路〈`export::build_model_proto`・facade `OnnxModel::to_bytes`／
  `to_path`〉へ拡げた）:

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
  | `interp::compute_constant`（`decode_tensor` の `collect` ＋ `raw_to_value` の `data.clone()`。実行ごと一時 2 倍） | `collect`／clone | external 由来で巨大化しうる（実行経路。codex P0 2 回目の指摘箇所） | 5.: `decode_constant_tensor`（`try_decode_tensor` で失敗可能復号し `Tensor` へ move。一時 1 倍） |
  | `interp::run_impl` が実行ごとに initializer を env へ変換（`raw_to_value` の `data.clone()`） | clone | external 由来で巨大化しうる（実行ごと ＋1 倍） | 5.: `try_clone_slice`（複製自体は `Graph` 借用・`Vec` 所有の構造上残す。上記の不採用理由） |
  | `onnx::autograd` の `Constant` 腕（`decode_tensor` ＋ `raw_to_value` の clone） | `collect`／clone | 同上（学習実行経路） | 5.: `try_decode_tensor` ＋ move |
  | `onnx::autograd::BoundGraph::bind` の initializer 変換（`raw_to_value` の clone） | clone | external 由来で巨大化しうる（bind ごと 1 回） | 5.: `try_clone_slice` |
  | `export::build_model_proto` の `graph.nodes.clone()`（external 由来の `Constant` 属性テンソルの `raw_data`） | clone | external 由来で巨大化しうる（facade `to_bytes`／`to_path` から到達） | 5.: `try_clone_nodes` |
  | `export::encode_tensor` の `raw_data`（`Vec::with_capacity`／BOOL は `collect`） | 無条件確保 | 同上（initializer 1 件ごと） | 5.: `encode_le`（`try_alloc_vec`） |
  | `proto::encode_model`（`encode_to_vec`）の全体バイト列 | 無条件確保 | 同上（モデル全体） | 5.: facade は `export::try_encode_model` を使う（`encode_model` は既存シグネチャのまま残す） |
  | `tape.var(&t)`／`var_no_grad`（`autograd::bind`）・`run` 結果の `env.get(..).cloned()`・`BoundGraph::run` の `init` 複製・facade の `OnnxValue` 変換 | `Tensor` の clone | `Arc<Storage>` のポインタ複製のみ（要素列を複製しない） | 対象外 |
  | 演算カーネル（`ops::*`・`interp_ext`）の出力確保・`interp_device` のオペランド準備（`contiguous()` 等） | 無条件確保 | 一般の推論メモリ（op の結果テンソル・演算用の作業領域）。external data 固有ではなく、入力 feed の shape でも同様に巨大化しうる | 対象外（本件の範囲外。確保失敗時の挙動は推論エンジン全体の方針として扱う） |
  | `interp::compute_constant` の `value_floats`／`value_ints`（`attr.floats.clone()`／`attr.ints.clone()`）・`graph::decode_tensor` の typed data 分岐 | clone | `.onnx` 本体の長さで有界（external data にならない） | 対象外 |
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
  - 実行時（5. の是正後。いずれも失敗可能確保）: `interp::run` は実行
    ごとに initializer を env へ複製するため ＋`N`（initializer 分）、
    `Constant` 属性テンソルは実行時に失敗可能復号して move するため
    ＋`1 ×` 当該分（是正前は復号 ＋ clone で一時 `2 ×`）。確保失敗は
    `InterpError::AllocationFailed`（facade では `Io(OutOfMemory)`）。
  - export 時（5. の是正後）: `Graph`（およそ `N`）に加え、`ModelProto`
    （initializer の `raw_data` ＋ ノード列の複製でおよそ `N`）と encode
    結果（およそ `N`）で最大およそ `3N`。確保失敗は
    `ExportError::AllocationFailed`（facade では `Io(OutOfMemory)`）。
- **低メモリ環境での運用**: `onnx-interop` の `build_graph_with_external_
  data`／`resolve_external_data` を直接呼ぶ利用者は、`ExternalDataOptions
  { max_total_bytes, .. }` を利用可能メモリに合わせて下げて渡せる
  （既存オプション。上限既定値 64 GiB 自体は変更しない。疎ファイルで
  4 GiB の単一テンソルを宣言したモデルが下げた予算で確保前に拒否される
  ことを `sparse_file_huge_tensor_is_rejected_by_lowered_budget_before_
  allocation` で固定）。一方 facade `OnnxModel::from_path` は
  `ExternalDataOptions::default()` 固定のため、facade 利用者は
  `OnnxModel::from_path_with_limits`＋`OnnxExternalDataLimits` で予算を
  下げる（本 P0 では追加せず、イシュー #2360 へ切り出し済み・実装済み。
  `docs/facade-onnx-import-exposure-decision.md` §16）。この場合も確保失敗は abort ではなく
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
  5.（実行経路・export 経路）の回帰テスト:
  - 単体（全プラットフォーム）: `onnx::fallible_alloc::tests`（`usize::
    MAX`・`isize::MAX` 超の要求が確保前に `AllocFailure`、`try_clone_slice`
    の bit 一致と容量ちょうど、`try_decode_tensor` と `decode_tensor` の
    Ok／Err 一致〈4 dtype・typed data・空テンソル・長さ不一致・負の dim・
    乗算オーバーフロー・未知 dtype〉、`try_clone_nodes` と `clone` の
    一致）、`onnx::export::alloc_tests`（`encode_tensor` の `raw_data` が
    旧直列化と同一、`encode_le` の確保不能長が `AllocationFailed`、
    `try_encode_model` と `encode_model` のバイト列一致）、facade
    `map_graph_error_tests` の `interp_allocation_failed_maps_to_io_out_
    of_memory`・`export_allocation_failed_maps_to_io_out_of_memory`。
  - 統合（実確保失敗。Linux・64bit）: `tests/onnx_external_data.rs::
    runtime_allocation_failures_are_typed_errors_under_address_space_
    limit`。同じテストバイナリを子プロセスとして起動し（`runtime_
    allocation_failure_child`）、64 MiB の external テンソル（疎ファイル）
    を読み込んだ後に `setrlimit(RLIMIT_AS)` でアドレス空間の soft limit
    を「現在の `VmSize` ＋ テンソルの半分」へ下げ、テンソル 1 個分の
    追加確保を要する 4 シナリオ（`initializer`: `run` の initializer
    複製／`constant`: `run` の `Constant` 属性テンソル復号／`autograd`:
    `BoundGraph::bind`／`export`: `build_model_proto`）がいずれも
    `AllocationFailed`（テンソル名・要求バイト数つき）になることを
    検査する。`vm.overcommit_memory` の設定に依存しない。確保ヘルパを
    無条件確保（`Vec::with_capacity`）へ戻す変異で 4 シナリオとも子が
    SIGABRT（`memory allocation of 67108864 bytes failed`）で落ちることを
    確認済み。
  - 統合（読み込み段）: `sparse_file_unallocatable_constant_attribute_
    returns_allocation_failed`（4 TiB の疎ファイルで 4 TiB の `Constant`
    属性テンソルを宣言すると、属性テンソルの slot も `load` の区間
    バッファで `AllocationFailed` になる。**実行時の復号には到達しない**
    ため実行時経路の検証は上記の子プロセステストが担う。ゲートは既存の
    疎ファイルテストと同じ Linux・overcommit 0／2・64bit）。
  - 数値不変: `external_constant_attributes_run_bit_identical_to_inline_
    across_runs`（external 由来の `Constant` 属性テンソル 4 dtype〈NaN・
    `-0.0`・非正規化数・BOOL の非ゼロ値を含む〉と initializer の推論結果
    が inline モデルと bit 一致・3 回の `run` で不変・export バイト列が
    inline モデルの export と同一）。

### 4.6 Windows 向け経路解決（`win_contained_open`。イシュー #2349）

Windows の std にはディレクトリハンドル起点の相対オープン（`openat` 相当。
`NtCreateFile` の `RootDirectory`）が無く、`File::open` は常にフルパスの
`CreateFileW` になる（`FILE_FLAG_OPEN_REPARSE_POINT` が効くのは最終成分
だけで、途中の junction は辿ってしまう）。そのため unix 版（4 節・
`no_follow_open`）とは異なる構成で TOCTOU を閉じる
（`crates/onnx-interop/src/onnx/external_data.rs::win_contained_open`）:

- **祖先チェーンのハンドル保持**: `base_dir`（ボリュームルート
  `\\?\C:\` から）までの各ディレクトリ成分を開いたまま保持する
  （`BaseDirHandle { base, chain }`。`plan`・`load` を通して 1 つだけ
  保持する。unix のディレクトリ fd 単体保持とは異なり祖先全体を保持する
  構成になる）。対象を rename・削除するには DELETE アクセスで開く必要が
  あり、共有モード（`dwShareMode`）から `FILE_SHARE_DELETE` を意図的に
  外すことで保持中の成分は他プロセスから rename・削除できない。また
  NTFS では reparse point 化（`FSCTL_SET_REPARSE_POINT`）に対象
  ディレクトリが空であることを要求するため、保持中の次成分を含む
  ディレクトリは reparse point 化できない。
- **開いた直後の属性検査**: 各ハンドルを開いた直後に、そのハンドル自身
  （パスではない）に対する属性照会（`File::metadata` →
  `GetFileInformationByHandle` 相当）で reparse point 属性
  （`FILE_ATTRIBUTE_REPARSE_POINT`）を検査する。判定は
  `file_type().is_symlink()`（シンボリックリンク・マウントポイントの
  タグしか認識せず AppExecLink・cloud files 等を素通りさせる）ではなく
  属性ビットで行う。最終ファイルを開いた後は、保持中の全祖先ハンドル
  （`base_dir.chain` ＋ 本 location 専用に新規で開いた途中ディレクトリ）
  の属性を再取得する事後チェックも行う。
- **`base_dir` の受理範囲**: `canonicalize` の結果（`\\?\C:\...`）の
  先頭成分が `Prefix::VerbatimDisk` の場合のみ受け付ける。UNC
  （`\\server\share\...`）は SMB 越しの junction がサーバ側で評価され
  共有モードの意味論も異なるため、Volume GUID パス
  （`\\?\Volume{GUID}\...`）はフォルダにマウントされたボリュームを
  経由しうるため、いずれも `InvalidBaseDir` で拒否する。
- **定数**: Win32 SDK ヘッダ（`winnt.h`／`winbase.h`）の値を手書きする。
  Linux の `O_*` フラグ（CPU アーキテクチャごとに値が異なり、旧実装の
  手書き値誤りの原因になった。4 節参照）と異なり、Win32 API の定数は
  x86_64／aarch64 で ABI が固定され値が変わらないため、`libc` 相当の
  crate を追加せず std の `OpenOptionsExt` へ渡すだけで済み、この部分は
  `unsafe` を要しない（下記「FileKey・FileSnapshot」節・「最終ハンドルの
  実所在検証」〈5 節「Windows 版の残存リスク」項目 5〉は std が未安定化の
  API に依存するため kernel32.dll への手書き `extern "system"` 宣言を使い、
  `unsafe` を FFI 境界〈2 関数の FFI 呼び出しと `file_identity` の
  `assume_init` の計 3 つの `unsafe` 式のみ〉に限定する。事後監査は 10 節）。
- **FileKey・FileSnapshot（2026-09-28 codex-review 是正・PR #2351 で更新）**:
  dev/ino 相当（`file_index`）・volume serial number
  （`volume_serial_number`）・ctime 相当（`change_time`）は
  `std::os::windows::fs::MetadataExt` が 1.98.1 時点で安定化していない
  （`windows_by_handle`／`windows_change_time` の nightly-only feature。
  `file_attributes`／`creation_time`／`last_access_time`／
  `last_write_time`／`file_size` は安定）。当初はこれを理由に `FileKey`
  をパスベースにしていたが、ハードリンク・NTFS 8.3 短縮名等の別名パスで
  同一ファイルを参照する `location` が異なるキーへ分散し
  `OverlappingRegion` 検証をすり抜ける欠陥だったため（codex-review 指摘
  `PRRT_kwDOTuUCJc6my2Ie`）、kernel32.dll の `GetFileInformationByHandle`
  （winbase.h）への手書き `extern "system"` 宣言（`unsafe` を FFI 境界に
  限定。第三のクレート追加は不要——kernel32.dll は Windows の全プロセスが
  常にリンクする基盤 DLL）で直接呼び、`(dwVolumeSerialNumber,
  nFileIndex)` を実体識別子として使うよう是正した
  （`win_contained_open::file_identity`）。`FileSnapshot` は引き続き
  `file_attributes`／`creation_time`／`last_write_time` を使う
  （`change_time` 未安定化のため。unix の ctime ほど強い変更検知では
  ない）。
- **location の字句検査（`windows_component_reject_reason`。
  `cfg(any(windows, test))`。Linux でも `cfg(test)` ビルドに含まれ単体
  テスト可能）**: 代替データストリーム（`name:stream`）・予約デバイス名
  （`CON`／`PRN`／`AUX`／`NUL`／`COM0`〜`COM9`〈上付き数字含む〉／
  `LPT0`〜`LPT9`〈同上〉／`CONIN$`／`CONOUT$`。大文字小文字を区別せず
  拡張子・末尾の空白/ドットを除いた基底名で判定）・禁止文字
  （`<>"|?*` ・制御文字）・末尾のドット/空白を拒否する。配線は
  `validate_location_string` 内の `cfg(windows)` 限定で、unix の受理
  範囲（`a:b` 等）は変えない。
- **残存リスク**: TOCTOU の根拠・受容する残存リスクは 5 節「Windows 版の
  残存リスク」に記録する。

## 5. 残るリスク（受容済み）

- **確保成功後のページ実コミット時の OOM（2026-09-28・PR #2348 codex P0
  是正に伴い明記。実行経路・export 経路〈4.3 節 5.〉にも同じく適用）**:
  失敗可能確保（4.3 節）が型付きエラーにできるのは、
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
  - **Windows**: 4.6 節の封じ込めオープン（`win_contained_open`）で
    対応済み（イシュー #2349）。**それ以外**（wasm32 等）は上記いずれの
    安全な経路解決手段も持たないため、`resolve_and_open` は常に
    `UnsupportedPlatformForSecureResolve` で拒否する（fail-closed の
    まま変更なし）。
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

### Windows 版の残存リスク（2026-09-28・イシュー #2349）

`win_contained_open`（4.6 節）の TOCTOU 根拠:

1. **名前と実体の対応の固定**: 対象オブジェクトの名前を変えたり削除
   したりするには、そのオブジェクト自体を DELETE アクセスで開く必要が
   ある。ボリュームルートから最終ファイルまで全成分を DELETE 共有なしで
   保持しているので、どの成分も rename・削除・差し替えができない。
2. **その場での junction 化の防止**: NTFS ではディレクトリに reparse
   point を設定するにはそのディレクトリが空でなければならない（MS
   Learn "Reparse Points"）。保持中の途中ディレクトリは次の成分
   （同じく保持中で消せない）を必ず含むので空にできず、junction 化
   できない。
3. **最終成分**: `FILE_FLAG_OPEN_REPARSE_POINT` で開き、開いたハンドルの
   属性で reparse point でないことと通常ファイルであることを確かめる。
   `FILE_FLAG_BACKUP_SEMANTICS` も最終成分に付与する（実装時判明の必須
   追加。4.6 節の定数コメント参照）: `location` がディレクトリ・
   junction を指す場合、この flag が無いと `CreateFileW` は
   `ERROR_ACCESS_DENIED` で open 自体に失敗し、「開いてから属性で分類
   する」という契約そのものが機能しない（Rust std の `File::open` が
   ディレクトリに対し os error 5 を返す既知の挙動と同根）。通常ファイル
   の open には副作用が無いため無条件で付与する。
4. **読み込みの窓**: 読み込みハンドルは書き込み共有を拒否している。
   したがって、既存の writer ハンドルがあれば open 自体が失敗し、保持
   している間は新しい writer も開けない（unix より強い保証）。
5. **最終ハンドルの実所在検証（2026-09-28 codex-review 是正・PR #2351 で
   追加・同 PR のレビューで 2 度是正）**: 1.〜4. だけでは「held に積んで
   いるディレクトリであっても、共有モードが許す書き込みアクセスで別
   ハンドルから同一オブジェクトへ reparse タグを立てる（削除・rename を
   伴わないため 1. の防御が及ばない）→ 以後のフルパス文字列解決（`held`
   のハンドル経由ではなく毎回ファイルシステム名前空間を再解決する）が
   その reparse point を中間成分として追跡してしまう → 事後チェック
   （2.）が走る前に reparse タグを外して元へ戻す」という flip-and-revert
   （旧 (a) の残存リスク）を検出できなかった（codex-review 指摘
   `PRRT_kwDOTuUCJc6my2IX`・`PRRT_kwDOTuUCJc6mzcKZ`）。最終ファイルを
   開いた直後に、`GetFinalPathNameByHandleW`（winbase.h。std 未対応の
   ため kernel32.dll への手書き `extern "system"` 宣言で直接呼ぶ）で
   「そのハンドルが実際に指しているオブジェクトの所在」を取得する
   （`win_contained_open::final_real_path`）。初版はこれを `base_dir`
   配下・想定した**深さ（成分数）だけ**で判定していたが、`base_dir`
   配下の「同じ深さの別ディレクトリ」へ着地した場合（例: 成分 `A` を
   同じ深さの別ディレクトリ `B` への junction へ一時的に差し替えて最終
   ファイルを開かせ、事後チェックが走る前に `A` を元へ戻す）を見逃す
   欠陥が残っていた（P0 是正・codex-review 指摘
   `PRRT_kwDOTuUCJc6m0J-L`・PR #2351）。是正版
   （`win_contained_open::verify_final_path_within_base_dir`）は、深さだけ
   でなく `held`（本 location 専用に保持中の各祖先ディレクトリハンドル）
   自身の実所在（同じく `final_real_path` によるハンドル起点の逆引き）を
   最終ファイルの実所在と成分単位で対応づけ、加えてボリューム識別子
   （`file_identity` の `volume_serial_number`）を `base_dir` の
   ボリュームルートハンドルと突き合わせる。いずれもハンドルが指す
   オブジェクトそのものに基づく逆引きであり、経路文字列の再解決では
   ないため、reparse point を事後に元へ戻しても偽装できない。
   本 5. の事後監査と閉鎖状況の結論（pass 1・pass 2 の両方で検証が走る
   こと、検出型である点の限界）は 10 節（イシュー #2392）を正とする。

**残るリスク**:

- (a) 「最深ディレクトリの flip-and-revert 競合」は上記 5. の初版
  （深さのみの判定）で一旦閉じたが、`base_dir` 配下の同じ深さの別
  ディレクトリへの着地を見逃す欠陥が残っており、held ハンドルとの対応
  づけ・ボリューム識別子の突き合わせで最終的に閉じた
  （2026-09-28・PR #2351。codex-review 指摘 `PRRT_kwDOTuUCJc6m0J-L`）。
  閉鎖は検出型（事後検証）であり、`base_dir` 外のオブジェクトを 1 バイト
  も読まないことは保証するが、拒否されるまでの間の open そのもの
  （メタデータ取得・短時間の共有ロック保持）と、`is_file`／reparse の
  分類が実所在検証より先に走ることによるエラー種別の差（`base_dir`
  外の存在・種別の限定的なオラクル）は残る受容リスクである（10.3 節）。
- (b) **snapshot の弱さ**: `FileSnapshot`（4.6 節）の時刻フィールドは
  `SetFileTime` で利用者が書き換え可能なため、pass 1・pass 2 間の
  差し替え検出は unix の ctime より弱い。ただし封じ込め（`base_dir`
  配下・reparse なし・区間は有界）は snapshot に依存しない（pass 2 は
  同じ封じ込め手順で開き直し、書き込み共有を拒否した状態で読むため）。
  `ChangeTime` への切替も利用者設定が可能なため根本改善にならない。
  改善候補（USN による変更検知）は 10.4 節に起票候補として記録した。
- (c) 「実体同一性の欠如（`FileKey` パスベース）」は上記「FileKey・
  FileSnapshot」節の是正（`(dwVolumeSerialNumber, nFileIndex)` への
  切替）で閉じた（2026-09-28・PR #2351）。
- (d) NTFS 以外（ReFS・exFAT 等）で「reparse point 化には空ディレクトリ
  が必要」という規則が成り立つかは未確認（Windows 実機での確認事項）。
  2026-09-29 再分類（10.4 節）: 実所在検証（5. 項）はこの規則に依存
  しないため、flip-and-revert の検出に関する影響度は下がる。残る論点
  （ReFS の 64 bit `nFileIndex` の一意性・共有モードの意味論）は 10.4
  節の表に従い、FFI 改善候補と #2393 の実機確認項目に分ける。
  **2026-10-03 実測・#2393（実測で確認）**: Windows Server 2022 の GCE VM
  で NTFS・ReFS・exFAT を実測した（出典 `docs/perf/logs/windows-onnx-
  external-data-2393/README.md`）。`FSCTL_SET_REPARSE_POINT` は NTFS・ReFS
  とも空ディレクトリで成功・非空で 145（`ERROR_DIR_NOT_EMPTY`）となり、
  「非空ディレクトリは reparse 化できない」は **ReFS でも成立**した。
  exFAT は空・非空とも 1（`ERROR_INVALID_FUNCTION`＝reparse 非対応）で、
  junction・symlink・hard link のいずれも作成できない（このため exFAT 上の
  `onnx_external_data` の失敗 3 件と `--ignored` の失敗 2 件は fixture 作成
  失敗であり、封じ込め判定の失敗ではない）。flip-and-revert（途中成分を
  同じ深さの別ディレクトリへの junction へ入れ替える攻撃）は NTFS 475
  サイクル・ReFS 764 サイクルで `containment_breach_B=0`・`other_invalid=0`
  （load は全件型付きエラー）。ReFS の 64 bit `nFileIndex` の一意性は
  実測していない（10.4 節の表のまま）。
- **可用性への副作用**: 読み込み中は祖先ディレクトリを rename・削除
  しようとした他プロセスが共有違反で失敗する。他プロセスが書き込み
  ハンドルで開いているモデルデータは読めない。OneDrive のプレースホルダ
  や dedup ファイル（reparse point）も拒否される。いずれも fail-closed
  側の制約として記録する。
  **2026-10-03 実測・#2393（訂正を含む）**: 読み込み中の祖先 rename の
  失敗コードは実測では**主に `ERROR_ACCESS_DENIED`(5)** だった。
  `ERROR_SHARING_VIOLATION`(32) は 8 MB モデルの開始後約 0.4〜0.8 ms
  のみで観測した（NTFS `delay_us=395`: 32×10。ReFS `delay_us=387`: 32×9。
  祖先チェーンを開いている段階と解釈されるが、`from_path` 内部の段階の
  時刻は測っていない。256 MB は最初の遅延点が約 12.4〜12.9 ms のため
  それより早い時点は標本に入らず）。rename が失敗した遅延点は 8 MB で
  開始後約 0.4〜5.6 ms〈NTFS〉・約 0.4〜5.1 ms〈ReFS〉、256 MB〈`from_path`
  中央値 約 0.24 s〉で約 13〜129 ms〈NTFS〉・約 12〜124 ms〈ReFS〉で、
  それより後は `from_path` 実行中でも rename は成功し、load は正しい値で
  成功した（external data の読み出し完了後と解釈されるが、読み出し完了の
  時刻は測っていない）。
  全条件で `invalid_values=0`（不正値ゼロ。load は正しい値か型付きエラー）。
  **FILE_TRAVERSE 拒否 ACL**: 非管理者ユーザー（`SeChangeNotifyPrivilege`
  有効）で、途中ディレクトリに `(DENY)(S,X)` の ACE を付けると `from_path`
  は `Io:PermissionDenied` で fail-closed になった。同条件の通常の `copy`
  は成功する（traverse 回避特権）。封じ込めは各祖先を明示的に開くため、
  通常 API では読めるパスでも traverse 拒否 ACE があれば拒否する。これも
  可用性への副作用として記録する。

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
  で書き出す契約は不変（export 用バイト列の確保は 4.3 節 5. で失敗可能
  確保へ是正済み）。
- `checksum`（SHA-1）の検証。
- `max_external_files` の既定値（4096）の承認（6 節）。
- （イシュー #2360 へ切り出し済み・実装済み・2026-09-29。
  `OnnxModel::from_path_with_limits`／`OnnxExternalDataLimits`）
  （2026-09-28・PR #2348 codex P0 是正に伴う起票候補。4.3 節）facade
  `OnnxModel::from_path` から `max_total_bytes` を下げる公開手段（現状は
  `ExternalDataOptions::default()` 固定。公開 API 面の追加にはユーザー
  承認が要る）。
- （解消済み・2026-09-28・PR #2348 codex P0 是正 2 回目）実行経路の確保
  （`interp::run` の initializer 複製、`interp::compute_constant`／
  `onnx::autograd` の `Constant` 属性テンソル復号）は 4.3 節 5. で
  失敗可能確保へ是正した（export 経路も同時に是正）。`Graph` を借用の
  まま複数回 `run` する契約の下で initializer の実行ごとの複製自体を
  無くす（`Arc` 共有）には `tensor-core` の `Tensor` 構築 API か
  `RawTensor` の表現の変更が要るため、必要になった時点で別途扱う
  （現状は失敗可能確保で abort しない）。
- **（解消済み・2026-09-29）facade `OnnxModel::from_path` の Windows
  対応**（2026-09-28・イシュー #2349 で起票候補として記録）:
  `onnx-interop::onnx::external_data` 自体は Windows へ対応済み（4.6
  節）だが、facade（`fandhe-ai`）は `backend-cuda` への無条件依存で
  Windows ではビルドできなかった（`crates/backend-cuda/src/nvrtc.rs` の
  `compile_error!`。#509／PR #677）。方針決定は #2389、実装は #2390（非
  unix で NVRTC ディスクキャッシュを無効化。いずれも close 済み）へ切り
  出し済み。facade の Windows クロス clippy の CI 化は #2391 へ切り出し済み・解消
  （2026-09-29。`build` ジョブと Makefile `check-cross-windows-interop` に
  facade 行〈`--lib --tests`〉を追加。実機実行は #2393）。
- **（解消済み・2026-09-28・PR #2351。事後監査は 2026-09-29・イシュー
  #2392 の 10 節）Windows の残存 TOCTOU 経路の閉鎖**: 5 節「Windows 版
  の残存リスク」(a) の flip-and-revert 競合は、`GetFinalPathNameByHandleW`
  による実所在検証（`win_contained_open::final_real_path`・
  `verify_final_path_within_base_dir`）を PR #2351 で導入して閉じた。
  kernel32.dll への手書き `extern "system"` 宣言で実現し、依存の追加
  （`windows-sys` 等）はしていない。当初ここに記した「`windows-sys`
  依存の追加の可能性」は不要になった。`NtCreateFile(RootDirectory)`
  による予防型対策（W2）は 10.3 節の結論により不要とする。
- **本節に残る起票候補（2026-09-29・イシュー #2392。ユーザー承認待ち・
  起票していない）**: 10.4 節の表を正とする。(1) `extern` ブロックへの
  `#[link(name = "kernel32")]` の明示（コード修正・P2。実機リンクは
  #2393 で確認）、(2) USN（`FSCTL_READ_FILE_USN_DATA`）による変更検知、
  (3) `GetFileInformationByHandleEx(FileIdInfo)`（128 bit ID）への切替。

自動運転中はユーザー承認を取れないため Issue は起票せず、本節（「本節に
残る起票候補」）と PR 本文に起票候補として記録する。

## 8. テスト・実測

- 合成入力の網羅テスト: `crates/onnx-interop/tests/onnx_external_data.rs`
  （unix で 58 テスト〈うち Linux 限定 4 件〉＋非 unix 契約テスト 2 件。正常系〈FLOAT/INT64/BOOL/FLOAT16・offset 省略・length 省略・
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
  with_names`。上記 58 テストに含む）と、全プラットフォームで実行する
  単体テスト `external_data.rs::overlap_tests`（7 テスト）。メモリ確保の
  失敗可能化（4.3 節）の回帰テストは全プラットフォームで実行する
  `external_data.rs::alloc_tests`（6 テスト）・統合テスト 4 件（上記 58
  テストに含む）・facade `interop::onnx::map_graph_error_tests`（5 テスト。
  うち 2 件は 4.3 節 5. の実行時・export 時の写像）。実行経路・export
  経路の失敗可能化（4.3 節 5.）の回帰テストは、全プラットフォームで
  実行する `onnx::fallible_alloc::tests`（4 テスト）・`onnx::export::
  alloc_tests`（3 テスト）と統合テスト 4 件（上記 58 テストに含む。
  `external_constant_attributes_run_bit_identical_to_inline_across_runs`
  〈unix〉・`sparse_file_unallocatable_constant_attribute_returns_
  allocation_failed`・`runtime_allocation_failures_are_typed_errors_
  under_address_space_limit` とその子プロセス本体
  `runtime_allocation_failure_child`〈いずれも Linux・64bit〉。内容は
  4.3 節の回帰テスト欄）。
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
- **Windows 対応のテスト（イシュー #2349・2026-09-28）**: 正常系
  （FLOAT/INT64/BOOL/FLOAT16・offset 省略・length 省略・隣接区間・
  数値検証・範囲・上限・重複区間・`owned_build_matches_resolve_then_
  build_graph` 等）は `cfg(unix)` から `cfg(any(unix, windows))` へ広げ、
  unix・Windows 共通で実行する。symlink・FIFO・fd 数計測・
  hard link は `cfg(unix)` のまま残す（当初の除外理由「`FileKey` が
  パスベースのため Windows では別名扱いになる」は、`FileKey` を
  `(dwVolumeSerialNumber, nFileIndex)` へ切り替えた PR #2351〈5 節
  (c)〉で失効している。`overlapping_regions_via_hard_link_are_rejected`
  の Windows 実行は未充足のテスト網羅として #2393〈実機検証〉へ申し
  送る。イシュー #2392 ではテストを変更しない。**2026-10-03 追記: #2485 で
  cfg を `any(unix, windows)` へ広げた。Windows 実機〈NTFS〉での再確認は
  人手**）。Windows 固有の新規テスト
  （`onnx_external_data.rs`）: junction が途中成分・最終成分（ディレクトリ
  への junction）にある場合の `ReparsePoint` 拒否、字句検査（ADS・予約
  デバイス名・禁止文字・末尾ドット/空白。`windows_lexical_rejections_
  via_full_pipeline`）、他プロセスの書き込みハンドルによる共有違反拒否
  （`write_handle_causes_sharing_violation_rejection`）。シンボリックリンク
  拒否（最終・途中成分）・UNC `base_dir` の拒否は特権・管理共有の設定に
  依存するため `#[ignore = "..."]` で分離し通常 CI では走らない。
  「読み込み中の祖先ディレクトリの rename が共有違反で失敗すること」は
  ライブラリ内部のハンドル保持へ同一プロセスから割り込む手段が無いため
  本 PR では自動テスト化していない（Windows 実機での手動確認項目として
  下記の申し送りに含める）。字句検査の純関数
  `windows_component_reject_reason` の単体テスト（`external_data.rs::
  windows_lexical_tests`。5 テスト）は Linux の `cfg(test)` ビルドでも
  実行される（`cfg(any(windows, test))`）。`onnx_interp_pytorch_cnn_
  fixture.rs` の external data fixture テスト（R1）も `cfg(any(unix,
  windows))` へ広げた。**Windows 実機でのテスト実行はこの PR の Linux 上
  の自動運転では行えない**ため、実行コマンド・確認項目（`FILE_TRAVERSE`
  の ACL 可否、祖先の rename が共有違反で失敗すること、junction のテスト、
  `FSCTL_SET_REPARSE_POINT` が非空ディレクトリで失敗すること）を PR 本文
  に申し送りとして記録し、Issue #2349 は `Refs #2349` で紐付けて open の
  まま残した（当時）。2026-09-29 時点では #2349 は close 済みで、R1'
  （facade からの到達性）は #2389〜#2391、R4（Windows 実機結果）は
  #2393 へ付け替えた。
  **2026-10-03 実測・#2393（R4 の実機結果。出典 `docs/perf/logs/windows-
  onnx-external-data-2393/README.md`）**: Windows Server 2022（GCE VM。
  物理機ではない）・Rust 1.99.0 で次を確認した。
  - 実行結果（NTFS）: `onnx_external_data` 50 passed・3 ignored、
    `--ignored`（symlink 最終・途中成分、UNC base_dir。管理者・
    `\\localhost\C$` 到達可）3 passed、`onnx_interp_pytorch_cnn_fixture`
    44 passed、`--lib` 344 passed、onnx-interop 全体 all ok。facade
    `interop_onnx_external_data` 2 passed（R1'。facade `OnnxModel::from_path`
    で external data を読めた）・`interop_onnx_internal_parity` 16 passed。
    ReFS は `onnx_external_data` 51 passed・3 ignored・`--ignored` 3 passed。
    exFAT は fixture 作成失敗で 3 件（junction 2・hard link 1）＋
    `--ignored` 2 件が失敗した（上記 5 節 (d) の実測を参照。封じ込め判定の
    失敗ではない）。
  - **読み込み中の祖先 rename（上記の「共有違反で失敗」の訂正）**:
    実測では主に `ERROR_ACCESS_DENIED`(5) で失敗し、共有違反(32) は
    開始直後の約 1 ms 以内のみだった。5 節「可用性への副作用」に
    数値を記録した。**これは同一プロセスから内部ハンドルへ割り込む手段が
    ないため自動テスト化できず、別スレッドの rename 攻撃を行う使い捨て
    ハーネス（同 README の `rename_race`）で確認した**。
  - **hard link テスト網羅**: VM 上でのみ
    `overlapping_regions_via_hard_link_are_rejected` の cfg を
    `any(unix, windows)` へ変えて実行し、NTFS・ReFS で pass
    （リポジトリのテストは変更していない）。cfg を広げる変更は起票候補
    （README の起票候補 4）。**2026-10-03 追記: #2485 でリポジトリのテストの
    cfg を `any(unix, windows)` へ広げた。Windows 実機での再確認は人手**。ReFS・exFAT の
    `onnx_external_data` は、cfg 変更を戻した後 mtime が古いまま戻った
    ため cargo が再ビルドせず、hard link テストを含むバイナリ（54 件）で
    実行されている（NTFS の 53 件との差）。
  - facade の `--no-fail-fast` 全体では external data と無関係の 9 件
    （`fs_guard`／`model` の単体 5・`tests/model_registry.rs` 3・
    `tests/api_surface.rs` 1）が Windows で失敗した。非 Linux/macOS で設計
    どおり `Unsupported` を返す実装にテストが cfg 分離されていないことが
    主因で、起票候補（README の起票候補 2）。

## 9. OWASP Top 10 観点

4 節の 2 パス設計・5 節の残るリスクを参照。要点は `security.md` A03
（外部フォーマットのパース検証を長さ・形状の検証が先行する）・A04
（確保の前に合計上限を検査する）・A08（`checksum` を黙って無視しない・
no-silent-skip 契約）。`unsafe` は `no_follow_open`（`cfg(unix)` 限定）の
`libc::openat`／`libc::fstatat`／`libc::syscall(SYS_openat2, ...)` FFI
呼び出しに限定して使用する（2026-09-28・#2347 是正で `libc =0.2.189`
〈`.claude/rules/deps-policy.md`「OS FFI」区分〉を導入。呼び出し箇所には
`coding-rust.md` 準拠の `// SAFETY:` コメントを付与済み。5 節参照）。
**Windows 版（`win_contained_open`。イシュー #2349）は `unsafe` を
kernel32.dll への手書き `extern "system"` 宣言（`GetFileInformationByHandle`・
`GetFinalPathNameByHandleW`）の FFI 呼び出し 2 か所と `file_identity` の
`MaybeUninit::assume_init` 1 か所（計 3 つの `unsafe` 式）に限定する。
security-auditor 相当の事後監査は 10 節（イシュー #2392）で記録済み**（依存の
追加〈`windows-sys` 等〉はしない。4.6 節「FileKey・FileSnapshot」・
「最終ハンドルの実所在検証」参照）。ディレクトリ祖先チェーンの走査・
reparse point 属性検査そのものは std の
`OpenOptionsExt::{access_mode, share_mode, custom_flags}` と
`MetadataExt` のみで完結する安全な API で構成するが、`file_index`／
`volume_serial_number`（`FileKey` の実体識別）・実所在の逆引き
（TOCTOU 是正の flip-and-revert 対策。PR #2351 codex-review 指摘
`PRRT_kwDOTuUCJc6my2IX`・`PRRT_kwDOTuUCJc6mzcKZ`・`PRRT_kwDOTuUCJc6m0J-L`）
は std が 1.98.1 時点で未安定化のため、この 2 関数のみ `unsafe` な FFI
呼び出しで直接叩く（`.claude/rules/coding-rust.md`「`unsafe` は FFI
境界等の必要最小限に留め、理由をコメントで明記」に準拠。呼び出し箇所には
`// SAFETY:` コメントを付与済み）。A01（パストラバーサル）は 4.6 節の
祖先ハンドル保持・reparse point 属性検査・`base_dir` の `VerbatimDisk`
限定・最終ハンドルの実所在検証（`held` 各エントリとの対応づけ）で対処し、
字句検査（ADS・予約デバイス名・禁止文字）は A03 の一部として Windows
固有の非信頼入力検証に位置づける。

## 10. 既存 Windows FFI の事後監査と flip-and-revert 閉鎖状況の再評価（2026-09-29・イシュー #2392）

PR #2351 のレビュー是正で main に入った `win_contained_open`
（`crates/onnx-interop/src/onnx/external_data.rs`）の `unsafe` FFI は、
`.claude/rules/security.md`「unsafe」節が求める監査記録を欠いていた。本節は
その事後監査と、flip-and-revert の閉鎖状況の結論を記録する。参照は行番号
ではなくシンボル名で書く。

### 10.1 監査の範囲・方法

- **対象シンボル**: `win_contained_open` 内の `unsafe extern "system"` ブロック
  （`GetFileInformationByHandle`・`GetFinalPathNameByHandleW`）、
  `ByHandleFileInformation`・`RawFiletime`、`file_identity`、
  `final_real_path`、`verify_final_path_within_base_dir`、
  `open_base_dir_handle`、`resolve_and_open`、および上位の `cfg(windows)` 版
  `resolve_and_open`・`file_key_for`。
- **方法**: 上記のソース精読と MS Learn の戻り値契約との突き合わせ。
  `make check-cross-windows-interop`（`x86_64-pc-windows-msvc` への
  onnx-interop クロス clippy）は 2026-09-29 に警告なしで成功し、宣言が
  型検査を通ることを確認した。
- **実施体制の注記**: 本監査は自動運転の実装エージェントが security-auditor
  の観点（`.claude/agents`）に沿って行った。独立した security-auditor
  サブエージェントでの再監査は、ユーザーが望む場合に別途実施する。
- **限界**: クロス clippy はリンクを行わないため、シンボル解決と実行時挙動は
  Linux からは未検証である（#2393 の実機確認項目）。
  **2026-10-03 実測・#2393**: Windows Server 2022（x86_64-pc-windows-msvc・
  Rust 1.99.0・MSVC 14.44.35207）の実機ビルドで `cargo build -p
  fandhe-ai-onnx-interop --tests --locked`・`cargo build -p fandhe-ai
  --tests --locked` とも MSVC リンクまで成功し、実行時挙動も NTFS・ReFS で
  確認した（出典 `docs/perf/logs/windows-onnx-external-data-2393/README.md`）。
  ここに書いた限界は実機で解消した。

### 10.2 指摘の表

| 重要度 | シンボル | 内容 | 処置 |
|--------|---------|------|------|
| 指摘なし（P0／P1 なし） | `ByHandleFileInformation`・`RawFiletime` | `#[repr(C)]`・フィールド順と型幅（`u32`／FILETIME の 32 bit ペア）が winbase.h の `BY_HANDLE_FILE_INFORMATION` と一致 | なし |
| 指摘なし | `extern "system"` 宣言 | `HANDLE=*mut c_void`・`BOOL=i32`・`DWORD=u32`・`LPWSTR=*mut u16`・呼び出し規約が winbase.h と一致 | なし |
| P2 | `unsafe extern "system"` ブロック | `#[link(name = "kernel32")]` が無く、std が kernel32 をリンクすることに暗黙に依存する。クロス clippy はリンクしないため未検証 | #2393 で実機ビルド・リンクを確認。コード修正（`#[link]` 明示）は起票候補（7 節）。**2026-10-03 実測・#2393**: `#[link(name = "kernel32")]` の明示なしでも MSVC リンクは成功し、kernel32 の手書き `extern "system"` 宣言は解決された（実害なし）。明示化は起票候補のまま |
| 指摘なし | `final_real_path` | 成功時は終端 NUL を除く文字数、不足時は NUL 込みの必要文字数を返す契約に対し、成功判定 `n < buf.len()`・不足時 `resize(n)`・上限 8 回後の `Unsupported`（fail-closed）が整合。`buf.len() as u32` は 32K 文字規模のため切り詰めは起きない | なし |
| 指摘なし | `file_identity` | 全フィールドが整数で零値が有効なビットパターンのため、`zeroed` → 成功時のみ `assume_init` は健全。失敗時は構造体を使わない | なし |
| P3（文言） | 4.6・9 節・モジュール doc | `unsafe` 式は FFI 呼び出し 2 か所と `assume_init` 1 か所の計 3 つ。「2 か所」は不正確 | 4.6・9 節は本 PR で是正済み。ソースのコメントは範囲外（コード変更なし） |
| 指摘なし | `verify_final_path_within_base_dir` のパス比較 | `base_dir.base`（`canonicalize`。std 内部が同じ API を同じ既定フラグで呼ぶ）と `real_path`（`FILE_NAME_NORMALIZED \| VOLUME_NAME_DOS`）は同じ表記系（`\\?\` 接頭辞・on-disk の大小文字・長い名前）で、`strip_prefix` の成分比較が健全。食い違いは拒否側（可用性の低下であり安全性の低下ではない）に倒れる。`VerbatimDisk` 以外は `open_base_dir_handle` が拒否 | なし |
| 指摘なし | ボリューム識別子の照合 | `volume_serial_number` は 32 bit で利用者が変更し得るため単独では根拠にならず、パス接頭辞検査の補助として位置づけられている。途中のマウントポイントは reparse 属性の検査で先に拒否される | なし |

### 10.3 flip-and-revert の閉鎖状況

**検証が走る経路**: パス 1 の `plan` とパス 2 の `load` は、どちらも上位の
`resolve_and_open`（`cfg(windows)` 版）から `win_contained_open::resolve_and_open`
へ入る。後者は最終ファイルを開くたびに無条件で `final_real_path` と
`verify_final_path_within_base_dir` を実行するため、両パスで検証が走る。

**ケース別の確認**:

- (i) `parts.len() == 1`（`base_dir` 直下）: `held` は空で、防御は
  `strip_prefix(&base_dir.base)` と深さ 1 の検査になる。これは
  `base_dir.chain` の各ハンドルを DELETE 共有なしで保持し（rename・削除・
  同名の別オブジェクトによる占有ができない）、かつ各祖先が次の成分を含み
  空でないため reparse 化もできないことに依存し、成立する。
- (ii) 途中成分の一時的 junction 化: `held[i]` 自身の実所在と最終ファイルの
  実所在の接頭辞・深さの突き合わせで検出される。
- (iii) 最深の `held` の一時的 junction 化: 同じ突き合わせで検出される。

いずれも検出はハンドルからの逆引き（`GetFinalPathNameByHandleW`）に基づき、
経路文字列の再解決ではないため、reparse タグを事後に元へ戻しても偽装できない。
なお NTFS の「reparse 化には空ディレクトリが必要」という規則と、保持中の
成分の削除不能性により、flip 自体も二重に起こりにくい（検出はこれに依存しない）。

**「閉じている」の定義**: 対策は検出型（事後検証）である。したがって
「`base_dir` 外のデータを 1 バイトも読まない（読み込みは検証通過後のハンドル
に限る）」という意味で閉じている。拒否されるまでの間の open そのもの
（メタデータ取得・短時間の共有ロック保持）と、`is_file`／reparse の分類が
実所在検証より先に走ることによるエラー種別の差（`base_dir` 外の存在・種別の
限定的なオラクル）は、残存する副作用として受容する（5 節 (a)）。

**結論**: 検証通過後に `base_dir` 外のバイトを読める経路は見つからなかった。
よって flip-and-revert は閉じていると判断し、予防型対策（W2。
`NtCreateFile(RootDirectory)`）は不要とする（起票しない）。

### 10.4 残存リスク (b)・(d) の再棚卸し

| 項目 | 現状 | FFI で改善できるか | 起票候補または確認項目 |
|------|------|-------------------|----------------------|
| (b) snapshot の弱さ | `FileSnapshot` の時刻は `SetFileTime` で書き換え可能。`ChangeTime` も `SetFileInformationByHandle` で設定できるため切替は根本改善にならない。封じ込めは snapshot に依存しない | USN（`FSCTL_READ_FILE_USN_DATA`。利用者が任意値にできない）による変更検知で改善できる（`unsafe` は増えるが依存追加は不要） | 起票候補（ユーザー承認待ち） |
| (d) NTFS 以外: flip の検出 | 実所在検証は NTFS の空ディレクトリ規則に依存しないため、影響度は下がる | 不要 | なし |
| (d) ReFS の `nFileIndex` | 64 bit の `nFileIndex` は ReFS で一意とは限らず、`FileKey` の実体同一性に関わる | `GetFileInformationByHandleEx(FileIdInfo)`（128 bit ID）へ切替可能 | 起票候補（ユーザー承認待ち） |
| (d) ReFS・exFAT の意味論 | reparse・共有モードの意味論は未確認 | 不可（実機依存） | #2393 の実機確認項目 |
| リンク解決 | `#[link(name = "kernel32")]` の明示（10.2 の P2） | コード修正 | #2393 で確認、修正は起票候補 |
| テスト網羅 | `overlapping_regions_via_hard_link_are_rejected` は `cfg(unix)` のまま | テスト変更 | #2393 へ申し送り |

**2026-10-03 実測・#2393（上の表の #2393 行への結果。出典 `docs/perf/logs/windows-onnx-external-data-2393/README.md`）**:

| 項目 | 実測結果 |
|------|---------|
| (d) ReFS・exFAT の意味論 | ReFS: NTFS と同じ結果（非空ディレクトリへの reparse 設定は 145・`onnx_external_data` 51 passed・flip-and-revert 764 サイクルで breach 0）。exFAT: reparse 設定が空・非空とも 1（`ERROR_INVALID_FUNCTION`）で reparse point 自体を持たず、junction・symlink・hard link は作成できない（テストの fixture 作成失敗であり封じ込め判定の失敗ではない）。**ReFS の `nFileIndex` 一意性は実測していない（行はそのまま）** |
| リンク解決 | `#[link]` 明示なしでも実機リンク成功（上の P2 行）。実害なし・明示化は起票候補のまま |
| テスト網羅 | VM 上でのみ cfg を `any(unix, windows)` に変えて NTFS・ReFS で pass。リポジトリのテストは未変更で、cfg 拡張は起票候補（未起票・ユーザー承認待ち）。2026-10-03 追記: #2485 で対応済み |

### 10.5 PR #2351 の記録との関係

PR #2351 の本文と squash コミットのメッセージにある「対象外」の記載は、同 PR
内のレビュー是正より前の記述である。履歴は書き換えず、本決定記録を正とする。
起票候補（7 節）は自動運転中のため起票していない。
