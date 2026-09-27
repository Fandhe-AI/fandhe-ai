# tensor-core npy／npz 読み書きの設計判断記録

イシュー #2189（親 #2131）。`Tensor<f32>` を NumPy 互換の `.npy`（単一配列）・
`.npz`（複数配列を束ねた ZIP コンテナ）形式で読み書きする機能を
`tensor-core::io` に実装した記録。

## 1. 目的・スコープ

- `.npy`・`.npz` の読み書きを完全自作コア方針（REQ-1）に従い実装する。
  `tensor-core` の直接依存は `half` のみであるため、ZIP コンテナ解析
  （EOCD／zip64／central directory／local header）・CRC-32・DEFLATE
  伸長はすべて自作した（依存追加なし）。
- 対象は `Tensor<f32>` のみ。他 dtype は `NpyError::UnsupportedDtype`
  で拒否する（スコープ外。§6）。
- 外部フォーマット（ファイル）を扱うため、形式不正・非対応 dtype・
  改ざん・巨大サイズ宣言はすべて確保の前に検証し fail-closed で
  拒否する（`.claude/rules/security.md` A03/A04/A05/A08）。

## 2. モジュール構成

```
crates/tensor-core/src/io/
├── mod.rs   # NpyError（公開エラー型）・境界付き読み取りヘルパ（pub(crate) bounded）
├── npy.rs   # npy ヘッダ parse／emit・read_npy_bytes／write_npy_bytes・load_npy／save_npy
├── npz.rs   # ZIP（EOCD／zip64／central dir／local header）読み取り・STORED 書き出し・load_npz／save_npz
├── crc32.rs # CRC-32（IEEE 802.3、pub(crate)）
└── inflate.rs # RFC 1951 DEFLATE 伸長（読み込み専用・pub(crate)）
```

公開関数（`tensor-core` 内部限定。§5 参照）:

- `io::npy::{read_npy_bytes, write_npy_bytes, load_npy, save_npy}`
- `io::npz::{read_npz_bytes, write_npz_bytes, load_npz, save_npz}`
- `io::NpyError`

## 3. 自由関数か inherent メソッドかの判断

`Tensor` に inherent メソッド（`Tensor::<f32>::load_npy(path)` 形式）は
**追加しない**。理由は #2156（RNG 確率分布サンプラー）の前例と同じ:
`crates/facade/src/lib.rs` が `pub use fandhe_ai_tensor_core::{...,
Tensor, ...};` で `Tensor` を再エクスポートしているため、inherent
メソッドを足すとそれだけで facade の公開面が広がってしまう
（`docs/rng-distributions-generator-decision.md` §「facade 公開判断」
参照）。`tensor-core` 側は自由関数（`io::npy::load_npy` 等）として
実装し、facade 側の公開形式（自由関数か `Tensor` inherent メソッドか）
は承認事項として保留する（§7）。

## 4. npy ヘッダの受理形（専用最小パーサ）

汎用 Python リテラルパーサは作らず、次の形だけを受理する専用パーサ
（`io::npy::parse_header`）を実装した:

- `{` キー `:` 値 (`,` キー `:` 値)* `,`? `}` の後ろに空白と `\n` のみ
- キーは `descr`／`fortran_order`／`shape` の 3 つのみ。順不同・各 1 回
  ずつ必須。未知キー・重複・欠落はすべて `InvalidHeader`
- `descr`: `<f4`（リトルエンディアン）・`>f4`（ビッグエンディアン）の
  みを対応 dtype とする。それ以外（`<f8`・`<i4`・`<f2`・構造化配列・
  object 等）は `UnsupportedDtype` で拒否する
- `fortran_order`: `True`／`False` の literal のみ
- `shape`: 非負整数のタプル（`()`・`(3,)`・`(2, 3)`。末尾カンマ許容）。
  rank は 64 まで、各要素は `usize` の範囲内

ヘッダ長は NumPy `_MAX_HEADER_SIZE` 相当の 10000 バイトを上限とし、
超過または範囲外は確保前に `HeaderTooLarge` で拒否する。

## 5. npz（ZIP コンテナ）の読み込み方針

- **central directory を正とする**。local header はシグネチャ確認と
  「ファイル名長 + extra 長」の読み飛ばしにしか使わない。理由は
  NumPy の local header のサイズ欄が `force_zip64=True` により常に
  `0xFFFFFFFF`（zip64 プレースホルダ）になるため
- zip64 extra field（id=0x0001）・マルチディスク拒否・暗号化フラグ
  拒否・エントリ数上限（65535）・CRC-32 検証を実装
- 1 メンバでも失敗すれば npz 読み込み全体を失敗させる（fail-closed。
  部分的な `HashMap` は返さない）
- 圧縮方式は STORED（0）・DEFLATE（8）のみ対応。それ以外は
  `UnsupportedCompression`
- **伸長後サイズ上限（PR #2318 レビュー指摘・P0）**: central directory
  の宣言 `uncompressed_size` が `MAX_MEMBER_DECOMPRESSED_BYTES`
  （＝単体ファイルの読み込み上限 `MAX_FILE_READ_BYTES`。1 GiB）を
  超えるメンバ、またはアーカイブ全体の宣言サイズ累積が
  `MAX_TOTAL_DECOMPRESSED_BYTES`（同じく 1 GiB。新規閾値を持ち込まず
  既存ポリシーを流用）を超える場合は `DecompressedSizeExceeded` で
  拒否する。`read_npz_bytes` は central directory を 2 パスで扱う:
  1 パス目で全エントリを解析し宣言範囲・上記上限を検査し、2 パス目で
  初めて `read_member_bytes`／`inflate`（＝出力バッファの実確保）を
  呼ぶ。DEFLATE の圧縮比上限（`inflate::MAX_COMPRESSION_RATIO`）は
  圧縮入力に対する相対値に過ぎず、圧縮入力自体がアーカイブサイズ上限
  いっぱいまで大きい場合は理論上数百 GiB 級の出力を許してしまうため、
  絶対値の上限を別途設けた。`inflate` 自身にも同じ絶対上限
  （`inflate::MAX_INFLATE_OUTPUT_BYTES`）を多層防御として持たせている
  （`crates/tensor-core/src/io/npz.rs`
  `MAX_MEMBER_DECOMPRESSED_BYTES`／`MAX_TOTAL_DECOMPRESSED_BYTES`）
- **DEFLATE ストリーム終端検査（PR #2318 レビュー指摘・P2）**:
  `inflate` は `BFINAL` ブロックを読み終えた時点で、ビット位置を
  バイト境界へ切り上げた消費バイト数が入力全体（ZIP の
  `compressed_size`）と一致することを要求する（不一致は
  `InvalidDeflate`）。出力長のみを照合すると、宣言された圧縮領域の
  末尾に付加した任意の余剰バイトを検出できない（CRC は伸長後データ
  のみが対象のため）

## 6. 書き出しの制限（対象外事項）

- **書き出しは STORED のみ**（`np.savez` と同じ）。`np.savez_compressed`
  相当の DEFLATE 圧縮書き出しは対象外
- **zip64 の書き出しはしない**。1 エントリが `u32::MAX` 以上、
  エントリ数が `u16::MAX` 以上の場合は `EntryTooLarge`／
  `TooManyEntries` で拒否する
- `save_npy`／`save_npz` はバイト列をメモリ上で組み立ててから
  `std::fs::write` する（**原子的な書き込みではない**）
- `np.savez` の出力とはバイト完全一致しない（zip64 extra の強制に
  差があるため）。「NumPy 標準ツールとの bit 同一往復」の要件は、
  npy 部分のバイト完全一致・要素 bit 一致・自前 reader での往復一致で
  満たす（統合テスト `tests/io_npz.rs` 参照）
- 非 f32 dtype・structured array・object dtype は対象外
  （`UnsupportedDtype`／読み込み失敗）

## 7. 承認事項（本 PR では実施しない）

facade の公開面拡張（4 件: `load_npy`／`save_npy`／`load_npz`／
`save_npz`、および `NpyError` の再エクスポート）は未承認のため保留する。
`crates/facade/src/lib.rs::NpyIoHoldDoctestGuard`（doctest の正の
プローブ）と `crates/facade/tests/api_surface.rs` の 4 テスト
（`npy_io_hold_doctest_globs_all_pub_modules`・`npy_io_hold_doctest_
probe_body_matches_fixed_contract`・`facade_does_not_reexport_or_
declare_npy_io`・`workspace_declares_npy_io_names_only_in_allowed_
locations`）で多層防御している。

署名案（承認された場合に facade へ追加する想定）:

```rust
// crates/facade/src/tensor_io.rs（新設想定）
pub use fandhe_ai_tensor_core::io::NpyError;
pub fn load_npy<P: AsRef<Path>>(path: P) -> Result<Tensor<f32>, NpyError>;
pub fn save_npy<P: AsRef<Path>>(t: &Tensor<f32>, path: P) -> Result<(), NpyError>;
pub fn load_npz<P: AsRef<Path>>(path: P) -> Result<HashMap<String, Tensor<f32>>, NpyError>;
pub fn save_npz<P: AsRef<Path>>(map: &HashMap<String, Tensor<f32>>, path: P) -> Result<(), NpyError>;
```

承認された場合、`crates/facade/src/lib.rs` に上記 `pub use` を追加し、
`NpyIoHoldDoctestGuard` とその doctest・`api_surface.rs` の否定ガード 4
件を撤去（正ガードへ置き換え）する。`Tensor::<f32>::load_npy(path)` 形式
（イシュー本文の受け入れ基準の書式）を選ぶ場合は inherent メソッドでは
なく拡張トレイト（`facade::compat` 相当の薄いラッパー）として実装し、
`tensor-core::Tensor` 自体には手を入れない設計とする。

新規 `unsafe`: 追加しない。依存追加: なし。spec 変更の提案: なし。

## 8. テスト

- 単体テスト: ヘッダパーサ（キー順・空白・トレーリングカンマ・rank0/1・
  重複/欠落/未知キー・整数 overflow）、CRC-32 既知ベクタ、inflate
  （stored／fixed／dynamic Huffman の既知ベクタ〈python `zlib` 実出力
  から採取〉・出力上限超過・予約ブロック型・不正入力）
- 統合テスト（`tests/io_npy.rs`・`tests/io_npz.rs`）: numpy 2.3.5 実出力
  fixture（`tests/fixtures/npy/gen_fixtures.py`。コミット済み）との
  要素 bit 一致・`write_npy_bytes` 出力の `np.save` バイト完全一致・
  fortran/big-endian の読み込み一致・非対応 dtype／改ざんデータの
  fail-closed 拒否・パス版 IO・非 contiguous view の C 順書き出し
- `gen_fixtures.py --verify <dir>`（ローカル専用。numpy 未導入の CI では
  実行しない）で Rust 側の書き出し結果を numpy で読める往復確認を行う
  運用とする

## 9. リスクと留意点

- inflate（DEFLATE 伸長）はホスト側の作り込みが最大のリスク。実装は
  Mark Adler の public domain 参照実装 `puff.c` の Canonical Huffman
  復号方式を踏襲しつつ、全境界検査を `Result` 化した
- 並列実行中の兄弟イシューと `crates/facade/src/lib.rs`・
  `api_surface.rs` の末尾を編集して競合しうる。追記位置は既存ガード群
  の末尾とし、rebase 時は hold ガードを両方残す形で解消する
