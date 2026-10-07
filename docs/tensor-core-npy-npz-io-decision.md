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

公開関数（`tensor-core` 内部。うちパス版 4 関数と `NpyError` は #2590 で `fandhe_ai::interop::npy` として公開済み。§12 参照）:

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

（#2589 追記: facade 公開形の候補比較と推奨案は §10 に記録した。承認は未取得。）

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

> **#2590 実装記録**: 本節は #2189 時点の保留記録である。承認（ルート #2499・2026-10-07）を受け #2590 で公開した。
> 公開した名前・配置は `fandhe_ai::interop::npy`（`NpyError`・`load_npy`・`save_npy`・`load_npz`・`save_npz` の純再エクスポート）。
> 下記の署名案（`tensor_io.rs`・facade 側 `pub fn`）と拡張トレイト案は**不採用**。`Tensor` へのメソッド追加はしていない。
> ガード 4 件の反転内容は §12 を参照。

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

（#2589 追記: 上記の署名案は想定止まりで形が未確定のため、候補比較・推奨案・
ユーザー承認依頼を §10 に記録した。承認は未取得で、本節の保留は継続する。）

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

## 10. #2589（facade 公開形の決定）の候補比較・推奨案・承認依頼

### 10.1 経緯

- §3 は「自由関数か `Tensor` のメソッドか」を、§7 は公開形（署名案は想定止まり・
  拡張トレイト案を併記）を保留しており、形が 1 つに決まっていない。このため
  一括承認（2026-10-04）の対象外で、親 #2588 は「記録作成（#2589）→ ユーザー
  承認 → 実装（#2590）」の 2 段で進める
- 調査基準コミット: `origin/main` `1c98db9a`（2026-10-05）
- **本節は承認の取得を意味しない**。facade・tensor-core のコードと保留ガード
  （`NpyIoHoldDoctestGuard`・`api_surface.rs` の 4 テスト）は変更していない

### 10.2 確定済みの内部 API（突合結果）

| 項目 | 内容 | 出典 |
|------|------|------|
| パス版 4 関数 | `load_npy`／`save_npy`／`load_npz`／`save_npz` | `crates/tensor-core/src/io/npy.rs`・`io/npz.rs` |
| バイト列版 4 関数 | `read_npy_bytes`／`write_npy_bytes`／`read_npz_bytes`／`write_npz_bytes`（保留ガードの走査対象外） | 同上 |
| `NpyError` | `#[non_exhaustive]`・`#[derive(Debug)]` のみ（`std::io::Error` を内包し `Clone`／`PartialEq` なし）。`Display`・`std::error::Error`・`From<std::io::Error>`・`From<ShapeError>` を実装 | `crates/tensor-core/src/io/mod.rs` |
| 読み込み | `read_file_bounded`（1 回の `File::open` で上限付き読み込み。上限 1 GiB）。std のみで cfg 分岐なし。symlink は辿る | `io/mod.rs` |
| 書き出し | `std::fs::write`（原子的でない）。`interop::safetensors::save_safetensors_f32`（一時ファイル + rename）とは挙動が異なる | `io/npy.rs`・`io/npz.rs` |
| 出力の決定性 | `write_npz_bytes` はキー昇順。npz の型は `HashMap<String, Tensor<f32>>` で `Sequential::state_dict`・safetensors と同形 | `io/npz.rs` |
| 出荷状況 | `tensor-core::io` は `fandhe-ai-tensor-core 0.10.0` に既に含まれる（#2318 のコミット `9fe4b523` は `v0.10.0` の祖先であることを確認済み）。ただし内部クレートはサポート対象外（`docs/compat-api-scope.md` §0）で facade からは未到達 | タグ祖先関係の実測 |

facade は `Tensor`・`ShapeError` を `tensor-core` から再エクスポートしている
（`crates/facade/src/lib.rs`）。`Tensor` は他クレート定義の型なので facade から
inherent メソッドは足せない（E0116）。メソッド形にするなら `tensor-core` 側の
inherent 追加か、facade の拡張トレイトの 2 通りに限られる。

### 10.3 候補比較

| 候補 | 形 | 判定 |
|------|----|------|
| (a) メソッド委譲 | a-1: `Var` に生やす。読み書きはホスト常駐 `Tensor<f32>` の話で tape・勾配と無関係。`tape.var(load_npy(p)?)` の 1 式で足りる。a-2: `tensor-core` の `Tensor<f32>` に inherent 追加。facade が `Tensor` を再エクスポート済みのため内部変更がそのまま公開面になる（§3 の不採用理由）。利用者の同名拡張トレイトメソッドより inherent が優先され、挙動が黙って変わる。a-3: facade の拡張トレイト（§7 併記案）。`use` が必要で、`load_*` は `self` を取らない関連関数・`load_npz` は `Tensor` を返さないため `Tensor` に載せる意味が薄い。sealed にしないと後からメソッドを足せない | 不採用 |
| (b) モジュール純再エクスポート | `tensor-core::io` の関数と `NpyError` を facade の 1 モジュールから `pub use` するだけ。ロジック複製ゼロ。先例は `interop::safetensors`（案 A） | **推奨** |
| (c) facade 独自型 | c-1: `interop::onnx` 型の薄いラッパー。隠すべき内部型が無い（署名に出るのは `Tensor<f32>`・`HashMap`・`Path`・`NpyError` のみ）ため写像層が増えるだけ。c-2: §7 の署名案（`src/tensor_io.rs` に facade 側 `pub fn` を定義して委譲）。宣言元が 2 箇所になり、workspace インベントリの期待集合と doc が二重管理になる | 不採用 |

比較軸と結果: 0.10.0 非破壊性は (b) が追加のみで最小／名前衝突は (b) が
クレートルートに名前を足さず最小／既存保留群との整合は 10.6／ロジック重複は
(b) のみゼロ／#2590 のガード反転の手間は (b) が `interop::safetensors` の
正ガードと同型で最小。

### 10.4 推奨案（いずれも未承認）

| 論点 | 推奨 | 比較した他案 |
|------|------|--------------|
| P1 形 | 純再エクスポート（型・関数・`impl` を facade に定義しない） | 10.3 の (a)(c) |
| P2 配置・名前 | `fandhe_ai::interop::npy`（新ファイル `crates/facade/src/interop/npy.rs`、`interop/mod.rs` に `pub mod npy;` を 1 行）。npy・npz を 1 モジュールに平らに置く（エラー型が共通の `NpyError` で、関数名が `_npy`／`_npz` で区別済みのため） | `compat` 配下（`compat` は numpy／Keras 慣習の API 形状の層で、ファイル形式の入出力は safetensors・ONNX と同じく `interop` に置く整理）／トップレベル `fandhe_ai::tensor_io`（§7 案。外部フォーマット入出力が 2 箇所に分かれる）／`interop::numpy`（名前の代替）／`interop::npy` と `interop::npz` の 2 モジュール（`NpyError` の置き場が割れ `pub mod` が 2 つ増える）／クレートルート直下（`use fandhe_ai::*` の利用者スコープに 5 名が入る） |
| P3 再エクスポート集合 | `NpyError`・`load_npy`・`save_npy`・`load_npz`・`save_npz` の 5 名（親 #2588 と §7 が名指しする範囲）。別名は付けない | バイト列版 4 関数も出す（10.7 の B） |
| P4 `Tensor` のメソッド | 追加しない（§3 を維持）。#2590 では `NpyIoHoldDoctestGuard` のうち `Tensor<f32>` への関連関数追加を検出するプローブを残す部分反転を推奨 | 全撤去 |
| P5 書き出しの非原子性 | 現状のまま公開し、モジュール doc に「`std::fs::write` で原子的でない。`interop::safetensors` の一時ファイル + rename とは異なる」と明記。原子化は `tensor-core` の挙動変更のため別 issue | 公開前に原子化 |
| P6 読み込みのパス扱い | 現状のまま（symlink を辿る・上限 1 GiB・TOCTOU 対策済みの `read_file_bounded`）。`interop::safetensors` と同じ姿勢。信頼できないパスを渡すかは呼び出し側の責任と doc に明記。no-follow が要る用途の受け皿は B のバイト列版 | facade でラップして `fs_guard` を通す（純再エクスポートでなくなり、非対応 OS で fail-closed になる挙動差が生じる） |
| P7 ガード（#2590 で実施。本 issue ではしない） | 否定ガード 4 件を「`src/interop/npy.rs` が `fandhe_ai_tensor_core::io` 接頭辞の 5 名だけを別名なしで再エクスポートする」正ガードへ反転（先例: `interop_safetensors_module_is_pure_reexport`・`interop_safetensors_reexports_exactly_expected_surface`）。facade 経由の往復 doctest を 1 つ置く | なし |

バイト列版を推奨に含めない理由: (1) 親 #2588・§7・保留ガードの名指し範囲の外。
(2) 名前が `read_*_bytes`／`write_*_bytes` で、safetensors の
`load_*_from_bytes`／`save_*_to_bytes` と揃っておらず、公開すると名前が固定される。
(3) 後から足しても非破壊。含める利点（メモリ上の入出力・呼び出し側で安全な
オープンや原子的書き込みを組める）もあり、判断はユーザーに委ねる。

### 10.5 `fandhe-ai =0.10.0` 非破壊の確認

| 観点 | 確認結果 |
|------|----------|
| 追加の種類 | `pub mod` 1 件と `pub use` 5 名の追加のみ。既存の `pub mod`／`pub use`／メソッドの署名・意味論は不変 |
| 名前衝突 | クレートルートに名前を足さないので `use fandhe_ai::*` の利用者に影響しない。`use fandhe_ai::interop::*` の利用者スコープに `npy` が増えるが、glob 由来の名前はローカル定義に隠れ、glob 同士の曖昧さは使った箇所でしか出ない（minor 変更の通常範囲）。facade 内に同名の既存項目は無い（`facade_does_not_reexport_or_declare_npy_io` が現に green） |
| メソッド解決 | `Tensor` にメソッドを足さないので、利用者の拡張トレイトとの解決順は変わらない |
| 公開後に固定されるもの | 5 名の署名（`P: AsRef<Path>`・`HashMap<String, Tensor<f32>>`）と `NpyError` の既存 variant の形。variant 追加は `#[non_exhaustive]` で非破壊だが、既存 variant のフィールド変更・削除は破壊的。`Display` 文言は契約にしない旨を doc に書くことを推奨 |
| 依存・unsafe | 依存追加なし・新規 `unsafe` なし・`Cargo.toml` 不変 |
| 対応 OS | 実装は std のみで cfg 分岐が無く、facade の Windows ビルド（#2390・#2391）を壊さない見込み（#2590 で再確認） |

### 10.6 既存保留群との整合（#2590 の波及）

- `pub mod` を 1 つ足すと、その時点で残る保留 doctest すべての glob 一覧に
  `use fandhe_ai::interop::npy::*;` を足す必要がある（`*_globs_all_pub_modules`
  ガードが facade の全 `pub mod` との一致を要求するため。基準コミットでの
  対象は npy 以外の標準プローブ 21 件 + rng の入れ子スコープ 1 件の計 22 箇所。
  兄弟 PR のマージで増減しうるため #2590 で再計数する）。配置をどこにしても
  同じ。クレートルート直下案だけはこの手間が無いが、10.5 の衝突面が広がる
- 新モジュールの 5 名が他の保留プローブのローカル名と重なると E0659 になる。
  現時点で重なりは見当たらないが、#2590 で `cargo test -p fandhe-ai --doc` で実測する
- `interop_module_exposes_only_approved_onnx_surface` の「`pub mod` は丁度 2 件」を
  3 件へ更新する必要がある
- 同じ Phase 3 ツリー（#2542）で `pub mod` を足す兄弟（`nn::kv_cache`・
  `pub mod inference` 等）と glob 一覧の編集が競合しうる。rebase 時は双方の行を残す（§9 と同じ）
- `docs/compat-api-scope.md` §5 の保留記録（#2189）は #2590 で適用記録へ書き換える。本 issue では触れない
- `compat::save_model`／`load_model`・`interop::safetensors` とは名前も型も重ならない。
  `state_dict` の `HashMap<String, Tensor<f32>>` をそのまま `save_npz` に渡せる

### 10.7 ユーザーに決めてほしい事項

- A: 10.4 の推奨案（`fandhe_ai::interop::npy` に 5 名を純再エクスポート。P4〜P6 は現状維持で doc に明記）で承認する
- B: A に加えてバイト列版 4 関数も同じモジュールから再エクスポートする
- C: 別の形を指定する（例: `interop::numpy`／`fandhe_ai::tensor_io`／`Tensor::<f32>::load_npy(path)` 形の拡張トレイト）
- D: 保留のままにする

承認コメントが形（A〜D、C は具体形）を名指しするまで #2590 は着手しない。
承認は #2590（または #2589 と #2590 の両方）に残すこと（#2590 は着手前に自 issue 上の承認コメントを確認する条件のため）。

### 10.8 本 PR で行わないこと

facade／tensor-core のコード変更、保留ガードの削除・反転、`compat-api-scope.md`
への適用記録、Issue 起票、spec 提案、原子的書き込みへの変更、非 f32 dtype 対応（§6 のまま）。

## 11. #2590（facade 公開の実装）の着手時判定

本節は docs のみの停止記録であり、承認を得たことを意味しない。

### 11.1 判定

- 基準コミット: `origin/main` `6dc72e45`（2026-10-05 確認）。
- #2590 は「ユーザーが承認した形」での公開を条件とするが、承認コメントは未取得のため着手せず停止する。
- 依存 #2589 は PR #2764 でクローズ済みだが、成果物は §10 の候補比較・推奨案の記録のみで、facade には何も公開していない。
- 実測: `crates/facade/src/interop/` は `mod.rs`・`onnx.rs`・`safetensors.rs` の 3 ファイル（`npy.rs` なし）。`interop/mod.rs` の `pub mod` は `onnx`・`safetensors` の 2 件。facade に npy 系の公開名はない。
- 正ガードは「公開済みの形だけを許す」検査であり、公開形が未確定の現状では反転先が存在しない。

### 11.2 承認コメントの確認範囲

| issue | コメント数 |
|---|---|
| #2590 | 0 |
| #2589 | 0 |
| #2588（親） | 0 |
| #2542（Phase 3 親） | 0 |
| #2499（ルート） | 0 |

§10.4 の推奨案は承認ではない。2026-10-04 の一括承認も §10.1 のとおり対象外である。

### 11.3 現状維持するもの（撤去・縮小・反転しない）

- `crates/facade/src/lib.rs` の `NpyIoHoldDoctestGuard`（正のプローブ doctest）。
- `crates/facade/tests/api_surface.rs` の 4 テスト: `npy_io_hold_doctest_globs_all_pub_modules`・`npy_io_hold_doctest_probe_body_matches_fixed_contract`・`facade_does_not_reexport_or_declare_npy_io`・`workspace_declares_npy_io_names_only_in_allowed_locations`。
- 共用する `NPY_IO_HOLD_PROBE_BODY`・`NPY_IO_FN_NAMES`・`scan_npy_io_reexports_and_declarations`。

### 11.4 受入条件ごとの扱い

| 受入条件 | 扱い |
|---|---|
| 承認コメントの確認 | 実施。0 件のため停止 |
| 承認形での facade 公開 | 未承認のため不実施 |
| 保留ガードの正ガード反転 | 公開物がなく反転先がないため不可 |
| `compat-api-scope.md` §5 適用記録・§3／§7 実装記録 | 公開していないため書かない（§5 の #2189 保留記録はそのまま） |
| facade 経由の利用例 | 対象 API が未公開のため追加不可 |

### 11.5 解除の順序

1. ユーザーが §10.7 の A〜D を選び、#2590 に承認コメントを残す
2. #2590 を reopen するか再起票するかをユーザーが決める
3. §10.4（P1〜P7）・§10.6 の手順で実装する

### 11.6 本 PR で行わないこと

コード変更、ガードの削除・反転・縮小、`compat-api-scope.md` の変更、Issue 起票・コメント投稿、spec 提案、依存追加、`unsafe`、tolerance 変更。

## 12. #2590 の実装記録（facade 公開・ガード反転）

### 12.1 承認の根拠と解除

- §11 の停止は、ルート #2499 のリポジトリ所有者本人のコメント（2026-10-07、`issuecomment-6033824965`）で解除された。同コメントは #2588 の §10 推奨案（バイト列版は含めない）を承認し、「記録に形が書かれていない点は実装せずに止め、承認依頼に戻す」ことを条件とする。
- 本実装は §10.4（P1〜P7）に書かれた形だけを実装した。形が書かれていない点は実装していない。

### 12.2 公開した形

| 項目 | 内容 |
|---|---|
| 配置 | `fandhe_ai::interop::npy`（`crates/facade/src/interop/npy.rs`、`interop/mod.rs` に `pub mod npy;` を 1 行） |
| 公開名 | `NpyError`・`load_npy`・`save_npy`・`load_npz`・`save_npz` の 5 名のみ。純再エクスポート（facade に型・関数・`impl` なし）、別名なし |
| 非公開のまま | バイト列版 4 関数、クレートルート直下の名前、`Tensor` への inherent メソッド |
| doc 明記 | P5（書き出しは非原子的）・P6（symlink を辿る・上限 1 GiB・信頼できないパスは呼び出し側の責任） |
| 利用例 | `npy.rs` モジュール doc の往復 doctest、`crates/facade/tests/interop_npy_roundtrip.rs` |

### 12.3 ガード反転の内容

| ガード | 結果 |
|---|---|
| `NpyIoHoldDoctestGuard` | 部分反転。`Tensor<f32>` への同名関連関数追加を検出するトレイトプローブ（`__probe_tensor`）だけを残し、ローカルモジュール・自由関数・モジュール名・型名の glob 衝突プローブは削除。全 `pub mod` の glob 一覧に `use fandhe_ai::interop::npy::*;` を追加 |
| `facade_does_not_reexport_or_declare_npy_io` | `facade_reexports_npy_io_only_from_interop_npy` へ反転。5 名は `interop/npy.rs` にだけ各 1 件、他ファイル 0 件、バイト列版 0 件、facade 独自の `NpyError` 宣言・4 関数の `fn` 宣言 0 件 |
| `npy_io_hold_doctest_probe_body_matches_fixed_contract` | 縮小後の固定本文へ更新 |
| `workspace_declares_npy_io_names_only_in_allowed_locations` | 期待集合は不変（再エクスポートは `fn` 宣言ではない）の正のインベントリとして維持 |
| 新規 | `interop_npy_module_is_pure_reexport`・`interop_npy_reexports_exactly_expected_surface`（`pub use` 3 行の完全一致。取得元の接頭辞一致方式は `io::crc32::{..}` を受理しうるため不採用）・`interop_npy_types_are_reachable_via_facade`・`npy_io_scanner_detects_unapproved_forms` |
| 付随更新 | `interop/mod.rs` の `pub mod` 件数 2→3、`GRAD_SCALER_PROBE_MODULES` 12→13、`LOWERCASE_PUB_USE_LEAF_ALLOWLIST` に 4 関数を追加 |

### 12.4 再確認・非対象

- §10.6 の glob 衝突: 新 5 名とモジュール名 `npy` は他の保留プローブのローカル名と重ならず、`cargo test -p fandhe-ai --doc` が全保留 doctest について通った。
- Windows ターゲット: パス版は `std::fs` と `AsRef<Path>` のみで、doctest の一時パスも `std::env::temp_dir()` 基準にしている（OS 依存のパス区切りなし）。
- 実機（CUDA／Metal）parity の対象外（ホスト常駐テンソルのファイル入出力で、`BackendOps`／GPU カーネルを追加しない）。`docs/perf/logs/` への申し送りは不要。
- 本実装で行わないこと（スコープ外）: 書き出しの原子化、`fs_guard` 経由の読み込み、非 f32 dtype、DEFLATE／zip64 書き出し、`Tensor::<f32>::load_npy` 形の拡張トレイト。
