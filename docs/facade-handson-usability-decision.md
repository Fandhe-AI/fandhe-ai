# ハンズオン検証で見つかった使い勝手の公開形（決定記録）

## 1. 位置づけ・承認根拠・境界

- 横断位置: 親 #4347。本記録は #4348 として 3 件（#4351 safetensors のパス引数・#4352 `FitConfig` の `Default`・#4353 `EarlyStopping` の停止 epoch）の公開形を書く。残る 2 件（#4354・#4355）は #4349 が §10 以降へ追記する
- 出典: 2026-11-21 のハンズオン検証（crates.io の `fandhe-ai =0.15.0` だけを使う外部プロジェクトで学習・保存の流れを書いて見つけた使い勝手の問題）
- **承認: 未取得**。所有者の判断は #4350 で行う。本記録は承認の根拠にならず、`docs/compat-api-scope.md` §5 の適用記録も本 PR では書かない（承認 URL を持たない記録を残さないため）
- 境界: コード・テスト・CI の変更なし。シンボルは名前で引用し、行番号は書かない（基準 sha `1961434db`）

## 2. 横断の前提

- 互換判定は `docs/api-compat-0.14.0-decision.md` §8 の手順に従い、「追加のみ／破壊的で承認要／該当なし」の 3 区分で書く
- 次の版は workspace の lockstep で 0.16.0 の見込み。破壊的変更を採る場合に限り `docs/migration/0.16.0.md` が要る（未作成。作成は実装 issue）
- facade が唯一のサポートされる公開面（`compat-api-scope.md` §0）。`onnx-interop` は内部クレートだが crates.io 公開済みのため、既存シグネチャは同じ扱いで凍結する

## 3. #4351 safetensors の `impl AsRef<Path>`

### 3.1 現状

- 4 関数が `&Path` を固定で受ける: `load_safetensors_f32`・`load_safetensors_typed`（`crates/onnx-interop/src/st_load.rs`）、`save_safetensors_f32`・`save_safetensors_typed`（`st_save.rs`）
- facade の `crates/facade/src/interop/safetensors.rs` は `pub use` だけのモジュール。`api_surface.rs` の `interop_safetensors_module_is_pure_reexport` が facade 独自の定義を禁じ、`interop_safetensors_reexports_exactly_expected_surface` が識別子 14 件の完全一致を固定している。よって新しい入口は onnx-interop に置き、facade は再エクスポートする
- `interop_safetensors_types_are_reachable_via_facade` は `fn(&std::path::Path) -> …` への関数ポインタ型強制で既存シグネチャを固定している。その場でジェネリック化すると、このプローブ自体が壊れる
- 先例: `api-compat-0.14.0-decision.md` §4 は safetensors の入口を「ジェネリック化は不可。別名の入口を足す」とし、`facade-safetensors-exposure-decision.md` §13 も dtype 拡張で別名の入口（`*_typed`）を採って既存を凍結した。`impl AsRef<Path>` の先例は `compat::save_model`／`load_model`・`OnnxModel::from_path`／`to_path`

### 3.2 案

| 案 | 内容 | 利点 | 欠点・影響 |
|----|------|------|-----------|
| A | 既存 4 関数をその場で `path: impl AsRef<Path>` へ変える | 既存の名前で `&str`・`String`・`PathBuf` を受けられる | 既存シグネチャの変更で破壊的（api-compat §8）。関数ポインタ `fn(&Path) -> …` への型強制、`x.as_ref()` を渡す呼び出しの型推論、独自 `Deref<Target = Path>` 型の deref 強制が壊れうる（未実測。#4351 で確認）。§4 の「ジェネリック化は不可」と衝突し、移行ノートが要る |
| B（推奨） | 既存 4 関数は変えず、別名の入口 4 本を onnx-interop に足し facade で再エクスポート: `load_safetensors_f32_from_path`・`load_safetensors_typed_from_path`・`save_safetensors_f32_to_path`・`save_safetensors_typed_to_path`。`path: impl AsRef<Path>` を受け、既存関数へ `path.as_ref()` で委譲する | 追加のみ。既存の `*_from_bytes`／`*_to_bytes`・`OnnxModel::from_path`／`to_path` と命名が揃う。挙動は既存と同一 | 同機能の名前が 2 つになる。既存名に `&str` を渡したときのエラーは残るため、既存関数の rustdoc から新名を案内する |
| C | 対象外（rustdoc に `Path::new` と `&PathBuf` の渡し方を書く） | 公開面が増えない | 他の保存 API との作法の不一致が残る |

### 3.3 推奨・SemVer・`api_surface.rs`

- 推奨: **実装する（案 B）**。先例の方針と api-compat §8 の判定に合い、追加のみで済み、命名が既存と対称になるため。既存関数に `#[deprecated]` は付けない（既存利用者への警告という挙動変化になるため）
- SemVer: 案 B は追加のみ。案 A は破壊的で承認要（`BREAKING CHANGE:`・0.16.0 移行ノート）。案 C は該当なし
- `api_surface.rs`（案 B）: 識別子の期待集合を 14 件から 18 件にする。既存の関数ポインタプローブ（`interop_safetensors_types_are_reachable_via_facade`）が `&Path` で固定するのは `load_safetensors_typed`・`save_safetensors_typed` の 2 本だけで、f32 版 2 本（`load_safetensors_f32`・`save_safetensors_f32`）は固定されていない。既存プローブは変えず、この 2 本の範囲の凍結の証拠として残す。4 本すべてのシグネチャを固定するため、f32 版 2 本の `&Path` 関数ポインタプローブも #4351 の実装計画に含める。新 4 本には `&str`・`String`・`PathBuf`・`&Path` を渡す呼び出しのコンパイル可否プローブを足す（ジェネリック関数は高階関数ポインタへ強制できないため、関数本体内の呼び出しで確かめる）。純再エクスポート検査・接頭辞 allowlist・`pub use` 1 文 1 行の規約は維持する。スキャナの自己テスト（alias／swap 系）へ新名を足すかは #4351 で判断する

## 4. #4352 `FitConfig` の `Default`

### 4.1 現状

- `compat/training.rs` の `FitConfig` は `#[derive(Debug, Clone, Copy, PartialEq, Eq)]`・`#[non_exhaustive]` なし。フィールドはすべて `pub(super)`
- 入口は `FitConfig::new(epochs, batch_size)` とビルダー `shuffle`・`drop_last`・`accumulate_steps`・`validation_split` だけ。`epochs`・`batch_size` を設定するビルダーはない
- フィールドが非公開のため、`Default` を足しても構造体更新構文 `FitConfig { shuffle: true, ..Default::default() }` は下流でコンパイルできない
- `#[derive(Default)]` は `epochs = 0`・`batch_size = 0` となり、fit 呼び出しで `InvalidArgument` として拒否される値になる
- `api-compat-0.14.0-decision.md` §4 は `FitConfig` の `Copy`／`Eq` 維持を方針にしている。`api_surface.rs` の EMA／SWA 保留ガードが禁じるのは `use_ema` 等の名前のみで、`epochs`／`batch_size` は禁じられていない（inherent メソッド集合の完全一致固定がないかは #4352 で再確認する）

### 4.2 案

| 案 | 内容 | 利点 | 欠点・影響 |
|----|------|------|-----------|
| A（推奨） | 手書きの `impl Default` を足し `FitConfig::new(1, 32)` と等しくする（Keras `fit` の既定 `epochs=1`・`batch_size=32`）。あわせてビルダー `epochs(self, usize) -> Self`・`batch_size(self, usize) -> Self` を足す | `FitConfig::default().epochs(10).batch_size(64)` と既存ビルダーの作法で書ける。追加のみ。`Copy`／`Eq` を維持 | inherent メソッドが 2 本増える |
| B | `Default` だけ足す | 変更が最小 | `epochs=1`・`batch_size=32` 固定のため、作法を揃える目的をほぼ果たさない |
| C | `#[derive(Default)]` | 実装が最小 | 0／0 は拒否される値のため不採用 |
| D | 対象外（rustdoc に、構造体更新構文は使えず `new` が正の入口と書く） | 公開面が増えない | ハンズオンの不一致が残る |
| E | フィールドを `pub` にする | `..Default::default()` が成立 | 以後のフィールド追加がすべて破壊的になり、`validation_split_bits` の内部表現も露出する。不採用 |

### 4.3 推奨・SemVer・`api_surface.rs`

- 推奨: **実装する（案 A）**。フィールドを非公開に保つ案（A〜D）では構造体更新構文は成立せず、ビルダーなしの `Default` では目的を果たせないため。構造体更新構文が成立するのは案 E だけで、その場合も現行の `FitConfig` には `Default` がないため、フィールドの公開に加えて `Default` の追加（手書き `impl Default`）が必要になる。案 E は採らない。案 A の `Default` は他フィールドの既定も `FitConfig::new(1, 32)` と同じ値を引き継ぐ。rustdoc に既定値一覧と `FitConfig::default() == FitConfig::new(1, 32)` の doctest を書く
- SemVer: 追加のみ（trait impl と inherent メソッドの追加。derive 集合と既存メソッドは不変）
- `api_surface.rs`: `fit_types_are_reachable_via_facade_only` の近くに正のプローブを足す（`Default + Copy + Eq` の境界の具体化、`FitConfig::default() == FitConfig::new(1, 32)`、`epochs`／`batch_size` の `fn(FitConfig, usize) -> FitConfig` 型）

## 5. #4353 `EarlyStopping` の停止 epoch

### 5.1 現状（issue の前提の訂正を含む）

- `Sequential::fit_with_callbacks` は `callbacks: &mut [Callback]` を受けるため、所有権は呼び出し側に残る。fit の後に `let Callback::EarlyStopping(es) = &cbs[i] else { … }; es.stopped_epoch()` で今でも読める（`crates/facade/tests/compat_sequential_fit_swa_wiring.rs`・`compat_sequential_fit_ema.rs` が同形で読む。`docs/compat-callbacks-design.md` §4.2 も意図した設計として記録）。課題は「取得できない」ではなく「見つけにくい」である
- `History` は `#[non_exhaustive]` で pub フィールドは `loss`・`val_loss`・`lr`・`val_metrics` の 4 つ。下流は構築できないためフィールド追加は追加のみ。構築箇所は `compat/training.rs` の fit 本体、`compat/functional/train.rs`、`compat/callbacks.rs` の単体テストの 3 つ
- 学習を打ち切るのは `Callback::EarlyStopping` だけで、epoch 末に他の callback を処理してから `break` する。`EarlyStopping::stopped_epoch()` は fit ローカルの 0 始まりで、完走時は `None`。最終 epoch で停止しても `Some` なので `history.loss.len() < epochs` では判定できない

### 5.2 案

| 案 | 内容 | 利点 | 欠点・影響 |
|----|------|------|-----------|
| A（推奨） | `History` に `pub stopped_epoch: Option<usize>` を足す（`EarlyStopping::stopped_epoch()` と同じ意味）。既存の `&cbs[i]` で読む方法を rustdoc の例にも書く | fit の戻り値だけで判定できる。最終 epoch の停止も区別できる | 構築箇所 3 つを直す。`PartialEq` の比較対象が増える |
| B | 対象外（`fit_with_callbacks`・`EarlyStopping::stopped_epoch` の rustdoc に例を書くだけ） | 公開面が増えない | 見つけやすさは doc 頼み |
| C | `Callback::as_early_stopping(&self) -> Option<&EarlyStopping>` 等のアクセサ | パターンマッチより短い | variant ごとに同種が欲しくなり公開面が増える |

### 5.3 推奨・意味・SemVer・`api_surface.rs`

- 推奨: **実装する（案 A。案 B の rustdoc も併せて書く）**
- 等値比較への影響: `History` は `PartialEq` を derive しているため、案 A では停止 epoch が異なれば既存 4 フィールドが等しくても `==` の結果が変わる。推奨は停止 epoch を等値比較に含める（derive のまま。停止の有無も学習結果の差として扱う）。既存の `History` 同士の比較テストは構築箇所で `stopped_epoch: None` を与えれば結果が変わらない。この点は所有者への論点 7 とする
- 確定タイミングと観測契約: 停止 epoch は、その epoch 末の全 callback（`LambdaCallback` を含む）の処理が終わった後に確定し、fit の戻り値の `History` で公開する。callback 処理中に渡される `&History` には当該 epoch の停止要求が反映されない（`stopped_epoch` は `None` のまま。EarlyStopping の登録順に依存した値を callback に見せない）。callback が停止を知る方法は fit の戻り値と `EarlyStopping::stopped_epoch()` のみとする
- 意味: `EarlyStopping` が複数あっても 1 epoch で全 callback を処理してから抜けるため値は 1 つ。`fit_with_callbacks` を複数回呼ぶ場合はその呼び出しのローカル番号。callbacks を受けない入口（`fit`・`FunctionalModel` の学習）は常に `None`
- SemVer: 追加のみ（`History` は `#[non_exhaustive]`）。ただし `==` の判定対象が増える挙動変化があり、論点 7 で承認を得る
- `api_surface.rs`: `assert_is_history_type` 付近に `let _: &Option<usize> = &h.stopped_epoch;` 型のフィールドプローブを足す。停止／完走のテストは #4353 の範囲

## 6. 所有者へ戻す論点（#4349 が行を足す）

| 点 | 選択肢 | 推奨 | 確認がないときの扱い |
|----|--------|------|----------------------|
| 1 | safetensors: A（置換・破壊的）／B（別名の入口 4 本）／C（rustdoc のみ） | B | 確認があるまで実装へ進めない（保留のまま保護ガードを維持） |
| 2 | 別名の入口の名前（`*_from_path`／`*_to_path`） | 左記 | 同上 |
| 3 | 既存 4 関数に `#[deprecated]` を付けるか | 付けない | 同上 |
| 4 | `FitConfig::default()` の既定値（`epochs=1`・`batch_size=32`） | 左記 | 同上 |
| 5 | ビルダー `epochs`／`batch_size` を足すか（案 A／B） | 足す（A） | 同上 |
| 6 | `History::stopped_epoch` を足すか、rustdoc のみか（案 A／B） | A（B の doc も併記） | 同上 |
| 7 | `History` の等値比較（`PartialEq`）に停止 epoch を含めるか | 含める（derive のまま） | 同上 |

承認: 未取得（#4350 が `承認: <URL>・<日付>` の形で記入する）

## 7. 保守・運用チェックリスト（イシュー #4348）

### 7.1 公開形

- シグネチャ案・配置: §3〜§5。`fandhe_ai::interop::safetensors::*`・`fandhe_ai::compat::{FitConfig, History}`
- 正ガードへ反転する対象 / 否定プローブを残す対象: 新規公開名の正のプローブを足す。否定ガードの反転対象はなし（EMA／SWA 保留ガードは不変）

### 7.2 rustdoc と doctest の例

- 計画: 新 4 関数・`FitConfig::default`／`epochs`／`batch_size`・`History::stopped_epoch` に役割要約と例。既存 safetensors 関数の rustdoc から新名を案内する
- 失敗系: `compile_fail` に頼らず正のプローブで確かめる

### 7.3 エラー型

- variant の追加なし。safetensors は既存の `LoadError`／`SaveError` を返す委譲のみで、外部入力の検証順は既存の `st_load` のまま

### 7.4 テスト計画

すべて CI（ubuntu-latest）で動く。実機の列は該当なし（ホスト処理のみ）。

| 受け入れ基準 | テスト | 区分 |
|--------------|--------|------|
| #4351 別名の入口の追加 | `api_surface.rs` の期待集合 18 件・パス型 4 種のプローブ・往復テスト | CI |
| #4352 `Default` とビルダー | 正のプローブ・doctest（`default() == new(1, 32)`） | CI |
| #4353 停止 epoch | フィールド型プローブ・停止時 `Some`／完走時 `None` の統合テスト・最終 epoch で停止し履歴長が予定 epoch 数に等しくても `Some(epochs - 1)` となるケース（`history.loss.len() < epochs` からの推定実装を見逃さないため）・停止 epoch だけが異なる 2 つの `History` の `==` テスト（論点 7 の結論で期待値を切り替える。「含める」なら不一致、「含めない」〈`PartialEq` を手書きして除外〉なら一致）・`LambdaCallback` 内で観測する `History` の `stopped_epoch` が `None` であること | CI |

### 7.5 数値契約

- 該当なし（数値経路・tolerance・baseline に変更なし）

### 7.6 互換への影響

- 推奨案はすべて「追加のみ」。§3 案 A を採る場合だけ「破壊的で承認要」
- 公開面の区分（追加のみ）とは別に、§5.3 案 A では `History` の `==` の結果が変わる（既存 4 フィールドが同じでも停止 epoch が異なれば false）。`docs/api-compat-0.14.0-decision.md` §12.2 (c) の方針に沿い、これは「該当なし」相当の挙動変更として扱い、論点 7 で承認された場合はリリース時の「挙動のみの変更」欄に記載する

### 7.7 適用記録・索引・追跡表

- `docs/README.md` は更新済み。`docs/compat-api-scope.md` §5 の適用記録は承認後に実装 issue が書く（未承認のため未記入）。追跡表は対象外。公開状態: 未公開

### 7.8 ベンチと性能

- 既定経路への影響なし（委譲 1 段・trait impl・`Option` フィールド 1 つ）。前後比較は不要

### 7.9 依存の追加

- 依存追加なし。許容 12 区分・禁止リストに影響しない。新規 `unsafe` なし

## 8. セキュリティ観点

- A03: 推奨の safetensors 入口はパスを `AsRef<Path>` のまま既存関数へ渡し、文字列連結・シェル展開をしない。一時ファイル＋`rename` の書き込み契約と読み込み時の先行検証は不変
- 本記録に秘密情報・実機ホスト名・個人のパスを書かない

## 9. スコープ外・Closes／Refs

- 実装は #4351・#4352・#4353。#4354・#4355 の追記は #4349。承認は #4350
- PR は #4348 だけを `Closes` し、#4347・#4350〜#4355 は `Refs` のみ
