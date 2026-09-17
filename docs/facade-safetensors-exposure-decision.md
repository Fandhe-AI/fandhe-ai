# safetensors save／load（onnx-interop）の facade 公開可否の設計判断記録（#1754）

イシュー #1754「safetensors save／load の再公開と Tensor の Debug／Display を追加する」のうち、safetensors 再公開部分（(B)）に対応する。親: #1616。関連: #1652（ONNX import 公開可否）・#1775（ONNX export 公開可否）・ルート: #1570／#1601。

本ドキュメントは **コード変更を伴わない設計記録**を成果物とする。`crates/onnx-interop/src/st_load.rs`／`st_save.rs`（本体ロジック）・`docs/spec/`（正本 submodule）・依存（`Cargo.toml`／`Cargo.lock`）・ガードレール閾値・数値一致許容誤差（tolerance／baseline）はいずれも変更しない。`Tensor` の `Debug`／`Display`（(A) 部分）はコード実装済み（本 PR 内・別コミット）であり、本ドキュメントの対象は (B) のみである。

基準コミット: 本 PR ブランチ作成時点の `origin/main`（2026-09-15）。**イシュー #2019（2026-09-17）で案 A（素の再エクスポート）を確定・facade 公開実装を完了した（§11）。以下 §1〜§10 は #2019 以前の段階 0 記録として残し、最新状態は §11 を参照する。**

**結論（当初記録。#2019 以前）**: `docs/compat-api-scope.md` §5（2026-09-14 追記）は「#1754 は onnx-interop の crates.io publish 承認を前提として共有するため blocked のまま close しない」と明記しており、`docs/facade-onnx-import-exposure-decision.md` §6.2・§9 も同旨の読み替えを確定済みである。本判断はこれをそのまま踏襲する。facade（`fandhe_ai`）への公開面追加コードは一切書かない（前提となる `onnx-interop` の crates.io 公開が未承認のため）。既存の負の guard テスト（`crates/facade/tests/api_surface.rs::facade_does_not_depend_on_unpublished_onnx_interop`）は段階 0 のまま不変。

## 1. 背景

対応する PyTorch 機能: `torch.save`／`torch.load`（の内 safetensors 形式に限定した部分。`state_dict` 全体の save/load は兄弟イシュー #1752 の対象）。`docs/compat-feature-gap.md` §2.15 は「safetensors 読み書き: リポ内非公開（facade 未接続）」を記録しており、親 #1616 はこの解消を目的の 1 つとする。

`onnx-interop::st_load`／`st_save` は本体ロジック（REQ-7 契約に基づく実装・テスト）としてはすでに完成している（下記 2 節）。しかし facade（`fandhe_ai`）からは一切到達できない。

## 2. 現状のコード事実

| 事実 | 出典 |
|---|---|
| `onnx-interop` は `publish.workspace = true`（= `false` 継承）の非公開クレート。`facade`（`fandhe-ai`）は crates.io 公開クレート（`publish = true`） | `crates/onnx-interop/Cargo.toml`・`crates/facade/Cargo.toml`・`Cargo.toml` `[workspace.package]`（**#2013 で `fandhe-ai-onnx-interop` として公開対象へ追加済み・`publish = true` へ更新済み。§11 参照**） |
| `facade` の `src/`・`Cargo.toml` は `onnx-interop`（`onnx_interop`）を一切参照しない。`crates/facade/tests/api_surface.rs::facade_does_not_depend_on_unpublished_onnx_interop`／`facade_sources_do_not_reference_onnx_interop` が機械的に固定（#1775 で追加） | `crates/facade/Cargo.toml`・`crates/facade/src/`（grep 0 件）（**#2017 で通常依存へ変更済み。§10 参照**） |
| 公開クレートの `Cargo.toml` に非公開クレートへの通常依存を持てないため「facade から公開する」は `onnx-interop` 自体を crates.io へ公開することと構造的に等価になる | `docs/facade-onnx-import-exposure-decision.md` §3.1・`docs/crates-io-publishing-order.md` §6 |
| `st_load` の公開 API: `LoadError`（**`#[non_exhaustive]` 具象 enum**）・`load_safetensors_f32_from_bytes(&[u8]) -> Result<HashMap<String, Tensor<f32>>, LoadError>`・`load_safetensors_f32(&Path) -> Result<HashMap<String, Tensor<f32>>, LoadError>`・`require_keys(&HashMap<String, Tensor<f32>>, &[&str]) -> Result<(), LoadError>` | `crates/onnx-interop/src/st_load.rs:46,48,116,170,181` |
| `st_save` の公開 API: `SaveError`（**`#[non_exhaustive]` 具象 enum**）・`save_safetensors_f32_to_bytes(&HashMap<String, Tensor<f32>>, Option<HashMap<String, String>>) -> Result<Vec<u8>, SaveError>`（**第 2 引数は `__metadata__` pass-through**）・`save_safetensors_f32(&Path, &HashMap<String, Tensor<f32>>) -> Result<(), SaveError>`（**引数順は path が先**） | `crates/onnx-interop/src/st_save.rs:71,73,114,188` |
| REQ-7 契約: 暗黙アダプタなし（転置・キーリネームを行わない）・無言 skip 禁止（`require_keys` は不足キーを全件収集）・F32 のみ（型レベルで保証）・キー昇順ソートによる決定的出力・一時ファイル + rename による書き込み整合性（OWASP A08） | `crates/onnx-interop/src/st_load.rs`・`st_save.rs` 冒頭コメント |
| `safetensors =0.7.0` は許容依存 9 区分の「相互運用」区分に含まれる（無条件承認済み。ただし現状は `onnx-interop` の `[dependencies]` としてのみ） | `.claude/rules/deps-policy.md`・`crates/onnx-interop/Cargo.toml` |
| `onnx-interop` の `tensor-core` 依存は `{ path = "..." }` のみで `version` 併記なし（公開時は公開クレートの規則へ追従が必要） | `crates/onnx-interop/Cargo.toml`（**#2013 の公開準備で version 併記済み。§11 参照**） |
| `docs/compat-api-scope.md` §1.2「state_dict／safetensors」行は #1616 対応中（未実装）のまま | `docs/compat-api-scope.md` §1.2（**#2019 で safetensors 部分は実装済みへ更新済み**） |

## 3. 契約整理

### 3.1 publish 前提

「facade から公開する」が `onnx-interop` 自体の crates.io 公開と構造的に等価であること・この publish 承認が facade 公開面拡張の一般承認（#1616 等）の範囲に含まれない別個の事項であることは、`docs/facade-onnx-import-exposure-decision.md` §3.1・§6.1 の整理と同一である（import／export／safetensors はいずれも同一クレート `onnx-interop` に同居するため、publish 承認自体は共有する）。#1616 のユーザー承認コメント（2026-09-12「今承認するので進めてください」）は「依存クレートの追加・更新」を明示的に範囲外としており、`onnx-interop` を crates.io へ公開する判断（新規公開クレートの追加に相当）はこの承認の対象外である。

### 3.2 薄いラッパー原則との整合

`st_load`／`st_save` は REQ-7 の契約（暗黙アダプタなし・F32 限定・無言 skip 禁止）を体現した「自作コアの上の薄いラッパー」（REQ-9・`.claude/rules/coding-rust.md`）であり、facade 公開面としての設計自体には問題がない。障壁は publish 承認のみである。

## 4. 案比較

| 案 | 内容 | 判定 |
|---|---|---|
| A | `onnx-interop` を crates.io 公開（7 クレート目）し facade から `st_load`／`st_save` を素の再エクスポート（`pub use fandhe_ai_onnx_interop::{st_load, st_save}` 相当）で公開する | `docs/facade-onnx-import-exposure-decision.md` §6.1 の publish 承認未取得のため **blocked**（現状の確定状態） |
| B | facade（または tensor-core）が `safetensors.workspace = true` を直接追加し、`onnx-interop::st_load`／`st_save` と同一契約（暗黙アダプタなし・F32 のみ・キー昇順ソート・一時ファイル+rename・検証順序）の薄いラッパー `fandhe_ai::io::safetensors::{save, load, save_to_bytes, load_from_bytes, SafetensorsError}` 相当を独立実装する | publish 不要で構造的に成立するが、**crates.io 公開クレート `fandhe-ai` の依存ツリーへ `safetensors`（推移的に既存 `serde`／`serde_json` 依存とは独立の新規推移的依存を含みうる）を加える変更**であり、#1616 承認の明示除外「依存クレートの追加・更新」に該当しうる。さらに `onnx-interop::st_load`／`st_save` とロジックを事実上複製することになり、REQ-7 契約のドリフト（2 実装が将来乖離するリスク）を新たに抱える → **ユーザー承認待ちの推奨候補**として記録（`docs/license-matrix.md` の使用箇所注記更新もセット） |
| C | `tensor-core` へ safetensors ラッパーを配置 | 案 B と同じ承認課題に加え、全下流クレート（`autodiff`／`backend-*`）へ依存が波及するため非推奨 |
| D | facade から `onnx-interop` へ `path` 依存を張るが `publish = false` のまま（facade 自身も `publish = false` へ変更するか、`[dev-dependencies]` 限定にする） | facade を非公開化するのは crates.io 公開方針（`docs/crates-io-naming-decision.md`）と矛盾し不可。`[dev-dependencies]` 限定では利用者コードから到達できず要件を満たさない → 不採用 |
| E | 段階 0（現状維持・記録のみ） | **本 PR の確定状態** |

## 5. 推奨

案 B（facade 直接依存による独立実装）を推奨候補として記録する。理由: publish 承認（案 A）はユーザーにとって影響範囲が大きい判断（新規 crates.io クレートの追加・公開名の確保・依存順序の再設計）であり、safetensors 単体の公開ニーズに対して過大な可能性がある。一方 案 B は影響範囲を「`safetensors` 依存 1 件の追加」に限定できる。ただし採否・実装方式（配置クレート・API 形状・`onnx-interop` とのロジック重複の扱い）はユーザー承認が必要であり、本 PR では判断しない。

## 6. 再開条件

以下のいずれかのユーザー承認を得た時点で、対応する実装イシューを起票する:

1. `onnx-interop` の crates.io 公開一式（`docs/facade-onnx-import-exposure-decision.md` §9 と同一の承認事項一式。案 A 選択時）
2. facade（または tensor-core）への `safetensors` 直接依存追加・API 形状（案 B 選択時。`fandhe_ai::io::safetensors` のような配置か、既存 `fandhe_ai` 直下再エクスポートかを含む）

## 7. スコープ外

- `Var`／`Tape` の状態（学習パラメータ）を safetensors へ直接保存する `state_dict`／`load_state_dict` 相当（兄弟イシュー #1752 の対象）
- ONNX import／export の facade 公開可否（`docs/facade-onnx-import-exposure-decision.md`・`docs/facade-onnx-export-exposure-decision.md` が既に対象）
- f16 等 F32 以外の dtype 対応（`onnx-interop` 側スコープ外として既に記録済み。`st_save.rs` 冒頭コメント参照）

## 8. 出典

- `docs/facade-onnx-import-exposure-decision.md`
- `docs/facade-onnx-export-exposure-decision.md`
- `docs/compat-api-scope.md` §5（2026-09-14 追記）
- `docs/compat-feature-gap.md` §2.15
- `crates/onnx-interop/src/st_load.rs`・`st_save.rs`
- `.claude/rules/deps-policy.md`

## 9. 追補（イシュー #1963・2026-09-17）: publish 前提の充足

§3.1 の publish 前提（`onnx-interop` の crates.io 公開が facade 公開面拡張
の一般承認範囲外であるため段階 0 とする根拠）のうち、「`onnx-interop`
自体の crates.io 公開」はイシュー #1963 のユーザー承認（2026-09-17）を
受けて充足した（`docs/crates-io-publishing-order.md` §13）。ただし推奨案
（案 B: facade 直接 `safetensors` 依存の独立実装）自体の採否・実装着手は
別途ユーザー承認が必要であり、本追補は publish 前提の充足のみを記録する
（段階 0 は継続）。

## 10. 追補（イシュー #2017・2026-09-17）: import 側の facade 公開に伴う事実更新

§2 の事実表（「`facade` の `src/`・`Cargo.toml` は `onnx-interop` を一切
参照しない」行）は **ONNX import の facade 公開（#2017）により事実が
変わった**。facade は `fandhe_ai::interop::onnx::{OnnxModel, OnnxValue,
OnnxError}`（import 専用）のために `onnx-interop` への通常依存を正式に
持つようになった。旧 guard テスト名（`facade_does_not_depend_on_
unpublished_onnx_interop`）は「承認済み依存形状の検査」
（`facade_depends_on_onnx_interop_only_in_approved_shape`）へ差し替え
られた（詳細は `docs/facade-onnx-import-exposure-decision.md` §12.4）。

**本 issue（safetensors save／load）の判断自体は不変**: safetensors は
#2017 のスコープ外であり、`src/interop/onnx.rs`（ONNX 専用）には
safetensors 側の公開シグネチャを一切追加していない。案 B（facade 直接
`safetensors` 依存の独立実装）の採否・実装着手は依然としてユーザー承認
が前提のまま、段階 0 を継続する。

## 11. 追補（イシュー #2019・2026-09-17）: 案 A 確定・facade 公開実装

### 11.1 案の変更（案 B → 案 A）

§5 は当初「案 B（facade 直接 `safetensors` 依存の独立実装）を推奨候補」
としていたが、§5 が案 B を推した根拠（publish〈案 A〉はユーザーにとって
影響範囲が大きい判断であり、safetensors 単体の公開ニーズに対して過大）
は §9 の追補（#1963。2026-09-17 ユーザー承認）と `docs/onnx-interop`
の crates.io 公開準備完了（#2013。`fandhe-ai-onnx-interop` として 7
クレート目の公開対象へ追加済み）により**解消済み**である。さらに
facade → `onnx-interop` への通常依存自体は #2017（ONNX import）で
既に成立しており（§10 追補）、本 issue（#2019）の時点で追加の依存結線
コスト・承認コストは発生しない。

一方、案 B は `onnx-interop::st_load`／`st_save` とロジックを事実上
複製することになり、REQ-7 契約（暗黙アダプタなし・無言 skip 禁止・
キー昇順ソートの決定的出力・一時ファイル + rename）が将来的に facade
側実装と onnx-interop 側実装の 2 系統へドリフトするリスクを新たに抱える
（§4 の案 B 判定欄がもとより指摘していた懸念）。加えて facade へ
`safetensors` クレートへの直接依存を追加することになり、`fandhe-ai`
の依存ツリーが 1 本増える。

これらを踏まえ、ワークフロー起動時のユーザー指示「#2019 は推奨で OK」
（2026-09-17）により **案 A（`fandhe_ai_onnx_interop::st_load`／
`st_save` の素の再エクスポート）を確定**した。案 A はロジック重複ゼロ・
facade への `safetensors` 直接依存追加なし（`onnx-interop` 経由の
推移的依存のまま）という利点があり、当初 §5 が案 B を推した際に前提と
していた「publish は重い判断」という制約が解消された結果、素の
再エクスポートで足りるという判断に至った。

### 11.2 実装記録

- 公開パス: `fandhe_ai::interop::safetensors::{LoadError, SaveError,
  load_safetensors_f32, load_safetensors_f32_from_bytes, require_keys,
  save_safetensors_f32, save_safetensors_f32_to_bytes}`（計 7 アイテム。
  `crates/facade/src/interop/safetensors.rs`。`pub use` のみのモジュール
  で型・関数を一切定義しない）
- `crates/facade/src/interop/mod.rs` は `pub mod onnx;`／
  `pub mod safetensors;` の 2 件のみを許容（`pub use` は引き続き 0 件）
- クレートルート直下の別実装 `LoadError`／`require_keys`（`onnx::interp`
  用。private `require_keys.rs` 由来）は再エクスポート対象から明示的に
  除外（`st_load::`／`st_save::` 接頭辞限定の再エクスポートで型を混同
  しない設計）
- ガード: `crates/facade/tests/api_surface.rs` に
  `interop_safetensors_module_is_pure_reexport`（純再エクスポート検査。
  `optim_module_is_pure_reexport` と同型）・
  `interop_safetensors_reexports_exactly_expected_surface`（再エクス
  ポート識別子の完全一致検査。`optim_module_reexports_exactly_expected_
  surface` と同型）・`interop_safetensors_types_are_reachable_via_facade`
  （facade 経由の到達性・`#[non_exhaustive]` ワイルドカード腕検査）を
  追加
- 統合テスト: `crates/facade/tests/interop_safetensors_roundtrip.rs`
  （`fandhe_ai`／`std` のみ import。bytes／`compat::Sequential::
  state_dict`／ファイルの 3 往復が全 bit 完全一致・特殊値〈NaN／±inf／
  -0.0／非正規化数〉も bit 完全一致・決定的出力・暗黙アダプタなし・
  不足キー／dtype 不一致／形状不一致／壊れたバイト列／存在しないパスが
  すべて型付き `Err` で fail-closed）
- `Cargo.toml`／`Cargo.lock`／`deny.toml`／`docs/license-matrix.md` は
  無変更（依存結線は #2017 で完了済みのため、facade への `safetensors`
  直接依存追加＝案 B の機構的帰結は本 issue では発生しない）
- `compat::Sequential::state_dict`／`load_state_dict`（#1752）と組み
  合わせた save→load→load_state_dict の往復例をモジュール doc の
  doctest として整備済み

### 11.3 スコープ外（不変）

`compat::Sequential`／`compat::callbacks::ModelCheckpoint` へのファイル
保存の薄いラッパー（`Sequential::save`／`load` 等）は案 A の範囲外
（素の再エクスポートのみ）のまま次点の切り出し候補として記録する
（`docs/compat-callbacks-design.md` §8）。F32 以外の dtype・入力サイズ
上限の導入・`st_load`／`st_save` 本体ロジックの変更・ONNX export（#2018）
も引き続き対象外。
