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

### 11.3 スコープ外（不変・#2073 で一部解消）

`compat::callbacks::ModelCheckpoint` からの薄いラッパー結線
（`ModelCheckpoint::to_file`）は **#2073 で実装済み**（親 #2059。
`docs/compat-callbacks-design.md` §4.3・§9）。`compat::Sequential`
へのファイル保存ラッパー（`Sequential::save`／`load` 等）は引き続き
案 A の範囲外（素の再エクスポートのみ）のまま次点の切り出し候補として
記録する（`docs/compat-callbacks-design.md` §8）。F32 以外の dtype・
入力サイズ上限の導入・`st_load`／`st_save` 本体ロジックの変更・ONNX
export（#2018）も引き続き対象外。

## 12. 追補（#2080）: HF レイアウト復元ガイドの整備

本 issue で確定した公開面（案 A・素の再エクスポート・F32 限定）は
不変のまま、「Hugging Face（PyTorch）由来の safetensors を
`compat::Sequential` へ復元する」ハウツーを
`docs/huggingface-safetensors-interop-guide.md` として新設した
（イシュー #2080・親 #2059）。呼び出し側が明示的に行う変換
（`crates/facade/examples/hf_safetensors_sequential/convert.rs`）と
統合テスト（`crates/facade/tests/interop_safetensors_hf_layout.rs`）を
伴う。本モジュール（`crates/facade/src/interop/safetensors.rs`）自体・
`api_surface.rs` の allowlist 契約・公開 7 アイテムはいずれも変更して
いない。

## 13. 保守・運用チェックリスト（イシュー #3761）: safetensors の f32 以外の dtype（B5）

雛形は `docs/decision-record-template.md`。§1〜§12 は書き換えない。基準 sha は `5834b0958`（本節の `path:line` はこの sha のもの）。実機ホスト名は `<cuda-node>`・`<metal-node>`・`<rocm-node>`。本節は **コード変更を伴わない設計記録**であり、承認は **未承認**（#3770 論点 3 待ち。本節は承認の根拠にならない）。

### 13.0 位置づけ・課題・推奨

- 位置づけ: 横断ルート #3711・Phase 2。実装は #3845（読込側）・#3846（書出側と facade の再エクスポート）。
- 番号のずれ: イシュー本文と #3845・#3846 は「項目 28」と書くが、追跡表 `docs/compat-matrix-ari-tracker.md` では B5 は行 #27（#28 は HF Hub。`docs/api-compat-0.14.0-decision.md:68` も 27）。行 #27 だけを更新し、ほかの issue の記述は直さない。
- 課題: safetensors の save／load は `Tensor<f32>` 専用で、F32 以外は `LoadError::UnsupportedDtype` で拒否する（`crates/onnx-interop/src/st_save.rs:19-20`、`st_load.rs:143-149`）。`st_save.rs` が参照する追跡先 #274 は CLOSED で、現在は追跡先がない（`docs/compat-unlisted-holds-inventory.md:63`）。HF Hub 由来のチェックポイントは bf16／f16 が主流で、f32 の重みと i64 の buffer を混ぜたファイルもある。モデル取得側の検証は既に F32 以外を受け入れており（`st_load.rs:105-119`）、ロード API だけが F32 に縛られている。
- 確認した事実:
  - 既存 7 アイテムは `Tensor<f32>` 固定。ジェネリック化は破壊的で、別名の入口なら追加のみ（`docs/api-compat-0.14.0-decision.md:68`）。よって既存 7 アイテムのシグネチャと「F32 以外を `UnsupportedDtype` で拒否する」挙動は凍結する。既存テスト `dtype_mismatch_is_rejected_with_unsupported_dtype`（`crates/facade/tests/interop_safetensors_roundtrip.rs:275`）が無修正で通ることを不変条件にする。
  - facade の `src/interop/safetensors.rs` は `pub use` だけのモジュールに固定されている（`api_surface.rs:2683` `interop_safetensors_module_is_pure_reexport`）。新しい型・関数は onnx-interop の `st_load`／`st_save` に置き、接頭辞 allowlist（`api_surface.rs:2768-2771`）の 2 本のまま再エクスポートする。
  - `Element` は f32・f64・f16・bf16・i32・i64・bool に実装済み（`crates/tensor-core/src/element.rs`）。`CastElement` は f32・f64・i32・i64・bool に封印され f16／bf16 を持たない（`crates/tensor-core/src/cast.rs:88-96`）。
  - safetensors は `=0.7.0` 固定。onnx-interop は既に `half` に依存している。facade は `half` を再エクスポートしない（`TypedOps` の先例。`crates/facade/src/lib.rs`）。`Tensor<f16>`／`Tensor<bf16>` がシグネチャに現れるのは許容し、要素を名指ししたい利用者は `half` に直接依存する。
  - `Tensor` は `Clone` と手書き `Debug` を持ち、`PartialEq` を持たない（`crates/tensor-core/src/tensor.rs`）。
- 案比較:

| 案 | 内容 | 利点 | 欠点・影響 |
|----|------|------|-----------|
| A | dtype ごとの名前付き関数（`load_safetensors_f16` 等。約 24 本） | 既存の命名と同形 | 名前が爆発する。1 マップ 1 dtype のため混在ファイルを読めない |
| B | 封印 trait `SafetensorsElement` のジェネリック関数 | 名前が少ない | 混在ファイルを読めない。trait が公開面に乗る |
| C（推奨） | `#[non_exhaustive]` 列挙型 `SafetensorsTensor` と dtype タグ `SafetensorsDtype` を新設し `*_typed` の入口を足す。f32 変換は列挙型のメソッドで明示 opt-in | 混在ファイルを 1 回で読める。新しい型は 2 つ。追加のみ | 呼び出し側が `match` で取り出す |
| D | 変換つき読込（全 dtype を f32 へ昇格）だけ | `state_dict` へすぐ渡せる | 書出しを扱えない。非可逆変換が暗黙になる |
| E | 対象外にする | 変更なし | B5 が未公開のまま残り、相互運用行が「部分的」のまま |

- 推奨: **実装する（案 C）**。(1) 新しい型 2 つで f16・bf16・f64・整数・bool を追加のみで足せる。(2) 混在 dtype を 1 回で読める。(3) 既存 f32 経路を変えないので既定経路は bit 同一のまま。(4) 変換を暗黙にせず、案 D の利点（`state_dict` への接続）を明示 opt-in で取り込める。

### 13.1 公開形

- 配置: onnx-interop の `st_load`（型と読込）・`st_save`（書出）。facade は `fandhe_ai::interop::safetensors::*` へ `pub use` で再エクスポートするだけ。新モジュールは作らない（接頭辞 allowlist の変更が要るため）。
- シグネチャ案:
  - `#[non_exhaustive] #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)] pub enum SafetensorsDtype { F32, F16, BF16, F64, I32, I64, Bool }`（`header_name(&self) -> &'static str` が `"F32"`・`"F16"`・`"BF16"`・`"F64"`・`"I32"`・`"I64"`・`"BOOL"` を返す。safetensors の `Dtype` は公開面に出さない）
  - `#[non_exhaustive] #[derive(Debug, Clone)] pub enum SafetensorsTensor { F32(Tensor<f32>), F16(Tensor<f16>), BF16(Tensor<bf16>), F64(Tensor<f64>), I32(Tensor<i32>), I64(Tensor<i64>), Bool(Tensor<bool>) }`（`PartialEq` は derive しない）
  - メソッド: `dtype()`・`shape()`・`to_f32(&self) -> Result<Tensor<f32>, ShapeError>`（明示変換）・`from_f32(t: &Tensor<f32>, dtype: SafetensorsDtype) -> Result<Self, ShapeError>`（明示の縮小変換）
  - 読込: `load_safetensors_typed_from_bytes(&[u8])`・`load_safetensors_typed(&Path)`（ともに `Result<HashMap<String, SafetensorsTensor>, LoadError>`）・`require_keys_typed(&HashMap<String, SafetensorsTensor>, &[&str]) -> Result<(), LoadError>`
  - 書出: `save_safetensors_typed_to_bytes(&HashMap<String, SafetensorsTensor>, Option<HashMap<String, String>>) -> Result<Vec<u8>, SaveError>`・`save_safetensors_typed(&Path, &HashMap<String, SafetensorsTensor>) -> Result<(), SaveError>`。キー昇順の決定的出力と、一時ファイル＋`rename` の契約は既存 f32 版と同じ。f32 版の出力バイト列を変えないことをテストで固定する。
  - 対応外の dtype（U8・I8・I16・U16・U32・U64・F8_*・F4・F6_*・C64）は読込で `UnsupportedDtype` として fail-closed（無言 skip しない）。
- 正ガードへの反転計画（`crates/facade/tests/api_surface.rs`。B5 専用の `*HoldDoctestGuard` はなく、識別子の完全一致 allowlist が否定ガードの役を担う）:
  1. `interop_safetensors_reexports_exactly_expected_surface`（`:2764`。期待集合は現在 7 件）: #3845 で読込側 5 件を足して 12 件、#3846 で書出側 2 件を足して 14 件。どちらで再エクスポートしたかは実装記録に書く。
  2. `interop_safetensors_types_are_reachable_via_facade`（`:2854`）: 新しい `LoadError` variant の腕と、`SafetensorsTensor`・`SafetensorsDtype` をワイルドカード腕つきで `match` する到達性プローブを足す。
  3. 新設（すり替え検出の自己テスト）: onnx-interop 内で `SafetensorsDtype` の全 variant と `header_name` の対応を網羅 `match` で固定するインベントリテスト（例 `safetensors_dtype_inventory_is_exact`）。facade 側では `scan_safetensors_reexport_lines` に別名付け（`SafetensorsTensor as Tensor`）を渡して拒否される合成入力テストを足す（拒否されない実装なら走査器の修正を #3845 に含める）。
- 残す否定側: `interop_safetensors_module_is_pure_reexport`（`:2683`）、`interop_safetensors_reexport_scanner_rejects_root_path_and_glob`、接頭辞 allowlist 2 本、`interop/mod.rs` の pub 項目の検査。

### 13.2 rustdoc と doctest の例

- 新しい公開項目それぞれに 1〜2 行の役割要約を付ける。`SafetensorsTensor` の doc に、対応 dtype 表・変換の可逆性（§13.5）・「既存 `*_f32` は F32 以外を拒否したまま」・「`half` は facade から再エクスポートしない」を書く。
- doctest（CPU だけ・facade だけで通る形。facade は `half` を dev 依存にも持たないため f16／bf16 を名指ししない）: (a) `F64`・`I64`・`Bool` を混ぜたマップの `save_safetensors_typed_to_bytes` → `load_safetensors_typed_from_bytes` 往復、(b) `from_f32(.., SafetensorsDtype::BF16)` で bf16 チェックポイントを作る → 読込 → `to_f32()` → `compat::Sequential::load_state_dict`。
- GPU 経路はないので `no_run`／`#[ignore]` の例は該当なし。失敗系は `compile_fail` に頼らず、実行時テストと正のプローブで確かめる。

### 13.3 エラー型

- `LoadError`（`#[non_exhaustive]`）に 1 件追加: `InvalidBoolValue { key: String, index: usize, value: u8 }`（BOOL は 1 バイトで 0／1 以外は fail-closed で拒否する）。
- 既存 variant の再利用: 対応外 dtype は `UnsupportedDtype`（doc の「F32 以外」を「対応外の dtype」に直す。シグネチャは不変）、データ長不整合は `DataLengthMismatch`（`Dtype::size()` で割って数える）、shape は `Shape`。
- `SaveError` は追加なしを推奨: 入力が列挙型で dtype が閉じているため未対応 dtype は起きない。非 contiguous は既存どおり詰め直し、`DataUnavailable` を防御用に再利用する。`to_f32`／`from_f32` は `ShapeError` を返す。
- 検証順（A03。既存 `st_load.rs` と同じ 4 段）: (1) ヘッダ・レイアウト検査 → (2) dtype 検査 → (3) データ長と shape の要素数積 × 要素幅の一致 → (4) `Tensor::new` の shape 検査。バイト変換と確保は検証の後。BOOL の値検査は変換と同じループで行い、最初の不正バイトで止める。本番経路で `unwrap`／`expect` を使わない。

### 13.4 テスト計画（すべて案）

| 受け入れ基準 | テスト | 区分 | `<cuda-node>` | `<metal-node>` | `<rocm-node>` |
|---|---|---|---|---|---|
| 各 dtype の bytes 往復が bit 一致 | `crates/onnx-interop/tests/st_typed.rs::typed_roundtrip_is_bit_identical_per_dtype` | CI | 該当なし | 該当なし | 該当なし |
| 特殊値（NaN・±inf・-0.0・非正規化数、整数の最小・最大）が bit 一致 | 同 `typed_roundtrip_preserves_special_values` | CI | 該当なし | 該当なし | 該当なし |
| 混在 dtype ファイルを 1 回で読める・ヘッダの dtype 文字列が PyTorch と一致 | 同 `mixed_dtype_file_loads_in_one_pass` | CI | 該当なし | 該当なし | 該当なし |
| 対応外 dtype（U8・I8・F8_E4M3・C64 の代表）は `UnsupportedDtype` | 同 `unsupported_dtypes_are_rejected_fail_closed` | CI | 該当なし | 該当なし | 該当なし |
| BOOL の 0／1 以外は `InvalidBoolValue` | 同 `invalid_bool_byte_is_rejected` | CI | 該当なし | 該当なし | 該当なし |
| 決定的出力・非 contiguous の書出し | 同 `typed_save_is_deterministic`・`typed_save_handles_non_contiguous` | CI | 該当なし | 該当なし | 該当なし |
| F32 だけのマップを `*_typed` で書いたバイト列が既存と同一 | 同 `typed_f32_only_matches_legacy_f32_bytes` | CI | 該当なし | 該当なし | 該当なし |
| 明示変換の方向別規則 | 同 `explicit_conversion_follows_cast_contract` | CI | 該当なし | 該当なし | 該当なし |
| 既存 f32 経路が不変（F32 以外を拒否し続ける） | 既存 `st_load.rs`・`st_save.rs`・`st_roundtrip.rs`・facade `interop_safetensors_roundtrip.rs` が無修正で通る | CI | 該当なし | 該当なし | 該当なし |
| facade から往復・bf16 → `to_f32` → `load_state_dict` まで届く | `crates/facade/tests/interop_safetensors_typed_roundtrip.rs` | CI | 該当なし | 該当なし | 該当なし |
| 公開面の正ガード・自己テスト | §13.1 の 1〜3 | CI | 該当なし | 該当なし | 該当なし |

- 実機列が「該当なし」の理由: ホスト上のバイト列の読み替えだけで、バックエンドのカーネルや device 経路を通らない。Phase 10 で測る実機テストは該当なし（置き場もなし）。追跡表の「実機 parity」も「該当 issue なし」のまま。
- PyTorch 側の手動確認（`safetensors.torch.load_file` で dtype と shape を表示）は `st_save.rs` 冒頭と同様に `text` ブロックで doc に書き、CI では走らせない。

### 13.5 数値契約

- 読込・書出は算術を含まないため bit 完全一致（NaN もビット列ごと保つ）。REQ-2 の統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）が効くバックエンド間比較はない。
- 明示変換の方向別規則: f16／bf16 → f32 は完全表現。f32 → f16／bf16 は最近接偶数丸め・範囲超過は ±inf（`half`）。f64・i32・i64 → f32 は最近接偶数丸めで非可逆。bool・整数は `CastElement` の規則にそろえ、`cast_to_f32`／`cast_from_f32` を単一情報源にする。
- `RELATIVE_TOLERANCE`／`ABSOLUTE_RESCUE_THRESHOLD`・parity baseline・ガードレール閾値は変えない。baseline の追加はなし。FMA 契約・f64 長軸縮約の契約への影響もない。

### 13.6 互換への影響

- 判定: **追加のみ**（`docs/api-compat-0.14.0-decision.md` §8 の手順 1〜4）。(1) 触れる型は同 §4 の行 27。(2) `LoadError` は `#[non_exhaustive]` への variant 追加、`SaveError` と既存 7 アイテムのシグネチャは不変、新しい 2 型は `#[non_exhaustive]` で新設、関数は別名で新設。(3) 破壊的変更なし。(4) 移行ノート（#3780）の「追加」欄に載せる。`!`・`BREAKING CHANGE:` は付けない。
- `UnsupportedDtype` の doc 文言変更は挙動差なし。公開クレート `fandhe-ai-onnx-interop` 側も追加のみ。

### 13.7 適用記録・索引・追跡表

- 適用記録の案（実装 PR #3846 が承認後に `docs/compat-api-scope.md` §5 へ書く）: `**適用記録（イシュー #3846・親 #3711。承認: <#3770 の回答コメント URL>）**: SafetensorsTensor・SafetensorsDtype と load_safetensors_typed(_from_bytes)・save_safetensors_typed(_to_bytes)・require_keys_typed を fandhe_ai::interop::safetensors に公開。対応 dtype は F32・F16・BF16・F64・I32・I64・BOOL、それ以外は fail-closed。既存 *_f32 は不変。公開状態: 公開済み。`
- 現在の公開状態: 未公開。
- チェック: `docs/README.md` の索引［x］（本 PR）／追跡表 行 #27［x］（本 PR）／`docs/compat-api-scope.md` の適用記録［ ］（実装 PR）。

### 13.8 ベンチと性能への影響

既定経路（既存 `*_f32`）のコードは変えないので前後比較は不要。`*_typed` は opt-in の入口で、計測しないため数値は書かない。書出しの内部ヘルパーを共有する場合も、f32 版のバイト列同一テスト（§13.4）で挙動を固定する。速度が問題になれば別 issue（起票は承認後）。

### 13.9 依存の追加の有無

依存追加なし（onnx-interop は既に `safetensors =0.7.0`・`half` に依存、facade は `fandhe-ai-onnx-interop` に依存済み）。`Cargo.toml`・`Cargo.lock`・`deny.toml`・`docs/license-matrix.md` は無変更。許容 12 区分・禁止リストとの照合結果は不変。新しい `unsafe` なし（`from_le_bytes`／`to_le_bytes` で変換する既存方針を踏襲）。

### 13.10 所有者へ戻す論点（#3770 論点 3）

| 点 | 選択肢 | 推奨 | 確認がないときの扱い |
|----|--------|------|----------------------|
| 1. 入口の形 | A dtype ごとの関数／B 封印 trait／C 列挙型 `SafetensorsTensor`／D 変換つき読込だけ／E 対象外 | C | 実装へ進めない（期待集合 7 件の allowlist を維持） |
| 2. 対応 dtype の範囲 | `Element` 実装済みの 7 種／さらに U8 等を足す／浮動小数点だけ | 7 種。ほかは fail-closed | 同上 |
| 3. f32 への昇格 | 既存 `*_f32` で暗黙に昇格／`to_f32` で明示 opt-in／提供しない | 明示 opt-in。既存 `*_f32` は拒否したまま | 同上 |
| 4. f32 からの縮小（`from_f32`） | 提供する／提供しない | 提供する | 同上 |
| 5. BOOL の 0／1 以外 | `InvalidBoolValue` で拒否／非 0 を true とみなす | 拒否 | 同上 |
| 6. `require_keys` の扱い | 別名 `require_keys_typed` を足す／足さない | 足す | 同上 |
| 7. 名前 | `SafetensorsTensor`・`SafetensorsDtype`・`*_typed`／別名 | 左の案 | 同上 |
| 8. `half` の再エクスポート | しない／する | しない | 同上 |

### 13.11 セキュリティ・スコープ外・Closes／Refs

- OWASP: A03 は §13.3 の検証順（確保と変換の前に検証。BOOL の値も検査）。A04 は対応外 dtype を無言で飛ばさず fail-closed、非可逆変換を暗黙にしない。A06 は依存変更なし。A08 は決定的出力と一時ファイル＋`rename` を typed 側にも適用し、公開面を allowlist で固定する。秘密情報・実ホスト名は書かない。
- スコープ外（起票は所有者承認後）: U8／I8 等 `Element` 未実装の dtype、`__metadata__` の読出し API、入力サイズ上限（§11.3 から継続）、`Var`／`Tape` の f16 値型、`docs/huggingface-safetensors-interop-guide.md` への bf16 手順の追記。
- 本 PR は docs のみ。`Closes #3761`、`Refs #3770 #3845 #3846`。
