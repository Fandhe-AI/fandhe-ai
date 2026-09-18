# ONNX export（onnx-interop）の facade 公開可否の設計判断記録（#1775）

イシュー #1775「ONNX export の facade 公開（#1652 の判断を経て）」に対応する。親: #1653（ONNX export）・関連: #1652（ONNX import 公開可否）・ルート: #1570。

本ドキュメントは **コード変更を伴わない設計記録＋段階 0 を機械的に固定する guard テストのみ**を成果物とする。`crates/onnx-interop/src/**`（本体ロジック）・`docs/spec/`（正本 submodule）・依存（`Cargo.toml`／`Cargo.lock`）・ガードレール閾値・数値一致許容誤差（tolerance／baseline）はいずれも変更しない。`crates/facade/tests/api_surface.rs` への guard テスト追加のみコードに触れる（3 節参照）。仕様変更が必要になった場合は正本である `docs/spec/`（fandhe-ai-spec リポジトリ）側への提案とし、本リポでは `docs/spec/` を編集しない。

基準コミット: `e3b4953b`（2026-09-14）。`file_path:line` は同コミット時点のもの。後続の変更で行番号がずれる可能性があるため、参照する際は当該コミット、または近傍のコミットで再確認すること。

**結論（先に記す）**: #1652 の設計判断（`docs/facade-onnx-import-exposure-decision.md` §6.2）が「#1775 は『公開しない』という結論ではないため close しない・publish 承認が得られるまで blocked として扱う」と読み替え済みであり、本判断はこれをそのまま踏襲する。facade（`fandhe_ai`）への公開面追加コードは一切書かない（前提となる `onnx-interop` の crates.io 公開が未承認のため）。

## 1. 背景

対応する PyTorch 機能: `torch.onnx.export` 相当（読み込み側の `torch.onnx.load` 相当は #1652・`docs/facade-onnx-import-exposure-decision.md` が対象）。

親 #1653（ONNX export）は #1772（`Graph` -> `GraphProto`／`ModelProto` の構造的な組み立て）・#1773（`interp.rs` 対応 22 op の逆マッピング。`export_ops` モジュール）・#1774（import -> export -> import の roundtrip 構造一致テスト）を経て、`onnx-interop` 内部の export 機構自体は完成している（`crates/onnx-interop/src/onnx/export.rs`・`export_ops.rs`）。しかし facade（`fandhe_ai`）からは一切到達できない。

`docs/facade-onnx-import-exposure-decision.md` §6.2 は次のとおり読み替えを確定済みである:

> **#1775（ONNX export の facade 公開）**: 「公開しない」という結論ではないため close しない。§6.1 の publish 承認が得られるまで blocked として扱う。承認が却下された場合に限り close し、親 #1653 へ判断結果を記録する。

本 issue はこの読み替えを export 固有の設計論点（export 元の限定・橋渡しの配置候補・利用者向け注意事項）込みで確定記録し、後続（`onnx-interop` の publish 承認取得後）の実装 issue が迷わない状態にする。

## 2. 現状のコード事実

| 事実 | 出典 |
|---|---|
| `facade`（`fandhe-ai`）は crates.io 公開クレート（`publish = true`）。`onnx-interop` はルート `[workspace.package]` の `publish = false`（`publish.workspace = true` により継承）の非公開 5 クレートの 1 つ | `crates/facade/Cargo.toml:1-6`・`crates/onnx-interop/Cargo.toml:1-6`・`Cargo.toml:84` |
| `facade` の `src/`・`Cargo.toml` は `onnx-interop`（`onnx_interop`）を一切参照しない（2026-09-14 実測。本 PR で追加した `crates/facade/tests/api_surface.rs::facade_does_not_depend_on_unpublished_onnx_interop`／`facade_sources_do_not_reference_onnx_interop` が機械的に固定） | `crates/facade/Cargo.toml`・`crates/facade/src/`（grep 0 件） |
| 公開クレートの `Cargo.toml` に非公開クレートへの通常依存を持てないため「facade から公開する」は `onnx-interop` 自体を crates.io へ公開することと構造的に等価になる（詳細な整理は import 側 doc） | `docs/facade-onnx-import-exposure-decision.md` §3.1・`docs/crates-io-publishing-order.md` §6 |
| `onnx::export` の公開 API: `ExportError`（`#[non_exhaustive]`）・`ExportOptions`（`ir_version` 既定 8・`opset_version` 既定 17）・`encode_tensor`・`build_model_proto(&Graph, &ExportOptions) -> Result<ModelProto, ExportError>`・`check_exportable` | `crates/onnx-interop/src/onnx/export.rs` |
| `onnx::export` は `export_ops` の公開面を再エクスポートする: `pub use export_ops::{ConstantAttr, ExportNode, ExportOp, SUPPORTED_OP_TYPES, to_node_proto}` | `crates/onnx-interop/src/onnx/export.rs:57` |
| `build_model_proto` の入力は `Graph`（`graph::build_graph`。decode 由来、または #1773 以降に手組みされた `NodeProto` 列）のみ。autodiff の `Op`／`Tape`／facade `compat::Sequential` から `ExportNode`／`ExportOp` を構築する橋渡しコードは存在しない（grep 0 件） | `crates/onnx-interop/src/onnx/export.rs`・`export_ops.rs`（`autodiff`／`Sequential` への参照なし） |
| `export.rs` は `value_info` を常に空のまま出力する契約（`Graph` が中間テンソルの型／形状情報を保持していないため）。本クレート自身の decode（`graph::build_graph`）に対しては構造的にラウンドトリップ可能だが、`onnx.checker` 等の外部 ONNX ツールでの厳密な妥当性検証は保証しない | `crates/onnx-interop/src/onnx/export.rs` 冒頭コメント |
| `encode_tensor` は常に `raw_data`（リトルエンディアンのバイト列）のみへ書き出す（`float_data`／`int64_data` は常に空） | `crates/onnx-interop/src/onnx/export.rs` 冒頭コメント |
| import -> export -> import の総合 roundtrip（構造一致・bit 同一）・未対応 op の fail-closed 確認は `tests/onnx_export_roundtrip.rs` に実装済み（#1774） | `crates/onnx-interop/tests/onnx_export_roundtrip.rs`・`docs/onnx-export-op-mapping.md` §6 |
| `onnx::export`／`export_ops` は `BackendOps`／`Device` を一切参照しない（grep 該当なし。`ops::{GemmAttrs, LayerNormAttrs}` のみ使用しホスト側の属性表現に留まる） | `crates/onnx-interop/src/onnx/export_ops.rs` |
| `docs/compat-api-scope.md` §1.3 の Tier 2 表は「ONNX import 公開／export」の 1 行のみで、DDP／量子化のような除外事項への従属記載はない | `docs/compat-api-scope.md:249` |
| crates.io 名 `fandhe-ai-onnx-interop`・`fandhe-ai-interop` はいずれも未登録（2026-09-14 時点。import 側 doc で実測済み） | `docs/facade-onnx-import-exposure-decision.md` 2 節 |
| `onnx-interop` の `tensor-core`・`autodiff` 依存は `{ path = "..." }` のみで `version` 併記なし（公開時は公開 6 クレートの規則へ追従が必要） | `crates/onnx-interop/Cargo.toml` |

## 3. 契約整理

### 3.1 publish 前提（import 側 doc を参照。再導出しない）

「facade から公開する」が `onnx-interop` 自体の crates.io 公開と構造的に等価であること・この publish 承認が 2026-09-12 の facade 公開面拡張の承認範囲に含まれない別個の事項であることは、`docs/facade-onnx-import-exposure-decision.md` §3.1・§6.1 の整理をそのまま export 側にも適用する（export・import は同一クレート `onnx-interop` に同居するため、publish 承認自体は共有する）。

### 3.2 export 元と橋渡しの配置問題（export 固有の中核論点）

現状 export できるのは import 済み `Graph` の再書き出し（roundtrip）のみである。`torch.onnx.export` のユーザーストーリー（fandhe で学習したモデルを ONNX として書き出す）を満たすには、facade の `compat::Sequential`／`autodiff::nn::{Linear, activation::*}` のような学習済みモデル表現から `ExportNode`（`ExportOp` 列）を構築する橋渡しが別途必要になる。配置候補を比較する:

| 候補 | 概要 | 判定 |
|---|---|---|
| (a) facade 内で橋渡しを実装 | `compat::Sequential` を直接読める | 不採用（現時点）: facade は非公開クレート `onnx-interop` へ依存できない（3.1）。publish 承認が得られるまで着手できない |
| **(b) `onnx-interop` 内で `fandhe_ai_autodiff::nn` の型（`Linear` 等）から `ExportNode` 列を構築** | `onnx-interop` は既に `autodiff` を dev-dependency に持ち（`Cargo.toml`）、通常依存化も非公開クレート間の変更のためユーザー承認を要さない先例（`docs/crates-io-publishing-order.md` の対象は「公開クレートの依存」に限る）——**#2035 でこの根拠は陳腐化していると訂正・再導出した（§15.2）**。 | **推奨**。ただし `compat::Sequential` は facade 側の型のため直接は受け取れず、facade 側で `Sequential` -> 層列への薄い分解層が別途要る（facade ラッパー実装〈§4 の案 B〉と同時に確定させる）。**配置・対応層・公開面の設計確定は §15（#2035）参照** |
| (c) 橋渡しを設けず roundtrip 限定で公開 | import した `Graph` の再書き出しのみを公開 | 不採用: REQ-7 のユーザーストーリー「学習済みモデルを書き出す」を満たさない。roundtrip 限定なら PyTorch `torch.onnx.export` 相当と呼べない |

現時点ではいずれの候補も実装しない（publish 承認が前提のため）。橋渡しの実装 issue（承認後）は (b)＋facade 側の薄い分解層を軸に検討する。

### 3.3 REQ-12・薄いラッパー原則との整合

`onnx::export`／`export_ops` は `BackendOps`／`Device` を一切参照しないホスト側の純粋な組み立てロジックであるため、facade へ公開しても REQ-12（「任意 `BackendOps` 実装を注入できる公開 API を設けない」）・`api_surface.rs` の機械検査（`Tape`／`BackendOps`／`new_with_ops` を再エクスポートしない）と衝突しない。ただし `prost` 由来の型（`ModelProto`・`NodeProto`・`AttributeProto`）・`half::f16`（`RawTensor::F16`）をそのまま公開面に出す素の再エクスポート案は、import 側 doc の案 A と同じ理由（`=x.y.z` 固定下での `prost`／`half` バンプが facade の SemVer 破壊になる・薄いラッパー原則違反）で不採用とする。

### 3.4 数値契約

export は bit 同一の往復契約（#1774 の roundtrip テストが固定）であり、REQ-2（バックエンド間数値一致の統一複合判定）・REQ-7（ONNX 数値一致の判定式）のいずれの tolerance にも触れない。本判断は tolerance／baseline の変更を伴わない。

## 4. 案比較（facade 公開面の形状）

import 側 doc の案 A〜E（`docs/facade-onnx-import-exposure-decision.md` 4 節）をそのまま export 側にも適用する:

| 案 | 概要 | 判定 |
|---|---|---|
| A: 素の再エクスポート | `prost`／`half::f16` が漏れる | 不採用（3.3） |
| **B: 薄いラッパー型**（`OnnxModel::to_bytes(&self, &OnnxExportOptions) -> Result<Vec<u8>, OnnxError>`／`to_path` 等。import 側 `OnnxModel::from_bytes`／`run` と対を成す） | `prost` 型を公開面に出さない専用ラッパー | **方針として推奨（現時点未着手）**。publish 承認未取得のため着手不可 |
| C: `onnx-interop` のコードを `facade` クレートへ移設 | 依存構造を変えず公開面だけ確保 | 不採用: REQ-1 の 10 クレート構成違反 |
| D: facade を経由しない独立の公開面 | — | 不採用: 「facade が唯一のサポートされる公開 API 面」（`docs/compat-api-scope.md` §0）に反する |
| **E: 段階 0** — 非公開のまま維持し再開条件を記録するのみ | 現状維持 | **現状の確定状態**（B の前提承認が得られるまで） |

## 5. 推奨（方針／現状の二層）

- **方針**: 公開する（案 B）。理由: DDP／量子化と異なり spec 上の除外事項に従属していない（`docs/compat-api-scope.md` §1.3）・REQ-7 のユーザーストーリー（学習済みモデルの書き出し）に現状公開入口が皆無・`onnx::export`／`export_ops` が `BackendOps` 非依存で REQ-12 と無衝突。
- **現状**: 非公開・未実装。ブロッカーは `onnx-interop` の crates.io 公開というユーザー承認事項（＋命名確定。import 側 doc §3.1 と共有）であり、2026-09-12 の facade 公開面拡張の承認範囲には含まれない。

この二層は #1628（DDP。除外事項への従属で Won't 相当）とは性質が異なる: DDP は制度的な足かせが理由だが、ONNX export 公開は制度的な足かせではなく、単に crates.io 公開という別個の承認手続きが未完了という状態にすぎない（import 側 doc 5 節と同じ整理）。

## 6. 再開条件・起票候補（承認後。本 PR では起票しない）

`onnx-interop` の crates.io 公開承認取得後、以下を承認済みの範囲で起票する（`.claude/rules/out-of-scope-tracking.md` に従い、ユーザー承認を得たうえで起票する）:

1. **`onnx-interop` の crates.io 公開準備**: 命名確定・`publish = true`・`tensor-core`／`autodiff` 依存への `version = "=x.y.z"` 併記・`release-all.yml` の `RELEASE_CRATES` 追加・per-crate README。**#1754（safetensors）と承認依頼を 1 回にまとめることを推奨**（import 側 doc §6.2 と同じ理由。`onnx-interop` は ONNX と safetensors の両方を同一クレートで扱う）。
2. **facade ラッパー実装**（案 B）: `OnnxModel::to_bytes`／`to_path`（`OnnxModel::from_bytes`／`run` と対）・`OnnxExportOptions`・`OnnxError`（import 側と共通のエラー型か、export 専用型かは実装時に確定）・`crates/facade/tests/api_surface.rs` の guard を「承認済み依存形状の検査」へ差し替え（3 節）・`docs/compat-api-scope.md` §1.3 更新。
3. **`Sequential`／`nn` -> `ExportNode` の橋渡し**（3.2 の (b) を推奨）。

## 7. 利用者向け注意事項として残す設計要件（将来のラッパー実装が満たすべき事項）

- (a) `value_info` は常に空のまま出力される（`export.rs` 冒頭契約）。本クレート内での roundtrip は保証するが、`onnx.checker` 等の外部 ONNX ツールでの厳密な妥当性検証は保証しない。将来の facade doc comment・`site/guides/interop.md` に必ず明記する。
- (b) `ExportOptions` の既定値（`ir_version` 8・`opset_version` 17）は、そのまま公開 API の既定値になる。
- (c) `encode_tensor` は常に `raw_data`（リトルエンディアン）のみで書き出す（`float_data`／`int64_data` は使わない）。
- (d) export は現状ホスト CPU 実行のみで `BackendOps`／`Device` を一切参照しない（3.3）。将来のラッパー doc には「`Device::Cuda`／`Device::Metal` を渡しても export 処理自体はホスト実行のままである」旨を明示する。

## 8. セキュリティ考慮（OWASP Top 10。docs のみのため留意点として記録）

- **A08 ソフトウェア・データ整合性**: 本 PR の唯一のコード変更（`crates/facade/tests/api_surface.rs` の 2 guard テスト）自体が A08 対策である——`docs/crates-io-publishing-order.md` §6 の前提（公開クレートは非公開クレートへ通常依存しない）を、CI が `cargo publish --dry-run` を実行しないという盲点ごと機械的に固定する（3 節参照）。将来 `release-all.yml`（`RELEASE_CRATES` 追加）を変更する場合は CI ワークフロー変更として `.claude/rules/security.md` の security-auditor 監査対象になることを承認事項に明記する。
- **A03 インジェクション／不正入力**: ONNX は非信頼の外部フォーマットである（import 側と対称の懸念）。export 自体はホスト側の内部 `Graph`（既に検証済み）を入力とするため新たな非信頼入力の受け口は増えない。将来の facade ラッパー実装で ONNX バイト列を書き出す `to_bytes`／`to_path` の出力サイズ・パス検証（`to_path` がパストラバーサルを許さないか）は実装時に設計要件として追加検討する（本 issue のスコープ外）。
- **A06 脆弱コンポーネント**: 新規依存の追加なし。`Cargo.lock` 不変。
- **A05／情報漏えい**: 本 doc に内部ホスト名・秘密情報は含めない。

## 9. スコープ外（本 PR では実装しない。`.claude/rules/out-of-scope-tracking.md` に基づき承認後に別 issue で起票）

- `onnx-interop` の crates.io 公開作業そのもの（6 節 1）。
- facade ラッパー実装（`OnnxModel::to_bytes`／`to_path` 等。6 節 2）・`api_surface.rs` guard の差し替え。
- `compat::Sequential`／`autodiff::nn` -> `ExportNode` の橋渡し（6 節 3）。
- `value_info`／`TypeProto` 出力による外部ツール妥当性（#1772 既知事項。7 節 (a)）。

## 10. 承認事項（列挙のみ。本 PR ではユーザー承認を得ない）

1. `onnx-interop` の crates.io 公開そのもの（命名・`publish = true`・`RELEASE_CRATES` 追加を含む一式。import 側 #1652 の承認事項 1 と同一事項）。
2. 上記 1 が承認された場合の、facade ラッパー API 形状（案 B）の確定（export 側の `OnnxModel::to_bytes`／`to_path` 形状を含む）。
3. `Sequential`／`nn` -> `ExportNode` 橋渡しの配置（3.2 の (b) を推奨するが最終確定は未了）。**§15（#2035）で配置・対応層の初期範囲・facade 公開面 1 件を設計記録として確定し、承認事項を §15.7 へ再整理した（未承認のまま）**。

## 11. 出典一覧

- `docs/facade-onnx-import-exposure-decision.md`（#1652。publish 前提・案 A〜E 比較の出典）
- `docs/onnx-export-op-mapping.md`（#1773／#1774。export op マッピング・roundtrip テストの契約）
- `docs/compat-api-scope.md` §1.3・§5（Tier 2 列挙・#1652 適用記録）
- `docs/crates-io-publishing-order.md` §6（公開クレートの依存規則）
- `crates/onnx-interop/src/onnx/export.rs`・`export_ops.rs`（本判断が参照するコード事実）

## 12. 追補（イシュー #1963・2026-09-17）: publish 承認取得済み

§10 承認事項 1（`onnx-interop` の crates.io 公開そのもの。import 側 #1652の
承認事項 1 と同一事項）は**取得済み**（2026-09-17・イシュー #1963 承認
コメント。`docs/crates-io-publishing-order.md` §13）。公開準備（rename・
公開メタデータ整備）は #1963 自身が完了させた。

§10 承認事項 2・3（facade ラッパー API 形状・`Sequential`／`nn` ->
`ExportNode` 橋渡しの配置）は未承認のまま残り、facade 公開面は段階 0 を
継続する（唯一のコード変更である `crates/facade/tests/api_surface.rs` の
段階 0 固定 guard テスト 2 件は無変更）。

## 13. 追補（イシュー #2017・2026-09-17）: import 側の facade 公開に伴う事実更新

§2 の事実表（「`facade` の `src/`・`Cargo.toml` は `onnx-interop` を一切
参照しない」行）は **ONNX import の facade 公開（#2017）により事実が
変わった**。facade は `fandhe_ai::interop::onnx::{OnnxModel, OnnxValue,
OnnxError}`（import 専用）のために `onnx-interop` への通常依存を正式に
持つようになった。旧 guard テスト名（`facade_does_not_depend_on_
unpublished_onnx_interop`／`facade_sources_do_not_reference_onnx_
interop`）は「承認済み依存形状の検査」（`facade_depends_on_onnx_
interop_only_in_approved_shape`／`facade_sources_reference_onnx_
interop_only_in_interop_module`）へ差し替えられた（詳細は
`docs/facade-onnx-import-exposure-decision.md` §12.4）。

**本 issue（ONNX export）の判断自体は不変**: export（`OnnxModel::
to_bytes`／`to_path` 相当）は #2017 のスコープ外であり、`src/interop/
onnx.rs` には export 側の公開シグネチャを一切追加していない。
`crates/facade/tests/api_surface.rs::interop_module_exposes_only_
approved_onnx_surface` が `src/interop/onnx.rs` の `pub` シグネチャを
`OnnxModel`／`OnnxValue`／`OnnxError`／`from_bytes`／`from_path`／`run`
に限定して検査するが、export 用メソッド（`to_bytes` 等）の追加自体を
拒否する専用ガードは本 issue のスコープでは新設していない——export
実装は依然として §10 承認事項 2・3（facade ラッパー API 形状・
`Sequential`／`nn` -> `ExportNode` 橋渡しの配置）のユーザー承認が前提の
まま、後続イシュー（#2018）へ引き継ぐ。

## 14. 追補（イシュー #2018・2026-09-17）: facade ラッパー実装完了

§10 承認事項 2（facade ラッパー API 形状。export 側の `OnnxModel::
to_bytes`／`to_path` 形状）は 2026-09-17 にイシュー本文・承認コメントで
**承認済み**（API 形状: `OnnxModel::to_bytes`／`to_path` ＋
`OnnxExportOptions`〈`ir_version` 既定 8・`opset_version` 既定 17〉）。
**§10 承認事項 3（`Sequential`／`nn` -> `ExportNode` 橋渡し）は本 issue の
対象外のまま**（本 issue は import 済みモデルの roundtrip export ラッパー
に限定。橋渡しは別途承認が必要な起票候補として §6 に残る）。

実装は `crates/facade/src/interop/onnx.rs` に追加した:

- `OnnxExportOptions`（`#[non_exhaustive]`。`ir_version: i64`・
  `opset_version: i64` の 2 フィールドのみ。`ExportOptions` の
  `producer_name`／`graph_name`／`opset_domain` は公開せず内部既定値の
  まま private ヘルパ `to_internal()` で補う）
- `OnnxModel::to_bytes(&self, options: &OnnxExportOptions) ->
  Result<Vec<u8>, OnnxError>`（`export::build_model_proto`〈allowlist
  fail-closed 検査込み〉→ `proto::encode_model` への薄い委譲）
- `OnnxModel::to_path(&self, path, options) -> Result<(), OnnxError>`
  （`to_bytes` 完了後にのみ `std::fs::write`。export 失敗時にファイルを
  作成・切り詰めない順序。既存ファイルは上書き）
- `map_export_error(ExportError) -> OnnxError`（新規 `OnnxError`
  variant は追加しない。`ExportError::UnsupportedOp` は既存
  `OnnxError::UnsupportedOp { op_type }` へ写像し、非既定 domain の
  場合は `op_type` を `"{domain}::{op_type}"` 形式にして情報を保持する。
  それ以外の `ExportError` variant は `OnnxError::InvalidModel` へ写像）

`crates/facade/tests/api_surface.rs` の承認範囲を 6 件 → 9 件（型定義 4
件・メソッド 5 件）へ拡張し、正例テスト（`approved_onnx_surface_yields_
no_offenses`）・負例テストの改名（`to_bytes` が承認済みになったため合成
違反例を `input_names` へ差し替え）・`FORBIDDEN_INTERNAL_TYPE_SUBSTRINGS`
への `ExportError`／`onnx::export::` 追加・facade 単独の到達性テスト
（`onnx_export_types_are_reachable_via_facade`）を追加した。

正しさは `crates/facade/tests/interop_onnx_export.rs`（facade + std の
み。roundtrip bit 完全一致・決定性・不動点・`to_path` バイト一致・
allowlist 外 op の fail-closed 拒否〈手組み protobuf〉）と
`crates/facade/tests/interop_onnx_internal_parity.rs`（facade +
`fandhe_ai_onnx_interop` 両方 import。`to_bytes` が内部クレート直接
呼び出しとバイト完全一致・`value_info` 空／`raw_data` 限定契約の確認・
既定値ドリフトガード・domain 修飾 `UnsupportedOp` の確認）で検証済み。
`Cargo.toml`／`Cargo.lock` は無変更（新規依存なし）。

## 15. 追補（イシュー #2035）: `Sequential`／`nn` -> `ExportNode` 橋渡しの設計確定

**コード変更なし・設計記録のみ。基準コミット `d9868a62`。承認事項は
§15.7 に列挙するのみで本 issue では承認を得ない（§10 と同じ扱い）。**

`#1653`（ONNX export）・`#1775`（本 doc）が「`Sequential`／`nn` ->
`ExportNode` 橋渡しの配置」（§3.2）を推奨のみで最終確定していなかった
ことを受け、§10 承認事項 3 を対象に配置・対応層の初期範囲・facade 公開
面（1 件）を設計記録として確定する。#2018 の facade ラッパー実装
（roundtrip export 限定）は対象外のまま。

### 15.1 現状のコード事実

| # | 事実 | 出典 |
|---|---|---|
| F1 | `onnx-interop` は `publish = true`（#1963 以降。`fandhe-ai-onnx-interop`）で 7 クレート目の公開対象。`dev-dependencies` に `fandhe-ai-autodiff = { path = "../autodiff" }` を `version` 併記なしで既に持つ | `crates/onnx-interop/Cargo.toml:29,73` |
| F2 | 公開順は `fandhe-ai-tensor-core` → `fandhe-ai-autodiff` → `fandhe-ai-backend-cpu` → `fandhe-ai-backend-cuda` → `fandhe-ai-backend-metal` → `fandhe-ai-onnx-interop` → `fandhe-ai`（`RELEASE_CRATES`）。`autodiff` の normal dependency は `tensor-core` のみ（`cargo tree -p fandhe-ai-autodiff --edges normal` で実測確認・循環なし） | `.github/workflows/release-all.yml:91-94`・`crates/autodiff/Cargo.toml:29-` |
| F3 | `SUPPORTED_OP_TYPES`（22 op）に `Tanh` は含まれない（grep 0 件） | `crates/onnx-interop/src/onnx/export_ops.rs:148-171` |
| F4 | `compat::Sequential` は `inner: NnSequential`（`fandhe_ai_autodiff::nn::container::Sequential`）の薄いラッパー。`layers() -> &[Box<dyn Module>]` で層列を読める。`add_softmax` は存在しない（grep 0 件） | `crates/facade/src/compat/sequential.rs:135`・`crates/autodiff/src/nn/container.rs:267` |
| F5 | `Module` trait の閉集合フック: `as_linear`／`as_linear_mut`（`Option<&Linear>`）・`as_relu`（`bool`）。`as_sigmoid`／`as_tanh` は存在しない | `crates/autodiff/src/nn/module.rs:156,162,274` |
| F6 | `nn::Linear` の `weight` は `[in_features, out_features]`（`x.matmul(w)` 慣習・転置は持たない）、`bias` は `Some` なら `[out_features]`。`weight()`／`bias()` は `pub` | `crates/autodiff/src/nn/linear.rs:20-31,139,143` |
| F7 | 数値経路（Linear）: interp `ops::gemm` は `acc: f32 = 0.0` から `p` 昇順に `a[i,p].mul_add(b[p,j], acc)` → `alpha * acc`（+ `beta * c`）。`predict` 側 CPU BLIS GEMM（`gemm_naive`）も同じ `p` 昇順 `mul_add` 連鎖で bit 完全一致契約・出力は `zeroed_output` でゼロ初期化。`alpha = beta = 1.0` 固定なら両者は同一演算列 | `crates/onnx-interop/src/ops/gemm.rs:86-104`・`crates/backend-cpu/src/gemm.rs:230-250`（`gemm_naive`）・`crates/backend-cpu/src/ops.rs:207-224`（`zeroed_output`） |
| F8 | 数値経路（ReLU）: `predict` は `x.max(0.0)`（NaN→0.0）、interp `Relu` は `nan_propagating_max` で NaN 伝播。**ReLU への入力が NaN のときにのみ結果が異なる**。ここでの「NaN 入力」はモデル入力テンソルの値そのものが NaN な場合に限らず、有限のモデル入力から途中の演算（例: `f32::MAX` 級の重みによる `mul_add` の overflow で `inf` が生じ、後続層で `0.0 * inf` 等により NaN へ転化する）を経て ReLU への入力が NaN になる場合も含む（反例は §15.6 参照） | `crates/backend-cpu/src/elementwise.rs:121-132`・`crates/onnx-interop/src/ops/activation.rs:31-42` |
| F9 | 数値経路（Sigmoid）: `predict`（`eval::sigmoid`）は `x >= 0.0` で 2 分岐する数値安定形、interp `sigmoid` は全域 `1/(1+exp(-x))`。**負入力で丸めが異なりうるため bit 完全一致は保証できない** | `crates/autodiff/src/eval.rs:415-427`・`crates/onnx-interop/src/ops/activation.rs:46-48` |
| F10 | `build_model_proto` は graph input／output を名前のみの `ValueInfoProto` で出力し `value_info` は常に空。`OnnxModel::run` は入力を名前で束縛する。`OnnxError` は `#[non_exhaustive]` | `crates/onnx-interop/src/onnx/export.rs:351-369`・`crates/facade/src/interop/onnx.rs:115-118,242-250` |
| F11 | facade 公開面ガード: `api_surface.rs` の `ALLOWED_PUB_ITEMS`（9 件）・`FORBIDDEN_INTERNAL_TYPE_SUBSTRINGS`（6 件）・正例テスト `approved_onnx_surface_yields_no_offenses`・実ファイル検査 `interop_module_exposes_only_approved_onnx_surface` | `crates/facade/tests/api_surface.rs:1580,1595,1909,1967` |
| F12 | interp の `LayerNormalization` は `f32` アキュムレータ（`sq_acc: f32`）を使う実装で、autodiff 側 LayerNorm（`f64` 縮約契約）との bit 一致は未検証 | `crates/onnx-interop/src/ops/layer_norm.rs:127-132` |

### 15.2 配置: §3.2 (b) の採用と根拠の再導出

**採用**: 橋渡し本体は `onnx-interop` 内の新モジュール（名称は後続実装
issue で確定。例 `onnx::export_nn`）に置き、`fandhe_ai_autodiff::nn` の型
（`&[Box<dyn Module>]`）から `ExportNode` 列＋initializer
（`RawTensor::F32`）→ `Graph` を構築する。

**旧根拠の訂正（F1）**: §3.2 (b) 行の「非公開クレート間の変更のため
ユーザー承認を要さない」は `onnx-interop` が非公開だった当時の前提であり、
#1963 以降 `publish = true`（F1）のため成立しない。再導出した整理:

1. 外部依存の新規追加ではない（workspace 内 path 依存。`deps-policy.md`
   の許容 9 区分・`Cargo.lock` の外部クレート集合は不変）。
2. ただし公開クレートの依存グラフ変更のため、`crates.io-publishing-
   order.md` §1 の規則（path 依存へ `version = "=0.9.0"` 併記）が適用対象
   になる——`[dev-dependencies]` の `fandhe-ai-autodiff` は通常依存へ昇格
   し `version` を併記する必要がある。
3. 公開順（F2）は変更不要・循環なし（実測確認済み）。
4. crates.io 上の `fandhe-ai-onnx-interop` の依存ツリーに
   `fandhe-ai-autodiff` が normal dependency として加わる（利用者から
   見える変化）。
5. 単一クレートの `cargo publish --dry-run -p fandhe-ai-onnx-interop` は
   registry 版 `autodiff` を参照するため、後続実装が `autodiff` に新規
   フック（§15.3 の `as_sigmoid` 等）を足すと、次回リリースまで単体
   dry-run が失敗しうる（7 パッケージ一括 dry-run はローカル解決のため
   成立する。`docs/crates-io-publishing-order.md` §8.1 参照）。

安全側に倒し、この依存結線を承認事項として明示列挙する（§15.7 項 1）。

**比較代替案**:

| 案 | 概要 | 判定 |
|---|---|---|
| (b) | 上記。`onnx-interop` が `fandhe_ai_autodiff::nn` を直接走査 | **推奨** |
| (b′) | `onnx-interop` は `tensor-core` 型のみ受ける記述子 API（weight／bias／活性化種別）とし、層走査は facade が行う | 依存辺を増やさない利点はあるが、層走査ロジックが facade に入り薄いラッパー原則（REQ-9）に反する・対応層の閉集合が 2 クレートに分散する。不採用 |
| (a)／(c) | §3.2 の判定を踏襲（facade 内実装は非公開クレート依存で不可・roundtrip 限定は REQ-7 未達） | 不採用（§3.2 のまま不変） |

`Module` trait（autodiff）に `ExportNode` を返すメソッドを足す案は、
autodiff → onnx-interop 依存（循環）を要するため不可。

### 15.3 対応層の初期範囲と fail-closed 契約

**初期対応**（`compat::Sequential` から到達可能かつ `SUPPORTED_OP_TYPES`
に対応表あり）: `Linear`（→ `Gemm`）・`ReLU`（→ `Relu`）・`Sigmoid`
（→ `Sigmoid`）の 3 種のみ。

**対応表はあるが初期範囲外**（理由付きで後続候補として列挙のみ）:

- `Softmax`: `compat::Sequential::add_softmax` が存在しない（F4）・
  `Softmax::dim()` が非公開。
- `LayerNorm`: `LayerNormalization` は allowlist にあるが、interp 実装
  （F12）と autodiff 側 `f64` 縮約契約との数値一致が未検証。
- GELU: `Erf` 合成で表現可能だが interp の `erf` 近似実装との数値契約
  整理が未了。

**対象外（fail-closed 拒否）**: `Tanh`（F3）・SiLU／Hardswish／
LeakyReLU／ELU・Dropout・Conv1d／2d・Pooling 6 種・RMSNorm／
BatchNorm1d／2d・Embedding・MultiheadAttention・RNN 系。Dropout／
BatchNorm を拒否するため、初期範囲では `training` フラグは export 結果
に影響しない。

**拒否契約**: 非対応層を 1 つでも含むモデルは、`Graph` を一切構築せず
型付き `Err` を返す（無言 skip・部分 export をしない。
`.claude/rules/security.md` A08）。層の判定は既存の明示フック方式
（`as_linear`／`as_relu`）を踏襲し、Sigmoid 用に
`Module::as_sigmoid(&self) -> bool`（defaulted `false`。`as_relu` と
同型）を autodiff に 1 件追加する（後続実装 issue の変更範囲。`Any`
ダウンキャストは使わない）。既知の 3 フックのいずれにも該当しない層は
非対応と判定するため、将来の層追加は自動的に拒否側へ倒れる。

Linear の入力は rank 2（`[batch, in]`）限定（`Gemm` の制約）。層数 0 の
空 `Sequential` は型付き `Err` で拒否する（ノード 0 個・input==output
のグラフは意味のある export ではない）。先頭層が活性化のみ（`Linear`
を含まない）のモデルは構造上許容する（graph input は元々形状情報を
持たないため追加制約なし。F10）。

### 15.4 ONNX グラフへの写像規約

- `Linear` → `ExportOp::Gemm(GemmAttrs { alpha: 1.0, beta: 1.0,
  trans_a: false, trans_b: false })`、`inputs = [x, W, (b)]`（bias なし
  は 2 入力）。weight は `[in, out]` のまま initializer に書き、転置の
  実体化はしない（PyTorch の `transB=1` 慣習とは異なるが ONNX として
  正当）。alpha=beta=1.0 固定は F7 の演算列一致の前提条件。
- `predict` の Linear→ReLU 融合（`gemm_bias_act`）はグラフ構造へ持ち込
  まず、常に `Gemm` ＋ `Relu` の 2 ノードで出力する（融合・非融合は
  CPU で bit 完全一致が既存テストで確認済みのため）。
- 名前規約: initializer 名は `named_parameters()`／`state_dict()` の
  キーと同一。**graph input／output 名は公開契約として固定する**（推奨:
  input=`"input"`・output=`"output"`）——`OnnxModel::run` は feeds を
  名前で束縛し、facade には入力名を取得する公開手段がないため（F10・
  F11。`input_names` は `api_surface.rs` の合成違反例に使われている）、
  名前が doc 化された定数でなければ利用者は `from_sequential` → `run`
  を facade 単独で完結できない。中間テンソル名・ノード名は決定的な
  固定規約（例: `layer{i}_out`）とし、同一モデルからの export がバイト
  単位で決定的になること。initializer 名が固定の input／output 名・
  中間名と衝突しないことの検査は後続実装 issue の設計要件とする。
- 重みは `RawTensor::F32`（`raw_data` リトルエンディアン）。opset／
  ir_version は `OnnxExportOptions` の既定（17／8）をそのまま使う。
- **制約の明記（F10）**: graph input／output は型・形状情報なし・
  `value_info` 空のため、外部 ONNX ツール（onnx.checker／ONNX Runtime）
  での読み込みは保証しない（§7 (a) を継承）。

### 15.5 facade 側の薄い分解層と公開面

`compat::Sequential` は既に `NnSequential` を内包するため、facade 側は
`pub(crate)` の層列アクセサ（`self.inner.layers()` を返すだけ）を足し、
`interop::onnx` から onnx-interop の橋渡し関数へ渡す 1 段委譲のみとする。
`Module` trait への export 専用メソッド追加も、`Sequential` の新規
`pub` メソッド追加も行わない。

**facade 公開面（新規 1 件）**:

```rust
impl OnnxModel {
    pub fn from_sequential(
        model: &crate::compat::Sequential,
    ) -> Result<OnnxModel, OnnxError>;
}
```

`OnnxExportOptions` を引数に取らない（`Graph` 構築自体は opset 非依存
で、opset／ir_version は既存 `to_bytes(&self, &OnnxExportOptions)` が
既に受けるため重複引数を避ける）。書き出しは既存 `to_bytes`／`to_path`
をそのまま再利用する。

エラー型は `OnnxError`（#[non_exhaustive]。F10）を再利用する案（案 1）
と新 variant `UnsupportedLayer { index: usize, layer: &'static str }`
を追加する案（案 2）を比較する: 案 2 は import 時の「未対応 op」と意味が
異なり利用者が層番号で特定できる利点があるが、新 variant は公開面追加の
ため承認事項に独立して載せる（§15.7 項 4）。推奨は案 2。

`api_surface.rs` 更新箇所（後続実装 issue の作業として列挙）:
`ALLOWED_PUB_ITEMS` 9→10（`("fn", "from_sequential")`）・正例テスト
追加・到達性テスト追加・`FORBIDDEN_INTERNAL_TYPE_SUBSTRINGS` に
`ExportNode`／`ExportOp` 追加検討・`facade_sources_reference_onnx_
interop_only_in_interop_module` ガード維持のため実装は
`src/interop/onnx.rs` に置く。

### 15.6 数値契約

tolerance／baseline は一切変更しない。

- **`Linear`／`ReLU` のみで構成されるモデル**: export → `from_bytes` →
  `run` の出力が `Sequential::predict`（CPU）と**bit 完全一致**（根拠
  F7）。ただし一致が成り立つのは「グラフ中のいずれの ReLU 入力にも
  NaN が現れない」場合に限る。この条件はモデル入力テンソルが有限値
  であることだけでは保証されない（F8）。反例: bias なし
  `Linear(w=f32::MAX)` → `Linear(w=0.0)` → `ReLU` に有限入力 `2.0` を
  与えると、1 層目の `mul_add` で `f32::MAX * 2.0` が overflow して
  `inf` になり、2 層目で `0.0 * inf` が NaN を生む（両経路とも F7 の
  同一 `mul_add` 連鎖を辿るため中間値自体は一致する）。この NaN が
  ReLU へ入ると `predict` は `0.0`（NaN→0.0 の飽和）を返す一方 interp
  は NaN をそのまま伝播するため、最終出力が一致しなくなる。したがって
  bit 完全一致の主張は「モデル入力が有限値」ではなく「グラフ全体を
  通じて ReLU 入力に NaN が発生しない」ことを前提とする（overflow に
  よる NaN の混入を含む）。NaN が入力に含まれる場合、または上記のよう
  な中間 overflow で NaN が発生しうる場合は、ReLU の NaN 意味論差
  （F8）により一致を主張しない。
- **`Sigmoid` を含むモデル**: F9 により bit 完全一致は保証できない。
  選択肢: (α) REQ-2 既存の統一複合判定（相対 1e-3 未満 または 絶対
  1e-5 未満。定数不変）で検証し bit 一致は Linear／ReLU 限定とする、
  (β) interp の `sigmoid` を `eval::sigmoid` と同じ 2 分岐形へ揃える
  （import 済み全モデルの interp 数値が変わる契約変更のため別承認・
  PyTorch 参照 fixture 再検証が必要）、(γ) Sigmoid を初期範囲から外す。
  **推奨は (α)**（§15.7 項 5）。
- 構築した `OnnxModel` の `to_bytes` は決定的（同一モデル→同一バイト
  列）・`from_bytes(to_bytes())` の再 `to_bytes` は不動点（#2018 の
  既存契約を継承）。
- export はホスト実行のみで `BackendOps`／`Device` 非経由（§3.3・§7 (d)
  を継承。REQ-12 と無衝突）。

### 15.7 承認事項

**2026-09-18 ユーザー承認済み**（親 #2034 コメント
〈https://github.com/Fandhe-AI/fandhe-ai/issues/2034#issuecomment-5726738002〉）:
項 1（配置 (b)＋onnx-interop → autodiff 通常依存化）承認・項 2 は
Linear／ReLU の 2 種へ縮小して承認（Sigmoid は別 issue）・項 3
（`OnnxModel::from_sequential`）承認（facade 結線は #2037）・項 4
（`OnnxError::UnsupportedLayer` 新設）承認・項 5（Sigmoid 数値契約）は
保留。以下は起票時点の原文（承認範囲の記録として保持）。

1. **配置**: §3.2 (b)。付随して `fandhe-ai-onnx-interop` の
   `[dependencies]` へ `fandhe-ai-autodiff`（path＋`version = "=x.y.z"`
   併記）を追加する公開クレート間依存の結線（外部依存の追加なし・公開
   順不変）。承認されない場合の代替: (b′) を採用し facade 側で層走査
   を行う（REQ-9 逸脱を許容する判断が別途必要）。
2. **対応層の初期範囲**: `Linear`／`ReLU`／`Sigmoid` の 3 種。それ以外
   は型付きエラーで fail-closed 拒否。autodiff `Module` へ defaulted
   フック `as_sigmoid` を 1 件追加。承認されない場合の代替: `Linear`／
   `ReLU` の 2 種に縮小（Sigmoid の数値契約〈項 5〉が未承認の場合）。
3. **facade 公開面 1 件**: `OnnxModel::from_sequential(&Sequential) ->
   Result<OnnxModel, OnnxError>`。承認されない場合の代替: 段階 0 を
   維持し roundtrip export（#2018）のみで運用を続ける。
4. （付随・独立判断）`OnnxError` への新 variant `UnsupportedLayer`
   追加の可否。承認されない場合の代替: 既存 `UnsupportedOp` を再利用
   （層種別名を `op_type` フィールドへ間借りさせる）。
5. （付随・独立判断）Sigmoid の数値契約 (α)／(β)／(γ) の選択（推奨
   (α)）。承認されない場合の代替: (γ) を採用し Sigmoid を項 2 の初期
   範囲から外す。

### 15.8 後続 issue への引き継ぎ・スコープ外

- 実装 issue（承認後）: 名前衝突検査・空モデル拒否を含む onnx-interop
  側モジュール＋`as_sigmoid` フック＋Cargo.toml 依存結線
  ＋`docs/crates-io-publishing-order.md` §13 追補（併記 11 箇所・依存
  グラフ図）＋`docs/onnx-export-op-mapping.md` 追記。入力検証（weight
  rank 2・bias 長一致・名前一意性）を `Graph` 構築前に行う設計要件。
- facade 入口 issue: `from_sequential`・`api_surface.rs` 更新・parity
  テスト（Linear／ReLU は bit 一致、Sigmoid は承認結果に従う）・
  `docs/compat-api-scope.md` §1.3／§5 適用記録・`site/guides` の注意
  事項（§7 (a)〜(d)）。
- スコープ外のまま（`.claude/rules/out-of-scope-tracking.md` に基づき
  ユーザー承認後に別 issue で起票する候補の列挙。本 issue では起票し
  ない）: `value_info`／`TypeProto` 出力、Softmax／LayerNorm／GELU／
  Conv 等の対応拡大、`fandhe_ai::nn`（RNN 等）からの export、学習済み
  `DeviceParamStore`（常駐パラメータ）からの直接 export。

### 15.9 セキュリティ考慮（OWASP Top 10）

- **A08 ソフトウェア・データ整合性**: 公開クレートの依存グラフ変更
  （onnx-interop → autodiff）と facade 公開面追加を、コードより先に
  承認事項として記録しゲートする設計自体が A08 対策。公開面追加は
  必ず `api_surface.rs` の allowlist 更新と同一 PR にする要件を §15.5
  に明記。非対応層の無言 skip・部分 export を禁止し型付き `Err` で
  fail-closed。tolerance／baseline を緩めない。
- **A03 インジェクション／不正入力**: 橋渡しの入力は内部の学習済み
  モデルで非信頼バイト列ではないが、後続 issue への設計要件として
  `Graph` 構築前の形状検証（weight rank・bias 長）・テンソル名の一意性
  検証を課す。`to_path` のパス取り扱いは #2018 の既存契約（export
  成功後にのみ書き込み）を継承し新たな受け口は増やさない。
- **A06 脆弱・古いコンポーネント**: 外部依存の追加・更新なし。
  `Cargo.lock` の外部クレート集合不変。workspace 内依存結線も本 issue
  では行わない（記録のみ）。
- **A05／情報漏えい**: 本 doc に内部ホスト名・ユーザー名・絶対パス・
  メールアドレスを書かない。
- **A01／A04（設計）**: REQ-12（任意 `BackendOps` 注入口を公開しない）
  と無衝突であること、`ExportNode`／`ExportOp`／prost 型を facade
  公開シグネチャへ出さないことを §15.5 に明記済み。

### 15.10 承認依頼コメント文案（起票時点。実投稿・承認結果は §15.7 参照）

親 #2034 へ以下を投稿した（要点）:

> イシュー #2035 で `Sequential`／`nn` -> `ExportNode` 橋渡しの設計を
> 確定しました（`docs/facade-onnx-export-exposure-decision.md` §15）。
> コード変更はなく、設計記録・承認依頼のみです。
>
> 承認をお願いしたい事項（§15.7）:
> 1. 配置: `onnx-interop` が `fandhe_ai_autodiff::nn` を直接走査する
>    案（onnx-interop → autodiff の通常依存化。`version` 併記込み）
> 2. 対応層の初期範囲: `Linear`／`ReLU`／`Sigmoid` の 3 種
> 3. facade 公開面 1 件: `OnnxModel::from_sequential(&Sequential)`
> 4. `OnnxError::UnsupportedLayer` 新設の可否
> 5. Sigmoid の数値契約（推奨: REQ-2 統一複合判定を Sigmoid 込みモデル
>    に適用し、bit 完全一致は Linear／ReLU 限定と明記）
>
> 未承認の間は後続実装（#2036 等）の該当部分には着手しません。

## 16. 追補（イシュー #2036）: `onnx::export_nn` 実装完了（facade 未接続）

**§15.7 承認事項は 2026-09-18 に親 #2034 へユーザー承認コメントが
投稿され、項 1〜4 承認・項 5 保留で確定した**
（https://github.com/Fandhe-AI/fandhe-ai/issues/2034#issuecomment-5726738002）。
本追補時点（実装当初）では承認前だったため、issue #2036 本文の作業
項目指示に基づき安全側の範囲（Linear／ReLU の 2 種）へ縮小して実装
した。承認後の範囲もこの縮小版（項 2）と一致するため実装内容は不変。

### 16.1 実施した範囲・縮小した範囲

| §15.7 項 | 扱い | 理由 |
|---|---|---|
| 項 1（配置 (b)・`onnx-interop → autodiff` 通常依存化） | **実施**（2026-09-18 ユーザー承認済み。§15.7） | issue #2036 本文の第 1 作業項目・workspace 内 path 依存の追加のため `deps-policy.md` の外部承認フロー対象外（§15.2 の再導出根拠 1〜5 のとおり）。実装当初は承認前だったため PR 本文にその旨を明記していたが、その後 §15.7 の承認記録により解消 |
| 項 2／項 5（Sigmoid 対応・数値契約） | **対象外** | `Module` に `as_sigmoid` フックが無く（項 5 も未承認）判別手段が無いため。issue コメント記載の代替「Linear／ReLU の 2 種へ縮小」を適用 |
| 項 3（facade 公開面 `OnnxModel::from_sequential`） | 対象外（#2037） | 本 issue は facade 未接続限定 |
| 項 4（`OnnxError::UnsupportedLayer` 新設） | **実施**（variant 名は同一） | `ExportError` への追加は onnx-interop 内部の型付きエラーであり、facade `OnnxError` の variant 追加（§15.7 項 4 本来の対象）ではない。facade 側の `OnnxError::UnsupportedLayer` 新設可否は #2037 が対象のまま未承認 |

### 16.2 実装内容

- `crates/onnx-interop/Cargo.toml`: `[dependencies]` へ `fandhe-ai-autodiff
  = { path = "../autodiff", version = "=0.9.0" }` を追加し、旧
  `[dev-dependencies]` の同名 version 非併記エントリを削除（統合依存は
  dev としても有効）。`docs/crates-io-publishing-order.md` §13.5 に
  依存グラフ更新・実測記録を追記。
- `crates/onnx-interop/src/onnx/export_nn.rs`（新規）: `export_parts_
  from_layers`／`graph_from_layers`（`&[Box<dyn Module>] -> Graph`）。
  `Linear -> ExportOp::Gemm`（`alpha=beta=1.0`・転置なし）・
  `Relu -> ExportOp::Relu` の 2 種のみ対応。命名規約・検証順序・
  `autodiff` API バージョン制約の詳細は `docs/onnx-export-op-mapping.md`
  §7 を参照。
- `crates/onnx-interop/src/onnx/export.rs`: `ExportError` へ
  `EmptyModel`／`UnsupportedLayer { index, layer_kind }`／
  `InvalidLayerParameter { index, reason }`／
  `DuplicateTensorName { name }` の 4 variant を追加（`#[non_exhaustive]`
  のため非破壊）。
- `crates/onnx-interop/tests/onnx_export_nn.rs`（新規）: 2 層 MLP
  （`Linear -> Relu -> Linear`。小形状・CPU BLIS ブロックタイル境界
  〈KC=256〉を跨ぐ大形状の 2 パターン）の export → `interp::run` 出力を
  `Module::forward_host`（tape 不要経路）・`Module::forward`（tape 経路）
  という独立な 2 通りの手動 forward と bit 完全一致で検証。契約テスト
  （initializer 名・属性常時書き出し・決定性）・fail-closed テスト
  （空層列・未対応層・末尾未対応層での部分グラフ非返却）を含む計 10 件、
  いずれも pass。

### 16.3 検証結果

- `cargo test -p fandhe-ai-onnx-interop`: 全テスト pass（新規 10 件込み）。
- `cargo test -p fandhe-ai --test api_surface`: pass（facade 公開面は
  無変更）。
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`:
  green（`-p` スコープ単独実行時に現れる `backend-cuda` の dead-code
  警告は dev-dependency 経由の feature unification による HEAD 既存の
  環境依存アーティファクトで、本 issue 変更前の HEAD でも同一に再現する
  ことを確認済み・workspace 全体実行では発生しない）。
- `git diff --exit-code origin/main -- crates/facade crates/autodiff
  Cargo.toml Cargo.lock deny.toml`: 差分ゼロ（無変更確認）。
- `cargo tree -p fandhe-ai-onnx-interop --edges normal`:
  `fandhe-ai-autodiff` が normal 辺として現れ循環なし。
- 7 パッケージ一括 `cargo publish --dry-run --locked`（`env.
  RELEASE_CRATES` と同一順序）: 全 7 件成功。
- 新規 `unsafe`: 0 件。
