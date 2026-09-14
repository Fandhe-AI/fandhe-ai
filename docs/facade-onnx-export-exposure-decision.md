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
| **(b) `onnx-interop` 内で `fandhe_ai_autodiff::nn` の型（`Linear` 等）から `ExportNode` 列を構築** | `onnx-interop` は既に `autodiff` を dev-dependency に持ち（`Cargo.toml`）、通常依存化も非公開クレート間の変更のためユーザー承認を要さない先例（`docs/crates-io-publishing-order.md` の対象は「公開クレートの依存」に限る） | **推奨**。ただし `compat::Sequential` は facade 側の型のため直接は受け取れず、facade 側で `Sequential` -> 層列への薄い分解層が別途要る（facade ラッパー実装〈§4 の案 B〉と同時に確定させる） |
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
3. `Sequential`／`nn` -> `ExportNode` 橋渡しの配置（3.2 の (b) を推奨するが最終確定は未了）。

## 11. 出典一覧

- `docs/facade-onnx-import-exposure-decision.md`（#1652。publish 前提・案 A〜E 比較の出典）
- `docs/onnx-export-op-mapping.md`（#1773／#1774。export op マッピング・roundtrip テストの契約）
- `docs/compat-api-scope.md` §1.3・§5（Tier 2 列挙・#1652 適用記録）
- `docs/crates-io-publishing-order.md` §6（公開クレートの依存規則）
- `crates/onnx-interop/src/onnx/export.rs`・`export_ops.rs`（本判断が参照するコード事実）
