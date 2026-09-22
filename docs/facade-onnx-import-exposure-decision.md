# ONNX import（onnx-interop）の facade 公開可否の設計判断記録（#1652）

イシュー #1652「ONNX import（onnx-interop）の facade 公開可否を判断する」に対応する。親: #1629（Tier 2）・ルート: #1570。

本ドキュメントは **コード変更を伴わない設計記録のみ**である。`crates/**`・`docs/spec/`（正本 submodule）・依存（`Cargo.toml`／`Cargo.lock`）・ガードレール閾値・数値一致許容誤差（tolerance／baseline）はいずれも変更しない。仕様変更が必要になった場合は正本である `docs/spec/`（fandhe-ai-spec リポジトリ）側への提案とし、本リポでは `docs/spec/` を編集しない。

基準コミット: `da8e52c23c451055ced24ec4aedc688686ea56e0`（2026-09-14）。`file_path:line` は同コミット時点のもの。後続の変更で行番号がずれる可能性があるため、参照する際は当該コミット、または近傍のコミットで再確認すること。

## 1. 背景

対応する PyTorch 機能: `torch.onnx.load` 相当（ONNX モデルの読み込み・推論実行。逆方向の `torch.onnx.export` は #1653・#1775 の対象）。

`docs/compat-feature-gap.md` §1.9「リポ内非公開（`onnx-interop`。crates.io 非公開・facade から到達不可）」・§2「ONNX import」行（`docs/compat-feature-gap.md:347`）は、`onnx::interp`（22 種のオペをホスト参照実装として持つ推論専用グラフ解釈器。autograd 未接続）が facade から到達不可であることを明記している。一方 spec REQ-7（`docs/spec/04-requirements.md:172-186`）のユーザーストーリー「PyTorch 学習済みモデルを Rust 側へ移行したい」には、現状サポートされる唯一の公開入口である facade から到達できる ONNX 読み込み手段が存在しない。

spec REQ-9 の 2026-09-12 追記（`docs/spec/04-requirements.md:232`）は Tier 2（長尾）の一覧に「ONNX import 公開／export」を明示列挙している（`docs/compat-api-scope.md:249`）。**DDP（#1628）・量子化（#1627）と異なり、この行には除外事項への従属も格上げ条件表も付いていない**。ONNX import の facade 公開判断は、こうした構造的な足かせがない状態から独立に評価できる。

`crates/onnx-interop/src/onnx/export.rs`（#1772）冒頭コメントは「facade 公開は #1775（#1652 の判断待ち）」と明記しており、本判断は #1775 の前提でもある（`crates/onnx-interop/src/onnx/export.rs:18-19`）。

## 2. 現状のコード事実

| 事実 | 出典 |
|---|---|
| `facade`（`fandhe-ai`）は crates.io 公開クレート（`publish = true`）。`onnx-interop` はルート `[workspace.package]` の `publish = false`（`publish.workspace = true` により継承）の非公開 5 クレートの 1 つ（`onnx-interop`／`guardrail`／`self-repair`／`bench-harness`／`docs-site`） | `crates/facade/Cargo.toml:1-6`・`crates/onnx-interop/Cargo.toml:1-6`・`Cargo.toml:84` |
| 公開 6 クレートの `[dependencies]`（通常依存）に非公開クレートが現れないことは `docs/crates-io-publishing-order.md` §6 で実測確認済み（dev-dependency の path のみ依存に限られる）。**facade が `onnx-interop` を通常依存に持つと `cargo publish` が成立しない**（公開クレートの `Cargo.toml` に非公開クレートへの path 依存を含めることになるため） | `docs/crates-io-publishing-order.md:169-187` |
| 一括リリースは `release-all.yml` の `env.RELEASE_CRATES`（`fandhe-ai-tensor-core fandhe-ai-autodiff fandhe-ai-backend-cpu fandhe-ai-backend-cuda fandhe-ai-backend-metal fandhe-ai` の 6 クレート固定・依存順）で駆動する | `.github/workflows/release-all.yml:90-92` |
| `onnx-interop` の `tensor-core` 依存は `{ path = "../tensor-core" }` のみで `version` 併記なし（公開 6 クレートの規則は `{ path = "...", version = "=x.y.z" }` 併記。公開する場合は同規則への追従が必要） | `crates/onnx-interop/Cargo.toml:22`・`docs/crates-io-publishing-order.md` §5 |
| `onnx::interp` の公開 API: `pub fn run(&Graph, HashMap<String, Value>) -> Result<HashMap<String, Value>, InterpError>`（`interp.rs:793`）・`pub enum Value { F32(Tensor<f32>), I64(Tensor<i64>), Bool(Tensor<bool>), F16(Tensor<f16>) }`（`interp.rs:62`）・`pub enum InterpError`（`#[non_exhaustive]`。`interp.rs:71-73`） | `crates/onnx-interop/src/onnx/interp.rs:62`・`:71-73`・`:793` |
| `onnx::graph::Graph { pub nodes: Vec<NodeProto>, pub initializers: HashMap<String, RawTensor>, pub inputs: Vec<String>, pub outputs: Vec<String> }`。`NodeProto` は `onnx::proto` の手書き `#[derive(prost::Message)]` 型 | `crates/onnx-interop/src/onnx/graph.rs:183-187` |
| decode 入口は `onnx::proto::ModelProto::decode`（`prost::Message` トレイトの `use` を利用者側で要する） | `crates/onnx-interop/tests/onnx_interp.rs:22-24` |
| `onnx::interp` は `ops::*`（`Tensor<f32>` 上のホスト純関数群）へディスパッチするのみで、`BackendOps`／`Device` を一切参照しない（grep 該当なし。doc コメント中の `eval::relu` 言及のみ） | `crates/onnx-interop/src/onnx/interp.rs`・`crates/onnx-interop/src/ops/activation.rs` |
| 対応オペは 22 種（8 種＋14 種）。`transformer.onnx` end-to-end テストは REQ-7 判定式 `abs_err/(|ref|+1e-6) <= 1e-3` で pass（`#[ignore]` なし） | `docs/compat-feature-gap.md:169-173`・`crates/onnx-interop/src/onnx/interp.rs:12-18` |
| 数値契約は REQ-7 の別指標であり、REQ-2 統一複合判定（相対誤差 1e-3 未満 または絶対誤差 1e-5 未満）とは混同禁止と明記済み | `crates/onnx-interop/tests/onnx_interp.rs:11-16`・`.claude/rules/coding-rust.md` |
| facade の公開面ガード: `pub use` に `Tape`／`BackendOps`／`new_with_ops` を含めない・`pub fn` が `BackendOps` を引数として直接受け取らない（REQ-12） | `crates/facade/tests/api_surface.rs:57-110` |
| facade は `half::f16` を公開型として出していない（`Value::F16` の素通しは新規露出になる） | `docs/compat-feature-gap.md:52-53` |
| `onnx::export`（#1772）は「facade 公開は #1775（#1652 判断待ち）」と明記し、内部 API のみを提供する | `crates/onnx-interop/src/onnx/export.rs:11-19` |
| `docs/compat-api-scope.md` §1.3 の Tier 2 表は「ONNX import 公開／export \| #1629」の 1 行のみで、DDP／量子化のような除外事項への従属記載はない。§5 に #1628 型（DDP）の適用記録の先例がある | `docs/compat-api-scope.md:249`・`:456-462` |
| crates.io 名 `fandhe-ai-onnx-interop`・`fandhe-ai-interop` はいずれも未登録（HTTP 404。2026-09-14 実測・読み取りのみ） | `curl https://crates.io/api/v1/crates/<name>`（本 PR 実施） |
| 並列実行中の他 worktree・open PR に `onnx-interop`／#1754／#1616 を触るものは無い（2026-09-14 時点） | `git worktree list`・`gh pr list` |

## 3. 契約整理

### 3.1 「facade から公開する」の構造的な意味

`facade` は唯一のサポートされる公開 API 面である（REQ-9 2026-08-08 追記・`docs/compat-api-scope.md` §0）。facade は crates.io 公開クレートであり、公開クレートの `Cargo.toml` は非公開クレートへの通常依存を持てない（2 節の実測事実）。したがって「facade から `onnx-interop` を公開する」は、単なる再エクスポートの追加ではなく、**`onnx-interop` 自体を 7 クレート目として crates.io へ公開すること**と構造的に等価である。

これは 2026-09-12 の #1652 承認コメント（facade 公開面拡張。`docs/compat-api-scope.md` §5 手続き）が対象とする「公開済みクレート内での公開面拡張」とは異なる、**別個のユーザー承認事項**である:

- crates.io での命名確定（先例: #878／#879 の `fandhe-ai` prefix 命名判断）
- 公開対象クレート数の拡大（`docs/crates-io-naming-decision.md`・`docs/crates-io-publishing-order.md` が前提とする「公開 6 クレート」という構成そのものの変更）
- `release-all.yml`／`release.yml` の `RELEASE_CRATES` 変更（CI ワークフロー変更のため `.claude/rules/security.md` の監査対象）
- `onnx-interop/Cargo.toml` の `tensor-core` 依存への `version = "=x.y.z"` 併記（公開クレート規則への追従）

### 3.2 REQ-12・薄いラッパー原則との整合

`onnx::interp::run` は `ops::*`（`Tensor<f32>` 上のホスト純関数）へディスパッチするのみで `BackendOps`／`Device` を一切参照しない。したがって facade へ公開しても REQ-12（「任意 `BackendOps` 実装を注入できる公開 API を設けない」）・`api_surface.rs` の機械検査（`Tape`／`BackendOps`／`new_with_ops` を再エクスポートしない）と衝突しない——公開する場合の実装形（3.3 の案 B）は `BackendOps` を一切引数に取らない設計で成立する。

一方、`onnx::proto`（`prost::Message` derive 型）・`onnx::graph::Graph`（`NodeProto` を `pub` フィールドで保持）・`Value::F16`（`half::f16` の素通し）を facade の公開面へそのまま出す案（3.3 の案 A）は、薄いラッパー原則（`docs/compat-api-scope.md` §3）・`=x.y.z` 完全固定の下での SemVer 安定性（`prost`・`half` のバージョンアップが facade の破壊的変更に直結する）に反する。

### 3.3 REQ-7／REQ-2 の指標分離

REQ-7 の受け入れ判定式（`abs_err/(|ref|+1e-6) <= 1e-3`）は REQ-2 統一複合判定（バックエンド間数値一致）とは別指標であり、両者を混同してどちらかを緩和しない契約が既存テストのコメントで明記されている（2 節）。`onnx::interp` を facade へ公開しても、この指標分離自体は変更しない。`onnx::interp` は現状 CPU ホスト実行のみで `BackendOps` を経由しないため、公開後も REQ-2 parity（バックエンド間数値一致）の対象にはならない——`BackendOps` 経由化（GPU 実行化）を将来行う場合にのみ REQ-2 が発生する、独立した後続課題である。

## 4. 案比較

| 案 | 概要 | 判定 | 主な理由 |
|---|---|---|---|
| A: 素の再エクスポート（`fandhe_ai::interop::onnx::{proto, graph, interp}`） | 既存 API をそのまま `pub use` | 不採用 | `prost`（`ModelProto::decode`・`Graph.nodes: Vec<NodeProto>`）と `half::f16`（`Value::F16`）が facade 公開 API へ漏れ、`=x.y.z` 固定下で `prost`／`half` のバンプが facade の SemVer 破壊になる。薄いラッパー原則（`docs/compat-api-scope.md` §3）にも反する |
| **B: 薄いラッパー型**（`fandhe_ai::interop::onnx::{OnnxModel, OnnxValue, OnnxError}` 相当。`OnnxModel::from_bytes(&[u8])`／`from_path`・`run(feeds) -> outputs`） | `prost`／proto 型を公開面に出さない専用ラッパー | **方針として推奨（現時点では未着手）** | REQ-9 Tier 2 に列挙済み・REQ-7 ユーザーストーリーの唯一のサポート経路になる・`onnx::interp` は `BackendOps` 非依存のため `api_surface.rs` ガード／REQ-12 と無衝突。ただし案 A と同じく §3.1 の publish 前提（onnx-interop の crates.io 公開）が未承認のため**現時点では着手不可** |
| C: `onnx-interop` のコードを `facade` クレートへ移設 | 依存構造を変えず公開面だけ確保 | 不採用 | REQ-1 の 10 クレート構成（`docs/spec/04-requirements.md:50`）に反する。クレート境界の恣意的な崩しは他クレートへも波及する |
| D: `fandhe-ai-onnx-interop` を facade を経由しない独立の公開面として公開 | facade に一切結線しない第 2 の公開面 | 不採用 | 「facade が唯一のサポートされる公開 API 面」（REQ-9 2026-08-08 追記・`docs/compat-api-scope.md` §0）に反する |
| E: 段階 0 — 非公開のまま維持し再開条件を記録するのみ | 現状維持 | **現状の確定状態**（B の前提承認が得られるまで） | `site/guides/interop.md` 等の既存記述と整合・既存契約への影響ゼロ |

## 5. 推奨（方針／現状の二層）

- **方針**: 公開する（案 B）。理由: DDP／量子化と異なり spec 上の除外事項に従属していない・REQ-7 のユーザーストーリーに現状公開入口が皆無・`onnx::interp` が `BackendOps` 非依存で REQ-12 と無衝突。
- **現状**: 非公開・未実装。ブロッカーは §3.1 に列挙した「onnx-interop の crates.io 公開」というユーザー承認事項（＋命名確定）であり、これは 2026-09-12 の #1652 承認コメントの範囲には含まれない。

この二層は #1628（DDP。除外事項への従属で Won't 相当）とは性質が異なる点に注意する: DDP は「除外事項の格上げがない限り実装着手不可」という**制度的な足かせ**が理由だが、ONNX import 公開は制度的な足かせはなく、**単に crates.io 公開という別個の承認手続きが未完了**という状態にすぎない。

## 6. 再開条件・依存 issue の読み替え

### 6.1 再開条件

1. `onnx-interop` の crates.io 公開承認: 命名（候補 `fandhe-ai-onnx-interop`。2026-09-14 時点で未登録確認済み）・`publish = true`（`publish.workspace = true` からの上書き。#881 での公開 6 クレート化と同型の変更）・`onnx-interop/Cargo.toml` の `tensor-core` 依存への `version = "=x.y.z"` 併記・`release-all.yml` の `RELEASE_CRATES` への追加・per-crate README（#882 先例）・`docs/crates-io-naming-decision.md`／`docs/crates-io-publishing-order.md` の更新。
2. ラッパー API 形状（案 B）の確定: `OnnxValue` の dtype 表現（`half::f16` を素通しするか独自表現へ変換するか）・エラー型の `#[non_exhaustive]` 方針・`crates/facade/tests/api_surface.rs` への公開面検査追加。
3. 実装 issue の起票（本 PR では起票しない。承認取得後に `.claude/rules/out-of-scope-tracking.md` に従い起票する）: (i) publish 準備、(ii) facade ラッパー実装＋`docs/compat-api-scope.md` §1.3 更新＋`api_surface.rs` 拡張、(iii) 学習可能化（`Tape` への変換層）は別系統の issue。

### 6.2 依存 issue への読み替え

- **#1775（ONNX export の facade 公開）**: 「公開しない」という結論ではないため close しない。§6.1 の publish 承認が得られるまで blocked として扱う。承認が却下された場合に限り close し、親 #1653 へ判断結果を記録する。export 固有の設計記録（export 元の限定・橋渡しの配置候補・facade `api_surface.rs` guard テスト等）は `docs/facade-onnx-export-exposure-decision.md`（#1775 で完了）を参照する。
- **#1653 配下 #1773／#1774**: `onnx::export` の内部実装（`Graph -> GraphProto` の組み立て・op 属性の逆マッピング・roundtrip テスト）であり、facade 公開可否とは独立に進行できるため本判断の影響を受けない。
- **#1754（safetensors save／load の facade 再公開）**: `onnx-interop` は ONNX と safetensors の両方を扱う同一クレートであり、safetensors 側の facade 再公開も §3.1 と同じ publish 前提を共有する。本 doc の判断を再度導出せず、**承認依頼は「onnx-interop の crates.io 公開」として 1 回にまとめることを推奨**する（`st_load`／`st_save` と `onnx` モジュールは同一クレートに同居するため）。

### 6.3 記録すべき制約（将来のラッパー実装が満たすべき設計要件）

- (a) `onnx::interp` はホスト CPU 実行のみで `Device` を無視する。将来のラッパー doc には「`Device::Cuda`／`Device::Metal` を渡しても GPU 実行にはならない」旨を明示する設計要件を残す。
- (b) 数値契約は REQ-7 判定式のまま・REQ-2 parity は `BackendOps` 経由化を将来行う場合にのみ発生する（3.3 節）。本判断のスコープ外。
- (c) 学習可能化（`Tape`／`Var` への変換層）は別 issue（`docs/compat-feature-gap.md:347`「学習させるなら」の注記どおり）。
- (d) `Value::F16` の公開表現（`half::f16` を facade の公開型として素通しするか否か）は #1626（dtype dispatch）の進捗と整合させる副次判断として、ラッパー API 形状確定時（§6.1 の 2）に扱う。

## 7. セキュリティ考慮（OWASP Top 10。docs のみのため留意点として記録）

- **A03 インジェクション／不正入力**: ONNX は非信頼の外部フォーマットである。`onnx::graph`／`onnx::proto` は dims 非負・`checked_mul`・データ長一致を復号前に検証し fail-closed で失敗する設計になっている（`crates/onnx-interop/src/onnx/graph.rs` の decode 経路）。将来ラッパーを facade に置く場合も同じ契約を維持し、`checked_mul` が個別要素の overflow は防ぐが総メモリ量までは制限しないため、要素数・総バイト数の上限設定を設計要件として追加検討する必要がある（本 issue のスコープ外・将来のラッパー実装 issue で扱う）。
- **A06 脆弱コンポーネント**: 本判断は新規依存を追加しない。`safetensors`／`prost` は既に許容依存区分（`.claude/rules/deps-policy.md`）で `=x.y.z` 固定済み・`docs/license-matrix.md` に記載済み。`onnx-interop` を publish する場合も追加の外部依存は発生しない。
- **A08 データ整合性**: 将来の `release-all.yml`（`RELEASE_CRATES` 追加）変更は CI ワークフロー変更として `.claude/rules/security.md` に基づき security-auditor の監査対象になることを承認事項に明記する。
- **A05／情報漏えい**: 本 doc に内部ホスト名・秘密情報は含めない。

## 8. スコープ外（実装は含めない。`.claude/rules/out-of-scope-tracking.md` に基づき承認後に別 issue で起票）

- `onnx-interop` の crates.io 公開作業そのもの（命名確定・`publish = true`・`RELEASE_CRATES` 追加・README 整備）。
- facade ラッパー実装（`OnnxModel`／`OnnxValue`／`OnnxError`）・`crates/facade/tests/api_surface.rs` の拡張。
- 学習可能化（`Tape`／`Var` への変換層）・`BackendOps` 経由の GPU 実行化（REQ-2 parity を伴う）。
- ONNX export の facade 公開（#1775。本判断の読み替え〈§6.2〉に従う）。
- safetensors 再公開の API 形状（#1754。publish 前提のみを共有し、API 形状自体は別途判断する）。

## 9. 承認事項（列挙のみ。本 PR ではユーザー承認を得ない）

1. `onnx-interop` の crates.io 公開そのもの（命名・`publish = true`・`RELEASE_CRATES` 追加を含む一式）。
2. 上記 1 が承認された場合の、facade ラッパー API 形状（案 B）の確定。
3. `Value::F16`／`half::f16` を facade の公開型として素通しするか、独自表現へ変換するかの判断。

## 10. 出典一覧

| 出典 | 内容 |
|------|------|
| `crates/facade/Cargo.toml` | facade の crates.io 公開設定 |
| `crates/onnx-interop/Cargo.toml` | onnx-interop の非公開設定・依存構成 |
| `Cargo.toml` | workspace 既定 `publish = false` |
| `docs/crates-io-publishing-order.md` | 公開 6 クレートの依存規則・非公開クレートとの依存が公開を阻害しないことの実測確認 |
| `.github/workflows/release-all.yml` | `RELEASE_CRATES` の一覧 |
| `crates/onnx-interop/src/onnx/interp.rs` | `onnx::interp::run`／`Value`／`InterpError` の公開 API 定義 |
| `crates/onnx-interop/src/onnx/graph.rs` | `Graph` 構造体・decode 経路の境界検査 |
| `crates/onnx-interop/src/onnx/proto.rs` | 手書き `prost::Message` derive 型 |
| `crates/onnx-interop/src/onnx/export.rs` | ONNX export の facade 公開待ち状態の明記 |
| `crates/onnx-interop/tests/onnx_interp.rs` | REQ-7 判定式と REQ-2 の指標分離の明記 |
| `crates/facade/tests/api_surface.rs` | REQ-12 公開面ガード（`Tape`／`BackendOps`／`new_with_ops` 非再エクスポート） |
| `docs/compat-api-scope.md` | Tier 2 一覧（ONNX import 公開／export の除外事項非従属）・§5 適用記録の先例 |
| `docs/compat-feature-gap.md` | §1.9／§2 の ONNX import ギャップ記述 |
| `docs/spec/04-requirements.md` | REQ-7・REQ-9（2026-09-12 追記）・REQ-12 |
| `docs/facade-multi-gpu-ddp-decision.md` | 同型の設計判断記録（構成テンプレートの先例） |

## 11. 追補（イシュー #1963・2026-09-17）: publish 承認取得済み

§9 承認事項 1（「onnx-interop の crates.io 公開」自体の承認）は
**取得済み**（2026-09-17・イシュー #1963 承認コメント。`docs/crates-io-
publishing-order.md` §13・`docs/crates-io-naming-decision.md`「7 件目」節）。
`onnx-interop` は `fandhe-ai-onnx-interop` として 7 クレート目の公開準備が
完了し、次回リリースサイクルで `release-all.yml` を通じて crates.io へ
公開される見込みである。

ただし本追補は「facade からの ONNX import ラッパー実装」自体を承認する
ものではない。§9 承認事項 2・3（`OnnxModel`／`OnnxValue` 等のラッパー API
形状・`Value::F16` の扱い）は未承認のまま残り、facade 公開面（段階 0）は
継続する。`crates/facade/tests/api_surface.rs` の否定ガード（facade が
`onnx-interop` へ通常依存しないことの機械固定）も維持する。ラッパー実装の
起票はユーザー承認を得てから行う。

## 12. 実装記録（イシュー #2017・2026-09-17）

§9 承認事項 2・3 は issue 本文・2026-09-17 承認コメントで取得済みとなり、
以下のとおり実装した。§11 の「段階 0 は継続」は **import に限り解消**
（export は §8 のとおり引き続き段階 0。**safetensors は #2019 で解消済み**
——`fandhe_ai::interop::safetensors`。`docs/facade-safetensors-exposure-
decision.md` §11 参照）。

### 12.1 公開面

- `fandhe_ai::interop::onnx::{OnnxModel, OnnxValue, OnnxError}`
  （`crates/facade/src/interop/{mod.rs, onnx.rs}`。新規モジュール
  `pub mod interop;` を `crates/facade/src/lib.rs` に追加）。
- `OnnxModel::{from_bytes, from_path, run}` の 3 メソッドのみ（§4 案 B の
  設計どおり）。`input_names` 等の追加アクセサは承認範囲外のため未実装。
- `OnnxValue`（`#[non_exhaustive]` ではない exhaustive enum。承認文言
  どおり）: `F32(Tensor<f32>)`／`I64(Tensor<i64>)`／`Bool(Tensor<bool>)`／
  `F16(Tensor<half::f16>)`。§6.3 (d) の判断: `half::f16` を**素通し**する
  （facade は `half` を再エクスポートしない。`TypedOps<half::f16>`
  〈#1939〉と同じ扱い。利用者は `half` へ直接依存し version を
  `fandhe-ai` の固定版に合わせる）。
- `OnnxError`（`#[non_exhaustive]`）: `Io`／`Decode`／`UnsupportedDataType`／
  `UnsupportedOp`／`MissingFeed`／`UnknownFeed`／`InvalidModel`／
  `Execution` の 8 variant。

### 12.2 `OnnxError` の設計判断（自己完結型を採用）

内部クレートのエラー enum（`GraphError`／`InterpError`）をそのまま
ペイロードに持たせる案（`Graph(GraphError)`／`Interp(InterpError)`）は
**不採用**とした。理由:

1. 承認済み公開面は `OnnxModel`／`OnnxValue`／`OnnxError` の 3 型のみ。
   内部エラー enum をペイロードに含めると facade 利用者がそれらを
   名指しできず、「型付き `Err` を fail-closed に拒否する」という
   受け入れ条件（AC2）を facade 単独では満たせない。
2. 名指し可能にするには内部エラー型自体の再エクスポートが必要になり、
   承認範囲（3 型のみ）を超える。
3. `GraphError`／`InterpError` は `onnx-interop` のエラー面がそのまま
   facade の SemVer 面へ侵入することになり、薄いラッパー原則
   （`docs/compat-api-scope.md` §3）に反する。

代わりに `OnnxError` を「名指し variant ＋ ワイルドカード（内部
`Display` 文字列を保持する `InvalidModel`／`Execution` への退避）」の
自己完結型として定義した（`crates/facade/src/interop/onnx.rs::
map_graph_error`／`map_interp_error`）。`InterpError::Graph(GraphError::
UnknownDataType{..})`（`Constant` 属性テンソル decode 経由）も
モデル構築時と同じ写像規則（`OnnxError::UnsupportedDataType`）を適用する
よう統一した。

### 12.3 `prost` 非露出の機構

facade は `prost` へ直接依存せず（Cargo.toml に追加していない）、
`onnx-interop` 側に新設した `onnx::proto::{decode_model, encode_model}`
（`ModelProto::decode`／`encode_to_vec` への薄い委譲。呼び出し側が
`prost::Message` を `use` しなくても呼べる入口）を経由する。facade は
戻りの `prost::DecodeError` を `to_string()` するだけで `prost` 型を
名指ししない。

### 12.4 guard テストの差し替え（`crates/facade/tests/api_surface.rs`）

§6.1「実装 issue の起票」で予告したとおり、旧負ガード 2 件を承認済み
依存形状の正ガードへ差し替えた:

- `facade_does_not_depend_on_unpublished_onnx_interop` →
  `facade_depends_on_onnx_interop_only_in_approved_shape`
  （`fandhe-ai-onnx-interop` が素の `[dependencies]` に丁度 1 回・
  `path = "../onnx-interop"`・`version` が他の公開 path 依存と同一・
  `git`／`registry`／`branch`／`rev`／`tag`／`optional`／`features`
  キーを含まない、を fail-closed に検査する）。回帰テスト 5 件
  （承認形状 pass・ヘッダコメント付き build-dependencies 検出・
  version 欠落／ドリフト検出・`git` 取得元差し替え検出・
  `[dependencies.<name>]` テーブル形式検出）を追加した。
- `facade_sources_do_not_reference_onnx_interop` →
  `facade_sources_reference_onnx_interop_only_in_interop_module`
  （`onnx_interop`／`fandhe_ai_onnx_interop` 等への参照が `src/interop/`
  配下に閉じていることを検査する。`src/interop/` 以外からの参照は
  引き続き禁止）。
- 新設 `interop_module_exposes_only_approved_onnx_surface`（`src/
  interop/mod.rs` の公開アイテムが `pub mod onnx;` のみ・`src/interop/
  onnx.rs` の `pub` シグネチャに `ModelProto`／`NodeProto`／`prost::`／
  `onnx::graph::Graph` が現れないことを検査）・`onnx_import_types_are_
  reachable_via_facade`（3 型の名指し・`OnnxError` の `#[non_exhaustive]`
  ワイルドカード `match` をコンパイル時に固定）を追加した。

### 12.5 テスト構成

- `crates/facade/tests/interop_onnx_import.rs`（**`fandhe_ai` と `std`
  のみ import**。facade 単独到達性の直接的な裏付け）: `model.onnx`
  （8 サンプル・`onnx_reference.json` からの転記）・`slice_repro.onnx`
  （動的境界 Slice パターン。5×6 → 5×4）を `from_path`／`from_bytes`
  両方で REQ-7 判定式（`abs_err/(|ref|+1e-6) <= 1e-3`）突合。負例
  （壊れたバイト列・空バイト列・存在しないパス・feed 欠落・未知
  feed 名）・`OnnxError` の `#[non_exhaustive]` ワイルドカード `match`
  を検証。9 テストすべて green。
- `crates/facade/tests/interop_onnx_internal_parity.rs`（**facade と
  `fandhe_ai_onnx_interop` の両方を意図的に import**）: AC1（facade 経由
  と内部クレート直接呼び出し〈`decode_model → build_graph → interp::run`〉
  の出力が `to_bits()` 完全一致であることを `model.onnx`・
  `slice_repro.onnx` で確認）・AC2（`onnx::proto::*` の struct リテラル
  ＋ `encode_model` で合成した壊れたモデル〈未対応 op `LSTM`・
  `data_type=999` の initializer・負 dims〉が `OnnxError::UnsupportedOp`／
  `UnsupportedDataType`／`InvalidModel` へ正しく写像されることを確認）・
  `Cast(to=FLOAT16)` の合成モデル出力が `OnnxValue::F16` になることを
  確認。6 テストすべて green。
- 正しさは AC1（bit 完全一致）・AC2（負例の型付き `Err`）で検証済み。
  移動・コピーのみで再計算を挟まない設計（`onnx_value_to_interp`／
  `interp_value_to_onnx` は move）のため bit 一致は構造的に成立する。

### 12.6 §6.3 制約の充足状況

- (a) 推論専用・ホスト CPU 実行のみ・autograd 未接続——doc comment
  （`crates/facade/src/interop/onnx.rs` モジュール冒頭）に明記。
- (b) 数値契約は REQ-7 判定式のまま・REQ-2 parity 非経由（`BackendOps`
  を一切参照しない）。
- (c) 学習可能化（`Tape`／`Var` への変換層）は未実装のまま。
- (d) `Value::F16` は素通し方針で確定（12.1 参照）。

### 12.7 スコープ外（実装しない・issue は起票しない）

- ONNX export（**#2018 で実装済み**。`OnnxModel::to_bytes`／`to_path`・
  `docs/facade-onnx-export-exposure-decision.md` §14）は本 issue の
  スコープ外だったが別 issue で公開済み。safetensors 側も #2019 で
  facade 公開が完了済み（本 issue とは別ファイル
  `docs/facade-safetensors-exposure-decision.md` §11 参照）。`Sequential`／
  `nn` → `ExportNode` 橋渡し・学習可能化・`BackendOps` 経由の GPU 実行化は
  引き続きスコープ外。
- 入力総バイト数／要素数の明示上限導入（§7 A03 の懸念事項。値の決定に
  承認が必要なため見送り。既存の fail-closed 境界〈`build_graph` の長さ
  整合検査〉のみで運用）。
- `OnnxModel::input_names` 等の追加アクセサ・`OnnxValue` の
  `#[non_exhaustive]` 化（承認文言に無いため見送り）。

## 13. 追補（イシュー #2077・2026-09-22）: `BackendOps` 経由の GPU 実行 opt-in

12.6(a) の「`BackendOps`／`Device` 非経由（GPU 実行にはならない）」は
**opt-in（既定 OFF）で解除済み**。`OnnxModel::run` は既定ではこの節の
記述どおり不変（bit 完全一致）のままだが、`fandhe_ai::
set_cuda_onnx_gpu_execution_enabled`／`set_metal_onnx_gpu_execution_enabled`
（プロセスワイド・既定 `false`）を明示的に有効化した場合のみ `BackendOps`
経由の device 実行を試みる。設計判断・op 別結線表・parity 契約・承認事項の
詳細は `docs/onnx-gpu-execution-decision.md` を正とする（本節は追補ポイン
タのみ）。
