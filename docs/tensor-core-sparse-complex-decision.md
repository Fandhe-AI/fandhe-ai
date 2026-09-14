# sparse／complex テンソルの非対応を明文化する設計判断記録（#1633）

イシュー #1633「sparse／complex テンソルの非対応を明文化する」に対応する。親: #1573（Tier 2）・ルート: #1570。

本ドキュメントは **コード変更を伴わない設計記録のみ**である。`crates/**`・`docs/spec/`（正本 submodule）・依存（`Cargo.toml`／`Cargo.lock`）・ガードレール閾値・数値一致許容誤差（tolerance／baseline）はいずれも変更しない。仕様変更が必要になった場合は正本である `docs/spec/`（fandhe-ai-spec リポジトリ）側への提案とするが、本イシューでは §3 のとおり既に整合が確認できており提案は不要である。

基準コミット: `35104de681d9920aeab8b9bcdd59a9725a58ad91`。`file_path:line` は同コミット時点のもの。後続の変更で行番号がずれる可能性があるため、参照する際は当該コミット、または近傍のコミットで再確認すること。

## 1. 背景

対応する PyTorch／TensorFlow 機能:

- sparse: `torch.sparse_coo_tensor`／`torch.sparse.mm`／`Tensor.to_sparse()`（PyTorch）、`tf.sparse.SparseTensor`／`tf.sparse.sparse_dense_matmul`（TensorFlow）。
- complex: `torch.complex64`／`torch.complex128`・`torch.fft`（PyTorch）、`tf.complex64`／`tf.complex128`・`tf.signal.fft`（TensorFlow）。

spec REQ-9 の 2026-09-12 追記は、互換 API 層の対象範囲を PyTorch／TensorFlow の機能網羅へ Tier 1／Tier 2 の 2 段階で拡張した際、sparse／complex テンソルを一貫して「引き続き対象外」の列挙に含めている（`docs/spec/04-requirements.md:233`・改定履歴 `:432`。submodule commit `c5cf1ed5a547f47dd60d168c926bd8d376ee31da`）。`docs/compat-api-scope.md` §2「対象外」は「sparse／complex テンソル（非対応の明文化は #1633）」として本イシューへの forward reference を既に持っており、`docs/compat-feature-gap.md`（fandhe-ai 公開面の実装状況スナップショット）には対応する行がない。本ドキュメントはこの forward reference を解消し、非対応の判断根拠をコード事実に基づいて記録する。

出典として issue 本文が挙げる claude.ai artifact（低レイヤー診断・敗因分析）は本リポジトリ内に実体を持たない（`find` で該当なしを確認済み）ため、名称のみで参照し URL は転記しない。

## 2. 現状のコード事実

| 事実 | 出典 |
|---|---|
| `Tensor<T>` は厳密に dense: `Storage { data: Vec<T> }` + `offset`／`shape`／`strides`（行優先 + strides。stride 0 ブロードキャスト対応）。レイアウト種別（COO／CSR 等の sparse 表現）を表す型・enum は存在しない | `crates/tensor-core/src/tensor.rs:33-57` |
| `Element` trait は **open**（sealed ではない。`Copy + Send + Sync + Debug + PartialEq + 'static` + `zero()`/`one()`）。実装は `f32`/`f64`/`i32`/`i64`/`bool`/`half::f16`/`half::bf16`。外部クレートが complex newtype（例: `struct MyComplex { re: f32, im: f32 }`）へ `impl Element` して `Tensor<MyComplex>` を**生成すること自体は型システム上可能**だが、`BackendOps`・演算カーネル・VJP のいずれも `Element` ではなく後述の sealed `Scalar` に境界付けられているため、算術経路（`+`／`matmul`／勾配伝播）には一切到達しない | `crates/tensor-core/src/element.rs:1-84` |
| `Scalar` trait は `private::Sealed` で封印され、実装対象は `f32`/`f64`/`half::f16`/`half::bf16` の 4 型に限定（外部クレートは実装できない）。`ScalarDType` は `#[non_exhaustive]` 4 variant（`F32`/`F64`/`F16`/`Bf16`）。`dispatch::DType` は `F32`/`F16` のみ。`BackendOps` は f32 固定で、`TypedOps<T>` 拡張（イシュー #1687）も上記 4 実数型限定 | `crates/tensor-core/src/element.rs:111-187`・`crates/tensor-core/src/dispatch.rs:31-39`・`docs/backend-dtype-dispatch-design.md` |
| `Var<'t>` は tape 上の f32 ノード id（`tape: &'t Tape, id: NodeId`）のみを保持する。dtype・レイアウトの多重化は存在しない | `crates/autodiff/src/var.rs:101-105` |
| ONNX: `proto::data_type` モジュールは `FLOAT`(1)/`INT64`(7)/`BOOL`(9)/`FLOAT16`(10) のみを定数として宣言する。`COMPLEX64`(14)/`COMPLEX128`(15) は宣言されておらず、`graph::decode_tensor` の match は未対応の `data_type` 値をすべて `GraphError::UnknownDataType` で **fail-closed 拒否**する（「無言 skip は A03 の観点で禁止」と proto.rs 自身のコメントに明記） | `crates/onnx-interop/src/onnx/proto.rs:158-164`・`crates/onnx-interop/src/onnx/graph.rs:31,93,397` |
| ONNX: `GraphProto`（`crates/onnx-interop/src/onnx/proto.rs:72-90`）に `sparse_initializer`（onnx.proto3 の tag 15）が未宣言。`prost::Message::decode` は protobuf ワイヤフォーマット仕様どおり未宣言のフィールド番号を**無言でスキップ**する（同ファイル冒頭コメント `proto.rs:20-22` が明記する既存の設計方針）。このため sparse initializer のみを持つ ONNX モデルはエラーなしでデコードされ、当該テンソルが最初から存在しないものとして扱われる。complex（fail-closed 拒否）とは対照的に、sparse はこの経路単独では明示エラーにならない（本 issue では是正せず §7 の引き継ぎ候補として記録するのみ） | `crates/onnx-interop/src/onnx/proto.rs:15-22,72-90` |
| safetensors: `st_load.rs` は F32 以外の dtype を `LoadError::UnsupportedDtype` で明示エラーとして拒否する。`safetensors::Dtype` 自体に complex／sparse 相当の variant は存在しない | `crates/onnx-interop/src/st_load.rs:27,58,112,130` |
| `crates/**` 全体に `sparse`／`complex` を指す識別子・コメントは存在しない（grep で 0 件） | — |

## 3. spec 整合の確認結果

**整合している。spec への追加提案は不要。**

- REQ-9 の「引き続き対象外」列挙（`docs/spec/04-requirements.md:233`）と改定履歴（`:432`）の両方が、sparse／complex テンソルを明示的に対象外としている。文言・判断内容は本ドキュメントの結論（段階 0・非対応）と矛盾しない。
- 「除外事項（v1 スコープ外）」節（`:351-377`）には sparse／complex の bullet は存在せず、量子化・複数 GPU／DDP のような格上げ条件表（a〜e。`:404`）も定義されていない。これは矛盾ではなく、REQ-9 の「対象範囲リスト（互換 API 層の網羅対象から外す）」という区分と、「除外事項（Won't・格上げ条件付き）」という別区分の違いである。sparse／complex は前者にのみ現れ、後者には従属しない——#1652（ONNX import 公開可否）が量子化／DDP と異なり除外事項に従属しないと整理したのと同型の構造である（`docs/facade-onnx-import-exposure-decision.md` 参照）。
- したがって、sparse／complex を対象範囲へ再び組み入れるには `docs/compat-api-scope.md` §5 の範囲拡張手続き（経路 1: spec リポでの REQ-9 改定、または経路 2: 本リポでのユーザー承認 + issue 起票）を経る必要がある。量子化・DDP のような追加の除外事項ゲート（格上げ条件表の充足）は不要——ただし後述 §6 のとおり、REQ-1（完全自作コア・許容依存 8 区分）・REQ-2（数値一致複合判定・FMA 契約）という別の既存契約への抵触は避けられないため、実装着手には個別の技術的検討・承認が要る。

## 4. 契約整理（守るべき既存契約）

- **REQ-1 完全自作コア・許容依存 9 区分**（`.claude/rules/deps-policy.md`）: sparse 用の疎行列表現（COO／CSR 等の格納・演算）・complex 用の複素数型は、いずれも現行の許容依存区分に含まれない。自作 newtype／自作レイアウト型として実装するか、外部 crate 追加（区分外・ユーザー承認必須）を選ぶかの判断が必要になる。
- **REQ-2 統一複合判定と FMA 契約**（`.claude/rules/coding-rust.md`「バックエンド構成」節）: complex 乗算 `(a+bi)(c+di) = (ac-bd) + (ad+bc)i` には、現行の実数 `f32::mul_add` 単一契約に対応する丸め順序契約が存在しない。CPU／CUDA／Metal 間で complex 演算の丸め順序を新規に定義しない限り、既存の「相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満」という判定式自体は複素数の実部・虚部それぞれへ適用できるとしても、契約の拡張範囲（どの粒度で判定するか）が未定義。正規化統計・勾配長軸縮約の `f64` アキュムレータ契約も同様に、sparse／complex 固有の縮約経路に対する定義がない。
- **REQ-8 手動境界検査**（`.claude/rules/coding-rust.md`「カーネル実装の境界検査」節）: sparse テンソルは index 配列（行／列インデックス等）由来の境界検査が実数 dense テンソルとは異なる形で必要になり、既存のシェーダ・カーネル境界検査パターンをそのまま流用できない。
- **`Scalar` sealed trait の封印方針**（`docs/backend-dtype-dispatch-design.md` §4.1）: `Element` を open のまま維持し演算境界は `Scalar` で封印する、という既存設計判断（`docs/backend-dtype-dispatch-design.md` が「後から演算境界を非破壊追加できない」制約への対策として採用した）と、complex を新規 `Scalar` 実装として追加する案は原理的には両立するが、`ScalarDType`／`TypedOps<T>` の全カーネル多重化（4 型 → 5 型）という横断変更を要する。
- **tolerance／baseline・ガードレール閾値の不変**: 本記録はコード変更を伴わないため直接の影響はない。将来実装する場合も、既存の tolerance 定数・parity baseline は自己判断で緩和しない方針（`.claude/rules/coding-rust.md`）が適用される。

## 5. 設計案の比較

| 案 | 概要 | 新規依存 | REQ-1／REQ-2 への影響 | スコープ | 難度 |
|---|---|---|---|---|---|
| **A（採用）: 段階 0 — 現時点では非対応と明文化し、既存の fail-closed／無言スキップ挙動を事実として記録する** | 実装着手せず、本ドキュメントに契約整理・現状のコード事実・再開条件を記録する | なし | 既存契約への影響ゼロ | — | — |
| B: sparse をレイアウト型（`SparseCoo<T>` 等）として `tensor-core` に新設し `BackendOps` を拡張する | PyTorch `torch.sparse_coo_tensor` に最も近い設計 | 疎行列演算ライブラリを使う場合は許容依存区分外（要承認）。自作の場合は依存追加なし | `BackendOps` trait 拡張・REQ-8 境界検査の sparse 版設計が必要（承認事項） | Tier 1／2 に列挙なし・REQ-9 対象外 | XL |
| C: complex を `Element` 実装の自作 newtype（`Complex32 { re, im }` 等）で追加し `Scalar`／`ScalarDType`／`TypedOps` を拡張する | PyTorch `torch.complex64` に最も近い設計 | なし（自作 newtype のため） | sealed `Scalar` の拡張・全カーネル dtype dispatch の 5 型化・complex 乗算の FMA 契約新規定義が必要（いずれも承認事項） | 同上 | XL |
| D: 実部／虚部の 2 テンソル分解をユーザー側で合成する（新規公開 API を追加しない） | complex の代替手段として言及のみ。サポート対象とはしない | なし | 影響なし（既存 API の組み合わせ） | 非サポートの回避策の記録に留める | — |

## 6. 推奨（段階 0 確定）

**案 A を採用する。非対応の境界は以下 4 点に整理できる（すべて §2 のコード事実に基づく既存挙動）:**

1. `Tensor<T>` は dense のみ（レイアウト型は 1 種類）。
2. 算術演算は `Scalar` に封印された実数 4 型（`f32`/`f64`/`f16`/`bf16`）のみ。`Element` 自体は open だが算術経路には到達しない。
3. ONNX の complex dtype（`COMPLEX64`/`COMPLEX128`）は `GraphError::UnknownDataType` で fail-closed 拒否される。
4. safetensors は F32 以外を `LoadError::UnsupportedDtype` で拒否し、complex／sparse 相当の dtype はフォーマット自体に存在しない。

理由:

- `docs/compat-api-scope.md` §2 が既に本 issue への forward reference を持っており、対応する決定記録が存在しない状態を解消する必要がある。
- spec REQ-9 の「引き続き対象外」判断（§3）と一致しており、実装着手の前提（REQ-9 の対象範囲への組み入れ）がそもそも成立していない。
- 案 B／C はいずれも既存の封印方針（`Scalar` sealed trait）・許容依存区分・数値一致契約（FMA・境界検査）へ横断的な変更を要し、本 issue のスコープ（設計記録のみ）を超える。
- 案 D はサポート対象ではなく、非対応の代替手段としての言及に留める。

## 7. 数値一致・既存テストとの整合

コード変更を行わないため、既存の数値一致回帰テスト・parity baseline・tolerance 定数への影響はない。

## 8. 公開 API・spec 整合

- facade 新規公開面なし。
- `docs/compat-api-scope.md` §5 の範囲拡張手続きは、本イシューでは**適用しない**（対象外項目の forward reference を解消する整理であり、対象範囲の拡張ではないため。#1632「グラフ最適化スコープ」§7 と同じ整理）。
- §3 の spec 整合結果のとおり、spec への新規提案は不要。

## 9. 引き継ぎ（起票草案。本イシューでは起票しない）

- **(a) ONNX `sparse_initializer` 無言スキップの fail-closed 化**: `GraphProto` に tag 15（`sparse_initializer`）を型として宣言し、非空であれば `GraphError`（新規 variant または既存 `UnknownDataType` 相当）で明示拒否する。現状は「sparse initializer のみを持つテンソルが存在しないものとして扱われる」という意図しない黙認が起きうる（§2 参照）。A03（インジェクション／不正入力）・A08（ソフトウェア・データ整合性）の観点からの候補。起票は `.claude/rules/out-of-scope-tracking.md` の規約に従いユーザー承認後に行う。
- **(b) 将来 sparse／complex を対象範囲へ組み入れる場合の前提ゲート**: `docs/compat-api-scope.md` §5 経路 1（spec 側 REQ-9 改定）または経路 2（本リポでのユーザー承認 + issue 起票）。加えて §4 に列挙した REQ-1／REQ-2／REQ-8 との整合設計・依存追加判断が前提となる。

## 10. 承認事項（実装着手の前提。列挙のみ・本イシュー時点で未取得）

1. spec リポでの REQ-9 改定（sparse／complex を対象範囲へ組み入れる場合）。
2. sparse 用レイアウト型の新設と `BackendOps` trait 拡張（案 B を選ぶ場合）。
3. complex 用 `Element`／`Scalar`／`ScalarDType` 拡張と dtype dispatch の全カーネル多重化（案 C を選ぶ場合）。
4. complex 乗算の丸め（FMA）契約の新規定義（REQ-2 判定式自体は不変のまま、適用範囲を拡張する場合）。
5. 外部 crate を使う場合の依存追加（`.claude/rules/deps-policy.md` の許容依存 9 区分外）。
6. §9(a) の ONNX `sparse_initializer` fail-closed 化の起票。

## 11. スコープ外

- sparse／complex の実装（`Op`／`BackendOps`／VJP／parity テストの追加）。
- facade 公開面の拡張。
- `docs/spec/`（正本 submodule）の編集・spec リポへの新規提案（§3 のとおり整合済みのため不要）。
- ONNX `sparse_initializer` の fail-closed 化（§9(a) の引き継ぎ候補として記録のみ）。

## 12. 出典一覧

| 出典 | 内容 |
|---|---|
| `docs/spec/04-requirements.md:233` | REQ-9「引き続き対象外」列挙（sparse／complex テンソルを含む） |
| `docs/spec/04-requirements.md:351-377` | 「除外事項（v1 スコープ外）」節（sparse／complex の bullet・格上げ条件表がいずれも存在しないことの確認対象） |
| `docs/spec/04-requirements.md:404` | 量子化の格上げ条件表（a〜e）。sparse／complex には同等の表が存在しないことの対比根拠 |
| `docs/spec/04-requirements.md:432` | REQ-9 2026-09-12 追記に対応する改定履歴表エントリ（イシュー #66） |
| `crates/tensor-core/src/tensor.rs:33-57` | `Tensor<T>`／`Storage` の dense 専用構造 |
| `crates/tensor-core/src/element.rs:1-84` | `Element` trait（open）の定義・実装対象型 |
| `crates/tensor-core/src/element.rs:111-187` | `Scalar` trait（sealed）・`ScalarDType`・4 型限定の実装一覧 |
| `crates/tensor-core/src/dispatch.rs:31-39` | `dispatch::DType`（F32／F16 のみ） |
| `crates/autodiff/src/var.rs:101-105` | `Var<'t>` の f32 固定ノード表現 |
| `crates/onnx-interop/src/onnx/proto.rs:15-22,72-90,158-164` | `data_type` 定数宣言・`GraphProto` の `sparse_initializer` 未宣言・prost 無言スキップの設計方針コメント |
| `crates/onnx-interop/src/onnx/graph.rs:31,93,397` | `GraphError::UnknownDataType` による fail-closed 拒否 |
| `crates/onnx-interop/src/st_load.rs:27,58,112,130` | `LoadError::UnsupportedDtype`（F32 以外を明示エラー拒否） |
| `docs/backend-dtype-dispatch-design.md` | dtype dispatch 機構（`TypedOps<T>`）の設計・`Scalar` 封印方針の根拠 |
| `docs/compat-api-scope.md` §2 | sparse／complex テンソルへの本イシューへの forward reference |
| `docs/facade-onnx-import-exposure-decision.md` | 「除外事項に従属しない対象外項目」という同型の整理（#1652） |
| `docs/facade-multi-gpu-ddp-decision.md`・`docs/backend-int8-quantization-decision.md` | 除外事項に従属する Tier 2 項目（DDP・量子化）との対比 |
| `docs/autodiff-graph-optimization-scope-decision.md` §7 | 範囲拡張手続き（§5）を適用しない整理の precedent（#1632） |
| `.claude/rules/deps-policy.md` | 許容依存 9 区分 |
| `.claude/rules/coding-rust.md` | REQ-1 完全自作コア・REQ-2 バックエンド構成（数値一致複合判定・FMA 契約）・REQ-8 カーネル境界検査 |
| `.claude/rules/security.md`「A03」節 | 無言 skip 禁止の一般原則（ONNX proto.rs コメントが引用） |
| `.claude/rules/out-of-scope-tracking.md` | 対象外事項の Issue 追跡規約 |
