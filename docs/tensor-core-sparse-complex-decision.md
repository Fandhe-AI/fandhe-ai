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

## 13. 追補（イシュー #2079・2026-09-22）: §9(a) の引き継ぎは解消済み

§9(a)・§11 に記録した「ONNX `sparse_initializer` 無言スキップの
fail-closed 化」の引き継ぎ候補は、イシュー #2079 で実装済みである。
`GraphProto` に `sparse_initializer`（tag=15。検出専用の
`SparseTensorProto`）を宣言し、`graph::build_graph` が非空を
`GraphError::SparseInitializerNotSupported` で fail-closed に拒否する
ように是正した（facade 側は `OnnxError::SparseInitializerNotSupported`
へ写像。詳細は `docs/facade-onnx-import-exposure-decision.md` §13）。

§2 の「本 issue では是正せず §7 の引き継ぎ候補として記録するのみ」と
いう記述、および §9(a)・§11 の引き継ぎ候補としての記載自体は、当時の
事実の記録として変更しない。sparse テンソルの実装（COO 形式の
`values`／`indices` の解釈）自体は本追補後も引き続きスコープ外（REQ-9）
のままである。

## §14 追補（イシュー #2151・2026-09-25）: FFT 設計との相互参照

イシュー #2151 で FFT（`torch.fft.{fft, ifft, rfft, irfft}` 相当）の
設計判断記録 `docs/autodiff-fft-design.md` を作成した。同 doc は実部・
虚部を末尾次元 2 の実テンソル対（`f32`・`torch.view_as_real` 相当）で
表す方式を採り、本 doc §6 の結論（案 A・complex dtype 非対応）は
**覆していない**。`Tensor<complex64>` 等の complex dtype 自体は本
追補後も §6 の結論どおり非対応のままである。相互参照のみで §5・§6 の
判断内容・§10 の承認事項一覧は変更しない。

## §15 追補（イシュー #2616・2026-10-04）: spec 改定提案と規模見積り

基準コミット: `e30c7b2e700c69a00d66ec23494e1233d6c2b325`（`origin/main`）。`docs/spec` submodule ポインタ `2e998dd77117814f4af8ed160394ad1d6a8f888a`。本節の `file_path:line` と件数は同コミットで取得したもの（後続の変更で行番号は動くため、参照時は基準コミットで開く）。

### §15.0 結論（最初に読む）

- **コード変更なし**。`crates/**`・`Cargo.toml`／`Cargo.lock`・`docs/spec/`・tolerance／baseline・ガードレール閾値は一切変更していない。
- **spec 改定文案（§15.3）は起票していない。ユーザー承認は未取得で、承認の代行もしていない**。承認事項は §15.5 に列挙した。
- **方針の転換を明示する**。§3・§8・§11 は「spec と整合済みのため spec 提案は不要」と結論した（#1633 時点の判断）。本追補はこの立場を、ルート #2499 の到達目標（PyTorch／TensorFlow を Rust で置き換える）の追加を理由に**見直し、改定提案を新たに作る**ものである。§3・§8・§11 の本文は当時の記録として書き換えない（§13・§14 と同じ追補方式）。spec 側の追記（`docs/spec/04-requirements.md:235`、2026-09-29）と `docs/functorch-serving-hub-non-target-spec-proposal.md` §0 は、この「sparse／complex は spec 記載済みで変更なし」という結論を引用している。両者は本 PR では編集しない。提案が承認された場合は、両者の該当文言の更新も spec 側の改定に含める。
- **推奨**（未承認）: complex dtype は §15.3 の案 (ii)（「引き続き対象外」から外して除外事項〈Won't・条件付き〉へ移し、格上げ条件表を付ける）。sparse は案 (ii) に加えて、ONNX `sparse_initializer` の dense 化 (s2) を先行させる。
- 規模は大きい。complex は autodiff の dtype 一般化（`Var`／`Tape`）が前提で、全バックエンドの dtype dispatch に及ぶ。実装は本ツリーに含めない。

### §15.1 現行 spec の記述（変更前の事実）

| 箇所 | 内容 |
|---|---|
| `docs/spec/04-requirements.md:233` | REQ-9「引き続き対象外」列挙に「sparse／complex テンソル」が入っている |
| `docs/spec/04-requirements.md:235` | 2026-09-29 追記（#2194）。列挙へ 3 項目を追加。sparse／complex は変更しない |
| `docs/spec/04-requirements.md:232` | Tier 2 の dtype は「f64／f16／bf16 演算」まで。complex は含まれない |
| `docs/spec/04-requirements.md:358`〜`363` | 除外事項「分散学習・量子化の網羅対応」（Won't・条件付き）。格上げ条件表 a〜g の前例 |
| `docs/spec/04-requirements.md:430` | 改定履歴表の前例エントリ |

スコープ件数（Should 8・Could 1・Won't 11）は同 `:359` の記述による。除外事項へ項目を足す案を採ると、この件数が変わる（§15.3）。

### §15.2 規模見積り（構造で見積もる。工数の推測値は書かない）

件数は基準コミットで次のコマンドにより取得した。

| 指標 | コマンド | 値 |
|---|---|---|
| `TypedOps<T>` の fn 数 | trait 定義（`crates/tensor-core/src/typed_ops.rs:33`〜`:50`）のメソッドだけを数える。`gemm`・`add`・`mul`・`relu`・`exp`・`tanh`・`sum`・`max` の 8 つ。`grep -c "fn "` はファイル末尾のテスト用関数も数えるため使わない | 8 |
| CUDA の kernel ファイル | `ls crates/backend-cuda/src \| grep -c kernels` | 39 |
| Metal の shader | `ls crates/backend-metal/src/shaders \| wc -l` | 26 |
| backend-cpu の src | `ls crates/backend-cpu/src \| wc -l`（直下）／`git ls-files crates/backend-cpu/src \| wc -l`（再帰） | 41／50 |
| `ScalarDType::` の参照 | `git grep -n "ScalarDType::" -- crates \| wc -l` | 88 箇所（11 ファイル） |
| autodiff の `Op` variant | `crates/autodiff/src/tape.rs:91`〜`:1281` の簡易集計 | 約 90 |

#### complex

1. **tensor-core**
   - `Scalar` は `private::Sealed`（`crates/tensor-core/src/element.rs:111`）で封印され、実装は `f32`／`f64`／`f16`／`bf16` の 4 型（同 `:171`〜`:186`）である。complex 型の newtype を足し、`Sealed` と `Scalar` を実装する。
   - `ScalarDType`（同 `:141`）は `#[non_exhaustive]` なので、variant 追加は 0.10.0 の公開 API を壊さない。ただし `ScalarDType` は facade から再公開済み（`crates/facade/src/lib.rs:270`）である。公開面が広がるため、facade の公開形は `docs/compat-api-scope.md` §5 の手続きを経る。
   - `dispatch::DType`（`crates/tensor-core/src/dispatch.rs:31`）は `ScalarDType` とは別の enum で、`#[non_exhaustive]` が付いていない。complex を GEMM の経路選択へ載せるなら、`docs/backend-dtype-dispatch-design.md` が定める `ScalarDType → Option<dispatch::DType>` の明示マッピングの方式（complex は `None`）に従う。
   - `TypedOps<T>` の 8 メソッドのうち、complex で意味が定まる演算だけを 3 バックエンドへ実装する。
     - 対象にできる演算: `gemm`・`add`・`mul`・`exp`・`sum`（`tanh` は複素解析関数として定義できるが、数値契約〈実部・虚部の丸めと特異点付近の扱い〉を別途決める条件付き）。
     - 対象外にする演算: `max`（順序が必要）と `relu`（`max(0, x)` に基づく）。complex には全順序がなく、PyTorch も complex の `max` 系を未対応にしている。順序・出力契約を新しく作る案は採らない。
     - 非対応演算の扱い: 実装しない演算は `BackendError::Unsupported` で fail-closed に拒否し、無言で実部だけを使う等の代替はしない。
     - 公開 API と trait 設計の前提: 現行の `TypedOps<T: Scalar>` は 8 メソッドすべてを必須にしている。complex の `Scalar` 実装でこの trait をそのまま使うと、上の対象外 2 演算も実装を強いられる。選択肢は (1) 対象外演算を常に `Unsupported` で返す実装を許容する、(2) complex 用に演算を絞った別 trait（例: `ComplexOps`）を足す、の 2 つ。既存 trait にメソッドを足す・削る変更は 0.10.0 の公開 API を壊しうるため、既存 trait は変更せず追加 API で拡張する前提とし、(1)／(2) の採否は承認事項（§15.5）に加える。
2. **autodiff（支配的コスト）**
   - `Var<'t>` は f32 のノード id だけを持つ。`Tape` は `ops: Box<dyn BackendOps + Send>` を持つ（`docs/backend-dtype-dispatch-design.md` が `crates/autodiff/src/tape.rs:775` として引用）。`Tape<T>` 化は同 doc §8 が対象外と明記した作業であり、complex の勾配を扱うには**前提条件**になる。「4 型 → 5 型化」に含めて見積もってはならない。
   - VJP の規約を決める必要がある。PyTorch は共役 Wirtinger 規約である。実数の関数が complex を経由する場合（`abs`・`angle`・`real`・`imag`）の勾配の向きも定義が要る。
   - `Op` enum の約 90 variant について、complex 対応の要否を 1 つずつ決める。
3. **数値契約**
   - complex 乗算の丸め（FMA）契約を新しく定義する必要がある。`.claude/rules/coding-rust.md` の matmul 系 FMA 契約は実数を前提にしている。
   - REQ-2 の統一複合判定は、実部と虚部のそれぞれに当てはめる案とする。判定式と tolerance は変えない。
   - f64 アキュムレータ契約（縮約経路）も complex 版の定義が要る。
4. **バックエンド**: CUDA の kernel 39 ファイル、Metal の shader 26 本、backend-cpu のうち該当ファイルに complex 版を足す。REQ-8 により、各カーネルに手動の境界検査を維持する。Metal は MSL が `double` に対応しないため、complex128 は soft-f64 系（`crates/backend-metal/src/soft_f64.rs`）の扱いが別途要る。GPU 経路は実機 parity が必要で、`#[ignore]` 分離と申し送りの手順に従う。
5. **interop**
   - ONNX は `COMPLEX64`(14)／`COMPLEX128`(15) の定数宣言とデコードが要る。現状は `GraphError::UnknownDataType`（`crates/onnx-interop/src/onnx/graph.rs:43`）で fail-closed に拒否している。
   - safetensors のフォーマットには complex dtype がない。保存形式（実部・虚部の対にする等）は別途決める。

#### sparse

1. dense 専用の `Tensor<T>`（`crates/tensor-core/src/tensor.rs`）とは別に、COO／CSR のレイアウト型を新設する。あわせて `BackendOps` の拡張（spmm 等）と、VJP（values 側・dense 側の勾配）が要る。
2. REQ-8: index 配列から読む位置の境界検査を、全カーネルに手動で入れる。
3. ONNX `sparse_initializer` は、現状 #2079 で fail-closed に拒否している（`crates/onnx-interop/src/onnx/graph.rs:97` の `GraphError::SparseInitializerNotSupported`、`crates/facade/src/interop/onnx.rs:590` の `OnnxError::SparseInitializerNotSupported`）。候補は次の 3 つ。

   | 候補 | 内容 | 規模 |
   |---|---|---|
   | (s1) | 拒否を維持する | 変更なし |
   | (s2) | import 時に dense 化する | sparse テンソル型なしで成立する低コスト案。`SparseTensorProto` の `values`／`indices`／`dims` の長さ・形状・index 範囲の検証と、dense 化後の要素数の上限検査が要る。現状の検出専用型（`proto.rs` の `SparseTensorValueName`）を、中身を読む型へ拡張する |
   | (s3) | sparse 型としてデコードする | 上記のレイアウト型の新設が前提 |

4. facade での公開は `docs/compat-api-scope.md` §5 の手続きを経る。

#### 依存

自作の newtype とレイアウト型だけで実装でき、新規依存は不要と見込む（REQ-1 の完全自作コア）。外部 crate を使う場合は許容依存 10 区分の外になるため、別途の承認事項とする。

### §15.3 spec 改定案（(b) 形式の文案。未起票）

#### 候補の比較

| 案 | 内容 | 評価 |
|---|---|---|
| (i) | 「引き続き対象外」を維持する | ルート #2499 の目標と両立しない。dtype 行の「部分的」が解消できない |
| (ii) | 「引き続き対象外」から外し、除外事項（Won't・条件付き）へ移して格上げ条件表を付ける | 量子化・DDP の前例（`04-requirements.md:358`〜`363`）と同型。条件が満たされるまで実装しない扱いを保てる |
| (iii) | Tier 2 へ組み入れる | 規模（§15.2）に対して時期尚早。complex は autodiff の dtype 一般化が前提で、条件の整理が先 |

推奨（未承認）: complex は (ii)。sparse は (ii) とし、ONNX `sparse_initializer` の dense 化 (s2) を先行する選択肢を条件表に含める。(s2) 単独なら sparse テンソル型を足さずに済む。

#### 提案文案

````markdown
## 背景

REQ-9「引き続き対象外」列挙は sparse／complex テンソルを含む。実装リポ
Fandhe-AI/fandhe-ai のルート #2499 は PyTorch／TensorFlow の置き換えを到達
目標とし、dtype の網羅を対象に含める。実装リポの規模見積り
（`docs/tensor-core-sparse-complex-decision.md` §15）では、complex は autodiff の
dtype 一般化と全バックエンドの dtype dispatch に及び、sparse はレイアウト型の
新設を要する。着手の前提を条件として明文化したい。

## 提案

1. REQ-9「引き続き対象外」列挙から「sparse／complex テンソル」を外す。
2. 除外事項に「sparse／complex テンソル（Won't・条件付き）」を追加し、格上げ条件
   表（a〜）を付ける。条件に含める項目の案:
   (a) autodiff の dtype 一般化（`Tape`／`Var`）の設計が承認されていること。
   (b) complex の VJP 規約（共役 Wirtinger 等）と FMA 契約が定義されていること。
   (c) REQ-2 の統一複合判定を実部・虚部それぞれに適用する方式で、tolerance・
       判定式を変更せずに数値一致が成立すること。
   (d) sparse の REQ-8 境界検査（index 配列の範囲検査）が全カーネルに維持される設計であること。
   (e) 依存追加なし（自作の newtype とレイアウト型）で成立すること。外部 crate を使う場合は REQ-1 の依存追加ルールに従い別途承認。
   (f) ONNX `sparse_initializer` の扱いが決まっていること（拒否維持／dense 化／sparse 型）。
3. 既存の受け入れ基準・Tier 1／Tier 2 列挙・REQ-1・REQ-2（tolerance を含む）・
   REQ-8 は変更しない。除外事項の件数（Won't 11）が増える場合は、スコープ件数の
   更新を同時に承認事項とする。

## 改定履歴表のエントリ案

| 日付 | 内容 |
|---|---|
| （承認日） | 実装リポ Fandhe-AI/fandhe-ai イシュー #2616（ルート #2499）。REQ-9 の対象外列挙から sparse／complex を除外事項（Won't・条件付き）へ移し、格上げ条件表を追加。受け入れ基準・Tier 列挙・REQ-1／REQ-2／REQ-8 は変更しない。 |
````

文案は起票していない。spec リポ（Fandhe-AI/fandhe-ai-spec）への起票と `docs/spec/` の編集はいずれも行わない。

### §15.4 FFT（#2630）との関係

- 「表現」の軸と「dtype」の軸は独立している。FFT の表現は実部・虚部の実テンソル対（`[..., n, 2]`、`torch.view_as_real` 相当。`docs/autodiff-fft-design.md`）、dtype は complex dtype である。
- #2630 の方針は、本提案の承認結果にかかわらず変えない。§14 のとおり、案 A（complex dtype 非対応）を前提に FFT を実装できる。
- 将来 complex dtype を採用した場合は、`view_as_real`／`view_as_complex` 相当の相互変換を移行経路にする。既存の `[..., n, 2]` 表現との間で、メモリレイアウトを変えずに行き来できる。

### §15.5 承認事項（未取得。承認の代行はしていない）

1. 推奨する spec 改定案の採否（sparse と complex を別々に判断する）。
2. spec リポ（Fandhe-AI/fandhe-ai-spec）への起票の可否。
3. `Var`／`Tape` の dtype 一般化に着手するかどうか。
4. complex の FMA 契約と VJP 規約を新しく定義すること。
5. sparse のレイアウト型新設と `BackendOps` の拡張。
6. ONNX `sparse_initializer` の扱い（(s1) 拒否維持／(s2) dense 化／(s3) sparse 型）。
7. 外部 crate を使う場合の依存追加（現時点では不要と見込む）。
8. complex で対応する演算の選別（`gemm`・`add`・`mul`・`exp`・`sum` を対象、`max`・`relu` を対象外、`tanh` は条件付き）と、非対応演算の扱い（`Unsupported` での拒否）、およびその公開 API／trait 設計（§15.2 complex 1 の (1)／(2)）。
9. 承認後に実装 issue を起票すること（本ツリーには含めない）。

### §15.6 セキュリティ観点

本追補は実行時の挙動を変えない。ただし (s2) または sparse 型を採用すると、#2079 で閉じた `sparse_initializer` のパース経路が再び開く。その場合は、`indices`・`values`・`dims` の長さ・形状・index の範囲を使用前に検証し、不正なら fail-closed で拒否すること（OWASP A03）を実装 issue の受け入れ条件にする。依存は追加しない（A06）。
