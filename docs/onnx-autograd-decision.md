# ONNX import グラフへの autograd 接続（イシュー #2078）

## 1. 背景

`docs/facade-onnx-import-exposure-decision.md` §6.3 (c)・§12.6 (c) は「学習可能化
（`Tape`／`Var` への変換層）は未実装・別 issue」と記録していた。`crates/onnx-interop`
は `fandhe_ai_autodiff` を通常依存へ昇格済み（#2036）のため、依存追加なしで
autograd 変換層を追加できる。本 issue はその変換層 `onnx::autograd` を実装する。

## 2. 配置と公開範囲

- 実装は `crates/onnx-interop/src/onnx/autograd.rs`（単一ファイル）。
- `onnx-interop` は crates.io 公開クレート（公開名 `fandhe-ai-onnx-interop`）のため
  `pub mod autograd` は同クレートの公開面に入るが、**facade（唯一のサポート対象
  公開面）は再エクスポートしない**——`fandhe_ai_autodiff::Tape::custom`
  （#1946・`docs/autodiff-custom-function-decision.md` §12.5 (a)）と同じ位置づけ。
  facade 側の否定ガード（`crates/facade/tests/api_surface.rs`）は既存の
  `interop_module_exposes_only_approved_onnx_surface` が `crates/facade/src/interop/onnx.rs`
  自身の pub シグネチャのみを検査するため、`autograd` を facade へ一切結線して
  いない本 PR では追加のガードなしで「autograd 系が facade から到達不能」を
  機械的に担保できている（同テストは緑のまま）。

## 3. スコープ（実装計画からの縮小）

計画段階では 22 op すべてに autograd 経路（view／shape 系を含む）を持たせる
想定だったが、実装時に以下へ縮小した。理由は本 issue の実装エージェント実行
環境の作業時間制約であり、技術的な不可能性ではない。

### 3.1 勾配追跡対象（`Var` 経路。11 op）

`Gemm`・`MatMul`・`Add`・`Mul`・`Div`・`Sqrt`・`Relu`・`Sigmoid`・`Erf`・
`Softmax`・`LayerNormalization`。

いずれも `fandhe_ai_autodiff::Tape::custom`（`CustomFunction` trait）で実装し、
**forward は必ず `crate::ops` の同一関数をそのまま呼ぶ**ため `interp::run` と
bit 完全一致する（構造的に保証。#1946 の決定規則「forward = `ops::*`」を踏襲）。
backward は解析的勾配を手書きし、REQ-2 の統一複合判定（相対誤差 1e-3 未満または
絶対誤差 1e-5 未満）で検証する契約（bit 一致は要求しない）。

- `Gemm`／`MatMul`: 標準的な行列積 VJP を `ops::matmul`／`Tensor::permute` で
  組み立てる（`Y = alpha * A' @ B' + beta * C` の `trans_a`／`trans_b` 双方に対応）。
- `Add`／`Mul`／`Div`: broadcast 後の局所勾配を計算してから
  `reduce_to_shape`（本モジュール内ヘルパー。broadcast VJP の標準縮約）で入力
  shape へ縮約する。
- `Sqrt`／`Relu`／`Sigmoid`／`Erf`: `UnaryElemFn`（forward 関数ポインタ＋
  `grad_fn(x, y)` の組）で統一的に実装。
- `Softmax`: `dx = y ⊙ (g − sum_axis(g ⊙ y))`。軸方向縮約は `f64` アキュムレータ
  （`.claude/rules/coding-rust.md` の長軸縮約契約）。
- `LayerNormalization`: 平均・分散は `f64` アキュムレータ・二乗和は要素を先に
  `f64` へ昇格してから二乗する（同契約）。`dx`／`dscale`／`dbias` の解析式は
  標準的な LayerNorm VJP。**forward（`ops::layer_normalization`）自体もこの
  f64 統計契約へ統一済み**（PR #2223 codex-review 再指摘。backward のみ f32
  bit-match で揃える案は「forward 自体が f64 契約〈`.claude/rules/
  coding-rust.md`〉に違反したまま残る」ため採らず、forward・backward 双方を
  f64 統計へ揃える形で本節の記述と実装を一致させた）。

### 3.2 非勾配経路（`Const` 専用。3 op）

`Shape`・`Cast`・`Constant`。

- `Shape`: `Var` 入力を許容し（`.value()` で実体化して形状のみ読む。出力は
  勾配を持たない `I64` メタデータのため問題ない）。
- `Cast`: **`F32 → F32`（恒等）のみ `Var` のまま透過**し、`F32 → 非 F32`
  （`INT64`／`BOOL`／`FLOAT16`）は `.value()` で実体化して変換する——**ここで
  のみ意図的に勾配を切断する**（承認事項 4）。非 F32 → F32 等の残りの組合せは
  既存 `interp::compute_cast` と同じ dtype 組合せ表に従う。
- `Constant`: 入力なし・常に `Const` を返す（既存 `interp::compute_constant` と
  同じ属性解決ロジック）。

### 3.3 未実装（fail-closed。8 op）

`Gather`・`Unsqueeze`・`Concat`・`Slice`・`Mod`・`Reshape`・`Squeeze`・
`Transpose`。

これらのノードに到達すると、入力が `Var`／`Const` のいずれであっても
`AutogradError::UnsupportedInAutograd` で fail-closed に拒否する
（no-silent-skip 契約。`.claude/rules/coding-rust.md`）。「Const-only なら
`interp::run` と同じロジックで実行できる」という部分最適化はあえて実装せず、
「未対応 op は常に拒否」という単純な契約に倒した（Var/Const 判定漏れによる
静かな勾配欠落を型で防ぐため）。

view／shape 系演算の勾配対応（`Reshape`／`Transpose` 等は本来ちょうど
「勾配をそのまま並べ替えて返す」だけで実装できる比較的単純な拡張である）は
後続スコープとして PR 本文に記録する（`.claude/rules/out-of-scope-tracking.md`）。

## 4. `BindGraph` API 形状

```rust
pub struct BindOptions { pub trainable: Option<HashSet<String>> }
impl BindOptions { pub fn with_trainable(trainable: HashSet<String>) -> Self }

pub enum AutogradValue<'t> { Var(Var<'t>), Const(Value) }

pub struct BoundGraph<'g, 't> { /* ... */ }
impl<'g, 't> BoundGraph<'g, 't> {
    pub fn bind(graph: &'g Graph, tape: &'t Tape, options: &BindOptions) -> Result<Self, AutogradError>;
    pub fn run(&self, feeds: HashMap<String, AutogradValue<'t>>) -> Result<HashMap<String, AutogradValue<'t>>, AutogradError>;
    pub fn forward(&self, input: &Var<'t>) -> Result<Var<'t>, AutogradError>;
    pub fn params(&self) -> &HashMap<String, Var<'t>>;
    pub fn param(&self, name: &str) -> Option<Var<'t>>;
}
```

- `bind` は `graph.initializers` の F32 initializer を一度だけ葉化する
  （`trainable` に含まれる、または `trainable == None` なら `tape.var`
  ＝ `requires_grad = true`、含まれなければ `tape.var_no_grad`）。非 F32
  initializer はそのまま `Value` として保持する。
- `run` の feed 検証順序は `interp::run` と同一（1. `MissingFeed` →
  2. `UnknownFeed`）。initializer と同名の feed は initializer を上書きする
  （ONNX のデフォルト値セマンティクス）。
- `forward` は「initializer を除く非追跡入力がちょうど 1 個・出力が 1 個」の
  グラフに限り使える便宜 API（`nn::Linear::bind` 的な単純さのためのショート
  カット）。`model.onnx`（`Gemm→Relu→Gemm→Relu→Gemm→Sigmoid`）のような単純な
  MLP はこの形に当てはまる。

## 5. 検証（`crates/onnx-interop/tests/onnx_autograd.rs`。20 テスト）

- per-op forward bit 同一（`interp::run` vs `BoundGraph::run`）: `Relu`・
  `Sigmoid`・`Sqrt`・`Erf`・`Add`（broadcast あり）・`Mul`・`Div`・`MatMul`・
  `Softmax`・`LayerNormalization`・`Gemm`（`transB`・bias 付き）の 11 op。
- `model.onnx`（実 fixture。`Gemm→Relu→Gemm→Relu→Gemm→Sigmoid`）:
  - forward: 3 通りの入力で `interp::run` と bit 完全一致。
  - backward: `fc1.weight` への解析的勾配を中心差分（FD, h=1e-2）と突き合わせ、
    FD 自体の離散化誤差を見込んだ専用許容誤差（絶対誤差 5e-2 または相対誤差
    5e-2。REQ-2 の bit／複合判定とは別の本テスト専用の緩い基準）で検証。
- `trainable` フィルタ: 指定 param のみ `Gradients::get` が `Ok(Some)`。
- エラー経路: `MissingFeed`・`UnknownFeed`・クロステープ（`AutodiffError`）・
  `NotSingleInputOutput`・`UnsupportedInAutograd`（`Reshape`）・`Shape`／
  `Cast(F32→F32)` の非勾配経路が正しく動くこと。

CUDA／Metal 実機は本実装エージェント実行環境から到達不能のため未実測。
`docs/perf/logs/onnx-autograd-2078/README.md` に測定コマンドを申し送る。

## 6. ユーザー承認事項

1. **facade 公開の可否と API 形式**（未実施）。`OnnxModel::bind_autograd(&self,
   tape: &Tape, options: &BindOptions) -> BoundGraph` 相当の薄い委譲、または
   standalone 関数のいずれか。承認までは内部クレート限定（#1946 と同型）。
2. `Gather`・`Unsqueeze`・`Concat`・`Slice`・`Mod`・`Reshape`・`Squeeze`・
   `Transpose` を autograd 経路で未実装（fail-closed）とする本スコープの縮小
   （§3.3）。
3. `Erf`／`Softmax`／`LayerNormalization` を `Tape::custom`（ホスト実行。
   `CustomFunction` の契約上 GPU tape 上でも常にホスト実行になる）で実装する
   ため、GPU 上に構築した `Tape` でも当該 op はホスト実行になる点（性能
   非保証）の受容。
4. `Cast` による勾配切断（F32 → 非 F32 のみ）・`trainable` 既定（`None` の
   場合は全 F32 initializer を勾配追跡対象とする）の意味論。

## 7. スコープ外（`.claude/rules/out-of-scope-tracking.md` に従い記録）

- §3.3 の 8 op（`Gather`・`Unsqueeze`・`Concat`・`Slice`・`Mod`・`Reshape`・
  `Squeeze`・`Transpose`）の勾配対応。
- 制御フロー op・BatchNorm running stats 等の ONNX 固有数値契約・optimizer
  state 管理（イシュー本文どおり）。
- 学習後パラメータの `Graph` initializer への書き戻し（→ `to_bytes` で export
  する経路）。
- facade 公開（承認事項 1 の承認後に別 issue）。
- CUDA／Metal 実機実測（§5 参照）。

## 8. 保守・運用チェックリスト（イシュー #3759）: ONNX autograd の facade 公開形（B1）

雛形は `docs/decision-record-template.md`。§1〜§7 の本文は書き換えず、本節で追記・訂正する。基準 sha は main の `81e23f9ec`（`path:line` はこの sha のもの）。実機ホスト名は `<cuda-node>`・`<metal-node>`・`<rocm-node>` で表す。

### 8.0 位置づけ・課題・推奨

- 位置づけ: 横断ルート #3711・Phase 2。承認は #3770 論点 1（**未承認**。本節は承認の根拠にならない）。実装は #3842（公開）・#3843（受入テスト）、実機 parity は #3864（`<cuda-node>`）・#3865（`<metal-node>`）。
- 番号のずれ: イシュー本文は「項目 26」と書くが、追跡表 `docs/compat-matrix-ari-tracker.md` では B1 は行 #25（#26 は B3 Sigmoid。`docs/api-compat-0.14.0-decision.md` §1.2 も 25）。本 PR は行 #25 だけを更新し、ほかの issue の記述は直さない。
- 課題: 内部クレートの `onnx::autograd`（#2078）は `BoundGraph::{bind, run, forward, params, param}` を持つが、facade は未結線（`crates/facade/src/interop/onnx.rs:8` が「autograd 未接続」と書く）。§6 の承認事項 4 件が未回答のまま残っている。
- §6 の記述の訂正（実測。§3.3・§6 の本文は残す）:
  - 項 2「未対応 8 op」: 明示の fail-closed 腕（`UnsupportedInAutograd`）は **16 op**（8 op ＋ #2186 の `Clip`・`Tanh`・`Gelu`・`Where`・`Expand`・`ReduceMean`・`Pad`・`Resize`。`crates/onnx-interop/src/onnx/autograd.rs:1526-1530`）。さらに interp が import 対応する `Conv`・`MaxPool`・`AveragePool`・`BatchNormalization`・`GlobalAveragePool`・`Flatten` の 6 op は autograd の dispatch に腕がなく、一般の `InterpError::UnsupportedOp`（「未対応 op_type」という誤診断）に落ちる。計 22 op が学習できない
  - 項 3「`Erf`／`Softmax`／`LayerNormalization` がホスト実行」: 勾配経路の 11 op **すべて**が `tape.custom` を通る（`autograd.rs:1305`〜`:1370`）。`CustomFunction` はホストの `Tensor<f32>` だけを受け渡し、GPU 上の `Tape` でも常にホスト実行になる（`crates/autodiff/src/custom.rs:31-44`）
- 推奨: **実装する（案 A）**。理由は 4 つ。(1) `OnnxModel::{run, to_bytes, from_sequential}` と同じメソッドへの薄い委譲でロジックを複製しない。(2) 公開面のガードがメソッド名の allowlist 方式なので拒否側をそのまま延ばせる。(3) 追加のみで済む（§8.6）。(4) fail-closed の範囲が型付きエラーで固定され、静かな勾配欠落が起きない。

| 案 | 内容 | 利点 | 欠点・影響 |
|----|------|------|-----------|
| A（推奨） | `OnnxModel::bind_autograd(&self, tape, options)` と facade 側のラッパー型 | 既存メソッド群と同形。scanner へ足す名前が少ない。発見しやすい | 戻り値に `OnnxModel` の借用寿命が乗る |
| B | standalone 関数 `fandhe_ai::interop::onnx::bind_autograd(&OnnxModel, &Tape, &opts)` | `OnnxModel` の impl を太らせない | 入口が型の外に増える。scanner の `("fn", ...)` は名前でしか照合せずメソッドと関数を区別できない |
| C | 対象外にする | 変更なし | B1 が未公開のまま残り、相互運用の行が「部分的」のまま |

### 8.1 公開形

- シグネチャ案（すべて `crates/facade/src/interop/onnx.rs` に置く。別ファイルにすると scanner〈走査対象は同ファイルだけ〉を逃れるか `mod.rs` の pub 項目 4 件の検査が壊れる）:
  - `impl OnnxModel { pub fn bind_autograd<'m, 't>(&'m self, tape: &'t Tape, options: &OnnxAutogradOptions) -> Result<OnnxAutogradGraph<'m, 't>, OnnxError> }`（内部は `BoundGraph::bind(&self.graph, &tape.0, ..)` への委譲）
  - `#[non_exhaustive] #[derive(Clone, Debug, Default)] pub struct OnnxAutogradOptions`（フィールドは private）＋ `pub fn with_trainable(names: HashSet<String>) -> Self`
  - `pub struct OnnxAutogradGraph<'m, 't>`（内部 `BoundGraph` を private に持つ newtype）＋ `run(&self, feeds: HashMap<String, OnnxTapeValue<'t>>) -> Result<HashMap<String, OnnxTapeValue<'t>>, OnnxError>`・`forward(&self, input: &Var<'t>) -> Result<Var<'t>, OnnxError>`・`params(&self) -> Vec<(String, Var<'t>)>`（名前順で決定的。内部 `HashMap` を出さない）・`param(&self, name: &str) -> Option<Var<'t>>`
  - `#[non_exhaustive] #[derive(Clone, Debug)] pub enum OnnxTapeValue<'t> { Var(Var<'t>), Const(OnnxValue) }`（内部 `AutogradValue` とは private 関数で変換。`From` は公開しない）
  - `OnnxValue` には variant を足さない（`#[non_exhaustive]` なしのため足すと破壊的。`onnx.rs:515`）
- facade 内の配置: `fandhe_ai::interop::onnx::{OnnxAutogradGraph, OnnxAutogradOptions, OnnxTapeValue}` と `OnnxModel::bind_autograd`。`onnx.rs:8`・`:341` の「autograd 未接続」doc は実装 PR で書き換える。
- 名前の制約: `FORBIDDEN_INTERNAL_TYPE_SUBSTRINGS` は部分文字列照合。足す内部型名（`AutogradValue`・`BindOptions` 等）を含む名前は使えない（`OnnxAutogradValue`・`OnnxBindOptions` は不可）。最終名は §8.10 の点 7。
- 正ガードへの反転計画（実装 PR #3842 が行う。基準は `crates/facade/tests/api_surface.rs`）:
  1. `scan_unapproved_onnx_pub_items`（`:1959`）の `ALLOWED_PUB_ITEMS` を 12 件から 20 件へ（`bind_autograd`・`OnnxAutogradGraph`・`OnnxAutogradOptions`・`OnnxTapeValue`・`with_trainable`・`forward`・`params`・`param` の 8 件。`run` は既存エントリを共用）
  2. 同関数の `FORBIDDEN_INTERNAL_TYPE_SUBSTRINGS` に `onnx::autograd`・`BoundGraph`・`BindOptions`・`AutogradValue`・`AutogradError` を足す（否定プローブ）。`interop_module_exposes_only_approved_onnx_surface`（`:2424`）の行単位禁止リストにも同じ語を足す
  3. 同テストの期待シグネチャ一覧に新しい公開項目を足す
  4. `approved_onnx_surface_yields_no_offenses`（`:2303`）の合成ソースへ新項目を足す（空虚 pass の防止）
  5. 新設の正ガード `onnx_autograd_types_are_reachable_via_facade`（facade だけで bind → forward → backward → `Gradients::get`。先例 `:2596`）
  6. 新設の否定の自己テスト `unapproved_onnx_autograd_items_are_flagged`（`BoundGraph`・`AutogradError`・`onnx::autograd::BindOptions` を pub シグネチャに含む合成ソースが検出される）と、名前衝突しないことの自己テスト
- 残す否定側: `facade_sources_reference_onnx_interop_only_in_interop_module`（`:1891`）、`facade_depends_on_onnx_interop_only_in_approved_shape`、`mod.rs` の pub 項目 4 件の検査。autograd 専用の `*HoldDoctestGuard` はもともとなく、拒否側は上記 allowlist が担う。

### 8.2 rustdoc と doctest の例

- 新しい公開項目それぞれに 1〜2 行の役割要約を付ける。`bind_autograd` の doc には次を書く: 勾配経路 11 op・非勾配経路 3 op（`Shape`・`Cast`・`Constant`）・fail-closed 22 op／GPU tape でもホスト実行（性能非保証）／`set_cuda_onnx_gpu_execution_enabled` などの opt-in は `run` だけに効き `bind_autograd` は見ない／`Cast`（F32 から F32 以外）で勾配が切れる。
- doctest（CPU で CI 実行できる形）: `compat::Sequential`（Linear と ReLU）→ `OnnxModel::from_sequential` → `fandhe_ai::tape()` → `bind_autograd` → `forward` → 損失 → `tape.backward` → `Gradients::get(param)`。onnx-interop の fixture には頼らない（facade の doctest から届かない）。
- 学習ループの書き方: 書き戻しは §7 のとおりスコープ外。`Tape` はステップごとに作って捨てる前提（`crates/autodiff/src/compat/sequential.rs:125-148`）なので、利用者が更新後の重み `Tensor<f32>` を持ち、各ステップで新しい `Tape` に `bind_autograd` し、更新後の重みを `OnnxTapeValue::Var(tape.var(&w))` として **initializer と同名の feed** で `run` に渡す（`run` は同名 feed で initializer を上書きする。`autograd.rs:226-340`）。ここは単一入力専用の `forward` ではなく `run` を使う。
- GPU の例（`tape_for(Device::Cuda)` 等）は `no_run`。失敗系は `compile_fail` に頼らず（stable rustdoc はエラーコードを照合しない）、正のプローブと実行時テストで確かめる。

### 8.3 エラー型

- 新しい型は作らず、`OnnxError`（`#[non_exhaustive]`）に variant を足す: `UnsupportedInAutograd { node, op_type }`・`UnknownTrainable { name }`・`NotSingleInputOutput { inputs, outputs }`・`FeedTypeMismatch { name, expected }`・`Autodiff(AutodiffError)`（facade の公開型。クロステープも型で判別でき、`source()` は `Some`）。
- 写像: `AutogradError::{MissingFeed, UnknownFeed}` は既存 `OnnxError::{MissingFeed, UnknownFeed}` へ（`run` と同契約）。`Interp` は既存の interp 写像関数へ。`Shape`・`Op` と残り（`#[non_exhaustive]`）は fallback 腕で `Execution { message }` へ。
- 6 op（`Conv` ほか）は実装 PR で内部クレートの明示 fail-closed 腕へ足し `UnsupportedInAutograd` にそろえる（推奨。§8.10 点 3）。それまでは facade 上で `UnsupportedOp` のまま見える。
- 検証順（A03）: `bind` は `UnknownTrainable` を葉化と確保の前に検査（既存実装どおり）。`run` は `MissingFeed` → `UnknownFeed` を計算の前に検査。facade の委譲はこの順序を変えない。本番経路で `unwrap`／`expect` は使わない。

### 8.4 テスト計画（すべて案）

| 受け入れ基準 | テスト | 区分 | `<cuda-node>` | `<metal-node>` | `<rocm-node>` |
|---|---|---|---|---|---|
| facade の `forward`／`run` が内部 `BoundGraph::run` と bit 完全一致（`model.onnx` と `from_sequential`） | `crates/facade/tests/interop_onnx_internal_parity.rs::onnx_autograd_forward_matches_internal_bound_graph_bit_identical` | CI | 該当なし | 該当なし | 該当なし |
| facade の勾配が内部 `BoundGraph` の勾配と bit 完全一致 | 同 `onnx_autograd_gradients_match_internal_bound_graph_bit_identical` | CI | 該当なし | 該当なし | 該当なし |
| 学習が進む（決定的シード・固定ステップ SGD で損失が下がる。feed 上書き） | `crates/facade/tests/interop_onnx_autograd.rs::onnx_autograd_sgd_loop_decreases_loss`（#3843） | CI | 該当なし | 該当なし | 該当なし |
| 未対応 op を fail-closed で拒否（16 op の代表と 6 op の代表） | 同 `unsupported_op_in_autograd_is_rejected_fail_closed` | CI | 該当なし | 該当なし | 該当なし |
| `Cast` は F32 から非 F32 のみ勾配を切る | 同 `cast_severs_gradient_only_for_non_f32_target` | CI | 該当なし | 該当なし | 該当なし |
| `trainable` の絞り込みと既定（全 F32 initializer） | 同 `trainable_default_and_filter_semantics` | CI | 該当なし | 該当なし | 該当なし |
| 新 `OnnxError` variant 全て | 同 `onnx_autograd_error_paths_*` | CI | 該当なし | 該当なし | 該当なし |
| 既定経路 `OnnxModel::run` が変更前と bit 同一 | 既存 `interop_onnx_*` が無修正で通る | CI | 該当なし | 該当なし | 該当なし |
| 公開面の正ガード・否定ガード | §8.1 の 1〜6 | CI | 該当なし | 該当なし | 該当なし |
| GPU tape と CPU tape の forward／勾配が REQ-2 統一複合判定で一致（CUDA） | `crates/facade/tests/interop_onnx_autograd_gpu_parity.rs::cuda_onnx_autograd_tape_matches_cpu_tape_within_req2_on_real_device` | `#[ignore]` | 未実測（#3864） | 該当なし | 該当なし |
| 同（Metal） | 同 `metal_onnx_autograd_tape_matches_cpu_tape_within_req2_on_real_device`（`#[cfg(target_os = "macos")]` ＋ `#[ignore]`） | `#[ignore]` | 該当なし | 未実測（#3865） | 該当なし |

- 実機テストが測るもの: 11 op は GPU tape 上でもホスト実行のため、GPU のカーネルではなく (a) device 常駐入力の `materialize` とホスト往復での値の保存、(b) 利用者の損失など device 上で動く組み込み `Var` 演算と組み合わせたときの一致、の 2 点である。形は既存の `crates/facade/tests/interop_onnx_gpu_execution_parity.rs` に倣い、置き場は共通雛形 #3778（OPEN）が入ってから合わせる。`<rocm-node>` は facade に `Device::Rocm` がないため該当なし。
- §5 の有限差分用 5e-2 基準は REQ-2 ではないため facade のテストへ持ち込まない。

### 8.5 数値契約

- forward: CPU tape ではホスト `ops::*` を呼ぶので `interp::run` と bit 一致（#2078 の不変条件）。GPU opt-in オンの `OnnxModel::run` は device 実行であり、autograd 経路との bit 一致は主張しない（別経路として記録）。
- backward: 解析的勾配。facade と内部の比較は bit 一致、GPU tape と CPU tape の比較は REQ-2 統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満。正本は tensor-core の定数）。
- `RELATIVE_TOLERANCE`／`ABSOLUTE_RESCUE_THRESHOLD`・parity baseline・ガードレール閾値は変えない。baseline の追加はなし。FMA 契約・f64 長軸縮約の契約への影響はなし（Softmax・LayerNormalization は既に f64 アキュムレータ。§3.1）。

### 8.6 互換への影響

- 判定: **追加のみ**（`docs/api-compat-0.14.0-decision.md` §8 の手順 1〜4）。(1) 触れる型は §4 の行 25（`OnnxModel`・`OnnxValue`・`OnnxError`）。(2) `OnnxModel` はメソッド追加のみ、`OnnxError` は `#[non_exhaustive]` への variant 追加、新 3 型は新設（`OnnxAutogradOptions`・`OnnxTapeValue` は `#[non_exhaustive]`、`OnnxAutogradGraph` は private フィールドで外から作れない）、`OnnxValue` は不変。(3) 破壊的変更なし。(4) 移行ノート（#3780）の「追加」欄に載せる。`!`・`BREAKING CHANGE` は付けない。
- 内部クレートで 6 op のエラーが `UnsupportedOp` から `UnsupportedInAutograd` に変わる点は facade 外の挙動差として記録する（内部の `AutogradError` は `#[non_exhaustive]`）。

### 8.7 適用記録・索引・追跡表

- 適用記録の案（実装 PR #3842 が `docs/compat-api-scope.md` §5 へ書く）: `**適用記録（イシュー #3842・親 #3711。承認: <#3770 の回答コメント URL>）**: OnnxModel::bind_autograd と OnnxAutogradGraph・OnnxAutogradOptions・OnnxTapeValue を facade に公開。勾配経路 11 op（GPU tape でもホスト実行）、未対応 22 op は fail-closed。公開状態: 公開済み（実機 parity は #3864・#3865）。`
- 現在の公開状態: 未公開。
- チェック: `docs/README.md` の索引を更新した［x］／追跡表 行 #25 を更新した［x］／`docs/compat-api-scope.md` の適用記録［ ］（実装 PR が行う）。

### 8.8 ベンチと性能への影響

- 既定経路（`OnnxModel::run`・`from_*`・`to_*`）のコードは変えず bit 不変。`bind_autograd` は opt-in の新しい入口なので前後比較は要らない。計測はしないため数値は書かない。
- GPU tape でのホスト往復の性能は保証しない（rustdoc に書く）。性能計測は別 issue 候補（スコープ外。起票は承認後）。

### 8.9 依存の追加の有無

- 依存追加なし（facade は既に `fandhe-ai-onnx-interop`・`fandhe-ai-autodiff` に依存）。許容 12 区分・禁止リストの照合結果は不変。新しい `unsafe` なし。

### 8.10 所有者へ戻す論点（#3770 論点 1）

| 点 | 選択肢 | 推奨 | 確認がないときの扱い |
|----|--------|------|----------------------|
| 1. 公開形 | A メソッド `OnnxModel::bind_autograd`／B standalone 関数／C 対象外 | A | 確認があるまで実装へ進めない（allowlist 12 件のまま保護を維持） |
| 2. 未対応 op（22 op。§6 項 2 の 8 op を訂正） | fail-closed のまま公開／view・shape 系（`Reshape`・`Transpose`・`Squeeze`・`Unsqueeze`・`Flatten`）の勾配を先に実装／対象外 | fail-closed のまま公開。view・shape 系は後続（起票は承認後） | 同上 |
| 3. 6 op（`Conv` ほか）の誤診断 | #3842 で明示 fail-closed 腕へそろえる／いまのまま | そろえる | 同上 |
| 4. ホスト実行（11 op すべて。§6 項 3 を訂正） | 性能非保証として受け入れる／GPU 実装を待つ | 受け入れる | 同上 |
| 5. `Cast` の勾配切断・`trainable` 既定 | 現行の意味論のまま／既定を空集合にする | 現行のまま（PyTorch の `requires_grad=True` 既定と同じ） | 同上 |
| 6. パラメータ更新経路 | feed 上書きで学習（新 API なし）／initializer 書き戻し API を足す | feed 上書き。書き戻しと `to_bytes` は §7 のとおり別 issue | 同上 |
| 7. 型名 | `OnnxAutogradGraph`・`OnnxAutogradOptions`・`OnnxTapeValue`／別名 | 左の案（禁止部分文字列と衝突しない） | 同上 |
| 8. GPU opt-in との関係 | `bind_autograd` は opt-in フラグを見ない／見る | 見ない（device は `Tape` で決まる） | 同上 |

### 8.11 セキュリティ・スコープ外・Closes／Refs

- OWASP: A03 は §8.3 の検証順（信頼できない ONNX の initializer 葉化は失敗可能な確保のまま、検証を計算・確保の前に行う）。A04 は未対応 op を無言で飛ばさず fail-closed、`trainable` の typo を拒否（fine-tuning が何も更新しないまま成功するのを防ぐ）。A06 は依存変更なし。A08 はガードレール・tolerance を迂回する経路を作らず、公開面を allowlist と否定プローブで固定。
- スコープ外（起票は所有者承認後）: view・shape 系の勾配／CNN 系 op の勾配／書き戻しと export／GPU 上のネイティブ実行／性能計測／制御フロー op／optimizer state。
- 本 PR は docs のみで `Closes #3759`。`Refs #3770 #3842 #3843 #3864 #3865`。
