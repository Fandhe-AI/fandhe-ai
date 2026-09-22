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
