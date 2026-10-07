# SELU・CELU・Softsign・Hardsigmoid・LogSigmoid の CPU 実装記録（イシュー #2649）

親: #2648（不足している活性化関数）／兄弟: #2650（Softmin・Tanhshrink・RReLU・Threshold）。
`docs/autodiff-trig-ops-decision.md`（#2634）の方式を再適用した実装記録であり、**承認記録ではない**
（facade 公開形の承認は #2677 で依頼中。公開自体は承認後の #2678〈`Var` 委譲〉・#2679〈`compat::Sequential::add_*`〉）。

## 0. 結論

- 5 演算の forward と VJP を CPU 参照実装として内部クレートへ追加した。新規 `Op`・新規 `BackendOps`
  メソッドは追加せず、既存の `Op::ScalarUnary` へ薄く委譲する。
  - `tensor_core::ScalarUnaryOp` に 5 variant（`Selu`／`Celu { alpha }`／`Softsign`／`Hardsigmoid`／
    `LogSigmoid`）を追加（`#[non_exhaustive]` への variant 追加のため `fandhe-ai =0.10.0` の公開 API は壊さない）。
  - 入口は自由関数モジュール `fandhe_ai_autodiff::activation_scalar_ops`（`crates/autodiff/src/activation_scalar_ops.rs`）と、
    `nn::activation` の層 5 型（`Selu`・`Celu`・`Softsign`・`Hardsigmoid`・`LogSigmoid`。`Module` impl 付き）。
- facade 公開は行わない。`ActivationScalarOpsHoldDoctestGuard`（`crates/facade/src/lib.rs`）と
  `crates/facade/tests/api_surface.rs` の否定ガードで機械固定した。
- CUDA／Metal の専用カーネルは対象外。明示 `None` → 既定 `Unsupported` → ホスト参照実装への
  フォールバックで動作する。実機テストは `#[ignore]` のまま未実測。

## 1. 着手時の判定

- 着手時点で #2649 にコメントは 0 件、承認依頼 #2677 は open でコメント 0 件。**承認記録は確認できなかった**。
- よって内部実装と保留ガードまでに留めた（公開は #2678・#2679 の担当）。

## 2. 実装方式

- 5 演算とも shape 不変の要素ごと演算で、既存の `Op::ScalarUnary` に載る（#2634 型）。既存 Op の合成（#2146 型）は採らない。
  - PyTorch の境界勾配（Hardsigmoid は開区間 `(-3, 3)`、SELU／CELU は `x <= 0` 側）を係数で直接書ける
    （合成の `x + 3` は丸めで境界が 1 ulp ずれる）。
  - VJP 係数を `f64` 昇格・1 回 downcast で書ける（Softsign の `(1 + |x|)²` は `f32` だと `|x| ≈ 3e38` で overflow）。
  - 層の `forward_host` が `scalar_unary_with_fallback` を共有するため tape 経由 forward と構造的に bit 一致する。
- 命名: 自由関数 `selu`・`celu`・`softsign`・`hardsigmoid`・`log_sigmoid`。PyTorch の `F.logsigmoid` は既存 `log_softmax` の
  snake_case 規則に揃えて `log_sigmoid` とした（`kind_name` も `log_sigmoid`）。
- 既存 `activation_ops` へは足さず新モジュールにした（同モジュールの fn 集合は #2146／#2516／#2529 のガードが固定済み）。
- `backend-cpu::scalar_elementwise` は `apply` を汎用ループで呼ぶので src の変更は不要。区分定数ではないため
  `is_piecewise_constant` は不変（`false`）。

## 3. 数値契約

forward（単一情報源は `ScalarUnaryOp::apply`）。SELU 定数は `tensor_core::scalar_op::{SELU_ALPHA, SELU_SCALE}`
（PyTorch と同値。積は AlphaDropout の既存定数 `1.7580993408473766` と厳密一致することを単体テストで固定）。

| 演算 | forward（`f32`） | VJP 係数（`f64` で計算し 1 回 downcast） |
|---|---|---|
| `Selu` | `x > 0` なら `scale * x`、それ以外は `(alpha * scale) * expm1(x)`（各定数を先に `f32` へ narrow してから掛ける） | `x > 0` なら `scale`、それ以外は `alpha * scale * exp(x)`（`y` から復元しない。`Elu` の PR #1686 の教訓） |
| `Celu { alpha }` | `x > 0` なら `x`、それ以外は `alpha * expm1(x / alpha)` | `x > 0` なら `1`、それ以外は `exp(x / alpha)` |
| `Softsign` | `x / (1 + abs(x))` | `1 / (1 + abs(x))²` |
| `Hardsigmoid` | `relu6(x + 3) / 6` | 開区間 `-3 < x < 3` で `1/6`、それ以外（NaN 含む）は `0`（`tensor_core::scalar_op::hardsigmoid_grad`） |
| `LogSigmoid` | NaN は NaN、それ以外は `min(x, 0) - ln_1p(exp(-abs(x)))` | `1 / (1 + exp(x))`（= `sigmoid(-x)`） |

- `x == ±0` は SELU／CELU の負側の枝（PyTorch の `x <= 0` 判定と同じ）。NaN は forward で伝播する。
- LogSigmoid の NaN 明示分岐は `Relu`／`Clamp`／`Sign` と規約を揃えるためのもの（`f32::min(NaN, 0.0)` が `0.0` でも
  第 2 項が NaN になるため結果は元から NaN になる）。
- **式順**: SELU の負側係数の narrow 順・CELU の `x / alpha`（PyTorch は `x * (1/alpha)`）・Hardsigmoid 勾配の `upstream * (1/6)` は
  PyTorch と 1 ulp 程度ずれうる。bit 一致は主張せず REQ-2 判定で突合する。
- **CELU の引数検査**: 自由関数 `celu` と `Celu::new` の両方で `alpha == 0` と非有限 `alpha` を `AutodiffError::InvalidArgument` で拒否する
  （検査は tape 操作より前に終えるため孤児ノードを残さない。負の `alpha` は PyTorch と同じく受理）。`Celu::default()` は
  `alpha = 1.0` をフィールド直接構築する（本番経路の `expect` 禁止）。
- **Hardsigmoid の VJP は要素選択**: 係数 `0` を `vjp_elementwise_mul` へ渡すと上流が `inf`／`NaN` のとき `0 * inf = NaN` に
  汚染される（PR #1823・#2145・#2635 と同類型。PyTorch は領域外で literal `0`）。`grad.rs` の `Op::ScalarUnary` VJP に
  Hardsigmoid 専用の腕を置き、`NanToNum` と同じ `elementwise_mul_mask` で領域内だけ上流を通してから傾き `1/6` を掛ける。
  回帰テスト: `grad.rs::new_2649_hardsigmoid_backward_selects_instead_of_multiplying`・
  `activation_scalar_ops_parity.rs::hardsigmoid_backward_is_not_polluted_by_non_finite_upstream`。
- SELU／CELU／Softsign／LogSigmoid は恒等的にゼロの領域を持たない（係数が underflow で `0` になる点は PyTorch も乗算で同じ結果）ため
  汎用の係数乗算経路のまま。
- `eval/scalar.rs::unary_grad_factor` は末尾が `unreachable!()` のため、5 kind すべての backward を単体・統合テストで踏んで漏れを検出する。

## 4. GPU

- `crates/backend-cuda/src/kernels_scalar_op.rs`・`crates/backend-metal/src/scalar_op_source.rs` の `unary_expr` に
  5 kind の明示 `None` 腕を追加した（末尾ワイルドカードに任せない規約）。`new_2649_unary_kinds_are_unsupported` が両クレートで固定する。
- `CudaBackendOps` は対応 kind 判定をデバイス取得より前に行うため、CUDA 非搭載 CI でも `CudaUnavailable` ではなく `Unsupported` を返す
  （`crates/backend-cuda/tests/scalar_op_parity.rs::new_2649_kinds_return_unsupported_without_touching_device`。属性なし）。

## 5. PyTorch 2.14.0 との差分

fixture は実 PyTorch 2.14.0+cpu の実行値（`crates/autodiff/tests/fixtures/activation-scalar-ops-pytorch-reference/`。24 ケース＋境界 6 ケース＋
エラー 4 ケース）。`cases` は forward・勾配とも REQ-2 統一複合判定で一致。`edge_cases`（`±0`・`±3` と 1 ulp 内外・`±100`・`±3e38`・`±inf`・NaN）の
値クラスも下表の項目を除いて一致した（Hardsigmoid の `±3` 境界の勾配は PyTorch と同じ開区間判定で一致）。

| 項目 | PyTorch 2.14.0 | 本実装 | 扱い |
|---|---|---|---|
| SELU／CELU の NaN 入力の勾配 | 正側の係数（`x <= 0` が偽のため。SELU は `1.0507`・CELU は `1`） | NaN を伝播 | 意図的な差分。forward が NaN の位置で勾配だけ有限値を返さない。テストで明示 assert |
| Softsign の `±inf` 入力の勾配 | NaN（合成 autograd） | `0`（`f64` の係数 `1/(1+\|x\|)²`） | 意図的な差分。同上 |
| `F.celu(alpha=0)` | `RuntimeError` | `InvalidArgument` で拒否 | 一致（両方拒否） |
| `F.celu(alpha=NaN／±inf)` | 例外なし（`x <= 0` の要素が NaN、正の要素は恒等） | `InvalidArgument` で拒否 | 意図的な差分（非有限 `alpha` は入口で拒否） |
| Softsign の `±3e38` の勾配 | `0` | `0`（`f64` で `~1e-77` → `f32` で `0`） | 一致 |

tolerance・baseline は変更していない。

## 6. テスト構成

- `crates/tensor-core/src/scalar_op.rs`（単体）: 既知値・巨大有限入力・符号付きゼロ・NaN 伝播・`±inf`・CELU の桁落ち回避・
  `kind_name`（`Celu` のペイロード非依存）・区分定数でないこと・SELU 定数の積・`hardsigmoid_grad` の境界。
- `crates/autodiff/src/eval/scalar.rs`（単体）・`grad.rs`（中心差分・Hardsigmoid 汚染回帰）・`create_graph.rs`（`false` の固定）。
- `crates/autodiff/tests/activation_scalar_ops_parity.rs`: fixture 突合・境界クラス・エラーケース・f64 中心差分・`apply` との bit 一致・
  Hardsigmoid の非有限上流・形状／非連続 view／空テンソル・`Unsupported` フォールバック・それ以外のエラー伝播。
- `crates/autodiff/tests/nn_activation_scalar_2649.rs`: 層の forward が自由関数と bit 一致・`Module::forward`／`forward_host` の一致・`Celu::new` の検査。
- `crates/backend-cpu/tests/scalar_op_parity.rs`: 全 variant 一覧へ追加し、負・ゼロ・正・境界入力の専用テストを追加。
- `crates/facade/tests/activation_scalar_ops_backend_parity.rs`: CPU tape と NaiveOps tape の突合（属性なし）。CUDA／Metal 実機は `#[ignore]`（5 演算すべて）。

## 7. facade 公開形の推奨案（ルート #2499 の 2026-10-07 コメントで承認・#2679 で公開。§12 参照）

推奨は 1 つ。#2146 → #2516／#2529 で確定済みの形をそのまま当てはめる。

- 関数形: `Var` の inherent メソッド 5 個を `activation_scalar_ops` への 1 行委譲で公開する。
  `Var::selu(&self)`・`Var::celu(&self, alpha: f32)`・`Var::softsign(&self)`・`Var::hardsigmoid(&self)`・`Var::log_sigmoid(&self)`
  （いずれも `Result<Var<'t>, AutodiffError>`）。公開は承認後の #2678。
- 層: `compat::Sequential` の `add_selu(self) -> Self`・`add_celu(self, alpha: f32) -> Result<Self, AutodiffError>`・`add_softsign(self) -> Self`・
  `add_hardsigmoid(self) -> Self`・`add_log_sigmoid(self) -> Self`。公開は承認後の #2679。
- `activation_scalar_ops` モジュール・層型・`ScalarUnaryOp` は再エクスポートしない。
- inherent メソッドの追加のみで非破壊。承認依頼は #2677。本節は推奨案の記録であり承認記録ではない。
- 承認後は `ActivationScalarOpsHoldDoctestGuard` と否定ガードを承認形の正ガードへ反転する。

## 8. スコープ外

- facade 公開（#2677 の承認後に #2678・#2679）。
- CUDA／Metal の専用カーネルと実機 parity の実測。
- `create_graph`（高階微分）対応、activation checkpoint の対象化。`create_graph::scalar_unary_replayable` は 5 kind に対し `false`。
- f64 自動微分経路と f16／bf16 への展開。
- ONNX `Selu`／`Celu`／`Softsign`／`HardSigmoid` の import／export。
- 非追跡 `Tensor` や `compat::array` への同名メソッド追加、`Module` の `as_*` フック・`save_model`／`load_model` の kind・resident 経路。
- 既存 kind（`Clamp`／`Hardswish`／`Relu` 等）の VJP を要素選択へ揃える変更。
- `docs/compat-api-scope.md` 1 節の対象範囲表・`docs/compat-feature-gap.md` の判定変更、spec（REQ-9）の改定、`MIN_KNOWN_PROBE_BLOCKS` の更新。
- 兄弟 #2650 の 4 演算。

## 9. 多層防御（保留ガード）

| ガード | 内容 |
|---|---|
| `ActivationScalarOpsHoldDoctestGuard`（`crates/facade/src/lib.rs`） | 全 `pub mod` glob import 下で、同名の型・関数・モジュール・`Var`／`Tape`／`Tensor` のメソッド・`Sequential::add_*` が公開されるとコンパイルが失敗する正のプローブ |
| `activation_scalar_ops_hold_doctest_globs_all_pub_modules` | doctest の glob 集合と `pub mod` 宣言集合の一致 |
| `activation_scalar_ops_hold_doctest_probe_body_matches_fixed_contract` | プローブ本文の固定文言一致 |
| `facade_does_not_reexport_or_declare_activation_scalar_ops`（＋自己テスト） | facade src の再エクスポート・層型の独自宣言・`pub mod`・fn 5 名と `add_*` 5 名の `fn` 宣言の否定検査 |
| `workspace_declares_activation_scalar_ops_fn_names_only_in_allowed_locations` | workspace 全体で fn 5 名が `autodiff/src/activation_scalar_ops.rs` の各 1 件のみ・`add_*` は 0 件 |

検出範囲は列挙した名前・型に限り、マクロ生成や別名経由の公開までは保証しない。

**反証確認（2026-10-06・ガード追加直後にローカルで実施。コミットしない）**: `crates/facade/src/lib.rs` へ仮に
`pub use fandhe_ai_autodiff::activation_scalar_ops;` を足すと `facade_does_not_reexport_or_declare_activation_scalar_ops` と
`ActivationScalarOpsHoldDoctestGuard` の doctest が FAILED（`E0659: activation_scalar_ops is ambiguous`）になり、外すと green に戻る。
`pub use fandhe_ai_autodiff::nn::activation::Selu;` でも同様（`E0659: Selu is ambiguous`）。

## 10. 実機申し送り

CUDA（DGX Spark GB10）・Metal（Apple Silicon）の実機テスト（`cuda_activation_scalar_ops_matches_cpu_reference`・
`metal_activation_scalar_ops_matches_cpu_reference`）は `#[ignore]` のまま未実測。手順は `docs/perf/logs/activation-scalar-ops-2649/README.md`。

## 11. 出典

- `docs/autodiff-trig-ops-decision.md`（#2634）・`docs/autodiff-scalar-unary-ops-decision.md`（#2145）・`docs/scalar-op-dispatch-design.md`
- `docs/compat-api-scope.md` 5 節（適用記録）・`.claude/rules/coding-rust.md`（REQ-2 判定・カーネル境界検査）
- PyTorch 2.14.0 実行値: `crates/autodiff/tests/fixtures/activation-scalar-ops-pytorch-reference/README.md`

## 12. #2679 実装記録（`Sequential::add_*` の facade 公開）


- 状態: **§7 のうち `compat::Sequential::add_*` 5 本を #2679 で公開した。** 承認根拠はルート #2499 の 2026-10-07 ユーザー承認コメント（issuecomment-6033824965。「Phase 4（#2625）」節で `docs/compat-api-scope.md` §5.1 の行 17 を各決定記録の推奨形で承認）。本書中の「未承認」「承認依頼は #2677」の記述は、#2679 時点で当該コメントの承認に更新された（#2677 の「承認の記録」コメントの割り振りでは公開は #2679）。承認は推奨形に限り、記録に形が書かれていない点は実装せず承認依頼へ戻す条件つき。
- 公開した識別子（`crates/facade/src/compat/sequential.rs`）: `add_selu(self) -> Self`・`add_celu(self, alpha: f32) -> Result<Self, AutodiffError>`・
  `add_softsign(self) -> Self`・`add_hardsigmoid(self) -> Self`・`add_log_sigmoid(self) -> Self`。層は内部実装（`nn::activation::{Selu, Celu, Softsign, Hardsigmoid, LogSigmoid}`）を
  `LayerSpec::Unsupported` で積む。`Var` の委譲メソッド 5 本は #2678 の担当で本イシューでは追加していない。
- 記録に形が書かれていない点の扱い: **`save_model`／`load_model` の manifest kind** はどの記録にも形がなく（§8 が `model_io` への kind 追加をスコープ外とする）、
  永続形式を無承認で広げない方針で kind を追加していない。`add_conv_transpose1d`／`add_unflatten`（#2521）と同じく `ModelIoError::UnsupportedModel` で拒否し、
  保存先に何も作らない（受入条件の「非対応なら型付きエラーで拒否」）。kind の追加は後から非破壊に行えるが形の決定が要る。層は無状態のため
  `nn::Module` の `as_*` フックは追加せず、`bind`／`trainable_parameters`／常駐経路（`forward_resident` 等）は層を素通しする（`Mish` と同じ）。
- ガード（§9）の反転: `ActivationScalarOpsHoldDoctestGuard` から `add_*` の UFCS プローブ・トレイト・impl を削除し、`Var`／`Tape`／`Tensor<f32>` の同名メソッド・層型・モジュールの
  プローブだけを残した（#2678 が `Var` 分を外す）。`add_*` の `fn` 宣言禁止は、`compat/sequential.rs` に承認シグネチャで各 1 件という正ガード
  （`compat_sequential_phase4_activation_layers_add_methods_have_approved_signatures`・`..._exposes_..._issue_2679`・workspace 宣言インベントリの期待値）へ反転した。
- テスト: `crates/facade/tests/compat_sequential_activation_scalar_layers.rs`（f64 閉形式参照との統一複合判定・学習経路・RReLU 以外の層でのパラメータ素通し・`fit`・常駐経路・保存と ONNX の拒否）、
  `compat_sequential_activation_scalar_layers_backend_parity.rs`（`#[ignore]`。実機は未実測で `docs/perf/logs/compat-sequential-activation-scalar-layers-2679/README.md` へ申し送り）。
- 依存・tolerance・baseline・ガードレール閾値・`docs/spec` は変更していない。`fandhe-ai =0.10.0` の既存公開 API・`pub use` 行・署名は変更せず、追加のみ。
