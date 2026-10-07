# 逆三角関数・双曲線関数（`atan`・`asin`・`acos`・`atan2`・`sinh`・`cosh`・`asinh`・`acosh`・`atanh`）の CPU 実装記録（イシュー #2634）

親: #2625（Phase 4）／ルート: #2499。`docs/autodiff-scalar-unary-ops-decision.md`（#2145）と
`docs/autodiff-fft-ops-decision.md`（#2631）の方式を再適用した実装記録であり、**承認記録ではない**
（facade 公開形の承認は #2677 で依頼中。公開自体は承認後の #2678）。

## 0. 結論

- 9 演算の forward と VJP を CPU 参照実装として内部クレートへ追加した。新規 `Op`・新規 `BackendOps`
  メソッドは追加せず、既存の `Op::ScalarUnary`／`Op::ScalarBinary` へ薄く委譲する。
  - `tensor_core::ScalarUnaryOp` に 8 variant（`Atan`／`Asin`／`Acos`／`Sinh`／`Cosh`／`Asinh`／
    `Acosh`／`Atanh`）、`ScalarBinaryOp` に `Atan2` を追加（いずれもペイロードなし。`#[non_exhaustive]`
    への variant 追加のため `fandhe-ai =0.10.0` の公開 API は壊さない）。
  - 入口は自由関数モジュール `fandhe_ai_autodiff::trig_ops`（`crates/autodiff/src/trig_ops.rs`）。
- facade 公開は行わない。`TrigOpsHoldDoctestGuard`（`crates/facade/src/lib.rs`）と
  `crates/facade/tests/api_surface.rs` の否定ガードで機械固定した。
- CUDA／Metal の専用カーネルは対象外。明示 `None` → 既定 `Unsupported` → ホスト参照実装への
  フォールバックで動作する。実機テストは `#[ignore]` のまま未実測。

## 1. 着手時の判定

- 着手時点で #2634 にコメントは 0 件、承認依頼 #2677 は open でコメント 0 件。**承認記録は確認できなかった**。
- よって内部実装と保留ガードまでに留めた（公開は #2678 の担当）。

## 2. 実装方式

- 新規 `Op` なし。`Var::scalar_unary`／`scalar_binary`（`pub(crate)`）が forward 実体化・バックエンド dispatch・
  tape 記録を担う共通経路。`trig_ops` は variant を選ぶ 1 行委譲の自由関数のみ。
- 既存 `scalar_unary_ops` へは足さず新モジュールにした。同モジュールの fn 集合は facade の正ガードに
  固定されているため。
- `backend-cpu::scalar_elementwise` は `apply` を汎用ループで呼ぶので src の変更は不要。
- 区分定数ではないため `is_piecewise_constant`／`is_comparison` は不変（`false`）。

## 3. 数値契約

forward（単一情報源は `ScalarUnaryOp::apply`／`ScalarBinaryOp::apply`）:

- `f32` 標準ライブラリの同名関数をそのまま使い、定義域外・極・overflow は IEEE のまま伝播する
  （マスクも panic もしない。`Tan`／`Log` と同規約）。
- `atan2(y, x)` の引数順は `torch.atan2(input, other)` と同じ（`a = y`・`b = x`）。
- `asinh(±3e38)`・`acosh(3e38)` が有限値になることを単体テストで固定した（rustc 1.98.1 で実測）。
  失敗する toolchain が現れた場合は `f64` で計算して 1 回 downcast へ切り替える。

VJP 係数（`autodiff::eval::scalar`）は `f64` へ昇格して計算し、最後に 1 回だけ `f32` へ downcast する
（`erf_grad` と同型。PR #1686 で指摘された overflow／underflow 類型の先回り）:

| 演算 | 係数 |
|---|---|
| `Atan` | `1 / (1 + x²)` |
| `Asin` | `1 / sqrt((1 - x)(1 + x))` |
| `Acos` | `-1 / sqrt((1 - x)(1 + x))` |
| `Sinh` | `cosh(x)`（入力 `x` から計算。出力 `y` からは復元しない） |
| `Cosh` | `sinh(x)`（同上。`y` から復元すると符号が失われる） |
| `Asinh` | `1 / sqrt(x² + 1)` |
| `Acosh` | `1 / sqrt(x² - 1)` |
| `Atanh` | `1 / (1 - x²)` |
| `Atan2` | `da = b / (a² + b²)`・`db = -a / (a² + b²)`（原点は `(0, 0)`） |

- `Atan2` の broadcast 縮約は既存の `reduce_bias_grad`（`f64` アキュムレータ）を通る。
- `eval/scalar.rs` の `unary_grad_factor`／`binary_partials` は末尾が `unreachable!()` のため、全 9 variant の
  backward を単体テスト・統合テストで踏んで漏れを検出する。

## 4. GPU

- `crates/backend-cuda/src/kernels_scalar_op.rs`・`crates/backend-metal/src/scalar_op_source.rs` の
  `unary_expr`／`binary_expr` に新 9 kind の明示 `None` arm を追加した（末尾ワイルドカードに任せない
  リポジトリ規約）。`new_2634_kinds_are_unsupported` が両クレートで固定する。
- `CudaBackendOps` は対応 kind 判定をデバイス取得より前に行うため、CUDA 非搭載 CI でも `CudaUnavailable`
  ではなく `Unsupported` を返す（`crates/backend-cuda/tests/scalar_op_parity.rs::
  new_2634_kinds_return_unsupported_without_touching_device`。属性なしで通常 CI が実行する）。

## 5. PyTorch 2.14.0 との差分

fixture は実 PyTorch 2.14.0+cpu の実行値（`crates/autodiff/tests/fixtures/trig-ops-pytorch-reference/`）。
`cases` 28 件は forward・勾配とも REQ-2 統一複合判定で一致。`edge_cases`（境界・外側・巨大値・符号付きゼロ）の
値クラス（NaN／+inf／-inf／有限）も `atan2` の 2 項目を除いて一致した。

| 項目 | PyTorch 2.14.0 | 本実装 | 扱い |
|---|---|---|---|
| `atan2(0, 0)` の勾配 | `0` | `0`（実測に合わせて原点のみ `(0, 0)`） | 実装を実測へ合わせた |
| `atan2(1e-30, 1e-30)` の勾配 | `0`（`f32` で分母が underflow） | 数学的に正しい有限値（約 `5e29`） | 意図的な差分。テストで自前の挙動を明示 assert |
| `atan2(1e20, 1e20)` の勾配 | `0`（`f32` で分母が overflow） | 数学的に正しい有限値（約 `5e-21`） | 同上 |

tolerance・baseline は変更していない。

## 6. テスト構成

- `crates/tensor-core/src/scalar_op.rs`（単体）: 既知値・定義域境界・巨大有限入力・符号付きゼロ・NaN 伝播・`atan2` 4 象限・
  `kind_name`・区分定数でないこと。
- `crates/autodiff/src/eval/scalar.rs`（単体）・`grad.rs`（中心差分）: 係数の既知値・underflow／overflow 回帰。
- `crates/autodiff/tests/trig_ops_parity.rs`: fixture 突合・境界クラス突合・f64 中心差分（broadcast 縮約を含む）・
  `apply` との bit 一致・`Unsupported` フォールバックの forward／backward・`Unsupported` 以外のエラー伝播・形状／別 tape の拒否。
- `crates/backend-cpu/tests/scalar_op_parity.rs`: 新 kind を全 variant 一覧へ追加し、定義域内入力の専用テストを追加。
- `crates/facade/tests/trig_ops_backend_parity.rs`: CPU tape と NaiveOps tape の突合（属性なし）。CUDA／Metal 実機は `#[ignore]`。

## 7. facade 公開形の推奨案（未承認）

推奨は 1 つ。`Var` の inherent メソッドとして `trig_ops` への 1 行委譲で公開する。

- `Var::atan`／`asin`／`acos`／`sinh`／`cosh`／`asinh`／`acosh`／`atanh`（`&self -> Result<Var<'t>, AutodiffError>`）
- `Var::atan2(&self, other: &Var<'t>)`（`self = y`・`other = x`。`torch.atan2(input, other)` と同じ引数順）
- `trig_ops` モジュールと `ScalarUnaryOp`／`ScalarBinaryOp` は再エクスポートしない。層ではないので `Sequential::add_*` は設けない。
- inherent メソッドの追加のみで非破壊（#2512 と同型）。承認依頼は #2677、公開は承認後の #2678。
- 承認後は `TrigOpsHoldDoctestGuard` と否定ガードを承認形の正ガード（委譲本体の固定を含む）へ反転する。

## 8. スコープ外

- facade 公開（#2677 の承認後に #2678）。
- CUDA／Metal の専用カーネル（`atanf`／`asinhf`／`metal::precise::atan2` 等）と実機 parity の実測。
- `create_graph`（高階微分）対応、activation checkpoint の対象化。`create_graph::scalar_unary_replayable`／
  `scalar_binary_replayable` は新 kind に対し `false`（末尾ワイルドカード）。
- f64 自動微分経路（`f64_autograd`）と f16／bf16 への展開。
- ONNX `Atan`／`Asin`／`Acos`／`Sinh`／`Cosh`／`Asinh`／`Acosh`／`Atanh` の import／export。
- 非追跡 `Tensor` や `compat::array` への同名メソッド追加。
- `docs/compat-api-scope.md` 1 節の対象範囲表と `docs/compat-feature-gap.md` の判定変更、spec（REQ-9）の改定。

## 9. 多層防御（保留ガード）

| ガード | 内容 |
|---|---|
| `TrigOpsHoldDoctestGuard`（`crates/facade/src/lib.rs`） | 全 `pub mod` glob import 下で、同名の関数・メソッドが公開されるとコンパイルが失敗する正のプローブ |
| `trig_ops_hold_doctest_globs_all_pub_modules` | doctest の glob 集合と `pub mod` 宣言集合の一致 |
| `trig_ops_hold_doctest_probe_body_matches_fixed_contract` | プローブ本文の固定文言一致 |
| `facade_does_not_reexport_or_declare_trig_ops`（＋自己テスト） | facade src の再エクスポート・`pub mod trig_ops`・9 名の `fn` 宣言の否定検査 |
| `workspace_declares_trig_ops_fn_names_only_in_allowed_locations` | workspace 全体で 9 名の `fn` 宣言が `autodiff/src/trig_ops.rs` の各 1 件のみ |

## 10. 実機申し送り

CUDA（DGX Spark GB10）・Metal（Apple Silicon）の実機テスト（`cuda_trig_ops_matches_cpu_reference`・
`metal_trig_ops_matches_cpu_reference`）は `#[ignore]` のまま未実測。手順は `docs/perf/logs/trig-ops-2634/README.md`。

## 11. 出典

- `docs/autodiff-scalar-unary-ops-decision.md`（#2145）・`docs/autodiff-fft-ops-decision.md`（#2631）・`docs/scalar-op-dispatch-design.md`
- `docs/compat-api-scope.md` 5 節（適用記録）・`.claude/rules/coding-rust.md`（REQ-2 判定・カーネル境界検査）
- PyTorch 2.14.0 実行値: `crates/autodiff/tests/fixtures/trig-ops-pytorch-reference/README.md`

## 12. #2678 実装記録（Phase 4 の facade 公開）

- 状態: **§7 の公開形を #2678 で承認形どおり公開した。** 承認根拠はルート #2499 の 2026-10-07 ユーザー承認コメント（issuecomment-6033824965。「Phase 4（#2625）」節で `docs/compat-api-scope.md` §5.1 の行 3を各決定記録の推奨形で承認）。本書中の「未承認」「承認依頼は #2677」の記述は、#2678 時点で当該コメントの承認に更新された（承認は推奨形に限り、記録に形が書かれていない点は実装せず承認依頼へ戻す条件つき）。
- 公開した識別子: `Var::{atan,asin,acos,sinh,cosh,asinh,acosh,atanh}(&self)`・`Var::atan2(&self, other: &Var)`（`self` が y）。本体は `crate::<module>::<fn>` への 1 行委譲（`Var`）／`&self.0` を渡すだけの 1 行委譲（`Tape`）に固定し、新規 `Op`・`BackendOps` メソッド・`AutodiffError` variant・`unsafe` は追加していない。
- ガードの反転・縮小: `TrigOpsHoldDoctestGuard` から `Var` の impl・UFCS 行を外し、`trig_ops` モジュール名・`Tape` 上の同名メソッドのプローブだけを残した（先例 #2516）。`api_surface.rs` の否定ガードは、承認済みの型名を識別子表から外し、`Tape` の承認済みメソッドを `fn` 宣言走査から除外したうえで、承認形だけを許す正ガードへ反転した。宣言場所インベントリには `autodiff/src/var.rs`（`Tape` 分は `facade/src/lib.rs`）の各 1 件を追加した。
- テスト: `crates/facade/tests/phase4_ops_facade.rs`（trig_methods_values_and_gradient。`fandhe_ai::` だけを import し、fn ポインタ型でシグネチャを固定して厳密に決まる値を確認）と、`crates/facade/tests/api_surface.rs` の正ガード（`var_phase4_ops_methods_are_thin_delegations`・`facade_reexports_phase4_ops_types_only_in_approved_shape`・`facade_tape_phase4_methods_are_thin_delegations`・各 `workspace_declares_*`）。
- 依存・tolerance・baseline・ガードレール閾値・`docs/spec` は変更していない。`fandhe-ai =0.10.0` の既存公開 API・`pub use` 行・署名は変更せず、追加のみ。CUDA／Metal 実機 parity は未実測で、`docs/perf/logs/phase4-ops-autodiff-exposure-2678/README.md` へ申し送る（新しい数値経路はなく、1 行委譲のため既存の各 `*_backend_parity.rs` の結果がそのまま適用される）。
