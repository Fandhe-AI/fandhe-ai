# 非有限値の判定・置換（`isnan`・`isinf`・`isfinite`・`nan_to_num`）の CPU 実装記録（イシュー #2635）

親: #2625（Phase 4）／ルート: #2499。`docs/autodiff-scalar-unary-ops-decision.md`（#2145）・
`docs/autodiff-bool-ops-exposure-decision.md`（#2141）・`docs/autodiff-trig-ops-decision.md`（#2634）の方式を
再適用した実装記録であり、**承認記録ではない**（facade 公開形の承認は #2677 で依頼中。公開自体は承認後の #2678）。

## 0. 結論

- 非有限値の判定 3 種（`isnan`／`isinf`／`isfinite`。bool 出力・非微分）と置換 `nan_to_num`（微分可能）を
  CPU 参照実装として内部クレートへ追加した。新規 `Op`・新規 `BackendOps` メソッドは追加せず、既存の
  `Op::ScalarUnary` と `scalar_unary` 経路へ載せる。
  - `tensor_core::ScalarUnaryOp` に 4 variant を追加: `IsNan`／`IsInf`／`IsFinite`（`1.0`／`0.0` の f32 マスク）と
    `NanToNum { nan, posinf, neginf }`（置換値は解決済みの `f32`）。`#[non_exhaustive]` への variant 追加のため
    `fandhe-ai =0.10.0` の公開 API は壊さない。
  - 入口は自由関数モジュール `fandhe_ai_autodiff::nonfinite_ops`（`crates/autodiff/src/nonfinite_ops.rs`）。
- facade 公開は行わない。`NonfiniteOpsHoldDoctestGuard`（`crates/facade/src/lib.rs`）と
  `crates/facade/tests/api_surface.rs` の否定ガードで機械固定した。
- 依存・`unsafe`・tolerance・baseline・`docs/spec/`・ガードレール閾値は変更していない。
- CUDA／Metal の専用カーネルは対象外。明示 `None` → 既定 `Unsupported` → ホスト参照実装へのフォールバックで
  動作する。実機テストは `#[ignore]` のまま未実測。

## 1. 着手時の判定（事実のみ）

- 本実装は #2635 の受入条件（内部実装・決定記録・保留ガード）に限る。公開面・対象範囲の拡張の承認は確認して
  いない（承認依頼 #2677 は別途）。よって内部実装と保留ガードまでに留めた（公開は #2678 の担当）。

## 2. 実装方式

| 層 | 置き場所 | 内容 |
|---|---|---|
| forward 数式 | `tensor-core::ScalarUnaryOp::apply` | 単一情報源。`IsNan` は `x.is_nan()`、`IsInf` は `x.is_infinite()`、`IsFinite` は `x.is_finite()` を `1.0`／`0.0` で返す。`NanToNum` は「NaN → `nan`／`+inf` → `posinf`／`-inf` → `neginf`／それ以外は `x` をビットを変えず通す」の順 |
| CPU `BackendOps` | `backend-cpu::scalar_elementwise::scalar_unary` | 全 variant を `op.apply` へ委譲するためコード変更なし |
| 判定 3 種 | `autodiff::nonfinite_ops::{isnan, isinf, isfinite}` | 確保前サイズ検査 → 層 1 実体化 → `scalar_unary_with_fallback` → `cast_from_f32_with_fallback::<bool>` → shape 事後検査。tape へは積まない |
| `nan_to_num` | `autodiff::nonfinite_ops::nan_to_num` | 確保前サイズ検査 → 既定値解決 → `Var::scalar_unary`（`pub(crate)`）へ委譲。tape へ `Op::ScalarUnary` を記録 |
| VJP | `autodiff::grad` の `Op::ScalarUnary` 分岐 | `NanToNum` 専用腕（§3）。判定 3 種は `is_piecewise_constant()` で既存のゼロテンソル直接生成へ |
| GPU | `backend-cuda::kernels_scalar_op`・`backend-metal::scalar_op_source` | `unary_expr` の明示 `None` 腕 |

- `ScalarUnaryOp` 拡張を選んだ理由: `ScalarBinaryOp::Gt` 等が既に 0/1 マスクを返す契約と整合し、`Unsupported` →
  ホストフォールバックを持つ経路に載るため受入条件の「フォールバックへ到達する」をテストで検証できる（純ホスト
  ループでは検証対象のフォールバックが存在しない）。
- 命名規律: 関数名はちょうど `isnan`／`isinf`／`isfinite`／`nan_to_num` の 4 件とし、同名の `fn` を workspace の
  他の場所へ宣言しない（保留ガードのインベントリが各 1 件に固定する）。
- `NanToNum` は NaN を伝播しない最初の kind。既存の「NaN 伝播」テストの明示リストへは入れていない。
  置換値が NaN の場合、`PartialEq`（派生）は自分自身と等しくならない（`Clamp` と同じ性質。等値比較に依存する箇所は無い）。

## 3. 数値契約

forward（真偽表）:

| 入力 | `isnan` | `isinf` | `isfinite` |
|---|---|---|---|
| NaN（符号・ペイロード問わず） | 真 | 偽 | 偽 |
| `±inf` | 偽 | 真 | 偽 |
| `±0.0`・非正規化数・`f32::MAX`／`MIN`・通常値 | 偽 | 偽 | 真 |

- `nan_to_num` の既定値は PyTorch と同じ（`nan = 0.0`・`posinf = f32::MAX`・`neginf = f32::MIN`）。有限値（`±0`・
  非正規化数を含む）はビットを変えずに通す。丸めは一切入らない。置換値自体が非有限でも拒否せず書き込む
  （PyTorch と同じ）。

VJP（`nan_to_num`）:

- 勾配は有限入力位置のみ上流をそのまま通し、非有限入力位置は 0。`grad.rs` は `NanToNum` 専用の腕で
  `elementwise_mul_mask(upstream, x_val, |v| v.is_finite())`（乗算を経由しない要素選択。`Op::Relu` の VJP と同じ
  ヘルパ）を返す。
- 係数 0 を `vjp_elementwise_mul` へ渡すと、上流が `inf`／`NaN` のとき `0 * inf = NaN` に汚染される（PR #1823・
  #2145 で是正済みの類型の再発）ため選択方式にした。判定 3 種は区分定数（`is_piecewise_constant() == true`）で
  既存のゼロテンソル直接生成を使う。
- `eval::scalar::unary_grad_factor` にも 4 件の腕を追加した（判定 3 種は `0.0`、`NanToNum` は有限入力で `1.0`）。
  `Var::scalar_unary` を crate 内から直接呼んだ場合に `unreachable!` へ落ちないための腕。

## 4. 境界検査

- 判定 3 種・`nan_to_num` とも、入力 shape に対し `checked_bytes_for::<f32>`（判定は加えて `::<bool>`）を実体化・
  確保より前に呼び、`isize::MAX` 超過を `ShapeError::ElementCountOverflow` の型付きエラーで拒否する（要素数 1 の
  `Var` を巨大 shape へ `broadcast_to` した view を渡されても panic しない。`.claude/rules/coding-rust.md`）。
- フォールバック条件は `BackendError::Unsupported` のみ。それ以外のバックエンドエラーは握りつぶさず伝播する。
  バックエンドの戻り値 shape が入力と異なる場合は `BackendError::ShapeMismatch` で拒否する。

## 5. PyTorch 2.14.0 との差分

fixture は実 PyTorch 2.14.0+cpu の実行値（`crates/autodiff/tests/fixtures/nonfinite-pytorch-reference/`。f32 は
NaN／inf を運べるよう u32 ビットパターンで保存）。判定 3 種は bool 完全一致、`nan_to_num` の forward は bit 一致
（NaN はクラス一致）、勾配は REQ-2 統一複合判定で一致した。

| 項目 | PyTorch 2.14.0 | 本実装 | 扱い |
|---|---|---|---|
| 上流勾配が非有限かつ入力も非有限の位置の `nan_to_num` 勾配 | NaN（`grad * isfinite(x)` で `0 * inf`／`0 * NaN`。fixture の `upstream_nonfinite_observations` で実測） | 0 | 意図的な差分（要素選択のため汚染しない）。入力が有限の位置は上流の非有限値を通し PyTorch と一致 |
| 引数の型 | `float?` | `Option<f32>` | `None` は PyTorch 既定へ解決 |
| 整数 dtype・complex・`out=` 引数 | あり | 非対応 | `Var` は f32 のみ |

tolerance・baseline は変更していない。

## 6. テスト構成

- `crates/tensor-core/src/scalar_op.rs`（単体）: 真偽表（符号付き NaN・`±inf`・`±0`・非正規化数・`MAX`／`MIN`）、
  `NanToNum` の置換とビット不変、NaN を伝播しない最初の kind であること、区分定数フラグと `kind_name`。
- `crates/autodiff/src/eval/scalar.rs`・`grad.rs`（単体）: 係数、`NanToNum` の中心差分、上流 `inf`・入力非有限でも勾配が有限。
- `crates/autodiff/src/nonfinite_ops.rs`（単体）: 既知値・tape 非記録・巨大 broadcast view の拒否。
- `crates/autodiff/tests/nonfinite_parity.rs`: fixture 突合（36 件）・観測記録・モック `BackendOps` による
  `Unsupported` フォールバックと他エラーの伝播と誤 shape・中心差分・view 入力・要素数 0・スカラー・
  `create_graph` が型付きエラーであること・run-to-run 決定性。
- `crates/backend-cpu/tests/scalar_op_parity.rs`・`backend_ops_dispatch.rs`: 4 kind の逐次ホスト参照との bit 一致
  （非有限入力を含む）、CUDA（macOS では Metal も）の `scalar_unary` が `Unsupported` を返し panic しないこと。
- `crates/backend-cuda`／`backend-metal`: `new_2635_unary_kinds_are_unsupported`。
- `crates/facade/tests/nonfinite_ops_backend_parity.rs`: CPU tape と NaiveOps tape の突合（属性なし）。CUDA／Metal 実機は `#[ignore]`。

## 7. facade 公開形の推奨案（未承認）

推奨は 1 つ。`Var` の inherent メソッドとして `nonfinite_ops` への 1 行委譲で公開する。

- `Var::isnan`／`isinf`／`isfinite`（`&self -> Result<Tensor<bool>, AutodiffError>`）
- `Var::nan_to_num(&self, nan: Option<f32>, posinf: Option<f32>, neginf: Option<f32>)`（`-> Result<Var<'t>, AutodiffError>`）
- `nonfinite_ops` モジュールと `ScalarUnaryOp` は再エクスポートしない。層ではないので `Sequential::add_*` は設けない。
- inherent メソッドの追加のみで非破壊（#2510 の `Var::gt_bool` 等・#2512 の `Var::floor` 等と同型）。承認依頼は #2677、
  公開は承認後の #2678。
- 承認後は `NonfiniteOpsHoldDoctestGuard` と否定ガードを承認形の正ガード（委譲本体の固定を含む）へ反転する。

承認事項（**すべて未承認**）: 上記 4 メソッドの公開、およびメソッド名・引数形（`Option<f32>` 3 つ）。

## 8. スコープ外

- facade 公開（#2677 の承認後に #2678）。
- CUDA／Metal の専用カーネル。将来足す場合、`NanToNum` は 3 値ペイロード（既存の単一値ペイロードでは足りない）が
  必要になる。実機 parity の実測もそのとき。
- `create_graph`（高階微分）・activation checkpoint・f64 自動微分経路（型付きエラーで拒否されることだけテストで固定）。
- `docs/compat-api-scope.md` 1 節の対象範囲表と `docs/compat-feature-gap.md` の判定変更、spec（REQ-9）の改定。

## 9. 多層防御（保留ガード）

| ガード | 内容 |
|---|---|
| `NonfiniteOpsHoldDoctestGuard`（`crates/facade/src/lib.rs`） | 全 `pub mod` glob import 下で、同名の関数・メソッドが `Var`／`Tape`／`Tensor<f32>`／`Tensor<bool>` に公開されるとコンパイルが失敗する正のプローブ |
| `nonfinite_ops_hold_doctest_globs_all_pub_modules` | doctest の glob 集合と `pub mod` 宣言集合の一致 |
| `nonfinite_ops_hold_doctest_probe_body_matches_fixed_contract` | プローブ本文の固定文言一致 |
| `facade_does_not_reexport_or_declare_nonfinite_ops`（＋自己テスト） | facade src の再エクスポート・`pub mod nonfinite_ops`・4 名の `fn` 宣言の否定検査 |
| `workspace_declares_nonfinite_ops_fn_names_only_in_allowed_locations` | workspace 全体で 4 名の `fn` 宣言が `autodiff/src/nonfinite_ops.rs` の各 1 件のみ |

stable rustdoc は `compile_fail` のコードを照合しないため、否定ガードは正のプローブ＋インベントリで組んでいる。

## 10. 実機申し送り

CUDA（DGX Spark GB10）・Metal（Apple Silicon）の実機テスト（`cuda_nonfinite_ops_matches_cpu_reference`・
`metal_nonfinite_ops_matches_cpu_reference`）は `#[ignore]` のまま未実測。手順は `docs/perf/logs/nonfinite-ops-2635/README.md`。

## 11. 出典

- `docs/autodiff-scalar-unary-ops-decision.md`（#2145）・`docs/autodiff-bool-ops-exposure-decision.md`（#2141）・
  `docs/autodiff-trig-ops-decision.md`（#2634）・`docs/scalar-op-dispatch-design.md`
- `docs/compat-api-scope.md` 5 節（適用記録）・`.claude/rules/coding-rust.md`（REQ-2 判定・カーネル境界検査）
- PyTorch 2.14.0 実行値: `crates/autodiff/tests/fixtures/nonfinite-pytorch-reference/README.md`
