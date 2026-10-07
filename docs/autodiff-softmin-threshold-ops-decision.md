# Softmin・Tanhshrink・Threshold・RReLU（`softmin`・`tanhshrink`・`threshold`・`rrelu`）の CPU 実装記録（イシュー #2650）

- 対象: PyTorch の `F.softmin`／`F.tanhshrink`／`F.threshold`／`F.rrelu` と、層 `nn.Softmin`／`nn.Tanhshrink`／`nn.Threshold`／`nn.RReLU` 相当
- 親: #2648（不足している活性化関数）／Phase 親: #2625／ルート: #2499 Phase 4
- 実装: `crates/autodiff/src/softmin_threshold_ops.rs`（自由関数）・`crates/autodiff/src/nn/softmin_threshold.rs`（層 4 種と `Module` 実装）
- 範囲: **内部クレート限定**。facade 公開面（`Var` メソッド・`compat::Sequential::add_*`）の拡張は本イシューの範囲外（承認依頼 #2677・承認後の #2678・#2679）

## 0. 結論

- 新規 `Op`・`BackendOps` メソッド・VJP・GPU カーネルは追加しない。既存の `Var::neg`／`softmax`／`sub`／`tanh`／`masked_fill`／`mul` と `Tape::var_no_grad` の合成のみで実装する。CPU／CUDA／Metal の全バックエンドに到達できる。
- facade 非公開を保留ガード（`SoftminThresholdOpsHoldDoctestGuard` と `api_surface.rs` の否定ガード）で機械固定した。
- PyTorch 2.14.0 の実行値 fixture（`cases` 33 件・`edge_cases` 8 件）と REQ-2 統一複合判定で突合し、全件一致した。tolerance・baseline は変更していない。
- `threshold` と固定 noise の `rrelu_with_noise` は forward／backward とも fixture に bit 一致した。`rrelu` 推論は傾きの `f32` 丸めで PyTorch と 1 ulp ずれうるため REQ-2 判定のみを主張する（§5）。
- CUDA／Metal 実機の `#[ignore]` テストは未実測（§10）。

## 1. 着手時の判定

事実のみを記す。

- 既存の活性化 5 種（#2146）は `activation_ops.rs` の自由関数と `nn/activation.rs` の層で、いずれも新規 `Op` なしの合成。値依存マスクは `activation_ops::build_value_mask`（本イシューで `pub(crate)` へ可視性のみ変更し再利用）。
- `ScalarUnaryOp::LeakyRelu` は `x >= 0` を恒等側とする（`crates/tensor-core/src/scalar_op.rs`）。PyTorch の RReLU 推論は `x > 0` が恒等側で、`x == 0` の勾配は傾きになる。
- 乱数は `fandhe_ai_tensor_core::rng`（Xorshift64*）で、PyTorch の乱数列とは一致しない。
- 同名の fn 宣言は `crates/autodiff/src/nn/activation.rs` の `Softplus::threshold`（`pub(crate)` アクセサ）1 件のみで、本イシューとは無関係。
- 承認記録: 調査時点で #2650 にコメントはなく、#2677 は open。公開面の拡張は未承認として扱う。

## 2. 実装方式

| 演算 | 合成 | バックエンド到達性 |
|---|---|---|
| `softmin(x, dim)` | `x.neg()?.softmax(dim)` | `Op::ScalarUnary(Neg)`／`Op::Softmax`。既存経路（既定 `Unsupported` はホスト参照実装へ） |
| `tanhshrink(x)` | `x.sub(&x.tanh())` | `Op::Sub`／`Op::Tanh` |
| `threshold(x, th, v)` | `x.masked_fill(&mask(x <= th), v)` | `Op::MaskedFill`（`masked_fill_with_fallback`） |
| `rrelu_with_noise(x, noise)` | `x.mul(&var_no_grad(noise))` | `Op::Mul` |
| `rrelu(x, lower, upper, training)` | noise を構築して `rrelu_with_noise` へ委譲 | `Op::Mul`（ホストでマスク・noise 構築） |

引数順は PyTorch（`F.softmin(input, dim)`・`F.threshold(input, threshold, value)`・`F.rrelu(input, lower, upper, training)`）に合わせる。

nn 層は `Softmin { dim }`・`Tanhshrink`（ユニット）・`Threshold { threshold, value }`・`RRelu { lower, upper, training }`。各 `forward` は自由関数への 1 行委譲で、`Module` 実装も同ファイルに置く（`nn/dropout.rs` と同型）。`RRelu` は `set_training`／`training` を両方オーバーライドし、`Default` は `lower = 1/8`・`upper = 1/3`・学習モード。`forward_host` は 4 層とも未提供（`supports_forward_host() == false`。tape 経由 forward との bit 一致を検証していない経路を主張しない。#2146 の `Mish`／`Glu` と同じ判断）。

### RReLU の noise 構築

- 学習時: `rng::rand(shape)` を全要素分 1 回引く（データに依存しない消費量。`dropout_mask` と同型）。`r = lower + (upper - lower) * u` を `f64` で計算して `f32` へ落とし `[lower, upper]` へ clamp する。`x <= 0` の位置は `r`、それ以外（NaN を含む）は `1.0`。
- 推論時: RNG を消費しない。傾きは `((lower as f64 + upper as f64) / 2.0) as f32`。`x > 0` の位置は `1.0`、それ以外は傾き。

### 却下した案

| 案 | 却下理由 |
|---|---|
| 推論を `Var::leaky_relu` へ委譲 | 既存 `LeakyRelu` は `x >= 0` を恒等側とするため、`x == 0` の勾配が 1 になり PyTorch（傾き）・学習時（`x <= 0` で `r`）と食い違う。定数 noise の単一路にすれば境界の扱いが揃い、コード経路も 1 本になる。代償は推論時にホスト読み出しとマスク構築が入ること（`hardtanh`／`prelu` と同じ） |
| `activation_ops` への追記 | `activation_ops` の fn 集合は facade の正ガードに固定済みで追記できない。並列イシュー（#2649）との衝突も避ける |
| `Var` メソッド化の先行 | 公開面の拡張は承認依頼 #2677 の範囲。本イシューでは行わない |

## 3. 数値契約

| 演算 | forward | backward | 判定 |
|---|---|---|---|
| `softmin` | `softmax` の超越関数を含む | 同左 | REQ-2 統一複合判定 |
| `tanhshrink` | `tanh` を含む。0 近傍は桁落ち（真値は約 `x^3 / 3`）するが絶対誤差側で収まる | 同左 | REQ-2 判定 |
| `threshold` | 選択のみ | 選択のみ（置換位置は 0） | bit 一致（fixture 実測で成立） |
| `rrelu_with_noise` | IEEE 乗算 1 回 | noise と bit 一致 | bit 一致（fixture 実測で成立） |
| `rrelu` 推論 | 同上 | 同上 | PyTorch 比は REQ-2 判定（§5）。同一入力のバックエンド間は bit 一致 |

tolerance 定数（`RELATIVE_TOLERANCE`／`ABSOLUTE_RESCUE_THRESHOLD`）・baseline は不変。

## 4. 境界検査（REQ-8・A03）

- `softmin` の `dim` は rank 以上なら `AutodiffError::Shape(AxisOutOfRange)`。tape 操作より前に検査する。
- `rrelu`／`RRelu::new` の `lower`・`upper` は有限かつ `lower <= upper`。違反は `AutodiffError::InvalidArgument`。学習時・推論時の両方で、tape 操作・RNG 消費より前に検査する。
- `rrelu_with_noise` は `noise.shape() == x.shape()` を要求し、違反は `AutodiffError::Shape(ShapeMismatch)`。
- `threshold` の `threshold`／`value` は検証せず IEEE のまま扱う（`leaky_relu` の `negative_slope` と同じ規約）。
- いずれもエラー時に tape へ孤児ノードを残さない（単体テストで `tape.len()` 不変を確認）。確保量は入力要素数に比例し、外部値で膨らむ経路はない。本番経路で `unwrap()`／`expect()` を使わない。新規 `unsafe` なし。

## 5. PyTorch 2.14.0 との差分と実測で確定した点

fixture は実 PyTorch 2.14.0+cpu の実行値（生成条件・sha256 は `crates/autodiff/tests/fixtures/softmin-threshold-pytorch-reference/README.md`）。

### 実測で確定した PyTorch の挙動（本実装は一致）

- `threshold`: NaN は素通し（出力 NaN・勾配 1）。`x == threshold` は `value` へ置換され勾配 0。述語は `v <= threshold`（`!(v > threshold)` ではない）。
- `rrelu` 推論: `x > 0` が恒等側で、`x == 0`・NaN・`-0.0`・`-inf` の勾配は傾き。
- `rrelu` 学習: `x == 0` の勾配は noise（傾き）、NaN の勾配は 1（noise = 1）。
- `softmin`: `+inf` を含む行は当該要素が 0、`-inf` を含む行は全要素 NaN。

### 既知の差分

| 項目 | PyTorch | 本実装 |
|---|---|---|
| RReLU 学習時の乱数列 | PyTorch の乱数・`x <= 0` の要素だけ引く | Xorshift64*・全要素分を 1 回引く。列は一致しない。fixture の noise を `rrelu_with_noise` へ渡して検証し、乱数経路は `manual_seed` と `rand` で別途固定 |
| `lower > upper`・非有限 | 学習時の `uniform_` でのみ失敗する想定 | 推論時にも型付きエラーで拒否 |
| `softmin` の `dim` | 負の `dim`・省略可 | `usize` のみ（本クレートの慣例） |
| 学習時 noise の範囲 | 丸めの扱いに依存 | `[lower, upper]` へ clamp（丸めで範囲外へ出ない保証） |
| `rrelu` 推論の傾き | `(lower + upper) / 2` を `f64` の `Scalar` から計算 | `f32` 引数から `f64` で計算して `f32` へ丸める。`lower = 0.05`・`upper = 0.6` で出力が 1 ulp ずれる実測があり、bit 一致は主張せず REQ-2 判定のみ |

## 6. テスト構成

| テスト | 内容 |
|---|---|
| `autodiff` 単体（`softmin_threshold_ops::tests`・`nn::softmin_threshold::tests`） | 手計算参照・中心差分・境界（`x == threshold`・NaN・rank 0）・検査エラーと tape 不変・RReLU の構造検証・層と自由関数の bit 一致・`supports_forward_host() == false`・`threshold` の `value` が NaN／±inf のケース（JSON の `params` は NaN／inf を持てず fixture へ入れられないため、PyTorch 2.14.0 の実測〈置換位置は `value` のまま・勾配 0、非置換位置は素通し〉に合わせた単体テストで代替） |
| `crates/autodiff/tests/softmin_threshold_parity.rs` | fixture 全件を REQ-2 判定、`threshold`／`rrelu_train` は bit 一致も固定、`edge_cases` はクラス一致、乱数経路（再現性・全要素分の消費・推論の非消費）、層経由 forward |
| `crates/facade/tests/softmin_threshold_ops_backend_parity.rs` | CPU 対 NaiveOps の 4 テスト（bit 一致 forward／backward、REQ-2 forward／backward。全演算を網羅）と CUDA／Metal の `#[ignore]` 8 件 |
| `crates/facade/tests/api_surface.rs` | 保留ガードの 5 テスト（§9） |

## 7. facade 公開形の推奨案（ルート #2499 の 2026-10-07 コメントで承認・#2679 で公開。§12 参照）

**本節は推奨案の記録であり、承認記録ではない。** 実際に得ていない承認はここにも、コミット・PR にも書かない。

- #2678: `Var` の 1 行委譲メソッド 4 本（`Var::softmin(dim)`・`Var::tanhshrink()`・`Var::threshold(threshold, value)`・`Var::rrelu(lower, upper, training)`）
- #2679: `compat::Sequential::add_softmin(dim)`・`add_tanhshrink()`・`add_threshold(threshold, value)`・`add_rrelu(lower, upper) -> Result`
- 推奨しない: `softmin_threshold_ops` モジュールの再エクスポート、層型の再エクスポート、`rrelu_with_noise` の公開、`Tensor<f32>`／`Tape` への配置
- 根拠: #2516・#2529 の前例と同形。追加のみで非破壊。`Var` に同名の既存メソッドが無いことは確認済み。
- 層の公開時の検討事項: `forward_host`（tape 不要経路）を足す場合は tape 経由 forward との bit 一致を検証してから `supports_forward_host` を `true` にする。

## 8. スコープ外

- facade 公開（`Var` 委譲メソッド・`compat::Sequential::add_*`・保留ガードの正ガードへの反転）: 承認依頼 #2677、承認後の #2678・#2679
- `compat::Sequential` への結線・`model_io` の kind 追加・ONNX import／export
- 4 層の `forward_host`
- CUDA／Metal の専用カーネルと実機での実測
- RReLU の乱数列の PyTorch 互換・Generator 指定・inplace 版・負の `dim`
- `create_graph`（高階微分）・activation checkpoint・f64／f16／bf16 経路での保証
- 既存 `LeakyRelu` の `x == 0` 勾配の PyTorch との差（既存挙動。本イシューでは変更しない）

## 9. 多層防御（保留ガード）

| 層 | 内容 |
|---|---|
| doctest プローブ | `crates/facade/src/lib.rs` の `SoftminThresholdOpsHoldDoctestGuard`。全 `pub mod` を glob import したスコープで同名モジュール・型・`Var`／`Tape`／`Tensor<f32>`／`Sequential` のメソッドを置き、facade が同名を公開すると名前解決の曖昧性（E0659）でコンパイルが失敗する |
| 固定文言 | `softmin_threshold_ops_hold_doctest_probe_body_matches_fixed_contract`（プローブ本文を 1 行単位で固定）・`softmin_threshold_ops_hold_doctest_globs_all_pub_modules` |
| ソース走査 | `facade_does_not_reexport_or_declare_softmin_threshold_ops`（`pub use` の経路・内部クレートの glob 再エクスポート・型の独自宣言・`pub mod`・同名 `fn`）と自己テスト `..._detects_each_category`（`ThresholdMode` の再エクスポートなど別トークンを負例に含む） |
| インベントリ | `workspace_declares_softmin_threshold_ops_fn_names_only_in_allowed_locations`（`crates/*/src/` の同名 `fn` を承認済みの置き場所に固定） |

検出範囲は列挙した名前・型に限り、マクロ生成や別名経由のメソッドまでは保証しない。

**手動の反証確認**: `crates/facade/src/lib.rs` に `pub use fandhe_ai_autodiff::softmin_threshold_ops;` を仮に足し、`facade_does_not_reexport_or_declare_softmin_threshold_ops` が失敗し、doctest が `E0659`（`softmin_threshold_ops` is ambiguous）で失敗することを確認して元に戻した。

## 10. 実機申し送り

CUDA（`Device::Cuda(0)`）・Metal（`Device::Metal`）の `#[ignore]` テスト 8 件は未実測。手順・期待結果・記入欄は `docs/perf/logs/softmin-threshold-ops-2650/README.md`。新規 GPU カーネルは無く、既存カーネルとホストフォールバック経路の確認である。

## 11. 出典

- `crates/autodiff/src/softmin_threshold_ops.rs`・`crates/autodiff/src/nn/softmin_threshold.rs`
- `crates/autodiff/tests/softmin_threshold_parity.rs`・`crates/autodiff/tests/fixtures/softmin-threshold-pytorch-reference/`
- `crates/facade/tests/softmin_threshold_ops_backend_parity.rs`・`crates/facade/tests/api_surface.rs`・`crates/facade/src/lib.rs`
- `docs/autodiff-activation-ops-decision.md`（#2146）・`docs/autodiff-trig-ops-decision.md`（#2634）・`docs/autodiff-packed-sequence-decision.md`（#2647）
- `docs/compat-api-scope.md`（適用記録）

## 12. #2679 実装記録（`Sequential::add_*` の facade 公開）


- 状態: **§7 のうち `compat::Sequential::add_*` 4 本を #2679 で公開した。** 承認根拠はルート #2499 の 2026-10-07 ユーザー承認コメント（issuecomment-6033824965。「Phase 4（#2625）」節で `docs/compat-api-scope.md` §5.1 の行 18 を各決定記録の推奨形で承認）。本書中の「未承認」「承認依頼は #2677」の記述は、#2679 時点で当該コメントの承認に更新された（#2677 の「承認の記録」コメントの割り振りでは公開は #2679）。承認は推奨形に限り、記録に形が書かれていない点は実装せず承認依頼へ戻す条件つき。
- 公開した識別子（`crates/facade/src/compat/sequential.rs`）: `add_softmin(self, dim: usize) -> Self`（`dim` の範囲検査は forward 時）・`add_tanhshrink(self) -> Self`・
  `add_threshold(self, threshold: f32, value: f32) -> Self`・`add_rrelu(self, lower: f32, upper: f32) -> Result<Self, AutodiffError>`（`lower`／`upper` は構築時に検証）。
  `rrelu_with_noise`・層型・モジュールは非公開のまま。`Var` の委譲メソッドは #2678 の担当。
- `RRelu` の追加時点の `training` は `true`（`Dropout` と同じ）で、`Sequential::set_training`／`eval` の伝播で推論時の固定傾き `(lower + upper) / 2` へ切り替わる。
- 記録に形が書かれていない点の扱い: manifest kind（`activation-scalar-ops` 記録 §12 と同じ理由で追加せず `UnsupportedModel` で拒否）。**`forward_host` は 4 層とも未提供のまま**
  （§7「bit 一致を検証してから」）で、`Sequential::predict` は既存の tape 経路フォールバックで動く。GPU 専用カーネルは追加していない。
- ガード（§9）の反転: `SoftminThresholdOpsHoldDoctestGuard` から `add_*` のプローブを削除し、`Var`／`Tape`／`Tensor<f32>` のメソッド・層型・モジュールのプローブだけを残した。
  `add_*` の `fn` 宣言は `compat/sequential.rs` に承認シグネチャで各 1 件という正ガードへ反転した（workspace 宣言インベントリの期待値に登録）。
- テスト・実機申し送りは `autodiff-activation-scalar-ops-decision.md` §12 と共通（9 層を 1 つのテストファイルで検証）。
- 依存・tolerance・baseline・ガードレール閾値・`docs/spec` は変更していない。`fandhe-ai =0.10.0` の既存公開 API・`pub use` 行・署名は変更せず、追加のみ。
