# Rust コーディング規約

## 基盤方針（REQ-1 v2、変更禁止）

- **完全自作コア**（テンソル・autodiff・演算グラフ／カーネル融合機構・計算カーネル・バックエンド抽象層）とする。Burn 等の既存 ML フレームワークへの統合は行わない
- 依存は許容依存 8 区分のみ（詳細は [deps-policy.md](./deps-policy.md)）。禁止リスト（`burn` 系一式・`cubecl`・`candle`・`tch`・`ndarray`）は CI で機械検査する（TASK-1.2）
- 互換 API 層（`compat::array`／`compat::Sequential` 相当）は自作コアの上の薄いラッパーに徹する（REQ-9）

## バックエンド構成（REQ-2）

- バックエンド切替は **feature フラグなしの cfg ベース**を基本とする（PoC-v2-5 実証構成）。`cudarc` は無条件依存＋動的ロード（CUDA toolkit 非搭載環境でもビルド成立）、`objc2`・`objc2-foundation`・`objc2-metal` は `cfg(target_os = "macos")` 分離
- バックエンド間数値一致は統一複合判定「**相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満**」（全ペア共通。REQ-2 は TF32 前提の複合指標に改定済み）
- 丸め方針（FMA 契約）をバックエンド間で統一する: CPU 参照実装は `f32::mul_add` を用い、GPU 側（CUDA NVRTC・Metal `simdgroup_multiply_accumulate`）の既定 FMA 契約と揃える（PoC-v2-5 の K=4096 ストレスケースで実測確認済み）。matmul 系の FMA 契約はこの方針のまま不変とする。**例外**: CUDA GEMM の 3×TF32 opt-in モード（`fandhe_ai_backend_cuda::precision::CudaGemmPrecision::Tf32x3`。既定 OFF・facade `fandhe_ai::set_cuda_gemm_precision` 経由の明示 opt-in 限定）は、hi/lo 分割・3 回の `mma.sync` 累積（split-single 法）という構造上 f32 SIMT 参照実装と bit 一致しない。数値一致の複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）自体は変更しない（ユーザー承認 2026-09-06・イシュー #1338 コメント。詳細は `docs/cuda-tf32x3-split-single-decision.md`）
- **正規化統計・勾配の長軸縮約（rmsnorm の `rstd` 二乗和・dw の行方向蓄積等）は `f64` アキュムレータで統一する**。この一般原則は 2 系統で扱いが異なる（精密化。イシュー #1102・codex-review 指摘・PR #1120。`docs/perf/cuda-parity-baseline.md` §9.10）:
  - **正規化統計の二乗和**（rmsnorm の `rstd` 導出）は要素を**先に `f64` へ昇格してから二乗**する（`f32` のまま二乗すると有限入力〈例 `2e20f`〉でも overflow しうるため。CUDA は `fma((double)v, (double)v, acc)`）
  - **勾配の長軸縮約の要素積**（dw の行方向蓄積等）は overflow リスクが実用上小さいため、要素積を **`f32` で確定してから** `f64` へ昇格して蓄積する（CUDA は `float term = dyv * r * xv; acc = (double)term + acc;`）
  最終書き出しはいずれも 1 回だけ `f32` へ downcast する。Metal（MSL）は `double` 型非対応のため、Neumaier 改良版 Kahan 補償和 + **scale/ssq 方式**（LAPACK SLASSQ 系の overflow-safe な二乗和アルゴリズム。単純な Kahan 補償和のみでは要素の二乗を `f32` のまま先に計算するため、有限入力でも overflow して `NaN` を生む。scale/ssq 方式は最大絶対値を `scale` として括り出し残りを比の二乗で蓄積するため二乗を直接計算せず overflow を避ける。`NaN`／`inf` 入力の伝播も明示的に扱う）を正規化統計の「`f64` 相当」実装形として適用する。CUDA・CPU（NEON は倍精度 SIMD `float64x2_t`）は `double`／`f64` アキュムレータを直接使う。この契約は matmul 系の FMA 契約とは独立の軸であり、既存の丸め方針を変更するものではない（ユーザー承認 2026-09-01。実測記録は `docs/perf/cuda-parity-baseline.md` §9.8〜§9.10）
  さらに、**勾配の長軸縮約（dw の行方向蓄積・bias 勾配の行方向縮約等）の Metal 実装形**は、正規化統計とは別に規定する。MSL は `double` 型非対応のため二乗和と同じ scale/ssq 方式は転用できず（勾配縮約は二乗和ではなく符号付き値の和であるため）、代わりに **Neumaier 改良版 Kahan 補償和 + 2 の冪 scale**（`bias_pow2_floor`。列内の要素の最大絶対値以下の最大の 2 の冪で括り出し、除算・比が exact なので誤差項を追加しない）を勾配縮約の「`f64` 相当」実装形として適用する。この実装形は f32 のみで構成されるため、ホスト（CPU）`f64` 蓄積・CUDA `double` 蓄積と**厳密同値にはならない**。一致判定は事前に入力から計算できる述語による 2 層契約（正本 `docs/metal-grad-reduction-parity-judgment-decision.md`）で規定する。**Tier A**（全入力に常に適用・除外なし）: `|y_metal − y_ref| ≤ (3 + n·ε32)·ε32·Σ|x_i|`（ε32 = 2^-24。`n` は縮約対象要素数。有効範囲 `n·ε32 < 1`。係数 `3` は縮約段数 1〈完全逐次 Neumaier 補償和〉の誤差上界 `2u` + ホスト downcast 1 回分 `1u`〈安全側切り上げ〉から導出し、PR #1659 の逐語ホストモデル実測と整合する。`y_ref` は参照実装〈`reduce_bias_grad_rows_host`／`eval::reduce_bias_grad_rows`〉が index 順に `f64` で逐次加算した和 `S_ref` を 1 回 `f32` へ downcast した値で、無限精度の真値ではない）。**Tier B**（REQ-2 統一複合判定）: 事前判定可能な述語 `(3 + n·ε32)·ε32·Σ|x_i| ≤ max(1e-3·|S_ref|, 1e-5)` が成立する列にのみ適用し、不成立の列（`[2^48, 2^24, 1, -2^48, -2^24]` 等）は Tier A のみで判定する（除外ではなく判定方式の事前分岐。片側変更ではなく `double` 非対応バックエンドに対する実装形の規定である。ユーザー承認 2026-09-12・イシュー #1566）。契約の有効範囲は `n < 2^24` とし、範囲外の入力は Metal 実装・ホスト参照実装のいずれも `BackendError::InvalidArgument` で fail-closed に拒否する（無言のフォールバックはしない。実装は PR #1659 側）。非有限値（NaN／±inf）を含む場合は実数の上界式ではなくクラス一致（`y_ref` が NaN／±inf なら `y_metal` も同一クラス）を要求し、`y_ref` と `y_metal` のクラスが食い違う場合は実装のバグとする（除外なし。詳細は正本ドキュメント「2.4」「2.5」節）。

## コード品質

- `cargo fmt --all`・`cargo clippy --workspace --all-targets --all-features -- -D warnings` を通す（`#[allow]` の安易な追加で黙らせない）
- `unsafe` は FFI 境界（cudarc・objc2 系）等の必要最小限に留め、理由をコメントで明記しレビュー必須とする
- エラーは型付きエラーとし、本番経路で `unwrap()` / `expect()` を使わない
- 依存クレートの追加・更新はライセンス確認（`docs/license-matrix.md` 更新）とセットで行い、ユーザー承認必須（deps-policy.md）

## カーネル実装の境界検査（REQ-8）

- **性能下限・最適化の達成を理由に、シェーダ・カーネル側の手動境界チェックを省略しない**
- 境界検査を無効化する最適化（ベクトル化ロード・タイル端の分岐削減等）を適用する場合は、シェーダ側で手動境界チェックを維持したうえで行う
- 本規約は CPU（intrinsics）・CUDA（NVRTC/mma）・Metal（simdgroup）の全カーネルに適用する

## テスト・ベンチ

- 受け入れ基準（`docs/spec/04-requirements.md`）に対応するテストを同一 PR に含める
- 実機（DGX Spark GB10・Metal 実機）依存テストは `#[ignore]` で分離し、CI（GitHub ホステッド。[`ci.md`](./ci.md)）で実行可能なテストと区別する
- バックエンド間数値一致テストの許容誤差（tolerance）を単独で緩和しない（ポリシー除外リストのブラインドスポット対象）
- **TF32/f16 Tensor Core 経路の parity テスト判定方式（spec REQ-2 2026-09-02 追記の形状別判定方式）**: 受け入れ基準の正は `docs/spec/04-requirements.md` REQ-2「2026-09-02 追記・Tensor Core 経路の受け入れ判定方式」（fandhe-ai-spec PR #63）。厳密ゼロ fail 判定（`fandhe_ai_backend_cpu::assert_parity`）は実機実測で成立が確認された形状に限り、成立しない形状は実測 baseline 非後退方式（`crates/backend-cuda/tests/common/parity_baseline.rs::ParityBaseline`。GB10 実機実測値を伴う `fail_count`・総要素数一致・`mean_abs_diff`/`max_abs_diff`/`max_rel_err` ceiling の fail-closed 非後退検査）を正式な受け入れ判定とする。baseline の追加・更新は実機実測値のみ・人間承認必須。tolerance 定数（`RELATIVE_TOLERANCE`/`ABSOLUTE_RESCUE_THRESHOLD`）自体の変更は本規約の対象外で、引き続きユーザー承認必須（経緯・承認記録は `docs/cuda-tensor-core-parity-judgment-decision.md`・`docs/perf/cuda-parity-baseline.md`）
- ベンチは 5 回計測の中央値を採用し、学習系回帰テストには決定的シード設定ユーティリティを使う

## 関連ルール

- 依存管理は [deps-policy.md](./deps-policy.md)
- コメントは [code-comment-style.md](./code-comment-style.md)
- セキュリティは [security.md](./security.md)
- CI は [ci.md](./ci.md)
