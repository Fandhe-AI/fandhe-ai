# Rust コーディング規約

## 基盤方針（REQ-1 v2、変更禁止）

- **完全自作コア**（テンソル・autodiff・演算グラフ／カーネル融合機構・計算カーネル・バックエンド抽象層）とする。Burn 等の既存 ML フレームワークへの統合は行わない
- 依存は許容依存 8 区分のみ（詳細は [deps-policy.md](./deps-policy.md)）。禁止リスト（`burn` 系一式・`cubecl`・`candle`・`tch`・`ndarray`）は CI で機械検査する（TASK-1.2）
- 互換 API 層（`compat::array`／`compat::Sequential` 相当）は自作コアの上の薄いラッパーに徹する（REQ-9）

## バックエンド構成（REQ-2）

- バックエンド切替は **feature フラグなしの cfg ベース**を基本とする（PoC-v2-5 実証構成）。`cudarc` は無条件依存＋動的ロード（CUDA toolkit 非搭載環境でもビルド成立）、`objc2`・`objc2-foundation`・`objc2-metal` は `cfg(target_os = "macos")` 分離
- バックエンド間数値一致は統一複合判定「**相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満**」（全ペア共通。REQ-2 は TF32 前提の複合指標に改定済み）
- 丸め方針（FMA 契約）をバックエンド間で統一する: CPU 参照実装は `f32::mul_add` を用い、GPU 側（CUDA NVRTC・Metal `simdgroup_multiply_accumulate`）の既定 FMA 契約と揃える（PoC-v2-5 の K=4096 ストレスケースで実測確認済み）。matmul 系の FMA 契約はこの方針のまま不変とする
- **正規化統計・勾配の長軸縮約（rmsnorm の `rstd` 二乗和・dw の行方向蓄積等）は `f64` アキュムレータで統一する**。この一般原則は 2 系統で扱いが異なる（精密化。イシュー #1102・codex-review 指摘・PR #1120。`docs/perf/cuda-parity-baseline.md` §9.10）:
  - **正規化統計の二乗和**（rmsnorm の `rstd` 導出）は要素を**先に `f64` へ昇格してから二乗**する（`f32` のまま二乗すると有限入力〈例 `2e20f`〉でも overflow しうるため。CUDA は `fma((double)v, (double)v, acc)`）
  - **勾配の長軸縮約の要素積**（dw の行方向蓄積等）は overflow リスクが実用上小さいため、要素積を **`f32` で確定してから** `f64` へ昇格して蓄積する（CUDA は `float term = dyv * r * xv; acc = (double)term + acc;`）
  最終書き出しはいずれも 1 回だけ `f32` へ downcast する。Metal（MSL）は `double` 型非対応のため、Neumaier 改良版 Kahan 補償和 + **scale/ssq 方式**（LAPACK SLASSQ 系の overflow-safe な二乗和アルゴリズム。単純な Kahan 補償和のみでは要素の二乗を `f32` のまま先に計算するため、有限入力でも overflow して `NaN` を生む。scale/ssq 方式は最大絶対値を `scale` として括り出し残りを比の二乗で蓄積するため二乗を直接計算せず overflow を避ける。`NaN`／`inf` 入力の伝播も明示的に扱う）を正規化統計の「`f64` 相当」実装形として適用する。CUDA・CPU（NEON は倍精度 SIMD `float64x2_t`）は `double`／`f64` アキュムレータを直接使う。この契約は matmul 系の FMA 契約とは独立の軸であり、既存の丸め方針を変更するものではない（ユーザー承認 2026-09-01。実測記録は `docs/perf/cuda-parity-baseline.md` §9.8〜§9.10）

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
