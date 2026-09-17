# Metal argmax／argmin 実機ランブック（イシュー #1951）

argmax／argmin の Metal カーネル（`crate::reduce::MetalReduce::run_arg_all_f32`／
`run_arg_axis_f32`。`crate::ops::MetalBackendOps::argmax`／`argmin` 経由）の
実機（Apple Silicon）実測ランブック。本実装エージェントの実行環境
（Linux）には Apple Silicon 実機への到達手段がないため、実測は未実施の
まま Mac セッションへ申し送る（`docs/backend-metal-reduce-sum-design.md`
§11.4・`metal-reduce-1895/README.md` と同型）。

## 実行コマンド

```sh
# Linux（本実装セッション）で完了済みの型検査・非実機テスト:
cargo test -p fandhe-ai-backend-metal --lib reduce_model
cargo test -p fandhe-ai-backend-metal --test reduce_source_evidence
cargo test -p fandhe-ai-backend-metal --test backend_ops_real_device \
  argmax_argmin_shape_errors_without_device_init
cargo check -p fandhe-ai-backend-metal --tests --target aarch64-apple-darwin
cargo check -p fandhe-ai --tests --target aarch64-apple-darwin

# Apple Silicon 実機（Mac セッション。--release 推奨）:
cargo test -p fandhe-ai-backend-metal --release --test reduce_parity -- \
  --ignored --nocapture metal_arg
cargo test -p fandhe-ai-backend-metal --release --test backend_ops_real_device -- \
  --ignored --nocapture backend_ops_argmax_argmin_match_cpu_exact
cargo test -p fandhe-ai --release --test reduce_backend_parity -- \
  --ignored --nocapture metal_argmax_and_argmin_match_cpu_exact

# 既存 #[ignore] 群の非後退確認（新規カーネル追加による回帰がないこと）。
cargo test -p fandhe-ai-backend-metal --release -- --ignored --nocapture
```

## 保存すべきログ

- 上記各コマンドの標準出力（pass/fail・所要時間）
- `docs/real-hardware-verification-env.local.md.example` に準じた
  `env_info.txt`（macOS バージョン・チップ世代。内部ホスト名は含めない）

## 事前登録判定規則

- 上記 argext 関連テスト（`reduce_parity.rs::metal_arg_*`・
  `backend_ops_real_device.rs::backend_ops_argmax_argmin_match_cpu_exact`・
  `reduce_backend_parity.rs::metal_argmax_and_argmin_match_cpu_exact`）が
  すべて pass で合格（添字は CPU 参照実装〈`fandhe_ai_backend_cpu::
  reduction::{argmax, argmin}`〉と完全一致する契約。REQ-2 複合判定
  ではなく厳密一致）。
- 既存 `#[ignore]` テスト群（`sum`／`gemm`／`elementwise` 等）が新規
  カーネル追加により非後退であること（新規 FAIL がないこと）。
- FAIL は是正せず記録のみとする（事後緩和なし）。
