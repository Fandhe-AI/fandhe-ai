# Metal train forward の encode-only 合流 A/B 記録（イシュー #2113）

**状態: 未実測**（実装・テスト・A/B 基盤のみ。M4 Max・GB10 とも実機セッションへ申し送り）。

## 位置づけ

- #1980 §17.4 の施策 2。train reuse の forward（`DeviceParamStore::linear_forward_with_activation` → `BackendOps::gemm_resident_rhs_act`）は Metal では既定合成（`gemm_resident_rhs` → `relu`）が 2 回同期する。opt-in（既定 OFF）の Metal オーバーライドで gemm と relu を同一コマンドバッファへ encode-only で積み、同期を 1 回へ合流する（カーネル・入力は不変で bit 同一）。
- 設計・カウンタ見積りは `docs/backend-metal-command-batching-design.md` §7.6。
- 事前登録判定規則は `RULE.txt`（実測前に固定）。#1691（MSE）・#1563（backward d_input）・#1044（epilogue 融合）とは同期境界が異なり、再実行にあたらない。

## 手順（実機セッション向け）

1. 前提ゲート: `cargo test -p fandhe-ai-backend-metal --release --test train_forward_encode_only_parity -- --ignored --nocapture`、`cargo test -p fandhe-ai --release --test metal_train_forward_encode_only_bit_identity -- --ignored`、`cargo test -p fandhe-ai --release --test mnist_scale_train_reuse_bench -- --ignored --nocapture`（ON 11/6/6・OFF 11/7/7・backward_dinput_phase 5/3/3）。
2. A/B: before（main）と after（同一コミットで `TRAIN_FORWARD_ENCODE_ONLY_DEFAULT_ENABLED` のみ `true`。未コミット）の 2 worktree を用意し、`AB_BEFORE_FACADE_PATH`／`AB_AFTER_FACADE_PATH`（絶対パス）を指定して `scripts/bench/framework-compare/run_ab_train_forward_encode_metal.sh <label>` を実行する（M4 Max: `AB_DEVICE=metal`、GB10: `AB_DEVICE=cpu` と `cuda` を対照として）。
3. 生ログ・`env_info.txt`（`env_info.txt.example` を参照。ホスト名等はマスク）を `m4max/`・`gb10/` へ収録する。

## 収録物

- `RULE.txt`: 事前登録判定規則
- `env_info.txt.example`: 環境情報の記入欄
- `m4max/`・`gb10/`: 実測ログの収録先（未収録）
