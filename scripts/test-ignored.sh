#!/usr/bin/env bash
# 実機（CUDA / Metal）専用の #[ignore] 分離テスト導線（make test-ignored-cuda /
# test-ignored-metal-facade）。CI（Linux ホステッド・実機非搭載）では実行できない。
# usage: test-ignored.sh <cuda|metal-facade>
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./lib.sh
source "${DIR}/lib.sh"

case "${1:-}" in
  cuda)
    # TASK-1.7e（イシュー #36）: backend-cuda 限定。--release 既定（M=N=K=4096 の実測や
    # CPU 参照実装の計算量が大きく debug では著しく遅い）。
    # --all-features 必須（PR #685 是正）: specialized_mma_parity は required-features =
    # ["internal-diagnostics"] でゲートされ、feature 未指定だとテストバイナリ自体が
    # ビルドされず --ignored を渡しても暗黙に「パス」扱いになる（false-green）。
    #
    # graph_capture_real_device.rs の ON 側 2 テストは 2 回目の別呼び出しに分離する
    # （codex-review P2・PR #1390）: internal-diagnostics 有効の間は CudaDevice::new が
    # StreamKind を常に Legacy に固定するため、capturable stream 前提の 2 テストは
    # 1 回目（--all-features）では実機上で確実に失敗する。--skip で 1 回目から除外し、
    # internal-diagnostics を持たない 2 回目で実行する（--test-threads=1 を明示）。
    #
    # #3594（学習 step 全体 capture の既定 ON 反転）: 同じ理由で train_step_graph_diag_tests_3260 も
    # 1 回目（Legacy 固定）では stats の assert で失敗するため --skip し、既定 feature・環境変数なし
    # （既定 ON）の 3 回目で単独・単スレッド実行する（既定 ON の本番経路を通るのはこの呼び出しだけ）。
    cargo test -p fandhe-ai-backend-cuda --release --all-features -- --ignored --nocapture \
      --skip sgd_update_segment_captures_then_replays_bit_identically \
      --skip different_config_key_produces_a_different_segment_key \
      --skip train_step_graph_diag_tests_3260
    cargo test -p fandhe-ai-backend-cuda --release --test graph_capture_real_device -- \
      --ignored --nocapture --test-threads=1
    cargo test -p fandhe-ai-backend-cuda --release --lib train_step_graph_diag_tests_3260 -- \
      --ignored --nocapture --test-threads=1
    ;;
  metal-facade)
    # イシュー #1898: test-ignored-metal は backend-metal 単体限定で、facade の
    # DeviceParamStore 契約テストは対象外。`-p fandhe-ai --tests -- --ignored` は同一
    # バイナリに CUDA 実機必須の cuda_* が混在し Mac 上で必ず FAIL するため、
    # テストバイナリを名前で絞って実行する。
    cargo test -p fandhe-ai --release --test device_param_store_backend_parity -- --ignored --nocapture --test-threads=1 on_metal
    cargo test -p fandhe-ai --release --test device_param_store_metal_mixed_shape_grad -- --ignored --nocapture --test-threads=1
    ;;
  *)
    echo "usage: $(basename "$0") <cuda|metal-facade>" >&2
    exit 2
    ;;
esac
