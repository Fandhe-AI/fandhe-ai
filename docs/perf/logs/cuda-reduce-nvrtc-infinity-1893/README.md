# イシュー #1893 実測記録先（reduce カーネルの NVRTC `INFINITY` 未定義是正）

## 位置づけ（本 PR 時点では未実測）

本 PR の実行環境（Linux コンテナ／worktree）には DGX Spark GB10 実機
への到達手段（`docs/real-hardware-verification-env.local.md`・
`CUDA_NODE` 環境変数・`~/.ssh/config` のいずれも確認できず）がないため、
**実測値は一切含まれていない**。本ディレクトリは実行コマンド・保存
すべきログ一覧のみを提供し、実測は GB10 実機を持つセッションへ申し送る
（`docs/perf/logs/cuda-conv2d-1766/README.md` と同型の運用）。

## 目的

`crates/backend-cuda/src/kernels_reduce.rs` の max／min 系 8 カーネル
（単位元として C マクロ `INFINITY`／`-INFINITY` を使用）が NVRTC で
コンパイルエラーになり（NVRTC は `<math.h>` を暗黙に含めないため。
`docs/perf/logs/cuda-realdevice-phase2-2026-09-16/README.md` §3.1）、
`CudaReduce::new` 全体が `Err` となって sum／max／min の 12 カーネル
すべてが `CudaUnavailable` を返していた不具合を、単位元を bit パターン
直接構成（`__uint_as_float(0xff800000u)`／`__uint_as_float(0x7f800000u)`）
へ置換することで是正した。本ディレクトリは是正後の GB10 実機再実測の
記録先。

## 実行コマンド

```bash
# 1) reduce_parity（6 件）
cargo test -p fandhe-ai-backend-cuda --release --all-features \
  --test reduce_parity -- --ignored --nocapture \
  2>&1 | tee reduce_parity-ignored.log

# 2) typed f16／bf16 reduction parity
cargo test -p fandhe-ai-backend-cuda --release --all-features \
  --test typed_ops_f16_parity -- --ignored --nocapture \
  2>&1 | tee typed_ops_f16_parity-ignored.log
cargo test -p fandhe-ai-backend-cuda --release --all-features \
  --test typed_ops_bf16_parity -- --ignored --nocapture \
  2>&1 | tee typed_ops_bf16_parity-ignored.log

# 3) facade var_norm_backend_parity（cuda_ プレフィックスのみ）
cargo test -p fandhe-ai --release --test var_norm_backend_parity \
  -- --ignored --nocapture cuda_ \
  2>&1 | tee var_norm_backend_parity-ignored.log

# 4) Var::sum(None) を loss とする tape backward（7 件）
for t in conv1d_backend_parity conv2d_backend_parity nn_conv_backend_parity \
         attention_backend_parity no_grad_detach_backend_parity \
         backward_accumulate_backend_parity; do
  cargo test -p fandhe-ai --release --test "$t" -- --ignored --nocapture cuda_ \
    2>&1 | tee "${t}-cuda-ignored.log"
done

# 5) 全体非後退確認
make test-ignored-cuda 2>&1 | tee make-test-ignored-cuda.log
```

## 保存すべきログ

- `reduce_parity-ignored.log`
- `typed_ops_f16_parity-ignored.log`
- `typed_ops_bf16_parity-ignored.log`
- `var_norm_backend_parity-ignored.log`
- `conv1d_backend_parity-cuda-ignored.log`・`conv2d_backend_parity-cuda-ignored.log`・
  `nn_conv_backend_parity-cuda-ignored.log`・`attention_backend_parity-cuda-ignored.log`・
  `no_grad_detach_backend_parity-cuda-ignored.log`・`backward_accumulate_backend_parity-cuda-ignored.log`
- `make-test-ignored-cuda.log`
- `env_info.txt`（下記フォーマット。内部ホスト名は書かない）

## 事前登録判定規則

- 上記 (1)〜(4) の対象テスト 16 件すべてが pass すること。
- `make test-ignored-cuda` 相当の全体実行で、FAIL が既知 10 件
  （`docs/perf/logs/cuda-realdevice-phase2-2026-09-16/README.md` §3.2
  の既知分類：f16 Tensor Core K=4096 ストレス 3・TF32／3×TF32 厳密
  ゼロ fail 不成立 5・staged 対 opt 性能比較 1・
  `module_cache_wiring_tests` 並列干渉 1）を超えないこと。
- tolerance 定数（`RELATIVE_TOLERANCE`／`ABSOLUTE_RESCUE_THRESHOLD`）・
  `parity_baseline.rs` の baseline は一切変更しないこと（AC-4）。

## env_info.txt 記入欄

```
hostname: masked
gpu: <型番>
driver: <driver バージョン>
cuda: <CUDA バージョン>
rustc: <rustc --version>
date: <実測日 YYYY-MM-DD>
```
