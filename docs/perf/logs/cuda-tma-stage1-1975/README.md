# TMA ロード経路 Stage 1（イシュー #1975）実機実測ランブック

本ディレクトリは、TMA（`cp.async.bulk.tensor`）ロード経路 Stage 1
（64×64 pipeline・`shared::cta`。`crates/backend-cuda/src/kernels_tiled_
pipeline.rs`「TMA」節・`crates/backend-cuda/src/gemm.rs::
tma_tiled_pipeline` モジュール）の GB10 実機検証（イシュー #1976）を記録
する。実測は 2026-09-18 に完了済み。

設計の正は `docs/backend-cuda-tma-gemm-load-design.md` §6（事前登録
ゲート）・§10（実装記録。#1975 で追記）を参照する。本 README は本節と
重複させず、実行コマンド・保存すべきログ一覧のみを記す。

## 前提

- `internal-diagnostics` cargo feature を要求する
  （`crates/backend-cuda/Cargo.toml` の `[[test]]
  cpu_cuda_tiled_pipeline_tma_parity` 参照）。
- compute capability 9.0 以降（TMA 命令の前提。`CudaGemm::
  compile_tiled_pipeline_tma_variant` が構築時に検査し、未満の環境は
  `CudaError::TiledPipelineUnavailable` で fail-closed 拒否する）。

## 実行コマンド

```sh
# (a) None / B64 両腕の bit 一致・決定性・CPU parity・事前転置入力・
#     k=0 no-op（AC 本体。`tiled_pipeline_tma_b64_*` は不一致を記録
#     するのみで CI 失敗にはしない設計。テストファイル冒頭コメント参照）
cargo test -p fandhe-ai-backend-cuda --test cpu_cuda_tiled_pipeline_tma_parity \
  --all-features -- --ignored --nocapture

# (b) 意味論プローブ（cluster/cta variant のコンパイル・実行 probe。
#     Stage 1 本体の座標系・部分 OOB box・B64 swizzle 物理配置の直接
#     観測は本 issue 未実装。tests/tma_probe_real_device.rs 冒頭コメント
#     「位置づけ」節・design doc §8 参照）
cargo test -p fandhe-ai-backend-cuda --test tma_probe_real_device \
  --all-features -- --ignored --nocapture
```

## 保存すべきログ一覧

- `ignored_tests.log`: 上記 (a) の全出力（`tiled_pipeline_tma_b64_
  matches_pipeline_or_records_hypothesis_gap` の `mismatched_shapes`
  出力を含む。B64 仮説の成立・不成立を判定するための一次情報）。
- `probe.log`: 上記 (b) の全出力。
- `env_info.txt`: `nvidia-smi`／`nvcc --version`（存在すれば）／
  compute capability（`device.compute_capability()` を出力するアドホック
  スクリプトでよい）。内部ホスト名・ユーザー名は含めない
  （`.claude/rules/security.md`）。

## 判定規則（設計 doc §6 の転記。事前登録）

- **正しさ（必須）**: `tiled_pipeline_tma_none_matches_pipeline_bit_exact`
  が全形状で `assert_eq!` bit 一致・`assert_parity` 通過すること。1 件
  でも FAIL した場合、本番非結線（`select_tiled_f32_kernel`／
  `CudaGemm::new` 不変）は維持したまま、原因（座標系／プロローグの
  タイル起点ガード／部分 OOB box の fill 意味論／`fence.proxy.async`
  の配置）を記録し是正 PR へ引き継ぐ（本 README では是正しない）。
- **B64 swizzle 仮説（記録のみ）**: `tiled_pipeline_tma_b64_matches_
  pipeline_or_records_hypothesis_gap` の `mismatched_shapes` が空なら
  仮説成立、非空なら不成立として記録する（CI 失敗にしない）。不成立の
  場合、`ignored_tests.log` の出力（不一致形状一覧）を
  `docs/backend-cuda-tma-gemm-load-design.md` §10 へ転記する。
- **性能（#1976 実測済み）**: ゲート C（純カーネル時間・N=256〜4096 の
  5 回計測中央値）は `aggregate.md` に記録済み。ベンチ列（
  `tma_none_gpu_only_tflops`／`tma_b64_gpu_only_tflops`）を
  `crates/backend-cuda/examples/gemm_tiled_pipeline_bench.rs` へ追加
  実装済み。ゲート D（本番ディスパッチ非後退）は結線対象外のため
  省略。

## 未実施事項（#1975 のスコープ縮小。詳細は設計 doc §10）

- 意味論プローブ 3 件（要素座標の非ゼロ確認・部分 OOB box の fill 意味論・
  `B64` swizzle の smem 物理配置ダンプ）は未実装。
  `kernels_tiled_pipeline::tma_swizzled_chunk_a`
  （現時点 `#[cfg(test)]` 限定）を `pub` へ戻し `lib.rs` から
  re-export する作業を含む。
- `examples/gemm_tiled_pipeline_persistent_bench.rs` への `--tma
  off|none|b64` 列追加（純カーネル時間比較）は未実施。
- tensor map のキャッシュ（起動ごとの再生成を避ける最適化）は未実装
  （設計 doc §4.4・毎起動 encode の契約は `launch_tiled_pipeline_
  tma_f32` ドキュメンテーションコメント参照）。
