# CUDA GEMM tiled pipeline（FP32 SIMT + cp.async 多段パイプライン）計測記録（#1033）

イシュー #1033「perf(backend-cuda): cp.async 多段パイプライン（3〜4 stage）を FP32 SIMT 経路に導入する」の実測記録テンプレート。
親イシュー #1031（FP32 SIMT GEMM 強化）・ルート #1029「GEMM カーネルの candle 超え」Phase 2 の一環。
受け入れ条件「N=4096 での改善値の記録（5 回計測中央値）」「カーネル側の手動境界検査の維持（REQ-8）」に対応する。

## 状態: DGX Spark GB10 実機実測完了・本番結線済み（イシュー #1137。下記「#1137 本番結線判断」節）。以下は #1033 実装セッション時点の記録として残す

本実装セッションの実行環境には CUDA **driver**（`libcuda`）が実在し、`nvidia-smi` で以下の実機が確認できる:

```
NVIDIA GeForce RTX 3060（compute capability 8.6・Driver Version 595.71.05・CUDA Version 13.2 表記）
```

これは `cp.async`（`cp.async.cg.shared.global`/`.commit_group`/`.wait_group`）が要求する compute capability 8.0 以上（Ampere 以降）
を満たすため、`CudaDevice::new(0)` は成功する。しかし **NVRTC（`libnvrtc`）はこの環境に存在せず**、`compile_ptx` は
`CudaError::NvrtcUnavailable` を返す（`nvcc`／CUDA toolkit 自体が未導入）。

したがって、本ファイルが指すカーネルソース（`crates/backend-cuda/src/kernels_tiled_pipeline.rs::TILED_PIPELINE_F32_BODY`）は
**この実装セッション中に一度も NVRTC の構文検証を通過していない**。`docs/perf/cuda-gemm-mma-pipeline.md`・
`docs/perf/metal-gemm-dynamic-tile.md` の先例（「実機での最初の実行が構文検証を兼ねる」）と同じ位置づけであり、
**RTX 3060（sm_86）を含むいかなる実機でも未検証**である点に注意。sm_121（DGX Spark GB10。設計上のターゲット）での挙動は
さらに別途の検証が必要。

本実装セッションで検証済みの事項:

- `cargo build -p fandhe-ai-backend-cuda`／`cargo test -p fandhe-ai-backend-cuda`／
  `cargo clippy -p fandhe-ai-backend-cuda --all-targets --all-features -- -D warnings` が全て green（Rust 側の型・API
  契約・カーネル定数の内部整合性はコンパイル時 `const _: () = assert!(...)`（`kernels_tiled_pipeline.rs`）で検査済み）
- `cargo run -p fandhe-ai-backend-cuda --example gemm_tiled_pipeline_bench` を実際に実行し、CUDA driver 到達→NVRTC
  不在検出→graceful skip の分岐が意図どおり動作することを確認済み（出力: `CudaGemm::new failed
  (CUDA NVRTC library unavailable: ...); nothing to measure.`）
- `kernels_tiled_pipeline.rs` 内の `#[cfg(test)]`（cp.async 命令実在検査・REQ-8 境界ガード実在検査・Rust 側定数と
  カーネル `#define` の整合検査・`tiled_pipeline_f32_source_with_stages` の範囲検証・commit/wait_group 出現回数検査）が
  全て green
- `cargo test -p fandhe-ai-backend-cuda --test cpu_cuda_tiled_pipeline_parity` の環境適応スモーク
  （`tiled_pipeline_parity_smoke_env_adaptive`）が green（NVRTC 不在検出で early return）。実機必須の残り 6 テストは
  `#[ignore]`（未実行）

未検証の事項（DGX Spark GB10 実機セッションへ引き継ぐ）:

- NVRTC が `cp.async.cg.shared.global`／`.commit_group`／`.wait_group` を受理するかどうか（本カーネルはインライン
  PTX の命令セット自体は `kernels_mma_tf32.rs::mma_tf32_cp_async16` と全く同一のため、その経路がコンパイル可能であれば
  本カーネルも可能である蓋然性は高いが未確認）
- `as_tile`/`bs_tile` の共有メモリレイアウト（転置なし。本ファイル冒頭コメント参照）・レジスタブロッキング
  （4×4 外積）の実際の正しさ（`cargo test -- --ignored` の parity テストでのみ確認可能）
- 3 stage（既定）・4 stage（オンデマンドコンパイル変種）のスループット差・occupancy への影響

## 実測手順（DGX Spark GB10 セッション向け）

1. **正しさ検証を先に実行する**（性能計測より優先）:
   ```sh
   cargo test -p fandhe-ai-backend-cuda --test cpu_cuda_tiled_pipeline_parity -- --ignored
   ```
   全 PASS を確認してから性能計測へ進む。`tiled_pipeline_matches_tiled_f32` が既存 `run_tiled_f32` との相互比較も
   兼ねるため、この 1 テストの PASS だけでも「新カーネルが既存本番経路と同一の GEMM を計算している」ことの追加保証になる。

2. **性能計測**:
   ```sh
   cargo run -p fandhe-ai-backend-cuda --example gemm_tiled_pipeline_bench --release
   ```
   N=1024/2048/4096 それぞれについて、TFLOPS を 5 回計測中央値（`bench_harness::MeasurementConfig::default` の
   warmup/計測 20 回以上）で記録する。**計測区間は 2 段に分かれる**（codex-review P2／Cursor Bugbot 指摘。PR #1071
   `gemm_tiled_pipeline_bench.rs` モジュールコメント「計測区間の統一」参照。異なる区間の TFLOPS を同じ比率へ混ぜると
   「転送有無の違い」が「stage 数増加による改善」として誤計上されるため、実装は比較を分離済み）:
   - **転送込み同士**: `tiled_f32`（本番既定経路） vs `pipeline3`（既定 3 stage・転送込み）。`pipeline3_over_tiled`
     はこの区間の比率
   - **GPU-only 同士**: `pipeline3_gpu_only`（既定 3 stage） vs `pipeline4_gpu_only`（4 stage）。
     `pipeline4_over_pipeline3_gpu_only` はこの区間の比率で、cp.async ステージ数の増加そのものの効果を表す

3. **3 vs 4 stage の追加スイープが必要な場合**: `examples/gemm_tiled_pipeline_bench.rs` の `STAGE_4` 定数を変更するか、
   `CudaGemm::compile_tiled_pipeline_variant(&device, stages)` を直接呼ぶ小さな診断バイナリを追加する
   （`gemm_wmma_tf32_staged_stages_bench.rs` の段数スイープパターンを参考にできるが、tiled pipeline は静的共有メモリ
   予算内（`TP_MAX_STAGES=4` でも約 37KiB < 48KiB）に収まるため、staged 版のような動的共有メモリ opt-in 変種は
   不要である点に注意）。

4. **記録先**: 実測値・TFLOPS・対 tiled f32 比・対 candle 比（`docs/perf/cuda-gemm-kernel-improvement-policy.md` の
   比較基準）は本ファイルの「実測結果」節（プレースホルダ。以下）へ追記する。

## 実測結果（DGX Spark GB10 実機実測完了。#1031）

実機: DGX Spark GB10（compute capability (12, 1) = sm_121）・driver 580.173.02・CUDA 13.0.88
（`nvcc --version` 実測）・rustc 1.97.0。計測時 `nvidia-smi --query-gpu=utilization.gpu
--format=csv,noheader` で 0% を確認済み（他プロセス非同居）。commit
`10011cd4f8ef097351c0dc1244eb55c8a021040b`（rsync 転送元 worktree の HEAD）。

**正当性検証（手順 1）**: `cargo test -p fandhe-ai-backend-cuda --release --features
internal-diagnostics --test cpu_cuda_tiled_pipeline_parity -- --ignored --nocapture`
（`internal-diagnostics` feature 指定が本コマンドに必須である点は本ファイルの手順記載から
更新: feature 未指定では `cargo test` が `requires the features: internal-diagnostics` で
即座に失敗する）。**9 件全て PASS**（`tiled_pipeline_matches_tiled_f32`・
`tiled_pipeline_matches_reference_across_shapes`・`tiled_pipeline_k4096_stress` を含む）。
これは本カーネル（`TILED_PIPELINE_F32_BODY`。`cp.async.cg.shared.global`／
`.commit_group`／`.wait_group` を含む）が sm_121 実機で NVRTC 構文検証を初めて通過し、
かつ CPU 参照実装・`run_tiled_f32` の双方と数値一致（複合判定: 相対誤差 1e-3 未満 または
絶対誤差 1e-5 未満）することを確認した記録である（本ファイル冒頭「状態」節の
「いかなる実機でも未検証」は本実測により解消）。

**性能計測（手順 2）**: `cargo run -p fandhe-ai-backend-cuda --example
gemm_tiled_pipeline_bench --release --features internal-diagnostics`（同様に feature 指定が
必須。既存記載の手順ではエラーになったため実行時に追加した）。

列は `gemm_tiled_pipeline_bench.rs` の出力そのまま（区間を混ぜない。上記「実測手順」手順 2 参照）。

| size (M=N=K) | tiled_f32 TFLOPS（転送込み） | pipeline3 TFLOPS（転送込み） | pipeline3_over_tiled | pipeline3_gpu_only TFLOPS | pipeline4_gpu_only TFLOPS | pipeline4_over_pipeline3_gpu_only |
|---|---|---|---|---|---|---|
| 1024 | 3.9053 | 5.1523 | 1.3193 | 11.2165 | 10.7736 | 0.9605 |
| 2048 | 5.4793 | 7.8827 | 1.4386 | 13.0398 | 12.1491 | 0.9317 |
| 4096 | 0.2120 | 0.2104 | 0.9926 | 9.9062 | 9.7746 | 0.9867 |

**観察**:

- 転送込み区間（`tiled_f32` vs `pipeline3`）では N=1024・2048 で pipeline3 が
  1.32〜1.44 倍改善する一方、**N=4096 では両者とも 0.21 TFLOPS 台まで落ち込み
  `pipeline3_over_tiled` が 0.99 倍（実質横ばい）**になっている。この N=4096 の絶対値の
  低さは `docs/perf/cuda-fresh-gemm-n2048-overhead-diagnosis.md` §6.4 が記録する
  「N=4096 P4（fresh モードの D2H/確保オーバーヘッド）のばらつき（540 ms ↔ 39 ms、原因未特定）」
  と同系統の fresh モード転送込み経路の残存オーバーヘッド事象と整合する挙動であり、
  `tiled_f32`（本番既定経路）自体も同じ落ち込みを示していることから **本ファイルが対象と
  する cp.async パイプライン化固有の劣化ではない**（`tiled_f32` 側もほぼ同水準まで
  落ちているため、両者の相対比較である `pipeline3_over_tiled` の値自体は破綻していない）。
  原因調査は上記診断ドキュメントのスコープであり本ファイルでは追跡のみとする。
- **GPU-only 区間（`pipeline3_gpu_only` vs `pipeline4_gpu_only`）では 3 サイズとも
  4 stage が 3 stage を下回る**（`pipeline4_over_pipeline3_gpu_only` = 0.93〜0.99 倍）。
  cp.async ステージ数を 3→4 に増やしても sm_121 では改善せず、静的共有メモリ予算内
  （§「実測手順」手順 3 の記載どおり TP_MAX_STAGES=4 でも約 37KiB）に収まる範囲でも
  occupancy 等の他要因が支配的である可能性が高い。3 stage（既定）を据え置く判断を
  裏付ける結果であり、4 stage への切替は推奨されない。
- GPU-only 経路の絶対値（3 stage: 9.91〜13.04 TFLOPS）は転送込み経路（0.21〜5.15
  TFLOPS）を大きく上回り、H2D/D2H・出力バッファ確保のオーバーヘッドが依然として
  支配的であることを示す。この転送込みオーバーヘッドの削減自体は本イシューのスコープ外
  （「スコープ外事項」節・本番既定経路への結線判断は #1035 が担う）。

## #1137 本番結線判断（GB10 実測）

イシュー #1137「cp.async 多段パイプライン（#1033）の GB10 実測に基づき本番結線可否を
判断する」の実測記録・採否判断。実装は `crates/backend-cuda/src/gemm.rs::
CudaGemm::select_tiled_f32_kernel`（`tiled_f32_kernel_kind` 純粋関数による形状条件付き
選択）。詳細な計測コマンド・生ログは `docs/perf/logs/cuda-tiled-pipeline-wiring-1137/`
（`env_info.txt`・`gateA_bitexact.log`・`gateB_parity.log`・
`gateC_floor_bench_and_pipeline_bench.log`）を参照。

**計測環境**: DGX Spark GB10（sm_121）・driver 580.173.02・CUDA 13.0.88・rustc 1.97.0。
commit `062ca1e8d58ede1c50de22c5e7cbb41b7b5ee06a`（rsync `.rev-stamp` 実測一致確認済み）。
計測前 GPU utilization 0%。

### ゲート A（bit 一致・最優先）

`tests/cpu_cuda_tiled_pipeline_parity.rs::tiled_pipeline_matches_tiled_f32_classic_bit_exact`
（classic 版 `run_tiled_f32_classic` と pipeline 版 `run_tiled_pipeline_f32` の出力を
`assert_eq!` でビット完全一致検査）**PASS**。あわせて結線の end-to-end 回帰テスト
（`run_tiled_f32_dispatches_to_pipeline_for_aligned_shape`・
`run_tiled_f32_falls_back_to_classic_for_unaligned_shape`）も **PASS**（計 11 件全 PASS）。

これにより `docs/kernel-fusion.md` §2.2 の融合 epilogue（`gemm_bias_act`）bit 完全一致
契約が #1137 結線後も成立することが実機で裏付けられた（下記ゲート B の
`gemm_bias_act_parity` 0 fail と合わせて確認）。

### ゲート B（parity 0 fail・境界検査）

9 バイナリ（`gemm_tiled`・`gemm_bias_act_parity`・`backend_ops_real_device`・`gemm_auto`・
`gemm_resident_real_device`・`cpu_cuda_parity`・`gemm_tf32_optin`・`transpose_parity`・
`tensor_core_real_device`・`gemm_f32_variants`）を `--ignored --nocapture
--test-threads=1` で実行。tiled_f32 classic/pipeline 分岐を経由する 8 バイナリは**全 PASS
（0 failed）**。

2 件の既存事象を観測したが、いずれも `run_tiled_f32`／`tiled_pipeline` を経由しない
別カーネル系統（TF32 `mma.sync`／WMMA）に起因し #1137 と無関係と判断した:

- `gemm_tf32_optin_on_matches_cpu_across_shapes`（TF32 opt-in 経路の CPU-CUDA 複合判定）:
  既知の TF32 数値許容誤差事情（`docs/perf/cuda-tensor-core-tolerance-*.md`）。
- `tensor_core_parity_record`（同上の TF32 WMMA 経路）・`tensor_core_tflops_record`
  （エラーメッセージ自体が「GB10 実機実測〈2026-09-03〉で既知 red。イシュー #1131 の
  受け入れ条件へ引き渡す」と明記する WMMA(f16) 性能事象。`docs/perf/
  cuda-wmma-f16-perf-triage.md`・イシュー #1123/#1130/#1131）。

### ゲート C（性能・同一プロトコル）

`cuda_floor_bench` を §7.4 と同一プロトコルで 5 回実行した `tiled_f32_tflops`
（#1137 結線後 = 実際の本番ディスパッチ値。判定対象の N=1024/2048/4096 はいずれも
4 の倍数のため cp.async パイプラインへ分岐する）の中央値と、`docs/perf/
cuda-gemm-simt-register-blocking.md` §7.4 baseline（結線前 classic 固定。commit
`1a32082`）との比較:

| N (M=N=K) | after 中央値（TFLOPS） | before baseline §7.4（TFLOPS） | after/before |
|---|---|---|---|
| 1024 | 11.2278 | 6.7470 | **1.664** |
| 2048 | 12.9984 | 7.4819 | **1.737** |
| 4096 | 10.2188 | 6.7485 | **1.514** |

`gemm_tiled_pipeline_bench`（classic vs dispatch。GPU-only 起動込み・拡張形状
N=256/512/1024/2048/4096）でも全形状で `dispatch_over_classic` ≥ 1.15（1.15〜1.44 倍）
と後退なし。

### 判定・採否

事前宣言した判定基準（REQ-8 判定形状 N=2048/4096 の after/before ≥ 1.00、参考形状
N=256/512/1024 で ≥ 0.95）を、判定形状・参考形状のいずれも大きく上回って満たした
（最小 1.514 倍 ≫ 1.00・最小 1.15 倍 ≫ 0.95）ため、**最小形状閾値の追加なしでそのまま
採用（ADOPT）と判断した**。ゲート A（bit 一致）が PASS したため、融合 epilogue の
bit 完全一致契約（`docs/kernel-fusion.md` §2.2）も維持されている。結線コードは
`gemm.rs::CudaGemm::select_tiled_f32_kernel`（`perf(backend-cuda): cp.async 多段
パイプラインを tiled f32 経路へ形状条件付きで結線する (#1137)` コミット）として
既に main へ取り込み済み。

framework-compare 経由（candle 比・reuse モード）の #1137 反映後の値は
`docs/perf/cuda-gemm-candle-gate-remeasurement.md`（イシュー #1142）を参照。本節の
after/before はカーネル単体（launch-only）の比較であり、framework-compare の reuse
計測境界（H2D/D2H を含む `Tensor<f32>` ホスト常駐）とは前提が異なる点に注意する。

## スコープ外事項（本 PR では対応しない）

- **pipeline 版 `TILED_BIAS_ACT_F32`（epilogue 融合）**: 現行の融合カーネルは classic
  タイリングのみに対応する。pipeline 側の epilogue 融合・resident bias_act 経路への
  横展開は別イシューで扱う（#1137 実装計画 §8）。
- **`gemm_variant_selection`（#1035）の DoubleBuffer 閾値再補正**: Simple 経路が
  #1137 以降 pipeline を暗黙に含むようになるため、既存の形状別ヒューリスティックとの
  重複・閾値の妥当性は再検証が必要（#1137 実装計画 §8）。
- **N=4096 転送込み経路の fresh オーバーヘッド**: `cuda-fresh-gemm-n2048-overhead-diagnosis.md`
  のスコープのまま（本ファイルで追跡のみ）。
- **ブロック実行順スウィズル（#1034）の本番結線判断**: 別イシューのスコープ（#1137
  実装計画 §8）。**#1139 で GB10 実機によるゲート 0〜1 を実施し、ゲート 1（classic 内 A/B）が N=512 で
  判定基準 ≥0.95 を満たさず不合格のため不採用と判定した（`CudaGemm::new` への結線は行わない）。詳細は
  `docs/perf/cuda-gemm-tiled-f32-swizzle-ab.md` を参照**。
- **レジスタブロッキング拡大・bank conflict 対策**: 兄弟イシュー #1032（並行実装の可能性あり）のスコープ。本 PR は
  #1032 のマージ状況を実装開始時に確認したが未マージだったため、既存 `kernels.rs::TILED_F32` を書き換えず新規
  モジュール（`kernels_tiled_pipeline.rs`）に自己完結した実装を追加した（コンフリクト回避）。

## #1343 128×64×16 候補の追加（opt-in・未実測）

親イシュー #1342（sub-issue 2 件構成）配下のイシュー #1343「perf(backend-cuda):
128×64×16 f32 pipeline カーネル（8×4 レジスタブロック・XOR スウィズル・cp.async
zfill 境界）を opt-in で追加し現行 64×64 経路との出力 bit 同一を全形状で自己検証
する」の実装記録。**本イシューは実装・自己検証のみを扱い、GB10 実機の性能実測・
本番結線可否判断は兄弟イシュー #1344 が担う**。本節時点で本番既定経路
（`CudaGemm::new` → `select_tiled_f32_kernel`）は一切変更されていない。

### タイル構成・差分（既存 64×64 版との対比）

| 項目 | 64×64（既存・#1033/#1137） | 128×64（本イシュー・#1343） |
|------|---------------------------|------------------------------|
| ブロックタイル `BM×BN` | 64×64 | **128×64** |
| 1 スレッド担当（`THREAD_M×THREAD_N`） | 4×4 | **8×4** |
| ブロック内スレッド数 | 256（16×16） | 256（16×16。不変） |
| バンク衝突対策 | 行幅パディング（`TP_A_PAD`/`TP_B_PAD`） | **A フラグメントのみ 16B チャンク単位 XOR スウィズル（パディングなし）** |
| ステージ範囲 | 2〜4 | **2〜3**（下記「共有メモリ予算・occupancy 訂正」） |

実装は `crates/backend-cuda/src/kernels_tiled_pipeline_128x64.rs`（新規モジュール。
`kernels_tiled_pipeline.rs` と同型の位置づけ・opt-in 専用）。カーネルソース生成
（`tiled_pipeline_128x64_f32_source`）・const assert 群・A フラグメントスウィズル
の Rust 側参照実装（`swizzled_chunk_a`）・自己検証テストを同モジュール内に完結
させた。

### なぜパディングでなく XOR か（設計時に導出した論拠）

`THREAD_M` が 4→8 になったことで、A フラグメント読みが warp 内で 8 行離れた 2
グループへ分裂する（256 スレッド・16×16 格子・warp=32 は `ty` 2 値 × `tx` 16 値
から成るため）。A の行幅（`BK`=16 要素=64 バイト。cp.async 16B 転送粒度の倍数
という制約下）では、8 行差のバイトオフセット差（8×64B=512B）が常に 128B（32
バンク×4B）の倍数になり、**パディングでは 8 行差バンク衝突を解消できない**（行幅
をどう変えても cp.async 制約〈4 の倍数〉の下では 8×w×4 は常に 128 の倍数になる
ため）。

そこで 16 バイトチャンク（f32 4 要素）単位の XOR スウィズル `swz(row, chunk) =
chunk ^ ((row >> 3) & 3)` を採用した。`row` と `row+8` は `(row>>3)&3` の値が
必ず異なるため、スウィズル後は常に異なるバンクへ写る。この論拠は
`kernels_tiled_pipeline_128x64.rs::tests::a_fragment_swizzle_resolves_8_row_bank_conflict`
（全 row×chunk 組合せの機械検査）で固定した。B（行幅 256B）は読み・書きとも連続
16 チャンクで元々衝突しないためスウィズル不要と判断し、A のみへ適用した。

### 共有メモリ予算・occupancy 訂正（GB10 実測に基づく事実確認）

パディングなしのため 1 ステージあたり `(BM*BK + BK*BN) * 4B` = `(128*16 +
16*64) * 4` = 12,288 バイト。4 段では 49,152 バイトとなり全 compute capability
共通の静的 48KiB per-block 上限ちょうどで余裕がなく、かつ #1137（64×64 版）の
A/B 実測で 4 段は GB10 で劣化することが確認済みのため、**本カーネルのステージ
範囲は 2〜3 に限定した**（3 段時点で 36,864 バイト。コンパイル時 assert で 48KiB
上限を検査する）。

**親イシュー #1342 が想定していた「3 block/SM」は GB10 では成立しない**（本
イシューで判明した訂正）: `docs/perf/sm121-device-attributes.md` の GB10 実測
`MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = 102,400` バイトに対し、3 段の 36,864
バイト × 3 block = 110,592 バイト（> 102,400）のため、GB10 では smem 制約により
**同時常駐は 2 block/SM（16 warp/SM）に留まる**。実測（ptxas 資源値・実効
occupancy）による確認は #1344 の検証項目として引き継ぐ。

### bit 同一の論拠・opt-in 到達経路

各出力要素は `acc=+0.0` から `kk` 昇順・`t`（K タイル）昇順の単一 `fmaf()` 連鎖で
確定し、A のスウィズルは共有メモリの物理格納位置のみを変える純粋なアドレス置換
であるため演算順序に影響しない。したがって本カーネルは 64×64 版・classic 版
（`kernels.rs::TILED_F32`）のいずれとも bit 完全一致すると論証できる
（`kernels_tiled_pipeline_128x64.rs` 冒頭コメント「bit 同一の論拠」に詳細）。

自己検証は GPU 非依存（ホストモデルによる `matmul_reference_fma` との bit 一致。
`kernels_tiled_pipeline_128x64.rs::tests::host_model_matches_reference_fma_bit_exact`）
と、実機 `#[ignore]` テスト（`tests/cpu_cuda_tiled_pipeline_parity.rs` の
`tiled_pipeline_128x64_matches_pipeline_64x64_bit_exact`・
`tiled_pipeline_128x64_matches_classic_bit_exact`・
`tiled_pipeline_128x64_matches_reference_across_shapes`・
`tiled_pipeline_128x64_k4096_stress`・
`run_tiled_f32_optin_dispatches_128x64_for_aligned_shape`・
`compile_tiled_pipeline_128x64_variant_matches_run_tiled_pipeline_f32`）の双方で
用意した。**実機（DGX Spark GB10 等の compute capability 8.0 以降）は本実装セッ
ションの実行環境に存在せず、上記 `#[ignore]` テストは未実行のまま記録する**（#1344
で実施することを引き継ぐ）。

opt-in 到達経路（本番既定 `CudaGemm::new` は不変）:

- `gemm.rs::TILED_PIPELINE_128X64_PRODUCTION_ENABLED`（既定 `false`）: `true`
  へ切り替えると `CudaGemm::new` 自体が 128×64 版をコンパイルするようになる
  （#1344 が結線可否を判断した後にのみ切り替える運用。切替自体は本イシューの
  スコープ外）。
- `CudaGemm::new_with_tiled_pipeline_128x64(device)`（`internal-diagnostics`
  feature 限定）: 既定コンストラクタと同じ手順で構築したうえで `tiled_pipeline`
  スロットを 128×64 版へ差し替える診断専用インスタンス。`run_tiled_f32` 系
  3 入口はこのインスタンスに対しては整列形状で自動的に 128×64 経由になる。
- `CudaGemm::compile_tiled_pipeline_128x64_variant(device, stages)`
  （`internal-diagnostics` feature 限定）: GPU-only 常駐 API・bench 用の
  任意ステージ数変種。

### #1344 向け実測手順（引き継ぎ）

```sh
# GPU-only 性能比較（pipeline128x64_gpu_only 列。既定 3 stage）。
cargo run -p fandhe-ai-backend-cuda --example gemm_tiled_pipeline_bench --release \
  --features internal-diagnostics

# 実機 bit 一致自己検証（T7〜T11。本イシューで追加済み・GB10 実機未実行）。
cargo test -p fandhe-ai-backend-cuda --release --all-features \
  --test cpu_cuda_tiled_pipeline_parity -- --ignored --nocapture
```

`--all-features` を省くと `required-features` ゲートによりテストバイナリ自体が
ビルドされず false-green になる点は #1033 実装時と同じ注意（本節冒頭「実行手順」
節参照）。実測時は低負荷時に `uptime` を `docs/perf/logs/` 配下の env_info へ記録
する（`.claude/rules` の実機検証運用に準拠。内部ホスト名は含めない）。

## 関連ドキュメント

- `docs/perf/cuda-gemm-mma-pipeline.md`（TF32 `mma.sync`/`ldmatrix`/`cp.async` 経路の同型記録。本ファイルの
  パイプライン骨格の移植元）
- `docs/perf/cuda-gemm-kernel-improvement-policy.md`（FP32 SIMT 経路の背景・candle 比較基準）
- `crates/backend-cuda/src/kernels_tiled_pipeline.rs`（カーネルソース・設計コメント）
- `crates/backend-cuda/src/kernels_tiled_pipeline_128x64.rs`（イシュー #1343。128×64×16 候補のカーネルソース・設計コメント・自己検証テスト）
