# CUDA GEMM persistent タイルキュー版 f32 pipeline カーネルの opt-in 追加（#1346）

イシュー #1346「perf(backend-cuda): persistent タイルキュー版 f32 pipeline カーネル（grid=SM 数・
atomic タイル取得）を opt-in で追加し非 persistent 版との出力 bit 同一を確認する」の記録。**本イシューは
カーネル・opt-in API・bit 同一テスト・計測ハーネスの整備に閉じ、本番結線（`select_tiled_f32_kernel` への
分岐追加）は行わない**。GB10 実機での bit 同一自己検証・5 回計測中央値・採否判断は兄弟イシュー #1347 へ
引き継ぐ（本ドキュメントは実機記入欄を残す）。

## 1. 背景

- 親 #1345（祖 #1341 Phase 5「CUDA GEMM 構造再設計」）: GB10（sm_121）は SM 48 基のため、f32 pipeline
  経路（`kernels_tiled_pipeline.rs`・64×64 タイル・#1137 で本番結線済み）では「ブロックタイル数 / 48」
  の端数（wave quantization。`docs/cuda-streamk-decision.md` §2）が末尾 wave の GPU 遊休時間の一次要因
  になりうる。
- 本イシューは、grid を SM 数（または占有可能 block 数）に固定し、各 CTA がグローバル atomic カウンタから
  次の未処理出力タイルを動的に取得する persistent 版カーネルを opt-in で追加し、非 persistent 版との
  出力 bit 同一を機構として保証することを目的とする。K 分割は行わない（1 出力タイル = 1 CTA が K 全体を
  担当する）ため、タイル内の蓄積順序自体は非 persistent 版と変わらない。

## 2. 設計

### 2.1 ソース分割（`kernels_tiled_pipeline.rs`）

既存の単一定数 `TILED_PIPELINE_F32_BODY`（64×64 pipeline カーネル本体テンプレート）を 4 断片へ分割した:

| 断片 | 内容 |
|------|------|
| `TP_CP_ASYNC_HELPER` | `cp.async` 16 バイト転送ヘルパー（`tp_cp_async16`）。非 persistent・persistent 共有 |
| `TP_NON_PERSISTENT_PREFIX` | 非 persistent 版の関数シグネチャ・共有メモリ宣言・`blockIdx` 由来の `block_row0`/`block_col0` |
| `TP_TILE_CORE` | タイル内計算本体（プロローグ→K ループ→drain→エピローグ）。非 persistent・persistent 共有 |
| `TP_KERNEL_SUFFIX` | 非 persistent 版の関数閉じ括弧 |

これに加え persistent 版専用の `TP_KERNEL_PERSISTENT_PREFIX`（atomic タイルキュー取得ループ含む）・
`TP_KERNEL_PERSISTENT_SUFFIX`（ループ・関数の閉じ括弧）を追加した。`render_source`（非 persistent）は
`TP_CP_ASYNC_HELPER + TP_NON_PERSISTENT_PREFIX + TP_TILE_CORE + TP_KERNEL_SUFFIX` を連結し、
`render_persistent_source`（persistent）は `TP_CP_ASYNC_HELPER + TP_KERNEL_PERSISTENT_PREFIX +
TP_TILE_CORE + TP_KERNEL_PERSISTENT_SUFFIX` を連結する。

**AC-3（非 persistent 版のレンダリング結果がバイト同一）の検証**: 実装時に、分割前の
`TILED_PIPELINE_F32_BODY` 文字列と分割後 4 断片の連結が完全に一致することをビルド時スクリプトで機械検証
した（Python での文字列切り出し・再結合の assert）うえで実ファイルへ反映し、さらに実行時の回帰テスト
（`kernels_tiled_pipeline::tests::tiled_pipeline_fragments_reconstruct_non_persistent_source`）で継続的に
守る。これにより #1137 が確立した本番 PTX・モジュールキャッシュキー（ソース全文＋関数名）の意味は不変。

### 2.2 persistent 版カーネル（`gemm_tiled_pipeline_persistent_f32`）

```c
extern "C" __global__ void gemm_tiled_pipeline_persistent_f32(
    const float* a, const float* b, float* c,
    int m, int n, int k,
    unsigned int* tile_counter)
{
    __shared__ ... as_tile / bs_tile ...   // 非 persistent 版と同一宣言
    __shared__ unsigned int s_tile;

    int tiles_x = (n - 1) / TP_BN + 1;
    int tiles_y = (m - 1) / TP_BM + 1;
    int num_tiles = tiles_x * tiles_y;

    for (;;) {
        if (threadIdx.x == 0) { s_tile = atomicAdd(tile_counter, 1u); }
        __syncthreads();
        unsigned int tile_u = s_tile;
        if (tile_u >= (unsigned int)num_tiles) { break; }
        int tile = (int)tile_u;
        int block_row0 = (tile / tiles_x) * TP_BM;
        int block_col0 = (tile % tiles_x) * TP_BN;

        /* TP_TILE_CORE（非 persistent 版と完全共有） */
    }
}
```

- タイル→座標の写像（`tile / tiles_x`・`tile % tiles_x`）は非 persistent 版の `blockIdx.y`/`blockIdx.x`
  （grid x = n 方向）と同じ走査順に対応させている。
- `s_tile` の unsigned 比較（`tile_u >= (unsigned int)num_tiles`）は、カウンタが全 CTA 分の余剰
  `atomicAdd` により `num_tiles` を超えて進み続ける前提の下で、先に `int` へキャストすると理論上の
  桁あふれで負値化し比較をすり抜けうる（REQ-8・fail-closed）ため、比較後にのみ `int` へ変換する。
- `break` はブロック一様（全スレッドが `__syncthreads()` の後に同じ `s_tile` を読んでから判定）。
  `s_tile` への WAR（次イテレーションの thread-0 書き込みと前イテレーションの全スレッド読み出し）は
  `TP_TILE_CORE` 末尾の drain `__syncthreads()`（cp.async 完了待ちと共用）が担保する。

## 3. bit 同一の根拠

`TP_TILE_CORE`（プロローグ・K ループ・drain・エピローグ）は非 persistent 版・persistent 版で**完全に同じ
文字列**を共有する。両者の違いは「出力タイル→CTA」の割り当て方法（`blockIdx` 直接 対 atomic タイル
キュー）のみであり、これは各出力タイルの**どの CTA が計算するか**を変えるだけで**各要素がどう計算されるか**
は変えない。タイルキュー用の `atomicAdd` は `unsigned int` カウンタ（スケジューリング専用）にのみ作用し、
GEMM の数値蓄積（`float` の `acc[][]`）には一切触れない。よって `.claude/rules/coding-rust.md` の FMA
契約・`kernels_mse.rs`/`kernels_rmsnorm.rs` が禁止する「float `atomicAdd` による非決定的な結合順序」とは
別種の atomic であり、決定性契約を破らない。同一入力に対し persistent 版と非 persistent 版は出力 bit
同一になる。

この根拠を機構として固定するため、以下を静的テスト（GPU 不要。`kernels_tiled_pipeline.rs`）で検査済み:

- `tiled_pipeline_persistent_source_uses_single_integer_atomic_add`: persistent 版ソースの `atomicAdd`
  は `s_tile = atomicAdd(tile_counter, 1u);` のちょうど 1 箇所のみで、`acc[][]`／出力 `c` への
  `atomicAdd` が存在しないこと・非 persistent 版ソースには `atomicAdd` が一切現れないこと。
- `tiled_pipeline_persistent_source_retains_manual_bounds_checks`: REQ-8 の手動境界チェック（cp.async
  ゼロ充填・エピローグ guarded store）が `TP_TILE_CORE` 共有により persistent 版でも維持されていること。
- `tiled_pipeline_persistent_commit_wait_group_counts`: cp.async の `commit_group`/`wait_group` 会計
  回数（各 2 回）が非 persistent 版と同一であること。

実機での**出力そのものの** bit 同一検証（AC-2 本体）は
`tests/cpu_cuda_tiled_pipeline_persistent_parity.rs::tiled_pipeline_persistent_matches_non_persistent_bit_exact`
が担う（§5「実測」節参照。未実測明記）。

## 4. opt-in API（`internal-diagnostics` feature 限定）

- `kernels_tiled_pipeline::tiled_pipeline_persistent_f32_source()` / `_with_stages(stages)`:
  persistent 版カーネルソース生成。
- `gemm::PersistentTiledPipelineFunction`: コンパイル済みハンドル（`func`・生成元 `context_ptr`・
  タイルキューカウンタ `tile_counter: CudaSlice<u32>`・`num_sms`・`blocks_per_sm` を保持）。
- `CudaGemm::compile_tiled_pipeline_persistent_variant(device, stages, blocks_per_sm)`: オンデマンド
  コンパイル。`blocks_per_sm: Some(1)` で「grid = SM 数」（イシュー #1346 の受け入れ条件が挙げる構成）、
  `None` で `cudarc` の `occupancy_max_active_blocks_per_multiprocessor` による占有率実測既定。
  `Some(0)` は `CudaError::InvalidKernelConfig` で拒否。SM 数取得不能・占有率実測 `0` は
  `CudaError::TiledPipelineUnavailable` で拒否（grid=0 の driver launch エラーを未然に防ぐ）。
- `CudaGemm::launch_tiled_pipeline_persistent_f32(&self, &mut func, a_dev, b_dev, &mut c_dev, m, n, k)`:
  デバイス常駐バッファに対する GPU-only 起動（`launch_tiled_pipeline_f32` と同じ context 一致検証・
  形状検証・no-op 契約）。起動前にタイルキューカウンタをストリーム順序でゼロ化する。
- `CudaGemm::run_tiled_pipeline_persistent_f32(&self, &mut func, a, b, m, n, k) -> Vec<f32>`: ホスト
  スライス入出力の便宜 API。

いずれも `internal-diagnostics` feature（既定 off）でゲートし、`CudaGemm::new`（本番既定経路）はコンパイル
しない。純関数 `persistent_grid_blocks(num_tiles, num_sms, blocks_per_sm)`・
`persistent_tile_count(m, n)` は GPU 不要の単体テスト（`gemm.rs::tests`）で検査済み。

新規 `unsafe` は `launch_tiled_pipeline_persistent_f32` 内の 1 箇所のみ（既存 `launch_tiled_pipeline_f32`
と同型の起動パターン。カーネル引数は起動前検証済みの `m`/`n`/`k` と 1:1 対応し、カーネル内の手動境界
チェックと合わせて OOB 読み書きが起きない根拠とする）。

## 5. 実測

**本エージェント実行環境（macOS）に CUDA 実機なしのため、GB10 実機での bit 同一自己検証（AC-2）・性能
計測は本イシューでは未実施**。実行手順・記入欄は以下のとおり（兄弟イシュー #1347 が実施する）。

```sh
# bit 同一自己検証（AC-2 本体・受け入れ判定）
cargo test -p fandhe-ai-backend-cuda --release --locked --features internal-diagnostics \
  --test cpu_cuda_tiled_pipeline_persistent_parity -- --ignored --nocapture --test-threads=1

# 非 persistent 版の既存回帰も非後退確認（本 PR の変更が既存経路に影響しないこと）
cargo test -p fandhe-ai-backend-cuda --release --locked --features internal-diagnostics \
  --test cpu_cuda_tiled_pipeline_parity -- --ignored --nocapture --test-threads=1

# 性能疎通確認（本イシューでは 1 回のみ。5 回計測中央値は #1347）
cargo run -p fandhe-ai-backend-cuda --release --locked --features internal-diagnostics \
  --example gemm_tiled_pipeline_persistent_bench -- --blocks-per-sm auto
```

### 5.1 #1347 向け判定基準（事前宣言。実測後に動かさない）

- **ゲート A（bit 一致・必須）**: `tiled_pipeline_persistent_matches_non_persistent_bit_exact`（全形状
  ×`blocks_per_sm ∈ {Some(1), None}`）PASS。1 つでも fail → 不採用。
- **ゲート B（parity）**: CPU 参照実装との複合判定（`assert_parity`）0 fail。
- **ゲート C（性能。5 回計測中央値）**: N=1024/2048/4096 × タイル形状（本イシューでは 64×64 のみ実装。
  128×64 は Phase 3 未実装のため対象外。§6「申し送り」参照）で `persistent_over_pipeline3` を計測。
  判定基準の具体的な閾値（例: N=4096 で ≥1.05・N=1024/2048 で ≥1.00 等）は #1347 側で GB10 実機の wave
  quantization 実態（タイル数 / 48 の端数分布）を確認したうえで確定する（本イシューでは事前に確定しない
  ——48 SM に対する 1024/2048/4096 のブロックタイル数〈16×16=256／32×32=1024／64×64=4096〉はいずれも
  48 の倍数から外れており定性的には persistent 化が効きうる形状だが、実際の改善幅は実機依存のため）。

### 5.2 GB10 実機記入欄（#1347 が埋める）

| 形状 | pipeline3_gpu_only (TFLOPS) | persistent_gpu_only (TFLOPS. blocks_per_sm=auto) | persistent_over_pipeline3 |
|------|------|------|------|
| N=1024 | 未実測 | 未実測 | 未実測 |
| N=2048 | 未実測 | 未実測 | 未実測 |
| N=4096 | 未実測 | 未実測 | 未実測 |

## 6. 申し送り（スコープ外・#1347 への引き継ぎ）

- **本番結線可否判断**: `select_tiled_f32_kernel`／`CudaGemm::new` への結線は本イシューでは行わない。
  §5.1 のゲート C 実測・採否判断を経て #1347（またはその後続）で判断する。
- **タイル形状 2 種（親 #1345 の受け入れ条件）**: 実装時間の制約により、persistent 版のタイル形状
  パラメータ化（`(bm, bn)` を実行時検証付きで選べる診断専用レンダラ。計画 Phase 3）は**見送った**。
  現状 persistent 版は 64×64 タイル（`kernels_tiled_pipeline::TP_BM`/`TP_BN` 固定）のみ実装済みで、
  128×64（`kernels_tiled_pipeline_128x64.rs`。#1343）の persistent 化は未実装。#1347 は 64×64 のみで
  実測可能であり、128×64 persistent 版が必要な場合は #1347 側で追加実装が必要になる。
- **Stream-K／K 分割**（出力 bit 同一を崩す設計）: 別 issue・要承認（`docs/cuda-streamk-decision.md`）。
  本イシューは K 分割を行わない前提のまま。
- **タイル取得順の L2 局所性スウィズル**（#1139 の classic 版不採用判断と整合させて再検討）: 別 issue。

## 7. 変更ファイル一覧

- `crates/backend-cuda/src/kernels_tiled_pipeline.rs`: 断片分割・persistent 版レンダラ・静的テスト追加。
- `crates/backend-cuda/src/gemm.rs`: `PersistentTiledPipelineFunction`・
  `compile_tiled_pipeline_persistent_variant`・`launch_tiled_pipeline_persistent_f32`・
  `run_tiled_pipeline_persistent_f32`・純関数 `persistent_grid_blocks`／`persistent_tile_count`・単体
  テスト追加。
- `crates/backend-cuda/src/lib.rs`: `PersistentTiledPipelineFunction` の feature ゲート付き re-export。
- `crates/backend-cuda/Cargo.toml`: `[[example]] gemm_tiled_pipeline_persistent_bench`・
  `[[test]] cpu_cuda_tiled_pipeline_persistent_parity` を `required-features = ["internal-diagnostics"]`
  付きで追加。
- `crates/backend-cuda/tests/cpu_cuda_tiled_pipeline_persistent_parity.rs`（新規）: 環境適応スモーク＋
  実機 `#[ignore]` bit 同一・parity・fail-closed テスト群。
- `crates/backend-cuda/examples/gemm_tiled_pipeline_persistent_bench.rs`（新規）: 非 persistent vs
  persistent の GPU-only A/B 計測ハーネス（`--sizes`／`--stages`／`--blocks-per-sm` 引数対応）。
