# mma_f16 warp タイル拡大候補のレジスタ収支検証（#803）

親: GEMM 性能改善ツリー #479 → Phase 4 親 #789。イシュー #803「perf(backend-cuda):
mma_f16 warp タイル拡大の設計とレジスタ収支検証」の実測記録。

**本イシューは設計・事前検証のみ**を担う。本番カーネル定数
（`crates/backend-cuda/src/kernels_mma.rs::MMA_WARP_TILES_M`/`_N`）は本イシューでは
変更しない。本番結線（採用形状への定数変更・`__launch_bounds__` 付与・実機ベンチ）は
後続 #804（dependsOn: #803）のスコープ。

## 1. 背景

現行の `mma_f16` カーネルは warp あたり `2x2` 命令タイル（warp タイル `32x16`。
`kernels_mma.rs` の `MMA_WARP_TILES_M`/`_N`・`MMA_WARP_M`/`_N`）で、CUTLASS 標準
WarpShape `64x64` の 1/8 面積。Tensor Core 1 発行あたりの `ldmatrix` フラグメント
ロード比が高く、smem→レジスタ帯域が律速になる構造が課題（Phase 4 親 #789 の診断・
`docs/perf/cuda-gemm-bottleneck-diagnosis.md`）。

warp タイルを拡大すると 1 回の `ldmatrix` ロードあたりの `mma.sync` 発行数が増え
（下記 loads/mma 比）、smem 帯域負荷を相対的に下げられる一方、per-thread レジスタ
使用量が増えて occupancy（同時常駐ブロック数）が下がりうる（レジスタスピルが
発生すればさらに悪化する）。本ドキュメントはこのトレードオフを実機 `ptxas -v` で
定量化する。

## 2. 再現手順

1. `crates/backend-cuda/src/kernels_mma.rs::mma_f16_source_with_warp_tiles(warp_tiles_m,
   warp_tiles_n, launch_bounds)`（診断専用。`lib.rs::diagnostics` 経由で
   `internal-diagnostics` feature 限定で crate 外へ再公開）が、`mma_f16_source()`
   （本番既定カーネルソース）に対しアンカー完全一致置換で `WARP_TILES_M`/
   `WARP_TILES_N`/`WARPS_N` の `#define` を候補値へ差し替え、`launch_bounds` 指定時は
   シグネチャへ `__launch_bounds__(v)` を付与したソース文字列を返す。本番定数
   （`MMA_WARP_TILES_M`/`_N`）自体は変更しない。
2. `docs/real-hardware-verification-env.md` の手順（rsync 転送 → SSH →
   `cargo run -p backend-cuda --example mma_ptx_dump --release
   --features internal-diagnostics -- --out-dir <dir>`）で、下記候補表 4 形状 ×
   `__launch_bounds__`（なし／導出スレッド数で明示付与）2 通り = 8 ソースを NVRTC で
   コンパイルし `.ptx` としてダンプする（base／swizzle の既存 2 ダンプに追加）。
3. example が出力する `ptxas -arch=sm_121 -v <file> -o <file>.cubin` コマンド
   （8 ファイル分）を実機で実行し、`registers`・`spill stores`・`spill loads` を
   stderr ログから記録する。

## 3. 机上見積もり

### 3.1 候補表

warp あたり per-thread のコア配列レジスタ（f32/u32 = 1 レジスタ換算。
`kernels_mma.rs::MMA_F16_BODY` の `d[WARP_TILES_M][WARP_TILES_N][4]`・
`a_frag[2][WARP_TILES_M][4]`・`b_frag[2][WARP_TILES_N][2]` の要素数から算出）:
`d = WTM*WTN*4`・`a_frag = 2*WTM*4`・`b_frag = 2*WTN*2`。

ブロックタイルは現行 `MMA_BM=64`・`MMA_BN=128`（`kernels_mma.rs` の値）を固定し、
`MMA_M=16`・`MMA_N=8` から `warp_m = MMA_M*WTM`・`warp_n = MMA_N*WTN`、
`warps_m = MMA_BM/warp_m`・`warps_n = MMA_BN/warp_n`、
`threads/block = warps_m * warps_n * 32` を導出する。

| 候補 | WTM x WTN | warp タイル | d | a_frag | b_frag | コア計 | warp 構成（warps_m x warps_n） | threads/block | loads/mma 比 |
|------|-----------|------------|---|--------|--------|--------|-------------------------------|---------------|--------------|
| 現行 | 2x2 | 32x16 | 16 | 16 | 8 | 40 | 2x8 = 16 warp | 512 | 1.00 |
| 案 A | 2x4 | 32x32 | 32 | 16 | 16 | 64 | 2x4 = 8 warp | 256 | 0.75 |
| 案 B | 4x2 | 64x16 | 32 | 32 | 8 | 72 | 1x8 = 8 warp | 256 | 0.75 |
| 案 C | 4x4 | 64x32 | 64 | 32 | 16 | 112 | 1x4 = 4 warp | 128 | 0.50 |

- loads/mma 比 = `(WTM + WTN) / (WTM * WTN)`（1 kstep あたり `ldmatrix` 発行数 /
  `mma.sync` 発行数の相対値。CUTLASS WarpShape `64x64` = `4x8` は `0.375`）。
- コア配列レジスタ計はカーネル全体のレジスタ使用量（インデックス計算・ループ変数・
  `cp.async` パイプライン用一時変数等を含む ptxas 実測値）の一部に過ぎない
  机上下限であり、実測値はこれを上回る。

### 3.2 occupancy 導出式（実測後に埋める）

- レジスタ制約: `floor(65536 / (ptxas 実測 regs/thread × threads/block))`
  （sm_121 の SM あたり総レジスタ数 65,536。出典: `docs/perf/sm121-device-attributes.md`
  の `MAX_REGISTERS_PER_MULTIPROCESSOR` 実測値）。
- smem 制約: `floor(SM あたり smem 容量 / MMA_SHARED_MEM_BYTES)`（静的 smem は
  ブロックタイル・パディング定数が候補間で不変のため `41,472B`（`kernels_mma.rs::
  MMA_SHARED_MEM_BYTES` の現行値）のまま。SM あたり smem 容量は
  `docs/perf/sm121-device-attributes.md` の `MAX_SHARED_MEMORY_PER_MULTIPROCESSOR`
  実測値を出典とする）。
- blocks/SM = 上記 2 制約の min（CUDA の per-SM 常駐ブロック数上限も別途適用され
  うるが、本ドキュメントのレジスタ/smem 比較では支配的要因のみを扱う）。

### 3.3 spill 判定

`ptxas -v` の `spill stores`/`spill loads` が 0 bytes かどうかを候補ごとに記録する
（#804 受け入れ条件「spill 0 維持」の分母になる）。

## 4. `__launch_bounds__` 検証方針

CUTLASS `device_kernel.h` 方式に倣い `__launch_bounds__(<ブロックスレッド数>)`
（`minBlocksPerMultiprocessor` は指定しない）を基本案とし、各候補を
**launch_bounds なし / あり（値 = その候補の threads/block）** の 2 通りで PTX 生成・
`ptxas -v` 比較し、付与値と付与要否を確定する（`.maxntid` が PTX に載ることで ptxas の
レジスタ割り当て前提が本番起動構成と一致する）。

## 5. 実機実測結果

**実行待ち**（`docs/real-hardware-verification-env.md` の手順で DGX Spark GB10
（sm_121）実機へ到達し `mma_ptx_dump` example を実行、8 ファイル分の `ptxas -v` を
掛けて本節を埋めること。到達できない場合は推定で埋めず本記録のまま残す。
`docs/cuda-tensor-core-design.md` §12「TMA プローブ」・§13「setmaxnreg プローブ」と
同じ「実行待ち」記録方式）。

| 候補 | launch_bounds | registers/thread | spill stores (bytes) | spill loads (bytes) | blocks/SM（レジスタ制約） | blocks/SM（smem 制約） |
|------|---------------|-------------------|----------------------|----------------------|--------------------------|--------------------------|
| 現行 2x2 | なし | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 |
| 現行 2x2 | あり(512) | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 |
| 案 A 2x4 | なし | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 |
| 案 A 2x4 | あり(256) | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 |
| 案 B 4x2 | なし | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 |
| 案 B 4x2 | あり(256) | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 |
| 案 C 4x4 | なし | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 |
| 案 C 4x4 | あり(128) | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 |

## 6. 判断

実機実測が完了するまで採用形状は未確定。判断が確定した際は
`docs/cuda-tensor-core-design.md` §14 へ記録し、本ドキュメントへは実測表のみを残す
（判断の一次記録は design doc 側に集約する方針。`docs/perf/cuda-gemm-mma-block-tile.md`
と `cuda-tensor-core-design.md` の役割分担を踏襲）。

## 7. #804 への引き渡し事項

- 採用形状（spill 0 かつ loads/mma 比最小の候補。spill が出る候補は除外根拠として
  その値を記録する）
- `__launch_bounds__` 付与値・付与要否の決定
- swizzle 条件（`kernels_mma.rs::mma_f16_source_with_swizzle`）・`gemm_auto.rs` の
  warp 刻み定数（`MMA_WARP_M`/`_N` を候補列挙の刻みとして参照する箇所）への波及有無
