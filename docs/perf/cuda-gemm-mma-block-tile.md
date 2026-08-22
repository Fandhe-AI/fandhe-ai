# CUDA GEMM mma.sync ブロックタイル拡大（BM/BN/BK）計測記録（#494・B-3）

イシュー #494「perf(backend-cuda): レジスタブロッキング後のブロックタイル（BM/BN/BK）拡大と SMEM・レジスタ予算の再計算」の実測記録テンプレート。
GEMM 性能改善ツリー #479 → Phase 2 親 #490 の B-3。先行 B-2（#493・warp あたり 2x2 レジスタブロッキング）とのペアで、K=4096 のメモリ律速緩和（データ再利用率向上）を狙う。

## 状態: 未実測・実機実行待ち（#502 で再計測）

本実装セッションの実行環境は CUDA **driver**（`libcuda`。compute capability 8.6・RTX 3060 実機）は存在するが NVRTC（`libnvrtc`）が存在しないため（`crates/backend-cuda/src/kernels_mma.rs` 冒頭コメント「検証状態」参照）、本ファイルが記録する変更（`MMA_BM`/`MMA_BN` 定数とカーネルソース `#define` の差し替え）は **NVRTC による構文検証を一度も通過していない**。sm_86（この実機）・sm_121（DGX Spark GB10。設計上のターゲット）のいずれでも未検証。`docs/perf/cuda-gemm-mma-pipeline.md`（B-0〜B-2 時点の同種記録）・`docs/perf/metal-gemm-dynamic-tile.md` の先例（実機での最初の実行が構文検証を兼ねる）と同じ位置づけ。

本実装セッションで検証済みの事項:

- `cargo build --workspace`（`const _: () = assert!(...)` によるコンパイル時境界検査。§1 参照）
- `cargo fmt --all -- --check` / `cargo clippy --workspace --all-targets --all-features -- -D warnings`
- `cargo test -p fandhe-ai-backend-cuda`（`kernels_mma.rs` 内 `#[cfg(test)]` の `#define` 整合検査・REQ-8 needle・タイル定数 pin・`gemm_mma.rs` の launch config div_ceil 被覆テスト）
- `tests/parity_nonregression.rs` の通常 CI 実行分（tolerance 定数 pin・fixture 自己整合。無変更で green）

未検証・実機実行待ちの事項（#502「Phase B 完了時点の再計測」へ引き継ぐ）:

- NVRTC によるカーネルソースの構文検証そのもの
- 候補 A〜D（下記 §3）の段階的計測・B-1（#492）比の TFLOPS 改善判定
- K=2048→4096 のスループット低下率（wmma_tf32 経路で約 −29%。`docs/perf/cuda-gemm-bottleneck-diagnosis.md` §1 実測）の mma.sync 経路での改善実測
- sm_121（DGX Spark GB10）実機の SMEM/レジスタ属性（`docs/perf/sm121-device-attributes.md` は未実測のまま。本ファイルの候補算出は全 compute capability 共通の保証値ベース）

## 1. 背景

B-2（#493）は warp あたり 2x2 レジスタブロッキング（warp タイル `32x16`）を導入した結果、ブロックタイル（`MMA_BM=32`・`MMA_BN=64`）に対する warp 数が 16→4 へ減り、ブロックスレッド数が 512→128 に低下した（`kernels_mma.rs::MMA_WARPS_M`/`MMA_WARPS_N` 定数直下のドキュメンテーションコメント参照）。これにより `cp.async` 協調ロードの並列度が下がり、B-2 単体でのスループット改善は受け入れ条件とされていない（#493 本文が明示）。

本イシュー（B-3）はブロックタイル（`MMA_BM`/`MMA_BN`）を拡大してブロックスレッド数を 512 相当（B-1 時点の水準）へ回復させ、B-2+B-3 ペアとしての改善を成立させる。

## 2. 数値 bit 一致の論拠（BK 不変）

各出力要素のアキュムレート順序は「K タイル `t` 順 → kstep 順」の `mma.sync` 系列のみで決まり、`MMA_BM`/`MMA_BN`（ブロックタイルの M/N 方向拡大）は関与しない。本イシューは **`MMA_BK=32` を変更しない**ため、出力は B-1（#492）/B-2（#493）時点と bit 一致を保つ。したがって `tests/parity_nonregression.rs` のベースライン fixture・tolerance 定数（`§1.2` 契約）は変更不要（`git diff` で無差分であることを機械確認する。§6 参照）。

## 3. タイル候補の算出

`SMEM(BM,BN,BK,S=3) = (BM*BK + BK*BN) * 2B * 3`。レジスタ概算/thread = アキュムレータ `d[2][2][4]`=16 本 + A フラグメント（`WARP_TILES_M`=2 個 x 4 レジスタ）=8 本 + B フラグメント（`WARP_TILES_N`=2 個 x 2 レジスタ）=4 本 = 28 本 + 索引・パイプライン用（正確な値は実機コンパイル後の `--ptxas-options=-v` 相当の出力、または `examples/gemm_mma_bench.rs` の function 属性クエリで確認する。§7 参照）。

| 候補 | BM/BN/BK | warp 構成 | スレッド | SMEM | kSteps(=BK/16) | 備考 |
|------|----------|-----------|---------|------|----------------|------|
| A | 64/64/32 | 2x4=8 warp | 256 | 24,576B | 2 | 最小差分・低リスク |
| **B（既定採用）** | **64/128/32** | **2x8=16 warp** | **512** | **36,864B** | **2** | スレッド数 512 回復（B-1 相当）・CUTLASS 既定（`128x256` は静的 48KiB 不成立）へ最も近い縮小形。B タイル協調ロードが 512 チャンク=1 反復/スレッドで整合 |
| C | 128/64/32 | 4x4=16 warp | 512 | 36,864B | 2 | M 縦長の対称候補 |
| D（参考・不採用） | 64/64/64 | 2x4=8 warp | 256 | 49,152B | 4 | 静的上限 48KiB ちょうどで余裕ゼロ。`BK` 拡大は §2 の bit 一致論拠が崩れるため既定にしない |

候補 B の cp.async 協調ロード反復回数の非対称性: A タイル（`BM*BK/8` = `64*32/8` = 256 チャンク）はブロック 512 スレッドに対し 1 反復で完了し 256 スレッドが遊休（`idx += blockDim.x` ループが 1 回で終わるため 2 回目の反復に入らない）、B タイル（`BK*BN/8` = `32*128/8` = 512 チャンク）は 512 スレッド全稼働で 1 反復と厳密に一致する。候補 B の「スレッド数回復が cp.async 並列度を改善する」根拠は主に B タイル側の反復回数減少（B-2 時点の 0.5 反復相当→本構成 1 反復）に基づき、A タイル側は元々 1 反復のままで直接の恩恵は小さい（候補 C は逆に A 側が厳密一致・B 側に遊休が生じる対称形）。実測時はこの非対称を踏まえて解釈すること。

`mma_f16_source_stages_are_swappable_without_kernel_source_edits`（`kernels_mma.rs`）は `stages=4` の共有メモリ使用量が本構成（BM=64/BN=128）で `49,152B` ちょうど（48KiB 上限に対し余裕ゼロ）になることを `<=` 判定のみで検査していた（本イシュー #494 時点の記述）。`STAGES=4` への変更自体は本イシューのスコープ外だが、将来検討する場合は BM/BN をさらに縮小しない限り成立しない点に注意。

**#498 追記（バンクコンフリクト対策パディング適用後）**: 上記「余裕ゼロ」はパディング**前**の値であり、#498 で `MMA_A_PAD`/`MMA_B_PAD`（バンクコンフリクト対策の非 2 冪パディング）を適用した結果、共有メモリ使用量は `41,472B`（stages=3）へ増加した。`stages=4` の使用量は `(64*40+32*136)*2*4 = 55,296B` となり **48KiB 上限を超過（不成立）** になった（パディング前の「余裕ゼロ」から「そもそも成立しない」へ変化）。テスト自体もこの変化に合わせ `docs/perf/cuda-gemm-mma-bank-conflict.md` §1「STAGES=4 スワップテストへの影響」の記載どおり改訂済み（詳細は同ファイル参照）。

候補 B の占有率概算（参考: compute capability 8.6 の公称仕様値。実機実測ではない。sm_121 は `docs/perf/sm121-device-attributes.md` 記入後に再確認）: SMEM/SM 約 100KiB（仕様上限）→ 2 ブロック/SM（パディング前 73,728B ≤ 100KiB。#498 のパディング適用後は 82,944B ≤ 100KiB で同じく 2 ブロック/SM を維持。詳細は上記「#498 追記」および `docs/perf/cuda-gemm-mma-bank-conflict.md` §「占有率への影響」参照）、1,024 スレッド ≤ 1,536、レジスタは 64 本/thread 以下なら 2 ブロック常駐可。全候補が `MMA_K_STEPS_PER_STAGE >= 2`（CUTLASS `mma_base.h` の `kWarpGemmIterations >= 2` 相当。`kernels_mma.rs` の `const _: () = assert!(MMA_K_STEPS_PER_STAGE >= 2, ...)` で機械検査）を満たす。

**本実装は候補 B（`BM=64`・`BN=128`・`BK=32`）を既定値として `kernels_mma.rs` に反映済み**（§2 の bit 一致論拠により安全側）。候補 A/C/D は実機での段階的計測用に本ファイルへ記録するのみで、コードには反映していない。

## 4. 段階的計測手順（実機・CUDA driver + NVRTC 搭載・compute capability 8.0 以上）

```sh
git fetch origin
git checkout perf/494-mma-block-tile-expansion   # 本イシューの実装ブランチ
cargo test -p fandhe-ai-backend-cuda -- --ignored --nocapture   # parity 非後退の全行検査（数値一致確認を性能計測より先に実施）
cargo run -p fandhe-ai-backend-cuda --example gemm_mma_bench --release   # 候補 B（既定値）の TFLOPS 計測
```

候補 A/C/D を計測する場合は `kernels_mma.rs` の `MMA_BM`/`MMA_BN` 定数とカーネルソース内 `#define BM`/`#define BN`/`#define WARPS_N` を候補表の値に一時的に差し替えてから同じ手順を実行する（`WARPS_M` はソース内に現れず `warp_id / WARPS_N` で導出されるため変更不要）。差し替え後は必ず `cargo test -p fandhe-ai-backend-cuda`（コンパイル時 `const _: () = assert!(...)` の再検査）を先に実行すること。

### 記録欄（実機セッションで埋める）

| 候補 | BM/BN/BK | M=N=K=2048 TFLOPS（5 回中央値） | M=N=K=4096 TFLOPS（5 回中央値） | B-1（#492）比 | 2048→4096 低下率 |
|------|----------|-------------------------------|-------------------------------|---------------|-------------------|
| A | 64/64/32 | 未計測 | 未計測 | 未計測 | 未計測 |
| B（既定） | 64/128/32 | 未計測 | 未計測 | 未計測 | 未計測 |
| C | 128/64/32 | 未計測 | 未計測 | 未計測 | 未計測 |

判定基準（#494 受け入れ基準 4・5 項）: B-2+B-3 ペアで 4096 の TFLOPS が B-1 時点を上回ること、2048→4096 の低下率が `docs/perf/cuda-gemm-bottleneck-diagnosis.md` の既知値（wmma_tf32 経路で約 −29%）から改善すること（mma.sync 経路自身の 2048→4096 低下率はこの表の「未計測」欄に実測記録する）。

## 5. リスク・安全側判断の記録

- **実測なしでの既定値変更リスク**: 候補 B が実機で最速である保証はない。緩和策: (1) `BK=32` 維持により数値は bit 一致で正しさリスクなし（§2）、(2) 本ファイルの候補表と段階的計測手順により #502 の再計測で最終確定可能、(3) 定数 2 行 + `#define` 2 行の差し替えで即時ロールバック可能
- **NVRTC 未検証**: 変更は定数値のみでカーネル構造は B-2 検証済み形のまま（新規 PTX 構文を追加していない）。構文リスクは増加しない
- **sm_121 実機属性未実測**: 候補算出（§3）は全 compute capability 共通の保証値（静的 SMEM per-block 48KiB・ブロック 1024 スレッド上限）ベース。sm_121 固有の SM あたり SMEM・レジスタファイルサイズは `docs/perf/sm121-device-attributes.md` の実測後に占有率概算を再確認する

## 6. §1.2 parity 非後退契約の機械確認

```sh
git diff origin/main -- crates/backend-cuda/tests/parity_nonregression.rs crates/backend-cuda/tests/common/parity_baseline
```

無差分であることを確認する（§2 の bit 一致論拠の裏付け。tolerance 定数・ベースライン fixture を変更していないことをコミット前に検査する）。

## 7. レジスタ予算の実測（任意・引き継ぎ）

`examples/gemm_mma_bench.rs` にロード済みカーネルの function 属性（レジスタ数・静的 SMEM）を出力するオプションを追加できれば、§3 の概算式に代わり実機でレジスタ予算を直接確認できる。本実装セッションでは `cudarc` の function 属性クエリ API（`cuFuncGetAttribute` 相当）が現行 `=0.19.8` 固定バージョンで到達可能かどうかを未調査のまま、実機 NVRTC 非搭載のため検証しようがない本セッションでの実装は見送った（推定で「到達不可」と書かない。単に未調査。#502 の実機セッションで調査・到達可能であれば追加実装する）。
