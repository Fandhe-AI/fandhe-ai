# CUDA GEMM mma.sync 共有メモリバンクコンフリクト対策（#498）計測記録

イシュー #498「perf(backend-cuda): 共有メモリのバンクコンフリクト対策（パディング先行・XOR swizzle は計測で判断）」の理論分析・実測記録テンプレート。
`crates/backend-cuda/src/kernels_mma.rs`（f16 `mma.sync`/`ldmatrix`/`cp.async` GEMM）の共有メモリタイル `as_tile`/`bs_tile` へ非 2 冪パディングを適用した対策の記録。依存イシュー #494（B-3 ブロックタイル拡大）・#486（A-6 プロファイルベンチ）はともに CLOSED 済み。

## 状態: パディング適用済み・実機バンクコンフリクト計測は未実施（実機実行待ち）

本実装セッションの実行環境は CUDA **driver**（`libcuda`。compute capability 8.6・RTX 3060 実機）は存在するが NVRTC（`libnvrtc`）が存在しないため（`kernels_mma.rs` 冒頭コメント「検証状態」参照）、本ファイルが記録する変更（`MMA_A_PAD`/`MMA_B_PAD` 定数追加とカーネルソース `as_tile`/`bs_tile` 配列次元の差し替え）は **NVRTC による構文検証を一度も通過していない**。sm_86（この実機）・sm_121（DGX Spark GB10）のいずれでも未検証。カーネル構造（パイプライン・命令列）自体は不変で、変更は配列次元と `#define` のみのため新規 PTX 構文の追加はない（#494 と同じ整理）。

本実装セッションで検証済みの事項:

- `cargo build --workspace`（`const _: () = assert!(...)` によるコンパイル時境界検査。§1 参照）
- `cargo fmt --all -- --check` / `cargo clippy --workspace --all-targets --all-features -- -D warnings`
- `cargo test -p backend-cuda`（`kernels_mma.rs` 内 `#[cfg(test)]` の `#define` 整合検査・SMEM 41,472B 固定・バンク位相分散ロック・STAGES スワップ改訂版・REQ-8 needle 群）
- `cargo test --workspace`（他クレートへの波及なし確認）
- §5 の `git diff origin/main` 無差分確認（parity 非後退契約）

未検証・実機実行待ちの事項（#502 系 実機セッションへ引き継ぐ）:

- NVRTC によるカーネルソースの構文検証そのもの
- nsight-compute によるバンクコンフリクト実測（§3 の判定基準）
- パディング適用前後の TFLOPS 比較（§4 の計測手順）
- XOR swizzle の採否判断（§3）

## 1. 背景・理論分析

`kernels_mma.rs` の共有メモリタイルは変更前、パディングなしで宣言されていた（`as_tile[STAGES][BM][BK]` = `[3][64][32]`・`bs_tile[STAGES][BK][BN]` = `[3][32][128]`。f16 2B/要素）。行幅が 2 冪バイト数のため `ldmatrix` のレーン群行アドレスがバンク位相で重なり、理論上以下のコンフリクトが発生しうる（CUDA 共有メモリの標準 32 バンク・4B/バンク・128B 周期モデル）:

- **A タイル**（行ストライド 64B = 16 バンク）: `ldmatrix.sync.aligned.m8n8.x4.shared.b16` が読む 8 行の開始バンクが 2 巡回で同一位相へ収束し、4-way バンクコンフリクト
- **B タイル**（行ストライド 256B = 32 バンク = バンク位相 0 固定）: `ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16` が読む 8 行が全て同一バンクへ収束し、8-way バンクコンフリクト（A より深刻）

### 対策: 非 2 冪パディング

`kernels_wmma_opt.rs`（`WMMA_TF32_OPT_A_PAD`/`WMMA_TF32_OPT_B_PAD`。TF32/f32 opt 経路）が同種問題を `+4` 要素パディングで回避済みだが、`kernels_mma.rs` は f16（2B/要素）+ `cp.async` 16B 転送粒度の制約から **8 要素（16B）単位が最小パディング**（f32 opt の `+4` 要素 = 8B では `cp.async` 宛先アドレス・`ldmatrix` 行アドレスの 16B 整列が崩れる）。

- `MMA_A_PAD = MMA_BK + 8 = 40`（行ストライド 80B = 20 バンク）: 8 行の開始バンクが `0,20,8,28,16,4,24,12` と全て相異なり、各 16B 読み（4 バンク幅）が 32 バンクを完全被覆 → A の `ldmatrix.x4` コンフリクト理論上解消
- `MMA_B_PAD = MMA_BN + 8 = 136`（行ストライド 272B = バンク位相 +4/行）: 8 行の開始バンクが `0,4,8,...,28` と分散 → B の `ldmatrix.x2.trans` の 8-way コンフリクトを理論上大幅低減

`crates/backend-cuda/src/kernels_mma.rs::MMA_A_PAD`/`MMA_B_PAD` 定数直下のドキュメンテーションコメントに同じ算出根拠を記載済み。バンク分散の機械検査は `kernels_mma.rs::tests::mma_tile_padding_distributes_bank_phase_across_rows` がロックする。

### 共有メモリ使用量への影響

`(64*40 + 32*136) * 2B * 3 stages = 41,472B`（パディング前 36,864B から増加）。per-block 48KiB（49,152B）上限に対し依然余裕あり（コンパイル時 `const _: () = assert!(...)` が機械検査。`kernels_mma.rs::MMA_SHARED_MEM_BYTES` 参照）。

### 占有率への影響

SMEM 36,864B→41,472B の増加で SM あたり常駐ブロック数が変わりうる。compute capability 8.6 の概算（SMEM/SM 約 100KiB、公称仕様値）では 2 ブロック/SM を維持（82,944B ≤ 約 100KiB）。sm_121（DGX Spark GB10）実機属性は `docs/perf/sm121-device-attributes.md` が未実測のため、実機での再確認が必要（#502 系へ引き継ぎ）。

### 数値への影響（bit 一致）

パディングは共有メモリ上の配置のみを変え、`mma.sync` のアキュムレート順序（K タイル t 順 → kstep 順）・FMA 契約に一切関与しない。#494 の `docs/perf/cuda-gemm-mma-block-tile.md` §2 と同じ論拠で parity 非後退契約のベースライン fixture・tolerance 定数は変更不要（§5 参照）。

### STAGES=4 スワップテストへの影響

パディング後は `stages=4` の共有メモリが `(64*40+32*136)*2*4 = 55,296B > 48KiB` となり静的上限不成立になる。`kernels_mma.rs::tests::mma_f16_source_stages_are_swappable_without_kernel_source_edits` は「stages=2 は上限内・stages=4 はパディング後は上限超過（BM/BN 縮小なしでは不成立）」を明示的に assert する形へ改訂済み（`docs/perf/cuda-gemm-mma-block-tile.md` §3 の「STAGES=4 は余裕ゼロ」記述も本ファイルの追記で更新済み）。

## 2. バンク位相計算の詳細

バンク番号 = `(行バイトオフセット / 4) % 32`（4B/バンク・32 バンク・128B 周期モデル）。

| タイル | パディング前行幅 | パディング前ストライド | パディング前 8 行の開始バンク | パディング後行幅 | パディング後ストライド | パディング後 8 行の開始バンク |
|--------|-----------------|------------------------|-------------------------------|-------------------|-------------------------|-------------------------------|
| A（`as_tile`） | 32 要素（`MMA_BK`） | 64B = 16 バンク | `0,16,0,16,0,16,0,16`（4-way 衝突） | 40 要素（`MMA_A_PAD`） | 80B = 20 バンク | `0,20,8,28,16,4,24,12`（分散） |
| B（`bs_tile`） | 128 要素（`MMA_BN`） | 256B = 32 バンク | `0,0,0,0,0,0,0,0`（8-way 衝突） | 136 要素（`MMA_B_PAD`） | 272B = 34 バンク相当（`mod 32` で位相 +4） | `0,4,8,12,16,20,24,28`（分散） |

## 3. XOR swizzle の採否判断基準

本 PR（#498）ではコードとして実装しない（パディング先行）。CUTLASS の 2 段 XOR・MLX の `tgp_padding`・metal-flash-attention の leading dimension 実値調整・TileKernels のパディング + swizzle が参照実装として挙げられているが、パディングが最小差分・最低リスクのため段階的アプローチを取る。

**採否判断基準**（実機セッションで確認する）:

1. nsight-compute で `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` 系メトリクス（共有メモリロードのバンクコンフリクト実測値）を計測する
2. パディング適用前後で当該メトリクスが有意に減少していることを確認する（§1 の理論分析どおり A/B タイル双方でコンフリクトがほぼ解消される想定）
3. パディング適用後もコンフリクトが有意に残存する場合のみ、CUTLASS 2 段 XOR の適用を検討する（索引演算が複雑になりコンパイル未検証環境では誤り検出不能なリスクが高いため、実測で残存が確認された場合のみ着手する）
4. XOR swizzle を採用する場合は本ファイルへ適用案概要・実測結果を追記し、`kernels_mma.rs` 冒頭コメント「バンクコンフリクト対策」節を更新する

## 4. 実機計測手順（実機・CUDA driver + NVRTC 搭載・compute capability 8.0 以上）

```sh
git fetch origin
git checkout perf/498-mma-smem-bank-conflict-padding   # 本イシューの実装ブランチ
cargo test -p backend-cuda -- --ignored --nocapture     # parity 非後退の全行検査（数値一致確認を性能計測より先に実施）
cargo run -p backend-cuda --example gemm_mma_bench --release   # パディング後の TFLOPS 計測（5 回中央値）
```

nsight-compute でのプロファイル（`docs/perf/cuda-gemm-bottleneck-diagnosis.md` の手順を踏襲）:

```sh
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum \
    cargo run -p backend-cuda --example gemm_profile_target --release
```

### 記録欄（実機セッションで埋める）

| 項目 | パディング前 | パディング後 | 差分 |
|------|-------------|--------------|------|
| バンクコンフリクトメトリクス（ld.sum） | 未計測 | 未計測 | 未計測 |
| バンクコンフリクトメトリクス（st.sum） | 未計測 | 未計測 | 未計測 |
| M=N=K=2048 TFLOPS（5 回中央値） | 未計測 | 未計測 | 未計測 |
| M=N=K=4096 TFLOPS（5 回中央値） | 未計測 | 未計測 | 未計測 |
| XOR swizzle 採否判断 | — | 未確定 | — |

判定基準（§3）: パディング後にバンクコンフリクトメトリクスが有意に減少していること。TFLOPS 改善は付随的な確認事項（バンクコンフリクト削減が必ずしもエンドツーエンド TFLOPS へ線形に反映するとは限らないため、メトリクス自体を主判定とする）。

## 5. §1.2 parity 非後退契約の機械確認

```sh
git diff origin/main -- crates/backend-cuda/tests/parity_nonregression.rs crates/backend-cuda/tests/common/parity_baseline
```

無差分であることを確認する（§1「数値への影響」の裏付け。tolerance 定数・ベースライン fixture を変更していないことをコミット前に検査する）。本実装セッションでは無差分を確認済み。

## 6. リスクと安全側判断

- **実測なしの理論的コンフリクト解消**: パディングが実機で効果を持つ保証はないが、bit 一致論拠（§1）により正しさリスクはゼロ。効果測定と XOR swizzle 採否は本ファイルの手順・記録欄で実機セッションへ確実に引き継ぐ
- **NVRTC 未検証環境**: カーネル構造（パイプライン・命令列）は不変で、変更は配列次元と `#define` のみ。新規 PTX 構文なし → 構文リスク増加なし（#494 と同じ整理）
- **占有率への影響**: SMEM 増加（36,864B→41,472B）で SM あたり常駐ブロック数が変わりうるが、cc 8.6 概算では 2 ブロック/SM を維持（§1「占有率への影響」参照）。実機で確認する
- **ロールバック**: 定数 2 行（`MMA_A_PAD`/`MMA_B_PAD`）+ `#define` 2 行 + 配列宣言 2 行の差し戻しで即時ロールバック可能
