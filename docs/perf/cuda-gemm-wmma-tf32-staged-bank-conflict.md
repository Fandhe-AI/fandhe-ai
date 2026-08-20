# CUDA GEMM: TF32 opt-staged 経路の SMEM バンクコンフリクト対策（イシュー #743）

## 0. 背景

実機 ncu 計測（2026-08-19・DGX Spark GB10・sm_121）で TF32 opt-staged 経路
（`gemm_wmma_tf32_staged`。`crates/backend-cuda/src/kernels_wmma_opt.rs`）の
`l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` が以下の値を示した
（数値は Issue #743 記載値。本ドキュメント作成時点で生ログは未入手のため出典を
Issue 記載値と明記する）。

| M=N=K | ld.sum |
|------:|-------:|
| 2048  | 8.53M  |
| 4096  | 67.5M  |

同一計測で mma_f16 経路（`kernels_mma.rs::mma_f16_source`）は ld 38.3K・st 0
であり問題なし（本イシュー対象外）。

## 1. 定量解析

### 1.1 「非線形悪化」の再解釈

2048→4096 で FLOP 数（および SMEM ロード命令数）は N³ で 8 倍になる。
8.53M→67.5M（7.9 倍）はほぼ命令数に比例しており、「命令あたりの衝突率」は
一定である。したがって対策の目標は「命令あたりの衝突（余剰 wavefront）を
ゼロに近づける」ことであり、footprint（4 倍）との比較で「非線形に悪化した」
と捉えるのは誤りである。

### 1.2 衝突源の特定（理論モデル）

`wmma::load_matrix_sync` の実際の lowering は不透明だが、TF32 `m16n16k8` の
lane→(row,col) 対応が PTX ISA `mma.m16n8k8.tf32` のフラグメントレイアウト
（groupID = lane/4, thread-in-group = lane%4）に準ずると仮定し、row_major な
`as_tile[..][A_PAD]`/`bs_tile[..][B_PAD]` からの 1 回のフラグメントロード
命令が発行する 32 レーン分の SMEM アクセスを次のようにモデル化する（32
バンク・4B/バンク）。このモデルは
`crates/backend-cuda/src/kernels_wmma_opt.rs::wmma_tf32_staged_fragment_ld_wavefronts`
（および `wmma_tf32_staged_a_fragment_ld_wavefronts`/
`wmma_tf32_staged_b_fragment_ld_wavefronts`）として実装し、ロックテスト
（`wmma_tf32_staged_b_pad_72_is_bank_conflict_free_and_68_is_two_way`）で
値を固定してある。

- **A**（`as_tile[stage][row][A_PAD=20]`、row_major、1 命令で行 g=0..7・
  列 t=0..3）: `bank(g,t) = (20*g + t) mod 32` → {0-3, 20-23, 8-11, 28-31,
  16-19, 4-7, 24-27, 12-15} の 32 バンク全てが相異なる → **コンフリクトなし
  （wavefront=1）**
- **B**（`bs_tile[stage][k][B_PAD=68]`、row_major、1 命令で行 t=0..3・
  列 g=0..7）: `bank(t,g) = (68*t + g) mod 32 = (4*t + g) mod 32` →
  32 レーンが 20 バンクへ縮退 → **2-way バンクコンフリクト
  （wavefront=2、余剰 1）**

### 1.3 定量突合

4096 では block 数 4096（64×64 タイル・4096/64=64 ブロック/軸²）× k-tile
256（K_TILE=16 として 4096/16）× 4 warp/block × B フラグメントロード
（`ks`×`fj` の 2×2 = 4 回/タイル）× LDS 命令が K_GROUPS(2) に分割される
実装詳細を丸めた概算で、各命令に余剰 wavefront 1 が乗ると ≈67M 命令 ×
余剰 1 ≒ 67M 相当となり、実測 67.5M とほぼ一致する。2048 でも同様の比率で
≈8.4M ≒ 実測 8.53M。**B タイルの 2-way コンフリクトがほぼ全量**という
仮説が定量的に支持される。

### 1.4 対策候補と選定

B ストライド `S` が `S mod 32 ∈ {8, 24}` であれば `(S*t + g) mod 32`（t=0..3,
g=0..7）が 32 バンクを完全被覆しコンフリクトが理論上ゼロになる。

- 候補: `B_PAD = 64 + 8 = 72`（`BLOCK_N + 8`。272B→288B/行）
- A 側（`A_PAD = 20`）は既にコンフリクトフリーのため変更不要
- 制約確認: cp.async 16B 整列（pad が 4 の倍数）を満たす。WMMA
  `load_matrix_sync` の ldm（float は 4 の倍数）を満たす。
  `load_matrix_sync` のポインタ 32B 整列（行オフセット `8*S*4B`・
  `16*S*4B`、列オフセットは 64B 倍数、stage ベースは `64*S`B）を満たす

### 1.5 SMEM・occupancy への影響

`b_pad=72` の場合、1 段あたり A 5,120B + B `16*72*4=4,608B` = 9,728B
（既定 68 の 9,472B から +256B/段）。static 合計（stages=3）
= `3*9,728 + c_tile(16,384) = 29,184 + 16,384 = 45,568B ≤ 49,152B`
（48KiB 上限以内）。GB10 SMEM/SM 102,400B に対し常駐ブロック数
`floor(102400/45568)=2`（既定 `floor(102400/44800)=2` と同一）→
**occupancy は変化しない**（試算のみ。実機 ncu で最終確認する）。

`wmma_tf32_staged_dyn_smem_bytes`（`b_pad=72`）の回帰テスト
（`wmma_tf32_staged_dyn_smem_bytes_reflects_custom_b_pad`）は
`3*(64*20 + 16*72)*4 = 29,184B` を固定している。

### 1.6 エピローグ `c_tile` の store 側（対象外）

`wmma::store_matrix_sync` によるエピローグ書き込みは ld 指標に含まれず、
発行回数はブロックあたり数十命令程度（4096 で概算 ≈数十万件）で主ループの
ld 命令数（≈67M）に対し無視できる比率と見積もる。**本イシューでは
対象外**とし、実測 st.sum が有意であれば別イシューで `C_PAD` を検討する
（§4「スコープ外」）。

## 2. XOR swizzle 不採用の判断

`gemm_wmma_tf32_staged` は `nvcuda::wmma::load_matrix_sync(ptr, ldm)`
（線形 row_major + leading dimension を要求する不透明 API）で SMEM を
読む。CUTLASS 型の行パリティ×列ブロック XOR permute は「レーンごとに
任意アドレスを指定できる `ldmatrix` または手動 `LDS` + `mma.sync` PTX」が
前提であり、**WMMA API のまま適用すると `load_matrix_sync` の読み出し
対象が permute 前の線形レイアウトと食い違い数値が壊れる**。

適用するには TF32 経路を raw `mma.sync.m16n8k8.tf32` + 手動フラグメント
ロードへ全面移行する必要があり、本イシュー（アクセスパターン対策・4h
粒度）のスコープを大きく超える。したがって **本イシューではパディング
調整（§1.4）を採用し、XOR swizzle は「WMMA API 下では適用不可」として
不採用とする**。`mma.sync` への全面移行は別イシューの検討対象になりうる
（§4「スコープ外」。ユーザー承認なしに起票はしない）。

## 3. 計測手順（実機。DGX Spark GB10）

`docs/real-hardware-verification-env.md` の SSH/rsync 手順を前提とする。

```sh
git fetch origin
git checkout perf/743-wmma-tf32-staged-smem-padding

# 1) parity 非後退（数値一致を性能計測より先に確認する）
cargo test -p backend-cuda --release -- --ignored --nocapture

# 2) bit 一致 + TFLOPS（b_pad=68/72 双方を internal-diagnostics 経由の
#    render_wmma_tf32_staged_dyn 診断変種で計測し突合する。
#    gemm_wmma_tf32_staged_stages_bench.rs（#742）と同じ
#    WmmaTf32StagedKernelConfig { b_pad: N, ..default_tf32_staged() }
#    構築パターンを使う専用計測コードを実機セッション側で書く）

# 3) ncu（メトリクス名は --query-metrics で事前確認する。--b-pad は
#    gemm_profile_target 側で #743 のために追加した任意引数
#    〈--path wmma_tf32 限定〉。未指定〈既定〉は本番経路〈b_pad=68 固定・
#    static 共有メモリ〉、指定時は render_wmma_tf32_staged（**static**
#    共有メモリ変種。本番と同一の __shared__ 宣言・同一 occupancy）を
#    b_pad=N でコンパイル・起動する。PR #769 Bugbot 指摘 review id
#    4978031442 の是正: 当初は動的共有メモリ変種
#    〈render_wmma_tf32_staged_dyn。c_tile を as_tile/bs_tile へエイリアス
#    し約 29KiB・3 blocks/SM〉を使っており、3a〈本番・static・
#    44.8〜45.6KiB・2 blocks/SM〉との比較が b_pad の差と dyn/static の
#    occupancy 差を交絡していた。static 変種へ切替後は 3a/3b が b_pad の
#    みで差分化されるため、ld バンクコンフリクトの差分は occupancy 変化
#    を含まない）
cargo build -p backend-cuda --example gemm_profile_target --release \
    --features internal-diagnostics

# 3a) 既定（b_pad=68・本番経路・static 共有メモリ）
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,\
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,\
smsp__inst_executed_op_shared_ld.sum,\
sm__warps_active.avg.pct_of_peak_sustained_active \
    ./target/release/examples/gemm_profile_target --path wmma_tf32 --size 4096

# 3b) 候補（b_pad=72・static 共有メモリ変種。3a と __shared__
#     レイアウト／occupancy が同一のため b_pad の差分のみを計測する）
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,\
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,\
smsp__inst_executed_op_shared_ld.sum,\
sm__warps_active.avg.pct_of_peak_sustained_active \
    ./target/release/examples/gemm_profile_target --path wmma_tf32 --size 4096 \
    --b-pad 72

# 4) 採用時: REQ-8 下限余裕の確認
cargo run -p backend-cuda --example cuda_floor_bench --release
```

`.ncu-rep`・生ログ・実ホスト名はコミットしない（下記記録表への転記のみ）。

## 4. 記録欄（実機セッションで埋める）

| 項目 | b_pad=68（既定） | b_pad=72（候補） | 差分 |
|------|------------------|-------------------|------|
| M=N=K=2048 ld.sum | 8.53M（Issue 記載値） | 未計測 | 未計測 |
| M=N=K=4096 ld.sum | 67.5M（Issue 記載値） | 未計測 | 未計測 |
| st.sum（4096） | 未計測 | 未計測 | 未計測 |
| M=N=K=2048 TFLOPS（5 回中央値） | 未計測 | 未計測 | 未計測 |
| M=N=K=4096 TFLOPS（5 回中央値） | 未計測 | 未計測 | 未計測 |
| parity 非後退（全行 pass） | — | 未確認 | — |
| bit 一致（b_pad=68 変種との突合） | — | 未確認 | — |

### 採否基準

以下をすべて満たす場合のみ、`WMMA_TF32_STAGED_B_PAD`
（`crates/backend-cuda/src/kernels_wmma_opt.rs`）を `BLOCK_N + 8`
（72）へ変更する。

1. ncu で 4096 の ld バンクコンフリクトが有意に減少していること
2. TFLOPS が非劣化であること（2048・4096 いずれも 5 回計測中央値）
3. `parity_nonregression` の全行が pass すること
4. b_pad=68/72 の出力が bit 一致すること（パディングはアキュムレート順序
   を変えないため、この一致は理論的に成立するはずであり、不一致は実装
   バグの兆候として扱う）

満たさない場合は既定値 68 を維持し、本ドキュメントの記録欄へ「未計測」
または「棄却（理由）」を明記する（#497/#499/#741/#742 と同じ
「未計測の間は採用済みとして扱わない」判断）。

### ロールバック手順

採用した場合のロールバックは `WMMA_TF32_STAGED_B_PAD` 定数 1 行の書き戻し
のみで完結する（`crates/backend-cuda/src/kernels_wmma_opt.rs`。パディング
幅を `WmmaTf32StagedKernelConfig::a_pad`/`b_pad` フィールドへ config 化した
のはこのロールバックコストを最小化するため）。定数変更に追随して以下の
テスト期待値も更新が必要になる:

- `wmma_tf32_staged_constants_match_kernel_source_defines`
- `wmma_tf32_staged_dyn_smem_bytes_matches_expected_values`
- `validate_wmma_tf32_staged_config_accepts_default_and_rejects_smem_overflow`
  のコメント中の試算値

## 5. リスクと安全側判断

- **実測なしの理論的コンフリクト解消**: 本 PR は `WmmaTf32StagedKernelConfig`
  へのパディング config 化・検証・診断ヘルパの追加に留め、本番既定値
  （`WMMA_TF32_STAGED_B_PAD=68`）は変更しない。本番ディスパッチ経路
  （`render_wmma_tf32_staged`・`default_tf32_staged()`）の展開結果は
  byte 完全一致であることを
  `wmma_tf32_staged_default_config_render_is_byte_identical_to_production_source`
  が回帰検査する
- **理論モデルの限界**: `wmma::load_matrix_sync` の lowering は不透明な
  ため、§1.2 のバンクマッピングは ncu 実測との定量突合（§1.3）が支持する
  仮説にとどまる。実機 ncu 実測が最終的な正である
- **XOR swizzle**: §2 の理由により本イシューでは不採用。将来 `mma.sync`
  移行を検討する場合は別イシューとする

## 6. スコープ外（ユーザー承認を得てから起票する。自動起票しない）

- XOR swizzle の本格適用（TF32 経路の raw `mma.sync.m16n8k8.tf32` +
  手動フラグメントロードへの移行が前提）
- エピローグ `c_tile` の `C_PAD`（st 側コンフリクト。実機実測で有意なら
  別イシューで検討する）
- mma_f16 経路（ld 38.3K・st 0 で問題なし。本イシュー対象外）
