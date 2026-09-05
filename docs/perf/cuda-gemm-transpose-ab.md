# CUDA 転置カーネル（smem パディング + スウィズル）A/B 計測記録（#601）

イシュー #601「perf(backend-cuda): smem パディング + スウィズルによる転置カーネルを実装（GEMM epilogue
転置）」（親 #582 の G-11）の設計根拠・A/B 計測手順・記録テンプレート。`crates/backend-cuda/src/
kernels_transpose.rs` の転置カーネル群（naive・smem パディングのみ・smem パディング+スウィズル・GEMM
epilogue 融合）の効果を計測する。

## 1. 設計根拠

- **出自**: 親イシュー #582（Phase G）が TileLang の転置カーネル群から抽出したアルゴリズム（TileLang 自体
  には依存しない。整数式のみを NVRTC カーネルへ手動転写した独立実装）。
- **smem パディング**: 32×32 タイル（`kernels_transpose::TRANSPOSE_TILE`）を無パディングで確保すると、
  f32（4 バイト要素）は行幅が `32*4=128` バイト（ちょうど 32 バンク分）となり、転置ストア時の列方向
  アクセスで 32-way バンクコンフリクトが発生する。行幅を `ceil(4 / 要素バイト数)` 要素広げることで
  128 バイトの倍数から外し、行ごとにバンク位相をずらす（f32: `SMEM_PAD_F32=1`・f16: `SMEM_PAD_F16=2`。
  `kernels_transpose.rs` 内ドキュメンテーションコメント参照）。
- **dtype 依存スウィズル**: 「周期 = 8 / dtype バイト数」（f32: 2・f16: 4）を XOR ベースの列並べ替えとして
  実装する（`transpose::swizzled_smem_col` がホスト側の唯一の参照実装。全単射性は単体テスト
  `swizzle_is_bijective_per_row` で機械検査済み）。パディングのみでバンクコンフリクトが十分解消するのか、
  スウィズルの追加効果があるのかは実機計測でのみ判定できる（本イシューの主目的）。
- **GEMM epilogue 融合**: tiled GEMM（`kernels::TILED_F32` と同一のアキュムレーション）の epilogue で
  C タイルを smem 経由で転置ストアし、中間バッファ C を HBM に書かない（`kernels_transpose::
  TILED_TRANSPOSED_F32`）。積和ループ・FMA 契約は `TILED_F32` と完全同一のため、`run_tiled_f32` の出力を
  ホスト側で転置した結果と bit 完全一致することが期待される（数値契約。実機テスト
  `tests/transpose_parity.rs::tiled_transposed_f32_matches_host_transposed_tiled_and_cpu_reference` が
  `#[ignore]` で検証する）。

## 2. 状態: opt-in 実装のみ完了。未計測のまま本番カーネル・本番ディスパッチへ導入しない

`crates/backend-cuda/src/kernels_transpose.rs`（カーネルソース・ソース生成関数・needle テスト）・
`crates/backend-cuda/src/transpose.rs`（ホスト側スウィズル参照実装・`CudaTranspose` 起動 API）を実装した
が、**本番カーネル（`kernels.rs`・`kernels_mma.rs`・`kernels_wmma*.rs`）・本番ディスパッチ経路
（`ops.rs`／`gemm_auto.rs`）は 1 バイトも変更していない**。`CudaTranspose` は `CudaGemm` とは独立した
構造体であり、`CudaGemm::new` の eager コンパイル集合にも追加していない。

理由は #498（バンクコンフリクト対策）・#499（タイル→SM スウィズル）・#688（epilogue 融合）と同型の判断:
本実装セッションの実行環境（RTX 3060・compute capability 8.6・NVRTC 非搭載・`docs/real-hardware-
verification-env.local.md` 不在）では NVRTC コンパイル・実機実行・nsight-compute 計測がすべて到達不能
なため、コード実装（opt-in 経路・GPU 非依存の単体テストで検証可能な部分）のみをこの PR で完了し、実機
A/B 計測・nsight-compute 実測・採否確定は実機セッション（実機ツリー #408 系／G-12 #602）へ引き継ぐ。

**計測方式（advisor 指摘の是正）**: `examples/gemm_transpose_bench.rs` は当初 `run_*`（H2D/D2H 転送込み）
で 3 経路を直接比較していたが、size=4096 f32 では 1 回あたり 64MB 分の PCIe 転送（数 ms オーダー）が
カーネル本体の実行時間（数百 us オーダー）を支配し、naive/smem パディング/スウィズルの差がすべて転送
時間へ埋もれて比が常に約 1.00 に近い値しか観測できない欠陥があった（受け入れ基準「改善が実測で確認
できなければ不採用と記録して完了」に対し、この計測は偽の「不採用」判定を導きうる）。`gemm_mma_swizzle_
bench.rs` と同じ「H2D/D2H を計測区間の外へ出す」方針（`CudaTranspose::upload_f32`/`alloc_output_f32`/
`launch_naive_f32`/`launch_smem_f32`/`launch_tiled_transposed_f32`）へ改めた。現在の bench は 2 系統を
計測する（GB/s はいずれも **transpose カーネル本体（＋融合経路は tiled アキュムレーションを含む）のみ**
を対象とし、H2D/D2H を含まない）:

- **系統 A（transpose 単体）**: 同一デバイス常駐 `c_dev` に対する naive／smem(pad)／smem(pad+swizzle) の
  比較。パディング・スウィズルそのものの効果を GEMM 分を含めずに分離計測する。
- **系統 B（tiled+transpose 分離 vs 融合）**: `launch_tiled_f32`→`launch_naive_f32`（分離）と
  `launch_tiled_transposed_f32`（融合）を、両方ともデバイス常駐済みの同一 `a_dev`/`b_dev` に対して比較
  する。差分は「中間バッファ C の HBM 書き込み・再読み出しの有無」のみに絞られる。

**追記（イシュー #1214）**: 上記「未計測のまま本番カーネル・本番
ディスパッチへ導入しない」は、smem パディングのみ変種
（`transpose_smem_source_f32(false)`）に限り解消した。VJP 専用 NT/TN
転置入口（`crates/backend-cuda/src/gemm.rs::CudaGemm::run_tiled_f32_nt`／
`run_tiled_f32_tn`／`launch_tiled_f32_resident_nt`）へ、`CudaTranspose`
とは独立にロードした専用ハンドル（`CudaGemm::transpose_smem_f32`）として
結線した。GB10 実機 before/after は `docs/perf/cuda-gemm-vjp-transposed-
entry.md` を参照。**スウィズル変種・GEMM epilogue 融合転置
（`TILED_TRANSPOSED_F32`）は依然未計測のまま非結線**（本節の元の記述の
とおり）。

## 3. 計測手順（DGX Spark GB10・sm_121 実機。#498/#499 と同じ接続・転送手順）

接続・転送手順は `docs/real-hardware-verification-env.md` に従う（実ホスト名はローカル管理外ファイル
参照）。

```sh
git fetch origin
gh pr checkout <本イシューの実装 PR 番号>   # perf/601-cuda-smem-transpose
# PR 番号で明示する（`docs/perf/cuda-gemm-swizzle-ab.md` と同じ理由:
# ブランチ名固定だとマージ後にブランチが削除され checkout 不能になるため、
# `gh pr checkout` または `git fetch origin refs/pull/<N>/head && git checkout FETCH_HEAD`
# を使う。マージ後は `git checkout main` でも同じコードを指す）。

# 数値一致確認（TFLOPS/GB/s 比較より前に必須）。転置は演算を伴わない
# 純置換のため、複合誤差判定ではなく bit 完全一致で検証する。
cargo test -p fandhe-ai-backend-cuda --release --test transpose_parity -- --ignored --nocapture
# 既存回帰（本イシューはカーネル・tolerance 定数を変更していないことの
# 実機側再確認）。
cargo test -p fandhe-ai-backend-cuda --release --test parity_nonregression -- --ignored --nocapture

# A/B 計測（5 回計測中央値。3 経路 x 4 サイズ）。
cargo run -p fandhe-ai-backend-cuda --example gemm_transpose_bench --release
```

### nsight-compute バンクコンフリクト実測

`ncu` で以下のメトリクスを 4 対象（naive／smem パディングのみ／smem パディング+スウィズル／融合）に
ついて取得する:

```sh
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum \
    <bench バイナリ or 個別テストハーネス>
```

- `..._op_ld.sum`: 共有メモリ読み込み時のバンクコンフリクト回数
- `..._op_st.sum`: 共有メモリ書き込み時のバンクコンフリクト回数

## 4. 採否判定基準

- **採用**: smem パディングのみ変種が naive 比で明確な改善（目安: GB/s で 1.2 倍以上）を示し、かつ
  nsight-compute でバンクコンフリクトが有意に減少していること。スウィズル追加変種がパディングのみ変種
  からさらに改善する場合はスウィズルも含めて採用候補とする。**判定基準は size=2048/4096（帯域が支配的
  になるサイズ）の結果を主とする**: size=512 はカーネル起動オーバーヘッド（launch overhead）が全変種に
  共通してかかるため比が 1.00 付近へ圧縮されやすく、この安定化バイアスにより「改善なし」の偽判定を導き
  うる（advisor 指摘。系統 A・系統 B いずれも同様の注意が必要）。
- **不採用**: 改善が誤差範囲内（5 回計測の Q1/Q3 に基準点の中央値が収まる等）、またはバンクコンフリクト
  実測に有意差がない場合は「不採用」と記録して完了する（実装計画 8 節「改善が実測で確認できなければ
  採用しない判断を記録して完了」）。GEMM epilogue 融合変種についても同様に、中間バッファ削減の効果が
  実測で確認できない場合は不採用と記録する。
- 採否に関わらず、`ops.rs`／`gemm_auto.rs` への結線は本イシューのスコープ外（実装計画 8 節）であり、
  採用判断後の別イシューで扱う。

## 5. 記録テンプレート（実機セッションが追記する）

```
計測日: YYYY-MM-DD
実機: DGX Spark GB10（sm_121）/ その他
PR: #<番号>（perf/601-cuda-smem-transpose）
コミット: <SHA>

### transpose_parity（bit 完全一致）
naive f32/f16: PASS / FAIL
smem(pad) f32/f16: PASS / FAIL
smem(pad+swizzle) f32/f16: PASS / FAIL
fused (a) vs host-transposed tiled: PASS / FAIL
fused (b) vs CPU reference: PASS / FAIL

### 系統 A（transpose 単体・GB/s・5 回中央値）
| size | naive | smem(pad) | smem(pad+swizzle) | pad/naive | swizzle/naive |
|------|-------|-----------|--------------------|-----------|-----------------|
| 512  |       |           |                    |           |                 |
| 1024 |       |           |                    |           |                 |
| 2048 |       |           |                    |           |                 |
| 4096 |       |           |                    |           |                 |

### 系統 B（tiled+transpose 分離 vs 融合・秒・5 回中央値）
| size | separate | fused | fused/separate |
|------|----------|-------|-----------------|
| 512  |          |       |                 |
| 1024 |          |       |                 |
| 2048 |          |       |                 |
| 4096 |          |       |                 |

### nsight-compute バンクコンフリクト
| 対象 | ld.sum | st.sum |
|------|--------|--------|
| naive | | |
| smem(pad) | | |
| smem(pad+swizzle) | | |
| fused | | |

### 採否判断
（採用 / 不採用・根拠）
```
