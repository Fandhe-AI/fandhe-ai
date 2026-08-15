# CUDA GEMM M=N=K=4096 データ再利用崩壊のボトルネック診断（#486・Phase A A-6）

イシュー #486「test(backend-cuda): M=N=K=4096 のデータ再利用崩壊を定量診断するベンチ・プロファイルを追加」
（親 #480・ルート #479）の実測記録テンプレート。受け入れ基準「6 通り（2 path × 3 size）× 4 指標の記録と、
主因 → Phase B タスクの対応づけ」に対応する。

## 1. 目的と「29% 低下」の定義

`docs/perf/cuda-floor-remeasurement.md` の実測により、CUDA GEMM は M=N=K を 2048→4096 に上げると
TFLOPS が明確に低下している:

| 経路 | 2048 | 4096 | 変化率 | 対 PyTorch 比（2048→4096） |
|------|------|------|--------|---------------------------|
| `wmma_tf32`（f32 最良経路。WMMA(TF32) opt） | 6.2995 TFLOPS | 4.4824 TFLOPS | **約 −29%**（主対象） | 36.78% → 25.68% |
| `mma_f16`（f16 `mma.sync` パイプライン） | 12.0214 TFLOPS | 11.4462 TFLOPS | 約 −4.8%（副対象） | 対 PyTorch 比は 2048 側が最小 |

（出典: `docs/perf/cuda-floor-remeasurement.md`「経路×形状 TFLOPS 実測」表）

本ドキュメントは、この低下の**主因**（L2 ミス／SMEM バンクコンフリクト／occupancy／命令発行のいずれか）を
`nsight-compute`（ncu）実測で特定し、Phase B の優先順位（B-2 #493 レジスタブロッキング／B-3 #494 タイル
拡大／B-7 #498 バンクコンフリクト／B-8 #499 L2 スウィズル）を実測根拠で決めることを目的とする。

**カーネル本体（`kernels_wmma_opt.rs`／`kernels_mma.rs`）は本イシューで一切変更しない**（診断専用）。

## 2. 方法論: MFA occupancy 判定式の CUDA 読み替え

参照実装 MFA（`GEMMDescriptor.swift:255-321`）は「実測 occupancy（`actualGroups = ceil(M/タイル) ×
ceil(N/タイル)` vs `idealGroups = コア数 × 係数`）→ 閾値でタイル切替」という判定式でタイル形状を選ぶ。
本診断ではこれを CUDA 向けに以下のように読み替え、`gemm_profile_target.rs` 起動時の
`occupancy estimate:` 行で機械的に出力する（実測値は ncu の `sm__warps_active.avg.pct_of_peak_sustained_active`
と突合する）:

- `actual_blocks = ceil(M / BLOCK_M) × ceil(N / BLOCK_N)`
  - `wmma_tf32`: `BLOCK_M = BLOCK_N = 64`（`kernels_wmma_opt.rs::WMMA_TF32_OPT_BLOCK_M`/`_N`）
  - `mma_f16`: `BLOCK_M = 32`・`BLOCK_N = 64`（`kernels_mma.rs::MMA_BM`/`MMA_BN`）
- `sm_count`: `CudaDevice::context().attribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)`（実機実測値）
- `blocks_per_sm = actual_blocks / sm_count`（MFA の `actualGroups/idealGroups` 比に相当する「1 SM
  あたり何ブロックが割り当たるか」の概算）

`size=4096` は `size=2048` に対し `actual_blocks` が概ね 4 倍（`ceil` の端数を除き `(4096/BLOCK)^2 =
4 × (2048/BLOCK)^2`）になるため、SM 数が固定である以上 `blocks_per_sm` も概ね 4 倍に増える。ブロック数の
増加自体は必ずしも悪化要因ではない（並列度が上がる）が、L2 キャッシュ容量は M/N に対して不変であるため、
**タイル数が増えるほど A/B タイルの L2 再利用ヒット率が下がりやすい**という仮説が成り立つ。本診断はこの
仮説を L2 hit rate の実測で検証する。

## 3. 採取手順

### 3.1 事前確認（ノード側。fail-closed）

```sh
ssh "$CUDA_NODE" 'ncu --version'
# GPU カウンタ権限（ERR_NVGPUCTRPERM が出る場合は sudo 実行可否を確認）
ssh "$CUDA_NODE" 'ncu --query-metrics | grep -E "sm__warps_active|lts__t_sector_hit_rate|l1tex__data_bank_conflicts|dram__bytes|gpu__dram_throughput|sm__inst_issued"'
```

`--query-metrics` の出力でメトリクス名の存在を確認してから採取する（sm_121/Blackwell 世代でのリネームに
備える。名称が変わっていた場合は本節の採取コマンド・下記「4. 記録表」の指標名を実際の名称へ読み替えて
記録し、変更した旨をこのドキュメントに追記する）。

### 3.2 コード転送・ビルド

`docs/real-hardware-verification-env.md` §3・§4 の手順に厳密に従う（`export CUDA_NODE=...` → `.rev-stamp`
生成 → 規定の rsync → ノード側リビジョン確認）。

```sh
env PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH \
    CARGO_TARGET_DIR=$HOME/work/target-rust-ai-library \
    cargo build -p backend-cuda --example gemm_profile_target --release \
    --features internal-diagnostics
```

`--features internal-diagnostics` は必須（PR #637 codex-review 指摘の是正）。
`backend_cuda::diagnostics`（内部カーネルのタイル定数を返す診断専用関数群）は
既定ビルドの通常の公開 API 面から除外されており、この feature を有効化した
ビルドでのみ `diagnostics` モジュールが存在する（`crates/backend-cuda/src/lib.rs`
`diagnostics` モジュール冒頭コメント参照）。

### 3.3 ncu 採取（6 通り: 2 path × 3 size）

`--launch-skip <warmup 起動数> --launch-count <iters>` で `gemm_profile_target` 内の warmup を除いた
計測区間のみをプロファイルする。`gemm.alloc_output_f32`／`alloc_output_f16`（各 `Path` 分岐で warmup
ループ直前に 1 回呼ばれる）は cudarc `alloc_zeros` 経由でデバイス側ゼロクリアを行うが、内部で呼ぶのは
`cuMemsetD*Async` 系のドライバ API（`cuLaunchKernel` を経由しない別経路）であり、ncu がプロファイルする
「カーネル起動」の通し番号には含まれない。旧版はこれを「memset も 1 回のカーネル起動として数えられる」と
誤って前提し `--launch-skip` を `warmup + 1` にしていたため、warmup 回数分だけ計測対象カーネル起動を余分に
スキップしてしまい、`--launch-count 5` に対し実際に計測される起動が 4 回になり 6 条件すべての診断結果が
不完全・誤りになっていた（PR #637 codex-review 指摘。`gemm_profile_target.rs` の `ALLOC_ZEROS_LAUNCHES`
定義コメント参照。値は常に 0 とし `--launch-skip = warmup` とする）。既定 `--warmup 2 --iters 5` なら
`--launch-skip 2 --launch-count 5`。この値は手計算せず、`gemm_profile_target` 実行時に
`path=... warmup=... iters=...` の直後へ出力される `ncu --launch-skip <値> --launch-count <値>` 行を
そのまま使う。ただし ncu のカーネル起動カウント仕様は cudarc バージョン・実機の compute capability に
依存しうるため、この前提を鵜呑みにせず §3.3.1 の事前確認を必ず行う。

```sh
# `set -o pipefail`: `gemm_profile_target` が opt カーネル不在等で非 0 終了
# しても `| tee` の終了コードは tee 自身の 0 で上書きされてしまう
# （bash の pipefail 既定 off の挙動）。`gemm_profile_target` は opt カーネル
# 不在時に基本カーネルへ黙ってフォールバックせず非 0 終了するよう変更済み
# のため（PR #637 codex-review 指摘。`gemm_profile_target.rs` 該当コメント
# 参照）、本ループ側も pipefail を有効化して非 0 終了を確実に検知し、
# 誤ったカーネルの ncu 結果を「正常計測」として次サイズへ進めないように
# する。
set -o pipefail

BIN=$HOME/work/target-rust-ai-library/release/examples/gemm_profile_target
METRICS="sm__warps_active.avg.pct_of_peak_sustained_active,\
lts__t_sector_hit_rate.pct,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,\
dram__bytes.sum.per_second,\
gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,\
sm__inst_issued.avg.pct_of_peak_sustained_active"

for path in wmma_tf32 mma_f16; do
  for size in 1024 2048 4096; do
    # --launch-skip 2 = --warmup 2（既定）。alloc_zeros の memset は
    # cuLaunchKernel を経由しないため ncu の起動通し番号に含まれず、
    # 加算は不要（§3.3 冒頭の説明・`gemm_profile_target.rs` の
    # `ALLOC_ZEROS_LAUNCHES` 定義コメント参照）。
    if ! ncu --launch-skip 2 --launch-count 5 --metrics "$METRICS" \
        "$BIN" --path "$path" --size "$size" \
        2>&1 | tee "ncu-${path}-${size}.log"; then
      echo "abort: path=${path} size=${size} で gemm_profile_target または ncu が非 0 終了した" \
           "（opt カーネル不在等の異常系。ログ ncu-${path}-${size}.log を確認する）。" >&2
      exit 1
    fi
  done
done
```

`.ncu-rep` ファイル自体・生ログはコミットしない（秘密情報・内部ホスト名は含まないが、実測記録は本
ドキュメントの「4. 記録表」への転記を正とする。`.claude/rules/security.md` A01/A09）。

#### 3.3.1 `ALLOC_ZEROS_LAUNCHES = 0` 前提の実機検証（6 通り採取ループの前に 1 回だけ実施）

`--launch-skip` の算出（§3.3 冒頭）は「`alloc_zeros` の memset は `cuLaunchKernel` を経由しないため ncu
の起動通し番号に一切現れない（`ALLOC_ZEROS_LAUNCHES = 0`）」という `gemm_profile_target.rs` の前提に
依存する。この前提は cudarc のバージョン・実機の compute capability（sm_121/Blackwell 世代）に固有の
実装詳細であり、ずれると `--launch-skip` が過不足し、意図しないカーネル（memset 自体・対象外の起動）を
計測してしまう（PR #637 codex-review 指摘: 旧版は逆に `ALLOC_ZEROS_LAUNCHES = 1` と誤って前提しており、
同種の失敗の再発を防ぐための事前確認）。6 通りの本採取ループ（§3.3）を回す前に、いずれか 1 通り
（例: `wmma_tf32`／`1024`）で `--launch-skip` を指定せず `--launch-count` のみ絞って全カーネル名を一覧し、
想定どおりの並び（memset 系カーネルの起動は現れず、先頭から対象カーネルが `warmup + iters` 回連続）に
なっていることを目視確認する。

```sh
# --launch-skip なしで先頭数回分の起動を全部並べ、カーネル名の並びを目視確認する。
# `--launch-count` は `warmup + iters`（既定なら 2 + 5 = 7）以上を指定する
# （memset がカーネル起動として現れない想定のため、旧版の `1 +` は不要）。
ncu --launch-count 7 --print-kernel-base full \
    "$BIN" --path wmma_tf32 --size 1024 2>&1 | tee ncu-verify-launch-skip.log
```

想定どおり（memset 系カーネルの起動が現れず、先頭から対象カーネルが `warmup + iters` 回連続）であれば
§3.3 の 6 通りループへ進む。並びが想定と異なる場合（memset らしきカーネルが 1 回以上現れる、または対象
カーネル名が warmup 区間から既に異なる等）は、`ALLOC_ZEROS_LAUNCHES` の値・`gemm_profile_target.rs` の
`--launch-skip` 算出式を実機の実際のカーネル起動順に合わせて見直してから 6 通りループを実行する（閾値・
受け入れ基準の変更ではなく診断ツール自体の前提修正のため人間承認は不要だが、修正した場合はコミット・PR
本文にその旨を明記する）。

## 4. 記録表

環境: commit SHA=`<.rev-stamp の値>`／GPU driver=`<実測>`／CUDA=`<実測>`／ncu=`<ncu --version 実測>`

### 4.1 TFLOPS（`gemm_profile_target` の wall-clock 出力。ncu 実行中はオーバーヘッドが乗るため単体実行値と併記）

| path | size | TFLOPS（単体実行） | TFLOPS（ncu 実行中） |
|------|------|--------------------|------------------------|
| wmma_tf32 | 1024 | (未採取) | (未採取) |
| wmma_tf32 | 2048 | (未採取) | (未採取) |
| wmma_tf32 | 4096 | (未採取) | (未採取) |
| mma_f16 | 1024 | (未採取) | (未採取) |
| mma_f16 | 2048 | (未採取) | (未採取) |
| mma_f16 | 4096 | (未採取) | (未採取) |

### 4.2 診断指標

候補 D（命令発行律速）の判定に用いる `sm__inst_issued.avg.pct_of_peak_sustained_active`（instruction issue
rate。§2 採取コマンド参照）は生ログを非コミット運用とするため、本表の列へ転記した値のみが実測記録として
残る（`.claude/rules/security.md` A01/A09。上記「4. 記録表」冒頭注記と同じ理由）。

| path | size | achieved occupancy（%） | L2 hit rate（%） | SMEM bank conflicts（ld+st, sum） | DRAM throughput（%peak） | instruction issue rate（%peak） |
|------|------|--------------------------|--------------------|-------------------------------------|-----------------------------|-------------------------------------|
| wmma_tf32 | 1024 | (未採取) | (未採取) | (未採取) | (未採取) | (未採取) |
| wmma_tf32 | 2048 | (未採取) | (未採取) | (未採取) | (未採取) | (未採取) |
| wmma_tf32 | 4096 | (未採取) | (未採取) | (未採取) | (未採取) | (未採取) |
| mma_f16 | 1024 | (未採取) | (未採取) | (未採取) | (未採取) | (未採取) |
| mma_f16 | 2048 | (未採取) | (未採取) | (未採取) | (未採取) | (未採取) |
| mma_f16 | 4096 | (未採取) | (未採取) | (未採取) | (未採取) | (未採取) |

## 5. 分析・結論

（実機採取後に記入。サイズ増加で悪化する指標を特定し、主因 → Phase B タスクへの効きと優先順位を記す）

- 主因候補 A: L2 ヒット率の低下（→ B-8 #499 L2 スウィズル）
- 主因候補 B: SMEM バンクコンフリクトの増加（→ B-7 #498 バンクコンフリクト）
- 主因候補 C: occupancy の低下（→ B-2 #493 レジスタブロッキング／B-3 #494 タイル拡大）
- 主因候補 D: 命令発行律速（→ B-2 #493 レジスタブロッキング）

## 6. 制約（自動運転の安全側判断）

SSH 不達・ncu 不在・カウンタ権限なしで採取不能な指標がある場合、閾値・仕様は変更せず、採取できた指標の
みで暫定結論を記す。受け入れ基準未達分があれば以下に記録し、人間判断へ引き継ぐ。

- 2026-08-14 時点: 本イシューの実装セッションは DGX Spark GB10 実機への SSH アクセスを持たないため、
  §4 の記録表は未採取（テンプレートのみ）。実機採取は別セッション・別実行で行う。
- **本 PR（#486）の受け入れ範囲の明確化（2026-08-15）**: 上記の理由により本 PR 単体では冒頭に記載した
  受け入れ基準（6 通り × 4 指標の記録と主因 → Phase B 対応づけ）を満たさない。本 PR のスコープは
  診断基盤（`gemm_profile_target` example・ncu 採取手順・本ドキュメントの記録テンプレート）の追加までとし、
  §4 の実測記録・§5 の分析結論の完成は実機検証ツリー（ルート #408 系）側の後続作業に委ねる。本 PR は
  §4／§5 が空欄のままでも完了とみなす（PR #637 の codex-review スレッド解決コメントに記録された
  2026-08-15 付のスコープ縮小承認に基づく。本セッションではその承認記録を直接検証していない）。

## 7. スコープ外

- カーネル最適化そのもの（B-2〜B-8 の各イシューで実施）
- REQ-8 下限・tolerance・ガードレール閾値の変更（人間承認タスク）
- Metal／CPU 側の同種診断（A-7 #487／A-8 #488）
