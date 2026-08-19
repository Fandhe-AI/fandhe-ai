# CUDA GEMM M=N=K=4096 データ再利用崩壊のボトルネック診断（#486・Phase A A-6）

イシュー #486「test(backend-cuda): M=N=K=4096 のデータ再利用崩壊を定量診断するベンチ・プロファイルを追加」
（親 #480・ルート #479）の実測記録テンプレート。**#486 自身が定義する受け入れ基準**は「6 通り（2 path ×
3 size）× 4 指標の記録と、主因 → Phase B タスクの対応づけ」であり、本ドキュメントはこの基準に対する
記録テンプレートである。

**本ドキュメント・PR #637 が実際に完了させる範囲**（=このドキュメント自身が担う受け入れ基準）は、
診断基盤（`gemm_profile_target` example・§3 の ncu 採取手順・下記「4. 記録表」の記録テンプレート）の
整備までである。§4 の実測記入・§5 の分析結論・#486 が定義する受け入れ基準の完了は、実機（DGX Spark
GB10）アクセスを要するため後続イシュー #653 に分離した（詳細・追跡条件は「6. 制約」節を参照）。
このため PR #637 は `Closes #486` ではなく `Refs #486` として扱い、#486 は #653 の完了をもってクローズする。

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

**2026-08-19 実測での読み替え**: GB10（sm_121/Blackwell）では `dram__bytes.sum` /
`gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed` 等の `dram__*` 系メトリクスが存在しなかった
ため、DRAM throughput の**代替ではなく**、L2（LTS）を通過するトラフィックの参考指標として
`lts__t_bytes.sum.per_second` を用いた（本節冒頭の想定どおりリネームを確認したうえでの読み替え。
出典: イシュー #739）。**`lts__t_bytes.sum.per_second` は L2 ヒットによる転送も含む L2/LTS 通過
トラフィックであり、L2 を経由しない DRAM 直接転送のみを表す真の DRAM throughput とは異なる**
（codex-review 指摘。L2 ヒット率が変化すると L2/LTS 通過量も連動して変化するため、L2 ヒット率の
変化と切り離してこの値単独で DRAM 帯域律速を判定しない）。以降本ドキュメントでは列名・注記を
「L2/LTS throughput（DRAM 非代替）」と表記する。**`lts__t_bytes.sum`（絶対量。per-second でない値）は
未計測**であり、下記 §4.2 の当該列は per-second 値のみの記録である旨に注意すること。

**sudo の要否（codex-review 指摘・出典: イシュー #739）**: `ncu --version`／`ncu --query-metrics` は
GPU パフォーマンスカウンタへアクセスしない単なるメタデータ照会（インストール済み ncu のバージョン表示・
対応メトリクス名の列挙）であり、`RmProfilingAdminOnly` 制限の対象外のため bare `ncu`（非 sudo）で実行
できる。一方、実際にカーネルを起動してカウンタを採取する §3.3／§3.3.1 の採取コマンドは
`RmProfilingAdminOnly` の対象であり `sudo ncu` が必須（下記のとおり両節とも `sudo ncu` で統一済み）。
このため本節（事前確認）のみ bare `ncu`、実採取（§3.3／§3.3.1）は `sudo ncu` という使い分けは意図的な
ものであり、統一漏れではない。

```sh
ssh "$CUDA_NODE" 'ncu --version'
# GPU カウンタ権限（ERR_NVGPUCTRPERM が出る場合は sudo 実行可否を確認）
# `dram__bytes`／`gpu__dram_throughput` はここでは「§3.3 の METRICS には含めない
# 旧名が実機に存在しないこと」を確認する目的でのみ列挙する（§3.1 実測どおり
# GB10 では非該当がヒットしない想定。実採取対象は §3.3 の METRICS＝
# `lts__t_bytes.sum.per_second` に一本化済み）。
ssh "$CUDA_NODE" 'ncu --query-metrics | grep -E "sm__warps_active|lts__t_sector_hit_rate|lts__t_bytes|l1tex__data_bank_conflicts|dram__bytes|gpu__dram_throughput|sm__inst_issued"'
```

`--query-metrics` の出力でメトリクス名の存在を確認してから採取する（sm_121/Blackwell 世代でのリネームに
備える。名称が変わっていた場合は本節の採取コマンド・下記「4. 記録表」の指標名を実際の名称へ読み替えて
記録し、変更した旨をこのドキュメントに追記する）。

**fail-closed 化（codex-review 指摘: grep のみでは §3.3 METRICS の各メトリクスが個別に存在することを
保証できない。出典: イシュー #739）**: 上記の `grep -E` は複数パターンの**いずれか 1 つでも**マッチすれば
成功（exit 0）してしまうため、§3.3 METRICS が実際に採取する 6 メトリクスのうち一部が欠けていても検出
できない。6 通り採取ループ（§3.3）へ進む前に、以下で `$METRICS`（§3.3 で定義するのと同じ値）の各要素が
個別に存在することを検証し、1 つでも欠けていれば非 0 終了する:

```sh
# §3.3 の METRICS と同一の 6 メトリクスを完全名（rollup サフィックス込み）で
# 1 つずつ照合し、いずれか 1 つでも見つからなければ fail-closed で中断する
# （grep -E の「いずれか 1 つでもマッチすれば成功」という性質では個々の
# メトリクス欠落を検出できないため。codex-review 指摘・出典: イシュー #739）。
# `ncu --query-metrics`（引数なし）はベースメトリクス名のみを列挙し
# `.sum`／`.avg.pct_of_peak_sustained_active` 等のサフィックス組み合わせを
# 含まないため、完全名の照合には `--query-metrics-mode all`（サフィックス
# 展開済みの完全名を列挙するモード）を用いる。完全名を 1 行 1 件で厳密一致
# （`grep -qxF`）させることで、存在しないサフィックス組み合わせの指定も
# 検出できる（codex-review 指摘・出典: イシュー #739）。
# POSIX sh 互換（配列・here-string は bash 固有のため使わない。改行区切り文字列 +
# printf | grep で代替）。
REQUIRED_METRICS='
sm__warps_active.avg.pct_of_peak_sustained_active
lts__t_sector_hit_rate.pct
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum
lts__t_bytes.sum.per_second
sm__inst_issued.avg.pct_of_peak_sustained_active
'
AVAILABLE=$(ssh "$CUDA_NODE" 'ncu --query-metrics-mode all')
for m in $REQUIRED_METRICS; do
  if ! printf '%s\n' "$AVAILABLE" | grep -qxF -- "$m"; then
    echo "abort: required metric '$m' が ncu --query-metrics-mode all の出力に" \
         "完全名で見つからない。§3.3 の METRICS または本リストを実機の実際の名称へ" \
         "読み替えること。" >&2
    exit 1
  fi
done
```

**`--query-metrics-mode all` の実機対応可否（codex-review 指摘・出典: イシュー #739）**:
`--query-metrics-mode`（`base`／`suffix`／`all` を選べ、`all` はサフィックス展開済みの完全名を列挙する）
は ncu CLI の一般的なオプションだが、GB10 実機に導入済みの ncu バージョンでの対応は**本セッションでは
未確認**（実機アクセスなしのため）。§3.3.1 の事前確認と同様、6 通り採取ループへ進む前に一度
`ssh "$CUDA_NODE" 'ncu --query-metrics-mode all | head'` 等で当該オプションが認識されることを確認し、
非対応（`ncu` 自体がオプション未知エラーを返す）の場合は導入済み ncu のバージョンに応じた代替手段
（例: `ncu --query-metrics-mode suffix` で個別メトリクスのサフィックス一覧を取得し完全名を組み立てて
照合する）へ読み替えること。いずれの場合も、完全名の誤りに対する最終防御は §3.3 側にある:
`sudo ncu --metrics "$METRICS" ...` は `$METRICS` に無効な完全名が 1 つでも含まれていれば ncu 自身が
非 0 終了し、§3.3 の `if [ "$status" -ne 0 ]; then abort ...; fi` により fail-closed で検知される
（黙って一部メトリクスを無視して続行することはない）。本節（§3.1）の完全名チェックは「6 通り採取
ループに入る前の早期検知（無駄な採取往復の削減）」を目的とした一次防御であり、最終的な保証は §3.3 の
ncu 実行自体の exit code 検査（二次防御）が担う。

**ncu の GPU カウンタアクセス権限（2026-08-19 実測での運用確立）**: GB10 実機の ncu GPU カウンタは
`RmProfilingAdminOnly` 制限下にあり、`ERR_NVGPUCTRPERM` を避けるには sudo 実行が必要。実運用では
（プロファイリング用途向けに）sudo NOPASSWD が設定済みで、非対話 `sudo ncu` 実行が可能な状態が確立されて
いる（`docs/real-hardware-verification-env.md` §6.2 の追記も参照。出典: イシュー #739）。

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
# `gemm_profile_target` は opt カーネル不在等の異常系で基本カーネルへ黙って
# フォールバックせず非 0 終了するよう変更済み（PR #637 codex-review 指摘。
# `gemm_profile_target.rs` 該当コメント参照）。この非 0 終了を本ループ側で
# 確実に検知する必要があるが、`if ! CMD 2>&1 | tee LOG; then` という
# パイプ構成は既定のシェル設定下では末尾の `tee` 自身の終了ステータス
# （常に 0 側に倒れやすい）を判定してしまい、`CMD`（ncu／対象バイナリ）が
# 非 0 終了しても検知できないまま次の採取へ進みかねない（`set -o pipefail`
# は bash 固有でありコードブロックの ```sh 表記〈POSIX sh 実行を含みうる〉
# 下では有効化が保証されない。codex-review 指摘・出典: イシュー #739）。
# このためパイプ経由の `tee` には依存せず、`ncu` の標準出力・標準エラーを
# 直接ログファイルへリダイレクトしたうえで `$?` を明示的に検査し、検査後に
# ログを表示する構成へ変更した（シェル実装・pipefail 設定に依存しない
# fail-closed 判定）。

BIN=$HOME/work/target-rust-ai-library/release/examples/gemm_profile_target
METRICS="sm__warps_active.avg.pct_of_peak_sustained_active,\
lts__t_sector_hit_rate.pct,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,\
lts__t_bytes.sum.per_second,\
sm__inst_issued.avg.pct_of_peak_sustained_active"
# `dram__bytes.sum.per_second` / `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed`
# はいずれも GB10（sm_121/Blackwell）に存在しない（§3.1 実測）ため METRICS
# から除外し、L2/LTS throughput（DRAM 非代替。§3.1 参照）の参考指標として
# `lts__t_bytes.sum.per_second` に一本化する（未対応メトリクス指定は ncu が
# fail-closed で拒否するため、実採取前に §3.1 の `--query-metrics` 確認を
# 必ず通す。出典: イシュー #739）。

for path in wmma_tf32 mma_f16; do
  for size in 1024 2048 4096; do
    # --launch-skip 2 = --warmup 2（既定）。alloc_zeros の memset は
    # cuLaunchKernel を経由しないため ncu の起動通し番号に含まれず、
    # 加算は不要（§3.3 冒頭の説明・`gemm_profile_target.rs` の
    # `ALLOC_ZEROS_LAUNCHES` 定義コメント参照）。
    # §3.1 のとおり GPU カウンタは RmProfilingAdminOnly 制限下にあり、
    # sudo NOPASSWD 運用が確立済みのため sudo 経由で起動する（bare ncu では
    # ERR_NVGPUCTRPERM になる。出典: イシュー #739）。
    LOG="ncu-${path}-${size}.log"
    sudo ncu --launch-skip 2 --launch-count 5 --metrics "$METRICS" \
        "$BIN" --path "$path" --size "$size" \
        > "$LOG" 2>&1
    status=$?
    cat "$LOG"
    if [ "$status" -ne 0 ]; then
      echo "abort: path=${path} size=${size} で gemm_profile_target または ncu が非 0 終了した" \
           "（opt カーネル不在等の異常系。ログ ${LOG} を確認する）。" >&2
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
# §3.3 と同じ理由（RmProfilingAdminOnly）で sudo 経由とする。
sudo ncu --launch-count 7 --print-kernel-base full \
    "$BIN" --path wmma_tf32 --size 1024 2>&1 | tee ncu-verify-launch-skip.log
```

想定どおり（memset 系カーネルの起動が現れず、先頭から対象カーネルが `warmup + iters` 回連続）であれば
§3.3 の 6 通りループへ進む。並びが想定と異なる場合（memset らしきカーネルが 1 回以上現れる、または対象
カーネル名が warmup 区間から既に異なる等）は、`ALLOC_ZEROS_LAUNCHES` の値・`gemm_profile_target.rs` の
`--launch-skip` 算出式を実機の実際のカーネル起動順に合わせて見直してから 6 通りループを実行する（閾値・
受け入れ基準の変更ではなく診断ツール自体の前提修正のため人間承認は不要だが、修正した場合はコミット・PR
本文にその旨を明記する）。

## 4. 記録表

環境: commit SHA=`cbc16e7`／計測日 2026-08-19／GPU driver・CUDA・ncu 版は `docs/real-hardware-verification-env.md`
§2.1 の既測値を参照（本イシュー〈#739〉本文には個別値の記載が無いため独自に埋めない）。

**`--launch-count 5` の性質に関する注記**: 本節の ncu 採取コマンド（§3.3）が指定する
`--launch-count 5` は §3.3 の説明どおり「連続 5 起動を対象とする 1 回のプロファイル実行」であり、
`.claude/rules/coding-rust.md` の「ベンチは 5 回計測の中央値を採用する」が要求する**独立した 5 回の
計測（プロセス起動を分けた再計測）ではない**。下記表の値は連続 5 起動プロファイルの結果である旨に
注意すること（出典: イシュー #739）。

### 4.1 TFLOPS（`gemm_profile_target` の wall-clock 出力。ncu 実行中はオーバーヘッドが乗るため単体実行値と併記）

| path | size | TFLOPS（単体実行） | TFLOPS（ncu 実行中） |
|------|------|--------------------|------------------------|
| wmma_tf32 | 1024 | (未採取) | (未採取) |
| wmma_tf32 | 2048 | (未採取) | (未採取) |
| wmma_tf32 | 4096 | (未採取) | (未採取) |
| mma_f16 | 1024 | (未採取) | (未採取) |
| mma_f16 | 2048 | (未採取) | (未採取) |
| mma_f16 | 4096 | (未採取) | (未採取) |

**2026-08-19 時点の注記**: イシュー #739 本文には本表（単体実行／ncu 実行中の TFLOPS）に対応する値の
記載が無いため「(未採取)」のまま残す。下記 §4.2 の診断指標（occupancy・L2 hit rate・SMEM bank
conflicts）のみが実測記入済み。

### 4.2 診断指標

候補 D（命令発行律速）の判定に用いる `sm__inst_issued.avg.pct_of_peak_sustained_active`（instruction issue
rate。§2 採取コマンド参照）は生ログを非コミット運用とするため、本表の列へ転記した値のみが実測記録として
残る（`.claude/rules/security.md` A01/A09。上記「4. 記録表」冒頭注記と同じ理由）。

**2026-08-19 実測**（出典: イシュー #739。1024 行は転記元に値が無いため「(未採取)」のまま残す）:

| path | size | achieved occupancy（%） | L2 hit rate（%） | SMEM bank conflicts（ld, sum） | L2/LTS throughput（bytes/s、DRAM 非代替） | instruction issue rate（%peak） |
|------|------|--------------------------|--------------------|-------------------------------------|-----------------------------|-------------------------------------|
| wmma_tf32 | 1024 | (未採取) | (未採取) | (未採取) | (未採取) | (未採取) |
| wmma_tf32 | 2048 | (未採取) | 96.77 | 8.53M | (未採取) | (未採取) |
| wmma_tf32 | 4096 | 16.6（脚注 1） | 76.51 | 67.5M | (未採取) | サイズ増で低下（脚注 2） |
| mma_f16 | 1024 | (未採取) | (未採取) | (未採取) | (未採取) | (未採取) |
| mma_f16 | 2048 | (未採取) | 96.92 | 10.9K（st=0） | (未採取) | (未採取) |
| mma_f16 | 4096 | 64（脚注 1） | 83.51 | 38.3K（st=0） | (未採取) | サイズ増で低下（脚注 2） |

- **脚注 1（occupancy）**: イシュー #739 本文に occupancy 値のサイズ別内訳の明記が無く、4096 での
  データ再利用崩壊診断（本ドキュメント §1）の文脈値として記載されたもの。よって上表では 4096 行に
  記入し、2048/1024 行は「(未採取)」のまま残す。
- **脚注 2（instruction issue rate）**: 具体的な数値はイシュー本文に記載が無く「サイズ増で低下」という
  定性的な記録に留まる（生ログ非コミット運用のため §4.2 冒頭注記のとおり本表へ数値転記できるのは
  イシューに明記された値のみ）。
- **SMEM bank conflicts の ld/st 区別**: mma_f16 は st=0 であることがイシュー本文に明記されている
  （出典: イシュー #743）。wmma_tf32 の st は転記元に個別値の記載が無いため、上表は ld（sum）のみを
  記録する。
- L2/LTS throughput 列は §3.1 のとおり `lts__t_bytes.sum.per_second`（L2 ヒット含む L2/LTS 通過
  トラフィックであり DRAM throughput の代替ではない）を記録する列だが、イシュー本文に個別数値の
  記載が無いため「(未採取)」のまま残す。

## 5. 分析・結論（2026-08-19 実測に基づく確定。出典: イシュー #739・#736・#740〜#743）

上記 §4.2 の 2048→4096 実測から、**wmma_tf32 の 4096 崩壊は単一要因ではなく以下の複合**と判断する:

- **(A) L2 ヒット率の崩壊**（96.77% → 76.51%）: 主因候補 A に該当。→ B-8／新ツリー #736 配下の
  L2 スウィズル対応（#741）へつなぐ
- **(B) SMEM バンクコンフリクト（ld: 8.53M → 67.5M。約 7.91 倍）**: GEMM の総仕事量（`2 × M × N × K`
  FLOP）は M=N=K を 2048→4096 にすると 8 倍（`(4096/2048)^3 = 8`）増えるため、タイル数比の 4 倍
  （〈本ドキュメント §2〉）ではなく**総仕事量比 8 倍を分母に正規化して比較する必要がある**（codex-review
  指摘。旧記述の「タイル数比 4 倍を上回る」は分母の選定が誤りだった）。実測の約 7.91 倍は総仕事量比
  8 倍とほぼ同率（7.91/8 ≈ 0.989）であり、**単位仕事量あたりのバンクコンフリクト発生率はほぼ横ばい**
  で「非線形悪化」とは言えない。よって (B) 単独をサイズ増に対する非線形な主因として確定はしない
  （バンクコンフリクトの絶対件数自体が大きいこと自体は事実であり、B-7／#743（バンクコンフリクト対策）
  は引き続き着手対象とするが、その根拠は「非線形悪化」ではなく「絶対件数が大きい」点に修正する）
- **(C) 低 occupancy**（4096 で 16.6%）: 上表 §4.2 のとおり 2048 側の occupancy は「(未採取)」であり、
  2048→4096 の**変化**（低下したかどうか）は実測比較できていない。確定できるのは「4096 単体の値が
  16.6% と低い」という絶対水準のみであり、これをサイズ増に伴う低下の**確定できる主因**として結論する
  ことはできない（codex-review 指摘。比較対象がない一点のみでの因果確定を避ける）。主因**候補** C として
  扱い、2048 側の occupancy 実測（再実機セッション）を待って確定判断する。→ B-2／#742（パイプライン
  段数スイープ）へつなぐ

**(A) L2 ヒット率の崩壊は現時点で最有力の主因候補だが、「実測から確定できる主因」とまでは言えない**
（codex-review 指摘。出典: イシュー #739）。L2 hit rate の低下（96.77% → 76.51%）と TFLOPS 低下との間に
相関は実測できているが、L2/LTS throughput は §4.2 のとおり未採取、instruction issue rate は「サイズ増で
低下」という定性記録に留まり数値化されておらず、occupancy も 2048 側が未採取であるため、他の候補
（(B)・(C)・命令発行律速）を対照実験や定量比較で切り分けて因果・寄与度を分離できていない。よって (A) は
確定した主因ではなく他候補より確度の高い**主因候補**として扱う。確定には L2/LTS throughput の実採取・
instruction issue rate の数値化・2048/4096 両サイズの occupancy 実測比較・L2 挙動のみを変える対照実験の
いずれかが必要である。B-8／新ツリー #736 配下の L2 スウィズル対応（#741）は現時点の最有力候補としての
優先着手対象に変わりないが、確定した主因としての結論づけは上記の追加実測後に行う。(C) 低 occupancy は
4096 単体の絶対値が低いことは実測済みだが、2048 側が未採取のため「サイズ増に伴う低下」としては確定せず
主因候補にとどめる（上記 (C) 参照）。(B) は「サイズ増に対する非線形悪化」としては確定しない（上記正規化
の結果、総仕事量に比例した増加とほぼ整合するため）。B-7／#743 は (B) の絶対件数の大きさを理由に着手対象
として残すが、優先順位づけの根拠を「非線形悪化」から修正する。B-2／#742 は (C) の確定判断待ちの候補として
着手対象に残す。

**mma_f16 の 4096 側の低下（軽微・約 −4.8%。本ドキュメント §1）も L2 ヒット率の低下**
（96.92% → 83.51%）との相関が最も目立つが、上記 wmma_tf32 と同じ理由（L2/LTS throughput 未採取・
instruction issue rate 未数値化・occupancy 2048 側未採取）により因果・寄与度を分離できていないため、
確定した主因ではなく**主因候補**の 1 つとして扱う（codex-review 指摘・出典: イシュー #739）。SMEM バンク
コンフリクト（ld: 10.9K → 38.3K・st は両サイズとも 0）・occupancy（64%）は wmma_tf32 ほど深刻ではないと
いう相対的な傾向は実測されているが、これらも同様に確定的な除外根拠ではない。この診断に基づき、mma_f16
側の L2 スウィズル（swizzle
A/B・#499・後述 §6 の兄弟ドキュメント `cuda-gemm-swizzle-ab.md`）は **4096 で ×1.5957 の実測改善値
自体は確立済み**だが、`cuda-gemm-swizzle-ab.md` §4 の既存判断基準（2048・4096 両方の改善が必要）には
2048 が未達のため**不採用が確定**しており、本番結線は行っていない（4096 限定基準への変更はユーザー
承認が必要。詳細は `docs/perf/cuda-gemm-swizzle-ab.md` §2・§4・§6）。

候補 D（命令発行律速）は §4.2 の脚注 2 のとおり具体的な数値がイシュー本文に無く定性記録（サイズ増で
低下）に留まるため、単独の主因としては確定しない。

**旧 Phase B 対応表（B-2／B-7／B-8）と新ツリー #736（#740〜#743）の対応づけ**:

| 旧 Phase B タスク | 内容 | 新ツリー #736 配下 |
|---|---|---|
| B-8（#499） | L2 スウィズル | #740（mma_f16。現行基準では不採用確定・承認後の基準改定検討）・#741（TF32 swizzle） |
| B-7（#498） | バンクコンフリクト対策 | #743 |
| B-2（#493） | レジスタブロッキング／パイプライン段数 | #742（段数スイープ） |
| B-3（#494） | タイル拡大 | #742 と関連（未分離） |

## 6. 制約（自動運転の安全側判断）

SSH 不達・ncu 不在・カウンタ権限なしで採取不能な指標がある場合、閾値・仕様は変更せず、採取できた指標の
みで暫定結論を記す。受け入れ基準未達分があれば以下に記録し、人間判断へ引き継ぐ。

- 2026-08-14 時点: 本イシューの実装セッションは DGX Spark GB10 実機への SSH アクセスを持たないため、
  §4 の記録表は未採取（テンプレートのみ）。実機採取は別セッション・別実行で行う。
- **本 PR（#486）の受け入れ範囲の明確化（2026-08-15 確定）**: #486 自身が定義する受け入れ基準
  （6 通り × 4 指標の記録と主因 → Phase B 対応づけ）は実機（DGX Spark GB10）でのみ実行可能であり、
  本 PR のセッションはそのアクセスを持たない。そのため本 PR は #486 をクローズせず（`Closes #486` では
  なく `Refs #486`）、#486 が定義する受け入れ基準の完了を後続イシュー #653
  （<https://github.com/Fandhe-AI/rust-ai-library/issues/653>・親 #480 配下の sub-issue）に分離した。
  本 PR（診断基盤の整備）のスコープは以下に限定する:
  - `gemm_profile_target` example（診断専用バイナリ）
  - §3 の ncu 採取手順（事前検証・転送・6 通りループ・`ALLOC_ZEROS_LAUNCHES` 前提の実機検証手順を含む）
  - 本ドキュメントの §4 記録表・§5 分析候補の**テンプレート自体**（値の記入は対象外）

  §4／§5 の実測記入・結論確定・Phase B タスクの優先順位更新は、上記完了条件を持つ #653 が担う。
  以前このドキュメントは review thread `PRRT_kwDOTuUCJc6Zf28l` への resolve コメント（スコープ縮小の
  ユーザー承認を主張する未検証の記述）のみを根拠にしていたが、後続の codex-review
  （thread `PRRT_kwDOTuUCJc6ZggWh`）で「差分から確認できない」と指摘されたため、根拠を diff から検証
  可能な形（#486 自身の受け入れ基準テキストとの突合・追跡先イシュー #653 の実在）へ差し替えた。
- **2026-08-15 時点（#653 実装セッション）**: G0（実行可否ゲート）判定のため
  `docs/real-hardware-verification-env.local.md`（メイン working copy `docs/` 配下・Git 管理外）の
  存在をメイン working copy の絶対パスで確認したが、**同ファイルは存在しなかった**
  （`docs/real-hardware-verification-env.local.md.example` テンプレートのみ存在）。`CUDA_NODE`
  実値が取得できず DGX Spark GB10 実機への SSH 接続を検証できないため、fail-closed 判定により
  実測フェーズ（§3.3.1 の事前検証・§3.3 の 6 通り ncu 採取・§4 記録表の記入・§5 の主因分析）は
  **未実施**のまま本追記のみに留める（値を捏造しない）。§4／§5 は引き続き未採取のテンプレートの
  ままであり、#653 の受け入れ基準（6 通り × 4 指標の記録・主因 → Phase B 対応づけ・Phase B
  イシューへの優先順位コメント）は本セッションでは完了しない。#653 はクローズせず、実機
  （DGX Spark GB10）アクセスを持つ後続セッション・人間判断へ引き継ぐ。
- **2026-08-19 時点（イシュー #739 実装セッション）**: 上記 §4.2 の診断指標（occupancy・L2 hit rate・
  SMEM bank conflicts）を実測値へ更新し、§5 の分析・結論を確定させた。ただし §4.1 の TFLOPS 表と
  §4.2 の 1024 行はイシュー本文に値の記載が無く「(未採取)」のまま残っており、**#653 が定義する受け入れ
  基準（6 通り × 4 指標の完全記録）は本セッションでも完全充足ではない**。よって本 PR は `Closes #653`
  ではなく `Refs #653` として扱い、#653 のクローズは 1024 行・TFLOPS 表の採取後に行う。

### 3.1 事前確認の運用確立（2026-08-19 追記）

`ERR_NVGPUCTRPERM` への対処は実運用で「`RmProfilingAdminOnly` 制限下で sudo ncu 実行（NOPASSWD 設定
済み）」として確立した（§3.1 冒頭の追記・`docs/real-hardware-verification-env.md` §6.2 の追記と同一
事実。出典: イシュー #739）。

## 7. スコープ外

- カーネル最適化そのもの（B-2〜B-8 の各イシューで実施）
- REQ-8 下限・tolerance・ガードレール閾値の変更（人間承認タスク）
- Metal／CPU 側の同種診断（A-7 #487／A-8 #488）
