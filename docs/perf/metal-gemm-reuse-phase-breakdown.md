# Metal GEMM reuse 計測境界のフェーズ分解（upload／encode／commit_wait／readback。イシュー #1189）

## §1 位置づけ

`#1147`（`docs/perf/metal-gemm-candle-gate-remeasurement.md` §4.1〜4.3）は
Metal GEMM reuse（N=1024/2048/4096）が candle fresh 比 0.589〜0.726 倍で
未達であることを確定したが、`#1036`/`#1103`（`docs/perf/
metal-gemm-bottleneck-rediagnosis.md` §4.1〜4.3・§7.1b）は「転送（アップ
ロード＋readback）＋同期」が reuse 計測境界の end-to-end 時間の約 38〜46%
（N=4096）を占めることまでは確認したが、candle 比未達の主因を転送と確定
できず、`bench-fandhe` の reuse 1 反復（`matmul`／`to_tensor`／
`host_copy`／`checksum`）のどこに固定費が乗っているかは未分解のまま
だった。

本イシューは CUDA 側 `#1182`（`docs/perf/cuda-gemm-reuse-phase-
breakdown.md`）と同じ 2 層計測（Layer A: 公開 API 境界のフェーズ分解、
Layer B: 非公開 API での `matmul` 内側分解）を Metal（M4 Max 実機）で
実施し、N=1024/2048/4096 の transfer／sync／kernel 内訳を実測確定する。

**結論の先出し**: CUDA（#1182 §6）は「主因はハーネス自身の
`host_copy`＋`checksum`（`matmul` を除いた固定費）であり、`matmul` 単体は
candle fresh を上回る」という非対称な結論だったが、**Metal では非対称の
向きが異なる**。N=1024/2048 では `matmul` 単体が candle fresh とほぼ同等
（0.90〜0.99 倍）まで縮まるが、**N=4096 では `matmul` 単体が candle
fresh より 1.52 倍遅い**——すなわち Metal の N=4096 ギャップはハーネス
測定境界の産物ではなく、GPU 実行（アップロード・カーネル・readback を
含む）自体に起因する（§6〜§7 参照）。

**追記（イシュー #1277）**: `#1276` が実装した GPU タイムスタンプ変種
`kernel_gpu` を N=1024/2048/4096 で 5 プロセス起動計測し、N=4096 の
candle 比ギャップ（1.52 倍）のうちカーネル専有時間に帰属できる分は
約 27.6%、残り約 72.4% は非カーネル要因（`commit_wait − kernel_gpu`・
アップロード・readback 等）であることを確定した（§11）。

**追記（イシュー #3182）**: Mac mini（Apple M4）で同じ窓の内訳（upload・commit_wait・
kernel_gpu・readback・host_copy・checksum）を N=256〜4096 で取り直した（§12。0.11.0＋HEAD）。
N=4096 の candle との差のうちカーネル（＋同期）由来は 65.5〜72.2%、fandhe 固有の
upload＋host_copy は 32.6%。M4 Max・0.6.0 の §3〜§11 とは別機体・別版の値。

## §2 計測環境・プロトコル

- 実機: Apple M4 Max（GPU 40 コア。`docs/perf/logs/metal-gemm-reuse-
  phase-1189/env_info.txt`）
- rustc/cargo 1.96.0、転送元コミット `fff8904`（origin/main）
- framework-compare ピン `fandhe-ai =0.6.0`（crates.io 公開版。ルート・
  framework-compare 双方の `Cargo.lock` 無変更）
- `pmset -g therm` は計測前後とも「記録なし」でサーマルスロットリングの
  兆候なし
- 対象サイズ N=1024/2048/4096（いずれも 8 の倍数のため
  `pad::pad_matrix` は `Cow::Borrowed` の no-op。`pad_a`／`pad_b`／
  `unpad` は計測対象に含めない）

### Metal における transfer／sync／kernel の定義（CUDA と 1:1 ではない）

本番経路 `facade Var::matmul` → `MetalBackendOps::gemm`（`ops.rs`）→
`MetalGemm::dispatch_auto`（`gemm.rs`）→ `dispatch_variant` の 1 反復は
次の区間へ分解できる。Apple Silicon は UMA（統合メモリ）のため、CUDA の
H2D/D2H に相当する区間は明示転送ではなく **統合メモリ上の memcpy** で
ある点が最大の違い:

| 区間名 | 実体 | CUDA 対応区間 |
| --- | --- | --- |
| `upload_a`／`upload_b` | `MetalBuffer::new_with_data`（`newBufferWithBytes_length_options`。統合メモリへの memcpy） | H2D A／B |
| `alloc_c` | `MetalBuffer::alloc_uninit_pooled`（プール経由） | C 確保（プール経由） |
| `encode` | `MetalContext::encode`（`pub(crate)`。エンコーダ生成＋バッファ結線＋ディスパッチ記録。commit しない） | launch_issue |
| `commit_wait` | `MetalContext::synchronize`（`flush_locked`＝`endEncoding`＋`commit` → `waitUntilCompleted`） | kernel_wait（ただし commit・カーネル専有・wait の合算。§8 参照） |
| `readback` | `MetalBuffer::read_to_vec`（`contents()` からの memcpy。D2H 固有の同期は存在しない） | D2H |
| `host_copy` | `Vec::to_vec()` | ホストコピー |

`crates/backend-metal/src/gemm_reuse_phase_diag_tests.rs` 冒頭コメントに
同じ表と実装判断の詳細を記す。

### Layer A（公開 API 境界。コード変更なし）

`scripts/bench/framework-compare` にて
`cargo run --release -p bench-fandhe -- --task gemm --device metal --size {1024,2048,4096} --mode reuse --phases`
を N ごとに 5 回（各 20 warmup＋20 測定の中央値）実行した。生 JSONL は
`docs/perf/logs/metal-gemm-reuse-phase-1189/layerA-phases.jsonl`
（= `scripts/bench/framework-compare/results/raw/
results-m4max-gemm-phases-0.6.0.jsonl`）。

**AC-2 確認**（既存経路・既存テスト無変更の裏付け）: 同一セッションで
`--phases` なしの `gemm --mode reuse` を N ごとに 1 回実行し
（`layerA-ac2.jsonl`）、`checksum` が phases 版と **全 N で bit 一致**・
`parity_fail_count=0`・JSONL キー集合不変を確認した。`git diff --stat`
で `scripts/bench/framework-compare/bench-fandhe/src/main.rs` に diff が
ないことも確認済み（§4 参照）。

### Layer B（`crates/backend-metal` 非公開 API。HEAD ツリー）

新規 `crates/backend-metal/src/gemm_reuse_phase_diag_tests.rs`
（`#[cfg(all(test, target_os = "macos"))]`）が `gemm.rs` へ新設した
`#[cfg(test)] pub(crate)` ヘルパ `MetalGemm::diag_encode_tiled_nn`
（`dispatch_tiled_prepared` と同じ NN 経路だが `encode` と
`synchronize` を個別に計時できるよう分離したもの）を通じて 1 反復を
upload_a／upload_b／alloc_c／encode／commit_wait／readback／host_copy
へ分解計測する。20 warmup＋20 測定を 1 run とし、テストバイナリを 5 回
起動して中央値の中央値を採った（CUDA #1182 と同一プロトコル）。

**GPU タイムスタンプ変種（`kernel_gpu`）を #1276 で実装済み**: 本文書
執筆時点（#1189）は `MetalContext::encode`/`synchronize` のバッチング
機構（イシュー #1017）がコマンドバッファを内部に閉じ込めていることを
理由に、`MTLCommandBuffer::GPUStartTime`/`GPUEndTime` で `commit_wait`
から純カーネル専有時間を分離する変種を見送っていた。イシュー #1276 は
自前のコマンドキュー経路を新設するのではなく、`MetalContext::
synchronize` の内部（完了バッチを `waitUntilCompleted` した直後・drop
する前）へオブザーバを差し込む方式（`context.rs::synchronize_
observed`／`#[cfg(test)] pub(crate) synchronize_with_gpu_timestamps`）
でこれを実装した。本番 `synchronize()` は no-op オブザーバのままの
ため、ディスパッチ挙動・数値結果は不変（AC-2。既存 parity／bit 一致
テスト全 pass を実機確認済み）。`commit_wait` は引き続き「commit＋
カーネル専有＋`waitUntilCompleted`」の合算値として扱い、`kernel_gpu`
（GPUEnd−GPUStart）はその内訳として別出力する。M4 Max 実機での自己
検証テスト（`gpu_timestamps_within_commit_wait_window`）は厳密判定
（`kernel_gpu <= commit_wait` に slack なし）で pass した。N=1024/2048/
4096 の 5 run 実測・本表への数表追記は本イシューのスコープ外（親
#1275 配下の後続 sub-issue が担う）。

### 相互検証（`gemm_bench` example）

同一セッションで `cargo run -p fandhe-ai-backend-metal --example
metal_gemm_bench --release` を実行し、`dynamic_tile_auto`（転送込み）の
TFLOPS 換算値を突合した（§7）。ログは
`docs/perf/logs/metal-gemm-reuse-phase-1189/gemm_bench-crosscheck.log`。

## §3 Layer A 実測（5 run。単位 ms、中央値。q1/q3 は付録ログ参照）

| N | run | matmul | to_tensor | host_copy | checksum | iter_total |
|---|---|---|---|---|---|---|
| 1024 | 1 | 2.067 | 0.0001 | 0.2368 | 0.5803 | 2.891 |
| 1024 | 2 | 2.039 | 0.0000 | 0.2453 | 0.5711 | 2.859 |
| 1024 | 3 | 1.989 | 0.0001 | 0.2397 | 0.5690 | 2.791 |
| 1024 | 4 | 2.058 | 0.0000 | 0.2425 | 0.5693 | 2.875 |
| 1024 | 5 | 1.485 | 0.0000 | 0.2334 | 0.5679 | 2.280 |
| 2048 | 1 | 5.587 | 0.0001 | 0.8810 | 2.2300 | 8.622 |
| 2048 | 2 | 4.832 | 0.0001 | 0.9218 | 2.2516 | 8.093 |
| 2048 | 3 | 5.549 | 0.0000 | 0.8916 | 2.2682 | 8.766 |
| 2048 | 4 | 6.306 | 0.0001 | 0.9196 | 2.2956 | 9.530 |
| 2048 | 5 | 4.967 | 0.0001 | 0.9201 | 2.2824 | 8.197 |
| 4096 | 1 | 34.687 | 0.0002 | 3.6534 | 9.6276 | 47.944 |
| 4096 | 2 | 34.800 | 0.0002 | 3.6574 | 9.2485 | 47.626 |
| 4096 | 3 | 36.491 | 0.0002 | 3.7191 | 9.6361 | 49.824 |
| 4096 | 4 | 33.861 | 0.0002 | 3.7960 | 9.6136 | 47.132 |
| 4096 | 5 | 35.521 | 0.0001 | 3.7690 | 9.7401 | 48.948 |

**中央値の中央値**（5 run → 各フェーズの中央値をさらに中央値）:

| N | matmul | host_copy（% of iter_total） | checksum（% of iter_total） | iter_total |
|---|---|---|---|---|
| 1024 | 2.039 ms | 0.2397 ms（8.4%） | 0.5693 ms（19.9%） | 2.859 ms |
| 2048 | 5.549 ms | 0.9196 ms（10.7%） | 2.2682 ms（26.3%） | 8.622 ms |
| 4096 | 34.800 ms | 3.7191 ms（7.8%） | 9.6276 ms（20.1%） | 47.944 ms |

`to_tensor` は `Var::matmul` が既にホスト `Tensor` を返す Metal の性質上
全 N で 0.1〜0.2 µs（無視できる水準。CUDA と同様）。

`python3 summarize.py --strict results/raw/results-m4max-gemm-phases-0.6.0.jsonl`
は exit 0（`docs/perf/logs/metal-gemm-reuse-phase-1189/
layerA-summarize-output.md` に全 15 run 分の詳細表を保存）。

## §4 Layer B 実測（`crates/backend-metal` 内部分解。5 回計測の中央値、単位 ms）

| N | run | upload_a | upload_b | alloc_c | encode | commit_wait | readback | host_copy | sum |
|---|---|---|---|---|---|---|---|---|---|
| 1024 | 1 | 0.2422 | 0.2446 | 0.0001 | 0.0030 | 1.2687 | 0.1043 | 0.2561 | 2.1189 |
| 1024 | 2 | 0.2405 | 0.2472 | 0.0001 | 0.0031 | 1.2782 | 0.0891 | 0.2510 | 2.1092 |
| 1024 | 3 | 0.2281 | 0.2205 | 0.0001 | 0.0028 | 0.4066 | 0.0547 | 0.2195 | 1.1322 |
| 1024 | 4 | 0.2371 | 0.2392 | 0.0001 | 0.0043 | 1.2449 | 0.0692 | 0.2430 | 2.0377 |
| 1024 | 5 | 0.2202 | 0.2190 | 0.0001 | 0.0027 | 0.4052 | 0.0543 | 0.2215 | 1.1231 |
| 2048 | 1 | 0.9873 | 0.9752 | 0.0001 | 0.0061 | 2.5519 | 0.2395 | 0.9553 | 5.7155 |
| 2048 | 2 | 0.9736 | 0.9567 | 0.0001 | 0.0082 | 3.5673 | 0.2487 | 0.9645 | 6.7192 |
| 2048 | 3 | 1.0095 | 0.9871 | 0.0002 | 0.0083 | 2.9670 | 0.2380 | 0.9660 | 6.1760 |
| 2048 | 4 | 1.0477 | 0.9684 | 0.0000 | 0.0065 | 2.3036 | 0.2375 | 0.9823 | 5.5459 |
| 2048 | 5 | 0.9597 | 0.9267 | 0.0001 | 0.0065 | 2.2336 | 0.2378 | 0.9137 | 5.2782 |
| 4096 | 1 | 3.7900 | 3.9392 | 0.0006 | 0.0162 | 15.7457 | 0.9410 | 3.8020 | 28.2347 |
| 4096 | 2 | 3.8854 | 4.0288 | 0.0009 | 0.0175 | 16.1335 | 0.9412 | 3.7973 | 28.8046 |
| 4096 | 3 | 3.9851 | 3.8821 | 0.0007 | 0.0171 | 15.8012 | 0.9442 | 3.7840 | 28.4142 |
| 4096 | 4 | 3.8727 | 3.8134 | 0.0004 | 0.0167 | 15.6113 | 0.9382 | 3.6132 | 27.8659 |
| 4096 | 5 | 3.7496 | 3.8409 | 0.0003 | 0.0135 | 15.7135 | 0.9452 | 3.7884 | 28.0514 |

**集計方式**: `upload_a`〜`host_copy` の各列は、その列単独で 5 run の値を
集めて中央値を取る（列ごと独立集計）。`upload_a+b` はその
`upload_a` 中央値と `upload_b` 中央値の和（＝列ごと中央値の和）である。
一方 `sum` 列は各 run の 7 フェーズ合計値（§4 生ログ表の `sum` 列。
1 run 内で 20 試行の中央値を取った 7 フェーズを足し合わせた値）を
5 run 分集めたうえでその中央値を取ったものであり、**列ごと中央値の単純
合計とは一致しない**（run 内の相関により両者は一般に異なる。本表の
`sum` は常に後者＝「各 run 合計の中央値」で統一する）。

**中央値の中央値**:

| N | upload_a+b | alloc_c | encode | commit_wait | readback | host_copy | sum |
|---|---|---|---|---|---|---|---|
| 1024 | 0.4763 ms | ~0 ms | 0.0030 ms | 1.2449 ms | 0.0692 ms | 0.2430 ms | 2.0377 ms |
| 2048 | 1.9557 ms | ~0 ms | 0.0065 ms | 2.5519 ms | 0.2380 ms | 0.9645 ms | 5.7155 ms |
| 4096 | 7.7548 ms | ~0.0006 ms | 0.0167 ms | 15.7457 ms | 0.9412 ms | 3.7884 ms | 28.2347 ms |

N=1024 の `commit_wait` は run3/5 のみ約 0.405 ms と run1/2/4（約
1.25〜1.28 ms）の約 1/3 に落ちる二峰性が見られた（`docs/perf/
metal-gemm-bottleneck-rediagnosis.md` §3.4「プロセス間で中央値が最大約
3 倍変動する」と同種の計測揺らぎ。原因は未特定のまま記録するにとどめ、
本イシューの結論では中央値の中央値のみを用いる）。N=2048/4096 では
同様の二峰性は見られなかった。

Layer B の `resolved_tile`（`tile::select_for_device` の解決結果）は
N=1024: `bm=64,bn=32,bk=8,wm=4,wn=1,staged=true`、N=2048:
`bm=64,bn=32,bk=16,wm=2,wn=2,staged=true`、N=4096:
`bm=32,bn=64,bk=16,wm=2,wn=2,staged=true`（`docs/perf/
metal-gemm-bottleneck-rediagnosis.md` の `CANDIDATES[3]`/`[5]`/`[6]`
と一致。§2「Metal と CUDA の対応表」参照）。

## §5 突合（Layer A `matmul` vs Layer B Σ、host_copy 除外）

Layer A の `matmul` 区間は `matmul(&b)` 完了時点で計時を終了し、
`to_vec()`（host_copy）はハーネス側で別区間として測定される（§3）。
したがって `matmul` 区間と境界を揃えるには、Layer B の Σ からも
`host_copy` を除いた値（upload_a+upload_b+alloc_c+encode+commit_wait+
readback）を使う必要がある。Σ（host_copy 除外）は各 run の
（`sum` − `host_copy`）を 5 run 分集めた中央値として求める（§4 の
集計方式と同じく「run 合計の中央値」方式）。

| N | Layer A `matmul` 中央値 | Layer B Σ（host_copy 除外）中央値 | 差分 |
|---|---|---|---|
| 1024 | 2.039 ms | 1.7947 ms | +13.6%（Layer A が大きい） |
| 2048 | 5.549 ms | 4.7602 ms | +16.6%（Layer A が大きい） |
| 4096 | 34.800 ms | 24.4327 ms | +42.4%（Layer A が大きい） |

**host_copy を含めた旧集計（Σ=2.0377/5.7155/28.2347 ms）は境界の異なる
量同士を比較しており、N=1024/2048 が「ほぼ一致」に見えたのは host_copy
分がたまたま両者の差を相殺していたことによる見かけ上の一致だった**。
host_copy を除いて境界を揃えると、N=1024/2048/4096 いずれも Layer A
`matmul` が Layer B Σ を明確に上回り、乖離幅は N とともに拡大する
（13.6%→16.6%→42.4%）。**N=4096 の乖離は旧記載の約19%ではなく約
42%であり、旧集計は乖離を過小評価していた**。

この残差（Layer A `matmul` にのみ含まれ Layer B の 6 フェーズ分解には
現れない時間）の発生源は本イシューでは特定できていない。Layer B は
`dispatch_tiled_prepared` と同じ NN 経路の診断ヘルパ
（`diag_encode_tiled_nn`）を通じた計測であり、本番経路
`dispatch_auto`（タイル選択・variant 分岐・`Var`/`Tensor` ラッピング
等を含む）の全ステップを計時していない。`docs/perf/
metal-gemm-bottleneck-rediagnosis.md` §3.4 が記録する「プロセス間で
中央値が最大約 3 倍変動する」計測揺らぎ、および本ファイル §4 で観測した
N=1024 の commit_wait 二峰性と同種の環境要因（GPU クロック遷移・他
プロセスとの競合等）に加え、`dispatch_auto` 内のタイル選択・`Var`
ラッピング等 Layer B 診断ヘルパが計時しない本番経路固有のオーバーヘッド
が寄与している可能性がある。いずれも特定はできていない。この乖離は
情報提供に留め、**固定費の帰属（§6）は Layer A の `iter_total` 比のみ
から述べる**方針を維持する（実装計画§7 リスク節が明示的に許容する
扱い）。

`commit_wait`（Layer B）を TFLOPS へ換算すると N=1024: 1.72 TFLOPS、
N=2048: 6.57 TFLOPS、N=4096: 8.51 TFLOPS。`gemm_bench` example の
`dynamic_tile_auto_tflops`（転送込み。§2「相互検証」参照）は N=1024:
1.852、N=2048: 3.319、N=4096: 4.925 TFLOPS で、`commit_wait` 単独の
値より小さい（転送・readback を含むため整合的）。両者とも
`docs/perf/metal-gemm-n4096-kernel-gap.md`（N=4096 カーネル純境界
candle 比ギャップ調査）の実測レンジと同じオーダーにある。

## §6 固定費の帰属（`iter_total` 比。CUDA #1182 §6 との対比）

| N | matmul（%） | host_copy（%） | checksum（%） |
|---|---|---|---|
| 1024 | 71.3%（median row） | 8.4% | 19.9% |
| 2048 | 64.4%（median row。run 間で 60〜66%） | 10.7% | 26.3% |
| 4096 | 72.6%（median row） | 7.8% | 20.1% |

`host_copy`＋`checksum`（ハーネス診断コスト。#965/#970 の縮退検出
契約）は `iter_total` の約 28〜37% を占め、CUDA（#1182 §6: 66〜75%）
より明確に小さい。すなわち **Metal は CUDA と異なり、`matmul` 区間
自体が iter_total の大半（60〜73%）を占めており、ハーネス固定費が
支配的ではない**。

**訂正（codex-review 指摘。旧稿の誤り）**: 旧稿はこの違いの原因を
「CUDA は `clone_htod`/`clone_dtoh` が別ホスト API 呼び出しへ分離
されるが Metal は 1 回の `ctx.dispatch_sync` 呼び出しに閉じている
ため」と説明していたが、これは実装と一致しない。CUDA 側も
`gemm_fp32_strict` → `CudaGemm::run_tiled_f32`（`crates/backend-cuda/
src/gemm.rs`）が `clone_htod`（A/B）・カーネル起動・`synchronize`・
`clone_dtoh` を**1 回の関数呼び出し内**で完了させてから `Tensor` を
返しており、Metal の `dispatch_variant` が `upload_a`/`upload_b`・
`encode`・`synchronize`・`readback` を 1 回の呼び出しで完結させるのと
API 呼び出し境界の粒度は同型である（§2 の対応表のとおり、いずれの
バックエンドも upload／kernel／sync／readback は Layer A `matmul`
区間の内側に含まれ、ハーネス側から見た「API 呼び出しの分離」の有無に
差はない）。したがって `matmul` が `iter_total` に占める比率が
バックエンド間で異なる理由を「API 呼び出しの分離」に帰属させることは
できない。実測で裏付けられているのは表の比率の違いそのものまでで
あり、その原因（GPU 実行内部の相対コスト構成の違い等）は本イシューの
計測範囲では特定できていない——原因未特定のまま記録するにとどめる
（§8 参照）。

## §7 kernel 専有時間ベースの candle 比（参考値。分母は #1147 正式系列
candle fresh 中央値）

| N | matmul 中央値（transfer+kernel+sync 込み。ハーネス host_copy／
    checksum を除く） | candle fresh 中央値（#1147 §4.1） | matmul/candle_fresh |
|---|---|---|---|
| 1024 | 2.039 ms | 2.071 ms | 0.985（ほぼ同等） |
| 2048 | 5.549 ms | 6.151 ms | 0.902（matmul が上回る） |
| 4096 | 34.800 ms | 22.948 ms | **1.516（matmul が 1.52 倍遅い）** |

**非対称性の明記（CUDA との対比）**: CUDA（#1182 §7）は `matmul` 単体
が candle fresh を **全 N で上回る**（N=1024: 1.59 倍・N=4096: 1.47
倍）ことを確認し、「reuse ゲート未達の主因はハーネス自身の
`host_copy`＋`checksum`」と結論した。**Metal は逆で、N=1024/2048 では
`matmul` 単体が candle fresh とほぼ同等（0.90〜0.99 倍）まで縮まるが、
N=4096 では `matmul` 単体がむしろ candle fresh より遅い**。すなわち
Metal の N=4096 candle 比未達（#1147: 0.589 倍）は、ハーネス測定境界
の固定費を除去しても解消しない——GPU 実行自体（アップロード・カーネル・
readback の合算）に起因するギャップである。この結論は
`docs/perf/metal-gemm-n4096-kernel-gap.md`（N=4096 カーネル純境界の
candle 比ギャップ調査。約 9.9 対 13.17 TFLOPS）と整合する。

分母（candle fresh）と分子（matmul。reuse 内の 1 区間）は計測境界が
非対称（fresh vs reuse の tape/デバイス初期化コストの扱いが異なる）
であるため、本表は既存の正式ゲート判定（`#1037`・#1147 の `reuse` 全体
中央値による判定）を置き換えるものではなく **参考値**として扱う。

## §8 スコープ外

- GPU タイムスタンプ変種（`kernel_gpu`。`MTLCommandBuffer::
  GPUStartTime`/`GPUEndTime` で `commit_wait` から純カーネル専有時間を
  分離する）はイシュー #1276 で実装済み（§2「Layer B」参照。
  `MetalContext::synchronize_with_gpu_timestamps`。本番 `synchronize()`
  は no-op オブザーバのため AC-2 は不変）。`commit_wait` は commit＋
  カーネル専有＋`waitUntilCompleted` の合算値のまま。N=1024/2048/4096
  の 5 run 実測・本表への数表追記は #1276 のスコープ外だったが、
  イシュー #1277 で実施済み（§11）
- N=1024 の `commit_wait` 二峰性（run3/5 のみ約 1/3 に低下）・全 N での
  Layer A `matmul` と Layer B Σ（host_copy 除外）の乖離（§5。
  13.6%〜42.4%）はいずれも原因未特定のまま記録するにとどめる
- `matmul` が `iter_total` に占める比率が CUDA より Metal で高い理由
  （§6 訂正後）は、API 呼び出し境界の分離差ではないことまでは確認した
  が、真因（GPU 実行内部の相対コスト構成の違い等）は本イシューの計測
  範囲外であり原因未特定のまま記録するにとどめる（codex-review 指摘。
  #1189）
- `Tensor<f32>` デバイス常駐化等の設計変更は対象外（計画共通節）
- tolerance／baseline／依存の追加変更は行っていない

## §9 ユーザー判断事項（AC-3。事実／選択肢／推奨／影響範囲）

### 事実

1. Metal reuse 計測境界のハーネス固定費（`host_copy`＋`checksum`）は
   `iter_total` の約 28〜37%（N 依存）で、CUDA（66〜75%）より小さい
2. `matmul` 単体（transfer＋kernel＋sync）は N=1024/2048 で candle
   fresh とほぼ同等（0.90〜0.99 倍）だが、**N=4096 では candle fresh
   より 1.52 倍遅い**
3. Metal の N=4096 candle 比未達（#1147: 0.589 倍）は、CUDA と異なり
   ハーネス測定境界の産物ではなく GPU 実行自体のギャップである
4. Layer A `matmul` と Layer B Σ（host_copy 除外・境界を揃えた比較）は
   N=1024/2048/4096 いずれも Layer A が上回り、乖離幅は N とともに
   拡大する（13.6%→16.6%→42.4%。原因未特定。§5）

### 選択肢

- (i) reuse 判定境界の再定義（REQ-8 影響）: **Metal では見送りが妥当**
  （事実 3 のとおり、境界を `matmul` のみへ絞っても N=4096 の未達は
  解消しないため、CUDA のような再定義の動機がない）
- (ii) ハーネス `host_copy`／`checksum` の削減: Metal では効果が限定的
  （事実 1。CUDA ほどの改善余地がない）
- (iii) N=4096 の `matmul` 内部ギャップの追加調査（別イシュー化）:
  `docs/perf/metal-gemm-n4096-kernel-gap.md`（既存）の延長線上の調査を
  継続する。本イシューの Layer B 分解（§4）は upload/encode/
  commit_wait/readback という粗い粒度に留まるため、カーネル専有時間
  自体（GPU タイムスタンプ変種。§8 参照）の分離が次の手がかりになりうる
  ——イシュー #1277 で実施済み。§11 の分母表（Phase 2 候補評価用）を
  参照
- (iv) candle 側 kernel-only 計測との突合（#1103 追補）: 既存の候補と
  して残す
- (v) 現状維持（本イシューは診断のみで完結し、本番結線・ゲート判定の
  変更は行わない）

### 推奨

**(v) 現状維持 ＋ (iii) の別イシュー化を推奨**。事実 3 が示すとおり、
Metal の candle 比未達は CUDA と異なりハーネス境界の見直しでは解決
しない構造的な問題であり、(i)/(ii) は Metal には適用の動機がない。
N=4096 の `matmul` 内部（カーネル専有時間そのもの）の追加分解は
既存の `metal-gemm-n4096-kernel-gap.md` 系列の延長として別イシューで
継続するのが妥当。

### 影響範囲

- 本イシューはコード変更を `#[cfg(test)]` 限定の診断ヘルパ・診断テスト
  ファイルの純追加に限定しており、本番経路（`dispatch_auto`・
  `dispatch_variant`・`dispatch_tiled_prepared`）・既存テスト・
  ゲート判定基準（`#1037`・`#1147`）への影響はない
- (iii) を別イシュー化する場合、対象は `crates/backend-metal` の
  N=4096 タイル構成・カーネル実装（`shaders/gemm.metal`）に限定され、
  `facade`／`tensor-core`／他バックエンドへの影響はない

## §10 関連ドキュメント

- `docs/perf/cuda-gemm-reuse-phase-breakdown.md`（CUDA 側同型調査。
  #1182）
- `docs/perf/metal-gemm-candle-gate-remeasurement.md`（#1147。candle 比
  ゲート判定の確定記録）
- `docs/perf/metal-gemm-bottleneck-rediagnosis.md`（#1036/#1103。転送
  寄与の先行調査）
- `docs/perf/metal-gemm-n4096-kernel-gap.md`（#1143。N=4096 カーネル
  純境界ギャップ調査）
- `docs/backend-metal-command-batching-design.md`（#1017。`encode`/
  `synchronize` のバッチング機構。GPU タイムスタンプ変種を実装しない
  判断の根拠）
- `docs/perf/metal-bench-noise-protocol.md`（計測衛生プロトコル）
- `docs/perf/logs/metal-gemm-reuse-phase-1277/`（本節 §11 の実行ログ・
  env_info・集計。イシュー #1277）
- `docs/perf/logs/metal-gemm-window-macmini-3182/`（§12 の事前登録規則・
  生ログ・集計。Mac mini。イシュー #3182）

## §11 GPU タイムスタンプによる純カーネル専有時間の分離（イシュー #1277）

`#1276`（§8）が実装した `kernel_gpu`（`MTLCommandBuffer::GPUStartTime`/
`GPUEndTime` の差分。`MetalContext::synchronize_with_gpu_timestamps`。
本番 `synchronize()` は no-op オブザーバのため不変）を用い、
`gemm_reuse_phase_diag_production_batch`（`crates/backend-metal/src/
gemm_reuse_phase_diag_tests.rs`。`#[ignore]`）を 5 プロセス起動（各回
20 warmup + 20 測定）して N=1024/2048/4096 の `kernel_gpu`・
`commit_wait`・`commit_wait − kernel_gpu` を確定した。**コード変更は
ゼロ**（診断テスト・診断ヘルパは #1276 で実装済み。本イシューは計測
実行と本節の追記のみ）。

### §11.1 環境・プロトコル

- 実機: Apple M4 Max（GPU 40 コア）。転送元コミット `c71264f`
  （origin/main。#1276「GPU タイムスタンプ取得」#1371・#1372 反映後）
- rustc/cargo 1.96.0、macOS 26.6.2 (BuildVersion 25G83)
- 事前ビルド（`cargo test -p fandhe-ai-backend-metal --release
  --no-run`）→ `pmset -g therm`（前）→ 負荷確認（他セッションの
  並走 `cargo test --workspace --locked` を検出。load average が
  1 分値 9.25 から約 7 分で 2.85 まで低下したのを待って計測開始）→
  スモーク（`gpu_timestamps_within_commit_wait_window` 1 回。pass）→
  本計測（`gemm_reuse_phase_diag_production_batch` を 5 プロセス起動。
  run 間 30 秒クールダウン、`--release --test-threads=1 --ignored
  --nocapture`）→ `pmset -g therm`（後）
- `pmset -g therm` は前後とも「記録なし」でサーマルスロットリングの
  兆候なし
- `resolved_tile` は全 5 run・全 3 サイズで `requested_tile` と完全
  一致（フォールバック NOTE なし。§4 と同一構成: N=1024
  `bm=64,bn=32,bk=8,wm=4,wn=1`、N=2048 `bm=64,bn=32,bk=16,wm=2,wn=2`、
  N=4096 `bm=32,bn=64,bk=16,wm=2,wn=2`）
- 詳細: `docs/perf/logs/metal-gemm-reuse-phase-1277/env_info.txt`（各
  run 直前の uptime を含む）

### §11.2〜§11.3 実測表・中央値の中央値

各列は「run ごとの中央値」を 5 run 分集め、列ごとに独立して中央値
（中央値の中央値）を取る（`commit_wait − kernel_gpu` は各 run が反復
ごとの差分から算出した中央値を用いる。`median(commit_wait) −
median(kernel_gpu)` では算出しない）。生値は
`docs/perf/logs/metal-gemm-reuse-phase-1277/layerB-run1.log`〜
`layerB-run5.log`、集計は `layerB-aggregate.md` を参照。

| N | commit_wait 中央値 (ms) | **kernel_gpu 中央値 (ms)** | commit_wait−kernel_gpu 中央値 (ms) | kernel_gpu 基準 TFLOPS |
|---|---|---|---|---|
| 1024 | 1.2606 | **1.0267** | 0.2339 | 2.092 |
| 2048 | 3.8869 | **3.1849** | 0.7020 | 5.394 |
| 4096 | 15.7892 | **13.7051** | 2.0593 | 10.028 |

**N=2048/4096 の run 間変動**: N=2048 は run 間で kernel_gpu が 1.60〜
7.28 ms（二峰性寄り: run1/3/4 が高値側、run2/5 が低値側）と大きく
変動した。N=4096 は run4 のみ 17.6802 ms の外れ値（他 4 run は
13.70〜14.09 ms）。中央値集計のためこれらの変動は最終値への影響は
限定的だが、原因は未特定のまま記録する（#1189 §8 の「N=1024
commit_wait 二峰性」と同種の環境要因の可能性がある）。

**妥当性帯チェック**: `docs/perf/metal-gemm-n4096-kernel-gap.md` の
N=4096 カーネル純境界 9.76〜9.91 TFLOPS（2·4096³ 換算で 13.87〜14.08
ms 帯）に対し、本計測の kernel_gpu 中央値 13.7051 ms（10.028 TFLOPS）
はこの帯よりわずかに高速側（帯下限比 約 1.2% 短い）。計測プロトコルが
異なる（#1143 は `gemm_bench` 系の独立ベンチ、本計測は GPU
タイムスタンプ）ため「大きく外れる」水準ではないと判断し採用した。

**セッション間ドリフト検査**: N=4096 の `commit_wait` 中央値
15.7892 ms は #1189 §4 の実測値 15.7457 ms と乖離約 0.28%（10% 閾値を
大きく下回る）のため、Layer A 再計測（任意項目）は実施しなかった。

### §11.4 N=4096 ギャップ内訳（AC-2）

2 つの数値を定義してラベル付きで併記する（分母・分子の出典は §7）。

- **(a) fandhe 内部のカーネル比率**: `kernel_gpu(4096)` 13.7051 ms /
  `matmul`（Layer A。§7）34.800 ms = **約 39.4%**。残り約 60.6%が
  `commit_wait − kernel_gpu`（2.0593 ms）・アップロード（upload_a+b
  中央値 7.4987 ms）・readback（0.9191 ms）・encode・Layer A/B 残差
  （§5。N=4096 で約 42.4%）の合算に相当する
- **(b) candle 比ギャップの帰属**（AC-2 が求める値）: ギャップ =
  `matmul` 34.800 ms − candle fresh 22.948 ms（#1189 §7。fandhe-ai
  `=0.6.0` 正式系列）= 11.852 ms。candle カーネル純境界（`docs/perf/
  metal-gemm-n4096-kernel-gap.md` §0: 13.17 TFLOPS）を時間換算すると
  2·4096³/13.17e12 ≈ 10.4358 ms。カーネル帰属分 = `kernel_gpu(4096)`
  13.7051 ms − 10.4358 ms = **3.2693 ms（ギャップの約 27.6%）**。
  非カーネル帰属分 = 11.852 − 3.2693 = **8.5827 ms（ギャップの約
  72.4%）**

  前提・限界:
  - candle カーネル純境界（13.17 TFLOPS）は `metal-gemm-n4096-
    kernel-gap.md` の別セッション・別計測境界（`gemm_bench` 系の壁時計
    計測であり GPU タイムスタンプではない）の値である
  - 分母のアンカー（`matmul` 34.800 ms・candle fresh 22.948 ms）は
    fandhe-ai `=0.6.0` 正式系列（#1147・#1189）。**脚注（0.7.0 正式
    系列との参考比較）**: `docs/perf/metal-gemm-candle-gate-
    remeasurement.md` §11（2026-09-06・共有負荷下）の candle fresh
    中央値は 23.100 ms（0.6.0 の 22.948 ms とほぼ同水準）で 0.509 倍。
    本節の (a)(b) の算術には 0.6.0 系列の値のみを用い、0.7.0 の値は
    混ぜない
  - (b) は「候補評価で回収可能な上限」を与えるものであり、
    `commit_wait − kernel_gpu`（commit・スケジューリング・
    `waitUntilCompleted` 復帰の固定費。約 2.06 ms）はカーネル側変更
    では回収できない非対象領域である

### §11.5 Phase 2 候補評価の分母表

`#1273` 配下の候補評価イシュー（#1286/#1291/#1297/#1323/#1324/#1325/
#1368 等）が「Phase 1 で確定した純カーネル時間を分母に比較」する際は
以下を分母として使う。**#1370（hfrag 候補の純カーネル時間比較）で分母
として使用済み**（`docs/perf/metal-gemm-hfrag-candidate.md` §9.5「妥当性
帯チェック」）。

| N | 純カーネル時間（kernel_gpu 中央値。ms） | TFLOPS | commit_wait−kernel_gpu（回収不能固定費。ms） |
|---|---|---|---|
| 1024 | 1.0267 | 2.092 | 0.2339 |
| 2048 | 3.1849 | 5.394 | 0.7020 |
| 4096 | 13.7051 | 10.028 | 2.0593 |

- **計測境界**: `GPUEndTime − GPUStartTime`（`MTLCommandBuffer` の GPU
  タイムスタンプ。`MetalContext::synchronize_with_gpu_timestamps`。
  本番 `synchronize()` は no-op オブザーバで計測経路にのみ使う）
- **起動経路**: `gemm_reuse_phase_diag_production_batch` →
  `diag_encode_tiled_nn`（本番 `dispatch_tiled_prepared` と同じ NN
  経路の診断ヘルパ）。`resolved_tile` は §11.1 のとおり本番選択と一致
- **再現コマンド**: `cargo test -p fandhe-ai-backend-metal --release
  gemm_reuse_phase_diag_production_batch -- --ignored --nocapture
  --test-threads=1`
- **比較時の規約**: 候補側も同一プロトコル（`--release`・
  `--test-threads=1`・20 warmup + 20 測定・5 プロセス起動の中央値の
  中央値）で `kernel_gpu` を取得し、本表と比較すること。単一プロセス・
  少数反復の比較は §11.3 が示す run 間変動（N=2048/4096 で顕著）により
  誤判定しうる

### §11.6 未確定事項・スコープ外

- Layer A/B 残差（§5。13.6%〜42.4%）は本イシューでも未解消のまま
- candle 側の GPU タイムスタンプ計測（`kernel_gpu` 相当値の candle
  側取得）は未実施。§11.4 (b) の candle カーネル純境界は既存の壁時計
  計測値（#1143）を参照するに留める
- N=2048/4096 の run 間変動（§11.3）の原因特定・N=1024 二峰性（#1189
  §8）との関係整理は対象外
- Phase 2 候補（E2〜E9 等）の実装・結線判断は対象外（各候補イシュー
  が担う）
- tolerance／baseline／依存の追加変更は行っていない


## §12 Mac mini（Apple M4・GPU 10 コア）の計測窓の内訳（イシュー #3182）

§3〜§4・§11 は M4 Max・`fandhe-ai =0.6.0` の値だった。本節は **Mac mini** で同じ窓の内訳を取り直した記録で、
診断のみ（計測窓の構成・判定式・tolerance・baseline・`scripts/bench/framework-compare/**`・本番コードは不変）。
規則の正は `docs/perf/logs/metal-gemm-window-macmini-3182/RULE.txt`（計測前に固定）、生ログ・集計は同ディレクトリ
（`aggregate.md` が機械出力）。M4 Max の §3・§4・§11 は書き換えず、並置は §12.8 に版と機体を付けて置く。

### §12.1 環境・腕・プロトコル

- 機体: Mac mini（Apple M4・10 コア・16 GB・macOS 27.0.1・rustc 1.99.0）。5 run（独立プロセス群）・負荷ゲート 5/5 `pass`
  （load1 < 3.0。`load_gate.log`）。`run{1..5}.err` は全て空。
- 腕 A（Layer A）: 計測前コミット `3114cdda2fe6c073184ad704537a5ed1076f6a10` の `scripts/bench/framework-compare`。
  `bench-fandhe` は registry `fandhe-ai =0.11.0`（`tree.txt` が 1 行で確認）、`bench-candle` は `candle-core =0.11.0`。
- 腕 B（Layer B）: 同ツリーの main workspace（`backend-metal` の診断テスト。HEAD ソースの path 依存）。
  **腕 A（0.11.0）と腕 B（HEAD）はソース版が違う**ため、A の列と B の列を足して 1 つの窓として扱わない。
- ツリー指紋（`find crates scripts/bench/framework-compare -type f ! -path '*/target*'` を sort した sha256 一覧の sha256）は
  送信側・Mac mini 側とも `063846beb…ad7909` で一致。
- `gemm --mode fresh --phases` は `bench-fandhe` が MEASURE_ERROR で拒否する仕様（`tests::phases_with_gemm_fresh_is_measure_error`
  で固定）で、ハーネスは変えない方針のため、fresh は phases なしの窓合計のみを取った（§12.7）。区間内訳は reuse の `--phases` と Layer B。
- 経路: 診断経路（`diag_encode_tiled_nn`）が本番 `dispatch_auto` を代表するかを `gemm_window_route_precheck` で確認した。
  5 サイズ × 5 run とも `ROUTE_ACTUAL=Classic`・出力 bit 一致（`DIAG_BITEQ equal=true`）で、全 N で Layer B は代表。
  Mac mini の resolved_tile は N=256〜4096 の全てで `bm32 bn32 bk16 wm2 wn2 staged`。M4 Max（§4）とは構成が違う。

### §12.2 Layer A（bench-fandhe 0.11.0 `gemm reuse --phases`。5 run の中央値の中央値、ms）

| N | matmul | host_copy | checksum | iter_total |
|---|---|---|---|---|
| 256 | 0.3308 | 0.0221 | 0.0546 | 0.4157 |
| 512 | 0.6354 | 0.0729 | 0.1580 | 0.8655 |
| 1024 | 2.1986 | 0.2406 | 0.5665 | 3.0144 |
| 2048 | 10.8913 | 0.9032 | 2.2352 | 14.0395 |
| 4096 | 78.1085 | 3.7062 | 8.6361 | 90.1252 |

`to_tensor` は全 N で 0.0001 ms 以下。5 run 分の生値と min–max は `aggregate.md`。

### §12.3 Layer B（backend-metal 診断テスト・HEAD。5 run の中央値の中央値、ms）

| N | upload(a+b) | commit_wait | kernel_gpu | commit_wait−kernel_gpu | readback | encode | Σ（host_copy 除く） |
|---|---|---|---|---|---|---|---|
| 256 | 0.0370 | 0.2328 | 0.0197 | 0.2132 | 0.0045 | 0.0022 | 0.2766 |
| 512 | 0.1248 | 0.3471 | 0.1148 | 0.2318 | 0.0195 | 0.0022 | 0.4934 |
| 1024 | 0.5203 | 1.2534 | 0.8607 | 0.3912 | 0.1024 | 0.0027 | 1.8732 |
| 2048 | 1.8262 | 8.6615 | 8.0716 | 0.5814 | 0.3732 | 0.0043 | 10.8482 |
| 4096 | 7.6416 | 67.5694 | 65.2521 | 2.2951 | 1.4676 | 0.0110 | 76.6727 |

- `alloc_c` は 0.0003 ms 以下（4096）。`host_copy`（B）は A の値と同水準（突合用）。
- N=1024 の run1 だけ外れ値（kernel_gpu 1.85 ms、他 4 run は 0.86 ms 付近。`aggregate.md`）。中央値集計のため最終値への影響は限定的。原因は未特定。
- A の `matmul` と B の Σ の残差（A−B。未説明）: N=256 +0.054、512 +0.142、1024 +0.325、2048 +0.043、4096 +1.436 ms。原因は断定しない（A と B の版が違う点を含む）。

### §12.4 checksum の ns／要素

A の `checksum` 区間中央値 / N²: N=256 0.834、512 0.603、1024 0.540、2048 0.533、4096 0.515 ns／要素。
checksum は全フレームワーク共通の Rust 側 f64 逐次総和で、N=4096 では 8.64 ms。

### §12.5 割合表（分母は A の `iter_total`）

| N | kernel_gpu(B) | upload(B) | host_copy(A) | checksum(A) | その他(B) | 未説明 |
|---|---|---|---|---|---|---|
| 256 | 4.7% | 8.9% | 5.3% | 13.1% | 52.9% | 15.0% |
| 512 | 13.3% | 14.4% | 8.4% | 18.3% | 29.3% | 16.3% |
| 1024 | 28.6% | 17.3% | 8.0% | 18.8% | 16.6% | 10.9% |
| 2048 | 57.5% | 13.0% | 6.4% | 15.9% | 6.8% | 0.3% |
| 4096 | 72.4% | 8.5% | 4.1% | 9.6% | 4.2% | 1.2% |

- 区分: カーネル = kernel_gpu、**fandhe-ai 固有の非カーネルコスト = upload・host_copy**、**全フレームワーク共通 = checksum**、
  その他 = alloc_c＋encode＋(commit_wait−kernel_gpu)＋readback（B）、未説明 = iter_total − 上記合計。
- 各列は中央値のため合計は iter_total と一致しない（差は「未説明」に入れ、吸収させない）。A（0.11.0）と B（HEAD）を同じ分母で割っている。
- N=4096 の fandhe 固有分は upload 7.64 ms＋host_copy 3.71 ms＝11.35 ms、共通の床 checksum は 8.64 ms。
  `host_copy` は Metal が `readout_uses_borrowed_view("metal", None) == false` のため全面コピーが 1 回入る分（N=4096 で 64 MiB）。

### §12.6 candle との差とカーネル由来の割合

G = fandhe reuse の窓合計（phases なし）− candle `gemm`（fresh）の窓合計（同一セッション・5 run の中央値）。

| N | fandhe reuse | candle fresh | G | candle transfer_excl | (a) (kernel_gpu−excl)/G | (b) ((encode+commit_wait)−excl)/G | 固有(upload+host_copy)/G |
|---|---|---|---|---|---|---|---|
| 256 | 0.3334 | 0.3054 | +0.0281 | 0.2046 | −658.4%（負値） | 108.2%（1 超） | 210.4%（1 超） |
| 512 | 0.7091 | 0.5403 | +0.1688 | 0.3234 | −123.6%（負値） | 15.3% | 117.1%（1 超） |
| 1024 | 3.0510 | 2.2893 | +0.7617 | 0.9503 | −11.8%（負値） | 40.2% | 99.9% |
| 2048 | 14.0746 | 9.4911 | +4.5835 | 5.4463 | 57.3% | 70.2% | 59.5% |
| 4096 | 90.0727 | 55.3087 | +34.7640 | 42.4764 | 65.5% | 72.2% | 32.6% |

- **N=4096 のカーネル由来は (a) 65.5%・(b) 72.2%**。candle との差 34.76 ms のうち約 3 分の 2 から 7 割強がカーネル（＋同期）側、
  fandhe 固有の upload＋host_copy が 32.6%。N=2048 は (a) 57.3%・(b) 70.2%。
- N=256〜1024 は G が小さく（0.03〜0.76 ms）、負値・1 超が出ているため**割合として読まない**（丸めずフラグ付きで出力）。
- 限界: candle の `gemm_transfer_excl` は壁時計（matmul＋synchronize）、`kernel_gpu` は GPU タイムスタンプなので (a) はカーネル差を小さく見積もりやすい。
  (b) は同じ壁時計の境界（エンコード＋同期）に揃えた値。candle 側の readback・fandhe 側の A/B 版差は分離していない。
- M4 Max・0.6.0 の §11.4(b)（candle 13.17 TFLOPS 基準で 27.6%）とは機体・版・基準が違うため算術に混ぜない。
- ドリフト検査（N=4096）: fandhe reuse 90.07 ms は系列 B（122.82 ms）比 −26.7%、candle fresh 55.31 ms は系列 B（71.19 ms）比 −22.3%。
  どちらも **10% 超の乖離**（規則どおり記録のみで判定は不変）。両者とも系列 B より速く、fandhe/candle 比は 1.63（系列 B は 1.73）。原因は切り分けていない。

### §12.7 fresh と reuse の窓合計

| N | fresh | reuse | reuse−fresh | (reuse−fresh)/出力 MiB |
|---|---|---|---|---|
| 256 | 0.3354 | 0.3334 | −0.0019 | −0.0078 |
| 512 | 0.6742 | 0.7091 | +0.0349 | +0.0349 |
| 1024 | 2.8566 | 3.0510 | +0.1944 | +0.0486 |
| 2048 | 13.5325 | 14.0746 | +0.5422 | +0.0339 |
| 4096 | 84.8794 | 90.0727 | +5.1933 | +0.0811 |

N=512 以上で reuse の方が fresh より遅い（N=4096 で +5.19 ms）。反復ごとの解放をしない keep-alive の影響などは**推測であり、本記録では断定しない**
（窓側の是正は #3265）。fresh の区間内訳は `--phases` が使えないため取っていない。

### §12.8 M4 Max・0.6.0 との並置（別機体・別版。混ぜない）

| N | 項目 | M4 Max 0.6.0（§3・§11.2） | Mac mini（A: 0.11.0／B: HEAD） |
|---|---|---|---|
| 2048 | iter_total | 8.622 | 14.0395 |
| 2048 | kernel_gpu | 3.1849 | 8.0716 |
| 2048 | host_copy / checksum | 0.9196 / 2.2682 | 0.9032 / 2.2352 |
| 4096 | iter_total | 47.944 | 90.1252 |
| 4096 | kernel_gpu | 13.7051 | 65.2521 |
| 4096 | host_copy / checksum | 3.7191 / 9.6276 | 3.7062 / 8.6361 |

CPU 側の host_copy／checksum は機体間でほぼ同水準、GPU 側（kernel_gpu／commit_wait）は Mac mini が大きい。全項目の並置は `aggregate.md`。

### §12.9 後続イシューへの含意（事実のみ。目標値の決定は #3211）

- #3181（純カーネル差）: Mac mini の N=4096 では candle との差の 65.5〜72.2% がカーネル（＋同期）側に当たる。N≤1024 は差が小さく割合では測れない。
- #3265（窓側の是正）: fandhe 固有の upload＋host_copy は N=4096 で 11.35 ms（G の 32.6%）。共通の checksum 8.64 ms は削れない床。
- #3211: 上の値は Mac mini・0.11.0（A）＋HEAD（B）の実測で、M4 Max の値とは別に扱う。

### §12.10 未確定事項・スコープ外

- A と B の版差（0.11.0／HEAD）、A−B 残差、N=1024 run1 の外れ値、系列 B とのドリフト（−26.7%／−22.3%）、reuse が fresh より遅い原因は未特定。
- fresh の `--phases`（ハーネス変更を要する）と candle 側の GPU タイムスタンプ計測は行っていない。
- `aggregate.py` は計測後に 1 点だけ直した（checksum は負値がありうるため符号を許す）。計測値・判定式には影響しない。
- checksum が一致しても bit 同一の証拠ではない。tolerance／baseline／依存の変更は行っていない。
- 生ログ: `docs/perf/logs/metal-gemm-window-macmini-3182/`。
