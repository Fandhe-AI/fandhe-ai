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

**GPU タイムスタンプ変種（`kernel_gpu`）を実装しない判断**: 実装計画は
`MTLCommandBuffer::GPUStartTime`/`GPUEndTime` で `commit_wait` から
純カーネル専有時間を分離する変種 B を「実装可能なら必須」としていたが、
`MetalContext::encode`/`synchronize` はバッチング機構（イシュー
#1017）によりコマンドバッファを内部に閉じ込めており、これを分離する
には自前のコマンドキュー経路（`ctx.queue()`）を新設する必要がある。
AC-2「既存の本番経路・既存テストを変更しない」の安全側判断（計画§7
リスク節が明示的に許容する縮退先）に従い、本イシューでは変種 A
（`ProductionBatch`。`encode`＋`synchronize` を本番同一経路で個別計時）
のみを実装した。`commit_wait` は「commit＋カーネル専有＋
`waitUntilCompleted`」の合算値として扱う（§8）。

### 相互検証（`gemm_bench` example）

同一セッションで `cargo run -p fandhe-ai-backend-metal --example
gemm_bench --release` を実行し、`dynamic_tile_auto`（転送込み）の
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

**中央値の中央値**:

| N | upload_a+b | alloc_c | encode | commit_wait | readback | host_copy | sum |
|---|---|---|---|---|---|---|---|
| 1024 | 0.474 ms | ~0 ms | 0.0030 ms | 1.2449 ms | 0.0692 ms | 0.2397 ms | 2.0377 ms |
| 2048 | 1.973 ms | ~0 ms | 0.0065 ms | 2.5519 ms | 0.2380 ms | 0.9645 ms | 5.7155 ms |
| 4096 | 7.734 ms | ~0.0006 ms | 0.0167 ms | 15.7457 ms | 0.9412 ms | 3.7973 ms | 28.2347 ms |

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

## §5 突合（Layer A `matmul` vs Layer B Σ）

| N | Layer A `matmul` 中央値 | Layer B Σ 中央値 | 差分 |
|---|---|---|---|
| 1024 | 2.039 ms | 2.0377 ms | +0.1%（ほぼ一致） |
| 2048 | 5.549 ms | 5.7155 ms | -3.0%（Layer B がやや大きい） |
| 4096 | 34.800 ms | 28.2347 ms | +18.9%（Layer A が明確に大きい） |

N=1024/2048 は両者ほぼ一致し、Σ（upload_a+upload_b+alloc_c+encode+
commit_wait+readback+host_copy）が `matmul` 区間の内訳として妥当である
ことを裏付ける。**N=4096 のみ Layer A `matmul` が Layer B Σ より約
19% 大きい**——`docs/perf/metal-gemm-bottleneck-rediagnosis.md` §3.4
が記録する「プロセス間で中央値が最大約 3 倍変動する」計測揺らぎ、および
本ファイル§4 で観測した N=1024 の commit_wait 二峰性と同種の環境要因
（GPU クロック遷移・他プロセスとの競合等）が寄与している可能性が高いが
特定はできていない。この乖離は情報提供に留め、**固定費の帰属（§6）は
Layer A の `iter_total` 比のみから述べる**方針を維持する（実装計画§7
リスク節が明示的に許容する扱い）。

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
支配的ではない**。この違いは Metal の `matmul` 区間内部にすでに
「転送＋カーネル＋同期」の全コストが集約されており、CUDA のように
`clone_htod`/`clone_dtoh` が別ホスト API 呼び出しへ分離されず 1 回の
`ctx.dispatch_sync` 呼び出しに閉じているため（§2 の対応表参照）と考え
られる。

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
  分離する）は本イシューでは未実装（§2「Layer B」参照。AC-2 の安全側
  判断による意図的な縮退）。`commit_wait` は commit＋カーネル専有＋
  `waitUntilCompleted` の合算値のまま
- N=1024 の `commit_wait` 二峰性（run3/5 のみ約 1/3 に低下）・N=4096 の
  Layer A/B 乖離（§5）はいずれも原因未特定のまま記録するにとどめる
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
4. Layer A `matmul` と Layer B Σ は N=1024/2048 でほぼ一致するが、
   N=4096 のみ約 19% 乖離する（原因未特定の計測揺らぎ。§5）

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
