# sm_121（DGX Spark GB10）デバイス属性・L1/L2 実効帯域 実測記録（#482）

イシュー #482「spike(backend-cuda): sm_121 実機のデバイス属性（SMEM/SM・SM 数・レジスタ・L2）を実測記録」
（親 #480 Phase A・A-2）の実測記録テンプレート。実装変更を伴わない spike であり、後続の CUDA GEMM 最適化
（Phase C: 共有メモリ予算からのパイプライン段数逆算・#521／タイル候補列挙・#524／L2 スウィズル・#499 等）
が参照するコストモデル定数の実測材料を提供する。

## 目的・出典

DeepGEMM（`csrc/jit_kernels/heuristics/sm90.hpp`）が持つ SMEM 予算・L1/L2 帯域の per-cycle コストモデル
定数は Hopper（SM90）固有値であり、DGX Spark GB10（sm_121）へはそのまま流用できない。本ドキュメントは
`crates/backend-cuda/examples/device_attributes_dump.rs`（本イシューで新規追加）の出力を転記し、C-8/C-9
系タスクが参照する sm_121 実測値を 1 箇所に集約する。

計測環境: GB10・sm_121・driver 版・CUDA 13.0（`docs/real-hardware-verification-env.md` の実機。実ホスト名
は内部管理外ファイル `docs/real-hardware-verification-env.local.md` を参照し本ドキュメントには書かない）。

## ステータス: 未実測・要実機実行

**本 PR 時点では DGX Spark GB10 実機への接続情報（`docs/real-hardware-verification-env.local.md`）が本
作業環境に存在せず、実機実行ができなかった。** 実装計画（#482）§4 Step 3 の安全側フォールバック
「実機に接続できない場合は example とドキュメント骨子までを本 PR のスコープとし、実測値の記入が残る旨を
明記する」に従い、以下の表は骨子（属性名・単位・実測欄）のみを用意し、実測値は空欄のまま残す。

`docs/real-hardware-verification-env.md` の手順（`.rev-stamp` → rsync → SSH 実行 →
`cargo run -p backend-cuda --example device_attributes_dump --release` → 出力回収）に従って実機実行し、
以下の表を埋めること。

### 動作検証（sm_121 ではない代替 CUDA GPU 上での機能確認。参考値・DGX Spark GB10 の代替ではない）

`device_attributes_dump.rs` の実装自体は、本作業環境で検出可能だった別の CUDA GPU（NVIDIA GeForce RTX
3060・compute capability 8.6・**sm_121 ではない**）上で実行し、属性取得・帯域マイクロベンチともに
正常終了することを確認済み（コンパイル・実行の動作確認のみが目的。以下の数値は sm_121 の実測値としては
**使用しないこと**）。

**注意（Review 指摘対応・#482）**: 下記の出力例は `BW_LAUNCH_REPEATS`（起動・同期オーバーヘッド償却の
ための連続起動回数）導入**前**のコードで採取した記録であり、`l2` 行の帯域が `global` 行の帯域を下回る
（190.42 GB/s < 324.26 GB/s。L2 常駐コピーが global コピーより遅いのは物理的にありえない）非物理的な
結果になっている。これは `median_secs=0.000006`（6µs）という極小の計測区間でカーネル起動＋
`stream.synchronize()` のオーバーヘッドが支配的になり、帯域ではなくオーバーヘッドを計測していたことが
原因（下記「限界・注意」参照）。この事象を受けて `device_attributes_dump.rs::measure_bandwidth_secs` は
1 計測サンプルあたり `BW_LAUNCH_REPEATS` 回連続起動してから 1 回だけ同期する方式へ修正済みだが、
本作業環境には NVRTC（CUDA toolkit のコンパイラライブラリ）が存在せず本 PR 時点で再実行による出力更新
ができなかったため、旧コードでの出力例をそのまま「動作確認済み（コンパイル・実行が正常終了する）」の
記録として残す（帯域の数値自体は上記の理由で参考にしないこと）。修正後コードでの再実行・出力更新は
実機実行時（sm_121 実測時）にあわせて行うこと。

```
device: name=NVIDIA GeForce RTX 3060 compute_capability=(8, 6) arch=compute_86
  total_memory_bytes = 12490440704 (11.63 GiB)
  MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 101376
  MAX_SHARED_MEMORY_PER_BLOCK = 49152
  MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = 102400
  RESERVED_SHARED_MEMORY_PER_BLOCK = 1024
  MULTIPROCESSOR_COUNT = 28
  MAX_REGISTERS_PER_MULTIPROCESSOR = 65536
  MAX_REGISTERS_PER_BLOCK = 65536
  L2_CACHE_SIZE = 2359296
  CLOCK_RATE = 1792000
  MEMORY_CLOCK_RATE = 7501000
  GLOBAL_MEMORY_BUS_WIDTH = 192
  MAX_THREADS_PER_MULTIPROCESSOR = 1536
  MAX_THREADS_PER_BLOCK = 1024
global: n=67108864 median_secs=0.001656 bandwidth=324.26 GB/s bytes_per_cycle=180.9503
l2: n=147456 (src+dst=1179648 bytes, L2_CACHE_SIZE=Some(2359296) bytes) median_secs=0.000006 bandwidth=190.42 GB/s bytes_per_cycle=106.2608
```

## デバイス属性実測表（要実機記入）

| 属性名（`CUdevice_attribute`） | 実測値（生値） | 単位換算値 |
|---|---|---|
| `MAX_SHARED_MEMORY_PER_BLOCK_OPTIN` | 未実測 | bytes |
| `MAX_SHARED_MEMORY_PER_BLOCK` | 未実測 | bytes |
| `MAX_SHARED_MEMORY_PER_MULTIPROCESSOR` | 未実測 | bytes |
| `RESERVED_SHARED_MEMORY_PER_BLOCK` | 未実測 | bytes |
| `MULTIPROCESSOR_COUNT`（SM 数） | 未実測 | 個 |
| `MAX_REGISTERS_PER_MULTIPROCESSOR` | 未実測 | 32bit レジスタ数 |
| `MAX_REGISTERS_PER_BLOCK` | 未実測 | 32bit レジスタ数 |
| `L2_CACHE_SIZE` | 未実測 | bytes |
| `CLOCK_RATE` | 未実測 | kHz |
| `MEMORY_CLOCK_RATE` | 未実測 | kHz |
| `GLOBAL_MEMORY_BUS_WIDTH` | 未実測 | bit |
| `MAX_THREADS_PER_MULTIPROCESSOR` | 未実測 | スレッド数 |
| `MAX_THREADS_PER_BLOCK` | 未実測 | スレッド数 |
| デバイス名 | 未実測 | — |
| compute capability | 未実測 | (major, minor) |
| 総メモリ容量 | 未実測 | bytes |

## L1/L2/global 実効帯域（要実機記入）

`device_attributes_dump.rs` の grid-stride コピーカーネル（読み出し N・書き込み N の f32 トラフィック、
`bench_harness::run` warmup 20 回・計測 20 回の中央値。`.claude/rules/coding-rust.md`「ベンチは 5 回計測
の中央値」の下限を満たす）による実測。1 計測サンプルは `BW_LAUNCH_REPEATS`（64）回連続でカーネルを
起動してから 1 回だけ `stream.synchronize()` する構成であり（起動・同期オーバーヘッドの償却。下記
「限界・注意」参照）、表・出力の `secs_per_launch` は計測した中央値秒を `BW_LAUNCH_REPEATS` で割った
「起動 1 回あたり」の値。

| 区分 | バッファサイズ | 実効帯域（中央値） | bytes/cycle | 備考 |
|---|---|---|---|---|
| global（L2 超） | n=67108864（256 MiB/バッファ） | 未実測 GB/s | 未実測 | L2 非依存の参照帯域 |
| L2（L2 未満） | 未実測（`L2_CACHE_SIZE/4` 要素） | 未実測 GB/s | 未実測 | src+dst 合計が L2 に収まる設定 |
| L1（SM あたり） | — | スペック値＋出典を記録（下記参照） | — | 本バイナリでは未実装（下記「限界」参照） |

### L1 帯域: スペック値＋出典の記入欄

`device_attributes_dump.rs` は L1 単体の実効帯域を実測しない（下記「限界・注意」参照）。sm_121（Blackwell
系）の L1/shared memory 実効帯域について、NVIDIA 公式アーキテクチャホワイトペーパー等のスペック値と出典
（URL・版・該当節）をここに記入すること。DeepGEMM Hopper 実装（下記コストモデル定数表）が用いる
`128 bytes/cycle/SM` 相当の仮定と対比できる形（bytes/cycle/SM）で記録する。

- スペック値: 未記入
- 出典: 未記入

## コストモデル定数表（C-8/C-9 参照用）

DeepGEMM Hopper（SM90）定数と sm_121 実測値の対比表。**後続タスク（#521・#524 等）がコード定数化する際は
本表の値を正とする。本表の値を変更する場合は再実測を要す。**

| 定数 | DeepGEMM Hopper（SM90）値 | 出典 | sm_121 実測値 |
|---|---|---|---|
| SMEM 容量（`smem_capacity`） | 232448 bytes | DeepGEMM `csrc/jit_kernels/heuristics/sm90.hpp` 14 行付近 | 未実測（上表 `MAX_SHARED_MEMORY_PER_BLOCK_OPTIN`／`MAX_SHARED_MEMORY_PER_MULTIPROCESSOR` 参照） |
| L2 帯域（`l2_bandwidth_per_cycle` 相当） | Hopper 固有値（同ファイル 201-238 行付近） | 同上 | 未実測（上表「L2」行参照） |
| L1 帯域（per-SM per-cycle 相当） | Hopper 固有値（同ファイル 201-238 行付近） | 同上 | 未実測（スペック値＋出典欄参照） |
| SM 数 | Hopper 固有（機種依存） | — | 未実測 |

## 限界・注意

- **L1 帯域は本バイナリでは実測しない**: 1 SM を単独占有して L1 のみを計測する信頼できるマイクロベンチは
  ウォームアップ・占有率制御など実装コストが高く、本イシュー（4h 見積）を圧迫するため、受入基準が許容する
  「スペック値＋出典」側の記録に倒した（実装計画 §4 Step 2 の安全側フォールバック）。
- **L2 マイクロベンチの測定境界（Review 指摘対応・#482）**: バッファサイズを `L2_CACHE_SIZE` 未満に抑えて
  L2 常駐を狙うが、実際に全アクセスが L2 ヒットする保証はない（ウォームアップ・アクセスパターンに依存）。
  修正前のコードでは動作検証（上記「動作検証」節参照）で、バッファが小さくカーネル実行時間が数マイクロ秒台
  に留まるため、起動レイテンシ・完了待ちのオーバーヘッドが支配的になり L2 実効帯域が global 実効帯域を
  下回る非物理的な結果が出た。この問題に対応するため `measure_bandwidth_secs` は 1 計測サンプルあたり
  `BW_LAUNCH_REPEATS`（64）回連続でカーネルを起動してから 1 回だけ同期する構成へ修正済み（起動・同期
  コストを繰り返し回数で償却し実データ転送時間の比率を高める）。ただし L2 常駐そのものが保証される
  わけではない点は変わらないため、sm_121 実機での記入時は中央値だけでなく Q1/Q3
  （`bench_harness::Measurement`）や複数バッファサイズでの追試も検討すること。加えて、`L2_CACHE_SIZE`
  属性取得に失敗した場合のフォールバックバッファ（`global_n / 16`）での計測結果は L2 実測値として
  信頼できないため、出力ラベルを `l2_FALLBACK_SIZE_UNRELIABLE_DO_NOT_TRANSCRIBE` に区別している
  （このラベルの行は本表へ転記しないこと）。
- **global 測定もカーネル完了待ち込み**: `stream.synchronize()` を計測区間（`bench_harness::run` へ渡す
  クロージャ内）に含めている。含めない場合、カーネル起動は非同期のため見かけ上の帯域が実際より 1〜3 桁
  過大に出る（実装時に実際に踏んだ不具合。`device_attributes_dump.rs::measure_bandwidth_secs` のコメント
  参照）。
- 本イシューはコストモデル定数のコードへの組み込みを行わない（C-8/C-9・#521・#524 等のスコープ）。
- REQ-8 の性能下限・tolerance・ガードレール閾値には一切影響しない（計測記録のみ）。
