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

## ステータス: 部分実測完了（2026-08-19・commit `cbc16e7`）

**2026-08-19、DGX Spark GB10（sm_121）実機（commit `cbc16e7`）で `device_attributes_dump` を実行し、
イシュー #739 の受け入れ条件に記載された 5 属性（`MAX_SHARED_MEMORY_PER_BLOCK_OPTIN`・
`MAX_SHARED_MEMORY_PER_MULTIPROCESSOR`・`MULTIPROCESSOR_COUNT`・`MAX_REGISTERS_PER_MULTIPROCESSOR`・
`L2_CACHE_SIZE`）と global／L2 実効帯域の実測値を確定させた。** 転記元はイシュー #739 本文（複数の
兄弟イシュー #736・#740〜#743 本文の数値と相互一致することを確認済み）であり、**転記元に個別値の記載が
無い属性（`MAX_SHARED_MEMORY_PER_BLOCK`・`RESERVED_SHARED_MEMORY_PER_BLOCK`・`MAX_REGISTERS_PER_BLOCK`・
`CLOCK_RATE`・`MEMORY_CLOCK_RATE`・`GLOBAL_MEMORY_BUS_WIDTH`・`MAX_THREADS_PER_MULTIPROCESSOR`・
`MAX_THREADS_PER_BLOCK`・デバイス名・compute capability・総メモリ容量）は推定で埋めず「未実測」のまま
残す**（`out-of-scope-tracking.md`・捏造禁止の安全側フォールバック）。これら残欄は
`device_attributes_dump` の出力全文（生ログ）を実機セッションで回収し転記することで充足する。

`docs/real-hardware-verification-env.md` の手順（`.rev-stamp` → rsync → SSH 実行 →
`cargo run -p backend-cuda --example device_attributes_dump --release` → 出力回収）に従って実機実行し、
残る「未実測」欄を埋めること。

### 動作検証（sm_121 ではない代替 CUDA GPU 上での機能確認。参考値・DGX Spark GB10 の代替ではない）

`device_attributes_dump.rs` の実装自体は、本作業環境で検出可能だった別の CUDA GPU（NVIDIA GeForce RTX
3060・compute capability 8.6・**sm_121 ではない**）上で実行し、属性取得・帯域マイクロベンチともに
正常終了することを確認済み（コンパイル・実行の動作確認のみが目的。以下の数値は sm_121 の実測値としては
**使用しないこと**）。

**注意（Review 指摘対応・#482）**: 下記の出力例は `BW_LAUNCH_REPEATS`（カーネル**内部**での外側ループに
よるコピー反復回数。起動は 1 回のみで、この定数は連続起動回数ではない。下記「限界・注意」参照）導入
**前**のコードで採取した記録であり、`l2` 行の帯域が `global` 行の帯域を下回る
（190.42 GB/s < 324.26 GB/s。L2 常駐コピーが global コピーより遅いのは物理的にありえない）非物理的な
結果になっている。これは `median_secs=0.000006`（6µs）という極小の計測区間でカーネル起動＋
`stream.synchronize()` のオーバーヘッドが支配的になり、帯域ではなくオーバーヘッドを計測していたことが
原因（下記「限界・注意」参照）。この事象を受けて `device_attributes_dump.rs::measure_bandwidth_secs` は
カーネルを 1 回だけ起動し、コピーの `BW_LAUNCH_REPEATS` 回反復はカーネル**内部**の外側ループへ委ねてから
1 回だけ同期する方式（さらに反復ごとに `a`/`b` を読み出し役・書き込み役として入れ替える ping-pong 構成。
`BW_COPY_F32` ドキュメンテーションコメント参照。イシュー #482 codex-review 指摘・PR #635）へ修正済みだが、
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

## デバイス属性実測表（2026-08-19・GB10 実機・commit `cbc16e7`。部分実測。出典: イシュー #739）

| 属性名（`CUdevice_attribute`） | 実測値（生値） | 単位換算値 |
|---|---|---|
| `MAX_SHARED_MEMORY_PER_BLOCK_OPTIN` | 101376 | bytes |
| `MAX_SHARED_MEMORY_PER_BLOCK` | 未実測 | bytes |
| `MAX_SHARED_MEMORY_PER_MULTIPROCESSOR` | 102400 | bytes |
| `RESERVED_SHARED_MEMORY_PER_BLOCK` | 未実測 | bytes |
| `MULTIPROCESSOR_COUNT`（SM 数） | 48 | 個 |
| `MAX_REGISTERS_PER_MULTIPROCESSOR` | 65536 | 32bit レジスタ数 |
| `MAX_REGISTERS_PER_BLOCK` | 未実測 | 32bit レジスタ数 |
| `L2_CACHE_SIZE` | 25165824 | bytes |
| `CLOCK_RATE` | 未実測 | kHz |
| `MEMORY_CLOCK_RATE` | 未実測 | kHz |
| `GLOBAL_MEMORY_BUS_WIDTH` | 未実測 | bit |
| `MAX_THREADS_PER_MULTIPROCESSOR` | 未実測 | スレッド数 |
| `MAX_THREADS_PER_BLOCK` | 未実測 | スレッド数 |
| デバイス名 | 未実測 | — |
| compute capability | 未実測 | (major, minor) |
| 総メモリ容量 | 未実測 | bytes |

上記「未実測」の行は #739 実測作業では転記元（イシュー #739・#736・#740〜#743 本文）に個別値の記載が
無いため未実測のまま残す（推定値を書かない）。実機での `device_attributes_dump` 出力全文の回収により
充足する。

**`MULTIPROCESSOR_COUNT`（SM 数）48 の再確認（2026-08-20・イシュー #777）**: 上表の SM 数実測値 48 は、
2026-08-20 の GB10 実機再計測（main `0bca711` 時点）でベンチ起動診断からも再確認された。
`gemm_mma_swizzle_bench`（`crates/backend-cuda/examples/gemm_mma_swizzle_bench.rs` L117-120）は起動時に
`device.multiprocessor_count()` を実行時取得してログ出力しており、`num_sms=48` を出力した（#781
codex-review 指摘是正・この再確認当時点〈2026-08-20 GB10 再計測〉の記録: `cuda_floor_bench` は
`CudaMmaGemm::new` の `swizzle_group_width()` が実機検証未了のため常に `None` であることを診断するのみ
で、`multiprocessor_count()` の取得・`num_sms` のログ出力は行っていなかった。再確認元は
`gemm_mma_swizzle_bench` のみであり `cuda_floor_bench` は含まない）。**イシュー #782（2026-08-21 GB10 実機
受け入れゲート通過）で `CudaMmaGemm::new` へサイズ条件付き swizzle 選択機構を本番結線した後は、
`cuda_floor_bench` も `swizzle_group_width()`／`swizzle_applies()` の実測値を診断出力する（上記時点の
「`cuda_floor_bench` は診断しない」という記述はこの結線前の状態を指す。`docs/perf/
cuda-gemm-swizzle-ab.md` §6.2 参照）**。本番経路（`gemm_auto`・`swizzle`）は同じ実行時取得値を動的に
使うため、この再確認は実測値の裏付けであり値の変更・カーネル挙動の変更は伴わない。

## L1/L2/global 実効帯域（要実機記入）

`device_attributes_dump.rs` の grid-stride コピーカーネル（読み出し N・書き込み N の f32 トラフィック、
`bench_harness::run` warmup 20 回・計測 20 回の中央値。`.claude/rules/coding-rust.md`「ベンチは 5 回計測
の中央値」の下限を満たす）による実測。1 計測サンプルはカーネルを **1 回だけ起動**し、コピーの
`BW_LAUNCH_REPEATS`（64）回反復はカーネル**内部**の外側ループで行ってから 1 回だけ
`stream.synchronize()` する構成であり（起動・同期オーバーヘッドの償却。反復ごとに 2 バッファを
読み出し役・書き込み役として入れ替える ping-pong 構成にすることで、コンパイラによる冗長ストア除去で
反復が縮約されるのを防ぐ。`BW_COPY_F32` ドキュメンテーションコメント参照。下記「限界・注意」参照）、
表・出力の `secs_per_iter` は計測した中央値秒を `BW_LAUNCH_REPEATS` で割った「内部反復 1 回あたり」
の値（イシュー #482 Review 指摘: ラベルが誤って「1 回の起動あたり」を意味する `secs_per_launch` に
なっていたため `secs_per_iter` へ改称し、本文の説明と整合させた）。

| 区分 | バッファサイズ | 実効帯域（中央値） | bytes/cycle（device-wide） | 備考 |
|---|---|---|---|---|
| global（L2 超） | n=67108864（256 MiB/バッファ） | 212.34 GB/s | 未実測（`CLOCK_RATE` 未実測のため算出せず） | L2 非依存の参照帯域。出典: イシュー #739（`device_attributes_dump` 実行値）。`bytes_per_cycle_device_wide` は `CLOCK_RATE` が未実測のため未算出（推定計算をしない） |
| L2（L2 未満） | `L2_CACHE_SIZE/16` 要素（`device_attributes_dump.rs` の算出式 `l2_bytes / 4 / size_of::<f32>()`） | 1237.62 GB/s | 未実測（同上） | src+dst 合計が L2 に収まる設定。出典: イシュー #739 |
| L1（SM あたり） | — | スペック値＋出典を記録（下記参照） | — （per-SM。上 2 行とは基準が異なる） | 本バイナリでは未実装（下記「限界」参照） |

**転記前チェックの合否（2026-08-19 実測）**: 上記「転記前チェック」の条件（`l2` の実効帯域が `global`
の実効帯域を下回った場合は転記しない）を判定すると、1237.62 GB/s（L2）> 212.34 GB/s（global）で
**合格**しており、旧版で観測された非物理的な逆転（L2 < global）は解消している。よって上表への転記を
実施した。

**単位に関する注意（Review 指摘対応・#482）**: 上表の `bytes/cycle` 列は global/L2 行が **device-wide**
（デバイス全体の実効帯域をデバイスクロックで割った値。`device_attributes_dump.rs::bytes_per_cycle`
のドキュメンテーションコメント参照）、L1 行のみ **per-SM**（DeepGEMM の `128 bytes/cycle/SM` 相当と
対比するための単位）であり、同一列でも基準が異なる。global/L2 の実測値を per-SM 換算する場合は
`device_wide_value / SM 数`（上表 `MULTIPROCESSOR_COUNT` の実測値）で割ること。単位を揃えずに下記
コストモデル定数表へ転記しない。

**転記前チェック（Review 指摘対応・#482）**: `l2` の実効帯域が `global` の実効帯域を**下回った場合は
本表へ転記しない**こと。起動・同期オーバーヘッドがまだ支配的である兆候（`BW_LAUNCH_REPEATS` を増やして
再計測し、それでも下回るなら実測を保留し本ドキュメントの「限界・注意」に事象を追記のうえユーザーへ
相談する）。それでも解消しない場合は `volatile`（下記「限界・注意」の該当項目参照）が疑わしい原因の
1 つになりうる: `BW_COPY_F32` の `volatile` アクセスは L1／non-coherent キャッシュを経由させないため、
non-volatile 実装であれば `l2` 側が L1 ヒットにより本来より高い値を報告していた可能性があり、
`volatile` あり／なしの比較計測（`nvdisasm` でのロード／ストア命令確認込み）を追加の診断手順として
実施すること。

### L1 帯域: スペック値＋出典の記入欄

`device_attributes_dump.rs` は L1 単体の実効帯域を実測しない（下記「限界・注意」参照）。sm_121（Blackwell
系）の L1/shared memory 実効帯域について、NVIDIA 公式アーキテクチャホワイトペーパー等のスペック値と出典
（URL・版・該当節）をここに記入すること。DeepGEMM Hopper 実装（下記コストモデル定数表）が用いる
`128 bytes/cycle/SM` 相当の仮定と対比できる形（bytes/cycle/SM）で記録する。

- スペック値: 未記入（調査済み・未確定。下記「調査状況」参照）
- 出典: 未記入
- **調査状況（2026-08-15 実施）**: NVIDIA 公式の Blackwell/GB10 アーキテクチャホワイトペーパーおよび
  サードパーティ解析記事（chipsandcheese "Analyzing Nvidia GB10's GPU"・"Blackwell: Nvidia's Massive GPU"
  等）を確認したが、L1/shared memory の**容量**（128 KiB/SM〈GPU 側〉、SoC 全体では最大 192 KiB/SM 相当）
  ・ヒットレイテンシ（約 30〜40 サイクル）の記載はあるものの、DeepGEMM の `128 bytes/cycle/SM` 相当と
  対比可能な **L1 実効帯域（bytes/cycle/SM）を明記した一次情報源は見つからなかった**。この数値は
  一般に公開されていない可能性が高く、実装計画の安全側フォールバック（推定値を書かない）に従い、
  推測での穴埋めは行わない。sm_121 実機アクセス時にマイクロベンチ（1 SM 占有の L1 限定測定。上記
  「限界・注意」参照）で実測するか、NVIDIA から追加のアーキテクチャ資料が公開された場合に転記すること。

## コストモデル定数表（C-8/C-9 参照用）

DeepGEMM Hopper（SM90）定数と sm_121 実測値の対比表。**後続タスク（#521・#524 等）がコード定数化する際は
本表の値を正とする。本表の値を変更する場合は再実測を要す。**

| 定数 | DeepGEMM Hopper（SM90）値 | 出典 | sm_121 実測値 |
|---|---|---|---|
| SMEM 容量（`smem_capacity`） | 232448 bytes | DeepGEMM `csrc/jit_kernels/heuristics/sm90.hpp` 14 行付近 | `MAX_SHARED_MEMORY_PER_BLOCK_OPTIN`＝101376 bytes／`MAX_SHARED_MEMORY_PER_MULTIPROCESSOR`＝102400 bytes（2026-08-19 実測。出典: イシュー #739） |
| L2 帯域（`l2_bandwidth_per_cycle` 相当） | Hopper 固有値（同ファイル 201-238 行付近） | 同上 | 1237.62 GB/s（**device-wide**。DeepGEMM 側定数の単位基準〈device-wide か per-SM か〉は本ドキュメントでは未確認のため、転記時に両者の基準を揃えること。上記「単位に関する注意」参照。出典: イシュー #739） |
| L1 帯域（per-SM per-cycle 相当） | Hopper 固有値（同ファイル 201-238 行付近） | 同上 | 未実測（スペック値＋出典欄参照。per-SM） |
| SM 数 | Hopper 固有（機種依存） | — | 48（2026-08-19 実測。出典: イシュー #739。2026-08-20 ベンチ起動診断の `num_sms=48` 出力で再確認。出典: イシュー #777） |

**C-8（#521）注記**: 本表の sm_121 SMEM 容量は上記のとおり 2026-08-19 に実機実測済み
（`MAX_SHARED_MEMORY_PER_BLOCK_OPTIN`＝101376 bytes／`MAX_SHARED_MEMORY_PER_MULTIPROCESSOR`＝
102400 bytes。出典: イシュー #739）であり、DeepGEMM の Hopper 固有値（232448）を sm_121 向けに
流用・推定で定数化することはしない。C-8 の `derive_pipeline_stages`
（`crates/backend-cuda/src/nvrtc.rs`）は SMEM 容量をコード定数として持たず、
`gemm_auto::derive_stages_for_device` が `device.context().attribute(
CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)` で**実行時に**取得した値を（静的
`__shared__` 構成の per-block 上限 49,152 バイトでクランプしたうえで）渡す方式を採る。ただし
`CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK`（非 OPTIN・静的 `__shared__` の既定上限）自体は
上表のとおり本ドキュメントでは依然「未実測」であり、`MAX_SHARED_MEMORY_PER_BLOCK_OPTIN`（動的確保
opt-in 時の上限。両者は異なる属性で非 OPTIN の方が通常小さい）とは別物のため両者を混同しないこと。
実機で `MAX_SHARED_MEMORY_PER_BLOCK` を実測記入する際は、実機上の取得値が上表の
`MAX_SHARED_MEMORY_PER_BLOCK_OPTIN`（101376 bytes）以下であることを確認すること（超過は属性クエリ側
かドキュメント記載側のいずれかに誤りがあることを示す）。

## 限界・注意

- **L1 帯域は本バイナリでは実測しない**: 1 SM を単独占有して L1 のみを計測する信頼できるマイクロベンチは
  ウォームアップ・占有率制御など実装コストが高く、本イシュー（4h 見積）を圧迫するため、受入基準が許容する
  「スペック値＋出典」側の記録に倒した（実装計画 §4 Step 2 の安全側フォールバック）。
- **L2 マイクロベンチの測定境界（Review 指摘対応・#482）**: バッファサイズを `L2_CACHE_SIZE` 未満に抑えて
  L2 常駐を狙うが、実際に全アクセスが L2 ヒットする保証はない（ウォームアップ・アクセスパターンに依存）。
  修正前のコードでは動作検証（上記「動作検証」節参照）で、バッファが小さくカーネル実行時間が数マイクロ秒台
  に留まるため、起動レイテンシ・完了待ちのオーバーヘッドが支配的になり L2 実効帯域が global 実効帯域を
  下回る非物理的な結果が出た。この問題に対応するため `measure_bandwidth_secs` はカーネルを 1 回だけ
  起動し、コピーの `BW_LAUNCH_REPEATS`（64）回反復をカーネル内部の外側ループへ委ねてから 1 回だけ
  同期する構成へ修正済み（起動・同期コストを繰り返し回数で償却し実データ転送時間の比率を高める。
  さらに反復ごとに 2 バッファを読み出し役・書き込み役として入れ替える ping-pong 構成にすることで、
  各反復が同一アドレスへ同一値を書くだけのループ不変ストアと化してコンパイラの冗長ストア除去で
  縮約されるのを防いでいる。`BW_COPY_F32` ドキュメンテーションコメント参照。イシュー #482
  codex-review 指摘・PR #635）。ただし L2 常駐そのものが保証される
  わけではない点は変わらないため、sm_121 実機での記入時は中央値だけでなく Q1/Q3
  （`bench_harness::Measurement`）や複数バッファサイズでの追試も検討すること。加えて、`L2_CACHE_SIZE`
  属性取得に失敗した場合のフォールバックバッファ（`global_n / 16`）での計測結果は L2 実測値として
  信頼できないため、出力ラベルを `l2_FALLBACK_SIZE_UNRELIABLE_DO_NOT_TRANSCRIBE` に区別している
  （このラベルの行は本表へ転記しないこと）。
- **global 測定もカーネル完了待ち込み**: `stream.synchronize()` を計測区間（`bench_harness::run` へ渡す
  クロージャ内）に含めている。含めない場合、カーネル起動は非同期のため見かけ上の帯域が実際より 1〜3 桁
  過大に出る（実装時に実際に踏んだ不具合。`device_attributes_dump.rs::measure_bandwidth_secs` のコメント
  参照）。
- **`volatile` による過小評価方向の歪み（Review 指摘対応・#482）**: `BW_COPY_F32` は上記の冗長ストア除去
  対策として `a`/`b` を `volatile` ポインタ経由でアクセスするが、`volatile` はベクトル化ロード／ストアや
  レジスタ経由の最適化も全面的に禁止するため、本表の `global`/`l2` 実効帯域は実ハードウェアの真の実効
  帯域より**主に過小評価する方向へ**歪みうる（`BW_COPY_F32` ドキュメンテーションコメント「`volatile` の
  トレードオフ」参照）。ping-pong による RAW 依存だけでも冗長ストア除去を防ぐには十分である可能性が高く、
  `volatile` はその上に安全側で追加した保護のため、この歪みは過大評価方向（64 倍バグ）より安全ではある
  が、ゼロではない。ただし `l2` 行は事情が単純ではない: CUDA の `volatile` は L1／non-coherent
  キャッシュも経由させないため、non-volatile 実装であれば L1 ヒットにより本来の L2 帯域より**過大**な
  値を報告していた可能性もあり、歪みの向きは `global` 行ほど一意ではない（L1 バイパスと冗長ストア除去
  防止が同時に効くため）。**転記時は `global` 行の実測値をハードウェアの実効帯域の保守的な下限値寄り、
  `l2` 行の実測値を参考値として扱うこと。** 本 PR 時点では NVRTC が利用可能な環境がなく `volatile`
  あり／なしの比較計測ができなかったため、sm_121 実機実行時に両版を比較計測し、`volatile` を安全に
  外せるかを検証すること（検証手順は `BW_COPY_F32` ドキュメンテーションコメント参照。上記
  「転記前チェック」の追加診断手順も参照）。
- 本イシューはコストモデル定数のコードへの組み込みを行わない（C-8/C-9・#521・#524 等のスコープ）。
- REQ-8 の性能下限・tolerance・ガードレール閾値には一切影響しない（計測記録のみ）。
- **残課題（2026-08-19 時点）**: 上記「転記前チェック」が要求する `volatile` あり／なしの比較計測、
  および複数バッファサイズでの追試は、#739 実測セッションでは実施した証跡がイシュー本文に無いため
  未実施として扱う。次回実機セッションで実施し本ドキュメントへ追記すること。同様に `MAX_SHARED_MEMORY_PER_BLOCK`・
  `RESERVED_SHARED_MEMORY_PER_BLOCK`・`MAX_REGISTERS_PER_BLOCK`・`CLOCK_RATE`・`MEMORY_CLOCK_RATE`・
  `GLOBAL_MEMORY_BUS_WIDTH`・`MAX_THREADS_PER_MULTIPROCESSOR`・`MAX_THREADS_PER_BLOCK`・デバイス名・
  compute capability・総メモリ容量（上表「未実測」欄）は `device_attributes_dump` 出力全文の回収により
  充足する残課題。
