# 初期化除外後の CUDA GEMM カーネル単体性能ベースライン記録（vs candle / Burn）（#928）

イシュー #928「初期化除外後の CUDA GEMM カーネル単体性能ベースライン計測（vs candle / Burn）」の
記録。ルート #920・親 #921 配下。#925（`bench-fandhe` の `--mode reuse` 実装。PR #944 でマージ済み）・
#926（tape 初期化コスト内訳診断。PR #945 でマージ済み）の後続として、fandhe-ai の CUDA GEMM を
**初期化コストを含まないカーネル単体性能**として candle / Burn・REQ-8 下限と突合する。

## 状態: 受け入れ条件は部分充足（N=512・1024・2048・4096 は充足。N=256 のみカーネル単体データが欠落。GB10 新規実測は未実施）

本ドキュメント作成時点で `docs/real-hardware-verification-env.local.md`（実機ホスト名のローカル
管理ファイル）が実装セッションに存在せず、DGX Spark GB10 実機へは SSH 到達不能だった。実測値の
捏造は行わない方針（PR #713 で確立）に従い、GB10 での `--mode reuse` 新規実測は実施していない。
受け入れ条件（N=256〜4096 の分離計測記録・vs candle/Burn 比較・REQ-8 下限との突合）のうち、
「N=256〜4096 の分離計測記録」は**既存コミット済み一次データ**（環境 2: DGX Spark GB10 の
candle/Burn 実測、環境 3: RTX 3060 の fresh/reuse 分離実測）の突合で全範囲を充足できる（§3）。
一方「vs candle/Burn 比較」「REQ-8 下限との突合」を**カーネル実行時間そのもの**（tape・ホスト実体化
を含まない launch-only 計測）で行うには `cuda_floor_bench` の実測値が要る。既存コミット済みの
`cuda-optimized-remeasurement.md` は `cuda_floor_bench` の `JUDGED_SIZES`（M=N=K=2048・4096。
REQ-8 判定対象）に加え `REFERENCE_ONLY_SIZES`（512・1024。判定対象外の参考値）も計測対象と
しており（`crates/backend-cuda/examples/cuda_floor_bench.rs:144,153`）、**N=512・1024 のカーネル
単体（`cuda_floor_bench`）実測値も既に存在する**（§4・§5 に追加転記）。したがって当初の想定
（N=256〜1024 が丸ごと欠落）は誤りであり、実際に欠落しているのは **N=256 のみ**である。
`cuda_floor_bench` は `JUDGED_SIZES`／`REFERENCE_ONLY_SIZES` とも定数配列に固定されたサイズ集合
（512・1024・2048・4096）を計測する実装であり、CLI 引数でサイズを指定する経路を持たないため、
N=256 の値を得るには同バイナリへ 256 を計測対象として追加するコード変更が必要になる（§6.1・§7）。
§3 の N=256〜1024 fresh/reuse 実測値は §2 の定義どおり tape 構築・per-call 固定オーバーヘッドを
含むため本文自身がカーネル実行時間とはみなせないと説明しており、これをカーネル単体の代用とする
ことはできない。したがって本ドキュメントは**N=512・1024・2048・4096 についてカーネル単体の
vs candle/Burn 比較・REQ-8 下限突合を充足**し、N=256 についてのみカーネル単体の受け入れ条件を
**未充足のまま §7 にスコープ外として記録**する（`cuda_floor_bench` への 256 追加というコード変更、
および GB10 実機での該当サイズ新規実測が必要。§6.1 に再現手順を残し実機セッションへ引き継ぐ）。

## 1. 目的・位置づけ

- **目的**: fandhe-ai の CUDA GEMM について、tape 初期化コスト（約 440〜460 ms。#926 診断対象）を
  除いた「カーネル実行そのものの性能」を candle / Burn の CUDA GEMM 実測、および REQ-8 の CUDA f32
  下限（`docs/performance-targets.md` §2）と突合し、「fandhe-ai の CUDA カーネルは他フレームワーク・
  REQ-8 下限に対しどの水準にあるか」を確定記録する
- **#925 との関係**: `bench-fandhe --mode reuse`（gemm タスクのみ）により tape 構築を `init_s` として
  分離計測する手段が確立した（PR #944・コミット `d156e72`）。ただし環境 3（RTX 3060）の実測では reuse
  モードでも per-call 数百 ms の固定オーバーヘッドが残ることが判明しており（§3）、`--mode reuse` の
  数値それ自体はまだ「カーネル実行時間」とみなせない。本ドキュメントのカーネル単体実測（§4）は
  `--mode reuse` ではなく `cuda_floor_bench`（backend API 直・launch-only 同期。REQ-8 用に別途整備済み）
  の実測値を採用する
- **#926 との関係**: tape 初期化コスト（約 440〜460 ms）の内訳診断（`docs/perf/cuda-tape-init-cost-diagnosis.md`）
  は実機未到達のため静的分析までで止まっている。本ドキュメントは初期化コストの内訳ではなく、
  初期化コストを含まない実行時間側（カーネル単体・reuse per-call）の比較に焦点を当てる

## 2. 計測境界の定義（3 層の区別）

同じ「CUDA GEMM の実行時間」でも計測境界が異なる 3 層のデータが存在する。混同を避けるため定義する。

| 層 | 内容 | 含むもの | 出典 |
| --- | --- | --- | --- |
| (a) fresh | `bench-fandhe --mode fresh`（既定）。ループ毎回 `tape_for(Device::Cuda(0))` を新規構築 | tape/デバイス初期化 + カーネル実行 + ホスト実体化 | `scripts/bench/framework-compare/results/summary.md` 環境 2・環境 3 |
| (b) reuse per-call | `bench-fandhe --mode reuse`（#925）。tape 構築を `init_s` として分離し、2 回目以降の呼び出しを計測 | tape 経由 matmul 呼び出し 1 回（初期化は含まないはずだが、環境 3 実測では per-call 固定オーバーヘッドが残存） + ホスト実体化 | `scripts/bench/framework-compare/results/summary.md` 環境 3（`results/raw/results-rtx3060.jsonl`） |
| (c) カーネル単体 | `cuda_floor_bench`（`crates/backend-cuda/examples/cuda_floor_bench.rs`）。backend API を直接呼び launch-only 同期境界で計測 | カーネル launch + 完了同期のみ（tape・ホスト実体化を含まない） | `docs/perf/cuda-optimized-remeasurement.md`（#571・PR #725） |

candle / Burn は環境 2・環境 3 いずれも「デバイス・入力テンソルをループ外で 1 回構築し、matmul +
ホスト実体化のみを計測」するため、その実測値は (a) fresh の中でも (b)・(c) に近い性質を持つ
（tape/デバイス構築を含まない）。ただし (c) のような launch-only 同期ではなくホスト実体化を含む点は
fandhe-ai の (c) 実測と異なる（§4 の限定条件で明記）。

## 3. N=256〜4096 の分離計測記録

### 3.1 環境 3（RTX 3060）: fresh vs reuse の実測

`scripts/bench/framework-compare/results/summary.md` 環境 3（計測日 2026-08-28、
`results/raw/results-rtx3060.jsonl`）より転記。

| N | mode | 初期化(init_s) | 中央値 | GFLOP/s |
| --- | --- | --- | --- | --- |
| 256 | fresh | - | 267.254 ms | 0.1 |
| 256 | reuse | 477.869 ms | 295.149 ms | 0.1 |
| 512 | fresh | - | 333.016 ms | 0.8 |
| 512 | reuse | 484.626 ms | 260.525 ms | 1.0 |
| 1024 | fresh | - | 275.626 ms | 7.8 |
| 1024 | reuse | 465.571 ms | 300.868 ms | 7.1 |
| 2048 | fresh | - | 291.567 ms | 58.9 |
| 2048 | reuse | 573.555 ms | 297.019 ms | 57.8 |
| 4096 | fresh | - | 506.611 ms | 271.3 |
| 4096 | reuse | 684.670 ms | 474.396 ms | 289.7 |

**観察**: reuse モードの `init_s`（tape 構築 + 初回 matmul + ホスト実体化）は 465〜685 ms とサイズ
非依存の大きな値であり、かつ 2 回目以降の呼び出し（中央値列）も 260〜474 ms とほぼ同水準にとどまる。
すなわち本環境では「tape 構築 1 回限りで以降はカーネル実行のみの短時間になる」という当初仮説
（#925 計画時点の想定）は成立せず、**行列積 1 回ごとに数百 ms の固定オーバーヘッドが繰り返し発生する**
ことが実測された（詳細な原因切り分けは未実施。§7 参照）。

### 3.2 環境 2（DGX Spark GB10）: fresh のみ（既存実測）

同じ summary.md 環境 2 より、fandhe-ai fresh モードと candle / Burn の実測（計測日 2026-08-28）。

| N | フレームワーク | 中央値 | GFLOP/s |
| --- | --- | --- | --- |
| 256 | fandhe-ai (fresh) | 440.042 ms | 0.1 |
| 256 | candle | 79.3 µs | 423.0 |
| 256 | burn | 381.4 µs | 88.0 |
| 512 | fandhe-ai (fresh) | 450.692 ms | 0.6 |
| 512 | candle | 242.0 µs | 1109.3 |
| 512 | burn | 314.1 µs | 854.6 |
| 1024 | fandhe-ai (fresh) | 435.171 ms | 4.9 |
| 1024 | candle | 946.4 µs | 2269.1 |
| 1024 | burn | 1.098 ms | 1956.4 |
| 2048 | fandhe-ai (fresh) | 458.350 ms | 37.5 |
| 2048 | candle | 4.086 ms | 4204.3 |
| 2048 | burn | 4.096 ms | 4194.7 |
| 4096 | fandhe-ai (fresh) | 593.890 ms | 231.4 |
| 4096 | candle | 60.676 ms | 2265.1 |
| 4096 | burn | 46.819 ms | 2935.5 |

GB10 では fandhe-ai の `--mode reuse` 実測は未実施（本ドキュメント作成時点で GB10 到達不能）。
再現手順・再計測キャンペーン表は §6.2 に記録する。

## 4. vs candle / Burn（GB10・既存一次データ突合。カーネル単体。N=512・1024・2048・4096）

fandhe-ai の CUDA GEMM「カーネル単体」実測は `docs/perf/cuda-optimized-remeasurement.md`（#571・
PR #725、計測日 2026-08-18。同ドキュメント内の PyTorch 参照値・`cuda_floor_bench` 実測は同一計測
セッション内で同一 GB10 個体上のもの）にある。これと環境 2 の candle / Burn 実測（fresh。
`scripts/bench/framework-compare/results/summary.md`、計測日 2026-08-28。tape/デバイス構築を
含まないためカーネル実行に近い）を GFLOP/s 換算で突合する。**両者は計測日・計測セッションが異なる
別実測であり、GB10 個体が同一であることは確認できていない**（framework-compare 側の一次記録は
実ホスト名を伏せた「内部クラスタの 1 ノード」としか記録しておらず、`cuda-optimized-remeasurement.md`
側の実機個体と照合できる provenance が無い。GPU ドライバ版数も前者 580.159.03・後者 580.173.02 と
異なる）。したがって本突合は「同一 GPU 型（NVIDIA GB10）だが個体は未確認のクロスセッション比較」
として解釈する必要がある（限定条件 7 参照）。

| N | 経路 | fandhe-ai（カーネル単体） | candle（fresh） | burn（fresh） | fandhe-ai / candle | fandhe-ai / burn |
| --- | --- | --- | --- | --- | --- | --- |
| 512（参考値） | wmma_tf32（f32 最良経路） | 8268.7 GFLOP/s (8.2687 TFLOPS) | 1109.3 GFLOP/s | 854.6 GFLOP/s | **約 7.45 倍** | **約 9.68 倍** |
| 512（参考値） | tiled f32（基準経路） | 2089.6 GFLOP/s (2.0896 TFLOPS) | 1109.3 GFLOP/s | 854.6 GFLOP/s | **約 1.88 倍** | **約 2.45 倍** |
| 1024（参考値） | wmma_tf32（f32 最良経路） | 12645.3 GFLOP/s (12.6453 TFLOPS) | 2269.1 GFLOP/s | 1956.4 GFLOP/s | **約 5.57 倍** | **約 6.46 倍** |
| 1024（参考値） | tiled f32（基準経路） | 2383.0 GFLOP/s (2.3830 TFLOPS) | 2269.1 GFLOP/s | 1956.4 GFLOP/s | **約 1.05 倍** | **約 1.22 倍** |
| 2048 | wmma_tf32（f32 最良経路） | 14332.6 GFLOP/s (14.3326 TFLOPS) | 4204.3 GFLOP/s | 4194.7 GFLOP/s | **約 3.41 倍** | **約 3.42 倍** |
| 2048 | tiled f32（基準経路） | 2342.5 GFLOP/s (2.3425 TFLOPS) | 4204.3 GFLOP/s | 4194.7 GFLOP/s | 約 0.56 倍 | 約 0.56 倍 |
| 4096 | wmma_tf32（f32 最良経路） | 9065.5 GFLOP/s (9.0655 TFLOPS) | 2265.1 GFLOP/s | 2935.5 GFLOP/s | **約 4.00 倍** | **約 3.09 倍** |
| 4096 | tiled f32（基準経路） | 1972.3 GFLOP/s (1.9723 TFLOPS) | 2265.1 GFLOP/s | 2935.5 GFLOP/s | 約 0.87 倍 | 約 0.67 倍 |

（倍率は `wmma_tf32(or tiled_f32) GFLOP/s ÷ candle(or burn) GFLOP/s` を python で機械計算。例:
2048 wmma_tf32/candle = 14332.6 / 4204.3 ≈ 3.409。512・1024 行（**参考値**）は `cuda_floor_bench` の
`REFERENCE_ONLY_SIZES` 実測値（`cuda-optimized-remeasurement.md` 242〜243 行）であり、REQ-8 の
判定対象形状（`JUDGED_SIZES` = 2048・4096）には含まれない。tiled f32 行が示すとおり、公開 API が
実際に実行する経路（tiled f32 固定。限定条件 4）の candle/burn 対比は **N=512・1024 では上回り
（約 1.05〜2.45 倍）、N=2048・4096 では下回る（約 0.56〜0.87 倍）**というサイズ依存の傾向であり、
「常に candle/Burn 未満」ではない点に注意する）

**限定条件（比較の解釈に必須）**:

1. **計測境界差**: fandhe-ai の値は `cuda_floor_bench` の launch-only 同期（カーネル完了待ちのみ）、
   candle/burn は matmul + ホスト実体化を含む（§2 表）。ホスト実体化コストは行列サイズに対し
   カーネル実行より小さいと考えられるが未定量化であり、上記倍率は「同一境界での比較」ではない
2. **バージョン差**: fandhe-ai 側は crates.io 公開版（v0.3.0）ではなく本リポジトリ実装（#571 実測時点、
   最適化後カーネル）の実測。framework-compare（§3）の fandhe-ai facade は v0.3.0 のため、同一
   バージョンでの直接比較ではない
3. **TF32 使用有無未確認**: 環境 2 の備考（summary.md）で candle/burn の CUDA GEMM checksum に
   下位桁のずれがあり、TF32 等の低精度アキュムレーションを使っている可能性があるが未確認。
   fandhe-ai の wmma_tf32 経路も TF32 入力変換を伴うため、両者が同種の精度トレードオフを取っている
   可能性がある一方、confirmed ではない
4. **【最重要】既定カーネル変種は tiled f32 固定であり、上表 wmma_tf32 行は公開 API が実際に通る経路
   ではない**: `crates/backend-cuda/src/ops.rs`（`CudaBackendOps::gemm`, `run_tiled_f32` 呼び出し
   1 箇所のみ）を確認したところ、facade → tape → backend-cuda の公開 API 経由の matmul（framework-compare
   の fandhe-ai 実測が通る経路）は `CudaGemm::run_tiled_f32` に固定されている。上表 wmma_tf32 行の
   実測値は `cuda_floor_bench`（`crates/backend-cuda/examples/cuda_floor_bench.rs`）が `CudaGemm` の
   計測 API（`launch_wmma_tf32` 等）を直接呼ぶ経路によるもので、`CudaGemmAuto`（`crates/backend-cuda/src/gemm_auto.rs`）
   を経由しない（`cuda-optimized-remeasurement.md`「役割分担」節の記述どおり、`CudaGemmAuto` は
   `cuda_floor_bench.rs` からも既定ビルドからも呼ばれていない）。また現行の `CudaGemmAuto::run_f32`
   （`gemm_auto.rs:1615-1631`）は `KernelKind::MatrixUnit`・`KernelKind::Tiled` のいずれも
   `run_tiled_f32` へ委譲する実装（TF32/f32 Tensor Core 経路は #62 未実装のためコメントで明記の上
   tiled へフォールバック）であり、**`CudaGemmAuto` を経由しても f32 の Tensor Core 経路
   （`wmma_tf32`）へは現状到達できない**。したがって「`CudaGemmAuto` を公開 API へ接続すれば
   Tensor Core 経路へ切り替わる」は成立せず、公開 API を wmma_tf32 経路へ切り替えるには単なる
   `CudaGemmAuto` 接続ではなく、`CudaGemmAuto::run_f32` 側に f32 の `MatrixUnit` 分岐を実装する
   こと・カーネル選択契約（`select_gemm_kernel`）の見直し・§4 限定条件の parity 検証を伴う変更が
   必要である。**すなわち上表の tiled f32 行こそが「公開 API が実際に実行するカーネルの性能」であり、
   wmma_tf32 行（512: 約 7.45〜9.68 倍、1024: 約 5.57〜6.46 倍、2048: 約 3.41〜3.42 倍、4096: 約 3.09〜
   4.00 倍。いずれも優位）は公開 API 経由では享受できない未接続の候補経路の実測値である**。
   tiled f32 行自体の candle/Burn 対比は**サイズ依存**であり一様に「未満」ではない点に注意する
   （512: 約 1.88〜2.45 倍・1024: 約 1.05〜1.22 倍で candle/Burn を上回る一方、2048: 約 0.56 倍・
   4096: 約 0.87〜0.67 倍で下回る。§5 の REQ-8 判定対象形状は 2048・4096 に限定されるため、REQ-8
   突合の文脈では「下回る」側が該当する）
5. **tape 経由 reuse per-call とカーネル単体の乖離**: §3.1 の環境 3 実測が示すとおり、fandhe-ai の
   公開 API（tape）経由では reuse per-call でも数百 ms の固定オーバーヘッドが残る。上記限定条件 4
   により、この乖離を解消しても N=2048・4096 で到達するのは tiled f32（candle/burn 未満）であり wmma_tf32
   （3.09〜4.00 倍優位）ではない点に注意する
6. **wmma_tf32 実測値（51.96%／83.53%）は `wmma_tf32_staged` 経路の値であり、この経路固有の
   数値一致（parity）状態には経緯がある**: `cuda-optimized-remeasurement.md`「数値一致（parity）
   状態の限定条件」節は当初（2026-08-18 実測時点）`wmma_tf32_staged`（512×512×4096）を
   `baseline_provenance_unconfirmed == true` により **判定不能（fail-closed）** と記録していたが、
   同ドキュメント追記（イシュー #726・2026-08-19）および正本 `docs/perf/performance-floor-decision.md`
   §10 限定条件 4 のとおり、DGX Spark GB10 実機で staged 固有の確定ベースラインを確立し
   （fail_count=43019/262144・mean_abs_diff=4.463436e-3）、`parity_baselines_do_not_regress` が
   staged 行を含む全対象行で pass することを確認済みであるため**この判定不能状態は解消済み**である。
   ただし `performance-floor-decision.md` §10 が明記するとおり限定条件 1〜3 は #726 のスコープ外で
   継続する: (a) 候補算出経路（`wmma_tf32`・`mma_f16`）は #389 §5.3 が記録した数値一致 parity の
   恒常 fail 対象と一致する（`assert_parity` による絶対比較では K=4096 ストレス等で 16〜17% 台の
   fail が既知事象として残る）、(b) TF32/f16 Tensor Core 経路の複合判定改定（REQ-2 改定）は
   #186 close 後も閾値定数自体は変更されておらず spec リポジトリ側対応待ちのまま、(c) 50% の
   採用は「実測基準でゲートを機能させ、今後の最適化で性能を改善していく」という 2026-08-18
   ユーザー承認済みの方針判断であること。したがって §5 の候補下限突合・優位性比較は**継続する
   限定条件 1〜3 付きの承認済み候補値**に基づくものであり、無条件に確定した性能値の比較ではない
7. **クロスセッション比較で GB10 個体の同一性は未確認**: §4 冒頭のとおり、`cuda-optimized-remeasurement.md`
   （2026-08-18 計測）と framework-compare（2026-08-28 計測）は別々の計測セッションの一次記録であり、
   同一 GB10 個体であることを確認できる識別子（実ホスト名等）が framework-compare 側に残っていない
   （`docs/real-hardware-verification-env.md` の運用によりホスト名はローカル管理でドキュメントへ
   書かない方針のため）。GPU ドライバ版数も 580.159.03（前者）と 580.173.02（後者）で異なり、同一個体か
   別個体かのいずれとも断定できない。したがって §4・§5 の比較は「同一 GPU 型（GB10）だが個体は
   未確認のクロスセッション比較」として扱う。個体同一性を確認するには、両セッションの一次記録に
   共通の個体識別子（`nvidia-smi -q` の GPU UUID 等）を追記する実機セッションでの追試が必要
   （§7 のスコープ外事項として記録）

## 5. REQ-8 下限との突合

`docs/performance-targets.md` §2 は CUDA f32 最適化後下限を **50%**（対 PyTorch、size=4096 が最小、
`wmma_tf32` 実測 51.96%）と確定している。`cuda-optimized-remeasurement.md`「PyTorch 参照値の再集計」
節の PyTorch f32 参照値（4096: 17.4467 TFLOPS、5 run 中央値。同ドキュメント内の同一計測セッション・
同一 GB10 個体上の値）を分母に、candle / Burn の対 PyTorch 比を同一形式で並べる。**この分母
（2026-08-18 計測）と candle/Burn の分子（framework-compare、2026-08-28 計測）は限定条件 7 のとおり
別セッションの実測であり GB10 個体の同一性が未確認のクロスセッション比較である**点に注意する。

| N | 経路 | 対 PyTorch 比 | 出典 |
| --- | --- | --- | --- |
| 4096 | fandhe-ai wmma_tf32（カーネル単体） | **51.96%**（REQ-8 f32 下限 50% の根拠） | `cuda-optimized-remeasurement.md` |
| 4096 | candle（fresh、GB10 環境 2） | 12.98% | 本ドキュメント §4 換算（2265.1 / 17446.7 GFLOP/s） |
| 4096 | burn（fresh、GB10 環境 2） | 16.83% | 同上（2935.5 / 17446.7 GFLOP/s） |
| 2048 | fandhe-ai wmma_tf32（カーネル単体） | 83.53% | `cuda-optimized-remeasurement.md` |
| 2048 | candle（fresh、GB10 環境 2） | 24.50% | 本ドキュメント §4 換算（4204.3 / 17158.2 GFLOP/s） |
| 2048 | burn（fresh、GB10 環境 2） | 24.45% | 同上（4194.7 / 17158.2 GFLOP/s） |
| 1024（参考値・REQ-8 判定対象外） | fandhe-ai wmma_tf32（カーネル単体） | 80.80% | `cuda-optimized-remeasurement.md` |
| 1024（参考値・REQ-8 判定対象外） | candle（fresh、GB10 環境 2） | 14.50% | 本ドキュメント §4 換算（2269.1 / 15650.3 GFLOP/s） |
| 1024（参考値・REQ-8 判定対象外） | burn（fresh、GB10 環境 2） | 12.50% | 同上（1956.4 / 15650.3 GFLOP/s） |
| 512（参考値・REQ-8 判定対象外） | fandhe-ai wmma_tf32（カーネル単体） | 105.37% | `cuda-optimized-remeasurement.md` |
| 512（参考値・REQ-8 判定対象外） | candle（fresh、GB10 環境 2） | 14.14% | 本ドキュメント §4 換算（1109.3 / 7847.2 GFLOP/s） |
| 512（参考値・REQ-8 判定対象外） | burn（fresh、GB10 環境 2） | 10.89% | 同上（854.6 / 7847.2 GFLOP/s） |

512・1024 行は `cuda_floor_bench` の `REFERENCE_ONLY_SIZES`（判定対象外の参考形状。§0・§4）の
実測値であり、REQ-8 の 50%（f32）下限自体は引き続き `JUDGED_SIZES`（2048・4096 の実測比率の最小値。
4096 の 51.96%）のみを根拠とする。512・1024 の対 PyTorch 比は同じクロスセッション比較の限定条件
（限定条件 7）を負うため参考情報として併記する。

**両論併記の確定事項（§4 限定条件 4・6 を踏まえ、2 つの原因を分離して記録する）**:

- **REQ-8 下限を満たす候補経路（wmma_tf32・Tensor Core）のカーネル単体性能は candle / Burn の
  CUDA GEMM 実測を上回る水準にある**（4096 で fandhe-ai wmma_tf32 51.96% 対 candle 12.98% /
  burn 16.83%、2048 で fandhe-ai wmma_tf32 83.53% 対 candle/burn 約 24.5%）。ただしこの 51.96%／
  83.53% は `wmma_tf32_staged` 経路の実測値であり、§4 限定条件 6 のとおり同経路固有の parity
  判定不能状態は #726 で解消済みである一方、`performance-floor-decision.md` §10 が継続と明記する
  限定条件 1〜3（数値一致 parity の恒常 fail・REQ-2 改定待ち・実測基準ゲートという運用方針）は
  解消していない。**すなわち 50%（REQ-8 f32 下限）自体が「限定条件付きでユーザー承認済みの
  候補下限」であり、本比較が示す candle/Burn 優位は §4 の限定条件 1〜3・6 を伴う候補経路の性能に
  ついてであって、無条件に確定した性能値としての優位ではない**点に注意する
- **しかし公開 API（facade → tape → backend-cuda）が実際に実行する既定カーネルは tiled f32 固定であり
  （§4 限定条件 4）、その candle/Burn 対比は REQ-8 の判定対象形状（N=2048・4096）では未満である**
  （4096 で tiled f32 GFLOP/s は candle の約 0.87 倍・burn の約 0.67 倍、2048 では約 0.56 倍）。
  一方 §4 の 512・1024（REQ-8 判定対象外の参考形状）では同じ tiled f32 経路でも candle/Burn を
  上回る（512: 約 1.88〜2.45 倍、1024: 約 1.05〜1.22 倍）ため、**tiled f32 の candle/Burn 対比は
  サイズ依存であり「常に未満」ではない**。framework-compare の fandhe-ai 実測はこの tiled f32
  経路を通っているため、**tape 初期化コストや per-call オーバーヘッドを仮に完全にゼロへ縮小
  できたとしても、REQ-8 判定対象形状（N=2048・4096）における公開 API 経由の実効性能は candle/Burn
  を下回ったままである**（N=512・1024 では逆に上回る）
- **原因は 2 つに分離される**: (i) tape 経由の固定オーバーヘッド（環境 2 fresh 458〜594 ms、環境 3
  reuse per-call でも 260〜474 ms。#926 系の後続対応が対象）、(ii) 既定カーネル変種が Tensor Core
  経路（`CudaGemmAuto`／`wmma_tf32`）へ未接続で tiled f32 に固定されていること（`CudaGemmAuto` の
  公開 API への接続自体が別スコープと `cuda-optimized-remeasurement.md` に明記済み）。(i) の解消
  だけでは N=2048・4096 における candle/Burn 対比の逆転（tiled f32 が下回る現状）を覆せず、(ii) の
  対応（既定カーネル変種を Tensor Core 経路へ切り替える判断）も併せて必要になる。いずれも本イシューの
  スコープ外であり、§7 に対象外事項として記録する

## 6. GB10 での未実施計測・再現手順

実装セッションから DGX Spark GB10 へは到達不能（`docs/real-hardware-verification-env.local.md` 不在）
のため、以下 2 種の実測はいずれも未実施。両者は計測境界（§2）が異なる別物であり、混同しないよう
節を分ける。

### 6.1 カーネル単体（`cuda_floor_bench`。N=256）の再現手順

§4・§5 の N=512・1024・2048・4096 のカーネル単体値は `cuda_floor_bench` の既存実測
（`cuda-optimized-remeasurement.md`）から転記済みであり新規実測は不要である。未計測なのは
**N=256 のみ**であり、その再現には `bench-fandhe --mode reuse`（§6.2。tape 経由・per-call
オーバーヘッドを含む）ではなく `cuda_floor_bench`（backend API 直・launch-only 同期）を使う
必要がある（§2 の層 (b) と (c) を混同しないこと。本節は codex-review 指摘 P2 の対応）。

`cuda_floor_bench`（`crates/backend-cuda/examples/cuda_floor_bench.rs`）は計測対象サイズを
`JUDGED_SIZES = [2048, 4096]`・`REFERENCE_ONLY_SIZES = [512, 1024]`（同ファイル 144・153 行）の
定数配列で固定しており、**CLI 引数でサイズを指定する経路を持たない**。したがって N=256 の値を
得るには、実測の前に以下のコード変更が必要（本ドキュメントのスコープ外。§7 参照）:

1. `REFERENCE_ONLY_SIZES` に `256` を追加する（`JUDGED_SIZES` は REQ-8 判定対象形状のみのため
   変更しない）
2. `pytorch_f32_fixed`／`pytorch_f16_fixed`（同ファイル `fn pytorch_f32_fixed` 起点。167 行〜）に
   256 の組み込み固定値の腕（arm）が無いため、対 PyTorch 比を得るには
   `CUDA_FLOOR_BENCH_PYTORCH_F32_256`／`_F16_256` と `CUDA_FLOOR_BENCH_PYTORCH_SOURCE`
   （同一実機での `gemm_bench_torch_cuda.py` 再計測値。ドキュメンテーションコメント「PyTorch
   参照値の再計測」節参照）を注入する必要がある（256 は `JUDGED_SIZES` に含まれないため候補下限
   判定には使われないが、`n/a` 表示を避け参考比率を得るには注入が要る）

上記コード変更後、実機セッションで次を実行する:

```bash
# docs/real-hardware-verification-env.md の rsync/SSH 手順で本リポジトリを転送後、
# nvidia-smi でアイドル確認のうえ実行する。
export CUDA_FLOOR_BENCH_PYTORCH_F32_256=<gemm_bench_torch_cuda.py 5 run 中央値>
export CUDA_FLOOR_BENCH_PYTORCH_SOURCE="gemm_bench_torch_cuda.py 実行, <計測日>, 同一 GB10 個体"
cargo run -p fandhe-ai-backend-cuda --example cuda_floor_bench --release
```

得られた N=256 の `wmma_tf32`／`tiled_f32` TFLOPS 値を §4・§7 へ追記し、対 PyTorch 比を §5 へ
追記する（512・1024 と同じく「参考値・REQ-8 判定対象外」として扱う）。

### 6.2 GB10 での reuse per-call 実測（未実施・再現手順）

環境 2 での `bench-fandhe --mode reuse`（§2 の層 (b)。tape 構築を含まないループ per-call の
計測。カーネル単体〈§6.1〉の代用にはならない点は §0・§2 参照）実測は未実施。実機セッションでの
追試手順を記録する。

```bash
# docs/real-hardware-verification-env.md の rsync/SSH 手順で framework-compare を転送後、
# nvidia-smi でアイドル確認のうえ実行する。
cd scripts/bench/framework-compare
for N in 256 512 1024 2048 4096; do
  cargo run --release -p bench-fandhe -- --task gemm --device cuda --size "$N" --mode reuse
done
```

### 再計測キャンペーン表（GB10・reuse モード。実測時に埋める）

| N | mode | 初期化(init_s) | 中央値 | Q1 | Q3 | GFLOP/s | 計測日 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 256 | reuse | | | | | | |
| 512 | reuse | | | | | | |
| 1024 | reuse | | | | | | |
| 2048 | reuse | | | | | | |
| 4096 | reuse | | | | | | |

実測後は raw JSONL を `scripts/bench/framework-compare/results/raw/` へ保存し、
`scripts/bench/framework-compare/results/summary.md` 環境 2 へ追記する（ホスト名は
`<cuda-node>` プレースホルダ規約を厳守する）。

## 7. 未計測・スコープ外

- **N=256 のカーネル単体（`cuda_floor_bench`）実測が欠落しており、この 1 点のみ vs candle/Burn
  比較・REQ-8 下限突合は受け入れ条件を未充足のまま残す**（§0「状態」・§4・§5）: `cuda_floor_bench`
  は `JUDGED_SIZES`／`REFERENCE_ONLY_SIZES`（512・1024・2048・4096）を計測対象としており、これらの
  値は既に `cuda-optimized-remeasurement.md` に存在し §4・§5 へ転記済みである。N=256 のみ同バイナリの
  計測対象外（定数配列固定・CLI 引数なし）のため値が存在しない。§3.1 の同サイズ fresh/reuse 実測値は
  tape 構築・per-call 固定オーバーヘッドを含むためカーネル実行時間とみなせず（§2・§3.1）、代用
  できない。`cuda_floor_bench` へ 256 を追加するコード変更・GB10 実機セッションでの新規実測が必要
  （§6.1 に再現手順とコード変更点を記録済み）
- **§4・§5 の GB10 個体同一性が未確認**（限定条件 7）: `cuda-optimized-remeasurement.md`（2026-08-18）
  と framework-compare（2026-08-28）の GB10 個体が同一かどうかを確認できる識別子（GPU UUID 等）が
  一次記録に残っていない。実機セッションで両計測に共通の個体識別子を記録し照合することが必要
- **GB10 での `--mode reuse` 新規実測**: 到達不能のため §6.2 に手順を残置。実機セッションでの追試が必要
- **reuse モードでも per-call 固定オーバーヘッドが残る原因の切り分け**（§3.1・§4 限定条件 5）:
  fandhe-ai 側のカーネル選択・ディスクキャッシュ照会等が候補だが未確認。`docs/perf/cuda-tape-init-cost-diagnosis.md`
  （#926）の後続対応として、原因調査を別イシューで追跡することを推奨する（本ドキュメントでは着手しない）
- **既定カーネル変種を Tensor Core 経路（`CudaGemmAuto` / `wmma_tf32`）へ切り替える判断**（§4 限定条件 4・
  §5）: 公開 API は現状 `CudaGemm::run_tiled_f32` 固定であり、`CudaGemmAuto` は未接続。この切替は
  `cuda-optimized-remeasurement.md` 自身が「別スコープ」と明記した既存の設計判断であり、本イシューの
  範囲外。切替判断自体はユーザー承認・別イシューでの追跡を推奨する
- **train/infer タスクの reuse モード対応**: `bench-fandhe --mode reuse` は gemm タスクのみに限定
  実装済み（#925・PR #944 で対象外と確定済み）
- **fandhe-ai 本体 API の初期化コスト削減実装**: #921 フェーズの後続フェーズ扱い（本イシューは計測・
  記録のみ）
