# CUDA 最適化後 f32/f16 対 PyTorch 比 確定計測 記録（#571・Phase F-1）

イシュー #571「bench(backend-cuda): 最適化後の CUDA f32/f16 対 PyTorch 比を確定計測」の実測記録。
GEMM 性能改善ツリー（ルート #479）の Phase F（親 #569「再計測・parity 非後退確認・REQ-8 下限再確定」）の
F-1 に対応する。Metal 側 F-2（#572・`docs/perf/metal-floor-remeasurement.md`）と対称の位置づけ。

## 目的・受け入れ条件対応

Phase B（mma/WMMA パイプライン強化。親 #490。B-0〜B-10 全 CLOSED）・Phase C（JIT テンプレート展開・
コストモデル選択・JIT キャッシュ。親 #503）適用後の CUDA f32/f16 GEMM スループットについて、
`docs/performance-targets.md` §4 の計測プロトコル（warmup 20 回以上・計測 20 回以上の中央値・Q1/Q3、
launch-only 同期境界、決定的シード、判定対象形状は M=N=K=2048/4096 の実測比率の最小値）で対 PyTorch
CUDA 比を**確定計測**し、`docs/perf/cuda-floor-remeasurement.md`（#157/#390 実測記録）と同形式で記録
する。

`docs/perf/cuda-floor-remeasurement.md` は TASK-8.3c（#157/#390）時点の記録であり、当時すでに候補下限
（f32=25%・f16=10%）が確定している（`docs/perf/performance-floor-decision.md` §9）。本ドキュメントは
それを書き換えず、**Phase B/C 適用後**の最新実装での再計測を別ファイルとして記録する（Metal F-2 が
`metal-floor-remeasurement.md` を新設した構成と対称）。候補下限値が変化した場合も、REQ-8 下限の最終
確定は F-5（#577・人間承認）へ引き継ぎ本ドキュメントでは確定させない。

## 実行環境の制約（本ドキュメント作成セッション）

**2026-08-18 追記**: 本節は 2026-08-17 のドキュメント初回整備セッション時点の記録であり、CUDA 実機
（DGX Spark GB10）に到達できないという制約は当時のものである。2026-08-18 に実機到達可能なセッション
で計測を完了した（「計測環境」節以降の実測値・「状態」節を参照）。本節は経緯の記録として書き換えず
残す。

本ドキュメントは Linux worktree で作成された。`docs/real-hardware-verification-env.local.md`（実ホスト
名・接続情報を記す Git 管理外ファイル）は本 worktree に**存在しない**（`.example` のみ存在。下記
「実機到達性の確認結果」参照）。これは `docs/perf/cuda-gemm-mma-pipeline.md`「Phase B 完了時点の再計測
（#502）」節（2026-08-16・実装セッション）が確認した状態と同一であり、その後も変化していない。

したがって本イシューは #502・#534（C-12）・Metal 側 #572 の確立済み先例と同方式を採る:

1. 経路カバレッジの確認（読み取り調査。下記「実測バイナリ」参照）
2. 実機到達性ゲートの再確認（結果: 不達）
3. 計測手順＋記録テンプレートの完全整備・相互参照の整備

**実測値の記入は CUDA 実機（DGX Spark GB10）到達可能なセッションへ申し送る**（下記「状態」節参照）。

### 実機到達性の確認結果（2026-08-17・本実装セッション）

- `ls docs/real-hardware-verification-env.local.md` は `No such file or directory`（`.example` のみ
  存在）。実機ホスト名（`CUDA_NODE`）を解決する前提が満たせないため、SSH 到達性確認（実装計画手順 3）
  に進めない。
- よって実機到達性ゲートは**不達**と判定し、実測は行わず安全側（推定値を記載しない）に倒す
  （#502・#656・#500 §7・#572 の先例と同じ判断）。

## 実測バイナリ（経路カバレッジ確認結果）

`crates/backend-cuda/examples/cuda_floor_bench.rs`（#157 新設・#390 で実機実測・#502 で 1024 参考形状
追加）を再利用する。**本イシューでの追加変更は不要と判断した**（下記の確認結果による）。

- 計測経路（4 経路。無変更）: tiled f32（基準）／WMMA(TF32) opt（f32 最良候補）／WMMA f16 opt／
  `mma.sync` f16 パイプライン（f16 最良候補）
- **Phase B の最適化は既存 launch API 経由で計測に自動反映されることを確認した**（`grep -n
  "run_wmma_tf32\|launch_wmma_tf32\|launch_f16\|upload_" crates/backend-cuda/examples/cuda_floor_bench.rs`
  で確認）: Phase B（`kernels_wmma_opt.rs`・`kernels_mma.rs` のパイプライン強化。PR #678 で TF32 経路へ
  も横展開済み）は `CudaGemm::run_wmma_tf32`／`CudaWmmaGemm::run_f16`／`CudaMmaGemm::run_f16` の内部
  実装であり、`cuda_floor_bench.rs` はこれらを入口として呼ぶだけで新しい実装を自動的に計測する
  （カーネル選択ロジック・呼び出しシグネチャの変更なし）
- **Phase C は 2 系統に分かれ、計測への反映状況が異なることを `grep -rn "pub fn " crates/backend-cuda/src/gemm*.rs`
  ・`crates/backend-cuda/src/nvrtc.rs::compile_ptx` の実装確認で判明させた**:
  - **NVRTC ディスクキャッシュ／プロセス内 LRU キャッシュ（C-1〜C-4。`nvrtc.rs`・`module_cache.rs`）は
    計測に無関係**: `cuda_floor_bench.rs` が呼ぶ `CudaGemm::new`／`CudaWmmaGemm::new`／`CudaMmaGemm::new`
    はいずれも `nvrtc.rs::compile_ptx` を直接呼び、ディスクキャッシュ・LRU キャッシュを経由しない
    （`compile_ptx` 実装〈`nvrtc.rs:3847`〉にキャッシュ参照なし）。これらのキャッシュはコンパイル
    レイテンシ（起動コスト）にのみ寄与し、GEMM 本体のスループットには影響しないため
    `docs/perf/cuda-jit-cache-benchmark.md` が別レイヤとして計測する対象であり、本ドキュメントの
    スコープと重複しない
  - **コストモデルによるタイル選定・JIT shape 特化カーネル（C-5〜C-9 系。`gemm_auto.rs`
    `CudaGemmAuto`／`select_best_tile_candidate`／`run_specialized_mma_f16`）は
    `cuda_floor_bench.rs` から呼ばれておらず、既定ビルドにも含まれない**: `run_specialized_mma_f16`・
    `SpecializedMmaKernelHandle` は `internal-diagnostics` feature（既定 off。
    `crates/backend-cuda/Cargo.toml:57`）でゲートされている。さらに `lib.rs` 冒頭ドキュメンテーション
    コメントが明記するとおり、`CudaBackendOps::gemm`（`facade` 経由の本番エントリ）自体も
    「既定カーネル変種の選択は保守的に tiled 固定とし、`CudaGemmAuto` を介した Tensor Core 経路の
    自動選択への切替は別スコープ」（`ops.rs:183` `CudaGemm::run_tiled_f32` 固定）であり、
    `CudaGemmAuto`（TASK-11.2b・#68。Phase B/C ツリーとは別系統の旧タスク）はいずれの本番経路からも
    現時点で到達しない。したがって `cuda_floor_bench.rs` がこの経路を計測しないことは**取りこぼしでは
    なく現状の実装構成を正しく反映している**。この構成が今後変わりコストモデル選定経路が本番化された
    場合は、`cuda_floor_bench.rs` への経路追加が必要になる（下記「役割分担」節へ申し送り）
  - opt カーネル可用性の検証（`wmma_tf32_opt_available`／`wmma_f16_opt_available`）・f32/f16 最良経路の
    実測比較選出（`best_of`／`f16_candidate_floor_value`）・計測境界の launch-only 統一は #157/#390
    時点で既に確立済みで、上記確認の結果、変更を要しない
- 形状: M=N=K = 512（参考）／1024（参考。#502 追加）／2048／4096（判定対象）
- 計測プロトコル: `bench_harness::protocol::run`（warmup 20 回・計測 20 回・中央値/Q1/Q3。TASK-8.1）・
  決定的シード `0xC0FFEE`（`cuda_floor_bench.rs::SEED`）
- 丸め規則: `bench_harness::floor_lower_bound`（`docs/perf/performance-floor-decision.md` §6 で一本化
  済み）
- PyTorch 参照値の扱い・GPU 名警告・計測境界統一の詳細は `docs/perf/cuda-floor-remeasurement.md`
  「実測バイナリ」節・「PyTorch 参照値の扱い」節を正とし、本ドキュメントでは二重管理しない（#157/#390
  時点から仕様変更なし）。

## 数値一致（parity）確認

性能値採用の前提ゲートとして、実機セッションは計測前に `docs/perf/cuda-gemm-mma-pipeline.md`「Phase B
完了時点の再計測（#502）」節の手順 3 と同一コマンド
（`cargo test -p backend-cuda --test parity_nonregression -- --ignored --test-threads=1`。
`--test-threads=1` は必須: `docs/perf/cuda-floor-remeasurement.md`「tiled f32 @4096 のバイナリ間乖離の
突合結果」節が記録するとおり、同一バイナリ内 `#[test]` の並列実行は GPU 時間分割により計測値を約 5 倍
歪ませた実績があるため）で parity テスト群を実行し、**非後退**（tolerance 定数不変・fail 比率/
mean_abs_diff が `docs/perf/cuda-floor-remeasurement.md`「数値一致（parity）状態の限定条件」節の
ベースライン以下）を確認すること。既存 tolerance 定数・REQ-2 統一複合判定（相対誤差 1e-3 未満 または
絶対誤差 1e-5 未満）は本イシューでは変更しない。

**既知の前提**: `wmma_tf32`・`mma_f16` は #389 §5.3 が記録した parity 恒常 fail 対象である（TF32 経路
5 件・f16 K=4096 tail 3 件。REQ-2 閾値改定は #186〈spec リポジトリ側対応待ち〉へ引き渡し済み）。後退が
無いことを確認できた場合でも、この既知 fail が解消したことにはならない。性能値採用は「非後退」の確認
のみを条件とし、fail 自体の解消は本イシューのスコープ外。後退を検出した場合は性能値を採用せず打ち切り、
その旨を本ドキュメントへ記録して #575（Phase F-4・parity 非後退最終確認）へ申し送る。

## 計測手順（DGX Spark GB10 実機）

`docs/perf/cuda-gemm-mma-pipeline.md`「Phase B 完了時点の再計測（#502）」節「実機セッションでの再実行
手順」を踏襲する。

```sh
git fetch origin

# 本イシューの実装ブランチ（bench/571-cuda-optimized-remeasurement）は PR マージ後に削除される
# 一時ブランチのため、恒久参照として使わない。以下のいずれかで対象コミットを取得する:
#   a) PR #710 が未マージ・ブランチ現存の場合: 上記ブランチを直接 checkout してよい
#   b) マージ済みの場合: 本ドキュメントが記録するコミット SHA を main 上で checkout する
#      （17ff13ab8590e404cf7ef8d3f36f339e86178d72。本イシュー実装完了時点の HEAD）
#   c) 上記 SHA 時点より後の実装状態を計測対象としたい場合: 最新 main を対象とする契約とし、
#      その旨（「実測時点の最新 main、コミット <SHA>」）を「状態」節に明記する
git checkout 17ff13ab8590e404cf7ef8d3f36f339e86178d72   # 本イシュー実装完了時点のコミット（b の場合）
# あるいは: git checkout main && git pull                # 最新 main を対象とする場合（c の場合）

# 1. 到達性・GPU 排他性の確認（docs/real-hardware-verification-env.local.md から CUDA_NODE を取得）
ssh -o BatchMode=yes -o ConnectTimeout=10 "$CUDA_NODE" \
  'hostname && nvidia-smi --query-gpu=name,utilization.gpu --format=csv,noheader'

# 2. docs/real-hardware-verification-env.md §3 の rsync 手順でコードを転送し、.rev-stamp でリビジョン一致を確認する

# 3. 数値一致確認を性能値採用より先に行う（非後退確認。既存 tolerance は緩和しない。
#    --test-threads=1 は同一バイナリ内並列実行による GPU 時間分割歪みを避けるため必須。
#    「数値一致（parity）確認」節参照）
cargo test -p backend-cuda --test parity_nonregression -- --ignored --test-threads=1

# 4. 同一実機で PyTorch 参照値を再計測する（size ∈ {512, 1024, 2048, 4096} × {f32, f16}）
# `<size>` はプレースホルダーであり、そのまま貼り付けると POSIX shell が入力リダイレクトと
# 誤解釈し `size: No such file or directory` で停止する。SIZE 変数へ実値を入れて渡すこと。
for SIZE in 512 1024 2048 4096; do
  python3 docs/spec/03-poc/poc-v2-3-cuda-gemm/code/pytorch/gemm_bench_torch_cuda.py "$SIZE" 20 20
done

# 5. env override を設定し cuda_floor_bench を 3 回反復実行する
export CUDA_FLOOR_BENCH_PYTORCH_SOURCE="gemm_bench_torch_cuda.py 再実行 (warmup=20 iters=20), <実施日>, 同一 GB10 個体"
export CUDA_FLOOR_BENCH_PYTORCH_F32_512=<再計測値>
export CUDA_FLOOR_BENCH_PYTORCH_F32_1024=<再計測値>
export CUDA_FLOOR_BENCH_PYTORCH_F32_2048=<再計測値>
export CUDA_FLOOR_BENCH_PYTORCH_F32_4096=<再計測値>
export CUDA_FLOOR_BENCH_PYTORCH_F16_512=<再計測値>
export CUDA_FLOOR_BENCH_PYTORCH_F16_1024=<再計測値>
export CUDA_FLOOR_BENCH_PYTORCH_F16_2048=<再計測値>
export CUDA_FLOOR_BENCH_PYTORCH_F16_4096=<再計測値>
cargo run -p backend-cuda --example cuda_floor_bench --release --locked
# ↑ を 3 回反復実行し、run 間中央値を代表値として下表へ機械転記する（stdout からの転記のみ。
#   後付け調整は行わない）
```

実機個体名は公開ドキュメントでは `<cuda-node>` にマスクする（`docs/git-history-exposure-decision.md`
の方針）。実測時の原文は `docs/real-hardware-verification-env.local.md` へ記録する。

## 計測環境（実測時に記入）

| 項目 | 値 |
|------|-----|
| GPU（`CudaDevice::name()`） | NVIDIA GB10 |
| compute capability（`CudaDevice::compute_capability()`） | (12, 1) |
| driver バージョン（`nvidia-smi`） | 580.159.03（CUDA 13.0・`nvcc V13.0.88`） |
| rustc | 1.97.0 (2d8144b78 2026-07-07) |
| commit SHA（`.rev-stamp` と転送後の値が一致確認済みであること） | `abaa94e`（下記「計測対象コミットの補足」参照） |
| 実施日 | 2026-08-18 |
| PyTorch 参照値の出典（`pytorch reference provenance:` 行を転記。実機個体名はマスク） | `measured this run (gemm_bench_torch_cuda.py 再実行 (warmup=20 iters=20), 2026-08-18T14:26:04Z (UTC), 同一 GB10 個体 <cuda-node>, torch=2.13.0+cu130)` |
| 計測プロトコル | `bench_harness::protocol::run`（warmup 20 回・計測 20 回・中央値/Q1/Q3。TASK-8.1） |
| 決定的シード | `0xC0FFEE`（`cuda_floor_bench.rs::SEED`） |
| GPU 排他性（実行前後） | 確認済み。`nvidia-smi --query-gpu=utilization.gpu` は計測前後とも 0% で一貫（実質アイドル）。常駐 2 プロセス（ノード管理用）は計測前後で変化なし、計測対象ジョブ以外の GPU 使用プロセスは検出されず |
| 反復回数 | `cuda_floor_bench` を 3 回反復実行し、経路×形状のセルごとに run1〜run3 の中央値を代表値として採用した（「経路×形状 TFLOPS 実測」節参照。セルによって中央値を与える run が異なる）。f16 は floor 境界近傍のため run4/run5 を追加した計 5 run（下記「f16 境界注記」参照） |

### 計測対象コミットの補足

計測手順「状態」節が挙げる SHA `17ff13ab8590e404cf7ef8d3f36f339e86178d72` は PR #710 ブランチ側の
コミットであり、squash マージ後の main には存在しない（マージコミットは `86e7e7e`）。よって手順オプ
ション (c)（実測時点の最新 main を対象とし、その旨を明記する）を採用し、実測時点の main tip
`abaa94e`（bench(backend-cpu) コミット #713）を計測対象とした。`git diff 86e7e7e..abaa94e` を確認し、
parity 関連ファイル（tolerance 定数・`BASELINES`・parity 判定ロジック）に差分がないことを確認済み
（本イシューの数値一致確認前提を崩していない）。

## 経路×形状 TFLOPS 実測（実測時に記入）

各セルは `<中央値>(q1=<Q1由来値>,q3=<Q3由来値>)` の形式で `size=<N> ...` 出力行から転記する
（`cuda_floor_bench.rs::TflopsSample`）。経路×形状のセルごとに run1〜run3 の 3 値（各 run の
`median_tflops`）を独立に中央値化し、その中央値を与えた run の出力行から `<中央値>(q1=,q3=)` を転記
する（中央値を与える run はセルごとに異なりうるため、q1/q3 も当該 run のものを採用する。
`docs/perf/cuda-floor-remeasurement.md`「経路×形状 TFLOPS 実測」節と同形式）。括弧内 `〔〕` へは
run1〜run3 の中央値レンジ（3 値の最小〜最大）を注記する。

| M=N=K | tiled f32（中央値/Q1/Q3、run 間レンジ） | WMMA(TF32) opt（同左） | WMMA f16 opt（同左） | mma.sync f16（同左） | f32 最良経路 | f16 candidate 経路 |
|-------|-----------------------------|-----------------------------|-----------------------------|-----------------------------|---------------|---------------------|
| 512（参考値） | 2.0896(q1=2.0904,q3=2.0880)〔2.0896〜2.1037〕 | 8.2687(q1=8.3135,q3=8.2161)〔8.1960〜8.3220〕 | 4.1191(q1=4.1630,q3=4.1060)〔4.1111〜4.1323〕 | 15.2937(q1=15.3217,q3=15.2382)〔15.1556〜15.3638〕 | wmma_tf32 | mma_f16 |
| 1024（参考値） | 2.3820(q1=2.3826,q3=2.3687)〔2.3811〜2.3838〕 | 12.6453(q1=12.6584,q3=12.6025)〔12.6168〜12.7353〕 | 8.8470(q1=8.8522,q3=8.7867)〔8.8191〜8.8645〕 | 33.5460(q1=33.8763,q3=33.4041)〔33.4457〜33.7995〕 | wmma_tf32 | mma_f16 |
| 2048 | 2.3422(q1=2.3428,q3=2.3412)〔2.3404〜2.3432〕 | 14.3540(q1=14.3719,q3=14.2383)〔14.3201〜14.3842〕 | 7.1973(q1=7.2126,q3=7.1702)〔5.6792〜8.9661〕 | 48.3274(q1=48.5547,q3=46.2002)〔47.9647〜48.3383〕 | wmma_tf32 | mma_f16 |
| 4096 | 1.9798(q1=1.9804,q3=1.9789)〔1.9723〜1.9847〕 | 9.0164(q1=9.0225,q3=9.0061)〔9.0109〜9.0655〕 | 4.3521(q1=4.3528,q3=4.3511)〔4.3508〜4.3620〕 | 33.2252(q1=33.9328,q3=31.0918)〔32.7499〜34.5145〕 | wmma_tf32 | mma_f16 |

代表値の出典 run（セルごとの中央値を与えた run）は次のとおり: 512 の `tiled_f32` は run2/run3 が同値
（2.0896。q1/q3 は run2 側を採用）、512 の `mma_f16` は run3（15.2937）、2048 の `wmma_f16` は run3
（7.1973）、2048 の `mma_f16` は run1（48.3274）、4096 の `tiled_f32` は run3（1.9798）、4096 の
`wmma_tf32` は run3（9.0164）。それ以外のセルは run2 が中央値と一致する。4096 の `mma_f16` レンジ
〔32.7499〜34.5145〕は run1〜run3 のみの範囲であり、境界注記（下記「f16 境界注記」節）で追加した
run4（31.4362 TFLOPS）・run5（32.6191 TFLOPS）は含まない（Appendix の生ログ抜粋参照）。

`wmma_f16`（f16 opt。候補経路ではない）は 2048 形状で run 間ばらつきが大きい（5.68〜8.97 TFLOPS）が、
これは `docs/perf/cuda-gemm-wmma-tf32-phase-b.md` 系列・#391 が既に記録した既知の変動であり、f16 最良
経路として採用するのは常に `mma_f16` であるため候補下限値の算出には影響しない。

## 対 PyTorch 比（実測時に記入）

PyTorch 参照値は本セッション内で一度だけ計測し全 run で共通のため、「経路×形状 TFLOPS 実測」節で
セルの中央値を与えた run の出力行がそのまま当該セルの正しい対 PyTorch 比になる。したがって
`f32_best_over_pytorch=`/`f16_candidate_over_pytorch=` はセルごとの中央値出典 run（前節の注記）から
転記する。

| M=N=K | f32 最良（実測大小比較で選出） / PyTorch f32 比 | f16 candidate（実測大小比較で選出） / PyTorch f16 比 |
|-------|----------------------------------------------------|------------------------------------------------------|
| 512（参考値） | wmma_tf32 = 105.32%（run2） | mma_f16 = 89.88%（run3） |
| 1024（参考値） | wmma_tf32 = 80.80%（run2） | mma_f16 = 60.06%（run2） |
| 2048 | wmma_tf32 = 82.82%（run2） | mma_f16 = 52.15%（run1） |
| 4096 | wmma_tf32 = **51.60%**（run3） | mma_f16 = **39.42%**（run2） |

判定対象形状（2048/4096）の最小比率: f32 = 51.60%（4096）・f16 = 39.42%（4096）。

## 丸め適用後の候補下限値（実測時に記入）

| 精度 | 判定対象形状の最小比率（2048/4096） | 丸め規則適用後の候補下限値 | #390 実測値（f32=25%・f16=10%）との比較 |
|------|--------------------------------------|------------------------------|------------------------------|
| f32  | 51.60%（4096） | **50%** | #390 の 25% を 25pt 上回る |
| f16  | 39.42%（4096） | **35%**（境界注記あり。下記参照） | #390 の 10% を 25pt 上回る |

### f16 境界注記（必読）

f16 candidate（4096 形状）の対 PyTorch 比は `bench_harness::floor_lower_bound` の丸め刻み（5% 刻み切り
下げ）の境界近傍（35% 台後半〜40% 台前半）に位置するため、run1〜run3 の 3 run に加えて run4・run5 を
追加した **計 5 run** を実行し境界跨ぎの有無を確認した。

| run | 4096 f16 candidate 対 PyTorch 比 | 丸め後 floor |
|-----|-----------------------------------|--------------|
| run1 | 40.95% | 40 |
| run2 | 39.42% | 35 |
| run3 | 38.86% | 35 |
| run4 | 37.30% | 35 |
| run5 | 38.70% | 35 |

5 run 中央値 = 38.86%（run3）→ 丸め後候補下限値 **35%**。run1 のみが 40% 境界を跨いだ事実（run1 だけ
40.95% で floor=40 相当）を明記する。3 run のセル単位中央値運用（「経路×形状 TFLOPS 実測」節）では
この形状（4096・f16 candidate）はたまたま run2 が中央値（39.42%）と一致するため、本ドキュメントの
確定候補値は 35% とするが、**境界近傍のため 5 run 中 1/5 が隣接刻みへ振れる程度の run 間変動がある
ことを申し送る**。採否・最終確定は F-5（#577）へ引き継ぐ。

**候補下限値は参考算出に留める。** REQ-8 下限値（現行確定値: f32=25%・f16=10%。
`docs/perf/performance-floor-decision.md` §9）の変更判断は本ドキュメントでは行わない。変更は F-5
（#577・人間承認タスク）のみが行う。

## 数値一致（parity）状態の限定条件

`docs/perf/cuda-gemm-mma-pipeline.md`「Phase B 完了時点の再計測（#502）」節の手順 3 と同一コマンド
（`cargo test -p backend-cuda --test parity_nonregression -- --ignored --test-threads=1`。debug/release
両プロファイルで実行し同一結果を確認）を実行した結果、`parity_baselines_do_not_regress` は
`wmma_tf32_staged 512×512×4096 seed=0xC0FFEE` の 1 件で FAIL した。この FAIL は数値乖離ではなく
`baseline_provenance_unconfirmed == true`（基本版カーネル専用の確定ベースラインが未整備のための
fail-closed プレースホルダ。#500 由来。参考実測 `fail_count=43019/262144, mean_abs_diff=4.463436e-3`
は合否判定に不使用）であり、**tolerance 定数・parity ロジック自体は無変更**（「計測環境」節の
`git diff 86e7e7e..abaa94e` 確認結果）のため後退ではない。

`--ignored` の非 `parity_nonregression` 系（`cargo test -p backend-cuda --lib -- --ignored`・
`cargo test -p backend-cuda --test cpu_cuda_mma_parity -- --ignored`）でも同様に選出経路の恒常 fail を
確認した:

| 候補下限の経路 | テスト | fail 内容（実測） | #389 §5.3 の恒常 fail 対象との一致 |
|---|---|---|---|
| `wmma_tf32`（基本版） | `wmma_tf32_basic_kernel_parity_does_not_regress` | 32×32×32: fail_count=154/1024（15.04%）／256×256×4096 stress: fail_count=10647/65536（16.25%） | 一致（`baseline_provenance_unconfirmed` 経路） |
| `wmma_tf32` opt | `wmma_tf32_opt_kernel_k4096_stress` | 512×512×4096: fail_count=43019/262144（**16.41%**） | 一致（#389 §5.3 K4096 stress） |
| `wmma_tf32` opt | `wmma_tf32_opt_kernel_matches_reference_across_shapes` | m=n=k=64: fail_count=699/4096（**17.06%**） | 一致（#389 §5.3 shape grid） |
| `mma_f16`（f16 candidate） | `mma_f16_k4096_stress` | 256×256×4096: fail_count=101/65536（**0.154%**） | 一致（#389 §5.3 K4096 tail） |

いずれも #389 §5.3 が記録した恒常 fail 範囲内（TF32 系 16〜17% 台・f16 K4096 stress 0.15% 台）で
**後退なし**と判定した。`wmma_tf32_opt_kernel_parity_does_not_regress`・`wmma_tf32_staged_kernel_...`・
`mma_f16_cross_check_against_wmma_f16`・`mma_f16_matches_reference_across_shapes` は pass。

（`nvrtc::jit_cache_bench_tests::*` の 2 件 FAIL は `/tmp` 配下の cache root pin に関する環境依存
エラーで、GEMM スループット・parity とは無関係のため本ドキュメントのスコープ外。#534〈C-12〉の
JIT キャッシュベンチ側の既知事象として申し送る。）

- **後退の有無**: tolerance 定数不変（コミット確認済み）・fail 比率/mean_abs_diff が #389 §5.3
  ベースライン以下 → **後退なし**
- 後退が無い場合でも、既知の parity 恒常 fail（#389 §5.3）自体は解消されていない
- 本 candidate floor（f32=50%・f16=35%）は数値一致未達の経路（`wmma_tf32`／`mma_f16`）の実測値であり、
  #186（REQ-2 閾値改定。spec リポジトリ側対応待ち）の解決前は #577 の下限確定根拠として**単独採用
  できない**（#186 限定条件は継続）

## 状態: 実測完了（2026-08-18・DGX Spark GB10）

本ドキュメントは当初 Linux worktree で計測手順・記録テンプレートのみを整備していたが（#502・
#534（C-12）・#572 先例と同方式）、2026-08-18 に CUDA 実機（DGX Spark GB10。`<cuda-node>`。実名は
`docs/real-hardware-verification-env.local.md` 参照）到達可能なセッションで上記「計測環境」「経路×
形状 TFLOPS 実測」「対 PyTorch 比」「丸め適用後の候補下限値」「数値一致（parity）状態の限定条件」の
各表・節を実測値で埋めた。

ノード同期は public 化に伴い GitHub 匿名 fetch（`git fetch origin && git reset --hard FETCH_HEAD`）で
行い、`abaa94e` 一致を確認した。`docs/spec`（private submodule）はノード上で初期化不能なため、
PyTorch 参照値計測スクリプト（`gemm_bench_torch_cuda.py`）は Mac から scp 転送して実行した。

内部ホスト名等の実値は書かない（#461 のプレースホルダ方針。実測時の原文は
`docs/real-hardware-verification-env.local.md` へ記録済み）。

## 動作確認（実機セッションで実施済み）

- `cargo build --workspace --locked` — `cudarc` 動的ロード契約（CUDA toolkit 非搭載環境でもビルド成立
  する。`.claude/rules/coding-rust.md`）を崩していないことを確認済み（実機は CUDA 13.0 搭載のため本
  確認は Linux worktree 側の当初整備セッションの結果を踏襲）
- `cargo build -p backend-cuda --example cuda_floor_bench --release` — example のビルド成立（無変更）
- `cargo test -p backend-cuda --test parity_nonregression -- --ignored --test-threads=1`（debug/release
  両方）・`cargo test -p backend-cuda --lib -- --ignored`・
  `cargo test -p backend-cuda --test cpu_cuda_mma_parity -- --ignored` — 上記「数値一致（parity）状態
  の限定条件」節のとおり既知 fail のみ、後退なし
- `cargo run -p backend-cuda --example cuda_floor_bench --release --locked` を計 5 回実行（生ログ
  `floor_bench_run{1..5}.log`）
- `git diff 86e7e7e..abaa94e -- crates/backend-cuda/src crates/backend-cuda/tests/common crates/bench-harness`
  が tolerance 定数（`RELATIVE_TOLERANCE`・`ABSOLUTE_RESCUE_THRESHOLD`）・parity fixture・`FloorSpec`・
  カーネルソースに差分を持たないことを確認（#502・#390 の検証項目を踏襲）

## Appendix: 生ログ抜粋

実測実行時の生ログはセッション限定の scratchpad 配下（`bench571/`）にのみ存在しコミット対象外のため、
本節へ最小限を転記する（内部ホスト名は `<cuda-node>` へマスク済み）。

### 環境情報（`env_info.txt`）

```
rustc 1.97.0 (2d8144b78 2026-07-07)
cargo 1.97.0 (c980f4866 2026-06-30)
NVIDIA GB10, 580.159.03, 12.1
```

### PyTorch 参照値（`pytorch_size{512,1024,2048,4096}.log`）

```
torch=2.13.0+cu130 numpy=2.2.6 cuda=13.0 device=NVIDIA GB10
kernel=torch_matmul_cuda dtype=f32 size=512  warmup=20 iters=20 median_tflops=7.8508  q1=7.8033  q3=7.8618
kernel=torch_matmul_cuda dtype=f16 size=512  warmup=20 iters=20 median_tflops=17.0153 q1=16.8278 q3=17.1024
kernel=torch_matmul_cuda dtype=f32 size=1024 warmup=20 iters=20 median_tflops=15.6503 q1=15.5794 q3=15.6650
kernel=torch_matmul_cuda dtype=f16 size=1024 warmup=20 iters=20 median_tflops=55.8542 q1=55.7149 q3=55.8772
kernel=torch_matmul_cuda dtype=f32 size=2048 warmup=20 iters=20 median_tflops=17.3309 q1=17.2416 q3=17.4023
kernel=torch_matmul_cuda dtype=f16 size=2048 warmup=20 iters=20 median_tflops=92.6679 q1=92.2691 q3=92.8516
kernel=torch_matmul_cuda dtype=f32 size=4096 warmup=20 iters=20 median_tflops=17.4750 q1=17.2059 q3=17.5175
kernel=torch_matmul_cuda dtype=f16 size=4096 warmup=20 iters=20 median_tflops=84.2850 q1=77.8918 q3=84.8805
```

### `cuda_floor_bench` run2（生ログ全文。`floor_bench_run2.log`）

run2 は 1 回の実行として全形状を通しで実行しており生ログの原文はそのまま残すが、下記「経路×形状
TFLOPS 実測」節のセル中央値は run ごとに独立に算出するため、run2 の `f32_best_over_pytorch`/
`f16_candidate_over_pytorch`（末尾 2 行の丸め結果を含む）が全セルの代表値と一致するとは限らない
（一致しないセルの裏付けは次項参照）。

```
device: name=NVIDIA GB10 compute_capability=(12, 1)
pytorch reference provenance: measured this run (gemm_bench_torch_cuda.py 再実行 (warmup=20 iters=20), 2026-08-18T14:26:04Z (UTC), 同一 GB10 個体 <cuda-node>, torch=2.13.0+cu130)
size=512  tiled_f32_tflops=2.0896 wmma_tf32_tflops=8.2687 wmma_f16_tflops=4.1191 mma_f16_tflops=15.3638 f32_best_over_pytorch=105.32% f16_candidate_over_pytorch=90.29%
size=1024 tiled_f32_tflops=2.3820 wmma_tf32_tflops=12.6453 wmma_f16_tflops=8.8470 mma_f16_tflops=33.5460 f32_best_over_pytorch=80.80% f16_candidate_over_pytorch=60.06%
size=2048 tiled_f32_tflops=2.3422 wmma_tf32_tflops=14.3540 wmma_f16_tflops=5.6792 mma_f16_tflops=48.3383 f32_best_over_pytorch=82.82% f16_candidate_over_pytorch=52.16%
size=4096 tiled_f32_tflops=1.9847 wmma_tf32_tflops=9.0109 wmma_f16_tflops=4.3521 mma_f16_tflops=33.2252 f32_best_over_pytorch=51.56% f16_candidate_over_pytorch=39.42%
CUDA f32 candidate optimized floor (rounding rule applied to min ratio 51.56%) = 50%
CUDA f16 candidate optimized floor (rounding rule applied to min ratio 39.42%) = 35%
```

run2 単体の丸め結果（f32=50%・f16=35%）は「経路×形状 TFLOPS 実測」節のセル中央値を使った判定対象
形状の最小比率（f32=51.60%・f16=39.42%）から導く丸め結果と一致する（下記「丸め適用後の候補下限値」
節参照）。

### セル中央値が run2 と異なるセルの裏付け（`floor_bench_run{1,3}.log` 該当行）

PyTorch 参照値は全 run 共通のため、各 run が自身の出力行で報告する
`f32_best_over_pytorch`/`f16_candidate_over_pytorch` は、そのまま当該 run の値をセル中央値として採用
した場合の正しい対 PyTorch 比になる。

```
run3 size=512:  wmma_tf32_tflops=8.1960 mma_f16_tflops=15.2937 f32_best_over_pytorch=104.40% f16_candidate_over_pytorch=89.88%
run1 size=2048: wmma_tf32_tflops=14.3842 mma_f16_tflops=48.3274 f32_best_over_pytorch=83.00% f16_candidate_over_pytorch=52.15%
run3 size=2048: wmma_tf32_tflops=14.3201 wmma_f16_tflops=7.1973 mma_f16_tflops=47.9647 f32_best_over_pytorch=82.63% f16_candidate_over_pytorch=51.76%
run3 size=4096: tiled_f32_tflops=1.9798 wmma_tf32_tflops=9.0164 f32_best_over_pytorch=51.60% f16_candidate_over_pytorch=38.86%
```

### f16 境界注記の裏付け（run1・run3〜run5 の size=4096 行）

```
run1: mma_f16_tflops=34.5145 f16_candidate_over_pytorch=40.95%
run3: mma_f16_tflops=32.7499 f16_candidate_over_pytorch=38.86%
run4: mma_f16_tflops=31.4362 f16_candidate_over_pytorch=37.30%
run5: mma_f16_tflops=32.6191 f16_candidate_over_pytorch=38.70%
```

### parity 実行結果（抜粋）

- `parity_nonregression`（debug/release 両方・`--test-threads=1`）: `wmma_tf32_staged 512x512x4096` の
  1 件 FAIL（`baseline_provenance_unconfirmed`。参考実測 `fail_count=43019/262144,
  mean_abs_diff=4.463436e-3`）
- `--lib -- --ignored`: `wmma_tf32_basic_kernel_parity_does_not_regress`
  （32×32×32 `fail_count=154/1024`・256×256×4096 stress `fail_count=10647/65536`）・
  `wmma_tf32_opt_kernel_k4096_stress`（512×512×4096 `fail_count=43019/262144, mean_abs_diff=4.463e-3`）・
  `wmma_tf32_opt_kernel_matches_reference_across_shapes`（m=n=k=64 `fail_count=699/4096,
  mean_abs_diff=5.676e-4`）が FAIL。他 7 件は pass（`nvrtc::jit_cache_bench_tests::*` 2 件の FAIL は
  `/tmp` cache root pin の環境依存エラーで GEMM parity と無関係。「動作確認」節参照）
- `cpu_cuda_mma_parity -- --ignored`: `mma_f16_k4096_stress`（256×256×4096
  `fail_count=101/65536, mean_abs_diff=7.646e-5`）のみ FAIL。`mma_f16_cross_check_against_wmma_f16`・
  `mma_f16_matches_reference_across_shapes` は pass

## 役割分担（二重管理を避ける）

- **`docs/perf/cuda-floor-remeasurement.md`（#157/#390）**: TASK-8.3c 時点（Phase B/C 適用前）の確定
  記録。本ドキュメントでは書き換えない
- **本ドキュメント（#571・Phase F-1）**: Phase B/C 適用後の再計測記録。2026-08-18 実測完了
- **`docs/perf/cuda-gemm-mma-pipeline.md`「Phase B 完了時点の再計測（#502）」節**: Phase B 単独の実機
  未到達記録。実機セッションが本ドキュメントと合わせて埋める判断は実機セッション側に委ねる
- **`docs/perf/cuda-jit-cache-benchmark.md`（#534・C-12）**: Phase C（JIT キャッシュ）固有の初回コンパ
  イル／2 回目ロード時間の実機未到達記録。本ドキュメントのスループット計測とは別レイヤ
- **#575（Phase F-4）**: parity 非後退の最終確認。記録先は
  `docs/perf/cuda-parity-baseline.md` §8「Phase F-4 最終確認（#575）」。本
  ドキュメントは性能値採用の前提ゲートとして非後退を確認するに留め、最終
  確認は #575（同 §8）が行う
- **#577（Phase F-5・人間承認）**: REQ-8 下限値の最終確定・`docs/spec/04-requirements.md` への反映判断
  （`docs/spec/` は本リポでは編集しない）
- **#569（Phase F 親）・#579**: 全バックエンド横断の集約・`docs/performance-targets.md` 更新

## 未実施・後続作業

- **実機実測**: 「状態」節のとおり 2026-08-18 実測完了。本節は完了扱い
- **候補下限値の最終確定・REQ-8 反映判断**: F-5（#577・人間承認）が本ドキュメントの実測結果
  （f32 候補 50%・f16 候補 35%〈境界注記あり〉）を受けて対応する
- **parity 非後退の最終確認**: F-4（#575）が本ドキュメントの非後退確認結果を受けて最終確認する
- **コストモデル選定・JIT shape 特化経路（`gemm_auto.rs::CudaGemmAuto`／`run_specialized_mma_f16`）の
  本番化判断**: 「実測バイナリ（経路カバレッジ確認結果）」節のとおり、この経路は現時点で
  `internal-diagnostics` feature 限定かつ `CudaBackendOps::gemm` からも到達しないため本イシューの
  計測対象から意図的に除外した。本番経路へ組み込む判断自体は本イシューのスコープ外（別イシュー）であり、
  組み込まれた場合は `cuda_floor_bench.rs` への経路追加が再度必要になる
- **`docs/perf/cuda-gemm-wmma-tf32-phase-b.md` §7 の未計測テンプレート**: 実機セッションが本ドキュメント
  と合わせて転記対象とするかは実機セッション側の判断に委ねる（本イシューのスコープ外）
