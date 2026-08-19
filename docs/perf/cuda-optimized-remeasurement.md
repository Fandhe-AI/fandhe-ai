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

# 4. 同一実機で PyTorch 参照値を計 5 回計測する（size ∈ {512, 1024, 2048, 4096} × {f32, f16}）。
#    リポジトリ規約「ベンチは 5 回計測の中央値」（coding-rust.md）に合わせ、PyTorch 参照値も
#    Rust 側 cuda_floor_bench と同数の 5 run を計測し、size×dtype ごとに run 間中央値を採る
#    （当初は 1 run のみで確定していたが、codex レビュー P1 指摘を受け run2〜run5 を追加計測し
#    5 run 中央値ベースへ統一した。「PyTorch 参照値の再集計」節参照）。
# `<size>` はプレースホルダーであり、そのまま貼り付けると POSIX shell が入力リダイレクトと
# 誤解釈し `size: No such file or directory` で停止する。SIZE 変数へ実値を入れて渡すこと。
for RUN in 1 2 3 4 5; do
  for SIZE in 512 1024 2048 4096; do
    python3 docs/spec/03-poc/poc-v2-3-cuda-gemm/code/pytorch/gemm_bench_torch_cuda.py "$SIZE" 20 20
  done
done
# ↑ run ごとに出力を保存し、size×dtype ごとに 5 run の median_tflops を独立に中央値化する。

# 5. env override へ size×dtype ごとの PyTorch 5 run 中央値を設定し cuda_floor_bench を 5 回反復実行する
export CUDA_FLOOR_BENCH_PYTORCH_SOURCE="gemm_bench_torch_cuda.py 5 run 再実行 (warmup=20 iters=20) の run 間中央値, <実施日>, 同一 GB10 個体"
export CUDA_FLOOR_BENCH_PYTORCH_F32_512=<5 run 中央値>
export CUDA_FLOOR_BENCH_PYTORCH_F32_1024=<5 run 中央値>
export CUDA_FLOOR_BENCH_PYTORCH_F32_2048=<5 run 中央値>
export CUDA_FLOOR_BENCH_PYTORCH_F32_4096=<5 run 中央値>
export CUDA_FLOOR_BENCH_PYTORCH_F16_512=<5 run 中央値>
export CUDA_FLOOR_BENCH_PYTORCH_F16_1024=<5 run 中央値>
export CUDA_FLOOR_BENCH_PYTORCH_F16_2048=<5 run 中央値>
export CUDA_FLOOR_BENCH_PYTORCH_F16_4096=<5 run 中央値>
cargo run -p backend-cuda --example cuda_floor_bench --release --locked
# ↑ を 5 回反復実行し、経路×形状のセルごとに run1〜run5 の median_tflops を独立に中央値化した
#   ものを代表値として下表へ機械転記する（stdout からの転記のみ。後付け調整は行わない）
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
| PyTorch 参照値の出典（`pytorch reference provenance:` 行を転記。実機個体名はマスク） | `measured this run (gemm_bench_torch_cuda.py 再実行 (warmup=20 iters=20), 2026-08-18T14:26:04Z (UTC), 同一 GB10 個体 <cuda-node>, torch=2.13.0+cu130)`（run1 実行時の出典文字列。size×dtype ごとの正式な参照値は run1〜run5 の 5 run 中央値を採用する。下記「PyTorch 参照値の再集計（5 run 中央値）」節参照） |
| 計測プロトコル | `bench_harness::protocol::run`（warmup 20 回・計測 20 回・中央値/Q1/Q3。TASK-8.1） |
| 決定的シード | `0xC0FFEE`（`cuda_floor_bench.rs::SEED`） |
| GPU 排他性（実行前後） | 確認済み。`nvidia-smi --query-gpu=utilization.gpu` は計測前後とも 0% で一貫（実質アイドル）。常駐 2 プロセス（ノード管理用）は計測前後で変化なし、計測対象ジョブ以外の GPU 使用プロセスは検出されず |
| 反復回数 | `cuda_floor_bench` を 5 回反復実行し、経路×形状のセルごとに run1〜run5 の中央値を代表値として採用した（「経路×形状 TFLOPS 実測」節参照。セルによって中央値を与える run が異なる）。PyTorch 参照値も同じく 5 run 計測し size×dtype ごとに中央値を採用した（下記「PyTorch 参照値の再集計（5 run 中央値）」節）。当初はリポジトリ規約「ベンチは 5 回計測の中央値」（coding-rust.md）に対し f32 が 3 run・PyTorch が 1 run のまま確定していたが、codex レビュー P1 指摘を受け PyTorch 側の追加 4 run を計測し、Rust・PyTorch とも 5 run 中央値ベースへ統一した |

### 計測対象コミットの補足

計測手順「状態」節が挙げる SHA `17ff13ab8590e404cf7ef8d3f36f339e86178d72` は PR #710 ブランチ側の
コミットであり、squash マージ後の main には存在しない（マージコミットは `86e7e7e`）。よって手順オプ
ション (c)（実測時点の最新 main を対象とし、その旨を明記する）を採用し、実測時点の main tip
`abaa94e`（bench(backend-cpu) コミット #713）を計測対象とした。`git diff 86e7e7e..abaa94e` を確認し、
parity 関連ファイル（tolerance 定数・`BASELINES`・parity 判定ロジック）に差分がないことを確認済み
（本イシューの数値一致確認前提を崩していない）。

## PyTorch 参照値の再集計（5 run 中央値。codex レビュー P1 対応）

初回計測時は PyTorch 参照値を 1 run（`pytorch_size{512,1024,2048,4096}.log`）のみで確定し、Rust 側
（`cuda_floor_bench`）の f32 経路も 3 run で確定していた。この構成はリポジトリ規約「ベンチは 5 回計測
の中央値」（`.claude/rules/coding-rust.md`）と不整合であるという codex レビュー P1 指摘を受け、PyTorch
参照値の追加 4 run（`pytorch_size*_run{2..5}.log`）を計測し、Rust・PyTorch とも 5 run 分のデータを揃え
たうえで**全セルを 5 run 中央値ベースへ再集計した**（本節以降がその再集計結果であり、以下の全ての表・
判定・候補下限値は 5 run 中央値ベースの確定値である）。

size×dtype ごとの PyTorch 5 run 中央値（`<中央値>(q1=<Q1由来値>,q3=<Q3由来値>)`、括弧内 `〔〕` は
run1〜run5 の最小〜最大レンジ）:

| size | f32（5 run 中央値） | f16（5 run 中央値） |
|------|----------------------|----------------------|
| 512  | 7.8472(q1=7.8070,q3=7.8729)〔7.7997〜7.8767〕（run3） | 17.0153(q1=16.8278,q3=17.1024)〔16.8956〜17.1021〕（run1） |
| 1024 | 15.6503(q1=15.5794,q3=15.6650)〔15.6121〜15.7329〕（run1） | 55.8542(q1=55.7149,q3=55.8772)〔48.7885〜56.0644〕（run1） |
| 2048 | 17.1582(q1=17.0755,q3=17.2727)〔17.0845〜17.3309〕（run5） | 92.6039(q1=92.2147,q3=92.8121)〔92.4767〜92.7398〕（run4） |
| 4096 | 17.4467(q1=17.2560,q3=17.4687)〔17.3663〜17.4750〕（run4） | 87.4117(q1=86.2003,q3=88.0560)〔81.9354〜88.5662〕（run2） |

**PyTorch 異常値の記録**: size=1024 f16 run4 = 48.7885（q1=40.8702, q3=54.4714。他 4 run は
55.8309〜56.0644）は一過性ジッタと推定される（q1/q3 の広がり自体が計測中の外れ値混入を示唆する）。
5 値中央値化のため採用値（55.8542・run1）へは影響しない。

**f16 4096 の run 間分散**: PyTorch f16 4096 は 81.9354〜88.5662（約 8% の広がり）と他 size・他 dtype
より分散が大きい。`docs/perf/cuda-floor-remeasurement.md`「f16 ヒューリスティクスのばらつき」系列・
#391 が既に記録した PyTorch cuDNN/cuBLAS ヒューリスティクス選択のばらつきと同種の事象と考えられ、下記
「対 PyTorch 比」節の比率にもこの分散が反映される（4096 f16 の比率は PyTorch 側の run 間変動の影響を
受けやすい点に留意する）。

## 経路×形状 TFLOPS 実測（実測時に記入）

各セルは `<中央値>(q1=<Q1由来値>,q3=<Q3由来値>)` の形式で `size=<N> ...` 出力行から転記する
（`cuda_floor_bench.rs::TflopsSample`）。経路×形状のセルごとに run1〜run5 の 5 値（各 run の
`median_tflops`）を独立に中央値化し、その中央値を与えた run の出力行から `<中央値>(q1=,q3=)` を転記
する（中央値を与える run はセルごとに異なりうるため、q1/q3 も当該 run のものを採用する。
`docs/perf/cuda-floor-remeasurement.md`「経路×形状 TFLOPS 実測」節と同形式）。括弧内 `〔〕` へは
run1〜run5 の中央値レンジ（5 値の最小〜最大）を注記する。**当初は 3 run（run1〜run3）で確定していたが、
codex レビュー P1 指摘を受け run4・run5 を含む 5 run 中央値ベースへ統一した**（上記「PyTorch 参照値の
再集計」節参照）。

| M=N=K | tiled f32（中央値/Q1/Q3、run 間レンジ） | WMMA(TF32) opt（同左） | WMMA f16 opt（同左） | mma.sync f16（同左） | f32 最良経路 | f16 candidate 経路 |
|-------|-----------------------------|-----------------------------|-----------------------------|-----------------------------|---------------|---------------------|
| 512（参考値） | 2.0896(q1=2.0904,q3=2.0880)〔2.0893〜2.1037〕 | 8.2687(q1=8.3135,q3=8.2161)〔8.1960〜8.3220〕 | 4.1191(q1=4.1630,q3=4.1060)〔4.1100〜4.1323〕 | 15.2937(q1=15.3217,q3=15.2382)〔15.1556〜15.4202〕 | wmma_tf32 | mma_f16 |
| 1024（参考値） | 2.3830(q1=2.3841,q3=2.3811)〔2.3811〜2.3840〕 | 12.6453(q1=12.6584,q3=12.6025)〔12.6096〜12.7486〕 | 8.8470(q1=8.8522,q3=8.7867)〔8.7959〜8.8892〕 | 33.4457(q1=33.6385,q3=33.3709)〔32.6009〜33.7995〕 | wmma_tf32 | mma_f16 |
| 2048 | 2.3425(q1=2.3448,q3=2.3414)〔2.3404〜2.3432〕 | 14.3326(q1=14.3634,q3=14.1159)〔14.1789〜14.3842〕 | 7.7525(q1=7.7799,q3=7.6902)〔5.6792〜8.9661〕 | 48.3274(q1=48.5547,q3=46.2002)〔47.9647〜48.6295〕 | wmma_tf32 | mma_f16 |
| 4096 | 1.9723(q1=1.9726,q3=1.9722)〔1.9682〜1.9847〕 | 9.0655(q1=9.0768,q3=9.0581)〔9.0109〜9.0745〕 | 4.3546(q1=4.3567,q3=4.3543)〔4.3508〜4.3634〕 | 32.7499(q1=34.2161,q3=29.9705)〔31.4362〜34.5145〕 | wmma_tf32 | mma_f16 |

代表値の出典 run（セルごとの中央値を与えた run）は次のとおり: 512 の `tiled_f32` は run2/run3 が同値
（2.0896。q1/q3 は run2 側を採用）、512 の `wmma_tf32`・`wmma_f16` は run2、512 の `mma_f16` は run3
（15.2937）、1024 の `tiled_f32` は run5（2.3830）、1024 の `wmma_tf32`・`wmma_f16` は run2、1024 の
`mma_f16` は run1（33.4457）、2048 の `tiled_f32`・`wmma_tf32`・`wmma_f16` は run4、2048 の `mma_f16`
は run1（48.3274）、4096 の `tiled_f32`・`wmma_tf32` は run1、4096 の `wmma_f16` は run5（4.3546）、
4096 の `mma_f16` は run3（32.7499）。

`wmma_f16`（f16 opt。候補経路ではない）は 2048 形状で run 間ばらつきが大きい（5.68〜8.97 TFLOPS）が、
これは `docs/perf/cuda-gemm-wmma-tf32-phase-b.md` 系列・#391 が既に記録した既知の変動であり、f16 最良
経路として採用するのは常に `mma_f16` であるため候補下限値の算出には影響しない。

上表の「WMMA(TF32) opt」列ラベルは `cuda_floor_bench.rs` の起動時診断メッセージ（`wmma_tf32_opt_available()`
由来）をそのまま踏襲したものであり、本セッションで実際に選択・計測された経路は staged である（「数値
一致（parity）状態の限定条件」節「f32 候補下限（50%）への影響に関する重要な注記」参照）。

## 対 PyTorch 比（実測時に記入）

対 PyTorch 比 = Rust セル 5 run 中央値（「経路×形状 TFLOPS 実測」節）÷ PyTorch 5 run 中央値（「PyTorch
参照値の再集計（5 run 中央値）」節）で size×dtype ごとに新規に計算する（Rust・PyTorch のセル中央値は
出典 run が size×dtype ごとに異なりうるため、両者とも独立に中央値化した値どうしを組み合わせる）。

| M=N=K | f32 最良（実測大小比較で選出） / PyTorch f32 比 | f16 candidate（実測大小比較で選出） / PyTorch f16 比 |
|-------|----------------------------------------------------|------------------------------------------------------|
| 512（参考値） | wmma_tf32 = 8.2687 / 7.8472 = 105.37% | mma_f16 = 15.2937 / 17.0153 = 89.88% |
| 1024（参考値） | wmma_tf32 = 12.6453 / 15.6503 = 80.80% | mma_f16 = 33.4457 / 55.8542 = 59.88% |
| 2048 | wmma_tf32 = 14.3326 / 17.1582 = 83.53% | mma_f16 = 48.3274 / 92.6039 = 52.19% |
| 4096 | wmma_tf32 = 9.0655 / 17.4467 = **51.96%** | mma_f16 = 32.7499 / 87.4117 = **37.47%** |

判定対象形状（2048/4096）の最小比率: f32 = 51.96%（4096）・f16 = 37.47%（4096）。

## 丸め適用後の候補下限値（実測時に記入）

| 精度 | 判定対象形状の最小比率（2048/4096） | 丸め規則適用後の候補下限値 | #390 実測値（f32=25%・f16=10%）との比較 |
|------|--------------------------------------|------------------------------|------------------------------|
| f32  | 51.96%（4096） | **50%** | #390 の 25% を 25pt 上回る |
| f16  | 37.47%（4096） | **35%**（境界注記あり。下記参照） | #390 の 10% を 25pt 上回る |

丸め規則（`bench_harness::rounding::floor_lower_bound`。10% 以上は 5% 刻み切り下げ）を適用すると、
f32: `floor(51.96 / 5) * 5 = 50`、f16: `floor(37.47 / 5) * 5 = 35` となり、3 run・PyTorch 1 run 時点の
確定値（f32=50%・f16=35%）から**変化しない**（下記「f16 境界注記」節参照）。

### f16 境界注記（必読）

f16 candidate（4096 形状）の対 PyTorch 比は `bench_harness::floor_lower_bound` の丸め刻み（5% 刻み切り
下げ）の境界近傍に位置するため、PyTorch 参照値を 5 run 中央値へ再集計した後も引き続き境界跨ぎの有無を
確認する。

**正式な確定実測値**は Rust 5 run 中央値（32.7499 TFLOPS・run3）÷ PyTorch 5 run 中央値（87.4117
TFLOPS・run2）= **37.47%**（丸め後 floor **35**）である。

参考として、Rust 側の run1〜run5 各 run の `mma_f16`（4096）TFLOPS 実測値を、正式な PyTorch 参照値
（5 run 中央値 87.4117）に対する比率として並べる（Rust 側の run 間ばらつきが丸め境界へ与える影響を
確認するための感度分析であり、正式集計は上記の中央値どうしの比較 37.47% である）:

| run | mma_f16 (4096) TFLOPS | PyTorch 5 run 中央値比 | 丸め後 floor |
|-----|------------------------|--------------------------|--------------|
| run1 | 34.5145 | 39.48% | 35 |
| run2 | 33.2252 | 38.01% | 35 |
| run3 | 32.7499 | 37.47% | 35 |
| run4 | 31.4362 | 35.96% | 35 |
| run5 | 32.6191 | 37.32% | 35 |

**PyTorch 参照値を正しく 5 run 中央値化した結果、Rust 側の run1〜run5 は全て floor=35 に収まり境界跨ぎ
は生じない**（旧版は PyTorch を run1 単独の 1 run 値〈84.2850〉のまま固定していたため、Rust run1 の
比率だけが 40.95% となり floor=40 相当に跨ぐ事実があったが、これは分母〈PyTorch 参照値〉が本来必要な
5 run 中央値へ未集計だったことによる見かけ上の跨ぎであり、PyTorch 側を正しく 5 run 中央値化した本表
では再現しない）。正式値 37.47% と感度分析レンジ（35.96%〜39.48%）はいずれも丸め後 **35%** で一致し、
候補下限値の安定性を裏付ける。採否・最終確定は F-5（#577）へ引き継ぐ。

**候補下限値は参考算出に留める。** REQ-8 下限値（現行確定値: f32=25%・f16=10%。
`docs/perf/performance-floor-decision.md` §9）の変更判断は本ドキュメントでは行わない。変更は F-5
（#577・人間承認タスク）のみが行う。

## 数値一致（parity）状態の限定条件

`docs/perf/cuda-gemm-mma-pipeline.md`「Phase B 完了時点の再計測（#502）」節の手順 3 と同一コマンド
（`cargo test -p backend-cuda --test parity_nonregression -- --ignored --test-threads=1`。debug/release
両プロファイルで実行し同一結果を確認）を実行した結果、`parity_baselines_do_not_regress` は
`wmma_tf32_staged 512×512×4096 seed=0xC0FFEE` の 1 件（`WmmaTf32Staged` 行を検査する内部ヘルパー
`check_wmma_tf32_staged_baseline`）で FAIL した（`MmaF16` 行は同じテスト内で検査され pass。下記表参照）。
この FAIL は `ParityBaseline::baseline_provenance_unconfirmed == true`
（`crates/backend-cuda/tests/common/parity_baseline.rs::assert_no_parity_regression`）による
fail-closed 判定であり、正本 `docs/perf/cuda-parity-baseline.md` §3 のベースライン表でもこの行
（`wmma_tf32_staged` 512×512×4096 seed=0xC0FFEE）は **`fail_count`/`mean_abs_diff` とも「未計測」**
のまま（実機再測定待ち）であることを確認した。panic メッセージの「この行は基本版カーネル専用の確定
ベースラインが未整備です」という文言は `assert_no_parity_regression` が `baseline_provenance_unconfirmed
== true` の全行（`wmma_tf32`〈基本版〉2 行・`wmma_tf32_staged` 1 行）へ共通で出す定型文であり、
staged 行固有の理由ではない。staged 行が未確定な実際の理由は、staged カーネル追加（#500）時点で
実機未到達だったため**staged 経路自体の確定ベースラインがまだ一度も記録されていない**ことである
（`crates/backend-cuda/tests/common/parity_baseline.rs:289-300` のコメント参照）。

参考実測 `fail_count=43019/262144, mean_abs_diff=4.463436e-3`（panic メッセージが出力する
`report.fail_count`/`report.mean_abs_diff`）は、`check_wmma_tf32_staged_baseline` が本セッションで
`run_wmma_tf32`（cp.async 16 バイト整列条件を満たす 512×512×4096 形状のため公開 API が staged 経路を
自動選択する）を実際に実行して得た**staged 経路自身の今回実測値**である（`wmma_tf32_opt` の記録済み
ベースライン値をそのまま転記したものではない）。この値が下表の `wmma_tf32_opt_kernel_k4096_stress`
（同一形状・同一シードで `fail_count=43019/262144, mean_abs_diff=4.463e-3`）と一致しているのは転記
ミスではなく、両者を独立に実行して得た実測値がたまたま一致したものである。**一致の理由（カーネル実装
上、staged と opt の演算結果が本当に一致するのか等）は本ドキュメントでは調査・断定しない**（推定値の
記載を禁止する `docs/perf/cuda-parity-baseline.md` §6 の方針に従う）。いずれにせよこの一致は staged
経路固有の**確定ベースラインではない**（正本にまだ記録されていない参考値に過ぎない）ため、この PR の
範囲では正本 `docs/perf/cuda-parity-baseline.md` への新規ベースライン追加は行わず、後続課題として
「未実施・後続作業」節へ申し送る。

**tolerance 定数・parity ロジック自体は無変更**（「計測環境」節の `git diff 86e7e7e..abaa94e` 確認結果）
だが、これは非後退の**必要条件**であって**十分条件ではない**——比較対象の確定ベースラインが正本に
存在しない以上、「後退していないこと」自体を確認する手段がない。したがって `wmma_tf32_staged`
512×512×4096 は「後退なし」ではなく**判定不能（fail-closed）**として扱う。

**f32 候補下限（50%）への影響に関する重要な注記**: `crates/backend-cuda/src/gemm.rs::launch_wmma_tf32`
（「経路×形状 TFLOPS 実測」節の計測に使う `cuda_floor_bench.rs::measure_wmma_tf32` が呼ぶ関数）は
`run_wmma_tf32` と同一の 3 段選択（staged が利用可能かつ整列形状なら staged を最優先。分岐条件は
`self.wmma_tf32_staged.is_some() && wmma_tf32_staged_alignment_ok(n, k)`）で経路を選ぶ
（`gemm.rs:1458-1461`）。今回の実機セッションでは `wmma_tf32_staged_available() == true`
（`check_wmma_tf32_staged_baseline` の事前 assert が通過した事実から確認済み）であり、判定対象形状
（512/1024/2048/4096）はすべて `n%4==0 && k%4==0`（cp.async 整列条件）を満たす。staged 分岐は
`validate_wmma_tf32_staged_k_bound(k)?` が `Err` を返せば早期リターンし opt へフォールスルーしない
（`gemm.rs:1463-1475`）ため、4096 形状で TFLOPS が実測できている以上 staged 分岐が実際に成功実行され
たことも確定している。よって**本ドキュメントの「WMMA(TF32) opt」列として記録した f32 最良経路の実測
値は、実際には staged 経路を計測したものである**（推測ではなく上記のコード上の分岐条件・実測データ
から確定できる事実）。`cuda_floor_bench.rs` の起動時診断メッセージ「WMMA(TF32) opt AVAILABLE」は
`wmma_tf32_opt_available()` のみを確認する表示であり staged の選択有無は出力しないため、実行ログ単体
からはこの事実を読み取れない（`floor_bench_run1.log:6`）。したがって f32 候補下限（50%）の性能値
採用ゲートは、**staged 経路の parity 判定不能（上記）の影響を直接受ける**——f16 候補（`mma_f16`。
`CudaMmaGemm::run_f16` に staged/opt の分岐はなく本注記の対象外）とは異なり、区分 1（後退なし確認済み。
`wmma_tf32` opt）ではなく区分 2（判定不能）の経路の実測値である。後続課題は「未実施・後続作業」節へ
申し送る。

本セッションで実行した parity 系テスト全体（`parity_nonregression`・`cargo test -p backend-cuda --lib
-- --ignored`・`cargo test -p backend-cuda --test cpu_cuda_mma_parity -- --ignored`）の結果を、各テスト
が実際に比較する対象（正本ベースラインとの相対比較 `assert_no_parity_regression` か、REQ-2 tolerance
との絶対比較 `backend_cpu::assert_parity` か）に基づき 3 区分に分けて整理する:

| 候補下限の経路 | テスト | 判定方式 | fail 内容（実測） | 区分 |
|---|---|---|---|---|
| `wmma_tf32`（基本版） | `wmma_tf32_basic_kernel_parity_does_not_regress` | `assert_no_parity_regression`（`WmmaTf32` 行。`baseline_provenance_unconfirmed == true`） | 32×32×32: fail_count=154/1024（15.04%）／256×256×4096 stress: fail_count=10647/65536（16.25%） | **判定不能（fail-closed）** |
| `wmma_tf32_staged` | `parity_baselines_do_not_regress`（内部で `check_wmma_tf32_staged_baseline`） | `assert_no_parity_regression`（`WmmaTf32Staged` 行。`baseline_provenance_unconfirmed == true`） | 512×512×4096: fail_count=43019/262144（16.41%） | **判定不能（fail-closed）** |
| `wmma_tf32` opt（3 行） | `wmma_tf32_opt_kernel_parity_does_not_regress` | `assert_no_parity_regression`（`WmmaTf32Opt` 3 行。いずれも `baseline_provenance_unconfirmed == false`） | 全 3 行 pass（512×512×512／64×64×64／512×512×4096） | **後退なしを確認できた** |
| `mma_f16` | `parity_baselines_do_not_regress`（内部で `check_mma_f16_baseline`） | `assert_no_parity_regression`（`MmaF16` 行。`baseline_provenance_unconfirmed == false`） | 256×256×4096: pass（staged 行のみ FAIL、`MmaF16` 行は非後退確認済み） | **後退なしを確認できた** |
| `wmma_tf32` opt（単体テスト） | `wmma_tf32_opt_kernel_k4096_stress` | `backend_cpu::assert_parity`（REQ-2 tolerance との絶対比較。ベースライン相対比較ではない） | 512×512×4096: fail_count=43019/262144（**16.41%**） | **既知恒常 fail の再現** |
| `wmma_tf32` opt（単体テスト） | `wmma_tf32_opt_kernel_matches_reference_across_shapes` | `backend_cpu::assert_parity`（同上） | m=n=k=64: fail_count=699/4096（**17.06%**） | **既知恒常 fail の再現** |
| `mma_f16`（単体テスト） | `mma_f16_k4096_stress`（`cpu_cuda_mma_parity.rs`） | `assert_parity` 相当（同上） | 256×256×4096: fail_count=101/65536（**0.154%**） | **既知恒常 fail の再現** |

`wmma_tf32_staged_kernel_exceeds_opt_kernel_tflops_at_4096`（parity ではなく TFLOPS 比較の性能テスト。
staged 経路の parity 非後退とは無関係）・`mma_f16_cross_check_against_wmma_f16`・
`mma_f16_matches_reference_across_shapes` も pass だが、いずれも上表のいずれの行とも異なる独立した検査
（形状網羅・相互検算用）であり非後退判定の根拠には数えない。

（`nvrtc::jit_cache_bench_tests::*` の 2 件 FAIL は `/tmp` 配下の cache root pin に関する環境依存
エラーで、GEMM スループット・parity とは無関係のため本ドキュメントのスコープ外。#534〈C-12〉の
JIT キャッシュベンチ側の既知事象として申し送る。）

**非後退判定の最終整理（3 区分）**:

1. **後退なしを確認できた経路**（`assert_no_parity_regression` によるベースライン相対比較で
   `baseline_provenance_unconfirmed == false` の行が全て pass）: `wmma_tf32_opt`（512×512×4096・
   64×64×64・512×512×512。`wmma_tf32_opt_kernel_parity_does_not_regress`）・`mma_f16`（256×256×4096。
   `parity_baselines_do_not_regress` の `MmaF16` 行）。いずれも記録済み確定ベースラインを上回って
   おらず後退なしと判定できる
2. **判定不能（fail-closed）**: `wmma_tf32`（基本版）2 行・`wmma_tf32_staged`（512×512×4096）1 行。
   `baseline_provenance_unconfirmed == true` のため正本に比較対象となる確定ベースラインがなく、
   tolerance 定数・parity ロジック不変というだけでは非後退を主張できない。`wmma_tf32_staged` は
   上記「f32 候補下限（50%）への影響に関する重要な注記」のとおり、f32 候補下限（`wmma_tf32` 最良経路）
   の実測に使われた経路そのもの（`launch_wmma_tf32` の 3 段選択・cp.async 整列条件・staged 分岐の
   非フォールスルーから確定）であるため、この判定不能は f32=50% 候補下限に**直接影響する**（f16 候補
   `mma_f16` は区分 1 で確認済みのため対象外）。staged 固有ベースラインの確立は本 PR のスコープ外の
   後続課題とする（「未実施・後続作業」節参照）
3. **既知恒常 fail の再現（#389 §5.3 範囲内）**: `backend_cpu::assert_parity`（REQ-2 tolerance との
   絶対比較。ベースライン相対比較ではない）による直接判定で、TF32 opt 単体テストの k4096 stress
   16.41%・shape grid 17.06%、f16 単体テストの K4096 stress 0.154% がいずれも #389 §5.3 が記録した
   恒常 fail 範囲内で再現した。これらは非後退ゲート自体の合否とは別建てのテストであり、区分 1 の
   `assert_no_parity_regression` 判定結果とは独立した事実である（数値は同一形状・同一シードの
   ベースライン記録元でもあるため一致するが、判定方式が異なる点に注意）

- 区分 1（後退なし確認済み）の経路でも、既知の parity 恒常 fail（#389 §5.3・区分 3）自体は解消されて
  いない
- 本 candidate floor（f32=50%・f16=35%）のうち f16=35%（`mma_f16`）は区分 1（後退なし確認済み）の経路
  の実測値である。f32=50%（`wmma_tf32` 最良経路）は上記注記のとおり実際には `wmma_tf32_staged` 経路の
  実測値であり、区分 2（判定不能）に属する
- 候補下限は #186（REQ-2 閾値改定。spec リポジトリ側対応待ち）の解決前は #577 の下限確定根拠として
  単独採用できない（#186 限定条件は継続）。本 PR ではこの限定に加え f32=50% が区分 2（判定不能）の
  実測値であることも踏まえ、性能値を確定させない（候補値としての記録に留める）

### 追記（#726・2026-08-19）: `wmma_tf32_staged` の判定不能状態は解消済み

上記 3 区分の区分 2 のうち `wmma_tf32_staged`（512×512×4096 seed=0xC0FFEE）は、イシュー #726 で
DGX Spark GB10 実機（コミット 06b24b4）にて確定ベースラインを確立した（fail_count=43019/262144
〈16.4%〉・mean_abs_diff=4.463436e-3。release/debug 各 2 回・計 4 回で同一値。正本
`docs/perf/cuda-parity-baseline.md` §3 表・§8.5 表参照）。fixture
（`crates/backend-cuda/tests/common/parity_baseline.rs` の `WmmaTf32Staged` 行）は確定値 +
`baseline_provenance_unconfirmed: false` へ更新済みで、`parity_baselines_do_not_regress` は staged
行を含む全対象行で pass する（更新後の実機再実行で確認済み）。これにより f32 候補下限 50% の根拠
経路（staged）の parity 非後退判定は可能になり、`docs/perf/performance-floor-decision.md` §10 の
限定条件 4 は解消された。本節上記の「判定不能（fail-closed）」記述は 2026-08-18 実測時点の記録として
保存する。なお `wmma_tf32`（基本版）2 行の判定不能（基本版カーネル単独の再測定未了）と、#186・
#389 §5.3 由来の限定条件 1〜3 は #726 のスコープ外であり継続する。

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
  `cargo test -p backend-cuda --test cpu_cuda_mma_parity -- --ignored` — 既知 fail のみで新規 fail は
  ない。非後退の判定可否は経路別（上記「数値一致（parity）状態の限定条件」節の 3 区分）に従う——
  `wmma_tf32` opt・`mma_f16` は後退なしを確認できたが、`wmma_tf32`（基本版）・`wmma_tf32_staged`
  （f32 候補下限の実測経路）は `baseline_provenance_unconfirmed` により判定不能
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

### PyTorch 参照値・run1（`pytorch_size{512,1024,2048,4096}.log`）

初回（run1）計測時の生ログ。この時点では 1 run のみで確定していたため、`cuda_floor_bench` の
`CUDA_FLOOR_BENCH_PYTORCH_*` env override は本 run の値で固定して 5 回の `cuda_floor_bench` 実行
（`floor_bench_run{1..5}.log`）に共通適用していた（下記「`cuda_floor_bench` 生ログ全文」節の各 run の
`f32_best_over_pytorch`/`f16_candidate_over_pytorch` はこの run1 単独の PyTorch 値に対する比率であり、
本ドキュメント本文の確定値〈PyTorch 5 run 中央値に対する比率〉とは異なる。詳細は次項）。

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

### PyTorch 参照値・run2〜run5（codex レビュー P1 対応の追加計測）

`pytorch_size{512,1024,2048,4096}_run{2,3,4,5}.log`（`median_tflops` のみ抜粋。q1/q3 は上記「PyTorch
参照値の再集計（5 run 中央値）」節の表を参照）。

```
                f32 median_tflops                  f16 median_tflops
size   run2     run3     run4     run5     run2     run3     run4     run5
512    7.8767   7.8472   7.7997   7.8435   17.1021  16.9809  16.8956  17.0500
1024   15.6980  15.7329  15.6321  15.6121  55.9006  56.0644  48.7885  55.8309
2048   17.1662  17.1169  17.0845  17.1582  92.5320  92.4767  92.6039  92.7398
4096   17.4721  17.4108  17.4467  17.3663  87.4117  81.9354  88.0821  88.5662
```

（1024 f16 run4 = 48.7885 は上記「PyTorch 異常値の記録」節のとおり一過性ジッタと推定し、5 値中央値化
のため採用値へは影響しない。）

### `cuda_floor_bench` 生ログ全文（`floor_bench_run{1,2,3,4,5}.log`）

各 run は 1 回の実行として全形状を通しで実行しており生ログの原文はそのまま残す。ログ中の
`f32_best_over_pytorch`/`f16_candidate_over_pytorch`（末尾 2 行の丸め結果を含む）は、`cuda_floor_bench`
バイナリが内部で参照する `CUDA_FLOOR_BENCH_PYTORCH_*`（run1 単独の PyTorch 1 run 値。上記「PyTorch 参照
値・run1」節）に対する比率であり、本ドキュメント本文の確定値（Rust 5 run 中央値 ÷ PyTorch 5 run 中央値。
「対 PyTorch 比」節）とは分母が異なるため一致しない。本文の確定値の再現には、下表の各セルの
`*_tflops` 値（Rust 側）と「PyTorch 参照値の再集計」節の 5 run 中央値（PyTorch 側）を独立に組み合わせる
必要がある。

```
run1 size=512  tiled_f32_tflops=2.1037 wmma_tf32_tflops=8.3220 wmma_f16_tflops=4.1323 mma_f16_tflops=15.1556 f32_best_over_pytorch=106.00% f16_candidate_over_pytorch=89.07%
run1 size=1024 tiled_f32_tflops=2.3838 wmma_tf32_tflops=12.7353 wmma_f16_tflops=8.8645 mma_f16_tflops=33.4457 f32_best_over_pytorch=81.37% f16_candidate_over_pytorch=59.88%
run1 size=2048 tiled_f32_tflops=2.3432 wmma_tf32_tflops=14.3842 wmma_f16_tflops=8.9661 mma_f16_tflops=48.3274 f32_best_over_pytorch=83.00% f16_candidate_over_pytorch=52.15%
run1 size=4096 tiled_f32_tflops=1.9723 wmma_tf32_tflops=9.0655 wmma_f16_tflops=4.3620 mma_f16_tflops=34.5145 f32_best_over_pytorch=51.88% f16_candidate_over_pytorch=40.95%
run1 CUDA f32 candidate optimized floor (rounding rule applied to min ratio 51.88%) = 50%
run1 CUDA f16 candidate optimized floor (rounding rule applied to min ratio 40.95%) = 40%

run2 size=512  tiled_f32_tflops=2.0896 wmma_tf32_tflops=8.2687 wmma_f16_tflops=4.1191 mma_f16_tflops=15.3638 f32_best_over_pytorch=105.32% f16_candidate_over_pytorch=90.29%
run2 size=1024 tiled_f32_tflops=2.3820 wmma_tf32_tflops=12.6453 wmma_f16_tflops=8.8470 mma_f16_tflops=33.5460 f32_best_over_pytorch=80.80% f16_candidate_over_pytorch=60.06%
run2 size=2048 tiled_f32_tflops=2.3422 wmma_tf32_tflops=14.3540 wmma_f16_tflops=5.6792 mma_f16_tflops=48.3383 f32_best_over_pytorch=82.82% f16_candidate_over_pytorch=52.16%
run2 size=4096 tiled_f32_tflops=1.9847 wmma_tf32_tflops=9.0109 wmma_f16_tflops=4.3521 mma_f16_tflops=33.2252 f32_best_over_pytorch=51.56% f16_candidate_over_pytorch=39.42%
run2 CUDA f32 candidate optimized floor (rounding rule applied to min ratio 51.56%) = 50%
run2 CUDA f16 candidate optimized floor (rounding rule applied to min ratio 39.42%) = 35%

run3 size=512  tiled_f32_tflops=2.0896 wmma_tf32_tflops=8.1960 wmma_f16_tflops=4.1111 mma_f16_tflops=15.2937 f32_best_over_pytorch=104.40% f16_candidate_over_pytorch=89.88%
run3 size=1024 tiled_f32_tflops=2.3811 wmma_tf32_tflops=12.6168 wmma_f16_tflops=8.8191 mma_f16_tflops=33.7995 f32_best_over_pytorch=80.62% f16_candidate_over_pytorch=60.51%
run3 size=2048 tiled_f32_tflops=2.3404 wmma_tf32_tflops=14.3201 wmma_f16_tflops=7.1973 mma_f16_tflops=47.9647 f32_best_over_pytorch=82.63% f16_candidate_over_pytorch=51.76%
run3 size=4096 tiled_f32_tflops=1.9798 wmma_tf32_tflops=9.0164 wmma_f16_tflops=4.3508 mma_f16_tflops=32.7499 f32_best_over_pytorch=51.60% f16_candidate_over_pytorch=38.86%
run3 CUDA f32 candidate optimized floor (rounding rule applied to min ratio 51.60%) = 50%
run3 CUDA f16 candidate optimized floor (rounding rule applied to min ratio 38.86%) = 35%

run4 size=512  tiled_f32_tflops=2.1027 wmma_tf32_tflops=8.2769 wmma_f16_tflops=4.1252 mma_f16_tflops=15.2382 f32_best_over_pytorch=105.43% f16_candidate_over_pytorch=89.56%
run4 size=1024 tiled_f32_tflops=2.3840 wmma_tf32_tflops=12.7486 wmma_f16_tflops=8.8892 mma_f16_tflops=32.6009 f32_best_over_pytorch=81.46% f16_candidate_over_pytorch=58.37%
run4 size=2048 tiled_f32_tflops=2.3425 wmma_tf32_tflops=14.3326 wmma_f16_tflops=7.7525 mma_f16_tflops=48.6295 f32_best_over_pytorch=82.70% f16_candidate_over_pytorch=52.48%
run4 size=4096 tiled_f32_tflops=1.9722 wmma_tf32_tflops=9.0725 wmma_f16_tflops=4.3634 mma_f16_tflops=31.4362 f32_best_over_pytorch=51.92% f16_candidate_over_pytorch=37.30%
run4 CUDA f32 candidate optimized floor (rounding rule applied to min ratio 51.92%) = 50%
run4 CUDA f16 candidate optimized floor (rounding rule applied to min ratio 37.30%) = 35%

run5 size=512  tiled_f32_tflops=2.0893 wmma_tf32_tflops=8.2403 wmma_f16_tflops=4.1100 mma_f16_tflops=15.4202 f32_best_over_pytorch=104.96% f16_candidate_over_pytorch=90.63%
run5 size=1024 tiled_f32_tflops=2.3830 wmma_tf32_tflops=12.6096 wmma_f16_tflops=8.7959 mma_f16_tflops=33.3124 f32_best_over_pytorch=80.57% f16_candidate_over_pytorch=59.64%
run5 size=2048 tiled_f32_tflops=2.3428 wmma_tf32_tflops=14.1789 wmma_f16_tflops=8.2042 mma_f16_tflops=48.3013 f32_best_over_pytorch=81.81% f16_candidate_over_pytorch=52.12%
run5 size=4096 tiled_f32_tflops=1.9682 wmma_tf32_tflops=9.0745 wmma_f16_tflops=4.3546 mma_f16_tflops=32.6191 f32_best_over_pytorch=51.93% f16_candidate_over_pytorch=38.70%
run5 CUDA f32 candidate optimized floor (rounding rule applied to min ratio 51.93%) = 50%
run5 CUDA f16 candidate optimized floor (rounding rule applied to min ratio 38.70%) = 35%
```

各セルの q1/q3 は「経路×形状 TFLOPS 実測」節の表へ転記済み（当該セルの中央値を与えた run のもの）。
上記 5 run 全文と PyTorch 5 run 中央値（前項）を独立に組み合わせることで、「対 PyTorch 比」節の全セルを
第三者が再計算・検算できる。

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
  ドキュメントは性能値採用の前提ゲートとして経路別の非後退確認結果（上記「数値一致（parity）状態の
  限定条件」節の 3 区分。`wmma_tf32_staged` は判定不能）を記録するに留め、最終確認は #575（同 §8）が
  行う
- **#577（Phase F-5・人間承認）**: REQ-8 下限値の最終確定・`docs/spec/04-requirements.md` への反映判断
  （`docs/spec/` は本リポでは編集しない）
- **#569（Phase F 親）・#579**: 全バックエンド横断の集約・`docs/performance-targets.md` 更新

## 未実施・後続作業

- **実機実測**: 「状態」節のとおり 2026-08-18 実測完了。本節は完了扱い
- **候補下限値の最終確定・REQ-8 反映判断**: F-5（#577・人間承認）が本ドキュメントの実測結果
  （f32 候補 50%・f16 候補 35%〈境界注記あり〉）を受けて対応する（f32=50% は `wmma_tf32_staged` 経路の
  実測値であり、同経路の parity 非後退は判定不能。「数値一致（parity）状態の限定条件」節「f32 候補
  下限（50%）への影響に関する重要な注記」参照）
- **parity 非後退の最終確認**: F-4（#575）が本ドキュメントの経路別の非後退確認結果（3 区分。上記
  「数値一致（parity）状態の限定条件」節）を受けて最終確認する
- **コストモデル選定・JIT shape 特化経路（`gemm_auto.rs::CudaGemmAuto`／`run_specialized_mma_f16`）の
  本番化判断**: 「実測バイナリ（経路カバレッジ確認結果）」節のとおり、この経路は現時点で
  `internal-diagnostics` feature 限定かつ `CudaBackendOps::gemm` からも到達しないため本イシューの
  計測対象から意図的に除外した。本番経路へ組み込む判断自体は本イシューのスコープ外（別イシュー）であり、
  組み込まれた場合は `cuda_floor_bench.rs` への経路追加が再度必要になる
- **`docs/perf/cuda-gemm-wmma-tf32-phase-b.md` §7 の未計測テンプレート**: 実機セッションが本ドキュメント
  と合わせて転記対象とするかは実機セッション側の判断に委ねる（本イシューのスコープ外）
- **`cuda_floor_bench.rs` の staged 可用性診断出力の追加・`wmma_tf32_staged` 固有ベースラインの確立
  （codex レビュー P1 対応で判明。「数値一致（parity）状態の限定条件」節「f32 候補下限（50%）への
  影響に関する重要な注記」参照）**: 本ドキュメントの f32 候補下限（50%）は `wmma_tf32_staged` 経路の
  実測値であることをコード上の分岐条件（`gemm.rs::launch_wmma_tf32` の 3 段選択・cp.async 整列条件・
  staged 分岐の非フォールスルー）から確定済みだが、`cuda_floor_bench.rs` の起動時診断メッセージは
  `wmma_tf32_opt_available()` のみを表示し staged の選択有無を出力しないため、実行ログ単体からは
  この経路を再構成できない。後続セッションで (a) `cuda_floor_bench.rs` の診断メッセージへ
  `wmma_tf32_staged_available()` の出力を追加し、(b) `wmma_tf32_staged` 512×512×4096 の確定
  ベースラインを正本 `docs/perf/cuda-parity-baseline.md` へ実機実測とセットで記録することが必要
  （推定値の記載は禁止）。本 PR のスコープ外（別イシューへの切り出しはユーザー承認を得て行う）。
  → **(b) はイシュー #726（2026-08-19）で完了**（「数値一致（parity）状態の限定条件」節の追記参照）。
  **(a) はイシュー #732（2026-08-19）で完了**（起動時診断へ `wmma_tf32_staged_available()` の
  AVAILABLE／UNAVAILABLE + 理由を出力し、staged 不能時は実測の経路 provenance が REQ-8 根拠・#726
  ベースラインと異なる旨を警告する。DGX Spark GB10 実機で出力確認済み）。本項は完了扱い
