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
python3 docs/spec/03-poc/poc-v2-3-cuda-gemm/code/pytorch/gemm_bench_torch_cuda.py <size> 20 20

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
| GPU（`CudaDevice::name()`） | （未計測） |
| compute capability（`CudaDevice::compute_capability()`） | （未計測） |
| driver バージョン（`nvidia-smi`） | （未計測） |
| rustc | （未計測） |
| commit SHA（`.rev-stamp` と転送後の値が一致確認済みであること） | （未計測） |
| 実施日 | （未計測） |
| PyTorch 参照値の出典（`pytorch reference provenance:` 行を転記。実機個体名はマスク） | （未計測） |
| 計測プロトコル | `bench_harness::protocol::run`（warmup 20 回・計測 20 回・中央値/Q1/Q3。TASK-8.1） |
| 決定的シード | `0xC0FFEE`（`cuda_floor_bench.rs::SEED`） |
| GPU 排他性（実行前後） | （未計測。`utilization.gpu` 0% ・第三プロセス非介在を確認すること） |
| 反復回数 | `cuda_floor_bench` を 3 回反復実行し、run 間中央値を代表値として採用する（#390 先例と同方式） |

## 経路×形状 TFLOPS 実測（実測時に記入）

各セルは `<中央値>(q1=<Q1由来値>,q3=<Q3由来値>)` の形式で `size=<N> ...` 出力行から転記する
（`cuda_floor_bench.rs::TflopsSample`）。3 run（run1/run2/run3）の中央値ベース TFLOPS 値のうち run 間の
中央値を代表値として記載し、括弧内に run1〜run3 の中央値レンジを注記する（`docs/perf/cuda-floor-remeasurement.md`
「経路×形状 TFLOPS 実測」節と同形式）。

| M=N=K | tiled f32（中央値/Q1/Q3、run 間レンジ） | WMMA(TF32) opt（同左） | WMMA f16 opt（同左） | mma.sync f16（同左） | f32 最良経路 | f16 candidate 経路 |
|-------|-----------------------------|-----------------------------|-----------------------------|-----------------------------|---------------|---------------------|
| 512（参考値） | （未計測） | （未計測） | （未計測） | （未計測） | （未計測） | （未計測） |
| 1024（参考値） | （未計測） | （未計測） | （未計測） | （未計測） | （未計測） | （未計測） |
| 2048 | （未計測） | （未計測） | （未計測） | （未計測） | （未計測） | （未計測） |
| 4096 | （未計測） | （未計測） | （未計測） | （未計測） | （未計測） | （未計測） |

## 対 PyTorch 比（実測時に記入）

| M=N=K | f32 最良（実測大小比較で選出） / PyTorch f32 比 | f16 candidate（実測大小比較で選出） / PyTorch f16 比 |
|-------|----------------------------------------------------|------------------------------------------------------|
| 512（参考値） | （未計測） | （未計測） |
| 1024（参考値） | （未計測） | （未計測） |
| 2048 | （未計測） | （未計測） |
| 4096 | （未計測） | （未計測） |

## 丸め適用後の候補下限値（実測時に記入）

| 精度 | 判定対象形状の最小比率（2048/4096） | 丸め規則適用後の候補下限値 | #390 実測値（f32=25%・f16=10%）との比較 |
|------|--------------------------------------|------------------------------|------------------------------|
| f32  | （未計測） | （未計測） | （未計測） |
| f16  | （未計測） | （未計測） | （未計測） |

**候補下限値は参考算出に留める。** REQ-8 下限値（現行確定値: f32=25%・f16=10%。
`docs/perf/performance-floor-decision.md` §9）の変更判断は本ドキュメントでは行わない。変更は F-5
（#577・人間承認タスク）のみが行う。

## 数値一致（parity）状態の限定条件

実測完了後、選出された経路（`wmma_tf32`／`mma_f16`）が `docs/perf/cuda-floor-remeasurement.md`「数値
一致（parity）状態の限定条件」節の恒常 fail 対象と一致するかを確認し、以下を明記すること:

- 後退の有無（tolerance 定数不変・fail 比率/mean_abs_diff がベースライン以下であること）
- 後退が無い場合でも、既知の parity 恒常 fail（#389 §5.3）自体は解消されていない旨
- 本 candidate floor は数値一致未達の経路の実測値であり、#186（REQ-2 閾値改定。spec リポジトリ側対応
  待ち）の解決前は #577 の下限確定根拠として単独採用できない旨

## 状態: 未計測。実機セッションで消化

本ドキュメントは Linux worktree で作成され、CUDA 実機（DGX Spark GB10）が同一セッションで到達できない
ため計測手順・記録テンプレートのみを整備した（#502・#534（C-12）・#572 先例と同方式）。実機到達可能な
セッションが「計測手順」節の手順で計測し、上記「計測環境」「経路×形状 TFLOPS 実測」「対 PyTorch 比」
「丸め適用後の候補下限値」「数値一致（parity）状態の限定条件」の各表・節を実測値で埋めること。

内部ホスト名等の実値は書かない（#461 のプレースホルダ方針。実測時の原文は
`docs/real-hardware-verification-env.local.md` へ記録する）。

## 動作確認（Linux セッションで実施済み）

- `cargo build --workspace --locked` — `cudarc` 動的ロード契約（CUDA toolkit 非搭載環境でもビルド成立
  する。`.claude/rules/coding-rust.md`）を崩していないことを確認済み
- `cargo build -p backend-cuda --example cuda_floor_bench --release` — example のビルド成立（無変更）
- `cargo fmt --all -- --check`
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`
- `cargo test --workspace`（Linux 実行分。実機依存 `#[ignore]` テストは除外）
- `git diff origin/main -- crates/backend-cuda/src crates/backend-cuda/tests/common crates/bench-harness`
  が tolerance 定数（`RELATIVE_TOLERANCE`・`ABSOLUTE_RESCUE_THRESHOLD`）・parity fixture・`FloorSpec`・
  カーネルソースに差分を持たないことを確認（#502・#390 の検証項目を踏襲）

## 役割分担（二重管理を避ける）

- **`docs/perf/cuda-floor-remeasurement.md`（#157/#390）**: TASK-8.3c 時点（Phase B/C 適用前）の確定
  記録。本ドキュメントでは書き換えない
- **本ドキュメント（#571・Phase F-1）**: Phase B/C 適用後の再計測記録。実測値記入は実機セッションへ
  申し送り
- **`docs/perf/cuda-gemm-mma-pipeline.md`「Phase B 完了時点の再計測（#502）」節**: Phase B 単独の実機
  未到達記録。実機セッションが本ドキュメントと合わせて埋める判断は実機セッション側に委ねる
- **`docs/perf/cuda-jit-cache-benchmark.md`（#534・C-12）**: Phase C（JIT キャッシュ）固有の初回コンパ
  イル／2 回目ロード時間の実機未到達記録。本ドキュメントのスループット計測とは別レイヤ
- **#575（Phase F-4）**: parity 非後退の最終確認。本ドキュメントは性能値採用の前提ゲートとして非後退を
  確認するに留め、最終確認は #575 が行う
- **#577（Phase F-5・人間承認）**: REQ-8 下限値の最終確定・`docs/spec/04-requirements.md` への反映判断
  （`docs/spec/` は本リポでは編集しない）
- **#569（Phase F 親）・#579**: 全バックエンド横断の集約・`docs/performance-targets.md` 更新

## 未実施・後続作業

- **実機実測**: 「状態」節のとおり本イシューでは未実施。CUDA 実機（DGX Spark GB10）到達可能なセッション
  へ申し送る
- **候補下限値の最終確定・REQ-8 反映判断**: F-5（#577・人間承認）が実測完了後に対応する
- **parity 非後退の最終確認**: F-4（#575）が実測完了後に対応する
- **コストモデル選定・JIT shape 特化経路（`gemm_auto.rs::CudaGemmAuto`／`run_specialized_mma_f16`）の
  本番化判断**: 「実測バイナリ（経路カバレッジ確認結果）」節のとおり、この経路は現時点で
  `internal-diagnostics` feature 限定かつ `CudaBackendOps::gemm` からも到達しないため本イシューの
  計測対象から意図的に除外した。本番経路へ組み込む判断自体は本イシューのスコープ外（別イシュー）であり、
  組み込まれた場合は `cuda_floor_bench.rs` への経路追加が再度必要になる
- **`docs/perf/cuda-gemm-wmma-tf32-phase-b.md` §7 の未計測テンプレート**: 実機セッションが本ドキュメント
  と合わせて転記対象とするかは実機セッション側の判断に委ねる（本イシューのスコープ外）
