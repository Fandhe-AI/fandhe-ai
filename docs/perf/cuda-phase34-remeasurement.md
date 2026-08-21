# CUDA Phase 3/4 完了後 f32/f16 対 PyTorch 比 確定計測 記録（#807）

イシュー #807「perf(backend-cuda): Phase 3/4 完了後の実機確定計測とベースライン追補」の実測記録。
GEMM OSS 比較ギャップ改修ツリー（ルート #785）Phase 4（親 #789「CUDA タイル形状拡大」）の締め計測に
対応する。前回ベースライン `docs/perf/cuda-optimized-remeasurement.md`（#571・Phase F-1・
2026-08-18 実測）と同形式で記録し、書き換えない（別ファイル新設。#571 が `cuda-floor-remeasurement.md`
を書き換えず新設した構成、Metal 側 #572 が `metal-floor-remeasurement.md` を新設した構成と対称）。

## 1. 位置づけ・#804/#806 との関係（必読）

依存イシュー #804（mma_f16 ブロックタイル拡大・ステージ数増）・#806（TF32 タイル拡大）はいずれも
CLOSED（PR #831・#832 マージ済み）だが、**両 PR とも「診断機構・机上候補表の整備」までで完了しており、
本番カーネル定数（ブロックタイル・ステージ数）は変更されていない**（Step F フォールバック。実機・
ローカル CUDA toolkit の双方に到達できず実測が「実行待ち」のまま引き継がれた。出典:
`docs/perf/cuda-gemm-mma-block-tile-stages.md` §4「実機実測結果」〈全欄「未実測」〉・§5「判断」
〈「実機実測が完了するまで採用構成は未確定」〉・§6「引き継ぎ事項」、`docs/perf/
cuda-gemm-mma-tf32-block-tile.md` §7「実測表（実行待ち）」・§8「引き継ぎ事項」）。

したがって本イシュー（#807）の確定計測は **main HEAD の本番経路そのまま**（#804/#806 のタイル拡大
候補は本番未結線）を対象とする。この記録は「タイル拡大適用後の性能」ではなく「Phase 3/4 ツリー
（#785 配下の一連のイシュー）完了時点での実機到達性確認と、到達可能になった際に確定計測を行うための
計測線の整備」である。実測値が得られた場合も、その値は #804/#806 適用前の本番経路（main HEAD）の値
であることを明記して数値の解釈を誤らせない。

## 2. 経路カバレッジ再確認(読み取り調査。本実装セッションで実施)

`crates/backend-cuda/examples/cuda_floor_bench.rs`（計測対象コミット `6259f95`）を対象に、以下を
確認した:

- **計測入口は無変更**: `grep -n "run_wmma_tf32\|launch_wmma_tf32\|launch_f16"
  crates/backend-cuda/examples/cuda_floor_bench.rs` の結果、`CudaGemm::run_wmma_tf32`／
  `CudaGemm::launch_wmma_tf32`／`CudaWmmaGemm::launch_f16`／`CudaMmaGemm::launch_f16` を入口とする
  構成は #571 時点（`docs/perf/cuda-optimized-remeasurement.md`「実測バイナリ」節）から変わっていない。
  4 経路（tiled f32／WMMA(TF32) opt／WMMA f16 opt／`mma.sync` f16 パイプライン）・形状集合
  （512/1024/2048/4096）・計測プロトコル（`bench_harness::protocol::run`・warmup 20/計測 20・
  決定的シード `0xC0FFEE`）も無変更
- **`internal-diagnostics` feature は既定ビルドに混入しない**: `crates/backend-cuda/Cargo.toml` の
  `[features]` に `internal-diagnostics = []` が定義され、`default` feature リストには含まれない
  （`[package]` 節に `default-features` 指定なし＝既定で無効）。`#804`／`#806` が追加した診断専用
  example（`mma_ptx_dump` 系。`required-features = ["internal-diagnostics"]`）は
  `cargo build --example cuda_floor_bench`（feature 指定なし）では到達しないことを確認した
- 結論: **本イシューでの `cuda_floor_bench.rs` への追加変更は不要**（#571 が下した同判断を踏襲）

## 3. 実機到達性ゲート判定（本実装セッション・2026-08-21）

- `ls docs/real-hardware-verification-env.local.md` → `No such file or directory`（実ホスト名を記す
  Git 管理外ファイルが本 worktree に存在しない。`.example` テンプレートのみ存在）
- `CUDA_NODE` 環境変数・SSH config 上のノード alias も未設定のため、SSH 到達性確認
  （`docs/real-hardware-verification-env.md` の手順）に進めない
- よって実機到達性ゲートは**不達**と判定し、実測は行わず安全側（推定値・外挿を記載しない）に倒す
  （#502・#571・#572・#799・#803・#804・#806 の確立済み先例と同じ判断）

**実測値の記入は CUDA 実機（DGX Spark GB10）到達可能なセッションへ申し送る**（下記「状態」節参照）。

## 4. 数値一致（parity）確認（実機セッションで性能値採用より先に実行すること）

性能値採用の前提ゲートとして、実機セッションは計測前に次のコマンドで parity テスト群を実行し、
**非後退**（tolerance 定数不変・fail 比率/mean_abs_diff が `docs/perf/cuda-parity-baseline.md` §3
のベースライン以下）を確認すること。既存 tolerance 定数・REQ-2 統一複合判定（相対誤差 1e-3 未満
または絶対誤差 1e-5 未満）は本イシューでは変更しない。

```sh
cargo test -p backend-cuda --test parity_nonregression -- --ignored --test-threads=1
```

`--test-threads=1` は必須: 同一バイナリ内 `#[test]` の並列実行は GPU 時間分割により計測値を約 5 倍
歪ませた実績がある（`docs/perf/cuda-floor-remeasurement.md`「tiled f32 @4096 のバイナリ間乖離の突合
結果」節）。debug/release 両プロファイルで実行し同一結果を確認すること。

**既知の前提**: `wmma_tf32`・`mma_f16` は #389 §5.3 が記録した parity 恒常 fail 対象である
（TF32 経路 5 件・f16 K=4096 tail 3 件。REQ-2 閾値改定は #186〈spec リポジトリ側対応待ち〉へ引き渡し
済み）。後退が無いことを確認できた場合でも、この既知 fail が解消したことにはならない。性能値採用は
「非後退」の確認のみを条件とし、fail 自体の解消は本イシューのスコープ外。

`wmma_tf32_staged`（512×512×4096 seed=0xC0FFEE）の確定ベースラインは #726 で確立済み（`docs/perf/
cuda-parity-baseline.md` §3・§8.5）であり、`docs/perf/cuda-optimized-remeasurement.md`「数値一致
（parity）状態の限定条件」節が記録した「判定不能（fail-closed）」状態は解消されている。実機セッションで
後退を検出した場合は性能値を採用せず打ち切り、その旨を本ドキュメントへ記録すること。

## 5. PyTorch 参照値の実測手順（size×dtype ごと 5 run 中央値）

リポジトリ規約「ベンチは 5 回計測の中央値」（`.claude/rules/coding-rust.md`）に合わせ、PyTorch 参照値も
`cuda_floor_bench` と同数の 5 run を計測し、size×dtype ごとに run 間中央値を採る
（`docs/perf/cuda-optimized-remeasurement.md`「PyTorch 参照値の再集計」節が確立した方式を踏襲）。

```sh
# `<size>` はプレースホルダーであり、そのまま貼り付けると POSIX shell が入力リダイレクトと
# 誤解釈し `size: No such file or directory` で停止する。SIZE 変数へ実値を入れて渡すこと。
for RUN in 1 2 3 4 5; do
  for SIZE in 512 1024 2048 4096; do
    python3 docs/spec/03-poc/poc-v2-3-cuda-gemm/code/pytorch/gemm_bench_torch_cuda.py "$SIZE" 20 20
  done
done
# ↑ run ごとに出力を保存し、size×dtype ごとに 5 run の median_tflops を独立に中央値化する。
```

## 6. `cuda_floor_bench` 実測手順（env override・5 回反復）

```sh
git fetch origin
# 本イシューの実装ブランチ（bench/807-cuda-phase34-remeasurement）は PR マージ後に削除される
# 一時ブランチのため、恒久参照として使わない。以下のいずれかで対象コミットを取得する:
#   a) PR が未マージ・ブランチ現存の場合: 上記ブランチを直接 checkout してよい
#   b) マージ済みの場合: 本ドキュメントが記録するコミット SHA（6259f95c8e39b826d17deb780ea72232e00c12b6）
#      を main 上で checkout する
#   c) 上記 SHA 時点より後の実装状態を計測対象としたい場合: 最新 main を対象とする契約とし、
#      その旨（「実測時点の最新 main、コミット <SHA>」）を「状態」節に明記する

# 1. 到達性・GPU 排他性の確認（docs/real-hardware-verification-env.local.md から CUDA_NODE を取得）
ssh -o BatchMode=yes -o ConnectTimeout=10 "$CUDA_NODE" \
  'hostname && nvidia-smi --query-gpu=name,utilization.gpu --format=csv,noheader'

# 2. docs/real-hardware-verification-env.md §3 の rsync 手順でコードを転送し、.rev-stamp でリビジョン一致を確認する

# 3. 数値一致確認を性能値採用より先に行う（§4 参照）
cargo test -p backend-cuda --test parity_nonregression -- --ignored --test-threads=1

# 4. PyTorch 参照値を計 5 回計測する（§5 参照）

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

## 7. 計測環境（実測時に記入）

| 項目 | 値 |
|------|-----|
| GPU（`CudaDevice::name()`） | 未実測 |
| compute capability（`CudaDevice::compute_capability()`） | 未実測 |
| driver バージョン（`nvidia-smi`） | 未実測 |
| rustc | 未実測 |
| commit SHA（`.rev-stamp` と転送後の値が一致確認済みであること） | `6259f95c8e39b826d17deb780ea72232e00c12b6`（本ドキュメント作成時点の main HEAD。実測時に「計測対象コミットの補足」を追記） |
| 実施日 | 未実測 |
| PyTorch 参照値の出典 | 未実測 |
| 計測プロトコル | `bench_harness::protocol::run`（warmup 20 回・計測 20 回・中央値/Q1/Q3。TASK-8.1） |
| 決定的シード | `0xC0FFEE`（`cuda_floor_bench.rs::SEED`） |
| GPU 排他性（実行前後） | 未実測 |
| 反復回数 | 未実測（`cuda_floor_bench` 5 回反復・PyTorch 参照値も 5 run の予定） |

## 8. 経路×形状 TFLOPS 実測（実測時に記入）

各セルは `<中央値>(q1=<Q1由来値>,q3=<Q3由来値>)` の形式で `size=<N> ...` 出力行から転記する
（`cuda_floor_bench.rs::TflopsSample`）。経路×形状のセルごとに run1〜run5 の 5 値（各 run の
`median_tflops`）を独立に中央値化し、その中央値を与えた run の出力行から `<中央値>(q1=,q3=)` を転記
する。括弧内 `〔〕` へは run1〜run5 の中央値レンジ（5 値の最小〜最大）を注記する
（`docs/perf/cuda-optimized-remeasurement.md`「経路×形状 TFLOPS 実測」節と同形式）。

| M=N=K | tiled f32（中央値/Q1/Q3、run 間レンジ） | WMMA(TF32) opt（同左） | WMMA f16 opt（同左） | mma.sync f16（同左） | f32 最良経路 | f16 candidate 経路 |
|-------|-----------------------------|-----------------------------|-----------------------------|-----------------------------|---------------|---------------------|
| 512（参考値） | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 |
| 1024（参考値） | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 |
| 2048 | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 |
| 4096 | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 |

## 9. 対 PyTorch 比（実測時に記入）

対 PyTorch 比 = Rust セル 5 run 中央値（§8）÷ PyTorch 5 run 中央値（§5 実測結果）で size×dtype ごとに
計算する。

| M=N=K | f32 最良（実測大小比較で選出） / PyTorch f32 比 | f16 candidate（実測大小比較で選出） / PyTorch f16 比 |
|-------|----------------------------------------------------|------------------------------------------------------|
| 512（参考値） | 未実測 | 未実測 |
| 1024（参考値） | 未実測 | 未実測 |
| 2048 | 未実測 | 未実測 |
| 4096 | 未実測 | 未実測 |

判定対象形状（2048/4096）の最小比率: 未実測。

**直近の確定値（前回ベースライン。#571・2026-08-18 実測）**: 判定対象形状の対 PyTorch 比最小値
f32=51.96%（4096）・f16=37.47%（4096）。本イシューの実測はこの前回値と比較し、変化があれば
「変化の解釈」節（実測時に追記）へ記録する。前回値は #804/#806 適用前（Phase B/C 適用後・Phase 3/4
適用前）の本番経路の値であり、本イシューの実測対象（Phase 3/4 完了後・#804/#806 タイル拡大候補は
本番未結線のままの main HEAD）と同一経路である点に注意する（§1 参照。したがって理論上は前回値から
大きく変化しないと予想されるが、これは予想であり実測で確認するまで確定させない）。

## 10. 丸め適用後の候補下限値（実測時に記入。記録のみ・下限反映はユーザー承認事項）

| 精度 | 判定対象形状の最小比率（2048/4096） | 丸め規則適用後の候補下限値 | 前回値（#571・f32=50%・f16=35%）との比較 |
|------|--------------------------------------|------------------------------|------------------------------|
| f32  | 未実測 | 未実測 | 未実測 |
| f16  | 未実測 | 未実測 | 未実測 |

丸め規則は `bench_harness::rounding::floor_lower_bound`（10% 以上は 5% 刻み切り下げ）を適用する。

**本節は記録のみに留める。REQ-8 下限値（現行確定値: f32=25%・f16=10%。
`docs/perf/performance-floor-decision.md` §9）の変更判断は本ドキュメントでは行わない。** 変更は
F-5（#577・人間承認タスク）と同様の人間承認プロセスへ引き継ぐ（`.claude/rules/security.md`「自己修復
ループ固有のガードレール」・`.claude/rules/deps-policy.md` 系のユーザー承認原則と同じ扱い）。

## 11. 状態: 未実測・実機セッション待ち（2026-08-21・本実装セッション）

本ドキュメントは Linux worktree（`docs/real-hardware-verification-env.local.md` 不在・SSH 到達不能）
で計測手順・記録テンプレートの整備のみを行った（#502・#571・#572・#799・#803・#804・#806 の確立済み
先例と同方式）。実測値の記入は CUDA 実機（DGX Spark GB10）到達可能なセッションへ申し送る。

## 12. 動作確認（本実装セッションで実施済み）

- `cargo build --example cuda_floor_bench -p backend-cuda`（cudarc 動的ロードにより CUDA toolkit
  非搭載環境でもビルド成立することを確認。実行結果は下記参照）
- 経路カバレッジ再確認（§2）: `grep` によるコード上の確認のみ。GPU 実行は伴わない
- `cargo fmt --all -- --check`・`cargo clippy --workspace --all-targets --all-features -- -D
  warnings`・`cargo test --workspace`（実機依存は `#[ignore]` 分離済みのため CI 可）は本 PR のコミット
  前チェックとして実行する（コード変更なし・ドキュメントのみのためビルド影響はない前提だが、
  workspace 全体の green を確認する）

## 13. 引き継ぎ事項（次に実機到達できたセッションへ）

- §7〜§10 の各表を §4〜§6 の手順どおり実測して埋める
- 実測後、判定対象形状の対 PyTorch 比最小値を §9 の「直近の確定値」と比較し、Phase 3/4 ツリー
  （#804/#806 の机上検討）の実施が実測値そのものへ影響していない（本番未結線のため）ことを再確認する
- `docs/perf/oss-gemm-comparison-baseline.md` §7.2・`docs/perf/gemm-optimization-baseline.md` §1 補足
  （本 PR で追記済み。§14 参照）へ実測後の日付・commit・比率を追記する
- 実測比率が候補下限値帯へ影響する場合は `docs/perf/performance-floor-decision.md` へ「下限値変更なし・
  変更提案は承認事項として分離」の整理追補を追加する（#799 §11 先例と同型）。**下限値そのものは変更
  しない**
- #804/#806 のタイル拡大候補の実機選定・本番結線は本イシューのスコープ外（両ドキュメントの「引き継ぎ
  事項」節が正）

## 14. 相互参照

- 前回ベースライン: `docs/perf/cuda-optimized-remeasurement.md`（#571・Phase F-1）
- parity 正本: `docs/perf/cuda-parity-baseline.md`
- REQ-8 突合基準: `docs/perf/gemm-optimization-baseline.md` §1
- OSS 比較キャンペーン表: `docs/perf/oss-gemm-comparison-baseline.md` §7.2
- #804/#806 の未結線状態の詳細: `docs/perf/cuda-gemm-mma-block-tile-stages.md`・`docs/perf/
  cuda-gemm-mma-tf32-block-tile.md`
- 実機接続手順: `docs/real-hardware-verification-env.md`（実ホスト名は `docs/
  real-hardware-verification-env.local.md`。Git 管理外）
