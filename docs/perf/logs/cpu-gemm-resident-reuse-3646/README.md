# GB10 CPU gemm 常駐入力経路の reuse>fresh 逆転の切り分け（#3646）

0.12.0 の 6 フレームワーク同席計測（#3273・`../framework-compare-0.12.0-remeasure/README.md`）で、
bench-fandhe の `gemm --mode reuse` を常駐入力経路（#3272。`Tape::upload_resident` → `matmul_resident`）へ
切り替えたところ、GB10 の CPU gemm だけ reuse が同 round の fresh より 1.23〜1.79 倍遅くなった。
本ディレクトリはその原因を GB10 実測で切り分けるための事前登録規則・診断基盤・集計器を置く。
結論・帰属・修正方針の提案は `docs/perf/cpu-gemm-reuse-fresh-inversion.md` §9 に書く。

## 状態

| 項目 | 状態 |
|---|---|
| 診断テスト（`crates/facade/tests/cpu_gemm_resident_reuse_diag.rs`）・RULE.txt・orchestrate.sh・aggregate.py | 整備済み（CI の bit 一致テスト・`aggregate.py --self-test`・shellcheck は通過） |
| GB10 正式系列（5 run）・補助系列（minflt／tunables／taskset）・ブリッジ（bench `--phases`） | **未実測**（本作業の実行環境に GB10 への接続がなく、Linux・M4 Max の代替値は使わない） |
| G-CPU-INF の 3 区分分類（要件 2） | **未実測のため「保留」**。規則は RULE.txt 末尾に事前登録済み。足りない数値は GB10 同一セッションの PyTorch CPU 純カーネル・fandhe 層別区間・PyTorch CPU BLAS 経路の実行時確認 |
| 修正（要件 3） | 実装しない（所有者への方針提示が先。提案は §9 に記載） |

## 構成

- `RULE.txt`: 事前登録規則（腕・区間・代表値・checksum 規則・再現判定・帰属 rho・補助系列・ゲート・G-CPU-INF 分類）
- `orchestrate.sh <m4max|gb10> [--aux minflt|tunables|taskset] [--out <dir>]`: 1 回ビルドして独立 5 プロセスで実行
  （`--aux minflt` のみ 1 run）。`.rev-stamp` 必須・`RAYON_NUM_THREADS` 拒否・既存 run 上書き禁止・
  GB10 は load1 < 1.0 ゲート・ホスト名／`$HOME`／ユーザー名をマスク
- `aggregate.py <dir>`: 標準ライブラリのみ。欠落・n_samples 不一致・checksum bit 不一致・ゲート／系列種別欠落は fail-closed。
  判定は `series_kind=main` かつ全 run `gate=pass` の系列のみ

## 腕（窓は `iter_total`）

| 腕 | 内容 | 分離する仮説 |
|---|---|---|
| fresh | bench `run_gemm` と同じ列（Tape drop は窓の外） | 基準 |
| resident_public | facade `matmul_resident` → contiguous 借用 → checksum（bench 新腕の写し） | 再現対象 |
| resident_split | `gemm_device`／`download`／`out_drop`／readout／checksum の区間分解 | 区間の内訳 |
| gemm_device_hold | `gemm_device` のみ（drop は窓の外） | カーネル＋確保の基準 |
| gemm_device_drop_in | `gemm_device` ＋窓内 drop | H1b（munmap・TLB） |
| gemm_device_download | `gemm_device` ＋ `download`（drop は窓の外） | H1（`to_vec` の確保とコピー） |
| resident_reupload | 毎反復 窓の外で upload し直す | H2・H3（入力の寿命・配置） |
| backend_drop | `BackendOps::gemm` 直接（#3187 の対照） | カーネル基準 |

## 計測ツリーの等価性（実装時の確認）

0.12.0 の bench（registry `fandhe-ai =0.12.0`）と HEAD の診断テストが同じコードを測ることを、
`git diff --stat v0.12.0 HEAD -- crates/backend-cpu/src crates/autodiff/src/optim/resident_input.rs crates/autodiff/src/optim/device_store.rs crates/facade/src/lib.rs crates/tensor-core/src/alloc.rs`
が空であることで確かめる（実装時点の結果は §9 に記録）。差分が出た場合は計測ツリーを `v0.12.0` tag に固定する。

## 実行手順（GB10。`docs/real-hardware-verification-env.md` の手順に従う）

1. RULE.txt を含むコミットを計測ツリーとして rsync し、ルートに `.rev-stamp`（1 行目 HEAD・2 行目 `dirty=<件数>`）を置く。
   ツリー指紋を `git archive` の展開物と突合して `rev-stamp-verification-3646.md` に残す
2. `setsid nohup` で切り離し hw-lock を取得、PATH を明示、計測中はビルド並走禁止
3. `bash orchestrate.sh gb10` → `python3 aggregate.py gb10`
4. 補助: `--aux minflt`・`--aux tunables`・`--aux taskset`（別ディレクトリへ出力）
5. ブリッジ: bench-fandhe `--task gemm --device cpu --mode reuse|fresh --phases` を N=256〜4096 各 5 run
6. G-CPU-INF: RULE.txt 末尾の測定 (i)〜(iii)（既存の承認済み venv のみ。新規パッケージ導入はしない）
