# CPU backward 非 GEMM 内訳・iteration 集計の診断計装（イシュー #2100）

`docs/perf/lowlayer-diagnosis-2026-09-12.md` §4（backward の非 GEMM が 18〜34%。097bff19 時点・M4 Max のみ）
の診断計装を、旧パッチ（`docs/perf/logs/lowlayer-diagnosis-2026-09-12/diag-instrumentation.patch`）が
HEAD に当たらなくなったため**手作業で移植**し、train の各 step（iteration）ごとの backward 内訳を
CSV で出せるようにしたもの。`docs/perf/train-step-phase-breakdown.md` §17.4／§17.5／§17.6.4 の
「registry 0.9.0 ビルドでは GEMM／非 GEMM 内訳が取れない」を解消する前提診断で、施策 1（#1980・#1981 系）の
入力になる。**本 PR は計装・スクリプト・集計器・記録の整備まで**で、両実機（M4 Max・GB10）の実測は
Phase B として申し送る（本ランの環境は x86 Linux・RTX 3060 のみ）。

## 成果物の形（本番コード無変更）

- 本ディレクトリだけを追加する。`crates/**`・`Cargo.toml`／`Cargo.lock`・
  `scripts/bench/framework-compare/{Cargo.toml,Cargo.lock}`・`.github/workflows/**`・`docs/spec/**`・
  tolerance・baseline・ガードレール閾値は**変更しない**（受入基準 4 点）。
- 計装は `diag-instrumentation-head.patch`（crates/ 配下のみ・`Cargo.lock` の hunk なし）としてだけ保持し、
  `prepare_tree.sh` が作る**使い捨て worktree にだけ当てる**（スコープ外: 本番コードへの計装の残存）。
  旧パッチは `scripts/bench/framework-compare/Cargo.lock` の `source`／`checksum` を消していたが、
  これは `scripts/check-forbidden-deps.sh lock-all`（`check_framework_compare`。registry source 必須。
  deps-policy 第 9 区分）に反するため本パッチには含めない。path patch は `--config
  'patch.crates-io.fandhe-ai.path="<abs>/crates/facade"'` として CLI だけで与え、
  `bench_fandhe_lock_restore.sh` で Cargo.lock を退避・EXIT trap で復元する（`run_ab_1578.sh` と同方式）。
- **ディレクトリ名**: イシュー受入基準は `backward-nongemm-1/<device>/` と書き、同イシューの契約節は
  `<slug>-<issue 番号>` と書いていて食い違う。リポジトリ慣例（`cpu-mse-backward-threshold-1578` 等）の
  `<slug>-<issue>` を採り `backward-nongemm-2100/{m4max,gb10}/` とした。

## パッチの内容（`diag-instrumentation-head.patch`。HEAD `a24e1c32` で `git apply --check` 確認済み）

| ファイル | 内容 |
|---|---|
| `crates/autodiff/src/diag.rs`（新規） | `thread_local!` カウンタ・`OnceLock` の有効判定（`FANDHE_DIAG_BACKWARD` の有無のみ。値は解釈しない。`unsafe`・`set_var` なし）・`time`／`time_vjp`／`maybe_start`／`record`／`record_fill`／`report_backward`。DIAG 行に単調増加の `seq=` を付ける |
| `crates/autodiff/src/backward.rs` | `backward_impl` を「計時ラッパー + `backward_impl_inner`（挙動不変）」に分割。ノード値の実体化（Materialize）と勾配 accumulate（**新カテゴリ Accumulate**）を計時 |
| `crates/autodiff/src/grad.rs` | `vjp` を「`time_vjp` ラッパー + `vjp_inner`」に分割。MatMul／Add／Mul／Relu／Exp／Tanh／Sigmoid／MseLoss／CrossEntropyLoss／LinearResident／LinearAct を計時。`materialize_fallible` は `diag_materialize` 経由 |
| `crates/autodiff/src/lib.rs` | `mod diag;` |
| 診断テスト 4 本（すべて `#[ignore]`・参考用） | `grad.rs::diag_elementwise_mask_bench`・`optim/device_store.rs::diag_two_layer_resident_relu_backward_upstream_contiguity`・`backend-cpu/src/mse.rs::diag_backward_bench`・`backend-cpu/tests/mse_backward_fixed_cost_diag.rs` |

`crates/backend-cuda`・`crates/backend-metal` の**ソース変更は不要**。計時は autodiff 側で `ops.*` 呼び出しを
包むため、GPU バックエンドもそのまま対象に入る（壁時計計時のため、非同期 GPU 処理は同期点のカテゴリへ計上される）。

### DIAG 行と CSV 列

```
DIAG_BACKWARD seq=.. total_ns=.. vjp_ns=.. gemm_ns=.. fill_ns=.. mask_ns=.. ewise_ns=.. transpose_ns=..
              materialize_ns=.. accumulate_ns=.. loss_ns=.. vjp_calls=.. gemm_calls=.. mask_calls=..
```

| カテゴリ | 対象 |
|---|---|
| `gemm` | `matmul_vjp`・`gemm_resident_lhs`・`gemm_fp32_strict`・resident grad staging 書き込み（`fill_resident_weight_grad`）。`matmul_vjp` 内部の `transpose2d` は Gemm に内包 |
| `fill`（`gemm` の内訳） | `fill_resident_weight_grad`（#1563 で `gemm_resident_lhs` より前に移動。#1566 で bias 縮約を含むようになったため内訳を別掲）。旧診断 §4 と比較できるよう `gemm_ns` にも計上する（二重計上ではなく部分集合） |
| `mask` | `elementwise_mul_mask`・`tanh_grad_factor`・`sigmoid_grad_factor` |
| `ewise` | `vjp_elementwise_mul`・`reduce_to_shape`・`reduce_bias_grad`（Add／Mul／bias の VJP） |
| `transpose` | `Op::LinearResident` の gemm 外側の `transpose2d` |
| `materialize` | `materialize_fallible`（各 arm・ノード自身の forward 値） |
| `accumulate` | `backward.rs::accumulate`（fan-out 勾配合算。旧パッチは未計装で `total − vjp` に埋もれていた） |
| `loss` | `ops.mse_loss_backward`／`mse_loss_vjp`・`cross_entropy_loss_vjp` |
| `vjp` | `vjp()` 全体（参考。カテゴリ和との突合に使う） |

CSV 列（`iterations.csv`）: `machine,device,mode,run,step,is_warmup,seq,total_ns,vjp_ns,gemm_ns,fill_ns,mask_ns,
ewise_ns,transpose_ns,materialize_ns,accumulate_ns,loss_ns,vjp_calls,gemm_calls,mask_calls,nongemm_ns`
（`nongemm_ns = total_ns − gemm_ns`。`gemm_ns` は bias 縮約を含む `fill` を含むため非 GEMM の**下界**で、fill 全体を非 GEMM 側へ戻した上界 `nongemm_upper_ns = total_ns − gemm_ns + fill_ns` を末尾列に併記する。GEMM 計算と bias 縮約は分離計時していない）。`train --phases` は 1 プロセス 100 step（うち先頭 20 が warmup。
`bench-fandhe/src/main.rs` の `TRAIN_STEPS`／`TRAIN_WARMUP`）なので DIAG 行はちょうど 100 行になり、
集計器は件数・`seq` の単調増加を fail-closed で検査する。`aggregate.md` は各 run で step 20..99 の中央値 →
5 run の中央値（µs・非 GEMM %・カテゴリ和との残差）。

## 事前登録規則（`RULE.txt`。実測開始前に固定・事後に緩めない）

`orchestrate_backward_diag.sh` が実測開始前に UTC 時刻付きで各出力ディレクトリへ書く。要点:

- 判定種別は **`record_only`**（診断のみ。ADOPT／REJECT は判定しない）
- 唯一の fail-closed 検査: 計装あり／計装なし（同一 ref・どちらも path patch）の JSONL `checksum` が全セル・全 run で
  完全一致（計装が数値を変えないことの証明。不一致なら系列を無効とし集計器が exit 1）
- 計装オーバーヘッド比（`step_total` 計装あり÷なし）は記録するだけで判定しない
- 5 run はそれぞれ独立プロセス（差し替え・追加起動なし。run ごとに計装あり／なしの起動順を反転）
- 負荷ゲート: M4 Max = load1 < 8.0／GB10 = load1 < 1.0 かつ GPU util 0%。30 秒間隔・最大 30 分待つ。
  1 run でも不通過ならその系列は参考扱い（`gate.tsv`・`aggregate.md` に明記）
- セル: 必須 = cpu × {fresh, reuse}。参考 = metal（M4 Max）・cuda（GB10）× {fresh, reuse}（`DIAG_DEVICES` で指定。record_only）
- 既存 REJECT と重複する実験は再実行しない: `diag_backward_bench`（MSE の direct 対 naive）は #1578（REJECT・
  既定 `MSE_BACKWARD_PARALLEL_MIN_ELEMS=0`）の機構計測と、`diag_elementwise_mask_bench`・
  `diag_two_layer_resident_relu_backward_upstream_contiguity`（マスク contiguity）は #1667（マージ済み）と重複するため、
  本イシューの実測プロトコルから外し**参考用（`#[ignore]`・判定に使わない）**とする。プロトコルは
  `FANDHE_DIAG_BACKWARD=1 bench-fandhe --task train --phases` のセルだけで構成する
- ホスト名・絶対パスは `<host>`／`<home>` へマスクして収録する

## 手順

```bash
# 0. スクラッチ worktree（計装あり／なし。同じ ref）。実測対象の ref をそろえる
bash docs/perf/logs/backward-nongemm-2100/prepare_tree.sh instr origin/main /abs/path/wt-instr
bash docs/perf/logs/backward-nongemm-2100/prepare_tree.sh plain origin/main /abs/path/wt-plain

# 1. 5 run 収集 + 集計（machine = m4max | gb10 | smoke-x86）。出力先は空のディレクトリ
#    DIAG_DEVICES: M4 Max は "cpu metal"、GB10 は "cpu cuda"（cpu が必須セル）
DIAG_INSTR_FACADE_PATH=/abs/path/wt-instr/crates/facade \
DIAG_PLAIN_FACADE_PATH=/abs/path/wt-plain/crates/facade \
DIAG_OUT_DIR=$PWD/docs/perf/logs/backward-nongemm-2100/m4max \
DIAG_DEVICES="cpu metal" \
  bash docs/perf/logs/backward-nongemm-2100/orchestrate_backward_diag.sh m4max

# 2. 後片付け
git worktree remove --force /abs/path/wt-instr && git worktree remove --force /abs/path/wt-plain
```

GB10 へは `docs/real-hardware-verification-env.md` §3 の手順（rsync）で転送する。集計器の自己検査:
`python3 docs/perf/logs/backward-nongemm-2100/aggregate_backward_diag.py --self-test`。
（`ci.yml` は変更しない。required context・ruleset の更新と security-auditor 監査を誘発するため。）

## 本 PR の検証（x86 Linux・RTX 3060・非公式）

- パッチは clean な HEAD（`a24e1c32`）に `git apply --check` で当たる
- パッチ適用ツリーで `cargo fmt --all --check`・`cargo clippy -p fandhe-ai-autodiff -p fandhe-ai-backend-cpu
  --all-targets --locked -- -D warnings`・`cargo test -p fandhe-ai-autodiff --lib --release`（1449 passed・3 ignored）・
  `cargo test -p fandhe-ai-backend-cpu --release`（診断テストは ignored）・
  `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --locked` が通る。`FANDHE_DIAG_BACKWARD` 未設定では DIAG 行は出ない
- `smoke-x86/`（元ログ〈`run*-instr.err`・計装あり／なし JSONL〉は収録しない。集計結果のみで、掲載値と checksum 一致は本ディレクトリ単独では再検証できない。詳細は `smoke-x86/aggregate.md` 末尾）: 上記オーケストレーターを `smoke-x86`・3 run・cpu × {fresh, reuse} で実行した出力
  （1 プロセス 100 行・checksum 一致・CSV 生成を確認）。**非公式・判定外・実機（M4 Max／GB10）ではない**。
  共有負荷下（load average 15〜23 の開発機。開始時 19.59／終了時 22.37）の値であり、負荷ゲートは適用していない
  （`gate.tsv` の load1 は全行同一値の参考記録で判定に使わない）。絶対値を実機の結論として扱わない

## 受入基準の状況（Phase B へ申し送り）

| 受入基準 | 状況 |
|---|---|
| 計装パッチを HEAD へ移植しビルド成功 | 達成（x86 Linux で確認） |
| 両実機（M4 Max・GB10）でコンパイル・実行が成功 | **未達（実機待ち）**。M4 Max・GB10 で `orchestrate_backward_diag.sh` を実行する |
| backward 内訳の iteration 集計を CSV で出力 | 達成（`aggregate_backward_diag.py`・`smoke-x86/iterations.csv`） |
| tolerance・baseline・`Cargo.toml`・ガードレール閾値・`docs/spec/` を変更しない | 達成 |
| 実機依存テストは `#[ignore]` 分離・ログディレクトリへ `RULE.txt`・`env_info.txt`・生ログを収録 | 診断テストは `#[ignore]`。`m4max/`・`gb10/` の収録は**実機待ち** |

Phase B（実機）完了後: 結果を `docs/perf/lowlayer-diagnosis-2026-09-12.md` §4 の HEAD 版として
`train-step-phase-breakdown.md` §17.4／§17.6.4 から参照できる形で記録する。
