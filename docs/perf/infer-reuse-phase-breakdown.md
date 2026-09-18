# 推論 `predict_resident` reuse モード・フェーズ分解の初回実測（イシュー #1217）

## 1. 目的・対応

イシュー #1217「`bench-fandhe` の infer に `predict_resident` reuse モードと
`--phases` 対応を追加する」に対応する実装・初回実測記録。`docs/perf/
train-step-phase-breakdown.md` §13・§15.5 が指摘したとおり、`--task infer`
はこれまで fresh 経路のみで `--task infer --phases` は `dispatch()` が
MEASURE_ERROR を返しており、infer の candle 比未達（DGX CUDA 0.27 倍・
M4 Max Metal 0.51 倍。`results/summary.md` 環境 10/11）の内訳（H2D/D2H・
forward の寄与）が実測できていなかった。

本イシューで `bench-fandhe --task infer --mode reuse`（`Sequential::
predict_resident` 経由。facade 公開 API・0.6.0 で公開済み）と `--task infer
--phases`（fresh/reuse 双方）を追加し、`summarize.py` に (c') reuse 節・
(c'') フェーズ分解節を追加した。計測ハーネスの区間定義・JSONL スキーマ・
`summarize.py` (c')/(c'') 節の読み方は `scripts/bench/framework-compare/
README.md`「`infer --mode reuse` / `infer --phases`」節を正とし、本
ドキュメントでは二重管理しない。

## 2. 判断事項（実装計画 D1〜D8 の要旨）

| # | 判断 |
|---|------|
| D1 | facade Phase 2（`predict_resident` 内部の `linear_forward_device` 差し替え）は本イシューのスコープ外。ハーネス拡張に閉じる |
| D2 | infer reuse の `init_s` は `train --mode reuse` と同一定義（`init_device_param_store` + `sync_device_param_store_to_host` の完了保証同期）。`predict_resident` は呼び出しごとに内部で tape を生成・破棄するため、gemm/train reuse と異なり warmup を init 側で消費する必要が無く、fresh と同じ `WARMUP_ITERS`/`MEASURE_ITERS`（20/20）をそのまま使う |
| D3 | 目標達成ゲート（`_pick_row_for_gate`）は infer でも reuse 優先を採用（gemm/train と同じ規則をそのまま適用。ロジック変更なし・reuse 行の存在で自然に優先される） |
| D4 | fresh `--phases` の計測窓は既存 `run_infer` と同一（GPU の `make_tape` は計測窓外・phase として emit しない）。既存 (c) 数値の前提を変えない |
| D5 | phase 集合は `(mode, device_class)` で定義（README 表参照）。CPU fresh の `predict`・reuse の `predict_resident` は公開 API 上単一呼び出しで分解不能。GPU fresh は `leaf_register`/`forward`/`to_tensor`/`host_copy`/`checksum`/`iter_total` に分解可能 |
| D6 | `results/summary.md` への反映は新環境節への追記＋「目標達成ゲート総括」への前方注記とし、既存環境節（infer reuse 未対応の記述）は履歴として書き換えない |
| D7 | `run_all.sh`/`run_all_cuda.sh` の bench-fandhe 限定ブロックへ (c') infer reuse・(c'') infer phases を追加（train と同じ扱い） |
| D8 | 本 PR に本体ライブラリの本番結線・性能に影響する変更はないため before/after 5 回比較は不要。代わりに本ドキュメントの初回実測を記録する |

## 3. 計測プロトコル

- モデル: 784→256→10（ReLU）・バッチ 64（`--size` は infer では無視され
  `BATCH=64` 固定。`bench-fandhe/src/main.rs` 参照）
- `warmup=20` / `iters=20`（producer 側の既定値。fresh/reuse 共通）
- **5 回計測**（`.claude/rules/coding-rust.md`）: `results/raw/
  results-m4max-infer-reuse-0.6.0-run{1..5}.jsonl` へ 1 run = 1 ファイルで
  記録（`_reuse_row_invalid_reason`/`_pick_row_for_gate` の重複キー検出が
  「1 ファイル = 1 run」を前提とするため。§5 参照）。本表の数値は各ランの
  `median_s` 5 個の中央値（`statistics.median`）と範囲（min–max）
- `--phases` は fresh/reuse 各 1 run を `results/raw/
  results-m4max-infer-reuse-0.6.0-phases.jsonl` に分離して記録
- 実行コマンド:

  ```bash
  ./target/release/bench-fandhe --task infer --device <cpu|metal> \
    --mode <fresh|reuse> [--phases] --out <dest.jsonl>
  ./target/release/bench-candle --task infer --device <cpu|metal> \
    --mode fresh --out <dest.jsonl>
  ```

- 計測環境: Apple M4 Max（MacBook Pro）・macOS 26.6.2 (25G83)・
  rustc/cargo 1.96.0（ローカル直接実行。デバイス cpu・metal）。**共有・
  多利用者環境**（計測時 `uptime`: `up 17 days, 17:02, 19 users, load
  averages: 2.55 3.17 4.10`）であり、他プロセスの負荷混入により run 4・
  run 5 の一部値が run 1〜3 より大きく劣化している（§4 の range 参照。
  `train-step-phase-breakdown.md` §10.4 と同じ注意）。DGX Spark GB10
  （CUDA）は本エージェント実行環境に実機アクセスが無いため未実測（§6）
- fandhe-ai は `fandhe-ai =0.6.0`（crates.io 公開版。deps-policy 第 9
  区分の承認ピン）に完全固定。CUDA/Metal の `predict_resident` 内部が
  `linear_forward_device`（#1216）未結線であることの影響は §6 を参照

## 4. M4 Max 実測（5 run 中央値・範囲）

| フレームワーク | デバイス | mode | 中央値の中央値 (µs) | 範囲 (µs) |
| --- | --- | --- | --- | --- |
| candle 0.11.0 | cpu | fresh | 176.7 | 141.6–1133.5 |
| candle 0.11.0 | metal | fresh | 402.3 | 303.2–957.7 |
| fandhe-ai 0.6.0 | cpu | fresh | 507.9 | 482.9–1886.1 |
| fandhe-ai 0.6.0 | cpu | reuse | 345.6 | 312.1–1821.7 |
| fandhe-ai 0.6.0 | metal | fresh | 756.4 | 664.4–7573.7 |
| fandhe-ai 0.6.0 | metal | reuse | 723.2 | 623.3–3443.2 |

- fresh→reuse 改善（中央値の中央値ベース）: cpu 1.47 倍（507.9→345.6 µs）、
  metal 1.05 倍（756.4→723.2 µs）。cpu は「ホスト経由の重み再構築を伴わない
  常駐パラメータ forward」の効果が明確。metal は §6 の理由（`predict_resident`
  内部が `gemm_resident_rhs_act` を使い、`linear_forward_device`〈#1216〉
  未結線のため中間活性化を層ごとにホスト実体化する構造）により改善が
  小さい
- candle 比（reuse・中央値の中央値）: cpu 0.51 倍（176.7/345.6）、
  metal 0.56 倍（402.3/723.2）— fresh 比（cpu 0.35 倍・metal 0.53 倍）から
  改善するが未達のまま。§6 参照
- `--strict`（無効データ判定）は 5 run とも exit 0（checksum 突合・
  時間値・`init_s` はすべて有効）

## 5. `infer --phases` 実測（1 run。fresh/reuse 各区間の中央値）

`results/raw/results-m4max-infer-reuse-0.6.0-phases.jsonl`（`--strict` exit 0）。

### CPU

| mode | phase | 中央値 | iter_total 比 |
| --- | --- | --- | --- |
| fresh | predict | 522.4 µs | 99.8% |
| fresh | host_copy | 0.25 µs | 0.05% |
| fresh | checksum | 0.33 µs | 0.06% |
| fresh | iter_total | 523.2 µs | 100.0% |
| reuse | predict_resident | 389.8 µs | 99.8% |
| reuse | host_copy | 0.31 µs | 0.08% |
| reuse | checksum | 0.33 µs | 0.09% |
| reuse | iter_total | 390.7 µs | 100.0% |

init_s（reuse・`DeviceParamStore` 構築）: 162.8 µs

### Metal

| mode | phase | 中央値 | iter_total 比 |
| --- | --- | --- | --- |
| fresh | leaf_register | 0.13 µs | 0.02% |
| fresh | forward | 605.0 µs | 81.0% |
| fresh | to_tensor | 140.6 µs | 18.8% |
| fresh | host_copy | 0.25 µs | 0.03% |
| fresh | checksum | 0.42 µs | 0.06% |
| fresh | iter_total | 747.4 µs | 100.0% |
| reuse | predict_resident | 718.9 µs | 99.9% |
| reuse | host_copy | 0.21 µs | 0.03% |
| reuse | checksum | 0.42 µs | 0.06% |
| reuse | iter_total | 719.7 µs | 100.0% |

init_s（reuse・`DeviceParamStore` 構築。Metal デバイスハンドル初期化コスト
を含む）: 28.855 ms

- fresh・metal の内訳: `forward`（81.0%）が支配的、`to_tensor`（18.8%。
  `Var::to_tensor()` の実体化。0.6.0 の `Tensor<f32>` はホスト常駐なため
  Metal 側の readback を含む）が次点。`leaf_register`/`host_copy`/
  `checksum` は無視できる規模
- reuse・metal は `predict_resident` 単独区間（99.9%）で、内部の
  `gemm_resident_rhs_act` 呼び出し・層ごとのホスト実体化はこれ以上
  分解できない（README「分離不能な内訳」列参照）
- reuse・metal の `init_s`（28.9 ms）は fresh には存在しない
  `DeviceParamStore` 構築コストで、`train --mode reuse` の Metal 実測
  （`train-step-phase-breakdown.md`）と同オーダー

## 6. 分離不能な内訳・#1216 効果測定の前提

- `predict_resident` 内部（`forward_from_flat_leaves` → `store.
  linear_forward_with_activation` → `gemm_resident_rhs_act`）は private
  ヘルパーのため公開 API からこれ以上分解できない。同型の内訳が要る場合は
  `crates/facade` 内のリポジトリ内ベンチ（`crates/facade/tests/
  infer_fixed_cost_bench.rs` 系）で取る必要がある
- **#1216 の効果は本ドキュメントの実測には反映されていない**: #1216 で
  HEAD に追加された CUDA/Metal `BackendOps::linear_forward_device`
  （中間活性化をデバイス常駐のまま連鎖させる非破壊拡張）は facade
  （`predict_resident`/`forward_resident`）へ未結線であり（`docs/
  linear-forward-device-gpu.md` §5「Phase 2 未実施」）、かつ本ハーネスは
  crates.io 公開版 `fandhe-ai =0.6.0` に固定されているため、#1216 の
  内部実装がどちらでも本ドキュメントの数値は変わらない。効果を測定する
  には次がすべて必要（いずれも本 PR のスコープ外）:
  1. facade Phase 2（`predict_resident`/`forward_resident` 内部を
     `linear_forward_device` へ差し替える結線）
  2. 新版 crates.io 公開
  3. `scripts/bench/framework-compare` の承認ピン更新（ユーザー承認必須。
     deps-policy.md 第 9 区分）
  4. 本ドキュメントと同一プロトコルでの再計測

## 7. DGX Spark GB10（CUDA）実測欄

本エージェント実行環境に CUDA 実機アクセスが無いため未実測。再現手順は
§3 の実行コマンドをそのまま使う（`--device cuda`）。実測後は本節を置き換え、
§4/§5 の表に CUDA 列・区間（`fresh` は `leaf_register`/`forward`/
`to_tensor`/`host_copy`/`checksum`/`iter_total` の 6 区間。§3.2 の GPU 区分
は metal/cuda 共通）を追記する。

## 8. 検証結果サマリ

- `cargo test --release -p bench-fandhe`: 19 passed（infer reuse/phases
  関連 9 件を含む）・6 ignored（実機 smoke。うち Metal 3 件は本エージェント
  実行環境〈M4 Max〉で `--ignored` 指定により実行し全 pass 確認済み。CUDA
  smoke は実機なしのため未実行）
- `python3 -m unittest summarize_test.py compare_ab_test.py
  compare_gemm_gate_test.py`: 229 passed（infer reuse/phases 関連 16 件を
  含む）
- `cargo fmt --all -- --check`・`cargo clippy --release -p bench-fandhe -p
  bench-common --all-targets -- -D warnings`: いずれも警告なし
- `bash scripts/check-forbidden-deps.sh lock-all`（本体リポジトリルート
  から実行）: `check_framework_compare` の契約検査 pass（ピン・members
  構成に変更なし）
- 上記 §4/§5 の JSONL は `--strict` exit 0（無効データなし）。
  `--target candle` は infer が未達のため exit 3（期待どおり。データ不正
  ではない）

## 9. 後続（イシュー #1689）

facade Phase 2（#1216 の `linear_forward_device` 結線・上記本文が
「metal 改善は限定的」と明記していた原因）は #1688 で実装済み
（`DeviceParamStore::predict_device_chain`・`Sequential::predict_
resident` の chain 経路優先化。`docs/inference-chain-single-sync-
design.md` §9）。その効果を CUDA 実機で計測する A/B・bit 同一確認は
#1689（`docs/perf/infer-chain-single-sync-cuda-ab.md`。同ドキュメント
の `--task infer` A/B スクリプト `run_ab_infer_chain_cuda.sh` は
`scripts/bench/framework-compare/run_ab_resident_grad_cuda.sh`〈#1560〉
と同じ「before／after 2 ツリーを path patch でビルドして比較する」
方式へ、本ファイル §3 が定義する `infer --mode reuse`／`infer --phases`
の計測プロトコルを拡張したもの）が担当する。本エージェント実行環境に
GB10 実機がないため未実測のまま記入欄を残す。

## 10. v0.9.0 ピン再計測（Apple M4 Max・DGX Spark GB10・2026-09-18。イシュー #1981）

### 10.1 目的・条件

registry `fandhe-ai =0.9.0`（`scripts/bench/framework-compare/` の
承認済みピン）にビルドした `bench-fandhe --task infer --phases` を
Apple M4 Max で cpu／metal × fresh／reuse の 4 セル・5 プロセス独立
起動で再計測した。事前登録規則・生ログ・env_info は
`docs/perf/logs/train-infer-phases-0.9.0-1980-1981/`（README 参照。
train 分と同一の run・同一 JSONL を共有する）。**採否判定を伴わない
記録**（tolerance・判定規則・本番定数は変更しない）。DGX Spark GB10
側は 2026-09-18 に別セッションで実測済み（§10.6。専有ゲート 5/5
通過）で、両実機が揃った。§10.2〜§10.4 は Mac 分（数値・文言は GB10
実測後も変更していない。§10.5 は GB10 項のみ更新）。v0.9.0 には #1688（`DeviceParamStore::
predict_device_chain` の実装）・#1580／#1911（同機構の facade
`predict_resident` への結線・Metal 実機 ADOPT 確定。`docs/perf/
metal-infer-chain-single-sync.md`）が含まれており、§4/§5 の v0.6.0
実測より metal reuse が改善している可能性がある点に留意する
（本節では原因帰属の判定は行わない）。

### 10.2 内訳表（5 run 中央値・min–max・iter_total 比）

`docs/perf/logs/train-infer-phases-0.9.0-1980-1981/m4max/aggregate.md`
からの転記。単位は µs。

#### infer_phases / cpu / fresh

| phase | 中央値 (µs) | min–max (µs) | 合計比 |
|---|---:|---|---:|
| predict | 184.9 | 158.1–220.2 | 99.7% |
| host_copy | 0.0 | 0.0–0.1 | 0.0% |
| checksum | 0.4 | 0.3–0.4 | 0.2% |
| iter_total | 185.5 | 158.6–220.9 | 100.0% |

トップ 3: predict（99.7%・184.9 µs）・checksum（0.2%・0.4 µs）・
host_copy（0.0%・0.0 µs）。フェーズ和／合計: 99.9%（差分は各フェーズ中央値の
非加法性を含み、固定費は未測定）。

#### infer_phases / cpu / reuse

| phase | 中央値 (µs) | min–max (µs) | 合計比 |
|---|---:|---|---:|
| predict_resident | 175.3 | 163.9–199.5 | 99.6% |
| host_copy | 0.0 | 0.0–0.0 | 0.0% |
| checksum | 0.3 | 0.3–0.3 | 0.2% |
| iter_total | 176.0 | 164.5–200.1 | 100.0% |

トップ 3: predict_resident（99.6%・175.3 µs）・checksum（0.2%・
0.3 µs）・host_copy（0.0%・0.0 µs）。フェーズ和／合計: 99.8%
（差分は各フェーズ中央値の非加法性を含み、固定費は未測定）。

#### infer_phases / metal / fresh

| phase | 中央値 (µs) | min–max (µs) | 合計比 |
|---|---:|---|---:|
| leaf_register | 0.1 | 0.1–0.1 | 0.0% |
| forward | 431.6 | 311.9–507.6 | 80.2% |
| to_tensor | 110.2 | 88.9–115.4 | 20.5% |
| host_copy | 0.2 | 0.2–0.2 | 0.0% |
| checksum | 0.4 | 0.3–0.4 | 0.1% |
| iter_total | 538.2 | 398.5–634.0 | 100.0% |

トップ 3: forward（80.2%・431.6 µs）・to_tensor（20.5%・110.2 µs）・
checksum（0.1%・0.4 µs）。フェーズ和／合計: 100.8%（差分は各フェーズ中央値の非加法性を含み、
固定費は未測定。個別フェーズの 5 run 中央値の和が合計フェーズの中央値を
上回る統計上の事象で、aggregate.md の値をそのまま転記する）。

#### infer_phases / metal / reuse

| phase | 中央値 (µs) | min–max (µs) | 合計比 |
|---|---:|---|---:|
| predict_resident | 365.2 | 353.6–369.3 | 99.8% |
| host_copy | 0.2 | 0.2–0.2 | 0.1% |
| checksum | 0.4 | 0.4–0.4 | 0.1% |
| iter_total | 366.1 | 354.4–370.0 | 100.0% |

トップ 3: predict_resident（99.8%・365.2 µs）・checksum（0.1%・
0.4 µs）・host_copy（0.1%・0.2 µs）。フェーズ和／合計: 99.9%
（差分は各フェーズ中央値の非加法性を含み、固定費は未測定）。

### 10.3 所見

- cpu は fresh／reuse とも `predict`／`predict_resident` がほぼ
  iter_total を占め（99.6〜99.7%）、他区間は無視できる規模
- metal fresh は `forward`（80.2%）が支配的で `to_tensor`（20.5%。
  `Var::to_tensor()` の実体化・Metal 側 readback を含む）が次点。
  v0.6.0 実測（§5「forward 81.0%・to_tensor 18.8%」）とほぼ同じ比率
- **metal reuse（365.2 µs）は cpu reuse（175.3 µs）の約 2.1 倍**。
  `predict_resident` 単独区間（99.8%）でこれ以上分解できない
  （§6「分離不能な内訳」）。size=64 という小形状では GPU 起動固定費
  が相対的に支配的になっていると推定される（推定であり本計測では
  検証していない）
- metal reuse（365.2 µs）は v0.6.0 実測（§5「718.9 µs」）よりおよそ
  半分に改善している。v0.9.0 には #1688（実装）・#1580／#1911（Metal
  推論単一同期化。`predict_resident` chain 経路。実機 ADOPT 確定）が
  含まれるため、この改善は同機構に起因する可能性があるが、本節では
  before/after A/B を取っていないため原因帰属の確定判定はしない

### 10.4 施策の起票案（列挙のみ。起票はしていない。新規 issue 化は
ユーザー承認が必要）

- metal reuse の GPU 起動固定費が支配的という推定を検証する
  マイクロベンチ（`predict_resident` 内部の同期回数・encode 回数の
  診断カウンタ。`docs/perf/metal-infer-chain-single-sync.md` が近縁の
  診断手法を持つ）
- metal fresh の `to_tensor`（110.2 µs・20.5%）削減余地の確認
  （`docs/perf/logs/metal-readout-legacy-regression-four-arm-diag.md`
  が readback 経路の診断を扱う既存記録）
- backward 側と同様、infer でも診断計装パッチによる内部内訳取得の
  要否確認（本節の限界と同じ制約）
- cpu／metal 双方で size を変えたスイープ（size=64 固定の本計測では
  小形状固定費と GEMM 本体コストの分離ができない）

### 10.5 限界

- registry `fandhe-ai =0.9.0`（cargo を起動しない事前ビルド済み
  バイナリ）を使うため、`predict_resident`／`forward` 内部の
  GEMM／非 GEMM 内訳・診断計装は取得不能。本計測では未取得
- DGX Spark GB10 側は §10.6 で実測済み（2026-09-18）。両実機の
  フェーズ表が揃い、#1981 の受け入れ条件「両実機 5 run 中央値の
  フェーズ表」は本節で充足する

### 10.6 DGX Spark GB10 分（2026-09-18。専有ゲート付き系列）

#### 10.6.1 条件

train 分（`docs/perf/train-step-phase-breakdown.md` §17.6.1）と同一
の run・同一 JSONL を共有する。要点: NVIDIA GB10（driver 580.173.02・
sm_121・CUDA 13.0・Ubuntu 24.04.4 aarch64・rustc 1.97.0）・registry
`fandhe-ai =0.9.0` をノード上で再ビルドした事前ビルド済みバイナリ
（path patch なし・計測中 cargo 非起動・転送元コミット `536c56a8`）・
cpu／cuda × fresh／reuse の 4 セル・5 プロセス独立起動・負荷ゲート
「load1 < 1.0 かつ GPU utilization 0%」5/5 通過（常駐 2 プロセスは
停止せず記録のみ）・`run{1..5}.err` 全 5 本 0 バイト・事前登録規則
`gb10/RULE.txt`（2026-09-18T01:33:53Z 固定）。`gb10/aggregate.md` は
`aggregate.py --devices cpu,cuda --machine "DGX Spark GB10"` の再生成
で byte 同一を確認済み。**採否判定を伴わない記録**。§7 の記入欄は
v0.6.0 当時の未実測記録としてそのまま残す（本節は v0.9.0 の実測）。

#### 10.6.2 内訳表（5 run 中央値・min–max・iter_total 比）

`docs/perf/logs/train-infer-phases-0.9.0-1980-1981/gb10/aggregate.md`
からの転記。単位は µs。

##### infer_phases / cpu / fresh（GB10）

| phase | 中央値 (µs) | min–max (µs) | 合計比 |
|---|---:|---|---:|
| predict | 176.3 | 167.1–184.6 | 99.5% |
| host_copy | 0.1 | 0.1–0.2 | 0.1% |
| checksum | 0.3 | 0.3–0.5 | 0.2% |
| iter_total | 177.1 | 167.8–186.1 | 100.0% |

トップ 3: predict（99.5%・176.3 µs）・checksum（0.2%・0.3 µs）・
host_copy（0.1%・0.1 µs）。フェーズ和／合計: 99.8%（差分は各フェーズ
中央値の非加法性を含み、固定費は未測定）。

##### infer_phases / cpu / reuse（GB10）

| phase | 中央値 (µs) | min–max (µs) | 合計比 |
|---|---:|---|---:|
| predict_resident | 195.0 | 167.6–195.6 | 99.5% |
| host_copy | 0.2 | 0.1–0.2 | 0.1% |
| checksum | 0.3 | 0.3–0.3 | 0.2% |
| iter_total | 196.0 | 168.2–196.2 | 100.0% |

トップ 3: predict_resident（99.5%・195.0 µs）・checksum（0.2%・
0.3 µs）・host_copy（0.1%・0.2 µs）。フェーズ和／合計: 99.7%
（差分は各フェーズ中央値の非加法性を含み、固定費は未測定）。

##### infer_phases / cuda / fresh（GB10）

| phase | 中央値 (µs) | min–max (µs) | 合計比 |
|---|---:|---|---:|
| leaf_register | 0.2 | 0.2–0.3 | 0.2% |
| forward | 138.1 | 133.0–139.4 | 87.2% |
| to_tensor | 19.0 | 18.7–19.3 | 12.0% |
| host_copy | 0.1 | 0.1–0.1 | 0.1% |
| checksum | 0.6 | 0.6–0.8 | 0.4% |
| iter_total | 158.4 | 153.0–160.0 | 100.0% |

トップ 3: forward（87.2%・138.1 µs）・to_tensor（12.0%・19.0 µs）・
checksum（0.4%・0.6 µs）。フェーズ和／合計: 99.9%（差分は各フェーズ
中央値の非加法性を含み、固定費は未測定）。

##### infer_phases / cuda / reuse（GB10）

| phase | 中央値 (µs) | min–max (µs) | 合計比 |
|---|---:|---|---:|
| predict_resident | 98.8 | 97.9–99.1 | 98.9% |
| host_copy | 0.1 | 0.1–0.1 | 0.1% |
| checksum | 0.8 | 0.8–0.9 | 0.8% |
| iter_total | 100.0 | 99.0–100.1 | 100.0% |

トップ 3: predict_resident（98.9%・98.8 µs）・checksum（0.8%・
0.8 µs）・host_copy（0.1%・0.1 µs）。フェーズ和／合計: 99.8%
（差分は各フェーズ中央値の非加法性を含み、固定費は未測定）。

#### 10.6.3 所見（数値から言えることのみ）

- cpu は fresh／reuse とも `predict`／`predict_resident` が iter_total
  の 99.5% を占め、他区間は無視できる規模（Mac 分 §10.3 と同じ構成）
- **cpu reuse（196.0 µs）は cpu fresh（177.1 µs）より遅い**
  （+18.9 µs。reuse の min は 168.2 µs で fresh の min〈167.8 µs〉
  付近まで下がる run がある）。Mac 分（reuse 176.0 対 fresh 185.5 で reuse
  が速い）とは方向が逆
- cuda fresh は `forward`（87.2%・138.1 µs）が支配的で `to_tensor`
  （12.0%・19.0 µs。`Var::to_tensor()` の実体化・D2H readback を含む）
  が次点。Mac metal fresh（forward 80.2%・to_tensor 20.5%）より
  `to_tensor` の比重が小さい
- **cuda reuse（100.0 µs）は cpu reuse（196.0 µs）の約 0.51 倍で
  GPU 側が速い**。Mac 分（metal reuse が cpu reuse の約 2.1 倍）とは
  逆の関係で、`predict_resident` 単独区間（98.9%）でこれ以上分解
  できない点は同じ（§6「分離不能な内訳」）。size=64 という小形状で
  GPU 起動固定費がどの程度を占めるかは本計測では検証していない
- cuda fresh→reuse は 158.4→100.0 µs（−58.4 µs）で、fresh の
  `forward`＋`to_tensor`（157.1 µs）が reuse の `predict_resident`
  （98.8 µs）へ置き換わる差にほぼ一致する
- v0.9.0 には #1688（`DeviceParamStore::predict_device_chain`）・
  #1689／#1905（同機構の CUDA 実機 A/B。`docs/perf/
  infer-chain-single-sync-cuda-ab.md`。ADOPT）が含まれるが、本節は
  before/after A/B を取っていないため原因帰属の判定はしない。#1981
  の出典スコアボード（リポジトリ外 Artifact。最速相手比）は本節では
  参照・検証していない

#### 10.6.4 施策の起票案（列挙のみ。起票はしていない。新規 issue 化は
ユーザー承認が必要）

- infer でも診断計装パッチによる `predict`／`predict_resident`／
  `forward` 内部の内訳取得の要否確認（§10.4 第 3 項と重複。GB10
  でも同様に未取得）
- cpu／cuda 双方で size を変えたスイープ（§10.4 第 4 項と重複。
  size=64 固定では小形状固定費と GEMM 本体コストの分離ができない）
- cpu reuse が fresh より遅い点（+18.9 µs・Mac とは逆方向）の原因
  切り分け（新規。`Sequential::predict_resident` の CPU 経路
  〈`docs/perf/cpu-infer-predict-profile.md`（#1218）が fresh 側の
  近縁記録〉と `predict` の差分。run 間の min–max 幅〈167.6〜
  195.6 µs〉が大きいため負荷変動の寄与も未分離）
- cuda fresh の `to_tensor`（19.0 µs・12.0%）削減余地の確認（§10.4
  第 2 項の cuda 版。`docs/perf/cuda-host-view-staging-readout.md`
  〈#1336／#1478〉が D2H readback 経路の既存記録）
- metal 固有の GPU 起動固定費検証（§10.4 第 1 項）は GB10 では
  cuda reuse が cpu reuse より速いため同じ形では該当しない

#### 10.6.5 限界

- `predict_resident`／`forward` 内部の内訳は Mac 分と同じく未取得
  （registry ビルド）
- 常駐 2 プロセス（`compute_apps=2`）を停止していないため完全な
  専有ではない（gpu_util は各 run 開始時 0%。RULE.txt は存在の記録
  のみと事前宣言）
- §7（v0.6.0 当時の GB10 記入欄）は未実測のまま残す。本節は v0.9.0
  の実測であり §4／§5 の v0.6.0 Mac 実測と同一条件の比較ではない
