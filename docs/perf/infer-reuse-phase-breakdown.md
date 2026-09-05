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
