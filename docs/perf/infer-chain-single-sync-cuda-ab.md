# CUDA 推論 forward チェーン単一同期化の GB10 A/B・bit 同一確認（イシュー #1689）

## 0. 位置づけ（本 PR 時点では未実測）

本 PR の実行環境（Linux。`docs/real-hardware-verification-env.local.md`・
`CUDA_NODE` 環境変数のいずれも確認できず、ローカル GPU も driver/library
version mismatch により NVML／CUDA 初期化不可）には DGX Spark GB10 実機
への到達手段がないため、**本ドキュメントに実測値は一切含まれていない**。
本文書は事前登録判定規則・実行手順・記入欄のみを提供し、実測は GB10 実機
を持つセッションへ申し送る（`docs/perf/train-resident-grad-device-update.md`
§7・`docs/perf/logs/train-resident-grad-cuda-1560/`・`docs/backend-cuda-mse-
backward-stream-contract.md` と同型の運用）。

## 1. 目的

イシュー #1689「CUDA 推論 forward チェーンの単一同期化を GB10 実機で A/B
し bit 同一を確認する」を満たす。対象は #1579（設計。`docs/inference-
chain-single-sync-design.md`）・#1688（実装。PR #1788＝`87b1e338`）で
導入した `DeviceParamStore::predict_device_chain`・`BackendOps::
linear_forward_device_tracked`（default メソッド）・facade
`Sequential::predict_resident` の chain 経路優先化。

推論 forward の層境界ごとのホスト往復（`gemm_resident_rhs_act` の
per-layer upload／download）が支配的というボトルネック（`docs/perf/
lowlayer-diagnosis-2026-09-12.md` §7・`docs/perf/infer-reuse-phase-
breakdown.md`）を、単一同期チェーン（`upload` 1 回 → `linear_forward_
device_tracked` ×N → `download` 1 回）へ差し替えることで解消できるかを
GB10 実機で検証する。CPU 実測・Linux 実行可能なテストは #1688 で完了
済み（`docs/inference-chain-single-sync-design.md` §9）。

## 2. 比較対象 2 腕（事前登録・固定）

- **before 腕**: `edb85c43`（PR #1788〈#1688 実装〉マージ直前の main。
  `predict_resident` は `forward_from_flat_leaves` 経由＝層境界ごとに
  同期する従来経路）
- **after 腕**: 本イシューのブランチ（`crates/*/src` は評価対象コミット
  と同一。単一同期チェーン経由）

両腕とも workspace `version = "0.8.0"`。`crates/facade` への
`[patch.crates-io.fandhe-ai]` path patch（CLI `--config` 引数のみ・
`Cargo.lock` は非コミット。deps-policy.md 第 9 区分）で `bench-fandhe`
をビルドする。

`edb85c43` から本ブランチまでの区間には #1786（Gather／Scatter Op 追加。
`ops_shape.rs`／`error.rs`／`backend_ops.rs` の default メソッド）も含ま
れるが、推論チェーンからは非到達であり、`crates/backend-cuda/src`・
`crates/autodiff/src/optim`・`crates/facade/src`・`crates/tensor-core/src`
への到達差分は #1688（PR #1788）のみである旨を `docs/perf/logs/
infer-chain-single-sync-cuda-1689/diff_edb85c43_<after>_cuda_infer_
path.txt`（実測実行時に生成）で確認する。

## 3. 事前登録規則（本節を計測前に固定・以後変更しない）

- **比較器**: `scripts/bench/framework-compare/compare_gemm_ab.py --task
  infer --device cuda`（イシュー #1689 で `_VALID_TASKS` に `"infer"`
  を追加し、`--task train` と同型の単一形状〈`size=BATCH=64`〉判定
  ロジック・`--phases`〈`task:"infer_phases"` 行〉診断表対応を実装
  済み。`compare_gemm_ab_test.py::TaskInferTest` で 7 件の単体テストを
  追加し `python3 -m unittest compare_gemm_ab_test.py` で確認済み）
- **Tier 1（必須・判定対象）**: `size=64 / reuse` セルの `median_s`
  （`predict_resident` 区間を含む反復全体の壁時計時間）5 run 中央値比
  after/before ≤ 1.00、かつ checksum 完全一致（`compare_gemm_ab.py
  --task infer --device cuda --threshold 1.00 --per-run --modes reuse
  --require-checksum-exact`）
- **対照（非判定）**: `fresh` セル（`predict_resident` 非到達。
  `model.forward` 経路。`fresh` は #1688 の変更対象外のため差が出ない
  はずで、1.00 超はノイズ帯として記録するのみ・規則の緩和ではない）
- **診断（非判定）**: `--phases`（reuse: `predict_resident`／
  `host_copy`／`checksum`／`iter_total`・fresh: `leaf_register`／
  `forward`／`to_tensor`／…。`bench-fandhe` README「`infer --phases`」
  節参照）を各腕 1 回計測して内訳を記録する
- **計測プロトコル**: 5 round・run 単位で before／after の起動順を反転・
  各腕別バイナリ・別 `--target-dir`・`uptime`／`nvidia-smi` を run ごとに
  記録（`scripts/bench/framework-compare/run_ab_infer_chain_cuda.sh`）
- **専有ゲート（既定 ON。#1560／#1489 と同型）**: 1 分 load average <
  1.0 かつ `utilization.gpu == 0 %` を 30 秒間隔 3 サンプル連続で通過。
  最大 20 サンプル不成立なら `verdict=undetermined` を 1 回記録して終了
  （再試行ループなし）。`AB_LOAD_GATE_MODE=record_only` はユーザー明示
  指示時のみ使う（#1519 の運用）
- **R0（前提ゲート）**: `crates/backend-cuda/tests/linear_forward_device_
  real_device.rs` の `#[ignore]` 4 件（#1216 で追加済み・**本イシュー
  以前は未実測**）＋ record-only bench（`linear_forward_device_bench_
  cuda`）が after ツリーで green であること。R0 が失敗した場合は R1
  以降を実行せず「R0 未達」として記録する（`predict_device_chain` が
  依拠する基礎カーネル自体の正しさが崩れている状態で chain 経路の bit
  一致を検証しても意味がないため）
- **R1（bit 同一・単一ツリー）**: `crates/facade/tests/predict_device_
  chain_cuda_bit_identity.rs`（本イシューで新設。`predict_device_chain_
  cpu_bit_exact.rs` の CUDA 版）で、小形状 2 種（`Linear→ReLU→Linear`・
  `Linear→Linear`）＋ bench 形状（batch=64・784→256→ReLU→10）の chain
  経路（`predict_resident`）と旧経路（`forward_resident`）が `f32::
  to_bits` で完全一致すること・`predict_resident` 連続 2 回が bit 同一
  であることを確認する。**bit 一致契約であり REQ-2 複合判定へ退避しない**
  （崩れた場合は REJECT として記録する）
- **R2（bit 同一・cross-tree）**: `docs/perf/logs/infer-chain-single-sync-
  cuda-1689/run_bitdump.sh` が `predict_resident_bit_dump_cuda`
  （bench 形状の出力を `out[<i>].bits=<hex>` 形式で 1 要素 1 行印字する
  テスト）を before／after 両ツリーで実行し、`^out\[` 行を抽出して diff
  する。期待行数は `BATCH=64 * D_OUT=10` = 640 行（診断専用ファイルが
  before ツリーに無いため after ツリーからコピーして実行する。README
  参照）
- **REQ-2 複合判定の適用範囲**: `predict_resident_matches_cpu_reference_
  on_cuda`（`predict_device_chain_cuda_bit_identity.rs` 内。CUDA chain
  経路の出力を CPU `Sequential::predict` と `fandhe_ai_backend_cpu::
  assert_parity` で突合）のみ。tolerance 定数（相対誤差 1e-3 未満 または
  絶対誤差 1e-5 未満）・`docs/spec/` は不変
- **verdict の 3 値**: ADOPT（Tier 1 成立・R0/R1 green）／REJECT（R1
  不一致、または Tier 1 で `ratio>1.00` か checksum 不一致）／
  undetermined（専有ゲート不成立・件数不足・R0 未実行）。3 値とも正式
  結果として記録し、規則は事後に緩めない（security.md A08）

## 4. 実行手順

`docs/perf/logs/infer-chain-single-sync-cuda-1689/README.md`「GB10 実機
での実行手順」節を正とする。要約:

1. before（`edb85c43`）／after（本ブランチ）2 ツリーを用意
2. `run_ignored_tests.sh`（R0 → R1 → 既存回帰）を after ツリーで実行
3. `run_bitdump.sh`（R2。before ツリーへ新規テストファイルをコピーして
   から実行）
4. `orchestrate.sh 1689`（専有ゲート → Tier 1 A/B）
5. 帰属 diff（`edb85c43` ↔ after の到達差分）を生成
6. 生成物を `ab/`／`bitdump/`／`ignored/`・`env_info.txt` へ回収し、
   内部ホスト名・ユーザー名・実パスをマスクする

## 5. 記入欄

### 5.1 R0（前提ゲート）

| テスト | 結果 |
|---|---|
| `linear_forward_device_matches_cpu_reference_on_real_device` | 未実測 |
| `linear_forward_device_matches_gemm_resident_rhs_act_bit_exact_on_real_device` | 未実測 |
| `linear_forward_device_two_layer_chain_matches_cpu_reference_on_real_device` | 未実測 |
| `linear_forward_device_rejects_shape_mismatches_and_handles_empty_input_on_real_device` | 未実測 |
| `linear_forward_device_bench_cuda`（record-only bench） | 未実測 |

R0 総合判定: 未実測

### 5.2 R1（chain bit 一致・単一ツリー）

| テスト | 結果 |
|---|---|
| `predict_device_chain_matches_legacy_path_bit_exact_on_cuda_relu_fusion` | 未実測 |
| `predict_device_chain_matches_legacy_path_bit_exact_on_cuda_no_activation_fusion` | 未実測 |
| `predict_device_chain_matches_legacy_path_bit_exact_on_cuda_bench_shape` | 未実測 |
| `predict_resident_matches_cpu_reference_on_cuda`（REQ-2 複合判定） | 未実測 |
| `predict_resident_bit_dump_cuda`（動作確認のみ・R2 で cross-tree diff） | 未実測 |

R1 総合判定: 未実測

### 5.3 R2（bit 同一・cross-tree）

| 項目 | 値 |
|---|---|
| before_lines | 未実測 |
| after_lines | 未実測 |
| 期待行数 | 640 |
| diff | 未実測 |

R2 判定: 未実測

### 5.4 Tier 1（size=64 / reuse。判定対象）

| 指標 | 値 |
|---|---|
| before median_s（5 run 中央値） | 未実測 |
| after median_s（5 run 中央値） | 未実測 |
| ratio（after/before） | 未実測 |
| checksum_exact_match | 未実測 |
| 判定 | 未実測 |

### 5.5 対照（size=64 / fresh。非判定・参考）

| 指標 | 値 |
|---|---|
| before median_s | 未実測 |
| after median_s | 未実測 |
| ratio | 未実測 |

### 5.6 診断（`--phases`。非判定）

reuse: `predict_resident`／`host_copy`／`checksum`／`iter_total` の
before/after 内訳は未実測。fresh: `leaf_register`／`forward`／
`to_tensor`／`host_copy`／`checksum`／`iter_total` の内訳も未実測。

### 5.7 env_info

`docs/perf/logs/infer-chain-single-sync-cuda-1689/env_info.txt` を参照
（未実測のため未記入）。

## 6. verdict

**undetermined**（本エージェント実行環境に DGX Spark GB10 実機への到達
手段がなく未実測のため。R0/R1/R2/Tier 1 いずれも実測完了後に本節・§5 を
更新し、事前登録規則（§3）に基づき ADOPT／REJECT／undetermined のいずれ
かを確定する）。

## 7. スコープ外

- Metal 側の同型 A/B は #1580 が担当（本イシューの対象外）
- `Sigmoid`／`Tanh` 等 `Linear`／`ReLU` 以外の層を含む chain 対応の拡張
- crates.io 公開・framework-compare の承認ピン更新（本イシューは path
  patch 限定の計測であり公開版への反映は別イシュー）
- CUDA バックエンドの `synchronize()` 呼び出し回数を数える診断カウンタの
  新設（Metal の `dispatch_count` 契約〈`docs/backend-metal-command-
  batching-design.md`〉と異なり、CUDA はストリーム順序契約（`docs/
  backend-cuda-async-execution-design.md`）＋bit 同一＋A/B で代替する
  方針を維持する）
