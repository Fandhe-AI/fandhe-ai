# イシュー #1689 実測スキャフォールド（CUDA 推論 forward チェーン単一同期化）

## 位置づけ（本 PR 時点では未実測）

本 PR の実行環境（Linux。`docs/real-hardware-verification-env.local.md`・
`CUDA_NODE` 環境変数のいずれも確認できず、ローカル GPU も driver/library
version mismatch により NVML／CUDA 初期化不可）には DGX Spark GB10 実機
への到達手段がないため、**実測値は一切含まれていない**。本ディレクトリは
スキャフォールド（実行スクリプト・事前登録判定規則・記入欄）のみを提供し、
実測は GB10 実機を持つセッションへ申し送る（`docs/perf/logs/
train-resident-grad-cuda-1560/`・`docs/perf/logs/cuda-mse-backward-1692/`
と同型の運用）。

## 目的

イシュー #1689「CUDA 推論 forward チェーンの単一同期化を GB10 実機で
A/B し bit 同一を確認する」の実測記録先。#1579（設計）・#1688（実装。
`DeviceParamStore::predict_device_chain`・facade `Sequential::
predict_resident` の chain 経路優先化・CUDA `linear_forward_device_
tracked` 結線）の bit 同一検証・性能 A/B を行う。

## 比較対象 2 腕（事前登録・固定）

- **before 腕**: `edb85c43`（#1788〈#1688 実装 PR〉マージ直前の main。
  `predict_resident` は `forward_from_flat_leaves` 経由＝層境界ごとに
  upload/download する従来経路）
- **after 腕**: 本イシューのブランチ（`crates/*/src` は評価対象コミットと
  同一。単一同期チェーン経由）

両腕とも workspace `version = "0.8.0"`。`crates/facade` への
`[patch.crates-io.fandhe-ai]` path patch（CLI `--config` 引数のみ。
deps-policy.md 第 9 区分）で `bench-fandhe` をビルドする。

`edb85c43` から本ブランチまでの区間には #1786（Gather／Scatter Op 追加）
も含まれるが、`crates/backend-cuda/src`・`crates/autodiff/src/optim`・
`crates/facade/src`・`crates/tensor-core/src` への到達差分は #1788 のみ
であることを `diff_edb85c43_<after>_cuda_infer_path.txt`（本ディレクトリ・
実測実行時に生成）で確認する。

## 事前登録判定規則（計測前に固定・結果を見て変更しない）

`docs/perf/infer-chain-single-sync-cuda-ab.md` の「事前登録規則」節が正。
要約:

- **Tier 1（必須・判定対象）**: `size=64 / reuse` セルの `median_s`
  （`predict_resident` 区間を含む反復全体）5 run 中央値比 after/before
  ≤ 1.00、かつ checksum 完全一致（`compare_gemm_ab.py --task infer
  --device cuda --threshold 1.00 --per-run --modes reuse
  --require-checksum-exact`）
- **対照（非判定）**: `fresh` セル（`predict_resident` 非到達。
  `model.forward` 経路）。1.00 超はノイズ帯として記録するのみ
- **診断（非判定）**: `--phases`（reuse: `predict_resident`／
  `host_copy`／`checksum`／`iter_total`・fresh: `leaf_register`／
  `forward`／`to_tensor`／…）を各腕 1 回
- **計測プロトコル**: 5 round・run 単位で before／after の起動順を反転・
  各腕別バイナリ・別 `--target-dir`
- **専有ゲート（既定 ON。#1560／#1489 と同型）**: 1 分 load average < 1.0
  かつ `utilization.gpu == 0 %` を 30 秒間隔 3 サンプル連続で通過。最大
  20 サンプル不成立なら `verdict=undetermined` を 1 回記録して終了
  （再試行ループなし）。`AB_LOAD_GATE_MODE=record_only` はユーザー明示
  指示時のみ使う（#1519 の運用）
- **R0（前提ゲート）**: `crates/backend-cuda/tests/linear_forward_device_
  real_device.rs` の `#[ignore]` 4 件＋record-only bench が after ツリー
  で green。fail なら R1 以降を実行せず「R0 未達」として記録する
- **R1（bit 同一・単一ツリー）**: `crates/facade/tests/predict_device_
  chain_cuda_bit_identity.rs` で chain 経路（`predict_resident`）と旧
  経路（`forward_resident`）が bit 完全一致・`predict_resident` 連続 2
  回が bit 同一であることを確認する（bit 一致契約であり REQ-2 複合判定
  へ退避しない。崩れた場合は REJECT として記録する）
- **R2（bit 同一・cross-tree）**: `run_bitdump.sh` が `predict_resident_
  bit_dump_cuda`（bits 印字テスト）を before／after 両ツリーで実行し
  `^out\[` 行を diff する（期待行数はモデル形状〈batch=64・出力次元
  10〉から機械計算した 640 行）
- **REQ-2 複合判定の適用範囲**: `predict_resident_matches_cpu_reference_
  on_cuda`（CUDA chain 経路 vs CPU `Sequential::predict`）のみ。
  tolerance 定数・`docs/spec/` は不変
- **verdict**: ADOPT（Tier 1 成立・R0/R1 green）／REJECT（R1 不一致、
  または Tier 1 で `ratio>1.00` か checksum 不一致）／undetermined
  （専有ゲート不成立・件数不足・R0 未実行）。3 値とも正式結果として記録
  し、規則は事後に緩めない

## ディレクトリ構成

| パス | 内容 |
|------|------|
| `orchestrate.sh` | 専有ゲート → `run_ab_infer_chain_cuda.sh` の実行ラッパー（`--dry-run` あり） |
| `run_ignored_tests.sh` | R0（前提ゲート）→ R1（chain bit 一致）→ 既存回帰の順に個別プロセス実行しログを保存する（R0 失敗時は即座に打ち切る） |
| `run_bitdump.sh` | before/after 両ツリーで `predict_resident_bit_dump_cuda` を実行し bit 同一を diff で確認する（R2） |
| `env_info.txt` | 実行環境・sha・バイナリ sha256・判定結果の記入欄（未実測のため未記入） |
| `ab/` | `run_ab_infer_chain_cuda.sh` の出力回収先（JSONL・compare md・sha・tree・uptime・skipped。未生成） |
| `bitdump/` | `run_bitdump.sh` の出力回収先（`{before,after}_raw.log`／`{before,after}_bits.txt`／`bitdump_diff.txt`。未生成） |
| `ignored/` | `run_ignored_tests.sh` の出力回収先（テストごとのログ。未生成） |

`scripts/bench/framework-compare/run_ab_infer_chain_cuda.sh`（本ディレク
トリではなくスクリプト本体はそちらに配置。deps-policy.md の慣例に従い
framework-compare 配下に置く）が実際の A/B 実行スクリプト。

## GB10 実機での実行手順

1. before/after 2 本のツリーを用意する（`git archive edb85c43 | tar -x
   -C <scratch>/before`・作業ブランチの worktree を after として使う）。
   `docs/real-hardware-verification-env.md` §3 の rsync 手順（`.env*`／
   `settings.local.json`／`.local.md` 除外）に従う
2. 両ツリーの `crates/facade` 絶対パスを控える
3. R0（前提ゲート）→ R1（chain bit 一致）→ 既存回帰: after ツリーで
   `./run_ignored_tests.sh`（`REPO_ROOT` を明示指定する場合は環境変数で
   上書き可能）。R0 が失敗した場合は R1 以降を実行せず記録する
4. R2（cross-tree bit 同一確認）:

   ```sh
   BEFORE_TREE=/absolute/path/to/before AFTER_TREE=/absolute/path/to/after \
     ./run_bitdump.sh
   ```

5. Tier 1 A/B:

   ```sh
   AB_BEFORE_FACADE_PATH=/absolute/path/to/before/crates/facade \
   AB_AFTER_FACADE_PATH=/absolute/path/to/after/crates/facade \
     ./orchestrate.sh 1689
   ```

   `AB_LOAD_GATE_MODE=record_only` を付けると専有ゲートを opt-out できる
   （ユーザー明示指示がある場合のみ）
6. 帰属 diff（before ↔ after の到達差分）:

   ```sh
   git diff edb85c43 <after_sha> -- crates/backend-cuda/src \
     crates/autodiff/src/optim crates/facade/src crates/tensor-core/src \
     > diff_edb85c43_<after_sha>_cuda_infer_path.txt
   ```

7. 生成物を `ab/`／`bitdump/`／`ignored/`・`env_info.txt` へ回収し、内部
   ホスト名・ユーザー名・実パスが含まれないことを確認してマスクする
8. 結果を `docs/perf/infer-chain-single-sync-cuda-ab.md` の記入欄・
   `docs/inference-chain-single-sync-design.md` §9 へ転記する

## 注意

- 内部ホスト名・ユーザー名・絶対パスを成果物（ログ・env_info・README）
  に書かない
- 実測値の捏造・事前登録規則の事後緩和は禁止（security.md A08）
- 本番コード（`crates/*/src`）は本イシューでは変更しない
