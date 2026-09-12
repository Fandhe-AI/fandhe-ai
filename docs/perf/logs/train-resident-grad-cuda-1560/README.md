# イシュー #1560 実測スキャフォールド（CUDA resident weight 勾配経路）

## 位置づけ（本 PR 時点では未実測）

本 PR の実行環境（Linux・QEMU VM）には DGX Spark GB10 実機への到達手段
（`docs/real-hardware-verification-env.local.md`・`CUDA_NODE` 環境変数・
`~/.ssh/config` のいずれも確認できず）がないため、**実測値は一切含まれて
いない**。本ディレクトリはスキャフォールド（実行スクリプト・事前登録
判定規則・記入欄）のみを提供し、実測は GB10 実機を持つセッションへ
申し送る（#1512／#1513／#1515 と同型の運用）。

## 目的

イシュー #1560「CUDA resident weight 勾配経路（#1559 実装）を GB10 実機で
bit 同一確認し train reuse A/B を計測する」の実測記録先。#1559
（`crates/backend-cuda/src/ops.rs::CudaBackendOps::gemm_fp32_strict_into_impl`）
の bit 同一検証・性能 A/B を行う。

## 比較対象 2 腕（事前登録・固定）

- **before 腕**: `d77f8bde`（#1569 マージ直前の main。`gemm_fp32_strict_into`
  の CUDA 実装なし＝既定 `Unsupported` フォールバック経由）
- **after 腕**: 本イシューのブランチ（`origin/main` `e41db903` + テスト／
  スクリプト／docs のみ。`crates/*/src` は `e41db903` と同一。
  `git diff e41db903 -- crates/*/src` が空であることは本 PR の diff で
  確認できる）

両腕とも workspace `version = "0.8.0"`。`crates/facade` への
`[patch.crates-io.fandhe-ai]` path patch（CLI `--config` 引数のみ。
deps-policy.md 第 9 区分）で `bench-fandhe` をビルドする。

## 事前登録判定規則（計測前に固定・結果を見て変更しない）

- **Tier 1（必須・判定対象）**: `size=64 / reuse` セルの `step_total`
  （bench-fandhe `median_s`）5 run 中央値比 after/before ≤ 1.00、かつ
  checksum 完全一致（`compare_gemm_ab.py --device cuda --task train
  --threshold 1.00 --per-run`）
- **対照（非判定）**: `fresh` セル（resident 経路非到達）。符号不一致・
  1.00 超はノイズ帯として記録するのみ。規則の緩和ではない
- **診断（非判定）**: `--phases` の `backward`／`device_update`／
  `step_total` 内訳（同期点移動の観察用）
- **計測プロトコル**: 5 round・run 単位で before/after の起動順を反転・
  各腕別バイナリ・別 `--target-dir`
- **専有ゲート（既定 ON。#1489 と同型）**: 1 分 load average < 1.0 かつ
  `utilization.gpu == 0 %` を 30 秒間隔で 3 サンプル連続 → 通過。最大 20
  サンプルで不成立なら `verdict=undetermined` を 1 回記録して終了
  （再試行ループなし）。`AB_LOAD_GATE_MODE=record_only` はユーザーの
  明示指示がある場合のみ使う（#1519 の運用）
- **bit 同一（R2）**: `cuda_graph_step_bit_identity::eager_baseline`
  （`--exact`・opt-in OFF）を before／after 両ツリーで実行し、
  `^(step\[|final\.param\[)` 行を抽出して diff。期待 4782 行（#1480
  実績値）・差分 0
- **副次観測（事前登録・非判定）**: after 腕で `bench-fandhe --graph on`
  を 1 回起動し `graph_captured`／`graph_replayed`／
  `graph_sgd_kernel_launches` カウンタを記録する。`#1569` により CUDA
  reuse は NT/TN 層で `any_resident == true` になるため、
  `DeviceParamStore::step` の CUDA Graph capture 分岐（`!any_resident`
  限定）が非到達になる可能性がある（`docs/backend-cuda-graph-step-capture-
  design.md:98` 参照）。判定には用いない

## ディレクトリ構成

| パス | 内容 |
|------|------|
| `orchestrate.sh` | 専有ゲート → `run_ab_resident_grad_cuda.sh` の実行ラッパー（`--dry-run` あり） |
| `run_bitdump.sh` | before/after 両ツリーで `eager_baseline` を実行し bit 同一を diff で確認する |
| `run_ignored_tests.sh` | R1 の `#[ignore]` テスト群を個別実行しログを保存する |
| `env_info.txt` | 実行環境・sha・バイナリ sha256 の記入欄（未実測のため未記入） |
| `ab/` | `run_ab_resident_grad_cuda.sh` の出力回収先（JSONL・compare md・sha・tree・uptime・skipped。未生成） |
| `bitdump/` | `run_bitdump.sh` の出力回収先（`{before,after}_raw.log`／`{before,after}_bits.txt`／`bitdump_diff.txt`。未生成） |
| `ignored/` | `run_ignored_tests.sh` の出力回収先（テストごとのログ。未生成） |

`scripts/bench/framework-compare/run_ab_resident_grad_cuda.sh`
（本ディレクトリではなくスクリプト本体はそちらに配置。deps-policy.md の
慣例に従い framework-compare 配下に置く）が実際の A/B 実行スクリプト。

## GB10 実機での実行手順

1. before/after 2 本のツリーを用意する（`git archive d77f8bde | tar -x -C
   <scratch>/before`・作業ブランチの worktree を after として使う）。
   `docs/real-hardware-verification-env.md` §3 の rsync 手順（`.env*`／
   `settings.local.json`／`.venv*/`／`.local.md` 除外）に従う
2. 両ツリーの `crates/facade` 絶対パスを控える
3. bit 同一確認:

   ```sh
   BEFORE_TREE=/absolute/path/to/before AFTER_TREE=/absolute/path/to/after \
     ./run_bitdump.sh
   ```

4. `#[ignore]` 群（R1）: after ツリーで `./run_ignored_tests.sh`
   （`REPO_ROOT` を明示指定する場合は環境変数で上書き可能）。失敗があれば
   before ツリーで同テストを再実行し pre-existing かを切り分ける
5. A/B（R3）:

   ```sh
   AB_BEFORE_FACADE_PATH=/absolute/path/to/before/crates/facade \
   AB_AFTER_FACADE_PATH=/absolute/path/to/after/crates/facade \
     ./orchestrate.sh 1560
   ```

   `AB_LOAD_GATE_MODE=record_only` を付けると専有ゲートを opt-out できる
   （ユーザー明示指示がある場合のみ）
6. 副次観測: after ツリーで
   `cargo build --release -p bench-fandhe --features graph-step --config
   'patch.crates-io.fandhe-ai.path="<after facade>"'` の後
   `--task train --device cuda --size 64 --mode reuse --graph on` を
   1 回起動し `graph_*` カウンタを記録する
7. 生成物を `ab/`／`bitdump/`／`ignored/`・`env_info.txt` へ回収し、内部
   ホスト名・ユーザー名・実パスが含まれないことを確認してマスクする
8. 結果を `docs/perf/train-resident-grad-device-update.md` §7 へ転記する

## 注意

- 内部ホスト名・ユーザー名・絶対パスを成果物（ログ・env_info・README）
  に書かない
- 実測値の捏造・事前登録規則の事後緩和は禁止（security.md A08）
- 本番コード（`crates/*/src`）は本イシューでは変更しない
