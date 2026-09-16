# イシュー #1590 実測スキャフォールド（CUDA GEMM VJP NT／TN 転置入口 train A/B）

## 位置づけ（本 PR 時点では未実測）

本 PR の実行環境（Linux。`docs/real-hardware-verification-env.local.md`・
`CUDA_NODE` 環境変数のいずれも確認できず、ローカル `/dev/nvidia0` も
driver/library version mismatch により NVML／CUDA 初期化不可）には
DGX Spark GB10 実機への到達手段がないため、**train A/B の実測値は一切
含まれていない**。本ディレクトリはオーケストレーション（実行スクリプト・
事前登録判定規則・記入欄）のみを提供し、実測は GB10 実機を持つセッション
へ申し送る（`docs/perf/logs/train-resident-grad-cuda-1560/`・
`docs/perf/logs/infer-chain-single-sync-cuda-1689/` と同型の運用）。

一方、parity（5/5 pass）・補助 A/B（8 形状・NT/TN・単一起動）は
低レイヤー診断（イシュー #1574・2026-09-12・GB10 専有・tree `097bff19`）
で既に実測済みであり、`docs/perf/cuda-gemm-vjp-transposed-entry.md`
§3.1／§3.2 へ転記済み（出典 `docs/perf/logs/lowlayer-diagnosis-2026-09-12/
dgx/gemm_transposed_{parity,perf}.log`）。本イシューが埋めるのは
train A/B（§3.3）のみである。

## 目的

イシュー #1590「#1214 NT／TN 転置入口の train フェーズ A/B を GB10 で
実測する」の実測記録先。#1214（PR #1226・マージコミット `ab0b77d0`）が
追加した CUDA GEMM VJP 専用 NT／TN 転置入口の train fresh／reuse
（`size=64`）性能への影響を確認する。

## 比較対象 2 腕（事前登録・固定）

- **before 腕**: `82058501`（#1214 マージ直前の main）
- **after 腕**: `ab0b77d0`（#1214 のマージコミット自身）

両腕とも workspace `version = "0.6.0"`。既存 run_ab_*.sh（#1560／#1689）
と異なり、**単一チェックアウト（本ブランチ HEAD）への facade path
patch ではなく、before／after それぞれ丸ごと展開したツリー**（`git
archive 82058501`／`git archive ab0b77d0` で得られる状態と同一）を使う。
理由: 本ブランチ HEAD の `bench-fandhe` ハーネスは `82058501` 時点の
facade に存在しない API を呼ぶため、HEAD ハーネス＋旧 facade の path
patch はビルド不能（`docs/perf/cuda-gemm-vjp-transposed-entry.md` §3.3
の設計注記）。各ツリー**自身の** `scripts/bench/framework-compare/` で
`cargo build`（同一ツリー内の `crates/facade` への
`[patch.crates-io.fandhe-ai]` path patch。CLI `--config` のみ・
非コミット）することでハーネスと facade のバージョンが常に一致する。

両ツリーの `scripts/bench/framework-compare/` は完全同一（`git diff
--stat 82058501 ab0b77d0 -- scripts/bench/framework-compare` が空。
workspace `version = "0.6.0"`・bench-fandhe のピンは `fandhe-ai =
"=0.6.0"`）。ただし `compare_gemm_ab.py` の判定は**本リポジトリ（呼び出し
元。本イシューのブランチ HEAD）のコピー**を使う（`--require-checksum-
exact` 等、#1214 当時になかった新しいフラグに対応するため。JSONL の
読み取りのみなので cargo は不要で、本リポジトリの framework-compare へ
一切書き込まない）。

## 事前登録判定規則（計画確定・実測前に固定・結果を見て変更しない）

`docs/perf/cuda-gemm-vjp-transposed-entry.md` §3.3 が正。要約:

- **Tier 1（必須・判定対象）**: `size=64` の **fresh／reuse 両方**の
  `step_total`（`median_s`）5 run 中央値比 after/before ≤ 1.00、かつ
  checksum 完全一致（`compare_gemm_ab.py --task train --device cuda
  --threshold 1.00 --per-run --modes {fresh,reuse} --require-checksum-
  exact`。mode ごとに 2 回呼ぶ）
- **診断（非判定）**: `--phases`（`backward`／`step_total` 等）を各腕・
  各 mode 1 起動
- **正式な補助 A/B（§3.2 の確定用・非判定）**: `gemm_transposed_perf`
  （`crates/backend-cuda/tests/gemm_transposed_perf.rs`）を after ツリー
  で 5 プロセス起動し、形状ごとの speedup 中央値を記録する（診断系列
  ＝低レイヤー診断の 1 起動値を正式値へ昇格させない）
- **計測プロトコル**: 5 round・run 単位で before／after の起動順を反転・
  腕ごとに別バイナリ・別 `--target-dir`（各ツリー内）・run ごとに
  `uptime`／`nvidia-smi` を記録
- **専有ゲート（既定 ON。#1560／#1689 と同型）**: 1 分 load average < 1.0
  かつ `utilization.gpu == 0 %` を 30 秒間隔 3 サンプル連続で通過。最大
  20 サンプル不成立なら `verdict=undetermined` を 1 回記録して終了
  （`AB_LOAD_GATE_MODE=record_only` はユーザー明示指示時のみ）。
  `run_ab_vjp_transposed_cuda.sh` は before／after 両腕のリリースビルド
  （数分規模の CPU 負荷を伴う）完了後・計測ループ（5 round）開始直前に
  **同一条件のゲートを再実行する**（`gate-postbuild-<label>.log`。
  ビルド中の負荷変動でオーケストレータの事前ゲート確認が計測開始時点
  では成立しなくなっている可能性を排除するため。イシュー #1590 PR
  #1812 codex-review 指摘）。再ゲートが不成立の場合は計測を実行せず
  `verdict=undetermined` を記録して終了する（`AB_LOAD_GATE_MODE` は
  `orchestrate.sh` から環境変数として引き継がれ、同じ opt-out 判断を
  尊重する）
- **R1（HEAD tree の `#[ignore]` 非後退）**: 下記「対象テスト」参照
- **verdict**: ADOPT（Tier 1 両セル成立・R1 green）／REJECT（Tier 1 で
  `ratio>1.00` または checksum 不一致、または R1 で #1214 起因の fail）／
  undetermined（専有ゲート不成立・件数不足・未実行）

## ディレクトリ構成

| パス | 内容 |
|------|------|
| `orchestrate.sh` | 専有ゲート → `run_ab_vjp_transposed_cuda.sh` の実行ラッパー（`--dry-run` あり） |
| `run_ignored_tests.sh` | R1（対象 `#[ignore]` テスト群）を HEAD（`REPO_ROOT`）で個別プロセス実行しログを保存する。`AUX_TREE=<after ツリー絶対パス>` を指定すると §3.2 正式補助 A/B（`gemm_transposed_perf` 5 起動）も **after ツリー内で** 続けて実行する（未指定時は aux/ をスキップし fail-closed。PR #1812 Cursor Bugbot 指摘: R1 の一部は post-#1214 API 依存のため HEAD 限定・aux はコンタミ防止のため after ツリー限定で、両者を同一 `REPO_ROOT` に一本化できない） |
| `aggregate_aux_ab.py` | `gemm_transposed_perf` の 5 プロセス起動ログから形状ごとの speedup 中央値表を生成する（`--self-test` あり） |
| `env_info.txt` | 実行環境・sha・バイナリ sha256・判定結果の記入欄（未実測のため未記入） |
| `ab/` | `run_ab_vjp_transposed_cuda.sh` の出力回収先（JSONL・compare md・sha・tree・uptime・skipped。未生成） |
| `ab/rev-stamp-verification-1590.md` | 両腕の `.rev-stamp`（`rev-stamp-{before,after}-1590.txt`）とツリー内容指紋（`tree-hashes-*.txt`）の独立検証記録（PR #1909 codex P2 対応・2026-09-16） |
| `ignored/` | `run_ignored_tests.sh` の出力回収先（テストごとのログ。未生成） |
| `aux/` | `gemm_transposed_perf` 5 プロセス起動ログの回収先（未生成） |

`scripts/bench/framework-compare/run_ab_vjp_transposed_cuda.sh`（本
ディレクトリではなくスクリプト本体はそちらに配置。deps-policy.md の
慣例に従い framework-compare 配下に置く）が実際の A/B 実行スクリプト。

## 対象テスト（R1・`#[ignore]`）

- `crates/backend-cuda/tests/gemm_transposed_parity.rs`（5 件）
- `crates/backend-cuda/tests/gemm_transposed_perf.rs`（2 件。§3.2 の
  正式補助 A/B と兼用。5 プロセス起動）
- `crates/backend-cuda/tests/gemm_fp32_strict_into_parity.rs`（#1559 の
  NT／TN 経路が既存経路の上に成立していることの非後退確認）
- `crates/backend-cuda/tests/transpose_parity.rs`（転置カーネル自体の
  非後退確認）
- `crates/backend-cuda/src/ops.rs` 内 `repack_count_tests`（`--lib`。
  env-adaptive・`pub(crate)` カウンタへは触れない）
- `crates/facade/tests/device_param_store_backend_parity.rs` の CUDA
  対象 2 件（既存回帰の非後退確認）

## GB10 実機での実行手順

1. before／after 2 本のツリーを用意する: `git archive 82058501 | tar -x
   -C <scratch>/before`・`git archive ab0b77d0 | tar -x -C <scratch>/
   after`。`docs/real-hardware-verification-env.md` §3 の rsync 手順
   （`.env*`／`settings.local.json`／`.local.md` 除外）に従う。各ツリーに
   `git rev-parse <sha> > .rev-stamp` を書いておく（展開ツリーは `.git`
   を持たないため版確認は `.rev-stamp` のみ）
2. R1（本ブランチ HEAD 限定）＋正式補助 A/B（§3.2・after ツリー限定）を
   まとめて実行する:

   ```sh
   AUX_TREE=/absolute/path/to/after \
     docs/perf/logs/cuda-gemm-vjp-transposed-entry-1590/run_ignored_tests.sh
   ```

   `run_ignored_tests.sh` は R1 の各テストを本ブランチ HEAD（`REPO_ROOT`。
   `gemm_fp32_strict_into_parity` 等 #1214 より後発の API に依存するため
   HEAD 限定）で実行し、`gemm_transposed_perf` の 5 プロセス起動のみを
   `AUX_TREE`（after ツリー。手順 1 で用意した `ab0b77d0` 展開先）へ
   `cd` してから実行する（post-#1214 の CUDA 変更が正式補助 A/B の数値
   へ混入するのを防ぐため。REPO_ROOT 自体を after ツリーへ向ける方式は
   R1 の後発ケースが存在せずビルド不能になるため採らない）。`AUX_TREE`
   を省略すると `aux/` は fail-closed でスキップされる（`aux/SKIPPED.txt`
   に理由を記録。R1 のみを先に確認したい場合に使う）
3. `aggregate_aux_ab.py` で `aux/gemm_transposed_perf_run{1..5}.log` を
   集計する
4. Tier 1 A/B:

   ```sh
   AB_BEFORE_TREE=/absolute/path/to/before \
   AB_AFTER_TREE=/absolute/path/to/after \
     docs/perf/logs/cuda-gemm-vjp-transposed-entry-1590/orchestrate.sh 1590
   ```

   `AB_LOAD_GATE_MODE=record_only` を付けると専有ゲートを opt-out できる
   （ユーザー明示指示がある場合のみ）
5. 生成物を `ab/`／`ignored/`／`aux/`・`env_info.txt` へ回収し、内部
   ホスト名・ユーザー名・実パスが含まれないことを確認してマスクする
6. 結果を `docs/perf/cuda-gemm-vjp-transposed-entry.md` §3.3／§4 へ転記
   する

## 注意

- 内部ホスト名・ユーザー名・絶対パスを成果物（ログ・env_info・README）
  に書かない
- 実測値の捏造・事前登録規則の事後緩和は禁止（security.md A08）
- 本番コード（`crates/*/src`）は本イシューでは変更しない
