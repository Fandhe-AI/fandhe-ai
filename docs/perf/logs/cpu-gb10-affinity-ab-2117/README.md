# GB10 大コア affinity（#1576）A/B の事前登録規則・実行基盤（イシュー #2117）

状態: **GB10 で実測済み（2026-10-21・イシュー #3187。ラベル `2117-gb10-3187`）。verdict は REJECT**
（正式系列。14 セル中 6 セルで中央値 ratio > 1.00。詳細は `docs/perf/cpu-gemm-gb10-affinity-ab.md` の記入欄）。
`GB10_AFFINITY_ENABLED = false`（`crates/backend-cpu/src/gb10_affinity.rs`）は維持している。

## 収録物

| パス | 内容 |
|---|---|
| `RULE.txt` | 実測前に固定した判定規則（14 セル・5 round 中央値・checksum 完全一致・専有ゲート load1<1.0・機構発火の前提） |
| `on-arm.patch` | after 腕用の 1 行差分（`GB10_AFFINITY_ENABLED = false -> true`） |
| `smoke-x86/` | x86 でのパイプライン疎通 smoke（**系列ではない**。機構は発火せず ratio に意味はない） |
| `gb10/` | 正式系列（`2117-gb10-3187`）の生ログ（`results-{before,after}-*-cpu-{gemm,train,infer}.jsonl`）・`compare-{gemm,train,infer}-*.md`・`compare-exit-*.log`・`load_gate-*.log`・`affinity-report-*.txt`・`env_info-*.txt`・`tree-*`・`sha-*`・`uptime-*`・`build-*.err`・`skipped-*.log`。ELF（`bench-fandhe-2117-*`）は収録しない。ホスト名・`$HOME` はマスク済み。計測ツリーの同定は `../cpu-gemm-reuse-fresh-3187/rev-stamp-verification-3187.md` |

## GB10 での実行手順

1. main の worktree（before）と、同一コミットに `on-arm.patch` を当てた worktree（after。コミットしない）を用意する。
   `git -C <after> apply docs/perf/logs/cpu-gb10-affinity-ab-2117/on-arm.patch`
2. `RAYON_NUM_THREADS` を unset し、他の重い処理を止める。
3. 実行（`crates/facade` の絶対パスを渡す）:
   ```
   AB_BEFORE_FACADE_PATH=<before>/crates/facade \
   AB_AFTER_FACADE_PATH=<after>/crates/facade \
     bash scripts/bench/framework-compare/run_ab_gb10_affinity_cpu.sh 2117-gb10
   ```
   既定は `AB_AFFINITY_PRECHECK=assert`（機構が after だけで実際に発火していなければベンチ前に停止）。
4. `results/raw/` の JSONL・`load_gate-2117-*`・`affinity-report-2117-*`・`env_info-2117-*`・`compare-*-2117-cpu-*.md` を
   マスク（ホスト名 `masked`・`$HOME` -> `<home>`・ユーザー名除去）して `gb10/` へ収録する。
5. `RULE.txt` に従い判定し、`docs/perf/cpu-gemm-gb10-affinity-ab.md` の記入欄・verdict を更新する。
   ADOPT なら `RULE.txt` の「ADOPT 時の結線」を別 PR で実施する。

## 注意

- 実測では before／after の checkout に `.rev-stamp`（HEAD の sha）を置き、`.git/info/exclude` へ追記して `verify_no_untracked_build_inputs` を通した（未追跡扱いにならない）。

- after worktree では `--lib` テストを実行しない（`GB10_AFFINITY_ENABLED` が false 前提の単体テストが落ちる）。
  機構発火の確認は結合テスト `--test gb10_affinity_report` のみ（スクリプトが実行する）。
- N=256 は `m*n*k` が `GB10_AFFINITY_MAX_WORK` 以下でルーティング対象（処置セル）。N>=512 が非ルーティングのガード。
- `Cargo.toml`／`Cargo.lock`（framework-compare）はコミットしない（`bench_fandhe_lock_restore.sh` が復元する）。
