# GB10 大コア affinity（#1576）A/B の事前登録規則・実行基盤（イシュー #2117）

状態: **未実測**（実行ホストが x86_64 で DGX Spark GB10 への到達手段が無い）。
`GB10_AFFINITY_ENABLED = false`（`crates/backend-cpu/src/gb10_affinity.rs`）は維持し、実測・判定・結線は GB10 セッションへ申し送る。
実測値・推定値は本ディレクトリにも `docs/perf/cpu-gemm-gb10-affinity-ab.md` にも記入していない（verdict は undetermined のまま）。

## 収録物

| パス | 内容 |
|---|---|
| `RULE.txt` | 実測前に固定した判定規則（14 セル・5 round 中央値・checksum 完全一致・専有ゲート load1<1.0・機構発火の前提） |
| `on-arm.patch` | after 腕用の 1 行差分（`GB10_AFFINITY_ENABLED = false -> true`） |
| `smoke-x86/` | x86 でのパイプライン疎通 smoke（**系列ではない**。機構は発火せず ratio に意味はない） |
| `gb10/` | （GB10 セッションで追加）正式系列の生ログ・env_info・compare 表 |

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

- after worktree では `--lib` テストを実行しない（`GB10_AFFINITY_ENABLED` が false 前提の単体テストが落ちる）。
  機構発火の確認は結合テスト `--test gb10_affinity_report` のみ（スクリプトが実行する）。
- N=256 は `m*n*k` が `GB10_AFFINITY_MAX_WORK` 以下でルーティング対象（処置セル）。N>=512 が非ルーティングのガード。
- `Cargo.toml`／`Cargo.lock`（framework-compare）はコミットしない（`bench_fandhe_lock_restore.sh` が復元する）。
