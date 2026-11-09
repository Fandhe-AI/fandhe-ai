# cpu-gemm-reuse-release-sizegate-4158

イシュー #4158（eager release にサイズゲートを付けて GB10 で A/B する）の記録置き場。
設計・結果の本文は `docs/perf/cpu-gemm-reuse-fresh-inversion.md` §18。

状態: **部品のみ・未実測**。`RULE.txt` は部品 PR のマージ後、計測より前に単独コミットで固定する。
GB10 の実測と判定は未実施（このディレクトリに `gb10/` と `judge-4158.md` はまだ無い）。

## 収録物

| パス | 内容 |
|---|---|
| `on-arm.patch` | after 腕の 1 行差分（`EAGER_MATMUL_RELEASE_DEFAULT_ENABLED` を `true` へ）。サイズゲート閾値は両腕で共通 |
| `RULE.txt` | 事前登録規則（計測前に別コミットで固定。後から追加） |
| `gb10/`・`judge-4158.md`・`rev-stamp-verification-4158.md` | 実測後に追加 |

## 実行手順

ツール: `scripts/bench/framework-compare/run_ab_eager_release_cpu.sh`（`AB_ISSUE=4158`）・`judge_eager_release_ab.py --rule-set 4158`。
GB10 は専有機体で実施する（ホスト名は公開面で `<cuda-node>`。実名は Git 管理外の `docs/real-hardware-verification-env.local.md`）。
手順 1〜2 は #3226 の `../cpu-gemm-reuse-release-3226/README.md` と同じ（bundle 配布・3 クローン・`.rev-stamp`・`RAYON_NUM_THREADS` を unset）。

```sh
AB_ISSUE=4158 AB_PROFILE=gb10 AB_BEFORE_FACADE_PATH=<before>/crates/facade AB_AFTER_FACADE_PATH=<after>/crates/facade \
  bash run_ab_eager_release_cpu.sh 4158-<label>
```

- 診断経路のセルは `--diag-legacy-tape-reuse`（旧条件: ホスト入力・Tape 経路で結果を溜め続ける）で測り、`gemmdiag`／`traindiag` の JSONL に記録する
- 専有ゲートは round ごとに load1 < 1.0 と GPU compute プロセス 0 件（ログに `compute_procs=`）。不通過は record-only
- precheck は両腕の `size_gate_max_bytes=33554432` を assert する
- 開発機の疎通確認は `AB_PRECHECK=report-only`（正式系列にならない）
