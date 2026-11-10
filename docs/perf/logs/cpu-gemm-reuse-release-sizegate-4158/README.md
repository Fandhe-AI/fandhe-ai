# cpu-gemm-reuse-release-sizegate-4158

イシュー #4158（eager release にサイズゲートを付けて GB10 で A/B する）の記録置き場。
設計・結果の本文は `docs/perf/cpu-gemm-reuse-fresh-inversion.md` §18。

状態: **実測済み・判定 REJECT（2026-11-10）**。`RULE.txt`（#4224 で単独コミット）に従い <cuda-node> で 5 round を実測した（全 round ゲート通過）。
処置 4 セルはすべて ADOPT 側の条件を満たした（中央値 0.505〜0.798・5/5 round で < 1.00）が、ガードの既定経路 gemm N=4096 reuse
（構造上 A/A）が 5/5 round で後退（1.004〜1.011・中央値 1.007）したため、RULE の判定式どおり REJECT。既定は OFF のまま。
詳細は `judge-4158.md`・`docs/perf/cpu-gemm-reuse-fresh-inversion.md` §18.4。

## 収録物

| パス | 内容 |
|---|---|
| `on-arm.patch` | after 腕の 1 行差分（`EAGER_MATMUL_RELEASE_DEFAULT_ENABLED` を `true` へ）。サイズゲート閾値は両腕で共通 |
| `RULE.txt` | 事前登録規則（#4224 で計測前に単独コミット） |
| `gb10/` | 実測の生データ（ラベル `4158-gb10`。JSONL・ゲートログ・env_info・precheck・ビルドログ・バイナリ sha256。ELF は収録しない。ホーム・ホスト名・ユーザー名はマスク済み） |
| `judge-4158.md` | `judge_eager_release_ab.py --rule-set 4158` の出力（逐語） |
| `rev-stamp-verification-4158.md` | `.rev-stamp`・ツリー指紋・RULE の sha256 の突合 |

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
