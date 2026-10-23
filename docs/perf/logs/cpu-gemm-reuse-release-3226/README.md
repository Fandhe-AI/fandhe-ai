# cpu-gemm-reuse-release-3226

イシュー #3226（reuse が fresh より遅い逆転の是正）の実測記録置き場。設計・結果の本文は
`docs/perf/cpu-gemm-reuse-fresh-inversion.md` §8。

## 収録物

| パス | 内容 |
|---|---|
| `RULE.txt` | 事前登録規則（計測より前に固定。事後に緩めない） |
| `on-arm.patch` | after 腕の 1 行差分（`EAGER_MATMUL_RELEASE_DEFAULT_ENABLED` を `true` へ） |
| `diag-gb10/run1.jsonl` | GB10 での補助 diag（1 run・record-only・判定外） |
| `gb10/`・`macmini/` | 正式 A/B 系列の生ログ（マスク済み）。JSONL・compare・judge・load_gate・env_info・eager-release-report・tree・sha・uptime・build.err・skipped・実行ログ。ELF と `target*` は収録しない |
| `gb10/minflt-run.txt` | GB10 の minflt 補助 run（`cpu_gemm_reuse_release_phases`・判定外） |
| `judge-3226-combined.md` | 両機体をまたぐ最終判定（`judge_eager_release_ab.py --require-both` の出力そのまま） |
| `rev-stamp-verification-3226.md` | `.rev-stamp`・ツリー指紋・`rule_sha256` の突合 |

## 実行手順

ツール: `scripts/bench/framework-compare/run_ab_eager_release_cpu.sh`（A/B）・`judge_eager_release_ab.py`（判定）。
機体は GB10 と Mac mini（ホスト名・手順は `docs/real-hardware-verification-env.md`・Git 管理外の `.local.md`）。

1. ローカルで計測 commit の `git bundle` を作り、ノードへ rsync する。ノード上で runner・before・after の 3 クローンを作る
   （`git init` → `git fetch <bundle> <ref>` → `git checkout --detach FETCH_HEAD`）。各ツリーに `.rev-stamp`
   （1 行目 HEAD の sha・2 行目 `dirty=0`）を置き `.git/info/exclude` へ追記する。after のみ `git apply on-arm.patch`。
2. `RAYON_NUM_THREADS` を unset し、他の重い処理を止める。GB10 は `PATH` に `~/.cargo/bin` を明示し `setsid nohup`、
   Mac mini は `zsh -lc` 経由で nohup サブシェルから起動して生存を確認する。
3. runner の `scripts/bench/framework-compare/` で実行する。

   ```sh
   AB_PROFILE=gb10|macmini AB_BEFORE_FACADE_PATH=<before>/crates/facade AB_AFTER_FACADE_PATH=<after>/crates/facade \
     bash run_ab_eager_release_cpu.sh 3226-<label>
   ```

4. `results/raw/` の 3226 系ファイル・`compare-*.md`・`judge-3226-*.md` をマスク（ホスト名は `masked`、`$HOME` は `<home>`、ユーザー名は `<user>`）して `gb10/`・`macmini/` へ収録する。
5. 両機体の JSONL を同時に渡して最終判定を出す。

   ```sh
   python3 -I judge_eager_release_ab.py --require-both \
     --gb10-before <gemm> <train> <infer> --gb10-after <gemm> <train> <infer> \
     --macmini-before <gemm> <train> <infer> --macmini-after <gemm> <train> <infer>
   ```

## 状態

実測済み（2026-10-23）。計測 sha は `3c5549027edd58c15917432add08dfd2d34f9d6d`。両機体とも 5/5 round が `gate_ok=1`・
`series=official`・precheck（before は未解放・after は解放）通過・checksum 全一致。総合判定は **REJECT**
（GB10 の gemm reuse N=4096 と train reuse が 5/5 round で後退）。既定は OFF のまま。結果は
`docs/perf/cpu-gemm-reuse-fresh-inversion.md` §8.4。

実施メモ:

- GB10 の最初の起動（label `3226-gb10`）は非対話 SSH で `cargo` が `PATH` に無く、precheck の時点（計測前）で停止した。
  計測は 1 回も行われていない。同ラベルは空の JSONL が残り再利用できないため、`PATH` を明示して label `3226-gb10-r2` で
  起動し直した（以降の系列はこの 1 本のみ。差し替え・追加起動なし）。
- 計測ツールの初版 commit（`ab8f74b37f86`）は `read_gate` の参照先が誤っており両機体でビルド前に停止した。修正して
  amend した commit（上記計測 sha）で全て計測し直している（それ以前に計測値は無い）。
- マスク後に `spark`・`mac-mini`・`/home/`・`/Users/` が 0 件であることを grep で確認した。
