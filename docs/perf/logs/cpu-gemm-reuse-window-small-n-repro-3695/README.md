# #3695: N=512 の vafter と tunmmap を別セッションで再現確認し分離する

状態: **実測済み**（2026-10-29・`<cuda-node>` の専有 GB10・3 系列 s1〜s3・ゲート不通過なし。全体判定は「再現しない（N=512 の差が再現しない）」）。
判定規則は `RULE.txt` が正（実測と部品より前に単独コミットで固定。固定後は変えない）。
経緯と読解は `docs/perf/cpu-gemm-reuse-fresh-inversion.md` §13.7。#3680（`../cpu-gemm-reuse-window-small-n-3680/`）の続き。

## 構成

| ファイル | 内容 |
|---|---|
| `RULE.txt` | 事前登録規則（単独コミット） |
| `session-scripts/session-3695-excl.sh` | 1 系列 = 1 セッション（hw-lock・30 秒 poller・診断 5 run・bench 4 腕・env_info・専有判定）。引数は系列ラベル `s1`〜`s5`（allowlist） |
| `judge_3695.py` | 複数系列の判定器（標準ライブラリのみ・fail-closed・`--self-test`）。`judge_3680.py` と #3671 のローダーを編集せず流用する |

流用して編集しない部品: `../cpu-gemm-reuse-window-small-n-3680/build_bench_head.sh`・`../cpu-gemm-reuse-window-3671/orchestrate.sh`・`../cpu-gemm-resident-reuse-3646/excl_check.py`。

## 腕（bench 側。同一バイナリ。差は CLI フラグと環境変数だけ）

| 腕（ディレクトリ名） | 内容 |
|---|---|
| `base` | 変更なし |
| `vafter` | reuse／phases にだけ `--diag-verify-after-loop` |
| `tunmmap` | 3 モードすべてに `GLIBC_TUNABLES=glibc.malloc.mmap_threshold=33554432` |
| `vafter+tunmmap`（`vafter_tunmmap`） | 上の環境変数（3 モード）と `--diag-verify-after-loop`（reuse／phases）の両方 |

## 実行手順（実機に接続できるセッションで）

1. 計測木を `git archive` で展開し、ルートに `.rev-stamp`（1 行目 HEAD の sha・2 行目 `dirty=0`）を置く。HEAD は RULE.txt を含むこと。`$HOME/work/r3695-excl/runner` に rsync する（openrsync の `--delete-excluded` で framework-compare の target が消えうるため、転送後に先にビルドが通ることを確かめる）
2. hw-lock の**前に**ビルドを済ませる（計測中のビルド並走は禁止）:
   - `build_bench_head.sh <木の絶対パス> $HOME/work/r3695-excl/bench-bin`（`Cargo.lock` は trap で戻る）
   - 診断テストバイナリ: 木で `cargo test --release --locked -p fandhe-ai --test cpu_gemm_reuse_window_diag --no-run`
3. 系列ごとに切り離して起動する: `setsid nohup bash <木>/docs/perf/logs/cpu-gemm-reuse-window-small-n-repro-3695/session-scripts/session-3695-excl.sh s1 >/dev/null 2>&1 &`。完了は `out-s1/session-scripts/steps.log` の `^done\. rc=` で確かめる。`hwlock released` から 30 分以上あけて `s2`、同様に `s3`。ゲートを通らない系列があれば `s4`・`s5` まで（最初の 3 系列を判定に使う）
4. 常駐プロセスのあるノードは使わず、常駐を止めない。専有機体が埋まっていれば待つ
5. 回収時にホスト名・`/home`・ユーザー名をマスクし、`gb10-excl/<label>/` に置く
6. `python3 -I judge_3695.py --series-root gb10-excl --series s1,s2,s3[,s4,s5] > judge-3695.md`（起動順。手で並べ替えない）

所要時間の目安: **推定であり実測ではない**。#3680 は 5 腕で 1 系列約 2 時間 17 分（`steps.log`）だった。本系列は 4 腕なので 1 系列 2 時間前後、3 系列と間隔 2 回で 7〜8 時間以上を見込む。

## 検証済み／未検証

| 項目 | 状態 |
|---|---|
| `judge_3695.py --self-test`（全体判定の各行・ゲート不通過の系列の飛ばし・系列間隔・RULE sha・バイナリ sha・checksum・データ欠損） | 検証済み |
| `shellcheck`／`bash -n`（セッションスクリプト）、ラベル不正・`RAYON_NUM_THREADS` 設定時の拒否 | 検証済み |
| `session-3695-excl.sh` の実機実行・`build_bench_head.sh` の実ビルド | 検証済み（s1〜s3 とも `done. rc=0`。起動し直した系列・ゲート不通過の系列はなし） |
| GB10 の 3 系列の実測・判定 | 検証済み（`judge-3695.md`。マスク前後で全行一致） |
| 別機体での再現・`GLIBC_TUNABLES` の閾値スイープ | 未実測（対象外） |

## 成果物

- `gb10-excl/s1`〜`s3/`: マスク済みの回収データ（`gb10/`〈診断 5 run〉・`bench/<腕>/`・`session-scripts/`・`.rev-stamp`）
- `judge-3695.md`: `judge_3695.py --series-root gb10-excl --series s1,s2,s3` の標準出力そのまま
- `rev-stamp-verification-3695.md`: 木・バイナリ・実行物・専有ゲート・時刻表の同定記録

結果の転記と読み方は `docs/perf/cpu-gemm-reuse-fresh-inversion.md` §13.7。
