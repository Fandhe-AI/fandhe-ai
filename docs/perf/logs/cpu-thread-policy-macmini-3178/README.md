# Mac mini の CPU スレッド方針 A/B ログ（イシュー #3178）

2 ゲート（`BIG_CORE_LIMIT_ENABLED`・`SMALL_SHAPE_CAP_ENABLED`）を個別に反転した A/B の事前登録規則・patch・集計ツール・実行手順。
結論は [`docs/perf/cpu-thread-policy-macmini.md`](../../cpu-thread-policy-macmini.md)。

## 収録物

| ファイル | 内容 |
|---|---|
| `RULE.txt` | 事前登録規則（計測前に固定。過去の REJECT を覆さない再測定であること・条件差・判定 4 値） |
| `on-arm-big-core.patch` | after 腕（`thread_limit.rs` の 1 行） |
| `on-arm-small-cap.patch` | after 腕（`small_shape_thread_cap.rs` の 2 行。brand は暫定 `Apple M4`、計測前に実測値へ合わせる） |
| `aggregate.py` | 集計（after/before 比・PyTorch 比・checksum 一致。`--self-test` あり） |
| `macmini/` | 実測ログ（マスク済み）。**未実測のため未生成** |

## 実行手順（Mac mini 実機。ホスト名と手順は `docs/real-hardware-verification-env.md` §7・Git 管理外の `.local.md`）

1. ローカルで `git bundle create <tmp>/3178.bundle <branch>` を作り rsync で送る。Mac mini 上で `git clone <bundle> before`・`git clone <bundle> after` を作る
   （差分検証に git checkout が必要なため、`.rev-stamp` 方式の展開ツリーは使わない）。
2. ゲートごとに after へ `git apply on-arm-<gate>.patch` を当てる（small-cap は先に `sysctl -n machdep.cpu.brand_string` を読み、patch の brand を実測値へ合わせる。P コア 4・論理 10 でなければ前提不成立として RULE に記録する）。
3. `RAYON_NUM_THREADS` を unset し、他の重い処理を止める。macOS には `setsid` がないので `nohup` のサブシェルで `zsh -lc` 経由で起動し、`ps` で生存を確認する。
4. `scripts/bench/framework-compare/` で、big-core を先に、small-cap を後に実行する。

   ```sh
   AB_GATE=big-core AB_BEFORE_FACADE_PATH=<before>/crates/facade AB_AFTER_FACADE_PATH=<after>/crates/facade \
     AB_TORCH_PYTHON=<venv-torch>/bin/python bash run_ab_thread_policy_macmini_cpu.sh 3178-big-core
   ```

5. `results/raw/` の 3178 系ファイル（JSONL・precheck・env_info・load_gate・uptime・compare 表・sha・tree）をマスクして `macmini/` へ収録する。
   `python3 -I aggregate.py <results/raw> 3178-<gate>` の出力を `aggregate.md` として保存する。

## 状態

本 PR を作った実行環境から Mac mini へ到達できなかったため、計測は**未実施**（判定は undetermined）。
基盤（スクリプト・プリチェックテスト・`--sizes threadpolicy`・集計ツール）は M4 Max 開発機で `bash -n`・単体テスト・プリチェックテストの report 出力まで確認した。
