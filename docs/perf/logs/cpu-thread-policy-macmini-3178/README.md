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
| `macmini/{big-core,small-cap}/` | 実測ログ（マスク済み）。JSONL（fandhe before／after・torch）・`precheck-*`・`env_info-*`・`load_gate-*`・`uptime-*`・`compare-*`・`sha-*`・`tree-*`・`build-*.err`・`skipped-*`・`gate_diff.txt`（after の差分）・`run_ab-<gate>.log`／`.rc`。`bench-fandhe-*` のバイナリと `target*` は収録しない |
| `macmini/rev-stamp.txt`・`macmini/preflight.txt` | 計測に使った commit（4 クローン共通）と、ssh 疎通（`uname -a`）・sysctl の記録 |
| `aggregate-big-core.md`・`aggregate-small-cap.md` | `python3 -I aggregate.py macmini/<gate> 3178-<gate>` の出力そのまま（手で直さない） |

## 実行手順（Mac mini 実機。ホスト名と手順は `docs/real-hardware-verification-env.md` §7・Git 管理外の `.local.md`）

1. ローカルで `git bundle create <tmp>/3178.bundle origin/main` を作り rsync で送る。Mac mini 上で、**ゲートごとに** before／after の 2 クローンを作る
   （`git init` → `git fetch <bundle> refs/remotes/origin/main` → `git checkout --detach FETCH_HEAD`。ref が remote-tracking のため `git clone` では空リポジトリになる）。
   差分検証に git checkout が必要なため `.rev-stamp` 方式の展開ツリーは使わない。ゲートごとにクローン対を分ける理由は、スクリプトの出力（`results/raw/`・`compare-*.md`・`target-ab-*`）が
   gitignore 対象外で、同じ before ツリーで 2 本目を走らせると未追跡ファイル検査（`verify_no_untracked_build_inputs`）で止まるため。
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

**実測済み**（2026-10-21。Mac mini M4・P4＋E6・論理 10。2 ゲートとも専有ゲート 5/5 round `gate_ok=1`・`series=official`・機構発火の前提成立・checksum 全一致）。
before の commit は `3637b4af283f72bde163fb5b799603b856247ef7`（`macmini/rev-stamp.txt`）。総合判定は 2 ゲートとも **REJECT**（big-core は 6 セル、small-cap は 3 セルで中央値比 > 1.00）。
結果と考察は [`docs/perf/cpu-thread-policy-macmini.md`](../../cpu-thread-policy-macmini.md)。

実施メモ（規則・閾値は変更していない）:

- `AB_LOAD_GATE`・`AB_THREADPOLICY_PRECHECK` は設定しない既定（閾値 3.0・precheck=assert）の正式系列。big-core を先、small-cap を後に直列実行した。
- small-cap の brand は計測前の実測値（`Apple M4`）が暫定値と一致したため patch は無修正。
- torch venv は自専用に新規作成（torch 2.14.0・numpy 2.5.4。`env_info-*` の `torch=` 行）。
- マスク: 実ホスト名は `masked`、`$HOME` は `<home>`、ユーザー名は `<user>`、作業ディレクトリは `<work>` に置換した（収録前に grep で 0 件を確認）。
