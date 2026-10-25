# 常駐チェーンの学習・推論の計測（イシュー #3257・参考系列）

## 状態

- **実機計測（Mac mini・GB10）は 2026-10-24 に実施済み**。両機体とも 5 round・全 8 セルが判定可能（`judge-3257.md`、終了コード 0）。結果表・比・残差の分解は決定記録 `docs/compat-resident-train-step-decision.md` §17.1〜§17.4。
- 判定の要旨: IMPROVE 4（Mac mini train Metal・GB10 infer CPU・train CUDA・infer CUDA）、NEUTRAL 2（Mac mini infer CPU・infer Metal）、REGRESS 2（CPU train が両機体。+0.4%・+5.9%）。INVALID／REJECT なし。
- 正式な 6 フレームワーク同席計測は #3273（0.12.0 のピン更新後）。

## 収録物

| ファイル | 内容 |
|---|---|
| `RULE.txt` | 事前登録規則（計測前に固定） |
| `bench-resident-chain.patch` | 計測専用パッチ（`bench-fandhe/src/main.rs`・`bench-common/src/lib.rs`。runner ツリーにだけ適用し、コミットしない） |
| `scripts/bench/framework-compare/run_ab_resident_chain.sh` | 両機体共用ランナー（1 ツリー・1 ビルド・4 腕・5 round） |
| `scripts/bench/framework-compare/judge_resident_chain_ab.py` | RULE を fail-closed に適用する判定（`judge_resident_chain_ab_test.py` は CI で実行） |

## 準備でのローカル確認（開発機 M4 Max。数値は記録に使わない）

- パッチは `origin/main` に `git apply --check` が通り、`--config` の path patch で `cargo build --release -p bench-fandhe` が通る（Cargo.lock は復元）。
- CPU の煙試験で、train は reuse・chain の loss checksum（0.080541）と PARAM_DIGEST（`0x7d9d7aefaeaf43d5`）が一致、infer は checksum（13.976574）が一致した。

## 実施メモ（2026-10-24）

- runner: `786d18b50`（#3584）。以降の main の差分は skills 更新のみ。ランナーは実 `.git` を前提にするため、bundle を各ノードの専用ディレクトリへ転送して clone し、`.rev-stamp`（`dirty=0`）を作ってパッチを当てた（rsync の `--exclude .git` や archive 方式では precheck で落ちる）。GitHub 認証はノードに置いていない。
- 機体ロック: 両ノードで `#3257 2026-10-24T04:28:52Z` を owner に取得。計測は 04:29〜04:36Z で完了した。torch は専用 venv（Mac mini 2.14.0・GB10 2.14.0+cu130）。版は各 `torch-version.txt`（`env_info` では `torch_python=masked` になるため別に保存）。
- マスク: `build-*.err`・`tree-*.txt`・`sha-*.txt`・`run-*.log` のホームの絶対パスを `<home>` に置換した。JSONL の数値・`rule_sha256`・`patch_sha256`・`rev_stamp_head_runner`・`bin_sha256`・PARAM_DIGEST は置換していない。
- 収録しないもの: bench バイナリ（`bench-fandhe-*`・`bench-candle-*`）、`target-ab-chain-*`、`.local.md`・`.env*`。
- 回収は 2026-10-25。ノード上の `ab-3257/`（target・venv を含む）を削除し、`#3257` 自身が残した古いロック（4 時間超・該当プロセスなし）を外した。
- 計測時の `nvidia-smi` の記録（`gpu_apps_launch.txt`）は取れていない。

## 実行手順

1. runner ツリー（本 RULE を含むコミット。実施時は bundle で clone）を実機へ置き、`.rev-stamp`（1 行目 HEAD・2 行目 `dirty=0`）を作り `.git/info/exclude` へ足し、パッチを `git apply` する。
2. `docs/real-hardware-verification-env.md` §7 に従い機体ロックを取る（`mkdir ~/work/.fandhe-hw-lock`）。
3. `AB_PROFILE=macmini|gb10 AB_FACADE_PATH=<runner>/crates/facade AB_TORCH_PY=<python> AB_BENCH_PY=<bench_py.py> bash scripts/bench/framework-compare/run_ab_resident_chain.sh <label>`
   （GB10 は `setsid nohup`、Mac mini は nohup サブシェルで切り離す）。
4. 生ログを回収し、ホスト名・`$HOME`・ユーザー名をマスクして `macmini/`・`gb10/` に置く（sha・digest・JSONL の数値は置換しない）。
5. `python3 -I scripts/bench/framework-compare/judge_resident_chain_ab.py --rule RULE.txt --machine macmini=<dir>:<label> --machine gb10=<dir>:<label> --require-both` の出力を終了コードつきで保存する。
