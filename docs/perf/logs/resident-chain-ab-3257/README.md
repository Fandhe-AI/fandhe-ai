# 常駐チェーンの学習・推論の計測（イシュー #3257・参考系列）

## 状態

- **実機計測（Mac mini・GB10）は未実施**。本ディレクトリは計測基盤（事前登録規則・計測専用パッチ・ランナー・判定ツール）までを収録する。
  結果表・比・残差の分解は計測後に決定記録 §17 へ追記する（未取得の値を推定で埋めない）。
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

## 実行手順

1. runner ツリー（本 RULE を含むコミット）を実機へ置き、`.rev-stamp`（1 行目 HEAD・2 行目 `dirty=0`）を作り `.git/info/exclude` へ足し、パッチを `git apply` する。
2. `docs/real-hardware-verification-env.md` §7 に従い機体ロックを取る（`mkdir ~/work/.fandhe-hw-lock`）。
3. `AB_PROFILE=macmini|gb10 AB_FACADE_PATH=<runner>/crates/facade AB_TORCH_PY=<python> AB_BENCH_PY=<bench_py.py> bash scripts/bench/framework-compare/run_ab_resident_chain.sh <label>`
   （GB10 は `setsid nohup`、Mac mini は nohup サブシェルで切り離す）。
4. 生ログを回収し、ホスト名・`$HOME`・ユーザー名をマスクして `macmini/`・`gb10/` に置く（sha・digest・JSONL の数値は置換しない）。
5. `python3 -I scripts/bench/framework-compare/judge_resident_chain_ab.py --rule RULE.txt --machine macmini=<dir>:<label> --machine gb10=<dir>:<label> --require-both` の出力を終了コードつきで保存する。
