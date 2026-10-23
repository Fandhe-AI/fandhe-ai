# d_input 省略（#3246）の前後 A/B（イシュー #3247）

`Tape::backward_device_param_store` 経路で入力ノードが `var_no_grad` のとき `Op::LinearResident` VJP の d_input GEMM を省く変更（#3246・PR #3559）を、Mac mini（CPU・Metal）と GB10（CPU・CUDA）の train セルで A/B 計測するための事前登録と道具一式。

## 状態

- **実測は未実施**（Mac mini・GB10 の実機ロック取得と実行は別セッションで行う）。8 セルすべて未計測で、判定は出していない。
- 計測値・M4 Max（開発機）の値はここに書かない。

## 収録物

| ファイル | 内容 |
|---|---|
| `RULE.txt` | 事前登録規則（計測前に固定。変更しない） |
| `bench-var-no-grad.patch` | 計測専用パッチ（コミットしない一時適用。reuse の入力を `var_no_grad` にし、`BENCH_PARAM_DIGEST=1` で更新後パラメータの FNV-1a 64 を stderr に出す）。bench の入力切り替えのコミットは #3265 の担当 |
| `scripts/bench/framework-compare/run_ab_dinput_skip_train.sh` | 両機体共用ランナー（`AB_PROFILE=macmini` または `gb10`） |
| `scripts/bench/framework-compare/judge_dinput_skip_ab.py`（と `_test.py`） | RULE を fail-closed に適用する判定 |
| `macmini/`・`gb10/`（計測後に追加） | マスク済み生ログ（JSONL・digest・gate・env_info・tree・sha・compare md・phases） |
| `judge-3247-combined.md`（計測後に追加） | judge の出力そのまま |

## 実行手順（計測セッション向け）

1. 実機ロック（`~/work/.fandhe-hw-lock`）を取り、`docs/real-hardware-verification-env.md` §7 に従う。
2. 3 ツリーを作る: runner（本 RULE を含むコミット）・before（`b6579c3f5`）・after（`4fb2a39c8`）。各ツリーに `.rev-stamp`（1 行目 HEAD sha・2 行目 `dirty=0`）を置き `.git/info/exclude` へ追加。runner だけパッチを `git apply` する。
3. `AB_PROFILE=<macmini|gb10> AB_BEFORE_FACADE_PATH=<before>/crates/facade AB_AFTER_FACADE_PATH=<after>/crates/facade [AB_ROUND_COOLDOWN_MAX_LOAD1=3.0] bash scripts/bench/framework-compare/run_ab_dinput_skip_train.sh <label>`（GB10 は `setsid nohup`、Mac mini は nohup サブシェル）。
4. after ツリーで `cargo test -p fandhe-ai --release --test device_param_store_nograd_input_dinput_skip -- --ignored --nocapture --test-threads=1` を実行しログを保存。
5. 生ログを回収し、ホスト名・`$HOME`・ユーザー名をマスクして `macmini/`・`gb10/` へ置く。
6. `python3 -I scripts/bench/framework-compare/judge_dinput_skip_ab.py --rule RULE.txt --machine macmini=<dir>:<label> --machine gb10=<dir>:<label> --require-both --ref macmini=<0.11.0 再計測 median5 JSONL> ...` を実行し `judge-3247-combined.md` に保存。
7. 決定記録 `docs/autodiff-nograd-leaf-dinput-skip-decision.md` へ「実測記録（#3247）」節を追記し、`docs/README.md` を更新。
