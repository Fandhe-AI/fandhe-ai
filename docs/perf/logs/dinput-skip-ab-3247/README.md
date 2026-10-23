# d_input 省略（#3246）の前後 A/B（イシュー #3247）

`Tape::backward_device_param_store` 経路で入力ノードが `var_no_grad` のとき `Op::LinearResident` VJP の d_input GEMM を省く変更（#3246・PR #3559）を、Mac mini（CPU・Metal）と GB10（CPU・CUDA）の train セルで A/B 計測するための事前登録と道具一式。

## 状態

- **実測済み（2026-10-23）**。8 セルすべてを計測し、judge の終了コードは 0（処置セル 4 つ = train reuse がすべて ADOPT・ガードセル 4 つ = train fresh がすべて guard-pass）。checksum・PARAM_DIGEST は全 run・両腕で一致。要約と読み方の注意は `docs/autodiff-nograd-leaf-dinput-skip-decision.md`「実測記録（#3247）」。
- 計測値は Apple M4 の Mac mini と GB10 のもの。M4 Max（開発機）の値は含まない。

## 収録物

| ファイル | 内容 |
|---|---|
| `RULE.txt` | 事前登録規則（計測前に固定。変更しない） |
| `bench-var-no-grad.patch` | 計測専用パッチ（コミットしない一時適用。reuse の入力を `var_no_grad` にし、`BENCH_PARAM_DIGEST=1` で更新後パラメータの FNV-1a 64 を stderr に出す）。bench の入力切り替えのコミットは #3265 の担当 |
| `scripts/bench/framework-compare/run_ab_dinput_skip_train.sh` | 両機体共用ランナー（`AB_PROFILE=macmini` または `gb10`） |
| `scripts/bench/framework-compare/judge_dinput_skip_ab.py`（と `_test.py`） | RULE を fail-closed に適用する判定 |
| `macmini/`・`gb10/` | マスク済み生ログ（JSONL・digest・gate・env_info・tree・sha・compare md・phases） |
| `judge-3247-combined.md` | judge の出力そのまま（終了コード併記） |
| `rev-stamp-verification-3247.md` | `.rev-stamp`・RULE・パッチ・バイナリの sha256 の突き合わせ |

## 実行手順（計測セッション向け）

1. 実機ロック（`~/work/.fandhe-hw-lock`）を取り、`docs/real-hardware-verification-env.md` §7 に従う。
2. 3 ツリーを作る: runner（本 RULE を含むコミット）・before（`b6579c3f5`）・after（`4fb2a39c8`）。各ツリーに `.rev-stamp`（1 行目 HEAD sha・2 行目 `dirty=0`）を置き `.git/info/exclude` へ追加。runner だけパッチを `git apply` する。
3. `AB_PROFILE=<macmini|gb10> AB_BEFORE_FACADE_PATH=<before>/crates/facade AB_AFTER_FACADE_PATH=<after>/crates/facade [AB_ROUND_COOLDOWN_MAX_LOAD1=3.0] bash scripts/bench/framework-compare/run_ab_dinput_skip_train.sh <label>`（GB10 は `setsid nohup`、Mac mini は nohup サブシェル）。
4. after ツリーで、機体ごとにテスト名を絞って実行しログを保存（`_on_cuda` は `cfg` で除外されないため、ファイル全体に `--ignored` を付けると Mac 上で失敗する）。Mac mini: `cargo test -p fandhe-ai --release --test device_param_store_nograd_input_dinput_skip -- --ignored --exact nograd_input_dinput_skip_is_bit_identical_on_metal --nocapture --test-threads=1`、GB10: 同じ形で `nograd_input_dinput_skip_is_bit_identical_on_cuda`。計測のビルドと並走させないため、ランナーの完了後に実行する。
5. 生ログを回収し、ホスト名・`$HOME`・ユーザー名をマスクして `macmini/`・`gb10/` へ置く。
6. `python3 -I scripts/bench/framework-compare/judge_dinput_skip_ab.py --rule RULE.txt --machine macmini=<dir>:<label> --machine gb10=<dir>:<label> --require-both --ref macmini=<0.11.0 再計測 median5 JSONL> ...` を実行し `judge-3247-combined.md` に保存。
7. 決定記録 `docs/autodiff-nograd-leaf-dinput-skip-decision.md` へ「実測記録（#3247）」節を追記し、`docs/README.md` を更新。

## 実施メモ

- ラベルは `3247-macmini`・`3247-gb10`。起動し直し（`-r2`）はない。run の差し替え・追加起動・round の追加・セルの除外はしていない。
- runner は `a5fa15fd3`。ノードは外部の Git ホスティングに届かないため、ローカルで作った bundle を rsync で送り、ノード上で bundle からの fetch により 3 ツリーを作った。計測パッチは runner だけに当てた。
- 実機ロック: GB10 は 2026-10-23T17:36:06Z 取得・17:43:48Z 解放、Mac mini は 17:38:36Z 取得・17:47:58Z 解放（ノード時計の UTC）。計測と実機テストはロック保持中に行った。
- Mac mini の load1 は計測中 2.43〜2.59（ゲート 3.0 未満で全 round pass）。OS のプロセス `mediaanalysisd` が CPU を占有していた。GB10 CPU は load1 0.86〜0.97（ゲート 1.0 未満）で、ゲートぎりぎりだった。
- 収録時のマスク: ホスト名は env_info の `host=masked`（ランナーが出力）、ホーム配下の絶対パスは `<home>` に置換した（10 ファイル。tree・build・実機テストのログ）。`rule_sha256`・`patch_sha256`・`rev_stamp_head_*`・`bin_sha256_*` と JSONL・digest の数値は置換していない。マスク後に judge を流し直した結果が上の判定出力である。ホスト名・ホームのパス・ユーザー名の grep は 0 件。
- 収録していないもの: bench バイナリ（`bench-fandhe-*`）、`target*`、`.rev-stamp` 本体（値は `rev-stamp-verification-3247.md`）。
