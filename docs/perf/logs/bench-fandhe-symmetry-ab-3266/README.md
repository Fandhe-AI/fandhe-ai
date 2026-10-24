# bench-fandhe 是正前後の併記計測（イシュー #3266）

#3265（PR #3566）で bench-fandhe に入れた是正（(1) GEMM reuse の反復ごとの `Tape::reset`、(3) readout の借用ビュー既定化、(6) train reuse の入力 `var_no_grad`）の前後を、相手フレームワークと同じラウンドで測り、旧条件・新条件の判定を並べる。規則は計測前に固定した `RULE.txt`。ADOPT／REJECT の二値判定は行わない（腕比は参考表示）。

## 状態

**実測済み（Mac mini 系列 A・系列 B・GB10 の 3 系列、いずれも 5/5 ラウンド完走）。** 集計器の終了コードは 3 系列とも 0（全セル判定可能・hard 条件は全セルで両腕とも一致）。結果の表は集計器の出力そのもの（手書きの表は作らない）。

| 系列 | 機体 | ゲート | 集計器の出力 | 終了コード |
|---|---|---|---|---|
| series-a | Mac mini（Apple M4） | なし | `judge-3266-series-a.md` | 0（`.exit`） |
| series-b | Mac mini（Apple M4） | load1 < 3.0（5 ラウンドとも通過。load1 2.42〜2.80） | `judge-3266-series-b.md` | 0 |
| gb10 | GB10（NVIDIA GB10） | load1 < 1.0 かつ GPU 使用率 0%（5 ラウンドとも通過。load1 0.70〜0.92） | `judge-3266-gb10.md` | 0 |

- 正式系列: Mac mini は系列 B（5/5 でゲートを通過したため。RULE.txt の規則どおり）。系列 A は併記の参考。
- 各ラウンドの行数は 3 系列とも期待どおり（Mac mini は old 26／new 26／相手 22／Python 32、GB10 は 28／28／28／28）。想定外の skipped は無し（`skipped.log`）。
- hard 条件（checksum・parity_total・fail_count・max_abs/rel_err・scaled_abs_bound/rescued の run 間・腕間一致）は全セル OK。tolerance・baseline・判定規則は変更していない。
- 本計測は性能の ADOPT／REJECT を判定しない。表の腕比 new/old は参考値。

### 実施メモ

- ラベル: `series-a`・`series-b`・`gb10`。runner の SHA は `39db29dba3ede799738defdc6b7e7f5b2607129b`（RULE.txt を登録したコミット）。`rev-stamp-{old,new,runner}.txt`（各系列のディレクトリ）が展開ツリーの `.rev-stamp` で、`env_info.txt` の `rev_stamp_*` と一致する。
- 実機ロック（`~/work/.fandhe-hw-lock`）: 両機体とも 2026-10-24T02:18Z（ノード時計の UTC）に取得し、同 03:01Z に解放した。取得から解放まで、転送・ビルド・計測・回収はすべてロックの保持中に行った。
- 展開ツリーは `scripts/bench/framework-compare`・`bench_py.py`（runner には RULE のディレクトリも）だけを `git archive` で作った（腕間の比較・ビルドに使う範囲）。専用ディレクトリ `~/work/ab-3266/` へ転送し、計測後にそのディレクトリ（`target-*`・venv を含む）を削除した。
- Python 参照環境: Mac mini・GB10 とも既存の参照用 venv が見つからなかったため、0.11.0 の記録の版に合わせて計測用に新規構築した。Mac mini は Python 3.12.15・torch 2.14.0・scipy 1.18.1・numpy 2.5.4、別 venv で tensorflow 2.16.2・tensorflow-metal 1.2.0・numpy 1.26.4・keras 3.15.1（0.11.0 と同版）。GB10 は Python 3.12.3・torch 2.14.0+cu130・tensorflow 2.21.0（CPU のみ）・scipy 1.18.1（0.11.0 と同版）。版は各 `env_info.txt` にも記録されている。
- GB10 は計測前に `nvidia-smi` で GPU を占有するプロセスが無いこと（`gb10/gpu_apps_launch.txt`）と使用率 0% を確認した。
- マスク: ランナーの出力に含まれるホームの絶対パス（`tree-*.txt`・`build-*.err`）を `<home>` に置換した。ホスト名・ユーザー名・venv の絶対パスは成果物に現れない（`env_info.txt` は `host=masked`）。JSONL の数値・`rule_sha256`・`rev_stamp_*`・`bin_sha256_*` は置換していない。マスク後のコピーで集計器を回し直し、マスク前の出力とバイト単位で一致することを確認した。
- 収録していないもの: `target-*`・bench バイナリ・`err.tmp`。

### ノイズ帯（系列 A と系列 B で区分が反転したセル）

- 新条件腕: gemm metal N=512、infer metal
- 旧条件腕: gemm metal N=512
- 上記以外の Mac mini のセルは両系列で同じ区分。GB10 は 1 系列のみ。数値は各 `judge-3266-*.md` を参照。

## 腕

| 腕 | SHA | 内容 |
|---|---|---|
| old | `61c69352d4b465c53c5265a1eb8adbd2afb2297f` | #3265 直前 |
| new | `40453506ae65c81f8f065853c50eda9caa7ae459` | #3265 |

両腕とも registry の `fandhe-ai =0.11.0`。腕間で変わるのは `bench-fandhe/src/main.rs` と README だけである。Metal の fresh 行（gemm・infer）の readout も (3) で変わるため、#3265 の「fresh は変更しない」は Metal には当てはまらない。fresh 行は「参考・Metal は変化あり」として読む。

## 実行手順

1. 実機ロックを取る（`mkdir ~/work/.fandhe-hw-lock` → owner を書く → 終了後に `rm -rf`。取れなければ 60 秒おきに取り直す。ロックなしで測った値は使わない）。ホスト名は `METAL_NODE`・`CUDA_NODE` の変数で扱い、接続情報は `docs/real-hardware-verification-env.local.md`（コミットしない）から読む。
2. old・new・runner（RULE を含むコミット）の 3 ツリーを `git archive` で展開し、各ツリー直下に `.rev-stamp`（フル SHA）を置く。runner には `docs/perf/logs/lowlayer-diagnosis-2026-09-12/bench_py.py` も含める。`.git`・`*.local.md`・`.env*` は転送しない。
3. 各機体で次を実行する。Mac mini は `zsh -lc` で nohup 起動（setsid なし）、GB10 は `setsid nohup`・PATH 明示・`~/work/ab-3266/` の専用ディレクトリ。計測中はビルドを並走させない。
   - Mac mini 系列 A: `AB_PROFILE=macmini AB_GATE=0 ... bash run_ab_bench_fandhe_symmetry.sh series-a`
   - Mac mini 系列 B: `AB_PROFILE=macmini AB_GATE=1 ... bash run_ab_bench_fandhe_symmetry.sh series-b`
   - GB10: `AB_PROFILE=gb10 AB_GATE=1 ... bash run_ab_bench_fandhe_symmetry.sh gb10`
4. `results/raw/ab3266-<label>/` を回収し、実ホスト名・ホームの絶対パス・venv のパスをマスクして `macmini-series-a/`・`macmini-series-b/`・`gb10/` へ置く。`env_info.txt` と各ツリーの `.rev-stamp` も含める。
5. `judge_bench_fandhe_symmetry_ab.py --profile ... --series-dir ... --env-info ... --rule RULE.txt [--require-gate --gate-log ...] --ref-dir <0.11.0 の run 群>` の出力を `judge-*.md`・`.exit` として保存する。終了コードは 0 が全セル判定可能、2 が undetermined あり、3 が hard 条件の不一致あり。手書きの表は作らない。

0.11.0 の旧値の出典: Mac mini は `docs/perf/logs/framework-compare-0.11.0-macmini-remeasure/`、GB10 は `docs/perf/logs/framework-compare-0.11.0-remeasure/gb10/`。
