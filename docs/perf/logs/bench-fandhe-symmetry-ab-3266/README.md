# bench-fandhe 是正前後の併記計測（イシュー #3266）

#3265（PR #3566）で bench-fandhe に入れた是正（(1) GEMM reuse の反復ごとの `Tape::reset`、(3) readout の借用ビュー既定化、(6) train reuse の入力 `var_no_grad`）の前後を、相手フレームワークと同じラウンドで測り、旧条件・新条件の判定を並べる。規則は計測前に固定した `RULE.txt`。ADOPT／REJECT の二値判定は行わない（腕比は参考表示）。

## 状態

**実測は未実施。** 本ディレクトリには規則（`RULE.txt`）と手順だけがある。ランナー `scripts/bench/framework-compare/run_ab_bench_fandhe_symmetry.sh` と集計器 `judge_bench_fandhe_symmetry_ab.py`（テスト付き）は用意済みで、CI が回すのは集計器のテストだけである。Mac mini（Apple M4）と GB10 での 5 ラウンドの結果は、計測後にこの節と `judge-*.md`・`docs/framework-compare-harness-decision.md` §7.8 へ追記する。

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
