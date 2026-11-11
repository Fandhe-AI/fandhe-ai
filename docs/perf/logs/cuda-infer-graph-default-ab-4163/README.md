# CUDA Graph を推論で既定にした場合の影響評価（イシュー #4163）

**状態: PR-1 時点。計測部品のみで実測なし。GB10 上の動作はすべて未検証。** 記録 doc は `docs/perf/infer-chain-graphcapture-cuda-ab.md` §6。

#4167 は ADOPT 入力がないため既定経路への結線を見送った（記録は doc §7）。PR-2・PR-3 は未着。

## 目的

推論チェーンの Graph capture（opt-in の `FANDHE_AI_CUDA_GRAPH_INFER`）を既定にしたら、推論・学習の双方にどう効くかを事前登録の A/B で測る。結果は #4167（既定化の結線または見送り。所有者承認）の判断材料にする。`INFER_GRAPH_DEFAULT_ENABLED` は本イシューでは変えない。

## 読み方の注意

- 本番の G-CUDA-INF reuse セル（常駐入力）は capture に届かない。効果は stream 変更だけ。capture まで含む効果は診断腕 reuse-host（`--diag-host-input-reuse`）で見る。
- checksum 一致は bit 同一性の証明ではない（R2 で補う）。単一機体・batch 3 種・1 スレッド。

## ファイル

| ファイル | 内容 |
|---|---|
| `RULE.template.txt` | 事前登録規則の雛形（プレースホルダ付き） |
| `RULE.txt` | （PR-2 で作る。測る前に単独コミット） |
| `run_gates_4163.sh` | 前提ゲート R0〜R3（R3 は `r3_repeats` 回の反復とダンプテストの単独実行） |
| `default-on.patch` | R3 専用の計測パッチ（const を true にし、コンパイル時表明テストを反転に合わせる）。crates へはコミットしない |
| `default-off-assumption-tests.txt` | 「既定 OFF 前提」のテストの事前リスト（R3 の失敗分類に使う） |
| `aggregate_4163.py` | 厳格パースの集計と判定（`--self-test`・`--check-template`・`--check-rule`） |
| `scripts/bench/framework-compare/run_ab_infer_graph_default_cuda.sh` | A/B ランナー |

## 手順

1. PR-2: PR-1 マージ後の main で `RULE.template.txt` から `RULE.txt` を作り（`fixed-at-utc`・`source_commit`・各 sha256・`r3_repeats` を埋める）、`python3 -I aggregate_4163.py --check-rule RULE.txt` を通して単独コミットする。
2. PR-3: GB10（`<cuda-node>`）で実機ロックを取り、compute プロセス 0 を確認し、`HW_LOCK_HELD=1 HW_LOCK_OWNER_FILE=<abs> AB_FACADE_PATH=<abs>/crates/facade bash run_ab_infer_graph_default_cuda.sh <label>`。
3. `python3 -I aggregate_4163.py --dir scripts/bench/framework-compare/results/raw --rule RULE.txt --label <label>` で `judge-4163.md` を作る。ホスト名・パスをマスクして記録する。

## R3 の作り直し（固定前の試走で見つかった点。#4163 所有者判断 2026-11-11）

RULE を固定する前の捨て試走で、R3 の集計が入力不正で止まった。原因は次の 3 つで、いずれも RULE 固定前に直した。

| 原因 | 対処 |
|---|---|
| `crates/backend-cuda/src/jit_cache_regression_tests.rs` の `multiprocess_concurrent_store_absorbs_rename_race` が自分を子プロセスで起動し直し、子の標準出力がそのまま親のログに混ざる | 子の stdout・stderr を pipe で捕捉し、失敗時だけ assert のメッセージに出す（テストの意味と製品コードは変えない） |
| facade の `tests/cpu_sme_gate_bit_dump.rs::dump_cpu_sme_gate_bits` が標準出力へ直接書き、`test <name> ... out[...]` の 1 行になる | 本体の実行から `--exact --skip` で外し、同じ回の中で `--exact` で単独に流す。ダンプ本体の行はログから落として件数だけ残し、結果は集計行から取って R3 の集合に含める（テストは変えない） |
| edition 2024 の merged doctests が 1 つの `Doc-tests` 節に結果行を 2 本出す | `aggregate_4163.py` は Doc-tests 節に限り結果行が複数あってよいとする（Running 節は 1 本のまま） |

あわせて R3 を反復にした。腕 C・A を `r3_repeats`（N。RULE の機械可読ブロック。所有者が決める）回ずつ流し、腕 A で N 回とも失敗し腕 C で N 回とも成功したテストだけを「阻害」（事前リストに載れば「既定 OFF 前提」）とする。回ごとに結果が混ざったテストは「不安定」として `judge-4163.md` に記録し、判定には使わない。腕 B（補助・判定外）は 1 回のまま。

ログは `r3-<腕>-<単位>-rep<k>-<label>.log/.tsv`（単位は `fandhe-ai-backend-cuda`・`fandhe-ai`・`smedump`）。テスト側の修正が対象差分に入るため、RULE.txt の対象差分は「差分あり（テストのみ）」になる。

## 既定 OFF 前提テストの走査

`git grep -nE "INFER_GRAPH_DEFAULT_ENABLED|infer_graph_stats|infer_graph_enabled|infer_graph_stream_requested|StreamKind::Legacy|is_capturable_stream|GRAPH_INFER" -- crates` で `crates/*/tests` と `graph.rs` の単体テストを走査し、既定値が OFF であること・Legacy stream であることを表明するものだけを `default-off-assumption-tests.txt` に載せる。

## 結果

（未実測）
