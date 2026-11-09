# CUDA Graph を推論で既定にした場合の影響評価（イシュー #4163）

**状態: PR-1 時点。計測部品のみで実測なし。GB10 上の動作はすべて未検証。** 記録 doc は `docs/perf/infer-chain-graphcapture-cuda-ab.md` §6。

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
| `run_gates_4163.sh` | 前提ゲート R0〜R3 |
| `default-on.patch` | R3 専用の計測パッチ（const を true にし、コンパイル時表明テストを反転に合わせる）。crates へはコミットしない |
| `default-off-assumption-tests.txt` | 「既定 OFF 前提」のテストの事前リスト（R3 の失敗分類に使う） |
| `aggregate_4163.py` | 厳格パースの集計と判定（`--self-test`・`--check-template`・`--check-rule`） |
| `scripts/bench/framework-compare/run_ab_infer_graph_default_cuda.sh` | A/B ランナー |

## 手順

1. PR-2: PR-1 マージ後の main で `RULE.template.txt` から `RULE.txt` を作り（`fixed-at-utc`・`tree_sha`・各 sha256 を埋める）、`python3 -I aggregate_4163.py --check-rule RULE.txt` を通して単独コミットする。
2. PR-3: GB10（`<cuda-node>`）で実機ロックを取り、compute プロセス 0 を確認し、`HW_LOCK_HELD=1 HW_LOCK_OWNER_FILE=<abs> AB_FACADE_PATH=<abs>/crates/facade bash run_ab_infer_graph_default_cuda.sh <label>`。
3. `python3 -I aggregate_4163.py --dir scripts/bench/framework-compare/results/raw --rule RULE.txt --label <label>` で `judge-4163.md` を作る。ホスト名・パスをマスクして記録する。

## 既定 OFF 前提テストの走査

`git grep -nE "INFER_GRAPH_DEFAULT_ENABLED|infer_graph_stats|infer_graph_enabled|infer_graph_stream_requested|StreamKind::Legacy|is_capturable_stream|GRAPH_INFER" -- crates` で `crates/*/tests` と `graph.rs` の単体テストを走査し、既定値が OFF であること・Legacy stream であることを表明するものだけを `default-off-assumption-tests.txt` に載せる。

## 結果

（未実測）
