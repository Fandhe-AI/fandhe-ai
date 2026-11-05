# 実機 parity run スクリプトの雛形（イシュー #4044）

実機（`<cuda-node>`・`<metal-node>`）で `#[ignore]` の parity テストを測るときの run スクリプトの雛形。**これは雛形であり、実機では未実行**である。
共有 `CARGO_TARGET_DIR`（`$HOME/work/target-fandhe-ai`）の workspace メンバー成果物が前の回のものの再利用にならないことを、スクリプトが fail-closed で確かめる。
背景・再現・Phase 10 の棚卸しは [`docs/real-hardware-shared-target-reuse-decision.md`](../../../real-hardware-shared-target-reuse-decision.md)、手順は
[`docs/real-hardware-verification-env.md`](../../../real-hardware-verification-env.md) §4.11。

## 何が入っているか

- `run_parity_template.sh`: `../metal-batch-norm-2e20-4031-rerun2/run_metal_parity.sh`（回収物。編集しない）の R1〜R5 を一般化したもの。次を直した
  - 成果物の名前を手書き glob でなく `cargo metadata --no-deps` のメンバーの lib ターゲット名から作る。rerun2 の `libfandhe_ai_*` は facade 自身の lib（`libfandhe_ai-<hash>.rlib`）に一致しなかった
  - プロファイルを `PROFILE`（`dev`／`release`）で切り替える。Phase 10 の facade テストは `--release` なし（`target/debug/deps`）、#4031 は `--release`
  - テストを複数本（`TESTS`）にできる。1 本ずつ別プロセスで実行し、`--list` の 1 件確認は全テスト分、実行したバイナリと確かめたバイナリの照合は各テストで行う

## 使い方

1. `docs/perf/logs/<名前>-<issue 番号>/run_<cuda|metal>_parity.sh` へ複製する（1 本の自己完結ファイルのまま。補助ファイルを source しない）
2. 冒頭の「固定値」を埋める。`__FILL__` や空のままだと `--self-test` 以外では開始前に exit 2 で止まる

   | 変数 | 内容 |
   |---|---|
   | `NODE_KIND` | `cuda` または `metal` |
   | `PROFILE` | `dev`（`--release` なし）または `release` |
   | `PACKAGE`・`TEST_TARGET`・`TESTS` | 測るパッケージ・テストターゲット・テスト名（`--exact`） |
   | `LOG_BASE` | 回収ログ名の接頭辞 |
   | `EXPECTED_SOURCE_FP` | `RULE.txt` の `source_fingerprint`（`git archive` から計算した値） |
   | `REQUIRED_LIBS` | 必ず存在すべき lib 名。facade を測るなら `fandhe_ai`（facade 自身の lib）を入れる。メンバーの lib 名でなければ停止する |
   | `EXTERNAL_TIMEOUT` | `cuda` のみ。`timeout(1)` の秒数 |

3. `RULE.txt` に `build_freshness:` ブロックを写す（`../create-graph-gpu-parity-template/RULE.txt`・`../train-gpu-parity-template/RULE.txt` の末尾にある）。実機での実行より前にコミットする
4. 送信側が実機ロックを取って転送した後、ノード上で `CARGO_TARGET_DIR=$HOME/work/target-fandhe-ai bash run_<...>_parity.sh <tree> <out> "<lock-owner>"` を呼ぶ
5. 回収物に `build_artifacts.txt`（成果物の更新時刻とロック取得時刻の照合）・`clean.log`（clean したメンバーの一覧）が含まれることを確かめる

## 検証状況

| 項目 | 状況 |
|---|---|
| `bash -n`・`--self-test`（Linux、bash 5） | pass |
| 固定値が未記入のままの通常起動 | exit 2 で停止することを確認 |
| 通し実行（Linux、`NODE_KIND=cuda`・`PROFILE=dev`、tensor-core に足した使い捨ての `#[ignore]` テスト 1 本。他の cargo 検査だけ無効化した複製で、実行後に削除） | pass。`clean.log`・`build_artifacts.txt`・`summary.tsv`（verdict=pass）が回収され、ロックが解放された |
| macOS の bash 3.2 での `--self-test`・通し実行 | **未検証**（Linux からは確かめられない）。初回の `<metal-node>` での実行時に、実行前に `--self-test` を通して確かめる |
| 実機（GB10・Mac mini）での実行 | 未実施 |

## 既知の限界

- 時刻検査は「成果物がロック取得より後に書かれた」ことを確かめる。内容が現在のソースと対応することまでは、`EXPECTED_SOURCE_FP`（ソースの指紋）と clean の組で担保する
- メンバー以外（crates.io の依存）の成果物は clean しない。依存の再利用は意図どおり（`Cargo.lock` で固定された版）
