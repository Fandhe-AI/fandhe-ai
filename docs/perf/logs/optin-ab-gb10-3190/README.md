# optin-ab-gb10-3190

イシュー #3190（未計測の opt-in 候補を GB10 で A/B する）の補遺規則・patch・補助ツール・一部の生ログ。
状態: **B1〜B4・B6 は GB10 で実測済み、B5 は未実施**。結論は `docs/perf/optin-ab-gb10.md`。

時刻の注記: ログのタイムスタンプは UTC、`RULE.txt` の `fixed-at: 2026-10-21` は JST（日付表記のみ）。規則を固定したコミット `230937f5f` の作者日時は 2026-10-20T15:20:29Z で、最初の計測ログ（B1 の負荷ゲート 15:22:50Z）より前。

## ファイル

| ファイル | 役割 |
|---|---|
| `RULE.txt` | 補遺の事前登録規則（計測前に単独コミットで固定。継承 RULE 6 件の sha256 を記録） |
| `on-host-arena.patch` | B1 after 腕（`HOST_ARENA_DEFAULT_ENABLED` を true。#3184 と同一内容） |
| `on-reduction-gb10.patch` | B2 Stage 1 after 腕（ゲート true・`REDUCTION_PARALLEL_MIN_ELEMS = 16384`。Stage 0 の T_gb10 を記入済み） |
| `aggregate_sweep_single.py` | B2 Stage 0 の GB10 単機集計（`python3 -I`。`--self-test` あり。継承の `aggregate_sweep.py` を import して再利用） |
| `readback-n256/` | B6 の生ログ（標準出力・bit 一致テスト）と再構成した JSONL・compare 表 |
| `session-scripts/` | GB10 上で実行したセッションスクリプトと各ステップの終了状態（`steps-session*.log`） |
| `tools/` | `recon.py`（標準出力から JSONL を無加工で再構成）・`normalize_b4.py`（B4 の融合行を分離した参考コピー作成） |

## 生ログの置き場（GB10 は継承 RULE が指定する機体なので、継承ディレクトリへ入れる）

| ゲート | label／置き場 |
|---|---|
| B1 | `../tape-arena-reuse-2104/*2104-gb10-3190*` |
| B2 Stage 0 | `../elemental-reduction-threshold-2101/gb10/`（`aggregate.md` は GB10 単機） |
| B2 Stage 1 | `../elemental-reduction-ab-2102/gb10/*2102-gb10-3190*` |
| B3 | `../cuda-infer-chain-graphcapture-2115/gb10/`（第 1 系列は `invalid-attempt1-2115-gb10-3190/`、判定系列は label `2115-gb10-3190b`） |
| B4 | `../cuda-gemm-small-launch-cost-2109/gb10/`（無効系列の生ログ）・`gb10-normalized-reference/`（参考集計） |
| B6 | `readback-n256/`（label `head-d89a093-3190-n256`） |

## 実行手順（GB10。実施した順）

1. `git bundle`（`origin/main` + 本ブランチ）を送り、ノード上で `before`／`after`／`runner` の 3 clone を作る。`.rev-stamp` は未追跡ファイルとして
   継承スクリプトの検査に引っかかるため clone 内には置かず、`git rev-parse HEAD` を同定に使う。スクリプトは `runner` から実行し、`before`／`after` は clean に保つ。
2. CPU 系（load1 < 8.0）: B1 `run_ab_tape_arena_cpu.sh 2104-gb10-3190` → B2 Stage 0 `run_reduction_sweep.sh gb10 <abs>` → 集計 → Stage 1 `AB_THRESHOLD_T=16384 run_ab_reduction_threshold_cpu.sh 2102-gb10-3190`。
3. CUDA 系（専有ゲート）: B3 R0→R1→R2→`run_ab_infer_graph_cuda.sh`、B4 `LOG_DIR=<空> RUN_NSYS=1 ./orchestrate.sh`（`PATH` に `/usr/local/cuda/bin` が要る）、
   B6 `AB_READBACK_SIZES=n256 run_ab_readback_reuse_cuda.sh`（先に `readback_reuse_bit_match_2108` を `--all-features --test-threads=1` で実行）。
4. 回収したログを、置換スクリプト（ホスト名・ユーザー名がリテラルになるため収録しない）でマスクしてから収録（ホスト名 → `masked`、`$HOME` → `<home>`、ELF は除外）。

## 作業中の事故（正直な記録）

- 第 1 CUDA セッション: `CARGO_TARGET_DIR` を export したため、スクリプトの `bench-fandhe` 探索が壊れ、B3 A/B（61 run が起動不能）・B4・B6 A/B が計測ゼロで失敗した。
  結果は一度も見ていない。B3 は別 label `2115-gb10-3190b`、B4 は別の空ディレクトリ、B6 は同 label で再実行した（B6 は第 1 回に出力が生成されていない）。
- 第 2 セッションの後、成果物の掃除で `git clean` を使い、B3（`3190b`）と B6 の compare 表・JSONL・ゲートログを消した。標準出力ログに全 run の JSON 行が残っていたため、
  `tools/recon.py` で無加工に再構成して同じ `compare_gemm_ab.py` コマンドで表を再生成した（`*.reconstructed.*`）。B3 のゲートログは復元できない
  （通過の証跡は、不成立なら計測せず終了する設計と計測の実行）。
- B4 の `nvcc` が PATH に無く第 1 回の `orchestrate.sh` が失敗した（`LOG_DIR` に `load_gate.log` のみ生成）。PATH を足して別の空ディレクトリで再実行した。
- 2101 の RULE が求める CUDA reduce 系 `--ignored` parity の同セッション実施は行っていない。
