# optin-ab-gb10-3190

イシュー #3190（未計測の opt-in 候補を GB10 で A/B する）の補遺規則・patch・補助ツール・一部の生ログ。
状態: **B1〜B6 は GB10 で実測済み**（B5 と B4 の有効系列は再開分・2026-10-21。B5 は不採用）。B3 の再計測は #3205 へ切り出し。結論は `docs/perf/optin-ab-gb10.md`。

時刻の注記: ログのタイムスタンプは UTC、`RULE.txt` の `fixed-at: 2026-10-21` は JST（日付表記のみ）。規則を固定したコミット `230937f5f` の作者日時は 2026-10-20T15:20:29Z で、最初の計測ログ（B1 の負荷ゲート 15:22:50Z）より前。

## ファイル

| ファイル | 役割 |
|---|---|
| `RULE.txt` | 補遺の事前登録規則（計測前に単独コミットで固定。継承 RULE 6 件の sha256 を記録） |
| `RULE-reopen.txt` | 再開分の事前登録（B4 の再計測・B5 の具体化・条件 4 の機体別併記。元 RULE と継承 RULE の sha256 を記録。コミット `fbf6ecbbe` で固定） |
| `RULE-reopen-2.txt`・`RULE-reopen-3.txt` | B5 前提ゲート (a) のコマンド訂正（`--features internal-diagnostics`・`--no-fail-fast`）。いずれも再実行の前にコミット |
| `on-boundary-predicate.patch` | B5 head 腕（`MMA_F16` の境界述語の事前計算。計測専用。`230937f5f` に当たる） |
| `aggregate_b5.py` | B5 の判定集計（`python3 -I`。`--self-test` あり。前提ゲート (a)(b)・TFLOPS・fail-closed） |
| `b5/` | B5 の生ログ（`parity-{base,head}.log`・旧 (a) の `parity-attempt{1,2}-*.log`・`ptxas-*`・`run{1..5}-{base,head}.log`・`gate.log`・`env_info.txt`・`rev-stamp.txt`・`preflight-procs.txt`・`aggregate.md`） |
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
| B4 | `../cuda-gemm-small-launch-cost-2109/gb10-r2/`（**有効系列**・`aggregate.py` exit 0）。旧 `gb10/`（無効系列）・`gb10-normalized-reference/`（参考集計）は履歴 |
| B5 | `b5/`（本ディレクトリ） |
| B6 | `readback-n256/`（label `head-d89a093-3190-n256`） |

## 実行手順（GB10。実施した順）

1. `git bundle`（`origin/main` + 本ブランチ）を送り、ノード上で `before`／`after`／`runner` の 3 clone を作る。`.rev-stamp` は未追跡ファイルとして
   継承スクリプトの検査に引っかかるため clone 内には置かず、`git rev-parse HEAD` を同定に使う。スクリプトは `runner` から実行し、`before`／`after` は clean に保つ。
2. CPU 系（load1 < 8.0）: B1 `run_ab_tape_arena_cpu.sh 2104-gb10-3190` → B2 Stage 0 `run_reduction_sweep.sh gb10 <abs>` → 集計 → Stage 1 `AB_THRESHOLD_T=16384 run_ab_reduction_threshold_cpu.sh 2102-gb10-3190`。
3. CUDA 系（専有ゲート）: B3 R0→R1→R2→`run_ab_infer_graph_cuda.sh`、B4 `LOG_DIR=<空> RUN_NSYS=1 ./orchestrate.sh`（`PATH` に `/usr/local/cuda/bin` が要る）、
   B6 `AB_READBACK_SIZES=n256 run_ab_readback_reuse_cuda.sh`（先に `readback_reuse_bit_match_2108` を `--all-features --test-threads=1` で実行）。
4. 回収したログを、置換スクリプト（ホスト名・ユーザー名がリテラルになるため収録しない）でマスクしてから収録（ホスト名 → `masked`、`$HOME` → `<home>`、ELF は除外）。

## 再開分の手順（セッション 3。`session-scripts/` の `setup3.sh`・`cuda_session3_stage*.sh`・`steps-session3.log`）

1. 事前登録（`RULE-reopen.txt`・`aggregate.py` の修正・`aggregate_b5.py`・patch）を GB10 のログより前にコミットする。
2. `git bundle` を新しい作業ディレクトリ `r3190-reopen/` へ送り、`runner`／`b5-base`（`230937f5f`）／`b5-head`（`230937f5f` + patch。未コミット）の 3 clone を作る（`setup3.sh`）。`CARGO_TARGET_DIR` は export しない。
3. 段 1: B4 `LOG_DIR=<空> RUN_NSYS=1 ./orchestrate.sh` → B5 (a) parity → (b) ptxas。(a) のコマンドは 2 回訂正した（段 1b・段 1c。下記）。
4. 段 2: 前提ゲート成立後に、gemm_mma_bench を 5 round（奇数 round は base→head、偶数 round は head→base）。各プロセスの前に専有ゲート。
5. 回収したログをマスク（ホスト名 → `masked`、`$HOME` → `<home>`）してから収録し、`aggregate.py`／`aggregate_b5.py` を実行する。

## 作業中の事故（正直な記録）

- 第 1 CUDA セッション: `CARGO_TARGET_DIR` を export したため、スクリプトの `bench-fandhe` 探索が壊れ、B3 A/B（61 run が起動不能）・B4・B6 A/B が計測ゼロで失敗した。
  結果は一度も見ていない。B3 は別 label `2115-gb10-3190b`、B4 は別の空ディレクトリ、B6 は同 label で再実行した（B6 は第 1 回に出力が生成されていない）。
- 第 2 セッションの後、成果物の掃除で `git clean` を使い、B3（`3190b`）と B6 の compare 表・JSONL・ゲートログを消した。標準出力ログに全 run の JSON 行が残っていたため、
  `tools/recon.py` で無加工に再構成して同じ `compare_gemm_ab.py` コマンドで表を再生成した（`*.reconstructed.*`）。B3 のゲートログは復元できない
  （通過の証跡は、不成立なら計測せず終了する設計と計測の実行）。
- B4 の `nvcc` が PATH に無く第 1 回の `orchestrate.sh` が失敗した（`LOG_DIR` に `load_gate.log` のみ生成）。PATH を足して別の空ディレクトリで再実行した。
- 再開分（2026-10-21）: B5 前提ゲート (a) の登録コマンドが 2 回実行不能だった。1 回目は `specialized_mma_parity` が `internal-diagnostics` feature を要求し cargo に拒否された
  （integration テストが 1 件も実行されなかった。旧ログ `b5/parity-attempt1-*.log`）。2 回目は最初の失敗バイナリで cargo が停止した（base・head 双方の
  `mma_f16_k4096_stress` の FAILED。旧ログ `parity-attempt2-*.log`）。どちらも結果を見る前に判定式を変えず、コマンドの訂正だけを再実行の前にコミットした
  （`RULE-reopen-2.txt`・`RULE-reopen-3.txt`）。1 回目の後に `aggregate_b5.py` へ「必須テストバイナリの実行」検査を足した（空の成立を防ぐ追加で、判定は厳しくなる方向のみ）。
  また (b) の ptxas 結果は、RULE-reopen-2 のコミット時点で目視済みだった（(a) の判定とは独立）。
- 再開分: `aggregate_b5.py` の test 行の正規表現は、実ログの `ignored, <理由>` 形式を読めず 1 回 fail-closed で止まった。形式の認識だけを直した（判定ロジックは不変。`--self-test` に追加）。
- 再開分: 2101 の RULE が「CUDA reduce 系 `--ignored` parity の同セッション実施」を求めるという前版の記述は、RULE の全文を検索して該当文言が無かったため取り下げた
  （詳細は `docs/perf/optin-ab-gb10.md` 申し送りの訂正）。
