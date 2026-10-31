# 0.13.0 の 6 フレームワーク同席計測（GB10）

**状態: 実測済み（2026-10-31。GB10 5 ラウンド・Mac mini 系列 B 5 run）。** crates.io `fandhe-ai =0.13.0`（2026-10-31 公開・release-all publish run 41168270894・タグ `v0.13.0` = `db2b385fb`）で framework-compare スコアボードを再計測した。事前登録規則 `RULE.txt` は計測前の 2026-10-31T04:56:14Z に固定し、以後編集していない（RULE.v2.txt なし）。計測の最初の記録は GB10 のゲート 04:56:36Z、Mac mini の run1 開始 04:58:54Z で、いずれも固定の後。計測腕（PIN_SHA）は main の #3709 マージ `5d05a50235138645b0b069060ed54dfd2c46db7f`（`v0.13.0` ピン入り。タグ `v0.13.0` の後のコミット）。Mac mini 側の生ログは `../framework-compare-0.13.0-macmini-remeasure/`。Mac mini と GB10 の値は同じ表に混ぜていない（機体ごとに節を分ける）。計測・生成のみで、コミット・push・公開は本記録の範囲外（スコアボード HTML は未公開）。

## 目的

0.12.0 のスコアボード（`../framework-compare-0.12.0-remeasure/`）を、registry ピンだけ `=0.13.0` に上げて取り直す。0.12.0 のような旧条件腕（stage4・oldarm）は置かない単腕構成（0.10.0・0.11.0 と同じ）。判定規則は 0.12.0 から変えない（比 = 同一 run 内の最速の有効な相手 / fandhe-ai reuse の 5 run 中央値。1 超で 1 位・0.90 以上で僅差・相手側の parity fail は判定不能。判定側 fandhe-ai の要素検証不合格は生成停止）。「0.12.0 比」は同一セル・同一モードの fandhe-ai 中央値同士の比（日またぎで負荷の揺れを含む。表示のみで勝敗には使わない）。

## 0.12.0 版からの差分

| 項目 | 0.12.0 | 0.13.0 |
|---|---|---|
| 腕 | 新条件腕＋旧条件腕（stage4・28 行・swap） | 単腕（`ROOT` と `EXPECT_SHA` のみ） |
| 旧腕に付随する不変条件 | 共有ファイルの新旧 diff・旧腕に ARM_RE が無いことの検査・tree-old・OLD_ROOT・旧腕の事前ビルド | 削除。ARM_RE が計測ツリーにあることの検査は残した |
| 期待行数 | stage1 112・skipped 0・stage2 8・stage3 28・stage4 28 | stage1 112・skipped 0・stage2 8・stage3 28 |
| ピン検査 | `fandhe-ai v0\.12\.0$` | `fandhe-ai v0\.13\.0$` |
| Mac mini の系列 | A（対照）＋B | B のみ。ノイズ帯は「対照系列なし・未算出」（`noise na`。0 件ではない） |
| 比較欄 | 0.11.0 比 | 0.12.0 比（sha256 を RULE に固定。`run_0130.sh` が照合） |
| スコアボード | 3 条件（新・旧・0.11.0） | 2 条件（0.13.0・0.12.0） |
| 実測依存の文字列 | `gen_args.txt` に実測値を記入済み | `gen_args.txt` に実測値を記入済み（本計測で記入） |

## 受け入れ条件

| # | 要件 | 状態 | 根拠 |
|---|---|---|---|
| A1 | 全 run の tree が v0.13.0・rev-stamp が同一 | **達成** | GB10 5 run・Mac mini 5 run の計 10 run で `tree.txt` が `└── fandhe-ai v0.13.0`、`rev-stamp.txt` が `5d05a50235138645b0b069060ed54dfd2c46db7f`。`aggregate/check_identity.py` が両機体とも rc=0（行数 GB10 112／8／28、Mac mini 48／32 を含む） |
| A2 | 5 run 中央値と 0.12.0 比 | **達成** | `gb10/*-median5.jsonl`・Mac mini 側の同名ファイル・`scoreboard/gen_0130.out`。GB10 本体と Mac mini 本体の median5 は `median_rounds.py` で run1〜run5 から再生成して byte 一致を確認した |
| A3 | 全セルの比と 1.0 以下のセルの理由 | **達成（2 セルは分類保留）** | 全 27 セルの比は `scoreboard/gen_0130.out`。1.0 以下の 14 セルの理由は「1.0 以下のセルの理由」。M-MTL-TRN と G-CPU-INF は数字で 3 分類に割り当てられず「分類保留」（`scoreboard/unreached_0130.tsv`） |
| A4 | 事前登録規則を計測前に固定 | **達成（事前確認 1 は条件を満たさなかった。下の「事前確認」）** | `RULE.txt`（固定 2026-10-31T04:56:14Z）。固定後に規則・期待行数・tolerance は変えていない |
| A5 | 機体を混ぜない | **達成** | Mac mini は `macmini-run-0130.sh` が run ごとに `machdep.cpu.brand_string == "Apple M4"` を検査し、不一致なら run を止める（5/5 完走）。GB10 は 5 run の `env_info.txt` 先頭行が `NVIDIA GB10, 580.173.02` で一致。M4 Max の値・Linux の代替値は使っていない |

## 事前確認（計測前の確認項目と、本記録時点で確かめた結果）

- [ ] **bench-fandhe が 0.12.0 の新条件腕 `bdaca14cc…` から変わっていないこと — 文言どおりの条件は満たさなかった（計測区間は同じ。下記）。** `git diff bdaca14cc 5d05a5023 -- scripts/bench/framework-compare` の実測（本記録時点）: `*.rs` の差は 2 ファイル（`bench-common/src/lib.rs`・`bench-fandhe/src/main.rs`）で、内容は #3685 の診断フラグ `--diag-verify-after-loop`（診断専用・既定 OFF。フラグなしでは `verify_strict` は従来どおりループ内に残り、行の出力も `diag` キーなし）と `emit_with_diag`・そのテスト、`VERSION` 定数とピン表記の更新。`bench-fandhe/Cargo.toml` の差は fandhe-ai の版の行とコメント内の版表記だけ。`Cargo.lock` の差は fandhe-ai 系 7 クレート（fandhe-ai・autodiff・backend-cpu・backend-cuda・backend-metal・onnx-interop・tensor-core）の version／checksum だけ。`deny.toml` はコメント 1 行。ほかにスクリプト・Python・README の差がある（`run_all.sh`・`run_all_cuda.sh` はコメントの版表記のみ）。本計測ではフラグを使っておらず（計測スクリプトに無く、全 run の結果行に `diag` キーなし）、計測区間への影響は diff で確認した: フラグ OFF（既定）のとき、`measure_gemm_reuse_resident` の計測ループ内の処理は `bdaca14cc` と同じで、増えたのは `elapsed` を記録した後（計測区間外）の `if verify_after_loop { … } else { verify_strict … }` の分岐と、ループ前の容量 0 の `Vec::with_capacity` だけ（`scripts/bench/framework-compare/bench-fandhe/src/main.rs` の `durations.push(elapsed)` 以降）。したがって「計測区間のコードは 0.12.0 の計測時と同じ」として扱う。ただし RULE.txt の事前確認の文言（`*.rs` の差 0 件）とは一致しないので、逸脱としてここに記録する。RULE.txt の事前確認の条件「`*.rs` の差 0 件」とは一致しない
- [x] PIN_SHA（計測ツリーの `.rev-stamp`）を決めて RULE.txt（両ディレクトリ）に記入: `5d05a50235138645b0b069060ed54dfd2c46db7f`（両 RULE.txt に記入済み。全 10 run の `rev-stamp.txt` が一致）
- [x] 行数スモーク（ビルド込み 1 round・値は判定に使わない）: GB10 stage1 112・skipped 0・stage2 8・stage3 28・rc 0（2026-10-31T04:51:35Z〜04:55:50Z）、Mac mini rows 48・pyrows 32・skipped 4（burn Metal GEMM 512〜4096 fresh のみ）・rc 0（04:49:42Z〜04:53:51Z）。`smoke/` に収録（RULE.txt にも記入済み）
- [x] 専有ノード（`<cuda-node>`）で compute プロセスが無いことを確認し、実機ロックを取る: 実機ロック owner `rel-0.13.0 2026-10-31T04:49:10Z`（05:35:42Z に解除）。全 5 run の `env_info.txt` の compute-apps は空
- [x] Python 3 FW の版が 0.12.0 と同じこと: JSONL の `version` 欄は 0.12.0 と一致した（GB10 `pytorch 2.14.0+cu130`・`tensorflow 2.21.0`・`scipy 1.18.1`、Mac mini `pytorch 2.14.0`・`tensorflow 2.16.2`・`scipy 1.18.1`）。計測後（2026-10-31T06 時台 UTC）に同じ venv の `pip freeze`（GB10 は `pip freeze`＋`python --version`、Mac mini は `uv pip freeze`）を取り `env/` に収録した。0.12.0 の `env/` と byte 一致（numpy・keras を含む全パッケージが同じ版）。

## 計測条件

| 項目 | GB10（`<cuda-node>`） | Mac mini（`<metal-node>`） |
|---|---|---|
| 機体 | NVIDIA GB10・driver 580.173.02（`env_info.txt`。5 run 一致） | Apple M4・10 コア・16 GB（計測専用機。0.12.0 と同じ機体） |
| 腕 | PIN_SHA `5d05a5023`（単腕）。`git archive` で展開し `.rev-stamp`（40 桁）を置いて転送 | 同左 |
| registry ピン | `fandhe-ai =0.13.0`・`candle-core =0.11.0`・`burn =0.21.0`（`tree.txt` で run ごとに検査） | 同左 |
| Python 3 FW | pytorch 2.14.0+cu130・tensorflow 2.21.0（CPU のみ）・scipy 1.18.1（JSONL の `version` 欄。0.12.0 と一致）。numpy・keras も 0.12.0 と同じ版（`env/` の pip freeze が byte 一致） | pytorch 2.14.0・tensorflow 2.16.2（Metal プラグイン）・scipy 1.18.1（同） |
| bench_py.py | sha256 `9fd2c08f…2409b`（0.12.0 と同値。スクリプトが run ごとに照合） | 同左 |
| 既定設定 | SME OFF・Graph capture OFF（既定のまま） | 同左 |
| 日時 | 2026-10-31 04:59:06Z〜05:33:28Z（5 ラウンド。run1 開始〜run5 終了） | 2026-10-31 04:58:54Z〜05:08:30Z（run1 開始〜run5 終了） |
| ゲート | 各ラウンド前に load1 < 1.0 かつ GPU 0% の 3 連続（`gb10/gate.log` 38 行）。ラウンド開始時 load1 = 0.82／0.56／0.33／0.36／0.63・GPU 0%。全ラウンドで 2.5〜4 分待機した（最初のサンプルは load1 3.33／7.71／5.94／5.04／5.59。run2 の最初のサンプルだけ GPU 1%。通過した 3 サンプルは全て GPU 0%）。`GATE-INVALID`・`GATE-TIMEOUT` なし | 系列 B は各 run 前に load1 < 3.0（`gate.log` 8 行。run2・run4・run5 の前に 4.21・3.15・3.10 で 1 分ずつ待機）。開始時 load1 = 1.39／2.57／2.35／2.00／1.77。系列 A（対照）は行わない |
| 完走 | `gb10/loop.log` `ALL-DONE completed=5/5`。skipped 0 件（`skipped-dgx-0.13.0.log` 5 本とも 0 バイト） | `loop-macmini.log` `ALL-DONE completed=5/5`。全 5 run の最終行 `done. rows=48 pyrows=32 skipped=4`（skipped は burn Metal GEMM N=512〜4096 fresh の既知の記録拒否 4 件だけ。0.12.0 と同一の文言） |
| 実機ロック | `~/work/.fandhe-hw-lock`（owner `rel-0.13.0 2026-10-31T04:49:10Z`）。05:35:42Z に解除 | 同左 |

- **判定不能 1 セル**: GB10 PyTorch CPU gemm N=4096 の 1 要素が複合判定を外れる（`parity_fail_count` 1／16777216。0.12.0 と同じ）。順位に含めない。fandhe-ai／candle の CUDA TF32 opt-in 行は f32 判定から外す（0.12.0 と同じ扱い）。
- **ノイズ帯**: Mac mini の系列 A（対照）を行っていないため算出しない（判定一覧の tally は `noise na`。0 件ではない）。GB10 は 0.12.0 も対照系列なし。

## 結果（同一 run 内比の 5 run 中央値。判定は `scoreboard/gen_0130.py`）

| 条件 | 1 位 | 僅差 | 負け | 判定不能 | ノイズ帯 | 計 |
|---|---|---|---|---|---|---|
| 0.13.0（正式値） | **13** | 3 | 11 | 1 | 未算出（対照系列なし） | 27 |
| 0.12.0（`../framework-compare-0.12.0-remeasure/scoreboard/gen_0120.out`） | 10 | 4 | 13 | 1 | 0 | 27 |

`gen_0130.out` の tally 行: `tally 13 3 11 invalid 1 noise na total 27`。

### 0.12.0 からの区分変化（5 セル）

| セル | 0.12.0 | 0.13.0 |
|---|---|---|
| GB10 gemm CPU N=512 | 2 位 0.68× | 1 位 1.14× |
| GB10 gemm CPU N=1024 | 2 位 0.87× | 1 位 1.25× |
| GB10 gemm CPU N=2048 | 僅差 0.92× | 1 位 1.74× |
| GB10 gemm CPU N=4096 | 僅差 0.96× | 1 位 1.20× |
| Mac mini train Metal | 1 位 1.01× | 僅差 0.95× |

- 1 位 10 → 13（GB10 gemm CPU の 4 セルが 1 位に、Mac mini train Metal が 1 位から僅差へ）、僅差 4 → 3、負け 13 → 11、判定不能 1 → 1。上記 5 セル以外の 22 セルは区分が変わらない（比の変化は最大 0.05。GB10 gemm CUDA N=4096 の 1.92→1.97）。
- Mac mini train Metal は fandhe-ai の 0.12.0 比が 1.00（変化なし）で、相手 PyTorch MPS の窓が縮んだ分だけ比が下がった（「1.0 以下のセルの理由」の M-MTL-TRN 参照）。
- GB10 gemm CPU の 4 セルの 1 位化は、reuse が fresh と並んだ 3 サイズ（N=512・2048・4096）と、reuse/fresh が 1.14 に下がった N=1024 に当たる（下の「GB10 CPU gemm の reuse／fresh」）。この変化と 0.12.0→0.13.0 の本番変更の対応は、本計測では示さない。

### セル表（`scoreboard/gen_0130.out`）

各列は「区分 比（vs 最速相手）」。比は同一 run 内比の 5 run 中央値。★ = 0.13.0 の比が 1.0 以下。「0.12.0 比」は fandhe-ai 中央値同士の時間比（推論はスループット比。<1 が高速化・推論は >1 が改善）。

#### Mac mini（M4）

| セル | 0.13.0 | 0.12.0 | 0.13.0 の run 別比 | 0.12.0 比（fandhe-ai 中央値同士） |
|---|---|---|---|---|
| gemm Metal N=256 | 1 位 1.15×（vs candle） | 1 位 1.16×（vs candle） | 1.12 1.09 1.15 1.15 1.16 | 1.00× |
| gemm Metal N=512 | 1 位 1.08×（vs candle） | 1 位 1.06×（vs candle） | 1.45 1.16 1.08 1.08 1.08 | 1.00× |
| gemm Metal N=1024 ★ | 僅差 0.99×（vs candle） | 僅差 0.99×（vs candle） | 1.00 0.99 1.02 0.98 0.99 | 0.98× |
| gemm Metal N=2048 ★ | 僅差 0.94×（vs candle） | 僅差 0.94×（vs candle） | 0.94 0.94 0.94 0.94 0.94 | 1.00× |
| gemm Metal N=4096 ★ | 2 位 0.88×（vs candle） | 2 位 0.85×（vs candle） | 0.94 0.86 0.88 0.89 0.85 | 0.96× |
| gemm CPU N=256 ★ | 3 位 0.71×（vs pytorch） | 3 位 0.70×（vs pytorch） | 0.68 0.70 0.71 0.72 0.71 | 0.99× |
| gemm CPU N=512 ★ | 2 位 0.76×（vs pytorch） | 2 位 0.77×（vs pytorch） | 0.74 0.76 0.77 0.76 0.75 | 0.99× |
| gemm CPU N=1024 ★ | 2 位 0.70×（vs pytorch） | 2 位 0.67×（vs pytorch） | 0.68 0.80 0.70 0.80 0.66 | 0.98× |
| gemm CPU N=2048 ★ | 2 位 0.52×（vs pytorch） | 2 位 0.53×（vs pytorch） | 0.52 0.52 0.54 0.54 0.52 | 0.97× |
| train Metal ★ | 僅差 0.95×（vs pytorch） | 1 位 1.01×（vs pytorch） | 0.95 0.95 0.96 0.95 0.96 | 1.00× |
| infer Metal | 1 位 1.02×（vs candle） | 1 位 1.02×（vs candle） | 1.03 1.02 0.91 1.03 0.98 | 1.01× |
| train CPU ★ | 3 位 0.28×（vs pytorch） | 3 位 0.25×（vs pytorch） | 0.27 0.28 0.27 0.28 0.28 | 0.98× |
| infer CPU ★ | 4 位 0.15×（vs pytorch） | 4 位 0.15×（vs pytorch） | 0.16 0.16 0.15 0.15 0.15 | 1.02× |

#### GB10

| セル | 0.13.0 | 0.12.0 | 0.13.0 の run 別比 | 0.12.0 比（fandhe-ai 中央値同士） |
|---|---|---|---|---|
| gemm CUDA N=256 | 1 位 1.16×（vs candle） | 1 位 1.17×（vs candle） | 1.14 1.19 1.16 1.18 1.16 | 1.00× |
| gemm CUDA N=512 | 1 位 1.14×（vs candle） | 1 位 1.12×（vs candle） | 1.15 1.14 1.15 1.13 0.99 | 0.99× |
| gemm CUDA N=1024 | 1 位 1.15×（vs candle） | 1 位 1.15×（vs candle） | 1.14 1.15 1.19 1.15 1.14 | 1.00× |
| gemm CUDA N=2048 | 1 位 1.13×（vs burn） | 1 位 1.13×（vs burn） | 1.13 1.14 1.13 1.13 1.13 | 1.00× |
| gemm CUDA N=4096 | 1 位 1.97×（vs burn） | 1 位 1.92×（vs burn） | 1.97 1.94 1.97 1.92 1.98 | 0.99× |
| gemm CPU N=256 ★ | 3 位 0.72×（vs pytorch） | 3 位 0.71×（vs pytorch） | 0.72 0.69 0.89 0.93 0.62 | 1.04× |
| gemm CPU N=512 | 1 位 1.14×（vs candle） | 2 位 0.68×（vs candle） | 1.15 1.14 0.99 0.93 1.43 | 0.76× |
| gemm CPU N=1024 | 1 位 1.25×（vs candle） | 2 位 0.87×（vs candle） | 1.21 1.25 1.21 1.40 1.57 | 0.70× |
| gemm CPU N=2048 | 1 位 1.74×（vs tensorflow） | 僅差 0.92×（vs tensorflow） | 1.68 1.77 1.74 1.71 1.84 | 0.54× |
| gemm CPU N=4096 | 1 位 1.20×（vs tensorflow） | 僅差 0.96×（vs tensorflow） | 1.22 1.21 1.20 1.17 1.19 | 0.79× |
| train CUDA | 1 位 1.99×（vs candle） | 1 位 2.01×（vs candle） | 1.99 1.99 2.00 1.99 1.44 | 1.01× |
| infer CUDA ★ | 3 位 0.51×（vs pytorch） | 3 位 0.51×（vs pytorch） | 0.51 0.51 0.53 0.51 0.52 | 0.99× |
| train CPU ★ | 2 位 0.52×（vs pytorch） | 2 位 0.55×（vs pytorch） | 0.52 0.55 0.56 0.48 0.42 | 0.98× |
| infer CPU ★ | 2 位 0.76×（vs pytorch） | 2 位 0.78×（vs pytorch） | 0.81 0.78 0.76 0.74 0.72 | 1.03× |

- 0.13.0: `tally 13 3 11 invalid 1 noise na total 27`
- 0.12.0: `tally 10 4 13 invalid 1 noise 0 total 27`

### GB10 CPU gemm の reuse／fresh（実測）

fandhe-ai の 5 run 中央値（ms）。0.13.0 は reuse が fresh より遅いのが N=256（1.91）と N=1024（1.14）で、N=512・2048・4096 は 0.99〜1.00 で並ぶ。0.12.0 は 5 サイズとも 1.23〜1.79 倍遅かった（`../framework-compare-0.12.0-remeasure/README.md`「GB10 CPU gemm の新旧差」の新腕）。

| N | 0.13.0 fresh | 0.13.0 reuse | 0.13.0 reuse/fresh | 0.12.0 fresh | 0.12.0 reuse | 0.12.0 reuse/fresh |
|---|---|---|---|---|---|---|
| 256 | 0.191 | 0.366 | 1.91 | 0.196 | 0.351 | 1.79 |
| 512 | 1.750 | 1.745 | 1.00 | 1.832 | 2.311 | 1.26 |
| 1024 | 3.895 | 4.443 | 1.14 | 4.641 | 6.344 | 1.37 |
| 2048 | 16.720 | 16.611 | 0.99 | 17.598 | 30.973 | 1.76 |
| 4096 | 121.661 | 121.973 | 1.00 | 125.016 | 153.687 | 1.23 |

- 0.12.0 から 0.13.0 で入った本番変更は `matmul_resident` の D2H 経路（#3677 の `download_owned` 追加・#3706 の出力 256 KiB 以下は copy／超は owned）。**本計測では区間の切り分けをしておらず、この変更と上表の変化の対応（効果の帰属）は本計測では示さない。** 変更時の A/B 記録は `docs/perf/logs/cpu-gemm-download-owned-size-branch-3694/judge-3694.md` と `docs/perf/cpu-gemm-reuse-fresh-inversion.md`。#3706 の A/B 判定は REJECT で、所有者判断により採用した（「改善を確認した」とは書かない）。
- N=256 の reuse が fresh より遅い後退は 0.13.0 でも残る（G-CPU-G256）。

### 1.0 以下のセルの理由（0.13.0・14 セル）

分類は `docs/perf/loss-attribution-matrix.md` §10.5 の 3 区分（ハードの上限／計測窓の定義／自分たちの制約）と、数字で 3 分類に割り当てられないセルの表示専用ラベル「分類保留」。0.12.0（#3273）の分類を引き継いだ（0.12.0 にあった 13 セル）。数字の「0.13.0」は本計測の 5 run 中央値（実測）、§10.3〜§10.5 の値は 0.11.0 時点の実測・推定（タグは同節のまま）。(3)(4) は §10.3・§10.4 の「再現で届く比」「工夫で届く比」で、目標値の列であり合否には使わない。制約の緩和は提案しない。内容は `scoreboard/unreached_0130.tsv` と同じ。

- 内訳: ハードの上限 1・計測窓の定義 2（Mac mini gemm Metal N=1024 と GB10 gemm CPU N=256）・自分たちの制約 9（うち確度低 4）・分類保留 2（M-MTL-TRN・G-CPU-INF）。
- 0.12.0 の 17 セルから外れた 4 セル（GB10 gemm CPU N=512・1024・2048・4096）は 1 位になったため理由表にない。新規は M-MTL-TRN（Mac mini train Metal。0.12.0 は 1 位）。
- M-MTL-TRN: fandhe-ai は 0.12.0 比 1.00（reuse 0.5118→0.5135 ms）で、PyTorch MPS の窓が 0.5180→0.4902 ms に縮んだ（5 run とも 0.13.0 の PyTorch が 0.12.0 の最小値より速い）。相手の経路が未測定のため「分類保留」。なお表示専用ラベル「分類保留」の注釈（title）は生成器に「所有者判断 2026-10-27」と固定で入っており、M-MTL-TRN の分類保留は 0.13.0 で新たに生じたもので、所有者判断を経ていない。

| セル | 比 | 分類 | 数字・根拠（0.13.0 の 5 run 中央値の実測と、0.11.0 時点の §10.3〜§10.5） |
|---|---|---|---|
| Mac mini gemm Metal N=1024 | 僅差 0.99 | 計測窓の定義 | (3)／(4)（§10.3・§10.4。目標値の列で合否に使わない）: F2 0.76／F4 1.00（仮想上限）。数字: 0.13.0: fandhe-ai reuse 2.308 ms 対 candle fresh 2.298 ms（0.12.0 は 2.354 ms 対 2.300 ms）。0.11.0 の自作カーネル kernel_gpu 0.861 ms（2.50 TFLOPS）は candle excl 0.950 ms（2.26 TFLOPS）より速い（MM-182）。0.12.0 以降は常駐入力で upload が窓から外れ、残る固有分は readout（0.11.0 の host_copy 0.241 ms）と checksum。F4（upload・host_copy を 0）の 1.00 と実測 0.99 が並ぶ |
| Mac mini gemm Metal N=2048 | 僅差 0.94 | 自分たちの制約 | (3)／(4)（§10.3・§10.4。目標値の列で合否に使わない）: F2 0.81／F3 0.76〜0.78・F4 0.84。数字: 0.13.0: 10.131 ms（窓で 1.70 TFLOPS）対 candle 9.523 ms（1.80 TFLOPS）（0.12.0 は 10.113 ms 対 9.550 ms）。0.11.0 の kernel_gpu 2.13 TFLOPS 対 candle excl 3.15・MPS 純時間 3.237 TFLOPS（MM-182・MM-179）。MPS は REQ-1 により使わない構成。実測 0.94 は (4) の推定 0.84 を上回った |
| Mac mini gemm Metal N=4096 | 2 位 0.88 | 自分たちの制約 | (3)／(4)（§10.3・§10.4。目標値の列で合否に使わない）: F2 0.62／F3 0.71〜0.73・F4 0.70。数字: 0.13.0: 62.712 ms（2.19 TFLOPS）対 candle 55.303 ms（2.49 TFLOPS）（0.12.0 は 65.389 ms 対 55.601 ms）。0.11.0 の kernel_gpu 2.11 TFLOPS（最良タイル 2.58）対 candle excl 3.24・MPS 3.429 TFLOPS（MM-182・MM-181・MM-179）。REQ-1（MPS 不使用） |
| Mac mini gemm CPU N=256 | 3 位 0.71 | ハードの上限 | (3)／(4)（§10.3・§10.4。目標値の列で合否に使わない）: Accelerate は再現不能・F2 0.95／SME は対象外で未測定。数字: 0.13.0: fandhe-ai 0.235 ms 対 PyTorch 0.165 ms（0.12.0 は 0.237 ms 対 0.167 ms）。相手の純 mm は 0.0240 ms（閉源 Accelerate・MM-179）。0.11.0 の 1 位（1.04）は PyTorch の窓が 0.320 ms だったセッションの値で、0.12.0 以降の計測では PyTorch の窓が 0.52 倍（0.13.0 は 0.165 ms。MM-179 が記録した 0.52〜0.54 倍の窓と同じ水準） |
| Mac mini gemm CPU N=512 | 2 位 0.76 | 自分たちの制約 | (3)／(4)（§10.3・§10.4。目標値の列で合否に使わない）: F2 1.05／SME ON の A/B で PyTorch 比 1.307／1.251。数字: 0.13.0: 0.941 ms 対 PyTorch 0.716 ms（0.12.0 は 0.951 ms 対 0.731 ms）。SME は既定 OFF（SME_PRODUCTION_ENABLED = false）。SME ON の A/B（MM-176）は同一ラウンド内の PyTorch 比 0.737／0.742 → 1.307／1.251 |
| Mac mini gemm CPU N=1024 | 2 位 0.70 | 自分たちの制約 | (3)／(4)（§10.3・§10.4。目標値の列で合否に使わない）: F2 0.91／SME ON で 1.406／1.246。数字: 0.13.0: 5.329 ms 対 PyTorch 3.734 ms（0.12.0 は 5.423 ms 対 3.649 ms）。SME 既定 OFF。MM-176 の PyTorch 比 0.650／0.619 → 1.406／1.246 |
| Mac mini gemm CPU N=2048 | 2 位 0.52 | 自分たちの制約 | (3)／(4)（§10.3・§10.4。目標値の列で合否に使わない）: F2 0.68／SME ON で 1.109／1.030。数字: 0.13.0: 35.671 ms（482 GFLOPS）対 PyTorch 18.823 ms（913 GFLOPS）（0.12.0 は 36.688 ms 対 19.231 ms）。SME 既定 OFF。MM-176 の PyTorch 比 0.525／0.494 → 1.109／1.030 |
| Mac mini train Metal | 僅差 0.95 | 分類保留 | 数字で 3 分類に割り当てられなかった（相手 PyTorch MPS の経路は未測定。§10.4・§10.11 も同一セッションの相手側内訳を取っていない）。(3)／(4)（§10.3・§10.4。目標値の列で合否に使わない）: (3) 未測定／F3 0.36（推定）。これらは 0.11.0 時点（比 0.32）の値で、0.13.0 の比 0.95 に対する目標値ではない。数字: 0.13.0: fandhe-ai reuse 0.514 ms 対 PyTorch MPS 0.490 ms（中央値同士）。0.12.0 は 0.512 ms 対 0.518 ms（比 1.01 で 1 位）。fandhe-ai の 0.12.0 比は 1.00（0.512→0.514 ms）、相手の PyTorch MPS の窓が 0.518→0.490 ms（0.95 倍）に縮んだ分だけ比が下がった。PyTorch の run 別は 0.13.0 が 0.4848〜0.4983 ms、0.12.0 が 0.5081〜0.5316 ms で重ならない（fandhe-ai の run 別は 0.5112〜0.5170 ms 対 0.5083〜0.5154 ms）。相手の窓が縮んだ原因（経路・セッション）は未測定で、本計測からは分けられない |
| Mac mini train CPU | 3 位 0.28 | 自分たちの制約 | 確度低。(3)／(4)（§10.3・§10.4。目標値の列で合否に使わない）: (3) 未測定／SME ON 0.90〜0.91・F3 0.22。数字: 0.13.0: fandhe-ai reuse 0.496 ms 対 PyTorch 0.137 ms（0.12.0 は 0.508 ms 対 0.129 ms）。§10.5.2: fandhe-ai は第 1 層の dX を計算してから捨てる形（推定）、L2 の細い形状の単スレッド 29〜40 GFLOPS（MM-177）。相手の純カーネルは未測定で、ハードの上限は数字で示せない |
| Mac mini infer CPU | 4 位 0.15 | 自分たちの制約 | 確度低。(3)／(4)（§10.3・§10.4。目標値の列で合否に使わない）: (3) 未測定／F3 0.29（REJECT）。数字: 0.13.0: 0.200 ms 対 PyTorch 0.0310 ms（0.12.0 は 0.205 ms 対 0.0310 ms）。SME は m=64 に効かない（0.9821／1.0100・MM-176）。L2 形状の単スレッド 29〜40 GFLOPS（MM-177）。相手の純カーネルは未測定 |
| GB10 gemm CPU N=256 | 3 位 0.72 | 計測窓の定義 | (3)／(4)（§10.3・§10.4。目標値の列で合否に使わない）: F2 1.39／affinity REJECT。数字: 0.13.0: reuse/fresh 1.91（reuse 0.366 ms・fresh 0.191 ms。5 run 中央値同士）。fresh 0.191 ms 対 PyTorch 0.264 ms（中央値同士・参考）。0.12.0 は reuse/fresh 1.79（新腕。旧腕は 0.95）。reuse が fresh より遅い後退は 0.13.0 でも N=256 に残る（N=1024 は 1.14、N=512・2048・4096 は 0.99〜1.00。0.12.0 は 5 サイズとも 1.23〜1.79）。区間の切り分けは本計測では行っていない |
| GB10 infer CUDA | 3 位 0.51 | 自分たちの制約 | 確度低。(3)／(4)（§10.3・§10.4。目標値の列で合否に使わない）: (3) 未測定／Graph capture 0.861（undetermined）・F3 0.47。数字: 0.13.0: fandhe-ai reuse 0.0785 ms 対 PyTorch 0.0402 ms・candle 0.0415 ms（0.12.0 は 0.0776 ms 対 0.0400 ms・0.0413 ms）。0.11.0 の chain 95.9 µs のうち導出 kernel_wait 63.2 µs、launch 往復 6.14 µs（GB-189）。相手は cuBLAS（閉源）で、REQ-1 により使わない構成。相手側の区間分解は未測定 |
| GB10 train CPU | 2 位 0.52 | 自分たちの制約 | 確度低。(3)／(4)（§10.3・§10.4。目標値の列で合否に使わない）: (3) 未測定／F3 0.39。数字: 0.13.0: 0.723 ms 対 PyTorch 0.379 ms（0.12.0 は 0.734 ms 対 0.363 ms）。§10.5.3: dX を計算してから捨てる形（推定）・NEON のみ（SME 非対応）。相手の経路は未確認 |
| GB10 infer CPU | 2 位 0.76 | 分類保留 | 数字で 3 分類に割り当てられなかった（0.12.0 の所有者判断 2026-10-27 を引き継ぐ。相手の経路の確認は #3191 の (1) が未測定）。(3)／(4)（§10.3・§10.4。目標値の列で合否に使わない）: (3) 未測定／F3 0.77（REJECT）。数字: 0.13.0: 0.162 ms 対 PyTorch 0.126 ms（0.12.0 は 0.167 ms 対 0.131 ms）。0.11.0 で記録された reuse が fresh より遅い逆転は本計測では見えない（reuse 0.1621・fresh 0.1687 ms）。相手の経路と純カーネルは未測定で、制約の欄にも該当する項目がない（§10.5.3）。3 区分のどれにも数字で割り当てられない |

## 既知の訂正（0.12.0 のページの学習 loss 注記）

0.12.0 のスコアボード（`../framework-compare-0.12.0-remeasure/scoreboard/out/scoreboard_0120.html`）の学習の最終 loss の注記は「fandhe-ai 0.080541・他 5 フレームワーク 0.117549〜0.117550」としていたが、実データでは burn の GB10 CUDA の train 行（5 run とも 0.117643）がこの範囲の外にあった（burn は 0.117549〜0.117643。0.12.0 のデータでも 0.13.0 のデータでも同じ）。0.13.0 のスコアボードは生成器が全 run の min〜max を数えて出すため、burn 0.117549〜0.117643 と表示される。0.12.0 の docs・公開済みページは直していない。

## 実機テスト（#3256 の C1〜C9・M1〜M6、#3272 の X1）

未実施（範囲外）。RULE.txt のとおり 0.13.0 の本計測の範囲に含めていない。

## 収録物

| パス | 内容 |
|---|---|
| `RULE.txt` | GB10 の事前登録規則（固定 2026-10-31T04:56:14Z） |
| `scripts/dgx-run-0130.sh`・`scripts/dgx-0130-loop.sh` | 1 round・5 round ループ（本計測で使った版。`dgx-run-0120.sh`／`dgx-0120-loop.sh` の単腕版） |
| `gb10/run{1..5}/` | 各 round の JSONL（`results-dgx-0.13.0.jsonl`＝run_all_cuda.sh 112 行、`-extra` 8 行、`-py` 28 行）・`tree.txt`・`rev-stamp.txt`・`env_info.txt`・`uptime_*.txt`・`gpu_util_before.txt`・`run_all_cuda.log`・`*.err`・`skipped-dgx-0.13.0.log` |
| `gb10/run{1..5}.log`・`gb10/loop.log`・`gb10/gate.log`・`gb10/build.log` | round のログ・ループのログ・ゲート・系列前のビルド |
| `gb10/*-median5.jsonl` | 5 run 中央値（本体・extra・Python） |
| `smoke/` | RULE 固定前の行数スモーク（判定に使わない） |
| `aggregate/median_rounds.py`・`check_identity.py` | 集計器・同一性検査（単腕版。`--self-test` あり） |
| `scoreboard/gen_0130.out` | 判定一覧（`gen_0130.py` の標準出力。`run_0130.sh` で再生成して byte 一致を確認） |
| `scoreboard/gen_args.txt` | 負荷・日付・skipped・公開日・コミット数・GB10 CPU gemm の所見の注記引数（実測値を記入済み） |
| `scoreboard/gen_0130.py`・`body_0130.html`・`style.css` | スコアボード生成器・テンプレート・CSS（`body_0130.html` は bench-fandhe の記述〔診断フラグの追加〕と env/ の記述を実測に合わせて修正した） |
| `scoreboard/unreached_0130.tsv` | 1.0 以下の 14 セルの届かなかった理由（3 分類＋分類保留） |
| `scoreboard/run_0130.sh`・`check_0130.py` | 判定一覧と HTML の一括生成・機械照合 |
| `scoreboard/out/scoreboard_0130.html` | 0.13.0 のスコアボード（1 ファイル・CSS インライン。公開はメインのセッション） |

`env/`（`pip freeze`）: 計測後（2026-10-31T06 時台 UTC）に同じ venv の `pip freeze`（GB10 は `pip freeze`＋`python --version`、Mac mini は `uv pip freeze`）を取り `env/` に収録した。0.12.0 の `env/` と byte 一致（numpy・keras を含む全パッケージが同じ版）。

## 再現手順

```bash
# 0. 展開ツリー（単腕）。.rev-stamp は 40 桁 1 行
PIN=5d05a50235138645b0b069060ed54dfd2c46db7f
git archive $PIN | tar -x -C tree && echo $PIN > tree/.rev-stamp
# 1. GB10: ~/work/r0130/tree、~/work/dgx-run-0130.sh、~/work/bench_py.py、~/work/.venv-bench を置き、実機ロックを取ってから
ROOT=~/work/r0130/tree EXPECT_SHA=$PIN setsid nohup bash dgx-0130-loop.sh > loop-gb10.log 2>&1 &
#    ログは ~/work/dgx-0130-logs/（run{1..5}/・gate.log・build.log）。回収して gb10/ に置く
# 2. 集計（このディレクトリで）
g=gb10; python3 aggregate/median_rounds.py $g/results-dgx-0.13.0-median5.jsonl $g/run{1..5}/results-dgx-0.13.0.jsonl   # extra・py も同様
python3 aggregate/check_identity.py $PIN results-dgx-0.13.0.jsonl=112,results-dgx-0.13.0-extra.jsonl=8,results-dgx-py-0.13.0.jsonl=28 gb10/run{1..5}
# 3. スコアボード（Mac mini 側の中央値・run も揃っていること）
bash scoreboard/run_0130.sh skeleton        # 1.0 以下のセルの雛形 TSV（unreached_0130.template.tsv）を作る（作業用。コミットしない）
cp scoreboard/unreached_0130.template.tsv scoreboard/unreached_0130.tsv   # 分類・理由・出典を実測に基づいて埋める
#    scoreboard/gen_args.txt の <要記入 を実測値で埋める。注記引数はプレーンテキスト（HTML エスケープされる）
bash scoreboard/run_0130.sh                 # gen_0130.out・out/scoreboard_0130.html を作り check_0130.py で照合
```

## スコアボード

**状態: 生成済み（2026-10-31）。** `scoreboard/run_0130.sh` は、前版（0.12.0）の入力 5 ファイルの sha256 を RULE.txt の値と照合したうえで、`gen_0130.py` を 1 回かけて判定一覧（`gen_0130.out`）と HTML（`out/scoreboard_0130.html`）を作り、`check_0130.py` で HTML の判定行・tally・2 条件表・届かなかったセルの表を判定一覧と照合する。

- 生成後の `run_0130.sh` の再実行で `gen_0130.out` は byte 一致（MISMATCH なし）、`check_0130.py` は `check ok: 判定行 27（Mac mini 13・GB10 14）・tally 新 (13, 3, 11, 1) 前版 (10, 4, 13, 1)・届かなかったセル 14（分類保留 2）`。
- 生成器の判定規則・tolerance・baseline は変えていない。変えたのはテンプレート `body_0130.html` の 4 文のみ: bench-fandhe の「変えず registry ピンだけ」「0.12.0 から変更なし」（3 か所）を「registry ピンの更新と既定 OFF の診断フラグ（#3685）の追加以外は変えない」に、「pip freeze は計測ディレクトリの env/」の文は計測後に env/ を収録したので元に戻した（上の事前確認 1・5 の実測と食い違うため）。

## 所有者判断を仰ぐ点

1. **bench-fandhe の差分（記録のみ・判断は不要と考える）**: RULE.txt の事前確認の文言（`*.rs` の差 0 件）は満たしていないが、差は #3685 の診断フラグ（既定 OFF・本計測では未使用）とピン表記で、OFF 時の計測区間内のコードは 0.12.0 の計測時と同じ（上の事前確認 1）。測り直しは不要と判断した。異論があれば知らせてほしい。
2. **M-MTL-TRN の分類保留**: Mac mini train Metal は 1 位（1.01）から僅差（0.95）に下がった。fandhe-ai は変化なし（1.00）で、PyTorch MPS の窓だけが縮んだが、相手の経路が未測定のため 3 分類に割り当てていない。相手側の計測（#3191 の (1)）を待つか。
3. **GB10 gemm CPU N=256**: reuse が fresh より 1.91 倍遅い後退が残る（0.12.0 は 1.79）。N=1024 も 1.14。切り分け（#3646）の続きをするか。
