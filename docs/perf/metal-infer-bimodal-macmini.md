# Mac mini の infer Metal reuse の二峰性の原因特定（イシュー #3183）

**状態（第 2 ラウンドまで完了・2026-10-21）: Mac mini で 5 run を 2 ラウンド実測済み（負荷ゲートは両ラウンドとも pass・集計は exit 0）。** AC1（分布）は充足し、二峰というより「単峰の本体 + 完了待ちの裾 + run1 の低性能状態」と読める（§6.3）。AC2（帰属）は第 2 ラウンドで `commit_wait - gpu_busy`（ホスト側の完了待ち。GPU 実行時間は不変）まで絞れた。それ以上の内訳と、待ちが伸びる理由は未特定。AC3（系列 A／B の説明）は事前登録の機械判定が「説明できる」だが、A 水準は run1 の低性能状態で再現し、B 水準の再現は弱く、第 1 ラウンドの bench 対照 0.427 ms は再現していない（§6.3・§6.4。第 1 ラウンド単独の評価は §3.3 の「部分的」）。AC4（ライブラリのコードと既定値を変えない）は充足している。生ログは `logs/metal-infer-bimodal-macmini-3183/macmini/`（第 1 ラウンド）と `macmini-r2/`（第 2 ラウンド）。

## 1. 背景

- 0.11.0 の Mac mini 計測（`docs/perf/logs/framework-compare-0.11.0-macmini-remeasure/`）では、infer Metal reuse（784→256→ReLU→10・batch 64・`Sequential::predict_resident`）の 1 反復時間が run 内で q1 約 0.385 ms／q3 約 0.61 ms に分かれ、run 中央値が 0.382〜0.609 ms に散らばる。0.478 ms 付近にまとまった run もあり、山が 2 つとは限らない
- 系列 B の 5 run 中央値は 0.388 ms（candle 比 1.13 倍）、系列 A は 0.539 ms（0.72 倍）。fresh は 0.85〜0.91 ms で安定している
- 既存の `metal-tape-build-infer-fixedcost.md`（#2114）はセル単位の集計値しか出さず、高い側のモードがどの区間と同時に現れるかを分けられない。`infer-reuse-phase-breakdown.md` §10 は `predict_resident` が単一区間のため分解できないと報告している

## 2. 設計（事前登録規則は `logs/metal-infer-bimodal-macmini-3183/RULE.txt`）

1 反復 1 行の JSONL を出す 4 腕を、腕ごとに warmup 20 + 計測 500 反復の長い系列で回す。

| 腕 | 内容 | 判別する仮説 | 実装 |
|---|---|---|---|
| R | `predict_resident` → bench と同じ legacy 読み出し → checksum | 二峰性が main HEAD と診断の枠組みでも再現するか | `crates/facade/tests/metal_infer_bimodal_diag.rs` |
| D | provider_select → tape_new → snapshot → chain → host_copy → checksum（キャッシュ OFF = 既定） | 高モードがどの区間で起きるか | 同上 |
| Don | D をデバイス存在確認キャッシュ ON のスコープ内で実行（診断専用） | provider_select（IOKit・`MTLCopyAllDevices`）を外しても二峰が残るか | 同上 |
| G | upload → `linear_forward_device_tracked` × 2 → `synchronize_with_gpu_timestamps` → download | GPU 実行時間／ホスト完了待ち／encode のどれが増えるか | `crates/backend-metal/src/infer_fixed_cost_diag_tests.rs`（`cfg(test)`・`#[ignore]`） |

- カウンタ（encode／command_buffers／wait／probe／hit／uploads／downloads）は既存の診断面の差分を反復ごとに出す。診断カウンタ・既定値・本番コードは追加も変更もしない
- モード検出は「2 つ」を前提にせず、相対ギャップ 10% 以上（両側 5% 以上が残る位置）で分割する。時間方向（交互出現／プロセス単位の固着）、帰属（区間ごとの Δ_P/Δ_total ≥ 0.5 が多峰 run の過半数）、系列 A/B の窓判定（連続 20 反復の窓中央値）の閾値は RULE.txt に実測前に固定した
- 腕の起動順は run ごとにローテーションし、順序効果を打ち消す。集計器 `aggregate.py` は fail-closed（`--self-test` 22 項目 pass）

## 3. 結果（Mac mini 実測・2026-10-21）

実測条件: Mac mini（Apple M4 10 コア・macOS 27.0.0・rustc 1.99.0）、転送ツリーの `.rev-stamp` = `26e60f6322e438acb007c6024c237224ddceb008`（`macmini/rev-stamp.txt`）、独立 5 run × 4 腕 × 500 反復（`run{1..5}.jsonl` は各 2000 行）、負荷ゲートは 5 run とも pass（load1 2.14〜2.15。`macmini/load_gate_status.txt`）。前提テスト `metal_infer_decomposition_matches_public_api_bit_exact` は pass（`macmini/prereq_bit_exact.log`）。集計は `aggregate.py` の出力そのもの（`macmini/aggregate.md`。exit 0・手編集なし）。powermetrics は `sudo -n` 不可のため未取得、thermal／performance 警告はいずれも「記録なし」、低電力モードは 0（`macmini/power_run*.txt`・`env_run*.txt`）。

### 3.1 分布（AC1）

1 反復時間の run 別中央値（クラスタ割合は `aggregate.md` を参照）。

| 腕 | run1 | run2 | run3 | run4 | run5 | 多峰の run |
|---|---|---|---|---|---|---|
| R（predict_resident 全体） | 0.550 | 0.324 | 0.323 | 0.324 | 0.323（低 93.6% / 高 0.445 ms 6.4%） | 5 のみ |
| D（キャッシュ OFF） | 0.322 | 0.340（低 92.6% / 高 0.458 ms 7.4%） | 0.322 | 0.323 | 0.322 | 2 のみ |
| Don（キャッシュ ON） | 0.312 | 0.328 | 0.314（低 90.8% / 高 0.444 ms 9.2%） | 0.314 | 0.311 | 3 のみ |
| G（GPU のみ・wall） | 0.545 | 0.310 | 0.311 | 0.311 | 0.309 | なし |

単位は ms。所見は次のとおり。

- 多峰の run は R／D／Don で各 1 run（5 run 中）だけ。いずれも高モードは 0.44〜0.46 ms（低モード比 約 +0.12〜0.14 ms）で、反復の 6〜9% を占め、プロセス内で 29〜37 回切り替わる（モード連続長の中央値 12〜22 反復）。プロセス単位の固着ではなく、同一プロセス内で交互に出現する
- run1 は別の現象で、R と G が run 全体で 0.55 ms 付近（Q3/Q1 約 1.75）に張り付いた。同じ run1 の D と Don（R の直後に起動、G の直前）は 0.31〜0.32 ms で、腕固有ではなく時間局所の状態に見える。原因は特定できていない（OS 要因は取得できた範囲では差がない）
- 腕 G（facade 層を含まない GPU 実行）は、run1 の全体上振れを除き多峰の run がない
- 追記（第 2 ラウンドでの訂正）: 上の「多峰の run」は RULE.txt の閾値（両側 5% 以上が残る位置の相対ギャップ 10% 以上）による判定で、各腕 1 run だけになる。一方、0.4 ms を超える反復は 20 の腕×run のうち 18 で 1.6〜8.2% 出ており（`logs/metal-infer-bimodal-macmini-3183/macmini-r2/posthoc-r1.md`。事後・探索）、高モードに当たる反復はほぼ常に現れている。5% 規則が「多峰」を稀に見せているだけで、分布が単峰だったわけではない

### 3.2 帰属（AC2）

- 帰属の機械判定: R は `predict_resident`（比 1.00）、D と Don は `chain`（比 1.00）。`host_copy`・`checksum`・`provider_select`・`tape_new`・`snapshot` の寄与は 0.00。つまり高モードの増分は、Metal の encode から完了待ちまでを担う GPU チェーン区間で起きる
- 診断カウンタ（encode／command_buffers／wait／probe／hit／uploads／downloads）の高低差はない。回数や dispatch 数が増えて遅くなるのではなく、同じ回数の処理に掛かる時間が伸びている
- デバイス存在確認キャッシュ ON の腕（Don）でも二峰が残る（run3）。`provider_select`（IOKit）は原因ではない
- 未特定: チェーン区間の内訳（encode／GPU 実行／完了待ち）のどれが増えているか。腕 G は多峰を再現しなかったため、GPU タイムスタンプとの突合ができなかった。OS 要因（クロック・電力状態）も sudo なしでは反復単位で取れず、特定できなかった。以後の切り分けは、多峰が出た run で再現する条件（連続実行・負荷）を変えた別イシューが必要

### 3.3 系列 A／B の差の説明（AC3）

`aggregate.py` の事前登録判定（腕 R・連続 20 反復の窓）の機械判定は「説明できる」。ただしこれは事前登録の機械判定であり、次の限定付きで **「部分的」** と読み替える（第 2 ラウンドでの訂正。冒頭の状態行も同じ評価）。

- 窓中央値の 5〜95% は 0.315〜0.586 ms で、bench 観測の B=0.388 ms・A=0.539 ms の両方を含む。窓中央値が低側に入る割合は 84.9%、高側は 15.1%（条件 (a)(b)(c) はすべて成立）
- 限定: 条件 (c) と窓中央値の 5〜95% の広がりは run1 の張り付き（R が 0.55 ms）にほぼ依存している。B 水準 0.388 ms も bench 対照 0.427 ms も、run1 を除く R の水準では再現していない。R の run 別中央値は 0.323〜0.550 ms で、系列 A に近い値は run1 の張り付き（0.550 ms）だけである。5 run 中央値の中央値のブートストラップ 5〜95% は 0.322〜0.326 ms（参考・判定に不使用）で、系列 B の 0.388 ms には届かない。R の水準が bench 側より低いことは §4 の水準比較も参照

- bench 比（B・A を低モード中央値 0.323 ms で割った値）は 1.20／1.67。第 1 ラウンドの `aggregate.md` は分母を定数 0.385 ms にしており 1.01／1.40 と出ていたが、RULE.txt は「低モード中央値で割った比」と定めており、同じ表の低モード中央値と食い違っていたため第 2 ラウンドで分母を実測値に訂正した（記録のみで判定は変わらない。`macmini/aggregate.md` は再生成済み）

### 3.4 #3191 への申し送り（安定させたときの比）

低モード中央値 0.323 ms（腕 R）に安定させられた場合の candle 比は、別セッション観測の candle（系列 B 0.507 ms・系列 A 0.349 ms）に対して、B で 1.57 倍、A で 1.08 倍。ただしこの数値は R 腕（main HEAD・診断の枠組み）の水準で、同じセッションで測った bench 対照（§4）の水準 0.427 ms で置き換えると B で 1.19 倍、A で 0.82 倍となり、A では負けのまま残る。#3191 に書く値は、どちらの水準を前提にするかを明記すること。candle は別セッションの値で、本書の実測と同一条件ではない。

## 4. 限界

- 腕 G は facade 層（tape・snapshot・legacy 読み出し）を含まない。`linear_forward_device_tracked` は本番チェーンと同じ呼び出しを使う
- 反復単位の GPU クロック・電力状態は sudo なしでは取れない。`powermetrics` は `sudo -n` が通る場合だけ run 単位で取得し、通らなければ未取得と記録する（sudo は使わない）
- bench-fandhe は crates.io の `fandhe-ai =0.11.0`、本診断は main HEAD を使う。`v0.11.0..HEAD` には `crates/autodiff/src/optim/device_store.rs`（約 380 行）、`crates/backend-metal/src/ops.rs`（約 590 行）、`crates/facade/src/compat/sequential.rs` ほかの変更があり、infer チェーンへの影響の有無は実測側で R 腕の水準が bench 対照（0.11.0）と同程度かで確かめる。同程度でなければ原因特定の前提が崩れるため、その旨を結果に書く
- M4 Max での同じ診断は対象外
- 実測時の補足（2026-10-21）:
  - `env_info.txt` の `git_head` 行が空で `fatal: gitfile does not point to a valid repository` が出ているのは、転送ツリーに worktree 由来の `.git` ファイル（ディレクトリではなく gitfile）が rsync の除外パターン `.git/` をすり抜けて渡り、ノード側で解決できないため。リビジョンは転送時の `.rev-stamp`（`macmini/rev-stamp.txt` = 計測時の HEAD）で特定する。転送物に秘密情報（`.env*`・`settings.local.json`・`.local.md`）が渡っていないことは find で確認した。ツリー指紋の突合は取っていなかった。第 2 ラウンドで対処した: `stage-tree.sh` が `git archive` から転送ツリーを作り（gitfile も追跡外のファイルも入らない）、`.rev-stamp` と `.tree-manifest` を書く。`orchestrate.sh` は `.rev-stamp` から `git_head` を、`.tree-manifest` の照合と sha256 から `tree_fingerprint` を env_info へ書く（`git rev-parse` は使わない。無い・不一致なら計測前に停止）。rsync の除外は `--exclude '.git'`（末尾スラッシュなし）にする。共有 doc `docs/real-hardware-verification-env.md` §3 の `.git/` 除外は別イシューで修正を提案する（本書では手を入れない）
  - ホスト名と `$HOME` は収録前にマスクした。前提テストのビルドログだけは orchestrate.sh を通らないため、`/Users/<ユーザー>` を `<home>` へ手置換した
  - bench 対照（crates.io 0.11.0・`infer metal 64 reuse`・5 回）の中央値は 0.418／0.422／0.480／0.465／0.427 ms（中央値の中央値 0.427 ms。`macmini/bench_control/`）。R 腕の低モード 0.323 ms より約 32% 高く、水準は同程度とは言えない。差の原因（`v0.11.0..HEAD` の変更、bench のウォームアップ 20・計測 20 反復の集計方式、診断の枠組みの違い）は切り分けていない。そのため R 腕の絶対値を bench の系列 A／B と直接比べる結論（§3.3・§3.4）は、この水準差を前提に読むこと
  - 帰属の機械判定（§3.2）は、各腕の多峰 run が 1 run ずつで過半数の根拠が薄い。再現性の確認には run 数を増やした再測が要る

## 5. 申し送り・スコープ外

- 実測は完了（§3・§6）。#3191 には §3.4・§6.5 の水準を前提つきで併記して反映する（本イシューでは #3191 へコメントしない）
- 二峰性の是正そのもの（既定値や本番経路の変更）は本イシューの対象外。原因が特定できたら別イシューで A/B 後に判断する
- 関連: `infer-reuse-phase-breakdown.md` §10・`metal-tape-build-infer-fixedcost.md`（#2114）・#3191（セル別の到達値の表）

## 6. 第 2 ラウンド（2026-10-21・内訳の切り分けと系列 A／B の水準の再現）

第 1 ラウンドで残った課題（AC2 の内訳が「GPU チェーン全体」までだったこと、AC3 が run1 の張り付きに依存し B 水準も bench 対照も再現しなかったこと）を、診断だけで切り分ける。規則は `logs/metal-infer-bimodal-macmini-3183/RULE2.txt`（実測前の単独コミット `60abbd553` で固定。RULE.txt は変更しない）。ライブラリのコード・既定値・診断テスト・診断カウンタは変えていない（AC4）。

### 6.1 設計

- 第 1 ラウンドの生ログを事後に探索して仮説を立てた（事後・探索で判定には使わない。`macmini-r2/posthoc-r1.md`）。H1 run 内の裾は GPU 実行時間ではなくホスト側の完了待ちで増える／H2 run 全体の低性能状態（run1）／H3 bench と同じ窓（計測 iter 0〜19）の立ち上がり／H4 v0.11.0 と main HEAD の版差／H5 train 直後という実行順
- 9 条件を 5 run、run ごとに 2×(n-1) 個ずつ巡回して起動する: HEAD の R／D／Don／G（500 反復）、v0.11.0 ツリー（`git archive v0.11.0` に HEAD の診断テスト 1 ファイルだけを重ねた）の R／D／Don、bench-fandhe（crates.io 0.11.0・変更なし）の `infer metal 64 reuse` 単独（Bc）と `train metal 64 reuse` の直後（BcT）
- 判定: 裾（中央値の 1.25 倍超）の帰属（Δ_P/Δ_total ≥ 0.5 が対象 run の過半数）・G の「GPU 実行時間は不変」（|Δ gpu_busy|/本体中央値 < 5%）・run 全体の低性能状態（5 run 中央値の 1.5 倍以上を GPU・CPU 両方で満たす run）・立ち上がり窓（W/S ≥ 1.10）・(v) 版差／(h) bench 固有の差／(o) 実行順の効果（いずれも 1.10 倍）・A／B 水準 ±10% の再現

### 6.2 実測条件

Mac mini（Apple M4・macOS 27.0.0・rustc 1.99.0）。転送ツリーは `stage-tree.sh` で `git archive` から作り、ノードの `orchestrate.sh` が `.rev-stamp` と `.tree-manifest` を照合してから計測した（照合に通らなければ停止する）。

| 項目 | 値 | 出典 |
|---|---|---|
| HEAD の `git_head`／`tree_fingerprint` | `60abbd553d466883aa9151cec250ddc0612506f2`／`dcf8eb81…c137db1` | `macmini-r2/env_info.txt`・`rev-stamp.txt`・`tree_manifest.sha256` |
| v0.11.0 ツリー | `6b14fdb420f1d5ecb5d118b3451f3fe0aacd333b`（+ 診断テスト 1 ファイルを HEAD から重ねた）／`d04df64d…a14d` | `env_info.txt`・`v0110_rev-stamp.txt`・`v0110_tree_manifest.sha256` |
| 負荷ゲート | 5 run とも pass（load1 2.21〜2.34） | `load_gate_status.txt` |
| run1 の開始 | ビルド終了の 180 秒後・load5 5.99（run2〜5 は load5 5.84〜5.91） | `run_meta.txt`・`env_run1.txt` |
| 前提テスト | `metal_infer_decomposition_matches_public_api_bit_exact` は pass | `prereq_bit_exact.log` |
| bench 対照 | `fandhe-ai v0.11.0`（`--locked`・別 `CARGO_TARGET_DIR`） | `bench_control/tree.txt` |

- 生ログ: `head/run{1..5}.jsonl`（各 2000 行）・`v0110/run{1..5}.jsonl`（各 1500 行）・`bench_control/run{N}_Bc.jsonl`／`run{N}_BcT.jsonl`（BcT の train 行は `run{N}_BcT_train.jsonl`）。集計は `aggregate.py --rule 2` の出力そのもの（`macmini-r2/aggregate.md`。exit 0・手編集なし）。主要な数値（HEAD-G run3 の裾の帰属・run1 の R の checksum 中央値）は生ログから独立に再集計して一致を確認した。マスク漏れ（実ホスト名・`/Users/<名前>`）の grep は 0 件
- v0.11.0 ツリーのビルドは成功し、checksum は HEAD と一致した。powermetrics は `sudo -n` 不可のため未取得

### 6.3 結果

**裾の帰属（AC2）**: 全腕で、裾は run の中央値より約 +0.1〜0.16 ms の区間として出る。

| 腕 | 帰属した区間（対象 run の過半数で比 ≥ 0.5） | 出典 |
|---|---|---|
| HEAD-G（GPU のみ） | `commit_wait`（4/4 run）と `commit_wait-gpu_busy`（4/4 run）。Δ commit_wait は +94〜123 µs（run1 は +232 µs）で、`upload`／`encode`／`download` の比は 0.00。**GPU 実行時間は不変**（|Δ gpu_busy| は本体中央値の 5% 未満、4/4 run。run3 は Δ gpu_busy +0.2 µs 対 Δ commit_wait +122.7 µs） | `aggregate.md` の HEAD-G 節 |
| HEAD-R／v0110-R | `predict_resident`（比 0.99〜1.00）。`host_copy`・`checksum` は 0.00〜0.05 | 同 HEAD-R・v0110-R 節 |
| HEAD-D／Don、v0110-D／Don | `chain`（比 0.99〜1.00）。`provider_select`・`tape_new`・`snapshot` は 0.00〜0.01 | 同各節 |

- 読み: 増えるのは encode でも GPU の実行時間でもなく、commit してから完了を受け取るまでのホスト側の待ち（`commit_wait - gpu_busy`）である。第 1 ラウンドで「GPU チェーン全体」までだった区間を、この範囲まで絞れた。0.4 ms を超える反復は、run1 を除いて HEAD の各腕で 1.6〜6.2%、v0110 の各腕で 1.6〜9.6%（記録のみ）。5% 規則による多峰判定は今回の 35 の腕×run のすべてで「なし」だった（第 1 ラウンドでは 3 つが「あり」）。分布が二峰として分かれているのではなく、単峰の本体に完了待ちの長い裾が一定割合で乗っている形である
- 特定できなかった点: `commit_wait - gpu_busy` をさらに「GPUStartTime より前（スケジューリング）」と「GPUEndTime より後（完了通知）」に分けることはしていない。`context.rs` に `cfg(test)` のアクセサが要り、ライブラリのファイルを変えるため本ラウンドの対象外とした（§6.5）。また、なぜ待ちが伸びる反復が 2〜10% の割合で出るのか（OS のスケジューリング・電力状態など）は診断だけでは特定できていない

**run 全体の低性能状態（H2）**: run1 だけが、GPU と CPU の両方で低性能だった（5 run 中央値の 1.5 倍以上）。

| 指標 | run1 | run2〜5 | 出典 |
|---|---|---|---|
| HEAD-G の gpu_busy 中央値 | 175.6 µs | 90.0〜90.9 µs | `aggregate.md` 低性能状態の表 |
| HEAD-R の checksum 中央値 | 4.33 µs | 2.29 µs | 同 |
| 各条件の run 中央値 | HEAD-R 0.555 ms・HEAD-G 0.534 ms・v0110-R 0.475 ms・Bc 0.551 ms | HEAD-R 0.322〜0.325 ms・Bc 0.315〜0.359 ms | `aggregate.md`・`bench_control/run{N}_Bc.jsonl` |

- run1 は 2 ラウンドとも、ビルド終了の直後（第 2 ラウンドは 180 秒後・load5 5.99、第 1 ラウンドは直前の load5 4.28）に始めた最初の run で、Bc（bench そのもの）も 0.551 ms に張り付いた。同じ run1 の中でも D と Don は平常だった。ビルド直後という相関は 2 ラウンドで再現したが、「ビルド直後」を単独で操作した検証（待機時間だけを変える等）はしていないため、原因とは断定しない

**立ち上がり窓（H3）**: bench と同じ計測 iter 0〜19 の中央値 W が定常 S の 1.10 倍以上になった腕×run は 35 のうち 11（HEAD-R 0・HEAD-D 2・HEAD-Don 1・HEAD-G 2・v0110-R 2・v0110-D 1・v0110-Don 3）。上振れは R の run だけに偏らず、全腕に散らばって出る。

**系列 A／B の説明（AC3）**: 5 run 中央値で次のとおり（`aggregate.md` 末尾）。

| 判定 | 値 |
|---|---|
| (v) 版差 | **なし**（v0110-R の S ÷ HEAD-R の S = 1.006。S は 0.325 ms 対 0.323 ms） |
| (h) bench 固有の差 | なし（Bc ÷ v0110-R の W = 0.788。Bc は 0.321 ms） |
| (o) 実行順の効果 | **なし**（BcT ÷ Bc = 0.985。BcT は 0.316 ms） |
| B 水準 0.388 ms ±10% の再現 | v0110-R の W（0.407 ms）のみ |
| A 水準 0.539 ms ±10% の再現 | 低性能 run1 の R の中央値（0.555 ms） |
| 機械判定 | 説明できる（両水準を再現） |

- 読み（機械判定への限定）: A 水準の再現は強い。run1 では Bc・HEAD-R・HEAD-G・v0110-R が同時に 0.48〜0.56 ms へ遅れ、Bc 自体が 0.551 ms だった。一方 B 水準の再現は弱い。再現した値は v0110-R の W の 5 run 中央値（0.407 ms）で、これは run2・run3 の立ち上がり上振れ（W/S 1.26・1.28）による。同じ上振れは HEAD-D・HEAD-G・v0110-D・v0110-Don の別の run にも出ており、版差（S が一致）ではなく立ち上がりの確率的なばらつきと読める。bench そのもの（Bc）の 5 run 中央値は 0.321 ms で、B 水準にも第 1 ラウンドの bench 対照（0.418〜0.480 ms・中央値 0.427 ms）にも届かない
- **第 1 ラウンドの bench 対照 0.427 ms は第 2 ラウンドで再現しなかった**。第 2 ラウンドの Bc は run1 の 0.551 ms と run3 の 0.359 ms を除き 0.315〜0.321 ms で、HEAD-R の定常と同じ水準である。このため、第 1 ラウンドの「R 腕の水準が bench 対照より約 32% 低い」という差は、版差・bench の集計方式・実行順では説明できず（(v)(h)(o) はいずれもなし）、第 1 ラウンドの bench 対照を測った時点のノードの状態（全 5 回が 0.42〜0.48 ms という、run1 の低性能状態とは違う持続した状態）に帰する可能性が残る。この点は原因を特定できていない

### 6.4 結論（確認できたものとできなかったもの）

| 要因 | 結論 |
|---|---|
| run 内の裾（完了待ち） | **確認**。増えるのは `commit_wait - gpu_busy`（ホスト側の完了待ち。+約 0.1〜0.12 ms）で、GPU の実行時間（gpu_busy）は不変。provider_select・tape_new・snapshot・host_copy・checksum・encode・upload・download は寄与しない。さらに細かい内訳（スケジューリングか完了通知か）と、待ちが伸びる反復が出る理由は未特定 |
| run 全体の低性能状態 | **確認（相関）**。run1 で GPU・CPU の両方が約 1.9 倍遅くなり、A 水準（0.539 ms）を再現する。2 ラウンドとも run1（ビルド直後）で起きたが、ビルド直後が原因かは未検証 |
| bench の窓の立ち上がり | 一部確認。11/35 の腕×run が W/S ≥ 1.10 になり、B 水準の弱い再現に寄与しうる。ただし系統的な差ではなく確率的なばらつき |
| 版差（v0.11.0 と HEAD） | **なし**（S の比 1.006）。v0.11.0 でも同じ裾・同じ帰属が出る |
| 実行順（train 直後） | **なし**（BcT/Bc = 0.985） |
| 第 1 ラウンドの bench 対照 0.427 ms | 再現せず・原因未特定 |

- AC1: 反復ごとの分布と 5 run の生ログを残した（`macmini-r2/`）。多峰は 5% 規則では検出されず、「二峰」より「単峰の本体 + 完了待ちの裾 + run1 の低性能状態」と記述するのが実測に合う
- AC2: 高い側と同時に増える区間は `commit_wait - gpu_busy`（G）。GPU 実行時間が不変であることまで確認した。さらに細かい内訳は上記のとおり未特定
- AC3: 事前登録の機械判定は「説明できる」。ただし A 水準は run1 の低性能状態で再現し、B 水準の再現は弱い（立ち上がり窓のばらつき）。第 1 ラウンドの bench 対照の水準は再現していない
- AC4: ライブラリのコード・既定値は変えていない

### 6.5 申し送り・スコープ外

- #3191 に反映する「安定させたときの比」は、前提とする水準を併記する。本ラウンドで bench そのもの（Bc）が定常で示した約 0.32 ms（第 1 ラウンドの R 腕の低モード 0.323 ms と同水準）を前提にすると、別セッション観測の candle（系列 B 0.507 ms・系列 A 0.349 ms。candle は同一条件ではない）に対して B で約 1.58 倍、A で約 1.09 倍。第 1 ラウンドの bench 対照 0.427 ms を前提にすると B で 1.19 倍、A で 0.82 倍で、こちらは本ラウンドでは再現しなかった。run1 のような低性能状態（約 0.55 ms）のままなら両方で負け側に倒れる。#3191 に書く値は前提とする水準を明記すること（本イシューでは #3191 へコメントしない）
- 二峰性（裾と低性能状態）の是正そのもの（既定値や本番経路の変更）は本イシューの対象外。別イシューで A/B のうえ判断する
- 切り出し候補（起票はしない）: (1) `commit_wait - gpu_busy` を GPUStartTime より前と GPUEndTime より後に分ける `cfg(test)` アクセサ（`context.rs`）。(2) run1 の低性能状態が「ビルド直後」に由来するかの切り分け（ビルド後の待機時間を変えて run1 相当を測る）。(3) 共有 doc `docs/real-hardware-verification-env.md` §3 の rsync 除外 `--exclude '.git/'` は worktree の gitfile を素通りさせる。`--exclude '.git'`（末尾スラッシュなし）への修正と、`stage-tree.sh` 方式（`.rev-stamp`・`.tree-manifest`）の共有を、他イシューの計測にも関わるため別イシューで提案する
