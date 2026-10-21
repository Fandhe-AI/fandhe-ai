# Mac mini の infer Metal reuse の二峰性の原因特定（イシュー #3183）

**状態: Mac mini で 5 run 実測済み（2026-10-21・負荷ゲート 5 run とも pass・集計は exit 0）。** AC1（分布）・AC3（系列 A／B の説明）は充足、AC2（帰属）は「GPU チェーン区間まで特定・その内訳は未特定」と明記して充足。AC4（ライブラリのコードと既定値を変えない）も充足している。生ログは `logs/metal-infer-bimodal-macmini-3183/macmini/`。

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

### 3.2 帰属（AC2）

- 帰属の機械判定: R は `predict_resident`（比 1.00）、D と Don は `chain`（比 1.00）。`host_copy`・`checksum`・`provider_select`・`tape_new`・`snapshot` の寄与は 0.00。つまり高モードの増分は、Metal の encode から完了待ちまでを担う GPU チェーン区間で起きる
- 診断カウンタ（encode／command_buffers／wait／probe／hit／uploads／downloads）の高低差はない。回数や dispatch 数が増えて遅くなるのではなく、同じ回数の処理に掛かる時間が伸びている
- デバイス存在確認キャッシュ ON の腕（Don）でも二峰が残る（run3）。`provider_select`（IOKit）は原因ではない
- 未特定: チェーン区間の内訳（encode／GPU 実行／完了待ち）のどれが増えているか。腕 G は多峰を再現しなかったため、GPU タイムスタンプとの突合ができなかった。OS 要因（クロック・電力状態）も sudo なしでは反復単位で取れず、特定できなかった。以後の切り分けは、多峰が出た run で再現する条件（連続実行・負荷）を変えた別イシューが必要

### 3.3 系列 A／B の差の説明（AC3）

`aggregate.py` の事前登録判定（腕 R・連続 20 反復の窓）の結論は「説明できる」。

- 窓中央値の 5〜95% は 0.315〜0.586 ms で、bench 観測の B=0.388 ms・A=0.539 ms の両方を含む。窓中央値が低側に入る割合は 84.9%、高側は 15.1%（条件 (a)(b)(c) はすべて成立）
- ただし限定付き。R の run 別中央値は 0.323〜0.550 ms で、系列 A に近い値は run1 の張り付き（0.550 ms）だけである。5 run 中央値の中央値のブートストラップ 5〜95% は 0.322〜0.326 ms（参考・判定に不使用）で、系列 B の 0.388 ms には届かない。R の水準が bench 側より低いことは §4 の水準比較も参照

### 3.4 #3191 への申し送り（安定させたときの比）

低モード中央値 0.323 ms（腕 R）に安定させられた場合の candle 比は、別セッション観測の candle（系列 B 0.507 ms・系列 A 0.349 ms）に対して、B で 1.57 倍、A で 1.08 倍。ただしこの数値は R 腕（main HEAD・診断の枠組み）の水準で、同じセッションで測った bench 対照（§4）の水準 0.427 ms で置き換えると B で 1.19 倍、A で 0.82 倍となり、A では負けのまま残る。#3191 に書く値は、どちらの水準を前提にするかを明記すること。candle は別セッションの値で、本書の実測と同一条件ではない。

## 4. 限界

- 腕 G は facade 層（tape・snapshot・legacy 読み出し）を含まない。`linear_forward_device_tracked` は本番チェーンと同じ呼び出しを使う
- 反復単位の GPU クロック・電力状態は sudo なしでは取れない。`powermetrics` は `sudo -n` が通る場合だけ run 単位で取得し、通らなければ未取得と記録する（sudo は使わない）
- bench-fandhe は crates.io の `fandhe-ai =0.11.0`、本診断は main HEAD を使う。`v0.11.0..HEAD` には `crates/autodiff/src/optim/device_store.rs`（約 380 行）、`crates/backend-metal/src/ops.rs`（約 590 行）、`crates/facade/src/compat/sequential.rs` ほかの変更があり、infer チェーンへの影響の有無は実測側で R 腕の水準が bench 対照（0.11.0）と同程度かで確かめる。同程度でなければ原因特定の前提が崩れるため、その旨を結果に書く
- M4 Max での同じ診断は対象外
- 実測時の補足（2026-10-21）:
  - `env_info.txt` の `git_head` 行が空で `fatal: gitfile does not point to a valid repository` が出ているのは、転送ツリーに worktree 由来の `.git` ファイル（ディレクトリではなく gitfile）が rsync の除外パターン `.git/` をすり抜けて渡り、ノード側で解決できないため。リビジョンは転送時の `.rev-stamp`（`macmini/rev-stamp.txt` = 計測時の HEAD）で特定する。転送物に秘密情報（`.env*`・`settings.local.json`・`.local.md`）が渡っていないことは find で確認した。ツリー指紋の突合は取っていない
  - ホスト名と `$HOME` は収録前にマスクした。前提テストのビルドログだけは orchestrate.sh を通らないため、`/Users/<ユーザー>` を `<home>` へ手置換した
  - bench 対照（crates.io 0.11.0・`infer metal 64 reuse`・5 回）の中央値は 0.418／0.422／0.480／0.465／0.427 ms（中央値の中央値 0.427 ms。`macmini/bench_control/`）。R 腕の低モード 0.323 ms より約 32% 高く、水準は同程度とは言えない。差の原因（`v0.11.0..HEAD` の変更、bench のウォームアップ 20・計測 20 反復の集計方式、診断の枠組みの違い）は切り分けていない。そのため R 腕の絶対値を bench の系列 A／B と直接比べる結論（§3.3・§3.4）は、この水準差を前提に読むこと
  - 帰属の機械判定（§3.2）は、各腕の多峰 run が 1 run ずつで過半数の根拠が薄い。再現性の確認には run 数を増やした再測が要る

## 5. 申し送り・スコープ外

- 実測は完了（§3）。#3191 には §3.4 の 2 通りの水準を併記して反映する（本イシューでは #3191 へコメントしない）
- 二峰性の是正そのもの（既定値や本番経路の変更）は本イシューの対象外。原因が特定できたら別イシューで A/B 後に判断する
- 関連: `infer-reuse-phase-breakdown.md` §10・`metal-tape-build-infer-fixedcost.md`（#2114）・#3191（セル別の到達値の表）
