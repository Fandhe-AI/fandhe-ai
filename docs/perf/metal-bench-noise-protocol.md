# Metal ベンチ計測プロトコルのノイズ対策（イシュー #746）

親イシュー #737（Metal 第 2 次最適化）配下。2026-08-19 の M4 Max 実機計測で、tgid swizzle（#540）の A/B と無関係な
対照カーネル（naive/tiled/simdgroup）が計測実行間で最大 70% 超変動し（256/512 で顕著・2048 のみ 2〜4%）、
`docs/perf/metal-gemm-tgid-swizzle-ab.md` の「劣化中央値 5% 以内」判定が成立しなかった。サーマル・GPU クロック
（DVFS）挙動が計測順序に系統的に乗ることが原因とみられる。本ドキュメントは、この系統誤差を抑える計測プロトコルの
設計と根拠を記録する。実装は `bench_harness::ab`（`crates/bench-harness/src/ab.rs`）。

## 設計方針

### 1. checkout 切替方式ではなく同一プロセス内 interleaved 比較

旧 A/B 手順（`docs/perf/metal-gemm-tgid-swizzle-ab.md` の旧版）は base（変更前コミット）・head（変更後コミット）を
`git checkout` で切り替えて別々に計測していた。base/head の計測が時間的に分離されるため、サーマルドリフト・DVFS
挙動が計測順序へそのまま系統誤差として乗る。

`bench_harness::ab::run_ab` は base/head（あるいは任意の A/B ペア）を**同一プロセス内**で interleaved に計測する。
Metal 側は `MetalGemm::new_with_swizzle(ctx, bool)` で swizzle off/on の 2 インスタンスを構築し、CUDA 側の
`CudaMmaGemm::new_with_swizzle`（`crates/backend-cuda/examples/gemm_mma_swizzle_bench.rs`）と同型の設計に揃えた。
これにより A/B 双方が同一プロセス・同一時間帯で計測され、コミット切替を伴わないため一時的なローカル変更
（コミット禁止制約の事故源だった `SWIZZLE_ENABLED` の一時トグル）も不要になる。

### 2. ラウンド交互（順序反転）による order-bias 相殺

1 回の計測を「A→B」の固定順で繰り返すと、後半に計測される側が先に計測される側よりサーマル上昇の影響を系統的に
多く受ける。`run_ab` はラウンドごとに A→B / B→A の順序を反転させる（`crates/backend-metal/examples/gemm_bench.rs`
の occupancy 比較・`ROUNDS=6` 偶数固定と同じ手法の再利用）。`AbConfig::rounds` は偶数必須で、奇数は
`BenchError::ProtocolViolation` として fail-closed に拒否する（A 先頭ラウンド数 = B 先頭ラウンド数が順序バイアス
相殺の前提のため）。

### 3. 時間ベースの追加ウォームアップ

`crate::protocol::run`（TASK-8.1）の `MeasurementConfig::warmup` は「回数」下限（20 回以上）のみを規定する。
小サイズのワークロードは 20 回の呼び出しがごく短時間で終わってしまい、GPU クロック（DVFS）が定常状態
（ブースト後の安定クロック）へ昇圧しきる前に計測へ入ってしまう懸念がある。`AbConfig::min_warmup` は
「最低経過時間まで追加ウォームアップを継続する」下限を提供し、`crate::protocol::run` 自体のセマンティクスは
変更せず、その前段で `min_warmup` 経過まで `workload` を呼び続ける（回数下限・時間下限のいずれか厳しい方を満たす
まで継続する設計。`crates/bench-harness/src/ab.rs::extended_warmup` 参照）。

### 4. 判定統計は変更しない・ばらつきは定量報告する

判定統計（中央値ベース）・許容誤差・#540 の既存採否判定基準（size 2048/4096 の中央値改善で採用、なければ revert）
は本イシューの範囲では変更しない（`.claude/rules/security.md`: ガードレール閾値・テスト許容誤差の変更は
ユーザー承認必須）。`bench_harness::relative_spread`（`(max − min) / median`）でラウンド間ばらつきを定量報告し、
`bench_harness::ab::run_stability` が対照カーネルの安定性セルフチェックとして使う（下記「安定性ゲート」参照）。
補助統計（`StabilityResult::aux` → `trimmed_spread_k1`／`iqr_spread`／`mad_spread`。イシュー #1483/#1484/#1485）は
**定量報告のみ**であり、判定へ転用してはならない（§8.6・`ab.rs` の `AuxiliarySpread` doc contract と同文意）。

### 5. 安定性ゲートと不成立時の中断規定

対照カーネルの spread が概ね 5%（`bench_harness::ab::STABILITY_SPREAD_GATE` が単一真実源。
`crates/bench-harness/src/ab.rs`）を超えるサイズがある計測セッションは、A/B 判定の土台となる計測プロトコル自体が
まだノイズを十分抑えられていないとみなし、**A/B 判定へ進まない**（判定を無効化して中断する、安全側の設計）。
`crates/backend-metal/examples/gemm_swizzle_ab_bench.rs` のフェーズ 1（安定性セルフチェック）が
`STABILITY_SPREAD_GATE` を参照してこのゲートを実装し、不成立時はフェーズ 2（swizzle A/B）をスキップして
「判定不可」を出力する。閾値を変更する場合は `STABILITY_SPREAD_GATE` の定義（コメント含む）と本節の両方を
更新すること（ガードレール閾値相当のためユーザー承認必須。`.claude/rules/security.md`）。

不成立の場合の調整手順: `crates/backend-metal/examples/gemm_swizzle_ab_bench.rs` の `ROUNDS`・`COOLDOWN`・
`MIN_WARMUP` 定数を**増やす方向のみ**調整して再実行する（減らす調整は spread 実測 green が条件。実装計画 §4.2）。
`within_gate` は `spread ≤ STABILITY_SPREAD_GATE` のみで決まり、補助統計（§4 参照）を `STABILITY_SPREAD_GATE`
と比較して判定に転用してはならない（イシュー #1485。§8.6・§8.8）。

### 6. 実行前の環境ガード（イシュー #1264・親 #1263）

#1186／#1187（Metal 転置ルーティング A/B）は、同一マシンで並走する他セッションの負荷（`uptime` 実測 load
average 3.4〜8.6・実行中の再上昇を `docs/perf/logs/metal-gemm-transpose-route-ab-1187/uptime_during_run4.txt`
で確認）により「5. 安定性ゲートと不成立時の中断規定」のゲートが 4 試行とも不成立のまま終わった。環境状態の
確認が手動記録（同ディレクトリの `env_info.txt`）に依存していたことが一因のため、`bench_harness::env_guard`
（`crates/bench-harness/src/env_guard.rs`）がその確認を機械化する **API 層**を提供する。load average・他
GPU プロセス検出・uptime 記録の取得と判定を行う設定型・取得関数・判定結果型を提供する（#1264）。
ガード不成立時のバックオフ再試行・env_info への記録出力・
`crates/backend-metal/examples/gemm_transpose_route_ab_bench.rs` への結線はイシュー #1265 で実装済み
（詳細は「7. ガード不成立時の再試行規定」節）。

## 7. ガード不成立時の再試行規定（イシュー #1265）

- **再判定するのは `EnvGuardReport::is_blocking()`（`overall == GuardVerdict::Fail`）の場合のみ**。
  `GuardVerdict::Undetermined`（取得不能）は記録のみで続行し、再試行しない（#1264 の「取得不能は未判定・
  ブロック要因にしない」契約を再試行にも一貫適用する）
- **待機間隔は増える方向のみ**: `RetryConfig`（`initial_wait`・`growth_factor`・`max_wait`・`max_attempts`）は
  `initial_wait * growth_factor^i` を `max_wait` で cap した非減少列を生成する。呼び出し側 example
  （`gemm_transpose_route_ab_bench.rs`）の既定値 `GUARD_INITIAL_WAIT=30s`・`GUARD_GROWTH_FACTOR=1.5`・
  `GUARD_MAX_WAIT=300s`・`GUARD_MAX_ATTEMPTS=10` の調整は `ROUNDS`／`COOLDOWN`／`MIN_WARMUP` と同様
  **増やす方向のみ**許容する
- **上限回数到達で中断**: `max_attempts` 回すべて `Fail` のままなら `BenchError::EnvGuardExhausted` を返す。
  呼び出し側 example は `verdict=undetermined` を出力して非ゼロ終了する（フェーズ 1 不成立時の既存経路と同じ
  `verdict=` grep 運用に揃える）
- **閾値は既定値なし**: `--max-load-avg` を指定しない実行は **record_only**（判定なし・記録のみ）で動作する。
  具体的な閾値の既定化は本イシューでも行わず（下記「#1265 向けの提案閾値」参照）、CLI 明示指定のみで有効化する

## 8. ロバスト統計の設計提案（イシュー #1266 提案・#1485 で確定（2026-09-09 ユーザー承認））

**本節は設計・提案のみであり、コード（`crates/bench-harness/src/stats.rs`・
`ab.rs`）は一切変更しない。ユーザー承認を得るまで統計量・閾値は現行の
`relative_spread`（`(max − min) / median`）・`STABILITY_SPREAD_GATE = 0.05`
のまま不変とする（`.claude/rules/security.md`: ガードレール閾値・テスト
許容誤差の変更はユーザー承認必須）。**

### 8.1 背景と前提

「5. 安定性ゲートと不成立時の中断規定」の統計量 `relative_spread` は
レンジ統計量（`(max − min) / median`）のため、n=10 ラウンド中 1 ラウンドの
外れ値だけでゲート（0.05）を容易に超える。#1186（1 試行）／#1187（4 試行）・#1255
（負荷環境 5 run）はいずれもほぼ全セルで不成立のまま、A/B 判定（フェーズ 2）
へ一度も到達していない（`metal-gemm-transpose-tiled.md` §5.2〜§5.8。#1253
の排他環境確保も 2 attempt とも `valid_runs=0` で分離計測データ自体が
得られていない）。

本節は「単発スパイクの影響を抑えるロバスト統計量」を候補として定義し、
既存実測データ（#1255 の 5 run・#1186/#1187 の 5 試行。計 10 セッション×
5 サイズ = 50 セル）へ再適用した場合にゲートが成立するかを定量化する。
**新規計測は行わない**。

**最重要事実（先に明記する）**: 下記のどの候補統計量を用いても、**セッション
単位のゲート成立は 10 セッション中 0**（`docs/perf/logs/metal-bench-robust-stats-1266/reapply.md`
「セッション単位の成立可否」節）。セル単位では raw 3/50 → 候補次第で
10〜18/50 まで改善するが、size=256 は**全候補で 0/10** のまま変わらない
（下記 §8.3）。つまり「単発スパイク」という前提は部分的にしか成り立たず
（例: `1255-run1`・size=1024 は 3 ラウンド連続の落ち込み＋1 ラウンドの
跳ね上がりという多峰性を示す。§8.3 の per-cell 表参照）、ロバスト統計は
`docs/perf/metal-gemm-transpose-tiled.md` §5.7〜§5.8 が示した GPU タイム
スタンプ分離計測（#1257 系）や排他環境確保（#1253）の代替にはならない。
承認だけでは #1267（フェーズ 2 完走）は解除されない。

### 8.2 候補統計量の定義と閾値との関係

すべて入力は `StabilityResult::round_medians_secs`（n = `AbConfig::rounds`。
現行 example は 10）。分母は**全系列の median-of-halves 中央値**
（`stats::median_q1_q3` と同一定義）に固定する。

| 案 | 定義 | 0.05 との関係 |
|---|---|---|
| A: トリム済みレンジ（k=1） | 昇順ソート後に上下各 k=1 を除外した残り n−2k のレンジ ÷ 全系列 median | 対称トリムは median-of-halves の選択要素を変えない（n=10→8: idx `round(4.5)`=5 と `round(3.5)`=4 が同一元要素を指す）ため分母不変。閾値 0.05 の意味が現行と最も近い |
| A′: トリム済みレンジ（k=2） | 同上 k=2 | n=10 の 40% を捨てる。ROUNDS 増（既に許容される調整方向）と組み合わせる前提 |
| B: Tukey フェンス IQR 除去後レンジ（1.5・3.0） | 全系列の Q1/Q3 から `[Q1 − c·IQR, Q3 + c·IQR]`（c=1.5 または 3.0）外を除外した残りのレンジ ÷ 全系列 median | 除去数が 0〜3 と可変（n=10 では分位点が粗い）。1.5 版と 3.0 版を併記 |
| C: IQR/median | `(Q3 − Q1) / median` | レンジより狭い量のため同じ 0.05 は**実質緩和**。ゲート置換には閾値の再導出（別承認）が必要 |
| D: 2·MAD/median | `2 · median(|x − median|) / median` | 同上。最も緩い |
| E: サイズ別部分ゲート（その他） | 統計量は変えず、成立したサイズのみフェーズ 2（A/B 判定）で判定する | 判定範囲を縮小するプロトコル変更。承認事項 |
| F: ROUNDS 増＋固定 k トリム（その他） | A と F（ROUNDS 増）の併用 | ROUNDS 増は承認不要、トリムは承認事項 |

### 8.3 実測データへの再適用結果

データセットは 2 系列を**分離して**扱う（合算しない）:

- **正確値**: #1255 の 25 セル（`1255-phase1_run{1..5}.log` の
  `phase1_round_stats` 行。`round_medians_secs` は本番コードの実測値
  そのもの・秒単位）
- **参考・近似値**: #1186 run1・#1187 run1〜4 の 25 セル（`round_tflops`
  行の 4 桁値から `secs = 1/tflops` で近似復元。丸め誤差により
  例えば `1187-run3`・size=256 はログ `spread=3.3070` に対し再計算
  `3.3016` とわずかにずれる。「参考」ラベル必須）

再現手順・スクリプトは `docs/perf/logs/metal-bench-robust-stats-1266/`
（`reapply.py`・`reapply.md`・`README.md`）。**以下の数値は
`reapply.md` からの転記**であり、乖離があれば `reapply.md` を正とする。

自己検証（`reapply.py --self-test`）: #1255 の 25 セルで raw spread
再計算値がログ `spread=` と相対誤差 1e-3 未満で一致することを確認済み
（25/25 pass）。

#### サイズ別成立数（分母: #1255 正確値のみ・5 run）

| size | raw | A(k=1) | A'(k=2) | B(1.5) | B(3.0) | C | D |
|---|---|---|---|---|---|---|---|
| 256 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 |
| 512 | 0/5 | 2/5 | 2/5 | 2/5 | 1/5 | 2/5 | 4/5 |
| 1024 | 0/5 | 1/5 | 2/5 | 2/5 | 1/5 | 2/5 | 3/5 |
| 2048 | 0/5 | 1/5 | 3/5 | 1/5 | 1/5 | 3/5 | 3/5 |
| 4096 | 1/5 | 3/5 | 4/5 | 3/5 | 2/5 | 4/5 | 4/5 |

#### サイズ別成立数（分母: #1255 正確値 + #1186/#1187 参考値込み・10 run）

| size | raw | A(k=1) | A'(k=2) | B(1.5) | B(3.0) | C | D |
|---|---|---|---|---|---|---|---|
| 256 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| 512 | 1/10 | 3/10 | 3/10 | 3/10 | 2/10 | 3/10 | 5/10 |
| 1024 | 0/10 | 1/10 | 2/10 | 2/10 | 1/10 | 2/10 | 4/10 |
| 2048 | 1/10 | 2/10 | 5/10 | 2/10 | 2/10 | 5/10 | 4/10 |
| 4096 | 1/10 | 4/10 | 5/10 | 3/10 | 2/10 | 5/10 | 5/10 |

#### セッション単位（5 サイズ全成立して初めてゲート通過）

全候補・全 10 セッションで**不成立（0/10）**（`reapply.md` 「セッション
単位の成立可否」節。個別セッション×候補の PASS/fail 内訳は同ファイル参照）。

#### セル単位合計（50 セル中）

`raw=3・A(k=1)=10・A'(k=2)=15・B(1.5)=10・B(3.0)=7・C=15・D=18`

#### 代表例（#1255 正確値。raw → A(k=1) → A'(k=2) → B(1.5) → C → D の順）

- `1255-run4`・size=1024: `0.8371 → 0.0184 → 0.0110 → 0.0230 → 0.0110 → 0.0139`
  （典型的な単発スパイク型。A で救済される）
- `1255-run1`・size=1024: `0.6265 → 0.4223 → 0.3992 → 0.6265（除去 0）→ 0.3992 → 0.0404`
  （3 ラウンド連続のレジーム切替。D 以外は救済不能）
- `1255-run1`・size=256: `0.6127 → 0.5799 → 0.1338 → 0.1703 → 0.1338 → 0.1154`
  （両側外れ値。全候補とも不成立）

全 50 セルの詳細は `docs/perf/logs/metal-bench-robust-stats-1266/reapply.md`
「per-cell 表」節を参照。

### 8.4 各案の利点・欠点・採否推奨

- **A（k=1）: 条件付き推奨**（ゲート統計量を置換するなら唯一の候補。
  最小変更・決定的・分母不変・0.05 の意味が現行に最も近い）。ただし
  本データではセッション成立 0/10 のため、**承認しても #1267（フェーズ 2
  完走）の解除には不十分**と明記する
- **A′（k=2）: 非推奨**（n=10 では 40% を捨てすぎる。ROUNDS ≥ 20 等へ
  増やした後に再検討する余地はある）
- **B（Tukey フェンス）: 非推奨**（除去数が可変・n=10 では分位点が粗く
  A と同等以下の効果しかない。Tukey の分位点定義が `median_q1_q3` に
  依存するため二重管理になる）
- **C・D: ゲート置換としては非推奨**（閾値 0.05 の意味が変わる＝閾値
  再導出という第 2 の承認が別途必要）。**判定に使わない補助レポート**
  （`phase1_round_stats` 行への追加キー等）としては推奨
- **E（サイズ別部分ゲート）: 検討価値あり**だが判定範囲の縮小という
  プロトコル変更のため承認事項。#1267 の設計判断に委ねる
- **F（ROUNDS 増＋トリム）: ROUNDS 増（承認不要）を先行**し、その上で
  A を再評価する順序を推奨

**総合推奨**: (1) 現行 `max − min`（レンジ）ベースの `relative_spread` を
維持する、(2) 補助統計（A・C・D）を判定には使わないレポート項目として
追加する後続イシューを先行させる、(3) A へのゲート置換は、セッション
成立 0/10 という本データの事実を踏まえ「排他環境データ（#1253 再試行）
または GPU タイムスタンプ分離計測（#1257 系）の結果を待ってから判断する」
と提案する。

**イシュー #1484 での先行実施**: `crates/bench-harness`（`StabilityResult::aux`・
`AuxiliarySpread`）は #1483 で追加済み。それを受けて #1484 が本節の C・D
（判定に使わない補助レポート）を出力側へ配線した: 呼び出し側 example 4 本
（`gemm_transpose_route_ab_bench.rs`・`gemm_swizzle_ab_bench.rs`・
`gemm_fine_barrier_ab_bench.rs`・`gemm_unroll_acc_ab_bench.rs`）の
`phase1_round_stats` 行へ `trimmed_spread_k1`（案 A・k=1 も併記）・
`iqr_spread`・`mad_spread` を追記し、`aggregate.py`・`1255-aggregate.py`
を後方互換な形で追従させた（`STABILITY_SPREAD_GATE`・判定式・既存キーは
不変。詳細は `docs/perf/metal-gemm-transpose-tiled.md` のキー一覧節）。
本節（§8.5）が挙げる A の**ゲート置換**（統計量そのものの置き換え）は
**不採用と確定**（再検討には新たな承認が必要。§8.6 参照）。

### 8.5 採用時の変更範囲（承認が下りた場合の見積もり。ゲート置換は不採用確定のため未実施のまま）

- `crates/bench-harness/src/stats.rs`: 新関数（例
  `trimmed_relative_spread(samples, trim_per_side)`）を追加し、
  `relative_spread` はそのまま残す。`n < 2k + 2` は `BenchError::ProtocolViolation`
  で fail-closed とする（`AbConfig::new` は `rounds=2` を許容するため、
  そのままではトリム後が空になりうる）
- `crates/bench-harness/src/ab.rs`: `StabilityResult`／`AbResult` への
  フィールド追加（補助レポート案）または `spread` の意味切替（置換案）・
  `STABILITY_SPREAD_GATE`（`ab.rs:57`）の doc comment・必要なら
  `STABILITY_TRIM_PER_SIDE` 定数の新設と `AbConfig::new` の rounds 下限
  検証の見直し
- 呼び出し側 example 4 本: `crates/backend-metal/examples/gemm_swizzle_ab_bench.rs`・
  `gemm_fine_barrier_ab_bench.rs`・`gemm_unroll_acc_ab_bench.rs`・
  `gemm_transpose_route_ab_bench.rs`（フェーズ 1 の該当箇所・
  `phase1_round_stats` 行契約）。`gemm_unroll_acc_ab_bench.rs`・
  `gemm_swizzle_ab_bench.rs` の `REGRESSION_TOLERANCE_RATIO = 1.0 − STABILITY_SPREAD_GATE`
  は閾値 0.05 据え置きなら数値は不変だが意味的に結合しているため
  変更時は要確認
- ログ集計スクリプト（`docs/perf/logs/metal-gemm-transpose-route-ab-1242/aggregate.py`・
  `1255-aggregate.py`・`1261-aggregate.py`）は `gate=`／`within_gate=`
  行をそのまま転記する設計のため、行契約を変更する場合は追従が必要
- docs: 本ファイルの §4・§5・API 概要節、`metal-gemm-transpose-tiled.md`、
  `CLAUDE.md`、各 A/B 記録（`metal-gemm-tgid-swizzle-ab.md`・
  `metal-gemm-fine-barrier-ab.md` 等の「`STABILITY_SPREAD_GATE=0.05`」
  記述）

### 8.6 承認結果（2026-09-09）

ルート #1468・Phase 親 #1472 のユーザー承認により、以下の 5 項目が確定した
（イシュー #1485）。tolerance／ガードレール変更ではなく「ゲート・閾値・
判定意味は不変のまま補助レポートを追加する」だけの変更であるため、
本件は `.claude/rules/security.md` のガードレール閾値・テスト許容誤差の
承認規約（変更を対象とする規約）の対象外である（(5) の根拠）。

1. **ゲート統計量は案 A（k=1）へ置換せず、現行 `max − min`（レンジ）ベースの
   `relative_spread` を維持する**
2. **置換は行わないため、閾値 `STABILITY_SPREAD_GATE = 0.05` も不変**
   （C・D の閾値再導出は不要・未実施のまま）
3. **補助統計（A・C・D）を判定に使わないレポート項目として先行追加する
   後続イシューは承認済み・実装完了**: `crates/bench-harness`
   （`StabilityResult::aux`・`AuxiliarySpread`・`AUXILIARY_TRIM_PER_SIDE = 1`）
   は #1483 で追加済み、呼び出し側 example 4 本（`gemm_transpose_route_ab_bench.rs`・
   `gemm_swizzle_ab_bench.rs`・`gemm_fine_barrier_ab_bench.rs`・
   `gemm_unroll_acc_ab_bench.rs`）の `phase1_round_stats` 行への
   `trimmed_spread_k1`／`iqr_spread`／`mad_spread` 追記・`aggregate.py`／
   `1255-aggregate.py` の表 D 追加は #1484 で実装済み（§8.4「イシュー #1484
   での先行実施」参照）
4. **案 E（サイズ別部分ゲート）は不採用**: #1267 が既にクローズ済みで
   あり、判定範囲を縮小するプロトコル変更を持ち込む先が存在しないため
5. **tolerance／ガードレール承認規約（`.claude/rules/security.md`）の対象外
   である根拠**: 本承認はゲート・閾値・判定意味（`within_gate` は
   `spread ≤ gate` のみで決まる）を一切変更せず、判定に使わない補助
   レポート項目を追加するのみであるため、性能ゲート閾値の変更を要する
   承認手続きの対象にはならない（§4・§5 の doc contract 明文化・#1485
   の M4 Max 実機記録は §8.8 参照）

### 8.7 再現手順

```sh
python3 docs/perf/logs/metal-bench-robust-stats-1266/reapply.py --self-test
python3 docs/perf/logs/metal-bench-robust-stats-1266/reapply.py \
  > docs/perf/logs/metal-bench-robust-stats-1266/reapply.md
```

**落とし穴**: Rust `f64::round` は half-away-from-zero、Python 組み込み
`round()` は銀行家丸め（round-half-to-even）。median-of-halves の
`idx = round(p · (n−1))` は n=10 で `p=0.5` のとき `4.5` となり、
Rust は 5・Python は 4 へ分岐しうる（例: `1255-run1`・size=256 で
素朴な Python `round()` 実装では 0.6170 になり、ログの 0.6127 と
食い違う）。`reapply.py` は `floor(x + 0.5)` で Rust 側の丸めを再現する
（詳細はスクリプト冒頭コメント）。

### 8.8 M4 Max 実機非後退確認（イシュー #1485）

§8.6 の承認結果を受け、`StabilityResult::aux`（#1483）・`phase1_round_stats`
行への補助キー出力（#1484）が実機で機能し、かつ `within_gate` 判定が
補助統計の影響を一切受けないことを M4 Max 実機で機械確認した
（コード変更なし。性能ゲート達成の確認は目的としない）。

**実行コマンド**:

```sh
cargo run -p fandhe-ai-backend-metal --example gemm_transpose_route_ab_bench \
  --release --features internal-diagnostics -- --phase1-only
```

`--max-load-avg` を付けないため環境ガードは `record_only`（判定なし・
待機なし）。

**環境**: Apple M4 Max（arm64）・共有負荷下（実行前 load average 1min
1.86 → 実行中 3.84 → 実行後 7.91。他セッション並走）。

**結果表**（`phase1_round_stats` 行の転記。詳細な機械検査結果は
`docs/perf/logs/metal-bench-robust-stats-1485/aggregate.md` を参照）:

| size | spread | gate | within_gate | trimmed_spread_k1 | iqr_spread | mad_spread |
|---|---|---|---|---|---|---|
| 256 | 2.1418e-01 | 5.0000e-02 | false | 1.9120e-1 | 1.3225e-2 | 1.9510e-2 |
| 512 | 5.9134e-01 | 5.0000e-02 | false | 4.7441e-1 | 5.2024e-2 | 7.1282e-2 |
| 1024 | 1.2784e-01 | 5.0000e-02 | false | 6.8816e-2 | 3.2653e-2 | 4.4504e-2 |
| 2048 | 3.6217e-01 | 5.0000e-02 | false | 1.9826e-1 | 1.4264e-1 | 4.3319e-2 |
| 4096 | 5.9607e-01 | 5.0000e-02 | false | 5.7022e-2 | 3.7271e-2 | 3.9314e-2 |

共有負荷下のため全 5 サイズで `within_gate=false`（想定内。ゲート成立
自体は本イシューの目的ではない）。

**判定不変の機械確認**: `phase1_round_stats` 行 5 件すべてで
`within_gate == (spread <= gate)` が一致し（補助統計が判定に影響して
いないことの直接証拠）、既存キー順（`size rounds spread gate within_gate
median_secs min_secs min_round_idx max_secs max_round_idx
round_medians_secs`）に続き `trimmed_spread_k1`／`iqr_spread`／
`mad_spread` の 3 補助キーがこの順で出力されることも確認した
（検査スクリプト・結果全文は `docs/perf/logs/metal-bench-robust-stats-1485/`
の `verify_1485.py`・`aggregate.md`・`README.md` を参照）。

**既存 `#[ignore]` 群・関連テストの非後退確認**（コード変更なしのため）:
`gemm_swizzle_bit_match`・`gemm_fine_barrier_bit_match`・
`gemm_transposed_parity` の実機 `#[ignore]` テスト（計 9 件）・example
単体テスト（`gemm_transpose_route_ab_bench`・`gemm_swizzle_ab_bench`・
`gemm_fine_barrier_ab_bench`・`gemm_unroll_acc_ab_bench`。計 76 件）・
`bench-harness` クレートテスト（16 passed・6 ignored〈実機依存分〉）・
`aggregate.py`／`1255-aggregate.py`／`reapply.py` の `--self-test` が
すべて非後退（全 pass。`docs/perf/logs/metal-bench-robust-stats-1485/
ignored_tests.log`）であることを確認した。

**ログ所在**: `docs/perf/logs/metal-bench-robust-stats-1485/`
（`phase1_only_run1.log`・`env_info.txt`・`uptime_{before,after}.txt`・
`pmset_therm_{before,after}.txt`・`ignored_tests.log`・`aggregate.md`・
`verify_1485.py`・`README.md`）。内部ホスト名・ユーザーパスは含めない。

## 熱・電源状態の記録

計測実行前後に `pmset -g therm`（非特権・`sudo` 不要）でサーマル状態を記録する。`powermetrics` は `sudo` 必須の
ため使用しない（A03 インジェクション対策の一環でもあり、シェル（`sh -c`）・`sudo` を使わず固定バイナリを直接
実行する設計。`env_guard` の外部コマンド起動〈`sysctl`・`ioreg`・`uptime`〉も同じ方針に従う。`pmset`／
`powermetrics` 自体は引き続き手順書側の手動実行に留める）。

```sh
pmset -g therm
cargo run -p fandhe-ai-backend-metal --example gemm_swizzle_ab_bench --release
pmset -g therm
```

## API 概要（`bench_harness::ab`）

- `AbConfig::new(rounds, cooldown, min_warmup)`: `rounds` は偶数・2 以上必須（fail-closed 検証）
- `run_stability(&AbConfig, &MeasurementConfig, workload) -> StabilityResult`: 単一ワークロードを `rounds` ラウンド
  計測し、ラウンド中央値の列と `relative_spread`（`StabilityResult::spread`）を返す
- `run_ab(&AbConfig, &MeasurementConfig, workload_a, workload_b) -> AbResult`: A/B を interleaved に計測し、
  各 side のラウンド中央値列・全体中央値・`b_over_a_ratio`・各 side の spread を返す。
  `b_over_a_ratio` は `median_b_secs / median_a_secs`（**実行時間の比**。1.0 未満なら B が速い）であり、
  TFLOPS 等スループット指標の head/base 比はその**逆数**になる点に注意（`ab.rs` の `AbResult::b_over_a_ratio`
  doc comment 参照。イシュー #746 PR #763 で取り違えによる判定逆転が指摘された）

いずれも `crate::protocol::run`（既存の warmup 20 回以上・計測 20 回以上・中央値/Q1/Q3 プロトコル）をラウンドごとに
呼ぶ上位ユーティリティであり、`guardrail`／`self-repair` が依存する `protocol::run`・`MeasurementConfig` の
セマンティクス自体は変更しない。

## API 概要（`bench_harness::env_guard`。`ab` から再公開。イシュー #1264）

実行前の環境確認を機械化する API 層（取得・判定を分離。設計は
`crates/bench-harness/src/env_guard.rs` モジュール doc 参照）。

- `EnvGuardConfig::new(max_load_avg_1min: f64) -> Result<Self, BenchError>`: 呼び出し側が明示するガード条件の
  検証付きコンストラクタ。**既定閾値・`Default` 実装は持たない**（ガードレール閾値相当のためユーザー承認事項。
  `.claude/rules/security.md`）。`with_gpu_process_watchlist(Vec<String>)`（名前部分一致。空なら記録のみ）・
  `with_max_gpu_device_utilization_percent(u8) -> Result<Self, BenchError>`（0〜100 検証）を builder 形式で追加設定する
- `EnvSample::collect() -> EnvSample`: load average・uptime・GPU プロセス（macOS のみ。`ioreg -r -c IOAccelerator -l`）
  を実測する I/O 層。全フィールドが `Option`（GPU は理由付き `Unavailable`）で、取得失敗を `panic`／`Err` にしない
- `EnvGuardConfig::evaluate(&EnvSample) -> EnvGuardReport`: 実測値へガード条件を適用する純粋関数（I/O なし）
- `EnvGuardConfig::check() -> EnvGuardReport`: `collect()` + `evaluate()` の合成
- `EnvGuardReport::is_blocking() -> bool`: `overall == GuardVerdict::Fail` のときのみ `true`。
  **取得不能（`GuardVerdict::Undetermined`）はブロック要因にしない**（イシュー #1264 本文の GPU 検出要件を
  load average にも一貫適用した設計）

判定規則: load average は `observed.one > max_1min` で `Fail`・取得不能で `Undetermined`。GPU は watchlist に
部分一致するプロセスがある、または使用率が上限超過で `Fail`。watchlist・上限とも未設定なら記録のみで `Pass`。
GPU 取得自体が不能（Linux 等）なら `Undetermined`。

### バックオフ再試行・env_info 記録 API（イシュー #1265）

- `RetryConfig::new(initial_wait, growth_factor, max_wait, max_attempts) -> Result<Self, BenchError>`:
  検証付きコンストラクタ（`max_attempts >= 1`・`initial_wait > 0`・`growth_factor` 有限かつ `>= 1.0`・
  `max_wait >= initial_wait`）。既定値・`Default` 実装は持たない
- `RetryConfig::wait_for_attempt(attempt_index: usize) -> Duration`: 待機列を返す純粋関数（非減少・`max_wait` cap）
- `run_guard_with_retry_with(&RetryConfig, check, sleep) -> Result<GuardRetryOutcome, BenchError>`:
  バックオフ再試行のコア実装（`check`／`sleep` を注入。I/O なしでユニットテスト可能）
- `run_guard_with_retry(&EnvGuardConfig, &RetryConfig) -> Result<GuardRetryOutcome, BenchError>`:
  実 I/O 版（`EnvGuardConfig::check` + `std::thread::sleep` の合成）
- `GuardRetryOutcome::record_only(EnvSample) -> Self`: 判定を行わない記録専用の単一試行結果を構築する
  （`--max-load-avg` 未指定時の record_only モード向け）
- `format_env_info_text(label, &GuardRetryOutcome, Option<&EnvGuardConfig>) -> String`:
  `env_info.txt` 準拠のテキストブロックを生成する純粋関数（I/O なし。GPU プロセスは件数と flagged 名のみ記録し、
  全プロセス名の列挙は行わない）

### #1265 向けの提案閾値（未承認・記録のみ）

`EnvGuardConfig::new` の `max_load_avg_1min` に既定値はなく、#1265（バックオフ再試行・結線）でも既定値化は行わず
CLI 明示指定（`--max-load-avg`）にのみ opt-in する形で実装した。参考として、
`crates/backend-cpu/src/thread_limit.rs::ThreadLimitReport`（大コア数判定。#1363）の実測値を踏まえ
「大コア数の 0.5 倍程度」を出発点とする案が考えられるが、**未承認・未検証のまま提案として記すに留める**。
承認後に既定値化するなら別イシューで行う。

## 適用対象・スコープ

本プロトコルは Metal の tgid swizzle A/B（`docs/perf/metal-gemm-tgid-swizzle-ab.md`）向けに整備したが、
`bench_harness::ab` 自体はバックエンド非依存（クロージャでワークロードを受け取る設計）であり、将来 CUDA 側の
A/B 計測（`crates/backend-cuda/examples/gemm_mma_swizzle_bench.rs` 等）へも適用しうる。本イシューでは Metal 側の
適用（`gemm_swizzle_ab_bench.rs`）のみを行い、CUDA 側の切り替えは別イシューのスコープとする。

## 実機実測・採否確定の状態

本ドキュメント・`bench_harness::ab`・`crates/backend-metal/examples/gemm_swizzle_ab_bench.rs` は Linux worktree で
整備した（Metal 実機が同一セッションで使用できないため）。実測（安定性セルフチェック・swizzle A/B・採否確定・
`docs/perf/metal-gemm-tgid-swizzle-ab.md` への記録）は Mac 実機セッションで消化する
（`docs/perf/metal-gemm-serpentine-ab.md`〈#536〉・PR #760 と同じ運用）。
