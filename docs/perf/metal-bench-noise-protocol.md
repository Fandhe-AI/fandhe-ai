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
