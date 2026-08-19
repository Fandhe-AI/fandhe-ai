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

対照カーネルの spread が概ね 5% を超えるサイズがある計測セッションは、A/B 判定の土台となる計測プロトコル自体が
まだノイズを十分抑えられていないとみなし、**A/B 判定へ進まない**（判定を無効化して中断する、安全側の設計）。
`crates/backend-metal/examples/gemm_swizzle_ab_bench.rs` のフェーズ 1（安定性セルフチェック）がこのゲートを実装し、
不成立時はフェーズ 2（swizzle A/B）をスキップして「判定不可」を出力する。

不成立の場合の調整手順: `crates/backend-metal/examples/gemm_swizzle_ab_bench.rs` の `ROUNDS`・`COOLDOWN`・
`MIN_WARMUP` 定数を**増やす方向のみ**調整して再実行する（減らす調整は spread 実測 green が条件。実装計画 §4.2）。

## 熱・電源状態の記録

計測実行前後に `pmset -g therm`（非特権・`sudo` 不要）でサーマル状態を記録する。`powermetrics` は `sudo` 必須の
ため使用しない（A03 インジェクション対策の一環でもあり、コード側にシェル呼び出しを埋め込まず手順書側の手動実行に
留める設計）。

```sh
pmset -g therm
cargo run -p backend-metal --example gemm_swizzle_ab_bench --release
pmset -g therm
```

## API 概要（`bench_harness::ab`）

- `AbConfig::new(rounds, cooldown, min_warmup)`: `rounds` は偶数・2 以上必須（fail-closed 検証）
- `run_stability(&AbConfig, &MeasurementConfig, workload) -> StabilityResult`: 単一ワークロードを `rounds` ラウンド
  計測し、ラウンド中央値の列と `relative_spread`（`StabilityResult::spread`）を返す
- `run_ab(&AbConfig, &MeasurementConfig, workload_a, workload_b) -> AbResult`: A/B を interleaved に計測し、
  各 side のラウンド中央値列・全体中央値・`b_over_a_ratio`（head/base 比）・各 side の spread を返す

いずれも `crate::protocol::run`（既存の warmup 20 回以上・計測 20 回以上・中央値/Q1/Q3 プロトコル）をラウンドごとに
呼ぶ上位ユーティリティであり、`guardrail`／`self-repair` が依存する `protocol::run`・`MeasurementConfig` の
セマンティクス自体は変更しない。

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
