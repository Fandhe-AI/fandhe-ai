# CPU GEMM 小形状スレッドプール上限（イシュー #1575）

## 背景

低レイヤー診断（#1574。`docs/perf/lowlayer-diagnosis-2026-09-12.md` §3・
§7 A-1a）で、Apple M4 Max の batch 64 学習・推論（小形状 GEMM 主体）は
プロセス全体の `RAYON_NUM_THREADS` をグローバルプール既定値（16）から
`4` へ絞ると train 約 1.47 倍・infer 約 1.30 倍速いことが観測された。

既存機構（`crate::thread_limit`〈コア種別ベース。#1363/#1364 実測により
`BIG_CORE_LIMIT_ENABLED=false`〉・`GEMM_THREADING_THRESHOLD`/
`should_serialize`〈#811/#1027・`#[cfg(test)]` 限定・本番未結線〉）は
いずれも本目的に使えない（判定軸・粒度が異なる。設計理由の詳細は
`crates/backend-cpu/src/small_shape_thread_cap.rs` モジュール冒頭コメント）。

本イシューは **M×N×K の仕事量に基づき、M4 Max（自機判定）でのみ小形状
GEMM を小さい専用 rayon プールで実行する**新機構
（`crates/backend-cpu/src/small_shape_thread_cap.rs`）を、既存 2 機構を
置き換えずに新設する。

## 設計

`docs/spec/` 変更なし・依存追加なし・`unsafe` 追加なし。

- `SMALL_SHAPE_CAP_ENABLED`（単一 const ゲート。#1313
  `TWO_D_DYNAMIC_PRODUCTION_ENABLED` と同型）
- `should_cap(m, n, k)`: `m * n.max(N_CLAMP) * k < SMALL_SHAPE_CAP_MAX_WORK`
  の純関数判定
- `eligible_from_probe`: P/E 非対称構成（`hw.perflevel0.logicalcpu <
  hw.logicalcpu`）かつ `machdep.cpu.brand_string` が allowlist
  （`"Apple M4 Max"` のみ）に完全一致する場合のみ有効
- `run_capped`: 条件を満たす場合のみ専用 `rayon::ThreadPool`
  （`SMALL_SHAPE_CAP_THREADS` スレッド）へ `install` する。`RAYON_NUM_THREADS`
  明示設定時は上書きしない（`crate::thread_limit` と同じ契約）
- 結線箇所: `gemm_blis_parallel_with_transpose`／`gemm_blis_bias_act_parallel`
  の `dispatch_two_d_dynamic` 呼び出しのみ（forward・VJP の d_input/d_weight
  を含む本番全入口を網羅。`linear_forward_device`／`gemm_resident_rhs` 等は
  いずれかを内部で呼ぶため自動的にカバーされる）

bit 完全一致契約: 本機構は並列度のみを変え GEMM カーネル本体を変えない
ため、専用プール実行はグローバルプール実行と常に bit 完全一致する
（`small_shape_thread_cap::tests::dedicated_pool_execution_matches_global_pool_bit_exact`
で検証。既存の `gemm_blis_parallel_*_matches_naive_bit_exact_across_thread_pools`
系テストが検証する不変条件と同種）。

## 事前登録規則

実装着手前に固定した判定規則の全文はイシュー #1575 のコメントを正とする:
<https://github.com/Fandhe-AI/fandhe-ai/issues/1575#issuecomment-5644510967>

## Phase 0（M4 Max・定数確定スイープ）

**実施内容**: `examples/small_shape_cap_sweep.rs`
（`cargo run --release -p fandhe-ai-backend-cpu --example
small_shape_cap_sweep -- <global|dedicated:N>`）を用い、学習 5 形状
（64×256×784・64×10×256・784×256×64 相当・64×256×10 相当・
256×10×64 相当。実際の VJP 転置入口〈`gemm_blis_parallel_nt`/`_tn`〉では
なく同一 `(m,n,k)` の `gemm_blis_parallel`〈NN〉で計測している点に注意
——`should_cap` の判定は転置パターンに依存せず `(m,n,k)` の積のみで
決まるため、専用プール切替の性能効果を近似する上で妥当な代理計測とした）
＋交差確認用正方 128/256/512 を、腕 {`global`（既定 16 スレッド）・
`dedicated:{2,4,6,8}`・プロセス全体 `RAYON_NUM_THREADS=4`（参照）} ×
各 5 プロセス起動で計測した（record_only。専有ゲートなし。共有負荷下
`load average` 6.4〜9.1 → 計測後 4.7〜7.4。`docs/perf/logs/
cpu-gemm-small-shape-thread-cap-1575/`）。

**checksum**: 全腕・全形状で `f64` 逐次和のビット表現が完全一致
（`aggregate_phase0.md` 冒頭。bit 完全一致契約を計測レベルでも確認）。

**選択規則適用**: 学習 5 形状すべてで off 比 `ratio<=1.00` を満たしたのは
`dedicated:6` のみ（`dedicated:2`／`4` は `train_64x256x784_nn` で
`ratio>1.00`〈1.52／1.056〉・`dedicated:8` は同形状で `ratio=1.0272` に
より不成立）。成立候補が 1 つのみのため幾何平均最小の選択は自動的に
`dedicated:6`（幾何平均 ratio ≈0.62）に確定した。

→ **`SMALL_SHAPE_CAP_THREADS = 6`**

`dedicated:6` の交差確認形状 ratio は 正方 128: 0.7995・正方 256: 0.7902・
**正方 512: 1.0980（後退）**。事前登録の tie-break（学習最大仕事量
12,845,056 を含み・後退開始仕事量 134,217,728 を含まない 2 のべき乗の
うち最小＝最も保守的なもの）に従い `1 << 24`（16,777,216）を採用した
（`1 << 25` も条件を満たすが tie-break により不採用。正方 256
〈16,777,216〉はこの境界とちょうど等しく cap 対象外〈グローバルプール
のまま〉になるが、同形状は元々改善方向だったため後退はしない）。

→ **`SMALL_SHAPE_CAP_MAX_WORK = 1 << 24`（16,777,216）**

実測全文は `docs/perf/logs/cpu-gemm-small-shape-thread-cap-1575/
aggregate_phase0.md`・生ログ（`phase0_*.log`）・`env_info.txt` を参照。

Phase 0 完了後に上記 2 定数を確定・コミットし、Phase 1 の結果を見て
再調整しない（事前登録規則）。

## Phase 1（framework-compare・同一バイナリ on/off）

事前登録手順: `SMALL_SHAPE_CAP_ENABLED=true` へ切り替えたバイナリを
`env -u RAYON_NUM_THREADS`（after・cap 有効）／`RAYON_NUM_THREADS=16`
（before・明示設定により cap 無効）で interleave 計測する。

<!-- Phase 1 実測結果はここに追記する -->

## GB10（DGX Spark）

`cfg(target_os = "macos")` により構造的に非到達。Linux での
`eligible() == false` は `small_shape_thread_cap::tests::
small_shape_cap_report_bounds`（`#[cfg(not(target_os = "macos"))]` 分岐）
が構造的根拠として担保する。

<!-- GB10 実機到達時の健全性記録（record_only・判定には用いない）はここに追記する -->
