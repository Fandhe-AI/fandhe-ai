# CUDA readback 既定反転（PR #3552）の GB10 確認（イシュー #3227・第 3 回）

事前登録の記録規則は `RULE.txt`（計測前にコミット済み）。これは確認計測で、ADOPT／REJECT の判定は出さない
（#3225 の判定は ADOPT で凍結済み）。考察は `docs/perf/cuda-gemm-reuse-phase-breakdown.md` §15.4 に書く。

## 実行条件

- ノード: GB10（NVIDIA GB10・driver 580.173.02・aarch64・rustc 1.97.1）。実機ロック取得後に実施し、終了後に解放した。
- 実測 HEAD: `405e08358f0233a346b2abc494b407ec727d46c6`（PR head。`crates/*/src` は origin/main `b6579c3f5` と同一）。
  転送は git bundle。ノード側 `git rev-parse HEAD` と `.rev-stamp` が上記と一致し、`git ls-tree -r HEAD | sha256sum`
  （`4889542a…24d624e5`）がローカルの同一コミットと一致した。
- bin の sha256: `3974756ae778f8ea7be9126cc40797136ec168a162e5bd65a268a6278ec092e6`（`raw/manifest-*.json`）。腕の間でビルドし直していない。
- 専有ゲート: `record_only`。計測開始時の load1 は 9.64（直前の並列ビルドの余韻）、計測中の load1 は 2.77〜9.15
  （`raw/uptime-*.log`）、GPU 使用率 0%。**静かな環境での系列ではない**。腕と round の interleave で相殺される前提だが、
  絶対値（us）は他の記録と直接比べない。

## 1. 既定経路の確認（`readback_default_prefetch_3227`）

env 未設定（既定腕）と `FANDHE_AI_CUDA_READBACK_DEST=pretouched`（opt-out 腕）で、本番の `MemoryOps::download` を
各 numel で 5 回呼び、元データと `to_bits()` を全要素比較した（全て一致）。

| numel | 対象（1 MiB 以上） | 既定腕 hits_delta | 既定腕 cached_bytes | opt-out 腕 hits_delta |
|---|---|---|---|---|
| 37 | いいえ | 0 | 0 | 0 |
| 4099 | いいえ | 0 | 0 | 0 |
| 262144 | はい | 3 | 1048576 | 0 |
| 1048576 | はい | 3 | 5242880 | 0 |
| 4194304 | はい | 3 | 22020096 | 0 |
| 16777216 | はい | 3 | 89128960 | 0 |

- 既定腕の保持量の最大は 89128960 B（約 85 MiB。上限 256 MiB の内）。`CudaBackendOps::release_cached_device_memory()`
  の後は `cached_bytes == 0`（既定腕・opt-out 腕とも）。REQ-14 の解放が本番経路で働くことを実機で確かめた。
- `readback_prefetch_bit_match_3225`（診断入口で宛先を固定する既存テスト）も pass（hits_delta 3、freed 1048576〜67108864）。

## 2. 14 セルの確認計測（5 round・record_only）

before = `FANDHE_AI_CUDA_READBACK_DEST=pretouched`（旧既定）、after = env 未設定（新既定）。ratio = after/before。
表の全文は `summary-cells.md`、生データは `raw/`。checksum は全セル・全 run・両腕で完全一致。

| task | mode | size | before | after | ratio |
|---|---|---|---|---|---|
| gemm | fresh | 256 | 88.2 us | 87.8 us | 0.9955 |
| gemm | fresh | 512 | 261.7 us | 258.9 us | 0.9892 |
| gemm | fresh | 1024 | 954.0 us | 959.6 us | 1.0059 |
| gemm | fresh | 2048 | 4173.4 us | 4190.2 us | 1.0040 |
| gemm | fresh | 4096 | 39091.2 us | 22062.8 us | 0.5644 |
| gemm | reuse | 256 | 91.2 us | 91.7 us | 1.0052 |
| gemm | reuse | 512 | 557.1 us | 263.4 us | 0.4729 |
| gemm | reuse | 1024 | 2272.6 us | 973.3 us | 0.4283 |
| gemm | reuse | 2048 | 8832.5 us | 4233.3 us | 0.4793 |
| gemm | reuse | 4096 | 38810.5 us | 22673.6 us | 0.5842 |
| train | fresh | 64 | 529.1 us | 531.0 us | 1.0036 |
| train | reuse | 64 | 316.3 us | 316.0 us | 0.9991 |
| infer | fresh | 64 | 153.3 us | 154.1 us | 1.0052 |
| infer | reuse | 64 | 90.7 us | 90.0 us | 0.9918 |

**後退セル（ratio > 1.00。全て併記）**: gemm fresh N=1024（1.0059）・gemm fresh N=2048（1.0040）・gemm reuse N=256（1.0052）・
train fresh（1.0036）・infer fresh（1.0052）。いずれも 1.01 未満で、run 内比は 1 を挟んで符号が揺れる
（train fresh は 0.9524〜1.0155、infer fresh は 0.9565〜1.0189）。gemm reuse N=512 の run 内比に 1.0267 が 1 run ある
（残り 4 run は 0.46〜0.50）。N=256 はプール対象外でコード経路が同一。

### 計測手順上の記録（正直な申告）

- 初回の比較ステップで `compare_gemm_ab.py` が gemm・train・infer 混在の JSONL と N=256／512 を含む集合を扱えず
  「判定不能」を出した（生の計測は 140 run 全て成功・公開済み。`raw/` の JSONL が正）。再計測はしていない。
  集計は新設の `scripts/bench/framework-compare/summarize_readback_confirm.py` を同じ生 JSONL に適用して作った
  （スクリプトの比較ステップも同ツールへ差し替えた。計測ロジックは変更なし）。
- 判定語は使っていない。`compare_gemm_ab.py` は変更していない。

## 3. `#[ignore]` 全件（env 未設定・PR head）

「main との突き合わせ」の定義は `RULE.txt`。反転は origin/main に入っているため、PR head（`crates/*/src` は main と同一）を
主系列とし、FAIL は 3 段で切り分けた（単独再実行・`pretouched` で再実行・反転前 `c00d27a11` で実行）。
範囲外: hf-hub（ネットワーク）・backend-rocm（ROCm 非搭載）・onnx-interop・bench-harness・guardrail・self-repair（readback に届かない）。

| 範囲 | pass | FAIL | 備考 |
|---|---|---|---|
| backend-cuda（`--all-features`・setmaxnreg 4 件と 2 件の別契約テストを除く） | 465 | 13 | 下の分類 |
| backend-cpu・autodiff・tensor-core・facade（`--tests`） | 335 | 5 | 全て環境要因 |
| setmaxnreg 4 ファイル（1 つずつ `timeout 120`） | 4 | 0 | |
| facade `cuda_graph_step_bit_identity`（`FANDHE_AI_CUDA_GRAPH_STEP=1`）・`cuda_train_infer_sync_census_3189`（`CENSUS_STEPS=20 CENSUS_WINDOW=full`） | 3 | 0 | env 契約どおりの個別実行（`graph_step` の `eager_baseline` は env 未設定が契約で、別行程） |

最初の実行は `--no-fail-fast` なしで最初の失敗バイナリで止まったため、`--no-fail-fast` 付きで全件を取り直した
（旧ログは収録せず、この事実のみ記録）。

### FAIL の分類（件数とテスト名）

| 分類 | 件数 | テスト |
|---|---|---|
| 反転が原因 | **0** | なし（`pretouched` 指定でも同じ結果） |
| 既存の不具合（main 由来。反転前 `c00d27a11` でも `pretouched` でも同一に FAIL） | 8 | `mma_f16_k4096_stress`・`wmma_f16_k4096_stress`・`mma_tf32x3_matches_reference_across_shapes`・`mma_tf32x3_k4096_stress`・`gemm_tf32_optin_on_matches_cpu_across_shapes`・`gemm_tf32x3_optin_on_matches_cpu_across_shapes`・`wmma_f16_opt_k4096_stress`・`tensor_core_parity_record`（Tensor Core 経路の厳密判定。これらは baseline 非後退方式の対象で、`*_non_regression` 版は pass） |
| 並走による一過性（単独の `--test-threads=1` で pass） | 3 | `module_cache_wiring_tests::cuda_gemm_new_second_construction_reuses_module_cache`・`init_cost_diag_tests::init_cost_diag_gemm_new_lru_cold_vs_warm`・`small_shape_matrix_unit_has_no_floor_tflops_record`（CPU レーンと並走した負荷による時間・キャッシュ依存の判定。反転前でも単独で pass） |
| 環境要因 | 7 | cuda 2: `int8_mma_probe_exec_selected`・`sm121_isa_probe_exec_selected`（`*_PROBE_ID` の環境変数が契約）。cpu-lane 5: `graph_capture`・`cuda_train_infer_sync_census`（env 契約。上表のとおり env 付きで pass）、`poc_evidence_cross_check`・`gemm_parallel_matches_poc_evidence_within_tolerance`・`dump_case_512_layout_round_trip_and_matmul_matches_pytorch_verified_output`（bundle に `docs/spec` submodule を含めないため evidence ファイルが無い） |

集計は `ignored-summary/`（`test result:` 行の件数合計と FAIL 一覧。`classify-*` は切り分けの個別実行結果）。
FAIL 18 件の分類合計は 8 + 3 + 7 = 18 で、cuda 13 + cpu-lane 5 と一致する。
生ログ全文（約 260 MB）は収録していない。

## 収録物

- `RULE.txt`（計測前に固定）・`summary-cells.md`・`raw/`（JSONL・manifest・gate・uptime・skipped）・`ignored-summary/`
