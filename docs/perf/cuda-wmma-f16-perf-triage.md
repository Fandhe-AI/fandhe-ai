# CUDA WMMA(f16) 性能外れ値の診断・実測記録（イシュー #1123）

イシュー #1123「`wmma_f16_opt` の性能外れ値（dim=2048 で `mma_sync_f16` の約 1/5、
dim=4096 の転送のみ計測に二峰性）」の診断記録。診断専用テスト
`crates/backend-cuda/tests/wmma_f16_opt_perf_triage.rs`（`CudaWmmaGemm::launch_f16_basic`。
`crates/backend-cuda/src/gemm_wmma.rs`）による GB10 実機実測（2026-09-03）と、その結果を
受けた既存テスト是正内容を記す。**tolerance・カーネル・ディスパッチ規則は変更しない**
（`.claude/rules/coding-rust.md`）。

## 状態: 実機実測完了（2026-09-03・DGX Spark GB10・sm_121・GPU アイドル。
是正後テスト〈§4.3〉も実機実測済み。イシュー #1160（2026-09-04）で
`tensor_core_tflops_record` の f16 assert の比較対象を「本番選択結果に
追従する方式」へ整備し、`MMA_PRIORITY_PRODUCTION_ENABLED = true`（mma
優先）を一時的に有効化した状態での GB10 実機実測では本 assert が pass
することを確認した（§8）。**本番結線（`MMA_PRIORITY_PRODUCTION_ENABLED
= true`）は PR #1179 codex-review 指摘〈K=4096 非後退ゲートの `MmaF16`
baseline ceiling 未承認〉により `false` へ差し戻し済みであり、この
本番既定のまま本テストを実機実行すると比較対象が `wmma_f16_opt` へ
戻るため本 assert は依然として red である**（`wmma_f16_opt` カーネル
単体 4.391〜4.496 TFLOPS が tiled f32 カーネル単体 6.776〜6.790 TFLOPS
を下回る。§4.3 と同一数値）。本テストは実機 `#[ignore]` 分離のため
通常 CI には現れず、この既知 red は baseline ceiling 承認・
`MMA_PRIORITY_PRODUCTION_ENABLED` の `true` 復帰まで持ち越しである。
詳細は `docs/perf/cuda-gemm-auto-f16-mma-switch.md` §0・
`docs/perf/cuda-parity-baseline.md` §12.6 追記を参照。以下 §8 の記述は
`MMA_PRIORITY_PRODUCTION_ENABLED = true` を一時的に有効化した際の
実測記録として維持する）

## 1. 症状（発端）

`crates/backend-cuda/tests/dispatch_boundary.rs::large_shape_mma_pipeline_vs_wmma_tflops_record`
の GB10 実機実行で、以下が観測された。

- dim=2048: `wmma_f16_opt` 3.60 ms・`mma_sync_f16` 0.80 ms（2 回とも同値で再現的。約 4.5 倍差）
- dim=4096: 転送のみ計測 0.263 s／0.275 s に対し、`wmma` 合算計測 0.30 s・`mma` 合算計測が
  1 回目 0.0185 s・2 回目 0.260 s という **二峰性**（同一プロセス内の連続 2 回実行で
  約 14 倍のばらつき）

旧計測プロトコル（`run_f16` による転送込み合算計測から、`clone_htod`/`alloc_zeros`/
`clone_dtoh` のみの「転送のみ」計測を差し引いて「計算のみ」時間を出す方式。PR #258 由来）
では、この外れ値が (a) カーネル本体の性能問題か、(b) 転送・アロケーション側
（`cudarc` の `cuMemAllocAsync` プールのトリムや unified memory のページマッピング等）に
起因するプロトコル外要因かを切り分けられなかった。

## 2. 診断方法

`wmma_f16_opt_perf_triage.rs`（`#[ignore]`・診断専用。受け入れ判定には使わない）で、
以下 3 系統を同一実行内で計測した（warmup 20・計測 20。TASK-8.1 準拠）。

1. **カーネル単体**: `CudaWmmaGemm::upload_f16`／`alloc_output_f16`／`launch_f16`
   （opt 優先。イシュー #1123 で追加した `launch_f16_basic` は基本版カーネルを強制）／
   `synchronize`（`CudaMmaGemm` も同名 API）による、H2D/D2H・バッファ確保を計測区間の
   外に置いた「launch → synchronize」のみの計測。
2. **転送込み**: 既存 `run_f16`（比較用。旧プロトコルとの対比のために残す）。
3. **転送のみ**: `clone_htod`/`alloc_zeros`/`synchronize`/`clone_dtoh` のみを
   dim ごとに **連続 3 回** 計測（二峰性が転送・アロケーション側由来かを可視化する）。

対象形状は #1123 の観測形状 2048/4096 に加え、外れ値の出現境界を探るため
512/768/1024/1536 も含めた（`DIMS = [512, 768, 1024, 1536, 2048, 4096]`）。

## 3. GB10 実測結果（sm_121・2026-09-03・warmup 20/iters 20）

### 3.1 カーネル単体（常駐バッファ・launch+synchronize）TFLOPS

| dim | wmma_f16_opt | wmma_f16_basic | mma_sync_f16 |
|---|---|---|---|
| 512  | 4.106 (0.0654 ms) | 5.966 (0.0450 ms) | 17.015 (0.0158 ms) |
| 768  | 9.284 (0.0976 ms) | 7.402 (0.1224 ms) | 30.707 (0.0295 ms) |
| 1024 | 8.915 (0.2409 ms) | 7.616 (0.2820 ms) | 38.904 (0.0552 ms) |
| 1536 | 8.758 (0.8275 ms) | 8.103 (0.8945 ms) | 49.512 (0.1464 ms) |
| 2048 | 7.135 (2.4077 ms) | 7.513 (2.2868 ms) | 51.732 (0.3321 ms) |
| 4096 | 4.664 (29.4674 ms) | 5.183 (26.5188 ms) | 50.411 (2.7264 ms) |

括弧内は `kernel_only_ms`（中央値）。

### 3.2 転送込み `run_f16`（ms・中央値）

| dim | wmma_f16_opt | mma_sync_f16 |
|---|---|---|
| 512  | 0.1136 | 0.0651 |
| 768  | 0.1857 | 0.1103 |
| 1024 | 0.4391 | 0.1856 |
| 1536 | 1.5945 | 0.4148 |
| 2048 | 3.7680 | 0.8060 |
| 4096 | 299.9533 | 269.0792 |

### 3.3 転送のみ（3 回連続計測・ms・中央値）

| dim | 1 回目 | 2 回目 | 3 回目 |
|---|---|---|---|
| 512  | 0.0553 | 0.0555 | 0.0552 |
| 1024 | 0.1347 | 0.1347 | 0.1343 |
| 2048 | 0.4529 | 0.4531 | 0.4538 |
| 4096 | 260.8906 | 262.9407 | 262.7663 |

### 3.4 既存テスト（是正前・転送込みプロトコル）の実測

- dim=2048: `wmma_f16_opt` 3.60 ms／`mma_sync_f16` 0.80 ms（2 回とも同値）
- dim=4096: 転送のみ 0.263 s／0.275 s、`wmma` 合算 0.30 s、`mma` 合算が
  1 回目 0.0185 s／2 回目 0.260 s（二峰性）

## 4. 結論

### 4.1 `wmma_f16`（opt ≈ basic）の恒常的な低速性

`wmma_f16_opt` と `wmma_f16_basic` はほぼ同水準（§3.1）で、全 dim にわたり
`mma_sync_f16`（`mma.sync`/`ldmatrix`/`cp.async` パイプライン）を恒常的に下回る。倍率
（`mma_sync_f16` ÷ `wmma_f16_opt`。§3.1 のカーネル単体 TFLOPS 実測値から算出）は
**形状依存で約 3〜11 倍**の範囲に分布する: 512 で約 4.1 倍・768 で約 3.3 倍・1024 で
約 4.4 倍・1536 で約 5.7 倍・2048 で約 7.3 倍・4096 で約 10.8 倍と、dim が大きいほど
倍率も拡大する傾向を示す。この差は dim=2048 固有ではなく（512〜4096 の全域で一貫して
`mma_sync_f16` が優位）、JIT キャッシュ・実行順にも非依存（カーネル単体計測はプロセス内の
起動順序に関わらず安定した値を示した）であることを確認した。opt カーネルの共有メモリ
最適化（`wmma_f16_opt`）は f16 経路では `wmma_f16_basic` に対して有意な改善効果を
示さない（一部 dim では basic の方が速い）。

**本番経路への到達性**: `fandhe_ai_backend_cuda::ops::BackendOps::gemm`（本番ディスパッチ）
は f32 tiled カーネル固定であり、`wmma_f16`／`mma_sync_f16` のいずれにも到達しない。
`CudaGemmAuto::run_f16`（f16 auto 経路。`crates/backend-cuda/src/gemm_auto.rs`）の
`KernelKind::MatrixUnit` 判定分岐（`select_f16_matrix_unit_impl`）自体は、cc >= 8.0
かつ整列形状（`validate_mma_alignment`／`validate_mma_grid_bounds` 充足）で
`mma_sync_f16`（`CudaMmaGemm`）を、それ以外では `wmma_f16`（`CudaWmmaGemm`）を
優先する `docs/dispatch-rules-design.md` §5.6 の mma 優先設計どおり実装済み（#1156）。
この優先順位は `gemm_auto::MMA_PRIORITY_PRODUCTION_ENABLED` でゲートされており、
イシュー #1160 の GB10 実機実測（`docs/perf/cuda-gemm-auto-f16-mma-switch.md`）で
非後退を確認した一方、mma 優先の本番有効化（`true`）は K=4096 非後退ゲートの
`MmaF16` baseline ceiling 未承認（PR #1179 codex-review 指摘）により
`false`（wmma 優先・#1156 以前と同じ従来経路）のまま保留している（`run_f16` は
現状 cc・整列形状に関わらず `wmma_f16` を優先して呼ぶ）。ただし `CudaGemmAuto` 自体は
本イシュー #1160 の時点でも facade／`BackendOps::gemm`／`bench-harness` のいずれ
からも呼ばれておらず本番未結線のまま（`gemm_auto.rs::new` doc コメント参照。
`BackendOps::gemm` は引き続き f32 tiled カーネル固定）である。この
`CudaGemmAuto` 自体の facade への結線は本イシュー・#1156 いずれのスコープ外
（ディスパッチ規則自体は変更しない）。TFLOPS 正式記録・`wmma_f16_opt` の維持・
格下げ判断は #1160 が §8 で扱った（結論: フォールバック限定で維持。§8「7. の
決定」参照）。

### 4.2 dim=4096 のプロトコル検査失敗（二峰性）の原因

旧プロトコル（「転送のみ計測を合算計測から差し引く」方式。§1）の破綻は、dim=4096 で
1 個あたり 32 MB 超（f16・4096×4096×2 byte）のバッファを **毎回新規に** `clone_htod`/
`alloc_zeros`/`clone_dtoh` する per-call アロケーション＋転送が、260 ms 前後・かつ
実行ごとに大きくばらつく（二峰性を示す）病態を持つためと判明した。

**確認できた事実**: §3.3 の転送のみ計測（3 回連続）は dim=4096 で 260.89／262.94／
262.77 ms と 3 回とも安定しており（dim=2048 の 0.4529〜0.4538 ms 比で約 580 倍）、
転送のみ計測単体では二峰性は見られない。二峰性が観測されたのは §3.4 の旧テスト
（`mma_sync_f16` の転送込み `run_f16` 合算計測）のみで、同一プロセス内の連続 2 回実行で
1 回目 0.0185 s・2 回目 0.260 s という大きな乖離を示した。

**未特定の仮説**: 上記の事実からは、二峰性の原因がカーネル本体ではなく `cudarc` の
アロケーション・転送経路（`cuMemAllocAsync` プールのトリムや unified memory の
ページマッピング等、状態依存で高速経路が使われる場合とそうでない場合がある可能性）に
あることが示唆されるが、具体的な機序は本イシューでは未特定である。この病態自体の
追加調査は別イシュー（#1130。§6）へ引き継ぐ。

### 4.3 是正後テスト（§5 の常駐バッファ・カーネル単体プロトコル）の GB10 実機実測

2026-09-03・sm_121・GPU アイドル環境で、是正後の 2 テストを実機実行した。

**`dispatch_boundary.rs::large_shape_mma_pipeline_vs_wmma_tflops_record`（kernel_only）:
pass**（本関数に受け入れ条件の assert はなく、実測記録のみ）。

| dim | wmma_f16_opt (TFLOPS) | mma_sync_f16 (TFLOPS) | mma_over_wmma |
|---|---|---|---|
| 2048 | 7.463 | 52.070 | 6.977 |
| 4096 | 4.414 | 55.513 | 12.577 |

`mma_sync_f16` の優位性（dim2048 で約 6.98 倍・dim4096 で約 12.58 倍。§4.1 の初回
診断計測〈dim2048 で約 7.3 倍・dim4096 で約 10.8 倍〉とおおむね整合し、dim が
大きいほど倍率が拡大する傾向も一致する）を、是正後のカーネル単体プロトコルでも
再確認した。

**`tensor_core_real_device.rs::tensor_core_tflops_record`（kernel_only、4096）:
TF32 assert は pass、f16 assert（f16 > tiled f32）は FAIL**。

| path | 1 回目 (TFLOPS) | 2 回目 (TFLOPS) |
|---|---|---|
| tiled_f32 | 6.790 | 6.776 |
| wmma_tf32_staged | 14.095 | — |
| wmma_f16_opt | 4.496 | 4.391 |

TF32 経路（`wmma_tf32_staged`。M=N=K=4096 は整列形状のため staged 経路が選択された）は
tiled f32 を約 2.1 倍上回り pass。一方 f16 経路（`wmma_f16_opt`）は tiled f32
（6.776〜6.790 TFLOPS）を **下回り**（4.391〜4.496 TFLOPS）、assert は red。

**記録する判断**: 旧プロトコル（合算計測から転送のみ計測を減算する方式）は、f16 の
転送バイト量が f32 の半分であることにより「計算のみ」時間を過大評価し、見かけ上 pass
していた（§3.4 の旧テストは「転送のみ ≥ 合算」プロトコル検査自体が §4.2 の病態で
不安定だったため、この過大評価が顕在化しないまま推移していた）。カーネル単体（転送を
計測区間の外へ完全に排除したプロトコル）では、GB10 の `wmma_f16_opt` は tiled f32 を
恒常的に下回ることが判明した（§4.1 の `wmma_f16`≈`wmma_f16_basic` が `mma_sync_f16` を
約 3〜11 倍〈形状依存〉下回るという結論と整合する: tiled f32 との比較でも
`wmma_f16_opt` は優位性を持たない）。

この f16 assert（`tensor_core_tflops_record` 内 `f16_kernel_tflops > tiled_kernel_tflops`）
は `.claude/rules/coding-rust.md`「性能下限・最適化の達成を理由に…緩和しない」と同じ
方針で **緩和せず red のまま維持**する。本番 f16 Tensor Core 経路を `mma.sync`
パイプラインへ結線するイシュー #1131（§6）の受け入れ条件へ本 assert の pass を引き渡し、
**#1131 完了時に本 assert が pass することを #1131 の完了条件とする**。TF32 assert は
pass のため #186 への引き渡し対象外（TF32 経路自体に既知の問題はない）。

## 5. 実施したテスト是正

`.claude/rules/coding-rust.md`「性能下限・最適化の達成を理由に…境界チェックを省略しない」
「バックエンド間数値一致テストの許容誤差を単独で緩和しない」のいずれにも抵触しない
（tolerance・カーネル・ディスパッチは無変更）。

1. **`crates/backend-cuda/tests/dispatch_boundary.rs::large_shape_mma_pipeline_vs_wmma_tflops_record`**:
   常駐バッファ（`CudaWmmaGemm`／`CudaMmaGemm` の `upload_f16`／`alloc_output_f16`／
   `launch_f16`／`synchronize`）による「launch → synchronize」のみのカーネル単体計測へ
   切り替えた。転送のみ計測の差し引き・「転送のみ ≥ 合算」のプロトコル整合性検査
   （§4.2 の病態により破綻していた）を廃止した。計測記録 JSON（`BenchReport` の name）・
   stdout ラベルはいずれも `_kernel_only` を付け、旧記録（転送込み値）と区別できるようにした。
   本関数は転送込みの参考値・「転送のみ ≥ 合算」検査に依存する受け入れ条件を持たないため、
   assert 自体の変更はない。
2. **`crates/backend-cuda/tests/tensor_core_real_device.rs::tensor_core_tflops_record`**:
   同様に `CudaGemm::upload_f32`／`alloc_output_f32`／`launch_tiled_f32`／
   `launch_wmma_tf32`、`CudaWmmaGemm` の同名 API による常駐バッファ・カーネル単体計測へ
   切り替えた。#64 受け入れ条件の 2 つの `assert!`（Tensor Core 経路が tiled f32 を
   上回ること）は判定式を変更せず、比較対象の量を「計算のみ」（旧: 合算計測 − 転送のみ
   計測）から「カーネル単体」（新: `launch → synchronize` 区間のみ）へ差し替えた
   （**assert 自体は緩和していない**。判定対象量の定義変更であり、判定式・意図
   〈Tensor Core 経路が tiled f32 を上回る〉は不変）。dtype 別の転送のみ計測・減算・
   「転送のみ ≥ 合算」プロトコル整合性検査はいずれも廃止した（転送そのものを計測区間の
   外へ置いたため、転送バイト数差の補正〈PR #258〉自体が不要になった）。
3. **`crates/backend-cuda/tests/wmma_f16_opt_perf_triage.rs`**（新規・診断専用）と
   `CudaWmmaGemm::launch_f16_basic`（`crates/backend-cuda/src/gemm_wmma.rs`。基本版
   カーネルを強制する診断専用の最小 `pub` 入口。本番ディスパッチには影響しない）は
   診断専用として維持する。**PR #1132 codex-review P2 指摘対応**: 診断専用ランチャーを
   恒久的な公開 API として無条件公開しないよう、`launch_f16_basic` は `internal-
   diagnostics` feature（既定 off。`Cargo.toml` の `[features]`）でゲートし、
   `wmma_f16_opt_perf_triage` テストは `Cargo.toml` の `[[test]]` セクションで
   `required-features = ["internal-diagnostics"]` を指定する（`specialized_mma_
   parity` 等の既存パターンと同一）。実行は
   `cargo test -p fandhe-ai-backend-cuda --features internal-diagnostics --test
   wmma_f16_opt_perf_triage -- --ignored --nocapture`。

### 5.1 #64 受け入れ条件（カーネル単体比較）の実測結果

`tensor_core_tflops_record`・`large_shape_mma_pipeline_vs_wmma_tflops_record` の是正後版を
GB10（sm_121・2026-09-03・GPU アイドル）で実機実行した。実測値・記録する判断は §4.3 の
とおり。要約:

- `large_shape_mma_pipeline_vs_wmma_tflops_record`: **pass**（受け入れ条件の assert なし）
- `tensor_core_tflops_record`: TF32 assert **pass**（`wmma_tf32_staged` 14.095 TFLOPS
  > tiled f32 6.790 TFLOPS）。f16 assert **FAIL**（`wmma_f16_opt` 4.391〜4.496 TFLOPS
  が tiled f32 6.776〜6.790 TFLOPS を下回る）

f16 assert は `.claude/rules/coding-rust.md` の tolerance 非緩和方針に従い red のまま
維持し、緩和せずイシュー #1131（本番 f16 Tensor Core 経路を `mma.sync` パイプラインへ
結線する設計判断・§6）の受け入れ条件へ引き渡す（#1131 完了時に本 assert が pass する
ことを #1131 の完了条件とする。`tensor_core_real_device.rs` 内の該当コメント参照）。

実機実行コマンド:

```sh
cargo test -p fandhe-ai-backend-cuda --test tensor_core_real_device -- --ignored --nocapture tensor_core_tflops_record
cargo test -p fandhe-ai-backend-cuda --test dispatch_boundary -- --ignored --nocapture large_shape_mma_pipeline_vs_wmma_tflops_record
```

### 5.2 記録の上書き範囲

本ドキュメントは `docs/perf/dispatch-boundary-measurement.md`・
`docs/perf/cuda-tensor-core-measurement.md`（両ファイルとも `tests/dispatch_boundary.rs`・
`tests/tensor_core_real_device.rs` の `#[ignore]` 理由文字列が転記先として引用している）
の **大形状（2048/4096）行のみ** を、本ドキュメントの実測（カーネル単体プロトコル）で
上書きする。小形状（128〜512。`small_shape_matrix_unit_has_no_floor_tflops_record`）・
`tensor_core_parity_record`（複合判定）は対象外（プロトコル変更なし）。

## 6. 切り出し先（スコープ外事項の追跡）

`.claude/rules/out-of-scope-tracking.md` に従い、以下はユーザー承認済みで別イシューへ
切り出し済み。

- **大容量バッファ per-call アロケーション病態の調査**（#1130。#1102 配下）: §4.2 の
  dim=4096 二峰性の調査は
  `docs/perf/cuda-large-buffer-percall-alloc-transfer-threshold.md`（#1146。P0〜P5 の
  同期漏れ是正〈コミット `dcf08e4`〉後のテストコードで GB10 実機再実測を完了済み
  〈2026-09-03・同ドキュメント §4.7〉）で行った。GPU 非関与の純ホスト計測（P6）で
  確認した **glibc mmap しきい値（既定 32 MiB）がホスト側の主因であること自体**、
  および `cuMemAllocAsync` プール（H-B）の棄却・H2D 側（H-C）の棄却・「D2H 転送先の
  未タッチ `Vec` に限定される」という発症箇所の特定は、いずれも sync 是正後の GB10
  再実測で確定できた（同ドキュメント §5・§6 結論 1〜3）。一方、この閾値効果が
  #1123 の元症状（dim4096 合算・約 261〜263 ms）の**主因であるという結論自体**は、
  dim4096 相当規模での直接再現・§4.4 の降順走査限定の確率的スパイクの発生条件特定が
  未完了のため、事実と切り分けた**未確定の仮説**として残る（同ドキュメント §5 H-A・
  §6 結論 3）。加えて、32 MiB 以上のフェーズ通過直後の降順走査でのみ確率的に発生する
  追加スパイク（二峰性そのもの）自体は、GPU 非関与の純ホスト計測（P6）ではなく
  GPU 関与の D2H 転送（P4・宛先が毎回新規 `Vec`）側で観測されており（同ドキュメント
  §4.4。P6 は sync 是正後の 3 ラン全てで安定）、かつ sync 是正前後で「どのフェーズが」
  「どの頻度で」発生するかのパターン自体が一致しないことが同ドキュメント §4.4 で
  明記されている。したがって二峰性の発生条件は本ドキュメント時点でも未確定のままであり、
  確定した実測記録として扱ってはならない（発生条件の特定は §7 の引き継ぎ対象）。
- **`mma_sync_f16`（`CudaMmaGemm`）の性能改善の本番結線検討**（#1131。#1007 配下）:
  §4.1 で確認した `mma.sync`/`ldmatrix`/`cp.async` パイプラインの約 3〜13 倍（形状依存。
  dim2048 で約 7 倍・dim4096 で約 11〜13 倍。§3.1 の初回診断計測・§4.3 の是正後実測
  いずれも同傾向）の優位性を、
  f16 auto 経路（`CudaGemmAuto::run_f16`）へ結線するかどうかの設計判断。#1156 で
  `select_f16_matrix_unit_impl` の判定ロジック自体は結線済み（cc>=8.0・整列形状で
  `mma_sync_f16` を優先する設計。§4.1 参照）。完了条件（2026-09-03 GB10 実機実測後に
  追加）: `tensor_core_real_device.rs::tensor_core_tflops_record` の f16 assert
  （§4.3 で FAIL）が pass すること。**#1160（2026-09-04 GB10 実機実測）で f16 assert
  の比較対象を本番経路が実際に選ぶ実装から動的に決定する方式へ差し替え、
  `MMA_PRIORITY_PRODUCTION_ENABLED` を一時的に `true` にした状態で pass することを
  確認した（§8）が、mma 優先の本番有効化自体は K=4096 非後退ゲートの `MmaF16`
  baseline ceiling 未承認（PR #1179 codex-review 指摘）により `false`（wmma 優先）へ
  差し戻し済みである。完了条件（assert が pass すること）は「本番選択結果に追従する
  比較方式」の整備として満たされたが、mma 優先自体の本番有効化は baseline ceiling
  承認まで保留のままである。**

## 7. 関連ファイル

- `crates/backend-cuda/tests/wmma_f16_opt_perf_triage.rs`（診断専用テスト）
- `crates/backend-cuda/src/gemm_wmma.rs::CudaWmmaGemm::launch_f16_basic`（診断専用 API）
- `crates/backend-cuda/tests/dispatch_boundary.rs::large_shape_mma_pipeline_vs_wmma_tflops_record`（是正）
- `crates/backend-cuda/tests/tensor_core_real_device.rs::tensor_core_tflops_record`（是正）
- `docs/perf/dispatch-boundary-measurement.md`・`docs/perf/cuda-tensor-core-measurement.md`（大形状行の上書き対象）

## 8. f16 経路切替前後の TFLOPS 記録・f16 assert pass 転換・`wmma_f16_opt` の扱い確定（#1160）

イシュー #1160「f16 経路切替前後の TFLOPS を dispatch_boundary 系で記録し
`tensor_core_tflops_record` の f16 assert pass と `wmma_f16_opt` の扱いを
確定する」の実測記録。GB10（sm_121・CUDA 13.0・2026-09-04・GPU アイドル。
`nvidia-smi --query-compute-apps`／`utilization.gpu` を計測前後に確認）。

### 8.1 カーネル単体（本イシューでの追加計測なし。§3.1 の既存実測を再引用）

§3.1 の GB10 実機実測（2026-09-03。イシュー #1123）は本イシューの結線判断で
そのまま参照した（同一プロトコル・カーネルソース無変更のため再計測は行って
いない）。2048/4096 の `wmma_f16_opt` 対 `mma_sync_f16` は §4.1 記載のとおり
（2048: 約 7.3 倍・4096: 約 10.8 倍）。

### 8.2 転送込み auto 経路 A/B（`CudaGemmAuto::run_f16`。独立 5 回起動・run-median）

`docs/perf/cuda-gemm-auto-f16-mma-switch.md`「実測表」に記録した（本ファイル
では二重管理しない）。要約: 512/1024/2048 は after（mma 優先）が base（wmma
優先）を 1.75〜4.67 倍上回り、4096 は #1130 の per-call アロケーション病態
（本節参照）の影響で base・after とも run-median が低く出るが after の
run-median は base の 5 run 範囲内に収まった（後退なし）。

### 8.3 `tensor_core_tflops_record` の f16 assert pass 転換

`tests/tensor_core_real_device.rs::tensor_core_tflops_record` の f16 計測を、
`CudaWmmaGemm`（`wmma_f16_opt`）直接計測から「本番 f16 経路が実際に選ぶ
実装」（`CudaGemmAuto::mma_available()` に応じて `CudaMmaGemm` または
`CudaWmmaGemm` をカーネル単体計測。GB10 では `mma_available()=true` のため
`mma_sync_f16`）へ差し替えた（`internal-diagnostics` feature 込みで選択器
`f16_matrix_unit_impl` との整合性を fail-closed 検査）。GB10 実機実測結果:

| path | TFLOPS |
|---|---|
| tiled_f32（基準） | 10.013 |
| wmma_tf32_staged | 14.134 |
| wmma_f16_opt（参考行・assert 対象外） | 4.522 |
| mma_sync_f16（本番 f16 経路・assert 対象） | 55.449 |

f16 assert（`production_f16_kernel_tflops > tiled_kernel_tflops`）は
55.449 > 10.013 のため **pass**（`test result: ok. 1 passed`）。TF32 assert
も引き続き pass（14.134 > 10.013）。この計測は
`MMA_PRIORITY_PRODUCTION_ENABLED = true`（mma 優先）を一時的に有効化した
状態で行ったものである。イシュー #1131 の完了条件（本 assert が pass
すること）は「本番選択結果に追従する比較方式」の整備としては満たされたが、
本番既定（`MMA_PRIORITY_PRODUCTION_ENABLED = false`。PR #1179 codex-review
指摘により差し戻し済み）のまま実機実行した場合は比較対象が `wmma_f16_opt`
へ戻り、本 assert は red のままである（上記状態ブロック・§6 参照）。

### 8.4 `wmma_f16_opt` の扱いの決定: フォールバック限定で維持（コード変更なし）

- **根拠 1**: GB10 カーネル単体で opt≈basic（§4.1・§3.1 参照。512/2048/4096
  は basic が僅かに速く、768〜1536 は opt が僅かに速い。一貫した優劣なし）。
  共有メモリ最適化は f16 で効果を持たないが、後退もしていない
- **根拠 2**: 結線後（`MMA_PRIORITY_PRODUCTION_ENABLED = true`。その後 baseline
  ceiling 未承認により `false` へ差し戻し済み。上記状態ブロック参照）の WMMA 経路
  は「mma 事前形状ゲート非充足（`n % 8 != 0` または `k % 8 != 0`・grid 上限
  超過）」「cc 7.x（mma 非対応）」「`CudaMmaGemm` NVRTC 失敗」時の
  フォールバックとして本番到達性を保つため、証跡専用への格下げ（到達不能
  化）はできない
- **根拠 3**: `ParityPath::WmmaF16` baseline 行は opt 実効経路の実測値であり
  （`docs/perf/cuda-parity-baseline.md`）、`CudaWmmaGemm::run_f16` の既定を
  basic へ変える・opt を削除するには baseline 再実測とユーザー承認が必要
  （本 PR の権限外）
- **結論**: `wmma_f16_opt` は「フォールバック経路として維持（本番 f16 auto
  経路の主経路からは外れる）・追加最適化投資は行わない」。将来の opt 廃止／
  basic 一本化は承認付き別イシューの候補として PR 本文に記録する（本 PR で
  Issue は起票しない）

### 8.5 4096 転送込みの既知病態（#1130・スコープ外）

4096 は base・after いずれの転送込み計測（§8.2）でも #1130（本ファイル §4.2
「dim=4096 のプロトコル検査失敗（二峰性）の原因」）の per-call アロケーション
病態の影響を受け run 間ばらつきが大きい。本イシューはこの病態の原因調査・
修正を行わない（#1130 のスコープのまま）。

### 8.6 関連ファイル（本節分）

- `crates/backend-cuda/src/gemm_auto.rs`（本節実測当時は `MMA_PRIORITY_PRODUCTION_ENABLED
  = true` で計測。baseline ceiling 未承認のため現在は `false` へ差し戻し済み）
- `crates/backend-cuda/tests/tensor_core_real_device.rs`（f16 計測の本番経路追従）
- `crates/backend-cuda/tests/gemm_auto.rs`（`f16_matrix_unit_impl_reports_selected_implementation` 期待値更新）
- `scripts/bench/run_gemm_auto_f16_mma_switch_bench.sh`（`BIN` の `CARGO_TARGET_DIR` 対応是正）
- `docs/perf/cuda-gemm-auto-f16-mma-switch.md`（転送込み auto 経路 A/B 実測表）
- `docs/perf/cuda-parity-baseline.md` §12.6（route-aware テストの既知 red 相互参照）
