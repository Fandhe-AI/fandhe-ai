# CUDA 実機 `#[ignore]` テスト実行・結果記録（#389）

イシュー #389（親トラッキング #388）の実施記録。`crates/backend-cuda/tests/` の実機依存 `#[ignore]`
テストを DGX Spark GB10 で初めて実行し、結果を記録する。後続（#390 floor 実測 / #391 起動コスト /
#392 ピークメモリ / #393 REQ-8 下限再確定）が前提とする「数値一致が実機 green」の実態を確定する。

実行手順の正本は [`real-hardware-verification-env.md`](./real-hardware-verification-env.md) であり、
本ファイルでは二重管理しない。ここでは環境実測値・結果サマリ・失敗と対処・未解決事項のみを記す。

## 1. 実行環境

| 項目 | 値 |
|------|-----|
| ノード | `<cuda-node>`（DGX Spark GB10。実ホスト名は `docs/real-hardware-verification-env.local.md` 参照） |
| GPU | NVIDIA GB10（sm_121） |
| compute capability（`CudaDevice::compute_capability()`） | (12, 1) |
| arch（`CudaDevice::arch()`） | `compute_121` |
| driver | 580.159.03 |
| CUDA / NVRTC | `/usr/local/cuda`・nvcc release 13.0 V13.0.88 |
| OS | Ubuntu 24.04 aarch64 |
| rustc | 1.97.0 (2d8144b78 2026-07-07) |
| cargo | 1.97.0 (c980f4866 2026-06-30) |
| 検証対象 commit | `720bf633e12471526a31dbe632a86bbe2150a8f4`（`.rev-stamp` 一致確認済み） |
| 実施日 | 2026-08-10 |
| GPU 占有状況（実行前後） | `utilization.gpu` 0%。常駐サービス 2 プロセスのみ（実名・使用量はローカル版 `docs/real-hardware-verification-env.local.md` 参照。`nvidia-smi --query-compute-apps`） |

## 2. 実行コマンド

実行前に `export CUDA_NODE="<実ホスト名>"`（山括弧のクォート必須。未クォートだと `export` 自体がリダイレクトと誤解釈される。`docs/real-hardware-verification-env.md` 冒頭の注記・
`docs/real-hardware-verification-env.local.md` 参照）を設定する。

```sh
ssh "$CUDA_NODE" 'cd ~/work/rust-ai-library-run && \
  env PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH \
      CARGO_TARGET_DIR=$HOME/work/target-rust-ai-library \
  cargo test -p backend-cuda --release --locked --no-fail-fast -- --ignored --nocapture'
```

`--locked` でノード側の lockfile 暗黙変更を禁止（`deps-policy.md`）。`--no-fail-fast` で全 17 バイナリを
1 回の実行で完走させ、S3（件数突合）・S4（失敗分岐）を一括で確認できるようにした。

## 3. 件数突合: 「74 件」ではなく 51 件が実体

イシュー #389 タイトルの「74 件」は `#[ignore]` という**文字列**の grep 件数（属性以外に、ファイル冒頭の
`//!` 概要コメント・関数ドキュメンテーションコメント中の `` `#[ignore]` `` という言及も一致してしまう）。
実際に `#[test]` に付与された `#[ignore = "..."]` 属性は次の通り 51 件で、`cargo test` のサマリ
（`passed + failed` の合計。`filtered out` は非 ignored テストで今回対象外）とも一致する。

```
grep -rnE '^\s*#\[ignore' crates/backend-cuda/tests/*.rs | wc -l   # => 51
```

| 内訳 | 件数 |
|------|------|
| grep `#\[ignore` 全マッチ（doc コメント中の言及を含む） | 74 |
| うち `#[test]` 属性としての `#[ignore = "..."]`（実行対象） | **51** |
| `cargo test -p backend-cuda --release --locked --no-fail-fast -- --ignored --nocapture` 実行件数（全 17 バイナリの `passed+failed` 合計） | **51**（突合一致） |

数字を合わせるための後付け調整は行っていない。「74」はテストファイル群の実装過程で `#[ignore]` という
語がドキュメンテーションコメントにも多用された結果であり、イシュー起票時の grep が doc コメントも
拾ったことによる過大カウントと判断する。

## 4. 結果サマリ

17 テストバイナリ・51 件中 **42 件 pass・9 件 fail**（`--no-fail-fast` 単発実行時点。5 節で個別に検証）。

| テストファイル | 件数 | pass | fail | 備考 |
|---|---|---|---|---|
| `backend_ops_real_device.rs` | 1 | 1 | 0 | |
| `cpu_cuda_mma_parity.rs` | 3 | 2 | 1 | `mma_f16_k4096_stress`（parity） |
| `cpu_cuda_parity.rs` | 2 | 2 | 0 | |
| `cpu_cuda_wmma_parity.rs` | 3 | 2 | 1 | `wmma_f16_k4096_stress`（parity） |
| `device.rs` | 1 | 1 | 0 | |
| `device_init.rs` | 1 | 1 | 0 | |
| `dispatch_boundary.rs` | 2 | 2* | 0* | 直列再実行で pass（5.1 節） |
| `gemm_auto.rs` | 2 | 2 | 0 | |
| `gemm_mma.rs` | 5 | 5 | 0 | 初回実行は 1 件 fail →本 PR で修正済み（5.2 節） |
| `gemm_naive.rs` | 3 | 3 | 0 | |
| `gemm_tiled.rs` | 6 | 6* | 0* | 直列再実行で pass（5.1 節） |
| `gemm_wmma.rs` | 2 | 2 | 0 | |
| `gemm_wmma_f16_opt.rs` | 4 | 3 | 1 | `wmma_f16_opt_k4096_stress`（parity） |
| `gemm_wmma_tf32.rs` | 4 | 2 | 2 | `wmma_tf32_k4096_stress_poc_v2_5`・`wmma_tf32_matches_reference_across_shapes`（parity） |
| `gemm_wmma_tf32_opt.rs` | 5 | 2〜3* | 2〜3* | parity 2 件は恒常 fail。性能アサーション 1 件は直列で pass（5.1 節） |
| `memory_real_device.rs` | 5 | 5 | 0 | |
| `tensor_core_real_device.rs` | 2 | 1〜2* | 0〜1* | `tensor_core_parity_record` は恒常 fail。`tensor_core_tflops_record` は並列競合フレーキー（5.1 節） |
| **合計** | **51** | **42（単発）** | **9（単発）** | 直列再実行後の恒常 fail は 8 件（すべて parity） |

`*` は複数回実行で結果が揺れた項目。詳細は 5 節。

## 5. 失敗と根本原因・対処

許容誤差（`backend_cpu::assert_parity` の複合判定「相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満」）・
性能アサーションの定数は**一切変更していない**（`git diff` に該当箇所なし。`.claude/rules/coding-rust.md`・
`security.md` 準拠）。

### 5.1 性能・プロトコルアサーション: 並列実行の GPU 競合が原因（コード変更なし）

同一バイナリ内の複数 `#[test]` は既定で並列実行され、`bench_harness::protocol::run`（warmup 20・計測 20）
を使うテストが同時に GPU を叩くと計測が歪む（計画 S4 の一次対処）。以下 4 テストは単発の `--no-fail-fast`
実行では fail したが、`--test-threads=1` での直列再実行では pass した。

| テスト | 単発（並列）実行 | 直列（`--test-threads=1`）再実行 |
|---|---|---|
| `gemm_tiled.rs::tiled_f32_outperforms_naive_at_4096` | fail（speedup=0.280x、要求 1.1x 以上） | **ok**（10.85s） |
| `dispatch_boundary.rs::small_shape_matrix_unit_has_no_floor_tflops_record` | fail（転送のみ計測が合算計測を下回らない） | **ok**（43.27s、同バイナリのもう 1 件と合わせて 2/2 pass） |
| `dispatch_boundary.rs::large_shape_mma_pipeline_vs_wmma_tflops_record` | fail（同上） | **ok**（同上） |
| `gemm_wmma_tf32_opt.rs::wmma_tf32_opt_exceeds_tiled_f32_tflops_at_4096` | fail（0.259/0.250 TFLOPS、tiled f32 1.187〜1.237 TFLOPS を下回る） | **ok**（8.92s） |

いずれも GPU 占有状況を `nvidia-smi --query-compute-apps` で再確認したうえで直列再実行しており、常駐
サービス（実名はローカル版 `docs/real-hardware-verification-env.local.md` 参照）以外のプロセスは介在していない。実機の性能・計測プロトコル自体の問題では
なく、**同一テストバイナリ内の並列実行による GPU 時間分割が原因**と判断する。コード変更は行っていない
（テスト自体の並列度制御は本イシューのスコープ外。並列実行下での安定計測が必要な場合は別途 issue で
`#[test]` の直列化属性付与を検討する。7 節参照）。

**tiled f32 基準値の突合（バイナリ間で約 5 倍の乖離。数値の信頼性に関する重要な注記）**: 上表の
`wmma_tf32_opt_exceeds_tiled_f32_tflops_at_4096` は `gemm_wmma_tf32_opt.rs` 単体バイナリ内で計測した
tiled f32 基準値が 1.187〜1.237 TFLOPS だったのに対し、`docs/perf/cuda-tensor-core-measurement.md`
「TFLOPS 実測」節が記録する `tensor_core_real_device.rs::tensor_core_tflops_record`（別バイナリ）の
tiled f32 計測値は 0.189〜0.233 TFLOPS で、同じ M=N=K=4096 形状にもかかわらず約 5 倍乖離している。
`tensor_core_tflops_record` は同一バイナリ内に `tensor_core_parity_record`（GPU を使う別テスト）を
併載しており、直後の段落で述べる `tensor_core_tflops_record` 自体のフレーキー性（並列実行間で
pass/fail が入れ替わる）と整合する形で、この低い方の値も同根の**同一バイナリ内並列実行による GPU
時間分割**で歪んだ計測である可能性が高い。したがって `docs/perf/cuda-tensor-core-measurement.md` が
導出する「対 PoC-v2-3 約 10〜13%」という評価は、より高い基準値（1.187〜1.237 TFLOPS、対 PoC-v2-3 比
約 65〜68%）と比較すると実態を過小評価している可能性があり、**現時点でどちらの値が「実機の実性能」
に近いかを本イシューでは確定できない**（直列実行下での再測定が必要。5.1 節の他 4 件と同じ理由で
#390／#391 に引き継ぐ）。

**#390 での突合結果（結論）**: #390（`docs/perf/cuda-floor-remeasurement.md`「tiled f32 @4096 の
バイナリ間乖離の突合結果」節）が、並列競合のない単一プロセス逐次実行の `cuda_floor_bench` で
3 回反復計測した結果、tiled f32 @4096 は **1.9729〜1.9817 TFLOPS**（中央値 1.9775 TFLOPS）と、
上記いずれの既存値よりも高い値を記録した。これは「並列実行による GPU 時間分割が低い方の値
〈0.189〜0.233 TFLOPS〉を歪めた」との推定を裏付けるが、直列再実行値（1.187〜1.237 TFLOPS）との
約 1.6〜1.7 倍の残差は未解明のまま #391（起動コスト計測）に引き継がれている。詳細・要因分析の
候補は `docs/perf/cuda-floor-remeasurement.md` 側を正本とし、本ファイルでは二重管理しない。

`tensor_core_real_device.rs::tensor_core_tflops_record` はさらにフレーキーで、複数回の単発（並列）実行間で
pass/fail が入れ替わった（1 回目 fail・3 回目 pass。直列実行でも 1 回 fail を観測）。転送のみ計測（大きな
f32 バッファの H2D）と合算計測（カーネル込み）の相対関係が実行間で安定しないためで、上記 4 件と同根の
計測タイミング競合と考えられる。恒常的な parity 失敗ではないためこの節に含めるが、他 3 件ほど確実に
「直列化で解消する」とは言い切れない。#391（起動コスト計測）で計測プロトコルの頑健性ごと再検証する
必要がある（7 節）。

### 5.2 修正した 1 件: `mma_f16_zero_dim_shape_returns_empty_without_launch`（テスト側のバッファ長不備）

初回実行で `InvalidShape { detail: "b length mismatch: expected 64 (k*n), actual 8" }` により fail。
`CudaMmaGemm::run_f16` は m==0/n==0 no-op 形状であっても `validate_gemm_dims`（a.len()==m*k・
b.len()==k*n の厳密一致検証）を早期 return より先に呼ぶ契約（`crates/backend-cuda/src/gemm_mma.rs`
run_f16 のドキュメンテーションコメント参照。`gemm_wmma.rs::run_f16` 等と共通の一般契約）。同種テスト
（`gemm_wmma.rs::wmma_f16_zero_dim_shape_returns_empty_without_launch`）は k*n・m*k ぴったりの
バッファ長を渡しているのに対し、`gemm_mma.rs` の当該テストのみバッファ長が不足していた
（m=0,n=8,k=8 の no-op で b に 8 要素しか与えていなかった。必要な k*n=64 要素に修正）。

これはホスト側検証ロジック（`validate_gemm_dims`・`run_f16` の検証順序）のバグではなく、**テスト自身の
入力データ不備**であるため、コミット `720bf633e12471526a31dbe632a86bbe2150a8f4` で
`crates/backend-cuda/tests/gemm_mma.rs` のみを修正した。`validate_gemm_dims` の検証順序・契約・
許容誤差は変更していない。修正後は当該ファイル 5 件すべて pass（4 節参照）。

### 5.3 数値一致（parity）失敗 8 件: 恒常的、許容誤差は変更せず #186 へ引き渡し

以下 8 件は単発・直列いずれの実行でも恒常的に fail する。`backend_cpu::assert_parity`（REQ-2 統一複合
判定の唯一の実体）は編集していない。

| テスト | 形状 | fail_count | mean_abs_diff | mean_rel_err |
|---|---|---|---|---|
| `cpu_cuda_mma_parity.rs::mma_f16_k4096_stress` | 256×256×4096 | 101/65536 (0.15%) | 7.646e-5 | 3.071e-5 |
| `cpu_cuda_wmma_parity.rs::wmma_f16_k4096_stress` | 256×256×4096 | 99/65536 (0.15%) | 7.562e-5 | 1.653e-5 |
| `gemm_wmma_f16_opt.rs::wmma_f16_opt_k4096_stress` | 256×256×4096 | 81/65536 (0.12%) | 7.627e-5 | 1.987e-5 |
| `gemm_wmma_tf32.rs::wmma_tf32_k4096_stress_poc_v2_5` | 256×256×4096 | 10647/65536 (16.2%) | 4.476e-3 | 1.441e-3 |
| `gemm_wmma_tf32.rs::wmma_tf32_matches_reference_across_shapes` | 32×32×32（形状網羅の最小ケースで既に fail） | 154/1024 (15.0%) | 3.698e-4 | 7.736e-4 |
| `gemm_wmma_tf32_opt.rs::wmma_tf32_opt_k4096_stress` | 512×512×4096 | 43019/262144 (16.4%) | 4.463e-3 | 1.459e-3 |
| `gemm_wmma_tf32_opt.rs::wmma_tf32_opt_matches_reference_across_shapes` | 64×64×64 | 699/4096 (17.1%) | 5.676e-4 | 1.742e-3 |
| `tensor_core_real_device.rs::tensor_core_parity_record` | 512×512×512（TF32） | 42493/262144 (16.2%) | 1.574e-3 | 1.489e-3 |

**f16 経路（3 件）**: fail_count が全要素の 0.12〜0.15% に留まり、K=4096 ストレスケースでのみ発生する
（形状網羅テストは全 pass）。`mean_abs_diff` は 7.6e-5 前後と小さく、大 K での丸め誤差蓄積の裾（tail）が
複合判定の閾値をわずかに超える事例と考えられる。

**TF32 経路（5 件）**: fail_count が 15〜17% と f16 の 100 倍以上に大きく、K=4096 ストレスだけでなく
32×32×32・64×64×64 という WMMA タイル 1〜2 個ぶんの最小形状でも fail する。`mean_abs_diff` は f16 の
約 60 倍（4.5e-3 台）。`docs/cuda-tensor-core-design.md` 93 節・`tests/gemm_wmma_tf32.rs` 冒頭コメントは
「統一複合判定は TF32 前提の複合指標として改定済み」としているが、本実機実測はこの前提が現行閾値
（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）では TF32 経路に対して不十分であることを示している。

**参照実装（テスト側 CPU 計算）は意図的に非量子化のままである点の確認**: `tests/gemm_wmma_tf32.rs`・
`tests/gemm_wmma_tf32_opt.rs`・`tensor_core_real_device.rs::tensor_core_parity_record` はいずれも TF32
経路の参照値を `backend_cpu::matmul_reference_fma`（入力を f32 のまま、TF32 相当の 10bit 仮数丸めを
適用せずに計算する参照実装）で得ている（f16 経路が `tests/cpu_cuda_mma_parity.rs` 等で「f16→f32→
参照計算→f16 丸め→f32 化」と GPU 側の丸めを参照側にも反映させているのとは対照的）。これは実装漏れ
ではなく、`docs/perf/cuda-tensor-core-tolerance-evaluation.md`（#186 実測。RTX 3060・compute capability
8.6 実機）が既に理論的・実測的に示した結論と整合する意図的な設計である: 同ドキュメント §3.1 のとおり
TF32 経路は「入力 A・B をそれぞれ TF32 に丸めてから積和し、累算は FP32 で行う」ため、GEMM 全要素中
約 15〜16.5%（本実測の 16.2〜17.1% と同水準）が現行閾値を恒常的に外れることを最小形状（32×32×32）
から確認済みであり、絶対誤差救済閾値を実測ベースで引き上げる REQ-2 改定候補（同ドキュメント §4）を
既に提示している。したがって参照側を TF32 相当に量子化して緩和する対処は「テスト側参照モデルの
バグ修正」ではなく「複合判定を TF32 丸め後の値同士の比較に置き換える」という**判定基準そのものの
変更**に相当し、`.claude/rules/security.md`「ガードレール閾値・テスト許容誤差の変更は必ず人間の承認を
経る」の対象になる。本実機実測（DGX Spark GB10・sm_121）は #186 の実測環境（RTX 3060・compute
capability 8.6）と異なる世代の Tensor Core でも同様の fail 率（15〜17% 台）が再現することを追加で
確認した点に価値があり、#186 が示した「REQ-2 改定が必要」という結論を補強する新規データとして
引き渡す。

いずれも `.claude/rules/coding-rust.md`「バックエンド間数値一致テストの許容誤差を単独で緩和しない」・
`security.md`「テスト許容誤差の変更は必ず人間の承認を経る」に従い**許容誤差・アサーションを一切変更
せず**、実測値を本節に記録したうえで #186（Tensor Core 経路の数値一致閾値の実測再評価。
`docs/cuda-tensor-core-design.md` 10 節・93 節・106 節が実測後の再評価用サブタスクとして既に切り出し
済み）へ引き渡す。**受け入れ条件「実機テスト全件 pass」は本イシューの範囲では未達のまま確定する**
（TF32 経路 5 件・f16 K=4096 tail 3 件の計 8 件は #186 が既に示した閾値不足に起因する既知の恒常
fail であり、本イシューのスコープ内〈テスト実行・結果記録〉での修正対象ではない。REQ-2 改定
（閾値変更）が完了するまで解消しない）。

## 6. `#[ignore]` 分離が通常 CI で機械的に効いている根拠

`docs/backend-metal-real-device-testing.md` と同型の確認。Mac（CUDA 非搭載）・CI（self-hosted・CUDA
toolkit 非搭載、`.claude/rules/ci.md`）双方で `cargo test -p backend-cuda`（`--ignored` なし）を実行すると、
本ファイルが対象とする 51 件はすべて `#[ignore]` によりスキップされ、環境適応スモークテスト
（`*_parity_smoke_env_adaptive` 等）のみが実行される。これらは `CudaDevice::new` が
`CudaError::DriverUnavailable`／`CudaError::NvrtcUnavailable` を返す分岐で早期 return し green になる
契約（各テストファイル冒頭のドキュメンテーションコメント参照）。本イシューの実行はこの契約を変更しない。

## 7. 未解決事項・エスカレーション先

- **#186**（Tensor Core 経路の数値一致閾値の実測再評価）: 5.3 節の 8 件（f16 K=4096 tail 3 件・TF32 5 件）
  の実測値一式を引き渡す。`docs/perf/cuda-tensor-core-tolerance-evaluation.md`（#186。RTX 3060・compute
  capability 8.6 実機）が既に「TF32 経路は最小形状から現行閾値を外れる・REQ-2 改定が必要」と結論して
  おり、本実測（DGX Spark GB10・sm_121）はこの結論を異なる Tensor Core 世代で再現した追加データと
  して引き渡す（最初の発見ではない。5.3 節参照）
- **#390**（floor 実測）: 5.1 節の tiled f32 @4096 バイナリ間乖離の突合を実施済み（5.1 節「#390 での
  突合結果」参照。`docs/perf/cuda-floor-remeasurement.md` が正本）。**#391**（起動コスト計測）: 5.1 節の
  残る並列競合起因の性能アサーション不安定性（直列実行で解消する 4 件・解消が不確実な
  `tensor_core_tflops_record` 1 件、および #390 が新たに残した直列再実行値との約 1.6〜1.7 倍の残差）を
  計測プロトコル頑健性の論点として引き渡す。転送のみ計測と合算計測の相対関係が実行間で安定しない事象
  （`tensor_core_tflops_record`）
  は #391 側の計測手法見直しの対象になりうる
- 本イシューでは REQ-8 下限・ディスパッチ閾値・数値一致許容誤差のいずれも変更していない
