# CUDA parity 非後退契約のベースライン（イシュー #491）

## 1. 位置づけ

`wmma_tf32`・`wmma_tf32_opt`・`mma_f16` 系経路は REQ-2 統一複合判定（相対誤差
1e-3 未満 または 絶対誤差 1e-5 未満）で恒常 fail の既知状態にある
（#186 由来。`docs/backend-cuda-real-device-testing.md` §5.3 に DGX Spark
GB10・sm_121 実機実測を記録済み）。REQ-2 改定（#186）が完了するまでこの状態は
解消しないため、GEMM 性能改善ツリー（ルート #479 → Phase 2 親 #490）以降の
Phase B/C カーネル改修は「parity green」を受け入れ条件にできない。

本ドキュメントが定義する**非後退契約**は、カーネル改修の受け入れ判定を
「fail 比率・平均絶対誤差が記録済みベースラインを上回らないこと」に置き換える。
機械検査の正は
`crates/backend-cuda/tests/common/parity_baseline.rs`（fixture・検査
ユーティリティ）と `crates/backend-cuda/tests/parity_nonregression.rs`
（テスト本体）であり、本ドキュメントは出典・実測環境・更新規約の記録に徹する
（二重管理を避ける）。

**承認記録（PR #640 codex-review 指摘対応）**: この非後退契約（恒常 fail
経路を「parity green」ではなく「記録済みベースラインを上回らないこと」で
受け入れ判定する設計）は、`RELATIVE_TOLERANCE`/`ABSOLUTE_RESCUE_THRESHOLD`
（1 要素あたりの合否判定式）を緩和するものではない。契約自体はイシュー #491
本文「§1.2 parity 非後退契約」の受け入れ基準（`gh issue view 491` で参照
可能）としてユーザー承認済みの仕様であり、本 PR はその機械検査を実装した
だけで新たな緩和を導入していない。以後この上限を**上方更新（緩和）**する
場合のみ §6 のとおり別途ユーザー承認が必要（初回記録・下方更新〈改善の
反映〉と区別する）。

## 2. 非後退契約の定義（検査 5 項目）

1. **tolerance 定数不変**: `backend_cpu::RELATIVE_TOLERANCE`（1e-3）・
   `backend_cpu::ABSOLUTE_RESCUE_THRESHOLD`（1e-5）が変更されていないこと
   （`tolerance_constants_are_pinned` テストで bit 等値 assert）
2. **fail 比率非増加**: 各ベースライン行について `fail_count` が記録値を
   超えないこと
3. **mean_abs_diff 非増加**: 各ベースライン行について `mean_abs_diff` が
   記録値（表記丸め天井値。4 節）を超えないこと
4. **FMA 契約維持**: 参照値計算は `backend_cpu::matmul_reference_fma`
   （TF32 経路）／f16 丸め込み経路（`mma_f16` 経路。3.2 節）を用い、独自の
   参照実装で判定式を複製しない
5. **実測値の本ドキュメント追記**: ベースライン行を追加・更新する場合は
   実機実測とセットで本ドキュメントへ記録する（推定値の記載を禁止）

## 3. ベースライン表

出典: `docs/backend-cuda-real-device-testing.md` §5.3（DGX Spark GB10・
sm_121・2026 年 8 月時点実測）。関連: `docs/perf/cuda-tensor-core-measurement.md`
・`docs/perf/cuda-floor-remeasurement.md`。

| 経路 | エントリポイント | 形状（M×N×K） | seed | fail_count/total | mean_abs_diff | 出典テスト |
|---|---|---|---|---|---|---|
| `wmma_tf32` | `CudaGemm::run_wmma_tf32` | 32×32×32 | 2000 | 154/1024 (15.0%) | 3.698e-4 | `gemm_wmma_tf32.rs::wmma_tf32_matches_reference_across_shapes`（先頭ケース） |
| `wmma_tf32` | `CudaGemm::run_wmma_tf32` | 256×256×4096 | 8888 | 10647/65536 (16.2%) | 4.476e-3 | `gemm_wmma_tf32.rs::wmma_tf32_k4096_stress_poc_v2_5`（先頭呼出し） |
| `wmma_tf32_opt` | `CudaGemm::run_wmma_tf32_opt_kernel`（private。gemm.rs 内部テスト経由） | 64×64×64 | 3000 | 699/4096 (17.1%) | 5.676e-4 | `gemm_wmma_tf32_opt.rs::wmma_tf32_opt_matches_reference_across_shapes`（記録元・改名前。現行は `wmma_tf32_routed_path_matches_reference_across_shapes`。先頭ケース） |
| `wmma_tf32_opt` | `CudaGemm::run_wmma_tf32_opt_kernel`（private。gemm.rs 内部テスト経由） | 512×512×512 | 0x7A0 | 42493/262144 (16.2%) | 1.574e-3 | `tensor_core_real_device.rs::tensor_core_parity_record`（記録元。TF32 部分。計測前に `wmma_tf32_opt_available()` を assert） |
| `wmma_tf32_opt` | `CudaGemm::run_wmma_tf32_opt_kernel`（private。gemm.rs 内部テスト経由） | 512×512×4096 | 0xC0FFEE | 43019/262144 (16.4%) | 4.463e-3 | `gemm_wmma_tf32_opt.rs::wmma_tf32_opt_k4096_stress`（記録元・改名前。現行は `wmma_tf32_routed_path_k4096_stress`。先頭呼出し） |
| `wmma_tf32_staged` | `CudaGemm::run_wmma_tf32`（staged 利用可能・整列形状） | 512×512×4096 | 0xC0FFEE | 未計測 | 未計測 | `tests/gemm_wmma_tf32_staged.rs::wmma_tf32_staged_k4096_stress`（記録元候補。実機再測定待ち） |
| `mma_f16` | `CudaMmaGemm::run_f16` | 256×256×4096 | 9999 | 101/65536 (0.15%) | 7.646e-5 | `cpu_cuda_mma_parity.rs::mma_f16_k4096_stress`（先頭呼出し） |

**ルーティング変更（PR #678 codex-review P1 指摘対応・イシュー #500）**:
`wmma_tf32_opt` 行は記録時点（#500 の staged カーネル追加前）ではエント
リポイントが `wmma_tf32` と共通で、opt 可用性さえ確認すれば `run_wmma_tf32`
（公開 API）が実際に opt 経路を通っていた。#500 以降 `run_wmma_tf32` は
cp.async 16 バイト整列条件を満たす形状で staged 経路を最優先するため、この
記録値の非後退検査は公開 API 経由では行わず、
`backend_cuda::gemm::tests::wmma_tf32_opt_kernel_parity_does_not_regress`
（`src/gemm.rs`。private field `wmma_tf32_opt`・private fn
`run_wmma_tf32_opt_kernel` へ直接アクセスし 3 段選択を経由せず opt カーネル
を強制実行する）へ移設した（公開 API 経由の旧検査が黙って staged 経路へ
すり替わっていた欠陥の是正）。新設の `wmma_tf32_staged` 行は #500 以降の
`run_wmma_tf32` が整列形状で実際に選ぶ経路であるため、逆に公開 API 経由で
正しく検査できる（`tests/parity_nonregression.rs::
check_wmma_tf32_staged_baseline`）。

**opt カーネル単独の形状網羅（PR #678 codex-review P1 再指摘対応）**: 上記
非後退ゲートはこの表に記録済みの 3 形状（64×64×64・512×512×512・
512×512×4096）しか検査しない。旧 `gemm_wmma_tf32_opt.rs::
wmma_tf32_opt_matches_reference_across_shapes`／`wmma_tf32_opt_k4096_stress`
が検査していた opt カーネル固有のタイル境界網羅（128×128×128・63×65×33・
65×63×17・64×96×256・1×1×1 を含む）は、同じ private field 経由アクセスで
CPU 参照実装と直接照合する
`backend_cuda::gemm::tests::wmma_tf32_opt_kernel_matches_reference_across_shapes`・
`wmma_tf32_opt_kernel_k4096_stress`（いずれも `src/gemm.rs`。
`backend_cpu::assert_parity` による判定であり、この表の非後退ベースライン
機構〈`assert_no_parity_regression`〉は使わない——未計測形状を本表へ追加
すると `baseline_provenance_unconfirmed: true` の fail-closed 契約により
無条件 panic になるため）へ移設した。移設元の `gemm_wmma_tf32_opt.rs` 側は
`wmma_tf32_routed_path_matches_reference_across_shapes`・
`wmma_tf32_routed_path_k4096_stress` へ改名し、「実効ルーティング経路の
parity」検査として引き続き公開 API 経由で機能する。

**既知の限界（`wmma_tf32` 行 2 件・PR #640 Cursor Bugbot 指摘・codex-review
P1 指摘。未解決・実機再測定が必要）**: 上表 1〜2 行目（32×32×32 seed=2000・
256×256×4096 seed=8888）は出典テスト `gemm_wmma_tf32.rs` が
`CudaGemm::run_wmma_tf32` を opt 可用性の確認なしに呼び出しており、DGX
Spark GB10 実機実測環境では opt カーネルが利用可能であった可能性が高い
（`docs/perf/cuda-floor-remeasurement.md`「opt カーネル可用性の検証」節
参照。同実機で `wmma_tf32_opt_available()` が概ね true であることの
傍証）。そのため記録値が実際には opt カーネルの結果であり、基本版カーネル
専用の単体テスト
（`backend_cuda::gemm::tests::wmma_tf32_basic_kernel_parity_does_not_regress`。
`crates/backend-cuda/src/gemm.rs`。§7 関連）の非後退検査と比較した際に
K-tiling 差（基本版 8 / opt 16）由来の false-fail・false-pass を生む可能性が
ある。実機未到達のため本 PR では再測定できず、`wmma_tf32` 行の基本版
カーネル確定測定は実機到達時のフォローアップ課題として引き継ぐ（推定値で
上書きしない。§6「未計測形状・シードの行追加」と同じ原則を、既存行の
provenance 再確認にも適用する）。

**機械的な運用対応（codex-review P1 再指摘対応。fail-closed 契約への変更）**:
この provenance 不確実性を「わかったうえで放置」せず、
`ParityBaseline::baseline_provenance_unconfirmed` フィールド
（`common/parity_baseline.rs`）でこの 2 行を明示的に `true` へマークした。
初回実装（`pending_basic_remeasurement`）は該当行の判定を黙って skip し
「実機テストは正常終了と shape 一致だけで通過する」状態を許していたが、
これは非後退ゲートが機能していないのに green に見える回帰だったため、
`assert_no_parity_regression`（`common/parity_baseline.rs`）自身がこの
フラグを検査する構造に変更した: フラグが `true` の行を渡すと、実測値の
良否に関わらず**必ず panic する**（fail-closed。判定を呼び出し側で
迂回できない）。したがって基本版カーネル専用の単体テスト
（`wmma_tf32_basic_kernel_parity_does_not_regress`）は、実機再測定で
provenance を確定させ `baseline_provenance_unconfirmed: false` へ更新する
までの間、実機で実行するたびに fail し続ける契約であり、これは意図した
挙動である（本リポで既知の受け入れ済み状態。
`docs/backend-cuda-real-device-testing.md` §5.3・§7 参照。「実機テスト全件
pass」は本イシューのスコープでは未達のまま確定している既存の前例と同種）。

**§5.3 の記録は「各テストで最初に fail した (形状, シード) の値」のみ
（`assert_parity` が最初の fail で panic する契約のため）**。上表 6 行
（`wmma_tf32_staged` を除く）はその実測値の転記であり、未計測形状・シード
の行は本表に含めていない。`wmma_tf32_staged` 行は §6「未計測形状・シードの
行追加」の例外として、実機再測定を強制する fail-closed プレースホルダで
ある（数値は実測値ではない。実測完了までは実機で実行するたびに必ず fail
する契約）。

## 4. 表記丸め対応

§5.3 の記録値は 4 有効桁への丸め表記のため、fixture
（`baseline_mean_abs_diff_ceiling`）には**記録表記の最終桁を切り上げた
天井値**を格納する（例: 3.698e-4 → 3.699e-4、4.476e-3 → 4.477e-3）。これは
表記丸め誤差の吸収であり許容誤差（tolerance 定数）の緩和ではない。判定式・
`RELATIVE_TOLERANCE`/`ABSOLUTE_RESCUE_THRESHOLD` は一切変更しない。

`fail_count` は整数の実測値であり丸め対応の対象外（記録値をそのまま上限
とする）。

## 5. 参照値計算の経路（fixture・テストが再現する契約）

- **TF32 経路**（`wmma_tf32`/`wmma_tf32_opt`）: 入力を f32 のまま
  `backend_cpu::matmul_reference_fma` で計算する非量子化参照
  （`docs/backend-cuda-real-device-testing.md` §5.3「参照実装は意図的に
  非量子化のままである点の確認」参照。TF32 相当の 10bit 仮数丸めを参照側に
  適用しない設計判断を踏襲する）
- **f16 経路**（`mma_f16`）: f16→f32→`matmul_reference_fma`→f16 丸め→f32 の
  量子化込み経路（`cpu_cuda_mma_parity.rs::assert_mma_f16_parity` と同一
  手順。GPU 側エピローグ store の丸めを参照側にも反映させる）

## 6. ベースライン更新規約

- **下方更新（改善の反映）**: カーネル改修で fail_count・mean_abs_diff が
  実測で改善した場合、実機実測値の記録とセットで fixture・本表を更新
  してよい（後続タスクの通常フロー）
- **上方更新（緩和）**: fail_count・mean_abs_diff の上限を緩める変更は
  **ユーザー承認必須**（`.claude/rules/security.md` A08「ガードレール閾値・
  テスト許容誤差の変更は必ず人間の承認を経る」と同列のガードレール）
- **未計測形状・シードの行追加**: 実機実測とセットでのみ行う（推定値の
  捏造をしない。§3 の実測記録方針を継続する）。ただし、公開 API から新たに
  到達可能になった経路（例: `wmma_tf32_staged`。#500）を非後退ゲートの対象
  経路として明示するために行だけを先行追加する場合は例外的に許容する:
  `baseline_fail_count`/`baseline_mean_abs_diff_ceiling` に推定値ではなく
  プレースホルダ（0 等）を入れ、`baseline_provenance_unconfirmed: true` で
  必ず fail-closed に倒す（`ParityBaseline::baseline_provenance_unconfirmed`
  ドキュメンテーションコメント参照）。この状態は「実機未到達」を明示する
  ゲートであり、実測値を主張しない。実機実測が完了次第、確定値へ差し替えて
  `false` へ更新する（PR #678 codex-review P1 指摘対応。`WmmaTf32`〈基本版〉
  行の既存前例と同型）
- tolerance 定数（`RELATIVE_TOLERANCE`/`ABSOLUTE_RESCUE_THRESHOLD`）自体の
  変更は本契約のスコープ外（#186・REQ-2 改定側の判断。変更する場合は
  `crates/backend-cpu/src/parity.rs` のコメントに従いユーザー承認を得る）

## 7. 関連

- `docs/backend-cuda-real-device-testing.md` §5.3（実測記録本体）
- `crates/backend-cuda/tests/common/parity_baseline.rs`（fixture・検査
  ユーティリティ本体）
- `crates/backend-cuda/tests/parity_nonregression.rs`（非後退契約テスト。
  `wmma_tf32_staged`・`mma_f16` 経路。公開 API 経由）
- `crates/backend-cuda/src/gemm.rs`（`mod tests` 内、公開 API・feature を
  増やさず private field へ直接アクセスするライブラリ単体テスト群）:
  - `wmma_tf32_basic_kernel_parity_does_not_regress`（基本版 WMMA(TF32)
    カーネル〈`wmma_tf32` 経路〉専用。PR #640 codex-review P1 指摘対応）
  - `wmma_tf32_opt_kernel_parity_does_not_regress`（opt カーネル
    〈`wmma_tf32_opt` 経路〉専用。記録済みベースライン 3 形状の非後退検査。
    PR #678 codex-review P1 指摘対応・イシュー #500）
  - `wmma_tf32_opt_kernel_matches_reference_across_shapes`・
    `wmma_tf32_opt_kernel_k4096_stress`（opt カーネル単独のタイル境界
    形状網羅。`assert_parity` による直接照合〈非後退ベースラインは使わない〉。
    旧 `gemm_wmma_tf32_opt.rs::wmma_tf32_opt_matches_reference_across_shapes`／
    `wmma_tf32_opt_k4096_stress` からの移設。PR #678 codex-review P1
    再指摘対応）
  - `wmma_tf32_staged_kernel_exceeds_opt_kernel_tflops_at_4096`（staged 対
    opt の TFLOPS 直接比較。PR #678 codex-review P2 指摘対応）
- イシュー #186（Tensor Core 経路の数値一致閾値の実測再評価。REQ-2 改定
  候補の引き渡し先）
- イシュー #490（GEMM 性能改善ツリー Phase 2 親）
