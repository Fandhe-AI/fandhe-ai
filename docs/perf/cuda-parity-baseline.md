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
| `wmma_tf32_staged` | `CudaGemm::run_wmma_tf32`（staged 利用可能・整列形状） | 512×512×4096 | 0xC0FFEE | 43019/262144 (16.4%) | 4.463e-3 | `parity_nonregression.rs::parity_baselines_do_not_regress`（`check_wmma_tf32_staged_baseline`。#726・DGX Spark GB10 実機・コミット 06b24b4・2026-08-19。release/debug 各 2 回で同一値を確認。実測生値 mean_abs_diff=4.463436e-3。値は `wmma_tf32_opt` 同形状行と一致 — staged は opt と同一の FMA 契約・積和順序を保つ cp.async 二重バッファ版であることの実測裏付け） |
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
の行は本表に含めていない。`wmma_tf32_staged` 行は #500 時点では §6
「未計測形状・シードの行追加」の例外として実機再測定を強制する fail-closed
プレースホルダだったが、イシュー #726 の実機実測（8.5 表・§3 表参照）で
確定値へ差し替え済み（記録元は `parity_nonregression.rs::
parity_baselines_do_not_regress` の staged 検査。`assert_parity` 系と異なり
最初の fail で panic せず全要素集計の `CompareReport` を返す経路のため、
fail_count・mean_abs_diff は全 262144 要素の集計値である）。

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
- `docs/perf/cuda-optimized-remeasurement.md`「役割分担」節（#575 は性能値
  採用の前提ゲートとして非後退を確認する側であることの相互参照）

## 8. Phase F-4 最終確認（#575）

GEMM 性能改善ツリー（ルート #479）Phase F（親 #569）の F-4。B-0（#491・
PR #640）で確立した非後退契約に対し、Phase B/C のカーネル改修完了時点で
最終確認を行った記録。

### 8.1 B-0 基準に対する差分確認（実機非依存・完了）

B-0 の基準コミット `01872cb`（PR #640・#491）から本イシュー着手時点の
`origin/main`（`a0fc666f26a427758a30eed67cdccc88b2a54e67`。本イシュー用
ブランチの分岐元）までの差分を機械確認した。この確認は「契約側
（tolerance 定数・fixture 上限値）が緩和されていないこと」の静的検証で
あり、Phase B/C カーネル改修後の**実測値**が現行ベースラインを下回って
いることの証明ではない（8.6 で区別する）。

- **tolerance 定数**: `git diff 01872cb..HEAD -- crates/backend-cpu/src/parity.rs`
  は**差分なし**。`RELATIVE_TOLERANCE`（1e-3）・`ABSOLUTE_RESCUE_THRESHOLD`
  （1e-5）は B-0 から一度も変更されていない
- **ベースライン fixture**: `git diff 01872cb..HEAD --
  crates/backend-cuda/tests/common/parity_baseline.rs` の変更は
  `1df5c0a`（PR #678・イシュー #500）1 コミットのみ。内容は
  (a) `wmma_tf32_staged` 行の新規追加（fail-closed プレースホルダ。
  `baseline_fail_count: 0`・`baseline_mean_abs_diff_ceiling: 0.0`・
  `baseline_provenance_unconfirmed: true`。§6「未計測形状・シードの行
  追加」の例外条件どおり推定値ではなくプレースホルダ）、(b) opt カーネル
  単独検査の `src/gemm.rs` への移設に伴うコメント更新、(c) フィールド名
  `basic_kernel_baseline_unconfirmed` → `baseline_provenance_unconfirmed`
  への改名（`WmmaTf32Staged` 行にも同じ意味で使うための一般化）のみで、
  **既存行の `baseline_fail_count`・`baseline_mean_abs_diff_ceiling` に
  上方更新（緩和）は存在しない**（値はいずれも B-0 記録値のまま）。よって
  §6 の「上方更新はユーザー承認必須」規約に抵触する変更はなく、**契約
  自体（判定式・上限値）はカーネル改修（Phase B/C）を通じて一貫して
  緩和されていない**

一方、`1df5c0a`（PR #678・イシュー #500）は `kernels_wmma_opt.rs`（TF32
経路）へ Phase B 技法を横展開しており、TF32 経路のカーネル実装自体は
B-0 時点から変化している。したがって「契約が緩和されていないこと」と
「実測 fail_count・mean_abs_diff が現行ベースラインを実際に下回って
いること」は別の主張であり、後者は 8.3 の実機不達により本イシューでは
確認できていない（8.6 で明示的に区別する）

### 8.2 GPU 不要のロジック検査（実機非依存・完了・green）

`cargo test -p fandhe-ai-backend-cuda --test parity_nonregression` を実行し、
実機必須テスト（`parity_baselines_do_not_regress`。CUDA 実機・compute
capability 8.0 以降が必須で `#[ignore]` 分離済み）を除く 8 件（fixture
自己整合性・`tolerance_constants_are_pinned`・fail-closed 契約の
falsification 検査群）が全て green であることを確認した。

### 8.3 実機到達性ゲートの再確認（不達・#571/#572/#502 と同型）

`docs/real-hardware-verification-env.local.md`（実ホスト名を記録するローカル
用ファイル。`.gitignore` 対象）の存在を確認したところ、本セッションの作業
環境には存在しなかった。直前の F-1（#571・PR #710）・F-2（#572）・B-11
（#502）と同一の実機不達状態であり、同方式（実測せず安全側に倒し、推定値
を記載しない）を踏襲する。

**確定できないまま残る行（実機到達可能なセッションへの申し送り。2 状態を
区別する）**:

| 経路 | 形状・シード | 状態 | 未確定の理由 |
|---|---|---|---|
| `wmma_tf32`（基本版） | 32×32×32 seed=2000 | 要再測定（現行記録値は provenance 未確定） | §3 に実測値（154/1024・3.698e-4）は存在するが、記録元テストが opt 可用性を確認せず呼んでいるため opt 実測結果である疑いが残る（§「既知の限界」）。基本版カーネル専用の再測定が必要 |
| `wmma_tf32`（基本版） | 256×256×4096 seed=8888 | 要再測定（現行記録値は provenance 未確定） | 同上（§3 実測値: 10647/65536・4.476e-3） |
| `wmma_tf32_staged` | 512×512×4096 seed=0xC0FFEE | **確定済み（#726・2026-08-19）** | #500 で行のみ先行追加（プレースホルダ）→ #726 の実機実測（§3 表・8.5 表参照）で確定値へ差し替え・`baseline_provenance_unconfirmed: false` へ更新済み |

### 8.4 実機実測手順（申し送りテンプレート）

実機到達可能なセッションが引き継ぐ手順（`docs/real-hardware-verification-env.md`
§2〜3 に準拠）。

1. `.rev-stamp` 記録 → `rsync`（`--filter=':- .gitignore'` で `.env*`・
   `real-hardware-verification-env.local.md` 等の秘密情報・内部実値を除外）
   で DGX Spark GB10（`CUDA_NODE` プレースホルダ表記）へ転送する
2. 非ログイン shell の PATH を明示したうえで以下を実行する:
   ```bash
   cargo test -p fandhe-ai-backend-cuda --release --test parity_nonregression -- --ignored --nocapture
   cargo test -p fandhe-ai-backend-cuda --release --lib -- --ignored --nocapture
   ```
3. stdout の `fail_count/total`・`mean_abs_diff` を機械転記する（目測・
   推定を混ぜない）
4. **後退していない場合**: 8.3 表の該当行を実機実測値へ更新し、fixture
   （`crates/backend-cuda/tests/common/parity_baseline.rs`）の
   `baseline_fail_count`・`baseline_mean_abs_diff_ceiling`（4 節の表記丸め
   天井値対応を適用）を実測値へ差し替え、`baseline_provenance_unconfirmed`
   を `false` へ更新する（§6「下方更新」に該当し人間承認不要）
5. **後退している場合（実測値が現行のベースライン上限を上回った場合）**:
   ベースラインを緩めない。後退の事実を本節へ記録し、上方更新は
   `.claude/rules/security.md` A08 に従いユーザー承認事項として停止・
   申し送る（自動運転では実施不可）

### 8.5 最終値記入テンプレート（実機セッションが埋める）

| 経路 | 形状（M×N×K） | seed | fail_count/total | mean_abs_diff | 実測日 | 実測コミット |
|---|---|---|---|---|---|---|
| `wmma_tf32`（基本版） | 32×32×32 | 2000 | 要再測定・申し送り（現行記録値 154/1024 は provenance 未確定） | 要再測定・申し送り（現行記録値 3.698e-4 は provenance 未確定） | — | — |
| `wmma_tf32`（基本版） | 256×256×4096 | 8888 | 要再測定・申し送り（現行記録値 10647/65536 は provenance 未確定） | 要再測定・申し送り（現行記録値 4.476e-3 は provenance 未確定） | — | — |
| `wmma_tf32_staged` | 512×512×4096 | 0xC0FFEE | 43019/262144 (16.4%) | 4.463436e-3（記録表記 4.463e-3・fixture 天井値 4.464e-3） | 2026-08-19 | 06b24b4（#726。release/debug 各 2 回・計 4 回で同一値） |

### 8.6 結論（本イシュー #575 のスコープでの完了状態）

- 受け入れ基準 1「B-0 基準に対し全経路が非後退であること」: **2 つの主張
  に分けて評価する（8.1 の指摘どおり、契約の非緩和と実測の非後退は別命題
  のため両者を同一視しない）**
  - 1a. 契約側の非緩和（tolerance 定数・fixture 上限値の上方更新がない
    こと）: **確認完了**（8.1・8.2。fixture の静的検査・GPU 不要のロジック
    検査で B-0 からの緩和が存在しないことを機械確認済み）
  - 1b. 実測による非後退（Phase B/C 適用後のカーネル実測値が現行
    ベースラインを実際に下回っていること）: **実機不達のため未達**。
    `1df5c0a`（TF32 経路への Phase B 技法横展開）でカーネル実装自体は
    B-0 時点から変化しており、実測での裏取りが必要（8.3〜8.5 へ申し送り）
- 受け入れ基準 2「tolerance 定数が未変更であることの差分確認」: **完了**
  （8.1）
- 受け入れ基準 3「`fail_count` 比率・`mean_abs_diff` の最終値の本ドキュメ
  ントへの更新」: **実機不達のため未達。手順・テンプレートを整備し 8.3〜8.5
  で申し送り**（推定値は記載しない）
- 受け入れ基準 4「共通契約の遵守」: **遵守**（境界チェック・tolerance・
  依存関係・`docs/spec/`・REQ-8 下限のいずれも本イシューで変更していない）

## 9. rmsnorm_backward_dw_split の GB10 parity 失敗の切り分け・修正（イシュー #1105）

本節は §1〜8（GEMM `wmma`/`mma` 系経路専用のベースライン表・非後退契約）
とは別スコープの追補である。対象は RMSNorm 逆伝播 `dw` の単段フォール
バック経路（`RMSNORM_BWD_DW_F32`。`crates/backend-cuda/src/kernels_rmsnorm.rs`）
の parity で、本節 §2 の非後退契約の対象外（GEMM 専用の
ベースライン表・fixture に本節の経路は含まれない）。

### 9.1 検出（PR #1098 実機実測。#1102 ツリー・#1105 分解 1/3）

- `#[ignore]` テスト `rmsnorm_backward_dw_split_matches_cpu_across_shapes`
  （`crates/backend-cuda/tests/rmsnorm_backward_parity.rs`）が DGX Spark
  GB10（sm_121）実機で REQ-2 統一複合判定（相対誤差 1e-3 未満 または
  絶対誤差 1e-5 未満。`crates/backend-cpu/src/parity.rs`）を超過して FAIL
  した（fail 6/4096）。RTX 3060（sm_86）では同テストは pass
- dx（先に assert される。分母 8192×4096）は pass しているため、fail は
  ケース表 7 番目 `(rows=8192, hidden=4096, num_blocks=1)`（単段フォール
  バック dw 経路。split-K を要しない広い `hidden`）に限定されると判断した

### 9.2 切り分け（GB10 実機実測に基づく確定。#1105 実装セッションで実施）

**コード根拠（実測前の仮説）**:

- 単段 dw カーネル（`kernels_rmsnorm.rs` の `RMSNORM_BWD_DW_F32`）は
  行 0..rows を serial に `acc = fmaf(dyv * r, xv, acc)` で蓄積し、
  テスト内 CPU 参照実装（`(dy*rstd).mul_add(x, dw)`）と加算順・FMA 契約が
  完全に一致する。両者で唯一異なりうる入力は `rstd`（GPU 側は forward
  `run_rmsnorm_f32_train` が保存した値、CPU 参照は自前で再計算）
- forward カーネル `RMSNORM_F32_ONEPASS`／`RMSNORM_F32_TWOPASS` は当初
  `rstd = rsqrtf(fmaf(acc, inv_n, eps))` と近似 intrinsic `rsqrtf`
  （MUFU.RSQ。丸めがアーキテクチャ実装依存・最大 ~2 ulp）を使っていた

**GB10 実機での検証結果（2026-09-01 実測。`--release`・`--ignored`）**:

1. `rsqrtf` → `1.0f / sqrtf(...)`（9.3）へ置換した修正版で
   `rmsnorm_backward_dw_split_matches_cpu_across_shapes` を再実行したところ、
   該当ケース `(rows=8192, hidden=4096, num_blocks=1)` は
   `fail_count=5/4096, max_abs_diff=2.975e-4, max_rel_err=1.340e-2,
   mean_abs_diff=4.477e-5, p50_abs_diff=3.433e-5` で **依然 FAIL**
   （修正前は fail 6/4096。単一測定であり「改善」と主張はしない。
   `p50_abs_diff` が絶対救済閾値 1e-5 を上回っており、4096 要素の中央値
   ですでに閾値超過という規模の誤差である点に注意）
2. 追加の切り分け診断（コミットしない一時テストを転送ツリー上でのみ実行。
   受け入れ基準検証対象のテストファイル自体は変更していない）:
   GPU forward が保存した `rstd` をそのまま CPU 参照の
   `(dy*rstd).mul_add(x, dw)` 式（dw カーネルと同一の逐次蓄積順）に代入
   して計算した「CPU(GPU-rstd) dw」を GPU dw 出力と比較したところ、
   4096 要素全てが **bit 一致**（`dw_mismatch_count=0/4096`）した。
   これにより **dw カーネル自体は無罪**と確定できる（誤差が入り得る
   経路は `rstd` のみ）
3. 同診断で GPU `rstd` と CPU 再計算 `rstd`（両者とも IEEE 丸めの
   `1.0f32 / (...).sqrt()`）の行あたり最大相対差を実測したところ
   `max_rstd_rel_delta = 2.06e-6` だった。両者とも丸め方式は同一
   （IEEE 丸め）なため、この差は丸め方式の不一致ではなく **forward の
   二乗和 `acc` の縮約順序の違い**に由来する: forward カーネルは
   32 レーンのストライド部分和 + 5 段 butterfly `__shfl_xor_sync`
   reduction（並列木構造）で `acc` を求めるのに対し、CPU 参照実装は
   `0..hidden` の単純逐次 `mul_add` で求める。浮動小数点加算は結合則を
   満たさないため、たとえ個々の演算が全て IEEE 丸め・`fmaf` 契約で
   統一されていても、縮約順序が異なれば `acc`（ひいては `sqrt` の引数・
   `rstd`）は ULP レベルで一致しない
4. `rows = 8192` の逐次蓄積（dw カーネル）は行あたり ULP レベルの
   `rstd` 差をランダムウォーク的に増幅し、キャンセレーションで値が
   小さくなる `dw[j]` 要素では絶対救済・相対救済の双方を超過する
   （`max_rel_err=1.340e-2` は `max_abs_diff=2.975e-4` の要素で相対救済も
   破れたケース）

**切り分け結論（実測で確定・訂正）**:

- 実装前の仮説「`rsqrtf` の世代依存近似誤差が主因」は診断 3 の結果
  （GPU/CPU とも IEEE 丸めに統一した後も `rstd` に ULP 差が残る）により
  **反証された**。`rsqrtf` は丸め契約として CPU 参照・`backend-cpu` 本番
  実装と不一致だった点で独立に是正すべき問題ではあるが、GB10 での
  FAIL の主因ではなかった
- 真因は **forward の warp butterfly reduction（並列木構造）と CPU 参照
  実装の逐次和という、縮約順序の非結合性に起因する `rstd` の ULP 差**
  （診断 2 が dw カーネル自体の無罪を bit 一致で証明済み）。これが
  `rows = 8192` という長い逐次蓄積を持つケースでのみ複合判定を超過する
  規模まで増幅される
- 「カーネルの不具合」か「テスト前提の誤り」かの二択でいえば、
  古典的な意味でのバグ（誤った計算式・境界誤り等）ではなく、**性能
  志向の並列縮約アルゴリズムと逐次 CPU 参照実装の間に本質的に存在する
  非結合性**が、テストケース表 7 番目（`rows` が大きく `hidden` も広い
  単段 dw 経路）でのみ顕在化したものである。sm_86 で pass していた
  理由は `rsqrtf` の世代依存丸め分布が偶然この特定ケースの ULP 差を
  救済側へ寄せていたためと考えられる（両世代とも同じ縮約順序の非結合性
  自体は抱えている）
- この結論はテスト前提（ケース形状・シード・tolerance）が誤っていた
  ことを意味しない。むしろ「並列縮約 forward と逐次 CPU 参照実装の
  間で、十分長い行の逆伝播蓄積を通すと ULP 差が複合判定を超えうる」
  という、性能重視カーネル設計に内在する制約を実測で可視化したもの
  であり、**tolerance・テスト設計をどう扱うかはユーザー判断事項**
  として提示する（9.4）

### 9.3 適用した修正（丸め契約の是正。parity FAIL 自体は未解消）

対象: `crates/backend-cuda/src/kernels_rmsnorm.rs`

- `RMSNORM_F32_ONEPASS`・`RMSNORM_F32_TWOPASS` の
  `float rstd = rsqrtf(fmaf(acc, inv_n, eps));` を
  `float rstd = 1.0f / sqrtf(fmaf(acc, inv_n, eps));` へ変更した
  （forward・backward 双方に共通の `rstd` 計算式。CPU 参照・`backend-cpu`
  本番実装と丸め契約を統一）
- この変更は 9.2 の実測で判明したとおり **GB10 での parity FAIL 自体は
  解消しない**（真因は縮約順序の非結合性であり `rsqrtf` はその主因では
  なかった）。それでもなお、CPU 参照・`backend-cpu` 本番実装との丸め
  契約統一（`.claude/rules/coding-rust.md` の FMA 契約統一規約）として
  独立に妥当な是正であり、世代依存の近似 intrinsic をカーネルに残さない
  という意味で維持する
- 回帰テスト `forward_kernels_do_not_use_approximate_rsqrtf`
  （`kernels_rmsnorm.rs` の `mod tests`）を追加し、`rsqrtf(` への後退と
  `1.0f / sqrtf(...)` 計算式の欠落を CI（実機不要の文字列 assert）で検出
  する
- **`RELATIVE_TOLERANCE`・`ABSOLUTE_RESCUE_THRESHOLD`・テストのケース表
  ／シード／判定式は一切変更していない**（`.claude/rules/security.md`
  A08。`assert_tolerance_constants_pinned` が bit 等値で検査する契約は
  そのまま維持）

### 9.4 結論・ユーザー判断事項（イシュー #1105 は未クローズ）

GB10 実機実測（9.2）の結果、受け入れ基準 3「GB10 実機で該当 parity
テストが pass する」は**未達**である。`rmsnorm_backward_dw_split_matches_
cpu_across_shapes` はケース `(rows=8192, hidden=4096, num_blocks=1)` で
修正後も FAIL する（fail_count=5/4096。単一測定であり修正前 6/4096 との
差を「改善効果」としては主張しない）。

切り分け（9.2）により、残存原因は dw カーネル自体のバグではなく
**forward の warp 並列縮約と CPU 参照実装の逐次和という、縮約順序の
非結合性に起因する `rstd` の ULP 差が、長い逆伝播蓄積（rows=8192）で
増幅されるもの**と確定している。この性質は現行のカーネル設計
（warp 内 `__shfl_xor_sync` butterfly reduction。性能上の理由で採用）に
構造的に内在するため、次のいずれも自動運転の範囲外としてユーザー判断へ
委ねる（`.claude/rules/security.md` A08・`out-of-scope-tracking.md`）:

1. **tolerance の緩和**（`RELATIVE_TOLERANCE`／`ABSOLUTE_RESCUE_THRESHOLD`
   またはケース単位の例外）— 変更にはユーザー承認が必須
2. **forward の縮約順序を CPU 逐次和に近づける再設計**（性能への影響を
   伴うアーキテクチャ変更）
3. **テスト設計側の対応**（例: 当該ケースを parity 検証の対象形状から
   除外する、または非後退契約〈本節 §2 と同型の仕組み〉へ切り替える）

**実測記録（2026-09-01・GB10・sm_121・CUDA 13.0.88）**:

| 経路 | 形状 | fail_count | max_abs_diff | max_rel_err | mean_abs_diff | p50_abs_diff | 実測コミット |
|---|---|---|---|---|---|---|---|
| `rmsnorm_backward_dw_split_matches_cpu_across_shapes`（修正後） | rows=8192, hidden=4096, num_blocks=1, eps=1e-5 | 5/4096 | 2.975e-4 | 1.340e-2 | 4.477e-5 | 3.433e-5 | 289622d + 本 PR の未コミット変更（rsqrtf→1.0f/sqrtf 修正版） |

診断専用の一時テスト（コミットしない。転送ツリー上でのみ実行）の実測値:

- dw カーネル bit 一致検査: `dw_mismatch_count=0/4096`（dw カーネル無罪の確証）
- GPU/CPU 再計算 `rstd` の最大相対差: `2.06e-6`（縮約順序差由来。丸め方式の不一致ではない）

なお `rmsnorm_backward_matches_cpu_across_shapes`（dx 側）・
`rmsnorm_parity`（forward 側、`rmsnorm_matches_backend_cpu_directly`・
`rmsnorm_matches_cpu_across_shapes`）は同一実機実測で全て pass しており、
`rsqrtf` → `1.0f / sqrtf` の変更による非後退（regression）は確認されて
いない。
