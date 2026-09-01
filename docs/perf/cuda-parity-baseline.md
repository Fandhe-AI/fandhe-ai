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

**既知の限界は解消済み（イシュー #1106・GB10 実機実測 2026-08-31/09-01。
コミット `d39e03b`。旧内容は下記に残す）**: 上表 1〜2 行目（32×32×32
seed=2000・256×256×4096 seed=8888）は出典テスト `gemm_wmma_tf32.rs` が
`CudaGemm::run_wmma_tf32` を opt 可用性の確認なしに呼び出しており、記録値が
実際には opt カーネルの結果である可能性が懸念されていた。基本版カーネル
専用の単体テスト
（`backend_cuda::gemm::tests::wmma_tf32_basic_kernel_parity_does_not_regress`。
`crates/backend-cuda/src/gemm.rs`。§7 関連）を DGX Spark GB10（sm_121・
CUDA 13.0）実機で release 2 回実行し、いずれも 32×32×32 で
fail_count=154/1024・mean_abs_diff=3.697936e-4、256×256×4096 で
fail_count=10647/65536・mean_abs_diff=4.476030e-3 の完全一致を確認した。
実測値は記録済みの値（上表）と一致しており、`wmma_tf32`（基本版）・
`wmma_tf32_opt` が同一の parity 分布を持つという
`docs/perf/cuda-tensor-core-tolerance-gb10-scale-sweep.md`（#995。GB10 実機で
basic/opt/staged 数値完全一致）の結果を裏付ける。推定値の上書きではなく
実測による確認であり、`crates/backend-cuda/tests/common/parity_baseline.rs`
の該当 2 行を `baseline_provenance_unconfirmed: false` へ更新済み。

（旧記述・履歴として保持）PR #640 Cursor Bugbot 指摘・codex-review P1 指摘の
懸念事項: 記録値が opt カーネル実測である可能性（DGX Spark GB10 実機実測
環境では opt カーネルが利用可能であった可能性が高いことが根拠。
`docs/perf/cuda-floor-remeasurement.md`「opt カーネル可用性の検証」節参照）。
K-tiling 差（基本版 8 / opt 16）由来の false-fail・false-pass を生む可能性が
懸念されていたが、上記実測により両カーネルの parity 分布が実際には
一致することが確認され、この懸念は解消された。

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
| `wmma_tf32`（基本版） | 32×32×32 seed=2000 | **確定済み（#1106・2026-08-31/09-01）** | GB10 実機 release 2 回実行で 154/1024・3.697936e-4 の完全一致を確認し `baseline_provenance_unconfirmed: false` へ更新済み（§「既知の限界」参照） |
| `wmma_tf32`（基本版） | 256×256×4096 seed=8888 | **確定済み（#1106・2026-08-31/09-01）** | GB10 実機 release 2 回実行で 10647/65536・4.476030e-3 の完全一致を確認し `baseline_provenance_unconfirmed: false` へ更新済み（§「既知の限界」参照） |
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
| `wmma_tf32`（基本版） | 32×32×32 | 2000 | 154/1024 (15.0%) | 3.697936e-4（記録表記 3.699e-4・fixture 天井値 3.699e-4） | 2026-08-31/09-01 | c58d905（#1106。release 2 回・同一値） |
| `wmma_tf32`（基本版） | 256×256×4096 | 8888 | 10647/65536 (16.2%) | 4.476030e-3（記録表記 4.477e-3・fixture 天井値 4.477e-3） | 2026-08-31/09-01 | c58d905（#1106。release 2 回・同一値） |
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

### 9.5 テスト設計側の対応（責務分離。イシュー #1102。tolerance 非変更）

9.4 でユーザー判断へ委ねた 3 選択肢のうち、**選択肢 3「テスト設計側の
対応」**を、tolerance 緩和（選択肢 1）を伴わない形で適用した。

**方針**: `assert_rmsnorm_backward_dw_split_parity`
（`crates/backend-cuda/tests/rmsnorm_backward_parity.rs`）の CPU 参照実装
（`cpu_rmsnorm_backward_reference`）が dw／dx を計算する際に使う `rstd` を、
CPU 側で独自に逐次和から再計算する方式から、**forward（GPU）が実際に
生成した `rstd` を D2H で取得しそのまま供給する方式**へ変更した。

- 9.2 の切り分けにより「dw カーネル自体は同一 `rstd` 下で GPU 出力と
  bit 一致する（`dw_mismatch_count=0/4096`）」ことが実測済みであり、
  `rstd` を GPU 側の値へ揃えることで dw（および dx）カーネルの縮約式・
  境界処理の正しさという本来の検証責務は失われない
- **rstd バッファ自体の parity は失わない**: 上記の切替だけでは「GPU
  forward が保存した `rstd` バッファ自体の取り違え・破損」を検出する
  経路が失われる（advisor レビュー指摘）ため、`assert_rmsnorm_backward_
  dw_split_parity` 内で GPU `rstd` と CPU 独立再計算
  （`cpu_rmsnorm_rstd_reference`。縮約順序は GPU forward の warp
  butterfly reduction と異なる素朴な逐次和）の複合判定
  （`fandhe_ai_backend_cpu::parity::assert_parity`。REQ-2 統一複合判定を
  再定義せず流用）を明示的に追加した。9.2 実測の `max_rstd_rel_delta =
  2.06e-6`（相対許容 1e-3 の 3 桁下）はこの判定を通過することを裏付ける
- `rstd` 自体の縮約順序差（forward の warp butterfly reduction と CPU
  逐次和という非結合性に起因する ULP 差）は REQ-2 統一複合判定の許容
  範囲に収まる（9.2 実測で確認済み）ため、上記 rstd バッファ parity は
  この ULP 差では fail しない。正確な言い方をすれば「この複合判定は
  ULP 差を fail-closed に検出する」のではなく「ULP 差が複合判定の許容
  範囲内に収まることを判定式自体が保証する」——rstd の乖離が許容範囲を
  超える異常（実質的なバグ）であれば、この判定も forward parity テスト
  （`rmsnorm_parity.rs::rmsnorm_matches_cpu_across_shapes`・
  `rmsnorm_matches_backend_cpu_directly`）も同一の複合判定で fail する
- forward parity テスト（`rmsnorm_parity.rs`）は hidden ∈
  {8, 1024, 4096, 4097, 8192, 16384} を網羅するが、dw split 検証の
  ケース表（9.1）は hidden ∈ {64, 128, 4096, 4097, 8, 16} も含み両者の
  網羅形状は完全一致しない。上記 rstd バッファ parity の追加により、
  dw split 検証自身が全ケース形状で rstd の複合判定を独立に行うため、
  この形状網羅の差はカバレッジの穴にならない
- `RELATIVE_TOLERANCE`・`ABSOLUTE_RESCUE_THRESHOLD`・複合判定式
  （`fandhe_ai_backend_cpu::parity::assert_parity`）・ケース表・シードは
  一切変更していない（`assert_tolerance_constants_pinned` の bit 等値検査
  はそのまま維持。`.claude/rules/coding-rust.md` の tolerance 非緩和方針
  を遵守）
- `assert_rmsnorm_backward_parity`（dx 中心の `rmsnorm_backward_matches_
  cpu_across_shapes`。実機で既に pass 実績あり）は本変更の対象外とした
  （スコープを 9.1〜9.4 で FAIL が確定している dw split 経路に限定し、
  無関係な既存 pass 経路への変更を避けるため）
- **`assert_rmsnorm_backward_dw_split_parity` の dx 判定への副次影響**:
  本関数は dw と dx を同時に検証しており、`gpu_rstd` 切替は dx 側の CPU
  参照計算にも及ぶ（同一関数を共有するため）。dx 判定はこれまで GB10
  実機で pass していた経路であり、rstd を GPU 側へ揃えることは非結合性
  由来の ULP 差を縮める方向の変更であるため後退の懸念は小さいと判断する
  が、実機未確認である旨を明記する（次回実機セッションで dx 側の非後退
  も合わせて確認する）

**検証範囲の変更を明文化するコメント**: 上記契約はテストファイル冒頭の
モジュールコメント・`cpu_rmsnorm_backward_reference`／
`cpu_rmsnorm_rstd_reference` のドキュメンテーションコメント・
`assert_rmsnorm_backward_dw_split_parity` 内の呼び出し箇所コメントに記録
した（後続の読み手が「なぜ rstd を GPU から渡すのか」「rstd バッファ
自体の parity をどこで検証しているか」を本ファイル単体で追跡できるように
するため。`.claude/rules/code-comment-style.md`）。

**GB10 実機再検証（2026-09-01・9.5 の rstd 供給責務分離を適用したブランチ
`fix/1102-rmsnorm-dw-parity-contract` 時点。コーディネータによる実機実行）**:

- `rmsnorm_backward_matches_cpu_across_shapes`（非 split・dx 中心）: **pass**
- `rmsnorm_backward_dw_split_matches_cpu_across_shapes`: **旧 FAIL ケース
  （`rows=8192, hidden=4096, num_blocks=1`）は通過した**（9.2 診断
  「dw カーネル無罪・rstd の ULP 差が真因」という切り分け、および 9.5 の
  rstd 供給責務分離の妥当性を実機で裏付ける結果）。ただし**別ケースで
  FAIL が新規発生**した:
  `(rows=4096, hidden=4097, num_blocks=8, eps=1e-5)` で
  `fail_count=1/4097, max_abs_diff=3.052e-4, max_rel_err=1.315e-3,
  mean_abs_diff=3.147e-5`（`rmsnorm_backward_parity.rs:405` 相当）

### 9.6 第 2 の非結合性: dw split-K の二段縮約順序（イシュー #1102 継続）

**切り分け**: 9.5 は `rstd` の非結合性（forward の warp butterfly
reduction 対 CPU 逐次和）を責務分離したが、`num_blocks >= 2`（split-K
経路）では **dw 自体の縮約にも別の非結合性**が存在する。GPU 側
（`kernels_rmsnorm.rs`）は二段構成である:

1. `RMSNORM_BWD_DW_PARTIAL_F32`: 行を `num_blocks` 個のブロック
   （`rows_per_block = ceil(rows / num_blocks)`）へ分割し、各ブロック内は
   行順の `fmaf` 逐次蓄積で部分和 `dw_partial[b, i]` を求める
2. `RMSNORM_BWD_DW_REDUCE_F32`: `dw_partial` をブロック番号
   `b = 0..num_blocks` の順に**単純な浮動小数点加算**
   （`acc += smem[buf][j][tid]`。`fmaf` ではない）で縮約し `dw` を書く

一方 9.5 以前の CPU 参照実装（`cpu_rmsnorm_backward_reference` の dw 蓄積）
は `num_blocks` に関わらず `0..rows` の単一の行順逐次 `mul_add` 蓄積で
あり、GPU の二段（ブロック内 fmaf・ブロック間単純加算）構造を再現して
いなかった。浮動小数点加算は結合則を満たさないため、`num_blocks` が
大きく `hidden` が vec4 非整列（4097）な形状ほどこの縮約順序差が複合判定
を超えうる規模まで蓄積しうる。

**対応（tolerance 非変更）**: `crates/backend-cuda/tests/
rmsnorm_backward_parity.rs` に `cpu_rmsnorm_dw_split_reference` を新設し、
GPU の二段縮約順序（ブロック内 `mul_add` 逐次蓄積 → ブロック間
`num_blocks` 順の単純加算）を CPU 側で明示的に再現した。
`assert_rmsnorm_backward_dw_split_parity` の dw 判定はこの関数の出力と
比較するよう変更し、9.5 の `cpu_rmsnorm_backward_reference`（単一の行順
逐次和）を dw 判定には使わないことにした（dx 判定は num_blocks に依存
しない独立カーネル `RMSNORM_BWD_DX_F32` を経由するため従来どおり
`cpu_rmsnorm_backward_reference` を使い続けて問題ない）。

- `num_blocks == 1` では `cpu_rmsnorm_dw_split_reference` は単一ブロックの
  行順逐次 `mul_add` 蓄積へ退化し、最終加算が `0.0 + partial ==
  partial`（浮動小数点の加法単位元。精度損失なし）となるため、
  `cpu_rmsnorm_backward_reference` の dw 出力と bit-for-bit 一致する
  （CPU 専用回帰テスト `cpu_dw_split_reference_matches_naive_reference_
  when_num_blocks_is_one` で確認。実機不要・通常 CI で実行）
- `rows` が `num_blocks` を割り切らない末尾ブロック（空範囲）の扱いも
  GPU 側コメント「末尾要素ブロックの扱い」と対応させ、CPU 専用回帰テスト
  `cpu_dw_split_reference_handles_non_divisible_num_blocks` で形状・
  有限性を確認した
- `RELATIVE_TOLERANCE`・`ABSOLUTE_RESCUE_THRESHOLD`・複合判定式・ケース表・
  シードは引き続き一切変更していない
- dx 側・非 split（`num_blocks=1` の `RMSNORM_BWD_DW_F32`）側は本対応の
  影響を受けない（dx は独立カーネル、`num_blocks=1` は上記のとおり
  bit-for-bit 退化）ため非後退が保たれる設計である

**実機確認コマンド（次回 GB10 セッション）**:

```sh
cargo test -p fandhe-ai-backend-cuda --release --test rmsnorm_backward_parity \
  -- --ignored --nocapture rmsnorm_backward_dw_split_matches_cpu_across_shapes
cargo test -p fandhe-ai-backend-cuda --release --test rmsnorm_backward_parity \
  -- --ignored --nocapture rmsnorm_backward_matches_cpu_across_shapes
make test-ignored-cuda
```

実機確認が完了し次第、本節に実測値（fail_count・max_abs_diff 等）を追記し、
イシュー #1102／#1105 のクローズ可否を判断する。

### 9.7 方針転換: テスト弱体化ではなくカーネル精度改善で解消する（PR #1120 codex P1・Bugbot Low 対応）

9.5〜9.6 の対応は「CPU 参照実装を GPU の内部縮約順序（`rstd` の
warp butterfly reduction・dw split-K の二段縮約）に合わせる」方式だった。
PR #1120 の codex-review が P1 として以下を指摘した:

> GPU forward の `rstd` を CPU 参照へ流用すると「独立 CPU 参照による
> forward→backward end-to-end 一致」の検査が失われる。`rstd` 単体 +
> 同一 `rstd` 下 dw/dx を別々に通しても、`rstd` の許容内差が `rows`
> 方向に蓄積した dw 誤差を拘束できない。カーネル単体検査の追加は可だが、
> 独立 CPU 参照の end-to-end 判定は残し、**実装側を修正して**統一複合
> 判定を満たせ、という内容。

この指摘は妥当と判断し、方針を転換した。

**新方針（tolerance 変更なし・独立 end-to-end 検査を弱体化しない）**:

1. **テスト側**: `assert_rmsnorm_backward_dw_split_parity`
   （`crates/backend-cuda/tests/rmsnorm_backward_parity.rs`）の**主検査**を
   独立 CPU 参照実装による end-to-end parity（`rstd`・dw・dx のいずれも
   GPU の内部実装詳細を一切参照しない独立計算。9.5 以前の元の契約）へ
   戻した。9.5・9.6 で追加した「GPU `rstd` を流用したカーネル単体検査」
   （`rstd` バッファ parity・同一 `rstd` 供給下の dx／dw カーネル単体
   検査。`cpu_rmsnorm_dw_split_reference` を含む）は**追加検査**として
   残し、主検査を置き換えない構成にした（イシュー #1102／#1105 の
   切り分けで有用性が実証済みのため削除はしない）
2. **カーネル側の精度改善**（`crates/backend-cuda/src/kernels_rmsnorm.rs`）:
   forward（`RMSNORM_F32_ONEPASS`／`RMSNORM_F32_TWOPASS`）の二乗和
   `acc` を `float` から **`double` アキュムレータ**へ変更した。レーン内
   部分和（`fma((double)v, (double)v, acc)`）・warp shuffle 縮約
   （`__shfl_xor_sync` は CUDA の組み込みオーバーロードで `double`
   〈8 byte〉に対応）・最終の除算＋平方根（`1.0 / sqrt(fma(acc,
   (double)inv_n, (double)eps))`）のすべてを `double` で行い、`rstd` へ
   代入する 1 回だけ `float` へ downcast する。SMEM への正規化前データ
   格納・出力・`rstd` 自体の型契約（`float`）は変更していない
3. **設計判断の根拠**: `hidden` 程度（数千要素）の総和では、`double`
   （53 bit 仮数）の丸め誤差は最終 `float32` downcast の 1 ULP 未満に
   収まる。9.2 実測の GPU/CPU `rstd` 相対差 `max_rstd_rel_delta =
   2.06e-6`（float32 同士の縮約順序差由来）は `double` 化により実質的に
   消滅する見込みであり、FAIL の余裕が僅か（9.5 実測: `max_rel_err =
   1.315e-3` に対し許容 `1e-3`、`max_abs_diff = 3.052e-4`）だったことから
   `rstd` 精度改善のみで複合判定を満たせる可能性が高いと判断した
   （dw split-K の部分和・最終縮約自体の `double` 化は、この `rstd`
   精度改善で解消しない場合の次段対応として保留する。9.6 に記録した
   `cpu_rmsnorm_dw_split_reference`〈追加検査〉は保留中もそのまま有効）
4. **他バックエンド・既存契約との整合**: FMA 契約統一
   （`.claude/rules/coding-rust.md`「CPU 参照実装は `f32::mul_add`」）は
   matmul 等の積和契約の話であり、縮約途中の**精度**（`double`
   アキュムレータの採用）を複合判定の許容範囲内で引き上げることとは
   独立の軸である。CPU 参照実装・`backend-cpu` 本番実装・Metal
   バックエンドはいずれも `rstd` を `f32` で計算する契約のままであり、
   本変更は「GPU 側の実装精度を独立 CPU 参照実装の精度により近づける」
   ものであって、バックエンド間の丸め契約自体（FMA 使用・`f32` 型）を
   変更するものではない。したがって既存の tolerance・複合判定の設計
   意図（実装依存の縮約順序差を許容しつつ実質的なバグは検出する）とも
   矛盾しない
5. **性能への影響**: forward カーネルは行あたり 1 回の二乗和縮約（`hidden`
   要素分のロード + 蓄積）でありメモリ帯域律速（`kernels_rmsnorm.rs`
   モジュールコメント「設計」節）。`double` 演算は SM の FP64 ユニットを
   使うが、演算数自体は `hidden` 要素程度でメモリロード量（`float4`
   ベクトル化ロードは変更なし）と比べて軽微であり、帯域律速の性質上
   全体スループットへの影響は無視できると見積もる（実機での定量計測は
   次回 GB10 セッションで REQ-8 性能下限との非後退確認と合わせて行う）
6. **Bugbot Low 対応**: `cpu_rmsnorm_rstd_reference`（追加検査で使う
   rstd バッファ parity 用の独立参照）が `hidden == 0` で `0.0f32` 埋め
   していたが、GPU 側の実際の契約（`rmsnorm.rs::run_rmsnorm_f32_inner`
   の早期 return。`hidden == 0` では `sum(x^2) == 0` が数学的に確定する
   ため `rstd = 1.0 / eps.sqrt()` を全行同一値で返す。カーネル自体は
   この縮退ケースでは起動されない）と不一致だった。`1.0f32 /
   eps.sqrt()` を返すよう修正した
7. CUDA カーネルの変更は NVRTC ソース文字列（Rust の `&str` 定数）の
   変更であり、Mac 環境では実機コンパイル確認ができない。ソース文字列
   の存在検査（`forward_kernels_do_not_use_approximate_rsqrtf`。`double
   acc = 0.0;`・新 `rstd` 計算式の文字列を回帰検出するよう更新済み）・
   CPU 側テスト・`cargo fmt`／`cargo clippy --workspace --all-targets
   --all-features -- -D warnings` の通過をローカルで確認した。GB10
   実機でのコンパイル成立・数値実測はコーディネータが実施する

**実機確認コマンド（変更なし。上記参照）**。実機確認完了後、本節に
実測値を追記し、`rstd` 精度改善のみで解消したか、dw split-K 側の
`double` 化が追加で必要かを判断する。

### 9.8 dw 縮約自体の double 化（PR #1120 追加コミット。GB10 実機実測 2026-09-01）

**GB10 実機再検証結果（0db80f6。§9.7 の rstd `double` 化のみを適用した
コミット）**: `rmsnorm_backward_matches_cpu_across_shapes`（非 split）は
pass した。しかし `rmsnorm_backward_dw_split_matches_cpu_across_shapes` は
旧ケース `(rows=8192, hidden=4096, num_blocks=1)` で FAIL 継続
（`fail_count=5/4096・max_abs_diff=3.052e-4・max_rel_err=1.345e-2`。
`rstd` の `double` 化前〈§9.4〉の `max_rel_err=1.340e-2` とほぼ同水準——
`rstd` 精度改善だけでは実質的な改善が見られなかった）。

**切り分け**: `rstd` 自体はほぼ厳密な値になった一方、dw の
`rows=8192` に渡る `f32` 逐次和（GPU: `fmaf` 蓄積／CPU 参照: `mul_add`
蓄積）が、相殺を含む出力要素（列）で支配的な誤差になっていた。GPU・CPU
とも同一の行順逐次蓄積だが、`rstd[r]`（行ごとのスカラー）が GPU
（warp butterfly reduction 由来）と CPU（逐次和由来）で `double`
精度でも非結合性に起因するごく僅かな差を持ち続ける限り、`rows` 方向へ
8192 回積み上げる `f32` の逐次和はこの僅かな入力差を打ち消しきれず、
特定の出力列（強い相殺が起きる列）で複合判定を超える絶対誤差へ増幅
されうる。

**ルーティングの実装確認（重要な訂正）**: 当初の対応方針は
`RMSNORM_BWD_DW_PARTIAL_F32`／`RMSNORM_BWD_DW_REDUCE_F32`（split-K
経路。`num_blocks >= 2`）の `double` 化のみを想定していたが、
`rmsnorm.rs::run_rmsnorm_bwd_f32_inner` のホスト側分岐
（`if num_blocks <= 1 { 単段カーネル RMSNORM_BWD_DW_F32 } else {
split-K 二段構成 }`）を確認したところ、**GB10 で FAIL していた
`(rows=8192, hidden=4096, num_blocks=1)` は `num_blocks <= 1` のため
実際には単段カーネル `RMSNORM_BWD_DW_F32`（split-K 側とは別カーネル）
を経由しており、split-K 側だけを `double` 化しても本ケースの FAIL は
解消しない**ことが判明した。このため対応範囲を単段カーネルへ拡張した
（`.claude/rules/out-of-scope-tracking.md` の趣旨に沿い、実装時に判明
した前提の誤りとして本節に明記する。既存の対応方針〈split-K 側の
`double` 化〉自体は誤りではなく、`num_blocks >= 2` の経路に引き続き
必要な改善のため維持する）。

**適用した修正（`crates/backend-cuda/src/kernels_rmsnorm.rs`。
tolerance 非変更）**:

1. **`RMSNORM_BWD_DW_F32`（単段フォールバック。`num_blocks <= 1` の
   実効経路）**: `rows` 方向の逐次蓄積 `acc` を `float` から `double`
   アキュムレータへ変更した（`fma((double)dyv * (double)r, (double)xv,
   acc)`。`dw[i]` へ代入する 1 回だけ `float` へ downcast）
2. **`RMSNORM_BWD_DW_PARTIAL_F32`（split-K 第 1 段。ブロック内蓄積）**:
   同様に `acc` を `double` 化。**部分和バッファ `dw_partial` 自体も
   `float` から `double` へ変更**した（ホスト側 `rmsnorm.rs` の
   `alloc_zeros::<f64>` に対応。中間で `float` へ downcast すると
   ブロック間縮約〈次段〉に渡す前に精度を失うため、バッファ型ごと
   `double` 化する設計を選んだ——コーディネータ指示の「partial 出力を
   float のまま downcast すると精度が落ちるので、可能なら partial
   バッファを double 化」に従う）
3. **`RMSNORM_BWD_DW_REDUCE_F32`（split-K 第 2 段。ブロック間縮約）**:
   `dw_partial` の読み出し型を `double` に合わせ、smem double buffer
   （`smem[2][4][256]`）・縮約アキュムレータ `acc` を `double` へ変更。
   `dw[col]` へ書く epilogue でのみ `float` へ downcast する
4. **ホスト側（`rmsnorm.rs`）**: `dw_partial_dev` の確保を
   `alloc_zeros::<f32>` から `alloc_zeros::<f64>` へ変更。部分和
   バッファのバイト数計算（`derive_dw_split` の `fits_budget`・
   `validate_dw_split_launch` の `partial_bytes`）の乗数を `4`（`float`）
   から `8`（`double`）へ更新した

**CPU 参照実装側の強化（`crates/backend-cuda/tests/
rmsnorm_backward_parity.rs`。テスト弱体化ではなく精度強化。codex P1
の趣旨「独立参照 + 実装側修正」に合致）**:

- `cpu_rmsnorm_backward_reference`（主検査が使う独立参照）: 二乗和
  （`rstd` 計算。`gpu_rstd = None` の場合）・dw の行方向蓄積を
  いずれも `f64` アキュムレータへ変更した。`dx` の計算式（`dot` の
  蓄積・最終の `dx_row[i]` 式）は変更していない（コーディネータ指示:
  dx 参照はそのままで可。dx は既に GB10 で pass しており、対応する
  GPU 側 `RMSNORM_BWD_DX_F32` も本 PR で変更していない）
- `cpu_rmsnorm_rstd_reference`（追加検査の rstd バッファ parity 用）:
  同様に `f64` アキュムレータ化した
- `cpu_rmsnorm_dw_split_reference`（追加検査の split-K dw カーネル単体
  検証用）: ブロック内・ブロック間の蓄積をいずれも `f64` 化し、GPU
  側の対応する 2 カーネルと精度前提を揃えた
- 独立性は維持している: いずれの関数も GPU の縮約順序（`rstd` の
  warp butterfly reduction・dw のブロック分割）を模倣するものではなく、
  素朴な逐次和（`rstd`・`cpu_rmsnorm_backward_reference` の dw）または
  独立に定義したブロック分割順序（`cpu_rmsnorm_dw_split_reference`。
  これは「追加検査」専用でありこの関数自体は元々 GPU の縮約順序を
  模倣する設計だったため対象外）のままである。変更したのは中間計算の
  **精度**（`float` → `double`）のみであり、GPU 実装のコピーではない

**回帰テストの更新**: `kernels_rmsnorm.rs` 側のソース文字列検査
（`split_k_dw_reduce_smem_size_matches_batch_and_block_dim_consts`・
`split_k_dw_reduce_writes_dw_exactly_once_in_epilogue` 等）を新しい
型・計算式に合わせて更新した。`rmsnorm.rs` 側のバッファ上限テスト
（`derive_dw_split_falls_back_when_partial_buffer_exceeds_cap`）も
`double` 要素サイズ（8 byte）に合わせて期待値を再計算した
（`num_blocks=16` → `8`。乗数 `4` → `8` により上限に収まる block 数が
半減する）。テスト用の CPU 専用回帰テスト
（`cpu_dw_split_reference_matches_naive_reference_when_num_blocks_is_one`）
は `f64` 化後も bit-for-bit 一致を維持することをローカルで確認済み
（`num_blocks == 1` では `0.0 + partial == partial` の加法単位元により
精度損失なく退化するため）。

**性能への影響の見積もり**: forward 同様、dw カーネル群もメモリ帯域
律速の性質を持つ（行あたり `hidden` 要素のロード＋蓄積、または
`num_blocks * hidden` 要素の部分和書き出し／読み出し）。`double` 演算
自体の追加コストは軽微と見積もるが、**split-K 経路は部分和バッファの
サイズが 2 倍化する**（`float` → `double`。`RMSNORM_DW_PARTIAL_BUFFER_
CAP_BYTES` は 64 MiB のまま変更していないため、同一 `hidden` に対し
選択される `num_blocks` の上限がおよそ半分になる——実測例:
`derive_dw_split(20_000, 100_000, 1_000_000)` は `16` から `8` へ
減少。これにより極端に広い `hidden` を持つ形状では split-K の並列度が
下がり性能へ影響しうる）。またブロック間縮約カーネルの静的 smem
使用量も 8 KiB から 16 KiB へ倍増したが、GPU の静的 smem 予算には
依然余裕を持って収まる。実機での定量計測（REQ-8 性能下限との非後退
確認を含む）は次回 GB10 セッションで実施する

**実機確認コマンド（変更なし。§9.7 参照）**。実機確認完了後、本節に
実測値（`fail_count`・`max_abs_diff` 等）を追記し、イシュー
#1102／#1105 のクローズ可否・REQ-8 性能非後退を判断する。

**GB10 実機再検証結果（コミット `1a356ea`。コーディネータ実行）**:
`rmsnorm_backward_matches_cpu_across_shapes`・`rmsnorm_backward_dw_split_
matches_cpu_across_shapes` の 2 テストが pass した。旧 FAIL ケース
`(rows=8192, hidden=4096, num_blocks=1)` は `fail_count=5/4096 → 0` へ
解消した。dw の `rows` 方向蓄積（単段カーネル `RMSNORM_BWD_DW_F32`。
9.8 のルーティング確認で判明したとおり `num_blocks<=1` の実効経路）を
`double` アキュムレータ化したことが、この特定ケースの解消に直接寄与した
ことを実機で確認した。

### 9.9 契約改定: 正規化統計・勾配の長軸縮約を全バックエンドで f64 統一（ユーザー承認 2026-09-01）

**背景**: 9.7〜9.8 の対応は CUDA カーネルと CPU 参照実装（テスト専用の
独立参照実装）のみを `f64`（`double`）化しており、`backend-cpu`・
`backend-metal` の**本番実装**（`crates/backend-cpu/src/rmsnorm.rs`・
`crates/backend-metal/src/shaders/rmsnorm.metal`）は `f32` のままだった。
これは「CUDA と参照実装だけを片側で `f64` 化する」契約の非対称性であり、
codex-review が P1 として 2 件（CUDA 側・CPU 参照実装側）指摘した内容の
根本原因でもある。ユーザー承認（2026-09-01）により、**正規化統計・勾配
の長軸縮約（rmsnorm の `rstd` 二乗和・dw の行方向蓄積等）は全バックエンド
で `f64` アキュムレータ契約へ統一する**（Metal は `double` 型非対応の
ため Kahan 補償和 `f32` を「`f64` 相当」の実装形として適用する）ことで
この非対称性を解消した。matmul 系の FMA 契約（CPU 参照 `f32::mul_add`）
は不変（`AGENTS.md`・`.claude/rules/coding-rust.md` に同内容を追記済み）。

**実装の棚卸し**: rmsnorm 実装は 3 バックエンドすべてに存在する
（`backend-cpu`・`backend-cuda`・`backend-metal`）。同型の「長軸縮約」を
持つ他の演算（softmax 等）は本イシューのスコープ外（rmsnorm の `rstd`・
dw のみが対象と確認済み。softmax の縮約〈max・sum〉は別イシューで扱う
判断とし、本 PR では変更していない）。CPU・Metal とも rmsnorm の
**backward（dw）専用カーネルは存在しない**（`backend-cpu`・
`backend-metal` いずれも forward のみを自作カーネルとして持ち、backward
は汎用 autodiff 合成〈elementwise・reduction〉経由。ダミー「同型演算の
列挙」としてはこの事実を記録するに留め、変更は forward の `rstd` のみに
限定した）。

**`backend-cpu`（`crates/backend-cpu/src/rmsnorm.rs`）**:

- スカラー経路（`rmsnorm_row_scalar`）: 二乗和の蓄積を `f32` から `f64`
  アキュムレータへ変更（要素の二乗自体も `f64` へ昇格してから計算し、
  CUDA 側の `fma((double)v, (double)v, acc)` と精度特性を揃えた）。
  `rstd` へ代入する 1 回だけ `f32` へ downcast する
- NEON 経路（`rmsnorm_row_neon`。aarch64 限定）: `vcvt_f64_f32`／
  `vcvt_high_f64_f32` で `float32x4_t` チャンクを下位・上位 2 要素ずつ
  `float64x2_t` へ拡張し、`vfmaq_f64`（倍精度 NEON FMA。ARMv8-A の
  Advanced SIMD がベースラインで倍精度演算を含むため追加の
  `#[target_feature]` 指定は不要）で二乗和を蓄積する。端数は `f64`
  逐次和で処理する
- 実機実測（Apple M4 Max・aarch64。本 PR 作業環境）: `cargo test -p
  fandhe-ai-backend-cpu` 全 pass（`neon_matches_scalar_various_hidden`
  という NEON/スカラー A/B 同値テストを含む。両経路とも `f64` 化後も
  1e-5 の既存許容誤差内で一致することを確認済み——`f64` 化は両経路を
  真値へ近づける変更のため、互いの差は `f64` 化前より縮む方向）

**`backend-metal`（`crates/backend-metal/src/shaders/rmsnorm.metal`）**:

- forward の 1 パス／2 パス両カーネル（`rmsnorm_f32_onepass`／
  `rmsnorm_f32_twopass`）の二乗和蓄積を、単純な `fma()` 直接蓄積から
  **Neumaier 改良版 Kahan 補償和 + scale/ssq 方式**（`rmsnorm_ssq_add`・
  `rmsnorm_ssq_combine`・`rmsnorm_reduce_ssq`・`rmsnorm_finalize_rstd`）
  へ変更した。レーン内蓄積（各レーンの `hidden/32` 要素程度の逐次和）・
  32 レーン間 butterfly 縮約（5 段シャッフルで `scale`・`ssq`・補償項
  `comp` の 3 つを交換し `rmsnorm_ssq_combine` で合成）の両方に適用した
  （CUDA 側「レーン内部分和を `double` 化・warp shuffle を `double` で
  行う」設計と対応する）
- Apple GPU family は `double` 型をサポートしないため MSL では `f64`
  アキュムレータを直接使えない。scale/ssq 方式（LAPACK SLASSQ 系の
  overflow-safe な二乗和アルゴリズム: 最大絶対値を `scale` として括り
  出し、残りを `(a/scale)^2`〈常に `[0,1]`〉として `ssq` へ蓄積する。
  各ステップの `ssq` 加算に Neumaier 改良版 Kahan 補償和を併用する）を
  「`f64` 相当」の実装形として適用する
- **当初は scale/ssq 方式を用いず単純な Kahan 補償和のみを適用していた
  が、codex-review が P1 として「`v.x * v.x` を `f32` のまま先に計算する
  ため、有限入力（例 `2e20f`）でも二乗が `f32` の表現範囲（最大約
  3.4e38）を超えて `inf` になり、`inf - inf` の Kahan 補償計算で `NaN`
  が発生する。CUDA・CPU は要素を `f64` へ昇格してから二乗するため有限
  値を保ち、意味論が片側で割れている」と指摘した**（2 度目の P1。1 度目
  は本節冒頭の「CUDA と参照実装だけの片側 `f64` 化」）。この指摘を受け
  scale/ssq 方式へ実装を訂正した。二乗を直接計算しない（比の二乗のみを
  計算する）ため、有限入力である限り中間計算が `f32` の表現範囲を超えて
  overflow することがない
- 最終的な `rstd` 導出（`rmsnorm_finalize_rstd`）も `scale` を明示的に
  二乗しない形（`rstd = 1 / (scale * sqrt(ssq * inv_n))`）へ整理した
  （`scale` が `2e20` 級の場合 `scale^2` 自体が `f32` 表現範囲外になる
  ため）。`eps` は最終段で別途加算するのではなく `sqrt(eps * n)` という
  追加の疑似要素として同じ `rmsnorm_ssq_add` へ通すことで、実要素の
  二乗和と同じ overflow-safe な経路に統一した（`scale == 0`〈実要素が
  全て 0〉かつ `eps > 0` のケースでも、この疑似要素が `scale` を
  `sqrt(eps * n)` へ更新するため、特別分岐なしに従来どおり
  `rstd = 1/sqrt(eps)` へ帰着する）
- **極値入力の実機テストを追加**（`tests/rmsnorm_parity.rs::
  rmsnorm_matches_backend_cpu_with_extreme_magnitude_values`。`#[ignore]`
  Metal 実機依存）: `hidden=17`（スカラー経路を含む非 4 の倍数）の行に
  `2e20f`／`-1.5e20f`（`f32` の二乗が overflow する大きさ）・`1e-38f`／
  `-3e-39f`（非正規化数級の極小値）を混在させ、GPU 出力が全て有限値
  （`is_finite()`）であること、および `backend-cpu`（`f64` 化済み）との
  複合判定一致を検証する。tolerance は変更していない
- 実機実測（Apple M4 Max・実機 GPU。本 PR 作業環境）:
  `cargo test -p fandhe-ai-backend-metal --release --test rmsnorm_parity
  -- --ignored --nocapture` 4 テスト全 pass（`rmsnorm_matches_cpu_
  across_shapes`・`rmsnorm_matches_backend_cpu_directly`〈`f64` 化した
  `backend-cpu` 本番実装との直接比較〉・`rmsnorm_run_fused_matches_cpu_
  composed`・上記の極値入力テスト）。`cargo test -p fandhe-ai-backend-
  metal --release -- --ignored --test-threads=1`（rmsnorm 以外を含む
  実機依存テスト全件）も全 pass（非後退確認）
- ソース文字列証跡テスト（`tests/rmsnorm_softmax_source_evidence.rs`）を
  更新: `rmsnorm_uses_fma_for_accumulation` →
  `rmsnorm_uses_overflow_safe_ssq_for_accumulation`（`rmsnorm_ssq_add`
  ヘルパー使用の検査に加え、単純な `v.x*v.x` 直接二乗〈Kahan 補償のみ〉
  への後退も明示的に検出する）、butterfly reduction の 5 段検査を
  softmax 側（展開済み 5 行。従来どおり）と rmsnorm 側（`scale`・`ssq`・
  `comp` 3 つの shuffle を検査する `rmsnorm_uses_five_stage_ssq_
  butterfly_reduction`）に分離した

**tolerance・ケース表・シードは一切変更していない**（`RELATIVE_TOLERANCE`・
`ABSOLUTE_RESCUE_THRESHOLD`・複合判定式は不変）。本節の変更はいずれも
「実装側の精度をより真値へ近づける・意味論を他バックエンドと揃える」
対応であり、CPU 参照実装の `f64` 化（9.8）と同じ精神——独立参照・各
バックエンド実装ともに精度・overflow 安全性を引き上げることで、縮約
順序の非結合性に由来する誤差を許容範囲内へ収める——に立つ。

### 9.10 codex-review 3 件目の P1 一括是正 + 4 件目の inf+inf 追加是正（PR #1120。eps overflow・NaN 非伝播・契約文言の精密化・inf+inf 汚染）

PR #1120 に対し codex-review が同時に P1 を 3 件指摘した。すべて妥当と
判断し 1 コミットで是正した。

**1. `rmsnorm.metal`: `eps * n` の中間 overflow**（`rmsnorm_finalize_
rstd`）: `eps` を `sqrt(eps * n)` という疑似要素として `scale/ssq` へ
折り込む際、`eps` が `f32::MAX` 級・`hidden`（`n`）がある程度大きい場合
に、平方根を取る前の `eps * n` 自体が `f32` の表現範囲（最大約 3.4e38）
を超えて `inf` になり、最終的な `rstd` が誤って `0` になっていた（CPU/
CUDA は `f64` で `sum_sq*inv_n + eps` を計算するため有限値を保つ）。
`sqrt(eps * n) = sqrt(eps) * sqrt(n)`（数学的に同値な恒等式。両辺とも
非負）という変形を使い、`eps`・`n` をそれぞれ個別に平方根を取ってから
掛け合わせることで中間 overflow を回避した（`eps`・`n` はいずれも個別
には `f32` の表現範囲内の有限値であるため、それぞれの平方根も表現範囲
内に収まる）。実機テスト
`tests/rmsnorm_parity.rs::rmsnorm_matches_backend_cpu_with_extreme_eps`
（`eps = f32::MAX`・`hidden = 17`）を追加し、GB10 実機実測: Apple M4 Max
実機で pass を確認（GPU 出力が有限・非ゼロであり `backend-cpu`〈`f64`
化済み〉と複合判定一致）。

**2. `rmsnorm.metal`: NaN 入力の非伝播**（`rmsnorm_ssq_add`／
`rmsnorm_ssq_combine`）: scale/ssq 方式の `a > scale` 比較は IEEE 754 の
規則により `a` が `NaN` の場合常に偽になる。`scale` が未だ `0.0f`
（このレーンで最初に処理される要素が `NaN` だった場合）だと `scale >
0.0f` も偽になり、いずれの分岐にも入らず `NaN` 入力が黙って捨てられて
いた（CPU/CUDA の `f64` 逐次和は `NaN` を含む行全体が `NaN` になる意味論
のため、これは意味論の不一致だった。実測で再現・確認済み——一時的に
NaN 検出分岐を無効化して再実行したところ、`NaN` 要素がそのまま出力へ
素通りし他の要素は正常な正規化値になるという、まさに指摘どおりの症状を
確認した）。`rmsnorm_ssq_add` に `isnan(a) || isnan(ssq) || isnan(scale)`
の検出を追加し、`ssq` を `NaN` へ確定・`scale` を**有限の正値**
（`1.0f`）へ強制する（`scale` を `0.0f` のままにすると
`rmsnorm_ssq_combine` の `other_scale == 0.0f` 早期 return で汚染情報が
握り潰されるため）。`rmsnorm_ssq_combine` にも同型の `isnan(ssq) ||
isnan(other_ssq)` 検出を追加した。**inf 入力**（`scale` が `+inf`）は
既存のアルゴリズムで自然に正しく扱える（`scale = inf` が最終的に
`rstd = 0` へ伝播する。CPU/CUDA の `f64` 意味論と一致）が、**両者とも
`+inf` の状態を結合する**特殊ケース（`ratio = other_scale / scale =
inf/inf = NaN` になり誤って `NaN` へ汚染してしまう）のみ
`isinf(scale) && isinf(other_scale)` の明示分岐で `+inf`（IEEE 754 の
`inf + inf = inf` に倣う）のまま維持するよう対応した。実機テスト
`tests/rmsnorm_parity.rs::rmsnorm_propagates_nan_matching_backend_cpu`
（先頭要素が `NaN` の行・非先頭要素が `NaN` の行・`NaN` なしの対照行の
3 行構成）を追加し、Apple M4 Max 実機で pass を確認。`crates/backend-cpu/
src/rmsnorm.rs` にも `run_rmsnorm_f32_propagates_nan_for_row_with_nan_
element`（CPU 単体・実機不要）を追加し、全バックエンドで NaN 伝播の
意味論が一致することを記録した。

**3. `kernels_rmsnorm.rs`: 要素積の `double` 化が契約文言と矛盾**: 「正規
化統計・勾配の長軸縮約は `f64` アキュムレータで統一する（要素ごとの
積和自体は `f32` のまま）」という契約文言に対し、`RMSNORM_BWD_DW_F32`・
`RMSNORM_BWD_DW_PARTIAL_F32` の実装が `acc = fma((double)dyv *
(double)r, (double)xv, acc);`（要素を先に `double` へ昇格してから積を
取る）になっており、文言と不一致だった。切り分けの結果、この不一致は
2 系統の縮約が本質的に異なる扱いを要することに起因すると判断した:
forward の二乗和は要素を `f32` のまま二乗すると overflow しうる
（Metal 側の scale/ssq 方式採用理由と同根）ため `f64` 昇格後に二乗する
必要がある一方、dw の 3 項積（`dy・rstd・x`）はそのような overflow
リスクが実用上小さいため、契約文言どおり要素積を `f32` で確定してから
`f64` へ昇格するのが妥当と判断した。**実装を契約文言（dw 側）に合わせて
是正**した（`float term = dyv * r * xv; acc = (double)term + acc;`）。
forward の二乗和側は実装（`f64` 昇格後に二乗）を維持し、**契約文言の側
をこの区別を明示する形へ精密化**した（`AGENTS.md`・`.claude/rules/
coding-rust.md`。本節冒頭を参照）。`kernels_rmsnorm.rs` 冒頭に
「精度契約の精密化（§9.10 追補）」節を追加し、実装の根拠をコード内にも
記録した。回帰テスト `dw_kernels_finalize_element_product_in_f32_before_
double_accumulation`（`RMSNORM_BWD_DW_F32`・`RMSNORM_BWD_DW_PARTIAL_F32`
双方のソース文字列を検査。実機不要）を追加した。テスト参照側
（`crates/backend-cuda/tests/rmsnorm_backward_parity.rs` の
`cpu_rmsnorm_backward_reference`・`cpu_rmsnorm_dw_split_reference`）も
同じ契約（要素積を `f32` で確定してから `f64` へ蓄積）へ統一した
（forward の二乗和側の CPU 参照実装は変更していない）。

**tolerance・ケース表・シードは一切変更していない**。実機検証は本作業
セッションが Apple M4 Max・macOS だったため Metal 側は実機で直接確認
できた（`cargo test -p fandhe-ai-backend-metal --release --test
rmsnorm_parity -- --ignored --nocapture` 全 6 テスト pass。うち 3 は本節
の是正で新規追加）。CUDA 側（`kernels_rmsnorm.rs` の要素積修正）は
ソース文字列検査・CPU 側テストで確認済みだが、GB10 実機での最終確認は
コーディネータに委ねる（確認コマンドは §9.7〜§9.8 と同一）。

**4. `rmsnorm.metal`: `scale == +inf` の状態に `a == +inf` が続くと NaN
汚染（codex-review P1 + Bugbot Medium。同根の 1 件。PR #1120 追加是正）**:
上記 2 番目の NaN 是正（`isnan` 検出）とは別に、`scale` が既に `+inf`
（先行する inf 要素で確定済み）の状態で次の要素 `a` も `+inf` だった
場合、`a > scale` が `inf > inf` で偽になり `else if` 分岐へ入る。そこで
計算する `ratio = a / scale = inf / inf` は IEEE 754 の不定形で `NaN`
になり、`rmsnorm_kahan_add(ssq, comp, ratio * ratio)` を通じて `ssq` を
`NaN` へ汚染してしまう。CPU/CUDA の `f64` 意味論では `inf + inf = inf`
（有限の不定形にならない）ため二乗和は `inf` のまま、`rstd` は有限の
`0` になり（`NaN` にはならない）、意味論が不一致だった。

`rmsnorm_ssq_add` に `isinf(scale) && isinf(a)` の明示分岐を追加し
（`isnan` チェックの直後・通常のリスケール分岐より前）、この場合は
リスケール計算自体を行わず `scale` を `+inf` のまま維持して `ssq` のみ
Neumaier 加算で更新する（呼び出し元は `fabs(v)` を渡すため `a` は常に
非負であり `isinf(a)` は必ず `+inf` を意味する。`rmsnorm_ssq_combine`
側は既存の `isinf(scale) && isinf(other_scale)` 分岐〈本節 3 番目是正の
延長で既に導入済み〉がレーン間結合の同型ケースを担っており、今回の
是正はレーン内蓄積（`rmsnorm_ssq_add`）側の抜け漏れを埋めるもの）。

**実機テスト**（`tests/rmsnorm_parity.rs::
rmsnorm_matches_backend_cpu_with_multiple_inf_in_same_row`）を追加した:
`hidden=65`（非 4 の倍数のためスカラー経路。`hidden > 32` でレーン割当
`idx % 32` が複数レーンにまたがる）で、**同一レーン配置**（`idx=0`・
`idx=32`。いずれもレーン 0 が同一ループ内で逐次処理し
`rmsnorm_ssq_add` を直撃する）と**別レーン配置**（`idx=0` はレーン 0・
`idx=1` はレーン 1。`rmsnorm_ssq_combine` の inf+inf 分岐を経由する）の
両方を検証する。`+inf` 要素自体の出力は `x_i * rstd(=0) = inf * 0 =
NaN`（IEEE 754 の不定形。CPU 参照実装でも同じ `NaN` になる契約）になる
一方、それ以外の有限要素の出力は `finite * 0 = 0.0`（厳密な等式）に
なるという区別を踏まえ、位置ごとに直接値検査で判定する（`tolerance`
は使わない）。

**回帰の実機再現確認**: 是正前に `isinf(scale) && isinf(a)` 分岐を一時的に
無効化して同テストを実行したところ、行 0（同一レーン配置）で有限要素の
GPU 出力が `NaN` になる（指摘どおりの症状）ことを Apple M4 Max 実機で
確認したうえで、是正を復元し全テスト pass を再確認した。

**実機実測（Apple M4 Max・macOS。本節 4 番目の是正時点）**: `cargo test
-p fandhe-ai-backend-metal --release --test rmsnorm_parity -- --ignored
--nocapture` 全 7 テスト pass（新規追加の
`rmsnorm_matches_backend_cpu_with_multiple_inf_in_same_row` を含む）。
`cargo test -p fandhe-ai-backend-metal --release -- --ignored
--test-threads=1`（実機依存テスト全件）も全 pass（非後退確認）。
`tolerance`・ケース表・シードは一切変更していない。

## 10. TF32/f16 mma・wmma・tensor_core_real_device テスト群の GB10 失敗解消（イシュー #1106）

本節は §1〜8（GEMM `wmma`/`mma` 系ベースライン表・非後退契約本体）の続き
として、8.3 で実機不達のまま申し送られた `WmmaTf32`（基本版）2 行の
provenance 未確定 fail-closed、および §5.3 記載の恒常 fail テスト群の
GB10 実機実測・解消記録を追記する（イシュー #1102 配下・親 #1007）。

### 10.1 実測環境

- ノード: DGX Spark GB10（実ホスト名は `docs/real-hardware-verification-env.local.md`
  参照。`.gitignore` 対象）。sm_121・CUDA 13.0.88・driver 580.159.03
- 転送元コミット: `d39e03b0f6f4a6488cfdf79732351b4af256eb13`（`origin/main`。
  `.rev-stamp` で記録）
- 実行方式: `docs/real-hardware-verification-env.md` §3 準拠の rsync 転送
  （`--filter=':- .gitignore'`・`--delete-excluded`・秘密情報・実ホスト名の
  多層除外）+ `env PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH cargo test
  -p fandhe-ai-backend-cuda --release -- --ignored --nocapture`

### 10.2 解消した項目

1. **`WmmaTf32`（基本版）provenance fail-closed の解消**: §8.3 で申し送られた
   2 行を基本版カーネル専用ゲート
   （`gemm::tests::wmma_tf32_basic_kernel_parity_does_not_regress`）で
   release 2 回実測し、既存記録値との完全一致（32×32×32:
   fail_count=154/1024・mean_abs_diff=3.697936e-4／256×256×4096:
   fail_count=10647/65536・mean_abs_diff=4.476030e-3）を確認。
   `baseline_provenance_unconfirmed: false` へ更新（§3・§「既知の限界」
   参照）。`parity_nonregression.rs` の fixture 自己整合テスト
   （`baseline_provenance_unconfirmed_is_scoped_to_unmeasured_paths_only`・
   `wmma_tf32_opt_and_mma_f16_rows_are_fully_enforced`）も全経路 enforced
   の状態へ強化方向で更新した。
2. **非後退監視の併設（3 件・PR #1115 の codex-review P1 指摘を受けて revert 済み）**:
   既存の確定済みベースライン行と形状・シードが完全一致するため新規実機測定が
   不要だった 3 件について、`assert_parity`（green 必須）から
   `assert_no_parity_regression` へ**変換する**変更・CPU 参照実装比較を
   `CudaGemm::run_wmma_tf32` 直接呼び出しとの bit-exact 比較へ**置換する**変更を
   当初適用したが、いずれも REQ-2 統一複合判定を受け入れ条件から外す片側変更
   （AGENTS.md「数値契約の片側変更」・`.claude/rules/coding-rust.md`「バックエンド間
   数値一致テストの許容誤差を単独で緩和しない」に抵触）との codex-review P1 指摘を
   受け、元の受け入れ条件（`assert_parity`・CPU 参照実装比較）を維持したまま
   revert し、非後退監視・配線検証は別テストとして併設する形に修正済み:
   - `tensor_core_real_device.rs::tensor_core_parity_record`（TF32 部分。
     512×512×512 seed=0x7A0 = `WmmaTf32Opt` 既存行）は `assert_parity` を維持し、
     `tensor_core_parity_record_tf32_non_regression` を併設（GB10 実機 release
     2 回 green）。**元の `assert_parity` 自体は §5.3 のとおり恒常 fail のまま
     未解消**（本項目が解消したのは基本版 provenance fail-closed〈上記 1〉のみ）。
     **追記（PR #1115 codex-review 再指摘対応。本項目自体の欠陥）**: この併設
     テストは `wmma_tf32_opt_available()` の確認後に公開 API `run_wmma_tf32` を
     呼んでいたが、対象形状（512×512×512）は cp.async 整列条件
     （`n%4==0 && k%4==0`）を満たすため `run_wmma_tf32` の 3 段選択（staged→
     opt→basic）が staged 経路を最優先で選ぶ（`gemm.rs::run_wmma_tf32`
     ドキュメンテーションコメント参照）。結果として実際には staged 経路の
     結果を `WmmaTf32Opt`（opt カーネル単独の非後退上限）に対して判定して
     しまっており、経路とベースラインが食い違う欠陥があった。opt カーネル
     単独の非後退監視は `fandhe_ai_backend_cuda::gemm::tests::
     wmma_tf32_opt_kernel_parity_does_not_regress`（`src/gemm.rs`。private
     field 経由で 3 段選択を経由せず opt カーネルを直接強制実行し、この
     512×512×512 seed=0x7A0 行を含む全 `WmmaTf32Opt` 行を検査する）が既に
     正しく行っているため、`tensor_core_parity_record_tf32_non_regression`
     は重複かつ誤判定だったとして削除した（実際に選ばれる staged 経路専用の
     非後退監視の追加には 512×512×512 形状の `WmmaTf32Staged` ベースライン
     行の新規実機実測が要るが未実施のため、推定値を書かず本ラウンドでは
     追加していない）
   - `cpu_cuda_mma_parity.rs::mma_f16_k4096_stress`（256×256×4096 seed=9999 =
     `MmaF16` 既存行）は `assert_parity` を維持し、
     `mma_f16_k4096_stress_non_regression` を併設（GB10 実機 release 2 回
     green）。**元の `assert_parity` 自体は §5.3 のとおり恒常 fail のまま未解消**
   - `gemm_tf32_optin.rs::gemm_tf32_optin_on_matches_cpu_across_shapes` は
     CPU 参照実装との tolerance 比較（`assert_tf32_optin_gemm_parity`）を維持し、
     bit-exact 配線検証を `gemm_tf32_optin_on_wiring_matches_run_wmma_tf32`
     （`assert_tf32_optin_wiring_bit_exact`）として併設。併設テストは GB10 実機
     release 2 回 green を確認したが、**復元した元の CPU 参照実装比較（TF32
     経路）自体は本 revert 時点で新規に実機再測定していない**。TF32 経路が
     複数テストで最小形状から恒常的に閾値超過している事実に鑑みると同種の
     fail が再現しうるため、pass/fail 確定は未実施のまま次ラウンドへ引き継ぐ
3. **未解消（新規実機測定を要するため本イシューのスコープ外）**: §5.3
   記載の残り 6 件（`wmma_f16_k4096_stress`〈WMMA〉・`wmma_f16_opt_k4096_stress`・
   `wmma_tf32_k4096_stress_poc_v2_5`・`wmma_tf32_matches_reference_across_shapes`・
   `wmma_tf32_opt_k4096_stress`・`wmma_tf32_opt_matches_reference_across_shapes`）
   に加え、`gemm_wmma_tf32_staged.rs`・`gemm_mma_tf32.rs`（mma_tf32
   系。`docs/perf/cuda-gemm-mma-tf32-ab.md` §8.4 の原因未確定残差）・
   `mma_tf32_vs_wmma_tf32_staged.rs`・`specialized_mma_parity.rs` は複数
   形状にわたる `assert_parity` 直接比較で、既存ベースライン行と一致しない
   形状・シードを含むため非後退契約化には新規ベースライン行の実機測定が
   要る。2026-08-31/09-01 のトリアージ実行で実測データ自体は採取済み
   （fail_count・mean_abs_diff 等。ログはコミットしていない）だが、
   fixture への転記・確定は後続イシューへ引き継ぐ。`tensor_core_tflops_record`
   （性能プロトコル。§5.1・#391 系の既知の不安定）はパリティ判定と無関係
   のため対象外のまま。

### 10.3 REQ-2 限定救済項の扱い

`docs/spec-proposal-req2-req8-revision.md` §5 が提案する専用比較関数
（`compare_uniform_probe_scaled` 等・`internal-diagnostics` 限定）の実装は
同提案が明記するとおり別イシュー扱いとし、本イシューでは実装しない
（`.claude/rules/out-of-scope-tracking.md`）。

### 10.4 reopen: opt カーネル `assert_parity` 厳密ゼロ fail 判定の不整合を解消（2026-09-02）

PR #1115 マージ後の GB10 実機再検証（2026-09-01）で `wmma_tf32_opt_kernel_matches_reference_across_shapes`・
`wmma_tf32_opt_kernel_k4096_stress`・`wmma_tf32_staged_kernel_exceeds_opt_kernel_tflops_at_4096`
の 3 テストが fail し、#1106 が reopen された。本節はその原因切り分け・
対応（案 A）を記録する。

#### 10.4.1 原因切り分け

`wmma_tf32_opt_kernel_matches_reference_across_shapes`・
`wmma_tf32_opt_kernel_k4096_stress` は `assert_wmma_tf32_opt_kernel_parity`
（内部で `fandhe_ai_backend_cpu::assert_parity`。**厳密ゼロ fail 判定**）を
使っていたが、`assert_parity` は最初に fail したケースで panic するため、
9 ケース中 2 ケース（64×64×64・512×512×4096）しか実機で実行されておらず、
残り 7 ケースは未計測のままだった。

reopen コメントの実測値（64x64x64 seed=3000: fail 699/4096・
mean_abs=5.676e-4／512x512x4096 seed=0xC0FFEE: fail 43019/262144・
mean_abs=4.463e-3）は `ParityBaseline::BASELINES` の既存記録値（同一形状・
同一シード）と**完全一致**した。さらに `wmma_tf32_opt_kernel_parity_does_not_regress`
（baseline 非後退方式・同一カーネル・同一形状・同一シード）は同じ実行で
pass していた。これは「厳密ゼロ fail 判定側だけが red、baseline 非後退
判定側は green」という状態であり、opt カーネル自体に数値バグがあるとは
考えにくいことを示す。

診断のため、同じ 9 ケースを `assert_parity` の代わりに
`fandhe_ai_backend_cpu::compare`（panic しない集計 API）で走らせる一時
テスト（`wmma_tf32_opt_kernel_parity_diagnostic_dump_issue_1106`。PR で
追加後、修正確定に伴い削除済み）を用意し、GB10（sm_121）実機・GPU アイドル
（`nvidia-smi --query-gpu=utilization.gpu` 0%）を確認したうえで 2 回実行
（2026-09-01 初回・2026-09-02 GPU アイドル再確認のうえ再計測）し、全値の
完全一致を確認した:

| shape | seed | fail/total | max_abs_diff | max_rel_err | mean_abs_diff |
|---|---|---|---|---|---|
| 64×64×64 | 0xBB8 (3000) | 699/4096 | 2.561e-3 | 7.772e-1 | 5.676e-4 |
| 128×128×128 | 0xBB9 (3001) | 2638/16384 | 4.413e-3 | 1.314e0 | 7.775e-4 |
| 512×512×512 | 0xBBA (3002) | 42799/262144 | 9.179e-3 | 1.923e0 | 1.568e-3 |
| 63×65×33 | 0xBBB (3003) | 698/4095 | 1.917e-3 | 7.411e-1 | 3.943e-4 |
| 65×63×17 | 0xBBC (3004) | 635/4095 | 1.436e-3 | 2.376e-1 | 2.777e-4 |
| 64×96×256 | 0xBBD (3005) | 967/6144 | 5.184e-3 | 1.169e0 | 1.117e-3 |
| 1×1×1 | 0xBBE (3006) | 0/1 | 1.419e-4 | 2.340e-4 | 1.419e-4 |
| 512×512×4096 | 0xC0FFEE | 43019/262144 | 2.408e-2 | 1.998e0 | 4.463e-3 |
| 4096×4096×4096 | 0xBEEF | 2725617/16777216 | 3.116e-2 | 1.997e0 | 4.453e-3 |

9 ケース中 8 ケースが非ゼロ fail_count を持ち、ゼロ fail が実際に成立
するのは 1×1×1（sub-K-tile。K 方向蓄積が発生せず TF32 丸め誤差が蓄積
しない）のみだった。この非ゼロ fail 率は
`docs/perf/cuda-tensor-core-tolerance-opt-remeasurement.md` §5〜§7 が
記録する opt/basic bit-identical・sm_86/GB10 世代間差分なしの既知の恒常
特性と整合しており、opt カーネルの数値バグではなく、**TF32 丸めの既知の
誤差特性を持つ形状に対して厳密ゼロ fail 判定（`assert_parity`）を適用して
いたテスト設計の不整合**と結論した。

`wmma_tf32_staged_kernel_exceeds_opt_kernel_tflops_at_4096`（性能比較）は
上記診断と同時に GPU アイドル状態で再実行し pass を確認した（初回計測は
他プロセスの GPU 使用〈84%〉と重なった疑いがある計測だったための再計測。
別イシュー化は不要と判断）。

#### 10.4.2 対応（案 A。ユーザー承認 2026-09-02）

tolerance 定数（`RELATIVE_TOLERANCE`/`ABSOLUTE_RESCUE_THRESHOLD`）は変更
せず、**形状ごとに実測特性へ合った判定方式へ再割り当てる**方針（案 A）を
採用した:

- `wmma_tf32_opt_kernel_matches_reference_across_shapes`: `cases` を
  1×1×1（seed=3000）のみへ縮小（厳密ゼロ fail が実際に成立する唯一の
  形状）
- `wmma_tf32_opt_kernel_k4096_stress`: 両ケースとも baseline 側へ移管済み
  のため削除
- `ParityBaseline::BASELINES` へ上表の非ゼロ fail 6 形状
  （128×128×128 seed=0xBB9・512×512×512 seed=0xBBA・63×65×33 seed=0xBBB・
  65×63×17 seed=0xBBC・64×96×256 seed=0xBBD・4096×4096×4096 seed=0xBEEF）
  を実測値付きで追加し、`wmma_tf32_opt_kernel_parity_does_not_regress`
  （既存の baseline 非後退方式。`ParityPath::WmmaTf32Opt` でフィルタして
  全件走査するためコード変更不要で対象拡大した）が検査する。ceiling
  （`baseline_mean_abs_diff_ceiling`）は §4「表記丸め対応」と同じ規約
  （表示 4 桁の最終桁 +1）で算出した
- 64×64×64（seed=3000）・512×512×4096（seed=0xC0FFEE）は既存の
  `ParityBaseline` 行がそのまま対応するため変更なし

これにより本非後退ゲートが 9 形状中 8 形状（1×1×1 を除く全て）をカバー
する。診断専用テスト（`wmma_tf32_opt_kernel_parity_diagnostic_dump_issue_1106`）
は修正確定に伴い削除した。

出典: イシュー #1106 reopen コメント（2026-09-01）・対応 PR のコミット
履歴。実測環境は §10.1 と同一（DGX Spark GB10・sm_121・CUDA 13.0）。

### 10.5 GB10 全件洗い出し（mma/wmma/tensor_core_real_device 全体。2026-09-02）

`cargo test -p fandhe-ai-backend-cuda --release --no-fail-fast -- --ignored
--test-threads=1`（直列・GPU アイドル）で `#[ignore]` テスト全体を実行し、
§10.4 の opt カーネル修正後に残る失敗を洗い出した結果、17 件 FAILED・
106 件 pass だった。内訳を数値系（A 群）と性能比較系（B 群）に分け、
A 群をさらに証跡の性質で 3 グループへ分類して対応した。

#### 10.5.1 グループ分け

- **グループ 1（ルーティング検証・既存記録値と一致）**: `gemm_wmma_tf32_opt.rs::
  wmma_tf32_routed_path_matches_reference_across_shapes`・
  `wmma_tf32_routed_path_k4096_stress`。先頭ケースの実測値（64×64×64:
  699/4096・512×512×4096: 43019/262144）が `ParityBaseline::BASELINES`
  の `WmmaTf32Opt` 既存行と完全一致。ただしこれらは公開 API
  `run_wmma_tf32`（3 段選択そのまま）経由の「実効経路の parity」検査で
  あり、opt カーネル単独強制検査とはテストの意図が異なる（ルーティング
  正しさの検証を兼ねる）ため、単純な重複として削除せず、残り 7 ケースの
  実測を診断テスト経由で別途確認したうえで判断する
- **グループ 2（TF32/f16 丸めの既知特性・追加実測が必要）**:
  `gemm_wmma_tf32.rs`（基本版。32×32×32・256×256×4096 は既存
  `WmmaTf32` baseline と一致、残り 7 ケース未計測）・
  `gemm_wmma_tf32_staged.rs`（staged。64×64×64: 633/4096 は新規値、
  残り 7 ケース未計測）・`cpu_cuda_wmma_parity.rs::wmma_f16_k4096_stress`
  （256×256×4096 seed=8888: 99/65536・単一ケースにつき実測完了）・
  `gemm_wmma_f16_opt.rs::wmma_f16_opt_k4096_stress`（256×256×4096
  seed=8889: 81/65536・単一ケースにつき実測完了）
- **グループ 3（原因未確定・本ラウンド対象外）**: `gemm_mma_tf32.rs`・
  `mma_tf32_vs_wmma_tf32_staged.rs`。いずれもファイル冒頭コメントが
  明記するとおり、残存 FAIL が TF32 丸め誤差由来か機能欠陥由来かが
  **未確定**（`docs/perf/cuda-gemm-mma-tf32-ab.md` §8.4 に分析記録）。
  `mma_tf32_vs_wmma_tf32_staged.rs` はさらに GPU-GPU 相互比較（CPU 参照
  実装ではなく `wmma_tf32_staged` を期待値とする）であり、両経路が
  TF32 丸めの既知特性で一致しない場合は #995 が確立した「basic/opt/staged
  数値完全一致」という bit-identical 性の想定と矛盾する。原因不明のまま
  baseline 化すると未解決のバグを恒久的な受け入れ上限へ変えてしまう
  （#1106 の provenance 問題そのものの再演）ため、本ラウンドでは一切
  変更しない
- **既存の「本体維持＋非後退監視併設」パターン**: `cpu_cuda_mma_parity.rs::
  mma_f16_k4096_stress`・`tensor_core_real_device.rs::tensor_core_parity_record`
  （tf32 部分）・`gemm_tf32_optin.rs::gemm_tf32_optin_on_matches_cpu_across_shapes`
  の 3 件は PR #1115 の codex-review P1 指摘により「元の `assert_parity`
  受け入れ条件は維持し、非後退監視は別テストとして併設する」設計が
  既に確定している（§10.2 項目 2）。これは「案 A」（`assert_parity` の
  対象形状を縮小・削除する）とは異なる、より保守的な既存の確定方針
  であるため、本ラウンドでは変更しない（変更する場合は当該 codex-review
  指摘を明示的に覆す追加のユーザー承認が要る）

#### 10.5.2 本ラウンドで実施した変更

- `ParityBaseline::ParityPath` へ `WmmaF16`・`WmmaF16Opt` を追加し、
  グループ 2 の単一ケース実測 2 件（`wmma_f16` 256×256×4096 seed=8888:
  fail 99/65536・`wmma_f16_opt` 256×256×4096 seed=8889: fail 81/65536。
  ceiling は §4 と同じ規約で算出）を `BASELINES` へ記録した
- `cpu_cuda_wmma_parity.rs::wmma_f16_k4096_stress_non_regression`・
  `gemm_wmma_f16_opt.rs::wmma_f16_opt_k4096_stress_non_regression` を、
  既存の「本体維持＋非後退監視併設」パターン（`mma_f16_k4096_stress_non_regression`
  と同型）で追加した。元の `wmma_f16_k4096_stress`／
  `wmma_f16_opt_k4096_stress`（`assert_parity`）は変更していない
  （引き続き red のまま。REQ-2 違反が解消されていないことを正しく表す）
- `parity_nonregression.rs::parity_baselines_do_not_regress` の
  `match baseline.path` を `WmmaF16`／`WmmaF16Opt` 追加に伴い更新した
  （これらの行は各記録元ファイルの `_non_regression` テストが直接検査
  するため、ここでは `WmmaTf32`／`WmmaTf32Opt` と同じ理由で重複検査を
  避けて skip する）
- グループ 1・グループ 2 の残りケース（`gemm_wmma_tf32.rs`・
  `gemm_wmma_tf32_opt.rs`〈routed〉・`gemm_wmma_tf32_staged.rs` の
  未計測 7 ケースずつ）を可視化する診断専用テスト（`#[ignore]`。panic
  しない `compare()` ベース。修正確定後に削除する一時コード）を追加した:
  `wmma_tf32_parity_diagnostic_dump_issue_1106`・
  `wmma_tf32_routed_path_parity_diagnostic_dump_issue_1106`・
  `wmma_tf32_staged_parity_diagnostic_dump_issue_1106`

#### 10.5.3 未完了（次ラウンドへ引き継ぎ）

- グループ 1・グループ 2 の残り約 21 ケース（`gemm_wmma_tf32.rs` 7 件・
  `gemm_wmma_tf32_opt.rs`〈routed〉7 件・`gemm_wmma_tf32_staged.rs` 7 件）
  の実測が完了次第、§10.4 と同型の「案 A」（`assert_parity` の対象を
  ゼロ fail 成立形状のみへ縮小し、非ゼロ fail 形状は
  `ParityBaseline::BASELINES` へ実測値付きで追加する）を適用する
- グループ 3（`gemm_mma_tf32.rs`・`mma_tf32_vs_wmma_tf32_staged.rs`）は
  原因切り分け自体が別イシューのスコープ（ユーザーと相談のうえ起票の
  要否を判断する。`.claude/rules/out-of-scope-tracking.md`）
- 「本体維持＋非後退監視併設」の既存 3 件（`mma_f16_k4096_stress`・
  `tensor_core_parity_record` tf32・`gemm_tf32_optin_on_matches_cpu_across_shapes`）
  を「案 A」相当（本体を green にする）へ転換するかはユーザー判断（PR
  #1115 の codex-review P1 指摘を明示的に上書きする追加承認が要る）
- B 群（性能比較 2 件）は §10.6 参照

### 10.6 B 群（性能比較テスト）所見

- `gemm.rs::wmma_tf32_staged_kernel_exceeds_opt_kernel_tflops_at_4096`:
  スイート内実行で staged 3.070 vs opt 3.186 TFLOPS（3.6% 差）で fail。
  単独実行では 2 回とも pass。ウォールクロック計測（`Instant::now()`）
  かつ直列実行でも同一プロセス内で他テストの CUDA コンテキストが
  残る・クロック/温度状態が引き継がれる等の影響を受けやすく、3.6% は
  計測ノイズが厳密な `>` 比較を跨いだ結果である可能性が高い。**扱い案:
  現状維持**（既存の受け入れ基準・実装を変更しない。スイート内での
  不安定性は既知の計測手法の限界として記録するに留める）
- `dispatch_boundary.rs::large_shape_mma_pipeline_vs_wmma_tflops_record`:
  `wmma_f16_opt` dim2048 が 5.401 TFLOPS（`mma` は 50.126 TFLOPS）と
  約 10 倍の外れ値。3.6% 差の計測ノイズとは性質が異なり、単なる
  スイート内不安定性として片付けられない規模。**扱い案: 別イシュー化を
  推奨**（`.claude/rules/out-of-scope-tracking.md` に従い、ユーザー確認の
  うえ起票する）


### 10.7 グループ1・2 の案 A 確定・実装（GB10 実機実測 2026-09-02）

§10.5.3 で未計測だった約 21 ケース（`gemm_wmma_tf32.rs` 8 件・
`gemm_wmma_tf32_opt.rs`〈routed〉9 件・`gemm_wmma_tf32_staged.rs` 9 件）
を、`wmma_tf32_parity_diagnostic_dump_issue_1106`・
`wmma_tf32_routed_path_parity_diagnostic_dump_issue_1106`・
`wmma_tf32_staged_parity_diagnostic_dump_issue_1106`（診断専用テスト。
GPU アイドル 1% で実行、いずれも ok）で GB10 実機実測した。結果:

- **basic（`gemm_wmma_tf32.rs`）**: 8 ケース中 7 ケースが非ゼロ fail
  （32×32×32 seed=2000: 154/1024・64×64×64 seed=2001: 687/4096・
  128×128×128 seed=2002: 2559/16384・512×512×512 seed=2003: 42550/262144・
  64×96×128 seed=2004: 1027/6144・17×23×19 seed=2006: 52/391・
  33×31×65 seed=2007: 171/1023）。1×1×1（seed=2005）のみ 0/1。K4096
  ストレス 2 ケースも非ゼロ（256×256×4096 seed=8888: 10647/65536・
  512×512×4096 seed=0xFACADE: 42688/262144）
- **routed（`gemm_wmma_tf32_opt.rs`）**: 9 ケース中 8 ケースが非ゼロ fail
  で、**全 8 ケースが既存の `ParityPath::WmmaTf32Opt` baseline 行と
  fail_count／mean_abs_diff 完全一致**（実効経路〈staged／opt〉の決定を
  含め再現的）。1×1×1（seed=0xBBE）のみ 0/1
- **staged（`gemm_wmma_tf32_staged.rs`）**: 9 ケース中 8 ケースが非ゼロ
  fail（64×64×64 seed=0xFA0: 633/4096・128×128×128 seed=0xFA1: 2631/16384・
  512×512×512 seed=0xFA2: 42782/262144・60×68×36 seed=0xFA3: 691/4080・
  68×60×20 seed=0xFA4: 620/4080・64×96×256 seed=0xFA5: 1008/6144）。
  K4096 ストレス 2 ケースも非ゼロ（512×512×4096 seed=0xC0FFEE:
  43019/262144〈既存 `WmmaTf32Staged` 行と一致〉・4096×4096×4096
  seed=0xBEEF: 2725617/16777216〈新規〉）。1×1×1（seed=0xFA6）のみ 0/1

いずれのファイルでも「ゼロ fail が実際に成立するのは 1×1×1（sub-K-tile）
のみ」という §10.4／§10.5 の結論が再確認された。

#### 10.7.1 実装したバグ修正（案 A 適用中に発見）

§10.4 で `gemm.rs::wmma_tf32_opt_kernel_matches_reference_across_shapes`
を 7 ケースから 1×1×1 のみへ縮小した際、`cases: &[(1, 1, 1)]` とした
うえで `3000 + idx as u64`（`idx` は縮小後の配列上のインデックス=0）で
seed を再計算していたため、実際には**未実測の seed=3000**（元は
64×64×64 用のシード）が (1,1,1) 形状に適用されていた（元の 7 ケース
配列で 1×1×1 が位置していた idx=6 に対応する seed=3006 ではなかった）。
本ラウンドでこの座標ずれに気づき、`gemm.rs`・本ラウンドで追加した
3 ファイルすべてで **縮小後の 1×1×1 テストは `idx` 由来の自動算出ではなく、
実測した seed を直接ハードコードする**方式へ修正した（`gemm.rs`:
seed=3006・`gemm_wmma_tf32.rs`: seed=2005・`gemm_wmma_tf32_opt.rs`
routed: seed=3006・`gemm_wmma_tf32_staged.rs`: seed=4006。いずれも実測で
fail_count=0/1 を確認済みの値）。

#### 10.7.2 実装内容

- **`crates/backend-cuda/tests/common/parity_baseline.rs`**: `WmmaTf32`
  へ 7 行（64×64×64・128×128×128・512×512×512・64×96×128・17×23×19・
  33×31×65・512×512×4096 seed=0xFACADE）、`WmmaTf32Staged` へ 7 行
  （64×64×64・128×128×128・512×512×512・60×68×36・68×60×20・64×96×256・
  4096×4096×4096 seed=0xBEEF）を追加した。`WmmaTf32Opt` は全 8 ケースが
  既存行と完全一致したため新規行は追加していない
- **`gemm_wmma_tf32.rs`**: `wmma_tf32_matches_reference_across_shapes` を
  1×1×1（seed=2005）のみへ縮小。`wmma_tf32_k4096_stress_poc_v2_5`
  （両ケースとも非ゼロ fail）は削除。公開 API `run_wmma_tf32` 経由で
  `ParityPath::WmmaTf32` 行を検査する非後退監視テスト
  `wmma_tf32_routed_path_baselines_do_not_regress` を新設した（`mod
  common;` を追加）。診断専用テストは削除
- **`gemm_wmma_tf32_opt.rs`**: `wmma_tf32_routed_path_matches_reference_across_shapes`
  を 1×1×1（seed=3006）のみへ縮小。`wmma_tf32_routed_path_k4096_stress`
  （両ケースとも既存行に対応）は削除。公開 API `run_wmma_tf32` 経由で
  `ParityPath::WmmaTf32Opt` 行を検査する非後退監視テスト
  `wmma_tf32_routed_path_baselines_do_not_regress` を新設した（`mod
  common;` を追加）。診断専用テストは削除
- **`gemm_wmma_tf32_staged.rs`**: `wmma_tf32_staged_matches_reference_across_shapes`
  を 1×1×1（seed=4006）のみへ縮小。`wmma_tf32_staged_k4096_stress`
  （両ケースとも baseline 側で検査可能）は削除。staged 経路は既存の
  `tests/parity_nonregression.rs::parity_baselines_do_not_regress`
  （`check_wmma_tf32_staged_baseline`）が `ParityPath::WmmaTf32Staged`
  を skip せず走査するため、新規の非後退監視テストは不要（コード変更
  なしで対象拡大）。診断専用テストは削除
- tolerance 定数（`RELATIVE_TOLERANCE`/`ABSOLUTE_RESCUE_THRESHOLD`）は
  変更していない（ユーザー承認 2026-09-02）

#### 10.7.3 残る扱い（次の意思決定待ち）

- グループ3（`gemm_mma_tf32.rs`・`mma_tf32_vs_wmma_tf32_staged.rs`）:
  原因未確定のため未着手（§10.5.1 参照）
- 「本体維持＋非後退監視併設」の既存 5 件（`cpu_cuda_mma_parity.rs::
  mma_f16_k4096_stress`・`tensor_core_real_device.rs::
  tensor_core_parity_record`〈tf32〉・`gemm_tf32_optin.rs::
  gemm_tf32_optin_on_matches_cpu_across_shapes`・本ラウンドで同型追加
  した `cpu_cuda_wmma_parity.rs::wmma_f16_k4096_stress`・
  `gemm_wmma_f16_opt.rs::wmma_f16_opt_k4096_stress`）は意図的に red の
  まま。案 A（本体を green にする）へ転換するかはユーザー判断
- B 群（性能比較 2 件）は §10.6 参照
- #1106 の受入条件「GB10 で該当テスト群が全 pass」は、上記未決定事項が
  残る限り本 PR 単独では満たされない（スコープ分割の要否をユーザーへ
  諮る）
