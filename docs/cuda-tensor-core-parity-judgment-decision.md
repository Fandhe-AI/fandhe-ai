# TF32/f16 Tensor Core 経路の parity テスト判定方式の決定記録（イシュー #1106）

イシュー #1106「TF32/f16 mma・wmma・tensor_core_real_device テスト群の GB10
失敗（provenance 未確定 fail-closed 含む）を解消する」の実装過程で判明した
「厳密ゼロ fail 判定がそもそも成立しない形状が存在する」という事実に対する
判定方式の決定を記録する。`docs/cuda-tf32-optin-api-decision.md` 等、本リポの
確立パターン（決定の背景・承認・スコープ限定を単一ドキュメントへ記録し、
`AGENTS.md`／`.claude/rules/` への反映は本ドキュメントを承認記録として参照した
後続 PR で行う）に倣う。

## 1. 背景

### 1.1 症状（イシュー #1102・#1106 の GB10 実測）

`backend-cuda` の TF32 Tensor Core 経路（`wmma_tf32`〈基本版〉・
`wmma_tf32_opt`〈共有メモリ・タイル最適化版〉・`wmma_tf32_staged`〈cp.async
多段パイプライン版〉。いずれも `crates/backend-cuda/src/gemm.rs`／
`kernels_wmma_opt.rs`）の parity テストは、当初 `fandhe_ai_backend_cpu::
assert_parity`（REQ-2 統一複合判定「相対誤差 1e-3 未満 または 絶対誤差
1e-5 未満」の**全要素**に対する厳密判定。1 要素でも不合格なら panic）で
CPU 参照実装（`matmul_reference_fma`）と照合していた。

イシュー #1106 の GB10（sm_121・DGX Spark GB10 実機。実ホスト名は
`docs/real-hardware-verification-env.local.md` 参照で本ドキュメントには
書かない）再検証（2026-09-01・2026-09-02）で、この厳密ゼロ fail 判定を
実際に GB10 実機で通しにいったところ、`wmma_tf32_opt`・`wmma_tf32`（基本
版）・`wmma_tf32_staged` の各形状網羅テストのうち多くの形状（1×1×1 を除く
ほぼ全形状）で非ゼロ `fail_count` を検出した。診断専用テスト（`assert_parity`
の代わりに panic しない `fandhe_ai_backend_cpu::compare` を使い、最初の
fail で打ち切らず全ケースを走らせて可視化する一時テスト。修正確定後に削除
済み）による実測表の要約は次のとおり（`fail/total` は複合判定不合格要素数
／全要素数。全形状で 2 回実行し値が完全一致することを確認済み）:

| 経路 | 形状 | fail/total | max_rel_err |
|---|---|---|---|
| `wmma_tf32_opt`（GB10 全件洗い出し。9 ケース） | 64×64×64 | 699/4096 | 7.772e-1 |
| | 128×128×128 | 2638/16384 | 1.314e0 |
| | 512×512×512 | 42799/262144 | 1.923e0 |
| | 63×65×33 | 698/4095 | 7.411e-1 |
| | 65×63×17 | 635/4095 | 2.376e-1 |
| | 64×96×256 | 967/6144 | 1.169e0 |
| | 1×1×1 | 0/1 | 2.340e-4 |
| | 512×512×4096 | 43019/262144 | 1.998e0 |
| | 4096×4096×4096 | 2725617/16777216 | 1.997e0 |
| `wmma_tf32`（基本版。8 ケース） | 32×32×32 | 154/1024 | 3.239e-2 |
| | 64×64×64 | 687/4096 | 1.101e0 |
| | 128×128×128 | 2559/16384 | 8.871e-1 |
| | 512×512×512 | 42550/262144 | 1.990e0 |
| | 64×96×128 | 1027/6144 | 1.778e0 |
| | 1×1×1 | 0/1 | 3.032e-5 |
| | 17×23×19 | 52/391 | 1.814e-2 |
| | 33×31×65 | 171/1023 | 3.964e-1 |
| `wmma_tf32_staged`（7 ケース） | 64×64×64 | 633/4096 | 1.515e0 |
| | 128×128×128 | 2631/16384 | 8.880e-1 |
| | 512×512×512 | 42782/262144 | 1.979e0 |
| | 60×68×36 | 691/4080 | 4.449e-1 |
| | 68×60×20 | 620/4080 | 7.802e-1 |
| | 64×96×256 | 1008/6144 | 1.034e0 |
| | 1×1×1 | 0/1 | 1.874e-4 |

**全経路・全形状に共通する傾向**: ゼロ fail が実際に成立するのは 1×1×1
（K タイル未満の sub-K-tile 形状。K 方向の蓄積誤差がそもそも発生しない）
のみであり、それ以外の全形状で 17〜100% 近い非ゼロ fail_count を持つ。

### 1.2 「カーネルのバグではない」ことの確認

上記の非ゼロ fail は、以下の既存実測データと整合しており、opt/basic/staged
各カーネルの実装差やハードウェア世代差に由来する回帰ではなく、TF32 丸め
（f32 仮数部 23bit → 10bit）自体が持つ既知の恒常特性であると判断できる:

- **opt/basic/staged 間の bit-identical 性**: `docs/perf/
  cuda-tensor-core-tolerance-opt-remeasurement.md` §6 は、`opt` 強制構成・
  `basic` 強制構成の生ログを突合した結果、全 15 形状・全シードで
  `fail_count`・`max_rel_err`・`max_fail_abs_diff` が完全一致（`diff` 差分
  0 行）したことを記録している。GB10 では TF32 WMMA カーネルのタイル戦略
  （共有メモリ・swizzle の有無）が誤差分布に一切影響しない
- **sm_86 世代との差分なし**: 同ドキュメント §7 は、`basic` 強制構成（sm_86
  実測時点のカーネルソースと同一）で対比した結果、fail 率・
  `max_fail_abs_diff` がいずれの形状も表示桁でほぼ完全一致し、系統的な世代
  差が見られなかったことを記録している
- **スケール依存の確認**: `docs/perf/cuda-tensor-core-tolerance-
  gb10-scale-sweep.md`（イシュー #995）は `s ∈ {0.1, 1, 10, 100}` の入力
  スケールスイープでも同様の傾向を確認している
- **本イシュー #1106 内での再現性**: 上記 1.1 節の実測表は、GB10 実機で
  GPU アイドル（`nvidia-smi --query-gpu=utilization.gpu` 0〜1%）を確認した
  うえで 2 回実行し、`fail_count`・`max_abs_diff`・`max_rel_err`・
  `mean_abs_diff` の全値が完全一致することを確認済み（2026-09-01 初回・
  2026-09-02 GPU アイドル再確認のうえ再計測）

以上から、この非ゼロ fail は「TF32 丸めの既知の恒常特性を持つ形状に、
成立しえない厳密ゼロ fail 判定を適用していた」というテスト設計上の
不整合であり、カーネル実装側の数値バグではないと判断する。

## 2. 決定

**2026-09-02 ユーザー承認**（イシュー #1106 のスコープ分割コメントが一次
記録: <https://github.com/Fandhe-AI/fandhe-ai/issues/1106#issuecomment-5497016249>）。

TF32/f16 Tensor Core 経路の parity テストについて、以下を正式な受け入れ
判定の方式として採用する（「案 A」）:

1. **厳密ゼロ fail 判定**（`fandhe_ai_backend_cpu::assert_parity`。REQ-2
   統一複合判定を全要素に適用し、1 要素でも不合格なら fail）は、**GB10
   実機実測でゼロ fail の成立が確認された形状にのみ適用する**（本イシュー
   の実測範囲では 1×1×1〈sub-K-tile〉のみ）
2. **実測でゼロ fail が成立しないと判明した形状**は、**実測 baseline
   非後退方式**（`crates/backend-cuda/tests/common/parity_baseline.rs::
   ParityBaseline`／`assert_no_parity_regression`。`fail_count`・
   `mean_abs_diff` の ceiling が既存の実測値を上回っていないかを機械検査
   する）を正式な受け入れ判定とする
3. baseline の新規追加・更新は必ず **GB10 実機実測値を伴う**（推定値の
   記入は禁止。`docs/perf/cuda-parity-baseline.md` §6「ベースライン更新
   規約」に従う）

## 3. 回帰検出契約（非後退方式が回帰を見逃さない根拠）

実測 baseline 非後退方式（項目 2）へ移行しても REQ-2 の回帰検出能力が
弱体化しないことを、以下の契約で担保する:

- **fail_count の増加で必ず fail する**: `assert_no_parity_regression`
  （`crates/backend-cuda/tests/common/parity_baseline.rs`）は
  `report.fail_count > baseline.baseline_fail_count` を検査し、記録済み
  ベースラインより 1 要素でも不合格要素数が増えれば panic する。カーネル
  実装の変更（誤ったリファクタリング・タイル境界の破損等）で fail_count
  が増加する回帰は確実に検出される
- **mean_abs_diff ceiling の超過でも必ず fail する**: 同様に
  `report.mean_abs_diff > baseline.baseline_mean_abs_diff_ceiling` を
  検査する。fail_count が変わらなくても誤差の平均値が悪化する回帰
  （境界は同じだが内部の丸め精度が劣化する等）を検出する
- **total（形状・要素数）の不一致でも必ず fail する**: `report.total !=
  baseline.total` を検査し、記録時と異なる形状で計測してしまう取り違え
  を防ぐ
- **baseline の更新自体に承認が要る**: `docs/perf/cuda-parity-baseline.md`
  §6「ベースライン更新規約」により、fail_count・mean_abs_diff の上限を
  緩める「上方更新」は**ユーザー承認必須**（`.claude/rules/security.md`
  A08 と同列のガードレール）。カーネル改修による改善（下方更新）は実機
  実測値とセットでのみ許容され、推定値の記入は禁止（未計測形状・シードの
  行追加も実機実測とセットでのみ行う。例外は「実機未到達」を明示する
  `baseline_provenance_unconfirmed: true` の fail-closed プレースホルダの
  みで、実測値を主張しない）
- **fail-closed 契約**: `baseline_provenance_unconfirmed: true` の行は
  `assert_no_parity_regression` が判定を試みず必ず panic する（黙って
  skip しない。「実機テストは正常終了と shape 一致だけで通過する」という
  非後退ゲートが機能していないのに green に見える状態を防ぐ）

以上により、非後退方式は「既知の不合格分布を許容する」という点で厳密ゼロ
fail 判定より緩いが、「その既知の分布から悪化していないか」を
fail-closed に検査する点で回帰検出契約としては維持される。

## 4. スコープ限定

- **tolerance 定数の変更は本決定の対象外**: `RELATIVE_TOLERANCE`・
  `ABSOLUTE_RESCUE_THRESHOLD`（`crates/backend-cpu/src/parity.rs`）自体の
  変更は、本決定が扱う「判定方式（テスト個別の合否基準）の使い分け」とは
  別軸であり、引き続きユーザー承認必須（`.claude/rules/coding-rust.md`
  「バックエンド間数値一致テストの許容誤差を単独で緩和しない」）
- **PR #1115（イシュー #1106）で確定した既存 5 テストの設計は現状維持**:
  `cpu_cuda_mma_parity.rs::mma_f16_k4096_stress`・
  `tensor_core_real_device.rs::tensor_core_parity_record`（tf32 部分）・
  `gemm_tf32_optin.rs::gemm_tf32_optin_on_matches_cpu_across_shapes`・
  および同型で追加した `cpu_cuda_wmma_parity.rs::wmma_f16_k4096_stress`・
  `gemm_wmma_f16_opt.rs::wmma_f16_opt_k4096_stress` の計 5 テストは、
  「本体 `assert_parity`（green 必須。REQ-2 受け入れ条件そのもの）は
  維持したまま、既知の不合格分布に対する非後退監視を別テストとして併設
  する」という PR #1115 の codex-review P1 指摘を受けた確定方針のまま
  変更しない。本決定（案 A: 対象形状を縮小し baseline 非後退方式へ移行
  する）はこの 5 テストの方式を覆すものではなく、**別の対応方式として
  並立する**（どちらを適用するかはテストの記録元・ヒストリーに応じて
  個別に判断してよく、両方式の間で優劣を定めない）

## 5. 反映方針

本ドキュメントは決定内容の一次記録であり、`AGENTS.md`（codex-review が
参照する権威ある規約）・`.claude/rules/coding-rust.md`・
`.github/codex/prompts/review.md` への反映（P1 分類「テストの弱体化」の
例外条件としての明記）は**別途、本ドキュメントをベース側から確認可能な
承認記録として参照する後続 PR で行う**（codex-review は制御ファイルを
base コミットから読むため、当の PR 自身の変更はその PR 自身のレビューには
反映されない特性がある。`.claude/rules/ci.md`「codex-review 判定との
非対称性」節と同種の理由により、本 PR〈決定記録〉と後続 PR〈規約反映〉を
分離する）。

## 6. 関連

- イシュー #1106（本決定の起点）・#1102（親トラッキング）
- `docs/perf/cuda-parity-baseline.md`（§3 ベースライン表・§6 ベースライン
  更新規約・§10 イシュー #1106 の実測記録本体）
- `docs/perf/cuda-tensor-core-tolerance-opt-remeasurement.md`（opt/basic
  bit-identical 性・sm_86 との世代差なしの実測根拠）
- `docs/perf/cuda-tensor-core-tolerance-gb10-scale-sweep.md`（イシュー
  #995。スケールスイープでの世代差記録）
- `crates/backend-cuda/tests/common/parity_baseline.rs`（`ParityBaseline`
  fixture・`assert_no_parity_regression` 検査ユーティリティ本体）
- `docs/cuda-tf32-optin-api-decision.md`（同型の決定記録ドキュメントの
  先例。承認ステータス節の書式を踏襲）
