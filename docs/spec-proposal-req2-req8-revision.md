# REQ-2 複合判定改定・REQ-8 表更新の REQ 改定提案（草案）

> **本文書は spec リポジトリ（Fandhe-AI/rust-ai-library-spec）への REQ-2・REQ-8 改定提案の草案**であり、
> 正本ではない。採否・spec 側への起票判断は人間（ユーザー）の承認事項（イシュー #580）。本リポジトリの
> `docs/spec/`（正本 submodule）は本提案作成にあたって一切編集していない。本リポジトリでは
> `RELATIVE_TOLERANCE`・`ABSOLUTE_RESCUE_THRESHOLD`・`threshold.rs::floor_spec` のいずれの値も本提案の
> 作成にあたって変更していない（変更はユーザー承認必須。`.claude/rules/coding-rust.md`・
> `.claude/rules/security.md` A08）。**

出典: イシュー #580「F-7: REQ-2 改定提案の spec リポジトリへの提案」（GEMM 性能改善ツリー #479 系列）。

## 1. REQ-2 複合判定の改定提案

### 1.1 未解消状態の整理

REQ-2 の統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満。`docs/spec/04-requirements.md:74`）は、
CUDA Tensor Core 経路（TF32 WMMA・f16 mma）で恒常的に不合格であることが実測済みである。

- イシュー #186（TASK-11.1g）は「変更が必要」と結論して 2026-08-06 に close 済みだが、閾値定数
  （`backend_cpu::parity::RELATIVE_TOLERANCE`＝1e-3・`ABSOLUTE_RESCUE_THRESHOLD`＝1e-5）は close 後も
  変更されていない（`docs/perf/cuda-tensor-core-tolerance-evaluation.md` §4「結論」）。
- TF32 経路（`CudaGemm::run_wmma_tf32`、基本版）は最小形状（32×32×32）を含む全 15 形状・全シードで
  複合判定の不合格率が約 15〜16.5% に達する（同ドキュメント §2.1 実測表・同「要旨」）。唯一の例外は
  要素数 1 の縮退形状（1×1×1）。
- f16 WMMA 経路（`CudaWmmaGemm::run_f16`）は K≤256 では不合格率 0% だが、K=512 以降で不合格が発生し、
  K=4096（PoC-v2-5 stress ケース）で不合格率 0.174% に達する（同 §2.2 実測表。K に対して単調増加）。
- `docs/perf/cuda-parity-baseline.md`（イシュー #389 系列。実測記録本体は `docs/backend-cuda-real-device-testing.md`
  §5.3）が定める数値一致 parity の恒常 fail 対象、および `docs/perf/cuda-optimized-remeasurement.md`
  「数値一致（parity）状態の限定条件」節が Phase F 後も再確認した内容は、いずれも上記 TF32／f16 経路の
  不合格が現時点でも解消していないことと整合する。

### 1.2 改定候補 3 案

`docs/perf/cuda-tensor-core-tolerance-evaluation.md` §4「結論（変更要否）」が整理した 3 案を、
出典を明記したうえでそのまま踏襲する（本提案が新たに数値判断を加えたものではない）。

1. **経路別の絶対誤差救済閾値の追加**: TF32 経路と f16 経路とで別々の絶対誤差救済閾値を REQ-2 に追加する。
   候補値は `max_fail_abs_diff`（不合格セル限定の絶対誤差最大値。同ドキュメント §2 の定義）の
   infimum（下限）実測値であり、TF32 は **2.535e-2 を厳密に超える値**、f16 は **4.883e-4 を厳密に超える値**
   が候補となる（同 §3.2）。**infimum そのものの値では救済されない**点に注意する: `compare` の判定式は
   `diff < ABSOLUTE_RESCUE_THRESHOLD`（狭義不等号）であり、`ABSOLUTE_RESCUE_THRESHOLD == max_fail_abs_diff`
   では当の最大値を持つセル自身が未救済のまま残る（同 §2 冒頭の注記）。両者は 1 桁以上開きがあるため、
   経路別の閾値設定が現実的である。
2. **f16 経路の K 依存スケーリングまたは K 実用上限の定義**: f16 経路の不合格率は K の増加に対して単調
   増加する（K=256: 0%、K=512: 0.0021%、K=1024: 0.025%、K=4096: 0.174%。同 §2.2）ため、K 依存の許容誤差
   スケーリング、または本ライブラリが許容する K の実用上限を REQ-2 側で定義し、その範囲内でのみ現行閾値
   を適用する案。
3. **高精度要求時に Tensor Core 経路を選択しないディスパッチ方針の明示**: TF32/f16 Tensor Core 経路
   そのものをディスパッチ規則（REQ-11 系）で高精度要求時に選択しない方針とし、経路ごとの精度トレード
   オフを利用者に明示する案。

### 1.3 最終閾値の確定前に必要な未実施検証

`docs/perf/cuda-tensor-core-tolerance-evaluation.md` が明記する未実施の検証を、閾値の具体値を確定する
前提条件としてそのまま引き継ぐ。**本提案はこれらの検証を実施していない**ため、閾値の具体値は spec 側で
実測を実施したうえで確定する段取りを提案する。

- **入力スケールスイープ（§3.3）**: 上記候補閾値（2.535e-2・4.883e-4）はいずれも
  `Xorshift64Star::fill_vec`／`fill_vec_f16` が生成する `[-1, 1)` 範囲の入力限定の暫定値である。GEMM の
  絶対誤差は入力スケール `s` に対しおおむね `s²` でスケールするため、代表的な複数のスケール
  （例: 0.1・1・10・100 倍）での実測スイープなしに固定の絶対閾値として採用しない。
- **opt 版カーネルでの再実測**: §2.1 の TF32 実測は TASK-11.1c（#62）時点の基本版
  `CudaGemm::run_wmma_tf32` を対象としており、TASK-11.1d（#63）で追加された共有メモリ・タイル最適化版
  （`kernels_wmma_opt::WMMA_TF32_F32_OPT`）は `run_wmma_tf32` が利用可能な場合に優先使用するため、
  現行実装で `wmma_tolerance_probe` を再実行すると異なるカーネルの誤差分布を計測することになる
  （同ドキュメント冒頭「測定条件の失効に関する注記」）。§2.1・§3・§4 の TF32 側数値は基本版限定の
  暫定値であり、opt 版での再実測が必要である。
- **GB10（sm_121）実機での再実測**: §1 の実測環境は compute capability 8.6（Ampere、NVIDIA GeForce
  RTX 3060）であり、GB10（sm_121・Blackwell 系譜）実機での再確認は別イシュー（#64・TASK-11.1e）の
  スコープである。Tensor Core の世代差（mantissa 丸め方式・累算精度）による差異が出る可能性があり、
  本ドキュメントの数値を sm_121 にそのまま適用しないこと（同 §5「制約事項」）。
- **累算誤差項の定量的切り分け（§3.1、参考）**: TF32 経路の不合格率が K に依存しない観察について、
  「入力丸め×条件数が支配項である」という帰属は現時点で未検証の仮説にとどまる（累算誤差項を分離した
  実測は未実施）。閾値改定の定量的根拠には現時点でこの帰属を用いない。

以上より、本提案は**閾値の具体値を確定しない**。改定の要否・改定候補 3 案・未実施検証の一覧を spec 側へ
提示し、最終値は spec 側での実測完了後に確定する段取りを提案する。

### 1.4 本リポジトリでの tolerance 非緩和の再掲

本提案の作成過程で、本リポジトリの `RELATIVE_TOLERANCE`・`ABSOLUTE_RESCUE_THRESHOLD`・既存 parity
テストの許容誤差はいずれも変更していない。バックエンド間数値一致テストの許容誤差の単独緩和は
`.claude/rules/coding-rust.md`「テスト・ベンチ」節・`.claude/rules/security.md`「自己修復ループ固有の
ガードレール」節によりユーザー承認必須であり、ポリシー除外リストのブラインドスポット対象でもある
（REQ-5・`crates/guardrail/tests/fixtures/labeled-changes/README.md` の `test-tolerance-loosening`
除外ルール）。

## 2. REQ-8 表の更新提案

### 2.1 spec 表と本リポ確定値の対照

`docs/spec/04-requirements.md`（2026-08-05 版、REQ-8 §受け入れ基準 1 の表）の下限値は、本リポジトリの
確定記録（`docs/perf/performance-floor-decision.md` §3・§8・§9・§10）と乖離している。出典は
`performance-floor-decision.md` を用いる（`docs/performance-targets.md` への転記整合は別イシュー #579
が OPEN で担当中のため、本提案の出典には使わない）。

| backend_dtype | 段階 | spec 現行値（2026-08-05 版） | 本リポ確定値 | 確定根拠 |
|---|---|---|---|---|
| CPU 対 PyTorch CPU | 初期リリース | 5% | **5%（変更なし）** | PoC-v2-1 実測 5.3%（`performance-floor-decision.md` §3） |
| CPU 対 PyTorch CPU | 最適化後 | 20% | **20%（変更なし）** | NEON intrinsics 実効効率見積もり（同 §3）。Phase F 再計測（実測比率 24.7%）でも同一値を再確認（同 §10） |
| CUDA f32 対 PyTorch CUDA | 初期リリース | 10% | **10%（変更なし）** | PoC-v2-3 実測 10.3%（同 §3） |
| CUDA f32 対 PyTorch CUDA | 最適化後 | 40%（暫定） | **50%** | §9（#393）で 25% に確定後、§10（#577・2026-08-18 承認記録）で Phase F 再計測（実測比率最小値 51.96%、size=4096、Rust/PyTorch とも 5 run 中央値）を根拠に 50% へ再確定 |
| CUDA f16 対 PyTorch f16 | 初期リリース | 下限を設定しない | **下限を設定しない（変更なし）** | tensor core 未使用のスカラー実装同士の比較（実測 1.9%）は指標として無意味（REQ-8 脚注、同 §3） |
| CUDA f16 対 PyTorch f16 | 最適化後 | 40%（暫定） | **35%** | §9（#393）で 10% に確定後、§10（#577）で Phase F 再計測（実測比率最小値 37.47%、size=4096、5 run 中央値）を根拠に 35% へ再確定 |
| Metal f32 対 PyTorch MPS | 初期リリース | 20% | **20%（変更なし）** | PoC-v2-4 実測 23.2%（同 §3） |
| Metal f32 対 PyTorch MPS | 最適化後 | 30% | **10%** | §10（#577）で計測境界を `docs/performance-targets.md` §4 準拠系列（`dispatch_tiled_prepared` prepared 入口）へ揃えた結果、実測比率最小値 13.01%（size=4096）を根拠に 30%→10% へ引き下げ |
| Metal f16 対 PyTorch MPS f16 | 初期リリース | 未設定 | **15%** | §8（#386）で PoC 後の実測（Metal/PyTorch 比 18.6%、size=4096）を根拠に確定 |
| Metal f16 対 PyTorch MPS f16 | 最適化後 | 未設定 | **15%（新設）** | §10（#577）で Phase F 実測（実測比率最小値 18.78%、size=4096）を根拠に新設 |

丸め規則（実測比率 10% 以上は 5% 刻み切り下げ、10% 未満は 1% 刻み切り下げ、条件付き追加ステップなしの
単調規則）は変更しない（`docs/spec/04-requirements.md:172`）。

### 2.2 CUDA 行の限定条件の継続記載

CUDA f32／f16 の最適化後下限（50%／35%）の候補算出経路（`wmma_tf32`・`mma_f16`）は、1 節で整理した
REQ-2 の parity 恒常 fail 対象と一致する（`docs/perf/performance-floor-decision.md` §10「CUDA 限定条件の
継続」節）。REQ-2 改定と REQ-8 表更新は次の関係で相互依存しているため、表更新後もこの限定条件を継続
記載することを提案する。

- §9（#393）・§10（#577）のいずれの承認記録も「#186（REQ-2 閾値改定）の解決後に本下限値を再確認する」
  ことを限定条件として明記している。#186 は close 済みだが閾値定数は未変更のため、この限定条件は
  **解消しておらず継続する**（同 §10「CUDA 限定条件の継続」節 3 点目）。
- REQ-2 改定（1 節）で parity が green 化した場合、CUDA f32／f16 の候補算出経路の実測値が変わりうる
  ため、REQ-8 側の下限値は REQ-2 改定解決後に再実測・再確認が必要になる。
- CudaF32 の 50% は `wmma_tf32_staged` 経路の実測値に基づくが、staged 経路は正本
  `docs/perf/cuda-parity-baseline.md` にベースライン未計測（`baseline_provenance_unconfirmed`）であり
  parity 非後退が判定不能という追加の限定条件がある（`performance-floor-decision.md` §10「CUDA 限定条件
  の継続」節 4 点目、2026-08-19 ユーザー承認済み）。この限定条件も REQ-8 表更新後に継続記載することを
  提案する。

### 2.3 Transformer 複合ワークロード行

Transformer 複合ワークロード行は「下限を設定しない」を維持することを提案する。`docs/kernel-fusion.md`
§7（#591 判断）は、Phase G（融合 IR 拡張）完了後も「matmul・softmax を含む複合 WL では融合効果を前提と
した性能目標を設定しない」という REQ-12 受け入れ基準が引き続き有効であると判断しており（採用: 分岐 (i)、
`docs/kernel-fusion.md` 改定のみで完結）、`docs/performance-targets.md` の Transformer 行も初期リリース・
最適化後のいずれも「下限を設定しない」のまま変更されていないことと整合する。

## 3. G-1・G-5 との一本化調整

受け入れ基準が求める、他の spec 提案文書との一本化調整を以下に記録する。

### 3.1 G-1（`docs/spec-proposal-fp8-int8-quant-gemm.md`）との一本化

G-1（FP8/INT8 新 REQ 追加提案）と本提案は、スコープが重複しない: G-1 は新 REQ の追加提案であり
REQ-2／REQ-8 の既存値の改定は扱わない（同文書 §8「相互参照・提案の一本化」）。一方本提案は既存
REQ-2／REQ-8 の改定を扱い、新 REQ の追加は扱わない。

G-1 §8 は「spec 側へは、無調整で複数件を別々に起票せず、F-7（#580）の提案と調整のうえ 1 回にまとめて
起票する運用を提案として記載する」としている。本提案はこれを引き継ぎ、spec リポジトリへの起票は
**親イシュー 1 件に本提案（REQ-2／REQ-8 改定）と G-1（新 REQ 追加提案）の 2 本の提案文書を添付する形**
での一本化を提案する。両提案のスコープ境界（既存 REQ の改定 vs 新 REQ の追加）に重複がないことを
起票時に明記する。

G-1 §8 の F-7 行（「`dependsOn` #577 のため未完」）は、#577 が CLOSED 済みかつ本提案文書が作成された
時点で解消しているため、`docs/spec-proposal-fp8-int8-quant-gemm.md` §8 の当該行を本文書への参照へ
更新する（本コミットで実施）。

### 3.2 G-5（#591）との一本化

G-5（#591）は上記 2.3 節のとおり分岐 (i)（本リポ `docs/kernel-fusion.md` の改定のみで完結）を採用済み
であり、REQ-12 改定提案は不要と判断されている。よって spec 側起票に REQ-12 改定は**含めない**。

`docs/kernel-fusion.md` §6 に残置されている「REQ-12 の文言が v1（`burn-wgpu` `fusion` feature）前提の
ままである」という既知の課題は、#591 の判断でも「従来どおり残置し、本判断で新たに拡大しない」とされて
おり、本提案でも参考情報としての言及に留め、REQ-12 改定の対象には含めない。

## 4. 結論・ユーザーへの提案事項

- 本文書は spec リポジトリへの改定提案の**草案**である。spec 側での起票判断（イシュー #580 の受け入れ
  基準が求める起票判断）は人間（ユーザー）に委ねる。
- 承認が得られるまで、本リポジトリでは閾値定数（`RELATIVE_TOLERANCE`・`ABSOLUTE_RESCUE_THRESHOLD`）・
  REQ-8 下限値（`crates/bench-harness/src/threshold.rs::floor_spec`）に関わるコード変更を行わない。
- REQ-2 改定の最終閾値は、1.3 節が示す未実施検証（入力スケールスイープ・opt 版再実測・GB10 実機再実測）
  の完了後に spec 側で確定する段取りを提案する。本文書はその具体値を断定しない。
- REQ-8 表更新は 2.1 節の対照表のとおり、本リポジトリで既にユーザー承認済みの確定値（`performance-floor-decision.md`
  §8・§9・§10）をそのまま spec 側へ反映する提案であり、新たな数値判断は含まない。ただし 2.2 節の
  CUDA 行限定条件は表更新後も継続記載することを提案する。
- 3 節の一本化調整（G-1 との親イシュー共有・G-5 の REQ-12 改定不要判断の反映）を、spec 側起票時の
  運用として提案する。
