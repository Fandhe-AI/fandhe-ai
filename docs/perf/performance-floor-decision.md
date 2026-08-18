# 性能下限 確定記録（#158・TASK-8.3d）

イシュー #158「chore(bench-harness): TASK-8.3d 下限確定・記録（人間判断）」の確定記録。
親 #154（TASK-8.3「未実測領域の実測・下限確定」）の最終工程であり、先行 3 イシュー
（#155 TASK-8.3a・#156 TASK-8.3b・#157 TASK-8.3c）の実測成果を集約し、REQ-8 段階的下限表の
各行について確定判断を記録する。

## 1. 位置づけ・承認の扱い

`docs/spec/05-tasks.md:290` は TASK-8.3 の担当欄を「共同（計測実行は Claude Code、下限値の
最終確定は人間）」と定めている。本ドキュメントはその最終工程の**判断案**であり、記載する
確定判断は本イシュー #158 の PR レビュー・マージ（人間承認）をもって成立する
（先例 #343「TASK-3.3e 結果評価・完了判定（人間）」と同じ扱い）。受け入れ条件
「承認済み下限確定値が記録されている」は、この PR のマージによって充足される。

本ドキュメントは**閾値の緩和を一切行わない**。`crates/bench-harness/src/threshold.rs::floor_spec`
の定数は本イシューで変更しない。

## 2. 入力の集約（先行 3 イシューの状態）

| イシュー | 対象 | 実測状態 | 記録 |
|---------|------|---------|------|
| #155（TASK-8.3a） | Transformer 複合ワークロード | **非実機**（QEMU 仮想 CPU）の参考実測のみ。対 PyTorch 比 約 6.1%。実機（Apple M4 Max・DGX Spark GB10）値は記入待ち | `docs/perf/transformer-workload-measurement.md` |
| #156（TASK-8.3b） | Metal f16 対 PyTorch MPS f16 | **実測未実施**（macOS 実機なし。Linux worktree で型検査のみ）。数値一致（`cpu_metal_f16_parity.rs`）も実機未実行 → **その後 #380（数値一致 6 件全 PASS）・#383（TFLOPS 実測）で完了し、#386 が §8 で下限確定**（歴史的記録として本行は書き換えない） | `docs/perf/metal-f16-vs-mps-f16.md` |
| #157（TASK-8.3c） | CUDA f32/f16 最適化後下限（暫定 40%） | 実測バイナリ `cuda_floor_bench.rs` は整備済みだが**実機（GB10+NVRTC）実測は記入待ち**。candidate optimized floor は「判定対象形状すべてで同一実機再計測値が揃った場合のみ」出力される設計で、現時点で `n/a` | `docs/perf/cuda-floor-remeasurement.md` |

いずれも REQ-8 が要求する「同一ハードウェア上の同一バックエンド比較」を満たす実機実測が
1 領域も揃っていない。

## 3. 確定判断（据え置き確定）

実機実測が存在しない以上、丸め規則を新たに適用して下限を引き上げ・引き下げる根拠は存在しない。
よって本イシューの確定案は**現行 REQ-8 下限表（`docs/spec/04-requirements.md:177-183`。
`threshold.rs::floor_spec` と同一値）の全行据え置き確定**とする。丸め規則を新規適用した行はなく、
閾値緩和もない。

| バックエンド・精度 | 段階 | 現行値 | 確定状態 | 根拠 |
|---|---|---|---|---|
| CPU 対 PyTorch CPU | 初期リリース | 5% | 確定（変更なし） | PoC-v2-1 実測 5.3%（10% 未満 1% 刻み切り下げ）。確定済み値であり本イシュー対象外 |
| CPU 対 PyTorch CPU | 最適化後 | 20% | 確定（変更なし）（**→ §10 で 20% を再確認〈#577〉**） | NEON intrinsics 実効効率見積もりに基づく確定値（暫定値ではない）。本イシュー対象外 |
| CUDA f32 対 PyTorch CUDA | 初期リリース | 10% | 確定（変更なし） | PoC-v2-3 実測 10.3%（10% 以上 5% 刻み切り下げ）。確定済み値であり本イシュー対象外 |
| CUDA f32 対 PyTorch CUDA | 最適化後 | 40%（暫定） | **暫定維持**（当時。**→ §9 で 25% に確定〈#393〉。→ §10 で 50% に再確定〈#577〉**） | #157: 実機（GB10+NVRTC）再実測なし。candidate floor は `n/a`。再確定条件は §4 に従う |
| CUDA f16 対 PyTorch f16 | 初期リリース | 下限を設定しない | **未設定維持** | tensor core 未使用のスカラー実装同士の比較（実測 1.9%）は指標として無意味（REQ-8 脚注）。本イシュー対象外 |
| CUDA f16 対 PyTorch f16 | 最適化後 | 40%（暫定） | **暫定維持**（当時。**→ §9 で 10% に確定〈#393〉。→ §10 で 35% に再確定〈#577〉**） | CUDA f32/最適化後と同一理由（#157） |
| Metal f32 対 PyTorch MPS | 初期リリース | 20% | 確定（変更なし） | PoC-v2-4 実測 23.2%（10% 以上 5% 刻み切り下げ）。確定済み値であり本イシュー対象外 |
| Metal f32 対 PyTorch MPS | 最適化後 | 30% | 確定（変更なし）（当時。**→ §10 で 10% に引き下げ〈#577〉**） | PoC-v2-4 事前固定判定基準を据え置いた確定値（暫定値ではない）。本イシュー対象外 |
| Metal f16 対 PyTorch MPS f16 | 初期リリース | 未設定 | **未設定維持**（当時。**→ §8 で 15% に確定〈#386〉**） | #156: 実測未実施（手順・テンプレート整備のみ） |
| Metal f16 対 PyTorch MPS f16 | 最適化後 | 未設定 | **未設定維持**（当時。**→ §8 でも未設定継続〈#386〉。→ §10 で 15% に新設〈#577〉**） | #156: 実測未実施。自作カーネルでの f16 実測後に丸め規則で設定する（REQ-8） |
| Transformer 複合ワークロード | — | 下限を設定しない | **未設定維持** | #155: 実機実測なし。QEMU 参考値（約 6.1%）は naive 経路混入・非実機の 2 重下振れ要因を含むため根拠に使わない（`transformer-workload-measurement.md` 明記に従う） |

Metal f16 の K=4096 ストレスケース許容誤差再評価（#156 が本イシューへ委ねた事項）についても、
実機結果が存在しないため「判断材料なし・実機実測後に再評価する（許容誤差は変更しない）」と
記録する（§5 (b)）。

## 4. 再確定条件・手順

実機実測が揃った際、以下の手順で再確定する:

1. 各記録テンプレート（`transformer-workload-measurement.md`・`metal-f16-vs-mps-f16.md`・
   `cuda-floor-remeasurement.md`）の記入待ち箇所に実機実測値（中央値・Q1/Q3）を転記する
2. 判定対象形状（M=N=K=2048・4096 の実測比率の最小値。512 は参考値。REQ-8「判定対象形状」節）を
   `bench_harness::floor_lower_bound`（本イシューで一本化。§6 参照）へ適用し候補下限値を得る
3. 本ドキュメント §3 の確定表を実測結果で更新する
4. ユーザー承認（PR レビュー・マージ）を経る
5. `docs/spec/04-requirements.md` REQ-8 節への反映は spec リポジトリ
   （Fandhe-AI/rust-ai-library-spec）側での対応をユーザーへ提案する（本リポの submodule は編集しない）

## 5. 申し送り

- (a) `docs/spec/04-requirements.md` REQ-8 節への本ドキュメントの反映は spec リポジトリ側の対応とする
  （`.claude/rules/out-of-scope-tracking.md`。本リポでは `docs/spec/` を編集しない）
- (b) Metal f16 K=4096 ストレスケースの複合判定逸脱時の許容誤差再評価（#156 が本イシューへ委ねた事項）は、
  実機実測が存在しないため「判断材料なし・実測後に再評価」とし、**許容誤差は変更しない**
  （`.claude/rules/coding-rust.md`「バックエンド間数値一致テストの許容誤差を単独で緩和しない」）。
  **解消（#386・§8）**: #383 の再確認実測で `f16_k4096_stress` を含む数値一致 6 件全件が
  複合判定 PASS のまま維持されていることを確認したため、本項の再評価は「不要」として解消する。
  許容誤差は #156 時点から変更していない
- (c) 実機実測タスク（Apple M4 Max・DGX Spark GB10）自体の追跡は親 #154 の残タスクとして扱う。
  新規 Issue 起票はユーザー承認事項のため、本ドキュメントでは提案に留め、PR 本文で改めて提案する

## 6. 丸め規則の一本化（本イシューで実施）

`crates/backend-cuda/examples/cuda_floor_bench.rs` のインライン丸め実装（`floor_round`）を
`bench_harness::floor_lower_bound`（TASK-8.2b・#153。fail-closed 入力検証付き）へ
一本化した（#157 の out-of-scope 申し送り「マージ後は #158/#159 で一本化する」への対応）。
既存の丸め規則単体テスト（仕様例突合・境界・非減少性・非有限値/負値防御）は一本化後の API に
対する回帰テストとして維持している。詳細は同ファイルの変更差分・`docs/perf/cuda-floor-remeasurement.md`
の該当節を参照。

## 7. 関連ドキュメント

- `docs/perf/transformer-workload-measurement.md`（#155）
- `docs/perf/metal-f16-vs-mps-f16.md`（#156・#380・#383）
- `docs/perf/cuda-floor-remeasurement.md`（#157）
- `docs/perf/cpu-gemm-optimized-remeasurement.md`（Phase F・CPU 分。§10 の入力）
- `docs/perf/cuda-optimized-remeasurement.md`（#571・Phase F-1。§10 の入力）
- `docs/perf/metal-floor-remeasurement.md`（#572・Phase F-2。§10 の入力）
- `docs/performance-targets.md`（TASK-8.4・#159。本ドキュメントを入力として全バックエンド横断の一覧を整備する。#386 §8・#393 §9 も入力に追加済み。§10〈#577〉の反映は #579 の担当）
- `crates/bench-harness/src/threshold.rs`（REQ-8 下限表のデータ化・自動合否判定。#386 で Metal f16 初期リリース行、#393 で CUDA f32/f16 最適化後行、#577 で Optimized 段 5 行を更新）
- `crates/bench-harness/src/rounding.rs`（丸め規則の公開 API。TASK-8.2b・#153）

## 8. 追補（#386・2026-08-10）: Metal f16 初期リリース下限の確定

§3 の Metal f16 初期リリース行は「実機実測が存在しない」ことを理由に未設定維持と据え置いたが、
その後の実測完了を受けてイシュー #386 で確定した。本節はその確定内容を追記する（§3 の
歴史的記録は書き換えず、該当行に本節への参照注記のみ付す）。

### 入力

- **TFLOPS 実測**: イシュー #383（PR #443）。計測環境 Apple M4 Max・macOS 26.6・torch 2.13.0
  （`docs/perf/metal-f16-vs-mps-f16.md`「実測結果」節）。判定対象形状（M=N=K=2048・4096）の
  Metal/PyTorch 比: 2048 → 21.6%、4096 → 18.6%。REQ-8 の判定対象形状の実測比率の最小値は
  **18.6%（size=4096）**
- **数値一致**: イシュー #380 で `cpu_metal_f16_parity.rs` 6 件（`f16_k4096_stress` を含む）が
  f32 累算化後に全 PASS。#383 の再確認実測でも同一 SHA で 6 件全件 PASS を再確認済み
  （`docs/perf/metal-f16-vs-mps-f16.md`「数値一致」節）。限定条件を要する未解消の逸脱はない

### 判断

`bench_harness::floor_lower_bound(18.6)` = **15%**（10% 以上のため 5% 刻み切り下げ。
`crates/bench-harness/src/rounding.rs::spec_metal_f16_measured_ratio` で機械的に固定）を
Metal f16 初期リリース段階の性能下限として**確定**する（暫定値ではない）。数値一致に
限定条件がないため、下限にも限定条件を付けない。最適化後段階は本イシューでは設定しない
（承認記録どおり、今後の最適化タスクの実測に基づき本規則で再確定する）。

### 承認記録

イシュー #386 のコメント（2026-08-10・リポジトリオーナー）に上記判断のユーザー承認記録が
存在する。先例 #158 §1 と同じく、本追補の最終成立は本イシュー #386 の PR レビュー・マージ
（人間承認）による。

### 反映箇所

- `crates/bench-harness/src/threshold.rs::floor_spec`: `(MetalF16, InitialRelease)` を
  `FloorSpec::NotSet` から `FloorSpec::Ratio { percent: 15.0, provisional: false }` へ更新
  （本追補とセットで実施）
- `docs/performance-targets.md` §2・§6: 転記整合（本追補とセットで実施）

### spec 反映

`docs/spec/04-requirements.md`（2026-08-05 版）REQ-8 表への反映は spec リポジトリ
（Fandhe-AI/rust-ai-library-spec）側での対応をユーザーへ提案する（§5(a) と同じ扱い。
本リポでは `docs/spec/` submodule を編集しない）。

## 9. 追補（#393・2026-08-10）: CUDA f32/f16 最適化後下限の確定

§3 の CUDA f32/f16 最適化後行は「実機（GB10+NVRTC）再実測なし」を理由に暫定 40% 維持と
据え置いたが、その後の実測完了を受けてイシュー #393 で再確定した。本節はその確定内容を
追記する（§3 の歴史的記録は書き換えず、該当行に本節への参照注記のみ付す）。

### 入力

- **TFLOPS 実測**: イシュー #390（PR #444）。計測環境 DGX Spark GB10・CUDA 13.0 系
  （`docs/perf/cuda-floor-remeasurement.md`「実測結果（#390 実機実測）」節）。3 run 反復実行の
  うえ、同一実機で PyTorch 参照値も再計測した。判定対象形状（M=N=K=2048・4096）の実測比率
  最小値: f32 = **25.64〜25.69%**（4096 側が最小、3 run とも安定）、f16 = **12.97%**
  （2048 側が最小、3 run とも同一値）。候補算出経路は f32 = `wmma_tf32`（WMMA(TF32) opt）、
  f16 = `mma_f16`（`mma.sync` パイプライン）（同ドキュメント「丸め適用後の候補下限値」節）
- **数値一致（parity）の限定条件**: 候補算出経路（`wmma_tf32`・`mma_f16`）はいずれも #389 §5.3
  が示す数値一致 parity の恒常 fail 対象と一致する（`cuda-floor-remeasurement.md`「数値一致
  （parity）状態の限定条件」節）。§8（Metal f16）と異なり、本追補は parity 未達のまま実測基準の
  下限を確定する点が承認記録の限定条件として明記されている

### 判断

`bench_harness::floor_lower_bound(25.64)` = **25%**、`floor_lower_bound(12.97)` = **10%**
（いずれも 10% 以上のため 5% 刻み切り下げ。`crates/bench-harness/src/rounding.rs::
spec_cuda_f32_optimized_measured_ratio`／`spec_cuda_f16_optimized_measured_ratio` で機械的に
固定）を CUDA f32/f16 最適化後段階の性能下限として**確定**する（暫定値ではない）。

`provisional: false` とする根拠: 暫定 40% の解消条件は「tensor core 実装完了後の実測で本値を
再確定すること」（`docs/spec/04-requirements.md:180-181`）であり、#390 の実機実測がこの条件を
満たしたため。ただし §8（Metal f16 初期リリース）とは異なり、本追補は数値一致 parity が
恒常 fail の経路をそのまま採用しており、下限値そのものに限定条件を付ける。

### 承認記録

イシュー #393 のコメント（2026-08-10・リポジトリオーナー aLiz-Nancy・author association: MEMBER）
に上記判断のユーザー承認記録が存在する。先例 #158 §1・#386 と同じく、本追補の最終成立は本
イシュー #393 の PR レビュー・マージ（人間承認）による。

承認コメントに明記された限定条件（本追補で追跡する）:

1. 候補算出経路（`wmma_tf32`・`mma_f16`）は #389 §5.3 の数値一致 parity 未達対象と一致する
2. 本承認は「実測基準でゲートを機能させ、今後の最適化で性能を改善していく」方針による
3. **#186（REQ-2 閾値改定）の解決後に本下限値を再確認する**こと（parity green の経路で
   再実測し、必要なら再確定する）。#186 は 2026-08-06 に close 済みだが、閾値定数
   （`RELATIVE_TOLERANCE`・`ABSOLUTE_RESCUE_THRESHOLD`）自体は変更されておらず
   （close コミット紐付けなし。`docs/perf/cuda-tensor-core-tolerance-evaluation.md` §4「結論」）、
   TF32/f16 Tensor Core 経路の複合判定改定は REQ-2 改定として spec リポジトリ側対応待ちの
   ままである。よって本限定条件は**解消しておらず、継続する**（本追補では下限値を変更しない。
   値の再変更は新たなユーザー承認事項）

### 反映箇所

- `crates/bench-harness/src/threshold.rs::floor_spec`: `(CudaF32, Optimized)` を
  `FloorSpec::Ratio { percent: 40.0, provisional: true }` から
  `FloorSpec::Ratio { percent: 25.0, provisional: false }` へ、`(CudaF16, Optimized)` を
  `FloorSpec::Ratio { percent: 40.0, provisional: true }` から
  `FloorSpec::Ratio { percent: 10.0, provisional: false }` へ更新（本追補とセットで実施）
- `crates/bench-harness/src/rounding.rs`: spec 実測値再現テスト 2 件を追加（本追補とセットで実施）
- `docs/performance-targets.md` §2・§6: 転記整合（本追補とセットで実施）

### spec 反映

`docs/spec/04-requirements.md`（2026-08-05 版）REQ-8 表への反映は spec リポジトリ
（Fandhe-AI/rust-ai-library-spec）側での対応をユーザーへ提案する（§5(a)・§8「spec 反映」と
同じ扱い。本リポでは `docs/spec/` submodule を編集しない）。

## 10. 追補（#577・2026-08-18）: Optimized 段 5 行の再確定（GEMM 性能改善ツリー Phase F）

GEMM 性能改善ツリー（ルート #479）Phase F（親 #569「再計測・parity 非後退確認・REQ-8 下限再確定」）
の再計測（F-1〜F-3: #571・#572・CPU 分）を踏まえ、Optimized 段の性能下限を再確定する。§8（Metal f16
初期リリース）・§9（CUDA f32/f16 最適化後）と同じ追補形式を踏襲する。

### 入力

| backend_dtype | 入力ドキュメント | 判定対象形状の実測比率最小値 | 候補算出経路 |
|---|---|---|---|
| CpuF32 | `docs/perf/cpu-gemm-optimized-remeasurement.md` | 24.7%（size=2048） | NEON intrinsics 適用済みカーネル |
| CudaF32 | `docs/perf/cuda-optimized-remeasurement.md`（#571・PR #725 系列） | 51.96%（size=4096・Rust/PyTorch とも 5 run 中央値） | `wmma_tf32`（WMMA(TF32) opt） |
| CudaF16 | 同上 | 37.47%（size=4096・Rust/PyTorch とも 5 run 中央値） | `mma_f16`（`mma.sync` パイプライン） |
| MetalF32 | `docs/perf/metal-floor-remeasurement.md`（#572・PR #725 系列） | 13.01%（size=4096） | `dispatch_tiled_prepared`（simdgroup タイル化） |
| MetalF16 | 同上 | 18.78%（size=4096） | `dispatch_f16_prepared_unverified` |

### 判断

`bench_harness::floor_lower_bound` を適用した結果（いずれも 10% 以上のため 5% 刻み切り下げ）:

| backend_dtype | 旧値 | 新値 | 変化 |
|---|---|---|---|
| CpuF32 | 20% | **20%** | 維持・確定（`floor_lower_bound(24.7)` = 20） |
| CudaF32 | 25%（#393） | **50%** | 引き上げ（`floor_lower_bound(51.96)` = 50） |
| CudaF16 | 10%（#393） | **35%** | 引き上げ（`floor_lower_bound(37.47)` = 35） |
| MetalF32 | 30% | **10%** | 引き下げ（`floor_lower_bound(13.01)` = 10） |
| MetalF16 | 未設定 | **15%**（新設） | 新設（`floor_lower_bound(18.78)` = 15） |

初期リリース段（`InitialRelease`）は全行変更しない。

**CpuF32（維持・確定）**: §3 では「本イシュー対象外」（#158 当時から既に確定済みの値）として
扱われていた行であり、本追補で新たに丸め規則を適用したわけではない。Phase F 実測
（24.7%・10% 以上のため 5% 刻み切り下げで 20%）は既存の確定値と一致することを再確認したのみで、
値は変更しない。

**CudaF32／CudaF16（引き上げ）**: §9 で 25%／10% に確定した後、Phase F の再計測で候補算出経路
（`wmma_tf32`／`mma_f16`）が変わらないままスループットが向上し、判定対象形状の実測比率が
51.96%／37.47%（4096 側が最小。Rust・PyTorch とも 5 run 中央値）へ改善した。§9 の限定条件（下記「CUDA 限定条件の継続」節）は
本追補でも**解消しておらず、継続する**。

CudaF16 は丸め刻み境界近傍のため、`cuda-optimized-remeasurement.md`「f16 境界注記」節のとおり
5 run 計測（Rust・PyTorch とも）で確認した。分母（PyTorch f16）を 5 run 中央値へ正しく集計すると
run1〜run5 の比はすべて 35% 帯に収まる（当初の 1 run 分母では run1 のみ 40.95% で見かけ上 40% 境界を
跨いでいた）。5 run 中央値 37.47% を採用根拠とし、35% を確定値とする。境界近傍のため run 間で隣接
刻みへ振れる程度の変動があることを申し送る。

**MetalF32（引き下げ）**: 現行 30% は PoC-v2-4 の事前固定判定基準（当時の旧カーネル・バッファ常駐
前提の実測 23.2% 系列）に基づく確定値だった。`docs/performance-targets.md` §4 が定める計測境界
（`dispatch_tiled_prepared` prepared 入口。エンコード＋コマンドバッファ完了待ちのみを計測）へ揃えた
現行計測系列（#572）は、当時の計測系列とは比較不能な非互換の系列であり、その現行系列では 30% が
恒常的に未達（判定対象形状の実測比率最小値 13.01%）だった。よって §4 準拠の現行計測系列を根拠に
30%→10% へ引き下げて再確定する。`metal-floor-remeasurement.md`「温度ドリフト注記」節の
worst-case ペアリング（最遅 Metal ÷ 最速 PyTorch）確認でも 10% は不変。数値一致
（`dispatch_tiled_prepared_matches_dispatch_variant`・`cpu_metal_f16_parity.rs` 系）はいずれも全 PASS
のため CUDA 行のような限定条件は付けない。

**MetalF16（新設）**: #386 承認記録どおり「今後の最適化タスクの実測に基づき丸め規則で再確定する」
段階として未設定だったが、Phase F（#572）で初の判定対象形状実測が揃ったため 15% を新設する。
`metal-floor-remeasurement.md`「温度ドリフト注記」節の worst-case ペアリング確認でも 15% は不変。
数値一致（`cpu_metal_f16_parity.rs` 6 件）は全 PASS のため限定条件は付けない。

### CUDA 限定条件の継続

§9 の承認記録が明記した 3 点の限定条件は、本追補でも解消せず継続する（値の再確定自体は本追補で
実施済みだが、限定条件そのものの解消は別途の判断事項）:

1. 候補算出経路（`wmma_tf32`・`mma_f16`）は #389 §5.3 の数値一致 parity 恒常 fail 対象と一致する
   （`cuda-optimized-remeasurement.md`「数値一致（parity）状態の限定条件」節で Phase F 後も再確認
   済み: `wmma_tf32_opt_kernel_k4096_stress`・`mma_f16_k4096_stress` 等が #389 §5.3 の恒常 fail 範囲内
   で後退なし）
2. 本承認は「実測基準でゲートを機能させ、今後の最適化で性能を改善していく」方針による
3. **#186（REQ-2 閾値改定）の解決後に本下限値を再確認する**こと。#186 は 2026-08-06 に close 済み
   だが、閾値定数（`RELATIVE_TOLERANCE`・`ABSOLUTE_RESCUE_THRESHOLD`）自体は変更されておらず、
   TF32/f16 Tensor Core 経路の複合判定改定は REQ-2 改定として spec リポジトリ側対応待ちのままである。
   よって本限定条件は §9 から**解消しておらず、継続する**
4. **（本追補で新規追加・CudaF32 のみ）f32 候補下限 50% の根拠実測は `wmma_tf32_staged` 経路の値
   である**（`launch_wmma_tf32` の 3 段選択が判定対象形状で staged を選ぶため。
   `cuda-optimized-remeasurement.md`「数値一致（parity）確認」節の経路対応を参照）。staged 経路は
   正本 `docs/perf/cuda-parity-baseline.md` にベースライン未計測（`baseline_provenance_unconfirmed`）
   のため **parity 非後退が判定不能**であり、staged 固有ベースラインの確立・非後退確認を後続タスク
   として追跡する（f16 側 `mma_f16` は非後退確認済みでこの限定条件の対象外）。本限定条件を承知の
   うえで 50% を維持する判断を 2026-08-19 にユーザーが承認した（#577 イシューコメントの追記参照）

### 承認記録

イシュー #577・2026-08-18・リポジトリオーナー（Nancy さん・GitHub: aLiz-Nancy）が対話セッションで
上記判断を承認した。§1・§8・§9 と同じく、本追補の最終成立は本イシュー #577 の PR レビュー・マージ
（人間承認）による。

先例（§8・§9）と同様、承認内容は #577 のイシューコメント（2026-08-18 の承認記録コメント。
backend_dtype 別の承認値・根拠実測・CUDA 限定条件の継続を明記）として監査可能な形で記録済み。
対話セッションでの承認を同コメントへ転記する形で残している。

### 反映箇所

- `crates/bench-harness/src/threshold.rs::floor_spec`: `(CpuF32, Optimized)` は値不変・根拠コメントの
  みを更新。`(CudaF32, Optimized)` を `FloorSpec::Ratio { percent: 25.0, provisional: false }` から
  `FloorSpec::Ratio { percent: 50.0, provisional: false }` へ、`(CudaF16, Optimized)` を
  `FloorSpec::Ratio { percent: 10.0, provisional: false }` から
  `FloorSpec::Ratio { percent: 35.0, provisional: false }` へ、`(MetalF32, Optimized)` を
  `FloorSpec::Ratio { percent: 30.0, provisional: false }` から
  `FloorSpec::Ratio { percent: 10.0, provisional: false }` へ、`(MetalF16, Optimized)` を
  `FloorSpec::NotSet` から `FloorSpec::Ratio { percent: 15.0, provisional: false }` へ更新
  （本追補とセットで実施）
- `crates/bench-harness/src/threshold.rs`（モジュール冒頭コメント「例外」節・`FloorSpec` ドキュメント
  コメント）: 上記変更に合わせて出典・NotSet 該当行の記述を更新（本追補とセットで実施）
- `crates/bench-harness/src/threshold.rs`（`#[cfg(test)]` 単体テスト）・
  `crates/bench-harness/tests/threshold_judgment.rs`（統合テスト）: 旧下限値をハードコードしていた
  期待値（`metal_f16_optimized_is_not_applicable` → `metal_f16_optimized_floor_15_percent_boundary` へ
  改名・境界固定へ変更／`cuda_f32_optimized_confirmed_floor_is_not_flagged_provisional` の
  `floor_percent` 期待値／`metal_f32_both_stages_use_distinct_floors` の実測値・期待値）を新値へ追従
  （判定ロジック・tolerance は変更していない）
- `docs/performance-targets.md` §2・§6 の転記整合は本追補のスコープ外とする（担当イシュー #579）

### spec 反映

`docs/spec/04-requirements.md`（2026-08-05 版）REQ-8 表への反映は spec リポジトリ
（Fandhe-AI/rust-ai-library-spec）側での対応をユーザーへ提案する（§5(a)・§8・§9「spec 反映」と
同じ扱い。本リポでは `docs/spec/` submodule を編集しない）。
