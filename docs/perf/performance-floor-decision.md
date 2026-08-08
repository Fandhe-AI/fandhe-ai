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
| #156（TASK-8.3b） | Metal f16 対 PyTorch MPS f16 | **実測未実施**（macOS 実機なし。Linux worktree で型検査のみ）。数値一致（`cpu_metal_f16_parity.rs`）も実機未実行 | `docs/perf/metal-f16-vs-mps-f16.md` |
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
| CPU 対 PyTorch CPU | 最適化後 | 20% | 確定（変更なし） | NEON intrinsics 実効効率見積もりに基づく確定値（暫定値ではない）。本イシュー対象外 |
| CUDA f32 対 PyTorch CUDA | 初期リリース | 10% | 確定（変更なし） | PoC-v2-3 実測 10.3%（10% 以上 5% 刻み切り下げ）。確定済み値であり本イシュー対象外 |
| CUDA f32 対 PyTorch CUDA | 最適化後 | 40%（暫定） | **暫定維持** | #157: 実機（GB10+NVRTC）再実測なし。candidate floor は `n/a`。再確定条件は §4 に従う |
| CUDA f16 対 PyTorch f16 | 初期リリース | 下限を設定しない | **未設定維持** | tensor core 未使用のスカラー実装同士の比較（実測 1.9%）は指標として無意味（REQ-8 脚注）。本イシュー対象外 |
| CUDA f16 対 PyTorch f16 | 最適化後 | 40%（暫定） | **暫定維持** | CUDA f32/最適化後と同一理由（#157） |
| Metal f32 対 PyTorch MPS | 初期リリース | 20% | 確定（変更なし） | PoC-v2-4 実測 23.2%（10% 以上 5% 刻み切り下げ）。確定済み値であり本イシュー対象外 |
| Metal f32 対 PyTorch MPS | 最適化後 | 30% | 確定（変更なし） | PoC-v2-4 事前固定判定基準を据え置いた確定値（暫定値ではない）。本イシュー対象外 |
| Metal f16 対 PyTorch MPS f16 | 初期リリース | 未設定 | **未設定維持** | #156: 実測未実施（手順・テンプレート整備のみ） |
| Metal f16 対 PyTorch MPS f16 | 最適化後 | 未設定 | **未設定維持** | #156: 実測未実施。自作カーネルでの f16 実測後に丸め規則で設定する（REQ-8） |
| Transformer 複合ワークロード | — | 下限を設定しない | **未設定維持** | #155: 実機実測なし。QEMU 参考値（約 6.1%）は naive 経路混入・非実機の 2 重下振れ要因を含むため根拠に使わない（`transformer-workload-measurement.md` 明記に従う） |

Metal f16 の K=4096 ストレスケース許容誤差再評価（#156 が本イシューへ委ねた事項）についても、
実機結果が存在しないため「判断材料なし・実機実測後に再評価する（許容誤差は変更しない）」と
記録する（§5 (b)）。

## 4. 再確定条件・手順

実機実測が揃った際、以下の手順で再確定する:

1. 各記録テンプレート（`transformer-workload-measurement.md`・`metal-f16-vs-mps-f16.md`・
   `cuda-floor-remeasurement.md`）の記入待ち箇所に実機実測値（中央値・Q1/Q3）を転記する
2. 判定対象形状（M=N=K=2048・4096 の実測比率の最小値。512 は参考値。REQ-8「判定対象形状」節）を
   `bench_harness::rounding::floor_lower_bound`（本イシューで一本化。§6 参照）へ適用し候補下限値を得る
3. 本ドキュメント §3 の確定表を実測結果で更新する
4. ユーザー承認（PR レビュー・マージ）を経る
5. `docs/spec/04-requirements.md` REQ-8 節への反映は spec リポジトリ
   （Fandhe-AI/rust-ai-library-spec）側での対応をユーザーへ提案する（本リポの submodule は編集しない）

## 5. 申し送り

- (a) `docs/spec/04-requirements.md` REQ-8 節への本ドキュメントの反映は spec リポジトリ側の対応とする
  （`.claude/rules/out-of-scope-tracking.md`。本リポでは `docs/spec/` を編集しない）
- (b) Metal f16 K=4096 ストレスケースの複合判定逸脱時の許容誤差再評価（#156 が本イシューへ委ねた事項）は、
  実機実測が存在しないため「判断材料なし・実測後に再評価」とし、**許容誤差は変更しない**
  （`.claude/rules/coding-rust.md`「バックエンド間数値一致テストの許容誤差を単独で緩和しない」）
- (c) 実機実測タスク（Apple M4 Max・DGX Spark GB10）自体の追跡は親 #154 の残タスクとして扱う。
  新規 Issue 起票はユーザー承認事項のため、本ドキュメントでは提案に留め、PR 本文で改めて提案する

## 6. 丸め規則の一本化（本イシューで実施）

`crates/backend-cuda/examples/cuda_floor_bench.rs` のインライン丸め実装（`floor_round`）を
`bench_harness::rounding::floor_lower_bound`（TASK-8.2b・#153。fail-closed 入力検証付き）へ
一本化した（#157 の out-of-scope 申し送り「マージ後は #158/#159 で一本化する」への対応）。
既存の丸め規則単体テスト（仕様例突合・境界・非減少性・非有限値/負値防御）は一本化後の API に
対する回帰テストとして維持している。詳細は同ファイルの変更差分・`docs/perf/cuda-floor-remeasurement.md`
の該当節を参照。

## 7. 関連ドキュメント

- `docs/perf/transformer-workload-measurement.md`（#155）
- `docs/perf/metal-f16-vs-mps-f16.md`（#156）
- `docs/perf/cuda-floor-remeasurement.md`（#157）
- `docs/performance-targets.md`（TASK-8.4・#159。本ドキュメントを入力として全バックエンド横断の一覧を整備する）
- `crates/bench-harness/src/threshold.rs`（REQ-8 下限表のデータ化・自動合否判定。本イシューでは不変更）
- `crates/bench-harness/src/rounding.rs`（丸め規則の公開 API。TASK-8.2b・#153）
