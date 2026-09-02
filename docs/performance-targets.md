# 性能目標（対 PyTorch 性能下限）

TASK-8.4（REQ-8・M5・イシュー #159）の成果物。REQ-8 が要求する「バックエンド別・対 PyTorch
性能下限」を、v2 段階的下限（初期リリース／最適化後）としてバックエンド横断で一覧化する。

## 1. 位置づけ・承認の扱い

`docs/spec/05-tasks.md` の TASK-8.4 は担当欄を「人間」と定めている。先例 #158
（`docs/perf/performance-floor-decision.md` §1）と同じく、本ドキュメントは**判断案（既存確定値の
転記・集約）**であり、記載内容は本イシュー #159 の PR レビュー・マージ（人間承認）をもって
成立する。

本ドキュメントは**閾値・許容誤差を一切変更しない**。`crates/bench-harness/src/threshold.rs::floor_spec`
の定数・`docs/spec/04-requirements.md` REQ-8 表・バックエンド間数値一致テストの許容誤差は
すべて据え置きで、`docs/perf/performance-floor-decision.md` §3（#158 確定記録）＋ §8（#386。
Metal f16 初期リリース下限の確定）＋ §9（#393。CUDA f32/f16 最適化後下限の確定）＋ §10（#577。
GEMM 性能改善ツリー Phase F による Optimized 段 5 行の再確定）で確定した値をそのまま転記する
（新しい判断・数値変更は行わない）。

## 2. v2 段階的下限表（正）

`docs/perf/performance-floor-decision.md` §3・§8（#386）・§9（#393）・§10（#577）・
`crates/bench-harness/src/threshold.rs::floor_spec` と同一値。

| バックエンド・精度（比較対象・実機） | 実測比率の最小値（2048/4096、出典） | 初期リリース下限 | 最適化後下限 | 状態 |
|---|---|---|---|---|
| CPU 対 PyTorch CPU（Apple M4 Max、PyTorch 2.13.0 macOS arm64） | 初期リリース: 5.3%（`03-poc/poc-v2-1-tensor-cpu-gemm/README.md`「計測結果」節）／最適化後: 24.7%（size=2048、`docs/perf/cpu-gemm-optimized-remeasurement.md`。#574・PR #713） | **5%** | **20%** | 確定（最適化後は #577・§10 で既存確定値 20% と一致を再確認。値は変更なし） |
| CUDA f32 対 PyTorch CUDA（DGX Spark GB10、PyTorch 2.13.0+cu130） | 初期リリース: 25.64〜25.69%（`docs/perf/cuda-floor-remeasurement.md`「実測結果（#390 実機実測）」節、size=4096 が最小）／最適化後: 51.96%（size=4096・Rust/PyTorch とも 5 run 中央値、`docs/perf/cuda-optimized-remeasurement.md`。#571・PR #725） | **10%** | **50%** | 初期リリース: 確定／最適化後: **確定**（#577・§10。25%→50% へ引き上げ。限定条件: §9 由来の限定条件 1〜3〈候補経路 `wmma_tf32` は #389 §5.3 の parity 恒常 fail 対象と一致・#186 解決後の再確認を継続〉に加え、限定条件 4〈根拠実測が `wmma_tf32_staged` 経路〉は #726・2026-08-19 の実機ベースライン確立で**解消済み**〈`cuda-parity-baseline.md` §3・staged 行の非後退 pass 確認済み〉） |
| CUDA f16 対 PyTorch f16（同上） | 初期リリース: 12.97%（同上、size=2048 が最小）／最適化後: 37.47%（size=4096・Rust/PyTorch とも 5 run 中央値、同上） | **下限を設定しない**（脚注参照） | **35%** | 初期リリース: 未設定（構造的に指標無意味）／最適化後: **確定**（#577・§10。10%→35% へ引き上げ。限定条件: §9 由来の限定条件 1〜3〈候補経路 `mma_f16` は #389 §5.3 の parity 恒常 fail 対象と一致・#186 解決後の再確認を継続〉。丸め刻み境界近傍のため 5 run 計測で確認済み。`mma_f16` は非後退確認済み） |
| Metal f32 対 PyTorch MPS（Apple M4 Max、PyTorch 2.13.0） | 初期リリース: 23.2%（`03-poc/poc-v2-4-metal-gemm/README.md`「PyTorch MPS 比」表、size=4096）／最適化後: 13.01%（size=4096、`docs/perf/metal-floor-remeasurement.md`。#572・PR #725） | **20%** | **10%** | 確定（最適化後は #577・§10。旧 30% は §4 準拠前の非互換な旧計測系列由来のため、現行 prepared 計測系列〈`dispatch_tiled_prepared`〉で 30%→10% へ再確定。限定条件なし） |
| Metal f16 対 PyTorch MPS f16（同上） | 初期リリース: 18.6%（`docs/perf/metal-f16-vs-mps-f16.md`「実測結果」節、size=4096。#383）／最適化後: 18.78%（size=4096、`docs/perf/metal-floor-remeasurement.md`。同上） | **15%** | **15%**（新設） | 初期リリース: 確定（#386・実機実測に基づく。数値一致 #380 全 PASS・限定条件なし）／最適化後: **確定**（#577・§10 で新設。数値一致〈`cpu_metal_f16_parity.rs` 6 件〉全 PASS・限定条件なし） |
| Transformer 複合ワークロード（attention/softmax/LayerNorm を含む複合演算） | 非実機参考値 約 6.1%（QEMU 仮想 CPU） | **下限を設定しない** | **下限を設定しない** | 未実測（`docs/perf/transformer-workload-measurement.md`・#155。QEMU 参考値は naive 経路混入・非実機の 2 重下振れ要因を含むため根拠に使わない） |

上記各行の対象カーネル関数名・計測境界・Metal 2 系列の対応関係・CPU 基準実機の突合は
`docs/perf/gemm-optimization-baseline.md`（#481）を参照（#481 自体は本表の数値・下限・状態列を
変更していない。最適化後下限・状態列のその後の更改は #577・§10 による）。

- **CUDA f16 初期リリースの扱い（脚注）**: 実測比率 1.9%（`03-poc/poc-v2-3-cuda-gemm/README.md`）
  は tensor core 未使用のスカラー実装同士の比較であり、下限値として意味を持たない。
  tensor core（WMMA/mma）実装完了後、下記 §3 の丸め規則を適用して初期リリース下限を確定する。
  それまでは実測値 1.9% を制約事項として記録するに留める（REQ-8 脚注）。
- 「確定」は暫定値ではないことを示す。多くは `docs/perf/performance-floor-decision.md`（#158）で
  実機実測なしのまま据え置き確定した値だが、Metal f16 初期リリースは実機実測（#383）に基づき
  #386（§8）で、CUDA f32/f16 最適化後は実機実測（#390）に基づき #393（§9）で確定した後、
  GEMM 性能改善ツリー（#479）Phase F の再計測に基づき #577（§10）で GEMM 5 行の最適化後下限を
  再確定した（CPU f32 は既存値と一致を再確認、CUDA f32/f16 は引き上げ、Metal f32 は引き下げ、
  Metal f16 は新設）。「暫定」は tensor core 実装完了後の実機再実測で再確定する値（本表には
  現存しない）。「未設定」は実機実測が未実施、または実測はあるが下限を設定しない判断のため
  下限自体を設定していない状態（GEMM 5 行には現時点で該当行はない。Transformer 複合ワークロード
  行は実機実測自体が未実施のため状態列では「未実測」と区別する。§6 参照）。CUDA f32/f16 最適化後の「確定」
  には限定条件が付く（候補算出経路が数値一致 parity 恒常 fail 対象と一致・#186 解決後の再確認を
  要する、に加え CUDA f32 は根拠実測の staged 経路 parity ベースライン未確定。§6 参照）。

## 3. 丸め規則

実測比率（対 PyTorch、パーセント値）から下限を導出する規則は次の通り（`docs/spec/04-requirements.md`
REQ-8）。

- 実測比率が 10% 以上の場合は **5% 刻み**で切り下げる
- 実測比率が 10% 未満の場合は **1% 刻み**で切り下げる（10% 未満の領域は 5% 刻みが粗すぎるため）
- 境界（10% ちょうど）は 10% 以上側（5% 刻み）を適用する
- 条件付き追加ステップは廃止済み（v1 の「境界近傍でさらに 1 段安全側へ」は非単調〈実測比率
  16.9% → 15% だが 17.0% → 15% からさらに 1 段下げて 10% となる逆転〉であったため v2 で解消。
  実測比率に対し非減少であることが保証される規則に統一した）
- 個別のバックエンド・精度ごとに異なる丸め規則を設けない。将来のバックエンド・精度追加時も
  同一規則を適用する

実装は `bench_harness::floor_lower_bound`（`crates/bench-harness/src/rounding.rs`）に一本化済み
（#158 §6。旧 `crates/backend-cuda/examples/cuda_floor_bench.rs` のインライン丸め実装から移行）。

## 4. 計測プロトコル

計測は 2 層構造を取る（§2 表の「最適化後」列・`docs/perf/cuda-optimized-remeasurement.md`・
`docs/perf/metal-floor-remeasurement.md` で採用した構成）。

- **内層（1 run 内）**: warmup 20 回以上・計測 20 回以上の中央値・Q1/Q3 を記録する
  （`bench_harness::protocol::run`。PoC-v2-1/3/4 はすべて本条件で実施済み）
- **外層（run 間）**: 内層プロトコルに従う 1 回の実行を「1 run」とし、これを独立に 5 回実行して
  run 間の中央値を代表値として採用する（コーディング規約 `.claude/rules/coding-rust.md`
  「ベンチは 5 回計測の中央値を採用」に合わせる）。§2 表の「Rust/PyTorch とも 5 run 中央値」等の
  記載はこの外層 5 run を指す。PoC-v2 時点の初期リリース確定値は外層 5 run 化以前の内層のみの
  計測であり、Phase F（GEMM 性能改善ツリー #479・`docs/perf/performance-floor-decision.md` §10）
  による最適化後の再計測から外層 5 run を追加適用した（内層の warmup 20／計測 20 という下限自体は
  変更していない）
- 同期方式は「ホスト転送を伴わない完了待ち」で統一する。具体的な完了待ち手段はバックエンド固有
  API に委ねる:
  - CUDA: `stream.synchronize()`（`03-poc/poc-v2-3-cuda-gemm/README.md`）
  - Metal: コマンドバッファ完了待ち・`device.poll(PollType::wait_indefinitely())`
    （`03-poc/poc-v2-4-metal-gemm/README.md`）
  - CPU: バックグラウンド実行がないため該当なし
- 決定的シード（xorshift64*、`crates/bench-harness/src/rng.rs`）を用いる
- **判定対象形状**: 演算律速域（M=N=K=2048・4096）の実測比率の最小値を採る。M=N=K=512 は
  ディスパッチ・起動オーバーヘッドが支配的で試行間ばらつきが大きいため参考値とし、判定には
  用いない（PoC-v2-1「PyTorch CPU 比は 5.3〜38.1%」・PoC-v2-4「size=512 は…比較の代表性が
  低い」の整理を踏襲）

## 5. v1 確定値の位置づけ（v2 下限の根拠に用いない）

v1 の性能下限（**f32 GEMM 30%**・**Transformer 45%**・**f16 GEMM 15%**、いずれも CUDA を
基準としたバックエンド間相対値、Burn 0.21.0／CubeCL 前提）は、Burn 0.21.0 上での到達可能性の
存在証明・回帰検知の目安としての**参考値**であり、**v2 下限の根拠には用いない**
（`docs/spec/04-requirements.md` REQ-8「v1 確定値の位置づけ」受け入れ基準・
`docs/spec/v1-assets-inventory.md` の方針を踏襲）。

v1 は Burn/CubeCL の共通カーネル基盤を前提に「CUDA を基準とした Metal 比」を主指標としていたが、
v2 では各バックエンドのカーネルを個別に自作するため最適化進度によって相対値が大きく変動し
比較の意味を失う。実際、自作実装の実測でも Metal simdgroup（3.134 TFLOPS、
`03-poc/poc-v2-4-metal-gemm/README.md`）が CUDA tiled（1.832 TFLOPS、
`03-poc/poc-v2-3-cuda-gemm/README.md`）を上回り、v1（CUDA が Metal を上回る）と大小関係が
逆転している。この逆転自体が、バックエンド間相対値を主指標に据えることの妥当性を損なう実測
根拠である。よって v2 では各バックエンドを「そのバックエンド上の PyTorch 実装」とのみ比較し、
バックエンド間相対値は参考指標に位置づける（REQ-8「概要」節）。

## 6. 未確定領域と再確定手順

未確定行（Transformer 複合ワークロードのみ。状態列は「未実測」。GEMM 5 行はすべて初期リリース・
最適化後の両段が確定済み。Metal f16 最適化後は #386 で未設定だったが Phase F 再計測に基づき
#577（§10）で 15% を新設して確定済みのため、GEMM 側の未確定領域は解消している）は、実機実測
（Apple M4 Max・DGX Spark GB10）が揃い次第、以下の手順で再確定する
（`docs/perf/performance-floor-decision.md` §4 を要約）。

**CUDA f32/f16 最適化後の限定付き再確認（#393 承認記録・#577 で値のみ更改）**: #393 で確定した
下限は候補算出経路（`wmma_tf32`・`mma_f16`）が #389 §5.3 の数値一致 parity 恒常 fail 対象と一致
したままの実測値であり、承認記録に「#186（REQ-2 閾値改定）の解決後に本下限値を再確認する」限定
条件が付されている。#186 は 2026-08-06 に close 済みだが、閾値定数
（`RELATIVE_TOLERANCE`・`ABSOLUTE_RESCUE_THRESHOLD`）自体は変更されておらず
（`docs/perf/cuda-tensor-core-tolerance-evaluation.md` §4「結論」）、TF32/f16 Tensor Core 経路の
複合判定改定は REQ-2 改定として spec リポジトリ側対応待ちのままである。よってこの限定条件は
継続しており、当該改定が解決し parity green の経路で再実測できるようになった時点で下限値の
再確認（必要なら再確定）を行う（値の再変更は新たなユーザー承認事項）。GEMM 性能改善ツリー
（#479）Phase F の再計測（`docs/perf/cuda-optimized-remeasurement.md`）を踏まえ、#577（§10）で
下限値そのものは 25%→50%（CUDA f32）・10%→35%（CUDA f16）へ更改済みだが、上記限定条件
（候補経路が parity 恒常 fail 対象と一致・#186 解決後の再確認）は解消しておらず継続する。加えて
CUDA f32 の 50% の根拠実測は `launch_wmma_tf32` の 3 段選択が判定対象形状で選ぶ
`wmma_tf32_staged` 経路の値である。#577 承認時点では staged 経路のベースライン未計測により
parity 非後退が判定不能（限定条件 4）だったが、#726（2026-08-19）で staged 固有ベースラインを
実機確立し非後退 pass を確認したため、この限定条件は**解消済み**である
（`docs/perf/performance-floor-decision.md` §10 限定条件 4 の解消追記参照）。

1. 記録テンプレート（`docs/perf/transformer-workload-measurement.md`）の記入待ち箇所に実機実測値
   （中央値・Q1/Q3）を転記する（`metal-f16-vs-mps-f16.md`・`cuda-floor-remeasurement.md` はすでに
   実測記入済み。#386・#393 で確定済みのため対象外）
2. 判定対象形状（§4 参照）の実測比率の最小値を `bench_harness::floor_lower_bound` へ適用し
   候補下限値を得る
3. `docs/perf/performance-floor-decision.md` へ §3・§8〜§10 と同形式の新規追補（節）を追加し、
   確定判断・根拠実測・限定条件を記録する（§3 の確定表そのものは書き換えない。追補方式は §8〜§10
   の先例を踏襲する）
4. ユーザー承認（PR レビュー・マージ）を経る
5. `docs/spec/04-requirements.md` REQ-8 節への反映は spec リポジトリ
   （Fandhe-AI/fandhe-ai-spec）側での対応をユーザーへ提案する（本リポの `docs/spec/`
   submodule は編集しない）

## 7. 関連参照

- `docs/perf/gemm-optimization-baseline.md`（#481。REQ-8 GEMM 5 行の対象カーネル・実機・PyTorch 版・出典の突合確定。Metal 2 系列の対応関係・CPU 基準実機の判断を含む）
- `docs/perf/performance-floor-decision.md`（#158 確定記録 §3・#386 §8〈Metal f16 初期リリース確定〉・#393 §9〈CUDA f32/f16 最適化後確定〉・#577 §10〈GEMM 性能改善ツリー Phase F・Optimized 段 5 行の再確定〉。本ドキュメントの §2・§6 の入力）
- `docs/perf/transformer-workload-measurement.md`（#155）
- `docs/perf/metal-f16-vs-mps-f16.md`（#156・#380・#383）
- `docs/perf/cuda-floor-remeasurement.md`（#157）
- `docs/perf/cpu-gemm-optimized-remeasurement.md`（#574・PR #713。CPU f32 最適化後段の Phase F 実測入力）
- `docs/perf/cuda-optimized-remeasurement.md`（#571・PR #725。CUDA f32/f16 最適化後段の Phase F 実測入力）
- `docs/perf/metal-floor-remeasurement.md`（#572・PR #725。Metal f32/f16 最適化後段の Phase F 実測入力）
- `crates/bench-harness/src/threshold.rs`（REQ-8 下限表のデータ化・自動合否判定）
- `crates/bench-harness/src/rounding.rs`（丸め規則の公開 API。`floor_lower_bound`）
- `docs/spec/04-requirements.md` REQ-8（正本。段階的下限の受け入れ基準）
- `scripts/bench/framework-compare/results/summary.md`（candle・burn との横並び計測。§8 参照）

## 8. フレームワーク横並び（candle 比）の達成状況（イシュー #1052）

- 本節は `scripts/bench/framework-compare`（イシュー #915／#1050／#1051）による candle・burn との
  横並び計測に関する追補であり、**§2 の PyTorch 比段階的下限表・§3 丸め規則は変更しない**
  （REQ-8 の下限自体はここでは動かない。本節は別ツール・別比較対象〈candle〉での達成状況の記録）
- fandhe-ai 0.5.0（crates.io 公開版・2026-08-31 公開）での実機再計測（DGX Spark GB10・Apple
  M4 Max。計測日 2026-09-01）に対し `python3 summarize.py --target candle` を実行した結果、
  **達成 1 件・未達 23 件・判定不能 2 件**（終了コード 3）。出典・詳細な (task, device, size)
  ごとの判定表は `scripts/bench/framework-compare/results/summary.md` 環境 8（DGX Spark GB10）・
  環境 9（Apple M4 Max）節「〜の目標達成ゲート」および「目標達成ゲート総括」節、生データは
  `scripts/bench/framework-compare/results/raw/results-dgx-0.5.0.jsonl`・
  `results-m4max-0.5.0.jsonl` を参照
- **収録範囲の限定**: `v0.5.0` タグ以降に main へ入った改善（#1108 Metal 選択テーブル・#1110
  Metal SGD バッチング・#1111 CUDA variant selection 修正）はこの達成状況に反映されていない。
  CUDA GEMM 改善トラッカー #1031・Metal GEMM 改善トラッカー #1037 も本計測時点で open のまま
  （未着手）であり、下記の未達は「0.5.0 時点・上記改善適用前」の現在地である
- 未達・判定不能項目の追跡状況（詳細は `results/summary.md`「目標達成ゲート総括」節）:
  - CUDA GEMM（N=256〜4096）: 既存トラッカー #1031（open）
  - Metal GEMM（N=256〜4096）: 既存トラッカー #1037（open）
  - CPU GEMM・学習（train）・推論（infer）の全項目: 既存の個別トラッカーなし。次回 crates.io
    公開後の再計測とあわせた Issue 化の要否はユーザー判断（本 PR では Issue 操作を行わず、
    未追跡のまま `out-of-scope-tracking.md` に従いスコープ外として引き継ぐ）


### 8.1 v0.6.0 追補

- fandhe-ai 0.6.0（crates.io 公開版・2026-09-02 公開）での実機再計測（DGX Spark GB10・Apple
  M4 Max。計測日 2026-09-02）に対し `python3 summarize.py --target candle` を実行した結果、
  **達成 3 件・未達 21 件・判定不能 2 件**（終了コード 3）。0.5.0 時点（達成 1 件・未達 23 件・
  判定不能 2 件）比では DGX Spark の CPU GEMM N=256・M4 Max の CPU GEMM N=512 が新規達成に
  転じた。出典・詳細な (task, device, size) ごとの判定表は
  `scripts/bench/framework-compare/results/summary.md` 環境 10（DGX Spark GB10）・環境 11
  （Apple M4 Max）節「〜の目標達成ゲート」および「目標達成ゲート総括」節、生データは
  `scripts/bench/framework-compare/results/raw/results-dgx-0.6.0.jsonl`・
  `results-m4max-0.6.0.jsonl` を参照
- **収録範囲の限定**: `v0.5.0` タグ以降に main へ入っていた改善（#1108 Metal 選択テーブル・
  #1110 Metal SGD バッチング・#1111 CUDA variant selection 修正）は今回初めて反映されている。
  CUDA GEMM 改善トラッカー #1031・Metal GEMM 改善トラッカー #1037・学習/推論 candle 比未達
  トラッカー #1118 は本計測時点で open のまま（未着手）であり、下記の未達はそれらの改善適用前の
  現在地である
- 未達・判定不能項目の追跡状況（詳細は `results/summary.md`「目標達成ゲート総括」節）:
  - CUDA GEMM（N=256〜4096）: 既存トラッカー #1031（open）
  - Metal GEMM（N=256〜4096）: 既存トラッカー #1037（open）
  - 学習（train）・推論（infer）の全項目: 既存トラッカー #1118（open）
  - CPU GEMM（N=512〈DGX のみ未達〉・N=1024・N=2048）: 既存の個別トラッカーなし。次回再計測と
    あわせた Issue 化の要否はユーザー判断（本 PR では Issue 操作を行わず、未追跡のまま
    `out-of-scope-tracking.md` に従いスコープ外として引き継ぐ）
