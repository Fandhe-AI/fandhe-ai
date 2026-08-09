# カーネル融合の適用範囲・限界（TASK-12.2b）

> 役割・参照元: 本文書は REQ-12（`docs/spec/04-requirements.md:246`）の
> v2 読み替え後タスク分解 TASK-12.2（`docs/spec/05-tasks.md:377`。
> 「TASK-12.1 の実装に対し、elementwise 連鎖・fan-out・transpose 混在
> ワークロードでの融合効果を実測する。matmul・softmax を含む複合
> ワークロードでは融合の効果を前提とした性能目標を設定しないこと
> （REQ-8 の複合ワークロード系目標との整合）を明記する」）の文書化部分
> （TASK-12.2b・本イシュー #168）の成果物である。親イシューは #166
> （TASK-12.2）、実測部分の兄弟イシューは #167（TASK-12.2a）。
>
> - **設計正本**: `docs/fusion-graph-design.md`（TASK-12.1a・#161。融合
>   IR・実体化境界・`BackendOps` 契約接続の確定設計）
> - **実測正本**: `docs/perf/cpu-elementwise-fusion-effect.md`
>   （TASK-12.2a・#167。elementwise 連鎖・fan-out・transpose 混在の実測）、
>   および `docs/perf/cpu-gemm-epilogue-fusion.md`（TASK-12.1f・#203。
>   GEMM epilogue 融合の実測）
> - **本文書の更新時点の状態**: TASK-12.2a（#167）で elementwise 連鎖
>   融合の実測を実施済み（実測正本: `docs/perf/
>   cpu-elementwise-fusion-effect.md`）。#167 の実装時点で
>   `CpuBackendOps::run_fused` が融合カーネルへ未結線（TASK-12.1 系列
>   〈#163・#164〉が結線を相手のスコープと委ね合っていたギャップ）で
>   あることが判明したため、同イシューの前提ステップとして結線を実施
>   したうえで実測している（詳細は同記録 §0）。

## 1. 判断サマリ

- (a) 融合の適用範囲は **elementwise 演算連鎖（4〜6 段程度。`add`／
  `mul`／`relu`／`exp`／`tanh` の混在を含む）** と **GEMM epilogue
  （bias・activation）** の 2 系統に限る（`docs/fusion-graph-design.md`
  §1・§2.1、TASK-12.1f・#203）。
- (b) **reduction エピローグ（`.sum()`／`.max()`）・matmul／softmax を
  挟む複合連鎖は初期スコープ外**とする（`docs/fusion-graph-design.md`
  §3.2 (a)(b)、PoC-9 実測が根拠）。
- (c) **matmul・softmax を含む複合ワークロード（attention 系連鎖等）
  では融合効果を前提とした性能目標を設定しない**（REQ-12 受け入れ
  基準 `docs/spec/04-requirements.md:255`。REQ-8 整合。§5 参照）。
- (d) **利用者向け融合制御 API は提供しない**（REQ-12 受け入れ基準
  `docs/spec/04-requirements.md:252`）。融合は `facade` クレートの
  composition root（`Device` → 具体 `BackendOps` の結線）を経由した
  既定バックエンド経由でのみ到達し、任意 `BackendOps` を注入できる
  公開 API は設けない（`docs/fusion-graph-design.md` §1「既定バック
  エンド供給は `facade` クレートの composition root が担う」・
  `docs/spec/05-tasks.md:316` TASK-9.3）。
- (e) transpose を挟む連鎖も初期スコープでは**融合しない（非融合
  フォールバックへ倒す）**（`docs/fusion-graph-design.md` §1・§2.3。
  §3 参照）。

## 2. 適用範囲

### 2.1 elementwise 連鎖融合

- **対象**: `add`／`mul`（二項）・`relu`／`exp`／`tanh`（単項）の
  elementwise 5 演算からなる 4〜6 段程度の連鎖。fan-out（1 つの中間
  テンソルを複数ノードから消費する分岐）も対象に含む（`docs/
  fusion-graph-design.md` §2.4。PoC-9 の `ew_fanout` 実測が根拠）。
- **成立条件**（`docs/fusion-graph-design.md` §2〜§3 の実装仕様を要約。
  実装ファイルは `crates/tensor-core/src/fusion/{graph,detect,mod}.rs`）:
  - 連鎖が単一の連結成分（`FusionOp` の DAG）として elementwise のみで
    閉じていること。`gemm`／`sum`／`max` は融合境界ノードであり連鎖を
    分断する（§3.2 (a)(b)）。
  - 各ノードの `NodeMeta.contiguous == true`（transpose／broadcast view
    を含まない）こと（§2.3・§2.1 の非融合フォールバック判定）。
  - 連鎖長が 4〜6 段の上限に収まること（§3.2 (d)。**設計上の想定であり、
    `crates/autodiff/src/tape.rs::build_lazy_plan` では未実装**。上限
    適用の実装は #404 で追跡する。詳細は §3 表 5 行目を参照）。
  - `dtype` は `DType::F32` 固定（`BackendOps` の現行スコープに合わせる。
    §2.1・§2.3）。f16 融合は §4 の未決事項。
- **利用者の記述形との対応**: 独立した複数回の公開 `Var` 呼び出し
  （`a.add(&b)?.relu().exp().tanh()` のように演算をまたぐ記述）が融合対象
  になりうる（`docs/fusion-graph-design.md` §1「演算跨ぎの遅延・二項
  elementwise 演算の遅延化」）。融合の成否は `Tape::new(ops)`（§1）へ渡
  された `ops` の具体的な実装によらず同一方針であり、利用者が明示的に
  融合を切り替える経路は存在しない（判断サマリ (d)）。
- **実測（CPU・TASK-12.2a・#167。出典: `docs/perf/
  cpu-elementwise-fusion-effect.md`「実測結果」節）**: `ew4`（4 段連鎖）・
  `ew6`（6 段連鎖）・`ew_fanout`（fan-out）のいずれも非融合（per-op
  フォールバック）比 **1.43〜1.53 倍**の改善（20 回中央値。全パターンで
  融合が非融合を上回る）。PoC-9（v1・Metal 実機）の定性的傾向（連鎖・
  fan-out で融合が有意に効く）と一致する（絶対値は実行系の違いにより
  異なるため直接比較しない）。数値一致は REQ-2 統一複合判定で確認済み。

### 2.2 GEMM epilogue 融合

- **対象**: `matmul` の直後に続く bias 加算・activation（Linear+bias+
  ReLU 相当）。`BackendOps::gemm_bias_act`（`CpuBackendOps` 実装）が
  行パネル並列の GEMM 完了直後・同一 `rayon` タスク内で bias 加算・
  activation を適用し、中間 `Tensor` 割当を出力 1 個に抑える（`docs/
  perf/cpu-gemm-epilogue-fusion.md`）。
- **実測（CPU・TASK-12.1f・#203。出典: `docs/perf/
  cpu-gemm-epilogue-fusion.md`「実測結果」節）**: Linear+bias+ReLU 相当
  形状・正方形状の計 5 形状で、非融合（`BackendOps::gemm_bias_act` の
  デフォルト実装＝ `gemm` → `add` → `relu` の逐次呼び出し）比 **1.46〜
  2.56 倍**の改善（5 回中央値。全形状で融合が非融合を上回る）。数値一致
  は bit 完全一致（epilogue が要素ごとに独立な演算で演算順序に依存しない
  ため。同文書「数値一致」節）。
- **GPU（CUDA NVRTC・Metal MSL）**: 本文書執筆時点で GEMM カーネルのみ
  実装済みで elementwise（`add`／`relu`）が未実装のため、`bias`／`act`
  指定時は透過的にデフォルト実装（非融合合成）へフォールバックする
  （`Unsupported` 経由。`docs/perf/cpu-gemm-epilogue-fusion.md`「スコープ
  外」節）。GPU カーネル内 epilogue 融合の実装自体は実機検証前提のため
  未着手（`out-of-scope-tracking.md` に従いユーザー承認取得後に別イシュー
  で追跡する）。

## 3. 限界

融合が働かない、または連鎖が分断されるパターンを列挙する。いずれも
`docs/fusion-graph-design.md` §1・§3.2 の実体化境界の設計根拠であり、
v1 PoC-9（Metal 実機、Burn/CubeCL 前提）の実測知見に基づく判断である
（下記「v1 PoC-9 実測の位置づけ」参照）。

| # | パターン | 挙動 | 根拠 |
|---|---------|------|------|
| 1 | reduction エピローグ（`.sum()`／`.max()`） | 融合セグメントが `sum`／`max` ノードで打ち切られる（連鎖部分のみ融合、reduction 自体は融合対象外） | `docs/fusion-graph-design.md` §3.2 (a)。PoC-9 `ew_reduce`（連鎖部分は `ElemwiseFuse` に融合されるが `.sum()` は別カーネル群） |
| 2 | matmul を挟む連鎖 | `gemm` ノードで融合セグメントが分断される（GEMM 本体は §2.2 の epilogue 融合対象だが、matmul をまたいだ elementwise 連鎖全体の融合ではない） | `docs/fusion-graph-design.md` §3.2 (b)。PoC-9 `ew_matmul_ew`（matmul 前後は個別カーネルのまま） |
| 3 | softmax 単体・attention 系連鎖 | 融合機構の対象外（softmax は組込み複合演算として別実装であり、本融合 IR の `FusionOp` enum に含まれない） | REQ-12 受け入れ基準（`docs/spec/04-requirements.md:255`）。PoC-9 `softmax`・`attention_chain`（融合の影響をほぼ受けない、または部分的） |
| 4 | transpose 混在連鎖 | `NodeMeta.contiguous == false` を検出した時点で融合セグメントを打ち切り、非融合フォールバックへ倒す（初期スコープでは transpose を融合対象に含めない）。**実測（#167）**: 融合を試みてから `Unsupported` で失敗し per-op フォールバックへ再実行する 2 段の経路を踏むため、最初から per-op 経路のみを通る非融合条件よりわずかに**遅くなる**という結果が得られた（速度比 0.707x。当初予測の「速度比 ≈ 1.0」との差分は、フォールバック試行コストが主要因である可能性を示唆するが、共有ホスト環境のノイズの影響を否定できない単一計測であり断定はしない。計測条件・Q1〜Q3 幅の詳細は `docs/perf/cpu-elementwise-fusion-effect.md` §5 を参照） | `docs/fusion-graph-design.md` §1「transpose を挟む連鎖は融合しない」・§2.3。実測: `docs/perf/cpu-elementwise-fusion-effect.md` §5 |
| 5 | 連鎖長が 4〜6 段の上限を超える | **設計上は**上限到達時点でその場の演算を実体化してから連鎖を再開する想定だが、`crates/autodiff/src/tape.rs::build_lazy_plan`（ライブで実行される遅延評価経路）ではこの上限適用ロジックが未実装（scope reduction。#164 実装ノート）。`tensor-core::fusion::detect::MAX_FUSED_CHAIN_LEN` は `detect_fusion`／`FusionGraph` 経由でのみ参照され、autodiff の遅延評価パスからは呼ばれない dead code（`#[allow(dead_code)]` 付き）。現状は連鎖長に上限なく単一 `FusionPlan` へ融合されるため、`build_lazy_plan` 内 `node_index` の線形探索により連鎖長 n に対し構築コストが O(n²) になる。実装は #404 で追跡する | `docs/fusion-graph-design.md` §3.2 (d)。実装ギャップの検出は #167 ブランチの Review、追跡は #404 |
| 6 | backward（VJP）の勾配式計算 | `grad.rs::vjp` が計算する勾配式そのもの（例: `tanh` の VJP `grad * (1 - y * y)`）は融合 IR に記録されず、常に具体 `Tensor` として直接計算される。融合されるのは forward が記録した elementwise 遅延グラフを `Tape::backward` が読み出す箇所のみ | `docs/fusion-graph-design.md` §3.3「backward（VJP）は融合対象外」 |

### v1 PoC-9 実測の位置づけ

`docs/spec/03-poc/poc-9-kernel-fusion/README.md` は Burn/CubeCL（`burn-
wgpu` の `fusion` feature）を前提とした v1 実測であり、**v2 自作融合
機構（本文書が扱う `tensor-core::fusion` モジュール）の保証値ではない**。
elementwise 連鎖・fan-out・transpose 混在パターンでの適用範囲判断
（表中 1〜5）は v1 実測が示した構造的な傾向（PoC-9「実施内容」節）を
参考に v2 の初期スコープを決定した根拠として引用しているが、v2 自作
機構での性能改善比・分断挙動は #167（TASK-12.2a）の実測を正とする。
v1 実測の代表数値（Metal 実機、`code/src/bin/elementwise_bench.rs`。
出典: 同 README「実行時間の計測」節）:

| パターン | 速度比（融合あり/なし） | 判定 |
|---|---|---|
| `ew4`（elementwise 4 段連鎖） | 2.25 倍 | 融合が有意に効く |
| `ew6`（elementwise 6 段連鎖） | 2.61 倍 | 融合が有意に効く |
| `ew_fanout`（fan-out を含む連鎖） | 3.19 倍 | 融合が有意に効く |
| `ew_reshape`（transpose 混在。v1 は融合有効時にメタデータ変換として取り込む） | 13.89 倍 | 融合が有意に効く（v1 実装依存。§3 表 4 参照） |
| `ew_matmul_ew`（matmul を挟む連鎖） | 0.82 倍（融合なしがわずかに高速） | 有意差なし |
| `softmax` 単体 | 0.94 倍（融合なしがわずかに高速） | 有意差なし |
| `attention_chain` | 0.98 倍（ほぼ差なし） | 有意差なし |

`ew_reshape` の 13.89 倍差は v1 の融合エンジン内部実装（ストライド付き
ビューを融合セグメント内で扱う機構）に依存した挙動であり、v2 自作融合
IR（`docs/fusion-graph-design.md` §2）は現時点でストライド付きビューを
表現・伝播する仕組みを持たないため、**v2 初期スコープはこの性能水準を
達成しない**という受け入れコストを明示的に記録する（`docs/
fusion-graph-design.md` §1・§6.2「transpose 混在連鎖のメタデータ融合」）。

## 4. 性能目標との関係（REQ-8 整合）

- REQ-12 受け入れ基準は「matmul・softmax を含む複合ワークロード
  （attention 系連鎖等）では融合の効果を前提とした性能目標を設定しない
  こと（PoC-9 で有意差なしと確認済み。REQ-8 の複合ワークロード系目標と
  整合させること）」と定める（`docs/spec/04-requirements.md:255`）。
- `docs/performance-targets.md` はこの整合を実装済みである: Transformer
  複合ワークロード（attention/softmax/LayerNorm を含む複合演算）の行
  （`docs/performance-targets.md:30`）は初期リリース・最適化後のいずれも
  **「下限を設定しない」**（融合の有無に依存させない）としており、
  QEMU 仮想 CPU での非実機参考値（約 6.1%）を性能目標の根拠に用いない
  ことを明記する。
- 融合を性能下限の前提とするのは §2 の elementwise 連鎖・GEMM epilogue
  という限定範囲のみであり、複合ワークロードの性能下限は
  `docs/performance-targets.md` を正とする（本文書はその根拠の一部を
  提供するのみで、下限値そのものを重複管理しない）。

## 5. 実測結果の参照

| 実測対象 | 記録 | 状態 |
|---|---|---|
| GEMM epilogue 融合（CPU、TASK-12.1f・#203） | `docs/perf/cpu-gemm-epilogue-fusion.md` | 実測済み（1.46〜2.56 倍、5 形状） |
| elementwise 連鎖・fan-out・transpose 混在の融合効果（v2、TASK-12.2a・#167） | `docs/perf/cpu-elementwise-fusion-effect.md` | 実測済み（連鎖・fan-out: 1.43〜1.53 倍改善。transpose 混在: 融合側が悪化〈0.707x〉） |
| v1 参考値（Burn/CubeCL 前提、Metal 実機） | `docs/spec/03-poc/poc-9-kernel-fusion/README.md` | 実測済み（v1 保証値であり v2 の保証値ではない。§3 参照） |

## 6. 将来拡張・スコープ外

- **reduction を含めた手動完全融合**: REQ-12 受け入れ基準は「性能クリ
  ティカルな箇所では、CubeCL カスタムカーネルによる手動融合（reduction
  を含めた完全融合）を組込み演算として提供する選択肢を将来検討課題と
  すること」と記載する（`docs/spec/04-requirements.md:254`。v1 CubeCL
  前提の文言だが、自作カーネルでの reduction epilogue 融合という論点
  自体は引き継ぐ）。本文書は §3 表 1 で reduction を実体化境界として
  扱う（初期スコープ外）に留め、将来対応は `docs/fusion-graph-
  design.md` §6.2「reduction エピローグの手動融合」への記録に留める。
- **backward（VJP）融合**: §3 表 6 のとおり初期スコープ外・未確定。
  VJP 計算式専用の融合グラフ構築・`FusedOpKind` への演算表現拡張
  （`sub` 等）を要する（`docs/fusion-graph-design.md` §6.2「backward
  （VJP）融合」）。
- **GPU バックエンドへの融合展開**: 現状 CUDA／Metal は GEMM カーネル
  のみ実装済みで elementwise 未実装のため、§2.2 の epilogue 融合を含め
  GPU 側の融合はデフォルト実装（非融合合成）へのフォールバックに留まる
  （§2.2）。実機（DGX Spark GB10・Metal 実機）での検証を要するため、
  ユーザー承認を得たうえで別イシューとして追跡する
  （`.claude/rules/out-of-scope-tracking.md`）。
- **f16 対応**: `BackendOps`・`NodeMeta.dtype` とも現状 f32 固定であり、
  f16 融合カーネルの型設計は未着手（`docs/fusion-graph-design.md`
  §6.2「f16 対応」）。
- **transpose 混在連鎖のメタデータ融合**: §3「v1 PoC-9 実測の位置づけ」
  のとおり、ストライド付きビューを融合 IR 内で表現・伝播する設計
  ができれば transpose を融合対象へ含められる可能性がある（`docs/
  fusion-graph-design.md` §6.2「transpose 混在連鎖のメタデータ融合」）。
- **REQ-12 自体の v2 書き直し**: REQ-12 の受け入れ基準文言は `burn-
  wgpu` の `fusion` feature・`CUBECL_DEBUG_LOG` を前提としたまま（v2
  全面改定未実施）である。本文書は TASK-12.1〜12.2 の読み替え（自作
  elementwise 融合機構）に基づいて記述しているが、REQ-12 自体の文言
  更新は正本リポジトリ（`Fandhe-AI/rust-ai-library-spec`）側の課題で
  あり、本文書のスコープ外である（`docs/spec/` は編集しない）。
