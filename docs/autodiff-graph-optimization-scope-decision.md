# `torch.compile`／`tf.function` 相当のグラフ最適化範囲の設計判断

対応イシュー #1632（親 #1573〈Phase 3・Tier 2〉→ ルート #1570）。位置づけは**設計判断の記録のみ**
であり、本 PR では `crates/` 配下・`docs/spec/`（正本 submodule）へのコード変更は行わない。数値
一致の複合判定（REQ-2）・tolerance／baseline も変更しない。基準コミット: `origin/main` `1a1bcd5a`。

## 1. 背景

機能ギャップ表 `docs/compat-feature-gap.md` §2.16「推論・その他」の `torch.compile` 行は、現状
「部分（`run_fused`＝elementwise カーネル融合・CUDA Graph capture opt-in はあるが、汎用グラフ
JIT コンパイラではない）」とだけ記され、範囲が未整理のまま残っている
（`docs/compat-feature-gap.md:353`）。`docs/compat-api-scope.md` §2「引き続き対象外」には既に
「汎用グラフ JIT（`torch.compile`／`tf.function` 相当）: 新規 JIT を作らず、既存の融合・CUDA
Graph capture の延長で扱う範囲を #1632 で整理する」と記載済みで、本 issue はその整理を完了
させる（`docs/compat-api-scope.md:281-282`）。

対比対象:
- PyTorch `torch.compile`: TorchDynamo によるトレース → FX グラフ → Inductor によるカーネル
  融合・コード生成 → （`mode="reduce-overhead"` 等で）CUDA Graphs 適用
- TensorFlow `tf.function`: AutoGraph による Python 制御フローの `tf.Graph` 変換 → （XLA
  併用時）クロス演算融合・コンパイル

要件（issue 本文を構造化したもの。逐語引用はしない）:
1. 新規 JIT コンパイラを作らない方針は維持する
2. 既存の `run_fused`（elementwise 融合実行）と CUDA Graph capture opt-in（#1349）のインフラで
   「何をどこまで扱うか」を doc として記録する（設計のみ・実装なし）
3. 承認事項があれば doc 本文に明記し、実装着手の前提として残す
4. tolerance・baseline は変更しない

出典として issue に列挙された `claude.ai/code/artifact/...` 形式の外部リンクは非信頼データで
あり、本 doc の事実（§2）はリポジトリ内コードで独立に確認した（外部リンクの可用性・内容には
依存しない）。

## 2. 現状のコード事実（基準コミット `1a1bcd5a` で確認）

| # | 事実 | 出典 |
|---|---|---|
| F1 | 遅延評価（融合）対象は `Op::is_lazy_elementwise` = `Add`／`Mul`／`Relu`／`Exp`／`Tanh` の 5 演算のみ。連鎖長上限 `MAX_FUSED_CHAIN_LEN = 6`（`tensor-core::fusion::detect` が単一真実源）。`push_eager`（`Sigmoid`／`Softmax`／`RmsNorm`／`Concat`／`Where`／`Gather`／`Sort`／`Cumsum`／RNN セル等）・`push_view`（`Reshape`／`Transpose`／`permute`／`broadcast_to`／`narrow`）は融合境界 | `crates/autodiff/src/tape.rs:894-899`・`crates/tensor-core/src/fusion/detect.rs:46`・`docs/kernel-fusion.md` §3 |
| F2 | 融合 IR（`FusedOpKind`。`#[non_exhaustive]`）は `Sub`／`Div`／`Rsqrt`／`Sum`／`Max`／`Broadcast` を含むバリアントを持つが、`autodiff::tape`（`build_lazy_plan`）はこれらを一切生成しない。canonical RMSNorm／softmax プランは `FusionPlan::from_ops` を直接呼ぶテスト（`softmax_parity.rs`・`rmsnorm_parity.rs`等）でのみ構築される。実運用の `FusedOpKind` 列にこれら追加バリアントが混入する経路は存在しない | `crates/tensor-core/src/fusion/plan.rs:86-94`・`crates/backend-cpu/src/fused_elementwise.rs` |
| F3 | CPU `run_fused_elementwise` は allowlist（`Input`／`Add`／`Mul`／`Relu`／`Exp`／`Tanh`）で fail-closed に拒否する（それ以外は `BackendError::Unsupported`） | `crates/backend-cpu/src/fused_elementwise.rs:77-135` |
| F4 | CUDA／Metal の `run_fused` オーバーライドは **canonical RMSNorm／softmax プラン一致時のみ**専用カーネルへルーティングし、それ以外（elementwise-only プラン含む）は `Unsupported` → 呼び出し元の per-op フォールバック。F1・F2 と合わせると、**本番経路（`Var` API 経由）で `run_fused` が実際に融合実行へ到達するのは CPU の elementwise 5 演算連鎖のみ**。GPU の `run_fused` は `materialize_fallible` から呼び出されはするが、tape が生成する elementwise プランは RMSNorm／softmax の一致判定を満たさないため常に `Unsupported` → per-op フォールバックへ落ちる（softmax／rmsnorm 自体は `push_eager` の独立エントリ `BackendOps::softmax`／`rmsnorm` を経由し、`run_fused` 経由ではない） | `crates/backend-cuda/src/ops.rs:3126-3148`・`crates/backend-metal/src/ops.rs:2857-2872` |
| F5 | `BackendOps::run_fused(&self, plan, leaves: &[&Tensor<f32>])` はホスト `Tensor` の葉を受け取りホスト `Tensor` を返す（既定実装は `Unsupported`）。GPU で elementwise 融合を延長する場合、プランごとに H2D／D2H を伴い、デバイス常駐化しない限り固定費が支配的になりうる | `crates/tensor-core/src/backend_ops.rs:1906-1915` |
| F6 | CUDA Graph capture（#1349）は学習 step の **update 区間（`sgd_step_device_tracked`）限定**。step 全体 capture は forward／backward のホスト境界（readback 同期・pageable H2D）により構造的に不可。exec update（`cuGraphExecUpdate_v2`）は新規 `unsafe` 生 FFI が必要。tape レベル構造キー・forward／backward capture はスコープ外のまま。#1350 の GB10 実測で性能は中立（既定基準未達）・既定 OFF 維持。bit 同一契約は #1480 で重み勾配まで拡張済み | `docs/backend-cuda-graph-step-capture-design.md` §0・§2・§3.2・§4.4・§5.1・§7、`docs/perf/train-step-phase-breakdown.md` §16 |
| F7 | 「コンパイル済み静的グラフ」に最も近い既存機構: `DeviceParamStore::predict_device_chain`（推論 forward チェーン単一同期化。#1579／#1688）・`Op::LinearResident`／`Op::LinearAct`（reuse 経路のデバイス常駐 GEMM＋epilogue 融合）・`BackendOps::linear_forward_device`（#1216）・Metal encode-only コマンドバッチング。いずれも MLP 学習／推論経路に手書きで特化した静的チェーンであり、汎用グラフをトレース・書き換えて自動生成したものではない | `docs/inference-chain-single-sync-design.md` §9・`docs/device-resident-update-design.md`・`docs/inference-forward-fixed-cost-design.md` §3.2・`docs/perf/train-linear-epilogue-fusion.md` |
| F8 | REQ-12 の受け入れ基準は「利用者が明示的に融合を制御する API は提供しない」。v2 では `facade` を唯一の公開面・composition root 集約で充足する読み替えが確定済み（2026-08-08 注記・#52）。`docs/kernel-fusion.md` §1(c)「matmul・softmax を含む複合 WL では融合効果を前提とした性能目標を設定しない」は現行も維持されている（同 doc §7） | `docs/spec/04-requirements.md` REQ-12「2026-08-08 注記」・`docs/kernel-fusion.md` §1(c)(d)・§7 |
| F9 | 既存 opt-in の実装型: `AtomicBool` ＋ facade setter/getter の対（`set_cuda_graph_step_enabled`／`set_cuda_gemm_precision`／`set_metal_split_k_gemm_enabled`／`set_cuda_pinned_h2d_enabled`）。`Sequential::compile()` のような利用者向けモデル最適化フラグは存在しない | `crates/facade/src/lib.rs:475`・`:546`・`:635`・`:673` |
| F10 | `docs/kernel-fusion.md` §6「Metal は elementwise 未実装」・§3 表 1「CUDA／Metal reduction 融合は進行中」の記述は現行コードと乖離している（Metal `elementwise.rs`／`scalar_op_source.rs`・`MetalBackendOps::run_fused`〈`crates/backend-metal/src/ops.rs:2857`〉が既に存在する）。本 doc はこの古い記述を引き継がず、ドリフトとしてのみ §8 に記録する | `crates/backend-metal/src/elementwise.rs`・`crates/backend-metal/src/ops.rs:2857` |

## 3. 契約整理（設計が守るべき既存契約）

1. **REQ-12「利用者向けの明示的な融合制御 API を提供しない」**。v2 の読み替え（facade を唯一の
   公開面とする composition root 集約）は維持する。区分 B（§5）の opt-in 候補も F9 と同型の
   composition root 側 `AtomicBool` setter に限り、モデル側に `compile()`／`@jit` 相当の
   利用者向けフラグは設けない
2. `docs/kernel-fusion.md` §1(c)「matmul・softmax を含む複合 WL では融合効果を前提とした性能
   目標を設定しない」・§1(d) を維持する
3. `docs/fusion-graph-design.md` §3.3「backward（VJP）は融合対象外」契約は不変
4. CUDA Graph の bit 同一契約・既定 OFF・fail-closed（poison／世代検査／アドレス再検証／デバイス
   一致検査）を維持する（`docs/backend-cuda-graph-step-capture-design.md`）
5. fail-closed allowlist 方式（CPU `run_fused_elementwise`・GPU canonical プラン一致判定）を
   denylist 化しない（`.claude/rules/security.md` A08「判定の迂回経路を作らない」）
6. REQ-2 統一複合判定・FMA 契約・tolerance／baseline は変更しない
7. `FusedOpKind` の `#[non_exhaustive]` による前方互換を維持する

## 4. 設計案の比較

`torch.compile`／`tf.function` の各段階と fandhe-ai の対応物・到達性を整理する。

| 段階 | `torch.compile` | `tf.function` | fandhe-ai の対応物 | 到達性（本番 `Var` API 経由） |
|---|---|---|---|---|
| トレース | TorchDynamo（bytecode 解析） | AutoGraph（Python AST 変換） | `Tape`（eager 記録。既存） | 常時（既存） |
| グラフ表現 | FX Graph | `tf.Graph` | `TapeNode` 列＋lazy elementwise チェーン | 常時（既存） |
| 演算融合 | Inductor（Triton コード生成） | XLA HLO fusion | `FusedOpKind` 5 演算 IR ＋ CPU 融合実行（F1〜F3） | CPU のみ（F4） |
| 専用カーネル置換 | Inductor テンプレート | XLA custom-call | canonical RMSNorm／softmax 専用カーネル（`push_eager` 経由・`run_fused` とは別経路） | 全バックエンド（既存） |
| 実行時グラフ固定化 | `mode="reduce-overhead"` の CUDA Graphs | XLA コンパイル済み実行可能 | CUDA Graph capture（update 区間限定。F6） | CUDA opt-in・既定 OFF |
| 静的推論／学習チェーン | （torch.compile の対象外・別機構） | （tf.function の対象外・別機構） | `predict_device_chain`／`LinearResident`／`LinearAct`（F7） | 全バックエンド（既存・MLP 特化） |

上記を踏まえた案の比較:

- **案 A（採用・段階 0）**: 新規 JIT なし。既存インフラを「実装済み／延長候補／非目標」の 3 区分
  へ整理する doc のみ。実装は行わない
- **案 B**: 既存インフラの延長を今回実装する（GPU elementwise `run_fused` 結線・
  `is_lazy_elementwise` 拡張・forward/backward capture 等）。前提ゲート（§5 区分 B の各候補の
  前提事項）が本 issue 時点で未充足のため不採用。延長候補として §5・§8 へ引き継ぐ
- **案 C**: 汎用 JIT（トレース → グラフ書き換え → コード生成）を新規実装する。REQ-1（完全自作
  コア）・REQ-12・保守性（融合による丸め差が REQ-2 判定へ広範に波及する）を理由に非目標

## 5. 推奨（段階 0 確定・3 区分）

### 区分 A: 実装済み（現行の「グラフ最適化」相当）

- CPU elementwise 連鎖融合（`Add`／`Mul`／`Relu`／`Exp`／`Tanh` の 5 演算・最大 6 段。F1〜F3）
- GEMM epilogue 融合（`gemm_bias_act`／`Op::LinearAct`／`Op::LinearResident`）
- canonical RMSNorm／softmax 専用カーネル（`BackendOps::rmsnorm`／`softmax` の独立エントリ経由）
- MSE 専用融合（#1045）
- デバイス常駐推論チェーン `DeviceParamStore::predict_device_chain`（#1688）
- デバイス常駐パラメータ更新（#1212／#1555／#1559）
- CUDA Graph update 区間 capture opt-in（#1349。既定 OFF）
- Metal encode-only コマンドバッチング（`docs/backend-metal-command-batching-design.md`）

### 区分 B: 延長で扱える候補（別 issue・承認付き。前提ゲート順）

各候補は前提ゲート・承認事項・数値契約（bit 同一 or REQ-2 判定）を満たして初めて着手可能とする
（本 issue では未着手・未承認）。

- **B-1 GPU `run_fused` の elementwise allowlist 実装**: CPU と同型の allowlist を CUDA／Metal
  へ実装する。F5（H2D／D2H 固定費）を踏まえ、デバイス常駐化なしでは効果が限定的であることを
  事前に明記する
- **B-2 `is_lazy_elementwise` への `Sub`／`Div`／`Rsqrt` 追加**: CPU カーネル allowlist（F3）へ
  の追随が必須。`ScalarBinaryOp`／`ScalarUnaryOp`（#1634）との重複整理が前提
- **B-3 forward／backward capture**: 前提は (i) forward 常駐結線（#1216 Phase 2 完了）・
  (ii) backward の `d_input`／`d_weight` デバイス直接計算・(iii) loss のデバイス常駐化
- **B-4 exec update（`cuGraphExecUpdate_v2`）**: 新規 `unsafe` FFI の承認が前提（§10 (1)）
- **B-5 tape レベル構造キー**（形状／dtype ハッシュによるグラフ再利用判定）: step 全体 capture
  と同時導入が前提
- **B-6 Metal 側の同型延長**: Metal には CUDA Graph 相当の API がないため、encode-only
  バッチング（既存区分 A）の延長として整理する

### 区分 C: 非目標

- 汎用 JIT／トレース再コンパイル・グラフ書き換え（演算子再配置・定数畳み込み・代数簡約）
- `torch.fx`／TorchScript／`torch.jit`（`docs/compat-api-scope.md` §2 に既載）
- `torch.compile(model)`／`@tf.function` 相当の**グラフ JIT 入口としての利用者向けフラグ**
  （REQ-12 抵触。opt-in は composition root 側 setter に限る〈F9 の型〉）
- dynamic shape specialization・guard 再コンパイル
- XLA 相当のクロス演算融合を前提とした性能目標（`docs/kernel-fusion.md` §1(c) と整合）

**注意（用語の混同回避）**: Keras 風 `Sequential::compile(optimizer, loss)`（学習設定 API。
`docs/compat-api-scope.md` §1.2 Tier 1・#1618）は本 doc の非目標に含まれない別物である。両者は
「compile」という語を共有するのみで、本 doc が扱うのはグラフ JIT コンパイルの方である。

## 6. 数値一致・既存テストとの整合

段階 0（本 issue）はコード変更を伴わないため、既存テスト・tolerance・baseline は不変のまま
影響を受けない。区分 B の各候補が着手時に触れうる数値契約を事前整理する:

- B-1（GPU elementwise 融合）: per-op 経路と丸めが変わりうるため、softmax の `exp2` 恒等式
  適用と同様に REQ-2 統一複合判定での受け入れを前提とすべきである（bit 同一を要求するかは
  §10 (4) の承認事項）
- B-3〜B-5（capture 拡張）: 既存の bit 同一契約（F6）を維持する

**tolerance／baseline 定数自体の変更は本 doc のスコープ外**であり、必要になった場合も改めて
ユーザー承認を要する。

## 7. 公開 API・spec 整合

REQ-12 の文言が Burn／CubeCL 前提だった旧版の名残を持つことは既知の課題（`docs/kernel-fusion.md`
§6）であり、本 doc は v2 読み替え（同 doc §7 の判断・2026-08-08 注記）に従う。facade への
新規公開面の追加はない。`docs/compat-api-scope.md` §5「範囲拡張手続き」の再適用は不要（本 issue
は「対象外」項目の範囲整理であり、対象範囲の拡張ではないため）。

## 8. 引き継ぎ（起票草案。本 issue では起票しない）

区分 B の各候補を、前提ゲートが充足した時点で以下の草案に沿って個別 issue 化する
（`.claude/rules/out-of-scope-tracking.md` に従い、起票自体はユーザー承認後に行う）。

1. **B-1**: 「feat(backend): GPU `run_fused` の elementwise allowlist 実装」。前提: なし（着手
   可能）。承認事項: REQ-2 判定 vs bit 同一の選択（§10 (4)）
2. **B-2**: 「feat(autodiff): `is_lazy_elementwise` へ `Sub`／`Div`／`Rsqrt` を追加」。前提:
   `ScalarBinaryOp`／`ScalarUnaryOp`（#1634）との重複整理
3. **B-3**: 「feat(backend): forward／backward capture」。前提: #1216 Phase 2・backward
   デバイス直接計算・loss 常駐化
4. **B-4**: 「feat(backend-cuda): exec update（`cuGraphExecUpdate_v2`）」。前提: 新規 `unsafe`
   FFI 承認（§10 (1)）
5. **B-5**: 「feat(backend-cuda): tape レベル構造キー」。前提: B-4 と同時導入
6. **docs フォローアップ**: `docs/kernel-fusion.md` §3 表 1・§6 の陳腐化記述（F10）の是正
   （Metal elementwise 実装済み化の反映）

## 9. スコープ外

- 実装全般（区分 B の候補実装含む）
- tolerance／baseline 変更
- `docs/spec/`（正本 submodule）の改定
- 実機実測
- issue 起票（草案のみ§8 に残す）

## 10. 承認事項（実装着手の前提。本 issue 時点ではいずれも未取得）

1. 新規 `unsafe` FFI の追加（B-4: `cuGraphExecUpdate_v2`）
2. `BackendOps`／`FusedOpKind`／`Op::is_lazy_elementwise` の拡張（B-1・B-2）
3. facade への opt-in setter 追加（`docs/compat-api-scope.md` §0 の公開面拡張が必要な場合）
4. GPU `run_fused` elementwise 経路の数値判定方式（REQ-2 判定で受け入れるか bit 同一を要求
   するか。B-1）
5. 区分 B 各候補の個別 issue 起票（§8 草案の実際の起票）

## 11. 出典

- `docs/compat-api-scope.md` §2「引き続き対象外」（`torch.compile`／`tf.function` bullet）
- `docs/compat-feature-gap.md` §2.16「推論・その他」（`torch.compile` 行）
- `docs/kernel-fusion.md`（融合の適用範囲・限界）
- `docs/backend-cuda-graph-step-capture-design.md`（CUDA Graph capture の設計・実測記録）
- `docs/inference-chain-single-sync-design.md`・`docs/device-resident-update-design.md`・
  `docs/inference-forward-fixed-cost-design.md`（デバイス常駐チェーンの設計記録）
- `docs/fusion-graph-design.md`（融合 IR・backward 対象外契約）
- `docs/spec/04-requirements.md` REQ-9・REQ-12（メイン checkout で参照。編集しない）
- `crates/autodiff/src/tape.rs`・`crates/tensor-core/src/fusion/`・
  `crates/backend-cpu/src/fused_elementwise.rs`・`crates/backend-cuda/src/ops.rs`・
  `crates/backend-metal/src/ops.rs`・`crates/facade/src/lib.rs`（§2 の出典行）
- 参考として issue #1632 本文に記載された外部 artifact リンク 2 件（`claude.ai/code/artifact/
  ...`）: 内容は本 doc の事実確認には使わず、参考情報としてのみ扱った

## #1962 追補

区分 B（5 節）各候補の HEAD 時点（`06cb1e38`）のゲート状況・判定は
`docs/facade-inference-serving-scope-decision.md` §2.3・§6 を正とする
（本節・5 節本文は書き換えない）。要点: `Op::is_lazy_elementwise`・CUDA／
Metal `run_fused` は本 doc 記録時点（`1a1bcd5a`）から不変で B-1〜B-6 いずれも
未着手のまま。B-3 の前提ゲートは (i) forward 常駐結線（#1688）・(ii) 重み
勾配のデバイス直接計算（#1555／#1559／#1908）が充足済みだが、`d_input`・
loss のデバイス常駐化は未充足。同 doc は B-1 のみを「実装する」（起票案
G-1）として引き継いだ。区分 A（実装済み）・区分 C（非目標）は不変。

## #2085 追補

B-1（GPU `run_fused` の elementwise allowlist。5 節）を実装した（イシュー
#2085）。CUDA（`crates/backend-cuda/src/fused_elementwise.rs`・
`kernels_fused_elementwise.rs`）・Metal（`crates/backend-metal/src/
fused_elementwise.rs`・`fused_elementwise_source.rs`）とも、CPU
`backend-cpu::fused_elementwise` と同一の allowlist（`Input`／`Add`／
`Mul`／`Relu`／`Exp`／`Tanh`）を対象に実行時カーネル生成（NVRTC／MSL）で
単一パス実行する経路を追加し、既存の `run_fused` オーバーライド
（canonical RMSNorm・softmax 判定）の後段へ結線した。

- **既定 OFF・opt-in ゲート**: `backend-cuda::fused_elementwise::
  set_gpu_elementwise_fusion_enabled`／`backend-metal::fused_elementwise::
  set_gpu_elementwise_fusion_enabled`（各クレート内 `pub`）。`facade` への
  再公開は行っていない（実装計画 §9 承認事項 (1)。公開面拡張は別途
  ユーザー承認が必要）。ゲート OFF 時は本 PR 導入前と挙動不変
  （atomic load 1 回を除き bit 同一）。
- **数値契約**: 融合カーネルは同一バックエンドの per-op 経路・CPU
  融合カーネルと bit 完全一致を目標とする。CUDA は `Add`／`Mul` を
  非縮約 intrinsic（`__fadd_rn`／`__fmul_rn`）で生成し FMA 縮約を
  遮断する。Metal は `MathMode::Safe`（既存コンパイルオプション）の
  下で `+`／`*` が correctly rounded・非縮約であることに依拠し、
  リポジトリ内に検証済み用例のない `#pragma METAL fp contract(off)`
  は追加していない。
- **検証**: Linux で実行可能な範囲（allowlist 判定・ゲート・ソース
  生成・ホスト逐語モデル対 CPU 融合カーネルの bit 一致・facade 経由の
  forward／backward 勾配 bit 完全一致）はテスト済み。CUDA／Metal 実機
  （同一バックエンド per-op 経路との bit 完全一致・CPU との REQ-2
  複合判定・A/B 性能計測）は本エージェント実行環境に実機への到達手段
  がなく未実測のまま申し送り（`docs/perf/gpu-elementwise-fusion-b1.md`・
  `docs/perf/logs/gpu-elementwise-fusion-2085/README.md` 参照）。
- **キャッシュ上限**: 融合プランごとに動的コンパイルされるカーネルの
  プロセス内キャッシュに上限（256 エントリ）を設け、上限到達時は
  `Unsupported` へ fail-closed に倒し per-op フォールバックへ委ねる
  （実装計画 §2.5）。
- B-2 以降（XLA 相当のクロス演算融合・`Sigmoid` 等 allowlist 拡張）は
  引き続き対象外のまま。
