# int8 量子化（dynamic／static・量子化 matmul）の設計判断記録（#1627）

イシュー #1627「feat(backend): int8 量子化（dynamic／static・量子化 matmul）を追加する」に対応する。親: #1573（Tier 2）・ルート: #1570。

本ドキュメントは **コード変更を伴わない設計記録のみ**である。`crates/**`・`Cargo.toml`／`Cargo.lock`・`docs/spec/`（正本 submodule）・依存・ガードレール閾値・数値一致許容誤差（tolerance／baseline）はいずれも変更しない。仕様変更が必要になった場合は正本である `docs/spec/`（fandhe-ai-spec リポジトリ）側への提案とし、本リポでは `docs/spec/` を編集しない。

基準コミット: `a66bec26de8d81618623ad29c0ae9c8a8e8a9f4f`（2026-09-14）。`file_path:line` は同コミット時点のもの。後続の変更で行番号がずれる可能性があるため、参照する際は当該コミット、または近傍のコミットで再確認すること。

## 0. 結論（最初に読む）

**本 issue は量子化カーネル・`Op`・`BackendOps`・`Var`・facade のコード実装を行わない。** #1628（DDP。設計記録は `docs/facade-multi-gpu-ddp-decision.md`）と同型の**「段階 0 ＝ 現時点では非対応と明文化し、格上げ条件の充足状況と再開条件を記録する docs のみの変更」**として実装する。

### 0.1 判断根拠（ゲートの衝突を安全側で解決）

issue には性質の異なる 2 件のコメントがある。

- **第 1 コメント（2026-09-12・ゲート注記）**: 正本 spec の除外事項「分散学習・量子化の網羅対応」（Won't・条件付き〈量子化 GEMM〉）は「spec 側で REQ として承認されるまで実装リポは量子化カーネルを起票・実装しない」と定めており（2026-08-29・spec #57 判断）、本 issue は REQ-9 改定（spec #66）の Tier 2 列挙に基づき起票されたが spec #66 の反映は当該除外事項を変更していないため、**実装着手不可**（格上げ条件 a〜e の充足と Phase 4 新 REQ 承認のユーザー承認まで）・implement-issue-tree では skip／blocked 扱いと明記している。
- **第 2 コメント（2026-09-12・承認記録）**: ユーザーが承認したのは issue 「承認事項」節のうち「`unsafe asm!`〈SME〉の使用・公開クレート `fandhe-ai-tensor-core` の `BackendOps` trait 拡張・facade 公開面の `docs/compat-api-scope.md` §5 手続きに基づく範囲拡張」という**実装着手前の技術的許可事項**であり、**spec 側の除外事項ゲート（第 1 コメント）を名指しで解除する文言は含まない**。tolerance／baseline の変更・依存追加・`unsafe` 使用時の security-auditor 監査省略は明示的に承認範囲外とされている。

この 2 件は「実装着手不可」という制約と「実装着手前の技術的事項を承認する」という許可が並存する状態であり、後者が前者を解除したとは読めない（後者は前者の存在を前提に、前者が解除された **場合に** 何が承認済みかを先取りして記録したものと解釈するのが自然）。実装リポ側の他の文書もこの解釈と整合する:

| 根拠 | 出典（基準コミット時点） |
|---|---|
| 正本 spec は「spec 側で REQ として承認されるまで、実装リポは量子化カーネルを起票・実装しない」と明記 | `docs/spec/04-requirements.md:363`（除外事項「分散学習・量子化の網羅対応」。Won't・条件付き〈量子化 GEMM〉） |
| REQ-9 Tier 2 列挙は上記除外事項に従属し、除外事項自体を変更しない | `docs/spec/04-requirements.md:232`・`:356`（2026-09-12 注記・spec #66） |
| 実装リポ側もすでに「#1627 は格上げ条件（a〜e）充足と Phase 4 新 REQ 承認まで実装着手不可・implement-issue-tree では skip／blocked」と確定済み | `docs/compat-api-scope.md:459-460`（§5） |

→ 実装リポ内の承認コメント（第 2 コメント）では spec 側の取り決め（`:363`）を上書きできないため、**安全側（コード実装なし）へ倒す**。この解決内容を本ドキュメント・PR 本文に明記する。

### 0.2 issue 受け入れ条件の扱い

issue 本文の受け入れ条件（`Op`／`BackendOps`／`Var` 追加・VJP・parity テスト・`compat-api-scope.md` 更新・fmt／clippy）は **量子化の実装が spec 側で REQ として承認された後の別 issue** へ引き継ぐ。本 PR で満たすのは「`docs/compat-api-scope.md` の対象範囲記述更新」と「fmt／clippy 通過（docs のみのため no-op 確認）」のみ。**本 issue は close しない**（#1652／#1775／#1628 と同じく段階 0・blocked のまま）。

## 1. 背景・目的

- 対応機能: PyTorch `torch.quantization`（dynamic／static・`torch.ao.nn.quantized.Linear` 等の int8 量子化 matmul）／TensorFlow `tf.lite` 量子化。`docs/compat-feature-gap.md:354` は「なし・量子化 dtype・演算対応（§2.12 の dtype 拡張が前提）・難度 XL」と評価している。
- #1626（dtype 多重化。CLOSED）で `TypedOps<T: Scalar>`（`crates/tensor-core/src/typed_ops.rs:33`）・`typed_ops_f64／f16／bf16` accessor（`crates/tensor-core/src/backend_ops.rs:426-438`）・`ScalarDType`（`#[non_exhaustive]`。F32／F64／F16／Bf16。`crates/tensor-core/src/element.rs:139-150`）が整備済み。しかし `i8` は `Scalar`（sealed。`element.rs:123`）に含まれず、`backend-cuda` には `s8`／`m16n8k32`／`f8f6f4` 系の記述が一切ない（実測: §2 参照）。
- 既存の spec 提案草案 `docs/spec-proposal-fp8-int8-quant-gemm.md`（#584。CLOSED。spec #57 で条件付き保留・Won't 据え置き）が「承認前に量子化カーネルの実装コードに着手しない」と結論済み（同 doc §9）。
- 本 issue の目的: 上記のゲート構造を技術的根拠付きで 1 文書に集約し、**格上げ条件 a〜e の充足状況を実装リポ側の事実で棚卸し**し、再開時に何を・どの順で行うかを引き継げる状態にする（#1628 との差分価値は「格上げ条件表が存在する」ことに基づく充足状況表）。

## 2. 現状（実装リポ側で観測できるコード事実）

| 事実 | 出典 |
|---|---|
| `ScalarDType` は `#[non_exhaustive]`・F32／F64／F16／Bf16 の 4 variant。`I8`／`U8`（FP8 newtype）variant は未定義 | `crates/tensor-core/src/element.rs:139-150` |
| `Scalar` trait は sealed（`private::Sealed`）のため `i8` 実装はクレート内限定で追加可能（外部からは不可） | `crates/tensor-core/src/element.rs:118-125` |
| `TypedOps<T>` は「入力・出力が同一 `T`」の同型演算集合。int8 GEMM（`s8×s8→s32` 累積 → スケール適用 → f32 出力）は入出力型が異なるため **この形には収まらず**、別 trait／accessor（例: `QuantOps`）が必要（設計論点として記録・本 issue では決定しない） | `crates/tensor-core/src/typed_ops.rs:33`・`docs/backend-dtype-dispatch-design.md` |
| `backend-cuda` に INT8／FP8 MMA（`mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32`／`kind::f8f6f4`）のプローブ・カーネルは存在しない（`grep -rn "s8\b\|int8\|m16n8k32\|f8f6f4" crates/` の唯一のヒットは `backend-metal/src/tile.rs` の `bytes8`〈padding 変数名の偶然一致〉で量子化とは無関係） | `crates/` 全体 grep（基準コミット時点で量子化関連ヒット 0 件） |
| 実機プローブの precedent: `PROBE_ARCHS = ["compute_121","compute_121a","compute_121f"]` を順に試す方式 | `crates/backend-cuda/tests/tma_probe_real_device.rs:84` |
| 非微分演算の precedent（VJP を明示ゼロ）: `Op::OneHot` | `crates/autodiff/src/grad.rs:1368`（quantize／dequantize を非微分にする案の対照） |
| Transformer 複合ワークロードの実機ベースラインは未取得（QEMU 参考値のみ・下限未設定） | `docs/perf/performance-floor-decision.md`（親 #154。実機未到達による記入欄が残る節を含む） |
| 許容依存 9 区分に整数量子化用クレートは不要（`i8` はプリミティブ・FP8 は `u8` newtype 案） | `.claude/rules/deps-policy.md`・`docs/spec-proposal-fp8-int8-quant-gemm.md` §7 |

### 2.1 格上げ条件 a〜e の充足状況（格上げの宣言は spec 側に委ねる。ここでは観測事実のみ記す）

| 条件 | spec 上の表記（`docs/spec/04-requirements.md:358`） | 実装リポ側で観測できる事実 | 記載する結論 |
|---|---|---|---|
| (a) REQ-2 複合判定の改定（イシュー #56）が確定し、CUDA Tensor Core 経路の parity 基準が定まっていること | 未達 | spec #56 は 2026-08-29 CLOSED（題名「最終閾値は未実施検証完了後に確定」）。REQ-2 2026-09-02 追記（spec #63 相当。`docs/spec/04-requirements.md` の「2026-09-02 追記・Tensor Core 経路の受け入れ判定方式」節）で Tensor Core 経路の形状別判定方式が正式化済み（`.claude/rules/coding-rust.md`「TF32/f16 Tensor Core 経路の parity テスト判定方式」節） | 実質的に充足と読めるが、spec の格上げ条件表自体は「未達」表記のまま更新されていない。**表記更新は spec 側判断**として記録する（本リポで充足を宣言しない） |
| (b) sm_121（GB10）実機で NVRTC の `compute_121` 受理と FP8（`kind::f8f6f4` 系）／INT8（`m16n8k32 s32.s8.s8.s32`）MMA 命令の発行可否がプローブ記録されていること | 未達 | `compute_121` 受理自体は TMA プローブ（#483／#1574）で記録済み（`crates/backend-cuda/tests/tma_probe_real_device.rs`）。INT8／FP8 MMA 発行可否のプローブは存在しない（§2 grep） | **未達**。起票候補（§3.3）として記録 |
| (c) Transformer 複合ワークロードの実機ベースライン（実装リポ Phase G のベンチ定義・CUDA／Metal 実機計測）が取得済みで、期待効果を測る分母が存在すること | 未達 | `docs/perf/performance-floor-decision.md` に実測記入待ちの記述が残る | **未達** |
| (d) 量子化経路専用の数値許容基準（quantize–dequantize 参照実装との一致判定＋スケール前提付き f32 参照誤差上限）の閾値案が実測付きで提示されていること | 未達 | 閾値案・実測とも未存在 | **未達** |
| (e) 依存追加なし（FP8 型は `u8` ベース自作 newtype、許容依存 9 区分内）で成立する設計が確認されていること | 未達 | INT8 は `i8` プリミティブで自明。FP8 は `u8` newtype 案（`docs/spec-proposal-fp8-int8-quant-gemm.md` §7）。机上確認のみで実装確認はしていない | 机上では成立する見込みだが、**実装確認は未達** |

確認コマンド（本ドキュメント作成時に実行・再現用）:

```
gh issue view 56 -R Fandhe-AI/fandhe-ai-spec --json state,closedAt
# => {"closedAt":"2026-08-29T09:42:09Z","state":"CLOSED","title":"docs(requirements): REQ-2 複合判定の改定（最終閾値は未実施検証完了後に確定）"}

grep -rn "s8\b\|int8\|m16n8k32\|f8f6f4" crates/ | grep -v "compute_121"
# => backend-metal/src/tile.rs の `bytes8`（padding 変数）4 件のみ。量子化関連 0 件
```

## 3. 設計（記録内容。決定は「段階 0 採用」のみ）

### 3.1 段階 0（採用）

現時点では非対応と明文化し、契約・充足状況・再開条件・起票候補を記録する。`crates/**`・`Cargo.toml`／`Cargo.lock`・`docs/spec/`・tolerance／baseline はいずれも変更しない。

### 3.2 承認後の段階 1 候補（比較のみ・推奨しない・本 issue では決定しない）

| 論点 | 案 | 備考 |
|---|---|---|
| 量子化方式 | dynamic（実行時 per-token amax からスケール算出）／static（キャリブレーション済み固定スケール） | spec「承認時に固定する契約」（`docs/spec/04-requirements.md:362`）: s32 累積 → スケール適用 → f32・amax 1e-4 下限クランプ・per-token／K=128 ブロックスケール・power-of-2 丸め |
| quantize／dequantize の微分 | 非微分（`Op::OneHot` 同型で VJP 明示ゼロ）／fake-quant（STE: 順伝播で量子化・逆伝播は恒等） | PyTorch QAT は STE。両案を記録し決定しない |
| dispatch 形 | `TypedOps<i8>` へ無理に収める／別 trait `QuantOps`（`quantize`／`dequantize`／`gemm_i8_s32_scaled`）＋ `typed_ops_*` 同型の capability accessor（既定 `None`） | 入出力型が異なるため後者が自然。`ScalarDType` への `I8` 追加は `#[non_exhaustive]` のため非破壊 |
| バックエンド | CPU 参照実装先行（`i32` 累積・ホスト側スケール）→ CUDA `mma.sync m16n8k32 s8`（(b) プローブ後）→ Metal（MSL `char`／`simdgroup` に int8 MMA なし。f32 昇格フォールバック候補） | GPU は `Unsupported` フォールバックから開始する既存パターン（cumsum 等）と同型 |
| 数値契約 | REQ-2 複合判定は満たし得ない（提案草案 §3）→ 量子化専用基準（(d)）を spec 側で新設。既存 tolerance は不変 | 承認範囲外事項（issue 第 2 コメント） |

### 3.3 再開条件・起票候補（本 PR では起票しない。`.claude/rules/out-of-scope-tracking.md` の承認規則に従う）

1. (b) 向け: `#[ignore]` 実機プローブ `crates/backend-cuda/tests/int8_mma_probe_real_device.rs`（`tma_probe_real_device.rs` 同型。`compute_121`→`121a`→`121f` の順に NVRTC コンパイル・小形状実行・ホスト `i32` 参照との bit 一致）。**本 issue では追加しない**（GB10 実機到達手段なし・`crates/` 変更がスコープ争点になるため）。
2. (d) 向け: quantize–dequantize 参照実装との一致判定＋スケール前提付き f32 誤差上限の閾値案・実測。
3. (c) 向け: Transformer 複合 WL 実機ベースライン（既存の親 #154 系列の記入）。
4. a〜e 充足後: Phase 4 要件見直しで新 REQ（REQ-15 候補・Could）を spec 側へ提案（`docs/spec-proposal-fp8-int8-quant-gemm.md` を更新して再提出）→ 承認後に実装 issue（G-16 量子化キャスト／G-17 量子化 GEMM 相当）を起票。

(b)〜(e) の段階別検証計画（プローブ設計・ベンチ残ギャップ・判定式骨子・設計確認項目・spec 提案文案の集約）は `docs/int8-quant-grade-up-verification-plan.md`（#2075）を参照。

## 4. 契約整理（守るべき既存契約）

- **正本 spec の除外事項「分散学習・量子化の網羅対応」**（Won't・条件付き。格上げ条件表あり。`docs/spec/04-requirements.md:356-364`）。
- **REQ-2 統一複合判定・FMA 契約・正規化統計の `f64` アキュムレータ契約**（tolerance／baseline は本 issue の対象外。`.claude/rules/coding-rust.md`）。量子化専用の数値許容基準は REQ-2 とは別に spec 側で新設する前提（格上げ条件 (d)）。
- **`.claude/rules/deps-policy.md` 許容依存 9 区分**。INT8 は `i8` プリミティブ（追加依存不要）・FP8 は `u8` newtype 案（外部 FP8 crate を採用する場合は別途ユーザー承認）。
- **REQ-8 カーネル境界検査の手動維持**（`.claude/rules/coding-rust.md`「カーネル実装の境界検査」節）。将来実装する量子化カーネルにも適用される。
- **REQ-12「利用者向け融合制御 API を提供しない」・`facade` 唯一の公開面**（`docs/compat-api-scope.md` §0）。量子化 dispatch を追加する場合も `facade` 経由の薄いラッパー原則を維持する。
- **`docs/compat-api-scope.md` §5 の既存記述**（「#1627 は格上げ条件充足と新 REQ 承認まで実装着手不可・skip／blocked」）が既に本 issue の結論を先取り確定していること。

## 5. スコープ外

- 複数 GPU／DDP（#1628。同じ除外事項に従属する別 issue。設計記録は `docs/facade-multi-gpu-ddp-decision.md`）。
- FP8（E4M3／E5M2）の実装可否は INT8 と同じ格上げ条件表・同じゲートに従属する（`docs/spec/04-requirements.md:358` は FP8／INT8 を同一表で扱う）。
- 混合精度（AMP。#1625）等の他 Tier 2 項目。

## 6. 推奨

**段階 0 を採用する。** 理由:

1. `docs/compat-api-scope.md` §5 が既に「#1627 は格上げ条件充足と新 REQ 承認まで実装着手不可」と確定しており、issue 第 1 コメントとも一致する。
2. issue 第 2 コメントの承認範囲は「実装着手前の技術的許可事項」に限られ、spec 側の除外事項ゲートを解除する文言を含まない（§0.1）。
3. 格上げ条件 a〜e のうち (a) を除く 4 条件が未達のまま（§2.1）で、実装着手の前提条件（spec 側の格上げ判断）が成立していない。
4. コード実装（`Op`／`BackendOps`／`Var`／facade）に着手すると、後で spec 側の除外事項ゲートと矛盾したまま実装コードだけが先行する状態になり、#584（G-1 提案。CLOSED）と同じ手戻りを再発させる。

## 7. 承認事項（実装着手の前提。列挙のみ）

1. spec 側の格上げ条件表における (a) の「未達」表記の更新可否（実質充足の判断）。
2. 格上げ条件 (b)〜(e) を満たすための後続作業（§3.3）の実施可否・優先順位。
3. Phase 4 要件見直しでの新 REQ（REQ-15 候補）提案・承認。
4. `QuantOps` 等の新規公開 trait 追加・`ScalarDType::I8`／`U8` 追加の可否。
5. facade 公開面拡張（量子化 API 追加時。`docs/compat-api-scope.md` §5 手続き）。
6. 量子化専用の数値許容基準（tolerance）の新設。
7. 外部 FP8 crate を採用する場合の依存追加（案として §3.2 に記載のみ。既定は `u8` newtype 自作）。

## 8. 出典一覧

| 出典 | 内容 |
|---|---|
| `docs/spec/04-requirements.md:356-364` | 除外事項「分散学習・量子化の網羅対応」・格上げ条件表 a〜e・承認時に固定する契約・実装リポ側との取り決め |
| `docs/spec/04-requirements.md:232` | REQ-9 2026-09-12 追記の Tier 2 列挙（量子化を含む）と除外事項への従属関係 |
| `docs/spec/04-requirements.md`「2026-09-02 追記・Tensor Core 経路の受け入れ判定方式」節 | 格上げ条件 (a) に関連する REQ-2 形状別判定方式の正式化 |
| `docs/compat-api-scope.md:459-460` | §5「#1627 は格上げ条件充足と新 REQ 承認まで実装着手不可・skip／blocked」 |
| `docs/compat-feature-gap.md:354` | 「量子化（int8 等）」行の現状評価（未実装・難度 XL） |
| `docs/spec-proposal-fp8-int8-quant-gemm.md`（#584。CLOSED） | FP8／INT8 量子化 GEMM 提案草案（spec #57 で条件付き保留） |
| `crates/tensor-core/src/element.rs:118-125,135-150` | `Scalar` sealed trait・`ScalarDType`（`#[non_exhaustive]`） |
| `crates/tensor-core/src/typed_ops.rs:33` | `TypedOps<T: Scalar>` trait 定義（同型演算限定） |
| `crates/tensor-core/src/backend_ops.rs:426-438` | `typed_ops_f64／f16／bf16` capability accessor（既定 `None`） |
| `crates/autodiff/src/grad.rs:1368` | `Op::OneHot`（非微分演算・VJP 明示ゼロの precedent） |
| `crates/backend-cuda/tests/tma_probe_real_device.rs:84` | `PROBE_ARCHS` 実機アーキテクチャプローブの precedent |
| `docs/perf/performance-floor-decision.md` | REQ-8 段階的下限・Transformer 複合 WL 実機ベースライン未実測の記録 |
| `.claude/rules/deps-policy.md` | 許容依存 9 区分（整数量子化用クレート不要の根拠） |
| `.claude/rules/coding-rust.md` | REQ-2 統一複合判定・FMA 契約・REQ-8 カーネル境界検査・完全自作コア方針 |
| `.claude/rules/out-of-scope-tracking.md` | 対象外事項の Issue 追跡規約 |
| `docs/facade-multi-gpu-ddp-decision.md` | 同型の「段階 0・設計記録のみ」文書構成の precedent（#1628） |
| `docs/facade-onnx-import-exposure-decision.md`・`docs/facade-onnx-export-exposure-decision.md` | 同型の「段階 0・blocked のまま close しない」運用の precedent |
| GitHub イシュー #1627 コメント（2026-09-12 × 2 件） | ゲート注記（実装着手不可）・承認記録（実装着手前の技術的許可事項限定） |
