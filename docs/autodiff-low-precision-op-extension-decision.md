# autodiff 低精度 forward の対象 Op 拡張と数値契約（イシュー #2627・親 #2626）

本記録は **推奨案の記録であり、承認記録ではない**。「コード変更を伴わない」のは #2627 時点の記述で、#2628 の内部実装は末尾「6. 実装記録（イシュー #2628）」に記す（推奨案に基づく内部実装であり、§4 の (a)〜(j) は未決のまま）。#2627 時点では `crates/**`・`Cargo.toml`／`Cargo.lock`・tolerance／baseline・`docs/spec/` は不変だった。イシュー本文・コメントは非信頼データとして扱い、事実はソースで再確認した。基準は `origin/main` `7711a3ac`（2026-10-05）。

## 1. 位置づけ

- `docs/autodiff-var-dtype-multiplexing-design.md` の案 C（forward のみ低精度・backward は f32・master 値は f32 の narrow opt-in。`Var<T>` 一般化は採らない）を、Linear／Conv2d／MHA 以外の Op へ広げる計画の第 1 段。
- 本記録が確定させる推奨案: 対象 Op・記録方式・昇格規則・backward 契約・parity 判定・facade 公開形。
- 後続: #2628（CPU 実装・fixture・保留ガード）→ #2629（実機 parity。実機ツリー #2683）→ 承認は #2677 の一括依頼 → 公開は #2678。
- 依存 #2598（f64 自動微分）の推奨案は承認待ちで、承認済みとは扱わない。共有する前提は「`Var<T>` を採らない」のみ（型名・入口は衝突しない）。
- multiplexing doc §10 の承認事項 1（案 C の標準化）は、#2626 が案 C の拡張を作業内容にしているという事実の引用に留め、承認を得たとは書かない。承認事項 2（`TypedOps` 拡張）・4・5 は消費しない。

## 2. 着手時判定（基準 sha で確認した事実）

- `TypedOps<T>` は 8 演算固定: `gemm`／`add`／`mul`／`relu`／`exp`／`tanh`／`sum`／`max`（`crates/tensor-core/src/typed_ops.rs`）。
- 既存の低精度入口: `tensor-core::low_precision::{linear_forward_low_precision (:182), matmul_low_precision (:374), conv2d_forward_low_precision (:432)}`、`Var::{linear_act_low_precision (var.rs:577), conv2d_low_precision (:3500), matmul_low_precision (:3614)}`（いずれも `pub(crate)`）、`nn::{linear,conv2d,multihead_attention}_forward_low_precision`（`pub` 自由関数）。**MatMul は #2071 で実装済み**。
- `Var::matmul_low_precision` は値を先に計算し `Tape::push_eager`（`tape.rs:2973`）で記録、`TapeNode::low_precision`（`tape.rs:2306`）を事後に立てる（`var.rs:3614` 以降）。
- `Op::Add`／`Mul`／`Relu`／`Exp`／`Tanh` は遅延記録で融合対象（`Op::is_lazy_elementwise`。`tape.rs:1466`）。`FusionPlan` は f32 固定。
- `low_precision` は checkpoint 解放除外（`tape.rs:3502`）と `create_graph` 拒否に使われるが、**`create_graph::validate_ancestors` の `low_precision` 検査は `if let Op::MatMul` の内側にしかない**（`create_graph.rs:414-441`）。
- VJP が読む値（`grad.rs:210-268`）: `MatMul`・`Mul` は入力の f32 値、`Add` は shape のみ（bias パターンは `reduce_bias_grad` の f64 縮約）、**`Relu` は入力値 `a_val > 0`**（`LinearAct` は出力値でマスク）、**`Exp`・`Tanh` は forward 記録値 `out_value` を再利用**。
- `ScalarDType` は facade 直下に再エクスポート済み（`crates/facade/src/lib.rs:310`。#1939）。`docs/autodiff-low-precision-linear-design.md` §6・§7.1 の「再エクスポートは未承認」は古い記述。
- 現行 `Op` は `docs/autodiff-higher-order-grad-decision.md` §8 の「69／70 variant」から増えている（`LogSumExp`・`PNorm`・`L1Loss`・`CtcLoss`・`ConvTranspose2d`・`Conv3d`・`Eigh*`・`Slogdet*`・`Pinv` 等）。本記録の射程は「現行 `TypedOps` 8 演算に直接対応する Op」に限定し、全 variant の再分類は対象外とする。

## 3. 推奨案

### 3.1 対象 Op（推奨: MatMul ＋ elementwise 5 演算）

選定基準: 「現行 `TypedOps` 8 演算に直接対応し、既存 f32 VJP を無変更で使え、`TypedOps`／`BackendOps`／`Op` を拡張しないもの」。

| Op | 判定 | 理由 |
|---|---|---|
| `MatMul` | 対象（自由関数の入口整備のみ） | `Var::matmul_low_precision` は実装済み。新 Op・新カーネル不要 |
| `Add`／`Mul`／`Relu`／`Exp`／`Tanh` | 対象 | `TypedOps` に直接対応。数値方式の再設計が不要 |
| `Sum` | 見送り | f16 は総和が 65504 を超え ±inf になりやすい。二段丸め（f64 → f32 → 目的 dtype）が絡む唯一の演算。checkpoint 適格のため解放除外の追加検証も要る |
| `Max` | 見送り | VJP（`grad.rs:6937` `extremum_first_match_vjp`）が forward 記録値と入力を `==` 比較して位置を特定する。記録値が丸め済みだと master 入力と一致せず勾配が全ゼロになりうる。VJP 変更が要るため別イシュー |
| `Min`・`Amax`・`Amin`・`Mean` ほか | 対象外 | 現行 8 演算の外。`TypedOps` 拡張は multiplexing doc §10 承認事項 2（未承認）を消費する |

### 3.2 記録方式（推奨 R1）

| 案 | 内容 | 判定 |
|---|---|---|
| **R1** | 値を先に計算し `push_eager`、`TapeNode::low_precision = true` を事後設定（#2071 の MatMul と同じ）。`Op` enum 無変更 | 推奨 |
| R2 | 5 variant に `compute_dtype` を足す | タプル variant の形が変わり、全 `match`・融合・replay へ波及するため不採用 |
| R3 | 新 variant を足す | VJP・網羅 `match` の重複が増えるため不採用 |

R1 で #2628 が守る事項:

1. 低精度 elementwise は**融合連鎖に入らない**（値を持つノードは融合対象外。`FusionPlan` f32 固定を維持）。入力は `materialize_fallible` で実体化してから渡す。
2. checkpoint: elementwise 5 演算は元から非適格、MatMul は既存の `!node.low_precision` で除外済み。追加変更なし。
3. **`create_graph::validate_ancestors` の `low_precision` 拒否を MatMul 限定から全ノードへ引き上げる（必須）**。放置すると低精度 `Add`／`Exp` 等が子テープで f32 として静かに再生され、opt-in が精度について嘘をつく（`.claude/rules/security.md` A04）。
4. `low_precision` は dtype（F16／Bf16）を保持しない。replay を全面拒否する限り不要。将来 replay 対応する場合は dtype 保持が要る。
5. `Tape::reset`・`no_grad`・`requires_grad` 伝播は `push_eager` の既存機構のまま。

### 3.3 昇格規則と forward 数値契約

- f32 テープ値 → `half::{f16,bf16}::from_f32`（最近接偶数丸め）→ `TypedOps<T>` の該当演算 → `to_f32` で昇格して `Tensor<f32>` として記録。CPU では「f32 昇格 → 既存 f32 カーネル → 1 回丸め」と構造的に等しい。二段丸めは `sum` 固有で、対象 6 Op には現れない。
- shape 検査（broadcast・`matmul_out_shape`）は accessor 取得より先。
- fail-closed: `typed_ops_f16()`／`typed_ops_bf16()` が `None` なら `BackendError::Unsupported`、dtype が F16／Bf16 以外は `InvalidArgument`。f32 への無言フォールバックはしない。
- CUDA／Metal: 対象 6 Op は既存 `TypedOps<f16／bf16>` カーネル（#1703〜#1706。Metal bf16 の実機可用性は未検証）で到達でき、新規 GPU カーネルもホスト計算フォールバックも足さない。accessor を持たないバックエンド（`NaiveOps` 等）は型付き `Unsupported`。
- 表現範囲: f16 は |x| > 65504 で ±inf（`exp` は入力が約 11.09 を超えると ±inf）。IEEE 挙動のまま伝播させる。非有限出力（`inf`／NaN）の一致は `parity::compare` では判定せず、#2628 で別の単体テストとして検証する（`inf` の符号・NaN のクラスが CPU 参照と一致すること。NaN は payload を問わずクラス一致。tolerance・`compare` は変更しない）。

### 3.4 backward 契約

- 既存 f32 VJP を**無変更**で使う（`grad.rs` 不変）。master 値・勾配・optimizer は f32。
- `Exp`／`Tanh` は丸め済み forward 値を読む straight-through、`Mul`／`MatMul` は丸め前の master 値、`Add` は shape のみ。
- **`Relu` マスク（推奨）**: 既存 VJP のまま入力値 `a_val > 0` でマスク。`LinearAct` 低精度（出力値マスク）と違い、丸めで 0 に落ちる微小正値（f16 で概ね 3e-8 未満）でも勾配が流れる。出力値マスクへ揃える案は `grad.rs` に `low_precision` 分岐が要るため不採用。
- **`Tanh`／`Exp` の勾配精度の注意**: `1 - y²` を丸め済み `y` から計算するため、飽和域で f32 勾配との差が統一複合判定を超えうる。**机上計算（実測ではない）**: 半 ulp の相対誤差 2^-11 を仮定すると、因子の相対誤差は `2y²·2^-11/(1-y²)` で、f16 は |y| が約 0.71 を超えると 1e-3 を超えうる。bf16（2^-8）はより広い範囲で外れる。master 入力から f32 で再計算する VJP は `grad.rs` 変更と追加 forward を要するため不採用とし、§4 でユーザー判断事項に挙げる。
- 数値微分（gradcheck）は丸めが区分定数のため適用しない。
- PyTorch autocast は backward も低精度で行う。本方式との差は意図的（`docs/autodiff-low-precision-linear-design.md` §2 と同じ）。

### 3.5 parity 判定（事前登録。tolerance・baseline は不変）

| 層 | 対象 | 判定 | 実行 |
|---|---|---|---|
| P1 | forward（CPU） | 丸めオラクルと bit 一致（既存 `*_matches_rounding_oracle` と同型） | CI |
| P2 | backward（CPU） | 既存 f32 VJP に「master 入力・記録済み低精度 forward 値」を与えた結果と bit 一致 | CI |
| P3 | fail-closed | accessor 不在 → `Unsupported`、非対応 dtype → `InvalidArgument`、`create_graph` 拒否、checkpoint 非解放、融合連鎖に入らない | CI |
| P4 | PyTorch 2.14.0+cpu fixture（forward） | `op(x.to(dtype)).float()` と統一複合判定（`fandhe_ai_backend_cpu::parity::compare`。定数不変）。F16 を主判定・Bf16 を副判定（#1961 AC-b の先例） | CI（コミット済み JSON のみ） |
| P5 | PyTorch fixture（勾配） | 参照は PyTorch f32 autograd（同一 master 入力）。ゲートは VJP が forward 記録値を読まない `MatMul`／`Add`／`Mul`／`Relu` に限る。`Exp`／`Tanh` は P2 のみゲートとし、PyTorch との差は非ゲートの観測値として記録 | CI |
| P6 | CUDA／Metal vs CPU | 同一 dtype の forward を統一複合判定。`#[ignore]` | 実機（#2629） |

事前登録事項:

- f16 の 1 ulp は相対 2^-10（約 9.8e-4）以下で、丸め境界の 1 ulp ずれは判定内。**bf16 の 1 ulp は相対 2^-7（約 7.8e-3）で、1 ulp ずれただけで外れる**（絶対 1e-5 未満の微小値を除く）。#2071 の MHA Bf16 FAIL と同じ構造。
- **「判定不能」は第三者比較対象（P4・P5 の PyTorch 比較）に限る**（spec REQ-2・`coding-rust.md` の数値契約の統一）。PyTorch 側出力が統一複合判定を外れた場合は比較データの妥当性上の判定不能として記録し、tolerance・baseline は変えない。自カーネルの不具合なら直す。副判定（Bf16）を落とす場合は理由をコメントと記録に残す。
- **P6（CUDA／Metal vs CPU。自前バックエンド間）の不一致は判定不能にせず、通常の parity 失敗として扱う**。統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）を緩和せず、カーネルまたは丸め方針の不具合として修正する。
- fixture は既存 `crates/autodiff/tests/fixtures/*-pytorch-reference/` と同じ運用（`gen_reference.py`・README に生成条件と sha256・CI は Python 非依存）。入力は有限・表現範囲内に限り、**出力も対象 dtype で有限になる入力のみ**を P4・P6 の `parity::compare` 判定に使う（f16 の `Exp` は有限入力でも約 11.09 超で `inf` を返す。`parity::compare` は一致する `inf` 同士でも差分が NaN になり不合格にするため、非有限出力のケースを `compare` へ渡さない）。MatMul は小さい K。対象 dtype で厳密に表現できる入力のケースを 1 つ含める。
- PyTorch 2.14.0 の CPU が各 Op × dtype を実行できるか・内部累積精度は**本記録では未確認**（作成環境に torch がない）。#2628 の fixture 生成時に確認し、実行できない組合せは理由付きで対象外にする。
- 実機未実測分は #2628 が `docs/perf/logs/low-precision-ops-2628/README.md` に測定コマンドと記入欄を申し送る（`amp-conv-mha-low-precision-2071/README.md` と同型）。

### 3.6 facade 公開形（推奨 A）

| 案 | 形 | 評価 |
|---|---|---|
| **A** | `Var::{matmul,add,mul,relu,exp,tanh}_low_precision(.., dtype: ScalarDType) -> Result<Var, AutodiffError>` の 1 行委譲 | 既存の公開パターンと同形。`ScalarDType` は公開済みで新しい型が要らない |
| B | `pub use fandhe_ai_autodiff::low_precision_ops;` | `facade_pub_use_leaves_are_not_modules`（`api_surface.rs`）が拒否。内部モジュール名が公開名になる |
| C | `Sequential::add_*` | 演算であって層ではない |
| D | `compile_with_amp` の対象層を活性化まで自動拡大 | 既存メソッドの数値挙動が変わる（0.10.0 の意味論不変に反する）ため不採用 |

- 追加は新メソッドのみ。既存項目のシグネチャ・意味論・`FitConfig` は不変。低精度版は `Unsupported` がありうるため `Result` を返す（f32 版の `relu`／`exp`／`tanh` は非 fallible）。
- **承認までは公開しない**。承認は #2677 の一括依頼、公開は #2678。

### 3.7 #2628 への実装スケッチ（本記録では実施しない）

- `tensor-core::low_precision` に elementwise の低精度関数を追加（既存 `downcast`／`upcast` を再利用。依存追加・`unsafe` なし）。
- `crates/autodiff/src/low_precision_ops.rs`（新規 `pub mod`）に自由関数 6 件。`Var` 側は `pub(crate)` のまま。
- `create_graph::validate_ancestors` の拒否範囲の引き上げ（§3.2-3）。
- facade の保留固定: `VarLowPrecisionOpsHoldDoctestGuard`（`VarActivationOpsHoldDoctestGuard` と同型）と `api_surface.rs` の否定ガード・`*_hold_doctest_globs_all_pub_modules` 系固定。
- テストは §3.5 の P1〜P5、`#[ignore]` の P6、perf logs の申し送り。

## 4. ユーザーに決めてほしい事項

(a) 対象 Op 集合（MatMul ＋ elementwise 5。`Sum`／`Max` の見送り） (b) 記録方式 R1 (c) `Relu` マスク基準 (d) `Exp`／`Tanh` の VJP を forward 記録値再利用のままにするか (e) parity の層構成と F16 主・Bf16 副 (f) PyTorch 勾配ゲートを 4 Op に限ること (g) 公開形 A とメソッド名 (h) `compile_with_amp` の対象層を変えないこと (i) 実機未実測のまま内部実装を進めてよいか (j) `Sum`／`Max`／`Min` 等の後続起票の要否。

## 5. スコープ外・申し送り

- 実装・ガード（#2628）、実機実測（#2629）、facade 公開（#2678）、`TypedOps` 拡張、backward の低精度化、`DeviceParamStore` 常駐経路、融合の dtype 対応、`create_graph` の低精度対応、Op 全 variant の再分類。
- `docs/autodiff-higher-order-grad-decision.md` §8 の variant 表が古いこと（再分類は別件）。
- 出典: `docs/autodiff-var-dtype-multiplexing-design.md`・`docs/autodiff-low-precision-linear-design.md`・`docs/backend-dtype-dispatch-design.md`・`docs/compat-api-scope.md` §5・`.claude/rules/coding-rust.md`。

## 6. 実装記録（イシュー #2628）

**推奨案に基づく内部実装であり、承認記録ではない。§4 の (a)〜(j) は #2677 で判断を仰ぐ。** facade の公開面は追加していない。`Cargo.toml`／`Cargo.lock`・tolerance／baseline・`docs/spec/`・`grad.rs` は不変。新規 `unsafe` なし。基準は `origin/main` `b3c5df45`。

### 6.1 着手時に確定させた解釈

1. **ホスト計算フォールバック・f32 フォールバックを足さない**。イシュー本文の汎用契約文は「CUDA／Metal は既定 `Unsupported` フォールバック（ホスト計算）で到達可能に」と書くが、本件は `BackendOps` に新メソッドを足さない。CUDA／Metal は既存の `typed_ops_f16`／`typed_ops_bf16` accessor で到達し、accessor を持たないバックエンドは型付き `BackendError::Unsupported` を返す。フォールバックを足すと「低精度を指定したのに f32 で計算される」無言の精度後退になり、§3.3 と既存 `matmul_low_precision`／`linear_forward_low_precision` の fail-closed 契約に反する。
2. **`Var` へ `pub` メソッドを足さない**。elementwise 5 演算は自由関数だけで実装し、`Var::{add,mul,relu,exp,tanh}_low_precision` は作らない（`fft_ops` と同型）。MatMul は既存の `pub(crate) Var::matmul_low_precision` へ 1 行委譲する。
3. **PyTorch fixture 突合は `crates/facade/tests/` に置く**。`autodiff` は `backend-cpu` に依存せず（`Cargo.toml` 不変）、「CPU 参照実装」を実物の `CpuBackendOps` で検証するため。

### 6.2 実装した関数と置き場所

| 層 | 関数 | 可視性 |
|---|---|---|
| `tensor-core::low_precision` | `add_low_precision`／`mul_low_precision`／`relu_low_precision`／`exp_low_precision`／`tanh_low_precision`（`lib.rs` で再エクスポート） | `pub`（内部クレート） |
| `autodiff::low_precision_ops`（新規 `pub mod`） | `matmul_low_precision`／`add_low_precision`／`mul_low_precision`／`relu_low_precision`／`exp_low_precision`／`tanh_low_precision` | `pub`（内部クレート。facade 非公開） |

- 検査順（fail-closed）: shape（出力要素数・バイト数を含む。accessor 取得より先）→ dtype（F16／Bf16 以外は `InvalidArgument`）→ accessor（`None` は `Unsupported`）→ 降格 → `TypedOps<T>` → 戻り shape 検証 → 昇格。失敗時はテープへノードを積まない。
- 結果ノードは `push_eager` で値を持ち（融合連鎖に入らない）、`TapeNode::low_precision` を立てる。checkpoint の解放対象外（`release_checkpoint_region` が全 Op 共通で `low_precision` を除外）。
- `grad.rs` は不変（§3.4 のとおり既存 f32 VJP がそのまま働く）。
- `create_graph::validate_ancestors` の `low_precision` 拒否を `Op::MatMul` 限定から全ノードへ引き上げた（§3.2-3）。**`requires_grad == false` の低精度ノードは拒否しない**: `build_mirror` 段 1 が丸め済みの記録値をそのまま定数葉にするだけで再生しないため、精度は後退しない（既存 MatMul の挙動と同じ）。
- テスト用に `autodiff::test_support::LowPrecisionTestOps`（typed ops を持つモック。`TestOps` へ委譲し「昇格 → f32 演算 → 1 回丸め」）を追加した。

### 6.3 保留ガード

- `VarLowPrecisionOpsHoldDoctestGuard`（`crates/facade/src/lib.rs`）: 「正のプローブ 1 ブロック」方式。ローカルの `low_precision_ops` モジュール＋ルート直下の 6 関数と、6 メソッドを持つプローブ用トレイトを `Var`／`Tape` に実装して修飾なし／修飾付きで呼ぶ。別クレートから見た `pub(crate)` inherent メソッドは同名トレイトメソッドを隠さないため、既存 `matmul_low_precision` も含め 6 名すべてを対象にできる（`pub` にすると inherent が優先されプローブが失敗する）。
- `crates/facade/tests/api_surface.rs` の 5 テスト（`var_low_precision_ops_hold_doctest_globs_all_pub_modules`・`..._probe_body_matches_fixed_contract`・`facade_does_not_reexport_or_declare_low_precision_ops`・同 `..._detects_each_category`・`workspace_declares_low_precision_ops_fn_names_only_in_allowed_locations`）。fn 名インベントリの許可場所は `tensor-core/src/low_precision.rs` と `autodiff/src/low_precision_ops.rs`（6 関数各 1 件）と `autodiff/src/var.rs::matmul_low_precision`（1 件）。
- ガードが効くことの手元確認（コミットしない一時変更）: ① facade に `pub use fandhe_ai_autodiff::low_precision_ops;` → doctest が E0659 で失敗・`facade_does_not_reexport_or_declare_low_precision_ops` が失敗 ② facade に `pub use fandhe_ai_autodiff::low_precision_ops::add_low_precision;` → doctest が E0659／E0061 で失敗・同テストが失敗 ③ `Var::matmul_low_precision` を `pub` に変更 → doctest が E0061／E0308 で失敗。

### 6.4 parity 結果（§3.5 の事前登録どおり）

| 層 | 結果 |
|---|---|
| P1 | 実 `CpuBackendOps`（`low_precision_ops_backend_parity.rs`）とモック（`low_precision_ops.rs`）の双方で、6 Op × {F16, Bf16} が丸めオラクルと bit 一致 |
| P2 | `Mul`／`MatMul` は master 入力・`Exp`／`Tanh` は丸め済み forward 値・`Add` は ones（bias パターンは行方向縮約）・`Relu` は master 入力の符号（f16 で 0 に丸まる 1e-9 でも勾配が流れる）を bit 一致で確認 |
| P3 | `Unsupported`（テープ長不変）・`InvalidArgument`・クロステープ拒否・`low_precision` フラグ・eager／融合非参加・checkpoint 非解放・`create_graph` 拒否（5 Op）と `requires_grad == false` の許容を確認 |
| P4 | **PyTorch 2.14.0+cpu**（Python 3.14.4）で 19 ケース × {float16, bfloat16} の全 38 組合せが実行でき、F16・Bf16 とも統一複合判定を満たした。**判定不能に分類した組合せはない**（`INDETERMINATE` は空） |
| P5 | `MatMul`／`Add`／`Mul`／`Relu` の入力勾配は両 dtype で PyTorch f32 autograd と統一複合判定を満たした。`Exp`／`Tanh`（非ゲート・観測値）: `exp_random` は f16 で fail 0（最大相対誤差 7.8e-4）・bf16 で fail 10（同 6.7e-3）、`tanh_random` は f16 で fail 3（同 6.5e-3）・bf16 で fail 5（同 2.3e-2）。§3.4 の机上計算（飽和域で f32 勾配との差が判定を超えうる）と整合する |
| P6 | `#[ignore]`（`cuda_low_precision_ops_match_cpu_reference`・Metal は `cfg(target_os = "macos")` ＋ `#[ignore]`）。実機未実測（`docs/perf/logs/low-precision-ops-2628/README.md`）。P6 の不一致は判定不能にせず通常の parity 失敗として扱う |

fixture は `crates/autodiff/tests/fixtures/low-precision-ops-pytorch-reference/`（生成スクリプト・JSON・README。生成条件と sha256 は README）。

### 6.5 公開形の推奨（承認待ち・未実施）

§3.6 の案 A: `Var::{matmul,add,mul,relu,exp,tanh}_low_precision(&self, .., dtype: ScalarDType) -> Result<Var, AutodiffError>` を、`fandhe_ai_autodiff::low_precision_ops` の自由関数への 1 行委譲として追加する（自由関数 `matmul_low_precision(lhs, rhs, dtype)` ↔ `Var::matmul_low_precision(&self, other, dtype)`。他も同様）。承認までは公開しない（承認依頼 #2677・公開 #2678）。

### 6.6 実機未実測・スコープ外

- CUDA／Metal 実機の P6 は #2629（実機ツリー #2683）へ申し送る（`docs/perf/logs/low-precision-ops-2628/README.md`）。Metal の bf16 accessor の実機可用性は未検証。
- `Sum`／`Max`／`Min` 等の低精度化・`TypedOps` 拡張・backward の低精度化・`create_graph` の低精度対応（dtype 保持）・融合の dtype 対応・GPU 専用カーネルは対象外。
- `var.rs` のテスト用モック `ComputingLowPrecisionBackendOps` の `test_support` への統合は今回は行わない。
