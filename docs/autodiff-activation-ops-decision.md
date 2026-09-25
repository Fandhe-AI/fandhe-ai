# mish・hardtanh・relu6・prelu・glu の設計判断記録

イシュー #2146（親 #2131）。`docs/autodiff-matrix-ops-decision.md`
（#2144）・`docs/autodiff-rearrange-ops-decision.md`（#2143）と同型の
記録。

## §0 結論

PyTorch 互換の活性化 5 種（`mish`／`hardtanh`／`relu6`／`prelu`／
`glu`）を、**`fandhe_ai_autodiff` のうち facade が再エクスポートしない
自由関数モジュール `activation_ops`**（`crates/autodiff/src/
activation_ops.rs`）として実装した（案 C。§2.1 参照。`matrix_ops`
〈#2144〉・`rearrange_ops`〈#2143〉・`bool_ops`〈#2141〉と同じ判断
枠組み）。5 演算を包む nn 層（`Mish`／`Hardtanh`／`Relu6`／`PRelu`／
`Glu`。`crates/autodiff/src/nn/activation.rs`）と `Module` trait 実装
（`crates/autodiff/src/nn/module.rs`）も追加した。`Var` に inherent の
`pub fn` は追加していない。新規 `Op`・`BackendOps` メソッド・VJP・
GPU 専用カーネルは追加していない——いずれも既存の `Var::mul`
（`Op::Mul`）・`Var::tanh`（`Op::Tanh`）・`Var::softplus`
（`Op::ScalarUnary`）・`Var::clamp`（`Op::ScalarUnary`）・
`Var::where_cond`（`Op::Where`）・`Var::detach`・`Var::narrow`・
`Var::sigmoid`（`Op::Sigmoid`）・`Var::reshape` の合成のみで構成した。
facade 公開（`Var` への委譲メソッド追加・`compat::Sequential::add_*`
5 種）は承認待ちのまま対象外とし、`crates/facade/src/
lib.rs::VarActivationOpsHoldDoctestGuard`（正のプローブ doctest。
`Var` への衝突プローブと `compat::Sequential::add_*` への衝突プローブ
の 2 種併用）と `crates/facade/tests/api_surface.rs` のソース走査・
workspace インベントリ（4 テスト）で多層固定している。

## §1 背景

イシュー #2146・親 #2131 にはコメントが 0 件で、facade 公開の承認
記録はない（2026-09-25 時点。着手前に `gh issue view 2146/2131` で
確認済み）。親 #2131 はこのツリーでの facade 公開面の拡張を「設計
判断記録 → 承認 → 実装」の 2 段階と定めているため、本実装は内部
クレート限定に倒す。同じツリーの先例（#2143・#2144）はいずれも
Tier 1 に列挙済みの機能だが facade 公開を承認待ちとした前例に従う。

## §2 設計判断

### §2.1 API の置き場所 — 案 C（自由関数）を採用

- **案 A（`Var::mish` 等の inherent メソッド）**: イシューの対象範囲欄
  は `var.rs` を挙げているが、`Var` は facade から再エクスポートされる
  ため、メソッドを足すと即座に facade の公開面が広がる。#2143・#2144
  と同じ理由で不採用。
- **案 C（採用）**: `crates/autodiff/src/activation_ops.rs` に
  `pub fn mish`／`hardtanh`／`relu6`／`prelu`／`glu` を置き、facade は
  再エクスポートしない。`nn::activation` の各層はこれらの関数へ薄く
  委譲する。承認後は `Var::mish` 等の委譲メソッドと `compat::
  Sequential::add_*` を追加し、保留ガードを撤去する。

### §2.2 compat::Sequential の add_* — 保留

イシューは `compat::Sequential::add_*` 5 種の追加を承認事項として
明示し「承認前に実施しない」と定めている。コメントでの承認もない
ため、`crates/facade/src/compat/sequential.rs` は変更していない。

### §2.3 5 演算の合成方式

| 演算 | 合成 | 数値・勾配の契約 |
|---|---|---|
| `mish(x)` | `x.mul(&x.softplus(1.0, 20.0)?.tanh())` | 超越関数を含むため REQ-2 統一複合判定 |
| `hardtanh(x, min, max)` | `where_cond(open_mask, x, clamp(x).detach())` | forward は `clamp` と bit 完全一致 |
| `relu6(x)` | `hardtanh(x, 0.0, 6.0)` | 同上 |
| `prelu(x, weight)` | `where_cond(pos_mask, x, w_b.mul(x))` | forward・入力勾配 bit 完全一致・weight 勾配は REQ-2 |
| `glu(x, dim)` | `narrow` → `a.mul(&b.sigmoid())` | `sigmoid` を経由するため REQ-2 |

**hardtanh の境界勾配（推奨案を採用）**: 既存の `Var::clamp` の VJP は
境界値ちょうどで勾配係数 1 を返す（境界を含む扱い）。PyTorch の
`hardtanh_backward` は開区間 `min < x < max` のときだけ 1 を返し、
境界では 0 になる。`Var::where_cond(&open_mask, x, &x.clamp(min,
max)?.detach()?)` を採用した:
- `open_mask` はホスト側で `min < x < max` から作る `Tensor<bool>`
  （`build_value_mask` ヘルパー。`x.host_view()` で contiguous な値を
  読み出し、`RefCell` 借用を関数内に閉じ込める）
- forward の値は、区間内では `x`（= `clamp(x)`）、それ以外（境界と
  `NaN` を含む）では `clamp(x)` となる。したがって `clamp` と bit
  完全一致する
- 勾配は区間内だけ流れ、境界と区間外では 0 になる（PyTorch と一致）。
  `detach()` が `clamp(x)` 側の勾配経路を切るため構造的に成立する

構築・呼び出し時の検査: `min`・`max` が `NaN` なら拒否する。`min <
max` を必須とする。違反は `AutodiffError::InvalidArgument` を返す。
`±inf` は受け付ける。

**prelu(x, weight)**: 却下した合成は `relu(x) + w * (x - relu(x))`
（`x = +inf` で `inf - inf = NaN` を生むため）。`weight` は rank 1・
`C = weight.shape()[0] >= 1` を要求する。`C == 1` は全チャネル共有、
`C > 1` は `x` の rank が 2 以上かつ `x.shape()[1] == C` を要求する
（PyTorch のチャネル軸＝軸 1 規約）。検査は計算・メモリ確保の前に
すべて行う。`weight` 勾配は `mul` の broadcast 縮約（汎用
`grad::reduce_to_shape`）を経由する——`.claude/rules/coding-rust.md`
の f64 アキュムレータ統一契約は bias パターン（`reduce_bias_grad`）
限定であり、汎用 `reduce_to_shape` 自体は対象外・不変のまま
（`activation_ops.rs` モジュール doc・実装計画 §2.3 補足と同じ
注記）。

**glu(x, dim)**: 検査を先に行う（rank 0・`dim >= rank` は
`ShapeError::AxisOutOfRange`、`shape[dim]` が奇数なら
`InvalidArgument`）。負の `dim` は受け付けない（本クレートの慣例に
従い `usize`）。`Var::sigmoid` はどのバックエンドでもホストの
`eval::sigmoid` を通る。

### §2.4 nn 層

| 層 | 持つ値 | 既定値（`Default`） |
|---|---|---|
| `Mish` | なし（ユニット構造体） | derive |
| `Hardtanh` | `min_val`, `max_val` | `-1.0, 1.0`（PyTorch と同じ） |
| `Relu6` | なし（ユニット構造体） | derive |
| `Glu` | `dim: usize` | なし |
| `PRelu` | `weight: Tensor<f32>`, `requires_grad` | — |

`PRelu` は初めて状態（学習可能パラメータ）を持つ活性化層で、
`RmsNorm`（`nn/norm.rs`）と同型の「本体 → `bind(&tape)` で `Var` 化
した `PReluVars`」分離パターンを踏襲する。

### §2.5 Module trait の実装

`forward_host`（tape を使わない推論専用経路）は、同じ値になることを
検証できたものだけがオーバーライドする:

- **`Hardtanh`／`Relu6`**: `crate::grad::scalar_unary_with_fallback`
  を `ScalarUnaryOp::Clamp{min,max}` で呼ぶ。tape 経由の forward も
  `clamp` と bit 完全一致するため構造的に一致する
  （`module.rs::hardtanh_forward_host_matches_tape_forward`・
  `relu6_forward_host_matches_tape_forward` で確認済み）
- **`Mish`／`Glu`／`PRelu`**: `softplus`／`tanh`／`sigmoid`／
  `where_cond` を経由する多段合成であり、手書きの `ops.*`／`eval::*`
  直接合成が tape 経由 forward と bit 一致するかを検証していない。
  確かめていない bit 一致を主張しないため、`forward_host` は既定
  （`Unsupported`）のまま・`supports_forward_host()` を `false` へ
  オーバーライドする（`Embedding`・`MultiheadAttention` と同型の
  契約。`module.rs::mish_glu_prelu_do_not_support_forward_host` で
  確認済み）

## §3 数値契約

`crates/autodiff/src/activation_ops.rs` モジュール doc §「数値契約」
に集約した（本節では重複記載しない）。要約は §0・§2.3 の表を参照。

## §4 PyTorch との既知の差異

- `mish` の softplus 閾値: PyTorch CPU 参照実装は softplus に閾値を
  持たないが、本実装は `threshold=20.0`（`nn.Softplus` 既定）を使う。
  `x > 20` では `exp(-20) ≈ 2e-9` が f32 の丸め誤差より小さいため
  両者の差は REQ-2 判定内に収まる
- `glu` は負の `dim` を受け付けない（本クレートの慣例に従い
  `usize`。PyTorch の既定 `dim=-1` は呼び出し側が明示的な軸番号で
  指定する）

## §5 スコープ外

- facade 公開（`Var::mish` 等の委譲メソッド化と保留ガードの撤去）
- `compat::Sequential::add_*` 5 種（承認事項として明示保留）
- GPU 専用カーネル（現状は既存カーネル・ホストフォールバックのみで
  到達）
- CUDA／Metal 実機での計測（§7 参照）

## §6 承認事項（未承認として列挙）

1. facade 公開（`Var` への委譲メソッド追加）
2. `compat::Sequential::add_*` 5 種
3. GPU 専用カーネル（別イシュー）

## §7 実機実測の申し送り

CUDA（DGX Spark GB10）・Metal 実機は本エージェント実行環境に無いため
未実測。`docs/perf/logs/activation-ops-2146/README.md` へ測定コマンド
案・期待結果を申し送る。

## §8 実装記録（イシュー #2146・2026-09-25）

- `crates/autodiff/src/activation_ops.rs`（新規）: `mish`／
  `hardtanh`／`relu6`／`prelu`／`glu`・モジュール doc（役割・facade
  非公開の理由・数値契約表・PyTorch との差分・REQ-8）・単体テスト
  12 件（forward・エッジケース・エラー系・勾配・境界勾配契約）
- `crates/autodiff/src/lib.rs`: `pub mod activation_ops;` を追加
  （アルファベット順先頭）・クレート doc にイシュー #2146 の要約を
  追記
- `crates/autodiff/src/nn/activation.rs`: `Mish`／`Hardtanh`／
  `Relu6`／`Glu`／`PRelu`／`PReluVars` を追加（`Hardtanh`／`Relu6` は
  `forward_host` 用の `op()` クレート内アクセサも持つ）
- `crates/autodiff/src/nn/module.rs`: `impl Module` 5 件・trait doc
  の実装数（16 → 18。`forward_host` をオーバーライドする実装のみの
  カウント）・`forward_host` bit 一致テスト 2 件・
  `supports_forward_host` 事前申告テスト・`PRelu` のパラメータ管理
  テスト
- `crates/facade/src/lib.rs`: `VarActivationOpsHoldDoctestGuard`
  （正のプローブ doctest。`Var`／`Tensor<f32>`／`Tape` への衝突
  プローブと `compat::Sequential::add_*` への UFCS 限定衝突プローブの
  2 種併用。`VarHooksHoldDoctestGuard` と同型）
- `crates/facade/tests/api_surface.rs`:
  `activation_ops_hold_doctest_globs_all_pub_modules`・
  `activation_ops_hold_doctest_probe_body_matches_fixed_contract`・
  `facade_does_not_reexport_or_declare_activation_ops`（`add_*` 5 種
  の非宣言も検査）・`workspace_declares_activation_ops_fn_names_
  only_in_autodiff_activation_ops`（4 テスト。着手前の実測 grep で
  `crates/tensor-core/src/scalar_op.rs::relu6`〈private〉との名前
  衝突を確認済みのため期待集合に含めた）
- `crates/facade/tests/activation_ops_backend_parity.rs`（新規）:
  CPU と NaiveOps の bit 完全一致 forward／backward（`hardtanh`／
  `relu6`／`prelu`）・REQ-2 統一複合判定 forward／backward（`mish`／
  `glu`／`prelu` weight 勾配）（属性なし 4 件）＋CUDA／Metal の
  `#[ignore]`（未実測。8 件。`docs/perf/logs/activation-ops-2146/
  README.md` 参照）
- `docs/compat-api-scope.md`: §1.2「GELU／SiLU 等の活性化」行へ
  追補
- `docs/README.md`: 本 doc・perf log README の索引行を追加

承認取得後の追随（本イシューでは未実施）: `Var::mish` 等の薄い委譲
メソッド追加、`compat::Sequential::add_*` 5 種追加、facade 保留
ガード（`VarActivationOpsHoldDoctestGuard`・対応する否定ガード 4 件）
の撤去。

**手動検証（実装計画「手順 6」）**: `crates/facade/src/lib.rs` へ
`pub use fandhe_ai_autodiff::activation_ops;` を仮に追加し、
`facade_does_not_reexport_or_declare_activation_ops`（ソース走査
ガード）が FAILED になること、`cargo test -p fandhe-ai --doc
VarActivationOpsHoldDoctestGuard`（正のプローブ doctest）が
E0659（名前解決の曖昧化）で FAILED になることを確認済み。同じ行を
外すと両テストとも green に戻ることも確認済み（2026-09-25）。

## §9 網羅契約の追加是正（codex-review 相当指摘・実装エージェント自己是正・2026-09-25）

初版の `crates/facade/tests/activation_ops_backend_parity.rs` は
CUDA／Metal の `#[ignore]` backward テスト 4 件（`metal_bit_exact_
backward_matches_cpu_reference`・`cuda_bit_exact_backward_matches_
cpu_reference`・`metal_req2_backward_matches_cpu_reference`・
`cuda_req2_backward_matches_cpu_reference`）が、対応する CPU 版
（`cpu_bit_exact_backward_matches_naive_reference`・`cpu_req2_
backward_matches_naive_reference_within_tolerance`）と異なり
代表 1 演算（`hardtanh`・`mish`）のみを検証し、`relu6`／`prelu`
（bit 完全一致 backward 側）・`glu`／`prelu` の `weight` 勾配
（REQ-2 backward 側）が空セルのまま残っていた。これはモジュール
doc が明記する「関数名は網羅表と一対一対応する」契約・#2144 の
codex-review 教訓「代表 1 演算での省略はしない」に反するため、
CPU 版のテスト本体をそのまま device 版へ複製し、4 テストとも CPU
版と同じ演算・セル数（3 セル）を網羅するよう是正した（`docs/perf/
logs/activation-ops-2146/README.md` の「代表とする」という誤記述も
同時に修正）。
