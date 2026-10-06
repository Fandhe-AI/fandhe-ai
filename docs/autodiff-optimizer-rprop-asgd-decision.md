# Rprop・ASGD の設計判断記録

イシュー #2655（親 #2654「不足 optimizer」・ルート #2499 Phase 4）。
`docs/autodiff-optimizer-adadelta-adamax-nadam-radam-decision.md`（#2171）と同型。

## 1. 背景

`fandhe_ai_autodiff::nn::optim` には PyTorch `torch.optim.Rprop`（符号ベースの
resilient backpropagation）と `torch.optim.ASGD`（averaged SGD）に相当する
optimizer がなかった。本イシューはこの 2 種を、既存 optimizer（`AdamW`・
`Adamax` 等）と同じ「`Tape`／`Var`／`BackendOps` に依存しないホスト側の値型・
純関数」として内部クレートへ追加する。facade への公開は行わず、保留ガードで
機械固定する（公開は承認依頼 #2677 → 承認後の #2679）。

## 2. 公開 API とシグネチャ

`crates/autodiff/src/nn/optim/{rprop,asgd}.rs`。`mod.rs` から
`Rprop`／`RpropConfig`／`Asgd`／`AsgdConfig` を `pub use` する。命名は既存の
`RmsProp`・`NAdam`・`Lbfgs` と同じ Rust 流の casing（`ASGD` は clippy
`upper_case_acronyms` に掛かるため `Asgd`）。呼び出し規約は既存 9 種と同じ:

```rust
pub struct RpropConfig { pub lr: f32, pub eta_minus: f32, pub eta_plus: f32,
                         pub step_size_min: f32, pub step_size_max: f32 }
// Default: lr=1e-2, eta_minus=0.5, eta_plus=1.2, step_size_min=1e-6, step_size_max=50.0
pub struct AsgdConfig { pub lr: f32, pub lambd: f32, pub alpha: f32,
                        pub t0: f32, pub weight_decay: f32 }
// Default: lr=1e-2, lambd=1e-4, alpha=0.75, t0=1e6, weight_decay=0.0

impl Rprop / Asgd {
    pub fn new(config) -> Result<Self, AutodiffError>;
    pub fn config(&self) -> &Config;
    pub fn set_lr(&mut self, new_lr: f32) -> Result<(), AutodiffError>;
    pub fn step_count(&self) -> u64;
    pub fn step(&mut self, params_and_grads: &[(&Tensor<f32>, &Tensor<f32>)])
        -> Result<Vec<Tensor<f32>>, AutodiffError>;
}
impl Asgd {
    /// 平均化パラメータ `ax`（スロット順）。初回 step 前は空列。
    pub fn averaged_params(&self) -> Result<Vec<Tensor<f32>>, AutodiffError>;
}
```

ASGD の成果物は平均化パラメータ `ax` であり PyTorch では
`optimizer.state[p]["ax"]` からしか取れないため、値型では `averaged_params()`
を用意する（内部型のメソッドで facade 公開面は増えない）。`step()` は
`adamax.rs` と同じ 2 段構成（副作用なしの検証 → 状態変更）と
`step_count.checked_add(1)` を遅延初期化より前に確定させるアトミック性を踏襲
する。

## 3. 演算順（PyTorch 2.14.0+cpu の `_single_tensor_*` を実装前に確認）

**Rprop**（状態: スロットごとに `prev`〈0 初期化〉・`step_size`〈初回 step の
`lr` で全要素初期化〉）

```
prod   = grad * prev                      # f32 の積（アンダーフローで 0 になりうる）
s      = sign(prod)                       # torch の sign
factor = s > 0 → eta_plus / s < 0 → eta_minus / それ以外 → 1
step_size = clamp(step_size * factor, step_size_min, step_size_max)
grad   = (s < 0 の要素は 0)               # 符号反転した要素は今回動かさない
param -= sign(grad) * step_size
prev   = grad                             # ゼロ化後の grad
```

**ASGD**（状態: スロットごとに `ax`〈0 初期化〉・`eta: f32`〈初期値 `lr`〉・
`mu: f32`〈初期値 1〉）

```
step += 1
weight_decay != 0 なら grad += weight_decay * param     # 減衰前の param
param *= (1 - lambd * eta)
param -= eta * grad
mu != 1 なら ax += (param - ax) * mu、mu == 1 なら ax = param
eta = lr / (1 + lambd * lr * step)^alpha                # 次の step で使う
mu  = 1 / max(1, step - t0)                             # 同上
```

更新には「前 step の終わりに計算した `eta`／`mu`」を使い、当 step の終わりに
次回分を計算する点が要点。

## 4. 数値方針

- **`f32::signum` を使わない**。torch の `sign` は `(0 < x) - (x < 0)` で
  `±0.0` と NaN は 0。Rust の `signum` は `+0.0` に 1.0・NaN に NaN を返すため、
  ゼロ勾配で更新が走る誤実装になる。`rprop.rs::torch_sign` を手書きし
  `sign(grad*prev)` と `sign(grad)` の両方に使う。
- ASGD の `eta`／`mu` は PyTorch が `float32` スカラーテンソルで保持する。係数は
  `f64` で計算して `f32` へ落としてから保持・使用する（#2171 §4 の方針の延長）。
  `mu != 1` の判定は保持した `f32` 値で行う。`eta`／`mu` はスロットごとに持つ
  （PyTorch の state 構造と一致）。
- 要素ごとの更新のみで長軸縮約を持たないため、正規化統計・勾配縮約の `f64`
  アキュムレータ契約は該当しない。matmul 系 FMA 契約にも触れない
  （`weight_decay`・`param - eta*grad` は `f32::mul_add`。`adamax.rs` と同型）。
- `f32::clamp` は `min > max` や NaN 境界で panic するため、`Rprop::new` が境界
  （有限・`0 <= min <= max`）を検証して到達不能にする。本番経路に
  `unwrap`／`expect`／panic はない。
- 内部ループは既存 optimizer と共通化しない（鏡写しの別実装）。

## 5. ハイパーパラメータ検証域と PyTorch との意図的差分

| 対象 | 本実装 | PyTorch |
|---|---|---|
| Rprop `lr` | 有限かつ `>= 0` | `>= 0` |
| Rprop `eta_minus`／`eta_plus` | 有限かつ `0 < eta_minus < 1 < eta_plus`（`f32` 値で判定） | 同条件 |
| Rprop `step_size_min`／`step_size_max` | 有限かつ `0 <= min <= max` | 無検査 |
| ASGD `lr`／`weight_decay` | 有限かつ `>= 0` | `>= 0` |
| ASGD `lambd`／`alpha`／`t0` | 有限かつ `>= 0`（`lambd < 0` はべき乗の底が負になりうる。`alpha < 0` は `eta` が発散しうる） | 無検査 |
| `set_lr`（Rprop） | `config.lr` を書き換えるだけ。`lr` は初回 step の `step_size` 初期化にしか使われないため初回 step 後は更新値に影響しない（`step_size` の再スケールはしない） | `param_group["lr"]` 変更と同じ |
| `set_lr`（ASGD） | `config.lr` を書き換えるだけ。保持済み `eta` は次の step でそのまま使われ、新 `lr` はその step の終わりに計算する `eta` から効く（1 step 遅れ） | 同じ |
| step カウンタ | `u64`（overflow は `checked_add` で型付きエラー） | `float32` テンソル（2^24 step 超で増分が飽和）。再現しない |
| 非対応 | `maximize`／`foreach`／`capturable`／`differentiable`・複素数・sparse 勾配 | あり |

## 6. 検証

- **fixture**: 実 PyTorch 2.14.0+cpu 実行値
  （`crates/autodiff/tests/fixtures/{rprop,asgd}-pytorch-reference/`。生成条件・
  sha256 は各 README）。Rprop は 4 ケース（`default`・`etas`・`tight_bounds`・
  `lr_change`）× 10 step、ASGD は 5 ケース（`default`・`small_t0`・
  `weight_decay`・`all`・`lr_change`）× 10 step を全要素突合し、ASGD は各 step の
  `ax` も突合する。判定は統一複合判定（相対 1e-3 未満 または 絶対 1e-5 未満。
  `common::req2_close`。tolerance 定数は不変）。生成時 assert で Rprop の 3 分岐
  （同符号・符号反転・ゼロ）と両 clamp 境界、ASGD の `mu` 両分岐が通ることを確認。
- **エッジケース実測**（`rprop_matches_pytorch_reference_edge_cases`。`f32`
  ビットパターンで保存し、期待値が NaN なら NaN クラス一致）: PyTorch 2.14.0 で
  `torch.sign(NaN) = torch.sign(±0.0) = 0`。NaN 勾配の要素は当 step 更新されず
  `prev` に NaN が残り、次 step は `sign(NaN*g) = 0` → `factor = 1` として再開する
  （ソース読解どおり。実測で確認し、`rprop.rs` の doc とユニットテストで固定）。
  積がアンダーフローして 0 になる要素（`1e-30 * 1e-30`）は `factor = 1` で
  `sign(grad)` による更新が走る。
- **ユニットテスト**: ハイパーパラメータ拒否、shape 不一致・スロット数変化・
  スロット shape 変化の拒否、失敗 step で状態が変わらないこと、初回失敗後の別
  shape 再試行、`step_count` overflow、閉形式（t=1）、Rprop のゼロ勾配・符号反転・
  clamp・NaN、ASGD の `eta`／`mu` スケジュール・平均化分岐・`averaged_params`、
  `set_lr` の意味論。
- **決定性**（同一入力 2 回で bit 完全一致）と **MLP 収束**（`Linear`＋`Relu`＋
  `mse_loss` の 2 層 MLP・100 step で loss 半減。ASGD は `lr = 0.1` を使用）。
- optimizer は計算グラフ上の演算ではないため VJP は対象外（#2171 と同じ整理）。

## 7. GPU・`DeviceParamStore`

本実装はホスト `Tensor<f32>` だけを扱い、新規 `Op`・`BackendOps` メソッド・
カーネルを持たない。したがって CUDA（GB10）・Metal（M4 Max）で測るべき parity
対象が存在せず、`#[ignore]` テストも `docs/perf/logs/` の申し送りも作らない
（#2171 §7 と同じ判断）。補足: バックエンド別の学習ループ比較を parity の代用に
すると、Rprop は勾配の符号で更新が不連続に変わるため、既存 backward カーネルの
許容誤差内の差が符号反転として増幅され、optimizer 自体の正しさと無関係に判定が
揺れる。`DeviceParamStore` への結線・`compat::Optimizer` への variant 追加は対象外。

## 8. facade 公開形の推奨案（未承認）

推奨形は 1 つ: `crates/facade/src/optim.rs` に
`pub use fandhe_ai_autodiff::nn::optim::{Asgd, AsgdConfig, Rprop, RpropConfig};`
を追加する素の再エクスポート（`docs/facade-optimizer-promotion-decision.md` §4
案 A、および #2501 で採った形と同じ）。出荷済みの公開 API に対しては追加のみで
非破壊。**本節は推奨案の記録であり承認記録ではない**。承認依頼は #2677、公開の
実施は承認後の #2679。

## 9. 保留ガードと検出範囲

- doctest 足場 `OptimizerRpropAsgdHoldDoctestGuard`（`crates/facade/src/lib.rs`）:
  全 `pub mod` を glob import したスコープへローカル型 4 個
  （`Rprop`・`RpropConfig`・`Asgd`・`AsgdConfig`）を置いて関数シグネチャで参照する。
  facade が glob 可能な位置へ同名を公開すると名前解決が曖昧（E0659）になり
  コンパイルが失敗する。型名のみが対象のためメソッドプローブは置かない。
- `crates/facade/tests/api_surface.rs`: ドリフト検査 2 件
  （`optimizer_rprop_asgd_hold_doctest_globs_all_pub_modules`・
  `..._probe_body_matches_fixed_contract`）、facade src のトークン完全一致走査
  （`facade_does_not_reexport_or_declare_optimizer_rprop_asgd`。コメント・文字列
  リテラルを除く。`pub use`・独自宣言に加え非公開 `use`＋公開シグネチャや enum
  variant 経由の露出も拾う）とその自己テスト、workspace 全体の型宣言インベントリ
  （`workspace_declares_optimizer_rprop_asgd_types_only_in_allowed_locations`）。
- **検出範囲の契約**: 「doctest プローブが名前解決で触れる 4 名」と「facade src の
  トークン完全一致」に限る。マクロ生成や、内部クレート側で別名を作ってからの公開
  までは保証しない。`MIN_KNOWN_PROBE_BLOCKS` は下限値のため変更していない。
- 実効確認: 一時的に `crates/facade/src/optim.rs` へ
  `pub use fandhe_ai_autodiff::nn::optim::Rprop;` を足し、doctest（E0659）と
  ソース走査の両方が fail することを確認して元へ戻した（コミットしない）。

## 10. スコープ外

- facade 公開（#2677 承認 → #2679）と保留ガードの正ガード反転
- `ParamGroupStep`・`OptimizerStateDict` の実装（先例も別イシュー #2298・#2174 で
  後追い。ASGD の `eta`／`mu`／`ax`・Rprop の `prev`／`step_size` は既存
  `decode_state_dict` のフラグ構成に載らないため設計が要る）
- `compat::Optimizer` への variant 追加・`compile()`／`fit()` 統合、
  `DeviceParamStore` 結線、GPU カーネル
- `maximize`／`foreach`／`capturable`／`differentiable`・複素数・sparse 勾配
- `docs/compat-api-scope.md` 1 節の対象範囲表・`docs/compat-feature-gap.md` の
  判定変更、spec 改定
