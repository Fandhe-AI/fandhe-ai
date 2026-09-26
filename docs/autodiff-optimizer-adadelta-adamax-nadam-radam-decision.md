# Adadelta・Adamax・NAdam・RAdam の設計判断記録

イシュー #2171（親 #2131「PyTorch／TF 置き換えの API 網羅（対応表の行内深掘り）」）。

## 1. 背景

`docs/compat-feature-gap.md` §2.9 optimizer 表の対応漏れのうち、PyTorch
`torch.optim.{Adadelta, Adamax, NAdam, RAdam}` に相当する 4 種が
`fandhe_ai_autodiff::nn::optim` に未実装だった（`Sgd`・`AdamW`・`Adam`・
`RmsProp`・`Adagrad`・`LAMB` の 6 種のみ）。本イシューはこの 4 種を追加
する。

## 2. 公開 API とシグネチャ（イシュー記述との差異）

イシュー本文は `step(lr, grads, params) -> new_params` という自由関数
形を示唆していたが、既存の `AdamW`・`RmsProp`・`Adagrad`・`Adam`・`LAMB`
はいずれも次の形に統一されている:

```rust
pub struct XxxConfig { pub lr: f32, /* 他のハイパーパラメータ */ }
pub struct Xxx { config: XxxConfig, step_count: u64, states: Vec<SlotState> }
impl Xxx {
    pub fn new(config: XxxConfig) -> Result<Xxx, AutodiffError>;
    pub fn config(&self) -> &XxxConfig;
    pub fn set_lr(&mut self, new_lr: f32) -> Result<(), AutodiffError>;
    pub fn step_count(&self) -> u64;
    pub fn step(&mut self, params_and_grads: &[(&Tensor<f32>, &Tensor<f32>)])
        -> Result<Vec<Tensor<f32>>, AutodiffError>;
}
```

本イシューもこの既存契約に揃えた（`lr` は `config` が持ち、`step` の
シグネチャは `AdamW::step` と同一）。理由: 既存 6 種との一貫性を崩すと
`compat::Sequential` の手動学習ループ・将来の `compile()` 統合
（#2170 系）が optimizer ごとに異なる呼び出し規約を扱う必要が生じる。

イシューは参照先として `docs/compat-api-scope.md` §1.3 を挙げていたが、
optimizer 行は実際には §1.2 の表（`optimizer（Adam／RMSprop／Adagrad／
LAMB）` 行）にある。§1.3 に optimizer 行は存在しないため、§1.2 の既存
行へ追記した（§5 参照）。

## 3. 4 種の演算順（torch 2.14.0+cpu のソースで確認したこと）

実装前に、導入した PyTorch 2.14.0+cpu の
`torch/optim/{adadelta,adamax,nadam,radam}.py::_single_tensor_*` を読み、
以下の演算順と完全一致することを確認した（各 fixture README にも
記録済み）。

### 3.1 Adadelta（Zeiler, 2012）

既定値: `lr=1.0, rho=0.9, eps=1e-6, weight_decay=0.0`。

```
weight_decay != 0 なら grad += weight_decay * param
square_avg = rho * square_avg + (1-rho) * grad^2
std   = sqrt(square_avg + eps)          # eps は sqrt の内側（RmsProp と逆）
delta = sqrt(acc_delta + eps) / std * grad
acc_delta = rho * acc_delta + (1-rho) * delta^2
param -= lr * delta
```

### 3.2 Adamax（Kingma & Ba, 2015 §7.1）

既定値: `lr=2e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0`。

```
weight_decay != 0 なら grad += weight_decay * param
exp_avg = lerp(exp_avg, grad, 1-beta1)
exp_inf = max(exp_inf*beta2, |grad|+eps)   # torch.maximum は NaN 伝播版
clr = lr / (1 - beta1^step)
param -= clr * exp_avg / exp_inf
```

`torch.maximum` は「どちらかが NaN なら NaN を返す」意味論であり、
`f32::max`（NaN を無視する）とは異なる。自前実装で NaN 伝播版 max を
再現した（`exp_inf_update_propagates_nan` テストで固定）。

### 3.3 NAdam（Dozat, 2016）

既定値: `lr=2e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0,
momentum_decay=4e-3, decoupled_weight_decay=false`。

```
bc2 = 1 - beta2^step
weight decay: decoupled なら param *= 1-lr*wd、それ以外は grad += wd*param
mu      = beta1 * (1 - 0.5 * 0.96^(step*momentum_decay))
mu_next = beta1 * (1 - 0.5 * 0.96^((step+1)*momentum_decay))
mu_product *= mu
exp_avg    = lerp(exp_avg, grad, 1-beta1)
exp_avg_sq = beta2*exp_avg_sq + (1-beta2)*grad^2
denom = sqrt(exp_avg_sq / bc2) + eps
param += (-lr*(1-mu)/(1-mu_product))               * grad    / denom
param += (-lr*mu_next/(1-mu_product*mu_next))      * exp_avg / denom
```

2 つの `addcdiv` は別々に適用する（`grad` 由来の項と `exp_avg` 由来の
項）。

### 3.4 RAdam（Liu et al., 2019）

既定値: `lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0,
decoupled_weight_decay=false`。

```
weight decay: decoupled なら param *= 1-lr*wd、それ以外は grad += wd*param
exp_avg    = lerp(exp_avg, grad, 1-beta1)
exp_avg_sq = beta2*exp_avg_sq + (1-beta2)*grad^2
bc1 = 1 - beta1^step
bc2 = 1 - beta2^step
bias_corrected_exp_avg = exp_avg / bc1
rho_inf = 2/(1-beta2) - 1
rho_t   = rho_inf - 2*step*beta2^step/bc2
if rho_t > 5.0:
    rect        = sqrt((rho_t-4)(rho_t-2)*rho_inf / ((rho_inf-4)(rho_inf-2)*rho_t))
    adaptive_lr = sqrt(bc2) / (sqrt(exp_avg_sq) + eps)
    param -= bias_corrected_exp_avg * lr * adaptive_lr * rect   # 左結合の評価順
else:
    param -= bias_corrected_exp_avg * lr
```

`rect` の分母は `rho_t > 5` の分岐でしか評価されないためゼロ除算は
起きない。

## 4. 数値方針

- **係数は `f64` で計算し最後に `f32` へ落とす**（PyTorch の Python
  float 演算を再現する方針。`AdamW::step` の `step_size`／
  `bias_correction2_sqrt` と同じ方針の延長）。
- **べき乗は `powf`（実数指数）で統一する**。`AdamW::step` の逐次積
  （`beta.powi` 相当の `beta1_pow_t *= beta1`）とは異なり、NAdam の
  `0.96^(step*momentum_decay)` と RAdam の `beta2^step`（`rho_t` 内）
  は非整数指数を要するため。`Adamax`・`RAdam` の `beta1_pow_t`／
  `beta2_pow_t` 自体は既存の逐次積方式（`f64` の掛け算を step 回繰り
  返す）を維持し、`powf` は momentum cache・`rho_t` 等の追加の実数指数
  項にのみ使う。
- **`lerp` は `rmsprop.rs` centered 分岐と同じ 2 分岐形**
  （`|weight| < 0.5` なら始点基準、それ以外は終点基準）で計算する。
  `weight = 1 - beta1` は検証済み範囲 `[0, 1)` の `beta1` から
  `(0, 1]` に収まるため NaN にならない。
- **Adamax の `exp_inf` 更新は NaN 伝播版 max を自前実装する**
  （§3.2 参照）。
- **NAdam の `mu_product` は optimizer 全体で `f32` 1 個として持つ**。
  PyTorch は `_get_scalar_dtype()`（既定 `torch.float32`）のパラメータ
  ごとの state として持つが、`mu`（`beta1`／`momentum_decay`／`step`
  のみに依存し `param`／`grad` を参照しない）の逐次積であるため、
  同一 optimizer インスタンス内の全スロットで値が一致する。この事実
  を利用して `NAdam` 構造体全体で 1 個の `f32` を持つ（`nadam.rs`
  モジュール冒頭 doc 参照）。
- 正規化統計・勾配長軸縮約の `f64` アキュムレータ統一契約
  （`.claude/rules/coding-rust.md`）は本実装のスカラー係数計算には
  該当しない（要素ごとの縮約〈`sum`／`norm` 等〉を行わないため）。

## 5. ハイパーパラメータの検証域（PyTorch より厳しい点）

- `lr >= 0`（有限）、`weight_decay >= 0`（有限）: 既存 6 種と同一。
- `eps > 0`（有限）: 0 を許すと分母がゼロになるため。
- `beta1`／`beta2`／`rho` は `[0, 1)`。PyTorch は範囲チェックをほぼ
  行わないが、`rho = 1`（Adadelta）・`beta1 = 1`／`beta2 = 1` は
  `square_avg`／`exp_avg`／`exp_avg_sq` が一切更新されない退化ケース
  のため、既存 `RmsProp::new` の `alpha` 検査と同じ理由で意図的に
  拒否する。
- NAdam の `momentum_decay >= 0`（有限）。

## 6. 検証

- **PyTorch 実行値 fixture との統一複合判定**（相対誤差 1e-3 未満
  または絶対誤差 1e-5 未満。`tests/common::req2_close`）: 実
  PyTorch 2.14.0+cpu を一時 venv に導入して生成した固定 JSON
  （`tests/fixtures/{adadelta,adamax,nadam,radam}-pytorch-reference/`。
  各 README に生成条件・sha256 を記録）と 10 step × 4〜5 ケース ×
  全要素を突合する（`tests/nn_optim_{adadelta,adamax,nadam,radam}.rs`）。
  全ケース・全 step・全要素で判定成立を確認済み。
- **RAdam の分岐境界**: fixture 生成スクリプトが各 step の `rho_t` を
  計算し、10 step 以内に rectified／non-rectified 両分岐を通ること、
  どの step の `rho_t` も 5.0 から `1e-3` 以上離れていることを生成時
  に assert する（実測: 全ケースで step 5→6 に境界がある。
  `radam-pytorch-reference/README.md` 参照）。
- **閉形式（t=1）一致**: 各 optimizer のユニットテストで固定。
- **決定性**: 同一入力で 2 回 `step()` を呼び bit 完全一致することを
  確認（`{adadelta,adamax,nadam,radam}_step_is_deterministic`）。
- **MLP 収束**: `Linear`＋`Relu`＋`MseLoss` の 2 層 MLP で 100 step
  後に loss が半減することを確認。RAdam のみ既定 `lr=1e-3` では
  non-rectified 分岐（`rho_t <= 5`）の間ほとんど更新が進まず、
  `lr=0.01` へ経験的に調整した（`rmsprop.rs` 冒頭コメントと同じ方針。
  収束判定の**形**は変更していない）。
- **facade 手動ループの bit 完全一致**（`crates/facade/tests/
  compat_sequential_optim_ext.rs`）: `compat::Sequential` の手動 step
  ループと `nn::Linear` 直組みの手動ループが、4 種すべてで 10 step
  にわたり loss・最終パラメータともビット完全一致することを確認。

## 7. GPU／`DeviceParamStore`

`crate::optim::device_store::DeviceParamStore::step` は
`BackendOps::sgd_step_device` 専用のデバイス常駐更新経路であり、本
イシューでは対応する `BackendOps` メソッドを追加していないため、
4 種とも **`DeviceParamStore` 非対応**（ホスト `Tensor<f32>` を介した
`step()` のみ）。

本実装は `Tape`／`Var`／`BackendOps` に一切依存しないホスト側の値型・
純関数であり、GPU カーネルを一切持たない。したがって CUDA（DGX Spark
GB10）・Metal 実機での parity テストは**構造上 N/A**であり、
`docs/perf/logs` への Mac／GB10 実測申し送りは作成しない。

## 8. 承認事項（本 PR では実施しない）

- **facade 純再エクスポート**（`docs/compat-api-scope.md` §5 経路 2）:
  `fandhe_ai::optim::{Adadelta, AdadeltaConfig, Adamax, AdamaxConfig,
  NAdam, NAdamConfig, RAdam, RAdamConfig}` の `crates/facade/src/
  optim.rs` への追加。`AdamW`／`Adam`／`RmsProp`／`Adagrad`／`LAMB`
  はすでにこの経路で公開済みだが、本 4 種は未承認のため保留する。
  保留固定は `crates/facade/src/lib.rs::OptimizerExtHoldDoctestGuard`
  （正のプローブ 1 ブロック方式。型名のみが対象で inherent メソッド
  追加を伴わないため trait プローブは不要）と `crates/facade/tests/
  api_surface.rs` の 3 テスト（doctest ドリフト検査 2 件・ソース
  走査 1 件＋自己テスト）で多層固定する。

承認後の作業: `optim.rs` へ `pub use fandhe_ai_autodiff::nn::optim::
{Adadelta, AdadeltaConfig, Adamax, AdamaxConfig, NAdam, NAdamConfig,
RAdam, RAdamConfig};` を追加、`api_surface.rs` の
`optim_module_reexports_exactly_expected_surface` 期待集合・
`optim_types_are_reachable_via_facade_only` への追加、
`OptimizerExtHoldDoctestGuard`・対応する 3 テストの削除。

## 9. スコープ外

- `compile()`（`compat::Optimizer` enum）への統合（#2170 系）。
- param groups（#2173。タイトルは "param groups" であり、実装計画
  立案時に一時誤認していた「CUDA／Metal 常駐実装」ではない）。
- `DeviceParamStore` への結線（§7 参照）。
- `maximize`／`foreach`／`capturable`／`differentiable`。
- 複素数パラメータ。
