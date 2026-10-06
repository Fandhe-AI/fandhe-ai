# Adafactor・Lion の設計判断記録

イシュー #2656（親 #2654「不足 optimizer」・ルート #2499 Phase 4）。
`docs/autodiff-optimizer-rprop-asgd-decision.md`（#2655）と同型。

## 1. 背景

`fandhe_ai_autodiff::nn::optim` には Adafactor（Shazeer & Stern, 2018。
factored second moment）と Lion（Chen et al., 2023。符号ベース）がなかった。
本イシューはこの 2 種を、既存 optimizer と同じ「`Tape`／`Var`／`BackendOps` に
依存しないホスト側の値型・純関数」として内部クレートへ追加する。facade への
公開は行わず、保留ガードで機械固定する（公開は承認依頼 #2677 → 承認後の
#2679）。

## 2. 公開 API とシグネチャ

`crates/autodiff/src/nn/optim/{adafactor,lion}.rs`。`mod.rs` から
`Adafactor`／`AdafactorConfig`／`Lion`／`LionConfig` を `pub use` する。呼び出し
規約は既存 11 種と同じ:

```rust
pub struct AdafactorConfig { pub lr: f32, pub beta2_decay: f32, pub eps1: f32,
                             pub eps2: f32, pub d: f32, pub weight_decay: f32 }
// Default: lr=1e-2, beta2_decay=-0.8, eps1=f32::EPSILON, eps2=1e-3, d=1.0, weight_decay=0.0
pub struct LionConfig { pub lr: f32, pub beta1: f32, pub beta2: f32, pub weight_decay: f32 }
// Default: lr=1e-4, beta1=0.9, beta2=0.99, weight_decay=0.0

impl Adafactor / Lion {
    pub fn new(config) -> Result<Self, AutodiffError>;
    pub fn config(&self) -> &Config;
    pub fn set_lr(&mut self, new_lr: f32) -> Result<(), AutodiffError>;
    pub fn step_count(&self) -> u64;
    pub fn step(&mut self, params_and_grads: &[(&Tensor<f32>, &Tensor<f32>)])
        -> Result<Vec<Tensor<f32>>, AutodiffError>;
}
```

`step()` は `rprop.rs` と同じ 2 段構成（副作用なしの検証 → 状態変更）で、
`step_count.checked_add(1)` を遅延初期化より前に確定させるアトミック性を踏襲
する。Adafactor の `step_count` は PyTorch ではパラメータごとの `float32`
テンソルだが全スロットで同値のため optimizer 全体で 1 個とする（`NAdam` の
`mu_product` と同じ理由）。

## 3. 演算順

**Adafactor**（PyTorch 2.14.0+cpu の `torch/optim/_adafactor.py::
_single_tensor_adafactor` を実装前に読んで確認。`t` は更新後の step 数）

```
w     = t^beta2_decay                 # 「1 - beta2_t」。lerp の重み
rho   = min(lr, 1/sqrt(t))
alpha = max(eps2, RMS(param)) * rho   # weight decay より前の param
weight_decay != 0 なら param *= (1 - lr*weight_decay)       # decoupled

rank >= 2（末尾 2 次元 [n, m] で因子分解。先頭次元はバッチ面として独立）:
  row_var = lerp(row_var, norm(g[..., i, :])^2 / m, w)
  col_var = lerp(col_var, norm(g[..., :, j])^2 / n, w)
  v       = row_var * col_var / clamp_min(mean_i(row_var), eps1)   # 積 → 除算の順
rank <= 1:
  variance = lerp(variance, g*g, w);  v = variance

u      = rsqrt(clamp_min(v, eps1^2)) * g
denom  = max(1, ||u|| / (sqrt(numel) * d))
param += u * (-alpha / denom)
```

状態は `step_count` と、スロットごとに rank >= 2 なら `row_var`（`B*n` 要素）・
`col_var`（`B*m` 要素。`B` は先頭次元の積）、rank 0・1 なら `variance`。いずれも
0 初期化で、初回 `step()` の `param` の rank で遅延確定する。

**Lion**（公式参照実装 google/automl `lion_pytorch.py` と同じ。状態は
`exp_avg`〈0 初期化〉）

```
param  *= (1 - lr*weight_decay)               # decoupled。wd=0 でも常に適用
update  = exp_avg*beta1 + grad*(1-beta1)      # 補間係数は beta1（更新前の exp_avg）
param  += sign(update) * (-lr)
exp_avg = exp_avg*beta2 + grad*(1-beta2)      # 保存する移動平均の係数は beta2
```

## 4. 数値方針

- スカラー係数（Adafactor の `w`・`rho`・`alpha`・`denom`・`1 - lr*wd`・
  `-alpha/denom`）は `f64` で計算し使用直前に 1 回だけ `f32` へ落とす（#2171
  §4 の延長）。
- **Adafactor は長軸縮約を持つ**（これまでの optimizer は持たなかった）ため
  `.claude/rules/coding-rust.md` の正規化統計の契約に従い `f64` アキュムレータを
  使う: ノルムは要素を先に `f64` へ昇格してから二乗して index 順に蓄積し、
  `f64` で `sqrt`、`f32` へ 1 回 downcast（torch が `f32` テンソルの norm を
  `.item()` で取り出すのと同じ精度段。先例は `lamb.rs`）。`row_mean`／
  `col_mean` は torch と同じ「norm（`f32`）→ 二乗 → size で除算」を `f32` で、
  `mean_i(row_var)` は `f64` で蓄積して `f32` へ落とす。
- `lerp` は `adamax.rs`／`rmsprop.rs` と同じ 2 分岐形。既定 `beta2_decay = -0.8`
  では `t=1,2` が終点基準、`t>=3` が始点基準で両分岐とも fixture が通る。
- **NaN 意味論の取り違えを避ける**: Python 組み込みの `max(a, x)` は `x` が NaN
  のとき `a` を返す（`py_max`）。torch の `clamp_min` は NaN を伝播する
  （`clamp_min_nan`）。`f32::max` は NaN を捨てるため使わない。
- **Lion は `f32::signum` を使わない**（torch の `sign` は `±0.0` と NaN を 0 に
  する。`lion.rs::torch_sign`。`rprop.rs` の同名関数とは意図的な重複で共通化
  しない）。**`update` の計算には `f32::mul_add` も `lerp` も使わない**: torch は
  `exp_avg*beta1` と `grad*(1-beta1)` を別テンソル演算で丸めてから加算する。
  `sign` は不連続で、相殺付近の 1 ulp 差がパラメータ差 `2*lr` の符号反転へ増幅
  されるため、ここだけは torch と同じ「積 2 回 → 加算 1 回」の `f32` 演算列にする。
- **非有限入力は PyTorch 準拠で伝播させ、黙って握りつぶさず契約化する**。
  Adafactor: NaN／inf 勾配は `row_var`／`col_var`／`variance` を汚染し、以後その
  スロット（因子分解では行・列を共有する要素）は回復しない。Lion: NaN 勾配の
  要素は `sign = 0` で当 step は動かず、`exp_avg` の NaN により以後 weight decay
  以外で動かない。fixture の `edge` ブロック（`f32` ビットパターン）で実測固定
  した。検討した代替（`lamb.rs` 型の「非有限入力は `Err`」の fail-closed）は
  PyTorch 実行値との突合という本ツリーの方式から外れるため採らなかった。
  学習ループ側では `nn/optim/mod.rs` の適用順序契約どおり非有限検出（`amp`）・
  `clip` が optimizer の前段にある。
- Adafactor の `param *= 係数` は通常の積、`param.add_(update, alpha=…)` は
  `asgd.rs` と同じ写し方（`f32::mul_add`）。matmul 系 FMA 契約には触れない。

## 5. ハイパーパラメータ検証域と PyTorch との意図的差分

| 対象 | 本実装 | PyTorch 2.14.0 |
|---|---|---|
| Adafactor `lr` | 有限かつ `>= 0` | `>= 0` |
| `beta2_decay` | 有限かつ `<= 0` | `<= 0` |
| `eps1` | 有限かつ `> 0`（0 は `rsqrt(0) = inf` を生むため拒否） | `None` または `>= 0` |
| `eps2` | 有限かつ `>= 0` | `>= 0` |
| `d` | 有限かつ `>= 1` | `>= 1` |
| Adafactor `weight_decay` | 有限かつ `>= 0` | `>= 0` |
| 要素数 0 のスロット | `step()` が状態変更前に `InvalidArgument` で拒否 | `ZeroDivisionError` |
| rank 0 | `variance` 経路で受理 | 同じ（`grad.dim() > 1` が偽） |
| `set_lr` | `config.lr` を書き換えるだけ。次の step から即時に `rho` と weight decay へ効く | `param_group["lr"]` 変更と同じ |
| step カウンタ | `u64`（overflow は `checked_add` で型付きエラー） | `float32` テンソル（2^24 超で飽和）。再現しない |
| Lion `lr`／`weight_decay` | 有限かつ `>= 0` | （PyTorch に Lion なし）公式実装は `lr >= 0` のみ、`weight_decay` 無検査 |
| Lion `beta1`／`beta2` | `[0, 1)` | 公式実装も `0 <= beta < 1` |
| 非対応 | `maximize`／`foreach`／`capturable`／`differentiable`・複素数・sparse 勾配 | あり |

## 6. 検証

- **Adafactor の fixture**: 実 `torch.optim.Adafactor` 2.14.0+cpu 実行値
  （`crates/autodiff/tests/fixtures/adafactor-pytorch-reference/`。生成条件・
  sha256 は README）。4 パラメータ（rank 2・1・3・0）× 7 ケース（`default`・
  `relative_step`・`weight_decay`・`beta2_decay`・`eps`・`clip_d`・`lr_change`）
  × 10 step を全要素突合。判定は統一複合判定（相対 1e-3 未満 または 絶対 1e-5
  未満。`common::req2_close`。tolerance 定数は不変）。生成時 assert で lerp 重みの
  両分岐・`rho` の両分岐・`alpha` の両分岐・両 clamp・`denom` の 1／`>1` の両方が
  通ることを確認。
- **Lion の参照値の出自（受け入れ基準の字義からの逸脱）**: **`torch.optim` 2.14.0
  に Lion は存在しない**。fixture は公式参照実装（google/automl
  `lion_pytorch.py`。Apache-2.0）の更新則を `gen_reference.py` に自前記述し、実
  PyTorch 2.14.0+cpu のテンソル演算で実行した値であり、`torch.optim` の実装との
  突合ではない。独立性の補強として (a) テスト内の論文の式どおりの独立 `f64` 参照
  実装（`lion_matches_independent_f64_reference`。`torch.optim` に無い LAMB の
  `nn_optim_lamb.rs` と同じ手当て）、(b) ユニットテストの閉形式（t=1・`beta1`／
  `beta2` の役割取り違えを検出する t=2 の手計算）を置く。サードパーティ
  `lion-pytorch` は導入していない。生成時 assert で `update` が厳密 0 または
  `|update| >= 1e-3`（符号が丸め差で反転しない余裕）であることと `sign` の 3 値を
  確認している。
- **エッジケース実測**（`*_matches_pytorch_reference_edge_cases`。ビット
  パターンで保存し、NaN は NaN クラス一致、inf は符号込み一致）: §4 の非有限
  契約を実測で固定。
- **ユニットテスト**: ハイパーパラメータ拒否、shape 不一致・スロット数変化・
  スロット shape 変化・要素数 0 の拒否、失敗 step で状態不変、初回失敗後の別
  shape 再試行、`step_count` overflow、閉形式（t=1。Adafactor は rank 1・rank 2）、
  ゼロ勾配、rank 0・rank 3 の面独立、`eps2`／`rho`／weight decay 順序、NaN 伝播
  ヘルパ、`set_lr` の即時反映。
- **決定性**（同一入力 2 回で bit 完全一致）と **MLP 収束**（`Linear`＋`Relu`＋
  `mse_loss` の 2 層 MLP・100 step で loss 半減。Adafactor は `lr = 0.1`、Lion は
  既定 `lr = 1e-4` が 100 step では小さいため `lr = 1e-2` を使用。判定の形は不変）。
- optimizer は計算グラフ上の演算ではないため VJP は対象外（#2171・#2655 と
  同じ整理）。

## 7. GPU・`DeviceParamStore`

本実装はホスト `Tensor<f32>` だけを扱い、新規 `Op`・`BackendOps` メソッド・
カーネルを持たない。したがって CUDA（GB10）・Metal（M4 Max）で測るべき parity
対象が構造上存在せず、`#[ignore]` テストも `docs/perf/logs/` の申し送りも作らない
（#2171 §7・#2655 §7 と同じ判断）。`DeviceParamStore` への結線・
`compat::Optimizer` への variant 追加は対象外。

## 8. facade 公開形の推奨案（未承認）

推奨形は 1 つ: `crates/facade/src/optim.rs` に
`pub use fandhe_ai_autodiff::nn::optim::{Adafactor, AdafactorConfig};` と
`pub use fandhe_ai_autodiff::nn::optim::{Lion, LionConfig};` を追加する素の
再エクスポート（`docs/facade-optimizer-promotion-decision.md` §4 案 A、#2501・
#2655 と同じ形）。出荷済みの公開 API に対しては追加のみで非破壊。**本節は推奨案
の記録であり承認記録ではない**。承認依頼は #2677、公開の実施は承認後の #2679。

## 9. 保留ガードと検出範囲

- doctest 足場 `OptimizerAdafactorLionHoldDoctestGuard`
  （`crates/facade/src/lib.rs`）: 全 `pub mod` を glob import したスコープへ
  ローカル型 4 個（`Adafactor`・`AdafactorConfig`・`Lion`・`LionConfig`）を置いて
  関数シグネチャで参照する。facade が glob 可能な位置へ同名を公開すると名前解決が
  曖昧（E0659）になりコンパイルが失敗する。
- `crates/facade/tests/api_surface.rs`: ドリフト検査 2 件
  （`optimizer_adafactor_lion_hold_doctest_globs_all_pub_modules`・
  `..._probe_body_matches_fixed_contract`）、facade src のトークン完全一致走査
  （`facade_does_not_reexport_or_declare_optimizer_adafactor_lion`）とその自己
  テスト、workspace 全体の型宣言インベントリ
  （`workspace_declares_optimizer_adafactor_lion_types_only_in_allowed_locations`）。
- **検出範囲の契約**: 「doctest プローブが名前解決で触れる 4 名」と「facade src の
  トークン完全一致」に限る。マクロ生成や、内部クレート側で別名を作ってからの公開
  までは保証しない。`MIN_KNOWN_PROBE_BLOCKS` は下限値のため変更していない。
- 実効確認: 一時的に `crates/facade/src/optim.rs` へ
  `pub use fandhe_ai_autodiff::nn::optim::Lion;` を足し、doctest（E0659）と
  ソース走査の両方が fail することを確認して元へ戻した（コミットしない）。

## 10. スコープ外

- facade 公開（#2677 承認 → #2679）と保留ガードの正ガード反転
- `ParamGroupStep`・`OptimizerStateDict` の実装（Adafactor の `row_var`／
  `col_var`／`variance` は既存 `decode_state_dict` のフラグ構成に載らず設計が要る。
  別イシュー候補。起票はユーザー承認が要るため未実施）
- `compat::Optimizer` への variant 追加・`compile()`／`fit()` 統合、
  `DeviceParamStore` 結線、GPU カーネル
- `maximize`／`foreach`／`capturable`／`differentiable`・複素数・sparse 勾配
- `docs/compat-api-scope.md` 1 節の対象範囲表・`docs/compat-feature-gap.md` の
  判定変更、spec 改定
