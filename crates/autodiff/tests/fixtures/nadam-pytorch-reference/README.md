# NAdam PyTorch 参照値フィクスチャの出自

イシュー #2171（親 #2131）の
`tests/nn_optim_nadam.rs::nadam_matches_pytorch_reference` が参照する
固定フィクスチャ。`rmsprop-pytorch-reference/README.md`（イシュー #1743
先例）と同じ方針で、生成条件・sha256 を記録した状態で JSON をコミット
する。CI は Python/PyTorch に依存せず、コミット済み JSON のみを読む
（`.claude/rules/ci.md`「グローバル状態を汚す処理を workflow に書か
ない」）。

## 生成条件

- **実 PyTorch 実行値**。`torch.__version__ == "2.14.0+cpu"`（`pip
  install --index-url https://download.pytorch.org/whl/cpu
  torch==2.14.0` で venv に導入し、`gen_reference.py` を 1 回実行して
  生成した）。
- **演算順のドリフト確認**: 導入した torch 2.14.0+cpu の
  `torch/optim/nadam.py::_single_tensor_nadam` を実装前に読み、演算順
  （`bc2 = 1 - beta2**step` → weight decay（`decoupled_weight_decay`
  なら `param *= 1 - lr*wd`、それ以外は `grad += wd*param`）→
  `mu = beta1*(1 - 0.5*0.96**(step*momentum_decay))`・
  `mu_next = beta1*(1 - 0.5*0.96**((step+1)*momentum_decay))` →
  `mu_product *= mu` → `exp_avg.lerp_(grad, 1-beta1)` →
  `exp_avg_sq = beta2*v + (1-beta2)*grad^2` →
  `denom = sqrt(exp_avg_sq / bc2) + eps` →
  `param.addcdiv_(grad, denom, value=-lr*(1-mu)/(1-mu_product))` →
  `param.addcdiv_(exp_avg, denom, value=-lr*mu_next/(1-mu_product*mu_next))`）
  と完全一致することを確認済み。`mu_product` は `_get_scalar_dtype()`
  （既定 `torch.float32`）のパラメータごとの state だが、同一 step・
  同一ハイパーパラメータでは全スロットで同一値になる（`beta1`／
  `momentum_decay`／`step` のみに依存するため）。
- 2 つの独立パラメータ: `param_a`（shape `[2, 3]`）・`param_b`
  （shape `[4]`）。`random.Random(seed=20260926)` で初期値・10 step 分
  の勾配系列を固定生成し、`torch.optim.NAdam` の `step()` を 10 回
  呼んで各 step 後のパラメータ値を記録した。
- 5 ケース（`gen_reference.py::CASES`）:
  - `default`: `lr=2e-3, beta1=0.9, beta2=0.999, eps=1e-8,
    weight_decay=0.0, momentum_decay=4e-3, decoupled_weight_decay=False`
    （`torch.optim.NAdam` の既定値そのもの）
  - `momentum_decay`: `momentum_decay=0.05`（他は既定）
  - `weight_decay`: `weight_decay=0.1`（coupled。他は既定）
  - `decoupled_weight_decay`: `weight_decay=0.1,
    decoupled_weight_decay=True`（他は既定）
  - `all`: `lr=0.01, beta1=0.8, beta2=0.99, eps=1e-6, weight_decay=0.02,
    momentum_decay=0.01, decoupled_weight_decay=True`
- 初期値・勾配系列はケース間で共通。生成された 10 step × 5 ケース ×
  全要素の中に NaN は含まれない。

## 再生成手順

```bash
python3 -m venv /path/to/venv
/path/to/venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.0
/path/to/venv/bin/python gen_reference.py   # このディレクトリで実行. nadam_reference.json を上書きする
sha256sum nadam_reference.json gen_reference.py   # 本 README の値と照合する
```

## sha256（改竄検知用）

```
d498ca4c0d28ef6d968717130a6afc2914fbdff883e7c40ae2510f2d72236682  nadam_reference.json
29d14734251bc3e52ac812d67581e7111afc6e3e5a6ca33ad3767861051592a9  gen_reference.py
```
