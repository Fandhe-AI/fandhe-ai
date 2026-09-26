# Adamax PyTorch 参照値フィクスチャの出自

イシュー #2171（親 #2131）の
`tests/nn_optim_adamax.rs::adamax_matches_pytorch_reference` が参照する
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
  `torch/optim/adamax.py::_single_tensor_adamax` を実装前に読み、演算順
  （`weight_decay != 0` なら `grad = grad.add(param, alpha=weight_decay)`
  → `exp_avg.lerp_(grad, 1-beta1)` →
  `exp_inf = max(exp_inf*beta2, |grad|+eps)`（`torch.maximum` は
  NaN 伝播）→ `clr = lr / (1 - beta1**step)` →
  `param.addcdiv_(exp_avg, exp_inf, value=-clr)`）と完全一致すること
  を確認済み。
- 2 つの独立パラメータ: `param_a`（shape `[2, 3]`）・`param_b`
  （shape `[4]`）。`random.Random(seed=20260926)` で初期値・10 step 分
  の勾配系列を固定生成し、`torch.optim.Adamax` の `step()` を 10 回
  呼んで各 step 後のパラメータ値を記録した。
- 4 ケース（`gen_reference.py::CASES`）:
  - `default`: `lr=2e-3, beta1=0.9, beta2=0.999, eps=1e-8,
    weight_decay=0.0`（`torch.optim.Adamax` の既定値そのもの）
  - `betas`: `beta1=0.8, beta2=0.99`（他は既定）
  - `weight_decay`: `weight_decay=0.1`（他は既定）
  - `all`: `lr=0.05, beta1=0.7, beta2=0.95, eps=1e-6, weight_decay=0.02`
- 初期値・勾配系列はケース間で共通。生成された 10 step × 4 ケース ×
  全要素の中に NaN は含まれない。

## 再生成手順

```bash
python3 -m venv /path/to/venv
/path/to/venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.0
/path/to/venv/bin/python gen_reference.py   # このディレクトリで実行. adamax_reference.json を上書きする
sha256sum adamax_reference.json gen_reference.py   # 本 README の値と照合する
```

## sha256（改竄検知用）

```
cb0651da22bc939da5cbb8e2f2c9dcf4063a1c8c631f6a31a90a4702de9cd09b  adamax_reference.json
538d2e581d504713ad028800abd80901b10e41da1df2d178828a218779c7a815  gen_reference.py
```
