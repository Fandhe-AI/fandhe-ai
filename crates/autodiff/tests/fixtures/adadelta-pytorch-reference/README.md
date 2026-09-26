# Adadelta PyTorch 参照値フィクスチャの出自

イシュー #2171（親 #2131）の
`tests/nn_optim_adadelta.rs::adadelta_matches_pytorch_reference` が参照
する固定フィクスチャ。`rmsprop-pytorch-reference/README.md`（イシュー
#1743 先例）と同じ方針で、生成条件・sha256 を記録した状態で JSON を
コミットする。CI は Python/PyTorch に依存せず、コミット済み JSON のみ
を読む（`.claude/rules/ci.md`「グローバル状態を汚す処理を workflow に
書かない」）。

## 生成条件

- **実 PyTorch 実行値**（アルゴリズム擬似コードからの再実装ではない）。
  `torch.__version__ == "2.14.0+cpu"`（`pip install --index-url
  https://download.pytorch.org/whl/cpu torch==2.14.0` で venv に導入し、
  `gen_reference.py` を 1 回実行して生成した）。
- **演算順のドリフト確認**: 導入した torch 2.14.0+cpu の
  `torch/optim/adadelta.py::_single_tensor_adadelta` を実装前に読み、
  演算順（`weight_decay != 0` なら `grad = grad.add(param,
  alpha=weight_decay)` →
  `square_avg.mul_(rho).addcmul_(grad, grad, value=1-rho)` →
  `std = square_avg.add(eps).sqrt_()`（**eps は sqrt の内側**。RMSprop
  とは逆）→ `delta = acc_delta.add(eps).sqrt_()` →
  `delta.div_(std).mul_(grad)` →
  `acc_delta.mul_(rho).addcmul_(delta, delta, value=1-rho)` →
  `param.add_(delta, alpha=-lr)`）と完全一致することを確認済み。
- 2 つの独立パラメータ（層としての関係は持たない単純な確認用テンソル）:
  `param_a`（shape `[2, 3]`）・`param_b`（shape `[4]`）。
  `random.Random(seed=20260926)` で初期値・10 step 分の勾配系列を
  固定生成し、`torch.optim.Adadelta` の `step()` を 10 回呼んで各 step
  後のパラメータ値を記録した。
- 4 ケース（`gen_reference.py::CASES`）:
  - `default`: `lr=1.0, rho=0.9, eps=1e-6, weight_decay=0.0`
    （`torch.optim.Adadelta` の既定値そのもの）
  - `lr_rho`: `lr=0.5, rho=0.8`（他は既定）
  - `weight_decay`: `weight_decay=0.1`（他は既定）
  - `all`: `lr=0.3, rho=0.7, eps=1e-5, weight_decay=0.05`
- 初期値・勾配系列はケース間で共通（差はハイパーパラメータのみにし、
  更新式そのものの一致検証に焦点を絞るため）。
- 生成された 10 step × 4 ケース × 全要素の中に NaN は含まれない。

## 再生成手順

```bash
python3 -m venv /path/to/venv
/path/to/venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.0
/path/to/venv/bin/python gen_reference.py   # このディレクトリで実行. adadelta_reference.json を上書きする
sha256sum adadelta_reference.json gen_reference.py   # 本 README の値と照合する
```

## sha256（改竄検知用）

```
54ddc7cb5df734e1f2e4fcb47a52bd25414b0ac8ab26b6ac53bd48c157d5cf3d  adadelta_reference.json
bd5a8b0479851e19235a0e7c1500065f869bbd71fa3c8ea3b5c78a6e5a333c28  gen_reference.py
```
