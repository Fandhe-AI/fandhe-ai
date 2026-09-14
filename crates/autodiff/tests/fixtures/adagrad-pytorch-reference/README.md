# Adagrad PyTorch 参照値フィクスチャの出自

イシュー #1743（親 #1610「optimizer（Adam／RMSprop／Adagrad／LAMB）」）の
`tests/nn_optim_adagrad.rs::adagrad_matches_pytorch_reference` が参照する
固定フィクスチャ。`adamw-pytorch-reference/README.md`（イシュー #194
先例）と同じ方針で、生成条件・sha256 を記録した状態で JSON をコミット
する。CI は Python/PyTorch に依存せず、コミット済み JSON のみを
読む（`.claude/rules/ci.md`「グローバル状態を汚す処理を workflow に
書かない」）。

## 生成条件

- **実 PyTorch 実行値**（アルゴリズム擬似コードからの再実装ではない）。
  `torch.__version__ == "2.14.0+cpu"`（`pip install --index-url
  https://download.pytorch.org/whl/cpu torch` で venv に導入し、
  `gen_reference.py` を 1 回実行して生成した）。
- **演算順のドリフト確認**: 導入した torch 2.14.0+cpu の
  `torch/optim/adagrad.py::_single_tensor_adagrad` を実装前に読み、
  実装計画 §3.3 で確認した演算順（`step += 1` → `weight_decay != 0`
  のとき `grad = grad.add(param, alpha=weight_decay)` →
  `clr = lr / (1 + (step - 1) * lr_decay)` →
  `state_sum.addcmul_(grad, grad, value=1)` →
  `std = state_sum.sqrt().add_(eps)` →
  `param.addcdiv_(grad, std, value=-clr)`）と完全一致することを確認済み。
  `initial_accumulator_value` は `state["sum"] = torch.full_like(p,
  initial_accumulator_value)` で `state_sum` の初期値として使われる
  （0 スタートではない）ことも確認済み。
- 2 つの独立パラメータ（層としての関係は持たない単純な確認用テンソル）:
  `param_a`（shape `[2, 3]`）・`param_b`（shape `[4]`）。
  `random.Random(seed=20260914)` で初期値・10 step 分の勾配系列を
  固定生成し、`torch.optim.Adagrad` の `step()` を 10 回呼んで各 step
  後のパラメータ値を記録した。
- 5 ケース（`gen_reference.py::CASES`）:
  - `default`: `lr=1e-2, lr_decay=0.0, weight_decay=0.0,
    initial_accumulator_value=0.0, eps=1e-10`（`torch.optim.Adagrad`
    の既定値そのもの）
  - `lr_decay`: `lr_decay=0.1`（他は既定）
  - `weight_decay`: `weight_decay=0.1`（他は既定）
  - `initial_accumulator`: `initial_accumulator_value=0.5`（他は既定）
  - `all`: `lr=0.05, lr_decay=0.05, weight_decay=0.01,
    initial_accumulator_value=0.1, eps=1e-8`
- 初期値・勾配系列はケース間で共通（差はハイパーパラメータのみにし、
  更新式そのものの一致検証に焦点を絞るため）。

## 再生成手順

```bash
python3 -m venv /path/to/venv
/path/to/venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
/path/to/venv/bin/python gen_reference.py   # このディレクトリで実行. adagrad_reference.json を上書きする
sha256sum adagrad_reference.json gen_reference.py   # 本 README の値と照合する
```

## sha256（改竄検知用）

```
589ea0e767fa2bae7cd86216163e09e1191a48353849f83829c31a812e77af7e  adagrad_reference.json
72505dbc434daf074e7eb23c53697327da91040545fe9b59af596c6908b8da7c  gen_reference.py
```
