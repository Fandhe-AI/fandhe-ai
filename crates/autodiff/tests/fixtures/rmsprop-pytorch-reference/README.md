# RMSprop PyTorch 参照値フィクスチャの出自

イシュー #1743（親 #1610「optimizer（Adam／RMSprop／Adagrad／LAMB）」）の
`tests/nn_optim_rmsprop.rs::rmsprop_matches_pytorch_reference` が参照する
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
  `torch/optim/rmsprop.py::_single_tensor_rmsprop` を実装前に読み、
  `docs/`（本 issue の実装計画 §3.3）で確認した演算順
  （`square_avg.mul_(alpha).addcmul_(grad, grad, value=1-alpha)` →
  `centered` 時は `grad_avg.lerp_(grad, 1-alpha)` →
  `avg = (square_avg - grad_avg^2).sqrt_()`、非 `centered` 時は
  `avg = square_avg.sqrt()` → `avg.add_(eps)`（sqrt の**後**に加算）→
  `momentum > 0` 時は `buf.mul_(momentum).addcdiv_(grad, avg)`；
  `param.add_(buf, alpha=-lr)`、それ以外は
  `param.addcdiv_(grad, avg, value=-lr)`）と完全一致することを確認済み。
- 2 つの独立パラメータ（層としての関係は持たない単純な確認用テンソル）:
  `param_a`（shape `[2, 3]`）・`param_b`（shape `[4]`）。
  `random.Random(seed=20260914)` で初期値・10 step 分の勾配系列を
  固定生成し、`torch.optim.RMSprop` の `step()` を 10 回呼んで各 step
  後のパラメータ値を記録した。
- 5 ケース（`gen_reference.py::CASES`）:
  - `default`: `lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0.0,
    momentum=0.0, centered=False`（`torch.optim.RMSprop` の既定値
    そのもの）
  - `momentum`: `momentum=0.9`（他は既定）
  - `centered`: `centered=True`（他は既定）
  - `weight_decay`: `weight_decay=0.1`（他は既定）
  - `all`: `lr=0.05, alpha=0.9, eps=1e-6, weight_decay=0.01,
    momentum=0.5, centered=True`
- 初期値・勾配系列はケース間で共通（差はハイパーパラメータのみにし、
  更新式そのものの一致検証に焦点を絞るため）。
- 生成された 10 step × 5 ケース × 全要素の中に NaN は含まれない
  （`centered` ケースの `square_avg - grad_avg^2` が丸めで負になり
  `sqrt` が NaN を返す懸念があったが、本フィクスチャの入力系列では
  発生しなかったことを確認済み）。

## 再生成手順

```bash
python3 -m venv /path/to/venv
/path/to/venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
/path/to/venv/bin/python gen_reference.py   # このディレクトリで実行. rmsprop_reference.json を上書きする
sha256sum rmsprop_reference.json gen_reference.py   # 本 README の値と照合する
```

## sha256（改竄検知用）

```
ba955ddacf78415bf0de84a6d638cffa13826ac5cbf66cbb6be0f3a18490dc8e  rmsprop_reference.json
8bbfe9c892a432f12b4607c3511b1b8ebdd0a9d529f2464df539b2b9cae08266  gen_reference.py
```
