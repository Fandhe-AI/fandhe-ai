# AdamW PyTorch 参照値フィクスチャの出自

イシュー #194（親 #192「optimizer（SGD・AdamW）・gradient clipping の
実装」）の `tests/nn_optim_adamw.rs::adamw_matches_pytorch_reference` が
参照する固定フィクスチャ。`onnx-interop` の
`tests/fixtures/pytorch-reference/README.md`（#73 先例）と同じ方針で、
生成条件・sha256 を記録した状態で JSON をコミットする。CI は
Python/PyTorch に依存せず、コミット済み JSON のみを
読む（`.claude/rules/ci.md`「グローバル状態を汚す処理を workflow に
書かない」）。

## 生成条件

- **実 PyTorch 実行値**（アルゴリズム擬似コードからの再実装ではない）。
  `torch.__version__ == "2.13.0+cpu"`（`pip install --index-url
  https://download.pytorch.org/whl/cpu torch` で venv に導入し、
  `gen_reference.py` を 1 回実行して生成した）。
- 2 つの独立パラメータ（層としての関係は持たない単純な確認用テンソル）:
  `param_a`（shape `[2, 3]`）・`param_b`（shape `[4]`）。
  `random.Random(seed=20260807)` で初期値・10 step 分の勾配系列を
  固定生成し、`torch.optim.AdamW` の `step()` を 10 回呼んで各 step 後の
  パラメータ値を記録した。
- 3 ケース（`gen_reference.py::CASES`）:
  - `default`: `lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8,
    weight_decay=0.01`（`torch.optim.AdamW` の既定値そのもの）
  - `custom`: `lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8,
    weight_decay=0.1`
  - `weight_decay_zero`: `lr=0.01, beta1=0.8, beta2=0.9, eps=1e-6,
    weight_decay=0.0`
- 初期値・勾配系列はケース間で共通（差はハイパーパラメータのみにし、
  更新式そのものの一致検証に焦点を絞るため）。

## 再生成手順

```bash
python3 -m venv /path/to/venv
/path/to/venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
/path/to/venv/bin/python gen_reference.py   # このディレクトリで実行. adamw_reference.json を上書きする
sha256sum adamw_reference.json gen_reference.py   # 本 README の値と照合する
```

## sha256（改竄検知用）

```
b18a9c758321bac4636f3709b65e2614ad884dd10dfe8b15b4cc4b1b4cb0ec83  adamw_reference.json
1807e319df4580b5e7694c3da06570d8b5bbc94fa26203665a5ee218d7f82c13  gen_reference.py
```
