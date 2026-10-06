# Adafactor PyTorch 参照値フィクスチャの出自

イシュー #2656（親 #2654）の
`tests/nn_optim_adafactor.rs::adafactor_matches_pytorch_reference`・
`adafactor_matches_pytorch_reference_edge_cases` が参照する固定フィクスチャ。
`rprop-pytorch-reference/README.md`（イシュー #2655 先例）と同じ方針で、
生成条件・sha256 を記録した状態で JSON をコミットする。CI は
Python/PyTorch に依存せず、コミット済み JSON のみを読む
（`.claude/rules/ci.md`「グローバル状態を汚す処理を workflow に書かない」）。

## 生成条件

- **実 `torch.optim.Adafactor` の実行値**。`torch.__version__ == "2.14.0+cpu"`
  （`pip install --index-url https://download.pytorch.org/whl/cpu
  torch==2.14.0` で一時 venv に導入し、`gen_reference.py` を 1 回実行して
  生成した。`foreach=False` を明示して単一テンソル経路を固定）。
- **演算順のドリフト確認**: 導入した torch 2.14.0+cpu の
  `torch/optim/_adafactor.py::_single_tensor_adafactor` を実装前に読み、
  `adafactor.rs` モジュール doc の演算順と一致することを確認済み
  （`rho = min(lr, 1/sqrt(t))`・`alpha = max(eps2, RMS(param)) * rho` は
  weight decay より前の param・decoupled weight decay・rank >= 2 は末尾 2
  次元で `vector_norm` → `square_` → `div_` → `lerp_`、`row_var @ col_var`
  を `row_var.mean(dim=-2).clamp_(min=eps1)` で除算、rank <= 1 は
  `variance.lerp_(grad*grad)`、`clamp_(min=eps1*eps1).rsqrt_()`、
  `update.norm / (sqrt(numel) * d)` で clip）。
- 4 つのパラメータ（rank 2 `[2,3]`・rank 1 `[4]`・rank 3 `[2,3,4]`・rank 0
  `[]`）。`random.Random(seed=20261006)` で初期値・10 step 分の勾配系列を固定。
- 7 ケース（`gen_reference.py::CASES`）: `default`・`relative_step`
  （`lr=1.0`。`t>=2` で `1/sqrt(t)` が勝つ）・`weight_decay`・`beta2_decay`
  （`-0.5`）・`eps`（`eps1=0.5`・`eps2=2.0`。両 clamp と `eps2` 側を効かせる）・
  `clip_d`（`d=2.5`）・`lr_change`（step 5 の前に
  `param_groups[0]["lr"] = 0.3`。即時反映を固定）。
- **生成時 assert**（`gen_reference.py::main`）: lerp 重みが `< 0.5` と
  `>= 0.5` の両方、`rho` の両分岐、`alpha` の両分岐（`eps2` 勝ち／RMS 勝ち）、
  `row_var.mean` の `eps1` clamp 発生、`var_estimate < eps1^2` の clamp 発生、
  `denom == 1` と `denom > 1` の両方が通ること。
- **エッジケース**（JSON の `edge` ブロック。rank 1 と rank 2 の 2 パラメータ・
  4 step）: NaN・`+0.0`／`-0.0`・`±inf`・極小値（`1e-30`）を含む系列。厳密 JSON
  は NaN／inf を表せないため `f32` のビットパターン（`u32`）で保存した。
  PyTorch 2.14.0 実測: NaN／inf 勾配は `variance`（rank 1 は要素単位）／
  `row_var`・`col_var`（rank 2 は行・列を共有する要素）を汚染し、以後回復しない。
  Python の `max(eps2, NaN)` は `eps2` を返し、`clamp_min` は NaN を伝播する。
- 状態（`row_var`／`col_var`／`variance`）そのものの突合は行わない（突合の
  ためだけに公開メソッドを増やさない。出力 param の突合で間接的に固定される）。

## 再生成手順

```bash
python3 -m venv /path/to/venv
/path/to/venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.0
/path/to/venv/bin/python gen_reference.py   # このディレクトリで実行. adafactor_reference.json を上書きする
sha256sum adafactor_reference.json gen_reference.py   # 本 README の値と照合する
```

## sha256（改竄検知用）

```
1906116aa1bad685dda3b4aad7e459964890ffa4980a71481ac28dbd0fe046ae  adafactor_reference.json
882e2d5df7993dfc4bd005c98dc4613d5ea5801658f8ea2c7cf945120dd8bd02  gen_reference.py
```
