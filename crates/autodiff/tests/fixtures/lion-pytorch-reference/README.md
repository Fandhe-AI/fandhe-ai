# Lion PyTorch 参照値フィクスチャの出自

イシュー #2656（親 #2654）の
`tests/nn_optim_lion.rs::lion_matches_pytorch_reference`・
`lion_matches_pytorch_reference_edge_cases` が参照する固定フィクスチャ。
CI は Python/PyTorch に依存せず、コミット済み JSON のみを読む。

## 重要: `torch.optim` に Lion は存在しない

**`torch.optim.Lion` は PyTorch 2.14.0+cpu に存在しない**（`torch/optim/
__init__.py::__all__` に無く、`torch/optim/` に該当ファイルもない）。したがって
本 fixture は `torch.optim` の実装との突合ではなく、**公式参照実装
（google/automl `lion/lion_pytorch.py`。Apache-2.0。
<https://github.com/google/automl/blob/master/lion/lion_pytorch.py>）の更新則を
`gen_reference.py` へ自前で記述し、実 PyTorch 2.14.0+cpu のテンソル演算で実行
した値**である。イシュー #2656 の受け入れ基準の字義（`torch.optim` の実行値）
からの逸脱であり、独立性の補強として `tests/nn_optim_lion.rs` に論文の式どおり
の `f64` 参照実装との突合（`lion_matches_independent_f64_reference`）を置いて
いる。サードパーティ `lion-pytorch`（lucidrains）は導入していない。

## 生成条件

- `torch.__version__ == "2.14.0+cpu"`（`pip install --index-url
  https://download.pytorch.org/whl/cpu torch==2.14.0` で一時 venv に導入し、
  `gen_reference.py` を 1 回実行して生成した）。
- 更新則（公式実装の演算順そのまま）:
  `p.mul_(1 - lr*wd)` → `update = exp_avg.mul(beta1).add(grad, alpha=1-beta1)` →
  `p.add_(update.sign_(), alpha=-lr)` →
  `exp_avg.mul_(beta2).add_(grad, alpha=1-beta2)`。
- 2 つの独立パラメータ（`[2,3]`・`[4]`）。`random.Random(seed=20261007)` で
  初期値・10 step 分の勾配系列を固定し、一部要素の系列を固定した（`[2,3]` の
  要素 0 は常に正・要素 1 は毎 step 符号反転・要素 2 と `[4]` の要素 3 は
  冒頭の厳密 0.0 を含む）。
- 5 ケース: `base`（`lr=1e-2`）・`lr`（`0.1`）・`betas`（`0.6, 0.8`）・
  `weight_decay`（`lr=0.05, wd=0.5`）・`lr_change`（step 5 の前に `lr=0.2`）。
- **生成時 assert**: 全 step・全要素で `update` が「厳密に 0」または
  `|update| >= 1e-3`（`sign` が丸め差で反転しない余裕）、`sign` の 3 値
  （`-1`・`0`・`+1`）がすべて出ること。
- **エッジケース**（JSON の `edge` ブロック。4 step）: NaN・`+0.0`／`-0.0`・
  `±inf`・極小値（`1e-30`）。`f32` のビットパターン（`u32`）で保存。実測:
  `sign(NaN) = sign(±0.0) = 0`。NaN 勾配の要素は当 step 動かず、`exp_avg` に
  NaN が残り以後 weight decay 以外で動かない。

## 再生成手順

```bash
python3 -m venv /path/to/venv
/path/to/venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.0
/path/to/venv/bin/python gen_reference.py   # このディレクトリで実行. lion_reference.json を上書きする
sha256sum lion_reference.json gen_reference.py   # 本 README の値と照合する
```

## sha256（改竄検知用）

```
edcf290972fe11a26ab6a340c2756c4d7caa22f89e69a7a320786a5a033c454d  lion_reference.json
85a33a4386bbeabd703837e0f15cb306fa0f9665cdae76fff2a3b18e6be8a564  gen_reference.py
```
