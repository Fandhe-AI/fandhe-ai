# ASGD PyTorch 参照値フィクスチャの出自

イシュー #2655（親 #2654）の
`tests/nn_optim_asgd.rs::asgd_matches_pytorch_reference` が参照する固定
フィクスチャ。`adamax-pytorch-reference/README.md`（イシュー #2171 先例）と
同じ方針で、生成条件・sha256 を記録した状態で JSON をコミットする。CI は
Python/PyTorch に依存せず、コミット済み JSON のみを読む
（`.claude/rules/ci.md`「グローバル状態を汚す処理を workflow に書かない」）。

## 生成条件

- **実 PyTorch 実行値**。`torch.__version__ == "2.14.0+cpu"`（`pip
  install --index-url https://download.pytorch.org/whl/cpu
  torch==2.14.0` で一時 venv に導入し、`gen_reference.py` を 1 回実行して
  生成した。`foreach=False` を明示して単一テンソル経路を固定）。
- **演算順のドリフト確認**: 導入した torch 2.14.0+cpu の
  `torch/optim/asgd.py::_single_tensor_asgd` を実装前に読み、演算順
  （`step += 1` → `weight_decay != 0` なら `grad = grad.add(param,
  alpha=weight_decay)` → `param.mul_(1 - lambd*eta)` →
  `param.add_(grad, alpha=-eta)` → `mu != 1` なら
  `ax.add_(param.sub(ax).mul_(mu))`、`mu == 1` なら `ax.copy_(param)` →
  `eta = lr / (1 + lambd*lr*step)**alpha`・`mu = 1 / max(1, step - t0)`）と
  完全一致すること、`eta`／`mu` が `float32` スカラーテンソル
  （`optimizer.py::_get_scalar_dtype`）で保持されること（初期値 `eta=lr`・
  `mu=1`）を確認済み。
- 2 つの独立パラメータ: `param_a`（shape `[2, 3]`）・`param_b`（shape
  `[4]`）。`random.Random(seed=20260928)` で初期値・10 step 分の勾配系列を
  固定生成した。各 step の `param_*` と平均化パラメータ `ax_*`
  （`opt.state[p]["ax"]`）の両方を記録している。
- 5 ケース（`gen_reference.py::CASES`）:
  - `default`: `lr=1e-2, lambd=1e-4, alpha=0.75, t0=1e6, weight_decay=0`
    （全 step で `mu == 1`）
  - `small_t0`: `t0=2.0`（`mu == 1` と `mu != 1` の両分岐を通る）
  - `weight_decay`: `weight_decay=0.1`
  - `all`: `lr=0.05, lambd=0.02, alpha=0.6, t0=3.0, weight_decay=0.05`
  - `lr_change`: `lambd=1e-2`。step 5 の前に `param_groups[0]["lr"] =
    0.05` へ変更（保持済み `eta` は次の step でそのまま使われ、新 `lr` は
    その step の終わりに計算する `eta` から効く 1 step 遅れを固定）
- **生成時 assert**（`gen_reference.py::run_case`）: `t0 < 10` のケースで
  10 step 内に `mu == 1` と `mu != 1` の両分岐を通ること。

## 再生成手順

```bash
python3 -m venv /path/to/venv
/path/to/venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.0
/path/to/venv/bin/python gen_reference.py   # このディレクトリで実行. asgd_reference.json を上書きする
sha256sum asgd_reference.json gen_reference.py   # 本 README の値と照合する
```

## sha256（改竄検知用）

```
bdcd0b8a493b368e73c49702302002e216fcbd852aa4f650297063d6d4a503b5  asgd_reference.json
46f1cdb8b0ddaf9d0844b65a2e360ed8e7dc19167ff74e22f5621ba0e8a71d98  gen_reference.py
```
