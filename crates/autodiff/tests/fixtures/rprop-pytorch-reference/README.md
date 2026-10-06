# Rprop PyTorch 参照値フィクスチャの出自

イシュー #2655（親 #2654）の
`tests/nn_optim_rprop.rs::rprop_matches_pytorch_reference`・
`rprop_matches_pytorch_reference_edge_cases` が参照する固定フィクスチャ。
`adamax-pytorch-reference/README.md`（イシュー #2171 先例）と同じ方針で、
生成条件・sha256 を記録した状態で JSON をコミットする。CI は
Python/PyTorch に依存せず、コミット済み JSON のみを読む
（`.claude/rules/ci.md`「グローバル状態を汚す処理を workflow に書かない」）。

## 生成条件

- **実 PyTorch 実行値**。`torch.__version__ == "2.14.0+cpu"`（`pip
  install --index-url https://download.pytorch.org/whl/cpu
  torch==2.14.0` で一時 venv に導入し、`gen_reference.py` を 1 回実行して
  生成した。`foreach=False` を明示して単一テンソル経路を固定）。
- **演算順のドリフト確認**: 導入した torch 2.14.0+cpu の
  `torch/optim/rprop.py::_single_tensor_rprop` を実装前に読み、演算順
  （`sign = (grad*prev).sign()` → `sign>0 → etaplus`・`sign<0 → etaminus`・
  `sign==0 → 1` → `step_size.mul_(sign).clamp_(min, max)` →
  `grad[sign == etaminus] = 0` → `param.addcmul_(grad.sign(), step_size,
  value=-1)` → `prev.copy_(grad)`）と完全一致することを確認済み。
  `step_size` は初回 step で `torch.full_like(grad, lr)` により全要素
  初期化され、以後 `lr` を参照しない。
- 2 つの独立パラメータ: `param_a`（shape `[2, 3]`）・`param_b`（shape
  `[4]`）。`random.Random(seed=20260927)` で初期値・10 step 分の勾配系列を
  固定生成し、一部要素の系列を固定した（`param_a[0]` は常に正＝同符号継続、
  `param_b[0]` は毎 step 符号反転、`param_a[1]`・`param_b[2]` は厳密 0.0 を
  各 2 step）。
- 4 ケース（`gen_reference.py::CASES`）:
  - `default`: `lr=1e-2, etas=(0.5, 1.2), step_sizes=(1e-6, 50.0)`
  - `etas`: `etas=(0.3, 1.5)`（他は既定）
  - `tight_bounds`: `step_sizes=(5e-3, 2e-2)`（下限・上限の両方へ張り付く）
  - `lr_change`: step 5 の前に `param_groups[0]["lr"] = 0.5` へ変更
    （`step_size` は初回で確定済みのため以後の更新値に影響しないことを固定）
- **生成時 assert**（`gen_reference.py::run_case`）: 各ケースで勾配×prev の
  符号が `{-1, 0, +1}` の 3 分岐すべてを通ること、`tight_bounds` で
  `opt.state[p]["step_size"]` が下限・上限の両方に一致する step があること。
- **エッジケース**（JSON の `edge` ブロック）: NaN・`+0.0`／`-0.0`・`±inf`・
  積がアンダーフローする微小値（`1e-30`）を含む 4 step 系列。厳密 JSON は
  NaN／inf を表せないため `f32` のビットパターン（`u32`）で保存した。
  PyTorch 2.14.0 実測: `torch.sign(NaN) == 0`・`torch.sign(±0.0) == 0`、
  NaN 勾配の要素は当 step 更新されず `prev` に NaN が残り、次 step は
  `sign(NaN*g) = 0` → `factor = 1` として再開する。

## 再生成手順

```bash
python3 -m venv /path/to/venv
/path/to/venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.0
/path/to/venv/bin/python gen_reference.py   # このディレクトリで実行. rprop_reference.json を上書きする
sha256sum rprop_reference.json gen_reference.py   # 本 README の値と照合する
```

## sha256（改竄検知用）

```
5271ebcc57be66a654b00cf0adda580c77c40f2ff906ee43aeec2d8442c1930a  rprop_reference.json
28653f758cb8da1cfb2d110d7945a11d3419d6b08284e0c8b72fbb823e864c79  gen_reference.py
```
