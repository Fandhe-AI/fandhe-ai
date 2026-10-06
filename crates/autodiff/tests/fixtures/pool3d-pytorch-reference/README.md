# 3D プーリング（max_pool3d・avg_pool3d）PyTorch 参照値フィクスチャの出自

イシュー #2643（親 #2625）の `tests/pool3d_parity.rs` が参照する固定フィクスチャ。
`cumulative-pytorch-reference/README.md` と同じ方針で、生成条件・sha256 を記録した状態で
JSON をコミットする。CI は Python／PyTorch に依存せず、コミット済み JSON のみを読む
（`.claude/rules/ci.md`）。

## 生成条件

- **実 PyTorch 実行値**。`torch.__version__ == "2.14.0+cpu"`・Python 3.14.4（JSON の
  `torch_version`／`python_version` に記録。スクリプトは 2.14.0 系でなければ assert で止まる）。
  numpy や手計算値で代替していない（numpy 未導入の使い捨て venv で実行）。
- **f32 はすべて u32 のビットパターン配列で保存**する（JSON は NaN／inf を運べないため）。
  入力 `x_bits`・上流勾配 `g_bits`・出力 `out_bits`・入力勾配 `grad_bits`、Max は索引 `index`
  （torch の int64。`(n, c)` 平面内 flat 添字 `d·H·W + h·W + w`）も保存する。損失は
  `(out * g).sum()`、dtype は float32、乱数は固定シード 2643。
- `finite_cases`（45 件）: 17 構成（等方・非等方の kernel／stride／padding、`stride < kernel` の
  重なり窓、`stride > kernel`、padding が `kernel/2` 境界、全要素同値・`±0` 混在・0.5 刻みの
  タイ・負値、kernel = 入力全体、単位 kernel、複数バッチ、Max の dilation）。Max は全 17 構成、
  Avg は dilation 無しの 14 構成 × `count_include_pad` 2 通り。
- `nonfinite_cases`（18 件）: 6 構成（NaN 単数・複数・窓内複数 NaN、`+inf`、`-inf`、窓全体が
  `-inf`、NaN と inf の混在）× Max／Avg（`count_include_pad` 2 通り）。
- `error_cases`（24 件）: padding 上限超過・kernel／stride／dilation の 0・カーネル超過・空間軸 0・
  `N=0`・`C=0`・rank 4（バッチなし）・rank 3・`ceil_mode=True` について、torch が例外を出すか
  否かを実測して記録（`torch_raises`）。実測で確定した点:
  - torch は `C=0` を拒否し（`Expected input's non-batch dimensions to have positive length`）、
    `N=0` は受理する。本実装は両方を受理する（空出力。差分）。
  - torch は rank 4（バッチなし）を受理して `[C, Dout, Hout, Wout]` を返す。本実装は rank 5 のみ
    （差分）。
  - torch は `ceil_mode=True` を受理する。本実装は v1 で拒否する（差分）。
  - `avg_pool3d` に dilation は無い（`avg_dilation_zero` は dilation を渡しても無関係）。
  - 本実装との突き合わせは `docs/autodiff-pool3d-ops-decision.md` §5。
- 空窓になる構成（`kernel = 2` かつ `dilation > 入力長`）は torch で実行していない
  （`docs/pooling-ops-design.md` §3 が torch 側の縮退動作を記録している）。本実装の拒否は
  Rust 側のテストだけで固定する。

## 再生成手順

```bash
python3 -m venv /path/to/venv
/path/to/venv/bin/pip install --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple torch==2.14.0
/path/to/venv/bin/python gen_reference.py > pool3d_reference.json
rm -rf /path/to/venv
```

## sha256

```
ab2acd7d575073a7355712bb78abc09f5b75f35c54604adb97d0e0bcb5e848f1  pool3d_reference.json
702c54d09ca96c0c9e9f82f0f390fd49d7ab3a611c283d4824077195b9bc970b  gen_reference.py
```

`gen_reference.py` を変更した場合は JSON を再生成し、上記を更新する。
