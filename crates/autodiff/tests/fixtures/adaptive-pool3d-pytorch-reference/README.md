# 3D 適応プーリング（adaptive_avg_pool3d・adaptive_max_pool3d）PyTorch 参照値フィクスチャの出自

イシュー #3823 の `tests/adaptive_pool3d_parity.rs` が参照する固定フィクスチャ。
`pool3d-pytorch-reference/README.md` と同じ方針で、生成条件・sha256 を記録した状態で JSON を
コミットする。CI は Python／PyTorch に依存せず、コミット済み JSON のみを読む（`.claude/rules/ci.md`）。

## 生成条件

- **実 PyTorch 実行値**。`torch.__version__ == "2.14.0+cpu"`・Python 3.14.4（JSON の
  `torch_version`／`python_version` に記録。スクリプトは 2.14 系でなければ assert で止まる）。
  numpy や手計算値で代替していない（numpy 未導入の使い捨て venv で実行）。
- **f32 はすべて u32 のビットパターン配列で保存**する（JSON は NaN／inf を運べないため）。
  入力 `x_bits`・上流勾配 `g_bits`・出力 `out_bits`・入力勾配 `grad_bits`、Max は索引 `index`
  （torch の int64。`(n, c)` 平面内 flat 添字 `d·H·W + h·W + w`）も保存する。損失は
  `(out * g).sum()`、dtype は float32、乱数は固定シード 3823。
- `finite_cases`（22 件）: 11 構成（割り切れる 4→2、割り切れない 5→3〈窓が重なる〉、非等方、全体縮約、
  恒等、出力が入力より大きい、複数バッチ、0.5 刻みのタイ、全要素同値、負値、空間 1×1×1）× Max／Avg。
- `nonfinite_cases`（12 件）: 6 構成（NaN 単数・窓内複数 NaN、`+inf`、`-inf`、窓全体が `-inf`、
  NaN と inf の混在）× Max／Avg。
- `error_cases`（16 件）: `output_size` の 0・空間軸 0・`N=0`・`C=0`・rank 4・rank 3 について、torch が
  例外を出すか否かを実測して記録（`torch_raises`）。本実装との差分は
  `docs/autodiff-pool3d-ops-decision.md` §21.6 の表を参照。

## 再生成手順

```bash
python3 -m venv /path/to/venv
/path/to/venv/bin/pip install --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cpu torch==2.14.0
/path/to/venv/bin/python gen_reference.py > adaptive_pool3d_reference.json
rm -rf /path/to/venv
```

## sha256

```
d6f5f6c335fce19a4925c443e13bfa46547e71fdf4d3e162d66ad26d2108ace6  adaptive_pool3d_reference.json
63d0a196c8f8092daa97074c7a679cb09995a369016e9bc8e7a800b82fbb1aa2  gen_reference.py
```

`gen_reference.py` を変更した場合は JSON を再生成し、上記を更新する。
