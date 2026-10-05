# 累積演算（cummax・cummin・logcumsumexp）PyTorch 参照値フィクスチャの出自

イシュー #2636（親 #2625）の `tests/cumulative_parity.rs` が参照する固定フィクスチャ。
`nonfinite-pytorch-reference/README.md` と同じ方針で、生成条件・sha256 を記録した状態で
JSON をコミットする。CI は Python／PyTorch に依存せず、コミット済み JSON のみを読む
（`.claude/rules/ci.md`）。

## 生成条件

- **実 PyTorch 実行値**。`torch.__version__ == "2.14.0+cpu"`・Python 3.14.4（JSON の
  `torch_version`／`python_version` に記録。スクリプトは 2.14.0 系でなければ assert で止まる）。
  numpy や手計算値で代替していない（numpy 未導入の使い捨て venv で実行）。
- **f32 はすべて u32 のビットパターン配列で保存**する（JSON は NaN／inf を運べないため）。
  入力 `x_bits`・上流勾配 `g_bits`・出力 `out_bits`・入力勾配 `grad_bits`、`cummax`／`cummin` は
  索引 `index`（torch の int64）も保存する。損失は `(values * g).sum()`、dtype は float32、
  乱数は固定シード 2636。
- `finite_cases`（49 件）: 3 演算 × 15 形状（軸長 1・通常長・連続／非連続タイ・単調増加／減少・
  全要素同値・`±0` 混在・負値・バッチ・rank 3 で `dim=0/1/2`）＋ `logcumsumexp` 専用 4 件
  （大振幅 ±50・±500・広いダイナミックレンジ）。
- `nonfinite_cases`（39 件）: 3 演算 × 13 ケース（NaN の先頭・中間・複数・末尾、`-inf` 全体・先頭・
  中間、`+inf`、`+inf`／`-inf` 混在、NaN と inf の混在、バッチ付き）。forward（値・索引）と勾配を保存。
- `error_cases`（15 件）: 軸長 0・`dim` 範囲外・0 次元入力・負の `dim` で torch が例外を出すか否かを
  実測して記録（`torch_raises`）。実測: **範囲外 `dim` のみ `IndexError`**。軸長 0・0 次元（`dim=0`／`-1`）・
  負の `dim` はいずれも例外にならない。本実装との突き合わせは `docs/autodiff-cumulative-ops-decision.md` §5。

## 再生成手順

```bash
python3 -m venv /path/to/venv
/path/to/venv/bin/pip install --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple torch==2.14.0
/path/to/venv/bin/python gen_reference.py > cumulative_reference.json
rm -rf /path/to/venv
```

## sha256

```
e718a701468984eaefe766d1e4e33a3bb13ab043aa5909572f2c95ed3bfed95b  cumulative_reference.json
d8c7fd5d61421bc5d0d5eb892ce202f7aa3699c89558d3957f42886e1e7d4027  gen_reference.py
```

`gen_reference.py` を変更した場合は JSON を再生成し、上記を更新する。
