# pad 非定数モード（reflect・replicate・circular）PyTorch 参照値フィクスチャの出自

イシュー #2642（親 #2625）の `tests/pad_modes_parity.rs` が参照する固定フィクスチャ。
`cumulative-pytorch-reference/README.md` と同じ方針で、生成条件・sha256 を記録した状態で
JSON をコミットする。CI は Python／PyTorch に依存せず、コミット済み JSON のみを読む
（`.claude/rules/ci.md`）。

## 生成条件

- **実 PyTorch 実行値**。`torch.__version__ == "2.14.0+cpu"`・Python 3.14.4（JSON の
  `torch_version`／`python_version` に記録。スクリプトは 2.14.0 系でなければ assert で止まる）。
  numpy や手計算値で代替していない（numpy 未導入の使い捨て venv で実行）。
- **f32 はすべて u32 のビットパターン配列で保存**する（JSON は NaN／inf を運べないため）。
  入力 `x_bits`・上流勾配 `g_bits`・出力 `out_bits`・入力勾配 `grad_bits` を保存する。損失は
  `(out * g).sum()`、dtype は float32、乱数は固定シード 2642。
- `torch_pad` は PyTorch 形式（末尾軸からの平坦リスト）、`pads` は Rust 形式（先頭軸から順の
  `[before, after]` を rank 個。パディングしない先頭軸は `[0, 0]`）。
- `finite_cases`（107 件）: 3 モード（reflect 30・replicate 42・circular 35）×
  形状 7 種（rank 2／3 の 1 軸、rank 3／4 の 2 軸、rank 4／5 の 3 軸、軸長 1）×
  パターン（対称・非対称・片側 0・境界ちょうど〈reflect は軸長 - 1、circular／replicate は軸長〉・
  replicate の軸長超え）。forward（出力 shape・bit）と勾配を保存。
- `nonfinite_cases`（6 件）: 3 モード × 2 ケース（NaN・`±inf`・`-0.0` を含む入力）。forward のみ。
- `error_cases`（14 件）: PyTorch が例外を出すか否かを実測して記録（`torch_raises`・例外型・文面）。
  実測: reflect の `pad >= 軸長`、circular の `pad > 軸長`、パディング軸の長さ 0、rank 1 入力、
  先頭軸へのパディング、PyTorch が受けない rank／pad 個数の組は `RuntimeError`／
  `NotImplementedError`。circular の `pad == 軸長` とバッチ軸長 0 は受理。本実装との突き合わせ
  （意図的な差分の分類表）は `docs/autodiff-pad-modes-decision.md` §5。

## 再生成手順

```bash
python3 -m venv /path/to/venv
/path/to/venv/bin/pip install --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple torch==2.14.0
/path/to/venv/bin/python gen_reference.py > pad_modes_reference.json
rm -rf /path/to/venv
```

## sha256

```
8d42c50210e1969b532f3f242021e1fe2445f67de1bde3d297b3e6f9ebd70596  pad_modes_reference.json
8321a0cf1db7dbc45f2b204283769cea1ce8e9c1972c9f3e8ea3faa4ea8a284e  gen_reference.py
```

`gen_reference.py` を変更した場合は JSON を再生成し、上記を更新する。
