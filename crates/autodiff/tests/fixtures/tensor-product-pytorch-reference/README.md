# テンソル積・距離・外積（kron・tensordot・cdist・cross）PyTorch 参照値フィクスチャの出自

イシュー #2640（親 #2625）の `tests/tensor_product_parity.rs` が参照する固定フィクスチャ。
`shape-view-pytorch-reference/README.md` と同じ方針で、生成条件・sha256 を記録した状態で
JSON をコミットする。CI は Python／PyTorch に依存せず、コミット済み JSON のみを読む
（`.claude/rules/ci.md`）。

## 生成条件

- **実 PyTorch 実行値**。`torch.__version__ == "2.14.0+cpu"`・Python 3.14.4（JSON の
  `torch_version`／`python_version` に記録。スクリプトは 2.14.0 系でなければ assert で止まる）。
  numpy や手計算値で代替していない。
- **f32 はすべて u32 のビットパターン配列で保存**する（JSON は NaN／inf を運べないため。NaN の payload も保存）。
  入力 `x_bits`・上流勾配 `g_bits`・出力 `bits`・入力勾配 `grads` を保存し、Rust 側で入力を再生成しない。
  損失は `(y * g).sum()`、dtype は float32、乱数は固定シード 2640。`cdist` の `p = inf` は JSON で数値にできないため文字列 `"inf"` で保存する。
- `cases`（93 件）: `kron` 14・`tensordot` 16・`tensordot_axes` 9・`cdist` 41・`cross` 13。
  軸長 0・1、rank 0、`pre_transpose`（先頭入力を transpose した非 contiguous view。`kron` は torch 2.14.0 が非 contiguous 入力で
  例外を出すため含めない）、同一点を含む `cdist`（名前に `identical`）、平行ベクトルの `cross`、
  非有限値（NaN payload `0x7FC00001`／`0xFFC00002`・`±inf`・`-0.0`）を差し込んだ入力（名前に `nonfinite`）を含む。
  `cdist` は常に `compute_mode="donot_use_mm_for_euclid_dist"`、`cross` は `torch.linalg.cross`。
- `error_cases`（18 件）: torch が例外を出すか否か（`torch_raises`）。実測の要点は
  `docs/autodiff-tensor-product-ops-decision.md` §5。

## 再生成手順

```bash
python3 -m venv /path/to/venv
/path/to/venv/bin/pip install --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple torch==2.14.0
/path/to/venv/bin/python gen_reference.py > tensor_product_reference.json
rm -rf /path/to/venv
```

## sha256

```
4332d853893d28074e96a278f40d3451f479defb83bb63e8ba4ff49ede36efe5  tensor_product_reference.json
ae8e9c21bb98f871c54f72f1e852163e50c8fbaaf3007870db4bd0487b77d7af  gen_reference.py
```

`gen_reference.py` を変更した場合は JSON を再生成し、上記を更新する。
