# 低精度 forward（MatMul・elementwise 5 演算）PyTorch 参照値フィクスチャの出自

イシュー #2628（親 #2626）の `crates/facade/tests/low_precision_ops_pytorch_parity.rs` が参照する固定フィクスチャ。`fft-pytorch-reference/README.md` と同じ方針で、生成条件・sha256 を記録した状態で JSON をコミットする。CI は Python／PyTorch に依存せず、コミット済み JSON のみを読む（`.claude/rules/ci.md`）。

## 生成条件

- **実 PyTorch 実行値**。`torch.__version__ == "2.14.0+cpu"`・Python 3.14.4（JSON の `torch_version`／`python_version` に記録）。numpy や手計算値で代替していない。
- forward 参照: 各ケース × {float16, bfloat16} で `op(x.to(dtype), ..).float()`。
- 勾配参照: float32 autograd で `(op(x, ..) * g).sum().backward()`（`g` は固定シード `20261005` の `torch.randn`）。入力勾配を 6 Op とも記録する（ゲート対象は MatMul／Add／Mul／Relu のみ。Exp／Tanh は Rust 側で非ゲートの観測値）。
- 入力・上流勾配・出力・入力勾配・shape をすべて JSON に保存する（Rust 側で入力を再生成しない）。master dtype は float32。記録する入力・出力が有限であることを生成時に assert 済み。
- ケース（`gen_reference.py::build_cases`、計 19 件）: matmul 4（rank 2 ×2・バッチ付き・厳密表現）／add・mul 各 4（同形・bias パターン・一般 broadcast・厳密表現）／relu 3（ランダム・ゼロ混在・厳密表現）／exp 2（出力が f16 で有限になる入力域に限定）／tanh 2。各 Op に対象 dtype で厳密表現できる入力のケースを 1 つ含む。

## Op × dtype の実行可否（実測）

PyTorch 2.14.0+cpu は **6 Op × {float16, bfloat16} の全組合せを実行できた**（19 ケース × 2 dtype = 38 組合せすべて `supported: true`）。Rust 側は実行された (op, dtype) 集合が全ケース × 2 dtype と完全一致することを assert する。

## 再生成手順

```bash
python3 -m venv /path/to/venv
/path/to/venv/bin/pip install --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple torch==2.14.0
/path/to/venv/bin/python gen_reference.py > low_precision_ops_reference.json
rm -rf /path/to/venv
```

## sha256

```
165e1e40d8e87489c1339f12a5290b4f6ab99712439664cbf8302b0d264d1275  low_precision_ops_reference.json
f3b6576044f065d22f263a2ab9cfc26fa6ac174fb1af66f58d492104ea44efbd  gen_reference.py
```

`gen_reference.py` を変更した場合は JSON を再生成し、上記を更新する。
