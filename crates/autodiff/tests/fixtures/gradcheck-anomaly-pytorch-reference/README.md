# gradcheck PyTorch 参照値フィクスチャの出自

イシュー #2671（親 #2668）の `tests/gradcheck_anomaly_parity.rs` が参照する固定フィクスチャ。
`jacobian-hessian-pytorch-reference/README.md` と同じ方針で、生成条件・sha256 を記録した状態で
JSON をコミットする。CI は Python／PyTorch に依存せず、コミット済み JSON のみを読む
（`.claude/rules/ci.md`）。

## 生成条件

- **実 PyTorch 実行値**。`torch.__version__ == "2.14.0+cpu"`・Python 3.14.4（JSON の
  `torch_version`／`python_version` に記録。スクリプトは 2.14.0 系でなければ assert で止まる）。
  `torch.autograd.functional.jacobian` を `vectorize=False`・`create_graph=False` で、float32（G1 の期待値）と
  float64（G3 の観測値）の両方について実行。numpy や手計算値で代替していない。
- **f32 は u32、f64 は u64 のビットパターン配列で保存**する。入力・定数・出力（`out`／`out_f64`）・期待値
  （`expected`／`expected_f64`）をすべて保存し、Rust 側で入力を再生成しない。乱数は固定シード 2671。
- `cases`（7 件）: 要素ごとの合成（`elementwise`）・matmul＋tanh で rank 2（`matmul_tanh`）・スカラー出力の縮約
  （`sum_reduce`）・小さな MLP（`mlp`）・rank 0 の入力（`scalar_in`）・複数入力（`multi_input`）・キンクを避けた
  relu（`relu_safe`）。relu を含むケースは前段値が 10·eps（eps=1e-3）より 0 から離れることをスクリプトが assert する。
- プログラム名は Rust 側 `build_program` が同名で組む。

## 再生成手順

```bash
python3 -m venv /path/to/venv
/path/to/venv/bin/pip install --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple torch==2.14.0
/path/to/venv/bin/python gen_reference.py gradcheck_anomaly_reference.json
rm -rf /path/to/venv
```

## sha256

```
fc57da0c71c9c7aa20c85cf91418e01ac7feaa27054f6dc069c4f5c787c01935  gradcheck_anomaly_reference.json
3069b0775bfca691b456461a671173889257e96969975c1a4a9a52982eb990a3  gen_reference.py
```

`gen_reference.py` を変更した場合は JSON を再生成し、上記を更新する。
