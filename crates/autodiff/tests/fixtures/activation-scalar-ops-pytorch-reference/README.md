# SELU・CELU・Softsign・Hardsigmoid・LogSigmoid PyTorch 参照値フィクスチャの出自

イシュー #2649（親 #2648）の `tests/activation_scalar_ops_parity.rs` が参照する固定フィクスチャ。
`trig-ops-pytorch-reference/README.md` と同じ方針で、生成条件・sha256 を記録した状態で JSON をコミットする。
CI は Python／PyTorch に依存せず、コミット済み JSON のみを読む（`.claude/rules/ci.md`）。

## 生成条件

- **実 PyTorch 実行値**。`torch.__version__ == "2.14.0+cpu"`・Python 3.14.4（JSON の `torch_version`／`python_version` に記録）。
  numpy や手計算値で代替していない。固定シード `2649`・`torch.set_num_threads(1)`・dtype float32。
- 損失は `(out * g).sum()`（`g` は一様乱数 `[-1, 1)`）。入力・上流勾配・出力・入力勾配・shape をすべて JSON に保存する
  （Rust 側で入力を再生成しない）。`cases` は NaN／inf を含まないことを生成時に assert 済み。
  `torch.nn.SELU`／`CELU`／`Softsign`／`Hardsigmoid`／`LogSigmoid` の出力が関数形（`F.*`）と一致することも生成時に assert 済み。
- `cases`（計 24 件）: `selu`・`softsign`・`hardsigmoid`・`log_sigmoid` × 形状 `[6]`・`[2,3]`・`[2,2,2]`（12 件）＋
  `celu`（`alpha ∈ {1.0, 0.5, 2.5, -1.5}`）× 同 3 形状（12 件）。入力は `[-5, 5)` の一様乱数。
  Hardsigmoid は `[-5,-3)`・`(-2.5,2.5)`・`[3,5)` の 3 領域から層化サンプリングして連結・shuffle する。
- `edge_cases`（6 件: `selu`・`celu`〈`alpha` 1.0 と 2.5〉・`softsign`・`hardsigmoid`・`log_sigmoid`。上流勾配 1）: 入力は u32 ビットパターンで保存
  （`±0`・`±3` とその 1 ulp 内側／外側〈float32 tensor の `torch.nextafter`〉・`±100`・`±3e38`・`±inf`・NaN）。出力・勾配は要素ごとに
  `{"class": nan|pos_inf|neg_inf|finite, "value"}`。主な実測クラス:
  - SELU／CELU の NaN 入力の勾配は正側の係数（SELU `1.0507`・CELU `1`）で、本実装（NaN 伝播）と異なる
  - Softsign の `±inf` 入力は forward・勾配とも NaN（本実装の勾配は `0`）
  - Hardsigmoid の勾配は `±3` ちょうどで `0`、`±3` の 1 ulp 内側で `1/6`（開区間判定）
  - LogSigmoid は `+inf` で `0`、`-inf` で `-inf`、`±3e38` で NaN にならない
- `error_cases`（4 件）: `F.celu` の `alpha = 0` は `RuntimeError`、`alpha = NaN／±inf` は例外なし（`x <= 0` の要素が NaN）。
  本実装は 4 件とも `InvalidArgument` で拒否する（差分は `docs/autodiff-activation-scalar-ops-decision.md` §5）。

## 再生成手順

```bash
python3 -m venv /path/to/venv
/path/to/venv/bin/pip install --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple torch==2.14.0
/path/to/venv/bin/python gen_reference.py > activation_scalar_ops_reference.json
rm -rf /path/to/venv
```

2 回連続で生成し JSON が一致する（決定的である）ことを確認済み。

## sha256

```
17be63718f77d1ef050789e38815a8e603400741ac9db7a66864fa47c365fcb8  activation_scalar_ops_reference.json
a32cf3134d764f5b65f208c89579da9fced4d1d71b1d5b02de0671b9f9ed01ba  gen_reference.py
```

`gen_reference.py` を変更した場合は JSON を再生成し、上記を更新する。
