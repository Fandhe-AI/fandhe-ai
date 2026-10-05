# 逆三角関数・双曲線関数 PyTorch 参照値フィクスチャの出自

イシュー #2634（親 #2625）の `tests/trig_ops_parity.rs` が参照する固定
フィクスチャ。`fft-pytorch-reference/README.md` と同じ方針で、生成条件・
sha256 を記録した状態で JSON をコミットする。CI は Python／PyTorch に
依存せず、コミット済み JSON のみを読む（`.claude/rules/ci.md`）。

## 生成条件

- **実 PyTorch 実行値**。`torch.__version__ == "2.14.0+cpu"`・Python 3.14.4
  （JSON の `torch_version`／`python_version` に記録）。numpy や手計算値で
  代替していない。
- 損失は `(out * g).sum()`（`g` は固定シード `2634` の一様乱数 `[-1, 1)`）。
  `atan2` は `torch.atan2(input, other)` の引数順（Rust 側の `a = y`・`b = x`）。
- 入力・上流勾配・出力・入力勾配・shape をすべて JSON に保存する（Rust 側で
  入力を再生成しない）。dtype は float32。`cases` は NaN／inf を含まない
  ことを生成時に assert 済み。
- `cases`（計 28 件）: 単項 8 演算 × 形状 `[6]`・`[2,3]`・`[2,2,2]`（入力は各演算の
  定義域の内側）＋ `atan2` 4 件（同形状 2・broadcast `[2,3]` 対 `[3]`・`[2,1]` 対
  `[1,3]`。原点から離した 4 象限）。
- `edge_cases`（9 件）: 定義域の境界・外側・巨大値（`3e38`）・符号付きゼロ・
  `atan2` の原点／符号付きゼロ／`1e-30`／`1e20`。NaN／inf は JSON で表せないため
  要素ごとに `{"class": nan|pos_inf|neg_inf|finite, "value"}` で保存する
  （勾配は上流勾配 1 で測定）。主な実測クラス:
  - `asin(±1)`・`acos(±1)` の勾配は ±inf（`asin` は +inf、`acos` は -inf）、`|x| > 1` は NaN
  - `atanh(±1)` の forward は ±inf、勾配は +inf。`atanh(±1.5)` の勾配は有限値 `-0.8`
  - `acosh(1)` の勾配は +inf、`acosh(0.5/0/-0.5)` は NaN、`acosh(-1.5)` の勾配は有限（`0.894…`）
  - `atan2(0, 0)` の勾配は `0`（NaN ではない）。`atan2(1e-30, 1e-30)`・
    `atan2(1e20, 1e20)` の勾配は PyTorch が `f32` の underflow／overflow で `0` を返す
    （本実装は `f64` 昇格で数学的に正しい有限値を返す。差分は
    `docs/autodiff-trig-ops-decision.md` §5）

## 再生成手順

```bash
python3 -m venv /path/to/venv
/path/to/venv/bin/pip install --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple torch==2.14.0
/path/to/venv/bin/python gen_reference.py > trig_ops_reference.json
rm -rf /path/to/venv
```

## sha256

```
d5419ac8c26dfb44ff0ee0c37d83f5795cbcdd16042e1f3675305b6cb93265be  trig_ops_reference.json
755edc3b537436f9757763f78667d626c4cd5958b1cc2c1eebc0e470351eb72c  gen_reference.py
```

`gen_reference.py` を変更した場合は JSON を再生成し、上記を更新する。
