# 結合 4 演算（Concatenate・Add・Multiply・Average）PyTorch 参照値フィクスチャの出自

イシュー #2666（親 #2663）の `tests/merge_ops_parity.rs` が参照する固定フィクスチャ。
`tensor-product-pytorch-reference/README.md` と同じ方針で、生成条件・sha256 を記録した状態で
JSON をコミットする。CI は Python／PyTorch に依存せず、コミット済み JSON のみを読む
（`.claude/rules/ci.md`）。

## 生成条件

- **実 PyTorch 実行値**。`torch.__version__ == "2.14.0+cpu"`・Python 3.14.4（JSON の
  `torch_version`／`python_version` に記録。スクリプトは 2.14.0 系でなければ assert で止まる）。
  numpy や手計算値で代替していない。
- **f32 はすべて u32 のビットパターン配列で保存**する。入力 `x_bits`・上流勾配 `g_bits`・出力 `bits`・
  入力勾配 `grads` を保存し、Rust 側で入力を再生成しない。損失は `(y * g).sum()`、dtype は float32、
  乱数は固定シード 2666。
- 参照式: Concatenate は `torch.cat`、Add は `x1 + x2 + …`、Multiply は `x1 * x2 * …`、
  Average は `(x1 + … + xn) / n` の左畳み込み（`torch.stack(..).mean(0)` は使わない）。
- `cases`（82 件）: Add／Multiply／Average は rank 0・1・2・4・軸長 1・入力 2／3／7／12 件・先頭入力を
  transpose した非 contiguous view・同一テンソルの重複指定（`uses`）。Concatenate は各 `dim`・不揃いな軸長・
  軸長 1・軸長 0・rank 1〜4・入力 2〜9 件・非 contiguous 入力・重複指定。
- `error_cases`（13 件）: `torch_raises` に torch が例外を出すか否かを記録する。torch が受理するが本実装が
  拒否するもの（broadcast 可能な shape・入力 1 件）は意図した差分として決定記録
  `docs/facade-functional-api-decision.md` §17 に列挙する。

## 再生成手順

```bash
python3 -m venv /path/to/venv
/path/to/venv/bin/pip install --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple torch==2.14.0
/path/to/venv/bin/python gen_reference.py > merge_ops_reference.json
rm -rf /path/to/venv
```

## sha256

```
70228fbfc893651ffa854dab81c02b89e252bf4f9570c92ee54cc3250344633d  merge_ops_reference.json
84bf92c9fedfd27d45eed483b77cfa9fd83981cb891c63d3860b863e1090ccff  gen_reference.py
```

`gen_reference.py` を変更した場合は JSON を再生成し、上記を更新する。
