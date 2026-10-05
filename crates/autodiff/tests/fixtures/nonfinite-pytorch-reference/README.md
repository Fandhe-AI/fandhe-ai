# 非有限値の判定・置換 PyTorch 参照値フィクスチャの出自

イシュー #2635（親 #2625）の `tests/nonfinite_parity.rs` が参照する固定
フィクスチャ。`trig-ops-pytorch-reference/README.md` と同じ方針で、生成条件・
sha256 を記録した状態で JSON をコミットする。CI は Python／PyTorch に
依存せず、コミット済み JSON のみを読む（`.claude/rules/ci.md`）。

## 生成条件

- **実 PyTorch 実行値**。`torch.__version__ == "2.14.0+cpu"`・Python 3.14.4
  （JSON の `torch_version`／`python_version` に記録）。numpy や手計算値で
  代替していない（numpy 未導入の使い捨て venv で実行）。
- **f32 はすべて u32 のビットパターン配列で保存**する。JSON（`serde_json`）は
  NaN／inf を運べないため。符号付き NaN（`0x7fc00000`・`0xffc00001`）・`±inf`・
  `±0.0`・非正規化数・`f32::MAX`／`MIN` も欠落なく保存できる。torch 側もビット
  パターンから float32 テンソルを組み立てる。
- `predicate_cases`（6 件）: 形状 `[0]`・`[1]`・`[10]`・`[2,5]`・`[2,3,4]` と、
  全特殊値を並べた `pred_specials`。`torch.isnan`／`isinf`／`isfinite` の bool 列。
- `nan_to_num_cases`（30 件）: 上記 5 形状 × 引数 6 通り（全省略・`nan` のみ・
  `posinf` のみ・`neginf` のみ・3 つ指定・置換値自体が非有限）。損失は
  `(out * g).sum()`（`g` は固定シード `2635` の有限一様乱数 `[-1, 1)`）。入力・
  `g`・出力・入力勾配をすべて保存し、入力勾配に NaN が無いことを生成時に
  assert 済み（Rust 側が無条件に統一複合判定を使えるようにするため）。
- `upstream_nonfinite_observations`（1 件）: **上流勾配が非有限**のときの PyTorch
  実測の観測記録。突合の合否には使わず、`docs/autodiff-nonfinite-ops-decision.md`
  §5 の事実記載の根拠とする。実測: PyTorch の `nan_to_num` の勾配は
  `grad * isfinite(x)` 相当のため、入力が非有限かつ上流が `±inf`／NaN の位置は
  `0 * inf = NaN` になる。本実装は要素選択のため同位置は 0。入力が有限の位置は
  上流の非有限値をそのまま通し PyTorch と一致する。

## 再生成手順

```bash
python3 -m venv /path/to/venv
/path/to/venv/bin/pip install --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple torch==2.14.0
/path/to/venv/bin/python gen_reference.py > nonfinite_reference.json
rm -rf /path/to/venv
```

## sha256

```
e5f09f2da1fae4e5829beaa3c94ab78325d663314f495ec41f079e209e83b8b2  nonfinite_reference.json
392026fda3c319101f96e13c2c554a5d1832bf0dd84e07369102b2a8836d5afe  gen_reference.py
```

`gen_reference.py` を変更した場合は JSON を再生成し、上記を更新する。
