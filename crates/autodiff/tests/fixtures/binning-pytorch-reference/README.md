# ヒストグラム・二分探索系（histc・bincount・searchsorted・bucketize）PyTorch 参照値フィクスチャの出自

イシュー #2638（親 #2625）の `tests/binning_parity.rs` が参照する固定フィクスチャ。
`stat-reduce-pytorch-reference/README.md` と同じ方針で、生成条件・sha256 を記録した状態で
JSON をコミットする。CI は Python／PyTorch に依存せず、コミット済み JSON のみを読む
（`.claude/rules/ci.md`）。

## 生成条件

- **実 PyTorch 実行値**。`torch.__version__ == "2.14.0+cpu"`・Python 3.14.4・CPU capability `AVX2`
  （JSON の `torch_version`／`python_version`／`cpu_capability` に記録。スクリプトは 2.14.0 系でなければ assert で
  止まる）。numpy や手計算値で代替していない（numpy 未導入の使い捨て venv で実行）。
- **f32 はすべて u32 のビットパターン配列で保存**する（JSON は NaN／inf を運べないため）。索引・カウント
  （torch の int64）は整数配列で保存する。例外になる呼び出しは `torch_raises: true` と例外文面を記録する。
  dtype は float32。乱数は固定シード 2638。
- `differentiability`（5 件）: `histc`・`bincount(weights)`・`searchsorted`（列／値）・`bucketize` の出力の
  `requires_grad`／`grad_fn` と backward の例外。`bincount(weights)` だけが `requires_grad=True`・
  `grad_fn=NotImplemented` で、backward が `derivative for aten::bincount is not implemented` で失敗する。
- `histc_cases`（37 件）: ビン境界ちょうどの値・`torch.linspace` の実エッジとその直上／直下（`nextafter`）・`bins`
  1〜1000・既定範囲（`min == max == 0`）・全要素同値・`min == max != 0`・範囲外・`x == max`・`±0`・空入力・0 次元・
  多次元・数千〜2 万要素の乱数・`max - min` が `f32` で溢れる範囲・NaN／`inf` 入力・0.1 刻み格子。
  `histc_errors`（6 件）: `min > max`・`bins = 0`・データ由来／明示の非有限範囲。
- `bincount_cases`（16 件）: 通常・`minlength` が大／小・空入力・大きな索引・重み付き（正負・相殺・非有限・
  空入力と長さ不一致の重み）・乱数。`bincount_errors`（4 件）: 負値・rank 2／0・重み長不一致。
- `searchsorted_cases`（34 件）: `right` 両方 × 重複列・空列・空 values・0 次元 values・1 次元列 × N 次元 values・
  バッチ（rank 2／3）・NaN・`±inf`・`±0`・未ソート列（記録用）・乱数。`searchsorted_errors`（4 件）: 0 次元列・
  先頭軸不一致・rank 不一致・N 次元列へのスカラー。
- `bucketize_cases`（14 件）・`bucketize_errors`（2 件）: 上記の 1 次元版と非 1 次元 boundaries。
  実測の要点は `docs/autodiff-binning-ops-decision.md` §5。

## 再生成手順

```bash
python3 -m venv /path/to/venv
/path/to/venv/bin/pip install --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple torch==2.14.0
/path/to/venv/bin/python gen_reference.py > binning_reference.json
rm -rf /path/to/venv
```

## sha256

```
2a6b3c81220e6080df3aa964d676413ab6f26b9e80bf3f786f96c95b8b6f5aa7  binning_reference.json
eca09e77f7a6618deded44ecd8e70c5b962aa39baa72505550326ce164f0287b  gen_reference.py
```

`gen_reference.py` を変更した場合は JSON を再生成し、上記を更新する。

出力は `dump_line_per_case` で整形済み（キーごと・16 要素ごとに改行。巨大な単一行 JSON が codex review の diff 読み込みを壊すため。データ内容は不変）。
