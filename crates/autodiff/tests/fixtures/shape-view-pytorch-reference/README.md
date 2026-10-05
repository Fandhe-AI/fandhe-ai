# 形状演算（unbind・movedim・swapaxes・tensor_split・meshgrid・rot90）PyTorch 参照値フィクスチャの出自

イシュー #2639（親 #2625）の `tests/shape_view_parity.rs` が参照する固定フィクスチャ。
`stat-reduce-pytorch-reference/README.md` と同じ方針で、生成条件・sha256 を記録した状態で
JSON をコミットする。CI は Python／PyTorch に依存せず、コミット済み JSON のみを読む
（`.claude/rules/ci.md`）。

## 生成条件

- **実 PyTorch 実行値**。`torch.__version__ == "2.14.0+cpu"`・Python 3.14.4（JSON の
  `torch_version`／`python_version` に記録。スクリプトは 2.14.0 系でなければ assert で止まる）。
  numpy や手計算値で代替していない（numpy 未導入の使い捨て venv で実行）。
- **f32 はすべて u32 のビットパターン配列で保存**する（JSON は NaN／inf を運べないため。NaN の payload も保存）。
  入力 `x_bits`・上流勾配 `g_bits`・各出力 `bits`・入力勾配 `grads` を保存し、Rust 側で入力を再生成しない。
  損失は `Σ_k (y_k * g_k).sum()`、dtype は float32、乱数は固定シード 2639。
- `cases`（186 件）: `unbind` 21・`movedim` 12・`swapaxes` 8・`tensor_split` 11・`tensor_split_indices` 17・`meshgrid` 22・`rot90` 95。
  軸長 0・1、`pre_transpose`（先頭入力を transpose した非 contiguous view）、0 次元の `meshgrid` 入力、
  `tensor_split_indices` の空・単調・非単調・範囲外・重複、`rot90` の `k ∈ {-5, -2, -1, 0, 1, 2, 3, 4, 5}` × 複数の `dims`、
  および非有限値（NaN payload `0x7FC00001`／`0xFFC00002`・`±inf`・`-0.0`）を差し込んだ入力（名前に `nonfinite`）を含む。
- `error_cases`（19 件）: rank 0・軸範囲外・`sections = 0`・重複軸・長さ不一致・空リスト・rank 2 の `meshgrid` 入力・`dims` 同値などで
  torch が例外を出すか否か（`torch_raises`）。実測の要点は `docs/autodiff-shape-view-ops-decision.md` §5。

## 再生成手順

```bash
python3 -m venv /path/to/venv
/path/to/venv/bin/pip install --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple torch==2.14.0
/path/to/venv/bin/python gen_reference.py > shape_view_reference.json
rm -rf /path/to/venv
```

## sha256

```
81368b514f0f3e4a9521af0c8bd16b6e343e043ddc872ee11c7051c03fee3a2f  shape_view_reference.json
2eb0409e35610bbfc801831ba4ddc6408fda2458d36fe6c2ff6ec15de774c119  gen_reference.py
```

`gen_reference.py` を変更した場合は JSON を再生成し、上記を更新する。
