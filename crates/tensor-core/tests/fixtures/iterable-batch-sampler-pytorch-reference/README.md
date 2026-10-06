# IterableDataset／BatchSampler PyTorch 参照値フィクスチャの出自

イシュー #2662（親 #2660）の `crates/facade/tests/data_iterable_batch_sampler.rs` が参照する固定フィクスチャ。
`dataset-compose-pytorch-reference/README.md` と同じ方針で、生成条件・sha256 を記録した状態で JSON を
コミットする。CI は Python／PyTorch に依存せず、コミット済み JSON のみを読む（`.claude/rules/ci.md`）。

## 生成条件

- **実 PyTorch 実行値**。`torch.__version__ == "2.14.0+cpu"`・Python 3.14.4（JSON の `torch_version`／
  `python_version` に記録。スクリプトは 2.14.0 系でなければ assert で止まる）。numpy 未導入の使い捨て venv で実行。
  生成時に torch の `BatchSampler.__init__`／`__len__` の実ソースを確認済み（`len` は `drop_last` なら
  `len // k`、それ以外は `(len + k - 1) // k`）。
- f32 は u32 のビットパターンで保存する（serde_json の既定 float パースは 1 ULP ずれうるため）。
- `batch_sampler_cases`（82 件）: 添字源（`range(n)` の n = 0, 1, 5, 6, 7, 10、逆順、重複あり、飛び飛び）×
  `batch_size`（1, 2, 3, 長さと等しい, 長さ超）× `drop_last` 両方の `list(BatchSampler(...))` と
  `len(BatchSampler(...))`。乱数 sampler（`RandomSampler`）経由は `randperm` の数列が本リポの RNG と一致しないため含めない。
- `iterable_cases`（35 件）: 行を順に `yield` する `IterableDataset` を `DataLoader(ds, batch_size=k, drop_last=d)`
  で回したバッチ列（shape とビット列）。n = 0, 5, 6, 7 × k = 1, 2, 3, 10 × `drop_last` 両方に加え、
  フィルタ付きジェネレータ（長さが事前に分からない例）3 件。
- `tuple_cases`（3 件）: f32 行と整数ラベルのタプルを供給する iterable のバッチ列。
- `error_cases`（7 件）: `BatchSampler` の `batch_size == 0`・iterable への `shuffle=True`／`sampler=`／
  `batch_sampler=` 指定・`len(DataLoader(iterable))`・サンプル shape 不揃い・ストリーム途中の例外
  （失敗前に完成したバッチ数 `batches_before_error` を併記）。各ケースは `{name, torch_raises, error}`。

## 再生成手順

```bash
python3 -m venv /path/to/venv
/path/to/venv/bin/pip install --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple torch==2.14.0
/path/to/venv/bin/python gen_reference.py > iterable_batch_sampler_reference.json
rm -rf /path/to/venv
```

## sha256

```
28c6571f3a007eb39959b9773efb00b502f190059da42d7c6380ee8bdfd7d3b6  iterable_batch_sampler_reference.json
854d8ac7caefe89f98a7367b356bf43723d6993b1b1eb8ebc70192749a30ced2  gen_reference.py
```

`gen_reference.py` を変更した場合は JSON を再生成し、上記を更新する。
