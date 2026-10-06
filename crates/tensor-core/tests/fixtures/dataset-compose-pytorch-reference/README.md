# Dataset 合成（Subset／ConcatDataset／random_split）PyTorch 参照値フィクスチャの出自

イシュー #2661（親 #2660）の `crates/facade/tests/data_dataset_compose.rs` が参照する固定フィクスチャ。
`packed-sequence-pytorch-reference/README.md` と同じ方針で、生成条件・sha256 を記録した状態で JSON を
コミットする。CI は Python／PyTorch に依存せず、コミット済み JSON のみを読む（`.claude/rules/ci.md`）。

## 生成条件

- **実 PyTorch 実行値**。`torch.__version__ == "2.14.0+cpu"`・Python 3.14.4（JSON の `torch_version`／
  `python_version` に記録。スクリプトは 2.14.0 系でなければ assert で止まる）。numpy 未導入の使い捨て venv で実行。
- f32 は u32、割合（f64）は u64 のビットパターンで保存する（serde_json の既定 float パースは 1 ULP ずれうるため）。
- `fraction_cases`（17 件）: `(n, fractions)` ごとの解決後の長さ列、または `torch_raises`。余りの round-robin
  配分・長さ 0 の分割・`[0.1]*10` の丸め境界・合計が 1 を外れる例・範囲外／NaN／空の割合を含む。
- `int_split_cases`（8 件）: 長さ列ごとの成否と長さ。分割が「同シードの `randperm` を累積 offset で切った連続区間」
  であることは生成時に assert 済み（`split_structure_verified`）。`randperm` の数列そのものは本リポの RNG と
  一致しないため保存しない。
- `subset_cases`（5 件）: 基本・重複添字・逆順全域・空・入れ子の取り出し行。
- `concat_cases`（3 件）: 長さ 0 成分を含む連結の `cumulative_sizes` と全域／逆順／交互／境界の添字での取り出し行、
  タプル（f32 特徴量＋整数ラベル）、`Subset` 同士の連結。
- `error_cases`（3 件）: 空リストの連結・`Subset` の範囲外添字（torch は参照時に遅延失敗）・連結の範囲外添字。

## 再生成手順

```bash
python3 -m venv /path/to/venv
/path/to/venv/bin/pip install --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple torch==2.14.0
/path/to/venv/bin/python gen_reference.py > dataset_compose_reference.json
rm -rf /path/to/venv
```

## sha256

```
57e26bdaf412249988e555dc3dcb40e6cb771ff799ca0f50659ee6a7de2cd5e7  dataset_compose_reference.json
39c1a3f4f40198c3b45e87689cadabc6f13db5667def61e9205aef8595ed52fc  gen_reference.py
```

`gen_reference.py` を変更した場合は JSON を再生成し、上記を更新する。
