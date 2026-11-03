# FractionalMaxPool2d／3d の PyTorch 参照値 fixture（イシュー #3826）

`F.fractional_max_pool{2,3}d(x, kernel, output_size=..., return_indices=True, _random_samples=s)` の実行値。
xorshift64* と PyTorch の生成器は一致しないため、乱数ではなく「同じ標本に対する選択結果」を突合する
（設計 `docs/autodiff-pool3d-ops-decision.md` §20.4・実装記録 §24）。

## 生成条件

| 項目 | 値 |
|---|---|
| PyTorch | `torch==2.14.0+cpu`（使い捨て venv。リポジトリの依存には加えない） |
| 生成器 | `gen_reference.py`（`python3 gen_reference.py > fractional_max_pool_reference.json`） |
| 入力 | `torch.manual_seed(3826)` の `randn`（値は JSON に収録済み） |
| 標本 | 各ケースで明示（`2d_overlap_equal_feasibility` は `[0.0, 1 - 2^-24]` の両端） |
| 内容 | `cases`: samples・input・output_values・indices・weights・input_grad（`sum(y * w)` の backward）、`probes`: 実行可能性の式と範囲外標本の受理可否 |

## sha256

```
0d29f39bc2284e5a82fc7e989fa5b716a4bb36ce1cd6fb1cb0a0b82afe58e90b  fractional_max_pool_reference.json
f6b542efba246dcac3fefaab82a0f6175deb33f0963421e76b19aee52fff9c4f  gen_reference.py
```

## 突合の結果

`crates/autodiff/src/fractional_max_pool_ops.rs` の単体テスト `matches_pytorch_reference_fixture` が、
全 5 ケースで値（bit 一致）・索引（完全一致）・入力勾配（bit 一致）を確かめる。tolerance は使わない。
NaN を含む窓は fixture に含めない（本実装は窓内で最初の NaN の索引、PyTorch は最後の NaN で既知差分）。
`feasibility_matches_pytorch_probes_and_out_of_range_samples_are_rejected` が実行可能性の式
（2d は `out + k - 1 <= in`、3d は `out + k - 1 < in`）と標本の範囲外の扱いを確かめる。
