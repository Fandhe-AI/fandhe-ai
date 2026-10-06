# PolynomialLR・ChainedScheduler PyTorch 参照値フィクスチャの出自

イシュー #2659（親 #2657）の `tests/nn_optim_lr_scheduler_poly_chained.rs` が参照する固定フィクスチャ。生成条件・
sha256 を記録した状態で JSON をコミットする。CI は Python／PyTorch に依存せず、コミット済み JSON のみを読む
（`.claude/rules/ci.md`）。

## 生成条件

- **実 PyTorch 実行値**。`torch.__version__ == "2.14.0+cpu"`・Python 3.14.4（JSON の `torch_version`／
  `python_version` に記録。スクリプトは 2.14.0 系でなければ assert で止まる）。numpy 未導入の使い捨て venv で実行し、
  numpy や手計算値で代替していない。再生成で同一 JSON になる（2 回実行して同一 sha256 を確認済み）。
- 各ケースは `SGD([p], lr=base_lr)` にスケジューラを組み、`step()` を 30 回呼んで epoch 0〜30 の
  `param_groups[0]["lr"]`（Python float = f64）を記録する（#2176 の `lr_sequence` と同型）。
- 入力値は f32 で厳密に表せる値のみ（base_lr 0.5／0.25／0.125、gamma 0.5／0.875／0.9375、start_factor 0.25、
  power 0／0.5／1／2／3）。torch は f64・Rust は f32 引数のため丸め差の増幅を避ける。
- `poly_cases`（6 件）: `PolynomialLR` の T=5/p=1、T=10/p=2、T=7/p=0.5、T=1/p=1、T=40/p=3（系列中に 0 へ未到達）、
  T=5/p=0。
- `chain_cases`（4 件）: `ChainedScheduler(..., optimizer=opt)` で StepLR+ExponentialLR、逆順の
  ExponentialLR+StepLR、LinearLR(end_factor=1)+MultiStepLR、LinearLR+PolynomialLR+ExponentialLR。
  PyTorch の `ConstantLR(factor, total_iters)` は Rust の `ConstantLr`（常に `base_lr`）と別物のためメンバーに使わない。
- `non_multiplicative_probe`: `CosineAnnealingLR(T_max=10, eta_min=0)` + `ExponentialLR(0.875)` のチェーン。
  Rust の積の閉形式が一致しない（意図的な制限）ことを固定する記録で、通常ケースには混ぜない
  （`CosineAnnealingLR` の再帰形は加算項を持つため）。
- NaN／inf の混入が無いことを生成時に assert している。

## `inspect.getsource` で確認した実際の演算経路（torch 2.14.0+cpu）

- `PolynomialLR.get_lr`: 再帰形。`last_epoch > total_iters` は現在値のまま、それ以外は
  `lr * ((1 - t/T) / (1 - (t-1)/T)) ** power`。閉形式 `base_lr·(1 − min(t,T)/T)^power` との差は 30 step で最大
  相対 6.0e-16（計画時実測）。
- `ChainedScheduler.step`: 各メンバーの `step()` を順に呼ぶ（同一 optimizer の lr へ乗算が連鎖する）。
  メンバーが現在値への純粋な乗算なら積の閉形式と一致する（最大 7.4e-16。メンバー順にも依存しない）。

## 再生成手順

```
python3 -m venv /tmp/venv && /tmp/venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.0
cd crates/autodiff/tests/fixtures/lr-scheduler-poly-chained-pytorch-reference && /tmp/venv/bin/python gen_reference.py
```

## sha256

```
e99acbbbf11183beb32b692c265f891023bc78d1c62e39968c8af14c693a2ed8  lr_scheduler_poly_chained_reference.json
90686a912e2f57de13de15b4ea0a708806374b4c37e74f06b8d9ecdfd1db4fdf  gen_reference.py
```
