# SWA（AveragedModel・SWALR）PyTorch 参照値フィクスチャの出自

イシュー #2658（親 #2657）の `tests/nn_swa.rs` が参照する固定フィクスチャ。生成条件・sha256 を記録した状態で
JSON をコミットする。CI は Python／PyTorch に依存せず、コミット済み JSON のみを読む（`.claude/rules/ci.md`）。

## 生成条件

- **実 PyTorch 実行値**。`torch.__version__ == "2.14.0+cpu"`・Python 3.14.4（JSON の `torch_version`／
  `python_version` に記録。スクリプトは 2.14.0 系でなければ assert で止まる）。numpy や手計算値で代替していない
  （numpy 未導入の使い捨て venv で実行）。乱数は固定シード 2658・`torch.set_num_threads(1)`。再生成で同一 JSON になる
  （2 回実行して同一 sha256 を確認）。
- f32 値（パラメータ入力・平均値）は **u32 のビットパターン配列**で保存する。学習率系列は Python float（f64）。
- `swalr_cases`（10 件）: `SGD([p], lr=base_lr)` に `SWALR` を組み、`step()` を 30 回呼んで epoch 0〜30 の
  `param_groups[0]["lr"]` を記録。cos／linear × `anneal_epochs` 5・10、`swa_lr > base_lr`（cos・linear）、
  `anneal_epochs` 1（cos・linear）、`anneal_epochs` 0（cos・linear）。`base_lr`／`swa_lr` は f32 で厳密に表せる値へ
  丸めてから torch へ渡す。
- `swalr_chain_case`: `CosineAnnealingLR(T_max=20)` を主スケジューラとし、PyTorch ドキュメントの定番ループ
  （`epoch > swa_start` なら `swa_scheduler.step()`、それ以外は `scheduler.step()`。`swa_start=9`）で系列を作る
  （`SequentialLR` は使わない）。主スケジューラの実ステップ回数は 10 で、これが切替 step（`switch_step`）。
- `avg_cases`（5 件）: 各反復で `update_parameters` へ渡したパラメータ（入力 `params`）と、更新後の平均値
  （`averaged`）・`n_averaged`（出力）の両方を記録する（Rust 側は torch の RNG を再現できないため）。
  `linear_sgd_12`（`nn.Linear(3, 2)` を実 SGD で 12 step）・`random_walk_200`（200 更新）・`single_update`・
  `constant_updates`・`mixed_scale`（1e-6〜1e6 の大小混在）。
- NaN／inf の混入が無いことを生成時に assert している。

## `inspect.getsource` で確認した実際の演算経路（torch 2.14.0+cpu）

- `AveragedModel.update_parameters`: `n_averaged == 0` はパラメータの `copy_`。以降は、CPU は
  `_get_foreach_kernels_supported_devices()` に含まれないため `get_swa_avg_fn()` の
  `averaged + (current - averaged) / (n_averaged + 1)`（**除算形**）を使う。本実装は lerp + `f32::mul_add` 形のため
  演算順が異なり、bit 一致は主張しない（統一複合判定のみ）。
- `SWALR.get_lr`: 直前の `group["lr"]` から初期 lr を `(lr - alpha*swa_lr)/(1 - alpha)` で逆算する再帰形。
  `anneal_epochs == 0` は `step = max(1, step)` により構築直後（epoch 0）から `swa_lr`。cos は
  `(1 - cos(pi*t))/2`、`t` は `[0, 1]` にクリップ。

## 再生成手順

```
python3 -m venv /tmp/venv && /tmp/venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.0
cd crates/autodiff/tests/fixtures/swa-pytorch-reference && /tmp/venv/bin/python gen_reference.py
```

## sha256

```
9439ac8c7474a962546f24645388fd6aaab030646ece98b12a029bbe54a68fc9  swa_reference.json
6c95262f8d84ef0eb165fb6157ac940121168975fff32e6782e80490dca518b7  gen_reference.py
```
