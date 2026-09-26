# LR scheduler 拡張 5 種 PyTorch 参照値フィクスチャの出自

イシュー #2176（親 #2131）の
`tests/nn_optim_lr_scheduler_ext.rs::*_matches_pytorch_reference` が参照
する固定フィクスチャ。`radam-pytorch-reference/README.md`（イシュー
#2171 先例）と同じ方針で、生成条件・sha256 を記録した状態で JSON を
コミットする。CI は Python/PyTorch に依存せず、コミット済み JSON のみ
を読む（`.claude/rules/ci.md`「グローバル状態を汚す処理を workflow に
書かない」）。

## 生成条件

- **実 PyTorch 実行値**。`torch.__version__ == "2.14.0+cpu"`（`pip
  install --index-url https://download.pytorch.org/whl/cpu
  torch==2.14.0` で一時 venv に導入し、`gen_reference.py` を 1 回実行
  して生成した）。
- 各ケースは `torch.optim.SGD([p], lr=base_lr)` に対応するスケジューラ
  を組み、引数なしの `step()` を 30 回呼んで `param_groups[0]['lr']`
  を epoch 0（構築直後）〜 30（`step()` 30 回後）の系列として記録した
  （`lr` フィールド。長さ 31）。`lr[e]` は本実装の `lr_at(e)`（`e` は
  `step()` 呼び出し回数と同一の 0 始まり通し番号）と対応する。
- 10 ケース（`gen_reference.py::main`）:
  - `multi_step`: `MultiStepLR`。未ソート・重複を含む milestones
    （`[3,3,7,2]`・`gamma=0.5`）と、ソート済み milestones
    （`[5,10,15]`・`gamma=0.1`）の 2 ケース
  - `cosine_annealing_warm_restarts`: `CosineAnnealingWarmRestarts`。
    `(T_0=4, T_mult=1)`・`(T_0=2, T_mult=2, eta_min=0.01)` の 2 ケース
  - `cyclic`: `CyclicLR`（`mode='triangular'`・`cycle_momentum=False`）。
    `step_size_up=3, step_size_down=None`・`step_size_up=2,
    step_size_down=5` の 2 ケース
  - `lambda`: `LambdaLR`。`0.95**e`（`geometric`）・`1/(e+1)`
    （`harmonic`）の 2 ケース
  - `sequential_2stage`: `SequentialLR([LinearLR(0.1, total_iters=3),
    ExponentialLR(0.9)], milestones=[3])`
  - `sequential_3stage`: `SequentialLR([LinearLR(0.1, total_iters=3),
    CosineAnnealingLR(T_max=5, eta_min=0.01*base_lr),
    MultiStepLR(milestones=[2,4], gamma=0.5)], milestones=[3, 8])`
- 生成された 31 epoch × 10 ケースの中に NaN は含まれない
  （`gen_reference.py::main` の生成時 assert）。
- **`lr_scheduler.step()` を `optimizer.step()` より先に呼ぶ警告**
  （`UserWarning: Detected call of lr_scheduler.step() before
  optimizer.step()`）が生成時に出力される。本フィクスチャはスケジュー
  ラの学習率系列のみを対象とし `optimizer.step()`（パラメータ更新）を
  一切呼ばないため無害（警告文どおり「最初の値をスキップする」効果は
  `param_groups[0]['lr']` の記録タイミングを epoch 0 とみなす本スクリ
  プトの設計と整合している。実際に epoch 0 の値は各スケジューラの
  初期学習率と一致することを手計算で確認済み）。

## 演算順の確認

- `MultiStepLR`: milestones を `Counter` で保持し `step` 以下の
  milestone の**多重**カウントを `gamma` の指数として使う
  （torch 2.14.0+cpu `torch/optim/lr_scheduler.py::MultiStepLR.get_lr`）。
  ソート済み milestones に対する `partition_point(|&m| m <= step)` は
  この多重カウントと同値（本実装の `MultiStepLr::lr_at` 実装コメント
  参照）。
- `CosineAnnealingWarmRestarts`: 引数なし `step()`（`last_epoch += 1`
  を内部の `T_cur`／`T_i` 更新式で追跡する経路。`epoch` 引数を渡す
  浮動小数 `log` 経路とは異なる）を使用。本実装はこの整数演算経路と
  一致する（`CosineAnnealingWarmRestarts::cycle_position` の doc
  「PyTorch との意図的な相違」節参照）。
- `CyclicLR`: `cycle_momentum=False` を明示（`SGD` は `momentum=0`
  既定で `cycle_momentum=True` でも動作しうるが、momentum cycling は
  本実装の対象外〈`CyclicLr` doc 参照〉のため無効化して生成した）。
- `SequentialLR`: 各段の境界（`milestones`）で内部スケジューラを
  `last_epoch=0` へ戻し `step(0)` 相当の処理を行う
  （`SequentialLR.step` 実装）。本実装の「局所 epoch = step -
  milestones[idx-1]」はこの再開挙動の閉形式である
  （`SequentialLr::lr_at` doc 参照）。

## 再生成手順

```bash
python3 -m venv /path/to/venv
/path/to/venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.0
/path/to/venv/bin/python gen_reference.py   # このディレクトリで実行. lr_scheduler_ext_reference.json を上書きする
sha256sum lr_scheduler_ext_reference.json gen_reference.py   # 本 README の値と照合する
```

## sha256（改竄検知用）

```
40a48c7f9af4ea38341c96a746ac98a35aa8f26db9d63dc3f5d4b943fcd77a27  lr_scheduler_ext_reference.json
04aa99c32e41c24c54d07af6e769baf0950732351aab6728edf4e9935c875922  gen_reference.py
```
