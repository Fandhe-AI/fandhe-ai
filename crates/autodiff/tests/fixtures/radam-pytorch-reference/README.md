# RAdam PyTorch 参照値フィクスチャの出自

イシュー #2171（親 #2131）の
`tests/nn_optim_radam.rs::radam_matches_pytorch_reference` が参照する
固定フィクスチャ。`rmsprop-pytorch-reference/README.md`（イシュー #1743
先例）と同じ方針で、生成条件・sha256 を記録した状態で JSON をコミット
する。CI は Python/PyTorch に依存せず、コミット済み JSON のみを読む
（`.claude/rules/ci.md`「グローバル状態を汚す処理を workflow に書か
ない」）。

## 生成条件

- **実 PyTorch 実行値**。`torch.__version__ == "2.14.0+cpu"`（`pip
  install --index-url https://download.pytorch.org/whl/cpu
  torch==2.14.0` で venv に導入し、`gen_reference.py` を 1 回実行して
  生成した）。
- **演算順のドリフト確認**: 導入した torch 2.14.0+cpu の
  `torch/optim/radam.py::_single_tensor_radam` を実装前に読み、演算順
  （weight decay（`decoupled_weight_decay` なら `param *= 1-lr*wd`、
  それ以外は `grad += wd*param`）→ `exp_avg.lerp_(grad, 1-beta1)` →
  `exp_avg_sq = beta2*v + (1-beta2)*grad^2` →
  `bc1 = 1-beta1**step`・`bc2 = 1-beta2**step` →
  `bias_corrected_exp_avg = exp_avg / bc1` →
  `rho_inf = 2/(1-beta2) - 1` →
  `rho_t = rho_inf - 2*step*beta2**step/bc2` →
  `rho_t > 5.0` なら
  `rect = sqrt((rho_t-4)(rho_t-2)*rho_inf / ((rho_inf-4)(rho_inf-2)*rho_t))`・
  `adaptive_lr = sqrt(bc2) / (sqrt(exp_avg_sq)+eps)` を用いて
  `param -= bias_corrected_exp_avg * lr * adaptive_lr * rect`（左結合の
  評価順）、それ以外は `param -= bias_corrected_exp_avg * lr`）と完全
  一致することを確認済み。
- **`rho_t` 分岐境界の検査**: `gen_reference.py` が各 step の `rho_t`
  を計算し、(a) 10 step 以内に両分岐（`<= 5.0` と `> 5.0`）を通ること、
  (b) どの step の `rho_t` も 5.0 から `1e-3` 以上離れていること
  （Rust 側が `beta2 as f64` で計算する際の丸めで分岐が反転しない
  ため）を生成時に assert 済み。JSON の各ケースに `rho_t`（step 順の
  リスト）を記録する。実測では全ケースで step 5→6（`beta2=0.999` は
  `4.996 → 5.994`、`beta2=0.9` は `4.581 → 5.390`）で分岐が切り替わる
  ことを確認した。
- 2 つの独立パラメータ: `param_a`（shape `[2, 3]`）・`param_b`
  （shape `[4]`）。`random.Random(seed=20260926)` で初期値・10 step 分
  の勾配系列を固定生成し、`torch.optim.RAdam` の `step()` を 10 回
  呼んで各 step 後のパラメータ値を記録した。
- 5 ケース（`gen_reference.py::CASES`）:
  - `default`: `lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8,
    weight_decay=0.0, decoupled_weight_decay=False`（`torch.optim.RAdam`
    の既定値そのもの）
  - `beta2_small`: `beta2=0.9`（rectified 分岐へ早く入る。他は既定）
  - `weight_decay`: `weight_decay=0.1`（coupled。他は既定）
  - `decoupled_weight_decay`: `weight_decay=0.1,
    decoupled_weight_decay=True`（他は既定）
  - `all`: `lr=0.01, beta1=0.8, beta2=0.95, eps=1e-6, weight_decay=0.02,
    decoupled_weight_decay=True`
- 初期値・勾配系列はケース間で共通。生成された 10 step × 5 ケース ×
  全要素の中に NaN は含まれない。

## 再生成手順

```bash
python3 -m venv /path/to/venv
/path/to/venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.0
/path/to/venv/bin/python gen_reference.py   # このディレクトリで実行. radam_reference.json を上書きする（境界 assert も再実行される）
sha256sum radam_reference.json gen_reference.py   # 本 README の値と照合する
```

## sha256（改竄検知用）

```
245634b0a5065d7976800e1e1bf2f0bbea276a3dedaa7aaa5911848ae6246588  radam_reference.json
73b428f150d67e5e262f939fa8be95a13bd8c03a29ef06b7d84bbbe4443cec2f  gen_reference.py
```
