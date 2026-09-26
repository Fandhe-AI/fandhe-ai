# LBFGS PyTorch 参照値フィクスチャの出自

イシュー #2197（親 #2172「LBFGS」）の
`tests/nn_optim_lbfgs.rs::lbfgs_matches_pytorch_reference` が参照する
固定フィクスチャ。`rmsprop-pytorch-reference/README.md`（イシュー #1743
先例）と同じ方針で、生成条件・sha256 を記録した状態で JSON をコミット
する。CI は Python/PyTorch に依存せず、コミット済み JSON のみを
読む（`.claude/rules/ci.md`「グローバル状態を汚す処理を workflow に
書かない」）。

## 生成条件

- **実 PyTorch 実行値**（アルゴリズム擬似コードからの再実装ではない）。
  `torch.__version__ == "2.14.0+cpu"`（`pip install --index-url
  https://download.pytorch.org/whl/cpu torch` で venv に導入し、
  `gen_reference.py` を 1 回実行して生成した）。
- **演算順のドリフト確認**: 導入した torch 2.14.0+cpu の
  `torch/optim/lbfgs.py`（`LBFGS.step`・`_strong_wolfe`・
  `_cubic_interpolate`）を実装前に読み、実装計画（イシュー #2197）§3
  の移植チェックリストと完全一致することを確認済み（two-loop
  recursion の `al`/`be` 添字順・`ys > 1e-10` の履歴更新条件・
  `state["n_iter"] == 1`〈**グローバル**〉での方向初期化・終了判定の
  順序 `n_iter_local == max_iter` → `current_evals >= max_eval` →
  `opt_cond` → `max|d·t| <= tolerance_change` → `|loss - prev_loss| <
  tolerance_change`）。
- **目的関数**: 凸・良条件の最小二乗回帰
  `loss = mean((X @ w + b - y) ** 2)`（`X` shape `[8, 4]`・`w` shape
  `[4]`・`b` shape `[1]`）。`random.Random(seed)` で `X`・真の重み・
  ノイズ付き `y`・初期 `w`/`b` を生成した（ケースごとに異なる seed）。
  Rust 側 parity テストの closure は同じ解析式を f32 で直接計算する
  （`d loss/d w_j = (2/N) Σ_n x_{n,j} (pred_n - y_n)`・
  `d loss/d b = (2/N) Σ_n (pred_n - y_n)`。Tape 駆動 closure での検証は
  本 fixture の対象外で、別途 `nn_optim_lbfgs.rs` の収束テストが担う）。
- **`lr` は 1.0 / 0.5 のみを使う**（2 の冪）。`_strong_wolfe` の
  `_cubic_interpolate` 内 `3*(f1-f2)/(x1-x2)` 等は初回ブラケット
  （`t_prev=0`、`t=lr`）では Python float（f64）の部分積が最終的に f32
  テンソルへキャストされる。`lr` が 2 の冪であれば f64→f32 の丸め値が
  純 f32 演算の結果と一致するため、Rust 側の全 f32 実装と bit 一致する
  （それ以外の `lr`〈例 `0.1`〉では ULP 差が生じうる）。
- 全ケースは `line_search_steps`（Rust 側の実装固有パラメータ。§2.3）
  `>= max_eval` を満たす設定でのみ生成しており、strong Wolfe の
  `max_ls` は PyTorch 側の `max_eval - current_evals` がそのまま効く
  （Rust 側キャップは拘束しない）。
- 各 outer `step()` 呼び出しごとに、更新後 `w`/`b`・累積
  `func_evals`・累積 `n_iter`・初回損失（`orig_loss`）・その step 内の
  closure 呼び出し回数（`closure_calls_this_step`。デバッグ用途で
  JSON には残すが Rust 側 parity 判定では使わない）を記録した。
- 生成された全 step・全要素の中に非有限値は含まれない
  （`gen_reference.py` の closure が `torch.isfinite` を全呼び出しで
  アサートしており、生成時に例外なく完走したことで確認済み）。
- **`strong_wolfe_default` は `max_iter=10`**（既定 `20` ではなく）に
  制限している。この問題設定は収束末期（loss の変化量が停止判定
  `tolerance_change`（`1e-9`）に近づく反復）で loss が
  `1.7e-4` 近辺に停滞し、f32/f64 混在演算に由来するごく僅かな ULP 差
  （実測: 12 反復目時点で PyTorch と Rust 実装の loss 値が
  `~7e-10` 差）が「打ち切り判定」の分岐そのものを反転させることを
  実装時（イシュー #2197）に確認した（advisor 助言の「ケースの入力・
  問題設定を変更する」対応。tolerance・判定式・移植ロジックは変更
  していない）。`max_iter=10` はこの近接領域に到達する前（loss の
  変化量が `1e-9` から十分離れている段階）で打ち切ることで回避する。

## ケース一覧（`gen_reference.py::CASES`）

| ケース | 設定 | 目的 |
|---|---|---|
| `fixed_step_default` | `lr=1.0, max_iter=20`, line search なし | 固定ステップモードの基本経路 |
| `strong_wolfe_default` | `lr=1.0, max_iter=10`, `line_search_fn="strong_wolfe"` | strong Wolfe の基本経路 |
| `small_history` | `history_size=2`, strong Wolfe | 履歴 FIFO 破棄経路 |
| `multi_step_carry_fixed` | `lr=0.5, max_iter=4` を 4 回 `step()` 呼び出し、line search なし | step 間の状態持ち越し・グローバル `n_iter==1` 分岐が最初の step の最初の反復のみで成立すること |
| `multi_step_carry_wolfe` | 同上・strong Wolfe | 同上（line search あり） |
| `tolerance_grad_early_return` | `tolerance_grad=10.0` | 初回勾配が閾値以下でパラメータ不変・`func_evals==1`・`n_iter==0` |

## 再生成手順

```bash
python3 -m venv /path/to/venv
/path/to/venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
/path/to/venv/bin/python gen_reference.py   # このディレクトリで実行. lbfgs_reference.json を上書きする
sha256sum lbfgs_reference.json gen_reference.py   # 本 README の値と照合する
```

## sha256（改竄検知用）

```
6627f04f1cc1b530ecc201153e50e4fbe5f3a3653866fcdf099ba4e4ece62d14  lbfgs_reference.json
85bc72ae37a10c8c42cf5952f4df022b75cc84ef656cb53cdf195e1b77480bb8  gen_reference.py
```
