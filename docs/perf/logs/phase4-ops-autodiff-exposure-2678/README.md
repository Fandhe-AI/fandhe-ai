# Phase 4 の演算・自動微分の facade 公開の実機 parity 申し送り（イシュー #2678）

対象は `Var` の委譲メソッド（FFT・低精度 forward・逆三角／双曲線・非有限値・累積・順序統計・ヒストグラム／二分探索・
形状・索引付き更新・テンソル積・pad モード・活性化）と facade `Tape` の `bincount`・`bincount_weighted`・`jacobian`・`hessian`・
`backward_detect_anomaly`。本エージェントの実行環境には CUDA／Metal 実機への到達手段がないため、実機テストは
**未実測**のまま GB10・M4 Max セッションへ申し送る。CPU 側の正しさは Linux で実行可能な
`crates/facade/tests/phase4_ops_facade.rs`（閉形式・手計算値）と各内部クレートのテストで検証済み。tolerance・baseline は変更しない。

## 新しい数値経路がないこと

公開は内部の自由関数（`crates/autodiff/src/<module>.rs`）への 1 行委譲だけで、新規 `Op`・`BackendOps` メソッド・カーネルを足していない。
このため facade 経由の実機 parity は、各機能の内部クレート側 `*_backend_parity.rs`（`#[ignore]`）の結果がそのまま適用される。

## 判定契約

REQ-2 統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満。`fandhe_ai_backend_cpu::assert_parity`）。結合順序が単一の連続 K ループと
異なる経路（Tensor Core・split-K 等）は `.claude/rules/coding-rust.md` の baseline 非後退方式に従う。新しい tolerance 定数は作らない。

## 実行コマンド

各機能の `crates/facade/tests/<名前>_backend_parity.rs`（`#[ignore]`）を実機で `--ignored` 実行する。機能ごとの手順・実測の詳細は
`docs/perf/logs/` の各機能のログ（`trig-ops-2634`・`fft-rfft-irfft-2631` 等）が正。

```
# CUDA（DGX Spark GB10）
for t in fft_ops low_precision_ops trig_ops nonfinite_ops cumulative_ops stat_reduce_ops binning_ops shape_view_ops \
         indexed_update_ops tensor_product_ops pad_modes_ops activation_scalar_ops softmin_threshold_ops \
         jacobian_hessian gradcheck_anomaly; do
  cargo test -p fandhe-ai --test ${t}_backend_parity -- --ignored cuda_
done
# Metal（M4 Max）: 上と同じ対象に `-- --ignored metal_`
```

## 結果記入欄（未実測）

| 環境 | 対象 | 結果 | 日付 |
|---|---|---|---|
| GB10 | 上記機能の `*_backend_parity.rs`（`--ignored cuda_`） | 未実測 | |
| M4 Max | 上記機能の `*_backend_parity.rs`（`--ignored metal_`） | 未実測 | |

## 保留（本イシューでは公開していない）

- `hinge_embedding_loss`・`soft_margin_loss`・`multilabel_margin_loss`: `Reduction` を `fandhe_ai::nn::loss::Reduction` で名指しできる前提（#2602 未マージ）
- `Tape::gradcheck` と `GradcheckOptions`／`GradcheckReport`: 決定記録に facade シグネチャ（レシーバ・テープ生成の受け方）が未記載
- 行 12〜15（3D プーリング・ConvTranspose3d／MaxUnpool・Fold／Unfold・LRN／weight_norm／spectral_norm）: 承認の対象外
