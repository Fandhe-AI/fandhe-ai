# イシュー #2071 実機ランブック（AMP 低精度 forward の Conv2d・MultiheadAttention 拡張・CUDA／Metal 実測）

## 位置づけ

本ディレクトリはイシュー #2071（AMP 低精度 forward〈#1960／#1961〉の
Conv2d・MultiheadAttention 拡張）の CUDA／Metal 実機 parity 実測の
受け皿である。本 PR の実行環境（Linux コンテナ／worktree）には DGX
Spark GB10・Apple Silicon いずれの実機への到達手段もないため、**実測
値は一切含まれていない**。実測は GB10（CUDA）・Mac（Metal）それぞれ
の実機を持つセッションへ申し送る。

## 対象テスト

`crates/facade/tests/amp_conv_mha_low_precision_backend_parity.rs`
（MultiheadAttention のみ。`#[ignore]`。CUDA 2 件・Metal 2 件
〈`cfg(target_os = "macos")` 限定〉）:

- `cuda_mha_low_precision_forward_f16_matches_cpu`
- `cuda_mha_low_precision_forward_bf16_matches_cpu`
- `metal_mha_low_precision_forward_f16_matches_cpu`（macOS 限定）
- `metal_mha_low_precision_forward_bf16_matches_cpu`（macOS 限定）

**Conv2d は本ファイルの対象外**（`docs/autodiff-low-precision-linear-
design.md` §8.5 に理由を記録済み）: `nn::conv2d_forward_low_precision`
が要求する `Conv2dVars` は `Conv2d::bind`（crate-internal。facade 非
公開の `&fandhe_ai_autodiff::Tape` を要求）以外に構築する公開経路を
持たず（`MultiheadAttentionVars::new` のような直接構築コンストラクタ
が存在しない）、新規 `pub` コンストラクタの追加は facade／autodiff
公開面拡張としてユーザー承認事項の判断になるため本イシューでは追加
しなかった。`Conv2dVars::new` 新設は別途ユーザー承認を得たうえでの
後続イシュー候補として記録する（`out-of-scope-tracking.md`）。

## 実行コマンド

```bash
# CUDA（GB10）
cargo test -p fandhe-ai --release --test amp_conv_mha_low_precision_backend_parity \
  -- --ignored --nocapture --test-threads=1 cuda_

# Metal（Mac）
cargo test -p fandhe-ai --release --test amp_conv_mha_low_precision_backend_parity \
  -- --ignored --nocapture --test-threads=1 metal_
```

`--test-threads=1` は他の実機ランブック（`conv-realdevice-1771`）と
同じ理由（run-to-run 決定性チェックのため）で付与する。本ファイルは
CUDA 専用テストが `cfg(target_os = "macos")` で除外されないため
（`Device::Cuda` の `.expect()` が非 CUDA 環境で panic する設計）、
Metal 実機での実行時は `metal_` 部分一致フィルタを付けて CUDA 専用
テストを除外する。

## 事前登録判定規則

- **判定**: `fandhe_ai_backend_cpu::parity::assert_parity`（REQ-2 統一
  複合判定。相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）が CUDA／
  Metal vs CPU（同一 dtype の低精度 forward）で fail 0 件であること。
- **対象**: `nn::multihead_attention_forward_low_precision` の forward
  のみ（backward は常に f32 のため対象外。`docs/autodiff-low-
  precision-linear-design.md` の「backward は意図的に f32 のまま」と
  同じ理由）。
- tolerance／baseline は変更しない（`.claude/rules/coding-rust.md`）。
- 実機到達不能のまま出荷する場合は `verdict=undetermined` とし、
  申し送り先（本イシュー #2071・親 #1626／#1648）を明記する。

## 記入欄

| 項目 | 値 |
|------|----|
| 実行日時（UTC） | 未実測 |
| ホスト | masked |
| CUDA / Metal 実測結果 | 未実測 |
| verdict | undetermined |
