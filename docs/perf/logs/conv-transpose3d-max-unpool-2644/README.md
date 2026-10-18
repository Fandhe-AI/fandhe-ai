# ConvTranspose3d・MaxUnpool（#2644）CUDA／Metal 実機未実測の申し送り

`docs/autodiff-conv-transpose3d-max-unpool-decision.md` §10 参照。本実装エージェント実行環境は CUDA／Metal 実機に
到達できないため、`fandhe_ai_autodiff::conv_transpose3d_ops`・`max_unpool_ops`（#2850 以降は同テストが facade 公開の `Var::conv_transpose3d`／`Var::max_unpool1d/2d/3d` 経由で同じ経路を測る）の
`crates/facade/tests/conv_transpose3d_max_unpool_backend_parity.rs` のうち CUDA（`Device::Cuda(0)`）・Metal
（`Device::Metal`。`cfg(target_os = "macos")` 限定）を対象とする 2 テスト
（`cuda_conv_transpose3d_max_unpool_match_cpu_reference`・`metal_conv_transpose3d_max_unpool_match_cpu_reference`）は
`#[ignore]` のまま未実測である。

## 測定コマンド案

```sh
# CUDA（DGX Spark GB10 等の実機上で）
cargo test -p fandhe-ai --test conv_transpose3d_max_unpool_backend_parity -- --ignored --nocapture cuda

# Metal（Apple Silicon 実機上で）
cargo test -p fandhe-ai --test conv_transpose3d_max_unpool_backend_parity -- --ignored --nocapture metal
```

CPU（`CpuBackendOps`）版は同テストファイルの属性なしテスト（`cpu_matches_naive_reference`）で既に検証済み（green）。

## 期待結果

本実装は新規 GPU カーネルを持たない。したがって本テストは **GPU カーネル新設の parity ではなく、既存フックを
実機が通る経路が CPU tape と同じ結果になることの確認**である。

- ConvTranspose3d: `gemm_batched`（forward）・`gemm_batched_fp32_strict`（VJP）は実機の GEMM が走る。
  `im2col3d`／`col2im3d` は CUDA／Metal とも既定 `Unsupported` を返す（`crates/backend-cpu/tests/
  backend_ops_dispatch.rs` の `*_im2col3d_col2im3d_are_unsupported_not_panic` で固定）ため、autodiff 側の
  ホストフォールバック（`eval::im2col3d`／`col2im3d`）へ到達する。forward・`d_input`・`d_weight`・`d_bias` は
  REQ-2 統一複合判定を満たす見込み（CUDA の TF32 opt-in を有効にした場合は `docs/spec/04-requirements.md` REQ-2 の
  Tensor Core 経路の判定方式に従う）。
- MaxUnpool: 既存の GPU `scatter`（forward。`Overwrite`）・`gather`（VJP）が実機で走る。重複索引を含む最後の書き手の
  契約が GPU カーネルでも保たれるかが確認点で、forward・勾配とも bit 一致する見込み。**GPU の `scatter` が重複索引で
  CPU と異なる勝者を返す場合は REQ-2 違反ではなく `Overwrite` の決定性契約の違反**であり、別イシューで扱う
  （tolerance の単独緩和は行わない。`.claude/rules/coding-rust.md`）。
- 将来 GPU 専用カーネル（`im2col3d`／`col2im3d`・unpool 専用）を実装する場合は、`col2im3d` の `f64` アキュムレータ契約
  （Metal は `soft_f64`）と `Overwrite` の最後の書き手契約の適用要否を別イシューの決定記録で判断する。

この前提が崩れる場合は本 README の「期待結果」を更新し、REQ-2 判定を外れた事実を PR へ記録すること。

## 実測記入欄

| 日付 | 実機 | コマンド | 結果 | 備考 |
|---|---|---|---|---|
| 未実測 | DGX Spark GB10 | 上記 CUDA | 未実測 | |
| 未実測 | Mac mini（Apple M4） | 上記 Metal | 未実測 | Linux 環境から Mac mini に届かない（接続先情報なし）。Mac 側へ申し送り（#3162）。事前登録規則は `metal/RULE.txt` |

## Metal 実測記入欄（#3162・準備済み・未実測）

- 判定規則・実行コマンド・期待件数（1 件）は事前登録規則 `metal/RULE.txt` を正とする（実機実行より前に固定済み）。
- 本セッションは Linux 環境で Mac mini（Apple M4）に到達できなかったため実機を実行していない。
  pass／fail・`.rev-stamp`・`env_info.txt`・ログは作っておらず、結果は「未実測」のままである。
- Mac 側の手順: 展開ツリーの `.rev-stamp` に HEAD の全桁 SHA を書いて転送（`docs/real-hardware-verification-env.md` §3・§7）、
  ノード側で `.rev-stamp` 一致と他の cargo／rustc 不在を確認、`RULE.txt` のコマンドを実行、
  ログ・env_info（before／after）をホスト名・ユーザー名・個人パスをマスクして `metal/` へ置き、
  `metal/rev-stamp.txt` に `.rev-stamp` を写し、本節と上の記入欄へ pass／fail 件数を記録する。
- fail の場合は tolerance・baseline・入力を変えず、失敗件数・テスト名・最大誤差を記録する。
