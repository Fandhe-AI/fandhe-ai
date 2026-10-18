# 3D プーリング 2 種（#2643）CUDA／Metal 実機未実測の申し送り

`docs/autodiff-pool3d-ops-decision.md` §10 参照。本実装エージェント実行環境は CUDA／Metal 実機に
到達できないため、`fandhe_ai_autodiff::pool3d_ops`（`max_pool3d`／`avg_pool3d`。#2850 以降は同テストが facade 公開の `Var::max_pool3d`／`avg_pool3d` 経由で同じ経路を測る）の
`crates/facade/tests/pool3d_ops_backend_parity.rs` のうち CUDA（`Device::Cuda(0)`）・Metal
（`Device::Metal`。`cfg(target_os = "macos")` 限定）を対象とする 2 テスト
（`cuda_pool3d_ops_match_cpu_reference`・`metal_pool3d_ops_match_cpu_reference`）は
`#[ignore]` のまま未実測である。

## 測定コマンド案

```sh
# CUDA（DGX Spark GB10 等の実機上で）
cargo test -p fandhe-ai --test pool3d_ops_backend_parity -- --ignored --nocapture cuda

# Metal（Apple Silicon 実機上で）
cargo test -p fandhe-ai --test pool3d_ops_backend_parity -- --ignored --nocapture metal
```

CPU（`CpuBackendOps`）版は同テストファイルの属性なしテスト（`cpu_matches_naive_reference`）で既に検証済み（green）。

## 期待結果

- CUDA／Metal は `BackendOps::pool3d_max`／`pool3d_avg` を持たず既定の `Unsupported` を返すため、実機テストは
  常に共有ホストカーネル（`fandhe_ai_tensor_core::pool3d`）へフォールバックする経路になる想定。よって本テストは
  **GPU カーネルの parity ではなく、フォールバック経路が CPU tape と同じ結果になることの確認**であり、
  Max は値 bit 一致・索引完全一致、Avg と各 backward は REQ-2 統一複合判定を満たす見込み。
- 将来 GPU 専用カーネルを実装する場合、Max はタイ先勝ち・NaN 伝播規則（最初の NaN の索引）を維持するか、
  Avg は `f64` アキュムレータ契約（Metal は `soft_f64`）の適用要否を、別イシューの決定記録で判断する。

この前提が崩れる場合は本 README の「期待結果」を更新し、REQ-2 判定を外れた事実を PR へ記録すること
（tolerance の単独緩和は行わない。`.claude/rules/coding-rust.md`）。

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
