# LocalResponseNorm・weight_norm・spectral_norm（#2646）CUDA／Metal 実機未実測の申し送り

`docs/autodiff-lrn-weight-reparam-decision.md` §10 参照。本実装エージェント実行環境は CUDA／Metal 実機に到達できないため、
`fandhe_ai_autodiff::{lrn_ops, weight_reparam_ops}` の `crates/facade/tests/lrn_weight_reparam_backend_parity.rs` のうち
CUDA（`Device::Cuda(0)`）・Metal（`Device::Metal`。`cfg(target_os = "macos")` 限定）を対象とする 2 テスト
（`cuda_lrn_weight_reparam_match_cpu_reference`・`metal_lrn_weight_reparam_match_cpu_reference`）は `#[ignore]` の
まま未実測である。

## 測定コマンド案

```sh
# CUDA（DGX Spark GB10 等の実機上で）
cargo test -p fandhe-ai --test lrn_weight_reparam_backend_parity -- --ignored --nocapture cuda

# Metal（Apple Silicon 実機上で）
cargo test -p fandhe-ai --test lrn_weight_reparam_backend_parity -- --ignored --nocapture metal
```

CPU（`CpuBackendOps`）版は同テストファイルの属性なしテスト（`cpu_matches_naive_reference`）で既に検証済み（green。
NaiveOps tape のホストフォールバックと bit 一致）。

## 期待結果

本実装は新規 GPU カーネルを持たず、CUDA／Metal は新規 3 フック（`lrn_forward`・`weight_norm_forward`・
`spectral_norm_forward`）を override しない（既定 `Unsupported`）。したがって本テストは **新規 GPU カーネルの parity では
なく、既定 `Unsupported` → ホストフォールバック経路が CPU tape と bit 一致することの確認**である。

- forward は共有ホストカーネルを通り、CPU tape（`CpuBackendOps` の override も同じ共有カーネル）と bit 一致する見込み。
  VJP・`SpectralNormState` の power iteration は両 tape ともホスト側のみで bit 一致する見込み。
- 周辺の演算（`mul`・`sum`・backward の勾配集約）は各バックエンドの既存カーネルを通るため、この部分が外れた場合は本機能
  ではなく当該バックエンドの既存契約違反として扱う。
- **外れた場合は REQ-2 違反として tolerance を緩めず**、原因を特定して別イシューで扱う（`.claude/rules/coding-rust.md`）。
- 将来 GPU 専用の LRN／weight_norm／spectral_norm カーネルを実装する場合は、正規化統計の `f64` 相当契約（Metal は
  `double` 非対応のため補償和または `soft_f64` 系）の適用要否を別イシューの決定記録で判断する。

この前提が崩れる場合は本 README の「期待結果」を更新し、判定を外れた事実を PR へ記録すること。

## 実測記入欄

| 日付 | 実機 | コマンド | 結果 | 備考 |
|---|---|---|---|---|
| 未実測 | DGX Spark GB10 | 上記 CUDA | 未実測 | |
| 未実測 | Mac mini（Apple M4） | 上記 Metal | 未実測 | Linux 環境から Mac mini に届かない（接続先情報なし）。Mac 側へ申し送り（#3162）。事前登録規則は `metal/RULE.txt` |

#2851 以降、本テストは `Var` のメソッド経由（`Var::local_response_norm`／`Var::weight_norm`／`Var::spectral_norm`）で同じ経路を測る（テスト名・`#[ignore]`・判定は不変。上表の記入欄は未実測のまま）。

## Metal 実測記入欄（#3162・準備済み・未実測）

- 判定規則・実行コマンド・期待件数（1 件）は事前登録規則 `metal/RULE.txt` を正とする（実機実行より前に固定済み）。
- 本セッションは Linux 環境で Mac mini（Apple M4）に到達できなかったため実機を実行していない。
  pass／fail・`.rev-stamp`・`env_info.txt`・ログは作っておらず、結果は「未実測」のままである。
- Mac 側の手順: 展開ツリーの `.rev-stamp` に HEAD の全桁 SHA を書いて転送（`docs/real-hardware-verification-env.md` §3・§7）、
  ノード側で `.rev-stamp` 一致と他の cargo／rustc 不在を確認、`RULE.txt` のコマンドを実行、
  ログ・env_info（before／after）をホスト名・ユーザー名・個人パスをマスクして `metal/` へ置き、
  `metal/rev-stamp.txt` に `.rev-stamp` を写し、本節と上の記入欄へ pass／fail 件数を記録する。
- fail の場合は tolerance・baseline・入力を変えず、失敗件数・テスト名・最大誤差を記録する。
