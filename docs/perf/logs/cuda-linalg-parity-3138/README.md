# CUDA GPU 線形代数の GB10 実機 parity（Stage C。#3138）申し送り

**状態: 未実測（Linux 側スキャフォールドのみ）。** 実測は DGX Spark GB10 での実施が必要。事前登録規則は [`RULE.txt`](./RULE.txt)（実測前に固定）。設計記録は `docs/backend-gpu-linalg-decision.md` §5・§10。

## 対象テスト

| 層 | ファイル | 備考 |
|---|---|---|
| backend（LU 系） | `crates/backend-cuda/tests/linalg_lu_parity.rs` | `#[ignore]` 6 件（NVRTC ロード・det・det 集合・solve・inv・特異写像） |
| backend（分解系） | `crates/backend-cuda/tests/linalg_factor_parity.rs` | `#[ignore]` 6 件（NVRTC ロード・cholesky・qr・eigh・非 SPD・eigh 非収束） |
| facade（ゲート ON） | `crates/facade/tests/linalg_gpu_optin_cuda_parity.rs` | 6 件。`CudaBackendOps` 直接呼びで `Unsupported` でないこと（デバイス経路）を確認してから `Var` で比較 |
| 参考（カーネルは通らない） | `crates/facade/tests/linalg_ops_backend_parity.rs` の `cuda_*` 2 件 | ゲート OFF のフォールバック経路 |

## 測定コマンド（GB10 上。手順は `docs/real-hardware-verification-env.md` §3・§4）

```sh
cargo test -p fandhe-ai-backend-cuda --release --test linalg_lu_parity -- --ignored --nocapture --test-threads=1
cargo test -p fandhe-ai-backend-cuda --release --test linalg_factor_parity -- --ignored --nocapture --test-threads=1
cargo test -p fandhe-ai --release --test linalg_gpu_optin_cuda_parity -- --ignored --nocapture --test-threads=1
cargo test -p fandhe-ai --release --test linalg_ops_backend_parity -- --ignored --nocapture --test-threads=1 cuda_
```

## 手順

1. 転送元で `.rev-stamp`（1 行目 sha・2 行目 `dirty=0`）を作り、rsync 除外フィルタ（`.env*`・`settings.local.json` 等）を守って転送する。ノード側で sha を確認する。
2. 上記 4 コマンドを 1 セットとして独立プロセスで 5 回実行し、`run1.log`〜`run5.log` に保存する（上書きしない。計測中は他のビルドを並走させない）。
3. ホスト名・ユーザー名・絶対パスを `<cuda-node>`・`<home>` へマスクして収録する。環境情報（ドライバ・NVRTC 版）も同梱する。

## ログ書式

- run ごとに `test result:` 行の pass／fail 件数を記録する。
- fail 時は `assert_parity` のメッセージ（文脈・`fail_count`・最大誤差）を事実のまま転記する。
- 1e±20 倍・`1e20 × I₂` での NaN／inf の有無を記録する。

## fail したとき

tolerance・判定式・入力集合・κ 上限は緩めない。fail は事実として記録し、原因と是正方針を下記に書いて #3135（LU 系）／#3136（分解系）の追加作業として報告する。baseline 方式へは、別の決定記録とユーザー承認なしに移らない。

原因と是正方針: （未記入）

## 実測記入欄

| 日付 | 実機 | コミット（.rev-stamp） | コマンド | 結果（5 run） | 1e±20 の NaN／inf | 備考 |
|---|---|---|---|---|---|---|
| 未実測 | DGX Spark GB10 | 未実測 | 上記 4 コマンド | 未実測 | 未実測 | スキャフォールド PR。実測は別 PR |
