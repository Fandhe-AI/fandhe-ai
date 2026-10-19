# compat::Sequential の Upsample・ZeroPad2d・Identity 実機 parity 申し送り（イシュー #2522）

> CUDA は #3163 で GB10 実測済み（2026-10-19・コミット b640875f・pass 2/2）。結果は末尾「CUDA 実測記入欄（#3163）」節。Metal は未実測（#3164）。
> Metal は Mac mini（Apple M4）で実測済み（#3164・2026-10-19）。結果は結果記入欄と末尾の「Metal 実測記入欄（#3164）」節、成果物は `metal/` を参照。

本エージェントの実行環境には CUDA／Metal 実機への到達手段がないため、実機テストは未実測のまま
GB10・M4 Max セッションへ申し送る。CPU 側の正しさは Linux 実行可能な
`crates/facade/tests/compat_sequential_spatial_layers.rs` で検証済み。tolerance・baseline は変更しない。

## 判定契約

- `ZeroPad2d`・`Identity`・`Upsample(Nearest)`: 算術を含まない整数添字・コピー系のため CPU と **bit 完全一致**
- `Upsample(Bilinear)`: REQ-2 統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満。`assert_parity`）

## 実行コマンド

```
# CUDA（DGX Spark GB10）
cargo test -p fandhe-ai --test compat_sequential_spatial_layers_backend_parity -- --ignored cuda_
# Metal（Mac mini〈Apple M4〉）
cargo test -p fandhe-ai --test compat_sequential_spatial_layers_backend_parity -- --ignored metal_
```

## 結果記入欄（CUDA は実測済み・Metal は未実測）

| 環境 | テスト | 結果 | 日付 |
|---|---|---|---|
| GB10 | cuda_spatial_layers_nearest_bit_exact | pass | 2026-10-19 |
| GB10 | cuda_spatial_layers_bilinear_matches_cpu | pass | 2026-10-19 |
| Mac mini（Apple M4） | metal_spatial_layers_nearest_bit_exact | pass | 2026-10-19 |
| Mac mini（Apple M4） | metal_spatial_layers_bilinear_matches_cpu | pass | 2026-10-19 |

## CUDA 実測手順（#3163）

- 判定規則は事前登録の `cuda/RULE.txt` を正とする（実測前に固定）。GB10 行は本節の手順で実測済み（下記「CUDA 実測記入欄」）。
- 実行: `cargo test -p fandhe-ai --test compat_sequential_spatial_layers_backend_parity -- --ignored --nocapture --test-threads=1 cuda_`
- 回収: ログ・`env_info.txt`・`.rev-stamp` をマスク（RULE.txt 記載）して `cuda/` に置き、上の結果記入欄の GB10 行と本節を実測値で更新する。

## CUDA 実測記入欄（#3163）

- 判定規則は事前登録の `cuda/RULE.txt` を正とし、実測後も変更していない。tolerance 定数（`RELATIVE_TOLERANCE`・`ABSOLUTE_RESCUE_THRESHOLD`）・baseline・判定式・テストコード・カーネルは不変。
- 実行: 展開ツリー内で `timeout 1800 cargo test -p fandhe-ai --test compat_sequential_spatial_layers_backend_parity -- --ignored --nocapture --test-threads=1 cuda_`（debug プロファイル・並列度 1・1 回実行・再実行なし）。本実測の 9 ファイルは表の順に直列で実行した。
- 対象コミット: `b640875f86399515b2dab4dfc38089caaa5e5c08`（`cuda/rev-stamp.txt`。`dirty=0`）。ファイルごとの実行後にノード側 `.rev-stamp` を読み出し、全 9 回とも同コミットであることを確認した。
- 環境: NVIDIA GB10（ドライバ 580.173.02）・CUDA 13.0（nvcc 13.0.88）・rustc 1.97.0・aarch64。詳細は `cuda/env_info.txt`。実行前後とも GPU 使用率 0 %で、compute-apps は常駐サービス 3 件のみ（before／after で同一。ホスト・プロセス名は `<resident-process>` にマスク）。常駐サービスは停止していない。
- 逸脱: 展開先は並列セッションとの衝突を避けるため、`docs/real-hardware-verification-env.md` §3 の既定ではなく専用ディレクトリ（`~/work/rust-ai-library-run-3163`）にした。`CARGO_TARGET_DIR` は §3 のとおり同期ツリー外の共有 target。
- ウォームアップ: 9 ファイル一括の `--no-run` ビルドを判定外で先に実施した（`cuda/build.log`）。ビルド成功（exit 0）。
- GPU 経路（GPU フック経由かホストフォールバックか）は本実測では確認していない。判定は各テストが `Device::Cuda(0)` で CPU と比較した結果（pass）に限る。

| ファイル | 関数 | 結果 | 備考 |
|---|---|---|---|
| `compat_sequential_spatial_layers_backend_parity.rs` | `cuda_spatial_layers_nearest_bit_exact` | pass（`cuda/compat_sequential_spatial_layers_backend_parity.log`） | bit 一致・`2 passed; 0 failed`・rc=0 |
| `compat_sequential_spatial_layers_backend_parity.rs` | `cuda_spatial_layers_bilinear_matches_cpu` | pass（`cuda/compat_sequential_spatial_layers_backend_parity.log`） | REQ-2・`2 passed; 0 failed`・rc=0 |

合計: 2 件 pass・0 件 fail・0 件 undetermined。

## Metal 実測記入欄（#3164）

- 判定規則: 事前登録の `metal/RULE.txt`（実機実行より前の独立コミットで固定。事後に変更していない）
- 実行日: 2026-10-19（UTC）。機体: Mac mini（Apple M4・16GiB・macOS 27.0.1・rustc 1.99.0）。ホスト名は記録しない
- 対象コミット: `b640875f86399515b2dab4dfc38089caaa5e5c08`（`dirty=0`。`metal/rev-stamp.txt`。`crates/` の指紋はローカルとノードで一致を確認）
- 実行コマンド: `cargo test -p fandhe-ai --test compat_sequential_spatial_layers_backend_parity -- --ignored --nocapture --test-threads=1 metal_`（debug・ノードに `timeout` が無いため perl alarm 1800 秒。各 1 回のみ実行。判定外の `--no-run` ビルドを先に実施: 9 ファイル共通で `spatial-2522/metal/build.log`）
- 実行前後の環境と `.rev-stamp`: `metal/env_info.txt`（実行前後とも他の cargo／rustc プロセスなし）
- 生ログ: `metal/compat_sequential_spatial_layers_backend_parity.log`（末尾に `rc=`。ホスト名・ユーザー名・ホームパスはマスク済み）

| ファイル | 関数 | 結果 |
|---|---|---|
| `compat_sequential_spatial_layers_backend_parity.rs` | `metal_spatial_layers_nearest_bit_exact` | pass（rc=0） |
| `compat_sequential_spatial_layers_backend_parity.rs` | `metal_spatial_layers_bilinear_matches_cpu` | pass（rc=0） |

- 合計: 2 件中 pass 2／fail 0／undetermined 0（`running 2 tests` が期待件数と一致）。
- tolerance 定数・baseline・判定式・テストコードは不変。
- 注記: Metal の GPU カーネル経路を通ったか、Unsupported 経由のホストフォールバック経路を通ったかは本ログから断定できず未確認。
