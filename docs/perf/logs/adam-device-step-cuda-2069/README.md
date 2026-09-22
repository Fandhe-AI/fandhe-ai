# Adam／AdamW 常駐 step CUDA カーネル（`CudaBackendOps::
adam_step_device`）実機ランブック（イシュー #2069）

イシュー #2069 は `docs/perf/logs/adam-device-step-1959/README.md`
「後続イシューが実装すべきカーネル（未着手）」が挙げていた CUDA 側
（`crates/backend-cuda/src/{adam.rs, kernels_adam.rs}`・
`context_cache::cached_adam`・`ops.rs::CudaBackendOps::
adam_step_device(_tracked)` override）を実装した。本ディレクトリは
その実機実測の記録先である（**本セッションは CUDA 実機
〈DGX Spark GB10 等〉に到達できないため、実測は未実施のまま申し送る**。
`out-of-scope-tracking.md` 対象）。

Metal 側は引き続き未実装のまま
（`docs/perf/logs/adam-device-step-1959/README.md` 参照）。

## 現状（本イシュー #2069 時点）

- CUDA カーネル実装は完了（`crates/backend-cuda/src/adam.rs`・
  `kernels_adam.rs`）。Linux（GPU・driver 不在）で実行可能な契約テスト
  （`crates/backend-cuda/tests/adam_device_contract.rs`・
  `crates/backend-cuda/src/{adam.rs, kernels_adam.rs}` 内 unit test・
  `ops.rs::tests::adam_step_device_rejects_on_poisoned_ordinal_before_
  device_handle_is_attempted`）はすべて pass 済み。
- 実機（DGX Spark GB10 等）実測は本イシューでは未実施。

## 実行コマンド（後続セッションでの実施を想定）

```sh
# Linux（実装セッション）で完了済みの型検査・非実機テスト:
cargo test -p fandhe-ai-backend-cuda --lib adam
cargo test -p fandhe-ai-backend-cuda --test adam_device_contract
cargo test -p fandhe-ai-backend-cuda --test adam_device_real_device --no-run

# DGX Spark GB10（CUDA 実機）:
cargo test -p fandhe-ai-backend-cuda --release --test adam_device_real_device -- \
  --ignored --nocapture

# 既存 #[ignore] 群の非後退確認（SGD 常駐経路・CUDA Graph capture 等）:
make test-ignored-cuda
```

## 保存すべきログ

- 上記各コマンドの標準出力（pass/fail・所要時間・`--nocapture` の
  bit-exact 要素数出力）
- `docs/real-hardware-verification-env.local.md.example` に準じた
  `env_info.txt`（内部ホスト名・パスを含めない範囲での実行環境情報。
  ログ内パスは `<home>` へ置換する）

## 事前登録判定規則

`docs/perf/logs/adam-device-step-1959/README.md`「事前登録判定規則」を
主として採用する（Issue #2069 本文の「5 run 中央値・checksum 完全一致・
bit 完全一致（性能判定なし）」は #1959 README 規則 2「run-to-run bit
同一」と同一事項と解釈する）。

1. **正しさ（REQ-2）**: `adam_device_real_device.rs` の各ケース
   （Coupled／Decoupled × wd=0／wd≠0 の 100 step 累積、奇数 numel
   〈65,549〉の複数ブロック境界ケース）で、GPU 常駐 N step の最終
   `param` が CPU `CpuBackendOps::adam_step_device`（ホスト
   `Adam::step`／`AdamW::step` と bit 一致済み）と統一複合判定（相対
   誤差 1e-3 未満 または 絶対誤差 1e-5 未満）で一致すること。
2. **run-to-run bit 同一**: 同一入力・同一設定で 5 回実行し、出力の
   `to_bits()` 列が 5 回とも完全一致すること
   （`run_to_run_bit_identical_across_5_runs`）。
3. **SGD 常駐経路の非後退**: `make test-ignored-cuda`（既存
   `#[ignore]` 群）で `sgd_device_real_device`・`graph_capture_real_
   device`・facade `device_param_store_backend_parity` が後退しない
   こと。
4. **付帯記録（判定には使わない）**: GPU vs CPU の bit 一致要素数／
   総要素数（設計目標は全要素 bit 一致。REQ-2 pass かつ bit 差がある
   場合はその旨を記録し、緩和・是正はしない）、各テストの所要時間
   （性能判定なし）。
5. FAIL は是正せず記録のみとする（事後緩和なし）。`env_info.txt` は
   内部ホスト名を含めず、ログ内パスは `<home>` へ置換する。

## 実測記録（未記入・後続セッションで追記）

- 実行日時／環境:
- コマンド出力サマリ:
- bit-exact 要素数（付帯記録）:
- 判定結果:
