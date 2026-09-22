# Adam／AdamW 常駐 step Metal カーネル（`MetalBackendOps::
adam_step_device`）実機ランブック（イシュー #2070）

イシュー #2070 は `docs/perf/logs/adam-device-step-1959/README.md`
「後続イシューが実装すべきカーネル（未着手）」・`docs/perf/logs/
adam-device-step-cuda-2069/README.md`「Metal 側は引き続き未実装」が
挙げていた Metal 側（`crates/backend-metal/src/{adam.rs, adam_model.rs,
shaders/adam.metal}`・`context_cache::cached_adam`・`ops.rs::
MetalBackendOps::adam_step_device(_tracked)` override）を実装した。
本ディレクトリはその実機実測の記録先である（**本セッションは Linux
環境のため Metal 実機〈Apple Silicon〉に到達できず、実測は未実施の
まま申し送る**。`out-of-scope-tracking.md` 対象）。

## 現状（本イシュー #2070 時点）

- Metal カーネル実装は完了（`crates/backend-metal/src/adam.rs`・
  `adam_model.rs`・`shaders/adam.metal`）。Linux で実行可能な契約
  テスト（`crates/backend-metal/tests/adam_device_contract.rs`・
  `crates/backend-metal/tests/adam_source_evidence.rs`・
  `crates/backend-metal/src/adam.rs`／`adam_model.rs` 内 unit test）は
  すべて pass 済み（`adam_step_host_model_bit_matches_cpu_reference_
  across_100_steps` により、カーネルへ写像した演算列が CPU 参照実装
  〈`CpuBackendOps::adam_step_device`〉と bit 一致することは実機なしで
  ロック済み）。
- `cargo check -p fandhe-ai-backend-metal --tests --target
  aarch64-apple-darwin`・`cargo check -p fandhe-ai --tests --target
  aarch64-apple-darwin` は green（cross コンパイル確認）。
- 実機（Apple Silicon・M4 Max 等）実測は本イシューでは未実施。

## 実行コマンド（後続セッションでの実施を想定）

```sh
# Linux（実装セッション）で完了済みの型検査・非実機テスト:
cargo test -p fandhe-ai-backend-metal --lib adam_model
cargo test -p fandhe-ai-backend-metal --test adam_device_contract
cargo test -p fandhe-ai-backend-metal --test adam_source_evidence
cargo check -p fandhe-ai-backend-metal --tests --target aarch64-apple-darwin
cargo check -p fandhe-ai --tests --target aarch64-apple-darwin

# Apple Silicon 実機（Mac セッション）:
cargo test -p fandhe-ai-backend-metal --release --test adam_device_parity -- \
  --ignored --nocapture

# 既存 #[ignore] 群の非後退確認（SGD 常駐経路等）:
make test-ignored-metal
make test-ignored-metal-facade
```

## 保存すべきログ

- 上記各コマンドの標準出力（pass/fail・所要時間・`--nocapture` の
  bit-exact 要素数出力）
- `docs/real-hardware-verification-env.local.md.example` に準じた
  `env_info.txt`（内部ホスト名・パスを含めない範囲での実行環境情報。
  ログ内パスは `<home>` へ置換する）

## 事前登録判定規則

`docs/perf/logs/adam-device-step-1959/README.md`「事前登録判定規則」・
`docs/perf/logs/adam-device-step-cuda-2069/README.md`（#1959 README
規則の CUDA 版解釈）と同じ解釈を Metal 側にも適用する（イシュー #2070
本文の「5 run・checksum 完全一致」は規則 2「run-to-run bit 同一」と
同一事項と解釈する）。

1. **正しさ（REQ-2）**: `adam_device_parity.rs` の各ケース（Coupled／
   Decoupled × wd=0／wd≠0 の 100 step 累積、奇数 numel〈65,549〉の
   複数 threadgroup 境界ケース）で、GPU 常駐 N step の最終 `param` が
   CPU `CpuBackendOps::adam_step_device`（ホスト `Adam::step`／
   `AdamW::step` と bit 一致済み）と統一複合判定（相対誤差 1e-3 未満
   または 絶対誤差 1e-5 未満）で一致すること。
2. **bit 完全一致**（Issue 受入条件。REQ-2 とは独立に判定）:
   `..._bit_identical_to_cpu_reference` として `to_bits()` 全要素一致
   を検証する。FAIL は是正せず記録のみとする（事前登録済み。緩和・
   是正はしない）。
3. **run-to-run bit 同一**: 同一入力・同一設定で 5 回実行し、出力の
   `to_bits()` 列が 5 回とも完全一致すること
   （`run_to_run_bit_identical_across_5_runs`）。
4. **SGD 常駐経路の非後退**: `make test-ignored-metal`・`make
   test-ignored-metal-facade`（既存 `#[ignore]` 群）で
   `sgd_device_parity`・facade `device_param_store_backend_parity` が
   後退しないこと。
5. **付帯記録（判定には使わない）**: Metal vs CPU の bit 一致要素数／
   総要素数、各テストの所要時間（性能判定なし）。
6. FAIL は是正せず記録のみとする（事後緩和なし）。`env_info.txt` は
   内部ホスト名を含めず、ログ内パスは `<home>` へ置換する。

## スコープ外（本イシューでは対応しない。将来の拡張）

- RMSprop・Adagrad の常駐 step（別 issue）。
- facade 経路（`Tape::step_device_param_store_adam`／`_adamw` on
  Metal）の end-to-end `#[ignore]` テスト。
- 本 README への Apple Silicon 実機実測の記入（Mac セッション）。

## 実測記録（未記入・後続セッションで追記）

- 実行日時／環境:
- コマンド出力サマリ:
- bit-exact 要素数（付帯記録）:
- 判定結果:
