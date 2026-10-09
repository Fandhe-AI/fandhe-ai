# facade `Tape::gradcheck`（#2847）CUDA／Metal 実機未実測の申し送り

`docs/autodiff-jacobian-hessian-gradcheck-decision.md` §12 参照。本実装エージェント実行環境は
CUDA／Metal 実機に到達できないため、`crates/facade/tests/tape_gradcheck_facade.rs` のうち
CUDA（`Device::Cuda(0)`）・Metal（`Device::Metal`。`cfg(target_os = "macos")` 限定）を対象とする 2 テスト
（`cuda_gradcheck_matches_cpu_reference`・`metal_gradcheck_matches_cpu_reference`）は `#[ignore]` のまま未実測である。

## 測定コマンド案

```sh
# CUDA（DGX Spark GB10 等の実機上で）
cargo test -p fandhe-ai --test tape_gradcheck_facade -- --ignored --nocapture cuda

# Metal（Apple Silicon 実機上で）
cargo test -p fandhe-ai --test tape_gradcheck_facade -- --ignored --nocapture metal
```

CPU（`Device::Cpu`）版は同テストファイルの属性なしテストで検証済み（内部実装の結果と全フィールド一致）。

## 期待結果

- 新規カーネルは存在しない。`Tape::gradcheck` は `tape_for(device)` で評価ごとにテープを作り、既存の
  `jacobian`／`backward` と forward の再評価を呼ぶだけである。確認対象は「実機テープでも CPU と同じ合否・
  同じ検査要素数になること」。
- Metal では評価ごとにデバイス存在確認（`tape_for` 内。#2114）が走るため、評価回数 `1 + 2·Σn_k` に比例して遅くなる。
- 前提が崩れる場合は本 README の「期待結果」を更新し、REQ-2 判定を外れた事実を PR へ記録すること
  （tolerance の単独緩和は行わない。`.claude/rules/coding-rust.md`）。

## 実測記入欄

| 日付 | 実機 | コマンド | 結果 | 備考 |
|---|---|---|---|---|
| 2026-10-09 | DGX Spark GB10 | `cargo test -p fandhe-ai --test tape_gradcheck_facade -- --ignored --nocapture cuda` | pass 1 / fail 0（`cuda_gradcheck_matches_cpu_reference` ok。running 1 test） | main `8bbeb874ceb4748cbcf01b52e7162d02812bf8ba`・`rustc 1.97.0 (2d8144b78 2026-07-07)` |
| 2026-10-09 | Apple Silicon（M4 Max） | `cargo test -p fandhe-ai --test tape_gradcheck_facade -- --ignored --nocapture metal` | pass 1 / fail 0（`metal_gradcheck_matches_cpu_reference` ok。running 1 test） | main `8bbeb874ceb4748cbcf01b52e7162d02812bf8ba`・Apple M4 Max・macOS 27.0（26A428）・`rustc 1.98.1 (48a229cea 2026-09-01)` |
