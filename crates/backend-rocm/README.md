# fandhe-ai-backend-rocm

ROCm（AMD GPU）バックエンドの雛形（イシュー #3154）。非公開（`publish = false`）。

- HIP ランタイムは `libloading` で実行時にロードする。ビルド時にはリンクしないため、
  ROCm 非搭載の環境（CI の ubuntu-latest・macOS・Windows）でもビルドとテストが通る。
- 現時点の公開 API は `is_available()` と `RocmError` のみ。HIP のカーネルは実行しない。
- 設計の正本: `docs/backend-rocm-hip-ffi-decision.md`
