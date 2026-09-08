# #1436 host-view-readout 後退診断ログ

`docs/perf/cuda-host-view-readout-small-shape-regression.md` の一次情報。
GB10（sm_121）実機実測（2026-09-08）。内部ホスト名は含めない。

## ファイル一覧

- `env_info.txt`: GPU/driver/CUDA/rustc・`getconf PAGESIZE`・THP 設定・glibc
  版・計測前後の `nvidia-smi --query-compute-apps`・`uptime`（load average）
- `layerB-n{1024,2048,4096}-run{1,2,3}.log`: Layer B（`crates/backend-cuda`
  非公開 API 直接呼び出し診断テスト
  `readout_regression_diag_tests_1436`）の 4 腕（`LegacyToVec`／
  `BorrowedKeepAlive`／`BorrowedWithDummyAllocFree`／`PretouchedReusedDest`）
  実行ログ。`--release --test-threads=1 -- --ignored --nocapture`
- `layerB-n1024-malloc-threshold-fixed.log`: `MALLOC_MMAP_THRESHOLD_=64MiB`
  固定時の N=1024 再計測（動的 mmap 閾値適応を無効化する確証実験）
- `off-N{1024,2048,4096}-run{1,2}.jsonl` / `on-N{1024,2048,4096}-run{1,2}.jsonl`:
  Layer A（`bench-fandhe --task gemm --device cuda --mode reuse --phases`）
  の off（既定）／on（`host-view-readout` feature 有効）実行結果。同一
  HEAD の facade path patch（`patch.crates-io.fandhe-ai.path`）でビルドした
  2 バイナリを使用

## 再現手順

```bash
# Layer B（backend-cuda 非公開 API 診断テスト。--test-threads=1 必須）
# 完全一致テスト名 + --exact を使うこと。部分一致フィルター
# （例: `readout_regression_diag_n1024`）は同名 prefix を持つ単腕
# 4 関数（`readout_regression_diag_n1024_legacy_to_vec` 等。#1442
# レビュー指摘対応で新設）にも一致してしまい、4 腕をまとめて実行する
# `readout_regression_diag_n1024` と単腕 4 関数の計 5 関数が同一プロセス
# 内で連続実行され、記録時（4 腕一括のみ）と異なる allocator 状態の
# 引き継ぎが起きる。
env PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH \
    CARGO_TARGET_DIR=$HOME/work/target-fandhe-ai \
cargo test -p fandhe-ai-backend-cuda --release --lib \
  readout_regression_diag_n1024 -- --ignored --nocapture --test-threads=1 --exact

# Layer A（off／on 2 バイナリを facade path patch でビルドしてから実行）
cd scripts/bench/framework-compare
FACADE="$(pwd)/../../../crates/facade"
env PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH \
    CARGO_TARGET_DIR=$HOME/work/target-fandhe-ai-off \
cargo build --release -p bench-fandhe \
  --config "patch.crates-io.fandhe-ai.path=\"$FACADE\""
env PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH \
    CARGO_TARGET_DIR=$HOME/work/target-fandhe-ai-on \
cargo build --release -p bench-fandhe --features host-view-readout \
  --config "patch.crates-io.fandhe-ai.path=\"$FACADE\""

$HOME/work/target-fandhe-ai-off/release/bench-fandhe \
  --task gemm --device cuda --size 1024 --mode reuse --phases --out off-N1024.jsonl
$HOME/work/target-fandhe-ai-on/release/bench-fandhe \
  --task gemm --device cuda --size 1024 --mode reuse --phases --out on-N1024.jsonl
```

## スコープの縮小（時間制約による明記）

- Layer B は N=1024/2048 を各 3 プロセス起動、N=4096 は 1 プロセス起動
  （計画の全サイズ 5 プロセス起動から縮小）。N=2048 の bimodal 挙動
  （2/3 が slow・1/3 が fast）を確認済み
- Layer A は各 off/on 1〜2 プロセス起動（計画の 5 プロセス起動から縮小）
- 腕 (b) の 4/8/12/16/24/32/64 MiB 相当の正方形状スイープ（非単調性の
  精密な可視化）は未実施。N=1024/2048/4096（4/16/64 MiB）の 3 点で
  bimodal 挙動と閾値通過後の収束（N=4096）は確認できたため、粗い
  スイープでも受入基準 2/3 は満たせると判断した
- `strace -c` 等の syscall 回数計測は未実施（`MALLOC_MMAP_THRESHOLD_`
  固定による確証実験で機構仮説を裏付けたため優先度を下げた）
