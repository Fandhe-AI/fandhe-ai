# イシュー #1336 実機ログ（GB10 実機実測完了）

`docs/perf/cuda-host-view-staging-readout.md` §5 の D2H＋読み出し時間
before/after 実測を GB10 実機（DGX Spark GB10・sm_121・CUDA 13.0）で
完了した記録。実測サマリ・採否は同 doc §5.2／§6 を正とし、本 README は
再現手順とファイル一覧のみを記す（内部ホスト名は含めない）。

## ファイル一覧

- `env_info.txt`: 計測前の `nvidia-smi`（utilization／compute-apps）・
  `uptime`・`rustc -V`・`nvcc --version`。
- `ignored-host_view_real_device.log`: ゲート A（`host_view_real_device`
  の `#[ignore]` 全 7 件。うち `with_host_view_cached_pinned_matches_
  download_bit_exact` はイシュー #1336 実機実測フェーズで追加）の実行
  ログ。7 件全 pass。
- `ab-run1.log`〜`ab-run5.log`: A/B 計測（`host_view_staging_readout_
  ab_1336`）の 5 プロセス起動生ログ。各 run は `cargo test` 実行を独立
  プロセスとして起動し、直前に `nvidia-smi --query-compute-apps` で
  他プロセスの不在を確認したうえで実行した（`env_info.txt` と同じ
  「常駐 2 サービス〈ComfyUI・Kokoro〉以外の GPU 使用プロセスなし」
  条件を毎回確認）。
- `aggregate.py`: 上記 5 run から CSV 行を抽出し `phase,n` ごとの
  5 run 中央値（`median_ms` の中央値）を集計する（Python3 標準ライブラリ
  のみ）。
- `aggregate.md`: `aggregate.py` の出力（表）。

## 再現手順

```sh
# 転送（Mac から）
rsync -az --delete --filter=':- .gitignore' \
  --exclude '.git/' --exclude '.codex/' --exclude '.env*' \
  --exclude '.claude/settings.local.json' --exclude '.venv*/' \
  --exclude 'real-hardware-verification-env.local.md' --exclude '_/' \
  ./ <cuda-node>:~/work/rust-ai-library-run-1336/

# ビルド（実機側）
ssh <cuda-node> 'export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH; \
  cd ~/work/rust-ai-library-run-1336 && \
  env CARGO_TARGET_DIR=$HOME/work/target-fandhe-ai \
  cargo build -p fandhe-ai-backend-cuda --release --all-features --tests'

# ゲート A
ssh <cuda-node> 'export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH; \
  cd ~/work/rust-ai-library-run-1336 && \
  env CARGO_TARGET_DIR=$HOME/work/target-fandhe-ai \
  cargo test -p fandhe-ai-backend-cuda --release --all-features \
  --test host_view_real_device -- --ignored --nocapture --test-threads=1'

# A/B（5 プロセス起動。各回の前に nvidia-smi --query-compute-apps で確認）
ssh <cuda-node> 'export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH; \
  cd ~/work/rust-ai-library-run-1336 && \
  env CARGO_TARGET_DIR=$HOME/work/target-fandhe-ai \
  cargo test -p fandhe-ai-backend-cuda --release --all-features \
  --test host_view_staging_readout_ab_1336 -- --ignored --nocapture --test-threads=1'

# 集計
python3 aggregate.py
```

## 結果要約（詳細は `cuda-host-view-staging-readout.md` §5.2／§6）

- ゲート A: 7/7 pass（bit 同一・キャッシュ再利用・`release_host_staging`
  の受け入れ条件を全て満たす）。
- ゲート B（本番 `Pageable` の非後退）: 全 N で `before` 比 +5% 以内
  （N=1024: 0.993x・N=2048: 1.018x）または大幅改善（N=4096: 0.097x）。
  N=4096（64 MiB）は glibc mmap 閾値（32 MiB）を超えるため約 10.3 倍
  改善、N=1024/2048（4/16 MiB）は閾値未満のため差なし相当（想定どおり）。
- ゲート C（情報のみ）: キャッシュ経由 `Pinned` は全 N で `Pageable` より
  さらに速い（N=1024: 約 20%・N=2048: 約 15%・N=4096: 約 6% 高速）。
  `HOST_STAGING_KIND` は本ラン結果のみでは切り替えない
  （既定 `Pageable` を維持。ユーザー判断事項として doc §6 に記録）。
