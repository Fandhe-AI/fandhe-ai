# ベンチマーク結果サマリー（fandhe-ai-introduction）

## 環境

- チップ: Apple M4 Max
- OS: macOS 26.6.2（Darwin 25.6.0）
- ツールチェーン: cargo 1.96.0 (30a34c682 2026-05-25)（`--release` ビルド）
- 計測日: 2026-08-28
- 計測プロトコル: warmup 20 回 → 計測 20 回（学習は 100 ステップ中先頭 20 を warmup、残り 80 を計測）。中央値・Q1・Q3 を記録
- 同期: 計測区間終端で結果テンソルをホストへ実体化し全要素を読み出す（checksum として記録）
- 入力データ: xorshift64* の同一シード・同一生成式で全フレームワーク共通

## 採用バージョン

| フレームワーク | クレート | バージョン |
| --- | --- | --- |
| fandhe-ai | fandhe-ai (facade) | 0.3.0 |
| candle | candle-core (metal feature) | 0.11.0 |
| Burn | burn (ndarray / wgpu backend) | 0.21.0 |

## (a) GEMM（C = A×B、f32、正方行列）

### CPU

| N | フレームワーク | 中央値 | Q1 | Q3 | GFLOP/s |
| --- | --- | --- | --- | --- | --- |
| 256 | fandhe-ai | 363.5 µs | 352.0 µs | 399.9 µs | 92.3 |
| 256 | candle | 339.8 µs | 306.1 µs | 403.1 µs | 98.8 |
| 256 | burn | 364.2 µs | 359.7 µs | 369.1 µs | 92.1 |
| 512 | fandhe-ai | 797.4 µs | 774.9 µs | 826.0 µs | 336.6 |
| 512 | candle | 818.7 µs | 772.6 µs | 876.4 µs | 327.9 |
| 512 | burn | 2.719 ms | 2.681 ms | 2.776 ms | 98.7 |
| 1024 | fandhe-ai | 3.536 ms | 3.339 ms | 3.597 ms | 607.3 |
| 1024 | candle | 3.677 ms | 3.502 ms | 3.787 ms | 584.0 |
| 1024 | burn | 20.679 ms | 20.519 ms | 20.917 ms | 103.8 |
| 2048 | fandhe-ai | 24.102 ms | 22.126 ms | 24.951 ms | 712.8 |
| 2048 | candle | 25.014 ms | 23.895 ms | 26.976 ms | 686.8 |
| 2048 | burn | 162.351 ms | 161.525 ms | 164.626 ms | 105.8 |

### Metal

| N | フレームワーク | 中央値 | Q1 | Q3 | GFLOP/s |
| --- | --- | --- | --- | --- | --- |
| 256 | fandhe-ai | 5.441 ms | 5.352 ms | 5.711 ms | 6.2 |
| 256 | candle | 257.6 µs | 242.3 µs | 286.6 µs | 130.2 |
| 256 | burn | 1.493 ms | 1.479 ms | 1.509 ms | 22.5 |
| 512 | fandhe-ai | 5.724 ms | 5.570 ms | 5.889 ms | 46.9 |
| 512 | candle | 519.0 µs | 499.5 µs | 613.8 µs | 517.2 |
| 512 | burn | 3.044 ms | 3.017 ms | 3.200 ms | 88.2 |
| 1024 | fandhe-ai | 7.894 ms | 7.738 ms | 8.029 ms | 272.1 |
| 1024 | candle | 1.086 ms | 1.079 ms | 1.113 ms | 1976.7 |
| 1024 | burn | 3.613 ms | 3.595 ms | 3.623 ms | 594.3 |
| 2048 | fandhe-ai | 14.576 ms | 14.075 ms | 17.032 ms | 1178.6 |
| 2048 | candle | 4.958 ms | 4.647 ms | 5.125 ms | 3464.9 |
| 2048 | burn | 5.548 ms | 5.532 ms | 5.586 ms | 3096.7 |
| 4096 | fandhe-ai | 46.415 ms | 45.855 ms | 47.355 ms | 2961.1 |
| 4096 | candle | 23.635 ms | 23.416 ms | 23.911 ms | 5815.1 |
| 4096 | burn | 13.194 ms | 13.171 ms | 13.241 ms | 10416.9 |

**Metal 表のプロトコル注意（イシュー #925 レビュー指摘）**: 上表の fandhe-ai 行は `--mode fresh`
（既定）の計測であり、`tape_for(Device::Metal)` をループの各回内で再構築する。candle
（`Device::new_metal(0)`）・Burn はデバイス・入力テンソルをループ外で 1 回だけ構築し `matmul` +
ホスト実体化のみを計測するため、fandhe-ai 行にのみ毎回のデバイス/tape 構築コストが乗る。
GEMM カーネル単体の速度としてではなく、fandhe-ai の「計測ごとに新規グラフを作る」運用コストを
含む数値として解釈する（`README.md` 計測プロトコル節参照。プロトコル完全一致の比較は
`--mode reuse` の `gemm` タスク。環境 3 の (a')「GEMM — CUDA（fresh vs reuse）」参照）。

## (b) MLP 学習（784→256→10、ReLU、バッチ 64、MSE、SGD lr=0.01、1 ステップあたり時間）

| デバイス | フレームワーク | 中央値 | Q1 | Q3 |
| --- | --- | --- | --- | --- |
| cpu | fandhe-ai | 18.185 ms | 17.955 ms | 18.417 ms |
| cpu | candle | 797.5 µs | 770.6 µs | 856.6 µs |
| cpu | burn | 626.5 µs | 625.5 µs | 634.8 µs |
| metal | fandhe-ai | 48.845 ms | 44.350 ms | 51.605 ms |
| metal | candle | 751.8 µs | 706.1 µs | 808.1 µs |
| metal | burn | 1.606 ms | 1.601 ms | 1.612 ms |

## (c) 推論スループット（同 MLP forward のみ、バッチ 64。表のスループットはバッチ/秒 = 1/中央値。1 バッチ = 64 件）

| デバイス | フレームワーク | 中央値 | Q1 | Q3 | バッチ/秒 |
| --- | --- | --- | --- | --- | --- |
| cpu | fandhe-ai | 505.7 µs | 477.6 µs | 537.5 µs | 1977 |
| cpu | candle | 195.7 µs | 131.5 µs | 240.4 µs | 5109 |
| cpu | burn | 282.1 µs | 282.0 µs | 282.3 µs | 3545 |
| metal | fandhe-ai | 24.125 ms | 22.486 ms | 25.086 ms | 41 |
| metal | candle | 251.3 µs | 244.9 µs | 270.3 µs | 3979 |
| metal | burn | 1.503 ms | 1.500 ms | 1.507 ms | 665 |

**(b)/(c) Metal 行のプロトコル注意**: GEMM 表と同様、fandhe-ai の学習・推論は毎ステップ / 毎回
新規 `tape_for(Device::Metal)` を構築する一方、candle / Burn はデバイスを使い回すため、上記
Metal 行の fandhe-ai 数値には毎回のデバイス/tape 構築コストが乗っている（`README.md` 計測
プロトコル節参照。`reuse` モードは `gemm` タスクのみ対応のため train/infer にはこの分離手段が
現時点でない。イシュー #925 のスコープ外）。

## 計測不可・未計測項目

- **CUDA（全フレームワーク）**: 環境 1（Apple M4 Max / macOS）では CUDA デバイスが存在せず計測不可 → **環境 2（DGX Spark、下記）で計測済み**
- **tch-rs（全タスク）**: 未計測。libtorch 依存のため（導入が制限時間内に完了しない見込みで省略）
- 実行時に失敗した組み合わせ: なし（skipped.log は空）

## 備考

- GEMM の入力は全フレームワークで同一（checksum が一致することを JSONL で確認できる）
- 学習・推論の重みは candle / Burn は共有 RNG で同一。fandhe-ai は `Sequential::add_linear` の内部初期化（シード指定）のため重みの値は異なるが、同一アーキテクチャ・同一入力・同一バッチであり実行時間の比較には影響しない
- fandhe-ai の学習ループは公開 API のみ（`compat::Sequential` + `tape.backward` + 手動 SGD）。パラメータ更新はホスト側で `param - lr * grad` を計算して `apply_parameters` で書き戻す実装であり、フレームワークにより更新方式が異なる（candle: `Var::set`、Burn: `from_inner + require_grad`）

## 環境 2: DGX Spark（CUDA / ARM CPU）

- ノード: DGX Spark 実機（内部クラスタの 1 ノード。実ホスト名は `docs/real-hardware-verification-env.local.md` 方式のローカル管理とし、本ドキュメントには書かない）
- SoC: NVIDIA GB10（Blackwell GPU + ARM CPU 20 コア: Cortex-X925 ×10 + Cortex-A725 ×10、ユニファイドメモリ 121 GB）
- OS / カーネル: Ubuntu 24.04、6.17.0-1031-nvidia（aarch64）
- GPU ドライバ: 580.173.02（CUDA 13.0）/ CUDA Toolkit: nvcc 13.0（V13.0.88、`/usr/local/cuda-13.0`）
- ツールチェーン: rustc 1.97.0 / cargo 1.97.0（`--release` ビルド）
- ビルド feature: bench-fandhe はそのまま（fandhe-ai は cfg + 実行時プローブで CUDA 有効化）、bench-candle / bench-burn は `--no-default-features --features cuda`
- 計測日: 2026-08-28
- 計測プロトコル: 環境 1（Apple M4 Max）と同一（warmup 20 → 計測 20、学習は 100 ステップ中先頭 20 を warmup。終端でホスト実体化 + checksum）
- **GPU の同居ワークロード**: 計測時の GPU 使用率 0%、常駐プロセスは ComfyUI（170 MiB）+ Kokoro（870 MiB）のみ。GB10 はユニファイドメモリのため `nvidia-smi --query-gpu=memory.used` は `[N/A]` を返す（プロセス単位の used_memory で確認）。システムメモリは 121 GB 中 available 113 GB
- ノード選定: クラスタ 6 台中 5 台は vLLM が 67〜99 GB を常駐確保していたため、常駐 GPU 利用が最小（約 1 GB）のノードを選定

### (a) GEMM — CUDA（NVIDIA GB10）

| N | フレームワーク | 中央値 | Q1 | Q3 | GFLOP/s |
| --- | --- | --- | --- | --- | --- |
| 256 | fandhe-ai | 440.042 ms | 439.193 ms | 442.070 ms | 0.1 |
| 256 | candle | 79.3 µs | 79.0 µs | 79.6 µs | 423.0 |
| 256 | burn | 381.4 µs | 210.6 µs | 515.0 µs | 88.0 |
| 512 | fandhe-ai | 450.692 ms | 448.581 ms | 452.824 ms | 0.6 |
| 512 | candle | 242.0 µs | 241.3 µs | 242.3 µs | 1109.3 |
| 512 | burn | 314.1 µs | 313.4 µs | 315.7 µs | 854.6 |
| 1024 | fandhe-ai | 435.171 ms | 434.322 ms | 436.265 ms | 4.9 |
| 1024 | candle | 946.4 µs | 944.1 µs | 953.0 µs | 2269.1 |
| 1024 | burn | 1.098 ms | 930.8 µs | 1.108 ms | 1956.4 |
| 2048 | fandhe-ai | 458.350 ms | 454.526 ms | 461.217 ms | 37.5 |
| 2048 | candle | 4.086 ms | 4.081 ms | 4.096 ms | 4204.3 |
| 2048 | burn | 4.096 ms | 4.065 ms | 4.136 ms | 4194.7 |
| 4096 | fandhe-ai | 593.890 ms | 592.453 ms | 598.015 ms | 231.4 |
| 4096 | candle | 60.676 ms | 59.941 ms | 61.133 ms | 2265.1 |
| 4096 | burn | 46.819 ms | 46.683 ms | 47.429 ms | 2935.5 |

### (a) GEMM — CPU（ARM、Cortex-X925/A725 20 コア）

| N | フレームワーク | 中央値 | Q1 | Q3 | GFLOP/s |
| --- | --- | --- | --- | --- | --- |
| 256 | fandhe-ai | 373.7 µs | 351.1 µs | 430.5 µs | 89.8 |
| 256 | candle | 302.8 µs | 276.6 µs | 496.3 µs | 110.8 |
| 256 | burn | 313.5 µs | 301.7 µs | 887.4 µs | 107.0 |
| 512 | fandhe-ai | 2.046 ms | 1.934 ms | 2.171 ms | 131.2 |
| 512 | candle | 1.953 ms | 1.469 ms | 2.270 ms | 137.5 |
| 512 | burn | 2.223 ms | 2.197 ms | 2.225 ms | 120.7 |
| 1024 | fandhe-ai | 7.709 ms | 7.525 ms | 7.865 ms | 278.6 |
| 1024 | candle | 5.351 ms | 5.176 ms | 5.434 ms | 401.4 |
| 1024 | burn | 16.912 ms | 16.908 ms | 16.917 ms | 127.0 |
| 2048 | fandhe-ai | 36.100 ms | 35.536 ms | 36.477 ms | 475.9 |
| 2048 | candle | 32.725 ms | 32.238 ms | 33.576 ms | 525.0 |
| 2048 | burn | 132.800 ms | 132.765 ms | 132.876 ms | 129.4 |
| 4096 | fandhe-ai | 167.429 ms | 166.182 ms | 167.978 ms | 820.9 |
| 4096 | candle | 221.490 ms | 220.051 ms | 224.325 ms | 620.5 |
| 4096 | burn | 1.198 s | 1.197 s | 1.198 s | 114.8 |

### (b) MLP 学習（784→256→10、ReLU、バッチ 64、MSE、SGD lr=0.01、1 ステップあたり時間）

| デバイス | フレームワーク | 中央値 | Q1 | Q3 |
| --- | --- | --- | --- | --- |
| cuda | fandhe-ai | 2.418 s | 2.394 s | 2.429 s |
| cuda | candle | 268.1 µs | 266.4 µs | 275.4 µs |
| cuda | burn | 679.8 µs | 500.1 µs | 867.1 µs |
| cpu | fandhe-ai | 14.133 ms | 13.772 ms | 14.992 ms |
| cpu | candle | 2.447 ms | 2.220 ms | 2.774 ms |
| cpu | burn | 963.2 µs | 943.6 µs | 970.5 µs |

### (c) 推論スループット（同 MLP forward のみ、バッチ 64。表のスループットはバッチ/秒 = 1/中央値。1 バッチ = 64 件）

| デバイス | フレームワーク | 中央値 | Q1 | Q3 | バッチ/秒 |
| --- | --- | --- | --- | --- | --- |
| cuda | fandhe-ai | 1.822 s | 1.813 s | 1.829 s | 0.5 |
| cuda | candle | 41.2 µs | 40.2 µs | 42.3 µs | 24243 |
| cuda | burn | 500.3 µs | 247.7 µs | 525.7 µs | 1999 |
| cpu | fandhe-ai | 360.3 µs | 332.0 µs | 396.7 µs | 2776 |
| cpu | candle | 301.3 µs | 206.8 µs | 386.2 µs | 3319 |
| cpu | burn | 237.0 µs | 236.5 µs | 237.5 µs | 4220 |

### 環境 2 の計測不可・未計測項目

- ビルド不可・実行時失敗の組み合わせ: なし（skipped-cuda.log は空。3 フレームワーク × cuda/cpu × 全タスクを計測）
- tch-rs: 環境 1 と同じ理由で未計測

### 環境 2 の備考

- 生データ: `results/raw/results-dgx.jsonl`（実行ログ: `results/run_all_cuda-dgx.log`）
- **fandhe-ai の CUDA はサイズ非依存の約 440〜460 ms の固定オーバーヘッドが支配的**（N=256〜2048 でほぼ一定、N=4096 でも 594 ms）。計測プロトコルが「計測ごとに新しい `tape_for(Device::Cuda(0))` を作る」ため、tape（CUDA コンテキスト/カーネル）初期化コストが毎回計測区間に入る。学習（毎ステップ新規 tape、2.418 s/step）・推論（1.822 s/回）も同様にこのオーバーヘッドを含む。candle / Burn はデバイス・グラフを使い回す API 設計のため同条件でも初期化を繰り返さない。この差は「同一プロトコル（毎回新規グラフ）」での実測であり、fandhe-ai の GEMM カーネル単体の性能を示すものではない点に注意
- checksum はフレームワーク間で概ね一致（CPU は 6 桁一致）。CUDA は candle / burn で下位桁が環境 1 の Metal と同様に揺れ、burn の CUDA GEMM は checksum のずれがやや大きい（例: N=256 で 237.5467 に対し 237.5806。TF32 等の低精度アキュムレーションの可能性があるが未確認。速度比較の解釈時に留意）
- candle の CUDA GEMM は N=2048（4204 GFLOP/s）→ N=4096（2265 GFLOP/s）で効率が低下する。burn は N=4096 で 2936 GFLOP/s
- ARM CPU の GEMM は環境 1（M4 Max）と同傾向（fandhe-ai と candle が同水準、burn の ndarray バックエンドが 1 桁遅い）。N=4096 では fandhe-ai（821 GFLOP/s）が candle(621 GFLOP/s) を上回った
- 計測は本番稼働サービス（ComfyUI / Kokoro / gateway）を停止せずに実施（GPU 使用率 0% のアイドル時間帯、干渉は上記常駐 1 GB のみ）

## 環境 3: NVIDIA GeForce RTX 3060（デバイス/tape 再利用モード fresh vs reuse 比較。イシュー #925）

- GPU: NVIDIA GeForce RTX 3060（12 GiB、compute capability 8.6）
- OS: Linux（x86_64、7.0.0-30-generic）
- GPU ドライバ: 595.71.05（CUDA 13.2 対応）
- CUDA NVRTC / ランタイムヘッダ: 完全な CUDA Toolkit が本機に未導入のため、`libnvrtc.so.13`（`nvidia-cuda-nvrtc` pip パッケージ v13.0.88）+ `cuda_fp16.h` 等のヘッダ（`nvidia-cuda-runtime`/`nvidia-cuda-cccl` pip パッケージ）を Python venv 内に取得し、`LD_LIBRARY_PATH`（`libnvrtc.so.13` の場所）・`CUDA_INCLUDE_PATH`（`compile_ptx` のフォールバック候補パス経由。`nvrtc.rs` の 2 段構えコンパイル）を実行時に指定して計測した。ドライバの CUDA 13.2 対応より新しい NVRTC（v13.3.x）は `CUDA_ERROR_UNSUPPORTED_PTX_VERSION` になったため v13.0.88 に固定（本体 workspace・framework-compare workspace の依存はいずれも変更していない。あくまで実行時ライブラリの一時的な提供手段）
- ツールチェーン: rustc/cargo 1.96.0（`--release` ビルド）
- 計測日: 2026-08-28
- 対象: `bench-fandhe` の GEMM（`--task gemm --device cuda`）のみ（受け入れ条件 1 の必須項目 N=2048 に加え、サイズ非依存性を確認するため 256/512/1024/4096 も計測）。MLP 学習・推論・candle/burn の CUDA ビルドは本機では `nvcc` 未導入のため実施していない（下記「計測不可・未計測項目」参照）
- 生データ: `results/raw/results-rtx3060.jsonl`（fresh・reuse 両方の行を含む）

### (a) / (a') GEMM — CUDA（fresh vs reuse）

| N | mode | 初期化(init_s) | 中央値 | Q1 | Q3 | GFLOP/s |
| --- | --- | --- | --- | --- | --- | --- |
| 256 | fresh | - | 267.254 ms | 263.324 ms | 268.936 ms | 0.1 |
| 256 | reuse | 477.869 ms | 295.149 ms | 285.541 ms | 311.193 ms | 0.1 |
| 512 | fresh | - | 333.016 ms | 288.506 ms | 350.769 ms | 0.8 |
| 512 | reuse | 484.626 ms | 260.525 ms | 259.086 ms | 261.049 ms | 1.0 |
| 1024 | fresh | - | 275.626 ms | 268.969 ms | 287.997 ms | 7.8 |
| 1024 | reuse | 465.571 ms | 300.868 ms | 283.915 ms | 316.275 ms | 7.1 |
| 2048 | fresh | - | 291.567 ms | 289.523 ms | 296.560 ms | 58.9 |
| 2048 | reuse | 573.555 ms | 297.019 ms | 294.843 ms | 301.696 ms | 57.8 |
| 4096 | fresh | - | 506.611 ms | 484.172 ms | 545.168 ms | 271.3 |
| 4096 | reuse | 684.670 ms | 474.396 ms | 469.370 ms | 480.577 ms | 289.7 |

### 環境 3 の計測不可・未計測項目

- Metal: macOS 実機がこのエージェント実行環境から到達不能のため未計測。再現コマンド:
  `cargo run --release -p bench-fandhe -- --task gemm --device metal --size 2048 --mode reuse`
  （macOS 実機での追試をユーザーへ案内する）
- MLP 学習・推論（train/infer）の reuse モード: `bench-fandhe` は受け入れ条件の範囲（gemm タスクのみ）に限定して実装したため未対応（`docs/spec` 変更を伴わないスコープ判断。PR 本文の対象外項目を参照）
- bench-candle / bench-burn の CUDA ビルド: 本機に `nvcc` が未導入のため `candle-kernels`（build.rs が nvcc を要求）のビルドが失敗し未計測。`--mode reuse` は API 設計上 candle/burn には適用されないため（README 参照）、この欠落は受け入れ条件の達成に影響しない

### 環境 3 の備考

- **reuse モードでも中央値は fresh モードとほぼ同水準（数十 ms 以内の差）で、DGX Spark GB10（環境 2）で観測された「tape 初期化コストの毎回計上」という仮説どおりには初期化コストが消えなかった**。むしろ reuse モードの `init_s`（tape 構築 + 初回 matmul + ホスト実体化）自体が 465〜685 ms とサイズ非依存の大きな値を示し、かつ同一 tape を使い回した 2 回目以降の呼び出し（`median_s`）も 260〜506 ms とほぼ同水準にとどまる。つまり本環境では「初期化コストは tape 構築 1 回限りで、以降の呼び出しはカーネル実行のみの短時間になる」という当初仮説は成立せず、**行列積 1 回ごとに数百 ms の固定オーバーヘッドが繰り返し発生している**ことが実測で判明した
- checksum は fresh/reuse で完全一致（同一入力に対する行列積であり期待どおり。数値的な副作用なし）
- 上記の性質（reuse でも per-call オーバーヘッドが消えない原因）は本イシューの受け入れ条件（fresh/reuse の差分を記録する）を満たす実測結果であり、原因の切り分け自体は未実施（fandhe-ai 側のカーネル選択・ディスクキャッシュ照会等が候補だが未確認）。原因調査は別イシューとして追跡することを推奨する（PR 本文のスコープ外項目を参照）
- 本環境は CUDA Toolkit を完全インストールしていない簡易構成（NVRTC・ヘッダのみを pip 経由で一時取得）のため、DGX Spark（環境 2。nvcc 込みの標準インストール）とはビルド・実行環境が異なる点に留意する
