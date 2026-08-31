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

## データ有効性の注記（イシュー #965）

**Burn(wgpu) Metal GEMM の N=512/1024/2048/4096 の 4 行は無効。実行時間・GFLOP/s を
性能値として扱わない。** 結果テンソルが全ゼロ（`results/raw/results.jsonl` の該当行は
`"checksum":0.000000`）であり、同一入力を使う他フレームワーク・他デバイスの同 N が
すべて 6 桁一致する中で当該 4 行のみ乖離する。原因は Burn 0.21.0（cubek-matmul 0.2.0）
の wgpu/Metal 経路に存在した upstream 既知バグ（tracel-ai/burn#4966 →
tracel-ai/cubek#283 で修正。承認済みピン `burn =0.21.0` の範囲では修正版を取得できない）
であり、本ハーネスの実装起因ではない。切り分けの詳細・追試手順は
`docs/perf/burn-wgpu-metal-gemm-zero-result.md` を参照。

- `results/raw/results.jsonl`（raw 記録）は変更せず残置する。無効データを黙って
  削除・改変しない（`.claude/rules/security.md` A08）
- 下表 Metal 節の該当 4 行は「（無効: checksum 不一致）」を付記し GFLOP/s 列を `-` にする
  （N=256 の burn 行は checksum 一致・有効）
- 再計測は、`tracel-ai/cubek#283`（修正 PR）を含むバージョンへの Burn ピン更新
  （`.claude/rules/deps-policy.md` 第 9 区分・ユーザー承認事項。対象バージョンの
  確定は `docs/perf/burn-wgpu-metal-gemm-zero-result.md` §2.5 参照）後に実施する
- 同種の不具合の再発防止として `summarize.py` が GEMM の checksum をフレームワーク間で
  相互突合し、不一致行を機械的に「無効」表示する（`--strict` で終了コード 2）。
  `bench-common::validate_gemm_checksum` は各バイナリ側でも縮退 checksum（全ゼロ・
  非有限）を emit 前に遮断する（「計測プロトコル」節・README.md 参照）

## データ有効性の注記（要素単位検証。イシュー #970）

本ページが参照する 3 つの raw JSONL（`results/raw/results.jsonl`・`results-dgx.jsonl`・
`results-rtx3060.jsonl`）は、GEMM 結果を参照実装と要素単位で突合する検証（`parity_total`・
`parity_fail_count`・`parity_max_abs_err`・`parity_max_rel_err`。README.md「要素単位検証」節）
の**追加前**に計測されたものであり、当該フィールドを持たない。`summarize.py` はこれを
「無効」ではなく「未検証（旧形式）」として区別して報告する（キー欠損と検証失敗を混同しない。
値そのものの正当性を否定するものではない）。**本 PR では既存の raw JSONL の再計測・数値の
書き換えは行っていない**（実測を捏造しない方針。`.claude/rules/security.md` A08）。次回の
実機再計測キャンペーンから要素単位検証が有効になる。

- **Burn(wgpu) Metal 経路の低精度の可能性**: 上記の checksum 乖離バグとは別に、Burn の
  wgpu/Metal 経路は内部で TF32 相当の低精度演算を使う場合がある（環境 2 の備考に記載の
  既知事項）。checksum 修正後の再計測で要素単位検証が閾値超過（無効）と判定された場合、
  それが実装の破損ではなく精度契約外（TF32 等）の低精度実装に起因する可能性を切り分ける
  必要がある。閾値（本体の数値一致契約と同値）は緩めず、**未検証・要追試**として記録する
  （閾値変更はユーザー承認が必須。`.claude/rules/coding-rust.md`）

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
| 512 | burn（無効: checksum 不一致。注記参照） | 3.044 ms | 3.017 ms | 3.200 ms | - |
| 1024 | fandhe-ai | 7.894 ms | 7.738 ms | 8.029 ms | 272.1 |
| 1024 | candle | 1.086 ms | 1.079 ms | 1.113 ms | 1976.7 |
| 1024 | burn（無効: checksum 不一致。注記参照） | 3.613 ms | 3.595 ms | 3.623 ms | - |
| 2048 | fandhe-ai | 14.576 ms | 14.075 ms | 17.032 ms | 1178.6 |
| 2048 | candle | 4.958 ms | 4.647 ms | 5.125 ms | 3464.9 |
| 2048 | burn（無効: checksum 不一致。注記参照） | 5.548 ms | 5.532 ms | 5.586 ms | - |
| 4096 | fandhe-ai | 46.415 ms | 45.855 ms | 47.355 ms | 2961.1 |
| 4096 | candle | 23.635 ms | 23.416 ms | 23.911 ms | 5815.1 |
| 4096 | burn（無効: checksum 不一致。注記参照） | 13.194 ms | 13.171 ms | 13.241 ms | - |

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
プロトコル節参照。`train` タスクの `reuse` モードは #958/#959 で対応済み（`DeviceParamStore`
によるデバイス常駐パラメータ更新）だが、Apple Silicon 実機での計測は本 PR 時点で未計測
（環境 4「計測不可・未計測項目」参照）。`infer` の `reuse` は引き続き対象外）。

## 計測不可・未計測項目

- **CUDA（全フレームワーク）**: 環境 1（Apple M4 Max / macOS）では CUDA デバイスが存在せず計測不可 → **環境 2（DGX Spark、下記）で計測済み**
- **tch-rs（全タスク）**: 未計測。libtorch 依存のため（導入が制限時間内に完了しない見込みで省略）
- 実行時に失敗した組み合わせ: なし（skipped.log は空）

## 備考

- GEMM の入力は全フレームワークで同一（checksum は Burn(wgpu) Metal N>=512 の 4 行
  （「データ有効性の注記」参照）を除き JSONL で一致することを確認できる）
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
- MLP 学習（train）の reuse モード: 本節（環境 3）の計測当時（イシュー #925）は gemm タスクのみ対応だったが、`train --mode reuse` は #958 で実装済み・#959 でスイープ・集計に統合済み（`DeviceParamStore` によるデバイス常駐パラメータ更新。実測は環境 4 参照）。推論（infer）の reuse は依然未対応（#958/#959 のスコープ外）
- bench-candle / bench-burn の CUDA ビルド: 本機に `nvcc` が未導入のため `candle-kernels`（build.rs が nvcc を要求）のビルドが失敗し未計測。`--mode reuse` は API 設計上 candle/burn には適用されないため（README 参照）、この欠落は受け入れ条件の達成に影響しない

### 環境 3 の備考

- **reuse モードでも中央値は fresh モードとほぼ同水準（数十 ms 以内の差）で、DGX Spark GB10（環境 2）で観測された「tape 初期化コストの毎回計上」という仮説どおりには初期化コストが消えなかった**。むしろ reuse モードの `init_s`（tape 構築 + 初回 matmul + ホスト実体化）自体が 465〜685 ms とサイズ非依存の大きな値を示し、かつ同一 tape を使い回した 2 回目以降の呼び出し（`median_s`）も 260〜506 ms とほぼ同水準にとどまる。つまり本環境では「初期化コストは tape 構築 1 回限りで、以降の呼び出しはカーネル実行のみの短時間になる」という当初仮説は成立せず、**行列積 1 回ごとに数百 ms の固定オーバーヘッドが繰り返し発生している**ことが実測で判明した
- checksum は fresh/reuse で完全一致（同一入力に対する行列積であり期待どおり。数値的な副作用なし）
- 上記の性質（reuse でも per-call オーバーヘッドが消えない原因）は本イシューの受け入れ条件（fresh/reuse の差分を記録する）を満たす実測結果であり、原因の切り分け自体は未実施（fandhe-ai 側のカーネル選択・ディスクキャッシュ照会等が候補だが未確認）。原因調査は別イシューとして追跡することを推奨する（PR 本文のスコープ外項目を参照）
- 本環境は CUDA Toolkit を完全インストールしていない簡易構成（NVRTC・ヘッダのみを pip 経由で一時取得）のため、DGX Spark（環境 2。nvcc 込みの標準インストール）とはビルド・実行環境が異なる点に留意する

## 環境 4: NVIDIA GeForce RTX 3060 / x86_64（train fresh vs reuse。イシュー #957/#958/#959）

- GPU: NVIDIA GeForce RTX 3060（12 GiB、compute capability 8.6）— 環境 3 と同一機
- OS: Linux（x86_64、7.0.0-30-generic）
- GPU ドライバ: 595.71.05（CUDA 13.2 対応）
- CUDA NVRTC / ランタイムヘッダ: 環境 3 と同じ一時プロビジョニング構成（`nvidia-cuda-nvrtc`
  pip パッケージ v13.0.88 由来の `libnvrtc.so.13` + `nvidia-cuda-runtime`/`nvidia-cuda-cccl` の
  ヘッダを `LD_LIBRARY_PATH`/`CUDA_INCLUDE_PATH` で実行時に指定。本体・framework-compare の
  依存は変更していない）
- ツールチェーン: rustc/cargo 1.96.0（`--release` ビルド）
- fandhe-ai バージョン: 0.4.0（環境 3 の 0.3.0 から更新済み。#958/#998 の `train --mode reuse` 実装を含む）
- 計測日: 2026-08-29
- 対象: `bench-fandhe` の `train`（cpu / cuda、fresh・reuse 両方）。加えて CPU 横並び参考として
  candle（`--no-default-features`）・Burn（`--no-default-features`。既定 feature が既に
  `ndarray`）の `train --mode fresh --device cpu` を手動で追加計測した（本体 `run_all_cuda.sh`
  は `bench-candle`/`bench-burn` を `--features cuda` でのみビルドするため、cuda ビルド失敗時は
  CPU 版も含め BINS に入らない。下記「計測不可・未計測項目」参照）
- 生データ: `results/raw/results-rtx3060-train.jsonl`（`run_all_cuda.sh` の 1 ラン分の train 行 +
  上記 candle CPU 手動計測 1 行）・`results/raw/skipped-rtx3060-train.log`・
  `results/run_all_cuda-rtx3060-train.log`
- ノイズ対策: train fresh/reuse（cuda・cpu）は `run_all_cuda.sh` の 1 回に加え個別コマンドで
  追加 4 回（計 5 回）実行した。JSONL には `run_all_cuda.sh` の 1 ラン分のみをコミットし
  （選別・捏造をしないため）、5 回分の傾向は下記「備考」に中央値のみ要約する

### (b) MLP 学習（cpu / cuda、fresh。summarize.py 生成）

| デバイス | フレームワーク | 中央値 | Q1 | Q3 |
| --- | --- | --- | --- | --- |
| cpu | fandhe-ai | 37.793 ms | 37.457 ms | 37.984 ms |
| cpu | candle | 1.551 ms | 1.443 ms | 1.665 ms |
| cpu | burn | 1.000 ms | 992.2 µs | 1.006 ms |
| cuda | fandhe-ai | 36.857 ms | 36.681 ms | 37.221 ms |
| cuda | candle | 計測不可 | - | - |
| cuda | burn | 計測不可 | - | - |

### (b') MLP 学習（デバイス常駐パラメータ更新モード。summarize.py 生成）

| デバイス | フレームワーク | 初期化(init_s) | 中央値 | Q1 | Q3 | fresh 中央値（参考） | fresh/reuse 比 | 最終 loss 突合（fresh） |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cpu | fandhe-ai | 535.0 µs | 37.978 ms | 37.714 ms | 38.208 ms | 37.793 ms | 1.00 倍 | 一致 |
| cuda | fandhe-ai | 132.333 ms | 36.898 ms | 36.755 ms | 37.294 ms | 36.857 ms | 1.00 倍 | 一致 |

### 環境 4 の計測不可・未計測項目

- Metal（Apple Silicon 実機）: 環境 5（Apple M4 Max / macOS。下記）で実測済み（本環境 4 の
  x86_64 Linux からは到達不能のため、本節の表には含めない）
- DGX Spark GB10（CUDA 実機）: `docs/real-hardware-verification-env.local.md` が本エージェント
  環境に存在せず SSH ホスト名が不明なため到達不能。再現コマンド: DGX Spark 実機で
  `./run_all_cuda.sh`
- bench-candle の CUDA ビルド: 本機に `nvcc` が未導入のため `--features cuda` ビルドが失敗
  （環境 3 と同一の既知制約）。CPU（`--no-default-features`）ビルドは成功し、`train fresh` の
  CPU 行のみ上表に含めた
- bench-burn の CUDA 実行: **ビルド自体は成功する**が（環境 3 の記述「candle/burn とも build
  失敗」は burn には当てはまらないことが本計測で判明した）、実行時に cubecl-cuda が
  「CUDA installation not found」（`CUDA_PATH` 未設定・本機に `/usr/local/cuda` 相当の完全な
  CUDA Toolkit が無い）で失敗する。`gemm` タスクは checksum 退化検出（`MEASURE_ERROR`）で
  自動的に拒否されるが、`train`/`infer` タスクには同種の検出が無いため `checksum=0.000000`
  という無効な結果が生成されうる（`skipped-rtx3060-train.log` 参照）。この無効行は
  `results-rtx3060-train.jsonl` に含めていない（性能値として提示すると誤解を招くため）。
  fandhe-ai は cudarc の動的ロード方式のためこの制約の影響を受けない

### 環境 4 の備考

- **fresh と reuse の中央値はほぼ同水準（cpu: 37.79 ms vs 37.98 ms、cuda: 36.86 ms vs 36.90 ms。
  比はいずれも 1.00 倍）で、gemm の reuse（環境 3・環境 2）で見られたような明確な高速化は
  train では観測されなかった**。これは `run_train_reuse`（`bench-fandhe/src/main.rs` モジュール
  doc「train --mode reuse」節）の設計上の理由による: reuse で排除するのはホスト経由 SGD の
  download/upload のみで、`register_resident_leaves` が毎 step 全パラメータを D2H download する
  経路は reuse でも残存する（#954 申し送り）。つまり本環境の実測は、この既知の設計制約
  （ホスト転送を伴わない完了待ち API が公開 API 面に無いギャップ）が実際の計測時間にも
  現れていることを裏付けるものであり、想定外の結果ではない
- 追加 4 回（計 5 回）の実行でも中央値の傾向は同様（cpu fresh: 37.60〜37.79 ms、
  cpu reuse: 37.98〜38.10 ms、cuda fresh: 36.72〜37.58 ms、cuda reuse: 36.90〜37.14 ms。
  いずれも fresh/reuse 差は数百 µs 以内でノイズの範囲内）
- 最終 loss（checksum）は fandhe-ai の cpu/cuda いずれも fresh/reuse で完全一致（0.080541。
  数値一致契約を満たす。上表「最終 loss 突合（fresh）」列参照）
- candle・Burn の CPU train 中央値（それぞれ 1.551 ms・1.000 ms）は fandhe-ai の CPU train
  （37.79 ms）よりおよそ 25〜38 倍高速である。この差は reuse では縮まらない（上記のとおり
  fresh/reuse 比が 1.00 倍のため）。candle/Burn は本 PR のスコープ外である reuse 相当の
  API（デバイス常駐更新）を標準で使っているのに対し、fandhe-ai の reuse も残存する毎 step
  D2H（上記備考）に律速されているためと考えられる。詳細な要因分解は別イシューで追跡することを
  推奨する（PR 本文の対象外項目を参照）

## 環境 5: Apple M4 Max / macOS（train fresh vs reuse。イシュー #957）

- チップ: Apple M4 Max（Metal GPU 統合メモリ）
- OS: macOS 26.6.2（25G83）
- ツールチェーン: cargo 1.96.0 (30a34c682 2026-05-25) / rustc 1.96.0 (ac68faa20 2026-05-25)（`--release` ビルド）
- fandhe-ai バージョン: 0.4.0（crates.io 公開版。#958/#998 の `train --mode reuse` 実装を含む）
- 計測日: 2026-08-29
- 対象: `bench-fandhe` の `train`（cpu / metal、fresh・reuse 両方）。イシュー #957 受け入れ条件 5
  （Metal 実機での train reuse 実測）を埋めるもので、candle / Burn は計測していない
  （reuse モードは `bench-fandhe` のみ対応。`run_all.sh` (b') ループと同じ扱い）
- 生データ: `results/raw/results-m4max-train.jsonl`（下記 4 組み合わせ各 1 ラン分）・
  `results/raw/skipped-m4max-train.log`（空。失敗なし）・`results/run_all-m4max-train.log`。
  追加 4 回分の一次データは `results/raw/results-m4max-train-extra.jsonl`（4 ラン × 4 行 = 16 行）・
  `results/run_all-m4max-train-extra.log`（4 ラン分の実行ログ。失敗なし）
- 実行方法: `./run_all.sh` は `results/raw/results.jsonl`（環境 1 の既存データ）を初期化するため
  実行せず、同スクリプトの `run()` と同一のコマンド形式
  （`./target/release/bench-fandhe --task train --device <cpu|metal> --size 64 --mode <fresh|reuse> --out <JSONL>`）
  で `train cpu 64 fresh` → `train metal 64 fresh` → `train cpu 64 reuse` → `train metal 64 reuse`
  の順に実行した。計測回数（warmup 20・iters 80）・シードはバイナリ既定のまま
- ノイズ対策: 環境 4 と同じ方式。上記 1 ランに加え同一手順で追加 4 回（計 5 回）実行した。
  summarize.py の表は 1 ラン目（`results-m4max-train.jsonl`）のみから生成し（選別・捏造を
  しないため）、追加 4 回分は `results-m4max-train-extra.jsonl` に別ファイルで保存する。
  5 回分の傾向は下記「備考」に中央値の範囲のみ要約する。計測中は他の重い処理を起動していない

### (b) MLP 学習（cpu / metal、fresh。summarize.py 生成）

| デバイス | フレームワーク | 中央値 | Q1 | Q3 |
| --- | --- | --- | --- | --- |
| cpu | fandhe-ai | 17.525 ms | 17.302 ms | 17.819 ms |
| cpu | candle | 計測不可 | - | - |
| cpu | burn | 計測不可 | - | - |
| metal | fandhe-ai | 19.699 ms | 19.087 ms | 20.284 ms |
| metal | candle | 計測不可 | - | - |
| metal | burn | 計測不可 | - | - |

### (b') MLP 学習（デバイス常駐パラメータ更新モード。summarize.py 生成）

| デバイス | フレームワーク | 初期化(init_s) | 中央値 | Q1 | Q3 | fresh 中央値（参考） | fresh/reuse 比 | 最終 loss 突合（fresh） |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cpu | fandhe-ai | 108.5 µs | 17.468 ms | 17.299 ms | 17.746 ms | 17.525 ms | 1.00 倍 | 一致 |
| metal | fandhe-ai | 30.440 ms | 20.381 ms | 19.996 ms | 20.762 ms | 19.699 ms | 0.97 倍 | 一致 |

### 環境 5 の計測不可・未計測項目

- candle / Burn（cpu・metal、train fresh）: 本環境の目的（#957 受け入れ条件 5 の Metal train reuse
  実測）の対象外のため計測していない。上表の「計測不可」は summarize.py が対象 JSONL に該当行が
  無い場合の既定表記であり、本環境で実行して失敗したものではない。環境 1（同一機 Apple M4 Max）
  の (b) 表に fresh の横並び値がある

### 環境 5 の備考

- **fresh と reuse の中央値は cpu で同水準（17.53 ms vs 17.47 ms、比 1.00 倍）、metal では reuse が
  わずかに遅い（19.70 ms vs 20.38 ms、比 0.97 倍）**。環境 4（RTX 3060 / cpu）と同様、train の
  reuse では明確な高速化は観測されなかった。設計上の理由は環境 4 の備考と同じ
  （`register_resident_leaves` が毎 step 全パラメータを D2H download する経路が reuse でも残存する。
  #954 申し送り）
- 追加 4 回（計 5 回）の中央値の範囲: cpu fresh 17.19〜17.53 ms、cpu reuse 17.19〜17.47 ms、
  metal fresh 19.05〜19.70 ms、metal reuse 19.64〜20.38 ms。metal の fresh/reuse の範囲は
  一部重なり（差は最大でも約 1.3 ms、Q1〜Q3 幅と同程度）、5 回すべてで reuse の中央値が同一ランの
  fresh より大きかった。この差が系統的かノイズかは本計測だけでは判別できない（要因分解は未実施）
- metal の `init_s`（30.4 ms。5 回の範囲 27.2〜32.5 ms）は cpu（108.5 µs）より大きい。
  初回 tape 構築 + 全パラメータの 1 回限りの H2D upload の経過時間（`bench-fandhe/src/main.rs`
  `run_train_reuse` の定義）であり、
  1 step あたり時間には含まれない
- 最終 loss（checksum）は cpu / metal いずれも fresh / reuse で完全一致（0.080541。環境 4 の
  cpu / cuda とも同値）。追加 4 回でも全 16 行が 0.080541 で一致した（数値一致契約を満たす）

## 環境 6: DGX Spark GB10（fandhe-ai 0.4.0 横並び再計測。イシュー #1050）

- ノード: 環境 2 と同一ノード（実ホスト名は `docs/real-hardware-verification-env.local.md`
  方式のローカル管理。環境 2 の SoC / OS / カーネル / GPU ドライバ / CUDA Toolkit / GPU 同居
  ワークロード / ノード選定の記載を参照。本節では未計測フィールドを推定で埋めない）
- ツールチェーン: 実行ログに rustc/cargo バージョンの記録なし。環境 2（rustc/cargo 1.97.0）を
  参照（同一ノード・直近の再ビルドのため大きな乖離は見込まない）
- fandhe-ai バージョン: 0.4.0（crates.io 公開版。2026-08-29 公開）
- 計測日: 2026-08-29
- 計測プロトコル: 環境 1・環境 2 と同一（f32、warmup 20 → 計測 20。学習は 100 ステップ中
  先頭 20 を warmup。終端でホスト実体化 + checksum。要素単位検証を含む）
- 対象: `scripts/bench/framework-compare` の GEMM（fresh/reuse）・MLP 学習・推論の 3 フレームワーク
  横並び（`./run_all_cuda.sh` 相当。fandhe-ai・candle・burn の cuda/cpu 全タスク）
- 生データ: `results/raw/results-dgx-0.4.0.jsonl`（46 行）・`results/raw/skipped-dgx-0.4.0.log`
  （空 = 実行時失敗なし）・`results/run_all_cuda-dgx-0.4.0.log`（実行ログ）

### (a) GEMM — CUDA

| N | フレームワーク | 中央値 | Q1 | Q3 | GFLOP/s |
| --- | --- | --- | --- | --- | --- |
| 256 | fandhe-ai | 114.2 µs | 114.0 µs | 114.7 µs | 293.8 |
| 256 | candle | 76.7 µs | 76.4 µs | 77.0 µs | 437.2 |
| 256 | burn（無効: 要素誤差超過 fail=10538/65536, max_abs=1.581e-03, max_rel=1.554e+00） | 180.7 µs | 179.9 µs | 181.1 µs | - |
| 512 | fandhe-ai | 398.6 µs | 398.0 µs | 399.4 µs | 673.4 |
| 512 | candle | 242.0 µs | 241.8 µs | 243.4 µs | 1109.2 |
| 512 | burn（無効: 要素誤差超過 fail=42361/262144, max_abs=2.343e-03, max_rel=1.972e+00） | 377.1 µs | 373.6 µs | 504.7 µs | - |
| 1024 | fandhe-ai | 1.889 ms | 1.887 ms | 1.891 ms | 1137.1 |
| 1024 | candle | 935.8 µs | 929.6 µs | 937.2 µs | 2294.8 |
| 1024 | burn（無効: 要素誤差超過 fail=169929/1048576, max_abs=3.643e-03, max_rel=1.965e+00） | 1.046 ms | 968.9 µs | 1.170 ms | - |
| 2048 | fandhe-ai | 183.997 ms | 178.863 ms | 186.809 ms | 93.4 |
| 2048 | candle（無効: 要素誤差超過 fail=2/4194304, max_abs=3.624e-05, max_rel=2.811e-01） | 4.228 ms | 4.217 ms | 4.247 ms | - |
| 2048 | burn（無効: 要素誤差超過 fail=681454/4194304, max_abs=4.941e-03, max_rel=1.987e+00） | 4.191 ms | 4.147 ms | 4.254 ms | - |
| 4096 | fandhe-ai | 131.228 ms | 130.815 ms | 131.935 ms | 1047.3 |
| 4096 | candle | 57.030 ms | 56.507 ms | 58.053 ms | 2409.9 |
| 4096 | burn（無効: 要素誤差超過 fail=2729050/16777216, max_abs=7.117e-03, max_rel=1.997e+00） | 39.082 ms | 39.018 ms | 39.124 ms | - |

### (a) GEMM — CPU（ARM、環境 2 と同一機）

| N | フレームワーク | 中央値 | Q1 | Q3 | GFLOP/s |
| --- | --- | --- | --- | --- | --- |
| 256 | fandhe-ai | 497.9 µs | 373.8 µs | 704.2 µs | 67.4 |
| 256 | candle | 344.9 µs | 335.2 µs | 367.0 µs | 97.3 |
| 256 | burn | 1.001 ms | 582.4 µs | 1.082 ms | 33.5 |
| 512 | fandhe-ai | 2.451 ms | 2.353 ms | 2.558 ms | 109.5 |
| 512 | candle | 1.872 ms | 1.668 ms | 1.965 ms | 143.4 |
| 512 | burn | 3.422 ms | 3.408 ms | 3.427 ms | 78.4 |
| 1024 | fandhe-ai | 7.526 ms | 7.268 ms | 7.723 ms | 285.3 |
| 1024 | candle | 5.439 ms | 5.313 ms | 5.550 ms | 394.8 |
| 1024 | burn | 21.953 ms | 21.938 ms | 21.975 ms | 97.8 |
| 2048 | fandhe-ai | 36.776 ms | 35.980 ms | 37.094 ms | 467.1 |
| 2048 | candle（無効: 要素誤差超過 fail=2/4194304, max_abs=3.815e-05, max_rel=3.944e-01） | 33.510 ms | 33.008 ms | 33.924 ms | - |
| 2048 | burn（無効: 要素誤差超過 fail=5/4194304, max_abs=3.529e-05, max_rel=3.052e-01） | 158.523 ms | 158.464 ms | 158.593 ms | - |

### (a') GEMM — CUDA（デバイス/tape 再利用モード。イシュー #925）

| N | フレームワーク | 初期化(init_s) | 中央値 | Q1 | Q3 | GFLOP/s | fresh 中央値（参考） |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 256 | fandhe-ai | 399.991 ms | 114.5 µs | 113.2 µs | 189.5 µs | 293.0 | 114.2 µs |
| 512 | fandhe-ai | 416.353 ms | 705.6 µs | 692.1 µs | 709.7 µs | 380.4 | 398.6 µs |
| 1024 | fandhe-ai | 405.698 ms | 3.295 ms | 3.127 ms | 3.443 ms | 651.7 | 1.889 ms |
| 2048 | fandhe-ai | 419.470 ms | 15.991 ms | 15.817 ms | 16.463 ms | 1074.4 | 183.997 ms |
| 4096 | fandhe-ai | 522.955 ms | 124.877 ms | 123.594 ms | 126.215 ms | 1100.6 | 131.228 ms |

### (b) MLP 学習（784→256→10、ReLU、バッチ 64、MSE、SGD lr=0.01、1 ステップあたり時間）

| デバイス | フレームワーク | 中央値 | Q1 | Q3 |
| --- | --- | --- | --- | --- |
| cpu | fandhe-ai | 13.874 ms | 13.694 ms | 14.637 ms |
| cpu | candle | 2.105 ms | 1.807 ms | 2.502 ms |
| cpu | burn | 996.3 µs | 990.8 µs | 999.3 µs |
| cuda | fandhe-ai | 12.440 ms | 12.431 ms | 12.454 ms |
| cuda | candle | 275.5 µs | 273.5 µs | 279.9 µs |
| cuda | burn | 670.6 µs | 506.9 µs | 835.1 µs |

### (b') MLP 学習（デバイス常駐パラメータ更新モード）

| デバイス | フレームワーク | 初期化(init_s) | 中央値 | Q1 | Q3 | fresh 中央値（参考） | fresh/reuse 比 | 最終 loss 突合（fresh） |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cpu | fandhe-ai | 625.5 µs | 13.877 ms | 13.643 ms | 14.172 ms | 13.874 ms | 1.00 倍 | 一致 |
| cuda | fandhe-ai | 214.625 ms | 12.420 ms | 12.406 ms | 12.440 ms | 12.440 ms | 1.00 倍 | 一致 |

### (c) 推論スループット（同 MLP forward のみ、バッチ 64。1 バッチ = 64 件）

| デバイス | フレームワーク | 中央値 | Q1 | Q3 | バッチ/秒 |
| --- | --- | --- | --- | --- | --- |
| cpu | fandhe-ai | 321.2 µs | 310.4 µs | 363.6 µs | 3114 |
| cpu | candle | 214.1 µs | 203.0 µs | 302.8 µs | 4671 |
| cpu | burn | 239.8 µs | 239.4 µs | 240.4 µs | 4170 |
| cuda | fandhe-ai | 225.7 µs | 225.3 µs | 227.1 µs | 4431 |
| cuda | candle | 41.5 µs | 40.2 µs | 42.3 µs | 24117 |
| cuda | burn | 365.5 µs | 227.0 µs | 425.8 µs | 2736 |

### 環境 6 のデータ有効性（checksum 突合・要素単位検証）

- 不一致なし（相互突合できた 32 行の checksum が参照値と一致）
- **無効（要素誤差超過。0 近傍の丸め差）**: candle/cuda/size=2048/fresh（fail=2/4194304,
  max_abs=3.624e-05, max_rel=2.811e-01）・candle/cpu/size=2048/fresh（fail=2/4194304,
  max_abs=3.815e-05, max_rel=3.944e-01）・burn/cpu/size=2048/fresh（fail=5/4194304,
  max_abs=3.529e-05, max_rel=3.052e-01）。いずれも 0 近傍要素の丸め差であり、閾値
  （本体の数値一致契約と同値。`.claude/rules/coding-rust.md`）は緩めず「無効」表示のまま
  参考値として記録する
- **無効（要素誤差超過。burn CUDA 経路の大幅超過）**: burn/cuda/size=256〜4096 の全 5 サイズ
  （fail は最大 2,729,050/16,777,216 要素、max_rel 最大 1.997e+00）。0 近傍の丸め差ではなく
  広範囲の要素で乖離しており、burn 0.21.0（cubecl-cuda 経路）が既定で TF32 相当の低精度
  アキュムレーションへ降格する既知事項（環境 2 の備考「checksum はフレームワーク間で概ね
  一致…burn の CUDA GEMM は checksum のずれがやや大きい」・`docs/burn-cuda-tf32.md` 系メモ）に
  起因する可能性が高いが、本 PR では実装破損か精度契約外かの切り分けは行わない。閾値は緩めず
  「無効」表示のまま未検証・要追試として記録する
- `summarize.py --strict results/raw/results-dgx-0.4.0.jsonl results/raw/results-m4max-0.4.0.jsonl`
  は終了コード 2（要素単位検証の閾値超過。環境 6/7 合算で警告 9 件。内訳は本節・環境 7 節参照）

### 環境 6 の計測不可・未計測項目

- 実行時に失敗した組み合わせ: なし（`skipped-dgx-0.4.0.log` は空）
- tch-rs: 環境 1〜5 と同じ理由で未計測

### 環境 6 の備考

- **fresh モード CUDA GEMM N=2048 が 184.0 ms と、同モード他サイズ（N=256〜1024 は 0.1〜1.9 ms、
  N=4096 は 131.2 ms）から突出して大きい。** 既知の fresh N=2048 固有オーバーヘッド
  （`docs/perf/cuda-fresh-gemm-n2048-overhead-diagnosis.md`・イシュー #956・#1025）と整合する
  再現であり、本 PR では原因調査を行わない（同 doc 参照）
- **reuse（(a') 表）N=4096 は 1100.6 GFLOP/s で、同一データの candle fresh 2409.9 GFLOP/s の
  約 46%。** fandhe-ai fresh N=4096 の 1047.3 GFLOP/s から reuse でわずかに改善するが、candle
  比の水準はほぼ変わらない。candle 比の目標達成判定そのものは本 PR のスコープ外であり、
  イシュー #1051（目標達成ゲート）・#1052（再計測）が担う
- 学習 1 step: cuda 12.440 ms（candle 275.5 µs・burn 670.6 µs）、cpu 13.874 ms（candle
  2.105 ms・burn 996.3 µs）。推論: cuda 225.7 µs（candle 41.5 µs・burn 365.5 µs）。いずれも
  candle・burn を大きく下回る現在地であり、環境 2（0.3.0 時点）からの傾向は継続している

## 環境 7: Apple M4 Max（fandhe-ai 0.4.0 横並び再計測。イシュー #1050）

- チップ・OS・ツールチェーン: 環境 1・環境 5 と同一機（Apple M4 Max、macOS 26.6.2）。実行ログに
  cargo/rustc バージョンの記録なし。環境 5（cargo/rustc 1.96.0）を参照
- fandhe-ai バージョン: 0.4.0（crates.io 公開版。2026-08-29 公開）
- 計測日: 2026-08-29
- 計測プロトコル: 環境 1・環境 2・環境 6 と同一
- 対象: `scripts/bench/framework-compare` の GEMM（fresh/reuse）・MLP 学習・推論の 3 フレームワーク
  横並び（`./run_all.sh` 相当。fandhe-ai・candle・burn の metal/cpu 全タスク）
- 生データ: `results/raw/results-m4max-0.4.0.jsonl`（42 行）・
  `results/raw/skipped-m4max-0.4.0.log`（4 行。下記参照）・`results/run_all_m4max-0.4.0.log`（実行ログ）

### (a) GEMM — CPU

| N | フレームワーク | 中央値 | Q1 | Q3 | GFLOP/s |
| --- | --- | --- | --- | --- | --- |
| 256 | fandhe-ai | 241.8 µs | 227.6 µs | 277.1 µs | 138.8 |
| 256 | candle | 348.3 µs | 308.5 µs | 375.3 µs | 96.3 |
| 256 | burn | 474.8 µs | 470.9 µs | 511.1 µs | 70.7 |
| 512 | fandhe-ai | 740.5 µs | 715.7 µs | 791.1 µs | 362.5 |
| 512 | candle | 731.5 µs | 684.9 µs | 785.8 µs | 367.0 |
| 512 | burn | 2.707 ms | 2.695 ms | 2.731 ms | 99.2 |
| 1024 | fandhe-ai | 3.410 ms | 3.341 ms | 3.578 ms | 629.7 |
| 1024 | candle | 3.107 ms | 2.827 ms | 3.384 ms | 691.2 |
| 1024 | burn | 20.628 ms | 20.516 ms | 20.768 ms | 104.1 |
| 2048 | fandhe-ai | 21.268 ms | 20.688 ms | 23.674 ms | 807.8 |
| 2048 | candle | 19.233 ms | 18.717 ms | 19.398 ms | 893.2 |
| 2048 | burn（無効: 要素誤差超過 fail=5/4194304, max_abs=3.529e-05, max_rel=3.052e-01） | 159.064 ms | 158.542 ms | 159.593 ms | - |

### (a) GEMM — Metal

| N | フレームワーク | 中央値 | Q1 | Q3 | GFLOP/s |
| --- | --- | --- | --- | --- | --- |
| 256 | fandhe-ai | 324.3 µs | 310.8 µs | 594.6 µs | 103.5 |
| 256 | candle | 253.2 µs | 238.7 µs | 268.5 µs | 132.5 |
| 256 | burn | 1.403 ms | 1.392 ms | 1.410 ms | 23.9 |
| 512 | fandhe-ai | 699.5 µs | 671.3 µs | 729.0 µs | 383.7 |
| 512 | candle | 378.6 µs | 369.8 µs | 501.9 µs | 709.0 |
| 512 | burn | 計測不可 | - | - | - |
| 1024 | fandhe-ai | 2.498 ms | 2.053 ms | 2.920 ms | 859.8 |
| 1024 | candle | 1.599 ms | 1.298 ms | 1.703 ms | 1342.6 |
| 1024 | burn | 計測不可 | - | - | - |
| 2048 | fandhe-ai | 9.736 ms | 7.999 ms | 10.772 ms | 1764.6 |
| 2048 | candle | 7.060 ms | 6.404 ms | 9.786 ms | 2433.6 |
| 2048 | burn | 計測不可 | - | - | - |
| 4096 | fandhe-ai | 40.481 ms | 39.944 ms | 41.030 ms | 3395.1 |
| 4096 | candle | 27.270 ms | 22.596 ms | 34.245 ms | 5039.9 |
| 4096 | burn | 計測不可 | - | - | - |

### (a') GEMM — Metal（デバイス/tape 再利用モード。イシュー #925）

| N | フレームワーク | 初期化(init_s) | 中央値 | Q1 | Q3 | GFLOP/s | fresh 中央値（参考） |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 256 | fandhe-ai | 33.535 ms | 325.6 µs | 319.1 µs | 329.9 µs | 103.1 | 324.3 µs |
| 512 | fandhe-ai | 33.947 ms | 722.7 µs | 711.8 µs | 734.4 µs | 371.4 | 699.5 µs |
| 1024 | fandhe-ai | 37.034 ms | 2.374 ms | 2.291 ms | 3.121 ms | 904.6 | 2.498 ms |
| 2048 | fandhe-ai | 48.193 ms | 10.164 ms | 9.786 ms | 10.773 ms | 1690.3 | 9.736 ms |
| 4096 | fandhe-ai | 97.232 ms | 45.339 ms | 45.067 ms | 45.938 ms | 3031.3 | 40.481 ms |

### (b) MLP 学習（784→256→10、ReLU、バッチ 64、MSE、SGD lr=0.01、1 ステップあたり時間）

| デバイス | フレームワーク | 中央値 | Q1 | Q3 |
| --- | --- | --- | --- | --- |
| cpu | fandhe-ai | 17.080 ms | 16.897 ms | 17.317 ms |
| cpu | candle | 814.0 µs | 739.1 µs | 929.3 µs |
| cpu | burn | 634.3 µs | 627.3 µs | 644.6 µs |
| metal | fandhe-ai | 18.631 ms | 18.411 ms | 19.049 ms |
| metal | candle | 608.0 µs | 592.7 µs | 629.0 µs |
| metal | burn | 1.598 ms | 1.590 ms | 1.605 ms |

### (b') MLP 学習（デバイス常駐パラメータ更新モード）

| デバイス | フレームワーク | 初期化(init_s) | 中央値 | Q1 | Q3 | fresh 中央値（参考） | fresh/reuse 比 | 最終 loss 突合（fresh） |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cpu | fandhe-ai | 111.6 µs | 18.054 ms | 17.931 ms | 18.175 ms | 17.080 ms | 0.95 倍 | 一致 |
| metal | fandhe-ai | 29.177 ms | 20.789 ms | 20.422 ms | 21.164 ms | 18.631 ms | 0.90 倍 | 一致 |

### (c) 推論スループット（同 MLP forward のみ、バッチ 64。1 バッチ = 64 件）

| デバイス | フレームワーク | 中央値 | Q1 | Q3 | バッチ/秒 |
| --- | --- | --- | --- | --- | --- |
| cpu | fandhe-ai | 536.7 µs | 491.7 µs | 549.7 µs | 1863 |
| cpu | candle | 166.5 µs | 155.9 µs | 238.5 µs | 6006 |
| cpu | burn | 282.1 µs | 281.8 µs | 284.8 µs | 3545 |
| metal | fandhe-ai | 955.6 µs | 919.9 µs | 1.135 ms | 1046 |
| metal | candle | 383.7 µs | 375.7 µs | 392.5 µs | 2606 |
| metal | burn | 1.509 ms | 1.500 ms | 1.516 ms | 663 |

### 環境 7 のデータ有効性（checksum 突合・要素単位検証）

- 不一致なし（相互突合できた 28 行の checksum が参照値と一致）
- **無効（要素誤差超過。0 近傍の丸め差）**: burn/cpu/size=2048/fresh（fail=5/4194304,
  max_abs=3.529e-05, max_rel=3.052e-01）。環境 6 の同一行と同種の丸め差であり、閾値は緩めず
  「無効」表示のまま参考値として記録する
- `summarize.py --strict` の合算終了コード・警告件数は環境 6 節を参照（環境 6/7 は同一
  `--strict` 呼び出しの対象）

### 環境 7 の計測不可・未計測項目

- **Burn(wgpu) Metal GEMM の N=512/1024/2048/4096**: `skipped-m4max-0.4.0.log` に 4 件の
  `MEASURE_ERROR: gemm checksum is degenerate (0)` が記録されており、`bench-common::validate_gemm_checksum`
  が結果テンソル全ゼロを emit 前に遮断したため行自体が JSONL に存在しない（無効データとして
  記録されるのではなく、そもそも計測不可）。原因は burn 0.21.0（cubek-matmul 0.2.0）の
  wgpu/Metal 経路の upstream 既知バグ（tracel-ai/burn#4966 → tracel-ai/cubek#283）で、
  「データ有効性の注記（イシュー #965）」節・`docs/perf/burn-wgpu-metal-gemm-zero-result.md` と
  同一事象の再現（承認済みピン `burn =0.21.0` の範囲では修正版を取得できない）
- tch-rs: 環境 1〜6 と同じ理由で未計測

### 環境 7 の備考

- **reuse（(a') 表）N=4096 は 3031.3 GFLOP/s で、同一データの candle fresh 5039.9 GFLOP/s の
  約 60%。** fandhe-ai fresh N=4096 の 3395.1 GFLOP/s から reuse で悪化する（初期化コストを
  分離しても改善しない点は環境 1 の Metal 表注記と同傾向）。candle 比の目標達成判定は本 PR の
  スコープ外（イシュー #1051・#1052 が担う）
- 学習 1 step: metal 18.631 ms（candle 608.0 µs・burn 1.598 ms）、cpu 17.080 ms（candle
  814.0 µs・burn 634.3 µs）。推論: metal 955.6 µs（candle 383.7 µs・burn 1.509 ms）。環境 1
  （0.3.0 時点）と同様、candle・burn を大きく下回る現在地が継続している
