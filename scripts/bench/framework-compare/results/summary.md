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
  参照（同一ノード・直近の再ビルドのため大きな乖離は見込まない）。cargo tree（依存関係の
  実測、推定ではない）は `results/versions.txt` の環境 6/7 節を参照。実行ログに cargo tree
  出力は残っていないため、計測後に確認した「以降変更のない Cargo.lock」から再取得した
  実際の出力を記録している（未記録部分〈rustc/cargo バージョン〉を推定で埋めていない点は
  上記と同様）
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
  cargo/rustc バージョンの記録なし。環境 5（cargo/rustc 1.96.0）を参照。cargo tree は
  `results/versions.txt` の環境 6/7 節（環境 6 節の注記と同じ取得方法）を参照
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

## 環境 8: DGX Spark GB10（fandhe-ai 0.5.0 横並び再計測・目標達成判定。イシュー #1052）

- ノード: 環境 2・環境 6 と同一ノード（実ホスト名は `docs/real-hardware-verification-env.local.md`
  方式のローカル管理。未計測フィールドを推定で埋めない）
- ツールチェーン: rustc 1.97.0 (2d8144b78 2026-07-07) / cargo 1.97.0 (c980f4866 2026-06-30)（実測。
  環境 6 の「記録なし」から改善）。CUDA 13.0 V13.0.88（nvcc）/ NVIDIA driver 580.173.02。
  cargo tree 実測は `results/versions.txt` の環境 8/9 節を参照
- fandhe-ai バージョン: 0.5.0（crates.io 公開版。2026-08-31 公開）
- **0.5.0 の収録範囲に関する重要な限定**: `v0.5.0` タグ以降に main へ入った改善（#1108 Metal
  選択テーブル・#1110 Metal SGD バッチング・#1111 CUDA variant selection 修正）は本計測に
  **含まれない**。また CUDA GEMM 改善トラッカー #1031・Metal GEMM 改善トラッカー #1037 は
  本計測時点で open のまま（未着手）であり、下記ゲート判定はそれらの改善前の現在地である
- 計測日: 2026-09-01
- 計測プロトコル: 環境 1・環境 2・環境 6 と同一（f32、warmup 20 → 計測 20。学習は 100 ステップ中
  先頭 20 を warmup。終端でホスト実体化 + checksum。要素単位検証を含む）
- 対象: `scripts/bench/framework-compare` の GEMM（fresh/reuse）・MLP 学習（フェーズ分解含む）・
  推論の 3 フレームワーク横並び（`./run_all_cuda.sh` 相当。fandhe-ai・candle・burn の cuda/cpu
  全タスク）
- 生データ: `results/raw/results-dgx-0.5.0.jsonl`（82 行）・`results/raw/skipped-dgx-0.5.0.log`
  （空 = 実行時失敗なし）・`results/run_all_cuda-dgx-0.5.0.log`（実行ログ）

### (a) GEMM — CUDA

| N | フレームワーク | 中央値 | Q1 | Q3 | GFLOP/s |
| --- | --- | --- | --- | --- | --- |
| 256 | fandhe-ai | 109.2 µs | 108.9 µs | 109.6 µs | 307.4 |
| 256 | candle | 76.6 µs | 76.2 µs | 76.9 µs | 438.2 |
| 256 | burn | 計測不可 | - | - | - |
| 512 | fandhe-ai | 328.7 µs | 328.5 µs | 329.7 µs | 816.6 |
| 512 | candle | 235.9 µs | 235.2 µs | 236.2 µs | 1138.1 |
| 512 | burn | 計測不可 | - | - | - |
| 1024 | fandhe-ai | 1.269 ms | 1.268 ms | 1.271 ms | 1692.8 |
| 1024 | candle | 924.4 µs | 923.1 µs | 925.6 µs | 2323.1 |
| 1024 | burn | 計測不可 | - | - | - |
| 2048 | fandhe-ai | 140.214 ms | 137.226 ms | 144.054 ms | 122.5 |
| 2048 | candle（無効: 要素誤差超過 fail=2/4194304, max_abs=3.624e-05, max_rel=2.811e-01） | 4.203 ms | 4.194 ms | 4.207 ms | - |
| 2048 | burn | 計測不可 | - | - | - |
| 4096 | fandhe-ai | 69.215 ms | 68.882 ms | 69.854 ms | 1985.7 |
| 4096 | candle | 55.509 ms | 53.932 ms | 56.125 ms | 2476.0 |
| 4096 | burn | 計測不可 | - | - | - |

burn の CUDA GEMM は本ラン（fresh 経路）では計測不可（`--tf32` opt-in 経路でのみ計測。下表
「(a-tf32)」参照）。環境 6 時点の「burn/cuda が広範囲で要素誤差超過」という事象は tf32 経路に
限定して継続している。

### (a-tf32) GEMM TF32（burn 0.21.0 の CUDA 既定精度。REQ-2 統一複合判定の対象外・参考値）

| N | フレームワーク | 中央値 | Q1 | Q3 | 備考 |
| --- | --- | --- | --- | --- | --- |
| 256 | burn（無効: 要素誤差超過 fail=10538/65536, max_abs=1.581e-03, max_rel=1.554e+00） | 701.7 µs | 279.1 µs | 839.7 µs | TF32 降格 |
| 512 | burn（無効: 要素誤差超過 fail=42361/262144, max_abs=2.343e-03, max_rel=1.972e+00） | 638.6 µs | 379.6 µs | 665.8 µs | TF32 降格 |
| 1024 | burn（無効: 要素誤差超過 fail=169929/1048576, max_abs=3.643e-03, max_rel=1.965e+00） | 962.4 µs | 958.4 µs | 1.015 ms | TF32 降格 |
| 2048 | burn（無効: 要素誤差超過 fail=681454/4194304, max_abs=4.941e-03, max_rel=1.987e+00） | 4.137 ms | 4.090 ms | 4.185 ms | TF32 降格 |
| 4096 | burn（無効: 要素誤差超過 fail=2729050/16777216, max_abs=7.117e-03, max_rel=1.997e+00） | 37.566 ms | 37.500 ms | 37.624 ms | TF32 降格 |

`docs/burn-cuda-tf32.md`（burn 0.21 CUDA の TF32 既定降格）の既知事項どおりで、閾値は緩めず
「無効」表示のまま参考値として記録する（本 PR で精度契約外の切り分けは行わない）。

### (a) GEMM — CPU（ARM、環境 2・環境 6 と同一機）

| N | フレームワーク | 中央値 | Q1 | Q3 | GFLOP/s |
| --- | --- | --- | --- | --- | --- |
| 256 | fandhe-ai | 544.6 µs | 414.9 µs | 671.7 µs | 61.6 |
| 256 | candle | 340.7 µs | 265.1 µs | 361.0 µs | 98.5 |
| 256 | burn | 513.8 µs | 499.4 µs | 516.1 µs | 65.3 |
| 512 | fandhe-ai | 2.632 ms | 2.327 ms | 2.714 ms | 102.0 |
| 512 | candle | 1.790 ms | 1.584 ms | 2.148 ms | 150.0 |
| 512 | burn | 3.303 ms | 3.228 ms | 3.363 ms | 81.3 |
| 1024 | fandhe-ai | 7.574 ms | 7.356 ms | 7.906 ms | 283.5 |
| 1024 | candle | 5.360 ms | 5.299 ms | 5.574 ms | 400.6 |
| 1024 | burn | 21.741 ms | 21.638 ms | 21.758 ms | 98.8 |
| 2048 | fandhe-ai | 36.029 ms | 35.559 ms | 37.082 ms | 476.8 |
| 2048 | candle（無効: 要素誤差超過 fail=2/4194304, max_abs=3.815e-05, max_rel=3.944e-01） | 33.490 ms | 32.758 ms | 34.592 ms | - |
| 2048 | burn（無効: 要素誤差超過 fail=5/4194304, max_abs=3.529e-05, max_rel=3.052e-01） | 155.015 ms | 153.732 ms | 155.659 ms | - |

### (a') GEMM — CUDA（デバイス/tape 再利用モード。イシュー #925）

| N | フレームワーク | 初期化(init_s) | 中央値 | Q1 | Q3 | GFLOP/s | fresh 中央値（参考） |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 256 | fandhe-ai | 389.345 ms | 109.0 µs | 108.2 µs | 174.0 µs | 307.9 | 109.2 µs |
| 512 | fandhe-ai | 415.681 ms | 614.1 µs | 611.0 µs | 623.5 µs | 437.1 | 328.7 µs |
| 1024 | fandhe-ai | 403.691 ms | 2.460 ms | 2.367 ms | 2.476 ms | 872.8 | 1.269 ms |
| 2048 | fandhe-ai | 423.617 ms | 10.522 ms | 10.277 ms | 10.583 ms | 1632.8 | 140.214 ms |
| 4096 | fandhe-ai | 464.911 ms | 67.918 ms | 64.896 ms | 68.457 ms | 2023.6 | 69.215 ms |

### (b) MLP 学習（784→256→10、ReLU、バッチ 64、MSE、SGD lr=0.01、1 ステップあたり時間）

| デバイス | フレームワーク | 中央値 | Q1 | Q3 |
| --- | --- | --- | --- | --- |
| cpu | fandhe-ai | 13.496 ms | 13.061 ms | 14.070 ms |
| cpu | candle | 2.568 ms | 2.227 ms | 2.829 ms |
| cpu | burn | 737.3 µs | 734.7 µs | 747.7 µs |
| cuda | fandhe-ai | 11.490 ms | 11.482 ms | 11.503 ms |
| cuda | candle | 268.7 µs | 266.6 µs | 271.9 µs |
| cuda | burn（TF32） | 774.7 µs | 522.0 µs | 927.1 µs |

### (b') MLP 学習（デバイス常駐パラメータ更新モード）

| デバイス | フレームワーク | 初期化(init_s) | 中央値 | Q1 | Q3 | fresh 中央値（参考） | fresh/reuse 比 | 最終 loss 突合（fresh） |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cpu | fandhe-ai | 900.0 µs | 7.887 ms | 7.275 ms | 8.730 ms | 13.496 ms | 1.71 倍 | 一致 |
| cuda | fandhe-ai | 211.199 ms | 5.374 ms | 5.364 ms | 5.381 ms | 11.490 ms | 2.14 倍 | 一致 |

環境 6（b'）比: cpu 900.0 µs／7.887 ms（環境 6: 625.5 µs／12.440 ms〈fresh 相当は当時 (b) 表の
値〉）・cuda 211.199 ms／5.374 ms（環境 6: 214.625 ms／12.420 ms）。cuda 側の reuse 中央値が
12.440 ms → 5.374 ms（約 2.3 倍）へ改善しており、#1078〜#1081（tape ノードクリア API・view
再計算方式化・Linear epilogue 融合・MSE reduction 融合。いずれも main 収録済みで 0.5.0 に
含まれる）の効果と整合する

### (b'') MLP 学習 1 step のフェーズ分解（イシュー #1009。詳細は `docs/perf/train-linear-epilogue-fusion.md`
等の個別ドキュメントを参照。本節では要点のみ）

- CPU reuse: backward 5.887 ms（76.7%）が支配項。fresh 比 forward 1.581 ms → reuse
  forward_resident 1.625 ms（ほぼ変化なし）、backward は 11.508 ms → 5.887 ms へ大幅改善
- CUDA reuse: backward 5.173 ms（96.3%）が支配項。fresh 比 backward 11.232 ms → 5.173 ms
  （約 2.2 倍改善）。device_update は 40.2 µs（0.7%）まで縮小

### (c) 推論スループット（同 MLP forward のみ、バッチ 64）

| デバイス | フレームワーク | 中央値 | Q1 | Q3 | バッチ/秒 |
| --- | --- | --- | --- | --- | --- |
| cpu | fandhe-ai | 403.6 µs | 342.6 µs | 680.4 µs | 2477 |
| cpu | candle | 229.8 µs | 216.4 µs | 261.0 µs | 4351 |
| cpu | burn | 230.5 µs | 230.0 µs | 231.4 µs | 4338 |
| cuda | fandhe-ai | 152.8 µs | 151.6 µs | 154.0 µs | 6542 |
| cuda | candle | 39.9 µs | 39.4 µs | 40.9 µs | 25065 |
| cuda | burn（TF32） | 432.6 µs | 210.2 µs | 483.6 µs | 2311 |

環境 6 比: cuda 推論 225.7 µs → 152.8 µs（約 1.48 倍改善。#1028 の推論固定費削減と整合）。cpu
推論は 403.6 µs（環境 6: 403.6 µs 相当データなし・環境 1 系との直接比較は行わない）

### 環境 8 のデータ有効性（checksum 突合・要素単位検証）

- 不一致なし（相互突合できた 27 行の checksum が参照値と一致）
- **無効（要素誤差超過。0 近傍の丸め差）**: candle/cuda/size=2048/fresh・candle/cpu/size=2048/fresh・
  burn/cpu/size=2048/fresh（詳細は上表参照）。環境 6 と同種の丸め差であり、閾値は緩めず「無効」
  表示のまま参考値として記録する
- **無効（要素誤差超過。burn CUDA TF32 経路）**: burn/cuda/size=256〜4096 の全 5 サイズ（tf32
  opt-in 経路。上表「(a-tf32)」参照）。環境 6 と同一事象の継続
- `summarize.py --strict` は終了コード 2（環境 8/9 合算で無効レコード 9 件〈ユニーク〉。burn
  CUDA TF32 経路の 5 行が通常 GEMM 検証と TF32 専用検証の双方で警告されるため、実際の警告出力
  行数は 14 件）

### 環境 8 の計測不可・未計測項目

- 実行時に失敗した組み合わせ: なし（`skipped-dgx-0.5.0.log` は空）
- tch-rs: 環境 1〜7 と同じ理由で未計測

### 環境 8 の目標達成ゲート（`summarize.py --target candle`。イシュー #1051）

| タスク | デバイス | N | fandhe-ai 中央値 | candle 中央値 | 比（target/fandhe） | 判定 |
| --- | --- | --- | --- | --- | --- | --- |
| gemm | CPU | 256 | 544.6 µs | 340.7 µs | 0.63 倍 | 未達 |
| gemm | CPU | 512 | 2.632 ms | 1.790 ms | 0.68 倍 | 未達 |
| gemm | CPU | 1024 | 7.574 ms | 5.360 ms | 0.71 倍 | 未達 |
| gemm | CPU | 2048 | - | - | - | 判定不能（candle 無効データ） |
| gemm | CUDA | 256 | 109.0 µs | 76.6 µs | 0.70 倍 | 未達 |
| gemm | CUDA | 512 | 614.1 µs | 235.9 µs | 0.38 倍 | 未達 |
| gemm | CUDA | 1024 | 2.460 ms | 924.4 µs | 0.38 倍 | 未達 |
| gemm | CUDA | 2048 | - | - | - | 判定不能（candle 無効データ） |
| gemm | CUDA | 4096 | 67.918 ms | 55.509 ms | 0.82 倍 | 未達 |
| train | CPU | 64 | 7.887 ms | 2.568 ms | 0.33 倍 | 未達 |
| train | CUDA | 64 | 5.374 ms | 268.7 µs | 0.05 倍 | 未達 |
| infer | CPU | 64 | 403.6 µs | 229.8 µs | 0.57 倍 | 未達 |
| infer | CUDA | 64 | 152.8 µs | 39.9 µs | 0.26 倍 | 未達 |

未達 11 件・判定不能 2 件（達成 0 件）。CUDA GEMM の未達は既存トラッカー #1031（open）、CUDA
train/infer の未達は #1031 の対象範囲外で新規トラッカー化が必要（下記「未達項目の追跡」参照）

### 環境 8 の備考

- **fresh モード CUDA GEMM N=2048 が 140.2 ms と、環境 6 の 184.0 ms から改善したが依然
  他サイズ（N=256〜1024 は 0.1〜1.3 ms、N=4096 は 69.2 ms）から突出して大きい**。既知の
  fresh N=2048 固有オーバーヘッド（`docs/perf/cuda-fresh-gemm-n2048-overhead-diagnosis.md`）と
  整合する再現であり、本 PR では原因調査を行わない
- reuse（(a') 表）N=4096 は 2023.6 GFLOP/s（環境 6: 1100.6 GFLOP/s から約 1.84 倍改善）。同一
  データの candle fresh 2476.0 GFLOP/s 比では約 82%まで縮まったが依然未達
- 学習・推論は環境 6 比で大きく改善した（cuda train 12.440 ms → 5.374 ms・infer 225.7 µs →
  152.8 µs）ものの、candle・burn 比では依然全項目未達

## 環境 9: Apple M4 Max（fandhe-ai 0.5.0 横並び再計測・目標達成判定。イシュー #1052）

- チップ・OS: 環境 1・環境 5・環境 7 と同一機（Apple M4 Max、macOS 26.6.2 / Darwin 25.6.0）
- ツールチェーン: rustc 1.96.0 (ac68faa20 2026-05-25) / cargo 1.96.0 (30a34c682 2026-05-25)（実測）
- fandhe-ai バージョン: 0.5.0（crates.io 公開版。2026-08-31 公開）
- **0.5.0 の収録範囲に関する重要な限定**: 環境 8 の同項目と同一（#1108/#1110/#1111 は 0.5.0 に
  含まれない。Metal GEMM 改善トラッカー #1037 は本計測時点で open のまま）
- 計測日: 2026-09-01
- 計測プロトコル: 環境 1・環境 2・環境 6・環境 7・環境 8 と同一
- 対象: `scripts/bench/framework-compare` の GEMM（fresh/reuse）・MLP 学習（フェーズ分解含む）・
  推論の 3 フレームワーク横並び（`./run_all.sh` 相当。fandhe-ai・candle・burn の metal/cpu
  全タスク）
- 生データ: `results/raw/results-m4max-0.5.0.jsonl`（78 行）・
  `results/raw/skipped-m4max-0.5.0.log`（4 行。下記参照）・`results/run_all_m4max-0.5.0.log`
  （実行ログ）

### (a) GEMM — CPU

| N | フレームワーク | 中央値 | Q1 | Q3 | GFLOP/s |
| --- | --- | --- | --- | --- | --- |
| 256 | fandhe-ai | 317.2 µs | 301.5 µs | 337.5 µs | 105.8 |
| 256 | candle | 335.8 µs | 309.6 µs | 383.1 µs | 99.9 |
| 256 | burn | 524.7 µs | 506.7 µs | 555.2 µs | 64.0 |
| 512 | fandhe-ai | 710.2 µs | 691.4 µs | 748.3 µs | 378.0 |
| 512 | candle | 704.9 µs | 671.9 µs | 729.1 µs | 380.8 |
| 512 | burn | 2.622 ms | 2.614 ms | 2.638 ms | 102.4 |
| 1024 | fandhe-ai | 3.489 ms | 3.342 ms | 3.645 ms | 615.4 |
| 1024 | candle | 2.669 ms | 2.635 ms | 2.704 ms | 804.7 |
| 1024 | burn | 20.005 ms | 19.961 ms | 20.031 ms | 107.3 |
| 2048 | fandhe-ai | 22.589 ms | 22.310 ms | 22.942 ms | 760.5 |
| 2048 | candle | 18.154 ms | 17.878 ms | 18.781 ms | 946.3 |
| 2048 | burn（無効: 要素誤差超過 fail=5/4194304, max_abs=3.529e-05, max_rel=3.052e-01） | 157.694 ms | 157.406 ms | 158.074 ms | - |

### (a) GEMM — Metal

| N | フレームワーク | 中央値 | Q1 | Q3 | GFLOP/s |
| --- | --- | --- | --- | --- | --- |
| 256 | fandhe-ai | 320.8 µs | 315.3 µs | 327.7 µs | 104.6 |
| 256 | candle | 267.2 µs | 245.7 µs | 274.9 µs | 125.6 |
| 256 | burn | 1.389 ms | 1.383 ms | 1.400 ms | 24.1 |
| 512 | fandhe-ai | 645.3 µs | 626.7 µs | 652.7 µs | 416.0 |
| 512 | candle | 498.6 µs | 489.2 µs | 507.6 µs | 538.4 |
| 512 | burn | 計測不可 | - | - | - |
| 1024 | fandhe-ai | 2.746 ms | 2.091 ms | 2.772 ms | 782.1 |
| 1024 | candle | 2.134 ms | 1.562 ms | 2.171 ms | 1006.1 |
| 1024 | burn | 計測不可 | - | - | - |
| 2048 | fandhe-ai | 7.531 ms | 7.326 ms | 12.810 ms | 2281.2 |
| 2048 | candle | 9.787 ms | 5.260 ms | 9.959 ms | 1755.4 |
| 2048 | burn | 計測不可 | - | - | - |
| 4096 | fandhe-ai | 36.962 ms | 36.652 ms | 37.281 ms | 3718.4 |
| 4096 | candle | 22.904 ms | 22.707 ms | 32.533 ms | 6000.7 |
| 4096 | burn | 計測不可 | - | - | - |

burn(wgpu) の Metal GEMM は N=512/1024/2048/4096 で計測不可（下記「環境 9 の計測不可・未計測
項目」参照。既知の upstream バグの継続）。fandhe-ai Metal GEMM N=2048 は 2281.2 GFLOP/s で、
環境 7 の 1764.6 GFLOP/s（fresh）から改善している

### (a') GEMM — Metal（デバイス/tape 再利用モード。イシュー #925）

| N | フレームワーク | 初期化(init_s) | 中央値 | Q1 | Q3 | GFLOP/s | fresh 中央値（参考） |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 256 | fandhe-ai | 35.812 ms | 307.0 µs | 300.9 µs | 310.4 µs | 109.3 | 320.8 µs |
| 512 | fandhe-ai | 36.516 ms | 680.0 µs | 671.5 µs | 687.7 µs | 394.8 | 645.3 µs |
| 1024 | fandhe-ai | 38.992 ms | 2.728 ms | 2.325 ms | 3.051 ms | 787.2 | 2.746 ms |
| 2048 | fandhe-ai | 52.898 ms | 10.521 ms | 8.065 ms | 12.312 ms | 1632.9 | 7.531 ms |
| 4096 | fandhe-ai | 91.964 ms | 41.214 ms | 40.838 ms | 49.076 ms | 3334.8 | 36.962 ms |

### (b) MLP 学習（784→256→10、ReLU、バッチ 64、MSE、SGD lr=0.01、1 ステップあたり時間）

| デバイス | フレームワーク | 中央値 | Q1 | Q3 |
| --- | --- | --- | --- | --- |
| cpu | fandhe-ai | 17.205 ms | 17.082 ms | 17.288 ms |
| cpu | candle | 818.0 µs | 750.0 µs | 921.5 µs |
| cpu | burn | 629.0 µs | 626.9 µs | 630.4 µs |
| metal | fandhe-ai | 19.078 ms | 18.899 ms | 19.202 ms |
| metal | candle | 619.5 µs | 610.3 µs | 638.5 µs |
| metal | burn | 1.604 ms | 1.595 ms | 1.617 ms |

### (b') MLP 学習（デバイス常駐パラメータ更新モード）

| デバイス | フレームワーク | 初期化(init_s) | 中央値 | Q1 | Q3 | fresh 中央値（参考） | fresh/reuse 比 | 最終 loss 突合（fresh） |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cpu | fandhe-ai | 160.6 µs | 7.972 ms | 7.883 ms | 8.054 ms | 17.205 ms | 2.16 倍 | 一致 |
| metal | fandhe-ai | 31.349 ms | 9.256 ms | 9.125 ms | 9.467 ms | 19.078 ms | 2.06 倍 | 一致 |

環境 7（b'）比: cpu 111.6 µs／18.054 ms → 160.6 µs／7.972 ms、metal 29.177 ms／20.789 ms →
31.349 ms／9.256 ms。cpu・metal とも reuse 中央値が約 2.2〜2.3 倍改善しており、環境 8 の CUDA
側改善と同じ #1078〜#1081 の効果と整合する

### (b'') MLP 学習 1 step のフェーズ分解（イシュー #1009。要点のみ）

- CPU reuse: backward 7.400 ms（91.7%）が支配項。fresh 比 backward 16.613 ms → 7.400 ms
  （約 2.2 倍改善）
- Metal reuse: backward 7.537 ms（81.9%）が支配項。fresh 比 backward 16.785 ms → 7.537 ms
  （約 2.2 倍改善）。device_update は 71.0 µs（0.8%）まで縮小

### (c) 推論スループット（同 MLP forward のみ、バッチ 64）

| デバイス | フレームワーク | 中央値 | Q1 | Q3 | バッチ/秒 |
| --- | --- | --- | --- | --- | --- |
| cpu | fandhe-ai | 526.6 µs | 519.0 µs | 543.2 µs | 1899 |
| cpu | candle | 163.1 µs | 146.2 µs | 205.4 µs | 6132 |
| cpu | burn | 282.2 µs | 282.1 µs | 283.8 µs | 3543 |
| metal | fandhe-ai | 608.1 µs | 581.4 µs | 733.1 µs | 1644 |
| metal | candle | 412.1 µs | 407.3 µs | 415.8 µs | 2427 |
| metal | burn | 1.502 ms | 1.494 ms | 1.506 ms | 666 |

環境 7 比: metal 推論 955.6 µs → 608.1 µs（約 1.57 倍改善）。cpu 推論は 536.7 µs → 526.6 µs
（ほぼ横ばい）

### 環境 9 のデータ有効性（checksum 突合・要素単位検証）

- 不一致なし（相互突合できた 28 行の checksum が参照値と一致）
- **無効（要素誤差超過。0 近傍の丸め差）**: burn/cpu/size=2048/fresh（環境 7 と同一事象の継続）
- `summarize.py --strict` の合算終了コード・警告件数は環境 8 節を参照（環境 8/9 は同一
  `--strict` 呼び出しの対象）

### 環境 9 の計測不可・未計測項目

- **Burn(wgpu) Metal GEMM の N=512/1024/2048/4096**: `skipped-m4max-0.5.0.log` に 4 件の
  `MEASURE_ERROR: gemm checksum is degenerate (0)` が記録されている。環境 7 と同一事象の継続
  （upstream 既知バグ tracel-ai/cubek#283。承認済みピン `burn =0.21.0` の範囲では修正版を
  取得できない。`docs/perf/burn-wgpu-metal-gemm-zero-result.md`）
- tch-rs: 環境 1〜8 と同じ理由で未計測

### 環境 9 の目標達成ゲート（`summarize.py --target candle`。イシュー #1051）

| タスク | デバイス | N | fandhe-ai 中央値 | candle 中央値 | 比（target/fandhe） | 判定 |
| --- | --- | --- | --- | --- | --- | --- |
| gemm | CPU | 256 | 317.2 µs | 335.8 µs | 1.06 倍 | **達成** |
| gemm | CPU | 512 | 710.2 µs | 704.9 µs | 0.99 倍 | 未達 |
| gemm | CPU | 1024 | 3.489 ms | 2.669 ms | 0.76 倍 | 未達 |
| gemm | CPU | 2048 | 22.589 ms | 18.154 ms | 0.80 倍 | 未達 |
| gemm | Metal | 256 | 307.0 µs | 267.2 µs | 0.87 倍 | 未達 |
| gemm | Metal | 512 | 680.0 µs | 498.6 µs | 0.73 倍 | 未達 |
| gemm | Metal | 1024 | 2.728 ms | 2.134 ms | 0.78 倍 | 未達 |
| gemm | Metal | 2048 | 10.521 ms | 9.787 ms | 0.93 倍 | 未達 |
| gemm | Metal | 4096 | 41.214 ms | 22.904 ms | 0.56 倍 | 未達 |
| train | CPU | 64 | 7.972 ms | 818.0 µs | 0.10 倍 | 未達 |
| train | Metal | 64 | 9.256 ms | 619.5 µs | 0.07 倍 | 未達 |
| infer | CPU | 64 | 526.6 µs | 163.1 µs | 0.31 倍 | 未達 |
| infer | Metal | 64 | 608.1 µs | 412.1 µs | 0.68 倍 | 未達 |

達成 1 件（gemm/CPU/N=256）・未達 12 件・判定不能 0 件。環境 8/9 合算（`--target candle`）:
**達成 1 / 未達 23 / 判定不能 2**（`summarize.py` の終了コード 3）

### 環境 9 の備考

- **CPU GEMM N=256 のみ candle を上回った（1.06 倍）。** 他 GEMM サイズ・全学習・全推論項目は
  未達のまま。環境 6/7 との比較では学習・推論の reuse 経路が大きく改善したが、candle・burn
  比の絶対水準はいずれも未達
- reuse（(a') 表）N=4096 は 3334.8 GFLOP/s（環境 7: 3031.3 GFLOP/s から微改善）。同一データの
  candle fresh 6000.7 GFLOP/s 比では約 56%

## 目標達成ゲート総括（環境 8/9・イシュー #1051/#1052）

- `python3 summarize.py results/raw/results-dgx-0.5.0.jsonl results/raw/results-m4max-0.5.0.jsonl --target candle`
  の終了コードは **3**（未達 23 件・判定不能 2 件・達成 1 件。全体 26 件）
- 未達・判定不能項目の追跡:
  - CUDA GEMM（gemm/CUDA/N=256,512,1024,2048〈判定不能〉,4096）: 既存トラッカー #1031（open）
  - Metal GEMM（gemm/Metal/N=256〜4096）: 既存トラッカー #1037（open）
  - CPU GEMM（gemm/CPU/N=256〜2048〈環境 9 の N=256 のみ達成〉）: 既存の CPU GEMM 個別トラッカー
    なし（`docs/perf/cpu-gemm-candle-cpu-retune.md` に再チューニング検討はあるが未実装・未
    Issue 化）
  - 学習・推論（train/infer の CPU/CUDA/Metal 全項目）: 個別の候補対応済みトラッカーなし
    （#1007 ツリー配下の後続 phase 候補）
  - **上記のうち既存トラッカーのない項目（CPU GEMM 全サイズ・学習/推論全項目）と「0.5.0 未収録
    改善〈#1108/#1110/#1111 等〉を含む次回 crates.io 公開後の再計測」は、本 PR では Issue 化を
    行わず `outOfScope` として引き継ぐ（`.claude/rules/out-of-scope-tracking.md` は新規 Issue
    起票・既存 Issue へのコメント追記のいずれもユーザー承認を要求するため。自動運転モードでは
    Issue 操作を行わず安全側に倒す）**

## 環境 10: DGX Spark GB10（fandhe-ai 0.6.0 横並び再計測・目標達成判定）

- ノード: 環境 2・環境 6・環境 8 と同一ノード（実ホスト名は `docs/real-hardware-verification-env.local.md`
  方式のローカル管理。未計測フィールドを推定で埋めない）
- ツールチェーン: 実行ログに rustc/cargo バージョンの記録なし。環境 8（rustc/cargo 1.97.0。同一
  ノード・直近の再ビルド）を参照。cargo tree（依存関係の実測）は `results/versions.txt` の
  環境 10/11 節を参照
- fandhe-ai バージョン: 0.6.0（crates.io 公開版。2026-09-02 公開。PR #1121 でピン更新）
- **0.6.0 の収録範囲**: `v0.5.0` タグ以降に main へ入っていた改善（#1108 Metal 選択テーブル・
  #1110 Metal SGD バッチング・#1111 CUDA variant selection 修正）は今回初めて反映される。CUDA
  GEMM 改善トラッカー #1031・Metal GEMM 改善トラッカー #1037・学習/推論 candle 比未達トラッカー
  #1118 は本計測時点で open のまま（未着手）であり、下記ゲート判定はそれらの改善適用前の現在地
  である
- 計測日: 2026-09-02（GPU アイドルを実行前後で確認・競合プロセスなしを確認したクリーン計測。
  初回計測〈同日〉は GPU 競合の影響を受けたため破棄し、本節は再計測分を採用する）
- 計測プロトコル: 環境 1・環境 2・環境 6・環境 8 と同一（f32、warmup 20 → 計測 20。学習は 100
  ステップ中先頭 20 を warmup。終端でホスト実体化 + checksum。要素単位検証を含む）
- 対象: `scripts/bench/framework-compare` の GEMM（fresh/reuse）・MLP 学習（フェーズ分解含む）・
  推論の 3 フレームワーク横並び（`./run_all_cuda.sh` 相当。fandhe-ai・candle・burn の cuda/cpu
  全タスク）
- 生データ: `results/raw/results-dgx-0.6.0.jsonl`（82 行）・`results/raw/skipped-dgx-0.6.0.log`
  （空 = 実行時失敗なし）・`results/run_all_cuda-dgx-0.6.0.log`（実行ログ）

### (a) GEMM — CUDA

| N | フレームワーク | 中央値 | Q1 | Q3 | GFLOP/s |
| --- | --- | --- | --- | --- | --- |
| 256 | fandhe-ai | 109.0 µs | 108.4 µs | 109.4 µs | 307.8 |
| 256 | candle | 76.5 µs | 76.3 µs | 76.7 µs | 438.7 |
| 256 | burn | 計測不可 | - | - | - |
| 512 | fandhe-ai | 327.5 µs | 327.1 µs | 328.0 µs | 819.7 |
| 512 | candle | 242.3 µs | 242.0 µs | 242.9 µs | 1107.7 |
| 512 | burn | 計測不可 | - | - | - |
| 1024 | fandhe-ai | 1.300 ms | 1.297 ms | 1.301 ms | 1652.4 |
| 1024 | candle | 933.8 µs | 932.3 µs | 937.4 µs | 2299.8 |
| 1024 | burn | 計測不可 | - | - | - |
| 2048 | fandhe-ai | 139.655 ms | 136.029 ms | 142.565 ms | 123.0 |
| 2048 | candle（無効: 要素誤差超過 fail=2/4194304, max_abs=3.624e-05, max_rel=2.811e-01） | 4.220 ms | 4.213 ms | 4.235 ms | - |
| 2048 | burn | 計測不可 | - | - | - |
| 4096 | fandhe-ai | 71.260 ms | 70.653 ms | 72.587 ms | 1928.7 |
| 4096 | candle | 57.370 ms | 57.170 ms | 58.517 ms | 2395.6 |
| 4096 | burn | 計測不可 | - | - | - |

burn の CUDA GEMM は本ラン（fresh 経路）では計測不可（`--tf32` opt-in 経路でのみ計測。下表
「(a-tf32)」参照）。環境 8 と同一事象の継続。

**fresh モード CUDA GEMM N=2048 が 139.655 ms と、他サイズ（N=256〜1024 は 0.1〜1.3 ms、
N=4096 は 71.3 ms）から突出して大きい。** reuse モード（下表 (a')）では 10.891 ms まで縮小
しており、環境 8（0.5.0）の fresh 140.214 ms／reuse 10.522 ms とほぼ同水準で、既知の fresh
N=2048 固有オーバーヘッド（`docs/perf/cuda-fresh-gemm-n2048-overhead-diagnosis.md`。#956/
#1025 で追跡した現象に近い水準）が引き続き残っている。目標達成ゲートの判定には下表 (a') の
reuse 中央値を用いる。本 PR では原因調査を行わない（記録のみ）。

### (a-tf32) GEMM TF32（burn 0.21.0 の CUDA 既定精度。REQ-2 統一複合判定の対象外・参考値）

| N | フレームワーク | 中央値 | Q1 | Q3 | 備考 |
| --- | --- | --- | --- | --- | --- |
| 256 | burn（無効: 要素誤差超過 fail=10538/65536, max_abs=1.581e-03, max_rel=1.554e+00） | 182.5 µs | 181.4 µs | 208.3 µs | TF32 降格 |
| 512 | burn（無効: 要素誤差超過 fail=42361/262144, max_abs=2.343e-03, max_rel=1.972e+00） | 487.3 µs | 477.0 µs | 678.8 µs | TF32 降格 |
| 1024 | burn（無効: 要素誤差超過 fail=169929/1048576, max_abs=3.643e-03, max_rel=1.965e+00） | 970.6 µs | 966.0 µs | 976.2 µs | TF32 降格 |
| 2048 | burn（無効: 要素誤差超過 fail=681454/4194304, max_abs=4.941e-03, max_rel=1.987e+00） | 4.179 ms | 4.160 ms | 4.226 ms | TF32 降格 |
| 4096 | burn（無効: 要素誤差超過 fail=2729050/16777216, max_abs=7.117e-03, max_rel=1.997e+00） | 39.274 ms | 39.199 ms | 39.711 ms | TF32 降格 |

`docs/burn-cuda-tf32.md`（burn 0.21 CUDA の TF32 既定降格）の既知事項どおりで、閾値は緩めず
「無効」表示のまま参考値として記録する（本 PR で精度契約外の切り分けは行わない）。

### (a) GEMM — CPU（ARM、環境 2・環境 6・環境 8 と同一機）

| N | フレームワーク | 中央値 | Q1 | Q3 | GFLOP/s |
| --- | --- | --- | --- | --- | --- |
| 256 | fandhe-ai | 379.9 µs | 363.4 µs | 398.6 µs | 88.3 |
| 256 | candle | 429.4 µs | 324.2 µs | 686.9 µs | 78.2 |
| 256 | burn | 975.4 µs | 580.0 µs | 1.071 ms | 34.4 |
| 512 | fandhe-ai | 2.425 ms | 2.046 ms | 2.513 ms | 110.7 |
| 512 | candle | 1.978 ms | 1.763 ms | 2.137 ms | 135.7 |
| 512 | burn | 3.294 ms | 3.287 ms | 3.297 ms | 81.5 |
| 1024 | fandhe-ai | 7.689 ms | 7.539 ms | 7.924 ms | 279.3 |
| 1024 | candle | 5.593 ms | 5.478 ms | 5.973 ms | 384.0 |
| 1024 | burn | 21.878 ms | 21.869 ms | 21.893 ms | 98.2 |
| 2048 | fandhe-ai | 36.081 ms | 35.813 ms | 36.611 ms | 476.2 |
| 2048 | candle（無効: 要素誤差超過 fail=2/4194304, max_abs=3.815e-05, max_rel=3.944e-01） | 34.153 ms | 33.575 ms | 34.654 ms | - |
| 2048 | burn（無効: 要素誤差超過 fail=5/4194304, max_abs=3.529e-05, max_rel=3.052e-01） | 155.117 ms | 154.983 ms | 155.381 ms | - |

**CPU GEMM N=256 で candle を上回った（fandhe-ai 379.9 µs 対 candle 429.4 µs、1.13 倍）。**
環境 8（0.5.0）は同サイズで未達（0.63 倍）だったため新規達成である（下記「目標達成ゲート」節
参照）。

### (a') GEMM — CUDA（デバイス/tape 再利用モード。イシュー #925）

| N | フレームワーク | 初期化(init_s) | 中央値 | Q1 | Q3 | GFLOP/s | fresh 中央値（参考） |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 256 | fandhe-ai | 410.535 ms | 110.6 µs | 110.2 µs | 111.4 µs | 303.4 | 109.0 µs |
| 512 | fandhe-ai | 419.364 ms | 650.3 µs | 641.6 µs | 657.7 µs | 412.8 | 327.5 µs |
| 1024 | fandhe-ai | 421.092 ms | 2.691 ms | 2.608 ms | 2.803 ms | 798.1 | 1.300 ms |
| 2048 | fandhe-ai | 427.057 ms | 10.891 ms | 10.697 ms | 10.957 ms | 1577.5 | 139.655 ms |
| 4096 | fandhe-ai | 470.451 ms | 71.146 ms | 69.498 ms | 569.759 ms | 1931.8 | 71.260 ms |

環境 8（0.5.0。reuse N=4096 67.918 ms／2023.6 GFLOP/s）とほぼ同水準（71.146 ms／1931.8
GFLOP/s）で、大きな後退・改善はない。

### (b) MLP 学習（784→256→10、ReLU、バッチ 64、MSE、SGD lr=0.01、1 ステップあたり時間）

| デバイス | フレームワーク | 中央値 | Q1 | Q3 |
| --- | --- | --- | --- | --- |
| cpu | fandhe-ai | 13.434 ms | 12.922 ms | 14.249 ms |
| cpu | candle | 2.592 ms | 2.276 ms | 2.911 ms |
| cpu | burn | 966.7 µs | 961.6 µs | 970.2 µs |
| cuda | fandhe-ai | 11.685 ms | 11.674 ms | 11.693 ms |
| cuda | candle | 274.2 µs | 272.4 µs | 278.2 µs |
| cuda | burn（TF32） | 788.0 µs | 517.4 µs | 1.001 ms |

### (b') MLP 学習（デバイス常駐パラメータ更新モード）

| デバイス | フレームワーク | 初期化(init_s) | 中央値 | Q1 | Q3 | fresh 中央値（参考） | fresh/reuse 比 | 最終 loss 突合（fresh） |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cpu | fandhe-ai | 1.150 ms | 7.997 ms | 7.294 ms | 8.657 ms | 13.434 ms | 1.68 倍 | 一致 |
| cuda | fandhe-ai | 216.819 ms | 5.505 ms | 5.497 ms | 5.514 ms | 11.685 ms | 2.12 倍 | 一致 |

環境 8（b'）比: cpu 900.0 µs／7.887 ms → 1.150 ms／7.997 ms（ほぼ横ばい）・cuda 211.199 ms／
5.374 ms → 216.819 ms／5.505 ms（ほぼ横ばい）。#1108/#1110/#1111 は主に Metal 側の改善のため
CUDA 学習には大きな寄与がなく、この横ばいと整合する。

### (b'') MLP 学習 1 step のフェーズ分解（イシュー #1009。要点のみ）

- CPU reuse: backward 6.010 ms（73.5%）が支配項。環境 8 の 5.887 ms（76.7%）とほぼ同水準
- CUDA reuse: backward 5.322 ms（96.3%）が支配項。環境 8 の 5.173 ms（96.3%）とほぼ同水準。
  device_update は 42.3 µs（0.8%）で環境 8（40.2 µs／0.7%）と同水準

### (c) 推論スループット（同 MLP forward のみ、バッチ 64）

| デバイス | フレームワーク | 中央値 | Q1 | Q3 | バッチ/秒 |
| --- | --- | --- | --- | --- | --- |
| cpu | fandhe-ai | 307.0 µs | 303.9 µs | 389.5 µs | 3257 |
| cpu | candle | 248.0 µs | 214.0 µs | 340.9 µs | 4033 |
| cpu | burn | 235.3 µs | 234.4 µs | 235.8 µs | 4249 |
| cuda | fandhe-ai | 156.9 µs | 155.7 µs | 159.3 µs | 6374 |
| cuda | candle | 42.4 µs | 41.0 µs | 43.6 µs | 23572 |
| cuda | burn（TF32） | 305.7 µs | 104.2 µs | 405.8 µs | 3271 |

環境 8 比: cuda 推論 152.8 µs → 156.9 µs（ほぼ横ばい）。cpu 推論は 403.6 µs → 307.0 µs
（改善）。

### 環境 10 のデータ有効性（checksum 突合・要素単位検証）

- 不一致なし（相互突合できた 27 行の checksum が参照値と一致）
- **無効（要素誤差超過。0 近傍の丸め差）**: candle/cuda/size=2048/fresh・candle/cpu/size=2048/fresh・
  burn/cpu/size=2048/fresh（環境 6/8 と同種の丸め差であり、閾値は緩めず「無効」表示のまま
  参考値として記録する）
- **無効（要素誤差超過。burn CUDA TF32 経路）**: burn/cuda/size=256〜4096 の全 5 サイズ（tf32
  opt-in 経路。上表「(a-tf32)」参照）。環境 6/8 と同一事象の継続
- `summarize.py --strict` は終了コード 2（環境 10/11 合算で無効レコード 9 件〈ユニーク〉。burn
  CUDA TF32 経路の 5 行が通常 GEMM 検証と TF32 専用検証の双方で警告されるため、実際の警告出力
  行数は 14 件）

### 環境 10 の計測不可・未計測項目

- 実行時に失敗した組み合わせ: なし（`skipped-dgx-0.6.0.log` は空）
- tch-rs: 環境 1〜9 と同じ理由で未計測

### 環境 10 の目標達成ゲート（`summarize.py --target candle`。イシュー #1051）

| タスク | デバイス | N | fandhe-ai 中央値 | candle 中央値 | 比（target/fandhe） | 判定 |
| --- | --- | --- | --- | --- | --- | --- |
| gemm | CPU | 256 | 379.9 µs | 429.4 µs | 1.13 倍 | **達成** |
| gemm | CPU | 512 | 2.425 ms | 1.978 ms | 0.82 倍 | 未達 |
| gemm | CPU | 1024 | 7.689 ms | 5.593 ms | 0.73 倍 | 未達 |
| gemm | CPU | 2048 | - | - | - | 判定不能（candle 無効データ） |
| gemm | CUDA | 256 | 110.6 µs | 76.5 µs | 0.69 倍 | 未達 |
| gemm | CUDA | 512 | 650.3 µs | 242.3 µs | 0.37 倍 | 未達 |
| gemm | CUDA | 1024 | 2.691 ms | 933.8 µs | 0.35 倍 | 未達 |
| gemm | CUDA | 2048 | - | - | - | 判定不能（candle 無効データ） |
| gemm | CUDA | 4096 | 71.146 ms | 57.370 ms | 0.81 倍 | 未達 |
| train | CPU | 64 | 7.997 ms | 2.592 ms | 0.32 倍 | 未達 |
| train | CUDA | 64 | 5.505 ms | 274.2 µs | 0.05 倍 | 未達 |
| infer | CPU | 64 | 307.0 µs | 248.0 µs | 0.81 倍 | 未達 |
| infer | CUDA | 64 | 156.9 µs | 42.4 µs | 0.27 倍 | 未達 |

達成 1 件（gemm/CPU/N=256）・未達 10 件・判定不能 2 件。gemm/CUDA・train/CUDA は reuse 中央値
で判定される（`summarize.py` の初期化コスト分離規約どおり）。CUDA GEMM の未達は既存トラッカー
#1031（open）、train/infer の未達は既存トラッカー #1118（open）が対象範囲。

### 環境 10 の備考

- **CPU GEMM N=256 が新規達成に転じた（環境 8: 0.63 倍未達 → 環境 10: 1.13 倍達成）。** 他 GEMM
  サイズ・全学習・全推論項目は未達のまま
- fresh モード CUDA GEMM N=2048 は 139.655 ms で環境 8（140.214 ms）とほぼ同水準の突出。reuse
  も 10.891 ms（環境 8: 10.522 ms）とほぼ同水準で、既知の fresh N=2048 固有オーバーヘッドが
  継続している（`docs/perf/cuda-fresh-gemm-n2048-overhead-diagnosis.md`）
- 学習・推論・reuse GEMM は総じて環境 8（0.5.0）比ほぼ横ばいであり、#1108/#1110/#1111（主に
  Metal 側の改善）が CUDA・CPU 側に大きな影響を与えていないことと整合する

## 環境 11: Apple M4 Max（fandhe-ai 0.6.0 横並び再計測・目標達成判定）

- チップ・OS: 環境 1・環境 5・環境 7・環境 9 と同一機（Apple M4 Max、macOS 26.6.2 / Darwin 25.6.0）
- ツールチェーン: rustc 1.96.0 (ac68faa20 2026-05-25) / cargo 1.96.0 (30a34c682 2026-05-25)（実測）
- fandhe-ai バージョン: 0.6.0（crates.io 公開版。2026-09-02 公開。PR #1121 でピン更新）
- **0.6.0 の収録範囲**: 環境 10 の同項目と同一（#1108/#1110/#1111 が今回初めて反映される。Metal
  GEMM 改善トラッカー #1037・学習/推論 candle 比未達トラッカー #1118 は本計測時点で open のまま）
- 計測日: 2026-09-02
- 計測プロトコル: 環境 1・環境 2・環境 6・環境 7・環境 8・環境 9・環境 10 と同一
- 対象: `scripts/bench/framework-compare` の GEMM（fresh/reuse）・MLP 学習（フェーズ分解含む）・
  推論の 3 フレームワーク横並び（`./run_all.sh` 相当。fandhe-ai・candle・burn の metal/cpu
  全タスク）
- 生データ: `results/raw/results-m4max-0.6.0.jsonl`（78 行）・
  `results/raw/skipped-m4max-0.6.0.log`（4 行。下記参照）・`results/run_all_m4max-0.6.0.log`
  （実行ログ）

### (a) GEMM — CPU

| N | フレームワーク | 中央値 | Q1 | Q3 | GFLOP/s |
| --- | --- | --- | --- | --- | --- |
| 256 | fandhe-ai | 313.5 µs | 277.2 µs | 331.9 µs | 107.0 |
| 256 | candle | 362.6 µs | 321.4 µs | 404.3 µs | 92.5 |
| 256 | burn | 509.8 µs | 506.7 µs | 555.3 µs | 65.8 |
| 512 | fandhe-ai | 746.9 µs | 730.3 µs | 776.6 µs | 359.4 |
| 512 | candle | 802.0 µs | 755.7 µs | 820.0 µs | 334.7 |
| 512 | burn | 2.618 ms | 2.615 ms | 2.630 ms | 102.5 |
| 1024 | fandhe-ai | 3.563 ms | 3.516 ms | 3.712 ms | 602.7 |
| 1024 | candle | 2.725 ms | 2.706 ms | 2.992 ms | 788.1 |
| 1024 | burn | 20.099 ms | 20.035 ms | 20.244 ms | 106.8 |
| 2048 | fandhe-ai | 21.454 ms | 21.051 ms | 22.034 ms | 800.8 |
| 2048 | candle | 17.690 ms | 17.480 ms | 19.730 ms | 971.2 |
| 2048 | burn（無効: 要素誤差超過 fail=5/4194304, max_abs=3.529e-05, max_rel=3.052e-01） | 158.390 ms | 157.827 ms | 159.318 ms | - |

### (a) GEMM — Metal

| N | フレームワーク | 中央値 | Q1 | Q3 | GFLOP/s |
| --- | --- | --- | --- | --- | --- |
| 256 | fandhe-ai | 285.5 µs | 263.5 µs | 294.9 µs | 117.5 |
| 256 | candle | 250.5 µs | 247.6 µs | 258.3 µs | 133.9 |
| 256 | burn | 1.397 ms | 1.369 ms | 1.548 ms | 24.0 |
| 512 | fandhe-ai | 693.9 µs | 681.7 µs | 716.9 µs | 386.9 |
| 512 | candle | 517.3 µs | 512.4 µs | 525.1 µs | 518.9 |
| 512 | burn | 計測不可 | - | - | - |
| 1024 | fandhe-ai | 2.792 ms | 2.051 ms | 2.814 ms | 769.0 |
| 1024 | candle | 2.149 ms | 1.582 ms | 2.199 ms | 999.4 |
| 1024 | burn | 計測不可 | - | - | - |
| 2048 | fandhe-ai | 8.656 ms | 7.049 ms | 11.059 ms | 1984.8 |
| 2048 | candle | 6.414 ms | 6.032 ms | 9.879 ms | 2678.5 |
| 2048 | burn | 計測不可 | - | - | - |
| 4096 | fandhe-ai | 34.333 ms | 34.025 ms | 44.611 ms | 4003.1 |
| 4096 | candle | 32.409 ms | 29.617 ms | 34.763 ms | 4240.8 |
| 4096 | burn | 計測不可 | - | - | - |

burn(wgpu) の Metal GEMM は N=512/1024/2048/4096 で計測不可（下記「環境 11 の計測不可・未計測
項目」参照。既知の upstream バグの継続）。fandhe-ai Metal GEMM N=4096 は 4003.1 GFLOP/s で、
環境 9 の 3718.4 GFLOP/s（fresh）から改善している（#1108 Metal 選択テーブルの効果と整合しうる
が、本 PR では因果関係の確認までは行わない）。

### (a') GEMM — Metal（デバイス/tape 再利用モード。イシュー #925）

| N | フレームワーク | 初期化(init_s) | 中央値 | Q1 | Q3 | GFLOP/s | fresh 中央値（参考） |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 256 | fandhe-ai | 35.970 ms | 306.7 µs | 294.1 µs | 322.7 µs | 109.4 | 285.5 µs |
| 512 | fandhe-ai | 35.965 ms | 738.5 µs | 726.4 µs | 758.6 µs | 363.5 | 693.9 µs |
| 1024 | fandhe-ai | 40.482 ms | 3.086 ms | 2.898 ms | 3.107 ms | 695.8 | 2.792 ms |
| 2048 | fandhe-ai | 52.157 ms | 9.888 ms | 9.032 ms | 10.685 ms | 1737.4 | 8.656 ms |
| 4096 | fandhe-ai | 101.236 ms | 48.865 ms | 40.644 ms | 49.925 ms | 2812.6 | 34.333 ms |

### (b) MLP 学習（784→256→10、ReLU、バッチ 64、MSE、SGD lr=0.01、1 ステップあたり時間）

| デバイス | フレームワーク | 中央値 | Q1 | Q3 |
| --- | --- | --- | --- | --- |
| cpu | fandhe-ai | 17.295 ms | 17.194 ms | 17.434 ms |
| cpu | candle | 880.8 µs | 814.6 µs | 949.3 µs |
| cpu | burn | 626.1 µs | 625.6 µs | 627.8 µs |
| metal | fandhe-ai | 19.136 ms | 18.984 ms | 19.245 ms |
| metal | candle | 661.6 µs | 622.4 µs | 1.267 ms |
| metal | burn | 1.603 ms | 1.595 ms | 1.625 ms |

### (b') MLP 学習（デバイス常駐パラメータ更新モード）

| デバイス | フレームワーク | 初期化(init_s) | 中央値 | Q1 | Q3 | fresh 中央値（参考） | fresh/reuse 比 | 最終 loss 突合（fresh） |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cpu | fandhe-ai | 166.1 µs | 8.073 ms | 7.971 ms | 8.129 ms | 17.295 ms | 2.14 倍 | 一致 |
| metal | fandhe-ai | 33.044 ms | 9.108 ms | 8.938 ms | 9.323 ms | 19.136 ms | 2.10 倍 | 一致 |

環境 9（b'）比: cpu 160.6 µs／7.972 ms → 166.1 µs／8.073 ms（ほぼ横ばい）、metal 31.349 ms／
9.256 ms → 33.044 ms／9.108 ms（ほぼ横ばい）。

### (b'') MLP 学習 1 step のフェーズ分解（イシュー #1009。要点のみ）

- CPU reuse: backward 7.293 ms（91.9%）が支配項。環境 9 の 7.400 ms（91.7%）とほぼ同水準
- Metal reuse: backward 7.580 ms（83.1%）が支配項。環境 9 の 7.537 ms（81.9%）とほぼ同水準。
  device_update は 76.1 µs（0.8%）で環境 9（71.0 µs／0.8%）と同水準

### (c) 推論スループット（同 MLP forward のみ、バッチ 64）

| デバイス | フレームワーク | 中央値 | Q1 | Q3 | バッチ/秒 |
| --- | --- | --- | --- | --- | --- |
| cpu | fandhe-ai | 567.5 µs | 519.7 µs | 576.0 µs | 1762 |
| cpu | candle | 202.8 µs | 148.7 µs | 285.6 µs | 4932 |
| cpu | burn | 282.2 µs | 282.0 µs | 283.2 µs | 3544 |
| metal | fandhe-ai | 805.9 µs | 783.3 µs | 811.8 µs | 1241 |
| metal | candle | 409.8 µs | 235.8 µs | 420.3 µs | 2440 |
| metal | burn | 1.501 ms | 1.495 ms | 1.510 ms | 666 |

環境 9 比: metal 推論 608.1 µs → 805.9 µs（悪化）。cpu 推論は 526.6 µs → 567.5 µs（微悪化）。

### 環境 11 のデータ有効性（checksum 突合・要素単位検証）

- 不一致なし（相互突合できた 28 行の checksum が参照値と一致）
- **無効（要素誤差超過。0 近傍の丸め差）**: burn/cpu/size=2048/fresh（環境 9 と同一事象の継続）
- `summarize.py --strict` の合算終了コード・警告件数は環境 10 節を参照（環境 10/11 は同一
  `--strict` 呼び出しの対象）

### 環境 11 の計測不可・未計測項目

- **Burn(wgpu) Metal GEMM の N=512/1024/2048/4096**: `skipped-m4max-0.6.0.log` に 4 件の
  `MEASURE_ERROR: gemm checksum is degenerate (0)` が記録されている。環境 7/9 と同一事象の
  継続（upstream 既知バグ tracel-ai/cubek#283。承認済みピン `burn =0.21.0` の範囲では修正版を
  取得できない。`docs/perf/burn-wgpu-metal-gemm-zero-result.md`）
- tch-rs: 環境 1〜10 と同じ理由で未計測

### 環境 11 の目標達成ゲート（`summarize.py --target candle`。イシュー #1051）

| タスク | デバイス | N | fandhe-ai 中央値 | candle 中央値 | 比（target/fandhe） | 判定 |
| --- | --- | --- | --- | --- | --- | --- |
| gemm | CPU | 256 | 313.5 µs | 362.6 µs | 1.16 倍 | **達成** |
| gemm | CPU | 512 | 746.9 µs | 802.0 µs | 1.07 倍 | **達成** |
| gemm | CPU | 1024 | 3.563 ms | 2.725 ms | 0.76 倍 | 未達 |
| gemm | CPU | 2048 | 21.454 ms | 17.690 ms | 0.82 倍 | 未達 |
| gemm | Metal | 256 | 306.7 µs | 250.5 µs | 0.82 倍 | 未達 |
| gemm | Metal | 512 | 738.5 µs | 517.3 µs | 0.70 倍 | 未達 |
| gemm | Metal | 1024 | 3.086 ms | 2.149 ms | 0.70 倍 | 未達 |
| gemm | Metal | 2048 | 9.888 ms | 6.414 ms | 0.65 倍 | 未達 |
| gemm | Metal | 4096 | 48.865 ms | 32.409 ms | 0.66 倍 | 未達 |
| train | CPU | 64 | 8.073 ms | 880.8 µs | 0.11 倍 | 未達 |
| train | Metal | 64 | 9.108 ms | 661.6 µs | 0.07 倍 | 未達 |
| infer | CPU | 64 | 567.5 µs | 202.8 µs | 0.36 倍 | 未達 |
| infer | Metal | 64 | 805.9 µs | 409.8 µs | 0.51 倍 | 未達 |

達成 2 件（gemm/CPU/N=256・N=512）・未達 11 件・判定不能 0 件。**gemm/CPU/N=512 は環境 9
（0.5.0）時点の 0.99 倍〈未達〉から 1.07 倍〈達成〉へ新規達成に転じた。** N=256 は環境 9 と
同様 1.06 倍 → 1.16 倍で達成を継続。環境 10/11 合算（`--target candle`）: **達成 3 / 未達 21 /
判定不能 2**（`summarize.py` の終了コード 3）。

### 環境 11 の備考

- **CPU GEMM N=256・N=512 の 2 サイズで candle を上回った。** 環境 9（0.5.0）の達成 1 件
  （N=256 のみ）から改善したが、他 GEMM サイズ・全学習・全推論項目は未達のまま
- reuse（(a') 表）N=4096 は 2812.6 GFLOP/s（環境 9: 3334.8 GFLOP/s から悪化）。同一データの
  candle fresh 4240.8 GFLOP/s 比では約 66%
- Metal 推論（fresh）が環境 9 比でやや悪化している（608.1 µs → 805.9 µs）。原因は本 PR では
  調査しない

## 目標達成ゲート総括（環境 10/11・v0.6.0 横並び再計測）

- `python3 summarize.py results/raw/results-dgx-0.6.0.jsonl results/raw/results-m4max-0.6.0.jsonl --target candle`
  の終了コードは **3**（未達 21 件・判定不能 2 件・達成 3 件。全体 26 件）
- 環境 8/9（0.5.0。達成 1／未達 23／判定不能 2）比では達成が 2 件増えた: DGX Spark
  gemm/CPU/N=256（新規達成。0.63 倍 → 1.13 倍）・M4 Max gemm/CPU/N=512（新規達成。0.99 倍 →
  1.07 倍）。M4 Max gemm/CPU/N=256 は達成を継続（1.06 倍 → 1.16 倍）
- CUDA GEMM・学習・推論（DGX）は環境 8 比ほぼ横ばいであり、#1108/#1110/#1111（主に Metal 側の
  改善）の CUDA・CPU 側への寄与は本ラウンドでは明確には確認できない
- 未達・判定不能項目の追跡:
  - CUDA GEMM（gemm/CUDA/N=256,512,1024,2048〈判定不能〉,4096）: 既存トラッカー #1031（open）
  - Metal GEMM（gemm/Metal/N=256〜4096）: 既存トラッカー #1037（open）
  - 学習・推論（train/infer の CPU/CUDA/Metal 全項目）: 既存トラッカー #1118（open）
  - CPU GEMM（gemm/CPU/N=512〈環境 10 のみ未達〉,1024,2048）: #1117 配下 #1148 で 5 回計測に
    より再判定済み（環境 14〈DGX〉・環境 15〈M4 Max〉参照。両実機とも N=512/1024 は未達、
    DGX N=2048 は candle 側要素誤差超過により判定不能、M4 Max N=2048 は未達）。詳細は
    `docs/perf/cpu-gemm-candle-gate-remeasurement.md` を参照

## 環境 12: DGX Spark GB10（GEMM 目標達成ゲート #1031 の 5 回計測再計測・イシュー #1142）

- ノード: 環境 2・環境 6・環境 8・環境 10 と同一ノード（実ホスト名は
  `docs/real-hardware-verification-env.local.md` 方式のローカル管理）
- ツールチェーン: rustc 1.97.0 (2d8144b78 2026-07-07) / cargo 1.97.0 (c980f4866 2026-06-30)（実測）
- NVIDIA driver 580.173.02 / CUDA (nvcc) 13.0 V13.0.88
- 対象: GEMM の N=1024/2048/4096（fandhe-ai reuse・candle fresh）のみ。環境 10 の全タスク横並び
  計測とは異なり、`run_gemm_gate_cuda.sh`（新規。イシュー #1142）による 5 回計測専用のスイープ
- **2 系列**（詳細は `docs/perf/cuda-gemm-candle-gate-remeasurement.md` §3）:
  - 正式系列（ラベル `0.6.0`）: 承認済みピン `fandhe-ai =0.6.0`（registry 解決）。生データ
    `results/raw/results-dgx-gemm-gate-0.6.0.jsonl`（30 行）・
    `results/raw/skipped-dgx-gemm-gate-0.6.0.log`（空）・
    `results/run_gemm_gate_cuda-dgx-0.6.0.log`
  - 参考系列（ラベル `head-7e3e4b6`）: 転送元コミット `7e3e4b6`（#1164 cp.async パイプライン
    結線後 HEAD）へ `--config patch.crates-io.fandhe-ai.path=...` で path 差し替えビルド
    （`[patch]`・`.cargo/config.toml` はコミットしていない）。生データ
    `results/raw/results-dgx-gemm-gate-head-7e3e4b6.jsonl`（30 行）・
    `results/raw/skipped-dgx-gemm-gate-head-7e3e4b6.log`（空）・
    `results/run_gemm_gate_cuda-dgx-head-7e3e4b6.log`
- GPU 競合確認: 両系列とも計測前後で `nvidia-smi utilization.gpu` 0%

### 5 回計測ゲート判定（正式系列 `0.6.0`）

| N | fandhe-ai reuse 中央値（min–max, n=5） | candle fresh 中央値（n=5） | candle/fandhe | GFLOP/s | 判定 |
| --- | --- | --- | --- | --- | --- |
| 1024 | 2.482 ms（2.391–2.617 ms） | 923.6 µs | 0.372 | 865.2 | 未達 |
| 2048 | - | - | - | - | 判定不能（candle 無効データ。下記参照） |
| 4096 | 68.337 ms（68.318–69.104 ms） | 56.324 ms | 0.824 | 2011.2 | 未達 |

### 5 回計測ゲート判定（参考系列 `head-7e3e4b6`。正式判定には用いない）

| N | fandhe-ai reuse 中央値（min–max, n=5） | candle fresh 中央値（n=5） | candle/fandhe | GFLOP/s | 判定 |
| --- | --- | --- | --- | --- | --- |
| 1024 | 2.414 ms（2.364–2.517 ms） | 923.5 µs | 0.383 | 889.7 | 未達 |
| 2048 | - | - | - | - | 判定不能（candle 無効データ。下記参照） |
| 4096 | 62.600 ms（60.252–63.437 ms） | 56.216 ms | 0.898 | 2195.5 | 未達（改善したが未達） |

### 環境 12 のデータ有効性（N=2048 candle 無効データ）

- 両系列とも `parity_fail_count=2, parity_total=4194304, parity_max_abs_err=3.623962e-05,
  parity_max_rel_err=2.811288e-01`（全 10 run で完全に決定的に一致。環境 10 の単発計測値とも
  一致）。原因は candle-core 0.11.0 の CUDA GEMM カーネル側にあり fandhe-ai 側は
  `parity_fail_count=0`（全 10 run）。詳細な分析は
  `docs/perf/cuda-gemm-candle-gate-remeasurement.md` §5 を参照。tolerance は緩めていない

### 環境 12 の #1031 ゲート判定（総括）

**未達 2 件（N=1024・N=4096）・判定不能 1 件（N=2048）。正式系列・参考系列のいずれも #1031
「reuse で candle 超え」は未達成。** N=4096 は参考系列（#1137 反映後）で 0.824→0.898 倍まで
改善したが 1.0 倍には届かない。詳細な突合・原因・ユーザー判断事項は
`docs/perf/cuda-gemm-candle-gate-remeasurement.md`（イシュー #1142）を参照。

## 環境 13: Apple M4 Max（GEMM 目標達成ゲート #1037 の 5 回計測再計測・イシュー #1147）

- ノード: 環境 3・環境 7・環境 9・環境 11 と同一ノード（実ホスト名は
  `docs/real-hardware-verification-env.local.md` 方式のローカル管理。ローカル直接実行のため
  rsync 転送は不要。`docs/real-hardware-verification-env.md` §1）
- ツールチェーン: rustc 1.96.0 (ac68faa20 2026-05-25) / cargo 1.96.0 (30a34c682 2026-05-25)（実測）
- macOS 26.6.2（BuildVersion 25G83）・Apple M4 Max・64GB
- 対象: GEMM の N=1024/2048/4096（fandhe-ai reuse・candle fresh）のみ。環境 11 の全タスク横並び
  計測とは異なり、`run_gemm_gate.sh metal <label>`（本 PR で device 汎用化。旧
  `run_gemm_gate_cuda.sh` の CUDA 専用実装を #1147 で device 汎用化し、呼び出し面は
  device 別薄い wrapper `run_gemm_gate_cuda.sh`／`run_gemm_gate_metal.sh` に分離。
  イシュー #1142→#1147）による 5 回計測専用のスイープ
- **2 系列**（詳細は `docs/perf/metal-gemm-candle-gate-remeasurement.md` §3）:
  - 正式系列（ラベル `0.6.0`）: 承認済みピン `fandhe-ai =0.6.0`（registry 解決）。生データ
    `results/raw/results-m4max-gemm-gate-0.6.0.jsonl`（30 行）・
    `results/raw/skipped-m4max-gemm-gate-0.6.0.log`（空）・
    `docs/perf/logs/metal-gemm-candle-gate-1147/run_gemm_gate_metal-m4max-0.6.0.log`
  - 参考系列（ラベル `head-bb7e35a`）: worktree HEAD `bb7e35a`（#1167/#1168 マージ後）へ
    `--config patch.crates-io.fandhe-ai.path=...` で path 差し替えビルド（`[patch]`・
    `.cargo/config.toml` はコミットしていない）。生データ
    `results/raw/results-m4max-gemm-gate-head-bb7e35a.jsonl`（30 行）・
    `results/raw/skipped-m4max-gemm-gate-head-bb7e35a.log`（空）
- 熱・電源状態確認: 両系列とも計測前後で `pmset -g therm` に thermal/performance warning なし
  （`docs/perf/logs/metal-gemm-candle-gate-1147/env_info.txt`）

### 5 回計測ゲート判定（正式系列 `0.6.0`）

| N | fandhe-ai reuse 中央値（min–max, n=5） | candle fresh 中央値（n=5） | candle/fandhe | GFLOP/s | 判定 |
| --- | --- | --- | --- | --- | --- |
| 1024 | 2.854 ms（2.673–2.966 ms） | 2.071 ms | 0.726 | 752.4 | 未達 |
| 2048 | 10.295 ms（9.225–12.090 ms） | 6.151 ms | 0.598 | 1668.8 | 未達 |
| 4096 | 38.941 ms（38.576–43.842 ms） | 22.948 ms | 0.589 | 3529.4 | 未達 |

### 5 回計測ゲート判定（参考系列 `head-bb7e35a`。正式判定には用いない）

| N | fandhe-ai reuse 中央値（min–max, n=5） | candle fresh 中央値（n=5） | candle/fandhe | GFLOP/s | 判定 |
| --- | --- | --- | --- | --- | --- |
| 1024 | 2.915 ms（2.366–3.058 ms） | 2.115 ms | 0.726 | 736.7 | 未達 |
| 2048 | 9.424 ms（9.070–9.946 ms） | 6.265 ms | 0.665 | 1823.0 | 未達（改善したが未達） |
| 4096 | 38.763 ms（38.673–39.459 ms） | 22.698 ms | 0.586 | 3545.6 | 未達 |

### 環境 13 のデータ有効性

- 両系列とも全 30 run で `parity_fail_count=0`・checksum が同一 N で一致（CUDA 側 #1142 の
  N=2048 candle 無効データは Metal では再現しなかった）。詳細は
  `docs/perf/metal-gemm-candle-gate-remeasurement.md` §5 を参照。tolerance は緩めていない

### 環境 13 の #1037 ゲート判定（総括）

**未達 3 件（N=1024・N=2048・N=4096）。正式系列・参考系列のいずれも #1037「reuse で candle
超え」は未達成。** N=2048 は参考系列（#1167/#1168 反映後）で 0.598→0.665 倍まで改善したが
1.0 倍には届かない。#1167/#1168 は `gemm metal` の NN 正方 GEMM 本番経路自体を変更していない
ため、系統的な改善は確認されなかった。詳細な突合・原因・ユーザー判断事項は
`docs/perf/metal-gemm-candle-gate-remeasurement.md`（イシュー #1147）を参照。


## 環境 14: DGX Spark GB10（GEMM 目標達成ゲート #1117 の 5 回計測再計測・イシュー #1148。CPU device 拡張）

- ノード: 環境 2・環境 6・環境 8・環境 10・環境 12 と同一ノード（実ホスト名は
  `docs/real-hardware-verification-env.local.md` 方式のローカル管理）
- ツールチェーン: rustc 1.97.0 (2d8144b78 2026-07-07) / cargo 1.97.0 (c980f4866 2026-06-30)（実測）
- CPU: Grace（Cortex-X925 ×10 + Cortex-A725 ×10、計 20 論理コア）
- 対象: GEMM の N=512/1024/2048（fandhe-ai reuse・candle fresh。加えて fandhe-ai fresh を
  参考記録）のみ。`run_gemm_gate.sh cpu <label>`（本 PR で cuda/metal に続き CPU device 対応
  拡張。呼び出し面は device 別薄い wrapper `run_gemm_gate_cpu.sh`〈新規〉）による 5 回計測
  専用のスイープ
- **単一系列**（正式系列 `fandhe-ai =0.6.0` のみ。詳細は
  `docs/perf/cpu-gemm-candle-gate-remeasurement.md` §3）
- **計測方式**: 共有作業ディレクトリ（`~/work/rust-ai-library-run`）に他の並列実行中セッションが
  更新した形跡を確認したため、本 Issue 専用の隔離ディレクトリ（`~/work/fc-1148/`。計測後に削除
  済み）へ必要なファイルのみ rsync して計測した（`docs/perf/cpu-gemm-candle-gate-remeasurement.md`
  §2 参照）
- 生データ: `results/raw/results-dgx-cpu-gemm-gate-0.6.0.jsonl`（45 行）、実行ログは
  `results/run_gemm_gate_cpu-dgx-0.6.0.log`

### 環境 14 の #1117 ゲート判定（5 回計測。`compare_gemm_gate.py --device cpu`）

| N | fandhe-ai reuse 中央値（min–max, n=5） | candle fresh 中央値（n=5） | candle/fandhe | GFLOP/s | 判定 | fandhe-ai fresh 中央値（参考。n=5） |
| --- | --- | --- | --- | --- | --- | --- |
| 512 | 2.376 ms（1.333–2.706 ms） | 1.805 ms | 0.760 | 113.0 | 未達 | 2.507 ms |
| 1024 | 7.085 ms（6.818–7.419 ms） | 5.604 ms | 0.791 | 303.1 | 未達 | 7.891 ms |
| 2048 | - | - | - | - | 判定不能（candle 無効データ。下記参照） | - |

### 環境 14 のデータ有効性（N=2048 candle 無効データ）

- fandhe-ai 側は全 45 run（reuse 15・candle fresh 15・fandhe fresh 15）で
  `parity_fail_count=0`。candle 側 N=2048 fresh のみ 5 run すべてで `parity_fail_count=2,
  parity_total=4194304, parity_max_abs_err=3.814697e-05, parity_max_rel_err=3.944416e-01`
  （run 間で完全に決定的に一致。環境 10 の単発計測値とも一致）。原因は candle-core 0.11.0 の
  CPU GEMM カーネル側にあり fandhe-ai 側は無関係。詳細な分析は
  `docs/perf/cpu-gemm-candle-gate-remeasurement.md` §5 を参照。tolerance は緩めていない

### 環境 14 の #1117 ゲート判定（総括）

**未達 2 件（N=512・N=1024）・判定不能 1 件（N=2048）。正式系列で #1117「reuse で candle
超え」は未達成。** 詳細な突合・原因・ユーザー判断事項は
`docs/perf/cpu-gemm-candle-gate-remeasurement.md`（イシュー #1148）を参照。

## 環境 15: Apple M4 Max（GEMM 目標達成ゲート #1117 の 5 回計測再計測・イシュー #1148。CPU device 拡張）

- ノード: 環境 3・環境 7・環境 9・環境 11・環境 13 と同一ノード（実ホスト名は
  `docs/real-hardware-verification-env.local.md` 方式のローカル管理。ローカル直接実行のため
  rsync 転送は不要）
- ツールチェーン: rustc 1.96.0 (ac68faa20 2026-05-25) / cargo 1.96.0 (30a34c682 2026-05-25)（実測）
- CPU: Apple M4 Max（P コア 12・E コア 4）
- 対象: GEMM の N=512/1024/2048（fandhe-ai reuse・candle fresh。加えて fandhe-ai fresh を
  参考記録）のみ。`run_gemm_gate.sh cpu <label>` による 5 回計測専用のスイープ
- **単一系列**（正式系列 `fandhe-ai =0.6.0` のみ。詳細は
  `docs/perf/cpu-gemm-candle-gate-remeasurement.md` §3）
- 負荷状態: 計測実行中、本マシンでは並列稼働する他の Claude Code エージェントセッションが
  複数存在した（`uptime` load average: 計測前 7.49/5.37/4.17、計測後 9.30/6.00/4.43）。
  計測専有ではないため GFLOP/s の絶対値には背景負荷によるノイズが乗っている可能性がある
- 生データ: `results/raw/results-m4max-cpu-gemm-gate-0.6.0.jsonl`（45 行）、実行ログは
  `results/run_gemm_gate_cpu-m4max-0.6.0.log`

### 環境 15 の #1117 ゲート判定（5 回計測。`compare_gemm_gate.py --device cpu`）

| N | fandhe-ai reuse 中央値（min–max, n=5） | candle fresh 中央値（n=5） | candle/fandhe | GFLOP/s | 判定 | fandhe-ai fresh 中央値（参考。n=5） |
| --- | --- | --- | --- | --- | --- | --- |
| 512 | 744.0 µs（720.4–771.1 µs） | 699.1 µs | 0.940 | 360.8 | 未達 | 727.8 µs |
| 1024 | 3.787 ms（3.667–3.864 ms） | 2.749 ms | 0.726 | 567.0 | 未達 | 3.494 ms |
| 2048 | 24.098 ms（23.154–24.730 ms） | 17.694 ms | 0.734 | 712.9 | 未達 | 23.120 ms |

### 環境 15 のデータ有効性

- 全 45 run で `parity_fail_count=0`・checksum が同一 N で fandhe-ai/candle 間一致。
  `compare_gemm_gate.py --device cpu` は 3 size とも「判定不能」を出さず確定判定（未達）。
  詳細は `docs/perf/cpu-gemm-candle-gate-remeasurement.md` §5 を参照。tolerance は緩めていない

### 環境 15 の #1117 ゲート判定（総括）

**未達 3 件（N=512・N=1024・N=2048）。正式系列で #1117「reuse で candle 超え」は未達成。**
未達原因分析（計測境界固定費・並列化・マイクロカーネル効率・packing）は
`docs/perf/cpu-gemm-candle-gate-remeasurement.md` §8 に記録した。詳細な突合・ユーザー判断
事項は同ドキュメント（イシュー #1148）を参照。

### 目標達成ゲート総括（CPU GEMM 追跡行の更新）

環境 10/11 の「目標達成ゲート総括」節に記載していた CPU GEMM 追跡行は、#1117 配下 #1148 で
5 回計測により再判定した（本節・環境 14/15 参照）。DGX Spark・M4 Max とも N=512/1024 は未達成、
DGX N=2048 は candle 側要素誤差超過により判定不能、M4 Max N=2048 は未達成が確定した。
