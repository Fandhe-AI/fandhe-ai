# Metal 4 `tensor<>`＋Metal Performance Primitives（MPP）行列積の可用性・純カーネル時間調査（#1326）

イシュー #1326「Metal 4 `tensor<>`＋Metal Performance Primitives 行列積の可用性・純カーネル時間を調査し
『完全自作コア』との整合の政策判断材料を整備する」に対応する。`docs/backend-metal-mlx-classic-nax-decision.md`
（#549。以下「#549」）が記録した MLX の NAX 経路（`MetalPerformancePrimitives` の `mpp::tensor_ops::matmul2d`
を使う経路）不採用判断の再訪条件のうち「M5 世代（Neural Accelerator 搭載）実機」以外の 2 点
（可用性・純カーネル時間の実測データ）を、本 M4 Max 実機（Neural Accelerator 非搭載）の範囲で先行して
整備する調査イシューである。**本ドキュメントは調査結果の記録であり、採否の結論は出さない**（§6 参照）。

## 判断サマリ

- **可用性**: Metal Toolchain 未導入の M4 Max 実機でも、**ランタイムコンパイル**（`newLibraryWithSource_
  options_error` に `MTLLanguageVersion::Version4_0` を指定）で `#include <metal_tensor>`・
  `#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>` を含む MSL 4 ソースがコンパイルでき、
  `mpp::tensor_ops::matmul2d` を使った GEMM カーネルが実際に動作することを確認した（§1・§2）。
- **バインディング到達性**: `objc2-metal =0.3.2`（依存追加・feature 変更なし）から到達可能な経路（Route C。
  カーネル内で `device float*` から `tensor_inline` を構築する方式）で classic `MTLComputeCommandEncoder`
  （既存 `MetalContext::encode` 経路）から起動できることを確認した（§2）。
- **正確性**: CPU 参照実装（`fandhe_ai_backend_cpu::parity::matmul_reference_fma`）との比較・本番選択構成
  との比較のいずれも REQ-2 複合判定に pass（多くは `mean_abs_diff=0` の bit 完全一致）した（§4）。
- **純カーネル時間**: 本番選択構成（`tile::select_for_device`。自作の `gemm_simdgroup_tiled`）に対し、
  MPP `matmul2d`（64×32 タイル・4 simdgroup。Apple ヘッダ既定の基本構成）は **N=1024 でほぼ同等
  （5 run 中央値 1.02 倍）、N=2048 で約 1.26 倍遅い、N=4096 で約 1.90 倍遅い**（M4 Max 実機・5 run 中央値。
  §3）。Neural Accelerator を持たない世代のためコンパイラ品質分のみでの比較になる点は #549 の理解と整合する
  （§5）。
- **「完全自作コア」との整合**: 採否・線引きの結論はユーザー判断に委ねる（§6）。本ドキュメントは選択肢の
  列挙と実測データの提供に留める。

## §1 実行環境・可用性

### §1.1 環境（内部ホスト名は含めない。詳細は `docs/perf/logs/metal-gemm-mpp-tensor-1326/env_info.txt`）

- macOS 26.6.2（25G83）・Apple M4 Max・GPU architecture `applegpu_g16s`
- `MTLDevice::supportsFamily(MTLGPUFamily::Metal4)` = `true`（実機実測。`mpp_metal4_compile_probe`）
- **Metal Toolchain は未導入**（`xcodebuild -showComponent MetalToolchain` → `Status: uninstalled`。
  `xcrun -sdk macosx metal --version` は `missing Metal Toolchain` エラー）。オフライン
  `xcrun metal -std=metal4.0` によるコンパイルは本調査では実施していない（Toolchain 導入は数 GB のシステム
  全体変更であり、他セッションと共有の実機環境のため本調査では見送った。§8 引き継ぎ）
- SDK に `MetalPerformancePrimitives.framework`（`Headers/MPPTensorOpsMatMul2d.h` 等）が存在し、
  ランタイムシェーダコンパイラ側ヘッダ（`GPUCompiler.framework` 配下）に `metal_tensor` が存在することを
  ファイルシステム上で確認済み

### §1.2 コンパイル可否（ランタイム。`mpp_metal4_compile_probe`）

`crate::pipeline::compile_options()`（`MathMode::Safe` + `MathFloatingPointFunctions::Precise`。本番
`compile_gemm_library` と同一の丸め方針）に `setLanguageVersion(MTLLanguageVersion::Version4_0)` を追加した
`MTLCompileOptions` で、`#include <metal_tensor>`・`#include <MetalPerformancePrimitives/
MetalPerformancePrimitives.h>`・`mpp::tensor_ops::matmul2d_descriptor` 参照を含む最小ソースを
`newLibraryWithSource_options_error` へ渡したところ、**`compile_result=ok`** で成功した（Metal Toolchain
未導入でも成立。実機実測ログ: `docs/perf/logs/metal-gemm-mpp-tensor-1326/compile_probe.log`）。

## §2 バインディング到達性（Route C の実装）

計画時点で整理した 3 経路（Route C／A'／B）のうち、本調査は **Route C のみ実装**した（A'／B は §8 へ引き継ぎ）。

| Route | 内容 | 本調査での扱い |
|---|---|---|
| **C** | カーネルが `device float*`（通常のバッファ引数）を受け取り、カーネル本体内で `tensor<device float, dextents<int32_t,2>, tensor_inline>(ptr, extents)` を構築してから `matmul2d::run` を呼ぶ。ホスト側は既存 `MetalContext::encode`（classic encoder）のみで完結 | **実装・実測済み**（本ドキュメントの実測データはすべて Route C） |
| A' | ホスト `MTLTensor`（`MTLBuffer::newTensorWithDescriptor_offset_error`）を作り `gpuResourceID` を argument buffer へ書き込んで classic encoder から参照する | 未実装（§8） |
| B | `MTL4CommandQueue`／`MTL4ComputeCommandEncoder`／`MTL4ArgumentTable` 等の Metal 4 専用コマンド経路 | 未実装（`objc2-metal =0.3.2` の既定 feature に `MTL4*` 系バインディングが含まれることのみソースコード上で確認済み。§8） |

Route C は `objc2-metal =0.3.2` の既定 feature（依存追加・feature 変更なし）から到達可能で、SDK 26 系
`MTLComputeCommandEncoder.h` に tensor を直接バインドする API が存在しない（実測確認済み）制約を回避できる。

### §2.1 実装（`crates/backend-metal/src/gemm_mpp_diag_tests.rs`。診断テスト限定・本番コード無変更）

- カーネル `mpp_gemm_nn_f32`: `device float* A_ptr/B_ptr/C_ptr [[buffer(0..2)]]` + `constant uint& N_DIM
  [[buffer(3)]]`（正方形状限定。§7 のスコープ判断）を受け取り、`tensor_inline` を構築して
  `matmul2d_descriptor(64, 32, dynamic_extent, false, false, false)`・`execution_simdgroups<4>` で `run` する
  （`MPPTensorOpsMatMul2d.h` 冒頭の基本例と同一のタイル・並列度）。`coding-rust.md`「境界検査を省略しない」
  （性能下限・最適化の達成を理由にカーネル側の手動境界チェックを省略しない。CPU/CUDA/Metal 全カーネルに
  適用）を満たすため、`tgid` から算出したタイルオフセットが行列サイズ以上の threadgroup を早期 return する
  明示的な境界検査をカーネル本体へ追加した（未検査のまま `slice()`／`matmulOp.run` へ渡さない。
  codex-review 指摘・イシュー #1326 対応）。端タイル（n が 64/32 で割り切れない場合の残余サイズ）自体の
  縮約処理は `matmul2d_descriptor` の `dynamic_extent` 契約に基づき `matmul2d::run` 側で行われる
  （ヘッダ冒頭コメント。MPP が公開する dynamic extent API を使う設計であり、上記の明示的な早期 return
  ガードと役割分担する）。
- ホスト側ディスパッチ（`encode_mpp_nn`）は `gemm.rs::encode_dispatch_tiled` と同一形の
  `setBuffer_offset_atIndex`／`setBytes_length_atIndex`／`dispatchThreadgroups_threadsPerThreadgroup` のみ。
  `MTLTensor`／`MTL4*` 型は一切使わない。

### §2.2 `tensor_inline` コンストラクタの const 制約（実装上の詰まり 1）

`tensor<device float, ...>` は `data_handle_type`（`device float*`。非 const）を要求し、`device const float*`
を渡すとコンパイルエラーになる（実機実測: `would lose const qualifier`）。A/B 入力バッファも `device float*`
（非 const）で宣言することで解消した（`gemm_mpp_diag_tests.rs:100-101` 相当）。

### §2.3 `tgid.x`/`tgid.y` の割当（実装上の詰まり 2。ヘッダ冒頭コメント例の誤記の疑い）

`MPPTensorOpsMatMul2d.h` 冒頭コメント例をそのまま実装（`A.slice(0, tgid.y*64)`／`B.slice(tgid.x*32, 0)`）
すると、M4 Max 実機で N=64 の出力の約半数（`fail_count=2048/4096`）が不一致になることを実測で確認した。
同ヘッダの別箇所にある境界検査コメント `if (tgid.x*64 + 63 < M && tgid.y*32 + 31 < N)`（`tgid.x` が M タイル・
`tgid.y` が N タイルを指す）と dispatch grid（`MTLSizeMake((M+63)/64, (N+31)/32, 1)`。width=M タイル数）は
整合する一方、冒頭のスライス例コメントはこれと矛盾する割当になっている。**冒頭のコメント例（非コンパイル対象）
を誤記と判断し、境界検査コメント・dispatch grid と整合する向き（`tgid.x`→M タイル・`tgid.y`→N タイル）へ
入れ替えて実装した**。入れ替え後は §4 の正確性スモークが全 pass（`mean_abs_diff=0`）した。

## §3 純カーネル時間 A/B（実機実測。M4 Max・5 run 中央値）

`MetalContext::synchronize_with_gpu_timestamps`（GPU タイムスタンプ。#1276 で導入・本番 `synchronize()` は
無変更）による `kernel_gpu_secs` を計測指標とし、E7/E8（`gemm_bk32_diag_tests.rs`／`gemm_bm128_diag_tests.rs`）
と同型の warmup 20／測定 20・trial ごとの base/head 開始順回転・5 プロセス起動で計測した
（`mpp_kernel_gpu_ab_vs_production_select`。head=MPP `matmul2d`〈Route C〉・base=`tile::select_for_device`
が選ぶ本番選択構成）。

| N | 本番選択構成（base） | `kernel_gpu_median_ms`（base） | `kernel_gpu_median_ms`（head=MPP） | `head_over_base`（5 run 中央値） | 符号一貫性 |
|---|---|---|---|---|---|
| 1024 | `bm64 bn32 bk8 wm4 wn1 staged` | 0.2246〜1.0306（外れ値含む。§3.1） | 0.2290〜0.9955 | **1.0223** | 4/5 head>base（1 run のみ逆転） |
| 2048 | `bm64 bn32 bk16 wm2 wn2 staged` | 1.6004〜2.9469 | 2.0304〜3.2794 | **1.2632** | 5/5 head>base |
| 4096 | `bm32 bn64 bk16 wm2 wn2 staged` | 13.5852〜13.6067 | 24.8969〜28.9961 | **1.9015** | 5/5 head>base |

詳細（全 5 run の生値・env_info・uptime・pmset）は `docs/perf/logs/metal-gemm-mpp-tensor-1326/aggregate.md` を
参照。

### §3.1 計測ノイズについて

本機は他セッションと共有の実機で、計測時の `uptime` は load average 約 2.5〜5.6（20 ユーザーログイン中。
`docs/perf/logs/metal-gemm-mpp-tensor-1326/uptime_before_run*.txt`）と非ゼロ負荷だった。run2/run3 の
N=1024/2048 base 側に外れ値が見えるが、`head_over_base` の**符号**（N=2048/4096 で head が一貫して遅い）は
5 run すべてで揃っており、E7/E8 が採用する「符号一貫性があれば負荷下でも判定不可とはしない」方針
（`docs/perf/metal-gemm-n4096-kernel-gap.md` §14・§16 の先例）に従い、上記中央値を実測結果として採用する。
N=1024 のみ 1 run で符号が逆転しており、実質的にほぼ互角（ノイズの範囲内）と解釈する。

## §4 正確性（REQ-2 複合判定）

- `mpp_matches_cpu_reference`（N=8/64/100。CPU 参照実装 `matmul_reference_fma` との比較）: 3/3 pass、いずれも
  `mean_abs_diff=0`（bit 完全一致。`docs/perf/logs/metal-gemm-mpp-tensor-1326/parity_smoke.log`）。
- `mpp_kernel_gpu_ab_vs_production_select`（本番選択構成〈自作 `gemm_simdgroup_tiled`〉との比較。N=1024/2048/
  4096）: 5 run すべて trial 0 の複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）に pass。**bit 完全
  一致は要求していない**（MPP は別カーネル実装のため加算順序が異なりうる。`coding-rust.md`
  「バックエンド間数値一致は統一複合判定」を単一バックエンド内の異実装比較にも適用する方針は E7/E8 と同じ）。

## §5 Neural Accelerator 非搭載世代であることの含意

M4 Max は Neural Accelerator を搭載しない世代のため、本調査の純カーネル時間差（§3）は MPP 実装の
「コンパイラ・スケジューラ品質」のみを反映し、専用ハードウェアパスの有無を反映しない。#549 §3 の
再訪条件（1. M5 世代実機の入手、2. 対応 macOS・MPP 利用可否確認、3. classic 経路が REQ-8 未達の場合に限る）
は本ドキュメントでは変更しない。本調査は 2 点目（利用可否）を M4 Max の範囲で先行実測したに過ぎず、
1 点目（M5 世代実機）が満たされない限り「専用ハードウェアパスを含めた比較」は行えない。

## §6 「完全自作コア」との整合（ユーザー判断事項。本ドキュメントは結論を出さない）

MPP はシェーダ側 vendor primitive（Cargo 依存ではなく Metal SDK フレームワークヘッダの `#include`）であり、
これを REQ-1「完全自作コア」の範囲内と見なすかは以下の選択肢を提示するのみに留める。

- **(a) 不採用**: `simdgroup_multiply_accumulate` 相当の simdgroup レベル primitive までを自作コアの範囲とし、
  MPP（`matmul2d` のようなブロック全体を代行する高レベル primitive）は採用しない。#549 の NAX 経路不採用判断
  と同じ立場の継続。
- **(b) 採用可**: シェーダ側 vendor primitive を `simdgroup_matrix`（現行 `gemm_simdgroup_tiled` が使う API）
  と同格の「GPU ベンダー提供の計算 primitive」と見なし、採用可とする。採用する場合の適用範囲案:
  - NN・f32・特定形状帯（本調査は N=1024 のみ互角。N=2048/4096 で後退のため、現状の実測だけでは
    `tile::select` への組み込みは正当化できない）に限定する
  - `tile::select` の候補としてではなく、明示的な opt-in API（既存 `set_cuda_tf32_gemm_enabled` 相当の
    pattern。`docs/cuda-tf32-optin-api-decision.md`）として提供する
  - REQ-2 の判定は本調査と同じ複合判定（bit 一致を要求しない）とする
- **(c) 限定採用**: cfg／環境変数による opt-in 限定（実験的機能として明示し既定 OFF）で許容する。

いずれの案も採用には**別途ユーザー承認が必要**（`.claude/rules/security.md`「自己修復ループ固有の
ガードレール」・`deps-policy.md` の精神と同旨。MPP 自体は Cargo 依存の追加ではないため deps-policy.md の
対象外だが、REQ-1 の解釈変更に相当するため同等の慎重さで扱う）。

## §7 スコープ・不変条件

- `tile.rs`／`gemm.rs`／`shaders/gemm.metal`／`Cargo.toml`／`Cargo.lock` への変更はない
  （`git diff origin/main --stat` で確認可能）。
- 本調査は**正方形状（M=N=K）のみ**を対象とした。extents 順序（`extent(0)` が連続次元）の取り違えは
  非正方形状で顕在化しやすいが、正方形状では回避できるため本調査のスコープでは検証していない
  （§8 引き継ぎ）。
- 新規 `unsafe` は `#[cfg(all(test, target_os = "macos"))]` 限定の診断テストモジュール内に閉じ、
  `gemm.rs::encode_dispatch_tiled` と同一形の FFI 呼び出し（`setBuffer_offset_atIndex`／
  `setBytes_length_atIndex`）のみを追加した。本番コードへの unsafe 追加はない。

## §8 スコープ外・引き継ぎ（`out-of-scope-tracking.md` 対応）

以下は本調査のスコープ外とし、起票はユーザー承認後に行う（自動運転モードでは起票しない）。

- Route A'（ホスト `MTLTensor`＋argument buffer 経由の classic encoder バインド）・Route B（`MTL4*`
  コマンド経路）の実装・計測
- MPP 採用時の本番結線・`tile::select` 候補化・f16／bf16／転置パターン（NT/TN/TT）・非正方形状の計測
  （extents 順序の非正方版検証を含む）
- Metal Toolchain 導入後のオフライン `xcrun metal -std=metal4.0` によるコンパイル可否の追加確認
  （ランタイムコンパイルで可用性は確認済みだが、`.metallib` 事前ビルド経路の可否は未確認）
- M5 世代（Neural Accelerator 搭載）実機での再計測（#549 §3 の再訪条件 1〜2 点目）
- 64×64 等、64×32 以外の `matmul2d_descriptor` タイル構成での再計測

## §9 参照

- `MetalPerformancePrimitives.framework/Headers/MPPTensorOpsMatMul2d.h`（SDK 26 系。ヘッダ本文はリポジトリへ
  複製していない）
- `GPUCompiler.framework` 配下 `metal_tensor` ヘッダ（ランタイムシェーダコンパイラ側）
- `docs/backend-metal-mlx-classic-nax-decision.md`（#549。NAX 経路不採用判断・再訪条件）
- `docs/perf/metal-gemm-n4096-kernel-gap.md`（E7/E8 の交互測定・符号一貫性判断の先例）
- `crates/backend-metal/src/gemm_mpp_diag_tests.rs`（本調査の診断テスト実装）
- `docs/perf/logs/metal-gemm-mpp-tensor-1326/`（実測ログ・env_info・aggregate.md）
