# CUDA Tensor Core（WMMA/mma）カーネル設計メモ

- 対応イシュー: #60（TASK-11.1a、親 #59 TASK-11.1 の再分解サブタスク先頭）／#485（GEMM 性能改善ツリー Phase A・親 #480 の A-5。11 節を追記）／#483（GEMM 性能改善ツリー Phase A・親 #480 の A-3。TMA sm_121 プローブ spike。12 節を追記）／#484（GEMM 性能改善ツリー Phase A・親 #480 の A-4。setmaxnreg プローブ spike。13 節を追記）
- 位置づけ: 本文書は**設計メモのみ**であり、実行可能なカーネル実装は含まない。受け入れ条件は「命令選定・タイル構成・根拠」の 3 要素が記録されていることの 1 点（#60 本文）。
- 対象外（後続サブタスクのスコープ。重複実装を避けるため明記する）:
  - #61（11.1b）: f16 WMMA GEMM の実装
  - #62（11.1c）: TF32/f32 経路の実装
  - #63（11.1d）: 共有メモリ・タイル基本最適化の実装
  - #187（11.1h）: `mma.sync` PTX 直叩き・`ldmatrix`・`cp.async` パイプライン・XOR swizzle（本設計では「将来経路」として言及に留める）
  - #186（11.1g）: TF32/f16 経路の数値一致閾値の実測再評価（閾値の変更はユーザー承認必須。本設計では論点整理のみ）
  - #66（TASK-11.2）: ディスパッチ規則の設計・実装（本設計では「引き渡し事項」の列挙に留める）

## 1. 前提・現状

- **PoC-v2-3 実測**（`docs/spec/03-poc/poc-v2-3-cuda-gemm/README.md`）: tiled f32 GEMM が 1.832 TFLOPS（M=N=K=4096）。同一実機・同一形状での PyTorch f16 実効値は 97.6 TFLOPS（tensor core 経路）。tiled/PyTorch 比は f32 で約 10.3%、f16（自作カーネルは tensor core 未使用のためスカラー実装のまま）で約 1.9%。tensor core 化により「現在の tiled 実装から 1 桁以上の改善余地がある」と見積もられている（同 README「tensor core 化の段階見積もり」節）。
- **REQ-11**（`docs/spec/04-requirements.md`「REQ-11: 行列演算ユニットの活用」）: CUDA バックエンドで Tensor Core（WMMA/mma）を用いた自作カーネルの実装を受け入れ基準とする。実装完了までは tiled 実装を暫定経路とし、tensor core 化を REQ-8 の CUDA 最適化後下限（暫定 40%）達成の前提条件として明記する。
- **`crates/backend-cuda` の現状**: `device.rs`（動的ロード・panic 回避ゲート、TASK-1.7a／#32・TASK-1.9a／#44）・`error.rs`・`nvrtc.rs`（NVRTC コンパイル基盤）・`kernels.rs`（`gemm_naive_f32`/`gemm_naive_f16` の CUDA C ソース文字列のみ。tiled 版は未着手）・`gemm.rs`（naive カーネル起動ラッパー）が存在する。`BackendOps`/`BackendError` へのフルマッピングは TASK-1.9c（#46）のスコープであり本クレートでは未実装。tensor core 版カーネルはソース・起動 API ともに未着手であり、本設計メモが実装（#61 以降）の起点になる。

## 2. sm_121（GB10）の Tensor Core 対応状況

- **compute capability**: DGX Spark GB10 は compute capability 12.1（`sm_121`）。RTX 50 系コンシューマ GPU と同じ Blackwell の「コンシューマ系譜」（SM12x）に属する（PoC-v2-3 実機ログ、CUDA SDK 13.0.3・ドライバ 580.159.03）。
- **命令セット系譜**: SM12x（sm_120/121）の Tensor Core プログラミングモデルは、データセンター系 Blackwell（SM100/`sm_100`）の `tcgen05` 命令・専用メモリ（TMEM）を要求せず、Hopper（SM90）の `wgmma` も要求しない。SM12x は Ampere（SM80）以来の `mma.sync`／WMMA 系プログラミングモデルを維持する（出典: [Analyzing Nvidia GB10's GPU — Chester Lam](https://chipsandcheese.com/p/analyzing-nvidia-gb10s-gpu)、[Day 3: DGX Spark Unpacked — Kubesimplify](https://blog.kubesimplify.com/day-3-the-dgx-spark-unpacked-gb10-unified-memory-sm-121-and-the-one-reason-this-hardware-exists)）。
  - この事実は本設計の中心的な前提を確定させる: sm_121 向けカーネルは `wmma::fragment`（C++ API）または `mma.sync.aligned`（PTX）のいずれかで実装可能であり、`tcgen05`/`wgmma` 系の新命令は選択肢に入らない。二次情報（技術ブログ）に基づく本節の記述は、11 節で CUTLASS 一次ソースの実測読解により裏付け・構造化している。
- **対応 fragment shape・精度**（Ampere 系譜の WMMA API 前提）:
  - f16 入力: `m16n16k16`・`m32n8k16`・`m8n32k16`、累算は f16 または f32
  - TF32 入力: `m16n16k8`（compute capability 8.0 以降で対応、sm_121 は満たす）
  - 5th-Gen Tensor Core（Blackwell 系譜共通）は FP8（E4M3）・FP6・FP4（NVFP4）にも対応するが（[NVIDIA Blackwell Architecture](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/)）、本イシュー（11.1a〜d）のスコープは PoC-v2-3 が既に f32/f16 で実測している範囲に合わせ f16・TF32・f32 累算に限定する。FP8/FP4 経路の採否は本設計では判断せず、将来検討事項として「6. 後続サブタスク」節に記録する。
- **NVRTC が `compute_121` を受理するか**: 未検証。PoC-v2-3 の `CudaGemm` は `CudaContext` から取得した compute capability を `--gpu-architecture=compute_XY` に反映する構成（ハードコードした sm 番号への依存を避ける設計、`docs/spec/03-poc/poc-v2-3-cuda-gemm/README.md` 実施内容 2 節）であり、この機構が `compute_121` に対しても正しく動作するかは実機での NVRTC コンパイル実行でのみ確認できる。本イシューでは実機プローブを見送った（3 節参照）。**未検証事項として #61 の着手初期に確認する。**
- **TMA（`cp.async.bulk.tensor`）が sm_121 で使えるか**: GEMM 性能改善ロードマップ（#479／#480）の Phase B 起票要否を判断するための独立プローブを #483 で実施した。詳細・記録表は 12 節「TMA（cp.async.bulk.tensor）sm_121 プローブ（#483）」を参照（未検証のまま本イシュー時点では記録待ち）。

## 3. 命令選定と根拠

### 3.1 比較する 2 方式

| 方式 | 概要 |
|------|------|
| A. WMMA C++ API（`#include <mma.h>`、`nvcuda::wmma::fragment`/`load_matrix_sync`/`mma_sync`/`store_matrix_sync`） | CUDA C++ の高レベル API。fragment 型が M/N/K・精度・レイアウトを型で表現し、ロード/ストア/積和が関数呼び出しで完結する |
| B. インライン PTX（`asm volatile` で `mma.sync.aligned.*` を直接記述） | ヘッダ非依存。PTX ISA の `mma` 命令を文字列アセンブリで直接発行する |

### 3.2 比較軸

1. **NVRTC ヘッダ問題**: NVRTC は CUDA ヘッダを同梱しない。`<mma.h>` を使うには `nvrtcCreateProgram` の呼び出し時に CUDA toolkit の include パス（`<toolkit>/include`）を渡す必要がある（[NVRTC 公式ドキュメント](https://docs.nvidia.com/cuda/nvrtc/index.html)。header 解決は `nvrtcCreateProgram` に渡したヘッダ一覧の後にサーチされる）。これは PoC-v2-3・現行 `crates/backend-cuda` が前提とする「CUDA toolkit 非搭載環境でもビルド成立する」設計（`cudarc` の動的ロード方式、`.claude/rules/deps-policy.md`）と緊張関係にある。**ビルド成立自体は toolkit 非搭載でも保たれる**（NVRTC 呼び出しはビルド時ではなく実行時のため）が、**実行時に toolkit の include パスが見つからない環境では `<mma.h>` を使うカーネルのコンパイルが失敗する**。実機側（DGX Spark）は CUDA SDK 13.0.3 が導入済みのため実行は成立する見込みだが、include パスの解決方法（環境変数 `CUDA_PATH` 由来か、既知パスの探索か）を実装時に確定する必要がある。方式 B はこの依存を持たない。
2. **記述コスト**: 方式 A は fragment 型・高レベル関数でロード/ストア/積和を表現でき、境界検査（後述 4 節）を通常の C++ 条件分岐で書ける。方式 B は PTX オペランドのレジスタ割当・データレイアウト（`mma.sync` が要求するスレッドあたりの断片配置）を手動管理する必要があり、記述・デバッグコストが高い。
3. **#187（11.1h: `mma.sync` PTX 直叩き・`ldmatrix`・`cp.async`）への発展性**: #187 は `ldmatrix`（共有メモリからレジスタへの効率的なロード）・`cp.async`（非同期コピーパイプライン）・XOR swizzle（バンクコンフリクト回避）を PTX レベルで扱う。これらは WMMA C++ API では表現できず、いずれ方式 B（PTX）への移行が必要になる。

### 3.3 判断

**#61（f16 WMMA GEMM）・#62（TF32/f32 経路）・#63（タイル最適化）では方式 A（WMMA C++ API、`<mma.h>`）を第一候補とする。** 判断理由:

- 3 軸のうち「NVRTC ヘッダ問題」は実装コスト増（include パス解決ロジックの追加）に留まり、ビルド成立可否そのものを損なわない（前述の通り実行時要求）。一方「記述コスト」「境界検査のしやすさ」は方式 A が明確に優位であり、#61〜#63 段階（初回 tensor core 実装・数値一致検証・基本最適化）では実装速度と正しさの検証しやすさを優先すべき局面である。
- #187 で PTX 直叩きへ移行する際は、方式 A で確立した fragment 構成・タイル構成・境界検査ロジックの設計知見（本メモの 4〜6 節）をそのまま引き継げる。方式 A → 方式 B の段階移行は「まず高レベル API で正しさを確立し、後で低レベル最適化を積む」という PoC-v2-3 の naive → tiled の段階と同じ考え方に沿う。
- 安全側判断: 計画時点でリスクとして挙げられていた「NVRTC で `<mma.h>` が使えない/環境依存が強い場合」に該当する事実（実機での動作不可）は未確認のため、現時点では方式 A を採用し、#61 着手時の実機検証（2 節「未検証事項」）で `<mma.h>` の実行時コンパイルが失敗することが判明した場合にのみ方式 B へ切り替える。この切替条件を #61 の受け入れ条件に含めることを推奨する（本メモはその推奨を記録するに留め、#61 の計画側で確定する）。

## 4. fragment・タイル構成

### 4.1 fragment 構成

| 精度 | fragment shape | 累算型 | 対応サブタスク |
|------|----------------|--------|----------------|
| f16 | `m16n16k16` | f32（`wmma::accumulator<..., float>`） | #61 |
| TF32 | `m16n16k8` | f32 | #62 |

- f16 の累算を f32 に固定する根拠: PoC-v2-3 の naive/tiled カーネルは既に「f16 入出力・f32 アキュムレータ」を採用している（`docs/spec/03-poc/poc-v2-3-cuda-gemm/README.md` 実施内容 2 節。理由は f16 の仮数部 10bit で K が大きい GEMM をそのままアキュムレートすると桁落ちが急速に蓄積するため）。PyTorch の f16 GEMM も内部で FP32 アキュムレートしており、比較対象と精度前提を揃える目的も PoC-v2-3 から引き継ぐ。tensor core 版でも同じ方針を維持し、既存カーネルとの精度前提の一貫性を保つ。
- TF32 の fragment shape（`m16n16k8`）は compute capability 8.0 以降の対応要件を満たす（sm_121 は満たす、2 節）。TF32 は FP32 の仮数部 23bit を仮数部 10bit（フォーマット全体では符号 1bit・指数 8bit・仮数 10bit の 19bit）に丸めて Tensor Core に投入する方式であり（詳細は 6 節「TF32 丸め特性との関係」）、f32 の入出力型を保ったまま高速化できる経路として #62 のスコープに位置づける。

### 4.2 タイル構成（候補値）

CUTLASS の階層構造（ブロックタイル → warp タイル → fragment タイル）の一般的な相場を参考に、以下を候補値とする（実測未検証、#61〜#63 での実測により確定・調整する）:

| 階層 | 候補値 | 根拠 |
|------|--------|------|
| ブロックタイル（thread block が担当する C の部分行列） | 128×128 | 手書き WMMA GEMM の一般的な出発点。PoC-v2-3 tiled 版の 32×32（共有メモリタイリングのみ）より大きく、Tensor Core 1 命令あたりの計算密度（16×16×16）を複数 warp 分束ねて occupancy を確保する |
| warp タイル（1 warp が担当する C の部分行列） | 64×64 | 128×128 ブロックタイルを 2×2 の warp グリッドで分割する一般的な構成。1 warp が複数の `m16n16k16` fragment（4×4 個）を反復してアキュムレートする |
| k タイル | f16: 16／TF32: 8 | fragment shape の K 次元に一致させ、共有メモリへの K 方向ロード単位をそのまま `mma_sync` の入力に使えるようにする |
| 共有メモリ使用量（見積もり） | A タイル 128×16（f16, 2byte）＋ B タイル 128×16（f16, 2byte）＝ 4KiB + 4KiB ≈ 8KiB 相当（ダブルバッファなし前提） | ブロックタイル 128×128・k タイル 16 の A/B 部分行列のみを保持する場合の概算。「ダブルバッファなし」は A・B 各 1 面のみを確保する構成を指し、ダブルバッファリング（#63 スコープ）を適用すればこの倍（約 16KiB）になる。SM あたり共有メモリ量（Blackwell 系は 100KiB 超級）に対し余裕があり、複数ブロックの同時常駐（occupancy）を妨げない規模と見積もる。正確な値は実装時に `nvcc`/NVRTC のレジスタ・共有メモリ使用量レポートで確定する |
| バンクコンフリクト回避 | 共有メモリタイルの行にパディングを加える（例: f16 タイルは 16 要素幅の行を 24 要素幅で確保。8 要素単位のパディング） | PoC-v2-3 tiled 版は素朴な 32×32 タイリングのみでパディング未適用。`load_matrix_sync`/`store_matrix_sync` は `ldm`（leading dimension）引数を取るが、half 型では `ldm` が 8 要素（16 バイト）の倍数であることが WMMA API の要件であり、f32/32-bank 前提の古典的な +1 パディング（17 要素幅）はそのまま half 型に転用できない（17 は 8 の倍数でない）。24 要素幅（8 の倍数）を暫定候補とし、実際のバンクコンフリクト低減効果・`ldm` 制約充足は #61/#63 実装時に実測で確認する（8 節の未検証事項に追記） |

- **#63（タイル基本最適化）との境界**: 上表は #61/#62 の初回実装で採用する初期候補値であり、レジスタブロッキング・ダブルバッファリング・ベクトル化ロードの適用は #63 のスコープとする（PoC-v2-3 README「要因分析」節が指摘する「tiled 実装が自明な最適化しか適用していない」点の解消は #63 で扱う）。

## 5. 境界検査設計（REQ-8）

- **規約**（`.claude/rules/coding-rust.md`「カーネル実装の境界検査（REQ-8）」）: 性能下限・最適化の達成を理由に、シェーダ・カーネル側の手動境界チェックを省略しない。境界検査を無効化する最適化を適用する場合は、手動境界チェックを維持したうえで行う。本規約は CPU（intrinsics）・CUDA（NVRTC/mma）・Metal（simdgroup）の全カーネルに適用される。
- **設計方針**（#61〜#63 実装時に適用すること）:
  1. **エッジタイル（M/N/K がタイル倍数でない形状）の guarded load**: ブロックタイル・warp タイルの端で、グローバルメモリの実データ範囲を超える読み出しが発生しうる（M/N/K がタイル寸法の倍数でない場合）。`load_matrix_sync` を無条件に呼ぶ前に、対象範囲がテンソルの実データ内かを判定し、範囲外のスレッド/要素はゼロ充填した共有メモリバッファから読ませる（グローバルメモリへの範囲外アクセス自体は発生させない）。
  2. **エピローグ store のガード条件**: `store_matrix_sync` で `wmma::accumulator` の内容を C 行列へ書き戻す際も同様に、書き戻し先が実データ範囲内かを判定してから store する。範囲外の fragment 要素は書き戻さない。
  3. **ベクトル化ロード・タイル端の分岐削減との両立**: #63 でベクトル化ロード（`float4`/`half2` 等）を適用する場合も、上記 1〜2 のガードは境界検査を無効化する形での省略対象にしない。ベクトル幅がタイル端で実データ幅を超える場合は、ベクトル化ロードを行単位のスカラーロードにフォールバックするか、ゼロ充填済み共有メモリ経由で読ませる設計とする。
  4. 上記は PoC-v2-3 の naive/tiled カーネル（`crates/backend-cuda/src/kernels.rs`）が既に採用している境界検査方針（グローバルメモリ範囲外アクセスを避ける条件分岐）を tensor core 版でも維持する形になる。

## 6. 数値契約

- **複合判定**: バックエンド間・cuBLAS 比較を含む数値一致は統一複合判定「相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満」（`.claude/rules/coding-rust.md`、REQ-2）を適用する。PoC-v2-3 の f32 vs `torch.matmul` 比較では、より厳しい閾値（相対 1e-4 or 絶対 1e-5）で 262,144 セル中 2 セルが僅かに外れた実績があり、tensor core 版（TF32 丸めを経由する f32 経路含む）でも同種の僅差が生じうることを想定しておく。
- **TF32 丸め特性との関係**: TF32 は FP32 の仮数部 23bit を TF32 の仮数部 10bit（フォーマット全体では符号 1bit・指数 8bit・仮数 10bit の 19bit）に丸めて Tensor Core に投入する（NVIDIA 公式仕様）。これは f32 の入出力型を保ちながら内部精度が TF32 相当に低下することを意味し、f32 tiled カーネル（フル精度の `f32::mul_add`）との比較では TF32 経路の方が誤差が大きくなる可能性がある。REQ-2 は「TF32 前提の複合指標に改定済み」（`.claude/rules/coding-rust.md`「バックエンド構成（REQ-2）」節）であり、現行の複合判定（相対 1e-3 or 絶対 1e-5）は TF32 前提を織り込み済みという位置づけである。tensor core 版 TF32 経路（#62）の実測でこの閾値内に収まるかを確認する。
- **FMA 契約との整理**: CPU 参照実装は `f32::mul_add` を用い、GPU 側は既定 FMA 契約と揃える方針（`.claude/rules/coding-rust.md`）。WMMA/mma の内部積和順序は cuBLAS 同様に非公開であり、PoC-v2-3 が観測した「積和順序差に起因する丸め誤差」（cuBLAS 比較で 0 近傍セルの相対誤差跳ね上がり）は tensor core 版でも同様に生じうる。この丸め誤差は「カーネル自体の正しさの問題ではない」（CPU 参照実装との比較は既存カーネルで 0 不一致）という PoC-v2-3 の整理を踏襲する。
- **閾値変更はスコープ外**: 数値一致閾値・ガードレール閾値の変更はユーザー承認必須（`.claude/rules/security.md`「自己修復ループ固有のガードレール」・`delegation-impl.md`「禁止事項」）。#186（TF32/f16 経路の数値一致閾値の実測再評価）は既存閾値が tensor core 実測後に不足すると判明した場合の再評価用に切り出されたサブタスクであり、本設計メモでは閾値の変更判断を行わない。

## 7. ディスパッチ規則への引き渡し事項（TASK-11.2／#66 向け）

TASK-11.2（#66）でディスパッチ規則を設計・実装する際、本メモから引き渡す事項:

- **compute capability 判定**: WMMA f16 経路は compute capability 7.0 以降、TF32 経路は 8.0 以降で有効化可能（sm_121 はいずれも満たす）。cc 7.0 未満（対象外だがフォールバック設計上の下限として記録）では tensor core 経路そのものを無効化し、既存の tiled 経路にフォールバックする。
- **形状境界**: v1 の autotune 実測知見（M=N=K=512 が境界となる傾向、`docs/spec/04-requirements.md` REQ-11「経路選択」受け入れ基準）は CubeCL 前提の参考値であり、v2 の自作ディスパッチでの再検証を要する。PoC-v2-3 の tiled 実測（512 で PyTorch 比 26.6% 達成、2048/4096 で未達）と合わせて考えると、小規模形状では tensor core 化前でも一定の相対性能を確保できている可能性があり、tensor core 経路への切替境界は「tiled で十分な形状」と「tensor core が必要な形状」を実測に基づき再設定する必要がある。
- **フォールバック条件**: (1) toolkit 非搭載・NVRTC が `<mma.h>` を解決できない環境、(2) compute capability が WMMA/TF32 の要件を満たさない環境、(3) M/N/K がタイル最小単位（fragment shape）に満たない極小形状、の 3 条件では tiled 経路（既存実装）へフォールバックする設計とすること。
- これらはディスパッチ規則の設計・実装そのものではなく、#66 が設計時に踏まえるべき前提条件の列挙に留める。

## 8. 後続サブタスク（11.1b〜h）の実装順・検証計画

- **実装順**: #61（f16 WMMA GEMM）→ #62（TF32/f32 経路）→ #63（共有メモリ・タイル基本最適化）→ #66（ディスパッチ規則）→ #186（数値一致閾値の再評価、必要な場合）→ #187（`mma.sync` PTX 直叩き・`ldmatrix`・`cp.async` パイプライン）。この順序は「まず正しさを確立し（f16→TF32/f32）、次にタイル最適化、次にディスパッチへの組み込み、最後に低レベル PTX 最適化」という段階を踏む（3.3 節の判断根拠と整合）。
- **実機依存テストの `#[ignore]` 分離**: PoC-v2-3・既存 `crates/backend-cuda` の方針を踏襲し、DGX Spark 実機（GB10）でのみ実行可能な数値一致テスト・ベンチマークは `#[ignore]` で分離し、通常 CI（GitHub ホステッド、CUDA 非搭載ランナー含む）ではビルド成立のみを検証する（`.claude/rules/ci.md`「実機依存」節）。
- **5 回計測中央値**: ベンチマークは `.claude/rules/coding-rust.md`「テスト・ベンチ」節に従い 5 回計測の中央値を採用する。PoC-v2-3 と同様、大規模形状（4096 等）で計測時間が過大な場合は計測回数の縮小を許容し、その旨をログ・レポートに明記する。
- **未検証事項の一覧**（#61 着手時に優先的に解消すべき事項）:
  1. NVRTC が `--gpu-architecture=compute_121` を受理するか（2 節）
  2. `<mma.h>` の実行時 include パス解決が DGX Spark 実機の CUDA SDK 13.0.3 環境で成立するか（3.2 節）。不成立の場合は方式 B（インライン PTX）へ切り替える（3.3 節の切替条件）
  3. 4 節の共有メモリ使用量・occupancy 候補値の実測確認
  4. TF32 経路の複合判定閾値（相対 1e-3 or 絶対 1e-5）内への収束（6 節）
  5. 4.2 節のバンクコンフリクト回避パディング（f16 タイル 24 要素幅）が `load_matrix_sync`/`store_matrix_sync` の `ldm` 制約（half 型で 8 要素単位）を満たしたうえで実際にバンクコンフリクトを低減するか

## 9. 実機検証プローブについて

- 計画（Step 2）は DGX Spark への到達可能性に応じたベストエフォートの実機 NVRTC コンパイル検証プローブを許容していた。本イシューの実行環境（サンドボックス化された git worktree、ネットワーク到達性は SSH エイリアス `local-server` 経由の実機接続を含め未確認・未実施）では実機プローブを実施しなかった。
- 未検証事項は 8 節の一覧に記録した。#61（f16 WMMA GEMM 実装）の着手初期に実機検証を行う方針は #187 本文の「NVRTC での sm_121 挙動は実装初期に実機検証する」と整合する。
- 接続情報（SSH エイリアス実体・ホスト名等）は本メモに一切記載していない（PoC-v2-3 の「接続情報は非記載」方針を踏襲、`.claude/rules/security.md`）。
- **TMA プローブ（#483）も同型の到達不能ベストエフォート**: 本節と同じ理由（サンドボックス化された git worktree からの実機到達性未確認）により、12 節「TMA（cp.async.bulk.tensor）sm_121 プローブ」の記録表は実行待ちのまま残している。

## 10. スコープ外・将来の unsafe 境界

- カーネル起動 API（`cudarc` の `unsafe fn` ラッパー）が唯一の `unsafe` 境界となる設計を、tensor core 版カーネルでも維持する。CUDA C ソース文字列（`kernels.rs` 相当）自体は Rust の `unsafe` を必要としない（NVRTC への文字列渡し・PTX ロードは `cudarc` 側の型で表現される）。実装（#61 以降）では、既存 `crates/backend-cuda/src/gemm.rs` のカーネル起動ラッパーと同様に、理由コメント付きで `unsafe` を最小化しレビュー必須とする（`.claude/rules/security.md`「unsafe」節）。
- **仕様変更が必要と判断した事項**: 現時点では発見していない。REQ-11 の受け入れ基準（明示切替 API を提供しない方針・証跡方式）と本設計は整合しており、`docs/spec/` 側の変更提案は不要と判断した。

## 11. SM120/sm_121 の機能制約（SM90/SM100 対比・CUTLASS ソース根拠）

- **目的・分担**: #485（GEMM 性能改善ツリー Phase A・親 #480 の A-5）でのイシューとして、以後の CUDA 最適化検討（Phase B/C）が SM90（Hopper）・SM100 系（データセンター Blackwell）専用技法を SM120/sm_121（DGX Spark GB10）向けの最適化候補に誤って含めないよう、CUTLASS 一次ソースの静的読解のみで確定できる制約をここに記録する。TMA・`setmaxnreg` の実機成否は本節では判断せず、A-3（#483）・A-4（#484）の受け入れ基準側で確定する（11.1 節の空欄行）。2 節の記述（技術ブログ由来の二次情報）はこの一次ソース読解で裏付けられている。
- **検証に用いた CUTLASS**: リポジトリ [NVIDIA/cutlass](https://github.com/NVIDIA/cutlass)、tag `v4.7.0`（commit `dcf215af68a2d08d305076c152a06f201728cd53`。2026-08-14 時点の最新リリースタグ）。scratchpad 内に `git clone --depth 1` 後 `git fetch --depth 1 origin tag v4.7.0` で当該タグへ checkout して検証した（読み取り専用。コード・コメントは転記せず、事実の指摘のみ `path:line` で記載する）。

### 11.1 機能対比表

| 機能 | SM90 (Hopper) | SM100 系 (DC Blackwell) | SM120/sm_121 (GB10) | CUTLASS ソース根拠（tag v4.7.0） |
|---|---|---|---|---|
| `wgmma` | 可 | —（`tcgen05` に置換） | **不可** | `include/cute/arch/mma_sm120.hpp`（全 3278 行）に `wgmma`・`tcgen05` の大小無視の文字列一致が 0 件（`grep -c -i` 実測）。同ファイル内の MMA 命令はすべて `mma.sync` 系（`grep -c "mma.sync"` で 78 件） |
| `tcgen05` / TMEM | 不可 | 可 | **不可** | `include/cute/arch/config.hpp`（起動源マクロは `CUTE_ARCH_TCGEN05_TMEM_ENABLED`。イシュー本文の `config.h` はファイル名の言い間違いで実体は `config.hpp`、内容面の主張は正しい）は同マクロを 4 箇所（115・140・180・188 行目）で定義するが、いずれの条件ブロックも起動源が `CUTLASS_ARCH_MMA_SM100A/100F/101A/101F/103A/103F_ENABLED` と `SM110A/110F_ENABLED` の組合せのみで、SM120/SM121 系マクロを起動源に含む条件ブロックは 1 件も存在しない（ファイル全 223 行を走査済み）。SM120/SM121 系条件ブロック（156〜161 行目）は `CUTE_ARCH_MMA_SM120_ENABLED`・`CUTE_ARCH_TMA_SM120_ENABLED` のみを定義する |
| cluster（実用） | 可 | 可 | **実用上不可**（1×1×1 のみ） | (1) `include/cutlass/gemm/collective/builders/` 配下の SM120 向け GEMM collective builder 6 ファイルのうち 5 ファイル（`blockwise_mma_builder.inl` を除く）が `static_assert(cute::size(ClusterShape_MNK{}) == Int<1>{}, ...)`（意訳: 「本アーキテクチャではプログラマブルなマルチキャストクラスタ不可」の主張）を課す。該当行: `sm120_mma_builder.inl:84`・`sm120_array_mma_builder.inl:87`・`sm120_sparse_mma_builder.inl:163`・`sm120_blockscaled_mma_builder.inl:104`・`sm120_blockscaled_sparse_mma_builder.inl:179`。(2) `examples/79_blackwell_geforce_gemm/` 配下の SM120 向け example 4 本（79a〜79d）は全て `using ClusterShape = Shape<_1,_1,_1>;` を使用（各ファイルの該当行を実測確認済み） |
| `mma.sync`（Ampere 形状） | 可 | 可 | **可** | `mma_sm120.hpp` の実体は `mma.sync.aligned` 系命令のみ（同上 78 件）＋本リポ `crates/backend-cuda/src/kernels_mma.rs`（#187。ファイル冒頭コメントに `mma.sync`/`ldmatrix`/`cp.async` GEMM カーネルソースと明記）の実装実績 |
| `cp.async` | 可 | 可 | **可** | 本リポ `crates/backend-cuda/src/kernels_mma.rs` の実装実績（#187。`cp.async.cg.shared.global` 命令を使用）。CUTLASS 側は本イシューでは個別検証していない（Ampere 系譜の一般的対応のため対比表に記載するが根拠は自リポ実績のみ） |
| `ldmatrix` | 可 | 可 | **可** | (1) 本リポ `crates/backend-cuda/src/kernels_mma.rs` の実装実績（#187。`ldmatrix.sync.aligned.m8n8.x4`/`.x2.trans` 命令を使用）。(2) CUTLASS 側でも `include/cute/arch/config.hpp:130-136` の条件ブロックが `CUTLASS_ARCH_MMA_SM120A_ENABLED`・`SM121A_ENABLED` を起動源に含み `CUTE_ARCH_LDSM_SM100A_ENABLED`（ldmatrix 相当）を定義する |
| TMA（`cp.async.bulk.tensor`） | 可 | 可 | **静的にはマクロ定義が存在**（実機成否は実機プローブ待ち → A-3・#483 で確定。本節では成否を断定しない） | `include/cute/arch/config.hpp:160` で SM120/SM121 系条件ブロックが `CUTE_ARCH_TMA_SM120_ENABLED` を定義している事実のみを記す。マクロ定義の存在は実機での動作成立を意味しない（コンパイル時定義と実行時動作は別事象） |
| `setmaxnreg` | 未検証 | 未検証 | **実機プローブ待ち（空欄）** → A-4（#484）で確定。プローブ実装は 13 節・`crates/backend-cuda/tests/setmaxnreg_probe_{dec,incdec}_{base,accel}_real_device.rs`（4 ファイル）に用意済み（実機実行は未了） | 本イシューでは検証していない（SM90/SM100 列も含め本イシューのスコープ外） |

### 11.2 f32/f16 標準精度向け SM120 専用 mainloop の不在

- `include/cutlass/gemm/collective/builders/` 配下の SM120 向け GEMM `CollectiveBuilder` 特殊化は 6 ファイル存在する（`sm120_mma_builder.inl`・`sm120_array_mma_builder.inl`・`sm120_sparse_mma_builder.inl`・`sm120_blockwise_mma_builder.inl`・`sm120_blockscaled_mma_builder.inl`・`sm120_blockscaled_sparse_mma_builder.inl`。各ファイルとも `struct CollectiveBuilder<...>` の特殊化定義は 1 つのみ）。このうち 4 ファイル（`mma`・`array_mma`・`sparse_mma`・`blockwise_mma`）が、入力型が f8f6f4 系（narrow-precision）要素であることを要求する `static_assert`（`is_sm10x_f8f6f4_element` 判定に基づく。該当行: `sm120_mma_builder.inl:80-81`・`sm120_array_mma_builder.inl:83-84`・`sm120_sparse_mma_builder.inl:159-160`・`sm120_blockwise_mma_builder.inl:132-133`）を課し、さらに 4 ファイル（`mma`・`array_mma`・`sparse_mma`・`blockwise_mma`）は追加で「blockscaled でない collective builder は F8F6F4 MMA のみサポートする」という趣旨の `static_assert`（該当行: `sm120_mma_builder.inl:114-115`・`sm120_array_mma_builder.inl:116-117`・`sm120_sparse_mma_builder.inl:192-194`・`sm120_blockwise_mma_builder.inl:177-178`）も課す。残り 2 ファイル（`blockscaled_mma`・`blockscaled_sparse_mma`）はファイル名・内部の `check_input_datatypes` 呼び出しが示すとおり MXFP/NVFP4 系のブロックスケール narrow-precision フォーマット専用であり、標準精度（f32/f16）は入力型として成立しない構成である。
- すなわち SM120 向け GEMM collective builder は 6 ファイルのいずれも f8f6f4／blockscaled／sparse のいずれかの narrow-precision 系に限定されており、f32/f16 のような標準精度では要件を満たせず選択されない。CUTLASS 自身も f32/f16 では SM120 専用の mainloop を持たないことが、この 6 ファイル全数の走査から確認できる（SM80 系 mainloop への依存経路そのものは本イシューでは SM80 側ソースを個別検証していないため、「SM120 専用 mainloop が narrow-precision 限定である」という否定的事実の確認に留める）。
- **帰結**: 本リポが CUDA バックエンドの tensor core 化（#61〜#63、`crates/backend-cuda`）で Ampere 世代 `mma.sync`（`m16n8k16` 等の WMMA/mma 形状、本ドキュメント 2〜4 節）を使い続ける設計判断は、CUTLASS 自身の構成（SM120 専用 collective builder が narrow-precision 限定であり、f32/f16 向けの SM120 専用 mainloop を持たない）とも整合する。

### 11.3 Phase B/C への含意

- SM90（`wgmma`・TMA 前提の warp specialization epilogue 等）・SM100 系（`tcgen05`・TMEM 系 epilogue）専用の最適化技法は、11.1 節の対比表が示すとおり SM120/sm_121 では利用不可のため、Phase B/C（本リポ CUDA 最適化検討の後続フェーズ）の最適化候補から除外する。
- TMA（`cp.async.bulk.tensor`）・`setmaxnreg` の採否は、11.1 節で「実機プローブ待ち」とした 2 行の結論が A-3（#483）・A-4（#484）で確定した後にのみ判断する。本節では静的なマクロ定義の存在のみを記録し、採否の断定は行わない。TMA の実機プローブ本体・記録表は 12 節「TMA（cp.async.bulk.tensor）sm_121 プローブ（#483）」を参照。

## 12. TMA（cp.async.bulk.tensor）sm_121 プローブ（#483）

- **位置づけ**: GEMM 性能改善トラッキング（ルート #479／Phase A 親 #480）の A-3 タスク。Phase B（TMA 前提の最適化タスク群 B-12〜B-14）を条件付き起票してよいかを、プロダクションコードへ触れる前に確定するための spike（調査・記録タスク）。#61 以降で確立した WMMA／`mma.sync` 経路（1〜10 節）とは独立した調査であり、本節はカーネル実装を追加しない。
- **CUTLASS 側の根拠**: CUTLASS では `CUTE_ARCH_TMA_SM120_ENABLED` が SM121（`"a"` サフィックス無し・`__CUDA_ARCH__ == 1210`）でも有効化される設計になっている（`include/cute/arch/config.hpp:154-158`・`include/cutlass/arch/config.h:197-204`。2026-08 時点の CUTLASS ソース調査）。CUTLASS は nvcc オフラインコンパイルの CuTe C++ DSL 経由であり、本リポジトリの NVRTC 実行時コンパイル・生 PTX インラインアセンブリ方式とは経路が異なるため、この根拠がそのまま NVRTC 経路にも当てはまるかは別途確認が必要（本プローブの目的そのもの）。
- **プローブテストの場所**: `crates/backend-cuda/tests/tma_probe_real_device.rs`（`#[ignore]` 分離。2 節「9 節」と同じく DGX Spark GB10 等 sm_121 実機必須）。
  - `tma_nvrtc_compile_probe`: mbarrier 初期化 + `cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes`（`cluster` variant）／`cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes`（`cta` variant）による 1 タイル転送の最小 inline PTX カーネル 2 variant を、`compute_121`／`compute_121a`／`compute_121f` の 3 arch × 2 variant（計 6 組み合わせ）で NVRTC コンパイルし、成否・エラーメッセージ全文を記録する（`cluster`／`cta` 両スコープを probe する理由は本ファイル該当ソースコメント参照。sm_120/121 では `CUTE_ARCH_TMA_SM120_ENABLED` パスが `shared::cta` opcode を発行する設計であるため、cluster のみの probe では「TMA は sm_121 で使えない」と誤って記録しうる）。
  - `tma_execution_probe`（cluster variant）／`tma_execution_probe_cta`（cta variant）: 各 variant について `compute_121` → `compute_121a` → `compute_121f` の順にコンパイルを試み、最初に成功した arch のみで実行する（`compute_121` 固定だと、`compute_121` が拒否され代替 arch のみ成功する環境で実行可能性を確認できないため。PR #634 codex-review 指摘対応）。64x64 f32 global テンソルから `cuTensorMapEncodeTiled` で生成した `CUtensorMap` 経由で 16x16 タイルを実転送し、ソース領域とのビット等値比較で検証する。
  - 実行コマンド: `cargo test -p backend-cuda --release -- --ignored --nocapture tma`
- **判定条件（記録表の読み方）**: 実行成否は variant（cluster／cta）ごとに独立して評価する。コンパイル成功 arch が複数あっても実行は「その variant で最初にコンパイル成功した 1 arch」のみで行うため、実行しなかった arch の実行成否列は「対象外（未選択 arch）」と記録する。cluster と cta は完全に独立した命令列（本節冒頭「CUTLASS 側の根拠」参照）のため、一方の失敗が他方の成否を意味しない。
- **実行環境（本イシューの実行結果）**: 実装セッションはサンドボックス化された git worktree であり、DGX Spark 実機への到達性（SSH 接続を含む）を確認できなかった（9 節と同型の制約）。よって以下の記録表は**実行待ち**のまま残す。**結論（B-12〜B-14 起票要否）は推測で埋めない**（実装計画 §3 Step 3「安全側フォールバック」の方針どおり）。
- **再実行セッション（2026-08-15、PR #634 の main 追従・完遂タスク）**: `docs/real-hardware-verification-env.local.md`（実ホスト名の正）・`~/.ssh/config` の該当エントリともに本セッションの worktree 環境には存在せず、実機（DGX Spark GB10）への到達性を確認できなかった（上記と同型の制約が継続）。記録表・結論欄は引き続き実行待ちのまま維持する。実機実行は到達可能な環境からの後続作業とする。

### 記録表（実行待ち）

| arch | variant | コンパイル成否 | エラーメッセージ要旨 | 実行成否 |
|------|---------|--------------|----------------------|----------|
| `compute_121` | cluster | 未実行 | — | 未実行 |
| `compute_121a` | cluster | 未実行 | — | 対象外（未選択 arch） |
| `compute_121f` | cluster | 未実行 | — | 対象外（未選択 arch） |
| `compute_121` | cta | 未実行 | — | 未実行 |
| `compute_121a` | cta | 未実行 | — | 対象外（未選択 arch） |
| `compute_121f` | cta | 未実行 | — | 対象外（未選択 arch） |

### 結論欄（実行待ち）

- B-12〜B-14 の起票要否: **未確定**。実機での `cargo test -p backend-cuda --release -- --ignored --nocapture tma` 実行後、上記記録表を実測値で更新する。判定基準は variant 別に以下のとおりとする（推測で埋めず、実測後にこの基準を機械的に適用する）。
  - **一次トリアージ（機械判定の前提）**: 「全 arch でコンパイルまたは実行が失敗」を確認した場合、下記の機械判定を適用する前に、記録したエラーメッセージ全文を (a) opcode／arch 非対応を示すもの（例: `unsupported`・`invalid instruction`・対象 opcode 名を含む NVRTC/ptxas エラー）か、(b) 本プローブ自身の構文・オペランドエラー（例: オペランド数不一致・レジスタ制約違反・`ptxas` の一般的な構文エラーで opcode 名を伴わないもの）かを目視で切り分ける。本プローブの inline PTX・`cuTensorMapEncodeTiled` 呼び出し（`&tensor_map` の `"l"` 制約渡しを含む）は本 PR 時点で実機コンパイル・実行を一度も通過しておらず（ファイル冒頭コメント参照）、構文・オペランド誤りの可能性が残るため、この切り分けを省略しない。(b) と判定される場合はプローブ自体を修正のうえ再実行し、(a) と判定できるまで「起票不要」を確定させない。
  - cluster・cta のいずれか一方でもコンパイル・実行が成功: 起票要（起票自体は `out-of-scope-tracking.md` に従いユーザー承認のうえ別途実施）。本ファイル冒頭コメント「CUTLASS 側の根拠」のとおり sm_120/121 では `CUTE_ARCH_TMA_SM120_ENABLED` パスが `shared::cta` opcode を発行する設計のため、**cta 単独の成功でも cluster の失敗は非ブロッキングとして扱う**（cta 成功のみで起票要と判定してよい）。
  - cluster・cta のいずれも全 arch でコンパイルまたは実行が失敗、かつ上記一次トリアージで全失敗理由が (a) opcode／arch 非対応と判定できた場合のみ: 起票不要。(b) プローブ自体のバグに起因する失敗が一件でも含まれる場合は「起票不要」と確定せず、プローブ修正・再実行後に再判定する。
- 実機実行手順は `docs/real-hardware-verification-env.md`（接続情報・実ホスト名は同ドキュメントの `*.local.md` 参照方式に従い本節には記載しない）。

## 13. setmaxnreg プローブ結果（#484）

- **位置づけ**: 親イシュー #480（Phase A: GEMM 最適化の前提確定・実機プローブ）の A-4。`setmaxnreg.inc/dec.sync.aligned.u32`（warp specialization レジスタ再配分。producer/consumer warp 間でレジスタ予算を非対称配分する PTX 命令）が sm_121（DGX Spark GB10）+ NVRTC（CUDA 13.0 系）で受理・実行可能かを確定させ、後続 B-3（タイル拡大時のレジスタ予算設計）の設計自由度の上限を明らかにする spike。
- **プローブ実装（PR #636 再指摘 P2 × 3・Bugbot Medium 対応で 4 ファイルへ再設計）**: `crates/backend-cuda/tests/setmaxnreg_probe_{dec,incdec}_{base,accel}_real_device.rs`（`#[ignore]` 分離。共有ヘルパーは `tests/setmaxnreg_common/mod.rs`）。`setmaxnreg.dec` 単独発行版・producer warpgroup が `setmaxnreg.dec`／consumer warpgroup が `setmaxnreg.inc` を発行する非対称版（1 ブロック 2 warpgroup・256 スレッド）の 2 カーネルを、基準 arch（`compute_121` 相当）・arch-accelerated 版（`compute_121a`）それぞれ独立のテストバイナリ（＝独立プロセス）で NVRTC コンパイル → ロード → 起動 → 同期し、成否を `SETMAXNREG_PROBE_RESULT` 形式で標準出力へ記録する。命令拒否と `libnvrtc` 不在等の toolchain 側要因を区別するため、`setmaxnreg` を含まない対照カーネルを同一 arch へ先にコンパイルしてから判定する。
  - **ファイル分割によるプロセス分離（再指摘 P2「同一 CUDA コンテキストでの連続実行」対応）**: 以前は 1 テスト関数内で基準 arch → arch-accelerated 版を同一 `CudaDevice`／CUDA コンテキストで連続実行しており、先行プローブのデバイス例外がコンテキストを汚染し後続の判定に伝播しうる構造だった。`tests/` 直下のトップレベル `.rs` ファイルはそれぞれ独立の cargo test バイナリ（＝独立プロセス）としてコンパイル・実行される cargo の既定動作を利用し、追加の自前プロセス分離機構（自己再実行ハーネス等）を実装せずに 4 ファイル（dec/incdec × base/accel）へ分割することで真のコンテキスト独立を得た。
  - **`num_regs` を実行ゲートから撤去し診断専用へ（再指摘 P2 × 2「静的 `num_regs` を実行可否ゲートに使わない」対応）**: 以前は `CU_FUNC_ATTRIBUTE_NUM_REGS`（`cudarc::driver::CudaFunction::num_regs`）から算出した CTA レジスタプール保存則等の静的判定で `launch`/`synchronize` 自体を skip していた。しかし `num_regs` は `cuModuleLoadData` 時点のドライバ JIT が確定させる静的な register/thread 割り当て値であり、単純なプローブカーネルでは `setmaxnreg` の対象値（24〜232）を大きく下回りやすく、有効な命令構成であっても実行経路が恒常的に skip され本スパイクの主目的（producer/consumer 非対称版の実際のロード・起動・同期・出力検証）を達成できていなかった。再設計後は `num_regs` を `source=diagnostic` の参考ログとしてのみ記録し（`stage=control_baseline_regs`／`stage=probe_self_regs`）、実行可否には一切使わない。ロードに成功したカーネルは常に起動・同期する。
  - **使用可否判断の一次根拠は変わらず** `nvrtc_compile`/`module_load`/`load_function`（受理）と `execute`（実行完走＋出力ビット完全一致。基準 arch 版・arch-accelerated 版とも `arch=` フィールドで区別して記録）の実測結果である。`result=corrupted`（実行完走したが出力が期待値と不一致）は引き続き `panic` させる（出力破壊は命令の受理可否の範疇を超えた危険なシグナルのため softening しない）。producer/consumer 間で実際にレジスタ予算が動的に再配分された証拠（例: `nsight-compute` 等によるレジスタ占有率の実行時プロファイル）は本プローブのスコープ外。詳細は `tests/setmaxnreg_common/mod.rs` 冒頭コメント参照。転記時は `result`・`arch` に加え `probe_self_regs`・`control_baseline_regs`（いずれも `source=diagnostic` の参考値）の実測値も併記すること。
  - **register/thread 数を構成でピン留めしていない点（既知の限界・PR #636 継続コミット時点の判断）**: dec/inc の対象値（dec 単独版 64、非対称版 24／232）が実機のベースライン静的レジスタ割り当てに対して ISA 上妥当（`.dec` 対象値 ≤ 現在値・`.inc` 対象値 ≥ 現在値）かどうかは、`--maxrregcount` 等はレジスタ数の**上限（cap）**を指定するのみで**下限（floor）**を保証しないため、構成のみでは確定できない。本プローブはこれを診断ログ（`source=diagnostic`）として記録するに留め、真の ISA 妥当性は実機実行の結果（受理・実行完走・出力一致）そのものに語らせる設計とした（構成による事前保証を諦める代わりに、静的推測に基づく恒常的な実行 skip という以前の失敗を繰り返さない）。この限界自体は実機実行後も残り得るため、実機結果の転記時は「命令自体が拒否された」（真の不可）と「値の組み合わせがこのベースラインでは不成立だった」（値の再選定が必要）を `probe_self_regs` の実測値と突き合わせて人間が判断すること。

### 13.1 実行契約

**ハング対策は静的ゲートではなく外部タイムアウトへ（Bugbot Medium「dec-only プローブが `dec_ok=false` でも常に launch する」対応）**: 旧実装は producer/consumer 非対称版のみ静的ゲートを持ち、dec 単独版は「常にゲートなしで実行」という非対称な扱いだった（旧コメント「単体（片道）プローブのため CTA レジスタプール保存則の対象外」）。再設計は両者を「診断ログのみ・ゲートなし」で統一し、ハング対策は各テストファイルを**外部タイムアウト付きで実行する運用契約**に一本化した: `timeout 120 cargo test -p backend-cuda --release --test <ファイル名> -- --ignored --nocapture`（4 ファイルそれぞれに適用。実行手順は `docs/real-hardware-verification-env.md` §4.5 に反映済み）。静的値からの予測に頼らず、実機自身に真の成否（命令拒否／ハング／実行完走）を語らせる設計とする。

- **実機実行**: **未了**（本イシューの実行環境ではローカル GPU が RTX 3060／compute capability 8.6 であり sm_121 実機ではないため、`docs/real-hardware-verification-env.md` が要求する `docs/real-hardware-verification-env.local.md`（実機ホスト名。`.gitignore` 対象・本 worktree には未配置）経由の DGX Spark GB10 接続を行っていない。9 節「実機検証プローブについて」と同じ理由で、到達できない実機の結果を推定で記載しない）。負対照実行（cc 8.6・本 worktree・2026-08-15）: `libnvrtc` 自体が本環境に未導入のため 4 ファイルとも `nvrtc_compile` 段階で `result=inconclusive`（`CudaError::NvrtcUnavailable`）として記録され、`panic` せず pass した。これは「非対応 arch での命令拒否」の実測ではなく「toolchain 不在」の実測に留まるが、4 ファイル分割後のプロセス分離・非ハング・構造化ログ出力という再設計後の実行経路自体が機能することは確認できた（一過性の負対照記録であり、恒久的な環境詳細としては記載しない）。
- **B-3 への引き渡し（プローブ実行前の fail-closed 既定）**: 実機プローブが完了し `setmaxnreg` の使用可否が確定するまで、B-3（タイル拡大時のレジスタ予算設計）は**対称レジスタ予算前提**でタイル上限を設計する（`setmaxnreg` の使用を前提にした非対称配分設計を仮定しない）。**使用可と確定した場合**、B-3 は producer/consumer 間の dealloc/alloc 量の設計自由度を本節の実測結果（`probe_self_regs`・`execute` の `result`）で更新した内容から引き継ぐ。確定自体は実機実行後の作業であり、本節はその条件付き方針のみを記す。

## 14. mma_f16 warp タイル拡大設計と __launch_bounds__ 方針（#803）

- **位置づけ**: 親イシュー #479（GEMM 性能改善ツリー）→ Phase 4 親 #789 配下の
  #803「mma_f16 warp タイル拡大の設計とレジスタ収支検証」。現行の `mma_f16`
  カーネル（`kernels_mma.rs`）は warp あたり `2x2` 命令タイル（warp タイル
  `32x16`）で、CUTLASS 標準 WarpShape `64x64` の 1/8 面積に留まり、`ldmatrix`
  フラグメントロード比が高く smem→レジスタ帯域が律速になりうる構造上の課題を
  持つ（Phase 4 診断・`docs/perf/cuda-gemm-bottleneck-diagnosis.md`）。
- **本イシューは設計・事前検証のみ**を担う。**本番カーネル定数
  （`MMA_WARP_TILES_M`/`_N`）は本イシューでは変更しない**。本番結線（採用形状
  への定数変更・`__launch_bounds__` 実付与・実機ベンチ 5 回中央値）は後続 #804
  （dependsOn: #803）のスコープであり、parity 非後退契約
  （`tests/parity_nonregression.rs`）・バックエンド間数値一致テストへの影響は
  ゼロに保たれる。
- **診断機構**: `kernels_mma.rs::mma_f16_source_with_warp_tiles(warp_tiles_m,
  warp_tiles_n, launch_bounds)`（`mma_f16_source_with_swizzle`〈#499・#782〉と
  同型の、`mma_f16_source()` に対するアンカー完全一致置換方式。アンカー不在・
  複数出現は `CudaError::InvalidKernelConfig` で fail-closed）を新設し、
  `internal-diagnostics` feature 限定で `lib.rs::diagnostics` 経由・
  `examples/mma_ptx_dump.rs` から到達可能にした。既定値
  `(MMA_WARP_TILES_M, MMA_WARP_TILES_N, None)` は `mma_f16_source()` とバイト
  一致することをユニットテストで固定しており、本番経路への影響がないことを
  機械的に担保する。
- **候補・実測結果・occupancy 導出**: `docs/perf/
  cuda-gemm-mma-warp-tile-register-budget.md` を参照（候補表: 2x2 現行・2x4
  案 A・4x2 案 B・4x4 案 C。loads/mma 比・机上レジスタ見積もり・実機 `ptxas -v`
  実測表〈本イシュー時点では実行待ち〉）。
- **`__launch_bounds__` 方針**: CUTLASS `device_kernel.h` 方式
  （`__launch_bounds__(<ブロックスレッド数>)`。`minBlocksPerMultiprocessor` は
  指定しない）を基本案とし、各候補を launch_bounds なし／あり（値 = 導出
  スレッド数）の 2 通りで比較して付与要否・付与値を確定する方針とした
  （`.maxntid` が PTX に載ることで ptxas のレジスタ割り当て前提が本番起動
  構成と一致するため）。付与値は診断関数側で「導出スレッド数と完全一致」を
  fail-closed で検査する（誤った `.maxntid` での計測を防ぐ）。実測に基づく
  最終決定は上記 perf ドキュメントへ記録する。
- **#804 への引き渡し事項**: 採用形状（spill 0 かつ loads/mma 比最小の候補）・
  `__launch_bounds__` 付与値と付与要否・swizzle 条件
  （`mma_f16_source_with_swizzle`）や `gemm_auto.rs` の warp 刻み定数
  （`MMA_WARP_M`/`_N` を候補列挙の刻みとして参照する箇所）への波及有無。

## 15. TF32 経路の生 `mma.sync`(m16n8k8) 移行（#801）

- **位置づけ**: `crates/backend-cuda/src/kernels_mma_tf32.rs`／
  `gemm_mma_tf32.rs`（`CudaMmaTf32Gemm`）。既存 TF32 本番経路
  （`CudaGemm::run_wmma_tf32` の WMMA C++ API ベース 3 段選択。11〜13 節）
  は無変更のまま並存させる独立経路であり、f16 `mma.sync`/`ldmatrix`/
  `cp.async` 経路（`kernels_mma.rs`。TASK-11.1h・#187）の構造を TF32
  精度へ移植した refactor。**本番ディスパッチ（`ops.rs`／`gemm.rs`／
  `gemm_auto.rs` の 3 段選択・JIT 特化基盤）へは結線しない**。数値一致
  回帰の実機確認・parity 非後退契約・本番採否判断は後続イシュー #802 の
  スコープ。

### 15.1 A フラグメント: ldmatrix の b16 流用と象限順序の再導出

TF32 の A タイル（16 行 x 8 列・f32 4B/要素）をビット等価な 16 行 x 16
列の b16（2B/要素）タイルとして再解釈すると、
`ldmatrix.sync.aligned.m8n8.x4.shared.b16` が読む 4 個の 8x8 b16 象限
（各象限 = 8 行 x 4 f32 列）にちょうど一致する（CUTLASS が用いる既知の
技法）。f16 `mma.sync.aligned.m16n8k16` の A オペランド（16x16 を直接
4 象限が覆う）は象限順序 **TL, BL, TR, BR**（`kernels_mma.rs::LDSM_A_FRAG`
の `a_quad_row = (lane/8)%2`・`a_quad_col = (lane/8)/2`）だが、TF32
`mma.sync.aligned.m16n8k8` の A オペランド（16x8 を b16 換算 16x16 として
異なる行/列分解で覆う）は PTX ISA の当該命令 A オペランドテーブルから
導出すると象限順序 **TL, TR, BL, BR** になる（`a0`=TL・`a1`=TR・`a2`=BL・
`a3`=BR。`row=groupID`/`groupID+8` が a0/a1 と a2/a3 を分け、
`col=tid_in_group`/`tid_in_group+4` が a0/a2 と a1/a3 を分ける分解）。

f16 版の定数（`a_quad_row = (lane/8)%2`, `a_quad_col = (lane/8)/2`）を
そのまま「列オフセットの掛け算数のみ 8→4 に変更」して移植すると、
BL/TR の位置が入れ替わったまま出力されるため誤った演算結果になる
（実装計画のレビューで検出。単純な定数流用は不可）。本実装は
`a_quad_row = (lane/8)/2`・`a_quad_col = (lane/8)%2`（f16 版から `/2`・
`%2` を入れ替え）とすることで、`ldmatrix.x4` がレーングループ
`g = lane/8`（0..3）の象限データを出力レジスタ `r_g` へ配る仕様と
組み合わせ、`a_frag[0..3] = {TL,TR,BL,BR}` を得ている
（`kernels_mma_tf32.rs::LDSM_A_FRAG` 定義部・冒頭コメント「命令選定」
参照）。

### 15.2 B フラグメント: 素の共有メモリロード

`ldmatrix .trans` は b16 粒度の転置命令であり 32bit 要素（tf32）を 2 個
の b16 に分断してしまうため使用できない。PTX ISA の m16n8k8 tf32 B
オペランド分解（`row = tid_in_group(+4)`・`col = groupID`）に従い、
row-major の共有メモリから `bs_tile[stage][k0+tid_in_group][col+group_id]`・
`bs_tile[stage][k0+tid_in_group+4][col+group_id]` を `__float_as_uint`
で直接レジスタへロードする（`LDS_B_FRAG` マクロ。`.trans` ldmatrix は
不使用）。

### 15.3 TF32 丸めの継承（#800）

`mma.sync` の tf32 オペランドは明示変換済みビットを要求し、cp.async は
生バイトコピーのため転送「中」に丸めを挟めない。`kernels_wmma_opt.rs::
CONVERT_A_STAGE_GROUP`/`CONVERT_B_STAGE_GROUP`（#800）と同一構造・同一の
正しさ論証（走査添字が LOAD マクロと完全一致することに依存する 3 点
論証）を移植し、各 compute イテレーション先頭・`cp.async.wait_group`
直後・`__syncthreads()` 前に stage の smem チャンクを 1 回だけ丸める。
変換関数は `wmma::__float_to_tf32`（`#include <mma.h>` 経由。既存カーネル
で NVRTC 構文検証実績のある経路を優先し、インライン PTX
`cvt.rna.tf32.f32` は採用しなかった）。

### 15.4 初期タイル定数

既存 TF32 opt-staged（`WMMA_TF32_STAGED_BLOCK_M/_N` = 64x64・K タイル
16・3 ステージ）と同一のブロック形状を採用し、#802 の A/B 比較を同条件に
する: `MMA_TF32_BM=64・BN=64・BK=16・STAGES=3`、warp タイル
`WARP_TILES_M=2・WARP_TILES_N=4`（warp タイル実寸 32x32・
`WARPS_M=2・WARPS_N=2`・128 スレッド）、`A_PAD=BK+4=20・B_PAD=BN+4=68`
（静的共有メモリ 28,416B）。Phase 4（タイル拡大）の起点として、これらの
定数の変更検討・実測はイシュー #802 以降へ引き継ぐ。

### 15.5 検証状態（未検証の明記）

本実装セッションの環境（CUDA driver あり・NVRTC なし）では上記カーネル
ソースは NVRTC による構文検証を一度も通過していない（`kernels_mma.rs`
冒頭コメント「検証状態」と同じ制約）。数値一致テスト（`tests/
gemm_mma_tf32.rs`・`tests/mma_tf32_vs_wmma_tf32_staged.rs`）は実装・同梱
済みだが、`#[ignore]` 実機テストは DGX Spark GB10 実機へ到達できず
未実行のまま残す。実機での最初の実行が構文検証を兼ね、数値一致の実機
確認は #802 のスコープとして引き継ぐ。

### 15.6 数値一致・parity・実機ベンチの確定状況（#802・2026-08-21 実装セッション）

イシュー #802（本節冒頭「数値一致回帰の実機確認・parity 非後退契約・本番採否判断は後続イシュー
#802 のスコープ」の引き継ぎ先）の実装セッションも、#792／#821 と同型の理由で DGX Spark GB10
実機へ到達できなかった（実行環境には対象外 GPU〈NVIDIA GeForce RTX 3060。sm_121 ではない〉のみ
存在し、`docs/real-hardware-verification-env.local.md` も未配置）。したがって §15.5 の「未検証の
明記」は本セッション終了時点でも解消していない: `#[ignore]` 実機テスト（`tests/gemm_mma_tf32.rs`・
`tests/mma_tf32_vs_wmma_tf32_staged.rs`）は未実行のまま、`docs/perf/cuda-parity-baseline.md` への
ベースライン追記もなし、本番結線（`gemm.rs::run_wmma_tf32` への `mma_tf32` 追加）も未実施。

本セッションで実施したのは、実機到達可能セッションで即座に A/B 計測へ進めるための準備のみ:
`crates/backend-cuda/examples/cuda_floor_bench.rs` に `measure_mma_tf32`（既存 4 経路と同一の
launch-only 計測境界）を追加し、`mma_tf32` を `wmma_tf32` との比較用**参考列**として出力する
（f32 候補下限の算出ロジック `best_f32` には組み込まない）。詳細な再開手順・記録テンプレは
`docs/perf/cuda-gemm-mma-tf32-ab.md` を参照。

### 15.7 採否判断の確定（#839・凍結。2026-08-22）

§15.6 が引き継いだ実機未到達の状態は #838 実装セッションで解消し、DGX Spark GB10 実機で数値
一致・parity・A/B ベンチの計測を完了させたが、**数値一致 6 本中 4 本 FAIL**（TF32 精度差では
説明できない機能欠陥。最小形状 `m=16 n=8 k=8` で `fail_count=128/128`）という結果になった。
これを受け #839「mma_tf32 の採否判断と本番ディスパッチ結線」は **不採用（凍結）** と確定した:
機能欠陥（correctness bug）が未解消のまま、数値的に誤った出力を返す経路を `gemm.rs`／
`gemm_auto.rs`／`ops.rs` の本番ディスパッチへ結線することはしない（現状の非結線を維持）。
再評価は (a) カーネルの機能欠陥修正、(b) 実機での数値一致 6 本 pass・`ParityPath::MmaTf32`
初回 parity 記録、(c) `cuda_floor_bench` 再計測での採用条件充足、の 3 条件が揃った後に行う。
実測値・判断根拠・再評価条件の詳細は `docs/perf/cuda-gemm-mma-tf32-ab.md` §2〜§5.1 を正とし、
本節では二重管理しない。

## 16. mma_f16 ブロックタイル拡大・ステージ数増（#804 → #840 実機 A/B → #842 採否確定）

**採否判断（#842・確定）: 不採用（現行 `MMA_BM=64`/`MMA_BN=128`/
`MMA_STAGES=3` を維持）**。#840 で DGX Spark GB10 実機 A/B を完了し、
4 候補すべてが不採用（`bt64x128_s4`／`bt128x128_s3_wt2x4` は数値一致
FAIL、`bt128x256_s3_wt4x4` は起動時リソース超過、`bt128x256_s4` は
机上除外）と判明した（詳細実測は `docs/perf/
cuda-gemm-mma-block-tile-stages.md` §4・§4.1・§5）。#842 は実機へ
再到達できず（本 worktree に `ptxas`/`nvcc` 不在・
`real-hardware-verification-env.local.md` 未配置）、原因調査は bench
診断出力の拡張（mismatch 件数・最大誤差・初回不一致座標）と机上検証に
留まったが（同 doc §7）、**#840 時点の実機不採用結果自体は確定済みの
事実であり、これを覆す新たな実機実測が得られない限り不採用判断は
変わらない**。よって #842 では本番カーネル定数・`swizzle.rs`・
`gemm_auto.rs` を変更せず不採用判断を確定記録する。以下は #804 時点
（実機未到達）の記録:

- **位置づけ**: 親イシュー #479（GEMM 性能改善ツリー）→ Phase 4 親 #789 配下の #804
  「perf(backend-cuda): mma_f16 ブロックタイル拡大とステージ数増」。§14（#803）の warp タイル
  拡大候補と組み合わせ、ブロックタイル（`MMA_BM`/`MMA_BN`）・`cp.async` パイプライン段数
  （`MMA_STAGES`）を拡大してデータ再利用と Tensor Core 発行密度を上げる狙い。
- **実機到達不能（Step F フォールバック）**: 本イシューの実装セッションでも #803（§14）・#801
  （§15.6）と同型の制約により DGX Spark GB10 実機へ到達できず、加えて本 worktree のローカル
  環境にも `ptxas`/`nvrtc`（CUDA toolkit 本体）が存在しなかった（`libcuda.so.1` の driver stub
  のみ）。したがって**本番カーネル定数（`MMA_BM`/`MMA_BN`/`MMA_STAGES`）・
  `swizzle.rs::SWIZZLE_APPLY_MIN_M_BLOCKS`/`_N_BLOCKS`・`gemm_auto.rs` の静的 SMEM 予算 assert は
  一切変更していない**。診断機構の拡張・机上候補表の記録のみを本イシューの成果物とした
  （実装計画 Step F）。
- **診断機構**: `kernels_mma.rs::mma_f16_source_with_block_tile(bm, bn, bk, stages, warp_tiles_m,
  warp_tiles_n, launch_bounds, optin_budget_bytes)`（`mma_f16_source_with_warp_tiles`〈#803・#822〉
  と同型のアンカー完全一致置換方式を `BM`/`BN`/`BK`/`STAGES`/`A_PAD`/`B_PAD` へ拡張）を新設し、
  `internal-diagnostics` feature 限定で `lib.rs::diagnostics` 経由・`examples/mma_ptx_dump.rs` から
  到達可能にした。共有メモリ予算は呼び出し元供給の `optin_budget_bytes`（デバイス実測値。
  `kernels_wmma_opt.rs` と同じ方針）に対し、静的 48KiB 以下ならそのまま、超過・opt-in 予算以下
  なら `extern __shared__` 変換（多次元添字構文を保つ `typedef` 配列型ポインタ方式）、opt-in 予算
  超過なら机上除外エラーの 3 分岐で判定する。既定値 `(MMA_BM, MMA_BN, MMA_BK, MMA_STAGES,
  MMA_WARP_TILES_M, MMA_WARP_TILES_N, None, MMA_SHARED_MEM_BYTES)` は `mma_f16_source()` とバイト
  一致することをユニットテストで固定しており、本番経路への影響がないことを機械的に担保する。
  **`extern __shared__` 変換経路は `nvrtc`/`ptxas` 実機での構文検証を一度も通過していない**。
- **候補・机上見積もり**: `docs/perf/cuda-gemm-mma-block-tile-stages.md` を参照（候補: ステージ増
  のみ `bt64x128_s4`〈55,296B〉・タイル拡大 `bt128x128_s3_wt2x4`〈56,832B〉・タイル拡大+
  `bt128x256_s3_wt4x4`〈81,408B〉。いずれも静的 48KiB を超え GB10 実測 opt-in 上限 101,376B 以下。
  `bt128x256_s4`〈108,544B〉は opt-in 上限超過のため机上除外）。実機 `ptxas -v` 実測表は本イシュー
  時点では実行待ち。
- **#804 の残作業（当初記録。#840/#842 で実測・判断は完了、残るのは原因調査のみ）**:
  `docs/perf/cuda-gemm-mma-block-tile-stages.md` §6・§7・§7.1 を参照（数値不一致 2 候補の実機
  原因調査〈`compute-sanitizer` 等の実行時観測が必要〉のみが未消化として残る。実測・採用構成決定・
  動的 SMEM opt-in の起動側結線・swizzle/`gemm_auto.rs` 追従は「不採用確定」により対象外になった）。

## 17. TF32 mma.sync タイル拡大（#806 → #841 実機 A/B → #842 判断材料なしのまま凍結継続）

**採否判断（#842・確定）: 不採用（現行 `MMA_TF32_BM=64`/`MMA_TF32_BN=64`/`MMA_TF32_BK=16`/
`MMA_TF32_STAGES=3` を維持）。判断材料なし（`CudaMmaTf32Gemm` 自体の既知 correctness bug〈#839〉
未解消のため全実測が参考値）につき凍結継続**。#841 で DGX Spark GB10 実機 A/B を完了したが、
全候補（比較基準行含む）が CPU 参照値との数値一致 FAIL（`parity_cpu=false`。§15.7 の #839 凍結
判断と同一原因）のため、TFLOPS 実測値は採否判断に使用不可の参考値に留まる（詳細は `docs/perf/
cuda-gemm-mma-tf32-block-tile.md` §7・§7.1・§9）。#842 は #839（`CudaMmaTf32Gemm` の
correctness bug 修正）とは独立に本イシューのスコープを閉じ、bug 修正 → 再計測 → 採否判断の順序が
必要という結論を再確認した（`docs/perf/cuda-gemm-mma-tf32-block-tile.md` §9「再評価条件」）。以下は
#806 時点（実機未到達）の記録:

- **位置づけ**: 親イシュー #479（GEMM 性能改善ツリー）→ Phase 4 親 #789 配下の #806
  「perf(backend-cuda): TF32 タイル拡大（mma.sync 化後）」。§16（#804）の f16 ブロックタイル拡大
  手法を、生 `mma.sync`(m16n8k8) TF32 経路（§15・#801→PR #823。`CudaMmaTf32Gemm`）へ展開する。
  現行 TF32 ブロックタイル（64x64x16・S3・28,416B）は f16 経路（64x128）よりさらに小さく、
  M=N=K=4096 で対 PyTorch 比 52.0% に留まる（§15.6・`docs/perf/gemm-optimization-baseline.md`）。
- **実機到達不能（Step F フォールバック）**: 本イシューの実装セッションも §15.6・§16 と同型の
  制約（本 worktree に `ptxas`/`nvcc` 不在・`docs/real-hardware-verification-env.local.md` 未配置）
  により DGX Spark GB10 実機へ到達できなかった。したがって**本番カーネル定数（`MMA_TF32_BM`/
  `MMA_TF32_BN`/`MMA_TF32_BK`/`MMA_TF32_STAGES` 等）・`gemm_mma_tf32.rs` の起動結線は一切変更して
  いない**。診断機構の新設・机上候補表の記録のみを本イシューの成果物とした（実装計画 Step F）。
  `CudaMmaTf32Gemm` は本番ディスパッチ非結線のままであり（§15 冒頭「位置づけ」参照）、本イシューの
  タイル拡大もこの非結線 API・診断経路上でのみ行う。
- **診断機構**: `kernels_mma_tf32.rs::mma_tf32_source_with_block_tile(bm, bn, bk, stages,
  warp_tiles_m, warp_tiles_n, launch_bounds, optin_budget_bytes)`（`kernels_mma.rs::
  mma_f16_source_with_block_tile`〈§16・#804〉と同型のアンカー完全一致置換方式を TF32 の
  `#define` 名前空間〈`MMA_TF32_*` 接頭辞〉へ適用）を新設し、`internal-diagnostics` feature 限定で
  `lib.rs::diagnostics` 経由・`examples/mma_tf32_ptx_dump.rs`（TF32 専用の新規 example。f16 用
  `mma_ptx_dump.rs` を直接拡張せず分離した理由は同ファイル冒頭コメント参照）から到達可能にした。
  f16 版との差分（cp.async 転送粒度が f32 4 要素/16B・`A_PAD`/`B_PAD` が `BK+4`/`BN+4`・SMEM 予算式
  の乗数が 4B/要素）は `docs/perf/cuda-gemm-mma-tf32-block-tile.md` §5 を参照。既定値は
  `mma_tf32_source()` とバイト一致することをユニットテストで固定しており、本番経路
  （`gemm_mma_tf32.rs`）への影響がないことを機械的に担保する。**`extern __shared__` 変換経路は
  `nvrtc`/`ptxas` 実機での構文検証を一度も通過していない**。
- **候補・机上見積もり**: `docs/perf/cuda-gemm-mma-tf32-block-tile.md` §4 を参照（候補: ステージ増
  のみ〈37,888B〉・M 拡大〈43,776B〉・N 拡大〈40,704B〉・両拡大〈56,064B〉・両拡大+ステージ増
  〈74,752B〉・BK 拡大〈53,760B〉。両拡大以降は静的 48KiB を超え opt-in 経路。全候補が GB10 実測
  opt-in 上限 101,376B 以下）。実機 `ptxas -v` 実測表は本イシュー時点では実行待ち。
- **#806 の残作業（当初記録。#841/#842 で実測・実機実行は完了、残るのは #839 の bug 修正待ち）**:
  `docs/perf/cuda-gemm-mma-tf32-block-tile.md` §7〜§9 を参照（spill 実測・数値一致 `#[ignore]`
  テスト・4096/2048 ベンチはいずれも #841 で完了済み。採用構成決定・動的 SMEM opt-in の起動側結線は
  `CudaMmaTf32Gemm` の correctness bug〈#839〉修正・再計測後に持ち越し）。TF32 mma.sync 経路自体の
  本番 3 段選択への結線・採否判断は #802 のスコープのまま変わらない。

## 18. クロスタイル先読み・XOR swizzle・StreamK の要否判断（#812）

イシュー #812「perf(backend-cuda): クロスタイル先読み・XOR swizzle・StreamK の要否判断」。GEMM OSS 比較
ギャップ改修ツリー #785 → Phase 5 親 #790 配下の CUDA 残候補 3 点。§14（#803）・§16（#804）と同じ制約
（`docs/real-hardware-verification-env.local.md`・`CUDA_NODE` 不在）により DGX Spark GB10 実機へ到達
できず、実測は全て「実行待ち」のまま。3 候補いずれも本イシューでは `crates/backend-cuda/src/` の
本番カーネル定数・実装（`gemm*.rs`・`swizzle.rs`・`crates/backend-cuda/tests/`）を変更しない（判断記録
のみ）。ただし `kernels_mma.rs` 冒頭モジュールドキュメントコメント（`//!`。クロスタイル先読み・XOR
swizzle 節への本イシュー判断の参照追記）は本イシュー内で更新しており、既存ソースの設計コメントのみの
更新である（本番カーネル定数・実装は不変）。

| 候補 | 判断 | 根拠 | 詳細 |
|------|------|------|------|
| クロスタイル先読み | 保留 | wait/sync 再構成の同期バグリスクが NVRTC 構文検証不能な環境では許容できない（露出比率は `K_STEPS=2` で 50% と大きいが、露出量ではなくリスク起点の判断）。より安価な代替（`MMA_BK` 拡大による `K_STEPS` 増）を優先候補として提示 | `docs/perf/cuda-gemm-mma-ldmatrix-double-buffer.md`「#812 追加判断」節 |
| XOR swizzle | 不採用（保留） | バンクコンフリクト残存の実測（既存基準）は実行待ちのまま。SMEM フットプリント差分（パディングでステージあたり `+1,536B`、`STAGES=4` が静的上限ぴったり適合〈49,152B〉から動的 SMEM opt-in 必須〈55,296B〉へ後退）を第 2 の再評価トリガーとして追加 | `docs/perf/cuda-gemm-mma-bank-conflict.md`「#812 追加判断」節 |
| StreamK | 不採用（保留） | 主要ワークロード（M=N=K=4096）では tail effect 解消の主効果が小さいが（quantization loss 約 3.0%）中間サイズ（2048・1024）では非小さい。加えて fixup がアキュムレート順序を変えるため、既存の bit 一致論拠（実測不要の十分条件）は使えなくなる。ただし `MmaF16` 経路は既に統一複合判定による非後退契約（`assert_no_parity_regression`。bit 一致は合否条件ではない）で運用済みのため、順序変更が既存 tolerance・fixture のまま非後退契約を通るかどうかは実機実測でしか確定できない。この未確定性（実機未到達で判定不能）そのものが不採用（保留）の根拠であり、通らなかった場合に限り fixture 再生成はユーザー承認必須事項となる | `docs/cuda-streamk-decision.md`（新規） |

3 候補とも「不採用」ではなく「保留（再評価条件付き）」である点が共通する: いずれも実機実測・追加承認が
得られれば再検討しうる余地を残し、`.claude/rules/out-of-scope-tracking.md` の方針に沿って各 doc に再評価
条件を明記した（採用判断が確定した場合の実装そのものは別イシューへ切り出す）。

## 参考文献

- [Analyzing Nvidia GB10's GPU — Chester Lam](https://chipsandcheese.com/p/analyzing-nvidia-gb10s-gpu)（SM12x の `mma.sync` 系譜、`tcgen05`/`wgmma` 非対応の根拠）
- [Day 3: DGX Spark Unpacked. GB10, Unified Memory, sm_121, and NVFP4 — Kubesimplify](https://blog.kubesimplify.com/day-3-the-dgx-spark-unpacked-gb10-unified-memory-sm-121-and-the-one-reason-this-hardware-exists)
- [NVIDIA Blackwell Architecture 公式ページ](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/)（5th-Gen Tensor Core の対応精度）
- CUTLASS（`Fandhe-AI` 外部リポジトリ調査。`include/cute/arch/config.hpp`・`include/cutlass/arch/config.h`）: SM121 での `CUTE_ARCH_TMA_SM120_ENABLED` 有効化根拠（#483）
- [NVRTC 13.3 公式ドキュメント](https://docs.nvidia.com/cuda/nvrtc/index.html)（ヘッダ解決の仕組み、`nvrtcCreateProgram` への渡し方）
- [NVIDIA/cutlass](https://github.com/NVIDIA/cutlass)（tag `v4.7.0`、commit `dcf215af68a2d08d305076c152a06f201728cd53`。11 節の一次ソース根拠。BSD-3-Clause ライセンス、コード・コメントの転記は行わず事実の指摘のみを記載）
- `docs/spec/03-poc/poc-v2-3-cuda-gemm/README.md`（CUDA tiled GEMM 実測・tensor core 化の段階見積もり）
- `docs/spec/04-requirements.md`（REQ-2・REQ-8・REQ-11）
