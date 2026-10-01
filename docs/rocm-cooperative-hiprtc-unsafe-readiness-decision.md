# cooperative 起動・HIPRTC 移植の unsafe 承認整理（ROCm readiness 3。#2127）

イシュー #2127「cooperative 起動・HIPRTC 移植の unsafe 承認整理（ROCm readiness 3）」に対応する設計記録。親: `docs/backend-abstraction-amd-readiness-decision.md`（#1340）§6 起票案 3。

## 1. 状態・前提

| 項目 | 内容 |
|------|------|
| 位置づけ | **設計記録（草案）のみ。承認済みではない**。`unsafe` の最終承認は実装イシューで security-auditor 監査とユーザー承認を経て行う（`.claude/rules/security.md`・`.claude/rules/coding-rust.md`） |
| 変更範囲 | `.md` のみ。`crates/**`・`Cargo.toml`・`Cargo.lock`・依存・tolerance・ガードレール閾値は変更しない。新しい `unsafe` は導入しない |
| 対象範囲 | ROCm/HIP は REQ-2 の対象外のまま（`docs/backend-matrix.md` §3.4）。仕様変更が要る場合は fandhe-ai-spec 側への提案とする |
| 基準 HEAD | `7df91509`（2026-09-30）。`file_path:line` は同コミット基準。ずれた場合は近傍コミットで再確認する |
| 確度の表記 | 「確認済み」= `.claude/skills/amd-rocm/references/hip/*.md`・`samples/`・cudarc ソース・本リポのコードで確認できたもの。「要出典確認」= 上記の範囲で確認できず、実装イシューで公式 ROCm ドキュメントまたは実機で確認すべきもの（断定しない） |

## 2. 背景・イシュー記載との違い

- #1340 §6 起票案 3 は cooperative 起動入口の `unsafe` 採用承認を後続事項として残している。本書はその材料（制約・HIPRTC 差異・SAFETY 根拠案・配置図）を先に整理する。
- **イシュー本文は「既存 CUDA の cooperative 起動の unsafe コメント」を参考にする前提だが、そのコメントは実在しない**。HEAD `7df91509` で `grep -rn -i cooperative crates` は 0 件。Stream-K の fixup も別カーネルの通常起動で行っている（`docs/cuda-streamk-decision.md`）。よって参考元を次に差し替える。
  - (a) 通常起動 `launch_builder(...).launch(cfg)` に付く SAFETY コメント（代表例 `crates/backend-cuda/src/elementwise.rs:252-266`）
  - (b) `cudarc =0.19.8` の `launch_cooperative`（`src/driver/safe/launch.rs:243`、`pub unsafe fn`）
  - (c) NVRTC 事前プローブの SAFETY コメント（`crates/backend-cuda/src/nvrtc.rs:3942-3950` 付近）
- イシュー本文が挙げる `hipew` 等の外部クレートは参照資料にとどめ、採用を前提にしない（§6.3）。

## 3. cooperative 起動の制約一覧（AC-1）

| # | 制約 | CUDA 側（cudarc 0.19.8 / 現行） | HIP 側 | 確度 |
|---|------|------|------|------|
| C1 | デバイス属性で対応可否を fail-closed 検査する | `CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH = 95`（`src/driver/sys/mod.rs:1179`）を `CudaContext::attribute`（`src/driver/safe/core.rs:362`）で取得 | 対応する device attribute 名は skill 参照範囲に記載なし | **要出典確認**（属性名・取得 API） |
| C2 | grid の総 block 数 ≤ 1 CU あたり常駐 block 数 × CU 数 | `CudaFunction::occupancy_max_active_blocks_per_multiprocessor`（`core.rs:2278`）と SM 数属性 | `hipOccupancyMaxActiveBlocksPerMultiprocessor`／module 版 `hipModuleOccupancyMaxActiveBlocksPerMultiprocessor`（`occupancy.md` 表） | 関数名は確認済み。cooperative 起動の上限が厳密にこの式で与えられるかは **要出典確認** |
| C3 | grid 全体同期（`grid_group`）は cooperative 起動でのみ有効。`<<<>>>` 相当の通常起動では使えない | `launch_cooperative`（`launch.rs:243`） | `grid_group` は `hipLaunchCooperativeKernel` を要する（`cooperative-groups.md` 表・Notes） | 確認済み |
| C4 | module API 経由（HIPRTC で得たコードオブジェクトを `hipModuleLoadData` したもの）の起動入口 | `launch_cooperative`（`CUfunction` 起動） | `hipModuleLaunchCooperativeKernel`（`cooperative-groups.md` 表）。シグネチャ詳細は記載なし | 関数名は確認済み・シグネチャは **要出典確認** |
| C5 | 通常起動への黙示フォールバック禁止（同期意味論が異なり正しさが壊れる） | 現行コードに該当経路なし | 同左（#1340 §4 方針 3 を再掲） | 設計要件 |
| C6 | マルチデバイス版は対象外 | — | `hipLaunchCooperativeKernelMultiDevice`（`cooperative-groups.md` 表）は採用しない | 設計判断 |
| C7 | wave 幅（RDNA 32／CDNA 64）がレーン導出・シャッフルに影響 | 32 固定（#1340 §2） | `warpSize`（`cpp-language-extensions.md:22`）。`__shfl_xor()` はマスク引数なし（同 `:27`） | 確認済み（#1340 参照） |
| C8 | 動的 shared memory 量・block 次元が occupancy（C2）の入力になる | `launch_cooperative` の LaunchConfig | `hipOccupancyAvailableDynamicSMemPerBlock` 等（`occupancy.md`） | 確認済み（関数名） |
| C9 | 総 work-item `gridDim × blockDim` は 2^32 未満 | — | `occupancy.md` Notes | 確認済み |

## 4. HIPRTC と NVRTC の互換性表（AC-2）

| 観点 | NVRTC（現行。`crates/backend-cuda/src/nvrtc.rs:3941` の `compile_ptx`） | HIPRTC | 移植時の扱い | 確度 |
|------|------|------|------|------|
| プログラム生成 | `nvrtcCreateProgram` 相当（cudarc 経由） | `hiprtcCreateProgram(&prog, src, name, numHeaders, headers, headerNames)`（`hiprtc.md`） | ソースは静的テンプレートと検証済み数値のみから組む（A03） | 確認済み |
| コンパイル | `compile_ptx_with_opts` | `hiprtcCompileProgram(prog, n, opts)` | 戻り値を必ず検査 | 確認済み |
| 成果物 | PTX（`Ptx`）。module ロードで JIT | `hiprtcGetCodeSize`／`hiprtcGetCode` = 対象 ISA のコードオブジェクト（`hiprtc.md`） | 成果物種別が違うためキャッシュ形式を分ける | 確認済み |
| ログ | NVRTC ログ | `hiprtcGetProgramLogSize`／`hiprtcGetProgramLog`。成功時も警告確認が推奨（`samples/hiprtc-runtime-compilation.md` Notes） | 成功時もログを取得・記録 | 確認済み |
| 名前解決 | `extern "C"` カーネルを名前直引き | `extern "C"` なら `hipModuleGetFunction` で直引き可（同 sample）。C++ の場合は `hiprtcAddNameExpression`／`hiprtcGetLoweredName` | 現行同様 `extern "C"` ＋ `&'static str` 名に統一 | 確認済み |
| arch 指定 | `CompileOptions::arch: Option<&'static str>`（`compute_XY`。cudarc `src/nvrtc/safe.rs:240`） | `--gpu-architecture=gfxNNN[:feature±]`（`hiprtc.md`）。sample はデバイスの `gcnArchName` を渡す | デバイス照会値から組む。未指定だと default が実機と不一致になりうる（sample Notes） | 確認済み |
| 固有フラグ | — | `-fgpu-rdc`（bitcode 出力）・`-mcumode`（RDNA の CU mode。既定は WGP mode） | 既定（rdc なし・既定 mode）から始め、採用は性能実測後 | 確認済み |
| include 解決 | `CUDA_INCLUDE_PATH` ＋既知パスで順に再試行（`nvrtc.rs:3967-3982`） | ヘッダは `hiprtcCreateProgram` の headers 配列、または include オプション | 環境変数由来パスをソースに連結しない（A03）。探索パス方針は実装時に決める | include オプションの仕様は **要出典確認** |
| 定義 macro | `__CUDACC_RTC__` 等 | HIPRTC 側の定義 macro（`__HIPCC_RTC__` 等）は skill 参照範囲に記載なし | カーネル文字列の先頭にバックエンド別プレフィクスで吸収（#1340 §3 のマクロ案と同方針） | **要出典確認** |
| `warpSize` | 32 固定コード | 実行時値（RDNA 32・CDNA 64。`cpp-language-extensions.md:22`） | レンダリング時 `#define` 注入（#1340） | 確認済み |
| pragma | `#pragma unroll` 等 | 互換性は skill 参照範囲に記載なし | 移植時にカーネルごとに実測 | **要出典確認** |
| 最適化レベル | 既定値任せ（明示なし） | 既定値は記載なし | 実装時に明示固定するか既定に任せるかを決める | **要出典確認** |
| FMA 契約（最重要） | `fmad` は明示せず NVRTC 既定（`nvrtc.rs:3935-3940` のコメント）。CPU 参照 `f32::mul_add` との丸め統一がこの既定を前提（`coding-rust.md`） | HIP-Clang の FP contraction 既定値は skill 参照範囲に記載なし | **本書では判定しない**。実装イシューで実機の bit 一致実測を経て統一可否を判断し、統一できない場合の扱いはユーザー承認事項とする。tolerance は緩めない | **要出典確認**（実測必須） |
| エラー | 型付き `CudaError::NvrtcUnavailable` 等 | `HIPRTC_SUCCESS`／`HIPRTC_ERROR_COMPILATION` 等。`hiprtcGetErrorString`（`hiprtc.md`） | 型付きエラー化。本番経路で `unwrap`／`expect` を使わない | 確認済み |
| キャッシュキー | `compile_flags` と `nvrtc_version` を含む | HIPRTC 版＋gfx target 文字列＋オプション列を含める | 版・target 欠落で旧成果物を再利用しない | 設計要件 |
| ライブラリ事前プローブ | `is_culib_present()`（dlopen ベース） | `libhiprtc`／`libamdhip64` の dlopen 可否 | 非搭載環境でもビルドが成立し実行時に型付きエラーを返す構成を踏襲 | 設計要件 |

## 5. unsafe FFI の分類と HIP 版の SAFETY 根拠案（AC-3）

cudarc は多くの FFI を safe wrapper で覆っている。HIP には許容依存の対応クレートがないため、同等の wrapper は自前で書くことになる。**「型安全性が CUDA と同等」と言えるのは、自前の safe wrapper 層が下表の不変条件を型・検証で担保した場合に限る**（無条件の主張はしない）。

| 分類 | CUDA 現行（cudarc／本リポの `unsafe`） | HIP で必要になる `unsafe` | 担保すべき不変条件 |
|------|------|------|------|
| U1 ライブラリ存在プローブ・dlopen | `nvrtc.rs:3942` の `unsafe { is_culib_present() }` | 手書き FFI の dlopen／シンボル解決 | 解決した関数ポインタの型が宣言と一致・非存在時は型付きエラー・ハンドルの寿命内でのみ呼ぶ |
| U2 program 生成・コンパイル | cudarc が safe 化 | `hiprtcCreateProgram`／`hiprtcCompileProgram` | C 文字列は NUL 終端・呼び出し中に生存・オプション配列の長さと個数が一致 |
| U3 コード取得 | cudarc が safe 化 | `hiprtcGetCodeSize`→確保→`hiprtcGetCode` | 取得サイズどおりのバッファ確保・サイズ 0／途中切れの検査 |
| U4 module ロード・関数 lookup | `module_cache.rs` 経由（cudarc safe） | `hipModuleLoadData`／`hipModuleGetFunction` | コードオブジェクトの寿命＞module の寿命・関数名は `&'static str` の固定名・ロード失敗を型付きエラーへ |
| U5 通常起動 | `elementwise.rs:252-266` などの `unsafe { launch_builder(..).launch(cfg) }` | `hipModuleLaunchKernel` | 引数ポインタ配列の型・順序がカーネルシグネチャと一致・バッファ長と 1:1・grid は `div_ceil` で包含・カーネル内手動境界チェック維持（REQ-8） |
| U6 cooperative 起動 | `launch_cooperative`（`unsafe fn`。Safety 節は「`launch()` を見よ」のみ） | `hipModuleLaunchCooperativeKernel` | U5 の全条件＋C1 の属性検査通過＋C2 の grid ≤ 上限＋通常起動へのフォールバック禁止 |
| U7 handle 破棄 | cudarc の Drop | `hiprtcDestroyProgram`／`hipModuleUnload` | 二重解放防止・module 破棄は実行中カーネルの完了後（stream 同期済み）・Drop 順序（関数ハンドル→module） |

### SAFETY コメント草案（コードへは入れない。草案であり承認済みではない）

**U5 通常起動**（`elementwise.rs:252-266` の書き方を踏襲）

> SAFETY: カーネル引数は検証済みの numel と 1:1 対応するデバイスバッファ長・値であり、引数の型・順序は kernel シグネチャと一致する。カーネル内の手動境界チェック（REQ-8）と、`div_ceil` で構築した grid により OOB 読み書きは起きない。

**U6 cooperative 起動**

> SAFETY: U5 の条件に加え、(1) この device で cooperative launch 対応可否の属性検査を通過済み、(2) grid の総 block 数は occupancy 上限 × CU 数以下に検証済み、(3) 通常起動へは落とさない（未対応・超過は型付きエラーで返す）。いずれも起動ラッパー内で検査し、呼び出し側へ前提を残さない。

**U2 HIPRTC program 生成**

> SAFETY: ソース・名前・オプションはいずれも NUL 終端の `CString` で、呼び出し中に生存する。オプション配列の長さは引数の個数と一致する。ソースは静的テンプレートと検証済みの数値・enum からのみ組み立てる。

最終承認は実装イシューで security-auditor 監査とユーザー承認を経る。

## 6. 将来実装時の配置図（AC-4）

### 6.1 レイアウト案

クレート名は仮称。新設自体にユーザー承認が要る（`CLAUDE.md` のクレート構成・`deps-policy.md`）。

```
crates/tensor-core/src/device.rs     DeviceInfo（warp_width 等。#1340 起票案 1）
        |  経路選択インターフェースのみ（LaunchKind::{Normal,Persistent,Cooperative}。#1340 §4）
        v
crates/backend-rocm/   (仮称)
  src/sys.rs          手書き FFI 宣言・dlopen                      ... U1
  src/hiprtc.rs       compile_code_object（nvrtc.rs の対応物）      ... U2, U3
  src/module_cache.rs module ロード・関数 lookup・キャッシュ         ... U4
  src/launch.rs       LaunchKind の入口（通常・cooperative）         ... U5, U6
  src/device.rs       属性取得（warp_width・cooperative 可否）
  src/error.rs        型付きエラー
```

### 6.2 対応表

| 新規ファイル | 対応する既存 CUDA 側 |
|------|------|
| `hiprtc.rs` | `crates/backend-cuda/src/nvrtc.rs`（`compile_ptx`） |
| `module_cache.rs` | `crates/backend-cuda/src/module_cache.rs` |
| `launch.rs` | 各カーネル呼び出し元（`elementwise.rs` など）の `launch_builder` 部分 |
| `device.rs` | `crates/backend-cuda/src/device.rs` |

### 6.3 依存方式の選択肢（どちらも承認待ち）

HIP の FFI を担う Rust クレートは許容依存 10 区分に存在しない（`deps-policy.md`。`libloading` は cudarc の推移的依存にすぎず直接依存としては許容されていない。第 10 区分の `libc` は onnx-interop の external data 用途に限定）。

1. 新しい依存区分を設け、FFI バインディングクレートを `=x.y.z` 完全固定で採用する（ライセンス確認・`docs/license-matrix.md` 更新を伴う）
2. 手書きの `unsafe extern` 宣言で済ませ、dlopen の手段（`libc` 区分の拡張か別手段か）も別途承認を得る

## 7. セキュリティ契約（OWASP 観点）

- **A03**: HIPRTC のソースは静的テンプレートと検証済みの数値・enum からのみ組み立てる。`--gpu-architecture` 値や include パスはオプション文字列としてのみ渡し、シェル展開・ソース連結に使わない。
- **A06**: 依存追加は `=x.y.z` 固定＋ライセンス確認＋ユーザー承認。本書では依存を変更しない。
- **A08**: コードオブジェクトのディスクキャッシュは既存 NVRTC キャッシュの対策（`O_NOFOLLOW`・fd 相対解決・キーへの toolchain 版と target の包含）を踏襲する。cooperative の通常起動への黙示フォールバックは正しさを壊すため fail-closed 要件とする。
- **unsafe**: 新設はしない。草案は承認済みではない（§1）。

## 8. 起票案（ユーザー承認後に起票する。本イシューでは起票しない）

1. HIP FFI の依存方式（§6.3）の承認
2. HIP-Clang の FP contraction 既定値の実測と FMA 契約統一の判定
3. cooperative 入口の実装と security-auditor 監査
4. ROCm を対象範囲に入れる場合の spec 提案（fandhe-ai-spec 側）（→ #2128 で (b) 形式 draft 化: `docs/rocm-grade-up-conditions-v2-spec-proposal.md`。未起票）

## 9. 出典

- `.claude/skills/amd-rocm/references/hip/{cooperative-groups,hiprtc,occupancy,cpp-language-extensions,porting-cuda-to-hip}.md`、`.claude/skills/amd-rocm/samples/hiprtc-runtime-compilation.md`
- `crates/backend-cuda/src/nvrtc.rs:3935-3993`・`crates/backend-cuda/src/elementwise.rs:252-266`
- `cudarc =0.19.8`: `src/driver/safe/launch.rs:243`・`src/driver/sys/mod.rs:1179`・`src/driver/safe/core.rs:362,2278`・`src/nvrtc/safe.rs:235-241`
- `docs/backend-abstraction-amd-readiness-decision.md`（#1340）・`docs/cuda-streamk-decision.md`・`docs/backend-matrix.md` §3.4
- 公式 ROCm ドキュメントは本書では未参照（「要出典確認」項目は実装イシューで確認する）
