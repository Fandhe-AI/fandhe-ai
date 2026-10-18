# backend-rocm: HIPRTC コンパイル経路と f32 GEMM の決定記録（イシュー #3155）

親: #3150。前提: `docs/backend-rocm-hip-ffi-decision.md`（FFI 方式・U1）、
`docs/rocm-cooperative-hiprtc-unsafe-readiness-decision.md`（U1〜U7・HIPRTC 互換性表）。
対象実機は MI300X（gfx942・CDNA3・wave64。`docs/rocm-cloud-amd-procurement-decision.md`）。
手元に AMD 実機がないため、本 Issue の検証はソース生成とホスト側モデルまでで、実機での実行は #3159 に送る。

## 1. 構成

| ファイル | 役割 |
|---|---|
| `hip.rs` | HIP ランタイムのロードと呼び出し（malloc/memcpy/module/launch。生ポインタを取るメソッドは `unsafe fn`） |
| `hiprtc.rs` | `libhiprtc` のロードと `compile_code_object`（メモリ上のコードオブジェクトのみ。ディスクキャッシュなし） |
| `target.rs` | `GfxTarget` allowlist（`--gpu-architecture` の唯一の生成元） |
| `kernels_gemm.rs` | `GemmF32Tile`（検証済みタイル）・`render_gemm_f32_source`（`GemmSource` newtype） |
| `module.rs` | Module／Function／DeviceBuffer の RAII（`Arc<HipRuntime>` 保持） |
| `gemm.rs` | ホスト側の形状・グリッド検証と `RocmGemmF32` |
| `gemm_model.rs` | `cfg(test)`。カーネルの添字計算の逐語モデル |

## 2. FMA（contraction）

- 積和は `__builtin_fmaf` で明示した。CUDA 版は `acc += a*b` で contraction に依存していたが、ROCm 側は
  HIP-Clang の既定 contraction 値に結果が左右されないようにした。累積は単一アキュムレータ・k 昇順。
- CPU 参照（`f32::mul_add`、`gemm_naive` も k 昇順）とは理論上 bit 一致する（末尾のゼロ埋めは
  `fma(0,0,acc)=acc`）。ホスト上のモデルでは bit 一致を assert している。実機テストでは REQ-2 の複合判定のみ
  assert し、bit 一致は観測値の出力にとどめる。
- `-ffp-contract` は HIPRTC が受け付けるか未確認のため渡さない（渡すとコンパイル失敗の恐れがあり、明示 FMA で不要）。
- tolerance 定数・FMA 契約の方針（GPU も FMA）・REQ-2 判定式は変更していない。
- #3159 へ: HIP-Clang の contraction 既定値の確認、`__builtin_fmaf` が `v_fma_f32` に落ちることの確認。

## 3. A03 と target

- ソースは静的テンプレートと検証済み u32 の `#define`（ちょうど 7 行）のみ。`hiprtc` は `GemmSource` しか受けない。
- オプションは `--gpu-architecture=` + allowlist 由来の固定リテラル 1 本。include パス・環境変数は使わない。
- `parse_gcn_arch_name` は先頭の `gfxNNN` を完全一致で照合し、`:feature±` は捨てる。
- デバイスからの target 自動取得は実装しない。`hipGetDeviceProperties` は ROCm 6 以降ヘッダのマクロで版付きシンボルに
  差し替わり構造体レイアウトも版依存、`hipDeviceGetAttribute` の enum 整数も版で並びが変わった経緯があり、
  ヘッダなしの手書き FFI では出典なしにハードコードできないため（#3157・#3159 で出典確認後）。
  実機テストは `GfxTarget::Gfx942` に固定する。

## 4. wave64

- 移植した GEMM は `__shfl*`・lane mask・`warpSize` を使わず warp 幅に非依存（256 スレッドは wave64 で 4 本）。
  `GemmF32Tile::new` はスレッド数が 1024 以下かつ 64 の倍数であることを fail-closed で検査する。
- warp 幅依存のカーネル（rmsnorm・softmax 等）の移植時は #2126 の `WarpGeometry` 同様に幅を検査するか、
  wave64 用の 64 bit マスク設計を別 Issue で行う。

## 5. 出典と未確認事項

- libhiprtc の名前: `libhiprtc.so`（<https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_rtc.html>
  「HIPRTC is available as a separate library」）。版付き soname（同文書に `so` 名は MAJOR 版とある）・Windows 名は採用しない。
- 関数シグネチャは HIP 公式 API リファレンスの C 宣言に基づく。`hipError_t`／`hiprtcResult` は 4 バイト enum と
  みなして `c_int` で受け、0 を成功とした。**実機での ABI 確認は #3159**。
- メモリ API は `hipMemcpy`（enum 整数が必要）ではなく `hipMemcpyHtoD`／`hipMemcpyDtoH` を使う。
- エラー文字列 API（`*GetErrorString`）は返却ポインタの寿命の扱いが増えるため使わず、コード値のみを持つ。
- HIPRTC ログは上限 1 MiB（超過時は取得せずサイズのみ記録）、コードオブジェクトは 256 MiB 上限。

## 6. unsafe 一覧（正式な監査記録は #3158）

| 箇所 | 不変条件 |
|---|---|
| `Library::new`／`get`（hip.rs・hiprtc.rs） | U1。固定名のみ、ポインタは同一構造内、型は C 宣言と一致 |
| HIPRTC create／compile（U2） | C 文字列は NUL 終端・呼び出し中生存、オプション個数一致、ヘッダ 0 件 |
| code 取得（U3） | サイズちょうどのバッファ、サイズ 0 を拒否 |
| Program の Drop（U7） | 所有権で 1 回だけ破棄 |
| module load／get_function／unload（U4・U7） | コードオブジェクトは module より長生き、関数名は固定、同期後に破棄 |
| malloc／memcpy／free | 長さは checked_mul で一致、Drop で 1 回だけ free |
| launch（U5） | `kernelParams` の型・順序・個数がカーネル宣言と一致（テストで固定）、同期完了まで生存、カーネル内で REQ-8 境界チェック |

## 7. 申し送り

- #3157: BackendOps・facade 統合、デバイスからの target 取得。
- #3158: unsafe 監査記録。
- #3159: 実機実行（`tests/gemm_real_device.rs`）、contraction・ABI・soname 確認、grid 上限の照会。
- 別 Issue（起票は承認後）: ディスクキャッシュ、MFMA 経路、f16・bias_act・split-K、版付き soname・Windows。
