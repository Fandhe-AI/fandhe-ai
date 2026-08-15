//! `setmaxnreg`（warp specialization レジスタ再配分）の sm_121 実機動作
//! 確認 spike（イシュー #484。親イシュー #480 の A-4）。
//!
//! ## 位置づけ
//!
//! PTX ISA の `setmaxnreg.inc/dec.sync.aligned.u32` は、producer/consumer
//! warp 間でレジスタ予算を非対称配分する warp specialization パターン
//! （warpgroup＝連続 4 warp／128 スレッド単位の命令）向けに導入された。
//! 歴代 `sm_90a` 系の arch-accelerated feature として案内されることが多く、
//! コンシューマ系 Blackwell（SM12x）の `compute_121`／DGX Spark GB10 +
//! NVRTC（CUDA 13.0 系）で NVRTC が受理し実行できるかは本リポジトリで
//! 未検証だった（`docs/cuda-tensor-core-design.md` §2 は tcgen05/wgmma の
//! 非対応のみ確定済みで setmaxnreg は未記載）。この結果は後続 B-3
//! （タイル拡大時のレジスタ予算設計）の設計自由度の上限に直接影響する。
//!
//! 「使えない」という結論も spike の正当な成果である。そのため本ファイルの
//! 2 テストは、NVRTC コンパイル・カーネル実行の失敗を `panic` させず
//! `SETMAXNREG_PROBE_RESULT: ...` 形式で標準出力へ構造的に記録したうえで
//! pass する設計とする（#64 の `tensor_core_real_device.rs` と同型の
//! 「実測記録テストは pass、CUDA デバイス自体の不在のみ fail-loud」方針。
//! 対して CUDA デバイス自体が使えない環境（`CudaDevice::new` 失敗）は
//! 既存規約どおり `.expect` で fail-loud にし、実機以外での silent green
//! を許さない）。
//!
//! ## 実機前提・CI 非実行
//!
//! 実機（DGX Spark GB10 等、compute capability 12.1 相当）必須の
//! `#[ignore]` 分離テストであり、通常 CI（GitHub ホステッド）では実行
//! されない（`.claude/rules/ci.md`「実機依存」）。実行は
//! `docs/real-hardware-verification-env.md` の手順に従い
//! `cargo test -p backend-cuda --release --test setmaxnreg_probe_real_device
//! -- --ignored --nocapture` で行い、出力を
//! `docs/cuda-tensor-core-design.md` の「setmaxnreg プローブ結果（#484）」
//! 節へ転記する運用とする。
//!
//! ## B-3 への引き渡し
//!
//! 使用可と判明した場合、producer warp の dealloc 量・consumer warp の
//! alloc 量をどこまで非対称化できるかの設計自由度は B-3
//! （タイル拡大時のレジスタ予算設計）が引き継ぐ。使用不可の場合、B-3 は
//! 対称レジスタ予算前提でタイル上限を設計する（本ファイルは判定のみを
//! 提供し、B-3 自体の設計判断は行わない）。
//!
//! ## A03（インジェクション）対応
//!
//! カーネルソースは本ファイル内の `&'static str` コンパイル時定数のみを
//! 使い、外部入力・環境変数をソース文字列へ連結しない
//! （`nvrtc.rs` の既存契約と同じ方針。`.claude/rules/security.md`）。

use backend_cuda::{CudaDevice, CudaError, compile_ptx};
use cudarc::driver::{LaunchConfig, PushKernelArg};

/// warpgroup（4 warp = 128 スレッド）1 個の起動を前提に、`setmaxnreg.dec`
/// のみを発行してから出力へ書き込むカーネル。
///
/// PTX ISA 上、`setmaxnreg` は warpgroup 全体が同一命令列を実行することを
/// 要求するため、ブロック次元は 128 の倍数（かつ本プローブでは 1 warpgroup
/// のみで足りるため 128 固定）とする。手動境界チェック（`if (idx < n)`）は
/// REQ-8（`.claude/rules/coding-rust.md`）に従い、プローブ用途であっても
/// 省略しない。
///
/// `__launch_bounds__(128)` を付与するのは [`PROBE_SETMAXNREG_INCDEC`] と
/// 同じ理由（イシュー #484 レビュー指摘 Low 対応）: [`report_control_baseline_regs`]
/// で [`CONTROL_DEC`] のベースライン register/thread 数を実測し、
/// `setmaxnreg.dec ... 64` が要求するレジスタ予算（64 以下への削減）が
/// 実際のベースラインと整合するかを [`setmaxnreg_dec_probe`] が
/// `stage=coherence_check` として記録する（`PROBE_SETMAXNREG_INCDEC` の
/// P2 対応が dec 単体側には未適用だった非対称を解消）。launch bounds を
/// 対照カーネルと揃えないと、ベースライン実測値が本プローブの実際の
/// コンパイル構成と異なる値になり誤読を招くため必須。
const PROBE_SETMAXNREG_DEC: &str = r#"
extern "C" __global__ void __launch_bounds__(128) probe_setmaxnreg_dec(
    const float* __restrict__ in,
    float* __restrict__ out,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    asm volatile("setmaxnreg.dec.sync.aligned.u32 64;");
    if (idx < n) {
        out[idx] = in[idx] + 1.0f;
    }
}
"#;

/// `setmaxnreg.dec`／`setmaxnreg.inc` を producer/consumer warpgroup で
/// 非対称に発行するパターン（CUTLASS の `warpgroup_reg_dealloc`/
/// `warpgroup_reg_alloc` に相当する使い方を模す。producer/consumer warp
/// specialization の典型形。以前の実装は全 warp が dec→inc を順に実行する
/// 対称往復のみだったため producer/consumer 間の非対称レジスタ再配分を
/// 検証できておらず、イシュー #484 レビュー指摘で本構成に修正した）。
///
/// PTX ISA 上 `setmaxnreg` は warpgroup（連続 128 スレッド）単位で全
/// スレッドが同一命令列を実行することを要求する（本ファイル冒頭コメント）
/// ため、producer/consumer の非対称性は同一 warpgroup 内の warp 間では
/// 表現できず、**warpgroup 単位**で表現する。本カーネルは 1 ブロック =
/// 2 warpgroup（256 スレッド。[`PRODUCER_CONSUMER_BLOCK_DIM`]）で起動し、
/// `threadIdx.x / 128`（warpgroup ごとに値が揃うため分岐は warpgroup 内で
/// 収束する）で判定した warpgroup 0 を producer（`setmaxnreg.dec` のみを
/// 発行してレジスタ予算を解放する側）、warpgroup 1 を consumer
/// （`setmaxnreg.inc` のみを発行して producer が解放した分を確保する側）
/// とする。
///
/// dec/inc の値（24 / 232。8 の倍数・[24, 256] の許容範囲内）は、
/// `__launch_bounds__(256)`（カーネル宣言に付与）によりコンパイラの
/// ベースライン割り当てをブロック内 256 スレッド構成に対してヒントした
/// うえで CUTLASS の一般的な producer/consumer 値域を踏襲したものである。
///
/// **ベースライン register/thread 数の確定（イシュー #484 レビュー指摘
/// P2 対応）**: `--maxrregcount` 等の明示指定は行っていないが、代わりに
/// [`report_control_baseline_regs`] が `CONTROL_INCDEC`（`setmaxnreg` を
/// 含まない対照カーネル。同一 `__launch_bounds__(256)`）を実際にロードし
/// `cuFuncGetAttribute(CU_FUNC_ATTRIBUTE_NUM_REGS)`（`CudaFunction::
/// num_regs`）でコンパイラが実際に割り当てたベースライン register/thread
/// 数を実測・記録する（NVRTC は PTX 生成のみでレジスタ割付を行わず、割付は
/// `cuModuleLoadData` 時のドライバ JIT〈ptxas 相当〉が行うため、この属性
/// クエリが「実際に成立したベースライン」を確定できる唯一の実測経路）。
/// これにより 24/232 という値がベースラインから見て妥当な増減幅かどうかを
/// `SETMAXNREG_PROBE_RESULT stage=control_baseline_regs` の実測値と突き合わせ
/// て判断できる。したがって `.dec`/`.inc` が要求するレジスタ予算総量が実際に
/// 成立するかは、このベースライン実測値と本プローブの実行結果
/// （`try_load_and_run` の launch/synchronize 失敗捕捉。本ファイル冒頭
/// 「使えないという結論も spike の正当な成果」）を組み合わせて判断する。
/// 失敗時は「命令自体が拒否された」（真の不可）と「本プローブの値の組み合わせ
/// がこのベースラインでは不成立だった」（値の再選定が必要）をベースライン
/// 実測値との比較で区別できる。`docs/cuda-tensor-core-design.md` への転記時は
/// `control_baseline_regs` の実測値も併記すること。
///
/// **`result=success` の判別力限界と `coherent` ログの位置づけ（イシュー
/// #484 レビュー指摘 Medium・PR #636 再指摘 P2 対応）**: 本カーネルの演算
/// 自体（`in[idx] * 2.0f`）は 3 引数・idx 計算程度でベースラインレジスタ数が
/// 数十/thread 程度に収まりうるため、`result=success`（`try_load_and_run` の
/// launch/synchronize/期待値検証がすべて通過）は「setmaxnreg による再配分が
/// 実際に成立した」ことと「レジスタ予算的に無風（`dec`/`inc` の対象値が
/// ベースラインに対して意味のある制約にならない）で通っただけ」を区別
/// できない。この判別力限界自体は演算を複雑化してレジスタ圧を上げても
/// 質的には解消しない（どの程度で「意味のある制約」になるかは ptxas の
/// 割付次第で決め打ちできないため）。そのため本ファイルは代わりに、
/// `.dec 24`/`.inc 232` との整合性を `setmaxnreg_incdec_probe` が
/// `stage=coherence_check` 行として **2 段階**でログする: (1)
/// [`report_control_baseline_regs`] が `setmaxnreg` を含まない対照カーネル
/// から実測した値（`source=control`）、(2) [`try_load_and_run`] が
/// `setmaxnreg` を実際に発行するプローブカーネル自身から実測した値
/// （`source=probe_self`）。
///
/// **`num_regs`（`CU_FUNC_ATTRIBUTE_NUM_REGS`）は両 `source` とも静的な
/// register/thread 割り当て値であり、`setmaxnreg` 命令が実行される時点の
/// 動的なレジスタ再配分後の実割当量ではない**（`cuModuleLoadData` 時点の
/// ドライバ JIT〈ptxas 相当〉がコンパイル時に確定させる値であり、
/// `setmaxnreg.dec/inc` はカーネル**実行中**に warp 単位でこの予算を
/// 動的に変更する命令のため、両者は原理的に異なる観測対象である）。
/// したがって `source=probe_self` を含め `coherent` は producer/consumer
/// 間の再配分が**実際に成立したこと**の証明にはならず、いずれの
/// `source` も **診断参考値**にとどまる（以前の実装は `source=probe_self` を
/// 「PTX ISA の dec/inc 制約が本来要求する対象そのものであり権威値」と
/// 位置付けていたが、静的値であるという性質は `source=control` と変わらない
/// ため、この位置付けは誤りであり本対応で撤回する）。`coherent=false` は
/// 「`.dec`/`.inc` の対象値が実測ベースラインと矛盾する（無風または UB 域）」
/// ことを示す客観的な**警告シグナル**として引き続き有用だが、
/// `coherent=true` は「矛盾は見つからなかった」以上の意味を持たず、それ
/// 単独でも `result=success` との組み合わせでも「setmaxnreg による再配分が
/// 実際に成立した」ことの根拠にはできない。
///
/// **本スパイクが確定できる範囲・できない範囲**: 本ファイルが実測で確定
/// できるのは (a) NVRTC が命令を**受理**するか（`stage=nvrtc_compile`・
/// `module_load`・`load_function`）と (b) 受理された場合に実行が**完走**し
/// 出力がビット完全一致するか（`stage=execute`。不一致は `result=corrupted`
/// として `panic`）の 2 点のみである。producer/consumer 間でレジスタ予算が
/// 実際に非対称再配分された動的証拠（例: 実行時の SM レジスタ占有率を
/// warp 単位で追跡する `nsight-compute` 等のプロファイラによる観測）は
/// 本ファイルのスコープ外であり、必要な場合は別途 Issue で追跡する
/// （`.claude/rules/out-of-scope-tracking.md`）。`docs/cuda-tensor-core-design.md`
/// への転記時は、使用可否の一次判断根拠を (a)・(b) の実測結果に置き、
/// `coherent` は「矛盾の有無を示す補助的な警告シグナル」として併記する
/// （`coherent=false` かつ `result=success` の場合に「使用可」と断定しない
/// 運用は維持する）。
const PROBE_SETMAXNREG_INCDEC: &str = r#"
extern "C" __global__ void __launch_bounds__(256) probe_setmaxnreg_incdec(
    const float* __restrict__ in,
    float* __restrict__ out,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int warpgroup_id = threadIdx.x / 128;
    if (warpgroup_id == 0) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 24;");
    } else {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 232;");
    }
    float v = 0.0f;
    if (idx < n) {
        v = in[idx] * 2.0f;
    }
    if (idx < n) {
        out[idx] = v;
    }
}
"#;

/// [`PROBE_SETMAXNREG_DEC`] から `asm volatile("setmaxnreg.dec...")` の
/// 1 行だけを除いた対照カーネル（それ以外はバイト単位で同一）。
/// （[`CONTROL_INCDEC`] は `PROBE_SETMAXNREG_INCDEC` の warpgroup 分岐
/// ごと asm 2 行を除いた対照カーネルであり、除く内容が異なる点に注意）。
///
/// `compile_ptx` の失敗は「`setmaxnreg` 命令自体が拒否された」以外に、
/// `libnvrtc` 不在（`CudaError::NvrtcUnavailable`）や include パス解決
/// 失敗等、命令と無関係な理由でも起こりうる。同一 arch に対する対照
/// カーネルのコンパイル成否を併記することで、プローブ結果を
/// 「コンパイル基盤は健全だが setmaxnreg のみ拒否された」（`control=accepted`
/// かつ `probe` が拒否）とそれ以外（toolchain 自体の問題で判定不能）を
/// 読み手が区別できるようにする。
///
/// `__launch_bounds__(128)` を [`PROBE_SETMAXNREG_DEC`] と揃えて付与する
/// （イシュー #484 レビュー指摘 Low 対応。[`CONTROL_INCDEC`] が
/// `PROBE_SETMAXNREG_INCDEC` と揃えているのと同じ理由。揃えないと
/// [`report_control_baseline_regs`] の実測値がプローブ本体のベースラインと
/// 異なる値になり誤読を招く）。
const CONTROL_DEC: &str = r#"
extern "C" __global__ void __launch_bounds__(128) control_dec(
    const float* __restrict__ in,
    float* __restrict__ out,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx] + 1.0f;
    }
}
"#;

/// [`PROBE_SETMAXNREG_INCDEC`] の対照カーネル（[`CONTROL_DEC`] と同じ目的）。
/// `warpgroup_id` による producer/consumer 分岐と両方の `asm volatile`
/// （dec・inc）を除いた点以外は [`PROBE_SETMAXNREG_INCDEC`] と同一の
/// 計算・書き込みを行う（同一ブロック次元 [`PRODUCER_CONSUMER_BLOCK_DIM`]
/// で起動することを前提とする）。
///
/// `__launch_bounds__(256)` を [`PROBE_SETMAXNREG_INCDEC`] と揃えて付与する
/// （イシュー #484 レビュー指摘 P2）。コンパイラのベースラインレジスタ
/// 割り当ては launch bounds ヒントに依存するため、これを揃えずに
/// [`report_control_baseline_regs`] で計測すると `PROBE_SETMAXNREG_INCDEC`
/// が実際に前提とするベースラインと異なる値を「確定値」として誤読しうる。
const CONTROL_INCDEC: &str = r#"
extern "C" __global__ void __launch_bounds__(256) control_incdec(
    const float* __restrict__ in,
    float* __restrict__ out,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float v = 0.0f;
    if (idx < n) {
        v = in[idx] * 2.0f;
    }
    if (idx < n) {
        out[idx] = v;
    }
}
"#;

/// 1 warpgroup 分のブロック次元（128 スレッド = 4 warp）。`setmaxnreg` は
/// warpgroup 単位の命令のため、ここが 128 の倍数からずれると一部 warp が
/// 命令列から欠落し未定義動作になる（`gemm.rs::WMMA_TF32_BLOCK_DIM` と
/// 同じ「ホスト側ブロック次元とカーネル内命令の 1:1 対応」契約）。
/// [`PROBE_SETMAXNREG_DEC`]／[`CONTROL_DEC`]（1 warpgroup のみで完結する
/// 片道パターン）に用いる。
const WARPGROUP_BLOCK_DIM: (u32, u32, u32) = (128, 1, 1);

/// producer/consumer 2 warpgroup 分のブロック次元（256 スレッド = 8 warp）。
/// [`PROBE_SETMAXNREG_INCDEC`] は `threadIdx.x / 128` が 0 の warpgroup を
/// producer（dec）・1 の warpgroup を consumer（inc）として扱うため、1
/// ブロックに必ず 2 warpgroup（256 スレッド）を割り当てる。
const PRODUCER_CONSUMER_BLOCK_DIM: (u32, u32, u32) = (256, 1, 1);

fn launch_config(n: u32, block_dim: (u32, u32, u32)) -> LaunchConfig {
    let grid_x = n.div_ceil(block_dim.0);
    LaunchConfig {
        grid_dim: (grid_x, 1, 1),
        block_dim,
        shared_mem_bytes: 0,
    }
}

/// `control_src`（[`CONTROL_DEC`]/[`CONTROL_INCDEC`]。`setmaxnreg` を含まない
/// 対照カーネル）を先にコンパイルしたうえで `src`（`setmaxnreg` を含む
/// プローブ本体）を NVRTC コンパイルし、`SETMAXNREG_PROBE_RESULT` 行として
/// 記録する。失敗は `panic` させず `None` を返す（本ファイル冒頭コメント
/// 「使えないという結論も spike の正当な成果」）。
///
/// `result` の判定基準（[`CONTROL_DEC`] ドキュメンテーションコメント
/// 参照）:
/// - `accepted`: `src` のコンパイルが成功
/// - `rejected`: 対照カーネルは成功したが `src` のみ失敗
///   （＝ `setmaxnreg` 命令自体が拒否されたと結論できる）
/// - `inconclusive`: 対照カーネルも失敗、または `libnvrtc` 不在
///   （`CudaError::NvrtcUnavailable`）。toolchain 側の問題であり
///   `setmaxnreg` の可否について何も結論できない
///
/// `control_ok`（対照カーネルのコンパイル成否）は呼び出し元が算出済みの値を
/// 渡す。基準 arch（`device.arch()`）向けの呼び出しは
/// [`report_control_baseline_regs`] が同一 `control_src`・同一 arch で
/// 既にコンパイル済みのため、ここで再コンパイルすると同一入力の NVRTC
/// コンパイルが 2 回走る（イシュー #484 レビュー指摘 Low 対応）。一方
/// arch-accelerated 版（`<arch>a`）の呼び出しは基準 arch と異なる arch で
/// コンパイルするため呼び出し元で別途コンパイルして渡す（真に別入力であり
/// 重複ではない）。
fn try_compile(label: &str, src: &str, control_ok: bool, arch: &str) -> Option<cudarc::nvrtc::Ptx> {
    println!(
        "SETMAXNREG_PROBE_RESULT stage=nvrtc_compile_control kernel={label} arch={arch} \
         control_accepted={control_ok}"
    );

    match compile_ptx(src, arch) {
        Ok(ptx) => {
            println!(
                "SETMAXNREG_PROBE_RESULT stage=nvrtc_compile kernel={label} arch={arch} \
                 result=accepted"
            );
            Some(ptx)
        }
        Err(err @ CudaError::NvrtcUnavailable { .. }) => {
            println!(
                "SETMAXNREG_PROBE_RESULT stage=nvrtc_compile kernel={label} arch={arch} \
                 result=inconclusive detail={err}"
            );
            None
        }
        Err(err) => {
            let result = if control_ok {
                "rejected"
            } else {
                "inconclusive"
            };
            println!(
                "SETMAXNREG_PROBE_RESULT stage=nvrtc_compile kernel={label} arch={arch} \
                 result={result} control_accepted={control_ok} detail={err}"
            );
            None
        }
    }
}

/// `control_src`（`setmaxnreg` を含まない対照カーネル）を `arch` でコンパイル・
/// ロードし、`CU_FUNC_ATTRIBUTE_NUM_REGS`（[`cudarc::driver::CudaFunction::
/// num_regs`]）でコンパイラが実際に割り当てたベースライン register/thread 数を
/// 実測して `SETMAXNREG_PROBE_RESULT` 行として記録する（イシュー #484
/// レビュー指摘 P2 対応。[`PROBE_SETMAXNREG_INCDEC`] ドキュメンテーション
/// コメント参照）。
///
/// カーネルを起動しない点が [`try_load_and_run`] と異なる: レジスタ割り当ては
/// `cuModuleLoadData` 時点のドライバ JIT（ptxas 相当）で確定するため、
/// `load_function` までで実測に十分であり、実行（launch/synchronize）は
/// 不要（かつ対照カーネルは `setmaxnreg` を含まないため実行結果自体に
/// 検証価値がない）。
///
/// コンパイル・ロードいずれかが失敗した場合も `panic` させず `result=failed`
/// として記録する（本ファイル冒頭「使えないという結論も spike の正当な
/// 成果」と同じ方針。対照カーネルの失敗は toolchain 側の問題である可能性が
/// 高く、[`try_compile`] 側の `control_accepted` ログと合わせて診断する）。
///
/// 戻り値は `(Option<i32>, bool)`。前者は測定成功時の `num_regs_per_thread`
/// （失敗時は `None`）で、呼び出し元（[`setmaxnreg_dec_probe`]／
/// [`setmaxnreg_incdec_probe`]）はこれを [`report_setmaxnreg_coherence`] へ
/// `source="control"`（参考値）として渡す（イシュー #484 レビュー指摘 Medium
/// 対応。[`PROBE_SETMAXNREG_INCDEC`] ドキュメンテーションコメント参照）。
/// 後者は `control_src` のコンパイル成否（`compile_ptx` の `Ok`/`Err`）で
/// あり、[`try_compile`] が基準 arch 向け呼び出しでそのまま再利用する
/// ことで同一 `control_src`・同一 arch の NVRTC コンパイルを 2 回走らせない
/// （イシュー #484 レビュー指摘 Low 対応）。ロード以降の失敗（`control_ok`
/// が `true` でも `num_regs` 側が `None` になりうる）は `control_ok` に
/// 影響しない: コンパイル自体は成功しているため `try_compile` の
/// `rejected`/`inconclusive` 判定基準（コンパイル成否）とは独立である。
fn report_control_baseline_regs(
    device: &CudaDevice,
    label: &str,
    control_src: &str,
    arch: &str,
) -> (Option<i32>, bool) {
    let ptx = match compile_ptx(control_src, arch) {
        Ok(ptx) => ptx,
        Err(err) => {
            println!(
                "SETMAXNREG_PROBE_RESULT stage=control_baseline_regs kernel={label} arch={arch} \
                 result=failed detail={err}"
            );
            return (None, false);
        }
    };
    let module = match device.context().load_module(ptx) {
        Ok(module) => module,
        Err(err) => {
            println!(
                "SETMAXNREG_PROBE_RESULT stage=control_baseline_regs kernel={label} arch={arch} \
                 result=failed detail={err:?}"
            );
            return (None, true);
        }
    };
    let func = match module.load_function(label) {
        Ok(func) => func,
        Err(err) => {
            println!(
                "SETMAXNREG_PROBE_RESULT stage=control_baseline_regs kernel={label} arch={arch} \
                 result=failed detail={err:?}"
            );
            return (None, true);
        }
    };
    match func.num_regs() {
        Ok(num_regs) => {
            println!(
                "SETMAXNREG_PROBE_RESULT stage=control_baseline_regs kernel={label} arch={arch} \
                 result=measured num_regs_per_thread={num_regs}"
            );
            (Some(num_regs), true)
        }
        Err(err) => {
            println!(
                "SETMAXNREG_PROBE_RESULT stage=control_baseline_regs kernel={label} arch={arch} \
                 result=failed detail={err:?}"
            );
            (None, true)
        }
    }
}

/// `SETMAXNREG_PROBE_RESULT` の `Option<i32>` 系の値を key=value ログの値
/// として扱いやすい裸のトークンへ整形する（`48` / `none`）。`{:?}` の
/// `Some(48)`／`None` 表記のまま出力すると、他フィールドの裸値
/// （`kernel=probe_setmaxnreg_dec` 等）と書式が混在し、grep・スクリプトでの
/// 機械的な値抽出を妨げる（イシュー #484 レビュー指摘 Low 対応）。
fn fmt_opt_regs(value: Option<i32>) -> String {
    match value {
        Some(n) => n.to_string(),
        None => "none".to_string(),
    }
}

/// `measured_regs`（register/thread 数の実測値）と、本プローブが発行する
/// `setmaxnreg.dec`/`.inc` の対象値（`dec_target`／`inc_target`。片方のみの
/// プローブは `None` を渡す）の整合性を判定し
/// `SETMAXNREG_PROBE_RESULT stage=coherence_check` として記録する
/// （イシュー #484 レビュー指摘 Medium 対応）。
///
/// `source` は `measured_regs` の由来を示す診断ラベルで、`"probe_self"`
/// （`setmaxnreg` を実際に発行するプローブカーネル自身を
/// [`try_load_and_run`] がロードして実測した値）と `"control"`
/// （[`report_control_baseline_regs`] が `setmaxnreg` を含まない対照カーネル
/// から実測した値。`__launch_bounds__` は揃えているが `setmaxnreg` 命令の
/// 有無自体が ptxas の静的レジスタ割り当てへ与える影響までは揃わない）の
/// いずれかを渡す。**両者とも `cuModuleLoadData` 時点のドライバ JIT が
/// 確定させた静的な register/thread 数であり、`setmaxnreg` 命令が実行時に
/// 動的へ変更するレジスタ予算そのものではない**（イシュー #484 レビュー
/// 指摘 Medium・PR #636 再指摘 P2 対応。[`PROBE_SETMAXNREG_INCDEC`]
/// ドキュメンテーションコメント参照）。したがって `source=probe_self` を
/// 「権威値」とする位置付けは撤回し、両 `source` とも**診断参考値**として
/// 扱う: `coherent=false` は `.dec`/`.inc` の対象値と実測ベースラインの
/// 矛盾を示す警告シグナルとして有用だが、`coherent=true` は「矛盾が
/// 見つからなかった」以上の意味を持たず、producer/consumer 間の再配分が
/// 実際に成立したことの証明にはならない。呼び出し元は両方の `source` で
/// 本関数を呼び、`docs/cuda-tensor-core-design.md` への転記時は
/// `source=probe_self` 行を参考値として併記しつつ、使用可否の一次判断根拠は
/// `nvrtc_compile`/`module_load`/`load_function`（受理）と `execute`
/// （実行完走＋出力一致）の実測結果に置くこと。
///
/// PTX ISA 上、`setmaxnreg.dec` の対象値は現在のレジスタ数**以下**、
/// `setmaxnreg.inc` の対象値は現在のレジスタ数**以上**であることが要求される
/// （さもなくば未定義動作になりうる）。本ファイルの `.dec`/`.inc` はいずれも
/// 同一カーネル関数内で発行され、`num_regs_per_thread` はカーネル関数単位で
/// 決まる値のためベースラインは producer/consumer 双方で共通である。よって
/// `dec_target <= measured_regs`（dec 側）・`inc_target >= measured_regs`
/// （inc 側）が両方成立する場合のみ `coherent=true` とする。
///
/// `coherent=false` は「`result=success` であってもレジスタ予算的に無風
/// （または PTX ISA 上不整合）だった」ことを示す客観的シグナルであり、
/// `docs/cuda-tensor-core-design.md` への転記時に `result=success` のみを
/// 見て「使用可」と誤読するのを防ぐ（[`PROBE_SETMAXNREG_INCDEC`]
/// ドキュメンテーションコメント参照）。実測自体が失敗している場合
/// （`measured_regs=None`）は判定不能として `coherent=unknown` を記録し
/// `false` 側へ丸めない（fail-closed で「不整合」と断定しない）。
fn report_setmaxnreg_coherence(
    label: &str,
    source: &str,
    measured_regs: Option<i32>,
    dec_target: Option<i32>,
    inc_target: Option<i32>,
) {
    let coherent = match measured_regs {
        Some(baseline) => {
            let dec_ok = dec_target.is_none_or(|dec| dec <= baseline);
            let inc_ok = inc_target.is_none_or(|inc| inc >= baseline);
            (dec_ok && inc_ok).to_string()
        }
        None => "unknown".to_string(),
    };
    println!(
        "SETMAXNREG_PROBE_RESULT stage=coherence_check kernel={label} source={source} \
         measured_regs={} dec_target={} inc_target={} coherent={coherent}",
        fmt_opt_regs(measured_regs),
        fmt_opt_regs(dec_target),
        fmt_opt_regs(inc_target),
    );
}

/// コンパイル成功済みカーネルをロード・起動・同期し、成否を記録する。
///
/// `setmaxnreg` が実機で受理されない場合（`CUDA_ERROR_ILLEGAL_INSTRUCTION`
/// 等）も `panic` させず結果として記録する。実行成功時は出力バッファを
/// 回収し「命令が実行を破壊していないか」の期待値検証も行う。
///
/// `label` は診断ログ用のラベルであると同時に、`module.load_function(label)`
/// へそのまま渡す **CUDA 側 `extern "C"` 関数シンボル名**でもある
/// （呼び出し元は `ptx` を生成した `src`（`PROBE_SETMAXNREG_DEC`/
/// `PROBE_SETMAXNREG_INCDEC`）内の `__global__` 関数名と必ず一致させること。
/// arch-accelerated 版〈`<arch>a`〉の PTX を実行する場合も、シンボル名は
/// 基準 arch 版と同一〈`probe_setmaxnreg_dec`/`probe_setmaxnreg_incdec`〉の
/// ため `label` はそのまま渡す。対照カーネル〈`CONTROL_DEC`/
/// `CONTROL_INCDEC`〉は本関数へは渡さない＝実行しない）。
///
/// `arch` は診断ログ専用（`load_function` には渡さない）で、基準 arch版と
/// arch-accelerated 版の呼び出しを `SETMAXNREG_PROBE_RESULT` の行から区別
/// するために付与する（PR #636 レビュー指摘 P2 対応。同一 `label` で 2 回
/// 呼ばれるため `arch` フィールドがないと `stage=execute kernel=...` 等の
/// 行がどちらの呼び出しに由来するか転記者が判別できない）。
///
/// `block_dim` は呼び出し元が `src` の warpgroup 構成に合わせて指定する
/// （[`WARPGROUP_BLOCK_DIM`]＝1 warpgroup／[`PRODUCER_CONSUMER_BLOCK_DIM`]
/// ＝producer・consumer 2 warpgroup）。`n`（256 要素固定）は
/// [`PRODUCER_CONSUMER_BLOCK_DIM`] でも 1 ブロックに収まり producer・
/// consumer 双方の warpgroup が確実に起動される値として選んでいる。
///
/// arch-accelerated 版も基準 arch 版と同じ「受理されても `panic` させない」
/// 方針を維持する（`module_load`/`load_function`/`launch`/`synchronize` の
/// 失敗は fail-loud にしない）。ただし `result=corrupted`（実行完走したが
/// 出力が期待値と不一致）は基準 arch 版と同様に `panic` させる: 出力破壊は
/// arch-accelerated 版であっても「命令の受理可否」の範疇を超えた危険な
/// シグナルであり、softening しない（本関数末尾のコメント参照）。
///
/// 戻り値の `Option<i32>` は、`setmaxnreg` を実際に発行する本プローブ
/// カーネル自身（`func`）を [`cudarc::driver::CudaFunction::num_regs`] で
/// 実測した `num_regs_per_thread`（診断参考値であり動的な実割当量では
/// ない。[`report_setmaxnreg_coherence`] ドキュメンテーションコメント
/// 参照）。`module_load`/`load_function` 失敗時は実測不能のため `None` を
/// 返す。実測は `module.load_function` 直後（起動前）に行う: レジスタ
/// 割り当ては `cuModuleLoadData` 時点のドライバ JIT で確定済みのため、
/// 実行結果の成否に左右されず記録できる（[`report_control_baseline_regs`]
/// と同じ理屈）。
fn try_load_and_run(
    device: &CudaDevice,
    label: &str,
    arch: &str,
    ptx: cudarc::nvrtc::Ptx,
    block_dim: (u32, u32, u32),
    expected: impl Fn(f32) -> f32,
) -> Option<i32> {
    let n: u32 = 256;
    let mut rng = bench_harness::rng::Xorshift64Star::new(0xC0FFEE);
    let input: Vec<f32> = rng.fill_vec(n as usize);

    let module = match device.context().load_module(ptx) {
        Ok(module) => module,
        Err(err) => {
            println!(
                "SETMAXNREG_PROBE_RESULT stage=module_load kernel={label} arch={arch} \
                 result=failed detail={err:?}"
            );
            return None;
        }
    };
    let func = match module.load_function(label) {
        Ok(func) => func,
        Err(err) => {
            println!(
                "SETMAXNREG_PROBE_RESULT stage=load_function kernel={label} arch={arch} \
                 result=failed detail={err:?}"
            );
            return None;
        }
    };

    // `setmaxnreg` が要求する dec/inc の PTX ISA 制約は、対照カーネルではなく
    // このプローブカーネル自身の静的レジスタ割り当てに対して定義される。
    // 起動前に実測して記録し、`probe_self_regs` を呼び出し元へ返す
    // （診断参考値。[`report_setmaxnreg_coherence`] 参照）。
    let probe_self_regs = match func.num_regs() {
        Ok(num_regs) => {
            println!(
                "SETMAXNREG_PROBE_RESULT stage=probe_self_regs kernel={label} arch={arch} \
                 result=measured num_regs_per_thread={num_regs}"
            );
            Some(num_regs)
        }
        Err(err) => {
            println!(
                "SETMAXNREG_PROBE_RESULT stage=probe_self_regs kernel={label} arch={arch} \
                 result=failed detail={err:?}"
            );
            None
        }
    };

    let in_dev = device
        .stream()
        .clone_htod(&input)
        .expect("clone_htod must succeed on CUDA-equipped test runner");
    let mut out_dev = device
        .stream()
        .alloc_zeros::<f32>(n as usize)
        .expect("alloc_zeros must succeed on CUDA-equipped test runner");

    let cfg = launch_config(n, block_dim);
    let n_i = n as i32;

    // SAFETY: カーネル引数（in_dev/out_dev/n_i）はホスト側で確保・検証済み
    // の固定長バッファであり、カーネル内の手動境界チェック（`if (idx < n)`。
    // 本ファイル冒頭 `PROBE_SETMAXNREG_DEC`/`PROBE_SETMAXNREG_INCDEC` 参照。
    // REQ-8）と合わせて OOB 読み書きが起きない根拠とする。`setmaxnreg` 命令
    // 自体が受理・実行されるかは本プローブが検証する対象そのものであり、
    // 実行時エラー（`CUDA_ERROR_ILLEGAL_INSTRUCTION` 等）は `launch`/
    // `synchronize` の `Result::Err` として捕捉し `panic` させない。
    let launch_result = unsafe {
        device
            .stream()
            .launch_builder(&func)
            .arg(&in_dev)
            .arg(&mut out_dev)
            .arg(&n_i)
            .launch(cfg)
    };
    if let Err(err) = launch_result {
        println!(
            "SETMAXNREG_PROBE_RESULT stage=launch kernel={label} arch={arch} result=failed \
             detail={err:?}"
        );
        return probe_self_regs;
    }

    if let Err(err) = device.stream().synchronize() {
        println!(
            "SETMAXNREG_PROBE_RESULT stage=synchronize kernel={label} arch={arch} \
             result=failed detail={err:?}"
        );
        return probe_self_regs;
    }

    let output = device
        .stream()
        .clone_dtoh(&out_dev)
        .expect("clone_dtoh must succeed on CUDA-equipped test runner");

    // `expected` は `x + 1.0f32`／`x * 2.0f32` のような単一 IEEE 754 単精度
    // 演算 1 回のみであり、GPU 側カーネルも同一の演算を 1 回行うだけで
    // 複数命令を組み合わせた累積誤差の余地がないため、期待値との一致は
    // ビット完全一致で判定する（バックエンド間数値一致の複合判定
    // 「相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満」〈`backend_cpu::
    // assert_parity`〉は GEMM 等の複数命令累積を前提にした閾値であり、
    // ローカルに緩い閾値を複製しない。`.claude/rules/coding-rust.md`）。
    let mismatch = input
        .iter()
        .zip(output.iter())
        .find(|&(&x, &y)| expected(x) != y);
    match mismatch {
        None => {
            println!(
                "SETMAXNREG_PROBE_RESULT stage=execute kernel={label} arch={arch} \
                 result=success output_matches_expected=true"
            );
            probe_self_regs
        }
        Some((&x, &y)) => {
            // `result=success` は「命令が受理され実行が完走した」ことしか
            // 意味しない。ここは実行が完走した *うえで* 出力が期待値と
            // 一致しない＝レジスタ再配分によるデータ破壊が疑われる最も
            // 危険なケースであり、`docs/cuda-tensor-core-design.md` への
            // 転記運用（grep・人間による転記）で `result=success` のみを見て
            // 「使用可」と誤読されるのを防ぐため、`result=corrupted` という
            // 別ラベルで記録したうえで `panic`（テスト失敗）させる。
            // 冒頭コメントの「命令の受理可否は panic させない」方針は
            // コンパイル・ロード・起動・同期の失敗（=setmaxnreg 自体が拒否
            // された）に限った例外であり、実行が完走したのに数値が壊れて
            // いるケースまでは対象外とする（arch-accelerated 版〈`arch`
            // フィールドで判別〉も同じ扱いとし softening しない）。
            panic!(
                "SETMAXNREG_PROBE_RESULT stage=execute kernel={label} arch={arch} \
                 result=corrupted output_matches_expected=false sample_input={x} \
                 sample_output={y} sample_expected={}",
                expected(x)
            );
        }
    }
}

/// `setmaxnreg.dec` のみを発行するカーネルの実機プローブ。
///
/// `device.arch()`（実機では `compute_121` 相当）に加え、arch-accelerated
/// 版 `<arch>a`（`compute_121a`）でもコンパイル・ロード・起動・同期・
/// 出力検証まで追試する（PR #636 レビュー指摘 P2 対応。コンパイル受理
/// 可否のみでは sm_121 実機での「受理・実行可能か」を確定できないため）。
/// 両方の結果を `SETMAXNREG_PROBE_RESULT` として記録する（本ファイル冒頭
/// コメント「実機実測」節）。
///
/// `try_compile`/`try_load_and_run` の前に [`report_control_baseline_regs`]
/// で対照カーネル（[`CONTROL_DEC`]）のベースライン register/thread 数を
/// 実測し、`setmaxnreg.dec ... 64`（[`PROBE_SETMAXNREG_DEC`]）の対象値との
/// 整合性を [`report_setmaxnreg_coherence`] で判定・記録する（イシュー #484
/// レビュー指摘 Low・Medium 対応。`setmaxnreg_incdec_probe` にのみ適用
/// されていた P2 是正を dec 単体側にも揃える）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10、compute capability 12.1 相当）必須。\
            実測記録は docs/cuda-tensor-core-design.md「setmaxnreg プローブ結果（#484）」節"]
fn setmaxnreg_dec_probe() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    println!(
        "SETMAXNREG_PROBE_RESULT stage=environment name={:?} compute_capability={:?} arch={:?}",
        device.name(),
        device.compute_capability(),
        device.arch()
    );

    let arch = device.arch();
    let arch_accelerated = format!("{arch}a");

    // `setmaxnreg.dec` の対象値（64）を検証する前に、対照カーネル
    // （`setmaxnreg` を含まない同一 `__launch_bounds__(128)` 構成）の実際の
    // ベースライン register/thread 数を実測する（イシュー #484 レビュー
    // 指摘 Low 対応。`setmaxnreg_incdec_probe` の `report_control_baseline_regs`
    // 呼び出しと対称の構成にする）。`control_ok` は基準 arch 向け
    // `try_compile` へそのまま渡し、同一 `control_src`・同一 arch の
    // 再コンパイルを避ける（同レビュー Low 対応）。両 `source` とも
    // 診断参考値であり「権威値」ではない（[`report_setmaxnreg_coherence`]
    // ドキュメンテーションコメント参照。PR #636 レビュー指摘 P2 対応）。
    let (baseline_regs, control_ok) =
        report_control_baseline_regs(&device, "control_dec", CONTROL_DEC, arch);
    report_setmaxnreg_coherence(
        "probe_setmaxnreg_dec",
        "control",
        baseline_regs,
        Some(64),
        None,
    );

    // 素の compute_XY（本命シナリオ: arch-accelerated feature のため
    // 拒否される可能性が高い）。
    if let Some(ptx) = try_compile(
        "probe_setmaxnreg_dec",
        PROBE_SETMAXNREG_DEC,
        control_ok,
        arch,
    ) {
        let probe_self_regs = try_load_and_run(
            &device,
            "probe_setmaxnreg_dec",
            arch,
            ptx,
            WARPGROUP_BLOCK_DIM,
            |x| x + 1.0,
        );
        report_setmaxnreg_coherence(
            "probe_setmaxnreg_dec",
            "probe_self",
            probe_self_regs,
            Some(64),
            None,
        );
    }

    // arch-accelerated 版（`compute_XYa`）: コンパイル受理可否だけでなく、
    // 受理された場合はロード・起動・同期・出力検証まで実行する（PR #636
    // レビュー指摘 P2 対応。以前はコンパイル確認のみで `try_compile` の
    // 戻り値を捨てており、sm_121 実機での「受理・実行可能か」を確定
    // できていなかった）。カーネル内 `__global__` 関数シンボル名は基準
    // arch 版と同一〈`probe_setmaxnreg_dec`〉のため [`try_load_and_run`]
    // へ渡す `label` はそのまま再利用し、`arch` 引数に `arch_accelerated`
    // を渡すことで `SETMAXNREG_PROBE_RESULT` の行を基準 arch 版と区別する。
    // 基準 arch とは異なる arch での確認のため、`control_src` は
    // `report_control_baseline_regs` のコンパイル結果を再利用できず、
    // ここで独立にコンパイルする（イシュー #484 レビュー指摘 Low 対応の
    // 適用範囲外＝真に別入力）。
    let control_ok_accelerated = compile_ptx(CONTROL_DEC, &arch_accelerated).is_ok();
    if let Some(ptx_accelerated) = try_compile(
        "probe_setmaxnreg_dec_arch_accelerated",
        PROBE_SETMAXNREG_DEC,
        control_ok_accelerated,
        &arch_accelerated,
    ) {
        let probe_self_regs_accelerated = try_load_and_run(
            &device,
            "probe_setmaxnreg_dec",
            &arch_accelerated,
            ptx_accelerated,
            WARPGROUP_BLOCK_DIM,
            |x| x + 1.0,
        );
        report_setmaxnreg_coherence(
            "probe_setmaxnreg_dec_arch_accelerated",
            "probe_self",
            probe_self_regs_accelerated,
            Some(64),
            None,
        );
    }
}

/// producer warpgroup が `setmaxnreg.dec`、consumer warpgroup が
/// `setmaxnreg.inc` を発行する非対称パターン（[`PROBE_SETMAXNREG_INCDEC`]
/// 参照）の実機プローブ。
///
/// `setmaxnreg_dec_probe` と同様、`device.arch()` に加え arch-accelerated
/// 版（`<arch>a`）でもコンパイル・ロード・起動・同期・出力検証まで追試
/// する（PR #636 レビュー指摘 P2 対応）。B-3（タイル拡大時のレジスタ予算
/// 設計）が引き継ぐのは producer/consumer 非対称パターン（本テスト）側で
/// あり、`setmaxnreg.dec` 単体（片道）版より情報価値が高いため、dec 版の
/// 拒否予測が的中した場合に備えてこちらでも追試を欠かさない。
///
/// `try_compile`/`try_load_and_run` の前に [`report_control_baseline_regs`]
/// で対照カーネルの実際のベースライン register/thread 数を実測・記録する
/// （イシュー #484 レビュー指摘 P2 対応。`PROBE_SETMAXNREG_INCDEC` が指定する
/// dec/inc の絶対値〈24/232〉がこのベースラインに対して妥当かどうかを
/// 転記時に突き合わせられるようにするため）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10、compute capability 12.1 相当）必須。\
            実測記録は docs/cuda-tensor-core-design.md「setmaxnreg プローブ結果（#484）」節"]
fn setmaxnreg_incdec_probe() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    println!(
        "SETMAXNREG_PROBE_RESULT stage=environment name={:?} compute_capability={:?} arch={:?}",
        device.name(),
        device.compute_capability(),
        device.arch()
    );

    let arch = device.arch();
    let arch_accelerated = format!("{arch}a");

    // `setmaxnreg.dec`/`.inc` の絶対値（24/232）を検証する前に、対照カーネル
    // （`setmaxnreg` を含まない同一 `__launch_bounds__(256)` 構成）の実際の
    // ベースライン register/thread 数を実測する（イシュー #484 レビュー
    // 指摘 P2 対応。[`PROBE_SETMAXNREG_INCDEC`] ドキュメンテーションコメント
    // 参照）。実測値は `report_setmaxnreg_coherence` へ `source=control`
    // （参考値）として渡し、24/232 との整合性を `stage=coherence_check` として
    // 明示的に記録する（同レビュー Medium 対応。`result=success` のみでは
    // 判別できない偽陽性を防ぐ）。`control_ok` は基準 arch 向け `try_compile`
    // へそのまま渡し、同一 `control_src`・同一 arch の再コンパイルを避ける
    // （同レビュー Low 対応）。
    let (baseline_regs, control_ok) =
        report_control_baseline_regs(&device, "control_incdec", CONTROL_INCDEC, arch);
    report_setmaxnreg_coherence(
        "probe_setmaxnreg_incdec",
        "control",
        baseline_regs,
        Some(24),
        Some(232),
    );

    // 素の compute_XY（本命シナリオ: arch-accelerated feature のため
    // 拒否される可能性が高い）。
    if let Some(ptx) = try_compile(
        "probe_setmaxnreg_incdec",
        PROBE_SETMAXNREG_INCDEC,
        control_ok,
        arch,
    ) {
        let probe_self_regs = try_load_and_run(
            &device,
            "probe_setmaxnreg_incdec",
            arch,
            ptx,
            PRODUCER_CONSUMER_BLOCK_DIM,
            |x| x * 2.0,
        );
        // `setmaxnreg` を実際に発行するプローブ自身の静的レジスタ割り当て
        // による判定（診断参考値。`source=control` の行は toolchain 健全性
        // の参考記録として残す。イシュー #484 レビュー指摘 Medium・
        // PR #636 レビュー指摘 P2 対応）。
        report_setmaxnreg_coherence(
            "probe_setmaxnreg_incdec",
            "probe_self",
            probe_self_regs,
            Some(24),
            Some(232),
        );
    }

    // arch-accelerated 版（`compute_XYa`）: コンパイル受理可否だけでなく、
    // 受理された場合はロード・起動・同期・出力検証まで実行する
    // （`setmaxnreg_dec_probe` と同型。PR #636 レビュー指摘 P2 対応。以前は
    // コンパイル確認のみで `try_compile` の戻り値を捨てており、sm_121 実機
    // での「受理・実行可能か」を確定できていなかった）。カーネル内
    // `__global__` 関数シンボル名は基準 arch 版と同一
    // 〈`probe_setmaxnreg_incdec`〉のため [`try_load_and_run`] へ渡す
    // `label` はそのまま再利用し、`arch` 引数に `arch_accelerated` を渡す
    // ことで `SETMAXNREG_PROBE_RESULT` の行を基準 arch 版と区別する。
    // 基準 arch と異なる arch のため `control_src` を独立にコンパイルする
    // （イシュー #484 レビュー指摘 Low 対応の適用範囲外＝真に別入力）。
    let control_ok_accelerated = compile_ptx(CONTROL_INCDEC, &arch_accelerated).is_ok();
    if let Some(ptx_accelerated) = try_compile(
        "probe_setmaxnreg_incdec_arch_accelerated",
        PROBE_SETMAXNREG_INCDEC,
        control_ok_accelerated,
        &arch_accelerated,
    ) {
        let probe_self_regs_accelerated = try_load_and_run(
            &device,
            "probe_setmaxnreg_incdec",
            &arch_accelerated,
            ptx_accelerated,
            PRODUCER_CONSUMER_BLOCK_DIM,
            |x| x * 2.0,
        );
        report_setmaxnreg_coherence(
            "probe_setmaxnreg_incdec_arch_accelerated",
            "probe_self",
            probe_self_regs_accelerated,
            Some(24),
            Some(232),
        );
    }
}
