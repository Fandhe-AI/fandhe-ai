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
const PROBE_SETMAXNREG_DEC: &str = r#"
extern "C" __global__ void probe_setmaxnreg_dec(
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
const CONTROL_DEC: &str = r#"
extern "C" __global__ void control_dec(
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
fn try_compile(
    label: &str,
    src: &str,
    control_src: &str,
    arch: &str,
) -> Option<cudarc::nvrtc::Ptx> {
    let control_ok = compile_ptx(control_src, arch).is_ok();
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
fn report_control_baseline_regs(device: &CudaDevice, label: &str, control_src: &str, arch: &str) {
    let ptx = match compile_ptx(control_src, arch) {
        Ok(ptx) => ptx,
        Err(err) => {
            println!(
                "SETMAXNREG_PROBE_RESULT stage=control_baseline_regs kernel={label} arch={arch} \
                 result=failed detail={err}"
            );
            return;
        }
    };
    let module = match device.context().load_module(ptx) {
        Ok(module) => module,
        Err(err) => {
            println!(
                "SETMAXNREG_PROBE_RESULT stage=control_baseline_regs kernel={label} arch={arch} \
                 result=failed detail={err:?}"
            );
            return;
        }
    };
    let func = match module.load_function(label) {
        Ok(func) => func,
        Err(err) => {
            println!(
                "SETMAXNREG_PROBE_RESULT stage=control_baseline_regs kernel={label} arch={arch} \
                 result=failed detail={err:?}"
            );
            return;
        }
    };
    match func.num_regs() {
        Ok(num_regs) => {
            println!(
                "SETMAXNREG_PROBE_RESULT stage=control_baseline_regs kernel={label} arch={arch} \
                 result=measured num_regs_per_thread={num_regs}"
            );
        }
        Err(err) => {
            println!(
                "SETMAXNREG_PROBE_RESULT stage=control_baseline_regs kernel={label} arch={arch} \
                 result=failed detail={err:?}"
            );
        }
    }
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
/// `PROBE_SETMAXNREG_INCDEC`）内の `__global__` 関数名と必ず一致させる
/// こと。対照カーネル〈`CONTROL_DEC`/`CONTROL_INCDEC`〉や
/// arch-accelerated 版のコンパイル確認専用ラベル
/// 〈`..._arch_accelerated`〉は本関数へは渡さない＝実行しない）。
///
/// `block_dim` は呼び出し元が `src` の warpgroup 構成に合わせて指定する
/// （[`WARPGROUP_BLOCK_DIM`]＝1 warpgroup／[`PRODUCER_CONSUMER_BLOCK_DIM`]
/// ＝producer・consumer 2 warpgroup）。`n`（256 要素固定）は
/// [`PRODUCER_CONSUMER_BLOCK_DIM`] でも 1 ブロックに収まり producer・
/// consumer 双方の warpgroup が確実に起動される値として選んでいる。
fn try_load_and_run(
    device: &CudaDevice,
    label: &str,
    ptx: cudarc::nvrtc::Ptx,
    block_dim: (u32, u32, u32),
    expected: impl Fn(f32) -> f32,
) {
    let n: u32 = 256;
    let mut rng = bench_harness::rng::Xorshift64Star::new(0xC0FFEE);
    let input: Vec<f32> = rng.fill_vec(n as usize);

    let module = match device.context().load_module(ptx) {
        Ok(module) => module,
        Err(err) => {
            println!(
                "SETMAXNREG_PROBE_RESULT stage=module_load kernel={label} result=failed \
                 detail={err:?}"
            );
            return;
        }
    };
    let func = match module.load_function(label) {
        Ok(func) => func,
        Err(err) => {
            println!(
                "SETMAXNREG_PROBE_RESULT stage=load_function kernel={label} result=failed \
                 detail={err:?}"
            );
            return;
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
            "SETMAXNREG_PROBE_RESULT stage=launch kernel={label} result=failed detail={err:?}"
        );
        return;
    }

    if let Err(err) = device.stream().synchronize() {
        println!(
            "SETMAXNREG_PROBE_RESULT stage=synchronize kernel={label} result=failed \
             detail={err:?}"
        );
        return;
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
                "SETMAXNREG_PROBE_RESULT stage=execute kernel={label} result=success \
                 output_matches_expected=true"
            );
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
            // いるケースまでは対象外とする。
            panic!(
                "SETMAXNREG_PROBE_RESULT stage=execute kernel={label} result=corrupted \
                 output_matches_expected=false sample_input={x} sample_output={y} \
                 sample_expected={}",
                expected(x)
            );
        }
    }
}

/// `setmaxnreg.dec` のみを発行するカーネルの実機プローブ。
///
/// `device.arch()`（実機では `compute_121` 相当）に加え、arch-accelerated
/// 版が存在し受理されるかの参考として `<arch>a`（`compute_121a`）でも
/// コンパイルを追試する。両方の結果を `SETMAXNREG_PROBE_RESULT` として
/// 記録する（本ファイル冒頭コメント「実機実測」節）。
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

    // 素の compute_XY（本命シナリオ: arch-accelerated feature のため
    // 拒否される可能性が高い）。
    if let Some(ptx) = try_compile(
        "probe_setmaxnreg_dec",
        PROBE_SETMAXNREG_DEC,
        CONTROL_DEC,
        arch,
    ) {
        try_load_and_run(
            &device,
            "probe_setmaxnreg_dec",
            ptx,
            WARPGROUP_BLOCK_DIM,
            |x| x + 1.0,
        );
    }

    // 参考追試: arch-accelerated 版（`compute_XYa`）が NVRTC に受理される
    // かどうかも記録する（実行までは行わず受理可否のみ。受理された場合の
    // 実行検証は B-3 着手時に改めて設計する）。
    let _ = try_compile(
        "probe_setmaxnreg_dec_arch_accelerated",
        PROBE_SETMAXNREG_DEC,
        CONTROL_DEC,
        &arch_accelerated,
    );
}

/// producer warpgroup が `setmaxnreg.dec`、consumer warpgroup が
/// `setmaxnreg.inc` を発行する非対称パターン（[`PROBE_SETMAXNREG_INCDEC`]
/// 参照）の実機プローブ。
///
/// `setmaxnreg_dec_probe` と同様、`device.arch()` に加え arch-accelerated
/// 版（`<arch>a`）でのコンパイル受理可否も追試する。B-3
/// （タイル拡大時のレジスタ予算設計）が引き継ぐのは producer/consumer
/// 非対称パターン（本テスト）側であり、`setmaxnreg.dec` 単体（片道）版より
/// 情報価値が高いため、dec 版の拒否予測が的中した場合に備えてこちらでも
/// 追試を欠かさない。
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
    // 参照）。
    report_control_baseline_regs(&device, "control_incdec", CONTROL_INCDEC, arch);

    // 素の compute_XY（本命シナリオ: arch-accelerated feature のため
    // 拒否される可能性が高い）。
    if let Some(ptx) = try_compile(
        "probe_setmaxnreg_incdec",
        PROBE_SETMAXNREG_INCDEC,
        CONTROL_INCDEC,
        arch,
    ) {
        try_load_and_run(
            &device,
            "probe_setmaxnreg_incdec",
            ptx,
            PRODUCER_CONSUMER_BLOCK_DIM,
            |x| x * 2.0,
        );
    }

    // 参考追試: arch-accelerated 版（`compute_XYa`）が NVRTC に受理される
    // かどうかも記録する（`setmaxnreg_dec_probe` と同型。実行までは行わず
    // 受理可否のみ。受理された場合の実行検証は B-3 着手時に改めて設計する）。
    let _ = try_compile(
        "probe_setmaxnreg_incdec_arch_accelerated",
        PROBE_SETMAXNREG_INCDEC,
        CONTROL_INCDEC,
        &arch_accelerated,
    );
}
