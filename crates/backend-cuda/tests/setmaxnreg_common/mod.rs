//! `setmaxnreg`（warp specialization レジスタ再配分）実機プローブ 4 ファイル
//! （`setmaxnreg_probe_dec_base_real_device.rs`／`..._dec_accel_...`／
//! `..._incdec_base_...`／`..._incdec_accel_...`）が共有するカーネル定義・
//! 実行ヘルパー（イシュー #484。`tests/common/mod.rs` の Rust 標準規約に
//! 従い、`tests/` 直下の非トップレベルパスとして各ファイルから
//! `#[path = "setmaxnreg_common/mod.rs"] mod common;` で取り込む＝本ファイル
//! 自体は独立の cargo test バイナリにならない）。
//!
//! ## レビュー指摘との対応関係（トレーサビリティ）
//!
//! PR #636 の未解決 review thread 3 件（`control_pool_conserved` 相当の
//! 静的 `num_regs` ゲート撤去 2 件・同一 CUDA コンテキスト連続実行の分離
//! 1 件）はいずれもコミット `9287ac2`
//! （`fix(backend-cuda): setmaxnreg プローブを num_regs ゲート撤去・
//! プロセス分離へ再設計`）で対応済み。各スレッドが指した「985 行目」
//! 「1132 行目」「1080 行以降」は、当時 1 ファイルに集約されていた旧
//! `setmaxnreg_probe_real_device.rs`（同コミットで本 4 ファイル + 本
//! `setmaxnreg_common/mod.rs` へ分割・撤去）内の行を指しており、分割後の
//! 現行ファイル群には該当箇所自体が存在しない。以降のコミット（`eb15f68`
//! 「レビュー指摘 3 点を修正」・`2bdccff` 「kernel ラベル不一致・corrupted
//! 時の stdout ログ欠落を修正」）は同設計を維持したまま追加指摘に対応した
//! ものであり、num_regs ゲート撤去・プロセス分離の設計自体を後退させて
//! いない。
//!
//! ## ファイル分割の理由（PR #636 レビュー指摘 P2「同一 CUDA コンテキストで
//! の連続実行」対応）
//!
//! 従来は 1 テスト関数内で基準 arch（`compute_121` 相当）→
//! arch-accelerated 版（`compute_121a`）を同一 `CudaDevice`／CUDA コンテキスト
//! で連続実行していた。`setmaxnreg` が実機で `CUDA_ERROR_ILLEGAL_INSTRUCTION`
//! 等のデバイス例外を起こすと、cudarc がバインドするプライマリコンテキスト
//! がエラー状態へ遷移し、後続の呼び出し（同一プロセス内の別カーネル実行）が
//! 自身の対応状況とは無関係に失敗しうる。`tests/` 直下のトップレベル `.rs`
//! ファイルは cargo によってそれぞれ独立のテストバイナリ（＝独立プロセス）
//! としてコンパイル・実行されるため、基準 arch 版・arch-accelerated 版を
//! 別ファイルへ分離するだけで追加の自前プロセス分離機構（`std::process::
//! Command` 自己再実行等）を実装せずに真のコンテキスト独立を得られる。
//! dec 単独版・producer/consumer 非対称版も同じ理由で別ファイルとする
//! （同一プロセスでは `CudaDevice::new(0)` が同一プライマリコンテキストを
//! 指す可能性があり、テスト関数間の独立性を保証できないため）。
//!
//! ## `num_regs` は診断専用（実行ゲートに使わない。PR #636 レビュー指摘
//! P2 × 3 対応）
//!
//! 以前の実装は `CU_FUNC_ATTRIBUTE_NUM_REGS`（[`cudarc::driver::
//! CudaFunction::num_regs`]）が返す静的な register/thread 割り当て値から
//! 「`setmaxnreg.dec`/`.inc` の対象値がこの静的値に対して整合するか」を
//! 判定し、不整合なら `launch`/`synchronize` 自体を skip していた。
//! しかし `num_regs` は `cuModuleLoadData` 時点のドライバ JIT（ptxas 相当）
//! が確定させる静的な値であり、単純なプローブカーネルでは 64〜232 のような
//! `setmaxnreg` の対象値を大きく下回りやすい。その結果、有効な命令構成で
//! あっても実行経路が恒常的に skip され、本スパイクの主目的（producer/
//! consumer 非対称版の実際のロード・起動・同期・出力検証）を達成できなく
//! なっていた（イシュー #484 3 件のレビュー指摘の core）。
//!
//! 本モジュールはこの反省を踏まえ、`num_regs` を**診断ログ専用**
//! （`source=diagnostic`）に限定し、実行可否の判定には一切使わない。
//! ロードに成功したカーネルは常に起動・同期する。ハング対策は静的値からの
//! 予測ではなく、実行そのものへの**外部タイムアウト**（プロセス単位）で
//! 行う契約とする（各テストファイルの実行コマンドは
//! `timeout <秒> cargo test ... -- --ignored --nocapture` を必須とする。
//! `docs/cuda-tensor-core-design.md` §13.1「実行契約」節参照）。これにより
//! Bugbot 指摘（dec 単独プローブが `dec_ok=false` でも常に launch していた
//! 非対称な扱い）も解消する: 本モジュールは dec 単独版・producer/consumer
//! 版のいずれも「診断ログのみ・ゲートなし・外部タイムアウトで保護」という
//! 単一の一貫した契約に統一する。
//!
//! ## A03（インジェクション）対応
//!
//! カーネルソースは本ファイル内の `&'static str` コンパイル時定数のみを
//! 使い、外部入力・環境変数をソース文字列へ連結しない
//! （`nvrtc.rs` の既存契約と同じ方針。`.claude/rules/security.md`）。
//!
//! ## `#![allow(dead_code)]` の理由
//!
//! 本モジュールは `#[path]` 経由で 4 つの独立バイナリへ**丸ごと**取り込ま
//! れるが、各バイナリは dec 単独版／producer・consumer 非対称版のいずれか
//! 一方の定数・ヘルパーしか使わない（例: dec 単独版バイナリは
//! `PROBE_SETMAXNREG_INCDEC`／`CONTROL_INCDEC`／
//! `PRODUCER_CONSUMER_BLOCK_DIM` を参照しない）。バイナリごとの未使用側は
//! `dead_code` lint の対象になるが、シナリオごとに本モジュールを分割する
//! と共有ロジック（`try_compile`／`try_load_and_run` 等）が 2 重管理になり
//! `.claude/rules/code-comment-style.md`「陳腐化しやすい実装詳細の重複を
//! 避ける」方針に反する。共有モジュールの意図的な設計上のトレードオフと
//! して本 lint に限り無効化する（`.claude/rules/coding-rust.md` が禁じる
//! 「安易な `#[allow]`」には当たらない）。

#![allow(dead_code)]

use cudarc::driver::{LaunchConfig, PushKernelArg};
use fandhe_ai_backend_cuda::{CudaDevice, CudaError, compile_ptx};

/// warpgroup（4 warp = 128 スレッド）1 個の起動を前提に、`setmaxnreg.dec`
/// のみを発行してから出力へ書き込むカーネル。
///
/// PTX ISA 上、`setmaxnreg` は warpgroup 全体が同一命令列を実行することを
/// 要求するため、ブロック次元は 128 の倍数（かつ本プローブでは 1 warpgroup
/// のみで足りるため 128 固定）とする。手動境界チェック（`if (idx < n)`）は
/// REQ-8（`.claude/rules/coding-rust.md`）に従い、プローブ用途であっても
/// 省略しない。
pub const PROBE_SETMAXNREG_DEC: &str = r#"
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
/// `warpgroup_reg_alloc` に相当する使い方を模す）。
///
/// PTX ISA 上 `setmaxnreg` は warpgroup（連続 128 スレッド）単位で全
/// スレッドが同一命令列を実行することを要求するため、producer/consumer の
/// 非対称性は**warpgroup 単位**で表現する。本カーネルは 1 ブロック =
/// 2 warpgroup（256 スレッド。[`PRODUCER_CONSUMER_BLOCK_DIM`]）で起動し、
/// `threadIdx.x / 128` で判定した warpgroup 0 を producer（`setmaxnreg.dec`
/// のみを発行してレジスタ予算を解放する側）、warpgroup 1 を consumer
/// （`setmaxnreg.inc` のみを発行して producer が解放した分を確保する側）
/// とする。
///
/// dec/inc の値（24 / 232。8 の倍数・[24, 256] の許容範囲内）は CUTLASS の
/// 一般的な producer/consumer 値域を踏襲したものである。この値がベース
/// ラインの静的レジスタ割り当てに対して ISA 上妥当かどうかは
/// **実測でしか確定できない**（本ファイル冒頭コメント「`num_regs` は診断
/// 専用」節）。本プローブはこれを実行前のゲートで判定・skip するのではなく、
/// 実測値を `source=diagnostic` として記録したうえで常に実行し、真の成否
/// （命令拒否／ハング／実行完走）を実機自身に語らせる設計とする。
pub const PROBE_SETMAXNREG_INCDEC: &str = r#"
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
///
/// `compile_ptx` の失敗は「`setmaxnreg` 命令自体が拒否された」以外に、
/// `libnvrtc` 不在（`CudaError::NvrtcUnavailable`）や include パス解決
/// 失敗等、命令と無関係な理由でも起こりうる。同一 arch に対する対照
/// カーネルのコンパイル成否を併記することで、プローブ結果を
/// 「コンパイル基盤は健全だが setmaxnreg のみ拒否された」とそれ以外
/// （toolchain 自体の問題で判定不能）を読み手が区別できるようにする。
pub const CONTROL_DEC: &str = r#"
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
pub const CONTROL_INCDEC: &str = r#"
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

/// 1 warpgroup 分のブロック次元（128 スレッド = 4 warp）。
pub const WARPGROUP_BLOCK_DIM: (u32, u32, u32) = (128, 1, 1);

/// producer/consumer 2 warpgroup 分のブロック次元（256 スレッド = 8 warp）。
pub const PRODUCER_CONSUMER_BLOCK_DIM: (u32, u32, u32) = (256, 1, 1);

pub fn launch_config(n: u32, block_dim: (u32, u32, u32)) -> LaunchConfig {
    let grid_x = n.div_ceil(block_dim.0);
    LaunchConfig {
        grid_dim: (grid_x, 1, 1),
        block_dim,
        shared_mem_bytes: 0,
    }
}

/// 実機環境（device 名・compute capability・device 基準 arch）を
/// `SETMAXNREG_PROBE_RESULT` 形式で記録する。各テストファイルの冒頭で
/// 呼ぶ（分割後もファイルごとに arch 情報を出力し、ログ単体から実行環境が
/// 分かるようにする）。
///
/// フィールド名はあえて `arch` ではなく `device_arch` にしている:
/// arch-accelerated 版（`*_accel_real_device.rs`）では、本関数は
/// `device.arch()`（例: `compute_121`）という**基準** arch を報告する一方、
/// 同ファイルの後続ログ行（`nvrtc_compile`/`execute` 等）は実際に使う
/// `<arch>a`（例: `compute_121a`）を `arch=` で記録する。同じログ内で
/// `arch=` が異なる値を指す（基準 arch と実効 arch の混在）と docs 転記時に
/// 誤転記を招くため、本関数側のフィールド名を分離して曖昧さを排除する
/// （PR #636 レビュー指摘対応）。
pub fn report_environment(device: &CudaDevice) {
    println!(
        "SETMAXNREG_PROBE_RESULT stage=environment name={:?} compute_capability={:?} device_arch={:?}",
        device.name(),
        device.compute_capability(),
        device.arch()
    );
}

/// `control_src`（`setmaxnreg` を含まない対照カーネル）を先にコンパイル
/// したうえで `src`（`setmaxnreg` を含むプローブ本体）を NVRTC コンパイル
/// し、`SETMAXNREG_PROBE_RESULT` 行として記録する。失敗は `panic` させず
/// `None` を返す（「使えないという結論も spike の正当な成果」方針。各
/// テストファイル冒頭コメント参照）。
///
/// `result` の判定基準:
/// - `accepted`: `src` のコンパイルが成功
/// - `rejected`: 対照カーネルは成功したが `src` のみ失敗
///   （＝ `setmaxnreg` 命令自体が拒否されたと結論できる）
/// - `inconclusive`: 対照カーネルも失敗、または `libnvrtc` 不在
///   （`CudaError::NvrtcUnavailable`）。toolchain 側の問題であり
///   `setmaxnreg` の可否について何も結論できない
pub fn try_compile(
    label: &str,
    src: &str,
    control_ok: bool,
    arch: &str,
) -> Option<cudarc::nvrtc::Ptx> {
    // `control_ok` は `report_control_baseline_regs` が別途コンパイルした
    // 対照カーネルの成否を呼び出し元から受け取って転記するのみで、この
    // 行自体は対照カーネルを再コンパイルしていない（本行は `try_compile`
    // 開始時点の context ログ）。stage 名を `nvrtc_compile_control` にすると
    // 「ここで対照カーネルをコンパイルした」ように読めてしまうため、
    // 独立した control 判定に依存しない stage 名にしている
    // （PR #636 レビュー指摘対応）。
    println!(
        "SETMAXNREG_PROBE_RESULT stage=nvrtc_compile_start kernel={label} arch={arch} \
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
/// **診断目的のみ**で実測して `SETMAXNREG_PROBE_RESULT` 行として記録する
/// （本ファイル冒頭コメント「`num_regs` は診断専用」節）。
///
/// カーネルを起動しない点が実行ヘルパーと異なる: レジスタ割り当ては
/// `cuModuleLoadData` 時点のドライバ JIT（ptxas 相当）で確定するため、
/// `load_function` までで実測に十分であり、実行（launch/synchronize）は
/// 不要（かつ対照カーネルは `setmaxnreg` を含まないため実行結果自体に
/// 検証価値がない）。
///
/// 戻り値は `bool`（`control_src` のコンパイル成否）のみで、`num_regs` の
/// 実測値そのものは呼び出し元へ返さない（診断ログとしてのみ用い、以前の
/// 実装のように呼び出し元がこれを実行可否ゲートへ転用できないようにする。
/// イシュー #484 レビュー指摘 P2 × 3 対応）。
pub fn report_control_baseline_regs(
    device: &CudaDevice,
    label: &str,
    control_src: &str,
    arch: &str,
) -> bool {
    let ptx = match compile_ptx(control_src, arch) {
        Ok(ptx) => ptx,
        Err(err) => {
            println!(
                "SETMAXNREG_PROBE_RESULT stage=control_baseline_regs kernel={label} arch={arch} \
                 source=diagnostic result=failed detail={err}"
            );
            return false;
        }
    };
    let module = match device.context().load_module(ptx) {
        Ok(module) => module,
        Err(err) => {
            println!(
                "SETMAXNREG_PROBE_RESULT stage=control_baseline_regs kernel={label} arch={arch} \
                 source=diagnostic result=failed detail={err:?}"
            );
            return true;
        }
    };
    let func = match module.load_function(label) {
        Ok(func) => func,
        Err(err) => {
            println!(
                "SETMAXNREG_PROBE_RESULT stage=control_baseline_regs kernel={label} arch={arch} \
                 source=diagnostic result=failed detail={err:?}"
            );
            return true;
        }
    };
    match func.num_regs() {
        Ok(num_regs) => println!(
            "SETMAXNREG_PROBE_RESULT stage=control_baseline_regs kernel={label} arch={arch} \
             source=diagnostic result=measured num_regs_per_thread={num_regs}"
        ),
        Err(err) => println!(
            "SETMAXNREG_PROBE_RESULT stage=control_baseline_regs kernel={label} arch={arch} \
             source=diagnostic result=failed detail={err:?}"
        ),
    }
    true
}

/// コンパイル成功済みカーネルをロード・起動・同期し、成否を記録する。
///
/// **ゲートなし（イシュー #484 レビュー指摘 P2 × 3・Bugbot Medium 対応）**:
/// ロードに成功したカーネルは `num_regs` の実測値に関わらず常に起動・同期
/// する。以前の実装は静的な `num_regs` からハングの可能性を予測して
/// `launch`/`synchronize` 自体を skip していたが、この静的値は
/// `setmaxnreg` が操作する動的なレジスタ予算そのものではなく、単純な
/// プローブでは恒常的に skip が発生し本スパイクの主目的（producer/consumer
/// 非対称版の実際の実行検証）を達成できなくなっていた（本ファイル冒頭
/// コメント参照）。ハング対策は本関数内のゲートではなく、呼び出し元
/// （各テストファイル）を **外部タイムアウト**（`timeout <秒> cargo
/// test ...`）付きで実行する運用契約に委ねる（`docs/cuda-tensor-core-
/// design.md` §13.1「実行契約」節）。
///
/// `setmaxnreg` が実機で受理されない場合（`CUDA_ERROR_ILLEGAL_INSTRUCTION`
/// 等）も `panic` させず結果として記録する。実行成功時は出力バッファを
/// 回収し「命令が実行を破壊していないか」の期待値検証も行う。
///
/// `label` は診断ログ用のラベルであると同時に、`module.load_function(label)`
/// へそのまま渡す **CUDA 側 `extern "C"` 関数シンボル名**でもある。
///
/// `arch` は診断ログ専用（`load_function` には渡さない）。
///
/// `result=corrupted`（実行完走したが出力が期待値と不一致）のみ `panic`
/// させる: 出力破壊は「命令の受理可否」の範疇を超えた危険なシグナルであり
/// softening しない。
pub fn try_load_and_run(
    device: &CudaDevice,
    ptx: cudarc::nvrtc::Ptx,
    label: &str,
    arch: &str,
    block_dim: (u32, u32, u32),
    expected: impl Fn(f32) -> f32,
) {
    let module = match device.context().load_module(ptx) {
        Ok(module) => module,
        Err(err) => {
            println!(
                "SETMAXNREG_PROBE_RESULT stage=module_load kernel={label} arch={arch} \
                 result=failed detail={err:?}"
            );
            return;
        }
    };
    let func = match module.load_function(label) {
        Ok(func) => func,
        Err(err) => {
            println!(
                "SETMAXNREG_PROBE_RESULT stage=load_function kernel={label} arch={arch} \
                 result=failed detail={err:?}"
            );
            return;
        }
    };

    // `setmaxnreg` を実際に発行するこのプローブカーネル自身の静的レジスタ
    // 割り当てを実測して記録する（`source=diagnostic`。実行可否には使わ
    // ない。本ファイル冒頭コメント参照）。
    match func.num_regs() {
        Ok(num_regs) => println!(
            "SETMAXNREG_PROBE_RESULT stage=probe_self_regs kernel={label} arch={arch} \
             source=diagnostic result=measured num_regs_per_thread={num_regs}"
        ),
        Err(err) => println!(
            "SETMAXNREG_PROBE_RESULT stage=probe_self_regs kernel={label} arch={arch} \
             source=diagnostic result=failed detail={err:?}"
        ),
    }

    let n: u32 = 256;
    let mut rng = bench_harness::rng::Xorshift64Star::new(0xC0FFEE);
    let input: Vec<f32> = rng.fill_vec(n as usize);

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
    // `PROBE_SETMAXNREG_DEC`/`PROBE_SETMAXNREG_INCDEC` 参照。REQ-8）と
    // 合わせて OOB 読み書きが起きない根拠とする。`setmaxnreg` 命令自体が
    // 受理・実行されるかは本プローブが検証する対象そのものであり、実行時
    // エラー（`CUDA_ERROR_ILLEGAL_INSTRUCTION` 等）は `launch`/
    // `synchronize` の `Result::Err` として捕捉し `panic` させない。
    // ハング自体はここではなく呼び出し元プロセス外の `timeout` に委ねる
    // （本関数ドキュメンテーションコメント参照）。
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
        return;
    }

    if let Err(err) = device.stream().synchronize() {
        println!(
            "SETMAXNREG_PROBE_RESULT stage=synchronize kernel={label} arch={arch} \
             result=failed detail={err:?}"
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
    // 「相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満」〈`fandhe_ai_backend_cpu::
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
        }
        Some((&x, &y)) => {
            // `result=success` は「命令が受理され実行が完走した」ことしか
            // 意味しない。ここは実行が完走した *うえで* 出力が期待値と
            // 一致しない＝レジスタ再配分によるデータ破壊が疑われる最も
            // 危険なケースであり、`docs/cuda-tensor-core-design.md` への
            // 転記運用（grep・人間による転記）で `result=success` のみを見て
            // 「使用可」と誤読されるのを防ぐため、`result=corrupted` という
            // 別ラベルで記録したうえで `panic`（テスト失敗）させる。
            //
            // `panic!` のメッセージは既定でテストハーネスの標準エラー
            // （stderr）に出力され、`success`／ソフト失敗（`--nocapture`
            // 付き `println!`）と異なり stdout には現れない。運用上
            // `SETMAXNREG_PROBE_RESULT` 行を stdout のみから grep で
            // 突合する場合、最も危険な `corrupted` 結果だけが取りこぼされる
            // （PR #636 レビュー指摘対応）。そのため `panic!` する前に
            // 同じペイロードを `println!` で stdout へも明示的に記録する。
            let expected_y = expected(x);
            println!(
                "SETMAXNREG_PROBE_RESULT stage=execute kernel={label} arch={arch} \
                 result=corrupted output_matches_expected=false sample_input={x} \
                 sample_output={y} sample_expected={expected_y}"
            );
            panic!(
                "SETMAXNREG_PROBE_RESULT stage=execute kernel={label} arch={arch} \
                 result=corrupted output_matches_expected=false sample_input={x} \
                 sample_output={y} sample_expected={expected_y}"
            );
        }
    }
}
