//! イシュー #482（親 #480 A-2）: sm_121 実機のデバイス属性・L1/L2 実効帯域を
//! 実測記録するための spike バイナリ。
//!
//! ## 背景・なぜ本バイナリが必要か
//!
//! 後続の CUDA GEMM 最適化（Phase C: 共有メモリ予算からのパイプライン段数
//! 逆算・#521／タイル候補列挙・#524／L2 スウィズル・#499 等）は、DeepGEMM
//! 型の「SMEM 予算逆算」「L1/L2 帯域コストモデル」を前提とする。DeepGEMM が
//! 持つ定数は Hopper（SM90）固有値（smem 容量 232448 バイト・L2/L1 帯域の
//! per-cycle モデル。参照: DeepGEMM `csrc/jit_kernels/heuristics/sm90.hpp`
//! の 14 行付近・201-238 行付近）であり、DGX Spark GB10（sm_121）へは
//! そのまま流用できない。本バイナリは実装変更を伴わない **spike（計測・
//! 記録）** であり、sm_121 実機のデバイス属性と L1/L2 実効帯域を実測し、
//! C-8/C-9 系タスクが参照できるコストモデル定数として
//! `docs/perf/sm121-device-attributes.md` へ記録する材料を出力する。
//!
//! ## 構成
//!
//! 1. デバイス属性ダンプ（SMEM/SM・SM 数・レジスタ・L2 サイズ・クロック等）
//! 2. L1/L2/global メモリの実効帯域マイクロベンチ（grid-stride コピー
//!    カーネル。5 回以上計測の中央値。`.claude/rules/coding-rust.md`）
//!
//! `examples/` に置くのは、通常の `cargo test`／CI では実行されずビルド
//! 検証（`cargo build --workspace --all-targets`）のみが CI で走るように
//! するため（`cuda_floor_bench.rs` と同じ判断。`.claude/rules/ci.md`）。
//! CUDA 非搭載環境では `CudaDevice::new` の失敗を検出した時点でスキップ
//! 終了する（`unwrap()`/`expect()` は本番経路に置かない。
//! `.claude/rules/coding-rust.md`）。
//!
//! ## 実行手順
//!
//! ```sh
//! cargo run -p backend-cuda --example device_attributes_dump --release
//! ```
//!
//! 出力（属性名・実測値・単位換算値、帯域 GB/s・bytes/cycle）を
//! `docs/perf/sm121-device-attributes.md` の表へ転記する。実機接続不能な
//! 環境では属性・帯域ともに取得できず「skipping」のみが出力される
//! （実測記録は別途実機実行が必要。イシュー #482 実装計画 §4 Step 3）。
//!
//! ## L1 帯域計測の限界（安全側フォールバック）
//!
//! 受入基準（#482）は「L1/L2 の実効帯域について、マイクロベンチ実測値
//! **または** スペック値＋出典を記録する」ことを許容している。SM 単体を
//! 占有して L1 のみを計測する信頼できるマイクロベンチはウォームアップ・
//! 占有率制御など実装コストが高く 4h 見積を圧迫するため、本バイナリは
//! global／L2 相当（デバイス側 L2 キャッシュサイズ未満のバッファに対する
//! 繰り返しアクセス）の 2 段のみを実測し、L1 は
//! `docs/perf/sm121-device-attributes.md` 側でスペック由来の値（出典明記）
//! として記録する方針とする（イシュー #482 実装計画 §4 Step 2 の安全側
//! フォールバック）。

use backend_cuda::{CudaDevice, CudaError, compile_ptx};
use bench_harness::{MeasurementConfig, run as bench_run};
use cudarc::driver::sys::CUdevice_attribute;
use cudarc::driver::{CudaFunction, LaunchConfig, PushKernelArg};

/// grid-stride の単純コピー（+1）カーネル。読み出し帯域・書き込み帯域を
/// 両方含めて 1 回の反復（内側ループ 1 周）で 2N 要素分のメモリトラフィック
/// を生成する（STREAM ベンチマークの `copy` 相当）。
///
/// `repeats` 引数でカーネル**内部**で同じコピーを繰り返す（外側ループ）。
/// カーネル起動そのものは呼び出し側で 1 回のみ行う設計とすることで、
/// 起動オーバーヘッドを `repeats` に依らず定数（起動 1 回分）に固定し、
/// 転送時間だけを `repeats` 倍に伸ばして相対的に希釈する（イシュー #482
/// codex-review 指摘: 呼び出し側で `repeats` 回連続 `launch` する旧方式は
/// 最後の `synchronize()` の固定コストは償却できるが、起動そのものの
/// コスト〈GPU 側キュー投入・カーネルディスパッチ〉は起動回数に比例して
/// 発生し続けるため、小サイズ計測〈L2 常駐バッファ〉では償却できていな
/// かった。`measure_bandwidth_secs` のドキュメンテーションコメント参照）。
///
/// 手動境界チェック（内側ループの `i < n` 条件）は最適化・計測目的を
/// 理由に省略しない（REQ-8。`.claude/rules/coding-rust.md`「カーネル実装の
/// 境界検査」は計測用カーネルにも適用する。イシュー #482 実装計画
/// §2 共通契約）。
const BW_COPY_F32: &str = r#"
extern "C" __global__ void bw_copy_f32(const float* __restrict__ src, float* __restrict__ dst, int n, int repeats) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int r = 0; r < repeats; ++r) {
        for (int i = idx; i < n; i += stride) {
            dst[i] = src[i] + 1.0f;
        }
    }
}
"#;

/// 起動 1 回あたりのブロック内スレッド数（帯域計測カーネルは演算特性が
/// 単純なため、GEMM カーネル群〈`gemm.rs`〉のようなタイルサイズ制約を
/// 持たない。一般的な occupancy 確保値として 256 を用いる）。
const BW_BLOCK_THREADS: u32 = 256;

/// 1 計測サンプルあたりの「カーネル内部での」コピー繰り返し回数
/// （`bw_copy_f32` の `repeats` 引数へ渡す）。L2 常駐サイズの小バッファは
/// カーネル実行そのものが数マイクロ秒程度になり、`stream.synchronize()`
/// 込みの 1 回計測ではカーネル起動・同期のオーバーヘッドが支配的になって
/// 帯域ではなくオーバーヘッドを測ってしまう（イシュー #482 Review 指摘:
/// RTX 3060 動作検証で `l2` 実効帯域〈190.42 GB/s〉が `global` 実効帯域
/// 〈324.26 GB/s〉を下回った事象。L2 常駐コピーが global より遅いのは
/// 物理的にありえず、median_secs=6µs という極小区間では起動＋同期コスト
/// が支配的だった）。
///
/// **旧方式の限界（イシュー #482 codex-review 指摘。PR #635）**: 当初は
/// 呼び出し側で `launch` を `BW_LAUNCH_REPEATS` 回連続実行してから 1 回
/// だけ `synchronize()` する方式だったが、これは `synchronize()` の
/// 固定コストしか償却できず、カーネル起動そのもの（GPU 側キュー投入・
/// ディスパッチ）のコストは起動回数に比例して発生し続けるため、小サイズ
/// 計測では依然として起動オーバーヘッドが支配的になり得た。現方式は
/// `repeats` をカーネル**内部**の外側ループへ渡し、呼び出し側の起動は
/// 常に 1 回のみとすることで、起動コストを `repeats` に依らない定数
/// （起動 1 回分）に固定し、転送時間だけを `repeats` 倍に伸ばして
/// 相対的に希釈する（`measure_bandwidth_secs` が返す秒数は計測した秒数を
/// `BW_LAUNCH_REPEATS` で割った「反復 1 回あたり」の値）。global
/// （256 MiB）側は元々転送時間がオーバーヘッドを大きく上回るため実測値
/// への影響は小さいが、両区分で同一の計測境界を使うことで比較可能性を
/// 保つ。
const BW_LAUNCH_REPEATS: u32 = 64;

/// grid-stride ループでバッファ全体を確実にカバーしつつ、SM 数に応じて
/// 十分な並列度（1 SM あたり複数ブロック）を確保するグリッド次元を返す。
/// `n` 要素を `BW_BLOCK_THREADS` で割った必要ブロック数と、SM 数ベースの
/// 占有率目安（`sm_count * 32`）の小さい方を採る
/// （バッファが小さい L2 計測時にブロック数だけが過大になるのを防ぐ）。
fn bw_grid_dim(n: usize, sm_count: u32) -> u32 {
    let needed = (n as u32).div_ceil(BW_BLOCK_THREADS);
    needed.min(sm_count.saturating_mul(32).max(1))
}

/// `BW_COPY_F32` を `arch` 向けにコンパイル・ロードする。
fn compile_bw_copy(device: &CudaDevice, arch: &str) -> Result<CudaFunction, CudaError> {
    let ptx = compile_ptx(BW_COPY_F32, arch)?;
    let func = device
        .context()
        .load_module(ptx)?
        .load_function("bw_copy_f32")?;
    Ok(func)
}

/// `n` 要素（f32）の src/dst バッファに対して `bw_copy_f32` を
/// `bench_harness::run`（warmup 20 回・計測 20 回以上・中央値/Q1/Q3。
/// `.claude/rules/coding-rust.md`「ベンチは 5 回計測の中央値」の下限を
/// 満たす）で計測する。1 サンプルは **カーネル起動 1 回**（`repeats`
/// 引数に `BW_LAUNCH_REPEATS` を渡し、コピーの反復はカーネル内部の
/// 外側ループで行う）+ 1 回同期のため、返す秒数は計測した中央値を
/// `BW_LAUNCH_REPEATS` で割った**反復 1 回あたり**の正規化値
/// （`bandwidth_gbps` が前提とする「1 回の反復で 2N 要素分のトラフィック」
/// という単位に揃えるため）。起動そのものが 1 回のみのため起動オーバー
/// ヘッドは `repeats` に依らない定数として計測秒数へ加算され、反復回数を
/// 増やすほど転送時間に対する相対比率が小さくなる（定数 `BW_LAUNCH_REPEATS`
/// のドキュメンテーションコメント参照。イシュー #482 codex-review 指摘。
/// PR #635）。バッファの初期値は帯域計測の正当性に無関係（読み出し値を
/// そのまま書き戻すだけ）のため `alloc_zeros` で確保し、ホスト⇔デバイス
/// 転送は計測区間の外に置く（`cuda_floor_bench.rs` と同じ「計測境界の
/// 統一」方針）。
fn measure_bandwidth_secs(
    device: &CudaDevice,
    func: &CudaFunction,
    n: usize,
    sm_count: u32,
) -> Result<f64, CudaError> {
    let stream = device.stream();
    let src = stream.alloc_zeros::<f32>(n)?;
    let mut dst = stream.alloc_zeros::<f32>(n)?;
    let n_i = n as i32;
    let repeats_i = BW_LAUNCH_REPEATS as i32;
    let cfg = LaunchConfig {
        grid_dim: (bw_grid_dim(n, sm_count), 1, 1),
        block_dim: (BW_BLOCK_THREADS, 1, 1),
        shared_mem_bytes: 0,
    };

    let measurement = bench_run(&MeasurementConfig::default(), || {
        // カーネル起動は 1 回のみ行い、コピーの `BW_LAUNCH_REPEATS` 回反復は
        // カーネル内部の外側ループ（`repeats` 引数）に委ねる。呼び出し側で
        // 複数回 `launch` する旧方式は起動コストそのものを償却できなかった
        // ため（定数 `BW_LAUNCH_REPEATS` のドキュメンテーションコメント
        // 参照）、起動 1 回・同期 1 回に統一する。
        //
        // SAFETY: `src`/`dst` は直前に `n` 要素で確保済みで `n_i == n`
        // （usize→i32 変換は `n` が呼び出し元でバッファ確保に使った値と
        // 同一のためオーバーフローしない）。カーネル側の内側ループは
        // `i < n` を毎回検査するため（`BW_COPY_F32` 定義参照）、起動側の
        // グリッド構成に関わらず OOB アクセスは発生しない。`repeats` 回の
        // カーネル内反復は `dst[i] = src[i] + 1.0f` を毎回同じ入力に対して
        // 計算し直すだけなので数値的にも安全（`src` は不変・`dst` は毎回
        // 同じ値へ上書きされる）。
        unsafe {
            stream
                .launch_builder(func)
                .arg(&src)
                .arg(&mut dst)
                .arg(&n_i)
                .arg(&repeats_i)
                .launch(cfg)
                .expect("bw_copy_f32 launch must succeed on CUDA-equipped runner");
        }
        // カーネル起動は非同期のため、`synchronize()` を計測区間（このクロージャ）
        // 内に含めないと `Instant` の計測が起動オーバーヘッドのみを捉え、実行完了を
        // 待たない見かけ上の帯域（実測で 1 桁以上過大な値）になる
        // （`gemm.rs::launch_tiled_f32` 等の GEMM 起動 API が内部で同期する契約と
        // 同じ理由。PyTorch 参照計測境界を踏襲する `cuda_floor_bench.rs` の
        // 「計測境界の統一」と同様、GPU 実行完了までを計測区間に含める）。
        stream
            .synchronize()
            .expect("stream synchronize must succeed on CUDA-equipped runner");
    })
    .expect("MeasurementConfig::default satisfies the 20/20 lower bound");
    // カーネル内で `BW_LAUNCH_REPEATS` 回分の反復を実行した 1 起動分の
    // 秒数を計測しているため、反復 1 回あたり（= `bandwidth_gbps` が
    // 前提とする 2N 要素分のトラフィック 1 回分）の秒数へ正規化して返す。
    Ok(measurement.median_secs / f64::from(BW_LAUNCH_REPEATS))
}

/// 2N 要素分（読み出し N・書き込み N）の f32 トラフィックから実効帯域
/// （GB/s）を算出する。
fn bandwidth_gbps(n: usize, secs: f64) -> f64 {
    let bytes = 2.0 * (n as f64) * (std::mem::size_of::<f32>() as f64);
    bytes / secs / 1e9
}

/// clock_rate（kHz。`CU_DEVICE_ATTRIBUTE_CLOCK_RATE`）を用いて GB/s を
/// bytes/cycle へ換算する（DeepGEMM の `*_bandwidth_per_cycle` 定数群と
/// 同じ単位に揃えるため。イシュー #482 実装計画 §4 Step 2）。
fn bytes_per_cycle(gbps: f64, clock_khz: i32) -> f64 {
    let hz = (clock_khz as f64) * 1e3;
    if hz <= 0.0 {
        return f64::NAN;
    }
    (gbps * 1e9) / hz
}

fn print_attr(device: &CudaDevice, label: &str, attr: CUdevice_attribute) {
    match device.context().attribute(attr) {
        Ok(v) => println!("  {label} = {v}"),
        Err(e) => println!("  {label} = <error: {e:?}>"),
    }
}

fn main() {
    let device = match CudaDevice::new(0) {
        Ok(dev) => dev,
        Err(CudaError::DriverUnavailable { detail }) => {
            println!(
                "backend-cuda device_attributes_dump: CUDA driver unavailable ({detail}); skipping."
            );
            return;
        }
        Err(other) => {
            println!(
                "backend-cuda device_attributes_dump: CudaDevice::new failed ({other}); skipping."
            );
            return;
        }
    };

    println!(
        "device: name={} compute_capability={:?} arch={}",
        device.name(),
        device.compute_capability(),
        device.arch()
    );
    if let Ok(total) = device.context().total_mem() {
        println!(
            "  total_memory_bytes = {total} ({:.2} GiB)",
            total as f64 / (1024.0 * 1024.0 * 1024.0)
        );
    }

    println!("--- device attributes (docs/perf/sm121-device-attributes.md へ転記) ---");
    use CUdevice_attribute as A;
    print_attr(
        &device,
        "MAX_SHARED_MEMORY_PER_BLOCK_OPTIN",
        A::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
    );
    print_attr(
        &device,
        "MAX_SHARED_MEMORY_PER_BLOCK",
        A::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
    );
    print_attr(
        &device,
        "MAX_SHARED_MEMORY_PER_MULTIPROCESSOR",
        A::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR,
    );
    print_attr(
        &device,
        "RESERVED_SHARED_MEMORY_PER_BLOCK",
        A::CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK,
    );
    print_attr(
        &device,
        "MULTIPROCESSOR_COUNT",
        A::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
    );
    print_attr(
        &device,
        "MAX_REGISTERS_PER_MULTIPROCESSOR",
        A::CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR,
    );
    print_attr(
        &device,
        "MAX_REGISTERS_PER_BLOCK",
        A::CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK,
    );
    print_attr(
        &device,
        "L2_CACHE_SIZE",
        A::CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE,
    );
    print_attr(&device, "CLOCK_RATE", A::CU_DEVICE_ATTRIBUTE_CLOCK_RATE);
    print_attr(
        &device,
        "MEMORY_CLOCK_RATE",
        A::CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE,
    );
    print_attr(
        &device,
        "GLOBAL_MEMORY_BUS_WIDTH",
        A::CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH,
    );
    print_attr(
        &device,
        "MAX_THREADS_PER_MULTIPROCESSOR",
        A::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
    );
    print_attr(
        &device,
        "MAX_THREADS_PER_BLOCK",
        A::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
    );

    let sm_count = device
        .context()
        .attribute(A::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
        .ok()
        .and_then(|v| u32::try_from(v).ok())
        .unwrap_or(1);
    let l2_cache_bytes = device
        .context()
        .attribute(A::CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE)
        .ok()
        .and_then(|v| u32::try_from(v).ok());
    let clock_khz = device
        .context()
        .attribute(A::CU_DEVICE_ATTRIBUTE_CLOCK_RATE)
        .unwrap_or(0);

    println!("--- L1/L2/global 実効帯域マイクロベンチ ---");
    let func = match compile_bw_copy(&device, device.arch()) {
        Ok(f) => f,
        Err(e) => {
            println!(
                "bw_copy_f32 kernel unavailable ({e}); bandwidth measurements skipped. \
                 See module doc \"L1 帯域計測の限界\" for the spec-value fallback policy."
            );
            return;
        }
    };

    // global: L2 サイズを大きく超えるバッファ（256 MiB/バッファ、src+dst
    // 合計 512 MiB）で計測し、L2 非依存の参照帯域とする。
    let global_n: usize = 64 * 1024 * 1024; // 256 MiB (f32 4 bytes)
    match measure_bandwidth_secs(&device, &func, global_n, sm_count) {
        Ok(secs) => {
            let gbps = bandwidth_gbps(global_n, secs);
            println!(
                "global: n={global_n} secs_per_launch={secs:.6} bandwidth={gbps:.2} GB/s \
                 bytes_per_cycle={:.4}",
                bytes_per_cycle(gbps, clock_khz)
            );
        }
        Err(e) => println!("global bandwidth measurement failed: {e}"),
    }

    // L2: L2_CACHE_SIZE 未満に src+dst の合計が収まるようバッファを
    // `l2_bytes / 4`（f32 要素数。src・dst 双方を L2 に収めるため 1/2 では
    // なくさらに余裕を持たせる）に抑える。属性取得に失敗した場合は
    // global より 1 桁小さい固定値へフォールバックする（fail-soft。
    // 属性ダンプ自体の失敗は上のセクションで既に可視化済みのため、ここでは
    // 帯域計測を継続する）。
    let (l2_n, l2_size_is_fallback): (usize, bool) = match l2_cache_bytes {
        Some(bytes) if bytes > 0 => (
            ((bytes as usize) / 4 / std::mem::size_of::<f32>()).max(1024),
            false,
        ),
        _ => {
            println!(
                "WARNING: L2_CACHE_SIZE attribute unavailable; falling back to a fixed small \
                 buffer size for the L2 measurement (result may not reflect actual L2 residency)."
            );
            (global_n / 16, true)
        }
    };
    match measure_bandwidth_secs(&device, &func, l2_n, sm_count) {
        Ok(secs) => {
            let gbps = bandwidth_gbps(l2_n, secs);
            // `l2_size_is_fallback` の場合、バッファサイズは L2_CACHE_SIZE
            // 実測値ではなく固定フォールバック値（`global_n / 16`）であり
            // L2 常駐を保証しない。そのままラベル "l2:" で出力すると
            // `docs/perf/sm121-device-attributes.md` の表へ誤って L2 実測値
            // として転記される危険があるため（イシュー #482 Review 指摘）、
            // ラベルを明確に区別し転記不可であることを明示する。
            let label = if l2_size_is_fallback {
                "l2_FALLBACK_SIZE_UNRELIABLE_DO_NOT_TRANSCRIBE"
            } else {
                "l2"
            };
            println!(
                "{label}: n={l2_n} (src+dst={} bytes, L2_CACHE_SIZE={:?} bytes) \
                 secs_per_launch={secs:.6} bandwidth={gbps:.2} GB/s bytes_per_cycle={:.4}",
                2 * l2_n * std::mem::size_of::<f32>(),
                l2_cache_bytes,
                bytes_per_cycle(gbps, clock_khz)
            );
        }
        Err(e) => println!("L2 bandwidth measurement failed: {e}"),
    }

    println!(
        "NOTE: L1 実効帯域は本バイナリでは実測しない（モジュール冒頭ドキュメンテーション\
         コメント「L1 帯域計測の限界」参照）。docs/perf/sm121-device-attributes.md 側で\
         スペック値＋出典として記録すること。"
    );
    println!(
        "NOTE: 上記の実測値は docs/perf/sm121-device-attributes.md の実測記録表へ転記する\
         こと（イシュー #482）。コストモデル定数へのコード組み込みは C-8/C-9（#521・#524 等）\
         のスコープであり本バイナリでは行わない。"
    );
}

#[cfg(test)]
mod tests {
    use super::{bandwidth_gbps, bw_grid_dim, bytes_per_cycle};

    // `bw_grid_dim` は grid-stride ループでのカバレッジ確保が目的のため
    // 厳密な一致は不要だが、少なくとも 1 以上を返し、SM 数ベースの
    // 占有率目安を超えない（過大なブロック数でスケジューラを圧迫しない）
    // ことを回帰確認する。
    #[test]
    fn bw_grid_dim_is_bounded_by_sm_occupancy_estimate() {
        let g = bw_grid_dim(1_000_000, 16);
        assert!(g >= 1);
        assert!(g <= 16 * 32);
    }

    #[test]
    fn bw_grid_dim_covers_small_buffers_without_over_allocating() {
        // n=100, block=256 なら 1 ブロックで足りる。
        let g = bw_grid_dim(100, 16);
        assert_eq!(g, 1);
    }

    #[test]
    fn bandwidth_gbps_computes_read_plus_write_traffic() {
        // n=1e9/8 要素・1 秒 → bytes = 2 * n * 4 = 1e9 bytes → 1 GB/s。
        let n = 125_000_000usize;
        let gbps = bandwidth_gbps(n, 1.0);
        assert!((gbps - 1.0).abs() < 1e-6);
    }

    #[test]
    fn bytes_per_cycle_converts_gbps_and_clock_khz() {
        // 1 GB/s @ 1 GHz(=1_000_000 kHz) -> 1 byte/cycle。
        let v = bytes_per_cycle(1.0, 1_000_000);
        assert!((v - 1.0).abs() < 1e-9);
    }

    #[test]
    fn bytes_per_cycle_handles_non_positive_clock_as_nan() {
        assert!(bytes_per_cycle(1.0, 0).is_nan());
    }
}
