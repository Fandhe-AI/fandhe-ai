//! raw 起動経路（`cuModuleLoadData`／`cuLaunchKernelEx`）。イシュー #2122 PR-B。
//!
//! TMA のプローブは `CUtensorMap` をカーネル引数へ値渡しし、`cluster` 次元を起動時に
//! 与える（`CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION`）必要がある。cudarc 0.19.8 の safe な
//! `CudaFunction` は raw の `CUfunction` を公開しない（`cu_function` が `pub(crate)`）ため、
//! 既存の `tests/tma_probe_real_device.rs` と同じ `cudarc::driver::result::module` の低レベル API
//! を使う（unsafe はこのファイルの FFI 呼び出しのみで、各所に SAFETY を付す。依存の追加なし）。
//!
//! 段の意味は `runner.rs` と同じ（S3＝ロード・S4＝`cuLaunchKernelEx`・S5 以降は共有の
//! `finish_after_launch`）。`CUtensorMap` の encoder は [`TmaSpec`]（OOB fill・swizzle・box・座標を
//! 引数化）から作る。

use std::ffi::{CString, c_void};

use cudarc::driver::sys::{
    self, CUfunction_attribute_enum as FuncAttr, CUlaunchAttribute, CUlaunchAttributeID,
    CUlaunchAttributeValue, CUlaunchConfig, CUtensorMap, CUtensorMapDataType,
    CUtensorMapFloatOOBfill, CUtensorMapInterleave, CUtensorMapL2promotion, CUtensorMapSwizzle,
};
use cudarc::driver::{DevicePtr, DevicePtrMut};
use cudarc::nvrtc::Ptx;
use fandhe_ai_backend_cuda::CudaDevice;

use super::model_tma::{Swz, TmaSpec};
use super::registry::ProbeSpec;
use super::runner::{Cell, SENTINEL, driver_code, finish_after_launch, skipped};
use super::types::{Policy, Stage, Status};

/// `Swz` → `CUtensorMapSwizzle`。
fn swizzle_enum(s: Swz) -> CUtensorMapSwizzle {
    match s {
        Swz::None => CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
        Swz::B32 => CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_32B,
        Swz::B64 => CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_64B,
        Swz::B128 => CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
    }
}

/// `spec` から 2 次元 f32 の `CUtensorMap` を作る（`global_ptr` は f32 の行優先テンソル）。
///
/// SAFETY（呼び出し側の契約）: `global_ptr` は `spec.global_rows * spec.global_cols` 語以上の
/// device メモリを指し、`CUtensorMap` を使うカーネルの実行完了まで生存する。
unsafe fn encode_tensor_map(
    spec: &TmaSpec,
    global_ptr: u64,
) -> Result<CUtensorMap, cudarc::driver::DriverError> {
    // SAFETY: `CUtensorMap` は POD（`opaque: [u64; 16]`）で、ゼロ初期化は有効な値。
    // `cuTensorMapEncodeTiled` が全フィールドを書く。
    let mut map: CUtensorMap = unsafe { std::mem::zeroed() };
    let global_dim: [u64; 2] = [u64::from(spec.global_cols), u64::from(spec.global_rows)];
    let global_strides: [u64; 1] = [u64::from(spec.global_cols) * 4];
    let box_dim: [u32; 2] = [spec.box_cols, spec.box_rows];
    let element_strides: [u32; 2] = [1, 1];
    let fill = if spec.oob_nan {
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA
    } else {
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    };
    // SAFETY: 配列はいずれもこの呼び出しの間だけ生存すればよいスタック上の値で、長さは
    // rank 2（globalDim 2・globalStrides 1〈rank-1〉・boxDim 2・elementStrides 2）に一致する。
    // `map` は書き込み先の有効なスタック領域。`global_ptr` は呼び出し側の契約どおり。
    // パラメータの妥当性（16 バイト整列・box 上限・swizzle 幅）は `TmaSpec::validate` が
    // registry テストで全 spec に対して検査済み。違反時は driver が `Err` を返す。
    unsafe {
        sys::cuTensorMapEncodeTiled(
            &mut map as *mut CUtensorMap,
            CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
            2,
            global_ptr as *mut c_void,
            global_dim.as_ptr(),
            global_strides.as_ptr(),
            box_dim.as_ptr(),
            element_strides.as_ptr(),
            CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
            swizzle_enum(spec.swizzle),
            CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
            fill,
        )
    }
    .result()?;
    Ok(map)
}

/// `CUmodule` の RAII ガード。`Drop` で `cuModuleUnload` を呼ぶため、S3 以降のどの早期 return でも
/// モジュールが漏れない（以前は末尾の 1 か所でのみ unload していた）。
struct ModuleGuard(sys::CUmodule);

impl Drop for ModuleGuard {
    fn drop(&mut self) {
        // SAFETY: `self.0` は `load_data` が返した有効な `CUmodule` で、このガードが唯一の所有者
        // （他へコピー・unload しない）。ガードは `func`（`get_function` の結果）やそれを使う起動・
        // 同期・読み戻しがすべて終わった後（宣言順の逆順 drop）で落ちる。アンロードの失敗は握りつぶす:
        // `Drop` はエラーを返せず、測定結果（S3〜S6 の記録）には影響せず、プロセス終了時に driver が
        // context ごと回収するため。panic もしない（二重 panic による異常終了を避ける）。
        unsafe { cudarc::driver::result::module::unload(self.0) }.ok();
    }
}

fn attr_or_err(f: sys::CUfunction, a: FuncAttr) -> String {
    // SAFETY: `f` は直前に `get_function` が返した有効な `CUfunction`（モジュールは未 unload）。
    match unsafe { cudarc::driver::result::function::get_function_attribute(f, a) } {
        Ok(v) => v.to_string(),
        Err(e) => format!("err:{}", driver_code(&e)),
    }
}

/// raw 経路の S3〜S6。`cluster` は起動時に与える cluster 次元（0 = 属性なし）。
pub fn device_stages_raw(
    device: &CudaDevice,
    probe: &ProbeSpec,
    ptx: Ptx,
    cluster: u32,
    sink: &mut dyn FnMut(Cell),
) {
    let policy = probe.policy;
    let fail_tail = |sink: &mut dyn FnMut(Cell), from: Stage, why: &str| {
        for st in [Stage::S4Launch, Stage::S5Sync, Stage::S6Verify] {
            if (st as u8) >= (from as u8) {
                sink(skipped(policy, st, why));
            }
        }
    };

    // S3: ロード（ドライバ JIT）。現スレッドへ context を bind してから raw API を呼ぶ。
    if let Err(e) = device.context().bind_to_thread() {
        sink(Cell::new(
            Stage::S3ModuleLoad,
            Status::Error,
            &driver_code(&e),
            format!("bind_to_thread: {e:?}"),
        ));
        fail_tail(sink, Stage::S4Launch, "S3 failed");
        return;
    }
    let ptx_src = match CString::new(ptx.to_src()) {
        Ok(c) => c,
        Err(e) => {
            sink(Cell::new(
                Stage::S3ModuleLoad,
                Status::Error,
                "PTX_NUL",
                format!("{e}"),
            ));
            fail_tail(sink, Stage::S4Launch, "S3 failed");
            return;
        }
    };
    // SAFETY: `ptx_src` は NUL 終端の PTX テキスト。呼び出しスレッドは直前に
    // `bind_to_thread` 済みの primary context を持つ（`cuModuleLoadData` はその context へロード）。
    let module = match unsafe {
        cudarc::driver::result::module::load_data(ptx_src.as_ptr() as *const c_void)
    } {
        Ok(m) => m,
        Err(e) => {
            sink(Cell::new(
                Stage::S3ModuleLoad,
                Status::Error,
                &driver_code(&e),
                format!("load_data: {e:?}"),
            ));
            fail_tail(sink, Stage::S4Launch, "S3 failed");
            return;
        }
    };
    // 以降の早期 return・正常終了のどれでもモジュールを unload する（RAII）。
    let _module_guard = ModuleGuard(module);
    let Ok(name) = CString::new(probe.symbol) else {
        sink(Cell::new(
            Stage::S3ModuleLoad,
            Status::Error,
            "SYMBOL_NUL",
            probe.symbol,
        ));
        fail_tail(sink, Stage::S4Launch, "S3 failed");
        return;
    };
    // SAFETY: `module` は直前に取得した有効な `CUmodule`（未 unload）。
    let func = match unsafe { cudarc::driver::result::module::get_function(module, name) } {
        Ok(f) => f,
        Err(e) => {
            sink(Cell::new(
                Stage::S3ModuleLoad,
                Status::Error,
                &driver_code(&e),
                format!("get_function({}): {e:?}", probe.symbol),
            ));
            fail_tail(sink, Stage::S4Launch, "S3 failed");
            return;
        }
    };
    sink(Cell::ok(
        Stage::S3ModuleLoad,
        format!(
            "binary_version={} ptx_version={} num_regs={}",
            attr_or_err(func, FuncAttr::CU_FUNC_ATTRIBUTE_BINARY_VERSION),
            attr_or_err(func, FuncAttr::CU_FUNC_ATTRIBUTE_PTX_VERSION),
            attr_or_err(func, FuncAttr::CU_FUNC_ATTRIBUTE_NUM_REGS),
        ),
    ));
    if policy == Policy::AcceptOnly {
        fail_tail(sink, Stage::S4Launch, "");
        return;
    }

    // S4: バッファ確保・tensor map の encode・cuLaunchKernelEx。
    let stream = device.stream();
    let input = (probe.make_input)();
    let spec = probe.tma;
    let global_words = spec
        .filter(|s| s.readback_global)
        .map_or(0, |s| s.global_words() as usize);
    let kernel_out_words = probe.out_words - global_words;
    let in_dev = match stream.clone_htod(&input) {
        Ok(b) => b,
        Err(e) => {
            sink(Cell::new(
                Stage::S4Launch,
                Status::Error,
                &driver_code(&e),
                format!("htod(in): {e:?}"),
            ));
            fail_tail(sink, Stage::S5Sync, "S4 failed");
            return;
        }
    };
    let mut out_dev = match stream.clone_htod(&vec![SENTINEL; kernel_out_words]) {
        Ok(b) => b,
        Err(e) => {
            sink(Cell::new(
                Stage::S4Launch,
                Status::Error,
                &driver_code(&e),
                format!("htod(out): {e:?}"),
            ));
            fail_tail(sink, Stage::S5Sync, "S4 failed");
            return;
        }
    };
    let launched = {
        let (in_ptr, _in_guard) = in_dev.device_ptr(stream);
        let (out_ptr, _out_guard) = out_dev.device_ptr_mut(stream);
        let mut in_ptr = in_ptr;
        let mut out_ptr = out_ptr;
        let mut n_in = input.len() as i32;
        let mut n_out = kernel_out_words as i32;
        // tensor map 系の引数（tm, out, n, cx, cy, expect_tx, dump_words）。plain ABI は (in, n_in, out, n)。
        let mut tm_holder: Option<CUtensorMap> = None;
        if let Some(s) = spec {
            // SAFETY: `in_ptr` は `in_dev`（`global_words` 語以上。`make_input` が `global_input` で
            // 生成）の device pointer で、`in_dev` はこのブロックの終わり（同期前）まで生存し、
            // 同期完了まで解放されない（下で `synchronize` する前に drop しない）。
            match unsafe { encode_tensor_map(s, in_ptr) } {
                Ok(m) => tm_holder = Some(m),
                Err(e) => {
                    sink(Cell::new(
                        Stage::S4Launch,
                        Status::Error,
                        &driver_code(&e),
                        format!("cuTensorMapEncodeTiled: {e:?}"),
                    ));
                    fail_tail(sink, Stage::S5Sync, "S4 failed");
                    return;
                }
            }
        }
        let (mut cx, mut cy, mut expect_tx, mut dump_words, mut expect_tx2) = spec
            .map_or((0, 0, 0u32, 0u32, 0u32), |s| {
                (s.cx, s.cy, s.expect_tx, s.dump_words, s.expect_tx2)
            });
        let mut params: Vec<*mut c_void> = match tm_holder.as_mut() {
            Some(tm) => vec![
                tm as *mut CUtensorMap as *mut c_void,
                &mut out_ptr as *mut u64 as *mut c_void,
                &mut n_out as *mut i32 as *mut c_void,
                &mut cx as *mut i32 as *mut c_void,
                &mut cy as *mut i32 as *mut c_void,
                &mut expect_tx as *mut u32 as *mut c_void,
                &mut dump_words as *mut u32 as *mut c_void,
                &mut expect_tx2 as *mut u32 as *mut c_void,
            ],
            None => vec![
                &mut in_ptr as *mut u64 as *mut c_void,
                &mut n_in as *mut i32 as *mut c_void,
                &mut out_ptr as *mut u64 as *mut c_void,
                &mut n_out as *mut i32 as *mut c_void,
            ],
        };
        let mut cluster_attr = CUlaunchAttribute {
            id: CUlaunchAttributeID::CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION,
            pad: [0; 4],
            value: CUlaunchAttributeValue { pad: [0; 64] },
        };
        // union への書き込みは safe。`id` を CLUSTER_DIMENSION にしているので driver は `clusterDim` として解釈する。
        cluster_attr.value.clusterDim.x = cluster;
        cluster_attr.value.clusterDim.y = 1;
        cluster_attr.value.clusterDim.z = 1;
        let cfg = CUlaunchConfig {
            gridDimX: probe.grid,
            gridDimY: 1,
            gridDimZ: 1,
            blockDimX: probe.block,
            blockDimY: 1,
            blockDimZ: 1,
            sharedMemBytes: 0,
            hStream: stream.cu_stream(),
            attrs: &mut cluster_attr as *mut CUlaunchAttribute,
            numAttrs: u32::from(cluster > 0),
        };
        // SAFETY: `params` は `func` のシグネチャ（tensor 系は (CUtensorMap 値・out ptr・int n・int cx・
        // int cy・uint expect_tx・uint dump_words・uint expect_tx2)、plain は (in ptr・int n_in・out ptr・int n)）と
        // 個数・型・順序が 1:1 対応し、各要素は引数値そのものへのポインタ（driver API の契約）で
        // 同期完了までスタックに生存する。カーネル側の全ストアは `ST` で `n` に対して境界チェック
        // 済み（REQ-8）。起動形状は registry の固定値。命令の拒否で起こる実行時エラーは
        // `Result::Err`／同期時のエラーとして捕捉し panic させない（ハングは外部 timeout）。
        let r = unsafe {
            sys::cuLaunchKernelEx(
                &cfg as *const CUlaunchConfig,
                func,
                params.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        }
        .result();
        // 同期（S5）の前に `params`・`tm_holder` を落としてよいのは、`cuLaunchKernelEx` が引数を
        // 起動時にコピーする契約だから。
        r
    };
    match launched {
        Ok(()) => sink(Cell::ok(
            Stage::S4Launch,
            format!(
                "grid={} block={} cluster={cluster} launch=raw",
                probe.grid, probe.block
            ),
        )),
        Err(e) => {
            sink(Cell::new(
                Stage::S4Launch,
                Status::Error,
                &driver_code(&e),
                format!("cuLaunchKernelEx: {e:?}"),
            ));
            fail_tail(sink, Stage::S5Sync, "S4 failed");
            return;
        }
    }

    let mut readback = || {
        let mut v = stream.clone_dtoh(&out_dev)?;
        if global_words > 0 {
            v.extend(stream.clone_dtoh(&in_dev)?);
        }
        Ok(v)
    };
    finish_after_launch(device, probe, &input, &mut readback, sink);
    // `_module_guard` が関数を抜けるときに unload する。
}
