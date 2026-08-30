//! イシュー #1084: `context_cache` の poison 回復経路（[`super::invalidate_with`]
//! を実 CUDA の `sync`／`probe` クロージャで駆動する経路）を実機
//! （DGX Spark GB10 等）で検証する。`async_ordering_poison_tests.rs`
//! （T3i・イシュー #1014）と同じ配置方式（`context_cache.rs` の子モジュール。
//! `[OrdinalRegistry]`／`[invalidate_with]`／`[ProbeFailure]`／
//! `[begin_driver_call]`／`[observe_cuda_result]`／`[is_poisoned]`／
//! `[current_generation]`／`[ordinal_registry]` がモジュール非公開のため
//! `tests/`（統合テスト・クレート外部扱い）からは到達できない）。
//!
//! # T3i との違い（本ファイルが追加する検証範囲）
//!
//! T3i は「回復成功時の drain・ストリーム完了同期の配線」を**独立
//! レジストリ**（`OrdinalRegistry::new()`。本番の `ordinal_registry()` とは
//! 無関係）で検証する。本ファイルはそれに対して以下 2 点を追加する:
//!
//! 1. [`poison_recovery_real_probe_restores_active_and_rejects_stale_
//!    generation`]（T-R1）: **本番のグローバルレジストリ**
//!    （[`super::ordinal_registry`]）を使い、「本番の観測経路
//!    （[`super::begin_driver_call`]／[`super::observe_cuda_result`]。
//!    `ops.rs`／`memory.rs` の演算入口が実際に呼ぶのと同じ関数）で
//!    poison → 実 CUDA の `sync`（`stream.synchronize()`）＋実処理
//!    プローブ（既知データの GPU 往復と値照合）で回復 → 新世代での
//!    演算成功・旧世代バッファの拒否」を通しで検証する。
//! 2. [`poison_from_real_cuda_error_fails_closed_to_unrecoverable`]
//!    （T-R2）: 実際に CUDA driver が返すエラー（block 次元超過の
//!    `CUDA_ERROR_INVALID_VALUE` 相当・メモリアクセスを伴わない
//!    `__trap()` カーネルによる `CUDA_ERROR_ASSERT`／`CUDA_ERROR_
//!    LAUNCH_FAILED` 相当）を発生させ、
//!    `classify_cuda_result`（`context_cache.rs` 本体・変更なし）の
//!    分類どおりに「operation-local は poison しない」「sticky は
//!    poison し、context 破壊後は `invalidate_with` の sync 自体が
//!    失敗して恒久 poison へ fail-closed に倒れる」ことを実機で確認する。
//!
//! # 検証できないこと（`docs/backend-cuda-async-execution-design.md`
//! §12b で別追跡と明記済みの範囲）
//!
//! production `invalidate(ordinal)` の**呼び出しタイミング**（poison
//! 検出後にいつ・どこから `invalidate_with` を呼ぶか）は本ファイルの
//! 対象外のまま残る。本ファイルは `invalidate_with` 自体を実 CUDA
//! クロージャで直接呼び出すことで回復経路そのものを検証するに留まる。
//!
//! # 運用契約（プロセス分離が必須。T-R2 はプロセス破壊的）
//!
//! `context_cache::ordinal_registry()` はプロセスワイド static であり、
//! 本ファイルの 2 テストはいずれも ordinal 0（本番のデフォルト ordinal。
//! `async_ordering_poison_tests.rs` の SGD 部分と同じ選択）を直接
//! poison／回復させる。他の実機テスト（`async_ordering_real_device`
//! （`tests/` 配下）等）が同一プロセス内で並走すると、ordinal 0 の状態
//! 機械を奪い合い、意図しない `DeviceContextRetiring`／
//! `StaleDeviceGeneration` を誘発しうる。したがって:
//!
//! - 各テストは**単独のプロセス**（`cargo test ... -- --ignored
//!   <test_name> --test-threads=1` のようにテスト名でフィルタした個別
//!   起動）で実行する（`nvrtc.rs` の `setmaxnreg` プローブの外部
//!   タイムアウト運用契約と同型の「1 テスト = 1 プロセス」契約）。
//! - [`poison_from_real_cuda_error_fails_closed_to_unrecoverable`]
//!   （T-R2）は CUcontext を意図的に破壊し ordinal 0 を恒久 poison
//!   （`unrecoverable: true`。プロセス再起動以外に回復手段がない）へ
//!   確定させる。**同一プロセス内で他のテスト（T-R1 を含む）より前に
//!   実行してはならない**。実機実行の全体順序（既存実機テストの非後退
//!   確認を含む）は `docs/backend-cuda-async-execution-design.md` §12c
//!   を正とする。
//!
//! 上記の運用契約はコメントだけでは強制されない（`cargo test -p
//! fandhe-ai-backend-cuda --lib -- --ignored` のようにテスト名フィルタ
//! なしでフィルタなしで起動すると、同一プロセス・マルチスレッドで T-R1・
//! T-R2 が並行または意図しない順序で実行されうる。review 指摘）ため、
//! 各テストの冒頭で opt-in 環境変数を fail-closed に検査する
//! （未設定なら即 return して何もしない。テストは実行されたが何も
//! 検証しなかった扱いになる＝安全側）。
//!
//! - `FANDHE_AI_CUDA_POISON_RECOVERY_REAL_DEVICE=1`: T-R1・T-R2 共通の
//!   基本 opt-in（実機・単独プロセス起動であることの明示的同意）。
//! - `FANDHE_AI_CUDA_POISON_RECOVERY_ALLOW_UNRECOVERABLE=1`: T-R2 のみが
//!   追加で要求する opt-in（本番グローバルレジストリの ordinal 0 を
//!   恒久 poison させ、以降そのプロセスでは CUDA を一切使えなくする
//!   ことへの明示的同意）。この 2 段構えにより、基本 opt-in のみを
//!   設定してテスト名フィルタなしで実行した場合でも T-R2 は自動では
//!   走らない。

use std::sync::Arc;

use cudarc::driver::{CudaFunction, LaunchConfig, PushKernelArg};

use fandhe_ai_tensor_core::BackendOps;
use fandhe_ai_tensor_core::Tensor;
use fandhe_ai_tensor_core::buffer::MemoryOps;
use fandhe_ai_tensor_core::device::BackendError;

use super::{
    ProbeFailure, begin_driver_call, current_generation, invalidate_with, is_poisoned,
    observe_cuda_result, ordinal_registry,
};
use crate::device::CudaDevice;
use crate::error::CudaError;
use crate::memory::CudaMemory;
use crate::nvrtc::compile_ptx;
use crate::ops::CudaBackendOps;
use crate::sgd::{CudaSgd, SgdKernelParams};

/// 本ファイルの実機テスト共通の基本 opt-in 環境変数（モジュール冒頭
/// コメント「運用契約」参照）。フィルタなしの `--ignored` 一括実行で
/// 意図せず本番グローバルレジストリを変異させないための fail-closed
/// ガード（未設定なら早期 return）。
const OPT_IN_ENV: &str = "FANDHE_AI_CUDA_POISON_RECOVERY_REAL_DEVICE";

/// [`poison_from_real_cuda_error_fails_closed_to_unrecoverable`]（T-R2）
/// のみが追加で要求する opt-in（恒久 poison への明示的同意）。
const OPT_IN_UNRECOVERABLE_ENV: &str = "FANDHE_AI_CUDA_POISON_RECOVERY_ALLOW_UNRECOVERABLE";

/// `env_var` が `"1"` に設定されているかを検査し、未設定・不一致なら
/// `test_name` を添えて標準エラーへ理由を出力したうえで `false` を返す
/// （呼び出し側はこれを受けて即 `return` する。何も検証せず正常終了する
/// ため、`--ignored` の一括実行に対して安全側＝fail-closed に倒れる）。
fn require_opt_in(env_var: &str, test_name: &str) -> bool {
    match std::env::var(env_var) {
        Ok(v) if v == "1" => true,
        _ => {
            eprintln!(
                "skipping {test_name}: set {env_var}=1 to opt in (単独プロセス・単独 \
                 テスト名フィルタでの実行を前提とする運用契約。モジュール冒頭コメント \
                 「運用契約」参照)"
            );
            false
        }
    }
}

/// テストローカルの NVRTC カーネル 1（実処理プローブ・「不正サイズの
/// launch」テスト共用）。`idx == 0` のスレッドだけが `out[0]` へ決定的な
/// 値（`7.0f`）を書く（実際に GPU 上でカーネルが実行され、結果が正しく
/// 読み出せることを確認するための最小限の実処理）。
///
/// [`poison_from_real_cuda_error_fails_closed_to_unrecoverable`]（T-R2）
/// の「block 次元超過」テストでも同じ関数を使い回す（起動そのものが
/// 拒否される前提のため、カーネル本体の意味論は問わない。有効な
/// `CudaFunction` を用意する目的のみ）。
const NOOP_PROBE_SRC: &str = r#"
extern "C" __global__ void poison_recovery_probe_write_f32(float* __restrict__ out) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        out[0] = 7.0f;
    }
}
"#;

/// テストローカルの NVRTC カーネル 2（[`poison_from_real_cuda_error_
/// fails_closed_to_unrecoverable`] 専用）。
///
/// 引数を取らず、メモリアクセスを一切行わない `__trap()`（PTX `trap`
/// 命令）のみを実行する。目的は sticky なデバイス異常（
/// `context_cache::classify_cuda_result` の分類で sticky 側へ倒れる
/// `CUDA_ERROR_ASSERT`／`CUDA_ERROR_LAUNCH_FAILED` 等。同関数の未知
/// コードを sticky 側へ倒す既定コメントに明記済み）を確実に発生させる
/// ことである。メモリアクセスが存在しないためカーネル自身に境界検査
/// 対象がなく、`.claude/rules/coding-rust.md`「カーネル実装の境界検査
/// （REQ-8）」規約（性能・テスト目的を理由に境界チェックを省略しない）
/// を回避ではなく自明に満たす。
///
/// review 指摘（P0・2 ラウンド目）: 以前の実装は null ポインタ
/// （アドレス 0）への意図的な範囲外書き込みだったが、境界チェック
/// 省略には「テスト・故障注入目的」の例外が規約上存在しないという
/// 指摘を受けた。本ラウンドでは書き込みそのものをなくし、`__trap()`
/// という「メモリアクセスを伴わない」故障注入手段へ置き換える
/// （driver API 経由でポインタを解放してダングリングにする代替案は、
/// `cuMemFree` 後もサブアロケータが VA を保持しマップされたままになり
/// うるため確実性を欠くとして不採用。値ではなく命令自体で確実に
/// 異常を発生させる本方式を採用した）。
const TRAP_SRC: &str = r#"
extern "C" __global__ void poison_recovery_trap() {
    __trap();
}
"#;

/// T-R1: `invalidate_with` を実 CUDA の `sync`（`stream.synchronize()`）＋
/// 実処理プローブ（既知データの GPU 往復と値照合）で駆動し、本番の
/// グローバルレジストリ上で「観測経路による poison → 回復 → 新世代での
/// 演算成功／旧世代バッファの拒否」を通しで検証する。
///
/// 手順は実装計画（イシュー #1084）§3 T-R1 のとおり。順序を崩すと
/// 世代・poison 状態の前提が変わるため、本関数内で分割しない。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須。本番グローバルレジストリの \
            ordinal 0 を変異させるため単独プロセスで実行すること（モジュール \
            冒頭コメント「運用契約」参照）"]
fn poison_recovery_real_probe_restores_active_and_rejects_stale_generation() {
    if !require_opt_in(
        OPT_IN_ENV,
        "poison_recovery_real_probe_restores_active_and_rejects_stale_generation",
    ) {
        return;
    }

    let ordinal = 0usize;
    let device =
        CudaDevice::new(ordinal).expect("CUDA device 0 must be available on ignored test runner");
    let stream = device.stream().clone();
    let memory = CudaMemory::new(&device);
    let cuda_ops = CudaBackendOps::new(ordinal);

    // 実処理プローブに使う SGD カーネルは `invalidate_with` 呼び出し前に
    // コンパイルしておく（advisor レビュー指摘: プローブクロージャの中で
    // NVRTC コンパイルを行うと、`Phase::Retiring` 中に
    // `begin_driver_call` を経由しない driver 呼び出しが混在し、
    // 「回復クロージャそのものの配線」という検証対象が曖昧になる）。
    let sgd = CudaSgd::new(&device).expect("CudaSgd::new must succeed on real hardware");

    // 1. ベースライン演算が成功することを確認する（poison していない
    //    健全な状態の前提確認）。
    let a = Tensor::new(vec![1.0f32, 2.0], &[2]).expect("tensor construction");
    let b = Tensor::new(vec![3.0f32, 4.0], &[2]).expect("tensor construction");
    let baseline = cuda_ops
        .add(&a, &b)
        .expect("baseline add must succeed before poisoning");
    assert_eq!(baseline.as_slice(), Some([4.0f32, 6.0f32].as_slice()));

    // 2. 実デバイスバッファを確保する（現行世代 g が刻印される。回復後の
    //    旧世代拒否検証に使う）。
    let g_before = current_generation(ordinal);
    let stale_tensor = Tensor::new(vec![1.0f32; 1024], &[1024]).expect("tensor construction");
    let stale_buffer = memory
        .upload(&stale_tensor)
        .expect("upload must succeed before poisoning");
    assert_eq!(
        stale_buffer.generation(),
        g_before,
        "upload 直後のバッファ世代は現行世代と一致するはず"
    );

    // 3. 本番の観測経路（`ops.rs`／`memory.rs` の演算入口が実際に呼ぶのと
    //    同じ `begin_driver_call`／`observe_cuda_result`）で sticky エラー
    //    を注入し poison する（`ops.rs::tests::sticky_driver_error` と
    //    同じ作り方）。`CallToken` はこのブロックを抜ける時点で drop し
    //    `in_flight` を確定させる（advisor レビュー指摘: drop し忘れると
    //    後段の `invalidate_with` の drain フェーズが
    //    `in_flight == 0` を待ち続け無期限に停止する）。
    {
        let token = begin_driver_call(ordinal, &[0])
            .expect("ordinal must be Active with generation 0 before poisoning");
        let sticky = cudarc::driver::result::DriverError(
            cudarc::driver::sys::CUresult::CUDA_ERROR_ILLEGAL_ADDRESS,
        );
        let observed: Result<(), CudaError> =
            observe_cuda_result(ordinal, &token, Err(CudaError::Driver(sticky)));
        assert!(
            observed.is_err(),
            "注入した sticky エラーはそのまま伝播するはず"
        );
        // token はブロック末尾で drop される。
    }

    // 4. poison 中の拒否を実機で確認する（本番演算経路）。
    let rejected = cuda_ops.add(&a, &b);
    assert!(
        matches!(rejected, Err(BackendError::DeviceContextPoisoned(_))),
        "poison 直後の演算は DeviceContextPoisoned で拒否されるはず: {rejected:?}"
    );

    // 5. `invalidate_with` を実 CUDA の sync ＋実処理プローブで駆動する。
    let sync_stream = Arc::clone(&stream);
    let probe_stream = Arc::clone(&stream);
    let recovery_result = invalidate_with(
        ordinal_registry(),
        ordinal,
        // b'. ストリーム完了同期（実 CUDA）。
        move || sync_stream.synchronize().map_err(|_| ProbeFailure::Sticky),
        // c. 実処理プローブ（設計文書 §9 item 5）: 既知データを転送し
        //    1 ステップの SGD カーネル（lr=1.0・grad=1.0・momentum=0.0 で
        //    決定的に `param -= 1.0` となる。T3i と同じ素材）を実行し、
        //    D2H で往復した値を照合する。
        move || {
            let mut param = probe_stream
                .clone_htod(&[0.0f32])
                .map_err(|_| ProbeFailure::OperationLocal)?;
            let grad = probe_stream
                .clone_htod(&[1.0f32])
                .map_err(|_| ProbeFailure::OperationLocal)?;
            let params = SgdKernelParams {
                lr: 1.0,
                momentum: 0.0,
                dampening: 0.0,
                weight_decay: 0.0,
                nesterov: false,
                is_first_step: true,
            };
            sgd.run(&mut param, &grad, None, &params)
                .map_err(|_| ProbeFailure::OperationLocal)?;
            let host = probe_stream
                .clone_dtoh(&param)
                .map_err(|_| ProbeFailure::OperationLocal)?;
            probe_stream
                .synchronize()
                .map_err(|_| ProbeFailure::Sticky)?;
            if (host[0] - (-1.0f32)).abs() < 1e-3 {
                Ok(())
            } else {
                Err(ProbeFailure::Mismatch)
            }
        },
    );
    assert!(
        recovery_result.is_ok(),
        "sync・実処理プローブの双方が成功する限り invalidate_with は Ok を \
         返すはず: {recovery_result:?}"
    );
    assert_eq!(
        current_generation(ordinal),
        g_before + 1,
        "回復成功時 generation はちょうど 1 進むはず"
    );
    assert!(
        !is_poisoned(ordinal),
        "回復成功後は poison 状態が解消され Active へ復帰するはず"
    );

    // 6. 旧世代バッファの拒否（`StaleDeviceGeneration`）を実バッファで
    //    確認する。
    let stale_download = memory.download(&stale_buffer);
    assert!(
        matches!(
            stale_download,
            Err(BackendError::StaleDeviceGeneration {
                ordinal: rejected_ordinal,
                resource_generation,
                current_generation: current,
            }) if rejected_ordinal == ordinal
                && resource_generation == g_before
                && current == g_before + 1
        ),
        "回復前の世代を刻印された旧バッファの download は \
         StaleDeviceGeneration で拒否されるはず: {stale_download:?}"
    );

    // 7. 新世代での演算成功（新たに確保したバッファは新世代 g+1 を刻印
    //    され、正常に往復できる）。キャッシュ済み `CudaDevice`（本テスト
    //    冒頭で構築した `device`）自体は evict されず再利用される設計
    //    である（`context_cache` モジュール冒頭コメント「所有モデル・
    //    生存期間」参照。`invalidate_with` は ordinal 単位の世代
    //    カウンタのみを進め、`cached_device` 等のキャッシュエントリを
    //    差し替えない）。
    let fresh_tensor = Tensor::new(vec![9.0f32; 8], &[8]).expect("tensor construction");
    let fresh_buffer = memory
        .upload(&fresh_tensor)
        .expect("upload after recovery must succeed");
    assert_eq!(fresh_buffer.generation(), g_before + 1);
    let fresh_readback = memory
        .download(&fresh_buffer)
        .expect("download after recovery must succeed");
    assert_eq!(fresh_readback.as_slice(), Some([9.0f32; 8].as_slice()));

    let post_recovery_add = cuda_ops
        .add(&a, &b)
        .expect("add after recovery must succeed");
    assert_eq!(
        post_recovery_add.as_slice(),
        Some([4.0f32, 6.0f32].as_slice())
    );
}

/// T-R2: 実 CUDA エラーに由来する poison と fail-closed 恒久化を検証
/// する。**プロセス破壊的**（本番グローバルレジストリの ordinal 0 を
/// 恒久 poison へ確定させる）ため、モジュール冒頭コメント「運用契約」の
/// とおり必ず最後・単独プロセスで実行する。
///
/// 手順 1（block 次元超過）の実エラーコードは環境依存でありうるため
/// 具体的な `CUresult` は固定検査しない（advisor レビュー指摘）。
/// 「起動が同期的に失敗すること」「その失敗が ordinal を poison しない
/// こと」（`classify_cuda_result` の operation-local 分類）の 2 点のみを
/// 検証する。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須。プロセス破壊的（本番グローバル \
            レジストリの ordinal 0 を恒久 poison へ確定させる）。単独プロセス \
            かつ他の実機テストより後に実行すること（モジュール冒頭コメント \
            「運用契約」参照）"]
fn poison_from_real_cuda_error_fails_closed_to_unrecoverable() {
    if !require_opt_in(
        OPT_IN_ENV,
        "poison_from_real_cuda_error_fails_closed_to_unrecoverable",
    ) || !require_opt_in(
        OPT_IN_UNRECOVERABLE_ENV,
        "poison_from_real_cuda_error_fails_closed_to_unrecoverable",
    ) {
        return;
    }

    let ordinal = 0usize;
    let device =
        CudaDevice::new(ordinal).expect("CUDA device 0 must be available on ignored test runner");
    let stream = device.stream().clone();
    let memory = CudaMemory::new(&device);
    let cuda_ops = CudaBackendOps::new(ordinal);

    // カーネル 2 種は context 破壊前にコンパイル・ロードしておく
    // （advisor レビュー指摘: NVRTC コンパイル自体を poison 後の状態
    // 遷移の検証と混同しないため）。
    let arch = device.arch();
    let noop_ptx = compile_ptx(NOOP_PROBE_SRC, arch).expect("noop kernel must compile");
    let noop_fn: CudaFunction = device
        .context()
        .load_module(noop_ptx)
        .expect("load_module must succeed")
        .load_function("poison_recovery_probe_write_f32")
        .expect("load_function must succeed");
    let trap_ptx = compile_ptx(TRAP_SRC, arch).expect("trap kernel must compile");
    let trap_fn: CudaFunction = device
        .context()
        .load_module(trap_ptx)
        .expect("load_module must succeed")
        .load_function("poison_recovery_trap")
        .expect("load_function must succeed");

    // 手順 0: healthy な状態で probe 用バッファを確保しておく（context
    // 破壊後の download 拒否確認に使う）。
    let probe_tensor = Tensor::new(vec![0.0f32; 4], &[4]).expect("tensor construction");
    let probe_buffer = memory
        .upload(&probe_tensor)
        .expect("baseline upload must succeed before corrupting the context");

    // 手順 1: 「不正サイズの launch」（block 次元超過）は同期的に
    // エラーを返し operation-local（poison しない）ことを実機で確認する
    // （イシュー本文の例に対する実測回答。§12c 検証項目）。
    {
        let token = begin_driver_call(ordinal, &[])
            .expect("ordinal must still be Active before this probe");
        let dummy = stream.clone_htod(&[0.0f32]).expect("alloc dummy arg");
        let oversized_cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            // 実機のデバイス上限（通常 1024）を大幅に超える block 次元。
            block_dim: (2_000_000_000, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: 引数 `dummy` は 1 要素の有効なデバイスポインタであり、
        // カーネル本体（`poison_recovery_probe_write_f32`）は範囲外
        // アクセスを行わない。本ブロックが検証したいのは「block 次元
        // 超過という起動パラメータ自体の不正」が同期的にエラーを返す
        // ことであり、カーネル本体の安全性はこの結果に影響しない
        // （起動自体が拒否され、カーネルは実行されない想定）。
        let launch_result: Result<(), CudaError> = unsafe {
            stream
                .launch_builder(&noop_fn)
                .arg(&dummy)
                .launch(oversized_cfg)
                .map(|_events| ())
                .map_err(CudaError::from)
        };
        let observed = observe_cuda_result(ordinal, &token, launch_result);
        assert!(
            observed.is_err(),
            "block 次元超過は同期的にエラーを返すはず: {observed:?}"
        );
        // token はブロック末尾で drop される。
    }
    // 本 assertion は「block 次元超過は CUDA_ERROR_INVALID_VALUE を返し
    // classify_cuda_result の operation-local 固定リストに含まれる」
    // （`context_cache.rs` の分類実装で確認済み）という前提に立った固定
    // 検査である。実 CUresult 自体は driver バージョン依存で理論上は
    // 未列挙コード（sticky 側デフォルト）が返る可能性が残るため、実機
    // 実測で本前提と食い違いが生じた場合は
    // `docs/backend-cuda-async-execution-design.md` §12c へ記録する
    // （review 指摘）。
    assert!(
        !is_poisoned(ordinal),
        "operation-local な起動エラー（classify_cuda_result の分類）は \
         ordinal を poison しないはず"
    );

    // 手順 2: 実 sticky エラーを発生させる。メモリアクセスを一切
    // 行わない `__trap()` カーネルを、本番の観測経路
    // （`begin_driver_call`／`observe_cuda_result`）を経由せず直接
    // 投入する（非同期投入契約〈イシュー #1013・設計文書 §5〉により
    // launch 自体の戻り値は多くの場合 `Ok` に見える。遅延エラーは
    // 後続の同期呼び出しで観測される）。
    //
    // SAFETY: `poison_recovery_trap`（本ファイル上部の `TRAP_SRC`）は
    // 引数を取らず、いかなるメモリへもアクセスしない（PTX `trap`
    // 命令のみを実行する）。したがってカーネル自身に境界検査対象は
    // 存在せず、`.claude/rules/coding-rust.md`「カーネル実装の境界検査
    // （REQ-8）」規約に抵触しない（review 指摘・2 ラウンド目: 以前の
    // 実装は null ポインタへの意図的な範囲外書き込みだったが、境界
    // チェック省略にテスト・故障注入目的の例外はないとの指摘を受け、
    // 書き込みを伴わない `__trap()` へ置き換えた。上部ドキュメント
    // コメント参照）。`__trap()` はデバイス異常（`CUDA_ERROR_ASSERT`
    // または `CUDA_ERROR_LAUNCH_FAILED` 相当。`context_cache::
    // classify_cuda_result` の分類でいずれも sticky 側へ倒れる）を
    // 確実に発生させ、後続の本番演算がこれを遅延エラーとして観測し
    // ordinal を poison することを確認する目的で使う。本呼び出し後は
    // CUcontext が破壊される前提であり、本テストは `#[ignore]` かつ
    // プロセス末尾専用（モジュール冒頭コメント「運用契約」参照）。
    unsafe {
        // 起動自体の Result は意図的に無視する（非同期投入契約。後続の
        // 本番演算がストリーム順序で先行するこの launch のエラーを
        // 遅延して観測する）。
        let _ = stream.launch_builder(&trap_fn).launch(LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        });
    }

    // 手順 2': 本番経路（`memory.download`）が遅延エラーを観測し
    // ordinal を poison することを確認する。
    let download_result = memory.download(&probe_buffer);
    assert!(
        download_result.is_err(),
        "__trap() 実行後の同期を含む本番演算は失敗するはず: \
         {download_result:?}"
    );
    assert!(
        is_poisoned(ordinal),
        "sticky エラーの観測により ordinal は poison されるはず"
    );

    // 手順 3: `invalidate_with` は sync（実 `stream.synchronize()`）自体
    // が失敗し、恒久 poison（`Poisoned{unrecoverable:true}`）へ
    // fail-closed で倒れる想定（設計文書 §5 の状態遷移「Retiring --stream
    // sync 失敗--> Poisoned{true}」。真に context を破壊する sticky
    // エラーでは同一 CUcontext 上の以降の呼び出しはすべて失敗し続ける
    // ため、probe クロージャに到達する前に sync 側で確定する想定）。
    let sync_stream = Arc::clone(&stream);
    let recovery_result = invalidate_with(
        ordinal_registry(),
        ordinal,
        move || sync_stream.synchronize().map_err(|_| ProbeFailure::Sticky),
        // probe に到達すること自体が §12c で追記すべき実測（設計文書の
        // 前提が崩れる可能性）だが、テストとしては「sync が失敗する限り
        // probe には到達しない」契約は `context_cache.rs::
        // poison_state_tests`（GPU 非依存モック）が既に検証済みのため、
        // ここでは to 到達時も安全に `Sticky` を返すだけに留める
        // （panic させない。実測との食い違いは §12c へ記録する）。
        || Err(ProbeFailure::Sticky),
    );
    assert!(
        matches!(
            recovery_result,
            Err(BackendError::DeviceContextUnrecoverable { .. })
        ),
        "context を破壊する sticky エラーの後、sync 自体が失敗し恒久 poison \
         （DeviceContextUnrecoverable）へ倒れるはず: {recovery_result:?}"
    );

    // 手順 4: 以降の本番演算は恒久 poison により拒否される。
    let a = Tensor::new(vec![1.0f32], &[1]).expect("tensor construction");
    let post_result = cuda_ops.add(&a, &a);
    assert!(
        matches!(
            post_result,
            Err(BackendError::DeviceContextUnrecoverable { .. })
        ),
        "恒久 poison 後の演算は DeviceContextUnrecoverable で拒否されるはず: \
         {post_result:?}"
    );
}
