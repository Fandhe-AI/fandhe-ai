//! `ModelCheckpoint::to_file`（イシュー #2073・親 #2059。safetensors
//! ファイル保存機構）の統合テスト。`compat_sequential_callbacks.rs`・
//! `interop_safetensors_roundtrip.rs` と同じ流儀で `fandhe_ai`
//! （facade）・`bench_harness::rng`・`std` のみを import する
//! （`callbacks.rs` は `src/interop/` 以外から `onnx_interop` を
//! 参照しない契約——`tests/api_surface.rs::
//! facade_sources_reference_onnx_interop_only_in_interop_module`——を
//! 持つため、テスト側も `fandhe_ai::interop::safetensors` 経由でのみ
//! safetensors 型へ触れる）。
//!
//! **決定的シード**: 重み初期化・データ生成は固定シードで駆動する
//! （`.claude/rules/coding-rust.md`）。実機（CUDA/Metal）非依存の
//! CPU 経路のみを本ファイルで検証し、CUDA／Metal 実機での往復は
//! `#[ignore]` 分離（§6.3 相当。下部）とする。

use std::collections::HashMap;

use bench_harness::rng::Xorshift64Star;
use fandhe_ai::compat::{
    Callback, EarlyStopping, FitConfig, Loss, ModelCheckpoint, Monitor, Optimizer, Sequential,
};
use fandhe_ai::interop::safetensors::load_safetensors_f32;
use fandhe_ai::{AutodiffError, Tensor};

const N: usize = 16;
const D_IN: usize = 4;
const D_HIDDEN: usize = 8;
const D_OUT: usize = 2;

const SEED_DATA: u64 = 0xC0FFEE;
const SEED_L1: u64 = 0x1111_1111;
const SEED_L2: u64 = 0x2222_2222;

/// `compat_sequential_callbacks.rs::gen_regression_data` と同型の
/// 決定的生成（本ファイルはテストバイナリが分かれるため独立に定義
/// する）。
fn gen_regression_data(seed: u64) -> (Tensor<f32>, Tensor<f32>) {
    let mut rng = Xorshift64Star::new(seed);
    let x = rng.fill_vec(N * D_IN);
    let y = rng.fill_vec(N * D_OUT);
    (
        Tensor::new(x, &[N, D_IN])
            .unwrap_or_else(|e| panic!("test fixture: x の shape 構築に失敗: {e}")),
        Tensor::new(y, &[N, D_OUT])
            .unwrap_or_else(|e| panic!("test fixture: y の shape 構築に失敗: {e}")),
    )
}

fn build_model() -> Sequential {
    Sequential::new()
        .add_linear(D_IN, D_HIDDEN, SEED_L1)
        .unwrap_or_else(|e| panic!("test fixture: 層 1 の構築に失敗: {e}"))
        .add_relu()
        .add_linear(D_HIDDEN, D_OUT, SEED_L2)
        .unwrap_or_else(|e| panic!("test fixture: 層 2 の構築に失敗: {e}"))
}

/// `interop_safetensors_roundtrip.rs::temp_dir_for` と同型（プロセス
/// ID + テスト名で衝突しない一時ディレクトリを作る。`tempfile`
/// クレートは不使用）。
fn temp_dir_for(test_name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "fandhe-ai-checkpoint-file-{}-{test_name}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn state_dict_bit_exact(
    a: &HashMap<String, Tensor<f32>>,
    b: &HashMap<String, Tensor<f32>>,
) -> bool {
    if a.len() != b.len() {
        return false;
    }
    for (key, av) in a {
        let Some(bv) = b.get(key) else {
            return false;
        };
        if av.shape() != bv.shape() {
            return false;
        }
        let a_bits: Vec<u32> = av
            .contiguous()
            .as_slice()
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect();
        let b_bits: Vec<u32> = bv
            .contiguous()
            .as_slice()
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect();
        if a_bits != b_bits {
            return false;
        }
    }
    true
}

// =====================================================================
// 1. fit 中の自動保存 → 読み戻し → best_state_dict と bit 一致
//    （tmp ファイルが残らないことも確認）
// =====================================================================

#[test]
fn to_file_saves_best_snapshot_and_roundtrips_bit_exact() {
    let (x, y) = gen_regression_data(SEED_DATA);
    let dir = temp_dir_for("best-only");
    let path = dir.join("best.safetensors");

    let mut model = build_model();
    model
        .compile(
            Optimizer::Sgd(fandhe_ai::optim::SgdConfig::new(0.05)),
            Loss::Mse,
        )
        .unwrap();
    let mut callbacks = [Callback::ModelCheckpoint(
        ModelCheckpoint::new().monitor(Monitor::Loss).to_file(&path),
    )];
    model
        .fit_with_callbacks(&x, &y, FitConfig::new(3, N), None, &mut callbacks)
        .unwrap();

    let Callback::ModelCheckpoint(mc) = &callbacks[0] else {
        unreachable!()
    };
    let expected = mc
        .best_state_dict()
        .expect("test fixture: 3 epoch 学習すれば best スナップショットは必ず存在する");

    let loaded = load_safetensors_f32(&path)
        .unwrap_or_else(|e| panic!("保存済みファイルの読み戻しに失敗: {e}"));
    assert!(
        state_dict_bit_exact(expected, &loaded),
        "ファイルから読み戻した state_dict が in-memory best_state_dict と bit 一致しない"
    );

    // 一時ファイル（`.tmp.` を含む名前）が残っていないこと（atomic
    // rename の契約。`save_safetensors_f32` doc 参照）。
    let leftover: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_name().to_string_lossy().contains(".tmp."))
        .collect();
    assert!(
        leftover.is_empty(),
        "一時ファイルが残っている: {leftover:?}"
    );
}

// =====================================================================
// 2. save_best_only(false): 毎 epoch 上書きされ最終 epoch の
//    state_dict と一致する
// =====================================================================

#[test]
fn to_file_with_save_best_only_false_persists_last_epoch() {
    let (x, y) = gen_regression_data(SEED_DATA);
    let dir = temp_dir_for("every-epoch");
    let path = dir.join("last.safetensors");

    let mut model = build_model();
    model
        .compile(
            Optimizer::Sgd(fandhe_ai::optim::SgdConfig::new(0.05)),
            Loss::Mse,
        )
        .unwrap();
    let mut callbacks = [Callback::ModelCheckpoint(
        ModelCheckpoint::new()
            .monitor(Monitor::Loss)
            .save_best_only(false)
            .to_file(&path),
    )];
    model
        .fit_with_callbacks(&x, &y, FitConfig::new(3, N), None, &mut callbacks)
        .unwrap();

    let final_state = model.state_dict();
    let loaded = load_safetensors_f32(&path)
        .unwrap_or_else(|e| panic!("保存済みファイルの読み戻しに失敗: {e}"));
    assert!(
        state_dict_bit_exact(&final_state, &loaded),
        "save_best_only(false) の最終ファイル内容が最終 epoch の state_dict と bit 一致しない"
    );
}

// =====================================================================
// 3. 復元フロー: 読み戻した state_dict を新規モデルへ load_state_dict
//    → predict 出力が元モデルと bit 一致
// =====================================================================

#[test]
fn to_file_then_load_state_dict_predict_matches_source_model_bit_exact() {
    let (x, y) = gen_regression_data(SEED_DATA);
    let dir = temp_dir_for("restore-predict");
    let path = dir.join("ckpt.safetensors");

    let mut model = build_model();
    model
        .compile(
            Optimizer::Sgd(fandhe_ai::optim::SgdConfig::new(0.05)),
            Loss::Mse,
        )
        .unwrap();
    let mut callbacks = [Callback::ModelCheckpoint(
        ModelCheckpoint::new()
            .monitor(Monitor::Loss)
            .save_best_only(false)
            .to_file(&path),
    )];
    model
        .fit_with_callbacks(&x, &y, FitConfig::new(2, N), None, &mut callbacks)
        .unwrap();

    let probe = Tensor::new(vec![0.1_f32, -0.2, 0.3, -0.4], &[1, D_IN]).unwrap();
    let expected_output = model.predict(&probe).unwrap();

    let loaded = load_safetensors_f32(&path)
        .unwrap_or_else(|e| panic!("保存済みファイルの読み戻しに失敗: {e}"));
    let mut restored = build_model();
    restored.load_state_dict(loaded).unwrap();
    let restored_output = restored.predict(&probe).unwrap();

    let expected_bits: Vec<u32> = expected_output
        .contiguous()
        .as_slice()
        .unwrap()
        .iter()
        .map(|v| v.to_bits())
        .collect();
    let restored_bits: Vec<u32> = restored_output
        .contiguous()
        .as_slice()
        .unwrap()
        .iter()
        .map(|v| v.to_bits())
        .collect();
    assert_eq!(
        expected_bits, restored_bits,
        "復元モデルの predict 出力が元モデルと bit 一致しない"
    );
}

// =====================================================================
// 4. restore_best_weights 併用: 同一 Monitor・min_delta 0.0 なら
//    fit 終了後の state_dict とファイル内容が bit 一致する
// =====================================================================

#[test]
fn to_file_combined_with_early_stopping_restore_best_weights_bit_exact() {
    let (x, y) = gen_regression_data(SEED_DATA);
    let dir = temp_dir_for("restore-best-weights");
    let path = dir.join("ckpt.safetensors");

    let mut model = build_model();
    model
        .compile(
            Optimizer::Sgd(fandhe_ai::optim::SgdConfig::new(0.05)),
            Loss::Mse,
        )
        .unwrap();
    let mut callbacks = [
        Callback::EarlyStopping(
            EarlyStopping::new(10) // patience は epoch 数超のため打ち切らない
                .monitor(Monitor::Loss)
                .min_delta(0.0)
                .unwrap()
                .restore_best_weights(true),
        ),
        Callback::ModelCheckpoint(ModelCheckpoint::new().monitor(Monitor::Loss).to_file(&path)),
    ];
    model
        .fit_with_callbacks(&x, &y, FitConfig::new(5, N), None, &mut callbacks)
        .unwrap();

    let final_state = model.state_dict();
    let loaded = load_safetensors_f32(&path)
        .unwrap_or_else(|e| panic!("保存済みファイルの読み戻しに失敗: {e}"));
    assert!(
        state_dict_bit_exact(&final_state, &loaded),
        "restore_best_weights 併用後の state_dict とファイル内容が bit 一致しない \
         （同一 Monitor・min_delta 0.0 なら EarlyStopping と ModelCheckpoint の \
         best epoch は一致するはず）"
    );
}

// =====================================================================
// 5. エラー伝播: 保存先の親を既存ファイルにして保存を失敗させると
//    fit_with_callbacks が InvalidArgument を返し、モデルは compiled
//    のまま・train/eval モードは呼び出し前に戻る
// =====================================================================

#[test]
fn to_file_save_failure_propagates_as_invalid_argument_and_restores_mode() {
    let (x, y) = gen_regression_data(SEED_DATA);
    let dir = temp_dir_for("save-failure");
    // 親ディレクトリ位置に通常ファイルを置き、`create_dir_all` を
    // 失敗させる（`persist` の親ディレクトリ作成契約。
    // `callbacks.rs::persist` doc 参照）。
    let blocking_file = dir.join("not-a-dir");
    std::fs::write(&blocking_file, b"not a directory").unwrap();
    let path = blocking_file.join("ckpt.safetensors");

    let mut model = build_model();
    model
        .compile(
            Optimizer::Sgd(fandhe_ai::optim::SgdConfig::new(0.05)),
            Loss::Mse,
        )
        .unwrap();
    let prev_training = model.training();
    let mut callbacks = [Callback::ModelCheckpoint(
        ModelCheckpoint::new().monitor(Monitor::Loss).to_file(&path),
    )];
    let err = model
        .fit_with_callbacks(&x, &y, FitConfig::new(2, N), None, &mut callbacks)
        .unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    assert!(model.is_compiled());
    assert_eq!(model.training(), prev_training);
}

// =====================================================================
// 5b. 一時的な保存失敗は best／best_epoch／state をロールバックし、
//     次回 fit_with_callbacks 呼び出しで再試行できる（イシュー #2073
//     codex-review 指摘: 保存失敗後も best を前進させたままだと
//     save_best_only(true) 下で以後その値を上回らない限り再保存が
//     試行されない）
// =====================================================================

#[test]
fn to_file_save_failure_rolls_back_best_and_retries_on_next_fit_call() {
    let (x, y) = gen_regression_data(SEED_DATA);
    let dir = temp_dir_for("save-failure-retry");
    // 親ディレクトリ位置に通常ファイルを置き `create_dir_all` を失敗
    // させる（1 回目の呼び出し用の障害物）。
    let blocking_file = dir.join("not-a-dir");
    std::fs::write(&blocking_file, b"not a directory").unwrap();
    let path = blocking_file.join("ckpt.safetensors");

    let mut model = build_model();
    model
        .compile(
            Optimizer::Sgd(fandhe_ai::optim::SgdConfig::new(0.05)),
            Loss::Mse,
        )
        .unwrap();
    let mut callbacks = [Callback::ModelCheckpoint(
        ModelCheckpoint::new().monitor(Monitor::Loss).to_file(&path),
    )];

    // 1 回目: 保存失敗 → InvalidArgument。best／best_epoch／state は
    // 一切コミットされていないはず（ロールバック契約）。
    let err = model
        .fit_with_callbacks(&x, &y, FitConfig::new(1, N), None, &mut callbacks)
        .unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    let Callback::ModelCheckpoint(mc) = &callbacks[0] else {
        unreachable!("callbacks[0] is always ModelCheckpoint in this test");
    };
    assert_eq!(
        mc.best_value(),
        None,
        "保存失敗後も best が前進していてはならない（ロールバック契約）"
    );
    assert!(mc.best_state_dict().is_none());

    // 障害物を取り除き、以後は正常に保存できるようにする。
    std::fs::remove_file(&blocking_file).unwrap();

    // 2 回目: 同じ mc（fit 呼び出しをまたいで状態継続）で再度 fit する。
    // best がロールバックされ None のままなので、今回観測する損失値は
    // 必ず「改善」と判定され、再保存が試行される契約。
    model
        .fit_with_callbacks(&x, &y, FitConfig::new(1, N), None, &mut callbacks)
        .unwrap_or_else(|e| panic!("障害物除去後の 2 回目 fit は成功するはず: {e}"));
    let Callback::ModelCheckpoint(mc) = &callbacks[0] else {
        unreachable!("callbacks[0] is always ModelCheckpoint in this test");
    };
    assert!(
        mc.best_value().is_some(),
        "障害物除去後は再保存が試行され best が前進するはず"
    );
    assert!(path.is_file(), "2 回目の fit でファイルが作成されるはず");

    let restored = load_safetensors_f32(&path)
        .unwrap_or_else(|e| panic!("保存済みファイルの読み戻しに失敗: {e}"));
    let best_state = mc
        .best_state_dict()
        .expect("2 回目の fit で state が更新されているはず");
    assert!(
        state_dict_bit_exact(&restored, best_state),
        "ファイルから読み戻した state_dict が in-memory best_state_dict と bit 一致しない"
    );
}

// =====================================================================
// 6. to_file を付けても in-memory 挙動は変わらない（History・最終
//    パラメータが to_file あり／なしで bit 一致）
// =====================================================================

#[test]
fn to_file_does_not_change_in_memory_training_behavior() {
    let (x, y) = gen_regression_data(SEED_DATA);
    let dir = temp_dir_for("no-side-effect");
    let path = dir.join("ckpt.safetensors");

    let mut model_a = build_model();
    model_a
        .compile(
            Optimizer::Sgd(fandhe_ai::optim::SgdConfig::new(0.05)),
            Loss::Mse,
        )
        .unwrap();
    let mut callbacks_a = [Callback::ModelCheckpoint(
        ModelCheckpoint::new().monitor(Monitor::Loss),
    )];
    let history_a = model_a
        .fit_with_callbacks(&x, &y, FitConfig::new(3, N), None, &mut callbacks_a)
        .unwrap();

    let mut model_b = build_model();
    model_b
        .compile(
            Optimizer::Sgd(fandhe_ai::optim::SgdConfig::new(0.05)),
            Loss::Mse,
        )
        .unwrap();
    let mut callbacks_b = [Callback::ModelCheckpoint(
        ModelCheckpoint::new().monitor(Monitor::Loss).to_file(&path),
    )];
    let history_b = model_b
        .fit_with_callbacks(&x, &y, FitConfig::new(3, N), None, &mut callbacks_b)
        .unwrap();

    assert_eq!(history_a.loss.len(), history_b.loss.len());
    for (a, b) in history_a.loss.iter().zip(history_b.loss.iter()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }

    let params_a = model_a.trainable_parameters();
    let params_b = model_b.trainable_parameters();
    assert_eq!(params_a.len(), params_b.len());
    for (pa, pb) in params_a.iter().zip(params_b.iter()) {
        let a_bits: Vec<u32> = pa
            .contiguous()
            .as_slice()
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect();
        let b_bits: Vec<u32> = pb
            .contiguous()
            .as_slice()
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect();
        assert_eq!(a_bits, b_bits, "to_file 指定の有無でパラメータが乖離した");
    }
}

// =====================================================================
// 7. fit をまたぐ継続: 2 回目の fit で改善がなければファイル内容は
//    不変（best 継続契約。`callbacks.rs` モジュール冒頭 doc「epoch
//    番号の数え方」節）
// =====================================================================

#[test]
fn to_file_across_multiple_fit_calls_keeps_best_continuation_contract() {
    let (x, y) = gen_regression_data(SEED_DATA);
    let dir = temp_dir_for("cross-fit");
    let path = dir.join("ckpt.safetensors");

    let mut model = build_model();
    model
        .compile(
            Optimizer::Sgd(fandhe_ai::optim::SgdConfig::new(50.0)),
            Loss::Mse,
        )
        .unwrap();
    let mut mc = ModelCheckpoint::new().monitor(Monitor::Loss).to_file(&path);

    // 1 回目の fit: 大きな lr で epoch 0 が best になり epoch 1 以降は
    // 改善しない（発散）ため、best は epoch 0 のまま固定される。
    {
        let mut callbacks = [Callback::ModelCheckpoint(mc)];
        model
            .fit_with_callbacks(&x, &y, FitConfig::new(3, N), None, &mut callbacks)
            .unwrap();
        let Callback::ModelCheckpoint(m) = callbacks.into_iter().next().unwrap() else {
            unreachable!()
        };
        mc = m;
    }
    let after_first_fit = std::fs::read(&path).unwrap();
    let best_epoch_after_first = mc.best_epoch();

    // 2 回目の fit: 同じ発散条件が続くため best は更新されない
    // （ファイル内容も不変のはず）。
    {
        let mut callbacks = [Callback::ModelCheckpoint(mc)];
        model
            .fit_with_callbacks(&x, &y, FitConfig::new(2, N), None, &mut callbacks)
            .unwrap();
        let Callback::ModelCheckpoint(m) = callbacks.into_iter().next().unwrap() else {
            unreachable!()
        };
        mc = m;
    }
    let after_second_fit = std::fs::read(&path).unwrap();

    assert_eq!(
        mc.best_epoch(),
        best_epoch_after_first,
        "発散が続く条件で best_epoch が更新されてしまった"
    );
    assert_eq!(
        after_first_fit, after_second_fit,
        "best が更新されない fit 呼び出しでファイル内容が変化した"
    );
}

// =====================================================================
// CUDA／Metal 実機 #[ignore] テスト（R6。実機未到達のため未実測。
// `docs/perf/logs/model-checkpoint-file-2073/README.md` へ申し送り）
// =====================================================================

/// CPU で fit + `to_file` → `load_safetensors_f32` → 新規 `Sequential`
/// へ `load_state_dict` → `device` 上で forward → CPU `predict` と
/// `fandhe_ai_backend_cpu::parity::assert_parity`（REQ-2 統一複合
/// 判定。tolerance 不変）で比較する共通本体
/// （`compat_sequential_layers_backend_parity.rs::run_norm_embedding_
/// parity` と同型）。state_dict の bit 一致自体はホスト側の上記
/// テストで既に確認済みのため、ここではデバイス forward の parity
/// のみを見る。
#[cfg(any(test, doctest))]
#[allow(dead_code)]
fn run_checkpoint_roundtrip_on_device(device: fandhe_ai::Device) {
    let (x, y) = gen_regression_data(SEED_DATA);
    let dir = temp_dir_for("device-roundtrip");
    let path = dir.join("ckpt.safetensors");

    let mut model = build_model();
    model
        .compile(
            Optimizer::Sgd(fandhe_ai::optim::SgdConfig::new(0.05)),
            Loss::Mse,
        )
        .unwrap();
    let mut callbacks = [Callback::ModelCheckpoint(
        ModelCheckpoint::new()
            .monitor(Monitor::Loss)
            .save_best_only(false)
            .to_file(&path),
    )];
    model
        .fit_with_callbacks(&x, &y, FitConfig::new(2, N), None, &mut callbacks)
        .unwrap();

    let probe = Tensor::new(vec![0.1_f32, -0.2, 0.3, -0.4], &[1, D_IN]).unwrap();
    let cpu_output = model.predict(&probe).unwrap();

    let loaded = load_safetensors_f32(&path).unwrap();
    let mut restored = build_model();
    restored.load_state_dict(loaded).unwrap();

    let tape = fandhe_ai::tape_for(device)
        .expect("実機必須（本テストは #[ignore]。実行時は事前に到達確認する）");
    let probe_var = tape.var(&probe);
    let device_output = restored.forward(&tape, &probe_var).unwrap().to_tensor();

    fandhe_ai_backend_cpu::parity::assert_parity(
        "compat::Sequential ModelCheckpoint::to_file roundtrip device vs CPU",
        device_output.contiguous().as_slice().unwrap(),
        cpu_output.contiguous().as_slice().unwrap(),
    );
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）依存。docs/perf/logs/model-checkpoint-file-2073/README.md 参照"]
fn checkpoint_file_roundtrip_on_cuda() {
    run_checkpoint_roundtrip_on_device(fandhe_ai::Device::Cuda(0));
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機依存。docs/perf/logs/model-checkpoint-file-2073/README.md 参照"]
fn checkpoint_file_roundtrip_on_metal() {
    run_checkpoint_roundtrip_on_device(fandhe_ai::Device::Metal);
}
