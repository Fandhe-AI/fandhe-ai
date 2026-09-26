//! optimizer state_dict（save/load。イシュー #2174・親 #2131）の統合
//! テスト。`fandhe_ai_autodiff::nn::optim::OptimizerStateDict` を 9
//! optimizer（`AdamW`・`Adam`・`RmsProp`・`Adagrad`・`Lamb`・
//! `Adadelta`・`Adamax`・`NAdam`・`RAdam`）へ適用したときの契約を
//! 固定する（`nn::optim::state_dict` モジュール冒頭 doc「キー配置」
//! 「符号化」「`load_state_dict` の検証順」節）。
//!
//! 各 optimizer 共通の契約（キー集合・往復・検証失敗時の非変更）は
//! [`optimizer_state_dict_tests`] マクロで 9 種横断的に固定し、種別
//! マーカーによる異種 optimizer 拒否（`Adam`/`AdamW` はバッファ名が
//! 同じ `m`/`v` のため、マーカーがないと黙って受理されてしまう）は
//! マクロ外の専用テストで固定する。
//!
//! 実機（CUDA/Metal）非依存のため `#[ignore]` 分離は行わない。

use std::collections::{BTreeSet, HashMap};

use fandhe_ai_autodiff::AutodiffError;
use fandhe_ai_autodiff::nn::optim::OptimizerStateDict;
use fandhe_ai_tensor_core::Tensor;

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn tensor_bits(tensor: &Tensor<f32>) -> (Vec<usize>, Vec<u32>) {
    let data = tensor
        .as_slice()
        .expect("test fixture: 生成直後の Tensor は contiguous のはず")
        .iter()
        .map(|v| v.to_bits())
        .collect();
    (tensor.shape().to_vec(), data)
}

/// 2 つの state_dict マップが「キー集合完全一致・各テンソルの shape・
/// bit 列完全一致」であることを固定する（イシュー #2174 受け入れ基準
/// 「safetensors 変換を往復して bit 同一」の facade を介さない側の
/// 対応物）。
fn assert_state_dicts_bit_equal(
    a: &HashMap<String, Tensor<f32>>,
    b: &HashMap<String, Tensor<f32>>,
) {
    let ka: BTreeSet<&String> = a.keys().collect();
    let kb: BTreeSet<&String> = b.keys().collect();
    assert_eq!(ka, kb, "state_dict のキー集合が一致しない");
    for k in ka {
        assert_eq!(
            tensor_bits(&a[k]),
            tensor_bits(&b[k]),
            "state_dict のキー `{k}` の値が bit 一致しない"
        );
    }
}

/// 9 optimizer 共通の state_dict 契約を固定するテストモジュールを 1 つ
/// 生成する（マクロ引数は各 optimizer の `nn::optim::state_dict` 冒頭
/// doc「キー配置」節の表と対応させる）。
macro_rules! optimizer_state_dict_tests {
    (
        mod_name: $mod_name:ident,
        ty: $Ty:ty,
        cfg: $cfg_expr:expr,
        kind: $kind:expr,
        buffers: [$($buf:literal),+ $(,)?],
        has_beta1: $has_beta1:expr,
        has_beta2: $has_beta2:expr,
        has_mu: $has_mu:expr $(,)?
    ) => {
        mod $mod_name {
            use super::*;

            const BUFFERS: &[&str] = &[$($buf),+];

            fn new_opt() -> $Ty {
                <$Ty>::new($cfg_expr).expect("test fixture: 既定 config は常に有効")
            }

            fn base_expected_keys() -> BTreeSet<String> {
                let mut expected = BTreeSet::new();
                expected.insert(format!("__optimizer__.{}", $kind));
                expected.insert("step_count.u64_u16x4".to_string());
                expected.insert("num_slots.u64_u16x4".to_string());
                if $has_beta1 {
                    expected.insert("beta1_pow_t.f64_u16x4".to_string());
                }
                if $has_beta2 {
                    expected.insert("beta2_pow_t.f64_u16x4".to_string());
                }
                if $has_mu {
                    expected.insert("mu_product".to_string());
                }
                expected
            }

            #[test]
            fn key_set_before_first_step_has_no_slot_keys() {
                let opt = new_opt();
                let sd = opt
                    .state_dict()
                    .expect("state_dict: shape とデータ長は常に整合するため成功する");
                let actual: BTreeSet<String> = sd.keys().cloned().collect();
                assert_eq!(actual, base_expected_keys());
            }

            #[test]
            fn key_set_after_two_slots_matches_fixed_expectation() {
                let mut opt = new_opt();
                let p0 = t(vec![1.0, -2.0], &[2]);
                let g0 = t(vec![0.1, -0.05], &[2]);
                let p1 = t(vec![0.5, 1.5, -1.0], &[3]);
                let g1 = t(vec![0.02, -0.01, 0.03], &[3]);
                for _ in 0..3 {
                    opt.step(&[(&p0, &g0), (&p1, &g1)])
                        .expect("test fixture: shape は固定");
                }
                let sd = opt.state_dict().unwrap();
                let mut expected = base_expected_keys();
                for i in 0..2 {
                    for buf in BUFFERS {
                        expected.insert(format!("state.{i}.{buf}"));
                    }
                }
                let actual: BTreeSet<String> = sd.keys().cloned().collect();
                assert_eq!(actual, expected);
            }

            #[test]
            fn key_set_after_empty_slot_step_has_no_slot_keys() {
                let mut opt = new_opt();
                opt.step(&[]).expect("空スロット列の step は許容される");
                assert_eq!(opt.step_count(), 1);
                let sd = opt.state_dict().unwrap();
                for key in sd.keys() {
                    assert!(
                        !key.starts_with("state."),
                        "空スロット列で step した後にスロットキーが出てはならない: {key}"
                    );
                }
            }

            #[test]
            fn roundtrip_bit_exact_and_continues_training_identically() {
                let mut a = new_opt();
                let p0 = t(vec![1.0, -2.0], &[2]);
                let g0 = t(vec![0.1, -0.05], &[2]);
                let p1 = t(vec![0.5, 1.5, -1.0], &[3]);
                let g1 = t(vec![0.02, -0.01, 0.03], &[3]);

                let mut cur0 = p0.clone();
                let mut cur1 = p1.clone();
                for _ in 0..3 {
                    let out = a.step(&[(&cur0, &g0), (&cur1, &g1)]).unwrap();
                    cur0 = out[0].clone();
                    cur1 = out[1].clone();
                }

                let sd = a.state_dict().unwrap();
                let mut b = new_opt();
                b.load_state_dict(sd.clone())
                    .expect("有効な state_dict の load は成功するはず");
                let sd_b = b.state_dict().unwrap();
                assert_state_dicts_bit_equal(&sd, &sd_b);
                assert_eq!(b.step_count(), a.step_count());

                // 続けて A・B を同じ入力で 2 step 進め、出力 param が
                // bit 完全一致することを確認する（イシュー #2174 受け入れ
                // 基準「checkpoint から再開した学習の軌跡が、中断しない
                // 学習と一致する」）。
                let mut a0 = cur0.clone();
                let mut a1 = cur1.clone();
                let mut b0 = cur0.clone();
                let mut b1 = cur1.clone();
                for _ in 0..2 {
                    let out_a = a.step(&[(&a0, &g0), (&a1, &g1)]).unwrap();
                    let out_b = b.step(&[(&b0, &g0), (&b1, &g1)]).unwrap();
                    assert_eq!(tensor_bits(&out_a[0]), tensor_bits(&out_b[0]));
                    assert_eq!(tensor_bits(&out_a[1]), tensor_bits(&out_b[1]));
                    a0 = out_a[0].clone();
                    a1 = out_a[1].clone();
                    b0 = out_b[0].clone();
                    b1 = out_b[1].clone();
                }
            }

            #[test]
            fn roundtrip_on_empty_state_before_first_step() {
                let a = new_opt();
                let sd = a.state_dict().unwrap();
                let mut b = new_opt();
                b.load_state_dict(sd.clone()).unwrap();
                assert_state_dicts_bit_equal(&sd, &b.state_dict().unwrap());
                assert_eq!(b.step_count(), 0);
            }

            #[test]
            fn load_state_dict_rejects_missing_key_without_mutation() {
                // 2 スロット（index 0・index 1）で学習させ、index 0 の
                // バッファを 1 本だけ欠落させる。`num_slots` メタデータ
                // （`state_dict` モジュール冒頭 doc「キー配置」節）が
                // 独立に「2 スロットあるはず」を保持しているため、単一
                // バッファしか持たない optimizer（`Adagrad` 等）でも
                // index 1 側のバッファを残さずに検出できる。
                let mut opt = new_opt();
                let p0 = t(vec![1.0], &[1]);
                let g0 = t(vec![0.1], &[1]);
                let p1 = t(vec![1.0, 2.0], &[2]);
                let g1 = t(vec![0.1, 0.1], &[2]);
                opt.step(&[(&p0, &g0), (&p1, &g1)]).unwrap();
                let step_count_before = opt.step_count();

                let mut sd = opt.state_dict().unwrap();
                sd.remove(&format!("state.0.{}", BUFFERS[0]));
                let err = opt.load_state_dict(sd).unwrap_err();
                assert!(matches!(err, AutodiffError::InvalidArgument(_)));
                assert_eq!(opt.step_count(), step_count_before);
            }

            #[test]
            fn load_state_dict_rejects_trailing_slot_fully_missing_without_mutation() {
                // P0 レビュー指摘（イシュー #2174 PR #2304）で固定した
                // 是正: 末尾スロット（index 1）の全バッファを削除しても、
                // `num_slots` メタデータが「2 スロットあるはず」を独立に
                // 保持しているため、単一バッファ optimizer（`Adagrad`
                // 等）を含めて必ず「欠落キー」として拒否される
                // （旧実装は実在バッファキーの最大添字から `num_slots`
                // を推測していたため、この入力は index 0 だけの
                // state_dict として黙って受理されてしまっていた）。
                let mut opt = new_opt();
                let p0 = t(vec![1.0], &[1]);
                let g0 = t(vec![0.1], &[1]);
                let p1 = t(vec![1.0, 2.0], &[2]);
                let g1 = t(vec![0.1, 0.1], &[2]);
                opt.step(&[(&p0, &g0), (&p1, &g1)]).unwrap();
                let step_count_before = opt.step_count();

                let mut sd = opt.state_dict().unwrap();
                for buf in BUFFERS {
                    sd.remove(&format!("state.1.{buf}"));
                }
                let err = opt.load_state_dict(sd).unwrap_err();
                assert!(matches!(err, AutodiffError::InvalidArgument(_)));
                assert_eq!(opt.step_count(), step_count_before);
            }

            #[test]
            fn load_state_dict_rejects_oversized_num_slots_claim_without_mutation() {
                // P0 レビュー指摘（イシュー #2174 PR #2304）で固定した
                // 是正: `num_slots` メタデータへ実際のキー総数と整合
                // しない巨大値を書き込む攻撃入力は、`expected` 集合を
                // 構築する巨大ループへ入る前に拒否される。
                let mut opt = new_opt();
                let p = t(vec![1.0], &[1]);
                let g = t(vec![0.1], &[1]);
                opt.step(&[(&p, &g)]).unwrap();
                let step_count_before = opt.step_count();

                let mut sd = opt.state_dict().unwrap();
                // `num_slots = 1_000_000`（下位から 16bit ずつ u16x4
                // 符号化。`state_dict` モジュール冒頭 doc「符号化」節と
                // 同じ形式）を直接組み立てる。`decode_u16x4_tensor` は
                // 各語が有限・整数・`0.0..=65535.0` であることのみを
                // 検証するため、この値は符号化としては正当だが、実際に
                // 存在するキー数（1 ステップ・1 バッファ規模）とは
                // 到底整合しない。
                sd.insert(
                    "num_slots.u64_u16x4".to_string(),
                    t(vec![16960.0, 15.0, 0.0, 0.0], &[4]),
                );
                let err = opt.load_state_dict(sd).unwrap_err();
                assert!(matches!(err, AutodiffError::InvalidArgument(_)));
                assert_eq!(opt.step_count(), step_count_before);
            }

            #[test]
            fn load_state_dict_rejects_unexpected_key_without_mutation() {
                let mut opt = new_opt();
                let p = t(vec![1.0], &[1]);
                let g = t(vec![0.1], &[1]);
                opt.step(&[(&p, &g)]).unwrap();
                let step_count_before = opt.step_count();

                let mut sd = opt.state_dict().unwrap();
                sd.insert("unexpected.key".to_string(), t(vec![0.0], &[1]));
                let err = opt.load_state_dict(sd).unwrap_err();
                assert!(matches!(err, AutodiffError::InvalidArgument(_)));
                assert_eq!(opt.step_count(), step_count_before);
            }

            #[test]
            fn load_state_dict_rejects_non_canonical_slot_index_without_mutation() {
                let mut opt = new_opt();
                let p = t(vec![1.0], &[1]);
                let g = t(vec![0.1], &[1]);
                opt.step(&[(&p, &g)]).unwrap();
                let step_count_before = opt.step_count();

                let mut sd = opt.state_dict().unwrap();
                for buf in BUFFERS {
                    let old_key = format!("state.0.{buf}");
                    let value = sd.remove(&old_key).expect("test fixture: 1 step 後は必ず存在する");
                    // `01` は非正規表記（`"1".parse::<usize>().to_string()
                    // != "01"`）のため、`state_dict` モジュール冒頭 doc
                    // 「検証順」節どおり「余剰キー」として拒否される。
                    sd.insert(format!("state.01.{buf}"), value);
                }
                let err = opt.load_state_dict(sd).unwrap_err();
                assert!(matches!(err, AutodiffError::InvalidArgument(_)));
                assert_eq!(opt.step_count(), step_count_before);
            }

            #[test]
            fn load_state_dict_rejects_slot_shape_mismatch_without_mutation() {
                // 1 バッファのみを持つ optimizer（`Adagrad` 等）は同一
                // スロット内でバッファ同士の shape を比較しようがないため
                // 対象外（早期 return）。
                if BUFFERS.len() < 2 {
                    return;
                }
                let mut opt = new_opt();
                let p = t(vec![1.0, 2.0], &[2]);
                let g = t(vec![0.1, 0.1], &[2]);
                opt.step(&[(&p, &g)]).unwrap();
                let step_count_before = opt.step_count();

                let mut sd = opt.state_dict().unwrap();
                sd.insert(
                    format!("state.0.{}", BUFFERS[0]),
                    t(vec![1.0, 2.0, 3.0], &[3]),
                );
                let err = opt.load_state_dict(sd).unwrap_err();
                assert!(matches!(err, AutodiffError::InvalidArgument(_)));
                assert_eq!(opt.step_count(), step_count_before);
            }

            #[test]
            fn load_state_dict_rejects_malformed_step_count_word_without_mutation() {
                let mut opt = new_opt();
                let p = t(vec![1.0], &[1]);
                let g = t(vec![0.1], &[1]);
                opt.step(&[(&p, &g)]).unwrap();
                let step_count_before = opt.step_count();

                let mut sd = opt.state_dict().unwrap();
                sd.insert(
                    "step_count.u64_u16x4".to_string(),
                    t(vec![70000.0, 0.0, 0.0, 0.0], &[4]),
                );
                let err = opt.load_state_dict(sd).unwrap_err();
                assert!(matches!(err, AutodiffError::InvalidArgument(_)));
                assert_eq!(opt.step_count(), step_count_before);
            }

            #[test]
            fn load_state_dict_rejects_scalar_shape_violation_without_mutation() {
                let mut opt = new_opt();
                let p = t(vec![1.0], &[1]);
                let g = t(vec![0.1], &[1]);
                opt.step(&[(&p, &g)]).unwrap();
                let step_count_before = opt.step_count();

                let mut sd = opt.state_dict().unwrap();
                sd.insert(
                    "step_count.u64_u16x4".to_string(),
                    t(vec![0.0, 0.0, 0.0], &[3]),
                );
                let err = opt.load_state_dict(sd).unwrap_err();
                assert!(matches!(err, AutodiffError::InvalidArgument(_)));
                assert_eq!(opt.step_count(), step_count_before);
            }

            #[test]
            fn load_state_dict_rejects_marker_value_mismatch_without_mutation() {
                let mut opt = new_opt();
                let p = t(vec![1.0], &[1]);
                let g = t(vec![0.1], &[1]);
                opt.step(&[(&p, &g)]).unwrap();
                let step_count_before = opt.step_count();

                let mut sd = opt.state_dict().unwrap();
                sd.insert(format!("__optimizer__.{}", $kind), t(vec![2.0], &[1]));
                let err = opt.load_state_dict(sd).unwrap_err();
                assert!(matches!(err, AutodiffError::InvalidArgument(_)));
                assert_eq!(opt.step_count(), step_count_before);
            }

            #[test]
            fn load_state_dict_rejects_beta_pow_t_out_of_range_without_mutation() {
                if !($has_beta1 || $has_beta2) {
                    return;
                }
                let mut opt = new_opt();
                let p = t(vec![1.0], &[1]);
                let g = t(vec![0.1], &[1]);
                opt.step(&[(&p, &g)]).unwrap();
                let step_count_before = opt.step_count();

                let key = if $has_beta1 {
                    "beta1_pow_t.f64_u16x4"
                } else {
                    "beta2_pow_t.f64_u16x4"
                };
                // 1.5 を u16x4（f64 bit 列）で符号化する。
                let bits = 1.5f64.to_bits();
                let words: Vec<f32> = (0..4)
                    .map(|i| (((bits >> (16 * i)) & 0xFFFF) as u32) as f32)
                    .collect();
                let mut sd = opt.state_dict().unwrap();
                sd.insert(key.to_string(), t(words, &[4]));
                let err = opt.load_state_dict(sd).unwrap_err();
                assert!(matches!(err, AutodiffError::InvalidArgument(_)));
                assert_eq!(opt.step_count(), step_count_before);
            }

            #[test]
            fn load_state_dict_rejects_non_finite_mu_product_without_mutation() {
                if !$has_mu {
                    return;
                }
                let mut opt = new_opt();
                let p = t(vec![1.0], &[1]);
                let g = t(vec![0.1], &[1]);
                opt.step(&[(&p, &g)]).unwrap();
                let step_count_before = opt.step_count();

                let mut sd = opt.state_dict().unwrap();
                sd.insert("mu_product".to_string(), t(vec![f32::NAN], &[1]));
                let err = opt.load_state_dict(sd).unwrap_err();
                assert!(matches!(err, AutodiffError::InvalidArgument(_)));
                assert_eq!(opt.step_count(), step_count_before);
            }

            #[test]
            fn load_state_dict_accepts_non_contiguous_buffer_view() {
                let mut opt = new_opt();
                let p = t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
                let g = t(vec![0.1, 0.1, 0.1, 0.1], &[2, 2]);
                opt.step(&[(&p, &g)]).unwrap();
                let sd = opt.state_dict().unwrap();

                // 転置 view（非 contiguous）を経由しても論理 row-major 順
                // で受理されることを固定する（`state_dict` モジュール
                // 冒頭 doc「非 contiguous のバッファ入力」節）。
                let mut view_sd: HashMap<String, Tensor<f32>> = HashMap::new();
                for (k, v) in &sd {
                    if k.starts_with("state.") {
                        let transposed = v
                            .transpose_2d()
                            .expect("test fixture: スロット buffer は 2 次元 shape");
                        let back = transposed
                            .transpose_2d()
                            .expect("test fixture: 2 回転置で元の論理値に戻る");
                        view_sd.insert(k.clone(), back);
                    } else {
                        view_sd.insert(k.clone(), v.clone());
                    }
                }

                let mut b = new_opt();
                b.load_state_dict(view_sd).unwrap();
                assert_state_dicts_bit_equal(&sd, &b.state_dict().unwrap());
            }
        }
    };
}

optimizer_state_dict_tests!(
    mod_name: adamw_state_dict,
    ty: fandhe_ai_autodiff::nn::optim::AdamW,
    cfg: fandhe_ai_autodiff::nn::optim::AdamWConfig::default(),
    kind: "adamw",
    buffers: ["m", "v"],
    has_beta1: true,
    has_beta2: true,
    has_mu: false,
);

optimizer_state_dict_tests!(
    mod_name: adam_state_dict,
    ty: fandhe_ai_autodiff::nn::optim::Adam,
    cfg: fandhe_ai_autodiff::nn::optim::AdamConfig::default(),
    kind: "adam",
    buffers: ["m", "v"],
    has_beta1: true,
    has_beta2: true,
    has_mu: false,
);

optimizer_state_dict_tests!(
    mod_name: rmsprop_state_dict,
    ty: fandhe_ai_autodiff::nn::optim::RmsProp,
    cfg: fandhe_ai_autodiff::nn::optim::RmsPropConfig::default(),
    kind: "rmsprop",
    buffers: ["square_avg", "grad_avg", "momentum_buffer"],
    has_beta1: false,
    has_beta2: false,
    has_mu: false,
);

optimizer_state_dict_tests!(
    mod_name: adagrad_state_dict,
    ty: fandhe_ai_autodiff::nn::optim::Adagrad,
    cfg: fandhe_ai_autodiff::nn::optim::AdagradConfig::default(),
    kind: "adagrad",
    buffers: ["state_sum"],
    has_beta1: false,
    has_beta2: false,
    has_mu: false,
);

optimizer_state_dict_tests!(
    mod_name: lamb_state_dict,
    ty: fandhe_ai_autodiff::nn::optim::Lamb,
    cfg: fandhe_ai_autodiff::nn::optim::LambConfig::default(),
    kind: "lamb",
    buffers: ["m", "v"],
    has_beta1: true,
    has_beta2: true,
    has_mu: false,
);

optimizer_state_dict_tests!(
    mod_name: adadelta_state_dict,
    ty: fandhe_ai_autodiff::nn::optim::Adadelta,
    cfg: fandhe_ai_autodiff::nn::optim::AdadeltaConfig::default(),
    kind: "adadelta",
    buffers: ["square_avg", "acc_delta"],
    has_beta1: false,
    has_beta2: false,
    has_mu: false,
);

optimizer_state_dict_tests!(
    mod_name: adamax_state_dict,
    ty: fandhe_ai_autodiff::nn::optim::Adamax,
    cfg: fandhe_ai_autodiff::nn::optim::AdamaxConfig::default(),
    kind: "adamax",
    buffers: ["exp_avg", "exp_inf"],
    has_beta1: true,
    has_beta2: false,
    has_mu: false,
);

optimizer_state_dict_tests!(
    mod_name: nadam_state_dict,
    ty: fandhe_ai_autodiff::nn::optim::NAdam,
    cfg: fandhe_ai_autodiff::nn::optim::NAdamConfig::default(),
    kind: "nadam",
    buffers: ["exp_avg", "exp_avg_sq"],
    has_beta1: false,
    has_beta2: true,
    has_mu: true,
);

optimizer_state_dict_tests!(
    mod_name: radam_state_dict,
    ty: fandhe_ai_autodiff::nn::optim::RAdam,
    cfg: fandhe_ai_autodiff::nn::optim::RAdamConfig::default(),
    kind: "radam",
    buffers: ["exp_avg", "exp_avg_sq"],
    has_beta1: true,
    has_beta2: true,
    has_mu: false,
);

// =========================================================================
// 種別マーカーによる異種 optimizer 拒否（マクロ外・専用テスト）。
// `Adam`／`AdamW` はバッファ名（`m`／`v`）・スカラー構造
// （`beta1_pow_t`／`beta2_pow_t` あり・`mu_product` なし）が完全に同一
// のため、マーカーがないと「キー集合完全一致」検査だけでは異種の
// 取り違えを検出できない。マーカー検証（`state_dict` モジュール冒頭
// doc「キー配置」節）がこれを fail-closed で拒否することを固定する。
// =========================================================================

#[test]
fn adam_state_dict_is_rejected_by_adamw_load_state_dict() {
    use fandhe_ai_autodiff::nn::optim::{Adam, AdamConfig, AdamW, AdamWConfig};

    let mut adam = Adam::new(AdamConfig::default()).unwrap();
    let p = t(vec![1.0, 2.0], &[2]);
    let g = t(vec![0.1, 0.2], &[2]);
    adam.step(&[(&p, &g)]).unwrap();
    let adam_sd = adam.state_dict().unwrap();

    let mut adamw = AdamW::new(AdamWConfig::default()).unwrap();
    adamw.step(&[(&p, &g)]).unwrap();
    let step_count_before = adamw.step_count();

    let err = adamw.load_state_dict(adam_sd).unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    assert_eq!(adamw.step_count(), step_count_before);
}

#[test]
fn radam_state_dict_is_rejected_by_nadam_load_state_dict() {
    // `NAdam`（`beta2_pow_t`／`mu_product`）と `RAdam`（`beta1_pow_t`／
    // `beta2_pow_t`）はバッファ名（`exp_avg`／`exp_avg_sq`）が同一だが、
    // スカラー構造も異なるため、この組は「キー集合完全一致」検査でも
    // 拒否される（マーカー検査は多層防御の一枚）。
    use fandhe_ai_autodiff::nn::optim::{NAdam, NAdamConfig, RAdam, RAdamConfig};

    let mut radam = RAdam::new(RAdamConfig::default()).unwrap();
    let p = t(vec![1.0, 2.0], &[2]);
    let g = t(vec![0.1, 0.2], &[2]);
    radam.step(&[(&p, &g)]).unwrap();
    let radam_sd = radam.state_dict().unwrap();

    let mut nadam = NAdam::new(NAdamConfig::default()).unwrap();
    nadam.step(&[(&p, &g)]).unwrap();
    let step_count_before = nadam.step_count();

    let err = nadam.load_state_dict(radam_sd).unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    assert_eq!(nadam.step_count(), step_count_before);
}
