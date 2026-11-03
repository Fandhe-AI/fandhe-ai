//! `Var::lp_pool1d`／`2d`／`3d`（イシュー #3825）の autodiff 統合テスト。閉形式・中央差分の `ops` と PyTorch
//! 実行値 fixture 突合の `parity` を 1 バイナリに束ね、リンクされるテストバイナリ数（CI の target 容量）を抑える。

mod common;

mod ops {
    //! `Var::lp_pool1d`／`lp_pool2d`／`lp_pool3d`（イシュー #3825・設計 `docs/autodiff-pool3d-ops-decision.md`
    //! §19・承認 #3768）の受け入れテスト（CPU・`NaiveOps`）。
    //!
    //! 合成のみ（新規 `Op` なし）なので、ここでは (1) 閉形式の値、(2) 中央差分との勾配突合、(3) 拒否系と
    //! 孤児ノードが残らないこと、(4) 決定性、(5) `create_graph` が panic せず型付きエラーになること
    //! （承認 Q3 (a)）を確かめる。PyTorch 参照値との突合は `fixtures/lp-pool-pytorch-reference/` を参照。

    use super::common;

    use fandhe_ai_autodiff::{AutodiffError, Tape};
    use fandhe_ai_tensor_core::Tensor;

    fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
        Tensor::new(data, shape).expect("test fixture: shape とデータ長は一致させている")
    }

    fn flat(tensor: &Tensor<f32>) -> Vec<f32> {
        tensor.contiguous().as_slice().unwrap().to_vec()
    }

    fn positive(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| 0.5 + ((i as f32) * 0.37).sin().abs())
            .collect()
    }

    #[test]
    fn closed_form_3d_with_stride() {
        let tape = Tape::new_with_ops(common::naive_ops());
        // [1,1,2,2,4] を stride=[1,1,2] の [2,2,2] 窓で走査 → 出力 [1,1,1,1,2]。
        let x = tape.var(&t(vec![1.0; 16], &[1, 1, 2, 2, 4]));
        let y = x.lp_pool3d(2.0, [2, 2, 2], Some([1, 1, 2]), false).unwrap();
        assert_eq!(y.to_tensor().shape(), &[1, 1, 1, 1, 2]);
        let v = flat(&y.to_tensor());
        for e in v {
            assert!((e - 8.0f32.sqrt()).abs() < 1e-6);
        }
    }

    #[test]
    fn gradient_matches_central_difference() {
        let shape = [1usize, 1, 4, 4];
        let n: usize = shape.iter().product();
        let base = positive(n);
        let loss_at = |x: &[f32], p: f32| -> f64 {
            let tape = Tape::new_with_ops(common::naive_ops());
            let xv = tape.var(&t(x.to_vec(), &shape));
            let y = xv.lp_pool2d(p, [2, 2], None, false).unwrap();
            flat(&y.to_tensor()).iter().map(|&v| v as f64).sum()
        };
        for p in [1.0f32, 2.0, 3.0, 1.5] {
            let tape = Tape::new_with_ops(common::naive_ops());
            let xv = tape.var(&t(base.clone(), &shape));
            let loss = xv
                .lp_pool2d(p, [2, 2], None, false)
                .unwrap()
                .sum(None)
                .unwrap();
            let grads = tape.backward(&loss).unwrap();
            let g = flat(grads.get(&xv).unwrap().unwrap());
            let h = 1e-2f32;
            for i in 0..n {
                let (mut xp, mut xm) = (base.clone(), base.clone());
                xp[i] += h;
                xm[i] -= h;
                let num = (loss_at(&xp, p) - loss_at(&xm, p)) / (2.0 * h as f64);
                assert!(
                    common::req2_close(g[i] as f64, num),
                    "p={p} i={i}: analytic={} numeric={num}",
                    g[i]
                );
            }
        }
    }

    #[test]
    fn zero_window_gradient_is_zero_not_nan() {
        for p in [2.0f32, 1.5] {
            let tape = Tape::new_with_ops(common::naive_ops());
            let xv = tape.var(&t(
                vec![0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0],
                &[1, 1, 2, 4],
            ));
            let loss = xv
                .lp_pool2d(p, [1, 4], None, false)
                .unwrap()
                .sum(None)
                .unwrap();
            let grads = tape.backward(&loss).unwrap();
            let g = flat(grads.get(&xv).unwrap().unwrap());
            assert_eq!(&g[..4], &[0.0; 4], "p={p}");
            assert!(g[4..].iter().all(|v| v.is_finite()));
        }
    }

    #[test]
    fn fractional_p_below_one_with_zero_in_window_gives_nonfinite_gradient() {
        // 記録用: 0 < p < 1 かつ窓内に 0 があると x^p の勾配が非有限になる（PyTorch も同様）。
        let tape = Tape::new_with_ops(common::naive_ops());
        let xv = tape.var(&t(vec![0.0, 1.0], &[1, 1, 2]));
        let loss = xv
            .lp_pool1d(0.5, 2, None, false)
            .unwrap()
            .sum(None)
            .unwrap();
        let grads = tape.backward(&loss).unwrap();
        let g = flat(grads.get(&xv).unwrap().unwrap());
        assert!(g.iter().any(|v| !v.is_finite()), "{g:?}");
    }

    #[test]
    fn invalid_arguments_are_rejected_without_orphan_nodes() {
        let tape = Tape::new_with_ops(common::naive_ops());
        let x = tape.var(&t(vec![1.0; 16], &[1, 1, 4, 4]));
        let x3 = tape.var(&t(vec![1.0; 8], &[1, 1, 2, 2, 2]));
        let before = tape.len();
        for bad in [0.0f32, -2.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(matches!(
                x.lp_pool2d(bad, [2, 2], None, false),
                Err(AutodiffError::InvalidArgument(_))
            ));
            assert!(x.lp_pool1d(bad, 2, None, false).is_err());
            assert!(x3.lp_pool3d(bad, [2, 2, 2], None, false).is_err());
        }
        assert!(x.lp_pool2d(2.0, [2, 2], None, true).is_err());
        assert!(x.lp_pool2d(2.0, [0, 2], None, false).is_err());
        assert!(x.lp_pool2d(2.0, [2, 2], Some([0, 1]), false).is_err());
        assert!(x.lp_pool2d(2.0, [8, 8], None, false).is_err());
        assert!(x.lp_pool1d(2.0, 2, None, false).is_err());
        assert!(x.lp_pool3d(2.0, [2, 2, 2], None, false).is_err());
        assert!(x3.lp_pool2d(2.0, [2, 2], None, false).is_err());
        assert!(x3.lp_pool3d(2.0, [usize::MAX, 2, 2], None, false).is_err());
        assert!(
            x3.lp_pool3d(2.0, [1 << 13, 1 << 13, 1], None, false)
                .is_err()
        );
        assert_eq!(tape.len(), before, "拒否時にノードを積まない");
    }

    #[test]
    fn forward_is_deterministic() {
        let run = || {
            let tape = Tape::new_with_ops(common::naive_ops());
            let x = tape.var(&t(positive(32), &[1, 2, 4, 4]));
            flat(
                &x.lp_pool2d(3.0, [2, 2], Some([1, 1]), false)
                    .unwrap()
                    .to_tensor(),
            )
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>()
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn create_graph_returns_typed_error_instead_of_panic() {
        // 承認 Q3 (a): `Sign` が replay できないため、二階微分は型付きエラーになる（現行挙動の固定）。
        let tape = Tape::new_with_ops(common::naive_ops());
        let child = Tape::new_with_ops(common::naive_ops());
        let x = tape.var(&t(positive(4), &[1, 1, 4]));
        let loss = x.lp_pool1d(2.0, 2, None, false).unwrap().sum(None).unwrap();
        assert!(tape.backward_create_graph(&loss, &child).is_err());
    }
}

mod parity {
    //! `Var::lp_pool1d`／`lp_pool2d`／`lp_pool3d`（イシュー #3825）を実 PyTorch 2.14 の実行値 fixture
    //! （`tests/fixtures/lp-pool-pytorch-reference/lp_pool_reference.json`。生成条件・sha256 は同ディレクトリの
    //! README.md）と突合する統合テスト。
    //!
    //! 判定は REQ-2 統一複合判定（`common::req2_close`。tolerance 定数は新設しない）。NaN は NaN 同士、
    //! ±inf は厳密一致。PyTorch は f32 累積・本実装は `avg_pool` の `f64` 累積のため bit 一致は求めない。
    //! 設計記録: `docs/autodiff-pool3d-ops-decision.md` §19 と LPPool 実施記録の節。

    use super::common;

    use fandhe_ai_autodiff::Tape;
    use fandhe_ai_tensor_core::Tensor;
    use serde::Deserialize;

    #[derive(Deserialize)]
    struct Fixture {
        torch_version: String,
        cases: Vec<Case>,
    }

    #[derive(Deserialize)]
    struct Case {
        name: String,
        dim: usize,
        norm_type: f32,
        kernel_size: Vec<usize>,
        stride: Option<Vec<usize>>,
        in_shape: Vec<usize>,
        out_shape: Vec<usize>,
        x_bits: Vec<u32>,
        g_bits: Vec<u32>,
        out_bits: Vec<u32>,
        grad_bits: Vec<u32>,
    }

    fn from_bits(bits: &[u32]) -> Vec<f32> {
        bits.iter().map(|&b| f32::from_bits(b)).collect()
    }

    /// NaN は NaN 同士、±inf は厳密、有限は REQ-2 統一複合判定。
    fn class_close(a: f32, e: f32) -> bool {
        if e.is_nan() {
            a.is_nan()
        } else if e.is_infinite() {
            a == e
        } else {
            common::req2_close(f64::from(a), f64::from(e))
        }
    }

    fn assert_close_all(actual: &[f32], expected: &[f32], context: &str) {
        assert_eq!(actual.len(), expected.len(), "{context}: 長さ不一致");
        for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
            assert!(class_close(a, e), "{context}[{i}]: actual={a} expected={e}");
        }
    }

    #[test]
    fn lp_pool_matches_pytorch_reference_forward_and_backward() {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/lp-pool-pytorch-reference/lp_pool_reference.json");
        let raw = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("fixture 読込に失敗: {} ({e})", path.display()));
        let fixture: Fixture =
            serde_json::from_str(&raw).unwrap_or_else(|e| panic!("fixture のパースに失敗: {e}"));
        assert!(
            fixture.torch_version.starts_with("2.14"),
            "fixture は PyTorch 2.14 系の実行値である必要がある: {}",
            fixture.torch_version
        );
        assert!(fixture.cases.len() >= 20);
        let mut dims = [0usize; 3];
        let mut nan_cases = 0usize;
        for case in &fixture.cases {
            dims[case.dim - 1] += 1;
            let tape = Tape::new_with_ops(common::naive_ops());
            let x = tape.var(&Tensor::new(from_bits(&case.x_bits), &case.in_shape).unwrap());
            let (p, k, s) = (case.norm_type, &case.kernel_size, &case.stride);
            let y = match case.dim {
                1 => x.lp_pool1d(p, k[0], s.as_ref().map(|v| v[0]), false),
                2 => x.lp_pool2d(p, [k[0], k[1]], s.as_ref().map(|v| [v[0], v[1]]), false),
                3 => x.lp_pool3d(
                    p,
                    [k[0], k[1], k[2]],
                    s.as_ref().map(|v| [v[0], v[1], v[2]]),
                    false,
                ),
                d => panic!("dim={d}"),
            }
            .unwrap_or_else(|e| panic!("{}: {e:?}", case.name));
            let out = y.to_tensor();
            assert_eq!(
                out.shape(),
                case.out_shape.as_slice(),
                "{}: shape",
                case.name
            );
            assert_close_all(
                &out.host_slice(),
                &from_bits(&case.out_bits),
                &format!("{} forward", case.name),
            );
            let out_has_nan = from_bits(&case.out_bits).iter().any(|v| v.is_nan());
            if out_has_nan {
                nan_cases += 1;
            }
            let g = tape.var_no_grad(&Tensor::new(from_bits(&case.g_bits), out.shape()).unwrap());
            let loss = y.mul(&g).unwrap().sum(None).unwrap();
            let grads = tape.backward(&loss).unwrap();
            let got = grads.get(&x).unwrap().unwrap().host_slice().into_owned();
            let want = from_bits(&case.grad_bits);
            if out_has_nan {
                // forward に NaN を含む（負の入力 × 非整数 p）ケース。PyTorch の勾配は NaN 要素を含むが、
                // 本実装は `pow_scalar` の VJP と上流 0 の積が 0 になる要素があり、全要素が NaN にはならない（既知の差分。実装記録に記載）。
                // 参照勾配が NaN の要素だけを比較から除外し、有限の参照勾配は検証する。除外した要素は
                // 本実装が NaN か 0 のいずれかを返す挙動を独立に固定する（特殊値処理の回帰を検出する）。
                assert_eq!(got.len(), want.len(), "{} grad 長", case.name);
                let mut known_diff = 0usize;
                let mut zero_diff = 0usize;
                for (i, (g_i, w_i)) in got.iter().zip(want.iter()).enumerate() {
                    if w_i.is_nan() {
                        known_diff += 1;
                        // 差分として記録した要素の挙動を固定する: NaN（PyTorch と一致）か厳密な 0（既知の差分）の
                        // いずれかで、有限の非零値へ変わったら回帰として検出する。
                        assert!(
                            g_i.is_nan() || *g_i == 0.0,
                            "{} grad[{i}]: NaN 参照要素は NaN または 0 のみ許す: {g_i}",
                            case.name
                        );
                        if *g_i == 0.0 {
                            zero_diff += 1;
                        }
                    } else {
                        assert_close_all(
                            std::slice::from_ref(g_i),
                            std::slice::from_ref(w_i),
                            &format!("{} grad[{i}]", case.name),
                        );
                    }
                }
                assert!(
                    known_diff >= 1,
                    "{}: NaN 参照勾配が 1 件以上ある",
                    case.name
                );
                assert!(
                    zero_diff >= 1,
                    "{}: 0 を返す既知の差分が 1 件以上ある",
                    case.name
                );
            } else {
                assert_close_all(&got, &want, &format!("{} grad", case.name));
            }
        }
        assert!(dims.iter().all(|&n| n >= 5), "1d/2d/3d の件数: {dims:?}");
        assert!(nan_cases >= 1, "NaN ケースが fixture に含まれる");
    }
}
