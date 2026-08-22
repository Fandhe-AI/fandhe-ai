//! `fandhe_ai_backend_cpu::reduction`（TASK-1.6c・#23）の受け入れ条件「軸指定 reduction の
//! 数値が期待値と一致する」に対応する統合テスト。
//!
//! `src/reduction.rs` のインライン単体テスト（決定性・空縮約・shape 検査）を
//! 補い、多次元（rank 3）データでの軸指定 sum/max/mean・rank0 全縮約・
//! 非 contiguous view の bit 一致を検証する。
//!
//! #25（TASK-1.6e）棚卸しで、非正方 rank3 の max/mean・非 contiguous
//! 全縮約の max/mean を追加した（既存は sum 中心のカバレッジだった）。
//! CHUNK 境界決定性・サイズ 1 軸は `CHUNK` 定数参照のため
//! `src/reduction.rs` のインライン `#[cfg(test)]` へ追加した。

use fandhe_ai_backend_cpu::reduction::{ReduceError, max, mean, sum};
use fandhe_ai_tensor_core::Tensor;

/// shape [2, 3, 4] の既知データで axis=0/1/2 それぞれの sum を手計算期待値と
/// 完全一致で検証する（受け入れ条件の直接対応）。
#[test]
fn sum_axis_matches_hand_computed_expected_values_rank3() {
    let data: Vec<f32> = (0..24).map(|v| v as f32).collect();
    let t = Tensor::<f32>::new(data, &[2, 3, 4]).unwrap();

    // axis=0: shape [3, 4]。出力[j][k] = t[0][j][k] + t[1][j][k]
    let out0 = sum(&t, Some(0)).unwrap();
    assert_eq!(out0.shape(), &[3, 4]);
    for j in 0..3 {
        for k in 0..4 {
            let expected = t.get(&[0, j, k]).unwrap() + t.get(&[1, j, k]).unwrap();
            assert_eq!(out0.get(&[j, k]).unwrap(), expected);
        }
    }

    // axis=1: shape [2, 4]。出力[i][k] = Σ_j t[i][j][k]
    let out1 = sum(&t, Some(1)).unwrap();
    assert_eq!(out1.shape(), &[2, 4]);
    for i in 0..2 {
        for k in 0..4 {
            let expected: f32 = (0..3).map(|j| t.get(&[i, j, k]).unwrap()).sum();
            assert_eq!(out1.get(&[i, k]).unwrap(), expected);
        }
    }

    // axis=2: shape [2, 3]。出力[i][j] = Σ_k t[i][j][k]
    let out2 = sum(&t, Some(2)).unwrap();
    assert_eq!(out2.shape(), &[2, 3]);
    for i in 0..2 {
        for j in 0..3 {
            let expected: f32 = (0..4).map(|k| t.get(&[i, j, k]).unwrap()).sum();
            assert_eq!(out2.get(&[i, j]).unwrap(), expected);
        }
    }
}

#[test]
fn max_axis_matches_hand_computed_expected_values_rank3() {
    // 決定的疑似データ（線形合同法もどきの単純な式）で max を検証する。
    let data: Vec<f32> = (0..24).map(|v| ((v * 37 + 11) % 24) as f32).collect();
    let t = Tensor::<f32>::new(data, &[2, 3, 4]).unwrap();

    let out = max(&t, Some(1)).unwrap();
    assert_eq!(out.shape(), &[2, 4]);
    for i in 0..2 {
        for k in 0..4 {
            let expected = (0..3)
                .map(|j| t.get(&[i, j, k]).unwrap())
                .fold(f32::NEG_INFINITY, f32::max);
            assert_eq!(out.get(&[i, k]).unwrap(), expected);
        }
    }
}

#[test]
fn mean_rounds_once_and_matches_expected() {
    let t = Tensor::<f32>::new(vec![1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
    let out = mean(&t, None).unwrap();
    assert_eq!(out.shape(), &[]);
    assert_eq!(out.get(&[]).unwrap(), 2.5);
}

#[test]
fn mean_axis_matches_expected() {
    // shape [2, 3]: [[1,2,3],[4,5,6]] -> axis=1 の mean は [2.0, 5.0]
    let t = Tensor::<f32>::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let out = mean(&t, Some(1)).unwrap();
    assert_eq!(out.shape(), &[2]);
    assert_eq!(out.get(&[0]).unwrap(), 2.0);
    assert_eq!(out.get(&[1]).unwrap(), 5.0);
}

/// `dim=None` は rank 0 スカラーを返し、期待値と一致する。
#[test]
fn full_reduction_returns_rank0_scalar() {
    let t = Tensor::<f32>::new((0..12).map(|v| v as f32).collect(), &[3, 4]).unwrap();
    let s = sum(&t, None).unwrap();
    assert_eq!(s.shape(), &[]);
    let expected: f32 = (0..12).map(|v| v as f32).sum();
    assert_eq!(s.get(&[]).unwrap(), expected);

    let m = max(&t, None).unwrap();
    assert_eq!(m.shape(), &[]);
    assert_eq!(m.get(&[]).unwrap(), 11.0);
}

/// transpose 済み（非 contiguous）view への軸指定 reduction が、
/// `contiguous()` 実体化後と bit 一致することを検証する。
#[test]
fn non_contiguous_view_matches_contiguous_bitwise() {
    let t = Tensor::<f32>::new((0..12).map(|v| v as f32 * 1.5).collect(), &[3, 4]).unwrap();
    let tt = t.transpose(0, 1).unwrap(); // shape [4, 3]
    assert!(tt.as_slice().is_none());
    let c = tt.contiguous();

    for axis in [0usize, 1] {
        let a = sum(&tt, Some(axis)).unwrap();
        let b = sum(&c, Some(axis)).unwrap();
        assert_eq!(a.shape(), b.shape());
        for i in 0..a.numel() {
            let idx = if a.rank() == 1 { vec![i] } else { vec![] };
            assert_eq!(
                a.get(&idx).unwrap().to_bits(),
                b.get(&idx).unwrap().to_bits()
            );
        }
    }

    // 全縮約（dim=None）でも非 contiguous と contiguous が bit 一致する。
    let full_a = sum(&tt, None).unwrap();
    let full_b = sum(&c, None).unwrap();
    assert_eq!(
        full_a.get(&[]).unwrap().to_bits(),
        full_b.get(&[]).unwrap().to_bits()
    );
}

/// 決定的疑似乱数データで CHUNK 超の全縮約 sum が、スレッド数 1/4 の rayon
/// プール間で to_bits() 完全一致し、かつ同一累積順序の逐次 naive 実装とも
/// 一致することを検証する（PoC-v2-5 の「逐次固定順序で bit 一致」前提）。
#[test]
fn full_reduction_deterministic_across_thread_pools() {
    // 固定シードの簡易線形合同法で決定的疑似乱数データを生成する。
    let mut seed: u64 = 0x243F_6A88_85A3_08D3;
    let n = 4096 * 3 + 101;
    let data: Vec<f32> = (0..n)
        .map(|_| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            (((seed >> 40) % 2000) as f32 - 1000.0) * 0.01
        })
        .collect();
    let t = Tensor::<f32>::new(data.clone(), &[n]).unwrap();

    let single = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("failed to build single-thread rayon pool");
    let multi = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .expect("failed to build 4-thread rayon pool");

    let a = single.install(|| sum(&t, None).unwrap());
    let b = multi.install(|| sum(&t, None).unwrap());
    assert_eq!(a.get(&[]).unwrap().to_bits(), b.get(&[]).unwrap().to_bits());

    // 参照実装（同一累積順序の逐次 naive 実装）との bit 一致。
    // 浮動小数点加算は結合則を満たさないため、単純な左から右への
    // fold ではなく src/reduction.rs の実装と同一の「CHUNK 単位で
    // チャンク内を逐次累積 → チャンク結果をチャンク番号順に逐次結合」
    // という累積順序を再現する（決定性契約はチャンク分割込みの
    // 演算順序についてのものであり、任意の累積順序との bit 一致を
    // 保証するものではない）。
    const CHUNK: usize = 4096;
    let naive = data
        .chunks(CHUNK)
        .map(|chunk| chunk.iter().fold(0.0f32, |acc, &v| acc + v))
        .fold(0.0f32, |acc, v| acc + v);
    assert_eq!(a.get(&[]).unwrap().to_bits(), naive.to_bits());
}

/// 非正方 rank3（各次元長が互いに異なる）形状での max/mean 軸別期待値
/// （#25 棚卸しで特定したギャップ）: 既存の rank3 テストは shape `[2,3,4]`
/// の sum/max のみで、mean の rank3 は未検証だった。
#[test]
fn max_mean_axis_matches_hand_computed_expected_values_non_square_rank3() {
    // shape [2, 5, 3]（各次元長が互いに異なる非正方形状）。
    let data: Vec<f32> = (0..30).map(|v| ((v * 53 + 7) % 30) as f32).collect();
    let t = Tensor::<f32>::new(data, &[2, 5, 3]).unwrap();

    let out_max = max(&t, Some(1)).unwrap();
    assert_eq!(out_max.shape(), &[2, 3]);
    for i in 0..2 {
        for k in 0..3 {
            let expected = (0..5)
                .map(|j| t.get(&[i, j, k]).unwrap())
                .fold(f32::NEG_INFINITY, f32::max);
            assert_eq!(out_max.get(&[i, k]).unwrap(), expected);
        }
    }

    let out_mean = mean(&t, Some(2)).unwrap();
    assert_eq!(out_mean.shape(), &[2, 5]);
    for i in 0..2 {
        for j in 0..5 {
            let expected: f32 = (0..3).map(|k| t.get(&[i, j, k]).unwrap()).sum::<f32>() / 3.0;
            assert_eq!(out_mean.get(&[i, j]).unwrap(), expected);
        }
    }
}

/// 非 contiguous view＋`dim=None` 全縮約（`gather_elements` フォールバック
/// 経路）が contiguous 実体化後と bit 一致することを確認する（#25 棚卸しで
/// 特定したギャップ: 既存の `non_contiguous_view_matches_contiguous_bitwise`
/// は全縮約も含むが対象は `sum` のみで、`max`/`mean` は未検証だった）。
#[test]
fn non_contiguous_full_reduction_max_mean_matches_contiguous_bitwise() {
    let t = Tensor::<f32>::new((0..12).map(|v| v as f32 * 1.5 - 3.0).collect(), &[3, 4]).unwrap();
    let tt = t.transpose(0, 1).unwrap(); // shape [4, 3]、非 contiguous
    assert!(tt.as_slice().is_none());
    let c = tt.contiguous();

    let max_view = max(&tt, None).unwrap();
    let max_contig = max(&c, None).unwrap();
    assert_eq!(
        max_view.get(&[]).unwrap().to_bits(),
        max_contig.get(&[]).unwrap().to_bits()
    );

    let mean_view = mean(&tt, None).unwrap();
    let mean_contig = mean(&c, None).unwrap();
    assert_eq!(
        mean_view.get(&[]).unwrap().to_bits(),
        mean_contig.get(&[]).unwrap().to_bits()
    );
}

/// 軸範囲外・空テンソルのエラー系。
#[test]
fn error_variants() {
    let t = Tensor::<f32>::zeros(&[2, 3]).unwrap();
    assert!(matches!(
        sum(&t, Some(9)).unwrap_err(),
        ReduceError::Shape(_)
    ));

    let empty = Tensor::<f32>::zeros(&[0]).unwrap();
    assert_eq!(sum(&empty, None).unwrap().get(&[]).unwrap(), 0.0);
    assert!(matches!(
        max(&empty, None).unwrap_err(),
        ReduceError::EmptyReduction { op: "max" }
    ));
    assert!(matches!(
        mean(&empty, None).unwrap_err(),
        ReduceError::EmptyReduction { op: "mean" }
    ));
}
