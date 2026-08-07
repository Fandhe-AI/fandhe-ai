//! TASK-4.4b の受け入れ条件「シード固定で学習系テストが再現する」の実証テスト。
//!
//! `guardrail` は依存方向上 `autodiff` に依存できない（`autodiff` → `guardrail` の
//! 一方向が層構造上の前提。`.claude/rules/delegation-impl.md`）ため、外部クレート
//! 非依存の最小学習ループ（手書き勾配降下による線形回帰）で
//! `guardrail::determinism::seeded_rng` の決定性を実証する。

use guardrail::determinism::seeded_rng;

/// 決定的乱数から初期パラメータ・合成データを生成し、数十 step の勾配降下で
/// 1 次関数 `y = 3x + 2` を学習する最小ループ。
///
/// 戻り値は `(学習後の (weight, bias), 最終 loss)`。学習ロジック自体は本イシューの
/// スコープではなく、`seeded_rng` を初期化前に使うことで学習系テストが決定的に
/// 再現することを示すための道具立てに過ぎない。
fn train_linear_regression(label: &str, steps: usize) -> (f32, f32, f32) {
    let mut rng = seeded_rng(label);

    // 合成データ（真の関係: y = 3x + 2 + ノイズ）。
    let xs: Vec<f32> = (0..32).map(|_| rng.next_f32() * 4.0).collect();
    let ys: Vec<f32> = xs
        .iter()
        .map(|&x| 3.0 * x + 2.0 + rng.next_f32() * 0.01)
        .collect();

    // 初期パラメータもモデル初期化前の決定的シードから生成する
    // （TASK-4.4 の要件そのもの: 「モデル初期化前に決定的シードを設定する」）。
    let mut w = rng.next_f32();
    let mut b = rng.next_f32();

    let lr = 0.01_f32;
    let n = xs.len() as f32;
    let mut loss = f32::MAX;

    for _ in 0..steps {
        let mut grad_w = 0.0_f32;
        let mut grad_b = 0.0_f32;
        loss = 0.0;
        for (&x, &y) in xs.iter().zip(ys.iter()) {
            let pred = w * x + b;
            let err = pred - y;
            grad_w += 2.0 * err * x / n;
            grad_b += 2.0 * err / n;
            loss += err * err / n;
        }
        w -= lr * grad_w;
        b -= lr * grad_b;
    }

    (w, b, loss)
}

#[test]
fn same_seed_reproduces_training_run_bit_for_bit() {
    // 受け入れ条件本体: 同一シード（同一 label）で 2 回学習し、
    // パラメータ軌跡・最終 loss がビット単位で一致することを確認する。
    let (w1, b1, loss1) = train_linear_regression("acceptance.same_seed", 50);
    let (w2, b2, loss2) = train_linear_regression("acceptance.same_seed", 50);

    assert_eq!(w1.to_bits(), w2.to_bits(), "weight がビット一致しない");
    assert_eq!(b1.to_bits(), b2.to_bits(), "bias がビット一致しない");
    assert_eq!(loss1.to_bits(), loss2.to_bits(), "loss がビット一致しない");
}

#[test]
fn different_seed_diverges_training_run() {
    // シードが実際に効いていることの確認（label 違い＝導出シード違い）。
    let (w1, b1, _) = train_linear_regression("acceptance.label_a", 50);
    let (w2, b2, _) = train_linear_regression("acceptance.label_b", 50);

    assert!(
        w1.to_bits() != w2.to_bits() || b1.to_bits() != b2.to_bits(),
        "異なる label でパラメータが一致してしまった（シードが効いていない）"
    );
}

#[test]
fn training_run_converges_toward_known_relationship() {
    // 決定的シードで再現するだけでなく、実際に学習が機能していることの
    // 健全性チェック（真の関係 y = 3x + 2 に収束する）。
    let (w, b, loss) = train_linear_regression("acceptance.convergence", 200);

    assert!((w - 3.0).abs() < 0.5, "weight が真値に収束していない: {w}");
    assert!((b - 2.0).abs() < 0.5, "bias が真値に収束していない: {b}");
    assert!(loss < 1.0, "loss が十分小さくなっていない: {loss}");
}
