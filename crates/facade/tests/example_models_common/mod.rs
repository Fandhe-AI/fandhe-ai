//! 合成 MNIST 相当データ生成器（イシュー #2201）。
//!
//! `crates/facade/tests/example_mlp_mnist.rs`・
//! `crates/facade/tests/example_lenet_mnist.rs` の両方が
//! `mod example_models_common;` で取り込む共通ヘルパー。**実 MNIST では
//! ない**（リポジトリにデータセットを同梱せずネットワーク取得も行わない
//! ため。`docs/reference-models-decision.md` 参照）。決定的シード駆動の
//! `bench_harness::rng::Xorshift64Star`（`.claude/rules/coding-rust.md`
//! 「学習系回帰テストには決定的シード設定ユーティリティを使う」）で、
//! クラスごとに固定した「プロトタイプ」画素パターンへ小さな一様ノイズを
//! 乗せたサンプルを生成する。
//!
//! 返り値は行優先の平坦な `Vec<f32>`（`[N, 28*28]` 相当の連続レイアウト）
//! で固定する。`[N, 784]`（MLP 入力）と `[N, 1, 28, 28]`（LeNet 入力）は
//! 同じメモリレイアウトの reshape に過ぎないため、呼び出し元が
//! `Tensor::new(data, &shape)` で望む shape を選べる（本モジュールが
//! shape 別に別関数を持つと、一方のテストバイナリでしか使われない関数が
//! `dead_code` になるため、単一関数に統一する）。

use bench_harness::rng::Xorshift64Star;

/// 1 サンプルあたりの画素数（28×28）。
pub const IMAGE_LEN: usize = 28 * 28;

/// クラス数（0..10）。
pub const NUM_CLASSES: usize = 10;

/// クラス `class` の固定プロトタイプ画素パターンを決定的に生成する。
/// クラスごとに異なる画素位置を「点灯」させることで、クラス間の分離を
/// 学習可能にする（実 MNIST の代替としての最低要件）。
fn class_prototype(class: usize) -> [f32; IMAGE_LEN] {
    let mut proto = [0.2f32; IMAGE_LEN];
    // クラス id をシードに使うことで、プロトタイプ自体もサンプル生成と
    // 同じ PRNG（xorshift64*）で決定的に導出する。
    let mut rng = Xorshift64Star::new(0x9000 + class as u64);
    for _ in 0..(IMAGE_LEN / 6) {
        let idx = (rng.next_u64() as usize) % IMAGE_LEN;
        proto[idx] = 0.6;
    }
    proto
}

/// 合成 MNIST 相当のサンプル `n` 件を生成する。
///
/// 戻り値は `(平坦画素データ〈長さ n * 784〉, クラス添字ラベル〈長さ
/// n〉)`。ラベルはクラス `i % NUM_CLASSES` の巡回割り当てで、
/// クラス間のサンプル数がほぼ均等になるようにする。
pub fn synthetic_mnist_flat(n: usize, seed: u64) -> (Vec<f32>, Vec<i32>) {
    let mut rng = Xorshift64Star::new(seed);
    let mut data = Vec::with_capacity(n * IMAGE_LEN);
    let mut labels = Vec::with_capacity(n);
    for i in 0..n {
        let class = i % NUM_CLASSES;
        let proto = class_prototype(class);
        for &p in proto.iter() {
            // ノイズ幅は [-0.25, 0.25)（`next_f32` の [-1, 1) を 0.25 倍）。
            // プロトタイプの点灯／非点灯の差（0.2 vs 0.6）と同程度の振幅の
            // ノイズを乗せることで、1 epoch で完全分離しきらない程度の
            // 難度に保つ（AC4 の判定式が実質的な学習進捗を確認できるように
            // するため）。
            let noise = rng.next_f32() * 0.25;
            data.push((p + noise).clamp(0.0, 1.0));
        }
        labels.push(class as i32);
    }
    (data, labels)
}
