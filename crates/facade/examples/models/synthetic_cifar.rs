//! 合成 CIFAR-10 相当データ生成器（イシュー #2202・親 #2190）。
//!
//! `mlp.rs`／`lenet.rs`（イシュー #2201）と同じ理由・同じ制約で、本
//! ファイルも**単独で完結する**（他ファイルへの `super::`／`crate::`
//! 参照なし）。`crates/facade/examples/main.rs`（runnable example）と
//! `crates/facade/tests/example_resnet_cifar10.rs`・
//! `crates/facade/tests/example_transformer_cifar10.rs`（`#[path]`
//! 取り込みの統合テスト）から読み込まれる。
//!
//! **実 CIFAR-10 は同梱しない。ネットワーク取得もしない**
//! （`example_models_common.rs`（イシュー #2201・合成 MNIST）と同じ方針。
//! 依存もネットワークアクセスも増やさない）。クラスごとに周波数・向き・
//! チャネルバイアスが異なる正弦波縞パターンへ、サンプルごとの巡回平行
//! 移動と一様ノイズを乗せて生成する。乱数源は呼び出し元が注入する
//! （`.claude/rules/coding-rust.md`「学習系回帰テストには決定的シード
//! 設定ユーティリティを使う」に沿い、テストは
//! `bench_harness::rng::Xorshift64Star`、`main.rs` は依存追加なしの
//! 局所 PRNG を注入する）。

use fandhe_ai::AutodiffError;

/// チャネル数（RGB）。
pub const IMG_C: usize = 3;
/// 画像の高さ。
pub const IMG_H: usize = 32;
/// 画像の幅。
pub const IMG_W: usize = 32;
/// クラス数（0..10。実 CIFAR-10 と同じ 10 分類）。
pub const NUM_CLASSES: usize = 10;

/// `next_u64()` の上位 24bit を仮数部に使い `[0, 1)` の一様分布を作る
/// （`bench_harness::rng::Xorshift64Star::next_f32` と同じ変換方式。
/// 本ファイルは `bench_harness` を import しない単独完結方針のため
/// 変換式のみ複製する）。
fn u64_to_unit(v: u64) -> f32 {
    let bits = (v >> 40) as u32; // 24bit
    bits as f32 / (1u32 << 24) as f32
}

/// クラス `class`・チャネル `c`・座標 `(h, w)` における決定的な基底輝度
/// （`[0, 1]` にクランプ済み）。クラスごとに縞の向き（`angle`）・周波数
/// （`freq`）・チャネル間の位相差（`phase`）を変えることでクラス間の
/// 分離を作る。乱数は使わない純関数（プロトタイプ自体を決定的にする
/// ため。`example_models_common.rs::class_prototype` と同じ考え方）。
fn class_pattern(class: usize, c: usize, h: usize, w: usize) -> f32 {
    let class_f = class as f32;
    let angle = class_f * std::f32::consts::PI / NUM_CLASSES as f32;
    let freq = 0.25 + 0.04 * class_f;
    let phase = c as f32 * 0.9 + class_f * 0.3;
    let proj = h as f32 * angle.cos() + w as f32 * angle.sin();
    let base = 0.5 + 0.42 * (freq * proj + phase).sin();
    let channel_bias = match (class + c) % 3 {
        0 => 0.08,
        1 => 0.0,
        _ => -0.08,
    };
    (base + channel_bias).clamp(0.0, 1.0)
}

/// `n * IMG_C * IMG_H * IMG_W`（1 サンプルあたりの平坦要素数 × サンプル
/// 数）を `checked_mul` で検証する（`synthetic_cifar10`・`to_row_tokens`
/// 共用。`n` は呼び出し元が渡す公開引数のため、素の `*` は debug では
/// panic・release では wrap-around して誤ったサイズのまま構築が進み
/// うる。Codex レビュー指摘・イシュー #2202 PR #2325。`IMG_C`／`IMG_H`／
/// `IMG_W` はモジュール内定数〈32・32・3〉のため `IMG_C*IMG_H*IMG_W` 自体
/// は現実的にオーバーフローしないが、`n` との積は検査する）。
fn checked_total_elements(n: usize) -> Result<usize, AutodiffError> {
    IMG_C
        .checked_mul(IMG_H)
        .and_then(|v| v.checked_mul(IMG_W))
        .and_then(|per_sample| n.checked_mul(per_sample))
        .ok_or_else(|| {
            AutodiffError::InvalidArgument(format!(
                "n（{n}）* IMG_C（{IMG_C}）* IMG_H（{IMG_H}）* IMG_W（{IMG_W}）が usize の範囲を超える"
            ))
        })
}

/// 合成 CIFAR-10 相当のサンプル `n` 件を生成する（`[N, 3, 32, 32]` 相当の
/// 行優先平坦データ・巡回ラベル `i % NUM_CLASSES`）。
///
/// `next_u64` は乱数源（呼び出し元が注入。本ファイル自体は PRNG を
/// 持たない）。サンプルごとに平行移動幅（`±3px`。巡回シフトで境界を
/// 折り返す）とチャネル・画素ごとの一様ノイズ（振幅 `±0.1`）を
/// この関数から引く。`n * IMG_C * IMG_H * IMG_W` が usize の範囲を超える
/// 場合は `InvalidArgument` を返す（Codex レビュー指摘・イシュー #2202
/// PR #2325）。
pub fn synthetic_cifar10(
    n: usize,
    next_u64: &mut impl FnMut() -> u64,
) -> Result<(Vec<f32>, Vec<i32>), AutodiffError> {
    let total = checked_total_elements(n)?;
    let mut data = Vec::with_capacity(total);
    let mut labels = Vec::with_capacity(n);
    for i in 0..n {
        let class = i % NUM_CLASSES;
        let dx = (u64_to_unit(next_u64()) * 7.0).floor() as i32 - 3;
        let dy = (u64_to_unit(next_u64()) * 7.0).floor() as i32 - 3;
        for c in 0..IMG_C {
            for h in 0..IMG_H {
                for w in 0..IMG_W {
                    let sh = (h as i32 + dy).rem_euclid(IMG_H as i32) as usize;
                    let sw = (w as i32 + dx).rem_euclid(IMG_W as i32) as usize;
                    let base = class_pattern(class, c, sh, sw);
                    let noise = (u64_to_unit(next_u64()) - 0.5) * 0.2;
                    data.push((base + noise).clamp(0.0, 1.0));
                }
            }
        }
        labels.push(class as i32);
    }
    Ok((data, labels))
}

/// `[N, 3, 32, 32]`（`(n, c, h, w)` 行優先）を `[N, 32, 96]`
/// （`(n, h, c, w)` 順。1 行を 1 トークン、チャネルを特徴次元へ連結）へ
/// 並べ替える。
///
/// `Var::reshape` は非 contiguous な入力を `NonContiguousReshape` で
/// 拒否するため（`crates/autodiff/src/var.rs`）、この並べ替えは
/// `Var::permute` ではなく tape に入れる前にホスト側の `Vec<f32>` を
/// 並べ替えて行う（`docs/reference-models-decision.md` #2202 節参照）。
///
/// `flat.len()` が `n * IMG_C * IMG_H * IMG_W` と一致することを実行時
/// 検証する（従来は `debug_assert_eq!` のみで、release ビルドでは
/// 検証されず、`flat` が短すぎれば index out of bounds で panic・
/// 長すぎれば余剰を黙って捨てたまま構築が進んでいた。Codex レビュー
/// 指摘・イシュー #2202 PR #2325）。
pub fn to_row_tokens(flat: &[f32], n: usize) -> Result<Vec<f32>, AutodiffError> {
    let expected_len = checked_total_elements(n)?;
    if flat.len() != expected_len {
        return Err(AutodiffError::InvalidArgument(format!(
            "to_row_tokens: flat.len()（{}）が期待値（n*IMG_C*IMG_H*IMG_W={expected_len}）と \
             一致しない",
            flat.len()
        )));
    }
    // 並べ替え（permutation）のため出力要素数も同じ expected_len。
    let mut out = vec![0.0f32; expected_len];
    for ni in 0..n {
        for c in 0..IMG_C {
            for h in 0..IMG_H {
                for w in 0..IMG_W {
                    let src = ((ni * IMG_C + c) * IMG_H + h) * IMG_W + w;
                    let dst = ((ni * IMG_H + h) * IMG_C + c) * IMG_W + w;
                    out[dst] = flat[src];
                }
            }
        }
    }
    Ok(out)
}
