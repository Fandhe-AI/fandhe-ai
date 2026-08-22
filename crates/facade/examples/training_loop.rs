//! `compat::Sequential` を使った学習ループの最小例（イシュー #875）。
//!
//! サイト原稿（`site/examples/training-loop.md`）に転記するコード例の
//! 一次ソース（`getting_started.rs`〈#874〉と同じ理由で二重実装を避ける。
//! `.claude/rules/code-comment-style.md`）。本 example の実行成功
//! （`cargo run -p facade --example training_loop`）が原稿の受け入れ
//! 条件（コード例がコンパイル・動作確認済みであること）を担保する。
//!
//! **手動 SGD にする理由**: `autodiff::optim::{Sgd, AdamW}` は
//! `facade` の公開面（`docs/compat-api-scope.md` §0）に含まれない
//! 内部 API のため、本 example は `facade::compat::Sequential` と
//! `facade::{Tape, Var, Tensor}` の公開面だけで完結する最小学習ループ
//! として `param - lr * grad` を自前で計算する。optimizer 実装自体の
//! 例は本 example のスコープ外（イシュー #875 実装計画 §7）。
//!
//! **決定的シード**: 重み初期化（`Sequential::add_linear` の `seed`
//! 引数）・データ生成（`bench_harness::rng::Xorshift64Star`）の双方を
//! 固定シードで駆動する（`.claude/rules/coding-rust.md`「学習系回帰
//! テストには決定的シード設定ユーティリティを使う」。ベンチ用途に
//! 限らず本 example のような再現可能なデモにも同じ方針を適用する）。
//!
//! **借用スコープ**: `Sequential::bind` が返す `SequentialVars` は
//! `&model`/`&tape` を借用するため、`Sequential::apply_parameters`
//! （`&mut model`）を呼ぶ前に必ずスコープを抜けて借用を解放する
//! （`crates/facade/tests/compat_sequential_train.rs` と同じ構成）。
//!
//! 本番経路で `unwrap()`/`expect()` を使わない方針（`.claude/rules/
//! coding-rust.md`）に合わせ、`main` は `Result` を返し `?` で伝播する。

use bench_harness::rng::Xorshift64Star;
use facade::Tensor;
use facade::compat::Sequential;

const BATCH: usize = 4;
const D_IN: usize = 8;
const D_HIDDEN: usize = 16;
const D_OUT: usize = 4;

const SEED_DATA: u64 = 0xC0FFEE;
const SEED_L1: u64 = 0x1111_1111;
const SEED_L2: u64 = 0x2222_2222;

const STEPS: usize = 100;
const LR: f32 = 0.05;

/// `x`（`[BATCH, D_IN]`）・`y`（`[BATCH, D_OUT]`）を Xorshift64Star から
/// 生成する（`crates/facade/tests/compat_sequential_train.rs::gen_regression_data`
/// と同一生成順）。
fn gen_regression_data(
    seed: u64,
) -> Result<(Tensor<f32>, Tensor<f32>), Box<dyn std::error::Error>> {
    let mut rng = Xorshift64Star::new(seed);
    let x = rng.fill_vec(BATCH * D_IN);
    let y = rng.fill_vec(BATCH * D_OUT);
    Ok((
        Tensor::new(x, &[BATCH, D_IN])?,
        Tensor::new(y, &[BATCH, D_OUT])?,
    ))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut model = Sequential::new()
        .add_linear(D_IN, D_HIDDEN, SEED_L1)?
        .add_relu()
        .add_linear(D_HIDDEN, D_OUT, SEED_L2)?;

    let (x_data, y_data) = gen_regression_data(SEED_DATA)?;

    let mut initial_loss = None;
    let mut final_loss = 0.0_f32;

    for step in 0..STEPS {
        // 1 ステップ分の Tape 上で forward → backward → 手動 SGD 更新値の
        // 計算までを行い、更新後テンソル列（所有 `Vec<Tensor<f32>>`）を
        // ブロック外へ持ち出す。ブロックを抜けると `bound`（`&model` を
        // 借用）と `tape` が drop され、直後の `apply_parameters`
        // （`&mut model`）呼び出しと借用が競合しない。
        let updated: Vec<Tensor<f32>> = {
            let tape = facade::tape();
            let bound = model.bind(&tape);
            let x = tape.var(&x_data);
            let y = tape.var(&y_data);

            let pred = bound.forward(&tape, &x)?;
            let loss = pred.mse_loss(&y)?;
            let loss_value = loss
                .to_tensor()
                .get(&[])
                .ok_or("loss はスカラー shape [] のはず")?;
            if step == 0 {
                initial_loss = Some(loss_value);
            }
            final_loss = loss_value;

            let grads = tape.backward(&loss)?;
            let grad_refs = bound.trainable_grads(&grads)?;
            let param_refs = model.trainable_parameters();

            let mut next_params = Vec::with_capacity(param_refs.len());
            for (param, grad) in param_refs.iter().zip(grad_refs.iter()) {
                // `contiguous()` の戻り値は同一文の間だけ生存すればよい
                // （`crates/facade/tests/compat_sequential_train.rs::dense_vec`
                // と同じ一時変数寿命の使い方）ため 1 文で `to_vec()` まで行う。
                let param_data = param.contiguous().as_slice().ok_or(
                    "trainable_parameters() の要素は contiguous() 直後は必ず as_slice() が Some",
                )?.to_vec();
                let grad_data = grad
                    .contiguous()
                    .as_slice()
                    .ok_or("trainable_grads() の要素は contiguous() 直後は必ず as_slice() が Some")?
                    .to_vec();
                let sgd_data: Vec<f32> = param_data
                    .iter()
                    .zip(grad_data.iter())
                    .map(|(p, g)| p - LR * g)
                    .collect();
                next_params.push(Tensor::from_slice(&sgd_data, param.shape())?);
            }
            next_params
        };
        model.apply_parameters(updated)?;
    }

    let initial_loss = initial_loss.ok_or("STEPS > 0 のため初期 loss は必ず記録される")?;
    println!("initial loss: {initial_loss}");
    println!("final loss: {final_loss}");
    let loss_decreased = final_loss < initial_loss;
    println!("loss decreased: {loss_decreased}");

    // 学習退行（勾配計算・パラメータ更新の不具合で loss が非有限化
    // する、または減少しなくなる）を表示するだけで見逃さないよう、
    // 満たさない場合は Err を返して `cargo run` を失敗終了させる
    // （イシュー #875 レビュー指摘）。
    if !final_loss.is_finite() {
        return Err(format!("final loss が有限値ではない: {final_loss}").into());
    }
    if !loss_decreased {
        return Err(format!(
            "loss が減少しなかった（退行の可能性）: initial={initial_loss}, final={final_loss}"
        )
        .into());
    }

    Ok(())
}
