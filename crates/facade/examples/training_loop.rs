//! `Tape`／`Var` と `fandhe_ai::optim::Sgd` で書く、PyTorch 風の手書き
//! 学習ループの最小例（イシュー #875・#4337）。
//!
//! サイト原稿（`site/examples/training-loop.md`）に転記するコード例の
//! 一次ソース（`getting_started.rs`〈#874〉と同じ理由で二重実装を避ける。
//! `.claude/rules/code-comment-style.md`）。本 example の実行成功
//! （`cargo run -p fandhe-ai --example training_loop`）が原稿の受け入れ
//! 条件（コード例がコンパイル・動作確認済みであること）を担保する。
//! `fandhe-ai` の公開面（`fandhe_ai::*`）だけで完結するため、`fandhe-ai`
//! だけに依存する利用者のプロジェクトへそのまま写して動く。
//!
//! **optimizer**: `fandhe_ai::optim::{Sgd, SgdConfig}` を使う。PyTorch の
//! `loss.backward(); optimizer.step()` に対応するのが `tape.backward` と
//! `Sgd::step` で、`step` は更新後パラメータを返すので
//! `Sequential::apply_parameters` で書き戻す。`AdamW` に替える場合は、
//! 設定型も `SgdConfig::new(LR)` から `AdamWConfig`（例: `AdamWConfig::default()`）
//! へ差し替えて `let mut optimizer = AdamW::new(AdamWConfig::default())?;` のように設定値を渡して呼ぶ（`SgdConfig` のままでは型が合わない）。
//! `step` の呼び出し方も異なる。`Sgd::step` は `&param_refs, &grad_refs` の 2 引数だが、
//! `AdamW::step` は `&[(&Tensor<f32>, &Tensor<f32>)]`（param と grad の組）の 1 引数なので、
//! `trainable_parameters` と `trainable_grads` を同じ順序で zip して組を作って渡す。
//! ループ自体を書かずに済ませたい場合は `Sequential::compile`／
//! `fit` を使う（その example はイシュー #4338 で追加予定）。
//!
//! **決定的シード**: データ生成は `fandhe_ai::manual_seed` ＋
//! `fandhe_ai::rand`（整数演算の xorshift64* で生成し、`next_unit_f32` は上位 24 ビットを
//! `f32` へ変換して 2^24 で除算する。24 ビット整数と 2 のべき乗による
//! スケーリングは `f32` で厳密に表現できるためプラットフォーム間で bit 同一）、
//! 重み初期化は `Sequential::add_linear` の `seed` 引数で固定する
//! （`.claude/rules/coding-rust.md`「学習系回帰テストには決定的シード設定
//! ユーティリティを使う」）。`rand` は暗号用途の乱数ではなく、再現可能な
//! デモ用途に限る。
//!
//! **借用スコープ**: `Sequential::bind` が返す `SequentialVars` は
//! `&model`/`&tape` を借用するため、`Sequential::apply_parameters`
//! （`&mut model`）を呼ぶ前に必ずスコープを抜けて借用を解放する
//! （`crates/facade/tests/compat_sequential_train.rs` と同じ構成）。
//!
//! 本番経路で `unwrap()`/`expect()` を使わない方針（`.claude/rules/
//! coding-rust.md`）に合わせ、`main` は `Result` を返し `?` で伝播する。

use fandhe_ai::Tensor;
use fandhe_ai::compat::Sequential;
use fandhe_ai::optim::{Sgd, SgdConfig};

const BATCH: usize = 4;
const D_IN: usize = 8;
const D_HIDDEN: usize = 16;
const D_OUT: usize = 4;

const SEED_DATA: u64 = 0xC0FFEE;
const SEED_L1: u64 = 0x1111_1111;
const SEED_L2: u64 = 0x2222_2222;

const STEPS: usize = 100;
const LR: f32 = 0.05;

/// `x`（`[BATCH, D_IN]`）・`y`（`[BATCH, D_OUT]`）を `[0, 1)` の一様乱数で
/// 生成する（`manual_seed` で系列を固定してから `rand` を呼ぶ）。
fn gen_regression_data(
    seed: u64,
) -> Result<(Tensor<f32>, Tensor<f32>), Box<dyn std::error::Error>> {
    fandhe_ai::manual_seed(seed);
    let x = fandhe_ai::rand(&[BATCH, D_IN])?;
    let y = fandhe_ai::rand(&[BATCH, D_OUT])?;
    Ok((x, y))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut model = Sequential::new()
        .add_linear(D_IN, D_HIDDEN, SEED_L1)?
        .add_relu()
        .add_linear(D_HIDDEN, D_OUT, SEED_L2)?;

    let (x_data, y_data) = gen_regression_data(SEED_DATA)?;
    let mut optimizer = Sgd::new(SgdConfig::new(LR))?;

    let mut initial_loss = None;
    let mut final_loss = 0.0_f32;

    for step in 0..STEPS {
        // 1 ステップ分の Tape 上で forward → backward → optimizer.step までを
        // 行い、更新後テンソル列（所有 `Vec<Tensor<f32>>`）をブロック外へ
        // 持ち出す。ブロックを抜けると `bound`（`&model` を借用）と `tape` が
        // drop され、直後の `apply_parameters`（`&mut model`）と借用が競合しない。
        let updated: Vec<Tensor<f32>> = {
            let tape = fandhe_ai::tape();
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

            // PyTorch の `loss.backward()` に相当。
            let grads = tape.backward(&loss)?;
            let grad_refs = bound.trainable_grads(&grads)?;
            let param_refs = model.trainable_parameters();

            // PyTorch の `optimizer.step()` に相当（更新後パラメータを返す）。
            optimizer.step(&param_refs, &grad_refs)?
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
