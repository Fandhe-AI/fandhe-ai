//! v2 自作コア（`autodiff::nn::Linear`）上の小型 MLP をライブラリ化したもの。
//! XOR 拡張タスク（2 入力 -> 8 -> 8 -> 1 出力）の回帰として学習・推論する。
//!
//! `forward` はベンチマーク（`benches/forward_bench.rs`）の対象であり、
//! TASK-4.2a 検証題材（D3・性能回帰、G2・アーキテクチャ変更）の注入対象でもある。

use autodiff::AutodiffError;
use autodiff::Tape;
use autodiff::Var;
use autodiff::nn::{Linear, LinearVars};

use crate::activations;

/// `Linear::new` の重み初期化シード導出に使う（`fc1`/`fc2`/`fc3` を
/// 独立系列にするための塩。`autodiff::nn::linear.rs::derive_seed` の
/// weight/bias 分離と同じ考え方をレイヤー間にも適用する）。
const FC1_SEED_SALT: u64 = 0x1;
const FC2_SEED_SALT: u64 = 0x2;
const FC3_SEED_SALT: u64 = 0x3;

pub struct Mlp {
    fc1: Linear,
    fc2: Linear,
    fc3: Linear,
}

impl Mlp {
    /// 隠れ層次元 8 の 3 層 MLP（`2 -> 8 -> 8 -> 1`）を決定的シードで
    /// 構築する。`seed` から `FC1_SEED_SALT`〜`FC3_SEED_SALT` で 3 系統の
    /// 導出シードを作り、各層の初期化に独立して使う。
    pub fn new(seed: u64) -> Result<Mlp, AutodiffError> {
        Ok(Mlp {
            fc1: Linear::new(2, 8, true, seed ^ FC1_SEED_SALT)?,
            fc2: Linear::new(8, 8, true, seed ^ FC2_SEED_SALT)?,
            fc3: Linear::new(8, 1, true, seed ^ FC3_SEED_SALT)?,
        })
    }

    pub fn fc1(&self) -> &Linear {
        &self.fc1
    }
    pub fn fc2(&self) -> &Linear {
        &self.fc2
    }
    pub fn fc3(&self) -> &Linear {
        &self.fc3
    }

    /// 各層のパラメータを差し替えた新しい `Mlp` を構築する（optimizer
    /// が返す更新後パラメータで置き換える運用。`autodiff::nn::linear.rs`
    /// の `Linear::from_parameters` を都度呼び直す設計と同じ理由:
    /// `LinearVars::forward` は勾配取得 API を持たず、パラメータ更新は
    /// 呼び出し元がテンソルを直接差し替える責務のため）。
    pub fn with_layers(fc1: Linear, fc2: Linear, fc3: Linear) -> Mlp {
        Mlp { fc1, fc2, fc3 }
    }

    /// 順伝播。fc1 -> ReLU -> fc2 -> ReLU -> fc3。
    ///
    /// 呼び出し元が「入力・パラメータ・出力すべてが同一 `tape` 上の
    /// ノード」という不変条件を保てるよう、内部で `bind()` した
    /// `LinearVars` をそのまま返す（呼び出し元がこれとは別に
    /// `bind()` し直すと、`backward()`／`Gradients::get()` の対象が
    /// forward の計算グラフに接続されないノードになってしまう）。
    /// `train.rs::train` はこの戻り値の `LinearVars` を勾配取得に使う。
    pub fn forward<'t>(
        &self,
        tape: &'t Tape,
        x: &Var<'t>,
    ) -> Result<(Var<'t>, LinearVars<'t>, LinearVars<'t>, LinearVars<'t>), AutodiffError> {
        let l1v = self.fc1.bind(tape);
        let l2v = self.fc2.bind(tape);
        let l3v = self.fc3.bind(tape);

        let h1 = l1v.forward(x)?;
        let a1 = activations::relu(&h1);
        let h2 = l2v.forward(&a1)?;
        let a2 = activations::relu(&h2);
        let out = l3v.forward(&a2)?;

        Ok((out, l1v, l2v, l3v))
    }
}
