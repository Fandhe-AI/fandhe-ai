//! `Mlp` と PyTorch `nn.Sequential` の層ごとの重み対応表（examples 限定。イシュー #2201・
//! 親 #2190。`Mlp` 本体は #2974 で `fandhe_ai::models::Mlp` として公開済み）。
//!
//! ## 位置づけ（重要）
//!
//! 本ファイルは `crates/facade/examples/reference_models.rs`（runnable example。対応表の
//! 目視確認）と `crates/facade/tests/example_mlp_mnist.rs`（統合テスト。
//! `#[path = "../examples/models/mlp.rs"] mod mlp;` で本ファイルを直接取り込む）の 2 箇所から
//! **単独で完結する**ファイルとして読み込まれる（`super::`／`crate::` による他ファイル参照は
//! しない）。
//!
//! 対応表は facade の公開面に含めない（`docs/reference-models-decision.md` §11）。
//! `Mlp` の構成値（`input_dim`／`hidden_dims`／`output_dim`）を読む公開 getter は
//! 承認されていないため、[`mlp_pytorch_param_map`] は構成値を引数で受け取る。
//!
//! ## 重みレイアウトの契約
//!
//! fandhe の `nn::Linear.weight` は `[in_features, out_features]`、PyTorch
//! `nn.Linear.weight` は `[out_features, in_features]` のため**転置の関係**にある
//! （`transpose: true`）。bias は両者とも `[out_features]` で転置不要。

use fandhe_ai::AutodiffError;
use fandhe_ai::models::Mlp;

/// PyTorch チェックポイントとの層ごとの対応 1 件（イシュー #2201 の受け入れ条件 AC5
/// 「層ごとの重み対応を目視で確認できる」を満たす表現）。
#[derive(Debug, Clone)]
pub struct MlpParamMap {
    /// fandhe 側キー（`compat::Sequential::named_parameters` の形式。
    /// `"{index}.weight"` / `"{index}.bias"`）。
    pub fandhe_key: String,
    /// 対応する PyTorch `state_dict` キー（`nn.Sequential` の同じ index 規約のため
    /// fandhe 側と同一の index を使う）。
    pub pytorch_key: String,
    /// fandhe 側 shape（構成値から導出。実パラメータの shape を読み返すと対応表の検証が
    /// トートロジーになるため、必ず構成値から計算する）。
    pub fandhe_shape: Vec<usize>,
    /// 対応する PyTorch 側 shape（`weight` は `[out, in]`）。
    pub pytorch_shape: Vec<usize>,
    /// `true` の場合 `weight` は fandhe → PyTorch で転置が必要。
    pub transpose: bool,
}

/// PyTorch 参照定義との層ごとの重み対応表（AC5）。`fandhe_shape` は引数の構成値
/// （`Mlp::new` に渡した `input_dim`／`hidden_dims`／`output_dim`）から計算し、実パラメータの
/// shape は読み返さない（対応表の検証をトートロジーにしないため）。そのうえで
/// `mlp.sequential().named_parameters()` と突き合わせ、キー集合・shape が完全一致することを
/// 検証する（`sequential_mut()` 経由で内部構成が差し替えられた場合や、渡した構成値が
/// 実モデルと食い違う場合に `Err` で検出するため。#2201 PR #2320）。
pub fn mlp_pytorch_param_map(
    mlp: &Mlp,
    input_dim: usize,
    hidden_dims: &[usize],
    output_dim: usize,
) -> Result<Vec<MlpParamMap>, AutodiffError> {
    let mut dims = Vec::with_capacity(hidden_dims.len() + 2);
    dims.push(input_dim);
    dims.extend(hidden_dims.iter().copied());
    dims.push(output_dim);

    let mut out = Vec::new();
    for (layer_no, w) in dims.windows(2).enumerate() {
        let (in_f, out_f) = (w[0], w[1]);
        // Sequential 上の index: 隠れ層ごとに Linear/ReLU/Dropout の 3 層を消費する
        // （`Mlp::with_seed` の構築順と一致させる）。
        let index = layer_no * 3;
        out.push(MlpParamMap {
            fandhe_key: format!("{index}.weight"),
            pytorch_key: format!("{index}.weight"),
            fandhe_shape: vec![in_f, out_f],
            pytorch_shape: vec![out_f, in_f],
            transpose: true,
        });
        out.push(MlpParamMap {
            fandhe_key: format!("{index}.bias"),
            pytorch_key: format!("{index}.bias"),
            fandhe_shape: vec![out_f],
            pytorch_shape: vec![out_f],
            transpose: false,
        });
    }

    let actual = mlp.sequential().named_parameters();
    if actual.len() != out.len() {
        return Err(AutodiffError::InvalidArgument(format!(
            "mlp_pytorch_param_map: 対応表のエントリ数（{}）が実パラメータ数\
             （{}）と一致しない（sequential_mut() 経由で内部構成が\
             差し替えられた、または構成値が実モデルと異なる可能性がある）",
            out.len(),
            actual.len()
        )));
    }
    for entry in &out {
        let found = actual
            .iter()
            .find(|(key, _)| *key == entry.fandhe_key)
            .ok_or_else(|| {
                AutodiffError::InvalidArgument(format!(
                    "mlp_pytorch_param_map: キー '{}' が実パラメータに\
                     存在しない（sequential_mut() 経由で内部構成が\
                     差し替えられた可能性がある）",
                    entry.fandhe_key
                ))
            })?;
        if found.1.shape() != entry.fandhe_shape.as_slice() {
            return Err(AutodiffError::InvalidArgument(format!(
                "mlp_pytorch_param_map: キー '{}' の shape が対応表\
                 （{:?}）と実パラメータ（{:?}）で不一致\
                 （sequential_mut() 経由で内部構成が差し替えられた\
                 可能性がある）",
                entry.fandhe_key,
                entry.fandhe_shape,
                found.1.shape()
            )));
        }
    }

    Ok(out)
}
