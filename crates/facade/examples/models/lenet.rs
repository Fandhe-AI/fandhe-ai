//! `LeNet` と PyTorch の層ごとの重み対応表（examples 限定。イシュー #2201・親 #2190。
//! `LeNet` 本体は #2974 で `fandhe_ai::models::LeNet` として公開済み）。
//!
//! ## 位置づけ
//!
//! `crates/facade/examples/models/mlp.rs` と同じ理由・同じ制約で、本ファイルも**単独で完結する**
//! （他ファイルへの `super::`／`crate::` 参照なし）。`reference_models.rs`（runnable example）と
//! `crates/facade/tests/example_lenet_mnist.rs`（`#[path]` 取り込みの統合テスト）から読み込まれる。
//! 対応表は facade の公開面に含めない（`docs/reference-models-decision.md` §11）。
//!
//! ## PyTorch 参照定義（Conv2d 2 層・Dense 2 層版。古典 LeNet-5 の fc 3 層版とは異なる）
//!
//! ```text
//! conv1 = Conv2d(1, 6, 5) -> relu -> max_pool2d(2)
//! conv2 = Conv2d(6, 16, 5) -> relu -> max_pool2d(2)
//! flatten(1)                               # 16*4*4 = 256
//! fc1 = Linear(256, 120) -> relu
//! fc2 = Linear(120, num_classes)
//! ```
//!
//! ## 重みレイアウトの契約
//!
//! `nn::Conv2d.weight` は PyTorch と同一 shape（転置不要）。`nn::Linear.weight` は
//! `[in_features, out_features]` で PyTorch の `[out, in]` とは**転置の関係**。

use fandhe_ai::AutodiffError;
use fandhe_ai::models::LeNet;

/// PyTorch チェックポイントとの層ごとの対応 1 件（`mlp.rs` の `MlpParamMap` と同型。
/// 単独完結の方針のため型自体は個別に定義する）。
#[derive(Debug, Clone)]
pub struct LeNetParamMap {
    pub fandhe_key: String,
    pub pytorch_key: String,
    pub fandhe_shape: Vec<usize>,
    pub pytorch_shape: Vec<usize>,
    /// `true` の場合 fandhe → PyTorch で `weight` の転置が必要（`Linear` 系の 2 層のみ
    /// `true`。`Conv2d` 系は shape が同一のため `false`）。
    pub transpose: bool,
}

/// PyTorch 参照定義との層ごとの重み対応表（AC5）。`fandhe_shape` は本モジュール doc 固定の
/// PyTorch 参照定義と `lenet.num_classes()` から直接書き下ろす（`named_parameters()` からの
/// 逆算はしない非トートロジー方針）。そのうえで `named_parameters()` と突き合わせ、
/// キー集合・shape が完全一致することを検証する（`sequential_mut()` 経由で内部構成が
/// 差し替えられた場合に `Err` で検出するため。#2201 PR #2320）。
pub fn lenet_pytorch_param_map(lenet: &LeNet) -> Result<Vec<LeNetParamMap>, AutodiffError> {
    let c = lenet.num_classes();
    let entry =
        |fk: &str, pk: &str, fs: Vec<usize>, ps: Vec<usize>, transpose: bool| LeNetParamMap {
            fandhe_key: fk.to_string(),
            pytorch_key: pk.to_string(),
            fandhe_shape: fs,
            pytorch_shape: ps,
            transpose,
        };
    let out = vec![
        entry(
            "0.weight",
            "conv1.weight",
            vec![6, 1, 5, 5],
            vec![6, 1, 5, 5],
            false,
        ),
        entry("0.bias", "conv1.bias", vec![6], vec![6], false),
        entry(
            "3.weight",
            "conv2.weight",
            vec![16, 6, 5, 5],
            vec![16, 6, 5, 5],
            false,
        ),
        entry("3.bias", "conv2.bias", vec![16], vec![16], false),
        entry(
            "7.weight",
            "fc1.weight",
            vec![16 * 4 * 4, 120],
            vec![120, 16 * 4 * 4],
            true,
        ),
        entry("7.bias", "fc1.bias", vec![120], vec![120], false),
        entry("9.weight", "fc2.weight", vec![120, c], vec![c, 120], true),
        entry("9.bias", "fc2.bias", vec![c], vec![c], false),
    ];

    let actual = lenet.sequential().named_parameters();
    if actual.len() != out.len() {
        return Err(AutodiffError::InvalidArgument(format!(
            "lenet_pytorch_param_map: 対応表のエントリ数（{}）が実\
             パラメータ数（{}）と一致しない（sequential_mut() 経由で\
             内部構成が差し替えられた可能性がある）",
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
                    "lenet_pytorch_param_map: キー '{}' が実パラメータに\
                     存在しない（sequential_mut() 経由で内部構成が\
                     差し替えられた可能性がある）",
                    entry.fandhe_key
                ))
            })?;
        if found.1.shape() != entry.fandhe_shape.as_slice() {
            return Err(AutodiffError::InvalidArgument(format!(
                "lenet_pytorch_param_map: キー '{}' の shape が対応表\
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
