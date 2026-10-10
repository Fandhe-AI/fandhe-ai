//! examples 限定の共通 trait `ReferenceModule`／`Trainable`（イシュー
//! #2202・親 #2190。AC3「各層が `nn::Module` trait を実装すること」の
//! 代替実装）。
//!
//! `facade` の `nn::Module` は #2338 で公開済み（`docs/facade-nn-module-
//! exposure-decision.md` §22）。ただし `forward` が `TapeRef<'t>` を取り、
//! 部品の `compat::Sequential::forward(&Tape, ..)` を呼べないため、現行の公開面では
//! 本 trait を facade `nn::Module` へ委譲で移行できない
//! （`docs/reference-models-decision.md` §10.8 (d)）。examples から
//! 内部の `fandhe_ai_autodiff::nn::Module` を impl することもしない
//! （公開パス限定の方針）。本ファイルはその代わりに、PyTorch
//! `nn.Module` に似せた examples 限定の利用者コードとして
//! `ReferenceModule` trait を定義し、`resnet.rs`／`transformer.rs` が
//! 公開型 `fandhe_ai::models::{ResNet, TransformerClassifier}`（#2975 で公開）へ
//! examples 側で実装する。
//!
//! `resnet.rs`・`transformer.rs`・`crates/facade/examples/main.rs`・
//! `crates/facade/tests/example_resnet_cifar10.rs`・
//! `crates/facade/tests/example_transformer_cifar10.rs` の 5 箇所から
//! `#[path]` で直接取り込まれる単独完結ファイル（`mlp.rs`／`lenet.rs`
//! と同じ方針）。取り込み側は `use super::reference_module::...;` で
//! 参照する（`docs/reference-models-decision.md` #2202 節「取り込み
//! 方」参照）。

use fandhe_ai::optim::Adam;
use fandhe_ai::{AutodiffError, Tape, Tensor, Var};

/// PyTorch `nn.Module` に似せた最小限の trait（AC3 の代替）。
///
/// `forward`（推論用。`compat::Sequential::forward` への委譲チェーン）・
/// `named_parameters`（階層名付きパラメータ一覧）・`set_training`
/// （BatchNorm の train/eval 切替の伝播）・`is_training`（現在の
/// training モード照会。`set_training` と対）の 4 メソッドに限定する。
/// 学習ステップ（backward・optimizer 適用）は複合モデルの構造ごとに
/// 演算列が異なる（`ResNetBlock` 単体には自然な loss がない）ため、
/// 本 trait には含めず [`Trainable`]（モデル本体限定）に分離する。
pub trait ReferenceModule {
    /// 外部 `Tape` 上で推論用 forward を計算する。
    fn forward<'t>(&self, tape: &'t Tape, x: &Var<'t>) -> Result<Var<'t>, AutodiffError>;

    /// PyTorch 風の階層名付きのパラメータ一覧を返す。命名規則の
    /// 対応関係は実装ごとに異なる: 例えば `resnet.rs::ResNet` は
    /// PyTorch の `layer1`/`layer2`/`layer3`（ステージ別に 0 起点で
    /// 再カウント）ではなく、全 block を通した単一の 0 起点連番
    /// `i` で `"layer.{i}.main.0.weight"` のように命名する（stage
    /// 番号は名前に現れない。`ResNet::named_parameters` 参照）。
    fn named_parameters(&self) -> Vec<(String, &Tensor<f32>)>;

    /// 内部の全 `compat::Sequential` 部品へ train/eval モードを伝える
    /// （`BatchNorm` の running stats 更新可否を切り替える。
    /// `compat::Sequential::train`／`eval` への委譲）。
    fn set_training(&mut self, training: bool);

    /// 現在の training モードを返す（`set_training` が全部品へ一様に
    /// 伝播する契約〈`ResNetBlock`／`ResNet`／`Transformer` の各
    /// `set_training` 実装参照〉に基づき、いずれか 1 部品
    /// （`ResNetBlock::main`／`ResNet::stem`／`Transformer::embed`）の
    /// `compat::Sequential::training()` を読むだけで代表値になる。
    /// `set_training` 以外に内部の `training` フラグを直接変更する
    /// 経路が無いことが前提。[`predict_in_eval`]／[`accuracy`] が
    /// 一時的な eval 切替の前後でモードを保存・復元するために使う
    /// （Codex レビュー指摘・イシュー #2202 PR #2325）。
    fn is_training(&self) -> bool;
}

/// [`ReferenceModule`] に学習ステップを追加した trait。モデル本体
/// （`ResNet`・`Transformer`）のみが実装し、部品（`ResNetBlock`）は
/// 実装しない。
pub trait Trainable: ReferenceModule {
    /// 1 学習ステップ（tape 構築 → forward → cross entropy → backward →
    /// 勾配抽出 → optimizer 適用 → パラメータ書き戻し）を実行し、
    /// ホスト側 loss 値を返す。
    fn train_step(
        &mut self,
        x: &Tensor<f32>,
        y: &Tensor<i32>,
        opt: &mut Adam,
    ) -> Result<f32, AutodiffError>;
}

/// mean cross-entropy loss（`-mean(gather(log_softmax(logits), y)))`）。
///
/// `facade` は `Reduction`（`Var::cross_entropy_loss` の引数）を
/// 再エクスポートしていないため（`docs/compat-api-scope.md`）、
/// 公開パスだけで呼べる形へ書き下ろす。正解クラスの log-probability
/// のみを [`Var::gather`] で選択する（one-hot との要素積で非正解
/// クラスを消す方式は、`log_softmax` が非正解クラスに返す `-inf` と
/// one-hot の `0` の積が `0 * -inf = NaN` になり、正解クラスの loss
/// が有限でも合計が NaN 汚染されうるため採らない。イシュー #2202
/// PR #2325 レビュー指摘）。`weight_var` は `1/N` の `[N, 1]`
/// 定数（`tape.var` で tape に載せるだけの非学習対象。
/// `Trainable::train_step` はモデル内部パラメータだけを
/// `trainable_grads` で抽出するため、この定数への勾配は無視される）
/// とすることで、スカラー乗算 op を使わずに mean 相当を実現する
/// （`docs/reference-models-decision.md` #2202 節参照）。
pub fn cross_entropy_mean<'t>(
    tape: &'t Tape,
    logits: &Var<'t>,
    targets: &Tensor<i32>,
    num_classes: usize,
) -> Result<Var<'t>, AutodiffError> {
    if num_classes == 0 {
        return Err(AutodiffError::InvalidArgument(
            "cross_entropy_mean: num_classes は 0 より大きい必要がある".to_string(),
        ));
    }
    let shape = targets.shape();
    if shape.len() != 1 {
        return Err(AutodiffError::InvalidArgument(format!(
            "cross_entropy_mean: targets の rank は 1 である必要がある（実際: {shape:?}）"
        )));
    }
    let n = shape[0];
    if n == 0 {
        return Err(AutodiffError::InvalidArgument(
            "cross_entropy_mean: targets は空であってはならない".to_string(),
        ));
    }

    let mut target_idx = vec![0i32; n];
    for (i, slot) in target_idx.iter_mut().enumerate() {
        let t = targets.get(&[i]).ok_or_else(|| {
            AutodiffError::InvalidArgument(format!(
                "cross_entropy_mean: targets[{i}] の読み出しに失敗した"
            ))
        })?;
        if t < 0 || (t as usize) >= num_classes {
            return Err(AutodiffError::InvalidArgument(format!(
                "cross_entropy_mean: targets[{i}]={t} が [0, {num_classes}) の範囲外"
            )));
        }
        *slot = t;
    }
    let target_idx_tensor = Tensor::new(target_idx, &[n, 1]).map_err(|e| {
        AutodiffError::InvalidArgument(format!(
            "cross_entropy_mean: target_idx テンソル構築に失敗: {e}"
        ))
    })?;

    let weight = vec![1.0f32 / n as f32; n];
    let weight_tensor = Tensor::new(weight, &[n, 1]).map_err(|e| {
        AutodiffError::InvalidArgument(format!(
            "cross_entropy_mean: weight テンソル構築に失敗: {e}"
        ))
    })?;
    let weight_var = tape.var(&weight_tensor);

    // logits の shape が [n, num_classes] であることを事前検証する
    // （検証しないと、列数の異なる logits でも in-range な target_idx
    // なら gather がそのまま通り、誤った列から loss を計算した結果が
    // 静かに成立してしまう。Codex レビュー指摘・イシュー #2202
    // PR #2325）。
    let logits_shape = logits.to_tensor().shape().to_vec();
    if logits_shape != [n, num_classes] {
        return Err(AutodiffError::InvalidArgument(format!(
            "cross_entropy_mean: logits の shape は [{n}, {num_classes}] である必要がある \
             （実際: {logits_shape:?}）"
        )));
    }

    let log_probs = logits.log_softmax(1)?;
    // 正解クラスの log-probability のみを選択（non-target の -inf を
    // 経由しないため、non-target の乗算由来の NaN 汚染が起きない）。
    let selected = log_probs.gather(1, &target_idx_tensor)?;
    let weighted = selected.mul(&weight_var)?;
    let summed = weighted.sum(None)?;
    summed.neg()
}

/// スカラー shape `[]` の `Tensor<f32>` から値を取り出す
/// （`Var::mse_loss` 系の既存テストと同じ `.get(&[])` パターン）。
pub fn scalar_of(t: &Tensor<f32>) -> Result<f32, AutodiffError> {
    t.get(&[]).ok_or_else(|| {
        AutodiffError::InvalidArgument(
            "scalar_of: loss テンソルの shape が [] ではない".to_string(),
        )
    })
}

/// eval モードで `x`／`y` の標本平均 cross-entropy loss を計算する（held-out loss の
/// 報告用）。バッチごとの mean loss にバッチ長を掛けて標本合計へ戻し、最後に標本数で
/// 割る。呼び出し前の training モードを保存し、成功・失敗いずれの経路でも復元する
/// （`accuracy` と同じ方針）。公開面の `train_step` は損失ヘルパーを持たないため、本関数が
/// examples 側の `cross_entropy_mean`／`scalar_of` の消費者になる（#2975）。
pub fn heldout_loss<M: ReferenceModule>(
    model: &mut M,
    x: &Tensor<f32>,
    y: &Tensor<i32>,
    batch_size: usize,
    num_classes: usize,
) -> Result<f32, AutodiffError> {
    if batch_size == 0 {
        return Err(AutodiffError::InvalidArgument(
            "heldout_loss: batch_size は 0 より大きい必要がある".to_string(),
        ));
    }
    let x_shape = x.shape();
    if x_shape.is_empty() || x_shape[0] == 0 {
        return Err(AutodiffError::InvalidArgument(
            "heldout_loss: x は rank 1 以上かつ空でない必要がある".to_string(),
        ));
    }
    let n = x_shape[0];
    if y.shape() != [n] {
        return Err(AutodiffError::InvalidArgument(format!(
            "heldout_loss: x の先頭軸長 {n} と y の shape {:?} が一致しない",
            y.shape()
        )));
    }
    let original_training = model.is_training();
    model.set_training(false);
    let result = heldout_loss_in_eval(model, x, y, n, batch_size, num_classes);
    model.set_training(original_training);
    result
}

/// [`heldout_loss`] の評価ループ本体（eval 設定済み・入力検証済みが前提）。
fn heldout_loss_in_eval<M: ReferenceModule>(
    model: &M,
    x: &Tensor<f32>,
    y: &Tensor<i32>,
    n: usize,
    batch_size: usize,
    num_classes: usize,
) -> Result<f32, AutodiffError> {
    let mut total = 0.0f32;
    let mut start = 0;
    while start < n {
        let len = batch_size.min(n - start);
        let xb = sub_tensor_f32(x, start, len)?;
        let yb = sub_tensor_i32(y, start, len)?;
        let tape = fandhe_ai::tape();
        let xv = tape.var(&xb);
        let logits = model.forward(&tape, &xv)?;
        let loss = cross_entropy_mean(&tape, &logits, &yb, num_classes)?;
        total += scalar_of(&loss.to_tensor())? * len as f32;
        start += len;
    }
    Ok(total / n as f32)
}

/// eval モードで `x` を推論し `Tensor<f32>` を返す（`ResNet::predict`／
/// `Transformer::predict` 相当の shape 確認・疎通確認向け）。呼び出し
/// 前の training モードを保存し、eval で `forward` した後、成功・失敗
/// いずれの経路でも呼び出し前のモードへ復元する。
///
/// `ResNet::predict`／`Transformer::predict`（`&self` を取り、現在の
/// モードをそのまま使う。`compat::Sequential::predict` と同じ
/// 「モード切り替えなし」契約）を、モデル構築直後（`training` の既定値
/// `true`）に shape 確認目的で呼ぶと、`BatchNorm` が train モードの
/// forward を実行し running stats を汚染してしまう。この関数はその
/// 呼び出しパターンを置き換える、呼び出し側の規律に頼らない安全な
/// 代替（Codex レビュー指摘・イシュー #2202 PR #2325）。
pub fn predict_in_eval<M: ReferenceModule>(
    model: &mut M,
    x: &Tensor<f32>,
) -> Result<Tensor<f32>, AutodiffError> {
    let original_training = model.is_training();
    model.set_training(false);
    let tape = fandhe_ai::tape();
    let xv = tape.var(x);
    let result = model.forward(&tape, &xv).map(|y| y.to_tensor());
    model.set_training(original_training);
    result
}

/// `x`（`[N, ...]`）の先頭軸から `[start, start+len)` 行を切り出した
/// `Tensor<f32>` を作る（ミニバッチ抽出用。`Var`/`Tensor` に行選択 API
/// が無いため、`contiguous().as_slice()` で取り出した生データを
/// 直接スライスして作り直す）。
pub fn sub_tensor_f32(
    x: &Tensor<f32>,
    start: usize,
    len: usize,
) -> Result<Tensor<f32>, AutodiffError> {
    let full_shape = x.shape();
    if full_shape.is_empty() {
        return Err(AutodiffError::InvalidArgument(
            "sub_tensor_f32: x の rank は 1 以上である必要がある（スカラーは不可）".to_string(),
        ));
    }
    let n = full_shape[0];
    // `start + len` を `checked_add` で検査する。素の `+` は
    // `start`・`len` が両方巨大な値のとき usize をオーバーフローし、
    // debug では panic・release ではラップアラウンドして本来の境界
    // 検査（`> n`）を素通りしてしまう（Codex レビュー指摘・イシュー
    // #2202 PR #2325。`sub_tensor_f32`／`sub_tensor_i32` は直接呼び出し
    // 可能な `pub fn` のため、呼び出し元が `fit_epochs`／`accuracy` の
    // 内部ループに限らない前提で検証する）。
    let end_idx = start.checked_add(len).ok_or_else(|| {
        AutodiffError::InvalidArgument(format!(
            "sub_tensor_f32: start（{start}）と len（{len}）の和が usize の範囲を超える"
        ))
    })?;
    if end_idx > n {
        return Err(AutodiffError::InvalidArgument(format!(
            "sub_tensor_f32: [{start}, {end_idx}) が先頭軸の長さ {n} を超える"
        )));
    }
    let item_len: usize = full_shape[1..].iter().product();
    let contiguous = x.contiguous();
    let flat = contiguous.as_slice().ok_or_else(|| {
        AutodiffError::InvalidArgument("sub_tensor_f32: contiguous() 直後は必ず Some".to_string())
    })?;
    let begin = start * item_len;
    let end = begin + len * item_len;
    let mut shape = full_shape.to_vec();
    shape[0] = len;
    Tensor::new(flat[begin..end].to_vec(), &shape).map_err(|e| {
        AutodiffError::InvalidArgument(format!("sub_tensor_f32: テンソル再構築に失敗: {e}"))
    })
}

/// [`sub_tensor_f32`] の `Tensor<i32>`（ラベル。常に rank 1）版。
pub fn sub_tensor_i32(
    y: &Tensor<i32>,
    start: usize,
    len: usize,
) -> Result<Tensor<i32>, AutodiffError> {
    let y_shape = y.shape();
    // ドキュメント上の契約（「常に rank 1」）に合わせ rank を厳密に
    // 検査する。`is_empty()`（rank >= 1 の検査）のままだと rank 2 以上
    // の y でも通ってしまい、後段の `flat[start..start+len]` が誤った
    // 要素をスライスして返す（Codex レビュー指摘・イシュー #2202
    // PR #2325）。
    if y_shape.len() != 1 {
        return Err(AutodiffError::InvalidArgument(format!(
            "sub_tensor_i32: y の rank は 1 である必要がある（実際: {y_shape:?}）"
        )));
    }
    let n = y_shape[0];
    // `sub_tensor_f32` と同じ理由で `checked_add` を使う（Codex レビュー
    // 指摘・イシュー #2202 PR #2325）。
    let end_idx = start.checked_add(len).ok_or_else(|| {
        AutodiffError::InvalidArgument(format!(
            "sub_tensor_i32: start（{start}）と len（{len}）の和が usize の範囲を超える"
        ))
    })?;
    if end_idx > n {
        return Err(AutodiffError::InvalidArgument(format!(
            "sub_tensor_i32: [{start}, {end_idx}) が先頭軸の長さ {n} を超える"
        )));
    }
    let contiguous = y.contiguous();
    let flat = contiguous.as_slice().ok_or_else(|| {
        AutodiffError::InvalidArgument("sub_tensor_i32: contiguous() 直後は必ず Some".to_string())
    })?;
    Tensor::new(flat[start..end_idx].to_vec(), &[len]).map_err(|e| {
        AutodiffError::InvalidArgument(format!("sub_tensor_i32: テンソル再構築に失敗: {e}"))
    })
}

/// 固定順（shuffle しない）のミニバッチで `epochs` 回学習し、epoch
/// ごとの平均 loss を返す（`.claude/rules/coding-rust.md`「ベンチは
/// 5 回計測の中央値」とは別軸——学習系は決定的シードのみを要件と
/// する）。
pub fn fit_epochs<M: Trainable>(
    model: &mut M,
    x: &Tensor<f32>,
    y: &Tensor<i32>,
    opt: &mut Adam,
    epochs: usize,
    batch_size: usize,
) -> Result<Vec<f32>, AutodiffError> {
    if epochs == 0 || batch_size == 0 {
        return Err(AutodiffError::InvalidArgument(
            "fit_epochs: epochs・batch_size はいずれも 0 より大きい必要がある".to_string(),
        ));
    }
    let x_shape = x.shape();
    if x_shape.is_empty() {
        return Err(AutodiffError::InvalidArgument(
            "fit_epochs: x の rank は 1 以上である必要がある（スカラーは不可）".to_string(),
        ));
    }
    let n = x_shape[0];
    if n == 0 {
        return Err(AutodiffError::InvalidArgument(
            "fit_epochs: x は空であってはならない".to_string(),
        ));
    }
    if y.shape() != [n] {
        return Err(AutodiffError::InvalidArgument(format!(
            "fit_epochs: x の先頭軸長 {n} と y の shape {:?} が一致しない",
            y.shape()
        )));
    }

    model.set_training(true);
    let mut history = Vec::with_capacity(epochs);
    for _ in 0..epochs {
        // 各バッチの mean loss（`cross_entropy_mean` はバッチ内平均）に
        // バッチサイズ `len` を掛けて標本合計へ戻し、epoch 終端でまとめて
        // 処理標本数 `n` で割ることで epoch loss を標本平均にする
        // （batch_size が n を割り切らない場合、最終バッチが小さいと
        // バッチ数単純平均では過大評価になるため。Codex レビュー指摘・
        // イシュー #2202 PR #2325）。
        let mut total = 0.0f32;
        let mut start = 0;
        while start < n {
            let len = batch_size.min(n - start);
            let xb = sub_tensor_f32(x, start, len)?;
            let yb = sub_tensor_i32(y, start, len)?;
            let loss = model.train_step(&xb, &yb, opt)?;
            total += loss * len as f32;
            start += len;
        }
        history.push(total / n as f32);
    }
    Ok(history)
}

/// eval モードで `x`／`y` 全件に対する正解率を計算する。呼び出し前の
/// training モードを保存し、成功・失敗いずれの経路でも復元する
/// （`predict_in_eval` と同じ方針。Codex レビュー指摘・イシュー #2202
/// PR #2325）。
pub fn accuracy<M: ReferenceModule>(
    model: &mut M,
    x: &Tensor<f32>,
    y: &Tensor<i32>,
    batch_size: usize,
    num_classes: usize,
) -> Result<f32, AutodiffError> {
    if batch_size == 0 {
        return Err(AutodiffError::InvalidArgument(
            "accuracy: batch_size は 0 より大きい必要がある".to_string(),
        ));
    }
    if num_classes == 0 {
        return Err(AutodiffError::InvalidArgument(
            "accuracy: num_classes は 0 より大きい必要がある".to_string(),
        ));
    }
    let x_shape = x.shape();
    if x_shape.is_empty() {
        return Err(AutodiffError::InvalidArgument(
            "accuracy: x の rank は 1 以上である必要がある（スカラーは不可）".to_string(),
        ));
    }
    let n = x_shape[0];
    if n == 0 {
        return Err(AutodiffError::InvalidArgument(
            "accuracy: x は空であってはならない".to_string(),
        ));
    }
    // `fit_epochs` と同様、評価ループの前にラベルの shape を検証する
    // （検証しないと、ラベルが x より多い場合に末尾を無視したまま
    // 正解率を返し、対応関係のずれを検出できない。Codex レビュー
    // 指摘・イシュー #2202 PR #2325）。
    if y.shape() != [n] {
        return Err(AutodiffError::InvalidArgument(format!(
            "accuracy: x の先頭軸長 {n} と y の shape {:?} が一致しない",
            y.shape()
        )));
    }
    // ラベルの範囲外検査も評価ループの前に一括で行う（forward の
    // コストを払う前に入力不備を検出する。`cross_entropy_mean` の
    // range 検証と同じ方針）。
    for i in 0..n {
        let label = y.get(&[i]).ok_or_else(|| {
            AutodiffError::InvalidArgument(format!("accuracy: y[{i}] の読み出しに失敗した"))
        })?;
        if label < 0 || (label as usize) >= num_classes {
            return Err(AutodiffError::InvalidArgument(format!(
                "accuracy: y[{i}]={label} が [0, {num_classes}) の範囲外"
            )));
        }
    }

    // eval モードで評価する。呼び出し前のモードを保存し、成功・失敗
    // いずれの経路でも復元する（`accuracy_in_eval` へ `?` で
    // 早期リターンさせず、戻り値を `result` に受けてから復元する
    // ことで、エラー経路を含め必ず復元させる。Codex レビュー指摘・
    // イシュー #2202 PR #2325。`predict_in_eval` と同じ方針）。
    let original_training = model.is_training();
    model.set_training(false);
    let result = accuracy_in_eval(model, x, y, n, batch_size, num_classes);
    model.set_training(original_training);
    result
}

/// [`accuracy`] の評価ループ本体（モデルが既に eval モードに設定され、
/// 入力（`x`／`y`／`batch_size`／`num_classes`）が検証済みであることを
/// 前提とする private ヘルパー）。
fn accuracy_in_eval<M: ReferenceModule>(
    model: &mut M,
    x: &Tensor<f32>,
    y: &Tensor<i32>,
    n: usize,
    batch_size: usize,
    num_classes: usize,
) -> Result<f32, AutodiffError> {
    let mut correct = 0usize;
    let mut start = 0;
    while start < n {
        let len = batch_size.min(n - start);
        let xb = sub_tensor_f32(x, start, len)?;
        let yb = sub_tensor_i32(y, start, len)?;

        let tape = fandhe_ai::tape();
        let xv = tape.var(&xb);
        let logits = model.forward(&tape, &xv)?;
        let out = logits.to_tensor();
        // 要素数だけを見ると、rank・各軸長が異なる shape でも積が同じ
        // なら通ってしまう（例: [len*num_classes] や [num_classes, len]
        // でも要素数は一致する）。shape そのものを `[len, num_classes]`
        // と完全一致検証する（Codex レビュー指摘・イシュー #2202
        // PR #2325。`cross_entropy_mean` の logits shape 検証と同じ
        // 方針）。
        let out_shape = out.shape().to_vec();
        if out_shape != [len, num_classes] {
            return Err(AutodiffError::InvalidArgument(format!(
                "accuracy: logits の shape は [{len}, {num_classes}] である必要がある \
                 （実際: {out_shape:?}）"
            )));
        }
        let contiguous = out.contiguous();
        let flat = contiguous.as_slice().ok_or_else(|| {
            AutodiffError::InvalidArgument("accuracy: contiguous() 直後は必ず Some".to_string())
        })?;
        for i in 0..len {
            let row = &flat[i * num_classes..(i + 1) * num_classes];
            // 非有限（NaN・inf）の logits を検出する（Codex レビュー指摘。
            // `v > best_val` は NaN に対して常に false を返すため、NaN
            // 汚染された行を無検査のまま通すと best_idx が初期値 0 の
            // ままクラス 0 の予測として誤って「正解」判定されうる
            // （AC4 の `check_ac4` と同様、非有限値を握り潰さず fail-fast
            // する方針。cross_entropy 側の NaN 対策は `d5e86226` 参照）。
            if let Some((idx, &v)) = row.iter().enumerate().find(|&(_, &v)| !v.is_finite()) {
                return Err(AutodiffError::InvalidArgument(format!(
                    "accuracy: logits[{}][{idx}] が非有限（{v}）",
                    start + i
                )));
            }
            let mut best_idx = 0usize;
            let mut best_val = row[0];
            for (idx, &v) in row.iter().enumerate().skip(1) {
                if v > best_val {
                    best_val = v;
                    best_idx = idx;
                }
            }
            let label = yb.get(&[i]).ok_or_else(|| {
                AutodiffError::InvalidArgument(format!("accuracy: targets[{i}] の読み出しに失敗"))
            })?;
            if best_idx as i32 == label {
                correct += 1;
            }
        }
        start += len;
    }
    Ok(correct as f32 / n as f32)
}
