//! 動的テープ式の自動微分エンジン。
//!
//! `tensor-core` が定義するテンソル型・演算グラフの上に、順伝播で実行した演算を
//! 動的テープへ記録し逆伝播で勾配を計算する（REQ-1 v2）。互換 API 層
//! （`compat::array`／`compat::Sequential` 等。REQ-9）はこのテープ機構を
//! 薄くラップして呼び出す。TASK-9.2a（#95）で `compat` モジュールへの隔離を
//! 確定した: 互換 API 層固有のロジック（numpy/Keras 慣習の入出力変換）は
//! `compat` モジュールに閉じ込め、コア（`tape`/`var`/`nn`）側へは互換慣習を
//! 一切漏らさない（依存方向は `compat` → `nn`/`var`/`tape` の一方向で、逆
//! 方向の `use` はない）という設計方針は継続する。
//!
//! **TASK-9.4（イシュー #411）で `compat` の唯一のサポート対象実装は
//! `fandhe_ai::compat` へ移設した**（10 クレート化・`facade` 新設〈TASK-9.3・
//! #410〉を受け、compat 公開面を composition root と同じ `facade`
//! クレートへ一本化するサポート境界の明文化。`docs/compat-api-scope.md`
//! 「サポート境界」節・`docs/spec/04-requirements.md:209-210` の
//! 2026-08-08 追記参照）。**移行期間中は本クレートの `compat` モジュール
//! （`pub mod compat`・非推奨シム）に旧実装を複製して残し、既存の
//! `fandhe_ai_autodiff::compat::{array, Sequential, SequentialVars}` 利用コードの
//! ソース互換性を保つ**（codex-review PR #424 P1 是正。詳細は
//! `crates/autodiff/src/compat/mod.rs` モジュール doc 参照）。本
//! クレート（`autodiff`）はこの移設後も compat 層が依拠する `Tape`/`Var`/
//! `nn`（`Module`・`Linear`・`activation` 等）を `pub` API として提供し
//! 続ける内部クレートである（利用者が `autodiff` を直接使うことは
//! サポート対象外。同ドキュメント参照）。
//!
//! TASK-1.5a（#16）でテープ構造（`Tape`/`TapeId`/`NodeId`）・
//! forward 演算群（`Var::matmul`/`add`/`mul`/`relu`/`exp`/`tanh`/`sum`/
//! `max`/`mse_loss`）の値計算とノード記録を実装した（spec 根拠:
//! `docs/spec/05-tasks.md` TASK-1.5、`docs/public-api-design.md` §3）。
//! `Op`（`tape.rs`・非公開）が各演算の入力 `NodeId` を保持する構造にする
//! ことで、後続タスクが発生順に記録されたノード列を逆走査できる下地とする。
//!
//! TASK-1.5b（#17）で各演算の勾配関数（VJP: vector-Jacobian product）と
//! `Op` 単位のディスパッチ入口 `vjp()`（`grad.rs`・非公開）を実装した。
//! 数値微分との突合テスト（受け入れ条件）は `grad.rs` 内のユニット
//! テストに含む。
//!
//! TASK-1.5c（本イシュー・#18）で勾配伝播 API（`Tape::backward`・
//! `Gradients`。`backward.rs`）を実装した。テープを発生順とは逆順に
//! 走査して `grad::vjp()` を呼び、複数経路から同一ノードへ流入する
//! 勾配を合算する（PoC-v2-2 の `accumulate()` 相当）。合成関数
//! end-to-end 勾配の受け入れ条件検証は `tests/backward.rs` に含む。
//!
//! TASK-1.5d（#19）で PoC-v2-2 数値突合の回帰テストを追加した
//! （`tests/poc_v2_2_parity.rs`）。PoC-v2-2 の確定ケース（2 層 MLP grad
//! check・50 step SGD 学習の決定性）を `Tape`/`Var`/`Tape::backward`/
//! `Gradients` 経由で再現し、PoC evidence（`docs/spec/03-poc/
//! poc-v2-2-autodiff/evidence/`）の判定結果と整合することを固定する。
//! これにより #16〜#19（TASK-1.5 全体）が完了する。
//!
//! TASK-9.1b（#92）で活性化関数プリミティブ `Var::sigmoid`
//! （`var.rs`・`Op::Sigmoid`・VJP は `grad.rs`）と、互換 API 層
//! （REQ-9）が積む薄いレイヤー実装群の入口 `nn`（`nn::activation`。
//! ReLU/Sigmoid/Tanh）を追加した。共通 `Module` trait の定義は
//! TASK-9.2（#94/#95・`compat::Sequential`）に委ねる。
//!
//! forward の値計算は `backend-cpu`（TASK-1.6・#20 以降。並行実装中で
//! 未完）が完成するまでの暫定参照実装（`eval.rs`、非公開）で行い、
//! TASK-1.9（バックエンド抽象層への接続）で backend 経由の実行に
//! 差し替える（PoC-v2-2 と同じ構成）。`grad.rs`・`backward.rs` も同じ
//! `eval.rs` のヘルパーを再利用するため、差し替えの影響範囲は
//! forward/backward 双方でこの 1 ファイルに閉じる。
//!
//! TASK-9.1a（#91）で `nn` モジュール（`Linear` 等、自作 NN モジュール）
//! を追加した。`nn` は `Tape`/`Var` に直接依存する自作コア側の部品で
//! あり、互換 API 層（`compat::array`/`compat::Sequential`。REQ-9・
//! TASK-9.2）とは区別される（`nn/mod.rs` の境界説明を参照）。上記の
//! 「互換レイヤ固有のロジックを持ち込まない」方針は `compat` 層本体を
//! 指しており、`nn` モジュールには適用されない。
//!
//! #190（親 #189「損失関数（MSE・CrossEntropy）の実装」）で
//! `Var::mse_loss_with`/[`Reduction`]（mean/sum 縮約）と
//! `nn::loss::MseLoss`（薄いラッパー）を追加した。既存 `Var::mse_loss`
//! は `mse_loss_with(target, Reduction::Mean)` への委譲に変更したが、
//! シグネチャ・既定の意味（mean）は維持する（公開 API 非破壊）。
//!
//! #191（親イシュー #189）で CrossEntropy 損失（log-sum-exp 安定化・
//! クラス次元指定）を追加した。`Var::cross_entropy_loss`（`var.rs`・
//! `Op::CrossEntropyLoss`）は log-softmax → NLL を個別オペ合成せず
//! `MseLoss` と同じ 1 個の融合オペとして実装し、VJP（`grad.rs`）は
//! 解析形 `softmax(x) − onehot(t)` で閉じる。`nn::loss::
//! CrossEntropyLoss` はその薄いラッパー。`Reduction`（`Mean`/`Sum`）は
//! #190 が `var.rs` に定義したものをそのまま再利用する（`nn::loss` 側に
//! 重複定義は置かない）。
//!
//! #193（親 #192「optimizer（SGD・AdamW）・gradient clipping の実装」）
//! で optimizer の第 1 分割 `optim::Sgd`/`optim::SgdConfig`（momentum・
//! dampening・weight decay・nesterov 対応。PyTorch `torch.optim.SGD`
//! 準拠）を追加した。`nn`（`Tape`/`Var` に直接依存する層プリミティブ）
//! とは別モジュールとし（`optim/mod.rs` 参照）、既存 `nn`/`lib.rs` 冒頭の
//! 記述は変更しない。AdamW（#194）・gradient clipping／LR スケジューラ
//! （#195）は `optim` 配下への後続分割。
//!
//! TASK-9.2a（#95）で互換 API 層（当時は本クレート内の `compat`
//! モジュール）を追加した。共通 `nn::Module` trait（`nn/module.rs`）を
//! 確定し、`compat::array`（numpy `np.array` 慣習のテンソル生成）・
//! `compat::Sequential`（Keras `Sequential` 慣習のレイヤー積み上げ
//! ビルダー）を実装した。`Sequential` は `nn::Module` 経由で `Linear`・
//! 活性化関数（ReLU・Sigmoid・Tanh）を均一に扱う（対象範囲は
//! `docs/compat-api-scope.md` 準拠。学習〈勾配取得・パラメータ更新〉は
//! 当時対象外）。**TASK-9.4（#411）で唯一のサポート対象実装を
//! `fandhe_ai::compat` へ移設し、本クレートの `compat` モジュールは移行期間中
//! のソース互換シムとして実装を複製して残した**（本ファイル冒頭の
//! クレート doc・`compat/mod.rs` 参照。`nn::Module`・`Linear`・
//! `activation` 等、`compat` が依拠する `nn` 側の部品は本クレートに残る）。

//! #1047（親 #1043「カーネル融合・autodiff 実行モデルの強化」）で
//! view 系ノード `Var::reshape`/`Var::transpose`（`tape::Op::Reshape`/
//! `Op::Transpose`）を追加した。`push_eager`/`push_lazy` に続く第 3 の
//! 登録経路 `Tape::push_view` はホスト値を一切保持せず、backward 時
//! （または後続の実体化要求時）に入力ノードから `tape::resolve_view`
//! で再導出する（burn-autodiff の `MemoryBound { retro_forward }` 相当。
//! `tensor-core::Tensor::reshape`/`transpose` の zero-copy 性質（`Arc`
//! 共有）を利用するため中間バッファを一切確保しない。設計判断・メモリ
//! 実測は `docs/autodiff-view-recompute-decision.md` を参照）。
//! elementwise 5 演算の融合連鎖には参加しない融合境界ノードであり、
//! これは `docs/kernel-fusion.md` が既に確定させた「transpose を挟む
//! 連鎖は融合しない」方針と整合する。

//! イシュー #1624 で activation checkpointing（`torch.utils.checkpoint`
//! 相当）を追加した。`Tape::checkpoint`（閉包版）／`Var::checkpoint_from`
//! （低儀式版。facade からはこちらのみ既存の `Var` 再エクスポート経由で
//! 到達可能）が区間内の再計算可能ノード（`Op::MatMul`／`Sigmoid`／
//! `Sum`／`Max`。`Op::is_checkpoint_eligible()`）の forward 値を解放し、
//! 上記 view 系ノードの `resolve_view` 機構を一般化した `tape::
//! recompute_value`／`recompute_fallible`／`recompute_infallible` が
//! backward 時に再導出する。設計判断・実装記録は `docs/
//! autodiff-checkpoint-design.md` を参照。
//! #1597 で同じ `push_view`／`resolve_view` 骨格を任意軸並べ替え・
//! ブロードキャストへ一般化した `Var::permute`（`tape::Op::Permute`。
//! zero-copy）・`Var::broadcast_to`／`expand`（`tape::Op::BroadcastTo`。
//! forward は stride 0 view で zero-copy だが VJP は `Op::Add`/`Op::Mul`
//! の暗黙ブロードキャストと同じ `reduce_to_shape` 縮約を使うため勾配
//! バッファを確保する）を追加した。`Var::squeeze`／`unsqueeze`／
//! `flatten` は新規 `Op` を持たず `Var::reshape` へ委譲する（案 A
//! 制約を継承。`docs/autodiff-view-recompute-decision.md` §5 が予告
//! した拡張）。

//! イシュー #1620 で `Var::einsum`（PyTorch `torch.einsum`／TF
//! `tf.einsum` 相当の汎用縮約記法）を追加した。新規カーネルは追加せず、
//! 既存の `Var::matmul`（GEMM）・`sum`（縮約）・`permute`／`reshape`
//! （view）・`mul`（broadcast 乗算）への分解（`einsum` モジュール）
//! として実装しているため、VJP は分解先各演算の VJP 合成として自動的
//! に成立する（`einsum` 専用の VJP は `grad.rs` に追加していない）。
//! 分解の前段で `permute` 後の非 contiguous view を明示実体化する
//! eager ノード `Var::contiguous`（`pub(crate)`。`tape::Op::Contiguous`）
//! も新設した。受理範囲（batch 添字を伴う縮約は rank≥3 `matmul`
//! 〈#1600〉未実装のため拒否等）・分解アルゴリズムは `einsum` モジュール
//! doc を参照。

//! イシュー #1946（親 #1944・設計確定は兄弟 #1945）でユーザー定義
//! forward／backward プラグイン機構（案 B。`Op` enum への trait object
//! variant）を追加した。[`CustomFunction`]（`custom.rs`）を実装し
//! `Tape::custom` へ渡すことで、組み込み演算では表現できない独自の
//! 勾配（straight-through estimator・gradient reversal 等）をグラフへ
//! 登録できる。facade（唯一のサポート対象公開面）へはイシュー #2549
//! （`docs/autodiff-custom-function-decision.md` §16.1）で
//! `fandhe_ai::CustomFunction` の再エクスポートと facade `Tape::custom`
//! の薄い委譲として公開済み（`Var::custom` は設けない）。

//! イシュー #2141（親 #2131）で bool を返す比較 6 種
//! （`gt_bool`／`ge_bool`／`lt_bool`／`le_bool`／`eq_bool`／`ne_bool`）・
//! logical 3 種（`logical_and`／`logical_or`／`logical_not`）・
//! `masked_select` を [`bool_ops`] へ追加した。いずれも非微分・tape
//! 非記録の自由関数で、比較 6 種と `masked_select` は #2510 で `Var` の
//! 委譲メソッドとして facade 公開済み。logical 3 種と自由関数自体は
//! 公開しない（`docs/autodiff-bool-ops-exposure-decision.md`・
//! モジュール doc 参照）。

//! イシュー #2195（親 #2142「f64 autograd の最小集合」の第 1 段）で
//! `f32` の `Tape`/`Var` とは完全に独立した f64 専用グラフ
//! [`f64_autograd::TapeF64`]/[`f64_autograd::VarF64`] を追加した。
//! elementwise 4 演算（add・mul・div・pow）の forward と VJP・1 step
//! backward を持つ（`add`/`mul` は `Tape::typed_ops_f64` が `Some`
//! ならネイティブ実装へ委譲し `None`/`Unsupported` ならホスト参照実装
//! へフォールバック、`div`/`pow` は常にホスト参照実装）。`Var`
//! （本ファイル既存の f32 版）へ inherent メソッドは追加せず、facade
//! も再エクスポートしない内部クレート限定 API（`docs/autodiff-
//! var-dtype-multiplexing-design.md` §10 の承認事項はいずれも未承認の
//! まま消費しない。設計判断・バックエンド別 dispatch 表は
//! `f64_autograd` モジュール doc・同 doc §13 を参照）。

//! イシュー #2144（親 #2131）で `tril`／`triu`／`diag`／`trace`／
//! `outer`／`dot` の 6 種形状・行列演算を [`matrix_ops`] へ追加した。
//! [`bool_ops`]／[`rearrange_ops`] と同じく非公開の自由関数群（`Var`
//! への inherent メソッドではない）で、facade 公開は承認待ちのため
//! 意図的に再エクスポートしない（`docs/autodiff-matrix-ops-decision.md`・
//! モジュール doc 参照）。

//! イシュー #2146（親 #2131）で `mish`／`hardtanh`／`relu6`／`prelu`／
//! `glu` の 5 活性化演算を [`activation_ops`] へ追加した。facade へは #2516 で
//! `Var` の委譲メソッドとして公開済み（モジュール自体は意図的に再エクスポート
//! しない。`compat::Sequential::add_*` は #2529 で公開済み。
//! `docs/autodiff-activation-ops-decision.md`・モジュール doc 参照）。

//! イシュー #2649（親 #2648）で `selu`／`celu`／`softsign`／`hardsigmoid`／
//! `log_sigmoid` の 5 活性化演算を [`activation_scalar_ops`] へ追加した
//! （`ScalarUnaryOp` 5 variant への薄い委譲・新規 `Op` ゼロ）。facade 公開は
//! 承認依頼 #2677 の承認待ちで保留（公開は #2678・#2679。
//! `docs/autodiff-activation-scalar-ops-decision.md`）。

//! イシュー #2147（親 #2131）で `prod`／`logsumexp`／`any`／`all`／
//! `norm_p`（p-ノルム）の 5 縮約を [`reduce_ops`] へ追加した。
//! #2514（ルート #2499 の一括承認）で `Var` の 1 行委譲メソッドとして
//! facade へ公開済み（モジュール自体は再エクスポートしない。
//! `docs/autodiff-reduce-ops-decision.md`・モジュール doc 参照）。

//! イシュー #2149（親 #2131）で、[`crate::var::Var::einsum`] が
//! rank≥3 `matmul`（#1600 未実装）を理由に拒否していた batch 添字
//! （両オペランドと出力に共通する添字。例 `"bij,bjk->bik"`）を伴う
//! 2 項縮約を、その後実装済みの rank≥3 `Var::matmul`（イシュー
//! #1715）へ分解する経路として `crate::einsum` 内に実装した。
//! 当初は承認待ちのため内部クレート限定の到達入口 [`einsum_batch`]
//! だけを公開していたが、イシュー #2517 で `Var::einsum` 自体が
//! batch 添字付き縮約を受理するよう拡張され facade へ公開された
//! （`docs/autodiff-einsum-batch-decision.md` §11）。[`einsum_batch`]
//! は公開済み 0.10.0 互換の同一挙動の薄い委譲として維持する。

//! イシュー #2153（親 #2131）で、`Var::topk`（`sorted=True` 固定・
//! 非負 `dim` のみ）・`Var::unique`（`dim`／`return_inverse`／
//! `return_counts` 非対応）のオプション拡張（`sorted=false`・負
//! `dim`・unique の `dim` 指定・`return_inverse`・`return_counts`・
//! `unique_consecutive`）を [`topk_unique_ops`] へ追加した。イシュー #2519 で
//! `Var::topk_with_options`／`unique_with_options`／`unique_consecutive` の
//! 1 行委譲メソッドとして facade へ公開済み（入出力型 `TopkOptions`／
//! `UniqueOptions`／`UniqueOutput` はルートから再エクスポート。モジュール自体は
//! facade へ再エクスポートしない。`docs/autodiff-topk-unique-ops-decision.md` §6）。

//! イシュー #2154（親 #2131）で `amax`／`amin`（PyTorch `torch.amax`／
//! `amin` 相当。タイに勾配を均等分配する VJP）を [`extremum_ops`] へ
//! 追加した。既存 `Var::max`／`min`／`max_dims`（先勝ち決定的方式。
//! イシュー #1718 で出荷済み挙動として維持を確定）とは独立の `Op`
//! （`tape::Op::Amax`／`Amin`）・VJP として実装し、既存経路の勾配値は
//! 変えない。#2514（ルート #2499 の一括承認）で `Var::amax`／`amin` の
//! 1 行委譲メソッドとして facade へ公開済み（モジュール自体は再エクスポート
//! しない。`docs/autodiff-amax-grad-distribution-decision.md`・モジュール
//! doc 参照）。

pub mod activation_ops;
pub mod activation_scalar_ops;
mod adaptive_max_pool_ops;
mod attention;
mod backward;
pub mod binning_ops;
pub mod bool_ops;
pub mod compat;
pub mod conv3d_ops;
mod create_graph;
pub mod cumulative_ops;
mod custom;
mod default_ops;
pub mod determinism;
mod einsum;
pub mod einsum_batch;
mod error;
mod eval;
pub mod extremum_ops;
pub mod f64_autograd;
pub mod fft_ops;
pub mod indexed_update_ops;
// LLM 推論向け自己回帰生成ループ（イシュー #2191。`nn` とは別の推論
// ループ層のためトップレベルに置く。`activation_ops`／
// `topk_unique_ops` と同型）。facade（`fandhe_ai`）への公開は
// `pub fn generate`／`GenerateConfig` の署名がユーザー承認事項のため
// 保留する（`docs/facade-generate-decision.md`）。
pub mod generate;
mod grad;
pub mod indexing_ops;
mod layout;
pub mod linalg_ops;
pub mod loss_ops;
// MatMul と elementwise 5 演算の opt-in 低精度 forward（イシュー #2628）。
// facade への公開は保留（承認依頼 #2677・公開 #2678。
// `docs/autodiff-low-precision-op-extension-decision.md`）。
pub mod low_precision_ops;
pub mod matrix_ops;
pub mod nn;
pub mod nonfinite_ops;
pub mod optim;
pub mod pad_ops;
// 3D プーリング 2 演算（`max_pool3d`・`avg_pool3d`。イシュー #2643）。facade への公開は保留
// （承認依頼 #2677・公開 #2678。`docs/autodiff-pool3d-ops-decision.md`）。
pub mod pool3d_ops;
// ConvTranspose3d と MaxUnpool1d／2d／3d（`conv_transpose3d`・`max_unpool{1,2,3}d`。イシュー #2644）。
// facade への公開は保留（承認依頼 #2677・公開 #2678。
// `docs/autodiff-conv-transpose3d-max-unpool-decision.md`）。
pub mod conv_transpose3d_ops;
// Fold／Unfold（`fold`・`unfold`。イシュー #2645）。既存 `im2col`／`col2im` の再利用。facade への公開は
// 保留（承認依頼 #2677・公開 #2678。`docs/autodiff-fold-unfold-decision.md`）。
pub mod fold_ops;
// LocalResponseNorm（`local_response_norm`。イシュー #2646）。facade への公開は保留（承認依頼 #2677・
// 公開 #2678。`docs/autodiff-lrn-weight-reparam-decision.md`）。
pub mod lrn_ops;
pub mod max_unpool_ops;
pub mod rearrange_ops;
mod reduce_dims;
pub mod reduce_ops;
pub mod scalar_unary_ops;
// 形状演算 6 種（`unbind`・`movedim`・`swapaxes`・`tensor_split`・`meshgrid`・`rot90`。
// イシュー #2639）。既存 Op の合成のみ。facade への公開は保留（承認依頼 #2677・公開
// #2678。`docs/autodiff-shape-view-ops-decision.md`）。
pub mod shape_view_ops;
// Softmin・Tanhshrink・RReLU・Threshold（`softmin`・`tanhshrink`・`threshold`・`rrelu`・
// `rrelu_with_noise`。イシュー #2650）。既存 Op の合成のみ。facade への公開は保留（承認依頼
// #2677・公開 #2678・層化 #2679。`docs/autodiff-softmin-threshold-ops-decision.md`）。
pub mod softmin_threshold_ops;
// pos_weight 付き BCEWithLogits・HingeEmbedding・SoftMargin・GaussianNLL（`bce_with_logits_loss_with`・
// `hinge_embedding_loss`・`soft_margin_loss`・`gaussian_nll_loss`。イシュー #2652）。新規 Op 4 種 +
// ホスト参照実装。facade への公開は保留（承認依頼 #2677・公開 #2678。
// `docs/autodiff-elementwise-loss-ops-decision.md`）。
pub mod elementwise_loss_ops;
// MultiMargin・MultiLabelMargin・MultiLabelSoftMargin・sigmoid focal loss（`multi_margin_loss`・
// `multilabel_margin_loss`・`multilabel_soft_margin_loss`・`sigmoid_focal_loss`。イシュー #2653）。
// 新規 Op 4 種 + ホスト参照実装。facade への公開は保留（承認依頼 #2677・公開 #2678。
// `docs/autodiff-margin-focal-loss-ops-decision.md`）。
pub mod margin_focal_loss_ops;
// 結合 4 演算（`merge_concatenate`・`merge_add`・`merge_multiply`・`merge_average`。
// イシュー #2666）。既存 Op の合成のみ。facade への公開は保留（承認依頼 #2677・公開 #2679。
// `docs/facade-functional-api-decision.md` §17）。
pub mod merge_ops;
pub mod stat_reduce_ops;
mod tape;
#[cfg(test)]
mod test_support;
// テンソル積・距離・外積 4 演算（`kron`・`tensordot`・`cdist`・`cross`。イシュー #2640）。
// 既存 Op の合成のみ。facade への公開は保留（承認依頼 #2677・公開 #2678。
// `docs/autodiff-tensor-product-ops-decision.md`）。
pub mod tensor_product_ops;
pub mod topk_unique_ops;
pub mod trig_ops;
mod var;
// 重み再パラメータ化（`weight_norm`・`norm_except_dim`・`spectral_norm`・`SpectralNormState`。
// イシュー #2646）。facade への公開は保留（承認依頼 #2677・公開 #2678・層化 #2679。
// `docs/autodiff-lrn-weight-reparam-decision.md`）。
pub mod weight_reparam_ops;

pub use backward::Gradients;
// 子テープ方式の高階微分（`create_graph`。イシュー #1942・設計
// `docs/autodiff-higher-order-grad-decision.md` §7〜§9）: `Tape::
// backward_create_graph`（`impl Tape` ブロック内で定義。`create_graph.rs`
// を参照）に加え、その戻り値型 `CreateGraphResult` を公開する。facade
// は `fandhe_ai::CreateGraphResult` として再エクスポート済み（#2545。
// 同 doc §17.2・§19。承認はルート #2499 の一括承認）。
pub use create_graph::CreateGraphResult;
pub use custom::CustomFunction;
pub use error::AutodiffError;
pub use tape::{NodeId, Tape, TapeId};
// `manual_seed`（イシュー #1724）・`randn`／`rand`／`randint`（イシュー
// #1725）: PyTorch `torch.manual_seed`／`randn`／`rand`／`randint` 相当の
// プロセスグローバル決定的 RNG 契約とその消費側（乱数テンソル生成）。
// 実体は `tensor-core::rng`（本クレートの `nn::init::Xorshift64Star` が
// 委譲する共通コアと同じ場所）にあり、`autodiff` はここで素通しするのみ。
// `facade` がさらにこれを再委譲する（composition root。
// `docs/rng-global-contract-design.md`）。
// `bernoulli`／`multinomial`／`normal`・`Generator`（イシュー #2156）:
// PyTorch `torch.bernoulli`／`torch.multinomial`／`torch.normal`／
// `torch.Generator` 相当の確率分布サンプラーと、グローバル RNG 状態と
// 完全に独立した乱数源。同じく `tensor-core::rng` が実体で本クレートは
// 素通しするのみ。**facade への再委譲は保留**（ユーザー承認待ち。
// `docs/rng-global-contract-design.md` §13・`docs/compat-api-scope.md`
// §1.2）——`fandhe_ai_autodiff::normal`（本関数。引数順
// `mean, std, shape`）と `nn::init::normal`（`shape, mean, std`）は
// パスが異なるため衝突しない。
pub use fandhe_ai_tensor_core::rng::{
    Generator, RngError, bernoulli, manual_seed, multinomial, normal, rand, randint, randn,
};
// `arange`／`linspace`／`eye`／`zeros_like`／`ones_like`（イシュー
// #1726）: PyTorch 相当の決定的テンソル生成 API。`rng` と同じく実体は
// `tensor-core::creation` にあり、本クレートは素通しするのみ（`facade`
// がさらに再委譲する。`docs/rng-global-contract-design.md` §11）。
pub use fandhe_ai_tensor_core::creation::{
    CreationError, arange, eye, linspace, ones_like, zeros_like,
};
// `topk_unique_ops` の入出力型（イシュー #2519。facade が `Var` の委譲メソッドと
// ともに再エクスポートする。`QrVars`／`SvdVars` と同じ「ルート経由」の形）。
pub use topk_unique_ops::{TopkOptions, UniqueOptions, UniqueOutput};
pub use var::{GateParams, QrVars, Reduction, SvdVars, Var, VarHostView};
