//! ONNX import ラッパー（イシュー #2017・`docs/facade-onnx-import-
//! exposure-decision.md` §4 案 B の実装）。
//!
//! `fandhe_ai_onnx_interop`（内部クレート。crates.io 公開名
//! `fandhe-ai-onnx-interop`）の `onnx::proto::decode_model` →
//! `onnx::graph::build_graph` → `onnx::interp::run`／`run_with_ops` を
//! 1 つの薄い型 [`OnnxModel`] に束ねる。**推論専用**であり
//! **autograd 未接続**（入出力は [`crate::Tensor`] であり `Var` ではない。
//! 勾配は取れない）。
//!
//! **既定はホスト CPU 実行のみ**（`BackendOps`／`Device` 非経由。
//! イシュー #2077 導入前と bit 完全に不変）。[`crate::
//! set_cuda_onnx_gpu_execution_enabled`]／`crate::
//! set_metal_onnx_gpu_execution_enabled`〈macOS 限定 cfg のため非 macOS
//! ビルドでは存在せずリンク化しない〉の opt-in（既定 OFF）が有効な
//! 場合のみ、[`OnnxModel::run`] は `BackendOps` 経由の device 実行
//! （op 単位。`Unsupported`／`ShapeMismatch` はホストへフォールバック・
//! それ以外のエラーは fail-closed）を試みる（詳細は [`OnnxModel::run`]
//! のドキュメンテーションコメント・`docs/onnx-gpu-execution-decision.md`
//! を参照）。
//!
//! 数値契約は REQ-7 判定式（`abs_err/(|ref|+1e-6) <= 1e-3`。
//! `crates/onnx-interop/tests/onnx_poc_v2_6_match.rs` 系と同一）であり、
//! `.claude/rules/coding-rust.md` の REQ-2 統一複合判定（バックエンド間
//! 数値一致）とは別指標である（両者を混同しない）。
//!
//! import 対応 op は 28 種（`fandhe_ai_onnx_interop::onnx::interp` 冒頭
//! コメント参照。イシュー #2200 で `GlobalAveragePool`／
//! `BatchNormalization`／`Flatten`、イシュー #2199 で `MaxPool`／
//! `AveragePool` を追加・`Conv` に 1D 対応を追加した）。export allowlist は
//! イシュー #2187 で 26 種へ拡大した（下記「ONNX export」節参照）が、
//! `MaxPool`／`AveragePool` の 2 op は import 専用のため export 非対応の
//! まま残る（import・export 非対称）。未対応 `op_type` は無言 skip せず
//! [`OnnxError::UnsupportedOp`]
//! で fail-closed に拒否する（no-silent-skip 契約。`.claude/rules/
//! security.md` A03）。`run` の `feeds` は ONNX の pre-IR-4 セマンティクス
//! どおり同名 initializer を上書きする。`GraphProto.sparse_initializer` が
//! 非空の場合も同じ契約に従い [`OnnxError::SparseInitializerNotSupported`]
//! で拒否する（sparse テンソルは非対応。イシュー #2079）。
//!
//! [`OnnxValue::F16`] は `half::f16` を素通しする。facade は `half` を
//! 再エクスポートしないため、`half::f16` を名指しして扱うには利用者側が
//! `half` クレートへ直接依存する必要がある（`TypedOps<half::f16>` と同じ
//! 扱い。`docs/backend-dtype-dispatch-design.md` §16 の同種注記参照）。
//!
//! ## ONNX export（イシュー #2018）
//!
//! [`OnnxModel::to_bytes`]／[`OnnxModel::to_path`] は import 済みモデルの
//! roundtrip export ラッパーである（`docs/facade-onnx-export-exposure-
//! decision.md` §4 案 B。承認事項は同 doc §10・#2018 承認コメント）。
//! `fandhe_ai_onnx_interop::onnx::export::build_model_proto`（allowlist
//! による fail-closed 検査込み）→ `onnx::export::try_encode_model`
//! （`onnx::proto::encode_model` と同一バイト列の失敗可能確保版。PR #2348
//! codex P0 是正）への薄い委譲のみで、以下を doc として明記する:
//!
//! - `value_info` は常に空。本クレート内 roundtrip は bit 同一で保証する
//!   が、`onnx.checker` 等の外部ツールでの厳密な妥当性検証は保証しない
//!   （`export.rs` モジュール冒頭コメント）
//! - [`OnnxExportOptions`] の既定値（`ir_version=8`／`opset_version=17`）
//!   が書き出される。**import 時に元モデルの `opset_import`／
//!   `ir_version`／`producer_name`／グラフ名は [`fandhe_ai_onnx_interop::
//!   onnx::graph::Graph`] に保持されない**ため、export 結果は options の
//!   値（と内部固定の producer／graph 名）になる。元モデルの opset と
//!   合わせる責任は利用者側にある
//! - tensor は常に `raw_data`（リトルエンディアン）のみで書き出す。
//!   initializer は名前順で決定的に出力する（同一モデルの `to_bytes` は
//!   何度呼んでも同一バイト列）
//! - ホスト CPU 実行のみ・`BackendOps`／`Device` 非経由。`to_bytes`／
//!   `to_path` で export できるのは import 済みモデル（`OnnxModel`）の
//!   みで、学習済み `Sequential`／`nn` から直接 `OnnxModel` を構築する
//!   経路は [`OnnxModel::from_sequential`]（次節・#2037）を使う
//! - export allowlist（26 op・既定 domain。イシュー #2187 で
//!   `GlobalAveragePool`／`BatchNormalization`／`Flatten` を追加し
//!   #2200 導入時点の import 対応 26 op と対称化した）外のノードを含む
//!   モデルは `from_bytes` では構築できても **export 時に**
//!   [`OnnxError::UnsupportedOp`] により fail-closed に拒否する
//!   （無言 skip しない。イシュー #2200・#2187）。import 対応はその後
//!   イシュー #2186 で `Clip`／`Tanh`／`Gelu`／`Where`／`Expand`／
//!   `ReduceMean`／`Pad`／`Resize` の 8 op、イシュー #2199 で
//!   `MaxPool`／`AveragePool` の 2 op を追加し 36 op へ拡大したが、
//!   `interp` のディスパッチ表への追加のみで本 export allowlist は
//!   未拡張のまま（26 op）。この 10 op を含むモデルは import はできても
//!   export では [`OnnxError::UnsupportedOp`] になる非対称が残る
//!   （追跡: 別イシューでの export 側拡張が必要。out-of-scope-tracking.md
//!   に従いユーザー承認を得たうえで Issue 化する）
//!
//! ## `Sequential` からの export（イシュー #2037・親 #2034）
//!
//! [`OnnxModel::from_sequential`] は学習済み [`crate::compat::Sequential`]
//! から直接 [`OnnxModel`] を構築する（`torch.onnx.export` 相当の
//! ユーザーストーリー。`docs/facade-onnx-export-exposure-decision.md`
//! §17）。`fandhe_ai_onnx_interop::onnx::export_nn::graph_from_layers`
//! （#2036）への 1 段委譲のみで、構築した [`Graph`] を encode／decode
//! による正規化を挟まず直接 [`OnnxModel`] が保持する。以下を doc として
//! 明記する:
//!
//! - **対応層は `Linear`／`ReLU`／`Softmax`／`LayerNorm`／
//!   `GELU`（erf 版）／`Conv2d` の 6 種**（イシュー #2076・親 #2034 で
//!   `Linear`／`ReLU` の 2 種から拡大。`Sigmoid` は数値契約
//!   〈`docs/facade-onnx-export-exposure-decision.md` §15.7 項 5〉が
//!   承認保留のため対象外のまま。Tanh・GeluTanh・LogSoftmax・
//!   Conv1d 等は引き続き非対応）。1 つでも非対応層を含む場合は `Graph`
//!   を一切構築せず [`OnnxError::UnsupportedLayer`] で fail-closed に
//!   拒否する（部分的に構築されたモデルを返さない）
//! - graph input 名は常に `"input"`・output 名は常に `"output"`
//!   （最終層の出力）。initializer 名は `{i}.weight`／`{i}.bias`
//!   （`i` は層の位置。`Sequential::state_dict` と同じ命名規約）
//! - 空の `Sequential`（層 0 個）は [`OnnxError::InvalidModel`] で拒否
//!   される（`ExportError::EmptyModel` の写像）
//! - `value_info` は常に空（前節と同じ制約）
//! - ホスト CPU 実行のみ・`BackendOps`／`Device` 非経由（学習済み
//!   パラメータの値をそのままコピーするのみで算術を行わない）
//! - 数値契約: `Linear`／`ReLU` のみのモデルは、`from_sequential(&m)
//!   .to_bytes(opts)` → `from_bytes` → `run` の出力が `Linear→ReLU`
//!   入力に NaN が現れず GEMM 出力に厳密な `±0.0` が現れない限り
//!   `m.predict(&x)` と bit 完全一致する（`export_nn` モジュール doc
//!   「bit 一致契約の前提」参照）。Softmax・LayerNorm・GELU・
//!   Conv2d を含むモデルは結合順序・実装経路が異なるため REQ-2 統一
//!   複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）で検証する
//!   （`crates/facade/tests/interop_onnx_export_layers_parity.rs`）
//!
//! ## 非信頼入力の扱い
//!
//! ONNX は非信頼な外部フォーマットである。本ラッパーは検証を迂回・複製
//! せず、`fandhe_ai_onnx_interop::onnx::graph::build_graph` の既存検査
//! （dims 非負・要素数 overflow 拒否・データ長／バイト長一致・名前
//! 重複／SSA／トポロジカル順検証）と `onnx::interp::run` の
//! no-silent-skip 契約をそのまま通す（迂回・複製しない）。`from_path` は
//! パスを `std::fs::read` へそのまま渡すのみでシェル展開・パス連結は
//! 行わない（external data 解決の基点ディレクトリ〈`base_dir`〉も `path`
//! の親ディレクトリをそのまま使うのみで、こちらもシェル展開・パス連結は
//! 行わない。external data 自体の非信頼入力検証は `onnx::external_data`
//! の 2 パス設計〈`docs/onnx-external-data-decision.md`〉に迂回・複製せず
//! 委譲する）。入力総バイト数・要素数の明示上限は導入していない
//! （`build_graph` の長さ整合検査がバイト長を初期入力長で抑える。値の
//! 決定にユーザー承認が要るため本 issue のスコープ外。
//! `docs/facade-onnx-import-exposure-decision.md` §6.3 参照）。`from_bytes`
//! は `onnx::proto::decode_model` の bounded 事前走査（イシュー #2079
//! codex-review 是正。`proto.rs` モジュール冒頭コメント「メモリ増幅対策」
//! 節）による `sparse_initializer` の早期 fail-closed 拒否を
//! `map_decode_error`（非公開関数）でそのまま [`OnnxError::SparseInitializerNotSupported`]
//! へ写像する（`build_graph` 側の同名エラーと同じ payload）。

use std::collections::HashMap;
use std::fmt;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

use fandhe_ai_onnx_interop::onnx::export::{
    ExportError, ExportOptions, build_model_proto, try_encode_model,
};
use fandhe_ai_onnx_interop::onnx::export_nn::graph_from_layers;
use fandhe_ai_onnx_interop::onnx::external_data::{
    DEFAULT_MAX_EXTERNAL_DATA_TOTAL_BYTES, DEFAULT_MAX_EXTERNAL_FILES, ExternalDataError,
    ExternalDataOptions, build_graph_with_external_data,
};
use fandhe_ai_onnx_interop::onnx::graph::{Graph, GraphError, build_graph};
use fandhe_ai_onnx_interop::onnx::interp::{
    InterpError, Value as InterpValue, run as interp_run, run_with_ops as interp_run_with_ops,
};
use fandhe_ai_onnx_interop::onnx::proto::{DecodeModelError, decode_model};
use fandhe_ai_tensor_core::f16;

use crate::Device;
use crate::Tensor;
use crate::compat::Sequential;

/// [`set_cuda_onnx_gpu_execution_enabled`]（`crate::lib` の薄い公開
/// ラッパー経由）が読み書きする opt-in 状態（イシュー #2077）。既定
/// `false`（ホスト CPU 実行のみ・bit 不変）。プロセスワイドの
/// `AtomicBool`（`SeqCst`）で、既存の `set_cuda_tf32_gemm_enabled` 等と
/// 同型（`crate::lib` の opt-in 群コメント参照）。`pub(crate)` のため
/// `crates/facade/tests/api_surface.rs::scan_unapproved_onnx_pub_items`
/// の allowlist 変更は不要。
pub(crate) static CUDA_ONNX_GPU_EXEC: AtomicBool = AtomicBool::new(false);

/// Metal 版の opt-in 状態（macOS 限定。`crate::Device::Metal` と同じ cfg
/// 境界）。
#[cfg(target_os = "macos")]
pub(crate) static METAL_ONNX_GPU_EXEC: AtomicBool = AtomicBool::new(false);

pub(crate) fn set_cuda_onnx_gpu_execution_enabled(enabled: bool) {
    CUDA_ONNX_GPU_EXEC.store(enabled, Ordering::SeqCst);
}

pub(crate) fn cuda_onnx_gpu_execution_enabled() -> bool {
    CUDA_ONNX_GPU_EXEC.load(Ordering::SeqCst)
}

#[cfg(target_os = "macos")]
pub(crate) fn set_metal_onnx_gpu_execution_enabled(enabled: bool) {
    METAL_ONNX_GPU_EXEC.store(enabled, Ordering::SeqCst);
}

#[cfg(target_os = "macos")]
pub(crate) fn metal_onnx_gpu_execution_enabled() -> bool {
    METAL_ONNX_GPU_EXEC.load(Ordering::SeqCst)
}

/// [`OnnxModel::run`] の Metal 分岐専用ヘルパ。`Device::Metal` variant
/// 自体が `#[cfg(target_os = "macos")]` 限定のため、cfg 境界を `run` 本体
/// から本関数へ隠蔽する（非 macOS ビルドでは opt-in フラグの値に関わらず
/// 常に `Ok(None)` = ホストへ）。opt-in が有効なのに driver 不在等で
/// `resolve_ops` が失敗した場合は fail-closed に `Err` を返す（ホストへの
/// 黙示フォールバックはしない）。
#[cfg(target_os = "macos")]
fn resolve_metal_ops_if_enabled()
-> Result<Option<Box<dyn fandhe_ai_tensor_core::BackendOps + Send>>, OnnxError> {
    if !metal_onnx_gpu_execution_enabled() {
        return Ok(None);
    }
    let ops = crate::resolve_ops(Device::Metal).map_err(|e| OnnxError::Execution {
        message: e.to_string(),
    })?;
    Ok(Some(ops))
}

#[cfg(not(target_os = "macos"))]
fn resolve_metal_ops_if_enabled()
-> Result<Option<Box<dyn fandhe_ai_tensor_core::BackendOps + Send>>, OnnxError> {
    Ok(None)
}

/// 読み込み済み ONNX モデル（内部的にはトポロジカル順検証済みの
/// `Graph` を保持する。フィールドは private——`fandhe_ai_onnx_interop`
/// の内部型〈`NodeProto` 等〉を facade の公開シグネチャへ出さないため）。
#[derive(Debug)]
pub struct OnnxModel {
    graph: Graph,
}

impl OnnxModel {
    /// `.onnx` ファイルのバイト列からモデルを構築する。
    ///
    /// protobuf デコード（[`OnnxError::Decode`]）→ 内部グラフ構築
    /// （形状・トポロジ検証。該当する `OnnxError` variant）の順で検証する。
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, OnnxError> {
        let model = decode_model(bytes).map_err(map_decode_error)?;
        let graph = build_graph(&model).map_err(map_graph_error)?;
        Ok(Self { graph })
    }

    /// ファイルパスから `.onnx` モデルを読み込む（`std::fs::read` →
    /// protobuf デコード → [`fandhe_ai_onnx_interop::onnx::external_data::
    /// build_graph_with_external_data`]。パスをシェル展開・連結せず
    /// そのまま渡す）。
    ///
    /// **external data 対応**（イシュー #2347・2026-09-28 ユーザー承認）:
    /// [`OnnxModel::from_bytes`] と異なり、`data_location = EXTERNAL` の
    /// initializer／Constant 属性テンソル（PyTorch の既定 exporter
    /// `torch.onnx.export(..., dynamo=True)` が出力する companion
    /// `.onnx.data` ファイル参照）を解決して読み込める。解決の基点
    /// ディレクトリ（`base_dir`）は `path` の親ディレクトリ（`path` が
    /// カレントディレクトリ相対の単純なファイル名で親コンポーネントを
    /// 持たない場合は `.`）とし、既定の読み込み予算（合計サイズ上限
    /// 64 GiB・distinct ファイル数上限 4096。[`OnnxExternalDataLimits::default`]）
    /// を使う。予算を変更する場合は [`OnnxModel::from_path_with_limits`]
    /// （イシュー #2360）を使う。
    /// external data を持たないモデルは従来どおり読み込める（`.onnx`
    /// 本体のみのモデルは [`OnnxModel::from_bytes`] と同じグラフ構築
    /// 経路〈`onnx::graph::build_graph`〉へ委譲される。`build_graph_
    /// with_external_data` は external テンソルが 0 件の場合
    /// `resolve_external_data` が早期 `Ok(())` を返すため `raw_data` の
    /// 書き換えを一切行わない）。
    ///
    /// **メモリ**（PR #2348 codex P0 是正）: external data の読み込み
    /// バッファ・復号先はすべて失敗可能確保であり、確保に失敗した場合は
    /// プロセスを終了させず [`OnnxError::Io`]（`ErrorKind::OutOfMemory`）
    /// を返す。合計サイズ上限（64 GiB）は読み込む raw バイト列の予算で、
    /// 読み込み中のピークは最大でおよそ「上限 ＋ 最大テンソル 1 個分」
    /// （`docs/onnx-external-data-decision.md` 4.3 節）。本関数は既定
    /// 予算固定であり、利用可能メモリがそれより小さい環境で予算を下げる
    /// 場合は [`OnnxModel::from_path_with_limits`] を使う（イシュー
    /// #2360）。Linux の
    /// overcommit 設定等により、確保自体は成功した後のページ実コミット時に
    /// OS がプロセスを終了させる可能性は残る。
    ///
    /// [`OnnxModel::from_bytes`] は external data を非対応のまま
    /// fail-closed に拒否する（`GraphError::RawDataByteLenMismatch`。
    /// 挙動不変。`onnx::external_data` モジュール冒頭コメント「不変条件」
    /// 節・回帰テスト参照）。
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, OnnxError> {
        Self::from_path_with_limits(path, &OnnxExternalDataLimits::default())
    }

    /// [`OnnxModel::from_path`] と同じ読み込みを、external data の読み込み
    /// 予算（合計バイト上限・distinct ファイル数上限）を指定して行う
    /// （イシュー #2360）。低メモリ環境で既定の 64 GiB を下げるための
    /// facade 単独の入口。
    ///
    /// 予算の検査は確保より前（計画段階）に行われ、超過したモデルは
    /// 確保・読み込みに進まず [`OnnxError::InvalidModel`] で拒否される
    /// （合計バイト超過・ファイル数超過とも。写像は `from_path` と同じ）。
    /// 確保に失敗した場合は従来どおり [`OnnxError::Io`]
    /// （`ErrorKind::OutOfMemory`）。external data を持たないモデルでは
    /// `limits` は結果に影響しない。`limits` の値は検証せずそのまま内部へ
    /// 渡す（上げる指定も素通し。上げるとピークメモリが増える）。
    pub fn from_path_with_limits(
        path: impl AsRef<Path>,
        limits: &OnnxExternalDataLimits,
    ) -> Result<Self, OnnxError> {
        let path = path.as_ref();
        let bytes = std::fs::read(path).map_err(OnnxError::Io)?;
        let model = decode_model(&bytes).map_err(map_decode_error)?;
        // デコード後は `.onnx` 本体のバイト列を参照しないため、external
        // data の読み込み（ピークメモリが最大になる区間）より前に解放する。
        drop(bytes);
        let base_dir = match path.parent() {
            Some(parent) if !parent.as_os_str().is_empty() => parent,
            _ => Path::new("."),
        };
        let graph = build_graph_with_external_data(&model, base_dir, &limits.to_internal())
            .map_err(map_graph_error)?;
        Ok(Self { graph })
    }

    /// 学習済み [`crate::compat::Sequential`] から [`OnnxModel`] を構築
    /// する（イシュー #2037。モジュール doc「`Sequential` からの
    /// export」節参照）。
    ///
    /// `fandhe_ai_onnx_interop::onnx::export_nn::graph_from_layers` への
    /// 1 段委譲のみ（薄いラッパー原則）。対応層は `Linear`／`ReLU`／
    /// `Softmax`／`LayerNorm`／`GELU`（erf 版）／`Conv2d` の
    /// 6 種（イシュー #2076・親 #2034 で拡大。`Sigmoid` は §15.7 項 5
    /// が承認保留のため対象外。`Linear`／`ReLU` のみの
    /// モデルは `predict` と bit 完全一致、それ以外を含むモデルは
    /// REQ-2 統一複合判定〈相対誤差 1e-3 未満 または 絶対誤差 1e-5
    /// 未満〉で検証する）。それ以外の層（Tanh・GeluTanh・LogSoftmax・
    /// Conv1d 等）を 1 つでも含む場合は [`Graph`] を一切構築せず
    /// [`OnnxError::UnsupportedLayer`] を返す（全層事前検証・
    /// fail-closed。`security.md` A08）。空の `Sequential` は
    /// [`OnnxError::InvalidModel`] で拒否される。
    pub fn from_sequential(model: &Sequential) -> Result<Self, OnnxError> {
        let graph = graph_from_layers(model.layers()).map_err(map_export_error)?;
        Ok(Self { graph })
    }

    /// グラフを実行する。`feeds` はグラフ入力名 → 値の対応（未対応の
    /// initializer 上書きを含む pre-IR-4 セマンティクス）。
    ///
    /// 推論専用・autograd 未接続（戻り値は [`crate::Tensor`] であり
    /// `Var` ではない）。**既定はホスト CPU 実行のみ**（`BackendOps`／
    /// `Device` 非経由。導入前と bit 完全に不変）。
    /// [`crate::set_cuda_onnx_gpu_execution_enabled`]／
    /// `crate::set_metal_onnx_gpu_execution_enabled`（macOS 限定 cfg のため非
    /// macOS ビルドでは存在せずリンク化しない。イシュー #2077）
    /// の opt-in が有効な場合のみ `BackendOps` 経由の device 実行を試みる
    /// （評価順は CUDA → Metal 固定。両方 ON なら CUDA 優先）。対象 op
    /// （`fandhe_ai_onnx_interop::onnx::interp_device` モジュール冒頭
    /// コメント参照）の f32 経路のみ device へ到達し、`Unsupported`／
    /// `ShapeMismatch` はホストへフォールバックする。device・driver 不在
    /// や範囲外 ordinal（CUDA ordinal は 0 固定）はこの `run` 呼び出し
    /// 自体を [`OnnxError::Execution`] として fail-closed に拒否する
    /// （ホストへの黙示フォールバックはしない。OWASP A08。`docs/
    /// onnx-gpu-execution-decision.md` §3.4）。
    ///
    /// **メモリ**（PR #2348 codex P0 是正）: 実行ごとに initializer を
    /// 実行時値へ複製する処理と `Constant` 属性テンソルの復号は失敗可能
    /// 確保であり、確保に失敗した場合はプロセスを終了させず
    /// [`OnnxError::Io`]（`ErrorKind::OutOfMemory`）を返す（external data
    /// 由来の巨大なテンソルを持つモデル向け。演算カーネルの出力確保は
    /// 対象外）。
    pub fn run(
        &self,
        feeds: HashMap<String, OnnxValue>,
    ) -> Result<HashMap<String, OnnxValue>, OnnxError> {
        let interp_feeds: HashMap<String, InterpValue> = feeds
            .into_iter()
            .map(|(k, v)| (k, onnx_value_to_interp(v)))
            .collect();

        let outputs = if cuda_onnx_gpu_execution_enabled() {
            let ops = crate::resolve_ops(Device::Cuda(0)).map_err(|e| OnnxError::Execution {
                message: e.to_string(),
            })?;
            interp_run_with_ops(&self.graph, interp_feeds, ops.as_ref())
                .map_err(map_interp_error)?
        } else if let Some(ops) = resolve_metal_ops_if_enabled()? {
            interp_run_with_ops(&self.graph, interp_feeds, ops.as_ref())
                .map_err(map_interp_error)?
        } else {
            interp_run(&self.graph, interp_feeds).map_err(map_interp_error)?
        };
        Ok(outputs
            .into_iter()
            .map(|(k, v)| (k, interp_value_to_onnx(v)))
            .collect())
    }

    /// 保持しているグラフを `.onnx`（protobuf）バイト列へ書き出す
    /// （イシュー #2018。roundtrip export 限定。モジュール doc「ONNX
    /// export」節参照）。
    ///
    /// `fandhe_ai_onnx_interop::onnx::export::build_model_proto`（allowlist
    /// による fail-closed 検査を含む）→ `export::try_encode_model`
    /// （`proto::encode_model` と同一バイト列の失敗可能確保版）への委譲の
    /// み。allowlist 外 op を含む場合は [`OnnxError::UnsupportedOp`] を
    /// 返す（無言 skip しない）。external data 由来の巨大なテンソルを含む
    /// モデル（[`OnnxModel::from_path`]）でバイト列の確保に失敗した場合は
    /// プロセスを終了させず [`OnnxError::Io`]（`ErrorKind::OutOfMemory`）を
    /// 返す（PR #2348 codex P0 是正）。
    pub fn to_bytes(&self, options: &OnnxExportOptions) -> Result<Vec<u8>, OnnxError> {
        let model =
            build_model_proto(&self.graph, &options.to_internal()).map_err(map_export_error)?;
        try_encode_model(&model).map_err(map_export_error)
    }

    /// [`OnnxModel::to_bytes`] の結果をファイルパスへ書き出す（`to_bytes`
    /// が成功してから `std::fs::write` する順序。export 失敗時にファイル
    /// を作成・切り詰めない）。パスをシェル展開・連結せずそのまま渡す
    /// （[`OnnxModel::from_path`] と対称）。既存ファイルは上書きされる。
    pub fn to_path(
        &self,
        path: impl AsRef<Path>,
        options: &OnnxExportOptions,
    ) -> Result<(), OnnxError> {
        let bytes = self.to_bytes(options)?;
        std::fs::write(path.as_ref(), bytes).map_err(OnnxError::Io)
    }
}

/// [`OnnxModel::to_bytes`]／[`OnnxModel::to_path`] の export オプション
/// （イシュー #2018・承認事項 1）。内部クレートの `ExportOptions`
/// （`producer_name`／`graph_name`／`opset_domain` を持つ）を承認済み 2
/// フィールドのみへ縮小した自己完結型（薄いラッパー原則）。
///
/// `#[non_exhaustive]`: 将来のフィールド追加を非破壊にするため
/// （`OnnxError` と同じ方針）。値の検証は行わない（薄いラッパー原則。
/// `ir_version`／`opset_version` はそのまま書き出される）。
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OnnxExportOptions {
    pub ir_version: i64,
    pub opset_version: i64,
}

impl Default for OnnxExportOptions {
    /// `fandhe_ai_onnx_interop::onnx::export::ExportOptions::default()` と
    /// 同じ既定値（`ir_version=8`・`opset_version=17`）。
    fn default() -> Self {
        OnnxExportOptions {
            ir_version: 8,
            opset_version: 17,
        }
    }
}

impl OnnxExportOptions {
    /// 内部クレートの `ExportOptions` へ変換する（`producer_name`／
    /// `graph_name`／`opset_domain` は内部既定値のまま。`ExportOptions`
    /// を facade 公開面へ出さないための private ヘルパ）。
    fn to_internal(&self) -> ExportOptions {
        ExportOptions {
            ir_version: self.ir_version,
            opset_version: self.opset_version,
            ..ExportOptions::default()
        }
    }
}

/// [`OnnxModel::from_path_with_limits`] の external data 読み込み予算
/// （イシュー #2360）。内部クレートの `ExternalDataOptions` を facade へ
/// 出さないための自己完結型（薄いラッパー原則・REQ-12。型名に内部型名を
/// 含めないのは `api_surface.rs` の内部型名検査との整合のため）。
///
/// - `max_total_bytes`: 読み込む raw バイト列の合計上限（既定 64 GiB）。
///   `0` は非ゼロ長の external テンソルを 1 件でも含めば拒否する
/// - `max_external_files`: 参照する distinct ファイル数の上限（既定 4096）。
///   `0` は external 参照を 1 件でも含めば拒否する
///
/// 値は検証せずそのまま渡す。上げるとピークメモリ（最悪でおよそ 2 倍。
/// `docs/onnx-external-data-decision.md` 4.3 節）が増え、確保後のページ
/// コミット時 OOM という残存リスクも消えない。`#[non_exhaustive]` のため
/// 下流では構造体リテラルで構築できず、`default()` へ代入して使う:
///
/// ```no_run
/// use fandhe_ai::interop::onnx::{OnnxExternalDataLimits, OnnxModel};
///
/// let mut limits = OnnxExternalDataLimits::default();
/// limits.max_total_bytes = 8 << 30;
/// let model = OnnxModel::from_path_with_limits("model.onnx", &limits);
/// ```
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OnnxExternalDataLimits {
    pub max_total_bytes: u64,
    pub max_external_files: usize,
}

impl Default for OnnxExternalDataLimits {
    /// 内部クレートの既定定数を直接参照する（二重管理しない）。
    fn default() -> Self {
        OnnxExternalDataLimits {
            max_total_bytes: DEFAULT_MAX_EXTERNAL_DATA_TOTAL_BYTES,
            max_external_files: DEFAULT_MAX_EXTERNAL_FILES,
        }
    }
}

impl OnnxExternalDataLimits {
    /// 内部クレートの `ExternalDataOptions` へ変換する（private ヘルパ）。
    fn to_internal(self) -> ExternalDataOptions {
        ExternalDataOptions {
            max_total_bytes: self.max_total_bytes,
            max_external_files: self.max_external_files,
        }
    }
}

/// [`OnnxModel::run`] が受け付ける／返す実行時値。ONNX の
/// `TensorProto.data_type` のうち本クレートが対応する 4 種類に対応する
/// （`fandhe_ai_onnx_interop::onnx::interp::Value` と同じ集合。
/// 内部クレートの `Value` 自体は facade の公開面に出さない）。
#[derive(Clone, Debug)]
pub enum OnnxValue {
    F32(Tensor<f32>),
    I64(Tensor<i64>),
    Bool(Tensor<bool>),
    F16(Tensor<f16>),
}

/// `OnnxValue` → 内部 `interp::Value` への変換（move。コピー・再計算を
/// 挟まないため出力は内部クレート直接呼び出しと bit 一致する）。`pub`
/// な `From` impl にはしない——内部クレートの `interp::Value` が facade の
/// 公開 trait impl として露出するのを避けるため（private 関数に留める）。
fn onnx_value_to_interp(v: OnnxValue) -> InterpValue {
    match v {
        OnnxValue::F32(t) => InterpValue::F32(t),
        OnnxValue::I64(t) => InterpValue::I64(t),
        OnnxValue::Bool(t) => InterpValue::Bool(t),
        OnnxValue::F16(t) => InterpValue::F16(t),
    }
}

/// `interp::Value` → `OnnxValue` への逆変換（同じく move）。
fn interp_value_to_onnx(v: InterpValue) -> OnnxValue {
    match v {
        InterpValue::F32(t) => OnnxValue::F32(t),
        InterpValue::I64(t) => OnnxValue::I64(t),
        InterpValue::Bool(t) => OnnxValue::Bool(t),
        InterpValue::F16(t) => OnnxValue::F16(t),
    }
}

/// [`OnnxModel`] の失敗を表す型付きエラー。`#[non_exhaustive]`:
/// 内部クレート（`GraphError`／`InterpError`）も `#[non_exhaustive]` で
/// あり、対応 op・対応 dtype の拡張に伴う variant 追加に備える。
///
/// 設計上の判断（`docs/facade-onnx-import-exposure-decision.md` §4.2）:
/// 内部クレートのエラー enum（`GraphError`／`InterpError`）をそのまま
/// ペイロードに持たせず、facade だけで名指し・`match` できる自己完結
/// 型として定義する。承認済み公開面は `OnnxModel`／`OnnxValue`／
/// `OnnxError` の 3 型のみであり、内部エラー型をペイロードに含めると
/// 利用者がそれらを名指しできず「型付き `Err` を fail-closed に拒否
/// する」という受け入れ条件を facade 単独では満たせないため。
#[non_exhaustive]
#[derive(Debug)]
pub enum OnnxError {
    /// ファイル I/O 失敗（[`OnnxModel::from_path`] でのモデル本体読み込み
    /// 失敗・external data 解決中の companion `.onnx.data` ファイルの
    /// 欠落／権限エラー等〈`ExternalDataError::Io`。イシュー #2347〉、
    /// external data の読み込みバッファ・復号先のメモリ確保失敗
    /// 〈`ExternalDataError::AllocationFailed`。`ErrorKind::OutOfMemory`〉、
    /// [`OnnxModel::run`] での initializer の実行時値・`Constant` 属性
    /// テンソルの確保失敗〈`InterpError::AllocationFailed`。同〉、
    /// [`OnnxModel::to_bytes`]／[`OnnxModel::to_path`] での export 用
    /// バイト列の確保失敗〈`ExportError::AllocationFailed`。同〉、
    /// または [`OnnxModel::to_path`] での書き込み失敗）。これらを区別する
    /// 専用 variant は設けず（薄いラッパー原則。`std::io::Error` 自体は
    /// 操作の別を保持しない）、[`fmt::Display`] 側で「I/O 失敗」と中立に
    /// 表現する。
    ///
    /// external data 由来の OS 起因の失敗では `raw_os_error()` が OS の
    /// エラーコード（unix は errno、Windows は Win32 エラーコード）を返し、
    /// 共有違反とアクセス拒否のような原因を区別できる（イシュー #2488）。
    /// OS コードを持たない合成エラー（読み込み不足の `UnexpectedEof`・確保
    /// 失敗の `OutOfMemory` 等）では `None`。
    Io(std::io::Error),
    /// protobuf デコード失敗（壊れたバイト列等）。`prost::DecodeError` は
    /// `Display` 文字列のみを保持する（`prost` 型を公開面に出さない）。
    Decode { message: String },
    /// テンソルの `data_type` が本クレートの対応範囲外（`GraphError::
    /// UnknownDataType`。initializer decode 経由・`Constant` 属性テンソル
    /// decode 経由〈`InterpError::Graph(GraphError::UnknownDataType)`〉の
    /// 両方をこの variant へ写像する）。
    UnsupportedDataType { tensor_name: String, data_type: i32 },
    /// `GraphProto.sparse_initializer` が非空（`GraphError::
    /// SparseInitializerNotSupported`）。sparse テンソルは非対応のため
    /// fail-closed に拒否する（イシュー #2079）。
    SparseInitializerNotSupported { tensor_name: String, count: usize },
    /// 未対応の `op_type`（`InterpError::UnsupportedOp`。import 実行時）、
    /// または export 時の allowlist 外 op（`ExportError::UnsupportedOp`。
    /// `op_type` が既定 domain 以外の場合は `"{domain}::{op_type}"`
    /// 形式で domain を含める。情報を落とさないための写像判断）。
    UnsupportedOp { op_type: String },
    /// `run` の呼び出し元がグラフ入力に対応する feed を渡さなかった
    /// （`InterpError::MissingFeed`）。
    MissingFeed { input: String },
    /// `run` に渡された feed 名がグラフ入力にも initializer にも属さない
    /// （`InterpError::UnknownFeed`）。
    UnknownFeed { name: String },
    /// [`OnnxModel::from_sequential`] が対応しない層（`Linear`／`ReLU`
    /// 以外）を含む場合の拒否（`ExportError::UnsupportedLayer`。イシュー
    /// #2037・承認事項 4）。`layer_kind` は `Module` trait の既存
    /// ダウンキャストフックで判別できる範囲のみ具体名を報告し、それ
    /// 以外は `"unknown"`（`export_nn` モジュール doc 参照）。
    UnsupportedLayer { index: usize, layer_kind: String },
    /// 上記以外のモデル構築時エラー（トポロジ矛盾・形状不正・名前重複
    /// 等。`GraphError` の `Display` 文字列を保持する）、または上記以外
    /// の export 時エラー（`ExportError` の `Display` 文字列を保持する。
    /// `check_exportable` 済みの `Graph` からは通常到達しないが
    /// `ExportError` は `#[non_exhaustive]` のため fallback として写像
    /// する）。
    InvalidModel { message: String },
    /// 上記以外の実行時エラー（型不一致・属性欠落・未対応 dtype の演算
    /// 等。`InterpError` の `Display` 文字列を保持する）。
    Execution { message: String },
}

impl fmt::Display for OnnxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OnnxError::Io(e) => write!(f, "ONNX ファイル I/O 失敗（読み込みまたは書き込み）: {e}"),
            OnnxError::Decode { message } => write!(f, "ONNX protobuf デコード失敗: {message}"),
            OnnxError::UnsupportedDataType {
                tensor_name,
                data_type,
            } => write!(
                f,
                "未対応の ONNX data_type（tensor={tensor_name}）: {data_type}"
            ),
            OnnxError::SparseInitializerNotSupported { tensor_name, count } => write!(
                f,
                "未対応の ONNX sparse_initializer（tensor={tensor_name}・count={count}）: sparse テンソルは非対応"
            ),
            OnnxError::UnsupportedOp { op_type } => write!(f, "未対応の ONNX op_type: {op_type}"),
            OnnxError::MissingFeed { input } => {
                write!(f, "グラフ入力 '{input}' に対応する feed がありません")
            }
            OnnxError::UnknownFeed { name } => {
                write!(
                    f,
                    "feed '{name}' はグラフ入力にも initializer にも属しません"
                )
            }
            OnnxError::UnsupportedLayer { index, layer_kind } => {
                write!(f, "未対応の層（index={index}）: {layer_kind}")
            }
            OnnxError::InvalidModel { message } => write!(f, "不正な ONNX モデル: {message}"),
            OnnxError::Execution { message } => write!(f, "ONNX 実行時エラー: {message}"),
        }
    }
}

impl std::error::Error for OnnxError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            OnnxError::Io(e) => Some(e),
            _ => None,
        }
    }
}

/// `DecodeModelError` → `OnnxError` 写像（`decode_model` のエラー経路。
/// イシュー #2079 codex-review 是正）。`sparse_initializer` の bounded
/// 事前走査による早期拒否（[`DecodeModelError::SparseInitializerNotSupported`]）
/// を `map_graph_error` の同名分岐と同じ payload で `OnnxError::
/// SparseInitializerNotSupported` へ写像することで、拒否が
/// `decode_model` 側・`build_graph` 側のどちらで起きても facade 利用者
/// から見た結果が同一になるようにする。
fn map_decode_error(e: DecodeModelError) -> OnnxError {
    match e {
        DecodeModelError::Wire(err) => OnnxError::Decode {
            message: err.to_string(),
        },
        DecodeModelError::SparseInitializerNotSupported { tensor_name, count } => {
            OnnxError::SparseInitializerNotSupported { tensor_name, count }
        }
    }
}

/// `GraphError` → `OnnxError` 写像（モデル構築時のエラー経路）。
fn map_graph_error(e: GraphError) -> OnnxError {
    match e {
        GraphError::UnknownDataType {
            tensor_name,
            data_type,
        } => OnnxError::UnsupportedDataType {
            tensor_name,
            data_type,
        },
        GraphError::SparseInitializerNotSupported { tensor_name, count } => {
            OnnxError::SparseInitializerNotSupported { tensor_name, count }
        }
        // external data 解決中（`OnnxModel::from_path`）の I/O エラー
        // （companion `.onnx.data` ファイルの欠落・権限エラー等）は、
        // 利用者が型で判別できるよう既存の `OnnxError::Io`（`std::fs::read`
        // 失敗と同じ variant。新規 variant は追加しない）へ写像する
        // （レビュー対応。以前は他の `ExternalDataError` 同様
        // `InvalidModel` へ畳み込まれ I/O 失敗と判別できなかった）。
        // `tensor_name` は `std::io::Error` に保持できないため落ちる
        // （`OnnxError::Io` は `std::fs::read` の I/O 失敗も同じ理由で
        // メッセージ以外のコンテキストを持たない設計であり整合する）。
        // OS 由来の失敗は `raw_os_error` から `from_raw_os_error` で復元し、
        // 利用者が `raw_os_error()` で OS コードを取得できるようにする
        // （イシュー #2488）。std には kind と OS コードを同時に持たせる
        // コンストラクタがないが、`from_raw_os_error` の kind は std が
        // コードから導出し、元の OS 由来 `io::Error` の `kind()` も同じ
        // コードから導出されるため、同一プラットフォームでは kind は一致する。
        // OS コードを持たない合成エラー（`UnexpectedEof` 等）は kind のみ。
        GraphError::ExternalData(ExternalDataError::Io {
            kind, raw_os_error, ..
        }) => OnnxError::Io(match raw_os_error {
            Some(code) => std::io::Error::from_raw_os_error(code),
            None => std::io::Error::from(kind),
        }),
        // external data の読み込みバッファ・復号先の確保失敗（PR #2348
        // codex P0 是正で abort から型付きエラーへ変更）は、既存の
        // `OnnxError::Io`（`ErrorKind::OutOfMemory`）へ写像する（新規
        // variant は追加しない）。同じ `from_path` 内の `std::fs::read` が
        // `.onnx` 本体の確保失敗を std の `try_reserve` 規約どおり
        // `Io(OutOfMemory)` で返すため、どちらの確保失敗も利用者が
        // `kind() == OutOfMemory` で一様に判別できる。`InvalidModel` へ
        // 畳み込むと資源不足を「モデル不正」と誤分類するため採らない。
        // `tensor_name`・`bytes` は上の `Io` 写像と同じ理由で落ちる。
        GraphError::ExternalData(ExternalDataError::AllocationFailed { .. }) => {
            OnnxError::Io(std::io::Error::from(std::io::ErrorKind::OutOfMemory))
        }
        other => OnnxError::InvalidModel {
            message: other.to_string(),
        },
    }
}

/// `InterpError` → `OnnxError` 写像（`run` 実行時のエラー経路）。
fn map_interp_error(e: InterpError) -> OnnxError {
    match e {
        InterpError::UnsupportedOp(op_type) => OnnxError::UnsupportedOp { op_type },
        InterpError::MissingFeed { input } => OnnxError::MissingFeed { input },
        InterpError::UnknownFeed { name } => OnnxError::UnknownFeed { name },
        // `Constant` 属性テンソルの decode 失敗（`GraphError` 由来）も
        // モデル構築時と同じ写像規則（未対応 dtype は名指し・それ以外は
        // InvalidModel）を適用する。
        InterpError::Graph(graph_err) => map_graph_error(graph_err),
        // 実行時値（initializer の複製・`Constant` 属性テンソルの復号）の
        // 確保失敗（PR #2348 codex P0 是正で abort から型付きエラーへ変更）
        // は、読み込み時の確保失敗（`map_graph_error` の
        // `ExternalDataError::AllocationFailed` 分岐）と同じ既存の
        // `OnnxError::Io(ErrorKind::OutOfMemory)` へ写像する（新規 variant
        // なし）。`Execution { message }` へ畳み込むと利用者が資源不足を
        // 型で判別できず、読み込み時と実行時で判別方法が分かれるため採らない。
        InterpError::AllocationFailed { .. } => {
            OnnxError::Io(std::io::Error::from(std::io::ErrorKind::OutOfMemory))
        }
        other => OnnxError::Execution {
            message: other.to_string(),
        },
    }
}

/// `ExportError` → `OnnxError` 写像（`OnnxModel::to_bytes`／`to_path`・
/// `OnnxModel::from_sequential` の export 時エラー経路）。承認済みの
/// 新規 variant は `UnsupportedLayer`（イシュー #2037・承認事項 4）の
/// 1 件のみ——`EmptyModel`／`InvalidLayerParameter`／
/// `DuplicateTensorName`（`from_sequential` 経由でのみ到達しうる）は
/// 既存 fallback（`InvalidModel { message }`）へ写像し、承認済み公開面
/// 〈#2018／#2037 承認事項〉の範囲に留める。
fn map_export_error(e: ExportError) -> OnnxError {
    match e {
        ExportError::UnsupportedOp {
            op_type, domain, ..
        } => {
            // 既定 domain（空文字列）以外は `"{domain}::{op_type}"` 形式で
            // domain 情報を落とさずに `op_type` へ畳み込む（`OnnxError`
            // へ新規 variant を追加しない設計上の制約下での情報保持）。
            let op_type = if domain.is_empty() {
                op_type
            } else {
                format!("{domain}::{op_type}")
            };
            OnnxError::UnsupportedOp { op_type }
        }
        ExportError::UnsupportedLayer { index, layer_kind } => OnnxError::UnsupportedLayer {
            index,
            layer_kind: layer_kind.to_string(),
        },
        // export 用バイト列の確保失敗（PR #2348 codex P0 是正）は、読み込み・
        // 実行時の確保失敗と同じ既存の `OnnxError::Io(OutOfMemory)` へ写像する
        // （新規 variant なし。`InvalidModel` へ畳み込むと資源不足を「モデル
        // 不正」と誤分類するため採らない）。
        ExportError::AllocationFailed { .. } => {
            OnnxError::Io(std::io::Error::from(std::io::ErrorKind::OutOfMemory))
        }
        other => OnnxError::InvalidModel {
            message: other.to_string(),
        },
    }
}

/// `map_graph_error` の external data 分岐と、実行時・export 時の確保失敗
/// （`map_interp_error`／`map_export_error`）の写像単体テスト（実確保・実
/// ファイル I/O を伴わずに写像規則だけを固定する）。
#[cfg(test)]
mod map_graph_error_tests {
    use super::{
        ExportError, ExternalDataError, GraphError, InterpError, OnnxError, map_export_error,
        map_graph_error, map_interp_error,
    };

    fn assert_out_of_memory(e: OnnxError) {
        match e {
            OnnxError::Io(io) => assert_eq!(io.kind(), std::io::ErrorKind::OutOfMemory),
            other => panic!("OnnxError::Io(OutOfMemory) を期待したが {other:?}"),
        }
    }

    /// 実行時値の確保失敗（`InterpError::AllocationFailed`。PR #2348 codex P0
    /// 是正）は読み込み時と同じ `Io(OutOfMemory)` へ写像される。
    #[test]
    fn interp_allocation_failed_maps_to_io_out_of_memory() {
        assert_out_of_memory(map_interp_error(InterpError::AllocationFailed {
            tensor_name: "w".to_string(),
            bytes: 1 << 40,
        }));
    }

    /// export 用バイト列の確保失敗（`ExportError::AllocationFailed`）も同じ
    /// `Io(OutOfMemory)` へ写像される。
    #[test]
    fn export_allocation_failed_maps_to_io_out_of_memory() {
        assert_out_of_memory(map_export_error(ExportError::AllocationFailed {
            tensor_name: "w".to_string(),
            bytes: 1 << 40,
        }));
    }

    /// distinct ファイル数上限超過（イシュー #2360）は既存の catch-all で
    /// `InvalidModel` へ写像される（新 variant なし）。
    #[test]
    fn too_many_external_files_maps_to_invalid_model() {
        let e = map_graph_error(GraphError::ExternalData(
            ExternalDataError::TooManyExternalFiles { limit: 1 },
        ));
        match e {
            OnnxError::InvalidModel { message } => assert!(message.contains("limit=1")),
            other => panic!("OnnxError::InvalidModel を期待したが {other:?}"),
        }
    }

    /// 確保失敗（PR #2348 codex P0 是正）は `InvalidModel` ではなく
    /// `Io(ErrorKind::OutOfMemory)` へ写像される（`std::fs::read` の確保
    /// 失敗と同じ判別方法を利用者へ提供する）。
    #[test]
    fn allocation_failed_maps_to_io_out_of_memory() {
        let e = map_graph_error(GraphError::ExternalData(
            ExternalDataError::AllocationFailed {
                tensor_name: "w".to_string(),
                bytes: 1 << 40,
            },
        ));
        match e {
            OnnxError::Io(io) => assert_eq!(io.kind(), std::io::ErrorKind::OutOfMemory),
            other => panic!("OnnxError::Io(OutOfMemory) を期待したが {other:?}"),
        }
    }

    /// 既存の `ExternalDataError::Io` 写像（kind 保持）は不変。
    #[test]
    fn external_io_error_keeps_kind() {
        let e = map_graph_error(GraphError::ExternalData(ExternalDataError::Io {
            tensor_name: "w".to_string(),
            kind: std::io::ErrorKind::NotFound,
            raw_os_error: None,
        }));
        match e {
            OnnxError::Io(io) => assert_eq!(io.kind(), std::io::ErrorKind::NotFound),
            other => panic!("OnnxError::Io(NotFound) を期待したが {other:?}"),
        }
    }

    /// OS 由来の `ExternalDataError::Io` は OS コードと kind を保持して
    /// `OnnxError::Io` へ写る（イシュー #2488）。
    #[test]
    fn external_io_error_keeps_raw_os_error() {
        let missing = std::env::temp_dir().join("fandhe-ai-2488-missing-file.data");
        let os_err = std::fs::File::open(&missing).expect_err("存在しないパスのはず");
        let code = os_err.raw_os_error().expect("OS 由来のエラーのはず");
        let e = map_graph_error(GraphError::ExternalData(ExternalDataError::Io {
            tensor_name: "w".to_string(),
            kind: os_err.kind(),
            raw_os_error: Some(code),
        }));
        match e {
            OnnxError::Io(io) => {
                assert_eq!(io.raw_os_error(), Some(code));
                assert_eq!(io.kind(), os_err.kind());
            }
            other => panic!("OnnxError::Io を期待したが {other:?}"),
        }
    }

    /// それ以外の external data エラーは従来どおり `InvalidModel`。
    #[test]
    fn other_external_errors_map_to_invalid_model() {
        let e = map_graph_error(GraphError::ExternalData(
            ExternalDataError::TotalSizeLimitExceeded {
                limit: 8,
                requested: 16,
            },
        ));
        assert!(matches!(e, OnnxError::InvalidModel { .. }), "{e:?}");
    }
}
