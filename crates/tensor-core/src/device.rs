//! バックエンド抽象層のデバイス列挙・選択の共通入口（TASK-1.9a・#44）。
//!
//! `backend-cpu`／`backend-cuda`／`backend-metal` はいずれも `tensor-core` に
//! 依存する（`crates/backend-cpu/Cargo.toml` 等）。3 バックエンドが「同一
//! trait でデバイス列挙・選択できる」（#44 受け入れ条件）ためには、共通の
//! trait 定義をこれら 3 クレートすべてが依存できる本クレートに置き、各
//! バックエンドクレート側で実装する**依存逆転**構成を取る必要がある
//! （`tensor-core` → `backend-*` の逆依存は作らない）。
//!
//! シグネチャは `docs/public-api-design.md` §4.1（`Device`）・§4.4
//! （`BackendError`）を正本としつつ、以下の点で同文書から拡張している
//! （同文書「TASK-1.9 実装イシューで本文書との突合を行うこと」に対応。
//! 突合結果は `docs/public-api-design.md` にも注記する）:
//!
//! - [`DeviceProvider`] trait・[`DeviceInfo`]・[`enumerate_all`]／
//!   [`select_from`] ヘルパーを新規追加した。§4.1 の `Device::available()`
//!   は `tensor-core` から 3 バックエンドを直接参照できないため本クレート
//!   では実装せず、`enumerate_all` を同等機能として提供する。集約入口
//!   （`Device::available()` をどの層で結線するか）は TASK-1.9c／1.9d
//!   （#46／#47）へ引き継ぐ。
//! - `BackendError` に §4.4 未記載の `DeviceUnavailable` variant を追加した
//!   （不在デバイス・範囲外 ordinal の選択失敗を表す）。`BackendError` は
//!   `#[non_exhaustive]` であり、§4.4 のコメントが「TASK-1.9 実装が進むに
//!   つれ想定される実行時失敗の variant 追加を非破壊にするため」と明記して
//!   おり、本追加はこれに整合する。
//! - 既定デバイス選択ロジック（§4.1「既定選択の方針（未決事項）」・§6-2）は
//!   実装しない。CUDA 既定有効化の構成決定はユーザー承認必須のため、本
//!   イシューは列挙と明示選択のみを提供する。
//!
//! `DeviceBuffer`／`upload`／`download`（§4.2）は本イシューのスコープ外
//! （TASK-1.9b・#45）。`BackendOps` カーネルディスパッチは
//! `crate::backend_ops`（TASK-1.9c・#46）を参照。

use std::fmt;

use crate::error::ShapeError;

/// 実行先デバイス（`docs/public-api-design.md` §4.1 のシグネチャをそのまま
/// 実装）。
///
/// `Metal` variant は `cfg(target_os = "macos")` 限定とし、feature フラグは
/// 設けない（PoC-v2-5 実証構成。`.claude/rules/coding-rust.md`）。`Cuda`
/// variant は `cudarc` が無条件依存＋動的ロード方式であるため、全 OS で
/// variant 自体は存在する（実行時に CUDA ドライバ不在なら
/// [`BackendError::CudaUnavailable`] を返す。実装は `backend-cuda` 側）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Device {
    /// CPU バックエンド（`backend-cpu`）。常に利用可能。
    Cpu,
    /// CUDA バックエンド（`backend-cuda`）。`usize` は `cudarc` の
    /// デバイス ordinal（`CudaContext::new(ordinal)` に対応）。
    Cuda(usize),
    /// Metal バックエンド（`backend-metal`）。macOS 限定
    /// （`deps-policy.md` の `objc2`／`objc2-foundation`／`objc2-metal`
    /// と同じ cfg 境界）。
    #[cfg(target_os = "macos")]
    Metal,
}

/// デバイスのプロパティ。取得可否がバックエンドごとに異なる項目
/// （メモリ容量・演算ユニット数）は `Option` とし、取得できない場合
/// （API 非対応・取得失敗）は `None` を返す設計とする。
///
/// `#[non_exhaustive]` を付す理由: 公開 API 非破壊はガードレール条件
/// （`.claude/rules/security.md`）であり、後続タスク（1.9b 以降）で
/// プロパティ項目が増えても呼び出し側の網羅的フィールドアクセスを
/// 破壊しないため（構築はこのモジュール内の関数のみが行う）。
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    /// このプロパティが属するデバイス。
    pub device: Device,
    /// デバイス名（例: `"cpu"` ／ CUDA デバイス名 ／ `MTLDevice.name`）。
    pub name: String,
    /// デバイスの総メモリ容量（バイト）。CPU は `None`
    /// （ホストメモリ容量の取得は本イシューのスコープ外）。
    pub total_memory_bytes: Option<u64>,
    /// 演算ユニット数（CPU: 論理コア数 ／ CUDA: SM 数）。取得できない
    /// 場合は `None`。
    pub compute_units: Option<u32>,
}

impl DeviceInfo {
    /// 新規 `DeviceInfo` を構築する。バックエンド実装（`CpuDeviceProvider`
    /// 等）の `enumerate`／`select` から呼ばれる。`#[non_exhaustive]` の
    /// ため構造体リテラルで直接構築できないバックエンドクレート向けの
    /// 構築入口。
    pub fn new(
        device: Device,
        name: impl Into<String>,
        total_memory_bytes: Option<u64>,
        compute_units: Option<u32>,
    ) -> Self {
        Self {
            device,
            name: name.into(),
            total_memory_bytes,
            compute_units,
        }
    }
}

/// 各バックエンド（CPU／CUDA／Metal）が実装するデバイス検出・選択の入口。
///
/// object-safe に設計している（`&dyn DeviceProvider` として扱える。
/// `enumerate_all`／`select_from` が複数バックエンドを横断する際に
/// 使用し、TASK-1.9c のディスパッチ機構・TASK-1.9d の統合テストが
/// `dyn` 経由で束ねる前提を担保する）。
pub trait DeviceProvider {
    /// バックエンド名（`"cpu"`／`"cuda"`／`"metal"`）。ログ・エラー
    /// メッセージでの識別用。
    fn backend_name(&self) -> &'static str;

    /// このバックエンドが実行環境で利用可能か（CUDA: ドライバ検出成功、
    /// Metal: デバイス 1 件以上検出、CPU: 常に `true`）。
    fn is_available(&self) -> bool;

    /// このバックエンドで利用可能なデバイスを列挙する。バックエンド不在
    /// （CUDA ドライバ未検出・Metal デバイスなし）でも `Err` にせず
    /// `Ok(vec![])` を返す（fail-safe。本番経路で `panic!`／`unwrap()`
    /// しない。`.claude/rules/coding-rust.md`）。
    fn enumerate(&self) -> Result<Vec<DeviceInfo>, BackendError>;

    /// 指定した [`Device`] を選択しプロパティを取得する。存在しない
    /// デバイス・範囲外 ordinal の場合は
    /// [`BackendError::DeviceUnavailable`] を返す。
    fn select(&self, device: Device) -> Result<DeviceInfo, BackendError>;
}

/// 複数の [`DeviceProvider`] を横断して利用可能なデバイスをすべて列挙する
/// （`docs/public-api-design.md` §4.1 `Device::available()` の同等機能。
/// `tensor-core` は 3 バックエンドクレートを直接参照できないため、
/// 呼び出し側（結線を担う上位クレート・テスト）が `providers` を注入する
/// 依存逆転構成を取る）。
///
/// 個々の provider の `enumerate` が `Err` を返した場合は、その provider
/// 分をスキップし他 provider の列挙結果を返す（1 バックエンドの異常が
/// 全体の列挙を止めない fail-safe 方針）。
pub fn enumerate_all(providers: &[&dyn DeviceProvider]) -> Vec<DeviceInfo> {
    providers
        .iter()
        .filter_map(|provider| provider.enumerate().ok())
        .flatten()
        .collect()
}

/// 複数の [`DeviceProvider`] を横断して `device` に一致するデバイスを選択
/// する。`device` の variant（[`Device::Cpu`]／[`Device::Cuda`]／
/// `Device::Metal`）に応じて対応する provider（`backend_name` が
/// `"cpu"`／`"cuda"`／`"metal"`）の `select` を呼び出す。対応する provider
/// が `providers` に含まれない場合は
/// [`BackendError::DeviceUnavailable`] を返す。
pub fn select_from(
    providers: &[&dyn DeviceProvider],
    device: Device,
) -> Result<DeviceInfo, BackendError> {
    let backend_name = match device {
        Device::Cpu => "cpu",
        Device::Cuda(_) => "cuda",
        #[cfg(target_os = "macos")]
        Device::Metal => "metal",
    };
    providers
        .iter()
        .find(|provider| provider.backend_name() == backend_name)
        .ok_or_else(|| {
            BackendError::DeviceUnavailable(format!(
                "no DeviceProvider registered for backend \"{backend_name}\""
            ))
        })?
        .select(device)
}

/// バックエンド抽象層のエラー型（`docs/public-api-design.md` §4.4 を実装。
/// `DeviceUnavailable` variant はモジュール冒頭に記載のとおり本イシューで
/// 追加した拡張）。
///
/// `#[non_exhaustive]` を付す理由: 公開 API 非破壊はガードレール条件
/// （`.claude/rules/security.md`）であり、CUDA／Metal 実装（TASK-1.9 以降）
/// が進むにつれ想定される実行時失敗（同期エラー・ドライババージョン
/// 不整合等）の variant 追加を非破壊にするため。
#[non_exhaustive]
#[derive(Debug)]
pub enum BackendError {
    /// `cudarc` の動的ロードに失敗した（CUDA ドライバ・toolkit 不在等）。
    CudaUnavailable(String),
    /// 入力テンソルの shape が演算の要求と合わない。
    ShapeMismatch(ShapeError),
    /// デバイス間でテンソルが混在している等の不整合。
    DeviceMismatch,
    /// デバイスメモリの確保に失敗した（`upload`／`DeviceBuffer` 確保等。
    /// VRAM 枯渇・アロケータ失敗を含む）。
    DeviceAllocationFailed(String),
    /// カーネル起動に失敗した（CUDA NVRTC のコンパイル・起動エラー、
    /// Metal `MTLComputeCommandEncoder` のディスパッチ失敗等）。
    KernelLaunchFailed(String),
    /// 指定した [`Device`] が選択時点で利用不可（存在しない ordinal・
    /// 対応する `DeviceProvider` 未登録等）。本イシュー（TASK-1.9a）で
    /// 追加した拡張 variant（モジュール冒頭のコメント参照）。
    DeviceUnavailable(String),
    /// ホスト⇔デバイス転送（`buffer::MemoryOps::upload`/`download`）が
    /// 失敗した。TASK-1.9b（#45）で追加した拡張 variant。
    ///
    /// `DeviceAllocationFailed` はバッファ確保自体の失敗（VRAM 枯渇等）を
    /// 表すのに対し、本 variant は確保済みバッファに対するコピー
    /// （CUDA `cuMemcpyHtoD`/`cuMemcpyDtoH`、Metal readback 等）の失敗を
    /// 表す（両者を区別することで呼び出し元が「確保からやり直すべきか・
    /// 転送のみ再試行すべきか」を判別できる）。`#[non_exhaustive]` の
    /// 非破壊拡張として追加した（モジュール冒頭コメント参照）。
    TransferFailed(String),
    /// 指定したバックエンドが当該演算のカーネルを未実装であることを表す
    /// （TASK-1.9c・#46 で追加した拡張 variant）。CUDA／Metal は本イシュー
    /// 時点で GEMM カーネルのみ実装済みのため、`crate::backend_ops::BackendOps`
    /// の elementwise・reduction メソッドはこの variant を返す
    /// fail-safe 実装とする（`panic!`／`unwrap()` しない。
    /// `.claude/rules/coding-rust.md`）。GPU 側カーネルの実装自体は本
    /// イシューのスコープ外（out-of-scope-tracking.md 対象）。
    Unsupported(String),
    /// デバイス上パラメータ更新（`fandhe_ai_autodiff::optim::device_store::
    /// DeviceParamStore`。イシュー #935・`docs/device-resident-update-design.md`
    /// §3.1）が「以前の `step()` 失敗により実行時エラー後の状態」で
    /// 呼ばれたことを表す。`DeviceParamStore` は `sgd_step_device` の
    /// 実行時エラー（GPU 側の実際の起動失敗等。件数・shape 不一致のような
    /// 事前検証で弾ける契約違反とは別）を検出すると内部状態を poisoned へ
    /// 遷移し、以降の `step`／`sync_to_host`／`register_resident_leaves`／
    /// `snapshot_resident_leaves` をすべてこの variant で fail-closed に
    /// 拒否する（「部分的に更新されたデバイス側パラメータをそのまま学習
    /// 継続・推論に使ってしまう」ことを構造的に防ぐ。`.claude/rules/
    /// security.md` A08）。回復は新しい `DeviceParamStore` の再構築のみ。
    StorePoisoned,
    /// `DeviceParamStore::step`／`sync_to_host` 等に、`DeviceParamStore`
    /// 構築時・直近の `register_resident_leaves` 呼び出しと異なる
    /// `Tape`（`TapeId` 不一致）を渡した（`fandhe_ai_autodiff::var::Var` の
    /// クロステープ検査〈`AutodiffError::TapeMismatch`〉と同種の契約を
    /// デバイス常駐パラメータ側にも課す。イシュー #935）。
    TapeMismatch,
    /// `DeviceParamStore::register_resident_leaves`（forward 用の
    /// デバイス→ホスト download・葉ノード登録）を、直前の登録が
    /// `step()` で消費（または `abandon_pending_forward()` で破棄）される
    /// 前に再度呼んだ。1 回の forward 記録に対し高々 1 回の `step()` が
    /// 対応する契約（イシュー #935・設計文書 §3.3a）を守るための
    /// fail-closed 拒否。
    PendingForwardUnconsumed,
    /// `DeviceParamStore::step` が `Gradients::get` で該当パラメータの
    /// 勾配を取得しようとしたが `Ok(None)`（loss へ未到達）だった。
    /// `fandhe_ai::compat::sequential::SequentialVars::trainable_grads` と
    /// 同じ fail-closed 方針（黙って一部パラメータの更新をスキップしない。
    /// `.claude/rules/security.md` A03）。
    MissingGradient(String),
    /// `DeviceParamStore` の構築・`step` 呼び出し引数が構造的に不正
    /// （パラメータ件数と勾配件数の不一致・momentum 構成〈velocity
    /// バッファの有無〉が前回 `step` から変化した等）。既存 variant
    /// （`ShapeMismatch`・`DeviceMismatch`）のいずれにも意味的に適合しない
    /// 引数検証失敗をここへ集約する（イシュー #935）。
    InvalidArgument(String),
}

impl fmt::Display for BackendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BackendError::CudaUnavailable(msg) => write!(f, "CUDA unavailable: {msg}"),
            BackendError::ShapeMismatch(err) => write!(f, "shape mismatch: {err}"),
            BackendError::DeviceMismatch => write!(f, "device mismatch between operands"),
            BackendError::DeviceAllocationFailed(msg) => {
                write!(f, "device allocation failed: {msg}")
            }
            BackendError::KernelLaunchFailed(msg) => write!(f, "kernel launch failed: {msg}"),
            BackendError::DeviceUnavailable(msg) => write!(f, "device unavailable: {msg}"),
            BackendError::TransferFailed(msg) => write!(f, "transfer failed: {msg}"),
            BackendError::Unsupported(msg) => write!(f, "unsupported operation: {msg}"),
            BackendError::StorePoisoned => write!(
                f,
                "device param store is poisoned after a previous step() failure; \
                 construct a new store"
            ),
            BackendError::TapeMismatch => {
                write!(f, "device param store operation used a mismatched Tape")
            }
            BackendError::PendingForwardUnconsumed => write!(
                f,
                "device param store has a pending forward registration not yet consumed by step()"
            ),
            BackendError::MissingGradient(msg) => write!(f, "missing gradient: {msg}"),
            BackendError::InvalidArgument(msg) => write!(f, "invalid argument: {msg}"),
        }
    }
}

impl std::error::Error for BackendError {}

#[cfg(test)]
mod tests {
    use super::*;

    /// テスト専用のモック `DeviceProvider`。実バックエンドに依存せず
    /// `enumerate_all`／`select_from` の成功・失敗経路を検証するために
    /// `tensor-core` 内で定義する（`backend-cpu` 等の実装は各バックエンド
    /// クレートの結合テストで別途検証する。TASK-1.9a 実装計画 §3.4）。
    struct MockProvider {
        name: &'static str,
        available: bool,
        devices: Vec<DeviceInfo>,
    }

    impl DeviceProvider for MockProvider {
        fn backend_name(&self) -> &'static str {
            self.name
        }

        fn is_available(&self) -> bool {
            self.available
        }

        fn enumerate(&self) -> Result<Vec<DeviceInfo>, BackendError> {
            if self.available {
                Ok(self.devices.clone())
            } else {
                Ok(vec![])
            }
        }

        fn select(&self, device: Device) -> Result<DeviceInfo, BackendError> {
            self.devices
                .iter()
                .find(|info| info.device == device)
                .cloned()
                .ok_or_else(|| {
                    BackendError::DeviceUnavailable(format!(
                        "{:?} not found on backend \"{}\"",
                        device, self.name
                    ))
                })
        }
    }

    fn cpu_provider() -> MockProvider {
        MockProvider {
            name: "cpu",
            available: true,
            devices: vec![DeviceInfo::new(Device::Cpu, "cpu", None, Some(8))],
        }
    }

    fn cuda_provider_unavailable() -> MockProvider {
        MockProvider {
            name: "cuda",
            available: false,
            devices: vec![],
        }
    }

    fn cuda_provider_available() -> MockProvider {
        MockProvider {
            name: "cuda",
            available: true,
            devices: vec![DeviceInfo::new(
                Device::Cuda(0),
                "mock-gpu-0",
                Some(1 << 30),
                Some(108),
            )],
        }
    }

    #[test]
    fn enumerate_all_merges_available_providers() {
        let cpu = cpu_provider();
        let cuda = cuda_provider_available();
        let providers: Vec<&dyn DeviceProvider> = vec![&cpu, &cuda];

        let all = enumerate_all(&providers);

        assert_eq!(all.len(), 2);
        assert!(all.iter().any(|info| info.device == Device::Cpu));
        assert!(all.iter().any(|info| info.device == Device::Cuda(0)));
    }

    #[test]
    fn enumerate_all_skips_unavailable_backend_without_error() {
        let cpu = cpu_provider();
        let cuda = cuda_provider_unavailable();
        let providers: Vec<&dyn DeviceProvider> = vec![&cpu, &cuda];

        // CUDA ドライバ不在でも enumerate_all は panic せず、CPU 分のみを
        // 返す（fail-safe。REQ-1「toolkit 非搭載環境でもビルド・実行成立」
        // の実行時側の受け皿）。
        let all = enumerate_all(&providers);

        assert_eq!(all.len(), 1);
        assert_eq!(all[0].device, Device::Cpu);
    }

    #[test]
    fn select_from_dispatches_to_matching_backend() {
        let cpu = cpu_provider();
        let cuda = cuda_provider_available();
        let providers: Vec<&dyn DeviceProvider> = vec![&cpu, &cuda];

        let selected = select_from(&providers, Device::Cuda(0)).expect("selection succeeds");

        assert_eq!(selected.device, Device::Cuda(0));
        assert_eq!(selected.name, "mock-gpu-0");
    }

    #[test]
    fn select_from_missing_backend_returns_device_unavailable() {
        let cpu = cpu_provider();
        let providers: Vec<&dyn DeviceProvider> = vec![&cpu];

        let err = select_from(&providers, Device::Cuda(0)).expect_err("no cuda provider");

        assert!(matches!(err, BackendError::DeviceUnavailable(_)));
    }

    #[test]
    fn backend_error_display_is_non_empty() {
        let errors: Vec<BackendError> = vec![
            BackendError::CudaUnavailable("no driver".into()),
            BackendError::DeviceMismatch,
            BackendError::DeviceAllocationFailed("oom".into()),
            BackendError::KernelLaunchFailed("nvrtc compile failed".into()),
            BackendError::DeviceUnavailable("ordinal 9 out of range".into()),
            BackendError::TransferFailed("clone_dtoh failed".into()),
        ];

        for err in errors {
            assert!(!err.to_string().is_empty());
        }
    }
}
