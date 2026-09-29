//! ONNX external data（initializer／Constant 属性テンソルが `.onnx` 本体の
//! 外に置く `.onnx.data` ファイル。onnx.proto3 `TensorProto.data_location`
//! （tag=14）・`external_data`（tag=13））を fail-closed に解決し、
//! `raw_data` へ inline 化する（イシュー #2347）。
//!
//! ## 背景・呼び出し文脈
//!
//! PyTorch の既定 exporter（`torch.onnx.export(..., dynamo=True)`）は
//! initializer を external data として出力する（432 バイト程度の小さな
//! テンソルでも external になる。#2329 実測）。`graph::decode_tensor` は
//! `data_location`／`external_data` を一切参照しないため（バイト列入口の
//! 契約は不変。下記「不変条件」節）、external data を持つモデルはそのまま
//! では `GraphError::RawDataByteLenMismatch`（raw_data・float_data とも
//! 空）で拒否される。本モジュールは、外部ファイルを解決する基点
//! ディレクトリ（`base_dir`）を受け取る新しい入口
//! （[`build_graph_with_external_data`]）を提供し、`ModelProto` を複製した
//! うえで external なテンソルを `raw_data` へ inline 化してから、
//! `graph::build_graph` と検証ロジックを共有する所有権ベースの
//! `graph::build_graph_owned` へ渡す（「全参照を検証 → 範囲を限定して
//! 読込 → raw_data へ inline 化 → build_graph と同一の検証でグラフ構築」
//! という設計。external テンソルが 0 件なら `graph::build_graph` へ
//! そのまま渡す。`docs/onnx-external-data-decision.md` が設計判断の正）。
//!
//! ## メモリ確保（PR #2348 codex P0 是正。security.md A04）
//!
//! external data 由来の長さから行う確保（区間の読み込みバッファ・
//! initializer の復号先）はすべて `Vec::try_reserve_exact` による失敗可能
//! 確保とし、失敗は [`ExternalDataError::AllocationFailed`] で返す
//! （`max_total_bytes` の既定値 64 GiB は利用可能メモリを超えうるため、
//! 小さな `.onnx` と疎ファイルでも巨大な単一テンソルを宣言できる。無条件
//! 確保〈`vec![..; n]`・`collect`〉は失敗時にプロセスを abort させる）。
//! `max_total_bytes` は読み込む raw バイト列の予算であり、読み込み経路の
//! ピークは最大でおよそ `max_total_bytes` ＋ 最大テンソル 1 個分
//! （`docs/onnx-external-data-decision.md` 4.3 節）。確保ヘルパ本体は
//! 読み込み後の実行経路（`interp::run` の initializer 複製・`Constant`
//! 属性テンソルの復号）・export 経路と共有する非公開モジュール
//! `onnx::fallible_alloc` にあり、本モジュールはその失敗を
//! `ExternalDataError::AllocationFailed` へ写像する（実行経路は
//! `InterpError::AllocationFailed`、export 経路は
//! `ExportError::AllocationFailed`。同節 5.）。
//!
//! ## 不変条件（A6・回帰テスト対象）
//!
//! `graph::decode_tensor` は本モジュールの追加後も
//! 1 バイトも変更しない。`onnx::proto::decode_model` → `graph::build_graph`
//! のバイト列入口は `data_location`／`external_data` を一切参照しないため、
//! external data を持つモデルをバイト列入口へ渡した場合の挙動（従来どおり
//! `RawDataByteLenMismatch` で拒否）は本モジュール導入前後で変わらない。
//!
//! ## 2 パス設計（検証と読み込みの分離。security.md A03／A04）
//!
//! 1. **パス 1（`plan`）**: ファイル内容を一切読まず、`data_location`／
//!    `external_data` のフィールド整合性・キーの重複や未知キー・
//!    `location` のパス安全性（絶対パス・`..`・symlink・base_dir 外への
//!    脱出を拒否）・`offset`／`length` の文法と範囲・期待バイト長との一致・
//!    重複区間・**外部データ合計の上限**（[`ExternalDataOptions::
//!    max_total_bytes`]。確保の前に検査する）・**distinct な external
//!    data ファイル数の上限**（[`ExternalDataOptions::
//!    max_external_files`]。合計バイト数の上限はサイズ 0 のテンソルを
//!    大量の異なる空ファイルへ分散させる入力に対しては無力なため、
//!    ファイルを開いた直後・既知ファイル集合へ登録する前に別途検査する。
//!    #2347 P0 是正・PR #2348 コードレビュー対応・PRRT_kwDOTuUCJc6mlxhy）
//!    をすべて検証する。1 件でも失敗すれば `Err` を返しファイルは一切
//!    読まない（A04 資源枯渇対策）。**パス 1 はファイルハンドルを保持
//!    しない**: 各 location を安全 open → ハンドルに対する `fstat` で
//!    `FileKey`（dev/ino）と `FileSnapshot`（ファイル長・unix では
//!    dev/ino・ctime・mtime）だけを記録 → 直ちに close する。
//! 2. **パス 2（`load`）**: パス 1 が全件成功した場合のみ、正規化済み
//!    location ごとに 1 ファイルずつ「パス 1 と同じ安全 open（同じ
//!    `base_dir` fd 起点の `openat2`／逐次 `openat(O_NOFOLLOW)`）→ 開いた
//!    ハンドルの `FileKey`・`FileSnapshot` をパス 1 の記録と完全一致で
//!    再照合 → その location を参照する全テンソルの区間だけを
//!    `read_region` で読む（`.data` ファイル全体は読まない。各区間の直前にも
//!    同じハンドルへ `fstat` を取り直して同じ照合を行う）→ close」を逐次に
//!    行う。再照合の不一致は [`ExternalDataError::FileChangedDuringLoad`]
//!    とする（fail-closed）。ただし最後の照合を通過した直後〜読み込みの
//!    間に truncate された場合は、照合ではなく `read_region` の読み込み
//!    不足（`UnexpectedEof`）として [`ExternalDataError::Io`] で拒否される
//!    （別 variant だが fail-closed。下記「TOCTOU の論拠」節）。
//!
//! ## ハンドル非保持の構成（PR #2348 codex P1 是正）
//!
//! 旧構成はパス 1 で開いた distinct ファイルのハンドルをすべて保持した
//! ままパス 2 で再利用していたため、`max_external_files` の既定値
//! （4096。ユーザー承認済みで下げない）が一般的なプロセスの fd 上限
//! （例: soft limit 1024）を上回り、小さなファイルを多数参照する
//! モデルでは `TooManyExternalFiles` に到達する前に `EMFILE` が発生して
//! 同一プロセスのほかの I/O にも影響し得た。現構成で同時に保持する fd は
//! `base_dir` のディレクトリ fd と、処理中の external data ファイル 1 つ
//! （パス 1 の `fstat` 中またはパス 2 の読み込み中）の高々 2 つ
//! （`openat2` 非対応時の逐次方式では経路途中のディレクトリ fd が一時的
//! に 1 つ加わる）であり、external data のファイル数に依存しない。
//!
//! 「単一パスでファイルごとに検証・読み込み・close を進める」構成は
//! 採らない: `length` 省略時の読み込み長は `file_len - offset` で確定する
//! ためファイルを開かずに総量上限（`max_total_bytes`）・ファイル数上限を
//! 判定できず、単一パスでは上限超過を途中まで読んでから検出する構成に
//! なるためである。検証パス（ハンドル非保持）と読み込みパスを分け、
//! 読み込みパスで再 open した直後にハンドル自身の `FileKey`・
//! `FileSnapshot` を検証パスの記録と照合する。
//!
//! **TOCTOU の論拠**: 読み込みは常に「パス 2 で安全 open し、その
//! ハンドル自身に対する `fstat` で `FileKey`〈dev, ino〉とファイル長・
//! ctime・mtime（秒＋ナノ秒。unix）がパス 1 の記録と完全一致した
//! ハンドル」からのみ行い、経路文字列を再解決しない（再 open も
//! `base_dir` fd 起点・シンボリックリンク拒否の同じ手段）。したがって
//! (1) `base_dir` 外・シンボリックリンク経由のファイルは読まない、
//! (2) 読む実体はパス 1 で当該 location について検証した実体と同一
//! （照合できる範囲。下記の残存リスク参照）で、長さもパス 1 の区間検証の
//! 前提と同一、(3) 読み込み量はパス 1 で上限検査済みの区間に有界、の 3 点を
//! 保証する。
//!
//! パス間でハンドルを保持しないことで、旧構成（ハンドル保持）には無かった
//! 差し替えの窓が生じる: パス 1 の close 後に元ファイルを unlink → 同じ
//! 長さの別ファイルを作成すると、ファイルシステムによっては inode 番号が
//! 再利用され dev/ino・長さが一致しうる。これは `base_dir` 配下の
//! ディレクトリ書き込み権（unlink／create）だけで起こせるため、dev/ino・
//! 長さの照合だけでは防げない（PR #2348 security-auditor P2-1）。そこで
//! ctime・mtime も照合する: 新しく作られた inode の ctime は作成時刻になり、
//! 同一 inode の in-place 改変（write・truncate・chmod・link 等）でも ctime
//! は更新される。ctime はユーザー空間から任意の値へ設定できない
//! （`utimensat` で mtime を書き戻す操作自体が ctime を更新する）ため、照合を
//! 通過する差し替えには「ディレクトリ書き込み権 ＋ inode 番号の再利用 ＋
//! 長さ一致 ＋ ctime 一致」、すなわち**ファイルシステム／カーネルの
//! タイムスタンプ粒度内（秒単位の粒度を持つファイルシステムでは同一秒内、
//! ナノ秒表現でもカーネルの粗いクロック刻み〈数 ms〉内）での再作成**が
//! 必要になる。この粒度内の再作成は残存リスクとして受容する
//! （`docs/onnx-external-data-decision.md` 5 節）。なお (a) パス 1 内で
//! 別ファイルが同じ dev/ino を得ても `FileKey` の併合は重複区間検出を
//! 増やす方向にしか働かず見逃しを生まない。
//!
//! 照合と読み込みの間の残存窓（旧構成と同じ）: 各区間の直前の照合を通過
//! した直後〜読み込みの間に truncate されると、`read_region` が読み込み
//! 不足を `UnexpectedEof` として返し [`ExternalDataError::Io`]（`FileChangedDuringLoad`
//! ではない別 variant）で fail-closed に拒否される。同じ窓で同じ長さの
//! まま in-place 書き換えされた場合は検出できない（旧構成のハンドル保持
//! でも同一で、`base_dir` 配下のファイルへの書き込み権を前提とする）。
//!
//!    **経路解決方式（unix 全般。イシュー #2347 是正版）**: パス 1 の
//!    ファイル解決は `base_dir` を 1 度だけディレクトリ fd として開き
//!    （`no_follow_open::open_base_dir`）、以後の全テンソルがその fd を
//!    起点に `location` を解決する。**Linux** では `openat2(2)`
//!    （`RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS | RESOLVE_NO_MAGICLINKS`。
//!    `no_follow_open::open_chain_openat2`）で経路解決全体を 1 回の
//!    システムコールとしてカーネルへアトミックに封じ込めさせる。
//!    `openat2` が未対応（`ENOSYS`／`EPERM`。古いカーネル・seccomp 等）の
//!    場合のみ、成分ごと逐次 `openat(O_NOFOLLOW)` で辿る方式
//!    （`no_follow_open::open_chain_component_walk`）へフォールバックする。
//!    **macOS・その他 unix**（`openat2` 非対応）では常に逐次方式を使う。
//!    いずれの方式も検証済みの fd をそのまま保持し経路文字列を**一切**
//!    再解決しないため（診断用のシンボリックリンク判別も、既に開いた
//!    親ディレクトリ fd を起点にした単一コンポーネントの `fstatat`
//!    〈`AT_SYMLINK_NOFOLLOW`〉のみを使う）、シンボリックリンク差し替えに
//!    よる TOCTOU 窓は構造的に生じない（P0 是正: PRRT_kwDOTuUCJc6mkUsI・
//!    PRRT_kwDOTuUCJc6mk30J。`docs/onnx-external-data-decision.md` の
//!    残タスクを解消）。
//!
//!    旧実装は `openat` のフラグ定数（`O_DIRECTORY`／`O_NOFOLLOW` 等）を
//!    OS ごとに手書きしていたが、Linux では同じ定数でも CPU アーキテク
//!    チャごとに値が異なり（例: x86 は `O_DIRECTORY=0o200000`・aarch64 は
//!    `O_DIRECTORY=0o40000`）、手書き値のまま aarch64 Linux（DGX Spark
//!    GB10）でビルドするとシンボリックリンク拒否が機能しない実装バグを
//!    生んでいた。`libc`（`.claude/rules/deps-policy.md`「OS FFI」区分。
//!    2026-09-28 ユーザー承認）の定数・`syscall` を使うことでこの
//!    プラットフォーム差異を自作せず解消する。
//!
//!    **Windows（イシュー #2349）**: std に `openat` 相当（ディレクトリ
//!    ハンドル起点の相対オープン）が無いため、`base_dir` から解決対象
//!    ファイルまでの全祖先ディレクトリのハンドルを開いたまま保持する
//!    （`win_contained_open::BaseDirHandle`。rename・削除・reparse point 化
//!    に必要な DELETE アクセスを共有モードから外して拒否し、reparse
//!    point 化に必要な「対象ディレクトリが空であること」〈NTFS〉を
//!    保持中の子ハンドルが妨げる）ことで、経路文字列を都度再解決する
//!    `CreateFileW` の限界を補う。各ハンドルは開いた直後に自身の属性
//!    照会で reparse point（シンボリックリンク・junction 等）を検出して
//!    拒否する。`FILE_SHARE_DELETE` を含めない共有モード・
//!    `FILE_FLAG_OPEN_REPARSE_POINT`（最終成分を追跡させない）を使う。
//!    さらに最終ファイルを開いた直後、そのハンドル自身が実際に指す
//!    オブジェクトの所在を `GetFinalPathNameByHandleW` で逆引きし
//!    `base_dir` 配下・想定した深さであることを検証する（属性再検査を
//!    「元に戻してから」すり抜ける flip-and-revert 型 TOCTOU 対策。
//!    2026-09-28 codex-review 是正・PR #2351。`win_contained_open` モジュール
//!    doc「TOCTOU の根拠」節 3.）。実体識別子（dev/ino 相当）は
//!    `(dwVolumeSerialNumber, nFileIndex)`（`GetFileInformationByHandle`）を
//!    使う。std の `file_index`／`volume_serial_number`
//!    （`windows_by_handle`）・`GetFinalPathNameByHandleW` 相当はいずれも
//!    1.98.1 時点で未安定化のため、kernel32.dll への手書き `extern
//!    "system"` 宣言（`win_contained_open` 内。`unsafe` を FFI 境界に限定）
//!    で直接呼ぶ。変更時刻は `change_time`（同じく未安定化）の代わりに
//!    `FileSnapshot` が `file_attributes`／`creation_time`／
//!    `last_write_time` を使う（unix の ctime ほど強い変更検知ではない。
//!    詳細・残存リスクは `docs/onnx-external-data-decision.md` 5 節）。
//!
//!    **それ以外（wasm32 等）**: `openat`／`openat2`・Windows 封じ込め
//!    オープンのいずれの安全な経路解決手段も持たないため、
//!    `resolve_and_open` は常に
//!    [`ExternalDataError::UnsupportedPlatformForSecureResolve`] で拒否
//!    する（fail-closed。`docs/onnx-external-data-decision.md` 5 節）。
//!
//! `checksum` キーは黙って無視せず [`ExternalDataError::
//! ChecksumUnsupported`] で fail-closed に拒否する（依存を追加できないため
//! SHA-1 検証は実装しない。no-silent-skip 契約。security.md A08）。
//!
//! ## facade への公開範囲
//!
//! `crate::facade::interop::onnx::OnnxModel::from_path`（本クレート外部）
//! が [`build_graph_with_external_data`] へ委譲する形で 2026-09-28 に
//! 公開済み（`docs/facade-onnx-import-exposure-decision.md` §6.3・
//! `docs/compat-api-scope.md` §5 の承認待ち事項を解消。イシュー #2347）。
//! [`resolve_external_data`]・[`ExternalDataOptions`] 自体は本モジュール
//! （`onnx-interop` 内部限定）のままであり、facade 側は薄いラッパーに徹する
//! （`from_path` は既定オプション〈[`ExternalDataOptions::default`]〉、
//! `from_path_with_limits` は facade 独自型 `OnnxExternalDataLimits` から
//! 変換して渡す。イシュー #2360。内部型は非公開のまま）。

use std::collections::HashMap;
use std::fmt;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Component, Path, PathBuf};

use super::graph::{GraphError, element_count};
use super::proto::{ModelProto, TensorProto, cap_sparse_tensor_diag_name, data_location};

/// external data の合計サイズ上限の既定値（64 GiB）。
///
/// 2026-09-28 ユーザー承認（イシュー #2347。当初の 4 GiB 暫定値から改定）。
/// 変更は本定数 1 行の書き換えで済む。`ExternalDataOptions::default()` が
/// 参照する。
pub const DEFAULT_MAX_EXTERNAL_DATA_TOTAL_BYTES: u64 = 64 * 1024 * 1024 * 1024;

/// 1 モデルあたりに参照を許す external data ファイル数（実体単位。
/// `FileKey` で畳み込んだ後の distinct 数）の既定上限。
///
/// `max_total_bytes` はバイト数のみを制限するため、要素数 0（サイズ 0）の
/// テンソルを大量に並べ、それぞれが別々の空ファイルを `location` で参照
/// するモデルに対しては合計サイズが 0 のまま歯止めが効かず、open／`fstat`
/// の回数が無制限に増えうる（security.md A04・AGENTS.md「外部フォーマット
/// のパース検証」。#2347 P0 是正・PR #2348 コードレビュー対応・
/// PRRT_kwDOTuUCJc6mlxhy）。なお本上限は同時保持 fd 数の上限ではない:
/// `plan`／`load` は external data ファイルのハンドルを 1 つずつ開いては
/// 閉じるため、同時保持 fd 数はこの値に依存せず有界（モジュール doc
/// 「ハンドル非保持の構成」節。PR #2348 codex P1 是正）であり、4096 が
/// プロセスの fd 上限（例: 1024）を上回っても `EMFILE` は生じない。この上限は
/// distinct なファイル実体（`FileKey`）の数を制限するため、1 ファイルを
/// 複数テンソルが参照する通常の分割形式（同一 `.onnx.data` を initializer
/// 群が共有する構成）は 1 件としてしか数えない。
///
/// 2026-09-28 ユーザー承認（イシュー #2347。当初の暫定値 4096 をそのまま
/// 正式な既定値として確定）。変更は本定数 1 行の書き換えで済む。
/// `ExternalDataOptions::default()` が参照する。
pub const DEFAULT_MAX_EXTERNAL_FILES: usize = 4096;

/// [`resolve_external_data`] の挙動を制御するオプション。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExternalDataOptions {
    /// 1 モデルあたりの external data 合計バイト数の上限。超過は
    /// [`ExternalDataError::TotalSizeLimitExceeded`] で拒否する（確保の
    /// 前に検査するため、この上限を超える `length` はメモリを確保しない）。
    ///
    /// 読み込み経路の raw バッファ合計の予算であり、変換後テンソルを含めた
    /// 読み込み中のピークは最大でおよそ本値 ＋ 最大テンソル 1 個分になる
    /// （`docs/onnx-external-data-decision.md` 4.3 節）。上限以内でも確保
    /// できない場合は [`ExternalDataError::AllocationFailed`] で返す（abort
    /// しない）。利用可能メモリが既定値（64 GiB）より小さい環境では、
    /// 呼び出し元が本値を下げて渡すこと。
    pub max_total_bytes: u64,
    /// 1 モデルあたりに参照を許す external data ファイル数
    /// （実体単位・`FileKey` で畳み込んだ後の distinct 数）の上限。
    /// 超過は [`ExternalDataError::TooManyExternalFiles`] で拒否する
    /// （パス 1 でファイルを開いて `FileKey` を得た直後・既知ファイル集合へ
    /// 登録する前に検査する。いずれのパスもハンドルを 1 つずつ開いては
    /// 閉じるため、同時保持 fd 数はこの値に依存しない）。
    pub max_external_files: usize,
}

impl Default for ExternalDataOptions {
    fn default() -> Self {
        ExternalDataOptions {
            max_total_bytes: DEFAULT_MAX_EXTERNAL_DATA_TOTAL_BYTES,
            max_external_files: DEFAULT_MAX_EXTERNAL_FILES,
        }
    }
}

/// `location` 文字列・パス解決が拒否される理由。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum LocationRejectReason {
    Empty,
    ContainsNul,
    TooLong,
    Backslash,
    DrivePrefix,
    Absolute,
    ParentDir,
    Symlink,
    NotRegularFile,
    OutsideBaseDir,
    /// Windows 版 `resolve_and_open`（`win_contained_open`）が、開いた
    /// ハンドル自身の属性照会で reparse point（シンボリックリンク・
    /// junction のいずれも区別せず検出する。`file_type().is_symlink()` は
    /// AppExecLink・cloud files 等の一部 reparse タグを認識しないため、
    /// 属性ビット `FILE_ATTRIBUTE_REPARSE_POINT` で判定する）を検出した
    /// 場合に返す（イシュー #2349）。unix の [`LocationRejectReason::
    /// Symlink`] に相当する Windows 版。
    ReparsePoint,
    /// Windows の代替データストリーム記法（`name:stream`）を拒否する
    /// （字句検査。イシュー #2349）。unix ではファイル名として正当な
    /// ため本検査は `cfg(windows)` でのみ配線する（R5: unix の受理範囲は
    /// 変えない）。
    AlternateDataStream,
    /// Windows の予約デバイス名（`CON`／`PRN`／`AUX`／`NUL`／`COM0`〜`COM9`／
    /// `LPT0`〜`LPT9`〈上付き数字を含む〉／`CONIN$`／`CONOUT$`）を拒否する
    /// （大文字小文字を区別せず、拡張子・末尾の空白/ドットを除いた基底名で
    /// 判定。イシュー #2349）。
    ReservedDeviceName,
    /// Windows で無効な成分名（末尾のドット・空白、または `<>"|?*` ・
    /// 制御文字を含む）を拒否する（イシュー #2349）。
    InvalidComponentName,
}

impl fmt::Display for LocationRejectReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            LocationRejectReason::Empty => "空文字列",
            LocationRejectReason::ContainsNul => "NUL を含む",
            LocationRejectReason::TooLong => "4096 バイトを超える",
            LocationRejectReason::Backslash => "バックスラッシュを含む",
            LocationRejectReason::DrivePrefix => "ドライブ文字の接頭辞",
            LocationRejectReason::Absolute => "絶対パス",
            LocationRejectReason::ParentDir => "親ディレクトリ参照（..）を含む",
            LocationRejectReason::Symlink => "経路にシンボリックリンクを含む",
            LocationRejectReason::NotRegularFile => "通常ファイルではない",
            LocationRejectReason::OutsideBaseDir => "base_dir の外へ解決される",
            LocationRejectReason::ReparsePoint => {
                "経路に reparse point（シンボリックリンク・junction 等）を含む"
            }
            LocationRejectReason::AlternateDataStream => "代替データストリーム（:）を含む",
            LocationRejectReason::ReservedDeviceName => "Windows の予約デバイス名",
            LocationRejectReason::InvalidComponentName => {
                "Windows で無効な文字、または末尾のドット・空白を含む"
            }
        };
        write!(f, "{s}")
    }
}

/// external data 解決時のエラー。`graph::GraphError::ExternalData` に包んで
/// 返す。非信頼文字列（`tensor_name`／`key`）は
/// `cap_sparse_tensor_diag_name` と同じ 256 バイト上限で切り詰める。
/// ホストの絶対パス・canonicalize 後のパスはいずれの variant にも含めない
/// （security.md A05）。
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ExternalDataError {
    /// `data_location` が `DEFAULT`（0）・`EXTERNAL`（1）以外。
    InvalidDataLocation { tensor_name: String, value: i32 },
    /// `EXTERNAL` なのに inline データが非空、または `DEFAULT` なのに
    /// `external_data` が非空。
    InconsistentDataFields {
        tensor_name: String,
        reason: &'static str,
    },
    /// `external_data` に `location` キーが無い。
    MissingLocationKey { tensor_name: String },
    /// `external_data` のキーが重複している。
    DuplicateKey { tensor_name: String, key: String },
    /// `location`／`offset`／`length` 以外のキー。
    UnknownKey { tensor_name: String, key: String },
    /// `checksum` キーは非対応のため fail-closed に拒否する。
    ChecksumUnsupported { tensor_name: String },
    /// `location` のパス検証に失敗した。
    InvalidLocation {
        tensor_name: String,
        reason: LocationRejectReason,
    },
    /// `offset`／`length` の文法・範囲が不正。
    InvalidNumber {
        tensor_name: String,
        field: &'static str,
    },
    /// `offset + length` がファイル長を超える。
    RangeOutOfFile { tensor_name: String },
    /// 実バイト長が `dims`／`data_type` から期待されるバイト長と一致しない。
    LengthMismatch {
        tensor_name: String,
        expected_bytes: u64,
        actual_bytes: u64,
    },
    /// 同一ファイル内で 2 つ以上のテンソルの読み込み区間が重なっている。
    OverlappingRegion {
        tensor_name: String,
        other_tensor_name: String,
    },
    /// external data の合計サイズが上限を超えた。
    TotalSizeLimitExceeded { limit: u64, requested: u64 },
    /// distinct な external data ファイル数（実体単位）が上限を超えた。
    /// `max_total_bytes` はバイト数のみを制限するため、サイズ 0 のテンソル
    /// を大量に異なるファイルへ分散させる攻撃はバイト数の上限をすり抜ける
    /// （A04 資源枯渇対策。`ExternalDataOptions::max_external_files`）。
    TooManyExternalFiles { limit: usize },
    /// ファイル I/O の失敗（`NotFound`／権限エラー等）。
    Io {
        tensor_name: String,
        kind: std::io::ErrorKind,
    },
    /// パス 2 で再 open したハンドル（または各区間の読み込み直前に同じ
    /// ハンドルへ取り直した `fstat`）のファイル長・実体識別子・変更時刻
    /// （Unix では dev/ino・ctime・mtime）がパス 1 の記録と食い違った
    /// （TOCTOU 検知）。照合通過直後〜読み込みの間の truncate は本
    /// variant ではなく `Io`（`UnexpectedEof`）で拒否される。
    FileChangedDuringLoad { tensor_name: String },
    /// `base_dir` の canonicalize に失敗した。
    InvalidBaseDir { kind: std::io::ErrorKind },
    /// initializer 名が重複している（I/O の前に検出する。`graph::
    /// build_graph` の `DuplicateInitializerName` と同一の欠陥クラス）。
    DuplicateInitializerName { tensor_name: String },
    /// `resolve_and_open` の TOCTOU 非後退実装は `cfg(unix)` 全般
    /// （`libc` 経由の `openat2`／ディレクトリ fd 起点の `O_NOFOLLOW`
    /// 追跡拒否オープン。Linux／macOS／その他 unix）に加え、`cfg(windows)`
    /// （`win_contained_open`。祖先ディレクトリのハンドル連鎖保持＋開いた
    /// ハンドル自身の reparse point 属性検査。イシュー #2349）でも提供
    /// 済みであり、拒否対象は**unix でも Windows でもないプラットフォーム**
    /// （wasm32-unknown-unknown 等。安全な経路解決手段そのものが無い）
    /// のみである。旧来の `symlink_metadata` 検証 → `canonicalize` →
    /// `File::open` 再解決経路は検証とオープンの間にシンボリックリンク
    /// 差し替えの窓が残るため、REQ-1 の完全自作コア方針・security.md の
    /// A08（自己修復ループが取り込む変更の整合性）と同じ fail-closed
    /// 原則に従い、対応不能なプラットフォームでは external data の
    /// 読み込みそのものを拒否する（P0・PRRT_kwDOTuUCJc6mk30J 是正）。
    UnsupportedPlatformForSecureResolve { tensor_name: String },
    /// external data の読み込みバッファ、または external 由来 initializer の
    /// 復号先（`RawTensor` の要素 Vec）の確保に失敗した（A04 資源枯渇対策。
    /// PR #2348 codex P0 是正）。`bytes` は確保しようとしたバイト数。
    ///
    /// `max_total_bytes`（既定 64 GiB）以内の宣言長であっても、小さな
    /// `.onnx` と疎ファイルから利用可能メモリを超える単一テンソルを宣言
    /// できるため、確保は `Vec::try_reserve_exact` による失敗可能確保とし、
    /// 失敗をプロセス終了（`handle_alloc_error` による abort）ではなく本
    /// variant で返す。`usize` へ変換できない長さ（32bit ターゲット）や
    /// `isize::MAX` を超える長さも確保を試みる前に本 variant で拒否する。
    AllocationFailed { tensor_name: String, bytes: u64 },
    /// 内部不変条件違反（本来発生しないはずの状態）。`coding-rust.md`
    /// の「本番経路で `unwrap()`/`expect()` を使わない」方針に従い、
    /// `panic!`／`unreachable!`／`.expect()` の代わりにこの型付きエラーで
    /// 表面化させる（`plan` と `load`／書き戻しの間の内部不変条件——
    /// `plan` が記録した location 添字・slot・`base_dir` ハンドルは
    /// `load`／書き戻し時にも有効なはず——が崩れた場合のみ到達する）。
    Internal { reason: &'static str },
}

impl fmt::Display for ExternalDataError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ExternalDataError::InvalidDataLocation { tensor_name, value } => {
                write!(f, "data_location が不正（tensor={tensor_name}）: {value}")
            }
            ExternalDataError::InconsistentDataFields {
                tensor_name,
                reason,
            } => write!(
                f,
                "data_location と実データフィールドの組み合わせが不正（tensor={tensor_name}）: {reason}"
            ),
            ExternalDataError::MissingLocationKey { tensor_name } => {
                write!(
                    f,
                    "external_data に location キーが無い（tensor={tensor_name}）"
                )
            }
            ExternalDataError::DuplicateKey { tensor_name, key } => {
                write!(
                    f,
                    "external_data のキー重複（tensor={tensor_name} key={key}）"
                )
            }
            ExternalDataError::UnknownKey { tensor_name, key } => {
                write!(
                    f,
                    "external_data の未知キー（tensor={tensor_name} key={key}）"
                )
            }
            ExternalDataError::ChecksumUnsupported { tensor_name } => {
                write!(f, "checksum キーは非対応のため拒否（tensor={tensor_name}）")
            }
            ExternalDataError::InvalidLocation {
                tensor_name,
                reason,
            } => write!(f, "location が不正（tensor={tensor_name}）: {reason}"),
            ExternalDataError::InvalidNumber { tensor_name, field } => {
                write!(f, "{field} の値が不正（tensor={tensor_name}）")
            }
            ExternalDataError::RangeOutOfFile { tensor_name } => {
                write!(
                    f,
                    "offset+length がファイル長を超える（tensor={tensor_name}）"
                )
            }
            ExternalDataError::LengthMismatch {
                tensor_name,
                expected_bytes,
                actual_bytes,
            } => write!(
                f,
                "external data のバイト長不整合（tensor={tensor_name}）: 期待={expected_bytes} 実際={actual_bytes}"
            ),
            ExternalDataError::OverlappingRegion {
                tensor_name,
                other_tensor_name,
            } => write!(
                f,
                "external data の区間重複（tensor={tensor_name} other={other_tensor_name}）"
            ),
            ExternalDataError::TotalSizeLimitExceeded { limit, requested } => write!(
                f,
                "external data 合計サイズが上限を超過: limit={limit} requested={requested}"
            ),
            ExternalDataError::TooManyExternalFiles { limit } => write!(
                f,
                "external data ファイル数（実体単位）が上限を超過: limit={limit}"
            ),
            ExternalDataError::Io { tensor_name, kind } => {
                write!(
                    f,
                    "external data の I/O エラー（tensor={tensor_name}）: {kind:?}"
                )
            }
            ExternalDataError::FileChangedDuringLoad { tensor_name } => write!(
                f,
                "読み込み直前にファイルが変化した（tensor={tensor_name}）: TOCTOU 検知"
            ),
            ExternalDataError::InvalidBaseDir { kind } => {
                write!(f, "base_dir の解決に失敗: {kind:?}")
            }
            ExternalDataError::DuplicateInitializerName { tensor_name } => {
                write!(f, "initializer 名の重複（tensor={tensor_name}）")
            }
            ExternalDataError::UnsupportedPlatformForSecureResolve { tensor_name } => write!(
                f,
                "external data の安全な解決（TOCTOU 非後退の封じ込めオープン）が \
                 このプラットフォームでは未対応のため拒否（tensor={tensor_name}）: unix・\
                 Windows 以外（wasm32 等）では external data 読み込みをサポートしない"
            ),
            ExternalDataError::AllocationFailed { tensor_name, bytes } => write!(
                f,
                "external data 用メモリの確保に失敗（tensor={tensor_name}）: bytes={bytes}"
            ),
            ExternalDataError::Internal { reason } => {
                write!(f, "external_data 内部不変条件違反: {reason}")
            }
        }
    }
}

impl std::error::Error for ExternalDataError {}

/// 診断用テンソル名を `cap_sparse_tensor_diag_name` と同じ上限で切り詰める。
fn cap_name(name: &str) -> String {
    cap_sparse_tensor_diag_name(name.as_bytes())
}

/// `location` 文字列を検証し、`base_dir` へ 1 段ずつ連結できる正規
/// コンポーネント列（`CurDir` は除去済み）を返す（`Path::components` の
/// 判定結果をそのまま返すことで、呼び出し元〈`resolve_and_open`〉が
/// 「ここには `Normal` しか来ないはず」という前提を `unreachable!` で
/// 表現せずに済む。`coding-rust.md` の本番経路 `unwrap`/`expect` 禁止と
/// 同じ理由で `unreachable!` も避ける）。
fn validate_location_string(loc: &str) -> Result<Vec<&std::ffi::OsStr>, LocationRejectReason> {
    if loc.is_empty() {
        return Err(LocationRejectReason::Empty);
    }
    if loc.len() > 4096 {
        return Err(LocationRejectReason::TooLong);
    }
    if loc.contains('\0') {
        return Err(LocationRejectReason::ContainsNul);
    }
    if loc.contains('\\') {
        return Err(LocationRejectReason::Backslash);
    }
    // ドライブ文字接頭辞（`C:` 等）: 先頭 2 バイトが ASCII 英字 + ':'。
    let bytes = loc.as_bytes();
    if bytes.len() >= 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':' {
        return Err(LocationRejectReason::DrivePrefix);
    }
    let mut parts = Vec::new();
    for component in Path::new(loc).components() {
        match component {
            Component::Normal(part) => {
                // Windows 固有の字句検査（ADS・予約デバイス名・禁止文字・
                // 末尾のドット/空白。イシュー #2349）。配線は `cfg(windows)`
                // 限定とし、unix での受理範囲（例: `a:b` は unix では正当な
                // ファイル名）は変えない（R5）。
                #[cfg(windows)]
                if let Some(reason) = windows_component_reject_reason(part) {
                    return Err(reason);
                }
                parts.push(part)
            }
            Component::CurDir => {}
            Component::RootDir | Component::Prefix(_) => {
                return Err(LocationRejectReason::Absolute);
            }
            Component::ParentDir => return Err(LocationRejectReason::ParentDir),
        }
    }
    Ok(parts)
}

/// Windows 向け成分の字句検査（イシュー #2349）。ADS（代替データ
/// ストリーム。`name:stream`）・予約デバイス名・禁止文字・末尾のドット/
/// 空白を拒否する。`cfg(any(windows, test))` を付けているため、Linux でも
/// `#[cfg(test)]` ビルドには含まれ単体テストできる（本体ビルドでは
/// Windows 限定）。呼び出し配線は [`validate_location_string`] 内の
/// `cfg(windows)` のみで、unix の受理範囲は変えない（R5）。
#[cfg(any(windows, test))]
fn windows_component_reject_reason(part: &std::ffi::OsStr) -> Option<LocationRejectReason> {
    // `part` は `loc: &str` から作った `Path` の `Component::Normal` の
    // 部分文字列であり、常に有効な UTF-8（`to_str()` は必ず `Some`）。
    let s = part.to_str()?;
    if s.contains(':') {
        return Some(LocationRejectReason::AlternateDataStream);
    }
    if s.ends_with('.') || s.ends_with(' ') {
        return Some(LocationRejectReason::InvalidComponentName);
    }
    if s.chars()
        .any(|c| matches!(c, '<' | '>' | '"' | '|' | '?' | '*') || (c as u32) < 0x20)
    {
        return Some(LocationRejectReason::InvalidComponentName);
    }
    if windows_is_reserved_device_name(s) {
        return Some(LocationRejectReason::ReservedDeviceName);
    }
    None
}

/// `s` が Windows の予約デバイス名かどうかを判定する（大文字小文字を
/// 区別せず、拡張子〈最初の `.` より後〉を除いた基底名で判定）。対象は
/// `CON`／`PRN`／`AUX`／`NUL`／`COM0`〜`COM9`（上付き数字 `¹²³` の
/// `COM¹`〜`COM³` を含む）／`LPT0`〜`LPT9`（同上）／`CONIN$`／`CONOUT$`
/// （MS Learn "Naming Files, Paths, and Namespaces" の予約名一覧）。
#[cfg(any(windows, test))]
fn windows_is_reserved_device_name(s: &str) -> bool {
    let base = s.split('.').next().unwrap_or(s);
    let upper = base.to_ascii_uppercase();
    if matches!(
        upper.as_str(),
        "CON" | "PRN" | "AUX" | "NUL" | "CONIN$" | "CONOUT$"
    ) {
        return true;
    }
    for prefix in ["COM", "LPT"] {
        let Some(rest) = upper.strip_prefix(prefix) else {
            continue;
        };
        let mut chars = rest.chars();
        let (Some(c), None) = (chars.next(), chars.next()) else {
            continue;
        };
        if c.is_ascii_digit() || matches!(c, '¹' | '²' | '³') {
            return true;
        }
    }
    false
}

/// [`windows_component_reject_reason`] の単体テスト（イシュー #2349）。
/// 対象関数自体は `cfg(any(windows, test))` のため Linux の `cfg(test)`
/// ビルドでも実体があり、Windows 実機を使わずに Linux CI で検証できる
/// （計画 §5 ステップ 2）。
#[cfg(test)]
mod windows_lexical_tests {
    use std::ffi::OsStr;

    use super::{LocationRejectReason, windows_component_reject_reason};

    fn reject(name: &str) -> Option<LocationRejectReason> {
        windows_component_reject_reason(OsStr::new(name))
    }

    #[test]
    fn reserved_device_names_are_rejected() {
        for name in [
            "CON",
            "con",
            "PRN",
            "AUX",
            "NUL",
            "con.txt",
            "COM1.data",
            "LPT9",
            "COM¹",
            "CONIN$",
            "CONOUT$",
        ] {
            assert!(
                reject(name).is_some(),
                "{name}: 予約デバイス名として拒否されるはず"
            );
        }
    }

    #[test]
    fn alternate_data_stream_is_rejected() {
        assert_eq!(
            reject("a:b"),
            Some(LocationRejectReason::AlternateDataStream)
        );
    }

    #[test]
    fn trailing_dot_or_space_is_rejected() {
        assert_eq!(
            reject("x."),
            Some(LocationRejectReason::InvalidComponentName)
        );
        assert_eq!(
            reject("x "),
            Some(LocationRejectReason::InvalidComponentName)
        );
        // "NUL." は拡張子除去前の末尾ドットで先に InvalidComponentName に
        // 分類される（`windows_component_reject_reason` 内の判定順）。
        assert_eq!(
            reject("NUL."),
            Some(LocationRejectReason::InvalidComponentName)
        );
    }

    #[test]
    fn forbidden_characters_are_rejected() {
        assert!(reject("a*b").is_some());
        assert!(reject("a?b").is_some());
        assert!(reject("a<b").is_some());
        assert!(reject("a>b").is_some());
        assert!(reject("a\"b").is_some());
        assert!(reject("a|b").is_some());
        assert!(reject("a\u{1}b").is_some(), "制御文字は拒否されるはず");
    }

    #[test]
    fn ordinary_names_are_accepted() {
        for name in [
            "CONSOLE",
            "COM10",
            "model.onnx.data",
            "weights.bin",
            "a",
            "foo.bar.baz",
        ] {
            assert_eq!(
                reject(name),
                None,
                "{name}: 通常のファイル名は拒否されないはず"
            );
        }
    }
}

/// ファイル実体の同一性・無変更性を照合するための `fstat` スナップショット。
///
/// パス 1（`plan`）で開いたハンドル自身の `fstat` から取り、
/// [`PlannedLocation::snapshot`] として記録する。パス 2（`load`）は再 open
/// したハンドルの値、および各区間の読み込み直前に同じハンドルへ取り直した
/// 値を [`ensure_unchanged`] で記録と**全フィールド完全一致**で照合する。
///
/// unix では dev/ino・長さに加えて ctime／mtime（秒＋ナノ秒）を持つ
/// （PR #2348 security-auditor P2-1 是正）。dev/ino・長さだけでは、パス 1 の
/// close 後に元ファイルを unlink → 同じ長さの別ファイルを作成 → inode 番号が
/// 再利用される経路（`base_dir` 配下のディレクトリ書き込み権〈unlink／
/// create〉だけで起こせる）を検出できない。新しい inode の ctime は作成
/// 時刻になり、in-place 改変（write・truncate・chmod・link 等）でも ctime が
/// 更新されるため、ctime を照合に加えることでいずれも検出できる。ctime は
/// ユーザー空間から任意の値へ設定できない（`utimensat` で mtime を書き戻す
/// 操作自体が ctime を更新する）ため照合の要は ctime で、mtime は補助で
/// ある。atime は読み込みだけで更新されうる（relatime 等）ため含めない
/// （誤検知を避ける）。残存リスク（同一 ctime 粒度内での再作成）は
/// モジュール doc「TOCTOU の論拠」節・`docs/onnx-external-data-decision.md`
/// 5 節を参照。
///
/// unix・Windows 以外（`resolve_and_open` が常に拒否しファイルを開かない
/// プラットフォーム）では、型の形だけを揃える長さのみを持つ。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FileSnapshot {
    len: u64,
    #[cfg(unix)]
    dev: u64,
    #[cfg(unix)]
    ino: u64,
    #[cfg(unix)]
    ctime: i64,
    #[cfg(unix)]
    ctime_nsec: i64,
    #[cfg(unix)]
    mtime: i64,
    #[cfg(unix)]
    mtime_nsec: i64,
    /// Windows 版（イシュー #2349）。dev/ino 相当（`file_index`）・ctime
    /// 相当（`change_time`）は `std::os::windows::fs::MetadataExt` が
    /// 1.98.1 時点で安定化していないため使えない（計画段階の実測）。その
    /// ため実体識別は `FileKey`（パスベース。`file_key_for` の
    /// `cfg(not(unix))` 共通実装）に委ね、本 snapshot は利用者が
    /// `SetFileTime` 等で書き換え可能な属性・作成時刻・更新時刻のみで
    /// 変化検知する（unix の ctime ほど強い保証ではない残存リスク。
    /// `docs/onnx-external-data-decision.md` 5 節）。
    #[cfg(windows)]
    file_attributes: u32,
    #[cfg(windows)]
    creation_time: u64,
    #[cfg(windows)]
    last_write_time: u64,
}

impl FileSnapshot {
    /// ハンドル自身に対する `fstat`（`File::metadata`）の結果から作る
    /// （経路文字列を再解決しない）。
    #[cfg(unix)]
    fn from_metadata(meta: &std::fs::Metadata) -> Self {
        use std::os::unix::fs::MetadataExt;
        FileSnapshot {
            len: meta.len(),
            dev: meta.dev(),
            ino: meta.ino(),
            ctime: meta.ctime(),
            ctime_nsec: meta.ctime_nsec(),
            mtime: meta.mtime(),
            mtime_nsec: meta.mtime_nsec(),
        }
    }

    /// Windows 版。ハンドル自身に対する属性照会（`File::metadata` ->
    /// `GetFileInformationByHandle` 相当）の結果から作る（経路文字列を
    /// 再解決しない。unix 版と同じ契約）。
    #[cfg(windows)]
    fn from_metadata(meta: &std::fs::Metadata) -> Self {
        use std::os::windows::fs::MetadataExt;
        FileSnapshot {
            len: meta.len(),
            file_attributes: meta.file_attributes(),
            creation_time: meta.creation_time(),
            last_write_time: meta.last_write_time(),
        }
    }

    /// unix・Windows 以外向け（長さのみ）。`resolve_and_open` が常に拒否
    /// するため実行時には到達しないが、`load` の照合コードを cfg で
    /// 分岐させずに共通化する。
    #[cfg(not(any(unix, windows)))]
    fn from_metadata(meta: &std::fs::Metadata) -> Self {
        FileSnapshot { len: meta.len() }
    }
}

/// パス 2（`load`）の照合: `now`（再 open 直後、または各区間の読み込み
/// 直前に同じハンドルへ取り直した `fstat`）が `planned`（パス 1 の記録）と
/// 全フィールド一致しなければ [`ExternalDataError::FileChangedDuringLoad`]
/// で拒否する（fail-closed）。比較ロジックを単体テスト可能にするため
/// `load` から切り出してある。
fn ensure_unchanged(
    tensor_name: &str,
    planned: &FileSnapshot,
    now: &FileSnapshot,
) -> Result<(), ExternalDataError> {
    if planned == now {
        Ok(())
    } else {
        Err(ExternalDataError::FileChangedDuringLoad {
            tensor_name: cap_name(tensor_name),
        })
    }
}

/// [`resolve_and_open`] が開いたファイルと、そのハンドル自身に対する
/// `fstat` で得たスナップショット。パス 1（`plan`）では `FileKey`・
/// スナップショットを記録した直後に破棄（close）し、パス 2（`load`）では
/// 再 open した同型の値をパス 1 の記録と照合してから読み込みに使う（どちらの
/// パスも本型をコレクションへ溜め込まない。モジュール doc「ハンドル非保持の
/// 構成」節）。
struct OpenFile {
    file: File,
    snapshot: FileSnapshot,
}

/// ディレクトリ fd（`base_dir` を開いたハンドル）を起点に `location` を
/// シンボリックリンク拒否で解決してファイルを開く実装（P0 対応。
/// discussion_r4119392011・イシュー #2347 是正版）。
///
/// 旧実装は手書きの `extern "C"` 宣言と手書きの `openat` フラグ定数を
/// 使っていたが、Linux では同じ定数でも CPU アーキテクチャごとに値が
/// 異なり（例: x86 は `O_DIRECTORY=0o200000`・aarch64 は
/// `O_DIRECTORY=0o40000`）、x86 向けの値のまま aarch64 Linux（DGX Spark
/// GB10）でビルドするとシンボリックリンク拒否が機能しない実装バグを
/// 生んでいた。本実装は `libc`（`.claude/rules/deps-policy.md`「OS FFI」
/// 区分。2026-09-28 ユーザー承認）の定数・関数を使い、この種の値の取り
/// 違えを自作せず解消する。
///
/// 経路は文字列として再解決しない。**Linux** は `openat2(2)`
/// （[`open_chain_openat2`]。`RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS |
/// RESOLVE_NO_MAGICLINKS`）で経路解決全体をカーネルへ 1 回のシステム
/// コールとしてアトミックに封じ込めさせる。`openat2` が未対応
/// （`ENOSYS`／`EPERM`）の場合のみ、成分ごと逐次 `openat(dirfd, name,
/// O_NOFOLLOW)` で辿る [`open_chain_component_walk`] へフォールバックする
/// （**macOS・その他 unix**〈`openat2` 非対応〉では常にこちらを使う）。
/// いずれの方式も、対象がシンボリックリンクなら `ELOOP` で即座に失敗する
/// ため検証済みの fd 連鎖以外を辿る余地が生じない。エラー分類用の診断
/// （シンボリックリンクか否かの判別）も、既に開いた親ディレクトリ fd を
/// 起点にした単一コンポーネントの `fstatat(AT_SYMLINK_NOFOLLOW)` のみを
/// 使い、`symlink_metadata`／`canonicalize` によるパス文字列の再解決は
/// 一切行わない（P0 是正: PRRT_kwDOTuUCJc6mkUsI・PRRT_kwDOTuUCJc6mk30J。
/// 旧実装はここで累積パス文字列 `symlink_metadata` を呼んでおり、その
/// 呼び出し自体が検証後の再解決だった）。
///
/// `base_dir` のディレクトリ fd は [`open_base_dir`] で 1 度だけ開き、
/// `plan`（全テンソル分の初回解決）と `load`（location ごとの再 open）が
/// 通して再利用する（呼び出しごとに開き直さない。advisor 指摘）。
#[cfg(unix)]
mod no_follow_open {
    use std::ffi::{CString, OsStr};
    use std::fs::File;
    use std::io;
    #[cfg(target_os = "linux")]
    use std::mem::size_of;
    use std::os::unix::ffi::OsStrExt;
    #[cfg(unix)]
    use std::os::unix::fs::OpenOptionsExt;
    use std::os::unix::io::{AsRawFd, FromRawFd};
    use std::path::Path;

    /// `base_dir_canonical` をディレクトリ fd として 1 度だけ開く。`plan`・
    /// `load` が全 external テンソル分を通してこの fd を再利用する。`O_NONBLOCK`
    /// は FIFO 混入時の無期限ハング防止（下記 `openat_no_follow` と同じ
    /// 理由）。`OpenOptionsExt::custom_flags` は安全な std API のため
    /// `unsafe` を要しない。
    pub(super) fn open_base_dir(base_dir_canonical: &Path) -> io::Result<File> {
        std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_DIRECTORY | libc::O_CLOEXEC | libc::O_NONBLOCK)
            .open(base_dir_canonical)
    }

    /// シンボリックリンク検知（`ELOOP`）かどうかを判定する。Linux／macOS
    /// 共通で POSIX が規定する errno のため `libc::ELOOP` をそのまま使う
    /// （アーキテクチャ間の値の取り違えは `libc` が吸収する）。
    fn is_eloop(e: &io::Error) -> bool {
        e.raw_os_error() == Some(libc::ELOOP)
    }

    /// `dir`（親ディレクトリ fd）に対して相対的な単一コンポーネント
    /// `name` の種別を診断する（シンボリックリンクか否か）。
    /// `open_chain_component_walk` が `ENOTDIR`（「シンボリックリンクを
    /// `O_DIRECTORY` 付きで開いた」場合と「単なる非ディレクトリの通常
    /// ファイルを `O_DIRECTORY` 付きで開いた」場合の両方で返り得る）の
    /// どちらのエラー種別として報告するかの分類にのみ使う診断専用の
    /// 関数。`dir` は呼び出し元が既に検証済みの親ディレクトリ fd であり、
    /// `fstatat(AT_SYMLINK_NOFOLLOW)` は `name` 単一コンポーネントのみを
    /// その fd に対して相対的に見るため、経路文字列の再解決（TOCTOU 窓）
    /// を伴わない。
    fn is_symlink_component(dir: &File, name: &OsStr) -> bool {
        let Ok(c_name) = CString::new(name.as_bytes()) else {
            return false;
        };
        // SAFETY: `zeroed()` で得る `libc::stat` はすべてのフィールドが
        // ビット表現 0 でも有効な値になる POD 構造体（POSIX の `stat`
        // 構造体は整数・配列フィールドのみで構成され、`0` 埋めのままでも
        // 不変条件を持たない）。`fstatat` は成功時のみこのバッファへ書き
        // 込むため、`ret == 0` を確認してから中身を読む下の判定は健全。
        let mut st: libc::stat = unsafe { std::mem::zeroed() };
        // SAFETY: `dir.as_raw_fd()` はこの呼び出しの間生存している `dir`
        // が所有する有効な open ディレクトリ fd。`c_name` は直前の
        // `CString::new` が NUL 終端を保証した有効な C 文字列でこの
        // 呼び出しの間生存する。`&mut st` は上で初期化済みの有効な出力
        // バッファへの排他参照。本関数は結果の真偽のみを診断（エラー
        // 種別の分類）に使い、安全性判断（fail-closed 拒否の可否）には
        // 使わない。
        let ret = unsafe {
            libc::fstatat(
                dir.as_raw_fd(),
                c_name.as_ptr(),
                &mut st,
                libc::AT_SYMLINK_NOFOLLOW,
            )
        };
        ret == 0 && (st.st_mode & libc::S_IFMT) == libc::S_IFLNK
    }

    /// `dir` に対して相対的に `name`（単一パス成分。`..`／`/` を含まない
    /// `Path::components()` の `Normal` 由来の値のみを渡す前提）を
    /// `O_NOFOLLOW` で開く。`want_dir` が true なら `O_DIRECTORY` を付け、
    /// 対象がディレクトリでなければ失敗する。
    fn openat_no_follow(dir: &File, name: &OsStr, want_dir: bool) -> io::Result<File> {
        let c_name = CString::new(name.as_bytes())
            .map_err(|_| io::Error::from(io::ErrorKind::InvalidInput))?;
        // `O_NONBLOCK` を無条件で付与する（High・Cursor Bugbot 指摘）:
        // `base_dir` 配下に FIFO（named pipe）等の特殊ファイルが置かれて
        // いた場合、`O_NONBLOCK` 抜きの `open`/`openat` は対向の
        // reader/writer が現れるまで無期限にブロックし得る。`is_file()`
        // による種別検証は open 成功後にしか行えないため、open 自体が
        // ハングすると検証に到達できない。通常ファイル・ディレクトリの
        // open には副作用が無い（POSIX: `O_NONBLOCK` は FIFO・キャラクタ
        // デバイス・ソケットにのみ意味を持つ）ため、中間ディレクトリ・
        // 最終ファイルのどちらの open にも無条件で付与してよい。
        let flags = if want_dir {
            libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC | libc::O_NONBLOCK
        } else {
            libc::O_RDONLY | libc::O_NOFOLLOW | libc::O_CLOEXEC | libc::O_NONBLOCK
        };
        // SAFETY: `dir.as_raw_fd()` はこの呼び出しの間生存している `dir` が
        // 所有する有効な open ディレクトリ fd。`c_name` は
        // `CString::new` が NUL 終端を保証した有効な C 文字列で、この
        // 呼び出しの間生存する。`libc::openat` は POSIX 標準の可変長引数
        // 関数だが本呼び出しは `O_CREAT` を渡さないため可変長引数は
        // 実際には使わない。返り値が非負なら新規に確保された fd の所有権
        // を呼び出し元へ渡す契約（POSIX `openat(2)`）であり、
        // `File::from_raw_fd` で即座に `File` へ委譲することで二重解放・
        // リークを防ぐ。
        let fd = unsafe { libc::openat(dir.as_raw_fd(), c_name.as_ptr(), flags) };
        if fd < 0 {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: 直前の `openat` が返した非負 fd は呼び出し元がここで
        // 一意に所有権を得る新規 fd であり、他のどのコードもまだ
        // 参照していない。
        Ok(unsafe { File::from_raw_fd(fd) })
    }

    /// `base_dir`（[`open_base_dir`] が開いたディレクトリ fd）を起点に
    /// `parts` を 1 段ずつ `O_NOFOLLOW` で辿り、最終ファイルの fd を得る
    /// （`openat2` 非対応環境向けのフォールバック方式。macOS・その他
    /// unix では常にこちらを使う）。中間段は `O_DIRECTORY` 付きで開く
    /// ためディレクトリでなければ失敗し、途中経路のシンボリックリンクは
    /// `ELOOP` で拒否される。エラーはシンボリックリンク検知か否かを
    /// 呼び出し元が判別できるよう `(io::Error, bool /* is_symlink */)`
    /// を返す。
    pub(super) fn open_chain_component_walk(
        base_dir: &File,
        parts: &[&OsStr],
    ) -> Result<File, (io::Error, bool)> {
        if parts.is_empty() {
            return Err((io::Error::from(io::ErrorKind::InvalidInput), false));
        }
        let last_idx = parts.len() - 1;
        // 直前段の fd。最初の反復は呼び出し元が既に開いた `base_dir` を
        // 起点にし、以後は毎段 `openat_no_follow` が返す新規 fd に置き
        // 換わる（経路文字列ではなく fd を起点に辿る）。
        let mut dir_owned: Option<File> = None;
        for (i, part) in parts.iter().enumerate() {
            let want_dir = i != last_idx;
            let current: &File = dir_owned.as_ref().unwrap_or(base_dir);
            match openat_no_follow(current, part, want_dir) {
                Ok(next) => dir_owned = Some(next),
                Err(e) => {
                    let is_sym = is_eloop(&e)
                        || (want_dir
                            && e.raw_os_error() == Some(libc::ENOTDIR)
                            && is_symlink_component(current, part));
                    return Err((e, is_sym));
                }
            }
        }
        // `parts` は上で非空を確認済みのため、ループは必ず 1 回以上
        // `dir_owned` へ書き込んでから正常終了する。`unwrap()` の代わりに
        // 型付きエラーで表面化させる（coding-rust.md「本番経路で
        // unwrap/expect を使わない」。到達しないはずの防御的分岐）。
        dir_owned.ok_or((io::Error::from(io::ErrorKind::InvalidInput), false))
    }

    /// Linux 限定: `openat2(2)`（`RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS |
    /// RESOLVE_NO_MAGICLINKS`）で経路解決全体をカーネルへ 1 回のシステム
    /// コールとしてアトミックに封じ込めさせる。`RESOLVE_NO_XDEV` は
    /// 意図的に付与しない（bind mount 越しのモデルディレクトリ配置を
    /// 壊さないため。`RESOLVE_BENEATH` と `RESOLVE_NO_SYMLINKS` だけで
    /// `base_dir` 外への脱出は既に閉じている）。`libc` は `SYS_openat2`・
    /// `open_how`・`RESOLVE_*` 定数は提供するが `openat2()` の関数
    /// ラッパー自体は未提供のため `libc::syscall` 経由で呼ぶ。
    #[cfg(target_os = "linux")]
    pub(super) enum Openat2Outcome {
        Opened(File),
        Failed(io::Error, bool /* is_symlink */),
        /// `openat2` 自体が未対応（`ENOSYS`／`EPERM`。古いカーネル・
        /// seccomp 等）。呼び出し元は [`open_chain_component_walk`] へ
        /// フォールバックする。
        Unsupported,
    }

    #[cfg(target_os = "linux")]
    pub(super) fn open_chain_openat2(base_dir: &File, parts: &[&OsStr]) -> Openat2Outcome {
        if parts.is_empty() {
            return Openat2Outcome::Failed(io::Error::from(io::ErrorKind::InvalidInput), false);
        }
        let mut rel: Vec<u8> = Vec::new();
        for (i, part) in parts.iter().enumerate() {
            if i > 0 {
                rel.push(b'/');
            }
            rel.extend_from_slice(part.as_bytes());
        }
        let Ok(c_rel) = CString::new(rel) else {
            return Openat2Outcome::Failed(io::Error::from(io::ErrorKind::InvalidInput), false);
        };
        // `libc::open_how` は `#[non_exhaustive]`（将来のカーネル ABI
        // 拡張に備えたフィールド追加余地）のため構造体リテラルで直接
        // 構築できない。POSIX の `open_how` は整数フィールドのみで構成
        // され `0` 埋めのままでも不変条件を持たない POD 構造体のため、
        // ゼロ初期化してから既知フィールドのみを設定する（下の
        // `is_symlink_component` の `libc::stat` ゼロ初期化と同じ根拠）。
        // SAFETY: `libc::open_how` はビット表現 0 が有効な値になる POD
        // 構造体（整数フィールドのみ・不変条件なし）。
        let mut how: libc::open_how = unsafe { std::mem::zeroed() };
        how.flags = (libc::O_RDONLY | libc::O_CLOEXEC | libc::O_NONBLOCK) as u64;
        how.resolve =
            libc::RESOLVE_BENEATH | libc::RESOLVE_NO_SYMLINKS | libc::RESOLVE_NO_MAGICLINKS;
        // rename race 等による一時的な `EAGAIN` のみ有限回再試行する
        // （`openat2(2)` man page: 経路解決が再試行を要して中断された
        // 場合に返る）。無限ループ化を避けるため上限を設ける。
        for _attempt in 0..4 {
            // SAFETY: `base_dir.as_raw_fd()` は生存している有効な
            // ディレクトリ fd。`c_rel` はこの呼び出しの間生存する有効な
            // C 文字列。`&how` は正しく初期化された `open_how`（POSIX
            // `openat2(2)` が要求するレイアウト）への参照で、第 4 引数に
            // 渡す構造体サイズ（`size_of::<libc::open_how>()`）も一致
            // する。返り値が非負なら新規に確保された fd の所有権をこの
            // 呼び出し元が唯一取得する契約（`openat2(2)`）であり、
            // `File::from_raw_fd` で即座に `File` へ委譲することで
            // 二重解放・リークを防ぐ。
            let ret = unsafe {
                libc::syscall(
                    libc::SYS_openat2,
                    base_dir.as_raw_fd(),
                    c_rel.as_ptr(),
                    &how as *const libc::open_how,
                    size_of::<libc::open_how>(),
                )
            };
            if ret >= 0 {
                // SAFETY: 上記と同じ根拠（直前の syscall が返した非負 fd
                // の所有権をここで一意に取得する）。
                return Openat2Outcome::Opened(unsafe { File::from_raw_fd(ret as i32) });
            }
            let e = io::Error::last_os_error();
            match e.raw_os_error() {
                Some(libc::ENOSYS) | Some(libc::EPERM) => return Openat2Outcome::Unsupported,
                Some(libc::EAGAIN) => continue,
                Some(libc::ELOOP) => return Openat2Outcome::Failed(e, true),
                _ => return Openat2Outcome::Failed(e, false),
            }
        }
        Openat2Outcome::Failed(io::Error::from(io::ErrorKind::WouldBlock), false)
    }
}

/// Windows 向け封じ込め open（イシュー #2349）。unix の [`no_follow_open`]
/// （dirfd 起点の `openat`／`openat2`）と同じ役割を Windows で果たす。
///
/// Windows の std にはディレクトリハンドル起点の相対オープン（`openat`
/// 相当。`NtCreateFile` の `RootDirectory`）が無く、`File::open` は常に
/// フルパスの `CreateFileW` になる（`FILE_FLAG_OPEN_REPARSE_POINT` が効く
/// のは最終成分だけで、途中の junction は辿ってしまう）。そのため本
/// モジュールは次の 3 つの手段を組み合わせて TOCTOU を閉じる
/// （`docs/onnx-external-data-decision.md` 5 節が設計判断の正。2026-09-28
/// codex-review 是正〈PR #2351〉で 3. を追加し、1.〜2. のみでは残っていた
/// 「途中成分・最終ディレクトリを reparse point 化してから即座に元へ
/// 戻す」flip-and-revert 型 TOCTOU を閉じた）:
///
/// 1. **祖先チェーンのハンドル保持**: `base_dir`（ボリュームルートから）
///    までの各ディレクトリ成分を開いたまま保持する（[`open_base_dir_handle`]
///    が返す [`super::BaseDirHandle`]）。対象を rename・削除するには
///    DELETE アクセスで開く必要があり、共有モードから `FILE_SHARE_DELETE`
///    を外すことで保持中の成分は他プロセスから rename・削除できない。
///    また NTFS では reparse point 化に対象ディレクトリが空であることを
///    要求するため、保持中の次成分を含むディレクトリは reparse point 化
///    できない。
/// 2. **開いた直後の属性検査**: 各ハンドルを開いた直後に、そのハンドル
///    自身（パスではない）に対する属性照会で reparse point 属性
///    （`FILE_ATTRIBUTE_REPARSE_POINT`）を検査する。判定は
///    `file_type().is_symlink()`（シンボリックリンク・マウントポイントの
///    タグしか認識せず AppExecLink・cloud files 等を素通りさせる）では
///    なく属性ビットで行う。
/// 3. **最終ハンドルの実所在検証**（[`final_real_path`]・
///    [`verify_final_path_within_base_dir`]）: 1.〜2. は「保持中の成分
///    自身」の破壊的操作は防ぐが、`open_component` は途中成分・最終成分の
///    いずれもフルパス文字列で `CreateFileW` するため、保持中（`held`）の
///    ディレクトリであっても、共有モードが許す書き込みアクセスで
///    別ハンドルから同一オブジェクトへ reparse タグを立てる（削除・
///    rename を伴わないため 1. の防御が及ばない）→ 以後のフルパス文字列
///    解決（`held` のハンドル経由ではなく毎回ファイルシステム名前空間を
///    再解決する）がその reparse point を中間成分として追跡してしまう →
///    事後の属性再検査（2.）が走る前に reparse タグを外して元へ戻す、
///    という flip-and-revert には 1.〜2. だけでは対応できない
///    （codex-review 指摘 `PRRT_kwDOTuUCJc6my2IX`・`PRRT_kwDOTuUCJc6mzcKZ`・
///    PR #2351）。最終ファイルを開いた直後に、そのハンドル自身が実際に
///    指しているオブジェクトの所在を `GetFinalPathNameByHandleW` で
///    （経路文字列の再解決ではなくハンドル起点の逆引きで）取得し、
///    `base_dir` から「想定した深さで解決されているか」という深さ数値
///    だけでなく、**`held` 各エントリ自身の実所在（同じくハンドル起点の
///    逆引き）と対応づけて**検証する（P0 是正・codex-review 指摘
///    `PRRT_kwDOTuUCJc6m0J-L`・PR #2351。深さだけの比較は `base_dir`
///    配下の同じ深さの別ディレクトリへ着地した場合を見逃す）。加えて
///    ボリューム識別子（`file_identity`）の突き合わせで別ボリュームへの
///    着地も拒否する。この検証はいずれもハンドルが指すオブジェクトその
///    ものに基づくため、reparse point を事後に元へ戻しても偽装できない。
///
/// 定数は Win32 SDK ヘッダ（`winnt.h`／`winbase.h`。MS Learn
/// "CreateFileA/W"・"File Security and Access Rights"・"File Access Rights
/// Constants"・"GetFileInformationByHandle function"・
/// "GetFinalPathNameByHandleW function"）の値・シグネチャを手書きする。
/// Linux の `O_*` フラグと異なり Win32 API の定数は x86_64／aarch64 で
/// ABI が固定されアーキテクチャ間で値が変わらないため、`libc` 相当の
/// crate を追加しない（`.claude/rules/deps-policy.md`）。定数・
/// `OpenOptionsExt` 経由の呼び出しは std のみで `unsafe` を要しないが、
/// 3. の実体識別（[`file_identity`]）・実所在検証
/// （[`final_real_path`]）は std が未安定化の API
/// （`file_index`／`volume_serial_number`〈`windows_by_handle`〉・
/// `GetFinalPathNameByHandleW` 相当）に依存するため、kernel32.dll への
/// 手書き `extern "system"` 宣言（`unsafe`。FFI 境界に限定。
/// `.claude/rules/coding-rust.md`）で直接呼ぶ。
#[cfg(windows)]
mod win_contained_open {
    use std::ffi::{OsStr, OsString, c_void};
    use std::fs::{File, OpenOptions};
    use std::io;
    use std::os::windows::ffi::OsStringExt;
    use std::os::windows::fs::{MetadataExt, OpenOptionsExt};
    use std::os::windows::io::AsRawHandle;
    use std::path::{Component, Path, PathBuf, Prefix};

    // winnt.h（`CreateFileW` の `dwShareMode`）。`FILE_SHARE_DELETE` は
    // 意図的に含めない: 本モジュールの TOCTOU 防止は「保持中のハンドルは
    // 他プロセスから rename・削除できない」ことに依存するため。
    const FILE_SHARE_READ: u32 = 0x0000_0001;
    const FILE_SHARE_WRITE: u32 = 0x0000_0002;

    // winbase.h（`CreateFileW` の `dwFlagsAndAttributes`）。
    // `FILE_FLAG_BACKUP_SEMANTICS` はディレクトリを開くために必要
    // （MS Learn "CreateFileA/W"）。`FILE_FLAG_OPEN_REPARSE_POINT` は
    // reparse point の最終成分をそのまま開かせ（追跡させない）、開いた
    // 直後の属性検査で検出可能にする。
    const FILE_FLAG_BACKUP_SEMANTICS: u32 = 0x0200_0000;
    const FILE_FLAG_OPEN_REPARSE_POINT: u32 = 0x0020_0000;

    // winnt.h（`GetFileInformationByHandle` 等が返す `dwFileAttributes`）。
    const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x0000_0400;

    // winnt.h（`CreateFileW` の `dwDesiredAccess`）。共有モードの判定に
    // 参加するアクセス権（read／execute／write／delete 系）を含める必要が
    // ある。`FILE_READ_ATTRIBUTES` 単独は共有モード判定に参加しないため、
    // execute 系の `FILE_TRAVERSE`（`FILE_EXECUTE` と同値）を必ず含める
    // （MS Learn "File Access Rights Constants"）。一般的な ACL で
    // `FILE_TRAVERSE` が拒否される場合に `FILE_LIST_DIRECTORY` へ切り替える
    // べきかは実機での確認事項とする（計画 §3.4。PR 申し送り）。
    const FILE_TRAVERSE: u32 = 0x0000_0020;
    const FILE_READ_ATTRIBUTES: u32 = 0x0000_0080;
    const SYNCHRONIZE: u32 = 0x0010_0000;
    const GENERIC_READ: u32 = 0x8000_0000;

    const DIR_ACCESS_MODE: u32 = FILE_TRAVERSE | FILE_READ_ATTRIBUTES | SYNCHRONIZE;
    const DIR_SHARE_MODE: u32 = FILE_SHARE_READ | FILE_SHARE_WRITE;
    // 最終ファイル成分にも `FILE_FLAG_BACKUP_SEMANTICS` を付ける（計画 §3.4
    // の表は最終成分に `FILE_FLAG_OPEN_REPARSE_POINT` のみを挙げていたが、
    // 実装時に判明した必須の追加。CVE 相当の実害ではなく起動失敗の是正）:
    // `location` がディレクトリ・junction（ディレクトリ属性を持つ reparse
    // point）を指す場合、`CreateFileW` は `FILE_FLAG_BACKUP_SEMANTICS`
    // 無しでは `ERROR_ACCESS_DENIED` で **open 自体が失敗する**（MS Learn
    // "CreateFileA/W" の Directories 節。Rust std の `File::open` がディレク
    // トリに対して os error 5 を返す既知の挙動と同根）。open 自体が失敗
    // すると、その後に続く「開いたハンドルの属性を見て reparse
    // point／非ファイルとして分類する」という本モジュールの契約
    // （`directory_as_location_is_rejected`・
    // `junction_as_final_directory_component_is_rejected` が要求する
    // `NotRegularFile`／`ReparsePoint`）を果たせず、単なる `Io` 拒否に
    // 縮退してしまう。`FILE_FLAG_BACKUP_SEMANTICS` は通常ファイルの open
    // には副作用が無い（`SeBackupPrivilege` を持たない呼び出し元でも
    // 通常のアクセスチェックへフォールバックする）ため、最終成分にも
    // 無条件で付与し、unix 版と同じ「まず open→属性で分類」の順序を保つ。
    const FINAL_CUSTOM_FLAGS: u32 = FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT;
    const DIR_CUSTOM_FLAGS: u32 = FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT;

    // winbase.h（`GetFinalPathNameByHandleW` の `dwFlags`）。両方とも値 0
    // （既定）で、`Path::canonicalize`（std 内部実装が同じ API を同じ既定
    // フラグで呼ぶ）が返す verbatim 絶対パスと同じ表記形式
    // （`\\?\C:\...`・正規化済み・DOS ドライブレター形式）を得るために
    // 明示しておく（値の意味を自明にするための命名であり、実効値は
    // 変えない）。
    const FILE_NAME_NORMALIZED: u32 = 0x0;
    const VOLUME_NAME_DOS: u32 = 0x0;
    const GET_FINAL_PATH_FLAGS: u32 = FILE_NAME_NORMALIZED | VOLUME_NAME_DOS;

    /// `GetFileInformationByHandle`（winbase.h）が書き込む構造体
    /// （`BY_HANDLE_FILE_INFORMATION`）を 1:1 再現する。C ABI ヘッダの
    /// フィールド順序・型幅（`u32`／`FILETIME` は winnt.h の 32bit ペア）を
    /// そのまま踏襲し、`#[repr(C)]` でレイアウトを固定する。
    #[repr(C)]
    struct RawFiletime {
        dw_low_date_time: u32,
        dw_high_date_time: u32,
    }

    #[repr(C)]
    struct ByHandleFileInformation {
        file_attributes: u32,
        creation_time: RawFiletime,
        last_access_time: RawFiletime,
        last_write_time: RawFiletime,
        volume_serial_number: u32,
        file_size_high: u32,
        file_size_low: u32,
        number_of_links: u32,
        file_index_high: u32,
        file_index_low: u32,
    }

    // SAFETY: この `extern "system"` 宣言は kernel32.dll が公開する Win32
    // API（`GetFileInformationByHandle`／`GetFinalPathNameByHandleW`。
    // MS Learn "GetFileInformationByHandle function"・
    // "GetFinalPathNameByHandleW function"）のシグネチャと一致させている
    // （戻り値・引数の型幅・呼び出し規約 `extern "system"` は winbase.h の
    // 宣言と 1:1 対応する）。std は `file_index`／`volume_serial_number`
    // （`windows_by_handle`）・`GetFinalPathNameByHandleW` 相当のいずれも
    // 1.98.1 時点で安定化していないため（本モジュール冒頭ドキュメント
    // 参照）、新規クレートを追加せず（`.claude/rules/deps-policy.md`。
    // kernel32.dll は Windows の全プロセスが常にリンクする基盤 DLL のため
    // `libc`（unix 側。第 10 区分）のような動的ロードも不要）手書き
    // `extern` で直接呼ぶ。`unsafe` は本ブロックが宣言する 2 関数の呼び
    // 出し箇所（[`file_identity`]・[`final_real_path`]）に限定する
    // （`.claude/rules/coding-rust.md`「`unsafe` は FFI 境界等の必要
    // 最小限に留め、理由をコメントで明記」）。
    unsafe extern "system" {
        fn GetFileInformationByHandle(
            h_file: *mut c_void,
            lp_file_information: *mut ByHandleFileInformation,
        ) -> i32;

        fn GetFinalPathNameByHandleW(
            h_file: *mut c_void,
            lp_sz_file_path: *mut u16,
            cch_file_path: u32,
            dw_flags: u32,
        ) -> u32;
    }

    /// `file` の実体識別子（`(dwVolumeSerialNumber, nFileIndex)`。unix の
    /// `(dev, ino)` 相当）を、ハンドル自身への `GetFileInformationByHandle`
    /// 照会で得る（経路文字列を再解決しない）。P0 是正
    /// （codex-review 指摘 `PRRT_kwDOTuUCJc6my2Ie`・PR #2351）: `FileKey` を
    /// 経路文字列ベースのままにすると、ハードリンク・NTFS 8.3 短縮名等の
    /// 別名パスで同一ファイルを参照する `location` が異なるキーへ分散し、
    /// [`super::OverlappingRegion`] 検証（区間重複検査）をすり抜けるため。
    pub(super) fn file_identity(file: &File) -> io::Result<(u32, u64)> {
        // SAFETY: `handle` はこの呼び出しの生存期間中有効な `File` から
        // 取得した生ハンドル。`info` は `size_of::<ByHandleFileInformation>()`
        // 分の初期化済み（`zeroed`）バッファで、`GetFileInformationByHandle`
        // は自身の構造体サイズを超えて書き込まない契約（MS Learn）。
        // 戻り値 0 は失敗を意味し、その場合 `info` の内容は使わず
        // `io::Error::last_os_error()` を返す。
        let handle = file.as_raw_handle();
        let mut info = std::mem::MaybeUninit::<ByHandleFileInformation>::zeroed();
        let ok = unsafe { GetFileInformationByHandle(handle, info.as_mut_ptr()) };
        if ok == 0 {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: 上の呼び出しが非 0（成功）を返したため、`info` は
        // `GetFileInformationByHandle` によって完全に初期化済み。
        let info = unsafe { info.assume_init() };
        let index = ((info.file_index_high as u64) << 32) | info.file_index_low as u64;
        Ok((info.volume_serial_number, index))
    }

    /// `file` が指すオブジェクトの「今開いているハンドルそのものが実際に
    /// 参照している」正規化済み絶対パスを `GetFinalPathNameByHandleW` で
    /// 取得する（経路文字列の再解決ではなく、ハンドルが指すオブジェクトを
    /// 起点に OS が逆引きする）。[`resolve_and_open`] が最終ファイルを
    /// 開いた直後に呼び、`base_dir` 配下へ実際に着地したことを検証する
    /// （P0 是正 ×2。codex-review 指摘 `PRRT_kwDOTuUCJc6my2IX`・
    /// `PRRT_kwDOTuUCJc6mzcKZ`・PR #2351。モジュール doc「TOCTOU の根拠」
    /// 節 3. 参照）。
    fn final_real_path(file: &File) -> io::Result<PathBuf> {
        let handle = file.as_raw_handle();
        // 初回は MAX_PATH で試し、不足なら API が返す必要長へ拡張して
        // 再試行する（MS Learn "GetFinalPathNameByHandleW function" の
        // 契約: バッファ不足時は必要文字数〈終端 NUL 込み〉を返す）。
        // 反復回数に上限を設け、想定外の戻り値の反復（本来起きない）で
        // 無限ループに陥らないようにする（fail-closed。
        // `.claude/rules/coding-rust.md`「本番経路で `unwrap()`/`expect()`
        // を使わない」と同じ精神で、ここでは無限ループを避ける）。
        let mut buf: Vec<u16> = vec![0u16; 260];
        for _ in 0..8 {
            // SAFETY: `buf` は `buf.len()` 個の `u16` を保持する有効な
            // バッファで、`cch_file_path` に `buf.len()` を渡すため API が
            // バッファ長を超えて書き込むことはない。`handle` は呼び出し元
            // `file` の生存期間中有効。
            let n = unsafe {
                GetFinalPathNameByHandleW(
                    handle,
                    buf.as_mut_ptr(),
                    buf.len() as u32,
                    GET_FINAL_PATH_FLAGS,
                )
            };
            if n == 0 {
                return Err(io::Error::last_os_error());
            }
            let n = n as usize;
            if n < buf.len() {
                buf.truncate(n);
                return Ok(PathBuf::from(OsString::from_wide(&buf)));
            }
            // バッファ不足: `n` は終端 NUL を含む必要文字数。
            buf.resize(n, 0);
        }
        Err(io::Error::from(io::ErrorKind::Unsupported))
    }

    /// `handle`（最終ファイル）が実際に `base_dir`（[`super::BaseDirHandle`]。
    /// ボリュームルートから検証済みの祖先チェーンを保持）配下へ、`held`
    /// （本 location 専用に開いた途中ディレクトリハンドル。`resolve_and_open`
    /// が `parts` の走査で積む）を実際に経由して着地したことを検証する。
    ///
    /// **P0 是正（codex-review 指摘 `PRRT_kwDOTuUCJc6m0J-L`・PR #2351）**:
    /// 旧実装は `real`（[`final_real_path`] が返す実所在）を `base` からの
    /// 深さ（成分数）だけで判定していた。これは、`base_dir` 配下の
    /// 「同じ深さの別ディレクトリ」（例: 成分 `A` を一時的に `base_dir`
    /// 直下の別ディレクトリ `B` への junction に差し替えて最終ファイルを
    /// 開かせ、祖先再検査（モジュール doc 2.）が走る前に `A` を元へ戻す
    /// flip-and-revert）を見逃す。`A` は既に `held` へ積まれ保持済みの
    /// ハンドルであり、その実体（カーネルオブジェクト）は差し替えられない
    /// ため、`A` 自身に junction タグを立てても `held` に積んだハンドル
    /// そのものは同じオブジェクトを指し続ける。本関数はこの性質を使い、
    /// `held[i]` 自身の実所在（`final_real_path(&held[i])`）を
    /// `real_path`（`handle` の実所在）の対応する深さの祖先成分列と
    /// 突き合わせる。両者はいずれも `GetFinalPathNameByHandleW` の同一
    /// フラグでの出力同士の比較のため、大小文字・8.3 短縮名等の表記ゆれ
    /// （NTFS は既定で大小文字を区別しない）に依らず安全に比較できる
    /// （`parts`〈呼び出し元が渡す生のパス文字列〉とは比較しない）。
    /// 加えてボリューム識別子（[`file_identity`] の
    /// `volume_serial_number`）を `handle` と `base_dir` の祖先チェーン先頭
    /// （ボリュームルートハンドル）とで突き合わせ、ボリュームマウント
    /// ポイント経由で別ボリュームへ着地した場合（`GetFinalPathNameByHandleW`
    /// の `VOLUME_NAME_DOS` 出力がドライブレターの無いボリュームでどう
    /// 振る舞うかは実機未検証のため、文字列比較だけに依存しない）も拒否
    /// する。
    fn verify_final_path_within_base_dir(
        base_dir: &super::BaseDirHandle,
        held: &[File],
        handle: &File,
        real_path: &Path,
    ) -> io::Result<bool> {
        let Some(root) = base_dir.chain.first() else {
            // 不変条件検査（fail-closed）: `open_base_dir_handle` は
            // ボリュームルートハンドルを無条件で `chain` の先頭へ積むため
            // 空にはならない。万一崩れていた場合は判定不能として拒否する。
            return Err(io::Error::from(io::ErrorKind::Unsupported));
        };
        let (final_volume, _) = file_identity(handle)?;
        let (root_volume, _) = file_identity(root)?;
        if final_volume != root_volume {
            return Ok(false);
        }

        let real_rel = match real_path.strip_prefix(&base_dir.base) {
            Ok(rel) => rel,
            Err(_) => return Ok(false),
        };
        // 深さ（成分数）は `held`（途中ディレクトリ。`parts` の最終成分を
        // 除いた分）＋最終ファイル自身の 1 成分。
        if real_rel.components().count() != held.len() + 1 {
            return Ok(false);
        }

        for (i, anc) in held.iter().enumerate() {
            let anc_real = final_real_path(anc)?;
            let anc_rel = match anc_real.strip_prefix(&base_dir.base) {
                Ok(rel) => rel,
                Err(_) => return Ok(false),
            };
            // `anc_rel`（`held[i]` 自身の実所在）が `real_rel`（最終
            // ファイルの実所在）の先頭 i+1 成分と一致し、かつ `anc_rel`
            // 自身の深さもちょうど i+1 であることを要求する。深さも見る
            // のは、`anc_rel` が `real_rel` の先頭 i+1 成分と偶然前方一致
            // しつつ実際には異なる深さのオブジェクトである退化ケースを
            // 排除するため。
            let matches_prefix = anc_rel.components().eq(real_rel.components().take(i + 1));
            if !matches_prefix || anc_rel.components().count() != i + 1 {
                return Ok(false);
            }
        }

        Ok(true)
    }

    /// ディレクトリ成分（`is_dir = true`）または最終ファイル成分
    /// （`is_dir = false`）を、モジュール doc の共有モード・フラグで
    /// 開く。`.read(true)` は `access_mode` が上書きするため実効を
    /// 持たないが、`OpenOptions` の一般契約（read/write のいずれかを
    /// 要求する経路がある）に備えて明示しておく。
    fn open_component(path: &Path, is_dir: bool) -> io::Result<File> {
        let mut opts = OpenOptions::new();
        opts.read(true);
        if is_dir {
            opts.access_mode(DIR_ACCESS_MODE)
                .share_mode(DIR_SHARE_MODE)
                .custom_flags(DIR_CUSTOM_FLAGS);
        } else {
            opts.access_mode(GENERIC_READ)
                .share_mode(FILE_SHARE_READ)
                .custom_flags(FINAL_CUSTOM_FLAGS);
        }
        opts.open(path)
    }

    /// `handle` 自身（パスではない）に対する属性照会で reparse point
    /// 属性を検査する。
    fn is_reparse_point(handle: &File) -> io::Result<bool> {
        Ok(handle.metadata()?.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0)
    }

    /// [`resolve_and_open`] の失敗理由。呼び出し元（`super::
    /// resolve_and_open`）が `ExternalDataError::InvalidLocation` の
    /// `reason`（`ReparsePoint`／`NotRegularFile`／`OutsideBaseDir`）と
    /// `Io` を区別できるよう、単なる `(io::Error, bool)` ではなく複数分岐に
    /// する（reparse point 検出・「open は成功したが通常ファイルではない」
    /// 検出・「実際に開かれたハンドルが `base_dir` の外を指していた」検出
    /// を、いずれも `io::Error` の種別へ押し込めずに表現するため）。
    pub(super) enum OpenError {
        Reparse,
        NotRegularFile,
        /// 最終ファイルハンドルの実際の所在（[`final_real_path`]）が
        /// `base_dir` 配下・`held` の実所在と対応する経路から外れていた
        /// （TOCTOU 是正。[`verify_final_path_within_base_dir`] の
        /// ドキュメント参照。codex-review 指摘 `PRRT_kwDOTuUCJc6my2IX`・
        /// `PRRT_kwDOTuUCJc6mzcKZ`・`PRRT_kwDOTuUCJc6m0J-L`・PR #2351）。
        EscapedBaseDir,
        Io(io::Error),
    }

    impl From<io::Error> for OpenError {
        fn from(e: io::Error) -> Self {
            OpenError::Io(e)
        }
    }

    /// `base_dir_canonical`（`Path::canonicalize` が返す verbatim 絶対パス
    /// `\\?\C:\...`）を受け取り、ボリュームルートから `base_dir` までの
    /// 各ディレクトリ成分を開いて [`super::BaseDirHandle`] として返す
    /// （モジュール doc 1.）。`VerbatimDisk`（`\\?\C:\`）以外の先頭成分
    /// （UNC・Volume GUID 等）は拒否する: UNC はサーバ側で junction が
    /// 評価され共有モードの意味論も異なり、Volume GUID パスはフォルダに
    /// マウントされたボリュームを経由しうるため（計画 §3.3）。
    pub(super) fn open_base_dir_handle(
        base_dir_canonical: &Path,
    ) -> io::Result<super::BaseDirHandle> {
        let mut components = base_dir_canonical.components();
        let Some(Component::Prefix(prefix)) = components.next() else {
            return Err(io::Error::from(io::ErrorKind::Unsupported));
        };
        if !matches!(prefix.kind(), Prefix::VerbatimDisk(_)) {
            return Err(io::Error::from(io::ErrorKind::Unsupported));
        }
        if !matches!(components.next(), Some(Component::RootDir)) {
            return Err(io::Error::from(io::ErrorKind::Unsupported));
        }

        // verbatim パスは `/` を区切り文字として扱わないため、文字列連結
        // ではなく `PathBuf::push` で `\` 区切りに連結する（計画 §3.3）。
        let mut cur_path = PathBuf::from(prefix.as_os_str());
        cur_path.push(Component::RootDir.as_os_str());
        let mut chain = Vec::new();

        // ボリュームルート自身（`\\?\C:\`）も祖先チェーンへ含めて開く
        // （計画 §3.3「ボリュームルートから base_dir までの各成分」）。
        let root_handle = open_component(&cur_path, true)?;
        if is_reparse_point(&root_handle)? {
            return Err(io::Error::from(io::ErrorKind::Unsupported));
        }
        if !root_handle.metadata()?.is_dir() {
            return Err(io::Error::from(io::ErrorKind::Unsupported));
        }
        chain.push(root_handle);

        for comp in components {
            let Component::Normal(name) = comp else {
                return Err(io::Error::from(io::ErrorKind::Unsupported));
            };
            cur_path.push(name);
            let handle = open_component(&cur_path, true)?;
            if is_reparse_point(&handle)? {
                return Err(io::Error::from(io::ErrorKind::Unsupported));
            }
            if !handle.metadata()?.is_dir() {
                return Err(io::Error::from(io::ErrorKind::Unsupported));
            }
            chain.push(handle);
        }

        // 不変条件検査（fail-closed。coding-rust.md の本番経路
        // `unwrap`/`expect` 禁止と同じ理由で `debug_assert!` ではなく
        // 型付きエラーで表面化させる）: `cur_path` は
        // `PathBuf::push(Component::RootDir.as_os_str())` を含む逐次連結で
        // `base_dir_canonical` を再構築したものであり、両者が一致しない
        // 場合は `PathBuf::push` の prefix/root 特殊処理（本関数冒頭
        // コメント参照）への前提が崩れている。
        if cur_path != base_dir_canonical {
            return Err(io::Error::from(io::ErrorKind::Unsupported));
        }

        Ok(super::BaseDirHandle {
            base: cur_path,
            chain,
        })
    }

    /// [`super::resolve_and_open`] の Windows 実装本体。`base_dir`
    /// （[`open_base_dir_handle`] が返した祖先チェーン）を起点に `parts`
    /// をフルパスの逐次オープンで辿る。dirfd 起点の相対オープンが無い
    /// ため、途中成分ごとに本 location 専用の祖先ハンドルを一時的に保持
    /// し、最終ファイルを開いた後の事後チェックが終わるまで rename・
    /// 削除・reparse point 化を防ぐ（モジュール doc 1.〜2.）。失敗理由は
    /// [`OpenError`]（`Reparse`／`NotRegularFile`／`Io`）で呼び出し元
    /// （`super::resolve_and_open`）へそのまま伝える。
    pub(super) fn resolve_and_open(
        base_dir: &super::BaseDirHandle,
        parts: &[&OsStr],
    ) -> Result<super::OpenFile, OpenError> {
        if parts.is_empty() {
            return Err(OpenError::Io(io::Error::from(io::ErrorKind::InvalidInput)));
        }
        let last_idx = parts.len() - 1;
        let mut cur_path = base_dir.base.clone();
        // 本 location（`parts`）専用の途中ディレクトリハンドル。
        // `base_dir.chain`（`plan`／`load` を通して保持）とは別に、この
        // 呼び出しの間だけ保持し、最終ファイルの事後チェックが終わったら
        // 破棄する。
        let mut held: Vec<File> = Vec::new();

        for (i, part) in parts.iter().enumerate() {
            cur_path.push(part);
            let is_dir_component = i != last_idx;
            // ディレクトリ・junction（ディレクトリ属性を持つ reparse
            // point）成分は `FILE_FLAG_BACKUP_SEMANTICS`（`open_component`
            // が最終成分にも付与済み）が無いと open 自体が
            // `ERROR_ACCESS_DENIED` で失敗しうる（`FINAL_CUSTOM_FLAGS`
            // 定義コメント参照）。open が失敗した場合は「開いたハンドルの
            // 属性で reparse point／非ディレクトリ・非ファイルを分類する」
            // 通常経路に到達できないため、そのまま `Io` として伝える
            // （実運用ではまず起きない想定だが fail-closed に拒否する）。
            let handle = open_component(&cur_path, is_dir_component)?;
            let attrs = handle.metadata()?;
            if attrs.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0 {
                return Err(OpenError::Reparse);
            }
            if is_dir_component {
                if !attrs.is_dir() {
                    return Err(OpenError::NotRegularFile);
                }
                held.push(handle);
                continue;
            }

            if !attrs.is_file() {
                return Err(OpenError::NotRegularFile);
            }

            // 事後チェック（モジュール doc 2.）: 最終ファイルを開いた後に、
            // `base_dir.chain`（祖先チェーン全体）＋本 location 専用に
            // 新規で開いた途中ディレクトリ（`held`）の全ハンドルの属性を
            // 再取得し、reparse point が付いていないことを確認する。
            for anc in base_dir.chain.iter().chain(held.iter()) {
                let m = anc.metadata()?;
                if m.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0 {
                    return Err(OpenError::Reparse);
                }
            }

            // 事後チェック（モジュール doc「TOCTOU の根拠」節 3.）: 上の
            // 祖先再検査は「元に戻された（flip-and-revert）」reparse point
            // 化を見逃しうる（codex-review 指摘 `PRRT_kwDOTuUCJc6my2IX`・
            // `PRRT_kwDOTuUCJc6mzcKZ`・`PRRT_kwDOTuUCJc6m0J-L`・PR #2351）。
            // 祖先ハンドルの属性ではなく、実際に開かれた `handle` 自身が
            // 指すオブジェクトの所在を `GetFinalPathNameByHandleW` で
            // 逆引きし、`held`（本 location 専用に保持中の各祖先ハンドル）
            // 自身の実所在と対応づけて検証する（`is_within_base_dir` の
            // 深さのみの比較では、`base_dir` 配下の同じ深さの別
            // ディレクトリへ着地した場合を見逃すため。
            // [`verify_final_path_within_base_dir`] のドキュメント参照）。
            let real_path = final_real_path(&handle)?;
            if !verify_final_path_within_base_dir(base_dir, &held, &handle, &real_path)? {
                return Err(OpenError::EscapedBaseDir);
            }

            let snapshot = super::FileSnapshot::from_metadata(&attrs);
            // `held`（本 location 専用の途中ディレクトリハンドル）は事後
            // チェックを終えたのでここで drop する（呼び出し元は最終
            // ファイルのハンドルのみを使う。モジュール doc 1. の保持は
            // `base_dir.chain` が引き続き担う）。
            drop(held);
            return Ok(super::OpenFile {
                file: handle,
                snapshot,
            });
        }

        Err(OpenError::Io(io::Error::from(io::ErrorKind::InvalidInput)))
    }
}

/// `base_dir` を起点に external data を解決するためのハンドル。unix では
/// [`no_follow_open::open_base_dir`] が開いたディレクトリ fd（`plan`・
/// `load` を通して 1 つだけ保持する）。
#[cfg(unix)]
type BaseDirHandle = File;

/// Windows 版（イシュー #2349）。ボリュームルートから `base_dir` までの
/// 祖先ディレクトリのハンドル連鎖（`win_contained_open::open_base_dir_handle`
/// が開く。`plan`・`load` を通して 1 つだけ保持する）。std に `openat`
/// 相当（ディレクトリハンドル起点の相対オープン）が無いため、unix の
/// ディレクトリ fd 単体とは異なり祖先全体を保持する構成になる
/// （`win_contained_open` モジュール doc「TOCTOU の根拠」節）。
#[cfg(windows)]
struct BaseDirHandle {
    /// `base_dir` の正規化済み verbatim 絶対パス（`Path::canonicalize` が
    /// 返す `\\?\C:\...` 形式）。`resolve_and_open` がこの下へ `location` の
    /// 各成分を `PathBuf::push` で連結する起点。
    base: PathBuf,
    /// ボリュームルート（`\\?\C:\`）から `base` までの各ディレクトリ
    /// 成分を開いたまま保持するハンドル（先頭がボリュームルート、末尾が
    /// `base` 自身）。DELETE を含まない共有モードで開くため、保持している
    /// 間はこれらの成分を他プロセスから rename・削除できない
    /// （`win_contained_open` モジュール doc 1.）。
    chain: Vec<File>,
}

/// unix・Windows 以外では安全な経路解決手段を持たず `resolve_and_open` が
/// 常に拒否するため、何も開かないゼロサイズのマーカーとする
/// （`plan`／`load` の制御フローを unix・Windows と共通化するためだけに
/// 存在する）。
#[cfg(not(any(unix, windows)))]
struct BaseDirHandle;

/// `base_dir_canonical` を external data 解決の起点として開く。`plan` が
/// 最初の external テンソルに到達した時点でのみ呼ぶ（遅延オープン。
/// `plan` 内コメント参照）。
#[cfg(unix)]
fn open_base_dir_handle(base_dir_canonical: &Path) -> Result<BaseDirHandle, ExternalDataError> {
    no_follow_open::open_base_dir(base_dir_canonical)
        .map_err(|e| ExternalDataError::InvalidBaseDir { kind: e.kind() })
}

/// Windows 版。[`win_contained_open::open_base_dir_handle`] へ委譲する。
#[cfg(windows)]
fn open_base_dir_handle(base_dir_canonical: &Path) -> Result<BaseDirHandle, ExternalDataError> {
    win_contained_open::open_base_dir_handle(base_dir_canonical)
        .map_err(|e| ExternalDataError::InvalidBaseDir { kind: e.kind() })
}

#[cfg(not(any(unix, windows)))]
fn open_base_dir_handle(_base_dir_canonical: &Path) -> Result<BaseDirHandle, ExternalDataError> {
    Ok(BaseDirHandle)
}

/// `base_dir_file`（[`no_follow_open::open_base_dir`] が開いたディレクトリ
/// fd）を起点に、既に検証済みの `parts`（[`validate_location_string`] の
/// 戻り値。`plan` が呼び出し元で 1 度だけ検証し、同一 location への
/// 2 件目以降の呼び出しをキャッシュで省略できるようにするため、検証を
/// 呼び出し元へ分離してある）を解決してファイルを開く。経路の途中を
/// 含めシンボリックリンクを拒否し、`openat2`／`O_NOFOLLOW` による
/// ディレクトリハンドル連鎖オープンで「検証した経路そのもの」を開くことを
/// 保証する（A2・P0 対応。discussion_r4119392011・イシュー #2347）。
/// パス 1（`plan`）の初回解決と、パス 2（`load`）の読み込み用再 open の
/// 両方から同じ `base_dir_file` を起点に呼ばれる（再 open でも経路文字列を
/// 再解決せず、同じシンボリックリンク拒否手段を使う）。
#[cfg(unix)]
fn resolve_and_open(
    tensor_name: &str,
    base_dir_file: &BaseDirHandle,
    parts: &[&std::ffi::OsStr],
) -> Result<OpenFile, ExternalDataError> {
    #[cfg(target_os = "linux")]
    let open_result = match no_follow_open::open_chain_openat2(base_dir_file, parts) {
        no_follow_open::Openat2Outcome::Opened(f) => Ok(f),
        no_follow_open::Openat2Outcome::Failed(e, is_symlink) => Err((e, is_symlink)),
        // `openat2` 未対応環境（古いカーネル・seccomp 等）: 成分ごと逐次
        // `openat(O_NOFOLLOW)` 方式へフォールバックする。
        no_follow_open::Openat2Outcome::Unsupported => {
            no_follow_open::open_chain_component_walk(base_dir_file, parts)
        }
    };
    #[cfg(not(target_os = "linux"))]
    let open_result = no_follow_open::open_chain_component_walk(base_dir_file, parts);

    let file = open_result.map_err(|(e, is_symlink)| {
        if is_symlink {
            ExternalDataError::InvalidLocation {
                tensor_name: cap_name(tensor_name),
                reason: LocationRejectReason::Symlink,
            }
        } else if e.kind() == std::io::ErrorKind::NotADirectory
            || e.raw_os_error() == Some(libc::ENOTDIR)
        {
            ExternalDataError::InvalidLocation {
                tensor_name: cap_name(tensor_name),
                reason: LocationRejectReason::NotRegularFile,
            }
        } else {
            ExternalDataError::Io {
                tensor_name: cap_name(tensor_name),
                kind: e.kind(),
            }
        }
    })?;

    let meta = file.metadata().map_err(|e| ExternalDataError::Io {
        tensor_name: cap_name(tensor_name),
        kind: e.kind(),
    })?;
    if !meta.is_file() {
        return Err(ExternalDataError::InvalidLocation {
            tensor_name: cap_name(tensor_name),
            reason: LocationRejectReason::NotRegularFile,
        });
    }

    Ok(OpenFile {
        file,
        snapshot: FileSnapshot::from_metadata(&meta),
    })
}

/// Windows 版（イシュー #2349）。[`win_contained_open::resolve_and_open`]
/// （祖先ディレクトリのハンドル連鎖保持＋開いたハンドル自身の reparse
/// point 属性検査による封じ込めオープン）へ委譲し、結果を
/// [`ExternalDataError`] へ写像する。パス 1（`plan`）の初回解決と、
/// パス 2（`load`）の読み込み用再 open の両方から同じ `base_dir_file` を
/// 起点に呼ばれる契約は unix 版と同じ。
#[cfg(windows)]
fn resolve_and_open(
    tensor_name: &str,
    base_dir_file: &BaseDirHandle,
    parts: &[&std::ffi::OsStr],
) -> Result<OpenFile, ExternalDataError> {
    win_contained_open::resolve_and_open(base_dir_file, parts).map_err(|e| match e {
        win_contained_open::OpenError::Reparse => ExternalDataError::InvalidLocation {
            tensor_name: cap_name(tensor_name),
            reason: LocationRejectReason::ReparsePoint,
        },
        win_contained_open::OpenError::NotRegularFile => ExternalDataError::InvalidLocation {
            tensor_name: cap_name(tensor_name),
            reason: LocationRejectReason::NotRegularFile,
        },
        win_contained_open::OpenError::EscapedBaseDir => ExternalDataError::InvalidLocation {
            tensor_name: cap_name(tensor_name),
            reason: LocationRejectReason::OutsideBaseDir,
        },
        win_contained_open::OpenError::Io(e) => ExternalDataError::Io {
            tensor_name: cap_name(tensor_name),
            kind: e.kind(),
        },
    })
}

/// [`resolve_and_open`] の unix・Windows 以外向け実装。
///
/// `openat`／`openat2`・Windows 封じ込めオープンのいずれの、ディレクトリ
/// ハンドル起点でシンボリックリンク／reparse point を拒否しながら経路
/// 解決する安全な手段も持たないため、`security.md` の A08（整合性の
/// 迂回経路を作らない）・本 crate の fail-closed 方針（イシュー #2347
/// タイトルのとおり external data 読み込みは fail-closed 前提）に従い、
/// **この関数は常にファイルを開かず拒否する**。
///
/// `location` の文法検証（`Path::components()` 等の OS 非依存な範囲）は
/// unix・Windows と共通に呼び出し元（`plan`）が本関数より前に行う:
/// 文字列自体が不正（絶対パス・`..`・NUL 等）な場合は具体的な理由を持つ
/// `InvalidLocation` が先に返り、文法上は正当な `location` のみが本関数へ
/// 到達して `UnsupportedPlatformForSecureResolve` になる。
#[cfg(not(any(unix, windows)))]
fn resolve_and_open(
    tensor_name: &str,
    _base_dir: &BaseDirHandle,
    _parts: &[&std::ffi::OsStr],
) -> Result<OpenFile, ExternalDataError> {
    Err(ExternalDataError::UnsupportedPlatformForSecureResolve {
        tensor_name: cap_name(tensor_name),
    })
}

/// ASCII 数字のみからなる非空文字列として `u64` を解釈する（符号・空白・
/// 先頭 `+` は拒否。checked 変換）。
fn parse_decimal_u64(
    raw: &str,
    tensor_name: &str,
    field: &'static str,
) -> Result<u64, ExternalDataError> {
    if raw.is_empty() || !raw.bytes().all(|b| b.is_ascii_digit()) {
        return Err(ExternalDataError::InvalidNumber {
            tensor_name: cap_name(tensor_name),
            field,
        });
    }
    raw.parse::<u64>()
        .map_err(|_| ExternalDataError::InvalidNumber {
            tensor_name: cap_name(tensor_name),
            field,
        })
}

/// `data_type` から要素サイズ（バイト）を返す。`graph::decode_tensor` が
/// 対応する 4 型（FLOAT／INT64／BOOL／FLOAT16）とだけ整合させる。
fn element_size(tensor_name: &str, data_type: i32) -> Result<u64, GraphError> {
    match data_type {
        t if t == super::proto::data_type::FLOAT => Ok(4),
        t if t == super::proto::data_type::INT64 => Ok(8),
        t if t == super::proto::data_type::BOOL => Ok(1),
        t if t == super::proto::data_type::FLOAT16 => Ok(2),
        other => Err(GraphError::UnknownDataType {
            tensor_name: tensor_name.to_string(),
            data_type: other,
        }),
    }
}

/// 要素数 `count` の空 `Vec<T>` を失敗可能確保（`try_reserve_exact`）で
/// 用意する（PR #2348 codex P0 是正。security.md A04）。
///
/// external data 由来の長さは `plan` で `max_total_bytes` 以内に抑えられて
/// いるが、既定上限（64 GiB）は利用可能メモリを超えうるため、`vec![..; n]`・
/// `Vec::with_capacity(n)`・`collect` のような無条件確保（失敗時は
/// `handle_alloc_error` でプロセスが abort する）を使わず、失敗を
/// [`ExternalDataError::AllocationFailed`] として返す。`count *
/// size_of::<T>()` が `isize::MAX` を超える場合、`try_reserve_exact` は
/// アロケータを呼ばずに `CapacityOverflow` で失敗するため、巨大な宣言長も
/// 確保を試みる前に同じ variant で拒否される。`bytes` は診断用の要求
/// バイト数（`u64` で飽和計算）。
///
/// 確保本体は読み込み・実行・export 経路で共有する
/// `fallible_alloc::try_alloc_vec` に委譲し、本関数は失敗を external data の
/// 型付きエラーへ写像する（テンソル名は他の `ExternalDataError` と同じく
/// [`cap_name`] で上限付きにする）。
fn try_alloc_vec<T>(tensor_name: &str, count: usize) -> Result<Vec<T>, ExternalDataError> {
    super::fallible_alloc::try_alloc_vec::<T>(tensor_name, count).map_err(alloc_failure_to_external)
}

/// `fallible_alloc::AllocFailure` → [`ExternalDataError::AllocationFailed`]。
fn alloc_failure_to_external(f: super::fallible_alloc::AllocFailure) -> ExternalDataError {
    ExternalDataError::AllocationFailed {
        tensor_name: cap_name(&f.tensor_name),
        bytes: f.bytes,
    }
}

/// `load` が 1 区間（`length` バイト）を読み込むための空バッファを失敗可能
/// 確保で用意する（容量は `length` ちょうど・長さ 0）。`length` が `usize`
/// に収まらない場合（32bit ターゲットで `max_total_bytes` を大きく設定した
/// 場合等）も内部不変条件違反ではなく「このプロセスでは確保できない
/// サイズ」として [`ExternalDataError::AllocationFailed`] を返す。
fn alloc_region_buf(tensor_name: &str, length: u64) -> Result<Vec<u8>, ExternalDataError> {
    let len = usize::try_from(length).map_err(|_| ExternalDataError::AllocationFailed {
        tensor_name: cap_name(tensor_name),
        bytes: length,
    })?;
    try_alloc_vec::<u8>(tensor_name, len)
}

/// `file` の現在位置から `length` バイトを、[`alloc_region_buf`] で失敗可能
/// 確保したバッファへ読み込む（`load` の 1 区間分）。
///
/// ゼロ初期化（`vec![0u8; n]` → `read_exact`）による二重書き込みを避け、
/// かつ未初期化メモリを `unsafe` で扱わないため、`Read::take(length)` で
/// 読み込み量を `length` に制限したうえで `read_to_end` へ渡す。容量は
/// 事前に `length` ちょうど確保済みで、`read_to_end` は容量ちょうどまで
/// 埋まった時点で小さなスタック上の探査読み込みにより EOF を確かめてから
/// 返るため再確保しない（`Take` の上限で EOF になる。単体テスト
/// `read_region_fills_exact_capacity_without_realloc` で容量不変を実測
/// 固定）。`File` に直接 `read_to_end` しない理由: `File` 向けの特殊化は
/// 残りファイル長を基に追加確保しうるため、上限を `length` に縛る `Take`
/// を必ず挟む。
///
/// 読み込めたバイト数が `length` に満たない（照合通過直後〜読み込みの間に
/// truncate された等）場合は、旧実装（`read_exact`）と同じ
/// `ExternalDataError::Io { kind: UnexpectedEof }` で fail-closed に拒否する
/// （モジュール doc「TOCTOU の論拠」節の契約を維持）。
fn read_region<R: Read>(
    file: &mut R,
    tensor_name: &str,
    length: u64,
) -> Result<Vec<u8>, ExternalDataError> {
    let mut buf = alloc_region_buf(tensor_name, length)?;
    let io_err = |kind: std::io::ErrorKind| ExternalDataError::Io {
        tensor_name: cap_name(tensor_name),
        kind,
    };
    let read = file
        .take(length)
        .read_to_end(&mut buf)
        .map_err(|e| io_err(e.kind()))?;
    if read as u64 != length {
        return Err(io_err(std::io::ErrorKind::UnexpectedEof));
    }
    Ok(buf)
}

/// external data から inline 化した initializer 1 件を、失敗可能確保で
/// `RawTensor` へ復号する（`build_graph_with_external_data` 専用。PR #2348
/// codex P0 是正）。
///
/// `graph::decode_tensor`（バイト列入口と共有。A6 により 1 バイトも変更
/// しない）は `collect` による無条件確保で要素 Vec を作るため、external
/// data 由来の巨大テンソルでは確保失敗がプロセス abort になる。本関数は
/// 同じ変換（リトルエンディアン。BOOL は 1 バイト/要素・非ゼロ→true）を
/// [`try_alloc_vec`] で確保した Vec へ行い、`t.raw_data` を
/// `mem::take` で取り出して復号直後に解放する（raw と復号後の要素 Vec が
/// 同時に存在するのは当該テンソル 1 件分だけになる）。
///
/// 長さ・形状の検証は `plan` で済んでいる（`element_count` による dims の
/// 非負性・積のオーバーフロー拒否、`length == expected_bytes`、`data_type`
/// が [`element_size`] の 4 型のいずれか）。本関数はそれを前提にしつつ、
/// 防御的にバイト長と要素サイズの整合（`as_chunks` の余り）を再検査し、
/// 崩れていれば `Internal` で返す。`decode_tensor` との出力一致は単体テスト
/// `try_decode_matches_decode_tensor_for_all_dtypes` で固定する。
fn try_decode_external_initializer(
    t: &mut TensorProto,
) -> Result<super::graph::RawTensor, GraphError> {
    use super::graph::RawTensor;
    use super::proto::data_type;

    let raw = std::mem::take(&mut t.raw_data);
    let name = t.name.as_str();
    let shape = t.dims.clone();
    let decoded = match t.data_type {
        dt if dt == data_type::FLOAT => RawTensor::F32 {
            data: try_decode_le::<4, f32>(name, &raw, f32::from_le_bytes)?,
            shape,
        },
        dt if dt == data_type::INT64 => RawTensor::I64 {
            data: try_decode_le::<8, i64>(name, &raw, i64::from_le_bytes)?,
            shape,
        },
        dt if dt == data_type::BOOL => RawTensor::Bool {
            data: try_decode_le::<1, bool>(name, &raw, |b| b[0] != 0)?,
            shape,
        },
        dt if dt == data_type::FLOAT16 => RawTensor::F16 {
            data: try_decode_le::<2, half::f16>(name, &raw, half::f16::from_le_bytes)?,
            shape,
        },
        other => {
            return Err(GraphError::UnknownDataType {
                tensor_name: t.name.clone(),
                data_type: other,
            });
        }
    };
    // `raw` はここで解放される（復号後の要素 Vec だけが残る）。
    drop(raw);
    Ok(decoded)
}

/// `raw` を `N` バイトずつのリトルエンディアン要素として `conv` で変換し、
/// [`try_alloc_vec`] で確保した Vec へ詰める。`raw.len()` が `N` の倍数で
/// ない場合は `plan` の長さ検証が崩れた内部不変条件違反として `Internal`
/// を返す（余りを黙って切り捨てない）。
fn try_decode_le<const N: usize, T>(
    tensor_name: &str,
    raw: &[u8],
    conv: impl Fn([u8; N]) -> T,
) -> Result<Vec<T>, GraphError> {
    if !raw.len().is_multiple_of(N) {
        return Err(GraphError::ExternalData(ExternalDataError::Internal {
            reason: "try_decode_le: raw_data のバイト長が要素サイズの倍数ではない",
        }));
    }
    // 変換・失敗可能確保は実行経路（`Constant` 属性テンソルの復号）と共有する
    // `fallible_alloc::decode_le_into` に委譲する（余りが無いことは直上で検査済み）。
    super::fallible_alloc::decode_le_into::<N, T>(tensor_name, raw, conv)
        .map_err(|f| GraphError::ExternalData(alloc_failure_to_external(f)))
}

/// `file_ids`（`plan` 内。区間の重複検査もこのキーで畳み込んだファイル単位で行う）のファイル識別キー。overlap 検出・
/// distinct ファイル数の計数が「同一ファイル実体」を正しく畳み込めるよう、
/// パス文字列ではなくファイルの実体識別子を使う（Cursor Bugbot 指摘・
/// PR #2348 review thread `PRRT_kwDOTuUCJc6mkYTr`: `base_dir.join(location)` という `location` の
/// 生文字列連結をキーにすると、`Path` の `Eq`/`Hash` はコンポーネント
/// 単位のため `foo.data`／`./foo.data` の表記ゆれ自体は畳み込まれるが、
/// ハードリンクのように**文字列としても正規化後の経路としても異なるが
/// 実体は同一のファイル**は別キーになり overlap 検出をすり抜ける。
/// dev/ino をキーにすることでこの実体単位の同一性を保証する）。Windows も
/// 同じ理由でパスベースキーを廃止した（P0 是正。codex-review 指摘
/// `PRRT_kwDOTuUCJc6my2Ie`・PR #2351: ハードリンク・NTFS 8.3 短縮名等の
/// 別名パスで同一ファイルを参照されると、パスベースキーでは別ファイル
/// 扱いになり `OverlappingRegion` 検証をすり抜けていた）。
///
/// newtype（`(u64, u64)`／`(u32, u64)`／`PathBuf` の型エイリアスではなく
/// 専用構造体）にする理由: 型エイリアスのままだと cfg ごとに `Copy` 性が
/// 変わり（`(u64, u64)`／`(u32, u64)` は `Copy`・`PathBuf` は非 `Copy`）、
/// 複数箇所で必要な `.clone()` が環境依存で clippy `clone_on_copy`
/// （`-D warnings` 対象）に触れて `#[allow]` が要った。`Clone` のみ導出する
/// newtype にすることで、どの cfg でも `.clone()` が常に非自明な複製と
/// なり `#[allow]` そのものが不要になる（レビュー指摘: `#[allow(clippy::
/// clone_on_copy)]` を型設計で解消する）。
#[cfg(unix)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct FileKey(u64, u64);
/// Windows 版: `(dwVolumeSerialNumber, nFileIndex)`（unix の
/// `(dev, ino)` 相当。[`win_contained_open::file_identity`] が
/// `GetFileInformationByHandle` で取得する）。
#[cfg(windows)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct FileKey(u32, u64);
#[cfg(not(any(unix, windows)))]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct FileKey(PathBuf);

/// `opened`（`resolve_and_open` が返したハンドル）から [`FileKey`] を
/// 作る。Unix では dev/ino（`OpenFile::snapshot` の `dev`／`ino`）を使い、
/// Windows では `(dwVolumeSerialNumber, nFileIndex)`
/// （[`win_contained_open::file_identity`]）を使う。いずれもシンボリック
/// リンク・junction や表記ゆれだけでなくハードリンクも実体単位で同一
/// キーへ畳み込む。両者とも実体識別子を持たない他プラットフォームでは
/// `normalized_rel`（`resolve_and_open` が `Path::components()` から
/// 再構築した正規化済み相対パス）を `base_dir_canonical` へ連結した値を
/// フォールバックキーに使う（ハードリンク識別はできないが、表記ゆれの
/// 畳み込みは維持する）。Windows の実体識別子取得は kernel32.dll への
/// FFI 呼び出し（`win_contained_open::file_identity`）を経るため失敗しうる
/// （`Result` を返す。unix・フォールバックの他 2 分岐は既存情報からのみ
/// 構築するため失敗しない）。
#[cfg(unix)]
fn file_key_for(
    _tensor_name: &str,
    _base_dir_canonical: &Path,
    opened: &OpenFile,
    _normalized_rel: &Path,
) -> Result<FileKey, ExternalDataError> {
    Ok(FileKey(opened.snapshot.dev, opened.snapshot.ino))
}
#[cfg(windows)]
fn file_key_for(
    tensor_name: &str,
    _base_dir_canonical: &Path,
    opened: &OpenFile,
    _normalized_rel: &Path,
) -> Result<FileKey, ExternalDataError> {
    win_contained_open::file_identity(&opened.file)
        .map(|(volume_serial_number, file_index)| FileKey(volume_serial_number, file_index))
        .map_err(|e| ExternalDataError::Io {
            tensor_name: cap_name(tensor_name),
            kind: e.kind(),
        })
}
#[cfg(not(any(unix, windows)))]
fn file_key_for(
    _tensor_name: &str,
    base_dir_canonical: &Path,
    _opened: &OpenFile,
    normalized_rel: &Path,
) -> Result<FileKey, ExternalDataError> {
    Ok(FileKey(base_dir_canonical.join(normalized_rel)))
}

/// パス 1 で確定した「どこから何バイト読むか」の 1 件分。
struct LoadPlanEntry {
    slot: TensorSlot,
    tensor_name: String,
    /// `plan` が返す `PlannedLocation` 列への添字（どの location から
    /// 読むか）。
    location_idx: usize,
    offset: u64,
    length: u64,
}

/// パス 1 で検証した distinct な正規化済み location 1 件分の記録。
/// ファイルハンドルは保持せず、パス 2 の再 open 時に照合する識別情報
/// （`FileKey`・[`FileSnapshot`]〈長さ・unix では dev/ino・ctime・mtime〉）と、
/// 再 open に使う検証済みの正規化済み相対パスだけを持つ（モジュール doc
/// 「ハンドル非保持の構成」節）。
struct PlannedLocation {
    /// [`validate_location_string`] が返した `Normal` 成分列を連結した
    /// 正規化済み相対パス（`CurDir` 除去済み・`..`／絶対パスを含まない）。
    /// パス 2 は `rel.iter()` で成分列へ戻して `resolve_and_open` へ渡す。
    rel: PathBuf,
    /// パス 1 で開いたハンドルの実体識別子。
    key: FileKey,
    /// パス 1 で開いたハンドルの `fstat` スナップショット（ファイル長は
    /// `snapshot.len`）。パス 2 は [`ensure_unchanged`] でこれと照合する。
    snapshot: FileSnapshot,
}

/// external なテンソルの所在（`ModelProto` 内の位置）。パス 2 の mutable
/// 書き戻しで使う。
#[derive(Clone, Copy)]
enum TensorSlot {
    Initializer(usize),
    NodeAttrTensor { node_idx: usize, attr_idx: usize },
}

/// `model` に含まれるすべてのテンソル（initializer・各ノードの
/// `attribute[].t`）を `(slot, tensor_name, &TensorProto)` として列挙する。
/// サブグラフ属性（`g`/`graphs`）は `proto::AttributeProto` に未定義のため
/// 対象外（`proto.rs` 冒頭コメント参照）。
fn enumerate_tensors(model: &ModelProto) -> Vec<(TensorSlot, String, &TensorProto)> {
    let mut out = Vec::new();
    let Some(g) = model.graph.as_ref() else {
        return out;
    };
    for (idx, init) in g.node.iter().enumerate() {
        for (attr_idx, attr) in init.attribute.iter().enumerate() {
            if let Some(t) = attr.t.as_ref() {
                let name = if !t.name.is_empty() {
                    t.name.clone()
                } else {
                    let node_label = if init.name.is_empty() {
                        init.op_type.as_str()
                    } else {
                        init.name.as_str()
                    };
                    format!("{node_label}:{}", attr.name)
                };
                out.push((
                    TensorSlot::NodeAttrTensor {
                        node_idx: idx,
                        attr_idx,
                    },
                    name,
                    t,
                ));
            }
        }
    }
    for (idx, init) in g.initializer.iter().enumerate() {
        out.push((TensorSlot::Initializer(idx), init.name.clone(), init));
    }
    out
}

/// `plan` の戻り値: 検証済みの読み込み計画・distinct location の記録・
/// （external テンソルが 1 件以上あった場合のみ）`base_dir` ハンドル。
type Plan = (
    Vec<LoadPlanEntry>,
    Vec<PlannedLocation>,
    Option<BaseDirHandle>,
);

/// パス 1: 全 external テンソルを検証する（ファイル内容は読まない）。
/// 検証済みの読み込み計画・distinct location ごとの識別情報
/// （`FileKey`・ファイル長）・`base_dir` ハンドルを返す。external data
/// ファイルのハンドルは location ごとに開いて `fstat` した直後に close
/// し、戻り値にも含めない（同時保持 fd 数をファイル数に依存させない。
/// PR #2348 codex P1 是正）。
fn plan(
    model: &ModelProto,
    base_dir_canonical: &Path,
    options: &ExternalDataOptions,
) -> Result<Plan, GraphError> {
    // initializer 名の重複は I/O の前に拒否する（A5）。
    if let Some(g) = model.graph.as_ref() {
        let mut seen = std::collections::HashSet::new();
        for init in &g.initializer {
            if !seen.insert(init.name.as_str()) {
                return Err(GraphError::ExternalData(
                    ExternalDataError::DuplicateInitializerName {
                        tensor_name: cap_name(&init.name),
                    },
                ));
            }
        }
    }

    // `base_dir` のディレクトリ fd は遅延で 1 度だけ開き、以後の全テンソル
    // が `resolve_and_open` 経由でこの fd を再利用する（呼び出しごとに
    // 開き直さない）。external なテンソルが 1 件も無いモデル（`from_path`
    // 経由の通常モデル import を含む）では一切開かない: `std::fs::read`
    // は親ディレクトリの実行〈x〉権限のみを要求するのに対し
    // `open(O_DIRECTORY)` は読み取り〈r〉権限を要求するため、無条件に
    // 開くと「実行のみ許可のディレクトリに置かれた external data 非使用
    // モデル」が `from_path` で読めなくなる後退を招く（advisor 指摘）。
    // 非 unix の `BaseDirHandle` は何も開かないマーカーで、
    // `resolve_and_open` が常に拒否する。パス 2（`load`）の再 open も同じ
    // ハンドルを起点にするため、戻り値としてそのまま返す。
    let mut base_dir_file: Option<BaseDirHandle> = None;

    // 正規化済み location（`Path::components()` の `Normal` 列。
    // `validate_location_string` の戻り値から再構築した相対パス）から
    // 既に解決済みの `locations` の添字へのキャッシュ。**同一の location
    // 文字列**を複数のテンソルが参照する通常の分割形式（1 ファイルを
    // initializer 群が共有する構成）で、2 件目以降の `resolve_and_open`
    // （`openat2` 等のシステムコール）を省略し、パス 2 でもその location を
    // 1 回の再 open でまとめて読めるようにするために使う。安全性は変え
    // ない: 異なる location 文字列がハードリンク等で同一実体を指す場合は、
    // このキャッシュではヒットせず必ず個別に開いて `file_key_for`
    // （dev/ino）で判定する既存の畳み込み・overlap 検出をそのまま経由する
    // （レビュー対応。#2347）。
    let mut location_cache: HashMap<PathBuf, usize> = HashMap::new();
    let mut locations: Vec<PlannedLocation> = Vec::new();

    // distinct なファイル実体（`FileKey`）→ 初出順のファイル番号。
    // `file_ids.len()` が `max_external_files` の判定に使う distinct 数で、
    // ハンドルは保持しない（旧構成の `files: HashMap<FileKey, OpenFile>` を
    // 置き換えた。PR #2348 codex P1 是正）。ファイル番号は
    // `regions_by_file` の添字であり、ループ後の重複区間検査をファイルの
    // 初出順に行うことで報告するペアを決定的にする（`HashMap` の反復順に
    // 依存させない）。
    let mut file_ids: HashMap<FileKey, usize> = HashMap::new();
    let mut regions_by_file: Vec<Vec<Region>> = Vec::new();
    let mut entries: Vec<LoadPlanEntry> = Vec::new();
    let mut total_requested: u64 = 0;

    for (slot, tensor_name, t) in enumerate_tensors(model) {
        match t.data_location {
            v if v == data_location::DEFAULT => {
                if !t.external_data.is_empty() {
                    return Err(GraphError::ExternalData(
                        ExternalDataError::InconsistentDataFields {
                            tensor_name: cap_name(&tensor_name),
                            reason: "data_location=DEFAULT なのに external_data が非空",
                        },
                    ));
                }
                continue;
            }
            v if v == data_location::EXTERNAL => {}
            other => {
                return Err(GraphError::ExternalData(
                    ExternalDataError::InvalidDataLocation {
                        tensor_name: cap_name(&tensor_name),
                        value: other,
                    },
                ));
            }
        }

        if !t.raw_data.is_empty() || !t.float_data.is_empty() || !t.int64_data.is_empty() {
            return Err(GraphError::ExternalData(
                ExternalDataError::InconsistentDataFields {
                    tensor_name: cap_name(&tensor_name),
                    reason: "data_location=EXTERNAL なのに inline データが非空",
                },
            ));
        }

        let mut location: Option<String> = None;
        let mut offset_raw: Option<String> = None;
        let mut length_raw: Option<String> = None;
        let mut seen_keys = std::collections::HashSet::new();
        for entry in &t.external_data {
            if !seen_keys.insert(entry.key.clone()) {
                return Err(GraphError::ExternalData(ExternalDataError::DuplicateKey {
                    tensor_name: cap_name(&tensor_name),
                    key: cap_name(&entry.key),
                }));
            }
            match entry.key.as_str() {
                "location" => location = Some(entry.value.clone()),
                "offset" => offset_raw = Some(entry.value.clone()),
                "length" => length_raw = Some(entry.value.clone()),
                "checksum" => {
                    return Err(GraphError::ExternalData(
                        ExternalDataError::ChecksumUnsupported {
                            tensor_name: cap_name(&tensor_name),
                        },
                    ));
                }
                other => {
                    return Err(GraphError::ExternalData(ExternalDataError::UnknownKey {
                        tensor_name: cap_name(&tensor_name),
                        key: cap_name(other),
                    }));
                }
            }
        }
        let location = location.ok_or_else(|| {
            GraphError::ExternalData(ExternalDataError::MissingLocationKey {
                tensor_name: cap_name(&tensor_name),
            })
        })?;
        let offset = match offset_raw {
            Some(raw) => {
                parse_decimal_u64(&raw, &tensor_name, "offset").map_err(GraphError::ExternalData)?
            }
            None => 0,
        };

        let expected_elements = element_count(&tensor_name, &t.dims)?;
        let esize = element_size(&tensor_name, t.data_type)?;
        let expected_bytes = (expected_elements as u64)
            .checked_mul(esize)
            .ok_or_else(|| GraphError::ElementCountOverflow {
                tensor_name: tensor_name.clone(),
            })?;

        // `file_key`／`file_len` の解決。location キャッシュがヒットした
        // 場合はパス 1 で既に記録した識別情報を再利用し、このテンソルの
        // ためには一切 open しない。ミスした場合のみ安全 open →
        // ハンドル自身の `fstat` で `FileKey`・長さを記録 → 直ちに close
        // する（ハンドルはどこにも格納しない。PR #2348 codex P1 是正）。
        let (location_idx, file_key, file_len): (usize, FileKey, u64) = {
            // 最初の external テンソルに到達した時点でのみ `base_dir` の
            // ハンドルを開く（上のコメント参照。遅延オープン）。
            if base_dir_file.is_none() {
                base_dir_file = Some(
                    open_base_dir_handle(base_dir_canonical).map_err(GraphError::ExternalData)?,
                );
            }
            let f = base_dir_file.as_ref().ok_or(GraphError::ExternalData(
                ExternalDataError::Internal {
                    reason: "plan: base_dir_file が None のまま resolve_and_open へ到達した",
                },
            ))?;

            // `location` の検証（文字列段階＋`Path::components()`）は
            // ここで 1 度だけ行い（unix・非 unix 共通。非 unix でも
            // 文法不正は `UnsupportedPlatformForSecureResolve` より先に
            // 具体的な `InvalidLocation` として返す）、`resolve_and_open`
            // （本体のオープン処理）へは検証済みの `parts` を渡す。正規化
            // 済み相対パスを `location_cache` のキーにすることで、
            // `foo.data`／`./foo.data` のような表記ゆれも同一キーへ畳み
            // 込む（`file_key_for` の表記ゆれ畳み込みと同じ設計）。
            let parts = validate_location_string(&location).map_err(|reason| {
                GraphError::ExternalData(ExternalDataError::InvalidLocation {
                    tensor_name: cap_name(&tensor_name),
                    reason,
                })
            })?;
            let normalized_rel: PathBuf = parts.iter().collect();

            if let Some(&cached_idx) = location_cache.get(&normalized_rel) {
                // 同一 location への 2 件目以降: パス 1 で記録済みの
                // 識別情報を再利用し、`openat2`／`openat` を再実行しない
                // （安全性は変えない。上の `location_cache` 定義コメント
                // 参照）。
                let planned = locations.get(cached_idx).ok_or(GraphError::ExternalData(
                    ExternalDataError::Internal {
                        reason: "plan: location_cache の添字が locations の範囲外",
                    },
                ))?;
                (cached_idx, planned.key.clone(), planned.snapshot.len)
            } else {
                let opened =
                    resolve_and_open(&tensor_name, f, &parts).map_err(GraphError::ExternalData)?;
                let key = file_key_for(&tensor_name, base_dir_canonical, &opened, &normalized_rel)
                    .map_err(GraphError::ExternalData)?;
                let snapshot = opened.snapshot;
                // ここで close する（明示 drop。以後このハンドルは使わず、
                // パス 2 は再 open したハンドルを `key`・`snapshot` と照合
                // してから読む）。
                drop(opened);
                let idx = locations.len();
                locations.push(PlannedLocation {
                    rel: normalized_rel.clone(),
                    key: key.clone(),
                    snapshot,
                });
                location_cache.insert(normalized_rel, idx);
                (idx, key, snapshot.len)
            }
        };

        let length = match length_raw {
            Some(raw) => {
                parse_decimal_u64(&raw, &tensor_name, "length").map_err(GraphError::ExternalData)?
            }
            None => file_len.checked_sub(offset).ok_or_else(|| {
                GraphError::ExternalData(ExternalDataError::RangeOutOfFile {
                    tensor_name: cap_name(&tensor_name),
                })
            })?,
        };

        let end = offset.checked_add(length).ok_or_else(|| {
            GraphError::ExternalData(ExternalDataError::RangeOutOfFile {
                tensor_name: cap_name(&tensor_name),
            })
        })?;
        if end > file_len {
            return Err(GraphError::ExternalData(
                ExternalDataError::RangeOutOfFile {
                    tensor_name: cap_name(&tensor_name),
                },
            ));
        }
        if length != expected_bytes {
            return Err(GraphError::ExternalData(
                ExternalDataError::LengthMismatch {
                    tensor_name: cap_name(&tensor_name),
                    expected_bytes,
                    actual_bytes: length,
                },
            ));
        }

        total_requested = total_requested
            .checked_add(length)
            .ok_or(GraphError::ExternalData(
                ExternalDataError::TotalSizeLimitExceeded {
                    limit: options.max_total_bytes,
                    requested: u64::MAX,
                },
            ))?;
        if total_requested > options.max_total_bytes {
            return Err(GraphError::ExternalData(
                ExternalDataError::TotalSizeLimitExceeded {
                    limit: options.max_total_bytes,
                    requested: total_requested,
                },
            ));
        }

        // distinct ファイル数の上限検査（A04）: `max_total_bytes` はバイト
        // 数のみを制限するため、サイズ 0 のテンソルを大量の異なるファイルへ
        // 分散させると合計サイズは 0 のままファイルハンドルだけが増え、
        // プロセスの fd 上限に達しうる（#2347 P0 是正・PR #2348 コード
        // レビュー対応・PRRT_kwDOTuUCJc6mlxhy）。`file_key` が既知の実体で
        // なければ、`file_ids` へ登録する前にここで拒否する（ハンドルは
        // 上で既に close 済みのため、この判定自体は fd を消費しない）。
        let file_id = match file_ids.get(&file_key) {
            Some(&id) => id,
            None => {
                if file_ids.len() >= options.max_external_files {
                    return Err(GraphError::ExternalData(
                        ExternalDataError::TooManyExternalFiles {
                            limit: options.max_external_files,
                        },
                    ));
                }
                let id = regions_by_file.len();
                regions_by_file.push(Vec::new());
                file_ids.insert(file_key, id);
                id
            }
        };

        // 同一ファイル内の読み込み区間はここでは収集だけ行い、重複判定は
        // 全テンソルの計画後に [`find_overlap`] でまとめて行う（codex P0
        // 是正。旧実装はテンソルごとに同一ファイルの既存区間を全走査して
        // おり、1 ファイルへ重ならない短い区間を多数並べた入力で O(n²) に
        // なった。`max_external_files` は同一ファイルを 1 件としか数えず、
        // `max_total_bytes` も短い区間の合計しか制限しないため、上限では
        // 抑えられない）。`length == 0` のテンソル（1 バイトも読まない）は
        // 区間としての幅を持たないため対象外とする（`offset` がファイル長
        // 以内であることは上の `end > file_len` 検査で既に検証済み。
        // レビュー対応: 0 バイト読み込みが既存区間の内側の offset を指す
        // だけで誤って `OverlappingRegion` になっていた不具合の是正。
        // #2347）。
        if length > 0 {
            regions_by_file
                .get_mut(file_id)
                .ok_or(GraphError::ExternalData(ExternalDataError::Internal {
                    reason: "plan: file_id が regions_by_file の範囲外",
                }))?
                .push(Region {
                    offset,
                    end,
                    entry_idx: entries.len(),
                });
        }

        entries.push(LoadPlanEntry {
            slot,
            tensor_name,
            location_idx,
            offset,
            length,
        });
    }

    // 同一ファイル内の読み込み区間の重複検査（同一区間の二重参照を含む）。
    // 上限判定（総量・ファイル数）を全テンソルについて終えた後、読み込み
    // （パス 2）より前に行う fail-closed 判定。ファイルは初出順に検査し、
    // 最初に見つかった重複を報告する。
    for regions in regions_by_file.iter_mut() {
        if let Some((later_idx, earlier_idx)) = find_overlap(regions) {
            let name_of = |idx: usize| {
                entries
                    .get(idx)
                    .map(|e| cap_name(&e.tensor_name))
                    .ok_or(GraphError::ExternalData(ExternalDataError::Internal {
                        reason: "plan: Region::entry_idx が entries の範囲外",
                    }))
            };
            return Err(GraphError::ExternalData(
                ExternalDataError::OverlappingRegion {
                    tensor_name: name_of(later_idx)?,
                    other_tensor_name: name_of(earlier_idx)?,
                },
            ));
        }
    }

    Ok((entries, locations, base_dir_file))
}

/// 同一ファイル内の読み込み区間 1 件分（`plan` が収集し [`find_overlap`]
/// が検査する）。`[offset, end)` の半開区間で、`entry_idx` は `plan` の
/// `entries`（external テンソルの入力順）への添字。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Region {
    offset: u64,
    end: u64,
    entry_idx: usize,
}

/// 同一ファイルの区間集合から重なり合う 2 区間を 1 組探す（O(n log n)。
/// codex P0 是正）。見つかれば `(後側の entry_idx, 既存側の entry_idx)` を
/// 返す。
///
/// `(offset, entry_idx)` の昇順にソートしてから走査し、それまでに現れた
/// 区間の最大 `end` とその持ち主を保持する。次の区間の `offset` がその
/// 最大 `end` 未満なら、持ち主の区間と重なっている（持ち主の `offset` は
/// ソート順で次の区間の `offset` 以下、かつ持ち主の `end` は次の区間の
/// `offset` を超えるため）。直前の区間ではなく最大 `end` と比べることで、
/// 長い区間が後続の複数区間を包含する場合も漏らさない。境界が接するだけ
/// （`end == 次の offset`）は重なりではない。同じ `offset` の区間は入力順
/// （`entry_idx`）で並べ、最大 `end` の持ち主は同値なら先に現れた区間の
/// まま更新しないため、報告するペアは入力に対して決定的である。長さ 0 の
/// 区間（`offset == end`）は幅を持たないため検査対象外とする（`plan` は
/// そもそも登録しないが、関数単体でもこの契約を守る）。
fn find_overlap(regions: &mut [Region]) -> Option<(usize, usize)> {
    regions.sort_unstable_by_key(|r| (r.offset, r.entry_idx));
    let mut max_end: Option<(u64, usize)> = None;
    for r in regions.iter().filter(|r| r.end > r.offset) {
        match max_end {
            Some((end, owner)) if r.offset < end => return Some((r.entry_idx, owner)),
            Some((end, _)) if r.end <= end => {}
            _ => max_end = Some((r.end, r.entry_idx)),
        }
    }
    None
}

/// パス 2: パス 1 が確定した計画に従い、該当区間だけを読み込む。
///
/// 正規化済み location ごとに「`base_dir` ハンドル起点の安全な再 open
/// （[`resolve_and_open`]。パス 1 と同じシンボリックリンク拒否手段）→
/// 開いたハンドル自身の `FileKey`・[`FileSnapshot`] をパス 1 の記録と
/// 照合（[`ensure_unchanged`]。各区間の読み込み直前にも再照合）→
/// その location を参照する全テンソルの区間だけを読む → close」を 1 件
/// ずつ逐次に行い、同時に開く external data ファイルは常に 1 つに保つ
/// （PR #2348 codex P1 是正。モジュール doc「ハンドル非保持の構成」節）。
/// 照合の不一致は [`ExternalDataError::FileChangedDuringLoad`]、再 open
/// 自体の失敗（削除による `NotFound`・シンボリックリンクへの差し替え等）は
/// パス 1 と同じ variant（`Io`／`InvalidLocation`）、照合通過直後〜
/// 読み込みの間の truncate は [`read_region`] の読み込み不足
/// （`Io { kind: UnexpectedEof }`）、区間バッファの確保失敗は
/// [`ExternalDataError::AllocationFailed`] でいずれも fail-closed に拒否する。戻り値は `entries` と同じ
/// 順序・同じ件数。
fn load(
    entries: &[LoadPlanEntry],
    locations: &[PlannedLocation],
    base_dir: Option<&BaseDirHandle>,
    base_dir_canonical: &Path,
) -> Result<Vec<Vec<u8>>, GraphError> {
    let base_dir = base_dir.ok_or(GraphError::ExternalData(ExternalDataError::Internal {
        reason: "load: entries が非空なのに base_dir ハンドルが無い",
    }))?;

    // location ごとに、それを参照する `entries` の添字をまとめる（読み込み
    // 順を location 単位へ並べ替えても、結果は `out[entry_idx]` へ格納する
    // ため戻り値の順序は `entries` のまま）。
    let mut by_location: Vec<Vec<usize>> = vec![Vec::new(); locations.len()];
    for (entry_idx, entry) in entries.iter().enumerate() {
        by_location
            .get_mut(entry.location_idx)
            .ok_or(GraphError::ExternalData(ExternalDataError::Internal {
                reason: "load: entry.location_idx が locations の範囲外",
            }))?
            .push(entry_idx);
    }

    let mut out: Vec<Option<Vec<u8>>> = (0..entries.len()).map(|_| None).collect();
    for (planned, entry_indices) in locations.iter().zip(by_location.iter()) {
        let Some(&first_idx) = entry_indices.first() else {
            // `plan` は location を記録した反復で必ず 1 件の entry も
            // 追加する（途中で失敗すれば `Err` で抜ける）ため到達しない。
            // 参照の無い location は開く必要が無いので読み飛ばす。
            continue;
        };
        let first_name = entries
            .get(first_idx)
            .map(|e| e.tensor_name.as_str())
            .ok_or(GraphError::ExternalData(ExternalDataError::Internal {
                reason: "load: entry 添字が entries の範囲外",
            }))?;

        let parts: Vec<&std::ffi::OsStr> = planned.rel.iter().collect();
        let mut opened =
            resolve_and_open(first_name, base_dir, &parts).map_err(GraphError::ExternalData)?;
        // 再 open したハンドル自身の識別子・スナップショット（長さ・unix
        // では dev/ino・ctime・mtime・Windows では volume serial／
        // file index）をパス 1 の記録と照合する（経路文字列ではなく
        // ハンドルに対する照会の結果。不一致なら 1 バイトも読まずに拒否
        // する）。`FileKey` の比較は unix・Windows でも実体単位の同一性
        // 照合として残す（`file_key_for` ドキュメント参照）。
        let key_now = file_key_for(first_name, base_dir_canonical, &opened, &planned.rel)
            .map_err(GraphError::ExternalData)?;
        if key_now != planned.key {
            return Err(GraphError::ExternalData(
                ExternalDataError::FileChangedDuringLoad {
                    tensor_name: cap_name(first_name),
                },
            ));
        }
        ensure_unchanged(first_name, &planned.snapshot, &opened.snapshot)
            .map_err(GraphError::ExternalData)?;

        for &entry_idx in entry_indices {
            let entry = entries.get(entry_idx).ok_or(GraphError::ExternalData(
                ExternalDataError::Internal {
                    reason: "load: entry 添字が entries の範囲外",
                },
            ))?;
            // 各区間の読み込み直前にも同じハンドルへ `fstat` を取り直し、
            // パス 1 の記録と同じ全フィールド照合を行う（同一ハンドルでも
            // 読み込み中の truncate・in-place 書き込み等は起こりうるため。
            // ctime／mtime を含めることで同じ長さのままの書き換えも検出
            // する）。この照合を通過した直後〜読み込みの間の変更は
            // 残存窓であり、truncate なら `read_region` の `UnexpectedEof`
            // として `ExternalDataError::Io` で fail-closed になる（モジュール
            // doc「TOCTOU の論拠」節）。
            let meta = opened.file.metadata().map_err(|e| {
                GraphError::ExternalData(ExternalDataError::Io {
                    tensor_name: cap_name(&entry.tensor_name),
                    kind: e.kind(),
                })
            })?;
            ensure_unchanged(
                &entry.tensor_name,
                &planned.snapshot,
                &FileSnapshot::from_metadata(&meta),
            )
            .map_err(GraphError::ExternalData)?;

            opened
                .file
                .seek(SeekFrom::Start(entry.offset))
                .map_err(|e| {
                    GraphError::ExternalData(ExternalDataError::Io {
                        tensor_name: cap_name(&entry.tensor_name),
                        kind: e.kind(),
                    })
                })?;
            // 区間バッファは失敗可能確保で用意する（PR #2348 codex P0 是正。
            // 旧実装の `vec![0u8; buf_len]` は確保失敗でプロセスが abort
            // しえた。`max_total_bytes` 以内でも既定 64 GiB は利用可能
            // メモリを超えうる）。`usize` へ変換できない長さ・`isize::MAX`
            // 超の長さも確保前に `AllocationFailed` で拒否し、読み込み不足は
            // 旧実装と同じ `Io { kind: UnexpectedEof }` になる
            // （[`read_region`]）。確保は上の照合を通過した後に限る
            // （変化を検知したファイルのためには確保しない）。
            let buf = read_region(&mut opened.file, &entry.tensor_name, entry.length)
                .map_err(GraphError::ExternalData)?;
            let slot = out.get_mut(entry_idx).ok_or(GraphError::ExternalData(
                ExternalDataError::Internal {
                    reason: "load: entry 添字が out の範囲外",
                },
            ))?;
            *slot = Some(buf);
        }
        // `opened` はこの反復の末尾で drop（close）される。次の location の
        // open より前に閉じるため、同時に開く external data ファイルは 1 つ。
        drop(opened);
    }

    out.into_iter()
        .map(|b| {
            b.ok_or(GraphError::ExternalData(ExternalDataError::Internal {
                reason: "load: 読み込まれなかった entry が残っている",
            }))
        })
        .collect()
}

/// `model` に含まれる external なテンソル（initializer・Constant 属性
/// テンソル）をすべて検証・読み込み、`raw_data` へ inline 化する（in-place。
/// `data_location` は `DEFAULT` に戻し `external_data` は空にする）。
///
/// 呼び出し元は [`build_graph_with_external_data`]。`base_dir` は
/// canonicalize してから使う（失敗は [`ExternalDataError::InvalidBaseDir`]）。
/// `base_dir` 自体は呼び出し元が与える信頼済み入力として扱う（location
/// 側だけを fail-closed に検証する。モジュール冒頭コメント参照）。
///
/// **読み込み経路のメモリ予算**: `options.max_total_bytes` は本関数が確保
/// する raw バッファ（全 external テンソルの `raw_data` の合計）の予算で
/// あり、`plan` がファイル内容を読む前・1 バイトも確保する前に判定する。
/// 各バッファは失敗可能確保（[`ExternalDataError::AllocationFailed`]）で
/// 用意する。利用可能メモリが既定上限（64 GiB）より小さい環境では、
/// 呼び出し元が `max_total_bytes` を下げて渡すこと（変換後テンソルを
/// 含めたピーク見積もりは `docs/onnx-external-data-decision.md` 4.3 節）。
/// 本関数の後に `graph::build_graph` を呼ぶと、external 由来 initializer の
/// 復号（`decode_tensor` の `collect`）が無条件確保になる。グラフ構築まで
/// 行う場合は、復号も失敗可能確保で行う [`build_graph_with_external_data`]
/// を使うこと。
pub fn resolve_external_data(
    model: &mut ModelProto,
    base_dir: &Path,
    options: &ExternalDataOptions,
) -> Result<(), GraphError> {
    resolve_external_data_slots(model, base_dir, options).map(|_| ())
}

/// [`resolve_external_data`] の本体。inline 化したテンソルの所在（slot）を
/// `plan` の列挙順で返す（[`build_graph_with_external_data`] が external
/// 由来の initializer だけを失敗可能確保で復号するために使う）。
fn resolve_external_data_slots(
    model: &mut ModelProto,
    base_dir: &Path,
    options: &ExternalDataOptions,
) -> Result<Vec<TensorSlot>, GraphError> {
    let base_dir_canonical = base_dir.canonicalize().map_err(|e| {
        GraphError::ExternalData(ExternalDataError::InvalidBaseDir { kind: e.kind() })
    })?;

    let (entries, locations, base_dir_file) = plan(model, &base_dir_canonical, options)?;
    if entries.is_empty() {
        return Ok(Vec::new());
    }
    let loaded = load(
        &entries,
        &locations,
        base_dir_file.as_ref(),
        &base_dir_canonical,
    )?;
    if loaded.len() != entries.len() {
        return Err(GraphError::ExternalData(ExternalDataError::Internal {
            reason: "load の戻り値件数が entries と一致しない",
        }));
    }

    // パス 2 完了後にのみ書き戻す（検証・読み込みが全件成功した場合限定）。
    // `entries` が非空の場合、`plan` は `enumerate_tensors` が返した
    // 実在の slot からのみ `entries` を構築しているため、ここで
    // `model.graph` は必ず `Some`（`enumerate_tensors` は graph が
    // `None` なら空 Vec を返す）。`unwrap`/`expect` の代わりに
    // `ok_or_else` で型付きエラーとして表面化させる（内部不変条件が
    // 崩れた場合のみ到達する防御的分岐）。
    let g = model
        .graph
        .as_mut()
        .ok_or(GraphError::ExternalData(ExternalDataError::Internal {
            reason: "entries が非空なのに model.graph が None",
        }))?;
    let mut slots = Vec::with_capacity(entries.len());
    for (entry, bytes) in entries.into_iter().zip(loaded) {
        slots.push(entry.slot);
        let target: &mut TensorProto = match entry.slot {
            TensorSlot::Initializer(idx) => g.initializer.get_mut(idx).ok_or(
                GraphError::ExternalData(ExternalDataError::Internal {
                    reason: "書き戻し先の initializer index が範囲外",
                }),
            )?,
            TensorSlot::NodeAttrTensor { node_idx, attr_idx } => g
                .node
                .get_mut(node_idx)
                .and_then(|n| n.attribute.get_mut(attr_idx))
                .and_then(|a| a.t.as_mut())
                .ok_or(GraphError::ExternalData(ExternalDataError::Internal {
                    reason: "書き戻し先の attribute テンソルが見つからない",
                }))?,
        };
        target.raw_data = bytes;
        target.data_location = data_location::DEFAULT;
        target.external_data.clear();
    }
    Ok(slots)
}

/// `ModelProto` を複製し external data を inline 化してから内部グラフを
/// 構築する新しい import 入口（イシュー #2347）。`base_dir` は external な
/// `location` の解決基点。
///
/// external テンソルが 1 件も無いモデルは、従来どおりバイト列入口と同じ
/// `graph::build_graph` へそのまま渡す（`resolve_external_data` は
/// external テンソルが 0 件なら `raw_data` を一切書き換えない）。
///
/// external テンソルを 1 件以上持つモデルは、複製を所有権ごと
/// `graph::build_graph_owned` へ渡す（PR #2348 codex P0 是正）。
/// `build_graph` と同じ検証（sparse 拒否・initializer 名重複・トポロジ・
/// グラフ出力）を同じ順序で行いつつ、(a) external 由来の initializer は
/// `try_decode_external_initializer` で失敗可能確保により復号し、raw を
/// 復号直後に解放する（`graph::decode_tensor` の `collect` による無条件確保
/// ＝確保失敗時の abort を external 由来の巨大テンソルで踏まない。
/// `decode_tensor` 自体は A6 により変更しない）、(b) inline の initializer
/// は従来どおり `decode_tensor` で復号する（入力バイト列長で有界）、
/// (c) ノード列は clone せず move する（external 由来の Constant 属性
/// テンソルの `raw_data` を 2 重に持たない）。結果の `Graph` は
/// `resolve_external_data` → `build_graph` の旧経路と同一（統合テスト
/// `owned_build_matches_resolve_then_build_graph` で固定）。
pub fn build_graph_with_external_data(
    model: &ModelProto,
    base_dir: &Path,
    options: &ExternalDataOptions,
) -> Result<super::graph::Graph, GraphError> {
    let mut cloned = model.clone();
    let slots = resolve_external_data_slots(&mut cloned, base_dir, options)?;
    if slots.is_empty() {
        return super::graph::build_graph(&cloned);
    }
    let initializer_count = cloned.graph.as_ref().map_or(0, |g| g.initializer.len());
    let mut external_initializer = vec![false; initializer_count];
    for slot in slots {
        if let TensorSlot::Initializer(idx) = slot {
            *external_initializer
                .get_mut(idx)
                .ok_or(GraphError::ExternalData(ExternalDataError::Internal {
                    reason: "build_graph_with_external_data: initializer slot が範囲外",
                }))? = true;
        }
    }
    super::graph::build_graph_owned(cloned, |idx, init| {
        if external_initializer.get(idx).copied().unwrap_or(false) {
            try_decode_external_initializer(init)
        } else {
            super::graph::decode_tensor(init)
        }
    })
}

/// `no_follow_open` の 2 方式（Linux 限定 `openat2` と unix 全般の逐次
/// `openat(O_NOFOLLOW)` フォールバック）を直接呼び出す単体テスト。
/// `resolve_and_open` は Linux では既定で `openat2` 側を優先するため、
/// `openat2` が未対応（`ENOSYS`／`EPERM`）にならない通常の開発・CI 環境
/// では統合テスト（`tests/onnx_external_data.rs`）だけではフォールバック
/// 関数（`open_chain_component_walk`）自体が実行されない。ここで両関数を
/// 直接呼び、フォールバック経路も独立して検証する。
/// [`find_overlap`]（同一ファイル内の区間重複検査。codex P0 是正で
/// O(n log n) のソート＋走査へ置き換えた）の単体テスト。ファイル I/O を
/// 伴わないため全プラットフォームで実行する。
#[cfg(test)]
mod overlap_tests {
    use super::{Region, find_overlap};

    fn r(offset: u64, end: u64, entry_idx: usize) -> Region {
        Region {
            offset,
            end,
            entry_idx,
        }
    }

    /// 境界が接するだけ（`end == 次の offset`）の区間は重ならない。
    #[test]
    fn adjacent_touching_regions_do_not_overlap() {
        let mut v = vec![r(4, 8, 1), r(0, 4, 0), r(8, 12, 2)];
        assert_eq!(find_overlap(&mut v), None);
    }

    /// 隣接区間が 1 バイトでも重なれば検出し、`(後側, 既存側)` を返す。
    #[test]
    fn adjacent_regions_sharing_one_byte_overlap() {
        let mut v = vec![r(0, 4, 0), r(3, 8, 1)];
        assert_eq!(find_overlap(&mut v), Some((1, 0)));
    }

    /// 長い区間が後続の区間を包含する場合も検出する（内側の区間を後側、
    /// 包含する区間を既存側として報告する）。
    #[test]
    fn containing_region_is_detected() {
        let mut v = vec![r(10, 20, 1), r(0, 100, 0), r(30, 40, 2)];
        assert_eq!(find_overlap(&mut v), Some((1, 0)));
    }

    /// 入力順がファイル内の位置順と一致しなくても（逆順・飛び飛び）、
    /// ソート後の位置関係だけで判定する。
    #[test]
    fn unsorted_input_is_judged_by_position() {
        let mut v = vec![r(50, 60, 0), r(0, 5, 1), r(5, 50, 2)];
        assert_eq!(find_overlap(&mut v), None);
        let mut w = vec![r(50, 60, 0), r(0, 5, 1), r(5, 51, 2)];
        assert_eq!(find_overlap(&mut w), Some((0, 2)));
    }

    /// 長さ 0 の区間は幅を持たないため、既存区間の内側・同じ offset を
    /// 指していても重なりとしない。
    #[test]
    fn zero_length_regions_are_excluded() {
        let mut v = vec![r(0, 8, 0), r(4, 4, 1), r(0, 0, 2), r(8, 8, 3)];
        assert_eq!(find_overlap(&mut v), None);
    }

    /// 同じ offset の区間は入力順（`entry_idx`）で並べ、後に現れた方を
    /// 後側として報告する（入力の並びに依らず決定的）。同一区間の二重
    /// 参照もここに含まれる。
    #[test]
    fn same_offset_regions_are_reported_deterministically() {
        let mut v = vec![r(0, 4, 1), r(0, 4, 0)];
        assert_eq!(find_overlap(&mut v), Some((1, 0)));
        let mut w = vec![r(0, 4, 0), r(0, 4, 1)];
        assert_eq!(find_overlap(&mut w), Some((1, 0)));
        let mut x = vec![r(0, 2, 1), r(0, 8, 0)];
        assert_eq!(find_overlap(&mut x), Some((1, 0)));
    }

    /// 1 ファイルに重ならない長さ 1 の区間を多数（逆順で）並べても重なり
    /// なしと判定する（旧実装の O(n²) 全走査では 20 万件で約 2×10^10 回の
    /// 比較になる規模。ソート＋1 回の走査で完了することを、件数を大きく
    /// 取って通ること自体で確認する。実時間の閾値判定はしない）。
    #[test]
    fn many_non_overlapping_unit_regions_are_accepted() {
        const N: usize = 200_000;
        let mut v: Vec<Region> = (0..N).rev().map(|i| r(i as u64, i as u64 + 1, i)).collect();
        assert_eq!(find_overlap(&mut v), None);
        // 末尾に 1 件だけ重なる区間を足すと検出する（空振りでないこと）。
        v.push(r(12_345, 12_347, N));
        assert_eq!(find_overlap(&mut v), Some((N, 12_345)));
    }
}

#[cfg(all(test, unix))]
mod tests {
    use std::ffi::OsStr;

    use super::no_follow_open;

    /// テスト専用の一時ディレクトリ（`tests/onnx_external_data.rs::
    /// TempDir` と同型。本ファイルはユニットテストのため独立実装する）。
    struct UnitTestDir(std::path::PathBuf);

    impl UnitTestDir {
        fn new(name: &str) -> Self {
            static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let dir = std::env::temp_dir().join(format!(
                "onnx-interop-external-data-unit-{}-{name}-{n}",
                std::process::id()
            ));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).expect("一時ディレクトリの作成に失敗した");
            UnitTestDir(dir)
        }

        fn path(&self) -> &std::path::Path {
            &self.0
        }
    }

    impl Drop for UnitTestDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// 両方式（`open_chain_component_walk`・Linux では `open_chain_openat2`
    /// も）が、通常ファイルへの正常な多段解決を成功させることを確認する。
    #[test]
    fn both_strategies_open_normal_nested_file() {
        let dir = UnitTestDir::new("normal");
        std::fs::create_dir_all(dir.path().join("sub")).unwrap();
        std::fs::write(dir.path().join("sub/f.data"), [1u8, 2, 3, 4]).unwrap();
        let base_dir_file = no_follow_open::open_base_dir(dir.path()).expect("base_dir open 失敗");

        let parts: Vec<&OsStr> = vec![OsStr::new("sub"), OsStr::new("f.data")];

        let walked = no_follow_open::open_chain_component_walk(&base_dir_file, &parts)
            .expect("component_walk は成功するはず");
        assert_eq!(
            std::io::Read::bytes(walked)
                .map(|b| b.unwrap())
                .collect::<Vec<u8>>(),
            vec![1, 2, 3, 4]
        );

        #[cfg(target_os = "linux")]
        {
            match no_follow_open::open_chain_openat2(&base_dir_file, &parts) {
                no_follow_open::Openat2Outcome::Opened(f) => {
                    assert_eq!(
                        std::io::Read::bytes(f)
                            .map(|b| b.unwrap())
                            .collect::<Vec<u8>>(),
                        vec![1, 2, 3, 4]
                    );
                }
                // 本 CI 環境の kernel が openat2 非対応の場合のみ許容する
                // （`ENOSYS`/`EPERM`。`resolve_and_open` 側は
                // `open_chain_component_walk` へフォールバックする経路と
                // 同じ判定）。
                no_follow_open::Openat2Outcome::Unsupported => {}
                no_follow_open::Openat2Outcome::Failed(e, is_symlink) => {
                    panic!("openat2 が失敗した（is_symlink={is_symlink}）: {e}")
                }
            }
        }
    }

    /// 両方式が、経路途中のシンボリックリンクを `ELOOP` 系の失敗として
    /// 拒否することを確認する（`is_symlink` フラグが true になること）。
    #[test]
    fn both_strategies_reject_symlink_component() {
        let dir = UnitTestDir::new("symlink");
        std::fs::write(dir.path().join("real.data"), [9u8; 4]).unwrap();
        std::os::unix::fs::symlink(dir.path().join("real.data"), dir.path().join("link.data"))
            .unwrap();
        let base_dir_file = no_follow_open::open_base_dir(dir.path()).expect("base_dir open 失敗");

        let parts: Vec<&OsStr> = vec![OsStr::new("link.data")];

        let (_e, is_symlink) = no_follow_open::open_chain_component_walk(&base_dir_file, &parts)
            .expect_err("component_walk はシンボリックリンクを拒否するはず");
        assert!(
            is_symlink,
            "component_walk: is_symlink フラグが立っていない"
        );

        #[cfg(target_os = "linux")]
        {
            match no_follow_open::open_chain_openat2(&base_dir_file, &parts) {
                no_follow_open::Openat2Outcome::Failed(_e, is_symlink) => {
                    assert!(is_symlink, "openat2: is_symlink フラグが立っていない");
                }
                no_follow_open::Openat2Outcome::Unsupported => {}
                no_follow_open::Openat2Outcome::Opened(_) => {
                    panic!("openat2 がシンボリックリンクを開いてしまった（TOCTOU 回帰）")
                }
            }
        }
    }

    // --- パス 1／パス 2 間の差し替え検知（PR #2348 security-auditor P2-1） ---
    //
    // `plan` と `load` の間に割り込めるのは内部関数を直接呼べる本単体
    // テストだけのため、ここで検査する（統合テスト `tests/
    // onnx_external_data.rs` は公開入口しか呼べない）。

    use super::{
        ExternalDataError, ExternalDataOptions, FileSnapshot, ensure_unchanged, load, plan,
    };
    use crate::onnx::graph::GraphError;
    use crate::onnx::proto::{
        GraphProto, ModelProto, StringStringEntryProto, TensorProto, data_location, data_type,
    };

    /// 照合前後の差分を 1 フィールドずつ作るための基準スナップショット。
    fn base_snapshot() -> FileSnapshot {
        FileSnapshot {
            len: 8,
            dev: 1,
            ino: 42,
            ctime: 1_700_000_000,
            ctime_nsec: 123_456_789,
            mtime: 1_700_000_000,
            mtime_nsec: 123_456_789,
        }
    }

    fn assert_changed(result: Result<(), ExternalDataError>, what: &str) {
        match result {
            Err(ExternalDataError::FileChangedDuringLoad { tensor_name }) => {
                assert_eq!(tensor_name, "w", "{what}: tensor_name が伝播していない");
            }
            other => panic!("{what}: FileChangedDuringLoad を期待したが {other:?}"),
        }
    }

    /// `ensure_unchanged` は全フィールド一致のときだけ通し、長さ・dev・
    /// ino・ctime（秒／ナノ秒）・mtime（秒／ナノ秒）のいずれか 1 つでも
    /// 異なれば `FileChangedDuringLoad` で拒否する。とくに dev/ino・長さが
    /// 一致し ctime だけが異なるケース（unlink → 同じ長さで再作成 → inode
    /// 番号再利用）を拒否することが P2-1 是正の要である。
    #[test]
    fn ensure_unchanged_rejects_any_single_field_difference() {
        let planned = base_snapshot();
        assert!(ensure_unchanged("w", &planned, &planned).is_ok());

        let cases: [(&str, FileSnapshot); 7] = [
            ("len", FileSnapshot { len: 9, ..planned }),
            ("dev", FileSnapshot { dev: 2, ..planned }),
            ("ino", FileSnapshot { ino: 43, ..planned }),
            (
                "ctime",
                FileSnapshot {
                    ctime: planned.ctime + 1,
                    ..planned
                },
            ),
            (
                "ctime_nsec",
                FileSnapshot {
                    ctime_nsec: planned.ctime_nsec + 1,
                    ..planned
                },
            ),
            (
                "mtime",
                FileSnapshot {
                    mtime: planned.mtime + 1,
                    ..planned
                },
            ),
            (
                "mtime_nsec",
                FileSnapshot {
                    mtime_nsec: planned.mtime_nsec + 1,
                    ..planned
                },
            ),
        ];
        for (what, now) in cases {
            assert_changed(ensure_unchanged("w", &planned, &now), what);
        }
    }

    /// `location` の FLOAT テンソル `w`（dims=[2]・8 バイト）を 1 件だけ
    /// external data として持つモデル。
    fn single_external_model(location: &str) -> ModelProto {
        let entry = |k: &str, v: &str| StringStringEntryProto {
            key: k.to_string(),
            value: v.to_string(),
        };
        let t = TensorProto {
            dims: vec![2],
            data_type: data_type::FLOAT,
            name: "w".to_string(),
            external_data: vec![entry("location", location), entry("length", "8")],
            data_location: data_location::EXTERNAL,
            ..Default::default()
        };
        ModelProto {
            ir_version: 8,
            graph: Some(GraphProto {
                name: "g".to_string(),
                initializer: vec![t],
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    fn f32_bytes(vals: [f32; 2]) -> Vec<u8> {
        vals.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    /// カーネルのファイルタイムスタンプは粗いクロック刻み（数 ms）で更新
    /// されうるため、`plan` の記録と差し替え後の ctime／mtime が同一刻み
    /// に収まって偶然一致しないよう、差し替え前に待つ。競合の再現待ちでは
    /// なく、タイムスタンプ粒度を跨ぐための待機（同一粒度内の差し替えは
    /// モジュール doc に記す受容済みの残存リスク）。
    fn wait_past_timestamp_granularity() {
        std::thread::sleep(std::time::Duration::from_millis(50));
    }

    fn assert_load_rejected_as_changed(result: Result<Vec<Vec<u8>>, GraphError>, what: &str) {
        match result {
            Err(GraphError::ExternalData(ExternalDataError::FileChangedDuringLoad {
                tensor_name,
            })) => assert_eq!(tensor_name, "w"),
            Err(other) => panic!("{what}: FileChangedDuringLoad を期待したが {other:?}"),
            Ok(bytes) => panic!("{what}: 差し替え後のファイルを読んでしまった: {bytes:?}"),
        }
    }

    /// 差し替えなしの対照: `plan` → `load` がそのまま元の内容を返す
    /// （下の差し替えテストが harness の不備で空振りしていないことの確認）。
    #[test]
    fn plan_then_load_without_change_succeeds() {
        let dir = UnitTestDir::new("no-change");
        std::fs::write(dir.path().join("w.data"), f32_bytes([1.0, 2.0])).unwrap();
        let base = dir.path().canonicalize().unwrap();
        let model = single_external_model("w.data");
        let (entries, locations, base_dir_file) =
            plan(&model, &base, &ExternalDataOptions::default()).expect("plan は成功するはず");
        wait_past_timestamp_granularity();
        let loaded = load(&entries, &locations, base_dir_file.as_ref(), &base)
            .expect("差し替えなしの load は成功するはず");
        assert_eq!(loaded, vec![f32_bytes([1.0, 2.0])]);
    }

    /// `plan` の後に元ファイルを unlink し、同じ長さの別ファイルを同じ名前で
    /// 作成すると `load` が `FileChangedDuringLoad` で拒否する。inode 番号が
    /// 再利用された場合は dev/ino・長さが一致し ctime／mtime だけが差し替えを
    /// 検出し、再利用されなかった場合は dev/ino が検出する（どちらでも
    /// 拒否されればよい）。
    #[test]
    fn unlink_and_recreate_same_length_between_plan_and_load_is_rejected() {
        let dir = UnitTestDir::new("unlink-recreate");
        let path = dir.path().join("w.data");
        std::fs::write(&path, f32_bytes([1.0, 2.0])).unwrap();
        let base = dir.path().canonicalize().unwrap();
        let model = single_external_model("w.data");
        let (entries, locations, base_dir_file) =
            plan(&model, &base, &ExternalDataOptions::default()).expect("plan は成功するはず");

        wait_past_timestamp_granularity();
        std::fs::remove_file(&path).unwrap();
        std::fs::write(&path, f32_bytes([7.0, 8.0])).unwrap();

        assert_load_rejected_as_changed(
            load(&entries, &locations, base_dir_file.as_ref(), &base),
            "unlink → 同長の別ファイル作成",
        );
    }

    /// `plan` の後に同じ inode を長さを変えずに in-place で書き換えると
    /// `load` が `FileChangedDuringLoad` で拒否する。dev/ino・長さは変わら
    /// ないため、ctime／mtime 照合の追加前（dev/ino・長さのみの照合）は
    /// 書き換え後の内容を黙って読んでいた経路である。
    #[test]
    fn in_place_same_length_overwrite_between_plan_and_load_is_rejected() {
        use std::io::Write;
        use std::os::unix::fs::MetadataExt;

        let dir = UnitTestDir::new("in-place");
        let path = dir.path().join("w.data");
        std::fs::write(&path, f32_bytes([1.0, 2.0])).unwrap();
        let before = std::fs::metadata(&path).unwrap();
        let base = dir.path().canonicalize().unwrap();
        let model = single_external_model("w.data");
        let (entries, locations, base_dir_file) =
            plan(&model, &base, &ExternalDataOptions::default()).expect("plan は成功するはず");

        wait_past_timestamp_granularity();
        {
            // truncate せず先頭から同じ長さを上書きする（同一 inode・同一長）。
            let mut f = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
            f.write_all(&f32_bytes([7.0, 8.0])).unwrap();
            f.sync_all().unwrap();
        }
        let after = std::fs::metadata(&path).unwrap();
        assert_eq!(
            (before.dev(), before.ino(), before.len()),
            (after.dev(), after.ino(), after.len()),
            "前提: in-place 上書きでは dev/ino・長さが変わらない"
        );

        assert_load_rejected_as_changed(
            load(&entries, &locations, base_dir_file.as_ref(), &base),
            "同一 inode の同長 in-place 上書き",
        );
    }

    /// `plan` で検証済みの計画の `length` を `isize::MAX + 1` へ書き換えて
    /// `load` へ渡すと、区間バッファを確保しようとする前に
    /// `AllocationFailed` で拒否される（PR #2348 codex P0 是正。旧実装の
    /// `vec![0u8; n]` は同じ入力で abort しえた）。`plan` の上限検査を
    /// 通過した後の `load` 単体の確保経路を、実確保せずに検証する。
    #[test]
    fn load_rejects_unallocatable_length_as_allocation_failed() {
        let dir = UnitTestDir::new("load-alloc");
        std::fs::write(dir.path().join("w.data"), f32_bytes([1.0, 2.0])).unwrap();
        let base = dir.path().canonicalize().unwrap();
        let model = single_external_model("w.data");
        let (mut entries, locations, base_dir_file) =
            plan(&model, &base, &ExternalDataOptions::default()).expect("plan は成功するはず");
        let huge = isize::MAX as u64 + 1;
        entries[0].length = huge;
        match load(&entries, &locations, base_dir_file.as_ref(), &base) {
            Err(GraphError::ExternalData(ExternalDataError::AllocationFailed {
                tensor_name,
                bytes,
            })) => {
                assert_eq!(tensor_name, "w");
                assert_eq!(bytes, huge);
            }
            other => panic!("AllocationFailed を期待したが {other:?}"),
        }
    }
}

/// 失敗可能確保のヘルパ（[`try_alloc_vec`]・[`alloc_region_buf`]・
/// [`read_region`]）と、external 由来 initializer の失敗可能復号
/// （[`try_decode_external_initializer`]）の単体テスト（PR #2348 codex P0
/// 是正）。数十 GiB の実確保は CI で危険なため行わず、`isize::MAX` 超の
/// 要求（`try_reserve_exact` がアロケータを呼ばずに `CapacityOverflow` で
/// 失敗する）で失敗経路を決定的に検証する。ファイル I/O を伴わないため全
/// プラットフォームで実行する。
#[cfg(test)]
mod alloc_tests {
    use super::{
        ExternalDataError, alloc_region_buf, read_region, try_alloc_vec,
        try_decode_external_initializer,
    };
    use crate::onnx::graph::{GraphError, decode_tensor};
    use crate::onnx::proto::{TensorProto, data_type};

    fn assert_alloc_failed<T: std::fmt::Debug>(
        r: Result<T, ExternalDataError>,
        expected_bytes: u64,
    ) {
        match r {
            Err(ExternalDataError::AllocationFailed { tensor_name, bytes }) => {
                assert_eq!(tensor_name, "t");
                assert_eq!(bytes, expected_bytes);
            }
            other => panic!("AllocationFailed を期待したが {other:?}"),
        }
    }

    #[test]
    fn try_alloc_vec_reports_capacity_overflow_as_allocation_failed() {
        assert_alloc_failed(try_alloc_vec::<u8>("t", usize::MAX), usize::MAX as u64);
        // 要素数自体は `isize::MAX` 未満でも、バイト数が溢れれば同様に拒否し、
        // 診断用 `bytes` は飽和計算する。
        let count = usize::MAX / 2;
        assert_alloc_failed(
            try_alloc_vec::<f32>("t", count),
            (count as u64).saturating_mul(4),
        );
        let ok = try_alloc_vec::<f32>("t", 16).expect("小さな確保は成功するはず");
        assert!(ok.is_empty() && ok.capacity() >= 16);
    }

    #[test]
    fn alloc_region_buf_rejects_lengths_beyond_isize_max_before_allocating() {
        let over = isize::MAX as u64 + 1;
        assert_alloc_failed(alloc_region_buf("t", over), over);
        assert_alloc_failed(alloc_region_buf("t", u64::MAX), u64::MAX);
        let ok = alloc_region_buf("t", 8).expect("小さな確保は成功するはず");
        assert!(ok.is_empty() && ok.capacity() >= 8);
    }

    /// `read_region` は事前確保した容量ちょうどまで埋め、`read_to_end` が
    /// 再確保しない（容量が変わらない）ことを実測で固定する（std 内部の
    /// 探査読み込みの挙動に依存するため、記憶ではなくテストで押さえる）。
    /// 余分なバイトは読まない（`Take` の上限）。
    #[test]
    fn read_region_fills_exact_capacity_without_realloc() {
        for len in [0usize, 1, 31, 32, 33, 4096, 70_000] {
            let src: Vec<u8> = (0..len + 5).map(|i| (i % 251) as u8).collect();
            let mut cur = std::io::Cursor::new(src.clone());
            let buf = read_region(&mut cur, "t", len as u64).expect("読み込みは成功するはず");
            assert_eq!(buf.as_slice(), &src[..len], "len={len}");
            let expected_cap = alloc_region_buf("t", len as u64).unwrap().capacity();
            assert_eq!(buf.capacity(), expected_cap, "len={len}: 再確保された");
            assert_eq!(cur.position(), len as u64, "len={len}: 余分に読んだ");
        }
    }

    /// 読み込めたバイト数が要求に満たない（truncate 等）場合は、旧実装の
    /// `read_exact` と同じ `Io { kind: UnexpectedEof }` で拒否する。
    #[test]
    fn read_region_short_read_is_unexpected_eof() {
        let mut cur = std::io::Cursor::new(vec![1u8, 2, 3]);
        match read_region(&mut cur, "t", 8) {
            Err(ExternalDataError::Io { tensor_name, kind }) => {
                assert_eq!(tensor_name, "t");
                assert_eq!(kind, std::io::ErrorKind::UnexpectedEof);
            }
            other => panic!("Io(UnexpectedEof) を期待したが {other:?}"),
        }
    }

    fn tensor(name: &str, dt: i32, dims: Vec<i64>, raw: Vec<u8>) -> TensorProto {
        TensorProto {
            dims,
            data_type: dt,
            name: name.to_string(),
            raw_data: raw,
            ..Default::default()
        }
    }

    /// 失敗可能復号は 4 型すべてで `graph::decode_tensor`（バイト列入口と
    /// 共有・A6 で不変）と同じ `RawTensor` を返し、raw を取り出して解放する。
    #[test]
    fn try_decode_matches_decode_tensor_for_all_dtypes() {
        let f32_raw: Vec<u8> = [1.5f32, -0.0, f32::INFINITY, f32::MIN_POSITIVE]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let i64_raw: Vec<u8> = [i64::MIN, -1, 0, 42]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let f16_raw: Vec<u8> = [0.5f32, -65504.0, 6.0e-8]
            .iter()
            .flat_map(|v| half::f16::from_f32(*v).to_le_bytes())
            .collect();
        let cases = [
            tensor("f", data_type::FLOAT, vec![2, 2], f32_raw),
            tensor("i", data_type::INT64, vec![4], i64_raw),
            tensor("b", data_type::BOOL, vec![5], vec![0, 1, 2, 128, 255]),
            tensor("h", data_type::FLOAT16, vec![3], f16_raw),
            tensor("empty", data_type::FLOAT, vec![0], Vec::new()),
        ];
        for t in cases {
            let expected = decode_tensor(&t).expect("decode_tensor は成功するはず");
            let mut owned = t.clone();
            let got =
                try_decode_external_initializer(&mut owned).expect("失敗可能復号は成功するはず");
            assert_eq!(got, expected, "tensor={}", t.name);
            assert!(owned.raw_data.is_empty(), "tensor={}: raw が残った", t.name);
        }
    }

    /// バイト長が要素サイズの倍数でない（`plan` の検証が崩れた）場合は余りを
    /// 黙って切り捨てず `Internal` で拒否する。
    #[test]
    fn try_decode_rejects_partial_element_as_internal() {
        let mut t = tensor("f", data_type::FLOAT, vec![1], vec![0u8; 5]);
        assert!(matches!(
            try_decode_external_initializer(&mut t),
            Err(GraphError::ExternalData(ExternalDataError::Internal { .. }))
        ));
    }
}
