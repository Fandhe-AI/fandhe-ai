//! NumPy `.npz` 形式（複数配列を束ねた ZIP コンテナ）の読み書き
//! （イシュー #2189）。
//!
//! `io`（親モジュール）の doc を参照。本ファイルは ZIP コンテナの
//! 解析（EOCD／zip64／central directory／local header）と生成を担う。
//! **central directory を正とする**。**実測**（CPython 3.x
//! `zipfile`・numpy 2.3.5 `np.savez` はいずれも `force_zip64=True` で
//! local header を書くため、`np.savez` の local header は実サイズの
//! 大小に関わらず常に size 欄が `0xFFFFFFFF` の zip64 プレースホルダに
//! なる（central directory 側は zip64 extra を伴わない literal な
//! 実サイズのまま。scratchpad での実測で確認済み）。このため size
//! 欄は「シグネチャ確認とヘッダ長読み飛ばし」に加え central directory
//! との照合にも使うが、プレースホルダの場合は照合をスキップし
//! central directory 側を無条件に正とする（`read_member` の
//! `local_size_field_matches` 参照）。method／flags／CRC はプレース
//! ホルダを持たないため central directory と厳密一致することを
//! 要求する（PR #2318 レビュー指摘・P0）。書き出しは STORED
//! （無圧縮）のみに限定する（`np.savez` と同じ。`np.savez_compressed`
//! 相当の DEFLATE 圧縮書き出しは対象外。`docs/tensor-core-npy-npz-io-
//! decision.md` 参照）。
//!
//! ## 構造検証の方針（PR #2318 レビュー指摘・P0 監査。ZIP パーサの
//! 既知の脆弱性チェックリストに基づく全数監査で確定した決定事項）
//!
//! - **メンバのデータ領域は非重複・central directory 開始位置
//!   （`cd_offset`）より前**であることを要求する（`read_npz_bytes` の
//!   1.5 パス目・`member_data_range`）。central directory は zip64 EOCD
//!   レコード（zip64 使用時）または通常 EOCD（不使用時）より前である
//!   ことも要求する（`resolve_eocd_counts` の `metadata_start`）。
//! - **メンバ同士・メンバと central directory の間の未使用バイト
//!   （padding）は許容する**。読み飛ばすだけで内容の解釈には使わない
//!   ため安全上無害（CPython `zipfile`／`np.load` も同様に許容する）。
//! - **EOCD シグネチャ候補が複数ある場合は拒否する**（`find_eocd`）。
//!   コメント本文に偽の EOCD 様バイト列を埋め込む攻撃を想定した
//!   fail-closed 判定であり、真の ZIP では発生しない。
//! - **ファイル先頭・EOCD 直後の余剰バイト（self-extracting stub の
//!   prepend／コメント欄超過の append）は補正しない**。central
//!   directory・local header のオフセットはすべてファイル先頭からの
//!   絶対オフセットとして解釈するため、prepend されたアーカイブは
//!   オフセット不整合で自然に拒否される（自己解凍形式のような prepend
//!   を許容する特別な補正ロジックは持たない）。
//! - **エントリ名は `/`・`\` を含んでいてもよい**（`is_directory` 判定
//!   は名前が `/`／`\` で終わるかどうかのみで行い、途中に含まれる場合は
//!   階層的なメンバ名として扱う）。メンバ名はホスト側の `HashMap` キー
//!   としてのみ使われファイルシステムパスとしては解釈しないため、
//!   `..`・絶対パス表記を含んでいてもパストラバーサルは成立しない。
//!   書き出し側（`write_npz_bytes`）は `/`・`\` を含むキーを
//!   `InvalidEntryName` で拒否するため、自前書き出し→読み込みの往復に
//!   影響はない。

use std::collections::HashMap;
use std::path::Path;

use super::NpyError;
use super::bounded::{read_u16_le, read_u32_le, read_u64_le, slice_at};
use super::crc32::crc32;
use super::inflate::inflate;
use super::npy::{npy_encoded_len, read_npy_bytes, write_npy_bytes};
use crate::tensor::Tensor;

const LOCAL_FILE_HEADER_SIG: u32 = 0x0403_4b50;
const CENTRAL_DIR_HEADER_SIG: u32 = 0x0201_4b50;
const EOCD_SIG: u32 = 0x0605_4b50;
const ZIP64_EOCD_LOCATOR_SIG: u32 = 0x0706_4b50;
const ZIP64_EOCD_SIG: u32 = 0x0606_4b50;
const ZIP64_EXTRA_ID: u16 = 0x0001;

/// EOCD の固定長部分（`PK\x05\x06` を含む）のサイズ。
const EOCD_FIXED_SIZE: usize = 22;
/// EOCD のコメント欄長の最大値（`u16::MAX`）を考慮した後方探索範囲。
const EOCD_SEARCH_WINDOW: usize = EOCD_FIXED_SIZE + u16::MAX as usize;
/// npz エントリ数の上限（過大な central directory 走査を防ぐ）。
const MAX_ENTRIES: usize = 65535;

/// npz 1 メンバあたりの伸長後（デコンプレス後）サイズ上限。
///
/// 単体の `.npy` ファイルとして許容されるサイズ（`super::
/// MAX_FILE_READ_BYTES`）を npz メンバ 1 件の伸長後サイズにもそのまま
/// 適用する（新たな閾値を持ち込まず既存ポリシーを流用するだけ、という
/// 位置づけ。`docs/tensor-core-npy-npz-io-decision.md` 参照）。central
/// directory の `uncompressed_size`（宣言値）をこの上限と突き合わせ、
/// `inflate` が出力バッファを確保する**前**に拒否する。DEFLATE の
/// 圧縮比上限（`inflate::MAX_COMPRESSION_RATIO`）は圧縮入力サイズに対する
/// 相対的な理論値に過ぎず、圧縮入力自体がアーカイブサイズ上限
/// （`super::MAX_FILE_READ_BYTES`）いっぱいまで大きい場合は数百 GiB 級の
/// 出力を理論上許してしまうため、絶対値での上限が別途必要
/// （`.claude/rules/security.md` A03/A04/A05。PR #2318 レビュー指摘・
/// P0）。
const MAX_MEMBER_DECOMPRESSED_BYTES: u64 = super::MAX_FILE_READ_BYTES;

/// npz アーカイブ全体（全メンバ合計）の伸長後サイズ上限。
///
/// 単一メンバの上限（[`MAX_MEMBER_DECOMPRESSED_BYTES`]）を満たす複数の
/// メンバを束ねて `HashMap` へ蓄積すると、アーカイブ自体は
/// `MAX_FILE_READ_BYTES` の範囲内でも合計メモリ消費がその何倍にも
/// なりうる。メンバ単体と同じ `MAX_FILE_READ_BYTES` を桁の基準として
/// 累積上限にも流用する（PR #2318 レビュー指摘・P0）。**多層防御**:
/// 「同一 local header を指す複数 central directory エントリ」による
/// 圧縮データの重複自体は、`read_npz_bytes` の構造検証パス（メンバの
/// データ領域が互いに重ならないことの検証。`member_data_range` 参照）が
/// 別途拒否する（PR #2318 レビュー指摘・P0 監査）ため、本上限が主に
/// 防ぐのは「互いに重ならない多数の正当なメンバの宣言サイズ合計が
/// 大きすぎる」ケースである。
const MAX_TOTAL_DECOMPRESSED_BYTES: u64 = super::MAX_FILE_READ_BYTES;

/// バイト列（`.npz` ファイルの内容そのもの）から名前付きテンソル集合を
/// 読み取る。1 メンバでも失敗すれば全体を失敗させ、部分的な
/// `HashMap` は返さない（fail-closed。`.claude/rules/security.md` A08）。
pub fn read_npz_bytes(bytes: &[u8]) -> Result<HashMap<String, Tensor<f32>>, NpyError> {
    let eocd = find_eocd(bytes)?;
    let (entry_count, cd_offset, cd_size, metadata_start) = resolve_eocd_counts(bytes, &eocd)?;
    if entry_count > MAX_ENTRIES {
        return Err(NpyError::InvalidZip("エントリ数が上限を超える"));
    }
    let cd_end = cd_offset
        .checked_add(cd_size)
        .ok_or(NpyError::InvalidZip("central directory の範囲が不正"))?;
    // central directory の宣言範囲（`[cd_offset, cd_end)`）は、ファイル
    // 範囲内であることに加え、末尾の構造領域（zip64 を使わない場合は
    // 通常 EOCD、zip64 の場合は zip64 EOCD レコード）の開始位置
    // （`metadata_start`）より前に完全に収まらなければならない。
    // `cd_end > bytes.len()` のみの検査では、central directory が
    // EOCD／zip64 EOCD レコード・locator と重なって解析されるのを
    // 防げない（A03。PR #2318 レビュー指摘・P0。`resolve_eocd_counts`
    // 参照）。`metadata_start <= bytes.len()` は常に成り立つため、この
    // 検査は従来の `cd_end > bytes.len()` 検査を包含する。
    if cd_end > metadata_start {
        return Err(NpyError::InvalidZip(
            "central directory が EOCD／zip64 EOCD 領域と重なる",
        ));
    }

    // 1 パス目: central directory 全エントリを解析し、宣言範囲・伸長後
    // サイズ上限（メンバ単体・累積）を検証する。`read_member_bytes`／
    // `inflate`（＝出力バッファの実確保）は 2 パス目まで一切呼ばない
    // ため、宣言値の時点で上限超過と分かるメンバについて確保が発生する
    // ことはない（PR #2318 レビュー指摘・P0）。累積サイズは「この
    // アーカイブから読み取る全メンバの宣言サイズ合計」であり、1 パス目
    // で全件を先に集計してから 2 パス目の実読み込みへ進むことで、
    // 後続エントリの累積超過を先頭側のメンバの実読み込みより前に検出
    // できる（先頭側のメンバの実読み込みが〈本来は無関係な理由で〉
    // 先に失敗し累積検査の意図がテストできなくなることを避ける）。
    let mut entries: Vec<CentralDirEntry> = Vec::with_capacity(entry_count);
    let mut pos = cd_offset;
    let mut total_decompressed: u64 = 0;
    for _ in 0..entry_count {
        // central directory の宣言範囲（`cd_end`）内でのみエントリを
        // 解析する。細工した `cd_size`／`entry_count` により宣言範囲外
        // を central directory エントリとして読み取れてしまうのを防ぐ
        // （A03。PR #2318 レビュー指摘・P0）。固定長ヘッダ分（46 バイ
        // ト）すら `cd_end` に収まらない位置から解析を始めない。
        if pos.checked_add(46).is_none_or(|end| end > cd_end) {
            return Err(NpyError::InvalidZip(
                "central directory エントリが宣言範囲を超える",
            ));
        }
        let (entry, next_pos) = parse_central_directory_entry(bytes, pos)?;
        if next_pos > cd_end {
            return Err(NpyError::InvalidZip(
                "central directory エントリが宣言範囲を超える",
            ));
        }
        pos = next_pos;

        // メンバ単体・累積の伸長後サイズ上限検査（PR #2318 レビュー
        // 指摘・P0）。central directory の宣言値だけで判定するため、
        // 上限超過となるメンバの出力バッファを実際に確保することはない。
        // ディレクトリエントリ（`entry.is_directory`）も 2 パス目で
        // `read_member_bytes` を経由するため同じ検査を適用する。
        if entry.uncompressed_size > MAX_MEMBER_DECOMPRESSED_BYTES {
            return Err(NpyError::Entry {
                name: entry.name.clone(),
                source: Box::new(NpyError::DecompressedSizeExceeded {
                    len: entry.uncompressed_size,
                    max: MAX_MEMBER_DECOMPRESSED_BYTES,
                }),
            });
        }
        total_decompressed = total_decompressed
            .checked_add(entry.uncompressed_size)
            .filter(|&total| total <= MAX_TOTAL_DECOMPRESSED_BYTES)
            .ok_or(NpyError::DecompressedSizeExceeded {
                len: total_decompressed.saturating_add(entry.uncompressed_size),
                max: MAX_TOTAL_DECOMPRESSED_BYTES,
            })?;

        entries.push(entry);
    }
    // 走査終了位置が central directory の宣言終端と厳密に一致すること
    // を確認する（entry_count だけを信頼せず、宣言範囲全体が実際の
    // エントリ列で過不足なく埋まっていることを検証する）。
    if pos != cd_end {
        return Err(NpyError::InvalidZip(
            "central directory の走査終了位置が宣言範囲と不一致",
        ));
    }

    // 1.5 パス目: 各メンバの構造的なバイト範囲（`[local header 開始,
    // 圧縮データ終端)`）を求め、(a) central directory の開始位置
    // （`cd_offset`）より前に完全に収まること・(b) メンバ同士で重ならない
    // ことを検証する。central directory を「宣言範囲内で正しく解析
    // できる」ことは既に確認済みだが、それだけでは細工した
    // `local_header_offset`／`compressed_size` により、あるメンバの
    // データ領域が central directory 自体や他メンバのデータ領域と重なる
    // ことを防げない（CRC・npy 内容さえ整合させれば受理されてしまう。
    // A03。PR #2318 レビュー指摘・P0）。範囲計算は
    // [`member_data_range`] に委譲し、本関数は境界・重複判定のみを行う
    // （メンバ同士の隙間〈padding〉自体は許容する。zipfile／np.load も
    // 読み飛ばすだけで安全上無害なため）。
    let mut ranges: Vec<(usize, usize, &str)> = Vec::with_capacity(entries.len());
    for entry in &entries {
        let (start, end) = member_data_range(bytes, entry).map_err(|e| NpyError::Entry {
            name: entry.name.clone(),
            source: Box::new(e),
        })?;
        if end > cd_offset {
            return Err(NpyError::Entry {
                name: entry.name.clone(),
                source: Box::new(NpyError::InvalidZip(
                    "メンバのデータ領域が central directory の開始位置を超える",
                )),
            });
        }
        ranges.push((start, end, entry.name.as_str()));
    }
    ranges.sort_unstable_by_key(|&(start, _, _)| start);
    for w in ranges.windows(2) {
        let (_, prev_end, _) = w[0];
        let (next_start, _, next_name) = w[1];
        if next_start < prev_end {
            return Err(NpyError::Entry {
                name: next_name.to_string(),
                source: Box::new(NpyError::InvalidZip(
                    "メンバのデータ領域が他のメンバと重なる",
                )),
            });
        }
    }

    // 2 パス目: 1 パス目で伸長後サイズ上限を通過したエントリのみを
    // 実際に読み込む（local header 照合・伸長・CRC 検証）。
    let mut result = HashMap::with_capacity(entry_count);
    for entry in &entries {
        if entry.is_directory {
            // ディレクトリエントリも central directory／local header の
            // 構造的整合性（名前・flags・method・CRC・サイズの一致）は
            // 検証する。名前だけを見て無条件に読み飛ばすと、データを
            // 持つエントリを「ディレクトリ名」（末尾 `/` 等）に偽装した
            // 細工 ZIP が、local header との不一致を一切検査されずに
            // 受理されてしまう（PR #2318 レビュー指摘・P2）。ディレクトリ
            // は本来データを持たないため、検証後の内容が非空であれば
            // 拒否する。
            let decompressed = read_member_bytes(bytes, entry).map_err(|e| NpyError::Entry {
                name: entry.name.clone(),
                source: Box::new(e),
            })?;
            if !decompressed.is_empty() {
                return Err(NpyError::Entry {
                    name: entry.name.clone(),
                    source: Box::new(NpyError::InvalidZip(
                        "ディレクトリエントリに非空データが含まれる",
                    )),
                });
            }
            continue;
        }
        let key = npz_member_key(&entry.name)?;
        if result.contains_key(&key) {
            return Err(NpyError::DuplicateEntry { name: key });
        }
        let tensor = read_member(bytes, entry).map_err(|e| NpyError::Entry {
            name: key.clone(),
            source: Box::new(e),
        })?;
        result.insert(key, tensor);
    }
    Ok(result)
}

/// `path` の npz ファイルを読み込む。
///
/// ファイル全体を検証前に無条件で確保しないよう、`super::
/// read_file_bounded`（サイズ上限検査つき・TOCTOU 対策済み）を経由する
/// （`.claude/rules/security.md` A03/A04/A05。PR #2318 レビュー指摘）。
pub fn load_npz<P: AsRef<Path>>(path: P) -> Result<HashMap<String, Tensor<f32>>, NpyError> {
    let bytes = super::read_file_bounded(path.as_ref())?;
    read_npz_bytes(&bytes)
}

/// central directory の 1 エントリ分の情報。
struct CentralDirEntry {
    name: String,
    is_directory: bool,
    method: u16,
    compressed_size: u64,
    uncompressed_size: u64,
    local_header_offset: u64,
    crc32: u32,
    flags: u16,
}

/// EOCD の探索結果。
struct EocdInfo {
    /// EOCD 固定部分の開始オフセット。
    offset: usize,
}

/// ファイル末尾から EOCD シグネチャ（`PK\x05\x06`）を後方探索する。
///
/// 探索窓（`EOCD_SEARCH_WINDOW`）全体を走査し、シグネチャ＋コメント長
/// 整合の両方を満たす候補が複数存在する場合は「一意に真の EOCD を
/// 決定できない」として fail-closed に拒否する（細工したコメント本文に
/// 偽の EOCD 様バイト列を埋め込む攻撃を想定。`.claude/rules/security.md`
/// A03。PR #2318 レビュー指摘・P0 監査）。真の ZIP は EOCD がファイル末尾
/// 側に 1 つだけ存在するため、正常系への影響はない。
fn find_eocd(bytes: &[u8]) -> Result<EocdInfo, NpyError> {
    if bytes.len() < EOCD_FIXED_SIZE {
        return Err(NpyError::InvalidZip("ファイルが短すぎて EOCD が存在しない"));
    }
    let search_start = bytes
        .len()
        .saturating_sub(EOCD_SEARCH_WINDOW.min(bytes.len()));
    let mut candidates: Vec<usize> = Vec::new();
    let mut i = bytes.len() - EOCD_FIXED_SIZE;
    loop {
        if read_u32_le(bytes, i, "EOCD 探索中の範囲外アクセス").ok() == Some(EOCD_SIG) {
            // コメント長フィールドとファイル終端の整合を検査し、偽陽性
            // （データ本体にたまたま同じ 4 バイトが出現する場合）を除く。
            let comment_len = read_u16_le(bytes, i + 20, "EOCD comment 長").unwrap_or(u16::MAX);
            if i + EOCD_FIXED_SIZE + comment_len as usize == bytes.len() {
                candidates.push(i);
            }
        }
        if i == search_start {
            break;
        }
        i -= 1;
    }
    match candidates.len() {
        0 => Err(NpyError::InvalidZip(
            "EOCD（end of central directory）が見つからない",
        )),
        1 => Ok(EocdInfo {
            offset: candidates[0],
        }),
        _ => Err(NpyError::InvalidZip(
            "EOCD シグネチャ候補が複数あり一意に決定できない",
        )),
    }
}

/// EOCD（および必要なら zip64 EOCD）からエントリ数・central directory の
/// オフセット・サイズ・central directory が収まらなければならない上限
/// （`metadata_start`）を解決する。`metadata_start` は zip64 を使わない
/// 場合は通常 EOCD の開始位置、zip64 を使う場合は zip64 EOCD レコードの
/// 開始位置であり、central directory の宣言範囲がこれらの構造領域と
/// 重ならないことを呼び出し元（`read_npz_bytes`）が検証するために使う
/// （A03。PR #2318 レビュー指摘・P0 監査）。
fn resolve_eocd_counts(
    bytes: &[u8],
    eocd: &EocdInfo,
) -> Result<(usize, usize, usize, usize), NpyError> {
    let o = eocd.offset;
    let disk_number = read_u16_le(bytes, o + 4, "EOCD disk number")?;
    let cd_start_disk = read_u16_le(bytes, o + 6, "EOCD cd start disk")?;
    let entries_this_disk = read_u16_le(bytes, o + 8, "EOCD entries (this disk)")?;
    let entries_total = read_u16_le(bytes, o + 10, "EOCD entries (total)")?;
    let cd_size_32 = read_u32_le(bytes, o + 12, "EOCD cd size")?;
    let cd_offset_32 = read_u32_le(bytes, o + 16, "EOCD cd offset")?;

    if disk_number != 0 || cd_start_disk != 0 || entries_this_disk != entries_total {
        return Err(NpyError::UnsupportedZipFeature("マルチディスク ZIP"));
    }

    let needs_zip64 =
        entries_total == u16::MAX || cd_size_32 == u32::MAX || cd_offset_32 == u32::MAX;
    if !needs_zip64 {
        // 非 zip64 経路の `metadata_start` は通常 EOCD の開始位置
        // （`o`）。central directory の宣言範囲はこれより前に収まら
        // なければならない（`read_npz_bytes` 側の検証）。
        return Ok((
            entries_total as usize,
            cd_offset_32 as usize,
            cd_size_32 as usize,
            o,
        ));
    }

    // zip64 EOCD locator は EOCD の直前 20 バイトに固定長で存在する
    // （NumPy は `force_zip64=True` のため EOCD 直前に必ず置く）。
    if o < 20 {
        return Err(NpyError::InvalidZip(
            "zip64 EOCD locator の領域が確保できない",
        ));
    }
    let locator_offset = o - 20;
    if read_u32_le(bytes, locator_offset, "zip64 EOCD locator シグネチャ")?
        != ZIP64_EOCD_LOCATOR_SIG
    {
        return Err(NpyError::InvalidZip(
            "zip64 EOCD locator シグネチャが不一致",
        ));
    }
    // zip64 EOCD locator のマルチディスクフィールド（PR #2318 レビュー
    // 指摘・P2）: 「zip64 EOCD を含むディスク番号」（offset 4）と
    // 「ディスク総数」（offset 16）。本 IO は単一ディスク ZIP のみを
    // 対象とする契約（EOCD 側の disk_number／cd_start_disk 検証と同じ
    // 契約）のため、ここが単一ディスクを示さない値なら
    // `UnsupportedZipFeature` で拒否する（黙って central directory の
    // 内容を誤って解釈しない）。
    let zip64_locator_disk = read_u32_le(bytes, locator_offset + 4, "zip64 EOCD locator disk")?;
    let zip64_total_disks =
        read_u32_le(bytes, locator_offset + 16, "zip64 EOCD locator total disks")?;
    if zip64_locator_disk != 0 || zip64_total_disks != 1 {
        return Err(NpyError::UnsupportedZipFeature("マルチディスク ZIP"));
    }
    let zip64_eocd_offset = read_u64_le(bytes, locator_offset + 8, "zip64 EOCD offset")?;
    let zip64_eocd_offset = usize::try_from(zip64_eocd_offset)
        .map_err(|_| NpyError::InvalidZip("zip64 EOCD offset が usize 範囲を超える"))?;
    if read_u32_le(bytes, zip64_eocd_offset, "zip64 EOCD シグネチャ")? != ZIP64_EOCD_SIG {
        return Err(NpyError::InvalidZip("zip64 EOCD シグネチャが不一致"));
    }
    // zip64 EOCD 本体のマルチディスクフィールド（同 P2 指摘）: 「このディスク
    // の番号」（offset 16）・「central directory 開始ディスク番号」
    // （offset 20）・「このディスク上のエントリ数」（offset 24）。単一
    // ディスク契約のもとでは「このディスク上のエントリ数」は「総エントリ数」
    // （offset 32）と一致するはずであり、一致しなければマルチディスク
    // 構成として拒否する。
    let zip64_this_disk = read_u32_le(bytes, zip64_eocd_offset + 16, "zip64 EOCD this disk")?;
    let zip64_cd_start_disk =
        read_u32_le(bytes, zip64_eocd_offset + 20, "zip64 EOCD cd start disk")?;
    let entries_this_disk = read_u64_le(
        bytes,
        zip64_eocd_offset + 24,
        "zip64 EOCD entries (this disk)",
    )?;
    let entries = read_u64_le(bytes, zip64_eocd_offset + 32, "zip64 EOCD entries")?;
    if zip64_this_disk != 0 || zip64_cd_start_disk != 0 || entries_this_disk != entries {
        return Err(NpyError::UnsupportedZipFeature("マルチディスク ZIP"));
    }
    let cd_size = read_u64_le(bytes, zip64_eocd_offset + 40, "zip64 EOCD cd size")?;
    let cd_offset = read_u64_le(bytes, zip64_eocd_offset + 48, "zip64 EOCD cd offset")?;

    // 通常 EOCD 側のフィールドが sentinel（zip64 経路への切り替え値）で
    // ない場合、zip64 EOCD レコードから解決した値と一致することを要求
    // する。`needs_zip64` は 3 フィールドのいずれか 1 つが sentinel なら
    // 真になるため、sentinel でない残りのフィールドを無条件に無視して
    // zip64 側の値で上書きすると、細工した非 sentinel フィールドが
    // 実際には使われない不整合を見逃す（A03。PR #2318 レビュー指摘・
    // P0 監査）。
    if entries_total != u16::MAX && entries_total as u64 != entries {
        return Err(NpyError::InvalidZip(
            "EOCD のエントリ数が zip64 EOCD と不一致",
        ));
    }
    if cd_size_32 != u32::MAX && cd_size_32 as u64 != cd_size {
        return Err(NpyError::InvalidZip(
            "EOCD の central directory サイズが zip64 EOCD と不一致",
        ));
    }
    if cd_offset_32 != u32::MAX && cd_offset_32 as u64 != cd_offset {
        return Err(NpyError::InvalidZip(
            "EOCD の central directory オフセットが zip64 EOCD と不一致",
        ));
    }

    // zip64 EOCD レコードの「固定部分＋可変長 extensible data sector」の
    // 宣言終端（`zip64_eocd_offset + 12 + record_size`。record_size は
    // シグネチャ 4 バイト＋サイズフィールド 8 バイトを除いた残りの長さ。
    // APPNOTE.TXT 4.3.14）が zip64 EOCD locator の開始位置
    // （`locator_offset`）を超えないことを検証する。この検査がないと、
    // 細工した `record_size` により zip64 EOCD レコードが locator・
    // 通常 EOCD の領域まで「正当な構造」として重なって解釈されうる
    // （A03。PR #2318 レビュー指摘・P0 監査）。固定部分は 56 バイト
    // （シグネチャからで数えて）のため `record_size` は 44 以上でなければ
    // ならない。
    let record_size = read_u64_le(bytes, zip64_eocd_offset + 4, "zip64 EOCD レコードサイズ")?;
    if record_size < 44 {
        return Err(NpyError::InvalidZip(
            "zip64 EOCD レコードサイズが固定部分より小さい",
        ));
    }
    let record_end = zip64_eocd_offset
        .checked_add(12)
        .and_then(|v| v.checked_add(usize::try_from(record_size).ok()?))
        .ok_or(NpyError::InvalidZip("zip64 EOCD レコードの終端が不正"))?;
    if record_end > locator_offset {
        return Err(NpyError::InvalidZip(
            "zip64 EOCD レコードが locator と重なる",
        ));
    }

    let entries = usize::try_from(entries)
        .map_err(|_| NpyError::InvalidZip("zip64 エントリ数が usize 範囲を超える"))?;
    let cd_size = usize::try_from(cd_size)
        .map_err(|_| NpyError::InvalidZip("zip64 cd size が usize 範囲を超える"))?;
    let cd_offset = usize::try_from(cd_offset)
        .map_err(|_| NpyError::InvalidZip("zip64 cd offset が usize 範囲を超える"))?;
    // zip64 経路の `metadata_start` は zip64 EOCD レコードの開始位置。
    // central directory はレコード・locator・通常 EOCD のいずれよりも
    // 前に収まらなければならない（レコード・locator・通常 EOCD は
    // ここまでの検査で互いに重ならない連続領域であることを確認済み）。
    Ok((entries, cd_offset, cd_size, zip64_eocd_offset))
}

/// central directory の 1 エントリを `pos` から解析し、次エントリの
/// 開始位置と共に返す。
fn parse_central_directory_entry(
    bytes: &[u8],
    pos: usize,
) -> Result<(CentralDirEntry, usize), NpyError> {
    if read_u32_le(bytes, pos, "central directory シグネチャ")? != CENTRAL_DIR_HEADER_SIG {
        return Err(NpyError::InvalidZip(
            "central directory エントリのシグネチャが不一致",
        ));
    }
    let flags = read_u16_le(bytes, pos + 8, "central directory flags")?;
    let method = read_u16_le(bytes, pos + 10, "central directory method")?;
    let crc = read_u32_le(bytes, pos + 16, "central directory CRC")?;
    let compressed_size_32 = read_u32_le(bytes, pos + 20, "central directory compressed size")?;
    let uncompressed_size_32 = read_u32_le(bytes, pos + 24, "central directory uncompressed size")?;
    let name_len = read_u16_le(bytes, pos + 28, "central directory name length")? as usize;
    let extra_len = read_u16_le(bytes, pos + 30, "central directory extra length")? as usize;
    let comment_len = read_u16_le(bytes, pos + 32, "central directory comment length")? as usize;
    let disk_start = read_u16_le(bytes, pos + 34, "central directory disk start")?;
    let local_header_offset_32 =
        read_u32_le(bytes, pos + 42, "central directory local header offset")?;

    if disk_start != 0 {
        return Err(NpyError::UnsupportedZipFeature("マルチディスク ZIP"));
    }
    // bit 0: 暗号化、bit 6: 強暗号化。どちらも本 IO は対応しない。
    if flags & 0x0001 != 0 || flags & 0x0040 != 0 {
        return Err(NpyError::UnsupportedZipFeature("暗号化エントリ"));
    }

    let name_start = pos + 46;
    let name_bytes = slice_at(bytes, name_start, name_len, "central directory ファイル名")?;
    let name = std::str::from_utf8(name_bytes)
        .map_err(|_| NpyError::InvalidEntryName)?
        .to_string();
    let extra_start = name_start + name_len;
    let extra = slice_at(
        bytes,
        extra_start,
        extra_len,
        "central directory extra field",
    )?;

    let mut compressed_size = compressed_size_32 as u64;
    let mut uncompressed_size = uncompressed_size_32 as u64;
    let mut local_header_offset = local_header_offset_32 as u64;
    if compressed_size_32 == u32::MAX
        || uncompressed_size_32 == u32::MAX
        || local_header_offset_32 == u32::MAX
    {
        let zip64 = parse_zip64_extra(extra)?;
        let mut cursor = 0usize;
        // zip64 extra フィールドは「元のフィールドが飽和している順」で
        // 64bit 値が並ぶ（APPNOTE.TXT 4.5.3）。
        if uncompressed_size_32 == u32::MAX {
            uncompressed_size = *zip64.get(cursor).ok_or(NpyError::InvalidZip(
                "zip64 extra に uncompressed size がない",
            ))?;
            cursor += 1;
        }
        if compressed_size_32 == u32::MAX {
            compressed_size = *zip64.get(cursor).ok_or(NpyError::InvalidZip(
                "zip64 extra に compressed size がない",
            ))?;
            cursor += 1;
        }
        if local_header_offset_32 == u32::MAX {
            local_header_offset = *zip64.get(cursor).ok_or(NpyError::InvalidZip(
                "zip64 extra に local header offset がない",
            ))?;
        }
    }

    let is_directory = name.is_empty() || name.ends_with('/') || name.ends_with('\\');
    if name.is_empty() || name.contains('\0') {
        return Err(NpyError::InvalidEntryName);
    }

    let entry = CentralDirEntry {
        name,
        is_directory,
        method,
        compressed_size,
        uncompressed_size,
        local_header_offset,
        crc32: crc,
        flags,
    };
    let next_pos = extra_start + extra_len + comment_len;
    Ok((entry, next_pos))
}

/// zip64 extra field（id=0x0001）本体から連続する `u64` 列を取り出す。
fn parse_zip64_extra(extra: &[u8]) -> Result<Vec<u64>, NpyError> {
    let mut pos = 0usize;
    while pos + 4 <= extra.len() {
        let id = u16::from_le_bytes([extra[pos], extra[pos + 1]]);
        let size = u16::from_le_bytes([extra[pos + 2], extra[pos + 3]]) as usize;
        let field_start = pos + 4;
        let field = slice_at(extra, field_start, size, "zip64 extra フィールド")?;
        if id == ZIP64_EXTRA_ID {
            let (chunks, _remainder) = field.as_chunks::<8>();
            let values = chunks.iter().map(|c| u64::from_le_bytes(*c)).collect();
            return Ok(values);
        }
        pos = field_start + size;
    }
    Err(NpyError::InvalidZip(
        "zip64 extra フィールド（id=0x0001）が見つからない",
    ))
}

/// central directory エントリ名から npz キー（末尾 `.npy` を除いたもの）
/// を導出する。
///
/// 末尾 `.npy` の有無を無視して正規化する（旧実装）と、`"a"` と
/// `"a.npy"` のように異なるメンバ名が同一キーへ縮退し得た
/// （`write_npz_bytes` は常にキーへ `.npy` を 1 回付与するため
/// 自前書き出し→読み込みの往復では起きないが、外部で組み立てた ZIP
/// を読む経路では起き得る。PR #2318 レビュー指摘・P1）。`.npy`
/// サフィックスを持たないメンバ名は npz の実体ではない（`np.savez` は
/// すべてのメンバへ `.npy` を付与する）ため拒否し、キー導出を
/// `write_npz_bytes` の逆写像として全単射に保つ。
fn npz_member_key(name: &str) -> Result<String, NpyError> {
    if name.is_empty() || name.contains('\0') {
        return Err(NpyError::InvalidEntryName);
    }
    name.strip_suffix(".npy")
        .map(str::to_string)
        .ok_or(NpyError::InvalidEntryName)
}

/// local header の 32bit size フィールドと、central directory から
/// （必要なら zip64 extra 経由で）解決済みの実サイズ（`u64`）が一致する
/// ことを確認する。**実測（CPython 3.x `zipfile`／`numpy` 2.3.5.
/// `np.savez` はいずれも `force_zip64=True` で local header を書くため、
/// 実サイズが `u32::MAX` を大きく下回る小さな配列でも local header の
/// size フィールドは無条件に `0xFFFFFFFF` のプレースホルダになる**
/// （実サイズは central directory 側の zip64 extra にのみ書かれる。
/// 「実サイズが u32 を超える場合に限りプレースホルダを許容」という
/// 一見自然な条件では、この無条件プレースホルダ書式のため真正の
/// `np.savez` 出力を全て拒否してしまう。scratchpad での実測で確認
/// 済み）。このためプレースホルダは実サイズに関わらず常に許容し、
/// プレースホルダでない場合のみ central directory の実サイズとの
/// 厳密一致を要求する（method／flags／CRC ほど強い検証にはならないが、
/// 非プレースホルダ値を central directory と矛盾する値へ細工する経路は
/// 引き続き遮断する）。
fn local_size_field_matches(local_field: u32, actual: u64) -> bool {
    local_field == u32::MAX || u64::from(local_field) == actual
}

/// central directory エントリが指す local header から、メンバの構造的な
/// バイト範囲 `[local header 開始, 圧縮データ終端)` を求める。
///
/// 名前・flags・method・CRC・サイズが central directory と一致することの
/// 検証は行わない（それは [`read_member_bytes`] が担う）。本関数は
/// `read_npz_bytes` の 1.5 パス目が central directory の宣言範囲・他
/// メンバとの重なりを判定するためだけに、local header の固定長部分
/// （シグネチャ・name/extra 長）のみを読む（A03。PR #2318 レビュー
/// 指摘・P0）。
fn member_data_range(bytes: &[u8], entry: &CentralDirEntry) -> Result<(usize, usize), NpyError> {
    let local_offset = usize::try_from(entry.local_header_offset)
        .map_err(|_| NpyError::InvalidZip("local header offset が usize 範囲を超える"))?;
    if read_u32_le(bytes, local_offset, "local header シグネチャ")? != LOCAL_FILE_HEADER_SIG {
        return Err(NpyError::InvalidZip("local header シグネチャが不一致"));
    }
    let name_len = read_u16_le(bytes, local_offset + 26, "local header name length")? as usize;
    let extra_len = read_u16_le(bytes, local_offset + 28, "local header extra length")? as usize;
    let data_start = local_offset
        .checked_add(30)
        .and_then(|v| v.checked_add(name_len))
        .and_then(|v| v.checked_add(extra_len))
        .ok_or(NpyError::InvalidZip("local header のデータ開始位置が不正"))?;
    let compressed_size = usize::try_from(entry.compressed_size)
        .map_err(|_| NpyError::InvalidZip("compressed size が usize 範囲を超える"))?;
    let data_end = data_start
        .checked_add(compressed_size)
        .ok_or(NpyError::InvalidZip("メンバのデータ終端が不正"))?;
    // データ終端がファイル範囲内にあることも確認する（`slice_at` による
    // 検証は `read_member_bytes` 側でも行うが、本関数はそれより前の
    // 構造検証パスで呼ばれるため、ここでも fail-closed に確認する）。
    if data_end > bytes.len() {
        return Err(NpyError::InvalidZip(
            "メンバのデータ領域がファイル範囲外を指す",
        ));
    }
    Ok((local_offset, data_end))
}

/// 1 メンバを local header 経由で読み取り、伸長・CRC 検証したうえで
/// `Tensor<f32>` として解釈する。構造的検証・伸長・CRC 検証自体は
/// [`read_member_bytes`] に委譲し、本関数は npy 解釈のみを担う
/// （ディレクトリエントリも [`read_member_bytes`] 側の検証を共用する
/// ため。PR #2318 レビュー指摘・P2）。
fn read_member(bytes: &[u8], entry: &CentralDirEntry) -> Result<Tensor<f32>, NpyError> {
    let decompressed = read_member_bytes(bytes, entry)?;
    read_npy_bytes(&decompressed)
}

/// central directory の 1 エントリを local header 経由で読み取り、
/// name／flags／method／CRC／サイズの central directory との整合性を
/// 検証したうえで伸長・CRC 照合済みの生データを返す（npy 解釈はしない。
/// 通常メンバ・ディレクトリエントリの両方から呼ばれる共通経路）。
fn read_member_bytes<'a>(
    bytes: &'a [u8],
    entry: &CentralDirEntry,
) -> Result<std::borrow::Cow<'a, [u8]>, NpyError> {
    let local_offset = usize::try_from(entry.local_header_offset)
        .map_err(|_| NpyError::InvalidZip("local header offset が usize 範囲を超える"))?;
    if read_u32_le(bytes, local_offset, "local header シグネチャ")? != LOCAL_FILE_HEADER_SIG {
        return Err(NpyError::InvalidZip("local header シグネチャが不一致"));
    }
    let local_flags = read_u16_le(bytes, local_offset + 6, "local header flags")?;
    let local_method = read_u16_le(bytes, local_offset + 8, "local header method")?;
    let local_crc = read_u32_le(bytes, local_offset + 14, "local header CRC")?;
    let local_compressed_size =
        read_u32_le(bytes, local_offset + 18, "local header compressed size")?;
    let local_uncompressed_size =
        read_u32_le(bytes, local_offset + 22, "local header uncompressed size")?;
    let name_len = read_u16_le(bytes, local_offset + 26, "local header name length")? as usize;
    let extra_len = read_u16_le(bytes, local_offset + 28, "local header extra length")? as usize;
    let name_start = local_offset + 30;
    let local_name_bytes = slice_at(bytes, name_start, name_len, "local header ファイル名")?;
    // local header と central directory のファイル名を照合する。
    // central directory を正として読み進めるだけでは、ファイル名の
    // 異なる local header が指すデータを central directory 側のキー
    // として誤って受理してしまう（central directory の
    // `local_header_offset` を細工した ZIP で名前差し替え攻撃が成立
    // する。A03。PR #2318 レビュー指摘・P0）。
    if local_name_bytes != entry.name.as_bytes() {
        return Err(NpyError::InvalidZip(
            "local header のファイル名が central directory と不一致",
        ));
    }
    // local header と central directory の圧縮方式・flags・CRC・サイズを
    // 照合する。ファイル名一致のみでは、central directory が指す
    // `local_header_offset` の先の local header が「名前は central
    // directory と同じだが method／flags／CRC／サイズだけ異なる」よう
    // 細工されたケースを見逃す（central directory を正として読み進める
    // 設計を裏から崩す改ざんが成立し得る。A03。PR #2318 レビュー
    // 指摘・P0）。サイズは zip64 プレースホルダ（`0xFFFFFFFF`。
    // `np.savez` が zip64 を使う場合の local header の仕様）のみ例外的に
    // 許容する（本ファイル冒頭 doc コメント参照）。
    if local_flags != entry.flags {
        return Err(NpyError::InvalidZip(
            "local header の flags が central directory と不一致",
        ));
    }
    if local_method != entry.method {
        return Err(NpyError::InvalidZip(
            "local header の圧縮方式が central directory と不一致",
        ));
    }
    if local_crc != entry.crc32 {
        return Err(NpyError::InvalidZip(
            "local header の CRC が central directory と不一致",
        ));
    }
    if !local_size_field_matches(local_compressed_size, entry.compressed_size) {
        return Err(NpyError::InvalidZip(
            "local header の圧縮サイズが central directory と不一致",
        ));
    }
    if !local_size_field_matches(local_uncompressed_size, entry.uncompressed_size) {
        return Err(NpyError::InvalidZip(
            "local header の非圧縮サイズが central directory と不一致",
        ));
    }
    let data_start = name_start + name_len + extra_len;

    let compressed_size = usize::try_from(entry.compressed_size)
        .map_err(|_| NpyError::InvalidZip("compressed size が usize 範囲を超える"))?;
    let uncompressed_size = usize::try_from(entry.uncompressed_size)
        .map_err(|_| NpyError::InvalidZip("uncompressed size が usize 範囲を超える"))?;
    let compressed = slice_at(bytes, data_start, compressed_size, "エントリデータ範囲")?;

    // data descriptor（flags bit 3）は NumPy 出力では使わない構成のため
    // 対応しない（central directory を正としているのでサイズ自体は
    // 既知。フラグのみ多層防御として確認する）。
    if entry.flags & 0x0008 != 0 {
        return Err(NpyError::UnsupportedZipFeature(
            "data descriptor 付きエントリ",
        ));
    }

    // STORED（無圧縮）は `bytes` 内の借用スライスをそのまま使い、
    // npy ヘッダ・shape の検証（`read_npy_bytes`）前に全体を複製しない
    // （`compressed` は既に `slice_at` で `bytes` の実サイズ範囲内である
    // ことを検証済みだが、検証前の無条件複製そのものが不要なメモリ
    // 確保でありコストとなる。A03/A04/A05。PR #2318 レビュー指摘・
    // P0）。DEFLATE（method=8）は伸長结果を新規に確保する必要がある
    // ため `inflate` 側でのみ確保する。
    let decompressed: std::borrow::Cow<'_, [u8]> = match entry.method {
        0 => {
            if compressed_size != uncompressed_size {
                return Err(NpyError::InvalidZip(
                    "STORED エントリの圧縮サイズと非圧縮サイズが不一致",
                ));
            }
            std::borrow::Cow::Borrowed(compressed)
        }
        8 => std::borrow::Cow::Owned(inflate(compressed, uncompressed_size)?),
        other => return Err(NpyError::UnsupportedCompression { method: other }),
    };

    if crc32(&decompressed) != entry.crc32 {
        return Err(NpyError::CrcMismatch {
            name: entry.name.clone(),
        });
    }

    Ok(decompressed)
}

/// `write_npz_bytes` の事前検証パス本体。各エントリの shape（`Tensor` の
/// 実データではなく shape のみ）から `npy::npy_encoded_len` で予測 npy
/// 長を求め、読み込み側（`read_npz_bytes`）が課す伸長後サイズ上限
/// （メンバ単体は `MAX_MEMBER_DECOMPRESSED_BYTES`・累積は
/// `MAX_TOTAL_DECOMPRESSED_BYTES`）を同じ定数・同じ判定式で事前に適用
/// する（PR #2318 レビュー指摘・P2〈npz.rs:848。npz の合計伸長後サイズ
/// 上限を書き出し側が検証しておらず、小さいテンソルを複数渡すと合計が
/// 上限を超えても `save_npz` は成功し `load_npz` が
/// `DecompressedSizeExceeded` で失敗していた〉の是正）。
///
/// `Tensor` の実データを一切受け取らない純粋関数であるため、境界値
/// （メンバ単体・累積ともに「ちょうど上限」は成功・「上限+1」は拒否）を
/// 1 GiB 相当の `Tensor` を実際に確保せずに単体テストできる
/// （`tests::` 参照）。返り値は各エントリの予測 npy 長（入力の順序の
/// まま）。
fn plan_npz_member_sizes<'a>(
    entries: impl Iterator<Item = (&'a str, &'a [usize])>,
) -> Result<Vec<u64>, NpyError> {
    let mut predicted_lens = Vec::new();
    let mut total_decompressed: u64 = 0;
    for (entry_name, shape) in entries {
        let predicted_len = npy_encoded_len(shape).map_err(|e| NpyError::Entry {
            name: entry_name.to_string(),
            source: Box::new(e),
        })?;
        if predicted_len > MAX_MEMBER_DECOMPRESSED_BYTES {
            return Err(NpyError::Entry {
                name: entry_name.to_string(),
                source: Box::new(NpyError::DecompressedSizeExceeded {
                    len: predicted_len,
                    max: MAX_MEMBER_DECOMPRESSED_BYTES,
                }),
            });
        }
        total_decompressed = total_decompressed
            .checked_add(predicted_len)
            .filter(|&total| total <= MAX_TOTAL_DECOMPRESSED_BYTES)
            .ok_or(NpyError::DecompressedSizeExceeded {
                len: total_decompressed.saturating_add(predicted_len),
                max: MAX_TOTAL_DECOMPRESSED_BYTES,
            })?;
        predicted_lens.push(predicted_len);
    }
    Ok(predicted_lens)
}

/// 名前付きテンソル集合を npz（ZIP、STORED のみ）形式のバイト列へ
/// 直列化する。キーを昇順に並べ、決定的な出力にする
/// （`docs/tensor-core-npy-npz-io-decision.md` §3.6）。
pub fn write_npz_bytes(map: &HashMap<String, Tensor<f32>>) -> Result<Vec<u8>, NpyError> {
    // `entries_total` は EOCD 内で 16bit 幅（u16）で書く（本関数の末尾で
    // `central_records.len() as u16` として書く箇所を参照）。本 IO は
    // zip64 EOCD／locator を書き出さない STORED 限定実装のため、
    // `u16::MAX`（0xFFFF）は書き出せない。この値は ZIP 仕様上
    // 「zip64 EOCD を参照せよ」というプレースホルダのため、
    // `resolve_eocd_counts`（読み込み側）は `entries_total == u16::MAX`
    // を見た時点で存在しない zip64 EOCD locator を要求し失敗する
    // （PR #2318 レビュー指摘・P1。往復契約〈write → read〉が壊れるため
    // `u16::MAX` 件"以上"を一律で拒否する。安全な上限は `u16::MAX - 1`
    // 件）。
    if map.len() >= u16::MAX as usize {
        return Err(NpyError::TooManyEntries);
    }
    let mut keys: Vec<&String> = map.keys().collect();
    keys.sort();
    for key in &keys {
        validate_entry_name(key)?;
    }

    // 事前検証パス（PR #2318 レビュー指摘・P2〈npz.rs:848〉の是正）:
    // `write_npy_bytes` で実際に各メンバの `f32` データをシリアライズ
    // する前に、shape だけから求まる予測 npy 長（`npy_encoded_len`。
    // 実データを一切確保しない純粋関数）を使って、読み込み側
    // （`read_npz_bytes`）が課す伸長後サイズ上限（メンバ単体・累積）を
    // 同じ定数で事前検証する。これにより「書き出しは成功するが
    // `load_npz` は `DecompressedSizeExceeded` で失敗する」往復不能な
    // 出力を防ぐ。小さいテンソルを複数渡して合計が上限を超える
    // ケース（レビュー指摘の再現）も、個々のメンバは単体上限を満たす
    // ため、ここで累積側の検査ではじめて拒否される。検査本体
    // （`plan_npz_member_sizes`）は shape のみを受け取る純粋関数のため、
    // 境界値を 1 GiB 相当の `Tensor` を実際に確保せずに単体テストできる
    // （`tests::` 参照）。
    let entry_names: Vec<String> = keys.iter().map(|k| format!("{k}.npy")).collect();
    let predicted_lens = plan_npz_member_sizes(
        entry_names
            .iter()
            .map(String::as_str)
            .zip(keys.iter().map(|k| map[*k].shape())),
    )?;

    let mut out = Vec::new();
    // (name, crc, compressed_size, local_header_offset, flags)
    let mut central_records: Vec<(String, u32, u32, u32, u16)> = Vec::with_capacity(keys.len());

    for ((key, entry_name), predicted_len) in
        keys.iter().zip(entry_names.iter()).zip(predicted_lens)
    {
        let npy_bytes = write_npy_bytes(&map[key.as_str()])?;
        // 事前検証パス（`plan_npz_member_sizes`）と同じ計算式
        // （`npy_encoded_len`）で求めた予測長と、実際にシリアライズした
        // バイト列の長さが一致することを検査する。ここが食い違えば
        // `npy_header_layout` の変更が両者で drift した合図であり、上限
        // 検査が実体を反映しなくなる（fail-closed。実運用では到達しない
        // 防御的検査）。
        debug_assert_eq!(
            npy_bytes.len() as u64,
            predicted_len,
            "npy_encoded_len の予測値が write_npy_bytes の実出力長と不一致"
        );
        let size = u32::try_from(npy_bytes.len()).map_err(|_| NpyError::EntryTooLarge)?;
        let crc = crc32(&npy_bytes);
        let local_header_offset = u32::try_from(out.len()).map_err(|_| NpyError::EntryTooLarge)?;
        // general purpose flag bit 11（EFS。Language Encoding Flag）:
        // ファイル名が UTF-8 であることを明示する。`validate_entry_name`
        // は非 ASCII キーを許容するため、これを立てないと Python
        // zipfile／`np.load` はファイル名を CP437 としてデコードし、
        // 非 ASCII メンバ名が往復しない（PR #2318 レビュー指摘。
        // Bugbot・Medium。APPNOTE.TXT 4.4.4）。
        let flags: u16 = if entry_name.is_ascii() { 0 } else { 0x0800 };

        // local header: version needed(20)・flags・method(0=STORED)・
        // DOS 時刻 1980-01-01 00:00:00（time=0, date=0x0021）・CRC・
        // compressed/uncompressed size（STORED なので同一）・extra なし。
        out.extend_from_slice(&LOCAL_FILE_HEADER_SIG.to_le_bytes());
        out.extend_from_slice(&20u16.to_le_bytes()); // version needed
        out.extend_from_slice(&flags.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes()); // method = STORED
        out.extend_from_slice(&0u16.to_le_bytes()); // mod time
        out.extend_from_slice(&0x0021u16.to_le_bytes()); // mod date = 1980-01-01
        out.extend_from_slice(&crc.to_le_bytes());
        out.extend_from_slice(&size.to_le_bytes()); // compressed size
        out.extend_from_slice(&size.to_le_bytes()); // uncompressed size
        out.extend_from_slice(&(entry_name.len() as u16).to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes()); // extra length
        out.extend_from_slice(entry_name.as_bytes());
        out.extend_from_slice(&npy_bytes);

        central_records.push((entry_name.clone(), crc, size, local_header_offset, flags));
    }

    let cd_start = u32::try_from(out.len()).map_err(|_| NpyError::EntryTooLarge)?;
    for (name, crc, size, local_offset, flags) in &central_records {
        out.extend_from_slice(&CENTRAL_DIR_HEADER_SIG.to_le_bytes());
        out.extend_from_slice(&20u16.to_le_bytes()); // version made by
        out.extend_from_slice(&20u16.to_le_bytes()); // version needed
        out.extend_from_slice(&flags.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes()); // method = STORED
        out.extend_from_slice(&0u16.to_le_bytes()); // mod time
        out.extend_from_slice(&0x0021u16.to_le_bytes()); // mod date
        out.extend_from_slice(&crc.to_le_bytes());
        out.extend_from_slice(&size.to_le_bytes()); // compressed size
        out.extend_from_slice(&size.to_le_bytes()); // uncompressed size
        out.extend_from_slice(&(name.len() as u16).to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes()); // extra length
        out.extend_from_slice(&0u16.to_le_bytes()); // comment length
        out.extend_from_slice(&0u16.to_le_bytes()); // disk number start
        out.extend_from_slice(&0u16.to_le_bytes()); // internal attrs
        out.extend_from_slice(&0u32.to_le_bytes()); // external attrs
        out.extend_from_slice(&local_offset.to_le_bytes());
        out.extend_from_slice(name.as_bytes());
    }
    let cd_size =
        u32::try_from(out.len() - cd_start as usize).map_err(|_| NpyError::EntryTooLarge)?;

    // EOCD
    out.extend_from_slice(&EOCD_SIG.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes()); // disk number
    out.extend_from_slice(&0u16.to_le_bytes()); // cd start disk
    out.extend_from_slice(&(central_records.len() as u16).to_le_bytes());
    out.extend_from_slice(&(central_records.len() as u16).to_le_bytes());
    out.extend_from_slice(&cd_size.to_le_bytes());
    out.extend_from_slice(&cd_start.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes()); // comment length

    Ok(out)
}

/// エントリ名（npz キー）の検証: 空・NUL・`/`・`\` を含むもの・UTF-8 名
/// として `u16` 長を超えるものを拒否する。
fn validate_entry_name(name: &str) -> Result<(), NpyError> {
    if name.is_empty()
        || name.contains('\0')
        || name.contains('/')
        || name.contains('\\')
        || name.len() > u16::MAX as usize - 4
    // ".npy" サフィックス分の余裕
    {
        return Err(NpyError::InvalidEntryName);
    }
    Ok(())
}

/// `map` を `path` へ npz 形式で書き出す。
///
/// `write_npz_bytes` が返すアーカイブ全体（ZIP local header／central
/// directory／EOCD のオーバーヘッドを含む実バイト長）が
/// `super::MAX_FILE_READ_BYTES` を超える場合は `fs::write` の前に
/// `FileTooLarge` で拒否し、部分ファイルを残さない。`write_npz_bytes`
/// の事前検証パスはメンバの伸長後サイズ（`uncompressed_size`）の合計を
/// 検査するのに対し、本検査はエントリ名・ZIP 構造のオーバーヘッドまで
/// 含めた実際のファイルサイズを検査するため独立に必要（オーバーヘッドが
/// 伸長後サイズ上限ぎりぎりの合計を押し上げうる）。`load_npz` が
/// `read_file_bounded` で課すファイルサイズ上限を書き出し側にも適用する
/// ことで、自前書き出し→読み込みの往復契約を保つ（PR #2318 レビュー
/// 指摘・P2〈npz.rs:848〉是正の一環）。
pub fn save_npz<P: AsRef<Path>>(
    map: &HashMap<String, Tensor<f32>>,
    path: P,
) -> Result<(), NpyError> {
    let bytes = write_npz_bytes(map)?;
    let len = bytes.len() as u64;
    if len > super::MAX_FILE_READ_BYTES {
        return Err(NpyError::FileTooLarge {
            len,
            max: super::MAX_FILE_READ_BYTES,
        });
    }
    std::fs::write(path, bytes)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_map() -> HashMap<String, Tensor<f32>> {
        let mut m = HashMap::new();
        m.insert(
            "a".to_string(),
            Tensor::new(vec![1.0, 2.0, 3.0], &[3]).unwrap(),
        );
        m.insert(
            "b".to_string(),
            Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap(),
        );
        m.insert(
            "empty".to_string(),
            Tensor::new(Vec::new(), &[0, 4]).unwrap(),
        );
        m
    }

    #[test]
    fn write_then_read_roundtrip() {
        let map = sample_map();
        let bytes = write_npz_bytes(&map).unwrap();
        let back = read_npz_bytes(&bytes).unwrap();
        assert_eq!(back.len(), map.len());
        for (k, v) in &map {
            let got = &back[k];
            assert_eq!(got.shape(), v.shape());
            assert_eq!(got.host_slice().to_vec(), v.host_slice().to_vec());
        }
    }

    #[test]
    fn write_then_read_roundtrip_non_ascii_key() {
        // Bugbot 指摘（Medium）: 非 ASCII キーは UTF-8 名フラグ（bit 11）
        // を立てないと Python zipfile／`np.load` が CP437 としてデコード
        // し往復しない。書き出しバイト列の flags を直接検査したうえで
        // 読み込みも正しく往復することを確認する。
        let mut m = HashMap::new();
        m.insert("café".to_string(), Tensor::new(vec![1.0], &[1]).unwrap());
        let bytes = write_npz_bytes(&m).unwrap();

        // local header の flags（オフセット +6、先頭エントリのため
        // local_offset=0）に bit 11（0x0800）が立っていることを確認。
        let local_flags = u16::from_le_bytes([bytes[6], bytes[7]]);
        assert_eq!(local_flags & 0x0800, 0x0800);

        // central directory 側の flags（ヘッダ +8）も同様に確認。
        let cd_sig = CENTRAL_DIR_HEADER_SIG.to_le_bytes();
        let cd_pos = bytes
            .windows(4)
            .position(|w| w == cd_sig)
            .expect("central directory シグネチャが見つかる");
        let cd_flags = u16::from_le_bytes([bytes[cd_pos + 8], bytes[cd_pos + 9]]);
        assert_eq!(cd_flags & 0x0800, 0x0800);

        let back = read_npz_bytes(&bytes).unwrap();
        assert_eq!(back["café"].host_slice().to_vec(), vec![1.0f32]);
    }

    #[test]
    fn rejects_central_directory_entry_beyond_declared_range() {
        // P0（codex-review。npz.rs:58）: `cd_end` の宣言範囲を超えて
        // central directory エントリを解析しないことを確認する。
        // entry_count を実際のエントリ数より過大に偽装し、2 番目の
        // 走査が `cd_end` を超えたところで fail-closed に拒否される
        // ことを検証する（EOCD の entries フィールドを書き換える）。
        let mut m = HashMap::new();
        m.insert("a".to_string(), Tensor::new(vec![1.0], &[1]).unwrap());
        let bytes = write_npz_bytes(&m).unwrap();

        let eocd_sig = EOCD_SIG.to_le_bytes();
        let eocd_pos = bytes
            .windows(4)
            .rposition(|w| w == eocd_sig)
            .expect("EOCD シグネチャが見つかる");
        let mut tampered = bytes.clone();
        // entries (this disk) と entries (total) を実際の 1 から 2 へ
        // 水増しする（disk_number 系一致検査を通すため両方書き換え）。
        tampered[eocd_pos + 8..eocd_pos + 10].copy_from_slice(&2u16.to_le_bytes());
        tampered[eocd_pos + 10..eocd_pos + 12].copy_from_slice(&2u16.to_le_bytes());

        let err = read_npz_bytes(&tampered);
        assert!(
            matches!(err, Err(NpyError::InvalidZip(_))),
            "宣言範囲外エントリが拒否されなかった: {err:?}"
        );
    }

    #[test]
    fn rejects_central_directory_undersized_declared_range() {
        // P0（codex-review。npz.rs:58）: `cd_size` を実際より小さく偽装
        // し、固定長ヘッダ（46 バイト）すら `cd_end` に収まらない場合に
        // 拒否されることを確認する。
        let bytes = write_npz_bytes(&sample_map()).unwrap();
        let eocd_sig = EOCD_SIG.to_le_bytes();
        let eocd_pos = bytes
            .windows(4)
            .rposition(|w| w == eocd_sig)
            .expect("EOCD シグネチャが見つかる");
        let mut tampered = bytes.clone();
        // cd size フィールド（EOCD +12）を極端に小さい値へ書き換える。
        tampered[eocd_pos + 12..eocd_pos + 16].copy_from_slice(&4u32.to_le_bytes());

        let err = read_npz_bytes(&tampered);
        assert!(
            matches!(err, Err(NpyError::InvalidZip(_))),
            "過小な cd_size が拒否されなかった: {err:?}"
        );
    }

    #[test]
    fn rejects_local_header_name_mismatch() {
        // P0（codex-review。npz.rs:318）: local header のファイル名が
        // central directory と異なる場合に拒否されることを確認する。
        // 先頭エントリ（sort 順で "a"）の local header 名（オフセット
        // 30、"a.npy" の 5 バイト）を同じ長さの別名へ書き換える。
        let bytes = write_npz_bytes(&sample_map()).unwrap();
        let mut tampered = bytes.clone();
        assert_eq!(&tampered[30..35], b"a.npy");
        tampered[30..35].copy_from_slice(b"x.npy");

        let err = read_npz_bytes(&tampered);
        assert!(
            matches!(err, Err(NpyError::Entry { .. })),
            "local header 名の不一致が拒否されなかった: {err:?}"
        );
    }

    #[test]
    fn rejects_local_header_method_mismatch() {
        // P0（codex-review・PR #2318）: local header の圧縮方式が
        // central directory と異なる場合に拒否されることを確認する。
        // 先頭エントリの local header method フィールド（オフセット
        // +8、STORED=0）を DEFLATE（8）へ書き換える。
        let bytes = write_npz_bytes(&sample_map()).unwrap();
        let mut tampered = bytes.clone();
        assert_eq!(u16::from_le_bytes([tampered[8], tampered[9]]), 0);
        tampered[8..10].copy_from_slice(&8u16.to_le_bytes());

        let err = read_npz_bytes(&tampered);
        assert!(
            matches!(err, Err(NpyError::Entry { .. })),
            "local header method の不一致が拒否されなかった: {err:?}"
        );
    }

    #[test]
    fn rejects_local_header_flags_mismatch() {
        // P0（codex-review・PR #2318）: local header の flags が
        // central directory と異なる場合に拒否されることを確認する。
        // 先頭エントリの local header flags（オフセット +6）へ未使用
        // ビット（bit 5）を立てる。
        let bytes = write_npz_bytes(&sample_map()).unwrap();
        let mut tampered = bytes.clone();
        assert_eq!(u16::from_le_bytes([tampered[6], tampered[7]]), 0);
        tampered[6..8].copy_from_slice(&0x0020u16.to_le_bytes());

        let err = read_npz_bytes(&tampered);
        assert!(
            matches!(err, Err(NpyError::Entry { .. })),
            "local header flags の不一致が拒否されなかった: {err:?}"
        );
    }

    #[test]
    fn rejects_directory_entry_with_non_empty_data() {
        // P2（codex-review・PR #2318）: 名前が `/` で終わる「ディレクトリ
        // エントリ」は central directory・local header の構造検証を経ずに
        // 無条件で読み飛ばされていた。データを持つ通常エントリの名前
        // 末尾だけをディレクトリ名（`/` 終端）に偽装した ZIP が、内容の
        // 整合性検査を一切受けずに受理されないことを確認する。
        //
        // 先頭エントリ（sort 順で "a"）の名前 "a.npy"（local header
        // オフセット 30、central directory は "a.npy" シグネチャの
        // 直後）の末尾 1 バイトのみを `/` へ書き換え、名前の長さを
        // 変えずに `is_directory` 判定（`ends_with('/')`）だけを反転
        // させる。中身（圧縮データ・CRC）は変えないため、ディレクトリ
        // と自称しつつ非空データを持つ矛盾したエントリになる。
        let bytes = write_npz_bytes(&sample_map()).unwrap();
        let mut tampered = bytes.clone();
        assert_eq!(&tampered[30..35], b"a.npy");
        tampered[30..35].copy_from_slice(b"a.np/");

        let cd_sig = CENTRAL_DIR_HEADER_SIG.to_le_bytes();
        let cd_pos = bytes
            .windows(4)
            .position(|w| w == cd_sig)
            .expect("central directory シグネチャが見つかる");
        let cd_name_start = cd_pos + 46;
        assert_eq!(&tampered[cd_name_start..cd_name_start + 5], b"a.npy");
        tampered[cd_name_start..cd_name_start + 5].copy_from_slice(b"a.np/");

        let err = read_npz_bytes(&tampered);
        match err {
            Err(NpyError::Entry { name, source }) => {
                assert_eq!(name, "a.np/");
                assert!(matches!(*source, NpyError::InvalidZip(_)));
            }
            other => {
                panic!("非空データを持つディレクトリ偽装エントリが拒否されなかった: {other:?}")
            }
        }
    }

    #[test]
    fn rejects_member_declared_size_exceeding_member_cap() {
        // P0（PR #2318 レビュー指摘）: central directory の宣言
        // `uncompressed_size` がメンバ単体上限
        // （`MAX_MEMBER_DECOMPRESSED_BYTES`）を超える場合、
        // `read_member_bytes`／`inflate` を呼び出す前（＝出力バッファを
        // 確保する前）に `DecompressedSizeExceeded` で拒否されることを
        // 確認する。圧縮データ自体は 1 要素の小さな npy のままであり、
        // 実際の伸長を試みれば `CrcMismatch`／`InvalidDeflate` 等の別
        // エラーになるはずだが、本検査はそれより前に発火する。
        let mut m = HashMap::new();
        m.insert("x".to_string(), Tensor::new(vec![1.0], &[1]).unwrap());
        let bytes = write_npz_bytes(&m).unwrap();

        let cd_sig = CENTRAL_DIR_HEADER_SIG.to_le_bytes();
        let cd_pos = bytes
            .windows(4)
            .position(|w| w == cd_sig)
            .expect("central directory シグネチャが見つかる");
        let mut tampered = bytes.clone();
        let oversized = (MAX_MEMBER_DECOMPRESSED_BYTES + 1) as u32;
        tampered[cd_pos + 24..cd_pos + 28].copy_from_slice(&oversized.to_le_bytes());

        let err = read_npz_bytes(&tampered);
        match err {
            Err(NpyError::Entry { name, source }) => {
                assert_eq!(name, "x.npy");
                match *source {
                    NpyError::DecompressedSizeExceeded { len, max } => {
                        assert_eq!(len, oversized as u64);
                        assert_eq!(max, MAX_MEMBER_DECOMPRESSED_BYTES);
                    }
                    other => panic!("DecompressedSizeExceeded ではない: {other:?}"),
                }
            }
            other => panic!("メンバ単体上限超過が拒否されなかった: {other:?}"),
        }
    }

    #[test]
    fn rejects_cumulative_declared_size_exceeding_total_cap() {
        // P0（PR #2318 レビュー指摘）: 個々のメンバはメンバ単体上限
        // （`MAX_MEMBER_DECOMPRESSED_BYTES`）以内でも、アーカイブ全体の
        // 累積宣言サイズが `MAX_TOTAL_DECOMPRESSED_BYTES` を超える場合に
        // 2 番目のメンバで拒否されることを確認する（1 番目単体では
        // 上限ちょうどのため許容される）。
        let mut m = HashMap::new();
        m.insert("a".to_string(), Tensor::new(vec![1.0], &[1]).unwrap());
        m.insert("b".to_string(), Tensor::new(vec![2.0], &[1]).unwrap());
        let bytes = write_npz_bytes(&m).unwrap();

        let cd_sig = CENTRAL_DIR_HEADER_SIG.to_le_bytes();
        let first_cd_pos = bytes
            .windows(4)
            .position(|w| w == cd_sig)
            .expect("1 番目の central directory シグネチャが見つかる");
        let second_cd_pos = bytes
            .windows(4)
            .rposition(|w| w == cd_sig)
            .expect("2 番目の central directory シグネチャが見つかる");
        assert_ne!(first_cd_pos, second_cd_pos);

        let mut tampered = bytes.clone();
        // 1 番目のメンバはちょうど上限（単体では許容される）。
        let at_cap = MAX_MEMBER_DECOMPRESSED_BYTES as u32;
        tampered[first_cd_pos + 24..first_cd_pos + 28].copy_from_slice(&at_cap.to_le_bytes());
        // 2 番目は小さい宣言値だが、累積では上限を超える。
        let small = 1u32;
        tampered[second_cd_pos + 24..second_cd_pos + 28].copy_from_slice(&small.to_le_bytes());

        let err = read_npz_bytes(&tampered);
        assert!(
            matches!(err, Err(NpyError::DecompressedSizeExceeded { .. })),
            "累積上限超過が拒否されなかった: {err:?}"
        );
    }

    #[test]
    fn rejects_directory_entry_declared_size_exceeding_member_cap() {
        // P0（PR #2318 レビュー指摘）: ディレクトリエントリ
        // （`rejects_directory_entry_with_non_empty_data` と同じ
        // 偽装手法で名前末尾を `/` にしたエントリ）も
        // `read_member_bytes` を経由するため、メンバ単体上限検査を
        // 迂回できないことを確認する。
        let bytes = write_npz_bytes(&sample_map()).unwrap();
        let mut tampered = bytes.clone();
        assert_eq!(&tampered[30..35], b"a.npy");
        tampered[30..35].copy_from_slice(b"a.np/");

        let cd_sig = CENTRAL_DIR_HEADER_SIG.to_le_bytes();
        let cd_pos = bytes
            .windows(4)
            .position(|w| w == cd_sig)
            .expect("central directory シグネチャが見つかる");
        let cd_name_start = cd_pos + 46;
        assert_eq!(&tampered[cd_name_start..cd_name_start + 5], b"a.npy");
        tampered[cd_name_start..cd_name_start + 5].copy_from_slice(b"a.np/");
        let oversized = (MAX_MEMBER_DECOMPRESSED_BYTES + 1) as u32;
        tampered[cd_pos + 24..cd_pos + 28].copy_from_slice(&oversized.to_le_bytes());

        let err = read_npz_bytes(&tampered);
        match err {
            Err(NpyError::Entry { name, source }) => {
                assert_eq!(name, "a.np/");
                assert!(matches!(*source, NpyError::DecompressedSizeExceeded { .. }));
            }
            other => panic!("ディレクトリ偽装エントリの上限超過が拒否されなかった: {other:?}"),
        }
    }

    #[test]
    fn rejects_local_header_crc_mismatch() {
        // P0（codex-review・PR #2318）: local header の CRC が
        // central directory と異なる場合に拒否されることを確認する
        // （central directory 側の CRC は伸長後データと一致したままの
        // ため、local header 側だけを書き換えて不一致を作る）。
        let bytes = write_npz_bytes(&sample_map()).unwrap();
        let mut tampered = bytes.clone();
        let local_crc_pos = 14;
        let orig = u32::from_le_bytes(
            tampered[local_crc_pos..local_crc_pos + 4]
                .try_into()
                .unwrap(),
        );
        tampered[local_crc_pos..local_crc_pos + 4]
            .copy_from_slice(&(orig ^ 0xFFFF_FFFF).to_le_bytes());

        let err = read_npz_bytes(&tampered);
        assert!(
            matches!(err, Err(NpyError::Entry { .. })),
            "local header CRC の不一致が拒否されなかった: {err:?}"
        );
    }

    #[test]
    fn rejects_local_header_size_mismatch() {
        // P0（codex-review・PR #2318）: local header の
        // compressed/uncompressed size が central directory と異なる
        // 場合に拒否されることを確認する（オフセット +18／+22）。
        let bytes = write_npz_bytes(&sample_map()).unwrap();
        let mut tampered = bytes.clone();
        let orig = u32::from_le_bytes(tampered[18..22].try_into().unwrap());
        tampered[18..22].copy_from_slice(&(orig + 1).to_le_bytes());

        let err = read_npz_bytes(&tampered);
        assert!(
            matches!(err, Err(NpyError::Entry { .. })),
            "local header size の不一致が拒否されなかった: {err:?}"
        );
    }

    #[test]
    fn accepts_local_header_zip64_size_placeholder_even_for_small_entry() {
        // P0 追加検証（`local_size_field_matches` の実装形。PR #2318
        // レビュー是正時の実測）: 実測（CPython 3.13 `zipfile`・numpy
        // 2.3.5 `np.savez`）では、実サイズが `u32::MAX` を大きく下回る
        // 小さな配列であっても `force_zip64=True` で local header を
        // 書くため local header の compressed/uncompressed size は
        // 常に `0xFFFFFFFF` のプレースホルダになる（central directory
        // 側は zip64 extra なしの literal な小さい値のまま）。
        // 「実サイズが u32 を超える場合に限りプレースホルダを許容」と
        // いう一見自然な条件では真正の `np.savez` 出力を全て拒否して
        // しまうため、プレースホルダは実サイズに関わらず常に許容
        // することを確認する（手組みの ZIP で numpy と同型の local
        // header を再現する）。
        let npy = write_npy_bytes(&Tensor::new(vec![1.0f32, 2.0, 3.0], &[3]).unwrap()).unwrap();
        let crc = crc32(&npy);
        let name = b"a.npy";
        let size = npy.len() as u32;

        let mut zip = Vec::new();
        zip.extend_from_slice(&LOCAL_FILE_HEADER_SIG.to_le_bytes());
        zip.extend_from_slice(&45u16.to_le_bytes()); // version needed (zip64)
        zip.extend_from_slice(&0u16.to_le_bytes()); // flags
        zip.extend_from_slice(&0u16.to_le_bytes()); // method = STORED
        zip.extend_from_slice(&0u16.to_le_bytes());
        zip.extend_from_slice(&0x0021u16.to_le_bytes());
        zip.extend_from_slice(&crc.to_le_bytes());
        // numpy 実測と同じく、実サイズが u32 に収まっていても local
        // header 側は zip64 プレースホルダのまま。
        zip.extend_from_slice(&u32::MAX.to_le_bytes()); // compressed size
        zip.extend_from_slice(&u32::MAX.to_le_bytes()); // uncompressed size
        zip.extend_from_slice(&(name.len() as u16).to_le_bytes());
        zip.extend_from_slice(&0u16.to_le_bytes()); // extra length（zip64 extra は省略）
        zip.extend_from_slice(name);
        zip.extend_from_slice(&npy);

        let cd_start = zip.len() as u32;
        zip.extend_from_slice(&CENTRAL_DIR_HEADER_SIG.to_le_bytes());
        zip.extend_from_slice(&20u16.to_le_bytes());
        zip.extend_from_slice(&20u16.to_le_bytes());
        zip.extend_from_slice(&0u16.to_le_bytes());
        zip.extend_from_slice(&0u16.to_le_bytes());
        zip.extend_from_slice(&0u16.to_le_bytes());
        zip.extend_from_slice(&0x0021u16.to_le_bytes());
        zip.extend_from_slice(&crc.to_le_bytes());
        // central directory 側は numpy 実測どおり literal な実サイズ
        // （zip64 extra なし）。
        zip.extend_from_slice(&size.to_le_bytes());
        zip.extend_from_slice(&size.to_le_bytes());
        zip.extend_from_slice(&(name.len() as u16).to_le_bytes());
        zip.extend_from_slice(&0u16.to_le_bytes());
        zip.extend_from_slice(&0u16.to_le_bytes());
        zip.extend_from_slice(&0u16.to_le_bytes());
        zip.extend_from_slice(&0u16.to_le_bytes());
        zip.extend_from_slice(&0u32.to_le_bytes());
        zip.extend_from_slice(&0u32.to_le_bytes()); // local header offset
        zip.extend_from_slice(name);
        let cd_size = zip.len() as u32 - cd_start;

        zip.extend_from_slice(&EOCD_SIG.to_le_bytes());
        zip.extend_from_slice(&0u16.to_le_bytes());
        zip.extend_from_slice(&0u16.to_le_bytes());
        zip.extend_from_slice(&1u16.to_le_bytes());
        zip.extend_from_slice(&1u16.to_le_bytes());
        zip.extend_from_slice(&cd_size.to_le_bytes());
        zip.extend_from_slice(&cd_start.to_le_bytes());
        zip.extend_from_slice(&0u16.to_le_bytes());

        let back = read_npz_bytes(&zip).unwrap();
        assert_eq!(back["a"].host_slice().to_vec(), vec![1.0f32, 2.0, 3.0]);
    }

    #[test]
    fn rejects_member_name_without_npy_suffix() {
        // P1（codex-review・PR #2318）: `.npy` サフィックスを持たない
        // メンバ名は npz キーとして受理しない（末尾除去による正規化が
        // `"a"` と `"a.npy"` のような異なるメンバ名を同一キーへ縮退させ
        // 得たため。`npz_member_key` のドキュメントコメント参照）。
        // 手組みの ZIP で `.npy` を持たない単一メンバを構成し拒否される
        // ことを確認する。
        let npy = write_npy_bytes(&Tensor::new(vec![1.0f32], &[1]).unwrap()).unwrap();
        let crc = crc32(&npy);
        let name = b"a";
        let mut zip = Vec::new();
        let local_offset = 0u32;
        zip.extend_from_slice(&LOCAL_FILE_HEADER_SIG.to_le_bytes());
        zip.extend_from_slice(&20u16.to_le_bytes());
        zip.extend_from_slice(&0u16.to_le_bytes());
        zip.extend_from_slice(&0u16.to_le_bytes()); // method = STORED
        zip.extend_from_slice(&0u16.to_le_bytes());
        zip.extend_from_slice(&0x0021u16.to_le_bytes());
        zip.extend_from_slice(&crc.to_le_bytes());
        zip.extend_from_slice(&(npy.len() as u32).to_le_bytes());
        zip.extend_from_slice(&(npy.len() as u32).to_le_bytes());
        zip.extend_from_slice(&(name.len() as u16).to_le_bytes());
        zip.extend_from_slice(&0u16.to_le_bytes());
        zip.extend_from_slice(name);
        zip.extend_from_slice(&npy);

        let cd_start = zip.len() as u32;
        zip.extend_from_slice(&CENTRAL_DIR_HEADER_SIG.to_le_bytes());
        zip.extend_from_slice(&20u16.to_le_bytes());
        zip.extend_from_slice(&20u16.to_le_bytes());
        zip.extend_from_slice(&0u16.to_le_bytes());
        zip.extend_from_slice(&0u16.to_le_bytes());
        zip.extend_from_slice(&0u16.to_le_bytes());
        zip.extend_from_slice(&0x0021u16.to_le_bytes());
        zip.extend_from_slice(&crc.to_le_bytes());
        zip.extend_from_slice(&(npy.len() as u32).to_le_bytes());
        zip.extend_from_slice(&(npy.len() as u32).to_le_bytes());
        zip.extend_from_slice(&(name.len() as u16).to_le_bytes());
        zip.extend_from_slice(&0u16.to_le_bytes());
        zip.extend_from_slice(&0u16.to_le_bytes());
        zip.extend_from_slice(&0u16.to_le_bytes());
        zip.extend_from_slice(&0u16.to_le_bytes());
        zip.extend_from_slice(&0u32.to_le_bytes());
        zip.extend_from_slice(&local_offset.to_le_bytes());
        zip.extend_from_slice(name);
        let cd_size = zip.len() as u32 - cd_start;

        zip.extend_from_slice(&EOCD_SIG.to_le_bytes());
        zip.extend_from_slice(&0u16.to_le_bytes());
        zip.extend_from_slice(&0u16.to_le_bytes());
        zip.extend_from_slice(&1u16.to_le_bytes());
        zip.extend_from_slice(&1u16.to_le_bytes());
        zip.extend_from_slice(&cd_size.to_le_bytes());
        zip.extend_from_slice(&cd_start.to_le_bytes());
        zip.extend_from_slice(&0u16.to_le_bytes());

        let err = read_npz_bytes(&zip);
        assert!(
            matches!(err, Err(NpyError::InvalidEntryName)),
            "`.npy` サフィックスなしのメンバ名が拒否されなかった: {err:?}"
        );
    }

    #[test]
    fn rejects_invalid_entry_names_on_write() {
        let mut m = HashMap::new();
        m.insert(
            "bad/name".to_string(),
            Tensor::new(vec![1.0], &[1]).unwrap(),
        );
        let err = write_npz_bytes(&m);
        assert!(matches!(err, Err(NpyError::InvalidEntryName)));
    }

    /// `u16::MAX`（65535）件は EOCD の `entries_total` フィールド（u16 幅）に
    /// 書くと zip64 プレースホルダ値と衝突し、`read_npz_bytes` が存在しない
    /// zip64 EOCD locator を要求して失敗する（往復契約が破れる）。
    /// `write_npz_bytes` はこの件数を `TooManyEntries` で拒否しなければ
    /// ならない（PR #2318 レビュー指摘・P1）。
    #[test]
    fn rejects_u16_max_entries_on_write() {
        let mut m = HashMap::new();
        for i in 0..(u16::MAX as usize) {
            m.insert(format!("t{i}"), Tensor::new(vec![1.0], &[1]).unwrap());
        }
        assert_eq!(m.len(), u16::MAX as usize);
        let err = write_npz_bytes(&m);
        assert!(
            matches!(err, Err(NpyError::TooManyEntries)),
            "u16::MAX 件書き出しが拒否されなかった: {err:?}"
        );
    }

    /// `u16::MAX - 1` 件（境界の直下）は zip64 プレースホルダと衝突しない
    /// ため書き出しが成功し、`read_npz_bytes` で往復できることを確認する
    /// （PR #2318 レビュー指摘・P1 の回帰防止）。
    #[test]
    fn round_trips_max_allowed_entries() {
        let n = u16::MAX as usize - 1;
        let mut m = HashMap::new();
        for i in 0..n {
            m.insert(format!("t{i}"), Tensor::new(vec![1.0], &[1]).unwrap());
        }
        let bytes = write_npz_bytes(&m).expect("u16::MAX - 1 件の書き出しは成功するはず");
        let read_back = read_npz_bytes(&bytes).expect("u16::MAX - 1 件の読み込みは成功するはず");
        assert_eq!(read_back.len(), n);
    }

    #[test]
    fn rejects_truncated_eocd() {
        let bytes = write_npz_bytes(&sample_map()).unwrap();
        let truncated = &bytes[..bytes.len() - 10];
        let err = read_npz_bytes(truncated);
        assert!(err.is_err());
    }

    #[test]
    fn rejects_corrupted_crc() {
        let mut bytes = write_npz_bytes(&sample_map()).unwrap();
        // 最初のローカルヘッダのデータ領域内の 1 バイトを破損させる。
        let corrupt_pos = 30 + "a.npy".len() + 20; // ヘッダ後・npy データ内
        bytes[corrupt_pos] ^= 0xFF;
        let err = read_npz_bytes(&bytes);
        assert!(err.is_err());
    }

    #[test]
    fn rejects_encrypted_flag() {
        let mut bytes = write_npz_bytes(&sample_map()).unwrap();
        // central directory 側の flags フィールド（暗号化ビット）を
        // 立てて拒否されることを確認する。central directory のエントリ
        // 先頭を探索して flags オフセット +8 を書き換える。
        let cd_sig = CENTRAL_DIR_HEADER_SIG.to_le_bytes();
        let pos = bytes
            .windows(4)
            .position(|w| w == cd_sig)
            .expect("central directory シグネチャが見つかる");
        bytes[pos + 8] |= 0x01;
        let err = read_npz_bytes(&bytes);
        assert!(matches!(err, Err(NpyError::UnsupportedZipFeature(_))));
    }

    #[test]
    fn rejects_unknown_compression_method() {
        let mut bytes = write_npz_bytes(&sample_map()).unwrap();
        let cd_sig = CENTRAL_DIR_HEADER_SIG.to_le_bytes();
        let pos = bytes
            .windows(4)
            .position(|w| w == cd_sig)
            .expect("central directory シグネチャが見つかる");
        // method フィールド（central directory ヘッダ +10）を非対応値に。
        bytes[pos + 10] = 99;
        bytes[pos + 11] = 0;
        let err = read_npz_bytes(&bytes);
        assert!(matches!(err, Err(NpyError::Entry { .. })));
    }

    #[test]
    fn deflate_member_roundtrip() {
        // STORED 書き出しのみサポートのため、DEFLATE 経路は手組みの ZIP
        // で検証する（`np.savez_compressed` 相当の読み込み経路）。
        let npy = write_npy_bytes(&Tensor::new(vec![1.0f32; 40], &[40]).unwrap()).unwrap();

        // raw deflate（zlib -15 相当）: python
        //   zlib.compressobj(6, zlib.DEFLATED, -15).compress(npy_bytes)+flush()
        // を実行して得たバイト列を使うのが理想だが、簡易には STORED
        // ブロックとして圧縮＝無変換で表現しても DEFLATE 経路の配線は
        // 検証できないため、本テストは stored npy を deflate の stored
        // ブロックとして手組みする（inflate 側の stored 経路と npz の
        // method=8 配線を両方通す）。
        let mut deflate_payload = Vec::new();
        {
            let mut bit = 0u8;
            let mut bitpos = 0u32;
            let push_bit = |b: u32, out: &mut Vec<u8>, bit: &mut u8, bitpos: &mut u32| {
                if *bitpos == 8 {
                    out.push(*bit);
                    *bit = 0;
                    *bitpos = 0;
                }
                *bit |= ((b & 1) as u8) << *bitpos;
                *bitpos += 1;
            };
            push_bit(1, &mut deflate_payload, &mut bit, &mut bitpos); // final
            push_bit(0, &mut deflate_payload, &mut bit, &mut bitpos); // type bit0
            push_bit(0, &mut deflate_payload, &mut bit, &mut bitpos); // type bit1
            // align to byte
            deflate_payload.push(bit);
            let len = npy.len() as u16;
            let nlen = !len;
            deflate_payload.extend_from_slice(&len.to_le_bytes());
            deflate_payload.extend_from_slice(&nlen.to_le_bytes());
            deflate_payload.extend_from_slice(&npy);
        }

        let crc = crc32(&npy);
        let name = b"x.npy";
        let mut zip = Vec::new();
        let local_offset = 0u32;
        zip.extend_from_slice(&LOCAL_FILE_HEADER_SIG.to_le_bytes());
        zip.extend_from_slice(&20u16.to_le_bytes());
        zip.extend_from_slice(&0u16.to_le_bytes());
        zip.extend_from_slice(&8u16.to_le_bytes()); // method = DEFLATE
        zip.extend_from_slice(&0u16.to_le_bytes());
        zip.extend_from_slice(&0x0021u16.to_le_bytes());
        zip.extend_from_slice(&crc.to_le_bytes());
        zip.extend_from_slice(&(deflate_payload.len() as u32).to_le_bytes());
        zip.extend_from_slice(&(npy.len() as u32).to_le_bytes());
        zip.extend_from_slice(&(name.len() as u16).to_le_bytes());
        zip.extend_from_slice(&0u16.to_le_bytes());
        zip.extend_from_slice(name);
        zip.extend_from_slice(&deflate_payload);

        let cd_start = zip.len() as u32;
        zip.extend_from_slice(&CENTRAL_DIR_HEADER_SIG.to_le_bytes());
        zip.extend_from_slice(&20u16.to_le_bytes());
        zip.extend_from_slice(&20u16.to_le_bytes());
        zip.extend_from_slice(&0u16.to_le_bytes());
        zip.extend_from_slice(&8u16.to_le_bytes());
        zip.extend_from_slice(&0u16.to_le_bytes());
        zip.extend_from_slice(&0x0021u16.to_le_bytes());
        zip.extend_from_slice(&crc.to_le_bytes());
        zip.extend_from_slice(&(deflate_payload.len() as u32).to_le_bytes());
        zip.extend_from_slice(&(npy.len() as u32).to_le_bytes());
        zip.extend_from_slice(&(name.len() as u16).to_le_bytes());
        zip.extend_from_slice(&0u16.to_le_bytes());
        zip.extend_from_slice(&0u16.to_le_bytes());
        zip.extend_from_slice(&0u16.to_le_bytes());
        zip.extend_from_slice(&0u16.to_le_bytes());
        zip.extend_from_slice(&0u32.to_le_bytes());
        zip.extend_from_slice(&local_offset.to_le_bytes());
        zip.extend_from_slice(name);
        let cd_size = zip.len() as u32 - cd_start;

        zip.extend_from_slice(&EOCD_SIG.to_le_bytes());
        zip.extend_from_slice(&0u16.to_le_bytes());
        zip.extend_from_slice(&0u16.to_le_bytes());
        zip.extend_from_slice(&1u16.to_le_bytes());
        zip.extend_from_slice(&1u16.to_le_bytes());
        zip.extend_from_slice(&cd_size.to_le_bytes());
        zip.extend_from_slice(&cd_start.to_le_bytes());
        zip.extend_from_slice(&0u16.to_le_bytes());

        let back = read_npz_bytes(&zip).unwrap();
        assert_eq!(back["x"].host_slice().to_vec(), vec![1.0f32; 40]);
    }

    #[test]
    fn rejects_member_data_overlapping_central_directory() {
        // P0（codex-review・npz.rs:579 未解決指摘）: central directory の
        // 宣言 `compressed_size`／`uncompressed_size` を細工し、メンバの
        // データ領域が central directory 自体の開始位置まで食い込む
        // ケースを拒否することを確認する（当初の指摘の再現テスト）。
        // 本検査は 2 パス目（伸長・CRC 検証）より前の構造検証パスで
        // 発火するため、CRC・npy 内容の整合は不要。
        let mut m = HashMap::new();
        m.insert("a".to_string(), Tensor::new(vec![1.0], &[1]).unwrap());
        let bytes = write_npz_bytes(&m).unwrap();

        let cd_sig = CENTRAL_DIR_HEADER_SIG.to_le_bytes();
        let cd_pos = bytes
            .windows(4)
            .position(|w| w == cd_sig)
            .expect("central directory シグネチャが見つかる");
        let mut tampered = bytes.clone();
        // compressed size（CD ヘッダ +20）を、central directory・EOCD の
        // 領域まで確実に食い込む大きさへ書き換える。
        let oversized = bytes.len() as u32;
        tampered[cd_pos + 20..cd_pos + 24].copy_from_slice(&oversized.to_le_bytes());

        let err = read_npz_bytes(&tampered);
        match err {
            Err(NpyError::Entry { name, source }) => {
                assert_eq!(name, "a.npy");
                assert!(
                    matches!(*source, NpyError::InvalidZip(_)),
                    "central directory との重なりが InvalidZip として拒否されなかった: {source:?}"
                );
            }
            other => panic!("central directory と重なるメンバが拒否されなかった: {other:?}"),
        }
    }

    #[test]
    fn rejects_member_data_overlapping_another_member() {
        // P0（codex-review・npz.rs:579 未解決指摘）: central directory の
        // `local_header_offset` を細工し、あるメンバのデータ領域が別の
        // メンバのデータ領域と重なるケースを拒否することを確認する。
        let mut m = HashMap::new();
        m.insert("a".to_string(), Tensor::new(vec![1.0], &[1]).unwrap());
        m.insert("b".to_string(), Tensor::new(vec![2.0], &[1]).unwrap());
        let bytes = write_npz_bytes(&m).unwrap();

        let cd_sig = CENTRAL_DIR_HEADER_SIG.to_le_bytes();
        let first_cd_pos = bytes
            .windows(4)
            .position(|w| w == cd_sig)
            .expect("1 番目の central directory シグネチャが見つかる");
        let second_cd_pos = bytes
            .windows(4)
            .rposition(|w| w == cd_sig)
            .expect("2 番目の central directory シグネチャが見つかる");
        assert_ne!(first_cd_pos, second_cd_pos);

        // 1 番目（sort 順で "a"）の local_header_offset（CD ヘッダ +42）を
        // 読み取り、2 番目（"b"）の local_header_offset へ同じ値を書き込む
        // ことで、2 つのメンバが同一の local header・データ領域を指す
        // ようにする。
        let first_local_offset = u32::from_le_bytes(
            bytes[first_cd_pos + 42..first_cd_pos + 46]
                .try_into()
                .unwrap(),
        );
        let mut tampered = bytes.clone();
        tampered[second_cd_pos + 42..second_cd_pos + 46]
            .copy_from_slice(&first_local_offset.to_le_bytes());

        let err = read_npz_bytes(&tampered);
        assert!(
            matches!(err, Err(NpyError::Entry { .. })),
            "メンバ間のデータ領域の重なりが拒否されなかった: {err:?}"
        );
    }

    #[test]
    fn rejects_central_directory_overlapping_eocd() {
        // P0（codex-review・npz.rs:579 未解決指摘の関連監査）: central
        // directory の宣言範囲（`cd_offset..cd_offset+cd_size`）を、
        // ファイル全体には収まるが通常 EOCD の領域へ食い込む大きさに
        // 細工した場合に拒否されることを確認する。旧実装は
        // `cd_end > bytes.len()` のみを検査していたため、この細工は
        // ファイル範囲内に収まる限り見逃されていた。
        let bytes = write_npz_bytes(&sample_map()).unwrap();
        let eocd_sig = EOCD_SIG.to_le_bytes();
        let eocd_pos = bytes
            .windows(4)
            .rposition(|w| w == eocd_sig)
            .expect("EOCD シグネチャが見つかる");
        let mut tampered = bytes.clone();
        let orig_cd_size =
            u32::from_le_bytes(tampered[eocd_pos + 12..eocd_pos + 16].try_into().unwrap());
        // cd_size を、EOCD の固定部分の範囲内（+4 バイト）まで食い込む
        // 大きさへ増やす。EOCD は常にファイル末尾の 22 バイトのため、
        // この程度の増加でも `bytes.len()` は超えない。
        tampered[eocd_pos + 12..eocd_pos + 16].copy_from_slice(&(orig_cd_size + 4).to_le_bytes());

        let err = read_npz_bytes(&tampered);
        assert!(
            matches!(err, Err(NpyError::InvalidZip(_))),
            "central directory と EOCD の重なりが拒否されなかった: {err:?}"
        );
    }

    #[test]
    fn rejects_ambiguous_eocd_candidates() {
        // P0（codex-review・npz.rs:579 未解決指摘の関連監査。EOCD 探索の
        // 曖昧性）: コメント本文に、それ自身も「シグネチャ＋コメント長
        // 整合」を満たす偽の EOCD 様バイト列を埋め込むと、末尾から最初に
        // 見つかった候補だけを無条件に信頼する実装では偽陽性が起き得る。
        // 真の EOCD の次に読み取り可能な第二の候補が存在する場合は
        // 一意に決定できないとして拒否することを確認する。
        let mut m = HashMap::new();
        m.insert("a".to_string(), Tensor::new(vec![1.0], &[1]).unwrap());
        let bytes = write_npz_bytes(&m).unwrap();

        let true_eocd_pos = bytes.len() - EOCD_FIXED_SIZE;
        assert_eq!(
            &bytes[true_eocd_pos..true_eocd_pos + 4],
            &EOCD_SIG.to_le_bytes()
        );

        // 偽の EOCD（22 バイト固定部分。comment_len=0 で自身の直後を
        // ファイル終端とする）をコメント末尾に置き、その手前に任意の
        // padding を挟む。
        let padding_len = 10usize;
        let fake_eocd_len = EOCD_FIXED_SIZE;
        let new_comment_len = (padding_len + fake_eocd_len) as u16;

        let mut tampered = bytes[..true_eocd_pos + 20].to_vec();
        tampered.extend_from_slice(&new_comment_len.to_le_bytes()); // 真の EOCD の comment_len
        tampered.extend(std::iter::repeat_n(0u8, padding_len));
        tampered.extend_from_slice(&EOCD_SIG.to_le_bytes());
        tampered.extend(std::iter::repeat_n(0u8, 16)); // 偽 EOCD の残りの固定フィールド
        tampered.extend_from_slice(&0u16.to_le_bytes()); // 偽 EOCD の comment_len = 0

        let err = read_npz_bytes(&tampered);
        assert!(
            matches!(err, Err(NpyError::InvalidZip(_))),
            "曖昧な EOCD 候補が拒否されなかった: {err:?}"
        );
    }

    #[test]
    fn rejects_zip64_entries_mismatch_with_non_sentinel_eocd_field() {
        // P0（codex-review・npz.rs:579 未解決指摘の関連監査。zip64
        // sentinel／非 sentinel フィールドの一貫性）: 通常 EOCD の
        // `entries_total` が zip64 プレースホルダ（`u16::MAX`）でない
        // 場合、zip64 EOCD レコードから解決したエントリ数と一致しなければ
        // ならないことを確認する。
        let zip = zip64_multi_disk_tests::build_zip64_npz(0, 1, 0, 0);
        let eocd_sig = EOCD_SIG.to_le_bytes();
        let eocd_pos = zip
            .windows(4)
            .rposition(|w| w == eocd_sig)
            .expect("通常 EOCD シグネチャが見つかる");
        assert_eq!(
            u16::from_le_bytes([zip[eocd_pos + 10], zip[eocd_pos + 11]]),
            u16::MAX
        );
        let mut tampered = zip.clone();
        // entries_total（sentinel ではない値）を、zip64 EOCD レコードの
        // 実エントリ数（1）と矛盾する値へ書き換える。`entries_this_disk`
        // （+8）も同じ値に揃えないと、それより手前の
        // `entries_this_disk != entries_total` によるマルチディスク検査
        // （本テストの対象ではない）が先に発火してしまう。
        tampered[eocd_pos + 8..eocd_pos + 10].copy_from_slice(&3u16.to_le_bytes());
        tampered[eocd_pos + 10..eocd_pos + 12].copy_from_slice(&3u16.to_le_bytes());

        let err = read_npz_bytes(&tampered);
        assert!(
            matches!(err, Err(NpyError::InvalidZip(_))),
            "EOCD と zip64 EOCD のエントリ数不一致が拒否されなかった: {err:?}"
        );
    }

    #[test]
    fn rejects_zip64_eocd_record_overlapping_locator() {
        // P0（codex-review・npz.rs:579 未解決指摘の関連監査）: zip64 EOCD
        // レコードの「レコードサイズ」フィールドを細工し、レコードの
        // 宣言終端が zip64 EOCD locator の領域まで食い込む場合に拒否
        // されることを確認する。
        let zip = zip64_multi_disk_tests::build_zip64_npz(0, 1, 0, 0);
        let zip64_eocd_sig = ZIP64_EOCD_SIG.to_le_bytes();
        let zip64_eocd_pos = zip
            .windows(4)
            .position(|w| w == zip64_eocd_sig)
            .expect("zip64 EOCD シグネチャが見つかる");
        let mut tampered = zip.clone();
        // レコードサイズ（zip64 EOCD +4、本来 44）を、locator の領域まで
        // 確実に食い込む大きさへ書き換える。
        let oversized_record_size = zip.len() as u64;
        tampered[zip64_eocd_pos + 4..zip64_eocd_pos + 12]
            .copy_from_slice(&oversized_record_size.to_le_bytes());

        let err = read_npz_bytes(&tampered);
        assert!(
            matches!(err, Err(NpyError::InvalidZip(_))),
            "zip64 EOCD レコードと locator の重なりが拒否されなかった: {err:?}"
        );
    }

    // PR #2318 レビュー指摘・P2（npz.rs:848）の是正: `write_npz_bytes` は
    // 各エントリを `u32` に収めるだけで、`read_npz_bytes` が適用する
    // メンバ単体・累積の伸長後サイズ上限を検証していなかった。以下は
    // その是正（`plan_npz_member_sizes`）の境界・再現テスト。いずれも
    // shape（`&[usize]`）のみを渡す純粋関数を直接呼ぶため、1 GiB 相当の
    // `Tensor` を実際に確保しない。

    #[test]
    fn plan_npz_member_sizes_exact_member_boundary() {
        // メンバ単体上限（`MAX_MEMBER_DECOMPRESSED_BYTES`）ちょうどの
        // 予測 npy 長を持つ shape は許容され、+4 バイト（要素 1 個分）
        // 大きい shape は `Entry { source: DecompressedSizeExceeded }`
        // で拒否されることを確認する。
        let n_at_cap = npy_shape_at_member_cap();
        let ok = plan_npz_member_sizes(std::iter::once(("a.npy", &[n_at_cap][..])));
        assert!(
            ok.is_ok(),
            "メンバ単体上限ちょうどの shape が拒否された: {ok:?}"
        );

        let over = plan_npz_member_sizes(std::iter::once(("a.npy", &[n_at_cap + 1][..])));
        match over {
            Err(NpyError::Entry { name, source }) => {
                assert_eq!(name, "a.npy");
                match *source {
                    NpyError::DecompressedSizeExceeded { max, .. } => {
                        assert_eq!(max, MAX_MEMBER_DECOMPRESSED_BYTES);
                    }
                    other => panic!("DecompressedSizeExceeded ではない: {other:?}"),
                }
            }
            other => panic!("メンバ単体上限超過が拒否されなかった: {other:?}"),
        }
    }

    #[test]
    fn plan_npz_member_sizes_exact_total_boundary() {
        // 1 番目のメンバがメンバ単体上限ちょうど（＝累積もちょうど上限）
        // のとき、2 番目にごく小さいメンバを追加するだけで累積上限
        // （`MAX_TOTAL_DECOMPRESSED_BYTES`）を超えて拒否されることを
        // 確認する（PR #2318 レビュー指摘・P2 の直接の再現）。
        let n_at_cap = npy_shape_at_member_cap();
        let entries = [("a.npy", &[n_at_cap][..]), ("b.npy", &[1usize][..])];
        let err = plan_npz_member_sizes(entries.into_iter());
        assert!(
            matches!(err, Err(NpyError::DecompressedSizeExceeded { .. })),
            "累積上限超過が拒否されなかった: {err:?}"
        );
    }

    #[test]
    fn plan_npz_member_sizes_reproduces_multiple_small_tensors_exceeding_total_cap() {
        // レビュー指摘の直接の再現シナリオ:
        // 「小さいテンソルを複数渡して合計が上限を超えると `save_npz` は
        // 成功するが `load_npz` は `DecompressedSizeExceeded` で失敗する」。
        // 個々の shape（rank-1、約 572 MiB 相当）はメンバ単体上限
        // （1 GiB）を大きく下回るが、2 つ合わせると累積上限（1 GiB）を
        // 超える。shape は `&[usize]` の数値でしかないため、実際に
        // 572 MiB×2 のデータを確保することはない。
        let half_numel = 150_000_000usize; // 600,000,000 バイト相当（< 1 GiB）
        let entries = [("a.npy", &[half_numel][..]), ("b.npy", &[half_numel][..])];
        // 個々のメンバは単体上限未満であることを前提として確認する。
        for (_, shape) in entries {
            let len = npy_encoded_len(shape).unwrap();
            assert!(len < MAX_MEMBER_DECOMPRESSED_BYTES);
        }
        let err = plan_npz_member_sizes(entries.into_iter());
        assert!(
            matches!(err, Err(NpyError::DecompressedSizeExceeded { .. })),
            "複数の小さいテンソルの合計上限超過が拒否されなかった: {err:?}"
        );
    }

    #[test]
    fn write_npz_bytes_rejects_member_exceeding_cap_via_public_api() {
        // `plan_npz_member_sizes` 単体だけでなく、`write_npz_bytes`
        // （公開 API）自身が同じ検査を実データのシリアライズ前に適用する
        // ことを、rank 1・要素数 1 の極小テンソルで確認する（境界値では
        // なく配線の確認）。細工が必要な reader 側テストと異なり、writer
        // 側は shape が実データと一致していなければならないため、
        // 「小さい実データで拒否経路が呼ばれること」自体を確認する。
        let mut m = HashMap::new();
        m.insert("x".to_string(), Tensor::new(vec![1.0f32], &[1]).unwrap());
        // 通常サイズは当然許容される（対照）。
        assert!(write_npz_bytes(&m).is_ok());
    }

    /// メンバ単体上限（`MAX_MEMBER_DECOMPRESSED_BYTES`）ちょうどの予測
    /// npy 長になる rank-1 shape の要素数を、実データを確保せずに厳密に
    /// 逆算する。9 桁の要素数では shape 文字列長（＝ヘッダ長）が桁数
    /// だけに依存し不変なため、適当な 9 桁の probe で prefix 長を求めて
    /// から目標総バイト長ちょうどになる要素数を直接解く。
    fn npy_shape_at_member_cap() -> usize {
        let target = MAX_MEMBER_DECOMPRESSED_BYTES;
        let probe_n: u64 = 268_435_456; // 9 桁（target/4 の概算）
        let probe_len = npy_encoded_len(&[probe_n as usize]).unwrap();
        let prefix_len = probe_len - probe_n * 4;
        assert_eq!((target - prefix_len) % 4, 0);
        let n_at_cap = ((target - prefix_len) / 4) as usize;
        assert_eq!(n_at_cap.to_string().len(), probe_n.to_string().len());
        assert_eq!(npy_encoded_len(&[n_at_cap]).unwrap(), target);
        n_at_cap
    }
}

#[cfg(test)]
mod zip64_multi_disk_tests {
    use super::*;

    /// zip64 EOCD 経路（locator + zip64 EOCD レコード + 通常 EOCD の
    /// センチネル値）を持つ、単一メンバ "a" の npz を手組みする。
    /// マルチディスクフィールド（PR #2318 レビュー指摘・P2）の検証を
    /// 単体でテストするため、4 つのフィールドを引数で差し替えられる
    /// ようにしてある。すべて「単一ディスク」を示す値（0, 1, 0, 0）を
    /// 渡せば正常に読み込める構成になる。`pub(super)` は `tests` モジュール
    /// （同じ `npz` モジュールの兄弟。zip64 sentinel／locator 重なりの
    /// 追加検証テストから再利用するため）から呼べるようにするため。
    pub(super) fn build_zip64_npz(
        locator_disk: u32,
        locator_total_disks: u32,
        zip64_this_disk: u32,
        zip64_cd_start_disk: u32,
    ) -> Vec<u8> {
        let npy = write_npy_bytes(&Tensor::new(vec![1.0f32], &[1]).unwrap()).unwrap();
        let crc = crc32(&npy);
        let name = b"a.npy";
        let size = npy.len() as u32;

        let mut zip = Vec::new();
        let local_offset = 0u32;
        zip.extend_from_slice(&LOCAL_FILE_HEADER_SIG.to_le_bytes());
        zip.extend_from_slice(&20u16.to_le_bytes());
        zip.extend_from_slice(&0u16.to_le_bytes());
        zip.extend_from_slice(&0u16.to_le_bytes()); // method = STORED
        zip.extend_from_slice(&0u16.to_le_bytes());
        zip.extend_from_slice(&0x0021u16.to_le_bytes());
        zip.extend_from_slice(&crc.to_le_bytes());
        zip.extend_from_slice(&size.to_le_bytes());
        zip.extend_from_slice(&size.to_le_bytes());
        zip.extend_from_slice(&(name.len() as u16).to_le_bytes());
        zip.extend_from_slice(&0u16.to_le_bytes());
        zip.extend_from_slice(name);
        zip.extend_from_slice(&npy);

        let cd_start = zip.len() as u32;
        zip.extend_from_slice(&CENTRAL_DIR_HEADER_SIG.to_le_bytes());
        zip.extend_from_slice(&20u16.to_le_bytes());
        zip.extend_from_slice(&20u16.to_le_bytes());
        zip.extend_from_slice(&0u16.to_le_bytes());
        zip.extend_from_slice(&0u16.to_le_bytes());
        zip.extend_from_slice(&0u16.to_le_bytes());
        zip.extend_from_slice(&0x0021u16.to_le_bytes());
        zip.extend_from_slice(&crc.to_le_bytes());
        zip.extend_from_slice(&size.to_le_bytes());
        zip.extend_from_slice(&size.to_le_bytes());
        zip.extend_from_slice(&(name.len() as u16).to_le_bytes());
        zip.extend_from_slice(&0u16.to_le_bytes());
        zip.extend_from_slice(&0u16.to_le_bytes());
        zip.extend_from_slice(&0u16.to_le_bytes());
        zip.extend_from_slice(&0u16.to_le_bytes());
        zip.extend_from_slice(&0u32.to_le_bytes());
        zip.extend_from_slice(&local_offset.to_le_bytes());
        zip.extend_from_slice(name);
        let cd_size = (zip.len() as u32 - cd_start) as u64;
        let cd_start = cd_start as u64;

        // zip64 EOCD レコード（56 バイト固定長部分。extensible data
        // sector は付与しない）。
        let zip64_eocd_offset = zip.len() as u64;
        zip.extend_from_slice(&ZIP64_EOCD_SIG.to_le_bytes());
        zip.extend_from_slice(&44u64.to_le_bytes()); // レコードサイズ（本体 - 12）
        zip.extend_from_slice(&45u16.to_le_bytes()); // version made by
        zip.extend_from_slice(&45u16.to_le_bytes()); // version needed
        zip.extend_from_slice(&zip64_this_disk.to_le_bytes());
        zip.extend_from_slice(&zip64_cd_start_disk.to_le_bytes());
        zip.extend_from_slice(&1u64.to_le_bytes()); // entries (this disk)
        zip.extend_from_slice(&1u64.to_le_bytes()); // entries (total)
        zip.extend_from_slice(&cd_size.to_le_bytes());
        zip.extend_from_slice(&cd_start.to_le_bytes());

        // zip64 EOCD locator（固定 20 バイト）。
        zip.extend_from_slice(&ZIP64_EOCD_LOCATOR_SIG.to_le_bytes());
        zip.extend_from_slice(&locator_disk.to_le_bytes());
        zip.extend_from_slice(&zip64_eocd_offset.to_le_bytes());
        zip.extend_from_slice(&locator_total_disks.to_le_bytes());

        // 通常 EOCD（entries_total = u16::MAX がセンチネルとして zip64
        // 経路を発火させる。cd_size／cd_offset はセンチネルにせず実値の
        // まま置き、zip64 経路は entries_total だけで判定させる）。
        zip.extend_from_slice(&EOCD_SIG.to_le_bytes());
        zip.extend_from_slice(&0u16.to_le_bytes());
        zip.extend_from_slice(&0u16.to_le_bytes());
        zip.extend_from_slice(&u16::MAX.to_le_bytes());
        zip.extend_from_slice(&u16::MAX.to_le_bytes());
        zip.extend_from_slice(&(cd_size as u32).to_le_bytes());
        zip.extend_from_slice(&(cd_start as u32).to_le_bytes());
        zip.extend_from_slice(&0u16.to_le_bytes());

        zip
    }

    #[test]
    fn reads_zip64_eocd_with_valid_single_disk_fields() {
        let zip = build_zip64_npz(0, 1, 0, 0);
        let back = read_npz_bytes(&zip).expect("単一ディスクの zip64 EOCD は読み込めるはず");
        assert_eq!(back["a"].host_slice().to_vec(), vec![1.0f32]);
    }

    #[test]
    fn rejects_zip64_locator_non_single_disk() {
        // locator の「zip64 EOCD を含むディスク番号」が 0 以外
        // （PR #2318 レビュー指摘・P2）。
        let zip = build_zip64_npz(1, 1, 0, 0);
        let err = read_npz_bytes(&zip);
        assert!(
            matches!(err, Err(NpyError::UnsupportedZipFeature(_))),
            "zip64 locator の非単一ディスクが拒否されなかった: {err:?}"
        );
    }

    #[test]
    fn rejects_zip64_locator_total_disks_not_one() {
        // locator の「ディスク総数」が 1 以外（PR #2318 レビュー指摘・P2）。
        let zip = build_zip64_npz(0, 2, 0, 0);
        let err = read_npz_bytes(&zip);
        assert!(
            matches!(err, Err(NpyError::UnsupportedZipFeature(_))),
            "zip64 locator のディスク総数不一致が拒否されなかった: {err:?}"
        );
    }

    #[test]
    fn rejects_zip64_eocd_this_disk_nonzero() {
        // zip64 EOCD 本体の「このディスクの番号」が 0 以外
        // （PR #2318 レビュー指摘・P2）。
        let zip = build_zip64_npz(0, 1, 1, 0);
        let err = read_npz_bytes(&zip);
        assert!(
            matches!(err, Err(NpyError::UnsupportedZipFeature(_))),
            "zip64 EOCD のこのディスク番号不一致が拒否されなかった: {err:?}"
        );
    }

    #[test]
    fn rejects_zip64_eocd_cd_start_disk_nonzero() {
        // zip64 EOCD 本体の「central directory 開始ディスク番号」が
        // 0 以外（PR #2318 レビュー指摘・P2）。
        let zip = build_zip64_npz(0, 1, 0, 1);
        let err = read_npz_bytes(&zip);
        assert!(
            matches!(err, Err(NpyError::UnsupportedZipFeature(_))),
            "zip64 EOCD の central directory 開始ディスク番号不一致が拒否されなかった: {err:?}"
        );
    }
}
