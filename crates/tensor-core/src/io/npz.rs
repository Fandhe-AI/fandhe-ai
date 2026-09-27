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

use std::collections::HashMap;
use std::path::Path;

use super::NpyError;
use super::bounded::{read_u16_le, read_u32_le, read_u64_le, slice_at};
use super::crc32::crc32;
use super::inflate::inflate;
use super::npy::{read_npy_bytes, write_npy_bytes};
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

/// バイト列（`.npz` ファイルの内容そのもの）から名前付きテンソル集合を
/// 読み取る。1 メンバでも失敗すれば全体を失敗させ、部分的な
/// `HashMap` は返さない（fail-closed。`.claude/rules/security.md` A08）。
pub fn read_npz_bytes(bytes: &[u8]) -> Result<HashMap<String, Tensor<f32>>, NpyError> {
    let eocd = find_eocd(bytes)?;
    let (entry_count, cd_offset, cd_size) = resolve_eocd_counts(bytes, &eocd)?;
    if entry_count > MAX_ENTRIES {
        return Err(NpyError::InvalidZip("エントリ数が上限を超える"));
    }
    let cd_end = cd_offset
        .checked_add(cd_size)
        .ok_or(NpyError::InvalidZip("central directory の範囲が不正"))?;
    if cd_end > bytes.len() {
        return Err(NpyError::InvalidZip(
            "central directory がファイル範囲外を指す",
        ));
    }

    let mut result = HashMap::with_capacity(entry_count);
    let mut pos = cd_offset;
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
        if entry.is_directory {
            // ディレクトリエントリも central directory／local header の
            // 構造的整合性（名前・flags・method・CRC・サイズの一致）は
            // 検証する。名前だけを見て無条件に読み飛ばすと、データを
            // 持つエントリを「ディレクトリ名」（末尾 `/` 等）に偽装した
            // 細工 ZIP が、local header との不一致を一切検査されずに
            // 受理されてしまう（PR #2318 レビュー指摘・P2）。ディレクトリ
            // は本来データを持たないため、検証後の内容が非空であれば
            // 拒否する。
            let decompressed = read_member_bytes(bytes, &entry).map_err(|e| NpyError::Entry {
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
        let tensor = read_member(bytes, &entry).map_err(|e| NpyError::Entry {
            name: key.clone(),
            source: Box::new(e),
        })?;
        result.insert(key, tensor);
    }
    // 走査終了位置が central directory の宣言終端と厳密に一致すること
    // を確認する（entry_count だけを信頼せず、宣言範囲全体が実際の
    // エントリ列で過不足なく埋まっていることを検証する）。
    if pos != cd_end {
        return Err(NpyError::InvalidZip(
            "central directory の走査終了位置が宣言範囲と不一致",
        ));
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
fn find_eocd(bytes: &[u8]) -> Result<EocdInfo, NpyError> {
    if bytes.len() < EOCD_FIXED_SIZE {
        return Err(NpyError::InvalidZip("ファイルが短すぎて EOCD が存在しない"));
    }
    let search_start = bytes
        .len()
        .saturating_sub(EOCD_SEARCH_WINDOW.min(bytes.len()));
    let mut i = bytes.len() - EOCD_FIXED_SIZE;
    loop {
        if read_u32_le(bytes, i, "EOCD 探索中の範囲外アクセス").ok() == Some(EOCD_SIG) {
            // コメント長フィールドとファイル終端の整合を検査し、偽陽性
            // （データ本体にたまたま同じ 4 バイトが出現する場合）を除く。
            let comment_len = read_u16_le(bytes, i + 20, "EOCD comment 長").unwrap_or(u16::MAX);
            if i + EOCD_FIXED_SIZE + comment_len as usize == bytes.len() {
                return Ok(EocdInfo { offset: i });
            }
        }
        if i == search_start {
            break;
        }
        i -= 1;
    }
    Err(NpyError::InvalidZip(
        "EOCD（end of central directory）が見つからない",
    ))
}

/// EOCD（および必要なら zip64 EOCD）からエントリ数・central directory の
/// オフセット・サイズを解決する。
fn resolve_eocd_counts(bytes: &[u8], eocd: &EocdInfo) -> Result<(usize, usize, usize), NpyError> {
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
        return Ok((
            entries_total as usize,
            cd_offset_32 as usize,
            cd_size_32 as usize,
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
    let entries = usize::try_from(entries)
        .map_err(|_| NpyError::InvalidZip("zip64 エントリ数が usize 範囲を超える"))?;
    let cd_size = usize::try_from(cd_size)
        .map_err(|_| NpyError::InvalidZip("zip64 cd size が usize 範囲を超える"))?;
    let cd_offset = usize::try_from(cd_offset)
        .map_err(|_| NpyError::InvalidZip("zip64 cd offset が usize 範囲を超える"))?;
    Ok((entries, cd_offset, cd_size))
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

    let mut out = Vec::new();
    // (name, crc, compressed_size, local_header_offset, flags)
    let mut central_records: Vec<(String, u32, u32, u32, u16)> = Vec::with_capacity(keys.len());

    for key in &keys {
        validate_entry_name(key)?;
        let entry_name = format!("{key}.npy");
        let npy_bytes = write_npy_bytes(&map[*key])?;
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

        central_records.push((entry_name, crc, size, local_header_offset, flags));
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
pub fn save_npz<P: AsRef<Path>>(
    map: &HashMap<String, Tensor<f32>>,
    path: P,
) -> Result<(), NpyError> {
    let bytes = write_npz_bytes(map)?;
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
}

#[cfg(test)]
mod zip64_multi_disk_tests {
    use super::*;

    /// zip64 EOCD 経路（locator + zip64 EOCD レコード + 通常 EOCD の
    /// センチネル値）を持つ、単一メンバ "a" の npz を手組みする。
    /// マルチディスクフィールド（PR #2318 レビュー指摘・P2）の検証を
    /// 単体でテストするため、4 つのフィールドを引数で差し替えられる
    /// ようにしてある。すべて「単一ディスク」を示す値（0, 1, 0, 0）を
    /// 渡せば正常に読み込める構成になる。
    fn build_zip64_npz(
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
