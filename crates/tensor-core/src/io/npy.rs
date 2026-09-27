//! NumPy `.npy` 形式（単一配列）の読み書き（イシュー #2189）。
//!
//! `io`（親モジュール）の doc を参照。本ファイルは npy のヘッダ
//! （magic・バージョン・`descr`／`fortran_order`／`shape` 辞書）の
//! 解析・生成と、`Tensor<f32>` との相互変換を担う。ヘッダ辞書は
//! 汎用 Python リテラルパーサではなく、受理する形（3 キー固定・
//! 決まった値の型）だけを扱う**専用の最小パーサ**で解析する
//! （`.claude/rules/security.md` A03。任意コード評価に相当する経路を
//! 作らない）。

use std::path::Path;

use super::NpyError;
use crate::tensor::Tensor;

/// npy magic（先頭 6 バイト）。
const MAGIC: [u8; 6] = [0x93, b'N', b'U', b'M', b'P', b'Y'];

/// NumPy `_MAX_HEADER_SIZE` 相当のヘッダ長上限（バイト）。ヘッダ長
/// フィールドを偽装した過大確保（`.claude/rules/security.md` A04/A05）を
/// 防ぐため、スライス取得前に検査する。
const MAX_HEADER_SIZE: usize = 10000;

/// npy ヘッダの `shape` タプルが取りうる rank の上限。異常に長い
/// タプル文字列によるパース処理の肥大化を防ぐ。
const MAX_RANK: usize = 64;

/// npy ヘッダ辞書のパース結果。
struct Header {
    descr: String,
    fortran_order: bool,
    shape: Vec<usize>,
}

/// バイト列（`.npy` ファイルの内容そのもの）から `Tensor<f32>` を
/// 読み取る。
///
/// 手順（`docs/tensor-core-npy-npz-io-decision.md` §3.3 準拠）:
/// 1. magic・バージョンを検証する
/// 2. ヘッダ長フィールド（v1: u16、v2/v3: u32）を読み、上限・範囲を
///    検査してからヘッダ本体を取り出す
/// 3. ヘッダ辞書を専用パーサで解析する
/// 4. dtype（`<f4`／`>f4` のみ対応）に応じてバイト列を `f32` へ変換する
/// 5. `fortran_order` なら逆順 shape で構築後 `permute` + `contiguous`
///    で C 順に変換する
pub fn read_npy_bytes(bytes: &[u8]) -> Result<Tensor<f32>, NpyError> {
    if bytes.len() < 8 || bytes[0..6] != MAGIC {
        return Err(NpyError::InvalidMagic);
    }
    let major = bytes[6];
    let minor = bytes[7];
    let (header_len_field_size, header_len) = match (major, minor) {
        (1, 0) => {
            let raw = bytes.get(8..10).ok_or(NpyError::HeaderTooLarge {
                len: 0,
                max: MAX_HEADER_SIZE,
            })?;
            (2usize, u16::from_le_bytes([raw[0], raw[1]]) as usize)
        }
        (2, 0) | (3, 0) => {
            let raw = bytes.get(8..12).ok_or(NpyError::HeaderTooLarge {
                len: 0,
                max: MAX_HEADER_SIZE,
            })?;
            (
                4usize,
                u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]) as usize,
            )
        }
        _ => return Err(NpyError::UnsupportedVersion { major, minor }),
    };
    if header_len > MAX_HEADER_SIZE {
        return Err(NpyError::HeaderTooLarge {
            len: header_len,
            max: MAX_HEADER_SIZE,
        });
    }
    let header_start = 8 + header_len_field_size;
    let header_end = header_start
        .checked_add(header_len)
        .ok_or(NpyError::HeaderTooLarge {
            len: header_len,
            max: MAX_HEADER_SIZE,
        })?;
    let header_bytes = bytes
        .get(header_start..header_end)
        .ok_or(NpyError::HeaderTooLarge {
            len: header_len,
            max: MAX_HEADER_SIZE,
        })?;
    // v1/v2 は ASCII（latin1 のうち非 ASCII は本パーサの受理形に現れない
    // ため実質 ASCII）、v3 は UTF-8 として検証する。
    let header_str = std::str::from_utf8(header_bytes)
        .map_err(|_| NpyError::InvalidHeader("ヘッダが有効な UTF-8/ASCII ではない"))?;
    if major == 1 && !header_str.is_ascii() {
        return Err(NpyError::InvalidHeader(
            "v1.0 ヘッダに非 ASCII バイトが含まれる",
        ));
    }

    let header = parse_header(header_str)?;
    let data_bytes = &bytes[header_end..];

    let big_endian = match header.descr.as_str() {
        "<f4" => false,
        ">f4" => true,
        other => {
            return Err(NpyError::UnsupportedDtype {
                descr: other.chars().take(64).collect(),
            });
        }
    };

    let numel: usize = header.shape.iter().try_fold(1usize, |acc, &d| {
        acc.checked_mul(d).ok_or(NpyError::InvalidHeader(
            "shape 要素数積が usize 範囲を超える",
        ))
    })?;
    let expected_bytes = numel.checked_mul(4).ok_or(NpyError::InvalidHeader(
        "shape のバイト長が usize 範囲を超える",
    ))?;
    if data_bytes.len() != expected_bytes {
        return Err(NpyError::DataLengthMismatch {
            expected: expected_bytes,
            actual: data_bytes.len(),
        });
    }

    let mut data = Vec::with_capacity(numel);
    let (chunks, _remainder) = data_bytes.as_chunks::<4>();
    for chunk in chunks {
        // NaN のペイロード・±inf・-0.0・非正規化数を保持するため、
        // ビットパターンをそのまま読み取るだけで算術は通さない。
        let bits = if big_endian {
            u32::from_be_bytes(*chunk)
        } else {
            u32::from_le_bytes(*chunk)
        };
        data.push(f32::from_bits(bits));
    }

    if header.fortran_order && header.shape.len() > 1 {
        // Fortran（列優先）順のデータを、逆順 shape で一旦構築してから
        // `permute` で軸を逆転し `contiguous()` で C 順に実体化する
        // （値のコピーのみで bit は変えない）。
        let rank = header.shape.len();
        let mut rev_shape: Vec<usize> = header.shape.clone();
        rev_shape.reverse();
        let mut rev_perm: Vec<usize> = (0..rank).collect();
        rev_perm.reverse();
        let t = Tensor::new(data, &rev_shape)?;
        let permuted = t.permute(&rev_perm)?;
        Ok(permuted.contiguous())
    } else {
        Ok(Tensor::new(data, &header.shape)?)
    }
}

/// `path` の npy ファイルを読み込む。
///
/// ファイル全体を検証前に無条件で確保しないよう、`super::
/// read_file_bounded`（サイズ上限検査つき・TOCTOU 対策済み）を経由する
/// （`.claude/rules/security.md` A03/A04/A05。PR #2318 レビュー指摘）。
pub fn load_npy<P: AsRef<Path>>(path: P) -> Result<Tensor<f32>, NpyError> {
    let bytes = super::read_file_bounded(path.as_ref())?;
    read_npy_bytes(&bytes)
}

/// ヘッダ長フィールドに書き込む値（`dict` 本体長 + パディング + 改行）
/// を計算する。NumPy の `_wrap_header`（`numpy/lib/format.py`）と同一の
/// 計算式にする必要がある。
///
/// パディング量は「64 引く（X を 64 で割った余り）」で、X（magic +
/// version + len フィールド + 辞書 + 改行の合計）が既に 64 の倍数の
/// ときも 0 ではなく 64（1 ブロック丸ごと）を返す仕様である（少なくと
/// も 1 バイトの空白と改行を保証するための意図的な非対称）。単純な
/// 「64 の倍数へ切り上げ」実装（`div_ceil` ベース）はこの境界で 0
/// パディングを返し `np.save` と食い違っていた（PR #2318 レビュー
/// 指摘。Bugbot）。
///
/// `dict_len` は shape の桁数に依存し、巨大な shape でこの境界に達し
/// うる（実際にそのサイズの `Tensor` を構築せずに済むよう、この関数を
/// `write_npy_bytes` から分離してテストから直接呼べるようにしてある）。
fn header_len_for(dict_len: usize, len_field_size: usize) -> usize {
    let unpadded_total = 6 + 2 + len_field_size + dict_len + 1;
    let pad_len = 64 - (unpadded_total % 64);
    dict_len + 1 + pad_len
}

/// `shape` から npy ヘッダのレイアウト（バージョン・ヘッダ長フィールド幅・
/// ヘッダ本体長・dict 文字列）を計算する純粋関数。
///
/// `write_npy_bytes`（実際にバイト列を組み立てる）と `npy_encoded_len`
/// （バイト列を組み立てずに総出力長だけを見積もる）の双方から呼ばれ、
/// 計算式を 1 箇所にまとめることで両者が食い違う（drift する）ことを
/// 防ぐ。
fn npy_header_layout(shape: &[usize]) -> (u8, u8, usize, usize, String) {
    let shape_str = format_shape_tuple(shape);
    let dict = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': {shape_str}, }}");

    // NumPy `_wrap_header` と同じ計算: magic(6) + version(2) + headerlen
    // フィールド + ヘッダ本体（末尾 `\n` を含む）の合計が 64 の倍数に
    // なるよう空白で埋める。通常は v1.0（headerlen が u16）を使い、
    // パディング後の header_len が u16 上限を超える場合のみ v2.0（u32）
    // にする（NumPy `_write_array_header` と同じ規則）。
    let v1_header_len = header_len_for(dict.len(), 2);
    if v1_header_len <= u16::MAX as usize {
        (1u8, 0u8, 2usize, v1_header_len, dict)
    } else {
        (2u8, 0u8, 4usize, header_len_for(dict.len(), 4), dict)
    }
}

/// `write_npy_bytes` が `shape` に対して実際に生成する総バイト長を、
/// `Tensor` のデータを一切確保・シリアライズせずに shape だけから
/// 見積もる純粋関数。読み込み側（`read_npy_bytes`）が課す上限
/// （`MAX_RANK`・`MAX_HEADER_SIZE`・shape 要素数積・バイト長の `usize`
/// 範囲検査）を、書き込み側でも同じ定数・同じ計算式で事前検証するために
/// 使う（`write_npy_bytes` 自身に加え、`npz::write_npz_bytes` の
/// 事前検証パスからも呼ばれる。PR #2318 レビュー指摘・P2〈npz.rs:848。
/// npz の合計伸長後サイズ上限を書き出し側が検証していなかった〉の是正）。
///
/// これが実データを触らない純粋関数であることにより、境界値
/// （ちょうど上限／上限+1）を 1 GiB 相当の `Tensor` を実際に確保せずに
/// 単体テストできる（`.claude/rules/coding-rust.md` テスト・ベンチ節の
/// 意図に沿う）。
pub(crate) fn npy_encoded_len(shape: &[usize]) -> Result<u64, NpyError> {
    if shape.len() > MAX_RANK {
        return Err(NpyError::InvalidHeader("shape の rank が上限を超える"));
    }
    let (_major, _minor, len_field_size, header_len, _dict) = npy_header_layout(shape);
    // rank ≤ MAX_RANK（64）である限り dict 文字列長は高々 1500 バイト
    // 程度に収まり `MAX_HEADER_SIZE`（10000）を超えることは実際には
    // ないが、将来 `MAX_RANK`／`MAX_HEADER_SIZE` の値がずれても
    // fail-closed であり続けるよう、読み込み側と同じ検査をここでも行う
    // （定数を複製せず、`MAX_HEADER_SIZE` をそのまま参照する）。
    if header_len > MAX_HEADER_SIZE {
        return Err(NpyError::HeaderTooLarge {
            len: header_len,
            max: MAX_HEADER_SIZE,
        });
    }
    let numel: u64 = shape
        .iter()
        .try_fold(1u64, |acc, &d| acc.checked_mul(d as u64))
        .ok_or(NpyError::InvalidHeader(
            "shape 要素数積が usize 範囲を超える",
        ))?;
    let data_len = numel.checked_mul(4).ok_or(NpyError::InvalidHeader(
        "shape のバイト長が usize 範囲を超える",
    ))?;
    let prefix_len = 6u64 + 2 + len_field_size as u64 + header_len as u64;
    prefix_len
        .checked_add(data_len)
        .ok_or(NpyError::InvalidHeader(
            "npy 出力の総バイト長が u64 範囲を超える",
        ))
}

/// `t`（C 順に実体化して）を npy 形式のバイト列へ直列化する。
///
/// 出力は NumPy `np.save` の C 順 `<f4` 出力と**バイト完全一致**する
/// ことを目標とする（`docs/tensor-core-npy-npz-io-decision.md` §3.4）。
/// 非 contiguous な view（transpose・narrow 等）は `host_slice()` で
/// C 順に実体化してから書く。
///
/// 出力全体が `super::MAX_FILE_READ_BYTES`（`load_npy` が読み込みを
/// 許容する上限）を超える場合は、`fs::write` の前に `save_npy` が
/// `FileTooLarge` で拒否する（本関数の呼び出し元。`.npy` バイト列
/// 単体には NumPy 仕様上の総サイズ上限はないため、本関数自体は shape・
/// ヘッダの上限（`npy_encoded_len` 経由の `MAX_RANK`／
/// `MAX_HEADER_SIZE`）のみを検査する）。
pub fn write_npy_bytes(t: &Tensor<f32>) -> Result<Vec<u8>, NpyError> {
    let shape = t.shape();
    let predicted_len = npy_encoded_len(shape)?;
    let (major, minor, len_field_size, header_len, dict) = npy_header_layout(shape);

    let pad_len = header_len - dict.len() - 1;
    let mut out = Vec::with_capacity(predicted_len as usize);
    out.extend_from_slice(&MAGIC);
    out.push(major);
    out.push(minor);
    if len_field_size == 2 {
        out.extend_from_slice(&(header_len as u16).to_le_bytes());
    } else {
        out.extend_from_slice(&(header_len as u32).to_le_bytes());
    }
    out.extend_from_slice(dict.as_bytes());
    out.extend(std::iter::repeat_n(b' ', pad_len));
    out.push(b'\n');

    let host = t.host_slice();
    out.reserve(host.len() * 4);
    for &v in host.iter() {
        out.extend_from_slice(&v.to_bits().to_le_bytes());
    }
    Ok(out)
}

/// `t` を `path` へ npy 形式で書き出す。バイト列をメモリ上で組み立てて
/// から `std::fs::write` する（原子的な書き込みではない）。
///
/// `write_npy_bytes` の出力全体が `super::MAX_FILE_READ_BYTES` を超える
/// 場合は `fs::write` の前に `FileTooLarge` で拒否し、部分ファイルを
/// 残さない。`load_npy` が `read_file_bounded` で課すファイルサイズ上限を
/// 書き出し側にも適用することで、自前書き出し→読み込みの往復契約を保つ
/// （PR #2318 レビュー指摘・P2〈npz.rs:848〉是正の一環。npy 単体は
/// `write_npy_bytes` 自体には総サイズ上限がないため、ここで初めて
/// `load_npy` の入口と同じ定数を適用する）。
pub fn save_npy<P: AsRef<Path>>(t: &Tensor<f32>, path: P) -> Result<(), NpyError> {
    let bytes = write_npy_bytes(t)?;
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

/// shape を NumPy の tuple repr（`()`／`(3,)`／`(2, 3)`）へ整形する。
fn format_shape_tuple(shape: &[usize]) -> String {
    match shape.len() {
        0 => "()".to_string(),
        1 => format!("({},)", shape[0]),
        _ => {
            let joined = shape
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            format!("({joined})")
        }
    }
}

/// npy ヘッダ辞書の専用最小パーサ。
///
/// 受理する形（`docs/tensor-core-npy-npz-io-decision.md` §3.3）:
/// `{` キー `:` 値 (`,` キー `:` 値)* `,`? `}` の後ろに空白と `\n` のみ。
/// キーは `descr`／`fortran_order`／`shape` の 3 つで、順不同・各 1 回
/// ずつ必須。未知キー・重複・欠落はすべて `InvalidHeader`。
fn parse_header(header_str: &str) -> Result<Header, NpyError> {
    let bytes = header_str.as_bytes();
    let mut pos = 0usize;

    skip_ws(bytes, &mut pos);
    expect_byte(bytes, &mut pos, b'{')?;

    let mut descr: Option<String> = None;
    let mut fortran_order: Option<bool> = None;
    let mut shape: Option<Vec<usize>> = None;

    loop {
        skip_ws(bytes, &mut pos);
        if peek(bytes, pos) == Some(b'}') {
            pos += 1;
            break;
        }
        let key = parse_quoted_string(header_str, bytes, &mut pos)?;
        skip_ws(bytes, &mut pos);
        expect_byte(bytes, &mut pos, b':')?;
        skip_ws(bytes, &mut pos);
        match key.as_str() {
            "descr" => {
                if descr.is_some() {
                    return Err(NpyError::InvalidHeader("descr キーが重複している"));
                }
                descr = Some(parse_quoted_string(header_str, bytes, &mut pos)?);
            }
            "fortran_order" => {
                if fortran_order.is_some() {
                    return Err(NpyError::InvalidHeader("fortran_order キーが重複している"));
                }
                fortran_order = Some(parse_bool_literal(bytes, &mut pos)?);
            }
            "shape" => {
                if shape.is_some() {
                    return Err(NpyError::InvalidHeader("shape キーが重複している"));
                }
                shape = Some(parse_shape_tuple(header_str, bytes, &mut pos)?);
            }
            _ => return Err(NpyError::InvalidHeader("未知のヘッダキー")),
        }
        skip_ws(bytes, &mut pos);
        match peek(bytes, pos) {
            Some(b',') => {
                pos += 1;
                continue;
            }
            Some(b'}') => {
                pos += 1;
                break;
            }
            _ => {
                return Err(NpyError::InvalidHeader(
                    "キー・値の後ろに `,` または `}` が必要",
                ));
            }
        }
    }

    // `}` 以降は空白のみで、末尾はちょうど 1 つの `\n`（本モジュールが
    // 追加確認済みのパディング契約。埋め込み改行による多重ヘッダ偽装を
    // 防ぐ。`.claude/rules/security.md` A03）。
    let tail = &bytes[pos..];
    if tail.iter().any(|&b| b != b' ' && b != b'\n') {
        return Err(NpyError::InvalidHeader(
            "`}` の後ろに空白・改行以外の文字がある",
        ));
    }
    if tail.iter().filter(|&&b| b == b'\n').count() != 1 || tail.last() != Some(&b'\n') {
        return Err(NpyError::InvalidHeader(
            "ヘッダの終端が単一の `\\n` ではない",
        ));
    }

    let descr = descr.ok_or(NpyError::InvalidHeader("descr キーが欠落している"))?;
    let fortran_order =
        fortran_order.ok_or(NpyError::InvalidHeader("fortran_order キーが欠落している"))?;
    let shape = shape.ok_or(NpyError::InvalidHeader("shape キーが欠落している"))?;

    Ok(Header {
        descr,
        fortran_order,
        shape,
    })
}

fn peek(bytes: &[u8], pos: usize) -> Option<u8> {
    bytes.get(pos).copied()
}

fn skip_ws(bytes: &[u8], pos: &mut usize) {
    while matches!(
        peek(bytes, *pos),
        Some(b' ') | Some(b'\t') | Some(b'\n') | Some(b'\r')
    ) {
        *pos += 1;
    }
}

fn expect_byte(bytes: &[u8], pos: &mut usize, expected: u8) -> Result<(), NpyError> {
    if peek(bytes, *pos) == Some(expected) {
        *pos += 1;
        Ok(())
    } else {
        Err(NpyError::InvalidHeader("期待するトークンが見つからない"))
    }
}

/// シングル／ダブルクォートの文字列リテラルを解析する（エスケープは
/// 扱わない。`descr`／キー名にクォート文字・バックスラッシュは
/// 現れないため必要十分）。
fn parse_quoted_string(
    header_str: &str,
    bytes: &[u8],
    pos: &mut usize,
) -> Result<String, NpyError> {
    let quote = match peek(bytes, *pos) {
        Some(b'\'') | Some(b'"') => bytes[*pos],
        _ => {
            return Err(NpyError::InvalidHeader(
                "文字列リテラルが `'` または `\"` で始まらない",
            ));
        }
    };
    let start = *pos + 1;
    let mut i = start;
    while i < bytes.len() && bytes[i] != quote {
        i += 1;
    }
    if i >= bytes.len() {
        return Err(NpyError::InvalidHeader("文字列リテラルが閉じていない"));
    }
    let s = header_str
        .get(start..i)
        .ok_or(NpyError::InvalidHeader(
            "文字列リテラルが有効な UTF-8 境界にない",
        ))?
        .to_string();
    *pos = i + 1;
    Ok(s)
}

fn parse_bool_literal(bytes: &[u8], pos: &mut usize) -> Result<bool, NpyError> {
    if bytes[*pos..].starts_with(b"True") {
        *pos += 4;
        Ok(true)
    } else if bytes[*pos..].starts_with(b"False") {
        *pos += 5;
        Ok(false)
    } else {
        Err(NpyError::InvalidHeader(
            "fortran_order の値が True/False ではない",
        ))
    }
}

fn parse_shape_tuple(
    header_str: &str,
    bytes: &[u8],
    pos: &mut usize,
) -> Result<Vec<usize>, NpyError> {
    expect_byte(bytes, pos, b'(')?;
    let mut dims = Vec::new();
    loop {
        skip_ws(bytes, pos);
        if peek(bytes, *pos) == Some(b')') {
            *pos += 1;
            break;
        }
        let start = *pos;
        while matches!(peek(bytes, *pos), Some(b) if b.is_ascii_digit()) {
            *pos += 1;
        }
        if *pos == start {
            return Err(NpyError::InvalidHeader("shape の要素が非負整数ではない"));
        }
        let digits = header_str.get(start..*pos).ok_or(NpyError::InvalidHeader(
            "shape の要素が有効な UTF-8 境界にない",
        ))?;
        let dim: usize = digits
            .parse()
            .map_err(|_| NpyError::InvalidHeader("shape の要素が usize 範囲を超える"))?;
        dims.push(dim);
        if dims.len() > MAX_RANK {
            return Err(NpyError::InvalidHeader("shape の rank が上限を超える"));
        }
        skip_ws(bytes, pos);
        match peek(bytes, *pos) {
            Some(b',') => {
                *pos += 1;
                continue;
            }
            Some(b')') => {
                // rank 1 は Python のタプルリテラルとして `(3,)` が正で
                // `(3)` は単一整数（タプルではない）。numpy が書き出す
                // shape は常に `(3,)` 形式のため、末尾カンマなしの
                // rank 1（要素 1 個のまま `,` を経ずに `)` へ到達した
                // 場合）は誤受理せず拒否する（PR #2318 レビュー指摘・
                // P2。npy.rs:454）。
                if dims.len() == 1 {
                    return Err(NpyError::InvalidHeader(
                        "shape が rank 1 の場合は `(N,)` のように末尾カンマが必要",
                    ));
                }
                *pos += 1;
                break;
            }
            _ => {
                return Err(NpyError::InvalidHeader(
                    "shape の要素の後ろに `,` または `)` が必要",
                ));
            }
        }
    }
    Ok(dims)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_v1_bytes(dict: &str) -> Vec<u8> {
        let unpadded = 6 + 2 + 2 + dict.len() + 1;
        let padded = unpadded.div_ceil(64) * 64;
        let header_len = padded - 6 - 2 - 2;
        let pad = header_len - dict.len() - 1;
        let mut out = Vec::new();
        out.extend_from_slice(&MAGIC);
        out.push(1);
        out.push(0);
        out.extend_from_slice(&(header_len as u16).to_le_bytes());
        out.extend_from_slice(dict.as_bytes());
        out.extend(std::iter::repeat_n(b' ', pad));
        out.push(b'\n');
        out
    }

    fn header_str_from(dict: &str) -> String {
        let bytes = make_v1_bytes(dict);
        let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
        String::from_utf8(bytes[10..10 + header_len].to_vec()).unwrap()
    }

    #[test]
    fn parses_key_order_variations() {
        let h1 = parse_header(&header_str_from(
            "{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3), }",
        ))
        .unwrap();
        assert_eq!(h1.descr, "<f4");
        assert!(!h1.fortran_order);
        assert_eq!(h1.shape, vec![2, 3]);

        let h2 = parse_header(&header_str_from(
            "{'shape': (2, 3), 'descr': '<f4', 'fortran_order': False}",
        ))
        .unwrap();
        assert_eq!(h2.shape, vec![2, 3]);
    }

    #[test]
    fn parses_whitespace_variations() {
        let h = parse_header(&header_str_from(
            "{'descr':'<f4','fortran_order':True,'shape':(3,)}",
        ))
        .unwrap();
        assert!(h.fortran_order);
        assert_eq!(h.shape, vec![3]);
    }

    #[test]
    fn parses_rank0_and_rank1_and_trailing_comma_shapes() {
        let h0 = parse_header(&header_str_from(
            "{'descr': '<f4', 'fortran_order': False, 'shape': (), }",
        ))
        .unwrap();
        assert_eq!(h0.shape, Vec::<usize>::new());

        let h1 = parse_header(&header_str_from(
            "{'descr': '<f4', 'fortran_order': False, 'shape': (3,), }",
        ))
        .unwrap();
        assert_eq!(h1.shape, vec![3]);
    }

    #[test]
    fn rejects_rank1_shape_without_trailing_comma() {
        // P2（codex-review。npy.rs:454）: Python では `(3)` は単一整数
        // リテラルでありタプルではない（`(3,)` が rank1 タプル）。numpy
        // が書き出す shape は常に `(3,)` 形式のため、末尾カンマなしの
        // `(3)` を rank1 shape として誤受理しないことを確認する。
        let err = parse_header(&header_str_from(
            "{'descr': '<f4', 'fortran_order': False, 'shape': (3), }",
        ));
        assert!(
            matches!(err, Err(NpyError::InvalidHeader(_))),
            "末尾カンマなしの rank1 shape が拒否されなかった"
        );
    }

    #[test]
    fn rejects_invalid_token() {
        let err = parse_header(&header_str_from(
            "{'descr': '<f4', 'fortran_order': maybe, 'shape': (2,), }",
        ));
        assert!(matches!(err, Err(NpyError::InvalidHeader(_))));
    }

    #[test]
    fn rejects_duplicate_key() {
        let err = parse_header(&header_str_from(
            "{'descr': '<f4', 'descr': '<f4', 'fortran_order': False, 'shape': (2,), }",
        ));
        assert!(matches!(err, Err(NpyError::InvalidHeader(_))));
    }

    #[test]
    fn rejects_missing_key() {
        let err = parse_header(&header_str_from("{'descr': '<f4', 'shape': (2,), }"));
        assert!(matches!(err, Err(NpyError::InvalidHeader(_))));
    }

    #[test]
    fn rejects_unknown_key() {
        let err = parse_header(&header_str_from(
            "{'descr': '<f4', 'fortran_order': False, 'shape': (2,), 'extra': 1, }",
        ));
        assert!(matches!(err, Err(NpyError::InvalidHeader(_))));
    }

    #[test]
    fn rejects_integer_overflow_in_shape() {
        let err = parse_header(&header_str_from(
            "{'descr': '<f4', 'fortran_order': False, 'shape': (99999999999999999999999,), }",
        ));
        assert!(matches!(err, Err(NpyError::InvalidHeader(_))));
    }

    #[test]
    fn header_len_for_matches_numpy_wrap_header_boundary() {
        // PR #2318 レビュー指摘（Bugbot）: パディング前のヘッダ長
        // （unpadded_total）がちょうど 64 の倍数になる場合、`div_ceil`
        // ベースの旧実装は pad_len=0 を返し `np.save`（`_wrap_header`）
        // と食い違っていた。境界を作る dict 長（実際に構築するには
        // 61 桁の shape 数字が要るため、`header_len_for` を直接呼んで
        // 検証する）で pad_len が 0 ではなく 64 になることを確認する。
        // dict_len=53 のとき unpadded_total = 6+2+2+53+1 = 64（ちょうど
        // 境界）。
        let dict_len = 53;
        let len_field_size = 2;
        assert_eq!((6 + 2 + len_field_size + dict_len + 1) % 64, 0);
        let header_len = header_len_for(dict_len, len_field_size);
        let pad_len = header_len - dict_len - 1;
        assert_eq!(
            pad_len, 64,
            "unpadded_total が 64 の倍数のとき pad_len は 64 でなければならない（0 は np.save と不一致）"
        );

        // 非境界の通常ケース（従来どおり 1..64 の範囲で正しく求まる）。
        for dict_len in 0..300usize {
            let header_len = header_len_for(dict_len, len_field_size);
            let total = 6 + 2 + len_field_size + header_len;
            assert_eq!(total % 64, 0, "dict_len={dict_len} で全体長が非整合");
            let pad_len = header_len - dict_len - 1;
            assert!(
                (1..=64).contains(&pad_len),
                "dict_len={dict_len} で pad_len={pad_len} が [1, 64] の範囲外"
            );
        }
    }

    #[test]
    fn write_npy_bytes_header_always_64_byte_aligned() {
        // 実際の write_npy_bytes 経路（`Tensor` → shape 文字列 → dict
        // 長）でも、多数の rank-1 shape に対しヘッダ全体長が常に 64 の
        // 倍数になることを回帰確認する（配線の健全性チェック）。
        for n in 0..300usize {
            let t = Tensor::new(vec![0.0f32; n], &[n]).unwrap();
            let bytes = write_npy_bytes(&t).unwrap();
            let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
            let prefix_total = 10 + header_len; // magic(6)+ver(2)+len_field(2)
            assert_eq!(
                prefix_total % 64,
                0,
                "n={n} でヘッダ全体長が 64 の倍数でない"
            );
        }
    }

    #[test]
    fn rank0_roundtrip_via_write_and_read() {
        let t = Tensor::new(vec![1.5f32], &[]).unwrap();
        let bytes = write_npy_bytes(&t).unwrap();
        let back = read_npy_bytes(&bytes).unwrap();
        assert_eq!(back.shape(), &[] as &[usize]);
        assert_eq!(back.host_slice().to_vec(), vec![1.5f32]);
    }

    #[test]
    fn empty_array_roundtrip() {
        let t = Tensor::new(Vec::<f32>::new(), &[0, 4]).unwrap();
        let bytes = write_npy_bytes(&t).unwrap();
        let back = read_npy_bytes(&bytes).unwrap();
        assert_eq!(back.shape(), &[0, 4]);
        assert!(back.host_slice().is_empty());
    }

    #[test]
    fn write_then_read_preserves_bits() {
        let vals = vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.0f32, 1.0f32];
        let t = Tensor::new(vals.clone(), &[5]).unwrap();
        let bytes = write_npy_bytes(&t).unwrap();
        let back = read_npy_bytes(&bytes).unwrap();
        let back_slice = back.host_slice();
        for (a, b) in vals.iter().zip(back_slice.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }

    #[test]
    fn rejects_unsupported_dtype() {
        let dict = "{'descr': '<f8', 'fortran_order': False, 'shape': (2,), }";
        let bytes = make_v1_bytes(dict);
        let mut full = bytes.clone();
        full.extend(std::iter::repeat_n(0u8, 16));
        let err = read_npy_bytes(&full);
        assert!(matches!(err, Err(NpyError::UnsupportedDtype { .. })));
    }

    #[test]
    fn rejects_data_length_mismatch() {
        let dict = "{'descr': '<f4', 'fortran_order': False, 'shape': (2,), }";
        let bytes = make_v1_bytes(dict);
        let mut full = bytes.clone();
        full.extend(std::iter::repeat_n(0u8, 4)); // 1 要素分しかない（期待 2 要素 = 8 バイト）
        let err = read_npy_bytes(&full);
        assert!(matches!(err, Err(NpyError::DataLengthMismatch { .. })));
    }

    #[test]
    fn rejects_invalid_magic() {
        let err = read_npy_bytes(b"NOTNUMPYxxxxxxxx");
        assert!(matches!(err, Err(NpyError::InvalidMagic)));
    }

    #[test]
    fn rejects_unsupported_version() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&MAGIC);
        bytes.push(9);
        bytes.push(9);
        bytes.extend_from_slice(&0u16.to_le_bytes());
        let err = read_npy_bytes(&bytes);
        assert!(matches!(err, Err(NpyError::UnsupportedVersion { .. })));
    }

    #[test]
    fn fortran_order_roundtrip_matches_c_order() {
        // C 順 [[1,2,3],[4,5,6]] を Fortran 順のバイト列（列優先: 1,4,2,5,3,6）
        // として手組みし、読み込み後に C 順と一致することを確認する。
        let dict = "{'descr': '<f4', 'fortran_order': True, 'shape': (2, 3), }";
        let mut bytes = make_v1_bytes(dict);
        let col_major: [f32; 6] = [1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        for v in col_major {
            bytes.extend_from_slice(&v.to_bits().to_le_bytes());
        }
        let t = read_npy_bytes(&bytes).unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.host_slice().to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn big_endian_roundtrip() {
        let dict = "{'descr': '>f4', 'fortran_order': False, 'shape': (2,), }";
        let mut bytes = make_v1_bytes(dict);
        bytes.extend_from_slice(&1.5f32.to_bits().to_be_bytes());
        bytes.extend_from_slice(&(-2.5f32).to_bits().to_be_bytes());
        let t = read_npy_bytes(&bytes).unwrap();
        assert_eq!(t.host_slice().to_vec(), vec![1.5, -2.5]);
    }

    // PR #2318 レビュー指摘・P2（npz.rs:848）の是正で追加した
    // `npy_encoded_len`／rank 上限の writer 側検証に対するテスト群。
    // 「reader が課す制約を writer も同じ定数で検証する」往復契約を
    // 固定する（`.claude/rules/coding-rust.md` テスト・ベンチ節）。

    #[test]
    fn npy_encoded_len_matches_write_npy_bytes_actual_len() {
        // 見積り関数（shape のみを見る純粋関数）と実際のシリアライズ
        // 結果が食い違わないことを、複数の小さな shape で確認する
        // （`write_npy_bytes` 内の `debug_assert_eq!` と同じ契約を、通常
        // テストの範囲でも直接検証する）。
        for shape in [vec![], vec![0usize], vec![3], vec![2, 3], vec![1; MAX_RANK]] {
            let numel: usize = shape.iter().product();
            let t = Tensor::new(vec![0.0f32; numel], &shape).unwrap();
            let predicted = npy_encoded_len(&shape).unwrap();
            let actual = write_npy_bytes(&t).unwrap();
            assert_eq!(
                predicted,
                actual.len() as u64,
                "shape={shape:?} で予測長と実出力長が不一致"
            );
        }
    }

    #[test]
    fn write_npy_bytes_accepts_rank_at_max_and_rejects_rank_above_max() {
        // reader（`parse_shape_tuple` の `MAX_RANK` 検査）が rank ≤ 64
        // までしか受理しないため、writer も同じ定数で rank 65 以上を
        // 事前に拒否しなければ「書き出せるが読み込めない」往復不能な
        // 出力になる（本 PR の是正対象。要素数は 1 のまま rank だけを
        // 動かすため、実データの確保量は無視できるほど小さい）。
        let at_max = vec![1usize; MAX_RANK];
        let t_ok = Tensor::new(vec![0.0f32; 1], &at_max).unwrap();
        assert!(write_npy_bytes(&t_ok).is_ok());

        let above_max = vec![1usize; MAX_RANK + 1];
        let t_over = Tensor::new(vec![0.0f32; 1], &above_max).unwrap();
        let err = write_npy_bytes(&t_over);
        assert!(
            matches!(err, Err(NpyError::InvalidHeader(_))),
            "rank が上限を超える shape が拒否されなかった: {err:?}"
        );
    }

    #[test]
    fn npy_encoded_len_exact_file_size_boundary_without_allocating_data() {
        // `save_npy` が `super::MAX_FILE_READ_BYTES`（1 GiB）超過を
        // 拒否する境界を、実際に 1 GiB 相当の `Tensor` を確保せずに
        // 検証する。`npy_encoded_len` は shape（`&[usize]`）のみを見る
        // 純粋関数であり、rank-1 の 1 要素スライスに対する呼び出しは
        // 巨大な実データを一切割り当てない。
        //
        // 9 桁の要素数（今回の対象領域）では、同じ桁数を持つ n に対する
        // header 長は不変（shape 文字列長が桁数だけに依存するため）。
        // これを利用して、まず適当な 9 桁の n でヘッダ込みの固定長
        // （prefix_len）を求め、その固定長から目標総バイト長ちょうどに
        // なる n を厳密に逆算する。
        let target = super::super::MAX_FILE_READ_BYTES;
        let probe_n: u64 = 268_435_456; // 9 桁（target/4 の概算）
        let probe_len = npy_encoded_len(&[probe_n as usize]).unwrap();
        let prefix_len = probe_len - probe_n * 4;
        assert_eq!(
            (target - prefix_len) % 4,
            0,
            "prefix_len は 64 バイト境界のため 4 の倍数のはず"
        );
        let n_at_cap = ((target - prefix_len) / 4) as usize;
        // 桁数が想定どおり 9 桁のままであること（逆算の前提条件）。
        assert_eq!(n_at_cap.to_string().len(), probe_n.to_string().len());

        let len_at_cap = npy_encoded_len(&[n_at_cap]).unwrap();
        assert_eq!(len_at_cap, target, "ちょうど上限になる n の逆算が外れた");

        let len_over_cap = npy_encoded_len(&[n_at_cap + 1]).unwrap();
        assert_eq!(
            len_over_cap,
            target + 4,
            "n+1（4 バイト分の 1 要素増）で総バイト長が 4 だけ増えるはず"
        );
        // `npy_encoded_len` 自体は shape・ヘッダの上限だけを検査し、
        // ファイルサイズ上限（`MAX_FILE_READ_BYTES`）は呼び出し元
        // （`write_npy_bytes`／`save_npy`）が単純な数値比較で適用する
        // （実データを持たないここでは、その比較対象になる正しい
        // 総バイト長が求まることまでを検証する）。
        assert!(len_at_cap <= target);
        assert!(len_over_cap > target);
    }
}
