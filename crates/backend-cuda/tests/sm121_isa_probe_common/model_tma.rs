//! TMA（`cp.async.bulk.tensor`）の encoder 引数とホスト側の候補モデル（イシュー #2122 PR-B。
//! RULE.txt R-TMA-BASE／R-TMA-SEM／R-TMA-XFER）。GPU を使わない純関数で、
//! `sm121_isa_probe_registry`（CI）が単体テストする。
//!
//! 方針: **期待値は持たない**。実機の観測値（smem のダンプ・global の読み戻し）を、
//! 事前登録した候補モデルと突き合わせ、一致したモデル名（複数可）または `NONE`
//! （ダンプ全文を記録）を `k=v` トークンで S5 の detail へ記録する。ただし `tma.base_cta`
//! （最も基本的な box 転送）・`tma.store`・`tma.bulk_*`・`tma.multicast`・`tma.prefetch` は
//! 「成立するか」を問うため、完走とビット一致（またはその目印）を判定する。
//!
//! データ: global テンソルは f32（要素値は行 r・列 c を表す正確な整数 `r * 1000 + c` の f32
//! ビット列）。値から行・列を逆算でき、OOB 要素の fill 値（0・NaN・番兵）と区別できる。

use super::registry::Outcome;

/// OOB の番兵（smem を転送前に埋める値。`kernels_tma.rs` の `TMA_SENTINEL` と一致）。
pub const SENTINEL: u32 = 0xFEED_FACE;
/// `tma.store` の完走の目印（カーネルの out[0]）。
pub const STORE_MAGIC: u32 = 0x5702_5E5E;
/// `tma.prefetch` の完走の目印。
pub const PREFETCH_MAGIC: u32 = 0x9E7F_3C11;
/// 出力ヘッダの語数（全 TMA カーネル共通で `[polls]` の 1 語。`kernels_tma.rs` の格納と registry テストが
/// 一致を検査する）。待ちが上限に達したカーネルは転送先を読まず終了もしない（外部 timeout で打ち切り）
/// ため、ヘッダに「待ちの失敗」の状態は持たない。
pub const HDR: usize = 1;
/// smem の box 最大語数（`kernels_tma.rs` の `smem[512]`）。
pub const SMEM_WORDS: u32 = 512;

/// swizzle 幅（`CUtensorMapSwizzle` の 32B／64B／128B）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Swz {
    None,
    B32,
    B64,
    B128,
}

impl Swz {
    /// swizzle アトム幅（バイト。`None` は 0）。
    pub const fn span_bytes(self) -> u32 {
        match self {
            Swz::None => 0,
            Swz::B32 => 32,
            Swz::B64 => 64,
            Swz::B128 => 128,
        }
    }

    /// アドレスビット [7, 7+B) を [4, 4+B) へ XOR する標準モデルの B（32B=1・64B=2・128B=3）。
    const fn xor_mask(self) -> u32 {
        match self {
            Swz::None => 0,
            Swz::B32 => 1,
            Swz::B64 => 3,
            Swz::B128 => 7,
        }
    }
}

/// 1 プローブ分の encoder 引数と起動パラメータ（`cuTensorMapEncodeTiled` の引数化）。
#[derive(Debug, Clone, Copy)]
pub struct TmaSpec {
    pub global_rows: u32,
    pub global_cols: u32,
    pub box_rows: u32,
    pub box_cols: u32,
    pub swizzle: Swz,
    /// OOB fill（`false` = NONE〈0 埋め〉・`true` = `NAN_REQUEST_ZERO_FMA`）。
    pub oob_nan: bool,
    /// 座標（要素単位。`cx` が内側〈列〉、`cy` が外側〈行〉）。
    pub cx: i32,
    pub cy: i32,
    /// `mbarrier.arrive.expect_tx` の総量。tx-count を実際の転送量より少なく期待するとアンダーフロー
    /// （未定義動作）になるため、load・multicast では box 全体のバイト数（OOB を含む）と一致させる
    /// （registry テストが検査する）。
    pub expect_tx: u32,
    pub dump_words: u32,
    /// global を起動後に読み戻して out の後ろへ連結する（store 用）。
    pub readback_global: bool,
    /// global の初期内容を番兵で埋める（store 用。それ以外は `row*1000+col`）。
    pub global_sentinel: bool,
}

impl TmaSpec {
    pub const fn box_words(&self) -> u32 {
        self.box_rows * self.box_cols
    }

    pub const fn global_words(&self) -> u32 {
        self.global_rows * self.global_cols
    }

    /// encoder の前提と smem・expect_tx の整合を検査する（`cuTensorMapEncodeTiled` の制約の
    /// うち本プローブが依存するもの。違反は `Err`。registry テストが全 spec へ適用する）。
    pub fn validate(&self) -> Result<(), String> {
        if self.global_rows == 0
            || self.global_cols == 0
            || self.box_rows == 0
            || self.box_cols == 0
        {
            return Err("ゼロ次元".into());
        }
        if self.box_rows > 256 || self.box_cols > 256 {
            return Err("boxDim は 256 以下".into());
        }
        if !(self.global_cols * 4).is_multiple_of(16) {
            return Err("global の行ストライドは 16 バイトの倍数".into());
        }
        if !(self.box_cols * 4).is_multiple_of(16) {
            return Err("box の内側次元のバイト幅は 16 の倍数".into());
        }
        if self.swizzle != Swz::None && self.box_cols * 4 > self.swizzle.span_bytes() {
            return Err("swizzle 使用時は box の内側次元のバイト幅が swizzle 幅以下".into());
        }
        if self.box_words() > SMEM_WORDS || self.dump_words > SMEM_WORDS {
            return Err("box／ダンプが smem 配列（512 語）を超える".into());
        }
        if self.expect_tx > self.box_words() * 4 {
            return Err("expect_tx は box 全体のバイト数以下".into());
        }
        Ok(())
    }
}

/// global データ（行優先）。要素 `(r, c)` は f32 値 `r * 1000 + c` のビット列。
pub fn global_data(rows: u32, cols: u32) -> Vec<u32> {
    (0..rows)
        .flat_map(|r| (0..cols).map(move |c| ((r * 1000 + c) as f32).to_bits()))
        .collect()
}

/// `spec` の global 初期内容。
pub fn global_input(spec: &TmaSpec) -> Vec<u32> {
    if spec.global_sentinel {
        vec![SENTINEL; spec.global_words() as usize]
    } else {
        global_data(spec.global_rows, spec.global_cols)
    }
}

/// box の各要素（行優先 `box_rows × box_cols`）の仮説上の値。範囲内なら `Some(bits)`、
/// OOB なら `None`。`transposed` は座標を（行, 列）= (cx, cy) と読む仮説。
pub fn box_elems(spec: &TmaSpec, data: &[u32], transposed: bool) -> Vec<Option<u32>> {
    let mut v = Vec::new();
    for r in 0..spec.box_rows as i64 {
        for c in 0..spec.box_cols as i64 {
            let (row, col) = if transposed {
                (i64::from(spec.cx) + r, i64::from(spec.cy) + c)
            } else {
                (i64::from(spec.cy) + r, i64::from(spec.cx) + c)
            };
            let inside = row >= 0
                && col >= 0
                && row < i64::from(spec.global_rows)
                && col < i64::from(spec.global_cols);
            v.push(inside.then(|| data[(row * i64::from(spec.global_cols) + col) as usize]));
        }
    }
    v
}

/// 標準の swizzle モデル（smem 上の物理バイトアドレスのビット [7,7+B) を [4,4+B) へ XOR。
/// タイル先頭が 1024 バイト整列であることが前提）。線形の語添字 → 物理の語添字。
pub fn swizzle_xor_phys_word(swz: Swz, linear_word: u32) -> u32 {
    let byte = linear_word * 4;
    let phys = byte ^ (((byte >> 7) & swz.xor_mask()) << 4);
    phys / 4
}

/// src の B64 仮説（`tma_swizzled_chunk_a`。`TmaSwizzleA::B64`）による線形の語添字 → 物理の語添字。
/// 1 行 = 16 語（64B）の box（`box_cols == 16`）にのみ意味を持つ。
pub fn swizzle_src_b64_phys_word(linear_word: u32) -> u32 {
    let (row, kk) = (linear_word / 16, linear_word % 16);
    row * 16 + fandhe_ai_backend_cuda::tma_swizzled_chunk_a(row, kk)
}

fn hex_list(words: &[u32]) -> String {
    words
        .iter()
        .map(|w| format!("0x{w:08x}"))
        .collect::<Vec<_>>()
        .join(",")
}

/// ヘッダの `k=v` トークン（ヘッダは `[polls]` の 1 語。データ語を混ぜない）。
fn header_tokens(out: &[u32]) -> String {
    format!("polls={}", out[0])
}

fn len_mismatch(spec: &TmaSpec, got: usize) -> Option<Outcome> {
    let want = HDR + spec.dump_words as usize;
    (got != want).then(|| Outcome::Mismatch {
        first: got.min(want),
        count: got.abs_diff(want).max(1),
        detail: format!("length expected={want} got={got}"),
    })
}

/// 基本の box 転送（`tma.base_cta`／`tma.base_cluster`）: mbarrier が完了し、box 全体が
/// 要素座標（内側次元が先）の仮説どおりにビット一致するか（成立の判定）。
pub fn check_base(spec: &TmaSpec, input: &[u32], out: &[u32]) -> Outcome {
    if let Some(m) = len_mismatch(spec, out.len()) {
        return m;
    }
    let elems = box_elems(spec, input, false);
    let dump = &out[HDR..];
    let bad: Vec<usize> = elems
        .iter()
        .zip(dump)
        .enumerate()
        .filter(|(_, (e, d))| **e != Some(**d))
        .map(|(i, _)| i)
        .collect();
    if bad.is_empty() {
        Outcome::Match(format!(
            "{} box_words={} bit_exact=true",
            header_tokens(out),
            elems.len()
        ))
    } else {
        Outcome::Mismatch {
            first: bad.first().copied().unwrap_or(0),
            count: bad.len().max(1),
            detail: format!(
                "{} mismatched_words={} dump={}",
                header_tokens(out),
                bad.len(),
                hex_list(dump)
            ),
        }
    }
}

/// `tma.coord`: 座標の解釈の仮説（内側次元が先の要素座標・転置）のどれと一致するかを記録する。
pub fn classify_coord(spec: &TmaSpec, input: &[u32], out: &[u32]) -> Outcome {
    if let Some(m) = len_mismatch(spec, out.len()) {
        return m;
    }
    let dump = &out[HDR..];
    let matches = |t: bool| {
        box_elems(spec, input, t)
            .iter()
            .zip(dump)
            .all(|(e, d)| *e == Some(*d))
    };
    let mut classes = Vec::new();
    if matches(false) {
        classes.push("ELEM_INNER_FIRST");
    }
    if matches(true) {
        classes.push("TRANSPOSED");
    }
    let class = if classes.is_empty() {
        "NONE".to_string()
    } else {
        classes.join("+")
    };
    let tail = if classes.is_empty() {
        format!(" dump={}", hex_list(dump))
    } else {
        String::new()
    };
    Outcome::Record(format!("{} class={class}{tail}", header_tokens(out)))
}

fn fill_class(v: u32) -> &'static str {
    if v == 0 {
        "ZERO"
    } else if v == SENTINEL {
        "SENTINEL"
    } else if f32::from_bits(v).is_nan() {
        "NAN"
    } else {
        "OTHER"
    }
}

/// `tma.oob_*`: 範囲内の要素が仮説どおりか（`inrange`）と、OOB 要素のビット列の種類を記録する。
pub fn classify_oob(spec: &TmaSpec, input: &[u32], out: &[u32]) -> Outcome {
    if let Some(m) = len_mismatch(spec, out.len()) {
        return m;
    }
    let dump = &out[HDR..];
    let elems = box_elems(spec, input, false);
    let mut in_ok = true;
    let mut oob: Vec<u32> = Vec::new();
    for (e, d) in elems.iter().zip(dump) {
        match e {
            Some(want) => in_ok &= want == d,
            None => oob.push(*d),
        }
    }
    let mut distinct = oob.clone();
    distinct.sort_unstable();
    distinct.dedup();
    // 順序を保ったまま全重複を除去する（`Vec::dedup` は連続した重複しか除かない）。
    let mut classes: Vec<&str> = Vec::new();
    for c in distinct.iter().map(|v| fill_class(*v)) {
        if !classes.contains(&c) {
            classes.push(c);
        }
    }
    let fill = if oob.is_empty() {
        "NO_OOB_ELEMENTS".to_string()
    } else {
        classes.join("+")
    };
    Outcome::Record(format!(
        "{} inrange={} oob_elems={} oob_fill={fill} oob_distinct={}",
        header_tokens(out),
        if in_ok { "MATCH" } else { "MISMATCH" },
        oob.len(),
        if distinct.is_empty() {
            "-".to_string()
        } else {
            hex_list(&distinct)
        },
    ))
}

/// `tma.swz*`: smem の線形ダンプが、候補モデル（無 swizzle・標準 XOR・src の B64 仮説）の
/// どれと一致するかを記録する。どれとも一致しなければダンプ全文を記録する。
pub fn classify_swizzle(spec: &TmaSpec, input: &[u32], out: &[u32]) -> Outcome {
    if let Some(m) = len_mismatch(spec, out.len()) {
        return m;
    }
    let dump = &out[HDR..];
    let elems = box_elems(spec, input, false);
    let fits = |phys: &dyn Fn(u32) -> u32| {
        elems
            .iter()
            .enumerate()
            .all(|(w, e)| dump.get(phys(w as u32) as usize).copied() == *e)
    };
    let mut classes = Vec::new();
    if fits(&|w| w) {
        classes.push("LINEAR");
    }
    if fits(&|w| swizzle_xor_phys_word(spec.swizzle, w)) {
        classes.push("XOR_ADDR_BITS");
    }
    if spec.swizzle == Swz::B64 && spec.box_cols == 16 && fits(&swizzle_src_b64_phys_word) {
        classes.push("SRC_B64_MODEL");
    }
    let class = if classes.is_empty() {
        "NONE".to_string()
    } else {
        classes.join("+")
    };
    // 線形と標準モデルが両方一致するのは swizzle が恒等になる配置のときのみ（記録のみ）。
    let tail = if classes.is_empty() {
        format!(" dump={}", hex_list(dump))
    } else {
        String::new()
    };
    Outcome::Record(format!("{} class={class}{tail}", header_tokens(out)))
}

/// `tma.store`: out[0] が完走の目印で、global の box 領域が smem の既知パターン
/// （`0xC0DE0000 | 添字`）と一致し、それ以外が初期の番兵のままか。out は
/// `[magic] ++ global 全語`。
pub fn check_store(spec: &TmaSpec, _input: &[u32], out: &[u32]) -> Outcome {
    let want_len = 1 + spec.global_words() as usize;
    if out.len() != want_len {
        return Outcome::Mismatch {
            first: out.len().min(want_len),
            count: out.len().abs_diff(want_len).max(1),
            detail: format!("length expected={want_len} got={}", out.len()),
        };
    }
    let global = &out[1..];
    let mut bad = 0usize;
    let mut first = None;
    for r in 0..spec.global_rows as i64 {
        for c in 0..spec.global_cols as i64 {
            let idx = (r * i64::from(spec.global_cols) + c) as usize;
            let (br, bc) = (r - i64::from(spec.cy), c - i64::from(spec.cx));
            let want = if (0..i64::from(spec.box_rows)).contains(&br)
                && (0..i64::from(spec.box_cols)).contains(&bc)
            {
                0xC0DE_0000u32 | (br * i64::from(spec.box_cols) + bc) as u32
            } else {
                SENTINEL
            };
            if global[idx] != want {
                bad += 1;
                first.get_or_insert(idx);
            }
        }
    }
    if out[0] == STORE_MAGIC && bad == 0 {
        Outcome::Match(format!("global_words={} bit_exact=true", global.len()))
    } else {
        Outcome::Mismatch {
            first: first.unwrap_or(0),
            count: bad.max(1),
            detail: format!("magic=0x{:08x} mismatched_global_words={bad}", out[0]),
        }
    }
}

/// `tma.prefetch`: 完走の目印のみ（prefetch の効果は観測できない）。
pub fn check_prefetch(_spec: &TmaSpec, _input: &[u32], out: &[u32]) -> Outcome {
    if out.len() == 1 && out[0] == PREFETCH_MAGIC {
        Outcome::Match("completion_magic=true".to_string())
    } else {
        Outcome::Mismatch {
            first: 0,
            count: 1,
            detail: format!("completion_magic expected=0x{PREFETCH_MAGIC:08x} got={out:08x?}"),
        }
    }
}

/// `tma.bulk_*`: 完走し（待ちが完了しなければカーネルは終了しない）、256 バイトが入力と一致するか。
/// out は `[polls, smem 64 語]`（ヘッダ 1 語）。
pub fn check_bulk(input: &[u32], out: &[u32]) -> Outcome {
    if out.len() != HDR + 64 || input.len() < 64 {
        return Outcome::Mismatch {
            first: 0,
            count: 1,
            detail: format!("length out={} in={}", out.len(), input.len()),
        };
    }
    let data = &out[HDR..];
    let bad: Vec<usize> = (0..64).filter(|&i| data[i] != input[i]).collect();
    let tokens = header_tokens(out);
    if bad.is_empty() {
        Outcome::Match(format!("{tokens} words=64 bit_exact=true"))
    } else {
        Outcome::Mismatch {
            first: bad[0],
            count: bad.len(),
            detail: format!("{tokens} mismatched_words={}", bad.len()),
        }
    }
}

/// `tma.multicast`: 2 CTA がそれぞれ完走し、どちらの smem も box が仮説どおりか。out は
/// CTA ごとのスロット（`HDR + dump_words` 語）を 2 つ連結したもの。
pub fn check_multicast(spec: &TmaSpec, input: &[u32], out: &[u32]) -> Outcome {
    let slot = HDR + spec.dump_words as usize;
    if out.len() != 2 * slot {
        return Outcome::Mismatch {
            first: out.len().min(2 * slot),
            count: out.len().abs_diff(2 * slot).max(1),
            detail: format!("length expected={} got={}", 2 * slot, out.len()),
        };
    }
    let elems = box_elems(spec, input, false);
    let mut details = Vec::new();
    let mut bad_total = 0usize;
    let mut first = None;
    for cta in 0..2 {
        let s = &out[cta * slot..(cta + 1) * slot];
        let bad = elems
            .iter()
            .zip(&s[HDR..])
            .filter(|(e, d)| **e != Some(**d))
            .count();
        bad_total += bad;
        if bad > 0 && first.is_none() {
            first = Some(cta * slot);
        }
        // 全トークンを k=v にする（CTA 接頭辞を付けたキー）。
        let prefixed: Vec<String> = header_tokens(s)
            .split_whitespace()
            .map(|t| format!("cta{cta}_{t}"))
            .collect();
        details.push(format!(
            "{} cta{cta}_mismatched_words={bad}",
            prefixed.join(" ")
        ));
    }
    if bad_total == 0 {
        Outcome::Match(details.join(" "))
    } else {
        Outcome::Mismatch {
            first: first.unwrap_or(0),
            count: bad_total,
            detail: details.join(" "),
        }
    }
}
