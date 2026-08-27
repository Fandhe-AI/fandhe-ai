//! PoC-v2-1（`docs/spec/03-poc/poc-v2-1-tensor-cpu-gemm/`）のテンソル型
//! 確定事項に対する productize 後 API の突合テスト（TASK-1.4d・#14）。
//!
//! `tensor-core` は PoC-v2-1 の `code/rust/src/tensor.rs`（`Vec<T>` 直接
//! 所有・`usize` strides）を「`Arc<Storage<T>>` + `offset` + `isize`
//! strides」構成へ productize したもの（`tensor.rs` 冒頭コメント・issue
//! #182 コメント 2026-08-05）。本ファイルは PoC の 5 テストベクタ
//! （row-major strides・実行時 shape 検査・NumPy ブロードキャスト規則・
//! broadcast add 期待値・非互換 shape 拒否）を公開 API 経由で再現し、
//! productize 前後の等価性を固定する。
//!
//! 契約: CI（self-hosted）は `docs/spec`（submodule）を checkout しない
//! （`.github/workflows/ci.yml` の Checkout ステップコメント）。このため
//! **本ファイルの非 `#[ignore]` テストは `docs/spec` 配下のいかなる
//! ファイルにも依存しない**（テストベクタは値としてこのファイルへ直接
//! 書き写す）。`docs/spec` 配下ファイル（`evidence/dump_case_512.bin`）を
//! 実際に読む数値突合テストのみ `#[ignore]` で分離し、submodule
//! checkout 済みのローカル環境でのみ実行する。

use fandhe_ai_tensor_core::{ShapeError, Tensor, broadcast_shape};

// --- PoC-v2-1 テストベクタ 1: row-major strides ---
// PoC: `strides_are_row_major`（tensor.rs:214）。
#[test]
fn poc_strides_are_row_major() {
    let t = Tensor::<f32>::new(vec![0.0; 24], &[2, 3, 4]).unwrap();
    // PoC は usize strides、現行は isize strides。値の等価性のみを確認する。
    assert_eq!(t.strides(), &[12isize, 4, 1]);
}

// --- PoC-v2-1 テストベクタ 2: 実行時 shape 検査 ---
// PoC: `shape_mismatch_is_detected_at_runtime`（tensor.rs:220、
// `ShapeError::DataLenMismatch`）。現行 API の対応 variant は
// `ShapeError::ElementCountMismatch`（`#[non_exhaustive]` のため
// `matches!` で照合する）。
#[test]
fn poc_shape_mismatch_is_detected_at_runtime() {
    let err = Tensor::<f32>::new(vec![0.0; 5], &[2, 3]).unwrap_err();
    assert!(matches!(
        err,
        ShapeError::ElementCountMismatch {
            expected: 6,
            actual: 5
        }
    ));
}

// --- PoC-v2-1 テストベクタ 3: NumPy 互換ブロードキャスト規則 ---
// PoC: `broadcast_shape_matches_numpy_rules`（tensor.rs:232）。
#[test]
fn poc_broadcast_shape_matches_numpy_rules() {
    assert_eq!(
        broadcast_shape(&[8, 1, 6], &[1, 5, 1]).unwrap(),
        vec![8, 5, 6]
    );
    assert_eq!(broadcast_shape(&[3, 4], &[4]).unwrap(), vec![3, 4]);
    assert!(broadcast_shape(&[3, 4], &[3]).is_err());
}

// --- PoC-v2-1 テストベクタ 4: broadcast add 期待値 ---
// PoC: `add_broadcasts_row_vector_over_matrix`（tensor.rs:242）。現行
// `tensor-core` は演算（`add`）を持たない（演算グラフ本体は後続タスク。
// `lib.rs` クレートコメント）ため、`broadcast_with` で両テンソルを
// 共通 shape [2,3] の view へ揃え、テスト内ヘルパの elementwise 加算
// （加算 1 回のみで丸め差なし・f32 完全一致）で PoC の `add` 相当を
// 再現する。stride 0 view がデータコピーなしで PoC と同じ結果を
// 与えることが本突合の本体。
#[test]
fn poc_add_broadcasts_row_vector_over_matrix() {
    let a = Tensor::<f32>::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let b = Tensor::<f32>::new(vec![10.0, 20.0, 30.0], &[3]).unwrap();
    let (ba, bb) = a.broadcast_with(&b).unwrap();
    assert_eq!(ba.shape(), &[2, 3]);
    assert_eq!(bb.shape(), &[2, 3]);

    let mut got = Vec::with_capacity(6);
    for i in 0..2 {
        for j in 0..3 {
            got.push(ba.get(&[i, j]).unwrap() + bb.get(&[i, j]).unwrap());
        }
    }
    assert_eq!(got, vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
}

// --- PoC-v2-1 テストベクタ 5: 非互換 shape 拒否 ---
// PoC: `add_rejects_incompatible_shapes`（tensor.rs:252）。現行 API は
// `broadcast_with` が addition 前の shape 整合を担う（ops_shape.rs 冒頭
// コメント参照）ため、ここで拒否を確認する。
#[test]
fn poc_add_rejects_incompatible_shapes() {
    let a = Tensor::<f32>::zeros(&[2, 3]).unwrap();
    let b = Tensor::<f32>::zeros(&[4]).unwrap();
    let err = a.broadcast_with(&b).unwrap_err();
    assert!(matches!(err, ShapeError::BroadcastIncompatible { .. }));
}

// --- PoC README「テンソル型の設計判断」確定事項の契約固定 ---

/// 行優先レイアウト: `from_slice` で構築したテンソルの `get()` 走査順が
/// 元スライスと一致することを確認する（PoC-v2-1 の行優先連続バッファ
/// 方針。`tensor.rs` 冒頭コメント参照）。
#[test]
fn row_major_traversal_matches_source_slice() {
    let src: Vec<f32> = (0..24).map(|v| v as f32).collect();
    let t = Tensor::<f32>::from_slice(&src, &[2, 3, 4]).unwrap();
    let mut i = 0;
    for a in 0..2 {
        for b in 0..3 {
            for c in 0..4 {
                assert_eq!(t.get(&[a, b, c]).unwrap(), src[i]);
                i += 1;
            }
        }
    }
}

/// 実行時 shape 検査方式（末尾軸比較・片方 1 で拡張）が `ops_shape` の
/// matmul／elementwise 出力 shape 計算でも PoC と同じ規則で成立する
/// ことを確認する。
#[test]
fn ops_shape_matmul_and_elementwise_match_poc_rules() {
    let out = fandhe_ai_tensor_core::matmul_out_shape(&[2, 3], &[3, 4]).unwrap();
    assert_eq!(out, vec![2, 4]);

    let out = fandhe_ai_tensor_core::elementwise_out_shape(&[2, 3], &[2, 3]).unwrap();
    assert_eq!(out, vec![2, 3]);
}

// --- dump_case_512.bin 数値突合（#[ignore]。docs/spec submodule 必須） ---

/// PoC-v2-1 `evidence/dump_case_512.bin` のヘッダ（m・n・k）を読み、
/// ファイル長との整合を検証してから本体（A・B・C の f32 LE 配列）を
/// 返す。パース時の境界検証を本体読み込みより先に行う（OWASP A03
/// 観点。`.claude/rules/security.md`「外部フォーマットパースは長さ・
/// 形状の検証を先に行う」）。
struct DumpCase {
    m: usize,
    n: usize,
    k: usize,
    a: Vec<f32>,
    b: Vec<f32>,
    c: Vec<f32>,
}

fn parse_dump_case(bytes: &[u8]) -> DumpCase {
    const HEADER_LEN: usize = 12;
    assert!(
        bytes.len() >= HEADER_LEN,
        "dump_case_512.bin: ヘッダ長（12 バイト）に満たない不正なファイル"
    );
    let m = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let n = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    let k = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;

    // 本体長を checked 演算で確定し、実ファイル長と突き合わせてから
    // 読み込む（不正・破損ファイルでの範囲外アクセスを未然に防ぐ）。
    let a_len = m
        .checked_mul(k)
        .expect("dump_case_512.bin: m*k がオーバーフローする不正なヘッダ");
    let b_len = k
        .checked_mul(n)
        .expect("dump_case_512.bin: k*n がオーバーフローする不正なヘッダ");
    let c_len = m
        .checked_mul(n)
        .expect("dump_case_512.bin: m*n がオーバーフローする不正なヘッダ");
    let total_f32 = a_len
        .checked_add(b_len)
        .and_then(|v| v.checked_add(c_len))
        .expect("dump_case_512.bin: 本体要素数がオーバーフローする不正なヘッダ");
    let body_bytes = total_f32
        .checked_mul(4)
        .expect("dump_case_512.bin: 本体バイト数がオーバーフローする不正なヘッダ");
    let expected_len = HEADER_LEN
        .checked_add(body_bytes)
        .expect("dump_case_512.bin: ファイル総バイト数がオーバーフローする不正なヘッダ");
    assert_eq!(
        bytes.len(),
        expected_len,
        "dump_case_512.bin: ヘッダ（m={m}, n={n}, k={k}）から期待される長さ \
         {expected_len} バイトと実ファイル長 {} バイトが一致しない",
        bytes.len()
    );

    let read_f32_le = |start: usize, len: usize| -> Vec<f32> {
        bytes[start..start + len * 4]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect()
    };
    let a_start = HEADER_LEN;
    let b_start = a_start + a_len * 4;
    let c_start = b_start + b_len * 4;
    let a = read_f32_le(a_start, a_len);
    let b = read_f32_le(b_start, b_len);
    let c = read_f32_le(c_start, c_len);

    DumpCase { m, n, k, a, b, c }
}

/// PoC-v2-1 の複合判定（相対誤差 1e-4 以内 または 絶対誤差 1e-5 以内。
/// `evidence/numeric_check.log`）をそのまま用いる。この許容誤差は
/// 単独緩和しない（`.claude/rules/coding-rust.md`）。
fn numeric_close(actual: f32, expected: f32) -> bool {
    const REL_TOL: f32 = 1e-4;
    const ABS_RESCUE: f32 = 1e-5;
    let abs_diff = (actual - expected).abs();
    if abs_diff <= ABS_RESCUE {
        return true;
    }
    let rel_err = abs_diff / expected.abs().max(f32::EPSILON);
    rel_err <= REL_TOL
}

#[test]
#[ignore = "docs/spec submodule の checkout が必要（CI は submodule 非取得）"]
fn dump_case_512_layout_round_trip_and_matmul_matches_pytorch_verified_output() {
    let path = format!(
        "{}/../../docs/spec/03-poc/poc-v2-1-tensor-cpu-gemm/evidence/dump_case_512.bin",
        env!("CARGO_MANIFEST_DIR")
    );
    let bytes = std::fs::read(&path).unwrap_or_else(|e| {
        panic!(
            "dump_case_512.bin を読めない（{path}）: {e}。\
             `git submodule update --init docs/spec` で submodule を \
             checkout してから再実行すること。"
        )
    });
    let case = parse_dump_case(&bytes);
    assert_eq!((case.m, case.n, case.k), (512, 512, 512));

    // 1. レイアウト round-trip: A を Tensor::from_slice で構築し、
    //    numel・strides・get() が生バッファ全数と一致することを確認する。
    let a = Tensor::<f32>::from_slice(&case.a, &[case.m, case.k]).unwrap();
    assert_eq!(a.numel(), case.m * case.k);
    assert_eq!(a.strides(), &[case.k as isize, 1]);
    for i in 0..case.m {
        for j in 0..case.k {
            assert_eq!(a.get(&[i, j]).unwrap(), case.a[i * case.k + j]);
        }
    }

    // 2. transpose -> contiguous -> transpose の round-trip が実データで
    //    元データと一致することを確認する。
    let at = a.transpose(0, 1).unwrap();
    let at_c = at.contiguous();
    let at_c_t = at_c.transpose(0, 1).unwrap();
    for i in 0..case.m {
        for j in 0..case.k {
            assert_eq!(a.get(&[i, j]).unwrap(), at_c_t.get(&[i, j]).unwrap());
        }
    }

    // 3. 参照 matmul（ikj 3 重ループ・f32::mul_add で FMA 契約を統一。
    //    `.claude/rules/coding-rust.md` の CPU 参照実装方針）で C の
    //    先頭 64x64 ブロック（全 K=512）を計算し、dump 内 C（PyTorch
    //    torch.matmul と PoC-v2-1 で複合判定 PASS 済み）と突合する。
    let b = Tensor::<f32>::from_slice(&case.b, &[case.k, case.n]).unwrap();
    const BLOCK: usize = 64;
    let mut fail_cells = 0usize;
    for i in 0..BLOCK {
        let mut row = vec![0.0f32; BLOCK];
        for p in 0..case.k {
            let a_ip = a.get(&[i, p]).unwrap();
            if a_ip == 0.0 {
                continue;
            }
            for (j, row_j) in row.iter_mut().enumerate() {
                let b_pj = b.get(&[p, j]).unwrap();
                *row_j = a_ip.mul_add(b_pj, *row_j);
            }
        }
        for (j, &row_j) in row.iter().enumerate() {
            let expected = case.c[i * case.n + j];
            if !numeric_close(row_j, expected) {
                fail_cells += 1;
            }
        }
    }
    assert_eq!(
        fail_cells, 0,
        "参照 matmul（先頭 64x64 ブロック）が dump 内 C と \
         複合判定（相対 1e-4 以内 または 絶対 1e-5 以内）で不一致"
    );
}
