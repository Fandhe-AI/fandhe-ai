//! TASK-7.1c（イシュー #75）: `transpose_2d()` 明示設計の検証・PoC-v2-6 数値突合。
//!
//! `tests/st_load.rs`（#73）は「`Tensor::get()` 経由の素朴なローカル forward」による
//! 突合に留まる。本ファイルはそのギャップを埋め、以下 4 点を production オペ
//! （`onnx_interop::ops::{gemm, relu, sigmoid}`）を用いて検証する（実装計画の 4 テスト群）。
//!
//! 1. 暗黙アダプタ不在の検証（キー集合完全一致・shape が `[out,in]` ネイティブのまま・
//!    転置省略時は `OpError::GemmDimMismatch` になる負のテスト）
//! 2. `transpose_2d()` の全 loaded weight（fc1/fc2/fc3、正方・単一行を含む縮退形状）横断の
//!    往復転置・値保存・`contiguous()` 実体化後の値保存
//! 3. 明示 `transpose_2d()` と ONNX `Gemm(transB=1)` の等価性（ビット一致）
//! 4. 受け入れ条件本体: production オペ経由の forward が PoC-v2-6 参照値と数値一致する
//!
//! ## 判定式についての注記（REQ-2 との混同禁止）
//!
//! 本ファイルのテスト 4 が用いる判定式は **REQ-7 事前固定基準**
//! `abs_err / (|ref| + 1e-6) <= 1e-3`（PoC-v2-6 `st_infer.rs`・v1 PoC-6・
//! `tests/st_load.rs` と同一）である。これは `.claude/rules/coding-rust.md` が定める
//! **REQ-2 のバックエンド間数値一致 OR 複合判定**（相対誤差 1e-3 未満 または絶対誤差
//! 1e-5 未満）とは**別指標**であり、本ファイルは REQ-7 側の基準のみを使う（両者を
//! 混同してどちらかを緩和しない）。
//!
//! fixture の出自・sha256 は `tests/fixtures/pytorch-reference/README.md` を参照（#73 と
//! 共有。本ファイルは fixture を変更しない）。CI（self-hosted）は `docs/spec`
//! （submodule）を checkout しないため、本ファイルは `docs/spec` 配下のいかなるファイルにも
//! 依存しない（`tests/st_load.rs` 冒頭コメントと同じ制約）。

use std::collections::{HashMap, HashSet};
use std::path::Path;

use onnx_interop::ops::{GemmAttrs, OpError, gemm, relu, sigmoid};
use onnx_interop::st_load::{load_safetensors_f32, require_keys};
use serde::Deserialize;
use tensor_core::Tensor;

const FIXTURE_DIR: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/pytorch-reference"
);

const WEIGHT_KEYS: [&str; 6] = [
    "fc1.weight",
    "fc1.bias",
    "fc2.weight",
    "fc2.bias",
    "fc3.weight",
    "fc3.bias",
];

#[derive(Deserialize)]
struct StReference {
    inputs: Vec<[f32; 2]>,
    outputs: Vec<f32>,
}

fn fixture_path(name: &str) -> std::path::PathBuf {
    Path::new(FIXTURE_DIR).join(name)
}

fn load_fixture_weights() -> HashMap<String, Tensor<f32>> {
    load_safetensors_f32(&fixture_path("model.safetensors")).expect("fixture のロードに失敗した")
}

fn load_weight_shapes() -> HashMap<String, Vec<usize>> {
    let shapes_json = std::fs::read_to_string(fixture_path("weight_shapes.json")).unwrap();
    serde_json::from_str(&shapes_json).unwrap()
}

// --- テスト群 1: 暗黙アダプタ不在の検証（REQ-7 契約 1） ---

#[test]
fn loader_preserves_pytorch_native_layout_and_exact_key_set() {
    let map = load_fixture_weights();

    // キー集合が PyTorch 側 6 キーと完全一致する（リネーム・追加・欠落なし）。
    // `require_keys` は不足のみを検査するため、余剰キーが無いことは別途 len 一致で確認する。
    require_keys(&map, &WEIGHT_KEYS).expect("6 キーすべてが揃っているはず");
    assert_eq!(
        map.len(),
        WEIGHT_KEYS.len(),
        "ローダーが余剰キーを追加している（暗黙アダプタが何らかのキーを生成した疑い）"
    );
    let expected_keys: HashSet<&str> = WEIGHT_KEYS.iter().copied().collect();
    let actual_keys: HashSet<&str> = map.keys().map(String::as_str).collect();
    assert_eq!(
        actual_keys, expected_keys,
        "ロード結果のキー集合が PyTorch 側と一致しない"
    );

    // 各 weight の shape が weight_shapes.json の [out,in] とそのまま一致する
    // （ローダーが転置を行っていない証拠。転置は呼び出し側の責務。REQ-7 契約 1）。
    let shapes = load_weight_shapes();
    for key in WEIGHT_KEYS {
        let expected = &shapes[key];
        let actual = map[key].shape();
        assert_eq!(
            actual,
            expected.as_slice(),
            "shape mismatch for key {key}（ローダーが暗黙に転置した疑い）"
        );
    }

    // 負のテスト: 転置なしの fc1.weight（[8,2]）を x[1,2] にそのまま渡すと
    // 内部次元不一致で拒否される（明示 transpose_2d() が省略不能であることの固定化。
    // 暗黙アダプタが存在しないため、呼び出し側が転置を忘れると必ずここで失敗する）。
    let x = Tensor::<f32>::new(vec![1.0, 2.0], &[1, 2]).unwrap();
    let fc1_w = &map["fc1.weight"]; // [8, 2]（転置していない）
    let err = gemm(&x, fc1_w, None, &GemmAttrs::default()).unwrap_err();
    assert!(
        matches!(err, OpError::GemmDimMismatch { .. }),
        "転置省略時に GemmDimMismatch 以外のエラーになった: {err:?}"
    );
}

// --- テスト群 2: transpose_2d() の全 loaded weight 横断検証 ---

#[test]
fn transpose_2d_roundtrip_on_all_loaded_weights() {
    let map = load_fixture_weights();

    // fc1: [8,2]（縦長・非正方）、fc2: [8,8]（正方）、fc3: [1,8]（単一行に縮退）の
    // 3 形状を横断して検証する（実装計画テスト群 2）。
    for key in ["fc1.weight", "fc2.weight", "fc3.weight"] {
        let w = &map[key];
        let (rows, cols) = (w.shape()[0], w.shape()[1]);

        // shape swap。
        let wt = w.transpose_2d().unwrap();
        assert_eq!(wt.shape(), &[cols, rows], "shape swap failed for {key}");

        // get() による全要素の値対応。
        for i in 0..rows {
            for j in 0..cols {
                assert_eq!(
                    wt.get(&[j, i]).unwrap(),
                    w.get(&[i, j]).unwrap(),
                    "transpose_2d が値の対応を保っていない for {key} (i={i}, j={j})"
                );
            }
        }

        // 二重転置は元とビット一致（f32 完全等価。往復転置は値の再計算を伴わず
        // strides の入れ替えのみのため、丸め誤差の混入余地がない）。
        let wtt = wt.transpose_2d().unwrap();
        assert_eq!(
            wtt.shape(),
            w.shape(),
            "round-trip shape mismatch for {key}"
        );
        for i in 0..rows {
            for j in 0..cols {
                assert_eq!(
                    wtt.get(&[i, j]).unwrap(),
                    w.get(&[i, j]).unwrap(),
                    "round-trip transpose がビット一致していない for {key} (i={i}, j={j})"
                );
            }
        }

        // 転置 view を contiguous() で実体化した後も値が保存される
        // （`ops::gemm` 内部の trans_b 経路が通るのと同じ contiguous() 実体化を検証する）。
        let wt_c = wt.contiguous();
        assert_eq!(wt_c.shape(), &[cols, rows]);
        for i in 0..cols {
            for j in 0..rows {
                assert_eq!(
                    wt_c.get(&[i, j]).unwrap(),
                    wt.get(&[i, j]).unwrap(),
                    "contiguous() 実体化が値を破壊した for {key} (i={i}, j={j})"
                );
            }
        }
    }
}

// --- テスト群 3: 明示 transpose_2d() と ONNX Gemm(transB=1) の等価性 ---

#[test]
fn explicit_transpose_matches_gemm_trans_b_bitwise() {
    let map = load_fixture_weights();
    let shapes = load_weight_shapes();

    // 各 weight について、x を weight の in_features 次元に合わせたダミー入力とし、
    // (a) 明示 transpose_2d() 済みの weight を trans_b なしで渡す経路と
    // (b) 転置前の weight を trans_b=true で渡す経路が、全要素ビット一致することを確認する。
    // `transpose_2d()` は `transpose(0,1)` への薄い委譲（tensor-core::tensor.rs）であり、
    // `gemm` 内部で trans_b=true も同じ `transpose(0,1)` を適用してから `contiguous()` する
    // （gemm.rs 参照）。両経路は最終的に同一の `transpose(0,1).contiguous()` 呼び出し列へ
    // 還元されるため、本テストは数値的な偶然の一致ではなく「明示転置と ONNX transB 属性が
    // 同じ内部変換に帰着する」という委譲契約を固定化するものである（FMA 契約の検証ではない。
    // FMA 契約自体は `gemm.rs` の `mul_add` 使用そのものが担保する）。
    for key in ["fc1.weight", "fc2.weight", "fc3.weight"] {
        let w = &map[key]; // [out, in]
        let out_dim = shapes[key][0];
        let in_dim = shapes[key][1];

        // 決定的なダミー入力（batch=1 x in_dim）。値そのものはビット一致検証には
        // 影響しないが、0 のみだと mul_add の丸め差異が検出できないため小数を混ぜる。
        let x_data: Vec<f32> = (0..in_dim).map(|i| 0.1 + i as f32 * 0.37).collect();
        let x = Tensor::<f32>::new(x_data, &[1, in_dim]).unwrap();

        let w_explicit_t = w.transpose_2d().unwrap();
        let y_explicit = gemm(&x, &w_explicit_t, None, &GemmAttrs::default()).unwrap();

        let attrs_trans_b = GemmAttrs {
            trans_b: true,
            ..GemmAttrs::default()
        };
        let y_trans_b = gemm(&x, w, None, &attrs_trans_b).unwrap();

        assert_eq!(y_explicit.shape(), &[1, out_dim]);
        assert_eq!(y_explicit.shape(), y_trans_b.shape());
        for j in 0..out_dim {
            assert_eq!(
                y_explicit.get(&[0, j]).unwrap(),
                y_trans_b.get(&[0, j]).unwrap(),
                "明示 transpose_2d() と Gemm(transB=1) がビット一致しない for {key} (j={j})"
            );
        }
    }
}

// --- テスト群 4（受け入れ条件本体）: production オペ経由 forward の PoC-v2-6 数値突合 ---
//
// PoC-v2-6 `Mlp::forward`（2->8->8->1, ReLU x2 + Sigmoid）と同一のレイヤー順・活性化配置を
// production オペ（gemm/relu/sigmoid）で再現する。bias は Gemm の `c` 引数として渡す
// （[out_dim] 形状は Gemm の出力 shape [batch, out_dim] へ NumPy 互換ブロードキャストされる）。

fn linear_via_gemm(
    x: &Tensor<f32>,
    w: &Tensor<f32>, // [out, in]（safetensors ロード直後のネイティブレイアウト）
    b: &Tensor<f32>, // [out]
) -> Tensor<f32> {
    // REQ-7 契約: ローダーは転置しないため、呼び出し側（本関数）が明示的に
    // transpose_2d() で [out,in] -> [in,out] へ変換してから Gemm に渡す。
    let w_t = w.transpose_2d().unwrap();
    gemm(x, &w_t, Some(b), &GemmAttrs::default()).unwrap()
}

fn mlp_forward_via_ops(map: &HashMap<String, Tensor<f32>>, input: &[f32; 2]) -> f32 {
    let x = Tensor::<f32>::new(input.to_vec(), &[1, 2]).unwrap();

    let h1 = relu(&linear_via_gemm(&x, &map["fc1.weight"], &map["fc1.bias"])).unwrap();
    let h2 = relu(&linear_via_gemm(&h1, &map["fc2.weight"], &map["fc2.bias"])).unwrap();
    let out = sigmoid(&linear_via_gemm(&h2, &map["fc3.weight"], &map["fc3.bias"])).unwrap();

    out.get(&[0, 0]).unwrap()
}

#[test]
fn ops_forward_matches_pytorch_reference_within_req7_tolerance() {
    let map = load_fixture_weights();
    require_keys(&map, &WEIGHT_KEYS).expect("6 キーすべてが揃っているはず");

    let reference_json = std::fs::read_to_string(fixture_path("st_reference.json")).unwrap();
    let reference: StReference = serde_json::from_str(&reference_json).unwrap();
    assert_eq!(reference.inputs.len(), reference.outputs.len());
    assert!(
        !reference.inputs.is_empty(),
        "fixture が空では突合にならない"
    );

    let mut max_rel_err = 0.0f32;
    for (input, &expected) in reference.inputs.iter().zip(reference.outputs.iter()) {
        let actual = mlp_forward_via_ops(&map, input);
        let abs_err = (actual - expected).abs();
        // REQ-7 事前固定判定式（本ファイル冒頭 `//!` 参照。REQ-2 の OR 複合判定とは別指標）。
        let rel_err = abs_err / (expected.abs() + 1e-6);
        max_rel_err = max_rel_err.max(rel_err);
        assert!(
            rel_err <= 1e-3,
            "REQ-7 数値一致基準を超過: input={input:?} expected={expected} actual={actual} \
             abs_err={abs_err} rel_err={rel_err}"
        );
    }
    // PoC-v2-6 実測（evidence/st_numeric_check.log）は全件で最大相対誤差 0.000000（閾値 1e-3）。
    // ループ内 assert が全件 <= 1e-3 を既に保証しているため、ここでは診断用に出力するのみで
    // 追加の assert は行わない（ループを抜けた時点で不成立はあり得ず、二重の assert は
    // 到達不能コードになるため）。
    eprintln!("max_rel_err={max_rel_err} (threshold=1e-3)");
}
