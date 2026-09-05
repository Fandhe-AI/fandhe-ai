//! TASK-11.3（#70）: 行列演算ユニット活用（REQ-11）の証跡として、
//! `gemm.metal` に Apple GPU の行列専用命令（`simdgroup_matrix` API）が
//! 実在することを機械検査する。
//!
//! CUDA 側 `backend-cuda` の `kernels_wmma.rs`／`kernels_wmma_opt.rs`／
//! `kernels_mma.rs` 末尾 `#[cfg(test)]` にある「tensor core 命令実在検査」
//! と対になる証跡であり、`docs/matrix-unit-dispatch.md` の命令実在一覧表が
//! 参照する（同ドキュメントは主張のみを記述し、その主張の機械検証は
//! ここで担う。二重管理を避けるため実測値・命令リストの重複は書かない）。
//!
//! `include_str!` によるビルド時文字列埋め込みへの contains 検査のみで
//! 完結するため、Metal 実機・`cfg(target_os = "macos")` を必要とせず
//! Linux CI（self-hosted）上でも green になる。既存の `tile.rs`（本クレート
//! 同様の Linux 実行可テスト）と同じ位置づり。

/// `crates/backend-metal/src/shaders/gemm.metal` のソース全文。
/// このファイル自体が実行対象ではなく、文字列としての内容検査のみに使う。
const GEMM_METAL_SOURCE: &str = include_str!("../src/shaders/gemm.metal");

/// REQ-11 の証跡: `gemm_simdgroup`／`gemm_simdgroup_tiled` カーネルが
/// Apple GPU の行列専用命令（`simdgroup_float8x8` 型・
/// `simdgroup_load`／`simdgroup_multiply_accumulate`／`simdgroup_store`
/// 関数）を実際に使用していることをロックする。将来の書き換えで
/// これらの命令が誤って除去された場合に検出する（CUDA 側の
/// `wmma_f16_source_uses_wmma_instructions` と同方針）。
#[test]
fn gemm_metal_source_uses_simdgroup_matrix_instructions() {
    for needle in [
        "#include <metal_simdgroup_matrix>",
        "simdgroup_float8x8",
        "simdgroup_load",
        "simdgroup_multiply_accumulate",
        "simdgroup_store",
    ] {
        assert!(
            GEMM_METAL_SOURCE.contains(needle),
            "gemm.metal に行列演算ユニット命令 `{needle}` が見つかりません"
        );
    }
}

/// `gemm_simdgroup_f16` の定義域（`MM_T`/`ACC_T` の `typedef simdgroup_half8x8`
/// 開始位置から、次に定義される `gemm_simdgroup_tiled` 開始位置まで）を
/// 切り出す。
///
/// 開始位置を `kernel void gemm_simdgroup_f16(` ではなく直前の `typedef` 行に
/// 取る理由: `MM_T`（A・B のタイル型）の実際の行列型 `simdgroup_half8x8` は
/// カーネル本体内ではなくこの typedef 行にのみ現れる（本体は型エイリアス
/// `MM_T`/`ACC_T` を使う）。`typedef` 行はこの f16 カーネル専用（f32 版
/// `gemm_simdgroup` は typedef より前で既に完結している）であり、かつ f16
/// カーネル本体全体を包含するため、命令実在検査・REQ-8 境界チェック検査の
/// 両方に安全に使える。
///
/// なお `ACC_T`（アキュムレータ型）はイシュー #380 で `simdgroup_float8x8`
/// （f32 累算）に変更済みだが、`simdgroup_half8x8` typedef 自体（`MM_T` の
/// 定義）はこの範囲の開始アンカーとして変更後も維持される（本ファイル末尾
/// `gemm_simdgroup_f16_source_uses_simdgroup_half_matrix_instructions` が
/// この typedef 行の存在に依存する）。
///
/// 検索対象を `GEMM_METAL_SOURCE` 全文のままにすると、`gemm_simdgroup`
/// （f32 版。L154〜）が同じ `simdgroup_load`/`simdgroup_multiply_accumulate`/
/// `simdgroup_store` 命令を先に使っているため、f16 カーネル側からこれらの
/// 命令を誤って取り除いてもテストが green のまま通過する偽陽性が生じる
/// （PR #346 Bugbot 指摘）。
fn gemm_simdgroup_f16_kernel_body() -> &'static str {
    let kernel_start = GEMM_METAL_SOURCE
        .find("typedef simdgroup_half8x8 MM_T;")
        .expect("gemm_simdgroup_f16 の MM_T typedef が見つかりません");
    let next_kernel_start = GEMM_METAL_SOURCE[kernel_start..]
        .find("kernel void gemm_simdgroup_tiled(")
        .map(|offset| kernel_start + offset)
        .expect("gemm_simdgroup_tiled カーネル本体が見つかりません（次カーネル境界の特定に失敗）");
    &GEMM_METAL_SOURCE[kernel_start..next_kernel_start]
}

/// REQ-11・TASK-8.3b（#156）の証跡: `gemm_simdgroup_f16` が半精度
/// simdgroup 行列型（`simdgroup_half8x8`）と行列演算ユニット命令
/// （`simdgroup_load`/`simdgroup_multiply_accumulate`/`simdgroup_store`）を
/// 実際に使用していることをロックする（CUDA 側
/// `wmma_f16_source_uses_wmma_instructions` と同方針。
/// `docs/matrix-unit-dispatch.md` の命令実在一覧表が参照する）。
///
/// 検索範囲は `gemm_simdgroup_f16_kernel_body` で f16 カーネル本体に限定する
/// （全文検索だと f32 版 `gemm_simdgroup` の同一命令に一致してしまう偽陽性を防ぐ。
/// PR #346 Bugbot 指摘）。
#[test]
fn gemm_simdgroup_f16_source_uses_simdgroup_half_matrix_instructions() {
    let kernel_body = gemm_simdgroup_f16_kernel_body();
    for needle in [
        "simdgroup_half8x8",
        "simdgroup_load",
        "simdgroup_multiply_accumulate",
        "simdgroup_store",
    ] {
        assert!(
            kernel_body.contains(needle),
            "gemm.metal の gemm_simdgroup_f16 に `{needle}` が見つかりません"
        );
    }
}

/// イシュー #380 の証跡: `gemm_simdgroup_f16` のアキュムレータ型 `ACC_T` が
/// `simdgroup_float8x8`（f32 累算）で定義されていることを CI（Linux
/// self-hosted）上でロックする。
///
/// 背景: `gemm_simdgroup_f16` の実機パリティテスト
/// （`crates/backend-metal/tests/gemm_simdgroup_parity.rs` 等）は Metal
/// 実機依存のため `#[ignore]` で分離されており通常 CI では実行されない
/// （本ファイル冒頭のとおり Linux CI では `include_str!` した文字列の
/// 検査のみで完結する）。
/// そのため #380 で導入した `ACC_T = simdgroup_float8x8` への変更
/// （`MM_T`＝`simdgroup_half8x8` はロード型のまま据え置き、累算のみ f32 化）
/// を、`typedef` 行の文字列としてここで固定しないと、将来
/// `simdgroup_half8x8` へ差し戻す退行が発生しても通常 CI は green のまま
/// 通過してしまい、実機 `#[ignore]` テストでのみ検出される弱い保護しか
/// 残らない（PR #434 Bugbot 指摘 review_id 4891802188 への対応）。
#[test]
fn gemm_simdgroup_f16_source_uses_f32_accumulator_type() {
    let kernel_body = gemm_simdgroup_f16_kernel_body();
    for needle in [
        "typedef simdgroup_half8x8 MM_T;",
        "typedef simdgroup_float8x8 ACC_T;",
    ] {
        assert!(
            kernel_body.contains(needle),
            "gemm.metal の gemm_simdgroup_f16 に `{needle}` が見つかりません（f32 累算への変更が失われていないか確認する）"
        );
    }
}

/// REQ-8 の証跡: `gemm_simdgroup_f16` が手動境界チェック
/// （タイル原点が実効次元を超える場合の早期 return）を維持していることを
/// ロックする（`gemm_metal_source_uses_simdgroup_matrix_instructions` と
/// 対になる検査。性能上の下限・最適化の達成を理由に境界チェックを省略
/// しない方針 `.claude/rules/coding-rust.md` の機械検証）。
///
/// カーネル本体の切り出しは `gemm_simdgroup_f16` 開始位置から EOF までではなく、
/// 次に定義される `gemm_simdgroup_tiled` の開始位置までに限定する（
/// `gemm_simdgroup_f16_kernel_body` を共用）。EOF までスライスすると後続
/// カーネル内の同一境界ガード文字列にもマッチしてしまい、f16 カーネル側の
/// 境界チェックを誤って削除しても本テストが green のまま通過する偽陽性が
/// 生じるため（PR #346 Bugbot 指摘）。
#[test]
fn gemm_simdgroup_f16_source_retains_req8_boundary_guard() {
    let kernel_body = gemm_simdgroup_f16_kernel_body();
    assert!(
        kernel_body.contains("if (row0 >= dims.m || col0 >= dims.n)"),
        "gemm_simdgroup_f16 に REQ-8 手動境界チェックが見つかりません"
    );
}

/// `gemm_simdgroup_tiled` カーネル本体（`kernel void gemm_simdgroup_tiled(`
/// 開始位置から、次に定義される `gemm_simdgroup_tiled_f16`
/// （イシュー #796）開始位置までを切り出す。
///
/// **イシュー #796 で境界の取り方を変更**: 本ファイル追加当初は
/// `gemm_simdgroup_tiled` がファイル内最後のカーネルだったため EOF までの
/// スライスで安全だったが、#796 で末尾に `gemm_simdgroup_tiled_f16`
/// （f32 版と同じ蛇行走査式・`bool group_in_bounds =`・
/// `simdgroup_float8x8 a_frag[MAX_ACC];` 相当の文言をコメントに含む）を
/// 追加したため、EOF までのスライスのままでは以下の既存 exact-occurrence
/// 検査（`gemm_simdgroup_tiled_source_uses_serpentine_scan_order` 等）が
/// f16 カーネル側の記述を誤って算入し偽陽性・偽陰性を招く
/// （`gemm_simdgroup_f16_kernel_body` が f32 版 `gemm_simdgroup` との混同を
/// 避けるために境界を絞っているのと同じ理由。PR #346 Bugbot 指摘の系譜）。
fn gemm_simdgroup_tiled_kernel_body() -> &'static str {
    let kernel_start = GEMM_METAL_SOURCE
        .find("kernel void gemm_simdgroup_tiled(")
        .expect("gemm_simdgroup_tiled カーネル本体が見つかりません");
    let next_kernel_start = GEMM_METAL_SOURCE[kernel_start..]
        .find("kernel void gemm_simdgroup_tiled_f16(")
        .map(|offset| kernel_start + offset)
        .expect(
            "gemm_simdgroup_tiled_f16 カーネル本体が見つかりません（次カーネル境界の特定に失敗）",
        );
    &GEMM_METAL_SOURCE[kernel_start..next_kernel_start]
}

/// `gemm_simdgroup_tiled_f16` カーネル本体（`kernel void
/// gemm_simdgroup_tiled_f16(` 開始位置から EOF まで）を切り出す
/// （イシュー #796）。本ファイル内で最後に定義されるカーネルのため EOF
/// までのスライスで安全。
fn gemm_simdgroup_tiled_f16_kernel_body() -> &'static str {
    let kernel_start = GEMM_METAL_SOURCE
        .find("kernel void gemm_simdgroup_tiled_f16(")
        .expect("gemm_simdgroup_tiled_f16 カーネル本体が見つかりません");
    &GEMM_METAL_SOURCE[kernel_start..]
}

/// REQ-11・イシュー #796 の証跡: `gemm_simdgroup_tiled_f16` が半精度
/// simdgroup 行列型（`MM_T`＝`simdgroup_half8x8`）・f32 累算
/// （`ACC_T`＝`simdgroup_float8x8`）・行列演算ユニット命令
/// （`simdgroup_load`/`simdgroup_multiply_accumulate`/`simdgroup_store`）を
/// 実際に使用していることをロックする（`gemm_simdgroup_f16_source_uses_*`
/// と対になる検査。`docs/matrix-unit-dispatch.md` の命令実在一覧表が
/// 参照する）。
#[test]
fn gemm_simdgroup_tiled_f16_source_uses_matrix_unit_instructions() {
    let kernel_body = gemm_simdgroup_tiled_f16_kernel_body();
    for needle in [
        "device const half* a",
        "device half* c",
        "MM_T a_frag[MAX_ACC];",
        "MM_T b_frag[MAX_ACC];",
        "ACC_T acc[MAX_ACC][MAX_ACC];",
        "simdgroup_load",
        "simdgroup_multiply_accumulate",
        "simdgroup_store",
    ] {
        assert!(
            kernel_body.contains(needle),
            "gemm.metal の gemm_simdgroup_tiled_f16 に `{needle}` が見つかりません"
        );
    }
}

/// REQ-8 の証跡（イシュー #796・協調ロードのベクトル化はイシュー #797・
/// 境界判定式の f32/f16 共通化はイシュー #1038）: `gemm_simdgroup_tiled_f16`
/// が手動境界チェック（ブロック原点の早期 return・協調ロードのグループ
/// 単位 in-bounds 判定＋要素単位スカラーフォールバック・エピローグ書き
/// 戻しの要素単位 in-bounds 判定）を維持していることをロックする
/// （`gemm_simdgroup_tiled_source_retains_*` と同種の検査。性能上の下限・
/// 最適化の達成を理由に境界チェックを省略しない方針
/// `.claude/rules/coding-rust.md` の機械検証）。
///
/// #1038 でブロック原点・グループ単位 in-bounds・要素単位フォールバックの
/// 各判定式は `gemm_simdgroup_tiled`（f32）と共有する述語関数
/// （`tiled_block_out_of_range`/`tiled_a_group_in_bounds`/
/// `tiled_b_group_in_bounds`/`tiled_a_elem_in_bounds`/
/// `tiled_b_elem_in_bounds`。ファイル冒頭 `Dims` 定義直後）へ抽出済みの
/// ため、ここではブール式そのものではなく**呼び出し**（ベクトル幅引数
/// `8` を含む）が本カーネル本体に実在することを検査する（抽出前は
/// ブール式自体がここに現れていたため、需要が消えたのではなく検査対象の
/// 形が変わった点に注意）。エピローグ（タイル粒度統合。#797）は f32 版の
/// タイル原点 `continue` 構造と異なり要素単位 `continue` を経ない即時判定
/// のため共有述語化していない（設計判断は `gemm.metal` 冒頭ヘルパ群
/// コメント参照）。
#[test]
fn gemm_simdgroup_tiled_f16_source_retains_req8_boundary_guards() {
    let kernel_body = gemm_simdgroup_tiled_f16_kernel_body();
    for needle in [
        "if (tiled_block_out_of_range(row0, col0, dims))",
        "tiled_a_group_in_bounds(kk, bk_eff, global_row, global_k, 8, dims)",
        "tiled_b_group_in_bounds(kk, bk_eff, global_k, global_col, 8, dims)",
        "tiled_a_elem_in_bounds(kk_e, bk_eff, global_row, global_k_e, dims)",
        "tiled_b_elem_in_bounds(kk, bk_eff, global_k, global_col_e, dims)",
        "if (out_row < dims.m && out_col < dims.n)",
    ] {
        assert!(
            kernel_body.contains(needle),
            "gemm_simdgroup_tiled_f16 に REQ-8 手動境界チェック `{needle}` が見つかりません"
        );
    }
}

/// イシュー #1038 の証跡: `gemm_simdgroup_tiled`（f32）・
/// `gemm_simdgroup_tiled_f16` の 2 系統が、共有の境界検査述語関数
/// （ファイル冒頭 `Dims` 定義直後。両カーネルより前方のため、いずれの
/// `_kernel_body()` スライスにも定義自体は含まれない）を**両方から**
/// 呼び出していることをロックする。片側のカーネルだけが共通化されて
/// もう片側が独自のインライン式へ後退する退行（境界検査ドリフト）を
/// 検出する目的（イシュー #1038 計画「3.2 節」）。
#[test]
fn gemm_simdgroup_tiled_variants_share_boundary_check_helpers() {
    for helper_def in [
        "inline bool tiled_block_out_of_range(",
        "inline bool tiled_a_group_in_bounds(",
        "inline bool tiled_b_group_in_bounds(",
        "inline bool tiled_a_elem_in_bounds(",
        "inline bool tiled_b_elem_in_bounds(",
    ] {
        assert!(
            GEMM_METAL_SOURCE.contains(helper_def),
            "gemm.metal に共通境界検査ヘルパ `{helper_def}` の定義が見つかりません（イシュー #1038）"
        );
    }

    let f32_body = gemm_simdgroup_tiled_kernel_body();
    let f16_body = gemm_simdgroup_tiled_f16_kernel_body();
    for (call, vec_w_f32, vec_w_f16) in [
        ("tiled_block_out_of_range(row0, col0, dims)", "", ""),
        (
            "tiled_a_group_in_bounds(kk, bk_eff, global_row, global_k, ",
            "4, dims)",
            "8, dims)",
        ),
        (
            "tiled_b_group_in_bounds(kk, bk_eff, global_k, global_col, ",
            "4, dims)",
            "8, dims)",
        ),
        (
            "tiled_a_elem_in_bounds(kk_e, bk_eff, global_row, global_k_e, dims)",
            "",
            "",
        ),
        (
            "tiled_b_elem_in_bounds(kk, bk_eff, global_k, global_col_e, dims)",
            "",
            "",
        ),
    ] {
        let f32_needle = format!("{call}{vec_w_f32}");
        let f16_needle = format!("{call}{vec_w_f16}");
        assert!(
            f32_body.contains(&f32_needle),
            "gemm_simdgroup_tiled（f32）が共通ヘルパ呼び出し `{f32_needle}` を含んでいません（境界検査ドリフトの疑い。イシュー #1038）"
        );
        assert!(
            f16_body.contains(&f16_needle),
            "gemm_simdgroup_tiled_f16 が共通ヘルパ呼び出し `{f16_needle}` を含んでいません（境界検査ドリフトの疑い。イシュー #1038）"
        );
    }
}

/// イシュー #1038 codex-review 指摘への対応
/// （PR #1074 レビュー r3888410843）: 上記
/// `gemm_simdgroup_tiled_variants_share_boundary_check_helpers` はヘルパの
/// **呼び出し**（シグネチャ・定義行の存在）のみを検査しており、ヘルパ
/// **本体**の境界条件式（例: `kk + 8 <= bk_eff`／`global_k + 8 <= dims.k`
/// 相当）が空実装・`return true;` 等へ後退しても検出できない。
///
/// さらに PR #1074 レビュー（codex-review P2）指摘: 当初の実装は各条件式を
/// `body.contains(condition)` で個別に部分一致検査していたため、条件式
/// 同士を結ぶ論理演算子（`&&` → `||` 等）が弱体化されても検出できなかった
/// （例: `tiled_a_group_in_bounds` が `return cond1 || cond2 || cond3;` へ
/// 書き換えられても、3 つの `contains` はいずれも真のまま通ってしまう）。
/// 本テストは空白を正規化した **完全な `return` 式**を期待値と厳密一致
/// 検査することで、条件式の脱落だけでなく結合演算子の弱体化・条件の
/// 順序入れ替えも検出する。加えて、A/B group・elem 系ヘルパ（4 個）は
/// `&&` のみで結合され `||` を含まないこと、block ヘルパ（1 個）は逆に
/// `||` のみで結合され `&&` を含まないことを個別に固定し、`contains`
/// 方式では見逃す論理演算子の入れ替えを構文レベルで遮断する。
///
/// 5 個のヘルパそれぞれの定義本体（シグネチャ直後の `{` から対応する `}`
/// まで。ネストした波括弧を持たない単純な `return ...;` 一文のみのため、
/// 最初に現れる `}` を対応する閉じ括弧として素朴に切り出せる）を個別に
/// 抽出し、REQ-8 の境界条件式そのものがヘルパ本体に実在することを検査
/// する（Metal 実機非依存・ホスト側で実行可能。`#[ignore]` の数値回帰
/// テストとは独立に通常 CI で退行を検出する）。
#[test]
fn gemm_metal_boundary_helpers_retain_req8_condition_expressions() {
    /// `signature` の直後に現れる `{` から、対応する `}` までの本体
    /// （中括弧含む）を切り出す。5 ヘルパはいずれも単純な `return` 一文
    /// のみでネストした波括弧を持たないため、最初の `{`/`}` ペアで
    /// 本体全体を過不足なく取得できる。
    fn extract_helper_body(source: &str, signature: &str) -> String {
        let sig_pos = source.find(signature).unwrap_or_else(|| {
            panic!("gemm.metal にヘルパシグネチャ `{signature}` が見つかりません")
        });
        let after_sig = &source[sig_pos..];
        let open = after_sig
            .find('{')
            .unwrap_or_else(|| panic!("ヘルパ `{signature}` の本体開始 `{{` が見つかりません"));
        let close = after_sig[open..]
            .find('}')
            .unwrap_or_else(|| panic!("ヘルパ `{signature}` の本体終端 `}}` が見つかりません"));
        after_sig[open..=open + close].to_string()
    }

    /// 本体中の連続空白（改行・インデント含む）を単一の半角スペースへ
    /// 正規化する。ソース側のフォーマット（改行位置・インデント幅）が
    /// 変わっても意味的に同一な `return` 式を同一文字列として比較できる
    /// ようにするため。
    fn normalize_whitespace(body: &str) -> String {
        body.split_whitespace().collect::<Vec<_>>().join(" ")
    }

    // (シグネチャ, 完全な return 式（正規化後の期待値）, 結合演算子)
    for (signature, expected_return, connective) in [
        (
            "inline bool tiled_block_out_of_range(",
            "return row0 >= dims.m || col0 >= dims.n;",
            "||",
        ),
        (
            "inline bool tiled_a_group_in_bounds(",
            "return (kk + vec_w <= bk_eff) && (global_row < dims.m) && (global_k + vec_w <= dims.k);",
            "&&",
        ),
        (
            "inline bool tiled_b_group_in_bounds(",
            "return (kk < bk_eff) && (global_k < dims.k) && (global_col + vec_w <= dims.n);",
            "&&",
        ),
        (
            "inline bool tiled_a_elem_in_bounds(",
            "return kk_e < bk_eff && global_row < dims.m && global_k_e < dims.k;",
            "&&",
        ),
        (
            "inline bool tiled_b_elem_in_bounds(",
            "return kk < bk_eff && global_k < dims.k && global_col_e < dims.n;",
            "&&",
        ),
    ] {
        let raw_body = extract_helper_body(GEMM_METAL_SOURCE, signature);
        let body = normalize_whitespace(&raw_body);

        // 空白正規化した本体全体が期待する `return` 式を含むことを厳密
        // 一致で検査する（部分式ごとの `contains` では検出できない結合
        // 演算子の弱体化・条件の脱落・順序入れ替えをまとめて検出する）。
        assert!(
            body.contains(expected_return),
            "ヘルパ `{signature}` の本体が期待する REQ-8 境界条件式 `{expected_return}` と一致しません（正規化後の本体: `{body}`）"
        );

        // 結合演算子そのものの入れ替え（`&&` → `||` 等）を構文レベルで
        // 固定する。上の完全一致検査だけでも検出できるが、意図（A/B
        // group・elem ヘルパは AND 結合、block ヘルパは OR 結合）を
        // 明示的に読み取れるよう独立した検査としても残す。
        let (required, forbidden) = if connective == "&&" {
            ("&&", "||")
        } else {
            ("||", "&&")
        };
        assert!(
            body.contains(required),
            "ヘルパ `{signature}` の本体に結合演算子 `{required}` が見つかりません（本体: `{body}`）"
        );
        assert!(
            !body.contains(forbidden),
            "ヘルパ `{signature}` の本体に本来含まれないはずの結合演算子 `{forbidden}` が含まれています（結合演算子の弱体化の疑い。本体: `{body}`）"
        );
    }
}

/// イシュー #797 の証跡: `gemm_simdgroup_tiled_f16` の協調ロードが
/// half 8 要素（128bit）幅のベクトルロードへ移行済みであることをロックする
/// （#796 時点はスカラーロードに留まっていた。上記
/// `gemm_simdgroup_tiled_f16_source_retains_req8_boundary_guards` の
/// group_in_bounds 判定と対になる証跡）。`reinterpret_cast<device const
/// float4*>`（half8 を 128bit ビットコピーする device 側読み出し）と、
/// threadgroup 側の 8 バイト境界 half4 分割 store（`as_type<half4>`）の
/// 両方が現れることを確認する。
#[test]
fn gemm_simdgroup_tiled_f16_source_uses_vectorized_staged_load() {
    let kernel_body = gemm_simdgroup_tiled_f16_kernel_body();
    for needle in ["reinterpret_cast<device const float4*>", "as_type<half4>"] {
        assert!(
            kernel_body.contains(needle),
            "gemm_simdgroup_tiled_f16 にベクトルロードの証跡 `{needle}` が見つかりません（イシュー #797 の実装漏れ）"
        );
    }
}

/// イシュー #797 の証跡: `gemm_simdgroup_tiled_f16` のエピローグが
/// サブタイル全体単位（`sub_bm*sub_bn`）へ統合され、`simdgroup_barrier` が
/// 1 simdgroup あたり 1 回だけ発生することをロックする（#796 時点は 8x8
/// acc タイル毎に store→barrier→書き戻し→barrier の 2 回、
/// `acc_rows*acc_cols` 個のタイル分＝ `2*acc_rows*acc_cols` 回発生していた）。
/// カーネル本体全体（すべて 1 個の staging スラブを扱うエピローグ節に
/// 属する）に現れる `simdgroup_barrier` の出現回数を数える。
#[test]
fn gemm_simdgroup_tiled_f16_source_epilogue_uses_single_barrier() {
    let kernel_body = gemm_simdgroup_tiled_f16_kernel_body();
    let barrier_count = kernel_body
        .matches("simdgroup_barrier(mem_flags::mem_threadgroup)")
        .count();
    assert_eq!(
        barrier_count, 1,
        "gemm_simdgroup_tiled_f16 の simdgroup_barrier 出現数が 1 ではありません（エピローグの barrier 粒度統合が崩れている可能性。イシュー #797）"
    );
}

/// イシュー #536 の証跡（#745 でも staged 経路の残存有無を再確認）:
/// `gemm_simdgroup_tiled` の MMA 発行ループが蛇行（serpentine）走査順
/// （`acc_cols - 1 - ci`。奇数行 r で列を逆順に辿る）を使用している箇所を
/// Linux CI（ubuntu-latest）上でロックする。
///
/// イシュー #745 でフラグメント（`a_frag`/`b_frag`）を kk ステップ先頭で
/// 一括ロードしてからレジスタ常駐のまま TM×TN の外積 MMA を発行する構造へ
/// staged 経路を再構成した結果、蛇行走査が狙っていた「tile_b 再ロード
/// アドレスの局所性向上」という前提（b_tile を (r, ci) の内側で毎回
/// 再ロードしていたこと）が構造的に消滅したため、staged 経路の蛇行走査は
/// 撤去し MMA 発行順を行優先へ戻した。direct-load 経路（else 節）は
/// フラグメント再ロードが引き続き残る構造のため #536 の蛇行走査を維持して
/// いる。よって出現数は 1（direct-load 経路のみ）へ変わる。
///
/// 走査順の並べ替えのみで `acc[r][c_]` ごとの累算オペランド列（K 方向の
/// 順序）は不変のため、既存の数値一致テスト（tolerance 変更なし）はこの
/// 変更と独立に green であるべきという前提のもとに追加した検査（本テスト
/// 自体は Metal 実機を必要としない文字列検査）。
#[test]
fn gemm_simdgroup_tiled_source_uses_serpentine_scan_order() {
    let kernel_body = gemm_simdgroup_tiled_kernel_body();
    let needle = "uint c_ = (r % 2 == 1) ? (acc_cols - 1 - ci) : ci;";
    let occurrences = kernel_body.matches(needle).count();
    assert_eq!(
        occurrences, 1,
        "gemm_simdgroup_tiled の直接ロード経路に蛇行走査式 `{needle}` が見つかりません（見つかった数: {occurrences}。イシュー #745 で staged 経路の蛇行走査はフラグメントレジスタ常駐化により撤去済み）"
    );
}

/// イシュー #745 の証跡: `gemm_simdgroup_tiled` の staged 経路が MLX
/// steel `mma.h` 型の「kk ステップ先頭でフラグメント配列を一括ロードして
/// からレジスタ常駐のまま TM×TN の外積 MMA を発行する」構造へ移植されて
/// いることを Linux CI（ubuntu-latest）上でロックする。従来は (r, ci) の
/// 内側ループで毎回 `b_tile` を再ロードしていたため 1 kk ステップあたり
/// TM + TM*TN 回のロードが発生していたが、本構造では TM + TN 回まで
/// 削減される（`docs/perf/metal-gemm-register-accumulator-ab.md` §2
/// 診断記録参照）。フラグメントロードと MMA 発行が分離された 3 段構成
/// （A ロード → B ロード → MMA 二重ループ）であることを検査することで、
/// 将来ロードが再び内側ループへ巻き戻される退行を検出する。
#[test]
fn gemm_simdgroup_tiled_source_uses_register_resident_fragment_arrays() {
    let kernel_body = gemm_simdgroup_tiled_kernel_body();
    for needle in [
        "simdgroup_float8x8 a_frag[MAX_ACC];",
        "simdgroup_float8x8 b_frag[MAX_ACC];",
        "simdgroup_load(a_frag[r], tile_a + (size_t)(wm_idx * sub_bm + r * 8) * (size_t)lda + (size_t)kk, lda);",
        "simdgroup_load(b_frag[c_], tile_b + (size_t)kk * (size_t)ldb + (size_t)(wn_idx * sub_bn + c_ * 8), ldb);",
        "simdgroup_multiply_accumulate(acc[r][c_], a_frag[r], b_frag[c_], acc[r][c_]);",
    ] {
        assert!(
            kernel_body.contains(needle),
            "gemm_simdgroup_tiled の staged 経路にフラグメントレジスタ常駐構造 `{needle}` が見つかりません"
        );
    }
    // 退行検出（インラインの a_tile 変数へ巻き戻されていないこと）: staged
    // 経路の kk ループ部分（`kernel_body` 先頭〜direct-load 経路〈else 節〉
    // 開始位置の手前まで）に検査範囲を限定する。`kernel_body` 全体（staged
    // ・direct-load 両経路を含む）を対象にすると、direct-load 経路側の
    // フラグメント変数宣言文言が偶然一致した場合に staged 経路の退行だと
    // 誤診断しうるため（現状 direct-load 側は
    // `simdgroup_float8x8 a_tile = simdgroup_float8x8(0.0f);` という別の
    // 宣言文言を使っており本 needle とは一致しないが、将来 direct-load 側の
    // 文言が変わった場合に備えて検査範囲自体を staged 経路へ絞る）。
    let direct_load_marker = "// 直接ロード: device メモリから simdgroup ごとに直接";
    let staged_scope = &kernel_body[..kernel_body.find(direct_load_marker).expect(
        "gemm_simdgroup_tiled の direct-load 経路（else 節）の目印コメントが見つかりません",
    )];
    assert!(
        !staged_scope.contains("simdgroup_float8x8 a_tile;"),
        "staged 経路の kk ループ内に旧来のインライン a_tile 宣言が残っています（フラグメント巻き上げの退行）"
    );
}

// イシュー #540「gemm_simdgroup_tiled の SWIZZLE_LOG/SWIZZLE_ENABLED 証跡
// 検査」は `crates/backend-metal/src/tile.rs` の crate 内 unit test
// （`gemm_simdgroup_tiled_source_uses_tgid_swizzle`）へ移設した（PR #661
// codex-review 指摘対応）。`crate::tile::SWIZZLE_LOG`/`SWIZZLE_ENABLED` を
// `pub(crate)` へ狭めたため（実験的な内部実装詳細を公開 API に露出しない
// 方針）、別コンパイル単位である本ファイル（`tests/` 配下の統合テスト）
// からは参照できなくなったことによる（直上の `CANDIDATES` 巡回テストが
// `tile.rs` 側に置かれている理由と同じ）。

/// イシュー #533 の証跡: `gemm_simdgroup_tiled` の staged ロード（協調
/// ロード）経路が A/B タイルとも `float4` ベクトルロード（1 要素ずつの
/// スカラーロードではなく `reinterpret_cast<device const float4*>` 経由の
/// 128bit 幅読み出し）へ移植されていることを Linux CI（ubuntu-latest）上で
/// ロックする。Mac 実機依存の parity・A/B 計測（
/// `docs/perf/metal-gemm-float4-staged-load.md`）は別途実施する。将来の
/// 書き戻し（スカラー化への retrograde）をこのテストで検出する。
///
/// ベクトルロードは A タイル・B タイルの 2 箇所で発行されるため、needle の
/// 出現数を 2 に固定する（境界外グループのスカラーフォールバック側には
/// この needle は現れないため、退行〈float4 化の巻き戻し〉があれば
/// occurrences が 0 になり検出できる）。
#[test]
fn gemm_simdgroup_tiled_source_uses_float4_staged_load() {
    let kernel_body = gemm_simdgroup_tiled_kernel_body();
    let needle = "reinterpret_cast<device const float4*>(";
    let occurrences = kernel_body.matches(needle).count();
    // イシュー #1138: NN 用（A/B）2 箇所 + 転置ロード用（TRANS_A/TRANS_B
    // 分岐。A タイル・B タイルそれぞれ 1 箇所ずつ）2 箇所の計 4 箇所。
    assert_eq!(
        occurrences, 4,
        "gemm_simdgroup_tiled の staged 経路（NN・転置ロード双方）の A/B タイルロードに float4 ベクトルロード `{needle}` が見つかりません（見つかった数: {occurrences}）"
    );
}

/// 上記 float4 ベクトルロードが REQ-8 の手動境界チェックを省略していない
/// ことのロック（境界グループの要素単位スカラーフォールバックが staged
/// 経路に残っていることを検査。`.claude/rules/coding-rust.md`「カーネル
/// 実装の境界検査」: 性能上の下限・最適化の達成を理由に境界チェックを
/// 省略しない）。
#[test]
fn gemm_simdgroup_tiled_source_retains_float4_load_boundary_fallback() {
    let kernel_body = gemm_simdgroup_tiled_kernel_body();
    assert!(
        kernel_body.contains("bool group_in_bounds ="),
        "float4 ベクトルロードのグループ単位 in-bounds 判定が見つかりません（境界チェック省略の疑い）"
    );
    // イシュー #1138: NN 用（A/B）2 箇所 + 転置ロード用（TRANS_A/TRANS_B。
    // A タイル・B タイルそれぞれ 1 箇所ずつ）2 箇所の計 4 箇所。
    assert_eq!(
        kernel_body.matches("bool group_in_bounds =").count(),
        4,
        "A タイル・B タイルの NN・転置ロード双方に group_in_bounds 判定が必要です"
    );
}

/// イシュー #538 の証跡: threadgroup memory のパディング幅 `TGP_PAD` が
/// MSL function constant（index 6）として宣言されていることを Linux CI
/// （ubuntu-latest）上でロックする。`crate::pipeline::make_pipeline_with_constants`
/// が渡す `TileConfig::pad` と 1:1 対応する契約（Rust 側は
/// `crates/backend-metal/src/tile.rs`・`crates/backend-metal/src/pipeline.rs`
/// の単体テストで別途検証する）。
#[test]
fn gemm_metal_source_declares_tgp_pad_function_constant() {
    assert!(
        GEMM_METAL_SOURCE.contains("constant uint TGP_PAD [[function_constant(6)]];"),
        "gemm.metal に TGP_PAD function constant（index 6）の宣言が見つかりません"
    );
}

/// イシュー #538 の証跡: `gemm_simdgroup_tiled` の staged 経路が
/// `simdgroup_load` の行ストライドに素の `BK`/`BN` ではなく `TGP_PAD` を
/// 含む `lda`/`ldb` を使用していることをロックする（バンクコンフリクト
/// 回避の要である「パディング込みストライドでのロード」自体が失われて
/// いないことの検査。`lda`/`ldb` の定義式自体〈`BK + TGP_PAD`〉も併せて
/// 固定する）。
#[test]
fn gemm_simdgroup_tiled_source_uses_tgp_padding_stride() {
    let kernel_body = gemm_simdgroup_tiled_kernel_body();
    for needle in [
        "uint lda = BK + TGP_PAD;",
        "uint ldb = BN + TGP_PAD;",
        // イシュー #745 でフラグメント配列（`a_frag`/`b_frag`）へ移植された後も
        // パディング込みストライド（`lda`/`ldb`）でのロードは維持される
        // （変数名のみ変更。ストライド式自体は不変）。
        "simdgroup_load(a_frag[r], tile_a + (size_t)(wm_idx * sub_bm + r * 8) * (size_t)lda + (size_t)kk, lda);",
        "simdgroup_load(b_frag[c_], tile_b + (size_t)kk * (size_t)ldb + (size_t)(wn_idx * sub_bn + c_ * 8), ldb);",
    ] {
        assert!(
            kernel_body.contains(needle),
            "gemm_simdgroup_tiled に TGP_PAD 込みストライド `{needle}` が見つかりません"
        );
    }
}

/// イシュー #538 の証跡: 上記パディング導入後も staged 経路の REQ-8
/// 手動境界チェック（float4 グループ単位 in-bounds 判定・境界外要素の
/// スカラーフォールバック 0 埋め）が維持されていることをロックする
/// （`gemm_simdgroup_tiled_source_retains_float4_load_boundary_fallback`
/// と同種の検査を、パディング導入後の書き込み先添字〈`dst_idx`〉に対して
/// 再確認する）。
#[test]
fn gemm_simdgroup_tiled_source_retains_boundary_guard_with_padding() {
    let kernel_body = gemm_simdgroup_tiled_kernel_body();
    assert!(
        kernel_body.contains("uint dst_idx = r * lda + kk;"),
        "A タイルのパディング込み書き込み先添字 dst_idx が見つかりません"
    );
    assert!(
        kernel_body.contains("uint dst_idx = kk * ldb + c_;"),
        "B タイルのパディング込み書き込み先添字 dst_idx が見つかりません"
    );
    // イシュー #1138: 上記コメントと同じ理由で 4 箇所（NN 2 + 転置 2）。
    assert_eq!(
        kernel_body.matches("bool group_in_bounds =").count(),
        4,
        "パディング導入後も A タイル・B タイル双方（NN・転置ロード）の group_in_bounds 判定が必要です"
    );
}

/// イシュー #809 の証跡: `FINE_BARRIER_ENABLED` function constant
/// （index 8。`SWIZZLE_ENABLED`〈#540・index 7〉の直後）がファイル冒頭で
/// 宣言されていることをロックする。
#[test]
fn gemm_metal_source_declares_fine_barrier_enabled_function_constant() {
    assert!(
        GEMM_METAL_SOURCE.contains("constant bool FINE_BARRIER_ENABLED [[function_constant(8)]];"),
        "gemm.metal に FINE_BARRIER_ENABLED function constant（index 8。SWIZZLE_ENABLED〈#540・index 7〉の \
         直後）宣言が見つかりません"
    );
}

/// イシュー #809 の証跡: `gemm_simdgroup_tiled` の staged 経路 kk ループが、
/// フラグメント一括ロード（`a_frag`/`b_frag`。#745）と MMA 発行の間に
/// `FINE_BARRIER_ENABLED` でゲートされた `simdgroup_barrier(mem_flags::mem_none)`
/// を挿入していることをロックする（本番既定 `false` のため実行時コストは
/// ないが、A/B 計測で `true` を渡した際に実際にこの挿入経路を通ることを
/// 保証する証跡）。挿入位置は B フラグメントロード（`b_frag[c_]` の
/// `simdgroup_load`）直後・MMA 発行ループ（`simdgroup_multiply_accumulate`）
/// 直前であることも合わせて検査する（barrier がフラグメントロード完了後・
/// MMA 発行前という契約〈本ファイル冒頭 FINE_BARRIER_ENABLED 宣言コメント
/// 参照〉から外れた位置への移動を検出するため）。
#[test]
fn gemm_simdgroup_tiled_source_gates_fine_barrier_between_fragment_load_and_mma() {
    let kernel_body = gemm_simdgroup_tiled_kernel_body();

    assert!(
        kernel_body.contains("if (FINE_BARRIER_ENABLED) {\n                    simdgroup_barrier(mem_flags::mem_none);\n                }"),
        "gemm_simdgroup_tiled に FINE_BARRIER_ENABLED ゲート付き simdgroup_barrier(mem_flags::mem_none) \
         挿入が見つかりません"
    );

    let b_frag_load_pos = kernel_body
        .find("simdgroup_load(b_frag[c_],")
        .expect("B フラグメントロード（b_frag）が見つかりません");
    let barrier_pos = kernel_body
        .find("simdgroup_barrier(mem_flags::mem_none);")
        .expect("FINE_BARRIER_ENABLED ゲート付き simdgroup_barrier が見つかりません");
    let mma_pos = kernel_body
        .find("simdgroup_multiply_accumulate(acc[r][c_], a_frag[r], b_frag[c_], acc[r][c_]);")
        .expect("MMA 発行（simdgroup_multiply_accumulate）が見つかりません");

    assert!(
        b_frag_load_pos < barrier_pos && barrier_pos < mma_pos,
        "simdgroup_barrier(mem_flags::mem_none) は B フラグメントロード直後・MMA 発行直前に \
         位置する契約です（b_frag_load={b_frag_load_pos} barrier={barrier_pos} mma={mma_pos}）"
    );
}

/// イシュー #1040 の証跡: `gemm_tiled_bias_act` が `GemmStrides`（lda/ldb/
/// trans_a/trans_b）を受け取り、転置パターン別の添字分岐（`crate::layout`
/// の `a_at`/`b_at` 参照実装と同じ式）を保持していることをロックする。
/// 将来の書き換えでこの分岐が誤って単純な密行優先添字へ巻き戻された
/// 場合に検出する（`gemm_metal_source_uses_simdgroup_matrix_instructions`
/// と同方針）。
#[test]
fn gemm_tiled_bias_act_source_retains_gemm_strides_transpose_branch() {
    for needle in [
        "struct GemmStrides {",
        "constant GemmStrides& st [[buffer(7)]]",
        "st.trans_a != 0",
        "st.trans_b != 0",
        "a[(size_t)a_col * (size_t)st.lda + (size_t)row]",
        "a[(size_t)row * (size_t)st.lda + (size_t)a_col]",
        "b[(size_t)col * (size_t)st.ldb + (size_t)b_row]",
        "b[(size_t)b_row * (size_t)st.ldb + (size_t)col]",
    ] {
        assert!(
            GEMM_METAL_SOURCE.contains(needle),
            "gemm.metal に転置パターン別添字 `{needle}` が見つかりません"
        );
    }
}

/// `gemm_tiled_bias_act` の REQ-8 手動境界チェック（`row < m && a_col < k`・
/// `b_row < k && col < n`・C 書き込み時の `row < m && col < n`）が
/// `GemmStrides` 導入後も維持されていることをロックする（イシュー
/// #1040。カーネル境界検査規約「性能・最適化を理由に手動境界チェックを
/// 省略しない」への準拠を機械検証する）。
#[test]
fn gemm_tiled_bias_act_source_retains_req8_boundary_guards() {
    let kernel_start = GEMM_METAL_SOURCE
        .find("kernel void gemm_tiled_bias_act(")
        .expect("gemm_tiled_bias_act カーネルが見つかりません");
    let next_kernel_start = GEMM_METAL_SOURCE[kernel_start..]
        .find("\nkernel void gemm_simdgroup(")
        .map(|offset| kernel_start + offset)
        .expect("gemm_tiled_bias_act の後続カーネル境界が見つかりません");
    let kernel_body = &GEMM_METAL_SOURCE[kernel_start..next_kernel_start];

    for needle in [
        "row < m && a_col < k",
        "b_row < k && col < n",
        "row < m && col < n",
    ] {
        assert!(
            kernel_body.contains(needle),
            "gemm_tiled_bias_act に REQ-8 境界チェック `{needle}` が見つかりません"
        );
    }
}

/// イシュー #1138 の証跡: `gemm_simdgroup_tiled` の転置ロードゲート
/// `TRANS_A`/`TRANS_B`（function constant index 9/10。FINE_BARRIER_ENABLED
/// 〈#809・index 8〉の直後）が index まで含めて宣言されていることを
/// ロックする（#540/#538 の index 衝突再発防止として本規約に沿う）。
#[test]
fn gemm_metal_source_declares_trans_a_trans_b_function_constants() {
    for needle in [
        "constant bool TRANS_A [[function_constant(9)]];",
        "constant bool TRANS_B [[function_constant(10)]];",
    ] {
        assert!(
            GEMM_METAL_SOURCE.contains(needle),
            "gemm.metal に転置ロードゲート `{needle}`（イシュー #1138）が見つかりません"
        );
    }
}

/// イシュー #1138 の証跡: `gemm_simdgroup_tiled` が `GemmStrides`
/// （`gemm_tiled_bias_act` と共用のレイアウト一致テスト対象。buffer index
/// 4）を受け取り、NT/TN/TT 用の転置フラグメントロード
/// （`simdgroup_load(..., true)`。A タイル・B タイルそれぞれ 1 箇所）を
/// staged 経路の kk ループに保持していることをロックする。
#[test]
fn gemm_simdgroup_tiled_source_retains_transpose_fragment_loads() {
    let kernel_body = gemm_simdgroup_tiled_kernel_body();
    for needle in [
        "constant GemmStrides& st [[buffer(4)]]",
        "simdgroup_load(a_frag[r], tile_a + (size_t)kk * (size_t)lda + (size_t)(wm_idx * sub_bm + r * 8), lda, ulong2(0), true);",
        "simdgroup_load(b_frag[c_], tile_b + (size_t)(wn_idx * sub_bn + c_ * 8) * (size_t)ldb + (size_t)kk, ldb, ulong2(0), true);",
        "if (TRANS_A) {",
        "if (TRANS_B) {",
    ] {
        assert!(
            kernel_body.contains(needle),
            "gemm_simdgroup_tiled に転置フラグメントロード `{needle}`（イシュー #1138）が見つかりません"
        );
    }
}

/// イシュー #1138 の証跡: 転置ロード側の新規境界検査ヘルパ
/// （`tiled_at_group_in_bounds`/`tiled_at_elem_in_bounds`/
/// `tiled_bt_group_in_bounds`/`tiled_bt_elem_in_bounds`）が実在し、
/// 既存 5 ヘルパの本体を変更していないことをロックする。
#[test]
fn gemm_metal_source_declares_transpose_boundary_helpers() {
    for needle in [
        "inline bool tiled_at_group_in_bounds(",
        "inline bool tiled_at_elem_in_bounds(",
        "inline bool tiled_bt_group_in_bounds(",
        "inline bool tiled_bt_elem_in_bounds(",
    ] {
        assert!(
            GEMM_METAL_SOURCE.contains(needle),
            "gemm.metal に転置ロード側境界ヘルパ `{needle}`（イシュー #1138）が見つかりません"
        );
    }
}

/// イシュー #1188 の証跡（E1 実験。親 #1037・参照 #1143）: `gemm_simdgroup_tiled`
/// （f32）のアキュムレータ系ループ（acc 初期化・staged 経路の
/// フラグメントロード〈a_frag/b_frag〉・MMA 発行〈staged/direct-load
/// 両経路〉・エピローグのストア）全 10 箇所へ `#pragma clang loop
/// unroll(full)` が付与されていることをロックする。
///
/// M4 Max 実機実測（`docs/perf/metal-gemm-n4096-e1-unroll-1188/`・
/// `docs/perf/metal-gemm-n4096-kernel-gap.md` §7）で、`acc_rows*acc_cols`
/// が大きい候補（cand0/cand4/cand8）が N=4096 で著しく劣化する現象が
/// unroll 未展開によるレジスタ配列の thread-local メモリへの降格（spill）
/// に起因するという仮説（H2）を支持する結果（当該候補が 4〜16 倍改善・
/// 本番 `dispatch_auto` 経路は非後退）を得たため採用した。出現数を
/// 10 に固定することで、将来ループ構造が変わった際に付与漏れ・過剰付与
/// （意図しない箇所への波及）の両方を検出する。
#[test]
fn gemm_simdgroup_tiled_source_unrolls_accumulator_loops() {
    let kernel_body = gemm_simdgroup_tiled_kernel_body();
    let needle = "#pragma clang loop unroll(full)";
    let occurrences = kernel_body.matches(needle).count();
    assert_eq!(
        occurrences, 10,
        "gemm_simdgroup_tiled の `{needle}` 出現数が 10 ではありません（見つかった数: {occurrences}。\
         イシュー #1188 の E1 実験で acc 初期化 2・staged フラグメントロード 2・\
         staged MMA 発行 2・direct-load MMA 発行 2・エピローグストア 2 の \
         計 10 箇所へ付与した契約が崩れている可能性）"
    );
}

/// イシュー #1188 の証跡: E1 の unroll pragma は f32 経路
/// （`gemm_simdgroup_tiled`）限定のスコープであり、f16 経路
/// （`gemm_simdgroup_tiled_f16`）へは波及していないことをロックする
/// （実験計画のスコープ外拡大を防ぐ）。
#[test]
fn gemm_simdgroup_tiled_f16_source_does_not_unroll_accumulator_loops() {
    let kernel_body = gemm_simdgroup_tiled_f16_kernel_body();
    assert!(
        !kernel_body.contains("#pragma clang loop unroll(full)"),
        "gemm_simdgroup_tiled_f16 に E1（#1188。f32 経路限定のはずの unroll pragma）が \
         波及しています"
    );
}
