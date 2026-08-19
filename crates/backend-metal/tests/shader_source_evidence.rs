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
/// 開始位置から EOF まで）を切り出す。本ファイル内で最後に定義される
/// カーネルのため EOF までのスライスで安全（次カーネル境界を考慮する
/// 必要がない）。
fn gemm_simdgroup_tiled_kernel_body() -> &'static str {
    let kernel_start = GEMM_METAL_SOURCE
        .find("kernel void gemm_simdgroup_tiled(")
        .expect("gemm_simdgroup_tiled カーネル本体が見つかりません");
    &GEMM_METAL_SOURCE[kernel_start..]
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
    // MMA 発行がフラグメントロードから分離され、内側ループで再ロード
    // されていないことの追加確認: `simdgroup_multiply_accumulate` の
    // 呼び出しは staged 経路で 1 箇所のみ（旧構造は kk ループ内の (r, ci)
    // 二重ループ内に 1 箇所だったが呼び出し自体の出現数は変わらないため、
    // ここでは a_frag[r]/b_frag[c_] を直接引数に取る形になっている
    // （インラインの a_tile/b_tile 変数を経由しない）ことを主目的とする。
    assert!(
        !kernel_body.contains("simdgroup_float8x8 a_tile;"),
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
    assert_eq!(
        occurrences, 2,
        "gemm_simdgroup_tiled の staged 経路の A/B タイルロードに float4 ベクトルロード `{needle}` が見つかりません（見つかった数: {occurrences}）"
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
    assert_eq!(
        kernel_body.matches("bool group_in_bounds =").count(),
        2,
        "A タイル・B タイルの両方に group_in_bounds 判定が必要です"
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
    assert_eq!(
        kernel_body.matches("bool group_in_bounds =").count(),
        2,
        "パディング導入後も A タイル・B タイル双方の group_in_bounds 判定が必要です"
    );
}
