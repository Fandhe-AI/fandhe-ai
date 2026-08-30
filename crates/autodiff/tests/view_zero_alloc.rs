//! 受け入れ条件「view 系ノード（reshape / transpose）の forward・
//! backward でバッファ確保が 0 であること」を
//! `bench_harness::alloc_tracker::TrackingAllocator`（`#[global_allocator]`
//! フック）で機械的に実測する統合テスト（イシュー #1047・親 #1043
//! 「カーネル融合・autodiff 実行モデルの強化」）。
//!
//! ## なぜ `harness = false` か
//!
//! `TrackingAllocator` はプロセス全体で共有される静的カウンタ
//! （`CURRENT_BYTES`／`PEAK_BYTES`）を持つ。libtest の既定ハーネスは
//! 各 `#[test]` をワーカースレッドへ並列ディスパッチするため、他の
//! テストの確保・解放が計測区間へ混入しうる（`crates/bench-harness/src/
//! alloc_tracker.rs` モジュール冒頭「テストでの計測検証（プロセス
//! 分離）」節・`crates/bench-harness/tests/alloc_tracker_serial.rs` と
//! 同じ対処方針）。本ファイルは `harness = false`（`Cargo.toml` の
//! `[[test]] name = "view_zero_alloc"`）とし、`fn main()` が各検査関数を
//! スレッドを生成せず順番に呼ぶ。
//!
//! ## 実測方針
//!
//! `reshape`/`transpose`（`Tape::push_view`・`tape::resolve_view`）は
//! 既存 `Arc<Storage>` を共有する zero-copy view であり、新しい
//! テンソルバッファを一切確保しない設計（`docs/
//! autodiff-view-recompute-decision.md` 参照）。この契約を「入力バッファ
//! （数 MiB 級）に対して純増分ピークが無視できるほど小さい」という
//! 定量的な閾値で検証する。閾値は絶対サイズではなく入力バッファに対する
//! 比率（1/1000）で表現し、`TapeNode` の `Vec` 成長等の無視できる雑音
//! （数百バイト〜数 KiB オーダー、入力バッファのサイズに依存しない）を
//! 吸収しつつ「もし view がバッファ全体をコピーしていたら確実に検出
//! できる」だけの分離度を持たせる。

mod common;

use bench_harness::alloc_tracker::TrackingAllocator;
use bench_harness::alloc_tracker::measure;
use fandhe_ai_autodiff::Tape;
use fandhe_ai_tensor_core::Tensor;

/// 本バイナリ限定で `TrackingAllocator` をプロセスの `#[global_allocator]`
/// として有効化する（`alloc_tracker_serial.rs` と同じ構成）。
#[global_allocator]
static GLOBAL_ALLOCATOR: TrackingAllocator = TrackingAllocator;

/// 入力バッファの一辺の要素数。`N * N * size_of::<f32>()` バイトの
/// リーフテンソルを作る（2048^2 * 4 = 16 MiB）。TapeNode の `Vec` 成長
/// 等の雑音（高々数 KiB）が閾値（バッファの 1/1000 = 16 KiB 強）に
/// 対して十分小さくなるよう、実測にかかる時間とのバランスでこの値を
/// 選んだ。
const N: usize = 2048;

/// 閾値算出の分母（`docs/autodiff-view-recompute-decision.md` §6 実測
/// 記録参照）。`docs` 側の記述と対応づけるため定数化する。
const THRESHOLD_DIVISOR: u64 = 1000;

fn buffer_bytes() -> u64 {
    (N * N * std::mem::size_of::<f32>()) as u64
}

fn make_leaf() -> Tensor<f32> {
    let data = vec![0.0f32; N * N];
    Tensor::new(data, &[N, N])
        .expect("view_zero_alloc: test fixture: shape とデータ長は事前に一致させている")
}

/// 1. `Var::transpose` の forward（ノード記録のみ）が、入力バッファに
///    対して無視できる量しか確保しないことを検証する（`tape::Op::
///    Transpose` の「ホスト値を持たない」契約の直接検証）。
fn check_transpose_forward_is_near_zero_alloc() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let leaf = make_leaf();
    let x = tape.var(&leaf); // リーフ登録（`Arc` 共有想定。測定区間の外）

    let (tr, peak) = measure(|| {
        let tr = x
            .transpose(0, 1)
            .expect("transpose(0,1) は常に成功する（rank 2）");
        std::hint::black_box(&tr);
        tr
    });
    std::hint::black_box(&tr);

    let peak =
        peak.expect("GLOBAL_ALLOCATOR がテストバイナリの #[global_allocator] のため Some のはず");
    let threshold = buffer_bytes() / THRESHOLD_DIVISOR;
    println!(
        "check_transpose_forward_is_near_zero_alloc: peak={peak} bytes, threshold={threshold} bytes, buffer={} bytes",
        buffer_bytes()
    );
    assert!(
        peak < threshold,
        "transpose forward の純増分ピーク（{peak} バイト）が閾値（{threshold} バイト。\
         入力バッファ {} バイトの 1/{THRESHOLD_DIVISOR}）を超えた——zero-copy 契約が破れている疑い",
        buffer_bytes()
    );
}

/// 2. `Var::reshape` の forward が同様に無視できる量しか確保しないことを
///    検証する。
fn check_reshape_forward_is_near_zero_alloc() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let leaf = make_leaf();
    let x = tape.var(&leaf);

    let (r, peak) = measure(|| {
        let r = x
            .reshape(&[N * N])
            .expect("要素数一致・contiguous な reshape は常に成功する");
        std::hint::black_box(&r);
        r
    });
    std::hint::black_box(&r);

    let peak =
        peak.expect("GLOBAL_ALLOCATOR がテストバイナリの #[global_allocator] のため Some のはず");
    let threshold = buffer_bytes() / THRESHOLD_DIVISOR;
    println!(
        "check_reshape_forward_is_near_zero_alloc: peak={peak} bytes, threshold={threshold} bytes, buffer={} bytes",
        buffer_bytes()
    );
    assert!(
        peak < threshold,
        "reshape forward の純増分ピーク（{peak} バイト）が閾値（{threshold} バイト。\
         入力バッファ {} バイトの 1/{THRESHOLD_DIVISOR}）を超えた——zero-copy 契約が破れている疑い",
        buffer_bytes()
    );
}

/// 3. view の連鎖を挟んだ `Tape::backward` が、連鎖長に比例した追加
///    バッファ確保を発生させないことを検証する（`tape::resolve_view`
///    が各ノードで `Arc` 共有のみを行い、実データコピーを重ねない
///    ことの回帰テスト）。
///
///    `x`（leaf）→ `transpose` を 5 回連鎖（正方行列のため shape は
///    不変・strides のみ入れ替わる）→ `sum(None)` で loss を作る。
///    forward（`loss` 構築まで）は測定区間の**外**で行い、
///    `Tape::backward(&loss)` のみを測定する。
///
///    backward が必要とする「本当のバッファ確保」は `Op::Sum` の VJP
///    （`unreduce_broadcast`。スカラー勾配を入力 shape へブロードキャスト
///    する 1 回のみ・入力バッファと同サイズ）のみであり、5 個の
///    `Op::Transpose` ノードの VJP はいずれも zero-copy
///    （`upstream.transpose(dim0, dim1)`）で閉じるはず。よって
///    「連鎖を経ても純増分ピークは入力バッファの高々定数倍（ここでは
///    安全マージンを見て 2 倍）に収まる」ことを検証する——もし view の
///    解決やその VJP が経由するたびにバッファを複製していれば、5 段の
///    連鎖で 5 倍以上のピークになり、この閾値を明確に超える。
fn check_backward_through_view_chain_is_bounded_alloc() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let leaf = make_leaf();
    let x = tape.var(&leaf);

    let mut cur = x;
    for _ in 0..5 {
        cur = cur
            .transpose(0, 1)
            .expect("正方行列の transpose(0,1) は常に成功する");
    }
    let loss = cur.sum(None).expect("sum(None) は常に成功する（全軸縮約）");

    let (grads, peak) = measure(|| tape.backward(&loss).expect("backward は常に成功する構成"));
    std::hint::black_box(&grads);

    let peak =
        peak.expect("GLOBAL_ALLOCATOR がテストバイナリの #[global_allocator] のため Some のはず");
    let threshold = buffer_bytes() * 2;
    println!(
        "check_backward_through_view_chain_is_bounded_alloc: peak={peak} bytes, threshold={threshold} bytes, buffer={} bytes",
        buffer_bytes()
    );
    assert!(
        peak < threshold,
        "view 連鎖を経た backward の純増分ピーク（{peak} バイト）が閾値（{threshold} バイト。\
         入力バッファ {} バイトの 2 倍）を超えた——view の VJP が連鎖のたびにバッファを \
         複製している疑い（zero-copy 契約の回帰）",
        buffer_bytes()
    );

    // 併せて、backward が実際に何らかの実データ確保を行った（トラッカーが
    // 無為に 0 を返しているのではない）ことを最低限確認する。`Op::Sum`
    // の VJP が入力バッファと同サイズの勾配を 1 回確保するはずなので、
    // ピークはその半分程度は超えるはず（噪音マージンを見て 1/4 とする）。
    assert!(
        peak > buffer_bytes() / 4,
        "view 連鎖を経た backward の純増分ピーク（{peak} バイト）が小さすぎる——\
         計測区間が実際の backward 実行を捉えていない疑い"
    );
}

fn main() {
    check_transpose_forward_is_near_zero_alloc();
    check_reshape_forward_is_near_zero_alloc();
    check_backward_through_view_chain_is_bounded_alloc();
    println!("view_zero_alloc: all checks passed");
}
