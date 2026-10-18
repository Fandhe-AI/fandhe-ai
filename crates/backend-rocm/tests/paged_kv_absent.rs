//! ROCm 非搭載環境（CI）での契約: `RocmPagedKv::new` が panic せず型付きエラーを返し、搭載環境では
//! 不正入力を HIP に触れる前に検証エラーで拒否すること（イシュー #3443・R12）。AMD 実機で壊れない
//! よう、成功・失敗のどちらでも許容する。`BackendOps` 経由の結線（ハンドル取り違え・検証順序）は
//! crate 内の HostModel テスト（`ops/paged_kv.rs`）が担う。

use fandhe_ai_backend_rocm::{GfxTarget, RocmError, RocmPagedKv};
use fandhe_ai_tensor_core::{PagedAttentionArgs, PagedKvLayout};

#[test]
fn constructor_does_not_panic_and_error_is_typed() {
    if let Err(e) = RocmPagedKv::new(GfxTarget::Gfx942) {
        assert!(!e.to_string().is_empty());
    }
}

#[test]
fn invalid_inputs_are_rejected_before_launch_when_available() {
    let Ok(r) = RocmPagedKv::new(GfxTarget::Gfx942) else {
        return;
    };
    let layout = PagedKvLayout::new(2, 2, 4).unwrap();
    let pool = 2 * 2 * 4;
    let mut kp = vec![0.0f32; pool];
    let mut vp = vec![0.0f32; pool];
    let rows = vec![0.5f32; 2 * 4];
    // 範囲外スロット・重複スロット・行数と slot_mapping の不一致。
    for bad in [&[0u32, 4][..], &[1, 1], &[0]] {
        assert!(
            matches!(
                r.append(&rows, &rows, &mut kp, &mut vp, bad, &layout),
                Err(RocmError::InvalidOpArgs { .. })
            ),
            "{bad:?}"
        );
    }
    // pool が書き換わっていない。
    assert!(kp.iter().chain(vp.iter()).all(|x| *x == 0.0));
    // 空の追記は HIP を起動せず成功する。
    r.append(&[], &[], &mut kp, &mut vp, &[], &layout).unwrap();

    let args = PagedAttentionArgs::new(layout, 1, 1, 2, 2, 0.5).unwrap();
    let q = vec![0.1f32; 4];
    // 番兵・範囲外ブロック ID、seq_len 範囲外、テーブル長不一致。
    for (tables, lens) in [
        (&[0u32, 2][..], &[3u32][..]),
        (&[0, u32::MAX], &[3]),
        (&[0, 1], &[5]),
        (&[0], &[3]),
    ] {
        assert!(
            matches!(
                r.attention(&q, &kp, &vp, tables, lens, &args),
                Err(RocmError::InvalidOpArgs { .. })
            ),
            "{tables:?} {lens:?}"
        );
    }
}
