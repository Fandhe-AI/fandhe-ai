//! ROCm 非搭載環境（CI）での契約: コンパイル経路の入口が panic せず型付きエラーを返すこと。
//! AMD 実機で壊れないよう、成功・失敗のどちらでも許容する（失敗の種類は環境依存）。

use fandhe_ai_backend_rocm::{
    GemmF32Tile, GfxTarget, RocmError, RocmGemmF32, parse_gcn_arch_name, render_gemm_f32_source,
};

#[test]
fn new_does_not_panic_and_errors_are_typed() {
    match RocmGemmF32::new(GfxTarget::Gfx942) {
        Ok(_) => {}
        Err(e) => assert!(!e.to_string().is_empty()),
    }
}

#[test]
fn invalid_shape_is_rejected_when_available() {
    if let Ok(g) = RocmGemmF32::new(GfxTarget::Gfx942) {
        let r = g.run(&[0.0; 3], &[0.0; 4], 2, 2, 2);
        assert!(matches!(r, Err(RocmError::InvalidGemmDims { .. })));
    }
}

#[test]
fn source_generation_and_target_parse_are_public_and_pure() {
    let t = GemmF32Tile::new((64, 64, 16), (4, 4), 4, (16, 16)).unwrap();
    assert_eq!(
        render_gemm_f32_source(&t),
        render_gemm_f32_source(&GemmF32Tile::DEFAULT)
    );
    assert!(parse_gcn_arch_name("gfx942:sramecc+:xnack-").is_ok());
    assert!(parse_gcn_arch_name("gfx000").is_err());
}
