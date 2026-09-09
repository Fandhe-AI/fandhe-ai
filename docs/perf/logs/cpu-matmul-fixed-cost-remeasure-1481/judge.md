# イシュー #1481 機械判定

事前登録閾値: RULE234_THRESHOLD=1.05 RULE2_OPS_GEMM_THRESHOLD=1.0 RULE4_MIN_RATIO=0.9523809524

- 規則1 checksum dgx N=512/fresh: checksum='完全一致' (== '完全一致') -> 満たす
- 規則1 checksum dgx N=512/reuse: checksum='完全一致' (== '完全一致') -> 満たす
- 規則1 checksum dgx N=1024/fresh: checksum='完全一致' (== '完全一致') -> 満たす
- 規則1 checksum dgx N=1024/reuse: checksum='完全一致' (== '完全一致') -> 満たす
- 規則1 checksum dgx N=2048/fresh: checksum='完全一致' (== '完全一致') -> 満たす
- 規則1 checksum dgx N=2048/reuse: checksum='完全一致' (== '完全一致') -> 満たす
- 規則1 checksum m4max N=512/fresh: checksum='完全一致' (== '完全一致') -> 満たす
- 規則1 checksum m4max N=512/reuse: checksum='完全一致' (== '完全一致') -> 満たす
- 規則1 checksum m4max N=1024/fresh: checksum='完全一致' (== '完全一致') -> 満たす
- 規則1 checksum m4max N=1024/reuse: checksum='完全一致' (== '完全一致') -> 満たす
- 規則1 checksum m4max N=2048/fresh: checksum='完全一致' (== '完全一致') -> 満たす
- 規則1 checksum m4max N=2048/reuse: checksum='完全一致' (== '完全一致') -> 満たす
- 規則2 DGX N=2048決定セル: layer_a(reuse)=0.709 (<= 1.05) alloc_c比=2.1199 ops_gemm比=0.8012 (<= 1.0) -> 不成立
- 規則3 対照セル dgx N=512/fresh: 比=0.9772 (<= 1.05) -> 満たす
- 規則3 対照セル dgx N=512/reuse: 比=1.0102 (<= 1.05) -> 満たす
- 規則3 対照セル dgx N=1024/fresh: 比=1.0549 (<= 1.05) -> 不成立
- 規則3 対照セル dgx N=1024/reuse: 比=0.9882 (<= 1.05) -> 満たす
- 規則3 対照セル m4max N=512/fresh: 比=1.0576 (<= 1.05) -> 不成立
- 規則3 対照セル m4max N=512/reuse: 比=1.078 (<= 1.05) -> 不成立
- 規則3 対照セル m4max N=1024/fresh: 比=0.9988 (<= 1.05) -> 満たす
- 規則3 対照セル m4max N=1024/reuse: 比=1.0156 (<= 1.05) -> 満たす
- 規則4改定版 dgx N=512: candle比 off=0.726 on=0.803 on/off比=1.1060606060606062 (>= 0.9524) -> 満たす
- 規則4改定版 dgx N=1024: candle比 off=0.982 on=1.007 on/off比=1.025458248472505 (>= 0.9524) -> 満たす
- 規則4改定版 dgx N=2048: candle比 off=1.218 on=1.709 on/off比=1.40311986863711 (>= 0.9524) -> 満たす
- 規則4改定版 m4max N=512: candle比 off=1.138 on=1.031 on/off比=0.9059753954305799 (>= 0.9524) -> 不成立
- 規則4改定版 m4max N=1024: candle比 off=0.979 on=1.012 on/off比=1.0337078651685394 (>= 0.9524) -> 満たす
- 規則4改定版 m4max N=2048: candle比 off=0.997 on=0.994 on/off比=0.9969909729187563 (>= 0.9524) -> 満たす
- 規則5 M4Max N=2048: layer_a(reuse)=1.0212 -> 後退なし（発火せず）
- 折り込み判定: checksum_ok=True dgx_rules_1_4_ok=False m4max_rules_1_4_ok=False r5_regression=False -> verdict=REJECT

**verdict=REJECT**
