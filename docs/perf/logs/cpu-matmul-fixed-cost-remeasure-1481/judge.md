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
- 規則2 DGX N=2048決定セル: layer_a(reuse)=0.7090086271821311 (md=0.709, raw=0.7090086271821311, incomplete=False) (<= 1.05) alloc_c比=2.1198577958354496 ops_gemm比=0.8011769028698661 (<= 1.0) -> 不成立
- 規則3 対照セル dgx N=512/fresh: 比=0.9772283959810419 (md=0.9772, raw=0.9772283959810419, incomplete=False) (<= 1.05) -> 満たす
- 規則3 対照セル dgx N=512/reuse: 比=1.0101950698322084 (md=1.0102, raw=1.0101950698322084, incomplete=False) (<= 1.05) -> 満たす
- 規則3 対照セル dgx N=1024/fresh: 比=1.0548854729256847 (md=1.0549, raw=1.0548854729256847, incomplete=False) (<= 1.05) -> 不成立
- 規則3 対照セル dgx N=1024/reuse: 比=0.988243081395619 (md=0.9882, raw=0.988243081395619, incomplete=False) (<= 1.05) -> 満たす
- 規則3 対照セル m4max N=512/fresh: 比=1.0575712251887612 (md=1.0576, raw=1.0575712251887612, incomplete=False) (<= 1.05) -> 不成立
- 規則3 対照セル m4max N=512/reuse: 比=1.0780068968696421 (md=1.078, raw=1.0780068968696421, incomplete=False) (<= 1.05) -> 不成立
- 規則3 対照セル m4max N=1024/fresh: 比=0.9988021400486745 (md=0.9988, raw=0.9988021400486745, incomplete=False) (<= 1.05) -> 満たす
- 規則3 対照セル m4max N=1024/reuse: 比=1.0156256025584462 (md=1.0156, raw=1.0156256025584462, incomplete=False) (<= 1.05) -> 満たす
- 規則4改定版 dgx N=512: candle比 on/off比（丸めなし raw）=1.1062959529506364 (md参考: off=0.726 on=0.803) (>= 0.9524) -> 満たす
- 規則4改定版 dgx N=1024: candle比 on/off比（丸めなし raw）=1.0247084746002495 (md参考: off=0.982 on=1.007) (>= 0.9524) -> 満たす
- 規則4改定版 dgx N=2048: candle比 on/off比（丸めなし raw）=1.4033708610534434 (md参考: off=1.218 on=1.709) (>= 0.9524) -> 満たす
- 規則4改定版 m4max N=512: candle比 on/off比（丸めなし raw）=0.9059650315431891 (md参考: off=1.138 on=1.031) (>= 0.9524) -> 不成立
- 規則4改定版 m4max N=1024: candle比 on/off比（丸めなし raw）=1.0334160680719124 (md参考: off=0.979 on=1.012) (>= 0.9524) -> 満たす
- 規則4改定版 m4max N=2048: candle比 on/off比（丸めなし raw）=0.9970867106200855 (md参考: off=0.997 on=0.994) (>= 0.9524) -> 満たす
- 規則5 M4Max N=2048: layer_a(reuse)=1.021193912877583 (md=1.0212) run別比=[1.03223932395284, 1.0391416770704545, 1.021193912877583, 1.117853165034165, 1.0240237343818477] 中央値超過=False 符号一貫=True -> 後退なし（発火せず）
- 折り込み判定: checksum_ok=True dgx_rules_1_4_ok=False m4max_rules_1_4_ok=False r5_regression=False r5_data_complete=True incomplete_cells=[] -> verdict=REJECT

**verdict=REJECT**
