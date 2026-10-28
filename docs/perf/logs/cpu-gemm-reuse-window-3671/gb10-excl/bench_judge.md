OFFICIAL: yes
N=256 [判定対象] fresh_bench=0.000187263 reuse_bench=0.000347815 G_bench=+0.000160552 fresh_diag=0.000193679 resident_public=0.000173599 G_diag=-0.000020080 G_bench/G_diag=-7.996 bridge_v2=fail E=2.072 matmul/G_bench=1.895
  AUX bench_base: reuse_median_s=0.000352111 reuse_minus_phases_iter_total=+0.000004296 to_tensor=0.000000456 host_copy=0.000000096 checksum=0.000033648
  AUX bench_tunables: reuse_median_s=0.000164816 reuse_minus_phases_iter_total=-0.000009536 to_tensor=0.000000256 host_copy=0.000000096 checksum=0.000033616
  AUX bench_headpatch: reuse_median_s=0.000362007 reuse_minus_phases_iter_total=-0.000010776 to_tensor=0.000000304 host_copy=0.000000128 checksum=0.000033631
  X rp_verify_touch: iter_total=0.000211439 G_X=+0.000017760 G_bench/G_X=9.040 支持=no iter_total/bench_reuse=0.608
  X rp_verify_gap: iter_total=0.000169951 G_X=-0.000023728 G_bench/G_X=-6.766 支持=no iter_total/bench_reuse=0.489
  X rp_hold_footprint: iter_total=0.000168368 G_X=-0.000025311 G_bench/G_X=-6.343 支持=no iter_total/bench_reuse=0.484
  X rp_ref_compute: iter_total=0.000156480 G_X=-0.000037199 G_bench/G_X=-4.316 支持=no iter_total/bench_reuse=0.450
  X rp_solo: iter_total=0.000191727 G_X=-0.000001952 G_bench/G_X=-82.250 支持=no iter_total/bench_reuse=0.551（G は主診断系列の fresh を使用）
  Y bench_tunables: reuse=0.000174352 fresh=0.000174959 G_Y=-0.000000607 G_Y/G_diag=0.030 支持=no
  Y bench_headpatch: reuse=0.000372783 fresh=0.000182240 G_Y=+0.000190543 G_Y/G_diag=-9.489 支持=no
N=512 [判定対象] fresh_bench=0.001709948 reuse_bench=0.002356401 G_bench=+0.000646453 fresh_diag=0.001135438 resident_public=0.001161710 G_diag=+0.000026272 G_bench/G_diag=24.606 bridge_v2=fail E=1.347 matmul/G_bench=3.350
  AUX bench_base: reuse_median_s=0.002465714 reuse_minus_phases_iter_total=+0.000109313 to_tensor=0.000000448 host_copy=0.000000080 checksum=0.000188872
  AUX bench_tunables: reuse_median_s=0.001685980 reuse_minus_phases_iter_total=-0.000081192 to_tensor=0.000000544 host_copy=0.000000128 checksum=0.000188608
  AUX bench_headpatch: reuse_median_s=0.002527365 reuse_minus_phases_iter_total=+0.000111983 to_tensor=0.000000320 host_copy=0.000000064 checksum=0.000134703
  X rp_verify_touch: iter_total=0.001405469 G_X=+0.000270031 G_bench/G_X=2.394 支持=no iter_total/bench_reuse=0.596
  X rp_verify_gap: iter_total=0.001270110 G_X=+0.000134672 G_bench/G_X=4.800 支持=no iter_total/bench_reuse=0.539
  X rp_hold_footprint: iter_total=0.001171578 G_X=+0.000036140 G_bench/G_X=17.887 支持=no iter_total/bench_reuse=0.497
  X rp_ref_compute: iter_total=0.000917903 G_X=-0.000217535 G_bench/G_X=-2.972 支持=no iter_total/bench_reuse=0.390
  X rp_solo: iter_total=0.001159005 G_X=+0.000023567 G_bench/G_X=27.430 支持=no iter_total/bench_reuse=0.492（G は主診断系列の fresh を使用）
  Y bench_tunables: reuse=0.001767172 fresh=0.001837077 G_Y=-0.000069905 G_Y/G_diag=-2.661 支持=no
  Y bench_headpatch: reuse=0.002415382 fresh=0.001676817 G_Y=+0.000738565 G_Y/G_diag=28.112 支持=no
N=1024 [判定対象] fresh_bench=0.004699717 reuse_bench=0.006272199 G_bench=+0.001572482 fresh_diag=0.004325010 resident_public=0.004493833 G_diag=+0.000168823 G_bench/G_diag=9.314 bridge_v2=fail E=1.284 matmul/G_bench=3.647
  AUX bench_base: reuse_median_s=0.006410313 reuse_minus_phases_iter_total=+0.000138114 to_tensor=0.000000736 host_copy=0.000000184 checksum=0.000540775
  AUX bench_tunables: reuse_median_s=0.004677406 reuse_minus_phases_iter_total=-0.000157170 to_tensor=0.000000776 host_copy=0.000000096 checksum=0.000551068
  AUX bench_headpatch: reuse_median_s=0.006327681 reuse_minus_phases_iter_total=+0.000009657 to_tensor=0.000000776 host_copy=0.000000176 checksum=0.000541174
  X rp_verify_touch: iter_total=0.004617945 G_X=+0.000292935 G_bench/G_X=5.368 支持=no iter_total/bench_reuse=0.736
  X rp_verify_gap: iter_total=0.004359170 G_X=+0.000034160 G_bench/G_X=46.033 支持=no iter_total/bench_reuse=0.695
  X rp_hold_footprint: iter_total=0.004271851 G_X=-0.000053159 G_bench/G_X=-29.581 支持=no iter_total/bench_reuse=0.681
  X rp_ref_compute: iter_total=0.004202147 G_X=-0.000122863 G_bench/G_X=-12.799 支持=no iter_total/bench_reuse=0.670
  X rp_solo: iter_total=0.004172580 G_X=-0.000152430 G_bench/G_X=-10.316 支持=no iter_total/bench_reuse=0.665（G は主診断系列の fresh を使用）
  Y bench_tunables: reuse=0.004834576 fresh=0.004601933 G_Y=+0.000232643 G_Y/G_diag=1.378 支持=yes
  Y bench_headpatch: reuse=0.006318024 fresh=0.004714421 G_Y=+0.001603603 G_Y/G_diag=9.499 支持=no
N=2048 [判定対象] fresh_bench=0.017806766 reuse_bench=0.025802946 G_bench=+0.007996180 fresh_diag=0.016451996 resident_public=0.021849420 G_diag=+0.005397424 G_bench/G_diag=1.481 bridge_v2=pass E=1.091 matmul/G_bench=2.930
  AUX bench_base: reuse_median_s=0.025265411 reuse_minus_phases_iter_total=-0.000537535 to_tensor=0.000001048 host_copy=0.000000128 checksum=0.002129539
  AUX bench_tunables: reuse_median_s=0.017978525 reuse_minus_phases_iter_total=-0.000675928 to_tensor=0.000001160 host_copy=0.000000128 checksum=0.003016825
  AUX bench_headpatch: reuse_median_s=0.027935006 reuse_minus_phases_iter_total=-0.002122096 to_tensor=0.000001152 host_copy=0.000000128 checksum=0.002159345
  X rp_verify_touch: iter_total=0.025737375 G_X=+0.009285379 G_bench/G_X=0.861 支持=yes iter_total/bench_reuse=0.997
  X rp_verify_gap: iter_total=0.021552999 G_X=+0.005101003 G_bench/G_X=1.568 支持=yes iter_total/bench_reuse=0.835
  X rp_hold_footprint: iter_total=0.021483463 G_X=+0.005031467 G_bench/G_X=1.589 支持=yes iter_total/bench_reuse=0.833
  X rp_ref_compute: iter_total=0.021582327 G_X=+0.005130331 G_bench/G_X=1.559 支持=yes iter_total/bench_reuse=0.836
  X rp_solo: iter_total=0.016611122 G_X=+0.000159126 G_bench/G_X=50.251 支持=no iter_total/bench_reuse=0.644（G は主診断系列の fresh を使用）
  Y bench_tunables: reuse=0.018654453 fresh=0.017220758 G_Y=+0.001433695 G_Y/G_diag=0.266 支持=no
  Y bench_headpatch: reuse=0.030057102 fresh=0.017493921 G_Y=+0.012563181 G_Y/G_diag=2.328 支持=no
N=4096 [対照（判定外）] fresh_bench=0.126705577 reuse_bench=0.152389650 G_bench=+0.025684073 fresh_diag=0.125358119 resident_public=0.151215549 G_diag=+0.025857430 G_bench/G_diag=0.993 bridge_v2=pass E=0.997 matmul/G_bench=5.599
  AUX bench_base: reuse_median_s=0.152758489 reuse_minus_phases_iter_total=+0.000368839 to_tensor=0.000001488 host_copy=0.000000128 checksum=0.008594940
  AUX bench_tunables: reuse_median_s=0.152084751 reuse_minus_phases_iter_total=-0.000001648 to_tensor=0.000001368 host_copy=0.000000128 checksum=0.008690845
  AUX bench_headpatch: reuse_median_s=0.153021724 reuse_minus_phases_iter_total=+0.000075344 to_tensor=0.000001432 host_copy=0.000000128 checksum=0.008689997
  X rp_verify_touch: iter_total=0.150624518 G_X=+0.025266399 G_bench/G_X=1.017 支持=yes iter_total/bench_reuse=0.988
  X rp_verify_gap: iter_total=0.150660721 G_X=+0.025302602 G_bench/G_X=1.015 支持=yes iter_total/bench_reuse=0.989
  X rp_hold_footprint: iter_total=0.150379754 G_X=+0.025021635 G_bench/G_X=1.026 支持=yes iter_total/bench_reuse=0.987
  X rp_ref_compute: iter_total=0.150767466 G_X=+0.025409347 G_bench/G_X=1.011 支持=yes iter_total/bench_reuse=0.989
  X rp_solo: iter_total=0.151489258 G_X=+0.026131139 G_bench/G_X=0.983 支持=yes iter_total/bench_reuse=0.994（G は主診断系列の fresh を使用）
  Y bench_tunables: reuse=0.152086399 fresh=0.124461244 G_Y=+0.027625155 G_Y/G_diag=1.068 支持=yes
  Y bench_headpatch: reuse=0.152946380 fresh=0.125927175 G_Y=+0.027019205 G_Y/G_diag=1.045 支持=yes
## 結論（RULE.v2 の規則どおり。因果の断定ではない）
N=256: 特定できず（どの腕も支持されない。各比は上記に記録）
N=512: 特定できず（どの腕も支持されない。各比は上記に記録）
N=1024: 支持された腕 = bench_tunables（交互作用は断定しない）
N=2048: 現象が再現しない（bridge_v2=pass）。帰属は行わない
