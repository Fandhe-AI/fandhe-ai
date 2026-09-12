=== checksum consistency ===
train_64x256x784_nn: OK ({'0x40b128afdd627000'})
train_64x10x256_nn: OK ({'0xc0b7bdb5992a8000'})
train_784x256x64_tn_shape: OK ({'0xc0f371874f013d8a'})
train_64x256x10_nt_shape: OK ({'0x4077ef0ce9c39000'})
train_256x10x64_tn_shape: OK ({'0xc0b94c72f98d4000'})
square_128: OK ({'0x40ad57b384d1ed00'})
square_256: OK ({'0xc08ed163c5cea000'})
square_512: OK ({'0xc0e9b13f0ea78000'})

=== medians (5-run median of per-process median) and ratio vs global ===
-- train_64x256x784_nn (global median=134.75 us) --
   global         median=  134.75 us  ratio_vs_global=1.0000
   dedicated:2    median=  204.83 us  ratio_vs_global=1.5201
   dedicated:4    median=  142.29 us  ratio_vs_global=1.0560
   dedicated:6    median=  130.00 us  ratio_vs_global=0.9647
   dedicated:8    median=  138.42 us  ratio_vs_global=1.0272
   rayon_env_4    median=  143.96 us  ratio_vs_global=1.0683
-- train_64x10x256_nn (global median=41.29 us) --
   global         median=   41.29 us  ratio_vs_global=1.0000
   dedicated:2    median=   17.08 us  ratio_vs_global=0.4137
   dedicated:4    median=   15.62 us  ratio_vs_global=0.3784
   dedicated:6    median=   20.83 us  ratio_vs_global=0.5045
   dedicated:8    median=   34.96 us  ratio_vs_global=0.8466
   rayon_env_4    median=   15.00 us  ratio_vs_global=0.3633
-- train_784x256x64_tn_shape (global median=309.58 us) --
   global         median=  309.58 us  ratio_vs_global=1.0000
   dedicated:2    median=  281.25 us  ratio_vs_global=0.9085
   dedicated:4    median=  226.54 us  ratio_vs_global=0.7318
   dedicated:6    median=  223.21 us  ratio_vs_global=0.7210
   dedicated:8    median=  233.50 us  ratio_vs_global=0.7542
   rayon_env_4    median=  227.92 us  ratio_vs_global=0.7362
-- train_64x256x10_nt_shape (global median=76.25 us) --
   global         median=   76.25 us  ratio_vs_global=1.0000
   dedicated:2    median=   28.17 us  ratio_vs_global=0.3694
   dedicated:4    median=   27.33 us  ratio_vs_global=0.3585
   dedicated:6    median=   42.58 us  ratio_vs_global=0.5585
   dedicated:8    median=   52.42 us  ratio_vs_global=0.6874
   rayon_env_4    median=   25.58 us  ratio_vs_global=0.3355
-- train_256x10x64_tn_shape (global median=61.96 us) --
   global         median=   61.96 us  ratio_vs_global=1.0000
   dedicated:2    median=   16.83 us  ratio_vs_global=0.2717
   dedicated:4    median=   17.12 us  ratio_vs_global=0.2764
   dedicated:6    median=   29.17 us  ratio_vs_global=0.4707
   dedicated:8    median=   44.83 us  ratio_vs_global=0.7236
   rayon_env_4    median=   14.17 us  ratio_vs_global=0.2286
-- square_128 (global median=70.25 us) --
   global         median=   70.25 us  ratio_vs_global=1.0000
   dedicated:2    median=   56.75 us  ratio_vs_global=0.8078
   dedicated:4    median=   49.50 us  ratio_vs_global=0.7046
   dedicated:6    median=   56.17 us  ratio_vs_global=0.7995
   dedicated:8    median=   70.96 us  ratio_vs_global=1.0101
   rayon_env_4    median=   47.92 us  ratio_vs_global=0.6821
-- square_256 (global median=203.75 us) --
   global         median=  203.75 us  ratio_vs_global=1.0000
   dedicated:2    median=  247.83 us  ratio_vs_global=1.2164
   dedicated:4    median=  178.00 us  ratio_vs_global=0.8736
   dedicated:6    median=  161.00 us  ratio_vs_global=0.7902
   dedicated:8    median=  168.25 us  ratio_vs_global=0.8258
   rayon_env_4    median=  181.71 us  ratio_vs_global=0.8918
-- square_512 (global median=642.29 us) --
   global         median=  642.29 us  ratio_vs_global=1.0000
   dedicated:2    median= 1461.08 us  ratio_vs_global=2.2748
   dedicated:4    median=  884.96 us  ratio_vs_global=1.3778
   dedicated:6    median=  705.25 us  ratio_vs_global=1.0980
   dedicated:8    median=  604.96 us  ratio_vs_global=0.9419
   rayon_env_4    median=  892.21 us  ratio_vs_global=1.3891
