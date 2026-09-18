# PyTorch cpu N=4096 parity fail 要素の真値突合（イシュー #1985）

判定・tolerance 変更なし（机上計算のみ）。参照は bench_py.py と同じ「k 昇順に f64 加算 → f32 丸め」方式。 `S_A=0.5`・`S_B=0.49999994`・`u=2^-24`・現行 bound（c=0.5・線形 K）=3.051757e-05

## 集計（bench_py.py の JSONL 列と同名）

| total | fail_count | max_abs_err | max_rel_err | scaled_abs_bound | scaled_abs_rescued |
|---:|---:|---:|---:|---:|---:|
| 16777216 | 1 | 8.964539e-05 | 1.685575e+00 | 3.051757e-05 | 643 |

## fail 要素の真値突合

| idx | row | col | actual (bits) | ref (bits) | ref_fma_bit_match | exact truth | \|actual−truth\| | \|ref−truth\| | \|ref−actual\| | 真値に近い側 | pass(真 FMA 参照) |
|---:|---:|---:|---|---|---|---:|---:|---:|---:|---|---|
| 343838 | 83 | 3870 | -1.325470209e-02 (0xbc592a40) | -1.322097145e-02 (0xbc589cc6) | True | -1.325068847e-02 | 4.014e-06 | 2.972e-05 | 3.373e-05 | actual | False |

## 救済可否表（`d=|ref−actual| <= bound`。u=2^-24・S_A·S_B は大域 max）

| c | 形 | bound | idx=343838 |
|---:|---|---:|---|
| 0.5 | 線形 K（現行契約） | 3.051757e-05 | fail |
| 0.5 | √K | 4.768371e-07 | fail |
| 1.0 | 線形 K | 6.103515e-05 | 救済 |
| 1.0 | √K | 9.536742e-07 | fail |
| 1.5 | 線形 K | 9.155272e-05 | 救済 |
| 1.5 | √K | 1.430511e-06 | fail |

