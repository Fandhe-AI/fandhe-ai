cell                             before_med_s    after_med_s  ratio(after/before)  checksum_match   judged
train:fresh                       0.000781459    0.000805437               1.0307              OK      yes
train:reuse                       0.000940772    0.000869563               0.9243              OK      yes
infer:fresh                       0.000201937    0.000155605               0.7706              OK      yes
infer:reuse                       0.000190146    0.000162292               0.8535              OK      yes
gemm:fresh:512                    0.000671271    0.000660709               0.9843              OK       no
gemm:reuse:512                    0.000638167    0.000643062               1.0077              OK       no
gemm:fresh:1024                   0.002987167    0.002976042               0.9963              OK       no
gemm:reuse:1024                   0.004357084    0.003889521               0.8927              OK       no
gemm:fresh:2048                   0.019987500    0.023424041               1.1719              OK       no
gemm:reuse:2048                   0.023663562    0.025603458               1.0820              OK       no

checksum全体一致: OK

判定: train:fresh の ratio=1.0307 (>1.00) が事前登録規則「判定対象 4
セルすべてで ratio<=1.00」に抵触 → REJECT（SMALL_SHAPE_CAP_ENABLED は
false のまま）。
