# #4150 集計（record_only の内訳。2 系列。tree_sha=d50251b385a32f0383c4f2b60a287b84cf0f95d8）

- status: complete
- round 状態: PASS PASS PASS PASS PASS（PASS 5 件を使用）
- 値の単位は µs。round 別中央値の中央値と NB-RANGE5（最小〜最大）。タグ: 実測／推定
- 2 系列（#4188）: seq3189 は P・E 腕（#3189 列）、prod0140 は PR・ER 腕（0.14.0 本番の経路）。分類は系列ごとに独立。E_sum と F 腕は共通

## 系列 seq3189: #3189 列（層ごとに確保＋ゼロ初期化・入力 H2D あり。§12 の分解の再現）

- 分類: **GPU 実行が支配**（確度: 高）
- 区間: E_sum（l1_iso＋l2_iso）は into 経路の融合カーネル幅で出力のゼロ初期化を含まない。chain_gpu・l1_issue／l2_issue は upload 後に計時し、層ごとの出力確保＋ゼロ初期化（#3189 の呼び出し列）と drop を含むため、その分は gpu_gap・share_wait 側に入る。入力 H2D は h2d_upload・up_gpu の別項目で、H2D を含むのはチェーン全体の total のみ（RULE 項 4・5）

| 量 | タグ | 中央値 | 最小 | 最大 |
|---|---|---|---|---|
| kernel_wait_derived | 実測 | 63.672 | 63.343 | 63.824 |
| chain_gpu | 実測 | 74.368 | 74.336 | 74.592 |
| gpu_gap | 実測 | 3.472 | 3.168 | 3.536 |
| host_issue_sum | 実測 | 13.775 | 13.744 | 14.112 |
| gpu_tail | 推定 | 60.577 | 60.480 | 60.688 |
| wait_resid | 推定 | 3.095 | 2.655 | 3.248 |
| share_exec | 推定 | 1.1182 | 1.1101 | 1.1207 |
| share_wait | 推定 | -0.1182 | -0.1207 | -0.1101 |
| event_perturbation | 実測 | -0.0009 | -0.0014 | -0.0002 |
| l1_span | 実測 | 54.048 | 54.016 | 54.272 |
| l2_span | 実測 | 20.320 | 20.288 | 20.320 |
| download | 実測 | 69.375 | 69.112 | 69.512 |
| download_again | 実測 | 5.760 | 5.664 | 5.952 |
| l1_issue | 実測 | 7.143 | 7.104 | 7.344 |
| l2_issue | 実測 | 6.640 | 6.608 | 6.768 |
| total | 実測 | 94.047 | 93.679 | 94.671 |
| up_gpu | 実測 | 11.200 | 11.008 | 11.328 |
| h2d_upload | 実測 | 10.896 | 10.768 | 10.968 |

## 系列 prod0140: 本番列（0.14.0 の predict_device_chain_resident 2 回目以降相当。常駐入力・層出力を into 経路で再利用）

- 分類: **GPU 実行が支配**（確度: 高）
- 区間: 区間内に確保・ゼロ初期化・H2D・drop を含まない（常駐入力・事前確保した層出力への into 経路）。gpu_gap は発行待ちによる GPU の空きだけで、層出力のゼロ初期化を含まない（RULE 項 4・5）

| 量 | タグ | 中央値 | 最小 | 最大 |
|---|---|---|---|---|
| kernel_wait_derived | 実測 | 62.328 | 62.223 | 62.760 |
| chain_gpu | 実測 | 71.152 | 70.928 | 71.248 |
| gpu_gap | 実測 | 0.096 | -0.192 | 0.400 |
| host_issue_sum | 実測 | 5.744 | 5.719 | 5.824 |
| gpu_tail | 推定 | 65.328 | 65.184 | 65.512 |
| wait_resid | 推定 | -3.000 | -3.193 | -2.449 |
| share_exec | 推定 | 1.1369 | 1.1307 | 1.1443 |
| share_wait | 推定 | -0.1369 | -0.1443 | -0.1307 |
| event_perturbation | 実測 | -0.0015 | -0.0021 | 0.0020 |
| l1_span | 実測 | 50.848 | 50.608 | 50.928 |
| l2_span | 実測 | 20.320 | 20.288 | 20.320 |
| download | 実測 | 68.399 | 68.288 | 68.639 |
| download_again | 実測 | 6.032 | 5.728 | 6.144 |
| l1_issue | 実測 | 2.912 | 2.880 | 2.960 |
| l2_issue | 実測 | 2.848 | 2.800 | 2.864 |
| total | 実測 | 74.119 | 74.031 | 74.423 |

## 共通量（I 腕・F 腕。両系列で共通）

| 量 | タグ | 中央値 | 最小 | 最大 |
|---|---|---|---|---|
| E_sum | 実測 | 71.040 | 70.848 | 71.200 |
| l1_iso | 実測 | 50.624 | 50.496 | 50.800 |
| l2_iso | 実測 | 20.400 | 20.352 | 20.448 |
| sync_idle | 実測 | 0.288 | 0.288 | 0.304 |
| tiny_roundtrip | 実測 | 9.976 | 9.912 | 10.112 |
| launch_floor_1x1 | 実測 | 2.880 | 2.848 | 2.912 |
| event_record_pair | 実測 | 2.752 | 2.736 | 2.800 |

## 本番列と #4149（PyTorch CUDA）の対応

| fandhe の量 | 節 | 相手の item | fandhe | 相手 | 差（実測の差） |
|---|---|---|---|---|---|
| host_issue_sum | prod0140 | issue | 5.744 | 27.940 | -22.196 |
| download | prod0140 | readout_sync | 68.399 | 7.360 | 61.039 |
| chain_gpu | prod0140 | gpu_span_event | 71.152 | 33.180 | 37.972 |
| l1_iso | common | mm_event_l1_fwd | 50.624 | 14.700 | 35.924 |
| l2_iso | common | mm_event_l2_fwd | 20.400 | 12.450 | 7.950 |
| sync_idle | common | sync_idle | 0.288 | 2.490 | -2.202 |
| tiny_roundtrip | common | tiny_roundtrip | 9.976 | 10.200 | -0.224 |
| launch_floor_1x1 | common | launch_floor_add | 2.880 | 3.240 | -0.360 |
| event_record_pair | common | event_record_pair | 2.752 | 8.510 | -5.758 |

参考（#3189 列と #4149 の対応。#3189 列は入力 H2D と層出力のゼロ初期化を含むため差を内訳として読まない）

| fandhe の量 | 節 | 相手の item | fandhe | 相手 | 差（実測の差） |
|---|---|---|---|---|---|
| host_issue_sum | seq3189 | issue | 13.775 | 27.940 | -14.165 |
| download | seq3189 | readout_sync | 69.375 | 7.360 | 62.015 |
| chain_gpu | seq3189 | gpu_span_event | 74.368 | 33.180 | 41.188 |

非対称（本番列）: fandhe の l1_iso／l2_iso は bias＋ReLU 融合込みの同じカーネル（into 経路）、PyTorch の mm_event は純 GEMM。fandhe の窓は facade 上位の固定費を含まない。launch_floor_1x1（1x1 linear の単発発行）と launch_floor_add（1 要素 add 100 回の 1 回あたり）は区間定義が異なる参考の並びで、差を内訳として読まない。
非対称（#3189 列のみ）: fandhe の窓は H2D を含み PyTorch 側は X が device 上で H2D を含まない。fandhe の chain_gpu は層出力のゼロ初期化を含む（PyTorch の caching allocator は層ごとのゼロ初期化をしない）。
