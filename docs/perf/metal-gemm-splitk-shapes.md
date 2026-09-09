# Metal GEMM split-K 対象形状（K 支配的非正方）の劣化定量化（#810）

イシュー #810「perf(backend-metal): split-K ディスパッチ分岐の設計検討」の実測記録テンプレート。
`crate::tile::select_for_device`（`MetalGemm::dispatch_auto` が使う本番タイル選択。`crates/backend-metal/src/tile.rs:1183`。#1308 で #1039 等の後続変更による行番号移動を反映して更新）
は tall／wide／正方立方／大形状の 4 分岐のみを持ち、K 方向は `TileConfig::bk`（K ループ刻み）にしか反映されない。
1 threadgroup が K 全域を直列にループする構造のため、M・N が小さく threadgroup 数が GPU コア数に対して不足する
形状では K がいくら大きくても並列度が上がらない。本ドキュメントはこの劣化を定量化し、split-K 専用経路導入の
採否判断材料を記録する（採否判断そのものは `docs/backend-metal-splitk-decision.md` §3）。

## 状態: M4 Max 実機実測完了（イシュー #1308）。**採否確定: 採用検討推奨**

本イシューは「設計検討（調査・計測・記録）であり、`dispatch_auto`・シェーダの本番経路変更は行わない」
（イシュー #810 受け入れ条件）。実行環境は Mac 実機（`docs/real-hardware-verification-env.md` §1・§7
「ローカル直接実行」）。#810 の解析値算出（Linux worktree）を受け、イシュー #1308 で M4 Max 実機（本エージェント
実行環境）にて `cargo run -p fandhe-ai-backend-metal --example gemm_splitk_shapes_bench --release` を
5 プロセス起動・5 回計測中央値で実行し、§4「実測結果」・§6「採否判断」を確定記録した。事前登録した §3.3
判定規則（下記）を機械的に適用した結果、対象 12 点中 9 点が「並列度不足の解析裏付け」（条件 2）に該当し、
うち 9 点全点で「劣化率 <0.7・5/5 run 一貫」（条件 1）も成立したため、**採用検討推奨**（過半数 7 点以上の
基準を大きく上回る 9/9）と判定した。split-K 実装自体は本イシューでは行わず、`docs/backend-metal-
splitk-decision.md` §3 の確定判断・別 issue 起票案（PR 本文で提示。ユーザー承認後に起票）へ引き継ぐ。

## 1. 計測手段

`crates/backend-metal/examples/gemm_splitk_shapes_bench.rs`（本イシューで新規作成）。

- **解析値**（`analytics` モジュール。`objc2` 系 FFI に触れない純粋関数のため非 macOS でも実行できる）:
  `tile::select(m, n, k)` の選択結果から threadgroup 数（`actual_groups`）・K ループタイル数
  （`k_tile_count`）・MLX `steel_gemm_splitk_axpby`（Case 1・非 NAX）選択条件への該当有無
  （`mlx_case1_domain`。式の出典は `docs/backend-metal-splitk-decision.md` §1）を算出する
- **実測値**（macOS 限定。`macos_impl` モジュール）: `MetalGemm::dispatch_tiled_prepared`（§4 準拠
  prepared 入口。イシュー #572）を使い、A・B バッファの確保・アップロードを計測ループの外で 1 回だけ
  行い、計測対象をエンコード＋コマンドバッファ完了待ちのみに限定する（readback も対象外）。対象
  （K 支配的非正方）・対照（正方立方）は総 FLOPs をほぼ揃えていても A・B の転送要素数は対象側が
  常に多く（2.0〜6.5536 倍。`gemm_splitk_shapes_bench.rs` モジュールドキュメント参照）、`dispatch_auto`
  （アップロード・readback を含む end-to-end 境界）で判定すると転送量差だけで §5 の判定基準
  （`< 0.7`）を跨ぎうるため（codex-review 指摘対応。#810 PR #829）、prepared 境界を採用した。
  併せて `bench_harness::ab::run_ab`（`gemm_swizzle_ab_bench.rs` フェーズ 2 と同型）で対象（side A）・
  対照（side B）をラウンド（`ROUNDS=6`）ごとに順序反転した interleaved 計測にし、サーマル状態・GPU
  クロック（DVFS）変動の順序バイアスをラウンド間で相殺する（`docs/perf/metal-bench-noise-protocol.md`
  参照。同じく codex-review 指摘対応。#810 PR #829）

## 2. 対象形状群

- **対象（K 支配的非正方。`M == N`）**: `(M,N) ∈ {32, 64, 128, 256}` × `K ∈ {2048, 4096, 8192}` の
  全 12 点
- **対照（同程度 FLOPs の正方立方形状）**: `analytics::matched_cube_side(m,n,k)` で `2*M*N*K` に最も
  近い `2*S^3`（`S` は 8 の倍数）を与える `S` を算出し `(S,S,S)` と比較する。TFLOPS は各形状自身の
  実 FLOPs で正規化する比率指標のため、`S` の丸めによる FLOPs のわずかな差異（下表 `flops_ratio` 列。
  実測範囲は 0.9537〜1.0136）は比較の妥当性を損なわない
- **除外**: `M=1` 等の gemv 領域は対象外（別軸の課題。#811 が CPU 側で扱う論点と同族）
- `M=32` は `tile::select` の `SMALL`（64）閾値未満のため `TileConfig::SINGLE_SIMDGROUP_8X8`
  （`bm=bn=bk=8`・単一 simdgroup）に縮退する既存挙動を確認する対照点として残す（本表の該当行）

## 3. 解析値（Linux 算出。2026-08-21 実行・`cargo run -p fandhe-ai-backend-metal --example gemm_splitk_shapes_bench --release`）

| target (M,N,K) | tile (bm×bn×bk, wm×wn) | actual_groups | k_tile_count | target MLX Case1 該当 | control (S,S,S) | control actual_groups | control MLX Case1 該当 | flops_ratio |
|---|---|---|---|---|---|---|---|---|
| (32,32,2048) | 8x8x8 (1x1) | 16 | 256 | true | (128,128,128) | 16 | true | 1.0000 |
| (32,32,4096) | 8x8x8 (1x1) | 16 | 512 | true | (160,160,160) | 25 | true | 0.9766 |
| (32,32,8192) | 8x8x8 (1x1) | 16 | 1024 | true | (200,200,200) | 49 | true | 0.9537 |
| (64,64,2048) | 32x32x16 (2x2) | 4 | 128 | true | (200,200,200) | 49 | true | 0.9537 |
| (64,64,4096) | 32x32x16 (2x2) | 4 | 256 | true | (256,256,256) | 64 | true | 1.0000 |
| (64,64,8192) | 32x32x16 (2x2) | 4 | 512 | true | (320,320,320) | 100 | true | 0.9766 |
| (128,128,2048) | 32x32x16 (2x2) | 16 | 128 | true | (320,320,320) | 100 | true | 0.9766 |
| (128,128,4096) | 32x32x16 (2x2) | 16 | 256 | true | (408,408,408) | 169 | true | 1.0120 |
| (128,128,8192) | 32x32x16 (2x2) | 16 | 512 | true | (512,512,512) | 256 | true | 1.0000 |
| (256,256,2048) | 32x32x16 (2x2) | 64 | 128 | true | (512,512,512) | 256 | true | 1.0000 |
| (256,256,4096) | 32x32x16 (2x2) | 64 | 256 | true | (648,648,648) | 441 | **false** | 1.0136 |
| (256,256,8192) | 32x32x16 (2x2) | 64 | 512 | true | (816,816,816) | 676 | **false** | 1.0120 |

**観察（解析値のみからの一次所見）**:

- **対象形状 12 点は全点** MLX の split-K 選択域（Case 1。`tm*tn<=min_tmn_threshold && tk>=8 &&
  k>=max(m,n)`。対象形状の `tm*tn` 最大は 256〈(256,256,*)〉であり `min_tmn_threshold` が
  1024／2048 いずれの分岐でも判定は変わらない — `devc`〈`docs/backend-metal-splitk-decision.md`
  §1〉の断定不能性は対象形状側の結論には影響しない）に該当する。一方、対照（正方立方）の 2 点は
  `min_tmn_threshold` の分岐で結果が変わる: `min_tmn_threshold=1024`（本実装が採用する前提。
  §1「解析値」節の実行結果）では `(648,648,648)`〈`tm*tn=41*41=1681`〉・`(816,816,816)`
  〈`tm*tn=51*51=2601`〉のいずれも閾値超過で域外になるが、`min_tmn_threshold=2048`（`devc` が
  `'s'`／`'d'` の場合）では `(648,648,648)`（1681≤2048）が域内に転じ、`(816,816,816)`
  （2601>2048）のみが域外のまま残る。したがって「対照は域外になる」という対比は
  `min_tmn_threshold=1024` の前提下でのみ成立する一次所見であり、`devc` 分岐の実機確認（本実装
  実機環境が `'s'`／`'d'` 系列に該当しないこと）を伴わない限り確定的な裏付けとしては扱わない
- 対象形状の `actual_groups`（4〜64）は実機検証環境（M4 Max・GPU コア 40。
  `docs/perf/metal-gemm-dynamic-tile.md:53`）のコア数を下回るか同程度に留まる点が多く、特に
  `(64,64,*)`（`actual_groups=4`）は 40 コアに対し著しく過小（コア稼働率の観点で 1/10 程度）
  であり、K 方向を分割して threadgroup 数を増やす split-K の理論的な有効性を示唆する
- 対照（正方立方）は同程度 FLOPs でも `actual_groups` が対象より多い（例: (64,64,4096) の対象
  `actual_groups=4` に対し対照 (256,256,256) は `actual_groups=64`）。同じ FLOPs でも形状によって
  並列度が大きく異なることが定量的に確認できる

## 4. 実測結果（M4 Max 実機実測。イシュー #1308・2026-09-09）

`cargo run -p fandhe-ai-backend-metal --example gemm_splitk_shapes_bench --release`（既定 `ROUNDS=6`・warmup 20
回・計測 20 回。`ROUNDS` 単位で `dispatch_tiled_prepared` を interleaved 計測するため、`--iters=N` の
引き上げは 12 形状組 × `ROUNDS` × 2 side 分の実行時間に直接乗る点に注意する。ノイズが大きい場合は
`--iters=200` より先に `docs/perf/metal-bench-noise-protocol.md` の cooldown／ROUNDS 調整手順を検討
すること）の出力（`target_tflops`／`control_tflops`／`target_over_control`／`spread_target`／
`spread_control`）を転記する。

- 実行コミット SHA: `60dc70a75e873382322b615b0c96e3236331f1cb`（base main。ブランチ
  `perf/1308-metal-splitk-shapes-measure`）
- 実機: Apple M4 Max（`docs/real-hardware-verification-env.md` §1 準拠。macOS 26.6.2 / rustc 1.96.0）
- 実行日時: 2026-09-09（UTC。run1〜run5 の各起動時刻は
  `docs/perf/logs/metal-gemm-splitk-shapes-1308/env_info.txt` 参照）
- **プロトコル**: `run{1..5}.log`（5 プロセス起動。各起動内部で `ROUNDS=6`・warmup/計測各 20 回の
  `run_ab` interleaved 計測）。集計は `docs/perf/logs/metal-gemm-splitk-shapes-1308/aggregate.py`
  （python3 標準ライブラリのみ。`--self-test` 済み）。`target_over_control` は各 run 内の対応比
  （run 内比の 5 run 中央値。下表 `target_over_control` 列）を主指標とし、両群中央値の比
  （`target_tflops` 5 run 中央値 ÷ `control_tflops` 5 run 中央値）は参考指標として併記する
  （PR #1389/#1392 codex-review 指摘の踏襲。両者の乖離が大きい行は run 間のばらつきが大きいことを
  示す — §3.1 の注記参照）
- **負荷ゲート**: 各 run 起動前に 1 分 load average <= 4.0 を確認（超過時は待機・再試行。
  `env_info.txt` に全試行を記録）。run3・run5 で 1 回ずつ超過を検出し待機後に再試行して通過
  （run3: 4.91→3.42、run5: 6.80→4.60→2.84）。それ以外は初回試行で通過（load average 1.76〜2.30）
- **§3.1 注記（重要）**: 下表 `control (S,S,S)` 列は §3 解析値算出時の `tile::select`（機種ゲートなし）
  基準の対照タイルを転記したものだが、実測ディスパッチは `select_for_device`（M4 Max 実測 GPU コア数
  検証込み）を使うため、一部の対照形状（`(512,512,512)`）で実際に選択されるタイルが異なる
  （解析値: `32x32x16`・`actual_groups=256` に対し、実測 resolved: `64x32x32`・
  `resolved_actual_groups=128`。`smoke_run.log`・`run1.log` 等の `resolved_tile=...
  resolved_actual_groups=...` 行で確認可能）。対象形状 12 点は `tile::select` と `select_for_device`
  が常に一致する形状クラス（厳密一致テーブルの発火域外）のため対象側には影響しないが、対照
  `(512,512,512)`（`(128,128,8192)`・`(256,256,2048)` の対照）の実測並列度は表記の 256 ではなく
  128 である点に注意する

| target (M,N,K) | target_tflops (median) | control (S,S,S) | control_tflops (median) | target_over_control (median) | spread_target (max) | spread_control (max) |
|---|---|---|---|---|---|---|
| (32,32,2048) | 0.0193 | (128,128,128) | 0.0389 | 0.5030 | 1.4377 | 0.5350 |
| (32,32,4096) | 0.0275 | (160,160,160) | 0.0736 | 0.3590 | 0.8078 | 0.5333 |
| (32,32,8192) | 0.0346 | (200,200,200) | 0.1417 | 0.2418 | 0.7814 | 0.6085 |
| (64,64,2048) | 0.0718 | (200,200,200) | 0.1431 | 0.5016 | 1.4075 | 18.6197 |
| (64,64,4096) | 0.1008 | (256,256,256) | 0.2947 | 0.3243 | 0.7815 | 4.0540 |
| (64,64,8192) | 0.1241 | (320,320,320) | 0.5496 | 0.2265 | 0.3924 | 5.4787 |
| (128,128,2048) | 0.2816 | (320,320,320) | 0.5601 | 0.4661 | 1.4747 | 0.8196 |
| (128,128,4096) | 0.3996 | (408,408,408) | 1.0925 | 0.3755 | 1.1405 | 0.8428 |
| (128,128,8192) | 0.4940 | (512,512,512) | 0.9005 | 0.5166 | 1.3702 | 1.2848 |
| (256,256,2048) | 1.1463 | (512,512,512) | 1.0134 | 0.6117 | 0.6245 | 0.7837 |
| (256,256,4096) | 1.0556 | (648,648,648) | 1.2149 | 0.8069 | 0.8822 | 1.4932 |
| (256,256,8192) | 1.9292 | (816,816,816) | 4.0235 | 0.4974 | 0.5584 | 1.9057 |

**spread（`bench_harness::ab::STABILITY_SPREAD_GATE`＝0.05 目安）は全 12 点で超過している**（実行時の
1 分 load average が 1.76〜6.80 で推移する共有負荷環境下の計測であり単発スパイクの影響を受けやすい。
`docs/perf/metal-bench-noise-protocol.md` 参照）。ただし §3.3 の事前登録判定は「5 run 中央値 <0.7 かつ
5/5 run すべて <0.7」という run 間符号一貫性を要求しており、対象 9 点（`(256,256,*)` を除く全点）は
いずれも 5 run すべてが 0.7 を明確に下回る（run 別生値・詳細判定は下記「5 run 生値・機械判定」節、または
`docs/perf/logs/metal-gemm-splitk-shapes-1308/aggregate.md` を参照）。spread 超過はノイズの大きさを示す
ものの、この 9 点については 0.7 の閾値を跨ぐような境界事例ではなく（中央値は最大でも 0.5166、最小
0.2265 と閾値から明確に離れている）、§5「実測ばらつきが判定を左右する水準の場合は不採用」の除外規定には
該当しないと判断した。一方 `(256,256,*)` 3 点は条件 2（`actual_groups=64 >= 40`）が非該当のため、
spread 超過の議論以前に判定対象から除外される。

### 5 run 生値・機械判定（`docs/perf/logs/metal-gemm-splitk-shapes-1308/aggregate.md` より転記）

| target (M,N,K) | control (S,S,S) | target_over_control (5 run) | median (alt: 両群中央値の比) | all runs <0.7 | actual_groups(target) | 条件1 | 条件2 | 判定対象 |
|---|---|---|---|---|---|---|---|---|
| (32,32,2048) | (128,128,128) | 0.4508,0.5030,0.4847,0.5104,0.5869 | 0.5030 (alt=0.4961) | yes | 16 | ○ | ○ | candidate |
| (32,32,4096) | (160,160,160) | 0.3369,0.3383,0.3903,0.3740,0.3590 | 0.3590 (alt=0.3736) | yes | 16 | ○ | ○ | candidate |
| (32,32,8192) | (200,200,200) | 0.1851,0.2418,0.2542,0.2403,0.2521 | 0.2418 (alt=0.2442) | yes | 16 | ○ | ○ | candidate |
| (64,64,2048) | (200,200,200) | 0.4941,0.5074,0.4992,0.5016,0.5099 | 0.5016 (alt=0.5017) | yes | 4 | ○ | ○ | candidate |
| (64,64,4096) | (256,256,256) | 0.3067,0.2283,0.3582,0.3243,0.3631 | 0.3243 (alt=0.3420) | yes | 4 | ○ | ○ | candidate |
| (64,64,8192) | (320,320,320) | 0.2276,0.2318,0.2228,0.1982,0.2265 | 0.2265 (alt=0.2258) | yes | 4 | ○ | ○ | candidate |
| (128,128,2048) | (320,320,320) | 0.5027,0.3201,0.4661,0.2265,0.5010 | 0.4661 (alt=0.5028) | yes | 16 | ○ | ○ | candidate |
| (128,128,4096) | (408,408,408) | 0.3803,0.4190,0.3537,0.3649,0.3755 | 0.3755 (alt=0.3658) | yes | 16 | ○ | ○ | candidate |
| (128,128,8192) | (512,512,512) | 0.2723,0.5166,0.2678,0.5486,0.5989 | 0.5166 (alt=0.5486) | yes | 16 | ○ | ○ | candidate |
| (256,256,2048) | (512,512,512) | 1.1794,0.6117,0.6199,0.6114,0.6100 | 0.6117 (alt=1.1311) | no | 64 | × | × | - |
| (256,256,4096) | (648,648,648) | 1.2194,0.8069,0.6312,0.5265,0.9481 | 0.8069 (alt=0.8689) | no | 64 | × | × | - |
| (256,256,8192) | (816,816,816) | 0.4885,1.1907,0.4974,0.4795,1.1154 | 0.4974 (alt=0.4795) | no | 64 | × | × | - |

条件2（並列度不足の解析裏付け。`actual_groups < 40`）該当: 9 / 12 点。うち条件1（劣化率中央値 <0.7
かつ 5/5 run 一貫）も成立: **9 / 9**（事前登録した「過半数（目安 7 点以上）」基準を明確に上回る）。

## 5. 判定基準（計測前に事前定義。ベンチ判定基準であり、ガードレール閾値・テスト許容誤差とは別軸）

以下をいずれも満たす形状が実測 12 点中で有意な割合（目安: 過半数）を占める場合、split-K 導入を
「採用検討推奨」と判定する（確定的な採用可否は別途実装 issue でのユーザー承認を要する。
`.claude/rules/out-of-scope-tracking.md`）:

1. **劣化率**: 中央値ベースで `target_tflops / control_tflops < 0.7`（`target_over_control` 列。
   同程度 FLOPs の正方立方形状比で 30% 以上の劣化）
2. **並列度不足の解析裏付け**: `actual_groups < 40`（実機 GPU コア数。§3 参照）

いずれも満たさない、または実測ばらつき（`spread_target`／`spread_control`。
`bench_harness::ab::STABILITY_SPREAD_GATE`＝0.05 を目安とする）が判定を左右する水準の場合は
「不採用（現状維持）」とし、その根拠を §6 に記録する。本判定基準は本イシューが新規に事前登録する数値であり、
`guardrail.toml`・バックエンド間数値一致テストの許容誤差（`.claude/rules/coding-rust.md`）とは
無関係である（`.claude/rules/security.md`「自己修復ループ固有のガードレール」の対象外）。

## 6. 採否判断（§4 実測結果を受けて確定。イシュー #1308・2026-09-09）

**採用検討推奨。** §3.3 で事前登録した判定規則（条件1: 劣化率中央値 <0.7 かつ 5/5 run 一貫、条件2:
`actual_groups < 40`。両方成立する形状が 12 点中過半数〈目安 7 点以上〉で「採用検討推奨」）を機械的に
適用した結果、条件2 該当 9 点（`(32,32,*)`・`(64,64,*)`・`(128,128,*)` の全 9 点。`(256,256,*)` 3 点は
`actual_groups=64 >= 40` のため対象外）のうち **9 点全点**で条件1も成立した（9/9 が事前登録閾値
7/12 を明確に上回る）。劣化率中央値は 0.2265〜0.5166 の範囲に収まり、いずれも 0.7 を大きく下回る
（境界事例なし）。

`docs/backend-metal-splitk-decision.md` §2 に記録済みの設計方針（2 パス方式・固定順序縮約・
`tile.rs` への純粋関数分離・境界検査維持・fail-closed フォールバック）に基づき、split-K 実装を別
issue へ切り出すことを提案する（`.claude/rules/out-of-scope-tracking.md`。起票案は本 PR の PR 本文へ
記載し、実際の起票はユーザー承認後に行う）。本イシュー自体は「調査・計測・記録」の範囲を超えないため
`crates/backend-metal/src/`・`shaders/gemm.metal` は変更しない。

**留保事項**: 実測は共有負荷環境（1 分 load average 1.76〜6.80 で推移）下で行っており、spread（0.05
目安）は全 12 点で超過している。ただし判定対象 9 点は 5 run すべてが 0.7 を明確に下回る一貫した符号を
示しており（§4「5 run 生値・機械判定」節参照）、ばらつきが 0.7 の閾値判定そのものを左右する境界事例
ではないと判断した。より低負荷な排他環境での再測は split-K 実装 issue 側での性能検証（結線前後
比較）時に併せて行うことが望ましい。

## 7. 参照

- `crates/backend-metal/examples/gemm_splitk_shapes_bench.rs`（本イシュー #810 新規作成・#1308 で
  resolved タイル出力〈`print_resolved_tile_line`〉を追加）
- `crates/backend-metal/src/tile.rs:1165`（`select`）・`crates/backend-metal/src/tile.rs:1183`
  （`select_for_device`。`crate::gemm::MetalGemm::dispatch_auto` の本番入口）・
  `crates/backend-metal/src/tile.rs:1239`（`select_with_occupancy_for_device`。4 分岐 match を含む
  occupancy 判定込み選択。K 方向の threadgroup 分割経路を持たないことの根拠。旧版〈#810 記録時点〉の
  行番号 `682-831` から #1039〈厳密一致テーブル追加〉等の後続変更で移動しているため #1308 で現行行番号へ
  更新した）
- `docs/backend-metal-splitk-decision.md`（MLX 選択条件との対比・採用時の設計方針・§3 採否判断の記録）
- `docs/perf/metal-gemm-bottleneck-diagnosis.md`（#487。同型の記録テンプレート・実測部分の設計判断の
  先例）
- `docs/perf/metal-bench-noise-protocol.md`（サーマルドリフト対策・順序バイアス相殺）
- `docs/perf/logs/metal-gemm-splitk-shapes-1308/`（本実測の生ログ・`aggregate.py`／`aggregate.md`・
  `env_info.txt`。イシュー #1308）
- `docs/real-hardware-verification-env.md` §1（実機検証環境）
- イシュー #810・#1308（M4 Max 実機実測・採否確定）・親 #1274（Phase 3 ゲート再判定）・
  ルート #1269・親系列 #479（GEMM 最適化）
