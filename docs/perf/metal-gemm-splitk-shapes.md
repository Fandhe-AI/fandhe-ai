# Metal GEMM split-K 対象形状（K 支配的非正方）の劣化定量化（#810）

イシュー #810「perf(backend-metal): split-K ディスパッチ分岐の設計検討」の実測記録テンプレート。
`crate::tile::select`（`MetalGemm::dispatch_auto` が使う本番タイル選択。`crates/backend-metal/src/tile.rs:788-793`）
は tall／wide／正方立方／大形状の 4 分岐のみを持ち、K 方向は `TileConfig::bk`（K ループ刻み）にしか反映されない。
1 threadgroup が K 全域を直列にループする構造のため、M・N が小さく threadgroup 数が GPU コア数に対して不足する
形状では K がいくら大きくても並列度が上がらない。本ドキュメントはこの劣化を定量化し、split-K 専用経路導入の
採否判断材料を記録する（採否判断そのものは `docs/backend-metal-splitk-decision.md` §3）。

## 状態: 解析値算出は完了（Linux worktree）。**壁時計計測・採否確定は未実施（Mac 実機実行待ち）**

本イシューは「設計検討（調査・計測・記録）であり、`dispatch_auto`・シェーダの本番経路変更は行わない」
（イシュー #810 受け入れ条件）。実行環境は Mac 実機（`docs/real-hardware-verification-env.md` §1・§7
「ローカル直接実行」）だが本セッション環境は Linux のため、#487・#549 の先例に従い、Linux 側で完了できる
範囲（診断 example の実装・解析値の算出・doc の計測手順・判定基準の確定）のみを本 PR で行う。
**§4「実測結果」・§5「採否判断」は Mac 実機セッションで `cargo run -p backend-metal --example
gemm_splitk_shapes_bench --release` を実行してから記入する。**

## 1. 計測手段

`crates/backend-metal/examples/gemm_splitk_shapes_bench.rs`（本イシューで新規作成）。

- **解析値**（`analytics` モジュール。`objc2` 系 FFI に触れない純粋関数のため非 macOS でも実行できる）:
  `tile::select(m, n, k)` の選択結果から threadgroup 数（`actual_groups`）・K ループタイル数
  （`k_tile_count`）・MLX `steel_gemm_splitk_axpby`（Case 1・非 NAX）選択条件への該当有無
  （`mlx_case1_domain`。式の出典は `docs/backend-metal-splitk-decision.md` §1）を算出する
- **実測値**（macOS 限定。`macos_impl` モジュール）: `MetalGemm::dispatch_auto` を
  `bench-harness::protocol::run`（`MeasurementConfig::default()` = warmup 20 回・計測 20 回・中央値/Q1/Q3。
  TASK-8.1）で壁時計計測する。`crate::pipeline::make_pipeline_with_constants` が `pub(crate)` で example
  から呼べないため（#487 と同一の制約）、転送時間の分離は試みず `wall_ms` を A・B アップロード＋カーネル
  実行＋C readback の end-to-end 時間として報告し、`tflops_lower_bound`（転送時間は非負という不等式のみ
  から導かれる健全な下限値）を算出する

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

## 3. 解析値（Linux 算出。2026-08-21 実行・`cargo run -p backend-metal --example gemm_splitk_shapes_bench --release`）

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

## 4. 実測結果（記入欄。Mac 実機セッションで記入）

`cargo run -p backend-metal --example gemm_splitk_shapes_bench --release`（既定 20/20 計測。ノイズが
大きい場合は `--iters=200` を付与）の出力を転記する。

- 実行コミット SHA: `______`
- 実機: `______`（`docs/real-hardware-verification-env.md` §1 準拠。想定 M4 Max）
- 実行日時: `______`

| target (M,N,K) | target wall_ms (median) | target tflops_lower_bound | control (S,S,S) | control wall_ms (median) | control tflops_lower_bound | 劣化率 (target/control) |
|---|---|---|---|---|---|---|
| (32,32,2048) | | | (128,128,128) | | | |
| (32,32,4096) | | | (160,160,160) | | | |
| (32,32,8192) | | | (200,200,200) | | | |
| (64,64,2048) | | | (200,200,200) | | | |
| (64,64,4096) | | | (256,256,256) | | | |
| (64,64,8192) | | | (320,320,320) | | | |
| (128,128,2048) | | | (320,320,320) | | | |
| (128,128,4096) | | | (408,408,408) | | | |
| (128,128,8192) | | | (512,512,512) | | | |
| (256,256,2048) | | | (512,512,512) | | | |
| (256,256,4096) | | | (648,648,648) | | | |
| (256,256,8192) | | | (816,816,816) | | | |

## 5. 判定基準（計測前に事前定義。ベンチ判定基準であり、ガードレール閾値・テスト許容誤差とは別軸）

以下をいずれも満たす形状が実測 12 点中で有意な割合（目安: 過半数）を占める場合、split-K 導入を
「採用検討推奨」と判定する（確定的な採用可否は別途実装 issue でのユーザー承認を要する。
`.claude/rules/out-of-scope-tracking.md`）:

1. **劣化率**: 中央値ベースで `target tflops_lower_bound / control tflops_lower_bound < 0.7`
   （同程度 FLOPs の正方立方形状比で 30% 以上の劣化）
2. **並列度不足の解析裏付け**: `actual_groups < 40`（実機 GPU コア数。§3 参照）

いずれも満たさない、または実測ばらつき（Q1/Q3）が判定を左右する水準の場合は「不採用（現状維持）」
とし、その根拠を §6 に記録する。本判定基準は本イシューが新規に事前登録する数値であり、
`guardrail.toml`・バックエンド間数値一致テストの許容誤差（`.claude/rules/coding-rust.md`）とは
無関係である（`.claude/rules/security.md`「自己修復ループ固有のガードレール」の対象外）。

## 6. 採否判断（記入欄。§4 実測結果を受けて確定する）

`______`

## 7. 参照

- `crates/backend-metal/examples/gemm_splitk_shapes_bench.rs`（本イシュー新規作成）
- `crates/backend-metal/src/tile.rs:682-831`（`select`／`select_with_occupancy`。K 方向分岐が
  存在しないことの根拠）
- `docs/backend-metal-splitk-decision.md`（MLX 選択条件との対比・採用時の設計方針の記録）
- `docs/perf/metal-gemm-bottleneck-diagnosis.md`（#487。同型の記録テンプレート・実測部分の設計判断の
  先例）
- `docs/perf/metal-bench-noise-protocol.md`（サーマルドリフト対策・順序バイアス相殺）
- `docs/real-hardware-verification-env.md` §1（実機検証環境）
- イシュー #810・親系列 #479（GEMM 最適化）
