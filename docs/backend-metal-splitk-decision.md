# Metal GEMM split-K ディスパッチ分岐: MLX 選択条件との対比・採否判断（#810）

イシュー #810「perf(backend-metal): split-K ディスパッチ分岐の設計検討」に対応する。ルート #479
（GEMM 性能改善）系列。`docs/backend-metal-mlx-classic-nax-decision.md`（#549）・
`docs/backend-metal-aligned-load-decision.md`（#752）と同型の決定記録として、(1) MLX の split-K
選択条件（非 NAX・Case 1）と本実装 `crate::tile::select` の構造対比、(2) 採用する場合の設計方針
（記録のみ・実装は別 issue）、(3) 採否判断（実測待ち）を残す。**本イシューは設計検討（調査・計測・
記録）であり、`crates/backend-metal/src/`・`shaders/gemm.metal` は一切変更しない。**

## 判断サマリ

**採用検討推奨（M4 Max 実機実測完了。イシュー #1308・2026-09-09。`docs/perf/metal-gemm-splitk-shapes.md`
§4・§6）。** 対象形状（K 支配的非正方。M=N=32〜256・K=2048〜8192）12 点中、並列度不足の解析裏付け
（`actual_groups < 40`）に該当する 9 点（`(32,32,*)`・`(64,64,*)`・`(128,128,*)`）**全点**で、5 プロセス
起動 5 回計測中央値による劣化率（対象/対照 TFLOPS 比）が事前登録判定基準の `< 0.7` を満たし、かつ 5/5
run すべてで一貫して下回った（劣化率中央値 0.2265〜0.5166）。事前登録した「過半数（目安 7 点以上）で
採用検討推奨」の基準を 9/9 で明確に上回ったため、split-K 導入は**採用検討推奨**と判定する。ただし
確定的な採用可否（実装の是非）は本ドキュメント §2 の設計方針に基づく別 issue でのユーザー承認を要する
（`.claude/rules/out-of-scope-tracking.md`）。本実装の `tile::select`／`select_for_device` は K 方向の
threadgroup 分割を一切持たない構造上の欠落があることは実装（`tile.rs:1165`／`tile.rs:1183`）から
確定的に確認できていた事実であり、今回の実測はこの構造上の欠落と符合する劣化（対象/対照の TFLOPS 比
< 0.7）が実機で実際に測定可能であることを裏付けた。**留保**: 計測したのは対象・対照双方の実行時間
（TFLOPS）のみであり、実際の occupancy（simdgroup／warp 稼働率等）や split-K 経路自体の改善効果を
計測したものではない（split-K カーネルは未実装のため計測不能。§2）。実行時間差だけから「並列度不足」
という原因への帰属を確定させることはできない — 例えば `(32,32,2048)` 対 `(128,128,128)` は
`actual_groups`（16 対 16）が同値でも劣化率 0.45〜0.59 が観測されており（`run1.log` の
`target_actual_groups=16 control_actual_groups=16` 行参照）、`actual_groups` 一致点ですら対象側が
一貫して遅い。これはタイル構成（対象 `8x8x8` 非 staged・対照 `32x32x16` staged）の違いなど
`actual_groups` 以外の要因も寄与しうることを示しており、「並列度不足の解析ヒューリスティックが真の
occupancy を表さない限界」（旧記述の留保）は解消されていない。原因の確定にはこの限界を踏まえた
split-K プロトタイプによる A/B 計測（§2 の設計方針に基づく別 issue）を要する。

## §1 MLX split-K（非 NAX・Case 1）選択条件

出典: MLX リポジトリ（`ml-explore/mlx`、参照時点コミット `a082cb91d5908e9d89a61a31ee90ee45875b8a1e`。
`gh api repos/ml-explore/mlx/commits/main --jq '.sha'` で解決）`mlx/backend/metal/matmul.cpp`。

### 選択条件式（変数定義 `matmul.cpp:913-921`・Case 1 `matmul.cpp:923-944`・Case 2 `matmul.cpp:947-966`）

```text
_tm = ceil(M / 16)                // matmul.cpp:913
_tn = ceil(N / 16)                // matmul.cpp:914
_tk = K / 16                      // matmul.cpp:915（整数除算・切り捨て）
use_nax = is_nax_available() && !complex && (tf32許可 || dtype != float32)  // matmul.cpp:917-920
devc    = MTLDevice::architecture() 名の末尾文字                             // matmul.cpp:920
min_tmn_threshold = (devc == 's' || devc == 'd') ? 2048 : 1024              // matmul.cpp:921

Case 1（非 NAX・SIMD split-K。条件式 matmul.cpp:925-926、ブロック 923-944）:
  !use_nax && batch_size_out == 1 && (_tm * _tn) <= min_tmn_threshold
    && _tk >= 8 && K >= max(M, N)
  → steel_gemm_splitk_axpby へディスパッチ

Case 2（NAX split-K。条件式 matmul.cpp:948-950、ブロック 947-966。本実装は
  NAX 経路を不採用確定済み — #549・docs/backend-metal-mlx-classic-nax-decision.md）:
  use_nax && batch_size_out == 1
    && (K >= 3*max(M,N) || (max(M,N) <= 1024 && K > 2*max(M,N)))
  → steel_gemm_splitk_axpby_nax へディスパッチ
```

`devc`（`'s'`／`'d'`）が具体的にどの Mac 系列（Studio／Pro〈Duo〉と推測されるが MLX ソース中に明示
コメントはない）を指すかは本ドキュメントでは断定しない（`docs/backend-metal-mlx-classic-nax-decision.md`
と同じ「推定で記述せず実測確認する」原則の準用）。本実装の実機検証環境（M4 Max。
`docs/real-hardware-verification-env.md` §1）はいずれにも該当しない前提で `min_tmn_threshold=1024`
を採用し、`gemm_splitk_shapes_bench.rs::analytics::mlx_case1_domain` もこの前提で実装している。

**本実装は Case 2（NAX 経路）を検討対象に含めない**: #549 で NAX 経路（`MetalPerformancePrimitives`
の `matmul2d`）自体が M4 Max（Neural Accelerator 非搭載）で実証不能と判断済みであり、その判断は
split-K の文脈でも変わらない。本ドキュメントは Case 1（非 NAX・SIMD split-K）のみを対比対象とする。

### split-K カーネルの 2 パス構造（`matmul.cpp:503-653`。`steel_gemm_splitk_axpby` 関数全体）

- **タイル構成**: `bm = M<40 ? 16 : 32`・`bn = N<40 ? 16 : 32`・`bk = 16`・`wm=2, wn=2` 固定
  （`matmul.cpp:527-530`。M/N に応じた 2 択のみで、classic 経路の `CANDIDATES`〈#549〉のような
  複数候補選択は行わない）
- **split 数**: `split_k_partitions = clamp(next_power_of_2(_tk / (_tm*_tn)), 2, 32)`
  （`_tm=ceil(M/32)`・`_tn=ceil(N/32)`・`_tk=K/16`。`matmul.cpp:523-533`。K ループ回数が多いほど・
  M×N の threadgroup 数が少ないほど split 数を増やす経験式）
- **パス 1（`steel_gemm_splitk_*` カーネル。`matmul.cpp:562-606`）**: `grid_dims = (tn, tm,
  split_k_partitions)` の 3 次元 dispatch（`matmul.cpp:599`）。各 `(tn,tm,split_k_partition)` の
  threadgroup が K 方向の担当区間（`split_k_partition_size = (K/bk/split_k_partitions)*bk`）のみを
  部分和として計算し、`C_split`（形状 `{split_k_partitions, M, N}`・dtype `float32`〈`out` が複素数
  なら `complex64`〉。`matmul.cpp:538-542`）へ書く
- **パス 2（`steel_gemm_splitk_accum_*` カーネル。`matmul.cpp:608-646`）**: `C_split` の
  `split_k_partitions` 枚を `out`（`M×N`）へ縮約する専用カーネル。`grid_dims = (N, M, 1)`
  （`matmul.cpp:643`）のシンプルな要素並列 dispatch で、各出力要素が担当スレッド内で
  `split_k_partitions` 個の部分和を**逐次加算**する（atomic は使わない。ファイル全体〈`matmul.cpp`
  3055 行〉を `grep -n atomic` で走査してもヒットせず、本関数に限らず atomic 系呼び出しが一切
  現れないことを確認済み）

## §2 採用する場合の設計方針（記録のみ・実装は別 issue）

以下は §1 の MLX 構造と本実装の既存契約（REQ-2 統一複合判定・FMA 契約・REQ-8 境界検査）を踏まえた
設計方針の記録であり、**本イシューでは実装しない**（採否確定後、別 issue へ切り出す。§3）。

1. **2 パス方式（MLX 同型）**: 分割 K それぞれの部分和 `C_split` を device スクラッチバッファ
   （`f32` 固定。REQ-2 の丸め方針〈CPU 参照実装 `f32::mul_add`・GPU 側既定 FMA 契約〉との整合を保つため
   `half::f16` 蓄積は行わない）へ書き、**固定順序（`split_k_partitions` を昇順に走査）の縮約カーネル**
   で加算する。atomic 加算は使わない — 浮動小数点加算は結合則を満たさないため、atomic による非決定的な
   加算順序は同一入力に対する非決定的な出力（bit-exact 再現性の喪失）を招き、`.claude/rules/
   coding-rust.md`「バックエンド間数値一致は統一複合判定」・数値一致回帰テストの再現性前提を損なう
   おそれがある。固定順序の逐次縮約（MLX の縮約カーネルと同型）はこのリスクを構造的に回避する
2. **選択ロジックの分離**: 分岐は `tile.rs` に純粋関数（例: `should_split_k(m, n, k) ->
   Option<SplitKConfig>`）として置き、`crate::tile::select`／`select_with_occupancy` と同じく
   `objc2` 系 FFI に触れない設計にする（Linux 単体テスト可能。`gemm_splitk_shapes_bench.rs::
   analytics::mlx_case1_domain` はこの純粋関数のプロトタイプを兼ねる診断専用実装であり、恒久実装は
   本方針に従い `tile.rs` へ改めて実装する — 診断 example への先取り実装で重複を作らない設計判断は
   `gemm_diagnosis.rs`〈#487〉の前例と同じ）
3. **シェーダ側手動境界チェックの維持（REQ-8・`.claude/rules/coding-rust.md`「カーネル実装の境界検査」）**:
   split-K パス 1 の K 方向担当区間の端数処理・パス 2 の縮約 grid の M×N 境界のいずれも、性能下限
   達成を理由に手動境界チェックを省略しない
4. **スクラッチバッファサイズ検証は fail-closed**: `split_k_partitions * M * N * 4` バイトの device
   バッファ確保に失敗した場合、または `TileConfig::validate` 相当の検証（threadgroup memory 上限・
   スレッド数上限）を満たさない構成が算出された場合は、本番経路では `unwrap`/`expect` を使わず
   （`.claude/rules/coding-rust.md`「コード品質」）、既存の `tile::fallback_chain` と同様に
   split-K 非適用の既存経路（現行 `tile::select` の結果）へ安全側でフォールバックする
5. **実装は別 issue へ切り出す**: 実装 issue の起票はユーザー承認を経てから行う
   （`.claude/rules/out-of-scope-tracking.md`）。本ドキュメントは PR 本文で切り出しを提案するに
   留め、本 PR 自体では起票しない

**追記（イシュー #1474）**: 上記 1〜4 の設計方針どおり opt-in で実装済み（`dispatch_auto` へは
未結線）。実装記録は `docs/perf/metal-gemm-splitk-two-pass.md` を参照。性能 A/B・本番結線可否は
それぞれ後続イシュー #1475／#1476 のスコープ。#1476 は結線せずと確定（§4）。

**追記（イシュー #1518）**: #1476 時点の「結線しない」判断はその後 #1513 で数値契約
ブロッカーが解消し、#1516 で `dispatch_auto` へ定数ゲート付き（当初既定 OFF。2026-09-11 に既定 `true` へ切替。§5）結線された。
現行状態は §5「本番結線（#1516）」を正とする。

## §3 採否判断

`docs/perf/metal-gemm-splitk-shapes.md` §4「実測結果」・§6「採否判断」で確定した（イシュー #1308・
M4 Max 実機実測・2026-09-09）。

**確定判断: 採用検討推奨。** 判定根拠:

- `crate::tile::select`（本番ディスパッチ入口 `crate::gemm::MetalGemm::dispatch_auto` が使う
  `select_for_device` の内部で呼ばれる形状クラス判定）は K 方向の threadgroup 分割経路を持たない
  （`tile.rs:1165`〈`select`〉・`tile.rs:1183`〈`select_for_device`〉・`tile.rs:1239`
  〈`select_with_occupancy_for_device`。4 分岐 match〉のいずれも M・N 方向の形状判定のみ）。これは
  実測を要さずコードから確定的に確認できる構造上の欠落である
- `docs/perf/metal-gemm-splitk-shapes.md` §3 の解析値は、対象形状 12 点全点が MLX の split-K
  選択域（Case 1）に該当し、`actual_groups`（4〜64）が実機 GPU コア数（40）を下回るか同程度に
  留まることを示していた（理論的な有効性を示唆する一次所見）
- **実測（§4・M4 Max・5 プロセス起動 5 回中央値）は「劣化が実機で測定可能な事実であること」を
  裏付けた**: 対象 9 点（条件2該当。`(256,256,*)` 3 点は `actual_groups=64 >= 40` のため除外）
  全点で対照（同程度 FLOPs の正方立方形状）比の劣化率中央値が 0.2265〜0.5166 の範囲に収まり、
  事前登録基準 `< 0.7` を 5/5 run すべてで一貫して満たした。共有負荷環境（1 分 load average
  1.76〜6.80）下の計測のため spread（0.05 目安）は全点で超過しているが、劣化率自体は 0.7 の閾値
  から明確に離れた値で一貫しており、ばらつきが判定を左右する境界事例ではない
- **ただし「並列度不足の解析ヒューリスティックが真の occupancy を表さない限界」（旧記述の留保）は
  解消されていない**。今回計測したのは対象・対照双方の実行時間（TFLOPS）のみであり、実際の
  occupancy（simdgroup／warp 稼働率等）そのものや split-K 経路自体の改善効果は計測していない
  （split-K カーネルは §2 の設計方針のとおり未実装のため計測不能）。速度差だけから原因を
  「並列度不足」へ帰属させることはできない — 例えば `(32,32,2048)` と対照 `(128,128,128)` は
  ともに `actual_groups=16`（`run1.log` の `target_actual_groups=16 control_actual_groups=16`
  行）だが、タイル構成（対象 `8x8x8` 非 staged・対照 `32x32x16` staged）が異なり劣化率
  0.45〜0.59 が観測されている。`actual_groups` が一致する点でも劣化が生じている以上、
  `actual_groups` 以外の要因（タイル構成差等）が寄与している可能性を否定できない。したがって
  本節で確定したのは「劣化率 `< 0.7` という実測事実」であり、「並列度不足（occupancy 不足）が
  原因である」という因果は仮説（構造上の欠落〈K 方向 threadgroup 分割の不在〉と方向性は整合する）
  にとどまる。因果の確定には split-K プロトタイプによる A/B 計測（§2 の設計方針に基づく別 issue）
  を要する

**追記（イシュー #1475。M4 Max 実機実測・2026-09-09）**: split-K 2 パス実装（#1474）を用い、
上記で留保していた因果を対象 9 形状の A/B 実測（A=classic〈`select_for_device`〉・A′=classic
〈`split_k_tile`。タイル構成のみ変更〉・B=split-K〈`should_split_k` の計画〉の 3 腕比較）で
検証した。**A′/A（タイル構成のみを変えた効果）は 0.99〜1.05 が大半でほぼ 1.0**（classic 経路の
速度はタイル構成差でほとんど変化しない）である一方、**target（A/B。K 分割を含む効果）は 9 形状
すべてで 1.5686〜4.4233 倍の改善**を示した（3 run 中央値。事前登録基準 `speedup >= 1.5` を全形状
で満たし 3/3 run 符号一貫）。この結果は「劣化の主因はタイル構成差ではなく K 方向の並列度不足
（threadgroup 分割の不在）である」という仮説を**支持する**方向で因果の留保を更新する（詳細・
実測表・数値契約の扱いは `docs/perf/metal-gemm-splitk-ab.md` §6 を参照）。ただし単一セッションの
時間制約により事前登録した 5 run のうち 3 run で打ち切ったため、正式確定ではなく暫定判定である
（同 doc §0／§5・§7 フォローアップ参照）。ADOPT（性能上の判定）が確定しても、本番結線
（`SPLIT_K_NUMERIC_CONTRACT_APPROVED` の切替）は別途ユーザー承認が必要（#1476）。
#1476 は性能判定〈undetermined〉・数値契約〈未承認〉の 2 ブロッカーにより結線せずと確定した
（§4）。

**追記（イシュー #1518）**: ブロッカー 2（数値契約）は #1513 で解消済み。ブロッカー 1
（性能の正式 ADOPT 判定）は #1518 時点でも未実測のまま残るが、結線自体は #1516 で
定数ゲート付き（当初既定 OFF・2026-09-11 に既定 `true`）に実施済み。詳細は §5 を参照。

## §4 本番結線可否（#1476）

**追記（イシュー #1518）**: 本節は #1476 時点（2026-09-09）の記録。ブロッカー 2
（数値契約）は #1513 で解消・ブロッカー 1（性能の正式 ADOPT 判定）は #1518 時点でも
未実測のまま残るが、結線自体は #1516 で定数ゲート付き（当初既定 OFF・2026-09-11 に既定 `true`）に実施済み。現行状態は
§5「本番結線（#1516）」を正とする。

**確定判断: 結線しない（`select_for_device`／`dispatch_auto`／`MetalBackendOps::gemm` は不変）。**
`SPLIT_K_NUMERIC_CONTRACT_APPROVED` は `false` を維持し、opt-in 実装（`crate::tile::
should_split_k`／`MetalGemm::dispatch_split_k_strided_prepared`・診断入口 `_with_plan`）は
そのまま残す。独立した 2 つのブロッカーがあり、いずれか一方が解消しても他方が結線を阻む。

1. **性能判定が正式 ADOPT ではない**（`docs/perf/metal-gemm-splitk-ab.md` §0／§9）: #1475
   の機械判定（`aggregate.md` の `verdict`）は `undetermined`（`n_runs=3 < MIN_FORMAL_RUNS=5`。
   PR #1499 の codex-review P1 対応で 5 run 未満は正式 ADOPT/REJECT を出力しない仕様。実測値
   自体〈対象 9 形状すべて speedup 1.57〜4.42・3/3 run 符号一貫〉は改善方向を示すが、Issue の
   結線条件「#1475 が ADOPT の場合のみ」に対し正式な ADOPT 判定は得られていない
2. **数値契約が未承認**（`docs/perf/metal-gemm-splitk-two-pass.md` §5）: split-K は K 分割の
   結合順序差により対象 11 形状中 8 形状が REQ-2 統一複合判定の厳密ゼロ fail を満たさない。
   結線には `SPLIT_K_NUMERIC_CONTRACT_APPROVED=true` への切替が必須で、これは CUDA TF32/f16
   と同型の実測ベースライン非後退方式（`tests/common/splitk_parity_baseline.rs::BASELINES`）
   を Metal f32 split-K へ適用拡張するという、`.claude/rules/coding-rust.md`「バックエンド間
   数値一致テストの許容誤差を単独で緩和しない」原則に基づくユーザー承認事項。memory
   `prod-wiring-preapproved`（性能結線の事前承認）も「tolerance 定数・baseline 行は引き続き
   ユーザー承認必須」と明示的にこの事項を除外している。一括承認（2026-09-09 06:10）の時点は
   PR #1496 レビューで parity 不成立が判明した時刻（同日 08:50 マージ）より前であり、数値契約の
   適用拡張を包含していたとは解釈できない（**追記: イシュー #1511 で適用拡張・baseline 値が2026-09-10 にユーザー承認された〈`docs/backend-metal-splitk-parity-judgment-decision.md`§7〉。受け入れテスト側の判定方式切替は #1512 で完了済みだが、`SPLIT_K_NUMERIC_CONTRACT_APPROVED` の `true` への切替・自動判定入口〈`dispatch_split_k_strided_prepared`〉のゲート解除自体は別イシュー #1513 のスコープであり、本ブロッカー 2 は #1513 完了まで有効のまま**）

### 承認依頼の要点（ユーザーが判断すべき項目）

- (a) REQ-2 実測ベースライン非後退方式の Metal f32 split-K への適用拡張の可否（spec 側への
  提案要否を含む。CUDA TF32/f16 は spec REQ-2 2026-09-02 追記の対象で Metal f32 split-K は
  対象外）
- (b) `splitk_parity_baseline.rs::BASELINES` に登録する具体値の承認（`docs/perf/
  metal-gemm-splitk-two-pass.md` §5.3 の実測表を出典とする）
- (c) 承認後の適用順序: (a)(b) の承認 → `SPLIT_K_NUMERIC_CONTRACT_APPROVED=true` へ切替 →
  結線

**イシュー #1513 で「切替」段階を完了した**: `SPLIT_K_NUMERIC_CONTRACT_APPROVED` を `true` へ
切り替え、自動判定入口 `dispatch_split_k_strided_prepared` のゲートを解除した（`docs/perf/
metal-gemm-splitk-two-pass.md` §5.9）。ブロッカー 2（数値契約）は解消済み。残る「結線」
段階（`select_for_device`／`dispatch_auto` への本番結線）はブロッカー 1（性能の正式 ADOPT
判定）が未解消のため実施しない。本節見出しの「結線しない」判断自体は #1513 時点でも不変。

### 再開条件・結線案メモ（コード変更なし）

結線位置は `MetalBackendOps::gemm`（`crates/backend-metal/src/ops.rs`）または
`MetalGemm::dispatch_auto`（`gemm.rs`）で、`tile::select_for_device` を呼ぶ前に
`tile::should_split_k(m, n, k)` を評価し `Some` なら `dispatch_split_k_strided_prepared` へ
委譲、`None` なら従来の classic 経路を通す構成を想定する。再開時のチェックリスト:

- 正方 N=512〜4096 が `should_split_k` で `None` を返すこと（既存 Linux 単体テスト
  `should_split_k_rejects_large_square_and_wide_shapes` で確認済み。`tile.rs`）
- classic 経路 bit 同一の非後退（`gemm_fine_barrier_bit_match`／`gemm_swizzle_bit_match`／
  `gemm_splitk_bit_match` の classic ケース）
- `gemm_splitk_parity.rs` を baseline 方式（`assert_parity` の厳密ゼロ fail ではなく
  `ParityBaseline` 非後退検査）へ戻す（(a)(b) 承認後。**#1512 で対応済み**。実機〈Apple
  M4 Max〉での全形状 pass 確認は `docs/perf/metal-gemm-splitk-two-pass.md` §5.8 に記入欄
  を残す）
- framework-compare gemm metal 8 セル（N=512〜4096 × fresh/reuse）の before/after 5 回
  中央値・checksum 完全一致（`run_ab_gemm_metal.sh`）
- #1515 で新規 5 run（`docs/perf/metal-gemm-splitk-ab.md` §10）により確定する
  （本 PR 時点では未実測。#1475 の 3 run とは混在させない）

### framework-compare A/B の扱い（「計測対象なし」）

本イシューはコメントのみの変更（コードロジック無変更）のため、`crates/backend-metal/src`
の before（`origin/main` 43a1e158）／after（本 PR HEAD）差分はコメント行のみであり、
framework-compare gemm metal の実行時計測は「計測対象なし」とする（#1272 §5.11 の先例と
同型）。根拠は `git diff --stat 43a1e158 HEAD -- crates/backend-metal/src` の出力:

```text
 crates/backend-metal/src/gemm.rs | 9 ++++++---
 crates/backend-metal/src/tile.rs | 6 ++++--
 2 files changed, 10 insertions(+), 5 deletions(-)
```

変更行はすべて `///`／`//` で始まるドキュメンテーション・通常コメント行であることを
`git diff 43a1e158 -- crates/backend-metal/src/gemm.rs crates/backend-metal/src/tile.rs`
の追加・削除行を機械確認した（コメント以外の追加・削除行は 0 件）。

### 既存 docs との整合確認

`select_for_device` が本イシューで不変であることを前提に記述している以下の 3 文書を確認し、
いずれも記述が真のままであることを確認した（編集なし）:

- `docs/perf/metal-gemm-transpose-tiled.md`
- `docs/perf/metal-gemm-n4096-kernel-gap.md`
- `docs/perf/metal-gemm-candle-gate-remeasurement.md`

**追記（イシュー #1518）**: `select_for_device` 自体は #1516 結線後も不変のままだが、
`dispatch_auto` は `select_route_for_device` を経由するようになったため、上記 3 文書の
該当箇所（`select_for_device`／`dispatch_auto` を参照する記述）へ #1516 以降の状態を
補足する追記を行った（§5「記述整合の棚卸し」参照）。

### スコープ外

- #1475 §7 のフォローアップ（run4/run5・B′ 選択関数呼び出し費用込み再実測・フェーズ 0
  同一入力再実測）は本イシューでは実施しない（独立したブロッカー 2 が結線を阻むため結線判断
  には影響せず、#1475 が自ら宣言したフォローアップであり、計測時の共有負荷〈1 分 load average
  約 6.5〉で事前登録ゲート 4.0 が枯渇しやすく再緩和は行わない）
- NT/TN/TT・f16／hfrag・`gemm_bias_act` 融合経路への split-K 適用（#1474 §8 と同じスコープ外）

## §5 本番結線（#1516）

`MetalGemm::dispatch_auto`（本番 NN 経路の自動入口。`MetalBackendOps::gemm` が呼ぶ唯一の
入口）へ split-K 2 パス経路の分岐を**定数ゲート付きで結線した**（実装 PR 時点は既定 OFF・2026-09-11 に既定 `true` へ切替〈本節末尾〉。実装 PR で
コード変更あり——本節は上記 §4「framework-compare A/B の扱い」が前提としていた「コメントの
みの変更」を、本イシューで正式に更新する）。

### 結線内容

- `crate::tile::GemmRoute`（`SplitK(SplitKPlan)` / `Classic(TileConfig)`）・
  `crate::tile::select_route_for_device(m, n, k, gpu_core_count)`（純関数。`should_split_k`
  が `Some` なら `SplitK`、`None` なら `select_for_device` の結果を `Classic` として返す）を
  新設した。`select_for_device` 自体のシグネチャ・戻り値・意味論は crates.io 公開 API
  互換性のため不変のまま（PR #1108 の互換維持方針を踏襲。Issue タイトルの「`select_for_device`
  へ結線する」は本関数を介して実現する）
- `MetalGemm` に `split_k_auto_enabled: bool` インスタンスフィールドを追加し、`new_with_gates`
  経由で構築する全コンストラクタ（`new`／`new_with_swizzle`／`new_with_fine_barrier`／
  `new_with_unroll_acc`／`new_with_frag_load`／`new_with_coop_load`／
  `new_with_tile_class`／`new_with_source_specialization`）は本番既定ゲート定数
  `SPLIT_K_DISPATCH_AUTO_PRODUCTION_ENABLED`（`crate::tile`。`gemm.rs` 側の全コンストラクタ
  からは `tile::` 経由で参照する。**既定 `false`**）を渡す。
  `MetalGemm::new_with_split_k_auto(ctx, enabled: bool)` を新設し、A/B 計測・実機テスト専用の
  明示 opt-in 入口とする（`new_with_swizzle` 等と同型の設計）
- `dispatch_auto` は `split_k_auto_enabled` が `true` の場合のみ、`should_split_k` の判定を
  **8 の倍数へパディング済みの実効次元**（`crate::pad::pad8`）で評価し、`GemmRoute::SplitK`
  が選ばれた形状のみ split-K 2 パス経路（`dispatch_split_k_strided_prepared_with_plan`。既存
  の internal-diagnostics 限定公開入口を crate 内部から呼ぶ）へ分岐する。`GemmRoute::Classic`
  が選ばれた形状・ゲート `false` の場合は、**`select_route_for_device` が返す `TileConfig` を
  使わず**、常に非パディング次元で `select_for_device(m, n, k, ..)` を呼び直したうえで
  `dispatch_variant` へ委譲する——結線前の `dispatch_auto` と 1 バイトも変わらない経路を保証
  する設計判断（下記「パディング整合の設計判断」参照）
- 診断専用入口 `dispatch_auto_with_route`（`internal-diagnostics` feature 限定の `pub`。既定
  ビルドは `pub(crate)`）を新設し、実際に採用した経路（`GemmRoute`）を返す。`dispatch_auto`
  自身もこの実装（`dispatch_auto_with_route_impl`）へ委譲することで、本番経路と診断経路の
  ロジックが乖離しない構造にしている

### パディング整合の設計判断

`dispatch_variant`（classic 経路）は 8 の倍数へパディングしてディスパッチするが、タイル選択
（`select_for_device`）自体は**非パディング次元**で行う。一方 split-K の判定（`should_split_k`）
は `strided_tiled_eligibility` が 8 の倍数の次元を要求し、`SplitKPlan::k_per_partition` も
渡した `k` から導出されるため、split-K の判定・スクラッチ確保は**実効次元**で行う必要がある。
承認済み対象 11 形状（`tests/common/splitk_parity_baseline.rs::BASELINES`）はすべて 8 の倍数
のためこの差は顕在化しないが、非 8 倍数形状で `select_route_for_device` の `Classic(cfg)` を
そのまま使うと、実効次元と非パディング次元とでタイル選択結果が食い違いうる。この差分を避ける
ため、classic 分岐は `select_route_for_device` の戻り値の `cfg` を捨てて `dispatch_variant`
（非パディング次元で自前に `select_for_device` を呼び直す既存実装）へ委譲する設計とした。
split-K 分岐の内部フォールバック（`dispatch_split_k_strided_prepared_with_plan` がスクラッチ
確保失敗等で classic へ縮退する稀な経路）のみ実効次元で `select_for_device` を呼ぶ非対称が
残るが、承認済み形状群では実効次元＝非パディング次元のため影響しない。

### ゲート既定値・切替条件（事前登録）

> **現行状態（2026-09-11・#1516 で更新）**: 下記は本セクション新設時点（ゲート既定
> `false`）の判断記録であり、そのまま残す。#1515 §10.4 の ADOPT 確定を受け、ゲートは
> 既定 `true` へ切替済み（詳細は本節末尾「ユーザー判断による本番結線（2026-09-11・#1516 マージ）」参照。以下の「既定 `false`」の記述は切替前の経緯として保持する）。

**既定 `SPLIT_K_DISPATCH_AUTO_PRODUCTION_ENABLED = false`**: 依存イシュー #1515（split-K vs
classic 経路の性能 A/B・#1475 §7 フォローアップの 5 run 正式確定）は、5 run 計測スキャフォー
ルド（`docs/perf/metal-gemm-splitk-ab.md` §10）のみを確立した段階で PR #1529 としてマージ・
クローズされ、**§10.4 の ADOPT 判定は「未実測」のまま確定していない**。§4 の結線判断（ブロッ
カー 1「性能判定が正式 ADOPT ではない」）は本イシュー時点でも解消していないため、本ゲートは
既定 OFF のまま維持する。ゲート `false` の間、`dispatch_auto` は本結線コード追加前と bit 同一
の classic 経路を通ることを実機 `#[ignore]` テスト（`tests/gemm_splitk_auto_wiring.rs::
wiring_off_is_bit_identical_to_new`）・ゲート定数のドリフト検出テスト（`gemm.rs::tests::
split_k_dispatch_auto_production_enabled_is_false_by_default`）で機械的に担保する。

`true` への切替手順（Mac セッションでの実施を事前登録する）:

1. #1515 §10（`docs/perf/metal-gemm-splitk-ab.md`）の 5 run 正式計測が ADOPT と確定する
2. `SPLIT_K_DISPATCH_AUTO_PRODUCTION_ENABLED` を `true` へ切替（`crate::tile`。`tile.rs`）
3. `tests/gemm_splitk_auto_wiring.rs`（本イシューで新設）・既存 `#[ignore]` split-K 群
   （`gemm_splitk_bit_match`／`gemm_splitk_parity`／`gemm_splitk_auto_entry_parity`）を実機
   （Apple Silicon）で pass 確認する
4. イシュー #1517（framework-compare 結線前後 A/B）の 5 回中央値で非後退・checksum 完全一致
   を確認する
5. 後退時は `false` へ差し戻し、理由を本節へ追記する

### 切替判断の記録（2026-09-11・#1515／#1516／#1517）

①は #1515 §10.4（M4 Max 5 run・ADOPT）で充足。②③は #1516 でブランチ
`perf/1516-metal-splitk-gate-on`（未 push）上に実装。実機 `#[ignore]` 群
pass（`gemm_splitk_auto_wiring` 4/4・`gemm_splitk_bit_match` 2/2・
`gemm_splitk_parity` 1/1・`splitk_parity_baseline_contract` 11/11・
`splitk_gemm_gate_shape_attribution` 3/3・`gemm_splitk_auto_entry_parity`
1/2〈FAIL 1 件は `(64,64,63)` 非 8 倍数形状を渡す既存無関係な不具合で切替前
HEAD でも再現〉）。④ #1517 run1 は checksum 全 10 セル（gemm 8 + train 2）
一致だが 4 セル `ratio > 1.00`（gemm 3 + train 1）→ 手順⑤に従い **`false`
維持**（ユーザー判断 2026-09-11）。main は既に `false` のため差し戻し
コミットなし。#1516 ブランチはマージしない。後退セルは帰属表上 split-K
非到達・符号非一貫・共有負荷下で計測ノイズと整合（参考情報。規則は
緩めない）。再開条件は #1517 doc §6 参照。

**#1516 ブランチ上のゲート ON 検証記録（2026-09-11。当初は未マージ）**: 手順①は #1515 §10.4（`docs/perf/metal-gemm-splitk-ab.md`。
2026-09-11 に M4 Max 実機 5 run で確定。対象 9 形状すべて中央値 1.55〜3.75 倍・5/5 run 一貫、
対照 3 形状 ≥0.95、phase0 checksum 5 run 完全一致）で **ADOPT** と正式確定し充足された。これを
受け、②`SPLIT_K_DISPATCH_AUTO_PRODUCTION_ENABLED`（`crate::tile`。`tile.rs`）を `true` へ切替
済み。ドリフト検出テスト（`tile.rs::tests::split_k_dispatch_auto_production_enabled_is_false_
by_default`）は `..._is_true_by_default` へ改名し `true` を固定する契約へ更新済み。③実機
（Apple M4 Max）での受け入れテスト実行結果は以下のとおり（すべて pass）:

- `cargo test -p fandhe-ai-backend-metal --lib tile::tests`: 134 passed（
  `split_k_dispatch_auto_production_enabled_is_true_by_default` 含む）
- `cargo test -p fandhe-ai-backend-metal --release --features internal-diagnostics --test
  gemm_splitk_auto_wiring -- --ignored`: 4 passed（`wiring_default_is_bit_identical_to_
  explicit_on`〈旧 `wiring_off_is_bit_identical_to_new` を `new()` の既定 ON 化に合わせて
  改名・再設計〉・`wiring_explicit_off_forces_classic_route_for_targets`〈新設。明示
  opt-out が対象形状でも classic に固定されることの直接検証〉を含む）
- `cargo test -p fandhe-ai-backend-metal --release --features internal-diagnostics --test
  gemm_splitk_bit_match -- --ignored`: 2 passed
- `cargo test -p fandhe-ai-backend-metal --release --features internal-diagnostics --test
  gemm_splitk_parity -- --ignored`: 1 passed
- `cargo test -p fandhe-ai-backend-metal --test splitk_parity_baseline_contract`: 11 passed
- `cargo test -p fandhe-ai-backend-metal --test splitk_gemm_gate_shape_attribution`: 3 passed
  （GEMM candle 比ゲート対象の正方 3 形状は、ゲート ON 後も `should_split_k` が `None` を
  返す独立した理由により classic 経路のまま）

`gemm_splitk_auto_entry_parity`（`--ignored`）は
`auto_entry_falls_back_to_classic_not_eligible_for_non_split_k_shapes` が
`dispatch_split_k_strided_prepared failed (m=64, n=64, k=63): ... must all be multiples of
8` で FAIL したが、本ゲート切替前（`SPLIT_K_DISPATCH_AUTO_PRODUCTION_ENABLED = false` の
HEAD）でも同一の FAIL を再現することを確認済みであり、本切替とは無関係の既存不具合
（`dispatch_split_k_strided_prepared` 自体が 8 の倍数でない次元を拒否する一方、テストが
`(64, 64, 63)` のような非 8 の倍数形状を渡している）。tolerance・テストの緩和は行っておらず、
本 PR のスコープ外として記録するに留める（同テストの `auto_entry_dispatches_split_k_for_
eligible_shapes_and_matches_baseline` は pass）。

④（イシュー #1517 の framework-compare A/B）は上記のとおり結線維持不可であった。

### ユーザー判断による本番結線（2026-09-11・#1516 マージ）

上記の #1517 判定（結線維持不可）は事前登録規則どおり確定した記録として
**書き換えない**。そのうえで、ユーザーは以下の別根拠により
`SPLIT_K_DISPATCH_AUTO_PRODUCTION_ENABLED = true` での本番結線を決定した
（性能規則の事後緩和ではなく、保守性を主眼とする上位判断）:

- #1517 の後退 4 セル（gemm 1024/reuse・2048/fresh・4096/fresh・train
  64/reuse）はいずれも帰属表（同 doc §2）上 split-K に構造的に非到達で、
  before/after が同一カーネル経路の比較・5 run 内で符号非一貫・共有負荷下
  （load1 6.3〜11.6）のため、結線の有無で差が生じないと判断した。checksum
  は全 10 セル完全一致・gemm parity 0 fail。
- 到達形状側の性能根拠は #1515 §10.4（ADOPT・対象 9 形状 1.55〜3.75 倍・
  5/5 run 一貫）で確定済み。数値契約は #1511 承認済み。
- **保守性**: ゲート OFF のままでは split-K 2 パスカーネル・
  `select_route_for_device`・parity baseline・数値契約テストを本番非到達の
  まま保守し続けることになる。結線によりこの死んだ本番隣接経路を解消し、
  次段でゲート定数・ドリフト検出テスト・「既定 false」前提の doc コメント
  を撤去できる（定数撤去は別 PR）。

**ゲート ON での既存 `#[ignore]` 群全件走査（M4 Max・2026-09-11・
backend-metal／facade／autodiff／tensor-core）**: 新規 FAIL は
`tests/gemm_bias_act_parity.rs` の (64, 64, 4096)・Relu 1 件のみ。融合カーネル
（classic 経路）と `MetalBackendOps::gemm`→`add`→`relu` の合成腕（`dispatch_auto`
→ split-K 到達）の Metal 内比較が複合判定 `fail_count=2/4096`
（`max_abs_diff=1.411e-4`・`max_rel_err=1.497e-3`・`mean_abs_diff=8.515e-6`。
CPU 参照との比較は pass）——split-K の K 分割に由来する既知特性（`docs/perf/
metal-gemm-splitk-two-pass.md` §5.5）がこのテストに現れたもの。ユーザー判断
（2026-09-11）により、この比較にも #1511 承認の baseline 非後退方式を適用し、
上記実測値を `tests/common/splitk_parity_baseline.rs::FUSED_VS_COMPOSED_
BASELINES`（CPU 参照用 `BASELINES` 11 行とは独立の表。契約テストの 11 行検査
には含めない）へ記録・切替済み（tolerance 定数・判定式は不変）。その他の FAIL
は main でも同一 assert で FAIL する既存（`command_batching_bench`・
`mnist_scale_train_reuse_metal_batch_counters` の dispatch 数契約）、singleton
プール共有の並列 flaky（`pool_real_device`。両ツリーで再実行 4/4 pass）、
worktree の `docs/spec` 未 checkout（autodiff／tensor-core の PoC evidence 参照）、
CUDA 実機なし（CUDA 専用テスト）のいずれかで、ゲート切替とは無関係。

事後監視として、低負荷時に #1517 と同一プロトコルの framework-compare を
再実行し記録する（結線の条件にはしない）。

### スコープ外（変更なし）

NT/TN/TT・f16／hfrag・`gemm_bias_act` 融合経路への split-K 適用は引き続きスコープ外（§4 の
既存整理を踏襲）。「結線しない記述の横断整合」（他 docs での「結線しない」記述の棚卸し）は
**#1518 で完了**（下記「記述整合の棚卸し」参照）。

### 記述整合の棚卸し（イシュー #1518）

`select_for_device` 不変を前提に「結線しない」「未結線」「`SPLIT_K_NUMERIC_CONTRACT_
APPROVED=false`」等と記述していた docs・`CLAUDE.md` 索引行を棚卸しし、#1513（数値契約
ゲート解除）・#1516（`dispatch_auto` への定数ゲート付き結線・既定 OFF）を反映した状態へ
追記した。判断方針:

- 日付付きの判断記録（例: §3／§4 の「#1476 時点」の確定記録）は歴史記録として本文を残し、
  現行状態への forward pointer のみ追記する（書き換えない）
- 「現在の状態」として読まれる行（冒頭要約・AC 対応表・見出し直下の結論文）は #1516 以降の
  状態へ更新する
- v0.8.0（crates.io registry 系列）に関する `SPLIT_K_NUMERIC_CONTRACT_APPROVED=false`
  等の記述は、v0.8.0 タグ時点の事実として真であるため訂正せず、HEAD の状態は別であることを
  補足するに留める
- 「本番有効」と読める表現は使わない（#1518 時点はゲート既定 OFF。2026-09-11 の既定 `true` 切替後は本項は失効・`SPLIT_K_DISPATCH_AUTO_PRODUCTION_
  ENABLED = false` の間は結線前と bit 同一の classic 経路のまま）

対象（更新・追記）:

| ファイル | 節 | 対応 |
|---|---|---|
| 本ファイル | §2／§3／§4／§5 | forward pointer 追記（本 PR） |
| `docs/backend-metal-splitk-parity-judgment-decision.md` | §8 | forward pointer 追記 |
| `docs/performance-targets.md` | §8.16 | v0.8.0 事実への補足 |
| `docs/perf/metal-gemm-splitk-two-pass.md` | 冒頭・AC-5 行・§6・§8 | forward pointer 追記 |
| `docs/perf/metal-gemm-splitk-ab.md` | §7・§9 | forward pointer 追記 |
| `docs/perf/metal-gemm-n4096-kernel-gap.md` | §19.4 | forward pointer 追記 |
| `docs/perf/metal-gemm-transpose-tiled.md` | §5.10 (b)・§5.11 | forward pointer 追記 |
| `docs/perf/metal-gemm-candle-gate-remeasurement.md` | §16.2・§16.8 | 補足追記 |
| `docs/perf/logs/metal-gemm-splitk-ab-5run-1515/README.md` | 手順 6 | 更新 |
| `CLAUDE.md` | 索引行（4 件） | forward pointer 追記 |

対象外（歴史記録・無関係のため編集しない）:

- `docs/perf/metal-gemm-tile-class-split.md`（E6 の「結線しない」。split-K と無関係）
- `docs/perf/cuda-gemm-f32-variant-selection.md`（CUDA 側 SplitK。Metal split-K とは無関係）
- `crates/backend-metal/src/shaders/gemm.metal` の hfrag「結線しない」コメント（別候補）
- `docs/perf/metal-gemm-splitk-shapes.md`（#1308 の設計検討スコープ記述。歴史記録）
- `docs/perf/metal-gemm-splitk-two-pass.md` §5.5〜§5.7（PR #1496 時点の日付付き記録。
  §5.9 に #1516 追記が既にある）
- `docs/perf/metal-gemm-splitk-framework-compare-1517.md`・
  `scripts/bench/framework-compare/README.md` の split-K 節（#1517 で現行状態に合わせて
  書かれており整合済み）

`.rs` doc comment のフォローアップ（本イシューでは編集しない。受け入れ条件「コード変更
なし」を優先）: `crates/backend-metal/src/gemm.rs`（`SPLIT_K_ROUTE_*` 診断カウンタ・
`SplitKParams`・`SplitKFallbackReason`・split-K パイプラインキャッシュ・
`dispatch_split_k_strided_prepared` の doc comment）・`crates/backend-metal/src/tile.rs`
（`split_k_tile` 付近の「`dispatch_auto` へは未結線」）・
`crates/backend-metal/tests/common/splitk_parity_baseline.rs`・
`tests/gemm_splitk_bit_match.rs`・`tests/gemm_splitk_parity.rs` の冒頭 `//!`。切り出し先:
未起票（ユーザー承認待ち。`.claude/rules/out-of-scope-tracking.md`）。

### 実行時トグル（#1545・2026-09-11）

`SPLIT_K_DISPATCH_AUTO_PRODUCTION_ENABLED = true`（#1544）により split-K 2 パス経路が
本番既定になったのを受け、プロセスワイドかつ実行時に classic 経路へ一時的に固定できる
opt-out スイッチを追加した。

- **API**: `fandhe_ai::set_metal_split_k_gemm_enabled(bool)` / `fandhe_ai::
  metal_split_k_gemm_enabled() -> bool`（`crates/facade/src/lib.rs`。`cfg(target_os =
  "macos")`）。実体は `fandhe_ai_backend_metal::split_k_runtime::set_split_k_enabled` /
  `split_k_enabled`（`crates/backend-metal/src/split_k_runtime.rs`。新設）への薄い委譲。
  `crates/backend-cuda/src/precision.rs`（`AtomicU8` による CUDA GEMM 精度モードの
  プロセスワイド切替）と同型の設計（`AtomicBool`・`Ordering::SeqCst`）。
- **既定値**: `true`（#1544 の本番既定と同一。導入前後でデフォルト挙動は完全に不変）。
- **3 段ゲートの関係**: split-K 到達は次の 3 つがすべて `true` の場合のみ成立する
  （`gemm.rs::dispatch_auto_with_route_impl` の分岐条件）。
  1. `tile::SPLIT_K_DISPATCH_AUTO_PRODUCTION_ENABLED`（コンパイル時定数。既定 `true`）
  2. `MetalGemm::split_k_auto_enabled`（インスタンス単位フィールド。`MetalGemm::new` は
     常に 1. の値を渡す。`new_with_split_k_auto` で個別に固定可能）
  3. `split_k_runtime::split_k_enabled()`（本節の実行時トグル。プロセスワイド・実行中に
     切り替え可能）
- **fail-closed**: `false` にすると、1.・2. の値に関わらず常に classic 経路
  （`GemmRoute::Classic`）へ固定され、split-K 結線前（`fandhe-ai =0.8.0` 相当）と bit
  同一の出力になる。「問題が起きたら `false` にすれば必ず既知の安全な経路へ戻せる」設計。
- **スレッド安全性**: `AtomicBool`（`Ordering::SeqCst`）でスレッド間共有。頻度が低い設定
  変更のため緩い順序による最適化は不要と判断（`precision.rs` の `AtomicU8` と同じ判断）。
- **bit 同一契約**: `false` の間の出力は結線前と bit 同一（classic 経路のみを通るため）。
  `true`（既定）時の split-K 到達形状は従来どおり実測 baseline 非後退方式（`docs/backend-
  metal-splitk-parity-judgment-decision.md` §7）で classic と bit 不一致を受け入れる。
- **`context_cache::cached_gemm` シングルトンとの関係**: `MetalGemm` はシェーダの実行時
  コンパイルを伴う重い構築コストを持つため、本トグルの切り替えのために作り直すことは
  しない。`split_k_auto_enabled` フィールド自体は構築時のまま不変で、呼び出しの都度
  実行時トグルを読み取ることで対応する。
- **テストの所在**:
  - 単体テスト（Linux。`split_k_runtime.rs` 内 `#[cfg(test)] mod tests`）: 既定 `true`・
    set/get 往復・RAII ガードによる原状復帰。
  - 実機 `#[ignore]` 統合テスト（`crates/backend-metal/tests/gemm_splitk_runtime_toggle.rs`。
    `internal-diagnostics` feature 限定）: (a) `false` で `MetalGemm::new()` が
    `new_with_split_k_auto(&ctx, false)` と bit 同一・`GemmRoute::Classic` 到達、
    (b) `true` で対象形状が `GemmRoute::SplitK` へ到達し `new_with_split_k_auto(&ctx,
    true)` と bit 同一、(c) false→true の往復後に既定と bit 同一に戻ることを検証する。
  - 既存 `tests/gemm_splitk_auto_wiring.rs`（インスタンス単位ゲートの検証が目的）は、
    本トグルの状態に左右されないよう各テスト冒頭で `true` へ固定する RAII ガードを追加
    済み。
- **bench-fandhe との関係**: `scripts/bench/` 配下の `--metal-split-k` フラグ（`feature
  metal-split-k-toggle` 限定。別イシュー実装）は本 facade API を経由して実行時トグルを
  操作する想定（本ドキュメント時点ではスコープ外・別エージェント実装）。

**#1549 追記（診断専用・判定基準なし）**: split-K 到達 11 形状（NN・単一乱数系列）で
split-K 経路・classic 経路・CPU f32 参照実装の出力をホスト `f64` 厳密解と突き合わせた
誤差実測を `docs/perf/metal-gemm-splitk-f64-truth.md` に記録した。本実測の範囲では
split-K 経路が classic 経路・CPU f32 参照実装より `f64` 真値に近かった（全 11 形状で
`max_abs`／`mean_abs` とも split-K が下回った）。`assert_no_split_k_parity_regression`
の baseline・tolerance 定数・既存テストは変更していない。

**#1548 事後監視（2026-09-12）**: run2（低負荷）で規則 1 不成立 7 セル（§3 規則上は
結線維持不可相当の記録）・符号一貫セル 0 件・checksum 全一致。原因帰属は判定と分けて
記録: 後退セルはいずれも split-K 非到達で計測ノイズと整合するが、トグル ON 時のみ実行
されるホスト側経路判定のオーバーヘッド寄与は未分離。既定 ON（#1544 ユーザー判断）は
不変。詳細は `docs/perf/metal-gemm-splitk-framework-compare-1517.md` §6a を参照。

## §6 参照

- `docs/perf/logs/metal-gemm-splitk-shapes-1308/`（M4 Max 実機実測の生ログ・`aggregate.py`／
  `aggregate.md`・`env_info.txt`。イシュー #1308）
- `docs/perf/metal-gemm-splitk-ab.md`・`docs/perf/logs/metal-gemm-splitk-ab-1475/`（split-K vs
  classic 経路の M4 Max 実機 A/B・因果検証。イシュー #1475）
- MLX リポジトリ `ml-explore/mlx`（参照時点コミット `a082cb91d5908e9d89a61a31ee90ee45875b8a1e`。
  `gh api repos/ml-explore/mlx/commits/main --jq '.sha'` で解決）
  - `mlx/backend/metal/matmul.cpp:503-660`（`steel_gemm_splitk_axpby`。2 パス構造・`C_split`
    スクラッチバッファ・逐次縮約カーネル）
  - `mlx/backend/metal/matmul.cpp:913-945`（Case 1 選択条件式）
  - `mlx/backend/metal/matmul.cpp:660-820`（`steel_gemm_splitk_axpby_nax`。NAX 版。本実装は不採用）
  - `mlx/backend/metal/matmul.cpp:947-963`（Case 2 選択条件式）
- 本実装 `crates/backend-metal/src/tile.rs:1165`（`select`）・`tile.rs:1183`（`select_for_device`）・
  `tile.rs:1239`（`select_with_occupancy_for_device`）。K 方向分岐が存在しないことの根拠（#1308 で
  #1039 等の後続変更による行番号移動を反映して更新）
- `crates/backend-metal/examples/gemm_splitk_shapes_bench.rs`（本イシューで新規作成。MLX Case 1
  条件式の突合・対象/対照形状の解析値算出・macOS 実機実測）
- `docs/perf/metal-gemm-splitk-shapes.md`（実測記録テンプレート。実測結果・採否判定基準）
- `docs/backend-metal-mlx-classic-nax-decision.md`（#549。同型の決定記録フォーマット踏襲元・
  NAX 経路不採用判断の参照元）
- `docs/backend-metal-aligned-load-decision.md`（#752。同型の決定記録）
- `.claude/rules/coding-rust.md`「バックエンド構成（REQ-2）」「カーネル実装の境界検査（REQ-8）」
  「コード品質」
- `.claude/rules/out-of-scope-tracking.md`
- イシュー #810・親系列 #479（GEMM 性能改善）・関連 #549（NAX 不採用判断）・#487（同型の実測
  記録テンプレート先例）
