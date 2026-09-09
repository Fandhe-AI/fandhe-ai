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
それぞれ後続イシュー #1475／#1476 のスコープ。

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

## §4 参照

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
