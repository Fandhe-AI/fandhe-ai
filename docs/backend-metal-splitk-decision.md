# Metal GEMM split-K ディスパッチ分岐: MLX 選択条件との対比・採否判断（#810）

イシュー #810「perf(backend-metal): split-K ディスパッチ分岐の設計検討」に対応する。ルート #479
（GEMM 性能改善）系列。`docs/backend-metal-mlx-classic-nax-decision.md`（#549）・
`docs/backend-metal-aligned-load-decision.md`（#752）と同型の決定記録として、(1) MLX の split-K
選択条件（非 NAX・Case 1）と本実装 `crate::tile::select` の構造対比、(2) 採用する場合の設計方針
（記録のみ・実装は別 issue）、(3) 採否判断（実測待ち）を残す。**本イシューは設計検討（調査・計測・
記録）であり、`crates/backend-metal/src/`・`shaders/gemm.metal` は一切変更しない。**

## 判断サマリ

**実測（`docs/perf/metal-gemm-splitk-shapes.md` §4）待ちのため、採否は未確定。** 解析値（同 doc §3）
は「対象形状（K 支配的非正方。M=N=32〜256・K=2048〜8192）12 点は全点 MLX の split-K 選択域に該当し、
かつ本実装の `actual_groups`（4〜64）は実機 GPU コア数（40）を下回るか同程度に留まる」ことを示しており、
split-K 導入の理論的な有効性を示唆する一次所見が得られている。ただし本実装の `tile::select` は K 方向の
threadgroup 分割を一切持たない構造上の欠落があることは実装（`tile.rs:788-793`）から確定的に確認できるが、
実機での劣化幅（TFLOPS 比）は未計測であり、`docs/perf/metal-gemm-splitk-shapes.md` §5 の判定基準に
基づく確定判断は Mac 実機セッションに委ねる。

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

## §3 採否判断

`docs/perf/metal-gemm-splitk-shapes.md` §4「実測結果」の記入を受けて確定する。

**本セッション環境は Linux のため実測は実施できない**（#487・#549 と同じ運用）。実装エージェントの
実行環境が macOS の場合のみ、その場で実測・記入まで行い採否を確定してよい（同 doc §5 の判定基準に
従う）。実測未実施の間、以下を構造的事実として記録する:

- `crate::tile::select`（本番ディスパッチ入口。`crate::gemm::MetalGemm::dispatch_auto` が使用）は
  K 方向の threadgroup 分割経路を持たない（`tile.rs:788-793` の 4 分岐はすべて M・N 方向の形状判定
  のみ）。これは実測を要さずコードから確定的に確認できる構造上の欠落である
- `docs/perf/metal-gemm-splitk-shapes.md` §3 の解析値は、対象形状 12 点全点が MLX の split-K
  選択域（Case 1）に該当し、`actual_groups`（4〜64）が実機 GPU コア数（40）を下回るか同程度に
  留まることを示す。これは split-K 導入の理論的な有効性を示唆する一次所見であり、確定的な採用判断
  ではない（並列度不足の解析ヒューリスティックが真の occupancy を表さない限界は
  `gemm_diagnosis.rs::analytics::DeviceProfile` ドキュメントと同じ限定を持つ）

## §4 参照

- MLX リポジトリ `ml-explore/mlx`（参照時点コミット `a082cb91d5908e9d89a61a31ee90ee45875b8a1e`。
  `gh api repos/ml-explore/mlx/commits/main --jq '.sha'` で解決）
  - `mlx/backend/metal/matmul.cpp:503-660`（`steel_gemm_splitk_axpby`。2 パス構造・`C_split`
    スクラッチバッファ・逐次縮約カーネル）
  - `mlx/backend/metal/matmul.cpp:913-945`（Case 1 選択条件式）
  - `mlx/backend/metal/matmul.cpp:660-820`（`steel_gemm_splitk_axpby_nax`。NAX 版。本実装は不採用）
  - `mlx/backend/metal/matmul.cpp:947-963`（Case 2 選択条件式）
- 本実装 `crates/backend-metal/src/tile.rs:682-831`（`select`／`select_with_occupancy`。K 方向分岐が
  存在しないことの根拠）
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
