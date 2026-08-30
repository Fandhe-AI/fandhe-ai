# B パネル packing のスレッド間共有化の設計検討（#565）

イシュー #565「B パネル packing のスレッド間共有化を設計検討」に対応する。GEMM 性能改善ツリー
（#479）の一環で、E-8（#564・PR #701）が確立した `BlockSizes` パラメータ化・NC 拡大候補グリッド
（`docs/perf/cpu-gemm-blocking-sweep.md`）を前提に、NC 拡大時の B packing 重複コスト・共有化の
設計案・適用可否を記録する。**本イシューは設計検討であり、コード（`crates/` 配下）は変更しない。
実装は別タスクとする**。

## 判断サマリ

現行の並列経路（`gemm_blis_parallel`／`gemm_blis_bias_act_parallel`）は C を行パネル一段のみで
分割し、各タスクが独立に B パネルを pack する。この構造は B packing 総量が理想値（k×n 要素 1 回）
の**実際の並列タスク数 Q 倍**（Q = 実際に生成される行パネルタスク数。定義は §A。スレッド数 T の
下限で Q ≤ T。REQ-8 判定形状では Q = T）になり、B パネルバッファも Q 本個別確保される（§A・§B）。
E-8 の候補グリッド
（`docs/perf/cpu-gemm-blocking-sweep.md` §3.4）のうち **KC=4096 かつ NC が大きい組合せ（候補
#6: NC=9600・候補 #8: NC=4096）**では 1 本あたり数十〜百 MiB 級（64〜150MiB）になるため、
**共有化なしでは NC 拡大が実運用に適用しにくい**（§B）。KC=256 の候補（#2・#3）は数 MiB 台、
NC=512・KC=4096 の候補 #4 は 8MiB、NC=4096・KC=1024 の候補 #7 は 16MiB に留まり、footprint 課題
が顕著に大きくなるのは KC・NC が双方とも大きい組合せ（#6・#8）に限られる。

設計案は rayon プリミティブのみ（`thread_tree` 等の外部依存は使わない。§C）で 3 案を比較し、
**案 B（BLIS 方式・共有 pack ＋ ic 並列）を推奨案とする**。案 A（jc 列分割）は C の列方向の非連続
アクセスに raw pointer 分割（unsafe 新規追加）が必要で REQ-8・unsafe 最小化方針との整合コストが
高く、案 C（`rayon::join` 再帰分割）は案 B と同じ効果を複雑な構成で得るのみで優位性がない。適用は
E-8 の NC 実機選定完了後、後続実装タスクとして着手することを条件とする（§E）。

## §A 現状構造の実測ベース整理

`crates/backend-cpu/src/gemm_blis/mod.rs`（origin/main `bae1f0f` 時点）の並列経路は次の構造。

- `gemm_blis_parallel`（`mod.rs:216-244`）・`gemm_blis_bias_act_parallel`（`mod.rs:278-323`）は
  いずれも `rayon::current_num_threads()` から `panel_rows = m.div_ceil(num_threads)` を算出し、
  `c.par_chunks_mut(panel_rows * n)` で C を**行パネル一段のみ**に分割する（`mod.rs:233-236`・
  `mod.rs:312-315`）
- 各タスク（行パネル 1 つ）は `dispatch_region` を個別に呼ぶ（`mod.rs:241`・`mod.rs:320`）。
  `dispatch_region` はカーネル型確定直後に `PanelBuffers::new` を呼び、A/B packing バッファを
  **タスクごとに 1 組ずつ**確保する（`mod.rs:372-448` の各 arch 版いずれも同型）
- `PanelBuffers` のドキュメントコメント（`mod.rs:113-127`）は「B packing のスレッド間重複計算
  （同じ B 列ブロックを複数タスクが個別に pack し直す）は本変更（#556）のスコープ外＝既存挙動の
  まま」と明記済みであり、本イシューはその後続検討にあたる
- B packing 自体は `gemm_blis_region`（`mod.rs:495-620`）内の jc→pc ループで、pc ブロックごとに
  1 回だけ実行され ic ループ全体で再利用される（`mod.rs:512-544`）。この再利用は**タスク内**で
  閉じており、タスク間（行パネル間）では共有されない。全タスクが同一の jc×pc 範囲（B の列・K
  ブロック全体）を担当するため、B の pack 対象領域は全タスクで完全に重複する
- **実際に並列実行される行パネルタスク数 Q の定義**: `panel_rows = m.div_ceil(T)`（`mod.rs:234`・
  `mod.rs:313`・`mod.rs:720`）で 1 パネルの行数を決めた後、`par_chunks_mut(panel_rows * n)`
  （`mod.rs:236`・`mod.rs:315`・`mod.rs:722`）が生成する実際のタスク（B packing の重複単位）数は
  `Q = m.div_ceil(panel_rows) = ceil(m / ceil(m/T))` であり、**スレッド数 T の下限（Q ≤ T）**に
  なる。m が T で割り切れない・m が T に対して小さい形状では Q < T になりうる（例: m=10・T=8 なら
  `panel_rows = ceil(10/8) = 2`・`Q = ceil(10/2) = 5 < 8`）。以降 §B の重複コスト・footprint の
  見積りは T ではなく Q を掛け算の基準とする。REQ-8 判定形状（m=2048／4096、T=8／12）はいずれも
  `Q = T`（後述 §B.1 で確認）だが、これは m が T の倍数に近い形状に限った一致であり一般には
  Q ≤ T である点に注意する

## §B 重複コストの見積り（解析モデル。実機実測は不要）

すべて解析値であり、実測値の捏造は行わない（`docs/cpu-gemm-prefetch-decision.md`・
`docs/perf/cpu-gemm-blocking-sweep.md` の fail-closed 前例と同方針）。

### §B.1 packing 総量モデル

- 現行方式: 各タスクが B 全体（k×n 要素）を pack するため、総 pack 要素数コピー量は **Q×k×n**
  （Q = §A で定義した実際の並列タスク数。Q ≤ T ＝ スレッド数）
- 共有方式（案 B）: B パネルを (jc,pc) ブロックごとに 1 回だけ pack して全タスクが共有読み出し
  するため、総 pack 要素数コピー量は **k×n**（Q に依存しない）
- 対計算量比: GEMM の乗算加算総数は `2·m·n·k`（FMA を 1 演算とすれば `m·n·k`）。B packing の
  相対コストは `pack量 / 計算量` で近似できる。REQ-8 判定形状（M=N=K=2048／4096）・代表スレッド数
  （Apple M4 Max 性能コア数相当。#481 で確定した対象実機の性能コア数を想定し T=8/12 の 2 点で
  試算）で数表化する。これらの形状は §A の Q 定義（`Q = ceil(m / ceil(m/T))`）に当てはめると
  m=2048・T=8 で `panel_rows=256, Q=8`、m=2048・T=12 で `panel_rows=171, Q=12`、m=4096・T=8 で
  `panel_rows=512, Q=8`、m=4096・T=12 で `panel_rows=342, Q=12` となり、**いずれも Q=T**（m が
  T のほぼ倍数になる形状のため）。よって下表は Q をそのまま T として計算してよい:

| 形状 (M=N=K) | 計算量 m·n·k | 現行 pack 総量 (Q=T=8) | 対計算量比 (Q=T=8) | 現行 pack 総量 (Q=T=12) | 対計算量比 (Q=T=12) | 共有 pack 総量 |
|---|---|---|---|---|---|---|
| 2048 | 8.59×10⁹ | 8×2048×2048 ≈ 3.36×10⁷ | ≈3.9×10⁻³ | 12×2048×2048 ≈ 5.03×10⁷ | ≈5.9×10⁻³ | 2048×2048 ≈ 4.19×10⁶ |
| 4096 | 6.87×10¹⁰ | 8×4096×4096 ≈ 1.34×10⁸ | ≈2.0×10⁻³ | 12×4096×4096 ≈ 2.01×10⁸ | ≈2.9×10⁻³ | 4096×4096 ≈ 1.68×10⁷ |

対計算量比は形状（M=N=K=2048／4096）・T（8／12）の組合せで**約 2.0×10⁻³〜5.9×10⁻³（10⁻³ の
オーダー。正方形かつ Q=T の場合 `T/m` に一致する）**である。m が T の
倍数から離れる形状では Q<T により実際の比率はこの表よりさらに小さくなりうる（§A）。10⁻⁴ オーダー以下という
過小評価はできないが、計算量全体（10⁹〜10¹⁰ オーダー）との比較では小さい部類に入るため、
compute-bound 領域では pack コピー自体の CPU サイクル影響は限定的と評価する。ただし帯域観点
（§B.3）では別の効き方をする点に注意する。

### §B.2 メモリ footprint モデル

B パネルバッファ 1 本のサイズは `nc_len_max × kc_len_max × 4B`（`panel_capacity` の B 長算出、
`mod.rs:174-191`）。`nc_len_max = blocks.nc.min(n)`（`mod.rs:183`）であり NC 候補値をそのまま
使うのは **`n >= NC`（行列 N 次元が NC 候補値以上）の場合に限る**。下表は NC・KC 候補
（`docs/perf/cpu-gemm-blocking-sweep.md` §3.4 の候補グリッド）と §A で定義した実際の並列タスク数
Q（`n >= NC` かつ m=n=k の正方形形状で T=8／12 のとき §B.1 で確認したとおり Q=T となるため、
以下 Q=T として表化する。一般形状では Q<T になりバッファ本数はこれより少なくなりうる）の
footprint を、**`n >= NC` を前提として** `nc_len_max = NC` の場合で表化したもの（現行方式は Q 本
個別確保、共有方式は 1 本）。REQ-8 判定形状（M=N=K=4096）のように `n < NC` となる候補では
`nc_len_max` が `n` にクランプされ実際の footprint はより小さくなる点に注意（下記補足を参照）。

| NC | KC | 前提 n | 1 本あたり footprint | 現行方式 (Q=T=8) | 現行方式 (Q=T=12) | 共有方式 |
|---|---|---|---|---|---|---|
| 512（候補 #1・#5。現行は #1） | 256 | n≥512 | 512×256×4B = 512KiB | 4MiB | 6MiB | 512KiB |
| 4096（候補 #2） | 256 | n≥4096 | 4096×256×4B = 4MiB | 32MiB | 48MiB | 4MiB |
| 9600（候補 #3） | 256 | n≥9600 | 9600×256×4B ≈ 9.4MiB | 75MiB | 112.5MiB | 9.4MiB |
| 512（候補 #4） | 4096 | n≥512 | 512×4096×4B = 8MiB | 64MiB | 96MiB | 8MiB |
| 4096（候補 #7） | 1024 | n≥4096 | 4096×1024×4B = 16MiB | 128MiB | 192MiB | 16MiB |
| 4096（候補 #8） | 4096 | n≥4096 | 4096×4096×4B = 64MiB | 512MiB | 768MiB | 64MiB |
| 9600（候補 #6） | 4096 | n≥9600 | 9600×4096×4B = 150MiB | 1200MiB (≈1.17GiB) | 1800MiB (≈1.76GiB) | 150MiB |

（候補 #5 は MC のみ firestorm 値〈NC/KC は候補 #1 と同じ〉のため B footprint は候補 #1 と同値。
候補グリッドの全定義は `docs/perf/cpu-gemm-blocking-sweep.md` §3.4 を参照）

`docs/perf/cpu-gemm-blocking-sweep.md` §6 は firestorm 参照値（候補 #6: NC=9600/KC=4096）で
「B パネルは約 150MiB × アクティブスレッド数」になると既に指摘しているが、本表はこれを候補
グリッド全体・Q=T=8/12 の両方へ拡張したものである。**候補 #6 は `n≥9600` の形状（`nc_len_max=NC`
がそのまま適用される場合）では現行方式で Q=T=12 時に 1.76GiB に達し、A パネル（別途確保・同様に
Q 倍化）・C（同形状なら別途 footprint）を合わせると実運用機のメモリ容量（M4 Max の統合メモリ）
を圧迫しうる**。共有方式ならこの候補でも 150MiB 固定である。

**REQ-8 判定形状（M=N=K=4096）での候補 #6 の実際値**: この形状では `n=4096 < NC=9600` のため
`nc_len_max = min(9600, 4096) = 4096` にクランプされ、1 本あたり footprint は
`4096×4096×4B = 64MiB`（候補 #8〈NC=4096・KC=4096〉自体の footprint と同値）であり、上表の
150MiB／1.76GiB は**適用されない**。現行方式では m=4096・T=8 で `Q=8`・T=12 で `Q=12`（§B.1）
のため 512MiB・768MiB（候補 #8 と同じ）に留まる（候補 #4〈8MiB〉・候補 #7〈16MiB〉とは別枠）。
1.76GiB という値は `n≥9600` の形状（例: N=9600 以上の非正方形・大規模 N 形状）にのみ当てはまり、
REQ-8 の 4096 正方形単体の圧迫根拠としては使えない。さらに `n≥9600` かつ m がスレッド数 T の
倍数から離れる非正方形形状では Q<T により実際の本数はここに示した Q=T 前提の値よりも少なくなり
うる（§A）。ただし `n≥9600` の形状は他候補（#3 等）や将来の大規模 N ワークロードで現実的に
発生しうるため、NC 拡大候補全般の footprint 課題としての本節の結論（共有化の必要性）自体は
変わらない。

### §B.3 帯域観点（定性）

packing は本質的に帯域律速の操作（B のストライド読み出し→連続書き込み）である。§B.1 のとおり
計算量比では小さいが、**Q 個のタスク（実際の並列行パネルタスク数。Q ≤ T。§A）が同時に同じ DRAM
領域（B 全体）へストライドアクセスする**ため、重複分は他タスクの GEMM 本体（compute-bound な
microkernel 実行）の帯域と干渉しうる。
特に NC・KC が双方とも大きい候補（§B.2 の候補 #6・#8 等。KC=4096 系のうち NC も拡大している
組合せ）では 1 タスクあたりの pack 対象が L2/L3 に収まらないサイズになり、DRAM 直読みの頻度が
増えるため、この干渉は NC/KC が小さい現行値よりもこれらの候補で相対的に強く効くと推定される
（定性評価であり実測ではない）。NC のみを拡大し KC=256 のまま据え置く候補（#2・#3）は pack
対象が KC=256 のまま小さく収まるため、この干渉増大は本節の対象外である。

## §C 設計案（rayon プリミティブのみ。`thread_tree` 不使用）

`matrixmultiply` の `thread_tree` 方式は許容依存 8 区分外のため使用不可（`.claude/rules/
deps-policy.md`）。rayon（許容依存内・既存依存）の `rayon::join`／`par_chunks_mut`／
`rayon::scope` のみで模す設計に限定する。

### 案 A: OpenBLAS js 方式（jc ブロックをスレッド間分配）

各スレッドが自分の jc（列ブロック）範囲のみを担当し、その範囲の B のみを pack する（重複解消）。

- **利点**: 実装が「並列軸を ic→jc に変える」だけで構造変更が小さい。B の重複は担当 jc 範囲内
  でのみ発生し、スレッド間では発生しない
- **欠点**: C は行優先（row-major）レイアウトのため、列ブロック（jc 範囲）は行方向に非連続
  （ストライド `n`）。`par_chunks_mut` は連続スライスの分割のみ対応するため、以下のいずれかが
  必要になる:
  - C を列ブロックごとに raw pointer で分割し `unsafe` で各タスクへ渡す（`.claude/rules/
    coding-rust.md`「unsafe は FFI 境界等の必要最小限」方針との整合確認が新たに必要。理由コメント
    ＋ security-auditor レビュー必須になる）
  - 列パネル分の一時バッファへ計算してから C へ書き戻す（メモリ・コピーオーバーヘッド増）
- A packing は逆に、並列軸を ic→jc へ変えることで**新たに重複が生じる**。現行方式（ic 並列。
  行タスクが自分の ic 範囲のみを担当）では A の pack 対象が各タスクで排他的な行範囲に限られる
  ため重複しないが、案 A（jc 並列）では各列タスクが自分の jc 範囲について pc→ic ループを
  独立に最後まで回す必要があり、全 m 行分の A を列タスクごとに個別に pack し直すことになる。
  列タスク数を J（現行の行タスク数 Q〈§A〉と同様、`n` を列ブロック粒度で分割した際の実タスク数。
  J ≤ T）とすると、A の総 pack 量は現行の `m×k`（重複なし）から `J×m×k`（J 倍重複）へ増加する。
  つまり案 A は B 側の重複問題（現行 Q 倍。§B）を解消する代わりに、対称的に A 側で同種の
  重複問題（J 倍）を新たに生じさせるトレードオフであり、「A 側の重複は元々発生しない」という
  単純化はできない

### 案 B: BLIS 方式（共有 pack ＋ ic 並列）※推奨

jc/pc ループは直列に回し、(jc,pc) ブロックごとに B パネルを **1 本だけ** pack する（pack 自体を
nr ブロック単位で `par_iter` 化することも可能）。pack 完了後、`rayon::scope` 等の同期境界を挟んで
共有 `&[f32]` の B パネルを全タスクが読みつつ、ic（行パネル）を `par_chunks_mut` で並列化する。

- **利点**: C の行チャンクは連続なので既存の `par_chunks_mut` 構造がそのまま使え、**safe Rust
  のみで成立する**（案 A のような raw pointer 分割が不要）。B packing は (jc,pc) の組み合わせ
  回数だけ発生し、NC/KC を拡大するほど (n/NC)×(k_dim/KC) の反復回数が減る（例: NC=KC=4096・
  M=N=K=4096 なら 1 回のみ）ため、NC 拡大とスレッド間共有化は相補的に効く
  （§B.1 の重複コストが構造的に縮小する方向）
- **欠点**: (jc,pc) ブロックの境目で「B pack → 全スレッド待ち合わせ → ic 並列 → 次の (jc,pc) へ」
  という同期点が入る。同期回数自体は (n/NC)×(k_dim/KC) 回であり、NC/KC を拡大すると同期回数は
  むしろ**減る**方向（現行 NC=512/KC=256 では M=N=K=4096 で 8×16=128 回、firestorm 値
  NC=9600/KC=4096 なら 1 回）。A packing は元々 ic ブロックごとにタスクローカルで行われるため、
  この設計変更でも A 側の構造（`PanelBuffers::a_panel`）は変えずに済む
- **融合 epilogue（bias/activation）との整合**: 現行の `gemm_blis_bias_act_parallel`
  （`mod.rs:278-323`）は行パネルタスクごとに `dispatch_region` が全 jc/pc ループを走らせた
  **直後に 1 回だけ** `apply_epilogue`（`mod.rs:336-`）を呼ぶ契約であり（`mod.rs:321`）、
  1 つの C 行範囲に対して bias 加算・活性化関数を **ちょうど 1 回**適用することを前提にしている。
  案 B は並列軸を「行パネルタスクが (jc,pc) を含む全域を担当」から「(jc,pc) ブロックごとに
  ic 並列を挟む」構造へ変えるため、素朴に epilogue 呼び出しを jc／pc ループの内側（各
  (jc,pc) ブロックの ic 並列直後）へ移すと、同じ C 行範囲に対して (n/NC)×(k_dim/KC) 回
  epilogue が重複適用されてしまう（bias が複数回加算される、活性化関数が非線形なら結果自体が
  変わる誤り）。案 B を bias/activation 融合カーネル（`gemm_blis_bias_act_parallel` 相当）へ
  適用する場合は、**全 (jc,pc) ブロックの ic 並列が完了した後（最終 pc ブロック処理後）にのみ、
  各 C 行範囲について 1 回だけ** `apply_epilogue` を呼ぶ構造を維持する必要がある。具体的には
  (a) epilogue 呼び出しを (jc,pc) ループの外（現行と同じく行パネルタスク単位の後処理として）に
  据え置き、jc/pc ループ側は純粋に GEMM 本体（B 共有 pack＋ic 並列）のみを担当する形に留めるか、
  (b) (jc,pc) ループを共有 pack 構造に変えた上で最終ブロック判定（残り pc・jc が無いことの判定）
  を明示的に行い、そのタイミングでのみ epilogue を適用するかのいずれかを実装タスク側で選択する。
  bias-free の `gemm_blis_parallel`（epilogue 呼び出しなし）にはこの制約は生じない

### 案 C: `rayon::join` ネスト 2 階層

`thread_tree` 相当（階層的なワーカーグループ分割）を `rayon::join` の再帰分割で模す。上位の
`rayon::join` で jc 範囲を 2 分割、その中でさらに `rayon::join` を再帰させてスレッドグループを
構成し、各グループ内で B pack を共有する案。

- **評価**: rayon のワークスティーリングスケジューラは `rayon::join` の再帰をタスクとして
  スティールする設計であり、「特定のスレッド集合に B パネルを固定して共有する」という
  `thread_tree` の前提（スレッドとデータの固定対応）とは相性が悪い（rayon はスレッド ID を
  明示的に扱う API を提供しない）。案 B と同等の効果（共有 pack＋並列読み出し）を、案 B より
  複雑な再帰構成で得るだけであり、**明確な優位性がない**ため不採用

### 比較まとめ

| 案 | unsafe 追加 | C アクセスパターン | 同期境界 | NC 拡大との相性 |
|---|---|---|---|---|
| A（jc 分配） | 必要（列分割）または一時バッファ | 列方向非連続（要対処） | タスク独立（同期なし） | 中（B側重複は解消するが A側に同種の重複〈J 倍〉が移るうえ構造変更コスト大） |
| B（共有 pack＋ic並列）**推奨** | 不要 | 行方向連続（既存構造のまま） | (n/NC)×(k/KC) 回。NC拡大で減少 | 高（相補的） |
| C（join ネスト） | 不要 | 案 B と同様 | 案 B相当だが構成複雑 | 中（優位性なし） |

## §D 契約・制約の確認

- **bit 完全一致契約（REQ-2）の維持根拠**: B packing の共有化は C 各要素の FMA 累積順序（p 昇順・
  レーン間縮約なし）に一切影響しない。`gemm_blis_region`（`mod.rs:495-620`）の C タイル ロード
  ／書き戻し構造・累積順序は「どのタスクが・いつ B パネルを pack したか」とは独立な関心事であり、
  案 A／B／C いずれも C の計算ロジック自体（`kernel.run` 呼び出し・FMA 連鎖）には触れない。
  `docs/perf/cpu-gemm-blocking-sweep.md` §3.2 が「MC/KC/NC の値は累積順序に一切影響しない」と
  示した論法と同型で、「並列分割の再構成」も同じ理由で bit 完全一致を破らない。tolerance の
  変更は不要かつ行わない
- **カーネル境界検査（REQ-8）**: 全案でカーネル本体（`microkernel::{neon,avx2,avx512}` の
  intrinsics 境界検査）には触れない。packing 層（`pack_a`／`pack_b`）・並列分割層の再構成に
  限定されるため、マイクロカーネルの手動境界チェックは不変
- **依存追加なし**: rayon は許容依存 8 区分内の既存依存（`.claude/rules/deps-policy.md`）。
  `thread_tree`（matrixmultiply 固有）等の外部依存は導入しない。案 A で unsafe が必要になる
  場合はその旨を明記し、`.claude/rules/security.md`「レビュー体制」に従い security-auditor
  レビュー必須とする
- **REQ-8 下限値・`docs/spec/` 不変更**: 本ドキュメントは性能下限・受け入れ基準を変更しない

## §E 適用可否の結論と後続

- **推奨案は案 B（共有 pack＋ic 並列）**。理由は §C 比較まとめのとおり、safe Rust のみで成立し
  （unsafe 追加不要）、NC/KC 拡大と相補的に効く（同期回数が拡大とともに減る）ため
- **適用条件**: E-8（#564）の M4 Max 実機での MC/KC/NC 実測・選定（`docs/perf/
  cpu-gemm-blocking-sweep.md` §状態節。本ドキュメント作成時点で環境ゲート未達・未実施）が
  完了し、NC 拡大の採否が確定した後に着手することを前提とする。NC が現行値（512）のまま
  据え置かれる場合でも、案 B 自体（重複コスト削減・footprint 削減）は独立に価値があるため、
  実測結果に関わらず適用候補として成立する
- **実装は別タスクである**。後続実装タスクの起票は `.claude/rules/out-of-scope-tracking.md`
  に従いユーザー承認を経て行う。自動運転モードの本セッションでは Issue 起票を行わず、
  切り出し候補として本ドキュメント・PR 本文に記録するに留める

## §F 実装済み・本番不採用（イシュー #750。採用ゲート未通過）

案 B を実装した（イシュー #750。`crates/backend-cpu/src/gemm_blis/mod.rs` の
`gemm_blis_shared_b_region`／`gemm_blis_ic_loop`／`IcLoopContext`／`dispatch_shared_b`）。
併せて B packing 自体を `nr` ブロック単位で `par_chunks_mut` により並列化した。

本番公開入口（`gemm_blis_parallel`／`gemm_blis_bias_act_parallel`）は本節で述べた「共有
pack」経路（`dispatch_shared_b`）を**採用しない**（codex-review P1 指摘・
thread `PRRT_kwDOTuUCJc6arIUt` を受けた是正）。実タスク数の多寡に関わらず常に従来の
`dispatch_region`（行パネルごとに B を個別 pack する PerTaskPrivateB 相当方式）を呼ぶ。
`gemm_blis_shared_b_region`／`dispatch_shared_b` は `#[cfg(test)]` 限定で実装・bit 完全一致
テストのみに使う（テスト専用入口 `gemm_blis_parallel_with_blocks` 経由）。

理由は §E の適用条件（E-8 実機実測完了）と同型の未充足に加え、REQ-8 の受け入れ条件 2
（Apple M4 Max 実機実測での M=N=K=2048/4096 非劣化確認）が本 PR 時点で環境ゲート未達・
未実施のため。本ドキュメント §E で「実装は別タスクである」としていた区切りに続き、
**「本番採用も実機実測を前提条件とする別タスク（別 PR）である」**という区切りを追加する
（PR #758〈イシュー #740・mma_f16 threadblock swizzle〉の最終マージ状態と整合する運用。
同 PR は commit `8269801`「mma_f16 threadblock swizzle の本番結線を差し戻す」により、一時的に
本番既定コンストラクタへ結線した swizzle 変種を `internal-diagnostics` feature 限定の経路へ
差し戻した状態でマージされている＝実機ゲート未達のうちは本番へ結線しない判断）。

数値一致（bit 完全一致契約）・実機性能実測の状況・採用ゲートの詳細は
`docs/perf/cpu-gemm-b-packing-sharing.md` を参照。

**追記（イシュー #793）**: 本節「実装済み・本番不採用」の状態は #793 でも変わらない。#793 は
本番結線（採用ゲート＝実機実測での非劣化確認）の実施を試みたが、実装セッションの環境が
Apple M4 Max 実機に到達できず（`docs/perf/cpu-gemm-b-packing-sharing.md` 追記節参照）、
`gemm_blis_shared_b_region`／`dispatch_shared_b` は引き続き `#[cfg(test)]` 限定のまま
本番未結線である。本節タイトルの「§F を『結線済み』へ更新する」は実機ゲート通過後の別セッション
に持ち越す。

**追記（イシュー #1041）**: 対 gemm crate（faer 実体）直接比較で判明した N=1024/2048 の
劣位（`docs/perf/oss-gemm-comparison-baseline.md` §7.2）を受け、A packing の重複（本節が
扱う B の重複とは別軸。jc 反復ごとに同一 A 行を再 pack する問題）を解消する pc 外側ループ
候補 `gemm_blis_shared_b_pc_outer_region`（`GemmDriverVariant::SharedBPcOuter`）を
A/B 一括計測ハーネスへ統合した。本節の B 共有化（案 B）と組み合わせた形で実装済みだが、
採用ゲート（実機 5 回中央値での受け入れ条件達成）は同じく未通過のため `#[cfg(test)]` 限定
のまま。詳細は `docs/perf/cpu-gemm-candle-cpu-retune.md` を参照。

## 出典

- イシュー #565（本ドキュメントの起票元）・#564／PR #701（E-8。MC/KC/NC パラメータ化・NC 拡大
  候補グリッドの根拠）・#556（`PanelBuffers` 導入・B packing スレッド間重複がスコープ外と
  明記した前例）・#479（GEMM 性能改善ツリー）
- `crates/backend-cpu/src/gemm_blis/mod.rs`（origin/main `bae1f0f` 時点。行番号は本コミット時点）
- `docs/perf/cpu-gemm-blocking-sweep.md`（MC/KC/NC 候補グリッド・footprint 指摘の先例〈§6〉）
- `docs/cpu-gemm-prefetch-decision.md`（fail-closed 記録様式・unsafe 導入時のレビュー体制の前例）
- `.claude/rules/coding-rust.md`（unsafe 最小化方針・カーネル境界検査規約）
- `.claude/rules/deps-policy.md`（許容依存 8 区分・rayon の位置づけ）
- `.claude/rules/security.md`（unsafe レビュー体制）
- `.claude/rules/out-of-scope-tracking.md`（後続タスク起票のユーザー承認要件）
