# 決定性モード（`set_deterministic`）設計判断記録

イシュー #2157（親 #2131「Phase 5: API 網羅の深掘り」）。PyTorch
`torch.use_deterministic_algorithms` に当たる決定性モードの API を
autodiff に追加するにあたり、非決定的な経路（rayon 縮約順・atomic
演算等）を棚卸しし、結果に基づいて実装の形を確定した設計判断記録。

## §0 結論

**CPU（`backend-cpu`）・ホスト参照実装（`autodiff::eval`）のいずれも、
本番経路に「結果がスレッド数・実行順に依存する非決定的な縮約」は
1 件も見つからなかった**（§2 棚卸し）。したがって:

- `fandhe_ai_autodiff::determinism::set_deterministic(bool)` は
  **状態を記録するのみで実行時の分岐・拒否は行わない no-op 契約**と
  する。呼び出し元のない `AutodiffError` 新規 variant・使われない
  `ensure_deterministic()` ヘルパー等の死蔵コードは追加しない
  （`.claude/rules/coding-rust.md` の `#[allow(dead_code)]` 濫用禁止
  方針に反するため）。
- facade（`fandhe_ai`）への公開は本イシューでは**未承認のまま保留**
  する（§6）。内部クレート（`fandhe_ai_autodiff`）限定の到達入口として
  実装する。
- GPU（CUDA／Metal）の `Tape` は棚卸し対象外（別 issue。§5・§7）。
  `set_deterministic` はどの `Tape` に対しても等しく no-op であり、
  GPU 経路を拒否も検査もしない。

## §1 背景・要件の要約

- PyTorch の `torch.use_deterministic_algorithms(True)` に相当する、
  プロセスワイドな決定性モード API を追加する。
- 非決定的な経路の棚卸しを行い設計 doc に記録する。棚卸し結果に
  よって実装の形が変わる（本 doc §0 の結論のとおり「該当なし」）。
- 受け入れ条件: (1) 棚卸しと fail-closed 契約を記録した設計 doc
  (2) 棚卸し結果に基づく仕組みの確定。
- 変更しないもの: tolerance・baseline・`Cargo.toml` 依存・ガードレール
  閾値・`docs/spec/`。`fandhe-ai =0.9.0` の公開 API は追加のみで破壊
  しない。新規 `unsafe` なし。

## §2 棚卸し

### §2.1 rayon 使用ファイルの棚卸し（`crates/backend-cpu/src`）

`par_iter`／`par_chunks`／`into_par_iter`／`rayon::` のいずれかを含む
ファイルは **24 件**（実装時の再 grep で確定。計画時点の見込み 22 件
から `gemm.rs`・`gb10_affinity.rs` の扱いを含め再集計して訂正）:

| ファイル | 分類 | 決定性の根拠 |
|---|---|---|
| `reduction.rs` | 縮約カーネル | モジュール doc「決定性契約」: 軸指定は出力要素側のみ並列化・縮約軸は逐次累積。全縮約は固定 `CHUNK` の `par_chunks` → `IndexedParallelIterator::collect`（rayon が入力順保持を保証）→ チャンク番号順に逐次結合。`logsumexp`／`vector_norm_p` の全縮約は `par_chunks` を使わず単一逐次 `f64` fold（eval 参照実装との bit 一致契約。イシュー #2147） |
| `mse.rs`・`bce.rs`・`huber.rs`・`kl_div.rs` | 損失関数 forward | `reduction.rs` と同型: 固定 `CHUNK` の `par_chunks` → チャンク内逐次 fold → `collect::<Vec<_>>()` → `into_iter().fold()` でチャンク番号順に逐次結合（`bce.rs` 129-141・`kl_div.rs` 97-109・`huber.rs` 149-161 実測） |
| `softmax.rs`・`layer_norm.rs`・`rmsnorm.rs`・`batch_norm.rs` | 正規化・活性化 | 行（サンプル）方向のみ並列化。行内の縮約（二乗和・max 等）は単一 worker 内で逐次完結するため共有アキュムレータへの並列書き込みが発生しない |
| `elementwise.rs`・`fused_elementwise.rs`・`scalar_elementwise.rs` | 要素独立演算 | 出力要素ごとに独立（`par_iter_mut` で書き込み先が要素ごとに排他）。縮約を伴わないため決定性契約の対象外（常に決定的） |
| `nll.rs`・`typed_f64.rs`・`rnn_cell.rs`・`pooling.rs`（`gemm.rs` に統合） | 損失・RNN セル・pooling | `reduction.rs`／損失関数群と同じチャンク分割＋逐次結合、または要素独立の並列化（`typed_f64.rs` モジュール doc「並列化・決定性」参照） |
| `gemm.rs`・`gemm_blis/mod.rs`・`gemm_blis/partition.rs` | GEMM | §2.2 参照（静的パーティション） |
| `gb10_affinity.rs`・`small_shape_thread_cap.rs`・`thread_limit.rs` | スレッドプール管理 | 縮約を一切行わない（専用 `rayon::ThreadPool` の構築・pin・スレッド数決定のみ）。既存 GEMM ロジックへ手を加えず `f()` をそのまま実行するラッパーのため決定性契約の対象外。GB10 大コア pin（`gb10_affinity.rs`）・macOS P/E 非対称キャップ（`small_shape_thread_cap.rs`）はいずれも実機検出ゲート付きで、非対象環境（本リポジトリの CI・ubuntu-latest 含む）では常に `f()` 直呼びへフォールバックする |
| `lib.rs` | クレートルート | コメント中の `par_chunks_mut` 言及のみ（コード本体に rayon 呼び出しなし） |
| `gemm_prefetch_bandwidth_diag_tests.rs` | テスト専用 | `crates/backend-cpu/src/lib.rs:191` で `#[cfg(all(test, target_arch = "aarch64"))]` ゲート済み（本番ビルドに一切含まれない診断用ベンチテスト）。本番インベントリの対象外として明示除外 |

### §2.2 GEMM（`gemm_blis`）のスケジューリング

- **静的パーティション経路（既定・本番結線）**: `dispatch_two_d_dynamic`
  → `gemm_blis_two_d_dynamic_region`
  （`crates/backend-cpu/src/gemm_blis/mod.rs:2849-2895`）は
  `partition::job_grid`（`num_threads`・`jobs_per_worker` から**純粋
  関数**として事前計算する静的な 2D ジョブ分割）→
  `jobs.par_iter_mut().try_for_each(...)` という構成。各ジョブは出力
  `C` の互いに素な領域を担当し、共有アキュムレータへの並列書き込みは
  発生しない。`num_threads`・`jobs_per_worker` が同一なら `job_grid`
  の出力は決定的（純粋関数）であり、`par_iter_mut` はワーカー間の
  ジョブ処理順序に依存しない出力（ジョブごとに書き込み先が排他）を
  返す。
- **動的カウンタ方式（未結線・テスト専用診断コード）**: `AtomicUsize`
  の `fetch_add` によるパネル動的配布（`gemm_blis_ic_dynamic_region`。
  `crates/backend-cpu/src/gemm_blis/mod.rs:2408-2506`）は関数自体・
  依存 `use`（`AtomicUsize`／`Mutex`。`mod.rs:81-84`）ともに
  **`#[cfg(test)]` ゲート**であり本番ビルドに一切含まれない。実測で
  `fetch_add` の出現は crate 全体でこの 1 箇所のみ（計画時点の見込み
  「本番 1 箇所」から実装時再確認で訂正: **本番 0 箇所**。テスト専用
  診断コードが 1 箇所）。
  行パネルをどの worker が・どの順序で claim するかは、互いに素な C
  要素集合の処理順序を並び替えるだけで、同一要素の pc 昇順・カーネル
  内 p 昇順の蓄積順序には影響しない（コメント `mod.rs:2395-2406`）ため、
  仮に本番結線されても数値結果はスレッド割り当てに依存しない。

### §2.3 rayon 並列イテレータと縮約マーカーの走査

`crates/backend-cpu/src` 全体を、rayon 並列イテレータ（`.par_iter(`／
`.par_chunks(`／`.par_chunks_mut(`／`.into_par_iter(`／`.par_iter_mut(`／
`.par_bridge(` ほか `.par_` 接頭辞のメソッド呼び出し全般・`T::par_iter`
のようなパス参照形・メソッド値形）と縮約マーカー（`.sum()`／
`.sum::<T>()`／`.reduce(`／`reduce_with(`／`.product(`／`.fold(` 系／
`.try_fold(`／`.try_reduce(` 系。`ParallelIterator::sum(...)` のような
UFCS〈fully-qualified〉呼び出し構文を含む）の共起を検出する条件で
走査した結果、**該当箇所は 0 件**（実装時の実測。
`crates/backend-cpu/tests/determinism_inventory.rs` が同条件を固定
fail-closed 検査する）。「同一文中に共起」という単純な判定は、6 ラウンド
にわたる検出漏れ指摘（PR #2274。関数本体最初の文・型注釈 `let`・
turbofish・`.par_bridge(`・ブロック式初期化子・未閉じ `{` の残余 `}` 等）、
さらにその後の敵対的レビューで指摘された過検出・検出漏れ（`fn` という
語の出現だけで汚染集合を全消去する不具合・UFCS 呼び出しの検出漏れ・
Vec collect で連鎖が切れた初期化式の誤汚染）、さらにそれらの是正後に
指摘された 3 件（無関係なタプル要素としての Vec collect が並列マーカー
の連鎖を過剰に遮断していた codex-review 指摘・ブロック式に連鎖した
縮約〈`{ .. }.sum()` 等〉を検出できなかった Cursor Bugbot 指摘・その
初回是正〈ブロック本体を合成マーカー文字列へ書き換える方式〉が短い
ブロック本体〈`{it}` 等・7 バイト未満〉を検出できなかった Critical
指摘）を経て、以下の走査方式へ置き換え済み（詳細な実装形は
`crates/backend-cpu/tests/determinism_inventory.rs` 冒頭の `//!` を正
とし、本節では要点のみを記す）:

1. **ブロック深さを考慮した文境界**: `()`／`[]`／`{}` の深さを追跡し、
   深さ 0 の `;` および深さ 0 に戻る `}`（直後が `.`／`?`／二項演算子／
   `else`／`catch` 等の式継続でないもの）を文の終端とする。
2. **縮約マーカーの括弧深さ条件**: 並列マーカーより後方かつ**同じか
   浅い**括弧深さに縮約マーカーが現れる場合のみ共起とみなす
   （`data.par_iter().map(..).sum::<f32>()` は検出、
   `data.par_chunks(n).map(|c| c.iter().sum::<f32>()).collect(..)`
   〈チャンク内逐次和〉は非検出）。加えて `ParallelIterator::sum(...)`・
   `Trait::reduce(...)` のような UFCS 呼び出しも別途走査し、呼び出し
   引数リストの内側に並列マーカーまたは汚染識別子があれば 1 件と
   カウントする（同一文内で他規則と二重計上しない）。
3. **`Vec` への collect による並列→逐次の連鎖の遮断（同一メソッド
   連鎖上に限る）**: 並列マーカー（または後述の汚染識別子）と縮約
   マーカーの間に `.collect::<Vec<`／`.collect_into_vec(` が挟まる
   場合は接続しないとみなす。遮断は順序保持が型で明示される `Vec`
   への collect に限り、型推論任せの `.collect()` や `HashMap`／
   `HashSet` 等への collect は反復順序が決まらない場合があるため
   遮断しない（fail-closed 側）。本クレートの縮約実装は
   `data.par_chunks(..).map(..).collect::<Vec<_>>().into_iter().fold(..)`
   （`mse.rs::mse_sum_sq_f32` ほか `bce.rs`／`huber.rs`／`kl_div.rs`／
   `nll.rs`／`reduction.rs`／`typed_f64.rs` で広く使われるイディオム）
   という形を取っており、`.collect(` でインデックス順の `Vec` へ一旦
   確定したあとの `.into_iter()` 以降は通常の逐次 `Iterator` になる
   ため決定的である。この collect を境界とみなさないと、この正当な
   パターンを誤って「並列縮約」と判定してしまう（実測で判明した
   スキャナ側の過検出）。ただし「同じか浅い深さの collect なら一律に
   遮断する」という単純な深さ比較は、`let it = (data.par_iter(),
   other.collect::<Vec<_>>()).0;` のように無関係なタプル要素として
   同居するだけの collect まで遮断してしまう過検出を生む
   （codex-review 指摘。回帰前は `it` が誤って非汚染と判定されていた）。
   `is_same_method_chain` が「到達点（collect）自身の深さを連鎖の
   合流点とみなし、そこまでテキスト深さが一度も下回らず、かつ合流点
   の深さちょうどに現れる文字がメソッド連鎖の構成要素（識別子・`.`／
   `:`・turbofish の `<>`・`?`・空白・呼び出しの `(`・`)`）だけである
   こと」を判定し、これを満たす collect のみを遮断とみなす（`,`・
   `;`・`=`・二項演算子・`|` 等が合流点の深さに現れたら遮断しない）。
   同じ判定は下記 4 の汚染**発生源**の判定にも適用する: `let parts =
   data.par_iter().map(f).collect::<Vec<f32>>();` のように初期化式が
   Vec 確定で終わる場合は束縛先を汚染しない（敵対的レビュー指摘。
   これを適用しないと `let s = parts.iter().sum::<f32>();` のような
   完全に逐次な後続コードまで並列縮約と誤検出する）。「到達点自身の
   深さ」を合流点とする設計は、`.zip(other.par_chunks(..))` のように
   起点より深い位置にある並列マーカーが、外側で閉じたあとの collect
   によって正しく遮断されることも保証する（実ソース実測で判明した
   過検出の是正。`crates/backend-cpu/tests/determinism_inventory.rs::
   count_par_reduce_does_not_flag_zip_argument_marker_blocked_by_outer_collect`
   固定）。
4. **レキシカルスコープを持つ関数単位の汚染追跡**: 文中の任意位置の
   `let`（型注釈・タプルパターン `(a, mut b)` を含む）または `let` を
   伴わない単純代入 `IDENT = <式>;` を検出し、初期化式（ネストした
   ブロックの中身も含む）が 3 の規則で汚染源を含めば束縛名を汚染
   する。クロージャ本体（`let f = |d| d.par_iter();`）や汚染識別子の
   再代入（`let it2 = it;`）も初期化式のテキストをそのまま見る規則で
   自然に捕捉する。汚染識別子が後続の文で識別子境界一致し、かつ
   同じか浅い括弧深さで縮約マーカーが現れたら 1 件とカウントする。
   汚染集合は**レキシカルスコープ単位**で再帰的に管理する（旧実装は
   「`fn` という語が文の先頭付近に現れたら丸ごと `clear()` する」
   ヒューリスティックだったため、ローカル `fn`・`fn(i32) -> i32` 型
   注釈・ローカル `impl` を挟むだけで汚染集合が全消去される不具合が
   あった。敵対的レビュー指摘）: fn アイテムの本体は fn がローカル
   変数をキャプチャしないため空集合から始め、それ以外のネストした
   ブロック（if／for／loop／match アーム／ブロック式／クロージャ
   本体／impl・mod 内の非 fn アイテム）は現在の汚染集合を引き継ぐ。
   ネストしたブロック内の `let` による汚染はそのブロックに閉じ親へ
   漏らさないが、`let` を伴わない単純代入による汚染だけは親スコープ
   へ伝播する（fn アイテムの境界をまたぐ場合は伝播しない）。
5. **字句前処理**: `//`／`/* */`（ネスト対応）コメント・`"..."` 文字列
   に加え、raw string（`r"..."`／`r#"..."#`）・byte string
   （`b"..."`／`br#"..."#`）・char/byte literal（`'{'`／`';'`／
   `'\''`／`'\u{7b}'` 等）を空白へ置換する。ライフタイム（`'a`／
   `'static`）は char literal と誤認せずそのまま残す。
6. **ブロック式に連鎖した縮約の検出（長さに依存しない仮想マーカー）**:
   同一文中の共起判定は、文内にネストした `{...}` ブロック本体を
   塗りつぶしたテキストに対して行う（ネスト内は上記 4 の再帰が別の
   レキシカルスコープとして検査するため二重計上しない）。しかし単純な
   空白塗りつぶしだけでは、`{ let x = 1; data.par_iter() }.sum()`・
   `if c { a.par_iter() } else { b.par_iter() }.sum()`・
   `unsafe { data.par_iter() }.sum::<f32>()`・`match k { _ =>
   data.par_iter() }.sum::<f32>()` のようにブロック式の直後に連鎖する
   縮約が、同一文判定・UFCS 判定のどちらからも見えなくなる（Cursor
   Bugbot 指摘）。初回の是正はブロック本体を合成の並列マーカー文字列
   （`.par_iter(` 等。長さを保つよう空白で埋める）へ実際に書き換える
   方式だったが、ブロック本体が合成マーカーの最小長（7 バイト）未満
   （`{it}`・`{ acc }` のような短い汚染識別子 1 つだけのブロック式）
   だと検出できない Critical な抜けがあった（敵対的レビュー指摘）。
   現行方式はテキストへ何も書き込まず、ブロック本体が「生きている
   （3 の規則で collect に遮断されていない）並列マーカー」または
   現スコープの汚染識別子を含む場合に、ブロックの開き `{` のバイト
   位置を「仮想マーカー位置」という別チャネル（深さはブロックの外側
   ——`{` 自身の位置——の深さ）として返し、同一文判定・UFCS 判定・
   連鎖の遮断判定のすべてが実マーカーと仮想マーカーの両方を起点として
   扱う。ブロック本体の長さに一切依存しないため、`{it}`・`{ acc }`の
   ような短いブロック式も正しく検出する。ブロック本体の中で
   par_chunks → collect が同一連鎖上で完結している場合（`if c {
   data.par_chunks(4).map(f).collect::<Vec<_>>() } else { vec![] }
   .into_iter().sum::<f32>();`）は仮想マーカーを記録しない（0 件の
   まま）。

見つかった `.sum()` 等はいずれも逐次 `std::iter::Iterator::sum()`
（`ops.rs::gemm_checksum` の `out.iter().map(|&x| x as f64).sum()` 等）
またはテストコード内の `naive_sum`／`naive` 参照実装で、rayon 並列
イテレータの縮約ではない。

### §2.4 atomic 使用の棚卸し

`crates/backend-cpu/src` 全体で `fetch_add`／`fetch_sub`／
`compare_exchange`／`fetch_or`／`fetch_and`／`fetch_max`／`fetch_min`
を検索した結果、実際のコード上の出現は §2.2 で述べた
`gemm_blis_ic_dynamic_region`（`#[cfg(test)]` ゲート）内の 1 箇所のみ
（他はすべてコメント中の言及）。本番経路に atomic 蓄積は存在しない。

### §2.5 `autodiff`／`tensor-core` の rayon 依存

`crates/autodiff/Cargo.toml`・`crates/tensor-core/Cargo.toml` の
`[dependencies]` に `rayon` は含まれない（`grep -rn rayon
crates/autodiff/src crates/tensor-core/src` はコメント中の言及
〈`default_ops.rs:33`・`backend_ops.rs:1534`〉のみで、実コード上の
使用なし）。backward はテープを逆順に逐次走査するのみ。

### §2.6 `HashMap`／`BTreeMap` 反復の棚卸し

- `crates/autodiff/src/grad.rs::scatter_overwrite_last_writer_mask`
  （4410-4441 行）: `last_writer: HashMap<usize, usize>` へは
  `index_data.iter().enumerate()`（決定的順序）で挿入するため
  `last_writer` 自体の中身は挿入順に依存せず確定する。その後
  `for &flat in last_writer.values() { mask[flat] = 1.0; }` は
  `HashMap` の内部反復順（ハッシュシードにより実行ごとに変わりうる）
  に依存するが、**書き込む値が一律 `1.0` かつ書き込み先 `flat` は
  `last_writer` の内容（挿入順に依存しない）そのもの**であるため、
  反復順が異なっても最終的な `mask` 配列は不変（冪等な上書き）。
- `crates/autodiff/src/einsum.rs`（174 行付近）: 省略添字の並びは
  `BTreeMap<char, usize>` で ASCII 昇順を保証しており非決定性なし。

### §2.7 グローバル RNG（`tensor-core/src/rng.rs`）

単一スレッドからの呼び出し列であれば決定的。複数スレッドから並行に
消費する順序は既存 doc で保証対象外と明記済み（PyTorch と同じ制約）。
本イシューは新たな保証を追加しない。

### §2.8 GPU（CUDA／Metal）

float の `atomicAdd` は使わない方針（ソース検査テストあり:
`kernels_huber.rs:231`・`kernels_rnn_cell.rs:258`・
`kernels_typed_f64.rs:451`）。persistent GEMM のタイルキューカウンタ
（`kernels_tiled_pipeline_128x64.rs:1185-1216`）はスケジューリング
専用。ただし GPU 全体の体系的棚卸しは本イシューのスコープ外（§5・
§7）。

## §3 契約

### §3.1 保証範囲

- 対象: `Tape::new()`（`NaiveOps`）と
  `Tape::new_with_ops(Box::new(CpuBackendOps::new()))`。
- 同一バイナリ・同一マシン上で、実行ごと（run-to-run）およびスレッド
  数に対して bit 同一になる。
- ISA 間・マシン間 bit 一致は本イシューでは保証を宣言しない
  （GEMM の ISA 間 bit 一致は既存の別契約〈`docs/*-parity-*-decision.md`
  群〉を参照するのみに留める）。
- 並行スレッドからのグローバル RNG 消費順は対象外（§2.7）。

### §3.2 no-op 契約

`set_deterministic(true)` は §2 の棚卸し結果（該当経路なし）に基づき
拒否も分岐もしない。`is_deterministic()` は現在の状態を返すのみ。

### §3.3 fail-closed 化の担保（将来の回帰防止）

`crates/backend-cpu/tests/determinism_inventory.rs`（54 テスト。すべて
green）に、次の fail-closed ソース走査テストを置く:

- `rayon_marker_files_match_fixed_allowlist`: rayon 並列イテレータ
  使用ファイル集合を allowlist（24 件。§2.1）で固定する。
- `no_rayon_parallel_reduce_cooccurrence_in_backend_cpu_src`: §2.3 の
  走査方式（ブロック深さ・括弧深さ・UFCS 呼び出し検出・同一メソッド
  連鎖上に限った `Vec` collect 遮断・レキシカルスコープを持つ関数
  単位の汚染追跡・ブロック式に連鎖した縮約の検出）で並列縮約の共起を
  検出し、`crates/backend-cpu/src` 実測で**0 件**であることを固定
  する。
- `atomic_rmw_occurrences_match_expected_test_only_count`: atomic
  蓄積箇所を 1 件（`gemm_blis/mod.rs` の `#[cfg(test)]` 限定診断コード
  のみ）で固定する。
- `no_parallel_iterator_valued_function_signature_in_backend_cpu_src`:
  §2.3 の走査（トークン走査）が追跡しない**既知の限界**——関数引数・
  戻り値経由で並列イテレータの値が流れる経路（例: 並列イテレータを
  引数として受け取り関数内部で縮約する設計）——を別軸で fail-closed
  に検出する。戻り値型・引数型に `ParallelIterator`／
  `IndexedParallelIterator` を含む関数定義（`use` 文・コメントは除く）
  が `crates/backend-cpu/src` に**0 件**であることを固定する。この
  経路が導入された場合、トークン走査による値の流れの追跡拡張か、
  本テストへの個別 exemption 追加のいずれかを検討する。
- 上記に加え、レキシカルスコープ管理・UFCS・`::par_` パス参照・
  Vec collect 遮断（同一メソッド連鎖上に限る）・ブロック式に連鎖した
  縮約の検出それぞれの単体回帰テスト（ローカル `fn`・`fn(i32) ->
  i32` 型注釈・ローカル `impl` を挟んでも汚染集合が維持されること／
  別々の fn 間で汚染が漏れないこと／ネストしたブロック内の `let` 汚染
  は漏れず単純代入の汚染だけが伝播すること／無関係なタプル要素として
  の collect が誤って連鎖を遮断しないこと／`.zip(other.par_chunks(..))`
  のような実ソースの合流パターンで誤検出しないこと／`{ .. }.sum()`・
  `if c { .. } else { .. }.sum()`・`unsafe { .. }.sum()`・`match k
  { .. }.sum()` の各形が検出されブロック内で collect 完結する場合は
  検出されないこと／`{it}`・`{ acc }`（詰めた形・rustfmt の空白入り
  形の両方）のような短い汚染識別子 1 つだけのブロック式・
  `match k { 0 => {a}, _ => {b} }.sum::<f32>()` の各腕が検出され、
  未汚染の `{x}.sum()` は検出されないこと等。敵対的レビューで実測
  確認した断片をそのまま固定）を置く。

新しい非決定的経路が将来追加された場合、上記いずれかの固定値との
不一致で CI が fail-closed に落ちる。その時点で本 doc §7 の規則に
従い、決定的実装への置き換えか決定性モード中の拒否（`AutodiffError`
への `#[non_exhaustive]` variant 追加）のいずれかを選ぶ。

## §4 置き場所の逸脱理由

Issue 本文は `var.rs` への追加を示唆していたが、`Var` への inherent
メソッド追加は facade（`fandhe_ai`）が `Var` を直接再エクスポート
している（`crates/facade/src/lib.rs:184`）ため即座に facade 公開面へ
出てしまう。先例 #2144（`matrix_ops`）・#2147（`reduce_ops`）と同じ
判断枠組みにより、承認（§6）が取れるまでは `Var` の外の自由関数
モジュール `crates/autodiff/src/determinism.rs` に置き、facade 側の
保留ガードで到達不能にする。

## §5 GPU の扱い

CUDA／Metal の `Tape` に対して `set_deterministic` を呼んでも拒否・
検査は行わない（未検証のまま no-op）。理由:

- GPU の非決定性棚卸しは Issue が別 issue へ明示的に切り出している
  （§7）。
- 拒否にすると `Tape::new_with_ops` のような fallible でない入口へ
  フックが必要になり、GPU の決定的 opt-in（TF32 等）まで一律に塞いで
  しまう。

## §6 承認事項（未承認）

- **facade 公開**（`fandhe_ai::set_deterministic`／
  `fandhe_ai::is_deterministic` の crate ルート公開）は未承認のまま
  対象外とする。`DeterminismHoldDoctestGuard`（`crates/facade/src/
  lib.rs`）＋`crates/facade/tests/api_surface.rs` の 4 テストで多層
  固定する。
- spec（`docs/spec/`）への変更提案はなし。

## §7 スコープ外・申し送り

- GPU（CUDA／Metal）の非決定性の体系的棚卸しと、必要なら fail-closed
  化は別 issue とする（起票はユーザー承認後）。
- facade 公開（§6）の承認後は `crates/facade/src/lib.rs::
  DeterminismHoldDoctestGuard`・`crates/facade/tests/api_surface.rs`
  の対応する 4 テストを削除し、`fandhe_ai::set_deterministic`／
  `fandhe_ai::is_deterministic` を追加する。

## §8 実装記録

- `crates/autodiff/src/determinism.rs`（新規）: `set_deterministic`・
  `is_deterministic`（`AtomicBool`・`SeqCst`）。
- `crates/autodiff/src/lib.rs`: `pub mod determinism;` を追加。
- `crates/autodiff/tests/determinism_mode.rs`（新規）: グローバル状態
  を扱う単一 `#[test]` に集約（既定 false・set/get・冪等性・no-op
  bit 一致を検証）。
- `crates/backend-cpu/tests/determinism_inventory.rs`（新規）: ソース
  走査インベントリ（§2 allowlist の固定）＋スレッド数不変性の
  end-to-end テスト。
- `crates/facade/src/lib.rs`: `DeterminismHoldDoctestGuard`（`#[cfg(
  doctest)]`）追加。
- `crates/facade/tests/api_surface.rs`: 対応する 4 テスト追加。
- `docs/README.md`・`docs/compat-api-scope.md`: 索引・Tier 1 表への
  最小追記。
