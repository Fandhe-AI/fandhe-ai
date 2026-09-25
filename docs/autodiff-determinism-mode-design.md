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

### §2.3 `.sum()`／`.reduce(`／`reduce_with` の走査

`crates/backend-cpu/src` 全体を「rayon 並列イテレータ（`par_iter`／
`par_chunks`／`par_chunks_mut`／`into_par_iter`／`par_iter_mut`）と
`.sum()`／`.reduce(`／`reduce_with` が同一文中に共起する」条件で走査
した結果、**該当箇所は 0 件**（実装時の実測。
`crates/backend-cpu/tests/determinism_inventory.rs` が同条件を固定
fail-closed 検査する。`par_chunks_mut` マーカーは codex-review 指摘
〈PR #2274〉により追加し、追加後も 0 件を再確認済み）。見つかった
`.sum()` はいずれも
逐次 `std::iter::Iterator::sum()`（`ops.rs::gemm_checksum` の
`out.iter().map(|&x| x as f64).sum()` 等）またはテストコード内の
`naive_sum` 参照実装で、rayon 並列イテレータの `.sum()`／`.reduce(`
ではない。

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

`crates/backend-cpu/tests/determinism_inventory.rs` に、rayon 使用
ファイル集合・atomic 蓄積箇所・`.sum()`／`.reduce(` 共起箇所を
allowlist で固定するソース走査テストを置く。新しい非決定的経路が
将来追加された場合、この allowlist との不一致で CI が fail-closed に
落ちる。その時点で本 doc §7 の規則に従い、決定的実装への置き換えか
決定性モード中の拒否（`AutodiffError` への `#[non_exhaustive]`
variant 追加）のいずれかを選ぶ。

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
