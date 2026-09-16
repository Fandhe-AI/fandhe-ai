# Metal f32 `sum` reduction（全要素・単一軸）の設計・実装記録

イシュー #1895・親 #1894（`MetalBackendOps::sum` 未実装により Metal 上の
`Var::sum` を loss とする backward テスト 11 件が判定不能。
`docs/perf/logs/metal-realdevice-phase2-2026-09-16/README.md` §3.1）。

## 1. 背景・目的

`MetalBackendOps::sum`（`crates/backend-metal/src/ops.rs`）は常に
`Unsupported` を返す。本イシューは是正の第 1 段として、Metal f32 `sum`
（全要素・単一軸）の**カーネル・起動 API・cfg なしホスト逐語モデル**を
追加する。`MetalBackendOps::sum` への結線・`Var::sum` のホスト
フォールバック可否の設計判断・M4 Max 実機実測は**#1896 のスコープ**であり
本イシューには含まない。

## 2. CPU 参照実装の演算順序（本実装が逐語再現する対象）

`fandhe_ai_backend_cpu::reduction::sum`（`crates/backend-cpu/src/
reduction.rs`）は `dim` の有無で異なる構造を持つ。

| 経路 | CPU 実装 | 意味 |
|------|---------|------|
| `dim=None`（`sum_slice_f64`） | `par_chunks(CHUNK=4096)` で分割 → 各チャンク内を `0.0f64` から index 順に逐次加算 → チャンク部分和（`Vec<f64>`。**narrow せず `f64` のまま**）を **チャンク番号順に `0.0f64` から逐次加算** → 最後に 1 回 `as f32` | 平坦な逐次和ではない。`numel > CHUNK` では単一スレッド逐次和と一般に bit がずれる |
| `dim=Some(axis)`（`axis_reduce_sum`） | 出力要素 `(o, i)` ごとに `0.0f64` から縮約軸 `k=0..axis_len` を昇順に逐次加算 → `as f32` | 平坦な逐次和（1 出力あたり） |

空縮約の意味論: `dim=None` で `numel==0` → `0.0`。`dim=Some` で
`axis_len==0` → 各出力 `0.0`（`acc=0.0` の downcast）。出力要素数 0
（`outer*inner==0`）→ 空。`-0.0` のみの入力は fold の初期値 `+0.0` に
より `+0.0` になる。

`CHUNK` は `backend-cpu::reduction` で `pub(crate)`（#1697 で可視化）の
ため backend-metal から直接参照できない。本実装の `reduce_model::
REDUCE_SUM_CHUNK` は同値の独立定数として持ち、ドリフトは
`reduce_source_evidence.rs::reduce_sum_chunk_constant_matches_rust_side`
（MSL 側リテラルとの一致）と `reduce_model` 単体テストのチャンク境界
感度テストの両方で検出する（値そのものが変わった場合、いずれも
CPU 参照実装との bit 不一致として顕在化する）。

## 3. カーネル構成（`shaders/reduce.metal`）

soft-f64 プリミティブ（`red_f64_clz64`／`red_f64_widen`／`red_f64_add`／
`red_f64_narrow`）は `scan.metal::scan_f64_*` から接頭辞置換で逐語複製
（`reduce_source_evidence.rs::
red_f64_primitives_match_scan_f64_primitives_verbatim_modulo_prefix` が
機械検証）。`mul`／`sub`／`div` は `sum` に不要なため含めない。

| カーネル | 並列単位 | 処理 |
|---------|---------|------|
| `reduce_sum_all_chunk_f32` | 1 スレッド = 1 チャンク（`REDUCE_SUM_CHUNK=4096` 要素） | `gid >= num_chunks` を境界検査 → `begin=gid*CHUNK`・`end=min(begin+CHUNK, numel)`（`ulong`）→ `acc=widen(0.0f)` から昇順に `red_f64_add` → `partial[gid]=acc`（`device ulong*`。**narrow しない**） |
| `reduce_sum_all_finalize_f32` | 単一スレッド（`gid != 0u` は早期 return） | `acc=widen(0.0f)` から `partial[0..num_chunks]` を昇順に `red_f64_add` → `out[0]=narrow(acc)` |
| `reduce_sum_axis_f32` | 1 スレッド = 1 lane（`lanes=outer*inner`） | `gid >= lanes` を境界検査 → `o=gid/inner`・`i=gid%inner` → `acc=widen(0.0f)` から `a=0..axis_len` を昇順に `red_f64_add` → `out[gid]=narrow(acc)` |

添字計算はすべて `ulong`（REQ-8）。この演算列はホスト `f64` 逐次参照
実装と **bit 完全一致**する（NaN のみクラス一致）。

### 既知の制約（並列度。`.claude/rules/out-of-scope-tracking.md` 対象）

`scan.metal` と同じ理由でブロック内対数段 reduction を採らない
（結合順序が変わり bit 一致契約が崩れるため）。したがって全要素縮約は
`numel/REDUCE_SUM_CHUNK` スレッドのみ・単一軸縮約は `outer*inner`
スレッドのみしか並列度が出ない。性能改善（並列度拡大）は結合順序を
維持したままでは実現できないため、別イシューへの切り出し候補として
記録するのみに留める。

## 4. 起動 API（`src/reduce.rs`）

- `MetalReduce::new(ctx)`: 3 パイプラインを実行時コンパイルして保持
  （`scan.rs::MetalScan::new` と同型）
- `MetalReduce::run_sum_all_f32(ctx, x) -> Result<f32, MetalError>`:
  `x` が空なら早期 `Ok(0.0)`。`reduce_model::plan_reduce_all` で
  `numel`／`num_chunks` の `u32` 収容を検証してから、単一の
  `ctx.dispatch_sync` クロージャ内で `reduce_sum_all_chunk_f32` →
  `reduce_sum_all_finalize_f32` を同一エンコーダへ順にエンコードする
  （2 段だが待つべきディスパッチは 1 回のため `mse.rs` の 2 回
  `ctx.encode` 方式ではなく `scan.rs` と同じ単一 `dispatch_sync` で
  完結できる）
- `MetalReduce::run_sum_axis_f32(ctx, x, outer, axis_len, inner) ->
  Result<Vec<f32>, MetalError>`: `lanes==0` は空 `Vec`、
  `axis_len==0 && lanes>0` は `vec![0.0; lanes]` を早期リターン（ともに
  ディスパッチを回避。`fandhe_ai_backend_cpu::reduction::
  axis_reduce_sum` の空縮約契約と同じ）
- 呼び出し元（`ops.rs`）が存在しない現時点では、本モジュール自身が
  `reduce_model::plan_reduce_*` による事前検査を行い
  `MetalError::InvalidReduceShape` へ写像する（`scan.rs` が
  「呼び出し元の検査結果を信頼しない」二重検査を行うのと対称の設計。
  #1896 で `ops.rs` 側にも同型の事前検査が追加される見込み）

encode-only（`ctx.encode` + `DispatchFailureCell`）は使わない:
`run_sum_all_f32`／`run_sum_axis_f32` はいずれも戻り値を同期的に消費
するため `ctx.dispatch_sync` で足りる（`scan.rs::MetalScan::run_scan`
doc コメントの理由をそのまま踏襲）。

`partial` バッファは `u64`（`f64` bit 表現の中間値。narrow しない）
のため `MetalIndexBuffer::new_zeroed_u64`（`sort.rs` が合成キー配列で
使う型）を流用する。

## 5. ホスト逐語モデル（`src/reduce_model.rs`。`cfg` なし）

- `sum_all_soft_f64(x: &[f32]) -> f32`: §2 のチャンク 2 段構成を逐語
  再現（`reduce_sum_all_chunk_f32`／`_finalize_f32` の Rust 側対応）
- `sum_axis_lane_soft_f64(xs: &[f32]) -> f32`: 1 lane 分のループ本体
  （`crate::soft_f64::sequential_sum_f32_bits` と数学的に同値だが、
  `reduce_sum_axis_f32` との対応を明示するため独立定義）
- `#[cfg(test)] sum_axis_soft_f64`: 形状全体への適用（テスト専用の
  Rust 側参照実装。`scan_model::scan_over_shape` と同型）
- `plan_reduce_all`／`plan_reduce_axis`・`ReducePrepareError`: `scan_
  model::plan_scan`／`ScanPrepareError` と同型のホスト側純関数検証
  （Linux でも単体テスト可能）

単体テスト（`reduce_model.rs` 内 `#[cfg(test)] mod tests`）は
`fandhe_ai_backend_cpu::reduction::sum` との bit 完全一致を以下の
観点で検証する: 複数サイズ（0・1・2・4095・4096・4097・8192・
`3*4096+1`）・チャンク境界を跨ぐ相殺列（`2^24` 等）・`-0.0` のみ・
NaN／inf 伝播・単一チャンク時の平坦逐次和との一致（構造の健全性）・
複数 rank・複数 `dim` の軸縮約・空縮約（`axis_len=0`）・空出力
（`outer=0`）・`plan_reduce_all`／`plan_reduce_axis` の境界値。

## 6. 数値契約

`.claude/rules/coding-rust.md`「勾配の長軸縮約」節・`crate::soft_f64`
と同じ binary64 逐次演算の 64bit 整数ソフトウェアエミュレーション。
CPU 参照実装（`fandhe_ai_backend_cpu::reduction::sum`）と **bit 完全
一致**（NaN のみクラス一致）。tolerance・REQ-2 判定・baseline は変更
しない（本イシューは新規カーネル追加のみで既存契約に触れない）。

## 7. テスト構成

- `crates/backend-metal/src/reduce_model.rs`（`#[cfg(test)] mod
  tests`）: Linux 実行可能・CPU 参照実装との bit 完全一致（§5）
- `crates/backend-metal/tests/reduce_source_evidence.rs`: Linux 実行
  可能・`include_str!` 文字列証跡（REQ-8 境界検査・soft-f64 使用・
  `REDUCE_SUM_CHUNK` 定数一致・`scan.metal` との soft-f64 プリミティブ
  逐語一致）
- `crates/backend-metal/tests/reduce_parity.rs`
  （`#![cfg(target_os = "macos")]` + `#[ignore]`）: 実機実測は本イシュー
  のスコープ外（Mac セッションへ申し送り。`docs/perf/logs/
  metal-reduce-1895/README.md` 参照）。Linux では
  `make check-cross-metal-tests`（`cargo check -p
  fandhe-ai-backend-metal --tests --target aarch64-apple-darwin`）で
  型検査のみ通す

## 8. スコープ外（#1896 等へ引き継ぐ）

- `MetalBackendOps::sum` 結線・`context_cache::cached_reduce`・
  `map_reduce_prepare_error`・`docs/backend-dtype-dispatch-design.md`
  §14.2 記述更新
- M4 Max 実機での `reduce_parity.rs` 実測（11 件の backward テスト
  判定不能の解消確認を含む）
- `Var::sum` のホストフォールバック可否（facade 公開契約に関わる
  設計判断。ユーザー承認を要する）
- `max`／`min`／`mean` の Metal カーネル
- `typed_f16`／`typed_bf16` の `sum` 自動有効化の実機確認
- 並列度改善（結合順序を変えずには実現できないため対象外のまま。
  §3「既知の制約」参照）
