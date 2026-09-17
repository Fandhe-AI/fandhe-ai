# Metal f32 `sum` reduction（全要素・単一軸）の設計・実装記録

イシュー #1895・親 #1894（`MetalBackendOps::sum` 未実装により Metal 上の
`Var::sum` を loss とする backward テスト 11 件が判定不能。
`docs/perf/logs/metal-realdevice-phase2-2026-09-16/README.md` §3.1）。

## 1. 背景・目的

`MetalBackendOps::sum`（`crates/backend-metal/src/ops.rs`）は常に
`Unsupported` を返していた。#1895 は是正の第 1 段として、Metal f32
`sum`（全要素・単一軸）の**カーネル・起動 API・cfg なしホスト逐語
モデル**を追加した。`MetalBackendOps::sum` への結線・`Var::sum` の
ホストフォールバック可否の設計判断・M4 Max 実機実測は**イシュー
#1896** のスコープであり、結線・設計判断は §9・§10 に、実機実測は
Mac セッションへの申し送り（`docs/perf/logs/
metal-reduce-sum-wiring-1896/README.md`）として記録する。

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
- `ops.rs::MetalBackendOps::sum`（イシュー #1896 で結線済み。§9）は
  起動前に `reduce_model::plan_reduce_*` を先出しして検査するが、本
  モジュール自身も同型の事前検査を維持し `MetalError::
  InvalidReduceShape` へ写像する（`scan.rs` の「呼び出し元の検査結果
  を信頼しない」二重検査と同じ設計）

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

## 8. スコープ外（後続イシューへ引き継ぐ）

- `max`／`min`／`mean` の Metal カーネル自体（`sum` は #1896 で結線
  済み。`argmax`／`argmin` は #1951 で実装済み〈§11〉。`mean` は
  `sum` の結果をホスト側で 1 回除算する合成実装のため対象外のまま
  自動的に有効化されている）
- M4 Max 実機での結線後の実測（§9・11 件の backward テスト判定不能の
  解消確認を含む。`docs/perf/logs/metal-reduce-sum-wiring-1896/
  README.md` へ申し送り）
- `Var::sum` ホストフォールバックの実装（§10 の設計判断は案 C〈実装
  しない〉で 2026-09-17 ユーザー承認済み・確定。実装対象から外れた）
- 並列度改善（結合順序を変えずには実現できないため対象外のまま。
  §3「既知の制約」参照）

## 9. 実装記録（`MetalBackendOps::sum` 結線。イシュー #1896）

`crates/backend-metal/src/ops.rs::MetalBackendOps::sum` を
`context_cache::cached_reduce`（新設。`cached_unique` と同型の
プロセス内キャッシュ）経由で `reduce::MetalReduce` へ結線した。
`context_cache::cached_scan`／`ops.rs::run_scan`（#1740）と同一の
検査順序を踏襲する:

1. [`fandhe_ai_tensor_core::ops_shape::reduce_out_shape`] で `dim` の
   範囲を検査し出力 shape を導出する（デバイス初期化前。範囲外 `dim`
   は `ShapeMismatch` として即座に返り、`context_cache::cached_context`
   に一切触れない）
2. **0 サイズ契約を要素数積の検査より先に処理する**: `shape` が 0 を
   含む場合、`dim=None` なら `0.0` を、`dim=Some` かつ出力 shape に
   0 を含むなら空テンソルを、`dim=Some` かつ `shape[axis]==0` のみ
   （他軸は非零）なら出力を `0.0` で埋めて早期リターンする。**この
   順序は巨大な非零軸を含む 0 サイズ shape（例
   `[1<<40, 0, 1<<40]`）で `checked_numel` の中間積 overflow により
   誤って `ShapeMismatch` を返すのを防ぐために必須**（`run_scan` と
   同じ理由。§7 の `backend_ops_sum_matches_cpu_bit_exact` が直接
   検証する）
3. [`crate::gather_scatter_model::checked_numel`] で要素数積の
   `usize` オーバーフローを検査（`a.numel()` を無検査で呼ぶ前）
4. `reduce_model::plan_reduce_all`／`plan_reduce_axis` を先出しし、
   カーネル `uint` 引数の上限超過（`u32::MAX` 超）を
   `map_reduce_prepare_error` で `BackendError::Unsupported` へ写像
   する
5. `context_cache::cached_context`／`cached_reduce` 経由で
   `reduce::MetalReduce::run_sum_all_f32`／`run_sum_axis_f32` へ委譲

エラー写像表:

| 状況 | 戻り値 |
|---|---|
| `dim` が軸範囲外 | `BackendError::ShapeMismatch(AxisOutOfRange)`（デバイス非接触） |
| 要素数積 `usize` オーバーフロー | `BackendError::ShapeMismatch(ElementCountOverflow)`（デバイス非接触） |
| カーネル `uint` 引数上限超過（`numel`／`num_chunks`／`lanes`／`axis_len`／`inner` が `u32::MAX` 超） | `BackendError::Unsupported`（`Var::sum` はホストフォールバックを持たず呼び出し元へそのまま伝播。§10 案 C・2026-09-17 ユーザー承認済み） |
| 内部契約違反（呼び出し元の検査をすり抜けた `reduce.rs` 側の二重検査失敗） | `BackendError::KernelLaunchFailed`（`map_metal_error` の wildcard arm） |
| 上記以外 | `Ok`。CPU 参照実装と bit 完全一致 |

**変更したテスト**（`sum` の `Unsupported` 前提を撤去し、`max` のみへ
縮小・改名。新規 `#[ignore]` 実機テストで `sum` を検証）:

| ファイル | 変更 |
|---|---|
| `crates/backend-metal/tests/backend_ops_real_device.rs` | `reduction_remains_unsupported_without_device_init` → `max_remains_unsupported_without_device_init`（`max` のみ）。新規 `#[ignore]` `backend_ops_sum_matches_cpu_bit_exact`（`BackendOps` 経由・非 contiguous view・0 サイズ契約・範囲外 dim・NaN・決定性） |
| `crates/backend-metal/src/typed_f16.rs` | `sum_max_inherit_f32_backend_ops_result_class` を `max_inherit_f32_backend_ops_unsupported_without_device_init`（`max` のみ・非 `#[ignore]`）と `sum_rejects_out_of_range_dim_before_touching_device`（非 `#[ignore]`）・`#[ignore]` `sum_matches_f32_backend_ops_rounded_bit_exact` へ分割 |
| `crates/backend-metal/src/typed_bf16.rs` | `sum_and_max_remain_unsupported_without_device_init` → `max_remains_unsupported_without_device_init`（`max` のみ）。新規 `#[ignore]` `sum_matches_f32_backend_ops_rounded_bit_exact` |
| `crates/backend-metal/tests/typed_ops_f16_parity.rs` | `sum_max_are_unsupported_matching_metal_f32_backend_ops` → `max_is_unsupported_matching_metal_f32_backend_ops`。新規 `#[ignore]` `sum_matches_f32_backend_ops_rounded_bit_exact` |
| `crates/backend-metal/tests/typed_ops_bf16_parity.rs` | `sum_and_max_remain_unsupported_without_device_init` → `max_remains_unsupported_without_device_init`。新規 `#[ignore]` `sum_matches_f32_backend_ops_rounded_bit_exact` |
| `crates/facade/tests/reduce_backend_parity.rs` | `#[cfg(target_os = "macos")]` `#[ignore]` 新規 3 テスト（`metal_sum_all_forward_and_backward_match_cpu_bit_exact`／`metal_sum_axis_forward_and_backward_match_cpu_bit_exact`／`metal_mean_forward_and_backward_match_cpu_bit_exact`。CPU tape との forward・backward bit 完全一致。`Op::Sum` の VJP は算術を伴わないホスト側のため勾配も bit 一致） |

`typed_f16`／`typed_bf16` の `sum` は本ファイル自体を変更せず（3 段
構成〈昇格 → `BackendOps::sum` へ委譲 → 丸め〉のまま）、委譲先の
`ops::MetalBackendOps::sum` 実装差し替えにより自動的に有効化された
（`TypedOps<f16|bf16>::sum` が成功結果を返すようになる）。

**Linux 検証結果**: `cargo fmt --all -- --check`・`cargo clippy
--workspace --all-targets --all-features -- -D warnings`・`cargo test
--workspace --all-features`・`make check-cross-metal-tests`（`cargo
check -p fandhe-ai-backend-metal --tests --target aarch64-apple-darwin`）・
`cargo check -p fandhe-ai --tests --target aarch64-apple-darwin`・
`RUSTDOCFLAGS="-D warnings" cargo doc -p fandhe-ai-backend-metal -p
fandhe-ai-backend-cpu --no-deps --locked --target aarch64-apple-darwin`
がいずれも green（本 PR 実装時点）。

**M4 Max 実機実測は未実施**。`docs/perf/logs/
metal-reduce-sum-wiring-1896/README.md`（事前登録判定規則・実行
コマンド・記入欄）へ申し送る。

**2026-09-16 M4 Max 実測済み（イシュー #1894）**: 上記 README の事前登録
判定規則どおり、非 `#[ignore]` 群（lib 567 pass／0 fail）・`reduce_parity`
3/3・新規 sum bit 一致テスト 6/6・#1902 §3.1 の 11 テスト 11/11 がすべて
pass。フル実行（`--all-features --no-fail-fast -- --ignored`）は 411 pass／
1 FAIL で、FAIL は `command_batching` の並列干渉による既知 FAIL 1 件のみ
（直列 3/3 pass）。#1902（398 pass）比の by-name 差は後退 0 件・新規 pass
14 件。§10 の `Var::sum` ホストフォールバックは同日時点では段階 0（未承認）
だったが、2026-09-17 に案 C（実装しない）でユーザー承認済み（§10 結論）。

## 10. `Var::sum` ホストフォールバックの設計判断（案 C 確定・2026-09-17 ユーザー承認済み）

`ops.rs::MetalBackendOps::sum` が `Unsupported` を返すのは（§9 の検査
順序どおり）カーネル `uint` 引数の上限超過（`numel`／`num_chunks`／
`lanes`／`axis_len`／`inner` が `u32::MAX` 超。f32 で概ね 16 GiB 超）の
場合のみに限定された。この残存 `Unsupported` に対し `Var::sum` が
ホストへフォールバックすべきかを検討する。

### 候補

- **案 A**: `eval::sum`（ホスト側 f32 逐次和の参照実装）へフォール
  バックする（`min_with_fallback`／`cumsum` と同型のパターン）
- **案 B**: フォールバック先を CPU 参照実装（`fandhe_ai_backend_cpu::
  reduction::sum`。f64 チャンク 2 段構成）と bit 一致する新規ホスト
  実装にする
- **案 C（推奨）**: フォールバックを実装しない（現状維持）

### 判断材料

1. `eval::sum` は素の f32 逐次和であり `fandhe_ai_backend_cpu::
   reduction::sum`（f64 チャンク 2 段構成。§2）と bit 一致しない。
   案 A を採ると、サイズ上限超過時のみ `Unsupported` が無言で異なる
   数値方式へすり替わる（silent fallback）ことになり、`.claude/
   rules/security.md` A08「判定の迂回経路を作らない」方針・
   `Var::mean`／`Op::Mean` 再計算・`var`／`std` 等 `sum` に依存する
   演算全体へこの非一貫性が波及する
2. `ops.rs::MetalBackendOps::sum` が `Unsupported` を返すのは実用上
   到達しないサイズ域（f32 で 16 GiB 超）に限定された（§9）ため、
   フォールバックの実利は小さい
3. CPU／CUDA は `sum` を実装済みであり、facade 公開契約
   （`docs/compat-api-scope.md`）上フォールバック必須の要件はない
4. 案 B は `autodiff` クレートが `backend-cpu` クレートへ依存しない
   設計（cfg ベースバックエンド切替。REQ-2）のため実装が重複する

### 結論（案 C 確定）

**案 C を採用**。サイズ上限超過は `AutodiffError::Backend(Unsupported)`
として呼び出し元へそのまま伝播することを受け入れ済み事項として明記
する。#1896 時点では段階 0（推奨のみ・未承認）として記録し実装しな
かったが、**2026-09-17 にユーザーが案 C を承認**（イシュー #1932・
親 #1930・ルート #1920）し確定した。案 A／B への転換は改めてユーザー
承認を要する別イシューとする。

承認に伴う反映（#1932。コード挙動変更なし）:

- `crates/autodiff/src/var.rs::Var::sum` の doc comment に「`BackendOps::
  sum` の `Unsupported` はホストへフォールバックせず伝播する（`cumsum`
  のフォールバック規律とは対になる）」旨を明記
- `crates/autodiff/tests/sum_no_host_fallback.rs`（Linux 実行可能）:
  `sum` が `Unsupported` を返すスタブ `BackendOps` 上で `Var::sum(None)`／
  `Var::sum(Some(dim))` が `Err(AutodiffError::Backend(BackendError::
  Unsupported(_)))` を返すことを機械的に固定（`cast.rs` のフィクス
  チャと同型。判定迂回経路を作らない `.claude/rules/security.md` A08）

## 11. argmax／argmin（イシュー #1951）

### 11.1 契約の核心

- **単一軸（`dim=Some(axis)`）**: `sum` と同じく CPU `axis_arg_reduce`
  （出力要素ごとに独立して縮約軸を昇順に逐次走査）を **逐語再現**する。
  1 スレッド = 1 lane で `best_idx=0`／`best` 未確定から開始し、NaN は
  スキップ、`v > best`（argmax）／`v < best`（argmin）の strict 比較の
  ときのみ更新する。
- **全要素（`dim=None`）**: CPU `arg_slice` は `sum_slice_f64`（`sum`）
  と異なり**平坦な逐次走査**（チャンク分割しない）。Metal は `sum` と
  同じ 2 段構成（チャンク内走査 → チャンク間走査）を採るため、逐語
  再現ではなく**等価性の証明**に依拠する: (1) チャンク内走査で
  「チャンク内最初に現れる極値」の (キー, グローバル添字) を求め、
  (2) チャンク番号昇順の走査で strict 置換のみ行う（同値は先行チャンク
  を保持）。平坦走査の結果は「極値を持つ最小添字」であり、(1) は
  チャンク内最小添字、(2) の strict 置換は同値時に先行チャンク（＝
  より小さい添字）を保持するため、2 段構成の結果は全域で平坦走査と
  一致する。`crate::reduce_model::argext_all_chunked_matches_flat_scan`
  が全域で機械的に裏付ける。
- **比較は整数ドメインで行う**: GPU 上の f32 非正規化数は flush され
  うるため、float の `<`／`>`・`isnan(` を直接使わず、ビットパターン
  から「NaN は除外・±0 は同値化」した単調 `uint` キー（`crate::
  reduce_model::arg_key`／`reduce.metal::red_arg_key`）へ変換してから
  整数比較する。`crate::sort_model::value_key`（NaN を最大キーへ写像）
  とは NaN の扱いが異なるため独立に定義する。

### 11.2 実装構成

- `crate::reduce_model`: `ArgExtKind`（Max／Min）・`arg_key`・
  `argext_axis_lane`・`argext_all_chunked`（ホスト逐語モデル・
  等価性検証テスト込み）・`plan_argext_all`／`plan_argext_axis`
  （既存 `plan_reduce_*` に `i32` 添字範囲検査を追加）。
- `shaders/reduce.metal`: `reduce_arg_all_chunk_f32`／
  `reduce_arg_all_finalize_f32`（全要素 2 段。中間値は `partial_key`／
  `partial_idx`〈`u32`〉2 本。無効チャンクは `partial_idx=0xFFFFFFFF`）・
  `reduce_arg_axis_f32`（単一軸 1 段）。`mode`（`constant uint&`。
  0=Max・1=Min）で切替。`red_f64_*` は使わない（加減算不要）。
- `crate::reduce::MetalReduce`: `run_arg_all_f32`／`run_arg_axis_f32`
  （新規パイプライン 3 種を保持）。
- `crate::ops::metal_argext`（`MetalBackendOps::argmax`／`argmin` 共通
  ヘルパ）: `reduce_out_shape` → 空縮約判定（CPU
  `reduce_error_to_backend_error` の `"argmax"`／`"argmin"` 分岐と同じ
  `BackendError::KernelLaunchFailed` 写像。単位元を持たないため）→
  `checked_numel` → `plan_argext_all`／`plan_argext_axis` 先出し
  （`i32` 範囲超過は `Unsupported` へ写像しホストフォールバックへ委ねる。
  `Var::argmax`／`argmin` は `sum` と異なりホスト参照実装
  〈`eval::argmax`／`argmin`〉フォールバックを持つ）→
  `context_cache::cached_reduce` 経由でカーネル起動。

### 11.3 サイズ上限・エラー契約

- カーネル引数は `uint`（既存 `plan_reduce_*` と同じ `u32::MAX` 上限）。
- 出力は `int`。CPU `build_index_tensor` の `IndexRangeOverflow` は
  **データ依存**（選ばれた添字が `i32::MAX` 超のときのみ）だが、
  Metal 側は形状だけから判定できる十分条件（全要素: `numel >
  i32::MAX`・単一軸: `axis_len > i32::MAX`）を `plan_argext_all`／
  `plan_argext_axis` で検査し、該当時は `Unsupported` へ写像して
  ホスト参照実装（同じデータ依存エラー契約を持つ）へ委ねる。
- 空縮約は CPU と同じくエラー（`sum` のようにゼロ埋めしない）。

### 11.4 テスト構成・実機実測

- Linux 実行可能: `reduce_model.rs` 単体テスト（CPU 参照実装との
  全域一致・チャンク境界タイ・NaN・±0・平坦走査との等価性・plan の
  `i32` 境界）・`reduce_source_evidence.rs`（MSL 文字列証跡: カーネル
  宣言・境界検査・`mode` 引数・`red_f64_*` 非参照・ビットキー比較）・
  `backend_ops_real_device.rs::argmax_argmin_shape_errors_without_
  device_init`（形状由来エラー経路がデバイス初期化を要しないこと）。
- macOS 実機 `#[ignore]`: `reduce_parity.rs`（起動 API 直叩き。全要素・
  単一軸・タイ・NaN・run-to-run 決定性）・`backend_ops_real_device.rs::
  backend_ops_argmax_argmin_match_cpu_exact`（`BackendOps` 経由。
  transpose view 込み）・`facade/tests/reduce_backend_parity.rs::
  metal_argmax_and_argmin_match_cpu_exact`。**M4 Max 実機実測は本
  実装セッション（Linux 環境。Apple Silicon 実機への到達手段なし）
  では未実施のまま Mac セッションへ申し送り**（`docs/perf/logs/
  metal-argext-1951/README.md` に実行手順・事前登録判定規則を記載）。
