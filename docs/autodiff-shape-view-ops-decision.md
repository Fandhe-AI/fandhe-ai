# 形状演算 6 種（`unbind`・`movedim`・`swapaxes`・`tensor_split`・`meshgrid`・`rot90`）の CPU 実装記録（イシュー #2639）

親: #2625（Phase 4）／ルート: #2499。`docs/autodiff-rearrange-ops-decision.md`（#2143）・`docs/autodiff-matrix-ops-decision.md`（#2144）の
「既存 `Op` の合成のみで構成する」方式を再適用した実装記録であり、**承認記録ではない**
（facade 公開形の承認は #2677 で依頼中。公開自体は承認後の #2678）。

## 0. 結論

- 6 種（実体は 7 関数: `tensor_split` は分割数形と境界添字列形を別関数にした）を内部クレート `fandhe_ai_autodiff::shape_view_ops` へ追加した。
  - 自由関数: `unbind`・`movedim`・`swapaxes`・`tensor_split`・`tensor_split_indices`・`meshgrid`・`rot90`、および型 `MeshgridIndexing { Ij, Xy }`。
  - **新規 `Op`・`BackendOps` メソッド・VJP はゼロ**。いずれも値のコピーか view だけの演算で、既存の `Var::transpose`／`permute`／`narrow`／
    `reshape`／`broadcast_to`／`Var::contiguous`（`pub(crate)`）と `rearrange_ops::flip` の合成で書ける。
  - `crates/tensor-core`・`crates/backend-*`・`tape.rs`・`grad.rs`・`error.rs`・`var.rs` は変更していない。
- facade 公開は行わない。`ShapeViewOpsHoldDoctestGuard`（`crates/facade/src/lib.rs`）と `crates/facade/tests/api_surface.rs` の否定ガードで機械固定した。
- 依存・`unsafe`・tolerance・baseline・`docs/spec/`・ガードレール閾値は変更していない。
- CUDA／Metal の専用カーネルは対象外。実機テストは `#[ignore]` のまま未実測（§10）。

## 1. 着手時の判定（事実のみ）

- 6 種は REQ-9 Tier の列挙に名前がない。
- 本実装はルート #2499 の Phase 4 方針（内部実装＋保留ガードまで先行し、facade 公開は承認後）に基づき、#2639 の受入条件
  （内部実装・決定記録・保留ガード）に限って行った。公開面・対象範囲の拡張について承認済みとは記録しない（承認依頼 #2677 は別途）。
- 直近の兄弟（#2631〜#2637）は `tensor-core` へ共有カーネルと `BackendOps::*` を足したが、本イシューは採らなかった。カーネルが不要で、
  公開済みクレート `fandhe-ai-tensor-core` の trait 面を無用に広げないため。

## 2. 実装方式・合成表・バックエンド到達性

| 関数 | 合成 | forward のバックエンド呼び出し | backward |
|---|---|---|---|
| `swapaxes` | `Var::transpose` | なし（view） | zero-copy の逆並べ替え |
| `movedim` | 順列を作り `Var::permute`（恒等順列はノードを積まず `*x`） | なし（view） | 同上 |
| `tensor_split`／`tensor_split_indices` | `Var::narrow` を出力本数ぶん | なし（view） | `Op::Narrow` の VJP が `concat_with_fallback` を呼ぶ |
| `unbind` | `narrow(dim, i, 1)` → 必要時 `Var::contiguous`（`Op::Contiguous` のホストコピー）→ `reshape` | `contiguous` 時のみホストコピー | 同上（`concat_with_fallback`） |
| `meshgrid` | 各入力を `reshape` → `broadcast_to`（stride 0 の view） | なし（view） | `reduce_to_shape` の縮約 |
| `rot90` | `rearrange_ops::flip`（`Var::index_select` → `Op::Gather`）と `transpose` | `gather` | `Op::Gather` の VJP が scatter 系の `*_with_fallback` を呼ぶ |

- `BackendOps::concat` は CPU／CUDA／Metal のどれも override していないため、`tensor_split`／`unbind` の backward は常にホスト参照実装へ
  フォールバックする。
- **`gather`／`scatter` は CPU・CUDA・Metal が実カーネルを持つ**。よって `rot90` を実機で動かすと GPU の gather／scatter が走り、
  フォールバックにはならない（`Unsupported` を返すのは既定実装のバックエンドだけ）。受入基準 2 の「`Unsupported` フォールバックで
  ホスト計算へ到達する」ことは、`gather`／`scatter`／`concat` を `Unsupported` にするモック `BackendOps` のテストで固定した（§6）。
- 命名規律: 素の `fn unbind`／`movedim`／`swapaxes`／`tensor_split`／`tensor_split_indices`／`meshgrid`／`rot90` は
  `autodiff/src/shape_view_ops.rs` の各 1 件のみ（workspace インベントリが固定。着手前の実測で workspace に同名の `fn` はなかった）。
  別名（`moveaxis`・`swapdims`）は作らない。
- `Var`／`Tape`／`Tensor` に inherent メソッドは足していない（足すと facade 公開面が広がる）。`MeshgridIndexing` は `autodiff` の
  クレートルートからも再エクスポートしない。
- 非 checkpoint・高階微分（`create_graph`）・f64／f16／bf16 自動微分経路での保証は対象外。挙動は合成先の `Op` に従う。

## 3. 数値契約

- forward はコピーのみで算術を含まず、3 バックエンド間で構造的に bit 完全一致する（NaN の payload・`±inf`・`-0.0` も保存。
  fixture で NaN payload `0x7FC00001`・`-0.0`・`±inf` を含む入力を bit 一致で確認した）。
- 勾配は各入力要素への寄与が 1 つだけ。`meshgrid` のみ broadcast 軸の `reduce_to_shape` による合算になる。勾配の比較は REQ-2 の
  統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）で行い、bit 一致は要求しない（`-0.0 + 0.0` の合算でビットが変わりうる）。
- FMA 契約・`f64` アキュムレータ契約（`.claude/rules/coding-rust.md`）の新たな対象はない。tolerance 定数は新設・変更していない。

## 4. 境界検査

- 外部入力（軸・`source`／`destination`・`sections`・境界添字・`k`・`dims`・入力リスト）は、tape へノードを積む前にすべて検査する。
  引数起因のエラーで孤児ノードを残さないことを `tape.len()` の不変で固定した。
  - 例外: `meshgrid` の非 contiguous 入力は、コピーが必要になった時点で確保上限を検査する。その検査が失敗した場合、先行入力で
    積んだノードは tape に残る（結果へは影響しない）。コピーの要否が `Var` から事前に分からないための割り切り。
- 加算・乗算は `checked_*`、`k` は `rem_euclid`（`isize::MIN` でも panic しない）。`meshgrid` の全体 shape の要素数積は `checked_mul`
  （`ElementCountOverflow`）。
- **出力本数の確保前検査**: `tensor_split` の `sections` は外部入力で、テンソルの大きさと無関係に巨大になりうる。`unbind` の軸長も
  stride 0 の broadcast view では実体なしに巨大になる。`Vec::with_capacity` の capacity overflow panic と tape ノードの過大確保 abort を
  防ぐため、確保前に「出力本数 ×（`Var` 本体 + 1 本あたり最大ノード数 × `TapeNode` 本体）」を 4 バイト単位へ換算し、
  `rearrange_ops::checked_index_alloc_len`（1 GiB 上限）へ渡す。新しい定数は作らない。超過は `ShapeError::ElementCountOverflow`。
  出力 `Vec` は `try_reserve_exact` で確保する。この見積もりは簿記コストの下限で、ノードが持つヒープ分は数えない。
- `rot90` は `flip` の内部検査（`checked_axis_len_as_i32`・`checked_index_alloc_len`）を、反転する全軸について `flip` を呼ぶ前に
  先行実行する（2 軸反転の途中で失敗して孤児ノードを残さないため）。
- 本番経路で `unwrap`／`expect`／無検査の `as` を使わない。`unsafe` なし。

## 5. PyTorch 2.14.0 との差分・実測で確定した点

fixture は実 PyTorch 2.14.0+cpu の実行値（`crates/autodiff/tests/fixtures/shape-view-pytorch-reference/`。f32 は u32 ビットパターンで保存）。
186 ケース（`unbind`・`movedim`・`swapaxes`・`tensor_split`・`tensor_split_indices`・`meshgrid`・`rot90`、非 contiguous 入力・軸長 0・
非有限値を含む）で forward の bit 一致と勾配の REQ-2 判定一致を確認した。`error_cases`（19 件）は torch が例外を出すかを記録した。

計画時の仮説は実測で次のとおり確認できた（修正は不要だった）:

| 項目 | 実測 |
|---|---|
| `tensor_split_indices` の境界の扱い | `start = min(直前の境界, n)`・`end = min(境界, n)`・`end < start` は長さ 0。`[5,3]`（n=10）は `[5,0,7]`、`[100,2]` は `[10,0,8]`、`[7,3,9]` は `[7,0,6,1]`、`[12]` は `[10,0]`。非単調・範囲外はエラーにならない |
| `unbind` の軸長 0 | 空のタプル（本実装は空の `Vec`） |
| `meshgrid` の 0 次元入力 | 長さ 1 として扱う。入力 1 本の 0 次元は shape `[1]` を返す |
| `meshgrid(indexing="xy")` | 入力 2 本以上で先頭 2 軸だけ入れ替える（`N=1` は入れ替えない） |
| 例外になるケース | rank 0 の `unbind`・軸範囲外・`movedim` の長さ不一致／重複／範囲外・`tensor_split` の `sections = 0`／rank 0／軸範囲外・rank 2 の `meshgrid` 入力・空リストの `meshgrid`・`rot90` の `dims` 同値／範囲外／rank 1 |

仮説から実装を直した点（実測に合わせた）: `meshgrid` の入力 1 本・0 次元を `*x` のまま返す近道を設けていたが、PyTorch が shape `[1]` を
返すため近道を外した。

差分（意図的）:

| 項目 | PyTorch 2.14.0 | 本実装 | 扱い |
|---|---|---|---|
| 軸・境界添字 | 負の値可 | `usize` のみ | 型として表現できず非対応（`Var::squeeze` 等と同じ規約） |
| `tensor_split` の引数形 | 分割数・境界添字列・テンソル | `tensor_split`（分割数）と `tensor_split_indices`（境界添字列）の 2 関数。テンソル引数形は非対応 | 差分 |
| `unbind` の戻り | view | 先頭軸以外のスライスはコピー（`Var::reshape` の contiguous 制約のため `Op::Contiguous` を経由）。値は bit 一致 | 差分。zero-copy 専用 `Op` はスコープ外 |
| `rot90` の `k mod 4 == 0` | clone | 入力そのもの（ノードを積まない） | 差分 |
| `swapaxes` の rank 0 入力 | 受理（`a = b = 0`） | `AxisOutOfRange` | 差分（`Var::transpose` の検査に従う）。テストの `INTENDED_DIFFS` と一対一 |
| 巨大な `sections`・軸長 | 受理（巨大なタプルを作る） | 確保前に `ElementCountOverflow` | 差分（§4） |

tolerance・baseline は変更していない。

## 6. テスト構成

- `crates/autodiff/src/shape_view_ops.rs`（単体 15 件）: 各関数の既知値・恒等時にノードを積まないこと・非 contiguous 入力・軸長 0・型付きエラー各種と
  エラー時に `tape.len()` が変わらないこと・巨大 `sections`／巨大 broadcast view の確保前拒否・`meshgrid` の 0 次元／非 contiguous 入力・
  `rot90` の 2 軸反転の事前検査・`k` の剰余（`isize::MIN` を含む）。
- `crates/autodiff/tests/shape_view_parity.rs`: fixture 突合（forward bit 一致・勾配 REQ-2）、`error_cases`（意図的な差分は
  `INTENDED_DIFFS`）、中心差分（損失は入力の線形関数のため刻み 0.5 でも厳密。7 関数）、モック `BackendOps`（`gather`／`scatter`／`concat`
  を差し替え）による `Unsupported` フォールバック（呼び出し回数 1 以上）・他エラー（`KernelLaunchFailed`）の伝播・view 演算の forward／backward で
  呼び出し 0 回、テープ記録数、run-to-run の bit 決定性、cross-tape の `meshgrid` 拒否。
- `crates/facade/tests/shape_view_ops_backend_parity.rs`: CPU tape と NaiveOps tape の突合（forward bit 一致・backward REQ-2）と手計算の期待値。
  CUDA／Metal 実機は `#[ignore]`。

## 7. facade 公開形の推奨案（未承認）

推奨は 1 つ。`Var` の inherent メソッドとして `shape_view_ops` への 1 行委譲で公開する。

- `Var::unbind(&self, dim: usize)`・`Var::tensor_split(&self, sections: usize, dim: usize)`・
  `Var::tensor_split_indices(&self, indices: &[usize], dim: usize)`（`-> Result<Vec<Var<'t>>, AutodiffError>`）
- `Var::movedim(&self, source: &[usize], destination: &[usize])`・`Var::swapaxes(&self, axis0: usize, axis1: usize)`・
  `Var::rot90(&self, k: isize, dims: [usize; 2])`（`-> Result<Var<'t>, AutodiffError>`）
- `Var::meshgrid(tensors: &[Var<'t>], indexing: MeshgridIndexing)`（`-> Result<Vec<Var<'t>>, AutodiffError>`。`Var::cat`／`stack` と同じ関連関数）
- 引数型 `MeshgridIndexing` だけをクレートルートから再エクスポートする（`FftNorm` と同じ扱い）。`shape_view_ops` モジュールは再エクスポートしない。
  `Sequential::add_*` は設けない。
- inherent メソッド 7 件と型 1 件の追加のみで非破壊。承認依頼は #2677、公開は承認後の #2678。
- 承認後は `ShapeViewOpsHoldDoctestGuard` と否定ガードを承認形の正ガード（委譲本体の固定を含む）へ反転する。

承認事項（**すべて未承認**）: 上記 7 メソッドの公開と `MeshgridIndexing` の再エクスポート、およびメソッド名・引数形
（`tensor_split` の 2 関数分割・軸と添字は非負の `usize`・`rot90` の `k` は `isize`・`meshgrid` は関連関数）。

## 8. スコープ外

- facade 公開（#2677 の承認後に #2678）。`Var` への inherent メソッド追加と `MeshgridIndexing` の再エクスポートを含む。
- GPU 専用カーネルと実機計測（§10）。
- zero-copy の `unbind`（専用 `Op::Select` や、非 contiguous の軸除去）。
- 負の軸と負の添字。
- `tensor_split` のテンソル引数形。
- 別名（`moveaxis`・`swapdims`・`dsplit`／`hsplit`／`vsplit`）。
- `rot90` の `k = 0` でのコピー。
- `create_graph`・activation checkpoint・f64／低精度の自動微分経路での保証。
- `docs/compat-api-scope.md` 1 節の対象範囲表・`docs/compat-feature-gap.md` の判定変更・spec（REQ-9）の改定。
- `MIN_KNOWN_PROBE_BLOCKS` の更新（下限値のため据え置き。#2637 も据え置いた）。

## 9. 多層防御（保留ガード）

| ガード | 内容 |
|---|---|
| `ShapeViewOpsHoldDoctestGuard`（`crates/facade/src/lib.rs`） | 全 `pub mod` glob import 下で、同名の関数・メソッド・型・モジュールが `Var`／`Tape`／`Tensor<f32>`／facade ルートに公開されるとコンパイルが失敗する正のプローブ |
| `shape_view_ops_hold_doctest_globs_all_pub_modules` | doctest の glob 集合と `pub mod` 宣言集合の一致 |
| `shape_view_ops_hold_doctest_probe_body_matches_fixed_contract` | プローブ本文の固定文言一致 |
| `facade_does_not_reexport_or_declare_shape_view_ops`（＋自己テスト） | facade src の再エクスポート・`MeshgridIndexing` の独自宣言・`pub mod shape_view_ops`・7 名の `fn` 宣言の否定検査 |
| `workspace_declares_shape_view_ops_fn_names_only_in_allowed_locations` | workspace 全体で 7 名の `fn` 宣言が `autodiff/src/shape_view_ops.rs` の各 1 件だけであること |

stable rustdoc は `compile_fail` のコードを照合しないため、否定ガードは正のプローブ＋インベントリで組んでいる。**有効性の確認**:
一時的に facade へ `pub use fandhe_ai_autodiff::shape_view_ops::MeshgridIndexing;` を足すと doctest（E0659）と
`facade_does_not_reexport_or_declare_shape_view_ops` が落ち、`Var` へ `pub fn unbind(&self, _: usize)` を足すと doctest（シグネチャ不一致）と
`workspace_declares_shape_view_ops_fn_names_only_in_allowed_locations` が落ちることを確認したうえで、いずれも元に戻した。

## 10. 実機申し送り

CUDA（DGX Spark GB10）・Metal（Apple Silicon）の実機テスト（`cuda_shape_view_ops_match_cpu_reference`・
`metal_shape_view_ops_match_cpu_reference`）は `#[ignore]` のまま未実測。手順は `docs/perf/logs/shape-view-ops-2639/README.md`。

## 11. 出典

- `docs/autodiff-rearrange-ops-decision.md`（#2143）・`docs/autodiff-matrix-ops-decision.md`（#2144）・
  `docs/autodiff-stat-reduce-ops-decision.md`（#2637。保留ガード一式の雛形）
- `docs/compat-api-scope.md` 5 節（適用記録）・`.claude/rules/coding-rust.md`（REQ-2 判定・カーネル境界検査）
- PyTorch 2.14.0 実行値: `crates/autodiff/tests/fixtures/shape-view-pytorch-reference/README.md`
