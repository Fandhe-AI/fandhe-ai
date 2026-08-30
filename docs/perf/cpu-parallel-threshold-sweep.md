# CPU elementwise・reduction の rayon 並列化閾値スイープ実測（イシュー #1027）

## 背景

親イシュー #1008（perf(phase-1) 学習・推論ループの固定費の診断と除去）
配下のイシュー #1027。GEMM（`docs/perf/cpu-gemm-small-shape-serial-fallback.md`）
と同種の rayon 直列フォールバックは、elementwise（`crates/backend-cpu/src/
elementwise.rs`）には既に `PARALLEL_THRESHOLD = 1 << 15`（要素数 32,768）
として実装済みだったが、閾値そのものの妥当性を裏付ける実測記録が
docs に無かった。reduction（`crates/backend-cpu/src/reduction.rs`）には
直列フォールバックが未実装だった。本 doc は両者の実測・実装記録である。

## 計測環境（重要な注記）

- ローカル QEMU x86_64 12 論理コア（AVX2+FMA あり）
- **REQ-8 の正式対象実機は Apple M4 Max**（`docs/perf/
  gemm-optimization-baseline.md` §3・イシュー #481 で確定）であり、
  本 doc の実測環境とは異なる。閾値の最終確定には M4 Max 実機での
  再スイープが残課題（GEMM 側と同じ事情。`cpu-gemm-small-shape-
  serial-fallback.md` 計測環境節参照）
- 計測プロトコル: `bench_harness::run`（warmup 20・iters 20・中央値。
  TASK-8.1 準拠）。ハーネス: `crates/backend-cpu/tests/
  elementwise_threshold_perf.rs`（`#[ignore]`）

## 実測: elementwise（`add`／`mul`／`exp`）閾値スイープ

`PARALLEL_THRESHOLD = 1 << 15 = 32,768` を挟む要素数 2^12〜2^18 で
`add_slice`／`mul_slice`／`exp_slice`（libm 経由の代表）を計測した:

| 要素数 | add（中央値） | mul（中央値） | exp（中央値） |
|---|---|---|---|
| 2^12=4,096 | 0.459µs | 0.734µs | 10.833µs |
| 2^13=8,192 | 1.679µs | 1.168µs | 21.794µs |
| 2^14=16,384 | 2.203µs | 2.227µs | 24.062µs |
| **2^15=32,768（閾値。並列化発動）** | **24.794µs** | **25.793µs** | 34.751µs |
| 2^16=65,536 | 27.554µs | 26.070µs | 42.423µs |
| 2^17=131,072 | 29.834µs | 28.961µs | 50.937µs |
| 2^18=262,144 | 35.789µs | 34.624µs | 77.173µs |

**観察**: `add`／`mul`（libm 非経由・加減乗算のみ）は 2^14→2^15 の遷移
（閾値到達＝並列化発動）で中央値が約 11〜12 倍に跳ね上がる（2.2µs →
24.8/25.8µs）。これは 1 要素あたりの計算コストが極小（加減乗算 1 回）
なため、rayon のタスク分割・スレッド同期オーバーヘッド自体が支配的に
なり、`PARALLEL_THRESHOLD` を跨いだ直後は並列化が明確に不利になる
ことを示す。`exp`（libm 経由で 1 要素あたりのコストが `add`/`mul` の
約 15〜20 倍）は同じ遷移での跳ね上がりが相対的に緩やか（24.1µs →
34.8µs。約 1.4 倍）で、1 要素あたりコストが高い演算ほど並列化
オーバーヘッドの相対的な影響が小さいという直感と整合する。

**判断: `PARALLEL_THRESHOLD = 1 << 15` は変更しない**。上記実測は
遷移点そのもの（2^15 到達直後に跳ね上がる）を捉えており、閾値を
現在値より大きくする方向（例: 2^16 や 2^17 へ引き上げ）であれば
`add`/`mul` の遷移直後の劣化区間を縮小できる可能性はあるが、
`PARALLEL_THRESHOLD` は softmax／rmsnorm／`fused_elementwise` にも
共有される値であり、変更の影響範囲は elementwise 単体に留まらない。
本イシューの受け入れ条件は「閾値スイープの実測記録」であり値の変更
そのものではないこと、および M4 Max 実機での再スイープが残課題である
こと（上記「計測環境」節）を踏まえ、現時点では実測記録に留め値は
保守的に維持する。

## reduction への直列フォールバックは未導入

`crates/backend-cpu/src/reduction.rs` の `sum_slice`／`max_slice`／
`axis_reduce` へ `crate::elementwise::PARALLEL_THRESHOLD` を再利用した
直列フォールバックを追加する実装を一度試みたが、`gemm_blis/mod.rs` の
`should_serialize`（イシュー #811・#1027）と同じ「reduction 専用の
実機直列/並列比較を実施していないうちは攻めた値を本番結線しない」
方針（PR #758 前例）に倣い、本番未結線へ差し戻した（`crates/
backend-cpu/src/reduction.rs` モジュール doc「小サイズ直列フォール
バック（未導入）」節参照）。`elementwise::PARALLEL_THRESHOLD` は
要素ごと独立・アキュムレータなしの契約に対する値であり、reduction
（累積を伴う別契約）へそのまま転用してよい根拠が実測として無いため。

- **現状**: `sum_slice`／`max_slice`（全縮約 `dim=None`）は常に
  `par_chunks` によるチャンク並列、`axis_reduce`（軸指定
  `dim=Some(axis)`）は常に出力要素側の `into_par_iter()` を経由する
  （サイズによる分岐なし）
- **テスト**: `parallel_threshold_boundary_deterministic_full_reduction`
  （`sum`/`max` の全縮約。`PARALLEL_THRESHOLD` 直下・直上という代表
  サイズでシングル／4 スレッドプール間の to_bits() 完全一致、および
  `chunk_boundary_deterministic_sum` と同じ naive 実装との bit 一致を
  確認。reduction 自体は閾値で分岐しないため、これは常時並列経路の
  決定性を確認するテストである）・
  `parallel_threshold_boundary_deterministic_axis_reduction`（軸指定
  reduction の同種確認。`outer=4` 固定で `axis_len` を調整し
  `outer*inner*axis_len` が代表サイズ境界を跨ぐ shape を構成）を
  `crates/backend-cpu/src/reduction.rs` に追加した
- reduction 専用の M4 Max 実機直列/並列比較を実施し閾値を確定・
  ユーザー承認を得られれば、直列フォールバックの導入を再検討する
  余地がある（上記モジュール doc 参照）

## 受け入れ条件との突合

- (a) 閾値スイープの実測記録: 本 doc「実測: elementwise 閾値スイープ」
- (b) 小形状ベンチで後退なし・大形状で非後退: reduction は直列
  フォールバック未導入（常に並列経路）のため後退の余地がない。
  追加した決定性テスト（`chunk_boundary_deterministic_sum`／
  `chunk_boundary_deterministic_max_and_mean` に加え上記 2 テスト）が
  全通過することで並列経路の bit 完全一致を確認済み。elementwise は
  実装済みのため変更なし

## 残課題（PR 本文にも記載）

- `PARALLEL_THRESHOLD`（elementwise・reduction 共有）・
  `GEMM_THREADING_THRESHOLD`（GEMM）いずれも M4 Max 実機での境界再
  スイープが残課題（`docs/perf/cpu-gemm-small-shape-serial-fallback.md`
  と同じ事情）
