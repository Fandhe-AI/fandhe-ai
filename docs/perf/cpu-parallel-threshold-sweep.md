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
  TASK-8.1 準拠）。ハーネス: 自動判定版スイープは `crates/backend-cpu/
  tests/elementwise_threshold_perf.rs`、同一サイズ逐次 vs 並列比較は
  `crates/backend-cpu/src/elementwise.rs` の `#[cfg(test)] mod
  bench_internal` 単体テスト（いずれも `#[ignore]`。強制版を公開面に
  出さないための配置。PR #1066 codex-review P1 対応）

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

**上記実測の限界（codex-review 指摘・後述の同一サイズ比較で補強）**:
上記の表は `add_slice`／`mul_slice`／`exp_slice`（`PARALLEL_THRESHOLD` に
よる自動判定を内蔵する公開関数）を異なる要素数でスイープしたものであり、
2^14→2^15 の遷移で観測される跳ね上がりには「要素数が単純に倍増した
影響」と「並列化オーバーヘッドの影響」が混在する。両者を切り分けるため、
下記「同一サイズでの逐次 vs 並列比較」の実測を追加した。

## 実測: 同一サイズでの逐次 vs 並列比較（codex-review 指摘）

上記スイープは要素数を変えながら自動判定版を計測するため、要素数増加の
影響と並列化オーバーヘッドの影響を切り分けられないという指摘
（PR #1066 codex-review・イシュー #1027）を受け、`elementwise` モジュールへ
`PARALLEL_THRESHOLD` 判定を経由しない逐次強制版・並列強制版
（`add_slice_force_serial`／`add_slice_force_parallel` 等。`gemm_blis`
〈直列専用入口〉／`gemm_blis_parallel`〈並列専用入口〉と同じ発想。
`crates/backend-cpu/src/elementwise.rs` の `#[cfg(test)] mod
bench_internal`。強制版を公開面に出さないため通常ビルドから消える
`#[cfg(test)]` の private モジュールに置く）を追加し、**同一要素数**で
両経路を計測する `elementwise_serial_vs_parallel_sweep`（同モジュール内の
単体テスト）を実行した:

| 要素数 | add 逐次 | add 並列 | add 比（並列/逐次） | mul 逐次 | mul 並列 | mul 比 | exp 逐次 | exp 並列 | exp 比 |
|---|---|---|---|---|---|---|---|---|---|
| 2^12=4,096 | 0.321µs | 20.665µs | 64.38 | 0.249µs | 21.330µs | 85.66 | 6.037µs | 21.946µs | 3.64 |
| 2^13=8,192 | 0.595µs | 25.421µs | 42.72 | 0.592µs | 22.884µs | 38.66 | 12.030µs | 24.053µs | 2.00 |
| 2^14=16,384 | 1.188µs | 23.251µs | 19.57 | 1.188µs | 25.636µs | 21.58 | 24.047µs | 27.877µs | 1.16 |
| **2^15=32,768（閾値）** | 2.308µs | 31.152µs | 13.50 | 2.761µs | 28.142µs | 10.19 | 49.944µs | 32.531µs | **0.65** |
| 2^16=65,536 | 4.725µs | 28.455µs | 6.02 | 4.710µs | 30.024µs | 6.38 | 96.081µs | 40.767µs | **0.42** |
| 2^17=131,072 | 10.610µs | 31.064µs | 2.93 | 10.671µs | 31.505µs | 2.95 | 198.232µs | 56.343µs | **0.28** |
| 2^18=262,144 | 49.386µs | 38.510µs | **0.78** | 36.781µs | 44.913µs | 1.22 | 386.595µs | 83.307µs | **0.22** |

**観察（要素数の影響を排除した並列化オーバーヘッド単体）**: `add`/`mul`
は本ハーネスの計測環境（QEMU x86_64・12 論理コア）では 2^12〜2^17 の
全域で並列強制版が逐次強制版より遅く（比 > 1）、`add` が 2^18 でようやく
比 0.78 まで改善する（`mul` は 2^18 でもなお比 1.22 で並列が不利）。
これは「タスク分割・スレッド同期オーバーヘッドが固定費として重く、
1 要素あたりの計算コスト（加減乗算 1 回）に対して現在の
`PARALLEL_THRESHOLD = 1 << 15` は本計測環境では小さすぎる」ことを
示す一方、`exp`（libm 経由でコスト大）は 2^15 で早くも比 0.65 と並列が
有利になり、`PARALLEL_THRESHOLD` 前後の遷移は演算コストに強く依存する
ことが定量的に裏付けられた。

**判断（変更なし）**: 上記の同一サイズ比較は当初のスイープ（異なる
要素数間の比較）が示唆した「2^15 到達直後に並列化オーバーヘッドが
支配的になる」という定性的な観察を、要素数増加の影響を排除した形で
裏付ける（`exp` は 2^15 で既に有利・`add`/`mul` は 2^15 でなお不利という
演算依存の傾向も一致する）。ただし本計測環境（QEMU x86_64）は REQ-8 の
正式対象実機（Apple M4 Max）と異なり、`add`/`mul` について本計測環境の
比較のみから `PARALLEL_THRESHOLD` を積極的に引き上げる根拠とはしない
（上記「判断」節と同じ理由）。M4 Max 実機での同一サイズ比較の再実測が
残課題である（下記「残課題」節に追記）。

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

- `PARALLEL_THRESHOLD`（elementwise 専用。reduction は上記のとおり
  直列フォールバック自体が未導入のため本閾値を使用していない）・
  `GEMM_THREADING_THRESHOLD`（GEMM）いずれも M4 Max 実機での境界再
  スイープが残課題（`docs/perf/cpu-gemm-small-shape-serial-fallback.md`
  と同じ事情）。reduction は閾値導入自体が残課題（上記「reduction
  への直列フォールバックは未導入」節参照）であり、上記 2 つの実機
  再スイープ対象には含まれない
- 上記「実測: 同一サイズでの逐次 vs 並列比較」も QEMU x86_64 実測であり、
  M4 Max 実機での同一サイズ比較（`elementwise_serial_vs_parallel_sweep`）
  の再実測が残課題（`PARALLEL_THRESHOLD` の境界再スイープと合わせて
  実施できる）
