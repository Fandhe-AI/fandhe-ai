# ブロッキングの実行時キャッシュ検出（sysctl）と 2 次元タイルジョブ分配（#753）

イシュー #753「perf(backend-cpu): ブロッキングの実行時キャッシュ検出（sysctl）と 2 次元
タイルジョブ分配」の実装記録。CPU GEMM（`gemm_blis_parallel`）は gemm crate（faer 実体）比
0.87〜0.95 の拮抗状態（2026-08-19 M4 Max 直接比較・`docs/perf/oss-gemm-comparison-baseline.md`）
にあり、gemm crate が持つ 2 つの上位構造（実行時キャッシュ検出・2 次元タイルジョブ分配）の
**技法を参照**（コードは転記しない）して導入することで差の縮小・逆転を狙う。

**本ドキュメントは REQ-8 の下限値・数値一致許容誤差を一切変更しない。**

## 状態: 実装済み・本番未結線（実機計測は未実施）

対象実機は Apple M4 Max（#481 §3 確定）。本セッション環境は Linux x86_64 で実機到達不能の
ため、受け入れ条件 2（実機 5 回中央値での非劣化・gemm crate との差の縮小または逆転）は本 PR
の範囲外。#750・#758 と同型の「実装は入れるが実機ゲート未通過のうちは本番結線しない」方式を
採る。本番 3 公開関数（`gemm_blis`／`gemm_blis_parallel`／`gemm_blis_bias_act_parallel`）の
既定挙動は本 PR では変更しない。

## §1 PR #766 の教訓とその反映

`docs/perf/cpu-gemm-blocking-sweep.md` §7 (ii) の通り、#749 で一旦実装した `sysctlbyname`
ベースの**機種判定**（`hw.model` 文字列一致による NC=9600 拡大の有効化判定）は、実測を行った
M4 Max 個体の正確な識別子が記録されておらず復元不能だったため、「識別子未記録のまま常に
`false` を返す判定機構を本番経路に残すこと自体が不要なリスク」という codex-review 指摘（P0/P1）
を受け PR #766 で撤去された。

本実装（#753）はこの教訓を踏まえ、以下の点で #749/#766 の構成と異なる:

| 観点 | #749（撤去済み） | #753（本実装） |
|------|------------------|----------------|
| 判定方式 | 機種識別子（`hw.model`）の文字列一致 | キャッシュサイズ（`hw.perflevel0.l1dcachesize`／`l2cachesize`）からの**算出式**（[`compute_blocks`](../../crates/backend-cpu/src/gemm_blis/cache_params.rs)） |
| 未知環境での挙動 | 常に `false`（発火しない） | 既知の環境依存の実測値をそのまま算出式へ通す（機種を選ばない） |
| 単体テスト可能性 | 機種判定ロジックは実機の `hw.model` 値に依存 | 純関数（`compute_blocks(l1d_bytes, l2_bytes, mr, nr)`）は任意のプラットフォームで合成値により単体テスト可能（`cache_params::tests`。5 件） |
| 本番未結線時の到達可能性 | `#[cfg(test)]` 化前は常時ビルドに含まれ「発火しない」状態だった | モジュール全体を `#[cfg(test)]` 化（下記 §3）し、`cargo test` の一部として非 `#[ignore]` テストが実行し続ける |

## §2 MC/KC/NC 算出式（`cache_params::compute_blocks`）

BLIS 解析モデル系の一般的なキャッシュ階層ブロッキング導出方針（gemm crate `cache.rs` の
「L1 連想度と A パネルの追い出し関係を整合させる」技法を**参照**のみ）に基づく:

- **KC**: A マイクロパネル（`MR × KC` 要素）と B マイクロパネル（`KC × NR` 要素）が L1D に
  共存し追い出し合わない条件から算出する。`sysctl` は連想度を報告しないため、保守的に
  「L1D 実容量の半分」を予算とする（残り半分は C アキュムレータタイル・連想度に由来する実効
  容量低下の余裕分）。
  ```
  kc = clamp((l1d_bytes / 2) / (4 * (mr + nr)), KC_MIN=128, KC_MAX=4096)
  ```
- **MC**: A パネル（`MC × KC × 4B`）が L2 実容量の一定割合（半分）に収まる条件から算出し、
  `MR` の倍数へ切り上げる。
  ```
  mc = clamp(round_up((l2_bytes / 2) / (4 * kc), mr), MC_MIN=64, MC_MAX=1024)
  ```
- **NC**: L2 残余（A パネル分を除いた残り半分）から B パネルが収まる上限を算出し、`NR` の
  倍数へ切り上げる。
  ```
  nc = clamp(round_up((l2_bytes / 2) / (4 * kc), nr), NC_MIN=256, NC_MAX=16384)
  ```

クランプ範囲の根拠: `KC_MAX`／`NC_MAX` は #749 実測（`cpu-gemm-blocking-sweep.md` §7）の
候補グリッドで検証済みの firestorm 参照値（KC=4096・NC=9600）を包含しつつ無制限の拡大を
防ぐ値。`NC_MAX=16384` は #749 実測（NC=9600 は n>=4096 でのみ改善・n=2048 では劣化）と
矛盾しないための上限。`l1d_bytes`／`l2_bytes` は外部入力（sysctl の戻り値）として扱い、
正当性検査範囲（L1D: 4KiB〜8MiB、L2: 128KiB〜256MiB）外・0・`mr`／`nr` が 0 の場合は
`None`（フォールバック。受け入れ条件 3。OWASP A03・`.claude/rules/security.md`）。

## §3 モジュール全体の `#[cfg(test)]` 化と unsafe FFI のコンパイル検証

`cache_params`・`partition` モジュールは、`gemm_blis_with_kernel_and_blocks`／
`gemm_blis_parallel_with_blocks`（#564）・`gemm_blis_shared_b_region`／`dispatch_shared_b`
（#750）と同じ「本番未結線の間はモジュール自体を `#[cfg(test)]` にする」既存パターンを
踏襲する（`crates/backend-cpu/src/gemm_blis/mod.rs` のモジュール宣言）。理由は、本番未結線
（受け入れ条件 2 未充足）の間は呼び出し元がテスト専用パラメータ化入口・実機 A/B ハーネスに
限られ、通常の `cargo build`（`cfg(test)` 無効）ではこれらの item が構造的に呼ばれない
ため、個別に `#[allow(dead_code)]` で黙らせる代わりにモジュール自体を条件付ける方が
`.claude/rules/coding-rust.md`（`#[allow]` の安易な追加で黙らせない）と整合する。

トレードオフ: この構成では CI の `cargo build (linux / aarch64-apple-darwin)` ジョブ
（`cargo build --workspace --locked --target aarch64-apple-darwin --lib`。`cfg(test)` 無効）
は `cache_params` モジュールをコンパイル対象に含めないため、sysctl FFI（`unsafe` を含む
唯一の箇所）の型・借用検査は同ジョブでは行われない。この検証は本セッションでローカルに
`cargo check -p backend-cpu --lib --tests --target aarch64-apple-darwin` を実行して確認
済み（`unsafe extern "C" fn sysctlbyname` 宣言・呼び出し双方が objc2 系依存を含むクロス
コンパイル環境で型・借用検査を通過することを確認。実行時の正当性〈実際に正しい値を返すか〉
は macOS 実機セッションでの検証が必要で、これは本 PR のスコープ外の「実機計測」に含まれる）。
CI で自動的に検証され続ける経路が必要な場合は、`aarch64-apple-darwin` 向けの `cargo check
--tests` ステップを追加する判断を別途行う（本 PR では追加しない。追加はスコープ外の
CI ワークフロー変更でありユーザー承認・`.claude/rules/ci.md` の運用注意〈ジョブ追加時の
ruleset required contexts 更新〉が関わるため）。

sysctl FFI 自体は `unsafe` を 1 箇所（`sysctlbyname` 呼び出し）に限定し、`// SAFETY:`
コメントで不変条件（NUL 終端 C 文字列・書き込み先の有効性・読み取り専用呼び出しの契約）を
明記する（`.claude/rules/coding-rust.md`「`unsafe` は FFI 境界等の必要最小限に留め、理由を
コメントで明記しレビュー必須」・PR #766 の P0 指摘「`unsafe` FFI に SAFETY コメントが必須」
の再発防止）。`libc` クレートは追加しない（許容 9 区分外。`sysctlbyname` は macOS 実行環境に
常にリンクされる libSystem の C ABI 関数のため、`objc2` 系〈`cfg(target_os = "macos")` 限定の
許容依存〉と同様に自前 `extern "C"` 宣言で足りる）。

## §4 2 次元タイルジョブ分配: unsafe を使わない設計判断

### 検討した案（不採用）: gemm crate 方式の完全な 2D 分配

gemm crate 本来の n_jobs 分配方式は、M×N のミニタイル格子を row-major でフラット化し、
worker 数へ均等分配したインデックス範囲を各 worker が処理する。この方式は列方向にも非連続な
タイル群を 1 worker が担当しうるため、C への書き込みが `&mut [f32]` の素朴な借用検査では
disjoint 性を表現できず、生ポインタ（`unsafe`）によるラッパーが必要になる。

**この案は #753 では採用しない。** 理由は PR #766 で「常に不活性な sysctl FFI」が P0/P1
指摘で撤去された経緯を踏まえ、`.claude/rules/coding-rust.md`「`unsafe` は FFI 境界等の
必要最小限に留める」方針との整合を優先したため（FFI 境界〈libSystem との ABI 境界〉は
`unsafe` の正当な用途だが、C の書き込み分配は Rust 内部のロジックであり FFI 境界ではない。
生ポインタによる disjoint 性の主張は実行時に不変条件が破れても borrow checker が検出できない
ため、レビューコスト・リスクが FFI 境界の `unsafe` より高い）。

### 採用した案: 行バンド外側・タイル数均等化による安全な分配

[`partition::row_ranges_for_workers`](../../crates/backend-cpu/src/gemm_blis/partition.rs)
は、`blocks.mc` 単位の行タイル**数**を [`partition::split_evenly`]（gemm crate `gemm.rs` の
n_jobs 分配方式を**参照**した均等割り。区間長の差は高々 1）で worker 数へ分配してから、
連続する行範囲へ変換する。

従来の `gemm_blis_parallel` の静的パネル分割（`panel_rows = m.div_ceil(num_threads)`）は
行**数**のみを均等化するため、MC タイル境界を考慮しない。MC タイル数が `num_threads` で
割り切れない形状では、パネル境界がタイル境界を跨ぐ worker とそうでない worker とでタイル
処理数に偏りが生じうる。`row_ranges_for_workers` はタイル**数**を先に均等化してから行範囲へ
変換することで、この偏りを ±1 タイルへ抑える（
[`gemm_blis_parallel_2d_with_blocks`](../../crates/backend-cpu/src/gemm_blis/mod.rs) が
この行範囲を `c.split_at_mut` の連鎖で切り出す。各 worker の担当範囲は `[0, m)` を隙間なく
連続分割したものであることが `row_ranges_for_workers` の契約〈`partition::tests` で被覆完全性・
disjoint 性・タイル数の偏り ±1 を検証〉のため、`unsafe` なしで disjoint 性がコンパイル時に
保証される）。

列方向の分配は行わない（各 worker は担当行範囲の全列を内部で処理する。既存の
`gemm_blis_ic_loop`／`gemm_blis_region` がそのまま対応する）。[`partition::tile_grid`] は
M×N の完全な 2 次元ミニタイル格子（「重複なし・被覆完全」を単体テストで検証済み）を提供する
純関数として残すが、実行時分配には使わない。これは #753 が指す「2 次元タイルジョブ分配」の
ジョブ空間の定義そのものを独立に固定・検証するための位置づけであり、実行時の安全な分配
（行方向のみ）とは役割が異なることを明示するために分離している。

### この判断の限界

- 列方向のタイル境界を跨ぐ worker 間の負荷分散は改善しない（元々 1 worker が全列を処理する
  設計のため対象外）。真に列方向も含めた偏り是正が必要と判明した場合は、`unsafe` 生ポインタ
  方式の再検討をユーザー承認事項として別途提起する
- MC タイル数が worker 数を下回る形状（例: `m` が小さく `mc` が大きい）では
  `row_ranges_for_workers` が返す区間数が `workers` を下回りうる（`split_evenly` が総タイル数
  以下の区間しか作らないため）。この場合は rayon の `into_par_iter` が単に少ない並列度で
  実行するのみで、正当性には影響しない（`partition::tests::split_evenly_workers_exceeding_total_yields_at_most_total_ranges`
  で固定）

## §5 実機計測手順（後続セッション向け）

1. `docs/real-hardware-verification-env.md` の手順で M4 Max 実機へ接続する
2. `cargo test -p backend-cpu --release -- --ignored runtime_cache_detect_and_2d_partition_ab_median_throughput`
   を実行し、`default`／`detected`／`2d-partition` の中央値を dim ∈ {512, 1024, 2048, 4096}
   で比較する（`crates/backend-cpu/src/gemm_blis/mod.rs` の同名テスト。5 回計測中央値・
   計測順インターリーブ）
3. gemm crate との比較は `scripts/bench/oss-gemm-compare/`（`docs/perf/oss-gemm-comparison-baseline.md`
   §1.1 手順）で追跡する
4. 受け入れ条件 2（全サイズ非劣化・gemm crate との差の縮小または逆転）を満たした場合、
   本番 3 公開関数への結線（`cache_params`／`partition` モジュールの `#[cfg(test)]` 解除・
   `gemm_blis_parallel` 等からの呼び出し追加）をユーザー承認を得たうえで別 PR で行う。結線
   時は `docs/oss-comparison-harness-decision.md` §（該当があれば）・本ドキュメント §3 の
   トレードオフ（aarch64-apple-darwin ビルドジョブでの FFI コンパイル検証追加要否）を
   再確認する

## §6 スコープ外（フォローアップ）

- 実機（M4 Max）での A/B 計測・非劣化確認と本番既定切替（実機セッション＋ユーザー承認後の
  後続 PR）
- NC=9600 形状依存分岐の再導入判断（`cpu-gemm-blocking-sweep.md` §7 (ii)。機種識別子判明が
  前提。本実装〈算出式方式〉とは独立の課題として残る）
- 共有 B 経路（#750）と 2 次元分配の統合最適化（`gemm_blis_parallel_2d_with_blocks` は
  共有 B 経路〈`gemm_blis_shared_b_region`〉を経由せず `dispatch_region` を worker ごとに
  独立呼び出しする。統合は別イシューで検討する）
- 列方向も含めた完全な 2 次元ジョブ分配（`unsafe` 採用の可否をユーザー承認事項として再提起
  する場合の検討。§4「この判断の限界」参照）
