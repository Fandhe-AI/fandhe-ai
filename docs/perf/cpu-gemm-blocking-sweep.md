# MC/KC/NC 実機スイープによる再選定（#564）

イシュー #564「MC/KC/NC を参照実装値の近傍で実機スイープして再選定」の実装記録。
`crates/backend-cpu/src/gemm_blis/mod.rs` のキャッシュブロッキング定数（MC=128／KC=256／
NC=512）は PoC-v2-1 実測環境で選定した起点値のままで、Grace（DGX Spark GB10）／M4 Max
実機での再チューニングは未実施だった。本イシューは BLIS/OpenBLAS の aarch64 向け参照実装値
近傍を含む候補グリッドを実機（Apple M4 Max。§1 参照）で 5 回計測中央値スイープし、有意な
改善があれば MC/KC/NC を再選定する。

**本ドキュメントは REQ-8 の下限値・数値一致許容誤差を一切変更しない**。

## 状態: 実測・選定完了、本番適用は識別子未記録のため未導入（2026-08-19 M4 Max 実測。詳細は §7）

**注記（PR #766・codex-review 再指摘。3 巡目で機種判定機構自体を撤去）**: #749 では一旦
`select_blocks`／`select_blocks_for`（判定ロジック）・`machine_detect`（`sysctlbyname` による
実行時機種判定）を実装したが、実測を行った M4 Max 個体の正確な `hw.model` 識別子が実測
セッション終了時点で記録されておらず、`Mac16,` prefix 等の広い一致条件へ後退させて有効化する
ことも本リポジトリの「実測値の捏造・placeholder 値での完了扱いは行わない」方針に反するため
PR #766 で採用できないと判断した。**識別子未記録のまま常に `false` を返す判定機構を本番経路に
残すこと自体が「発火しない unsafe FFI 呼び出しを常時ビルドに含める」不要なリスクであり
（codex-review P0/P1 指摘）、機構を有効化する見込みが立つまで導入を見送る方が安全側**と判断し、
PR #766 で `select_blocks`／`select_blocks_for`／`machine_detect`（`sysctlbyname` FFI 宣言・
`VERIFIED_M4_MAX_HW_MODEL` を含む）と対応する単体テストを撤去した。以下 §2〜§7 の実測記録・
選定根拠自体は有効（NC=9600・n>=4096 で約 9.9% 改善という実測結果は変わらない）。本番の
`gemm_blis`／`gemm_blis_parallel`／`gemm_blis_bias_act_parallel` は常に `default_blocks()`
（MC=128／KC=256／NC=512）を使う。実機の正確な `hw.model` 識別子が判明し機種判定を安全に
実装できるようになった段階で、本ドキュメントの実測値を根拠に再導入を検討する
（follow-up: #753〈sysctl ベース MC/KC/NC 動的算出〉）。
### 経緯（実装セッション時点の環境ゲート判定）

実測・選定は aarch64 実機（Apple M4 Max）でのみ有効という前提のもと、実装セッション開始時に
以下を判定した（#559・#552・#488 と同一のゲート判定手順）:

1. `uname -sm` → `Linux x86_64`（本開発環境。実測）。aarch64 実機ではないため NEON 経路での
   実性能計測は不可能。
2. `docs/real-hardware-verification-env.local.md`（Git 管理外のローカル用ファイル）→ 不存在
   （実測）。M4 Max・DGX Spark GB10 実機への接続情報が定義されていないため到達不可。
3. `qemu-aarch64` の存在確認（`which`）→ 不在（実測）。エミュレーション実行も不可能
   （なお QEMU では性能計測自体が元々無意味）。

**結論（実装セッション時点）**: 当該セッションでは M4 Max 実機に到達できないため、スイープ基盤の実装・パラメータ化・
テスト整備（`cargo test`／クロス `cargo check`／`cargo clippy` の green 確認）までを実施し、
**実測値の捏造・placeholder 値での完了扱いは行わない**（fail-closed）。実測・選定は M4 Max
実機へアクセス可能な後続セッション・Agent（`bench-runner` 委譲想定）が引き継いで実施する
方針とした。**2026-08-19 に後続セッションで実機スイープが実施され、下記「§7 実機実測結果」
に記録済み**（本節の記述は着手時点の判断としてそのまま残す）。

**追記（#749）**: 上記ゲート判定は本イシュー（#564）の実装セッション時点のものとして記録を
残す。実測自体は 2026-08-19 に別セッションで M4 Max へ到達して実施済みで、結果・採用判断は
§7 を参照。#749 の実装セッション（本改訂を行ったセッション）も同じ環境ゲート（`Linux x86_64`・
`docs/real-hardware-verification-env.local.md` 不在・`qemu-aarch64` 不在）で aarch64 実機に
到達できないため、§7 の記述は 2026-08-19 実測の引用であり本セッションでの再計測ではない
（実測値の捏造・placeholder 化はしない）。

## §1 対象実機・依存イシューの確定事項

- **対象実機**: **Apple M4 Max**（firestorm 系。`docs/perf/gemm-optimization-baseline.md` §3・
  イシュー #481 で確定。Grace（DGX Spark GB10）は参考系列）
- **NEON マイクロカーネル**: 既定 MR=8／NR=12（#559・closed）。スイープ候補の MC は 8 の倍数
  を基本とする（NC の NR=12 非整数倍は端タイル処理で機能上問題ない。`gemm_blis_region` の
  `nr_eff` 端数処理が任意の NC で成立することは §3.3 のパリティテストで検証済み）
- **記録様式・fail-closed 前例**: #488（A-8）`docs/perf/cpu-gemm-baseline-remeasurement.md`

## §2 参照実装値（自分の言葉での要約）

BLIS の aarch64 向けデフォルト config（`config/firestorm/bli_family_firestorm.h` 相当。Apple
M1 系高性能コア向け）と OpenBLAS の aarch64 系ターゲットの代表値を要約する（本リポでの実測
ではなく公開ソースの記述に基づく参考値。実機スイープの候補選定根拠として引用する）:

| 実装 | MC | KC | NC | 備考 |
|---|---|---|---|---|
| BLIS firestorm | 480 | 4096 | 9600 | Apple M1 高性能コア向け config。L1D/L2 サイズ・マイクロカーネル形状（MR=8, NR=12）に基づき選定されたパラメータ |
| OpenBLAS（aarch64 全ターゲット共通） | ターゲット依存 | ターゲット依存 | 4096 相当 | OpenBLAS は NC 相当のブロックサイズをターゲット横断で 4096 に統一する傾向がある（本リポ現行 NC=512 の 8 倍） |
| 本リポ現行値（PoC-v2-1） | 128 | 256 | 512 | QEMU Virtual CPU 環境での起点選定（#24。`docs/perf/cpu-gemm-rayon-tuning.md`） |

現行 NC=512 は firestorm 参照値（9600）の 1/18.75、OpenBLAS 相当（4096）の 1/8 と極端に小さい。
M=N=K=4096 では B パネル（NC×KC×4B 換算）の再パッキング回数が参照実装比で最大 8 倍になる
（NC が小さいほど `jc` ループの反復回数が増え、同じ B ブロックへの再アクセスが増える）。

## §3 実装内容（整備済み・コンパイル検証済み）

### §3.1 パラメータ化

`crates/backend-cpu/src/gemm_blis/mod.rs` を、既存の [`crate::gemm::BlockSizes`]（`gemm_blocked`
向けに #24 で導入済みの mc/kc/nc 3 つ組構造体）を再利用してパラメータ化した:

- `panel_capacity`／`PanelBuffers::new`／`gemm_blis_region`／`dispatch_region`（x86_64／
  aarch64／その他 arch の 3 変種）が `BlockSizes` 引数を受け取るようになった
- 公開 3 関数（`gemm_blis`／`gemm_blis_parallel`／`gemm_blis_bias_act_parallel`）のシグネチャは
  **不変**（公開 API 非破壊。既定値 `default_blocks()`＝旧 `MC`/`KC`/`NC` 定数と同値を内部で渡す）
- テスト・スイープ専用の入口を新設: `gemm_blis_with_kernel_and_blocks`（単一スレッド）・
  `gemm_blis_parallel_with_blocks`（行パネル並列。実運用経路 `gemm_blis_parallel` 相当）。
  いずれも `#[cfg(test)]` で駆動経路（本番 3 公開関数）からは到達不能

### §3.2 bit 完全一致契約（REQ-2）が維持される根拠

C タイルは pc（K ブロック）をまたいで現在値をロードして FMA 連鎖を継続するため、C 各要素の
累積順序は KC の値に依らず常に p 昇順・レーン間縮約なし。MC/NC は累積順序に一切影響しない。
したがって任意の MC/KC/NC でも `gemm_naive` との bit 完全一致は理論上維持される。

`gemm_blis_non_default_block_sizes_match_naive_bit_exact`（単一スレッド経路）・
`gemm_blis_parallel_non_default_block_sizes_match_naive_bit_exact`（並列経路）で、現行値・境界
跨ぎの小さい奇数系・firestorm 参照値そのもの（MC=480/KC=4096/NC=9600）を含む複数組を
`ScalarKernel`（単一スレッド版）／実行時検出 ISA（並列版）で `gemm_naive` と直接比較し、
x86_64 でも実行可能な形で検証済み（**許容誤差は一切変更していない**）。

### §3.3 パラメータ検証（fail-closed）

`validate_block_sizes`（`gemm_blis_with_kernel_and_blocks`／`gemm_blis_parallel_with_blocks` の
入口）が mc/kc/nc のいずれかが 0 の場合を `GemmError::ZeroBlockSize`（`crate::gemm` 側で既に
定義済み。`step_by(0)` パニック防止。Cursor Bugbot #231 と同種のバグの gemm_blis 版再発防止）
で早期拒否する。`gemm_blis_with_kernel_and_blocks_rejects_zero_block_size` テストで検証済み。

panel 容量計算（`panel_capacity`）の乗算オーバーフローについては、検討の結果**追加の検査を
設けなかった**（実装上到達不能と判断）。理由: `n`／`k_dim`／`mc_total` は呼び出し元
`validate_dims` を先に通過済み（`m*k`／`k*n`／`m*n` が `usize` に収まると検証済み）であり、
`panel_capacity` は常に `blocks.{mc,kc,nc}.min(dim)` でクランプしてから乗算する構造のため、
`blocks` 側にどれだけ大きな値（firestorm 参照値 KC=4096/NC=9600 等）を渡しても実際の乗算対象
は非オーバーフロー確定済みの dim 由来値に収まる。詳細は `crates/backend-cpu/src/gemm_blis/
mod.rs` の `validate_block_sizes` ドキュメントコメント参照。

### §3.4 候補グリッド（8 点・現行値と参照値近傍を含む）

M4 Max（firestorm 系）確定に基づき、firestorm 参照値を主候補・軸別分離で寄与を切り分ける
8 点を初期グリッドとした（MC は MR=8 の倍数）:

| # | MC | KC | NC | 意図 |
|---|---|---|---|---|
| 1 | 128 | 256 | 512 | 現行値（基準） |
| 2 | 128 | 256 | 4096 | NC のみ拡大（OpenBLAS 全ターゲット相当） |
| 3 | 128 | 256 | 9600 | NC のみ firestorm 値 |
| 4 | 128 | 4096 | 512 | KC のみ firestorm 値 |
| 5 | 480 | 256 | 512 | MC のみ firestorm 値 |
| 6 | 480 | 4096 | 9600 | BLIS firestorm 参照値そのまま |
| 7 | 256 | 1024 | 4096 | 中間点（現行と firestorm の対数中間近傍） |
| 8 | 480 | 4096 | 4096 | firestorm MC/KC × OpenBLAS NC |

計測形状は REQ-8 判定形状の M=N=K=2048／4096（+ 参考 1024）。各点 5 回計測の中央値
（`.claude/rules/coding-rust.md` 規約）とし、候補の計測順序を反復ごとにローテーションして
cache/TLB の系統的偏りを避ける（#559 の A/B テストのインターリーブ手法を候補数 N へ一般化）。
計測対象は実運用経路の `gemm_blis_parallel` 相当（`gemm_blis_parallel_with_blocks`。rayon 並列）。

テスト本体: `crates/backend-cpu/src/gemm_blis/mod.rs` の
`mc_kc_nc_blocking_sweep_median_throughput`（`#[cfg(target_arch = "aarch64")]` + `#[ignore]`）。
テストは勝敗を assert せず、標準出力へ形状 × 候補ごとの中央値を報告する（採用判断は人間／
後続セッションが行う。#559 §2.3 と同方針）。

## §4 実機での実行手順（M4 Max 到達可能セッション向け）

```bash
cargo test -p fandhe-ai-backend-cpu --release -- --ignored mc_kc_nc_blocking_sweep_median_throughput --nocapture
```

## §5 選定判断基準

- 中央値の改善が計測ばらつき（Q1〜Q3 幅相当）を超える候補があれば最良点を採用し、
  `crates/backend-cpu/src/gemm_blis/mod.rs` の `MC`/`KC`/`NC` 定数と選定コメントを更新する
- 有意差がなければ現行値を維持する（#24 の前例と同じ判断枠組み）
- REQ-8 下限値・許容誤差はいかなる結果でも変更しない（変更が必要な場合は停止して人間承認へ）
- 採用時は `tests/gemm_blis_parity.rs`（既定値経路のパリティ）の再実行で回帰がないことを
  確認する

### §5.1 判断結果（2026-08-19 実測後）

下記「§7 実機実測結果」のとおり、単一固定値の MC/KC/NC では 4096 と 2048 の双方で最良を
取れない（NC 拡大は 4096 で有利・2048 では不利という形状依存の逆転が生じる）ため、本イシュー
の範囲では現行値（MC=128／KC=256／NC=512）を変更しない。N（または問題サイズ）依存の NC 分岐
実装はイシュー #749（受け入れ条件: 4096 で 9% 級改善を獲得しつつ 512〜2048 の劣化を中央値
5% 以内に抑えること）へ引き継ぐ。

## §6 リスク・注意点（実装時に判明した事項）

- 並列経路（`gemm_blis_parallel`）は行パネル分割後に各タスクが `dispatch_region` を呼ぶため、
  MC が行パネル高さを超える候補では MC の効果が飽和しうる（スレッド数・パネル分割との相互
  作用は実機実測時に注記する）
- NC=9600／KC=4096 候補では panel バッファが数十〜百 MiB 級になる（B パネル: 9600×4096×4B ≈
  150MiB）。§3.3 のとおり確保前のオーバーフロー検査は不要と判断したが、確保失敗（OOM）自体は
  `vec![0.0f32; len]` のメモリ確保失敗として通常の Rust allocator エラー処理（プロセス abort）
  に委ねる。実機実測時にメモリ搭載量との兼ね合いで候補を絞る可能性がある
  - `gemm_blis_parallel`（および `gemm_blis_parallel_with_blocks`）は `par_chunks_mut` で
    分割した行パネルごとに `dispatch_region`→`PanelBuffers::new` を呼ぶため、B パネルは
    タスクごとに個別確保される。並列経路の実運用ピークメモリは上記単一パネル概算値の
    **アクティブスレッド数倍**になる（例: M4 Max の性能コア数相当で並列実行時、150MiB 級
    候補なら概算 150MiB × スレッド数）。実機実測時はスレッド数を含めて候補を絞る

## §7 実機実測結果（2026-08-19・M4 Max）と形状依存 NC 分岐の実装（#749。本番適用は §(ii) 参照）

### (i) 実測値表

出典: イシュー #749／親 issue #738・#735 に記載の 2026-08-19 M4 Max 実測（5 回計測中央値。
本リポ内スクリプトは §3・§4 の `mc_kc_nc_blocking_sweep_median_throughput`）。本セッション
（x86_64・aarch64 実機到達不可）は §0 追記のとおりこの実測値を引用するのみで再計測は行っていない。

| dim | 候補 | 中央値 | 対現行値 |
|---|---|---|---|
| 4096 | 現行値（#1 MC=128/KC=256/NC=512） | 0.134019 s | 基準 |
| 4096 | NC 拡大 firestorm（#3 MC=128/KC=256/**NC=9600**） | 0.120774 s | **約 9.9% 改善** |
| 2048 | 現行値 | （現行が最良） | 基準（NC 拡大は劣化） |
| 1024 | NC 拡大 firestorm（#3） | ― | 約 7.1% 改善（今回は未適用。理由は (iv)） |
| 全サイズ | KC 拡大単独（#4）・MC 拡大単独（#5）・firestorm 全軸（#6） | ― | 全サイズで劣化 |

### (ii) 採用ルール（未導入）と撤去の経緯

**本節は #749 実装セッション内で一時実装 → PR #766・codex-review 再指摘を受けて撤去した
経緯の記録であり、以下に記す `select_blocks`／`machine_detect` 等は HEAD の
`crates/backend-cpu/src/gemm_blis/mod.rs` にはもう存在しない**（撤去コミット: 「fix(backend-cpu):
未有効化の NC=9600 機種判定機構を撤去する」）。HEAD の本番 3 公開関数
（[`gemm_blis`]／[`gemm_blis_parallel`]／[`gemm_blis_bias_act_parallel`]）は常に
`default_blocks()`（MC=128／KC=256／NC=512）のみを使い、n 依存分岐・機種判定コードは
一切残っていない。

当初の採用ルール案は「`n >= LARGE_N_THRESHOLD`（= 4096）で `NC = NC_LARGE_N`（= 9600）・
MC/KC は現行値のまま、`n < LARGE_N_THRESHOLD` では従来どおり `default_blocks()`」という
n（B のパネル幅を分割する次元）のみを鍵にした分岐だった。実測が正方形状のみで m／k 依存の
知見がないため m／k は条件に含めない設計だった（非正方形状でのリスクは (vi) 参照）。

**適用対象の実行時機種判定を試みた経緯（PR #766・codex-review 再指摘 1 巡目への対応）**:
`NC_LARGE_N` は上表のとおり Apple M4 Max 実機 1 台でのみ実測した値であり、
`cfg(target_arch = "aarch64", target_os = "macos")`（Apple Silicon Mac 全般）は M1〜M3 等の
未検証機種も含んでしまう。そこで `select_blocks` に `machine_detect::is_m4_family`
（実測個体の `hw.model` を `machine_detect::VERIFIED_M4_MAX_HW_MODEL` と厳密一致で判定。
`sysctlbyname` FFI 経由）を組み合わせ、`true` を返した場合のみ `NC_LARGE_N` を適用し
それ以外は fail-closed で `default_blocks()` に留める実装を追加した。

**識別子未記録につき常に不活性だった（PR #766 再指摘・2 巡目）**: 当初 `hw.model` の
`"Mac16,"` prefix 一致で判定していたが、Apple の `Mac16,*` 識別子は 2024 発表の M4 世代
Mac 全機種（M4 無印・M4 Pro・M4 Max 搭載の MacBook Pro／Mac mini／iMac 各モデル）に
割り当てられており、M4 Max 以外の未検証機種にも一致してしまう問題があった。実測セッション
終了時点で個体の正確な `hw.model` 値が本 issue・親 issue #738・#735 のいずれにも記録されて
おらず復元不能なため、`machine_detect::VERIFIED_M4_MAX_HW_MODEL` を `None`（未記録）に
変更し、判明するまで `is_m4_family()` が常に `false` を返す（= `NC_LARGE_N` がどの実機にも
適用されない）状態にしていた。

**発火しない unsafe FFI を残すこと自体がリスクと判断し撤去（PR #766 再指摘・3 巡目）**:
識別子が判明する見込みが立たないまま「常に `false` を返すだけの `sysctlbyname` FFI 呼び出し」
を本番ビルドへ残すこと自体が codex-review P0（`unsafe extern "C"` 宣言への `// SAFETY:` 欠落）・
P1（最適化が一度も有効化されずデッドコード化）の指摘対象になったため、`select_blocks`／
`select_blocks_for`（判定ロジック）・`machine_detect`（`sysctlbyname` FFI 宣言・
`VERIFIED_M4_MAX_HW_MODEL` を含む）と対応する単体テストを撤去し、常に `default_blocks()`
を返す従来経路のみへ戻した。実測機の正確な識別子が判明し機種判定を安全に実装できる段階で、
本ドキュメントの実測値（§7 (i)）を根拠に再導入を検討する。M1〜M3 等での実測に基づく機種別
最適化（sysctl ベースの MC/KC/NC 動的算出への一般化）は引き続き #753 の対象とする。

### (iii) 512〜2048 が劣化 0% である構造的根拠（未導入案の時点の記録）

(ii) で撤去した当初案では、`n < LARGE_N_THRESHOLD` の場合 `select_blocks(n)` が
`default_blocks()` と完全に同一の `BlockSizes` を返す設計だった（`select_blocks_switches_nc_only_at_large_n_threshold`
テストで n ∈ {0, 1, 511, 512, 2048, 4095} を保証していた）ため、512〜2048 は分岐 1 回の
定数コストを除き変更前と完全に同一のコード経路・ブロック値で実行され、受け入れ条件
「512〜2048 の劣化が中央値 5% 以内」を実測を待たずに構造的に満たす、という設計上の性質
だった。HEAD では `select_blocks` 自体が撤去済みで、本番経路は常に `default_blocks()`
のみを使うため本節の分岐は存在しない（劣化 0% は自明に成立する）。

### (iv) 1024（7.1% 改善）を今回適用しない理由

512 が未計測・2048 は劣化という結果と 1024・4096 の改善が並ぶと非単調なテーブルになり、
512〜1536 帯（未計測区間）で劣化する可能性を否定できない。REQ-8 の判定形状は 2048／4096
であり 1024 は参考形状（§1・計画 §3.4）のため、安全側に倒して今回は 4096 のみを対象とした。
512／1536 を含む再計測を行ったうえで、#753（sysctl ベース MC/KC/NC 動的算出・2 次元ジョブ
分配）の検証材料として後続で判断する。

### (v) メモリ影響（概算）

B パネル容量は `panel_capacity` により `min(nc, n).div_ceil(NR) * min(kc, k) * NR` 要素。
n=4096・NR=12（NEON 既定）では `342 * 256 * 12 * 4B ≈ 4 MiB`／タスク（現行 NC=512 は
約 512 KiB／タスク）。並列実行時はアクティブタスク（スレッド）数倍（例: 16 スレッドで
約 64 MiB）。n > 9600 では NC=9600 クランプにより最大 約 9.4 MiB／タスク。REQ-14 の
4096³ 上限（理論最小 192 MiB の 2 倍＝384 MiB）に対し十分な余裕がある見込みだが、
`bench-harness` の peak-memory 計測（`make peak-memory-bench`・
`docs/perf/gemm-peak-memory-measurement.md`）による実機再確認は §7 の残課題とする。

### (vi) 残課題・リスク

- **512 未計測**: 512〜2048 帯で NC 拡大がどう振る舞うか未計測（(iv) 参照）
- **非正方形状未計測**: m／k を条件に含めない判断（(ii)）は正方形状の実測のみに基づく。
  m が極端に小さい／k が極端に大きい形状での NC=9600 の挙動は未検証
- **#753 への引き継ぎ**: sysctl ベースの MC/KC/NC 動的算出・2 次元ジョブ分配は #753 の
  スコープ（本イシューは実測テーブルベースの固定閾値分岐に留める）
- **B パネルのスレッド間共有**（NC 拡大でタスクごとの B パネル重複確保が増える問題）→ #750
- **MR12×NR8 不採用等の記録** → #751

### (vii) 再計測手順（未導入。再導入時に復元する手順の記録）

`select_blocks` を経由する A/B ハーネス（`shape_dependent_nc_vs_fixed_default_ab_median_throughput`
テスト）は (ii) の撤去に伴い削除済みで、HEAD には存在しない。再導入時（#753 での機種判定の
安全な実装完了後）は、本番経路（`gemm_blis_parallel`）と旧来固定既定値
（`gemm_blis_parallel_with_blocks(default_blocks())`）を dim ∈ {512, 1024, 2048, 4096} で
A/B 比較するハーネスを `crates/backend-cpu/src/gemm_blis/mod.rs` へ再実装し、受け入れ条件
（4096 で改善・512〜2048 で劣化 5% 以内）を実機再確認する。

REQ-8 下限値・数値一致許容誤差は本イシューでも一切変更しない。

## §8 #753 引き継ぎ状況（実装済み・実機計測待ち）

イシュー #753「ブロッキングの実行時キャッシュ検出（sysctl）と 2 次元タイルジョブ分配」で
実装した内容と状況を記録する。詳細な設計判断・算出式の根拠は
`docs/perf/cpu-gemm-runtime-cache-detect.md` を参照。

- **実装済み**: `sysctl`（`hw.perflevel0.l1dcachesize`／`l2cachesize`）実測値からの
  MC/KC/NC 算出（`crates/backend-cpu/src/gemm_blis/cache_params.rs`）・タイル境界に整列した
  2 次元寄りの行範囲分配（`crates/backend-cpu/src/gemm_blis/partition.rs`）・両者を組み合わせた
  実機 A/B 計測ハーネス（`crates/backend-cpu/src/gemm_blis/mod.rs` の
  `runtime_cache_detect_and_2d_partition_ab_median_throughput`。`#[ignore]`）
- **未実施**: Apple M4 Max 実機での A/B 計測（受け入れ条件 2）・本番 3 公開関数
  （`gemm_blis`／`gemm_blis_parallel`／`gemm_blis_bias_act_parallel`）への結線判断
- **本番未結線の理由**: #750・#758 と同型の「実装は入れるが実機ゲート未通過のうちは本番結線
  しない」判断（本セッション環境が Linux x86_64 で M4 Max 実機に到達不能なため）。§7 の
  機種判定方式（識別子未記録で撤去）とは異なり、本実装は機種判定を行わない算出式方式のため
  同じ理由での撤去リスクはない（詳細は `cpu-gemm-runtime-cache-detect.md` 参照）
- **再計測手順**: §(vii) と同じ位置づけで、実機セッションで
  `cargo test -p fandhe-ai-backend-cpu --release -- --ignored runtime_cache_detect_and_2d_partition_ab_median_throughput`
  を実行し、`default`／`detected`／`2d-partition` の中央値を dim ∈ {512, 1024, 2048, 4096} で
  比較・記録する
- **イシュー #794 への引き継ぎ（本イベント時点で未実施のまま）**: `compute_blocks` の算出式を
  「KC／MC は理論値と現行既定〈256／128〉の小さい方（キャップ）・NC は理論値をそのまま
  採用（キャップなし＝動的算出）」へ再較正した（M4 Max 実機未到達のため実機計測は #794 でも
  未実施。詳細・記録テンプレートは `cpu-gemm-runtime-cache-detect.md` §6）。本番未結線の
  状態・理由（実機ゲート未通過）は変わらない