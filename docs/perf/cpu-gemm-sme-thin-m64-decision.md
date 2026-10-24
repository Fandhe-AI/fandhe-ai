# m=64 専用 SME カーネル（4 タイル）の設計決定記録（イシュー #3234）

本書は docs のみの決定記録であり、`.rs`・`docs/spec`・tolerance・baseline・FMA 契約・REQ-8・`SME_MIN_*`・`SME_PRODUCTION_ENABLED` を変更せず、新規 `unsafe` も追加しない。実装は #3235、実測による既定の採否は #3236 が担う。

## 1. 位置づけと結論

- 判定は **(A) 4 タイルの m=64 専用カーネル**（§6 の所有者回答で確定）。
- #3235 は**既定 OFF の独自ゲート**で実装する。既定を反転するかは #3236 の事前登録 A/B（ADOPT）で決める。推定値は合格ラインに使わない。
- 承認された範囲は「設計を (A) とし #3235 へ進めてよい」までであり、新規 `unsafe asm!` の security-auditor 監査は**未実施**（#3235 で実施。§7）。

## 2. 入力

| 出典 | 使う内容 |
|---|---|
| `docs/perf/cpu-sme-thread-scaling-macmini.md`（#3175）§2・§4・§6 | スレッド別 GFLOPS・micro-call 固定費 |
| `docs/perf/cpu-gemm-sme-fmopa-microkernel.md` §3.2・§4・§5.10 | 4 タイル（32×32）を `MAX_TILE` を理由に見送った経緯・NaN の扱い |
| `docs/perf/cpu-gemm-sme-ab-macmini.md`（#3176） | Mac mini の SME A/B |
| `docs/perf/cpu-gemm-m64-jobgrid-diagnosis.md`（#3177）§2 | m=64 級 6 形状と転置の種類 |
| `docs/perf/cpu-gemm-neon-direct-decision.md` §3 | Mac mini infer/fresh 208.2 µs・PyTorch 比 0.155 |
| `docs/cpu-gemm-sme-unsafe-audit.md` §12 | 現行の監査範囲・残留リスク L3 |
| `crates/backend-cpu/src/gemm_blis/microkernel/sme.rs`・`gemm_blis/mod.rs`（origin/main `281597f58` 時点の行番号） | MR/NR=16（`sme.rs:63`・`:65`）・`MAX_TILE=256`（`mod.rs:125`）・`m == 1` 分岐（`mod.rs:467`・`:528`・`:780`）・`SME_PRODUCTION_ENABLED=true`（`mod.rs:3540`）・`SME_MIN_M/N/K=256/256/64`（`mod.rs:3551` 付近） |

## 3. 現状

- 現行 SME カーネルは MR=16×NR=16 で ZA0 の 1 タイルのみを使い、micro-call ごとに `smstart`→C プリロード→`fmopa` ループ→ストア→`smstop` を行う。
- `SME_PRODUCTION_ENABLED=true`（#3231）。m=64 は `sme_shape_eligible`（`SME_MIN_M=256`）を満たさず NEON へ落ちる。
- 仮に現行カーネルで (64,256,784) を処理すると micro-call は 4×16×4=256 回（#3175 §4 の算術）。

#3175 の実測（Mac mini・5 run 中央値）:

| 項目 | 値 |
|---|---|
| (64,256,784) SME の GFLOPS（T=1/2/4/10） | 220.2／154.8／194.0／272.7（T=2 で T=1 より遅い） |
| 同 NEON | 91.0／95.1／127.0／148.1 |
| 固定費（上限） | `kc=0` で 44.5 ns/call、切片 89.8 ns |
| 256 組が SME 時間に占める割合 | T=1 で 9.8〜19.7%、T=10 で 1.2〜2.4% |
| 定常の傾き | 0.978 ns/kc（1 タイル 512 FLOP/kc で 523 GFLOPS 相当） |
| 大形状の頭打ち | なし（(2048,2048,1024) で S(10)=3.07）。ただしピーク 2008 GFLOPS の 54% 止まり |

- #3175 自身が「固定費だけでは差を説明しきれない」と書いている。
- **推定**: 1 タイルの `fmopa` 連鎖は同一 ZA0 への依存で直列化しやすく、独立 4 タイル化で発行効率が上がる見込みがある。(A) の利得は `smstart`/`smstop` の削減だけでなくこの効果が主と推定するが、実測で確かめるまで確定しない。#3175 §6 も固定費の削減策として ZA0〜ZA3 化と streaming 維持を挙げている。

## 4. 設計（(A) の形）

### 4.1 タイル構成

SVL=512 bit（16 f32）を前提に ZA0.S〜ZA3.S の 4 タイルで 64 行×16 列の C ブロックを持つ（タイル t が行 16t〜16t+15）。p ごとに A の 4 ベクトルと B の 1 ベクトルを `ld1w` で読み、`fmopa` を 4 発撃つ。命令は SME1 の範囲に限り、SME2 の多ベクトルロードには頼らない（検出条件を増やさないため）。

### 4.2 MAX_TILE を避ける構造

64×16=1024 要素は `MAX_TILE=256` を超える。このため本カーネルは `Microkernel` の実装にせず、`gemm_blis_region` と端タイルのスタックバッファを通らない別の入口とする。分岐は `m == 1` 分岐（`gemm_row_vector` へ迂回）の直後に置き、C の 64×16 ブロックへ `ldc = n` で直接ストアする。これにより `MAX_TILE`・NEON の端タイル経路・GB10 の経路は変わらない。fmopa doc §3.2 が 32×32 を見送った理由（`MAX_TILE` 拡大による GB10 の端タイル経路への影響）は当てはまらない。

### 4.3 K の扱い

kc で分割せず K 全長（例 784）を 1 つの `smstart` 区間で処理する。(64,256,784) の `smstart`/`smstop` は列ブロック 16 個で 16 組（256 組から削減。算術）。B パネルは 784×16×4 B ≈ 49 KiB、A の pack は 784×64×4 B ≈ 196 KiB（算術）。

### 4.4 pack

- A は `[p][64]`（p-major。既存 `pack_a` の `dst[p*mr+i]` 契約と同型）へ `par_iter` の前に 1 回だけ pack し、全列ブロックの job で共有する（`share_a` と同じ考え方）。
- B は Nn なら `B[p][j0..j0+16]` が連続なので pack せず、ストライド `n*4` で直接 `ld1w` する。Nt は B^T の pack が要る。
- A の pack（約 5 万語）の時間は**推定**で十数 µs かかりうる。並列化するかは #3236 の実測で決める。

### 4.5 ゲートと適格条件

- 新しい `const`（例 `SME_THIN_M64_ENABLED`、既定 `false`）と純関数の適格判定の両方が真のときだけ `SmeKernel::try_new()` を呼ぶ（短絡評価）。環境変数・実行時設定では切り替えない（A03）。既存の `SME_MIN_*` は変えない。
- 適格条件の初版: `m == 64 && n % 16 == 0 && n >= 16`、転置は Nn と Nt、Tn は対象外。k の下限は #3236 の格子で決める。
- n%16≠0 はマスク付きストアを使わず従来経路へ落とす（asm を単純に保つため）。マスク付きストア案は #3236 以降の比較対象とする。

### 4.6 到達セル（#3177 の 6 形状）

| 形状 | 到達 |
|---|---|
| L1 fwd 64×256×784 Nn | 到達 |
| L1 dX 64×784×256 Nt | n=784=49×16 で Nt を扱えば到達 |
| L2 fwd 64×10×256 | n=10 で非到達 |
| L2 dX 64×256×10 Nt | 条件上は到達。k=10 と小さく k の下限しだい |
| dW（m≠64） | 非到達 |

train CPU に効くのは forward と dX の GEMM 経由だけである。

### 4.7 数値契約

各 `c[i][j]` は「初期値 = 呼び出し時の C、p 昇順に 1 回ずつ fma」の連鎖になる。現行カーネルは kc ブロック N のストアがブロック N+1 のプリロードになるため、K 全長を 1 回で流しても演算列は同じ。p ごとの 4 発は別タイルに当たり、要素ごとの順序は変わらない。NEON とも同じ連鎖で bit 一致する見込み（fmopa doc §3.2 の議論を継承。実機の bit 一致は #3235 の `#[ignore]` テストで確認する）。

- SME 非搭載（GB10 を含む）・SVL≠64・`TPIDR2_EL0≠0` のときは `compute` を呼ばず従来経路へ落ちる（fail-closed）。出力は bit 同一。
- NaN は fmopa doc §4 と同じくクラス一致の扱い。tolerance・baseline・FMA 契約は変えない。

### 4.8 epilogue（bias＋ReLU）

初版は asm へ融合しない。`smstop` の後に既存の `apply_epilogue`（`x += bias[j]` → `x.max(0.0)`）をブロック単位で当て、現行と同じ意味を構成上保つ。融合を将来採る場合の条件:

- Rust の `f32::max` の NaN の扱いに合わせ `fmax` ではなく `fmaxnm` を使う。
- ストリーミング SVE モードでの SVE `fadd`／`fmaxnm` の合法性は Arm ARM の一次ソースで確認する。**本書では未確認**。
- 符号付きゼロと NaN の bit テストを足す。

### 4.9 REQ-8 境界検査

`asm!` に入る前に Rust 側で `checked_mul`／`checked_add` により `a.len()`・`b.len()`（ストライド `n` を含む）・`c.len() >= 63*n + j0 + 16`・A pack の長さを検査し、違反は型付きエラーで返す（panic しない）。性能を理由に省かない。

### 4.10 新規 `unsafe asm!` の範囲

- asm ブロックは 1 つに局所化し、`smstart`/`smstop` を同一ブロックで対にする。v0〜v31・p0〜p15・ffr・使う w12〜w15 を clobber に宣言し、`preserves_flags`・`nomem` は付けない。
- **ZA0〜ZA3 の全タイルを壊す**ことを SAFETY 契約に明記する。
- 残留リスク L3（ストリーミング中のシグナル配送）は引き継ぐリスクで、ストリーミング区間が長くなる点を明記する。新しいリスクではない。

## 5. (A)(B)(C) の比較

| 案 | 内容 | 評価 |
|---|---|---|
| **(A)** | 4 タイルの専用カーネル | 推奨・確定。`smstart`/`smstop` は 256 組から 16 組へ。発行効率の改善も見込む（推定）。費用は新規 `unsafe asm!` 1 ブロック・監査・テスト |
| (B) | 現行 1 タイルのまま `SME_MIN_*` を下げる | 256 組が残り、定常は 1 タイル上限（523 GFLOPS 相当）のまま。M/N≤128 へ下げると Mac mini R4 格子で 5/5 を満たさない点が入る（`mod.rs` の const コメント）。#3231 の確定値を開き直す |
| (C) | 見送る | Mac mini の m=64 級 CPU セル（infer/fresh 208.2 µs・PyTorch 比 0.155）が改善しない |

期待利得の目安として「50,176 発×512 FLOP／約 2 TFLOPS ≒ 約 13 µs」という計算ができるが、これは**推定**であり合格ラインには使わない。

## 6. 判定と所有者回答

| 項目 | 内容 |
|---|---|
| 出所 | <https://github.com/Fandhe-AI/fandhe-ai/issues/3234#issuecomment-6111765895> |
| 日時 | 2026-10-11T17:34:53Z（2026-10-12 JST） |
| 回答 | 「全て推奨で」 |
| 確定 | (A) |

条件:

1. 既定の反転は事前登録規則で ADOPT されることが前提。推定値は合格ラインに使わない。
2. 個々の `unsafe` には理由コメントと security-auditor の監査が要る。
3. FMA 契約・tolerance・baseline・REQ-8 は変えない。#3175 の実測を引く。

承認されたのは設計を (A) とし #3235 へ進めてよいところまで。(B)・(C) にはならなかったため、その場合の扱いは該当なし。

## 7. 監査の手順（#3235 向け）

- 追記先は `docs/cpu-gemm-sme-unsafe-audit.md` の次の空き節（現状 §12.11 まで。§13 の見込みだが #3235 の時点で確かめ直す）。
- 実装 PR で security-auditor に読み取り専用の監査を依頼する。観点は既存の M/L/P/R/G/D、aarch64 2 ターゲットの clippy、OWASP 対応、SAFETY コメントと clobber の網羅、REQ-8 の事前検査、フォールバック。結果は同じ PR で audit doc に追記する。
- 本書の PR では audit doc を変更しない。

## 8. 後続と未確認事項

- #3235: 既定 OFF での実装と bit 一致テスト（CI は非 SME 経路、実機は `#[ignore]`）。
- #3236: 事前登録規則による Mac mini の A/B。ADOPT なら既定を反転する。GB10 は `try_new` が失敗するため経路は変わらず、非後退は bit 同一の確認で示す。
- 未確認: ストリーミング中の `fmaxnm`、A pack の並列化、スレッド方針（SME ユニットは P クラスタに 1 基のため並列でも伸びない可能性がある。推定）。

## 9. 変更していないもの

`.rs`・`docs/spec`・tolerance（`RELATIVE_TOLERANCE`・`ABSOLUTE_RESCUE_THRESHOLD`）・baseline・FMA 契約・REQ-8・`SME_MIN_*`・`SME_PRODUCTION_ENABLED`・ruleset／リポジトリ設定。

## 10. #3235 の実装での差分（2026-10-24 追記）

§1〜§9 の判定は変えない。実装で確定した差分のみを記す。

- **C の扱い**: §4.2 の「`ldc = n` で直接ストア」ではなく、job-local の連続バッファ（ldc=16）へ copy-in → カーネル → 並列区間後に直列で copy-out する。スレッド間で生ポインタを共有する新規 `unsafe` を増やさないため。copy は値の転写のみで演算列は変わらず bit 一致を保つ。カーネルは `ldc` を引数に取る汎用形のまま。直接ストアへの戻しは #3236 の結果を見て別 issue で扱う。
- **挿入位置**: `gemm_blis_parallel_with_transpose`（Nn／Nt）と `gemm_blis_bias_act_parallel_prepacked`（Nn。bias+ReLU は GEMM 完了後に `apply_epilogue` を 1 回）の `m == 1` 分岐の直後。GB10 affinity／`run_capped` の外にあり、現在の rayon プールで動く。スレッド方針は #3236 で扱う（GB10 は `try_new` が `None` でここへ入らない）。
- **適格条件**: `m == 64`・`n >= 16 && n % 16 == 0`・`k >= 1`・Nn／Nt。k の下限は #3236 の格子で決める。
- **ゲート**: `SME_THIN_M64_ENABLED = false`（既定 OFF）。監査は `docs/cpu-gemm-sme-unsafe-audit.md` §13。性能は本 issue では測っていない。

## 11. #3236 の A/B（2026-10-24 追記。状態: 実測済み・判定 undetermined）

§1〜§10 は変えない。計測基盤と事前登録規則（PR #3575）に加え、本節は 2026-10-24 の Mac mini 実測結果の追記を含む。

- **事前登録規則**: `docs/perf/logs/sme-thin-m64-ab-3236/RULE.txt`。腕は before = main（`SME_THIN_M64_ENABLED = false`）・after = `on-arm.patch`（同宣言 1 行を true）。ADOPT／ADOPT-k／REJECT／FAIL／undetermined を機械判定する。
- **計測**: GEMM 単体 7 形状（到達 3・非到達 4）、framework-compare の train／infer size=64（fresh／reuse）、k 下限用の K 格子（`cpu_sme_thin_m64_kgrid`、k = 10〜784）。Mac mini のみ。GB10 は SME 非搭載で経路が変わらないため測らない。
- **到達の静的確認**: infer は `ops.rs:464`・`:575` から thin 分岐、train は L1 fwd と L2 dX（`ops.rs:260/272/294`）。train の L1 dX が #3559 以降で実行されるかは未確認で、到達根拠に数えない。
- **bit 同一の根拠**: framework-compare の checksum（6 桁丸め）は bit 同一の証明にならない。根拠は P-BIT（Mac mini 実機での `sme_thin_m64` テスト）とする。
- **既定の扱い**: 判定 undetermined のため OFF のまま（下記「実測結果」）。反転 PR は ADOPT／ADOPT-k に加え、(i) `docs/cpu-gemm-sme-unsafe-audit.md` §13 の独立した security-auditor 監査（現状は実装担当の自己点検のみ）と (ii) 所有者の確認が前提。
- **相手比**: 下記「実測結果」に記録（PyTorch 同 round の値のみ「実測」。issue の到達目安は「推定」で合格ラインに使わない）。

### 11.1 実測結果（Mac mini・Apple M4・2026-10-24。機械出力は `docs/perf/logs/sme-thin-m64-ab-3236/aggregate.md`）

- **判定: undetermined**（`aggregate.py` の機械判定。FAIL なし・前提ゲートはすべて成立・5/5 一貫の後退セルなし・K 下限は決まらず）。**既定は OFF のまま、Rust は変えない**。
- **事前登録規則の訂正（RULE-amend-1・計測前）**: #3580 で `SME_PRODUCTION_ENABLED` が false に戻ったため、P-TREE の同条件を「両腕とも false」に改めた（所有者の「#3580 マージ後の main で計測」に従う）。薄型経路の分岐は `SME_THIN_M64_ENABLED` だけで決まるため腕は分離できる。副作用として非到達ガードセル（l1_dw・ref_square_512・train の L1 dW）は両腕とも NEON になる。分類・判定式・N=5・tolerance・baseline は不変。
- **計測ツリー**: コミット `3f122311f`（`rev-stamp.txt` の 1 行目と一致。after は `+on-arm.patch`）。5 round・負荷ゲート（load1 < 3.0）5/5 通過。
- **実機ロック**: 転送段 2026-10-24T01:08:28Z〜01:08:35Z、計測段 取得 01:08:41Z・解放確認 01:17Z 頃（`macmini/hwlock.log`。解放はスクリプトの trap で、終了後にロックディレクトリの不在を確認）。
- **到達セルの 5 round 中央値比（ON/OFF。<= 1.00 なら ON が速い）**:

| セル | 比 | 5 round 中 <=1.00 |
|---|---|---|
| GEMM l1_fwd（64x256x784） | 0.704 | 5/5 |
| GEMM l1_dx（64x784x256） | 0.537 | 5/5 |
| GEMM l2_dx（k<64 の到達セル） | 0.987 | 3/5（r1=1.092・r2=1.078 が後退） |
| train 64 fresh／reuse | 0.798／0.885 | 5/5／5/5 |
| infer 64 fresh／reuse | 0.747／0.709 | 5/5／5/5 |

- **undetermined になった理由**: 到達セルのうち GEMM l2_dx が「5/5 で <= 1.00」を満たさない（3/5）ため ADOPT ではなく、かつ 5/5 一貫の後退も無いため REJECT／ADOPT-k でもない。K 格子も k=10・16・32・784 で 5/5 を満たさず、「k >= K の全点が 5/5 で ON <= OFF」となる K が格子上に無い。k=64・128・256 は 5/5 で ON <= OFF（比 0.903・0.651・0.556）。l2_dx 系は 30 us 台と短く、round 間の揺らぎ（0.915〜1.092）が大きい点は事実として記録する（規則は緩めない）。
- **非到達ガードセル**（両腕とも NEON のため差は雑音）: l2_fwd 1.022・l1_dw 1.014・l2_dw 1.065・ref_square_512 1.061。いずれも 5/5 一貫の後退ではない。
- **checksum**: framework-compare は train／infer の全セルで「完全一致」。GEMM 単体・K 格子の判定も `aggregate.md` の前提ゲートで全一致。ただし checksum の一致は bit 同一の証明ではない（RULE §7）。
- **P-BIT（Mac mini 実機・main ツリー）**: `--lib sme_thin_m64` 8 passed／`--ignored sme_thin_m64 --test-threads=1` 4 passed（`p_bit exit=0`）。#3233 独立監査の P2 #3（非正規化数・NaN の実機判定）の根拠として、次の 3 件が ok: `…_matches_scalar_reference_denormal`・`…_denormal_result_from_normal_operands`・`…_nan_input_does_not_panic`。**判定範囲**: 非正規化数は `assert_eq` の値一致（非ゼロの非正規化数では bit 一致と等価。±0 の符号は区別しない）、NaN はクラス一致（`is_nan`）で payload は比較しない（既存の「NaN のみクラス一致」契約と同じ）、FPCR は既定（FZ=0・DN=0）。`to_bits` による NaN payload・±0 符号の厳密判定は未実施で、反転 PR 側の範囲とする（本 PR は Rust のテストを足さない）。
- **PyTorch との比（同 round の実測。PyTorch median / fandhe median の 5 round 中央値。値が大きいほど fandhe が近い。判定には使わない）**: train 64 fresh 0.197（OFF）→ 0.248（ON）、train reuse 0.273 → 0.306、infer fresh 0.157 → 0.216、infer reuse 0.155 → 0.221（torch 2.14.0・bench_py.py sha256 は `macmini/ab/env_info-3236-sme-3236-r1.txt`）。GEMM 単体は同 round の相手実測なし。issue に書かれた到達目安（infer 40〜55 us 等）は**推定**で、合格ラインに使っていない。
- **未実測**: GB10（SME 非搭載で経路が変わらないため対象外）、`SME_PRODUCTION_ENABLED = true` 下での A/B（#3232 の GB10 REJECT により既定 false）、l2_dx の round 数を増やした再計測。M4 Max の値は Mac mini の値として使っていない。
- **処置**: 既定 OFF のまま。再計測が必要なら RULE §5 のとおり末尾に RULE-reopen を追記してから新しいラベルで測る（既存系列は差し替えない）。反転 PR は作らない。仮に将来 ADOPT／ADOPT-k になった場合の前提は (i) 所有者の確認、(ii) 独立監査の P2 条件（`docs/cpu-gemm-sme-unsafe-audit.md` §13.6。P2 #1 のゲート結合 `SME_PRODUCTION_ENABLED && SME_THIN_M64_ENABLED` は production SME が false の現状では薄型経路を死に経路にするため、扱いは所有者と反転 PR の判断事項）。
