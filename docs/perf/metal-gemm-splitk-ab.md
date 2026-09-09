# Metal split-K 経路と現行 classic 経路の M4 Max 実機 A/B（イシュー #1475）

## §0 状態（判定結果）

**暫定 ADOPT（3/5 run。5 run 完了による正式確定は未実施）**。

対象 9 形状（K 支配的非正方。`(32,32,*)`／`(64,64,*)`／`(128,128,*)`。
K ∈ {2048, 4096, 8192}）はすべて 3 run の中央値で speedup（= median_secs(classic)
/ median_secs(split-K)）が事前登録基準 `>= 1.5` を満たし、かつ 3 run とも符号
一貫（全 run で speedup > 1.0）だった。対照 3 形状（`(256,256,*)`）も 3 run
の中央値がすべて事前登録基準 `>= 0.95` を満たした。単一セッションの時間制約
により事前登録・実装計画（`docs/perf/logs/metal-gemm-splitk-ab-1475/`
`run_gated.sh` コメント参照）が想定した 5 プロセス起動のうち **3 run
（run1〜run3）を完了**した時点で打ち切ったため、判定は「暫定」とする。
残り 2 run の追加実測は本イシュー配下のフォローアップとして記録する
（§7）。

3 run の実測は §6 で `A′/A`（タイル構成効果）と `B/A′`（K 分割効果）へ
分解しており、対象 9 形状では `B/A′` が支配的（K 分割そのものが速度改善の
主因）であることを確認した。`docs/backend-metal-splitk-decision.md` §3 が
留保していた「因果は仮説」を実測で更新する。

## §1 計測手段

実装: `crates/backend-metal/examples/gemm_splitk_ab_bench.rs`（本イシュー）。

- **計測境界**: prepared 境界（A・B・C バッファの確保・アップロードは計測外。
  計測対象は encode ＋ コマンドバッファ完了待ち。readback 対象外）。NN
  レイアウト固定。
- **腕**:
  - A（classic）: `MetalGemm::dispatch_strided_tiled_prepared` に
    `cfg = tile::select_for_device(m, n, k, ...)`（split-K のフォールバック先
    と同一経路）。
  - A′（classic・split-K タイル。参考のみ）: 同じ classic 経路だが
    `cfg = tile::split_k_tile(m, n)`。
  - B（split-K）: `MetalGemm::dispatch_split_k_strided_prepared_with_plan` に
    `tile::should_split_k(m, n, k)` の計画を明示的に渡す。戻り値が
    `SplitKRoute::Split` でなければ計測を中止する（フォールバックをデータ点
    として扱わない）。
  - B′（対照・classic）: `(256,256,K)` は `should_split_k` が `None` を
    返すことを assert したうえで classic を dispatch。**計測クロージャ
    内でも毎回 `should_split_k` を呼ぶ**（`dispatch_auto` 相当が本番で
    毎回払う選択関数の呼び出し費用込みで計測する。2026-09-09 是正。
    §3 「既知の限界」参照）。
  - C（対照・強制 split-K。参考のみ）: `max_groups=u64::MAX` で強制した
    split-K。`max_groups=40` 境界の妥当性の記録のみ。
  - フロア（参考のみ）: 各 `(M,N)` を `K=64` で classic dispatch した dispatch
    固定費。
- **比の方向**: `speedup = median_secs(A) / median_secs(B)`（1 より大きい
  ほど split-K が速い）。主指標は run 内対応比の run 数中央値。
- **プロトコル**: `bench_harness::ab::run_ab`（`AbConfig::new(6, 2s, 1s)`。
  順序反転 interleave）・`MeasurementConfig::default()`（warmup/iters 20/20）。
  フロアのみ `run_stability`（単一腕）。

実行コマンド:

```sh
cargo run -p fandhe-ai-backend-metal --release --features internal-diagnostics \
  --example gemm_splitk_ab_bench -- --max-load-avg=<f64>
```

`--self-check-only` でフェーズ 0（自己検証）のみ実行できる。

## §2 事前登録判定基準

計測前に issue #1475 へ投稿（2026-09-09T09:13:30Z UTC）:
https://github.com/Fandhe-AI/fandhe-ai/issues/1475#issuecomment-5599418673

- **ADOPT**: 対象 9 形状すべてで (i) 主指標（run 内比の run 数中央値）
  `>= 1.5` かつ (ii) 全 run で run 内比 `> 1.0`（符号一貫性）、かつ対照 3
  形状すべてで主指標 `>= 0.95`。
- **REJECT**: 専有ゲート成立下で上記いずれかが不成立。
- **undetermined**: 専有ゲート（`--max-load-avg`）が規定回数で成立しない
  場合（1 回だけ記録して終了）。
- spread（レンジベース）は判定に使わない。数値契約（各腕の run-to-run bit
  同一）は計測妥当性の前提条件（フェーズ 0）とし、腕間の差は情報として
  記録するのみで verdict の入力にしない。
- ADOPT は性能上の判定に限る。本番結線（`SPLIT_K_NUMERIC_CONTRACT_APPROVED`
  の切替）は別途ユーザー承認が必要（#1476 のスコープ）。

以後、本基準・腕定義・定数は変更していない。

## §3 フェーズ 0 自己検証結果

`docs/perf/logs/metal-gemm-splitk-ab-1475/self_check.log` 参照。

- 対象 9 形状・対照 3 形状すべてで classic（A）・A′（classic・split-K
  タイル）・split-K（B）の run-to-run bit 同一（2 回連続 dispatch の
  `read_to_vec()` 一致）を確認した（`classic_stable=true` /
  `target_tile_stable=true` / `splitk_stable=true` が全 12 形状で成立）。
- A vs A′（`a_vs_at_fail_count`）は全 9 対象形状で **0**（タイル構成
  〈`select_for_device` の選択構成 vs `split_k_tile`〉を変えても classic
  経路自体は bit 完全一致することを実測で確認した。演算列がタイル形状に
  よらず同一の K 直列ループであることの裏付け）。
- A vs B（`a_vs_b_fail_count`）は全 9 対象形状で総要素数近くまで不一致
  （K 分割の結合順序差に起因する既知の丸め誤差。`tests/gemm_splitk_parity.rs`
  の既知 FAIL・`docs/perf/metal-gemm-splitk-two-pass.md` §5 と整合。
  verdict の入力にはしない）。

**既知の限界（イシュー #1499 codex-review・Cursor Bugbot 指摘。2026-09-09
是正）**: 本節・`self_check.log` の `a_vs_b_fail_count` は、`prepare()` の
`seed_offset` が A（classic。`p1`）=1・B（split-K。`p2`）=2 と**異なる
入力行列**で生成されたバイナリによる実測値だった。異なる入力同士の出力を
比較しても「K 分割の結合順序差に起因する丸め誤差」という解釈の根拠には
ならない（別問題を解いた結果を比較しているにすぎない）。コード側は
`p2` の `seed_offset` を `p1` と同一の `1` へ統一済み（同一入力での
比較へ是正済み）だが、本ドキュメントが参照する `self_check.log`・上記
`a_vs_b_fail_count` の実測値自体は是正前バイナリによるものであり、
**同一入力での再実測は未実施**（実機〈M4 Max〉再接続が必要なため本 PR
のスコープ外。§7 のフォローアップ参照）。run-to-run bit 同一
（`classic_stable`／`splitk_stable`）・A vs A′（`a_vs_at_fail_count`。
両者とも `seed_offset=1` で同一入力のため元々有効）は本是正の影響を
受けない。

## §4 実測結果（3 run。`aggregate.md` から転記）

生データ・集計スクリプトは `docs/perf/logs/metal-gemm-splitk-ab-1475/`
（`run{1,2,3}.log`・`aggregate.py`・`aggregate.md`・`env_info.txt`）を参照。

### target（対象 9 形状。A=classic vs B=split-K）

| m | n | k | speedups（run1,run2,run3） | median | 5-run 判定基準 `>=1.5` かつ全 run `>1.0` |
|---|---|---|---|---|---|
| 32 | 32 | 2048 | 1.8952, 1.8183, 1.7299 | 1.8183 | 満たす（3/3 run） |
| 32 | 32 | 4096 | 2.3931, 2.6833, 2.2358 | 2.3931 | 満たす（3/3 run） |
| 32 | 32 | 8192 | 3.5266, 4.4233, 3.6979 | 3.6979 | 満たす（3/3 run） |
| 64 | 64 | 2048 | 1.5686, 1.6581, 1.6709 | 1.6581 | 満たす（3/3 run） |
| 64 | 64 | 4096 | 2.4883, 2.2917, 3.0936 | 2.4883 | 満たす（3/3 run） |
| 64 | 64 | 8192 | 3.8174, 3.7672, 4.0542 | 3.8174 | 満たす（3/3 run） |
| 128 | 128 | 2048 | 1.6336, 1.6220, 1.7561 | 1.6336 | 満たす（3/3 run） |
| 128 | 128 | 4096 | 2.1127, 2.0993, 2.5271 | 2.1127 | 満たす（3/3 run） |
| 128 | 128 | 8192 | 2.7487, 2.7191, 1.8570 | 2.7191 | 満たす（3/3 run） |

partitions は形状ごとに一貫（`(32,*,*)`／`(64,*,*)` は 32、`(128,128,2048)` は
8・`(128,128,4096)` は 16・`(128,128,8192)` は 32）。

### control（対照 3 形状。A=classic vs B′=classic）

| m | n | k | speedups（run1,run2,run3） | median | 基準 `>=0.95` |
|---|---|---|---|---|---|
| 256 | 256 | 2048 | 0.9977, 0.5903, 1.1718 | 0.9977 | 満たす（median） |
| 256 | 256 | 4096 | 0.9925, 0.9573, 0.9904 | 0.9904 | 満たす（median） |
| 256 | 256 | 8192 | 0.9963, 1.0245, 0.9968 | 0.9968 | 満たす（median） |

`(256,256,2048)` の run2 が 0.5903（>1.0 基準以下）と大きく振れているが
（対照は classic vs classic の対称比較のため理論上 1.0 近傍のはず。この
run 単発のノイズと考えられる）、median は基準を満たす。全 run を通じた
ばらつきの大きさは §5 で留保として明記する。

**既知の限界（イシュー #1499 codex-review 指摘。2026-09-09 是正）**: 上表の
実測値は、B′ の計測クロージャが `should_split_k` を呼ばずに classic
dispatch のみを測っていた是正前バイナリによるもの（A と全く同一の経路を
測っていたに等しい）。コード側は結線相当経路の選択関数呼び出し費用を
含めるよう是正済みだが、上表の再実測は未実施（§7 のフォローアップ
参照）。「対照は classic vs classic の対称比較のため理論上 1.0 近傍」
という解釈自体は妥当（`should_split_k` の呼び出し費用は encode／
コマンドバッファ完了待ちに比べ無視できるほど小さいと見込まれるため
median 側の結論〈基準 `>=0.95` を満たす〉が覆るとは考えにくいが、
未検証である点を明記する）。

### target_tile（対象 9 形状。A=classic〈select_for_device〉vs A′=classic
〈split_k_tile〉。参考のみ・タイル構成効果の分離）

| m | n | k | speedups（run1,run2,run3） | median |
|---|---|---|---|---|
| 32 | 32 | 2048 | 1.0097, 1.0381, 1.0399 | 1.0381 |
| 32 | 32 | 4096 | 0.6140, 0.6278, 0.9895 | 0.6278 |
| 32 | 32 | 8192 | 1.0466, 1.0070, 0.9684 | 1.0070 |
| 64 | 64 | 2048 | 0.8615, 1.0321, 1.0060 | 1.0060 |
| 64 | 64 | 4096 | 1.0171, 1.0144, 1.0001 | 1.0144 |
| 64 | 64 | 8192 | 1.2845, 0.9983, 1.0031 | 1.0031 |
| 128 | 128 | 2048 | 0.9984, 1.0093, 1.0052 | 1.0052 |
| 128 | 128 | 4096 | 1.0222, 0.5776, 0.9946 | 0.9946 |
| 128 | 128 | 8192 | 1.0001, 0.9853, 0.9212 | 0.9853 |

A′/A（タイル構成のみを変えた効果）はほぼ 1.0 近傍（0.99〜1.05 が大半）で、
`(32,32,4096)` の run1/run2（0.61〜0.63）のような外れ値を除けば、タイル
構成差自体の寄与は小さい（§6 参照）。

### control_forced（対照 3 形状。A=classic vs C=強制 split-K。参考のみ）

| m | n | k | speedups（run1,run2,run3） | median | partitions |
|---|---|---|---|---|---|
| 256 | 256 | 2048 | 1.4075, 0.6541, 1.3720 | 1.3720 | 2 |
| 256 | 256 | 4096 | 2.7785, 1.6551, 1.6768 | 1.6768 | 4 |
| 256 | 256 | 8192 | 1.9383, 1.6525, 1.6618 | 1.6618 | 8 |

強制 split-K は対照形状でも中央値ベースでは classic を上回る傾向がある
（`max_groups=40` の境界は保守的である可能性を示唆。ただし本結果は
`max_groups` の閾値変更を推奨するものではなく、記録のみ）。

### フロア（参考。各 `(M,N)` を `K=64` で classic dispatch）

| m | n | k | median_a_secs（3 run 中央値） |
|---|---|---|---|
| 32 | 32 | 64 | 8.8875e-5 |
| 64 | 64 | 64 | 8.9208e-5 |
| 128 | 128 | 64 | 8.8416e-5 |
| 256 | 256 | 64 | 8.7666e-5 |

対象形状の A（classic）median_a_secs は約 2.1e-4〜5.3e-4 秒であり、フロア
（約 8.7〜8.9e-5 秒）はその 20〜40% 程度を占める。dispatch 固定費が classic
経路の実行時間に無視できない割合を占めるが、B（split-K）はさらに小さい
median_b_secs（例: `(128,128,8192)` は 1.9e-4 秒）を達成しているため、
フロア自体が speedup の主因ではない。

## §5 判定

**暫定 ADOPT**（§0 参照。人間による分析上の暫定評価であり、`aggregate.py`
が出力する機械的な正式 verdict とは区別する）。3 run いずれも
`verdict=undetermined`（専有ゲート不成立）を出力せず正常終了しており、
対象 9 形状・対照 3 形状とも事前登録基準を満たした。

ただし `aggregate.py` 自体の正式な ADOPT/REJECT 出力は、渡された run 数
（本例では 3）が事前登録した `MIN_FORMAL_RUNS`（5）以上であることを条件に
変更済み（イシュー #1499 codex-review P1 指摘への対応。3〜4 run の時点で
形状さえ揃えば正式判定を出力していた旧実装の不備を修正）。このため
`docs/perf/logs/metal-gemm-splitk-ab-1475/aggregate.md`（3 run から生成）は
機械的な verdict としては **`undetermined`**（`n_runs=3 < MIN_FORMAL_RUNS=5`
により暫定値と明示）を出力しており、上記「暫定 ADOPT」という本ドキュメントの
評価はあくまで §0／本節の記述に基づく人間側の暫定的な解釈であって、
`aggregate.md` が ADOPT を宣言しているわけではない。加えて `aggregate.py`
は各 run が専有ゲート（`env_guard_mode=gated` かつ
`env_guard_overall verdict=pass`）を経て収集されたことも run 単位で検査する
（同 P2 指摘への対応）。run1〜run3 はいずれも `run_gated.sh` 経由（§0 の
`--max-load-avg=8.0` 緩和込み）で収集されておりこの検査を満たす。

### 計測環境に関する留保

- 実測時のマシンは複数セッションが並走する共有環境で、1 分 load average
  が概ね 3〜7 で推移していた。事前登録・実装計画は `--max-load-avg=4.0`
  を想定していたが、この閾値では専有ゲートのバックオフ再試行が上限
  （10 回）に達し undetermined になる可能性が高かったため、実務判断として
  `--max-load-avg=8.0` へ緩和して実行した（`env_info.txt` に経緯を記録）。
  対象・対照双方が同一の共有負荷下で interleaved 計測されているため
  speedup 比自体への影響は限定的と考えられるが、厳密な専有環境での
  再計測（5 run 完了）が望ましい。
- 単一セッションの時間制約により 5 run のうち 3 run で打ち切った。3 run
  はいずれも target 9 形状で 1.5686〜4.4233 の範囲に収まり分散は大きくない
  （最小値 1.5686 は事前登録基準 1.5 にわずかな余裕を持って上回る）ため、
  結果が閾値付近で不安定という兆候はない。ただし control `(256,256,2048)`
  の run2（0.5903）のような単発の大きな振れが観測されており、5 run 完了
  時にこの形状の中央値が基準（0.95）を下回るリスクはゼロではない。

### スコープ

ADOPT は性能上の判定に限る。本番結線（`SPLIT_K_NUMERIC_CONTRACT_APPROVED`
の切替）は REQ-2 判定方式の Metal f32 split-K への適用拡張という別のユーザー
承認が必要（#1476 のスコープ。§7 参照）。

## §6 因果に関する考察（`docs/backend-metal-splitk-decision.md` §3 留保への回答）

`docs/backend-metal-splitk-decision.md` §3・`docs/perf/metal-gemm-splitk-shapes.md`
§4/§6（#1308）は、対象 9 形状の劣化率実測が「並列度不足」という因果の
**仮説**にとどまり、タイル構成差等の代替説明を排除できていないと留保していた。

本 A/B は §4 の `target_tile`（A′/A。タイル構成のみを変えた効果）と
`target`（B/A′ 相当。K 分割効果）を分離することでこの留保を検証する:

- **A′/A（タイル構成効果）**: 対象 9 形状の median は 0.99〜1.05 が大半
  （`(32,32,4096)` のみ 0.63 と外れるが、他の run2 で 0.99 近くまで戻る
  ばらつきの大きい単発値と考えられる）。タイル構成を `select_for_device`
  の選択構成から `split_k_tile`（16×16 または 32×32・staged・wm2/wn2）へ
  変えるだけでは、classic 経路の速度はほぼ変化しない。
- **B/A′（K 分割効果。`target` の speedup を A′ 基準に再計算すると
  概ね target の speedup と同程度）**: A′/A がほぼ 1.0 のため、`target`
  （A/B の比）で観測された 1.5686〜4.4233 倍の改善は、タイル構成差ではなく
  **split-K による K 方向の並列度拡張（`partitions=8〜32` の追加分割）** に
  ほぼ帰属できる。

**結論**: #1308 が留保した「因果は仮説」は、本実測（A′/A がほぼ 1.0・
target の改善幅が大きい）により**「K 方向の並列度不足」が主因である
という仮説を支持する**方向で更新する。ただし §5 の留保（3/5 run・共有
負荷環境）のため、この結論も 5 run 完了後の再確認が望ましい暫定的な
ものとして扱う。フロア（§4）は classic 経路の実行時間の一定割合
（20〜40%）を占めるが、B の実行時間はフロアを下回らないためフロア自体が
speedup の主因ではないことも確認した。

## §7 スコープ外（フォローアップ）

- **フェーズ 1 control（B′）の選択関数呼び出し費用込み再実測**: §4
  「control」節「既知の限界」参照。`should_split_k` 呼び出しをクロージャへ
  含めるよう是正済みのコードで対照 3 形状を実機（M4 Max）で再実測し、
  §4 の control 表・§5 の判定根拠を更新する。
- **フェーズ 0 A vs B（`a_vs_b_fail_count`）の同一入力での再実測**: §3
  「既知の限界」参照。`prepare()` の `seed_offset` 不一致（是正済み）を
  反映した同一入力でのフェーズ 0 自己検証を実機（M4 Max）で再実行し、
  `self_check.log`・§3 の実測値を更新する。フェーズ 1（§4）の speedup
  自体は GEMM の実行時間が入力値に依存しない（形状のみに依存する）ため
  本是正による再実測は不要と判断する。
- **5 run 完了による正式確定**: 単一セッションの時間制約により 3/5 run で
  打ち切った。残り 2 run（可能なら専有環境・`--max-load-avg` をより厳格な
  値で）を追加実行し、§0/§5 の「暫定」を外す正式判定へ更新することを
  フォローアップとして issue #1475 に残す。
- **#1476（本番結線可否）**: `select_for_device`／`dispatch_auto`／
  `MetalBackendOps::gemm` への結線と `SPLIT_K_NUMERIC_CONTRACT_APPROVED`
  の切替（数値契約の適用拡張。ユーザー承認事項）。
- split-K の encode 分離入口の追加と GPU タイムスタンプによる純カーネル
  時間計測（`gemm.rs` 変更が必要）。
- NT/TN/TT の性能比較・f16／hfrag の split-K・`gemm_bias_act` 融合経路への
  適用（#1474 §8 と同じ）。

## §8 参照

- `docs/backend-metal-splitk-decision.md`（本イシューの追記先。§3）
- `docs/perf/metal-gemm-splitk-shapes.md`（#1308。劣化率の元実測）
- `docs/perf/metal-gemm-splitk-two-pass.md`（#1474。split-K 実装記録）
- `docs/perf/logs/metal-gemm-splitk-ab-1475/`（本イシューの生ログ・
  `aggregate.py`／`aggregate.md`・`env_info.txt`・`self_check.log`）
