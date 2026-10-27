# A を pack しない NEON 直接カーネルの設計と判定（決定記録）

イシュー #3241（親 #3240）。`backend-cpu` の m=64 級 GEMM 6 形状（MLP 784→256→10・batch 64 の forward／dW／dX）に対し、A を pack せず行優先のまま NEON で直読みするカーネルを追加するかどうかを決める。本書は docs のみで、`.rs`・`docs/spec`・tolerance・baseline・FMA 契約は変更しない。

## 1. 位置づけと結論

- **判定は (B)**: job 統合と事前 pack までで止め、直接カーネル・ゲート・新規 `unsafe` は追加しない（§6）
- 実装イシュー #3242 は (B) の場合は着手しない（同イシュー本文の定め）。#3242・#3243 の close は Issue ツリーの運用に委ね、本書では閉じない
- §4 の設計は、(A) を将来再開するときの正として残す

## 2. 入力

| 出典 | 使う節 |
|------|--------|
| `docs/perf/cpu-gemm-small-m-partition-ab.md`（#3201） | §1・§3・§4・§6.1・§6.3 |
| `docs/cpu-gemm-small-m-job-merge-prepack-decision.md`（#3198〜#3200） | §5〜§7・§13 |
| `docs/perf/cpu-gemm-m64-jobgrid-diagnosis.md`（#3177） | §2・§3 |
| `docs/perf/lowlayer-diagnosis-2026-09-12.md` | §3 |
| `docs/backend-cpu-gb10-affinity-design.md` | §1 |
| `crates/backend-cpu/src/gemm_blis/microkernel/neon.rs` | 既定 8×12 カーネル |
| `crates/backend-cpu/src/gemm_blis/pack.rs`・`mod.rs`・`prepack.rs` | `pack_a`（p-major）・`m == 1` 分岐・G2 |

## 3. 現状の数字

- 最新の基準は #3201 の値（`cpu-gemm-small-m-partition-ab.md` §3・§6.3）
  - GB10（off）: infer/fresh 174.4 µs・PyTorch 比 0.753（off-aff で 0.958）
  - Mac mini（off）: infer/fresh 208.2 µs・PyTorch 比 0.155
- 2026-09-12 時点の値（`lowlayer-diagnosis-2026-09-12.md` §3 の表。イシュー本文の「T8 pin の infer fresh 0.129」「無 pin T10 の train reuse 2.359」の出所）
  - 単位が食い違っている。lowlayer の表見出しは ms、`backend-cpu-gb10-affinity-design.md` §1（12 行目）は s で書かれる。どちらが正しいかは本書では決めず、他の doc も直さない（スコープ外）
- イシュー本文の「job グリッドは実ログとの突合が未実施」は古い。#3177 §2 で job 数 22／44・A の再 pack 11 倍が再計算と 36/36 件一致している
- 以下は**推定**であり、判定（§6）には使わない: GB10 大コア 10 基の NEON 上限が約 1.2 TFLOPS（推定）・PyTorch の逆算が約 300 GFLOPS（推定）・Mac mini の P コアの NEON ピークが約 560 GFLOPS（推定）

## 4. 設計（(A) を採った場合の形）

### 4.1 読み出しの形

- 対象は A を転置しない呼び出し（Nn・Nt）。Tn（dW。A^T が m 方向に連続）は対象外
- 8 行それぞれ `vld1q_f32(&a[(i0+i)*lda + p])` で k 方向の連続 4 要素を読む。レーンは k のオフセット p..p+3 に対応する（既存カーネルではレーンが行 i に対応する）

### 4.2 FMA の形と bit 一致の根拠

- ステップ q（0..4）で `acc[i][j] = vfmaq_laneq_f32::<q>(acc[i][j], b_{p+q}[j], a_i)` を当てる。オペランドの並び（`acc + b * a[lane]`）は既存の `neon::compute` と同じ
- 各 C 要素は p 昇順の単一 fused FMA の連鎖になり、乗数の値と並びが既存経路と同じなので、NaN を含めて bit 同一になる見込み。k を分割しない・レーン間の縮約をしない（#3198 の決定記録 §7 の (b)〜(e)）も満たす
- 実機での bit 一致確認は、実装する場合の義務とする（本書では未確認）

### 4.3 レジスタ配置

- A 8 本＋B 3 本＋アキュムレータ 24 本で 35 本となり、v0〜v31 を超えてスピルする
- 案: 4 行ずつ 2 回に分ける（A 4 本＋B 3 本＋acc 24 本＝31 本）。4k チャンクごとに B を 2 回読む（L1 から）ため、ロードは packed の既存経路（4 ステップで A 2 本＋B 3 本＝20 ロード）より増える（A 8 本＋B 24 本＝32 ロード）
- どちらが速いかは実機でしか分からない（未計測）

### 4.4 端の処理と REQ-8

- `unsafe` に入る前にスライス長を検査する（`a.len() >= (m-1)*lda + k` を `checked_mul`／`checked_add` で確かめ、`lda >= k`・B panel 長・C タイル境界を見る）。違反は `TileBoundsError` 型で返し panic しない
- m の端（m%8 行）: 行末を越えて読まない。残り行は既存の packed 経路に落とすか、スタック上の小バッファへスカラーでコピーする
- k の端（kc%4）: 4 要素ロードを使わず 1 要素ずつ読む（DUP＋`vfmaq_f32` か lane 0）。#552 に記録された「DUP は丸めに影響しない」同値性に依拠する
- n の端: 既存の端タイル（`ldc = NR` のスタックバッファ）と同じ
- 性能下限を理由に境界検査を省かない（`.claude/rules/coding-rust.md` の REQ-8）

### 4.5 B の出所と分岐位置

- イシュー本文の「事前 pack 済み `[kc][NR]` panel（#3200）」は G2（`RESIDENT_W_PREPACK_ENABLED`）を指す。G2 は #3201 で REJECT され既定 OFF のため、(A) を採るなら「G2 を ON にする」か「呼び出しごとに B を自前で pack する」のどちらかが要る
- 分岐は `gemm_blis_parallel_with_transpose`・`gemm_blis_bias_act_parallel` の `m == 1` 分岐の直後、GB10 affinity／thread cap の包みより手前。既定 OFF の `const` ゲートとし、環境変数では切り替えない

### 4.6 新規 `unsafe` の範囲（(A) の場合）

- 行ポインタからの `vld1q_f32` ロードと C のロード／ストア。各ブロックに SAFETY コメントを付け、security-auditor の監査が要る。(B) なので本書では追加しない

## 5. 採らない案

- 主因が確認できていない現時点で (A) を実装すること
- 環境変数や実行時設定でカーネルを切り替えること
- FMA 契約・tolerance・baseline を緩めること
- B の job 間共有（`docs/cpu-gemm-small-m-job-merge-prepack-decision.md` で REJECT 済み）

## 6. 判定

### 所有者の規則

- 出所: #3241 のコメント <https://github.com/Fandhe-AI/fandhe-ai/issues/3241#issuecomment-6111769113>（2026-10-12 JST）。回答は「全て推奨で」。内容は条件つきで、#3201 の結果で pack が残りの主因と確認できれば (A)、確認できなければ (B)

### 規則の当てはめ（実装側の判断）

1. G1（job 統合＋A の pack 共有）は forward 形状でむしろ遅い。l1_fwd／l2_fwd の比は Mac mini 1.153／1.982、GB10 1.300／1.613（`cpu-gemm-small-m-partition-ab.md` §3.1・§3.2）。§6.1 は原因の切り分けを未実施としている
2. G2（常駐 W の事前 pack）の上乗せは merge+prepack／merge で 0.85〜1.06（同 §1・§6.1）
3. pack と kernel の時間内訳を測った記録はない。#3177 が数えたのは pack の語数と確保回数で、時間ではない

- よって「pack が残りの主因」とは確認できておらず、規則により **(B)**
- 限界: #3201 が測ったのは pack の「共有」（同期のコストを含む）で、「省略」ではない。pack が主因かどうかを肯定も否定もしていない。規則上「確認できなかった」は (B) になる、というのが本判定の意味である
- **承認の範囲**: 所有者が確認したのは条件つきの規則であり、(B) への当てはめは実装側が行った。「所有者が (B) を承認した」ものではない。(A) へ切り替えるには新しい所有者コメントが要る
- (B) のため新規 `unsafe` は増えず、security-auditor の監査対象もない

## 7. 再開の条件（候補。本書では起票しない）

- 6 形状について pack と kernel の時間内訳を事前登録規則で測り、pack が主因と分かった場合に (A) を再検討する。起票は `.claude/rules/out-of-scope-tracking.md` に従い、ユーザー承認を経てから
- #3227 の既定反転判断には影響しない

### 7.1 内訳計測（#3614）

- **状態: 実測済み（2026-10-27。GB10・Mac mini の 2 機体とも。両機体で 5 round 完走・負荷ゲート各 20/20 pass・集計 fail-closed なし）**。判定は下の「判定の結果」
- 事前登録規則: `docs/perf/logs/cpu-pack-kernel-breakdown-3614/RULE.txt`（計測前に固定。6 形状・5 round・負荷ゲート・判定式〈Nn／Nt の A pack の時間加重割合 S_A が 0.50 以上で 5 round 中 4 以上なら PACK_MAIN〉・fail-closed 条件）
- 計測部品: example `cpu_pack_kernel_breakdown`（base 腕）・区間計時の診断パッチ `diag_timing.patch`（複製ツリーにだけ当てる。main には入れない）・`run_breakdown.sh`・`aggregate.py`。手順は同ディレクトリの `README.md`
- 判定の結果（native 腕。`docs/perf/logs/cpu-pack-kernel-breakdown-3614/aggregate.md` の転記。手で直していない）:

  | 機体 | S_A の round 別（1〜5） | S_A 中央値 | S_A >= 0.50 の round | 判定 |
  |---|---|---|---|---|
  | GB10（rayon 20） | 0.1491・0.1482・0.1493・0.1492・0.1497 | 0.1492 | 0/5 | **NOT_PACK_MAIN** |
  | Mac mini（rayon 10） | 0.1742・0.1737・0.1752・0.1730・0.1725 | 0.1737 | 0/5 | **NOT_PACK_MAIN** |

  - RULE §10 の式では、A pack が主因という条件を 2 機体とも満たさない（閾値 0.50 に対し約 0.15〜0.17）。RULE の閾値・tolerance・baseline は変えていない
  - 補助（判定に使わない）: T 全体の s_pack（A＋B pack）中央値は GB10 0.3752・Mac mini 0.3184。(A) が取り除くのは Nn／Nt の A pack だけ（§4.1）なので判定式は sA を使う
  - 形状別 sA／s_pack（native。GB10 / Mac mini）: L1 fwd 0.188/0.215・0.409/0.385、L1 dW 0.067/0.057・0.119/0.134、L1 dX 0.116/0.135・0.357/0.244、L2 fwd 0.186/0.152・0.410/0.472、L2 dW 0.145/0.156・0.304/0.366、L2 dX 0.073/0.097・0.180/0.193。kernel 比率 sK は L1 で 0.42〜0.60、L2 は 0.14〜0.30
  - 注記: L2 dW／L2 dX は区間が 10×tick 未満の項があり個別判定は判定不能（分解能。GB10 tick 16 ns・Mac mini tick 41 ns）。L2 dW／dX の timed/base wall 比は 1.5〜2.1 倍で計時の摂動が大きい（L2 は絶対時間が小さく、sO＝その他が 0.3〜0.7 を占める）。参照 512 角の timed/base は 0.97 前後。L2 の値は参考扱いで、判定は L1 の大きい形状が支配する
  - t1 腕（rayon 1。判定に使わない参考）: S_A 中央値は GB10 0.0330・Mac mini 0.0296。単スレッドでは kernel が支配的
  - 限界は RULE §13（plain GEMM 入口であり `bias_act` 融合経路ではない・L1 dX が本番で実行されるかは未確認・kernel 区間は端タイルのコピーを含む）。RULE §14 のとおり閾値は実装側が登録した値で、所有者が承認した定数ではない
  - 計測条件: `.rev-stamp` は両機体とも SHA `3e17739b0f0795677ce085ba4bdb8ea54db8e774`・ツリー指紋 `74d7c88267db8c531689212248094329b1a14b8fe4334f1ed1bfed0410b20af7`。診断パッチ sha256 `0b748dc0…7549` 一致。実機ロックの保持は GB10 07:47:12Z〜約 07:48:40Z・Mac mini 07:56:50Z〜約 08:01:40Z（ノード時計の UTC）。ログは `docs/perf/logs/cpu-pack-kernel-breakdown-3614/{gb10,macmini}/`（`bin_sha256.txt`・`load_gate.log`・`hwlock.log` を含む）
  - 事実整理（結論ではない）: 事前登録した式では pack 主因とは言えない。一方 L1 fwd・dX の s_pack（A＋B）は 0.24〜0.41 で、sA だけを見る判定式が (A) の効果の上限を表す点は RULE §10 の定義による
- (A) を再検討するかの所有者判断: **未記録（所有者確認待ち）**。本節の追記は所有者の承認を意味しない

## 8. 変更していないもの

- `.rs`・`docs/spec`・tolerance・baseline・FMA 契約（p 昇順）・ruleset・リポジトリ設定
