# m=64 級 GEMM の job 分割・再 pack・確保の回数診断（イシュー #3177）

MLP（784→256→10・batch 64）の CPU 学習・推論で、小形状 GEMM の固定費として疑われる
4 項目（(mc, nc) 2D 動的分配の job 細分化・A の job ごと再 pack・job ごとの `vec!` 確保・
C の job-local コピーイン／アウト）を、実行ログで数えるための診断基盤と結果を記録する。
**診断のみ**で、本番の既定値とコードパスは変えていない。事前登録規則は
`docs/perf/logs/cpu-jobgrid-3177/RULE.txt`。

## 1. 追加したもの

| 物 | 内容 |
|---|---|
| `crates/backend-cpu/src/gemm_blis/jobgrid_probe.rs` | `#[cfg(test)]` 限定の計数プローブ（atomic）。B バッファのアドレスをキーに arm し、他テストの GEMM を数えない |
| `gemm_blis/mod.rs` のフック 6 か所 | すべて文単位の `#[cfg(test)]`。非テストビルドではコンパイルされない |
| `examples/cpu_jobgrid_bench.rs` | cfg(test) なしの計時バイナリ（単スレッド GFLOPS・A/B・直列参照）。出力に checksum を含む |
| CI テスト 3 件 | 再計算（22／44 job ほか）の固定、計数と解析式の一致（Nn／Nt／Tn・Nn は直列と bit 一致）、未 arm 時に何も数えないこと |

確保回数は rayon 内部の確保を含まず、`vec!` の発行 site を解析的に数えた値である
（job 内 3 回 = `c_local`・`b_panel`・`a_panel`。`split_c_into_jobs` 側 = `4 + col_bands + jobs`。内訳は `partition::bands` の行帯・列帯 Vec 各 1、`per_col_band` 外側 1、内側 col_bands、`jobs` 1、job ごとの `c_rows`。
後者は今回新たに分かった事実で、Issue の再計算には無かった）。

## 2. job 数・再 pack の計数（NEON MR=8／NR=12・開発機 M4 Max で動作確認した値）

下表は **正式な記録ではなく参考**（Mac mini／GB10 の実測は未実施。§4）。job 数と pack 語数は
`job_grid` と形状だけで決まる決定的な値で、CI テスト `job_grid_recompute_matches_issue` が
行帯×列帯を固定している。`jpw=2`。

| 形状 | T=10 job（行帯×列帯） | T=20 job | A 再 pack 倍率（T=10／T=20） | B 再 pack 倍率（T=10／T=20） |
|---|---|---|---|---|
| L1 fwd 64×256×784 Nn | 22（2×11） | 44（4×11） | 11.0／11.0 | 2.0／4.0 |
| L1 dW 784×256×64 Tn | 20（10×2） | 40（10×4） | 2.0／4.0 | 10.0／10.0 |
| L1 dX 64×784×256 Nt | 22（1×22） | 44（2×22） | 22.0／22.0 | 1.0／2.0 |
| L2 fwd 64×10×256 Nn | 8（8×1） | 8 | 1.0／1.0 | 8.0／8.0 |
| L2 dW 256×10×64 Tn | 32（32×1） | 32 | 1.0／1.0 | 32.0／32.0 |
| L2 dX 64×256×10 Nt | 22（2×11） | 44（4×11） | 11.0／11.0 | 2.0／4.0 |

- Issue の再計算（L1 fwd の 22／44 job、A を 11 回再 pack、job 内 `vec!` 3 回）と**一致**した。
  job 内確保は全 job で 3 回（job 数×3）、C staging は 1 GEMM あたり `2×m×n` 語。
- T=1 では全形状で job 1 個。再 pack 倍率は 1.0（L1 dX の A のみ 2.0。KC 分割に伴う可能性があるが未調査）。
- L1 dX（64×784×256 Nt）は入力が勾配を要求しない学習では本番で実行されない可能性がある。
  未確認のため、本番での実行有無は結論に使わない。

## 3. 単スレッド実効 GFLOPS と A/B

開発機（M4 Max・非正式・負荷ゲートなし・`RAYON_NUM_THREADS=1`・1 プロセス）の動作確認値は
L1 fwd 46.7／L1 dW 61.5／L1 dX 71.7／L2 fwd 25.3／L2 dW 22.4／L2 dX 33.0／正方 512 113.6 GFLOPS。
**これは動作確認であり記録ではない**。5 run の生ログつき単スレッド GFLOPS と
`jobs_per_worker=1`／job 数上限の A/B 比は、規則どおり Mac mini と GB10 で測る必要がある（§4）。

## 4. 未実施（受け入れ条件の未充足）

- Mac mini・GB10 での 5 run 実測（計数・単スレッド GFLOPS・A/B 4 腕）は、本作業の自動実行環境から
  実機へ到達して行えないため**未実施**。推定値で埋めていない。`RULE.txt` の手順で後続で実施する。
- 参考: #3198（設計と A/B 基盤）・#3227（既定値変更）への入力として、上表の再 pack 倍率
  （L1 fwd の A 11 倍、L2 の B 8〜32 倍）が最初の削減候補になる。採否・既定値の変更はここでは決めない。
