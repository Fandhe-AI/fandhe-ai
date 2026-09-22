# イシュー #2072 実測記録先（`fit()` の分類 metrics 対応）

## 位置づけ（本 PR 時点では未実測）

本 PR の実行環境（Linux コンテナ／worktree）には CUDA（DGX Spark
GB10）・Metal（Apple Silicon）実機への到達手段がないため、**実測値は
一切含まれていない**。本ディレクトリは実行コマンド・保存すべきログ
一覧・事前登録判定規則のみを提供し、実測は GB10／Mac 実機を持つ
セッションへ申し送る。

## 目的

イシュー #2072「`fit()` の metrics 対応（accuracy・precision 等）」の
受入基準 F（CPU・CUDA・Metal で metrics 計算が bit 同一）を実機で確認
する。metrics 算術自体はホスト側整数カウント＋`f64` 導出でバックエンド
非依存（`docs/compat-metrics-design.md` §6）のため、本記録先の対象は
「forward（logits）が CPU と対象デバイスで一致すれば、そこから求めた
`MetricsResult` も bit 完全一致する」という end-to-end 契約の実機確認
であり、新規カーネルは追加していない。

## 実行コマンド

```bash
# CUDA（DGX Spark GB10 等）
cargo test -p fandhe-ai --release --test compat_sequential_metrics_backend_parity -- --ignored --nocapture cuda_

# Metal（Apple Silicon。macOS 限定でコンパイルされる）
cargo test -p fandhe-ai --release --test compat_sequential_metrics_backend_parity -- --ignored --nocapture metal_
```

## 保存すべきログ

- `compat_sequential_metrics_backend_parity-cuda.log`: 上記 CUDA 実行の
  生出力（`cuda_metrics_result_matches_cpu`）
- `compat_sequential_metrics_backend_parity-metal.log`: 上記 Metal 実行
  の生出力（`metal_metrics_result_matches_cpu`）
- `env_info.txt`: 実行環境記入欄（内部ホスト名は書かない。GPU 型番・
  driver／CUDA バージョン・macOS／Xcode バージョンの要約程度に留める）

## 事前登録判定規則

- **`MetricsResult` は CPU 参照実装（`Sequential::predict`）と対象
  デバイスの `forward` 出力から独立に求めた結果同士で `assert_eq!`
  （**bit 完全一致**）すること**（`assert_parity` の許容誤差付き判定
  ではない。metrics 算術自体がバックエンド非依存のため）。
- **不一致が出た場合の切り分け**: logits の近接タイ（parity レベルの
  forward 差で `Var::argmax` の結果が反転する）が第一の疑い箇所。
  `pred.argmax(Some(1))` はタイを先頭添字で決定的に解決するため、
  CPU・対象デバイスの logits がわずかでも異なれば argmax の選択が
  反転しうる。
- **上記の原因が確認できた場合でも、判定規則・tolerance・baseline は
  変更しない**（`.claude/rules/coding-rust.md`「バックエンド間数値一致
  テストの許容誤差を単独で緩和しない」方針。metrics 側の bit 一致契約
  自体は forward の parity 契約〈REQ-2 統一複合判定〉に従属するもので
  あり、forward 側が既に許容する誤差の範囲内でタイが反転することは
  想定内の事象として記録する——metrics 契約自体を緩める理由にはしない）。
