# guardrail / self-repair CLI

自己修復ループが AI 生成の変更を取り込むかどうかを判定する `guardrail`
と、検出・修正生成・検証・取り込み判断の 1 ループを起動する
`self-repair` の CLI コマンド要点です。詳しい引数・JSON スキーマ・終了
コード契約は、リポジトリ内のコマンド仕様
[`docs/guardrail-self-repair-cli.md`](https://github.com/Fandhe-AI/rust-ai-library/blob/main/docs/guardrail-self-repair-cli.md)
を正としてください（本ページはその要点整理です）。

## guardrail

| サブコマンド | 対象 | ポリシー除外リストの適用 |
|---|---|---|
| `guardrail check` | 単一変更セットの 3 分岐判定（自動適用／エスカレーション／却下）。CI・self-repair から呼ばれる本番相当経路 | 適用する |
| `guardrail eval` | ラベル付きデータセットの一括評価。判定器単体（機械判定のみ）の品質保証 | 適用しない |

`guardrail check` の終了コードは判定結果を表します。

| 終了コード | 意味 |
|---|---|
| `0` | 自動適用 |
| `10` | エスカレーション（人間承認待ち） |
| `20` | 却下 |
| `1` | 内部エラー（判定不能） |

`guardrail eval` は見逃し率 0%・誤検知率 30% 以下の 2 条件を評価し、
未達なら終了コード `30` で CI を fail させます。

判定閾値（`guardrail.toml`）・ポリシー除外リスト（`policy-exclusion.toml`）
の変更は必ず人間の承認を経ます。CLI の引数（`--config`／`--preset`）は
コミット済みの設定ファイルのうちどれを使うかを選択する手段であり、
CLI 呼び出しだけで閾値の数値そのものを緩めることはできません。

## self-repair

| サブコマンド | 役割 |
|---|---|
| `self-repair run` | 検出 → 修正候補の検証 → 取り込み判断までの 1 ループを実行する |
| `self-repair verify-log` | JSON Lines ログのハッシュチェーンを検証し、改竄がないことを確認する |

`self-repair run` は `guardrail` をサブプロセスとして起動せず、lib として
直接呼び出します。取り込み判断は必ず guardrail の 3 分岐判定を経由し、
判定を迂回する経路は設けていません。

`self-repair run` の終了コードは guardrail の 3 分岐契約と揃えています。

| 終了コード | 意味 |
|---|---|
| `0` | 自動適用で完走し、検証済み差分の反映も成功 |
| `10` | エスカレーション |
| `20` | 却下 |
| `1` | 内部エラー、または反映処理自体の失敗 |
| `2` | 引数エラー |

候補コード（`--candidates`）の実行にはホスト権限での `cargo build`／
`cargo test`／`cargo clippy` 起動を伴うため、`--allow-candidate-exec` の
明示指定が必須です。加えて環境変数の遮断・書き込み先の隔離といった
OS レベルの縦深防御を既定で適用し、`--isolate-network` 指定時はネット
ワーク namespace 分離も追加できます。

ログ・判定レポートの JSON スキーマ、各サブコマンドの全引数、判定迂回
防止の設計根拠は、上記リンク先の仕様文書を参照してください。
