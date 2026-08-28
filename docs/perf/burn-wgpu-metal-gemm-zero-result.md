# Burn(wgpu) Metal GEMM N>=512 全ゼロの原因切り分け（イシュー #965）

## 1. 現象

`scripts/bench/framework-compare/results/raw/results.jsonl`（計測日 2026-08-28。
Apple M4 Max / macOS 26.6.2）の Burn（`burn =0.21.0`。ndarray/wgpu backend）
Metal GEMM 行のうち、N=512/1024/2048/4096 の 4 行が結果テンソル全ゼロを返した:

| N | フレームワーク/デバイス | checksum |
| --- | --- | --- |
| 256 | burn/metal | 237.546660（他フレームワークと 6 桁一致・有効） |
| 512 | burn/metal | **0.000000** |
| 1024 | burn/metal | **0.000000** |
| 2048 | burn/metal | **0.000000** |
| 4096 | burn/metal | **0.000000** |

同一 N・同一入力（`bench-common::Xorshift64Star` の同一シード `SEED_A`/`SEED_B`
生成式）の fandhe-ai・candle・burn/cpu は全 N で checksum が一致する
（burn/cpu は下位桁のみ揺れ、本体の数値一致契約「相対誤差 1e-3 未満 または
絶対誤差 1e-5 未満」〈`.claude/rules/coding-rust.md`〉の範囲内）。壊れているのは
Burn の Metal（wgpu）経路のみであり、N=256 は正常に計算できている（境界は
N=256→512）。詳細は `results/summary.md`「データ有効性の注記（イシュー #965）」・
`results/raw/results.jsonl` 31〜35 行目を参照。

`results/raw/results-dgx.jsonl`（DGX Spark、CUDA）・`results-rtx3060.jsonl`
（RTX 3060、CUDA）には同種の全ゼロ行は無い。Metal（wgpu）経路に固有の現象である。

## 2. 原因: Burn/cubek-matmul の upstream 既知バグ

### 2.1 該当バージョンの実測確認

`scripts/bench/framework-compare/Cargo.lock`（本ハーネス限定の依存 9 区分適用範囲
拡張。`.claude/rules/deps-policy.md`）を実測すると、`burn =0.21.0` の Metal
（wgpu）行列積は `cubek-matmul =0.2.0`（`tracel-ai/cubek` リポジトリ由来。
`burn-wgpu` → `burn-cubecl` → `cubek` → `cubek-matmul` の依存経路）に解決される:

```
$ grep -A1 '^name = "cubek-matmul"' scripts/bench/framework-compare/Cargo.lock
name = "cubek-matmul"
version = "0.2.0"
```

### 2.2 upstream の既知イシュー・修正 PR

`gh` で実測確認済み（2026-08-28 時点）:

| 対象 | 内容 | 状態 |
| --- | --- | --- |
| `tracel-ai/burn#4966` | 「`Tensor::matmul` on wgpu returns near-zero output when M ≥ ~500」。2026-05-17 起票 → 2026-05-21 クローズ | クローズ（修正済み） |
| `tracel-ai/burn#4907` | 「WGPU backend matmul returns all zeros」（macOS M4 Pro / burn 0.21.0-pre.4） | クローズ（同種の既知不具合として認識） |
| `tracel-ai/cubek#283` | 「Fix/matmul/smem bust」。2026-05-20 マージ | マージ済み（`#4966` の修正 PR） |

`#4966` の本文・コメント（`gh issue view 4966 --repo tracel-ai/burn --json body,comments`
で実測確認）は本イシューと一致する環境・症状を報告している: 報告者の実測表は
「M=256（`rel_rms_err` 4.08e-7・正常）→ M=535（`rel_rms_err` 1.00・全ゼロ）」という
境界を示し、環境は「Burn 0.21.0（`burn-wgpu` 0.21.0・既定 cubecl 0.10.0 backend）・
macOS 26.3.1・Apple M1（Metal via wgpu）」。メンテナ（`laggui`）は 2026-05-19 の
コメントで「macOS で再現・wgpu(Linux) では非再現」と報告し、2026-05-21 のコメントで
「`tracel-ai/cubek#283` で修正済み」と述べている。`#4907`（本文実測確認）も同種の
症状（`Tensor::matmul` on wgpu が全ゼロ。環境「MacBook Pro M4 Pro・macOS 26.4.1・
Burn 0.21.0-pre.4」）を報告しており、`#4966` の報告者自身が本文で `#4907` に類似する
旨に言及している。

### 2.3 閾値の実測確認（512 境界の根拠）

`cubek-matmul =0.2.0` のソース（ローカル cargo レジストリキャッシュから直接確認。
`~/.cargo/registry/src/index.crates.io-*/cubek-matmul-0.2.0/src/launch/tune_key.rs`）
に以下の分類ロジックが存在する:

```rust
pub fn from_size(m: usize, n: usize, k: usize) -> Self {
    if m < 512 && k < 512 && n < 512 {
        MatmulGlobalScale::Small
    } else if ... {
        MatmulGlobalScale::Medium
    } else {
        MatmulGlobalScale::Large
    }
}
```

この `m/n/k < 512` の分岐がまさに autotune 候補群（`MatmulGlobalScale` に応じて
異なるカーネル戦略が選ばれる）の切り替え境界であり、本ハーネスで観測した
「N=256 は正常・N=512 から全ゼロ」という境界と完全に一致する。したがって本現象は
Small スケール以外で選択される特定カーネル戦略（`#4966`/`cubek#283` が修正した
smem 破損経路）に起因すると判断する。

### 2.4 本ハーネス構成と upstream 再現条件の一致

`scripts/bench/framework-compare/bench-burn/Cargo.toml` の feature 構成
（`["std", "ndarray", "autodiff"]` + `burn/wgpu`）は、`burn/metal`（MSL
コンパイラ経由の専用バックエンド）ではなく wgpu 既定の経路（macOS では Metal
API を wgpu 経由で叩く。WGSL/naga 変換を経由）であり、`#4966` の再現構成
（Burn 0.21.0・macOS・wgpu 経由の Metal）と一致する。ハーネス側の実装起因ではなく
upstream バグの再現と判断する根拠。

### 2.5 承認済みピンでの修正版入手可否

crates.io の登録メタデータ（index.crates.io 実測）によると `cubek-matmul 0.2.0`
の公開日時（`pubtime`）は 2026-05-07T17:53:00Z であり、`cubek#283`（修正 PR。
`tracel-ai/cubek` へのマージコミット `d713fd4`）のマージ日時は
2026-05-20T12:57:41Z（GitHub API `merged_at` 実測）である。修正 PR のマージは
`cubek-matmul 0.2.0` の公開より約 13 日後であるため、**`cubek-matmul 0.2.0` には
`#4966`/`cubek#283` の修正が含まれていない**と判断できる（公開済みの crates.io
パッケージは再公開されない限り内容が変わらないため、公開日時が修正マージより
前であれば当該修正を含み得ない）。したがって `scripts/bench/framework-compare/
Cargo.lock`（§2.1）が実測どおり `cubek-matmul =0.2.0` に解決される本ハーネスの
現行ピン構成では、本現象（Metal wgpu 経路 N≥512 GEMM 全ゼロ）は再現し続ける
はずであり、修正取り込みには `cubek-matmul`（推移的には `burn`）の後続版への
更新が必要になる。`.claude/rules/deps-policy.md` 第 9 区分の承認済みピンは
`burn =0.21.0`（→ `cubek-matmul =0.2.0`）に固定済みであり、ピンを動かす（`burn`
の別バージョンへ更新する）こと自体が依存追加・更新のユーザー承認事項に該当する。
よって本 PR ではピン更新は行わず、承認された手順でのピン更新を前提とした再計測を
§4 の追試事項として記録するに留める。

upstream `#4966` は既に該当事象で解決済みのイシューであるため、新規の upstream
起票は行わない（自動運転中は外部リポジトリへの書き込みを行わない方針とも整合）。

## 3. ハーネス側の再発防止（イシュー #965 受け入れ条件 3）

同種の不具合（フレームワーク側の内部バグによる壊れた計算結果）を将来検知
できるよう、2 段構えの防御を追加した:

1. **bench-common の縮退 checksum ガード**（`bench-common/src/lib.rs`
   `validate_gemm_checksum`）: 各バイナリ（`bench-fandhe`/`bench-candle`/
   `bench-burn`）の `run_gemm`（fandhe は reuse 経路含む）が `Record::emit`
   直前に呼ぶ。checksum が `0.0` または非有限（NaN/inf）なら
   `MEASURE_ERROR: gemm checksum is degenerate (…)` として JSONL への記録を
   拒否する（xorshift64* の一様乱数入力で総和が厳密に 0.0 になる確率は
   無視できるため、`== 0.0` を縮退の強いシグナルとして扱う）
2. **summarize.py の checksum 相互突合**（`gemm_checksum_reference` /
   `gemm_checksum_mismatches`）: 同一 size の gemm checksum を全フレームワーク
   間で突合し、本体の数値一致契約と同一の複合判定（相対誤差 1e-3 未満 または
   絶対誤差 1e-5 未満）から外れる行を Markdown 表で「無効」表示する
   （`--strict` で終了コード 2）

(1) は次回計測時に同種の不具合が発生した場合に該当行が JSONL へ書き込まれる
のを未然に防ぐ（`skipped.log` に理由付きで記録される）。(2) は既にコミット済み
の JSONL・将来 (1) を経ずに手動投入されたデータに対しても機械的に検知できる
ようにする、独立した第 2 の防御層である。

## 4. 未検証事項・追試手順（Mac 実機セッション待ち）

本ドキュメントは Linux worktree（このエージェント実行環境）で執筆しており、
Mac 実機が到達不能なため以下は未計測（結果欄「未計測」。数値を捏造しない）:

- `cargo build -p bench-burn --release --features burn/autotune-checks` で
  ビルドし `--task gemm --device metal --size 512` を実行した場合に、autotune
  候補間の出力不一致 assert が発火するか: `--未計測--`
- `bench-burn/Cargo.toml` の feature を `burn/metal`（MSL コンパイラ経由の
  専用 Metal バックエンド。wgpu を経由しない）に切り替えたビルドで同事象が
  再現するか: `--未計測--`（`Cargo.toml` の変更を伴うため、実施する場合は
  本ハーネスの依存契約〈`check_framework_compare`〉に抵触しない一時的な
  診断ビルドとして扱う）
- 環境変数 `CUBECL_AUTOTUNE_LEVEL` の変更、または autotune キャッシュ
  （`$HOME/.cache/` 配下）の削除で挙動が変わるか: `--未計測--`

再計測（正式な是正）については `results/summary.md`「データ有効性の注記」
節のとおり、`cubek#283` の修正を含むバージョンへの `burn` ピン更新（§2.5 参照）を
経て実施する。crates.io の登録メタデータ（index.crates.io 実測。2026-08-28
時点）を確認したところ、`cubek-matmul` に `0.2.0` 直後の \`0.2.x\` パッチ版は
存在せず、次の版は `cubecl =0.11.0-pre.*` を要求する `0.3.0-pre.1`
（2026-07-29 公開）以降のプレリリース版のみである。対応する `burn` 側も
同様に、`0.21.0`（2026-05-07 公開。現行ピン）の次の安定版 `0.22.0` は
2026-08-28 時点で未公開（プレリリース `0.22.0-pre.1`〜`0.22.0-pre.3` のみ存在。
最新 `0.22.0-pre.3` は 2026-08-25 公開）である。したがって**現時点では
修正を含む安定版 `burn` が存在しない**（プレリリース版のみ）ため、ピン更新を
実施する場合は「プレリリース版への更新を承認するか、安定版 `0.22.0` の公開を
待つか」を含めてユーザー判断を仰ぐ必要がある。ピン更新自体は
`.claude/rules/deps-policy.md` 第 9 区分の依存追加・更新に該当し、人間承認を
経てから実施する事項として記録する（`.claude/rules/out-of-scope-tracking.md`
準拠）。

## 5. 出典

- `scripts/bench/framework-compare/results/raw/results.jsonl`（計測日
  2026-08-28）31〜35 行目
- `scripts/bench/framework-compare/Cargo.lock`（`cubek-matmul` の解決版 0.2.0
  実測）
- `~/.cargo/registry/src/index.crates.io-*/cubek-matmul-0.2.0/src/launch/tune_key.rs`
  （`MatmulGlobalScale::from_size` の 512 境界。ローカル cargo レジストリ
  キャッシュから直接確認）
- `tracel-ai/burn#4966`（`gh issue view 4966 --repo tracel-ai/burn --json
  body,comments,state,createdAt,closedAt` で本文・全コメントを 2026-08-28 に
  実測確認）・`tracel-ai/burn#4907`（同 `--json body` で本文実測確認）・
  `tracel-ai/cubek#283`（`gh pr view 283 --repo tracel-ai/cubek --json
  number,title,state,mergedAt` でタイトル・マージ日時のみ実測確認。本文は
  未確認）
- `https://index.crates.io/cu/be/cubek-matmul`・`https://index.crates.io/bu/rn/burn`
  （crates.io 公式インデックスを 2026-08-28 に直接取得し、`cubek-matmul`・
  `burn` の全公開版と `pubtime` を実測。§2.5・§4 の版数・日時根拠）
- `scripts/bench/framework-compare/bench-burn/Cargo.toml`（feature 構成の実測）
- `scripts/bench/framework-compare/bench-common/src/lib.rs`
  `validate_gemm_checksum`・`scripts/bench/framework-compare/summarize.py`
  `gemm_checksum_reference`/`gemm_checksum_mismatches`（本イシューで追加した
  再発防止ロジック）
