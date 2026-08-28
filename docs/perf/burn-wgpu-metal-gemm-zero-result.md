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

`#4966` は本イシューと一致する環境・症状（Burn 0.21.0 / macOS / Metal（wgpu 経由）、
M=256 は正常・M=535 以降で出力ゼロ〜異常）で報告されており、メンテナが macOS で
再現・Linux(wgpu) では非再現と回答し、`cubek#283` で修正した経緯が記録されている。

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

`cubek-matmul` の公開版は 0.2.0（2026-05-07）の次が 0.3.0-pre.1（2026-07-29）
であり、0.2.x 系のパッチリリースは存在しない。`cubek#283` の修正は Burn
0.22.0-pre.x 系（`cubek` 0.3.0-pre.x 系）にのみ含まれ、`.claude/rules/deps-policy.md`
第 9 区分の承認済みピン `burn =0.21.0` の範囲では取得できない。ピン更新
（0.22 系以降への更新）は依存追加・更新のユーザー承認事項であり、本 PR の
スコープ外として次回再計測キャンペーン時に扱う（§4 参照）。

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
節のとおり、修正版を含む Burn（0.22 系以降）へのピン更新後に実施する。ピン
更新自体は `.claude/rules/deps-policy.md` 第 9 区分の依存追加・更新に該当し、
人間承認を経てから実施する事項として記録する
（`.claude/rules/out-of-scope-tracking.md` 準拠）。

## 5. 出典

- `scripts/bench/framework-compare/results/raw/results.jsonl`（計測日
  2026-08-28）31〜35 行目
- `scripts/bench/framework-compare/Cargo.lock`（`cubek-matmul` の解決版 0.2.0
  実測）
- `~/.cargo/registry/src/index.crates.io-*/cubek-matmul-0.2.0/src/launch/tune_key.rs`
  （`MatmulGlobalScale::from_size` の 512 境界。ローカル cargo レジストリ
  キャッシュから直接確認）
- `tracel-ai/burn#4966`・`tracel-ai/burn#4907`・`tracel-ai/cubek#283`（`gh
  issue view`/`gh pr view` で 2026-08-28 に実測確認）
- `scripts/bench/framework-compare/bench-burn/Cargo.toml`（feature 構成の実測）
- `scripts/bench/framework-compare/bench-common/src/lib.rs`
  `validate_gemm_checksum`・`scripts/bench/framework-compare/summarize.py`
  `gemm_checksum_reference`/`gemm_checksum_mismatches`（本イシューで追加した
  再発防止ロジック）
