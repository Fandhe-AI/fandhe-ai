# 依存管理規約（REQ-1 v2）

## 許容依存 12 区分（これ以外の追加はユーザー承認必須）

第 1〜8 区分・第 10〜12 区分は本体 workspace（ルート `Cargo.toml`／`Cargo.lock`）の
直接依存。第 9 区分（ベンチ比較対象）は `scripts/bench/oss-gemm-compare/`
および `scripts/bench/framework-compare/`（適用範囲拡張。下表参照）限定であり
本体 workspace には入らない。本体依存の区分数は第 1〜8・第 10〜12 区分の
合計 11 区分である（許容区分の総数は 12）。CLAUDE.md 等に残る「9 区分」（第 1〜8・第 10 区分）・「10 区分」
（第 11 区分新設前の許容区分の総数）・「11 区分」（第 12 区分新設前の許容区分の総数）の記述は
第 12 区分の新設（2026-10-17）以前の時点の記述であり、
`crates/` 配下のコード・`Cargo.toml` のコメントや過去の決定記録（`docs/`）に残る
「許容依存 8 区分」は第 10 区分の新設（2026-09-28）以前の時点の記述である。
本体 workspace の直接依存の範囲は本表を正とする。

| 区分 | クレート | 条件 |
|------|---------|------|
| CUDA | `cudarc`（`driver`／`nvrtc`／`dynamic-loading`／`cuda-13000`／`f16` feature） | 無条件依存。動的ロード方式（CUDA toolkit 非搭載環境でもビルド成立）。`cuda-13000` は cudarc の CUDA API バージョン feature（指定必須。未指定ではビルド不能）で、DGX Spark GB10 実機の CUDA 13.0 系・PoC-v2-3／PoC-v2-5 実測構成を踏襲した採用。イシュー #412 でユーザー承認済み。ライセンス実測は `docs/license-matrix.md` 4 節を参照 |
| Metal | `objc2`・`objc2-foundation`・`objc2-metal` | `cfg(target_os = "macos")` 限定 |
| 相互運用 | `safetensors` | ワイヤフォーマット処理のみ（テンソルへのマッピングは自作） |
| 相互運用 | `prost` | ONNX の protobuf デコードのみ。`prost-build`（`protoc` ビルド時依存）は使わない（手書き derive。PoC-v2-6） |
| シリアライズ | `serde`・`serde_json` | 構造化データのシリアライズ |
| CPU 並列 | `rayon` | PoC-v2-1 で採用（naive/blocked 比 約 6〜8.5 倍改善） |
| 数値型 | `half` | f16 型 |
| ベンチ | `criterion` | `dev-dependencies` 限定 |
| ベンチ比較対象（フレームワーク横並び） | `candle-core`・`burn`（およびその推移的依存ツリー。禁止リスト掲載クレートの推移的混入を含む） | **`scripts/bench/framework-compare/`（独自の `[workspace]` を持つ独立 Cargo workspace。本体 workspace 外）限定**の第 9 区分の適用範囲拡張。`=x.y.z` 完全固定（`candle-core =0.11.0`・`burn =0.21.0`・`fandhe-ai =0.11.0`〈crates.io 公開版の自社クレート。2026-10-09 公開・v0.11.0 リリースサイクルで `=0.10.0` から更新（2026-10-10 所有者承認）。推移的依存として公開済みの `fandhe-ai-onnx-interop`〈`prost`・`libc`・`safetensors` を含む承認済み区分のみ〉が加わる〉）で、同 workspace の `Cargo.lock` をコミットして再現性を確保する。目的はフレームワーク横並びベンチ（GEMM / MLP 学習 / 推論）の比較対象であり、比較対象という性質上、同 workspace の `Cargo.lock` には依存禁止リストのクレート（`candle-*`・`burn-*`・`cubecl`・`ndarray`・`tch` 等）が**意図的に含まれる**。このため同 workspace の `Cargo.lock` には禁止リスト grep（`check_lock`）を適用せず、代わりに `scripts/check-forbidden-deps.sh lock-all` が**専用の fail-closed 契約検査**（`check_framework_compare`: Cargo.lock の存在・`Cargo.toml` の独自 `[workspace]` 宣言〈本体 workspace への構造的非混入〉・承認済みピン〈burn 0.21.0・candle-core 0.11.0・fandhe-ai 0.11.0〉のドリフト検出（各エントリの `source = "registry+…crates.io-index"` 必須化を含む。path/git 依存への差し替えを fail-closed 検出。イシュー #982）・各メンバー crate の直接依存 allowlist〈承認済み比較対象の完全固定 + bench-common の path 依存のみ。`tch` 等の直接依存追加・非完全固定に加え、`@=` 付き承認済みエントリの `path`/`git`/`registry`/`rev`/`branch`/`tag`/`package` キーによる非 registry 取得元差し替えを fail-closed 検出。イシュー #982〉・セクションヘッダ allowlist〈`[dev-dependencies]`・`[build-dependencies]`・`[target.'cfg'.dependencies]`・`[dependencies.<crate>]` 等の代替依存宣言経路をセクション単位で遮断〉・workspace `members` 宣言の完全一致・配下 Cargo.toml ファイル集合の一致〈未登録 member crate の追加を遮断〉）を毎回実行する。本体 workspace（ルート `Cargo.toml`／`Cargo.lock`）への混入は引き続き禁止で、ルート `Cargo.lock`・`cargo tree` 検査が fail-closed に検出する。専用 `scripts/bench/framework-compare/deny.toml`（advisories / bans / licenses / sources。本 workspace 限定の allow 追加は MPL-2.0・CC0-1.0・BSL-1.0）による依存監査を CI（`ci.yml` の `deps-forbidden` ジョブ）の必須ステップとして実行する。**上記条件（独立 workspace・完全固定・専用契約検査・専用 deny.toml の CI 監査）を満たさない追加・変更、承認済みピン・allow リスト・検査の変更は通常どおりユーザー承認が必要。現行ピンは `fandhe-ai =0.11.0`（ピン更新も都度ユーザー承認）。承認履歴は `docs/framework-compare-harness-decision.md` を正とする。設計判断・ライセンス実測は `docs/framework-compare-harness-decision.md`・`docs/license-matrix.md` 8b 節を出典として参照する** |
| ベンチ比較対象（OSS GEMM） | `matrixmultiply`・`gemm` | `scripts/bench/oss-gemm-compare/`（`[workspace]` を空テーブルで持つ独立 Cargo プロジェクト）限定。`=x.y.z` 完全固定（`matrixmultiply =0.3.11`・`gemm =0.19.0`）。本体 workspace（ルート `Cargo.toml`／`Cargo.lock`）への追加は禁止（依存禁止リスト検査とは別に、この限定はレビューで担保する）。同ハーネスの `Cargo.lock` も依存禁止リスト（`burn` 系一式・`cubecl`・`candle`・`tch`・`ndarray`）の対象とし、`scripts/check-forbidden-deps.sh` の走査対象へ含める。本パッケージ専用の `deny.toml`（`scripts/bench/oss-gemm-compare/deny.toml`。allow リストは本区分と同一方針）による `cargo deny --manifest-path scripts/bench/oss-gemm-compare/Cargo.toml --locked check --config scripts/bench/oss-gemm-compare/deny.toml licenses sources` を CI（`.github/workflows/ci.yml` の `deps-forbidden` ジョブ）へ必須ステップとして組み込む。条件を満たさない追加・変更は通常どおりユーザー承認が必要。2026-08-20 ユーザー承認（イシュー #755）。設計判断・実測記録（ライセンス実測値を含む）は PR #770 で記録される `docs/oss-comparison-harness-decision.md`（イシュー #755）を出典として参照する |
| OS FFI（第 10 区分） | `libc` | `onnx-interop` の external data（イシュー #2347）の安全なファイルオープン（`openat`／`openat2`。`crates/onnx-interop/src/onnx/external_data.rs`）用途に限定（依存の実追加・同ファイルは PR #2348 で導入。本区分の新設は依存追加に先立って本規約へ確定させる）。`[target.'cfg(unix)'.dependencies]` で `cfg(unix)` 限定（Windows 等の非 unix ではリンクされない）。`=0.2.189` 完全固定。手書きの `openat` フラグ定数（`O_DIRECTORY`／`O_NOFOLLOW` 等）は Linux では arch ごとに値が異なり（例: x86 は `O_DIRECTORY=0o200000`・aarch64 は `O_DIRECTORY=0o40000`）、手書き値のまま aarch64 Linux（DGX Spark GB10）でビルドするとシンボリックリンク拒否が機能しない実装バグを生んでいたため、libc の定数・`syscall` 経由の `openat2`（`SYS_openat2`／`open_how`／`RESOLVE_BENEATH` 等）を使う。2026-09-28 ユーザー承認。本体 workspace の直接依存として第 10 区分に位置づける（第 9 区分〈ベンチ比較対象〉とは無関係。既存の第 1〜9 区分の番号・名称は変更しない）。**用途の拡張（B-1。イシュー #3124・#3123 所有者コメント 6111756315 の委任条項）**: `facade` のモデルキャッシュ書き込み（dirfd 系の `openat`／`openat2`／`mkdirat`／`renameat`）へ用途を広げ、`crates/facade/Cargo.toml` の `[target.'cfg(unix)'.dependencies]` に `libc.workspace = true` を置く（新規クレートではなく版・`cfg(unix)` 限定は不変。参照コードと libc 定数への移行は #3125） |
| HTTP クライアント／TLS（第 11 区分） | `ureq`（`default-features = false`・`features = ["native-tls", "vendored"]`） | `=3.4.2` 完全固定。**`cfg(unix)` 限定**（facade〈`ModelRegistry::download`／`download_with` の HTTPS 取得〉と hf-hub〈HF Hub API の JSON 取得。#3393、承認 #3128 コメント 6220036919〉の各 `Cargo.toml` の `[target.'cfg(unix)'.dependencies]` でのみ参照。Windows の依存ツリーには入らない。hf-hub の用途の詳細は末尾の「用途の拡張（#3393）」を参照）。`TlsConfig` は `provider(TlsProvider::NativeTls)` を常に明示し、証明書検証を無効化する設定（`disable_verification`）を facade から出さない。`https_only(true)` を付け、リダイレクトの各ホップを `https` に限る。接続・受信のタイムアウトとサイズ上限を明示する（既定は無期限）。panic 経路（プロバイダ・ルート証明書の feature 不足）を固定設定とテストで閉じる。HTTP crate の型を facade の公開面に出さない（`api_surface.rs` のテストで担保）。`deny.toml` の allow に追加するのは `CDLA-Permissive-2.0`（`webpki-root-certs`）のみ。`vendored` は OpenSSL（Apache-2.0）をソースから静的に同梱するため、OpenSSL の脆弱性は OS の更新では直らず `openssl-src` の版更新（人間承認）でしか直らない点を受容する。Linux ビルドに perl・make が要る（`libssl-dev` は不要。debug で +21〜42 秒）。HF Hub 連携（#3129）が同じ区分を再利用する場合も承認単位は別とする。**用途の拡張（#3393。#3128 コメント 6220036919・2026-10-16 承認）**: `crates/hf-hub` の `[target.'cfg(unix)'.dependencies]` に `ureq.workspace = true` を置き、用途は HF Hub API の JSON 取得（revision 解決・ファイル一覧）に限る。版・feature・固定設定（`NativeTls` 明示・`https_only(true)`・`proxy(None)`・全タイムアウトとサイズ上限の明示・`disable_verification` 非公開・HTTP crate の型を公開面に出さない）の条件は不変。リダイレクトは追従せず 3xx を拒否する（`max_redirects(0)`）。ファイル本体の取得は引き続き facade の `download_with` に委ねる。リダイレクト先ホストの allowlist・2 段目の公開・ほかの依存は別承認。承認根拠: #3123 所有者コメント 6111756315（2026-10-12）の委任条項（dirfd = B-1・`cfg(unix)` 限定・第 11 区分の新設を確定し、クレート・版・feature・配置は #3122 の実測推奨〈`docs/model-download-design.md` §16.8〉で進めてよいとする）に基づき確定（確定記録は同 doc §17、正式反映は #3124・同 §18）。実測は `docs/license-matrix.md` の追加実測（#3124）。2026-10-16 反映 |
| 動的ロード（第 12 区分） | `libloading`（default feature） | `=0.9.0` 完全固定（cudarc 0.19.8 の推移的依存と同じ版。`Cargo.lock` 上で版が重複しない）。**用途は HIP／HIPRTC の動的ロードに限り、利用クレートは `crates/backend-rocm` だけ**。**無条件依存**（cfg なし。cudarc の dynamic-loading と同じ構成で、ビルド時にリンクしないため ROCm 非搭載でもビルドが成立する）。ロードするライブラリ名とシンボル名は固定の `&'static str` に限り、環境変数・設定・外部入力から組み立てない（ファイル名のみを渡しローダーの探索規則に任せる）。**ライブラリハンドルと解決済み関数ポインタは 1 つの所有構造に同居させ、ポインタを構造の外へ出さない（不変条件 U1）**。`Library::new`／`Library::get` は `unsafe` で、SAFETY コメントと security-auditor の監査を必須とする（正式な監査記録は #3158）。版の更新は人間承認で、cudarc と版が並んだ場合は所有者へ戻す。新しい allow 識別子・推移的依存の増加が生じる変更は別承認。承認根拠: #3153 所有者コメント 6111762873（2026-10-12。「`libloading` を新区分の直接依存にする」「版は実装時点の `Cargo.lock` の版を `=x.y.z` で完全固定する」）と `docs/backend-rocm-hip-ffi-decision.md` §5（戻し条件 3 点に該当しない実測）。同 doc §9 の版・feature・`deny.toml` 影響の明示確認は記録されていない（#3154 の PR で人間が判断）。ライセンス実測は `docs/license-matrix.md` の追加実測（#3154）。2026-10-17 反映 |

## 依存禁止リスト（CI で機械検査。TASK-1.2）

- `burn` 系一式（`burn`・`burn-core`・`burn-store`・`burn-onnx`・`burn-import` 等）
- `cubecl`・`candle`・`tch`・`ndarray`

## バージョン固定

- 許容依存は `Cargo.toml` で **`=x.y.z` 完全固定**とし、`Cargo.lock` をコミットする
- 依存の追加・更新は `docs/license-matrix.md` の更新とセットで行い、**AI 自律メンテナンスの自動適用対象外（人間承認必須）**とする（REQ-5）

## ライセンス要件

- 適合基準は MIT OR Apache-2.0 系（`objc2-metal` は Zlib OR Apache-2.0 OR MIT の三重ライセンス）
- MPL-2.0 等コピーレフトの推移的混入は、feature 除外による回避を**推定で記述せず**、有効化しうる feature 組合せごとの `cargo tree` 実測で個別に適合確認する（旧 issue #2 の教訓）
- 新規追加時は境界の判断軸 a〜e（`docs/spec/01-brainstorm.md`）に基づく判断理由を記録する
