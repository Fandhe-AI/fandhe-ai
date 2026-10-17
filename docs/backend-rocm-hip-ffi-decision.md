# HIP FFI の依存方式（`libloading` による動的ロード）の決定記録と承認依頼（#3153）

状態: 方式 B は所有者回答（2026-10-12・イシュー #3153 コメント 6111762873）で確定済み。依存の実追加は #3154 で反映済み（§12）。版（`=0.9.0`）・feature（default）・`deny.toml` への影響は本 doc §5 の実測値で、所有者コメント 6111762873 の委任条項（版は実装時点の `Cargo.lock` の版を `=x.y.z` で固定・実測で戻し条件に該当しなければ #3154 へ進める）の範囲内として反映した。この具体構成への個別の明示確認コメントは存在しない（§9・§12 参照）。
更新履歴: #3154 で §12（反映記録）を追加。#3153 で初版（`docs/rocm-grade-up-conditions-v2-spec-proposal.md` §3c の比較を本 doc へ移して確定）。
実測日: 2026-10-17。toolchain: `rustc 1.98.1`・`cargo 1.98.1`・`cargo-deny 0.19.8`。

## §0 結論

- **方式は案 B**（`libloading` を直接依存にし、実行時に解決する関数ポインタ型宣言で HIP／HIPRTC を呼ぶ）。所有者回答（2026-10-12）で確定済み。
- **区分は第 12 区分**。所有者回答の採番規則（HTTPS 側 #3123 と比べ、先に `deps-policy.md` へ入った側に続けて採番する）に従う。`ureq` が 2026-10-16 に第 11 区分として先に入ったため、HIP の動的ロードは第 12 区分になる。`docs/rocm-grade-up-conditions-v2-spec-proposal.md` に残る「第 11 区分案」は #2614 時点の古い表記である。
- **ピン**: `libloading =0.9.0`（default feature。`Cargo.lock` の現行版。cudarc 0.19.8 の推移的依存と同一）。
- **戻し条件 3 点**（新しい allow 識別子が要る／cudarc 側と版が重複する／推移的依存が増える）は、§5 の実測ではいずれも該当しない。#3154 へ進める。
- 本 doc（#3153 時点）は依存ファイルを変えない。反映は #3154 で行った（§12）。
- 現状の許容依存の数え方: 区分は第 1〜12 の 12 区分（第 9 区分はベンチ比較対象で本体 workspace 外）。第 12 区分が入ると本体直接依存は第 1〜8・第 10〜12 区分の 11 区分になる。

## §1 位置づけ・前提

- ROCm を実装へ進める判断（D-B）の前提のうち、依存方式の部分だけを確定する。spec 側の改訂は #3151、`backend-rocm` の雛形と依存の実追加は #3154、unsafe の監査は #3158 が担う。
- 要件: ROCm を入れていない環境でもビルドが成立する。cudarc の `dynamic-loading` と CI の `build-no-cuda-toolkit` ジョブが同じ性質の先例である。

## §2 3 案の比較

| 比較軸 | A: 新区分でバインディングクレート | **B: `libloading` を直接依存＋実行時解決の関数ポインタ（確定）** | C: 手書き宣言＋`libc::dlopen`／`dlsym` |
|---|---|---|---|
| 1. toolkit 非搭載でビルド成立 | クレート次第（ビルド時リンクの有無は出典未確認） | 成立（ビルド時にリンクしない） | 成立 |
| 2. `Cargo.lock` 差分 | 新規 crate が入る | 新規 crate なし（§5 で実測。直接依存化でも新しい `[[package]]` は利用側メンバー自身のみ） | 新規 crate なし |
| 3. deps-policy への影響 | 新区分を新設 | 第 12 区分を新設 | 第 10 区分の用途条件を拡張 |
| 4. U1（ハンドル寿命内でのみ呼ぶ）の担保 | クレート任せ | ハンドルと関数ポインタを 1 つの所有構造に同居させる不変条件で担保（§4） | `*mut c_void` からの `transmute` で手動。`dlerror` がスレッド安全でない |
| 5. Windows | クレート次第 | `LoadLibrary` を含み拡張可能（§5 で windows-link 経由を確認） | unix に限られる |
| 6. ライセンス | 要調査 | ISC（`deny.toml` の allow に既存） | MIT／Apache-2.0（`libc`） |
| 7. 禁止リストとの衝突 | `cubecl-hip-sys` 等は `scripts/check-forbidden-deps.sh` の `FORBIDDEN_CRATES_ALT`（`cubecl-[a-z0-9-]+`）に当たる | なし | なし |
| 8. 保守性 | 外部クレートの保守状況が出典未確認 | cudarc と同じ機構 | unsafe の面積が最大 |

## §3 採用理由と不採用理由

- **B の採用理由**: ビルド時リンクがない／新規 crate がない／cudarc と同じ機構／Windows へ拡張できる／U1 を所有構造で担保できる。
- **A の不採用理由**: バインディングの有無・ビルド時リンク・保守状況の出典が未確認のままで、新規 crate が増える。禁止リスト（`cubecl` 系）に当たる候補がある。
- **C の不採用理由**: unsafe の面積が最大で、unix に限られ、`dlerror` がスレッド安全でない。

## §4 実装形の契約（#3154・#3158 が従う）

1. `#[link]` 付きの `extern "C"` ブロックは使わない（ビルド時に `libamdhip64` へリンクし、非搭載ビルドを壊すため）。実行時に解決する `unsafe extern "C" fn` の関数ポインタ型を手書きする。
2. **ライブラリハンドル（`libloading::Library`）と解決済み関数ポインタを 1 つの所有構造に同居させる**。ポインタを構造の外へ出さず、呼び出しはその構造のメソッド経由に限る。`Symbol<'lib, T>` のライフタイムは借用中しか守らず、関数ポインタをコピーして保持すれば制約が外れるため、型だけでは U1 を担保できない。
3. dlopen するライブラリ名とシンボル名は**固定の `&'static str`（定数配列）に限る**。環境変数・設定ファイル・外部入力から組み立てない。ファイル名のみを指定してローダーの探索規則に任せ、`LD_LIBRARY_PATH` 等をコード側で連結しない。
4. soname の候補とその並びは、#3154 で ROCm 公式文書に当たって出典つきで確定する（本 doc では推定で書かない。リポジトリ内に出典となる記述はない）。Windows の DLL 名と対応範囲も #3154 で確定する。
5. libloading 0.9 の `Library::new`／`Library::get` は `unsafe` である。SAFETY コメントと security-auditor の監査は #3154・#3158 で必須とする。
6. ロード失敗は型付きエラーまたは probe が false になる形で返す（panic しない・本番経路で `unwrap`／`expect` しない）。macOS など HIP のない環境でも probe が false を返すだけでビルドは通る。
7. 配置: **無条件依存**（cfg を付けない。cudarc の dynamic-loading と同じ構成）。ルート `[workspace.dependencies]` に `=0.9.0` で置き、`crates/backend-rocm` だけが参照する案とする。反映は #3154。

## §5 実測記録（2026-10-17）

### 5.1 現行 lock の状態（読み取りのみ）

| 項目 | コマンド | 結果 |
|---|---|---|
| 版・ライセンス・rust-version | `cargo metadata --locked --offline --format-version 1` | `libloading 0.9.0`（ISC・rust-version 1.88.0）／`cfg-if 1.0.4`（MIT OR Apache-2.0）／`windows-link 0.2.1`（MIT OR Apache-2.0）／`cudarc 0.19.8` |
| 現在の取り込み経路 | `cargo tree --locked -i libloading -e normal,build --target all` | `cudarc v0.19.8` 経由のみ（backend-cuda・bench-harness 等が間接利用） |
| 版の重複 | `cargo tree --locked -d --target all` を `libloading` で絞る | 0 件 |

libloading 0.9.0 の manifest: feature は `default = ["std"]`・`std = []` のみ（`std` は依存を追加しない）。依存は `cfg(unix)` で `cfg-if`、`cfg(windows)` で `windows-link`。`windows-sys` は dev-dependencies のみで取り込まれない。

### 5.2 直接依存化後の擬似実測

リポジトリ外の使い捨てコピー（`origin/main` のアーカイブ）に雛形メンバー `hip-probe-scratch`（`libloading = { version = "=0.9.0" }`）を足して測り、終了後に削除した。リポジトリ内のファイルは変更していない。

| feature | target | 依存ツリー（`cargo tree -p hip-probe-scratch -e normal,build`） |
|---|---|---|
| default | x86_64-unknown-linux-gnu | libloading 0.9.0 → cfg-if 1.0.4 |
| default | aarch64-apple-darwin | libloading 0.9.0 → cfg-if 1.0.4 |
| default | x86_64-pc-windows-msvc | libloading 0.9.0 → windows-link 0.2.1 |
| `default-features = false` | 上記 3 target | default と同一 |

- `Cargo.lock` の差分は scratch メンバー自身の `[[package]]` の追加だけで、他の crate の追加・版変更はない。
- 直接依存化後も `cargo tree -d --target all` の `libloading` は 0 件。
- `cargo deny check licenses bans sources`（scratch メンバーに `license` を付けて実行）: `bans ok, licenses ok, sources ok`。新しい allow は不要で、重複の警告も増えない。advisories はネットワークが要るため本記録では未実施。
- `rust-version 1.88.0` は `rust-toolchain.toml`（stable）および開発コンテナのベース（rust:1.88）と両立する。

### 5.3 戻し条件 3 点の照合

| 戻し条件 | 結果 |
|---|---|
| 新しい allow 識別子が要る | 該当しない（ISC は `deny.toml` の allow に既存。`licenses ok`） |
| cudarc 側と版が重複する | 該当しない（cudarc 0.19.8 の `libloading` も 0.9.0。`-d` で 0 件） |
| 推移的依存が増える | 該当しない（`cfg-if`・`windows-link` は既に lock にある。新規 crate は 0 件） |

## §6 版重複の確認手順（#3154 と cudarc 更新時）

```sh
cargo tree --locked -d --target all | grep -n libloading   # 0 件であること
cargo tree --locked -i libloading -e normal,build --target all
cargo deny check licenses bans sources
```

cudarc が libloading を上げて 2 つの版が並んだ場合は、`=0.9.0` の更新を含めて所有者へ戻す（版の更新は人間承認）。

## §7 license-matrix 追記案（#3154 で反映。本 PR では本体を変更しない）

- §2 の表へ: 区分「動的ロード（第 12 区分）」／`libloading`／`=0.9.0`／ISC／可／判断軸は「ビルド時リンクなし・新規 crate なし・cudarc と同一版」。
- §4 の実測表へ: 本 doc §5.2 のコマンドと結果（default／no-default × 3 target）。

## §8 deps-policy 追記案（#3154 で反映）

第 12 区分の条件文の案: 用途は HIP／HIPRTC の動的ロードに限る／`=0.9.0` 完全固定／ロードするライブラリ名とシンボル名は固定の `&'static str` に限る／ハンドルと関数ポインタの一体所有／unsafe の SAFETY コメントと監査が必須／版の更新は人間承認。

## §9 承認依頼

| 項目 | 内容 |
|---|---|
| クレート名 | `libloading` |
| 版 | `=0.9.0`（`Cargo.lock` の現行版。cudarc と同一） |
| feature | default（`std`。依存を追加しない） |
| ライセンス | ISC（推移的依存の `cfg-if`・`windows-link` は MIT OR Apache-2.0） |
| `deny.toml` への影響 | なし（新しい allow 不要・新規 crate なし。`bans`／`licenses`／`sources` ok） |

方式 B は 2026-10-12 の回答（コメント 6111762873）で確定済み。同コメントは版を「実装時点の `Cargo.lock` の版」とし、実測が戻し条件 3 点に該当しない限り #3154 へ進めてよいと委任している。上記の版・feature・影響は §5 の実測値で、戻し条件に該当しない（§0）ため委任範囲内と判断した。上表の個別値への明示確認コメントは記録されていない（承認を得たとは記述しない）。この判断に異議があれば #3154 の PR で依存追加を外す。

## §10 セキュリティ観点（OWASP）

- A03: dlopen 先とシンボル名は固定の `&'static str` のみ。外部入力でパスを組み立てない。将来の HIPRTC オプションは検証済みの数値と enum だけから作る（#2126 の方針を踏襲）。
- A06: `=0.9.0` で完全固定。更新は人間承認。重複は §6 で検出する。
- A08: 依存は crates.io のみ（`deny.toml [sources]`）。本 PR は依存ファイルを変えず、承認ゲートを迂回しない。
- unsafe: §4 の 2・5 を契約とする。

## §11 出典

- 所有者回答: イシュー #3153 コメント 6111762873（2026-10-12）
- 比較の元表: `docs/rocm-grade-up-conditions-v2-spec-proposal.md` §3c・`docs/rocm-cooperative-hiprtc-unsafe-readiness-decision.md` §6.3
- 区分の採番: `.claude/rules/deps-policy.md` 第 11 区分（`ureq`・2026-10-16）
- ライセンス: `docs/license-matrix.md` §6（`libloading` ISC 適合判断）・`deny.toml` の `[licenses] allow`
- 禁止リスト: `scripts/check-forbidden-deps.sh`（`FORBIDDEN_CRATES_ALT`）
- 実測: §5 のコマンド

## §12 反映記録（#3154）

- 反映内容: `crates/backend-rocm`（`publish = false`）の新設、ルート `[workspace.dependencies]` への `libloading = "=0.9.0"` 追加、`deps-policy.md` 第 12 区分・`docs/license-matrix.md`（2 節と末尾の追加実測）の更新、`deny.toml` はコメントの区分数のみ（allow は不変）。
- 承認根拠: #3153 所有者コメント 6111762873（2026-10-12）の「`libloading` を新区分の直接依存にする」「版は `Cargo.lock` の版を `=x.y.z` で完全固定する」と §5 の実測。**委任範囲の根拠**: 同コメントは版を「実装時点の `Cargo.lock` の版」（`=0.9.0`・cudarc と同一）、feature を実測記録の対象（default は依存を増やさない）とし、戻し条件 3 点（新しい allow 識別子・cudarc との版重複・推移的依存の増加）に該当しなければ #3154 へ進めると定めている。本実装は該当しないため委任範囲内である。**§9 の表の個別値（版・feature・`deny.toml` 影響）への所有者の明示確認コメントは存在しない**。承認を推定で記述せず、PR レビューで人間が最終判断する（異議があれば依存追加を外し足場を依存なしにする）。
- 戻し条件 3 点: 実装時に §6 の確認を再実行し、いずれも該当しなかった（`Cargo.lock` の差分は `fandhe-ai-backend-rocm` の `[[package]]` 追加のみ、`-d` で `libloading` 0 件、`cargo deny` の 4 検査 ok・新しい allow なし）。
- soname 候補（§4-4）: Linux は `libamdhip64.so` のみ採用した。出典は ROCm 公式文書のリンク例 `-lamdhip64`（<https://rocm.docs.amd.com/projects/HIP/en/latest/faq.html>）。版付き soname（`libamdhip64.so.6`／`.so.7` 等）は一次資料で確認できなかったため採用していない。開発用シンボリックリンクを持たないランタイムのみの環境では probe が false になりうる（安全側）。版付き名の追加は出典確認後に別途行う。
- Windows の DLL 名と対応範囲: 一次資料で確定できなかったため候補は空配列で、Windows と macOS の probe は常に false。追跡は別 Issue（起票はユーザー承認後）。
- `hipGetDeviceCount`: シグネチャは `hipError_t hipGetDeviceCount(int* count)`（<https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/device_management.html>）。`hipError_t` の基底型は同文書に明記がないため、雛形ではシンボル解決の成否だけで probe し、呼び出しは #3155 へ送った。
