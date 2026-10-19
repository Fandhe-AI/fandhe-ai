//! 実通信の opt-in テスト（イシュー #3127。`docs/model-download-design.md` §6・§21）。
//!
//! 公開 API `ModelRegistry::download_with` を実ネットワーク・実 TLS（OS ルート証明書）で確認する。
//! CI（GitHub ホステッド）では外部へ通信しないため全件 `#[ignore]`。URL と pin は環境変数だけで
//! 渡し、リポジトリ・CI・README に実値を書かない。失敗メッセージ・出力に URL・pin・環境変数の
//! 値は出さない（A09）。手順はルート README「実通信でのモデル取得確認（opt-in）」節。
//!
//! 環境変数
//! - `FANDHE_AI_DOWNLOAD_TEST_URL`／`FANDHE_AI_DOWNLOAD_TEST_SHA256`: 必須の対。両方未設定なら
//!   skip（`make test-ignored` の `--ignored` 一括実行で落とさない）、片方だけなら設定ミスで panic
//! - `FANDHE_AI_DOWNLOAD_TEST_BAD_TLS_URL`: 任意。証明書が不正なホストの URL（`Err(Network)` を確認）
//!
//! - `FANDHE_AI_DOWNLOAD_TEST_BLACKHOLE_URL`: 任意（#3173）。実ホストへ通信しない到達不能な https URL
//!   （RFC 5737 TEST-NET 等）。接続タイムアウトが `Err(Network)` で返ることと経過時間（観測値のみ）を確認
//!
//! 追加テスト（#3173）: `URL`／`SHA256` の対から pin を 1 桁だけ壊して `HashMismatch` での拒否と
//! 非公開を確認する。GB10・macOS 実機での実行結果は `docs/perf/logs/https-download-3173/`
//! と `docs/model-download-design.md` §22 に記録する。

#![cfg(unix)]

use std::path::PathBuf;

use fandhe_ai::model::{DownloadOptions, ModelError, ModelRegistry, Sha256Pin};

const ENV_URL: &str = "FANDHE_AI_DOWNLOAD_TEST_URL";
const ENV_SHA: &str = "FANDHE_AI_DOWNLOAD_TEST_SHA256";
const ENV_BAD_TLS: &str = "FANDHE_AI_DOWNLOAD_TEST_BAD_TLS_URL";
const ENV_BLACKHOLE: &str = "FANDHE_AI_DOWNLOAD_TEST_BLACKHOLE_URL";

/// 一時キャッシュルート（Drop で削除）。
struct TempRoot(PathBuf);

impl Drop for TempRoot {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// `create_dir`（排他的作成。既存なら失敗）で作る。予測可能な名前への symlink 先回りがあっても
/// 成功せず、Drop の `remove_dir_all` が別の場所を消さない。
fn temp_root() -> TempRoot {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let path = std::env::temp_dir().join(format!("fandhe-dl-real-{}-{nanos}", std::process::id()));
    if let Err(e) = std::fs::create_dir(&path) {
        panic!("一時ディレクトリの排他的作成に失敗しました: {:?}", e.kind());
    }
    TempRoot(path)
}

/// `ModelError` の種別名だけを返す（Display は URL を含みうるため使わない）。
fn error_kind(e: &ModelError) -> String {
    let dbg = format!("{e:?}");
    dbg.split(|c: char| !c.is_alphanumeric() && c != '_')
        .next()
        .unwrap_or("Unknown")
        .to_string()
}

/// 必須の対を読む。両方未設定は `None`（skip）、片方だけは設定ミスとして panic（値は出さない）。
fn required_pair() -> Option<(String, String)> {
    match (std::env::var(ENV_URL).ok(), std::env::var(ENV_SHA).ok()) {
        (Some(u), Some(s)) if !u.is_empty() && !s.is_empty() => Some((u, s)),
        (None, None) => None,
        _ => panic!("{ENV_URL} と {ENV_SHA} は両方設定するか両方未設定にしてください"),
    }
}

#[test]
#[ignore = "実ネットワーク必須（CI では実行しない）。手順は README"]
fn real_network_download_verifies_pin_and_reuses_cache() {
    let Some((url, sha)) = required_pair() else {
        eprintln!("skip: {ENV_URL} と {ENV_SHA} が未設定のため実通信テストを実行しません");
        return;
    };
    let root = temp_root();
    let registry = ModelRegistry::with_cache_dir(&root.0);
    let pin = Sha256Pin::from_hex(&sha).unwrap_or_else(|_| panic!("{ENV_SHA} の形式が不正です"));
    registry
        .download_with(
            &url,
            "live",
            "v1",
            DownloadOptions::default().expected_sha256(pin),
        )
        .unwrap_or_else(|e| panic!("download_with が失敗しました: {}", error_kind(&e)));
    let leaf = root.0.join("live/v1");
    assert!(leaf.join("model.safetensors").is_file());
    assert!(leaf.join("download.json").is_file());
    // 2 回目も成功する（サーバが ETag を返せば条件付き GET の 304 再利用経路）。
    registry
        .download_with(
            &url,
            "live",
            "v1",
            DownloadOptions::default().expected_sha256(pin),
        )
        .unwrap_or_else(|e| panic!("2 回目の download_with が失敗しました: {}", error_kind(&e)));
}

#[test]
#[ignore = "実ネットワーク必須（CI では実行しない）。手順は README"]
fn real_network_bad_tls_certificate_is_rejected() {
    let Some(bad) = std::env::var(ENV_BAD_TLS).ok().filter(|v| !v.is_empty()) else {
        eprintln!("skip: {ENV_BAD_TLS} が未設定のため TLS 検証テストを実行しません");
        return;
    };
    let root = temp_root();
    let registry = ModelRegistry::with_cache_dir(&root.0);
    let r = registry.download(&bad, "badtls", "v1");
    assert!(
        matches!(r, Err(ModelError::Network(_))),
        "証明書不正のホストは Network エラーで拒否される必要があります"
    );
    assert!(!root.0.join("badtls").exists());
}

/// pin の最後の 16 進 1 桁を反転した値で取得し、`HashMismatch` で拒否され何も公開されないことを確認する
/// （A08。issue #3173 の「ハッシュ不一致時の拒否」の実通信確認）。
#[test]
#[ignore = "実ネットワーク必須（CI では実行しない）。手順は README"]
fn real_network_wrong_pin_is_rejected() {
    let Some((url, sha)) = required_pair() else {
        eprintln!("skip: {ENV_URL} と {ENV_SHA} が未設定のため pin 不一致テストを実行しません");
        return;
    };
    let mut wrong = sha.clone();
    let Some(last) = wrong.pop() else {
        panic!("{ENV_SHA} が空です");
    };
    wrong.push(if last == '0' { '1' } else { '0' });
    let pin = Sha256Pin::from_hex(&wrong).unwrap_or_else(|_| panic!("{ENV_SHA} の形式が不正です"));
    let root = temp_root();
    let registry = ModelRegistry::with_cache_dir(&root.0);
    let r = registry.download_with(
        &url,
        "badpin",
        "v1",
        DownloadOptions::default().expected_sha256(pin),
    );
    match &r {
        Err(ModelError::HashMismatch { .. }) => {}
        Err(e) => panic!(
            "HashMismatch を期待しましたが別のエラーでした: {}",
            error_kind(e)
        ),
        Ok(_) => panic!("pin 不一致なのに取得が成功しました"),
    }
    assert!(!root.0.join("badpin/v1/model.safetensors").exists());
}

/// 到達しない宛先への接続が `Err(Network)` で返ることを確認し、経過時間を観測値として出す
/// （時間では assert しない。即時に RST／unreachable が返る環境では接続タイムアウトの実挙動は未確認）。
#[test]
#[ignore = "実ネットワーク必須（CI では実行しない）。手順は README"]
fn real_network_blackhole_connect_times_out() {
    let Some(url) = std::env::var(ENV_BLACKHOLE).ok().filter(|v| !v.is_empty()) else {
        eprintln!("skip: {ENV_BLACKHOLE} が未設定のため接続タイムアウトテストを実行しません");
        return;
    };
    let root = temp_root();
    let registry = ModelRegistry::with_cache_dir(&root.0);
    let started = std::time::Instant::now();
    let r = registry.download(&url, "blackhole", "v1");
    eprintln!("BLACKHOLE_ELAPSED_MS={}", started.elapsed().as_millis());
    assert!(
        matches!(r, Err(ModelError::Network(_))),
        "到達不能な宛先は Network エラーで返る必要があります"
    );
    assert!(!root.0.join("blackhole").exists());
}
