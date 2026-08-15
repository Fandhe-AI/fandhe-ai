//! Metal バックエンドのデバイス列挙・選択（TASK-1.9a・#44）。
//!
//! `tensor-core::device::DeviceProvider` の Metal 実装。`objc2`／
//! `objc2-foundation`／`objc2-metal` は `cfg(target_os = "macos")` 限定の
//! 許容依存であり（`.claude/rules/deps-policy.md`）、本モジュールも同じ
//! cfg でクレート全体（`lib.rs`）から分離する。非 macOS 環境ではこの
//! ファイル自体がコンパイル対象に入らない（`Device::Metal` variant の
//! cfg 境界と整合。TASK-1.9a 実装計画 §3.3）。
//!
//! `MTLCopyAllDevices()`（システム内の全 Metal デバイス列挙）・
//! `MTLDevice::name`／`recommendedMaxWorkingSetSize`／
//! `maxThreadgroupMemoryLength` はいずれも objc2-metal が safe 関数として
//! 提供する。
//!
//! **IOKit FFI に限り最小限の `unsafe` を持つ（イシュー #541）**:
//! `MTLDevice` に GPU コア数を取得する公開 API は存在しないため
//! （#487 codex-review P1 で「機種識別子からの対応表推定」は却下済み。
//! `crate::tile` モジュールドキュメント「occupancy 目標算出」節参照）、
//! [`probe_gpu_core_count`] は IOKit IORegistry の `AGXAccelerator`
//! サービスが公開する `gpu-core-count` プロパティを読む（MFA
//! 〈metal-flash-attention〉／`applegpuinfo` と同方式。機種識別子からの
//! 推定ではなく実機からの実測読み取りのため #487 の却下理由には該当
//! しない）。依存クレートを追加せず（`deps-policy.md` の許容 8 区分を
//! 変更しない）、`extern "C"` の手書き宣言＋`#[link(kind = "framework")]`
//! で `IOKit`／`CoreFoundation` framework を直接リンクする。

use objc2_metal::{MTLCopyAllDevices, MTLDevice};
use tensor_core::device::{BackendError, Device, DeviceInfo, DeviceProvider};

/// IOKit／CoreFoundation の手書き FFI 宣言（[`probe_gpu_core_count`] 専用）。
///
/// `objc2` 系はオブジェクト指向な Objective-C ランタイム API のバインディング
/// であり、C ベースの IOKit／CoreFoundation API とは別物のため流用できない。
/// 本モジュールは `unsafe` を IOKit FFI 境界のみへ局所化する
/// （`.claude/rules/coding-rust.md`・`.claude/rules/security.md`）。
mod iokit_ffi {
    use std::ffi::{c_char, c_void};

    /// `mach_port_t`（IOKit のサービス／ポートハンドルの共通表現）。
    pub(super) type MachPortT = u32;
    /// `io_service_t`（`IOServiceGetMatchingService` の戻り値。実体は
    /// `mach_port_t` の別名）。
    pub(super) type IoServiceT = MachPortT;
    /// `io_object_t`（`IOObjectRelease` が受け取るハンドル。`io_service_t`
    /// と同じ表現）。
    pub(super) type IoObjectT = MachPortT;
    /// `kern_return_t`（IOKit の呼び出し結果コード）。
    pub(super) type KernReturnT = i32;
    /// `IOOptionBits`（`IORegistryEntryCreateCFProperty` のオプション引数。
    /// 本実装では常に 0 を渡す）。
    pub(super) type IoOptionBits = u32;

    /// CoreFoundation の各種オブジェクト参照はすべてオペーク（不透明）な
    /// ポインタ型で表現する（`CFTypeRef`／`CFStringRef`／`CFDictionaryRef`／
    /// `CFAllocatorRef`／`CFNumberRef` はいずれもポインタ 1 個分のレイアウト
    /// のため、FFI 境界では同一の `*mut c_void` エイリアスで扱ってよい。
    /// objc2-foundation 等の型付き束縛を使わない代わりに、呼び出し側
    /// （[`super::probe_gpu_core_count`]）で型 ID 検査を必ず行う）。
    pub(super) type CfTypeRef = *mut c_void;
    pub(super) type CfStringRef = *mut c_void;
    pub(super) type CfDictionaryRef = *mut c_void;
    pub(super) type CfAllocatorRef = *mut c_void;
    /// `CFStringEncoding`（`UInt32`）。
    pub(super) type CfStringEncoding = u32;
    /// `CFNumberType`（`CFIndex` の別名。64bit Darwin では 8 バイト符号付き
    /// 整数）。
    pub(super) type CfNumberType = isize;
    /// `CFTypeID`（`unsigned long`。64bit Darwin では 8 バイト）。
    pub(super) type CfTypeId = usize;
    /// `Boolean`（`unsigned char`）。
    pub(super) type CfBoolean = u8;

    /// UTF-8（`kCFStringEncodingUTF8`）。CoreFoundation ヘッダのマクロ定数
    /// `0x08000100` をそのまま使う（シンボルとして提供されないため）。
    pub(super) const K_CF_STRING_ENCODING_UTF8: CfStringEncoding = 0x0800_0100;
    /// `kCFNumberSInt64Type`（`CFNumberType` 列挙の 4 番目）。
    pub(super) const K_CF_NUMBER_SINT64_TYPE: CfNumberType = 4;
    /// `kIOMainPortDefault`（≡ 旧 `kIOMasterPortDefault`）の実体値は `0`。
    /// シンボル参照だと SDK バージョン間の名称差異（Sonoma 以降
    /// `kIOMainPortDefault`・それ以前 `kIOMasterPortDefault`）に影響される
    /// ため、値が安定している `0` を直接使う（Apple 公式ヘッダ・MFA 等の
    /// 実装が同様に扱う既知の安定値）。
    pub(super) const K_IO_MAIN_PORT_DEFAULT: MachPortT = 0;

    // SAFETY: 各宣言のシグネチャは Apple の `IOKitLib.h`／`IOKitKeys.h`
    // ヘッダの ABI と一致させてある（C の関数ポインタ ABI は Rust の
    // `extern "C"` と同一表現のためリンク時に齟齬なく解決される前提）。
    // 引数・戻り値の所有権契約:
    // - `IOServiceMatching(name)`: `name` は呼び出し中のみ生存すればよい
    //   NUL 終端 C 文字列（借用）。戻り値の `CFDictionaryRef` は呼び出し側
    //   が所有権を得る（CF Create ルール。ヘッダ注釈 `CF_RETURNS_RETAINED`
    //   相当）が、本モジュールでは次段の `IOServiceGetMatchingService` へ
    //   譲渡するため個別に解放しない。
    // - `IOServiceGetMatchingService(main_port, matching)`: `matching` の
    //   所有権を引き取り内部で解放する（Apple ヘッダの明示契約。呼び出し後
    //   に呼び出し側が再度使用・解放してはならない）。戻り値
    //   `io_service_t` はハンドル値（`mach_port_t` の別名）であり借用の
    //   ようなポインタ安全性の懸念はないが、不在時は `0`（`IO_OBJECT_NULL`）
    //   を返しうる。
    // - `IORegistryEntryCreateCFProperty(entry, key, allocator, options)`:
    //   `entry` は有効なハンドル、`key` は非 NULL の `CFStringRef`
    //   （借用）、`allocator` は `NULL` 許容（既定アロケータを意味する）。
    //   戻り値は呼び出し側が所有権を得る（Create ルール。プロパティ不在時
    //   は `NULL`）。
    // - `IOObjectRelease(object)`: `object` は有効なハンドルであることが
    //   前提（無効ハンドルへの呼び出しは未定義動作になりうるため、呼び出し
    //   側が「取得済みかつ未解放」の不変条件を管理する）。
    unsafe extern "C" {
        pub(super) fn IOServiceMatching(name: *const c_char) -> CfDictionaryRef;
        /// `matching` の所有権を引き取り解放する（呼び出し後に
        /// 呼び出し側で再度 `CFRelease` しない）。
        pub(super) fn IOServiceGetMatchingService(
            main_port: MachPortT,
            matching: CfDictionaryRef,
        ) -> IoServiceT;
        pub(super) fn IORegistryEntryCreateCFProperty(
            entry: IoServiceT,
            key: CfStringRef,
            allocator: CfAllocatorRef,
            options: IoOptionBits,
        ) -> CfTypeRef;
        pub(super) fn IOObjectRelease(object: IoObjectT) -> KernReturnT;
    }

    // SAFETY: 各宣言のシグネチャは Apple の `CFString.h`／`CFNumber.h`／
    // `CFBase.h` ヘッダの ABI と一致させてある。引数・戻り値の所有権契約:
    // - `CFStringCreateWithCString(alloc, c_str, encoding)`: `alloc` は
    //   `NULL` 許容（既定アロケータ）。`c_str` は呼び出し中のみ生存すれば
    //   よい NUL 終端 C 文字列（借用）。戻り値は呼び出し側が所有権を得る
    //   （Create ルール。失敗時は `NULL`）。
    // - `CFGetTypeID(cf)`／`CFNumberGetValue(number, ..)`: `cf`／`number`
    //   は非 NULL の有効な `CFTypeRef` であることが前提（呼び出し側が
    //   NULL チェック後にのみ呼ぶ）。`CFNumberGetValue` の
    //   `value_ptr` は `the_type` に応じた書き込み先バイト数
    //   （本モジュールでは `kCFNumberSInt64Type` に対応する `i64` 分）を
    //   確保した有効なメモリを指すことが呼び出し側の責務。
    // - `CFNumberGetTypeID()`: 引数を取らない純関数。
    // - `CFRelease(cf)`: `cf` は Create ルールで取得した未解放の有効な
    //   `CFTypeRef` であることが前提（二重解放・無効ポインタへの呼び出し
    //   は未定義動作になりうるため、呼び出し側が「取得済みかつ未解放」の
    //   不変条件を管理する）。
    #[link(name = "CoreFoundation", kind = "framework")]
    unsafe extern "C" {
        pub(super) fn CFStringCreateWithCString(
            alloc: CfAllocatorRef,
            c_str: *const c_char,
            encoding: CfStringEncoding,
        ) -> CfStringRef;
        pub(super) fn CFGetTypeID(cf: CfTypeRef) -> CfTypeId;
        pub(super) fn CFNumberGetTypeID() -> CfTypeId;
        pub(super) fn CFNumberGetValue(
            number: CfTypeRef,
            the_type: CfNumberType,
            value_ptr: *mut c_void,
        ) -> CfBoolean;
        pub(super) fn CFRelease(cf: CfTypeRef);
    }
}

/// Metal バックエンドの `DeviceProvider` 実装。`Device::Metal` は ordinal
/// を持たない単一 variant のため（`docs/public-api-design.md` §4.1）、
/// `select` は「1 台以上の Metal デバイスが検出できるか」のみを判定する。
/// 複数 GPU を個別に選択する API（ordinal 拡張）は本イシューのスコープ外
/// （§4.1 は `Metal` に ordinal を持たせておらず、拡張する場合は設計書側
/// の変更が必要）。
#[derive(Debug, Default, Clone, Copy)]
pub struct MetalDeviceProvider;

impl MetalDeviceProvider {
    /// 新規 provider を構築する。macOS 上の Metal デバイス検出自体は
    /// `is_available`／`enumerate`／`select` 呼び出し時に行う。
    pub fn new() -> Self {
        Self
    }

    /// `MTLCopyAllDevices()` で検出したデバイスを `DeviceInfo` へ写像する。
    /// システムに Metal デバイスが 1 つも無い場合は空 `Vec` を返す
    /// （fail-safe。`.claude/rules/coding-rust.md`）。`compute_units` は
    /// [`probe_gpu_core_count`]（IOKit 実測。イシュー #541）を記録する。
    /// `AGXAccelerator` サービスは通常システムに 1 つのため、複数デバイス
    /// 列挙時（マルチ GPU 構成）でも同一値を全 `DeviceInfo` に写す
    /// （個々の GPU を区別した取得手段がないため。`docs/perf/
    /// metal-gemm-occupancy-target.md` 参照）。
    fn probe_all() -> Vec<DeviceInfo> {
        let gpu_core_count = probe_gpu_core_count();
        MTLCopyAllDevices()
            .to_vec()
            .into_iter()
            .map(|device| {
                DeviceInfo::new(
                    Device::Metal,
                    device.name().to_string(),
                    Some(device.recommendedMaxWorkingSetSize()),
                    gpu_core_count,
                )
            })
            .collect()
    }
}

/// IOKit IORegistry の `AGXAccelerator` サービスから GPU コア数
/// （`gpu-core-count` プロパティ）を読み取る（イシュー #541。
/// 本ファイル冒頭「IOKit FFI に限り最小限の `unsafe` を持つ」節参照）。
///
/// **fail-safe 契約**: サービス不在（Intel Mac・仮想環境等）・プロパティ
/// 不在・型不一致（`CFNumber` 以外）・値域外（`1..=1024` 外。GPU コア数の
/// 実在しうる範囲を大きく超える値は取得異常とみなす）はいずれも `None`
/// を返す。本番経路で `panic!`／`unwrap()`／`expect()` を使わない
/// （`.claude/rules/coding-rust.md`）。IORegistry から読む値は外部入力
/// として扱い、型検証→値域検証を経てから採用する
/// （OWASP A03・`.claude/rules/security.md`）。
///
/// `crate::device::MetalDeviceProvider::probe_all` の `DeviceInfo::
/// compute_units` へ記録するほか、`crate::tile::OccupancyParams`
/// （occupancy 目標算出。#542 で `select()` に組み込まれる）の入力
/// としても使う想定。
pub fn probe_gpu_core_count() -> Option<u32> {
    use std::ffi::{CString, c_void};

    use iokit_ffi as ffi;

    // 各 unsafe ブロックの前提: いずれも IOKit／CoreFoundation の C API
    // 呼び出しであり、Rust の借用チェッカの保護対象外（生ポインタの
    // ライフタイム・解放責務を呼び出し側が手動管理する）。CF の
    // Create/Copy 系関数（`IOServiceMatching`・`CFStringCreateWithCString`・
    // `IORegistryEntryCreateCFProperty`）が返すオブジェクトはいずれも
    // 呼び出し側が所有権を得る（Create ルール）ため、使用後に必ず
    // `CFRelease`／`IOObjectRelease` する。

    // 1. マッチング辞書を構築する（`IOServiceGetMatchingService` が
    //    所有権を引き取り解放するため、ここでの解放は不要）。
    let service_name = CString::new("AGXAccelerator").ok()?;
    // SAFETY: `service_name` は直前に構築した有効な NUL 終端 C 文字列
    // であり、呼び出し中生存する。戻り値の所有権は次のステップへ譲渡
    // される（`IOServiceGetMatchingService` が解放する）。
    let matching = unsafe { ffi::IOServiceMatching(service_name.as_ptr()) };
    if matching.is_null() {
        return None;
    }

    // 2. マッチするサービス（`AGXAccelerator`）を取得する。
    // SAFETY: `matching` はステップ 1 で得た有効な CFDictionaryRef の
    // 所有権を保持しており、本呼び出しへ譲渡する（呼び出し後に
    // `matching` を再度使用・解放しない）。
    let service =
        unsafe { ffi::IOServiceGetMatchingService(ffi::K_IO_MAIN_PORT_DEFAULT, matching) };
    if service == 0 {
        // サービス不在（Intel Mac・仮想環境等）。fail-safe で None。
        return None;
    }

    // 3. プロパティキー文字列を構築する。
    let key_cstr = CString::new("gpu-core-count").ok()?;
    // SAFETY: `key_cstr` は直前に構築した有効な NUL 終端 C 文字列。
    // 第 1 引数 `null_mut()` は `kCFAllocatorDefault` と等価
    // （CoreFoundation の仕様: アロケータ引数への NULL は既定アロケータ
    // を意味する）。戻り値は本関数が所有権を持ち、以下で確実に解放する。
    let key = unsafe {
        ffi::CFStringCreateWithCString(
            std::ptr::null_mut(),
            key_cstr.as_ptr(),
            ffi::K_CF_STRING_ENCODING_UTF8,
        )
    };
    if key.is_null() {
        // SAFETY: `service` はステップ 2 で取得した有効なハンドル。
        unsafe {
            ffi::IOObjectRelease(service);
        }
        return None;
    }

    // 4. プロパティを取得する。
    // SAFETY: `service`・`key` はいずれも有効なハンドル／オブジェクト。
    // `allocator=null_mut()`（既定アロケータ）・`options=0`（既定動作）。
    // 戻り値（存在すれば）は本関数が所有権を持つ。
    let prop =
        unsafe { ffi::IORegistryEntryCreateCFProperty(service, key, std::ptr::null_mut(), 0) };

    // SAFETY: `key`・`service` は使用済みのため、ここで解放する
    // （Create ルール: 生成側が解放責務を持つ）。
    unsafe {
        ffi::CFRelease(key);
        ffi::IOObjectRelease(service);
    }

    if prop.is_null() {
        return None;
    }

    // 5. 型検証: `CFNumber` 以外は破棄する（外部入力の型検証。OWASP A03）。
    // SAFETY: `prop` はステップ 4 で取得した有効な CFTypeRef。
    let type_id = unsafe { ffi::CFGetTypeID(prop) };
    // SAFETY: 引数を取らない CoreFoundation API 呼び出し。
    let number_type_id = unsafe { ffi::CFNumberGetTypeID() };
    if type_id != number_type_id {
        // SAFETY: `prop` は未解放の有効な所有権付きオブジェクト。
        unsafe {
            ffi::CFRelease(prop);
        }
        return None;
    }

    // 6. 値を取り出す（`kCFNumberSInt64Type` で `i64` として読む。
    //    `gpu-core-count` は非負の小さい整数のため `i64` で十分だが、
    //    `CFNumberGetValue` の失敗（変換不能）も fail-safe に扱う）。
    let mut value: i64 = 0;
    // SAFETY: `prop` は CFNumber であることをステップ 5 で検証済み。
    // `value_ptr` は `i64` を書き込むのに十分な有効なメモリ（スタック
    // 変数へのポインタ）を指す。
    let ok = unsafe {
        ffi::CFNumberGetValue(
            prop,
            ffi::K_CF_NUMBER_SINT64_TYPE,
            &mut value as *mut i64 as *mut c_void,
        )
    };

    // SAFETY: `prop` は使用済みのため解放する。
    unsafe {
        ffi::CFRelease(prop);
    }

    if ok == 0 {
        return None;
    }

    // 7. 値域検証: GPU コア数として妥当な範囲（1〜1024）外は破棄する
    //    （異常値・改竄・将来の想定外ハードウェアへの fail-safe）。
    if !(1..=1024).contains(&value) {
        return None;
    }

    // 範囲を 1..=1024 に限定済みのため `as u32` は非破壊的に成立する。
    Some(value as u32)
}

/// [`MetalDeviceProvider::probe_all`] とは独立に、occupancy 目標算出
/// （`crate::tile::OccupancyParams`）が必要とする実機値をまとめて取得する
/// 入口（イシュー #541）。`crate::tile::actual_groups`／
/// `crate::tile::OccupancyParams::ideal_groups` への `select()` 組み込みは
/// #542 のスコープであり、本構造体は算出に必要な実測値を保持するのみ。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MetalOccupancyInfo {
    /// [`probe_gpu_core_count`] の結果（取得不能なら `None`）。
    pub gpu_core_count: Option<u32>,
    /// `MTLDevice::maxThreadgroupMemoryLength()`（safe API。objc2-metal
    /// 提供）。`crate::tile::OccupancyParams::max_threadgroup_memory_bytes`
    /// へそのまま渡す入力値。
    pub max_threadgroup_memory_bytes: u64,
}

impl MetalOccupancyInfo {
    /// 指定デバイスから occupancy 算出用の実測値をまとめて取得する。
    /// `device` は `MTLCopyAllDevices()`（[`MetalDeviceProvider::probe_all`]
    /// 内部）等で得た `MTLDevice` プロトコルオブジェクトを想定する。
    pub fn probe(device: &objc2::runtime::ProtocolObject<dyn MTLDevice>) -> Self {
        Self {
            gpu_core_count: probe_gpu_core_count(),
            max_threadgroup_memory_bytes: device.maxThreadgroupMemoryLength() as u64,
        }
    }
}

impl DeviceProvider for MetalDeviceProvider {
    fn backend_name(&self) -> &'static str {
        "metal"
    }

    fn is_available(&self) -> bool {
        !Self::probe_all().is_empty()
    }

    fn enumerate(&self) -> Result<Vec<DeviceInfo>, BackendError> {
        Ok(Self::probe_all())
    }

    fn select(&self, device: Device) -> Result<DeviceInfo, BackendError> {
        match device {
            Device::Metal => Self::probe_all().into_iter().next().ok_or_else(|| {
                BackendError::DeviceUnavailable("no Metal device detected".to_string())
            }),
            other => Err(BackendError::DeviceUnavailable(format!(
                "MetalDeviceProvider cannot select {other:?}"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Metal 実機（Apple Silicon）依存のため CI（Linux ホステッド）では
    // 実行しない（`.claude/rules/coding-rust.md`「テスト・ベンチ」節。
    // 実行は Mac 実機セッション〈実機ツリー #408 系〉で行い、実測値を
    // `docs/perf/metal-gemm-occupancy-target.md` の記録表へ転記する）。

    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
    fn probe_gpu_core_count_returns_positive_value_on_apple_silicon() {
        let count = probe_gpu_core_count();
        println!("probe_gpu_core_count() = {count:?}");
        match count {
            Some(n) => assert!(
                n > 0 && n <= 1024,
                "GPU コア数が妥当域外: {n}（値域検証〈1..=1024〉を後段で通過したにも \
                 関わらず 0 以下は矛盾）"
            ),
            None => panic!(
                "Apple Silicon 実機で probe_gpu_core_count() が None を返した \
                 （AGXAccelerator サービス・gpu-core-count プロパティが取得できない \
                 環境変化の可能性。docs/perf/metal-gemm-occupancy-target.md 参照）"
            ),
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
    fn metal_occupancy_info_reports_smem_upper_bound_at_least_32kib() {
        use objc2_metal::MTLCopyAllDevices;

        let devices = MTLCopyAllDevices().to_vec();
        let device = devices
            .first()
            .expect("Metal 実機に 1 台以上のデバイスが検出されるはず");
        let info = MetalOccupancyInfo::probe(device);
        println!("MetalOccupancyInfo = {info:?}");
        assert!(
            info.max_threadgroup_memory_bytes >= 32 * 1024,
            "maxThreadgroupMemoryLength が Apple Silicon の一般的な下限 \
             （32KiB）を下回った: {}",
            info.max_threadgroup_memory_bytes
        );
    }
}
