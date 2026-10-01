#!/usr/bin/env python3
"""イシュー #2122 sm_121 ISA プローブの実測ログ集計（python3 標準ライブラリのみ）。

判定規則の正は同ディレクトリの RULE.txt（実測前に固定）。本ファイルは RULE.txt の
機械可読行（CLAUSE／STAGE／STATUS／VERDICT／INDETERMINATE／G0REASON／TARGET／
DEVTARGET／PROBE／PROCESS）を読み、次の入力から判定表を作る。

入力（`orchestrate.sh` の生成物。`--log-dir` 配下）:
  - env_info.txt                  provenance（mode・git_head・git_clean・時刻）
  - compile.log                   S1／S2／home／hopper（`SM121_PROBE_JSON` 行＋proc 行）
  - exec/<probe>@<target>.log     1 (プローブ, target) = 1 プロセスの全セル
  - legacy-<name>.log             既存 setmaxnreg_probe_* の再実行（R-LEGACY）
  - device_attributes_dump.log    記録のみ（存在とマスクだけ検査する）

fail-closed（RULE.txt 13）: 重複・未知・壊れた JSON・NaN・浮動小数点・巨大な整数・
キー集合の過不足・連鎖の矛盾・完了記録の欠落・未マスクのパスを検出したら
`LogIntegrityError` を送出し、正式な集計を一切出力せず exit 2。前提ゲート G0 が
不成立なら全セルを判定不能として表を stdout に出し exit 3（`--out` では書かない）。
ログが全く無い場合だけ「未実測」として exit 0。

検出範囲（RULE.txt 0）: 上記の入力に現れたレコードの完全性と、そこから導く判定に限る。
ログが実際にその実行から生成されたことの証明までは保証しない。
`--self-test` は固定 fixture による自己検証（実機不要。CI では走らせない）。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
RULE_PATH = HERE / "RULE.txt"
CLAIMS_PATH = HERE / "guide_claims.tsv"
ORCHESTRATE_PATH = HERE / "orchestrate.sh"

PREFIX_JSON = "SM121_PROBE_JSON "
PREFIX_PROC = "SM121_PROBE_PROC "
LEGACY_PREFIX = "SETMAXNREG_PROBE_RESULT "

# 以下の閉じた集合は RULE.txt の機械可読行と一致しなければならない
# （`check_rule_consistency` が実行のたびに検査する）。
STAGES = ("ctl", "nvrtc_ptx", "nvrtc_cubin", "module_load", "launch", "sync", "verify")
STATUSES = ("ok", "rejected", "error", "mismatch", "timeout", "process_failed", "not_run",
            "unavailable", "not_applicable_by_design")
CLAUSES = ("G0", "R-STAGE", "R-CTL", "R-HOME", "R-TC5", "R-GUIDE", "R-MMA", "R-CLU", "R-SNR",
           "R-HOPPER", "R-TMA-BASE", "R-TMA-SEM", "R-TMA-XFER", "R-LEGACY", "R-COMMON")
V_OK = "成立"
V_NVRTC = "NVRTC 拒否"
V_PTXAS = "ptxas 拒否（オフライン）"
V_LOAD = "ロード失敗"
V_RUNTIME = "実行時エラー"
V_MISMATCH = "結果不一致"
V_ACCEPT_ONLY = "受理のみ（実行意味論は未検証）"
V_INDET = "判定不能"
V_UNMEASURED = "未実測"
VERDICTS = (V_OK, V_NVRTC, V_PTXAS, V_LOAD, V_RUNTIME, V_MISMATCH, V_ACCEPT_ONLY, V_INDET,
            V_UNMEASURED)
REFUSED = (V_NVRTC, V_PTXAS, V_LOAD, V_RUNTIME)
G_MATCH = "一致（12.0 の記述を外挿）"
G_DIFF = "不一致"
GUIDE_WORDS = (G_MATCH, G_DIFF, "判定不能")
INDET_CODES = ("G0_FAILED", "TARGET_UNSUPPORTED", "CTL_FAILED", "HOME_REJECTED", "HOME_NOT_OK",
               "COMPILE_EXEC_DISAGREE", "OFFLINE_JIT_DISAGREE", "NVRTC_UNAVAILABLE", "NO_DEVICE",
               "UNSUPPORTED_PTX_VERSION", "STAGE_ERROR", "TIMEOUT", "PROCESS_FAILED",
               "LAYOUT_UNVERIFIED", "UNEXPECTED_ACCEPT", "LEGACY_CONTRADICTION", "LEGACY_INCONCLUSIVE",
               "HOPPER_HOME_DISAGREE", "TMA_BASE_NOT_ESTABLISHED", "CLU_DIMS2_NOT_ESTABLISHED", "GUIDE_NO_PROBE", "GUIDE_UNMEASURED")
G0_REASONS = ("G0_PROVENANCE", "G0_DIRTY_TREE", "G0_DEV_MODE", "G0_DEV_TARGET", "G0_NVRTC_MISSING",
              "G0_CC", "G0_NO_DEVICE")
POLICIES = ("verify", "accept_only", "record_only", "attr")
LAYOUTS = ("none", "verified", "unverified")
EXPECTS = ("none", "reject121")
HOPPER_ARCH = "sm_90a"
# 既存プローブの再実行（R-LEGACY）。name -> (対応する新プローブ, 固定の対象 target〈None はログから決める〉, 種別)。
# TMA の既存テストは 1 テストずつ別プロセスで実行する（名前は <テストバイナリ>@<テスト関数>）。
LEGACY_NAMES = {
    "setmaxnreg_probe_dec_base_real_device": ("snr.dec", "compute_121", "snr"),
    "setmaxnreg_probe_dec_accel_real_device": ("snr.dec", "compute_121a", "snr"),
    "setmaxnreg_probe_incdec_base_real_device": ("snr.incdec", "compute_121", "snr"),
    "setmaxnreg_probe_incdec_accel_real_device": ("snr.incdec", "compute_121a", "snr"),
    "tma_probe_real_device@tma_nvrtc_compile_probe": (None, None, "tma_compile"),
    "tma_probe_real_device@tma_execution_probe": ("tma.base_cluster", None, "tma_exec_cluster"),
    "tma_probe_real_device@tma_execution_probe_cta": ("tma.base_cta", None, "tma_exec_cta"),
}
# 依存（事前登録。RULE.txt 10b）: 自身が 成立／受理のみ になるプローブは、依存先が同 target で 成立 のときだけ採る。
TMA_BASE = "tma.base_cta"
TMA_DEPENDENT_EXCLUDE = ("tma.base_cta", "tma.base_cluster")
INT_LIMIT = 2 ** 63
UNMASKED_RE = re.compile(r"/home/[A-Za-z0-9_.-]+")
CODE_RE = re.compile(r"^(-|[A-Z0-9_]+)$")
ARCH_RE = re.compile(r"^(-|(compute|sm)_[0-9]+[af]?)$")
CC_RE = re.compile(r"^([0-9]+\.[0-9]+|none)$")
NVRTC_RE = re.compile(r"^([0-9]+\.[0-9]+|unavailable)$")
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
UTC_RE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$")


class LogIntegrityError(Exception):
    """完全性検査の失敗（黙って除外せず非ゼロ終了する。RULE.txt 13）。"""


def register_once(store: dict, key, value, where: str) -> None:
    """キー付きレコードの唯一の登録口。既出キーは値が同一でも拒否する。"""
    if key in store:
        raise LogIntegrityError(f"{where}: レコードが重複: {key!r}")
    store[key] = value


def require_exact_keys(got, want, where: str) -> None:
    """観測キー集合が期待集合と過不足なく一致することを要求する（欠落・余剰・未知を拒否）。"""
    got_s, want_s = set(got), set(want)
    if got_s != want_s:
        raise LogIntegrityError(
            f"{where}: キー集合が期待と不一致（欠落 {sorted(want_s - got_s)}・余剰 {sorted(got_s - want_s)}）")


def check_masked(text: str, name: str) -> None:
    m = UNMASKED_RE.search(text)
    if m:
        raise LogIntegrityError(f"{name}: 未マスクの絶対パス {m.group(0)!r}（RULE.txt 13）")


def parse_kv(tokens, where: str, strict: bool = False) -> dict:
    """`k=v` 形式のトークン列を辞書にする。重複キーは値が同一でも拒否する（register_once）。
    strict=True なら `k=v` でないトークンも拒否する。"""
    kv: dict = {}
    for tok in tokens:
        k, sep, v = tok.partition("=")
        if not sep or not k:
            if strict:
                raise LogIntegrityError(f"{where}: k=v 形式でないトークン: {tok!r}")
            continue
        register_once(kv, k, v, where)
    return kv


def expected_arch(rule: "Rule", pdef: "ProbeDef", target: str, stage: str) -> str:
    """exec のセルが記録すべき arch（runner.rs / sm121_isa_probe_exec_real_device.rs の規則）。
    ctl は target、S2 は実アーキ、それ以外は仮想アーキ。tc5.cross など固定アーキのプローブは
    S1〜S6 が固定アーキ（FIXEDARCH 行）になる。"""
    fixed = rule.fixed_arch.get(pdef.id)
    if stage == "ctl":
        return target
    if stage == "nvrtc_cubin":
        return fixed[1] if fixed else (rule.targets.get(target) or rule.dev_targets[target])
    return fixed[0] if fixed else target


# ------------------------------------------------------------------ RULE.txt

@dataclass(frozen=True)
class ProbeDef:
    id: str
    ac: str
    clause: str
    home: str
    policy: str
    layout: str
    expect: str

    @property
    def is_attr(self) -> bool:
        return self.policy == "attr"


@dataclass
class Rule:
    lines: dict
    probes: dict
    targets: dict
    dev_targets: dict
    processes: list = field(default_factory=list)
    fixed_arch: dict = field(default_factory=dict)


def parse_rule(text: str) -> Rule:
    """RULE.txt の機械可読行を読む。未知の KEY や書式違反は LogIntegrityError。"""
    keys = ("CLAUSE", "STAGE", "STATUS", "VERDICT", "GUIDE", "INDETERMINATE", "G0REASON",
            "TARGET", "DEVTARGET", "PROBE", "PROCESS", "FIXEDARCH")
    lines: dict = {k: [] for k in keys}
    for raw in text.splitlines():
        m = re.match(r"^([A-Z0-9]+): (.*)$", raw)
        if not m or m.group(1) not in keys:
            continue
        lines[m.group(1)].append(m.group(2).rstrip())
    probes: dict = {}
    for v in lines["PROBE"]:
        parts = v.split(" ")
        pid, kvs = parts[0], parts[1:]
        fields = {}
        for kv in kvs:
            k, sep, val = kv.partition("=")
            if not sep:
                raise LogIntegrityError(f"RULE.txt: PROBE 行の項目が k=v 形式でない: {v!r}")
            register_once(fields, k, val, f"RULE.txt PROBE {pid}")
        require_exact_keys(fields, ("ac", "clause", "home", "policy", "layout", "expect"),
                           f"RULE.txt PROBE {pid}")
        if fields["policy"] not in POLICIES or fields["layout"] not in LAYOUTS \
                or fields["expect"] not in EXPECTS or fields["clause"] not in CLAUSES:
            raise LogIntegrityError(f"RULE.txt: PROBE 行の値が未知: {v!r}")
        register_once(probes, pid, ProbeDef(pid, fields["ac"], fields["clause"], fields["home"],
                                            fields["policy"], fields["layout"], fields["expect"]),
                      "RULE.txt PROBE")
    targets, dev = {}, {}
    for key, store in (("TARGET", targets), ("DEVTARGET", dev)):
        for v in lines[key]:
            name, real = v.split(" ")
            register_once(store, name, real, f"RULE.txt {key}")
    fixed: dict = {}
    for v in lines["FIXEDARCH"]:
        pid, virt, real = v.split(" ")
        if pid not in probes:
            raise LogIntegrityError(f"RULE.txt: FIXEDARCH が未知のプローブ: {pid}")
        register_once(fixed, pid, (virt, real), "RULE.txt FIXEDARCH")
    for v in lines["PROCESS"]:
        process_timeout(v)  # 書式検査（timeout= の欠落・不正は LogIntegrityError）
    return Rule(lines, probes, targets, dev, lines["PROCESS"], fixed)


def load_rule() -> Rule:
    return parse_rule(RULE_PATH.read_text(encoding="utf-8"))


def check_rule_consistency(rule: Rule) -> None:
    """aggregate.py の閉じた集合が RULE.txt の機械可読行と一致することを要求する。"""
    pairs = (("CLAUSE", CLAUSES), ("STAGE", STAGES), ("STATUS", STATUSES), ("VERDICT", VERDICTS),
             ("GUIDE", GUIDE_WORDS), ("INDETERMINATE", INDET_CODES), ("G0REASON", G0_REASONS))
    for key, mine in pairs:
        if tuple(rule.lines[key]) != tuple(mine):
            raise LogIntegrityError(
                f"RULE.txt の {key} 行と aggregate.py が不一致: RULE={rule.lines[key]} 実装={list(mine)}")
    if not rule.probes or not rule.targets or not rule.dev_targets:
        raise LogIntegrityError("RULE.txt に PROBE／TARGET／DEVTARGET 行が無い")


def process_timeout(line: str) -> int | None:
    """PROCESS 行末尾の `timeout=<秒>`。env_info のみ timeout を持たない（None）。"""
    if line == "env_info":
        return None
    m = re.search(r" timeout=([0-9]+)$", line)
    if not m or int(m.group(1)) <= 0:
        raise LogIntegrityError(f"RULE.txt: PROCESS 行に正の timeout= が無い: {line!r}")
    return int(m.group(1))


def expand_processes(rule: Rule) -> list:
    """PROCESS 行を (PROBE × 正式 TARGET) まで展開した起動プロセス名の一覧にする
    （`timeout=<秒>` を含む。orchestrate.sh --dry-run の出力と一致しなければならない）。"""
    out = []
    for p in rule.processes:
        if p.startswith("exec_matrix "):
            tmo = p[len("exec_matrix "):]
            for pid in rule.probes:
                for t in rule.targets:
                    out.append(f"exec {pid} {t} {tmo}")
        else:
            out.append(p)
    return out


def stage_is_na(policy: str, stage: str) -> bool:
    """方針で設計上不実施の段（`types.rs::Policy::stage_is_na` と同じ表）。"""
    if policy == "accept_only":
        return stage in ("launch", "sync", "verify")
    if policy == "record_only":
        return stage == "verify"
    if policy == "attr":
        return stage in ("nvrtc_ptx", "nvrtc_cubin", "module_load", "launch", "sync")
    return False


# ------------------------------------------------------------------ JSON 読み取り

def _reject_constant(name):
    raise ValueError(f"NaN／Infinity は許容しない: {name}")


def _reject_float(text):
    raise ValueError(f"浮動小数点は許容しない: {text}")


def _bounded_int(text):
    v = int(text)
    if abs(v) >= INT_LIMIT:
        raise ValueError(f"整数が 64 bit を超える: {text}")
    return v


def _no_dup_pairs(pairs):
    d = {}
    for k, v in pairs:
        if k in d:
            raise ValueError(f"JSON オブジェクト内のキー重複: {k}")
        d[k] = v
    return d


def strict_json(text: str, where: str) -> dict:
    try:
        rec = json.loads(text, parse_constant=_reject_constant, parse_float=_reject_float,
                         parse_int=_bounded_int, object_pairs_hook=_no_dup_pairs)
    except (ValueError, RecursionError) as exc:
        raise LogIntegrityError(f"{where}: 壊れた／不正な JSON: {exc}: {text[:200]!r}") from exc
    if not isinstance(rec, dict):
        raise LogIntegrityError(f"{where}: レコードがオブジェクトでない")
    return rec


SCHEMAS = {
    "cell": {"v": int, "kind": str, "phase": str, "probe": str, "target": str, "arch": str,
             "stage": str, "status": str, "code": str, "detail": str},
    "env": {"v": int, "kind": str, "phase": str, "probe": str, "target": str, "device": str,
            "cc": str, "nvrtc": str},
    "done": {"v": int, "kind": str, "phase": str, "probe": str, "target": str, "cells": int,
             "mismatch": int},
    "proc": {"v": int, "kind": str, "phase": str, "probe": str, "target": str, "exit": int},
}


def validate_record(rec: dict, where: str) -> str:
    kind = rec.get("kind")
    if kind not in SCHEMAS:
        raise LogIntegrityError(f"{where}: 未知の kind: {kind!r}")
    schema = SCHEMAS[kind]
    require_exact_keys(rec.keys(), schema.keys(), f"{where} {kind} のフィールド")
    for k, typ in schema.items():
        v = rec[k]
        ok = (isinstance(v, int) and not isinstance(v, bool)) if typ is int else isinstance(v, str)
        if not ok:
            raise LogIntegrityError(f"{where}: {k} の型が不正: {v!r}")
    if rec["v"] != 1:
        raise LogIntegrityError(f"{where}: 未知のスキーマ版 v={rec['v']}")
    if kind == "cell":
        if rec["stage"] not in STAGES:
            raise LogIntegrityError(f"{where}: 未知の stage: {rec['stage']!r}")
        if rec["status"] not in STATUSES:
            raise LogIntegrityError(f"{where}: 未知の status: {rec['status']!r}")
        if not CODE_RE.match(rec["code"]) or not ARCH_RE.match(rec["arch"]):
            raise LogIntegrityError(f"{where}: code／arch の書式が不正: {rec['code']!r} {rec['arch']!r}")
        if (rec["status"] in ("ok", "not_applicable_by_design")) != (rec["code"] == "-"):
            raise LogIntegrityError(
                f"{where}: status と code の対応が不正（ok・not_applicable_by_design のみ code は -）:"
                f" {rec['status']} {rec['code']}")
        if rec["status"] == "rejected" and rec["stage"] not in ("nvrtc_ptx", "nvrtc_cubin"):
            raise LogIntegrityError(f"{where}: rejected は S1／S2 のみ")
        if rec["status"] == "mismatch" and rec["stage"] != "verify":
            raise LogIntegrityError(f"{where}: mismatch は verify のみ")
    elif kind == "env":
        if not CC_RE.match(rec["cc"]) or not NVRTC_RE.match(rec["nvrtc"]):
            raise LogIntegrityError(f"{where}: env の cc／nvrtc の書式が不正")
    elif kind == "done":
        if rec["cells"] < 0 or rec["mismatch"] < 0:
            raise LogIntegrityError(f"{where}: done の件数が負")
    if rec["phase"] not in ("compile", "exec", "legacy", "dump"):
        raise LogIntegrityError(f"{where}: 未知の phase: {rec['phase']!r}")
    return kind


@dataclass
class ParsedLog:
    cells: list = field(default_factory=list)
    envs: list = field(default_factory=list)
    dones: list = field(default_factory=list)
    procs: list = field(default_factory=list)
    other_lines: list = field(default_factory=list)


def parse_log(text: str, name: str) -> ParsedLog:
    """`SM121_PROBE_JSON`／`SM121_PROBE_PROC` 行を厳密に読む。それ以外の行は
    cargo の出力等として `other_lines` に残す（内容は判定に使わない）。"""
    check_masked(text, name)
    out = ParsedLog()
    for i, raw in enumerate(text.splitlines(), 1):
        where = f"{name}:{i}"
        if raw.startswith(PREFIX_JSON):
            rec = strict_json(raw[len(PREFIX_JSON):], where)
            kind = validate_record(rec, where)
            if kind == "proc":
                raise LogIntegrityError(f"{where}: proc レコードは SM121_PROBE_PROC 行でのみ許容")
            {"cell": out.cells, "env": out.envs, "done": out.dones}[kind].append(rec)
        elif raw.startswith(PREFIX_PROC):
            rec = strict_json(raw[len(PREFIX_PROC):], where)
            if validate_record(rec, where) != "proc":
                raise LogIntegrityError(f"{where}: SM121_PROBE_PROC 行に proc 以外のレコード")
            out.procs.append(rec)
        elif PREFIX_JSON.strip() in raw or PREFIX_PROC.strip() in raw:
            # 行頭以外に現れる識別子は、連結や混入の疑いとして拒否する。
            raise LogIntegrityError(f"{where}: 識別子が行頭にない（行の混入・連結の疑い）: {raw[:120]!r}")
        else:
            out.other_lines.append(raw)
    return out


# ------------------------------------------------------------------ モデル

@dataclass(frozen=True)
class Cell:
    status: str
    code: str
    detail: str
    arch: str = "-"


@dataclass
class ExecProc:
    probe: str
    target: str
    env: dict | None
    cells: dict
    exit_code: int
    synthesized: bool


@dataclass
class Run:
    measured: bool
    mode: str  # official / dev
    provenance: dict
    compile_env: dict | None
    compile_cells: dict          # (probe, target_or_home_or_hopper, stage) -> Cell
    compile_missing: bool        # NVRTC 不在で compile が完走しなかった
    execs: dict                  # (probe, target) -> ExecProc
    legacy: dict                 # name -> {"class": str, "exit": int}
    targets: tuple
    compile_failure: str | None = None   # compile が打ち切られた: TIMEOUT／PROCESS_FAILED（完走なら None）
    dump_status: str = "-"               # device_attributes_dump の記録状態: ok／TIMEOUT／PROCESS_FAILED


def parse_env_info(text: str) -> dict:
    check_masked(text, "env_info.txt")
    kv: dict = {}
    for raw in text.splitlines():
        line = raw.strip()
        m = re.match(r"^([a-z0-9_]+)=(.*)$", line)
        if m:
            register_once(kv, m.group(1), m.group(2), "env_info.txt")
    return kv


def validate_chain(pdef: ProbeDef, cells: dict, where: str) -> None:
    """連鎖の矛盾・policy 固定状態との不一致を拒否する（R-STAGE）。"""
    dead = any(cells[st].status in ("timeout", "process_failed") for st in STAGES)
    for stage in STAGES:
        c = cells[stage]
        if stage == "ctl":
            na_ok = pdef.id == "ctl.copy"
            if (c.status == "not_applicable_by_design") != na_ok:
                raise LogIntegrityError(f"{where}: ctl の not_applicable_by_design は ctl.copy のみ")
            # timeout／process_failed は aggregate が合成する（対照段でのハング・異常終了）。
            if c.status not in ("ok", "error", "timeout", "process_failed", "not_applicable_by_design"):
                raise LogIntegrityError(f"{where}: ctl の状態が不正: {c.status}")
            continue
        na = stage_is_na(pdef.policy, stage)
        if cells["ctl"].status in ("timeout", "process_failed") and not na:
            continue  # 打ち切り時の検査は下の専用分岐が行う
        if na != (c.status == "not_applicable_by_design"):
            raise LogIntegrityError(
                f"{where}: {stage} は policy={pdef.policy} で設計上不実施かどうかが状態と矛盾: {c.status}")
        if stage in ("nvrtc_ptx", "nvrtc_cubin") and (c.status == "mismatch" or (c.status == "not_run" and not dead)):
            raise LogIntegrityError(f"{where}: {stage} に不正な状態 {c.status}")
        if stage == "verify" and c.status == "unavailable" and not pdef.is_attr:
            raise LogIntegrityError(f"{where}: verify の unavailable は attr のみ")
    if cells["ctl"].status in ("timeout", "process_failed"):
        # 対照段で打ち切られたプロセス: 以降の段（設計上不実施を除く）はすべて実行されていない。
        for stage in STAGES[1:]:
            if cells[stage].status not in ("not_run", "not_applicable_by_design"):
                raise LogIntegrityError(f"{where}: ctl が打ち切られたのに {stage} が {cells[stage].status}")
        return
    if pdef.is_attr:
        return
    chain = ("nvrtc_ptx", "nvrtc_cubin", "module_load", "launch", "sync", "verify")
    upstream_ok = True
    for stage in chain:
        c = cells[stage]
        if c.status == "not_applicable_by_design":
            continue
        if stage == "nvrtc_cubin":
            # S2 は S3 を止めない独立の枝。プロセスが S2 で打ち切られた場合だけ後続を止める。
            if c.status in ("timeout", "process_failed"):
                upstream_ok = False
            continue
        if not upstream_ok and c.status != "not_run":
            raise LogIntegrityError(f"{where}: 前段が失敗なのに {stage} が {c.status}")
        if upstream_ok and c.status == "not_run":
            raise LogIntegrityError(f"{where}: 前段が ok なのに {stage} が not_run")
        if c.status != "ok":
            upstream_ok = False


def load_exec_proc(rule: Rule, pdef: ProbeDef, target: str, text: str, name: str) -> ExecProc:
    log = parse_log(text, name)
    cells: dict = {}
    order = []
    for rec in log.cells:
        if rec["phase"] != "exec" or rec["probe"] != pdef.id or rec["target"] != target:
            raise LogIntegrityError(f"{name}: セルの phase／probe／target がファイル名と不一致: {rec['probe']} {rec['target']}")
        want_arch = expected_arch(rule, pdef, target, rec["stage"])
        if rec["arch"] != want_arch:
            raise LogIntegrityError(
                f"{name}: {rec['stage']} の arch が期待 {want_arch} と不一致: {rec['arch']!r}")
        register_once(cells, rec["stage"], Cell(rec["status"], rec["code"], rec["detail"], rec["arch"]),
                      f"{name} cell")
        order.append(rec["stage"])
    envs: dict = {}
    for rec in log.envs:
        if rec["phase"] != "exec" or rec["probe"] != pdef.id or rec["target"] != target:
            raise LogIntegrityError(f"{name}: env の識別子がファイル名と不一致")
        register_once(envs, "env", rec, f"{name} env")
    dones: dict = {}
    for rec in log.dones:
        if rec["phase"] != "exec" or rec["probe"] != pdef.id or rec["target"] != target:
            raise LogIntegrityError(f"{name}: done の識別子がファイル名と不一致")
        register_once(dones, "done", rec, f"{name} done")
    procs: dict = {}
    for rec in log.procs:
        if rec["phase"] != "exec" or rec["probe"] != pdef.id or rec["target"] != target:
            raise LogIntegrityError(f"{name}: proc の識別子がファイル名と不一致")
        register_once(procs, "proc", rec, f"{name} proc")
    if "proc" not in procs:
        raise LogIntegrityError(f"{name}: proc 記録（終了コード）が無い")
    code = procs["proc"]["exit"]
    done = dones.get("done")
    complete = len(cells) == len(STAGES)
    if tuple(order) != STAGES[:len(order)]:
        raise LogIntegrityError(f"{name}: セルが段の実行順の接頭辞になっていない: {order}")
    n_mismatch = sum(1 for c in cells.values() if c.status == "mismatch")
    if done is not None:
        if not complete:
            raise LogIntegrityError(f"{name}: done があるのにセルが 7 件そろっていない")
        if done["cells"] != len(STAGES) or done["mismatch"] != n_mismatch:
            raise LogIntegrityError(f"{name}: done の件数が実際のセルと不一致")
    synthesized = False
    if code == 0:
        if "env" not in envs:
            raise LogIntegrityError(f"{name}: exit 0 なのに env レコードが無い")
        if done is None or not complete or n_mismatch:
            raise LogIntegrityError(f"{name}: exit 0 なのに完了記録・全セルがそろっていない／mismatch がある")
    elif code == 101 and done is not None:
        if n_mismatch < 1:
            raise LogIntegrityError(f"{name}: exit 101 で完了しているのに mismatch が無い（原因不明の panic）")
    else:
        if done is not None or complete:
            raise LogIntegrityError(f"{name}: exit {code} なのに完了記録・全セルがそろっている（矛盾）")
        # 未完了: 先頭の欠落段に timeout／process_failed を合成し、後続は not_run／設計上不実施。
        first = True
        status = "timeout" if code == 124 else "process_failed"
        for stage in STAGES:
            if stage in cells:
                continue
            if stage == "ctl" and pdef.id == "ctl.copy":
                # ctl.copy 自身の ctl は設計上不実施（対照そのものがプローブ）。打ち切りは次の段に付く。
                cells[stage] = Cell("not_applicable_by_design", "-", "this probe is the control")
            elif stage != "ctl" and stage_is_na(pdef.policy, stage):
                cells[stage] = Cell("not_applicable_by_design", "-", f"policy={pdef.policy}")
            elif first:
                cells[stage] = Cell(status, f"EXIT_{code}", f"aggregate が合成（exit={code}。外部 timeout は 124）")
                first = False
            else:
                cells[stage] = Cell("not_run", "UPSTREAM_FAILED", "aggregate が合成（前段が未完了）")
        synthesized = True
    validate_chain(pdef, cells, name)
    if pdef.is_attr and cells["verify"].status == "ok":
        parse_kv(cells["verify"].detail.split(), f"{name} attr detail", strict=True)
    if pdef.clause == "R-TMA-SEM" and cells["sync"].status == "ok":
        # 観測は status・polls を含む k=v トークン列（候補モデルとの一致を機械可読に残す）。
        kv = parse_kv(cells["sync"].detail.removeprefix("record: ").split(), f"{name} TMA 観測", strict=True)
        if kv.get("status") not in ("complete", "timeout") or not kv.get("polls", "x").isdigit():
            raise LogIntegrityError(f"{name}: TMA 観測に status／polls が無い: {cells['sync'].detail[:120]!r}")
    return ExecProc(pdef.id, target, envs.get("env"), cells, code, synthesized)


def load_compile(rule: Rule, text: str, targets: tuple) -> tuple:
    """compile.log を読む。戻り値は (env, cells, nvrtc_missing, failure)。

    exit 0 は完了記録と全セルが必須。exit 124／その他の非 0 は測定そのものの打ち切りで、
    打ち切りまでに出たセルを検証して採用し、未出力のセルは欠測とする（failure に TIMEOUT／
    PROCESS_FAILED を返し、欠測に依存する判定を verdict_for が判定不能にする）。
    exit 0 なのに欠落、完了しているのに非 0、proc 記録の欠落は完全性違反（RULE.txt 13）。"""
    log = parse_log(text, "compile.log")
    envs = [r for r in log.envs if r["phase"] == "compile"]
    if len(envs) != len(log.envs) or len(envs) > 1:
        raise LogIntegrityError("compile.log: env レコードが 2 件以上、または phase が compile でない")
    procs = [r for r in log.procs if r["phase"] == "compile"]
    if len(procs) != 1 or len(log.procs) != 1:
        raise LogIntegrityError("compile.log: proc 記録がちょうど 1 件でない")
    exit_code = procs[0]["exit"]
    if procs[0]["probe"] != "-" or procs[0]["target"] != "-":
        raise LogIntegrityError("compile.log: proc の probe／target が - でない")
    env = envs[0] if envs else None
    if env is None and exit_code == 0:
        raise LogIntegrityError("compile.log: exit 0 なのに env レコードが無い")
    if env is not None and (env["probe"] != "-" or env["target"] != "-"):
        raise LogIntegrityError("compile.log: env の probe／target が - でない")
    if env is not None and env["nvrtc"] == "unavailable":
        if log.cells or log.dones:
            raise LogIntegrityError("compile.log: NVRTC 不在なのにセルまたは done がある")
        return env, {}, True, None
    cells: dict = {}
    for rec in log.cells:
        if rec["phase"] != "compile":
            raise LogIntegrityError("compile.log: phase が compile でないセル")
        if rec["probe"] not in rule.probes or rule.probes[rec["probe"]].is_attr:
            raise LogIntegrityError(f"compile.log: 未知またはカーネルを持たないプローブ: {rec['probe']}")
        if rec["target"] not in tuple(targets) + ("home", "hopper"):
            raise LogIntegrityError(f"compile.log: 未知の target: {rec['target']}")
        if rec["stage"] not in ("nvrtc_ptx", "nvrtc_cubin"):
            raise LogIntegrityError(f"compile.log: S1／S2 以外の stage: {rec['stage']}")
        if rec["status"] not in ("ok", "rejected", "error", "unavailable"):
            raise LogIntegrityError(f"compile.log: 不正な状態 {rec['status']}")
        pdef = rule.probes[rec["probe"]]
        if rec["target"] == "home":
            want_arch = pdef.home
        elif rec["target"] == "hopper":
            want_arch = HOPPER_ARCH
        else:
            want_arch = expected_arch(rule, pdef, rec["target"], rec["stage"])
        if rec["arch"] != want_arch:
            raise LogIntegrityError(
                f"compile.log: {rec['probe']} {rec['target']} {rec['stage']} の arch が期待 {want_arch}"
                f" と不一致: {rec['arch']!r}")
        register_once(cells, (rec["probe"], rec["target"], rec["stage"]),
                      Cell(rec["status"], rec["code"], rec["detail"], rec["arch"]), "compile.log cell")
    want = set()
    for pid, p in rule.probes.items():
        if p.is_attr:
            continue
        for t in targets:
            want.add((pid, t, "nvrtc_ptx"))
            want.add((pid, t, "nvrtc_cubin"))
        want.add((pid, "home", "nvrtc_cubin"))
        want.add((pid, "hopper", "nvrtc_cubin"))
    for (pid, t, stage) in cells:
        if t in ("home", "hopper") and stage != "nvrtc_cubin":
            raise LogIntegrityError(f"compile.log: {t} は S2 のみ: {pid} {stage}")
    extra = set(cells) - want
    if extra:
        raise LogIntegrityError(f"compile.log のセル: 余剰・未知のキー {sorted(extra)}")
    complete = set(cells) == want
    done_ok = (len(log.dones) == 1 and log.dones[0]["phase"] == "compile"
               and log.dones[0]["probe"] == "-" and log.dones[0]["target"] == "-"
               and log.dones[0]["cells"] == len(cells) and log.dones[0]["mismatch"] == 0)
    if exit_code == 0:
        if not complete or not done_ok:
            raise LogIntegrityError("compile.log: exit 0 なのに全セル・完了記録がそろっていない")
        return env, cells, False, None
    if complete or log.dones:
        raise LogIntegrityError(f"compile.log: exit {exit_code} なのに全セルまたは完了記録がある（矛盾）")
    return env, cells, False, "TIMEOUT" if exit_code == 124 else "PROCESS_FAILED"


def classify_legacy(text: str, name: str, exit_code: int, want_arch: str) -> str:
    """SETMAXNREG_PROBE_RESULT 行から結論を 1 つに分類する（ちょうど 1 つの終端行を要求）。"""
    check_masked(text, name)
    terminals = []
    for raw in text.splitlines():
        if not raw.startswith(LEGACY_PREFIX):
            continue
        body = raw[len(LEGACY_PREFIX):]
        main, _, _detail = body.partition(" detail=")
        kv = parse_kv(main.split(), f"{name} 行 {len(terminals)}")
        stage, result = kv.get("stage"), kv.get("result")
        arch = kv.get("arch")
        if arch is not None and arch != want_arch and stage in ("nvrtc_compile", "execute", "module_load",
                                                                  "launch", "synchronize", "load_function"):
            raise LogIntegrityError(f"{name}: arch が期待 {want_arch} と不一致: {raw[:160]!r}")
        if stage == "execute" and result == "success":
            terminals.append("run_ok")
        elif stage == "execute" and result == "corrupted":
            terminals.append("mismatch")
        elif stage in ("module_load", "load_function") and result == "failed":
            terminals.append("load_failed")
        elif stage in ("launch", "synchronize") and result == "failed":
            terminals.append("runtime_error")
        elif stage == "nvrtc_compile" and result == "rejected":
            terminals.append("nvrtc_rejected")
        elif stage == "nvrtc_compile" and result == "inconclusive":
            terminals.append("inconclusive")
    # 測定そのものの打ち切り・異常終了は比較不能（timeout／process_failed。新プローブ側は
    # LEGACY_INCONCLUSIVE）。終端の結論が残っているのに異常終了、exit 0 なのに結論が無い等の
    # 矛盾は完全性違反（RULE.txt 13）。
    if exit_code == 124:
        if terminals:
            raise LogIntegrityError(f"{name}: timeout なのに終端の結論がある: {terminals}")
        return "timeout"
    if exit_code == 0:
        if len(terminals) != 1 or terminals[0] == "mismatch":
            raise LogIntegrityError(f"{name}: exit 0 で終端の結論がちょうど 1 つ（corrupted 以外）でない: {terminals}")
        return terminals[0]
    if exit_code == 101 and terminals == ["mismatch"]:
        return "mismatch"
    if terminals:
        raise LogIntegrityError(f"{name}: exit {exit_code} なのに終端の結論がある: {terminals}")
    return "process_failed"


def classify_tma_legacy(text: str, name: str, exit_code: int, kind: str) -> tuple:
    """既存 tma_probe_real_device の 1 テスト分のログを (分類, 対象 target) にする（RULE.txt 12）。

    tma_compile: 6 行（3 arch x 2 変種）の記録のみ（分類 recorded）。tma_exec_*: 選択された arch
    （`(selected for execution probe)` 行）を対象 target とし、成功行があれば run_ok。"""
    check_masked(text, name)
    variant_want = {"tma_exec_cluster": "cluster", "tma_exec_cta": "cta"}.get(kind)
    compile_lines: dict = {}
    selected = []
    successes = []
    for raw in text.splitlines():
        if raw.startswith("tma_compile_probe "):
            main, _, _detail = raw[len("tma_compile_probe "):].partition(" detail=")
            kv = parse_kv(main.split(), f"{name} compile 行")
            if kv.get("variant") not in ("cluster", "cta") or "arch" not in kv or kv.get("result") not in (
                    "success", "failure"):
                raise LogIntegrityError(f"{name}: tma_compile_probe 行の書式が不正: {raw[:160]!r}")
            if kind == "tma_compile":
                register_once(compile_lines, (kv["variant"], kv["arch"]), kv["result"], f"{name} compile 行")
            elif "(selected for execution probe)" in raw and kv["variant"] == variant_want:
                selected.append(kv["arch"])
        elif raw.startswith("tma_execution_probe ") or raw.startswith("tma_execution_probe_cta "):
            main, _, _detail = raw.partition(" detail=")
            kv = parse_kv(main.split()[1:], f"{name} exec 行")
            if kv.get("result") == "success" and kv.get("variant") == variant_want:
                successes.append(kv.get("arch"))
    if kind == "tma_compile":
        if exit_code == 124:
            return "timeout", None
        if exit_code == 0:
            if len(compile_lines) != 6:
                raise LogIntegrityError(f"{name}: exit 0 なのに compile 行が 6 件（3 arch x 2 変種）でない: {len(compile_lines)}")
            return "recorded", None
        return "process_failed", None
    if len(selected) > 1 or len(successes) > 1:
        raise LogIntegrityError(f"{name}: 選択 arch／成功行が複数: {selected} {successes}")
    target = selected[0] if selected else None
    if successes:
        if exit_code != 0:
            raise LogIntegrityError(f"{name}: 成功行があるのに exit {exit_code}")
        if target is None or successes[0] != target:
            raise LogIntegrityError(f"{name}: 成功行の arch が選択 arch と不一致: {successes} {selected}")
        return "run_ok", target
    if exit_code == 0:
        raise LogIntegrityError(f"{name}: exit 0 なのに成功行が無い")
    if exit_code == 124:
        return "timeout", target
    if exit_code == 101 and "TMA 転送結果が期待するタイル" in text:
        return "mismatch", target
    if exit_code == 101 and "failed to compile for every arch" in text:
        return "nvrtc_rejected", target
    return "process_failed", target


def load_run(rule: Rule, log_dir: Path) -> Run:
    """ログディレクトリを読み、完全性を検査して Run を作る。全ログ無しは measured=False。"""
    env_path = log_dir / "env_info.txt"
    compile_path = log_dir / "compile.log"
    exec_dir = log_dir / "exec"
    exec_files = sorted(exec_dir.glob("*.log")) if exec_dir.is_dir() else []
    legacy_files = sorted(log_dir.glob("legacy-*.log"))
    dump_path = log_dir / "device_attributes_dump.log"
    nothing = not (env_path.exists() or compile_path.exists() or exec_files or legacy_files
                   or dump_path.exists())
    if nothing:
        return Run(False, "official", {}, None, {}, False, {}, {}, tuple(rule.targets))
    for need in (env_path, compile_path):
        if not need.exists():
            raise LogIntegrityError(f"{need.name} が無い（一部だけ欠けている）")
    prov = parse_env_info(env_path.read_text(encoding="utf-8"))
    if "mode" not in prov or prov["mode"] not in ("gb10", "dev_smoke"):
        raise LogIntegrityError("env_info.txt: mode が gb10／dev_smoke でない")
    for k in ("git_head", "git_clean", "start_utc", "end_utc"):
        if k not in prov:
            raise LogIntegrityError(f"env_info.txt: {k} が無い（未完了の実行の疑い）")
    if prov["git_clean"] not in ("0", "1"):
        raise LogIntegrityError(f"env_info.txt: git_clean が 0／1 でない: {prov['git_clean']!r}")
    for k in ("start_utc", "end_utc"):
        if not UTC_RE.match(prov[k]):
            raise LogIntegrityError(f"env_info.txt: {k} の書式が不正: {prov[k]!r}")
    # exec ファイルのファイル名から target 集合（正式／開発機）を決める。
    names = {}
    for f in exec_files:
        m = re.match(r"^(.+)@(.+)\.log$", f.name)
        if not m:
            raise LogIntegrityError(f"exec/{f.name}: ファイル名が <probe>@<target>.log でない")
        pid, t = m.group(1), m.group(2)
        if pid not in rule.probes:
            raise LogIntegrityError(f"exec/{f.name}: 未知のプローブ")
        if t not in rule.targets and t not in rule.dev_targets:
            raise LogIntegrityError(f"exec/{f.name}: 未知の target")
        register_once(names, (pid, t), f, "exec ファイル")
    ts = {t for (_, t) in names}
    official, dev = ts & set(rule.targets), ts & set(rule.dev_targets)
    if official and dev:
        raise LogIntegrityError("正式 target と開発機 target のログが混在している")
    mode = "dev" if dev else "official"
    targets = tuple(rule.dev_targets) if dev else tuple(rule.targets)
    if prov["mode"] == "dev_smoke" and not dev and names:
        raise LogIntegrityError("env_info の mode=dev_smoke なのに正式 target のログがある")
    want = {(pid, t) for pid in rule.probes for t in targets}
    require_exact_keys(names.keys(), want, "exec ファイル集合")
    compile_env, compile_cells, compile_missing, compile_failure = load_compile(
        rule, compile_path.read_text(encoding="utf-8"), targets)
    execs: dict = {}
    for key, f in names.items():
        execs[key] = load_exec_proc(rule, rule.probes[key[0]], key[1], f.read_text(encoding="utf-8"),
                                    f"exec/{f.name}")
    legacy: dict = {}
    for f in legacy_files:
        m = re.match(r"^legacy-(.+)\.log$", f.name)
        lname = m.group(1) if m else ""
        if lname not in LEGACY_NAMES:
            raise LogIntegrityError(f"{f.name}: 未知の legacy ログ")
        text = f.read_text(encoding="utf-8")
        log = parse_log(text, f.name)
        if log.cells or log.envs or log.dones:
            raise LogIntegrityError(f"{f.name}: legacy ログに新形式のセルがある")
        procs = [r for r in log.procs if r["phase"] == "legacy" and r["probe"] == lname
                 and r["target"] == "-"]
        if len(procs) != 1 or len(log.procs) != 1:
            raise LogIntegrityError(f"{f.name}: proc 記録がちょうど 1 件でない")
        _probe, fixed_target, kind = LEGACY_NAMES[lname]
        if kind == "snr":
            cls, ltarget = classify_legacy(text, f.name, procs[0]["exit"], fixed_target), fixed_target
        else:
            cls, ltarget = classify_tma_legacy(text, f.name, procs[0]["exit"], kind)
        register_once(legacy, lname, {"class": cls, "exit": procs[0]["exit"], "target": ltarget}, "legacy")
    # R-LEGACY: official では legacy 7 件が必須（欠測を黙認すると legacy 照合が飛び「成立」になる）。
    # dev（開発機スモーク）では legacy を回さないので 0 件を要求する。
    require_exact_keys(legacy.keys(), LEGACY_NAMES.keys() if mode == "official" else (),
                       f"legacy ログの集合（mode={mode}）")
    # device_attributes_dump.log: 値は記録用だが、存在と proc 記録（exit 0）は必須（欠測の黙認をしない）。
    if not dump_path.exists():
        raise LogIntegrityError("device_attributes_dump.log が無い（記録用だが欠測は許容しない）")
    dump = parse_log(dump_path.read_text(encoding="utf-8"), "device_attributes_dump.log")
    if dump.cells or dump.envs or dump.dones:
        raise LogIntegrityError("device_attributes_dump.log に新形式のセルがある")
    dprocs = [r for r in dump.procs if r["phase"] == "dump" and r["probe"] == "-" and r["target"] == "-"]
    if len(dprocs) != 1 or len(dump.procs) != 1:
        raise LogIntegrityError("device_attributes_dump.log: proc 記録がちょうど 1 件でない")
    # 記録用（判定に使わない）なので、打ち切り（124）・異常終了は欠測として報告し集計は続ける。
    # exit 0 なのに本文が無い（proc 行だけ）は出力欠落として完全性違反。
    if dprocs[0]["exit"] == 0:
        if not [l for l in dump.other_lines if l.strip()]:
            raise LogIntegrityError("device_attributes_dump.log: exit 0 なのに出力が無い")
        dump_status = "ok"
    else:
        dump_status = "TIMEOUT" if dprocs[0]["exit"] == 124 else "PROCESS_FAILED"
    return Run(True, mode, prov, compile_env, compile_cells, compile_missing, execs, legacy, targets,
               compile_failure, dump_status)


# ------------------------------------------------------------------ 判定

@dataclass(frozen=True)
class Verdict:
    word: str
    code: str   # 判定不能のときの理由コード（それ以外は "-"）
    why: str


def evaluate_g0(run: Run) -> list:
    """前提ゲート G0。不成立の理由コードのリスト（空なら成立）。"""
    reasons = []
    if run.provenance.get("mode") != "gb10":
        reasons.append("G0_DEV_MODE")
    if not SHA_RE.match(run.provenance.get("git_head", "")):
        reasons.append("G0_PROVENANCE")
    if run.provenance.get("git_clean") != "1":
        reasons.append("G0_DIRTY_TREE")
    if run.mode != "official":
        reasons.append("G0_DEV_TARGET")
    if run.compile_missing or run.compile_env is None or run.compile_env.get("nvrtc") == "unavailable":
        reasons.append("G0_NVRTC_MISSING")
    envs = [p.env for p in run.execs.values()]
    if any(e is None or e["cc"] == "none" or e["device"] == "none" for e in envs):
        reasons.append("G0_NO_DEVICE")
    if any(e is not None and e["cc"] not in ("none", "12.1") for e in envs):
        reasons.append("G0_CC")
    return [r for r in G0_REASONS if r in reasons]


def new_class(cells: dict) -> str | None:
    """新プローブの結論を legacy と同じ分類語へ写す（比較できなければ None）。"""
    if cells["nvrtc_ptx"].status == "rejected":
        return "nvrtc_rejected"
    if cells["module_load"].status == "error":
        return "load_failed"
    if cells["launch"].status == "error" or cells["sync"].status == "error":
        return "runtime_error"
    if cells["verify"].status == "ok":
        return "run_ok"
    if cells["verify"].status == "mismatch":
        return "mismatch"
    if "timeout" in {c.status for c in cells.values()}:
        return "timeout"
    return None


def indet(code: str, why: str) -> Verdict:
    return Verdict(V_INDET, code, why)


def verdict_core(rule: Rule, run: Run, g0: list, pdef: ProbeDef, target: str) -> Verdict:
    """(プローブ, target) の判定（RULE.txt 6 の順序。依存〈10b〉は含まない）。"""
    if g0:
        return indet("G0_FAILED", "前提ゲート不成立: " + ",".join(g0))
    p = run.execs[(pdef.id, target)]
    c = p.cells
    for stage in STAGES:
        if c[stage].status == "timeout":
            return indet("TIMEOUT", f"{stage} で外部 timeout（記録できた段までを参照）")
        if c[stage].status == "process_failed":
            return indet("PROCESS_FAILED", f"{stage} で異常終了 exit={p.exit_code}")
    if pdef.id != "ctl.copy" and c["ctl"].status != "ok":
        code = c["ctl"].code if c["ctl"].code in ("TARGET_UNSUPPORTED", "CTL_FAILED") else "CTL_FAILED"
        return indet(code, f"対照 ctl.copy が失敗: {c['ctl'].detail}")
    if pdef.is_attr:
        v = c["verify"]
        if v.status == "ok":
            return Verdict(V_OK, "-", "属性を記録")
        if v.status == "unavailable":
            return indet("NO_DEVICE", v.detail)
        return indet("STAGE_ERROR", v.detail)
    # compile が打ち切られて必要なセル（home・S1／S2・hopper）が欠測なら、R-HOME・R-HOPPER に
    # 依存する判定は TIMEOUT／PROCESS_FAILED で判定不能（RULE.txt 6 (4)）。
    if run.compile_failure is not None:
        need = [(pdef.id, "home", "nvrtc_cubin"), (pdef.id, target, "nvrtc_ptx"),
                (pdef.id, target, "nvrtc_cubin")]
        if pdef.home == HOPPER_ARCH:
            need.append((pdef.id, "hopper", "nvrtc_cubin"))
        missing = [k for k in need if k not in run.compile_cells]
        if missing:
            return indet(run.compile_failure, f"compile が打ち切られ {missing[0][1]}／{missing[0][2]} が欠測")
    home = run.compile_cells.get((pdef.id, "home", "nvrtc_cubin"))
    if home is None or home.status != "ok":
        if home is not None and home.status == "rejected":
            return indet("HOME_REJECTED", f"本来の対応アーキ {pdef.home} で拒否（プローブ不良の疑い）")
        return indet("HOME_NOT_OK", f"本来の対応アーキ {pdef.home} の S2 が {home.status if home else '欠測'}")
    for stage in ("nvrtc_ptx", "nvrtc_cubin"):
        cc = run.compile_cells[(pdef.id, target, stage)]
        if cc.status != c[stage].status:
            return indet("COMPILE_EXEC_DISAGREE",
                         f"{stage}: compile={cc.status} exec={c[stage].status}")
    if pdef.home == HOPPER_ARCH:
        hop = run.compile_cells[(pdef.id, "hopper", "nvrtc_cubin")]
        if hop.status != home.status:
            return indet("HOPPER_HOME_DISAGREE", f"同一アーキなのに home={home.status} hopper={hop.status}")
    for lname, (pid, _fixed, kind) in LEGACY_NAMES.items():
        if pid is not None and pid == pdef.id and lname in run.legacy and run.legacy[lname]["target"] in (target, None):
            mine = new_class(c)
            theirs = run.legacy[lname]["class"]
            if kind.startswith("tma_exec"):
                # 既存 TMA プローブの失敗は panic の理由を細かく区別できない（ロード失敗・実行時エラー等が
                # すべて process_failed になる）ため、成功か否かだけを比較する。timeout は比較不能。
                if theirs == "timeout":
                    return indet("LEGACY_INCONCLUSIVE", "legacy が timeout（比較できない）")
                if mine is not None and (mine == "run_ok") != (theirs == "run_ok"):
                    return indet("LEGACY_CONTRADICTION", f"legacy={theirs} 新プローブ={mine}")
                continue
            if theirs in ("inconclusive", "timeout", "process_failed"):
                return indet("LEGACY_INCONCLUSIVE", f"legacy の結論が {theirs}（比較できない）")
            if mine is not None and mine != theirs:
                return indet("LEGACY_CONTRADICTION", f"legacy={theirs} 新プローブ={mine}")
    s1, s2, s3 = c["nvrtc_ptx"], c["nvrtc_cubin"], c["module_load"]
    if s1.status == "rejected":
        return Verdict(V_NVRTC, "-", "S1 で拒否")
    if s1.status == "unavailable":
        return indet("NVRTC_UNAVAILABLE", s1.detail)
    if s1.status == "error":
        return indet("STAGE_ERROR", s1.detail)
    if s2.status == "rejected":
        if s3.status == "ok":
            return indet("OFFLINE_JIT_DISAGREE", "オフライン ptxas は拒否、ドライバ JIT はロード成功")
        return Verdict(V_PTXAS, "-", "S2（オフライン ptxas）で拒否")
    if s2.status == "unavailable":
        return indet("NVRTC_UNAVAILABLE", s2.detail)
    if s2.status == "error":
        return indet("STAGE_ERROR", s2.detail)
    if s3.status == "unavailable":
        return indet("NO_DEVICE", s3.detail)
    if s3.status == "error":
        if s3.code == "CUDA_ERROR_UNSUPPORTED_PTX_VERSION":
            return indet("UNSUPPORTED_PTX_VERSION", s3.detail)
        return Verdict(V_LOAD, "-", f"S3 {s3.code}")
    for stage in ("launch", "sync"):
        if c[stage].status == "error":
            return Verdict(V_RUNTIME, "-", f"{stage} {c[stage].code}")
    v = c["verify"]
    if v.status == "mismatch":
        if pdef.layout == "unverified":
            return indet("LAYOUT_UNVERIFIED", f"レイアウト未検証の形状の不一致: {v.detail}")
        return Verdict(V_MISMATCH, "-", v.detail)
    if v.status == "error":
        return indet("STAGE_ERROR", v.detail)
    if pdef.expect == "reject121":
        return indet("UNEXPECTED_ACCEPT", "拒否される想定だったが全段が通った（実行意味論は未検証）")
    if pdef.policy == "verify":
        return Verdict(V_OK, "-", "S1〜S6 すべて ok")
    return Verdict(V_ACCEPT_ONLY, "-", "受理段まで（policy=" + pdef.policy + "）")


def tma_dependencies(pdef: ProbeDef) -> list:
    """事前登録の依存（RULE.txt 10b）: (依存先プローブ, 不成立のときの理由コード) の列。"""
    if not pdef.id.startswith("tma.") or pdef.id in TMA_DEPENDENT_EXCLUDE:
        return []
    deps = [(TMA_BASE, "TMA_BASE_NOT_ESTABLISHED")]
    if pdef.id == "tma.multicast":
        deps.append(("clu.dims2", "CLU_DIMS2_NOT_ESTABLISHED"))
    return deps


def verdict_for(rule: Rule, run: Run, g0: list, pdef: ProbeDef, target: str) -> Verdict:
    """(プローブ, target) の最終判定。自身が 成立／受理のみ になるときに限り、依存先が同 target で
    成立 でなければ判定不能にする（自身が拒否・失敗・不一致ならその判定をそのまま採る）。"""
    own = verdict_core(rule, run, g0, pdef, target)
    if own.word not in (V_OK, V_ACCEPT_ONLY):
        return own
    for dep_id, code in tma_dependencies(pdef):
        dep = verdict_core(rule, run, g0, rule.probes[dep_id], target)
        if dep.word != V_OK:
            return indet(code, f"依存先 {dep_id} が {dep.word}"
                         + (f"（{dep.code}）" if dep.code != "-" else ""))
    return own


def compute_verdicts(rule: Rule, run: Run) -> tuple:
    """全 (プローブ, target) の判定と G0 の理由。判定はここ 1 か所で作り、描画は読むだけ。"""
    if not run.measured:
        return [], {(pid, t): Verdict(V_UNMEASURED, "-", "ログなし") for pid in rule.probes
                    for t in rule.targets}
    g0 = evaluate_g0(run)
    verdicts = {}
    for pid, pdef in rule.probes.items():
        for t in run.targets:
            verdicts[(pid, t)] = verdict_for(rule, run, g0, pdef, t)
    return g0, verdicts


# ------------------------------------------------------------------ R-GUIDE

def load_claims(text: str) -> list:
    rows = []
    for raw in text.splitlines():
        if not raw.strip() or raw.startswith("#"):
            continue
        cols = raw.split("\t")
        if len(cols) != 6:
            raise LogIntegrityError(f"guide_claims.tsv: 6 列でない行: {raw[:100]!r}")
        rows.append(dict(zip(("id", "source", "quote", "probe", "key", "compare"), cols)))
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise LogIntegrityError("guide_claims.tsv: claim_id が重複")
    return rows


def check_claim_quotes(rows: list, repo_root: Path) -> None:
    """引用が引用元の当該行に実在することを照合する（self-test から呼ぶ）。"""
    for r in rows:
        path, _, line = r["source"].rpartition(":")
        if not line.isdigit():
            raise LogIntegrityError(f"{r['id']}: source の書式が path:line でない: {r['source']}")
        lines = (repo_root / path).read_text(encoding="utf-8").splitlines()
        n = int(line)
        if not (1 <= n <= len(lines)) or r["quote"] not in lines[n - 1]:
            raise LogIntegrityError(f"{r['id']}: 引用が {r['source']} に実在しない: {r['quote']!r}")


def attr_values(run: Run, probe: str, key: str) -> list | None:
    """全 target の属性値（整数）。1 つでも得られなければ None。"""
    vals = []
    for t in run.targets:
        p = run.execs.get((probe, t))
        if p is None or p.cells["verify"].status != "ok":
            return None
        kv = parse_kv(p.cells["verify"].detail.split(), f"{probe}@{t} attr detail", strict=True)
        raw = kv.get(key)
        if raw is None or not re.fullmatch(r"-?[0-9]+", raw):
            return None
        vals.append(int(raw))
    return vals


def evaluate_claim(rule: Rule, run: Run, verdicts: dict, g0: list, claim: dict) -> tuple:
    """(判定語, 理由コード, 測定値の要約)。"""
    if claim["probe"] == "-":
        return "判定不能", "GUIDE_NO_PROBE", "対応するプローブがない"
    if not run.measured or g0:
        return "判定不能", "GUIDE_UNMEASURED", "未実測または G0 不成立"
    cmp_ = claim["compare"]
    kind, _, arg = cmp_.partition(":")
    if claim["probe"] not in rule.probes:
        raise LogIntegrityError(f"{claim['id']}: 未知のプローブ {claim['probe']}")
    if kind in ("eq", "kib", "div"):
        vals = attr_values(run, claim["probe"], claim["key"])
        if vals is None:
            return "判定不能", "GUIDE_UNMEASURED", "属性値を取得できていない"
        if kind == "eq":
            ok = [v == int(arg) for v in vals]
        elif kind == "kib":
            ok = [v == int(arg) * 1024 for v in vals]
        else:
            d, _, n = arg.partition(":")
            ok = [v % int(d) == 0 and v // int(d) == int(n) for v in vals]
        summary = "測定値=" + ",".join(str(v) for v in vals)
        return (G_MATCH if all(ok) else G_DIFF), "-", summary
    if kind in ("verdict_is", "verdict_in"):
        allowed = {arg} if kind == "verdict_is" else set(arg.split("|"))
        words = []
        for t in run.targets:
            v = verdicts[(claim["probe"], t)]
            if v.word == V_INDET:
                return "判定不能", "GUIDE_UNMEASURED", f"{t}: 判定不能（{v.code}）"
            words.append(v.word)
        summary = "判定=" + ",".join(words)
        return (G_MATCH if all(w in allowed for w in words) else G_DIFF), "-", summary
    raise LogIntegrityError(f"{claim['id']}: 未知の compare: {cmp_}")


# ------------------------------------------------------------------ 描画

def tma_observation(run: Run, pid: str, target: str) -> str:
    """R-TMA-SEM の観測（S5 の detail の k=v トークン。S5 が ok でなければ `-`）。"""
    p = run.execs.get((pid, target))
    if p is None or p.cells["sync"].status != "ok":
        return "-"
    return p.cells["sync"].detail.removeprefix("record: ")


def render(rule: Rule, run: Run, g0: list, verdicts: dict, claims: list) -> str:
    out = ["# sm_121 ISA プローブ集計（イシュー #2122）", ""]
    if not run.measured:
        out += ["全セル: 未実測（ログなし）。", ""]
    out += ["## G0（前提ゲート）", ""]
    if not run.measured:
        out.append("- 未実測")
    elif g0:
        out.append("- **不成立**: " + ", ".join(g0) + "（全セルを判定不能とする。正式な結果として反映しない）")
    else:
        out.append("- 成立")
    out += ["", f"mode={run.mode if run.measured else '-'}　git_head={run.provenance.get('git_head', '-')}", ""]
    if run.measured:
        out += [f"- compile プロセス: {run.compile_failure or '完走'}"
                + ("（打ち切り。欠測セルに依存する判定は判定不能）" if run.compile_failure else ""),
                f"- device_attributes_dump（記録のみ）: {run.dump_status}"
                + ("（欠測。判定には使わない）" if run.dump_status != "ok" else ""), ""]
    targets = run.targets
    out += ["## プローブ別の判定", "",
            "| プローブ | AC | 条項 | home | " + " | ".join(targets) + " |",
            "|---|---|---|---|" + "|".join("---" for _ in targets) + "|"]
    for pid, p in rule.probes.items():
        cells = []
        for t in targets:
            v = verdicts[(pid, t)]
            cells.append(v.word + (f"（{v.code}）" if v.code != "-" else ""))
        out.append(f"| {pid} | {p.ac} | {p.clause} | {p.home} | " + " | ".join(cells) + " |")
    out += ["", "## R-GUIDE（ガイドとの突き合わせ）", "",
            "| claim | 出典 | 判定 | 理由コード | 測定 |", "|---|---|---|---|---|"]
    for cl in claims:
        word, code, summary = evaluate_claim(rule, run, verdicts, g0, cl)
        out.append(f"| {cl['id']} | {cl['source']} | {word} | {code} | {summary} |")
    out += ["", "## R-HOPPER（Hopper との差分）", "",
            "Hopper 列は GB10 上の NVRTC（ptxas）による sm_90a 受理の実測であり、Hopper 実機での実行は"
            "未検証。PTX ISA 9.0 の節番号は要確認。", "",
            "| プローブ | home | Hopper(sm_90a) S2 | sm_121 系 S2（121/121a/121f） | 判定 | 方向 | PTX ISA 9.0 節 |",
            "|---|---|---|---|---|---|---|"]
    for pid, p in rule.probes.items():
        if p.is_attr:
            continue
        hop = run.compile_cells.get((pid, "hopper", "nvrtc_cubin")) if run.measured else None
        s2 = [run.compile_cells.get((pid, t, "nvrtc_cubin")) for t in targets] if run.measured else []
        hop_s = hop.status if hop else "未実測"
        s2_s = "/".join(c.status if c else "未実測" for c in s2) if s2 else "未実測"
        vs = {verdicts[(pid, t)].word for t in targets}
        verdict_s = "/".join(sorted(vs))
        if not run.measured or g0 or hop is None:
            direction = "判定不能"
        else:
            sm121_ok = all(c is not None and c.status == "ok" for c in s2)
            sm121_rej = all(c is not None and c.status == "rejected" for c in s2)
            if hop.status == "ok" and sm121_ok:
                direction = "両方で受理"
            elif hop.status == "ok" and sm121_rej:
                direction = "Hopper のみ受理"
            elif hop.status == "rejected" and sm121_ok:
                direction = "sm_121 のみ受理（逆方向）"
            elif hop.status == "rejected" and sm121_rej:
                direction = "両方で拒否"
            else:
                direction = "判定不能（target 間で不一致）"
        out.append(f"| {pid} | {p.home} | {hop_s} | {s2_s} | {verdict_s} | {direction} | 要確認 |")
    if run.measured:
        out += ["", "## R-TMA 観測（候補モデルとの一致。値は S5 の detail に記録した観測）", "",
                "| プローブ | target | 判定 | 観測 |", "|---|---|---|---|"]
        for pid, p in rule.probes.items():
            if p.clause != "R-TMA-SEM":
                continue
            for t in run.targets:
                v = verdicts[(pid, t)]
                word = v.word + (f"（{v.code}）" if v.code != "-" else "")
                out.append(f"| {pid} | {t} | {word} | {tma_observation(run, pid, t)} |")
    if run.legacy:
        out += ["", "## R-LEGACY", "", "| legacy | 対応 | 結論 | exit |", "|---|---|---|---|"]
        for lname, d in sorted(run.legacy.items()):
            pid, fixed, _kind = LEGACY_NAMES[lname]
            out.append(f"| {lname} | {pid or '-'}@{d['target'] or fixed or '-'} | {d['class']} | {d['exit']} |")
    return "\n".join(out) + "\n"


def validate_output(md: str, rule: Rule, run: Run, verdicts: dict) -> None:
    """出力を書く前の検査（全セルの判定が描画され、欠測表示や None が混入していないこと）。"""
    if "None" in md or "nan" in md.lower().split():
        raise LogIntegrityError("出力に None／nan が混入している")
    for pid in rule.probes:
        if f"| {pid} |" not in md:
            raise LogIntegrityError(f"出力にプローブ {pid} の行が無い")
    for key, v in verdicts.items():
        if v.word not in VERDICTS or (v.word == V_INDET and v.code not in INDET_CODES):
            raise LogIntegrityError(f"出力に未知の判定語・理由コード: {key} {v}")


def write_atomic(path: Path, text: str) -> None:
    """一時ファイルへ書いてから `os.replace` する（途中状態を公開しない）。"""
    path = path.resolve()
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


# ------------------------------------------------------------------ self-test

def _synth_cells(pdef: ProbeDef, reject: bool) -> dict:
    """方針に従う基準セル。reject=True は ptxas が拒否し S3 のロードも失敗する典型。"""
    ok = Cell("ok", "-", "synthetic")
    na = Cell("not_applicable_by_design", "-", "policy")
    nr = Cell("not_run", "UPSTREAM_FAILED", "upstream")
    cells = {s: ok for s in STAGES}
    cells["ctl"] = na if pdef.id == "ctl.copy" else ok
    for stage in STAGES:
        if stage != "ctl" and stage_is_na(pdef.policy, stage):
            cells[stage] = na
    if pdef.clause == "R-TMA-SEM":
        cells["sync"] = Cell("ok", "-", "record: status=complete polls=3 class=ELEM_INNER_FIRST")
    if pdef.is_attr:
        cells["verify"] = Cell("ok", "-", "max_threads_per_multiprocessor=1536 max_blocks_per_multiprocessor=32"
                                          " max_shared_memory_per_multiprocessor=131072"
                                          " max_shared_memory_per_block=49152"
                                          " max_shared_memory_per_block_optin=101376")
        return cells
    if reject:
        cells["nvrtc_cubin"] = Cell("rejected", "NVRTC_COMPILE_ERROR", "ptxas: not supported")
        cells["module_load"] = Cell("error", "CUDA_ERROR_INVALID_PTX", "load_module failed")
        for stage in ("launch", "sync", "verify"):
            if not stage_is_na(pdef.policy, stage):
                cells[stage] = nr
    return cells


REJECT_BASE = {"tc5.alloc", "tc5.ld", "tc5.cross", "wgmma.m64n8k16"}


class Model:
    """合成ログのモデル。`write` が実ログと同じ書式でファイルへ書き、
    self-test の各 fixture はこのモデルを編集してから `load` する。"""

    def __init__(self, rule: Rule, dev: bool = False):
        self.rule = rule
        self.dev = dev
        targets = tuple(rule.dev_targets) if dev else tuple(rule.targets)
        self.targets = targets
        self.prov = {"mode": "dev_smoke" if dev else "gb10", "git_head": "a" * 40,
                     "git_clean": "1", "start_utc": "2026-10-01T00:00:00Z",
                     "end_utc": "2026-10-01T01:00:00Z"}
        self.compile_env = {"v": 1, "kind": "env", "phase": "compile", "probe": "-", "target": "-",
                            "device": "none", "cc": "none", "nvrtc": "13.0"}
        self.compile_cells: dict = {}
        self.compile_exit = 0
        self.compile_keep: int | None = None   # 先頭 N セルだけ書く（打ち切りの再現）
        self.compile_env_present = True
        self.execs: dict = {}
        self.legacy: dict = {}
        for pid, p in rule.probes.items():
            reject = pid in REJECT_BASE
            for t in targets:
                cells = _synth_cells(p, reject)
                self.execs[(pid, t)] = {
                    "env": {"v": 1, "kind": "env", "phase": "exec", "probe": pid, "target": t,
                            "device": "GB10", "cc": "8.6" if dev else "12.1", "nvrtc": "13.0"},
                    "cells": cells, "exit": 0, "done": True}
                if not p.is_attr:
                    for stage in ("nvrtc_ptx", "nvrtc_cubin"):
                        self.compile_cells[(pid, t, stage)] = cells[stage]
            if not p.is_attr:
                self.compile_cells[(pid, "home", "nvrtc_cubin")] = Cell("ok", "-", "home")
                hop_rejected = pid.startswith("tc5.") or pid in ("mma.f8f6f4.m16n8k32",
                                                                 "mma.block_scale.m16n8k64")
                self.compile_cells[(pid, "hopper", "nvrtc_cubin")] = (
                    Cell("rejected", "NVRTC_COMPILE_ERROR", "hopper") if hop_rejected else Cell("ok", "-", "hopper"))
        self.dump_exit: int | None = 0
        for lname in ([] if dev else LEGACY_NAMES):
            _pid, fixed, kind = LEGACY_NAMES[lname]
            if kind == "snr":
                lines = [f"SETMAXNREG_PROBE_RESULT stage=execute kernel=k arch={fixed} result=success"]
            elif kind == "tma_compile":
                lines = [f"tma_compile_probe variant={v} arch={a} result=success"
                         for v in ("cluster", "cta") for a in ("compute_121", "compute_121a", "compute_121f")]
            else:
                v = "cluster" if kind == "tma_exec_cluster" else "cta"
                lines = [f"tma_compile_probe variant={v} arch=compute_121 result=success (selected for execution probe)",
                         f"tma_execution_probe variant={v} arch=compute_121 result=success tile=16x16 bitwise_match=true"]
            self.legacy[lname] = {"exit": 0, "lines": lines}

    def exec_arch(self, pid, t, stage) -> str:
        return expected_arch(self.rule, self.rule.probes[pid], t, stage)

    def compile_arch(self, pid, t, stage) -> str:
        p = self.rule.probes[pid]
        if t == "home":
            return p.home
        if t == "hopper":
            return HOPPER_ARCH
        return expected_arch(self.rule, p, t, stage)

    def _cell_line(self, pid, t, phase, stage, c: Cell, arch="compute_121") -> str:
        rec = {"v": 1, "kind": "cell", "phase": phase, "probe": pid, "target": t, "arch": arch,
               "stage": stage, "status": c.status, "code": c.code, "detail": c.detail}
        return PREFIX_JSON + json.dumps(rec, ensure_ascii=True)

    def write(self, root: Path) -> None:
        (root / "exec").mkdir(parents=True, exist_ok=True)
        (root / "env_info.txt").write_text(
            "".join(f"{k}={v}\n" for k, v in self.prov.items()), encoding="utf-8")
        if self.dump_exit is not None:
            (root / "device_attributes_dump.log").write_text(
                "dump\n" + PREFIX_PROC + json.dumps({"v": 1, "kind": "proc", "phase": "dump", "probe": "-",
                                                    "target": "-", "exit": self.dump_exit}) + "\n",
                encoding="utf-8")
        lines = [PREFIX_JSON + json.dumps(self.compile_env)] if self.compile_env_present else []
        items = list(self.compile_cells.items())
        if self.compile_keep is not None:
            items = items[:self.compile_keep]
        for (pid, t, stage), c in items:
            lines.append(self._cell_line(pid, t, "compile", stage, c, self.compile_arch(pid, t, stage)))
        if self.compile_keep is None:
            lines.append(PREFIX_JSON + json.dumps({"v": 1, "kind": "done", "phase": "compile", "probe": "-",
                                                   "target": "-", "cells": len(self.compile_cells),
                                                   "mismatch": 0}))
        lines.append(PREFIX_PROC + json.dumps({"v": 1, "kind": "proc", "phase": "compile", "probe": "-",
                                               "target": "-", "exit": self.compile_exit}))
        (root / "compile.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
        for (pid, t), e in self.execs.items():
            ls = []
            if e["env"] is not None:
                ls.append(PREFIX_JSON + json.dumps(e["env"]))
            for stage in STAGES:
                if stage in e["cells"]:
                    ls.append(self._cell_line(pid, t, "exec", stage, e["cells"][stage],
                                              self.exec_arch(pid, t, stage)))
            if e["done"]:
                n_mis = sum(1 for c in e["cells"].values() if c.status == "mismatch")
                ls.append(PREFIX_JSON + json.dumps({"v": 1, "kind": "done", "phase": "exec", "probe": pid,
                                                    "target": t, "cells": len(e["cells"]),
                                                    "mismatch": n_mis}))
            ls.append(PREFIX_PROC + json.dumps({"v": 1, "kind": "proc", "phase": "exec", "probe": pid,
                                                "target": t, "exit": e["exit"]}))
            (root / "exec" / f"{pid}@{t}.log").write_text("\n".join(ls) + "\n", encoding="utf-8")
        for lname, d in self.legacy.items():
            ls = list(d["lines"])
            ls.append(PREFIX_PROC + json.dumps({"v": 1, "kind": "proc", "phase": "legacy", "probe": lname,
                                                "target": "-", "exit": d["exit"]}))
            (root / f"legacy-{lname}.log").write_text("\n".join(ls) + "\n", encoding="utf-8")


def _load(model: Model, mutate_files=None):
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        model.write(root)
        if mutate_files:
            mutate_files(root)
        rule = model.rule
        run = load_run(rule, root)
        g0, verdicts = compute_verdicts(rule, run)
        return run, g0, verdicts


def _expect(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _expect_integrity_error(fn, label: str) -> None:
    try:
        fn()
    except LogIntegrityError:
        return
    raise AssertionError(f"{label}: LogIntegrityError が送出されなかった")


def _verdict(model: Model, pid: str, t: str = "compute_121", mutate_files=None) -> Verdict:
    _, _, verdicts = _load(model, mutate_files)
    return verdicts[(pid, t)]


def fx_g0(rule):
    m = Model(rule)
    _, g0, v = _load(m)
    _expect(g0 == [] and v[("mma.tf32.m16n8k8", "compute_121")].word == V_OK, "G0 成立の基準が成立でない")
    m = Model(rule)
    m.execs[("ctl.copy", "compute_121")]["env"]["cc"] = "8.6"
    _, g0, v = _load(m)
    _expect(g0 == ["G0_CC"], f"G0_CC を検出できない: {g0}")
    _expect(all(x.word == V_INDET and x.code == "G0_FAILED" for x in v.values()), "G0 不成立で全セルが判定不能にならない")


def fx_g0_ind(rule):
    m = Model(rule, dev=True)
    _, g0, v = _load(m)
    _expect({"G0_DEV_MODE", "G0_DEV_TARGET", "G0_CC"} <= set(g0), f"開発機スモークの G0 理由が不足: {g0}")
    _expect(all(x.word == V_INDET and x.code == "G0_FAILED" for x in v.values()), "スモークで全セルが判定不能にならない")
    m = Model(rule)
    m.prov["git_clean"] = "0"
    _, g0, _ = _load(m)
    _expect(g0 == ["G0_DIRTY_TREE"], f"dirty tree: {g0}")
    m = Model(rule)
    m.prov["git_head"] = "zz"
    _, g0, _ = _load(m)
    _expect(g0 == ["G0_PROVENANCE"], f"provenance: {g0}")
    m = Model(rule)
    m.compile_env["nvrtc"] = "unavailable"
    m.compile_cells = {}
    m_run = None
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        m.write(root)
        # NVRTC 不在の compile.log は done なし・exit 非 0 の形になる（セルなし）。
        lines = [PREFIX_JSON + json.dumps(m.compile_env),
                 PREFIX_PROC + json.dumps({"v": 1, "kind": "proc", "phase": "compile", "probe": "-",
                                           "target": "-", "exit": 101})]
        (root / "compile.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
        m_run = load_run(rule, root)
    _expect(evaluate_g0(m_run) == ["G0_NVRTC_MISSING"], "NVRTC 不在の G0")


def fx_stage(rule):
    m = Model(rule)
    c = m.execs[("mma.tf32.m16n8k8", "compute_121")]["cells"]
    c["nvrtc_ptx"] = Cell("rejected", "NVRTC_COMPILE_ERROR", "x")
    for s in ("module_load", "launch", "sync", "verify"):
        c[s] = Cell("not_run", "UPSTREAM_FAILED", "x")
    m.compile_cells[("mma.tf32.m16n8k8", "compute_121", "nvrtc_ptx")] = c["nvrtc_ptx"]
    _expect(_verdict(m, "mma.tf32.m16n8k8").word == V_NVRTC, "S1 拒否 → NVRTC 拒否")


def fx_stage_ind(rule):
    m = Model(rule)
    m.compile_cells[("mma.tf32.m16n8k8", "compute_121", "nvrtc_cubin")] = Cell("rejected", "X", "x")
    v = _verdict(m, "mma.tf32.m16n8k8")
    _expect(v.word == V_INDET and v.code == "COMPILE_EXEC_DISAGREE", f"compile/exec 食い違い: {v}")
    m = Model(rule)
    c = m.execs[("mma.tf32.m16n8k8", "compute_121")]["cells"]
    c["nvrtc_cubin"] = Cell("rejected", "X", "x")
    m.compile_cells[("mma.tf32.m16n8k8", "compute_121", "nvrtc_cubin")] = c["nvrtc_cubin"]
    v = _verdict(m, "mma.tf32.m16n8k8")
    _expect(v.code == "OFFLINE_JIT_DISAGREE", f"S2 拒否かつ S3 成功: {v}")
    m = Model(rule)
    c = m.execs[("mma.tf32.m16n8k8", "compute_121")]["cells"]
    c["module_load"] = Cell("error", "CUDA_ERROR_UNSUPPORTED_PTX_VERSION", "x")
    for s in ("launch", "sync", "verify"):
        c[s] = Cell("not_run", "UPSTREAM_FAILED", "x")
    v = _verdict(m, "mma.tf32.m16n8k8")
    _expect(v.code == "UNSUPPORTED_PTX_VERSION", f"UNSUPPORTED_PTX_VERSION: {v}")
    m = Model(rule)
    m.execs[("mma.tf32.m16n8k8", "compute_121")]["cells"]["nvrtc_ptx"] = Cell("unavailable", "NVRTC_UNAVAILABLE", "x")
    _expect_integrity_error(lambda: _load(m), "S1 unavailable なのに後段 ok は連鎖矛盾")


def fx_ctl(rule):
    m = Model(rule)
    _expect(_verdict(m, "ctl.copy").word == V_OK, "ctl.copy 自身は成立")


def fx_ctl_ind(rule):
    m = Model(rule)
    m.execs[("mma.tf32.m16n8k8", "compute_121f")]["cells"]["ctl"] = Cell("error", "TARGET_UNSUPPORTED", "toolchain")
    v = _verdict(m, "mma.tf32.m16n8k8", "compute_121f")
    _expect(v.word == V_INDET and v.code == "TARGET_UNSUPPORTED", f"TARGET_UNSUPPORTED: {v}")
    m.execs[("mma.tf32.m16n8k8", "compute_121f")]["cells"]["ctl"] = Cell("error", "CTL_FAILED", "x")
    _expect(_verdict(m, "mma.tf32.m16n8k8", "compute_121f").code == "CTL_FAILED", "CTL_FAILED")
    # 対照段（env 行だけで打ち切られたプロセス）のハング・異常終了も判定不能（P1）。
    for pid in ("mma.tf32.m16n8k8", "ctl.copy"):
        for exit_code, want in ((124, "TIMEOUT"), (139, "PROCESS_FAILED")):
            m = Model(rule)
            e = m.execs[(pid, "compute_121")]
            e["cells"], e["done"], e["exit"] = {}, False, exit_code
            v = _verdict(m, pid)
            _expect(v.word == V_INDET and v.code == want, f"{pid} ctl 欠落 exit={exit_code}: {v}")


def fx_home(rule):
    m = Model(rule)
    v = _verdict(m, "wgmma.m64n8k16")
    _expect(v.word == V_PTXAS, f"home 受理のうえ sm_121 拒否 → ptxas 拒否: {v}")


def fx_home_ind(rule):
    m = Model(rule)
    m.compile_cells[("wgmma.m64n8k16", "home", "nvrtc_cubin")] = Cell("rejected", "X", "home")
    v = _verdict(m, "wgmma.m64n8k16")
    _expect(v.code == "HOME_REJECTED", f"HOME_REJECTED: {v}")
    m = Model(rule)
    m.compile_cells[("wgmma.m64n8k16", "home", "nvrtc_cubin")] = Cell("error", "X", "home")
    _expect(_verdict(m, "wgmma.m64n8k16").code == "HOME_NOT_OK", "HOME_NOT_OK")
    # compile プロセスの打ち切り: 出たセルは採用し、欠測に依存する判定だけを判定不能にする。
    for exit_code, want in ((124, "TIMEOUT"), (139, "PROCESS_FAILED")):
        m = Model(rule)
        m.compile_exit, m.compile_keep = exit_code, 10   # ctl.copy（8 セル）は完全・macro.arch は途中
        run, g0, v = _load(m)
        _expect(run.compile_failure == want and g0 == [], f"compile 打ち切り exit={exit_code}: {run.compile_failure} {g0}")
        _expect(v[("ctl.copy", "compute_121")].word == V_OK, "完全に出たセルの判定は継続する")
        _expect(v[("wgmma.m64n8k16", "compute_121")].code == want, f"欠測に依存する判定は {want}")
        _expect(v[("attr.limits", "compute_121")].word == V_OK, "compile に依存しない attr は継続する")
    # env 行の前に打ち切られた compile は NVRTC の存在を確認できず G0 不成立。
    m = Model(rule)
    m.compile_exit, m.compile_keep, m.compile_env_present = 124, 0, False
    run, g0, _ = _load(m)
    _expect(g0 == ["G0_NVRTC_MISSING"], f"compile の env 欠落: {g0}")
    # exit 0 なのに欠落、完了しているのに非 0 は完全性違反。
    m = Model(rule)
    m.compile_keep = 10
    _expect_integrity_error(lambda: _load(m), "compile: exit 0 なのにセルが欠落")
    m = Model(rule)
    m.compile_exit = 124
    _expect_integrity_error(lambda: _load(m), "compile: 完了しているのに exit 124")


def fx_tc5(rule):
    v = _verdict(Model(rule), "tc5.alloc")
    _expect(v.word == V_PTXAS, f"tc5.alloc の想定どおりの拒否: {v}")


def fx_tc5_ind(rule):
    m = Model(rule)
    m.execs[("tc5.alloc", "compute_121")]["cells"] = _synth_cells(rule.probes["tc5.alloc"], False)
    m.compile_cells[("tc5.alloc", "compute_121", "nvrtc_cubin")] = Cell("ok", "-", "x")
    v = _verdict(m, "tc5.alloc")
    _expect(v.code == "UNEXPECTED_ACCEPT", f"想定外の受理: {v}")


def fx_guide(rule):
    claims = load_claims(CLAIMS_PATH.read_text(encoding="utf-8"))
    m = Model(rule)
    run, g0, verdicts = _load(m)
    got = {c["id"]: evaluate_claim(rule, run, verdicts, g0, c) for c in claims}
    _expect(got["C01-warps-per-sm"][0] == G_MATCH, f"C01: {got['C01-warps-per-sm']}")
    _expect(got["C04-smem-per-block"][0] == G_MATCH, f"C04: {got['C04-smem-per-block']}")
    _expect(got["C06-tcgen05-sm100plus"][0] == G_DIFF, f"tc5 の基準（ptxas 拒否）は verdict_is:成立 と不一致: {got['C06-tcgen05-sm100plus']}")
    _expect(got["C08-tcgen05-unavailable"][0] == G_MATCH, f"C08: {got['C08-tcgen05-unavailable']}")
    m = Model(rule)
    m.execs[("attr.limits", "compute_121")]["cells"]["verify"] = Cell(
        "ok", "-", "max_threads_per_multiprocessor=1536 max_blocks_per_multiprocessor=24")
    run, g0, verdicts = _load(m)
    # 3 target のうち 1 つだけ違えば不一致（全 target を要求）。
    _expect(evaluate_claim(rule, run, verdicts, g0, claims[1])[0] == G_DIFF, "blocks/SM の不一致を検出できない")


def fx_guide_ind(rule):
    m = Model(rule)
    run, g0, verdicts = _load(m)
    word, code, _ = evaluate_claim(rule, run, verdicts, g0, {
        "id": "X", "source": "x:1", "quote": "q", "probe": "-", "key": "-", "compare": "eq:1"})
    _expect((word, code) == ("判定不能", "GUIDE_NO_PROBE"), "プローブ無しの claim は判定不能")
    run2, g02, v2 = _load(Model(rule, dev=True))
    word, code, _ = evaluate_claim(rule, run2, v2, g02, {
        "id": "X", "source": "x:1", "quote": "q", "probe": "attr.limits", "key": "warp_size",
        "compare": "eq:32"})
    _expect((word, code) == ("判定不能", "GUIDE_UNMEASURED"), "G0 不成立の claim は判定不能")
    rows = load_claims(CLAIMS_PATH.read_text(encoding="utf-8"))
    check_claim_quotes(rows, REPO_ROOT)
    bad = [dict(rows[0], quote="存在しない引用")]
    _expect_integrity_error(lambda: check_claim_quotes(bad, REPO_ROOT), "引用の実在照合")


def fx_mma(rule):
    _expect(_verdict(Model(rule), "mma.tf32.m16n8k8").word == V_OK, "mma 成立")
    m = Model(rule)
    c = m.execs[("mma.tf32.m16n8k8", "compute_121")]["cells"]
    c["verify"] = Cell("mismatch", "MISMATCH", "first_index=1")
    m.execs[("mma.tf32.m16n8k8", "compute_121")]["exit"] = 101
    v = _verdict(m, "mma.tf32.m16n8k8")
    _expect(v.word == V_MISMATCH, f"layout=verified の不一致は結果不一致: {v}")
    _expect(_verdict(Model(rule), "mma.s8.m16n8k32").word == V_ACCEPT_ONLY, "accept_only は受理のみ")


def fx_mma_ind(rule):
    m = Model(rule)
    m.execs[("mma.stmatrix.x4", "compute_121")]["cells"]["verify"] = Cell("mismatch", "MISMATCH", "x")
    m.execs[("mma.stmatrix.x4", "compute_121")]["exit"] = 101
    v = _verdict(m, "mma.stmatrix.x4")
    _expect(v.code == "LAYOUT_UNVERIFIED", f"LAYOUT_UNVERIFIED: {v}")


def fx_clu(rule):
    _expect(_verdict(Model(rule), "clu.dims2").word == V_OK, "clu.dims2 成立")


def fx_clu_ind(rule):
    m = Model(rule)
    e = m.execs[("clu.dsmem", "compute_121")]
    e["cells"] = {k: v for k, v in e["cells"].items() if k in ("ctl", "nvrtc_ptx", "nvrtc_cubin", "module_load", "launch")}
    e["done"], e["exit"] = False, 124
    v = _verdict(m, "clu.dsmem")
    _expect(v.code == "TIMEOUT", f"barrier.cluster のハング → TIMEOUT: {v}")
    m = Model(rule)
    e = m.execs[("clu.dsmem", "compute_121")]
    e["cells"] = {k: v for k, v in e["cells"].items() if k in ("ctl", "nvrtc_ptx")}
    e["done"], e["exit"] = False, 139
    _expect(_verdict(m, "clu.dsmem").code == "PROCESS_FAILED", "異常終了 → PROCESS_FAILED")


def fx_snr(rule):
    _expect(_verdict(Model(rule), "snr.dec").word == V_OK, "snr.dec 成立（legacy と一致）")


def fx_snr_ind(rule):
    m = Model(rule)
    m.legacy["setmaxnreg_probe_dec_base_real_device"]["lines"] = [
        "SETMAXNREG_PROBE_RESULT stage=launch kernel=k arch=compute_121 result=failed detail=X"]
    v = _verdict(m, "snr.dec")
    _expect(v.code == "LEGACY_CONTRADICTION", f"LEGACY_CONTRADICTION: {v}")


def fx_hopper(rule):
    m = Model(rule)
    run, g0, verdicts = _load(m)
    md = render(rule, run, g0, verdicts, load_claims(CLAIMS_PATH.read_text(encoding="utf-8")))
    _expect("| wgmma.m64n8k16 | sm_90a | ok | rejected/rejected/rejected | " in md, "Hopper 表の行")
    _expect("Hopper のみ受理" in md, "方向の分類（Hopper のみ受理）")
    _expect("両方で拒否" in md, "方向の分類（両方で拒否）")
    _expect("要確認" in md, "PTX ISA 節番号は要確認")


def fx_hopper_ind(rule):
    m = Model(rule)
    m.compile_cells[("wgmma.m64n8k16", "hopper", "nvrtc_cubin")] = Cell("ok", "-", "x")
    v = _verdict(m, "wgmma.m64n8k16")
    _expect(v.word == V_PTXAS, "home=hopper ともに ok なら通常判定（基準では hopper も ok）")
    m.compile_cells[("wgmma.m64n8k16", "hopper", "nvrtc_cubin")] = Cell("rejected", "X", "x")
    v = _verdict(m, "wgmma.m64n8k16")
    _expect(v.code == "HOPPER_HOME_DISAGREE", f"HOPPER_HOME_DISAGREE: {v}")


def fx_legacy(rule):
    fx_tma_legacy(rule)
    _expect(_verdict(Model(rule), "snr.incdec", "compute_121a").word == V_OK, "legacy 一致で成立")


def fx_legacy_ind(rule):
    m = Model(rule)
    m.legacy["setmaxnreg_probe_incdec_accel_real_device"]["lines"] = [
        "SETMAXNREG_PROBE_RESULT stage=module_load kernel=k arch=compute_121a result=failed detail=X"]
    v = _verdict(m, "snr.incdec", "compute_121a")
    _expect(v.code == "LEGACY_CONTRADICTION", f"legacy ロード失敗 vs 新 run_ok: {v}")
    m = Model(rule)
    m.legacy["setmaxnreg_probe_incdec_accel_real_device"]["lines"].append(
        "SETMAXNREG_PROBE_RESULT stage=launch kernel=k arch=compute_121a result=failed")
    _expect_integrity_error(lambda: _load(m), "legacy の終端が 2 つ")
    # 異常終了（124 以外の非 0）は比較不能。結論が残っている・exit 0 で結論が無いのは完全性違反。
    for exit_code in (139, 101):
        m = Model(rule)
        m.legacy["setmaxnreg_probe_dec_base_real_device"].update(lines=[], exit=exit_code)
        v = _verdict(m, "snr.dec")
        _expect(v.code == "LEGACY_INCONCLUSIVE", f"legacy exit {exit_code}: {v}")
    for exit_code, want in ((139, "process_failed"), (101, "process_failed"), (124, "timeout")):
        got = classify_legacy("", "unit", exit_code, "compute_121")
        _expect(got == want, f"classify_legacy 空出力 exit {exit_code}: {got}")
    m = Model(rule)
    m.legacy["setmaxnreg_probe_dec_base_real_device"]["exit"] = 1
    _expect_integrity_error(lambda: _load(m), "legacy: 結論があるのに exit 1")
    m = Model(rule)
    m.legacy["setmaxnreg_probe_dec_base_real_device"]["lines"] = []
    _expect_integrity_error(lambda: _load(m), "legacy: exit 0 なのに結論が無い")
    # legacy が inconclusive／timeout のときは矛盾ではなく比較不能（別コード）。
    m = Model(rule)
    m.legacy["setmaxnreg_probe_dec_base_real_device"]["lines"] = [
        "SETMAXNREG_PROBE_RESULT stage=nvrtc_compile kernel=k arch=compute_121 result=inconclusive detail=X"]
    v = _verdict(m, "snr.dec")
    _expect(v.word == V_INDET and v.code == "LEGACY_INCONCLUSIVE", f"legacy inconclusive: {v}")
    m = Model(rule)
    m.legacy["setmaxnreg_probe_dec_base_real_device"].update(lines=[], exit=124)
    v = _verdict(m, "snr.dec")
    _expect(v.code == "LEGACY_INCONCLUSIVE", f"legacy timeout: {v}")
    # official で legacy が 1 件でも欠ければ黙ってスキップせず完全性違反。
    m = Model(rule)
    del m.legacy["setmaxnreg_probe_dec_base_real_device"]
    _expect_integrity_error(lambda: _load(m), "official の legacy 欠落")
    m = Model(rule)
    m.legacy.clear()
    _expect_integrity_error(lambda: _load(m), "official の legacy 全欠落")
    m = Model(rule, dev=True)
    m.legacy["setmaxnreg_probe_dec_base_real_device"] = {"exit": 0, "lines": [
        "SETMAXNREG_PROBE_RESULT stage=execute kernel=k arch=compute_121 result=success"]}
    _expect_integrity_error(lambda: _load(m), "dev で legacy がある")
    _load(Model(rule, dev=True))  # 陽性: dev は legacy 0 件で読める


def fx_common(rule):
    run, g0, verdicts = _load(Model(rule))
    _expect(g0 == [], "統合: 基準は G0 成立")
    md = render(rule, run, g0, verdicts, load_claims(CLAIMS_PATH.read_text(encoding="utf-8")))
    validate_output(md, rule, run, verdicts)
    with tempfile.TemporaryDirectory() as d:
        dest = Path(d) / "agg.md"
        write_atomic(dest, md)
        _expect(dest.read_text(encoding="utf-8") == md, "atomic 書き出し")
        _expect([p.name for p in Path(d).iterdir()] == ["agg.md"], "一時ファイルが残っている")
    # 全ログ無し → 未実測（exit 0 の中間状態）。
    with tempfile.TemporaryDirectory() as d:
        r0 = load_run(rule, Path(d))
        _expect(not r0.measured, "ログ無しは未実測")
        _, v0 = compute_verdicts(rule, r0)
        _expect(all(x.word == V_UNMEASURED for x in v0.values()), "未実測の語")


def fx_common_ind(rule):
    """入力の完全性違反はすべて LogIntegrityError（fail-closed）。"""
    base = Model(rule)

    def patch(rel, fn):
        def go(root):
            p = root / rel
            p.write_text(fn(p.read_text(encoding="utf-8")), encoding="utf-8")
        return go

    for label, text in (("NaN", '{"a": NaN}'), ("Infinity", '{"a": Infinity}'),
                        ("-Infinity", '{"a": -Infinity}'), ("小数", '{"a": 1.5}'),
                        ("指数表記", '{"a": 1e5}'), ("巨大整数", '{"a": 9223372036854775808}'),
                        ("キー重複", '{"a": 1, "a": 2}'), ("配列", "[1]"), ("文字列", '"x"')):
        _expect_integrity_error(lambda text=text: strict_json(text, "unit"), f"strict_json: {label}")
    _expect(strict_json('{"a": 9223372036854775807}', "unit") == {"a": 9223372036854775807}, "境界の整数は許容")
    dup: dict = {}
    register_once(dup, "k", 1, "unit")
    _expect_integrity_error(lambda: register_once(dup, "k", 1, "unit"), "register_once は同値でも拒否")
    _expect_integrity_error(lambda: parse_env_info("mode=gb10\nmode=gb10\n"), "env_info の重複キー")
    _expect_integrity_error(lambda: require_exact_keys({"a"}, {"a", "b"}, "unit"), "require_exact_keys の欠落")
    _expect_integrity_error(lambda: require_exact_keys({"a", "c"}, {"a"}, "unit"), "require_exact_keys の余剰")
    _expect_integrity_error(lambda: check_masked("x /home/bob/y", "unit"), "check_masked")

    first_exec = "exec/ctl.copy@compute_121.log"
    cases = {
        "重複セル": patch(first_exec, lambda t: t + t.splitlines()[1] + "\n"),
        "壊れた JSON": patch(first_exec, lambda t: t.replace('"v": 1', '"v": 1,,', 1)),
        "NaN": patch(first_exec, lambda t: t.replace('"v": 1, "kind": "cell"', '"v": NaN, "kind": "cell"', 1)),
        "浮動小数点": patch(first_exec, lambda t: t.replace('"v": 1, "kind": "cell"', '"v": 1.5, "kind": "cell"', 1)),
        "巨大整数": patch(first_exec, lambda t: t.replace('"v": 1, "kind": "cell"', '"v": 99999999999999999999, "kind": "cell"', 1)),
        "未知 stage": patch(first_exec, lambda t: t.replace('"stage": "nvrtc_ptx"', '"stage": "bogus"', 1)),
        "未知 status": patch(first_exec, lambda t: t.replace('"status": "ok"', '"status": "weird"', 1)),
        "未知フィールド": patch(first_exec, lambda t: t.replace('"kind": "cell"', '"kind": "cell", "extra": 1', 1)),
        "未マスクパス": patch(first_exec, lambda t: t + "/home/alice/x\n"),
        "proc 欠落": patch(first_exec, lambda t: "\n".join(l for l in t.splitlines() if not l.startswith(PREFIX_PROC)) + "\n"),
        "識別子の行頭外": patch(first_exec, lambda t: "junk " + t),
        "env_info 欠落キー": patch("env_info.txt", lambda t: t.replace("end_utc", "end_x")),
    }
    for label, fn in cases.items():
        _expect_integrity_error(lambda fn=fn: _load(base, fn), label)

    def drop_file(root):
        (root / "exec" / "ctl.copy@compute_121.log").unlink()
    _expect_integrity_error(lambda: _load(base, drop_file), "exec ファイルの欠落")

    def extra_file(root):
        (root / "exec" / "bogus.probe@compute_121.log").write_text("x\n", encoding="utf-8")
    _expect_integrity_error(lambda: _load(base, extra_file), "未知プローブの exec ファイル")

    def mixed(root):
        src = root / "exec" / "ctl.copy@compute_121.log"
        dst = root / "exec" / "ctl.copy@compute_86.log"
        dst.write_text(src.read_text(encoding="utf-8").replace("compute_121", "compute_86"), encoding="utf-8")
    _expect_integrity_error(lambda: _load(base, mixed), "正式 target と開発機 target の混在")
    m = Model(rule)
    m.execs[("ctl.copy", "compute_121")]["exit"] = 124
    _expect_integrity_error(lambda: _load(m), "exit 124 なのに全セル完了")
    m = Model(rule)
    m.execs[("ctl.copy", "compute_121")]["done"] = False
    _expect_integrity_error(lambda: _load(m), "exit 0 で done 欠落")
    m = Model(rule)
    m.execs[("mma.tf32.m16n8k8", "compute_121")]["cells"]["launch"] = Cell("not_run", "UPSTREAM_FAILED", "x")
    _expect_integrity_error(lambda: _load(m), "前段 ok なのに not_run")
    m = Model(rule)
    m.execs[("mma.s8.m16n8k32", "compute_121")]["cells"]["launch"] = Cell("ok", "-", "x")
    _expect_integrity_error(lambda: _load(m), "accept_only の launch は設計上不実施で固定")
    m = Model(rule)
    m.compile_cells.pop(("mma.tf32.m16n8k8", "home", "nvrtc_cubin"))
    _expect_integrity_error(lambda: _load(m), "compile のキー欠落")
    # device_attributes_dump.log: 欠測・exit 非 0・proc 欠落は黙認しない（記録用でも必須）。
    m = Model(rule)
    m.dump_exit = None
    _expect_integrity_error(lambda: _load(m), "dump ログの欠落")
    m = Model(rule)
    m.dump_exit = 124   # 打ち切りは欠測として報告し、集計は続ける（判定に使わない）
    run, g0, _ = _load(m)
    _expect(run.dump_status == "TIMEOUT" and g0 == [], f"dump timeout は非致命: {run.dump_status} {g0}")
    m.dump_exit = 139
    _expect(_load(m)[0].dump_status == "PROCESS_FAILED", "dump 異常終了は非致命")
    _expect_integrity_error(lambda: _load(base, patch("device_attributes_dump.log",
        lambda t: "\n".join(l for l in t.splitlines() if l.startswith(PREFIX_PROC)) + "\n")),
        "dump: exit 0 なのに出力が無い")
    _expect_integrity_error(lambda: _load(base, patch("device_attributes_dump.log", lambda t: "\n".join(
        l for l in t.splitlines() if not l.startswith(PREFIX_PROC)) + "\n")), "dump の proc 欠落")
    # arch の期待値照合（exec・compile）。
    run_bad_arch = patch("exec/mma.tf32.m16n8k8@compute_121.log", lambda t: t.replace(
        '"arch": "compute_121", "stage": "module_load"', '"arch": "compute_86", "stage": "module_load"'))
    _expect_integrity_error(lambda: _load(base, run_bad_arch), "exec セルの arch 不一致")
    _expect_integrity_error(lambda: _load(base, patch("exec/tc5.cross@compute_121.log", lambda t: t.replace(
        '"arch": "compute_100a"', '"arch": "compute_121"'))), "固定アーキ（tc5.cross）の arch 不一致")
    _expect_integrity_error(lambda: _load(base, patch("compile.log", lambda t: t.replace(
        '"target": "home", "arch": "sm_80"', '"target": "home", "arch": "sm_90a"', 1))), "home の arch 不一致")
    _expect_integrity_error(lambda: _load(base, patch("compile.log", lambda t: t.replace(
        '"target": "hopper", "arch": "sm_90a"', '"target": "hopper", "arch": "sm_80"', 1))), "hopper の arch 不一致")
    _expect_integrity_error(lambda: _load(base, patch("compile.log", lambda t: t.replace(
        '"target": "compute_121", "arch": "sm_121"', '"target": "compute_121", "arch": "sm_121a"', 1))), "S2 の arch 不一致")
    # 重複キー（attr の detail・legacy の行）。
    m = Model(rule)
    m.execs[("attr.limits", "compute_121")]["cells"]["verify"] = Cell("ok", "-", "warp_size=32 warp_size=32")
    _expect_integrity_error(lambda: _load(m), "attr detail の重複キー")
    m = Model(rule)
    m.legacy["setmaxnreg_probe_dec_base_real_device"]["lines"] = [
        "SETMAXNREG_PROBE_RESULT stage=execute stage=launch arch=compute_121 result=success"]
    _expect_integrity_error(lambda: _load(m), "legacy 行の重複キー")
    _expect_integrity_error(lambda: parse_kv(["a=1", "a=2"], "unit"), "parse_kv の重複")
    # env_info の値検証・compile の識別子。
    m = Model(rule)
    m.prov["git_clean"] = "2"
    _expect_integrity_error(lambda: _load(m), "git_clean の値")
    m = Model(rule)
    m.prov["end_utc"] = "yesterday"
    _expect_integrity_error(lambda: _load(m), "end_utc の書式")
    _expect_integrity_error(lambda: _load(base, patch("compile.log", lambda t: t.replace(
        '"kind": "done", "phase": "compile", "probe": "-"', '"kind": "done", "phase": "compile", "probe": "x"'))), "compile done の probe")
    _expect_integrity_error(lambda: _load(base, patch("legacy-setmaxnreg_probe_dec_base_real_device.log",
        lambda t: t.replace('"target": "-"', '"target": "compute_121"'))), "legacy proc の target")
    # PROCESS 行の timeout 書式。
    _expect_integrity_error(lambda: process_timeout("compile"), "timeout 欠落")
    _expect_integrity_error(lambda: process_timeout("exec_matrix timeout=0"), "timeout=0")
    _expect(process_timeout("compile timeout=900") == 900 and process_timeout("env_info") is None, "timeout 解析")
    # 一部だけ欠けているログ（env_info があって compile.log が無い）。
    def drop_compile(root):
        (root / "compile.log").unlink()
    _expect_integrity_error(lambda: _load(base, drop_compile), "compile.log の欠落")


def _break(m, pid, stage="verify", status="mismatch", code="MISMATCH", exit_code=101, t="compute_121"):
    """プローブの 1 段を失敗させたモデル編集（連鎖の整合を保つ: mismatch は exit 101・完走）。"""
    e = m.execs[(pid, t)]
    e["cells"][stage] = Cell(status, code, "x")
    e["exit"] = exit_code


def _legacy_cta_mismatch(m):
    """base_cta を壊すフィクスチャでは、legacy（cta）も不一致にして整合させる。"""
    m.legacy["tma_probe_real_device@tma_execution_probe_cta"] = {"exit": 101, "lines": [
        "tma_compile_probe variant=cta arch=compute_121 result=success (selected for execution probe)",
        "TMA 転送結果が期待するタイル"]}


def fx_tma_base(rule):
    _expect(_verdict(Model(rule), "tma.base_cta").word == V_OK, "tma.base_cta 成立")
    _expect(_verdict(Model(rule), "tma.base_cluster").word == V_OK, "tma.base_cluster 成立")
    # 自身の不一致はそのまま結果不一致（判定不能にしない）。
    m = Model(rule)
    _break(m, "tma.base_cta")
    _legacy_cta_mismatch(m)
    _expect(_verdict(m, "tma.base_cta").word == V_MISMATCH, "base_cta 自身の不一致")


def fx_tma_base_ind(rule):
    m = Model(rule)
    _break(m, "tma.base_cta")
    _legacy_cta_mismatch(m)
    for pid in ("tma.coord", "tma.store", "tma.bulk_cta", "tma.multicast", "tma.swz64"):
        v = _verdict(m, pid)
        _expect(v.word == V_INDET and v.code == "TMA_BASE_NOT_ESTABLISHED", f"{pid}: {v}")
    # base_cluster は依存の対象外。他 target の base_cta が成立なら依存は target 単位で判定する。
    _expect(_verdict(m, "tma.base_cluster").word == V_OK, "base_cluster は依存しない")
    _expect(_verdict(m, "tma.coord", "compute_121a").word == V_ACCEPT_ONLY, "依存は target 単位")


def fx_tma_sem(rule):
    m = Model(rule)
    run, g0, v = _load(m)
    _expect(v[("tma.coord", "compute_121")].word == V_ACCEPT_ONLY, f"tma.coord: {v[('tma.coord', 'compute_121')]}")
    md = render(rule, run, g0, v, load_claims(CLAIMS_PATH.read_text(encoding="utf-8")))
    _expect("## R-TMA 観測" in md and "class=ELEM_INNER_FIRST" in md, "R-TMA 観測表に観測が出る")
    for pid in ("tma.oob_none", "tma.oob_nan", "tma.oob_tx_partial", "tma.oob_neg", "tma.swz32", "tma.swz64", "tma.swz128"):
        _expect(f"| {pid} | compute_121 |" in md, f"{pid} の観測行")
    # 拒否される target（NVRTC／ptxas 拒否）は拒否をそのまま採る。
    m = Model(rule)
    c = m.execs[("tma.swz64", "compute_121")]["cells"]
    c["nvrtc_cubin"] = Cell("rejected", "NVRTC_COMPILE_ERROR", "x")
    c["module_load"] = Cell("error", "CUDA_ERROR_INVALID_PTX", "x")
    c["launch"] = c["sync"] = Cell("not_run", "UPSTREAM_FAILED", "x")
    m.compile_cells[("tma.swz64", "compute_121", "nvrtc_cubin")] = c["nvrtc_cubin"]
    _expect(_verdict(m, "tma.swz64").word == V_PTXAS, "TMA 拒否は ptxas 拒否のまま")


def fx_tma_sem_ind(rule):
    m = Model(rule)
    _break(m, "tma.base_cta", "module_load", "error", "CUDA_ERROR_INVALID_PTX", exit_code=0)
    for s_ in ("launch", "sync", "verify"):
        m.execs[("tma.base_cta", "compute_121")]["cells"][s_] = Cell("not_run", "UPSTREAM_FAILED", "x")
    m.legacy["tma_probe_real_device@tma_execution_probe_cta"] = {"exit": 139, "lines": []}
    v = _verdict(m, "tma.coord")
    _expect(v.code == "TMA_BASE_NOT_ESTABLISHED", f"base がロード失敗なら SEM は判定不能: {v}")
    # 観測の書式不正（status／polls なし）は完全性違反。
    m = Model(rule)
    m.execs[("tma.coord", "compute_121")]["cells"]["sync"] = Cell("ok", "-", "record: class=ELEM_INNER_FIRST")
    _expect_integrity_error(lambda: _load(m), "TMA 観測に status／polls が無い")
    m = Model(rule)
    m.execs[("tma.coord", "compute_121")]["cells"]["sync"] = Cell("ok", "-", "record: status=complete polls=1 polls=2")
    _expect_integrity_error(lambda: _load(m), "TMA 観測の重複キー")
    m = Model(rule)
    m.execs[("tma.coord", "compute_121")]["cells"]["sync"] = Cell("ok", "-", "record: status=weird polls=1")
    _expect_integrity_error(lambda: _load(m), "TMA 観測の status が不正")


def fx_tma_xfer(rule):
    for pid in ("tma.store", "tma.prefetch", "tma.multicast", "tma.bulk_cta", "tma.bulk_cluster"):
        _expect(_verdict(Model(rule), pid).word == V_OK, f"{pid} 成立")


def fx_tma_xfer_ind(rule):
    m = Model(rule)
    _break(m, "clu.dims2", "launch", "error", "CUDA_ERROR_INVALID_CLUSTER_SIZE", exit_code=0)
    for s_ in ("sync", "verify"):
        m.execs[("clu.dims2", "compute_121")]["cells"][s_] = Cell("not_run", "UPSTREAM_FAILED", "x")
    v = _verdict(m, "tma.multicast")
    _expect(v.word == V_INDET and v.code == "CLU_DIMS2_NOT_ESTABLISHED", f"multicast: {v}")
    _expect(_verdict(m, "tma.store").word == V_OK, "store は clu.dims2 に依存しない")
    # 自身が失敗していれば依存より自身の判定を採る。
    m = Model(rule)
    _break(m, "tma.base_cta")
    _legacy_cta_mismatch(m)
    _break(m, "tma.store", "launch", "error", "CUDA_ERROR_ILLEGAL_INSTRUCTION", exit_code=0)
    for s_ in ("sync", "verify"):
        m.execs[("tma.store", "compute_121")]["cells"][s_] = Cell("not_run", "UPSTREAM_FAILED", "x")
    _expect(_verdict(m, "tma.store").word == V_RUNTIME, "自身の実行時エラーを採る")


def fx_tma_legacy(rule):
    """既存 TMA プローブの再実行との整合（R-LEGACY の拡張）。"""
    ok_cta = ["tma_compile_probe variant=cta arch=compute_121 result=success (selected for execution probe)",
              "tma_execution_probe_cta variant=cta arch=compute_121 result=success tile=16x16 bitwise_match=true"]
    cls = classify_tma_legacy("\n".join(ok_cta), "unit", 0, "tma_exec_cta")
    _expect(cls == ("run_ok", "compute_121"), f"成功行: {cls}")
    _expect(classify_tma_legacy("", "unit", 124, "tma_exec_cta") == ("timeout", None), "timeout")
    _expect(classify_tma_legacy("", "unit", 139, "tma_exec_cta") == ("process_failed", None), "異常終了")
    sel = "tma_compile_probe variant=cta arch=compute_121 result=success (selected for execution probe)"
    _expect(classify_tma_legacy(sel + "\nTMA 転送結果が期待するタイル", "unit", 101, "tma_exec_cta") == ("mismatch", "compute_121"), "不一致")
    _expect(classify_tma_legacy("failed to compile for every arch", "unit", 101, "tma_exec_cta") == ("nvrtc_rejected", None), "全 arch 失敗")
    six = [f"tma_compile_probe variant={v} arch={a} result=failure detail=X"
           for v in ("cluster", "cta") for a in ("compute_121", "compute_121a", "compute_121f")]
    _expect(classify_tma_legacy("\n".join(six), "unit", 0, "tma_compile") == ("recorded", None), "compile 記録")
    _expect_integrity_error(lambda: classify_tma_legacy("\n".join(six[:5]), "unit", 0, "tma_compile"), "compile 行が 5 件")
    _expect_integrity_error(lambda: classify_tma_legacy("\n".join(six + six[:1]), "unit", 0, "tma_compile"), "compile 行の重複")
    _expect_integrity_error(lambda: classify_tma_legacy("", "unit", 0, "tma_exec_cta"), "exit 0 で成功行なし")
    _expect_integrity_error(lambda: classify_tma_legacy("\n".join(ok_cta), "unit", 1, "tma_exec_cta"), "成功行があるのに exit 1")
    # 新プローブとの突き合わせ: 一致なら成立のまま、矛盾・比較不能は判定不能。
    _expect(_verdict(Model(rule), "tma.base_cta").word == V_OK, "legacy 一致")
    m = Model(rule)
    m.legacy["tma_probe_real_device@tma_execution_probe_cta"] = {"exit": 101, "lines": [
        sel, "TMA 転送結果が期待するタイル"]}
    v = _verdict(m, "tma.base_cta")
    _expect(v.code == "LEGACY_CONTRADICTION", f"legacy 不一致 vs 新 run_ok: {v}")
    m = Model(rule)
    m.legacy["tma_probe_real_device@tma_execution_probe"] = {"exit": 124, "lines": []}
    v = _verdict(m, "tma.base_cluster")
    _expect(v.code == "LEGACY_INCONCLUSIVE", f"legacy timeout: {v}")
    # 異常終了（process_failed）は成否のみ比較: 新が成立なら矛盾、新も失敗なら整合（判定は新のまま）。
    m = Model(rule)
    m.legacy["tma_probe_real_device@tma_execution_probe"] = {"exit": 139, "lines": []}
    _expect(_verdict(m, "tma.base_cluster").code == "LEGACY_CONTRADICTION", "legacy 異常終了 vs 新 成立")
    m = Model(rule)
    m.legacy["tma_probe_real_device@tma_execution_probe_cta"] = {"exit": 139, "lines": []}
    _break(m, "tma.base_cta", "module_load", "error", "CUDA_ERROR_INVALID_PTX", exit_code=0)
    for s_ in ("launch", "sync", "verify"):
        m.execs[("tma.base_cta", "compute_121")]["cells"][s_] = Cell("not_run", "UPSTREAM_FAILED", "x")
    _expect(_verdict(m, "tma.base_cta").word == V_LOAD, "legacy・新ともに失敗なら整合（新のロード失敗を採る）")


FIXTURES = {
    "G0": (fx_g0, fx_g0_ind),
    "R-STAGE": (fx_stage, fx_stage_ind),
    "R-CTL": (fx_ctl, fx_ctl_ind),
    "R-HOME": (fx_home, fx_home_ind),
    "R-TC5": (fx_tc5, fx_tc5_ind),
    "R-GUIDE": (fx_guide, fx_guide_ind),
    "R-MMA": (fx_mma, fx_mma_ind),
    "R-CLU": (fx_clu, fx_clu_ind),
    "R-SNR": (fx_snr, fx_snr_ind),
    "R-HOPPER": (fx_hopper, fx_hopper_ind),
    "R-TMA-BASE": (fx_tma_base, fx_tma_base_ind),
    "R-TMA-SEM": (fx_tma_sem, fx_tma_sem_ind),
    "R-TMA-XFER": (fx_tma_xfer, fx_tma_xfer_ind),
    "R-LEGACY": (fx_legacy, fx_legacy_ind),
    "R-COMMON": (fx_common, fx_common_ind),
}


def check_orchestrate_dry_run(rule: Rule) -> None:
    """orchestrate.sh --dry-run が出す起動プロセス一覧が RULE.txt の PROCESS 行の展開と一致することを要求する。"""
    with tempfile.TemporaryDirectory() as d:
        env = dict(os.environ, LOG_DIR=str(Path(d) / "logs"))
        res = subprocess.run(["bash", str(ORCHESTRATE_PATH), "--dry-run"], capture_output=True,
                             text=True, env=env, check=False, timeout=120)
        if res.returncode != 0:
            raise AssertionError(f"orchestrate.sh --dry-run が失敗: {res.stderr}")
        _expect(not (Path(d) / "logs").exists(), "--dry-run は何も作らない契約")
    got = [l[len("dry-run: process "):] for l in res.stdout.splitlines() if l.startswith("dry-run: process ")]
    want = expand_processes(rule)
    _expect(got == want, f"orchestrate.sh の起動プロセス一覧が RULE.txt と不一致（{len(got)} 件 vs {len(want)} 件）")


def self_test() -> None:
    rule = load_rule()
    check_rule_consistency(rule)
    # 条項 ID の 3 か所照合: RULE.txt（CLAUSE 行）・本ファイルの CLAUSES・FIXTURES。
    _expect(tuple(rule.lines["CLAUSE"]) == CLAUSES, "RULE.txt の CLAUSE 行と CLAUSES が不一致")
    require_exact_keys(FIXTURES.keys(), CLAUSES, "条項ごとの fixture")
    for pdef in rule.probes.values():
        _expect(pdef.clause in CLAUSES, f"{pdef.id}: 未知の条項")
    for clause, (positive, indeterminate) in FIXTURES.items():
        positive(rule)
        indeterminate(rule)
        print(f"self-test OK: {clause}（陽性・判定不能）")
    check_claim_quotes(load_claims(CLAIMS_PATH.read_text(encoding="utf-8")), REPO_ROOT)
    print("self-test OK: guide_claims の引用照合")
    check_orchestrate_dry_run(rule)
    print("self-test OK: orchestrate.sh の起動プロセス一覧")
    print("self-test: 全件 OK")


# ------------------------------------------------------------------ main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--log-dir", default=str(HERE))
    ap.add_argument("--out", default=None, help="検証後に atomic に書く出力先（未指定は stdout）")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    try:
        if args.self_test:
            self_test()
            return 0
        rule = load_rule()
        check_rule_consistency(rule)
        run = load_run(rule, Path(args.log_dir))
        g0, verdicts = compute_verdicts(rule, run)
        claims = load_claims(CLAIMS_PATH.read_text(encoding="utf-8"))
        md = render(rule, run, g0, verdicts, claims)
        validate_output(md, rule, run, verdicts)
    except (LogIntegrityError, AssertionError) as exc:
        print(f"aggregate: 完全性違反（正式な集計を出力しない）: {exc}", file=sys.stderr)
        return 2
    if g0:
        sys.stdout.write(md)
        print(f"aggregate: G0 不成立 {g0}（全セル判定不能。--out へは書かない）", file=sys.stderr)
        return 3
    if args.out:
        write_atomic(Path(args.out), md)
        print(f"aggregate: {args.out} へ書き出した")
    else:
        sys.stdout.write(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
