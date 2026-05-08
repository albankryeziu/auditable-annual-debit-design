from __future__ import annotations

import csv
import json
import logging
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


# ============================================================
# Configuration
# ============================================================

INCLUDED_TYPES = {"CONGESTIONMANAGEMENT", "TRANSPORTPROBLEM"}

CANONICAL_PROVINCES = [
    "Drenthe",
    "Flevoland",
    "Friesland",
    "Gelderland",
    "Groningen",
    "Limburg",
    "Noord-Brabant",
    "Noord-Holland",
    "Overijssel",
    "Utrecht",
    "Zeeland",
    "Zuid-Holland",
]

# ------------------------------------------------------------
# Public operator territories
# ------------------------------------------------------------
# IMPORTANT:
#   These are corroborating constraints only.
#   They must not be treated as province truth in step01.
#   They may be used for:
#     - consistency flags,
#     - fallback candidate generation,
#     - downstream weighting support,
#   but not for hard assignment.
# ------------------------------------------------------------
DSO_TERRITORIES: Dict[str, Set[str]] = {
    "Liander": {
        "Flevoland", "Friesland", "Gelderland", "Groningen", "Noord-Holland",
    },
    "Enexis": {
        "Drenthe", "Groningen", "Limburg", "Noord-Brabant", "Overijssel", "Zeeland",
    },
    "Stedin": {"Utrecht", "Zeeland", "Zuid-Holland"},
    "Westland Infra": {"Zuid-Holland"},
    "Coteq": {"Drenthe", "Overijssel"},
    "DNWG": {"Zeeland"},
    "Enduris": {"Zeeland"},
    "Delta": {"Zeeland"},
}

NATIONAL_OPERATORS: Set[str] = {"TenneT"}

DSO_ALIASES_RAW: Dict[str, str] = {
    "liander": "Liander",
    "alliander": "Liander",
    "enexis": "Enexis",
    "stedin": "Stedin",
    "westland infra": "Westland Infra",
    "westland": "Westland Infra",
    "coteq": "Coteq",
    "dnwg": "DNWG",
    "enduris": "Enduris",
    "delta": "Delta",
    "tennet": "TenneT",
}

# ------------------------------------------------------------
# Province aliases
# ------------------------------------------------------------
PROVINCE_ALIASES: Dict[str, str] = {
    "groningen": "Groningen",
    "friesland": "Friesland",
    "fryslan": "Friesland",
    "drenthe": "Drenthe",
    "overijssel": "Overijssel",
    "flevoland": "Flevoland",
    "gelderland": "Gelderland",
    "utrecht": "Utrecht",
    "noord holland": "Noord-Holland",
    "noord-holland": "Noord-Holland",
    "zuid holland": "Zuid-Holland",
    "zuid-holland": "Zuid-Holland",
    "zeeland": "Zeeland",
    "noord brabant": "Noord-Brabant",
    "noord-brabant": "Noord-Brabant",
    "limburg": "Limburg",
    "nh": "Noord-Holland",
    "zh": "Zuid-Holland",
    "nb": "Noord-Brabant",
}

# ------------------------------------------------------------
# Location aliases
# ------------------------------------------------------------
LOCATION_ALIASES: Dict[str, str] = {
    # Groningen
    "eemshaven": "Groningen",
    "delfzijl": "Groningen",
    "hoogezand": "Groningen",
    "stadskanaal": "Groningen",
    "veendam": "Groningen",
    "winschoten": "Groningen",
    "grijpskerk": "Groningen",
    "leek": "Groningen",
    "groningen stad": "Groningen",
    "marnezijl": "Groningen",
    "ulkesluis": "Groningen",

    # Friesland
    "leeuwarden": "Friesland",
    "heerenveen": "Friesland",
    "drachten": "Friesland",
    "sneek": "Friesland",
    "harlingen": "Friesland",
    "lemmer": "Friesland",
    "franeker": "Friesland",
    "joure": "Friesland",
    "waskemeer": "Friesland",
    "noordwolde": "Friesland",
    "damwoude": "Friesland",
    "damwald": "Friesland",
    "hallum": "Friesland",
    "dronrijp": "Friesland",
    "wommels": "Friesland",
    "stiens": "Friesland",
    "balk": "Friesland",
    "twijzelerheide": "Friesland",

    # Drenthe
    "assen": "Drenthe",
    "emmen": "Drenthe",
    "hoogeveen": "Drenthe",
    "meppel": "Drenthe",
    "coevorden": "Drenthe",
    "beilen": "Drenthe",
    "wijster": "Drenthe",

    # Overijssel
    "zwolle": "Overijssel",
    "enschede": "Overijssel",
    "hengelo": "Overijssel",
    "almelo": "Overijssel",
    "deventer": "Overijssel",
    "kampen": "Overijssel",
    "rijssen": "Overijssel",
    "ommen": "Overijssel",
    "hardenberg": "Overijssel",
    "twente": "Overijssel",
    "salland": "Overijssel",
    "dedemsvaart": "Overijssel",
    "oldenzaal": "Overijssel",
    "haaksbergen": "Overijssel",

    # Flevoland
    "almere": "Flevoland",
    "lelystad": "Flevoland",
    "dronten": "Flevoland",
    "emmeloord": "Flevoland",
    "zeewolde": "Flevoland",
    "urk": "Flevoland",
    "nop": "Flevoland",
    "noordoostpolder": "Flevoland",

    # Gelderland
    "arnhem": "Gelderland",
    "nijmegen": "Gelderland",
    "apeldoorn": "Gelderland",
    "ede": "Gelderland",
    "doetinchem": "Gelderland",
    "tiel": "Gelderland",
    "harderwijk": "Gelderland",
    "zutphen": "Gelderland",
    "barneveld": "Gelderland",
    "achterhoek": "Gelderland",
    "betuwe": "Gelderland",
    "veluwe": "Gelderland",
    "rivierenland": "Gelderland",
    "druten": "Gelderland",
    "wamel": "Gelderland",
    "buurmalsen": "Gelderland",
    "neerijnen": "Gelderland",
    "duiven": "Gelderland",
    "elst": "Gelderland",

    # Utrecht
    "amersfoort": "Utrecht",
    "nieuwegein": "Utrecht",
    "veenendaal": "Utrecht",
    "woerden": "Utrecht",
    "houten": "Utrecht",
    "ijsselstein": "Utrecht",
    "soest": "Utrecht",
    "baarn": "Utrecht",
    "rijnenburg": "Utrecht",
    "zeist": "Utrecht",
    "maarssen": "Utrecht",
    "bunschoten": "Utrecht",

    # Noord-Holland
    "amsterdam": "Noord-Holland",
    "haarlem": "Noord-Holland",
    "alkmaar": "Noord-Holland",
    "hoorn": "Noord-Holland",
    "zaandam": "Noord-Holland",
    "zaanstad": "Noord-Holland",
    "purmerend": "Noord-Holland",
    "hilversum": "Noord-Holland",
    "ijmuiden": "Noord-Holland",
    "den helder": "Noord-Holland",
    "schiphol": "Noord-Holland",
    "kop van noord holland": "Noord-Holland",
    "waterland": "Noord-Holland",
    "westfriesland": "Noord-Holland",
    "heiloo": "Noord-Holland",
    "vijfhuizen": "Noord-Holland",
    "middenmeer": "Noord-Holland",
    "wieringerwerf": "Noord-Holland",
    "beverwijk": "Noord-Holland",

    # Zuid-Holland
    "rotterdam": "Zuid-Holland",
    "den haag": "Zuid-Holland",
    "the hague": "Zuid-Holland",
    "delft": "Zuid-Holland",
    "leiden": "Zuid-Holland",
    "gouda": "Zuid-Holland",
    "dordrecht": "Zuid-Holland",
    "zoetermeer": "Zuid-Holland",
    "schiedam": "Zuid-Holland",
    "spijkenisse": "Zuid-Holland",
    "vlaardingen": "Zuid-Holland",
    "barendrecht": "Zuid-Holland",
    "ridderkerk": "Zuid-Holland",
    "hoeksche waard": "Zuid-Holland",
    "goeree overflakkee": "Zuid-Holland",
    "goeree-overflakkee": "Zuid-Holland",
    "alphen aan den rijn": "Zuid-Holland",
    "boskoop": "Zuid-Holland",
    "capelle aan den ijssel": "Zuid-Holland",
    "krimpen aan den ijssel": "Zuid-Holland",
    "nieuwerkerk aan den ijssel": "Zuid-Holland",
    "papendrecht": "Zuid-Holland",
    "zwijndrecht": "Zuid-Holland",
    "gorinchem": "Zuid-Holland",
    "katwijk": "Zuid-Holland",
    "noordwijk": "Zuid-Holland",
    "naaldwijk": "Zuid-Holland",
    "honselersdijk": "Zuid-Holland",
    "poeldijk": "Zuid-Holland",
    "de lier": "Zuid-Holland",
    "maasdijk": "Zuid-Holland",
    "maassluis": "Zuid-Holland",
    "botlek": "Zuid-Holland",
    "maasvlakte": "Zuid-Holland",
    "voorburg": "Zuid-Holland",

    # Zeeland
    "middelburg": "Zeeland",
    "vlissingen": "Zeeland",
    "goes": "Zeeland",
    "terneuzen": "Zeeland",
    "zierikzee": "Zeeland",
    "schouwen duiveland": "Zeeland",
    "schouwen-duiveland": "Zeeland",
    "zeeuws vlaanderen": "Zeeland",
    "zeeuws-vlaanderen": "Zeeland",
    "walcheren": "Zeeland",
    "hulst": "Zeeland",
    "axel": "Zeeland",
    "sas van gent": "Zeeland",
    "sluiskil": "Zeeland",
    "breskens": "Zeeland",
    "sluis": "Zeeland",
    "kapelle": "Zeeland",
    "yerseke": "Zeeland",
    "kruiningen": "Zeeland",
    "tholen": "Zeeland",

    # Noord-Brabant
    "eindhoven": "Noord-Brabant",
    "tilburg": "Noord-Brabant",
    "breda": "Noord-Brabant",
    "s hertogenbosch": "Noord-Brabant",
    "den bosch": "Noord-Brabant",
    "helmond": "Noord-Brabant",
    "waalwijk": "Noord-Brabant",
    "roosendaal": "Noord-Brabant",
    "bergen op zoom": "Noord-Brabant",
    "veghel": "Noord-Brabant",
    "uden": "Noord-Brabant",
    "moerdijk": "Noord-Brabant",
    "brainport": "Noord-Brabant",
    "geertruidenberg": "Noord-Brabant",
    "oosterhout": "Noord-Brabant",

    # Limburg
    "maastricht": "Limburg",
    "heerlen": "Limburg",
    "sittard": "Limburg",
    "venlo": "Limburg",
    "roermond": "Limburg",
    "weert": "Limburg",
    "kerkrade": "Limburg",
    "geleen": "Limburg",
    "parkstad": "Limburg",
    "zuid limburg": "Limburg",
    "noord limburg": "Limburg",
}

PRIMARY_PATH_TOKENS = {
    "province",
    "provincie",
    "region",
    "regio",
    "area",
    "gebied",
    "location",
    "locatie",
    "municipality",
    "gemeente",
    "city",
    "plaats",
    "station",
    "substation",
    "gridarea",
    "zone",
    "name",
}

TEXT_PATH_TOKENS = {
    "title",
    "subject",
    "summary",
    "description",
    "message",
    "text",
    "body",
    "content",
    "remark",
    "reason",
}

IGNORE_GENERIC_TERMS = {"netherlands", "nederland", "nl"}

_CAMEL_SPLIT_RE = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)")
_WORD_RE = re.compile(r"[a-z]{4,}")


# ============================================================
# Data classes
# ============================================================

@dataclass
class CrosswalkConfig:
    input_raw_path: Path
    output_candidate_path: Path
    output_audit_path: Path
    output_summary_path: Path
    output_unresolved_terms_path: Path


@dataclass
class Evidence:
    province: str
    score: int
    kind: str
    path: str
    term: str
    excerpt: str


# ============================================================
# Path helpers
# ============================================================

def _repo_root() -> Path:
    here = Path(__file__).resolve()
    return here.parent.parent if here.parent.name == "scripts" else here.parent


def _build_config() -> CrosswalkConfig:
    repo_root = _repo_root()
    return CrosswalkConfig(
        input_raw_path=repo_root / "data" / "GOPACS" / "raw" / "gopacs_announcements_raw.jsonl",
        output_candidate_path=repo_root / "data" / "derived" / "gopacs_location_candidate_crosswalk.csv",
        output_audit_path=repo_root / "qc" / "step01_crosswalk_audit.csv",
        output_summary_path=repo_root / "qc" / "step01_crosswalk_summary.json",
        output_unresolved_terms_path=repo_root / "qc" / "step01_unresolved_location_terms.csv",
    )


# ============================================================
# Generic utilities
# ============================================================

def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _epoch_ms_to_iso(value: Any) -> str:
    if value in (None, "", "null"):
        return ""
    try:
        dt = datetime.fromtimestamp(int(value) / 1000.0, tz=timezone.utc)
        return dt.isoformat()
    except Exception:
        return ""


def _normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.lower()
    value = value.replace("&", " and ")
    value = re.sub(r"[\r\n\t]+", " ", value)
    value = re.sub(r"[^a-z0-9\s\-_/]", " ", value)
    value = re.sub(r"([a-z])(\d)", r"\1 \2", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _preview_text(text: str, max_len: int = 180) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _iter_string_fields(obj: Any, path: str = "") -> Iterable[Tuple[str, str]]:
    if isinstance(obj, dict):
        for key, value in obj.items():
            next_path = f"{path}.{key}" if path else str(key)
            yield from _iter_string_fields(value, next_path)
    elif isinstance(obj, list):
        for idx, value in enumerate(obj):
            next_path = f"{path}[{idx}]"
            yield from _iter_string_fields(value, next_path)
    elif isinstance(obj, str):
        stripped = obj.strip()
        if stripped:
            yield path, stripped


def _path_weight(path: str) -> int:
    norm = _normalize_text(path)
    tokens = set(norm.replace(".", " ").replace("[", " ").replace("]", " ").split())
    if tokens & PRIMARY_PATH_TOKENS:
        return 3
    if tokens & TEXT_PATH_TOKENS:
        return 2
    camel_tokens = {p.lower() for p in _CAMEL_SPLIT_RE.findall(path)}
    if camel_tokens & PRIMARY_PATH_TOKENS:
        return 3
    if camel_tokens & TEXT_PATH_TOKENS:
        return 2
    return 1


def _compile_alias_patterns(alias_map: Dict[str, str]) -> List[Tuple[re.Pattern, str, str]]:
    compiled = []
    seen = set()
    for alias, province in sorted(alias_map.items(), key=lambda kv: (-len(kv[0]), kv[0])):
        alias_norm = _normalize_text(alias)
        if not alias_norm or alias_norm in IGNORE_GENERIC_TERMS:
            continue
        if alias_norm in seen:
            continue
        seen.add(alias_norm)
        pattern = re.compile(r"(?<![a-z0-9])" + re.escape(alias_norm) + r"(?![a-z0-9])")
        compiled.append((pattern, alias_norm, province))
    return compiled


PROVINCE_PATTERNS = _compile_alias_patterns(PROVINCE_ALIASES)
LOCATION_PATTERNS = _compile_alias_patterns(LOCATION_ALIASES)


def _best_preview(record: Dict[str, Any]) -> str:
    preferred_keys = {
        "title", "subject", "summary", "description",
        "message", "text", "content", "name",
    }
    collected = []
    for path, text in _iter_string_fields(record):
        leaf = path.split(".")[-1].lower()
        if "[" in leaf:
            leaf = leaf.split("[", 1)[0]
        if leaf in preferred_keys:
            collected.append(text)
        if len(collected) >= 3:
            break
    if not collected:
        for _, text in _iter_string_fields(record):
            collected.append(text)
            if len(collected) >= 2:
                break
    return _preview_text(" | ".join(collected))


def _add_evidence(
    evidence_by_province: Dict[str, List[Evidence]],
    province: str,
    score: int,
    kind: str,
    path: str,
    term: str,
    excerpt: str,
) -> None:
    evidence_by_province.setdefault(province, []).append(
        Evidence(
            province=province,
            score=score,
            kind=kind,
            path=path,
            term=term,
            excerpt=_preview_text(excerpt, 120),
        )
    )


def _extract_evidence(record: Dict[str, Any]) -> Dict[str, List[Evidence]]:
    """
    Evidence classes:
      explicit_province   — province name directly appears.
      structured_location — location alias found in a strong location-like field.
      text_alias          — location alias found only in text/generic fields.
    """
    evidence_by_province: Dict[str, List[Evidence]] = {}
    dedupe = set()

    for path, raw_text in _iter_string_fields(record):
        norm_text = _normalize_text(raw_text)
        if not norm_text:
            continue

        weight = _path_weight(path)

        for pattern, alias, province in PROVINCE_PATTERNS:
            if pattern.search(norm_text):
                key = (path, alias, province, "explicit_province")
                if key not in dedupe:
                    dedupe.add(key)
                    base_score = 120 if weight == 3 else 85 if weight == 2 else 45
                    _add_evidence(
                        evidence_by_province,
                        province,
                        base_score,
                        "explicit_province",
                        path,
                        alias,
                        raw_text,
                    )

        for pattern, alias, province in LOCATION_PATTERNS:
            if pattern.search(norm_text):
                if weight == 3:
                    kind = "structured_location"
                    base_score = 70
                elif weight == 2:
                    kind = "text_alias"
                    base_score = 35
                else:
                    kind = "text_alias"
                    base_score = 20

                key = (path, alias, province, kind)
                if key not in dedupe:
                    dedupe.add(key)
                    _add_evidence(
                        evidence_by_province,
                        province,
                        base_score,
                        kind,
                        path,
                        alias,
                        raw_text,
                    )

    return evidence_by_province


def _score_evidence(
    evidence_by_province: Dict[str, List[Evidence]],
) -> List[Tuple[str, int, List[Evidence]]]:
    scored = []
    for province, evidence_list in evidence_by_province.items():
        score = sum(ev.score for ev in evidence_list)
        scored.append(
            (
                province,
                score,
                sorted(evidence_list, key=lambda ev: (-ev.score, ev.kind, ev.term)),
            )
        )
    scored.sort(key=lambda x: (-x[1], x[0]))
    return scored


# ============================================================
# Operator helpers
# ============================================================

def _detect_operator(record: Dict[str, Any]) -> Tuple[str, str, Set[str]]:
    raw_org = _normalize_text(str(record.get("organisationName", "")).strip())

    for alias_raw, canonical in DSO_ALIASES_RAW.items():
        alias = _normalize_text(alias_raw)
        if alias and alias in raw_org:
            if canonical in NATIONAL_OPERATORS:
                return canonical, "national", set(CANONICAL_PROVINCES)
            if canonical in DSO_TERRITORIES:
                return canonical, "regional_strict", set(DSO_TERRITORIES[canonical])

    return "", "unknown", set()


def _allowed_provinces_label(constraint_mode: str, allowed_provinces: Set[str]) -> str:
    if constraint_mode == "national":
        return "ALL"
    if not allowed_provinces:
        return ""
    return " | ".join(sorted(allowed_provinces))


def _operator_consistency_label(
    province: str,
    constraint_mode: str,
    allowed_provinces: Set[str],
) -> str:
    if constraint_mode == "national":
        return "national"
    if constraint_mode != "regional_strict":
        return "unknown"
    return "consistent" if province in allowed_provinces else "inconsistent"


# ============================================================
# Candidate construction
# ============================================================

def _dominant_match_type(evidence_list: List[Evidence]) -> str:
    kinds = {ev.kind for ev in evidence_list}
    if "explicit_province" in kinds:
        return "explicit_province"
    if "structured_location" in kinds:
        return "structured_location"
    if "text_alias" in kinds:
        return "text_alias"
    return "unknown"


def _ambiguity_degree(scored_candidates: List[Tuple[str, int, List[Evidence]]], fallback_only: bool) -> str:
    if not scored_candidates:
        return "none"
    if fallback_only:
        return "operator_only_high"
    if len(scored_candidates) == 1:
        top = scored_candidates[0][1]
        return "unique_strong" if top >= 70 else "unique_weak"

    top = scored_candidates[0][1]
    second = scored_candidates[1][1]
    margin = top - second

    if margin >= 60:
        return "multi_clear"
    if margin >= 25:
        return "multi_moderate"
    return "multi_close"


def _candidate_status(scored_candidates: List[Tuple[str, int, List[Evidence]]], fallback_only: bool) -> str:
    if not scored_candidates:
        return "unresolved_no_signal"
    if fallback_only:
        return "operator_only_fallback"
    amb = _ambiguity_degree(scored_candidates, fallback_only=False)
    if amb in {"unique_strong", "multi_clear"}:
        return "candidate_clear"
    return "candidate_ambiguous"


def _build_candidate_rows_for_record(
    record: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    announcement_id = str(record.get("id", "")).strip()
    ann_type = str(record.get("type", "")).strip()
    ann_state_raw = record.get("announcementState")
    ann_state = "UNKNOWN" if ann_state_raw in (None, "") else str(ann_state_raw)
    organisation_name = str(record.get("organisationName", "")).strip()
    created_iso = _epoch_ms_to_iso(record.get("createdTimestamp"))
    updated_iso = _epoch_ms_to_iso(record.get("lastUpdatedTimestamp"))
    preview = _best_preview(record)

    operator_detected, operator_constraint_mode, allowed_provinces = _detect_operator(record)
    allowed_provinces_label = _allowed_provinces_label(operator_constraint_mode, allowed_provinces)

    evidence_by_province = _extract_evidence(record)
    scored = _score_evidence(evidence_by_province)

    fallback_only = False
    candidate_rows: List[Dict[str, Any]] = []

    # Main evidence-based candidate generation
    if scored:
        total_score = sum(max(score, 0) for _, score, _ in scored)
        if total_score <= 0:
            total_score = len(scored)

        ambiguity = _ambiguity_degree(scored, fallback_only=False)

        for rank, (province, score, evidence_list) in enumerate(scored, start=1):
            dominant_type = _dominant_match_type(evidence_list)
            evidence_terms = sorted({ev.term for ev in evidence_list})
            evidence_paths = sorted({ev.path for ev in evidence_list})
            source_field = " | ".join(evidence_paths)
            location_string = " | ".join(evidence_terms[:6])

            candidate_rows.append(
                {
                    "announcement_id": announcement_id,
                    "location_string": location_string,
                    "province": province,
                    "match_type": dominant_type,
                    "match_score": score,
                    "candidate_weight_norm": float(score) / float(total_score) if total_score > 0 else 0.0,
                    "is_fallback": False,
                    "source_field": source_field,
                    "ambiguity_degree": ambiguity,
                    "candidate_rank": rank,
                    "operator_detected": operator_detected,
                    "operator_constraint_mode": operator_constraint_mode,
                    "operator_consistency": _operator_consistency_label(
                        province, operator_constraint_mode, allowed_provinces
                    ),
                    "allowed_provinces": allowed_provinces_label,
                    "announcement_type": ann_type,
                    "announcement_state": ann_state,
                    "organisation_name": organisation_name,
                    "created_timestamp_utc": created_iso,
                    "last_updated_timestamp_utc": updated_iso,
                    "preview_text": preview,
                    "evidence_terms": " | ".join(evidence_terms),
                    "evidence_paths": source_field,
                    "evidence_count": len(evidence_list),
                }
            )

    # Weak fallback generation only when there is no locational evidence.
    # This is NOT province truth. It is an explicit low-confidence fallback.
    elif operator_constraint_mode == "regional_strict" and allowed_provinces:
        fallback_only = True
        fallback_provinces = sorted(allowed_provinces)
        n = len(fallback_provinces)
        ambiguity = _ambiguity_degree([], fallback_only=True)

        for rank, province in enumerate(fallback_provinces, start=1):
            candidate_rows.append(
                {
                    "announcement_id": announcement_id,
                    "location_string": "",
                    "province": province,
                    "match_type": "operator_fallback",
                    "match_score": 5,
                    "candidate_weight_norm": 1.0 / float(n),
                    "is_fallback": True,
                    "source_field": "organisationName",
                    "ambiguity_degree": ambiguity,
                    "candidate_rank": rank,
                    "operator_detected": operator_detected,
                    "operator_constraint_mode": operator_constraint_mode,
                    "operator_consistency": "consistent",
                    "allowed_provinces": allowed_provinces_label,
                    "announcement_type": ann_type,
                    "announcement_state": ann_state,
                    "organisation_name": organisation_name,
                    "created_timestamp_utc": created_iso,
                    "last_updated_timestamp_utc": updated_iso,
                    "preview_text": preview,
                    "evidence_terms": "",
                    "evidence_paths": "",
                    "evidence_count": 0,
                }
            )

    # Audit summary
    status = _candidate_status(scored, fallback_only=fallback_only)

    suggested_primary_province = ""
    suggested_primary_match_type = ""
    suggested_primary_score = 0
    ambiguity = "none"

    if candidate_rows:
        ambiguity = candidate_rows[0]["ambiguity_degree"]
        top_row = candidate_rows[0]
        if status == "candidate_clear":
            suggested_primary_province = str(top_row["province"])
            suggested_primary_match_type = str(top_row["match_type"])
            suggested_primary_score = int(top_row["match_score"])

    second_score = 0
    if scored and len(scored) >= 2:
        second_score = scored[1][1]

    audit_row = {
        "announcement_id": announcement_id,
        "announcement_type": ann_type,
        "announcement_state": ann_state,
        "organisation_name": organisation_name,
        "operator_detected": operator_detected,
        "operator_constraint_mode": operator_constraint_mode,
        "allowed_provinces": allowed_provinces_label,
        "status": status,
        "candidate_count": len(candidate_rows),
        "fallback_only": fallback_only,
        "ambiguity_degree": ambiguity,
        "suggested_primary_province": suggested_primary_province,
        "suggested_primary_match_type": suggested_primary_match_type,
        "suggested_primary_score": suggested_primary_score,
        "top_score": candidate_rows[0]["match_score"] if candidate_rows else 0,
        "second_score": second_score,
        "created_timestamp_utc": created_iso,
        "last_updated_timestamp_utc": updated_iso,
        "preview_text": preview,
    }

    return candidate_rows, audit_row


# ============================================================
# Unresolved diagnostics
# ============================================================

def _collect_unresolved_term_rows(
    record: Dict[str, Any],
    announcement_id: str,
    organisation_name: str,
    operator_detected: str,
    status: str,
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for path, raw_text in _iter_string_fields(record):
        if _path_weight(path) != 3:
            continue
        norm_text = _normalize_text(raw_text)
        if not norm_text:
            continue

        tokens = [tok for tok in _WORD_RE.findall(norm_text) if len(tok) >= 4]
        tokens = [tok for tok in tokens if tok not in IGNORE_GENERIC_TERMS]

        token_str = " | ".join(tokens[:12])
        if not token_str:
            continue

        rows.append(
            {
                "announcement_id": announcement_id,
                "organisation_name": organisation_name,
                "operator_detected": operator_detected or "",
                "status": status,
                "path": path,
                "field_preview": _preview_text(raw_text, 160),
                "candidate_tokens": token_str,
            }
        )
    return rows


# ============================================================
# IO
# ============================================================

def _load_raw_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing raw GOPACS archive: {path}")

    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON at line {line_number}: {exc}") from exc
            if not isinstance(obj, dict):
                raise RuntimeError(f"Expected JSON object at line {line_number}")
            rows.append(obj)
    return rows


def _atomic_write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", newline="", delete=False, dir=str(path.parent)) as tmp:
        writer = csv.DictWriter(tmp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def _atomic_write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=str(path.parent)) as tmp:
        json.dump(obj, tmp, indent=2, ensure_ascii=False)
        tmp.write("\n")
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


# ============================================================
# Main processing
# ============================================================

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger("step01_crosswalk")

    cfg = _build_config()
    raw_records = _load_raw_jsonl(cfg.input_raw_path)

    candidate_rows: List[Dict[str, Any]] = []
    audit_rows: List[Dict[str, Any]] = []
    unresolved_term_rows: List[Dict[str, str]] = []

    total_records = len(raw_records)
    eligible_records = 0
    excluded_type_records = 0

    seen_ids: Set[str] = set()

    candidate_status_counts: Counter = Counter()
    ambiguity_degree_counts: Counter = Counter()
    match_type_counts: Counter = Counter()
    province_counts: Counter = Counter()
    operator_detected_counts: Counter = Counter()
    operator_constraint_mode_counts: Counter = Counter()
    operator_consistency_counts: Counter = Counter()
    fallback_candidate_rows = 0

    for record in raw_records:
        announcement_id = str(record.get("id", "")).strip()
        ann_type = str(record.get("type", "")).strip()
        ann_state_raw = record.get("announcementState")
        ann_state = "UNKNOWN" if ann_state_raw in (None, "") else str(ann_state_raw)
        organisation_name = str(record.get("organisationName", "")).strip()
        preview = _best_preview(record)
        created_iso = _epoch_ms_to_iso(record.get("createdTimestamp"))
        updated_iso = _epoch_ms_to_iso(record.get("lastUpdatedTimestamp"))

        if not announcement_id:
            continue
        if announcement_id in seen_ids:
            raise RuntimeError(f"Duplicate announcement id in step01 input: {announcement_id}")
        seen_ids.add(announcement_id)

        if ann_type not in INCLUDED_TYPES:
            excluded_type_records += 1
            audit_rows.append(
                {
                    "announcement_id": announcement_id,
                    "announcement_type": ann_type,
                    "announcement_state": ann_state,
                    "organisation_name": organisation_name,
                    "operator_detected": "",
                    "operator_constraint_mode": "",
                    "allowed_provinces": "",
                    "status": "excluded_type",
                    "candidate_count": 0,
                    "fallback_only": False,
                    "ambiguity_degree": "none",
                    "suggested_primary_province": "",
                    "suggested_primary_match_type": "",
                    "suggested_primary_score": 0,
                    "top_score": 0,
                    "second_score": 0,
                    "created_timestamp_utc": created_iso,
                    "last_updated_timestamp_utc": updated_iso,
                    "preview_text": preview,
                }
            )
            candidate_status_counts["excluded_type"] += 1
            continue

        eligible_records += 1

        record_candidate_rows, audit_row = _build_candidate_rows_for_record(record)
        audit_rows.append(audit_row)

        status = str(audit_row["status"])
        candidate_status_counts[status] += 1
        ambiguity_degree_counts[str(audit_row["ambiguity_degree"])] += 1

        if status in {"unresolved_no_signal", "operator_only_fallback"}:
            unresolved_term_rows.extend(
                _collect_unresolved_term_rows(
                    record=record,
                    announcement_id=announcement_id,
                    organisation_name=organisation_name,
                    operator_detected=str(audit_row["operator_detected"]),
                    status=status,
                )
            )

        for row in record_candidate_rows:
            candidate_rows.append(row)
            province_counts[str(row["province"])] += 1
            match_type_counts[str(row["match_type"])] += 1
            operator_detected_counts[str(row["operator_detected"] or "UNKNOWN")] += 1
            operator_constraint_mode_counts[str(row["operator_constraint_mode"] or "UNKNOWN")] += 1
            operator_consistency_counts[str(row["operator_consistency"])] += 1
            if bool(row["is_fallback"]):
                fallback_candidate_rows += 1

    candidate_fieldnames = [
        "announcement_id",
        "location_string",
        "province",
        "match_type",
        "match_score",
        "candidate_weight_norm",
        "is_fallback",
        "source_field",
        "ambiguity_degree",
        "candidate_rank",
        "operator_detected",
        "operator_constraint_mode",
        "operator_consistency",
        "allowed_provinces",
        "announcement_type",
        "announcement_state",
        "organisation_name",
        "created_timestamp_utc",
        "last_updated_timestamp_utc",
        "preview_text",
        "evidence_terms",
        "evidence_paths",
        "evidence_count",
    ]

    audit_fieldnames = [
        "announcement_id",
        "announcement_type",
        "announcement_state",
        "organisation_name",
        "operator_detected",
        "operator_constraint_mode",
        "allowed_provinces",
        "status",
        "candidate_count",
        "fallback_only",
        "ambiguity_degree",
        "suggested_primary_province",
        "suggested_primary_match_type",
        "suggested_primary_score",
        "top_score",
        "second_score",
        "created_timestamp_utc",
        "last_updated_timestamp_utc",
        "preview_text",
    ]

    unresolved_terms_fieldnames = [
        "announcement_id",
        "organisation_name",
        "operator_detected",
        "status",
        "path",
        "field_preview",
        "candidate_tokens",
    ]

    _atomic_write_csv(cfg.output_candidate_path, candidate_rows, candidate_fieldnames)
    _atomic_write_csv(cfg.output_audit_path, audit_rows, audit_fieldnames)
    _atomic_write_csv(cfg.output_unresolved_terms_path, unresolved_term_rows, unresolved_terms_fieldnames)

    summary = {
        "generated_at_utc": _now_utc_iso(),
        "input_raw_path": str(cfg.input_raw_path),
        "output_candidate_path": str(cfg.output_candidate_path),
        "output_audit_path": str(cfg.output_audit_path),
        "output_unresolved_terms_path": str(cfg.output_unresolved_terms_path),
        "total_raw_records": total_records,
        "eligible_records": eligible_records,
        "excluded_type_records": excluded_type_records,
        "candidate_row_count": len(candidate_rows),
        "audit_row_count": len(audit_rows),
        "unresolved_term_rows": len(unresolved_term_rows),
        "fallback_candidate_rows": fallback_candidate_rows,
        "strict_policy": {
            "step01_is_candidate_resolver_not_truth": True,
            "one_row_per_candidate_not_per_announcement": True,
            "province_multi_assignment_allowed_in_candidates": True,
            "operator_constraint_used_as_metadata_not_truth": True,
            "operator_only_fallback_allowed": True,
            "hard_resolution_done_in_step01": False,
            "free_text_included": False,
        },
        "candidate_status_counts": dict(sorted(candidate_status_counts.items())),
        "ambiguity_degree_counts": dict(sorted(ambiguity_degree_counts.items())),
        "match_type_counts": dict(sorted(match_type_counts.items())),
        "province_counts": dict(sorted(province_counts.items())),
        "operator_detected_counts": dict(sorted(operator_detected_counts.items())),
        "operator_constraint_mode_counts": dict(sorted(operator_constraint_mode_counts.items())),
        "operator_consistency_counts": dict(sorted(operator_consistency_counts.items())),
        "included_types": sorted(INCLUDED_TYPES),
    }

    _atomic_write_json(cfg.output_summary_path, summary)

    print(f"Wrote {format(len(candidate_rows), ',')} candidate rows -> {cfg.output_candidate_path}")
    print(f"Wrote {format(len(audit_rows), ',')} audit rows     -> {cfg.output_audit_path}")
    print(f"Wrote {format(len(unresolved_term_rows), ',')} unresolved-term rows -> {cfg.output_unresolved_terms_path}")
    print(f"Eligible rows            : {format(eligible_records, ',')}")
    print(f"Excluded rows            : {format(excluded_type_records, ',')}")
    print(f"Fallback candidate rows  : {format(fallback_candidate_rows, ',')}")
    print(f"Candidate status counts  : {dict(sorted(candidate_status_counts.items()))}")


if __name__ == "__main__":
    main()