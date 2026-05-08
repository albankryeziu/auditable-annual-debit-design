from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import pandas as pd


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    return here.parent.parent if here.parent.name == "scripts" else here.parent


REPO_ROOT = _repo_root()
JSONL_PATH = REPO_ROOT / "data" / "GOPACS" / "raw" / "gopacs_announcements_raw.jsonl"
CROSSWALK_PATH = REPO_ROOT / "data" / "derived" / "gopacs_location_candidate_crosswalk.csv"
ANN_PANEL_PATH = REPO_ROOT / "data" / "derived" / "gopacs_announcement_province_hour.csv"
UNRESOLVED_QC_PATH = REPO_ROOT / "qc" / "step02_unresolved_announcements.csv"
SUMMARY_QC_PATH = REPO_ROOT / "qc" / "step02_announcement_panel_summary.json"

NL_PROVINCES = [
    "Groningen",
    "Friesland",
    "Drenthe",
    "Overijssel",
    "Flevoland",
    "Gelderland",
    "Utrecht",
    "Noord-Holland",
    "Zuid-Holland",
    "Zeeland",
    "Noord-Brabant",
    "Limburg",
]


@dataclass
class ReconciledCandidate:
    province: str
    allocation_weight: float
    match_type: str
    match_score: float
    is_fallback: bool
    ambiguity_degree: str
    operator_detected: str
    operator_constraint_mode: str
    operator_consistency: str


def parse_epoch_ms(value: Any) -> datetime | None:
    if value in (None, "", "null"):
        return None
    try:
        return datetime.fromtimestamp(int(value) / 1000.0, tz=timezone.utc)
    except Exception:
        return None


def _epoch_ms_to_iso(value: Any) -> str:
    dt = parse_epoch_ms(value)
    return dt.isoformat() if dt is not None else ""


def infer_direction(record: dict[str, Any]) -> str:
    buy = bool(str(record.get("requestAreaDescriptionBuyOrders", "") or "").strip())
    sell = bool(str(record.get("requestAreaDescriptionSellOrders", "") or "").strip())
    if buy and sell:
        return "both"
    if buy:
        return "up"
    if sell:
        return "down"
    return "both"


def hourly_profile_mwh(record: dict[str, Any]) -> dict[datetime, float]:
    """
    Convert remainingProblemProfileInMW quarter-hour profile into hourly MWh-equivalent.
    If the explicit quarter-hour profile is absent, create a zero-valued hourly support
    over the stated problem period so that timing can still be tracked separately from magnitude.
    """
    profile = record.get("remainingProblemProfileInMW")
    problem_period = record.get("problemPeriod") or {}

    start_dt = (
        parse_epoch_ms(problem_period.get("startTime"))
        or parse_epoch_ms(record.get("day"))
        or parse_epoch_ms(record.get("createdTimestamp"))
    )
    end_dt = parse_epoch_ms(problem_period.get("endTime"))

    if start_dt is None:
        return {}

    hourly: defaultdict[datetime, float] = defaultdict(float)

    if isinstance(profile, list) and len(profile) > 0:
        current = start_dt
        for raw_value in profile:
            try:
                mw = float(raw_value) if raw_value is not None else 0.0
            except (TypeError, ValueError):
                mw = 0.0
            hour = current.replace(minute=0, second=0, microsecond=0)
            hourly[hour] += mw * 0.25
            current += timedelta(minutes=15)
        return dict(hourly)

    if end_dt is None:
        end_dt = start_dt

    current = start_dt.replace(minute=0, second=0, microsecond=0)
    final = end_dt.replace(minute=0, second=0, microsecond=0)

    safety = 0
    while current <= final and safety < 24 * 14:
        hourly[current] += 0.0
        current += timedelta(hours=1)
        safety += 1

    if not hourly:
        hourly[start_dt.replace(minute=0, second=0, microsecond=0)] = 0.0

    return dict(hourly)


def load_records() -> list[dict[str, Any]]:
    if not JSONL_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {JSONL_PATH}")

    records: list[dict[str, Any]] = []
    with JSONL_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("announcementState") == "ANNOUNCEMENT_DELETED":
                continue
            records.append(record)
    return records


def load_crosswalk_candidates() -> pd.DataFrame:
    if not CROSSWALK_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {CROSSWALK_PATH}")

    crosswalk = pd.read_csv(CROSSWALK_PATH)

    required_cols = {
        "announcement_id",
        "province",
        "match_type",
        "match_score",
        "candidate_weight_norm",
        "is_fallback",
        "ambiguity_degree",
    }
    missing = required_cols.difference(crosswalk.columns)
    if missing:
        raise ValueError(f"Crosswalk missing required columns: {sorted(missing)}")

    crosswalk = crosswalk.copy()
    crosswalk["announcement_id"] = crosswalk["announcement_id"].astype(str).str.strip()
    crosswalk["province"] = crosswalk["province"].astype(str).str.strip()
    crosswalk["match_type"] = crosswalk["match_type"].astype(str).str.strip()
    crosswalk["ambiguity_degree"] = crosswalk["ambiguity_degree"].astype(str).str.strip()
    crosswalk["match_score"] = pd.to_numeric(crosswalk["match_score"], errors="coerce").fillna(0.0)
    crosswalk["candidate_weight_norm"] = pd.to_numeric(
        crosswalk["candidate_weight_norm"], errors="coerce"
    ).fillna(0.0)
    crosswalk["is_fallback"] = crosswalk["is_fallback"].astype(bool)

    if "operator_detected" not in crosswalk.columns:
        crosswalk["operator_detected"] = ""
    if "operator_constraint_mode" not in crosswalk.columns:
        crosswalk["operator_constraint_mode"] = ""
    if "operator_consistency" not in crosswalk.columns:
        crosswalk["operator_consistency"] = "unknown"

    crosswalk = crosswalk[crosswalk["province"].isin(NL_PROVINCES)].copy()

    # Keep one best row per (announcement_id, province) if duplicates exist.
    crosswalk = crosswalk.sort_values(
        by=["announcement_id", "province", "match_score", "candidate_weight_norm"],
        ascending=[True, True, False, False],
    )
    crosswalk = crosswalk.drop_duplicates(subset=["announcement_id", "province"], keep="first").copy()

    return crosswalk


def reconcile_candidates(
    candidates: pd.DataFrame,
) -> tuple[dict[str, list[ReconciledCandidate]], pd.DataFrame]:
    """
    Reconcile step01 candidate rows into per-announcement weighted province allocations.

    Policy:
      1. If any evidence-based candidates exist for an announcement_id, ignore fallback rows.
      2. If no evidence-based candidates exist but fallback rows exist, use fallback rows.
      3. Normalize weights within announcement_id so the allocation sums to 1.
      4. Never replicate a full announcement into every province with weight 1.
    """
    reconciled: dict[str, list[ReconciledCandidate]] = {}
    qc_rows: list[dict[str, Any]] = []

    if candidates.empty:
        return reconciled, pd.DataFrame(qc_rows)

    for announcement_id, grp in candidates.groupby("announcement_id", sort=True):
        grp = grp.copy()

        evidence_grp = grp[~grp["is_fallback"]].copy()
        fallback_grp = grp[grp["is_fallback"]].copy()

        if not evidence_grp.empty:
            chosen = evidence_grp
            candidate_policy = "evidence_only"
        elif not fallback_grp.empty:
            chosen = fallback_grp
            candidate_policy = "fallback_only"
        else:
            candidate_policy = "none"
            qc_rows.append(
                {
                    "announcement_id": announcement_id,
                    "candidate_policy": candidate_policy,
                    "candidate_count_total": int(len(grp)),
                    "candidate_count_used": 0,
                    "sum_raw_weight_used": 0.0,
                    "sum_alloc_weight": 0.0,
                    "max_match_score_used": 0.0,
                    "min_match_score_used": 0.0,
                }
            )
            continue

        raw_weights = chosen["candidate_weight_norm"].astype(float).clip(lower=0.0)
        if float(raw_weights.sum()) <= 0.0:
            raw_weights = chosen["match_score"].astype(float).clip(lower=0.0)

        if float(raw_weights.sum()) <= 0.0:
            raw_weights = pd.Series([1.0] * len(chosen), index=chosen.index, dtype=float)

        alloc_weights = raw_weights / float(raw_weights.sum())

        rows: list[ReconciledCandidate] = []
        for (_, row), alloc_weight in zip(chosen.iterrows(), alloc_weights.tolist()):
            rows.append(
                ReconciledCandidate(
                    province=str(row["province"]),
                    allocation_weight=float(alloc_weight),
                    match_type=str(row["match_type"]),
                    match_score=float(row["match_score"]),
                    is_fallback=bool(row["is_fallback"]),
                    ambiguity_degree=str(row["ambiguity_degree"]),
                    operator_detected=str(row.get("operator_detected", "") or ""),
                    operator_constraint_mode=str(row.get("operator_constraint_mode", "") or ""),
                    operator_consistency=str(row.get("operator_consistency", "unknown") or "unknown"),
                )
            )

        reconciled[str(announcement_id)] = rows

        qc_rows.append(
            {
                "announcement_id": announcement_id,
                "candidate_policy": candidate_policy,
                "candidate_count_total": int(len(grp)),
                "candidate_count_used": int(len(chosen)),
                "sum_raw_weight_used": float(raw_weights.sum()),
                "sum_alloc_weight": float(sum(x.allocation_weight for x in rows)),
                "max_match_score_used": float(chosen["match_score"].max()),
                "min_match_score_used": float(chosen["match_score"].min()),
            }
        )

    qc = pd.DataFrame(qc_rows)
    return reconciled, qc


def build_panel() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    records = load_records()
    candidates = load_crosswalk_candidates()

    records_by_id: dict[str, dict[str, Any]] = {}
    duplicate_raw_ids = 0

    for record in records:
        record_id = str(record.get("id", "")).strip()
        if not record_id:
            continue
        if record_id in records_by_id:
            duplicate_raw_ids += 1
        records_by_id[record_id] = record

    reconciled_candidates, candidate_qc = reconcile_candidates(candidates)

    event_rows: list[dict[str, Any]] = []
    unresolved_rows: list[dict[str, Any]] = []

    min_hour: datetime | None = None
    max_hour: datetime | None = None

    allocatable_record_count = 0
    allocated_record_hour_count = 0
    zero_profile_record_count = 0
    missing_raw_record_count = 0
    unresolved_no_candidate_count = 0
    evidence_based_record_count = 0
    fallback_only_record_count = 0

    national_hourly_total_from_raw = 0.0
    national_hourly_total_after_alloc = 0.0

    for announcement_id, cand_list in reconciled_candidates.items():
        record = records_by_id.get(announcement_id)
        if record is None:
            missing_raw_record_count += 1
            unresolved_rows.append(
                {
                    "announcement_id": announcement_id,
                    "issue": "candidate_id_missing_in_raw_jsonl",
                    "candidate_policy": "unknown",
                    "candidate_count_used": len(cand_list),
                    "announcement_type": "",
                    "organisation_name": "",
                    "created_timestamp_utc": "",
                    "last_updated_timestamp_utc": "",
                }
            )
            continue

        allocatable_record_count += 1
        candidate_policy = "fallback_only" if all(c.is_fallback for c in cand_list) else "evidence_only"
        if candidate_policy == "fallback_only":
            fallback_only_record_count += 1
        else:
            evidence_based_record_count += 1

        hourly_profile = hourly_profile_mwh(record)
        direction = infer_direction(record)
        ann_type = str(record.get("type", "") or "")
        operator = str(record.get("organisationName", "") or "").strip()
        created_iso = _epoch_ms_to_iso(record.get("createdTimestamp"))
        updated_iso = _epoch_ms_to_iso(record.get("lastUpdatedTimestamp"))

        if not hourly_profile:
            zero_profile_record_count += 1
            unresolved_rows.append(
                {
                    "announcement_id": announcement_id,
                    "issue": "no_hourly_profile_support",
                    "candidate_policy": candidate_policy,
                    "candidate_count_used": len(cand_list),
                    "announcement_type": ann_type,
                    "organisation_name": operator,
                    "created_timestamp_utc": created_iso,
                    "last_updated_timestamp_utc": updated_iso,
                }
            )
            continue

        # Re-check normalization defensively.
        alloc_sum = sum(c.allocation_weight for c in cand_list)
        if alloc_sum <= 0:
            unresolved_rows.append(
                {
                    "announcement_id": announcement_id,
                    "issue": "nonpositive_candidate_weight_sum",
                    "candidate_policy": candidate_policy,
                    "candidate_count_used": len(cand_list),
                    "announcement_type": ann_type,
                    "organisation_name": operator,
                    "created_timestamp_utc": created_iso,
                    "last_updated_timestamp_utc": updated_iso,
                }
            )
            continue

        if abs(alloc_sum - 1.0) > 1e-9:
            cand_list = [
                ReconciledCandidate(
                    province=c.province,
                    allocation_weight=float(c.allocation_weight / alloc_sum),
                    match_type=c.match_type,
                    match_score=c.match_score,
                    is_fallback=c.is_fallback,
                    ambiguity_degree=c.ambiguity_degree,
                    operator_detected=c.operator_detected,
                    operator_constraint_mode=c.operator_constraint_mode,
                    operator_consistency=c.operator_consistency,
                )
                for c in cand_list
            ]

        for hour, mwh_equiv in hourly_profile.items():
            min_hour = hour if min_hour is None or hour < min_hour else min_hour
            max_hour = hour if max_hour is None or hour > max_hour else max_hour

            national_hourly_total_from_raw += float(mwh_equiv)
            allocated_record_hour_count += 1

            alloc_check = 0.0
            for cand in cand_list:
                alloc_mwh = float(mwh_equiv) * float(cand.allocation_weight)
                alloc_check += alloc_mwh

                event_rows.append(
                    {
                        "province": cand.province,
                        "timestamp_utc": hour,
                        "announcement_id": announcement_id,
                        "announcement_type": ann_type,
                        "direction": direction,
                        "operator": operator,
                        "candidate_policy": candidate_policy,
                        "match_type": cand.match_type,
                        "ambiguity_degree": cand.ambiguity_degree,
                        "operator_detected": cand.operator_detected,
                        "operator_constraint_mode": cand.operator_constraint_mode,
                        "operator_consistency": cand.operator_consistency,
                        "is_fallback": int(cand.is_fallback),
                        "allocation_weight": float(cand.allocation_weight),
                        "announcement_mwh_equiv": alloc_mwh,
                    }
                )

            national_hourly_total_after_alloc += alloc_check

    # Log raw records that were not represented in the candidate map.
    reconciled_ids = set(reconciled_candidates.keys())
    for record in records:
        record_id = str(record.get("id", "")).strip()
        if record_id and record_id not in reconciled_ids:
            unresolved_no_candidate_count += 1
            unresolved_rows.append(
                {
                    "announcement_id": record_id,
                    "issue": "no_step01_candidates",
                    "candidate_policy": "none",
                    "candidate_count_used": 0,
                    "announcement_type": str(record.get("type", "") or ""),
                    "organisation_name": str(record.get("organisationName", "") or "").strip(),
                    "created_timestamp_utc": _epoch_ms_to_iso(record.get("createdTimestamp")),
                    "last_updated_timestamp_utc": _epoch_ms_to_iso(record.get("lastUpdatedTimestamp")),
                }
            )

    if min_hour is None or max_hour is None:
        raise RuntimeError("No allocatable announcement hours were produced from step01 candidates.")

    full_hours = pd.date_range(min_hour, max_hour, freq="h", tz="UTC")
    skeleton = pd.DataFrame(
        [(province, ts) for province in NL_PROVINCES for ts in full_hours],
        columns=["province", "timestamp_utc"],
    )

    events = pd.DataFrame(event_rows)
    events["timestamp_utc"] = pd.to_datetime(events["timestamp_utc"], utc=True)

    agg = events.groupby(["province", "timestamp_utc"], as_index=False).agg(
        # Weighted versions are the coherent aggregates.
        alloc_weight_sum=("allocation_weight", "sum"),
        announcement_mwh_equiv=("announcement_mwh_equiv", "sum"),
        weighted_record_hours=("allocation_weight", "sum"),
        weighted_records_fallback=("is_fallback", lambda x: float(x.sum())),
        n_unique_records_raw=("announcement_id", "nunique"),
        n_record_rows_raw=("announcement_id", "count"),
        n_operators=("operator", "nunique"),
        ann_type_TA=("announcement_type", lambda x: int((x == "TRANSPORTPROBLEM").any())),
        ann_type_CA=("announcement_type", lambda x: int((x == "CONGESTIONMANAGEMENT").any())),
        ann_type_FT=("announcement_type", lambda x: int((x == "FREE_TEXT").any())),
        direction_up=("direction", lambda x: int(any(v in {"up", "both"} for v in x))),
        direction_down=("direction", lambda x: int(any(v in {"down", "both"} for v in x))),
        any_fallback=("is_fallback", lambda x: int(any(int(v) == 1 for v in x))),
        any_evidence_based=("is_fallback", lambda x: int(any(int(v) == 0 for v in x))),
    )

    panel = skeleton.merge(agg, on=["province", "timestamp_utc"], how="left")

    float_zero_cols = [
        "alloc_weight_sum",
        "announcement_mwh_equiv",
        "weighted_record_hours",
        "weighted_records_fallback",
    ]
    int_zero_cols = [
        "n_unique_records_raw",
        "n_record_rows_raw",
        "n_operators",
        "ann_type_TA",
        "ann_type_CA",
        "ann_type_FT",
        "direction_up",
        "direction_down",
        "any_fallback",
        "any_evidence_based",
    ]

    for col in float_zero_cols:
        panel[col] = panel[col].fillna(0.0)
    for col in int_zero_cols:
        panel[col] = panel[col].fillna(0).astype(int)

    # Compatibility-oriented columns
    panel["n_record_hours"] = panel["weighted_record_hours"]
    panel["n_unique_records"] = panel["n_unique_records_raw"]
    panel["has_announcement"] = (panel["alloc_weight_sum"] > 0.0).astype(int)

    panel["year"] = panel["timestamp_utc"].dt.year
    panel["month"] = panel["timestamp_utc"].dt.month
    panel["weekday"] = panel["timestamp_utc"].dt.weekday
    panel["hour_of_day"] = panel["timestamp_utc"].dt.hour

    panel = panel.sort_values(["province", "timestamp_utc"]).reset_index(drop=True)

    unresolved = pd.DataFrame(unresolved_rows).drop_duplicates().reset_index(drop=True)

    province_counts_allocated_records: dict[str, int] = {}
    if not events.empty:
        province_counts_allocated_records = {
            str(k): int(v)
            for k, v in events.groupby("province")["announcement_id"].nunique().sort_index().to_dict().items()
        }

    # Global coherence checks
    candidate_qc_bad = 0
    if not candidate_qc.empty:
        candidate_qc_bad = int((candidate_qc["sum_alloc_weight"].sub(1.0).abs() > 1e-9).sum())

    national_mwh_gap = float(national_hourly_total_after_alloc - national_hourly_total_from_raw)

    summary = {
        "raw_records_total": int(len(records)),
        "duplicate_raw_id_encounters": int(duplicate_raw_ids),
        "crosswalk_candidate_rows_total": int(len(candidates)),
        "crosswalk_candidate_ids_total": int(candidates["announcement_id"].nunique()) if not candidates.empty else 0,
        "reconciled_candidate_ids_total": int(len(reconciled_candidates)),
        "candidate_qc_bad_weight_sum_count": candidate_qc_bad,
        "allocatable_record_count": int(allocatable_record_count),
        "evidence_based_record_count": int(evidence_based_record_count),
        "fallback_only_record_count": int(fallback_only_record_count),
        "missing_raw_record_count": int(missing_raw_record_count),
        "zero_profile_record_count": int(zero_profile_record_count),
        "unresolved_no_candidate_count": int(unresolved_no_candidate_count),
        "allocated_record_hour_count": int(allocated_record_hour_count),
        "allocated_event_rows": int(len(event_rows)),
        "unresolved_records_logged": int(len(unresolved)),
        "panel_rows": int(len(panel)),
        "panel_start": min_hour.isoformat(),
        "panel_end": max_hour.isoformat(),
        "province_hour_announcement_share": float(panel["has_announcement"].mean()),
        "total_announcement_mwh_equiv_panel": float(panel["announcement_mwh_equiv"].sum()),
        "national_hourly_total_from_raw": float(national_hourly_total_from_raw),
        "national_hourly_total_after_alloc": float(national_hourly_total_after_alloc),
        "national_mwh_gap": national_mwh_gap,
        "province_counts_allocated_records": province_counts_allocated_records,
        "allocation_policy": {
            "evidence_candidates_used_preferentially": True,
            "fallback_used_only_when_no_evidence": True,
            "many_to_many_full_copying_allowed": False,
            "per_announcement_weight_normalized_to_one": True,
        },
    }

    return panel, unresolved, summary


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=str(path.parent), newline="") as tmp:
        tmp.write(text)
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def main() -> None:
    panel, unresolved, summary = build_panel()

    ANN_PANEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    UNRESOLVED_QC_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_QC_PATH.parent.mkdir(parents=True, exist_ok=True)

    panel.to_csv(ANN_PANEL_PATH, index=False, encoding="utf-8")
    unresolved.to_csv(UNRESOLVED_QC_PATH, index=False, encoding="utf-8")
    _atomic_write_text(SUMMARY_QC_PATH, json.dumps(summary, indent=2) + "\n")

    print(f"Wrote panel      -> {ANN_PANEL_PATH}")
    print(f"Wrote unresolved -> {UNRESOLVED_QC_PATH}")
    print(f"Wrote summary    -> {SUMMARY_QC_PATH}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()