from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# 0) CONFIG
# ============================================================

REPO_ROOT = Path(r"C:\Users\Alban\OneDrive - University of Groningen\Desktop\Final Paper Script")

RAW_JSONL = REPO_ROOT / "data" / "GOPACS" / "raw" / "gopacs_announcements_raw.jsonl"
CROSSWALK_CSV = REPO_ROOT / "data" / "derived" / "gopacs_location_candidate_crosswalk.csv"

OUT_DIR = REPO_ROOT / "outputs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FIG_HOW_COUNT = OUT_DIR / "gopacs_heatmap_hour_of_week_counts.png"
FIG_HOW_EUR = OUT_DIR / "gopacs_heatmap_hour_of_week_eur.png"
FIG_HOW_MWH = OUT_DIR / "gopacs_heatmap_hour_of_week_mwh.png"

FIG_MONTH_COUNT = OUT_DIR / "gopacs_heatmap_monthly_counts.png"
FIG_MONTH_EUR = OUT_DIR / "gopacs_heatmap_monthly_eur.png"
FIG_MONTH_MWH = OUT_DIR / "gopacs_heatmap_monthly_mwh.png"

SUMMARY_JSON = REPO_ROOT / "qc" / "step01b_heatmap_summary.json"
PANEL_CSV = REPO_ROOT / "data" / "derived" / "gopacs_province_time_heatmap_panel.csv"

PROVINCES = [
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


# ============================================================
# 1) Helpers
# ============================================================

def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSONL file: {path}")
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for i, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON at line {i} in {path}: {exc}") from exc
            if not isinstance(obj, dict):
                raise RuntimeError(f"Expected JSON object at line {i} in {path}")
            out.append(obj)
    return out


def _read_csv_dicts(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV file: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _epoch_ms_to_dt(value: Any) -> Optional[datetime]:
    if value in (None, "", "null"):
        return None
    try:
        return datetime.fromtimestamp(int(value) / 1000.0, tz=timezone.utc)
    except Exception:
        return None


def _parse_iso_dt(value: str) -> Optional[datetime]:
    value = (value or "").strip()
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


def _iter_values(obj: Any) -> Iterable[Any]:
    if isinstance(obj, dict):
        for v in obj.values():
            for x in _iter_values(v):
                yield x
    elif isinstance(obj, list):
        for v in obj:
            for x in _iter_values(v):
                yield x
    else:
        yield obj


def _iter_path_values(obj: Any, path: str = "") -> Iterable[Tuple[str, Any]]:
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}" if path else str(k)
            for item in _iter_path_values(v, p):
                yield item
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            p = f"{path}[{i}]"
            for item in _iter_path_values(v, p):
                yield item
    else:
        yield path, obj


def _to_float(value: Any) -> Optional[float]:
    if value in (None, "", "null"):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if not s:
        return None
    s = s.replace(",", ".")
    s = re.sub(r"[^0-9eE+\-\.]", "", s)
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def _best_timestamp(record: Dict[str, Any]) -> Optional[datetime]:
    # Strict preference order
    candidate_keys = [
        "startTimestamp",
        "startTime",
        "eventStartTimestamp",
        "createdTimestamp",
        "lastUpdatedTimestamp",
    ]
    for key in candidate_keys:
        if key in record:
            val = record.get(key)
            dt = _epoch_ms_to_dt(val) if not isinstance(val, str) else _parse_iso_dt(val)
            if dt is not None:
                return dt

    # Fallback: scan all values for epoch-like integers or ISO timestamps
    for _, value in _iter_path_values(record):
        if isinstance(value, (int, float)):
            dt = _epoch_ms_to_dt(value)
            if dt is not None and 2019 <= dt.year <= 2030:
                return dt
        elif isinstance(value, str):
            dt = _parse_iso_dt(value)
            if dt is not None and 2019 <= dt.year <= 2030:
                return dt
    return None


def _extract_first_numeric_by_keyword(record: Dict[str, Any], keywords: List[str]) -> Optional[float]:
    keywords = [k.lower() for k in keywords]

    # First pass: path-sensitive search
    for path, value in _iter_path_values(record):
        path_l = path.lower()
        if any(k in path_l for k in keywords):
            num = _to_float(value)
            if num is not None:
                return num

    # Second pass: string payload search
    patt = re.compile(r"[-+]?\d+(?:[.,]\d+)?")
    for _, value in _iter_path_values(record):
        if isinstance(value, str):
            s = value.lower()
            if any(k in s for k in keywords):
                m = patt.search(value)
                if m:
                    num = _to_float(m.group(0))
                    if num is not None:
                        return num
    return None


def _extract_gopacs_eur(record: Dict[str, Any]) -> Optional[float]:
    return _extract_first_numeric_by_keyword(
        record,
        keywords=["eur", "euro", "price", "cost", "amount", "financial"],
    )


def _extract_gopacs_mwh(record: Dict[str, Any]) -> Optional[float]:
    return _extract_first_numeric_by_keyword(
        record,
        keywords=["mwh", "mw", "volume", "energy", "capacity"],
    )


def _month_label(dt: datetime) -> str:
    return f"{dt.year:04d}-{dt.month:02d}"


def _hour_of_week(dt: datetime) -> int:
    return dt.weekday() * 24 + dt.hour


def _how_labels() -> List[str]:
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    labels = []
    for d in days:
        for h in range(24):
            labels.append(f"{d}-{h:02d}")
    return labels


def _plot_heatmap(
    data: pd.DataFrame,
    title: str,
    cbar_label: str,
    out_path: Path,
    annotate: bool = False,
    fmt: str = ".0f",
    rotation: int = 0,
) -> None:
    if data.empty:
        raise RuntimeError(f"Cannot plot empty heatmap: {title}")

    arr = data.to_numpy(dtype=float)

    fig_w = max(10.5, 0.40 * data.shape[1] + 3.0)
    fig_h = max(6.2, 0.48 * data.shape[0] + 2.0)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(arr, aspect="auto")

    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_xticklabels(list(data.columns), rotation=rotation, ha="right" if rotation else "center", fontsize=8)

    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_yticklabels(list(data.index), fontsize=9)

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("")
    ax.set_ylabel("Province")

    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label(cbar_label)

    # cell grid
    ax.set_xticks(np.arange(-0.5, data.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, data.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.6, alpha=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)

    if annotate and data.shape[0] <= 12 and data.shape[1] <= 36:
        max_val = np.nanmax(arr) if arr.size else 0.0
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = arr[i, j]
                color = "white" if max_val > 0 and val > 0.55 * max_val else "black"
                ax.text(j, i, format(val, fmt), ha="center", va="center", fontsize=7, color=color)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 2) Build localized panel
# ============================================================

def build_panel() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    raw_records = _load_jsonl(RAW_JSONL)
    crosswalk_rows = _read_csv_dicts(CROSSWALK_CSV)

    raw_by_id: Dict[str, Dict[str, Any]] = {}
    for rec in raw_records:
        aid = str(rec.get("id", "")).strip()
        if aid:
            raw_by_id[aid] = rec

    panel_rows: List[Dict[str, Any]] = []

    missing_raw_match = 0
    missing_timestamp = 0
    missing_eur = 0
    missing_mwh = 0

    for row in crosswalk_rows:
        aid = str(row.get("announcement_id", "")).strip()
        province = str(row.get("province", "")).strip()
        resolution_method = str(row.get("resolution_method", "")).strip()
        resolution_bucket = str(row.get("resolution_bucket", "")).strip()

        if not aid or not province:
            continue
        if province not in PROVINCES:
            continue

        raw = raw_by_id.get(aid)
        if raw is None:
            missing_raw_match += 1
            continue

        ts = _best_timestamp(raw)
        if ts is None:
            missing_timestamp += 1
            continue

        eur_val = _extract_gopacs_eur(raw)
        mwh_val = _extract_gopacs_mwh(raw)

        if eur_val is None:
            missing_eur += 1
        if mwh_val is None:
            missing_mwh += 1

        panel_rows.append(
            {
                "announcement_id": aid,
                "province": province,
                "timestamp_utc": ts.isoformat(),
                "year": ts.year,
                "month": ts.month,
                "month_label": _month_label(ts),
                "weekday": ts.weekday(),
                "hour": ts.hour,
                "hour_of_week": _hour_of_week(ts),
                "activation_count": 1.0,
                "gopacs_eur": float(eur_val) if eur_val is not None else 0.0,
                "gopacs_mwh": float(mwh_val) if mwh_val is not None else 0.0,
                "has_eur": 0 if eur_val is None else 1,
                "has_mwh": 0 if mwh_val is None else 1,
                "resolution_method": resolution_method,
                "resolution_bucket": resolution_bucket,
                "announcement_type": raw.get("type", ""),
                "organisation_name": raw.get("organisationName", ""),
            }
        )

    panel = pd.DataFrame(panel_rows)
    if panel.empty:
        raise RuntimeError("Localized panel is empty. Heatmap generation cannot proceed.")

    # Stable province order
    panel["province"] = pd.Categorical(panel["province"], categories=PROVINCES, ordered=True)
    panel = panel.sort_values(["province", "timestamp_utc", "announcement_id"]).reset_index(drop=True)

    summary = {
        "raw_records_total": len(raw_records),
        "crosswalk_rows_total": len(crosswalk_rows),
        "panel_rows_total": int(len(panel)),
        "missing_raw_match": int(missing_raw_match),
        "missing_timestamp": int(missing_timestamp),
        "missing_eur": int(missing_eur),
        "missing_mwh": int(missing_mwh),
        "province_counts": {
            str(k): int(v) for k, v in panel.groupby("province", observed=True)["announcement_id"].count().items()
        },
        "resolution_bucket_counts": {
            str(k): int(v) for k, v in panel.groupby("resolution_bucket")["announcement_id"].count().items()
        },
        "time_min_utc": str(panel["timestamp_utc"].min()),
        "time_max_utc": str(panel["timestamp_utc"].max()),
    }

    return panel, summary


# ============================================================
# 3) Build heatmap tables
# ============================================================

def province_by_hour_of_week(panel: pd.DataFrame, value_col: str) -> pd.DataFrame:
    tbl = (
        panel.groupby(["province", "hour_of_week"], observed=True)[value_col]
        .sum()
        .unstack(fill_value=0.0)
        .reindex(index=PROVINCES, fill_value=0.0)
    )
    full_cols = list(range(168))
    tbl = tbl.reindex(columns=full_cols, fill_value=0.0)
    tbl.columns = _how_labels()
    return tbl


def province_by_month(panel: pd.DataFrame, value_col: str) -> pd.DataFrame:
    month_order = sorted(panel["month_label"].unique())
    tbl = (
        panel.groupby(["province", "month_label"], observed=True)[value_col]
        .sum()
        .unstack(fill_value=0.0)
        .reindex(index=PROVINCES, fill_value=0.0)
    )
    tbl = tbl.reindex(columns=month_order, fill_value=0.0)
    return tbl


# ============================================================
# 4) Main
# ============================================================

def main() -> None:
    print("Building province-localized GOPACS heatmaps from:")
    print(" ", RAW_JSONL)
    print(" ", CROSSWALK_CSV)

    panel, summary = build_panel()
    PANEL_CSV.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(PANEL_CSV, index=False, encoding="utf-8")

    # Heatmap tables
    how_count = province_by_hour_of_week(panel, "activation_count")
    how_eur = province_by_hour_of_week(panel, "gopacs_eur")
    how_mwh = province_by_hour_of_week(panel, "gopacs_mwh")

    month_count = province_by_month(panel, "activation_count")
    month_eur = province_by_month(panel, "gopacs_eur")
    month_mwh = province_by_month(panel, "gopacs_mwh")

    # Figures
    _plot_heatmap(
        how_count,
        title="Province heatmap — GOPACS localized activation count by hour of week",
        cbar_label="Localized announcement count",
        out_path=FIG_HOW_COUNT,
        annotate=False,
        rotation=90,
    )
    _plot_heatmap(
        how_eur,
        title="Province heatmap — GOPACS EUR by hour of week",
        cbar_label="Extracted GOPACS EUR",
        out_path=FIG_HOW_EUR,
        annotate=False,
        rotation=90,
    )
    _plot_heatmap(
        how_mwh,
        title="Province heatmap — GOPACS MWh by hour of week",
        cbar_label="Extracted GOPACS MWh",
        out_path=FIG_HOW_MWH,
        annotate=False,
        rotation=90,
    )

    _plot_heatmap(
        month_count,
        title="Province heatmap — monthly localized GOPACS activation count",
        cbar_label="Localized announcement count",
        out_path=FIG_MONTH_COUNT,
        annotate=True,
        fmt=".0f",
        rotation=90,
    )
    _plot_heatmap(
        month_eur,
        title="Province heatmap — monthly GOPACS EUR",
        cbar_label="Extracted GOPACS EUR",
        out_path=FIG_MONTH_EUR,
        annotate=False,
        rotation=90,
    )
    _plot_heatmap(
        month_mwh,
        title="Province heatmap — monthly GOPACS MWh",
        cbar_label="Extracted GOPACS MWh",
        out_path=FIG_MONTH_MWH,
        annotate=False,
        rotation=90,
    )

    summary.update(
        {
            "panel_csv": str(PANEL_CSV),
            "fig_hour_of_week_count": str(FIG_HOW_COUNT),
            "fig_hour_of_week_eur": str(FIG_HOW_EUR),
            "fig_hour_of_week_mwh": str(FIG_HOW_MWH),
            "fig_month_count": str(FIG_MONTH_COUNT),
            "fig_month_eur": str(FIG_MONTH_EUR),
            "fig_month_mwh": str(FIG_MONTH_MWH),
        }
    )
    SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_JSON.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print("\nSaved panel:")
    print(" ", PANEL_CSV)

    print("\nSaved figures:")
    print(" ", FIG_HOW_COUNT)
    print(" ", FIG_HOW_EUR)
    print(" ", FIG_HOW_MWH)
    print(" ", FIG_MONTH_COUNT)
    print(" ", FIG_MONTH_EUR)
    print(" ", FIG_MONTH_MWH)

    print("\nSummary:")
    print(" ", SUMMARY_JSON)

    print("\nPanel diagnostics:")
    print(f"  localized rows      = {len(panel):,}")
    print(f"  time min            = {summary['time_min_utc']}")
    print(f"  time max            = {summary['time_max_utc']}")
    print(f"  missing EUR rows    = {summary['missing_eur']:,}")
    print(f"  missing MWh rows    = {summary['missing_mwh']:,}")
    print(f"  missing time rows   = {summary['missing_timestamp']:,}")


if __name__ == "__main__":
    main()