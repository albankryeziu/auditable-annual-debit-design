from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Iterable

import requests

BASE_URL = "https://idcons.gopacs-services.eu/machineannouncements"
DEFAULT_USER_AGENT = "FinalPaperPipeline/step00_gopacs_fetch/1.0"


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    return here.parent.parent if here.parent.name == "scripts" else here.parent


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _parse_yyyy_mm_dd(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)

def _as_yyyy_mm_dd(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d")
    return str(value)
def _dt_to_epoch_ms(dt: datetime) -> int:
    return int(dt.astimezone(timezone.utc).timestamp() * 1000)


def _epoch_ms_to_dt(value: Any) -> datetime | None:
    if value in (None, "", "null"):
        return None
    try:
        return datetime.fromtimestamp(int(value) / 1000.0, tz=timezone.utc)
    except Exception:
        return None


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=str(path.parent), newline="") as tmp:
        tmp.write(text)
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("wb", delete=False, dir=str(path.parent)) as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    payload = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    _atomic_write_text(path, payload)


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    lines = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    _atomic_write_text(path, lines)

def _read_existing_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise FetchError(
                    f"Existing raw archive has invalid JSON on line {line_no}: {exc}"
                ) from exc

            if not isinstance(item, dict):
                raise FetchError(
                    f"Existing raw archive line {line_no} is not a JSON object."
                )

            records.append(item)

    return records


def _merge_latest_by_id(
    existing_records: list[dict[str, Any]],
    fetched_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    retained: dict[str, dict[str, Any]] = {}

    existing_missing_id_count = 0
    fetched_missing_id_count = 0
    duplicate_id_count = 0
    replacement_count = 0
    ignored_count = 0
    fetched_new_id_count = 0

    for record in existing_records:
        record_id = str(record.get("id", "")).strip()
        if not record_id:
            existing_missing_id_count += 1
            continue

        if record_id in retained:
            duplicate_id_count += 1
            chosen, replaced = _retain_better(retained[record_id], record)
            retained[record_id] = chosen
            if replaced:
                replacement_count += 1
            else:
                ignored_count += 1
        else:
            retained[record_id] = record

    for record in fetched_records:
        record_id = str(record.get("id", "")).strip()
        if not record_id:
            fetched_missing_id_count += 1
            continue

        if record_id in retained:
            duplicate_id_count += 1
            chosen, replaced = _retain_better(retained[record_id], record)
            retained[record_id] = chosen
            if replaced:
                replacement_count += 1
            else:
                ignored_count += 1
        else:
            retained[record_id] = record
            fetched_new_id_count += 1

    merged_records = sorted(retained.values(), key=_record_rank, reverse=True)

    merge_stats = {
        "existing_archive_record_total": len(existing_records),
        "fetched_archive_record_total": len(fetched_records),
        "final_archive_record_total": len(merged_records),
        "merge_existing_missing_id_count": existing_missing_id_count,
        "merge_fetched_missing_id_count": fetched_missing_id_count,
        "merge_duplicate_id_count": duplicate_id_count,
        "merge_replacement_count": replacement_count,
        "merge_ignored_count": ignored_count,
        "merge_fetched_new_id_count": fetched_new_id_count,
    }

    return merged_records, merge_stats
def _write_csv_rows(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=str(path.parent), newline="") as tmp:
        writer = csv.DictWriter(tmp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def _safe_getattr(module: Any, names: list[str], default: Any) -> Any:
    for name in names:
        if hasattr(module, name):
            return getattr(module, name)
    return default


def _load_optional_pipeline_config(repo_root: Path) -> dict[str, Any]:
    """
    Optional repo-local config loader.

    This does not assume a single schema. It tries a few likely names and
    falls back cleanly when pipeline_config.py is absent or does not define them.
    """
    cfg_path = repo_root / "pipeline_config.py"
    if not cfg_path.exists():
        return {}

    spec = importlib.util.spec_from_file_location("pipeline_config", cfg_path)
    if spec is None or spec.loader is None:
        return {}

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return {
        "from_date": _safe_getattr(
            module,
            ["STEP00_GOPACS_FROM_DATE", "GOPACS_FETCH_FROM_DATE", "FETCH_GOPACS_FROM_DATE"],
            None,
        ),
        "to_date": _safe_getattr(
            module,
            ["STEP00_GOPACS_TO_DATE", "GOPACS_FETCH_TO_DATE", "FETCH_GOPACS_TO_DATE"],
            None,
        ),
        "window_days": _safe_getattr(
            module,
            ["STEP00_GOPACS_WINDOW_DAYS", "GOPACS_FETCH_WINDOW_DAYS", "FETCH_GOPACS_WINDOW_DAYS"],
            None,
        ),
        "sleep_between": _safe_getattr(
            module,
            ["STEP00_GOPACS_SLEEP_BETWEEN", "GOPACS_FETCH_SLEEP_BETWEEN"],
            None,
        ),
        "request_timeout": _safe_getattr(
            module,
            ["STEP00_GOPACS_REQUEST_TIMEOUT", "GOPACS_FETCH_REQUEST_TIMEOUT"],
            None,
        ),
        "retries": _safe_getattr(
            module,
            ["STEP00_GOPACS_RETRIES", "GOPACS_FETCH_RETRIES"],
            None,
        ),
        "raw_dir": _safe_getattr(
            module,
            ["STEP00_GOPACS_RAW_DIR", "GOPACS_RAW_DIR", "RAW_GOPACS_DIR"],
            None,
        ),
    }


def _looks_like_machine_announcement(record: dict[str, Any]) -> bool:
    keys = set(record.keys())
    strong_keys = {
        "id",
        "createdTimestamp",
        "lastUpdatedTimestamp",
        "type",
        "announcementState",
        "organisationName",
        "problemPeriod",
    }
    return len(keys.intersection(strong_keys)) >= 3


@dataclass
class FetchConfig:
    from_date: str
    to_date: str
    window_days: int
    sleep_between: float
    request_timeout: int
    retries: int
    raw_path: Path
    manifest_path: Path
    manifest_latest_path: Path
    window_log_path: Path
    window_log_latest_path: Path
    summary_path: Path
    summary_latest_path: Path
    state_filter: str | None
    type_filter: str | None
    user_agent: str = DEFAULT_USER_AGENT


@dataclass
class WindowLogRow:
    window_start: str
    window_end: str
    http_status: int | None
    raw_count: int
    parsed_count: int
    new_id_count: int
    duplicate_id_count: int
    replacement_count: int
    duplicate_ignored_count: int
    missing_id_count: int
    non_dict_item_count: int
    malformed_record_count: int
    retry_count: int
    error: str | None


class FetchError(RuntimeError):
    pass


def _record_rank(record: dict[str, Any]) -> tuple[int, int, str]:
    updated = _epoch_ms_to_dt(record.get("lastUpdatedTimestamp"))
    created = _epoch_ms_to_dt(record.get("createdTimestamp"))
    return (
        int(updated.timestamp() * 1000) if updated else -1,
        int(created.timestamp() * 1000) if created else -1,
        json.dumps(record, sort_keys=True, ensure_ascii=False),
    )


def _retain_better(current: dict[str, Any], challenger: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    if _record_rank(challenger) > _record_rank(current):
        return challenger, True
    return current, False


def _normalize_payload(payload: Any) -> tuple[list[dict[str, Any]], int, int]:
    """
    Returns:
        records,
        non_dict_item_count,
        malformed_record_count
    """
    if payload is None:
        return [], 0, 0

    if isinstance(payload, list):
        records: list[dict[str, Any]] = []
        non_dict_item_count = 0
        malformed_record_count = 0
        for item in payload:
            if not isinstance(item, dict):
                non_dict_item_count += 1
                continue
            if not _looks_like_machine_announcement(item):
                malformed_record_count += 1
                continue
            records.append(item)
        return records, non_dict_item_count, malformed_record_count

    if isinstance(payload, dict):
        if not payload:
            return [], 0, 0

        # Defensive support for wrapped payloads, without trusting arbitrary dicts.
        for key in ("announcements", "items", "data", "results"):
            candidate = payload.get(key)
            if isinstance(candidate, list):
                return _normalize_payload(candidate)

        if _looks_like_machine_announcement(payload):
            return [payload], 0, 0

        raise FetchError(
            "HTTP 200 returned a JSON dict that does not look like a machine-announcement record."
        )

    raise FetchError(f"Unexpected JSON payload type: {type(payload).__name__}")


class GOPACSFetcher:
    def __init__(self, config: FetchConfig) -> None:
        self.config = config
        self.logger = logging.getLogger("gopacs_fetch")
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/json",
                "User-Agent": self.config.user_agent,
            }
        )

    def close(self) -> None:
        self.session.close()

    def fetch(self) -> tuple[list[dict[str, Any]], list[WindowLogRow], dict[str, int]]:
        start_dt = _parse_yyyy_mm_dd(self.config.from_date)
        end_dt_exclusive = _parse_yyyy_mm_dd(self.config.to_date) + timedelta(days=1)

        if start_dt >= end_dt_exclusive:
            raise ValueError("from_date must be <= to_date")
        if self.config.window_days <= 0:
            raise ValueError("window_days must be positive")
        if self.config.request_timeout <= 0:
            raise ValueError("request_timeout must be positive")
        if self.config.retries <= 0:
            raise ValueError("retries must be positive")
        if self.config.sleep_between < 0:
            raise ValueError("sleep_between must be nonnegative")

        retained: dict[str, dict[str, Any]] = {}
        window_logs: list[WindowLogRow] = []

        duplicate_id_encounters = 0
        retained_replacements = 0
        duplicate_ignored = 0
        missing_id_total = 0
        payload_record_total = 0
        non_dict_item_total = 0
        malformed_record_total = 0
        retry_total = 0

        current = start_dt
        while current < end_dt_exclusive:
            window_end = min(current + timedelta(days=self.config.window_days), end_dt_exclusive)

            (
                records,
                status,
                error,
                retry_count,
                non_dict_item_count,
                malformed_record_count,
            ) = self._fetch_window(current, window_end)

            retry_total += retry_count
            non_dict_item_total += non_dict_item_count
            malformed_record_total += malformed_record_count

            raw_count = len(records) + non_dict_item_count + malformed_record_count
            payload_record_total += len(records)

            parsed_count = 0
            new_id_count = 0
            duplicate_id_count = 0
            replacement_count = 0
            duplicate_ignored_count = 0
            missing_id_count = 0

            for record in records:
                record_id = str(record.get("id", "")).strip()
                if not record_id:
                    missing_id_count += 1
                    missing_id_total += 1
                    continue

                parsed_count += 1

                if record_id in retained:
                    duplicate_id_count += 1
                    duplicate_id_encounters += 1
                    chosen, replaced = _retain_better(retained[record_id], record)
                    retained[record_id] = chosen
                    if replaced:
                        replacement_count += 1
                        retained_replacements += 1
                    else:
                        duplicate_ignored_count += 1
                        duplicate_ignored += 1
                else:
                    retained[record_id] = record
                    new_id_count += 1

            window_logs.append(
                WindowLogRow(
                    window_start=current.isoformat(),
                    window_end=window_end.isoformat(),
                    http_status=status,
                    raw_count=raw_count,
                    parsed_count=parsed_count,
                    new_id_count=new_id_count,
                    duplicate_id_count=duplicate_id_count,
                    replacement_count=replacement_count,
                    duplicate_ignored_count=duplicate_ignored_count,
                    missing_id_count=missing_id_count,
                    non_dict_item_count=non_dict_item_count,
                    malformed_record_count=malformed_record_count,
                    retry_count=retry_count,
                    error=error,
                )
            )

            self.logger.info(
                "%s -> %s | status=%s parsed=%d new=%d dup=%d repl=%d kept_old=%d no_id=%d non_dict=%d malformed=%d retries=%d retained=%d",
                current.date(),
                (window_end - timedelta(seconds=1)).date(),
                status,
                parsed_count,
                new_id_count,
                duplicate_id_count,
                replacement_count,
                duplicate_ignored_count,
                missing_id_count,
                non_dict_item_count,
                malformed_record_count,
                retry_count,
                len(retained),
            )

            current = window_end
            if current < end_dt_exclusive and self.config.sleep_between > 0:
                time.sleep(self.config.sleep_between)

        bad_windows = [w for w in window_logs if (w.http_status != 200) or (w.error is not None)]
        if bad_windows:
            raise FetchError(f"Fetch failed: {len(bad_windows)} windows had unresolved HTTP/request errors.")

        records = sorted(retained.values(), key=_record_rank, reverse=True)

        self._validate_final_records(records)

        fetch_stats = {
            "payload_record_total": payload_record_total,
            "retained_record_total": len(records),
            "duplicate_id_encounters": duplicate_id_encounters,
            "retained_replacements": retained_replacements,
            "duplicate_ignored_count": duplicate_ignored,
            "missing_id_count": missing_id_total,
            "non_dict_item_count": non_dict_item_total,
            "malformed_record_count": malformed_record_total,
            "retry_total": retry_total,
        }
        return records, window_logs, fetch_stats

    def _fetch_window(
        self,
        start_dt: datetime,
        end_dt: datetime,
    ) -> tuple[list[dict[str, Any]], int | None, str | None, int, int, int]:
        params: dict[str, Any] = {
            "starttime": _dt_to_epoch_ms(start_dt),
            "endtime": _dt_to_epoch_ms(end_dt),
        }
        if self.config.state_filter:
            params["state"] = self.config.state_filter
        if self.config.type_filter:
            params["type"] = self.config.type_filter

        last_status: int | None = None
        last_error: str | None = None
        retry_count = 0

        for attempt in range(1, self.config.retries + 1):
            try:
                response = self.session.get(
                    BASE_URL,
                    params=params,
                    timeout=self.config.request_timeout,
                )
                last_status = response.status_code

                if response.status_code == 200:
                    payload = response.json()
                    records, non_dict_item_count, malformed_record_count = _normalize_payload(payload)
                    return (
                        records,
                        response.status_code,
                        None,
                        retry_count,
                        non_dict_item_count,
                        malformed_record_count,
                    )

                last_error = f"HTTP {response.status_code}"

                if response.status_code == 429:
                    retry_after = response.headers.get("Retry-After")
                    if attempt < self.config.retries:
                        retry_count += 1
                        sleep_s = self._retry_after_to_seconds(retry_after, attempt)
                        time.sleep(sleep_s)
                        continue

            except (requests.RequestException, ValueError, FetchError) as exc:
                last_error = str(exc)

            if attempt < self.config.retries:
                retry_count += 1
                time.sleep(min(2 ** attempt, 30))

        return [], last_status, last_error, retry_count, 0, 0

    @staticmethod
    def _retry_after_to_seconds(retry_after: str | None, attempt: int) -> float:
        if retry_after is None:
            return min(2 ** attempt, 30)
        try:
            return max(float(retry_after), 0.0)
        except Exception:
            return min(2 ** attempt, 30)

    @staticmethod
    def _validate_final_records(records: list[dict[str, Any]]) -> None:
        seen: set[str] = set()
        prev_rank: tuple[int, int, str] | None = None

        for idx, record in enumerate(records):
            record_id = str(record.get("id", "")).strip()
            if not record_id:
                raise FetchError("Final retained archive contains a record with missing id.")
            if record_id in seen:
                raise FetchError(f"Final retained archive is not unique on id; duplicate id={record_id!r}.")
            seen.add(record_id)

            rank = _record_rank(record)
            if prev_rank is not None and rank > prev_rank:
                raise FetchError(
                    f"Final retained archive is not deterministically sorted at position {idx}."
                )
            prev_rank = rank


def _summarize(
    records: list[dict[str, Any]],
    window_logs: list[WindowLogRow],
    fetch_stats: dict[str, int],
    raw_path: Path,
) -> dict[str, Any]:
    created_values = [_epoch_ms_to_dt(r.get("createdTimestamp")) for r in records]
    updated_values = [_epoch_ms_to_dt(r.get("lastUpdatedTimestamp")) for r in records]

    created_ok = [x for x in created_values if x is not None]
    updated_ok = [x for x in updated_values if x is not None]

    type_counts: dict[str, int] = {}
    state_counts: dict[str, int] = {}

    missing_created_timestamp_count = 0
    missing_last_updated_timestamp_count = 0

    for record in records:
        type_key = str(record.get("type", "UNKNOWN"))
        state_value = record.get("announcementState")
        state_key = "UNKNOWN" if state_value in (None, "") else str(state_value)

        type_counts[type_key] = type_counts.get(type_key, 0) + 1
        state_counts[state_key] = state_counts.get(state_key, 0) + 1

        if record.get("createdTimestamp") in (None, "", "null"):
            missing_created_timestamp_count += 1
        if record.get("lastUpdatedTimestamp") in (None, "", "null"):
            missing_last_updated_timestamp_count += 1

    non_200_window_count = sum(1 for w in window_logs if w.http_status != 200)
    error_window_count = sum(1 for w in window_logs if w.error is not None)

    return {
        "run_mode": fetch_stats.get("run_mode", "unknown"),
        "payload_record_total": fetch_stats["payload_record_total"],
        "fetched_retained_record_total": fetch_stats.get(
    "fetched_retained_record_total",
    fetch_stats["retained_record_total"],
),
"retained_record_total": fetch_stats["retained_record_total"],
"existing_archive_record_total": fetch_stats.get("existing_archive_record_total", 0),
"final_archive_record_total": fetch_stats.get(
    "final_archive_record_total",
    fetch_stats["retained_record_total"],
),
        "duplicate_id_encounters": fetch_stats["duplicate_id_encounters"],
        "retained_replacements": fetch_stats["retained_replacements"],
        "duplicate_ignored_count": fetch_stats["duplicate_ignored_count"],
        "missing_id_count": fetch_stats["missing_id_count"],
        "non_dict_item_count": fetch_stats["non_dict_item_count"],
        "malformed_record_count": fetch_stats["malformed_record_count"],
        "retry_total": fetch_stats["retry_total"],
        "missing_created_timestamp_count": missing_created_timestamp_count,
        "missing_last_updated_timestamp_count": missing_last_updated_timestamp_count,
        "type_counts": dict(sorted(type_counts.items())),
        "state_counts": dict(sorted(state_counts.items())),
        "created_timestamp_parse_success": len(created_ok),
        "created_timestamp_parse_failure": len(records) - len(created_ok),
        "last_updated_timestamp_parse_success": len(updated_ok),
        "last_updated_timestamp_parse_failure": len(records) - len(updated_ok),
        "created_timestamp_min": min(created_ok).isoformat() if created_ok else None,
        "created_timestamp_max": max(created_ok).isoformat() if created_ok else None,
        "last_updated_timestamp_min": min(updated_ok).isoformat() if updated_ok else None,
        "last_updated_timestamp_max": max(updated_ok).isoformat() if updated_ok else None,
        "window_count": len(window_logs),
        "non_200_window_count": non_200_window_count,
        "error_window_count": error_window_count,
        "raw_archive_sha256": _sha256_of_file(raw_path),
    }


def _window_log_rows_to_dicts(rows: list[WindowLogRow]) -> list[dict[str, Any]]:
    return [asdict(row) for row in rows]

def _detect_potential_truncation(window_logs: list[WindowLogRow]) -> dict[str, Any]:
    successful_counts = [
        row.raw_count
        for row in window_logs
        if row.http_status == 200 and row.error is None
    ]

    suspicious_thresholds = [100, 200, 500, 1000, 2000, 5000, 10000]

    threshold_hits = {
        str(threshold): successful_counts.count(threshold)
        for threshold in suspicious_thresholds
        if successful_counts.count(threshold) >= 2
    }

    return {
        "potential_truncation_warning": bool(threshold_hits),
        "suspicious_repeated_window_counts": threshold_hits,
        "max_successful_window_raw_count": max(successful_counts) if successful_counts else None,
    }
def build_config(args: argparse.Namespace) -> FetchConfig:
    repo_root = _repo_root()
    pipeline_cfg = _load_optional_pipeline_config(repo_root)

    from_date = (
            _as_yyyy_mm_dd(args.from_date)
            or _as_yyyy_mm_dd(pipeline_cfg.get("from_date"))
            or "2020-01-01"
    )

    to_date = (
            _as_yyyy_mm_dd(args.to_date)
            or _as_yyyy_mm_dd(pipeline_cfg.get("to_date"))
            or _now_utc().strftime("%Y-%m-%d")
    )

    window_days = args.window_days or pipeline_cfg.get("window_days") or 30

    sleep_between = (
        args.sleep_between
        if args.sleep_between is not None
        else pipeline_cfg.get("sleep_between")
    )
    if sleep_between is None:
        sleep_between = 1.0

    request_timeout = args.request_timeout or pipeline_cfg.get("request_timeout") or 30
    retries = args.retries or pipeline_cfg.get("retries") or 4

    if args.out_dir:
        raw_dir = Path(args.out_dir)
    elif pipeline_cfg.get("raw_dir"):
        raw_dir = Path(str(pipeline_cfg["raw_dir"]))
    else:
        raw_dir = repo_root / "data" / "GOPACS" / "raw"

    if not raw_dir.is_absolute():
        raw_dir = repo_root / raw_dir

    raw_dir = raw_dir.resolve()

    manifest_dir = repo_root / "manifests"
    qc_dir = repo_root / "qc"
    run_stamp = _now_utc().strftime("%Y%m%dT%H%M%SZ")

    return FetchConfig(
        from_date=from_date,
        to_date=to_date,
        window_days=int(window_days),
        sleep_between=float(sleep_between),
        request_timeout=int(request_timeout),
        retries=int(retries),
        raw_path=raw_dir / "gopacs_announcements_raw.jsonl",
        manifest_path=manifest_dir / f"step00_gopacs_fetch_manifest_{run_stamp}.json",
        manifest_latest_path=manifest_dir / "step00_gopacs_fetch_manifest_latest.json",
        window_log_path=manifest_dir / f"step00_gopacs_fetch_windows_{run_stamp}.csv",
        window_log_latest_path=manifest_dir / "step00_gopacs_fetch_windows_latest.csv",
        summary_path=qc_dir / f"step00_fetch_gopacs_announcements_summary_{run_stamp}.json",
        summary_latest_path=qc_dir / "step00_fetch_gopacs_announcements_summary.json",
        state_filter=args.state,
        type_filter=args.type,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch and audit GOPACS machine announcements.")
    parser.add_argument("--from-date", default=None, help="Inclusive YYYY-MM-DD")
    parser.add_argument("--to-date", default=None, help="Inclusive YYYY-MM-DD")
    parser.add_argument("--window-days", type=int, default=None)
    parser.add_argument("--sleep-between", type=float, default=None)
    parser.add_argument("--request-timeout", type=int, default=None)
    parser.add_argument("--retries", type=int, default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument(
        "--state",
        default=None,
        choices=["ANNOUNCEMENT_OPEN", "ANNOUNCEMENT_CLOSED", "ANNOUNCEMENT_DELETED"],
    )
    parser.add_argument(
        "--type",
        default=None,
        choices=["TRANSPORTPROBLEM", "CONGESTIONMANAGEMENT", "FREE_TEXT"],
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace the existing raw archive with only the records fetched in this run.",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Allow writing an empty archive. Use only for debugging.",
    )
    parser.add_argument(
        "--update-to-today",
        action="store_true",
        help="Use today's UTC date as to-date when no to-date is configured.",
    )
    return parser.parse_args()
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    run_started_utc = _now_utc()
    args = parse_args()
    config = build_config(args)
    logger = logging.getLogger("gopacs_fetch")

    logger.info("Endpoint         : %s", BASE_URL)
    logger.info("Date range       : %s -> %s", config.from_date, config.to_date)
    logger.info("Window days      : %d", config.window_days)
    logger.info("State filter     : %s", config.state_filter)
    logger.info("Type filter      : %s", config.type_filter)
    logger.info("Replace archive  : %s", args.replace)
    logger.info("Allow empty      : %s", args.allow_empty)
    logger.info("Raw output       : %s", config.raw_path)
    logger.info("Latest manifest  : %s", config.manifest_latest_path)
    logger.info("Latest window log: %s", config.window_log_latest_path)
    logger.info("Latest QC        : %s", config.summary_latest_path)

    fetcher = GOPACSFetcher(config)

    try:
        fetched_records, window_logs, fetch_stats = fetcher.fetch()

        if not fetched_records and not args.allow_empty:
            raise FetchError(
                "Fetch returned zero retained records. Refusing to write raw archive. "
                "Use --allow-empty only for debugging."
            )

        if args.replace:
            existing_records: list[dict[str, Any]] = []
            final_records = fetched_records
            merge_stats = {
                "existing_archive_record_total": 0,
                "fetched_archive_record_total": len(fetched_records),
                "final_archive_record_total": len(final_records),
                "merge_existing_missing_id_count": 0,
                "merge_fetched_missing_id_count": 0,
                "merge_duplicate_id_count": 0,
                "merge_replacement_count": 0,
                "merge_ignored_count": 0,
                "merge_fetched_new_id_count": len(fetched_records),
            }
            run_mode = "replace_full_rebuild"
        else:
            existing_records = _read_existing_jsonl(config.raw_path)
            final_records, merge_stats = _merge_latest_by_id(
                existing_records=existing_records,
                fetched_records=fetched_records,
            )
            run_mode = "merge_with_existing_archive"

        if not final_records and not args.allow_empty:
            raise FetchError(
                "Final archive would contain zero records. Refusing to write raw archive."
            )

        GOPACSFetcher._validate_final_records(final_records)

        _write_jsonl(config.raw_path, final_records)

        truncation_qc = _detect_potential_truncation(window_logs)

        summary_stats = dict(fetch_stats)
        summary_stats["run_mode"] = run_mode
        summary_stats["fetched_retained_record_total"] = fetch_stats["retained_record_total"]
        summary_stats["retained_record_total"] = len(final_records)
        summary_stats.update(merge_stats)

        summary = _summarize(final_records, window_logs, summary_stats, config.raw_path)
        summary["run_mode"] = run_mode
        summary["replace_mode"] = bool(args.replace)
        summary["allow_empty"] = bool(args.allow_empty)
        summary["fetched_retained_record_total"] = len(fetched_records)
        summary.update(merge_stats)
        summary.update(truncation_qc)

        window_log_rows = _window_log_rows_to_dicts(window_logs)
        fieldnames = [
            "window_start",
            "window_end",
            "http_status",
            "raw_count",
            "parsed_count",
            "new_id_count",
            "duplicate_id_count",
            "replacement_count",
            "duplicate_ignored_count",
            "missing_id_count",
            "non_dict_item_count",
            "malformed_record_count",
            "retry_count",
            "error",
        ]

        _write_csv_rows(config.window_log_path, fieldnames, window_log_rows)
        _write_csv_rows(config.window_log_latest_path, fieldnames, window_log_rows)

        _write_json(config.summary_path, summary)
        _write_json(config.summary_latest_path, summary)

        manifest = {
            "endpoint": BASE_URL,
            "fetch_started_utc": run_started_utc.isoformat(),
            "fetch_completed_utc": _now_utc().isoformat(),
            "run_mode": run_mode,
            "replace_mode": bool(args.replace),
            "allow_empty": bool(args.allow_empty),
            "from_date": config.from_date,
            "to_date": config.to_date,
            "window_days": config.window_days,
            "window_end_is_exclusive_assumption": True,
            "state_filter": config.state_filter,
            "type_filter": config.type_filter,
            "request_timeout": config.request_timeout,
            "retries": config.retries,
            "sleep_between": config.sleep_between,
            "user_agent": config.user_agent,
            "raw_path": str(config.raw_path),
            "window_log_path": str(config.window_log_path),
            "window_log_latest_path": str(config.window_log_latest_path),
            "summary_path": str(config.summary_path),
            "summary_latest_path": str(config.summary_latest_path),
            "fetched_retained_count": len(fetched_records),
            "final_retained_count": len(final_records),
            "existing_archive_record_total": merge_stats["existing_archive_record_total"],
            "window_count": len(window_logs),
            "window_raw_count_total": sum(x.raw_count for x in window_logs),
            "window_parsed_count_total": sum(x.parsed_count for x in window_logs),
            "window_new_id_total": sum(x.new_id_count for x in window_logs),
            "window_duplicate_id_total": sum(x.duplicate_id_count for x in window_logs),
            "window_replacement_total": sum(x.replacement_count for x in window_logs),
            "window_duplicate_ignored_total": sum(x.duplicate_ignored_count for x in window_logs),
            "window_missing_id_total": sum(x.missing_id_count for x in window_logs),
            "window_non_dict_item_total": sum(x.non_dict_item_count for x in window_logs),
            "window_malformed_record_total": sum(x.malformed_record_count for x in window_logs),
            "window_retry_total": sum(x.retry_count for x in window_logs),
            "non_200_window_count": sum(1 for x in window_logs if x.http_status != 200),
            "error_window_count": sum(1 for x in window_logs if x.error is not None),
            "raw_archive_sha256": _sha256_of_file(config.raw_path),
            "merge_stats": merge_stats,
            "truncation_qc": truncation_qc,
            "summary": summary,
        }

        _write_json(config.manifest_path, manifest)
        _write_json(config.manifest_latest_path, manifest)

        logger.info("Fetched records  : %s", format(len(fetched_records), ","))
        logger.info("Final records    : %s", format(len(final_records), ","))
        logger.info("Run mode         : %s", run_mode)
        logger.info("Raw archive      : %s", config.raw_path)
        logger.info("Manifest         : %s", config.manifest_path)
        logger.info("Manifest latest  : %s", config.manifest_latest_path)
        logger.info("Window log       : %s", config.window_log_path)
        logger.info("Window log latest: %s", config.window_log_latest_path)
        logger.info("QC summary       : %s", config.summary_path)
        logger.info("QC summary latest: %s", config.summary_latest_path)

        if truncation_qc["potential_truncation_warning"]:
            logger.warning(
                "Potential API truncation warning: repeated suspicious window counts = %s",
                truncation_qc["suspicious_repeated_window_counts"],
            )

    finally:
        fetcher.close()


if __name__ == "__main__":
    main()