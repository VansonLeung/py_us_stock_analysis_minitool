import datetime as dt
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm
import yfinance as yf

from .scan_models import (
    DEFAULT_METADATA_CACHE,
    DEFAULT_METADATA_SCOPE,
    DEFAULT_METADATA_TTL_DAYS,
    METADATA_ERROR_TTL_HOURS,
    METADATA_MISSING_RETRY_ATTEMPTS,
    SCAN_METADATA_COLUMNS,
    StockMetadata,
)


def _normalize_metadata_text(value) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split()).strip()


def _safe_float(value) -> float | None:
    if value is None or value == "":
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        return None


def _metadata_timestamp_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def build_empty_stock_metadata_model(symbol: str, status: str = "missing", note: str = "") -> StockMetadata:
    return StockMetadata(
        symbol=symbol.upper(),
        company_name="",
        sector=None,
        industry=None,
        business_summary="",
        country=None,
        market_cap=None,
        beta=None,
        trailing_pe=None,
        forward_pe=None,
        short_ratio=None,
        next_earnings_date=None,
        metadata_status=status,
        metadata_note=note,
        metadata_fetched_at=_metadata_timestamp_now(),
    )


def _stock_metadata_from_record(symbol: str, record: dict) -> StockMetadata:
    return StockMetadata(
        symbol=str(record.get("symbol") or symbol).upper(),
        company_name=_normalize_metadata_text(record.get("company_name")),
        sector=_normalize_metadata_text(record.get("sector")) or None,
        industry=_normalize_metadata_text(record.get("industry")) or None,
        business_summary=_normalize_metadata_text(record.get("business_summary")),
        country=_normalize_metadata_text(record.get("country")) or None,
        market_cap=_safe_float(record.get("market_cap")),
        beta=_safe_float(record.get("beta")),
        trailing_pe=_safe_float(record.get("trailing_pe")),
        forward_pe=_safe_float(record.get("forward_pe")),
        short_ratio=_safe_float(record.get("short_ratio")),
        next_earnings_date=_normalize_metadata_text(record.get("next_earnings_date")) or None,
        metadata_status=_normalize_metadata_text(record.get("metadata_status")) or "missing",
        metadata_note=_normalize_metadata_text(record.get("metadata_note")),
        metadata_fetched_at=_normalize_metadata_text(record.get("metadata_fetched_at")) or _metadata_timestamp_now(),
    )


def _parse_metadata_timestamp(value: str) -> dt.datetime | None:
    raw = _normalize_metadata_text(value)
    if not raw:
        return None
    try:
        return dt.datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def _is_metadata_cache_fresh(record: dict, ttl_days: int) -> bool:
    fetched_at = _parse_metadata_timestamp(str(record.get("metadata_fetched_at", "")))
    if fetched_at is None:
        return False
    now = dt.datetime.now(dt.timezone.utc)
    if fetched_at.tzinfo is None:
        fetched_at = fetched_at.replace(tzinfo=dt.timezone.utc)
    status = _normalize_metadata_text(record.get("metadata_status")).lower()
    ttl = dt.timedelta(hours=METADATA_ERROR_TTL_HOURS) if status in {"error", "missing"} else dt.timedelta(days=ttl_days)
    return now - fetched_at <= ttl


def _load_metadata_cache(cache_path: Path) -> dict[str, dict]:
    if not cache_path.exists():
        return {}
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Could not read metadata cache {cache_path}: {exc}")
        return {}
    if not isinstance(payload, dict):
        return {}

    cache: dict[str, dict] = {}
    for symbol, record in payload.items():
        if isinstance(record, dict):
            cache[str(symbol).upper()] = record
    return cache


def _save_metadata_cache(cache_path: Path, cache: dict[str, dict]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = {symbol: cache[symbol] for symbol in sorted(cache.keys())}
    cache_path.write_text(json.dumps(ordered, ensure_ascii=True, indent=2), encoding="utf-8")


def _fetch_stock_metadata(symbol: str) -> StockMetadata:
    last_missing = build_empty_stock_metadata_model(
        symbol,
        status="missing",
        note=f"empty yfinance metadata after {METADATA_MISSING_RETRY_ATTEMPTS} attempts",
    )

    for attempt in range(1, METADATA_MISSING_RETRY_ATTEMPTS + 1):
        ticker = yf.Ticker(symbol)

        info = {}
        info_exc = None
        try:
            info = ticker.get_info() or {}
        except Exception as exc:
            info_exc = exc

        fast_info = {}
        try:
            fast_info = ticker.fast_info  # type: ignore[attr-defined]
        except Exception:
            fast_info = {}

        def _fi(key: str):
            return fast_info.get(key) if hasattr(fast_info, "get") else getattr(fast_info, key, None)

        company_name = _normalize_metadata_text(
            info.get("shortName") or info.get("longName") or info.get("displayName") or info.get("name")
        )
        sector = _normalize_metadata_text(info.get("sector")) or None
        industry = _normalize_metadata_text(info.get("industry")) or None
        business_summary = ""
        country = _normalize_metadata_text(info.get("country")) or None
        trailing_pe = _safe_float(info.get("trailingPE"))
        forward_pe = _safe_float(info.get("forwardPE"))
        short_ratio = _safe_float(info.get("shortRatio") or info.get("shortPercentOfFloat"))
        market_cap = _safe_float(_fi("market_cap"))
        beta = _safe_float(_fi("beta"))

        next_earnings_date = None
        earnings_ts = info.get("earningsTimestamp")
        try:
            if earnings_ts:
                next_earnings_date = dt.datetime.fromtimestamp(int(earnings_ts), tz=dt.timezone.utc).strftime("%Y-%m-%d")
        except Exception:
            next_earnings_date = None

        status = "ok" if any([company_name, sector, industry, business_summary, market_cap, beta]) else "missing"
        note = ""
        if info_exc is not None:
            note = str(info_exc)
            if status != "ok":
                status = "error"
        elif status == "missing":
            note = (
                f"empty yfinance metadata after {METADATA_MISSING_RETRY_ATTEMPTS} attempts"
                if attempt == METADATA_MISSING_RETRY_ATTEMPTS
                else f"empty yfinance metadata response ({attempt}/{METADATA_MISSING_RETRY_ATTEMPTS})"
            )

        metadata = StockMetadata(
            symbol=symbol.upper(),
            company_name=company_name,
            sector=sector,
            industry=industry,
            business_summary=business_summary,
            country=country,
            market_cap=market_cap,
            beta=beta,
            trailing_pe=trailing_pe,
            forward_pe=forward_pe,
            short_ratio=short_ratio,
            next_earnings_date=next_earnings_date,
            metadata_status=status,
            metadata_note=note,
            metadata_fetched_at=_metadata_timestamp_now(),
        )
        if metadata.metadata_status == "missing":
            last_missing = metadata
            if attempt < METADATA_MISSING_RETRY_ATTEMPTS:
                continue
        return metadata

    return last_missing


def load_stock_metadata_by_symbol(symbols: Sequence[str], cache_path: str | None, ttl_days: int) -> dict[str, StockMetadata]:
    normalized = sorted({str(symbol).upper().strip() for symbol in symbols if str(symbol).strip()})
    if not normalized:
        return {}

    cache_file = Path(cache_path).resolve() if cache_path else None
    cache = _load_metadata_cache(cache_file) if cache_file else {}
    metadata_map: dict[str, StockMetadata] = {}
    to_fetch: list[str] = []
    for symbol in normalized:
        record = cache.get(symbol)
        if record and _is_metadata_cache_fresh(record, ttl_days):
            metadata_map[symbol] = _stock_metadata_from_record(symbol, record)
            continue
        to_fetch.append(symbol)

    cache_dirty = False
    if to_fetch:
        iterator = tqdm(to_fetch, desc="Metadata", disable=len(to_fetch) < 20)
        for symbol in iterator:
            metadata = _fetch_stock_metadata(symbol)
            metadata_map[symbol] = metadata
            if cache_file:
                cache[symbol] = asdict(metadata)
                cache_dirty = True

    if cache_file and cache_dirty:
        _save_metadata_cache(cache_file, cache)
    return metadata_map


def _select_symbols_for_metadata_enrichment(
    frame: pd.DataFrame,
    metadata_scope: str,
    always_include_symbols: Sequence[str] | None = None,
) -> list[str]:
    if frame.empty or "symbol" not in frame.columns or metadata_scope == "off":
        return []

    working = frame.copy()
    working["symbol"] = working["symbol"].astype(str).str.upper().str.strip()
    always = {str(symbol).upper().strip() for symbol in (always_include_symbols or []) if str(symbol).strip()}
    has_status = "status" in working.columns

    if metadata_scope == "all":
        eligible = working[working["status"].astype(str) != "fetch_error"]["symbol"].tolist() if has_status else working["symbol"].tolist()
        return sorted(set(eligible) | always)

    score_series = pd.to_numeric(working.get("score"), errors="coerce").fillna(0) if "score" in working.columns else pd.Series(0, index=working.index)
    if has_status:
        eligible_mask = (score_series >= 4) & (working["status"].astype(str) != "fetch_error")
    else:
        eligible_mask = score_series >= 4
    eligible = set(working.loc[eligible_mask, "symbol"].tolist())
    return sorted(eligible | always)


def attach_stock_metadata_to_scan_frame(
    frame: pd.DataFrame,
    metadata_scope: str = DEFAULT_METADATA_SCOPE,
    metadata_cache_path: str | None = DEFAULT_METADATA_CACHE,
    metadata_ttl_days: int = DEFAULT_METADATA_TTL_DAYS,
    always_include_symbols: Sequence[str] | None = None,
) -> pd.DataFrame:
    out = frame.copy()
    for col in SCAN_METADATA_COLUMNS:
        if col not in out.columns:
            out[col] = ""

    symbols = _select_symbols_for_metadata_enrichment(out, metadata_scope, always_include_symbols)
    if not symbols:
        return out

    metadata_map = load_stock_metadata_by_symbol(symbols, metadata_cache_path, metadata_ttl_days)
    if not metadata_map:
        return out

    metadata_frame = pd.DataFrame(
        {
            "symbol": [meta.symbol for meta in metadata_map.values()],
            "company_name": [meta.company_name for meta in metadata_map.values()],
            "sector": [meta.sector for meta in metadata_map.values()],
            "industry": [meta.industry for meta in metadata_map.values()],
            "business_summary": [meta.business_summary for meta in metadata_map.values()],
            "metadata_status": [meta.metadata_status for meta in metadata_map.values()],
            "metadata_note": [meta.metadata_note for meta in metadata_map.values()],
            "metadata_fetched_at": [meta.metadata_fetched_at for meta in metadata_map.values()],
        }
    )
    out["symbol"] = out["symbol"].astype(str).str.upper().str.strip()
    out = out.drop(columns=SCAN_METADATA_COLUMNS, errors="ignore").merge(metadata_frame, on="symbol", how="left")
    for col in SCAN_METADATA_COLUMNS:
        out[col] = out[col].fillna("")
    return out


attach_metadata_to_scan_frame = attach_stock_metadata_to_scan_frame
empty_stock_metadata = build_empty_stock_metadata_model
get_stock_metadata_map = load_stock_metadata_by_symbol
