import argparse
import concurrent.futures
import datetime as dt
import io
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import yfinance as yf
from tqdm import tqdm


NASDAQ_URL = "https://ftp.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt"
OTHER_URL = "https://ftp.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers_exchange.json"
SEC_TICKER_TXT_URL = "https://www.sec.gov/include/ticker.txt"
GITHUB_TICKERS_URL = "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/master/all/all_tickers.txt"
DEFAULT_METADATA_SCOPE = "filtered"
DEFAULT_METADATA_CACHE = "stock_metadata_cache.json"
DEFAULT_METADATA_TTL_DAYS = 30
METADATA_ERROR_TTL_HOURS = 12
SCAN_METADATA_COLUMNS = [
    "company_name",
    "sector",
    "industry",
    "business_summary",
    "metadata_status",
    "metadata_note",
    "metadata_fetched_at",
]


@dataclass
class VCPResult:
    symbol: str
    last_close: float
    pivot_high: float
    contractions: List[float]
    volume_trend_ok: bool
    days_looked: int
    bars: int
    score: float
    note: str


@dataclass
class ScanRow:
    symbol: str
    price_source: str
    status: str  # vcp, no_pattern, fetch_error
    last_close: Optional[float]
    day_change_pct: Optional[float]
    ema_5: Optional[float]
    ema_10: Optional[float]
    ema_20: Optional[float]
    ema_60: Optional[float]
    ema_250: Optional[float]
    price_above_ema_20: Optional[bool]
    price_above_ema_60: Optional[bool]
    price_above_ema_250: Optional[bool]
    ema_stack_bullish: Optional[bool]
    ema_signal_note: str
    trend_score: int
    daily_high: Optional[float]
    daily_low: Optional[float]
    daily_volume: Optional[float]
    daily_turnover: Optional[float]
    pivot_high: Optional[float]
    contractions_pct: str
    volume_trend_ok: Optional[bool]
    score: int
    bars: int
    note: str


@dataclass
class EnrichedRow:
    symbol: str
    status: str  # ok or error
    company_name: str
    business_summary: str
    metadata_status: str
    metadata_note: str
    last_close: Optional[float]
    avg_vol_20: Optional[float]
    ret_60d_pct: Optional[float]
    rs_spy_60d: Optional[float]
    dist_from_52w_high_pct: Optional[float]
    market_cap: Optional[float]
    sector: Optional[str]
    industry: Optional[str]
    country: Optional[str]
    beta: Optional[float]
    trailing_pe: Optional[float]
    forward_pe: Optional[float]
    short_ratio: Optional[float]
    next_earnings_date: Optional[str]
    buy_score: Optional[float]
    buy_target: Optional[float]
    sell_target: Optional[float]
    note: str


@dataclass
class StockMetadata:
    symbol: str
    company_name: str
    sector: Optional[str]
    industry: Optional[str]
    business_summary: str
    country: Optional[str]
    market_cap: Optional[float]
    beta: Optional[float]
    trailing_pe: Optional[float]
    forward_pe: Optional[float]
    short_ratio: Optional[float]
    next_earnings_date: Optional[str]
    metadata_status: str
    metadata_note: str
    metadata_fetched_at: str


def _session_with_retry(timeout: int = 15) -> requests.Session:
    retry = Retry(
        total=3,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    adapter = HTTPAdapter(max_retries=retry)
    sess = requests.Session()
    sess.mount("http://", adapter)
    sess.mount("https://", adapter)
    sess.request_timeout = timeout  # type: ignore[attr-defined]
    return sess


def _normalize_metadata_text(value) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split()).strip()


def _safe_float(value) -> Optional[float]:
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


def _empty_stock_metadata(symbol: str, status: str = "missing", note: str = "") -> StockMetadata:
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


def _parse_metadata_timestamp(value: str) -> Optional[dt.datetime]:
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
    meta = _empty_stock_metadata(symbol)
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
    business_summary = _normalize_metadata_text(info.get("longBusinessSummary"))
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

    return StockMetadata(
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


def get_stock_metadata_map(symbols: Sequence[str], cache_path: Optional[str], ttl_days: int) -> dict[str, StockMetadata]:
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


def _symbols_for_metadata(
    frame: pd.DataFrame,
    metadata_scope: str,
    always_include_symbols: Optional[Sequence[str]] = None,
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


def attach_metadata_to_scan_frame(
    frame: pd.DataFrame,
    metadata_scope: str = DEFAULT_METADATA_SCOPE,
    metadata_cache_path: Optional[str] = DEFAULT_METADATA_CACHE,
    metadata_ttl_days: int = DEFAULT_METADATA_TTL_DAYS,
    always_include_symbols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    out = frame.copy()
    for col in SCAN_METADATA_COLUMNS:
        if col not in out.columns:
            out[col] = ""

    symbols = _symbols_for_metadata(out, metadata_scope, always_include_symbols)
    if not symbols:
        return out

    metadata_map = get_stock_metadata_map(symbols, metadata_cache_path, metadata_ttl_days)
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


def _fetch_sec_symbols(session: requests.Session, timeout: int, user_agent: str) -> List[str]:
    headers = {"User-Agent": user_agent}
    # First try the JSON with exchange info
    try:
        resp = session.get(SEC_TICKERS_URL, timeout=timeout, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        frame = pd.DataFrame.from_dict(data, orient="index")
        frame.columns = [str(c).lower() for c in frame.columns]
        allowed_exchanges = {"NASDAQ", "NYSE", "NYSE MKT", "NYSE ARCA", "BATS"}
        if "exchange" in frame.columns:
            frame["exchange"] = frame["exchange"].astype(str).str.upper()
            frame = frame[frame["exchange"].isin(allowed_exchanges)]
        else:
            print("SEC list missing exchange field; using all tickers.")

        ticker_col = "ticker" if "ticker" in frame.columns else ("symbol" if "symbol" in frame.columns else None)
        if ticker_col is None:
            raise RuntimeError("SEC ticker payload missing ticker column")

        tickers = frame[ticker_col].dropna().astype(str).str.upper()
        tickers = tickers[tickers.str.isalpha()]
        result = sorted(set(tickers))
        if result:
            return result
    except Exception as exc:
        print(f"SEC JSON endpoint failed ({exc}); trying ticker.txt fallback.")

    # Fallback: plain text ticker list (ticker|cik per line)
    resp_txt = session.get(SEC_TICKER_TXT_URL, timeout=timeout, headers=headers)
    if resp_txt.status_code == 200:
        lines = resp_txt.text.strip().splitlines()
        tickers: List[str] = []
        for line in lines:
            if "|" in line:
                ticker = line.split("|")[0].strip().upper()
                if ticker.isalpha():
                    tickers.append(ticker)
        if tickers:
            return sorted(set(tickers))
        print("SEC ticker.txt returned no tickers; trying GitHub mirror.")
    else:
        print(f"SEC ticker.txt status {resp_txt.status_code}; trying GitHub mirror.")

    # Final fallback: community-maintained GitHub list
    resp_git = session.get(GITHUB_TICKERS_URL, timeout=timeout, headers=headers)
    resp_git.raise_for_status()
    tickers_git = [line.strip().upper() for line in resp_git.text.splitlines() if line.strip()]
    tickers_git = [t for t in tickers_git if t.isalpha()]
    if not tickers_git:
        raise RuntimeError("GitHub ticker fallback returned no tickers")
    return sorted(set(tickers_git))


def fetch_symbols(timeout: int = 15, source: str = "auto", sec_user_agent: str = "") -> List[str]:
    session = _session_with_retry(timeout=timeout)
    if source == "sec":
        return _fetch_sec_symbols(session, timeout, sec_user_agent)

    frames: List[pd.DataFrame] = []
    last_err: Optional[Exception] = None
    if source in {"nasdaq", "auto"}:
        for url, symbol_col in [(NASDAQ_URL, "NASDAQ Symbol"), (OTHER_URL, "ACT Symbol")]:
            for candidate_url in (url, url.replace("https://", "http://")):
                try:
                    resp = session.get(candidate_url, timeout=timeout)
                    resp.raise_for_status()
                    last_err = None
                    break
                except Exception as exc:  # noqa: PERF203 - explicit retry loop
                    last_err = exc
            if last_err:
                continue

            df = pd.read_csv(io.StringIO(resp.text), sep="|")
            df = df[(df["Test Issue"] == "N") & (df[symbol_col].notna())]
            df = df[df[symbol_col].str.isalpha()]
            frames.append(df[[symbol_col]].rename(columns={symbol_col: "Symbol"}))

    if frames:
        merged = pd.concat(frames, ignore_index=True)
        symbols = sorted(set(merged["Symbol"].str.upper()))
        return symbols

    if source == "nasdaq":
        raise RuntimeError(f"Failed to fetch symbols from Nasdaq sources ({last_err}).")

    # auto fallback to SEC
    try:
        print("Nasdaq sources unreachable; using SEC list fallback.")
        return _fetch_sec_symbols(session, timeout, sec_user_agent)
    except Exception as exc:
        raise RuntimeError("Failed to fetch symbols from Nasdaq and SEC.") from exc


def local_extrema(values: Sequence[float], window: int, find_max: bool) -> List[int]:
    idx: List[int] = []
    for i in range(window, len(values) - window):
        segment = values[i - window : i + window + 1]
        center = values[i]
        if find_max and center == max(segment):
            idx.append(i)
        if not find_max and center == min(segment):
            idx.append(i)
    return idx


def detect_vcp(df: pd.DataFrame) -> Optional[VCPResult]:
    closes = df["Close"].to_numpy()
    vols = df["Volume"].to_numpy()
    if len(closes) < 120:
        return None

    highs = local_extrema(closes, window=3, find_max=True)
    lows = local_extrema(closes, window=3, find_max=False)
    labeled = [(i, "H") for i in highs] + [(i, "L") for i in lows]
    labeled.sort(key=lambda x: x[0])
    pivots: List[tuple[int, str]] = []
    last_type = None
    for idx, kind in labeled:
        if kind == last_type:
            continue
        pivots.append((idx, kind))
        last_type = kind

    if not pivots or pivots[0][1] != "H":
        pivots = [p for p in pivots if p[1] == "H"] + [p for p in pivots if p[1] == "L"]
        pivots.sort(key=lambda x: x[0])
        if not pivots or pivots[0][1] != "H":
            return None

    drops: List[float] = []
    volumes: List[float] = []
    for i in range(len(pivots) - 1):
        (h_idx, h_type), (l_idx, l_type) = pivots[i], pivots[i + 1]
        if h_type == "H" and l_type == "L" and l_idx > h_idx:
            high = closes[h_idx]
            low = closes[l_idx]
            drop = (high - low) / high
            drops.append(drop)
            volumes.append(vols[h_idx:l_idx + 1].mean())
    if len(drops) < 3:
        return None

    drops = drops[-3:]
    volumes = volumes[-3:]
    decreasing_drops = drops[0] > drops[1] > drops[2] and drops[2] > 0
    volume_ok = volumes[0] > volumes[1] > volumes[2]

    last_pivot_high_idx = pivots[-1][0] if pivots[-1][1] == "H" else pivots[-2][0]
    pivot_high = closes[last_pivot_high_idx]
    last_close = closes[-1]
    near_pivot = last_close >= 0.95 * pivot_high

    score = 0
    score += 2 if decreasing_drops else 0
    score += 1 if volume_ok else 0
    score += 1 if near_pivot else 0

    # Optional elite rating: exceptionally tight/clean VCPs become score 5.
    if score == 4:
        drop_ratio_1 = drops[1] / drops[0] if drops[0] > 0 else 1.0
        drop_ratio_2 = drops[2] / drops[1] if drops[1] > 0 else 1.0
        very_near_pivot = last_close >= 0.985 * pivot_high
        strong_volume_dryup = volumes[2] <= 0.7 * volumes[0] if volumes[0] > 0 else False
        tight_final_drop = drops[2] <= 0.06
        smooth_shrink = drop_ratio_1 <= 0.8 and drop_ratio_2 <= 0.8
        if very_near_pivot and strong_volume_dryup and tight_final_drop and smooth_shrink:
            score = 5

    if score == 0:
        return None

    note_parts = []
    if not decreasing_drops:
        note_parts.append("drops not shrinking")
    if not volume_ok:
        note_parts.append("volume not contracting")
    if not near_pivot:
        note_parts.append("price not near pivot")
    if score == 5:
        note_parts.append("perfect vcp setup")

    return VCPResult(
        symbol=df.attrs.get("symbol", ""),
        last_close=float(last_close),
        pivot_high=float(pivot_high),
        contractions=[round(d * 100, 2) for d in drops],
        volume_trend_ok=volume_ok,
        days_looked=len(closes),
        bars=len(closes),
        score=score,
        note="; ".join(note_parts),
    )


def _normalize_ohlcv(df: pd.DataFrame, symbol: str) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return None

    # If MultiIndex columns (common when YF returns multiple tickers), pick the target symbol or first one.
    if isinstance(df.columns, pd.MultiIndex):
        syms = list(df.columns.get_level_values(-1).unique())
        pick = symbol if symbol in syms else syms[0]
        df = df.xs(pick, axis=1, level=-1)

    # Standardize column names
    col_map = {c.lower(): c for c in df.columns}
    def pick(colkey: str) -> Optional[pd.Series]:
        for k, original in col_map.items():
            if colkey == k:
                series = df[original]
                if isinstance(series, pd.DataFrame):
                    # pick first column if still wide
                    series = series.iloc[:, 0]
                return pd.to_numeric(series, errors="coerce")
        return None

    open_s = pick("open")
    high_s = pick("high")
    low_s = pick("low")
    close_s = pick("close")
    vol_s = pick("volume")
    if any(s is None for s in (open_s, high_s, low_s, close_s, vol_s)):
        return None

    out = pd.DataFrame(
        {
            "Open": open_s,
            "High": high_s,
            "Low": low_s,
            "Close": close_s,
            "Volume": vol_s,
        }
    ).dropna()
    out.attrs["symbol"] = symbol
    return out


def fetch_history_yahoo(symbol: str, lookback_days: int) -> Optional[pd.DataFrame]:
    data = yf.download(
        symbol,
        period=f"{lookback_days}d",
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    norm = _normalize_ohlcv(data, symbol)
    return norm


def fetch_history_yahoo_period(symbol: str, period: str) -> Optional[pd.DataFrame]:
    data = yf.download(
        symbol,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    norm = _normalize_ohlcv(data, symbol)
    return norm


def fetch_history_futu(symbol: str, lookback_days: int, host: str, port: int) -> Optional[pd.DataFrame]:
    try:
        from futu import OpenQuoteContext, RET_OK  # type: ignore
    except ImportError:
        return None

    end = dt.datetime.utcnow()
    start = end - dt.timedelta(days=lookback_days + 5)
    code = f"US.{symbol}"
    ctx = OpenQuoteContext(host=host, port=port)
    try:
        ret, data, _ = ctx.request_history_kline(
            code,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            ktype="K_DAY",
            autype="qfq",
        )
        if ret != RET_OK or data is None or data.empty:
            return None
        data = data.rename(columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        })
        data = data[["Open", "High", "Low", "Close", "Volume"]].dropna()
        data.attrs["symbol"] = symbol
        return data
    finally:
        ctx.close()


def fetch_history(
    symbol: str,
    lookback_days: int,
    price_source: str,
    futu_host: str,
    futu_port: int,
    futu_fallback_yahoo: bool,
) -> Optional[pd.DataFrame]:
    if price_source == "futu":
        data = fetch_history_futu(symbol, lookback_days, futu_host, futu_port)
        if data is not None:
            return data
        if not futu_fallback_yahoo:
            return None
        # fallback to Yahoo if requested
    return fetch_history_yahoo(symbol, lookback_days)


def _compute_latest_ema(close_series: pd.Series, span: int) -> Optional[float]:
    if close_series.empty:
        return None
    ema_series = close_series.ewm(span=span, adjust=False).mean().dropna()
    if ema_series.empty:
        return None
    return float(ema_series.iloc[-1])


def _build_ema_signal_note(
    last_close: float,
    ema_5: Optional[float],
    ema_10: Optional[float],
    ema_20: Optional[float],
    ema_60: Optional[float],
    ema_250: Optional[float],
) -> str:
    notes: list[str] = []
    if None not in (ema_5, ema_10, ema_20) and last_close > ema_5 > ema_10 > ema_20:
        notes.append("ultra-short strong")
    if None not in (ema_20, ema_60) and last_close > ema_20 > ema_60:
        notes.append("mid-trend healthy")
    if None not in (ema_60, ema_250) and last_close > ema_60 > ema_250:
        notes.append("long-trend bullish")
    elif ema_250 is not None and last_close < ema_250:
        notes.append("below long-term trend")
    return "; ".join(notes)


def _compute_trend_score(
    last_close: float,
    ema_5: Optional[float],
    ema_10: Optional[float],
    ema_20: Optional[float],
    ema_60: Optional[float],
    ema_250: Optional[float],
) -> int:
    score = 0
    if ema_5 is not None and last_close > ema_5:
        score += 1
    if None not in (ema_5, ema_10) and ema_5 > ema_10:
        score += 1
    if None not in (ema_10, ema_20) and ema_10 > ema_20:
        score += 1
    if ema_20 is not None and last_close > ema_20:
        score += 2
    if None not in (ema_20, ema_60) and ema_20 > ema_60:
        score += 2
    if ema_60 is not None and last_close > ema_60:
        score += 1
    if None not in (ema_60, ema_250) and ema_60 > ema_250:
        score += 1
    if ema_250 is not None and last_close > ema_250:
        score += 1
    return score


def analyze_symbol(
    symbol: str,
    lookback_days: int,
    price_source: str,
    futu_host: str,
    futu_port: int,
    futu_fallback_yahoo: bool,
) -> tuple[Optional[VCPResult], ScanRow]:
    try:
        df = fetch_history(symbol, lookback_days, price_source, futu_host, futu_port, futu_fallback_yahoo)
        if df is None or df.empty:
            row = ScanRow(
                symbol=symbol,
                price_source=price_source,
                status="fetch_error",
                last_close=None,
                day_change_pct=None,
                ema_5=None,
                ema_10=None,
                ema_20=None,
                ema_60=None,
                ema_250=None,
                price_above_ema_20=None,
                price_above_ema_60=None,
                price_above_ema_250=None,
                ema_stack_bullish=None,
                ema_signal_note="",
                trend_score=0,
                daily_high=None,
                daily_low=None,
                daily_volume=None,
                daily_turnover=None,
                pivot_high=None,
                contractions_pct="",
                volume_trend_ok=None,
                score=0,
                bars=0,
                note="no data",
            )
            return None, row

        vcp = detect_vcp(df)
        close_series = pd.to_numeric(df["Close"], errors="coerce").dropna()
        last_close = float(df["Close"].iloc[-1])
        prev_close = float(df["Close"].iloc[-2]) if len(df) >= 2 else None
        day_change_pct = ((last_close / prev_close) - 1.0) * 100.0 if prev_close not in (None, 0) else None
        ema_5 = _compute_latest_ema(close_series, 5)
        ema_10 = _compute_latest_ema(close_series, 10)
        ema_20 = _compute_latest_ema(close_series, 20)
        ema_60 = _compute_latest_ema(close_series, 60)
        ema_250 = _compute_latest_ema(close_series, 250)
        price_above_ema_20 = (last_close > ema_20) if ema_20 is not None else None
        price_above_ema_60 = (last_close > ema_60) if ema_60 is not None else None
        price_above_ema_250 = (last_close > ema_250) if ema_250 is not None else None
        ema_stack_bullish = (
            ema_5 is not None and ema_10 is not None and ema_20 is not None and ema_60 is not None and ema_250 is not None
            and ema_5 > ema_10 > ema_20 > ema_60 > ema_250
        )
        ema_signal_note = _build_ema_signal_note(last_close, ema_5, ema_10, ema_20, ema_60, ema_250)
        trend_score = _compute_trend_score(last_close, ema_5, ema_10, ema_20, ema_60, ema_250)
        daily_high = float(df["High"].iloc[-1])
        daily_low = float(df["Low"].iloc[-1])
        daily_volume = float(df["Volume"].iloc[-1])
        daily_turnover = float(last_close * daily_volume)
        bars = len(df)
        if vcp:
            row = ScanRow(
                symbol=symbol,
                price_source=price_source,
                status="vcp",
                last_close=last_close,
                day_change_pct=day_change_pct,
                ema_5=ema_5,
                ema_10=ema_10,
                ema_20=ema_20,
                ema_60=ema_60,
                ema_250=ema_250,
                price_above_ema_20=price_above_ema_20,
                price_above_ema_60=price_above_ema_60,
                price_above_ema_250=price_above_ema_250,
                ema_stack_bullish=ema_stack_bullish,
                ema_signal_note=ema_signal_note,
                trend_score=trend_score,
                daily_high=daily_high,
                daily_low=daily_low,
                daily_volume=daily_volume,
                daily_turnover=daily_turnover,
                pivot_high=vcp.pivot_high,
                contractions_pct="|".join(map(str, vcp.contractions)),
                volume_trend_ok=vcp.volume_trend_ok,
                score=int(vcp.score),
                bars=bars,
                note=vcp.note,
            )
            return vcp, row

        row = ScanRow(
            symbol=symbol,
            price_source=price_source,
            status="no_pattern",
            last_close=last_close,
            day_change_pct=day_change_pct,
            ema_5=ema_5,
            ema_10=ema_10,
            ema_20=ema_20,
            ema_60=ema_60,
            ema_250=ema_250,
            price_above_ema_20=price_above_ema_20,
            price_above_ema_60=price_above_ema_60,
            price_above_ema_250=price_above_ema_250,
            ema_stack_bullish=ema_stack_bullish,
            ema_signal_note=ema_signal_note,
            trend_score=trend_score,
            daily_high=daily_high,
            daily_low=daily_low,
            daily_volume=daily_volume,
            daily_turnover=daily_turnover,
            pivot_high=None,
            contractions_pct="",
            volume_trend_ok=None,
            score=0,
            bars=bars,
            note="no VCP pattern",
        )
        return None, row
    except Exception as exc:
        row = ScanRow(
            symbol=symbol,
            price_source=price_source,
            status="fetch_error",
            last_close=None,
            day_change_pct=None,
            ema_5=None,
            ema_10=None,
            ema_20=None,
            ema_60=None,
            ema_250=None,
            price_above_ema_20=None,
            price_above_ema_60=None,
            price_above_ema_250=None,
            ema_stack_bullish=None,
            ema_signal_note="",
            trend_score=0,
            daily_high=None,
            daily_low=None,
            daily_volume=None,
            daily_turnover=None,
            pivot_high=None,
            contractions_pct="",
            volume_trend_ok=None,
            score=0,
            bars=0,
            note=f"error: {exc}",
        )
        return None, row


def run_scan(
    symbols: Iterable[str],
    lookback_days: int,
    max_workers: int,
    price_source: str,
    futu_host: str,
    futu_port: int,
    futu_fallback_yahoo: bool,
) -> tuple[List[VCPResult], List[ScanRow]]:
    results: List[VCPResult] = []
    rows: List[ScanRow] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(analyze_symbol, sym, lookback_days, price_source, futu_host, futu_port, futu_fallback_yahoo): sym
            for sym in symbols
        }
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Scanning"):
            vcp, row = future.result()
            if vcp:
                results.append(vcp)
            rows.append(row)
    results.sort(key=lambda r: (-r.score, r.symbol))
    rows.sort(key=lambda r: r.symbol)
    return results, rows


def _load_symbols_from_csv(path: str) -> List[str]:
    df = pd.read_csv(path)
    sym_col = None
    for candidate in ["symbol", "Symbol", "ticker", "Ticker"]:
        if candidate in df.columns:
            sym_col = candidate
            break
    if sym_col is None:
        raise ValueError("CSV must contain a symbol/ticker column")
    return sorted(df[sym_col].dropna().astype(str).str.upper().unique())


def enrich_symbol(
    symbol: str,
    spy_hist: pd.DataFrame,
    rs_lookback: int,
    metadata_map: dict[str, StockMetadata],
) -> EnrichedRow:
    try:
        hist = fetch_history_yahoo_period(symbol, period="1y")
        metadata = metadata_map.get(symbol.upper(), _empty_stock_metadata(symbol))
        if hist is None or hist.empty:
            return EnrichedRow(
                symbol,
                "error",
                metadata.company_name,
                metadata.business_summary,
                metadata.metadata_status,
                metadata.metadata_note,
                None,
                None,
                None,
                None,
                None,
                metadata.market_cap,
                metadata.sector,
                metadata.industry,
                metadata.country,
                metadata.beta,
                metadata.trailing_pe,
                metadata.forward_pe,
                metadata.short_ratio,
                metadata.next_earnings_date,
                None,
                None,
                None,
                "no data",
            )

        last_close = float(hist["Close"].iloc[-1])
        avg_vol_20 = float(hist["Volume"].tail(20).mean()) if len(hist) >= 1 else None
        ret_60 = None
        rs_spy = None
        if len(hist) > rs_lookback and len(spy_hist) > rs_lookback:
            ret_60 = float(hist["Close"].pct_change(rs_lookback).iloc[-1])
            spy_ret = float(spy_hist["Close"].pct_change(rs_lookback).iloc[-1])
            rs_spy = ret_60 - spy_ret if spy_ret is not None else None

        high_52w = float(hist["High"].rolling(252, min_periods=1).max().iloc[-1])
        dist_high = (last_close / high_52w - 1.0) if high_52w else None

        def _clip01(val: float) -> float:
            return max(0.0, min(1.0, val))

        def _scale(val: Optional[float], low: float, high: float) -> float:
            if val is None or high == low:
                return 0.0
            return _clip01((val - low) / (high - low))

        rs_pct = rs_spy * 100 if rs_spy is not None else None
        ret_pct = ret_60 * 100 if ret_60 is not None else None
        dist_pct = dist_high * 100 if dist_high is not None else None

        proximity_score = _clip01((20 + dist_pct) / 20) if dist_pct is not None else 0.0
        rs_score = _scale(rs_pct, -10.0, 20.0)
        mom_score = _scale(ret_pct, -15.0, 30.0)
        buy_score = round((rs_score * 35) + (mom_score * 30) + (proximity_score * 35), 2)

        buy_target = None
        sell_target = None
        if last_close and high_52w:
            breakout_price = high_52w * 1.01
            buy_target = round(breakout_price, 2)
            sell_target = round(breakout_price * 1.15, 2)

        return EnrichedRow(
            symbol=symbol,
            status="ok",
            company_name=metadata.company_name,
            business_summary=metadata.business_summary,
            metadata_status=metadata.metadata_status,
            metadata_note=metadata.metadata_note,
            last_close=last_close,
            avg_vol_20=avg_vol_20,
            ret_60d_pct=ret_60 * 100 if ret_60 is not None else None,
            rs_spy_60d=rs_spy * 100 if rs_spy is not None else None,
            dist_from_52w_high_pct=dist_high * 100 if dist_high is not None else None,
            market_cap=metadata.market_cap,
            sector=metadata.sector,
            industry=metadata.industry,
            country=metadata.country,
            beta=metadata.beta,
            trailing_pe=metadata.trailing_pe,
            forward_pe=metadata.forward_pe,
            short_ratio=metadata.short_ratio,
            next_earnings_date=metadata.next_earnings_date,
            buy_score=buy_score,
            buy_target=buy_target,
            sell_target=sell_target,
            note="",
        )
    except Exception as exc:
        metadata = metadata_map.get(symbol.upper(), _empty_stock_metadata(symbol))
        return EnrichedRow(
            symbol,
            "error",
            metadata.company_name,
            metadata.business_summary,
            metadata.metadata_status,
            metadata.metadata_note,
            None,
            None,
            None,
            None,
            None,
            metadata.market_cap,
            metadata.sector,
            metadata.industry,
            metadata.country,
            metadata.beta,
            metadata.trailing_pe,
            metadata.forward_pe,
            metadata.short_ratio,
            metadata.next_earnings_date,
            None,
            None,
            None,
            f"error: {exc}",
        )


def run_enrichment(
    symbols: List[str],
    rs_lookback: int,
    metadata_cache_path: Optional[str] = DEFAULT_METADATA_CACHE,
    metadata_ttl_days: int = DEFAULT_METADATA_TTL_DAYS,
) -> List[EnrichedRow]:
    spy_hist = fetch_history_yahoo_period("SPY", period="1y")
    if spy_hist is None or spy_hist.empty:
        spy_hist = pd.DataFrame()
    metadata_map = get_stock_metadata_map(symbols, metadata_cache_path, metadata_ttl_days)
    rows: List[EnrichedRow] = []
    for sym in symbols:
        rows.append(enrich_symbol(sym, spy_hist, rs_lookback, metadata_map))
    return rows


def save_enrichment(rows: List[EnrichedRow], csv_path: Optional[str], excel_path: Optional[str]):
    frame = pd.DataFrame(
        {
            "symbol": [r.symbol for r in rows],
            "status": [r.status for r in rows],
            "company_name": [r.company_name for r in rows],
            "sector": [r.sector for r in rows],
            "industry": [r.industry for r in rows],
            "business_summary": [r.business_summary for r in rows],
            "metadata_status": [r.metadata_status for r in rows],
            "metadata_note": [r.metadata_note for r in rows],
            "last_close": [r.last_close for r in rows],
            "avg_vol_20": [r.avg_vol_20 for r in rows],
            "ret_60d_pct": [r.ret_60d_pct for r in rows],
            "rs_spy_60d": [r.rs_spy_60d for r in rows],
            "dist_from_52w_high_pct": [r.dist_from_52w_high_pct for r in rows],
            "market_cap": [r.market_cap for r in rows],
            "country": [r.country for r in rows],
            "beta": [r.beta for r in rows],
            "trailing_pe": [r.trailing_pe for r in rows],
            "forward_pe": [r.forward_pe for r in rows],
            "short_ratio": [r.short_ratio for r in rows],
            "next_earnings_date": [r.next_earnings_date for r in rows],
            "buy_score": [r.buy_score for r in rows],
            "buy_target": [r.buy_target for r in rows],
            "sell_target": [r.sell_target for r in rows],
            "note": [r.note for r in rows],
        }
    )
    if csv_path:
        frame.to_csv(csv_path, index=False)
        print(f"Saved enriched CSV to {csv_path}")
    if excel_path:
        try:
            frame.to_excel(excel_path, index=False)
            print(f"Saved enriched Excel to {excel_path}")
        except ImportError:
            print("openpyxl not installed; skipping enriched Excel export. Install with `pip install openpyxl`.")


def _load_previous_scores(path: str) -> dict:
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(f"Could not read prior scan file {path}: {exc}")
        return {}

    if "symbol" not in df.columns or "score" not in df.columns:
        print(f"Prior scan file {path} missing symbol/score columns; skipping deltas.")
        return {}

    scores = (
        df[["symbol", "score"]]
        .dropna()
        .assign(symbol=lambda x: x["symbol"].astype(str).str.upper())
    )
    return dict(zip(scores["symbol"], scores["score"].astype(float)))


def save_outputs(
    rows: List[ScanRow],
    csv_path: Optional[str],
    excel_path: Optional[str],
    prev_scores: Optional[dict] = None,
    metadata_scope: str = DEFAULT_METADATA_SCOPE,
    metadata_cache_path: Optional[str] = DEFAULT_METADATA_CACHE,
    metadata_ttl_days: int = DEFAULT_METADATA_TTL_DAYS,
    always_include_symbols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "symbol": [r.symbol for r in rows],
            "price_source": [r.price_source for r in rows],
            "status": [r.status for r in rows],
            "last_close": [r.last_close for r in rows],
            "day_change_pct": [r.day_change_pct for r in rows],
            "ema_5": [r.ema_5 for r in rows],
            "ema_10": [r.ema_10 for r in rows],
            "ema_20": [r.ema_20 for r in rows],
            "ema_60": [r.ema_60 for r in rows],
            "ema_250": [r.ema_250 for r in rows],
            "price_above_ema_20": [r.price_above_ema_20 for r in rows],
            "price_above_ema_60": [r.price_above_ema_60 for r in rows],
            "price_above_ema_250": [r.price_above_ema_250 for r in rows],
            "ema_stack_bullish": [r.ema_stack_bullish for r in rows],
            "ema_signal_note": [r.ema_signal_note for r in rows],
            "trend_score": [r.trend_score for r in rows],
            "daily_high": [r.daily_high for r in rows],
            "daily_low": [r.daily_low for r in rows],
            "daily_volume": [r.daily_volume for r in rows],
            "daily_turnover": [r.daily_turnover for r in rows],
            "pivot_high": [r.pivot_high for r in rows],
            "contractions_pct": [r.contractions_pct for r in rows],
            "volume_trend_ok": [r.volume_trend_ok for r in rows],
            "score": [r.score for r in rows],
            "bars": [r.bars for r in rows],
            "note": [r.note for r in rows],
        }
    )

    if prev_scores:
        frame["score_delta"] = frame.apply(
            lambda r: r["score"] - prev_scores.get(str(r["symbol"]).upper(), r["score"])
            if pd.notna(r["score"]) else None,
            axis=1,
        )

    frame = attach_metadata_to_scan_frame(
        frame,
        metadata_scope=metadata_scope,
        metadata_cache_path=metadata_cache_path,
        metadata_ttl_days=metadata_ttl_days,
        always_include_symbols=always_include_symbols,
    )

    if frame.empty:
        print("No VCP-like setups found. Writing empty outputs.")

    if csv_path:
        frame.to_csv(csv_path, index=False)
        print(f"Saved CSV to {csv_path}")
    if excel_path:
        try:
            frame.to_excel(excel_path, index=False)
            print(f"Saved Excel to {excel_path}")
        except ImportError:
            print("openpyxl not installed; skipping Excel export. Install with `pip install openpyxl`.")
    return frame


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan US equities for VCP-like setups using daily data.")
    parser.add_argument("--max-tickers", type=int, default=1000, help="Limit number of symbols to scan (after sorting). Use 0 for all.")
    parser.add_argument("--lookback-days", type=int, default=260, help="Daily bars to request from yfinance.")
    parser.add_argument("--workers", type=int, default=8, help="Thread workers for download/analysis.")
    parser.add_argument("--csv", dest="csv_path", default="vcp_scan.csv", help="Output CSV path.")
    parser.add_argument("--excel", dest="excel_path", default=None, help="Optional Excel output path.")
    parser.add_argument("--timeout", type=int, default=15, help="HTTP timeout seconds for symbol fetch.")
    parser.add_argument(
        "--symbol-source",
        choices=["auto", "nasdaq", "sec"],
        default="auto",
        help="Choose symbol universe source: Nasdaq feeds, SEC list, or auto fallback.",
    )
    parser.add_argument(
        "--sec-user-agent",
        dest="sec_user_agent",
        default="Mozilla/5.0 (compatible; VCPScanner/1.0; +https://example.com/contact)",
        help="Custom User-Agent for SEC endpoints (use your email/domain).",
    )
    parser.add_argument(
        "--price-source",
        choices=["yahoo", "futu"],
        default="yahoo",
        help="Price/volume data source for OHLCV history.",
    )
    parser.add_argument("--futu-host", default="127.0.0.1", help="Futu OpenD host (for price-source futu).")
    parser.add_argument("--futu-port", type=int, default=11111, help="Futu OpenD port (for price-source futu).")
    parser.add_argument(
        "--futu-fallback-yahoo",
        action="store_true",
        help="If set, fall back to Yahoo when Futu returns no data.",
    )
    parser.add_argument("--post-filter", default=None, help="Path to CSV of filtered symbols to enrich (skip scan).")
    parser.add_argument("--enrich-csv", default="vcp_enriched.csv", help="Output CSV for enriched analysis.")
    parser.add_argument("--enrich-excel", default=None, help="Output Excel for enriched analysis.")
    parser.add_argument("--rs-lookback", type=int, default=60, help="Lookback days for relative strength vs SPY.")
    parser.add_argument(
        "--metadata-scope",
        choices=["off", "filtered", "all"],
        default=DEFAULT_METADATA_SCOPE,
        help="Metadata enrichment scope for scan exports: off, score>=4 rows only, or all non-fetch-error rows.",
    )
    parser.add_argument(
        "--metadata-cache",
        default=DEFAULT_METADATA_CACHE,
        help="Local JSON cache file for company metadata.",
    )
    parser.add_argument(
        "--metadata-ttl-days",
        type=int,
        default=DEFAULT_METADATA_TTL_DAYS,
        help="Refresh successful cached metadata after this many days.",
    )
    parser.add_argument(
        "--score-delta-from",
        default=None,
        help="Path to prior vcp_scan.csv to compute score deltas against.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None):
    args = parse_args(argv)
    # If post-filter is provided, skip scanning and enrich the given symbols.
    if args.post_filter:
        symbols = _load_symbols_from_csv(args.post_filter)
        print(f"Enriching {len(symbols)} symbols from {args.post_filter}")
        rows = run_enrichment(
            symbols,
            rs_lookback=args.rs_lookback,
            metadata_cache_path=args.metadata_cache,
            metadata_ttl_days=args.metadata_ttl_days,
        )
        save_enrichment(rows, args.enrich_csv, args.enrich_excel)
        return

    print("Fetching symbol universe...")
    symbols = fetch_symbols(timeout=args.timeout, source=args.symbol_source, sec_user_agent=args.sec_user_agent)
    if args.max_tickers > 0:
        symbols = symbols[: args.max_tickers]
    print(f"Analyzing {len(symbols)} symbols with lookback {args.lookback_days} days using {args.workers} workers")

    start = dt.datetime.now()
    candidates, rows = run_scan(
        symbols,
        args.lookback_days,
        args.workers,
        args.price_source,
        args.futu_host,
        args.futu_port,
        args.futu_fallback_yahoo,
    )
    elapsed = (dt.datetime.now() - start).total_seconds()
    print(f"Finished scan in {elapsed:.1f}s. Candidates: {len(candidates)}")

    prev_scores = _load_previous_scores(args.score_delta_from) if args.score_delta_from else None
    save_outputs(
        rows,
        args.csv_path,
        args.excel_path,
        prev_scores,
        metadata_scope=args.metadata_scope,
        metadata_cache_path=args.metadata_cache,
        metadata_ttl_days=args.metadata_ttl_days,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
