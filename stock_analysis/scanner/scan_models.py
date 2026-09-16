from dataclasses import dataclass


DEFAULT_METADATA_SCOPE = "filtered"
DEFAULT_METADATA_CACHE = "stock_metadata_cache.json"
DEFAULT_METADATA_TTL_DAYS = 300
METADATA_ERROR_TTL_HOURS = 2400
METADATA_MISSING_RETRY_ATTEMPTS = 1
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
    contractions: list[float]
    volume_trend_ok: bool
    days_looked: int
    bars: int
    score: float
    note: str


@dataclass
class ScanRow:
    symbol: str
    price_source: str
    status: str
    last_close: float | None
    day_change_pct: float | None
    ema_5: float | None
    ema_10: float | None
    ema_20: float | None
    ema_60: float | None
    ema_250: float | None
    price_above_ema_20: bool | None
    price_above_ema_60: bool | None
    price_above_ema_250: bool | None
    ema_stack_bullish: bool | None
    ema_signal_note: str
    trend_score: int
    daily_high: float | None
    daily_low: float | None
    daily_volume: float | None
    daily_turnover: float | None
    pivot_high: float | None
    contractions_pct: str
    volume_trend_ok: bool | None
    score: int
    bars: int
    note: str


@dataclass
class EnrichedRow:
    symbol: str
    status: str
    company_name: str
    business_summary: str
    metadata_status: str
    metadata_note: str
    last_close: float | None
    avg_vol_20: float | None
    ret_60d_pct: float | None
    rs_spy_60d: float | None
    dist_from_52w_high_pct: float | None
    market_cap: float | None
    sector: str | None
    industry: str | None
    country: str | None
    beta: float | None
    trailing_pe: float | None
    forward_pe: float | None
    short_ratio: float | None
    next_earnings_date: str | None
    buy_score: float | None
    buy_target: float | None
    sell_target: float | None
    note: str


@dataclass
class StockMetadata:
    symbol: str
    company_name: str
    sector: str | None
    industry: str | None
    business_summary: str
    country: str | None
    market_cap: float | None
    beta: float | None
    trailing_pe: float | None
    forward_pe: float | None
    short_ratio: float | None
    next_earnings_date: str | None
    metadata_status: str
    metadata_note: str
    metadata_fetched_at: str
