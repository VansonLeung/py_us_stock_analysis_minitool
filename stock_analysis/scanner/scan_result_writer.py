from collections.abc import Sequence

import pandas as pd

from .market_data_fetchers import fetch_yahoo_symbol_history_for_period
from .scan_models import (
    DEFAULT_METADATA_CACHE,
    DEFAULT_METADATA_SCOPE,
    DEFAULT_METADATA_TTL_DAYS,
    EnrichedRow,
    ScanRow,
    StockMetadata,
)
from .stock_metadata_service import (
    attach_stock_metadata_to_scan_frame,
    build_empty_stock_metadata_model,
    load_stock_metadata_by_symbol,
)


def load_symbol_list_from_csv_file(path: str) -> list[str]:
    frame = pd.read_csv(path)
    symbol_column = None
    for candidate in ["symbol", "Symbol", "ticker", "Ticker"]:
        if candidate in frame.columns:
            symbol_column = candidate
            break
    if symbol_column is None:
        raise ValueError("CSV must contain a symbol/ticker column")
    return sorted(frame[symbol_column].dropna().astype(str).str.upper().unique())


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _scale(value: float | None, low: float, high: float) -> float:
    if value is None or high == low:
        return 0.0
    return _clip01((value - low) / (high - low))


def build_enriched_symbol_analysis(
    symbol: str,
    spy_hist: pd.DataFrame,
    rs_lookback: int,
    metadata_map: dict[str, StockMetadata],
) -> EnrichedRow:
    try:
        history_frame = fetch_yahoo_symbol_history_for_period(symbol, period="1y")
        metadata = metadata_map.get(symbol.upper(), build_empty_stock_metadata_model(symbol))
        if history_frame is None or history_frame.empty:
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

        last_close = float(history_frame["Close"].iloc[-1])
        avg_vol_20 = float(history_frame["Volume"].tail(20).mean()) if len(history_frame) >= 1 else None
        ret_60 = None
        rs_spy = None
        if len(history_frame) > rs_lookback and len(spy_hist) > rs_lookback:
            ret_60 = float(history_frame["Close"].pct_change(rs_lookback).iloc[-1])
            spy_ret = float(spy_hist["Close"].pct_change(rs_lookback).iloc[-1])
            rs_spy = ret_60 - spy_ret if spy_ret is not None else None

        high_52w = float(history_frame["High"].rolling(252, min_periods=1).max().iloc[-1])
        dist_high = (last_close / high_52w - 1.0) if high_52w else None
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
        metadata = metadata_map.get(symbol.upper(), build_empty_stock_metadata_model(symbol))
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


def build_enriched_symbol_analysis_list(
    symbols: list[str],
    rs_lookback: int,
    metadata_cache_path: str | None = DEFAULT_METADATA_CACHE,
    metadata_ttl_days: int = DEFAULT_METADATA_TTL_DAYS,
) -> list[EnrichedRow]:
    spy_history_frame = fetch_yahoo_symbol_history_for_period("SPY", period="1y")
    if spy_history_frame is None or spy_history_frame.empty:
        spy_history_frame = pd.DataFrame()
    metadata_map = load_stock_metadata_by_symbol(symbols, metadata_cache_path, metadata_ttl_days)
    return [build_enriched_symbol_analysis(symbol, spy_history_frame, rs_lookback, metadata_map) for symbol in symbols]


def save_enriched_analysis_outputs(rows: list[EnrichedRow], csv_path: str | None, excel_path: str | None) -> None:
    frame = pd.DataFrame(
        {
            "symbol": [row.symbol for row in rows],
            "status": [row.status for row in rows],
            "company_name": [row.company_name for row in rows],
            "sector": [row.sector for row in rows],
            "industry": [row.industry for row in rows],
            "business_summary": [row.business_summary for row in rows],
            "metadata_status": [row.metadata_status for row in rows],
            "metadata_note": [row.metadata_note for row in rows],
            "last_close": [row.last_close for row in rows],
            "avg_vol_20": [row.avg_vol_20 for row in rows],
            "ret_60d_pct": [row.ret_60d_pct for row in rows],
            "rs_spy_60d": [row.rs_spy_60d for row in rows],
            "dist_from_52w_high_pct": [row.dist_from_52w_high_pct for row in rows],
            "market_cap": [row.market_cap for row in rows],
            "country": [row.country for row in rows],
            "beta": [row.beta for row in rows],
            "trailing_pe": [row.trailing_pe for row in rows],
            "forward_pe": [row.forward_pe for row in rows],
            "short_ratio": [row.short_ratio for row in rows],
            "next_earnings_date": [row.next_earnings_date for row in rows],
            "buy_score": [row.buy_score for row in rows],
            "buy_target": [row.buy_target for row in rows],
            "sell_target": [row.sell_target for row in rows],
            "note": [row.note for row in rows],
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


def load_previous_scan_score_map(path: str) -> dict:
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        print(f"Could not read prior scan file {path}: {exc}")
        return {}

    if "symbol" not in frame.columns or "score" not in frame.columns:
        print(f"Prior scan file {path} missing symbol/score columns; skipping deltas.")
        return {}

    scores = (
        frame[["symbol", "score"]]
        .dropna()
        .assign(symbol=lambda data: data["symbol"].astype(str).str.upper())
    )
    return dict(zip(scores["symbol"], scores["score"].astype(float)))


def save_scan_output_files(
    rows: list[ScanRow],
    csv_path: str | None,
    excel_path: str | None,
    prev_scores: dict | None = None,
    metadata_scope: str = DEFAULT_METADATA_SCOPE,
    metadata_cache_path: str | None = DEFAULT_METADATA_CACHE,
    metadata_ttl_days: int = DEFAULT_METADATA_TTL_DAYS,
    always_include_symbols: Sequence[str] | None = None,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "symbol": [row.symbol for row in rows],
            "price_source": [row.price_source for row in rows],
            "status": [row.status for row in rows],
            "last_close": [row.last_close for row in rows],
            "day_change_pct": [row.day_change_pct for row in rows],
            "ema_5": [row.ema_5 for row in rows],
            "ema_10": [row.ema_10 for row in rows],
            "ema_20": [row.ema_20 for row in rows],
            "ema_60": [row.ema_60 for row in rows],
            "ema_250": [row.ema_250 for row in rows],
            "price_above_ema_20": [row.price_above_ema_20 for row in rows],
            "price_above_ema_60": [row.price_above_ema_60 for row in rows],
            "price_above_ema_250": [row.price_above_ema_250 for row in rows],
            "ema_stack_bullish": [row.ema_stack_bullish for row in rows],
            "ema_signal_note": [row.ema_signal_note for row in rows],
            "trend_score": [row.trend_score for row in rows],
            "daily_high": [row.daily_high for row in rows],
            "daily_low": [row.daily_low for row in rows],
            "daily_volume": [row.daily_volume for row in rows],
            "daily_turnover": [row.daily_turnover for row in rows],
            "pivot_high": [row.pivot_high for row in rows],
            "contractions_pct": [row.contractions_pct for row in rows],
            "volume_trend_ok": [row.volume_trend_ok for row in rows],
            "score": [row.score for row in rows],
            "bars": [row.bars for row in rows],
            "note": [row.note for row in rows],
        }
    )

    if prev_scores:
        frame["score_delta"] = frame.apply(
            lambda row: row["score"] - prev_scores.get(str(row["symbol"]).upper(), row["score"])
            if pd.notna(row["score"]) else None,
            axis=1,
        )

    frame = attach_stock_metadata_to_scan_frame(
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


_load_previous_scores = load_previous_scan_score_map
_load_symbols_from_csv = load_symbol_list_from_csv_file
enrich_symbol = build_enriched_symbol_analysis
load_previous_scores = load_previous_scan_score_map
load_symbols_from_csv = load_symbol_list_from_csv_file
run_enrichment = build_enriched_symbol_analysis_list
save_enrichment = save_enriched_analysis_outputs
save_outputs = save_scan_output_files
