import concurrent.futures
from collections.abc import Iterable, Sequence

import pandas as pd
from tqdm import tqdm

from .market_data_fetchers import fetch_symbol_price_history
from .scan_models import ScanRow, VCPResult


def find_local_extrema_indices(values: Sequence[float], window: int, find_max: bool) -> list[int]:
    indexes: list[int] = []
    for idx in range(window, len(values) - window):
        segment = values[idx - window : idx + window + 1]
        center = values[idx]
        if find_max and center == max(segment):
            indexes.append(idx)
        if not find_max and center == min(segment):
            indexes.append(idx)
    return indexes


def detect_vcp_pattern(df: pd.DataFrame) -> VCPResult | None:
    closes = df["Close"].to_numpy()
    volumes = df["Volume"].to_numpy()
    if len(closes) < 120:
        return None

    highs = find_local_extrema_indices(closes, window=3, find_max=True)
    lows = find_local_extrema_indices(closes, window=3, find_max=False)
    labeled = [(idx, "H") for idx in highs] + [(idx, "L") for idx in lows]
    labeled.sort(key=lambda item: item[0])

    pivots: list[tuple[int, str]] = []
    last_type = None
    for idx, kind in labeled:
        if kind == last_type:
            continue
        pivots.append((idx, kind))
        last_type = kind

    if not pivots or pivots[0][1] != "H":
        pivots = [pivot for pivot in pivots if pivot[1] == "H"] + [pivot for pivot in pivots if pivot[1] == "L"]
        pivots.sort(key=lambda item: item[0])
        if not pivots or pivots[0][1] != "H":
            return None

    drops: list[float] = []
    contraction_volumes: list[float] = []
    for idx in range(len(pivots) - 1):
        (high_idx, high_type), (low_idx, low_type) = pivots[idx], pivots[idx + 1]
        if high_type != "H" or low_type != "L" or low_idx <= high_idx:
            continue
        high = closes[high_idx]
        low = closes[low_idx]
        drops.append((high - low) / high)
        contraction_volumes.append(volumes[high_idx : low_idx + 1].mean())
    if len(drops) < 3:
        return None

    drops = drops[-3:]
    contraction_volumes = contraction_volumes[-3:]
    decreasing_drops = drops[0] > drops[1] > drops[2] and drops[2] > 0
    volume_ok = contraction_volumes[0] > contraction_volumes[1] > contraction_volumes[2]

    last_pivot_high_idx = pivots[-1][0] if pivots[-1][1] == "H" else pivots[-2][0]
    pivot_high = closes[last_pivot_high_idx]
    last_close = closes[-1]
    near_pivot = last_close >= 0.95 * pivot_high

    score = 0
    score += 2 if decreasing_drops else 0
    score += 1 if volume_ok else 0
    score += 1 if near_pivot else 0

    if score == 4:
        drop_ratio_1 = drops[1] / drops[0] if drops[0] > 0 else 1.0
        drop_ratio_2 = drops[2] / drops[1] if drops[1] > 0 else 1.0
        very_near_pivot = last_close >= 0.985 * pivot_high
        strong_volume_dryup = contraction_volumes[2] <= 0.7 * contraction_volumes[0] if contraction_volumes[0] > 0 else False
        tight_final_drop = drops[2] <= 0.06
        smooth_shrink = drop_ratio_1 <= 0.8 and drop_ratio_2 <= 0.8
        if very_near_pivot and strong_volume_dryup and tight_final_drop and smooth_shrink:
            score = 5

    if score == 0:
        return None

    notes = []
    if not decreasing_drops:
        notes.append("drops not shrinking")
    if not volume_ok:
        notes.append("volume not contracting")
    if not near_pivot:
        notes.append("price not near pivot")
    if score == 5:
        notes.append("perfect vcp setup")

    return VCPResult(
        symbol=df.attrs.get("symbol", ""),
        last_close=float(last_close),
        pivot_high=float(pivot_high),
        contractions=[round(drop * 100, 2) for drop in drops],
        volume_trend_ok=volume_ok,
        days_looked=len(closes),
        bars=len(closes),
        score=score,
        note="; ".join(notes),
    )


def _compute_latest_ema_value(close_series: pd.Series, span: int) -> float | None:
    if close_series.empty:
        return None
    ema_series = close_series.ewm(span=span, adjust=False).mean().dropna()
    if ema_series.empty:
        return None
    return float(ema_series.iloc[-1])


def _build_ema_signal_summary(
    last_close: float,
    ema_5: float | None,
    ema_10: float | None,
    ema_20: float | None,
    ema_60: float | None,
    ema_250: float | None,
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


def _calculate_trend_alignment_score(
    last_close: float,
    ema_5: float | None,
    ema_10: float | None,
    ema_20: float | None,
    ema_60: float | None,
    ema_250: float | None,
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


def _build_symbol_scan_row_model(symbol: str, price_source: str, status: str, note: str, **overrides) -> ScanRow:
    payload = {
        "symbol": symbol,
        "price_source": price_source,
        "status": status,
        "last_close": None,
        "day_change_pct": None,
        "ema_5": None,
        "ema_10": None,
        "ema_20": None,
        "ema_60": None,
        "ema_250": None,
        "price_above_ema_20": None,
        "price_above_ema_60": None,
        "price_above_ema_250": None,
        "ema_stack_bullish": None,
        "ema_signal_note": "",
        "trend_score": 0,
        "daily_high": None,
        "daily_low": None,
        "daily_volume": None,
        "daily_turnover": None,
        "pivot_high": None,
        "contractions_pct": "",
        "volume_trend_ok": None,
        "score": 0,
        "bars": 0,
        "note": note,
    }
    payload.update(overrides)
    return ScanRow(**payload)


def analyze_symbol_vcp_state(
    symbol: str,
    lookback_days: int,
    price_source: str,
    futu_host: str,
    futu_port: int,
    futu_fallback_yahoo: bool,
) -> tuple[VCPResult | None, ScanRow]:
    try:
        price_frame = fetch_symbol_price_history(symbol, lookback_days, price_source, futu_host, futu_port, futu_fallback_yahoo)
        if price_frame is None or price_frame.empty:
            return None, _build_symbol_scan_row_model(symbol, price_source, "fetch_error", "no data")

        vcp_result = detect_vcp_pattern(price_frame)
        close_series = pd.to_numeric(price_frame["Close"], errors="coerce").dropna()
        last_close = float(price_frame["Close"].iloc[-1])
        prev_close = float(price_frame["Close"].iloc[-2]) if len(price_frame) >= 2 else None
        day_change_pct = ((last_close / prev_close) - 1.0) * 100.0 if prev_close not in (None, 0) else None
        ema_5 = _compute_latest_ema_value(close_series, 5)
        ema_10 = _compute_latest_ema_value(close_series, 10)
        ema_20 = _compute_latest_ema_value(close_series, 20)
        ema_60 = _compute_latest_ema_value(close_series, 60)
        ema_250 = _compute_latest_ema_value(close_series, 250)
        price_above_ema_20 = (last_close > ema_20) if ema_20 is not None else None
        price_above_ema_60 = (last_close > ema_60) if ema_60 is not None else None
        price_above_ema_250 = (last_close > ema_250) if ema_250 is not None else None
        ema_stack_bullish = (
            ema_5 is not None
            and ema_10 is not None
            and ema_20 is not None
            and ema_60 is not None
            and ema_250 is not None
            and ema_5 > ema_10 > ema_20 > ema_60 > ema_250
        )
        ema_signal_note = _build_ema_signal_summary(last_close, ema_5, ema_10, ema_20, ema_60, ema_250)
        trend_score = _calculate_trend_alignment_score(last_close, ema_5, ema_10, ema_20, ema_60, ema_250)

        row_kwargs = {
            "last_close": last_close,
            "day_change_pct": day_change_pct,
            "ema_5": ema_5,
            "ema_10": ema_10,
            "ema_20": ema_20,
            "ema_60": ema_60,
            "ema_250": ema_250,
            "price_above_ema_20": price_above_ema_20,
            "price_above_ema_60": price_above_ema_60,
            "price_above_ema_250": price_above_ema_250,
            "ema_stack_bullish": ema_stack_bullish,
            "ema_signal_note": ema_signal_note,
            "trend_score": trend_score,
            "daily_high": float(price_frame["High"].iloc[-1]),
            "daily_low": float(price_frame["Low"].iloc[-1]),
            "daily_volume": float(price_frame["Volume"].iloc[-1]),
            "daily_turnover": float(last_close * float(price_frame["Volume"].iloc[-1])),
            "bars": len(price_frame),
        }

        if vcp_result:
            row = _build_symbol_scan_row_model(
                symbol,
                price_source,
                "vcp",
                vcp_result.note,
                pivot_high=vcp_result.pivot_high,
                contractions_pct="|".join(map(str, vcp_result.contractions)),
                volume_trend_ok=vcp_result.volume_trend_ok,
                score=int(vcp_result.score),
                **row_kwargs,
            )
            return vcp_result, row

        return None, _build_symbol_scan_row_model(symbol, price_source, "no_pattern", "no VCP pattern", **row_kwargs)
    except Exception as exc:
        return None, _build_symbol_scan_row_model(symbol, price_source, "fetch_error", f"error: {exc}")


def run_vcp_scan_for_symbol_list(
    symbols: Iterable[str],
    lookback_days: int,
    max_workers: int,
    price_source: str,
    futu_host: str,
    futu_port: int,
    futu_fallback_yahoo: bool,
) -> tuple[list[VCPResult], list[ScanRow]]:
    results: list[VCPResult] = []
    rows: list[ScanRow] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(analyze_symbol_vcp_state, symbol, lookback_days, price_source, futu_host, futu_port, futu_fallback_yahoo): symbol
            for symbol in symbols
        }
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Scanning"):
            vcp_result, scan_row = future.result()
            if vcp_result:
                results.append(vcp_result)
            rows.append(scan_row)
    results.sort(key=lambda result: (-result.score, result.symbol))
    rows.sort(key=lambda row: row.symbol)
    return results, rows


analyze_symbol = analyze_symbol_vcp_state
detect_vcp = detect_vcp_pattern
local_extrema = find_local_extrema_indices
run_scan = run_vcp_scan_for_symbol_list
