import datetime as dt
import html
from pathlib import Path

import pandas as pd

from stock_analysis.scanner import StockMetadata

from .daily_job_constants import SECTION_ROWS_PER_BLOCK
from .scan_history_file_helpers import calculate_consecutive_score4plus_streaks


def build_daily_watchlist_html_message(
    today: dt.date,
    full_scan_frame: pd.DataFrame,
    blacklisted_symbols: set[str],
    focus_symbols: set[str],
    history_dir: Path,
    fallback_metadata: dict[str, StockMetadata] | None = None,
    eligible_score_symbols: set[str] | None = None,
    message_mode: str = "full",
) -> str:
    if full_scan_frame.empty or "symbol" not in full_scan_frame.columns:
        return f"<b>{today.isoformat()} VCP watchlist</b>\n<pre>(no stocks)</pre>"

    frame = full_scan_frame.copy()
    frame["symbol"] = frame["symbol"].astype(str).str.upper()

    defaults = {
        "score": 0,
        "score_delta": 0,
        "last_close": None,
        "day_change_pct": None,
        "company_name": "",
        "trend_score": 0,
        "price_above_ema_20": None,
        "price_above_ema_60": None,
        "price_above_ema_250": None,
        "ema_stack_bullish": None,
    }
    for column, value in defaults.items():
        if column not in frame.columns:
            frame[column] = value

    if blacklisted_symbols:
        frame = frame[(~frame["symbol"].isin(blacklisted_symbols)) | (frame["symbol"].isin(focus_symbols))].copy()
    frame["score"] = pd.to_numeric(frame["score"], errors="coerce").fillna(0)
    frame["score_delta"] = pd.to_numeric(frame["score_delta"], errors="coerce").fillna(0)
    frame["prev_score"] = frame["score"] - frame["score_delta"]

    streak_days = calculate_consecutive_score4plus_streaks(history_dir, today)
    frame["days_ge_4"] = frame["symbol"].map(lambda symbol: streak_days.get(str(symbol).upper(), 0))

    focus_frame = frame[frame["symbol"].isin(focus_symbols)].copy()
    present_focus = set(focus_frame["symbol"].astype(str).str.upper()) if not focus_frame.empty else set()
    missing_focus = sorted(focus_symbols - present_focus)
    if missing_focus:
        extra_names = []
        for symbol in missing_focus:
            metadata = (fallback_metadata or {}).get(symbol.upper())
            extra_names.append(metadata.company_name if metadata is not None else "")
        extra = pd.DataFrame(
            {
                "symbol": missing_focus,
                "company_name": extra_names,
                "last_close": [None] * len(missing_focus),
                "day_change_pct": [None] * len(missing_focus),
                "score": [0] * len(missing_focus),
                "score_delta": [0.0] * len(missing_focus),
                "prev_score": [0.0] * len(missing_focus),
                "days_ge_4": [0] * len(missing_focus),
                "trend_score": [0] * len(missing_focus),
                "price_above_ema_20": [None] * len(missing_focus),
                "price_above_ema_60": [None] * len(missing_focus),
                "price_above_ema_250": [None] * len(missing_focus),
                "ema_stack_bullish": [None] * len(missing_focus),
            }
        )
        focus_frame = pd.concat([focus_frame, extra], ignore_index=True)

    non_focus = frame[~frame["symbol"].isin(focus_symbols)].copy()
    newly_frame = non_focus[(non_focus["score"] >= 4) & (non_focus["prev_score"] < 4)].copy()
    existing_frame = non_focus[(non_focus["score"] >= 4) & (non_focus["prev_score"] >= 4)].copy()
    dropped_frame = non_focus[(non_focus["score"] < 4) & (non_focus["prev_score"] >= 4)].copy()

    if eligible_score_symbols is not None:
        newly_frame = newly_frame[newly_frame["symbol"].isin(eligible_score_symbols)].copy()
        existing_frame = existing_frame[existing_frame["symbol"].isin(eligible_score_symbols)].copy()

    def flag_text(value) -> str:
        if value is True:
            return "Y"
        if value is False:
            return "N"
        return "-"

    def rows_from(df: pd.DataFrame) -> list[tuple[str, str, float, int, float, float | None, int, int, str, str, str, str]]:
        rows: list[tuple[str, str, float, int, float, float | None, int, int, str, str, str, str]] = []
        if df.empty:
            return rows
        sort_cols = ["score", "symbol"]
        ascending = [False, True]
        if "trend_score" in df.columns:
            sort_cols = ["score", "trend_score", "symbol"]
            ascending = [False, False, True]
        df = df.sort_values(by=sort_cols, ascending=ascending)
        for _, row in df.iterrows():
            symbol = str(row.get("symbol", "")).upper().strip()
            if not symbol:
                continue
            name = str(row.get("company_name") or "").strip()
            try:
                price = float(row.get("last_close")) if pd.notna(row.get("last_close")) else 0.0
            except Exception:
                price = 0.0
            try:
                score = int(row.get("score", 0))
            except Exception:
                score = 0
            try:
                delta = float(row.get("score_delta", 0.0))
            except Exception:
                delta = 0.0
            try:
                days = int(row.get("days_ge_4", 0))
            except Exception:
                days = 0
            try:
                trend_score = int(row.get("trend_score", 0))
            except Exception:
                trend_score = 0
            try:
                day_change_pct = float(row.get("day_change_pct")) if pd.notna(row.get("day_change_pct")) else None
            except Exception:
                day_change_pct = None
            rows.append(
                (
                    symbol,
                    name,
                    price,
                    score,
                    delta,
                    day_change_pct,
                    days,
                    trend_score,
                    flag_text(row.get("price_above_ema_20")),
                    flag_text(row.get("price_above_ema_60")),
                    flag_text(row.get("price_above_ema_250")),
                    flag_text(row.get("ema_stack_bullish")),
                )
            )
        return rows

    def render_section_blocks(
        title: str,
        rows: list[tuple[str, str, float, int, float, float | None, int, int, str, str, str, str]],
        show_days: bool,
        show_ema: bool,
    ) -> list[str]:
        if not rows:
            return [f"<b>{html.escape(title)} (0)</b>\n<pre>(none)</pre>"]

        symbol_width = max(6, max(len(row[0]) for row in rows))
        name_width = 24
        if show_days and show_ema:
            header = f"{'symbol':<{symbol_width}} {'name':<{name_width}} {'price':>10} {'score':>5} {'trnd':>4} {'e20':>3} {'e60':>3} {'e250':>4} {'stk':>3} {'delta':>6} {'chg%':>8} {'days':>5}"
        elif show_days:
            header = f"{'symbol':<{symbol_width}} {'name':<{name_width}} {'price':>10} {'score':>5} {'delta':>6} {'chg%':>8} {'days':>5}"
        elif show_ema:
            header = f"{'symbol':<{symbol_width}} {'name':<{name_width}} {'price':>10} {'score':>5} {'trnd':>4} {'e20':>3} {'e60':>3} {'e250':>4} {'stk':>3} {'delta':>6} {'chg%':>8}"
        else:
            header = f"{'symbol':<{symbol_width}} {'name':<{name_width}} {'price':>10} {'score':>5} {'delta':>6} {'chg%':>8}"

        blocks: list[str] = []
        total = len(rows)
        start = 0
        part = 1
        while start < total:
            chunk = rows[start : start + SECTION_ROWS_PER_BLOCK]
            lines = [header]
            for symbol, name, price, score, delta, change_pct, days, trend_score, e20, e60, e250, stack in chunk:
                short_name = (name[: name_width - 1] + "...") if len(name) > name_width else name
                change_text = f"{change_pct:+.2f}%" if change_pct is not None else "   n/a"
                if show_days and show_ema:
                    line = f"{symbol:<{symbol_width}} {short_name:<{name_width}} {price:>10.2f} {score:>5} {trend_score:>4} {e20:>3} {e60:>3} {e250:>4} {stack:>3} {delta:+6.1f} {change_text:>8} {days:>5}"
                elif show_days:
                    line = f"{symbol:<{symbol_width}} {short_name:<{name_width}} {price:>10.2f} {score:>5} {delta:+6.1f} {change_text:>8} {days:>5}"
                elif show_ema:
                    line = f"{symbol:<{symbol_width}} {short_name:<{name_width}} {price:>10.2f} {score:>5} {trend_score:>4} {e20:>3} {e60:>3} {e250:>4} {stack:>3} {delta:+6.1f} {change_text:>8}"
                else:
                    line = f"{symbol:<{symbol_width}} {short_name:<{name_width}} {price:>10.2f} {score:>5} {delta:+6.1f} {change_text:>8}"
                lines.append(html.escape(line))

            end = min(start + SECTION_ROWS_PER_BLOCK, total)
            section_title = f"{title} ({total}) [part {part}: {start + 1}-{end}]"
            blocks.append(f"<b>{html.escape(section_title)}</b>\n<pre>{'\n'.join(lines)}</pre>")
            start = end
            part += 1

        return blocks

    focus_count = 0 if focus_frame.empty else int(focus_frame["symbol"].nunique())
    new_count = 0 if newly_frame.empty else int(newly_frame["symbol"].nunique())
    existing_count = 0 if existing_frame.empty else int(existing_frame["symbol"].nunique())
    dropped_count = 0 if dropped_frame.empty else int(dropped_frame["symbol"].nunique())
    summary = f"<b>Summary:</b> Focus {focus_count} | New {new_count} | Existing {existing_count} | Dropped {dropped_count}"

    sections: list[str] = []
    sections.extend(render_section_blocks("Focused symbols (always shown)", rows_from(focus_frame), show_days=False, show_ema=False))
    if message_mode == "compact":
        compact_lines = [
            f"newly_ge_4   {new_count}",
            f"existing_ge_4 {existing_count}",
            f"dropped_lt_4 {dropped_count}",
        ]
        sections.append(f"<b>Compact score summary</b>\n<pre>{html.escape(chr(10).join(compact_lines))}</pre>")
    else:
        sections.extend(render_section_blocks("Newly achieved score >= 4", rows_from(newly_frame), show_days=False, show_ema=True))
        sections.extend(render_section_blocks("Existing score >= 4", rows_from(existing_frame), show_days=True, show_ema=True))
        sections.extend(render_section_blocks("Dropped below 4 (was >= 4 previously)", rows_from(dropped_frame), show_days=False, show_ema=False))

    return f"<b>{today.isoformat()} VCP watchlist</b>\n{summary}\n\n" + "\n\n".join(sections)


build_message = build_daily_watchlist_html_message
