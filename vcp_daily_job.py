import argparse
import datetime as dt
import html
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
import yfinance as yf

import main as scanner


WEBHOOK_URL = "https://tgbot.www.vanportdev.com/msg/1348940059"
BLACKLIST_FILE_NAME = "blacklisted_symbols.txt"
FOCUS_FILE_NAME = "focus_symbols.txt"
TELEGRAM_MSG_MAX_CHARS = 3500
SECTION_ROWS_PER_BLOCK = 20


def _dated_output_paths(base_dir: Path, day: dt.date) -> tuple[Path, Path]:
    date_key = day.strftime("%Y%m%d")
    return base_dir / f"{date_key}_vcp_scan.csv", base_dir / f"{date_key}_vcp_scan.xlsx"


def _filtered_output_paths(base_dir: Path, day: dt.date) -> tuple[Path, Path]:
    date_key = day.strftime("%Y%m%d")
    return (
        base_dir / f"{date_key}_vcp_scan_score4plus_filtered.csv",
        base_dir / f"{date_key}_vcp_scan_score4plus_filtered.xlsx",
    )


def _parse_dated_scan_filename(path: Path) -> Optional[dt.date]:
    name = path.name
    if not name.endswith("_vcp_scan.csv"):
        return None
    prefix = name.split("_", 1)[0]
    if len(prefix) != 8 or not prefix.isdigit():
        return None
    try:
        return dt.datetime.strptime(prefix, "%Y%m%d").date()
    except ValueError:
        return None


def _find_previous_scan_csv(base_dir: Path, today: dt.date) -> Optional[Path]:
    dated_files: list[tuple[dt.date, Path]] = []
    for path in base_dir.glob("*_vcp_scan.csv"):
        day = _parse_dated_scan_filename(path)
        if day is None:
            continue
        if day < today:
            dated_files.append((day, path))
    if not dated_files:
        return None
    dated_files.sort(key=lambda x: x[0], reverse=True)
    return dated_files[0][1]


def _list_dated_scan_csvs(base_dir: Path, today: dt.date) -> list[tuple[dt.date, Path]]:
    dated_files: list[tuple[dt.date, Path]] = []
    for path in base_dir.glob("*_vcp_scan.csv"):
        day = _parse_dated_scan_filename(path)
        if day is None or day > today:
            continue
        dated_files.append((day, path))
    dated_files.sort(key=lambda x: x[0])
    return dated_files


def _consecutive_score4plus_days(base_dir: Path, today: dt.date) -> dict[str, int]:
    days_by_symbol: dict[str, int] = {}
    dated_files = _list_dated_scan_csvs(base_dir, today)
    if not dated_files:
        return days_by_symbol

    # Traverse backwards so we count streaks ending at the most recent run.
    for _, path in reversed(dated_files):
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        if "symbol" not in frame.columns or "score" not in frame.columns:
            continue

        frame = frame[["symbol", "score"]].copy()
        frame["symbol"] = frame["symbol"].astype(str).str.upper()
        frame["score"] = pd.to_numeric(frame["score"], errors="coerce").fillna(0)
        today_symbols = set(frame.loc[frame["score"] >= 4, "symbol"].tolist())

        # Start new streaks and extend existing ones; symbols missing today break automatically.
        if not days_by_symbol:
            for sym in today_symbols:
                days_by_symbol[sym] = 1
            continue

        active = set(days_by_symbol.keys())
        for sym in list(active):
            if sym in today_symbols:
                days_by_symbol[sym] += 1
            else:
                del days_by_symbol[sym]
        for sym in (today_symbols - active):
            days_by_symbol[sym] = 1

        if not days_by_symbol:
            break

    return days_by_symbol


def _lookup_symbol_name(symbol: str) -> str:
    try:
        info = yf.Ticker(symbol).get_info()
    except Exception:
        return ""
    return str(info.get("shortName") or info.get("longName") or "").strip()


def _load_symbol_file(path: Path, label: str) -> set[str]:
    if not path.exists():
        print(f"{label} file not found at {path}; continuing with empty list.")
        return set()

    symbols: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        symbols.add(line.upper())
    return symbols


def _load_blacklisted_symbols(blacklist_path: Path) -> set[str]:
    return _load_symbol_file(blacklist_path, "Blacklist")


def _load_focus_symbols(focus_path: Path) -> set[str]:
    return _load_symbol_file(focus_path, "Focus")


def _fetch_price_change_pct(symbol: str) -> Optional[float]:
    try:
        hist = yf.download(
            symbol,
            period="7d",
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        if hist is None or hist.empty:
            return None
        close = hist["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close = pd.to_numeric(close, errors="coerce").dropna()
        if len(close) < 2:
            return None
        prev_close = float(close.iloc[-2])
        last_close = float(close.iloc[-1])
        if prev_close == 0:
            return None
        return (last_close / prev_close - 1.0) * 100.0
    except Exception:
        return None


def _save_score4plus_outputs(
    scan_csv: Path,
    out_csv: Path,
    out_xlsx: Path,
    blacklisted_symbols: set[str],
) -> pd.DataFrame:
    frame = pd.read_csv(scan_csv)
    if "score" not in frame.columns:
        filtered = frame.iloc[0:0].copy()
    else:
        filtered = frame[frame["score"] >= 4].copy()

    if blacklisted_symbols and "symbol" in filtered.columns:
        filtered = filtered[~filtered["symbol"].astype(str).str.upper().isin(blacklisted_symbols)].copy()

    filtered = filtered.sort_values(by=["score", "symbol"], ascending=[False, True]) if not filtered.empty else filtered
    filtered.to_csv(out_csv, index=False)
    filtered.to_excel(out_xlsx, index=False)
    print(f"Saved score>=4 filtered CSV to {out_csv}")
    print(f"Saved score>=4 filtered Excel to {out_xlsx}")
    return filtered


def _build_message(
    today: dt.date,
    full_scan_frame: pd.DataFrame,
    blacklisted_symbols: set[str],
    focus_symbols: set[str],
    base_dir: Path,
) -> str:
    if full_scan_frame.empty or "symbol" not in full_scan_frame.columns:
        return (
            f"<b>{today.isoformat()} VCP watchlist</b>\n"
            "<pre>(no stocks)</pre>"
        )

    frame = full_scan_frame.copy()
    frame["symbol"] = frame["symbol"].astype(str).str.upper()

    if "score" not in frame.columns:
        frame["score"] = 0
    if "score_delta" not in frame.columns:
        frame["score_delta"] = 0
    if "last_close" not in frame.columns:
        frame["last_close"] = None

    # Non-focused symbols still honor blacklist; focused symbols always stay.
    if blacklisted_symbols:
        frame = frame[(~frame["symbol"].isin(blacklisted_symbols)) | (frame["symbol"].isin(focus_symbols))].copy()
    frame["score"] = pd.to_numeric(frame["score"], errors="coerce").fillna(0)
    frame["score_delta"] = pd.to_numeric(frame["score_delta"], errors="coerce").fillna(0)
    frame["prev_score"] = frame["score"] - frame["score_delta"]

    streak_days = _consecutive_score4plus_days(base_dir, today)
    frame["days_ge_4"] = frame["symbol"].map(lambda s: streak_days.get(str(s).upper(), 0))

    # Focus section always visible and on top.
    focus_frame = frame[frame["symbol"].isin(focus_symbols)].copy()

    # If focus symbols are outside scan universe, still include with fallback fields.
    present_focus = set(focus_frame["symbol"].astype(str).str.upper()) if not focus_frame.empty else set()
    missing_focus = sorted(focus_symbols - present_focus)
    if missing_focus:
        extra = pd.DataFrame(
            {
                "symbol": missing_focus,
                "last_close": [None] * len(missing_focus),
                "score": [0] * len(missing_focus),
                "score_delta": [0.0] * len(missing_focus),
                "prev_score": [0.0] * len(missing_focus),
                "days_ge_4": [0] * len(missing_focus),
            }
        )
        focus_frame = pd.concat([focus_frame, extra], ignore_index=True)

    non_focus = frame[~frame["symbol"].isin(focus_symbols)].copy()
    newly_frame = non_focus[(non_focus["score"] >= 4) & (non_focus["prev_score"] < 4)].copy()
    existing_frame = non_focus[(non_focus["score"] >= 4) & (non_focus["prev_score"] >= 4)].copy()
    dropped_frame = non_focus[(non_focus["score"] < 4) & (non_focus["prev_score"] >= 4)].copy()

    def _rows_from(df: pd.DataFrame) -> list[tuple[str, str, float, int, float, Optional[float], int]]:
        rows: list[tuple[str, str, float, int, float, Optional[float], int]] = []
        if df.empty:
            return rows
        df = df.sort_values(by=["score", "symbol"], ascending=[False, True])
        for _, row in df.iterrows():
            symbol = str(row.get("symbol", "")).upper().strip()
            if not symbol:
                continue
            name = _lookup_symbol_name(symbol)
            price_raw = row.get("last_close")
            try:
                price = float(price_raw) if pd.notna(price_raw) else 0.0
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
            chg = _fetch_price_change_pct(symbol)
            rows.append((symbol, name, price, score, delta, chg, days))
        return rows

    def _render_section_blocks(
        title: str,
        rows: list[tuple[str, str, float, int, float, Optional[float], int]],
        show_days: bool,
    ) -> list[str]:
        if not rows:
            return [f"<b>{html.escape(title)} (0)</b>\n<pre>(none)</pre>"]

        symbol_w = max(6, max(len(r[0]) for r in rows))
        name_w = 24
        if show_days:
            header = f"{'symbol':<{symbol_w}} {'name':<{name_w}} {'price':>10} {'score':>5} {'delta':>6} {'chg%':>8} {'days':>5}"
        else:
            header = f"{'symbol':<{symbol_w}} {'name':<{name_w}} {'price':>10} {'score':>5} {'delta':>6} {'chg%':>8}"

        blocks: list[str] = []
        total = len(rows)
        start = 0
        part = 1
        while start < total:
            chunk = rows[start : start + SECTION_ROWS_PER_BLOCK]
            lines = [header]
            for symbol, name, price, score, delta, chg, days in chunk:
                name_short = (name[: name_w - 1] + "...") if len(name) > name_w else name
                chg_text = f"{chg:+.2f}%" if chg is not None else "   n/a"
                if show_days:
                    line = f"{symbol:<{symbol_w}} {name_short:<{name_w}} {price:>10.2f} {score:>5} {delta:+6.1f} {chg_text:>8} {days:>5}"
                else:
                    line = f"{symbol:<{symbol_w}} {name_short:<{name_w}} {price:>10.2f} {score:>5} {delta:+6.1f} {chg_text:>8}"
                lines.append(html.escape(line))

            end = min(start + SECTION_ROWS_PER_BLOCK, total)
            section_title = f"{title} ({total}) [part {part}: {start + 1}-{end}]"
            blocks.append(f"<b>{html.escape(section_title)}</b>\n<pre>{'\n'.join(lines)}</pre>")
            start = end
            part += 1

        return blocks

    sections: list[str] = []
    sections.extend(_render_section_blocks("Focused symbols (always shown)", _rows_from(focus_frame), show_days=False))
    sections.extend(_render_section_blocks("Newly achieved score >= 4", _rows_from(newly_frame), show_days=False))
    sections.extend(_render_section_blocks("Existing score >= 4", _rows_from(existing_frame), show_days=True))
    sections.extend(_render_section_blocks("Dropped below 4 (was >= 4 previously)", _rows_from(dropped_frame), show_days=False))
    focus_count = 0 if focus_frame.empty else int(focus_frame["symbol"].nunique())
    new_count = 0 if newly_frame.empty else int(newly_frame["symbol"].nunique())
    existing_count = 0 if existing_frame.empty else int(existing_frame["symbol"].nunique())
    dropped_count = 0 if dropped_frame.empty else int(dropped_frame["symbol"].nunique())
    summary = (
        f"<b>Summary:</b> "
        f"Focus {focus_count} | New {new_count} | Existing {existing_count} | Dropped {dropped_count}"
    )
    return f"<b>{today.isoformat()} VCP watchlist</b>\n{summary}\n\n" + "\n\n".join(sections)


def _split_message_chunks(message: str, max_chars: int = TELEGRAM_MSG_MAX_CHARS) -> list[str]:
    if len(message) <= max_chars:
        return [message]

    # Prefer splitting on section boundaries first.
    parts = message.split("\n\n")
    chunks: list[str] = []
    cur = ""
    for part in parts:
        candidate = part if not cur else (cur + "\n\n" + part)
        if len(candidate) <= max_chars:
            cur = candidate
            continue
        if cur:
            chunks.append(cur)
            cur = ""

        # Fallback split for a single oversized part.
        while len(part) > max_chars:
            chunks.append(part[:max_chars])
            part = part[max_chars:]
        cur = part
    if cur:
        chunks.append(cur)
    return chunks


def _post_webhook(message: str, timeout: int = 20) -> None:
    chunks = _split_message_chunks(message)
    for idx, chunk in enumerate(chunks, start=1):
        payload = {"msg": chunk}
        response = requests.post(WEBHOOK_URL, json=payload, timeout=timeout)
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            body = response.text[:500] if response.text else ""
            raise requests.HTTPError(f"Webhook chunk {idx}/{len(chunks)} failed: {exc}. Response: {body}") from exc


def _post_file(file_path: Path, timeout: int = 40) -> None:
    with file_path.open("rb") as fh:
        files = {
            "files": (
                file_path.name,
                fh,
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        }
        response = requests.post(WEBHOOK_URL, files=files, timeout=timeout)
        response.raise_for_status()


def run_vcp_job(
    base_dir: Path,
    max_tickers: int,
    lookback_days: int,
    workers: int,
    symbol_source: str,
    price_source: str,
    timeout: int,
    sec_user_agent: str,
    futu_host: str,
    futu_port: int,
    futu_fallback_yahoo: bool,
) -> Path:
    today = dt.date.today()
    blacklist_path = base_dir / BLACKLIST_FILE_NAME
    focus_path = base_dir / FOCUS_FILE_NAME
    blacklisted_symbols = _load_blacklisted_symbols(blacklist_path)
    focus_symbols = _load_focus_symbols(focus_path)
    csv_path, xlsx_path = _dated_output_paths(base_dir, today)
    score4plus_csv_path, score4plus_xlsx_path = _filtered_output_paths(base_dir, today)
    prev_csv = _find_previous_scan_csv(base_dir, today)

    print(f"[{dt.datetime.now().isoformat(timespec='seconds')}] Starting VCP scan...")
    print(f"Output CSV: {csv_path}")
    print(f"Output Excel: {xlsx_path}")
    print(f"Loaded blacklist symbols: {len(blacklisted_symbols)} from {blacklist_path}")
    print(f"Loaded focus symbols: {len(focus_symbols)} from {focus_path}")
    if prev_csv:
        print(f"Score delta base: {prev_csv}")
    else:
        print("Score delta base: none (first dated run)")

    symbols = scanner.fetch_symbols(timeout=timeout, source=symbol_source, sec_user_agent=sec_user_agent)
    if max_tickers > 0:
        symbols = symbols[:max_tickers]

    _, rows = scanner.run_scan(
        symbols=symbols,
        lookback_days=lookback_days,
        max_workers=workers,
        price_source=price_source,
        futu_host=futu_host,
        futu_port=futu_port,
        futu_fallback_yahoo=futu_fallback_yahoo,
    )

    prev_scores = scanner._load_previous_scores(str(prev_csv)) if prev_csv else None
    scanner.save_outputs(rows, str(csv_path), str(xlsx_path), prev_scores)
    score4plus_frame = _save_score4plus_outputs(
        csv_path,
        score4plus_csv_path,
        score4plus_xlsx_path,
        blacklisted_symbols,
    )

    full_scan_frame = pd.read_csv(csv_path)
    msg = _build_message(today, full_scan_frame, blacklisted_symbols, focus_symbols, base_dir)
    _post_webhook(msg)
    print(f"[{dt.datetime.now().isoformat(timespec='seconds')}] Sent HTML stock list message.")

    _post_file(score4plus_xlsx_path)
    print(f"[{dt.datetime.now().isoformat(timespec='seconds')}] Sent filtered XLSX file.")
    return csv_path


def _wait_until_next_run(hour: int, minute: int) -> int:
    now = dt.datetime.now()
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if now >= target:
        target = target + dt.timedelta(days=1)
    return int((target - now).total_seconds())


def run_scheduler(hour: int, minute: int, **job_kwargs) -> None:
    while True:
        sleep_seconds = _wait_until_next_run(hour, minute)
        run_at = dt.datetime.now() + dt.timedelta(seconds=sleep_seconds)
        print(
            f"[{dt.datetime.now().isoformat(timespec='seconds')}] "
            f"Next run at {run_at.isoformat(timespec='seconds')}"
        )
        time.sleep(sleep_seconds)
        try:
            run_vcp_job(**job_kwargs)
        except Exception as exc:
            print(f"[{dt.datetime.now().isoformat(timespec='seconds')}] Job failed: {exc}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run VCP scan once or on a daily schedule.")
    parser.add_argument("--mode", choices=["once", "schedule"], default="once")
    parser.add_argument("--schedule-hour", type=int, default=6)
    parser.add_argument("--schedule-minute", type=int, default=0)
    parser.add_argument("--base-dir", default=".")

    # Scan settings (mirrors main.py)
    parser.add_argument("--max-tickers", type=int, default=8000)
    parser.add_argument("--lookback-days", type=int, default=260)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--timeout", type=int, default=15)
    parser.add_argument("--symbol-source", choices=["auto", "nasdaq", "sec"], default="sec")
    parser.add_argument(
        "--sec-user-agent",
        default="Mozilla/5.0 (compatible; VCPDailyJob/1.0; +https://example.com/contact)",
    )
    parser.add_argument("--price-source", choices=["yahoo", "futu"], default="yahoo")
    parser.add_argument("--futu-host", default="127.0.0.1")
    parser.add_argument("--futu-port", type=int, default=11111)
    parser.add_argument("--futu-fallback-yahoo", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    job_kwargs = {
        "base_dir": Path(args.base_dir).resolve(),
        "max_tickers": args.max_tickers,
        "lookback_days": args.lookback_days,
        "workers": args.workers,
        "symbol_source": args.symbol_source,
        "price_source": args.price_source,
        "timeout": args.timeout,
        "sec_user_agent": args.sec_user_agent,
        "futu_host": args.futu_host,
        "futu_port": args.futu_port,
        "futu_fallback_yahoo": args.futu_fallback_yahoo,
    }

    if args.mode == "once":
        run_vcp_job(**job_kwargs)
        return

    run_scheduler(args.schedule_hour, args.schedule_minute, **job_kwargs)


if __name__ == "__main__":
    main(sys.argv[1:])
