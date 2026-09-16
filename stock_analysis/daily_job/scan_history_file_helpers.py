import datetime as dt
from pathlib import Path

import pandas as pd


def build_dated_scan_output_paths(history_dir: Path, day: dt.date) -> tuple[Path, Path]:
    date_key = day.strftime("%Y%m%d")
    return history_dir / f"{date_key}_vcp_scan.csv", history_dir / f"{date_key}_vcp_scan.xlsx"


def build_filtered_scan_output_paths(history_dir: Path, day: dt.date) -> tuple[Path, Path]:
    date_key = day.strftime("%Y%m%d")
    return (
        history_dir / f"{date_key}_vcp_scan_score4plus_filtered.csv",
        history_dir / f"{date_key}_vcp_scan_score4plus_filtered.xlsx",
    )


def parse_scan_date_from_dated_csv_filename(path: Path) -> dt.date | None:
    if not path.name.endswith("_vcp_scan.csv"):
        return None
    prefix = path.name.split("_", 1)[0]
    if len(prefix) != 8 or not prefix.isdigit():
        return None
    try:
        return dt.datetime.strptime(prefix, "%Y%m%d").date()
    except ValueError:
        return None


def find_previous_dated_scan_csv_path(history_dir: Path, today: dt.date) -> Path | None:
    dated_files: list[tuple[dt.date, Path]] = []
    for path in history_dir.glob("*_vcp_scan.csv"):
        day = parse_scan_date_from_dated_csv_filename(path)
        if day is None or day >= today:
            continue
        dated_files.append((day, path))
    if not dated_files:
        return None
    dated_files.sort(key=lambda item: item[0], reverse=True)
    return dated_files[0][1]


def list_available_dated_scan_csv_paths(history_dir: Path, today: dt.date) -> list[tuple[dt.date, Path]]:
    dated_files: list[tuple[dt.date, Path]] = []
    for path in history_dir.glob("*_vcp_scan.csv"):
        day = parse_scan_date_from_dated_csv_filename(path)
        if day is None or day > today:
            continue
        dated_files.append((day, path))
    dated_files.sort(key=lambda item: item[0])
    return dated_files


def calculate_consecutive_score4plus_streaks(history_dir: Path, today: dt.date) -> dict[str, int]:
    days_by_symbol: dict[str, int] = {}
    dated_files = list_available_dated_scan_csv_paths(history_dir, today)
    if not dated_files:
        return days_by_symbol

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

        if not days_by_symbol:
            for symbol in today_symbols:
                days_by_symbol[symbol] = 1
            continue

        active = set(days_by_symbol.keys())
        for symbol in list(active):
            if symbol in today_symbols:
                days_by_symbol[symbol] += 1
            else:
                del days_by_symbol[symbol]
        for symbol in (today_symbols - active):
            days_by_symbol[symbol] = 1

        if not days_by_symbol:
            break

    return days_by_symbol


def load_symbol_set_from_text_file(path: Path, label: str) -> set[str]:
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


def load_blacklisted_symbol_set(blacklist_path: Path) -> set[str]:
    return load_symbol_set_from_text_file(blacklist_path, "Blacklist")


def load_focus_symbol_set(focus_path: Path) -> set[str]:
    return load_symbol_set_from_text_file(focus_path, "Focus")


def save_score4plus_filtered_outputs(
    frame: pd.DataFrame,
    out_csv: Path,
    out_xlsx: Path,
    blacklisted_symbols: set[str],
    require_price_above_ema20: bool,
    require_price_above_ema60: bool,
    require_price_above_ema250: bool,
) -> pd.DataFrame:
    filtered = frame.iloc[0:0].copy() if "score" not in frame.columns else frame[frame["score"] >= 4].copy()

    if blacklisted_symbols and "symbol" in filtered.columns:
        filtered = filtered[~filtered["symbol"].astype(str).str.upper().isin(blacklisted_symbols)].copy()
    if require_price_above_ema20 and "price_above_ema_20" in filtered.columns:
        filtered = filtered[filtered["price_above_ema_20"] == True].copy()
    if require_price_above_ema60 and "price_above_ema_60" in filtered.columns:
        filtered = filtered[filtered["price_above_ema_60"] == True].copy()
    if require_price_above_ema250 and "price_above_ema_250" in filtered.columns:
        filtered = filtered[filtered["price_above_ema_250"] == True].copy()

    if not filtered.empty:
        sort_cols = ["score", "symbol"]
        ascending = [False, True]
        if "trend_score" in filtered.columns:
            sort_cols = ["score", "trend_score", "symbol"]
            ascending = [False, False, True]
        filtered = filtered.sort_values(by=sort_cols, ascending=ascending)
    filtered.to_csv(out_csv, index=False)
    filtered.to_excel(out_xlsx, index=False)
    print(f"Saved score>=4 filtered CSV to {out_csv}")
    print(f"Saved score>=4 filtered Excel to {out_xlsx}")
    return filtered


consecutive_score4plus_days = calculate_consecutive_score4plus_streaks
dated_output_paths = build_dated_scan_output_paths
filtered_output_paths = build_filtered_scan_output_paths
find_previous_scan_csv = find_previous_dated_scan_csv_path
list_dated_scan_csvs = list_available_dated_scan_csv_paths
load_blacklisted_symbols = load_blacklisted_symbol_set
load_focus_symbols = load_focus_symbol_set
load_symbol_file = load_symbol_set_from_text_file
parse_dated_scan_filename = parse_scan_date_from_dated_csv_filename
save_score4plus_outputs = save_score4plus_filtered_outputs
