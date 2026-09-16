import datetime as dt
import time
from pathlib import Path

from stock_analysis.scanner import (
    fetch_us_equity_symbols,
    load_previous_scan_score_map,
    load_stock_metadata_by_symbol,
    run_vcp_scan_for_symbol_list,
    save_scan_output_files,
)

from .daily_job_constants import BLACKLIST_FILE_NAME, FOCUS_FILE_NAME
from .daily_watchlist_message_formatter import build_daily_watchlist_html_message
from .scan_history_file_helpers import (
    build_dated_scan_output_paths,
    build_filtered_scan_output_paths,
    find_previous_dated_scan_csv_path,
    load_blacklisted_symbol_set,
    load_focus_symbol_set,
    save_score4plus_filtered_outputs,
)
from .supplemental_notification_service import (
    prepare_fear_greed_notification_message,
    prepare_financial_forecast_notification_messages,
)
from .webhook_delivery_client import send_webhook_file_attachment, send_webhook_html_message_batch


def run_daily_vcp_scan_job(
    base_dir: Path,
    history_dir: Path,
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
    metadata_scope: str,
    metadata_cache_path: Path,
    metadata_ttl_days: int,
    require_price_above_ema20: bool,
    require_price_above_ema60: bool,
    require_price_above_ema250: bool,
    message_mode: str,
    fear_greed_enabled: bool,
    fear_greed_charts: str,
    fear_greed_timeout: int,
    fear_greed_attempts: int,
    financial_forecast_enabled: bool,
    financial_forecast_timeout: int,
    financial_forecast_cli_timeout_ms: int,
) -> Path:
    today = dt.date.today()
    blacklist_path = base_dir / BLACKLIST_FILE_NAME
    focus_path = base_dir / FOCUS_FILE_NAME
    blacklisted_symbols = load_blacklisted_symbol_set(blacklist_path)
    focus_symbols = load_focus_symbol_set(focus_path)
    csv_path, xlsx_path = build_dated_scan_output_paths(history_dir, today)
    score4plus_csv_path, score4plus_xlsx_path = build_filtered_scan_output_paths(history_dir, today)
    prev_csv = find_previous_dated_scan_csv_path(history_dir, today)

    print(f"[{dt.datetime.now().isoformat(timespec='seconds')}] Starting VCP scan...")
    print(f"Output CSV: {csv_path}")
    print(f"Output Excel: {xlsx_path}")
    print(f"Loaded blacklist symbols: {len(blacklisted_symbols)} from {blacklist_path}")
    print(f"Loaded focus symbols: {len(focus_symbols)} from {focus_path}")
    if prev_csv:
        print(f"Score delta base: {prev_csv}")
    else:
        print("Score delta base: none (first dated run)")

    symbols = fetch_us_equity_symbols(timeout=timeout, source=symbol_source, sec_user_agent=sec_user_agent)
    if max_tickers > 0:
        symbols = symbols[:max_tickers]

    _, rows = run_vcp_scan_for_symbol_list(
        symbols=symbols,
        lookback_days=lookback_days,
        max_workers=workers,
        price_source=price_source,
        futu_host=futu_host,
        futu_port=futu_port,
        futu_fallback_yahoo=futu_fallback_yahoo,
    )

    prev_scores = load_previous_scan_score_map(str(prev_csv)) if prev_csv else None
    full_scan_frame = save_scan_output_files(
        rows,
        str(csv_path),
        str(xlsx_path),
        prev_scores,
        metadata_scope=metadata_scope,
        metadata_cache_path=str(metadata_cache_path),
        metadata_ttl_days=metadata_ttl_days,
        always_include_symbols=sorted(focus_symbols),
    )
    score4plus_frame = save_score4plus_filtered_outputs(
        full_scan_frame,
        score4plus_csv_path,
        score4plus_xlsx_path,
        blacklisted_symbols,
        require_price_above_ema20,
        require_price_above_ema60,
        require_price_above_ema250,
    )

    if full_scan_frame.empty or "symbol" not in full_scan_frame.columns:
        missing_focus = sorted(focus_symbols)
    else:
        available_symbols = set(full_scan_frame["symbol"].astype(str).str.upper())
        missing_focus = sorted(focus_symbols - available_symbols)
    fallback_metadata = load_stock_metadata_by_symbol(missing_focus, str(metadata_cache_path), metadata_ttl_days) if missing_focus else {}
    eligible_score_symbols = (
        set(score4plus_frame["symbol"].astype(str).str.upper())
        if not score4plus_frame.empty and "symbol" in score4plus_frame.columns
        else set()
    )

    outbound_messages = [
        f"<b>Here comes {today.isoformat()} financial analysis</b>",
        build_daily_watchlist_html_message(
            today,
            full_scan_frame,
            blacklisted_symbols,
            focus_symbols,
            history_dir,
            fallback_metadata,
            eligible_score_symbols,
            message_mode=message_mode,
        ),
    ]

    fear_greed_artifacts: list[Path] = []
    if fear_greed_enabled:
        fear_greed_message, fear_greed_artifacts = prepare_fear_greed_notification_message(
            base_dir=base_dir,
            charts_mode=fear_greed_charts,
            timeout_seconds=fear_greed_timeout,
            attempts=fear_greed_attempts,
        )
        outbound_messages.append(fear_greed_message)

    financial_forecast_messages: list[str] = []
    if financial_forecast_enabled:
        financial_forecast_messages = prepare_financial_forecast_notification_messages(
            today=today,
            base_dir=base_dir,
            timeout_seconds=financial_forecast_timeout,
            cli_timeout_ms=financial_forecast_cli_timeout_ms,
        )

    send_webhook_html_message_batch(outbound_messages)
    print(f"[{dt.datetime.now().isoformat(timespec='seconds')}] Sent HTML stock list message.")

    for artifact_path in fear_greed_artifacts:
        send_webhook_file_attachment(artifact_path)
        print(f"[{dt.datetime.now().isoformat(timespec='seconds')}] Sent Fear & Greed artifact: {artifact_path.name}")

    send_webhook_file_attachment(score4plus_xlsx_path)
    print(f"[{dt.datetime.now().isoformat(timespec='seconds')}] Sent filtered XLSX file.")
    if financial_forecast_messages:
        send_webhook_html_message_batch(financial_forecast_messages, split_messages=False)
        print(f"[{dt.datetime.now().isoformat(timespec='seconds')}] Sent financial forecast messages.")
    send_webhook_html_message_batch([f"<b>That's the end of {today.isoformat()} financial analysis. Thank you.</b>"], split_messages=False)
    print(f"[{dt.datetime.now().isoformat(timespec='seconds')}] Sent analysis ending message.")
    return csv_path


def calculate_seconds_until_next_scheduled_run(hour: int, minute: int) -> int:
    now = dt.datetime.now()
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if now >= target:
        target = target + dt.timedelta(days=1)
    return int((target - now).total_seconds())


def run_scheduled_daily_vcp_scan_job(hour: int, minute: int, **job_kwargs) -> None:
    while True:
        sleep_seconds = calculate_seconds_until_next_scheduled_run(hour, minute)
        run_at = dt.datetime.now() + dt.timedelta(seconds=sleep_seconds)
        print(f"[{dt.datetime.now().isoformat(timespec='seconds')}] Next run at {run_at.isoformat(timespec='seconds')}")
        time.sleep(sleep_seconds)
        try:
            run_daily_vcp_scan_job(**job_kwargs)
        except Exception as exc:
            print(f"[{dt.datetime.now().isoformat(timespec='seconds')}] Job failed: {exc}")


run_scheduler = run_scheduled_daily_vcp_scan_job
run_vcp_job = run_daily_vcp_scan_job
wait_until_next_run = calculate_seconds_until_next_scheduled_run
