import argparse
from collections.abc import Sequence
from pathlib import Path

from stock_analysis.scanner import DEFAULT_METADATA_CACHE, DEFAULT_METADATA_SCOPE, DEFAULT_METADATA_TTL_DAYS

from .daily_job_orchestrator import run_daily_vcp_scan_job, run_scheduled_daily_vcp_scan_job


def parse_daily_job_cli_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VCP scan once or on a daily schedule.")
    parser.add_argument("--mode", choices=["once", "schedule"], default="schedule")
    parser.add_argument("--schedule-hour", type=int, default=6)
    parser.add_argument("--schedule-minute", type=int, default=0)
    parser.add_argument("--base-dir", default=".")
    parser.add_argument("--history-dir", default="./history_data")
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
    parser.add_argument(
        "--require-price-above-ema20",
        action="store_true",
        help="Keep score>=4 filtered exports only when price is above EMA20.",
    )
    parser.add_argument(
        "--require-price-above-ema60",
        action="store_true",
        help="Keep score>=4 filtered exports only when price is above EMA60.",
    )
    parser.add_argument(
        "--require-price-above-ema250",
        action="store_true",
        help="Keep score>=4 filtered exports only when price is above EMA250.",
    )
    parser.add_argument(
        "--message-mode",
        choices=["full", "compact"],
        default="full",
        help="Telegram message layout: full lists or compact counts for new/existing/dropped score groups.",
    )
    parser.add_argument(
        "--fear-greed-enable",
        action="store_true",
        help="Append a separate Fear & Greed text message after the VCP Telegram messages.",
    )
    parser.add_argument(
        "--fear-greed-charts",
        choices=["off", "daily", "weekly", "both", "auto"],
        default="off",
        help="When Fear & Greed is enabled, optionally generate and send chart images.",
    )
    parser.add_argument(
        "--fear-greed-timeout",
        type=int,
        default=15,
        help="Per-request timeout in seconds for Fear & Greed fetches.",
    )
    parser.add_argument(
        "--fear-greed-attempts",
        type=int,
        default=3,
        help="Number of fetch attempts for Fear & Greed data (1-5).",
    )
    parser.add_argument(
        "--financial-forecast-enable",
        action="store_true",
        help="Append forecast AI Overview messages after the filtered XLSX file.",
    )
    parser.add_argument(
        "--financial-forecast-timeout",
        type=int,
        default=380,
        help="Outer subprocess timeout in seconds for the forecast CLI.",
    )
    parser.add_argument(
        "--financial-forecast-cli-timeout-ms",
        type=int,
        default=360000,
        help="Overall timeout passed through to the forecast CLI in milliseconds.",
    )
    parser.add_argument(
        "--metadata-scope",
        choices=["off", "filtered", "all"],
        default=DEFAULT_METADATA_SCOPE,
        help="Metadata enrichment scope for dated scan exports: filtered means score>=4 rows plus focus symbols only; all means all non-fetch-error rows.",
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
    return parser.parse_args(argv)


def build_daily_job_runtime_kwargs(args: argparse.Namespace) -> dict:
    base_dir = Path(args.base_dir).resolve()
    metadata_cache_path = Path(args.metadata_cache)
    return {
        "base_dir": base_dir,
        "history_dir": Path(args.history_dir).resolve(),
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
        "require_price_above_ema20": args.require_price_above_ema20,
        "require_price_above_ema60": args.require_price_above_ema60,
        "require_price_above_ema250": args.require_price_above_ema250,
        "message_mode": args.message_mode,
        "fear_greed_enabled": args.fear_greed_enable,
        "fear_greed_charts": args.fear_greed_charts,
        "fear_greed_timeout": args.fear_greed_timeout,
        "fear_greed_attempts": args.fear_greed_attempts,
        "financial_forecast_enabled": args.financial_forecast_enable,
        "financial_forecast_timeout": args.financial_forecast_timeout,
        "financial_forecast_cli_timeout_ms": args.financial_forecast_cli_timeout_ms,
        "metadata_scope": args.metadata_scope,
        "metadata_cache_path": (base_dir / metadata_cache_path) if not metadata_cache_path.is_absolute() else metadata_cache_path,
        "metadata_ttl_days": args.metadata_ttl_days,
    }


def run_daily_job_cli_controller(argv: Sequence[str] | None = None) -> None:
    args = parse_daily_job_cli_arguments(argv)
    job_kwargs = build_daily_job_runtime_kwargs(args)
    if args.mode == "once":
        run_daily_vcp_scan_job(**job_kwargs)
        return
    run_scheduled_daily_vcp_scan_job(args.schedule_hour, args.schedule_minute, **job_kwargs)


_build_job_kwargs = build_daily_job_runtime_kwargs
main = run_daily_job_cli_controller
parse_args = parse_daily_job_cli_arguments
