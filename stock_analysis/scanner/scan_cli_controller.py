import argparse
import datetime as dt
from collections.abc import Sequence

from .market_data_fetchers import fetch_us_equity_symbols
from .scan_models import DEFAULT_METADATA_CACHE, DEFAULT_METADATA_SCOPE, DEFAULT_METADATA_TTL_DAYS
from .scan_result_writer import (
    build_enriched_symbol_analysis_list,
    load_previous_scan_score_map,
    load_symbol_list_from_csv_file,
    save_enriched_analysis_outputs,
    save_scan_output_files,
)
from .vcp_pattern_analyzer import run_vcp_scan_for_symbol_list


def parse_scan_cli_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
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


def run_scan_cli_controller(argv: Sequence[str] | None = None) -> None:
    args = parse_scan_cli_arguments(argv)
    if args.post_filter:
        symbols = load_symbol_list_from_csv_file(args.post_filter)
        print(f"Enriching {len(symbols)} symbols from {args.post_filter}")
        rows = build_enriched_symbol_analysis_list(
            symbols,
            rs_lookback=args.rs_lookback,
            metadata_cache_path=args.metadata_cache,
            metadata_ttl_days=args.metadata_ttl_days,
        )
        save_enriched_analysis_outputs(rows, args.enrich_csv, args.enrich_excel)
        return

    print("Fetching symbol universe...")
    symbols = fetch_us_equity_symbols(timeout=args.timeout, source=args.symbol_source, sec_user_agent=args.sec_user_agent)
    if args.max_tickers > 0:
        symbols = symbols[: args.max_tickers]
    print(f"Analyzing {len(symbols)} symbols with lookback {args.lookback_days} days using {args.workers} workers")

    start = dt.datetime.now()
    candidates, rows = run_vcp_scan_for_symbol_list(
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

    prev_scores = load_previous_scan_score_map(args.score_delta_from) if args.score_delta_from else None
    save_scan_output_files(
        rows,
        args.csv_path,
        args.excel_path,
        prev_scores,
        metadata_scope=args.metadata_scope,
        metadata_cache_path=args.metadata_cache,
        metadata_ttl_days=args.metadata_ttl_days,
    )


main = run_scan_cli_controller
parse_args = parse_scan_cli_arguments
