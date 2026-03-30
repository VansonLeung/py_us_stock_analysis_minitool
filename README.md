# US Stock VCP Scanner

Python tool to scan US-listed equities for volatility contraction pattern (VCP) traits, export results to CSV/Excel, enrich filtered symbols with fundamentals/technicals, and compare score deltas against a prior run.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Core scan (VCP detection)

```bash
venv/bin/python main.py \
	--max-tickers 1200 \
	--lookback-days 260 \
	--workers 12 \
	--symbol-source sec \
	--price-source yahoo \
	--csv vcp_scan.csv \
	--excel vcp_scan.xlsx
```

```bash
venv/bin/python3 main.py \
--max-tickers 12000 \
--lookback-days 260 \
--workers 12 \
--csv 20260127_vcp_scan.csv \
--excel 20260127_vcp_scan.xlsx \
--symbol-source sec \
--price-source yahoo \
--score-delta-from vcp_scan.csv
```

Key flags (scan):
- `--symbol-source {auto,nasdaq,sec}`: universe source (auto falls back to SEC/GitHub).
- `--price-source {yahoo,futu}`: OHLCV source; for Futu, set `--futu-host`/`--futu-port`, optional `--futu-fallback-yahoo`.
- `--max-tickers`: cap symbols (0 = all). Default 1000.
- `--lookback-days`: daily bars requested. Default 260.
- `--workers`: thread pool size. Default 8.
- `--csv` / `--excel`: output paths (Excel needs `openpyxl`).
- `--score-delta-from path/to/old_vcp_scan.csv`: adds `score_delta` vs a prior scan.

## Post-filter enrichment (fundamentals + targets)

Provide a CSV with a `symbol`/`ticker` column to enrich only those names. Adds average volume, 60d return, RS vs SPY, distance to 52w high, market cap/sector/industry/country, beta, trailing/forward PE, short ratio, next earnings date, a composite buy score, and breakout/target prices.

```bash
venv/bin/python main.py \
	--post-filter vcp_scan_filtered/sheet.csv \
	--enrich-csv vcp_enriched.csv \
	--enrich-excel vcp_enriched.xlsx \
	--rs-lookback 60
```

## How it scores VCP

- Looks for three shrinking pullbacks between local highs/lows with contracting volume and price near the latest pivot high.
- Assigns a simple VCP score; outputs all symbols with status `vcp`, `no_pattern`, or `fetch_error`.
- Optional `score_delta` shows change vs a provided prior scan file.

## Notes

- Yahoo Finance may throttle; reduce `--workers` or split runs if rate-limited.
- Futu requires a running OpenD instance; use `--futu-fallback-yahoo` to fall back per-symbol.
- Outputs are heuristic, not trading advice—validate before use.

## Daily 6:00 job (dated files + webhook)

Use `vcp_daily_job.py` to run one-off or keep a process running that executes daily at 6:00.

One-time run:

```bash
venv/bin/python vcp_daily_job.py --mode once --base-dir .
```

Scheduled run (every day at 6:00):

```bash
venv/bin/python vcp_daily_job.py --mode schedule --schedule-hour 6 --schedule-minute 0 --base-dir .
```

What it does each run:
- Writes dated outputs: `YYYYMMDD_vcp_scan.csv` and `YYYYMMDD_vcp_scan.xlsx`.
- Finds the most recent prior dated scan CSV and computes `score_delta` from it.
- Filters symbols where `score == 4`.
- Sends the result to `https://tgbot.www.vanportdev.com/msg/1348940059` with JSON body `{ "msg": "..." }`.
