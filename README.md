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
- `--metadata-scope {off,filtered,all}`: attach cached company metadata to scan exports. `filtered` enriches score>=4 rows and daily focus symbols.
- `--metadata-cache`: JSON cache file for company metadata. Default `stock_metadata_cache.json`.
- `--metadata-ttl-days`: refresh successful cache entries after this many days. Default 30.
- `--score-delta-from path/to/old_vcp_scan.csv`: adds `score_delta` vs a prior scan.

Scan exports can now include these metadata columns when available:
- `company_name`
- `sector`
- `industry`
- `business_summary`
- `metadata_status`
- `metadata_note`
- `metadata_fetched_at`

Scan exports also include EMA-based trend columns derived from the fetched daily close series:
- `ema_5`, `ema_10`, `ema_20`, `ema_60`, `ema_250`
- `price_above_ema_20`, `price_above_ema_60`, `price_above_ema_250`
- `ema_stack_bullish`
- `ema_signal_note`
- `trend_score`
## Post-filter enrichment (fundamentals + targets)

Provide a CSV with a `symbol`/`ticker` column to enrich only those names. Adds average volume, 60d return, RS vs SPY, distance to 52w high, market cap/sector/industry/country, beta, trailing/forward PE, short ratio, next earnings date, a composite buy score, breakout/target prices, and company metadata including name and business summary.

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
- EMA signals are exported as additional trend context and do not change the VCP score.
- `trend_score` is a separate EMA-alignment score used for ranking, not for redefining VCP.

## Notes

- Yahoo Finance may throttle; reduce `--workers` or split runs if rate-limited.
- Metadata is cached locally to avoid refetching company details on every run.
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

Compact Telegram mode with appended Fear & Greed text:

```bash
venv/bin/python vcp_daily_job.py \
	--mode schedule \
	--schedule-hour 6 \
	--schedule-minute 0 \
	--base-dir . \
	--metadata-scope filtered \
	--message-mode compact \
	--fear-greed-enable
```

Add forecast AI Overview messages after the filtered XLSX attachment:

```bash
venv/bin/python vcp_daily_job.py \
	--mode schedule \
	--schedule-hour 8 \
	--schedule-minute 15 \
	--base-dir . \
	--metadata-scope filtered \
	--message-mode compact \
	--fear-greed-enable \
	--financial-forecast-enable
```

Useful daily-job filters:
- `--require-price-above-ema20`: keep score>=4 filtered exports only when price is above EMA20.
- `--require-price-above-ema60`: keep score>=4 filtered exports only when price is above EMA60.
- `--require-price-above-ema250`: keep score>=4 filtered exports only when price is above EMA250.
- `--message-mode compact`: keep focus-symbol details, but reduce new/existing/dropped score sections to counts only.
- `--fear-greed-enable`: append a Fear & Greed text message after the VCP Telegram messages.
- `--fear-greed-charts {daily,weekly,both,auto}`: when Fear & Greed is enabled, generate and send PNG charts if available.
- `--financial-forecast-enable`: run `npx --prefix vendor/playwright-altered financial_sector_forecast_fetch --headed` and append one Telegram message per available AI Overview after the XLSX attachment.
- `--metadata-scope all`: populate metadata for all non-fetch-error rows in the dated scan export. `filtered` only fills score>=4 rows plus focus symbols, so most rows in the full dated Excel will remain blank by design.

What it does each run:
- Writes dated outputs: `YYYYMMDD_vcp_scan.csv` and `YYYYMMDD_vcp_scan.xlsx`.
- Finds the most recent prior dated scan CSV and computes `score_delta` from it.
- Filters symbols where `score >= 4`.
- Orders filtered score>=4 exports by VCP score first and `trend_score` second.
- Shows compact EMA regime fields for score>=4 rows in the Telegram message.
- Supports a compact Telegram mode that sends counts instead of full new/existing/dropped score tables.
- Can append a separate Fear & Greed text summary and optional chart images to the same Telegram push sequence.
- Can append forecast AI Overview text messages from the local Playwright-based CLI after the filtered XLSX file.
- Reuses cached metadata in exports and message rendering to reduce extra Yahoo requests.
- Sends the result to `https://tgbot.www.vanportdev.com/msg/1348940059` with JSON body `{ "msg": "..." }`.
