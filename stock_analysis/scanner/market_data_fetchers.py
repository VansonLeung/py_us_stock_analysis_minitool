import datetime as dt
import io

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import yfinance as yf


NASDAQ_URL = "https://ftp.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt"
OTHER_URL = "https://ftp.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers_exchange.json"
SEC_TICKER_TXT_URL = "https://www.sec.gov/include/ticker.txt"
GITHUB_TICKERS_URL = "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/master/all/all_tickers.txt"


def _build_retry_http_session(timeout: int = 15) -> requests.Session:
    retry = Retry(
        total=3,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.request_timeout = timeout  # type: ignore[attr-defined]
    return session


def _fetch_sec_exchange_symbols(session: requests.Session, timeout: int, user_agent: str) -> list[str]:
    headers = {"User-Agent": user_agent}
    try:
        resp = session.get(SEC_TICKERS_URL, timeout=timeout, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        frame = pd.DataFrame.from_dict(data, orient="index")
        frame.columns = [str(column).lower() for column in frame.columns]
        allowed_exchanges = {"NASDAQ", "NYSE", "NYSE MKT", "NYSE ARCA", "BATS"}
        if "exchange" in frame.columns:
            frame["exchange"] = frame["exchange"].astype(str).str.upper()
            frame = frame[frame["exchange"].isin(allowed_exchanges)]
        else:
            print("SEC list missing exchange field; using all tickers.")

        ticker_column = "ticker" if "ticker" in frame.columns else ("symbol" if "symbol" in frame.columns else None)
        if ticker_column is None:
            raise RuntimeError("SEC ticker payload missing ticker column")

        tickers = frame[ticker_column].dropna().astype(str).str.upper()
        tickers = tickers[tickers.str.isalpha()]
        result = sorted(set(tickers))
        if result:
            return result
    except Exception as exc:
        print(f"SEC JSON endpoint failed ({exc}); trying ticker.txt fallback.")

    resp_txt = session.get(SEC_TICKER_TXT_URL, timeout=timeout, headers=headers)
    if resp_txt.status_code == 200:
        tickers: list[str] = []
        for line in resp_txt.text.strip().splitlines():
            if "|" not in line:
                continue
            ticker = line.split("|")[0].strip().upper()
            if ticker.isalpha():
                tickers.append(ticker)
        if tickers:
            return sorted(set(tickers))
        print("SEC ticker.txt returned no tickers; trying GitHub mirror.")
    else:
        print(f"SEC ticker.txt status {resp_txt.status_code}; trying GitHub mirror.")

    resp_git = session.get(GITHUB_TICKERS_URL, timeout=timeout, headers=headers)
    resp_git.raise_for_status()
    tickers_git = [line.strip().upper() for line in resp_git.text.splitlines() if line.strip()]
    tickers_git = [ticker for ticker in tickers_git if ticker.isalpha()]
    if not tickers_git:
        raise RuntimeError("GitHub ticker fallback returned no tickers")
    return sorted(set(tickers_git))


def fetch_us_equity_symbols(timeout: int = 15, source: str = "auto", sec_user_agent: str = "") -> list[str]:
    session = _build_retry_http_session(timeout=timeout)
    if source == "sec":
        return _fetch_sec_exchange_symbols(session, timeout, sec_user_agent)

    frames: list[pd.DataFrame] = []
    last_err: Exception | None = None
    if source in {"nasdaq", "auto"}:
        for url, symbol_column in [(NASDAQ_URL, "NASDAQ Symbol"), (OTHER_URL, "ACT Symbol")]:
            for candidate_url in (url, url.replace("https://", "http://")):
                try:
                    resp = session.get(candidate_url, timeout=timeout)
                    resp.raise_for_status()
                    last_err = None
                    break
                except Exception as exc:
                    last_err = exc
            if last_err:
                continue

            frame = pd.read_csv(io.StringIO(resp.text), sep="|")
            frame = frame[(frame["Test Issue"] == "N") & (frame[symbol_column].notna())]
            frame = frame[frame[symbol_column].str.isalpha()]
            frames.append(frame[[symbol_column]].rename(columns={symbol_column: "Symbol"}))

    if frames:
        merged = pd.concat(frames, ignore_index=True)
        return sorted(set(merged["Symbol"].str.upper()))

    if source == "nasdaq":
        raise RuntimeError(f"Failed to fetch symbols from Nasdaq sources ({last_err}).")

    try:
        print("Nasdaq sources unreachable; using SEC list fallback.")
        return _fetch_sec_exchange_symbols(session, timeout, sec_user_agent)
    except Exception as exc:
        raise RuntimeError("Failed to fetch symbols from Nasdaq and SEC.") from exc


def _normalize_ohlcv_price_frame(df: pd.DataFrame, symbol: str) -> pd.DataFrame | None:
    if df is None or df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        symbols = list(df.columns.get_level_values(-1).unique())
        selected = symbol if symbol in symbols else symbols[0]
        df = df.xs(selected, axis=1, level=-1)

    column_map = {column.lower(): column for column in df.columns}

    def pick(column_key: str) -> pd.Series | None:
        for normalized_key, original in column_map.items():
            if column_key != normalized_key:
                continue
            series = df[original]
            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0]
            return pd.to_numeric(series, errors="coerce")
        return None

    open_series = pick("open")
    high_series = pick("high")
    low_series = pick("low")
    close_series = pick("close")
    volume_series = pick("volume")
    if any(series is None for series in (open_series, high_series, low_series, close_series, volume_series)):
        return None

    out = pd.DataFrame(
        {
            "Open": open_series,
            "High": high_series,
            "Low": low_series,
            "Close": close_series,
            "Volume": volume_series,
        }
    ).dropna()
    out.attrs["symbol"] = symbol
    return out


def fetch_yahoo_symbol_history(symbol: str, lookback_days: int) -> pd.DataFrame | None:
    data = yf.download(
        symbol,
        period=f"{lookback_days}d",
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    return _normalize_ohlcv_price_frame(data, symbol)


def fetch_yahoo_symbol_history_for_period(symbol: str, period: str) -> pd.DataFrame | None:
    data = yf.download(
        symbol,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    return _normalize_ohlcv_price_frame(data, symbol)


def fetch_futu_symbol_history(symbol: str, lookback_days: int, host: str, port: int) -> pd.DataFrame | None:
    try:
        from futu import OpenQuoteContext, RET_OK  # type: ignore
    except ImportError:
        return None

    end = dt.datetime.utcnow()
    start = end - dt.timedelta(days=lookback_days + 5)
    code = f"US.{symbol}"
    context = OpenQuoteContext(host=host, port=port)
    try:
        ret, data, _ = context.request_history_kline(
            code,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            ktype="K_DAY",
            autype="qfq",
        )
        if ret != RET_OK or data is None or data.empty:
            return None
        data = data.rename(
            columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            }
        )
        data = data[["Open", "High", "Low", "Close", "Volume"]].dropna()
        data.attrs["symbol"] = symbol
        return data
    finally:
        context.close()


def fetch_symbol_price_history(
    symbol: str,
    lookback_days: int,
    price_source: str,
    futu_host: str,
    futu_port: int,
    futu_fallback_yahoo: bool,
) -> pd.DataFrame | None:
    if price_source == "futu":
        data = fetch_futu_symbol_history(symbol, lookback_days, futu_host, futu_port)
        if data is not None:
            return data
        if not futu_fallback_yahoo:
            return None
    return fetch_yahoo_symbol_history(symbol, lookback_days)


fetch_history = fetch_symbol_price_history
fetch_history_futu = fetch_futu_symbol_history
fetch_history_yahoo = fetch_yahoo_symbol_history
fetch_history_yahoo_period = fetch_yahoo_symbol_history_for_period
fetch_symbols = fetch_us_equity_symbols
