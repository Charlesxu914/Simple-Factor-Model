"""
Ticker universe selection for the equity factor model.

Supports multiple universes:
  - sp500:       S&P 500 constituents (scraped from Wikipedia)
  - russell3000: Russell 3000 proxy via iShares IWV ETF holdings
"""

import io
import logging
import urllib.request

import pandas as pd

logger = logging.getLogger("ticker_universe")

UNIVERSE_SP500 = "sp500"
UNIVERSE_RUSSELL3000 = "russell3000"
SUPPORTED_UNIVERSES = (UNIVERSE_SP500, UNIVERSE_RUSSELL3000)

# ---------------------------------------------------------------------------
# Fallback ticker lists (used when web scraping fails)
# ---------------------------------------------------------------------------
_SP500_FALLBACK = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "BRK-B", "UNH",
    "XOM", "JPM", "JNJ", "V", "PG", "MA", "HD", "CVX", "MRK", "LLY",
    "AVGO", "KO",
]

_RUSSELL3000_FALLBACK_URL = (
    "https://stockanalysis.com/list/russell-3000-stocks/"
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def get_tickers(universe: str = UNIVERSE_SP500) -> list[str]:
    """Return a list of ticker symbols for the requested universe.

    Parameters
    ----------
    universe : str
        One of ``"sp500"`` or ``"russell3000"``.

    Returns
    -------
    list[str]
        Ticker symbols with dots replaced by dashes (yfinance convention).
    """
    if universe == UNIVERSE_SP500:
        return _get_sp500_tickers()
    elif universe == UNIVERSE_RUSSELL3000:
        return _get_russell3000_tickers()
    else:
        raise ValueError(
            f"Unknown universe: {universe!r}. "
            f"Supported: {SUPPORTED_UNIVERSES}"
        )


# ---------------------------------------------------------------------------
# S&P 500 — scrape Wikipedia
# ---------------------------------------------------------------------------
def _get_sp500_tickers() -> list[str]:
    """Fetch current S&P 500 constituents from Wikipedia."""
    try:
        logger.info("Fetching S&P 500 tickers from Wikipedia")
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        html = urllib.request.urlopen(req).read().decode("utf-8")
        tables = pd.read_html(io.StringIO(html))
        tickers = tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
        logger.info(f"Found {len(tickers)} S&P 500 constituents")
        return tickers
    except Exception as e:
        logger.error(f"Error fetching S&P 500 tickers: {e}")
        logger.warning(f"Using fallback list of {len(_SP500_FALLBACK)} tickers")
        return list(_SP500_FALLBACK)


# ---------------------------------------------------------------------------
# Russell 3000 — iShares IWV ETF holdings CSV
# ---------------------------------------------------------------------------
_IWV_CSV_URL = (
    "https://www.ishares.com/us/products/239714/"
    "ishares-russell-3000-etf/1467271812596.ajax"
    "?fileType=csv&fileName=IWV_holdings&dataType=fund"
)


def _parse_ishares_csv(raw_text: str) -> list[str]:
    """Parse iShares holdings CSV and return cleaned ticker list."""
    # iShares CSVs have ~9 header rows of metadata before the actual table.
    # Find the row that starts with "Ticker" or "Name" to locate the header.
    lines = raw_text.splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith("Ticker,") or line.startswith('"Ticker"'):
            header_idx = i
            break

    if header_idx is None:
        # Try to find a line containing "Ticker" anywhere
        for i, line in enumerate(lines):
            if "Ticker" in line and "Name" in line:
                header_idx = i
                break

    if header_idx is None:
        raise ValueError("Could not find header row in iShares CSV")

    csv_text = "\n".join(lines[header_idx:])
    df = pd.read_csv(io.StringIO(csv_text))

    # Normalise column names
    df.columns = df.columns.str.strip()

    ticker_col = None
    for candidate in ("Ticker", "ticker", "Symbol"):
        if candidate in df.columns:
            ticker_col = candidate
            break
    if ticker_col is None:
        raise ValueError(f"No ticker column found. Columns: {list(df.columns)}")

    # Also grab the Asset Class column if it exists to filter equities
    asset_col = None
    for candidate in ("Asset Class", "asset_class", "assetClass"):
        if candidate in df.columns:
            asset_col = candidate
            break

    if asset_col:
        df = df[df[asset_col].astype(str).str.strip().str.lower() == "equity"]

    tickers = df[ticker_col].dropna().astype(str).str.strip().tolist()

    # Filter out non-equity entries (cash, index futures, etc.)
    # Valid equity tickers: 1-5 alpha chars, possibly with a dash (e.g. BRK-B)
    import re
    valid_ticker = re.compile(r"^[A-Z]{1,5}(-[A-Z]{1,2})?$")
    cleaned = []
    for t in tickers:
        t = t.replace(".", "-")
        if valid_ticker.match(t):
            cleaned.append(t)

    return cleaned


def _get_russell3000_from_iwv() -> list[str]:
    """Download iShares IWV ETF holdings CSV and extract tickers."""
    logger.info("Fetching Russell 3000 tickers from iShares IWV ETF holdings")
    req = urllib.request.Request(_IWV_CSV_URL, headers={"User-Agent": "Mozilla/5.0"})
    resp = urllib.request.urlopen(req, timeout=30)
    raw = resp.read().decode("utf-8", errors="replace")
    tickers = _parse_ishares_csv(raw)
    logger.info(f"Parsed {len(tickers)} equity tickers from IWV holdings CSV")
    return tickers


def _get_russell3000_from_stockanalysis() -> list[str]:
    """Fallback: scrape Russell 3000 tickers from stockanalysis.com."""
    logger.info("Fallback: fetching Russell 3000 tickers from stockanalysis.com")
    req = urllib.request.Request(
        _RUSSELL3000_FALLBACK_URL,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    html = urllib.request.urlopen(req, timeout=30).read().decode("utf-8")
    tables = pd.read_html(html)
    if not tables:
        raise ValueError("No tables found on stockanalysis.com Russell 3000 page")
    df = tables[0]
    # The first column is usually "Symbol" or "Ticker"
    col = df.columns[0]
    tickers = df[col].dropna().astype(str).str.replace(".", "-", regex=False).tolist()
    logger.info(f"Parsed {len(tickers)} tickers from stockanalysis.com")
    return tickers


def _get_russell3000_tickers() -> list[str]:
    """Fetch Russell 3000 tickers with fallback chain."""
    # Try iShares IWV first
    try:
        tickers = _get_russell3000_from_iwv()
        if len(tickers) >= 1000:
            return tickers
        logger.warning(f"IWV returned only {len(tickers)} tickers, trying fallback")
    except Exception as e:
        logger.warning(f"iShares IWV fetch failed: {e}")

    # Fallback: stockanalysis.com
    try:
        tickers = _get_russell3000_from_stockanalysis()
        if len(tickers) >= 1000:
            return tickers
        logger.warning(f"stockanalysis.com returned only {len(tickers)} tickers")
    except Exception as e:
        logger.warning(f"stockanalysis.com fetch failed: {e}")

    # Last resort: use SP500 + warn loudly
    logger.error(
        "All Russell 3000 sources failed! Falling back to S&P 500 tickers. "
        "The analysis will NOT cover the full Russell 3000."
    )
    return _get_sp500_tickers()
