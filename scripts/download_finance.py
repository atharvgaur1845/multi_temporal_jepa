"""Download a daily S&P-500 cross-sectional panel from Yahoo Finance -> data_root/FINANCE/panel.npz.

The financial analogue of scripts/download_pastis.sh. PASTIS is a *spatial* cross-section (a grid
of pixels) observed over time; here the cross-section is a basket of **sector ETFs** (the nine
original Select-Sector SPDRs, all trading since Dec-1998) observed over the same trading days. That
gives a genuine cross-section to mask spatially (Spatial JEPA) AND a time axis to predict forward
(Temporal JEPA) — the same two-axis structure the satellite pipeline exploits, so the
temporal-vs-spatial comparison transfers unchanged.

We also pull the index (^GSPC) and volatility index (^VIX) purely to DEFINE the downstream task
labels (regime / volatility / anomaly / forecasting); the encoder only ever sees the sector panel.

Yahoo notes (why this isn't a one-liner):
  * the v8 chart endpoint 429s aggressive callers, so we warm a cookie + crumb once and then space
    requests out with retry/backoff;
  * series are aligned to the set of trading days for which EVERY symbol has a real close (inner
    join) so the panel is a dense (T, N) matrix with no NaNs.

If the network is blocked, this script exits non-zero and prints a hint; the dataset
(data/finance_dataset.py) then falls back to a documented synthetic regime-switching generator so
the rest of the pipeline (train + eval + tests) is runnable and reproducible offline.

Usage:
    python scripts/download_finance.py                       # full history -> data_root/FINANCE/panel.npz
    python scripts/download_finance.py --start 2005-01-01    # restrict start date
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import http.cookiejar
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np

# Cross-section = the 9 original Select-Sector SPDR ETFs (each ~1/11th of the S&P by GICS sector,
# all listed Dec-1998 -> long, clean, survivorship-bias-free history). These are the "assets" /
# tokens the encoder sees. Order is fixed and saved with the panel.
SECTOR_ETFS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
# Index + vol index used ONLY to build downstream labels (never fed to the encoder).
LABEL_SYMBOLS = ["^GSPC", "^VIX"]

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")


def _opener():
    cj = http.cookiejar.CookieJar()
    op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
    op.addheaders = [("User-Agent", UA), ("Accept", "*/*"), ("Accept-Language", "en-US,en;q=0.9")]
    return op


def _get(op, url, timeout=30):
    with op.open(url, timeout=timeout) as r:
        raw = r.read()
        if r.headers.get("Content-Encoding") == "gzip":
            raw = gzip.decompress(raw)
        return r.status, raw.decode("utf-8", "replace")


def _warmup(op):
    """Establish the Yahoo cookie + crumb (best-effort; chart API often works without it but the
    warmup markedly reduces 429s)."""
    try:
        _get(op, "https://fc.yahoo.com")
    except Exception:
        pass
    time.sleep(1.0)
    try:
        _, crumb = _get(op, "https://query1.finance.yahoo.com/v1/test/getcrumb")
        return crumb.strip()
    except Exception:
        return ""


def _fetch_symbol(op, symbol, period1, period2, retries=5):
    """Return (dates_yyyymmdd: np.int64[T], adjclose: float[T], volume: float[T]) for one symbol.

    Uses adjusted close (splits + dividends) so multi-year returns are correct. 429 -> exp backoff.
    """
    enc = urllib.parse.quote(symbol, safe="")
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{enc}"
           f"?period1={period1}&period2={period2}&interval=1d&events=div%2Csplit")
    last_err = None
    for attempt in range(retries):
        try:
            st, body = _get(op, url)
            j = json.loads(body)
            res = j["chart"]["result"][0]
            ts = res["timestamp"]
            quote = res["indicators"]["quote"][0]
            close = quote["close"]
            vol = quote["volume"]
            adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose", close)
            dates, ac, vv = [], [], []
            for i, t in enumerate(ts):
                c = adj[i] if adj[i] is not None else close[i]
                if c is None:
                    continue                      # skip holidays / missing prints
                d = dt.datetime.fromtimestamp(t, dt.timezone.utc).date()
                dates.append(d.year * 10000 + d.month * 100 + d.day)
                ac.append(float(c))
                vv.append(float(vol[i]) if vol[i] is not None else 0.0)
            return np.array(dates, np.int64), np.array(ac, np.float64), np.array(vv, np.float64)
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code == 429:
                wait = 3.0 * (2 ** attempt)
                print(f"  {symbol}: 429, backoff {wait:.0f}s (attempt {attempt+1}/{retries})")
                time.sleep(wait)
            else:
                raise
        except Exception as e:
            last_err = e
            time.sleep(2.0)
    raise RuntimeError(f"failed to fetch {symbol}: {last_err}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("FINANCE_ROOT", "data_root/FINANCE"))
    ap.add_argument("--start", default="1999-01-01", help="earliest date (YYYY-MM-DD)")
    ap.add_argument("--end", default=None, help="latest date (YYYY-MM-DD); default today")
    ap.add_argument("--sleep", type=float, default=2.0, help="seconds between symbol requests")
    args = ap.parse_args()

    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end) if args.end else dt.date.today()
    period1 = int(dt.datetime(start.year, start.month, start.day).timestamp())
    period2 = int(dt.datetime(end.year, end.month, end.day).timestamp()) + 86400

    symbols = SECTOR_ETFS + LABEL_SYMBOLS
    op = _opener()
    print("[download_finance] warming up cookie/crumb ...")
    _warmup(op)

    series = {}
    for sym in symbols:
        print(f"[download_finance] fetching {sym} ...")
        d, c, v = _fetch_symbol(op, sym, period1, period2)
        series[sym] = (d, c, v)
        print(f"  {sym}: {len(d)} rows  {d[0]}..{d[-1]}")
        time.sleep(args.sleep)

    # Inner-join on dates present for EVERY symbol -> dense (T, N) panel (no NaNs).
    common = None
    for sym in symbols:
        s = set(series[sym][0].tolist())
        common = s if common is None else (common & s)
    dates = np.array(sorted(common), np.int64)
    print(f"[download_finance] {len(dates)} common trading days across {len(symbols)} symbols")

    def aligned(sym):
        d, c, v = series[sym]
        idx = {int(dd): i for i, dd in enumerate(d)}
        rows = [idx[int(dd)] for dd in dates]
        return c[rows], v[rows]

    close = np.stack([aligned(s)[0] for s in SECTOR_ETFS], axis=1)     # (T, N)
    volume = np.stack([aligned(s)[1] for s in SECTOR_ETFS], axis=1)    # (T, N)
    index_close = aligned("^GSPC")[0]                                  # (T,)
    vix = aligned("^VIX")[0]                                           # (T,)

    os.makedirs(args.root, exist_ok=True)
    out = os.path.join(args.root, "panel.npz")
    np.savez_compressed(
        out, symbols=np.array(SECTOR_ETFS), dates=dates,
        close=close.astype(np.float32), volume=volume.astype(np.float32),
        index_close=index_close.astype(np.float32), vix=vix.astype(np.float32),
        source=np.array("yahoo"),
    )
    print(f"[download_finance] saved {out}  close{close.shape}  "
          f"{dates[0]}..{dates[-1]}  (source=yahoo, real market data)")


if __name__ == "__main__":
    main()
