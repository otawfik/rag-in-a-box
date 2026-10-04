"""SEC EDGAR 10-K downloader and cleaner.

Downloads the most recent 10-K filing for each given ticker and converts the
filing's HTML into clean plain text plus a metadata sidecar.

Usage:
    SEC_USER_AGENT="Your Name you@example.com" \\
        python -m ragbox.edgar --tickers AAPL MSFT GOOGL NVDA --out corpus/sec10k

The SEC requires a descriptive User-Agent identifying who you are; the script
refuses to run without ``SEC_USER_AGENT`` set. Requests are throttled to stay
well under the SEC's 10-requests/second fair-use guideline.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

SEC = "https://www.sec.gov"
DATA_SEC = "https://data.sec.gov"
REQUEST_DELAY = 0.35  # seconds between requests; SEC asks for <= 10 req/s

_TICKER_MAP: dict[str, str] | None = None


def _user_agent() -> str:
    ua = os.environ.get("SEC_USER_AGENT", "").strip()
    if not ua:
        sys.exit(
            "ERROR: set the SEC_USER_AGENT environment variable first, e.g.\n"
            '  export SEC_USER_AGENT="Your Name you@example.com"\n'
            "The SEC blocks anonymous/bot user agents; identify yourself."
        )
    return ua


def _get_json(url: str) -> dict:
    req = urllib.request.Request(
        url, headers={"User-Agent": _user_agent(), "Accept": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    time.sleep(REQUEST_DELAY)
    return data


def _get_bytes(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": _user_agent()})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = resp.read()
    time.sleep(REQUEST_DELAY)
    return data


def ticker_to_cik(ticker: str) -> str:
    """Map a stock ticker to its zero-padded 10-digit CIK."""
    global _TICKER_MAP
    if _TICKER_MAP is None:
        data = _get_json(f"{SEC}/files/company_tickers.json")
        _TICKER_MAP = {v["ticker"]: str(v["cik_str"]).zfill(10) for v in data.values()}
    cik = _TICKER_MAP.get(ticker.upper())
    if not cik:
        raise ValueError(f"Unknown ticker: {ticker}")
    return cik


def recent_10k(cik: str) -> dict:
    """Return accession, primary document, filing date and period for the
    most recent 10-K in the company's submissions history."""
    subs = _get_json(f"{DATA_SEC}/submissions/CIK{cik}.json")
    recent = subs["filings"]["recent"]
    for i, form in enumerate(recent["form"]):
        if form == "10-K":
            accession = recent["accessionNumber"][i]
            return {
                "cik": cik,
                "company": subs.get("name", ""),
                "accession": accession,
                "filing_date": recent["filingDate"][i],
                "report_date": recent["reportDate"][i],
                "primary_doc": recent["primaryDocument"][i],
            }
    raise ValueError(f"No 10-K found for CIK {cik}")


def filing_url(cik: str, accession: str, primary_doc: str) -> str:
    acc_nodash = accession.replace("-", "")
    return f"{SEC}/Archives/edgar/data/{int(cik)}/{acc_nodash}/{primary_doc}"


def html_to_text(html: bytes | str) -> str:
    """Convert a 10-K filing's HTML into clean plain text.

    Drops scripts, styles, and hidden XBRL payloads; collapses whitespace;
    keeps paragraph breaks so the chunker still sees structure.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.decompose()
    # 10-Ks embed XBRL facts in hidden elements; they read as garbage text.
    for tag in soup.find_all(style=re.compile(r"display\s*:\s*none", re.I)):
        tag.decompose()
    for tag in soup.find_all(attrs={"hidden": True}):
        tag.decompose()

    text = soup.get_text(separator="\n")
    # collapse whitespace, but keep paragraph breaks
    text = re.sub(r"[ \t\xa0\u200b]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    kept: list[str] = []
    for ln in (ln.strip() for ln in text.split("\n")):
        if not ln:
            continue
        # drop page numbers (bare 1-4 digit lines)...
        if re.fullmatch(r"\d{1,4}", ln):
            continue
        # ...and dot leaders / horizontal rules, but KEEP table figures
        # like "$ 201,183" or "42.5%": dropping those guts financial tables
        if re.fullmatch(r"[.\-–—_]+", ln):
            continue
        kept.append(ln)
    text = "\n".join(kept)
    # squeeze 3+ newlines that survived
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def download_10k(ticker: str, out_dir: str | Path) -> Path:
    """Download + clean the most recent 10-K for ``ticker``.

    Writes ``{TICKER}_10K_FY{year}.txt`` and a ``.meta.json`` sidecar into
    ``out_dir``. Returns the text path.
    """
    ticker = ticker.upper()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cik = ticker_to_cik(ticker)
    info = recent_10k(cik)
    url = filing_url(cik, info["accession"], info["primary_doc"])
    print(f"[{ticker}] {info['company']} — 10-K filed {info['filing_date']} "
          f"(period {info['report_date']})")
    print(f"[{ticker}] downloading {url}")
    html = _get_bytes(url)
    text = html_to_text(html)
    print(f"[{ticker}] cleaned to {len(text):,} chars")

    fy = (info["report_date"] or info["filing_date"])[:4]
    stem = f"{ticker}_10K_FY{fy}"
    text_path = out_dir / f"{stem}.txt"
    text_path.write_text(text, encoding="utf-8")
    meta = {
        "ticker": ticker,
        "company": info["company"],
        "cik": cik,
        "form": "10-K",
        "accession": info["accession"],
        "filing_date": info["filing_date"],
        "report_date": info["report_date"],
        "source_url": url,
        "chars": len(text),
    }
    (out_dir / f"{stem}.meta.json").write_text(json.dumps(meta, indent=2))
    return text_path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Download recent 10-K filings from SEC EDGAR")
    ap.add_argument("--tickers", nargs="+", required=True, help="e.g. AAPL MSFT GOOGL NVDA")
    ap.add_argument("--out", default="corpus/sec10k", help="output directory")
    args = ap.parse_args(argv)
    _user_agent()  # fail fast before any network
    for ticker in args.tickers:
        try:
            path = download_10k(ticker, args.out)
            print(f"[{ticker.upper()}] wrote {path}")
        except Exception as exc:  # noqa: BLE001 — report and continue with others
            print(f"[{ticker.upper()}] FAILED: {exc}")


if __name__ == "__main__":
    main()
