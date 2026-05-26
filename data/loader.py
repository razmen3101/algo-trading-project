from dataclasses import dataclass
from typing import NamedTuple
import yfinance as yf
import pandas as pd
import numpy as np
from config import SECTORS, ALL_TICKERS, DEFAULT_START, DEFAULT_END, DEFAULT_INTERVAL, OUTLIER_THRESHOLD, MAX_FILL_DAYS


class SectorView(NamedTuple):
    name:        str
    etf:         str
    target:      str
    predictors:  list[str]
    prices:      pd.DataFrame
    returns:     pd.DataFrame
    correlation: pd.DataFrame


@dataclass
class MarketData:
    prices:  pd.DataFrame
    returns: pd.DataFrame
    highs:   pd.DataFrame
    lows:    pd.DataFrame
    volumes: pd.DataFrame
    summary: pd.DataFrame

    def sector(self, etf: str) -> SectorView:
        cfg     = SECTORS[etf]
        tickers = [cfg['target']] + cfg['predictors']
        prices  = self.prices[tickers]
        returns = self.returns[tickers]
        return SectorView(cfg['name'], etf, cfg['target'], cfg['predictors'], prices, returns, returns.corr())

    def sectors(self) -> dict[str, SectorView]:
        return {etf: self.sector(etf) for etf in SECTORS}


class Loader:
    def __init__(self, tickers=ALL_TICKERS, start=DEFAULT_START, end=DEFAULT_END, interval=DEFAULT_INTERVAL):
        self.tickers  = tickers
        self.start    = start
        self.end      = end
        self.interval = interval

    def load(self) -> MarketData:
        raw                                   = self._fetch()
        prices, returns, highs, lows, volumes = self._clean(raw)
        summary                               = self._summary(prices, returns)
        print(f"[load] Ready — {len(prices)} trading days x {len(prices.columns)} stocks")
        return MarketData(prices=prices, returns=returns, highs=highs, lows=lows, volumes=volumes, summary=summary)

    def _fetch(self) -> pd.DataFrame:
        print(f"[fetch] {len(self.tickers)} tickers | {self.start} -> {self.end or 'today'} | {self.interval}")
        raw = yf.download(self.tickers, start=self.start, end=self.end, interval=self.interval,
                          group_by='ticker', auto_adjust=True, progress=False, threads=True)
        if raw.empty:
            raise ValueError("yfinance returned empty DataFrame")
        return raw

    def _clean_ticker(self, raw: pd.DataFrame, ticker: str) -> pd.DataFrame | None:
        try:
            df = raw[ticker][['Close', 'High', 'Low', 'Volume']].copy()
        except KeyError:
            print(f"[WARN] {ticker} missing — skipped")
            return None
        df = df[~df.index.duplicated(keep='first')].sort_index()
        return df[df['Close'] > 0].dropna(subset=['Close'])

    def _clean(self, raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        dfs = {t: df for t, df in ((t, self._clean_ticker(raw, t)) for t in self.tickers) if df is not None}

        def build(col):
            out = pd.DataFrame({t: dfs[t][col] for t in dfs})
            out.index = pd.to_datetime(out.index)
            return out

        prices  = build('Close').ffill(limit=MAX_FILL_DAYS).bfill(limit=1)
        highs   = build('High').ffill(limit=MAX_FILL_DAYS).bfill(limit=1)
        lows    = build('Low').ffill(limit=MAX_FILL_DAYS).bfill(limit=1)
        volumes = build('Volume').fillna(0)

        valid_idx                       = prices.dropna(how='all').index
        prices, highs, lows, volumes    = (df.loc[valid_idx] for df in [prices, highs, lows, volumes])

        returns = prices.pct_change(fill_method=None).iloc[1:]
        n = int((returns.abs() > OUTLIER_THRESHOLD).sum().sum())
        if n:
            print(f"[INFO] {n} returns exceed {OUTLIER_THRESHOLD*100:.0f}% — kept")
        return prices, returns, highs, lows, volumes

    def _summary(self, prices: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
        last  = prices.index[-1]
        roles = {cfg['target']: ('target', cfg['name']) for cfg in SECTORS.values()} | \
                {t: ('predictor', cfg['name']) for cfg in SECTORS.values() for t in cfg['predictors']}

        def ret(since):
            sub = prices[prices.index >= since]
            return pd.Series(np.nan, index=prices.columns) if sub.empty else \
                   ((prices.loc[last] / sub.iloc[0]) - 1) * 100

        df = pd.DataFrame(index=prices.columns)
        df.index.name     = 'Ticker'
        df['Sector']      = df.index.map(lambda t: roles[t][1])
        df['Role']        = df.index.map(lambda t: roles[t][0])
        df['Last Price']  = prices.loc[last].round(2)
        df['YTD %']       = ret(pd.Timestamp(f'{last.year}-01-01')).round(2)
        df['1Y %']        = ret(last - pd.DateOffset(years=1)).round(2)
        df['Ann. Vol %']  = (returns.std() * np.sqrt(252) * 100).round(2)
        df['Sharpe (1Y)'] = ((returns.tail(252).mean() * 252) / (returns.tail(252).std() * np.sqrt(252))).round(3)
        df['Days']        = prices.count()
        return df.sort_values(['Sector', 'Role'])
