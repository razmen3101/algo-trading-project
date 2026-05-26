import pandas as pd
import numpy as np


class Indicators:
    def __init__(self, prices: pd.DataFrame, highs: pd.DataFrame,
                 lows: pd.DataFrame, volumes: pd.DataFrame):
        self.prices  = prices
        self.highs   = highs
        self.lows    = lows
        self.volumes = volumes

    # --- Trend ---

    def sma(self, periods=(10, 25, 50)) -> dict[int, pd.DataFrame]:
        return {n: self.prices.rolling(n).mean() for n in periods}

    def ema(self, periods=(10, 25, 50)) -> dict[int, pd.DataFrame]:
        return {n: self.prices.ewm(span=n, adjust=False).mean() for n in periods}

    def macd(self, fast=12, slow=26, signal=9) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        fast_ema    = self.prices.ewm(span=fast,   adjust=False).mean()
        slow_ema    = self.prices.ewm(span=slow,   adjust=False).mean()
        macd_line   = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        return macd_line, signal_line, macd_line - signal_line  # line, signal, histogram

    # --- Momentum ---

    def rsi(self, n=14) -> pd.DataFrame:
        delta = self.prices.diff()
        gain  = delta.clip(lower=0).ewm(com=n - 1, min_periods=n).mean()
        loss  = (-delta).clip(lower=0).ewm(com=n - 1, min_periods=n).mean()
        return 100 - 100 / (1 + gain / loss)

    def pfe(self, n=14) -> pd.DataFrame:
        """Polarized Fractal Efficiency: 100 = perfectly linear trend, 0 = pure chop."""
        num   = np.sqrt(self.prices.diff(n) ** 2 + n ** 2)
        steps = np.sqrt(self.prices.diff(1) ** 2 + 1)
        den   = steps.rolling(n - 1).sum()
        return 100 * num / den

    # --- Volume ---

    def obv(self) -> pd.DataFrame:
        return (np.sign(self.prices.diff()) * self.volumes).cumsum()

    def volume_profile(self, bins=50) -> dict[str, dict]:
        """Distributes daily volume evenly across High-Low range. Returns POC and profile per ticker."""
        result = {}
        for ticker in self.prices.columns:
            h, l, v  = self.highs[ticker].values, self.lows[ticker].values, self.volumes[ticker].values
            all_px   = np.concatenate([h[~np.isnan(h)], l[~np.isnan(l)]])
            edges    = np.linspace(all_px.min(), all_px.max(), bins + 1)
            centers  = (edges[:-1] + edges[1:]) / 2
            vol_hist = np.zeros(bins)
            for hi_, lo_, vi in zip(h, l, v):
                if np.isnan(hi_) or np.isnan(lo_) or np.isnan(vi) or vi == 0:
                    continue
                mask   = (centers >= lo_) & (centers <= hi_)
                n_bins = mask.sum()
                if n_bins:
                    vol_hist[mask] += vi / n_bins
            result[ticker] = {
                'poc':     round(float(centers[vol_hist.argmax()]), 2),
                'profile': pd.Series(vol_hist, index=centers.round(2)),
            }
        return result

    # --- Volatility / Risk ---

    def atr(self, n=14) -> pd.DataFrame:
        prev = self.prices.shift(1)
        tr   = pd.DataFrame(
            np.maximum(self.highs.values - self.lows.values,
                       np.maximum(np.abs(self.highs.values - prev.values),
                                  np.abs(self.lows.values - prev.values))),
            index=self.prices.index, columns=self.prices.columns,
        )
        return tr.ewm(span=n, adjust=False).mean()

    # --- Structural levels ---

    def fibonacci(self, window=100) -> dict[str, pd.DataFrame]:
        hi  = self.highs.rolling(window).max()
        lo  = self.lows.rolling(window).min()
        rng = hi - lo
        return {
            '0.0':   lo,
            '23.6':  lo + 0.236 * rng,
            '38.2':  lo + 0.382 * rng,
            '50.0':  lo + 0.500 * rng,
            '61.8':  lo + 0.618 * rng,
            '100.0': hi,
        }

    # --- Composite signals ---

    def signals(self) -> pd.DataFrame:
        """
        Latest signal per ticker based on RSI, MACD cross, and price vs SMA50.
        Score -3 (strong sell) to +3 (strong buy).
        """
        rsi_val           = self.rsi().iloc[-1]
        macd_l, sig_l, _  = self.macd()
        sma50             = self.prices.rolling(50).mean().iloc[-1]

        rsi_sig  = pd.Series(
            np.where(rsi_val < 30, 1, np.where(rsi_val > 70, -1, 0)),
            index=rsi_val.index,
        )
        macd_sig = np.sign(macd_l.iloc[-1] - sig_l.iloc[-1]).astype(int)
        sma_sig  = np.sign(self.prices.iloc[-1] - sma50).astype(int)

        return pd.DataFrame({
            'RSI':         rsi_val.round(1),
            'RSI_signal':  rsi_sig,
            'MACD_cross':  macd_sig,
            'Above_SMA50': sma_sig,
            'Score':       rsi_sig + macd_sig + sma_sig,
        }).sort_values('Score', ascending=False)
