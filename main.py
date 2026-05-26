import pandas as pd
from data.loader import Loader
from analysis.indicators import Indicators
from tabulate import tabulate


def print_section(title: str) -> None:
    print(f"\n{'='*72}\n  {title}\n{'='*72}")


def run():
    md  = Loader().load()
    ind = Indicators(md.prices, md.highs, md.lows, md.volumes)

    # --- Full summary ---
    print_section("ALL STOCKS SUMMARY")
    print(tabulate(md.summary, headers='keys', tablefmt='rounded_outline',
                   floatfmt='.2f', numalign='right'))

    # --- Per-sector: target vs predictors correlation ---
    print_section("TARGET CORRELATION WITH PREDICTORS (per sector)")
    rows = []
    for etf, sv in md.sectors().items():
        corr_with_target = sv.correlation[sv.target].drop(sv.target).sort_values(ascending=False)
        for pred, val in corr_with_target.items():
            rows.append({'Sector': sv.name, 'Target': sv.target, 'Predictor': pred, 'Corr': round(val, 4)})
    print(tabulate(pd.DataFrame(rows), headers='keys', tablefmt='rounded_outline',
                   floatfmt='.4f', showindex=False, numalign='right'))

    # --- Composite signals ---
    print_section("COMPOSITE SIGNALS  (Score: -3 sell -> +3 buy)")
    sig = ind.signals()
    print(tabulate(sig, headers='keys', tablefmt='rounded_outline',
                   floatfmt='.1f', numalign='right'))

    # --- RSI extremes ---
    print_section("RSI EXTREMES")
    rsi_last = ind.rsi().iloc[-1].round(1).rename('RSI')
    oversold  = rsi_last[rsi_last < 30].sort_values()
    overbought = rsi_last[rsi_last > 70].sort_values(ascending=False)
    print("  Oversold  (RSI < 30):")
    print(tabulate(oversold.reset_index().rename(columns={'index': 'Ticker'}),
                   headers='keys', tablefmt='rounded_outline', floatfmt='.1f',
                   showindex=False, numalign='right'))
    print("  Overbought (RSI > 70):")
    print(tabulate(overbought.reset_index().rename(columns={'index': 'Ticker'}),
                   headers='keys', tablefmt='rounded_outline', floatfmt='.1f',
                   showindex=False, numalign='right'))

    # --- ATR + PFE for sector targets ---
    print_section("SECTOR TARGETS — ATR (daily $-risk)  &  PFE (trend efficiency)")
    atr_last = ind.atr().iloc[-1].round(2)
    pfe_last = ind.pfe().iloc[-1].round(1)
    target_rows = []
    for etf, sv in md.sectors().items():
        target_rows.append({
            'Sector':  sv.name,
            'Target':  sv.target,
            'Price':   round(md.prices[sv.target].iloc[-1], 2),
            'ATR ($)': atr_last[sv.target],
            'ATR %':   round(atr_last[sv.target] / md.prices[sv.target].iloc[-1] * 100, 2),
            'PFE':     pfe_last[sv.target],
        })
    print(tabulate(pd.DataFrame(target_rows), headers='keys', tablefmt='rounded_outline',
                   floatfmt='.2f', showindex=False, numalign='right'))

    # --- Fibonacci levels for sector targets ---
    print_section("FIBONACCI LEVELS — SECTOR TARGETS (100-day rolling window)")
    fib   = ind.fibonacci()
    price = md.prices.iloc[-1]
    fib_rows = []
    for _, sv in md.sectors().items():
        t = sv.target
        fib_rows.append({
            'Sector': sv.name,
            'Target': t,
            'Price':  round(price[t], 2),
            '38.2%':  round(fib['38.2'][t].iloc[-1], 2),
            '50.0%':  round(fib['50.0'][t].iloc[-1], 2),
            '61.8%':  round(fib['61.8'][t].iloc[-1], 2),
        })
    print(tabulate(pd.DataFrame(fib_rows), headers='keys', tablefmt='rounded_outline',
                   floatfmt='.2f', showindex=False, numalign='right'))


if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    run()
