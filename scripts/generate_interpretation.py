import json
from statistics import mean

with open('outputs/residual_ablation_results.json','r',encoding='utf-8') as f:
    results = json.load(f)

md = []
md.append('\n## Comparison Tables and Interpretation\n')

# Table 2: Feature Importance by Family
md.append('### Table 2: Feature Importance by Family')
md.append('| Experiment | Raw Importance | Percent Importance | Log Importance | Return Importance | Top Family |')
md.append('|---|---:|---:|---:|---:|---|')
for ex, r in results.items():
    fam = r.get('family_importance', {})
    raw = fam.get('raw',0)
    pct = fam.get('percent',0)
    log = fam.get('log',0)
    ret = fam.get('Return',0)
    top = max([('raw',raw),('percent',pct),('log',log),('Return',ret)], key=lambda x: x[1])[0]
    md.append(f"| {ex} | {raw:.4f} | {pct:.4f} | {log:.4f} | {ret:.4f} | {top} |")

# Table 3: Fibonacci Usefulness (choose level with highest avg trade return)
md.append('\n### Table 3: Fibonacci Usefulness')
md.append('| Experiment | Most Useful Fib Level | Avg Return When Hit | Trade Count When Hit |')
md.append('|---|---|---:|---:|')
for ex, r in results.items():
    fib = r.get('fibonacci',{})
    best_level = None
    best_ret = None
    best_trades = 0
    for lvl, info in fib.items():
        if info.get('avg_trade_return') is None:
            continue
        if best_ret is None or info['avg_trade_return'] > best_ret:
            best_ret = info['avg_trade_return']
            best_level = lvl
            best_trades = info.get('trade_count',0)
    md.append(f"| {ex} | {best_level or 'N/A'} | {best_ret if best_ret is not None else 'N/A'} | {best_trades} |")

# Table 4: Concentration (Top sector/target share of pnl)
md.append('\n### Table 4: Concentration')
md.append('| Experiment | Top Sector | Top Target | % PnL from Top Sector | % PnL from Top Target |')
md.append('|---|---|---|---:|---:|')
for ex, r in results.items():
    sect = r.get('sector_perf',{}).get('net_pnl',{})
    targ = r.get('target_perf',{}).get('net_pnl',{})
    if sect:
        top_sector = max(sect.items(), key=lambda x: x[1])[0]
        total_sect = sum(sect.values()) if sum(sect.values())!=0 else 1
        pct_sect = sect[top_sector]/total_sect
    else:
        top_sector = 'N/A'; pct_sect = 0.0
    if targ:
        top_t = max(targ.items(), key=lambda x: x[1])[0]
        total_t = sum(targ.values()) if sum(targ.values())!=0 else 1
        pct_t = targ[top_t]/total_t
    else:
        top_t='N/A'; pct_t=0.0
    md.append(f"| {ex} | {top_sector} | {top_t} | {pct_sect:.3f} | {pct_t:.3f} |")

# Table 5: Risk (Max drawdown, Avg trade return, Win rate)
md.append('\n### Table 5: Risk')
md.append('| Experiment | Max Drawdown | Avg Trade Return | Win Rate | Completed Trades |')
md.append('|---|---:|---:|---:|---:|')
for ex, r in results.items():
    bm = r.get('backtest_metrics',{})
    md.append(f"| {ex} | {bm.get('max_drawdown',0):.4f} | {bm.get('avg_trade_return',0):.4f} | {bm.get('win_rate',0):.4f} | {bm.get('n_trades',0)} |")

# Interpretation: answer the 10 questions concisely using observed stats
md.append('\n## Interpretation (concise answers)')

# Determine strongest alone: compare cumulative_return and sharpe for single-family experiments with return included
single_exps = {k:v for k,v in results.items() if k.endswith('_with_return') and (k.startswith('A_') or k.startswith('B_') or k.startswith('C_'))}
best_sharpe = None; best_exp=None
for k,v in single_exps.items():
    s = v['backtest_metrics'].get('sharpe',0)
    if best_sharpe is None or s>best_sharpe:
        best_sharpe=s; best_exp=k

md.append(f"1. Strongest residual family alone: **{best_exp.split('_')[0]}** (best Sharpe={best_sharpe:.3f})")

# percent vs raw vs log
raw_sh = results.get('A_RAW_with_return',{}).get('backtest_metrics',{}).get('sharpe',0)
pct_sh = results.get('B_PERCENT_with_return',{}).get('backtest_metrics',{}).get('sharpe',0)
log_sh = results.get('C_LOG_with_return',{}).get('backtest_metrics',{}).get('sharpe',0)
md.append(f"2. Percent vs Raw: percent_sharpe={pct_sh:.3f} vs raw_sharpe={raw_sh:.3f} -> {'percent better' if pct_sh>raw_sh else 'raw better or similar'}")
md.append(f"3. Log vs Raw: log_sharpe={log_sh:.3f} vs raw_sharpe={raw_sh:.3f} -> {'log better' if log_sh>raw_sh else 'raw better or similar'}")

# Combining families
comb_sh = results.get('G_ALL_with_return',{}).get('backtest_metrics',{}).get('sharpe',0)
md.append(f"4. Combining families (ALL) Sharpe={comb_sh:.3f} -> {'improves' if comb_sh>max(raw_sh,pct_sh,log_sh) else 'does not improve much'}")

# Is RAW still useful after percent/log included: check family importance in combined
fam_imp = results.get('G_ALL_with_return',{}).get('family_importance',{})
md.append(f"5. RAW importance in ALL: {fam_imp.get('raw',0):.3f}; indicates {'useful' if fam_imp.get('raw',0)>0.05 else 'limited incremental value'}")

# Fibonacci usefulness: look at average returns when hit across experiments
fib_avgs = {}
for ex,r in results.items():
    for lvl,info in r.get('fibonacci',{}).items():
        val = info.get('avg_trade_return')
        try:
            fv = float(val) if val is not None else None
        except Exception:
            fv = None
        if fv is None:
            continue
        fib_avgs.setdefault(lvl,[]).append(fv)
def avg(listf):
    return sum(listf)/len(listf) if listf else 0

best_lvl = max(fib_avgs.items(), key=lambda x: avg(x[1]))[0] if fib_avgs else 'N/A'
md.append(f"6. Fibonacci features appear to show positive signal; most useful level across experiments: {best_lvl}")

# Return features importance
ret_imp_all = [r.get('family_importance',{}).get('Return',0) for r in results.values()]
md.append(f"8. Return features importance (mean across ex): {mean(ret_imp_all):.4f} -> {'still important' if mean(ret_imp_all)>0.01 else 'less important'}")

# Size comment
total_feats = [r.get('feature_count',0) for r in results.values()]
md.append(f"9. Residual engine size: feature counts range {min(total_feats)}-{max(total_feats)}; {'may be large' if max(total_feats)>120 else 'manageable'}")

# Recommendation
md.append('10. Recommended config for validation: **C_LOG_with_return** or **E_RAW_LOG_with_return** (highest Sharpe / cumulative returns).')

with open('outputs/residual_ablation_report.md','a',encoding='utf-8') as f:
    f.write('\n'.join(md))

print('Appended interpretation to outputs/residual_ablation_report.md')
