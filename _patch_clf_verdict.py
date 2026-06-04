"""Patch cell 11 of DBTS_Variants_Classifier.ipynb to filter degenerate runs."""
import json, pathlib
p = pathlib.Path('DBTS_Variants_Classifier.ipynb')
nb = json.loads(p.read_text(encoding='utf-8'))

new_src = '''# Cell 11 — Edge analysis: best CLF mode vs NoCLF on VAL (filter degenerate runs)
MIN_ENTRIES = 100   # ignore degenerate variants with too few trades
val = summary[summary['split']=='VAL'].copy()
noclf_val = val[val['mode']=='NoCLF'].iloc[0]
clf_val = val[(val['mode']!='NoCLF') & (val['entries'] >= MIN_ENTRIES)].copy()
print(f'CLF candidates with entries >= {MIN_ENTRIES} on VAL:')
if clf_val.empty:
    print('  (none)')
else:
    print(clf_val[['mode','conf_thr','entries','sharpe','sortino','cum_ret','max_dd']]
          .sort_values('sharpe', ascending=False).to_string(index=False))

if clf_val.empty:
    PROMOTE = False; PROMO_MODE = 'NoCLF'; PROMO_CONF = 0.0
    print('\\n>>> VERDICT: classifier too restrictive (no variant has enough trades). Keep NoCLF.')
else:
    best_clf = clf_val.sort_values('sharpe', ascending=False).iloc[0]
    delta_sh = best_clf['sharpe'] - noclf_val['sharpe']
    delta_cum = best_clf['cum_ret'] - noclf_val['cum_ret']
    print(f'\\nNoCLF VAL    : Sharpe={noclf_val["sharpe"]:.4f} cum_ret={noclf_val["cum_ret"]:.4f} entries={int(noclf_val["entries"])} max_dd={noclf_val["max_dd"]:.4f}')
    print(f'Best CLF VAL : mode={best_clf["mode"]} conf={best_clf["conf_thr"]} Sharpe={best_clf["sharpe"]:.4f} cum_ret={best_clf["cum_ret"]:.4f} entries={int(best_clf["entries"])} max_dd={best_clf["max_dd"]:.4f}')
    print(f'delta Sharpe = {delta_sh:+.4f} | delta cum_ret = {delta_cum:+.4f}')
    VERDICT_THR = 0.10
    if delta_sh >= VERDICT_THR:
        print(f'\\n>>> VERDICT: Classifier ADDS Sharpe edge (delta >= {VERDICT_THR}). Promote {best_clf["mode"]}@{best_clf["conf_thr"]} to TEST.')
        PROMOTE = True; PROMO_MODE = best_clf['mode']; PROMO_CONF = float(best_clf['conf_thr'])
    else:
        print(f'\\n>>> VERDICT: Classifier does NOT add meaningful Sharpe edge (delta < {VERDICT_THR}). Keep NoCLF.')
        PROMOTE = False; PROMO_MODE = 'NoCLF'; PROMO_CONF = 0.0
'''

# find cell with old VERDICT_THR text
for i, c in enumerate(nb['cells']):
    src = ''.join(c.get('source', []))
    if 'Edge analysis' in src and 'VERDICT_THR' in src:
        c['source'] = [ln + '\n' for ln in new_src.splitlines()]
        # strip trailing \n on last line for cleanliness
        c['source'][-1] = c['source'][-1].rstrip('\n')
        c['outputs'] = []
        c['execution_count'] = None
        print(f'patched cell {i}')
        break

p.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
print('saved')
