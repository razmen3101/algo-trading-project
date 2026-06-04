"""Add Cell 14 to Regression_Bakeoff.ipynb: dense-horizon R^2 grid + heatmap.
Uses cached ALL dict (pred_1 is contemporaneous prediction; compare to price.shift(-h))."""
import nbformat
from pathlib import Path

P = Path("Regression_Bakeoff.ipynb")
nb = nbformat.read(P, as_version=4)

cell_a = nbformat.v4.new_code_cell(source="""# Cell 14 — Dense forecast-horizon grid (R^2 for h=1..15) using cached pred_1
HORIZONS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15]

def horizon_metrics(df, window_idx, h):
    sub = df.loc[df.index.isin(window_idx)].dropna(subset=['pred_1','price']).copy()
    sub['actual_h'] = sub['price'].shift(-h)
    sub = sub.dropna(subset=['actual_h'])
    if len(sub) < 20: return None
    y_true, y_pred = sub['actual_h'].values, sub['pred_1'].values
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    r2  = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    mae = float(abs(y_true - y_pred).mean())
    rmse = float(((y_true - y_pred) ** 2).mean() ** 0.5)
    return dict(h=h, n=len(sub), r2=r2, mae=mae, rmse=rmse)

rows = []
for (sec, tgt, mname, regime), df in ALL.items():
    for label, widx in [('TRAIN', train_idx), ('VAL', val_idx), ('TEST', test_idx)]:
        for h in HORIZONS:
            r = horizon_metrics(df, widx, h)
            if r is None: continue
            rows.append({'sector': sec, 'target': tgt, 'model': mname,
                         'regime': regime, 'window': label, **r})
horizon_df = pd.DataFrame(rows)
horizon_df.to_csv(CACHE_DIR / 'horizon_metrics.csv', index=False)
print(f'Wrote horizon_metrics.csv ({len(horizon_df)} rows)')

for win in ('VAL', 'TEST'):
    pv = (horizon_df[horizon_df['window']==win]
          .groupby(['model','regime','h'])['r2'].mean()
          .unstack('h').round(3))
    pv = pv[HORIZONS]
    print(f'\\n=== {win} — mean R^2 across 10 targets, per horizon ===')
    print(pv.to_string())
""")

cell_b = nbformat.v4.new_code_cell(source="""# Cell 15 — Heatmap: model x horizon, mean R^2 on VAL (+ TEST)
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
for ax, win in zip(axes, ('VAL', 'TEST')):
    pv = (horizon_df[horizon_df['window']==win]
          .groupby(['model','regime','h'])['r2'].mean()
          .unstack('h'))
    pv = pv[HORIZONS]
    pv.index = [f'{m}|{r}' for (m, r) in pv.index]
    pv = pv.sort_values(pv.columns[0], ascending=False)
    arr = pv.values.astype(float)
    vmax = float(np.nanmax(np.abs(arr))) if np.isfinite(arr).any() else 1.0
    im = ax.imshow(arr, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(HORIZONS))); ax.set_xticklabels(HORIZONS)
    ax.set_yticks(range(len(pv.index))); ax.set_yticklabels(pv.index, fontsize=8)
    ax.set_title(f'{win}  mean R^2(h)  across 10 targets')
    ax.set_xlabel('horizon (days)')
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            if np.isfinite(v):
                ax.text(j, i, f'{v:+.2f}', ha='center', va='center',
                        fontsize=7, color='black' if abs(v) < vmax*0.5 else 'white')
    fig.colorbar(im, ax=ax, fraction=0.04)
plt.tight_layout(); plt.show()
""")

cell_c = nbformat.v4.new_code_cell(source="""# Cell 16 — Line plot: R^2 vs horizon, one panel per model
fig, axes = plt.subplots(1, len(MODEL_FACTORIES), figsize=(4.2*len(MODEL_FACTORIES), 4), sharey=True)
if len(MODEL_FACTORIES) == 1: axes = [axes]
for ax, mname in zip(axes, MODEL_FACTORIES):
    for regime in ('rolling', 'expanding'):
        for win, style in [('VAL', '-'), ('TEST', '--')]:
            sub = horizon_df[(horizon_df['model']==mname) &
                             (horizon_df['regime']==regime) &
                             (horizon_df['window']==win)]
            if sub.empty: continue
            ser = sub.groupby('h')['r2'].mean().reindex(HORIZONS)
            ax.plot(ser.index, ser.values, style, marker='o', markersize=3,
                    label=f'{regime}-{win}', linewidth=1.2, alpha=0.85)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_title(f'{mname}'); ax.set_xlabel('horizon (days)')
    if ax is axes[0]: ax.set_ylabel('mean R^2 across targets')
    ax.grid(alpha=0.3); ax.legend(fontsize=7, loc='best')
plt.tight_layout(); plt.show()
""")

nb.cells.extend([cell_a, cell_b, cell_c])
nbformat.write(nb, P)
print(f"appended 3 cells -> {P.name} (total: {len(nb.cells)})")
