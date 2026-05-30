# TRAIN-ONLY TARGET SELECTION AUDIT — RESULTS


## Part2 Current vs Best (by Residual Sharpe)

| sector           | current_target   |   current_sharpe | best_target   |   best_sharpe |   improvement_pct |
|:-----------------|:-----------------|-----------------:|:--------------|--------------:|------------------:|
| Communication    | META             |       -1.83867   | META          |    -1.83867   |                 0 |
| Consumer Disc.   | AMZN             |       -3.54157   | AMZN          |    -3.54157   |                 0 |
| Consumer Staples | PG               |       -3.12974   | PG            |    -3.12974   |                 0 |
| Energy           | XOM              |        1.87629   | XOM           |     1.87629   |                 0 |
| Financials       | JPM              |        2.08274   | JPM           |     2.08274   |                 0 |
| Health Care      | UNH              |       -1.72375   | UNH           |    -1.72375   |                 0 |
| Materials        | FCX              |        0.624604  | FCX           |     0.624604  |                 0 |
| Real Estate      | PLD              |       -0.0443737 | PLD           |    -0.0443737 |                 0 |
| Technology       | NVDA             |        0.809356  | NVDA          |     0.809356  |                 0 |
| Utilities        | NEE              |        1.97527   | NEE           |     1.97527   |                 0 |


## Part6 Shadow vs Tradability correlations

corr(resid_sharpe, avg_shadow_r2)=0.16206969655866424
corr(resid_sharpe, avg_return_r2)=0.03791425736078953



## Part8 Decisions

| sector           | decision   | reason                            | current   | best   |
|:-----------------|:-----------|:----------------------------------|:----------|:-------|
| Communication    | KEEP       | Sharpe diff 0.0000; Opp 16.0-16.0 | META      | META   |
| Consumer Disc.   | KEEP       | Sharpe diff 0.0000; Opp 24.0-24.0 | AMZN      | AMZN   |
| Consumer Staples | KEEP       | Sharpe diff 0.0000; Opp 21.0-21.0 | PG        | PG     |
| Energy           | KEEP       | Sharpe diff 0.0000; Opp 16.0-16.0 | XOM       | XOM    |
| Financials       | KEEP       | Sharpe diff 0.0000; Opp 24.0-24.0 | JPM       | JPM    |
| Health Care      | KEEP       | Sharpe diff 0.0000; Opp 26.0-26.0 | UNH       | UNH    |
| Materials        | KEEP       | Sharpe diff 0.0000; Opp 21.0-21.0 | FCX       | FCX    |
| Real Estate      | KEEP       | Sharpe diff 0.0000; Opp 21.0-21.0 | PLD       | PLD    |
| Technology       | KEEP       | Sharpe diff 0.0000; Opp 22.0-22.0 | NVDA      | NVDA   |
| Utilities        | KEEP       | Sharpe diff 0.0000; Opp 18.0-18.0 | NEE       | NEE    |

## Detailed Decisions
| sector           | decision   | current   | best   |   cur_sharpe |   best_sharpe |   sharpe_diff |   cur_half |   best_half |   half_diff |   cur_opp |   best_opp |   opp_diff |   cur_pred_turn |   best_pred_turn |   pred_turn_diff |   cur_tgt_stab |   best_tgt_stab |   tgt_stab_diff | reason                       |
|:-----------------|:-----------|:----------|:-------|-------------:|--------------:|--------------:|-----------:|------------:|------------:|----------:|-----------:|-----------:|----------------:|-----------------:|-----------------:|---------------:|----------------:|----------------:|:-----------------------------|
| Communication    | KEEP       | META      | META   |   -1.83867   |    -1.83867   |             0 |        nan |         nan |         nan |        16 |         16 |          0 |       0.047619  |        0.047619  |                0 |              1 |               1 |               0 | SharpeDiff=0.0000, OppDiff=0 |
| Consumer Disc.   | KEEP       | AMZN      | AMZN   |   -3.54157   |    -3.54157   |             0 |        nan |         nan |         nan |        24 |         24 |          0 |       0.190476  |        0.190476  |                0 |              1 |               1 |               0 | SharpeDiff=0.0000, OppDiff=0 |
| Consumer Staples | KEEP       | PG        | PG     |   -3.12974   |    -3.12974   |             0 |        nan |         nan |         nan |        21 |         21 |          0 |       0.047619  |        0.047619  |                0 |              1 |               1 |               0 | SharpeDiff=0.0000, OppDiff=0 |
| Energy           | KEEP       | XOM       | XOM    |    1.87629   |     1.87629   |             0 |        nan |         nan |         nan |        16 |         16 |          0 |       0.190476  |        0.190476  |                0 |              1 |               1 |               0 | SharpeDiff=0.0000, OppDiff=0 |
| Financials       | KEEP       | JPM       | JPM    |    2.08274   |     2.08274   |             0 |        nan |         nan |         nan |        24 |         24 |          0 |       0.0952381 |        0.0952381 |                0 |              1 |               1 |               0 | SharpeDiff=0.0000, OppDiff=0 |
| Health Care      | KEEP       | UNH       | UNH    |   -1.72375   |    -1.72375   |             0 |        nan |         nan |         nan |        26 |         26 |          0 |       0.047619  |        0.047619  |                0 |              1 |               1 |               0 | SharpeDiff=0.0000, OppDiff=0 |
| Materials        | KEEP       | FCX       | FCX    |    0.624604  |     0.624604  |             0 |        nan |         nan |         nan |        21 |         21 |          0 |       0.238095  |        0.238095  |                0 |              1 |               1 |               0 | SharpeDiff=0.0000, OppDiff=0 |
| Real Estate      | KEEP       | PLD       | PLD    |   -0.0443737 |    -0.0443737 |             0 |        nan |         nan |         nan |        21 |         21 |          0 |       0.190476  |        0.190476  |                0 |              1 |               1 |               0 | SharpeDiff=0.0000, OppDiff=0 |
| Technology       | KEEP       | NVDA      | NVDA   |    0.809356  |     0.809356  |             0 |        nan |         nan |         nan |        22 |         22 |          0 |       0         |        0         |                0 |              1 |               1 |               0 | SharpeDiff=0.0000, OppDiff=0 |
| Utilities        | KEEP       | NEE       | NEE    |    1.97527   |     1.97527   |             0 |        nan |         nan |         nan |        18 |         18 |          0 |       0.1       |        0.1       |                0 |              1 |               1 |               0 | SharpeDiff=0.0000, OppDiff=0 |