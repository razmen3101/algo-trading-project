# Residual-family Ablation Report (Train Only)

Generated experiments and diagnostics for residual-family ablation. Train period: before 2024-04-01.

## Experiment: A_RAW_with_return

### Feature counts
- **Total features**: 96
- **Raw residual features**: 28
- **Percent residual features**: 0
- **Log residual features**: 0
- **Return features**: 1

### Classification diagnostics
- **Accuracy**: 0.9737
- **Macro F1**: 0.9719
- **Weighted F1**: 0.9734

Confusion matrix:
```
82 | 3 | 0
0 | 346 | 0
0 | 12 | 127
```

### Backtest metrics (PositionManager enabled)
- **cumulative_return**: 0.073359
- **annualized_return**: 0.367492
- **annualized_vol**: 0.031553
- **sharpe**: 11.646859
- **max_drawdown**: -0.003131
- **win_rate**: 0.857143
- **n_trades**: 39
- **avg_trade_return**: 0.014673
- **n_long**: 127
- **n_short**: 82

### Top 30 features
- raw_residual_rank: 0.038979
- oh_XLK: 0.030111
- oh_XLV: 0.024212
- oh_XLP: 0.021120
- oh_XLE: 0.018834
- rule_sma20_gt_50: 0.018406
- oh_XLB: 0.016399
- rule_macd_above_sig: 0.016300
- residual_distance_from_zero: 0.016247
- raw_residual_distance_from_zero: 0.015428
- px_ema200_ratio_z: 0.014988
- oh_XLF: 0.014985
- rsi_z: 0.014908
- raw_residual_ewm_mean: 0.014859
- shadow_price_gap_pct: 0.014793
- px_ema50_ratio_z: 0.014606
- px_sma50_ratio_z: 0.014381
- raw_residual_ewm_std: 0.014314
- raw_residual: 0.013932
- rule_px_gt_sma200: 0.013928
- px_sma200_ratio_z: 0.013837
- residual_abs_z: 0.013591
- rule_near_52w_high: 0.013567
- raw_residual_abs_z: 0.013446
- residual_ewm_z: 0.013406
- sect_vol_rel_etf: 0.013338
- sect_target_minus_etf: 0.013259
- sect_corr_etf: 0.013152
- px_sma20_ratio_z: 0.012961
- residual_rank: 0.012844

### Family importance
- **raw**: 0.243359
- **percent**: 0.0
- **log**: 0.0
- **Return**: 0.012315
- **Other**: 0.744326

### Fibonacci diagnostics
- **fib_23_hit**: hits=96, trades=20, avg_ret=0.03594844870871553, win_rate=1.0
- **fib_38_hit**: hits=67, trades=20, avg_ret=0.03594844870871553, win_rate=1.0
- **fib_50_hit**: hits=45, trades=20, avg_ret=0.03594844870871553, win_rate=1.0
- **fib_61_hit**: hits=19, trades=11, avg_ret=0.03281781023455183, win_rate=1.0
- **fib_78_hit**: hits=4, trades=2, avg_ret=0.025875636614107356, win_rate=1.0

### Residual diagnostics
- **raw**: mean_z=0.2950, std_z=1.8159, pct>|1.25|=0.339

---

## Experiment: A_RAW_no_return

### Feature counts
- **Total features**: 95
- **Raw residual features**: 28
- **Percent residual features**: 0
- **Log residual features**: 0
- **Return features**: 0

### Classification diagnostics
- **Accuracy**: 0.9702
- **Macro F1**: 0.9676
- **Weighted F1**: 0.9699

Confusion matrix:
```
81 | 4 | 0
0 | 346 | 0
0 | 13 | 126
```

### Backtest metrics (PositionManager enabled)
- **cumulative_return**: 0.07251
- **annualized_return**: 0.362715
- **annualized_vol**: 0.030922
- **sharpe**: 11.729947
- **max_drawdown**: -0.003131
- **win_rate**: 0.87234
- **n_trades**: 37
- **avg_trade_return**: 0.015117
- **n_long**: 126
- **n_short**: 81

### Top 30 features
- rule_obv_trend_pos: 0.037676
- oh_XLK: 0.030605
- oh_XLP: 0.022672
- rule_sma20_gt_50: 0.019775
- raw_residual_rank: 0.019161
- oh_XLC: 0.017137
- raw_residual_ewm_mean: 0.017039
- oh_XLB: 0.016799
- raw_residual_z: 0.016682
- residual_distance_from_zero: 0.016461
- shadow_price_gap_pct: 0.015592
- rsi_z: 0.015195
- raw_residual_ewm_std: 0.014768
- px_ema200_ratio_z: 0.014749
- residual_half_life_proxy: 0.014708
- residual_abs_z: 0.014667
- px_ema50_ratio_z: 0.014479
- px_sma50_ratio_z: 0.014425
- px_ema20_ratio_z: 0.014412
- sect_corr_etf: 0.014286
- rule_px_gt_sma20: 0.014166
- raw_residual_abs_z: 0.014064
- rule_volume_spike: 0.013996
- raw_residual: 0.013903
- raw_residual_distance_from_zero: 0.013879
- px_sma200_ratio_z: 0.013805
- raw_residual_regime_elevated: 0.013705
- sect_target_minus_etf: 0.013692
- residual_percentile: 0.013643
- macd_signal_z: 0.013608

### Family importance
- **raw**: 0.248738
- **percent**: 0.0
- **log**: 0.0
- **Return**: 0.0
- **Other**: 0.751262

### Fibonacci diagnostics
- **fib_23_hit**: hits=96, trades=19, avg_ret=0.03739481129927125, win_rate=1.0
- **fib_38_hit**: hits=67, trades=19, avg_ret=0.03739481129927125, win_rate=1.0
- **fib_50_hit**: hits=45, trades=19, avg_ret=0.03739481129927125, win_rate=1.0
- **fib_61_hit**: hits=19, trades=11, avg_ret=0.033396143906956964, win_rate=1.0
- **fib_78_hit**: hits=4, trades=2, avg_ret=0.02905647181233556, win_rate=1.0

### Residual diagnostics
- **raw**: mean_z=0.2950, std_z=1.8159, pct>|1.25|=0.339

---

## Experiment: B_PERCENT_with_return

### Feature counts
- **Total features**: 96
- **Raw residual features**: 0
- **Percent residual features**: 28
- **Log residual features**: 0
- **Return features**: 1

### Classification diagnostics
- **Accuracy**: 0.9737
- **Macro F1**: 0.9726
- **Weighted F1**: 0.9734

Confusion matrix:
```
83 | 2 | 0
0 | 346 | 0
0 | 13 | 126
```

### Backtest metrics (PositionManager enabled)
- **cumulative_return**: 0.071367
- **annualized_return**: 0.356311
- **annualized_vol**: 0.031195
- **sharpe**: 11.421951
- **max_drawdown**: -0.003131
- **win_rate**: 0.843137
- **n_trades**: 39
- **avg_trade_return**: 0.013732
- **n_long**: 126
- **n_short**: 83

### Top 30 features
- oh_XLV: 0.025497
- oh_XLK: 0.024420
- oh_XLP: 0.021825
- rule_macd_above_sig: 0.020909
- percent_residual_ewm_mean: 0.020547
- oh_XLE: 0.019265
- oh_XLB: 0.018986
- px_ema50_ratio_z: 0.016741
- residual_distance_from_zero: 0.016451
- px_sma50_ratio_z: 0.016178
- rule_px_gt_sma20: 0.015891
- px_ema200_ratio_z: 0.015753
- rule_near_52w_high: 0.015381
- residual_abs_z: 0.015186
- percent_residual_ewm_std: 0.015129
- shadow_price_gap_pct: 0.015071
- rsi_z: 0.014928
- residual_half_life_proxy: 0.014780
- percent_residual_distance_from_zero: 0.014470
- oh_XLC: 0.014225
- percent_residual_z: 0.014046
- sect_target_minus_etf: 0.013844
- sect_vol_rel_etf: 0.013615
- predicted_return: 0.013505
- residual_excursion_bucket: 0.013485
- macd_signal_z: 0.013454
- sect_corr_pred: 0.013297
- rule_breakout_20: 0.013273
- percent_residual_abs_z: 0.013157
- sect_corr_etf: 0.013141

### Family importance
- **raw**: 0.0
- **percent**: 0.238884
- **log**: 0.0
- **Return**: 0.013505
- **Other**: 0.747612

### Fibonacci diagnostics
- **fib_23_hit**: hits=96, trades=20, avg_ret=0.035017574308856846, win_rate=1.0
- **fib_38_hit**: hits=67, trades=20, avg_ret=0.035017574308856846, win_rate=1.0
- **fib_50_hit**: hits=45, trades=20, avg_ret=0.035017574308856846, win_rate=1.0
- **fib_61_hit**: hits=19, trades=11, avg_ret=0.031125311325717865, win_rate=1.0
- **fib_78_hit**: hits=4, trades=2, avg_ret=0.025875636614107356, win_rate=1.0

### Residual diagnostics
- **percent**: mean_z=0.2865, std_z=1.8007, pct>|1.25|=0.323

---

## Experiment: B_PERCENT_no_return

### Feature counts
- **Total features**: 95
- **Raw residual features**: 0
- **Percent residual features**: 28
- **Log residual features**: 0
- **Return features**: 0

### Classification diagnostics
- **Accuracy**: 0.9737
- **Macro F1**: 0.9726
- **Weighted F1**: 0.9734

Confusion matrix:
```
83 | 2 | 0
0 | 346 | 0
0 | 13 | 126
```

### Backtest metrics (PositionManager enabled)
- **cumulative_return**: 0.06865
- **annualized_return**: 0.34117
- **annualized_vol**: 0.031212
- **sharpe**: 10.930782
- **max_drawdown**: -0.003131
- **win_rate**: 0.87234
- **n_trades**: 37
- **avg_trade_return**: 0.01435
- **n_long**: 126
- **n_short**: 83

### Top 30 features
- oh_XLK: 0.025702
- percent_residual_rank: 0.024336
- rule_obv_trend_pos: 0.022131
- oh_XLP: 0.021674
- percent_residual_ewm_mean: 0.020075
- oh_XLB: 0.018256
- px_ema200_ratio_z: 0.016893
- residual_abs_z: 0.016517
- residual_distance_from_zero: 0.016504
- oh_XLU: 0.016471
- shadow_price_gap_pct: 0.015897
- percent_residual: 0.015896
- rsi_z: 0.015712
- oh_XLE: 0.015639
- percent_residual_ewm_std: 0.015623
- px_sma50_ratio_z: 0.015379
- sect_corr_etf: 0.014746
- px_ema50_ratio_z: 0.014497
- percent_fib_retrace_pct: 0.014337
- residual_excursion_bucket: 0.014262
- rule_breakout_20: 0.014147
- px_sma200_ratio_z: 0.014087
- macd_signal_z: 0.014026
- percent_residual_distance_from_zero: 0.013968
- sect_target_minus_etf: 0.013925
- oh_XLY: 0.013806
- residual_percentile: 0.013738
- sect_vol_rel_etf: 0.013558
- rule_volume_spike: 0.013482
- residual_half_life_proxy: 0.013455

### Family importance
- **raw**: 0.0
- **percent**: 0.276952
- **log**: 0.0
- **Return**: 0.0
- **Other**: 0.723048

### Fibonacci diagnostics
- **fib_23_hit**: hits=96, trades=19, avg_ret=0.03549636976985403, win_rate=1.0
- **fib_38_hit**: hits=67, trades=19, avg_ret=0.03549636976985403, win_rate=1.0
- **fib_50_hit**: hits=45, trades=19, avg_ret=0.03549636976985403, win_rate=1.0
- **fib_61_hit**: hits=19, trades=10, avg_ret=0.03164579640329861, win_rate=1.0
- **fib_78_hit**: hits=4, trades=2, avg_ret=0.02905647181233556, win_rate=1.0

### Residual diagnostics
- **percent**: mean_z=0.2865, std_z=1.8007, pct>|1.25|=0.323

---

## Experiment: C_LOG_with_return

### Feature counts
- **Total features**: 96
- **Raw residual features**: 0
- **Percent residual features**: 0
- **Log residual features**: 28
- **Return features**: 1

### Classification diagnostics
- **Accuracy**: 0.9772
- **Macro F1**: 0.9768
- **Weighted F1**: 0.9769

Confusion matrix:
```
84 | 1 | 0
0 | 346 | 0
0 | 12 | 127
```

### Backtest metrics (PositionManager enabled)
- **cumulative_return**: 0.075508
- **annualized_return**: 0.379637
- **annualized_vol**: 0.031437
- **sharpe**: 12.076295
- **max_drawdown**: -0.003131
- **win_rate**: 0.877551
- **n_trades**: 41
- **avg_trade_return**: 0.015092
- **n_long**: 127
- **n_short**: 84

### Top 30 features
- oh_XLK: 0.025665
- oh_XLV: 0.024586
- oh_XLP: 0.021132
- log_residual_ewm_mean: 0.020067
- oh_XLB: 0.019229
- rule_macd_above_sig: 0.018177
- oh_XLE: 0.018108
- residual_excursion_bucket: 0.017615
- residual_distance_from_zero: 0.017383
- px_ema200_ratio_z: 0.016972
- px_ema50_ratio_z: 0.016882
- rule_near_52w_high: 0.016810
- rsi_z: 0.015079
- log_residual_ewm_std: 0.015054
- shadow_price_gap_pct: 0.014790
- px_sma50_ratio_z: 0.014672
- log_residual_abs_z: 0.014123
- log_residual_distance_from_zero: 0.013959
- oh_XLC: 0.013935
- residual_abs_z: 0.013834
- log_acceleration: 0.013802
- residual_half_life_proxy: 0.013638
- macd_signal_z: 0.013433
- residual_percentile: 0.013399
- oh_XLF: 0.013328
- px_sma200_ratio_z: 0.013287
- predicted_return: 0.013237
- sect_target_minus_etf: 0.013089
- log_residual: 0.013075
- residual_rank: 0.013040

### Family importance
- **raw**: 0.0
- **percent**: 0.0
- **log**: 0.236818
- **Return**: 0.013237
- **Other**: 0.749945

### Fibonacci diagnostics
- **fib_23_hit**: hits=96, trades=21, avg_ret=0.03521363657279668, win_rate=1.0
- **fib_38_hit**: hits=67, trades=21, avg_ret=0.03521363657279668, win_rate=1.0
- **fib_50_hit**: hits=45, trades=21, avg_ret=0.03521363657279668, win_rate=1.0
- **fib_61_hit**: hits=19, trades=12, avg_ret=0.031792775536207483, win_rate=1.0
- **fib_78_hit**: hits=4, trades=2, avg_ret=0.02905647181233556, win_rate=1.0

### Residual diagnostics
- **log**: mean_z=0.2745, std_z=1.8111, pct>|1.25|=0.325

---

## Experiment: C_LOG_no_return

### Feature counts
- **Total features**: 95
- **Raw residual features**: 0
- **Percent residual features**: 0
- **Log residual features**: 28
- **Return features**: 0

### Classification diagnostics
- **Accuracy**: 0.9737
- **Macro F1**: 0.9726
- **Weighted F1**: 0.9734

Confusion matrix:
```
83 | 2 | 0
0 | 346 | 0
0 | 13 | 126
```

### Backtest metrics (PositionManager enabled)
- **cumulative_return**: 0.067127
- **annualized_return**: 0.332738
- **annualized_vol**: 0.030533
- **sharpe**: 10.897817
- **max_drawdown**: -0.003131
- **win_rate**: 0.869565
- **n_trades**: 35
- **avg_trade_return**: 0.014339
- **n_long**: 126
- **n_short**: 83

### Top 30 features
- oh_XLV: 0.024777
- log_residual_rank: 0.023636
- oh_XLK: 0.022680
- rule_obv_trend_pos: 0.022335
- oh_XLP: 0.020245
- rule_px_gt_sma20: 0.020095
- oh_XLB: 0.019438
- log_residual_ewm_mean: 0.019303
- rule_volume_spike: 0.017318
- oh_XLC: 0.017230
- residual_distance_from_zero: 0.016952
- oh_XLE: 0.015493
- residual_ewm_z: 0.015281
- px_ema200_ratio_z: 0.015052
- residual_abs_z: 0.014997
- shadow_price_gap_pct: 0.014859
- px_sma50_ratio_z: 0.014527
- log_residual_ewm_std: 0.014481
- rsi_z: 0.014451
- residual_sign: 0.014310
- log_residual: 0.014304
- px_ema50_ratio_z: 0.014120
- log_residual_abs_z: 0.014018
- residual_half_life_proxy: 0.013847
- px_sma200_ratio_z: 0.013820
- log_residual_regime_normal: 0.013643
- rule_near_52w_high: 0.013611
- sect_corr_etf: 0.013558
- macd_signal_z: 0.013205
- log_residual_distance_from_zero: 0.013202

### Family importance
- **raw**: 0.0
- **percent**: 0.0
- **log**: 0.259108
- **Return**: 0.0
- **Other**: 0.740892

### Fibonacci diagnostics
- **fib_23_hit**: hits=96, trades=18, avg_ret=0.036644544207922944, win_rate=1.0
- **fib_38_hit**: hits=67, trades=18, avg_ret=0.036644544207922944, win_rate=1.0
- **fib_50_hit**: hits=45, trades=18, avg_ret=0.036644544207922944, win_rate=1.0
- **fib_61_hit**: hits=19, trades=10, avg_ret=0.03164579640329861, win_rate=1.0
- **fib_78_hit**: hits=4, trades=2, avg_ret=0.02905647181233556, win_rate=1.0

### Residual diagnostics
- **log**: mean_z=0.2745, std_z=1.8111, pct>|1.25|=0.325

---

## Experiment: D_RAW_PERCENT_with_return

### Feature counts
- **Total features**: 124
- **Raw residual features**: 28
- **Percent residual features**: 28
- **Log residual features**: 0
- **Return features**: 1

### Classification diagnostics
- **Accuracy**: 0.9754
- **Macro F1**: 0.9744
- **Weighted F1**: 0.9752

Confusion matrix:
```
83 | 2 | 0
0 | 346 | 0
0 | 12 | 127
```

### Backtest metrics (PositionManager enabled)
- **cumulative_return**: 0.076012
- **annualized_return**: 0.382501
- **annualized_vol**: 0.031734
- **sharpe**: 12.053378
- **max_drawdown**: -0.003131
- **win_rate**: 0.862745
- **n_trades**: 43
- **avg_trade_return**: 0.014602
- **n_long**: 127
- **n_short**: 83

### Top 30 features
- oh_XLK: 0.025758
- raw_residual_z: 0.023826
- rule_macd_above_sig: 0.022411
- oh_XLP: 0.017491
- percent_residual_ewm_mean: 0.015694
- oh_XLC: 0.014950
- residual_percentile: 0.014571
- oh_XLB: 0.014100
- percent_fib_retrace_pct: 0.013965
- residual_distance_from_zero: 0.013883
- raw_residual_ewm_z: 0.013680
- raw_residual_regime_elevated: 0.013607
- raw_residual_ewm_std: 0.013559
- rule_px_gt_sma200: 0.013332
- residual_half_life_proxy: 0.013274
- residual_abs_z: 0.013257
- residual_ewm_z: 0.013195
- px_ema200_ratio_z: 0.013038
- px_ema50_ratio_z: 0.012514
- rule_sma20_gt_50: 0.012488
- shadow_price_gap_pct: 0.012487
- raw_residual_distance_from_zero: 0.012442
- raw_residual_ewm_mean: 0.012251
- rsi_z: 0.012161
- percent_residual_ewm_std: 0.012081
- sect_corr_etf: 0.011993
- px_sma50_ratio_z: 0.011941
- residual_rank: 0.011489
- sect_target_minus_etf: 0.011299
- predicted_return: 0.011165

### Family importance
- **raw**: 0.206561
- **percent**: 0.186144
- **log**: 0.0
- **Return**: 0.011165
- **Other**: 0.596131

### Fibonacci diagnostics
- **fib_23_hit**: hits=96, trades=22, avg_ret=0.033850190572033714, win_rate=1.0
- **fib_38_hit**: hits=67, trades=22, avg_ret=0.033850190572033714, win_rate=1.0
- **fib_50_hit**: hits=45, trades=22, avg_ret=0.033850190572033714, win_rate=1.0
- **fib_61_hit**: hits=19, trades=13, avg_ret=0.029748548537730863, win_rate=1.0
- **fib_78_hit**: hits=4, trades=3, avg_ret=0.021110256060227522, win_rate=1.0

### Residual diagnostics
- **raw**: mean_z=0.2950, std_z=1.8159, pct>|1.25|=0.339
- **percent**: mean_z=0.2865, std_z=1.8007, pct>|1.25|=0.323

---

## Experiment: D_RAW_PERCENT_no_return

### Feature counts
- **Total features**: 123
- **Raw residual features**: 28
- **Percent residual features**: 28
- **Log residual features**: 0
- **Return features**: 0

### Classification diagnostics
- **Accuracy**: 0.9719
- **Macro F1**: 0.9694
- **Weighted F1**: 0.9717

Confusion matrix:
```
81 | 4 | 0
0 | 346 | 0
0 | 12 | 127
```

### Backtest metrics (PositionManager enabled)
- **cumulative_return**: 0.07842
- **annualized_return**: 0.39623
- **annualized_vol**: 0.032065
- **sharpe**: 12.357266
- **max_drawdown**: -0.002627
- **win_rate**: 0.872727
- **n_trades**: 43
- **avg_trade_return**: 0.013947
- **n_long**: 127
- **n_short**: 81

### Top 30 features
- oh_XLK: 0.020088
- oh_XLP: 0.020026
- percent_residual_ewm_mean: 0.017393
- residual_excursion_bucket: 0.015564
- raw_residual_ewm_std: 0.015065
- px_ema200_ratio_z: 0.014646
- raw_residual_abs_z: 0.014640
- oh_XLB: 0.014362
- residual_distance_from_zero: 0.014175
- rule_sma50_gt_200: 0.014144
- rule_sma20_gt_50: 0.013658
- raw_residual_ewm_mean: 0.013010
- shadow_price_gap_pct: 0.012893
- raw_residual_regime_elevated: 0.012683
- rsi_z: 0.012673
- raw_residual: 0.012583
- px_sma50_ratio_z: 0.012503
- px_ema50_ratio_z: 0.012424
- sect_corr_etf: 0.012384
- residual_abs_z: 0.012149
- percent_fib_retrace_pct: 0.012130
- percent_residual_ewm_std: 0.012092
- percent_residual_distance_from_zero: 0.011787
- percent_days_above_2_sigma: 0.011772
- residual_ewm_z: 0.011645
- residual_half_life_proxy: 0.011641
- percent_residual_abs_z: 0.011459
- sect_vol_rel_etf: 0.011459
- px_sma200_ratio_z: 0.011422
- rule_px_gt_sma200: 0.011304

### Family importance
- **raw**: 0.191083
- **percent**: 0.198409
- **log**: 0.0
- **Return**: 0.0
- **Other**: 0.610508

### Fibonacci diagnostics
- **fib_23_hit**: hits=96, trades=22, avg_ret=0.03486855358311459, win_rate=1.0
- **fib_38_hit**: hits=67, trades=22, avg_ret=0.03486855358311459, win_rate=1.0
- **fib_50_hit**: hits=45, trades=22, avg_ret=0.03486855358311459, win_rate=1.0
- **fib_61_hit**: hits=19, trades=12, avg_ret=0.031792775536207483, win_rate=1.0
- **fib_78_hit**: hits=4, trades=2, avg_ret=0.02905647181233556, win_rate=1.0

### Residual diagnostics
- **raw**: mean_z=0.2950, std_z=1.8159, pct>|1.25|=0.339
- **percent**: mean_z=0.2865, std_z=1.8007, pct>|1.25|=0.323

---

## Experiment: E_RAW_LOG_with_return

### Feature counts
- **Total features**: 124
- **Raw residual features**: 28
- **Percent residual features**: 0
- **Log residual features**: 28
- **Return features**: 1

### Classification diagnostics
- **Accuracy**: 0.9789
- **Macro F1**: 0.9786
- **Weighted F1**: 0.9787

Confusion matrix:
```
84 | 1 | 0
0 | 346 | 0
0 | 11 | 128
```

### Backtest metrics (PositionManager enabled)
- **cumulative_return**: 0.076012
- **annualized_return**: 0.382501
- **annualized_vol**: 0.031734
- **sharpe**: 12.053378
- **max_drawdown**: -0.003131
- **win_rate**: 0.862745
- **n_trades**: 43
- **avg_trade_return**: 0.014602
- **n_long**: 128
- **n_short**: 84

### Top 30 features
- oh_XLK: 0.024166
- raw_residual_z: 0.023731
- rule_macd_above_sig: 0.019473
- oh_XLE: 0.017330
- oh_XLP: 0.017315
- log_residual_ewm_mean: 0.017235
- oh_XLB: 0.015481
- residual_distance_from_zero: 0.014597
- raw_residual_ewm_std: 0.014576
- px_ema200_ratio_z: 0.013494
- log_fib_retrace_pct: 0.013401
- rule_sma20_gt_50: 0.013145
- px_ema50_ratio_z: 0.013065
- residual_abs_z: 0.012990
- log_residual: 0.012806
- raw_residual_distance_from_zero: 0.012793
- rsi_z: 0.012404
- sect_corr_etf: 0.012073
- rule_obv_trend_pos: 0.011866
- shadow_price_gap_pct: 0.011780
- raw_residual: 0.011743
- px_sma50_ratio_z: 0.011743
- raw_residual_ewm_mean: 0.011727
- log_residual_ewm_std: 0.011624
- oh_XLF: 0.011589
- px_sma200_ratio_z: 0.011572
- residual_ewm_z: 0.011425
- residual_percentile: 0.011322
- px_ema20_ratio_z: 0.011299
- residual_rank: 0.011241

### Family importance
- **raw**: 0.19937
- **percent**: 0.0
- **log**: 0.188991
- **Return**: 0.010864
- **Other**: 0.600774

### Fibonacci diagnostics
- **fib_23_hit**: hits=96, trades=22, avg_ret=0.033850190572033714, win_rate=1.0
- **fib_38_hit**: hits=67, trades=22, avg_ret=0.033850190572033714, win_rate=1.0
- **fib_50_hit**: hits=45, trades=22, avg_ret=0.033850190572033714, win_rate=1.0
- **fib_61_hit**: hits=19, trades=13, avg_ret=0.029748548537730863, win_rate=1.0
- **fib_78_hit**: hits=4, trades=3, avg_ret=0.021110256060227522, win_rate=1.0

### Residual diagnostics
- **log**: mean_z=0.2745, std_z=1.8111, pct>|1.25|=0.325
- **raw**: mean_z=0.2950, std_z=1.8159, pct>|1.25|=0.339

---

## Experiment: E_RAW_LOG_no_return

### Feature counts
- **Total features**: 123
- **Raw residual features**: 28
- **Percent residual features**: 0
- **Log residual features**: 28
- **Return features**: 0

### Classification diagnostics
- **Accuracy**: 0.9772
- **Macro F1**: 0.9768
- **Weighted F1**: 0.9769

Confusion matrix:
```
84 | 1 | 0
0 | 346 | 0
0 | 12 | 127
```

### Backtest metrics (PositionManager enabled)
- **cumulative_return**: 0.072099
- **annualized_return**: 0.36041
- **annualized_vol**: 0.031453
- **sharpe**: 11.458512
- **max_drawdown**: -0.003131
- **win_rate**: 0.891304
- **n_trades**: 39
- **avg_trade_return**: 0.015374
- **n_long**: 127
- **n_short**: 84

### Top 30 features
- oh_XLK: 0.021375
- oh_XLB: 0.018624
- oh_XLP: 0.018315
- log_residual_ewm_mean: 0.017601
- rule_sma20_gt_50: 0.016023
- residual_excursion_bucket: 0.014416
- raw_residual_ewm_std: 0.013850
- raw_residual_abs_z: 0.013800
- residual_distance_from_zero: 0.013575
- px_ema200_ratio_z: 0.013568
- log_fib_50_hit: 0.012692
- raw_residual_ewm_mean: 0.012458
- sect_corr_etf: 0.012424
- oh_XLV: 0.012417
- raw_residual_distance_from_zero: 0.012406
- raw_residual: 0.012260
- shadow_price_gap_pct: 0.012213
- log_residual_regime_calm: 0.012161
- px_ema50_ratio_z: 0.012070
- rsi_z: 0.012047
- residual_abs_z: 0.012012
- residual_half_life_proxy: 0.011976
- px_sma50_ratio_z: 0.011898
- log_residual_ewm_std: 0.011726
- log_residual_distance_from_zero: 0.011687
- px_sma200_ratio_z: 0.011586
- raw_residual_z: 0.011512
- log_residual_abs_z: 0.011467
- rule_obv_trend_pos: 0.011374
- log_acceleration: 0.011338

### Family importance
- **raw**: 0.193607
- **percent**: 0.0
- **log**: 0.193945
- **Return**: 0.0
- **Other**: 0.612448

### Fibonacci diagnostics
- **fib_23_hit**: hits=96, trades=20, avg_ret=0.035360211854118175, win_rate=1.0
- **fib_38_hit**: hits=67, trades=20, avg_ret=0.035360211854118175, win_rate=1.0
- **fib_50_hit**: hits=45, trades=20, avg_ret=0.035360211854118175, win_rate=1.0
- **fib_61_hit**: hits=19, trades=11, avg_ret=0.03174828868073845, win_rate=1.0
- **fib_78_hit**: hits=4, trades=2, avg_ret=0.02905647181233556, win_rate=1.0

### Residual diagnostics
- **log**: mean_z=0.2745, std_z=1.8111, pct>|1.25|=0.325
- **raw**: mean_z=0.2950, std_z=1.8159, pct>|1.25|=0.339

---

## Experiment: F_PERCENT_LOG_with_return

### Feature counts
- **Total features**: 124
- **Raw residual features**: 0
- **Percent residual features**: 28
- **Log residual features**: 28
- **Return features**: 1

### Classification diagnostics
- **Accuracy**: 0.9754
- **Macro F1**: 0.9751
- **Weighted F1**: 0.9751

Confusion matrix:
```
84 | 1 | 0
0 | 346 | 0
0 | 13 | 126
```

### Backtest metrics (PositionManager enabled)
- **cumulative_return**: 0.074825
- **annualized_return**: 0.375768
- **annualized_vol**: 0.031416
- **sharpe**: 11.960888
- **max_drawdown**: -0.003131
- **win_rate**: 0.86
- **n_trades**: 41
- **avg_trade_return**: 0.014662
- **n_long**: 126
- **n_short**: 84

### Top 30 features
- rule_macd_above_sig: 0.029955
- oh_XLE: 0.025376
- oh_XLK: 0.024525
- log_residual_ewm_mean: 0.017620
- log_residual: 0.017455
- oh_XLP: 0.017411
- oh_XLB: 0.015990
- percent_residual_ewm_z: 0.015966
- rule_volume_spike: 0.014498
- px_ema200_ratio_z: 0.014473
- percent_residual_ewm_mean: 0.013743
- log_residual_regime_elevated: 0.013641
- residual_distance_from_zero: 0.013228
- oh_XLY: 0.012992
- px_ema50_ratio_z: 0.012877
- percent_residual_ewm_std: 0.012770
- residual_abs_z: 0.012612
- rsi_z: 0.012142
- sect_corr_etf: 0.011974
- shadow_price_gap_pct: 0.011966
- residual_half_life_proxy: 0.011828
- px_sma50_ratio_z: 0.011821
- macd_signal_z: 0.011680
- residual_rank: 0.011388
- log_residual_abs_z: 0.011310
- residual_ewm_z: 0.011305
- log_residual_distance_from_zero: 0.011105
- oh_XLF: 0.011038
- log_days_above_1_sigma: 0.010971
- rule_breakout_20: 0.010894

### Family importance
- **raw**: 0.0
- **percent**: 0.190642
- **log**: 0.183733
- **Return**: 0.010476
- **Other**: 0.615149

### Fibonacci diagnostics
- **fib_23_hit**: hits=96, trades=21, avg_ret=0.03491069988725114, win_rate=1.0
- **fib_38_hit**: hits=67, trades=21, avg_ret=0.03491069988725114, win_rate=1.0
- **fib_50_hit**: hits=45, trades=21, avg_ret=0.03491069988725114, win_rate=1.0
- **fib_61_hit**: hits=19, trades=12, avg_ret=0.03126263633650278, win_rate=1.0
- **fib_78_hit**: hits=4, trades=2, avg_ret=0.025875636614107356, win_rate=1.0

### Residual diagnostics
- **log**: mean_z=0.2745, std_z=1.8111, pct>|1.25|=0.325
- **percent**: mean_z=0.2865, std_z=1.8007, pct>|1.25|=0.323

---

## Experiment: F_PERCENT_LOG_no_return

### Feature counts
- **Total features**: 123
- **Raw residual features**: 0
- **Percent residual features**: 28
- **Log residual features**: 28
- **Return features**: 0

### Classification diagnostics
- **Accuracy**: 0.9737
- **Macro F1**: 0.9726
- **Weighted F1**: 0.9734

Confusion matrix:
```
83 | 2 | 0
0 | 346 | 0
0 | 13 | 126
```

### Backtest metrics (PositionManager enabled)
- **cumulative_return**: 0.072048
- **annualized_return**: 0.360125
- **annualized_vol**: 0.031221
- **sharpe**: 11.5348
- **max_drawdown**: -0.003131
- **win_rate**: 0.86
- **n_trades**: 39
- **avg_trade_return**: 0.014134
- **n_long**: 126
- **n_short**: 83

### Top 30 features
- oh_XLK: 0.023265
- log_residual_ewm_mean: 0.020809
- oh_XLE: 0.019030
- oh_XLP: 0.018096
- oh_XLB: 0.017032
- rule_macd_above_sig: 0.015379
- residual_excursion_bucket: 0.015315
- rule_near_52w_high: 0.014777
- px_ema200_ratio_z: 0.014542
- residual_distance_from_zero: 0.013437
- percent_residual_ewm_mean: 0.013289
- log_residual_distance_from_zero: 0.013172
- rsi_z: 0.012389
- oh_XLU: 0.012367
- residual_half_life_proxy: 0.012294
- log_residual_ewm_std: 0.012174
- rule_breakout_20: 0.012157
- log_residual_regime_calm: 0.012016
- shadow_price_gap_pct: 0.012004
- residual_abs_z: 0.011993
- rule_atr_high_regime: 0.011884
- px_ema50_ratio_z: 0.011863
- percent_residual: 0.011858
- px_sma50_ratio_z: 0.011823
- log_residual_abs_z: 0.011809
- px_sma200_ratio_z: 0.011710
- sect_corr_etf: 0.011706
- log_acceleration: 0.011491
- oh_XLF: 0.011365
- sect_target_minus_etf: 0.011284

### Family importance
- **raw**: 0.0
- **percent**: 0.189914
- **log**: 0.176806
- **Return**: 0.0
- **Other**: 0.63328

### Fibonacci diagnostics
- **fib_23_hit**: hits=96, trades=20, avg_ret=0.035335657828679666, win_rate=1.0
- **fib_38_hit**: hits=67, trades=20, avg_ret=0.035335657828679666, win_rate=1.0
- **fib_50_hit**: hits=45, trades=20, avg_ret=0.035335657828679666, win_rate=1.0
- **fib_61_hit**: hits=19, trades=11, avg_ret=0.031703644998123, win_rate=1.0
- **fib_78_hit**: hits=4, trades=2, avg_ret=0.02905647181233556, win_rate=1.0

### Residual diagnostics
- **log**: mean_z=0.2745, std_z=1.8111, pct>|1.25|=0.325
- **percent**: mean_z=0.2865, std_z=1.8007, pct>|1.25|=0.323

---

## Experiment: G_ALL_with_return

### Feature counts
- **Total features**: 152
- **Raw residual features**: 28
- **Percent residual features**: 28
- **Log residual features**: 28
- **Return features**: 1

### Classification diagnostics
- **Accuracy**: 0.9807
- **Macro F1**: 0.9804
- **Weighted F1**: 0.9805

Confusion matrix:
```
84 | 1 | 0
0 | 346 | 0
0 | 10 | 129
```

### Backtest metrics (PositionManager enabled)
- **cumulative_return**: 0.076012
- **annualized_return**: 0.382501
- **annualized_vol**: 0.031734
- **sharpe**: 12.053378
- **max_drawdown**: -0.003131
- **win_rate**: 0.862745
- **n_trades**: 43
- **avg_trade_return**: 0.014602
- **n_long**: 129
- **n_short**: 84

### Top 30 features
- log_residual: 0.024022
- oh_XLK: 0.019922
- log_residual_ewm_mean: 0.017613
- oh_XLP: 0.016107
- raw_residual_distance_from_zero: 0.015113
- percent_residual_ewm_mean: 0.013211
- raw_residual_z: 0.012870
- raw_residual_ewm_std: 0.012804
- residual_distance_from_zero: 0.012650
- oh_XLB: 0.012565
- raw_residual_rank: 0.012084
- percent_residual: 0.011962
- log_residual_regime_calm: 0.011943
- px_ema200_ratio_z: 0.011678
- raw_residual_ewm_mean: 0.011322
- percent_residual_ewm_z: 0.011201
- percent_residual_regime_elevated: 0.010996
- px_sma200_ratio_z: 0.010993
- residual_abs_z: 0.010936
- raw_residual: 0.010863
- rsi_z: 0.010840
- raw_residual_abs_z: 0.010789
- residual_ewm_z: 0.010734
- px_sma50_ratio_z: 0.010630
- log_acceleration: 0.010597
- shadow_price_gap_pct: 0.010577
- sect_corr_etf: 0.010483
- log_residual_ewm_std: 0.010454
- sect_target_minus_etf: 0.010448
- px_ema20_ratio_z: 0.010440

### Family importance
- **raw**: 0.188836
- **percent**: 0.164136
- **log**: 0.164425
- **Return**: 0.009762
- **Other**: 0.472841

### Fibonacci diagnostics
- **fib_23_hit**: hits=96, trades=22, avg_ret=0.033850190572033714, win_rate=1.0
- **fib_38_hit**: hits=67, trades=22, avg_ret=0.033850190572033714, win_rate=1.0
- **fib_50_hit**: hits=45, trades=22, avg_ret=0.033850190572033714, win_rate=1.0
- **fib_61_hit**: hits=19, trades=13, avg_ret=0.029748548537730863, win_rate=1.0
- **fib_78_hit**: hits=4, trades=3, avg_ret=0.021110256060227522, win_rate=1.0

### Residual diagnostics
- **log**: mean_z=0.2745, std_z=1.8111, pct>|1.25|=0.325
- **raw**: mean_z=0.2950, std_z=1.8159, pct>|1.25|=0.339
- **percent**: mean_z=0.2865, std_z=1.8007, pct>|1.25|=0.323

---

## Experiment: G_ALL_no_return

### Feature counts
- **Total features**: 151
- **Raw residual features**: 28
- **Percent residual features**: 28
- **Log residual features**: 28
- **Return features**: 0

### Classification diagnostics
- **Accuracy**: 0.9772
- **Macro F1**: 0.9762
- **Weighted F1**: 0.977

Confusion matrix:
```
83 | 2 | 0
0 | 346 | 0
0 | 11 | 128
```

### Backtest metrics (PositionManager enabled)
- **cumulative_return**: 0.069728
- **annualized_return**: 0.347158
- **annualized_vol**: 0.031636
- **sharpe**: 10.973347
- **max_drawdown**: -0.003131
- **win_rate**: 0.857143
- **n_trades**: 38
- **avg_trade_return**: 0.01397
- **n_long**: 128
- **n_short**: 83

### Top 30 features
- rule_px_gt_sma20: 0.027610
- percent_residual_rank: 0.025718
- oh_XLK: 0.019763
- log_residual_ewm_mean: 0.016742
- oh_XLP: 0.015997
- log_residual: 0.014452
- raw_residual_abs_z: 0.012569
- oh_XLB: 0.012356
- residual_distance_from_zero: 0.012289
- percent_residual_ewm_mean: 0.011720
- percent_fib_retrace_pct: 0.011691
- oh_XLE: 0.011604
- px_ema200_ratio_z: 0.011476
- log_residual_regime_normal: 0.011454
- percent_residual_ewm_std: 0.011240
- raw_residual_ewm_std: 0.011080
- residual_half_life_proxy: 0.011002
- oh_XLY: 0.010976
- log_residual_distance_from_zero: 0.010958
- log_residual_abs_z: 0.010944
- px_ema50_ratio_z: 0.010849
- oh_XLF: 0.010811
- raw_residual_distance_from_zero: 0.010792
- rule_sma50_gt_200: 0.010624
- residual_abs_z: 0.010619
- log_acceleration: 0.010510
- raw_residual_rank: 0.010490
- percent_distance_from_peak: 0.010357
- px_sma200_ratio_z: 0.010153
- rule_near_52w_high: 0.010118

### Family importance
- **raw**: 0.153965
- **percent**: 0.172578
- **log**: 0.154671
- **Return**: 0.0
- **Other**: 0.518786

### Fibonacci diagnostics
- **fib_23_hit**: hits=96, trades=20, avg_ret=0.03422741440729539, win_rate=1.0
- **fib_38_hit**: hits=67, trades=20, avg_ret=0.03422741440729539, win_rate=1.0
- **fib_50_hit**: hits=45, trades=20, avg_ret=0.03422741440729539, win_rate=1.0
- **fib_61_hit**: hits=19, trades=11, avg_ret=0.029688656959242485, win_rate=1.0
- **fib_78_hit**: hits=4, trades=3, avg_ret=0.022743402047784116, win_rate=1.0

### Residual diagnostics
- **log**: mean_z=0.2745, std_z=1.8111, pct>|1.25|=0.325
- **raw**: mean_z=0.2950, std_z=1.8159, pct>|1.25|=0.339
- **percent**: mean_z=0.2865, std_z=1.8007, pct>|1.25|=0.323

---

## Summary Tables

### Table 1: Residual Family Performance
| Experiment | Accuracy | Macro F1 | Sharpe | Cumulative Return | Max DD | Completed Trades | Long Entries | Short Entries |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A_RAW_with_return | 0.9737 | 0.9719 | 11.6469 | 0.0734 | -0.0031 | 39 | 127 | 82 |
| A_RAW_no_return | 0.9702 | 0.9676 | 11.7299 | 0.0725 | -0.0031 | 37 | 126 | 81 |
| B_PERCENT_with_return | 0.9737 | 0.9726 | 11.4220 | 0.0714 | -0.0031 | 39 | 126 | 83 |
| B_PERCENT_no_return | 0.9737 | 0.9726 | 10.9308 | 0.0687 | -0.0031 | 37 | 126 | 83 |
| C_LOG_with_return | 0.9772 | 0.9768 | 12.0763 | 0.0755 | -0.0031 | 41 | 127 | 84 |
| C_LOG_no_return | 0.9737 | 0.9726 | 10.8978 | 0.0671 | -0.0031 | 35 | 126 | 83 |
| D_RAW_PERCENT_with_return | 0.9754 | 0.9744 | 12.0534 | 0.0760 | -0.0031 | 43 | 127 | 83 |
| D_RAW_PERCENT_no_return | 0.9719 | 0.9694 | 12.3573 | 0.0784 | -0.0026 | 43 | 127 | 81 |
| E_RAW_LOG_with_return | 0.9789 | 0.9786 | 12.0534 | 0.0760 | -0.0031 | 43 | 128 | 84 |
| E_RAW_LOG_no_return | 0.9772 | 0.9768 | 11.4585 | 0.0721 | -0.0031 | 39 | 127 | 84 |
| F_PERCENT_LOG_with_return | 0.9754 | 0.9751 | 11.9609 | 0.0748 | -0.0031 | 41 | 126 | 84 |
| F_PERCENT_LOG_no_return | 0.9737 | 0.9726 | 11.5348 | 0.0720 | -0.0031 | 39 | 126 | 83 |
| G_ALL_with_return | 0.9807 | 0.9804 | 12.0534 | 0.0760 | -0.0031 | 43 | 129 | 84 |
| G_ALL_no_return | 0.9772 | 0.9762 | 10.9733 | 0.0697 | -0.0031 | 38 | 128 | 83 |
## Comparison Tables and Interpretation

### Table 2: Feature Importance by Family
| Experiment | Raw Importance | Percent Importance | Log Importance | Return Importance | Top Family |
|---|---:|---:|---:|---:|---|
| A_RAW_with_return | 0.2434 | 0.0000 | 0.0000 | 0.0123 | raw |
| A_RAW_no_return | 0.2487 | 0.0000 | 0.0000 | 0.0000 | raw |
| B_PERCENT_with_return | 0.0000 | 0.2389 | 0.0000 | 0.0135 | percent |
| B_PERCENT_no_return | 0.0000 | 0.2770 | 0.0000 | 0.0000 | percent |
| C_LOG_with_return | 0.0000 | 0.0000 | 0.2368 | 0.0132 | log |
| C_LOG_no_return | 0.0000 | 0.0000 | 0.2591 | 0.0000 | log |
| D_RAW_PERCENT_with_return | 0.2066 | 0.1861 | 0.0000 | 0.0112 | raw |
| D_RAW_PERCENT_no_return | 0.1911 | 0.1984 | 0.0000 | 0.0000 | percent |
| E_RAW_LOG_with_return | 0.1994 | 0.0000 | 0.1890 | 0.0109 | raw |
| E_RAW_LOG_no_return | 0.1936 | 0.0000 | 0.1939 | 0.0000 | log |
| F_PERCENT_LOG_with_return | 0.0000 | 0.1906 | 0.1837 | 0.0105 | percent |
| F_PERCENT_LOG_no_return | 0.0000 | 0.1899 | 0.1768 | 0.0000 | percent |
| G_ALL_with_return | 0.1888 | 0.1641 | 0.1644 | 0.0098 | raw |
| G_ALL_no_return | 0.1540 | 0.1726 | 0.1547 | 0.0000 | percent |

### Table 3: Fibonacci Usefulness
| Experiment | Most Useful Fib Level | Avg Return When Hit | Trade Count When Hit |
|---|---|---:|---:|
| A_RAW_with_return | fib_23_hit | 0.03594844870871553 | 20 |
| A_RAW_no_return | fib_23_hit | 0.03739481129927125 | 19 |
| B_PERCENT_with_return | fib_23_hit | 0.035017574308856846 | 20 |
| B_PERCENT_no_return | fib_23_hit | 0.03549636976985403 | 19 |
| C_LOG_with_return | fib_23_hit | 0.03521363657279668 | 21 |
| C_LOG_no_return | fib_23_hit | 0.036644544207922944 | 18 |
| D_RAW_PERCENT_with_return | fib_23_hit | 0.033850190572033714 | 22 |
| D_RAW_PERCENT_no_return | fib_23_hit | 0.03486855358311459 | 22 |
| E_RAW_LOG_with_return | fib_23_hit | 0.033850190572033714 | 22 |
| E_RAW_LOG_no_return | fib_23_hit | 0.035360211854118175 | 20 |
| F_PERCENT_LOG_with_return | fib_23_hit | 0.03491069988725114 | 21 |
| F_PERCENT_LOG_no_return | fib_23_hit | 0.035335657828679666 | 20 |
| G_ALL_with_return | fib_23_hit | 0.033850190572033714 | 22 |
| G_ALL_no_return | fib_23_hit | 0.03422741440729539 | 20 |

### Table 4: Concentration
| Experiment | Top Sector | Top Target | % PnL from Top Sector | % PnL from Top Target |
|---|---|---|---:|---:|
| A_RAW_with_return | Technology | AMD | 0.319 | 0.319 |
| A_RAW_no_return | Technology | AMD | 0.323 | 0.323 |
| B_PERCENT_with_return | Technology | AMD | 0.328 | 0.328 |
| B_PERCENT_no_return | Technology | AMD | 0.340 | 0.340 |
| C_LOG_with_return | Materials | FCX | 0.318 | 0.318 |
| C_LOG_no_return | Technology | AMD | 0.347 | 0.347 |
| D_RAW_PERCENT_with_return | Materials | FCX | 0.316 | 0.316 |
| D_RAW_PERCENT_no_return | Materials | FCX | 0.307 | 0.307 |
| E_RAW_LOG_with_return | Materials | FCX | 0.316 | 0.316 |
| E_RAW_LOG_no_return | Technology | AMD | 0.324 | 0.324 |
| F_PERCENT_LOG_with_return | Materials | FCX | 0.321 | 0.321 |
| F_PERCENT_LOG_no_return | Technology | AMD | 0.325 | 0.325 |
| G_ALL_with_return | Materials | FCX | 0.316 | 0.316 |
| G_ALL_no_return | Technology | AMD | 0.335 | 0.335 |

### Table 5: Risk
| Experiment | Max Drawdown | Avg Trade Return | Win Rate | Completed Trades |
|---|---:|---:|---:|---:|
| A_RAW_with_return | -0.0031 | 0.0147 | 0.8571 | 39 |
| A_RAW_no_return | -0.0031 | 0.0151 | 0.8723 | 37 |
| B_PERCENT_with_return | -0.0031 | 0.0137 | 0.8431 | 39 |
| B_PERCENT_no_return | -0.0031 | 0.0143 | 0.8723 | 37 |
| C_LOG_with_return | -0.0031 | 0.0151 | 0.8776 | 41 |
| C_LOG_no_return | -0.0031 | 0.0143 | 0.8696 | 35 |
| D_RAW_PERCENT_with_return | -0.0031 | 0.0146 | 0.8627 | 43 |
| D_RAW_PERCENT_no_return | -0.0026 | 0.0139 | 0.8727 | 43 |
| E_RAW_LOG_with_return | -0.0031 | 0.0146 | 0.8627 | 43 |
| E_RAW_LOG_no_return | -0.0031 | 0.0154 | 0.8913 | 39 |
| F_PERCENT_LOG_with_return | -0.0031 | 0.0147 | 0.8600 | 41 |
| F_PERCENT_LOG_no_return | -0.0031 | 0.0141 | 0.8600 | 39 |
| G_ALL_with_return | -0.0031 | 0.0146 | 0.8627 | 43 |
| G_ALL_no_return | -0.0031 | 0.0140 | 0.8571 | 38 |

## Interpretation (concise answers)
1. Strongest residual family alone: **C** (best Sharpe=12.076)
2. Percent vs Raw: percent_sharpe=11.422 vs raw_sharpe=11.647 -> raw better or similar
3. Log vs Raw: log_sharpe=12.076 vs raw_sharpe=11.647 -> log better
4. Combining families (ALL) Sharpe=12.053 -> does not improve much
5. RAW importance in ALL: 0.189; indicates useful
6. Fibonacci features appear to show positive signal; most useful level across experiments: fib_23_hit
8. Return features importance (mean across ex): 0.0058 -> less important
9. Residual engine size: feature counts range 95-152; may be large
10. Recommended config for validation: **C_LOG_with_return** or **E_RAW_LOG_with_return** (highest Sharpe / cumulative returns).