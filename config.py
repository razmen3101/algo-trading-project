import pandas as pd

SECTORS = {
    'XLB': {
        'name':       'Materials',
        'target':     'FCX',
        'predictors': ['SCCO', 'NEM', 'AA', 'CLF', 'NUE', 'VMC', 'MLM', 'ALB', 'SQM', 'TECK'],
    },
    'XLC': {
        'name':       'Communication',
        'target':     'META',
        'predictors': ['GOOGL', 'PINS', 'SNAP', 'TTD', 'NFLX', 'DIS', 'WBD', 'ROKU', 'SPOT', 'MTCH'],
    },
    'XLE': {
        'name':       'Energy',
        'target':     'XOM',
        'predictors': ['CVX', 'VLO', 'MPC', 'PSX', 'COP', 'SLB', 'HAL', 'BKR', 'WMB', 'KMI'],
    },
    'XLF': {
        'name':       'Financials',
        'target':     'JPM',
        'predictors': ['BAC', 'C', 'WFC', 'GS', 'MS', 'BLK', 'SCHW', 'V', 'MA', 'AXP'],
    },
    'XLK': {
        'name':       'Technology',
        'target':     'NVDA',
        'predictors': ['AMD', 'INTC', 'TSM', 'ASML', 'AVGO', 'MU', 'SMCI', 'ANET', 'ARM', 'ORCL'],
    },
    'XLP': {
        'name':       'Consumer Staples',
        'target':     'PG',
        'predictors': ['WMT', 'TGT', 'COST', 'KO', 'PEP', 'CL', 'KMB', 'HSY', 'MDLZ', 'PM'],
    },
    'XLRE': {
        'name':       'Real Estate',
        'target':     'PLD',
        'predictors': ['AMT', 'CCI', 'EQIX', 'DLR', 'O', 'SPG', 'PSA', 'AVB', 'EQR', 'ARE'],
    },
    'XLU': {
        'name':       'Utilities',
        'target':     'NEE',
        'predictors': ['DUK', 'SO', 'D', 'AEP', 'SRE', 'EXC', 'XEL', 'ED', 'AWK', 'PEG'],
    },
    'XLV': {
        'name':       'Health Care',
        'target':     'UNH',
        'predictors': ['ELV', 'HUM', 'CVS', 'CI', 'JNJ', 'PFE', 'MRK', 'ABBV', 'TMO', 'DHR'],
    },
    'XLY': {
        'name':       'Consumer Disc.',
        'target':     'AMZN',
        'predictors': ['HD', 'LOW', 'NKE', 'LULU', 'SBUX', 'MCD', 'BKNG', 'EXPE', 'MAR', 'TSLA'],
    },
}

# Flat lists derived from SECTORS
ALL_TARGETS    = [v['target']              for v in SECTORS.values()]
ALL_PREDICTORS = [t for v in SECTORS.values() for t in v['predictors']]
ALL_TICKERS    = sorted(set(ALL_TARGETS + ALL_PREDICTORS))

DEFAULT_START    = '2022-01-01'
DEFAULT_END      = None
DEFAULT_INTERVAL = '1d'

OUTLIER_THRESHOLD = 0.15
MAX_FILL_DAYS     = 3
