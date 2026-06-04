import json
nb = json.load(open('DBTS_Train_Only_Diagnostic_FIXED.ipynb', encoding='utf-8'))
for i, c in enumerate(nb['cells'][:20]):
    src = ''.join(c['source'])
    head = src.split('\n')[0] if src else '(empty)'
    print(f"[{i:2}] {c['cell_type']:8} | {head[:120]}")
