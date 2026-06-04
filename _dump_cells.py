import json
nb = json.load(open('DBTS_Train_Only_Diagnostic_FIXED.ipynb', encoding='utf-8'))
# Print cells 1, 5, 6, 7 in full (these set up cfg, data, panel, classifier, model_store)
for i in [1, 2, 3, 4, 5, 6, 15]:
    print(f"\n{'='*70}\n== CELL {i}\n{'='*70}")
    print(''.join(nb['cells'][i]['source']))
