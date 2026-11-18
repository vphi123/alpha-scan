import json
from pathlib import Path

nb = json.loads(Path('AlphaScan_Plan2A_Colab.ipynb').read_text(encoding='utf-8'))
for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'code':
        print(f"==================== CODE CELL {i} ====================")
        print(''.join(c.get('source', [])))
