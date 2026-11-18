import json
from pathlib import Path

nb = json.loads(Path('AlphaScan_Plan2A_Colab.ipynb').read_text(encoding='utf-8'))
print(f"Total cells: {len(nb['cells'])}")
for i, c in enumerate(nb['cells']):
    src = ''.join(c.get('source', []))
    lines = [l.strip() for l in src.split('\n') if l.strip()]
    first_line = lines[0] if lines else "EMPTY"
    print(f"Cell {i} [{c.get('cell_type')}]: {first_line[:80]}")
