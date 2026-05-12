import json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

path = sys.argv[1]
nb = json.load(open(path, encoding='utf-8'))
print(f'TOTAL CELLS: {len(nb["cells"])}')
for i, c in enumerate(nb['cells']):
    src = ''.join(c['source'])
    print(f'\n===== CELL {i} ({c["cell_type"]}) len={len(src)} =====')
    print(src[:3000])
    if len(src) > 3000:
        print(f'... [TRUNC {len(src)-3000} more chars]')
