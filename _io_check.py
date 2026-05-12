import json, sys
sys.stdout.reconfigure(encoding='utf-8')
with open(r'C:\Users\user\Desktop\데이콘 4월\code_share\solution.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

io_patterns = ['open(', 'pickle.load', 'np.load("', "np.load('"]

print('=== 실제 파일 I/O 코드 검색 (주석 제외) ===')
found = False
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    src = ''.join(cell['source'])
    for line in src.split('\n'):
        stripped = line.strip()
        if stripped.startswith('#'):
            continue
        hits = [p for p in io_patterns if p in stripped]
        if hits:
            found = True
            print(f'Cell {i}: {hits}')
            print(f'  >> {stripped[:120]}')

if not found:
    print('없음 — 외부 파일을 여는 코드 없음')

print()
print('=== pd.read_csv 사용처 ===')
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    src = ''.join(cell['source'])
    for line in src.split('\n'):
        if 'pd.read_csv' in line and not line.strip().startswith('#'):
            print(f'Cell {i}: {line.strip()}')
