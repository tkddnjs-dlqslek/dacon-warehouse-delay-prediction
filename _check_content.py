import json, sys
sys.stdout.reconfigure(encoding='utf-8')
nb_path = r'C:\Users\user\Desktop\데이콘 4월\code_share\solution.ipynb'
with open(nb_path, encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    # 실제로 old 값들 중 문제가 되는 것만 정확히 찾기
    old_patterns = [
        '~8.38', '~8.41', '~8.45',      # 구버전 OOF
        '~0.92 |', '~0.91 |', '~0.89 |',  # 구버전 corr (테이블 내)
        '< 0.95',                           # 구버전 설명 문장
    ]
    found = [p for p in old_patterns if p in src]
    if found:
        print(f'Cell {i} [{cell["cell_type"]}] 구버전 텍스트 잔존: {found}')
        for line in src.split('\n'):
            if any(p.strip() in line for p in found):
                print(f'  >> {repr(line)}')
    else:
        if 'oracle' in src and ('corr' in src.lower() or 'MAE' in src):
            print(f'Cell {i} [{cell["cell_type"]}] OK — 구버전 텍스트 없음')
