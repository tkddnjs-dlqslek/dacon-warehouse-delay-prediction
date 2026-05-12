import json, sys
sys.stdout.reconfigure(encoding='utf-8')
nb_path = r'C:\Users\user\Desktop\데이콘 4월\code_share\solution.ipynb'
with open(nb_path, encoding='utf-8') as f:
    nb = json.load(f)

OLD = '세 oracle 모두 `corr(FIXED) < 0.95` → FIXED와 블렌드 시 다양성 기여.'
NEW = '세 oracle 모두 `corr(FIXED) ≈ 0.97` — FIXED 대비 다양성은 제한적이나 oracle 간 corr ≈ 0.90~0.92로 상호 보완.'

fixed = False
for cell in nb['cells']:
    if cell['cell_type'] == 'markdown':
        src = ''.join(cell['source'])
        if OLD in src:
            cell['source'] = [src.replace(OLD, NEW)]
            fixed = True
            print('수정 완료')
            break

if not fixed:
    print('OLD 문자열 없음 — 이미 수정됐거나 다른 텍스트')
    for cell in nb['cells']:
        if cell['cell_type'] == 'markdown':
            src = ''.join(cell['source'])
            if 'corr(FIXED)' in src:
                for line in src.split('\n'):
                    if 'corr(FIXED)' in line and ('0.95' in line or '0.97' in line):
                        print(' ', repr(line))

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print('저장 완료')
