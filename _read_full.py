import json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

path = sys.argv[1]
out_prefix = sys.argv[2]
nb = json.load(open(path, encoding='utf-8'))
total = 0
for i, c in enumerate(nb['cells']):
    src = ''.join(c['source'])
    total += len(src)
    open(f'{out_prefix}_cell{i:03d}.txt', 'w', encoding='utf-8').write(src)
print(f'wrote {len(nb["cells"])} cells, total {total} chars')
