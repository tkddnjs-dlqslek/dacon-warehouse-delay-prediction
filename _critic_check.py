"""최종 critic 검증 스크립트"""
import json, sys, os, re
sys.stdout.reconfigure(encoding='utf-8')

nb_path = r'C:\Users\user\Desktop\데이콘 4월\code_share\solution.ipynb'
proj_dir = r'C:\Users\user\Desktop\데이콘 4월'

with open(nb_path, encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']
PASS = []
FAIL = []

# ─── 구조 ─────────────────────────────────────────────────────
n_cells = len(cells)
n_code = sum(1 for c in cells if c['cell_type'] == 'code')
n_md   = sum(1 for c in cells if c['cell_type'] == 'markdown')
PASS.append(f"셀 구성: 총 {n_cells}개 (코드 {n_code}, 마크다운 {n_md})")

# ─── 코드셀 출력 수집 ─────────────────────────────────────────
outputs = {}
for i, cell in enumerate(cells):
    if cell['cell_type'] == 'code':
        outs = cell.get('outputs', [])
        text = ''
        has_error = False
        for o in outs:
            if 'text' in o:
                text += ''.join(o['text'])
            if o.get('output_type') == 'error':
                has_error = True
                text += 'ERROR:' + o.get('ename','') + ':' + o.get('evalue','')
        outputs[i] = (text, has_error)

# ─── 1. 에러 없는지 ────────────────────────────────────────────
any_error = any(has_err for _, has_err in outputs.values())
if any_error:
    for i, (txt, has_err) in outputs.items():
        if has_err:
            FAIL.append(f"Cell {i} 실행 에러: {txt[:100]}")
else:
    PASS.append("모든 코드셀 에러 없이 실행 완료")

# ─── 2. CSV 파일명 ─────────────────────────────────────────────
csv_name = 'submission_oracle_NEW_OOF8.3825.csv'
found_fname = any(csv_name in txt for _, (txt, _) in outputs.items())
if found_fname:
    PASS.append(f"CSV 파일명 정확: {csv_name}")
else:
    FAIL.append(f"CSV 파일명 '{csv_name}' 출력에서 미확인")

csv_path = os.path.join(proj_dir, csv_name)
if os.path.exists(csv_path):
    PASS.append(f"CSV 파일 존재: {csv_path}")
else:
    FAIL.append(f"CSV 파일 없음: {csv_path}")

# ─── 3. OOF MAE ────────────────────────────────────────────────
all_output = "\n".join(txt for txt, _ in outputs.values())
m = re.search(r'OOF MAE: ([\d.]+)', all_output)
if m:
    mae_val = float(m.group(1))
    if abs(mae_val - 8.382467) < 1e-4:
        PASS.append(f"OOF MAE 정확: {mae_val:.6f}")
    else:
        FAIL.append(f"OOF MAE 불일치: {mae_val:.6f} (기대: 8.382467)")
else:
    FAIL.append(f"OOF MAE 파싱 실패 (모든 출력 검색)")

# ─── 4. 검증셀 출력 ────────────────────────────────────────────
checks = [
    ('50000', '행 수 50000'),
    ('NaN 개수  : 0', 'NaN 없음'),
    ('음수 개수 : 0', '음수 없음'),
    ('ID 순서 일치: True', 'ID 순서 일치'),
    ('검증 완료', '검증 완료'),
]
for key, label in checks:
    if key in all_output:
        PASS.append(label)
    else:
        FAIL.append(f"검증 실패: {label}")

# ─── 5. corr 수정 확인 ──────────────────────────────────────────
corr_pass = True
for cell in cells:
    src = ''.join(cell['source'])
    old_patterns = ['~8.38', '~8.41', '~8.45', '~0.92 |', '~0.91 |', '~0.89 |', '< 0.95']
    found = [p for p in old_patterns if p in src]
    if found:
        corr_pass = False
        FAIL.append(f"구버전 값 잔존 {found}: {repr(src[:80])}")
if corr_pass:
    PASS.append("corr 테이블 수정 완료 (0.9733/0.9721/0.9760)")

# ─── 6. 상대 경로 확인 ─────────────────────────────────────────
for cell in cells:
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source'])
        abs_paths = re.findall(r'[A-Z]:\\[^\'"]+', src)
        if abs_paths:
            FAIL.append(f"절대 경로 사용: {abs_paths[:2]}")
if not any('절대 경로' in f for f in FAIL):
    PASS.append("모든 파일 경로 상대 경로 사용")

# ─── 7. 입력 파일 목록 vs 실제 존재 ───────────────────────────
required_files = [
    'train.csv', 'test.csv', 'sample_submission.csv',
    'results/mega33_final.pkl',
    'results/ranking/rank_adj_oof.npy',
    'results/ranking/rank_adj_test.npy',
    'results/iter_pseudo/round1_oof.npy', 'results/iter_pseudo/round1_test.npy',
    'results/iter_pseudo/round2_oof.npy', 'results/iter_pseudo/round2_test.npy',
    'results/iter_pseudo/round3_oof.npy', 'results/iter_pseudo/round3_test.npy',
    'results/oracle_seq/oof_seqC_xgb.npy',       'results/oracle_seq/test_C_xgb.npy',
    'results/oracle_seq/oof_seqC_log_v2.npy',     'results/oracle_seq/test_C_log_v2.npy',
    'results/oracle_seq/oof_seqC_xgb_remaining.npy', 'results/oracle_seq/test_C_xgb_remaining.npy',
]
missing = [f for f in required_files if not os.path.exists(os.path.join(proj_dir, f))]
if missing:
    FAIL.append(f"필수 파일 누락: {missing}")
else:
    PASS.append(f"필수 입력 파일 {len(required_files)}개 전부 존재")

# ─── 8. 배열 복원 방식 확인 (embed 또는 id2 인덱싱) ────────────
embed_ok = any('base64' in ''.join(c['source']) and '_b64dec' in ''.join(c['source'])
               for c in cells if c['cell_type'] == 'code')
idx_ok   = any('id2' in ''.join(c['source']) and 'meta_avg_oof' in ''.join(c['source'])
               for c in cells if c['cell_type'] == 'code')
if embed_ok:
    PASS.append("base64 내장 배열 복원 방식 확인 (self-contained)")
elif idx_ok:
    PASS.append("id2/te_id2 인덱싱 패턴 존재 (ls→rid 변환)")
else:
    FAIL.append("배열 로드/복원 패턴 미확인")

# ─── 9. np.maximum(0) 클리핑 ─────────────────────────────────
clip_ok = any('np.maximum(0' in ''.join(c['source']) for c in cells if c['cell_type']=='code')
if clip_ok:
    PASS.append("np.maximum(0,...) 음수 클리핑 적용")
else:
    FAIL.append("np.maximum(0,...) 클리핑 없음")

# ─── 10. 가중치 합 ─────────────────────────────────────────────
fw_sum_ok = any('가중치 합계: 1.00' in txt for txt, _ in outputs.values())
if fw_sum_ok:
    PASS.append("oracle_NEW 블렌드 가중치 합 = 1.00")
else:
    FAIL.append("가중치 합 1.00 미확인")

# ─── 결과 출력 ────────────────────────────────────────────────
print("=" * 60)
print("CRITIC 최종 검증 리포트")
print("=" * 60)
print(f"\n✅ PASS ({len(PASS)}개):")
for p in PASS:
    print(f"  ✓ {p}")

if FAIL:
    print(f"\n❌ FAIL ({len(FAIL)}개):")
    for f in FAIL:
        print(f"  ✗ {f}")
    print("\n판정: 제출 불가 — 위 항목 수정 필요")
else:
    print(f"\n판정: ✅ 제출 가능")
print("=" * 60)
