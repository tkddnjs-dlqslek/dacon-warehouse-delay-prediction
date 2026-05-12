"""제출 파일 현황 및 ref3-only 후보 요약."""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import glob, re, os

# 모든 submission 파일 OOF 파싱
files = glob.glob('submission_*.csv')
results = []
for f in files:
    m = re.search(r'OOF(\d+\.\d+)', f)
    if m:
        oof = float(m.group(1))
        size = os.path.getsize(f)
        results.append((oof, f, size))

results.sort()

# ref3-safe 컴포넌트만 사용하는 파일 패턴
REF3_PATTERNS = ['cascade_refined3', 'R3A', 'R3B', 'R3C', 'R3D', 'R3E', 'R3F', 'R3G', 'R3H',
                 'CC', 'DD', 'EE', 'FF', 'GG', 'HH', 'II', 'Q_no_rankadj', 'A_refined3',
                 'P_mega33only', 'T_ref3_exact', 'U_nogate', 'R_no_gate']

def is_ref3_safe(fname):
    return any(p in fname for p in REF3_PATTERNS)

print(f"{'='*70}")
print(f"전체 submission 파일: {len(results)}개")
print(f"{'='*70}")

ref3_files = [(oof, f) for oof, f, s in results if is_ref3_safe(f)]
other_files = [(oof, f) for oof, f, s in results if not is_ref3_safe(f)]

print(f"\n[Ref3-safe 파일: {len(ref3_files)}개]")
print(f"{'OOF':>10}  {'파일명'}")
print(f"{'-'*70}")
for oof, f in ref3_files:
    marker = ' ← BEST LB' if 'cascade_refined3' in f else ''
    print(f"  {oof:.5f}   {f}{marker}")

print(f"\n[기타 파일 (LB 위험): {len(other_files)}개]")
for oof, f in other_files[:10]:
    print(f"  {oof:.5f}   {f}")
if len(other_files) > 10:
    print(f"  ... (총 {len(other_files)}개)")

print(f"\n{'='*70}")
print("오늘 제출 추천 순서 (ref3-only, OOF 기준):")
for i, (oof, f) in enumerate(ref3_files[:5], 1):
    print(f"  {i}. {f}  (OOF={oof:.5f})")
