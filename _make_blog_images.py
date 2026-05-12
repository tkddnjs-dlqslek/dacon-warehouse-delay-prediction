"""블로그용 이미지 4개 생성 (v2 — 라벨 겹침 수정)"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import rcParams

rcParams['font.family'] = 'Malgun Gothic'
rcParams['axes.unicode_minus'] = False

OUT = r'C:\Users\user\Desktop\데이콘 4월\blog_images'

# ────────────────────────────────────────────────────────────
# 이미지 1: LB 진화 라인 차트
# ────────────────────────────────────────────────────────────
print("[1/4] LB 진화 라인 차트…")

milestones = [
    # (name, date, lb, big_jump, label_offset_y, label_dx)
    ("v4 baseline",      "4/3", 10.86, False,  0.05, 0),
    ("v5 FE+앙상블",     "4/3", 10.26, False,  -0.07, 0),
    ("v8 GroupKFold",    "4/4", 10.23, False,   0.07, 0),
    ("v14 log target",   "4/5", 10.17, False,  -0.07, 0),
    ("v23 SC features",  "4/8", 9.952, True,    0.10, 0),
    ("mega20 stacking",  "4/12", 9.875, False, -0.10, 0),
    ("mega27 offset",    "4/14", 9.807, False,  0.07, 0),
    ("mega33 + neural",  "4/16", 9.7759, False, -0.10, 0),
    ("+ Ranking loss",   "4/18", 9.7711, True,  0.13, -0.3),
    ("oracle_NEW 최종",  "5/3",  9.7527, True,  -0.10, 0),
]

x = list(range(len(milestones)))
y = [m[2] for m in milestones]

fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(x, y, color='#2c3e50', linewidth=2, zorder=2)

# 점 그리기
for i, (name, date, lb, big, dy, dx) in enumerate(milestones):
    if big:
        ax.plot(i, lb, marker='*', color='#e74c3c', markersize=24,
                markeredgecolor='white', markeredgewidth=1.5, zorder=5)
    else:
        ax.plot(i, lb, marker='o', color='#2c3e50', markersize=10, zorder=4)

# 라벨 — 별 위치에 따라 분기
for i, (name, date, lb, big, dy, dx) in enumerate(milestones):
    color = '#c0392b' if big else '#34495e'
    weight = 'bold' if big else 'normal'
    ax.annotate(f"{name}\n({date}, LB {lb:.4f})",
                xy=(i + dx, lb + dy),
                ha='center', fontsize=9.5,
                color=color, fontweight=weight)

ax.set_xticks([])
ax.set_xlabel("실험 진행 →", fontsize=12)
ax.set_ylabel("Public LB MAE  (낮을수록 좋음 ↓)", fontsize=12)
ax.set_title("LB 점수 진화 — 한 달간의 점프", fontsize=15, fontweight='bold', pad=18)
ax.grid(True, axis='y', linestyle='--', alpha=0.4)
ax.invert_yaxis()
ax.set_ylim(11.05, 9.55)
ax.set_xlim(-0.5, len(milestones) - 0.5)

# 범례
ax.plot([], [], marker='*', color='#e74c3c', markersize=18, linestyle='',
        markeredgecolor='white', markeredgewidth=1.5, label='큰 점프 (★)')
ax.plot([], [], marker='o', color='#2c3e50', markersize=10, linestyle='', label='일반 진전')
ax.legend(loc='lower left', fontsize=11, framealpha=0.95)

plt.tight_layout()
plt.savefig(f"{OUT}/01_lb_timeline.png", dpi=150, bbox_inches='tight')
plt.close()
print("   저장 완료")

# ────────────────────────────────────────────────────────────
# 이미지 2: Sweet spot 산점도 (라벨 정리판)
# ────────────────────────────────────────────────────────────
print("[2/4] Sweet spot 산점도…")

# 분리: 클러스터 영역의 회색 점들은 작게 표시 + 대표 1개만 라벨
# 강조 포인트만 라벨

cluster_high_corr = [
    ("Cumsum",           0.991, 8.62),
    ("V_ORTH",           0.966, 8.83),
    ("Retrieval kNN",    0.978, 8.64),
    ("FFT 독립",         0.967, 8.81),
    ("Queuing theory",   0.980, 8.61),
    ("Tweedie 1.7",      0.971, 8.75),
    ("Q60 quantile",     0.973, 8.74),
    ("9-quantile meta",  0.977, 8.52),
    ("Error-driven",     0.989, 8.56),
]

low_quality = [
    ("Q90 quantile",     0.663, 14.66),
    ("Q80 quantile",     0.836, 10.85),
    ("Q70 quantile",     0.930, 9.35),
    ("Survival Cox",     0.565, 28.70),
    ("Layout kNN",       0.917, 10.67),
    ("residual ranking", 0.910, 9.59),
]

fig, ax = plt.subplots(figsize=(12, 7))

# Sweet spot 박스 (먼저)
sweet = patches.Rectangle((0.93, 7.98), 0.04, 8.40 * 1.05 - 7.98,
                           linewidth=2.5, edgecolor='#27ae60',
                           facecolor='#27ae60', alpha=0.13,
                           zorder=1, linestyle='--')
ax.add_patch(sweet)

# 가이드 라인
ax.axhline(8.40, color='#3498db', linestyle=':', linewidth=1.5, alpha=0.7,
           label='mega33 baseline MAE (8.40)', zorder=2)
ax.axhline(8.40 * 1.05, color='#3498db', linestyle=':', linewidth=0.8, alpha=0.4, zorder=2)
ax.axvline(0.95, color='#e74c3c', linestyle='--', linewidth=1.2, alpha=0.5,
           label='corr 0.95 벽', zorder=2)

# 회색 클러스터 — 점은 모두, 라벨은 1개만
for i, (name, corr, mae) in enumerate(cluster_high_corr):
    ax.scatter(corr, mae, c='#95a5a6', marker='o', s=110,
               edgecolors='white', linewidths=0.8, zorder=3, alpha=0.85)
ax.annotate("9개 실험 클러스터\n(corr 0.97~0.99,\nMAE 8.5~8.85)",
            xy=(0.985, 8.7), xytext=(0.88, 13),
            fontsize=9.5, color='#555', ha='center',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#ecf0f1',
                      edgecolor='#bdc3c7', alpha=0.9),
            arrowprops=dict(arrowstyle='->', color='#7f8c8d', lw=1))

# 주황색 (quality 부족) — 모두 라벨
label_offsets = {
    "Q90 quantile":     (0.025, 0.5),
    "Q80 quantile":     (-0.10, 0.6),
    "Q70 quantile":     (0.013, 0.4),
    "Survival Cox":     (0.025, 0.5),
    "Layout kNN":       (0.013, -0.5),
    "residual ranking": (-0.135, 0.0),
}
for name, corr, mae in low_quality:
    ax.scatter(corr, mae, c='#e67e22', marker='s', s=130,
               edgecolors='white', linewidths=0.8, zorder=3)
    dx, dy = label_offsets.get(name, (0.012, 0.3))
    ax.annotate(name, xy=(corr, mae), xytext=(corr + dx, mae + dy),
                fontsize=9.5, color='#a04000', fontweight='bold')

# Ranking 별 ★
ax.scatter(0.977, 8.50, c='#e74c3c', marker='*', s=600,
           edgecolors='black', linewidths=2, zorder=10)
ax.annotate("Ranking ★\n(corr 0.977, MAE 8.50,\nblend Δ=-0.005)",
            xy=(0.977, 8.50), xytext=(0.65, 8.7),
            fontsize=11, color='#c0392b', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#fff5f5',
                      edgecolor='#c0392b', linewidth=2),
            arrowprops=dict(arrowstyle='->', color='#c0392b', lw=2))

# Sweet spot 텍스트 (박스 밖 좌측)
ax.text(0.948, 7.55, 'Sweet Spot',
        fontsize=11, color='#27ae60', fontweight='bold')

ax.set_xlabel("residual corr (vs mega33) — 낮을수록 직교 →", fontsize=12)
ax.set_ylabel("single OOF MAE — 낮을수록 품질 ↑", fontsize=12)
ax.set_title("앙상블 다양성 — corr↓ AND quality↑를 동시 만족하는 곳이 답",
             fontsize=13.5, fontweight='bold', pad=15)
ax.set_xlim(0.48, 1.005)
ax.set_ylim(7.0, 31)
ax.grid(True, linestyle='--', alpha=0.3)

ax.legend(loc='upper right', fontsize=10, framealpha=0.95)

plt.tight_layout()
plt.savefig(f"{OUT}/02_sweet_spot.png", dpi=150, bbox_inches='tight')
plt.close()
print("   저장 완료")

# ────────────────────────────────────────────────────────────
# 이미지 3: 타겟 분포 — 그대로 유지
# ────────────────────────────────────────────────────────────
print("[3/4] 타겟 분포 히스토그램…")

train = pd.read_csv(r'C:\Users\user\Desktop\데이콘 4월\train.csv')
y = train['avg_delay_minutes_next_30m'].values

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

ax1 = axes[0]
ax1.hist(y[y < 100], bins=80, color='#3498db', edgecolor='white', linewidth=0.5)
zero_count = int((y == 0).sum())
ax1.axvline(0, color='#e74c3c', linestyle='-', linewidth=2.5, alpha=0.8,
            label=f'y=0 spike ({zero_count:,}행, {zero_count/len(y)*100:.2f}%)')
ax1.axvline(np.median(y), color='#27ae60', linestyle='--', linewidth=2,
            label=f'median = {np.median(y):.2f}')
ax1.axvline(np.mean(y), color='#e67e22', linestyle='--', linewidth=2,
            label=f'mean = {np.mean(y):.2f}')
ax1.set_xlabel("avg_delay_minutes_next_30m (분)", fontsize=11)
ax1.set_ylabel("빈도", fontsize=11)
ax1.set_title("타겟 분포 (0~100 구간) — 우편향 + y=0 spike",
              fontsize=12, fontweight='bold')
ax1.legend(loc='upper right', fontsize=10)
ax1.grid(True, axis='y', linestyle='--', alpha=0.3)

ax2 = axes[1]
ax2.hist(y, bins=100, color='#9b59b6', edgecolor='white', linewidth=0.5)
ax2.set_yscale('log')
ax2.axvline(50, color='#e74c3c', linestyle=':', linewidth=2, alpha=0.7,
            label=f'y=50 (전체 5%, error의 30%)')
ax2.set_xlabel("avg_delay_minutes_next_30m (분)", fontsize=11)
ax2.set_ylabel("빈도 (log scale)", fontsize=11)
ax2.set_title(f"전체 분포 (max={y.max():.1f}) — heavy tail",
              fontsize=12, fontweight='bold')
ax2.legend(loc='upper right', fontsize=10)
ax2.grid(True, axis='y', linestyle='--', alpha=0.3)

plt.suptitle(f"타겟 분포: mean={np.mean(y):.2f}, median={np.median(y):.2f}, std={np.std(y):.2f}, max={y.max():.2f}",
             fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(f"{OUT}/03_target_dist.png", dpi=150, bbox_inches='tight')
plt.close()
print("   저장 완료")

# ────────────────────────────────────────────────────────────
# 이미지 4: mega33 스태킹 구조도 (범례 위치 수정)
# ────────────────────────────────────────────────────────────
print("[4/4] mega33 스태킹 구조도…")

fig, ax = plt.subplots(figsize=(13, 9))
ax.set_xlim(0, 100)
ax.set_ylim(0, 105)
ax.axis('off')

COL_GBDT   = '#3498db'
COL_NEURAL = '#e74c3c'
COL_DOMAIN = '#27ae60'
COL_OFFSET = '#9b59b6'
COL_META   = '#f39c12'
COL_FINAL  = '#2c3e50'

# 타이틀 (위)
ax.text(50, 102, "mega33 스태킹 구조 (3-level)",
        fontsize=18, ha='center', fontweight='bold')

# 범례 (타이틀 아래, 따로 한 줄로)
legend_items = [
    (COL_GBDT, "GBDT (17개)"),
    (COL_NEURAL, "Neural (10개)"),
    (COL_DOMAIN, "Domain (3개)"),
    (COL_OFFSET, "Offset (3개)"),
]
legend_x_starts = [10, 32, 54, 76]
for (color, text), x_start in zip(legend_items, legend_x_starts):
    ax.add_patch(patches.Rectangle((x_start, 95.5), 2.5, 2.2, facecolor=color))
    ax.text(x_start + 3.5, 96.6, text, fontsize=10, va='center')

# Level 1 라벨
ax.text(50, 90, "Level 1: 33 base models (각자 5-fold GroupKFold OOF)",
        fontsize=12.5, ha='center', fontweight='bold', color='#555')

# Base 그룹들 (Y 좌표 조정)
groups = [
    ("9 × v23 GBDT\n(LGB_Huber/XGB/CB × 3 seeds)", 12, 78, COL_GBDT),
    ("4 × v24 GBDT\n(cumsum 변형)",                 32, 78, COL_GBDT),
    ("4 × v26 variants\n(Tuned/sqrt/pow/DART)",     52, 78, COL_GBDT),
    ("3 × MLP/CNN",                                  72, 78, COL_NEURAL),
    ("3 × domain-specific",                          88, 78, COL_DOMAIN),
    ("1 × mlp_aug",                                  12, 63, COL_NEURAL),
    ("3 × offset decomposition\n(scenario_mean + row_offset)", 32, 63, COL_OFFSET),
    ("6 × neural army\n(BiLSTM/DeepCNN/TCN/MLP)",   60, 63, COL_NEURAL),
]
for text, cx, cy, color in groups:
    box = patches.FancyBboxPatch((cx-9, cy-3.5), 18, 7,
                                  boxstyle="round,pad=0.3",
                                  facecolor=color, edgecolor='white',
                                  linewidth=2, alpha=0.88)
    ax.add_patch(box)
    ax.text(cx, cy, text, fontsize=8.7, ha='center', va='center',
            color='white', fontweight='bold')

# 화살표 down
for cx in [12, 32, 52, 72, 88]:
    ax.annotate('', xy=(cx, 53), xytext=(cx, 59),
                arrowprops=dict(arrowstyle='->', color='#7f8c8d', lw=1.5))

# Level 2 라벨
ax.text(50, 48, "Level 2: 3 meta-learners (입력 = 33 base OOF, log space)",
        fontsize=12.5, ha='center', fontweight='bold', color='#555')

metas = [("LGB meta", 25), ("XGB meta", 50), ("CatBoost meta", 75)]
for name, cx in metas:
    box = patches.FancyBboxPatch((cx-10, 37), 20, 7.5,
                                  boxstyle="round,pad=0.3",
                                  facecolor=COL_META, edgecolor='white',
                                  linewidth=2, alpha=0.92)
    ax.add_patch(box)
    ax.text(cx, 40.7, name, fontsize=11.5, ha='center', va='center',
            color='white', fontweight='bold')

# 화살표 to meta_avg
for cx in [25, 50, 75]:
    ax.annotate('', xy=(50, 25), xytext=(cx, 36.5),
                arrowprops=dict(arrowstyle='->', color='#7f8c8d', lw=1.5))

# Level 3
ax.text(50, 31, "Level 3", fontsize=12.5, ha='center',
        fontweight='bold', color='#555')

box = patches.FancyBboxPatch((30, 14), 40, 9.5,
                              boxstyle="round,pad=0.3",
                              facecolor=COL_FINAL, edgecolor='white', linewidth=2)
ax.add_patch(box)
ax.text(50, 18.7, "meta_avg = (LGB + XGB + CB) / 3\n→ mega33  OOF 8.40, LB 9.7759",
        fontsize=11.5, ha='center', va='center', color='white', fontweight='bold')

# 옆에 추가 블렌드 표시
ax.annotate('', xy=(50, 9), xytext=(50, 13.5),
            arrowprops=dict(arrowstyle='->', color='#7f8c8d', lw=1.5))
box2 = patches.FancyBboxPatch((6, 1), 88, 7,
                               boxstyle="round,pad=0.3",
                               facecolor='#ecf0f1', edgecolor='#2c3e50', linewidth=2)
ax.add_patch(box2)
ax.text(50, 4.5,
        "최종: 0.64 × FIXED(mega33+rank+iter) + 0.12 × oracle_xgb + 0.16 × oracle_log + 0.08 × oracle_rem  →  LB 9.7527",
        fontsize=10.3, ha='center', va='center',
        color='#2c3e50', fontweight='bold')

plt.savefig(f"{OUT}/04_mega33_structure.png", dpi=150, bbox_inches='tight')
plt.close()
print("   저장 완료")

print(f"\n4개 이미지 v2 생성 완료 → {OUT}")
