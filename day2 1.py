import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv("C:/java/tst.csv")
vr = df[df['P(member)']==1]['vr'].dropna()
n = len(vr)
table_data = {}
for i in np.arange(-380, -360, 0.05):
    table_data[i] = {}
    for j in np.arange(1, 20, 0.05):
        log_pdf_each = (-0.5*np.log(2*np.pi)-np.log(j)-(vr-i)**2/(2*j**2))
        logL = log_pdf_each.sum()
        table_data[i][j] = logL

v_mean_grid = sorted(table_data.keys())
sigma_grid = sorted(table_data[v_mean_grid[0]].keys())


logL_2d = np.zeros((len(sigma_grid),len(v_mean_grid)))
for i_idx,v_mean in enumerate(v_mean_grid):
    for j_idx,sigma in enumerate(sigma_grid):
        logL_2d[j_idx,i_idx] = table_data[v_mean][sigma]



logL_2d = np.zeros((len(sigma_grid), len(v_mean_grid)))
for i_idx, v_mean in enumerate(v_mean_grid):
    for j_idx, sigma in enumerate(sigma_grid):
        logL_2d[j_idx, i_idx] = table_data[v_mean][sigma]

logL_max = np.max(logL_2d)
max_index = np.unravel_index(np.argmax(logL_2d), logL_2d.shape)
best_sigma = sigma_grid[max_index[0]]
best_vmean = v_mean_grid[max_index[1]]

# ============================================
# 3. 计算对数似然差值（图中显示的是这个）
# ============================================
logL_diff = logL_2d - logL_max  # 范围: -20 到 0

print(f"最优: v̄={best_vmean:.2f}, σ_v={best_sigma:.2f}")
print(f"最大对数似然: {logL_max:.4f}")
threshold = -20
logL_masked = logL_diff.copy()
logL_masked[logL_diff < threshold] = np.nan
# ============================================
# 5. 画图（模仿图中风格）
# ============================================
X, Y = np.meshgrid(v_mean_grid, sigma_grid)

fig, ax = plt.subplots(figsize=(10, 8))
threshold=-20

logL_masked=logL_diff.copy()

logL_masked[logL_diff < threshold]=np.nan

contour=ax.contourf(
    X,
    Y,
    logL_masked,
    levels=np.linspace(-20,0,100),
    cmap="viridis"
)

# ---- 修改2: 颜色条显示 -2.5, -5.0, -7.5, ...（和图中一致） ----
cbar = plt.colorbar(contour, ax=ax, ticks=np.arange(-20, 1, 2.5))
cbar.set_label(r'$\ln \mathcal{L} - \ln \mathcal{L}_\mathrm{max}$', fontsize=13)

# ---- 修改3: 等高线（白色细线，和图中风格一致） ----
levels_contour = [-15, -12.5, -10, -7.5, -5, -2.5,-1]
lines = ax.contour(X, Y, logL_masked, levels=levels_contour, 
                    colors="white", linewidths=1.0, linestyles='-')
ax.clabel(lines, fontsize=9, fmt='%.1f', inline=True, inline_spacing=5)

# ---- 修改4: 标记最佳拟合点（白色星号，和图中一致） ----
ax.scatter(
    best_vmean,
    best_sigma,
    color="white",
    marker="*",
    s=300,
    edgecolors="black",
    linewidths=1,
    zorder=10,
    label='Best Fit'
)

ax.set_xlim(-376, -366)
ax.set_ylim(6,15)

# ---- 修改6: 坐标轴标签 ----
ax.set_ylabel(r'$\sigma_{vr}$ (km s$^{-1}$)', fontsize=14)
ax.set_xlabel(r'$\bar{v}_r$ (km s$^{-1}$)', fontsize=14)

# ---- 修改7: 标题 ----
ax.set_title(r'Relative Log-Likelihood Contour Heatmap', fontsize=15)

# ---- 修改8: 添加信息框（图中右上角） ----
info_text = f"Best mean vr = {best_vmean:.2f} km/s\nBest sigma_vr = {best_sigma:.2f} km/s"
props = dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='gray')
ax.text(0.97, 0.97, info_text, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', horizontalalignment='right',
        bbox=props, fontfamily='monospace')

# ---- 修改9: 去掉网格线 ----
ax.grid(False)

# ---- 修改10: 设置刻度 ----
ax.set_xticks(np.arange(-382, -358, 2))
ax.set_yticks(np.arange(2, 19, 2))

plt.tight_layout()
plt.savefig("C:/java/Likelihood_heatmap.png", dpi=300, bbox_inches='tight')
plt.show()
print("✅ 已保存: C:/java/Likelihood_heatmap.png")