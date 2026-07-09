import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from scipy.stats import truncnorm

target_sample=10000
vr0=round(float(input()), 2)  
sigma0=round(float(input()), 2)
df = pd.read_csv("C:/java/tst.csv")
vr = df[df['P(member)']==1]['vr'].dropna()
n = len(vr)
table_data = {}
step = 0.1
vr_min, vr_max = -380, -360
sigma_min, sigma_max = 1, 20
vr_range = np.linspace(vr_min, vr_max, int((vr_max - vr_min)/step) + 1)
sigma_range = np.linspace(sigma_min, sigma_max, int((sigma_max - sigma_min)/step) + 1)

for i in vr_range:
    key_i = round(i, 2)
    table_data[key_i] = {}
    for j in sigma_range: 
        key_j=round(j,2)
        log_pdf_each = (-0.5*np.log(2*np.pi)-np.log(j)-(vr-i)**2/(2*j**2))
        logL = log_pdf_each.sum()
        table_data[key_i][key_j] = logL 

vr_step=1
sigma_step=0.5

def truncated_normal(loc, scale, lower, upper):
    """
    生成截断正态分布随机数
    loc: 均值
    scale: 标准差
    lower: 下限
    upper: 上限
    """
    a = (lower - loc) / scale
    b = (upper - loc) / scale
    return truncnorm.rvs(a, b, loc=loc, scale=scale)

datachose={}
all_points = []  
point_counter = Counter()
accepted_points = [] 

nn=10000
vrchose=vr0
sigmachose=sigma0
for i in range(0,10000):
    datachose[i]={}
    newvr = truncated_normal(vrchose, vr_step, -380, -360)
    newsigma = truncated_normal(sigmachose, sigma_step, 1, 20)
    
    if newvr < -380:
        newvr = -380
    elif newvr > -360:
        newvr = -360
    
    if newsigma < 1:
        newsigma = 1
    elif newsigma > 20:
        newsigma = 20
        
    # 四舍五入到最近的网格点（步长0.05）
    step_vr = 0.1
    step_sigma = 0.1
    newvr = round(newvr / step_vr) * step_vr
    newsigma = round(newsigma / step_sigma) * step_sigma
    newvr = round(newvr, 2)
    newsigma = round(newsigma, 2)
    
    alp=np.random.uniform(0,1)
    
    vr_key = round(vrchose, 2)
    sigma_key = round(sigmachose, 2)
    new_vr_key = round(newvr, 2)
    new_sigma_key = round(newsigma, 2)
    
    current_logL = table_data[round(vrchose,2)][round(sigmachose,2)]
    new_logL = table_data[round(newvr,2)][round(newsigma,2)]


    log_ratio = new_logL - current_logL
    
    if log_ratio > 0 or np.exp(log_ratio) > alp: 
        vrchose=newvr
        sigmachose=newsigma
        accepted_points.append((vrchose, sigmachose))
    else:
        nn=nn-1
    all_points.append((vrchose, sigmachose))
    point_counter[(round(vrchose, 2), round(sigmachose, 2))] += 1
    datachose[i][0]=vrchose
    datachose[i][1]=sigmachose
    
 
vr_values = [p[0] for p in accepted_points]
sigma_values = [p[1] for p in accepted_points]

# 为每个被接受的点获取访问次数
counts = [point_counter[(round(vr_values[i], 2), round(sigma_values[i], 2))] for i in range(len(vr_values))]


# 创建图形
plt.figure(figsize=(10, 8))

# 使用色阶变化绘制散点图
scatter = plt.scatter(vr_values, sigma_values, 
                      c=counts,           # 颜色映射到访问次数
                      cmap='plasma',      # 使用plasma色阶
                      s=15,               # 点的大小
                      alpha=0.7,          # 透明度
                      edgecolors='black', # 黑色边框
                      linewidth=0.1)      # 边框宽度



# 设置标签和标题
plt.xlabel('vr值')
plt.ylabel('sigma值')
plt.title('被接受点的散点图（颜色表示访问次数）')

cbar = plt.colorbar(scatter, label='Selection Count')

# x轴标签改为带单位的英文（径向速度）
plt.xlabel(r'$v_r$ (km s$^{-1}$)', fontsize=12)

# y轴标签改为带单位的英文（速度弥散）
plt.ylabel(r'$\sigma_{vr}$ (km s$^{-1}$)', fontsize=12)



# 标题改为英文
plt.title('Scatter plot of accepted points (color represents selection count)', fontsize=12)


plt.grid(True, alpha=0.5)

plt.tight_layout()
plt.show()

probility=float(nn/10000)
print(probility)
    
    
    
