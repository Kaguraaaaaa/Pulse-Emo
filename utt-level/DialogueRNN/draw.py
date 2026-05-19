import matplotlib.pyplot as plt
import numpy as np

# 情绪及对应角度（度）
emotions = ['Anger', 'Disgust', 'Sad', 'Fear', 'Happy', 'Surprise']
angles_deg = [0, 45, 90, 180, 270, 300]  # 角度
angles_rad = np.radians(angles_deg)

# 圆心 (Neutral)
neutral_pos = (0, 0)

fig, ax = plt.subplots(figsize=(8,8), subplot_kw={'projection': 'polar'})
ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)

# 画圆心的 Neutral 点
ax.plot(0, 0, 'o', color='gray', markersize=10)
ax.text(0, 0, 'Neutral', ha='center', va='center', fontsize=12, fontweight='bold', color='gray')

# 画从圆心到每个情绪点的连线，情绪点放在线上中间
for i, angle in enumerate(angles_rad):
    # 线条：从圆心到半径1位置
    ax.plot([0, angle], [0, 1], color='black', linewidth=1)

    # 计算情绪点位置在边上的中点（半径0.5）
    emotion_r = 0.5
    ax.plot(angle, emotion_r, 'o', markersize=8, color='blue')

    # 情绪标签稍微偏离点位置
    label_r = emotion_r + 0.1
    ax.text(angle, label_r, emotions[i], ha='center', va='center', fontsize=12, fontweight='bold')

# 标注边与边之间的夹角（度数），计算相邻两点间的角度差
for i in range(len(angles_deg)):
    start = angles_rad[i]
    end = angles_rad[(i+1) % len(angles_rad)]
    # 夹角（度）
    # 注意这里用正角度，考虑顺时针方向，夹角不能为负
    angle_diff_deg = (angles_deg[(i+1) % len(angles_deg)] - angles_deg[i]) % 360
    # 夹角位置：两个角度的中点，半径取0.75，稍微靠外
    mid_angle = (start + end) / 2
    mid_r = 0.75
    ax.text(mid_angle, mid_r, f'{angle_diff_deg}°', fontsize=10, ha='center', va='center', color='red')

# 取消极坐标的刻度
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.grid(False)

ax.set_title("简化情感轮示意图", fontsize=16, pad=20)
plt.show()