import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.cluster import KMeans
import colorsys
import sys
import os
import struct

def save_ps_aco(raw_colors, filename):
    try:
        with open(filename, 'wb') as f:
            f.write(struct.pack('>HH', 1, len(raw_colors)))
            for color in raw_colors:
                f.write(struct.pack('>HHHHH', 0, int(color[0])*257, int(color[1])*257, int(color[2])*257, 0))
        print(f"\n🎉【成功】已生成色板: {filename}")
    except Exception as e:
        print(f"\n❌ 失败: {e}")

def analyze_image_colors(image_path, threshold=0.05, n_clusters=20):
    """
    threshold: 提取颜色的最小占比 (例如 0.05 表示占比大于5%的颜色才会被保留)
    n_clusters: 初始聚类数量 (越大越精细)
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 聚类提取
    pixels = cv2.resize(img, (100, 100)).reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(pixels)
    
    # 筛选大于阈值的颜色
    counts = np.bincount(kmeans.labels_)
    proportions = counts / len(kmeans.labels_)
    mask = proportions >= threshold
    
    colors = kmeans.cluster_centers_[mask] / 255.0
    proportions = proportions[mask]
    
    # 按占比排序
    idx = np.argsort(proportions)[::-1]
    colors, proportions = colors[idx], proportions[idx]

    # 绘图区域划分
    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)

    # 1. 原始图
    ax1 = fig.add_subplot(gs[:, 0])
    ax1.imshow(img)
    ax1.axis('off')
    ax1.set_title('原始图片')

    # 2. 占比饼图
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.pie(proportions, labels=[f"{p:.1%}" for p in proportions], colors=colors, autopct='', startangle=140)
    ax2.set_title('色彩占比分布')

    # 3. 条形色卡
    ax3 = fig.add_subplot(gs[0, 2])
    start = 0
    for color, prop in zip(colors, proportions):
        ax3.barh(0, prop, left=start, color=color, height=0.5)
        start += prop
    ax3.axis('off')
    ax3.set_title('色卡序列')

    # 4. 连续渐变
    ax4 = fig.add_subplot(gs[1, 1:])
    cmap = mcolors.LinearSegmentedColormap.from_list("custom", colors)
    ax4.imshow(np.linspace(0, 1, 512).reshape(1, -1), aspect='auto', cmap=cmap)
    ax4.axis('off')
    ax4.set_title('主色提取渐变')

    # 保存色板
    base_name = os.path.splitext(image_path)[0]
    save_ps_aco(colors * 255, f"{base_name}_palette.aco")

    plt.suptitle(f"色彩分析报告 | 阈值: {threshold*100}%", fontsize=16)
    plt.show()

if __name__ == '__main__':
    # 使用方法: python script.py image.jpg 0.05
    import streamlit as st

# ... (保持原有的 analyze_image_colors 函数不变) ...

st.title("🎨 色彩分析工具")
uploaded_file = st.file_uploader("请上传一张图片...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # 将上传的文件转为 OpenCV 可读的格式
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    # 临时保存或直接传入 img 对象进行分析
    # 为了兼容你现有的函数，建议稍微修改一下 analyze_image_colors 
    # 让它接受 img 对象而不是文件路径
    analyze_image_colors_from_obj(img) 

def analyze_image_colors_from_obj(img, threshold=0.05, n_clusters=20):
    # 这里的逻辑和之前的 analyze_image_colors 一模一样
    # 只是第一行把 cv2.imread(image_path) 改为直接使用传入的 img
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # ... (其余代码保持不变)
