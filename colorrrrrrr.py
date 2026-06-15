import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.cluster import KMeans
import colorsys
import struct
import json
import io

# --- 核心辅助函数 ---
def get_aco_bytes(raw_colors):
    """生成 ACO 文件字节流"""
    buf = io.BytesIO()
    buf.write(struct.pack('>HH', 1, len(raw_colors)))
    for color in raw_colors:
        buf.write(struct.pack('>HHHHH', 0, int(color[0]*257), int(color[1]*257), int(color[2]*257), 0))
    return buf.getvalue()

def analyze_image(img_obj, threshold=0.01, n_clusters=20):
    img_rgb = cv2.cvtColor(img_obj, cv2.COLOR_BGR2RGB)
    pixels = cv2.resize(img_rgb, (100, 100)).reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(pixels)
    
    counts = np.bincount(kmeans.labels_)
    proportions = counts / len(kmeans.labels_)
    mask = proportions >= threshold
    
    colors = kmeans.cluster_centers_[mask] / 255.0
    proportions = proportions[mask]
    
    # 按明度排序 (Y = 0.299R + 0.587G + 0.114B)
    luminance = np.array([0.299*c[0] + 0.587*c[1] + 0.114*c[2] for c in colors])
    sort_idx = np.argsort(luminance)
    return colors[sort_idx], proportions[sort_idx]

# --- 页面显示 ---
st.set_page_config(page_title="专业色彩分析", layout="wide")
st.title("🎨 深度色彩分析报告")

uploaded_file = st.file_uploader("上传图片以分析...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    col_l, col_r = st.columns([1, 2])
    with col_l:
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="原图", use_container_width=True)
        threshold = st.slider("最小占比阈值", 0.005, 0.10, 0.02)
    
    colors, props = analyze_image(img, threshold=threshold)
    
    # 绘图区域
    fig = plt.figure(figsize=(12, 8))
    
    # 1. 渐变条
    ax1 = fig.add_subplot(211)
    cmap = mcolors.LinearSegmentedColormap.from_list("grad", colors)
    ax1.imshow(np.linspace(0, 1, 512).reshape(1, -1), aspect='auto', cmap=cmap)
    ax1.set_title("按明度排序的渐变条")
    ax1.axis('off')
    
    # 2. 色环分布
    ax2 = fig.add_subplot(212, projection='polar')
    for color in colors:
        h, s, v = colorsys.rgb_to_hsv(*color)
        ax2.plot(h*2*np.pi, s, 'o', color=color, markersize=15, markeredgecolor='white')
    ax2.set_title("色相/饱和度色环分布")
    
    st.pyplot(fig)
    
    # 导出功能
    st.subheader("导出")
    c1, c2 = st.columns(2)
    # JSON
    json_data = json.dumps([{"hex": mcolors.to_hex(c), "rgb": [int(x*255) for x in c]} for c in colors])
    c1.download_button("📥 下载 JSON", json_data, "palette.json")
    # ACO
    c2.download_button("📥 下载 Photoshop 色板 (.aco)", get_aco_bytes(colors*255), "palette.aco")
