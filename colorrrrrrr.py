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

# --- 工具函数 ---
def get_aco_bytes(raw_colors):
    buf = io.BytesIO()
    buf.write(struct.pack('>HH', 1, len(raw_colors)))
    for color in raw_colors:
        buf.write(struct.pack('>HHHHH', 0, int(color[0]*257), int(color[1]*257), int(color[2]*257), 0))
    return buf.getvalue()

def analyze_image(img_obj, threshold=0.005, n_clusters=20):
    img_rgb = cv2.cvtColor(img_obj, cv2.COLOR_BGR2RGB)
    pixels = cv2.resize(img_rgb, (100, 100)).reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(pixels)
    
    counts = np.bincount(kmeans.labels_)
    proportions = counts / len(kmeans.labels_)
    
    # 筛选
    mask = proportions >= threshold
    colors = kmeans.cluster_centers_[mask] / 255.0
    proportions = proportions[mask]
    
    # 占比排序
    sort_idx_prop = np.argsort(proportions)[::-1]
    
    # 明度排序 (Y = 0.299R + 0.587G + 0.114B)
    luminance = np.array([0.299*c[0] + 0.587*c[1] + 0.114*c[2] for c in colors])
    sort_idx_lum = np.argsort(luminance)
    
    return colors[sort_idx_prop], proportions[sort_idx_prop], colors[sort_idx_lum]

# --- 页面逻辑 ---
st.set_page_config(page_title="色彩分析大师", layout="wide")
st.title("🎨 专业级色彩分析报告")

uploaded_file = st.file_uploader("上传图片...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    # 参数侧边栏
    with st.sidebar:
        threshold = st.slider("最小占比阈值", 0.001, 0.10, 0.01)
        gen_aco = st.checkbox("导出为 Photoshop 色板 (.aco)", value=True)
    
    colors_prop, props, colors_lum = analyze_image(img, threshold=threshold)
    
    # 1. 展示原图
    col1, col2 = st.columns([1, 1])
    col1.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="原图", use_container_width=True)
    
    # 2. 占比色卡
    col2.subheader("色彩占比排序")
    fig1, ax1 = plt.subplots(figsize=(6, 1))
    start = 0
    for c, p in zip(colors_prop, props):
        ax1.barh(0, p, left=start, color=c, height=1)
        start += p
    ax1.axis('off')
    col2.pyplot(fig1)
    
    # 5. 明度渐变
    st.subheader("按明度排序的渐变")
    fig2, ax2 = plt.subplots(figsize=(10, 1))
    cmap = mcolors.LinearSegmentedColormap.from_list("lum", colors_lum)
    ax2.imshow(np.linspace(0, 1, 512).reshape(1, -1), aspect='auto', cmap=cmap)
    ax2.axis('off')
    st.pyplot(fig2)
    
    # 6. 色环分布 (含真实色环显示)
    st.subheader("色环分布示意")
    fig3, ax3 = plt.subplots(figsize=(6, 6), subplot_kw={'projection': 'polar'})
    # 绘制背景色环
    angles = np.linspace(0, 2*np.pi, 360)
    for a in angles:
        ax3.bar(a, 0.2, width=2*np.pi/360, bottom=0.8, color=colorsys.hsv_to_rgb(a/(2*np.pi), 1, 1))
    # 绘制主色点
    for c in colors_prop:
        h, s, v = colorsys.rgb_to_hsv(*c)
        ax3.plot(h*2*np.pi, s, 'o', color=c, markersize=15, markeredgecolor='white', markeredgewidth=2)
    ax3.set_yticklabels([])
    st.pyplot(fig3)
    
    # 下载区
    c1, c2 = st.columns(2)
    c1.download_button("📥 下载 JSON 数据", json.dumps([mcolors.to_hex(c) for c in colors_prop]), "data.json")
    if gen_aco:
        c2.download_button("📥 下载 .aco 色板", get_aco_bytes(colors_prop*255), "palette.aco")
