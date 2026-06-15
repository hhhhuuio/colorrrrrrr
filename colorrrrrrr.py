import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.cluster import KMeans
import colorsys
import json
import io
import struct

# --- 核心分析逻辑 ---
def analyze_image(img_obj, threshold=0.001, n_clusters=20):
    img_rgb = cv2.cvtColor(img_obj, cv2.COLOR_BGR2RGB)
    pixels = cv2.resize(img_rgb, (100, 100)).reshape(-1, 3)
    # n_clusters 越大，颜色提取越精细，即容差调节
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(pixels)
    
    counts = np.bincount(kmeans.labels_)
    proportions = counts / len(kmeans.labels_)
    
    mask = proportions >= threshold
    colors = kmeans.cluster_centers_[mask] / 255.0
    proportions = proportions[mask]
    
    sort_idx = np.argsort(proportions)[::-1]
    return colors[sort_idx], proportions[sort_idx]

# --- 页面 UI ---
st.set_page_config(page_title="色彩分析大师", layout="wide")
st.title("🎨 极致色彩分析报告")

# 上传
uploaded_file = st.file_uploader("上传图片以分析...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    # 排版优化：侧边控制栏
    with st.sidebar:
        st.header("⚙️ 调节参数")
        threshold = st.slider("最小占比阈值", 0.001, 0.05, 0.005)
        clusters = st.slider("颜色容差(聚类数)", 5, 50, 20)
        gen_aco = st.checkbox("导出 Photoshop 色板 (.aco)", True)

    # 布局：左侧图，右侧数据
    col1, col2 = st.columns([1, 1.5])
    colors, props = analyze_image(img, threshold=threshold, n_clusters=clusters)
    
    with col1:
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="原图预览", use_container_width=True)
        st.metric("提取到颜色数量", len(colors))
        
    with col2:
        st.subheader("占比色卡")
        fig1, ax1 = plt.subplots(figsize=(8, 1))
        start = 0
        for c, p in zip(colors, props):
            ax1.barh(0, p, left=start, color=c, height=1)
            start += p
        ax1.axis('off')
        st.pyplot(fig1)

    # 渐变与色环
    st.divider()
    c3, c4 = st.columns(2)
    
    with c3:
        st.subheader("明度渐变 (由暗至亮)")
        lum = np.array([0.299*c[0]+0.587*c[1]+0.114*c[2] for c in colors])
        sorted_lum = colors[np.argsort(lum)]
        fig2, ax2 = plt.subplots(figsize=(8, 1))
        ax2.imshow(np.linspace(0, 1, 512).reshape(1, -1), aspect='auto', cmap=mcolors.LinearSegmentedColormap.from_list("lum", sorted_lum))
        ax2.axis('off')
        st.pyplot(fig2)

    with c4:
        st.subheader("全饱和色环分布")
        fig3, ax3 = plt.subplots(figsize=(5, 5), subplot_kw={'projection': 'polar'})
        # 填充色环内部
        for r in np.linspace(0.1, 1, 20):
            for a in np.linspace(0, 2*np.pi, 100):
                ax3.plot(a, r, 'o', color=colorsys.hsv_to_rgb(a/(2*np.pi), r, 1), markersize=3, alpha=0.3)
        # 标注主色
        for c in colors:
            h, s, v = colorsys.rgb_to_hsv(*c)
            ax3.plot(h*2*np.pi, s, 'o', color=c, markersize=12, markeredgecolor='white')
        ax3.set_yticklabels([])
        st.pyplot(fig3)

    # 下载功能
    if st.button("生成色板下载"):
        st.download_button("下载 JSON", json.dumps([mcolors.to_hex(c) for c in colors]), "palette.json")
        if gen_aco:
            buf = io.BytesIO()
            buf.write(struct.pack('>HH', 1, len(colors)))
            for c in colors:
                buf.write(struct.pack('>HHHHH', 0, int(c[0]*257), int(c[1]*257), int(c[2]*257), 0))
            st.download_button("下载 .aco", buf.getvalue(), "palette.aco")
