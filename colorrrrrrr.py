import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import json

# --- 核心分析函数 ---
def analyze_image(img_obj, threshold=0.05, n_clusters=20):
    # 转换颜色空间
    img_rgb = cv2.cvtColor(img_obj, cv2.COLOR_BGR2RGB)
    # 缩小图片以加快计算
    pixels = cv2.resize(img_rgb, (100, 100)).reshape(-1, 3)
    
    # 聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(pixels)
    
    # 计算占比并筛选
    counts = np.bincount(kmeans.labels_)
    proportions = counts / len(kmeans.labels_)
    mask = proportions >= threshold
    
    colors = kmeans.cluster_centers_[mask] / 255.0
    proportions = proportions[mask]
    
    # 排序
    idx = np.argsort(proportions)[::-1]
    return colors[idx], proportions[idx]

# --- Streamlit 界面 ---
st.set_page_config(page_title="色彩提取工具", layout="wide")
st.title("🎨 在线色彩提取分析")

uploaded_file = st.file_uploader("请上传图片...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # 读取内存中的图片数据
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    # 调节参数
    threshold = st.slider("最小颜色占比阈值", 0.01, 0.20, 0.05)
    
    # 执行分析
    colors, proportions = analyze_image(img, threshold=threshold)
    
    # 展示色卡图
    fig, ax = plt.subplots(figsize=(10, 2))
    start = 0
    for color, prop in zip(colors, proportions):
        ax.barh(0, prop, left=start, color=color, height=0.5)
        start += prop
    ax.axis('off')
    st.pyplot(fig)
    
    # 准备 JSON 数据
    json_data = []
    for color, prop in zip(colors, proportions):
        r, g, b = map(int, color * 255)
        json_data.append({
            "rgb": [r, g, b],
            "hex": '#{:02x}{:02x}{:02x}'.format(r, g, b),
            "proportion": round(float(prop), 4)
        })
    
    # 展示 JSON 并提供下载
    st.subheader("提取的颜色数据 (JSON)")
    st.json(json_data)
    st.download_button("📥 下载 JSON 数据", json.dumps(json_data), "palette.json", "application/json")
