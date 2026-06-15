import streamlit as st
import cv2, numpy as np, matplotlib.pyplot as plt, matplotlib.colors as mcolors
from sklearn.cluster import KMeans
import colorsys, json, io, struct, plotly.graph_objects as go

# 强制紧凑 UI 布局
st.set_page_config(page_title="色彩分析", layout="wide")
st.markdown("<style>div[data-testid='stBlock']{padding:2px!important} .stMetric{font-size:0.8rem!important}</style>", unsafe_allow_html=True)

# 核心算法
def process_image(img, threshold, n_clusters):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pixels = cv2.resize(img_rgb, (80, 80)).reshape(-1, 3)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=8).fit(pixels)
    counts = np.bincount(km.labels_)
    props = counts / len(km.labels_)
    mask = props >= threshold
    cols = km.cluster_centers_[mask] / 255.0
    props = props[mask]
    # 占比排序
    idx = np.argsort(props)[::-1]
    return cols[idx], props[idx]

# UI 界面
st.title("🎨 极致色彩协同看板")
uploaded = st.file_uploader("上传设计图", type=["jpg", "png"])

if uploaded:
    img = cv2.imdecode(np.asarray(bytearray(uploaded.read()), dtype=np.uint8), 1)
    # 参数栏
    cols_ctrl = st.columns(4)
    thr = cols_ctrl[0].slider("阈值", 0.0001, 0.03, 0.001)
    clus = cols_ctrl[1].slider("容差", 5, 60, 24)
    exc = cols_ctrl[2].checkbox("剔除低占比焦点色", True)
    aco = cols_ctrl[3].checkbox("导出 .aco", True)

    cols, props = process_image(img, thr, clus)
    main_c = cols[0]

    # 并排显示
    c1, c2 = st.columns([1, 1])
    c1.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), width=350)
    
    # 交互色环 (Plotly)
    fig = go.Figure()
    # 绘制背景光谱
    for r in np.linspace(0.1, 1, 10):
        for t in np.linspace(0, 360, 60):
            fig.add_trace(go.Scatterpolar(r=[r], theta=[t], mode='markers', marker=dict(color=mcolors.to_hex(colorsys.hsv_to_rgb(t/360, r, 1)), size=5), hoverinfo='skip'))
    
    # 绘制色点
    for c in cols:
        h, s, v = colorsys.rgb_to_hsv(*c)
        diff = np.linalg.norm(c - main_c)
        fig.add_trace(go.Scatterpolar(r=[s], theta=[h*360], mode='markers', marker=dict(size=12, color=mcolors.to_hex(c), line=dict(color='white', width=2)), 
                      text=f"对比相似度:{max(0, 100-diff*50):.1f}%", hoverinfo='text'))
    
    fig.update_layout(polar=dict(radialaxis=dict(showticklabels=False), angularaxis=dict(showticklabels=False)), showlegend=False, width=350, height=350, margin=dict(l=0,r=0,t=0,b=0))
    c2.plotly_chart(fig, use_container_width=True)

    # 线性色卡
    st.subheader("📊 色彩演化面板")
    # 1. 占比色卡
    f1, a1 = plt.subplots(figsize=(10, 0.5)); start=0
    for c, p in zip(cols, props): a1.barh(0, p, left=start, color=c, height=1); start+=p
    a1.axis('off'); st.pyplot(f1)
    
    # 2. 明度排序 (剔除焦点色逻辑)
    lums = [0.299*c[0]+0.587*c[1]+0.114*c[2] for c in cols]
    idx = np.argsort(lums)
    c_lum, p_lum = cols[idx], props[idx]
    
    if exc:
        c_grad = [c for c, p in zip(c_lum, p_lum) if p >= 0.05]
    else:
        c_grad = c_lum
        
    f2, a2 = plt.subplots(figsize=(10, 0.5))
    a2.imshow(np.linspace(0, 1, 512).reshape(1, -1), aspect='auto', cmap=mcolors.LinearSegmentedColormap.from_list("g", c_grad))
    a2.axis('off'); st.pyplot(f2)
