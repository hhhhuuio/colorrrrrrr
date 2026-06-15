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

# --- 核心分析与排序算法 ---
def analyze_image(img_obj, threshold=0.001, n_clusters=20):
    img_rgb = cv2.cvtColor(img_obj, cv2.COLOR_BGR2RGB)
    pixels = cv2.resize(img_rgb, (100, 100)).reshape(-1, 3)
    
    # K-Means 聚类（数量即为容差精细度）
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(pixels)
    
    counts = np.bincount(kmeans.labels_)
    proportions = counts / len(kmeans.labels_)
    
    # 过滤低占比颜色
    mask = proportions >= threshold
    colors = kmeans.cluster_centers_[mask] / 255.0
    proportions = proportions[mask]
    
    # 1. 按占比从大到小排序
    sort_idx_prop = np.argsort(proportions)[::-1]
    colors_prop = colors[sort_idx_prop]
    props_prop = proportions[sort_idx_prop]
    
    # 2. 按明度由暗到亮排序 (Y = 0.299R + 0.587G + 0.114B)
    luminance = np.array([0.299*c[0] + 0.587*c[1] + 0.114*c[2] for c in colors_prop])
    sort_idx_lum = np.argsort(luminance)
    colors_lum = colors_prop[sort_idx_lum]
    
    # 3. 计算焦点色（与其他主要颜色平均距离最远的颜色）
    if len(colors_prop) > 1:
        avg_dists = []
        for i, c in enumerate(colors_prop):
            dists = [np.linalg.norm(c - other) for j, other in enumerate(colors_prop) if i != j]
            avg_dists.append(np.mean(dists))
        focus_idx = np.argmax(avg_dists)
        focus_color = colors_prop[focus_idx]
    else:
        focus_color = colors_prop[0] if len(colors_prop) > 0 else np.array([0.0, 0.0, 0.0])

    return colors_prop, props_prop, colors_lum, focus_color

# --- Streamlit 界面配置 ---
st.set_page_config(page_title="高级色彩看板", layout="wide")
st.title("🎨 极简色彩协同分析看板")

# 1. 顶层控制面板
st.markdown("### ⚙️ 控制参数")
c_ctrl1, c_ctrl2, c_ctrl3, c_ctrl4 = st.columns(4)
with c_ctrl1:
    threshold = st.slider("最小占比阈值", 0.0001, 0.05, 0.002, format="%.4f")
with c_ctrl2:
    clusters = st.slider("色彩融合容差 (聚类数)", 5, 60, 25)
with c_ctrl3:
    show_focus_grad = st.checkbox("启用焦点色互补渐变", value=True)
with c_ctrl4:
    gen_aco = st.checkbox("准备 PS 色板 (.aco) 下载", value=True)

uploaded_file = st.file_uploader("选择并上传你的设计图/摄影作品...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # 转换内存图片
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    # 执行色彩核心算法
    colors_prop, props_prop, colors_lum, focus_color = analyze_image(img, threshold=threshold, n_clusters=clusters)
    
    st.divider()
    
    # 2. 核心大模块：原图 与 实色色环 并排显示
    col_img, col_wheel = st.columns(2)
    
    with col_img:
        st.subheader("🖼️ 原始图像")
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
        st.metric("核心提取色板数量", len(colors_prop))
        
    with col_wheel:
        st.subheader("⭕ 实色光谱色环分布")
        fig_wheel, ax_wheel = plt.subplots(figsize=(6, 6), subplot_kw={'projection': 'polar'})
        
        # 绘制极坐标实色填充背景
        r_space = np.linspace(0, 1, 40)
        theta_space = np.linspace(0, 2*np.pi, 120)
        C_rgb = np.zeros((len(r_space)-1, len(theta_space)-1, 3))
        for i in range(len(r_space)-1):
            r_c = (r_space[i] + r_space[i+1]) / 2
            for j in range(len(theta_space)-1):
                t_c = (theta_space[j] + theta_space[j+1]) / 2
                C_rgb[i, j] = colorsys.hsv_to_rgb(t_c / (2*np.pi), r_c, 1.0)
                
        ax_wheel.pcolormesh(theta_space, r_space, C_rgb, shading='flat', zorder=1)
        
        # 标出提取出来的主色点
        for c in colors_prop:
            h, s, v = colorsys.rgb_to_hsv(*c)
            # 使用双色高对比度边框确保在任何背景色下都清晰可见
            ax_wheel.plot(h*2*np.pi, s, 'o', color=c, markersize=14, 
                          markeredgecolor='black', markeredgewidth=2, zorder=10)
            ax_wheel.plot(h*2*np.pi, s, 'o', color=c, markersize=10, 
                          markeredgecolor='white', markeredgewidth=1, zorder=11)
            
        ax_wheel.set_yticklabels([])
        ax_wheel.set_xticklabels([])
        ax_wheel.grid(False)
        st.pyplot(fig_wheel)

    st.divider()

    # 3. 线性面板：色卡与渐变上下同列垂直展示
    st.subheader("📊 线性色彩演化条")
    
    # Palette 1: 占比排序色卡
    st.markdown("**1. 主色占比分配色卡 (按画面覆盖率由大到小)**")
    fig_p1, ax_p1 = plt.subplots(figsize=(12, 0.8))
    start = 0
    for c, p in zip(colors_prop, props_prop):
        ax_p1.barh(0, p, left=start, color=c, height=1, edgecolor='none')
        start += p
    ax_p1.axis('off')
    st.pyplot(fig_p1)
    
    # Palette 2: 明度排序离散色卡
    st.markdown("**2. 等宽明度阶梯色卡 (离散型：由暗至亮排序)**")
    fig_p2, ax_p2 = plt.subplots(figsize=(12, 0.8))
    n_cls = len(colors_lum)
    for i, c in enumerate(colors_lum):
        ax_p2.barh(0, 1/n_cls, left=i/n_cls, color=c, height=1, edgecolor='none')
    ax_p2.axis('off')
    st.pyplot(fig_p2)
    
    # Palette 3: 明度连续渐变
    st.markdown("**3. 空间明度连续平衡渐变条**")
    fig_p3, ax_p3 = plt.subplots(figsize=(12, 0.8))
    cmap_lum = mcolors.LinearSegmentedColormap.from_list("lum_grad", colors_lum)
    ax_p3.imshow(np.linspace(0, 1, 768).reshape(1, -1), aspect='auto', cmap=cmap_lum)
    ax_p3.axis('off')
    st.pyplot(fig_p3)
    
    # Palette 4: 焦点色对比渐变（按选项开启）
    if show_focus_grad and len(colors_prop) > 0:
        st.markdown(f"**4. 🎯 焦点色交互对比渐变 (左侧互补色 ➡️ 右侧逆向焦点色: `{mcolors.to_hex(focus_color)}`)**")
        h_f, s_f, v_f = colorsys.rgb_to_hsv(*focus_color)
        comp_h = (h_f + 0.5) % 1.0
        comp_color = colorsys.hsv_to_rgb(comp_h, s_f, v_f)
        
        fig_p4, ax_p4 = plt.subplots(figsize=(12, 0.8))
        cmap_focus = mcolors.LinearSegmentedColormap.from_list("focus_grad", [comp_color, focus_color])
        ax_p4.imshow(np.linspace(0, 1, 768).reshape(1, -1), aspect='auto', cmap=cmap_focus)
        ax_p4.axis('off')
        st.pyplot(fig_p4)

    st.divider()

    # 4. 数据资产下载区域
    st.subheader("💾 资产导出")
    col_d1, col_d2 = st.columns(2)
    
    hex_list = [mcolors.to_hex(c) for c in colors_prop]
    col_d1.download_button("📥 下载标准配色 JSON 数据", json.dumps(hex_list, indent=4), "color_palette.json")
    
    if gen_aco:
        buf = io.BytesIO()
        buf.write(struct.pack('>HH', 1, len(colors_prop)))
        for c in colors_prop:
            buf.write(struct.pack('>HHHHH', 0, int(c[0]*255)*257, int(c[1]*255)*257, int(c[2]*255)*257, 0))
        col_d2.download_button("📥 下载 Adobe Photoshop 色板 (.aco)", buf.getvalue(), "palette.aco")
