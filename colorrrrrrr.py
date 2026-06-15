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
import plotly.graph_objects as go

# --- 全局紧凑样式注入 ---
st.markdown("""
    <style>
    html, body, [data-testid="stMarkdownContainer"] { font-size: 0.85rem !important; }
    h1 { font-size: 1.6rem !important; font-weight: 700; padding-top: 0px; }
    h2 { font-size: 1.2rem !important; margin-top: 10px; }
    h3 { font-size: 1.0rem !important; }
    .stSlider, .stCheckbox { padding: 0px !important; margin: 0px !important; }
    div[data-testid="stBlock"] { padding: 5px !important; }
    </style>
""", unsafe_allow_html=True)

# --- 核心算法引擎 ---
def analyze_image(img_obj, threshold=0.001, n_clusters=25):
    img_rgb = cv2.cvtColor(img_obj, cv2.COLOR_BGR2RGB)
    pixels = cv2.resize(img_rgb, (80, 80)).reshape(-1, 3)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=8).fit(pixels)
    counts = np.bincount(kmeans.labels_)
    proportions = counts / len(kmeans.labels_)
    
    mask = proportions >= threshold
    colors = kmeans.cluster_centers_[mask] / 255.0
    proportions = proportions[mask]
    
    # 1. 占比排序
    sort_idx_prop = np.argsort(proportions)[::-1]
    colors_prop = colors[sort_idx_prop]
    props_prop = proportions[sort_idx_prop]
    
    # 2. 计算焦点色（与其他主色平均色彩距离最远的孤立色）
    focus_idx = 0
    if len(colors_prop) > 1:
        avg_dists = [np.mean([np.linalg.norm(c - other) for j, other in enumerate(colors_prop) if i != j]) for i, c in enumerate(colors_prop)]
        focus_idx = np.argmax(avg_dists)
    focus_color = colors_prop[focus_idx]
    focus_prop = props_prop[focus_idx]
    
    return colors_prop, props_prop, focus_color, focus_prop

# --- UI 渲染界面 ---
st.title("🎨 极致紧凑型专业色彩协同画布")

uploaded_file = st.file_uploader("导入设计资产 (JPG / PNG)...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    # 1. 置顶紧凑控制面板
    st.markdown("### ⚙️ 调控中心")
    c_ctrl1, c_ctrl2, c_ctrl3, c_ctrl4 = st.columns(4)
    with c_ctrl1:
        threshold = st.slider("微量色过滤阈值", 0.0001, 0.03, 0.001, format="%.4f")
    with c_ctrl2:
        clusters = st.slider("色彩融合容差 (聚类中心数)", 5, 60, 24)
    with c_ctrl3:
        exclude_focus = st.checkbox("渐变条剔除低占比焦点色 (<5%)", value=True)
    with c_ctrl4:
        gen_aco = st.checkbox("准备 Adobe .aco 色板下载", value=True)

    # 执行核心色彩分析
    colors_prop, props_prop, focus_color, focus_prop = analyze_image(img, threshold=threshold, n_clusters=clusters)
    dominant_color = colors_prop[0]  # 占比最大的主色
    
    st.divider()
    
    # 2. 核心并排联动层：原图 VS 交互色环
    col_img, col_wheel = st.columns([1, 1])
    
    with col_img:
        st.subheader("🖼️ 原图预览")
        # 适当缩小图片展示尺寸
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=False, width=380)
        st.metric("有效提取色板总数", len(colors_prop))
        
    with col_wheel:
        st.subheader("⭕ 交互式光谱色环 (支持悬停放大与主色对比)")
        
        # 利用 Plotly 构建具备全交互特性的实色填充色环
        fig_json = go.Figure()
        
        # 映射生成背景光谱网格点
        rs = np.linspace(0.1, 1.0, 15)
        thetas = np.linspace(0, 360, 72, endpoint=False)
        bg_theta, bg_r, bg_color = [], [], []
        for r in rs:
            for t in thetas:
                bg_theta.append(t)
                bg_r.append(r)
                bg_color.append(mcolors.to_hex(colorsys.hsv_to_rgb(t/360.0, r, 1.0)))
                
        fig_json.add_trace(go.Scatterpolar(
            r=bg_r, theta=bg_theta, mode='markers',
            marker=dict(size=4, color=bg_color, opacity=0.25),
            hoverinfo='skip', showlegend=False
        ))
        
        # 动态计算并标记主色交互节点
        pts_theta, pts_r, pts_color, pts_hover = [], [], [], []
        for c in colors_prop:
            h, s, v = colorsys.rgb_to_hsv(*c)
            pts_theta.append(h * 360.0)
            pts_r.append(s)
            pts_color.append(mcolors.to_hex(c))
            
            # 计算当前色与占比最大主色的色彩感知差异
            c_diff = np.linalg.norm(c - dominant_color)
            similarity = max(0.0, 100.0 - (c_diff * 50.0))
            
            hover_text = (
                f"<b>十六进制:</b> {mcolors.to_hex(c).upper()}<br>"
                f"<b>RGB 权重:</b> {[int(x*255) for x in c]}<br>"
                f"<b>主色对比相似度:</b> {similarity:.1f}%"
            )
            pts_hover.append(hover_text)
            
        fig_json.add_trace(go.Scatterpolar(
            r=pts_r, theta=pts_theta, mode='markers',
            marker=dict(
                size=14, color=pts_color, line=dict(color='#ffffff', width=2),
                customdata=pts_color
            ),
            text=pts_hover, hovertemplate="%{text}<extra></extra>",
            # 关键：配置鼠标悬停时节点放大预览机制
            hoverlabel=dict(bgcolor="whitesmoke", font_size=11),
            showlegend=False
        ))
        
        # 优化交互动效与视窗尺寸
        fig_json.update_traces(selector=dict(mode='markers+text'), unselected=dict(marker_opacity=0.7))
        fig_json.update_layout(
            width=360, height=360, margin=dict(l=10, r=10, t=10, b=10),
            polar=dict(
                angularaxis=dict(showticklabels=False, ticks='', showgrid=False),
                radialaxis=dict(showticklabels=False, ticks='', showgrid=False)
            ),
            hovermode='closest'
        )
        st.plotly_chart(fig_json, config={'displayModeBar': False})

    st.divider()

    # 3. 垂直同列线性分析面板
    st.subheader("📊 垂直演化色级面板")
    
    # Panel 1: 原始占比排序
    st.markdown("**1. 画面覆盖率原始分配色卡 (按占比由大到小)**")
    fig_m1, ax_m1 = plt.subplots(figsize=(11, 0.5))
    start = 0
    for c, p in zip(colors_prop, props_prop):
        ax_m1.barh(0, p, left=start, color=c, height=1)
        start += p
    ax_m1.axis('off')
    st.pyplot(fig_m1)
    
    # 计算明度排序索引
    lums = np.array([0.299*c[0] + 0.587*c[1] + 0.114*c[2] for c in colors_prop])
    idx_lum_asc = np.argsort(lums)
    colors_lum = colors_prop[idx_lum_asc]
    props_lum = props_prop[idx_lum_asc]
    
    # Panel 2: 保留空间比例宽度的明度排序色卡
    st.markdown("**2. 加权明度梯度色卡 (保留面积占比 ➡️ 依明度由暗至亮重排)**")
    fig_m2, ax_m2 = plt.subplots(figsize=(11, 0.5))
    start_lum = 0
    for c, p in zip(colors_lum, props_lum):
        ax_m2.barh(0, p, left=start_lum, color=c, height=1)
        start_lum += p
    ax_m2.axis('off')
    st.pyplot(fig_m2)
    
    # Panel 3: 等宽离散明度阶梯
    st.markdown("**3. 标准明度等宽离散色卡 (由暗至亮排列)**")
    fig_m3, ax_m3 = plt.subplots(figsize=(11, 0.5))
    n_total = len(colors_lum)
    for i, c in enumerate(colors_lum):
        ax_m3.barh(0, 1/n_total, left=i/n_total, color=c, height=1)
    ax_m3.axis('off')
    st.pyplot(fig_m3)
    
    # Panel 4: 空间平衡连续渐变 (支持智能过滤选项)
    st.markdown("**4. 平滑明度平衡连续渐变条**")
    fig_m4, ax_m4 = plt.subplots(figsize=(11, 0.5))
    
    # 执行智能剔除逻辑
    if exclude_focus and focus_prop < 0.05 and len(colors_lum) > 2:
        # 如果焦点色占比小于5%，从渐变序列中抽离，避免破坏整体平滑过渡
        colors_for_grad = [c for c in colors_lum if not np.array_equal(c, focus_color)]
    else:
        colors_for_grad = colors_lum
        
    cmap_custom = mcolors.LinearSegmentedColormap.from_list("custom_lum", colors_for_grad)
    ax_m4.imshow(np.linspace(0, 1, 1024).reshape(1, -1), aspect='auto', cmap=cmap_custom)
    ax_m4.axis('off')
    st.pyplot(fig_m4)

    st.divider()

    # 4. 数据资产输出
    st.subheader("💾 工业资产导出")
    col_d1, col_d2 = st.columns(2)
    hex_data = [mcolors.to_hex(c) for c in colors_prop]
    col_d1.download_button("📥 导出生产环境 JSON", json.dumps(hex_data, indent=2), "palette.json")
    
    if gen_aco:
        buf = io.BytesIO()
        buf.write(struct.pack('>HH', 1, len(colors_prop)))
        for c in colors_prop:
            buf.write(struct.pack('>HHHHH', 0, int(c[0]*255)*257, int(c[1]*255)*257, int(c[2]*255)*257, 0))
        col_d2.download_button("📥 导出 Photoshop (.aco) 色板", buf.getvalue(), "palette.aco")
