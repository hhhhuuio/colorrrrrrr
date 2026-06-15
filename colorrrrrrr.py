import streamlit as st
from PIL import Image  
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.cluster import KMeans
import colorsys
import json
import io
import struct
import plotly.graph_objects as go
import os

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
    img_resized = img_obj.resize((80, 80))
    pixels = np.array(img_resized).reshape(-1, 3)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=8).fit(pixels)
    counts = np.bincount(kmeans.labels_)
    proportions = counts / len(kmeans.labels_)
    
    mask = proportions >= threshold
    colors = kmeans.cluster_centers_[mask] / 255.0
    proportions = proportions[mask]
    
    sort_idx_prop = np.argsort(proportions)[::-1]
    colors_prop = colors[sort_idx_prop]
    props_prop = proportions[sort_idx_prop]
    
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
    img = Image.open(uploaded_file).convert('RGB')
    
    # 提取默认图片名称 (去除后缀)
    default_img_name = os.path.splitext(uploaded_file.name)[0]
    
    # 1. 置顶紧凑控制面板
    st.markdown("### ⚙️ 调控中心")
    
    # 【新增】色板组命名输入框
    palette_name = st.text_input("📝 色板组名称 (默认使用图片名，支持修改)", value=default_img_name)
    
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
    dominant_color = colors_prop[0]  
    
    st.divider()
    
    # 2. 核心并排联动层：原图 VS 交互色环
    col_img, col_wheel = st.columns([1, 1])
    
    with col_img:
        st.subheader("🖼️ 原图预览")
        st.image(img, use_container_width=False, width=380)
        st.metric("有效提取色板总数", len(colors_prop))
        
    with col_wheel:
        st.subheader("⭕ 交互式光谱色环")
        fig_json = go.Figure()
        
        rs = np.linspace(0.05, 1.0, 25) 
        thetas = np.linspace(0, 360, 180, endpoint=False) 
        bg_theta, bg_r, bg_color = [], [], []
        for r in rs:
            for t in thetas:
                bg_theta.append(t)
                bg_r.append(r)
                bg_color.append(mcolors.to_hex(colorsys.hsv_to_rgb(t/360.0, r, 1.0)))
                
        fig_json.add_trace(go.Scatterpolar(
            r=bg_r, theta=bg_theta, mode='markers',
            marker=dict(size=10, color=bg_color, opacity=1.0),
            hoverinfo='skip', showlegend=False
        ))
        
        pts_theta, pts_r, pts_color, pts_hover = [], [], [], []
        for c in colors_prop:
            h, s, v = colorsys.rgb_to_hsv(*c)
            pts_theta.append(h * 360.0)
            pts_r.append(s)
            pts_color.append(mcolors.to_hex(c))
            
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
                size=18, color=pts_color, line=dict(color='#ffffff', width=3)
            ),
            customdata=pts_color,  
            text=pts_hover, hovertemplate="%{text}<extra></extra>",
            hoverlabel=dict(bgcolor="whitesmoke", font_size=11),
            showlegend=False
        ))
        
        fig_json.update_traces(selector=dict(mode='markers'), unselected=dict(marker_opacity=0.9))
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
    
    st.markdown("**1. 画面覆盖率原始分配色卡 (按占比由大到小)**")
    fig_m1, ax_m1 = plt.subplots(figsize=(11, 0.5))
    start = 0
    for c, p in zip(colors_prop, props_prop):
        ax_m1.barh(0, p, left=start, color=c, height=1)
        start += p
    ax_m1.axis('off')
    st.pyplot(fig_m1)
    
    lums = np.array([0.299*c[0] + 0.587*c[1] + 0.114*c[2] for c in colors_prop])
    idx_lum_asc = np.argsort(lums)
    colors_lum = colors_prop[idx_lum_asc]
    props_lum = props_prop[idx_lum_asc]
    
    st.markdown("**2. 加权明度梯度色卡 (保留面积占比 ➡️ 依明度由暗至亮重排)**")
    fig_m2, ax_m2 = plt.subplots(figsize=(11, 0.5))
    start_lum = 0
    for c, p in zip(colors_lum, props_lum):
        ax_m2.barh(0, p, left=start_lum, color=c, height=1)
        start_lum += p
    ax_m2.axis('off')
    st.pyplot(fig_m2)
    
    st.markdown("**3. 标准明度等宽离散色卡 (由暗至亮排列)**")
    fig_m3, ax_m3 = plt.subplots(figsize=(11, 0.5))
    n_total = len(colors_lum)
    for i, c in enumerate(colors_lum):
        ax_m3.barh(0, 1/n_total, left=i/n_total, color=c, height=1)
    ax_m3.axis('off')
    st.pyplot(fig_m3)
    
    st.markdown("**4. 平滑明度平衡连续渐变条**")
    fig_m4, ax_m4 = plt.subplots(figsize=(11, 0.5))
    if exclude_focus and focus_prop < 0.05 and len(colors_lum) > 2:
        colors_for_grad = [c for c in colors_lum if not np.array_equal(c, focus_color)]
    else:
        colors_for_grad = colors_lum
    cmap_custom = mcolors.LinearSegmentedColormap.from_list("custom_lum", colors_for_grad)
    ax_m4.imshow(np.linspace(0, 1, 1024).reshape(1, -1), aspect='auto', cmap=cmap_custom)
    ax_m4.axis('off')
    st.pyplot(fig_m4)

    st.divider()

    # --- 4. 数据资产输出 (全新升级) ---
    st.subheader("💾 工业资产导出")
    col_d1, col_d2 = st.columns(2)
    
    # 构建高规格 JSON
    structured_json = {
        "palette_group": palette_name,
        "colors": []
    }
    for c, p in zip(colors_prop, props_prop):
        hex_code = mcolors.to_hex(c).upper()
        # 色块命名规则：编码 + 比例
        color_name = f"{hex_code} ({p*100:.1f}%)"
        structured_json["colors"].append({
            "name": color_name,
            "hex": hex_code,
            "rgb": [int(x*255) for x in c],
            "proportion": float(p)
        })
    
    col_d1.download_button(
        "📥 导出生产环境 JSON", 
        json.dumps(structured_json, indent=2, ensure_ascii=False), 
        f"{palette_name}_palette.json" # 文件名动态跟随
    )
    
    if gen_aco:
        buf = io.BytesIO()
        n_colors = len(colors_prop)
        
        # 写入 .aco Version 1 (向后兼容无名数据)
        buf.write(struct.pack('>HH', 1, n_colors))
        for c in colors_prop:
            r, g, b = [int(x*65535) for x in c]
            buf.write(struct.pack('>HHHHH', 0, r, g, b, 0))
            
        # 写入 .aco Version 2 (注入动态命名数据)
        buf.write(struct.pack('>HH', 2, n_colors))
        for c, p in zip(colors_prop, props_prop):
            r, g, b = [int(x*65535) for x in c]
            buf.write(struct.pack('>HHHHH', 0, r, g, b, 0))
            
            # 生成色块名称并以 UTF-16-BE 编码 (Photoshop 规范)
            hex_code = mcolors.to_hex(c).upper()
            color_name = f"{hex_code} ({p*100:.1f}%)"
            name_bytes = color_name.encode('utf-16-be') + b'\x00\x00'
            
            # 写入名称长度(含空字符)和名称字节
            buf.write(struct.pack('>I', len(color_name) + 1))
            buf.write(name_bytes)
            
        col_d2.download_button(
            "📥 导出 Photoshop (.aco) 色板", 
            buf.getvalue(), 
            f"{palette_name}_swatches.aco" # 文件名动态跟随
        )
