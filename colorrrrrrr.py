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
import math

# --- 全局紧凑样式注入 ---
st.markdown("""
    <style>
    html, body, [data-testid="stMarkdownContainer"] { font-size: 0.85rem !important; }
    h1 { font-size: 1.6rem !important; font-weight: 700; padding-top: 0px; }
    h2 { font-size: 1.2rem !important; margin-top: 10px; }
    h3 { font-size: 1.0rem !important; }
    .stSlider, .stCheckbox { padding: 0px !important; margin: 0px !important; }
    div[data-testid="stBlock"] { padding: 5px !important; }
    .palette-card {
        border-radius:12px;
        padding:10px;
        border:1px solid rgba(128,128,128,.2);
    }
    .color-preview-box {
        width: 30px; 
        height: 20px; 
        border-radius: 4px; 
        border: 1px solid #ddd;
        display: inline-block;
    }
    </style>
""", unsafe_allow_html=True)


# --- 布局模式选择 ---
layout_mode = st.sidebar.radio(
    "🖥️ 显示布局",
    ["12.4寸平板模式", "15.6寸电脑模式"],
    index=0
)

if layout_mode == "12.4寸平板模式":
    PREVIEW_WIDTH = 420
    WHEEL_SIZE = 420
    PANEL_RATIO = [1, 1]
    st.set_page_config(layout="centered")
else:
    PREVIEW_WIDTH = 650
    WHEEL_SIZE = 650
    st.set_page_config(layout="wide")


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
    
    default_img_name = os.path.splitext(uploaded_file.name)[0]
    
    # 1. 置顶紧凑控制面板
    st.sidebar.markdown("### ⚙️ 调控中心")
    palette_name = st.sidebar.text_input("📝 色板组名称", value=default_img_name)
    
    c_ctrl1, c_ctrl2 = st.sidebar.columns(2)
    c_ctrl3, c_ctrl4 = st.sidebar.columns(2)
    with c_ctrl1:
        threshold = st.slider("微量色过滤阈值", 0.0001, 0.03, 0.001, format="%.4f")
    with c_ctrl2:
        clusters = st.slider("色彩融合容差 (聚类数)", 5, 60, 24)
    with c_ctrl3:
        exclude_focus = st.checkbox("渐变条剔除低占比焦点色 (<5%)", value=True)
    with c_ctrl4:
        gen_aco = st.checkbox("准备 Adobe .aco 下载", value=True)

    colors_prop, props_prop, focus_color, focus_prop = analyze_image(img, threshold=threshold, n_clusters=clusters)
    dominant_color = colors_prop[0]  
    
    st.divider()
    
    # 2. 核心联动层：布局控制
    if layout_mode == "15.6寸电脑模式":
        with st.sidebar:
            st.markdown("---")
            st.subheader("🖼️ 原图预览")
            st.image(img, use_container_width=True)
            st.metric("有效提取色板总数", len(colors_prop))
        # 电脑模式下主界面直接显示色环，去掉原图
        col_wheel = st.container()
    else:
        col_img, col_wheel = st.columns(PANEL_RATIO)
        with col_img:
            st.subheader("🖼️ 原图预览")
            st.image(img, width=PREVIEW_WIDTH)
            st.metric("有效提取色板总数", len(colors_prop))
        
    with col_wheel:
        st.subheader("⭕ 3D 交互式光谱色柱 (含明度 Z 轴)")
        fig_json = go.Figure()
        
        # 绘制背景色盘 (放置在 Z=1 的最高明度层)
        rs = np.linspace(0.05, 1.0, 15) 
        thetas = np.linspace(0, 360, 90, endpoint=False) 
        bg_x, bg_y, bg_z, bg_color = [], [], [], []
        for r in rs:
            for t in thetas:
                bg_x.append(r * math.cos(math.radians(t)))
                bg_y.append(r * math.sin(math.radians(t)))
                bg_z.append(1.0) # 背景盘在最上方
                bg_color.append(mcolors.to_hex(colorsys.hsv_to_rgb(t/360.0, r, 1.0)))
                
        fig_json.add_trace(go.Scatter3d(
            x=bg_x, y=bg_y, z=bg_z, mode='markers',
            marker=dict(size=4, color=bg_color, opacity=0.15),
            hoverinfo='skip', showlegend=False
        ))
        
        # 绘制中心明度轴 (从黑到白)
        axis_z = np.linspace(0, 1, 20)
        fig_json.add_trace(go.Scatter3d(
            x=np.zeros_like(axis_z), y=np.zeros_like(axis_z), z=axis_z, 
            mode='markers+lines',
            line=dict(color='gray', width=2),
            marker=dict(size=3, color=[mcolors.to_hex((v, v, v)) for v in axis_z]),
            hoverinfo='skip', showlegend=False
        ))
        
        # 绘制提取出的颜色 3D 散点
        pts_x, pts_y, pts_z, pts_color, pts_hover = [], [], [], [], []
        for c in colors_prop:
            h, s, v = colorsys.rgb_to_hsv(*c)
            x = s * math.cos(h * 2 * math.pi)
            y = s * math.sin(h * 2 * math.pi)
            z = v
            pts_x.append(x)
            pts_y.append(y)
            pts_z.append(z)
            pts_color.append(mcolors.to_hex(c))
            
            c_diff = np.linalg.norm(c - dominant_color)
            similarity = max(0.0, 100.0 - (c_diff * 50.0))
            hover_text = (
                f"<b>十六进制:</b> {mcolors.to_hex(c).upper()}<br>"
                f"<b>RGB:</b> {[int(x*255) for x in c]}<br>"
                f"<b>HSV(明度):</b> {v:.2f}<br>"
                f"<b>主色相似度:</b> {similarity:.1f}%"
            )
            pts_hover.append(hover_text)
            
            # 添加向底部的投影线，增强立体感
            fig_json.add_trace(go.Scatter3d(
                x=[x, x], y=[y, y], z=[0, z],
                mode='lines', line=dict(color=mcolors.to_hex(c), width=3, dash='dot'),
                showlegend=False, hoverinfo='skip'
            ))
            
        fig_json.add_trace(go.Scatter3d(
            x=pts_x, y=pts_y, z=pts_z, mode='markers',
            marker=dict(size=12, color=pts_color, line=dict(color='#ffffff', width=2), opacity=1.0),
            text=pts_hover, hovertemplate="%{text}<extra></extra>",
            hoverlabel=dict(bgcolor="whitesmoke", font_size=11),
            showlegend=False
        ))
        
        fig_json.update_layout(
            width=WHEEL_SIZE, height=WHEEL_SIZE, margin=dict(l=0, r=0, t=0, b=0),
            scene=dict(
                xaxis=dict(title='色相/饱和度 (X)', showticklabels=False, range=[-1.1, 1.1]),
                yaxis=dict(title='色相/饱和度 (Y)', showticklabels=False, range=[-1.1, 1.1]),
                zaxis=dict(title='明度轴 (Value)', range=[0, 1.1]),
                camera=dict(eye=dict(x=1.3, y=1.3, z=1.0)) # 默认倾斜视角
            ),
            hovermode='closest'
        )
        # 将 displayModeBar 设为 True，右上角会出现原生菜单，支持“恢复默认(Reset Camera)”
        st.plotly_chart(fig_json, config={'displayModeBar': True}, use_container_width=True)

    st.divider()

    # 3. 垂直同列线性分析面板 (修正：放进 tabs 内确保等宽)
    st.subheader("📊 垂直演化色级面板")
    tab1, tab2, tab3, tab4 = st.tabs(["覆盖率色卡", "明度加权", "等宽色卡", "连续渐变"])
    
    with tab1:
        st.markdown("**1. 画面覆盖率原始分配色卡 (按占比由大到小)**")
        fig_m1, ax_m1 = plt.subplots(figsize=(11, 0.6))
        start = 0
        for c, p in zip(colors_prop, props_prop):
            ax_m1.barh(0, p, left=start, color=c, height=1)
            start += p
        ax_m1.axis('off')
        st.pyplot(fig_m1, use_container_width=True)
    
    lums = np.array([0.299*c[0] + 0.587*c[1] + 0.114*c[2] for c in colors_prop])
    idx_lum_asc = np.argsort(lums)
    colors_lum = colors_prop[idx_lum_asc]
    props_lum = props_prop[idx_lum_asc]
    
    with tab2:
        st.markdown("**2. 加权明度梯度色卡 (保留面积占比 ➡️ 依明度由暗至亮重排)**")
        fig_m2, ax_m2 = plt.subplots(figsize=(11, 0.6))
        start_lum = 0
        for c, p in zip(colors_lum, props_lum):
            ax_m2.barh(0, p, left=start_lum, color=c, height=1)
            start_lum += p
        ax_m2.axis('off')
        st.pyplot(fig_m2, use_container_width=True)
    
    with tab3:
        st.markdown("**3. 标准明度等宽离散色卡 (由暗至亮排列)**")
        fig_m3, ax_m3 = plt.subplots(figsize=(11, 0.6))
        n_total = len(colors_lum)
        for i, c in enumerate(colors_lum):
            ax_m3.barh(0, 1/n_total, left=i/n_total, color=c, height=1)
        ax_m3.axis('off')
        st.pyplot(fig_m3, use_container_width=True)
    
    with tab4:
        st.markdown("**4. 平滑明度平衡连续渐变条**")
        fig_m4, ax_m4 = plt.subplots(figsize=(11, 0.6))
        if exclude_focus and focus_prop < 0.05 and len(colors_lum) > 2:
            colors_for_grad = [c for c in colors_lum if not np.array_equal(c, focus_color)]
        else:
            colors_for_grad = colors_lum
        cmap_custom = mcolors.LinearSegmentedColormap.from_list("custom_lum", colors_for_grad)
        ax_m4.imshow(np.linspace(0, 1, 1024).reshape(1, -1), aspect='auto', cmap=cmap_custom)
        ax_m4.axis('off')
        st.pyplot(fig_m4, use_container_width=True)

    
    st.divider()
    st.subheader("🎯 色板数据总览 (含预览)")

    # 生成自定义 HTML 列表以支持色卡预览
    html_table = """
    <table style="width:100%; text-align:left; border-collapse: collapse;">
        <tr style="border-bottom: 1px solid #ddd; background-color: rgba(128,128,128,0.1);">
            <th style="padding: 8px;">颜色预览</th>
            <th style="padding: 8px;">HEX</th>
            <th style="padding: 8px;">RGB</th>
            <th style="padding: 8px;">面积占比</th>
        </tr>
    """
    for c, p in zip(colors_prop, props_prop):
        hex_code = mcolors.to_hex(c).upper()
        rgb_code = f"{int(c[0]*255)}, {int(c[1]*255)}, {int(c[2]*255)}"
        prop_str = f"{p*100:.2f}%"
        html_table += f"""
        <tr style="border-bottom: 1px solid #eee;">
            <td style="padding: 8px;"><div class="color-preview-box" style="background-color: {hex_code};"></div></td>
            <td style="padding: 8px; font-family: monospace;">{hex_code}</td>
            <td style="padding: 8px; font-family: monospace;">{rgb_code}</td>
            <td style="padding: 8px;">{prop_str}</td>
        </tr>
        """
    html_table += "</table>"
    
    st.markdown(html_table, unsafe_allow_html=True)

    st.divider()

    # --- 4. 数据资产输出 ---
    st.subheader("💾 工业资产导出")
    col_d1, col_d2 = st.columns(2)
    
    structured_json = {
        "palette_group": palette_name,
        "colors": []
    }
    for c, p in zip(colors_prop, props_prop):
        hex_code = mcolors.to_hex(c).upper()
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
        f"{palette_name}_palette.json" 
    )
    
    if gen_aco:
        buf = io.BytesIO()
        n_colors = len(colors_prop)
        
        buf.write(struct.pack('>HH', 1, n_colors))
        for c in colors_prop:
            r, g, b = [int(x*65535) for x in c]
            buf.write(struct.pack('>HHHHH', 0, r, g, b, 0))
            
        buf.write(struct.pack('>HH', 2, n_colors))
        for c, p in zip(colors_prop, props_prop):
            r, g, b = [int(x*65535) for x in c]
            buf.write(struct.pack('>HHHHH', 0, r, g, b, 0))
            
            hex_code = mcolors.to_hex(c).upper()
            color_name = f"{hex_code} ({p*100:.1f}%)"
            name_bytes = color_name.encode('utf-16-be') + b'\x00\x00'
            
            buf.write(struct.pack('>I', len(color_name) + 1))
            buf.write(name_bytes)
            
        col_d2.download_button(
            "📥 导出 Photoshop (.aco) 色板", 
            buf.getvalue(), 
            f"{palette_name}_swatches.aco" 
        )
