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
import plotly.express as px
import os
import math

# ═══════════════════════════════════════════════════════════
# 页面配置
# ═══════════════════════════════════════════════════════════
st.set_page_config(page_title="色彩协同画布", layout="wide")

# --- 全局紧凑样式注入 ---
st.markdown("""
    <style>
    /* 全局基础字号 */
    html, body, [data-testid="stMarkdownContainer"], .stMarkdown, .stText, .stCodeBlock {
        font-size: 0.95rem !important;
        line-height: 1.4 !important;
    }
    h1 { font-size: 1.8rem !important; font-weight: 700; padding-top: 0px; margin-bottom: 0.5rem; }
    h2 { font-size: 1.3rem !important; margin-top: 8px; margin-bottom: 0.4rem; }
    h3 { font-size: 1.1rem !important; margin-top: 6px; margin-bottom: 0.3rem; }
    .stSlider, .stCheckbox { padding: 0px !important; margin: 0px !important; }
    div[data-testid="stBlock"] { padding: 4px !important; }
    
    /* 侧边栏紧凑 */
    [data-testid="stSidebar"] .block-container {
        padding: 1rem 0.8rem !important;
        font-size: 0.88rem !important;
    }
    [data-testid="stSidebar"] h3 { font-size: 0.95rem !important; margin: 6px 0 4px 0 !important; }
    [data-testid="stSidebar"] .stRadio > div { padding: 2px 0 !important; }
    [data-testid="stSidebar"] .stSlider { padding: 0 !important; margin: -2px 0 !important; }
    [data-testid="stSidebar"] .stCheckbox { padding: 0 !important; margin: -2px 0 !important; }
    [data-testid="stSidebar"] .stButton { padding: 2px 0 !important; }
    [data-testid="stSidebar"] hr { margin: 6px 0 !important; }
    [data-testid="stSidebar"] .stTextInput { padding: 0 !important; margin: -2px 0 !important; }
    
    /* 色板卡片 */
    .compact-card {
        background: rgba(128,128,128,0.03);
        border-radius: 6px;
        padding: 6px 8px;
        margin-bottom: 8px;
        border: 1px solid rgba(128,128,128,0.05);
        transition: all 0.1s ease;
    }
    .compact-card:hover {
        background: rgba(128,128,128,0.08);
        border-color: rgba(128,128,128,0.15);
    }
    .color-swatch {
        width: 28px; height: 22px; border-radius: 4px;
        border: 1px solid rgba(0,0,0,0.1); flex-shrink: 0;
    }
    .color-code {
        font-family: monospace; font-size: 0.75rem;
        background: rgba(0,0,0,0.03); padding: 1px 4px; border-radius: 3px;
    }
    .color-percent { font-weight: 600; font-size: 0.8rem; white-space: nowrap; }
    .color-meta {
        font-size: 0.7rem; color: #888; margin-top: 4px;
        border-top: 1px dashed rgba(128,128,128,0.2); padding-top: 4px;
        display: flex; justify-content: space-between;
    }
    .flex-row { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
    .right-group { margin-left: auto; display: flex; align-items: center; gap: 8px; }
    </style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# Session State
# ═══════════════════════════════════════════════════════════
if "wheel_reset_counter" not in st.session_state:
    st.session_state.wheel_reset_counter = 0
if "wheel_display_mode" not in st.session_state:
    st.session_state.wheel_display_mode = "仅色相环"

def reset_wheel_view():
    st.session_state.wheel_reset_counter += 1

# --- 核心算法 ---
@st.cache_data
def analyze_image(img_resized, threshold=0.001, n_clusters=25):
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

# ═══════════════════════════════════════════════════════════
# 色环构建函数 (悬停放大色块 + 动态白边)
# ═══════════════════════════════════════════════════════════
def build_hue_wheel(colors_prop, props_prop, dominant_color, focus_main, wheel_size, uirev, dot_size=18):
    fig = go.Figure()
    n_rings, n_sectors = 30, 180
    dr, dtheta = 1.0/n_rings, 360.0/n_sectors
    for ring in range(n_rings):
        r_inner = ring*dr + 0.02
        r_mid = r_inner + dr/2
        thetas, rs, bar_colors, widths = [], [], [], []
        for sec in range(n_sectors):
            theta_mid = sec*dtheta + dtheta/2
            h = theta_mid/360.0
            s = r_mid
            thetas.append(theta_mid); rs.append(dr)
            r,g,b = colorsys.hsv_to_rgb(h,s,1.0)
            bar_colors.append(mcolors.to_hex((r,g,b)))
            widths.append(dtheta)
        fig.add_trace(go.Barpolar(
            r=rs, theta=thetas, width=widths, base=r_inner,
            marker=dict(color=bar_colors, line=dict(width=0)),
            opacity=1.0, hoverinfo='skip', showlegend=False
        ))
    pts_theta, pts_r, pts_color, pts_hover = [], [], [], []
    for c,p in zip(colors_prop, props_prop):
        h,s,v = colorsys.rgb_to_hsv(*c)
        pts_theta.append(h*360.0); pts_r.append(s)
        hex_code = mcolors.to_hex(c).upper()
        pts_color.append(hex_code)
        c_diff = np.linalg.norm(c - dominant_color)
        similarity = max(0.0, 100.0 - (c_diff*50.0))
        hover_text = f"""
        <div style="display: flex; align-items: center; gap: 12px; min-width: 220px;">
            <div style="width: 56px; height: 56px; background: {hex_code}; border-radius: 8px; border: 1px solid #aaa;"></div>
            <div>
                <b>HEX:</b> {hex_code}<br>
                <b>RGB:</b> {[int(x*255) for x in c]}<br>
                <b>占比:</b> {p*100:.2f}%<br>
                <b>主色相似度:</b> {similarity:.1f}%
            </div>
        </div>
        """
        pts_hover.append(hover_text)
    line_width = max(1, min(4, int(dot_size/6)))   # 白边随色点大小变化
    fig.add_trace(go.Scatterpolar(
        r=pts_r, theta=pts_theta, mode='markers',
        marker=dict(size=dot_size, color=pts_color, line=dict(color='white', width=line_width)),
        text=pts_hover, hovertemplate="%{text}<extra></extra>",
        hoverlabel=dict(bgcolor="rgba(0,0,0,0.8)", font_size=12, font_color="white"),
        showlegend=False
    ))
    if focus_main and len(pts_theta)>0:
        h_dom = colorsys.rgb_to_hsv(*dominant_color)[0]
        ang = h_dom*360.0
        fig.update_layout(polar=dict(
            angularaxis=dict(rotation=90-ang, direction="clockwise", showticklabels=False, ticks='', showgrid=False),
            radialaxis=dict(range=[0,1.05], showticklabels=False, ticks='', showgrid=False)
        ), uirevision=uirev)
    else:
        fig.update_layout(polar=dict(
            angularaxis=dict(showticklabels=False, ticks='', showgrid=False),
            radialaxis=dict(range=[0,1.05], showticklabels=False, ticks='', showgrid=False)
        ), uirevision=uirev)
    fig.update_layout(width=wheel_size, height=wheel_size, margin=dict(l=10,r=10,t=10,b=10),
                      hovermode='closest', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig

def build_value_wheel(colors_prop, props_prop, dominant_color, focus_main, wheel_size, uirev, dot_size=18):
    fig = go.Figure()
    h_base = colorsys.rgb_to_hsv(*dominant_color)[0]
    n_rings, n_sectors = 30, 180
    dr, dtheta = 1.0/n_rings, 360.0/n_sectors
    for ring in range(n_rings):
        r_inner = ring*dr + 0.02
        r_mid = r_inner + dr/2
        v = r_mid
        thetas, rs, bar_colors, widths = [], [], [], []
        for sec in range(n_sectors):
            theta_mid = sec*dtheta + dtheta/2
            s = theta_mid/360.0
            thetas.append(theta_mid); rs.append(dr)
            r,g,b = colorsys.hsv_to_rgb(h_base, s, v)
            bar_colors.append(mcolors.to_hex((r,g,b)))
            widths.append(dtheta)
        fig.add_trace(go.Barpolar(
            r=rs, theta=thetas, width=widths, base=r_inner,
            marker=dict(color=bar_colors, line=dict(width=0)),
            opacity=1.0, hoverinfo='skip', showlegend=False
        ))
    pts_theta, pts_r, pts_color, pts_hover = [], [], [], []
    for c,p in zip(colors_prop, props_prop):
        h,s,v = colorsys.rgb_to_hsv(*c)
        pts_theta.append(s*360.0); pts_r.append(v)
        hex_code = mcolors.to_hex(c).upper()
        pts_color.append(hex_code)
        c_diff = np.linalg.norm(c - dominant_color)
        similarity = max(0.0, 100.0 - (c_diff*50.0))
        hover_text = f"""
        <div style="display: flex; align-items: center; gap: 12px; min-width: 220px;">
            <div style="width: 56px; height: 56px; background: {hex_code}; border-radius: 8px; border: 1px solid #aaa;"></div>
            <div>
                <b>HEX:</b> {hex_code}<br>
                <b>RGB:</b> {[int(x*255) for x in c]}<br>
                <b>占比:</b> {p*100:.2f}%<br>
                <b>主色相似度:</b> {similarity:.1f}%
            </div>
        </div>
        """
        pts_hover.append(hover_text)
    line_width = max(1, min(4, int(dot_size/6)))
    fig.add_trace(go.Scatterpolar(
        r=pts_r, theta=pts_theta, mode='markers',
        marker=dict(size=dot_size, color=pts_color, line=dict(color='white', width=line_width)),
        text=pts_hover, hovertemplate="%{text}<extra></extra>",
        hoverlabel=dict(bgcolor="rgba(0,0,0,0.8)", font_size=12, font_color="white"),
        showlegend=False
    ))
    if focus_main and len(pts_theta)>0:
        _, s_dom, v_dom = colorsys.rgb_to_hsv(*dominant_color)
        ang = s_dom*360.0
        fig.update_layout(polar=dict(
            angularaxis=dict(rotation=90-ang, direction="clockwise", showticklabels=False, ticks='', showgrid=False),
            radialaxis=dict(range=[0,1.05], showticklabels=False, ticks='', showgrid=False)
        ), uirevision=uirev)
    else:
        fig.update_layout(polar=dict(
            angularaxis=dict(showticklabels=False, ticks='', showgrid=False),
            radialaxis=dict(range=[0,1.05], showticklabels=False, ticks='', showgrid=False)
        ), uirevision=uirev)
    fig.update_layout(width=wheel_size, height=wheel_size, margin=dict(l=10,r=10,t=10,b=10),
                      hovermode='closest', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig

# ═══════════════════════════════════════════════════════════
# 交互式色级面板 (Plotly 堆叠条形图，支持悬停预览)
# ═══════════════════════════════════════════════════════════
def make_stacked_bar(colors, proportions, title, xlabel="占比"):
    """生成水平堆叠条形图，每个色块独立悬停"""
    fig = go.Figure()
    cumulative = 0.0
    for c, p in zip(colors, proportions):
        hex_c = mcolors.to_hex(c).upper()
        rgb_vals = [int(x*255) for x in c]
        hover = f"""
        <div style="display:flex; align-items:center; gap:10px;">
            <div style="width:40px; height:40px; background:{hex_c}; border-radius:4px;"></div>
            <div>
                <b>HEX:</b> {hex_c}<br>
                <b>RGB:</b> {rgb_vals}<br>
                <b>占比:</b> {p*100:.2f}%
            </div>
        </div>
        """
        fig.add_trace(go.Bar(
            x=[p], y=[0], base=cumulative, orientation='h',
            marker=dict(color=hex_c, line=dict(width=0)),
            name=hex_c, showlegend=False, hovertemplate=hover+"<extra></extra>"
        ))
        cumulative += p
    fig.update_layout(barmode='stack', height=80, margin=dict(l=0,r=0,t=30,b=0),
                      xaxis_title=xlabel, yaxis=dict(showticklabels=False, showgrid=False, visible=False),
                      plot_bgcolor='rgba(0,0,0,0)', hoverlabel=dict(bgcolor="white", font_size=11))
    return fig

def make_equal_width_bar(colors, title):
    """等宽色块，每个宽度相同"""
    n = len(colors)
    width = 1.0 / n if n > 0 else 0
    fig = go.Figure()
    cumulative = 0.0
    for i, c in enumerate(colors):
        hex_c = mcolors.to_hex(c).upper()
        rgb_vals = [int(x*255) for x in c]
        hover = f"""
        <div style="display:flex; align-items:center; gap:10px;">
            <div style="width:40px; height:40px; background:{hex_c}; border-radius:4px;"></div>
            <div>
                <b>HEX:</b> {hex_c}<br>
                <b>RGB:</b> {rgb_vals}<br>
                <b>位置:</b> {i+1}/{n}
            </div>
        </div>
        """
        fig.add_trace(go.Bar(
            x=[width], y=[0], base=cumulative, orientation='h',
            marker=dict(color=hex_c, line=dict(width=0)),
            name=hex_c, showlegend=False, hovertemplate=hover+"<extra></extra>"
        ))
        cumulative += width
    fig.update_layout(barmode='stack', height=80, margin=dict(l=0,r=0,t=30,b=0),
                      xaxis=dict(range=[0,1]), yaxis=dict(showticklabels=False, showgrid=False, visible=False),
                      plot_bgcolor='rgba(0,0,0,0)', hoverlabel=dict(bgcolor="white", font_size=11))
    return fig

# --- UI 渲染 ---
st.title("色彩协同画布")
uploaded_file = st.file_uploader("导入设计资产 (JPG / PNG)...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    img_resized = img.resize((80,80))
    default_img_name = os.path.splitext(uploaded_file.name)[0]

    # 侧边栏（仅保留基础参数）
    st.sidebar.markdown("**调控中心**")
    palette_name = st.sidebar.text_input("色板名称", value=default_img_name)
    col1, col2 = st.sidebar.columns(2)
    with col1:
        threshold = st.slider("过滤阈值", 0.0001, 0.03, 0.001, format="%.4f")
    with col2:
        clusters = st.slider("聚类数", 5, 60, 24)
    col3, col4 = st.sidebar.columns(2)
    with col3:
        exclude_focus = st.checkbox("剔除低占比焦点色", value=True)
    with col4:
        gen_aco = st.checkbox("生成 .aco 文件", value=True)
    st.sidebar.markdown("<hr>", unsafe_allow_html=True)
    st.sidebar.image(img, use_container_width=True)
    st.sidebar.metric("提取色板数", None)  # placeholder, 稍后更新

    # 分析图像
    colors_prop, props_prop, focus_color, focus_prop = analyze_image(img_resized, threshold=threshold, n_clusters=clusters)
    dominant_color = colors_prop[0]
    st.sidebar.metric("提取色板数", len(colors_prop))

    # ═══════════════════════════════════════════════════════════
    # 主体色环控制栏（集成展示模式、聚焦主色、色点大小、恢复默认）
    # ═══════════════════════════════════════════════════════════
    st.markdown("### 色彩环视")
    control_cols = st.columns([2,2,2,1])
    with control_cols[0]:
        wheel_display_mode = st.radio(
            "展示模式", ["仅色相环", "仅明度环", "色相+明度并列"],
            horizontal=True, label_visibility="collapsed"
        )
    with control_cols[1]:
        focus_main = st.checkbox("聚焦主色区域", value=False)
    with control_cols[2]:
        dot_size = st.slider("色点大小", 8, 24, 14, label_visibility="collapsed")
    with control_cols[3]:
        st.button("恢复默认视图", on_click=reset_wheel_view, use_container_width=True)

    uirev = str(st.session_state.wheel_reset_counter)
    WHEEL_SIZE = 560

    if wheel_display_mode == "仅色相环":
        fig_hue = build_hue_wheel(colors_prop, props_prop, dominant_color, focus_main, WHEEL_SIZE, uirev, dot_size)
        st.plotly_chart(fig_hue, config={'displayModeBar': False}, use_container_width=False)
    elif wheel_display_mode == "仅明度环":
        fig_val = build_value_wheel(colors_prop, props_prop, dominant_color, focus_main, WHEEL_SIZE, uirev, dot_size)
        st.plotly_chart(fig_val, config={'displayModeBar': False}, use_container_width=False)
    else:  # 并列
        col_h, col_v = st.columns(2)
        with col_h:
            st.markdown("<div style='text-align:center; font-weight:600'>色相环 (H-S)</div>", unsafe_allow_html=True)
            fig_hue = build_hue_wheel(colors_prop, props_prop, dominant_color, focus_main, WHEEL_SIZE, uirev, dot_size)
            st.plotly_chart(fig_hue, config={'displayModeBar': False}, use_container_width=False)
        with col_v:
            st.markdown("<div style='text-align:center; font-weight:600'>明度环 (S-V)</div>", unsafe_allow_html=True)
            fig_val = build_value_wheel(colors_prop, props_prop, dominant_color, focus_main, WHEEL_SIZE, uirev, dot_size)
            st.plotly_chart(fig_val, config={'displayModeBar': False}, use_container_width=False)

    st.divider()

    # ═══════════════════════════════════════════════════════════
    # 演化色级面板 (交互式)
    # ═══════════════════════════════════════════════════════════
    st.subheader("演化色级面板")
    # 准备明度排序数据
    lums = np.array([0.299*c[0] + 0.587*c[1] + 0.114*c[2] for c in colors_prop])
    idx_lum_asc = np.argsort(lums)
    colors_lum = colors_prop[idx_lum_asc]
    props_lum = props_prop[idx_lum_asc]

    # 剔除低占比焦点色（用于连续渐变）
    if exclude_focus and focus_prop < 0.05 and len(colors_lum) > 2:
        colors_for_grad = [c for c in colors_lum if not np.array_equal(c, focus_color)]
    else:
        colors_for_grad = colors_lum

    # 四个面板
    fig1 = make_stacked_bar(colors_prop, props_prop, "覆盖率原始分配", "占比")
    fig2 = make_stacked_bar(colors_lum, props_lum, "加权明度梯度", "占比")
    fig3 = make_equal_width_bar(colors_lum, "标准明度等宽离散")
    # 连续渐变仍使用 matplotlib (支持平滑渐变，无需悬停预览)
    fig4, ax4 = plt.subplots(figsize=(10, 0.5))
    fig4.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax4.axis('off')
    cmap_custom = mcolors.LinearSegmentedColormap.from_list("custom_lum", colors_for_grad)
    ax4.imshow(np.linspace(0,1,1024).reshape(1,-1), aspect='auto', cmap=cmap_custom, extent=[0,1,-0.5,0.5])

    # 按标签页或全览展示
    view_mode = st.radio("显示模式", ["标签页", "全览"], horizontal=True, label_visibility="collapsed")
    if view_mode == "标签页":
        t1,t2,t3,t4 = st.tabs(["覆盖率原始分配", "加权明度梯度", "标准明度等宽离散", "连续渐变"])
        with t1:
            st.plotly_chart(fig1, use_container_width=True, config={'displayModeBar': False})
        with t2:
            st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})
        with t3:
            st.plotly_chart(fig3, use_container_width=True, config={'displayModeBar': False})
        with t4:
            st.pyplot(fig4)
    else:
        st.markdown("**1. 覆盖率原始分配**")
        st.plotly_chart(fig1, use_container_width=True, config={'displayModeBar': False})
        st.markdown("**2. 加权明度梯度**")
        st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})
        st.markdown("**3. 标准明度等宽离散**")
        st.plotly_chart(fig3, use_container_width=True, config={'displayModeBar': False})
        st.markdown("**4. 平滑明度平衡连续渐变**")
        st.pyplot(fig4)

    st.divider()
    st.subheader("色板数据总览")

    # 三列紧凑卡片 (不变)
    card_data = []
    for c,p in zip(colors_prop, props_prop):
        hex_code = mcolors.to_hex(c).upper()
        rgb_str = f"{int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)}"
        c_diff = np.linalg.norm(c - dominant_color)
        similarity = max(0.0, 100.0 - (c_diff*50.0))
        card_data.append({
            "hex": hex_code, "rgb": rgb_str, "prop": p*100,
            "similarity": similarity, "distance": c_diff, "color": hex_code
        })
    chunk_size = math.ceil(len(card_data)/3)
    chunks = [card_data[i:i+chunk_size] for i in range(0,len(card_data),chunk_size)]
    while len(chunks) < 3:
        chunks.append([])
    cols = st.columns(3)
    for col_idx, col in enumerate(cols):
        with col:
            for item in chunks[col_idx]:
                st.markdown(f"""
                <div class="compact-card">
                    <div class="flex-row">
                        <div class="color-swatch" style="background:{item['color']};"></div>
                        <span class="color-code">{item['hex']}</span>
                        <span class="color-code">rgb({item['rgb']})</span>
                        <div class="right-group"><span class="color-percent">{item['prop']:.2f}%</span></div>
                    </div>
                    <div class="color-meta">
                        <span>相似度 {item['similarity']:.1f}%</span>
                        <span>距离 {item['distance']:.3f}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    st.divider()
    st.subheader("资产导出")
    col_d1, col_d2 = st.columns(2)
    structured_json = {
        "palette_group": palette_name,
        "colors": [{"name": f"{mcolors.to_hex(c).upper()} ({p*100:.1f}%)",
                    "hex": mcolors.to_hex(c).upper(),
                    "rgb": [int(x*255) for x in c],
                    "proportion": float(p)} for c,p in zip(colors_prop, props_prop)]
    }
    col_d1.download_button("导出 JSON", json.dumps(structured_json, indent=2), f"{palette_name}_palette.json")
    if gen_aco:
        buf = io.BytesIO()
        n_colors = len(colors_prop)
        buf.write(struct.pack('>HH', 1, n_colors))
        for c in colors_prop:
            r,g,b = [int(x*65535) for x in c]
            buf.write(struct.pack('>HHHHH', 0, r, g, b, 0))
        buf.write(struct.pack('>HH', 2, n_colors))
        for c,p in zip(colors_prop, props_prop):
            r,g,b = [int(x*65535) for x in c]
            buf.write(struct.pack('>HHHHH', 0, r, g, b, 0))
            hex_code = mcolors.to_hex(c).upper()
            color_name = f"{hex_code} ({p*100:.1f}%)"
            name_bytes = color_name.encode('utf-16-be') + b'\x00\x00'
            buf.write(struct.pack('>I', len(color_name)+1))
            buf.write(name_bytes)
        col_d2.download_button("导出 Photoshop (.aco)", buf.getvalue(), f"{palette_name}_swatches.aco")
