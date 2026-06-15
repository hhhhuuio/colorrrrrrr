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
import math  # 用于分页计算

# ═══════════════════════════════════════════════════════════
# 页面配置
# ═══════════════════════════════════════════════════════════
st.set_page_config(page_title="色彩协同画布", layout="wide")

# --- 全局紧凑样式注入（移除卡片悬浮预览样式，新增按钮选中样式）---
st.markdown("""
    <style>
    /* 全局基础字号提升 */
    html, body, [data-testid="stMarkdownContainer"], .stMarkdown, .stText, .stCodeBlock {
        font-size: 0.95rem !important;
        line-height: 1.4 !important;
    }
    h1 { font-size: 1.8rem !important; font-weight: 700; padding-top: 0px; margin-bottom: 0.5rem; }
    h2 { font-size: 1.3rem !important; margin-top: 8px; margin-bottom: 0.4rem; }
    h3 { font-size: 1.1rem !important; margin-top: 6px; margin-bottom: 0.3rem; }
    .stSlider, .stCheckbox { padding: 0px !important; margin: 0px !important; }
    div[data-testid="stBlock"] { padding: 4px !important; }
    
    /* 侧边栏极致紧凑但字号稍大 */
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
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] .stSlider label {
        font-size: 0.85rem !important;
    }
    
    /* 新卡片按钮样式（无指针、无虚线、保持紧凑） */
    .color-card-btn {
        width: 100%;
        background: rgba(128,128,128,0.03);
        border: 1px solid rgba(128,128,128,0.05);
        border-radius: 6px;
        padding: 6px 8px;
        margin-bottom: 8px;
        text-align: left;
        cursor: default !important;          /* 无手型指针 */
        transition: all 0.1s ease;
        font-family: inherit;
        font-size: 0.9rem;
    }
    .color-card-btn:hover {
        background: rgba(128,128,128,0.08);
        border-color: rgba(128,128,128,0.15);
    }
    .color-card-btn:focus, .color-card-btn:active {
        outline: none !important;             /* 无虚线轮廓 */
        box-shadow: none !important;
    }
    .selected-color-card {
        background: rgba(128,128,128,0.12);
        border: 2px solid rgba(0,0,0,0.2);
        transform: scale(1.02);
        transition: all 0.2s ease;
    }
    .color-swatch {
        width: 28px;
        height: 22px;
        border-radius: 4px;
        border: 1px solid rgba(0,0,0,0.1);
        flex-shrink: 0;
    }
    .color-code {
        font-family: monospace;
        font-size: 0.75rem;
        background: rgba(0,0,0,0.03);
        padding: 1px 4px;
        border-radius: 3px;
        display: inline-block;
    }
    .color-percent {
        font-weight: 600;
        font-size: 0.8rem;
        color: #1f1f1f;
        white-space: nowrap;
    }
    .flex-row {
        display: flex;
        align-items: center;
        gap: 8px;
        flex-wrap: wrap;
    }
    .right-group {
        margin-left: auto;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    @media (max-width: 900px) {
        .flex-row { gap: 6px; }
        .color-code { font-size: 0.7rem; }
    }
    
    /* 侧边栏选中色块预览放大样式 */
    .selected-preview {
        background: #ffffff;
        border-radius: 12px;
        padding: 12px;
        margin-top: 16px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border: 1px solid #e0e0e0;
    }
    .big-swatch {
        width: 100%;
        height: 120px;
        border-radius: 10px;
        margin-bottom: 10px;
        border: 1px solid rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# Session State 初始化
# ═══════════════════════════════════════════════════════════
if "selected_color" not in st.session_state:
    st.session_state.selected_color = None

# ═══════════════════════════════════════════════════════════
# 核心算法引擎
# ═══════════════════════════════════════════════════════════
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
# 色环构建函数
# ═══════════════════════════════════════════════════════════
def build_hue_wheel(colors_prop, props_prop, dominant_color, focus_main, wheel_size, dot_size=18):
    fig = go.Figure()
    n_rings = 30
    n_sectors = 180
    dr = 1.0 / n_rings
    dtheta = 360.0 / n_sectors

    for ring in range(n_rings):
        r_inner = ring * dr + 0.02
        r_mid = r_inner + dr / 2
        thetas, rs, bar_colors, widths = [], [], [], []
        for sec in range(n_sectors):
            theta_mid = sec * dtheta + dtheta / 2
            h = theta_mid / 360.0
            s = r_mid
            thetas.append(theta_mid)
            rs.append(dr)
            r, g, b = colorsys.hsv_to_rgb(h, s, 1.0)
            r = max(0.0, min(1.0, r))
            g = max(0.0, min(1.0, g))
            b = max(0.0, min(1.0, b))
            bar_colors.append(mcolors.to_hex((r, g, b)))
            widths.append(dtheta)
        fig.add_trace(go.Barpolar(
            r=rs, theta=thetas, width=widths, base=r_inner,
            marker=dict(color=bar_colors, line=dict(width=0)),
            opacity=1.0, hoverinfo='skip', showlegend=False
        ))

    pts_theta, pts_r, pts_color, pts_hover = [], [], [], []
    for c, p in zip(colors_prop, props_prop):
        h, s, v = colorsys.rgb_to_hsv(*c)
        pts_theta.append(h * 360.0)
        pts_r.append(s)
        pts_color.append(mcolors.to_hex(c))
        hex_val = mcolors.to_hex(c).upper()
        rgb_val = [int(x*255) for x in c]
        hover_text = (
            f"<b>{hex_val}</b><br>"
            f"RGB: {rgb_val}<br>"
            f"占比: {p*100:.2f}%"
        )
        pts_hover.append(hover_text)

    for c, p, r, theta in zip(colors_prop, props_prop, pts_r, pts_theta):
        hex_val = mcolors.to_hex(c).upper()
        rgb_val = [int(x*255) for x in c]
        hover_text = (
            f"<span style='color:{hex_val};font-size:20px;'>██████</span><br>"
            f"<b>{hex_val}</b><br>"
            f"RGB: {rgb_val}<br>"
            f"占比: {p*100:.2f}%"
        )
        fig.add_trace(go.Scatterpolar(
            r=[r], theta=[theta], mode='markers',
            marker=dict(size=dot_size, color=hex_val, line=dict(color='#ffffff', width=2)),
            text=[hover_text], hovertemplate="%{text}<extra></extra>",
            hoverlabel=dict(bgcolor="white", font_size=12, bordercolor="rgba(0,0,0,0.15)"),
            showlegend=False
        ))

    fig.update_traces(selector=dict(mode='markers'), unselected=dict(marker_opacity=0.9))

    if focus_main and len(pts_theta) > 0:
        h_dom = colorsys.rgb_to_hsv(*dominant_color)[0]
        ang = h_dom * 360.0
        fig.update_layout(
            polar=dict(
                barmode="overlay", bargap=0,
                angularaxis=dict(rotation=90 - ang, direction="clockwise",
                                 showticklabels=False, ticks='', showgrid=False),
                radialaxis=dict(range=[0, 1.05], showticklabels=False, ticks='', showgrid=False)
            )
        )
    else:
        fig.update_layout(
            polar=dict(
                barmode="overlay", bargap=0,
                angularaxis=dict(showticklabels=False, ticks='', showgrid=False),
                radialaxis=dict(range=[0, 1.05], showticklabels=False, ticks='', showgrid=False)
            )
        )

    fig.update_layout(
        width=wheel_size, height=wheel_size, margin=dict(l=10, r=10, t=10, b=10),
        hovermode='closest', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def build_value_wheel(colors_prop, props_prop, dominant_color, focus_main, wheel_size, dot_size=18):
    fig = go.Figure()
    h_base = colorsys.rgb_to_hsv(*dominant_color)[0]
    n_rings = 30
    n_sectors = 180
    dr = 1.0 / n_rings
    dtheta = 360.0 / n_sectors

    for ring in range(n_rings):
        r_inner = ring * dr + 0.02
        r_mid = r_inner + dr / 2
        v = r_mid
        thetas, rs, bar_colors, widths = [], [], [], []
        for sec in range(n_sectors):
            theta_mid = sec * dtheta + dtheta / 2
            s = theta_mid / 360.0
            thetas.append(theta_mid)
            rs.append(dr)
            r, g, b = colorsys.hsv_to_rgb(h_base, s, v)
            r = max(0.0, min(1.0, r))
            g = max(0.0, min(1.0, g))
            b = max(0.0, min(1.0, b))
            bar_colors.append(mcolors.to_hex((r, g, b)))
            widths.append(dtheta)
        fig.add_trace(go.Barpolar(
            r=rs, theta=thetas, width=widths, base=r_inner,
            marker=dict(color=bar_colors, line=dict(width=0)),
            opacity=1.0, hoverinfo='skip', showlegend=False
        ))

    pts_theta, pts_r, pts_color, pts_hover = [], [], [], []
    for c, p in zip(colors_prop, props_prop):
        h, s, v = colorsys.rgb_to_hsv(*c)
        pts_theta.append(s * 360.0)
        pts_r.append(v)
        pts_color.append(mcolors.to_hex(c))
        hex_val = mcolors.to_hex(c).upper()
        rgb_val = [int(x*255) for x in c]
        hover_text = (
            f"<b>{hex_val}</b><br>"
            f"RGB: {rgb_val}<br>"
            f"占比: {p*100:.2f}%"
        )
        pts_hover.append(hover_text)

    for c, p, r, theta in zip(colors_prop, props_prop, pts_r, pts_theta):
        hex_val = mcolors.to_hex(c).upper()
        rgb_val = [int(x*255) for x in c]
        hover_text = (
            f"<span style='color:{hex_val};font-size:20px;'>██████</span><br>"
            f"<b>{hex_val}</b><br>"
            f"RGB: {rgb_val}<br>"
            f"占比: {p*100:.2f}%"
        )
        fig.add_trace(go.Scatterpolar(
            r=[r], theta=[theta], mode='markers',
            marker=dict(size=dot_size, color=hex_val, line=dict(color='#ffffff', width=2)),
            text=[hover_text], hovertemplate="%{text}<extra></extra>",
            hoverlabel=dict(bgcolor="white", font_size=12, bordercolor="rgba(0,0,0,0.15)"),
            showlegend=False
        ))

    fig.update_traces(selector=dict(mode='markers'), unselected=dict(marker_opacity=0.9))

    if focus_main and len(pts_theta) > 0:
        _, s_dom, v_dom = colorsys.rgb_to_hsv(*dominant_color)
        ang = s_dom * 360.0
        fig.update_layout(
            polar=dict(
                barmode="overlay", bargap=0,
                angularaxis=dict(rotation=90 - ang, direction="clockwise",
                                 showticklabels=False, ticks='', showgrid=False),
                radialaxis=dict(range=[0, 1.05], showticklabels=False, ticks='', showgrid=False)
            )
        )
    else:
        fig.update_layout(
            polar=dict(
                barmode="overlay", bargap=0,
                angularaxis=dict(showticklabels=False, ticks='', showgrid=False),
                radialaxis=dict(range=[0, 1.05], showticklabels=False, ticks='', showgrid=False)
            )
        )

    fig.update_layout(
        width=wheel_size, height=wheel_size, margin=dict(l=10, r=10, t=10, b=10),
        hovermode='closest', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

# ═══════════════════════════════════════════════════════════
# UI 渲染
# ═══════════════════════════════════════════════════════════
st.title("色彩协同画布")

uploaded_file = st.file_uploader("导入设计资产 (JPG / PNG)...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    img_resized = img.resize((80, 80))
    default_img_name = os.path.splitext(uploaded_file.name)[0]

    # ═══════════════════════════════════════════════════════════
    # 侧边栏 — 紧凑优化版 + 选中色块放大预览
    # ═══════════════════════════════════════════════════════════
    st.sidebar.markdown("**调控中心**")
    palette_name = st.sidebar.text_input("色板名称", value=default_img_name)

    c1, c2 = st.sidebar.columns(2)
    with c1:
        threshold = st.slider("过滤阈值", 0.0001, 0.03, 0.001, format="%.4f")
    with c2:
        clusters = st.slider("聚类数", 5, 60, 24)

    c3, c4 = st.sidebar.columns(2)
    with c3:
        exclude_focus = st.checkbox("剔除低占比焦点色", value=True)
    with c4:
        gen_aco = st.checkbox("生成 .aco 文件", value=True)

    focus_main = False

    c_hue, c_val = st.sidebar.columns(2)
    with c_hue:
        show_hue = st.checkbox("色相环", value=True)
    with c_val:
        show_val = st.checkbox("明度环", value=False)

    if show_hue and show_val:
        wheel_display_mode = "色相+明度并列"
    elif show_val:
        wheel_display_mode = "仅明度环"
    else:
        wheel_display_mode = "仅色相环"

    dot_size = st.sidebar.slider("色点大小", 8, 24, 14)

    colors_prop, props_prop, focus_color, focus_prop = analyze_image(
        img_resized, threshold=threshold, n_clusters=clusters
    )
    dominant_color = colors_prop[0]

    st.sidebar.markdown("<hr style='margin:6px 0;'>", unsafe_allow_html=True)
    st.sidebar.image(img, use_container_width=True)
    st.sidebar.metric("提取色板数", len(colors_prop))

    # ----- 侧边栏选中色块放大预览区域 -----
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🎨 选中色块 (放大预览)**")
    selected = st.session_state.selected_color
    if selected is not None:
        st.sidebar.markdown(f"""
        <div class="selected-preview">
            <div class="big-swatch" style="background: {selected['hex']};"></div>
            <div style="font-weight:bold; font-size:1rem;">{selected['hex']}</div>
            <div style="font-size:0.85rem; color:#555;">rgb({selected['rgb']})</div>
            <div style="font-size:0.85rem; color:#555;">占比 {selected['prop']:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
        if st.sidebar.button("清除选中", use_container_width=True):
            st.session_state.selected_color = None
            st.rerun()
    else:
        st.sidebar.info("点击下方任意色卡，此处放大显示")
    st.sidebar.divider()

    # ═══════════════════════════════════════════════════════════
    # 核心区域：色环
    # ═══════════════════════════════════════════════════════════
    WHEEL_SIZE = 560

    if wheel_display_mode == "仅色相环":
        st.subheader("光谱色相环")
        fig_hue = build_hue_wheel(colors_prop, props_prop, dominant_color, focus_main, WHEEL_SIZE, dot_size)
        st.plotly_chart(fig_hue, config={'displayModeBar': False}, use_container_width=False, key="hue_wheel")

    elif wheel_display_mode == "仅明度环":
        st.subheader("明度环 (S-V 平面)")
        fig_val = build_value_wheel(colors_prop, props_prop, dominant_color, focus_main, WHEEL_SIZE, dot_size)
        st.plotly_chart(fig_val, config={'displayModeBar': False}, use_container_width=False, key="val_wheel")

    else:  # 并列
        st.subheader("双环并列")
        col_h, col_v = st.columns(2)
        with col_h:
            st.markdown("<div style='text-align:center;font-weight:600;font-size:0.9rem;'>色相环 (H-S)</div>", unsafe_allow_html=True)
            fig_hue = build_hue_wheel(colors_prop, props_prop, dominant_color, focus_main, WHEEL_SIZE, dot_size)
            st.plotly_chart(fig_hue, config={'displayModeBar': False}, use_container_width=False, key="hue_wheel_dual")
        with col_v:
            st.markdown("<div style='text-align:center;font-weight:600;font-size:0.9rem;'>明度环 (S-V)</div>", unsafe_allow_html=True)
            fig_val = build_value_wheel(colors_prop, props_prop, dominant_color, focus_main, WHEEL_SIZE, dot_size)
            st.plotly_chart(fig_val, config={'displayModeBar': False}, use_container_width=False, key="val_wheel_dual")

    st.divider()

    # ═══════════════════════════════════════════════════════════
    # 色级面板 (条形图，保持不变)
    # ═══════════════════════════════════════════════════════════
    cv_1, cv_2 = st.columns([3, 2])
    with cv_1: st.markdown("<h3 style='margin:0;padding:0;font-size:1.1rem;'>色级面板</h3>", unsafe_allow_html=True)
    with cv_2: view_mode = st.radio("显示模式", ["标签页", "全览"], horizontal=True, label_visibility="collapsed")

    def build_color_bar(colors, props, title, key):
        fig = go.Figure()
        start = 0
        for c, p in zip(colors, props):
            hex_val = mcolors.to_hex(c).upper()
            rgb_val = [int(x*255) for x in c]
            fig.add_trace(go.Bar(
                x=[p], y=[0], base=[start], orientation='h',
                marker=dict(color=hex_val, line=dict(width=0)),
                hovertemplate=(
                    f"<span style='color:{hex_val};font-size:20px;'>██████</span><br>"
                    f"<b>{hex_val}</b><br>"
                    f"RGB: {rgb_val}<br>"
                    f"占比: {p*100:.2f}%<extra></extra>"
                ),
                showlegend=False,
                hoverlabel=dict(bgcolor="white", font_size=12, bordercolor="rgba(0,0,0,0.15)")
            ))
            start += p
        fig.update_layout(
            barmode='stack', height=48, margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[0, 1], fixedrange=True),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False, fixedrange=True),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            hovermode='x unified'
        )
        return fig

    def build_equal_bar(colors, title, key):
        fig = go.Figure()
        n_total = len(colors)
        for i, c in enumerate(colors):
            hex_val = mcolors.to_hex(c).upper()
            rgb_val = [int(x*255) for x in c]
            fig.add_trace(go.Bar(
                x=[1/n_total], y=[0], base=[i/n_total], orientation='h',
                marker=dict(color=hex_val, line=dict(width=0)),
                hovertemplate=(
                    f"<span style='color:{hex_val};font-size:20px;'>██████</span><br>"
                    f"<b>{hex_val}</b><br>"
                    f"RGB: {rgb_val}<extra></extra>"
                ),
                showlegend=False,
                hoverlabel=dict(bgcolor="white", font_size=12, bordercolor="rgba(0,0,0,0.15)")
            ))
        fig.update_layout(
            barmode='stack', height=48, margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[0, 1], fixedrange=True),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False, fixedrange=True),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            hovermode='x unified'
        )
        return fig

    def build_gradient_bar(colors, title, key):
        fig = go.Figure()
        if exclude_focus and focus_prop < 0.05 and len(colors) > 2:
            colors_for_grad = [c for c in colors if not np.array_equal(c, focus_color)]
        else:
            colors_for_grad = colors

        n = len(colors_for_grad)
        n_segments = 256

        for i in range(n_segments):
            t = i / n_segments
            idx = int(t * (n - 1))
            idx = min(idx, n - 1)
            next_idx = min(idx + 1, n - 1)
            local_t = t * (n - 1) - idx
            c = colors_for_grad[idx] * (1 - local_t) + colors_for_grad[next_idx] * local_t
            r = max(0, min(1, c[0]))
            g = max(0, min(1, c[1]))
            b = max(0, min(1, c[2]))
            hex_val = mcolors.to_hex((r, g, b))
            fig.add_trace(go.Bar(
                x=[1/n_segments], y=[0], base=[i/n_segments], orientation='h',
                marker=dict(color=hex_val, line=dict(width=0)),
                hoverinfo='skip', showlegend=False
            ))

        fig.add_trace(go.Scatter(
            x=[0.5], y=[0], mode='markers',
            marker=dict(size=0, opacity=0),
            hovertemplate=f"<b>连续渐变</b><br>基于 {n} 种颜色<extra></extra>",
            showlegend=False,
            hoverlabel=dict(bgcolor="white", font_size=12, bordercolor="rgba(0,0,0,0.15)")
        ))

        fig.update_layout(
            barmode='stack', height=48, margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[0, 1], fixedrange=True),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False, fixedrange=True),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            hovermode='closest'
        )
        return fig

    lums = np.array([0.299*c[0] + 0.587*c[1] + 0.114*c[2] for c in colors_prop])
    idx_lum_asc = np.argsort(lums)
    colors_lum = colors_prop[idx_lum_asc]
    props_lum = props_prop[idx_lum_asc]

    fig_m1 = build_color_bar(colors_prop, props_prop, "覆盖率", "m1")
    fig_m2 = build_color_bar(colors_lum, props_lum, "明度加权", "m2")
    fig_m3 = build_equal_bar(colors_lum, "等宽", "m3")
    fig_m4 = build_gradient_bar(colors_lum, "渐变", "m4")

    labels = ["覆盖率", "明度加权", "等宽离散", "连续渐变"]
    figs = [fig_m1, fig_m2, fig_m3, fig_m4]
    keys_tab = ["m1_tab", "m2_tab", "m3_tab", "m4_tab"]
    keys_full = ["m1_full", "m2_full", "m3_full", "m4_full"]

    if view_mode == "标签页":
        tab1, tab2, tab3, tab4 = st.tabs(labels)
        for tab, fig, k in zip([tab1, tab2, tab3, tab4], figs, keys_tab):
            with tab:
                st.plotly_chart(fig, config={"displayModeBar": False}, use_container_width=True, key=k)
    else:
        for label, fig, k in zip(labels, figs, keys_full):
            c_l, c_r = st.columns([1, 20])
            with c_l:
                st.markdown(f"<div style='font-size:0.7rem;color:#888;white-space:nowrap;padding-top:4px;'>{label}</div>", unsafe_allow_html=True)
            with c_r:
                st.plotly_chart(fig, config={"displayModeBar": False}, use_container_width=True, key=k)

    st.divider()
    st.subheader("色板数据总览")

    # ═══════════════════════════════════════════════════════════
    # 色板列表 — 三列紧凑可点击按钮（无指针虚线 + 选中放大效果）
    # ═══════════════════════════════════════════════════════════
    card_data = []
    for idx, (c, p) in enumerate(zip(colors_prop, props_prop)):
        hex_code = mcolors.to_hex(c).upper()
        rgb_str = f"{int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)}"
        card_data.append({
            "index": idx,
            "hex": hex_code,
            "rgb": rgb_str,
            "prop": p * 100,
            "color": hex_code
        })

    # 分三列
    chunk_size = math.ceil(len(card_data) / 3)
    chunks = [card_data[i:i+chunk_size] for i in range(0, len(card_data), chunk_size)]
    while len(chunks) < 3:
        chunks.append([])

    cols = st.columns(3)

    # 定义点击回调函数（通过 session state 更新选中色块）
    def on_color_click(color_info):
        st.session_state.selected_color = color_info
        # 无需 rerun，streamlit 会自动重新运行

    for col_idx, col in enumerate(cols):
        with col:
            for item in chunks[col_idx]:
                is_selected = (st.session_state.selected_color is not None and 
                               st.session_state.selected_color.get("hex") == item["hex"])
                # 动态附加选中样式类
                btn_class = "color-card-btn"
                if is_selected:
                    btn_class += " selected-color-card"
                # 使用 st.button 来获得干净的可点击组件
                button_label = f"""
                <div class="flex-row">
                    <div class="color-swatch" style="background: {item['color']};"></div>
                    <span class="color-code">{item['hex']}</span>
                    <span class="color-code">rgb({item['rgb']})</span>
                    <div class="right-group">
                        <span class="color-percent">{item['prop']:.2f}%</span>
                    </div>
                </div>
                """
                # 关键：按钮无 hand 光标、无轮廓、点击时更新选中状态
                clicked = st.button(
                    button_label,
                    key=f"color_card_{item['index']}_{item['hex']}",
                    use_container_width=True,
                    help="点击放大预览"
                )
                # 应用自定义样式（移除默认按钮样式）
                st.markdown(f"""
                <style>
                    div[data-testid="stButton"] button[key="color_card_{item['index']}_{item['hex']}"] {{
                        background: rgba(128,128,128,0.03);
                        border: 1px solid rgba(128,128,128,0.05);
                        border-radius: 6px;
                        padding: 6px 8px;
                        text-align: left;
                        cursor: default !important;
                        transition: all 0.1s ease;
                        font-family: inherit;
                        font-size: 0.9rem;
                        width: 100%;
                        margin-bottom: 8px;
                    }}
                    div[data-testid="stButton"] button[key="color_card_{item['index']}_{item['hex']}"]:hover {{
                        background: rgba(128,128,128,0.08);
                        border-color: rgba(128,128,128,0.15);
                    }}
                    div[data-testid="stButton"] button[key="color_card_{item['index']}_{item['hex']}"]:focus,
                    div[data-testid="stButton"] button[key="color_card_{item['index']}_{item['hex']}"]:active {{
                        outline: none !important;
                        box-shadow: none !important;
                    }}
                </style>
                """, unsafe_allow_html=True)
                if clicked:
                    on_color_click(item)

    st.divider()

    # --- 数据资产输出 ---
    st.subheader("资产导出")
    col_d1, col_d2 = st.columns(2)

    structured_json = {
        "palette_group": palette_name,
        "colors": []
    }
    for c, p in zip(colors_prop, props_prop):
        hex_code = mcolors.to_hex(c).upper()
        color_name = f"{hex_code} ({p*100:.1f}%)"
        structured_json["colors"].append({
            "name": color_name, "hex": hex_code,
            "rgb": [int(x*255) for x in c], "proportion": float(p)
        })

    col_d1.download_button(
        "导出 JSON",
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
            "导出 Photoshop (.aco)", buf.getvalue(),
            f"{palette_name}_swatches.aco"
        )
