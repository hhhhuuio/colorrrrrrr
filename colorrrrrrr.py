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

# ═══════════════════════════════════════════════════════════
# 页面配置
# ═══════════════════════════════════════════════════════════
st.set_page_config(page_title="色彩协同画布", layout="wide")

# --- 全局紧凑样式注入 ---
st.markdown("""
    <style>
    html, body, [data-testid="stMarkdownContainer"] { font-size: 0.82rem !important; }
    h1 { font-size: 1.5rem !important; font-weight: 700; padding-top: 0px; }
    h2 { font-size: 1.1rem !important; margin-top: 8px; }
    h3 { font-size: 0.95rem !important; }
    .stSlider, .stCheckbox { padding: 0px !important; margin: 0px !important; }
    div[data-testid="stBlock"] { padding: 4px !important; }
    /* 侧边栏极致紧凑 */
    [data-testid="stSidebar"] .block-container { padding: 1rem 0.8rem !important; }
    [data-testid="stSidebar"] h3 { font-size: 0.85rem !important; margin: 4px 0 2px 0 !important; }
    [data-testid="stSidebar"] .stRadio > div { padding: 2px 0 !important; }
    [data-testid="stSidebar"] .stSlider { padding: 0 !important; margin: -4px 0 !important; }
    [data-testid="stSidebar"] .stCheckbox { padding: 0 !important; margin: -2px 0 !important; }
    [data-testid="stSidebar"] .stButton { padding: 2px 0 !important; }
    [data-testid="stSidebar"] hr { margin: 6px 0 !important; }
    [data-testid="stSidebar"] .stTextInput { padding: 0 !important; margin: -2px 0 !important; }
    /* 色板表格紧凑双列 */
    .color-table { width: 100%; border-collapse: collapse; margin-top: 6px; font-size: 0.85rem; }
    .color-table th { border-bottom: 1px solid rgba(128,128,128,0.2); padding: 4px 6px; text-align: left; font-size: 0.8rem; }
    .color-table td { border-bottom: 1px solid rgba(128,128,128,0.08); padding: 3px 6px; vertical-align: middle; }
    .color-preview { width: 28px; height: 20px; border-radius: 3px; border: 1px solid rgba(128,128,128,0.3); }
    .color-row { transition: background 0.15s; }
    .color-row:hover { background: rgba(128,128,128,0.06); }
    .color-hover-info { display: none; font-size: 0.75rem; color: #666; margin-top: 2px; }
    .color-row:hover .color-hover-info { display: block; }
    /* 双列色板布局 */
    .palette-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0; }
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


# --- 核心算法引擎 ---
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
# 色环构建函数 — 实色填充版
# ═══════════════════════════════════════════════════════════

def build_hue_wheel(colors_prop, props_prop, dominant_color, focus_main, wheel_size, uirev, dot_size=18):
    """构建实色色相环（HSV 中 H-S 平面，V=1）"""
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
        c_diff = np.linalg.norm(c - dominant_color)
        similarity = max(0.0, 100.0 - (c_diff * 50.0))
        hover_text = (
            f"<b>HEX:</b> {mcolors.to_hex(c).upper()}<br>"
            f"<b>RGB:</b> {[int(x*255) for x in c]}<br>"
            f"<b>占比:</b> {p*100:.2f}%<br>"
            f"<b>主色相似度:</b> {similarity:.1f}%"
        )
        pts_hover.append(hover_text)

    fig.add_trace(go.Scatterpolar(
        r=pts_r, theta=pts_theta, mode='markers',
        marker=dict(size=dot_size, color=pts_color, line=dict(color='#ffffff', width=2)),
        customdata=pts_color, text=pts_hover, hovertemplate="%{text}<extra></extra>",
        hoverlabel=dict(bgcolor="whitesmoke", font_size=10),
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
            ), uirevision=uirev
        )
    else:
        fig.update_layout(
            polar=dict(
                barmode="overlay", bargap=0,
                angularaxis=dict(showticklabels=False, ticks='', showgrid=False),
                radialaxis=dict(range=[0, 1.05], showticklabels=False, ticks='', showgrid=False)
            ), uirevision=uirev
        )

    fig.update_layout(
        width=wheel_size, height=wheel_size, margin=dict(l=10, r=10, t=10, b=10),
        hovermode='closest', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig


def build_value_wheel(colors_prop, props_prop, dominant_color, focus_main, wheel_size, uirev, dot_size=18):
    """构建实色明度环（HSV 中 S-V 平面，H=画面占比最多的主色色相）"""
    fig = go.Figure()

    # 使用画面占比最多的颜色（dominant_color）的色相
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
        c_diff = np.linalg.norm(c - dominant_color)
        similarity = max(0.0, 100.0 - (c_diff * 50.0))
        hover_text = (
            f"<b>HEX:</b> {mcolors.to_hex(c).upper()}<br>"
            f"<b>RGB:</b> {[int(x*255) for x in c]}<br>"
            f"<b>占比:</b> {p*100:.2f}%<br>"
            f"<b>主色相似度:</b> {similarity:.1f}%"
        )
        pts_hover.append(hover_text)

    fig.add_trace(go.Scatterpolar(
        r=pts_r, theta=pts_theta, mode='markers',
        marker=dict(size=dot_size, color=pts_color, line=dict(color='#ffffff', width=2)),
        customdata=pts_color, text=pts_hover, hovertemplate="%{text}<extra></extra>",
        hoverlabel=dict(bgcolor="whitesmoke", font_size=10),
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
            ), uirevision=uirev
        )
    else:
        fig.update_layout(
            polar=dict(
                barmode="overlay", bargap=0,
                angularaxis=dict(showticklabels=False, ticks='', showgrid=False),
                radialaxis=dict(range=[0, 1.05], showticklabels=False, ticks='', showgrid=False)
            ), uirevision=uirev
        )

    fig.update_layout(
        width=wheel_size, height=wheel_size, margin=dict(l=10, r=10, t=10, b=10),
        hovermode='closest', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig


# --- UI 渲染界面 ---
st.title("色彩协同画布")

uploaded_file = st.file_uploader("导入设计资产 (JPG / PNG)...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    img_resized = img.resize((80, 80))
    default_img_name = os.path.splitext(uploaded_file.name)[0]

    # ═══════════════════════════════════════════════════════════
    # 侧边栏 — 紧凑优化版
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

    st.sidebar.markdown("<hr style='margin:6px 0;'>", unsafe_allow_html=True)
    st.sidebar.markdown("**色环视图**")
    focus_main = st.sidebar.checkbox("聚焦主色区域", value=False)

    wheel_display_mode = st.sidebar.radio(
        "展示模式", ["仅色相环", "仅明度环", "色相+明度并列"], index=0
    )

    dot_size = st.sidebar.slider("色点大小", 8, 24, 14)

    if st.sidebar.button("恢复默认视图", use_container_width=True, on_click=reset_wheel_view):
        pass

    colors_prop, props_prop, focus_color, focus_prop = analyze_image(
        img_resized, threshold=threshold, n_clusters=clusters
    )
    dominant_color = colors_prop[0]

    st.sidebar.markdown("<hr style='margin:6px 0;'>", unsafe_allow_html=True)
    st.sidebar.image(img, use_container_width=True)
    st.sidebar.metric("提取色板数", len(colors_prop))

    st.divider()

    # ═══════════════════════════════════════════════════════════
    # 核心区域：色环
    # ═══════════════════════════════════════════════════════════
    uirev = str(st.session_state.wheel_reset_counter)
    WHEEL_SIZE = 560

    if wheel_display_mode == "仅色相环":
        st.subheader("光谱色相环")
        fig_hue = build_hue_wheel(colors_prop, props_prop, dominant_color, focus_main, WHEEL_SIZE, uirev, dot_size)
        st.plotly_chart(fig_hue, config={'displayModeBar': False}, use_container_width=False, key="hue_wheel")

    elif wheel_display_mode == "仅明度环":
        st.subheader("明度环 (S-V 平面)")
        fig_val = build_value_wheel(colors_prop, props_prop, dominant_color, focus_main, WHEEL_SIZE, uirev, dot_size)
        st.plotly_chart(fig_val, config={'displayModeBar': False}, use_container_width=False, key="val_wheel")

    else:  # 并列
        st.subheader("双环并列")
        col_h, col_v = st.columns(2)
        with col_h:
            st.markdown("<div style='text-align:center;font-weight:600;font-size:0.9rem;'>色相环 (H-S)</div>", unsafe_allow_html=True)
            fig_hue = build_hue_wheel(colors_prop, props_prop, dominant_color, focus_main, WHEEL_SIZE, uirev, dot_size)
            st.plotly_chart(fig_hue, config={'displayModeBar': False}, use_container_width=False, key="hue_wheel_dual")
        with col_v:
            st.markdown("<div style='text-align:center;font-weight:600;font-size:0.9rem;'>明度环 (S-V)</div>", unsafe_allow_html=True)
            fig_val = build_value_wheel(colors_prop, props_prop, dominant_color, focus_main, WHEEL_SIZE, uirev, dot_size)
            st.plotly_chart(fig_val, config={'displayModeBar': False}, use_container_width=False, key="val_wheel_dual")

    st.divider()

    # ═══════════════════════════════════════════════════════════
    # 色卡面板
    # ═══════════════════════════════════════════════════════════
    cv_1, cv_2 = st.columns([3, 2])
    with cv_1: st.subheader("演化色级面板")
    with cv_2: view_mode = st.radio("显示模式", ["标签页", "全览"], horizontal=True, label_visibility="collapsed")

    def create_aligned_axis():
        fig, ax = plt.subplots(figsize=(11, 0.5))
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        ax.axis('off')
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, 0.5)
        return fig, ax

    fig_m1, ax_m1 = create_aligned_axis()
    start = 0
    for c, p in zip(colors_prop, props_prop):
        ax_m1.barh(0, p, left=start, color=c, height=1)
        start += p

    lums = np.array([0.299*c[0] + 0.587*c[1] + 0.114*c[2] for c in colors_prop])
    idx_lum_asc = np.argsort(lums)
    colors_lum = colors_prop[idx_lum_asc]
    props_lum = props_prop[idx_lum_asc]

    fig_m2, ax_m2 = create_aligned_axis()
    start_lum = 0
    for c, p in zip(colors_lum, props_lum):
        ax_m2.barh(0, p, left=start_lum, color=c, height=1)
        start_lum += p

    fig_m3, ax_m3 = create_aligned_axis()
    n_total = len(colors_lum)
    for i, c in enumerate(colors_lum):
        ax_m3.barh(0, 1/n_total, left=i/n_total, color=c, height=1)

    fig_m4, ax_m4 = create_aligned_axis()
    if exclude_focus and focus_prop < 0.05 and len(colors_lum) > 2:
        colors_for_grad = [c for c in colors_lum if not np.array_equal(c, focus_color)]
    else:
        colors_for_grad = colors_lum
    cmap_custom = mcolors.LinearSegmentedColormap.from_list("custom_lum", colors_for_grad)
    ax_m4.imshow(np.linspace(0, 1, 1024).reshape(1, -1), aspect='auto', cmap=cmap_custom, extent=[0, 1, -0.5, 0.5])

    if view_mode == "标签页":
        tab1, tab2, tab3, tab4 = st.tabs(["覆盖率色卡", "明度加权", "等宽色卡", "连续渐变"])
        with tab1:
            st.markdown("**1. 覆盖率原始分配 (按占比由大到小)**")
            st.pyplot(fig_m1)
        with tab2:
            st.markdown("**2. 加权明度梯度 (保留面积占比 → 依明度由暗至亮)**")
            st.pyplot(fig_m2)
        with tab3:
            st.markdown("**3. 标准明度等宽离散 (由暗至亮)**")
            st.pyplot(fig_m3)
        with tab4:
            st.markdown("**4. 平滑明度平衡连续渐变**")
            st.pyplot(fig_m4)
    else:
        st.markdown("**1. 覆盖率原始分配**")
        st.pyplot(fig_m1)
        st.markdown("**2. 加权明度梯度**")
        st.pyplot(fig_m2)
        st.markdown("**3. 标准明度等宽离散**")
        st.pyplot(fig_m3)
        st.markdown("**4. 平滑明度平衡连续渐变**")
        st.pyplot(fig_m4)

    st.divider()
    st.subheader("色板数据总览")

    # ═══════════════════════════════════════════════════════════
    # 色板列表 — 紧凑双列 + hover 显示比例和主色对比
    # ═══════════════════════════════════════════════════════════
    half = (len(colors_prop) + 1) // 2
    colors_left = list(zip(colors_prop, props_prop))[:half]
    colors_right = list(zip(colors_prop, props_prop))[half:]

    table_html = '<div class="palette-grid">'
    for side_data in [colors_left, colors_right]:
        table_html += '<table class="color-table">'
        table_html += '<tr><th>色块</th><th>HEX</th><th>RGB</th><th>占比</th></tr>'
        for c, p in side_data:
            hex_code = mcolors.to_hex(c).upper()
            rgb_str = f"{int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)}"
            c_diff = np.linalg.norm(c - dominant_color)
            similarity = max(0.0, 100.0 - (c_diff * 50.0))
            table_html += f"""
            <tr class="color-row">
                <td><div class="color-preview" style="background-color: {hex_code};"></div></td>
                <td><code>{hex_code}</code></td>
                <td><code>{rgb_str}</code></td>
                <td>{p*100:.2f}%</td>
            </tr>
            <tr class="color-row">
                <td colspan="4" style="padding:0 6px 4px 6px; border:none;">
                    <div class="color-hover-info">
                        占比: {p*100:.2f}% | 主色相似度: {similarity:.1f}% | 与主色距离: {c_diff:.3f}
                    </div>
                </td>
            </tr>
            """
        table_html += '</table>'
    table_html += '</div>'

    st.markdown(table_html, unsafe_allow_html=True)

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
