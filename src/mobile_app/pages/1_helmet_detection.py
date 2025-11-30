"""
eScooterAI - 헬멧 탐지 페이지
Helmet Detection Page - Mobile First Design

이미지 업로드 기반 헬멧 착용 상태 탐지
"""

import streamlit as st

# ============================================================================
# 페이지 설정
# ============================================================================

st.set_page_config(
    page_title="헬멧 탐지 - eScooterAI",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================================
# 커스텀 CSS - 모바일 디자인
# ============================================================================

st.markdown("""
<style>
    /* 전체 배경 */
    .main {
        background: #F8FAFC;
    }

    .main > div {
        padding-top: 0rem;
    }

    /* 헤더 - 밝은 파란색 */
    .header-container {
        background: #3B82F6;
        padding: 2.5rem 1.5rem;
        border-radius: 0 0 25px 25px;
        margin: -1rem -1rem 1.5rem -1rem;
        text-align: center;
    }

    .header-icon {
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }

    .header-title {
        color: white;
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
    }

    .header-subtitle {
        color: rgba(255, 255, 255, 0.95);
        font-size: 1rem;
        margin-top: 0.5rem;
        font-weight: 400;
    }

    /* 기능 카드 */
    .feature-card {
        background: white;
        padding: 2rem;
        border-radius: 18px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
        margin-bottom: 1.5rem;
        text-align: center;
    }

    .feature-icon-large {
        font-size: 3.5rem;
        margin-bottom: 1rem;
    }

    /* 버튼 스타일 - 밝은 파란색 */
    .stButton > button {
        width: 100%;
        height: 55px;
        font-size: 1rem;
        font-weight: 600;
        border-radius: 15px;
        border: none;
        background: #3B82F6;
        color: white;
        box-shadow: 0 2px 6px rgba(59, 130, 246, 0.3);
        transition: all 0.2s;
    }

    .stButton > button:hover {
        background: #2563EB;
        transform: translateY(-1px);
        box-shadow: 0 4px 10px rgba(59, 130, 246, 0.4);
    }

    /* 상태 배지 */
    .status-badge {
        background: #EFF6FF;
        color: #3B82F6;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin: 0.5rem;
    }

    /* 작은 카드 */
    .small-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
        margin-bottom: 1rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# 헤더
# ============================================================================

st.markdown("""
<div class="header-container">
    <div class="header-icon">📸</div>
    <h1 class="header-title">헬멧 탐지</h1>
    <p class="header-subtitle">AI로 헬멧 착용 여부를 감지합니다</p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# 개발 진행 중 안내
# ============================================================================

st.markdown("""
<div class="feature-card">
    <div class="feature-icon-large">🚧</div>
    <h2 style="color: #1E293B; margin-bottom: 0.5rem; font-size: 1.5rem;">개발 진행 중</h2>
    <p style="color: #64748B; font-size: 0.9rem;">
        곧 사용 가능합니다!
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# 성능 배지
# ============================================================================

st.markdown("""
<div style="text-align: center; margin: 1.5rem 0;">
    <span class="status-badge">🎯 정확도 93.7%</span>
    <span class="status-badge">⚡ 32ms</span>
    <span class="status-badge">📦 6MB</span>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# 예정 기능
# ============================================================================

st.markdown("<h3 style='font-size: 1.3rem; font-weight: 700; color: #1E293B; margin: 1.5rem 0 1rem 0; padding-left: 0.3rem;'>🎯 예정 기능</h3>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="small-card">
        <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">📤</div>
        <p style="color: #1E293B; font-weight: 600; font-size: 0.9rem; margin: 0;">이미지<br/>업로드</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="small-card">
        <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">📹</div>
        <p style="color: #1E293B; font-weight: 600; font-size: 0.9rem; margin: 0;">실시간<br/>웹캠</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="small-card">
        <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">📱</div>
        <p style="color: #1E293B; font-weight: 600; font-size: 0.9rem; margin: 0;">Telegram<br/>알림</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================================
# 탐지 클래스 설명
# ============================================================================

with st.expander("ℹ️ 탐지 정보"):
    st.markdown("""
    **🔵 Helmet** - 헬멧 착용 (안전)

    **🔴 Head** - 헬멧 미착용 (위험, 알림)
    """)

# ============================================================================
# 하단 메뉴
# ============================================================================

st.markdown("<br>", unsafe_allow_html=True)

if st.button("🏠 홈", use_container_width=True):
    st.switch_page("app.py")
