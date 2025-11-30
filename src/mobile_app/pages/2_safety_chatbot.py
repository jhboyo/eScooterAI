"""
eScooterAI - 안전 가이드 챗봇 페이지
Safety Chatbot Page - Mobile First Design

RAG 기반 헬멧 안전 질의응답 시스템
"""

import streamlit as st
import os
import sys
from pathlib import Path
from datetime import datetime

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.rag.vector_store import FAISSVectorStore
from src.rag.query_engine import RAGQueryEngine

# ============================================================================
# 페이지 설정
# ============================================================================

st.set_page_config(
    page_title="안전 챗봇 - eScooterAI",
    page_icon="🤖",
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
        padding: 2rem 1.5rem;
        border-radius: 0 0 25px 25px;
        margin: -1rem -1rem 1.5rem -1rem;
        text-align: center;
    }

    .header-icon {
        font-size: 2.5rem;
        margin-bottom: 0.3rem;
    }

    .header-title {
        color: white;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0;
    }

    .header-subtitle {
        color: rgba(255, 255, 255, 0.95);
        font-size: 0.9rem;
        margin-top: 0.3rem;
        font-weight: 400;
    }

    /* 채팅 메시지 */
    .stChatMessage {
        border-radius: 18px;
        padding: 1rem;
        margin: 0.5rem 0;
    }

    /* 버튼 스타일 - 밝은 파란색 */
    .stButton > button {
        width: 100%;
        height: 50px;
        font-size: 0.95rem;
        font-weight: 600;
        border-radius: 12px;
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
        margin: 0.5rem 0;
    }

    /* 질문 예시 카드 */
    .question-card {
        background: white;
        padding: 1rem;
        border-radius: 15px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
        margin-bottom: 1rem;
        text-align: center;
        transition: all 0.2s;
    }

    .question-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# RAG 엔진 초기화 (캐싱)
# ============================================================================

@st.cache_resource
def load_rag_engine():
    """RAG 엔진 로드 (한 번만 실행)"""
    try:
        vector_db_dir = project_root / "vector_db"
        vector_store = FAISSVectorStore()
        vector_store.load(vector_db_dir)

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None, "OpenAI API 키가 설정되지 않았습니다."

        rag_engine = RAGQueryEngine(
            vector_store=vector_store,
            model="gpt-3.5-turbo",
            temperature=0.3,
            max_tokens=500,
            top_k=3,
            api_key=api_key
        )

        return rag_engine, None

    except Exception as e:
        return None, f"RAG 엔진 초기화 실패: {str(e)}"


# ============================================================================
# 세션 상태 초기화
# ============================================================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "show_sources" not in st.session_state:
    st.session_state.show_sources = True

# ============================================================================
# 헤더
# ============================================================================

st.markdown("""
<div class="header-container">
    <div class="header-icon">💬</div>
    <h1 class="header-title">안전 가이드 챗봇</h1>
    <p class="header-subtitle">헬멧 법규·착용법·사고 사례 질의응답</p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# RAG 엔진 로드 확인
# ============================================================================

rag_engine, error_msg = load_rag_engine()

if error_msg:
    st.error(f"❌ {error_msg}")
    st.info("📝 .env 파일에 OPENAI_API_KEY를 설정하세요")
    st.stop()

# 상태 표시
st.markdown("""
<div style="text-align: center; margin-bottom: 1rem;">
    <span class="status-badge">✅ 온라인</span>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# 질문 예시 버튼
# ============================================================================

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("💰 과태료", use_container_width=True):
        st.session_state.current_question = "헬멧을 안 쓰면 과태료가 얼마인가요?"
        st.rerun()

with col2:
    if st.button("🎓 착용법", use_container_width=True):
        st.session_state.current_question = "헬멧을 올바르게 착용하는 방법을 알려주세요"
        st.rerun()

with col3:
    if st.button("📊 사고", use_container_width=True):
        st.session_state.current_question = "전동킥보드 사고 통계를 알려주세요"
        st.rerun()

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================================
# 채팅 인터페이스
# ============================================================================

# 채팅 히스토리 표시
chat_container = st.container()

with chat_container:
    if len(st.session_state.chat_history) == 0:
        st.info("👋 안녕하세요! 헬멧 관련 질문을 해주세요.", icon="💡")

    for chat in st.session_state.chat_history:
        # 사용자 질문
        with st.chat_message("user", avatar="👤"):
            st.markdown(chat["question"])

        # AI 답변
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(chat["answer"])

            # 출처 문서 표시
            if st.session_state.show_sources and "sources" in chat and chat["sources"]:
                with st.expander("📄 참고 문서", expanded=False):
                    for i, source in enumerate(chat["sources"], 1):
                        st.markdown(f"""
                        **{i}. {source.get('category', 'Unknown')}**
                        (유사도: {source.get('score', 0):.2f})

                        > {source.get('content', '')[:150]}...

                        *출처: {source.get('source', 'Unknown')}*
                        """)

# 질문 입력 폼
question = st.chat_input("💬 헬멧 관련 질문을 입력하세요...", key="question_input")

# 예시 버튼 클릭 처리
if "current_question" in st.session_state:
    question = st.session_state.current_question
    del st.session_state.current_question

# 질문 처리
if question:
    # 사용자 질문 표시
    with st.chat_message("user", avatar="👤"):
        st.markdown(question)

    # AI 답변 생성
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("🤔 답변 생성 중..."):
            try:
                # RAG 쿼리 실행
                result = rag_engine.query(question)

                answer = result.get("answer", "답변을 생성할 수 없습니다.")
                sources = result.get("sources", [])

                # 답변 표시
                st.markdown(answer)

                # 출처 문서 표시
                if st.session_state.show_sources and sources:
                    with st.expander("📄 참고 문서", expanded=False):
                        for i, source in enumerate(sources, 1):
                            st.markdown(f"""
                            **{i}. {source.get('category', 'Unknown')}**
                            (유사도: {source.get('score', 0):.2f})

                            > {source.get('content', '')[:150]}...

                            *출처: {source.get('source', 'Unknown')}*
                            """)

                # 히스토리에 추가
                st.session_state.chat_history.append({
                    "question": question,
                    "answer": answer,
                    "sources": sources,
                    "timestamp": datetime.now().isoformat()
                })

                st.rerun()

            except Exception as e:
                st.error(f"❌ 오류 발생: {str(e)}")

# ============================================================================
# 하단 메뉴
# ============================================================================

st.markdown("<br>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    if st.button("🗑️ 대화 지우기", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

with col2:
    if st.button("🏠 홈", use_container_width=True):
        st.switch_page("app.py")

# ============================================================================
# 시스템 정보
# ============================================================================

with st.expander("ℹ️ 시스템 정보"):
    st.markdown("""
    **📚 지식 베이스**
    35개 문서 • FAISS • GPT-3.5

    **🔍 검색 방식**
    벡터 변환 → Top-3 검색 → LLM 답변 생성

    **✅ 특징**
    문서 기반 답변 • 출처 표시 • 빠른 응답
    """)
