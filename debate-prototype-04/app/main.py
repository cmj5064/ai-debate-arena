import streamlit as st
from components.debate import render_debate_view, render_start_view
from components.sidebar import render_sidebar
from utils.state_manager import init_session_state

# 페이지 설정
st.set_page_config(page_title="AI 토론", page_icon="🤖", layout="wide")

st.title("🤖 AI 토론")
st.markdown(
    """**LangGraph 기반 AI 토론 시스템**으로 찬성/반대/심판 역할의 AI가 지식 검색(RAG)을 활용해 논리적인 토론을 펼칩니다."""
)

# 세션 스테이트 초기화
init_session_state()

# 사이드바 렌더링
render_sidebar()

# 메인 UI 렌더링
if not st.session_state.debate_active:
    render_start_view()
else:
    render_debate_view()
