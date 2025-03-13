# app_langgraph.py - 챕터 3: LangGraph를 활용한 AI 토론 시스템 구현
import streamlit as st

from components.debate import render_debate_view, render_start_view
from utils.state_manager import init_session_state

# 페이지 설정
st.set_page_config(page_title="AI 토론", page_icon="🤖", layout="wide")

# 제목 및 소개
st.title("🤖 AI 토론 - LangGraph 버전")


init_session_state()

# 메인 UI 렌더링
if not st.session_state.debate_active:
    render_start_view()
else:
    render_debate_view()
