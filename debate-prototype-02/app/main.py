# app_multi_agent.py - 챕터 2: Streamlit을 활용한 멀티 에이전트 구현
import streamlit as st

from components.debate import handle_con_round, handle_judge, handle_pro_round
from components.options import render_options
from components.progress import show_progress
from utils.state_manager import init_session_state, reset_session_state

# 페이지 설정
st.set_page_config(page_title="AI 토론", page_icon="🤖", layout="wide")

# 제목 및 소개
st.title("🤖 AI 토론 - 멀티 에이전트")
st.markdown(
    """
### 프로젝트 소개
이 애플리케이션은 3개의 AI 에이전트(찬성, 반대, 심판)가 사용자가 제시한 주제에 대해 토론을 진행합니다.
각 AI는 서로의 의견을 듣고 반박하며, 마지막에는 심판 AI가 토론 결과를 평가합니다.
"""
)

init_session_state()

render_options()

# 토론 시작 버튼
if not st.session_state.debate_active:
    if st.button("토론 시작"):
        reset_session_state()
        st.session_state.debate_active = True
        st.rerun()
else:
    # 토론 진행
    debate_topic = st.session_state.ui_debate_topic
    # 토론 주제 표시
    st.header(f"토론 주제: {debate_topic}")

    # 현재 라운드 정보 - 심판 단계에서는 라운드 표시 방식을 다르게 처리
    if (
        st.session_state.current_step == "judge"
        or st.session_state.current_step == "completed"
    ):
        st.subheader("최종 평가 단계")
    else:
        st.subheader(f"라운드 {st.session_state.round} / {st.session_state.max_rounds}")

    show_progress()

    # 진행 단계별 처리
    if st.session_state.current_step.startswith("pro_round_"):
        handle_pro_round(debate_topic)
        st.rerun()  # 페이지 리로드하여 다음 단계로 진행

    elif st.session_state.current_step.startswith("con_round_"):
        handle_con_round(debate_topic)
        st.rerun()  # 페이지 리로드하여 다음 단계로 진행

    elif (
        st.session_state.current_step == "judge"
        and st.session_state.judge_verdict is None
    ):
        handle_judge(debate_topic)
        st.rerun()  # 페이지 리로드하여 결과 표시

    # 토론 내용 표시
    st.header("토론 진행 상황")
    for i, entry in enumerate(st.session_state.debate_history):
        round_num = (i // 2) + 1

        st.subheader(f"라운드 {round_num} - {entry['role']}")
        st.write(entry["content"])
        st.divider()

    # 심판 판정 표시
    if st.session_state.judge_verdict:
        st.header("🧑‍⚖️ 심판 평가")
        st.write(st.session_state.judge_verdict)

    # 다시 시작 버튼
    if st.session_state.current_step == "completed":
        if st.button("새 토론 시작"):
            reset_session_state()
            st.rerun()
