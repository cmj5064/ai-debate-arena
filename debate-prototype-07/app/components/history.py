import streamlit as st
from utils.state_manager import set_debate_to_state
from database.repository import (
    delete_all_debates,
    delete_debate_by_id,
    fetch_debate_by_id,
    fetch_debate_history,
)


def render_history_tab():
    st.subheader("이전 토론 이력")

    col1, col2 = st.columns([1, 1])

    # 이력 새로고침 버튼
    with col1:
        if st.button("이력 새로고침", use_container_width=True):
            st.rerun()

    # 전체 이력 삭제 버튼
    with col2:
        if st.button("전체 이력 삭제", type="primary", use_container_width=True):
            if delete_all_debates():
                st.rerun()

    debate_history = fetch_debate_history()

    if not debate_history:
        st.info("저장된 토론 이력이 없습니다.")
    else:
        render_history_list(debate_history)


def render_history_list(debate_history):
    for id, topic, date, rounds in debate_history:
        with st.container(border=True):
            col1, col2, col3 = st.columns([3, 1, 1])

            # 토론 정보
            with col1:
                st.write(f"***{topic}***")
                st.caption(f"날짜: {date} | 라운드: {rounds}")

            # 보기 버튼
            with col2:
                if st.button("보기", key=f"view_{id}", use_container_width=True):
                    load_debate_view(id)

            # 삭제 버튼
            with col3:
                if st.button("삭제", key=f"del_{id}", use_container_width=True):
                    if delete_debate_by_id(id):
                        st.rerun()


def load_debate_view(debate_id):
    # DB에서 토론 데이터 로딩
    topic, messages = fetch_debate_by_id(debate_id)
    if topic and messages:
        # 세션 스테이트에 토론 데이터 로드
        set_debate_to_state(topic, messages, debate_id)
        st.rerun()
