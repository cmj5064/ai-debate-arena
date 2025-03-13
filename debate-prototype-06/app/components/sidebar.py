"""
사이드바 컴포넌트 - 설정 및 토론 이력 관리
"""

import streamlit as st

from typing import Dict, Any

from .history import render_history_tab


def render_settings_tab():
    """설정 탭 렌더링"""
    # 토론 주제 입력
    st.text_input(
        label="토론 주제를 입력하세요:",
        value="인공지능은 인간의 일자리를 대체할 수 있습니다.",
        key="ui_debate_topic",
    )

    st.slider("토론 라운드 수", min_value=1, max_value=5, value=1, key="ui_max_rounds")


def render_sidebar() -> Dict[str, Any]:
    """사이드바 렌더링"""
    with st.sidebar:
        st.header("토론 설정")

        # 탭 추가: 새 토론 / 이력 조회
        tab1, tab2 = st.tabs(["새 토론", "토론 이력"])

        with tab1:
            settings = render_settings_tab()

        with tab2:
            render_history_tab()

    return settings
