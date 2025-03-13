import streamlit as st


def render_options():
    # 토론 주제 입력
    st.text_input(
        label="토론 주제를 입력하세요:",
        value="인공지능은 인간의 일자리를 대체할 수 있습니다.",
        key="ui_debate_topic",
    )

    st.slider("토론 라운드 수", min_value=1, max_value=5, value=1, key="ui_max_rounds")
