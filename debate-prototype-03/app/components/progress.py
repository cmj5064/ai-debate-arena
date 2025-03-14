import streamlit as st


def show_progress():
    # 전체 단계 수 계산: 각 라운드는 찬성과 반대 발언으로 2단계, 마지막에 심판 단계 1개 추가
    total_steps = st.session_state.max_rounds * 2 + 1

    # 현재까지 진행된 단계 수 계산
    if st.session_state.current_step == "judge":
        current_steps = total_steps - 1
    elif st.session_state.current_step == "completed":
        current_steps = total_steps

    else:
        # 현재 라운드의 이전 라운드들은 모두 완료됨 (각 라운드당 2단계)
        completed_rounds_steps = (st.session_state.round - 1) * 2

        # 현재 라운드에서 어느 단계인지 확인
        current_round_steps = (
            1 if st.session_state.current_step.startswith("con") else 0
        )
        current_steps = completed_rounds_steps + current_round_steps
    progress_ratio = current_steps / total_steps

    # 진행 상태 표시
    st.progress(progress_ratio)
