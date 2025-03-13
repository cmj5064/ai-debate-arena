import streamlit as st

from utils.state_manager import reset_session_state
from database.repository import save_debate_to_db
from retrieval.vector_store import create_vector_store
from workflow.graph import create_debate_graph
from workflow.state import AgentType, DebateState
from langfuse.callback import CallbackHandler


# 토론 시작 뷰
def render_start_view():
    if st.button("토론 시작"):
        start_debate()


# 토론 시작 처리
def start_debate():

    # 세션 상태 초기화
    st.session_state.vector_store = None
    st.session_state.retrieved_docs = {"pro": [], "con": []}
    st.session_state.debate_messages = []
    st.session_state.streaming_active = True

    # 사용자 입력값 가져오기
    debate_topic = st.session_state.ui_debate_topic
    enable_rag = st.session_state.ui_enable_rag
    max_rounds = st.session_state.ui_max_rounds

    # RAG 기능 활성화 시 외부 지식 수집
    if enable_rag:
        with st.spinner("외부 지식을 수집하고 분석 중입니다..."):
            vector_store = create_vector_store(debate_topic)
            st.session_state.vector_store = vector_store

            if vector_store:
                st.success("외부 지식 검색 준비 완료!")
            else:
                st.warning(
                    "외부 지식 검색을 위한 준비에 실패했습니다. 기본 토론으로 진행합니다."
                )

    debate_graph = create_debate_graph()

    initial_state: DebateState = {
        "topic": debate_topic,
        "messages": [],
        "current_round": 1,
        "max_rounds": max_rounds,
        "current_speaker": AgentType.PRO,
        "vector_store": st.session_state.vector_store,
        "retrieved_docs": {"pro": [], "con": []},
    }

    with st.spinner("토론이 진행 중입니다... 완료까지 잠시 기다려주세요."):

        langfuse_handler = CallbackHandler()
        result = debate_graph.invoke(
            initial_state, config={"callbacks": [langfuse_handler]}
        )

        # 결과를 세션 스테이트에 저장
        st.session_state.debate_messages = result["messages"]
        st.session_state.debate_active = True
        # 외부 지식 검색 결과 저장
        st.session_state.retrieved_docs = result.get(
            "retrieved_docs", {"pro": [], "con": []}
        )
        st.session_state.viewing_history = False
        save_debate_to_db(debate_topic, max_rounds, result["messages"])

    # 페이지 새로고침하여 결과 표시
    st.rerun()


def render_debate_view():
    if st.session_state.viewing_history:
        st.info("📚 이전에 저장된 토론을 보고 있습니다.")
        debate_topic = st.session_state.debate_topic
    else:
        debate_topic = st.session_state.ui_debate_topic

    st.header(f"토론 주제: {debate_topic}")

    messages = st.session_state.debate_messages
    total_rounds = len([m for m in messages if m["role"] == AgentType.PRO])

    # 라운드별로 그룹화하여 표시
    for round_num in range(1, total_rounds + 1):
        st.subheader(f"라운드 {round_num}")

        # 이 라운드의 찬성측 메시지 찾기 (인덱스는 (라운드-1)*2)
        pro_index = (round_num - 1) * 2
        if pro_index < len(messages) and messages[pro_index]["role"] == AgentType.PRO:
            st.markdown("**🙆🏻‍♀️ 찬성 측**")
            st.write(messages[pro_index]["content"])

        # 이 라운드의 반대측 메시지 찾기 (인덱스는 (라운드-1)*2 + 1)
        con_index = (round_num - 1) * 2 + 1
        if con_index < len(messages) and messages[con_index]["role"] == AgentType.CON:
            st.markdown("**🙅🏻‍♂️ 반대 측**")
            st.write(messages[con_index]["content"])

        st.divider()

    # 심판 평가 표시 (마지막 메시지)
    if messages and messages[-1]["role"] == AgentType.JUDGE:
        st.subheader("👩🏼‍⚖️ 최종 평가")
        st.write(messages[-1]["content"])

    def reset_debate():
        st.session_state.debate_active = False
        st.session_state.debate_messages = []

    if st.session_state.ui_show_sources:
        render_source_materials()

    render_control_buttons()


def render_source_materials():
    """참고 자료 렌더링"""
    if st.session_state.retrieved_docs and (
        st.session_state.retrieved_docs.get("pro")
        or st.session_state.retrieved_docs.get("con")
    ):
        with st.expander("사용된 참고 자료 보기"):
            st.subheader("찬성 측 참고 자료")
            for i, doc in enumerate(st.session_state.retrieved_docs.get("pro", [])[:3]):
                st.markdown(f"**출처 {i+1}**")
                st.text(doc[:300] + "..." if len(doc) > 300 else doc)
                st.divider()

            st.subheader("반대 측 참고 자료")
            for i, doc in enumerate(st.session_state.retrieved_docs.get("con", [])[:3]):
                st.markdown(f"**출처 {i+1}**")
                st.text(doc[:300] + "..." if len(doc) > 300 else doc)
                st.divider()


def render_control_buttons():
    """제어 버튼 렌더링"""
    col1, col2 = st.columns(2)

    with col1:
        if st.button("새 토론 시작", use_container_width=True):
            reset_session_state()
            st.rerun()

    # 이력 조회 모드일 때만 삭제 버튼 표시
    if st.session_state.viewing_history and st.session_state.loaded_debate_id:
        with col2:
            if st.button("현재 토론 삭제", use_container_width=True, type="secondary"):
                delete_current_debate()


def delete_current_debate():
    """현재 토론 삭제"""
    from app.database.repository import delete_debate_by_id

    if (
        st.session_state.viewing_history
        and st.session_state.loaded_debate_id
        and delete_debate_by_id(st.session_state.loaded_debate_id)
    ):
        st.success("토론이 삭제되었습니다.")
        reset_session_state()
        st.rerun()
    else:
        st.error("토론 삭제에 실패했습니다.")
