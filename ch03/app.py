# app_langgraph.py - 챕터 3: LangGraph를 활용한 AI 토론 시스템 구현
from enum import Enum, auto
import streamlit as st
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END
from typing import Dict, List, TypedDict, Literal
import os
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드
load_dotenv()

# 페이지 설정
st.set_page_config(page_title="AI 토론", page_icon="🤖", layout="wide")

# 제목 및 소개
st.title("🤖 AI 토론 - LangGraph 버전")
st.markdown(
    """
    ### 프로젝트 소개
    이 애플리케이션은 LangGraph를 활용하여 AI 에이전트 간의 토론 워크플로우를 구현합니다.
    찬성 측, 반대 측, 그리고 심판 역할의 AI가 주어진 주제에 대해 체계적으로 토론을 진행합니다.
    """
)

# LLM 설정
llm = AzureChatOpenAI(
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15"),
    temperature=0.7,
)


# 토론자 역할을 위한 Enum
class SpeakerRole(Enum):
    PRO = "pro_agent"
    CON = "con_agent"
    JUDGE = "judge"
    COMPLETED = "completed"


# 토론 상태를 위한 Enum
class DebateStatus(Enum):
    ACTIVE = auto()
    JUDGED = auto()
    COMPLETED = auto()


# LangGraph 상태 정의
class DebateState(TypedDict):

    topic: str
    messages: List[Dict]
    current_round: int
    max_rounds: int
    current_speaker: SpeakerRole
    debate_status: DebateStatus


# 찬성 측 에이전트 노드
def pro_agent(state: DebateState) -> DebateState:
    """찬성 측 에이전트 함수"""
    # 시스템 프롬프트 설정
    system_prompt = "당신은 논리적이고 설득력 있는 찬성 측 토론자입니다."

    # 메시지 준비
    messages = [SystemMessage(content=system_prompt)]

    # 기존 대화 기록 추가
    for msg in state["messages"]:
        if msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
        else:
            messages.append(HumanMessage(content=f"{msg['role']}: {msg['content']}"))

    # 프롬프트 구성
    if state["current_round"] == 1:
        prompt = f"""
        당신은 '{state['topic']}'에 대해 찬성 입장을 가진 토론자입니다.
        논리적이고 설득력 있는 찬성 측 주장을 제시해주세요.
        2 ~ 3문단, 각 문단은 100자내로 작성해주세요.
        """
    else:
        # 이전 발언자의 마지막 메시지를 가져옴
        previous_messages = [m for m in state["messages"] if m["role"] == "반대 측"]
        if previous_messages:
            last_con_message = previous_messages[-1]["content"]
            prompt = f"""
            당신은 '{state['topic']}'에 대해 찬성 입장을 가진 토론자입니다.
            반대 측의 다음 주장에 대해 반박하고, 찬성 입장을 더 강화해주세요:

            반대 측 주장: "{last_con_message}"

            2 ~ 3문단, 각 문단은 100자내로 작성해주세요.
            """

    # LLM에 요청
    messages.append(HumanMessage(content=prompt))
    response = llm.invoke(messages)

    # 상태 업데이트
    # TODO: new_state 셋팅
    new_state = state.copy()
    new_state["messages"].append({"role": "찬성 측", "content": response.content})
    new_state["current_speaker"] = SpeakerRole.CON
    return new_state


# 반대 측 에이전트 노드
def con_agent(state: DebateState) -> DebateState:
    """반대 측 에이전트 함수"""
    # 시스템 프롬프트 설정
    system_prompt = "당신은 논리적이고 설득력 있는 반대 측 토론자입니다. 찬성 측 주장에 대해 적극적으로 반박하세요."

    # 메시지 준비
    messages = [SystemMessage(content=system_prompt)]

    # 기존 대화 기록 추가
    for msg in state["messages"]:
        if msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
        else:
            messages.append(HumanMessage(content=f"{msg['role']}: {msg['content']}"))

    # 프롬프트 구성
    # 찬성 측 마지막 메시지를 가져옴
    previous_messages = [m for m in state["messages"] if m["role"] == "찬성 측"]
    last_pro_message = previous_messages[-1]["content"]
    prompt = f"""
    당신은 '{state['topic']}'에 대해 반대 입장을 가진 토론자입니다.
    찬성 측의 다음 주장에 대해 반박하고, 반대 입장을 제시해주세요:

    찬성 측 주장: "{last_pro_message}"

    2 ~ 3문단, 각 문단은 100자내로 작성해주세요.
    """

    # LLM에 요청
    messages.append(HumanMessage(content=prompt))
    response = llm.invoke(messages)

    # 상태 업데이트
    # TODO: new_state 셋팅
    new_state = state.copy()
    new_state["messages"].append({"role": "반대 측", "content": response.content})
    new_state["current_round"] += 1

    # 다음 라운드 여부 결정
    if new_state["current_round"] <= new_state["max_rounds"]:
        new_state["current_speaker"] = SpeakerRole.PRO
    else:
        new_state["current_speaker"] = SpeakerRole.JUDGE

    return new_state


# 심판 에이전트 노드
def judge_agent(state: DebateState) -> DebateState:
    """심판 에이전트 함수"""
    # 시스템 프롬프트 설정
    system_prompt = "당신은 공정하고 논리적인 토론 심판입니다. 양측의 주장을 면밀히 검토하고 객관적으로 평가해주세요."

    # 메시지 준비
    messages = [SystemMessage(content=system_prompt)]

    # 프롬프트 구성
    prompt = f"""
    다음은 '{state['topic']}'에 대한 찬반 토론입니다. 각 측의 주장을 분석하고 평가해주세요.

    토론 내용:
    """
    for msg in state["messages"]:
        prompt += f"\n\n{msg['role']}: {msg['content']}"

    prompt += """
    
    위 토론을 분석하여 다음을 포함하는 심사 평가를 해주세요:
    1. 양측 주장의 핵심 요약
    2. 각 측이 사용한 주요 논리와 증거의 강점과 약점
    3. 전체 토론의 승자와 그 이유
    4. 양측 모두에게 개선점 제안
    """

    # LLM에 요청
    messages.append(HumanMessage(content=prompt))
    response = llm.invoke(messages)

    # 상태 업데이트
    # TODO: new_state 셋팅
    new_state = state.copy()
    new_state["messages"].append({"role": "심판", "content": response.content})
    new_state["debate_status"] = DebateStatus.COMPLETED
    new_state["current_speaker"] = SpeakerRole.COMPLETED
    return new_state


# 라우터 함수: 다음 노드 결정
def router(state: DebateState) -> Literal["pro_agent", "con_agent", "judge", "END"]:
    if state["debate_status"] == DebateStatus.COMPLETED:
        return "END"
    return state["current_speaker"].value


# LangGraph 워크플로우 정의
def create_debate_graph() -> StateGraph:
    # 그래프 생성
    workflow = StateGraph(DebateState)

    # TODO: 노드 추가
    workflow.add_node("pro_agent", pro_agent)
    workflow.add_node("con_agent", con_agent)
    workflow.add_node("judge", judge_agent)

    # TODO: routing_map 정의
    routing_map = {
       SpeakerRole.PRO.value: "pro_agent",
       SpeakerRole.CON.value: "con_agent",
       SpeakerRole.JUDGE.value: "judge",
       "END": END
    }

    # 엣지 정의
    workflow.add_conditional_edges(
        SpeakerRole.PRO.value,
        router,
        routing_map,
    )

    workflow.add_conditional_edges(
        SpeakerRole.CON.value,
        router,
        routing_map,
    )

    workflow.add_conditional_edges(
        SpeakerRole.JUDGE.value,
        router,
        routing_map,
    )

    # TODO: 시작 노드 설정
    workflow.set_entry_point(SpeakerRole.PRO.value)

    # TODO: 그래프 컴파일
    return workflow.compile()


# Streamlit 애플리케이션 로직
# 세션 스테이트 초기화
if "debate_active" not in st.session_state:
    st.session_state.debate_active = False
    st.session_state.debate_messages = []

# 토론 주제 입력 섹션
st.header("토론 주제 입력")
debate_topic = st.text_input(
    "토론 주제를 입력하세요:", "인공지능이 인간의 일자리를 대체해야 한다"
)
max_rounds = st.slider("토론 라운드 수", min_value=1, max_value=5, value=1)

# 토론 시작 버튼
if not st.session_state.debate_active and st.button("토론 시작"):
    # 그래프 생성
    debate_graph = create_debate_graph()

    # 초기 상태 설정
    initial_state: DebateState = {
        "topic": debate_topic,
        "messages": [],
        "current_round": 1,
        "max_rounds": max_rounds,
        "current_speaker": SpeakerRole.PRO,
        "debate_status": DebateStatus.ACTIVE,
    }

    # 토론 시작
    with st.spinner("토론이 진행 중입니다... 완료까지 잠시 기다려주세요."):
        # 그래프 실행 - stream 대신 invoke 사용
        result = debate_graph.invoke(initial_state)

        # 결과를 세션 스테이트에 저장
        st.session_state.debate_messages = result["messages"]
        st.session_state.debate_active = True

    # 페이지 새로고침하여 결과 표시
    st.rerun()

# 토론 내용 표시
if st.session_state.debate_active:
    # 토론 주제 표시
    st.header(f"토론 주제: {debate_topic}")

    # 토론 내용 표시
    st.header("토론 진행 상황")

    messages = st.session_state.debate_messages
    total_rounds = len([m for m in messages if m["role"] == "찬성 측"])

    # 라운드별로 그룹화하여 표시
    for round_num in range(1, total_rounds + 1):
        st.subheader(f"라운드 {round_num}")

        # 이 라운드의 찬성측 메시지 찾기 (인덱스는 (라운드-1)*2)
        pro_index = (round_num - 1) * 2
        if pro_index < len(messages) and messages[pro_index]["role"] == "찬성 측":
            st.markdown("**찬성 측:**")
            st.write(messages[pro_index]["content"])

        # 이 라운드의 반대측 메시지 찾기 (인덱스는 (라운드-1)*2 + 1)
        con_index = (round_num - 1) * 2 + 1
        if con_index < len(messages) and messages[con_index]["role"] == "반대 측":
            st.markdown("**반대 측:**")
            st.write(messages[con_index]["content"])

        st.divider()

    # 심판 평가 표시 (마지막 메시지)
    if messages and messages[-1]["role"] == "심판":
        st.subheader("최종 평가")
        st.write(messages[-1]["content"])

    # 다시 시작 버튼
    if st.button("새 토론 시작"):
        st.session_state.debate_active = False
        st.session_state.debate_messages = []
        st.rerun()
