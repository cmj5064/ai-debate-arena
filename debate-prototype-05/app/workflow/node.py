"""
agents.py - 토론 에이전트 관련 기능
"""

import time
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from models.enums import DebateStatus, SpeakerRole
from workflow.state import DebateState
from workflow.config import get_llm
import logging

logger = logging.getLogger(__name__)

ROLE_PRO = "pro"
ROLE_CON = "con"
ROLE_JUDGE = "judge"

ROLE_NAMES = {ROLE_PRO: "찬성 측", ROLE_CON: "반대 측", ROLE_JUDGE: "심판"}


def create_agent_prompt(state: DebateState, agent_type: str) -> str:
    context = state.get("current_context", "")
    topic = state["topic"]

    if agent_type == ROLE_PRO:
        if state["current_round"] == 1:
            return f"""
            당신은 '{topic}'에 대해 찬성 입장을 가진 토론자입니다.
            다음은 이 주제와 관련된 정보입니다:
            {context}
            논리적이고 설득력 있는 찬성 측 주장을 제시해주세요.
            가능한 경우 제공된 정보에서 구체적인 근거를 인용하세요.
            2 ~ 3문단, 각 문단은 100자내로 작성해주세요.
            """
        else:
            previous_messages = [m for m in state["messages"] if m["role"] == "반대 측"]
            if previous_messages:
                last_con_message = previous_messages[-1]["content"]
                return f"""
                당신은 '{topic}'에 대해 찬성 입장을 가진 토론자입니다.
                다음은 이 주제와 관련된 정보입니다:
                {context}
                반대 측의 다음 주장에 대해 반박하고, 찬성 입장을 더 강화해주세요:
                반대 측 주장: "{last_con_message}"
                가능한 경우 제공된 정보에서 구체적인 근거를 인용하세요.
                2 ~ 3문단, 각 문단은 100자내로 작성해주세요.
                """

    elif agent_type == ROLE_CON:
        previous_messages = [m for m in state["messages"] if m["role"] == "찬성 측"]
        if previous_messages:
            last_pro_message = previous_messages[-1]["content"]
            return f"""
            당신은 '{topic}'에 대해 반대 입장을 가진 토론자입니다.
            다음은 이 주제와 관련된 정보입니다:
            {context}
            찬성 측의 다음 주장에 대해 반박하고, 반대 입장을 제시해주세요:
            찬성 측 주장: "{last_pro_message}"
            가능한 경우 제공된 정보에서 구체적인 근거를 인용하세요.
            2 ~ 3문단, 각 문단은 100자내로 작성해주세요.
            """

    elif agent_type == ROLE_JUDGE:
        prompt = f"""
        다음은 '{topic}'에 대한 찬반 토론입니다. 각 측의 주장을 분석하고 평가해주세요.
        다음은 이 주제와 관련된 객관적인 정보입니다:
        {context}
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
        가능한 경우 제공된 객관적 정보를 참고하여 평가해주세요.
        """
        return prompt

    return ""


def get_system_prompt(agent_type: str) -> str:
    prompts = {
        "pro": "당신은 논리적이고 설득력 있는 찬성 측 토론자입니다.",
        "con": "당신은 논리적이고 설득력 있는 반대 측 토론자입니다. 찬성 측 주장에 대해 적극적으로 반박하세요.",
        "judge": "당신은 공정하고 논리적인 토론 심판입니다. 양측의 주장을 면밀히 검토하고 객관적으로 평가해주세요.",
    }
    return prompts.get(agent_type, "")


def stream_response(messages, system_prompt, prompt, role, container):

    llm = get_llm()

    ai_messages = []
    # System Prompt
    ai_messages.append(SystemMessage(content=system_prompt))

    # History
    for msg in messages:
        if msg["role"] == "assistant":
            ai_messages.append(AIMessage(content=msg["content"]))
        else:
            ai_messages.append(HumanMessage(content=f"{msg['role']}: {msg['content']}"))

    # User Prompt
    ai_messages.append(HumanMessage(content=prompt))

    response_area = container.empty()

    # 응답 스트리밍
    full_response = ""
    for chunk in llm.stream(ai_messages):
        if chunk.content:
            full_response += chunk.content
            response_area.markdown(f"**{role}**: {full_response}")
            time.sleep(0.01)
    # 마지막 결과 반환
    return full_response


def process_agent_response(state: DebateState, agent_type: str) -> DebateState:

    system_prompt = get_system_prompt(agent_type)

    container_key = f"{agent_type}_container"

    container = state.get("ui_placeholders", {}).get(container_key)
    status_messages = {
        "pro": "⭕ **찬성 측 의견 생성 중...**",
        "con": "❌ **반대 측 의견 생성 중...**",
        "judge": "⚖️ **심판의 평가 생성 중...**",
    }

    role_names = {"pro": "찬성 측", "con": "반대 측", "judge": "심판"}

    # 프롬프트 생성
    prompt = create_agent_prompt(state, agent_type)

    # 스트리밍 응답 처리
    with container:
        container.markdown(status_messages[agent_type])
        response_content = stream_response(
            state["messages"],
            system_prompt,
            prompt,
            role_names[agent_type],
            container,
        )

    # 상태 업데이트
    new_state = state.copy()
    new_state["messages"].append(
        {"role": role_names[agent_type], "content": response_content}
    )

    # 각 에이전트 특수 상태 업데이트
    if agent_type == ROLE_PRO:
        # 찬성 측이 끝나면 반대 측으로 변경
        new_state["current_speaker"] = SpeakerRole.CON
    elif agent_type == ROLE_CON:
        # 반대 측이 끝나면 다음 라운드로 변경
        new_state["current_round"] += 1
        if new_state["current_round"] <= new_state["max_rounds"]:
            # 다음 라운드가 있으면 찬성 측으로 변경
            new_state["current_speaker"] = SpeakerRole.PRO
        else:
            # 마지막 라운드면 심판으로 변경
            new_state["current_speaker"] = SpeakerRole.JUDGE
    elif agent_type == ROLE_JUDGE:
        new_state["debate_status"] = DebateStatus.COMPLETED
        new_state["current_speaker"] = SpeakerRole.COMPLETED

    return new_state


# 찬성 측 에이전트 노드
def pro_agent(state: DebateState) -> DebateState:
    """찬성 측 에이전트 함수"""
    return process_agent_response(state, ROLE_PRO)


# 반대 측 에이전트 노드
def con_agent(state: DebateState) -> DebateState:
    """반대 측 에이전트 함수"""
    return process_agent_response(state, ROLE_CON)


# 심판 에이전트 노드
def judge_agent(state: DebateState) -> DebateState:
    """심판 에이전트 함수"""
    return process_agent_response(state, ROLE_JUDGE)


# 찬성 측을 위한 검색 노드
def retrieve_pro_info(state: DebateState) -> DebateState:
    """찬성 측 정보 검색 노드"""
    return retrieve_info_for_role(
        state, role=ROLE_PRO, perspective="찬성 장점 이유 근거"
    )


# 반대 측을 위한 검색 노드
def retrieve_con_info(state: DebateState) -> DebateState:
    """반대 측 정보 검색 노드"""
    return retrieve_info_for_role(
        state, role=ROLE_CON, perspective="반대 단점 문제점 근거"
    )


# 심판을 위한 검색 노드
def retrieve_judge_info(state: DebateState) -> DebateState:
    """심판 정보 검색 노드"""
    return retrieve_info_for_role(
        state, role=ROLE_JUDGE, perspective="평가 기준 객관적 사실"
    )


# 관련 정보 검색
def retrieve_relevant_info(query, vector_store, k=3):
    if not vector_store:
        return "", []

    try:
        retrieved_docs = vector_store.similarity_search(query, k=k)

        context = ""
        for i, doc in enumerate(retrieved_docs):
            # 문서 정보 추출
            source = doc.metadata.get("source", "Unknown")
            # 섹션 정보가 있는 경우 추가
            section = doc.metadata.get("section", "")
            # 문서 정보와 내용을 컨텍스트에 추가
            context += f"[문서 {i+1}] 출처: {source}"
            # 섹션 정보가 있는 경우 추가
            if section:
                context += f", 섹션: {section}"
            # 문서 내용 추가
            context += f"\n{doc.page_content}\n\n"

        return context, retrieved_docs
    except Exception as e:
        logger.error(f"Error retrieving info: {str(e)}")
        return "", []


def retrieve_info_for_role(state, role: str, perspective: str):

    base_query = f"{state['topic']} {perspective}"

    if role == ROLE_PRO and state["current_round"] > 1:
        # 이전 반대 측 주장이 있으면 참고
        prev_con_arguments = [m for m in state["messages"] if m["role"] == "반대 측"]
        if prev_con_arguments:
            last_con = prev_con_arguments[-1]["content"]
            base_query += f" {last_con}에 대한 반박"

    elif role == ROLE_CON and state["messages"]:
        # 이전 찬성 측 주장이 있으면 참고
        prev_pro_arguments = [e for e in state["messages"] if e["role"] == "찬성 측"]
        if prev_pro_arguments:
            last_pro = prev_pro_arguments[-1]["content"]
            base_query += f" {last_pro}에 대한 반박"

    context, docs = "", []
    # 벡터 검색을 통해 관련 정보 검색
    if state["vector_store"]:
        k = 2 if role == ROLE_JUDGE else 3
        context, docs = retrieve_relevant_info(base_query, state["vector_store"], k=k)

    new_state = state.copy()
    new_state["current_query"] = base_query
    new_state["current_context"] = context

    # 검색된 문서 저장
    if "retrieved_docs" not in new_state:
        new_state["retrieved_docs"] = {"pro": [], "con": []}

    # 검색 유형에 따라 저장 위치 결정
    if role in [ROLE_PRO, ROLE_CON]:
        new_state["retrieved_docs"][role] = new_state["retrieved_docs"].get(
            role, []
        ) + ([doc.page_content for doc in docs] if docs else [])

    return new_state
