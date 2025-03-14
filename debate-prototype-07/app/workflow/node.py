from langchain.schema import HumanMessage, SystemMessage, AIMessage
from app.workflow.config import get_llm
from app.workflow.state import DebateState, AgentType


# 찬성 측 에이전트 노드
def pro_agent(state: DebateState) -> DebateState:
    # 시스템 프롬프트 설정
    system_prompt = "당신은 논리적이고 설득력 있는 찬성 측 토론자입니다."

    # 메시지 준비
    messages = [SystemMessage(content=system_prompt)]

    # 기존 대화 기록 추가
    for message in state["messages"]:
        if message["role"] == "assistant":
            messages.append(AIMessage(content=message["content"]))
        else:
            messages.append(
                HumanMessage(content=f"{message['role']}: {message['content']}")
            )

    # 프롬프트 구성
    if state["current_round"] == 1:
        prompt = f"""
        당신은 '{state['topic']}'에 대해 찬성 입장을 가진 토론자입니다.
        논리적이고 설득력 있는 찬성 측 주장을 제시해주세요.
        2 ~ 3문단, 각 문단은 100자내로 작성해주세요.
        """
    else:
        # 이전 발언자의 마지막 메시지를 가져옴
        previous_messages = [m for m in state["messages"] if m["role"] == AgentType.CON]
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
    response = get_llm().invoke(messages)

    # 상태 업데이트
    new_state = state.copy()
    new_state["messages"].append({"role": AgentType.PRO, "content": response.content})
    return new_state


# 반대 측 에이전트 노드
def con_agent(state: DebateState) -> DebateState:
    """반대 측 에이전트 함수"""
    # 시스템 프롬프트 설정
    system_prompt = "당신은 논리적이고 설득력 있는 반대 측 토론자입니다. 찬성 측 주장에 대해 적극적으로 반박하세요."

    # 메시지 준비
    messages = [SystemMessage(content=system_prompt)]

    # 기존 대화 기록 추가
    for message in state["messages"]:
        if message["role"] == "assistant":
            messages.append(AIMessage(content=message["content"]))
        else:
            messages.append(
                HumanMessage(content=f"{message['role']}: {message['content']}")
            )

    # 프롬프트 구성
    # 찬성 측 마지막 메시지를 가져옴
    previous_messages = [m for m in state["messages"] if m["role"] == AgentType.PRO]
    last_pro_message = previous_messages[-1]["content"]
    prompt = f"""
    당신은 '{state['topic']}'에 대해 반대 입장을 가진 토론자입니다.
    찬성 측의 다음 주장에 대해 반박하고, 반대 입장을 제시해주세요:

    찬성 측 주장: "{last_pro_message}"

    2 ~ 3문단, 각 문단은 100자내로 작성해주세요.
    """

    # LLM에 요청
    messages.append(HumanMessage(content=prompt))
    response = get_llm().invoke(messages)

    # 상태 업데이트
    new_state = state.copy()
    new_state["messages"].append({"role": AgentType.CON, "content": response.content})
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
    for message in state["messages"]:
        role_text = AgentType.to_korean(message["role"])
        prompt += f"\n\n{role_text}: {message['content']}"

    prompt += """
    
    위 토론을 분석하여 다음을 포함하는 심사 평가를 해주세요:
    1. 양측 주장의 핵심 요약
    2. 각 측이 사용한 주요 논리와 증거의 강점과 약점
    3. 전체 토론의 승자와 그 이유
    4. 양측 모두에게 개선점 제안
    """

    # LLM에 요청
    messages.append(HumanMessage(content=prompt))
    response = get_llm().invoke(messages)

    # 상태 업데이트
    new_state = state.copy()
    new_state["messages"].append({"role": AgentType.JUDGE, "content": response.content})
    return new_state


# 찬성 측을 위한 검색 노드
def retrieve_pro_info(state: DebateState) -> DebateState:
    """찬성 측 정보 검색 노드"""
    return retrieve_info_for_role(
        state, role=AgentType.PRO, perspective="찬성 장점 이유 근거"
    )


# 반대 측을 위한 검색 노드
def retrieve_con_info(state: DebateState) -> DebateState:
    """반대 측 정보 검색 노드"""
    return retrieve_info_for_role(
        state, role=AgentType.CON, perspective="반대 단점 문제점 근거"
    )


# 심판을 위한 검색 노드
def retrieve_judge_info(state: DebateState) -> DebateState:
    """심판 정보 검색 노드"""
    return retrieve_info_for_role(
        state, role=AgentType.JUDGE, perspective="평가 기준 객관적 사실"
    )


# 역할에 따른 정보 검색
def retrieve_info_for_role(state, role: str, perspective: str):

    base_query = f"{state['topic']} {perspective}"

    if role == AgentType.PRO and state["current_round"] > 1:
        # 이전 반대 측 주장이 있으면 참고
        prev_con_arguments = [m for m in state["messages"] if m["role"] == "반대 측"]
        if prev_con_arguments:
            last_con = prev_con_arguments[-1]["content"]
            base_query += f" {last_con}에 대한 반박"

    elif role == AgentType.CON and state["messages"]:
        # 이전 찬성 측 주장이 있으면 참고
        prev_pro_arguments = [e for e in state["messages"] if e["role"] == "찬성 측"]
        if prev_pro_arguments:
            last_pro = prev_pro_arguments[-1]["content"]
            base_query += f" {last_pro}에 대한 반박"

    context, docs = "", []
    # 벡터 검색을 통해 관련 정보 검색
    if state["vector_store"]:
        k = 2 if role == AgentType.JUDGE else 3
        context, docs = retrieve_relevant_info(base_query, state["vector_store"], k=k)

    new_state = state.copy()
    new_state["current_query"] = base_query
    new_state["current_context"] = context

    # 검색된 문서 저장
    if "retrieved_docs" not in new_state:
        new_state["retrieved_docs"] = {"pro": [], "con": []}

    # 검색 유형에 따라 저장 위치 결정
    if role in [AgentType.PRO, AgentType.CON]:
        new_state["retrieved_docs"][role] = new_state["retrieved_docs"].get(
            role, []
        ) + ([doc.page_content for doc in docs] if docs else [])

    return new_state


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
    except Exception:
        return "", []
