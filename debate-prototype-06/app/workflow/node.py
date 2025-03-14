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
