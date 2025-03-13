import time
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from workflow.state import DebateState, AgentType
from workflow.config import get_llm
import logging

logger = logging.getLogger(__name__)


def create_agent_prompt(state: DebateState, agent_type: str) -> str:
    context = state.get("current_context", "")
    topic = state["topic"]

    if agent_type == AgentType.PRO:
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
            previous_messages = [
                m for m in state["messages"] if m["role"] == AgentType.CON
            ]
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

    elif agent_type == AgentType.CON:
        previous_messages = [m for m in state["messages"] if m["role"] == AgentType.PRO]
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

    elif agent_type == AgentType.JUDGE:
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

    # 응답 스트리밍
    full_response = ""
    for chunk in llm.stream(ai_messages):
        if chunk.content:
            full_response += chunk.content
            container.markdown(f"**{role}**: {full_response}")
            time.sleep(0.01)
    # 마지막 결과 반환
    return full_response


def process_agent_response(state: DebateState, agent_type: str) -> DebateState:

    def get_system_prompt(agent_type: str) -> str:
        prompts = {
            AgentType.PRO: "당신은 논리적이고 설득력 있는 찬성 측 토론자입니다.",
            AgentType.CON: "당신은 논리적이고 설득력 있는 반대 측 토론자입니다. 찬성 측 주장에 대해 적극적으로 반박하세요.",
            AgentType.JUDGE: "당신은 공정하고 논리적인 토론 심판입니다. 양측의 주장을 면밀히 검토하고 객관적으로 평가해주세요.",
        }
        return prompts.get(agent_type, "")

    system_prompt = get_system_prompt(agent_type)

    container_key = f"{agent_type}_container"

    container = state.get("ui_placeholders", {}).get(container_key)

    status_messages = {
        AgentType.PRO: "⭕ **찬성 측 의견 생성 중...**",
        AgentType.CON: "❌ **반대 측 의견 생성 중...**",
        AgentType.JUDGE: "⚖️ **심판의 평가 생성 중...**",
    }

    # 프롬프트 생성
    prompt = create_agent_prompt(state, agent_type)

    # 스트리밍 응답 처리
    with container:
        container.markdown(status_messages[agent_type])
        response_content = stream_response(
            state["messages"],
            system_prompt,
            prompt,
            agent_type,
            container,
        )

    # 상태 업데이트
    new_state = state.copy()
    new_state["messages"].append({"role": agent_type, "content": response_content})
    if agent_type == AgentType.CON:
        new_state["current_round"] += 1
    return new_state


# 찬성 측 에이전트 노드
def pro_agent(state: DebateState) -> DebateState:
    logger.info("pro_agent")
    return process_agent_response(state, AgentType.PRO)


# 반대 측 에이전트 노드
def con_agent(state: DebateState) -> DebateState:
    return process_agent_response(state, AgentType.CON)


# 심판 에이전트 노드
def judge_agent(state: DebateState) -> DebateState:
    return process_agent_response(state, AgentType.JUDGE)
