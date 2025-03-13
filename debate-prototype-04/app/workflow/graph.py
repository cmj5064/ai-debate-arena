from workflow.state import DebateState, AgentType
from workflow.node import pro_agent, con_agent, judge_agent
from langgraph.graph import StateGraph, END


def create_workflow():

    # 메인 그래프 생성
    workflow = StateGraph(DebateState)

    # 에이전트 노드 추가
    workflow.add_node(AgentType.PRO, pro_agent)
    workflow.add_node(AgentType.CON, con_agent)
    workflow.add_node(AgentType.JUDGE, judge_agent)

    # 라우팅 엣지 추가
    workflow.set_entry_point(AgentType.PRO)

    # 라우터 함수: 토론의 흐름 결정
    def router(state: DebateState) -> str:
        """토론 흐름을 결정하는 라우터 함수"""

        # 1. 메시지 존재 확인
        messages = state["messages"]
        if not messages:
            return AgentType.PRO  # 메시지가 없으면 PRO부터 시작

        # 2. 마지막 메시지 확인
        last_speaker_role = messages[-1]["role"]

        # 3. 일반적인 토론 흐름 처리
        if last_speaker_role == AgentType.PRO:
            return AgentType.CON  # 찬성 다음은 항상 반대
        elif last_speaker_role == AgentType.CON:
            # 마지막 라운드에 도달했는지 확인
            if state["current_round"] > state["max_rounds"]:
                return AgentType.JUDGE  # 마지막 라운드 도달 시 심판으로
            return AgentType.PRO  # 아니면 다음 라운드의 찬성으로

        # 4. 심판 케이스 처리
        elif last_speaker_role == AgentType.JUDGE:
            return END  # 심판 발언 후 종료

        # 5. 예상치 못한 상태 처리 (명시적으로 오류 상황 처리)
        else:
            print(f"WARNING: Unexpected speaker role: {last_speaker_role}")
            return AgentType.PRO  # 기본값으로 찬성 측 선택

    workflow.add_conditional_edges(AgentType.PRO, router)
    workflow.add_conditional_edges(AgentType.CON, router)
    workflow.add_conditional_edges(AgentType.JUDGE, router)

    return workflow


def create_debate_graph() -> StateGraph:

    workflow = create_workflow()
    return workflow.compile()
