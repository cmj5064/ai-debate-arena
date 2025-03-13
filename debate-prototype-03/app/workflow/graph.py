from workflow.state import DebateState, AgentType
from workflow.node import pro_agent, con_agent, judge_agent
from langgraph.graph import StateGraph, END


"""
    토론 그래프 구조:
    
    [시작점]
        |
    [PRO AGENT] --┐
        |         |
    [agent_router]--┐
        |          |
    [CON AGENT] ---┤
        |          |
    [agent_router]--┐
        |          |
    [INCREMENT_ROUND]
        |
    [check_rounds] -------┐
        |                |
        | (라운드 < max)   |  (라운드 > max)
        ▼                ▼
    [PRO AGENT]      [JUDGE AGENT]
                         |
                         ▼
                     [agent_router]
                         |
                         ▼
                       [END]
"""


def create_debate_graph() -> StateGraph:
    # 그래프 생성
    workflow = StateGraph(DebateState)

    # 기존 노드 추가
    workflow.add_node(AgentType.PRO, pro_agent)
    workflow.add_node(AgentType.CON, con_agent)
    workflow.add_node(AgentType.JUDGE, judge_agent)

    def increment_round(state: DebateState) -> DebateState:
        new_state = state.copy()
        new_state["current_round"] = state["current_round"] + 1
        return new_state

    workflow.add_node("INCREMENT_ROUND", increment_round)

    def check_rounds(state: DebateState) -> str:
        """라운드 수를 확인하고 다음 단계를 결정하는 노드"""
        if state["current_round"] > state["max_rounds"]:
            return AgentType.JUDGE
        return AgentType.PRO

    def agent_router(state: DebateState) -> str:
        """토론자 순서를 결정하는 라우터"""
        messages = state["messages"]
        if not messages:
            return AgentType.PRO

        last_speaker_role = messages[-1]["role"]

        if last_speaker_role == AgentType.PRO:
            return AgentType.CON
        elif last_speaker_role == AgentType.CON:
            return "INCREMENT_ROUND"
        elif last_speaker_role == AgentType.JUDGE:
            return END
        else:
            print(f"WARNING: Unexpected speaker role: {last_speaker_role}")
            return AgentType.PRO

    workflow.add_conditional_edges(AgentType.PRO, agent_router)
    workflow.add_conditional_edges(AgentType.CON, agent_router)
    workflow.add_conditional_edges(AgentType.JUDGE, agent_router)
    workflow.add_conditional_edges("INCREMENT_ROUND", check_rounds)

    workflow.set_entry_point(AgentType.PRO)

    return workflow.compile()
