from typing import Literal
from models.enums import DebateStatus, SpeakerRole
from workflow.state import DebateState
from workflow.node import (
    pro_agent,
    con_agent,
    judge_agent,
    retrieve_con_info,
    retrieve_judge_info,
    retrieve_pro_info,
)
from langgraph.graph import StateGraph, END


def create_workflow():
    """워크플로우 생성 (컴파일 전)"""
    # 메인 그래프 생성
    workflow = StateGraph(DebateState)

    # 검색 노드 추가
    workflow.add_node("retrieve_pro_info", retrieve_pro_info)
    workflow.add_node("retrieve_con_info", retrieve_con_info)
    workflow.add_node("retrieve_judge_info", retrieve_judge_info)

    # 에이전트 노드 추가
    workflow.add_node("pro_agent", pro_agent)
    workflow.add_node("con_agent", con_agent)
    workflow.add_node("judge", judge_agent)

    # 라우팅 엣지 추가
    workflow.set_entry_point("retrieve_pro_info")

    # 검색 노드에서 에이전트 노드로 라우팅
    workflow.add_edge("retrieve_pro_info", "pro_agent")
    workflow.add_edge("retrieve_con_info", "con_agent")
    workflow.add_edge("retrieve_judge_info", "judge")

    # 라우터 함수
    def router(
        state: DebateState,
    ) -> Literal[
        "retrieve_pro_info",
        "retrieve_con_info",
        "retrieve_judge_info",
        "END",
    ]:
        if state["debate_status"] == DebateStatus.COMPLETED:
            return "END"

        current_speaker = state["current_speaker"]

        # 현재 화자에 따라 검색 노드로 먼저 라우팅
        if current_speaker == SpeakerRole.PRO:
            return "retrieve_pro_info"
        elif current_speaker == SpeakerRole.CON:
            return "retrieve_con_info"
        elif current_speaker == SpeakerRole.JUDGE:
            return "retrieve_judge_info"
        elif current_speaker == SpeakerRole.COMPLETED:
            return "END"

    router_map = {
        "retrieve_pro_info": "retrieve_pro_info",
        "retrieve_con_info": "retrieve_con_info",
        "retrieve_judge_info": "retrieve_judge_info",
        "END": END,
    }
    workflow.add_conditional_edges("pro_agent", router, router_map)
    workflow.add_conditional_edges("con_agent", router, router_map)
    workflow.add_conditional_edges("judge", router, router_map)

    return workflow


def create_debate_graph() -> StateGraph:

    workflow = create_workflow()
    return workflow.compile()
