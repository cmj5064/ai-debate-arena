# LangGraph 상태 정의 - RAG 관련 필드 추가
from typing import Any, Dict, List, TypedDict


class AgentType:
    PRO = "pro_agent"
    CON = "con_agent"
    JUDGE = "judge"
    COMPLETED = "END"


class DebateState(TypedDict):
    topic: str
    messages: List[Dict]
    current_round: int
    max_rounds: int
    current_speaker: AgentType
    ui_placeholders: Dict[str, Any]  # UI 스트리밍을 위한 플레이스홀더
