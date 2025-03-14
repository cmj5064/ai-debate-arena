# LangGraph 상태 정의
from typing import Dict, List, TypedDict


class AgentType:
    PRO = "PRO_AGENT"
    CON = "CON_AGENT"
    JUDGE = "JUDGE_AGENT"

    @classmethod
    def to_korean(cls, role: str) -> str:
        if role == cls.PRO:
            return "찬성"
        elif role == cls.CON:
            return "반대"
        elif role == cls.JUDGE:
            return "심판"
        else:
            return role


class DebateState(TypedDict):

    topic: str
    messages: List[Dict]
    current_round: int
    max_rounds: int
    current_speaker: AgentType
    vector_store: object  # RAG 벡터 스토어
    retrieved_docs: Dict[str, List]  # RAG 검색 결과
    current_query: str  # 현재 검색 쿼리
    current_context: str  # 검색된 컨텍스트
