# LangGraph 상태 정의 - RAG 관련 필드 추가
from typing import Any, Dict, List, TypedDict

from models.enums import DebateStatus, SpeakerRole


class DebateState(TypedDict):
    topic: str
    messages: List[Dict]
    current_round: int
    max_rounds: int
    current_speaker: SpeakerRole
    debate_status: DebateStatus
    vector_store: object  # RAG 벡터 스토어
    retrieved_docs: Dict[str, List]  # RAG 검색 결과
    current_query: str  # 현재 검색 쿼리
    current_context: str  # 검색된 컨텍스트
    ui_placeholders: Dict[str, Any]  # UI 스트리밍을 위한 플레이스홀더
