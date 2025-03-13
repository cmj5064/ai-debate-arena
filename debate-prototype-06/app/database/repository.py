import datetime
import json
import logging
from typing import List, Tuple, Dict

from database.model import Debate
from database.session import get_db_session
from database.exceptions import DatabaseError

logger = logging.getLogger(__name__)


def save_debate_to_db(topic: str, rounds: int, messages: List[Dict]) -> int:
    try:
        with get_db_session() as session:
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            messages_json = json.dumps(messages, ensure_ascii=False)

            debate = Debate(
                topic=topic, date=now, rounds=rounds, messages=messages_json
            )

            session.add(debate)
            session.commit()
        return True
    except Exception as e:
        logger.error(f"토론 저장 중 오류 발생: {str(e)}")
        raise DatabaseError(f"토론 저장 오류: {str(e)}") from e


def fetch_debate_history() -> List[Tuple[int, str, str, int]]:
    try:
        with get_db_session() as session:
            debates = (
                session.query(Debate.id, Debate.topic, Debate.date, Debate.rounds)
                .order_by(Debate.date.desc())
                .all()
            )
            return [(d.id, d.topic, d.date, d.rounds) for d in debates]
    except Exception as e:
        logger.error(f"토론 이력 조회 중 오류 발생: {str(e)}")
        raise DatabaseError(f"토론 이력 조회 오류: {str(e)}") from e


def fetch_debate_by_id(debate_id: int) -> Tuple[str, List[Dict], Dict]:

    try:
        with get_db_session() as session:
            debate = session.query(Debate).filter(Debate.id == debate_id).first()

            if debate:
                messages = json.loads(debate.messages)
                return debate.topic, messages
            return None, None, None
    except json.JSONDecodeError as e:
        logger.error(f"JSON 디코딩 오류: {str(e)}")
        raise DatabaseError(f"토론 데이터 변환 오류: {str(e)}") from e
    except Exception as e:
        logger.error(f"토론 불러오기 중 오류 발생: {str(e)}")
        raise DatabaseError(f"토론 불러오기 오류: {str(e)}") from e


def delete_debate_by_id(debate_id: int) -> bool:
    try:
        with get_db_session() as session:
            result = session.query(Debate).filter(Debate.id == debate_id).delete()
            return result > 0
    except Exception as e:
        logger.error(f"토론 삭제 중 오류 발생: {str(e)}")
        raise DatabaseError(f"토론 삭제 오류: {str(e)}") from e


def delete_all_debates() -> int:
    try:
        with get_db_session() as session:
            result = session.query(Debate).delete()
            return result
    except Exception as e:
        logger.error(f"전체 토론 삭제 중 오류 발생: {str(e)}")
        raise DatabaseError(f"전체 토론 삭제 오류: {str(e)}") from e
