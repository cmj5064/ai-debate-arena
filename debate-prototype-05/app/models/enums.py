"""
models.py - 데이터 모델 및 Enum 정의
"""

from enum import Enum, auto


# 토론자 역할을 위한 Enum
class SpeakerRole(Enum):
    PRO = "pro_agent"
    CON = "con_agent"
    JUDGE = "judge"
    COMPLETED = "completed"


# 토론 상태를 위한 Enum
class DebateStatus(Enum):
    ACTIVE = auto()
    JUDGED = auto()
    COMPLETED = auto()
