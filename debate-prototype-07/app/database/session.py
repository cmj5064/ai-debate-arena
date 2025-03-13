import os
import logging
import streamlit as st
from sqlalchemy import create_engine
from contextlib import contextmanager

from database.model import Base

logger = logging.getLogger(__name__)

DB_PATH = os.getenv("DB_PATH")
CONNECTION_NAME = "sqlite_db"


def initialize_database():
    # 데이터베이스 스키마 초기화
    logger.info("데이터베이스 스키마 초기화 중...")
    engine = create_engine(f"sqlite:///{DB_PATH}")
    Base.metadata.create_all(engine)
    logger.info("데이터베이스 초기화 완료")


@st.cache_resource
def get_connection():
    # Streamlit의 캐시를 사용하여 연결을 캐시
    return st.connection(CONNECTION_NAME, type="sql", url=f"sqlite:///{DB_PATH}")


def get_session():
    return get_connection().session


@contextmanager
def get_db_session():
    session = get_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
