"""
위키피디아 데이터 수집 - 토론 주제에 대한 위키피디아 정보 검색
"""

import streamlit as st
import wikipedia
from langchain.schema import Document
from typing import List, Literal
from retrieval.query_generator import improve_search_query


def get_wikipedia_content(
    topic: str,
    language: str = "ko",
    search_type: Literal["pro", "con", "general"] = "general",
) -> List[Document]:

    try:
        wikipedia.set_lang(language)

        # LLM을 사용하여 검색어 개선
        improved_queries = improve_search_query(topic, search_type, language)

        documents = []

        # 각 개선된 검색어에 대해 검색 수행
        for query in improved_queries:
            # 검색 결과 가져오기
            search_results = wikipedia.search(query, results=3)

            if not search_results:
                continue

            # 검색 결과에 대한 위키피디아 페이지 정보 수집
            for page_title in search_results:
                try:
                    # 페이지 정보 가져오기
                    page = wikipedia.page(page_title, auto_suggest=False)

                    # 요약 추가
                    if page.summary:
                        documents.append(
                            Document(
                                page_content=page.summary,
                                metadata={
                                    "source": "wikipedia",
                                    "section": "summary",
                                    "topic": page_title,
                                    "query": query,
                                },
                            )
                        )

                    # 본문 섹션 추가
                    content = page.content
                    if content:
                        max_length = 5000
                        if len(content) > max_length:
                            content = content[:max_length]

                        documents.append(
                            Document(
                                page_content=content,
                                metadata={
                                    "source": "wikipedia",
                                    "section": "content",
                                    "topic": page_title,
                                    "query": query,
                                },
                            )
                        )
                except Exception as e:
                    st.warning(f"위키피디아 페이지 정보 수집 중 오류 발생: {str(e)}")

        return documents

    except Exception as e:
        st.error(f"위키피디아 검색 중 오류 발생: {str(e)}")
        return []
