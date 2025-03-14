from langchain.schema import HumanMessage, SystemMessage
from typing import List, Literal
from workflow.config import get_llm


def improve_search_query(
    topic: str,
    search_type: Literal["pro", "con", "general"] = "general",
    language: str = "en",
) -> List[str]:
    llm = get_llm()
    prompt_by_type = {
        "pro": f"'{topic}'에 대해 찬성하는 입장을 뒷받침할 수 있는 사실과 정보를 찾고자 합니다. 위키피디아 검색에 적합한 3개의 검색어를 제안해주세요. 각 검색어는 25자 이내로 작성하고 콤마로 구분하세요. 검색어만 제공하고 설명은 하지 마세요.",
        "con": f"'{topic}'에 대해 반대하는 입장을 뒷받침할 수 있는 사실과 정보를 찾고자 합니다. 위키피디아 검색에 적합한 3개의 검색어를 제안해주세요. 각 검색어는 25자 이내로 작성하고 콤마로 구분하세요. 검색어만 제공하고 설명은 하지 마세요.",
        "general": f"'{topic}'에 대한 객관적인 사실과 정보를 찾고자 합니다. 위키피디아 검색에 적합한 3개의 검색어를 제안해주세요. 각 검색어는 25자 이내로 작성하고 콤마로 구분하세요. 검색어만 제공하고 설명은 하지 마세요.",
    }

    messages = [
        SystemMessage(
            content="당신은 검색 전문가입니다. 주어진 주제에 대해 가장 관련성 높은 검색어를 제안해주세요."
        ),
        HumanMessage(content=prompt_by_type[search_type]),
    ]

    # 스트리밍 응답 받기
    response = llm.invoke(messages)

    suggested_queries = [q.strip() for q in response.content.split(",")]

    return suggested_queries[:3]
