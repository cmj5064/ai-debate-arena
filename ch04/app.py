# app_langgraph.py - 챕터 3: LangGraph를 활용한 AI 토론 시스템 구현
from enum import Enum, auto
import streamlit as st
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import HumanMessage, SystemMessage, AIMessage, Document
from langgraph.graph import StateGraph, END
from typing import Dict, List, TypedDict, Literal
import os
from dotenv import load_dotenv
import wikipedia

# .env 파일에서 환경 변수 로드
load_dotenv()

# 페이지 설정
st.set_page_config(page_title="AI 토론", page_icon="🤖", layout="wide")

# 제목 및 소개
st.title("🤖 AI 토론 - LangGraph & RAG 버전")
st.markdown(
    """
    ### 프로젝트 소개
    이 애플리케이션은 LangGraph를 활용하여 AI 에이전트 간의 토론 워크플로우를 구현합니다.
    찬성 측, 반대 측, 그리고 심판 역할의 AI가 주어진 주제에 대해 체계적으로 토론을 진행합니다.
    RAG(Retrieval-Augmented Generation)를 통해 외부 지식을 검색하여 더 강력한 논리를 펼칩니다.
    """
)

# LLM 설정
llm = AzureChatOpenAI(
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15"),
    temperature=0.7,
)

# 임베딩 설정
embeddings = AzureOpenAIEmbeddings(
    model=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
    openai_api_version="2024-02-01",
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)


# 위키피디아 데이터 수집 함수
def get_wikipedia_content(topic, language="en"):
    try:
        # 언어 설정
        wikipedia.set_lang(language)

        # 검색 결과 가져오기 (먼저 관련 페이지 찾기)
        # search_results = # TODO> wikipedia search
        search_results = wikipedia.search(topic)

        if not search_results:
            st.warning(f"{topic}에 대한 위키피디아 페이지를 찾을 수 없습니다.")
            return []

        # 디버깅용 메시지
        st.info(f"'{topic}'에 대한 검색 결과: {', '.join(search_results[:3])}...")

        documents = []

        # 최대 3개의 가장 관련성 높은 페이지 처리
        for i, page_title in enumerate(search_results[:3]):
            try:
                # 페이지 정보 가져오기
                page = wikipedia.page(page_title, auto_suggest=False)

                # 요약 추가
                if page.summary:
                    documents.append(
                        Document(
                            page_content=page.summary,
                            metadata={
                                "source": f"wikipedia-{language}",
                                "section": "summary",
                                "topic": page_title,
                            },
                        )
                    )

                # 본문 섹션 추가 (일부만)
                content = page.content
                if content:
                    # 본문을 적당한 길이로 잘라서 추가
                    max_length = 5000  # 최대 길이 제한
                    if len(content) > max_length:
                        content = content[:max_length]

                    documents.append(
                        Document(
                            page_content=content,
                            metadata={
                                "source": f"wikipedia-{language}",
                                "section": "content",
                                "topic": page_title,
                            },
                        )
                    )
            except (
                wikipedia.exceptions.DisambiguationError,
                wikipedia.exceptions.PageError,
            ) as e:
                # 단일 페이지를 가져오는 데 실패해도 계속 진행
                st.warning(f"{page_title} 페이지 처리 중 오류 발생: {str(e)}")
                continue

        if documents:
            st.success(f"{language} 언어로 {len(documents)}개의 문서를 찾았습니다.")

        return documents

    except Exception as e:
        st.error(f"위키피디아 검색 중 오류 발생: {str(e)}")
        return []


# Vector Store 생성 함수
# TODO: cache
@st.cache_resource
def create_vector_store(topic):
    documents = []

    # 영어 위키피디아 데이터 수집
    wiki_docs_en = get_wikipedia_content(topic, "en")
    # TODO: wiki_docs_en가 None이 아니면 documents에 추가
    if wiki_docs_en:
        documents.extend(wiki_docs_en)
    

    # 한국어 위키피디아 데이터 수집
    wiki_docs_ko = get_wikipedia_content(topic, "ko")
    # TODO: wiki_docs_en가 None이 아니면 documents에 추가
    if wiki_docs_ko:
        documents.extend(wiki_docs_ko)

    # 토픽이 영어가 아닌 경우 영어로도 검색 시도
    if not any(c.isascii() for c in topic):
        # 한국어 주제를 영어로 변환하려는 시도
        try:
            english_topic = f"{topic} in English"
            additional_docs = get_wikipedia_content(english_topic, "en")
            if additional_docs:
                documents.extend(additional_docs)
        except:
            pass

    # Vector DB 생성
    if documents:
        try:
            vector_store = FAISS.from_documents(documents, embeddings)
            return vector_store
        except Exception as e:
            st.error(f"Vector DB 생성 중 오류 발생: {str(e)}")
            return None
    else:
        return None


# 관련 정보 검색
def retrieve_relevant_info(query, vector_store, k=3):
    if not vector_store:
        return "", []

    try:
        # TODO: 질의에 관련된 문서 검색
        retrieved_docs = vector_store.similarity_search(query, k=k)

        # 검색 결과 텍스트 형식으로 변환
        context = ""
        for i, doc in enumerate(retrieved_docs):
            source = doc.metadata.get("source", "Unknown")
            section = doc.metadata.get("section", "")
            context += f"[문서 {i+1}] 출처: {source}"
            if section:
                context += f", 섹션: {section}"
            context += f"\n{doc.page_content}\n\n"

        return context, retrieved_docs
    except:
        return "", []


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


# LangGraph 상태 정의 - RAG 관련 필드 추가
class DebateState(TypedDict):
    topic: str
    messages: List[Dict]
    current_round: int
    max_rounds: int
    current_speaker: SpeakerRole
    debate_status: DebateStatus
    vector_store: object  # RAG 벡터 스토어
    retrieved_docs: Dict[str, List]  # RAG 검색 결과


# 찬성 측 에이전트 노드 - RAG 활용
def pro_agent(state: DebateState) -> DebateState:
    """찬성 측 에이전트 함수"""
    # 시스템 프롬프트 설정
    system_prompt = "당신은 논리적이고 설득력 있는 찬성 측 토론자입니다."

    # 메시지 준비
    messages = [SystemMessage(content=system_prompt)]

    # 기존 대화 기록 추가
    for msg in state["messages"]:
        if msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
        else:
            messages.append(HumanMessage(content=f"{msg['role']}: {msg['content']}"))

    # RAG: 관련 정보 검색
    query = f"{state['topic']} 찬성 장점 이유 근거"
    if state["current_round"] > 1 and state["messages"]:
        # 이전 반대 측 주장이 있으면 참고
        prev_con_arguments = [m for m in state["messages"] if m["role"] == "반대 측"]
        if prev_con_arguments:
            last_con = prev_con_arguments[-1]["content"]
            query += f" {last_con}에 대한 반박"

    context, docs = "", []
    # if "vector_store" in state and state["vector_store"]:
    vector_store = state.get("vector_store")
    if vector_store:
        context, docs = retrieve_relevant_info(query, state["vector_store"])

    # 프롬프트 구성
    if state["current_round"] == 1:
        prompt = f"""
        당신은 '{state['topic']}'에 대해 찬성 입장을 가진 토론자입니다.
        
        다음은 이 주제와 관련된 정보입니다:
        {context}
        
        논리적이고 설득력 있는 찬성 측 주장을 제시해주세요.
        가능한 경우 제공된 정보에서 구체적인 근거를 인용하세요.
        2 ~ 3문단, 각 문단은 100자내로 작성해주세요.
        """
    else:
        # 이전 발언자의 마지막 메시지를 가져옴
        previous_messages = [m for m in state["messages"] if m["role"] == "반대 측"]
        if previous_messages:
            last_con_message = previous_messages[-1]["content"]
            prompt = f"""
            당신은 '{state['topic']}'에 대해 찬성 입장을 가진 토론자입니다.
            
            다음은 이 주제와 관련된 정보입니다:
            {context}
            
            반대 측의 다음 주장에 대해 반박하고, 찬성 입장을 더 강화해주세요:

            반대 측 주장: "{last_con_message}"

            가능한 경우 제공된 정보에서 구체적인 근거를 인용하세요.
            2 ~ 3문단, 각 문단은 100자내로 작성해주세요.
            """

    # LLM에 요청
    messages.append(HumanMessage(content=prompt))
    response = llm.invoke(messages)

    # 상태 업데이트
    new_state = state.copy()
    new_state["messages"].append({"role": "찬성 측", "content": response.content})
    new_state["current_speaker"] = SpeakerRole.CON

    # 검색된 문서 저장
    if "retrieved_docs" not in new_state:
        new_state["retrieved_docs"] = {"pro": [], "con": []}
    else:
        new_state["retrieved_docs"] = {
            "pro": state["retrieved_docs"].get("pro", [])
            + ([doc.page_content for doc in docs] if docs else []),
            "con": state["retrieved_docs"].get("con", []),
        }

    return new_state


# 반대 측 에이전트 노드 - RAG 활용
def con_agent(state: DebateState) -> DebateState:
    """반대 측 에이전트 함수"""
    # 시스템 프롬프트 설정
    system_prompt = "당신은 논리적이고 설득력 있는 반대 측 토론자입니다. 찬성 측 주장에 대해 적극적으로 반박하세요."

    # 메시지 준비
    messages = [SystemMessage(content=system_prompt)]

    # 기존 대화 기록 추가
    for msg in state["messages"]:
        if msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
        else:
            messages.append(HumanMessage(content=f"{msg['role']}: {msg['content']}"))

    # RAG: 관련 정보 검색
    query = f"{state['topic']} 반대 단점 문제점 근거"
    if state["messages"]:
        # 가장 최근 찬성 측 주장 참고
        pro_arguments = [e for e in state["messages"] if e["role"] == "찬성 측"]
        if pro_arguments:
            last_pro = pro_arguments[-1]["content"]
            query += f" {last_pro}에 대한 반박"

    context, docs = "", []
    if "vector_store" in state and state["vector_store"]:
        context, docs = retrieve_relevant_info(query, state["vector_store"])

    # 프롬프트 구성
    # 찬성 측 마지막 메시지를 가져옴
    previous_messages = [m for m in state["messages"] if m["role"] == "찬성 측"]
    last_pro_message = previous_messages[-1]["content"]
    prompt = f"""
    당신은 '{state['topic']}'에 대해 반대 입장을 가진 토론자입니다.
    
    다음은 이 주제와 관련된 정보입니다:
    {context}
    
    찬성 측의 다음 주장에 대해 반박하고, 반대 입장을 제시해주세요:

    찬성 측 주장: "{last_pro_message}"

    가능한 경우 제공된 정보에서 구체적인 근거를 인용하세요.
    2 ~ 3문단, 각 문단은 100자내로 작성해주세요.
    """

    # LLM에 요청
    messages.append(HumanMessage(content=prompt))
    response = llm.invoke(messages)

    # 상태 업데이트
    new_state = state.copy()
    new_state["messages"].append({"role": "반대 측", "content": response.content})
    new_state["current_round"] += 1

    # 검색된 문서 저장
    if "retrieved_docs" not in new_state:
        new_state["retrieved_docs"] = {"pro": [], "con": []}
    else:
        new_state["retrieved_docs"] = {
            "pro": state["retrieved_docs"].get("pro", []),
            "con": state["retrieved_docs"].get("con", [])
            + ([doc.page_content for doc in docs] if docs else []),
        }

    # 다음 라운드 여부 결정
    if new_state["current_round"] <= new_state["max_rounds"]:
        new_state["current_speaker"] = SpeakerRole.PRO
    else:
        new_state["current_speaker"] = SpeakerRole.JUDGE

    return new_state


# 심판 에이전트 노드 - RAG 활용
def judge_agent(state: DebateState) -> DebateState:
    """심판 에이전트 함수"""
    # 시스템 프롬프트 설정
    system_prompt = "당신은 공정하고 논리적인 토론 심판입니다. 양측의 주장을 면밀히 검토하고 객관적으로 평가해주세요."

    # 메시지 준비
    messages = [SystemMessage(content=system_prompt)]

    # RAG: 객관적인 평가를 위한 정보 검색
    query = f"{state['topic']} 평가 기준 객관적 사실"
    context, _ = "", []
    if "vector_store" in state and state["vector_store"]:
        context, _ = retrieve_relevant_info(query, state["vector_store"], k=2)

    # 프롬프트 구성
    prompt = f"""
    다음은 '{state['topic']}'에 대한 찬반 토론입니다. 각 측의 주장을 분석하고 평가해주세요.
    
    다음은 이 주제와 관련된 객관적인 정보입니다:
    {context}

    토론 내용:
    """
    for msg in state["messages"]:
        prompt += f"\n\n{msg['role']}: {msg['content']}"

    prompt += """
    
    위 토론을 분석하여 다음을 포함하는 심사 평가를 해주세요:
    1. 양측 주장의 핵심 요약
    2. 각 측이 사용한 주요 논리와 증거의 강점과 약점
    3. 전체 토론의 승자와 그 이유
    4. 양측 모두에게 개선점 제안
    
    가능한 경우 제공된 객관적 정보를 참고하여 평가해주세요.
    """

    # LLM에 요청
    messages.append(HumanMessage(content=prompt))
    response = llm.invoke(messages)

    # 상태 업데이트
    new_state = state.copy()
    new_state["messages"].append({"role": "심판", "content": response.content})
    new_state["debate_status"] = DebateStatus.COMPLETED
    new_state["current_speaker"] = SpeakerRole.COMPLETED
    return new_state


# TODO: 라우터 함수 작성
def router(state: DebateState) -> Literal["pro_agent", "con_agent", "심판", "END"]:
    if state["debate_status"] == DebateStatus.COMPLETED:
        return "END"
    return state["current_speaker"].value

# LangGraph 워크플로우 정의
def create_debate_graph() -> StateGraph:
    # 그래프 생성
    workflow = StateGraph(DebateState)

    # 노드 추가
    workflow.add_node(SpeakerRole.PRO.value, pro_agent)
    workflow.add_node(SpeakerRole.CON.value, con_agent)
    workflow.add_node(SpeakerRole.JUDGE.value, judge_agent)

    routing_map = {
        SpeakerRole.PRO.value: SpeakerRole.PRO.value,
        SpeakerRole.CON.value: SpeakerRole.CON.value,
        SpeakerRole.JUDGE.value: SpeakerRole.JUDGE.value,
        "END": END,
    }

    # 엣지 정의
    workflow.add_conditional_edges(
        SpeakerRole.PRO.value,
        router,
        routing_map,
    )

    workflow.add_conditional_edges(
        SpeakerRole.CON.value,
        router,
        routing_map,
    )

    workflow.add_conditional_edges(
        SpeakerRole.JUDGE.value,
        router,
        routing_map,
    )

    # 시작 노드 설정
    workflow.set_entry_point(SpeakerRole.PRO.value)

    # 그래프 컴파일
    return workflow.compile()


# Streamlit 애플리케이션 로직
# 세션 스테이트 초기화
if "debate_active" not in st.session_state:
    st.session_state.debate_active = False
    st.session_state.debate_messages = []
    st.session_state.vector_store = None
    st.session_state.retrieved_docs = {"pro": [], "con": []}

# 사이드바: 설정
with st.sidebar:
    st.header("토론 설정")

    # 토론 주제 입력
    debate_topic = st.text_input(
        "토론 주제를 입력하세요:", "인공지능이 인간의 일자리를 대체해야 한다"
    )
    max_rounds = st.slider("토론 라운드 수", min_value=1, max_value=5, value=1)

    # RAG 기능 활성화 옵션
    enable_rag = st.checkbox(
        "RAG 활성화", value=True, help="외부 지식을 검색하여 토론에 활용합니다."
    )
    show_sources = st.checkbox(
        "출처 표시", value=True, help="검색된 정보의 출처를 표시합니다."
    )

# 토론 시작 버튼
if not st.session_state.debate_active and st.button("토론 시작"):
    st.session_state.vector_store = None
    st.session_state.retrieved_docs = {"pro": [], "con": []}

    # Vector Store 생성 (RAG 활성화된 경우)
    if enable_rag:
        with st.spinner("외부 지식을 수집하고 분석 중입니다..."):
            vector_store = create_vector_store(debate_topic)
            st.session_state.vector_store = vector_store

            if vector_store:
                st.success("외부 지식 검색 준비 완료!")
            else:
                st.warning(
                    "외부 지식 검색을 위한 준비에 실패했습니다. 기본 토론으로 진행합니다."
                )

    # 그래프 생성
    debate_graph = create_debate_graph()

    # 초기 상태 설정
    initial_state: DebateState = {
        "topic": debate_topic,
        "messages": [],
        "current_round": 1,
        "max_rounds": max_rounds,
        "current_speaker": SpeakerRole.PRO,
        "debate_status": DebateStatus.ACTIVE,
        "vector_store": st.session_state.vector_store,
        "retrieved_docs": {"pro": [], "con": []},
    }

    # 토론 시작
    with st.spinner("토론이 진행 중입니다... 완료까지 잠시 기다려주세요."):
        # 그래프 실행 - stream 대신 invoke 사용
        result = debate_graph.invoke(initial_state)

        # 결과를 세션 스테이트에 저장
        st.session_state.debate_messages = result["messages"]
        st.session_state.debate_active = True
        st.session_state.retrieved_docs = result.get(
            "retrieved_docs", {"pro": [], "con": []}
        )

    # 페이지 새로고침하여 결과 표시
    st.rerun()

# 토론 내용 표시
if st.session_state.debate_active:
    # 토론 주제 표시
    st.header(f"토론 주제: {debate_topic}")

    # 토론 내용 표시
    st.header("토론 진행 상황")

    messages = st.session_state.debate_messages
    total_rounds = len([m for m in messages if m["role"] == "찬성 측"])

    # 라운드별로 그룹화하여 표시
    for round_num in range(1, total_rounds + 1):
        st.subheader(f"라운드 {round_num}")

        # 이 라운드의 찬성측 메시지 찾기 (인덱스는 (라운드-1)*2)
        pro_index = (round_num - 1) * 2
        if pro_index < len(messages) and messages[pro_index]["role"] == "찬성 측":
            with st.container(border=True):
                st.markdown("**🔵 찬성 측:**")
                st.write(messages[pro_index]["content"])

        # 이 라운드의 반대측 메시지 찾기 (인덱스는 (라운드-1)*2 + 1)
        con_index = (round_num - 1) * 2 + 1
        if con_index < len(messages) and messages[con_index]["role"] == "반대 측":
            with st.container(border=True):
                st.markdown("**🔴 반대 측:**")
                st.write(messages[con_index]["content"])

        st.divider()

    # 심판 평가 표시 (마지막 메시지)
    if messages and messages[-1]["role"] == "심판":
        with st.container(border=True):
            st.subheader("🧑‍⚖️ 최종 평가")
            st.write(messages[-1]["content"])

    # 검색된 출처 정보 표시 (옵션이 활성화된 경우)
    if (
        show_sources
        and st.session_state.retrieved_docs
        and (
            st.session_state.retrieved_docs.get("pro")
            or st.session_state.retrieved_docs.get("con")
        )
    ):
        with st.expander("사용된 참고 자료 보기"):
            st.subheader("찬성 측 참고 자료")
            for i, doc in enumerate(
                st.session_state.retrieved_docs.get("pro", [])[:3]
            ):  # 최대 3개만 표시
                st.markdown(f"**출처 {i+1}**")
                st.text(doc[:300] + "..." if len(doc) > 300 else doc)
                st.divider()

            st.subheader("반대 측 참고 자료")
            for i, doc in enumerate(
                st.session_state.retrieved_docs.get("con", [])[:3]
            ):  # 최대 3개만 표시
                st.markdown(f"**출처 {i+1}**")
                st.text(doc[:300] + "..." if len(doc) > 300 else doc)
                st.divider()

    # 다시 시작 버튼
    if st.button("새 토론 시작"):
        st.session_state.debate_active = False
        st.session_state.debate_messages = []
        st.session_state.vector_store = None
        st.session_state.retrieved_docs = {"pro": [], "con": []}
        st.rerun()
