# app.py - 챕터 1: 프로젝트 소개 & Streamlit으로 기본 UI 개발
import streamlit as st
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import os
from dotenv import load_dotenv


# .env 파일에서 환경 변수 로드
load_dotenv()

# LangChain Azure OpenAI 설정
llm = AzureChatOpenAI(
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    temperature=0.7,
)


# LLM 응답 생성 함수
def generate_response(prompt, system_prompt):
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=prompt)]
    response = llm.invoke(messages)
    return response.content


# 페이지 설정
st.set_page_config(page_title="AI 토론", page_icon="🤖", layout="wide")

# 제목 및 소개
st.title("🤖 AI 토론")

st.markdown(
    """
    - 이 애플리케이션은 사용자가 제시한 주제에 대해 찬성과 반대 입장을 취하는
    AI 에이전트 간의 토론을 진행합니다.
    """
)

# 토론 주제 입력
st.header("토론 주제 입력")

debate_topic = st.text_input(
    "토론 주제를 입력하세요:", "인공지능이 인간의 일자리를 대체해야 한다"
)

# 토론 시작 버튼
if st.button("토론 시작"):

    # 세션 스테이트 초기화
    if "debate_started" not in st.session_state:
        st.session_state.debate_started = True
        st.session_state.debate_history = []

    # 토론 주제 표시
    st.header(f"토론 주제: {debate_topic}")

    # 1라운드 찬성 측 의견 생성
    with st.spinner("찬성 측 의견을 생성 중입니다..."):
        pro_prompt = f"""
            당신은 '{debate_topic}'에 대해 찬성 입장을 가진 토론자입니다.
            논리적이고 설득력 있는 찬성 측 주장을 제시해주세요.
            1-2 문단 정도로 간결하게 작성해주세요.
            """

        pro_argument = generate_response(
            pro_prompt, "당신은 논리적이고 설득력 있는 토론자입니다."
        )

        st.session_state.debate_history.append(
            {"role": "찬성 측", "content": pro_argument}
        )

    # 1라운드 반대 측 의견 생성
    with st.spinner("반대 측 의견을 생성 중입니다..."):
        con_prompt = f"""
            당신은 '{debate_topic}'에 대해 반대 입장을 가진 토론자입니다.
            논리적이고 설득력 있는 반대 측 주장을 제시해주세요.
            1-2 문단 정도로 간결하게 작성해주세요.
            """

        con_argument = generate_response(
            con_prompt, "당신은 논리적이고 설득력 있는 토론자입니다."
        )

        st.session_state.debate_history.append(
            {"role": "반대 측", "content": con_argument}
        )

    # 토론 결과 표시
    st.header("토론 결과")
    for entry in st.session_state.debate_history:
        st.subheader(entry["role"])
        st.write(entry["content"])
        st.divider()
