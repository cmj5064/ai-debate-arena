# app_multi_agent.py - 챕터 2: Streamlit을 활용한 멀티 에이전트 구현
import streamlit as st
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
import os
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드
load_dotenv()

# LangChain Azure OpenAI 설정
llm = AzureChatOpenAI(
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15"),
    temperature=0.7,
)

# todo: 페이지 설정
# st. ...

# 제목 및 소개
st.title("🤖 AI 토론 - 멀티 에이전트")
st.markdown(
    """
### 프로젝트 소개
이 애플리케이션은 3개의 AI 에이전트(찬성, 반대, 심판)가 사용자가 제시한 주제에 대해 토론을 진행합니다.
각 AI는 서로의 의견을 듣고 반박하며, 마지막에는 심판 AI가 토론 결과를 평가합니다.
"""
)

# 세션 스테이트 초기화

# step : start, pro_round_{n}, con_round_{n}, judge, completed
if "debate_started" not in st.session_state:
    st.session_state.debate_started = False
    st.session_state.round = 0
    st.session_state.max_rounds = 3
    st.session_state.debate_history = []
    st.session_state.judge_verdict = None
    st.session_state.current_step = "start"  # 현재 단계 추적


# AI 응답 생성 함수 (메시지 히스토리 포함)
def generate_response(prompt, system_prompt, message_history=None):
    messages = [SystemMessage(content=system_prompt)]

    # 메시지 히스토리가 있으면 추가
    if message_history:
        for message in message_history:
            if message["role"] == "assistant":
                messages.append(AIMessage(content=message["content"]))
            else:
                messages.append(
                    HumanMessage(content=f"{message['role']}: {message['content']}")
                )

    # 현재 프롬프트 추가
    messages.append(HumanMessage(content=prompt))

    # todo: LLM 호출
    # response = ...
    return response.content


# 토론 주제 입력 섹션
st.header("토론 주제 입력")

# todo: 토론 주제 입력
# debate_topic = ...

max_rounds = st.slider("토론 라운드 수", min_value=1, max_value=5, value=1)
st.session_state.max_rounds = max_rounds

# 토론 시작 버튼
if not st.session_state.debate_started and st.button("토론 시작"):
    # 세션 스테이트 초기화
    st.session_state.debate_started = True
    st.session_state.round = 1
    st.session_state.debate_history = []
    st.session_state.judge_verdict = None
    st.session_state.current_step = "pro_round_1"
    # todo: 페이지 리로드하여 다음 단계로 진행
    # st. ...


# 토론 단계별 기능을 함수로 분리
def handle_pro_round():
    with st.spinner("찬성 측 의견을 생성 중입니다..."):

        if st.session_state.round == 1:
            # 첫 번째 찬성 측 의견 생성
            pro_prompt = f"""
            당신은 '{debate_topic}'에 대해 찬성 입장을 가진 토론자입니다.
            논리적이고 설득력 있는 찬성 측 주장을 제시해주세요.
            200자 내로 작성해주세요.
            """
            system_prompt = "당신은 논리적이고 설득력 있는 찬성 측 토론자입니다."
        else:
            # 이전 반대 측 의견에 대한 반박
            previous_argument = st.session_state.debate_history[-1]["content"]
            pro_prompt = f"""
            당신은 '{debate_topic}'에 대해 찬성 입장을 가진 토론자입니다.
            반대 측의 다음 주장에 대해 반박하고, 찬성 입장을 더 강화해주세요:

            반대 측 주장: "{previous_argument}"

            200자 내로 작성해주세요.
            """
            system_prompt = "당신은 논리적이고 설득력 있는 찬성 측 토론자입니다. 반대 측 주장에 대해 적극적으로 반박하세요."

        pro_argument = generate_response(
            pro_prompt, system_prompt, st.session_state.debate_history
        )

        st.session_state.debate_history.append(
            {"role": "찬성 측", "content": pro_argument}
        )
        st.session_state.current_step = f"con_round_{st.session_state.round}"


def handle_con_round():
    # todo: 반대 측 의견 생성
    # ...
    # handle_pro_round 참조
    # ...


def handle_judge():
    with st.spinner("심판이 토론을 평가 중입니다..."):
        judge_prompt = f"""
        다음은 '{debate_topic}'에 대한 찬반 토론입니다. 각 측의 주장을 분석하고 평가해주세요.

        토론 내용:
        """
        for entry in st.session_state.debate_history:
            judge_prompt += f"\n\n{entry['role']}: {entry['content']}"

        judge_prompt += """
        
        위 토론을 분석하여 다음을 포함하는 심사 평가를 해주세요:
        1. 양측 주장의 핵심 요약
        2. 각 측이 사용한 주요 논리와 증거의 강점과 약점
        3. 전체 토론의 승자와 그 이유
        4. 양측 모두에게 개선점 제안
        """
        system_prompt = "당신은 공정하고 논리적인 토론 심판입니다. 양측의 주장을 면밀히 검토하고 객관적으로 평가해주세요."

        judge_verdict = generate_response(judge_prompt, system_prompt, [])

        st.session_state.judge_verdict = judge_verdict
        st.session_state.current_step = "completed"


def show_progress():
    # 토론 진행 상태 표시
    if (
        st.session_state.debate_started
        and st.session_state.round <= st.session_state.max_rounds
    ):
        total_steps = (
            st.session_state.max_rounds * 2 + 1
        )  # 각 라운드 마다 찬성/반대 + 심판
        current_steps = (st.session_state.round - 1) * 2 + (
            1 if st.session_state.current_step.startswith("con") else 0
        )
        if (
            st.session_state.current_step == "judge"
            or st.session_state.current_step == "completed"
        ):
            current_steps = total_steps - 1

        progress = current_steps / total_steps
        st.progress(progress)


# 토론 진행
if st.session_state.debate_started:
    # 토론 주제 표시
    st.header(f"토론 주제: {debate_topic}")

    # 현재 라운드 정보 - 심판 단계에서는 라운드 표시 방식을 다르게 처리
    if (
        st.session_state.current_step == "judge"
        or st.session_state.current_step == "completed"
    ):
        st.subheader(f"최종 평가 단계")
    else:
        st.subheader(f"라운드 {st.session_state.round} / {st.session_state.max_rounds}")

    show_progress()

    # 진행 단계별 처리
    if st.session_state.current_step.startswith("pro_round_"):
        handle_pro_round()
        st.rerun()  # 페이지 리로드하여 다음 단계로 진행

    elif st.session_state.current_step.startswith("con_round_"):
        handle_con_round()
        st.rerun()  # 페이지 리로드하여 다음 단계로 진행

    elif (
        st.session_state.current_step == "judge"
        and st.session_state.judge_verdict is None
    ):
        handle_judge()
        st.rerun()  # 페이지 리로드하여 결과 표시

    # 토론 내용 표시
    st.header("토론 진행 상황")
    for i, entry in enumerate(st.session_state.debate_history):
        round_num = (i // 2) + 1

        st.subheader(f"라운드 {round_num} - {entry['role']}")
        st.write(entry["content"])
        st.divider()

    # 심판 판정 표시
    if st.session_state.judge_verdict:
        st.header("🧑‍⚖️ 심판 평가")
        st.write(st.session_state.judge_verdict)

    # 다시 시작 버튼
    if st.session_state.current_step == "completed":
        if st.button("새 토론 시작"):
            st.session_state.debate_started = False
            st.session_state.round = 0
            st.session_state.debate_history = []
            st.session_state.judge_verdict = None
            st.session_state.current_step = "start"
            st.rerun()
