import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI

# .env 파일에서 환경 변수 로드
load_dotenv()


def get_llm():
    return AzureChatOpenAI(
        openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15"),
        temperature=0.7,
    )
