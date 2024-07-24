import os
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv


# 주어진 모델 이름으로 ChatGPT APU모델 불러오는 함수
def load_model(model_name: str) -> ChatOpenAI:
    """
    주어진 모델 이름을 기반으로 ChatOpenAI 모델을 로드합니다.

    Args:
        model_name (str): 사용할 모델의 이름.

    Returns:
        ChatOpenAI: 로드된 ChatOpenAI 모델.
    """

    # env로 key불러오기
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(api_key=OPENAI_API_KEY, model_name=model_name)
    return llm


# 사전에 저장해둔 prompt file을 읽어와서 str으로 반환하는 함수
def load_prompt(character_name: str) -> str:
    """캐릭터의 이름을 입력받아서 그에 해당하는 프롬프트를 문자열로 반환합니다.

    Args:
        character_name (str): 불러올 캐릭터의 이름을 입력받습니다.

    Returns:
        prompt(str): 불러온 프롬프트 내용을 반환합니다
    """
    with open(f"prompts/{character_name}.prompt", "r", encoding="utf-8") as file:
        prompt = file.read().strip()
    return prompt


# 대화를 메모리하는 함수
def set_memory() -> ConversationBufferMemory:
    """
    대화 히스토리를 저장하기 위한 메모리를 설정합니다.

    Returns:
        ConversationBufferMemory: 초기화된 대화 메모리.
    """
    return ConversationBufferMemory(memory_key="chat_history", return_messages=True)


# Langchain의 chain을 만들어주는 함수
def initialize_chain(
    llm: ChatOpenAI, character_name: str, memory: ConversationBufferMemory
) -> LLMChain:
    """
    주어진 LLM과 캐릭터 이름, 메모리를 기반으로 체인을 초기화합니다.

    Args:
        llm (ChatOpenAI): 사용할 언어 모델.
        character_name (str): 캐릭터의 이름.
        memory (ConversationBufferMemory): 대화 메모리.

    Returns:
        LLMChain: 초기화된 LLM 체인.
    """
    system_prompt = load_prompt(character_name)
    custom_prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{input}"),
        ]
    )
    chain = LLMChain(llm=llm, prompt=custom_prompt, verbose=True, memory=memory)
    return chain


# LLM의 답변을 생성하는 함수 (invoke)
def generate_message(chain: LLMChain, user_input: str) -> str:
    """
    사용자 입력을 기반으로 메시지를 생성합니다.

    Args:
        chain (LLMChain): 사용할 체인.
        user_input (str): 사용자의 입력.

    Returns:
        str: 생성된 응답 메시지.
    """
    result = chain({"input": user_input})
    response_content = result["text"]
    return response_content
