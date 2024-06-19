import os
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, HumanMessage

from .class_library.processing import get_embedder_class
from .class_library.chroma import Chroma

from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
BASE_URL_OPENAI_API = os.getenv('BASE_URL_OPENAI_API')
BASE_PATH = os.getenv('BASE_PATH')

CHAT_ID = 'chat_id_1'
embedding = get_embedder_class({"group": "openai"})
client = Chroma(embedding, f'{BASE_PATH}/make_jarvis/data/vectorsDB/chroma_{CHAT_ID}', CHAT_ID)

prompt = hub.pull("hwchase17/openai-functions-agent")
prompt.messages[0].prompt.template = 'Вы полезный ассистент, который помогает анализировать статьи и разговаривает с пользователем. Ваши ответы должны быть не более двух предложений.'

@tool
def get_relevant_chunks_tools(query):
    """Поиск ответов на запросы пользователя на основе подгруженных документов
    Полезная функция, если пользователь спрашивает что-то по документам.
    Args:
        query: запрос пользователя
    """
    return str(client.get_relevant_docs(query)['documents'][0])

tools = [get_relevant_chunks_tools]

llm = ChatOpenAI(model="gpt-3.5-turbo-0125", api_key=OPENAI_API_KEY, base_url=BASE_URL_OPENAI_API)
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


def run_agent(query : str) -> str:
    return agent_executor.invoke({
        "input": query,
        })
