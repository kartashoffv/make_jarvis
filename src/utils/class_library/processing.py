import os
import json
import hashlib

from langchain_text_splitters import TokenTextSplitter
from unstructured.partition.auto import partition
from chromadb.utils import embedding_functions
from chromadb.api.types import EmbeddingFunction

from langchain_openai import ChatOpenAI
from langchain_community.chat_models import GigaChat
from langchain_core.messages import HumanMessage, AIMessage

from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
BASE_URL_OPENAI_API = os.getenv('BASE_URL_OPENAI_API')

def file_to_text(filepath: str) -> str | None:
    """
    Converts a file to text.

    Supported file formats:
     * PDF
     * DOCX (DOC has not yet been processed)
     * MD
     * TXT

    Parameters
    ----------
    filepath : str
        Path to the file to be converted.

    Returns
    -------
    output : str
        A string that contains all the text from the document.
    """
    solid_texts = None

    extnt = os.path.splitext(filepath)[1].lower()

    if extnt in (".pdf", ".docx", ".md", ".xlsx", ".txt"):
        try:
            reader = partition(filepath)
            solid_texts = " ".join([line.text for line in reader])
        except Exception as e:
            print(e)

    return solid_texts


def text_splitter_by_tokens(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 0,
    encoding_name: str = "cl100k_base",
) -> list[str]:
    """
    Divides a string into chunks of a certain size according
    to the number of tokens in the text. Uses tokenizer to determine
    the number of tokens.

    Parameters
    ----------
    text : str
        Path to the file to be converted.
    chunk_size : int
        Chunk size in tokens should be used to split text.
    chunk_overlap : int
        How many tokens from the end of the previous chunk
        should go to the beginning of the next chunk.
    encoding_name : str
        Tokenizer, which is used to split text into tokens.
        Must be the same as the one used in the model used
        to create embeddings.

    Returns
    -------
    output : list
        List of strings, each of which contains ``n=chunk_size`` tokens.
    """

    text_splitter = TokenTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, encoding_name=encoding_name
    )

    return text_splitter.split_text(text)


def hash_string(stroke: str, num: int) -> str:
    """
    Creates a hash of the string connected to the number n via underscore.

    Parameters
    ----------
    stroke : str
        Path to the file to be converted.
    num : int
        Chunk size in tokens should be used to split text.

    Returns
    -------
    output : str
        A string in the form of: `hashedstring_num`.
    """

    hash_object = hashlib.sha256()
    hash_object.update(stroke.encode())
    hashed_output = hash_object.hexdigest()
    return f"{hashed_output}_{num}"


def get_llm_class(llm_info: dict) -> ChatOpenAI | GigaChat | None:
    """
    Function that return class of large language model.

    Parameters
    ----------
    config : dict_like
        Dictionaries are config['model_config']['llm']. They contains info
        about llm configuration.

    Returns
    -------
    output : class object
        Return the class for specific LLM (openai, gigachat etc.)
    """
    if llm_info["group"] == "openai":
        return ChatOpenAI(
            model=llm_info["name"],
            api_key=llm_info["api_key"],
            temperature=llm_info["temp"],
        )
    elif llm_info["group"] == "gigachat":
        return GigaChat(
            credentials=llm_info["api_key"],
            verify_ssl_certs=False,
            model=llm_info["name"],
            scope=llm_info["scope"],
            temperature=llm_info["temp"],
        )


def get_embedder_class(embd_info: dict) -> EmbeddingFunction:
    """
    Function that return class of embedder (function for getting embeddings).

    Parameters
    ----------
    config : dict_like
        Dictionaries are config['model_config']['embedder']. They contains info
        about embedder configuration.

    Returns
    -------
    output : class object
        Return the class for specific embedder (openai, open_source etc.)
    """
    
    if embd_info["group"] == "openai":
        return embedding_functions.OpenAIEmbeddingFunction(
            api_key=OPENAI_API_KEY,
            api_base=BASE_URL_OPENAI_API
        )
    elif embd_info["group"] == "open_source":
        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embd_info["name"]
        )
    else:
        return embedding_functions.DefaultEmbeddingFunction()


def get_history(chat_history: list[dict[str, str]]) -> list:
    """
    Function that convert user's chat history to usage in following code.

    Parameters
    ----------
    config : list[dict]
        list with dictionaries with format like {"text" : value, "from" : value}

    Returns
    -------
    output : list
        Return the list with list with classes HumanMessage and AIMessage
    """
    chat_history = []
    for u_msg, ai_msg in zip(chat_history):
        chat_history.extend([HumanMessage(content=u_msg), AIMessage(ai_msg)])
    return chat_history


def check_values(data, threshold):
    # TODO add documentation
    for sublist in data:
        for value in sublist:
            if value > threshold:
                return True
    return False


def return_const_msg(response: str, finish_reason: str) -> str:
    # TODO add documentation
    return json.dumps({
            "response": response,
            "sources": {
                "paths": [],
                "documents": []
                },
            "metadata": {
                "token_usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                    },
                "model_name": "SYSTEM",
                "finish_reason": finish_reason
                }
            }, ensure_ascii=False)
