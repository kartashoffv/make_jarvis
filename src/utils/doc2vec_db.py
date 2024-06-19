import os
from class_library.processing import get_embedder_class
from class_library.chroma import Chroma

BASE_PATH = os.getenv('BASE_PATH')

embedding = get_embedder_class({"group": "openai"})
client = Chroma(embedding, f'{BASE_PATH}/make_jarvis/data/vectorsDB/chroma_chat_id_1', 'chat_id_1')
client.add_docs([f'{BASE_PATH}/make_jarvis/data/raw/Cross-encoder for RAG fa18ed4758ed442da9e27a1e8e1a53de.pdf'])
