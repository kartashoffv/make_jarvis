from .processing import text_splitter_by_tokens, file_to_text, hash_string


class ChunkData:
    chunked_texts: list[str]
    chunk_metadatas: list[dict]
    chunk_ids: list[str]

    def __init__(self, filepaths: str | list[str]) -> None:
        if not isinstance(filepaths, list):
            filepaths = [filepaths]
        self.chunked_texts = []
        self.chunk_metadatas = []
        self.chunk_ids = []

        for path in filepaths:
            chunks = text_splitter_by_tokens(file_to_text(path))
            for pos, chunk in enumerate(chunks):
                self.chunked_texts.append(chunk)
                self.chunk_metadatas.append({"source": path})
                self.chunk_ids.append(hash_string(path, pos))
