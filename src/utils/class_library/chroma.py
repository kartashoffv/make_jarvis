import os
import sys
import json
import chromadb
from chromadb.utils import embedding_functions
from .chunk_data import ChunkData
from .processing import get_embedder_class

import warnings
warnings.filterwarnings('ignore')

import onnxruntime as ort
ort.set_default_logger_severity(3)

class Chroma:
    def __init__(self, embedding, persist_dir, chat_id) -> None:
        self.embedding = embedding
        self.persist_dir = persist_dir
        self.chat_id = chat_id

    def init_db(self, files_dir: list[str]) -> None:
        """
        Initializes the Chroma DB using all the documents that
        the user uploaded the first time after registration.

        Supported file formats:
         * PDF
         * DOCX (DOC has not yet been processed)
         * MD
         * TXT

        Parameters
        ----------
        config : dict_like
            Dict with user data.

            chat_id : str
                chat_id
            persist_dir : str
                path to the folder where the vector database
                of a particular user will be stored
            files_dir : str
                path to the folder where all documents
                uploaded by the user are located

        Returns
        -------
        output : None
            Saves the database file to `persist_dir`.
        """

        chroma_client = chromadb.PersistentClient(path=self.persist_dir)

        collection = chroma_client.create_collection(
            name=f"{self.chat_id}",
            metadata={"hnsw:space": "cosine"},
            embedding_function=self.embedding
        )
        try:
            files = os.listdir(files_dir[0])
            filepaths = [os.path.join(files_dir[0], file) for file in files]

            chunk_data = ChunkData(filepaths)

            collection.add(
                documents=chunk_data.chunked_texts,
                metadatas=chunk_data.chunk_metadatas,
                ids=chunk_data.chunk_ids,
            )
            print('Non-empty folder were initialized')
        except Exception:
            print('Empty folder were initialized')
            pass

    def add_docs(self, files_dir: list[str]) -> None:
        """
        Add one or several documents into an existing user's Chroma database.

        Supported file formats:
         * PDF
         * DOCX (DOC has not yet been processed)
         * MD
         * TXT

        Parameters
        ----------
        config : dict_like
            Dict with user data.

            chat_id : str
                chat_id
            persist_dir : str
                path to the folder where the vector database
                of a particular user will be stored
            files_dir : list of str
                Paths to files to be added to vector storage.

        Returns
        -------
        output : None
            Update the database file in `persist_dir`.
        """
    
        chroma_client = chromadb.PersistentClient(path=self.persist_dir)
        collection = chroma_client.get_collection(
            name=f"{self.chat_id}", embedding_function=self.embedding
        )

        chunk_data = ChunkData(files_dir)

        collection.add(
            documents=chunk_data.chunked_texts,
            metadatas=chunk_data.chunk_metadatas,
            ids=chunk_data.chunk_ids,
        )
        print('File(s) were added ')

    def remove_docs(self, files_dir: list[str]) -> None:
        """
        Add one or several documents into an existing user's Chroma database.

        Supported file formats:
         * PDF
         * DOCX (DOC has not yet been processed)
         * MD
         * TXT

        Parameters
        ----------
        config : dict_like
            Dict with user data.

            chat_id : str
                chat_id
            persist_dir : str
                path to the folder where the vector database
                of a particular user will be stored
            files_dir : list of str
                paths to files to be removed from vector storage.

        Returns
        -------
        output : None
            Update the database file in `persist_dir`.
        """
        
        embedding = embedding_functions.DefaultEmbeddingFunction()

        chroma_client = chromadb.PersistentClient(
            path=self.persist_dir)
        collection = chroma_client.get_collection(
            name=f"{self.chat_id}", embedding_function=embedding)
        for path in files_dir:
            collection.delete(
                where={"source": path}
            )
        print('File(s) were removed')
            
    
    def get_relevant_docs(self, query:str) -> dict:
        """
        get_relevant_docs(config)

        Function that retrieve relevant chunks from vector db.

        Parameters
        ----------
        config : dict_like
            dictionary that contains info about persistent directory
            (where our vector db is), model configuration - info about LLM and
            embedding configuration, chat history and user's query.

        Returns
        -------
        output : dict_like
            Return the dictionary with information of relevant docs
        """
        chroma_client = chromadb.PersistentClient(path=self.persist_dir)

        collection = chroma_client.get_collection(
            name=f"{self.chat_id}", embedding_function=self.embedding
        )
        if collection.count() > 0:
            return collection.query(
                query_texts=query,
                include=["metadatas", "documents", "distances"],
                n_results=5,
            )
        else:
            return collection.query(
                query_texts=query,
                include=["metadatas", "documents", "distances"],
                n_results=1,
            )
