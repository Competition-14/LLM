import os
from typing import List, Any

from tqdm import tqdm
from langchain_core.documents import Document
from overrides import override
from langchain_openai import OpenAIEmbeddings
from langchain_milvus import Milvus

# The easiest way is to use Milvus Lite where everything is stored in a local file.
# If you have a Milvus server you can use the server URI such as "http://localhost:19530".
URI: str = "./milvus_database.db"


class MilvusDB(Milvus):
    def __init__(self, em: OpenAIEmbeddings) -> None:
        super().__init__(embedding_function=em,
                         connection_args={"uri": URI},
                         auto_id=True)

    @override
    def add_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        """Run more documents through the embeddings and add to the vectorstore.

        Args:
            documents: Documents to add to the vectorstore.

        Returns:
            List of IDs of the added texts.
        """
        # TODO: Handle the case where the user doesn't provide ids on the Collection
        texts: str = [doc.page_content for doc in documents]
        metadatas: dict = [doc.metadata for doc in documents]
        # show process
        if "show_process" in kwargs and kwargs["show_process"]:
            # create a var to store result
            results: List[str] = []
            for text, metadata in tqdm(zip(texts, metadatas), total=len(texts), desc="Adding documents"):
                results.extend(self.add_texts([text], [metadata], **kwargs))
        else:
            return self.add_texts(texts, metadatas, **kwargs)

    @override
    async def aadd_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        """Run more documents through the embeddings and add to the vectorstore.

        Args:
            documents: Documents to add to the vectorstore.

        Returns:
            List of IDs of the added texts.
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        if "show_process" in kwargs and kwargs["show_process"]:
            # create a var to store result
            results: List[str] = []
            for text, metadata in tqdm(zip(texts, metadatas), total=len(texts), desc="Adding documents"):
                results.extend(await self.aadd_texts([text], [metadata], **kwargs))
        else:
            return await self.aadd_texts(texts, metadatas, **kwargs)