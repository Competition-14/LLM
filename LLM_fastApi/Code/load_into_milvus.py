from dotenv import load_dotenv

# load_dotenv() loads the .env file and makes the variables available to the program
load_dotenv()

import os
from typing import List
from langchain_core.documents import Document
# from rag.embeding_model import BGE_M3
from MilvusCode import MilvusDB
from loader import DirLoader
from splitter import TableSplitter, TextSplitter
from langchain_openai import OpenAIEmbeddings

# const var
# DATA_DIR: str = os.getenv("DATA_DIR")
DATA_DIR: str = r"C:\Users\ASUS\Desktop\DATA_DIR"

loader: DirLoader = DirLoader(DATA_DIR)
docs: list[Document] = loader.load()

# table_splitter:TableSplitter = TableSplitter(max_length=2000)
# table_documents:List[Document] = table_splitter.create_documents([doc.page_content for doc in docs],
#                                         metadatas=[doc.metadata for doc in docs])
table_documents: List[Document] = []
for doc in docs:
    table_splitter: TableSplitter = TableSplitter(max_length=2000)
    tmp_table_documents: List[Document] = table_splitter.create_documents([doc.page_content],
                                                                          metadatas=[doc.metadata])
    table_documents.extend(tmp_table_documents)

print("length of tabl chunks: ", len(table_documents))
# print("table metadata:",tabl_documents[0].metadata)

# text_splitter:TextSplitter = TextSplitter(
#     # Set a really small chunk size, just to show.
#     chunk_size=500,
#     chunk_overlap=200,
#     length_function=len,
#     is_separator_regex=False,
# )
# text_documents:List[Document] = text_splitter.create_documents([doc.page_content for doc in docs],
#                                            metadatas=[doc.metadata for doc in docs])
text_documents: List[Document] = []
for doc in docs:
    # text splitter
    text_splitter: TextSplitter = TextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=600,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    tmp_text_documents: List[Document] = text_splitter.create_documents([doc.page_content],
                                                                        metadatas=[doc.metadata])
    text_documents.extend(tmp_text_documents)
print("length of text chunks: ", len(text_documents))
# print("text metadata:",text_documents[0].metadata)

# create an instance of the MilvusDB class with the embedding model
openai_embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002",  # 指定模型名
    base_url="https://api.gptsapi.net/v1"  # 指定API URL，如果是默认的OpenAI URL，可以省略
)
embedding_model: openai_embeddings
vector_db = MilvusDB(em=embedding_model)

# load documents into the vector database
print("loading text documents into vector database...")
vector_db.add_documents(text_documents, show_process=True)

print("loading table documents into vector database...")
vector_db.add_documents(table_documents, show_process=True)