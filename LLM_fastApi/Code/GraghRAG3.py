from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Tuple, List, Optional
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
import os
from langchain_community.graphs import Neo4jGraph
# from langchain.document_loaders import WikipediaLoader
from langchain_community.document_loaders import WikipediaLoader, TextLoader, AthenaLoader
from langchain.text_splitter import TokenTextSplitter
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from neo4j import GraphDatabase
from yfiles_jupyter_graphs import GraphWidget
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_core.runnables import ConfigurableField, RunnableParallel, RunnablePassthrough

# try:
#   import google.colab
#   from google.colab import output
#   output.enable_custom_widget_manager()
# except:
#   print("1")
#   pass

# 获取环境变量
# os.environ["OPENAI_API_KEY"] = "sk-"
# os.environ["NEO4J_URI"] = "bolt://18.212.197.205:7687"
# os.environ["NEO4J_USERNAME"] = "neo4j"
# os.environ["NEO4J_PASSWORD"] = "tendencies-meanings-enlistment"
openai_api_key = os.getenv("OPENAI_API_KEY")
# neo4j_uri = os.getenv("NEO4J_URI")
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = "neo4j"
"""
    这里的Neo4jGragh()源码中是这么定义的：
            url = get_from_dict_or_env({"url": url}, "url", "NEO4J_URI")
        username = get_from_dict_or_env(
            {"username": username}, "username", "NEO4J_USERNAME"
        )
        password = get_from_dict_or_env(
            {"password": password}, "password", "NEO4J_PASSWORD"
        )
        database = get_from_dict_or_env(
            {"database": database}, "database", "NEO4J_DATABASE", "neo4j"
        )
"""
# graph = Neo4jGraph()

# Read the wikipedia article（阅读维基百科的文章）——————————————但是我不一定保证能够连接上维基百科
# raw_documents = WikipediaLoader(query="Elizabeth I").load()
# raw_documents = AthenaLoader(query="Elizabeth I").load()
raw_documents = WikipediaLoader(query="Elizabeth I").load()
# 定义文件路径列表
# file_paths = [
#     r"O:\Python\Flask\Competition\LLM_fastApi\Code\extracted_text_all_pages.txt",
#     r"O:\Python\Flask\Competition\LLM_fastApi\Code\extracted_text_all_pages1.txt",
#     r"O:\Python\Flask\Competition\LLM_fastApi\Code\extracted_text_all_pages2.txt",
#     # 添加更多文件路径
# ]
# raw_documents = []
# raw_documents = TextLoader("").load()
# 分别加载每个文件
# for file_path in file_paths:
#     text_loader = TextLoader(file_path)
#     documents = text_loader.load()
#     # 将加载的文档添加到列表中
#     raw_documents.extend(documents)
# Define chunking strategy（定义分块策略，用来切割文档）
print(raw_documents)

text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=24)
# 这里进限于前三个文档
documents = text_splitter.split_documents(raw_documents[:3])

print(documents)

"""
    测试线---------------------------------------------------------------------------------
"""
# # 这里存在一个问题，LLMGraphTransformer()是langchain的一个接口，但是当时说的是只支持 OpenAI 和 Mistral 的函数调用模型
# llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0125")  # gpt-4-0125-preview occasionally has issues
# llm_transformer = LLMGraphTransformer(llm=llm)
#
# # 这个是在提取图数据，然后add到neo4j
# graph_documents = llm_transformer.convert_to_graph_documents(documents)
# graph.add_graph_documents(
#     graph_documents,
#     baseEntityLabel=True,
#     include_source=True
# )
#
# # directly show the graph resulting from the given Cypher query（）
# default_cypher = "MATCH (s)-[r:!MENTIONS]->(t) RETURN s,r,t LIMIT 50"
#
#
# # 种类的id？
# def showGraph(cypher: str = default_cypher):
#     # create a neo4j session to run queries
#     driver = GraphDatabase.driver(
#         uri=NEO4J_URI,
#         auth=(NEO4J_USERNAME,
#               NEO4J_PASSWORD))
#     session = driver.session()
#     widget = GraphWidget(graph=session.run(cypher).graph())
#     widget.node_label_mapping = 'id'
#     # display(widget)
#     return widget
#
#
# showGraph()
#
# # 非结构化数据检索器————这里的Neo4jVector.from_existing_graph()为文档添加关键字和向量检索功能
# vector_index = Neo4jVector.from_existing_graph(
#     OpenAIEmbeddings(),
#     search_type="hybrid",
#     node_label="Document",
#     text_node_properties=["text"],
#     embedding_node_property="embedding"
# )
#
# # Retriever
#
# graph.query(
#     "CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]")
#
# '''
#     这里有个问题，这边的 Entities 还有下面的 prompt 都是要根据具体的变过的，那么这里可以考虑添加一步让 LLM 生成提示词模版
# '''
#
#
# # Extract entities from text 从文本中提取实体
# class Entities(BaseModel):
#     """Identifying information about entities."""
#
#     names: List[str] = Field(
#         ...,
#         description="All the person, organization, or business entities that "
#                     "appear in the text",
#     )
#
#
# # prompt定义，这里还要修改一下，或者我再加入一个提示词模版生成器【防止用户语言表述不清楚】
# prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             "You are extracting organization and person entities from the text.",
#         ),
#         (
#             "human",
#             "Use the given format to extract information from the following "
#             "input: {question}",
#         ),
#     ]
# )
#
# entity_chain = prompt | llm.with_structured_output(Entities)
#
# entity_chain.invoke({"question": "Where was Amelia Earhart born?"}).names
#
#
# # 这里是生成全文检索，把检索映射到知识图谱
# def generate_full_text_query(input: str) -> str:
#     """
#     Generate a full-text search query for a given input string.
#
#     This function constructs a query string suitable for a full-text search.
#     It processes the input string by splitting it into words and appending a
#     similarity threshold (~2 changed characters) to each word, then combines
#     them using the AND operator. Useful for mapping entities from user questions
#     to database values, and allows for some misspelings.
#     """
#     full_text_query = ""
#     words = [el for el in remove_lucene_chars(input).split() if el]
#     for word in words[:-1]:
#         full_text_query += f" {word}~2 AND"
#     full_text_query += f" {words[-1]}~2"
#     return full_text_query.strip()
#
#
# # Fulltext index query
# def structured_retriever(question: str) -> str:
#     """
#     Collects the neighborhood of entities mentioned
#     in the question
#     """
#     result = ""
#     entities = entity_chain.invoke({"question": question})
#     for entity in entities.names:
#         response = graph.query(
#             """CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
#             YIELD node,score
#             CALL {
#               WITH node
#               MATCH (node)-[r:!MENTIONS]->(neighbor)
#               RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
#               UNION ALL
#               WITH node
#               MATCH (node)<-[r:!MENTIONS]-(neighbor)
#               RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output
#             }
#             RETURN output LIMIT 50
#             """,
#             {"query": generate_full_text_query(entity)},
#         )
#         result += "\n".join([el['output'] for el in response])
#     return result
#
#
# print(structured_retriever("Who is Elizabeth I?"))
#
#
# # 最终的检索器，把非结构/结构化检索合并。
# def retriever(question: str):
#     print(f"Search query: {question}")
#     structured_data = structured_retriever(question)
#     unstructured_data = [el.page_content for el in vector_index.similarity_search(question)]
#     final_data = f"""Structured data:
# {structured_data}
# Unstructured data:
# {"#Document ".join(unstructured_data)}
#     """
#     return final_data
#
#
# # Condense a chat history and follow-up question into a standalone question
# _template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question,
# in its original language.
# Chat History:
# {chat_history}
# Follow Up Input: {question}
# Standalone question:"""  # noqa: E501
# CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)
#
#
# # 引入查询重写部分——————其实这个我不是很理解他的作用
# def _format_chat_history(chat_history: List[Tuple[str, str]]) -> List:
#     buffer = []
#     for human, ai in chat_history:
#         buffer.append(HumanMessage(content=human))
#         buffer.append(AIMessage(content=ai))
#     return buffer
#
#
# _search_query = RunnableBranch(
#     # If input includes chat_history, we condense it with the follow-up question
#     (
#         RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
#             run_name="HasChatHistoryCheck"
#         ),  # Condense follow-up question and chat into a standalone_question
#         RunnablePassthrough.assign(
#             chat_history=lambda x: _format_chat_history(x["chat_history"])
#         )
#         | CONDENSE_QUESTION_PROMPT
#         | ChatOpenAI(temperature=0)
#         | StrOutputParser(),
#     ),
#     # Else, we have no chat history, so just pass through the question
#     RunnableLambda(lambda x: x["question"]),
# )
#
#
# template = """Answer the question based only on the following context:
# {context}
#
# Question: {question}
# Use natural language and be concise.
# Answer:"""
# prompt = ChatPromptTemplate.from_template(template)
#
# chain = (
#     RunnableParallel(
#         {
#             "context": _search_query | retriever,
#             "question": RunnablePassthrough(),
#         }
#     )
#     | prompt
#     | llm
#     | StrOutputParser()
# )
#
# chain.invoke({"question": "Which house did Elizabeth I belong to?"})
#
# chain.invoke(
#     {
#         "question": "When was she born?",
#         "chat_history": [("Which house did Elizabeth I belong to?", "House Of Tudor")],
#     }
# )
"""
    测试线---------------------------------------------------------------------------------
"""
