from langchain.agents import AgentType, initialize_agent, Tool
from langchain.chains.sql_database.query import create_sql_query_chain
from langchain_community.document_loaders.pdf import BasePDFLoader, PyPDFLoader
from langchain_community.llms.ollama import Ollama
from langchain_community.tools import QuerySQLDataBaseTool
from langchain_community.utilities import SQLDatabase
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
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
import os
from langchain_community.graphs import Neo4jGraph
# from langchain.document_loaders import WikipediaLoader
from langchain_community.document_loaders import WikipediaLoader, TextLoader
from langchain.text_splitter import TokenTextSplitter
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from neo4j import GraphDatabase
import json
from yfiles_jupyter_graphs import GraphWidget
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_core.runnables import ConfigurableField, RunnableParallel, RunnablePassthrough


# 获取环境变量
openai_api_key = os.getenv("OPENAI_API_KEY")    # 通义千问应该也行
# neo4j_uri = os.getenv("NEO4J_URI")
# NEO4J_URI = "bolt://localhost:7687"
NEO4J_URI = "neo4j://localhost:7687"
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = "neo4j"
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
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

# 这里存在一个问题，LLMGraphTransformer()是langchain的一个接口，但是当时说的是只支持 OpenAI 和 Mistral 的函数调用模型
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0125", base_url="https://api.gptsapi.net/v1")  # gpt-4-0125-preview occasionally has issues
# Ollama模型导入
Ollama_llm_nl2sql = Ollama(model="HridaAI/hrida-t2sql:latest")
Ollama_llm_nl2Cypher = Ollama(model="tomasonjo/llama3-text2cypher-demo:latest")
Ollama_llm_glm4 = Ollama(model="glm4")
# 配置 OpenAI 嵌入模型
openai_embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002",  # 指定模型名
    base_url="https://api.gptsapi.net/v1"  # 指定API URL，如果是默认的OpenAI URL，可以省略
)
llm_transformer = LLMGraphTransformer(llm=llm)



import mysql.connector
# conn=mysql.connector.connect(host = '127.0.0.1' # 连接名称，默认127.0.0.1
# ,user = 'root' # 用户名
# ,passwd='password' # 密码
# ,port= 3306 # 端口，默认为3306
# ,db='test' # 数据库名称
# ,charset='utf8' # 字符编码
# )
# cur = conn.cursor() # 生成游标对象
# sql="select * from `student` " # SQL语句
# cur.execute(sql) # 执行SQL语句
# data = cur.fetchall()   # 通过fetchall方法获得数据
# for i in data[:2]: # 打印输出前2条数据
#     print(i)
# cur.close() # 关闭游标
# conn.close() # 关闭连接
# 创建多个数据库连接，这里到时候可能会找个链接或者别的东西动态获取
databases = {
    # "sqlite_db": SQLDatabase.from_uri("sqlite:///./chinook_sqlite.db"),
    "mysql_db1": SQLDatabase.from_uri("mysql+mysqlconnector://root:123456@localhost:3306/llm_1"),
    "mysql_db2": SQLDatabase.from_uri("mysql+mysqlconnector://root:123456@localhost:3306/llm_2"),
    # 添加更多数据库连接
}
databases_info = []
# 测试连接，查看是否成功___注意这里输出的是str
for db_name, database in databases.items():
    try:
        table_info = database.get_table_info()
        print(f"Tables in {db_name}: {table_info}")
        # 在 \n 和 \t 前后加空格
        processed_info = table_info.replace('\n', ' \n ').replace('\t', ' \t ')
        databases_info.append({
            "database_name": db_name,
            "table_info": processed_info
        })
    except Exception as e:
        print(f"Error querying {db_name}: {e}")

print(databases_info)
# 引入数据库工具包
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities.sql_database import SQLDatabase


# # SQL 查询工具函数
# def sql_query_tool_function(query: str) -> str:
#     # 假设这里是执行 SQL 查询的逻辑
#     # 实际代码中可能会连接数据库并执行查询
#     return f"SQL query result for: {query}"
#
#
# # 初始化 SQL 查询 Agent
# sql_agent = initialize_agent(
#     agent_type=AgentType.SINGLE_AGENT,
#     tools=[Tool(name="sql_query_tool", func=sql_query_tool_function, description="Tool for SQL queries")]
# )
#
#
# # Cypher 查询工具函数
# def cypher_query_tool_function(query: str) -> str:
#     # 假设这里是执行 Cypher 查询的逻辑
#     # 实际代码中可能会连接 Neo4j 数据库并执行查询
#     return f"Cypher query result for: {query}"
#
#
# # 初始化 Cypher 查询 Agent
# cypher_agent = initialize_agent(
#     agent_type=AgentType.SINGLE_AGENT,
#     tools=[Tool(name="cypher_query_tool", func=cypher_query_tool_function, description="Tool for Cypher queries")]
# )
#
#
# # 主 Agent 工具函数
# def main_agent_tool_function(request: dict) -> str:
#     if 'sql_query' in request:
#         # 调用 SQL 查询 Agent
#         return sql_agent({"query": request['sql_query']})
#     elif 'cypher_query' in request:
#         # 调用 Cypher 查询 Agent
#         return cypher_agent({"query": request['cypher_query']})
#     else:
#         return "Unknown query type."
#
#
# # 初始化主 Agent
# main_agent = initialize_agent(
#     agent_type=AgentType.SINGLE_AGENT,
#     tools=[Tool(name="main_agent_tool", func=main_agent_tool_function, description="Main agent tool to route queries.")]
# )


# 如果不用agent而用链的形式
# prompt里面记得拼接RAG内容和历史记录（如果能加个引用功能会更好）
chat_template = """
You are an AI assistant with the ability to generate SQL queries. Based on the provided databases info and the user's question, please generate a valid SQL query.

Databases Info:
{databases_info}

Question: {question}

Please provide the SQL query that answers the question based on the given schema. Make sure your query is valid and addresses the question accurately.
"""
chat_prompt = ChatPromptTemplate.from_template(chat_template)

final_chain = (
    chat_prompt
    | Ollama_llm_nl2Cypher
    | StrOutputParser()
)


print(final_chain.invoke({"databases_info": databases_info, "question": "我想知道我们公司的用户姓名和国籍，按年龄从大到小排序"}))


# 原来只能实现一个库的提取
# 现在要实现多表的查询
# 使用 Ollama_glm4 分析需求并优化提示词
"""
    databases_info是个列表，每个元素里面包含“database_name”和“table_info”
"""


def generate_optimized_prompts(question, databases_info):
    # 假设 Ollama_glm4 返回一个优化后的提示词列表
    # 每个提示词都包含特定数据库的上下文
    optimized_prompts = []
    for db_info in databases_info:
        prompt = f"根据以下表信息，请为数据库 '{db_info['database_name']}' 生成一个 SQL 查询：\n{db_info['table_info']}\n\n问题：{question}"
        optimized_prompts.append(prompt)
    return optimized_prompts

optimized_prompts = generate_optimized_prompts(question, databases_info)



print("其他程序正常")
