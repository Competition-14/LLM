import os
from flask import Flask, request, jsonify
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
app = Flask(__name__)


"""
    另外的tool定义：
"""
# 为了返回提示词列表
def generate_optimized_prompts(question, databases_info):
    # 假设 Ollama_glm4 返回一个优化后的提示词列表
    # 每个提示词都包含特定数据库的上下文
    optimized_prompts = []
    for db_info in databases_info:
        prompt = f"根据以下表信息，请为数据库 '{db_info['database_name']}' 生成一个 SQL 查询：\n{db_info['table_info']}\n\n问题：{question}"
        optimized_prompts.append(prompt)
    return optimized_prompts


@app.route('/chat', methods=['POST'])
def chat_with_llm():
    # 获取用户提问
    question = request.values.get("question")
    # 这里存在一个问题，LLMGraphTransformer()是langchain的一个接口，但是当时说的是只支持 OpenAI 和 Mistral 的函数调用模型
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0125",
                     base_url="https://api.gptsapi.net/v1")  # gpt-4-0125-preview occasionally has issues
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
    # 提示词
    analyse_template = """
    You are a data analyst with the ability to understand user questions and analyze the provided database information to determine which databases should be queried. Although the database names in the question may not exactly match those in the database info, your task is to identify the most relevant databases to query. 

    Databases Info:
    {databases_info}

    Question: {question}

    Based on the above database information and the question, please identify the databases that are likely needed for the query. However, ensure that your response strictly uses the exact 'database_name' as provided in the 'databases_info'. Make sure your selection is accurate and directly addresses the question.
    """
    analyse_prompt = ChatPromptTemplate.from_template(analyse_template)
    # You are a data analyst with the ability to understand user questions and analyze the provided database information to determine which databases should be queried.Although the database names in the question may not exactly match those in the database info, your task is to identify the most relevant databases to query.
    """     
    旧提示词
    You are an AI assistant with the ability to understand user questions and generate SQL queries. Based on the provided databases info and the user's question, please identify which databases are relevant for answering the question. You must strictly use the database names provided in the 'database_name' field of the databases info.

    Databases Info:
    {databases_info}

    Question: {question}

    Please identify the relevant databases using only the exact 'database_name' provided in the databases info. Make sure your queries address the question accurately.
    """
    # 使用 Ollama_glm4 分析要查询的数据库
    database_analysis_chain = (
            analyse_prompt
            | Ollama_llm_glm4
            | StrOutputParser()
    )
    # 获取分析结果，判断要查询的数据库
    database_to_query = database_analysis_chain.invoke(
        {"databases_info": databases_info, "question": question}
    )

    # 打印分析结果（这里你需要解析模型返回的数据库列表）
    print(database_to_query)

    # 提示词
    sql_template = """
    You are an AI assistant with the ability to generate SQL queries. Based on the provided databases info and the user's question, please generate a valid SQL query.

Databases Info:
{databases_info}

Question: {question}

Please provide the SQL query that answers the question based on the given schema. The query should only include tables and fields that are present in the provided 'Databases Info'. **Do not assume or create any fields that are not explicitly listed in the 'Databases Info'.**
    """
    sql_prompt = ChatPromptTemplate.from_template(sql_template)
    # 原本提示词在前面，但是现在我觉得会有问题，所以要改一下

    # 为每个相关数据库生成 SQL 查询
    sql_queries = []
    for db_info in databases_info:
        if db_info["database_name"] in database_to_query:  # 判断分析结果中是否包含此数据库
            query = (
                    sql_prompt
                    | Ollama_llm_nl2Cypher
                    | StrOutputParser()
            ).invoke({
                "databases_info": db_info["table_info"],
                "question": question
            })
            sql_queries.append((db_info["database_name"], query))
    print(sql_queries)
    # 执行 SQL 查询并合并结果
    results = []
    for db_name, query in sql_queries:
        database = databases[db_name]
        try:
            # 使用 QuerySQLDataBaseTool 运行查询
            query_tool = QuerySQLDataBaseTool(db=database)
            result = query_tool.run(query)
            print("-----------------------------")
            print(result)
            print("-----------------------------")
            results.extend(result)
        except Exception as e:
            print(f"Error running query on {db_name}: {e}")
    print(results)
    """
        对于每个数据库要有一个data的json返回
        prompt里面要有data还要有图片类型（这里定义几个，然后让llm从里面选）
    """
    json_template = """
    You are an expert data analyst. Based on the given SQL query results, you will generate chart data in JSON format for a chart. The JSON must follow this exact structure:

{
    "chart_type": "bar",  // The type of the chart, 
    "data": { 
        "labels": ["Category A", "Category B", "Category C"],  // The x-axis labels
        "values": [10, 20, 30]  // The y-axis values corresponding to each label
    },
    "sql_results": [  // The original SQL query results formatted as an array of dictionaries
        {"Category": "Category A", "Value": 10},
        {"Category": "Category B", "Value": 20},
        {"Category": "Category C", "Value": 30}
    ]
}

Ensure that:
- The field `chart_type` is always "bar".
- The `data.labels` array corresponds to the distinct categories from the SQL results.
- The `data.values` array contains the values corresponding to each label.
- The `sql_results` field contains the original SQL results in a structured format, with keys like "Category" and "Value".
    """
    json_prompt = ChatPromptTemplate.from_template(json_template)

    return jsonify(results)


#   如果当前模块是作为主入口来运行则条件成立
if __name__ == '__main__':
    app.run()
