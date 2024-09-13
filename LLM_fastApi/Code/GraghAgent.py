from langchain.agents import AgentType, initialize_agent, Tool
from langchain_community.document_loaders.pdf import BasePDFLoader, PyPDFLoader
from langchain_community.llms.ollama import Ollama
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
# os.environ["OPENAI_API_KEY"] = "sk-"
# os.environ["NEO4J_URI"] = "bolt://18.212.197.205:7687"
# os.environ["NEO4J_USERNAME"] = "neo4j"
# os.environ["NEO4J_PASSWORD"] = "tendencies-meanings-enlistment"
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
graph = Neo4jGraph()


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


"""
    测试线---------------------------------------------------------------------------------
"""


vector_index = Neo4jVector.from_existing_graph(
    openai_embeddings,
    search_type="hybrid",
    node_label="Document",
    text_node_properties=["text"],
    embedding_node_property="embedding"
)


# Retriever
# 这里是抽取question里面的实体，为了结构化查询
# Extract entities from text 从文本中提取实体
class Entities(BaseModel):
    """Identifying information about entities."""

    names: List[str] = Field(
        ...,
        description="All entities in the text that could be used to create data reports, such as persons, "
                    "organizations, etc",
        # All the person, organization, or business entities that "
        #                     "appear in the text
    )

    @classmethod
    def from_json(cls, json_data: dict):
        _names = []
        if 'entities' in json_data:
            for entity in json_data['entities']:
                _names.append(entity['entity'])
        return cls(names=_names)


# prompt定义，这里还要修改一下，或者我再加入一个提示词模版生成器【防止用户语言表述不清楚】

# 这里格式化一下，Set up a parser + inject instructions into the prompt template.
parser = PydanticOutputParser(pydantic_object=Entities)    # 转换一下

# 更新 ChatPromptTemplate 以包含新的提示词
chat_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are extracting all relevant entities from the text, including but not limited to persons, "
            "organizations, locations, dates, and other important entities that might be useful for generating data "
            "reports.",
            # "You are extracting organization and person entities from the text.",
        ),
        (
            "human",
            "Use the given format to extract information from the following "
            "input: {question}. Please ensure the output is in valid JSON format.",
        ),
    ]
)

# 创建提示模板
prompt_template = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)


# 这里是生成全文检索，把检索映射到知识图谱————————这里的input改不改再说吧
def generate_full_text_query(input: str) -> str:
    """
    Generate a full-text search query for a given input string.

    This function constructs a query string suitable for a full-text search.
    It processes the input string by splitting it into words and appending a
    similarity threshold (~2 changed characters) to each word, then combines
    them using the AND operator. Useful for mapping entities from user questions
    to database values, and allows for some misspelings.
    """
    full_text_query = ""
    words = [el for el in remove_lucene_chars(input).split() if el]
    for word in words[:-1]:
        full_text_query += f" {word}~2 AND"
    full_text_query += f" {words[-1]}~2"
    return full_text_query.strip()


# Fulltext index query
def structured_retriever(question: str) -> str:
    """
    Collects the neighborhood of entities mentioned
    in the question
    """
    result = ""

    entity_query = question
    # 格式化输入
    _input = chat_prompt.format_prompt(question=entity_query)

    # 获取 LLM 输出
    output = llm(_input.to_string())

    # 解析输出
    parsed_output = json.loads(output)

    # 使用自定义的 from_json 方法解析输出
    entities = Entities.from_json(parsed_output)

    # 获取 names 列表
    names = entities.names
    print(names)

    for entity in names:
        response = graph.query(
            """CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
            YIELD node,score
            CALL {
              WITH node
              MATCH (node)-[r:!MENTIONS]->(neighbor)
              RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
              UNION ALL
              WITH node
              MATCH (node)<-[r:!MENTIONS]-(neighbor)
              RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output
            }
            RETURN output LIMIT 50
            """,
            {"query": generate_full_text_query(entity)},
        )
        result += "\n".join([el['output'] for el in response])
    return result


# 最终的检索器，把非结构/结构化检索合并。
def retriever(question: str):
    print(f"Search query: {question}")
    structured_data = structured_retriever(question)
    unstructured_data = [el.page_content for el in vector_index.similarity_search(question)]
    final_data = f"""Structured data:
{structured_data}
Unstructured data:
{"#Document ".join(unstructured_data)}
    """
    return final_data


# 接下来组装RAG链，有个记忆模块需要搞一下
# Condense a chat history and follow-up question into a standalone question
_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question,
in its original language.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""  # noqa: E501
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)


# 引入查询重写部分
def _format_chat_history(chat_history: List[Tuple[str, str]]) -> List:
    buffer = []
    for human, ai in chat_history:
        buffer.append(HumanMessage(content=human))
        buffer.append(AIMessage(content=ai))
    return buffer


# 分两种情况，分为有无历史对话
_search_query = RunnableBranch(
    # If input includes chat_history, we condense it with the follow-up question
    (
        RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
            run_name="HasChatHistoryCheck"
        ),  # Condense follow-up question and chat into a standalone_question
        RunnablePassthrough.assign(
            chat_history=lambda x: _format_chat_history(x["chat_history"])
        )
        | CONDENSE_QUESTION_PROMPT
        | ChatOpenAI(temperature=0)
        | StrOutputParser(),
    ),
    # Else, we have no chat history, so just pass through the question
    RunnableLambda(lambda x: x["question"]),
)


template = """Answer the question based only on the following context:
{context}

Question: {question}
Use natural language and be concise.
Answer:"""
prompt = ChatPromptTemplate.from_template(template)

graghrag_chain = (
    # RunnableParallel是一个并行处理任务，context就是前面链和检索的答案，RunnablePassthrough()表示问题直接传递到下一个环节——————后面prompt使用了这两个变量
    RunnableParallel(
        {
            "context": _search_query | retriever,
            "question": RunnablePassthrough(),
        }
    )
    | prompt
    | llm
    | StrOutputParser()
)


# 下面两个只是测试一下的
# graghrag_chain.invoke({"question": "Which house did Elizabeth I belong to?"})
#
# graghrag_chain.invoke(
#     {
#         "question": "When was she born?",
#         "chat_history": [("Which house did Elizabeth I belong to?", "House Of Tudor")],
#     }
# )
"""
    测试线---------------------------------------------------------------------------------
"""

"""
    以上为GraghRAG的链构建，接下来要拼接整个Agent
    注意，还可以结合CRAG（添加一个分解、重组算法）
"""

"""
    LEDVR工作流，但是我这里想着一个问题，也不是说检索了这边，我应该是意图识别，根据我的问题，LLM决定我是要制作报表还是要普通问答
"""

"""
    1.prompt：把RAG的结果结合到prompt中
"""

"""
    2.记忆模块添加，采用向量数据库
"""

"""
    3.组装LLM链/封装agent.（采用什么链再斟酌一下）
"""

# # 创建多个数据库连接，这里到时候可能会找个链接或者别的东西动态获取
# databases = {
#     "sqlite_db": SQLDatabase.from_uri("sqlite:///./chinook_sqlite.db"),
#     "mysql_db": SQLDatabase.from_uri("mysql+mysqlconnector://root:admin@localhost:3306/chinook_mysql"),
#     # 添加更多数据库连接
# }


# SQL 查询工具函数
def sql_query_tool_function(query: str) -> str:
    # 假设这里是执行 SQL 查询的逻辑
    # 实际代码中可能会连接数据库并执行查询
    return f"SQL query result for: {query}"


# 初始化 SQL 查询 Agent
sql_agent = initialize_agent(
    agent_type=AgentType.SINGLE_AGENT,
    tools=[Tool(name="sql_query_tool", func=sql_query_tool_function, description="Tool for SQL queries")]
)


# Cypher 查询工具函数
def cypher_query_tool_function(query: str) -> str:
    # 假设这里是执行 Cypher 查询的逻辑
    # 实际代码中可能会连接 Neo4j 数据库并执行查询
    return f"Cypher query result for: {query}"


# 初始化 Cypher 查询 Agent
cypher_agent = initialize_agent(
    agent_type=AgentType.SINGLE_AGENT,
    tools=[Tool(name="cypher_query_tool", func=cypher_query_tool_function, description="Tool for Cypher queries")]
)


# 主 Agent 工具函数
def main_agent_tool_function(request: dict) -> str:
    if 'sql_query' in request:
        return sql_agent({"query": request['sql_query']})
    elif 'cypher_query' in request:
        return cypher_agent({"query": request['cypher_query']})
    else:
        return "Unknown query type."


# 初始化主 Agent
main_agent = initialize_agent(
    llm=Ollama_llm_glm4,
    agent_type=AgentType.SINGLE_AGENT,
    tools=[Tool(name="main_agent_tool", func=main_agent_tool_function, description="Main agent tool to route queries.")]
)


# 测试主 Agent
def test_main_agent():
    sql_result = main_agent({"sql_query": "SELECT * FROM users"})
    cypher_result = main_agent({"cypher_query": "MATCH (n:Person) RETURN n"})

    print("SQL Result:", sql_result)
    print("Cypher Result:", cypher_result)


print("其他程序正常")
# agent = initialize_agent(
#
# )

# 完成后关闭 Neo4j 连接
graph.close()
driver.close()


# 一直有报错；
# Exception ignored in: <function Driver.__del__ at 0x000001A0A2F635B0>
# Traceback (most recent call last):
#   File "O:\Python\Flask\Competition\LLM_fastApi\.venv\lib\site-packages\neo4j\_sync\driver.py", line 544, in __del__
#   File "O:\Python\Flask\Competition\LLM_fastApi\.venv\lib\site-packages\neo4j\_sync\driver.py", line 641, in close
# TypeError: catching classes that do not inherit from BaseException is not allowed

