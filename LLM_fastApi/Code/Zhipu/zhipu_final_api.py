import re
from flask import Flask, request, jsonify
from flask_cors import cross_origin, CORS
from zhipuai import ZhipuAI
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
openai_api_key = os.getenv("OPENAI_API_KEY")  # 通义千问应该也行
zhipu_api_key = os.getenv("ZhipuAI")
api_key = zhipu_api_key
# 初始化ZhipuAI客户端
client = ZhipuAI(api_key=api_key)
# neo4j_uri = os.getenv("NEO4J_URI")
# NEO4J_URI = "bolt://localhost:7687"
NEO4J_URI = "neo4j://localhost:7687"
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = "neo4j"
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
app = Flask(__name__)
(CORS(app))
CORS(app, resources=r'/*', supports_credentials=True)

def str_json(raw_data):

    # 尝试将字符串解析为 JSON 对象
    try:
        data = json.loads(raw_data)
    except json.JSONDecodeError:
        return jsonify({"success": False, "message": "Invalid JSON format."}), 40
    if not data:
        return jsonify({"success": False, "message": "No data received."}), 400
    data = data.get('data')
    return data


# 初始化对话上下文
context = [
    {
        "role": "system",
        "content": "你是一个对话式数据报表智能助手，你的任务是专业、严谨地回答用户的问题，你必须确保阅读了用户提供的数据或者网络数据，正确处理用户提出的制作图表请求，最后进行数据分析。",
    }
]

# history_content = {
#     {
#         "role": "system",
#         "content": ""
#     }
# }

"""
    另外的tool定义：
"""


def generate_chat(content, question, databases_info, sql_result):
    global context

    # 将用户输入添加到上下文中
    context.append({"role": "user", "content": content})

    # 发送请求到模型
    response = client.chat.completions.create(
        model="glm-4-alltools",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": question
                    }
                ]
            }
        ],
        stream=True,  # 保持流式响应
        tools=[
            {
                "type": "web_browser"
            },
            {
                "type": "code_interpreter",  # 智能编程助手
            },
        ]
    )

    for chunk in response:
        print("-------------------chunk-------------------")
        print(chunk)
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="")
        for choice in chunk.choices:
            # print("-------------------choice-------------------")
            # print(choice)
            yield json.dumps({"message": choice.delta.content})
        #     # 如果finish_reason是'stop'，说明这是最后一个Chunk或者句子已经完成
        #     if choice.finish_reason == "stop":
        #         return
        # else:
        #     # 如果这个Chunk中没有finish_reason为'stop'的Choice，则继续下一个Chunk
        #     continue
    return


# 为了返回提示词列表
def generate_optimized_prompts(question, databases_info):
    # 假设 Ollama_glm4 返回一个优化后的提示词列表
    # 每个提示词都包含特定数据库的上下文
    optimized_prompts = []
    for db_info in databases_info:
        prompt = f"根据以下表信息，请为数据库 '{db_info['database_name']}' 生成一个 SQL 查询：\n{db_info['table_info']}\n\n问题：{question}"
        optimized_prompts.append(prompt)
    return optimized_prompts


@app.route('/chat', methods=['POST', 'GET'])
#@cross_origin()
def chat_with_llm():
    # 获取用户提问
    raw_data = request.data.decode('utf-8')
    data = str_json(raw_data)
    # question = request.values.get("question")
    question = data['question']
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
        "mysql_db3": SQLDatabase.from_uri("mysql+mysqlconnector://root:123456@localhost:3306/llm_3"),
        # 添加更多数据库连接
    }
    table_descriptions = {
        "mysql_db1": {
            "user": "表user用于存储。",
        },
        "mysql_db2": {
            "feedback": "该表记录用户对模型输出的反馈信息。",
            "evaluation": "模型性能评估数据。",
        },
        "mysql_db3": {
            "logs": "系统交互日志表。",
            "user_sessions": "记录用户的会话历史。",
        }
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
    你是一名数据分析师，能够理解用户问题，并分析提供的数据库信息，以确定应该查询哪些数据库。如果我的问题与数据库的内容不相关，你就不需要分析了，就输出一个“无”。

    数据库信息：
    {databases_info}

    问题：{question}

    根据上述数据库信息和问题，请识别可能需要查询的数据库。然而，请确保在回应中严格使用数据库信息中提供的确切'database_name'。确保你的选择准确无误，直接针对问题作出回应。
    """
    # 另外的内容：
    """
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

    你是我的服务的一部分，你需要实现我的服务的部分功能
    你是**严谨的**，不要输出除了SQL语句以外的建议等内容
    你的任务是**根据我的数据库数据和问题，写出SQL查询语句，这个SQL语句直接会被QuerySQLDataBaseTool工具用于query_tool.run(query)完成数据库查询**
    - 注意，你只需要写**SQL查询部分的内容，你不要写额外的任何内容**

    Databases Info:
    {databases_info}

    Question: {question}

    Please provide the SQL query that answers the question based on the given schema. The query should only include tables and fields that are present in the provided 'Databases Info'. **Do not assume or create any fields that are not explicitly listed in the 'Databases Info'.**
    """
    sql_prompt = ChatPromptTemplate.from_template(sql_template)
    # 原本提示词在前面，但是现在我觉得会有问题，所以要改一下
    number = 0
    # 为每个相关数据库生成 SQL 查询
    sql_queries = []
    # query_database = []
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
            # 这个是图表数量
            number += 1
            # query_database.append()
    print(sql_queries)
    # 执行 SQL 查询并合并结果
    if number > 0:
        # 有要查询的数据库
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
                results.append(result)
            except Exception as e:
                print(f"Error running query on {db_name}: {e}")
        print(results)
        """
            对于每个数据库要有一个data的json返回
            prompt里面要有data还要有图片类型（这里定义几个，然后让llm从里面选）
        """
        # 根据前面返回的数量
        json_template = """
        System:
            你是我的服务的一部分，你需要实现我的服务的部分功能。
            你是**严谨的**，我的data数量和内容是根据数据库来的。虽然实例只给3条示范，但你要根据实际情况灵活改写。
            你的任务是**根据我的输入返回对应图表的JSON格式的数据，数据条数必须和数据库查询到的数据条数一致。**
            - 注意，你只需要**根据对应图表JSON格式，填充与数据库相符合的data，数据条数和数据库内容一致，不要写任何额外的说明或注解。**
            下面是一个例子，仅作格式参考，不代表具体的data数量：
            {example}
        User:
            我的问题是{question}
            我的字段名和数据库示例信息是{databases_info}
            执行过的SQL查询为{sql_queries}，结果为{results}
        """

        #     json_template = """
        # System:
        #     你是我的服务的一部分，你需要实现我的服务的部分功能
        #     你是**灵活的**，我的data数量和内容是根据数据库来的，实例只给3条示范，你要根据实际情况灵活改写
        #     你的任务是**根据我的输入返回对应图表的JSON格式的数据，例子只给出格式，不代表具体的data数量，这个JSON数据会被用作前端表格的生成**
        #     - 注意，你只需要**根据对应图表JSON格式，填充与数据库相符合的data，其中"data"数据条数必须要和数据库查询到的一样，不要写任何额外的说明、注解，但是请你注意示例中的data要求**
        #     下面是例子：
        #     {example}
        # User:
        #     我的问题是{question}
        #     我的数据信息是{databases_info}
        #     """
        json_prompt = ChatPromptTemplate.from_template(json_template)
        json_chain = (
                json_prompt
                | llm
                | StrOutputParser()
        )

        example: str = """
        柱状图:
    \"\"\"
    {
      "xAxis": {
        "data": ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
      },
      "yAxis": {},
      "series": [
        {
          "type": 'bar',
          "data": [23, 24, 18, 25, 27, 28, 25]
        }
      ]
    };
    \"\"\"
    折线图:
    \"\"\"
    {
      "xAxis": {
        "type": 'category',
        "data": ['A', 'B', 'C', 'D']
      },
      "yAxis": {
        "type": 'value'
      },
      "series": [
        {
          "data": [120, 200, 150, 250],
          "type": 'line'
        }
      ]
    };
    \"\"\"
    饼图:
    \"\"\"
    {
      "series": [
        {
          "type": 'pie',
          "data": [
            {
              "value": 335,
              "name": '直接访问'
            },
            {
              "value": 234,
              "name": '联盟广告'
            },
            {
              "value": 1548,
              "name": '搜索引擎'
            }
          ]
        }
      ]
    };
    \"\"\"
    散点图:
    \"\"\"
    {
      "xAxis": {
        "data": ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
      },
      "yAxis": {},
      "series": [
        {
          "type": 'scatter',
          "data": [220, 182, 191, 234, 290, 330, 310]
        }
      ]
    };
    \"\"\"
    """

        llm_result = json_chain.invoke(
            {"databases_info": databases_info, "question": question, "example": example, "results": results,
             "sql_queries": sql_queries}
        )
        # "请根据世界人口统计表，查询各国2021年的人口信息，画成柱状图"
        print(llm_result)
        # 这里转换一下字符串，因为模型输出有点问题
        try:
            # 使用正则表达式提取最大花括号内的内容
            match = re.search(r'{.*}', llm_result, re.DOTALL)
            if match:
                json_str = match.group(0)

                # 替换单引号为双引号以符合 JSON 标准
                json_str = json_str.replace("'", '"')
                print("----------------json_str----------------")
                print(json_str)
                print("----------------------------------------")
                # 解析清理后的 JSON 数据
                parsed_result = json.loads(json_str)
                print("----------------parsed_result----------------")
                print(parsed_result)
                print("---------------------------------------------")
                # 提取最外层的 JSON 对象
                if isinstance(parsed_result, dict):
                    result_data = parsed_result
                else:
                    # 如果最外层不是字典，处理异常情况
                    result_data = {}

                # 最后一步：数据分析，根据我的问题以及数据图表分析。
                final_template = """
                        System:
                            你是我的服务的一部分，你需要实现我的服务的部分功能。
                            你是**认真的**，你不仅会对我的数据写出理解和分析，包括趋势等，还会联网查询一些有用的内容，对分析进行补充。
                            我已经根据问题查询到了需要的数据，但是我还需要一段数据分析和说明，希望你用和我问题领域相关的专业语言，我的图表会放在文末，所以你也要承上启下，做个简单的文字说明。
                            -注意 图表是在文末，所以有类似“图表放在此处”的填空，默认图表在最后。
                        User:
                            我的问题是{question}
                            我的字段名和数据库示例信息是{databases_info}
                            执行过的SQL查询为{sql_queries}，结果为{results}
                            我用于生成图表的数据为{parsed_result}
                """
                final_prompt = ChatPromptTemplate.from_template(final_template)
                final_chain = (
                        final_prompt
                        | llm
                        | StrOutputParser()
                )
                illustration = final_chain.invoke({"question": question, "databases_info": databases_info, "sql_queries": sql_queries, "results": results, "parsed_result": parsed_result})
                # 使用 jsonify 返回 JSON 数据
                result_data['illustration'] = illustration
                return jsonify(result_data)
            else:
                return jsonify({"error": "No valid JSON found"}), 400

        except json.JSONDecodeError as e:
            # 处理 JSON 解析错误
            return jsonify({"error": f"Invalid JSON data:{e}"}), 400
        # 还有的问题是这个出来的json数据外面会带有
        # return jsonify(llm_result)
    # 没有要查询的数据库
    # 调用网络搜索吧，这里使用智谱的
    return jsonify({"error": f"none"}), 400


#   如果当前模块是作为主入口来运行则条件成立
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

