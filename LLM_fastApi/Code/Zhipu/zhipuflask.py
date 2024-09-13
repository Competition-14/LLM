import os
from flask import Flask, jsonify, request, Response
from zhipuai import ZhipuAI

app = Flask(__name__)

# 从环境变量获取 ZhipuAI 的 API Key
zhipu_api_key = os.getenv("ZhipuAI")

# 初始化 ZhipuAI 客户端
client = ZhipuAI(api_key=zhipu_api_key)


# 定义一个路由，处理查询旅游数据的请求
@app.route('/query_data', methods=['POST'])
def query_tourist_data():
    # 定义 AI 请求体
    response = client.chat.completions.create(
        model="glm-4-alltools",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Please help me query the national travel data for the Labor Day holiday from 2018 to 2024, and present the data trend in a bar chart."
                    }
                ]
            }
        ],
        stream=True,  # 保持流式响应
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_tourist_data_by_year",
                    "description": ("Used to query the national travel data for each year, "
                                    "input the year range (from_year, to_year), and return the corresponding travel data, "
                                    "including the total number of trips, the number of trips by different modes of transportation, etc."),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "type": {
                                "description": ("Mode of transportation, default is by_all, "
                                                "train = by_train, plane = by_plane, self-driving = by_car."),
                                "type": "string"
                            },
                            "from_year": {
                                "description": "Start year, formatted as yyyy.",
                                "type": "string"
                            },
                            "to_year": {
                                "description": "End year, formatted as yyyy.",
                                "type": "string"
                            }
                        },
                        "required": ["from_year", "to_year"]
                    }
                }
            },
            {
                "type": "code_interpreter"
            }
        ]
    )

    # 使用流式响应返回结果
    def stream():
        for chunk in response:
            # # 将每个块逐个返回
            # # 获取choices中的delta
            # delta = chunk.choices[0].delta
            #
            # # 检查是否有content
            # if delta.content:
            #     content = delta.content
            #     yield f"Content: {content}\n".encode('utf-8')
            # yield chunk.encode('utf-8')
            content = chunk.choices[0].delta.content
            yield content

    # 返回流式数据
    return Response(stream(), content_type='application/json')


# 启动 Flask 应用
if __name__ == '__main__':
    app.run()
