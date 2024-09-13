import json
import os

from flask import Flask, Response, render_template, request
from zhipuai import ZhipuAI

app = Flask(__name__)
zhipu_api_key = os.getenv("ZhipuAI")
api_key = zhipu_api_key
model_name = "glm-4-alltools"

# 初始化ZhipuAI客户端
client = ZhipuAI(api_key=api_key)

# 初始化对话上下文
context = [
    {
        "role": "system",
        "content": "你是一个对话式数据报表智能助手，你的任务是专业、严谨地回答用户的问题，完美处理用户提出的制作图表等请求，并进行数据分析。",
    }
]


# @app.route("/")
# def index():
#     return render_template("index.html")  # 渲染前端模板


def generate_chat(content):
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
                        "text": "帮我查询慕尼黑07/05至07/15的日平均气温。并将所有平均气温数值组成数列，绘出折线图显示趋势。"
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


@app.route("/chat", methods=["POST"])
def chat():
    global context

    # 获取前端发送的用户输入
    # user_input = request.json.get("input", "")
    user_input = ""
    # 如果用户输入是'exit'，则结束对话
    if user_input.lower() == "exit":
        context = []
        return Response(
            json.dumps({"message": "对话已结束"}), content_type="application/json"
        )

    return Response(generate_chat(user_input), content_type="application/json")


if __name__ == "__main__":
    app.run()