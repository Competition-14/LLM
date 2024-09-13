from langchain.tools import BaseTool
import requests


class BaiduSearchTool(BaseTool):
    name = "BaiduSearch"
    description = "A tool for performing web searches using Baidu."

    def _run(self, query: str):
        response = requests.get("https://www.baidu.com/s", params={"wd": query})
        return response.text  # 返回网页内容，可以用 BeautifulSoup 或其他工具解析

    async def _arun(self, query: str):
        raise NotImplementedError("BaiduSearchTool does not support async")


# 使用自定义工具
baidu_tool = BaiduSearchTool()
result = baidu_tool.run("LangChain 使用指南")
print(result)