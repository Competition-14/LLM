from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_community.document_loaders.pdf import BasePDFLoader, PyPDFLoader
import os
from langchain_community.graphs import Neo4jGraph
from langchain.text_splitter import TokenTextSplitter
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer

openai_api_key = os.getenv("OPENAI_API_KEY")
NEO4J_URI = "neo4j://localhost:7687"
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = "neo4j"

"""
    本代码是为了创建图的，想法是引入ai智能构建图，根据导入的pdf或者其他文件进行建图
    目前PDF如果有很多奇怪的符号效果不好
"""
app = Flask(__name__)
CORS(app)


@app.route('/upload', methods=['POST'])
def addgragh():  # put application's code here
    llm = ChatOpenAI(temperature=0,
                     model_name="gpt-3.5-turbo-0125",
                     base_url="https://api.gptsapi.net/v1")
    try:
        # 尝试建图的逻辑
        files = request.files.getlist('files')
        if not files:
            return jsonify({"error": "No files uploaded"}), 400
        responses = []
        # 有文件上传
        # 这里要针对不同文件处理吧
        pdf_raw_documents = []
        doc_raw_documents = []
        txt_raw_documents = []
        for file in files:
            # 这里是针对PDF文件
            if file.filename.endswith('.pdf'):
                # 保存文件到临时路径
                temp_file_path = os.path.join(r"C:\Users\ASUS\Desktop\服务外包\省赛14\大模型", file.filename)
                file.save(temp_file_path)
                # 加载 PDF 文件
                pdf_loader = PyPDFLoader(temp_file_path)
                # pdf_loader = PyPDFLoader(file)
                pdf_documents = pdf_loader.load()
                # 将加载的文档添加到列表中
                pdf_raw_documents.extend(pdf_documents)
        print(pdf_raw_documents)
        text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=24)
        pdf_documents = text_splitter.split_documents(pdf_raw_documents)
        print(pdf_documents)
        # 接下来要尝试建图了
        graph = Neo4jGraph()
        # 使用 AI 进行分析，生成节点和关系提示词
        prompt = (f"I am a data analyst, and my goal is to store data in a Neo4j database to facilitate future data queries and the creation of data charts. I am particularly focused on the significance of the data and the relationships between different data points. Based on the documents I have uploaded, please return appropriate node names. Note that the response should only be a Python-syntax-compliant list of strings, without any additional code or variables: {pdf_documents}.")
        ai_response = llm.invoke(prompt)
        print(ai_response)
        ai_response_content = ai_response.content

        # Convert ai_response_content to a list (if needed)
        # allowed_node = eval(ai_response_content)  # Convert to list, ensuring the content is safe
        row = []
        row = [item.strip("'") for item in ai_response_content.strip("[]").split(',')]
        # row =
        allowed_node = row
        llm_transformer_filtered = LLMGraphTransformer(
            llm=llm,
            allowed_nodes=allowed_node,
            # allowed_relationships=["NATIONALITY", "LOCATED_IN", "WORKED_AT", "SPOUSE"],
        )
        graph_documents_filtered = llm_transformer_filtered.convert_to_graph_documents(
            pdf_documents
        )
        graph.add_graph_documents(
            graph_documents_filtered,
            baseEntityLabel=True,
            include_source=True
        )
        # for i, d in tqdm(enumerate(documents), total=len(documents)):
        #     try:
        #         extract_and_store_graph(d, allowed_nodes)
        #     except:
        #         traceback.print_exc()
    except Exception as e:  # 捕获所有类型的异常
        print(f"An error occurred: {e}")
        return jsonify({"error": "No files uploaded"}), 400
    else:
        # 如果没有异常发生，执行这里的代码
        print("Operation was successful.")
        return jsonify({"success": "OK"}), 200


#   如果当前模块是作为主入口来运行则条件成立
if __name__ == '__main__':
    app.run()
