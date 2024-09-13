import json
import re

# 读取 JSON 文件内容
with open(r"O:\LLama_Factory\Online_datasets\TransGPT-sft\TransGPT-sft - convert2.json", 'r', encoding='utf-8') as file:
    data = file.read()

# 使用正则表达式提取每个独立的 JSON 对象
json_objects = re.findall(r'{.*?}', data)

# 解析每个 JSON 对象
parsed_json_objects = [json.loads(obj) for obj in json_objects]

# 格式化输出
formatted_output = json.dumps(parsed_json_objects, indent=2, ensure_ascii=False)

# 打印结果
print(formatted_output)
