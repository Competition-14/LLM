import json
import re

# 输入的字符串
data = ''''''

# 1. 使用正则表达式将每个对象分离出来
json_objects = re.findall(r'{.*?}', data)
# print("-------------re.findall--------------")
# print(json_objects)
print("-------------------------------------")
# 2. 解析每个 JSON 对象
parsed_json_objects = [json.loads(obj) for obj in json_objects]

# 3. 格式化输出
formatted_output = json.dumps(parsed_json_objects, indent=2, ensure_ascii=False)

# 打印结果
print(formatted_output)
print("-------------------------------------")