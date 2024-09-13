# import pdfplumber
#
# # 打开PDF文件
# with pdfplumber.open(r"C:\Users\ASUS\Desktop\伊丽莎白·科顿.pdf") as pdf:
#     # 提取每一页的文本
#     for page in pdf.pages:
#         text = page.extract_text()
#         # 将文本保存到文件
#         with open(f"extracted_text_{page.page_number}.txt", "w", encoding="utf-8") as text_file:
#             text_file.write(text)
# # 现在你可以使用langchain的TextLoader来加载这些文本文件
import pdfplumber

# 打开PDF文件
with pdfplumber.open(r"C:\Users\ASUS\Desktop\600015_20240430_华夏银行2023年年度报告.pdf") as pdf:
    # 初始化一个空字符串用于存储所有页面的文本
    full_text = ""

    # 提取每一页的文本并累加到full_text字符串
    for page in pdf.pages:
        text = page.extract_text()
        full_text += text + "\n"  # 在每页文本后添加换行符

# 将累加的文本保存到一个文本文件中
with open("../600015_20240430_华夏银行2023年年度报告.txt", "w", encoding="utf-8") as text_file:
    text_file.write(full_text)
