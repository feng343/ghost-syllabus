import pandas as pd
import pickle
import os
from openai import OpenAI


# 初始化 OpenAI 客户端（记得填入你的 API Key）
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# 读取 CSV 文件（用逗号分隔）
df = pd.read_csv("ss2025.csv")

# 拼接每门课程的描述为完整文本（可以自定义顺序和格式）
def row_to_text(row):
    return f"""Course: {row['TITLE']}
Semester: {row['YEAR']}
Lecturer(s): {row['LECTURER']}
Major: {row['MAJOR']}
Location: {row['LOCATION']}
Dates: {row['TIME']} ({row['START']} – {row['END']})
Credit: {row['CREDIT']}
Description: {row['DESCRIPTION']}
Contact: {row['CONTACT']}"""

# 生成每门课的文本描述列表，结构为 (title, lecturer, major, text)
texts = [(row['TITLE'], row['LECTURER'], row['MAJOR'], row_to_text(row)) for _, row in df.iterrows()]

with open("course_texts.pkl", "wb") as f:
    pickle.dump(texts, f)

print("✅ 成功生成课程文本数据，共处理课程数：", len(texts))