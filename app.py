from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek
from agent.llm_core.api_keys import OPENAI_API_KEY, DEEPSEEK_API_KEY

import os
os.environ["DEEPSEEK_API_KEY"] = DEEPSEEK_API_KEY

llm = ChatDeepSeek(
    model="deepseek-chat",
    max_tokens=1024,
    max_retries=2,
)

prompt = ChatPromptTemplate.from_messages(
    [("system", "你是世界级的技术专家"),
     ("user", "{question}")],
)

chain = prompt | llm

result = chain.invoke({"question": "如何使用langchain和openai创建一个简单的聊天机器人？"})

print(result)