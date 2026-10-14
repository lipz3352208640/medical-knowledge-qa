from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain_core.tools import tool
from datetime import datetime

model = ChatOllama(
    model="qwen3.5:latest",  # 替换为你想使用的Ollama模型名称
    temperature=0.7,  # 控制随机性，0.7比较平
    base_url="http://localhost:11434",
)


@tool
def get_current_time(input_text: str) -> str:
    """返回当前的日期和时间。当用户问'现在几点'、'今天日期'时调用此工具。"""
    date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"当前日期和时间: {date_time}"


agent = create_agent(
    model=model,
    tools=[get_current_time],  # 将工具传递给代理
    system_prompt="你是一个通用的查询助手，当用户发送查询请求的时候，你会根据用户的请求调用相应的工具来获取信息。当没有相应工具的时候，直接返回用户无法做出回答，不要编造答案",
)
result = agent.invoke({"messages": [{"role": "user", "content": "今天天气如何？"}]})
print(result["messages"][-1].content)
