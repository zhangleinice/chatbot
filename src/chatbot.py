from src.faq import faq
from src.order import search_order
from src.prd_recommend import recommend_product
from langchain.agents import initialize_agent
from langchain.memory import ConversationBufferMemory
# from langchain.chat_models import ChatOpenAI
from models.model import llama2

tools = [
    search_order,
    recommend_product,
    faq
]

# chatllm = ChatOpenAI(temperature=0)

memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True)

# llama2 进行 ai agent判断
conversation_agent = initialize_agent(
    tools,
    llama2,
    agent="conversational-react-description",
    memory=memory,
    verbose=True
)


# zero-shot-react-description：零样本
# agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

# question = "请问你们的货，能送到三亚吗？大概需要几天？"
# result = conversation_agent.run(question)
# print(result)


# question = "我想买一件衣服，想要在春天去公园穿，但是不知道哪个款式好看，你能帮我推荐一下吗？"
# answer = conversation_agent.run(question)
# print(answer)


# question = "我有一张订单，订单号是 2022ABCDE，一直没有收到，能麻烦帮我查一下吗？"
# answer = conversation_agent.run(question)
# print(answer)


# question1 = "我有一张订单，一直没有收到，能麻烦帮我查一下吗？"
# answer1 = conversation_agent.run(question1)
# print(answer1)


# question2 = "我的订单号是20230101ABC"
# answer2 = conversation_agent.run(question2)
# print(answer2)


# question3 = "你们的退货政策是怎么样的？"
# answer3 = conversation_agent.run(question3)
# print(answer3)
