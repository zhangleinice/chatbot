import json
import re, os
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import initialize_agent, tool
from langchain.document_loaders.csv_loader import CSVLoader
# from models.use import  embeddings_zh, llama2_7b, llama2_7b_chat
from langchain.callbacks import AsyncIteratorCallbackHandler, StreamlitCallbackHandler
import openai
from langchain.chat_models import ChatOpenAI

from langchain.callbacks.streaming_stdout_final_only import (
    FinalStreamingStdOutCallbackHandler,
)
from langchain.agents.agent_types import AgentType
from dotenv import load_dotenv

# load_dotenv()

# 异步请求添加代理
openai.proxy = {
            "http": "http://127.0.0.1:7890",
            "https": "http://127.0.0.1:7890"
        }

# callback必须共用一个
callback = AsyncIteratorCallbackHandler()

# langchain 的 callbacks进行流式处理
llm = OpenAI(
    openai_api_key= os.environ["OPENAI_API_KEY"],
    temperature=0, 
    streaming=True, 
    callbacks=[callback]
)
embeddings = OpenAIEmbeddings()

# 切换开源模型
# llm = llama2_7b
# embeddings = embeddings_zh

# 问答
loader = TextLoader("static/faq/ecommerce_faq.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

docsearch = Chroma.from_documents(texts, embeddings)

faq_chain = RetrievalQA.from_chain_type(
    llm, 
    chain_type="stuff", 
    retriever=docsearch.as_retriever(),
)

@tool("FAQ")
def faq(input):
    """"useful for when you need to answer questions about shopping policies, like return policy, shipping policy, etc."""
    print('faq input', input)
    return faq_chain.arun(input)

#  商品推荐 llMchain
product_loader = CSVLoader(file_path='static/faq/ecommerce_products.csv')
product_documents = product_loader.load()

product_text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

product_texts = text_splitter.split_documents(product_documents)

product_search = Chroma.from_documents(product_texts, embeddings)

product_chain = RetrievalQA.from_chain_type(
    llm, 
    chain_type="stuff", 
    retriever=product_search.as_retriever()
)

@tool("Recommend Product")
def recommend_product(input: str) -> str:
    """"useful for when you need to search and recommend products and recommend it to the user"""
    return product_chain.acall(input)

# 订单查询
ORDER_1 = "20230101ABC"
ORDER_2 = "20230101EFG"

ORDER_1_DETAIL = {
    "order_number": ORDER_1,
    "status": "已发货",
    "shipping_date" : "2023-01-03",
    "estimated_delivered_date": "2023-01-05",
} 

ORDER_2_DETAIL = {
    "order_number": ORDER_2,
    "status": "未发货",
    "shipping_date" : None,
    "estimated_delivered_date": None,
}

answer_order_info = PromptTemplate(
    template="请把下面的订单信息回复给用户： \n\n {order}?", input_variables=["order"]
)
answer_order_llm = LLMChain(llm=llm,  prompt=answer_order_info)

@tool("Search Order", return_direct=True)
def search_order(input:str)->str:

    # 加提示语：找不到订单的时候，防止重复调用 OpenAI 的思考策略，来敷衍用户
    """useful for when you need to answer questions about customers orders"""
    pattern = r"\d+[A-Z]+"
    match = re.search(pattern, input)

    order_number = input
    if match:
        order_number = match.group(0)
    else:
        return "请问您的订单号是多少？"
    if order_number == ORDER_1:        
        return answer_order_llm.run(json.dumps(ORDER_1_DETAIL))
    elif order_number == ORDER_2:
        return answer_order_llm.run(json.dumps(ORDER_2_DETAIL))
    else:
        return f"对不起，根据{input}没有找到您的订单"

# res = faq('如何更改帐户信息')
# print('res', res)
# <coroutine object Chain.acall at 0x13755e820>


tools = [
    faq,
    search_order,
    recommend_product,
]

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

conversation_agent = initialize_agent(
    tools, 
    # 用OpenAI，不要用ChatOpenAI，会报错 Could not parse LLM output: `{text}`
    llm, 
    # agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    agent="conversational-react-description", 
    memory=memory, 
    verbose=True,
    streaming=True,
)

# question3 = "写一首关于大海的诗歌"
# answer3 = conversation_agent.run('你好')
# print('res', answer3)

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
