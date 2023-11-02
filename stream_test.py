# 流式输出
# 非LLMChain： llm._stream
# LLMChain


# from langchain.chat_models import ChatOpenAI
# from langchain.schema import ChatMessage, AIMessage
# from langchain.llms import OpenAI


# llm = ChatOpenAI(temperature=0)
# messages= [ChatMessage(role="user",content="你好")]
# result = llm._stream(messages) # 这里会得到生成器
# print('result',result)
# text = ""
    # generator object可以直接使用for遍历
# for i in result:
#       text = text+i.message.content
#       print('text', text)

# result <generator object ChatOpenAI._stream at 0x116b66700>
# text 
# text 你
# text 你好
# text 你好！
# text 你好！有

# 未跑通
import gradio as gr
import asyncio
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.callbacks.base import AsyncCallbackHandler, BaseCallbackHandler


import os
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.document_loaders.csv_loader import CSVLoader
# from models.use import  embeddings_zh, llama2_7b, llama2_7b_chat
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler



embeddings = OpenAIEmbeddings()

loader = TextLoader("data/faq/ecommerce_faq.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

docsearch = Chroma.from_documents(texts, embeddings)

callback = AsyncIteratorCallbackHandler()

llm = OpenAI(
    openai_api_key= os.environ["OPENAI_API_KEY"],
    temperature=0, 
    streaming=True, 
    callbacks=[callback]
)

faq_chain = RetrievalQA.from_chain_type(
    llm, 
    chain_type="stuff", 
    retriever=docsearch.as_retriever()
)


async def f():
   res =  faq_chain.acall('如何更改帐户信息')
   print('res', res)
#    res <coroutine object Chain.acall at 0x13722e980>

# 将一个协程对象 res 添加到事件循环的任务队列中执行
   asyncio.create_task(res)

   print(asyncio.all_tasks())

   print(callback.aiter())

#    text = ""
# #    coroutine object 使用async遍历
# #    异步回调中逐步获取数据
    
   async for token in callback.aiter():
       print('token', token)
#     #    text = text+token
#     #    yield text



asyncio.run(f())

