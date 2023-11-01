# 流式输出
# 直接用llm： llm._stream
# 使用LLMChain


# from langchain.chat_models import ChatOpenAI
# from langchain.schema import ChatMessage, AIMessage
# from langchain.llms import OpenAI


# llm = ChatOpenAI(temperature=0)
# messages= [ChatMessage(role="user",content="你好")]
# result = llm._stream(messages) # 这里会得到生成器
# print('result',result)
# text = ""
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

async def f():
   callback = AsyncIteratorCallbackHandler()

   llm = ChatOpenAI(streaming=True,callbacks=[callback])

   coro = llm.apredict("写一首关于大海的文章，100字以内")  # 这里如果是 LLMChain的话 可以 换成  chain.acall()

   asyncio.create_task(coro)

   text = ""
   async for token in callback.aiter():
       text = text+token
       yield gr.TextArea.update(value=text)

with gr.Blocks() as demo:
    with gr.Column():
         摘要汇总 = gr.TextArea(value="",label="摘要总结",)
         bn = gr.Button("触发", variant="primary")
    bn.click(f,[],[摘要汇总])

demo.queue().launch(share=False, inbrowser=False, server_name="127.0.0.1", server_port=8001)


# import os
# from langchain.chains import RetrievalQA
# from langchain.document_loaders import TextLoader
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.llms import OpenAI
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.vectorstores import Chroma
# from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain
# from langchain.document_loaders.csv_loader import CSVLoader
# # from models.use import  embeddings_zh, llama2_7b, llama2_7b_chat
# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# # 流式传输
# # 使用langchain 的 callbacks进行流式处理
# llm = OpenAI(
#     openai_api_key= os.environ["OPENAI_API_KEY"],
#     temperature=0, 
#     streaming=True, 
#     callbacks=[StreamingStdOutCallbackHandler()]
# )
# embeddings = OpenAIEmbeddings()

# # 切换开源模型
# # llm = llama2_7b
# # embeddings = embeddings_zh

# # 问答
# loader = TextLoader("data/faq/ecommerce_faq.txt")
# documents = loader.load()
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# texts = text_splitter.split_documents(documents)

# docsearch = Chroma.from_documents(texts, embeddings)

# faq_chain = RetrievalQA.from_chain_type(
#     llm, 
#     chain_type="stuff", 
#     retriever=docsearch.as_retriever()
# )

# # res = faq_chain.acall('如何更改帐户信息')
# res = faq_chain.run('如何更改帐户信息')
# # print('res', res)
# #  登录您的帐户，然后点击“我的帐户”以更改个人信息、收货地址等。text  
