import openai
import os
import base64
import gradio as gr
from langchain import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryBufferMemory, ConversationBufferWindowMemory
from chatbot import conversation_agent, faq, recommend_product
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
# from models.use import whisper_asr, llama2_7b, llama2_7b_predict, bark_tts

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from langchain.callbacks import AsyncIteratorCallbackHandler
import asyncio
import openai
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage


# 流式输出需要添加代理
openai.proxy = {
            "http": "http://127.0.0.1:7890",
            "https": "http://127.0.0.1:7890"
        }

async def wait_done(fn, event):
    try:
        await fn
    except Exception as e:
        print('error', e)
        event.set()
    finally:
        event.set()


async def call_openai(question):

    callback = AsyncIteratorCallbackHandler()

    # chain = faq(question)
    # chain = conversation_agent.acall(input)


    # print(111111, model.agenerate(messages=[[HumanMessage(content=question)]]))
    # 111111 <coroutine object BaseChatModel.agenerate at 0x137586f20>

    # print(222222, faq(question))
    # 222222 <coroutine object Chain.acall at 0x1375489e0>

    # coroutine = wait_done(model.agenerate(messages=[[HumanMessage(content=question)]]), callback.done)

    coroutine = wait_done(faq(question), callback.done)

    task = asyncio.create_task(coroutine)

    print('coroutine', callback.aiter())
    # <async_generator object AsyncIteratorCallbackHandler.aiter at 0x11e4ffa40>

    async for token in callback.aiter():
        yield f"{token}"

    await task

app = FastAPI()

@app.get("/")
async def homepage():
    return FileResponse('static/index.html')

@app.post("/ask")
def ask(body: dict):
    return StreamingResponse(call_openai(body['question']), media_type="text/event-stream")

if __name__ == "__main__":
   uvicorn.run(host="127.0.0.1", port=8888, app=app)
