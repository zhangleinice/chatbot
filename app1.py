import openai
import os
import base64
import gradio as gr
from langchain import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryBufferMemory, ConversationBufferWindowMemory
from chatbot import conversation_agent, callback, faq
# from models.use import whisper_asr, llama2_7b, llama2_7b_predict, bark_tts
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
import asyncio

openai.proxy = {
            "http": "http://127.0.0.1:7890",
            "https": "http://127.0.0.1:7890"
        }

async def wait_done(fn, event):
    try:
        await fn
    except Exception as e:
        print('error', e)
        # event.set()
    finally:
        event.set()



async def call_openai(question):

    # chain = faq(question)
    chain = conversation_agent.arun(question)
    print('conversation_agent', chain)
    # conversation_agent <coroutine object Chain.arun at 0x133319f10>

    # 直接使用openai模型
    # coroutine = wait_done(model.agenerate(messages=[[HumanMessage(content=question)]]), callback.done)

    # 使用llmchain
    coroutine = wait_done(chain, callback.done)

    task = asyncio.create_task(coroutine)

    print('coroutine', callback.aiter())
    # <async_generator object AsyncIteratorCallbackHandler.aiter at 0x11e4ffa40>

    async for token in callback.aiter():
        # print('token', token)
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
