import uvicorn
import os
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
import asyncio
import openai

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

    model = ChatOpenAI(
        openai_api_key= os.environ["OPENAI_API_KEY"],
        streaming=True, 
        callbacks=[callback]
    )

    # ChatOpenAI.agenerate
    coroutine = wait_done(model.agenerate(messages=[[HumanMessage(content=question)]]), callback.done)

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