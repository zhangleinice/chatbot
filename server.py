import uvicorn
import os

from typing import AsyncIterable, Awaitable
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

app = FastAPI()

@app.get("/")
async def homepage():
    return FileResponse('static/index.html')

if __name__ == "__main__":
   uvicorn.run(host="127.0.0.1", port=8888, app=app)