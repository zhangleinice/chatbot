
import openai
import os
import gradio as gr
from langchain import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.agents.tools import Tool
from src.chatbot import conversation_agent

openai.api_key = os.environ["OPENAI_API_KEY"]

memory = ConversationSummaryBufferMemory(
    llm=ChatOpenAI(), max_token_limit=2048
)

conversation = ConversationChain(
    llm=OpenAI(max_tokens=2048, temperature=0.5),
    memory=memory,
)


def conversation_history(input, history=[]):
    history.append(input)
    # input输入有符号时报错
    res = conversation_agent.run(input)
    # response = conversation.predict(input=input)
    history.append(res)
    responses = [(u, b) for u, b in zip(history[::2], history[1::2])]
    return responses, history


def create_demo():
    with gr.Blocks(css="#chatbot{height:800px} .overflow-y-auto{height:800px}") as demo:
        chatbot = gr.Chatbot(elem_id="chatbot")
        state = gr.State([])

        with gr.Row():
            txt = gr.Textbox(
                show_label=False, placeholder="Enter text and press enter", container=False)

        txt.submit(conversation_history, [txt, state], [chatbot, state])

    return demo
