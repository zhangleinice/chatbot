
import openai
import os
import base64
import gradio as gr
from langchain import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chat_models import ChatOpenAI
from src.chatbot import conversation_agent
from models.model import llama2, transcribe

openai.api_key = os.environ["OPENAI_API_KEY"]

memory = ConversationSummaryBufferMemory(
    llm=llama2, 
    max_token_limit=2048
)

conversation = ConversationChain(
    llm=llama2,
    # llm=OpenAI(max_tokens=2048, temperature=0.5),
    memory=memory,
)

def predict(input, history=[]):
    
    history.append(input)
    print('history', history)
    # input输入有符号时报错
    # ai agent
    # res = conversation_agent.run(input)
    # print('res', res)
    # llama2直接返回
    res = conversation.predict(input=input)
    print('res',res)
    history.append(res)
    # print('history', history)
    
    responses = [(u, b) for u, b in zip(history[::2], history[1::2])]
    # responses: [('用户输入1', '聊天机器人回复1'), ('用户输入2', '聊天机器人回复2'), ...]

    return responses, history


# asr
def process_audio(audio, history=[]):

    text = transcribe(audio)
    history.append(text)
    # 语音已经识别出来了，llama2返回有问题
    # text [' Hello.'] 可能是空格造成的
    print('text', text)
    print('history', history)
    return predict('hello', history)


def create_demo():
    with gr.Blocks(css="#chatbot{height:800px} .overflow-y-auto{height:800px}") as demo:
        chatbot = gr.Chatbot(elem_id="chatbot")
        state = gr.State([])

        with gr.Row():
            txt = gr.Textbox(
                show_label=False, placeholder="Enter text and press enter", container=False)

        with gr.Row():
            audio = gr.Audio(source="microphone", type="filepath")

        with gr.Row():
            # 本地图片转base64加载
            with open("data/avatar.png", "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
            video = gr.HTML(
                f'<img src="data:image/png;base64,{image_data}" width="320" height="240" alt="avatar">', live=False)

        txt.submit(predict, [txt, state], [chatbot, state])
        audio.change(process_audio, [audio, state], [chatbot, state])

    return demo
