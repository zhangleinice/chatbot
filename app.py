import openai
import os
import base64
import gradio as gr
from langchain import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryBufferMemory, ConversationBufferWindowMemory
from langchain.chat_models import ChatOpenAI
from chatbot import conversation_agent
from models.use import whisper_asr, llama2_7b, llama2_7b_predict, bark_tts

# openai.api_key = os.environ["OPENAI_API_KEY"]

# BufferWindowMemory： 仅支持过去几轮对话
memory = ConversationBufferWindowMemory(
    llm=llama2_7b, 
    max_token_limit=2048
)

conversation = ConversationChain(
    llm=llama2_7b,
    # llm=OpenAI(max_tokens=2048, temperature=0.5),
    memory=memory,
    verbose=True
)


def predict(input, history=[]):
    if input is None: 
        print("error: 输入不能为空")
        return {"error": "输入不能为空"}

    history.append(input)
    # res = conversation_agent.run(input)
    # print('res', res)

    # 使用 langchain.ConversationChain 后返回数据有问题
    # res = conversation.predict(input=input)

    res = llama2_7b_predict(input)
    # print('res',res)

    history.append(res)
    # print('history', history)

    # tts 生成语音文件
    bark_tts(res)

    with open("data/bark_out.wav", "rb") as audio_file:
        audio_data = base64.b64encode(audio_file.read()).decode("utf-8")

    # 更新audio
    audio_html = f"""<audio controls="controls" src="data:audio/wav;base64,{audio_data}"></audio>"""

    # responses: [('用户输入1', '聊天机器人回复1'), ('用户输入2', '聊天机器人回复2'), ...]
    responses = [(u, b) for u, b in zip(history[::2], history[1::2])]

    return responses, audio_html, history


# asr
def process_audio(audio, history=[]):
    text = whisper_asr(audio)
    return predict(text, history)


def create_demo():
    with gr.Blocks(css="#chatbot{height:800px} .overflow-y-auto{height:800px}") as demo:
        chatbot = gr.Chatbot(elem_id="chatbot")
        state = gr.State([])

        with gr.Row():
            txt = gr.Textbox(
                show_label=False, placeholder="Enter text and press enter", container=False)

        with gr.Row():
            input_audio = gr.Audio(source="microphone", type="filepath")

        with gr.Row():
            # 本地图片，音频文件转base64加载
            with open("data/avatar.png", "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')

            with open("data/bark_out.wav", "rb") as audio_file:
                audio_data = base64.b64encode(audio_file.read()).decode("utf-8")

            output_audio = gr.HTML(
                f'<audio controls="controls" src="data:audio/wav;base64,{audio_data}"></audio>')

            video = gr.HTML(
                f'<img src="data:image/png;base64,{image_data}" width="320" height="240" alt="avatar">')

        txt.submit(predict, [txt, state], [chatbot, output_audio, state])
        input_audio.change(process_audio, [input_audio, state], [chatbot, output_audio, state])

    return demo


def main():
    demo = create_demo()

    demo.launch(server_name="127.0.0.1", server_port=8888, share=True)


if __name__ == "__main__":
    main()
