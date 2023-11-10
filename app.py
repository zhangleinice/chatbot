import openai
import os
import base64
import gradio as gr
from langchain import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryBufferMemory, ConversationBufferWindowMemory
from langchain.chat_models import ChatOpenAI
from chatbot import conversation_agent, faq
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
# from models.use import whisper_asr, llama2_7b, llama2_7b_predict, bark_tts

llm = OpenAI(
    openai_api_key= os.environ["OPENAI_API_KEY"],
    temperature=0, 
    streaming=True, 
    callbacks=[StreamingStdOutCallbackHandler()]
)

memory = ConversationBufferWindowMemory(
    llm=llm, 
    max_token_limit=2048
)

conversation = ConversationChain(
    llm=llm,
    # llm=OpenAI(max_tokens=2048, temperature=0.5),
    memory=memory,
    verbose=True
)

# def text2speech(text):
#     _, _, file_path = bark_tts(text)

#     with open(file_path, "rb") as audio_file:
#         audio_data = base64.b64encode(audio_file.read()).decode("utf-8")

#     audio_html = f"""<audio controls="controls" src="data:audio/wav;base64,{audio_data}"></audio>"""
    
#     return audio_html


def predict(input, history=[]):
    if input is None: 
        print("error: 输入不能为空")
        return {"error": "输入不能为空"}

    history.append(input)
    res = faq('如何更改帐户信息')
    # res = conversation_agent.run(input)
    print('res', res)

    # 使用 langchain.ConversationChain 后返回数据有问题
    # res = conversation.predict(input=input)

    # res = llama2_7b_predict(input)
    # print('res',res)

    history.append(res)

    # print('history', history)

    # audio_html = text2speech(res)

    # responses: [('用户输入1', '聊天机器人回复1'), ('用户输入2', '聊天机器人回复2'), ...]
    responses = [(u, b) for u, b in zip(history[::2], history[1::2])]

    return responses, history


# def speech2text(audio, history=[]):
#     text = whisper_asr(audio)
#     return predict(text, history)


def create_demo():
    with gr.Blocks(css="#chatbot{height:800px} .overflow-y-auto{height:800px}") as demo:
        chatbot = gr.Chatbot(elem_id="chatbot")
        state = gr.State([])

        with gr.Row():
            txt = gr.Textbox(
                show_label=False, placeholder="Enter text and press enter", container=False)

        # with gr.Row():
        #     input_audio = gr.Audio(source="microphone", type="filepath")

        # with gr.Row():
        #     # 本地图片，音频文件转base64加载
        #     with open("static/avatar.png", "rb") as image_file:
        #         image_data = base64.b64encode(image_file.read()).decode('utf-8')

        #     with open("static/bark_out.wav", "rb") as audio_file:
        #         audio_data = base64.b64encode(audio_file.read()).decode("utf-8")

        #     output_audio = gr.HTML(
        #         f'<audio controls="controls" src="data:audio/wav;base64,{audio_data}"></audio>')

        #     video = gr.HTML(
        #         f'<img src="data:image/png;base64,{image_data}" width="320" height="240" alt="avatar">')

        txt.submit(predict, [txt, state], [chatbot, state])
        # input_audio.change(speech2text, [input_audio, state], [chatbot, output_audio, state])

    return demo


def main():
    demo = create_demo()

    # 流式输出
    demo.queue()

    demo.launch(server_name="127.0.0.1", server_port=8888, share=True)


if __name__ == "__main__":
    main()
