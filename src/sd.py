import torch
import uuid
import os
from diffusers import DiffusionPipeline
from langchain.agents import initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain.agents.tools import Tool
from langchain import OpenAI
import argparse


def prompts(name, description):
    def decorator(func):
        func.name = name
        func.description = description
        return func

    return decorator


class Text2Image:
    def __init__(self, device):
        print(f"Initializing Text2Image to {device}")
        self.device = device
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32

        self.pipe = DiffusionPipeline.from_pretrained("Linaqruf/anything-v3.0")
        self.pipe.to(device)

        self.a_prompt = 'masterpiece, best quality, masterpiece, (1girl), <lora:LLCharV2-8:1>, (LLChar), (takami chika:1.1), (orange hair, red eyes), standing, (school uniform, pleated skirt, grey skirt, short sleeves, white shirt, red neckerchief), happy'

        self.n_prompt = '(EasyNegative:1.4)'

    @prompts(name="Generate Image From User Input Text",
             description="useful when you want to generate an image from a user input text and save it to a file. "
                         "like: generate an image of an object or something, or generate an image that includes some objects. "
                         "The input to this tool should be a string, representing the text used to generate image. ")
    def inference(self, text):
        image_filename = os.path.join('image', f"{str(uuid.uuid4())[:8]}.png")
        prompt = text + ', ' + self.a_prompt
        image = self.pipe(prompt, negative_prompt=self.n_prompt).images[0]
        image.save(image_filename)
        print(
            f"\nProcessed Text2Image, Input Text: {text}, Output Image: {image_filename}")
        return image_filename


class ConversationBot:
    def __init__(self, load_dict):
        self.models = {}
        self.tools = []

        # load models
        for class_name, device in load_dict.item():
            self.models[class_name] = globals()[class_name](device=device)

        for instance in self.models.values():
            for e in dir(instance):
                if e.startswith('inference'):
                    func = getattr(instance, e)
                    self.tools.append(
                        Tool(name=func.name, description=func.description, func=func))

        self.llm = OpenAI(temperature=0)
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True)

    # langchain ai agent
    def init_agent(self):
        self.agent = initialize_agent(
            self.tools,
            self.llm,
            agent="conversational-react-description",
            memory=self.memory,
            verbose=True
        )

    def run_text(self, text, state):
        return 'xxx'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # 指定了不同的工具类应该使用 CPU 还是 GPU

    parser.add_argument('--load', type=str,
                        default="ImageCaptioning_cuda:0,Text2Image_cuda:0")
    args = parser.parse_args()

    load_dict = {e.split('_')[0].strip(): e.split(
        '_')[1].strip() for e in args.load.split(',')}

    print('load_dict', load_dict)
    # load_dict {'ImageCaptioning': 'cuda:0', 'Text2Image': 'cuda:0'}

    bot = ConversationBot(load_dict=load_dict)
