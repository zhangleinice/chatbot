
# 需要升级python3.10以上
from langchain.llms import HuggingFacePipeline
from langchain import OpenAI

llama2 = HuggingFacePipeline.from_model_id(
    model_id="meta-llama/Llama-2-7b-chat-hf",
    task="text-generation",
    model_kwargs={"max_length": 64},
    device=0,
    batch_size=4
)

gpt = OpenAI(temperature=0)
