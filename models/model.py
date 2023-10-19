# 加载模型
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
import torch

# 中文embeddings
embeddings_zh = HuggingFaceEmbeddings(
    model_name='llm/text2vec-base-chinese',
    # model_kwargs={'device': 'cuda'}
)

# llama2
llama2_tokenizer = AutoTokenizer.from_pretrained(
    'llm/Llama-2-7b-chat-hf',
)
llama2_model = AutoModelForCausalLM.from_pretrained(
    "llm/Llama-2-7b-chat-hf",
)

llama2_pipe = pipeline(
    "text-generation",
    model=llama2_model,
    torch_dtype=torch.bfloat16,
    device_map='auto',
    max_new_tokens=512,
    do_sample=True,
    top_k=30,
    num_return_sequences=1,
    tokenizer=llama2_tokenizer,
    eos_token_id = llama2_tokenizer.eos_token_id
)

llama2 = HuggingFacePipeline(
    pipeline=llama2_pipe, 
    model_kwargs={'temperature':0}
)