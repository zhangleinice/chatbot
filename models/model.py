# 加载模型
import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import pipeline
import torchaudio
import torch


# embeddings
embeddings_zh = HuggingFaceEmbeddings(
    model_name='llm/text2vec-base-chinese',
    # model_kwargs={'device': 'cuda'}
)

# nlp
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

# asr
processor = WhisperProcessor.from_pretrained("llm/whisper-tiny")
model = WhisperForConditionalGeneration.from_pretrained("llm/whisper-tiny")

def transcribe(audio, desired_sample_rate=16000):
    os.rename(audio, audio + '.wav')
    audio_file = open(audio + '.wav', "rb")

    # 加载保存的文件
    waveform, sample_rate = torchaudio.load(audio + '.wav', normalize=True)
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=desired_sample_rate)
    waveform = resampler(waveform)

    # 使用处理器编码音频数据
    input_features = processor(waveform[0], sampling_rate=desired_sample_rate, return_tensors="pt").input_features

    # 执行生成操作
    predicted_ids = model.generate(input_features)

    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    return transcription


