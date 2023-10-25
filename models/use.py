import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import pipeline
from transformers import AutoProcessor, AutoModel
from IPython.display import Audio
import azure.cognitiveservices.speech as speechsdk
import torchaudio
import torch
import scipy

# Load Tokenizer and Model
whisper_processor = WhisperProcessor.from_pretrained("models/whisper-tiny")
whisper_model = WhisperForConditionalGeneration.from_pretrained("models/whisper-tiny")

llama2_7b_chat_tokenizer = AutoTokenizer.from_pretrained('models/Llama-2-7b-chat-hf')
llama2_7b_chat_model = AutoModelForCausalLM.from_pretrained("models/Llama-2-7b-chat-hf")

llama2_7b_tokenizer = AutoTokenizer.from_pretrained('models/Llama-2-7b-hf')
llama2_7b_model = AutoModelForCausalLM.from_pretrained("models/Llama-2-7b-hf")

# 中文合成效果不太好
bark_processor = AutoProcessor.from_pretrained("models/bark-small")
bark_model = AutoModel.from_pretrained("models/bark-small")

# embeddings
embeddings_zh = HuggingFaceEmbeddings(
    model_name='models/text2vec-base-chinese',
    # model_kwargs={'device': 'cuda'}
)


llama2_7b_chat_pipe = pipeline(
    "text-generation",
    model=llama2_7b_chat_model,
    torch_dtype=torch.bfloat16,
    device_map='auto',
    max_new_tokens=512,
    do_sample=True,
    top_k=30,
    num_return_sequences=1,
    tokenizer=llama2_7b_chat_tokenizer,
    eos_token_id = llama2_7b_chat_tokenizer.eos_token_id
)

llama2 = HuggingFacePipeline(
    pipeline=llama2_7b_chat_pipe, 
    model_kwargs={'temperature':0}
)


llama2_7b_pipe = pipeline(
    "text-generation",
    model=llama2_7b_model,
    torch_dtype=torch.bfloat16,
    device_map='auto',
    max_new_tokens=512,
    do_sample=True,
    top_k=30,
    num_return_sequences=1,
    tokenizer=llama2_7b_tokenizer,
    eos_token_id = llama2_7b_tokenizer.eos_token_id
)

llama2_7b = HuggingFacePipeline(
    pipeline=llama2_7b_pipe, 
    model_kwargs={'temperature':0}
)

# asr
def transcribe(audio, desired_sample_rate=16000):
    os.rename(audio, audio + '.wav')
    audio_file = open(audio + '.wav', "rb")

    # 加载保存的文件
    waveform, sample_rate = torchaudio.load(audio + '.wav', normalize=True)
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=desired_sample_rate)
    waveform = resampler(waveform)

    # encode
    input_features = whisper_processor(waveform[0], sampling_rate=desired_sample_rate, return_tensors="pt").input_features

    # generate
    predicted_ids = whisper_model.generate(input_features)

    # decode
    transcription = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)

    return transcription[0]

# nlp
def llama2_7b_predict(prompt):
    
    # encode
    input = llama2_7b_tokenizer(prompt, return_tensors="pt")

    # predict
    res = llama2_7b_model.generate(input_ids=input.input_ids, max_length=30)

    # decode
    generated_text = llama2_7b_tokenizer.decode(res[0], skip_special_tokens=True)

    return generated_text


# tts
def play_voice():

    inputs = bark_processor(
        text=["Hello, my name is Suno"],
        return_tensors="pt",
    )

    speech_values = bark_model.generate(**inputs, do_sample=True)

    sampling_rate = bark_model.generation_config.sample_rate

    return speech_values, sampling_rate

# speech_values, sampling_rate = play_voice()

# scipy.io.wavfile.write("data/bark_out.wav", rate=sampling_rate, data=speech_values.cpu().numpy().squeeze())

# Audio(speech_values.cpu().numpy().squeeze(), rate=sampling_rate)

# print('res', speech_values)





