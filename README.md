# chatbot
llama2开源聊天数字人

## 简介

```
ASR: 使用 openai/whisper-tiny 语音识别 
NLP: 使用 meta-llama/Llama-2-7b-hf 模型推理  
TTS: 使用 suno/bark-small 语音合成  
本地知识库问答: langchain RAG 
AI Agent: langchain agent 

```


## Quick Start

```
# clone the repo
git clone https://github.com/zhangleinice/chatbot.git

# Go to directory
cd chatbot/

# create a new environment
python3 -m venv venv

# activate the new environment
source venv/bin/activate

#  prepare the basic environments
pip install -r requirements.txt

# prepare your private OpenAI key (for Linux)
# export OPENAI_API_KEY={Your_Private_Openai_Key}

# login huggingface
huggingface-cli login --token {Your_Huggingface_Token}

# download models
cd models/
git clone https://huggingface.co/suno/bark-small
git clone https://huggingface.co/meta-llama/Llama-2-7b-hf
git clone https://huggingface.co/shibing624/text2vec-base-chinese
git clone https://huggingface.co/openai/whisper-tiny 

# run
python app.py

```
 