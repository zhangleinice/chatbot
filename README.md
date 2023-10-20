# chatbot
llama2开源聊天机器人

## 快速开始

```
# 下载仓库
git clone https://github.com/zhangleinice/chatbot.git
cd chatbot/

# 创建,激活虚拟环境
python3 -m venv venv
source venv/bin/activate

#  安装依赖
pip install -r requirements.txt

# openai环境变量
export OPENAI_API_KEY={Your_Private_Openai_Key}

# 登录 huggingface
huggingface-cli login --token {Your_Huggingface_Token}

# 下载模型到llm文件夹下
cd llm/
git lfs clone https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
git clone https://huggingface.co/shibing624/text2vec-base-chinese

# 大文件使用wget下载
wget https://huggingface.co/GanymedeNil/text2vec-large-chinese/resolve/main/pytorch_model.bin

# 启动
python main.py

```

## 注意事项
1. 下载模型到本地或服务器下，直接在gpu缓存中运行，容易导致显存不足  
2. 下载模型使用 git lfs + wget  
3. 使用llama2模型，需要到meta ai网站申请权限，https://ai.meta.com/resources/models-and-libraries/llama-downloads/  
4. 登录huggingface，添加一个token，llama2需要登录校验  