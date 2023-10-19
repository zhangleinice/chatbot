# chatbot
客服聊天机器人


## Quick Start

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