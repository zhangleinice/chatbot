# chatbot
客服聊天机器人


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
export OPENAI_API_KEY={Your_Private_Openai_Key}

# huggingface-cli login
huggingface-cli login --token {Your_Huggingface_Token}

```