# download the model to this file

```
# clone the repo:  git lfs clone
git clone https://huggingface.co/suno/bark-small
git clone https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
git clone https://huggingface.co/meta-llama/Llama-2-7b-hf
git clone https://huggingface.co/shibing624/text2vec-base-chinese
git clone https://huggingface.co/openai/whisper-tiny

# use wget to download large files
eg：wget https://huggingface.co/GanymedeNil/text2vec-large-chinese/resolve/main/pytorch_model.bin

# llama-2-7b requires personal token verification

git clone https://x-access-token:your-token@huggingface.co/meta-llama/Llama-2-7b-hf

wget --header="Authorization: Bearer your-token" https://huggingface.co/meta-llama/Llama-2-7b-hf/resolve/main/pytorch_model-00001-of-00002.bin

Replace "your-token" with your huggingface personal access token .

```
