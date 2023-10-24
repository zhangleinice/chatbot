# 下载模型到这个文件夹下
```
克隆模型 git lfs clone
git clone https://huggingface.co/suno/bark-small
git clone https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
git clone https://huggingface.co/meta-llama/Llama-2-7b-hf
git clone https://huggingface.co/shibing624/text2vec-base-chinese
git clone https://huggingface.co/openai/whisper-tiny

大文件下载慢可以使用 wget下载
如：wget https://huggingface.co/GanymedeNil/text2vec-large-chinese/resolve/main/pytorch_model.bin

llama-2-7b需要个人令牌方式验证
git clone https://x-access-token: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@huggingface.co/meta-llama/Llama-2-7b-hf

wget --header="Authorization: Bearer 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" https://huggingface.co/meta-llama/Llama-2-7b-hf/resolve/main/pytorch_model-00001-of-00002.bin

替换 “your-token” 为你的huggingface个人访问令牌

```
