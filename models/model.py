from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings

# 开源中文embeddings
embeddings_zh = HuggingFaceEmbeddings(
    model_name='shibing624/text2vec-base-chinese'
)

embeddings_en = OpenAIEmbeddings()
