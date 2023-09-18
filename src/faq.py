
from langchain.document_loaders import TextLoader
from langchain.text_splitter import SpacyTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain import VectorDBQA, OpenAI
from langchain.agents import tool

llm = OpenAI(temperature=0)

# 问答llmchain
loader = TextLoader('../data/faq/ecommerce_faq.txt')

documents = loader.load()

text_splitter = SpacyTextSplitter(chunk_size=256, pipeline="zh_core_web_sm")

texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()

docsearch = FAISS.from_documents(texts, embeddings)

faq_chain = VectorDBQA.from_chain_type(
    llm=llm,
    vectorstore=docsearch,
    verbose=True
)


@tool("FAQ")
def faq(intput: str) -> str:
    """"useful for when you need to answer questions about shopping policies, like return policy, shipping policy, etc."""
    return faq_chain.run(intput)
