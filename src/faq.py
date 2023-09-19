
from langchain.document_loaders import TextLoader
from langchain.text_splitter import SpacyTextSplitter
from langchain.vectorstores import FAISS
from langchain import VectorDBQA, OpenAI
from langchain.agents import tool
from models.model import embeddings_zh

llm = OpenAI(temperature=0)

# 问答llmchain
loader = TextLoader('data/faq/ecommerce_faq.txt')

documents = loader.load()

text_splitter = SpacyTextSplitter(chunk_size=256, pipeline="zh_core_web_sm")

texts = text_splitter.split_documents(documents)


docsearch = FAISS.from_documents(texts, embeddings_zh)

faq_chain = VectorDBQA.from_chain_type(
    llm=llm,
    vectorstore=docsearch,
    verbose=True
)


@tool("FAQ")
def faq(intput: str) -> str:
    """"useful for when you need to answer questions about shopping policies, like return policy, shipping policy, etc."""
    return faq_chain.run(intput)
