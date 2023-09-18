
from langchain.document_loaders import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain import VectorDBQA, OpenAI
from langchain.agents import tool

llm = OpenAI(temperature=0)

product_loader = CSVLoader('../data/faq/ecommerce_products.csv')

product_documents = product_loader.load()

product_text_splitter = CharacterTextSplitter(chunk_size=1024, separator="\n")

product_texts = product_text_splitter.split_documents(product_documents)

product_search = FAISS.from_documents(product_texts, OpenAIEmbeddings())

product_chain = VectorDBQA.from_chain_type(
    llm=llm,
    vectorstore=product_search,
    verbose=True
)


@tool("Recommend Product")
def recommend_product(input: str) -> str:
    """"useful for when you need to search and recommend products and recommend it to the user"""
    return product_chain.run(input)
