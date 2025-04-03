import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv() 

if __name__ == "__main__":
    print('Ingesting...')
    loader = TextLoader('medium-blog.txt')
    document = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    embeddings = MistralAIEmbeddings(model="mistral-embed", api_key=os.environ['MISTRAL_API_KEY'])
    PineconeVectorStore.from_documents(texts, embeddings, index_name=os.environ['INDEX_NAME'])
    
    print('finish')