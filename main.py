import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_pinecone import PineconeVectorStore

from langchain import hub

#we are using stuff_document chain but we may want to avoid it
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

load_dotenv()
import time

if __name__=="__main__":
    
    print('retrieving...') 
    query = "What is pinecone in machine learning?"

    embeddings = MistralAIEmbeddings(model="mistral-embed", api_key=os.environ['MISTRAL_API_KEY'])
    llm = ChatMistralAI(api_key=os.environ['MISTRAL_API_KEY'])
    chain = PromptTemplate.from_template(template=query) | llm
    result = chain.invoke(input={})
    print(result.content)
    
    print('/n')
    print('With retrieval...')
    
    index_name = os.environ['INDEX_NAME']
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm, prompt=retrieval_qa_chat_prompt)
    retrieval_chain = create_retrieval_chain(
        retriever = vectorstore.as_retriever(),
        combine_docs_chain=combine_docs_chain
    )
    time.sleep(5)

    result = retrieval_chain.invoke(input={"input": query})
    print(result)
    