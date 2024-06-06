from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


llm = Ollama(model='snickers8523/llama3-taide-lx-8b-chat-alpha1-q4-0:latest')
embeddings = OllamaEmbeddings()

import pandas as pd
df = pd.read_csv('./交大地圖.csv')
s = df['output']

docs = [Document(page_content=e) for e in s]


vectordb = FAISS.from_documents(docs, embeddings)
retriever = vectordb.as_retriever()

prompt = ChatPromptTemplate.from_messages([
    ('system', 'I want you to act as a Text Adventure Game. Below are the game rules, you will strictly follow them at all times: You must answer in Chinese. First, show the location where the player inputed, then list where the user will arrive if they go forward, backward, left, and right, according to the context below:\n\n{context}.'),
    ('user', 'Question: {input}'),
])
document_chain = create_stuff_documents_chain(llm, prompt)

retrieval_chain = create_retrieval_chain(retriever, document_chain)

context = []
input_text = input('>>> ')
while input_text.lower() != 'bye':
    response = retrieval_chain.invoke({
        'input': input_text,
        'context': context
    })
    print(response['answer'])
    context = response['context']
    input_text = input('>>> ')
