import os
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import GooglePalmEmbeddings
from langchain_community.llms.google_palm import GooglePalm
from langchain_google_genai.llms import GoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv


load_dotenv()

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY

def get_pdf_text(pdf_doc):
    text = ''
    for pdf in pdf_doc:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunk(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunk):
    embeddings = GooglePalmEmbeddings()
    vector_store = FAISS.from_texts(text_chunk,embedding=embeddings)
    return vector_store

def get_conversational_chain(vector_store):
    llm = GoogleGenerativeAI(model="gemini-1.5-flash")
    memory = ConversationBufferMemory(memory_key='chat_history',return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm,retriever=vector_store.as_retriever(),memory=memory)
    return conversation_chain
