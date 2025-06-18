# --- Magic patch for SQLite on Streamlit Cloud ---
# This must be the very first import in your app
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    print("✅ Successfully patched sqlite3")
except ImportError:
    print("⚠️ pysqlite3 not found, using default sqlite3. This may fail on Streamlit Cloud.")
# --------------------------------------------------

import streamlit as st
import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# โหลด API Key จาก Secrets ของ Streamlit หรือ .env ไฟล์
load_dotenv()

DATA_PATH = "data"
DB_PATH = "db"

# --- ฟังก์ชันสำหรับเตรียมฐานข้อมูลและ RAG Chain ---

def setup_vector_database():
    """
    ตรวจสอบว่ามี Vector DB อยู่หรือไม่ ถ้าไม่มีให้สร้างขึ้นมาใหม่แบบเงียบๆ
    """
    if not os.path.exists(DB_PATH):
        # st.write("ยังไม่มีฐานข้อมูล Vector DB, กำลังสร้างขึ้นใหม่...") # <-- ซ่อนข้อความนี้
        loader = DirectoryLoader(DATA_PATH, glob="*.txt")
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_documents(documents)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = Chroma.from_documents(
            documents=texts,
            embedding=embeddings,
            persist_directory=DB_PATH
        )
        # st.write("สร้างฐานข้อมูล Vector DB สำเร็จ!") # <-- ซ่อนข้อความนี้
    # else:
        # st.write("กำลังโหลดฐานข้อมูล Vector DB ที่มีอยู่...") # <-- ซ่อนข้อความนี้ด้วย

@st.cache_resource
def load_rag_chain():
    """
    โหลด RAG Chain ทั้งหมด (ทำครั้งเดียวและ cache ไว้)
    """
    setup_vector_database()
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    retriever = vector_store.as_retriever()
    template = """
    คุณคือผู้ช่วย AI ของบริษัท AI สยาม จำกัด มีหน้าที่ตอบคำถามเกี่ยวกับนโยบายภายในบริษัทเท่านั้น
    โปรดใช้ข้อมูลจาก 'บริบท' ที่ให้มาเพื่อตอบคำถาม หากข้อมูลในบริบทไม่เกี่ยวข้องกับคำถาม ให้ตอบว่า "ฉันไม่สามารถให้ข้อมูลได้จากเอกสารที่มีอยู่"
    บริบท: {context}
    คำถาม: {question}
    คำตอบ:
    """
    prompt = PromptTemplate.from_template(template)
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# --- ส่วนของหน้าเว็บ (Streamlit UI) ---
st.title("🤖 ระบบถาม-ตอบข้อมูล HR")
st.header("บริษัท AI สยาม จำกัด")

try:
    # แสดง Spinner ขณะโหลด Chain ในครั้งแรก ซึ่งรวมการสร้าง DB ด้วย
    with st.spinner("กำลังเตรียมระบบ..."):
        rag_chain = load_rag_chain()

    user_question = st.text_input("ถามคำถามเกี่ยวกับนโยบายบริษัทที่นี่:")
    if user_question:
        with st.spinner("กำลังค้นหาข้อมูลและสร้างคำตอบ..."):
            answer = rag_chain.invoke(user_question)
            st.write("### คำตอบ:")
            st.markdown(answer)
except Exception as e:
    st.error("เกิดข้อผิดพลาดในการทำงานของแอปพลิเคชัน:")
    st.error(e)