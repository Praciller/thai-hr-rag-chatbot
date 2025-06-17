from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# โหลด API Key จากไฟล์ .env
load_dotenv()

# กำหนดค่าตัวแปร
DB_PATH = "db"

# --- ตั้งค่า RAG Pipeline ---

# 1. โหลด LLM Model
#  vvv  นี่คือบรรทัดที่แก้ไข  vvv
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0)
#  ^^^  นี่คือบรรทัดที่แก้ไข  ^^^

# 2. โหลด Vector Store ที่สร้างไว้
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector_store = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
retriever = vector_store.as_retriever()

# 3. สร้าง Prompt Template
template = """
คุณคือผู้ช่วย AI ของบริษัท AI สยาม จำกัด มีหน้าที่ตอบคำถามเกี่ยวกับนโยบายภายในบริษัทเท่านั้น
โปรดใช้ข้อมูลจาก 'บริบท' ที่ให้มาเพื่อตอบคำถาม หากข้อมูลในบริบทไม่เกี่ยวข้องกับคำถาม ให้ตอบว่า "ฉันไม่สามารถให้ข้อมูลได้จากเอกสารที่มีอยู่"

บริบท: {context}

คำถาม: {question}

คำตอบ:
"""
prompt = PromptTemplate.from_template(template)

# 4. สร้าง RAG Chain ด้วย LangChain Expression Language (LCEL)
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# --- สร้าง FastAPI Application ---

app = FastAPI(
    title="HR Q&A System with RAG",
    description="API สำหรับระบบถาม-ตอบข้อมูล HR ด้วยเทคโนโลยี RAG และ Gemini"
)

# Pydantic model สำหรับ Request Body
class QueryRequest(BaseModel):
    question: str

# สร้าง API Endpoint
@app.post("/ask", summary="ถามคำถามเกี่ยวกับนโยบาย HR")
def ask_question(request: QueryRequest):
    """
    รับคำถามและตอบโดยอ้างอิงจากฐานข้อมูลเอกสารของบริษัท
    - **question**: คำถามที่ต้องการถาม (เช่น "ลาพักร้อนได้กี่วัน?")
    """
    answer = rag_chain.invoke(request.question)
    return {"answer": answer}

@app.get("/", summary="Root endpoint for health check")
def read_root():
    return {"message": "HR Q&A System is running."}