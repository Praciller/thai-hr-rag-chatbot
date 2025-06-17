import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

# โหลด API Key จากไฟล์ .env
load_dotenv()
print("API Key loaded.")

# กำหนดค่าตัวแปร
DATA_PATH = "data"
DB_PATH = "db"

def create_vector_database():
    """
    ฟังก์ชันสำหรับสร้าง Vector Database จากเอกสารในโฟลเดอร์ data
    """
    # 1. โหลดเอกสารทั้งหมดจากโฟลเดอร์
    loader = DirectoryLoader(DATA_PATH)
    documents = loader.load()
    print(f"Loaded {len(documents)} documents.")

    # 2. แบ่งเอกสารเป็นส่วนย่อยๆ (Chunking)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks.")

    # 3. สร้าง Embeddings และจัดเก็บใน ChromaDB
    print("Creating embeddings and storing in ChromaDB...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # สร้างและบันทึก Vector Store ลงดิสก์
    vector_store = Chroma.from_documents(
        documents=texts, 
        embedding=embeddings, 
        persist_directory=DB_PATH
    )
    
    print("------------------------------------------")
    print(f"Vector database created successfully at {DB_PATH}")
    print("------------------------------------------")


if __name__ == "__main__":
    create_vector_database()