# ใช้ Python 3.10-slim เป็น Base Image
FROM python:3.10-slim

# กำหนด Working Directory ภายใน Container
WORKDIR /app

# คัดลอกไฟล์ requirements.txt เข้าไปก่อน เพื่อใช้ประโยชน์จาก Docker cache
COPY requirements.txt .

# ติดตั้ง Dependencies
RUN pip install --no-cache-dir -r requirements.txt

# คัดลอกไฟล์ทั้งหมดในโปรเจกต์เข้าไปใน /app
# รวมถึงโฟลเดอร์ db ที่สร้างโดย ingest.py และโค้ด ui.py
COPY . .

# เปิด Port 8501 ซึ่งเป็น Port หลักของ Streamlit
EXPOSE 8501

# กำหนด Health Check เพื่อให้ Docker รู้ว่าแอปพร้อมใช้งาน
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# คำสั่งสำหรับรันแอปพลิเคชัน Streamlit
# --server.port 8501: กำหนด port
# --server.address 0.0.0.0: ทำให้เข้าถึงได้จากภายนอก Container
# --server.headless true: ปิดการแสดงข้อความที่ไม่จำเป็นใน log
CMD ["streamlit", "run", "ui.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]