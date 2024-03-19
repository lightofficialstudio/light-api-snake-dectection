# ใช้ภาพพื้นฐานของ Ubuntu
FROM ubuntu:20.04

# ติดตั้ง python และ pip
RUN apt-get update && apt-get install -y python3-pip python3-dev libgl1-mesa-dev

# อัพเกรด pip และติดตั้ง wheel ซึ่งจำเป็นสำหรับบางไลบรารีที่จะคอมไพล์จาก source
RUN pip3 install --upgrade pip && pip3 install wheel

# คัดลอกไฟล์โปรเจคทั้งหมดไปยัง container
COPY . /app

# ตั้งค่า working directory ให้เป็น /app
WORKDIR /app

# ติดตั้งไลบรารีจาก requirements.txt
RUN pip3 install -r requirements.txt

# เปิด port 8080 สำหรับการเชื่อมต่อเข้าสู่ container
EXPOSE 8080

# ตั้งค่าคำสั่งสำหรับการรันแอปผ่าน Gunicorn
CMD ["gunicorn", "--workers=4", "--bind", "0.0.0.0:8080", "app:app"]
