# app.py
import os
import json
import pickle
import streamlit as st
import pandas as pd

from PyPDF2 import PdfReader
from pptx import Presentation
from openai import OpenAI

from drive_utils import get_drive_from_secrets, list_files_in_folder, download_file

# --------------------------
# Cấu hình trang
# --------------------------
st.set_page_config(page_title="VNA Tech: Hỗ trợ tra cứu thông tin từ Google Drive", layout="wide")
st.title("✈️ VNA Tech: Hỗ trợ tra cứu thông tin tài liệu từ Google Drive")

# --------------------------
# Đọc secrets
# --------------------------
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
DRIVE_FOLDER_ID = st.secrets["DRIVE_FOLDER_ID"]
GSA_RAW = st.secrets["GOOGLE_SERVICE_ACCOUNT_JSON"]

client = OpenAI(api_key=OPENAI_API_KEY)

# --------------------------
# Hàm đọc file PDF/PPTX
# --------------------------
def read_pdf(path):
    reader = PdfReader(path)
    return "\n".join([page.extract_text() or "" for page in reader.pages])

def read_pptx(path):
    prs = Presentation(path)
    texts = []
    for slide in prs.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                texts.append(shape.text)
    return "\n".join(texts)

# --------------------------
# Chunking text
# --------------------------
def chunk_text(text, chunk_size=1000, overlap=200):
    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# --------------------------
# Tạo embedding & cache
# --------------------------
def build_or_load_embeddings(file_id, file_name, local_path):
    cache_name = f"{file_id}.pkl"
    if os.path.exists(cache_name):
        with open(cache_name, "rb") as f:
            return pickle.load(f)

    # đọc nội dung
    if file_name.endswith(".pdf"):
        text = read_pdf(local_path)
    elif file_name.endswith(".pptx"):
        text = read_pptx(local_path)
    else:
        return None

    chunks = chunk_text(text)

    embeddings = []
    for ch in chunks:
        if ch.strip():
            emb = client.embeddings.create(model="text-embedding-3-small", input=ch).data[0].embedding
            embeddings.append((ch, emb))

    with open(cache_name, "wb") as f:
        pickle.dump(embeddings, f)

    return embeddings

# --------------------------
# Tìm top-k chunk
# --------------------------
from numpy import dot
from numpy.linalg import norm

def search_chunks(question, embeddings, top_k=10):
    q_emb = client.embeddings.create(model="text-embedding-3-small", input=question).data[0].embedding

    scored = [(ch, float(dot(q_emb, emb) / (norm(q_emb) * norm(emb)))) for ch, emb in embeddings]
    scored = sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]
    return [ch for ch, _ in scored]

# --------------------------
# Giao diện Streamlit
# --------------------------
# Kết nối Google Drive
drive = get_drive_from_secrets(GSA_RAW)

files = list_files_in_folder(drive, DRIVE_FOLDER_ID)
file_map = {f["title"]: f for f in files}

st.sidebar.header("📂 Chọn tài liệu")
file_choice = st.sidebar.selectbox("File trên Google Drive:", list(file_map.keys()) if file_map else [])

question = st.text_input("Đặt câu hỏi:")
if st.button("Tìm câu trả lời") and file_choice and question:
    file_meta = file_map[file_choice]
    local_path = f"tmp_{file_meta['id']}.pdf" if file_choice.endswith(".pdf") else f"tmp_{file_meta['id']}.pptx"

    if not os.path.exists(local_path):
        download_file(drive, file_meta["id"], local_path)

    embeddings = build_or_load_embeddings(file_meta["id"], file_choice, local_path)
    top_chunks = search_chunks(question, embeddings, top_k=10)

    # Tạo câu trả lời
    context = "\n\n".join(top_chunks)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Bạn là trợ lý kỹ thuật, hãy trả lời dựa trên nội dung tài liệu."},
            {"role": "user", "content": f"Câu hỏi: {question}\n\nDữ liệu liên quan:\n{context}"}
        ]
    )
    st.subheader("💡 Trả lời")
    st.write(response.choices[0].message.content)
