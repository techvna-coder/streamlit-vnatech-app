# RR-V01.py (hoặc file chính Streamlit gọi tới)
import os, tempfile, time
import numpy as np
import streamlit as st
from openai import OpenAI
from drive_utils import get_drive_from_service_account, list_files_in_folder, download_file
from index_builder import (get_client, load_meta, save_meta, load_faiss, save_faiss,
                           add_document_to_index, md5_of_file, DIM, l2_normalize, embed_texts)

# === Đọc secrets
GSA_JSON = st.secrets["GOOGLE_SERVICE_ACCOUNT_JSON"]
FOLDER_ID = st.secrets["DRIVE_FOLDER_ID"]
OPENAI_KEY = st.secrets["OPENAI_API_KEY"]

# === Kết nối Drive & OpenAI
drive = get_drive_from_service_account(GSA_JSON)
client = get_client(OPENAI_KEY)

# === Tải meta & index
meta = load_meta("embeddings_meta.pkl")
index = load_faiss(meta.get("index_path", "faiss_index.bin"))

# === Liệt kê file trên Drive
files = list_files_in_folder(drive, FOLDER_ID)

# === Kiểm tra file mới/đổi
new_or_changed = []
for f in files:
    fid, title = f["id"], f["title"]
    # tải tạm về để tính md5 cục bộ (ổn định nhất). Nếu tin md5Checksum của Drive có sẵn thì dùng luôn.
    with tempfile.TemporaryDirectory() as td:
        local_path = os.path.join(td, title)
        download_file(drive, fid, local_path)
        md5_now = md5_of_file(local_path)

    if fid not in meta["files"] or meta["files"][fid]["md5"] != md5_now:
        new_or_changed.append((fid, title))

# === Xử lý (chỉ) file mới/đổi
added = 0
for fid, title in new_or_changed:
    with tempfile.TemporaryDirectory() as td:
        local_path = os.path.join(td, title)
        download_file(drive, fid, local_path)
        kind = "pdf" if title.lower().endswith(".pdf") else "pptx"
        added += add_document_to_index(client, index, meta, fid, title, local_path, kind)

# === Lưu lại index & meta nếu có thay đổi
if added > 0:
    save_faiss(index, meta.get("index_path", "faiss_index.bin"))
    save_meta(meta, "embeddings_meta.pkl")

# === HÀM TRUY HỒI
def search(query: str, top_k=10):
    # embed câu hỏi → chuẩn hoá → tìm trong FAISS
    qv = embed_texts(client, [query])
    qv = l2_normalize(qv)
    scores, idx = index.search(qv, top_k)  # FlatIP với vector đã chuẩn hoá = cosine sim
    idx = idx[0].tolist(); scores = scores[0].tolist()
    # ánh xạ ngược từ vị trí vector → file & chunk
    results = []
    offset = 0
    for fid, finfo in meta["files"].items():
        for (start, end) in finfo["ranges"]:
            for k, gid in enumerate(idx):
                if start <= gid < end:
                    results.append({
                        "file_id": fid,
                        "title": finfo["title"],
                        "global_id": gid,
                        "score": float(scores[k]),
                    })
    results = sorted(results, key=lambda r: -r["score"])[:top_k]
    return results
