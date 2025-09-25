import streamlit as st
from rag_index import build_or_load, retrieve
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
st.set_page_config(page_title="VNA TECH SUPPORT", page_icon="📚", layout="wide")
st.title("📚 VNA TECH SUPPORT Q&A")

uploaded = st.file_uploader("Tải file PDF/PPTX", type=["pdf","pptx"], accept_multiple_files=True)
query = st.text_input("Đặt câu hỏi:")

if st.button("Trả lời"):
    if uploaded and query:
        paths = []
        for f in uploaded:
            path = f.name
            with open(path,"wb") as w: w.write(f.getbuffer())
            paths.append(path)

        index = build_or_load(paths)  # tạo/lấy cache index.pkl
        top = retrieve(query, index, k=10)

        # Tạo context từ 10 chunks
        context = "\n\n".join([c for _,_,c in top])

        # Gọi GPT để trả lời dựa trên context
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role":"system","content":"Bạn là trợ lý kỹ thuật, trả lời ngắn gọn bằng tiếng Việt chỉ dựa trên context."},
                {"role":"user","content": f"CONTEXT:\n{context}\n\nCÂU HỎI: {query}\n\nNếu context thiếu, nói 'không đủ dữ liệu'."}
            ],
            temperature=0.2
        )

        st.subheader("✨ Trả lời")
        st.write(resp.choices[0].message.content)

        st.subheader("🔟 Các đoạn trích liên quan")
        for score, meta, chunk in top:
            st.markdown(f"**{meta['file']} – chunk {meta['id']} (score={score:.2f})**")
            st.write((chunk[:800] + "...") if len(chunk) > 800 else chunk)
    else:
        st.warning("Hãy tải file và nhập câu hỏi.")
