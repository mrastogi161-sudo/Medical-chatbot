import streamlit as st
from PIL import Image
import base64
from io import BytesIO

from medical_chatbot_with_multimodel_rag import multimodal_rag_w_sources

st.set_page_config(page_title="🩺 Medical Chatbot with RAG", page_icon="🧬")
st.title("🩺 Multimodal Medical Chatbot with RAG")
st.caption("Ask medical questions; get answers from text + extracted images!")

query = st.text_input("🔍 Enter your medical question:")

if st.button("Ask") and query:
    with st.spinner("Retrieving..."):
        response = multimodal_rag_w_sources.invoke({'input': query})

    st.subheader("✅ Answer")
    st.write(response['answer'])

    st.subheader("📚 Retrieved text chunks")

    # Showing retrieved images if present
    if 'images' in response and response['images']:
        st.subheader("🖼️ Retrieved images")
        for img_b64 in response['images']:
            img_data = base64.b64decode(img_b64)
            img = Image.open(BytesIO(img_data))
            st.image(img, use_column_width=True)

