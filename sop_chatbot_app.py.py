import os
import streamlit as st
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import GPT4All
from langchain.chains import RetrievalQA

# === STEP 1: Utility to load Word document ===
@st.cache_data
def load_docx_text(file_path):
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])

# === STEP 2: Load and embed SOP ===
@st.cache_resource
def prepare_qa_chain(sop_path, model_path):
    text = load_docx_text(sop_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.create_documents([text])

    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma.from_documents(chunks, embedding_model, persist_directory="sop_db_web")
    db.persist()

    llm = GPT4All(model=model_path, backend="gptj", verbose=False)
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())
    return qa

# === STREAMLIT UI ===
st.set_page_config(page_title="SOP Chatbot", layout="wide")
st.title("🤖 SOP Chatbot (Local & Offline)")
st.markdown("Ask questions based on your SOP document. Powered by GPT4All + HuggingFace Embeddings. No API needed.")

# === File Upload ===
uploaded_file = st.file_uploader("Upload your SOP Word (.docx) file", type=["docx"])

# === Model Path ===
default_model_path = "ggml-gpt4all-j-v1.3-groovy.bin"
model_path = st.text_input("Path to GPT4All model (.bin):", default_model_path)

# === Load chatbot ===
if uploaded_file and os.path.exists(model_path):
    with open("temp_sop.docx", "wb") as f:
        f.write(uploaded_file.read())

    with st.spinner("Processing SOP and loading model..."):
        qa_chain = prepare_qa_chain("temp_sop.docx", model_path)
    st.success("Chatbot is ready! Start asking questions.")

    # === Chat Interface ===
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("You:", key="input")
    if st.button("Ask") and user_input:
        response = qa_chain.run(user_input)
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Bot", response))

    for role, message in reversed(st.session_state.chat_history):
        st.markdown(f"**{role}:** {message}")
else:
    st.warning("Upload a DOCX file and make sure the model path is correct.")



