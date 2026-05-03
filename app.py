import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
import base64
import os
from datetime import datetime

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ─── Helpers ────────────────────────────────────────────────────────────────

def get_pdf_text(pdf_docs: list) -> str:
    """Extract raw text from a list of uploaded PDF files."""
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted
    return text


def get_text_chunks(text: str) -> list[str]:
    """Split text into overlapping chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(text)


def build_vector_store(text_chunks: list[str], api_key: str) -> FAISS:
    """Embed text chunks and save a FAISS index locally."""
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key,
    )
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store


def get_rag_chain(vector_store, api_key: str):
    retriever = vector_store.as_retriever()

    prompt_template = """
Answer the question as detailed as possible from the provided context.
If the answer is not in the context, say "Answer is not available in the context."

Context:
{context}

Question:
{question}

Answer:
"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"],
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview",
        temperature=0.3,
        google_api_key=api_key,
    )

    chain = (
        {"context": retriever, "question": lambda x: x}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


# ─── Chat UI ─────────────────────────────────────────────────────────────────

CHAT_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

body, .stApp { font-family: 'Inter', sans-serif; }

.chat-wrap { display: flex; flex-direction: column; gap: 1rem; margin-bottom: 1.5rem; }

.chat-message {
    padding: 1.2rem 1.5rem;
    border-radius: 12px;
    display: flex;
    align-items: flex-start;
    gap: 1rem;
    animation: fadeIn 0.3s ease;
}
@keyframes fadeIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; } }

.chat-message.user  { background: #1e2330; border-left: 3px solid #4f8ef7; }
.chat-message.bot   { background: #252d3d; border-left: 3px solid #34d399; }

.chat-message .avatar img {
    width: 42px; height: 42px;
    border-radius: 50%;
    object-fit: cover;
    flex-shrink: 0;
}
.chat-message .message { color: #e8eaf0; font-size: 0.95rem; line-height: 1.6; }
.label { font-size: 0.7rem; font-weight: 600; letter-spacing: 0.08em;
         text-transform: uppercase; margin-bottom: 4px; opacity: 0.55; }
.user .label  { color: #4f8ef7; }
.bot  .label  { color: #34d399; }
</style>
"""


def render_message(question: str, answer: str):
    """Render a single Q/A exchange as styled chat bubbles."""
    st.markdown(
        f"""
        {CHAT_CSS}
        <div class="chat-wrap">
            <div class="chat-message user">
                <div class="avatar">
                    <img src="https://i.ibb.co/CKpTnWr/user-icon-2048x2048-ihoxz4vq.png">
                </div>
                <div class="message">
                    <div class="label">You</div>
                    {question}
                </div>
            </div>
            <div class="chat-message bot">
                <div class="avatar">
                    <img src="https://i.ibb.co/wNmYHsx/langchain-logo.webp">
                </div>
                <div class="message">
                    <div class="label">Assistant</div>
                    {answer}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─── Core Q&A ────────────────────────────────────────────────────────────────

def process_question(user_question: str, api_key: str, pdf_docs: list):
    """
    1. Extract + chunk PDF text.
    2. Build / reload FAISS index.
    3. Run similarity search & QA chain.
    4. Persist result in session history.
    5. Render chat UI.
    """
    if not api_key:
        st.warning("Please enter your Google API key.")
        return
    if not pdf_docs:
        st.warning("Please upload at least one PDF.")
        return

    with st.spinner("Thinking…"):
        # Build vector store fresh each call (avoids stale index issues)
        raw_text = get_pdf_text(pdf_docs)
        if not raw_text.strip():
            st.error("Could not extract any text from the uploaded PDFs.")
            return

        chunks = get_text_chunks(raw_text)
        build_vector_store(chunks, api_key)

        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=api_key,
        )
        db = FAISS.load_local(
            "faiss_index", embeddings, allow_dangerous_deserialization=True
        )
        chain = get_rag_chain(db, api_key)
        answer = chain.invoke(user_question)

    # Persist to history
    pdf_names = ", ".join(p.name for p in pdf_docs)
    st.session_state.conversation_history.append(
        {
            "question": user_question,
            "answer": answer,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "pdfs": pdf_names,
        }
    )

    # Render history (newest first)
    for entry in reversed(st.session_state.conversation_history):
        render_message(entry["question"], entry["answer"])

    _offer_csv_download()
    st.snow()


# ─── CSV Download ─────────────────────────────────────────────────────────────

def _offer_csv_download():
    history = st.session_state.conversation_history
    if not history:
        return
    df = pd.DataFrame(history, columns=["question", "answer", "timestamp", "pdfs"])
    df.columns = ["Question", "Answer", "Timestamp", "PDF Names"]
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = (
        f'<a href="data:file/csv;base64,{b64}" download="conversation_history.csv">'
        '<button style="margin-top:8px;padding:6px 14px;border-radius:6px;'
        'background:#4f8ef7;color:#fff;border:none;cursor:pointer;font-size:0.85rem;">'
        "⬇ Download Conversation CSV"
        "</button></a>"
    )
    st.sidebar.markdown(href, unsafe_allow_html=True)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="Chat with PDFs", page_icon="📚", layout="wide")
    st.title("📚 Chat with Multiple PDFs")

    # Session state init
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Settings")

        # Social links
        linkedin = "put url here"
        kaggle   = "put url here"
        github   = "put url here"
        st.markdown(
            f"[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)]({linkedin}) "
            f"[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)]({kaggle}) "
            f"[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)]({github})"
        )
        st.divider()

        # API key
        api_key = st.text_input(
            "Google API Key", type="password", placeholder="Paste your key here…"
        )
        st.markdown("Get a key at [ai.google.dev](https://ai.google.dev/)")
        if not api_key:
            st.warning("API key required to proceed.")

        st.divider()

        # PDF upload
        pdf_docs = st.file_uploader(
            "Upload PDF files",
            accept_multiple_files=True,
            type=["pdf"],
        )
        if st.button("✅ Submit & Process", use_container_width=True):
            if pdf_docs:
                st.success(f"{len(pdf_docs)} file(s) ready.")
            else:
                st.warning("Please upload at least one PDF first.")

        st.divider()

        # Reset
        col1, col2 = st.columns(2)
        if col1.button("🔄 Rerun", use_container_width=True):
            if st.session_state.conversation_history:
                st.session_state.conversation_history.pop()
                st.rerun()
            else:
                st.info("Nothing to rerun.")

        if col2.button("🗑 Reset All", use_container_width=True):
            st.session_state.conversation_history = []
            st.rerun()

    # ── Main area ────────────────────────────────────────────────────────────
    user_question = st.text_input(
        "💬 Ask a question about your PDFs",
        placeholder="e.g. Summarise the key findings…",
        key="question_input",
    )

    if user_question:
        process_question(user_question, api_key, pdf_docs)


if __name__ == "__main__":
    main()












# import streamlit as st
# from PyPDF2 import PdfReader
# import pandas as pd
# import base64
# from datetime import datetime

# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_ollama import OllamaEmbeddings, ChatOllama
# from langchain_community.vectorstores import FAISS
# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import StrOutputParser


# # ─── Helpers ────────────────────────────────────────────────────────────────

# def get_pdf_text(pdf_docs: list) -> str:
#     text = ""
#     for pdf in pdf_docs:
#         reader = PdfReader(pdf)
#         for page in reader.pages:
#             extracted = page.extract_text()
#             if extracted:
#                 text += extracted
#     return text


# def get_text_chunks(text: str) -> list[str]:
#     splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     return splitter.split_text(text)


# def build_vector_store(text_chunks: list[str], embedding_model: str) -> FAISS:
#     embeddings = OllamaEmbeddings(model=embedding_model)
#     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
#     vector_store.save_local("faiss_index")
#     return vector_store


# def get_rag_chain(vector_store, llm_model: str):
#     retriever = vector_store.as_retriever(search_kwargs={"k": 5})

#     prompt_template = """Answer the question as detailed as possible from the provided context.
# If the answer is not in the context, say "Answer is not available in the context."

# Context:
# {context}

# Question:
# {question}

# Answer:"""

#     prompt = PromptTemplate(
#         template=prompt_template,
#         input_variables=["context", "question"],
#     )

#     llm = ChatOllama(model=llm_model, temperature=0.3)

#     chain = (
#         {"context": retriever, "question": lambda x: x}
#         | prompt
#         | llm
#         | StrOutputParser()
#     )
#     return chain


# # ─── Chat UI ─────────────────────────────────────────────────────────────────

# CHAT_CSS = """
# <style>
# @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
# body, .stApp { font-family: 'Inter', sans-serif; }
# .chat-wrap { display: flex; flex-direction: column; gap: 1rem; margin-bottom: 1.5rem; }
# .chat-message {
#     padding: 1.2rem 1.5rem; border-radius: 12px;
#     display: flex; align-items: flex-start; gap: 1rem;
#     animation: fadeIn 0.3s ease;
# }
# @keyframes fadeIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; } }
# .chat-message.user { background: #1e2330; border-left: 3px solid #4f8ef7; }
# .chat-message.bot  { background: #252d3d; border-left: 3px solid #34d399; }
# .chat-message .avatar img {
#     width: 42px; height: 42px; border-radius: 50%; object-fit: cover; flex-shrink: 0;
# }
# .chat-message .message { color: #e8eaf0; font-size: 0.95rem; line-height: 1.6; }
# .label {
#     font-size: 0.7rem; font-weight: 600; letter-spacing: 0.08em;
#     text-transform: uppercase; margin-bottom: 4px; opacity: 0.55;
# }
# .user .label { color: #4f8ef7; }
# .bot  .label { color: #34d399; }
# </style>
# """


# def render_message(question: str, answer: str):
#     st.markdown(
#         f"""
#         {CHAT_CSS}
#         <div class="chat-wrap">
#             <div class="chat-message user">
#                 <div class="avatar">
#                     <img src="https://i.ibb.co/CKpTnWr/user-icon-2048x2048-ihoxz4vq.png">
#                 </div>
#                 <div class="message">
#                     <div class="label">You</div>
#                     {question}
#                 </div>
#             </div>
#             <div class="chat-message bot">
#                 <div class="avatar">
#                     <img src="https://i.ibb.co/wNmYHsx/langchain-logo.webp">
#                 </div>
#                 <div class="message">
#                     <div class="label">Assistant (Ollama)</div>
#                     {answer}
#                 </div>
#             </div>
#         </div>
#         """,
#         unsafe_allow_html=True,
#     )


# # ─── Core Q&A ────────────────────────────────────────────────────────────────

# def process_question(
#     user_question: str,
#     llm_model: str,
#     embedding_model: str,
#     pdf_docs: list,
# ):
#     if not pdf_docs:
#         st.warning("Please upload at least one PDF.")
#         return

#     with st.spinner(f"Running locally — LLM: `{llm_model}` | Embeddings: `{embedding_model}`…"):
#         raw_text = get_pdf_text(pdf_docs)
#         if not raw_text.strip():
#             st.error("Could not extract any text from the uploaded PDFs.")
#             return

#         chunks = get_text_chunks(raw_text)

#         try:
#             build_vector_store(chunks, embedding_model)
#             db = FAISS.load_local(
#                 "faiss_index",
#                 OllamaEmbeddings(model=embedding_model),
#                 allow_dangerous_deserialization=True,
#             )
#             chain = get_rag_chain(db, llm_model)
#             answer = chain.invoke(user_question)
#         except Exception as e:
#             st.error(
#                 f"**Ollama error:** {e}\n\n"
#                 "**Checklist:**\n"
#                 "- Is Ollama running? → `ollama serve`\n"
#                 f"- LLM pulled? → `ollama pull {llm_model}`\n"
#                 f"- Embeddings pulled? → `ollama pull {embedding_model}`"
#             )
#             return

#     pdf_names = ", ".join(p.name for p in pdf_docs)
#     st.session_state.conversation_history.append(
#         {
#             "question": user_question,
#             "answer": answer,
#             "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#             "pdfs": pdf_names,
#         }
#     )

#     for entry in reversed(st.session_state.conversation_history):
#         render_message(entry["question"], entry["answer"])

#     _offer_csv_download()
#     st.snow()


# # ─── CSV Download ─────────────────────────────────────────────────────────────

# def _offer_csv_download():
#     history = st.session_state.conversation_history
#     if not history:
#         return
#     df = pd.DataFrame(history, columns=["question", "answer", "timestamp", "pdfs"])
#     df.columns = ["Question", "Answer", "Timestamp", "PDF Names"]
#     csv = df.to_csv(index=False)
#     b64 = base64.b64encode(csv.encode()).decode()
#     href = (
#         f'<a href="data:file/csv;base64,{b64}" download="conversation_history.csv">'
#         '<button style="margin-top:8px;padding:6px 14px;border-radius:6px;'
#         'background:#4f8ef7;color:#fff;border:none;cursor:pointer;font-size:0.85rem;">'
#         "⬇ Download Conversation CSV"
#         "</button></a>"
#     )
#     st.sidebar.markdown(href, unsafe_allow_html=True)


# # ─── Model lists ─────────────────────────────────────────────────────────────

# LLM_MODELS = [
#     "llama3.2", "llama3.1", "llama3",
#     "mistral", "mistral-nemo",
#     "gemma3", "gemma2",
#     "phi4", "phi3",
#     "qwen2.5", "deepseek-r1",
# ]

# EMBEDDING_MODELS = [
#     "nomic-embed-text",        # recommended default — fast & accurate
#     "mxbai-embed-large",       # higher quality, slower
#     "all-minilm",              # very lightweight
#     "snowflake-arctic-embed",  # strong retrieval perf
# ]


# # ─── Main ─────────────────────────────────────────────────────────────────────

# def main():
#     st.set_page_config(
#         page_title="Chat with PDFs — Local Ollama",
#         page_icon="📚",
#         layout="wide",
#     )
#     st.title("📚 Chat with Multiple PDFs")
#     st.caption("100% local · no API keys · powered by [Ollama](https://ollama.com)")

#     if "conversation_history" not in st.session_state:
#         st.session_state.conversation_history = []

#     # ── Sidebar ──────────────────────────────────────────────────────────────
#     with st.sidebar:
#         st.header("⚙️ Settings")

#         linkedin = "https://www.linkedin.com/in/snsupratim/"
#         kaggle   = "https://www.kaggle.com/snsupratim/"
#         github   = "https://github.com/snsupratim/"
#         st.markdown(
#             f"[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)]({linkedin}) "
#             f"[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)]({kaggle}) "
#             f"[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)]({github})"
#         )
#         st.divider()

#         with st.expander("🛠 Ollama setup guide", expanded=False):
#             st.markdown(
#                 "1. [Download Ollama](https://ollama.com/download)\n"
#                 "2. Start server: `ollama serve`\n"
#                 "3. Pull LLM: `ollama pull llama3.2`\n"
#                 "4. Pull embeddings: `ollama pull nomic-embed-text`"
#             )

#         st.divider()

#         # LLM selector
#         st.subheader("🤖 LLM Model")
#         llm_choice = st.selectbox("Preset", LLM_MODELS, index=0)
#         llm_custom = st.text_input("Custom name (overrides preset)", placeholder="e.g. codellama")
#         llm_model  = llm_custom.strip() or llm_choice

#         st.divider()

#         # Embedding selector
#         st.subheader("🔢 Embedding Model")
#         emb_choice = st.selectbox("Preset", EMBEDDING_MODELS, index=0)
#         emb_custom = st.text_input("Custom name (overrides preset)", placeholder="e.g. bge-m3")
#         embedding_model = emb_custom.strip() or emb_choice

#         st.divider()
#         st.caption(f"**Active LLM:** `{llm_model}`")
#         st.caption(f"**Active Embeddings:** `{embedding_model}`")
#         st.divider()

#         # PDF upload
#         pdf_docs = st.file_uploader(
#             "Upload PDF files",
#             accept_multiple_files=True,
#             type=["pdf"],
#         )
#         if st.button("✅ Submit & Process", use_container_width=True):
#             st.success(f"{len(pdf_docs)} file(s) ready.") if pdf_docs else st.warning("Upload at least one PDF first.")

#         st.divider()

#         col1, col2 = st.columns(2)
#         if col1.button("🔄 Undo Last", use_container_width=True):
#             if st.session_state.conversation_history:
#                 st.session_state.conversation_history.pop()
#                 st.rerun()
#             else:
#                 st.info("Nothing to undo.")
#         if col2.button("🗑 Reset All", use_container_width=True):
#             st.session_state.conversation_history = []
#             st.rerun()

#     # ── Main area ────────────────────────────────────────────────────────────
#     user_question = st.text_input(
#         "💬 Ask a question about your PDFs",
#         placeholder="e.g. Summarise the key findings…",
#         key="question_input",
#     )

#     if user_question:
#         process_question(user_question, llm_model, embedding_model, pdf_docs)


# if __name__ == "__main__":
#     main()