import streamlit as st
import uuid
import time
from datetime import datetime

from pinecone import Pinecone
from openai import OpenAI
from pypdf import PdfReader

# ==================================================
# FIXED CONFIG (YOUR INDEX)
# ==================================================
INDEX_NAME = "wolf"
PINECONE_HOST = "https://wolf-b79cc48.svc.aped-4627-b74a.pinecone.io"
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536
NAMESPACE = "default"

# ==================================================
# STREAMLIT CONFIG
# ==================================================
st.set_page_config(
    page_title="🐺 WOLF • RAG Intelligence",
    page_icon="🐺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==================================================
# SIDEBAR
# ==================================================
with st.sidebar:
    st.title("🐺 WOLF AI")
    st.caption("RAG • Chat • Vector Control")

    st.divider()

    PINECONE_API_KEY = st.text_input("Pinecone API Key", type="password")
    OPENAI_API_KEY = st.text_input("OpenAI API Key", type="password")

    st.divider()

    CHUNK_SIZE = st.slider("Chunk Size", 300, 2000, 900, 100)
    CHUNK_OVERLAP = st.slider("Chunk Overlap", 0, 500, 150, 50)

    st.divider()
    st.info(
        f"**Index:** wolf\n\n"
        f"**Host:** connected\n\n"
        f"**Embedding:** {EMBED_MODEL}\n\n"
        f"**Namespace:** default"
    )

# ==================================================
# CLIENT INIT
# ==================================================
@st.cache_resource(show_spinner=False)
def init_clients(pinecone_key, openai_key):
    pc = Pinecone(api_key=pinecone_key)
    index = pc.Index(name=INDEX_NAME, host=PINECONE_HOST)
    openai_client = OpenAI(api_key=openai_key)
    return pc, index, openai_client

# ==================================================
# HELPERS
# ==================================================
def extract_text(file):
    if file.type == "text/plain":
        return file.read().decode("utf-8")

    if file.type == "application/pdf":
        reader = PdfReader(file)
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    return ""

def chunk_text(text, size, overlap):
    words = text.split()
    chunks = []

    for i in range(0, len(words), size - overlap):
        chunk = " ".join(words[i:i + size])
        if chunk.strip():
            chunks.append(chunk)

    return chunks

def embed(text):
    res = openai_client.embeddings.create(
        model=EMBED_MODEL,
        input=text
    )
    return res.data[0].embedding

def rag_answer(question, matches):
    context_blocks = []
    citations = []

    for i, m in enumerate(matches, 1):
        text = m.metadata.get("text", "")
        source = m.metadata.get("source", "unknown")
        context_blocks.append(f"[{i}] {text}")
        citations.append(f"[{i}] {source}")

    prompt = f"""
You are WOLF, an expert document analyst.

Answer the question using ONLY the context below.
Cite sources using bracket numbers like [1], [2].

Context:
{chr(10).join(context_blocks)}

Question:
{question}

Answer:
"""

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return response.choices[0].message.content, citations

# ==================================================
# MAIN HEADER
# ==================================================
st.markdown("<h1 style='text-align:center;'>🐺 WOLF RAG Intelligence</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Upload • Search • Chat • Control</p>", unsafe_allow_html=True)

# ==================================================
# AUTH CHECK
# ==================================================
if not PINECONE_API_KEY or not OPENAI_API_KEY:
    st.warning("🔑 Enter API keys to activate WOLF")
    st.stop()

pc, index, openai_client = init_clients(PINECONE_API_KEY, OPENAI_API_KEY)

# ==================================================
# FILE UPLOAD
# ==================================================
st.header("📤 Document Upload")

files = st.file_uploader(
    "Upload TXT or PDF",
    type=["txt", "pdf"],
    accept_multiple_files=True
)

if files and st.button("🚀 Index Documents", use_container_width=True):
    vectors = []
    progress = st.progress(0)

    for i, file in enumerate(files):
        progress.progress(i / len(files))
        text = extract_text(file)
        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)

        for idx, chunk in enumerate(chunks):
            vectors.append({
                "id": str(uuid.uuid4()),
                "values": embed(chunk),
                "metadata": {
                    "source": file.name,
                    "chunk": idx,
                    "text": chunk[:1000],
                    "indexed_at": datetime.utcnow().isoformat()
                }
            })

    for i in range(0, len(vectors), 100):
        index.upsert(vectors=vectors[i:i + 100], namespace=NAMESPACE)

    st.success(f"✅ Indexed {len(vectors)} chunks")

# ==================================================
# DELETE BY FILE
# ==================================================
st.header("🗂️ Delete by File")

delete_file = st.text_input("Exact filename to delete (case-sensitive)")

if st.button("🗑️ Delete File Vectors", use_container_width=True):
    if delete_file:
        index.delete(
            namespace=NAMESPACE,
            filter={"source": {"$eq": delete_file}}
        )
        st.success(f"Deleted vectors for {delete_file}")
    else:
        st.warning("Enter a filename")

# ==================================================
# CHAT UI (RAG)
# ==================================================
st.header("💬 Chat With Your Documents")

if "chat" not in st.session_state:
    st.session_state.chat = []

for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Ask WOLF...")

if query:
    st.session_state.chat.append({"role": "user", "content": query})

    with st.spinner("🐺 WOLF is thinking..."):
        q_embed = embed(query)
        results = index.query(
            vector=q_embed,
            top_k=5,
            include_metadata=True,
            namespace=NAMESPACE
        )

        answer, citations = rag_answer(query, results.matches)

    st.session_state.chat.append({
        "role": "assistant",
        "content": f"{answer}\n\n**Sources:** {', '.join(citations)}"
    })

    st.rerun()

# ==================================================
# FOOTER
# ==================================================
st.divider()
st.caption("🐺 WOLF • Pinecone • OpenAI • RAG • Chat • Control")
