import streamlit as st
import uuid
from datetime import datetime

from pinecone import Pinecone
from openai import OpenAI
from pypdf import PdfReader
from pinecone.exceptions import PineconeApiException

# ==================================================
# FIXED CONFIG
# ==================================================
INDEX_NAME = "wolf"
PINECONE_HOST = "https://wolf-b79cc48.svc.aped-4627-b74a.pinecone.io"

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536

# ==================================================
# STREAMLIT CONFIG
# ==================================================
st.set_page_config(
    page_title="🐺 WOLF RAG Intelligence",
    page_icon="🐺",
    layout="wide",
)

# ==================================================
# SESSION STATE
# ==================================================
if "files_ingested" not in st.session_state:
    st.session_state.files_ingested = set()

if "chat" not in st.session_state:
    st.session_state.chat = []

# ==================================================
# SIDEBAR
# ==================================================
with st.sidebar:
    st.title("🐺 WOLF AI")

    PINECONE_API_KEY = st.text_input("Pinecone API Key", type="password")
    OPENAI_API_KEY = st.text_input("OpenAI API Key", type="password")

    st.divider()
    CHUNK_SIZE = st.slider("Chunk Size", 300, 2000, 900, 100)
    CHUNK_OVERLAP = st.slider("Chunk Overlap", 0, 500, 150, 50)

# ==================================================
# CLIENT INIT
# ==================================================
@st.cache_resource(show_spinner=False)
def init_clients(pinecone_key, openai_key):
    pc = Pinecone(api_key=pinecone_key)
    index = pc.Index(host=PINECONE_HOST)
    openai_client = OpenAI(api_key=openai_key)
    return index, openai_client

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

def safe_query(index, vector, top_k=5):
    try:
        return index.query(
            vector=vector,
            top_k=top_k,
            include_metadata=True
        )
    except PineconeApiException:
        return None

def rag_answer(question, matches):
    context = []
    for i, m in enumerate(matches, 1):
        context.append(f"[{i}] {m.metadata.get('text', '')}")

    prompt = f"""
Answer ONLY using the context below.
Cite sources using [1], [2], etc.

Context:
{chr(10).join(context)}

Question:
{question}

Answer:
"""

    res = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    return res.choices[0].message.content

# ==================================================
# UI
# ==================================================
st.markdown("<h1 style='text-align:center;'>🐺 WOLF RAG Intelligence</h1>", unsafe_allow_html=True)

if not PINECONE_API_KEY or not OPENAI_API_KEY:
    st.warning("Enter API keys to activate WOLF")
    st.stop()

index, openai_client = init_clients(PINECONE_API_KEY, OPENAI_API_KEY)

# ==================================================
# UPLOAD
# ==================================================
st.header("📤 Upload & Index")

files = st.file_uploader(
    "Upload TXT or PDF",
    type=["txt", "pdf"],
    accept_multiple_files=True
)

if files and st.button("🚀 Index Documents", use_container_width=True):
    vectors = []

    for file in files:
        text = extract_text(file)
        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)

        for idx, chunk in enumerate(chunks):
            embedding = embed(chunk)

            if len(embedding) != EMBED_DIM:
                st.error("Embedding dimension mismatch")
                st.stop()

            vectors.append({
                "id": str(uuid.uuid4()),
                "values": embedding,
                "metadata": {
                    "source": file.name,
                    "chunk": idx,
                    "text": chunk[:800],
                    "indexed_at": datetime.utcnow().isoformat()
                }
            })

        st.session_state.files_ingested.add(file.name)

    for i in range(0, len(vectors), 100):
        index.upsert(vectors=vectors[i:i + 100])

    st.success(f"Indexed {len(vectors)} chunks")

# ==================================================
# VIEW INGESTED FILES (FIXED)
# ==================================================
st.header("📂 Ingested Files")

if st.session_state.files_ingested:
    for f in sorted(st.session_state.files_ingested):
        st.write(f"• {f}")
else:
    st.info("No files indexed yet")

# ==================================================
# DELETE BY FILE
# ==================================================
st.header("🗑️ Delete by File")

delete_file = st.selectbox(
    "Select file to delete",
    options=[""] + sorted(st.session_state.files_ingested)
)

if delete_file and st.button("Delete File Vectors", use_container_width=True):
    index.delete(filter={"source": {"$eq": delete_file}})
    st.session_state.files_ingested.remove(delete_file)
    st.success(f"Deleted {delete_file}")

# ==================================================
# CHAT
# ==================================================
st.header("💬 Chat")

for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask WOLF...")

if question:
    st.session_state.chat.append({"role": "user", "content": question})

    q_embed = embed(question)

    results = safe_query(index, q_embed)

    if not results or not results.matches:
        answer = "⚠️ No indexed data available yet."
    else:
        answer = rag_answer(question, results.matches)

    st.session_state.chat.append({
        "role": "assistant",
        "content": answer
    })

    st.rerun()

st.divider()
st.caption("🐺 WOLF • Pinecone Serverless • Stable")
