import streamlit as st
import uuid
from datetime import datetime
import time

from pinecone import Pinecone
from pypdf import PdfReader
from openai import OpenAI

# --------------------------------------------------
# Global Clients
# --------------------------------------------------
pc = None
index = None
openai_client = None

# --------------------------------------------------
# Streamlit Config
# --------------------------------------------------
st.set_page_config(
    page_title="Pinecone Document Manager",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# Initialize Clients
# --------------------------------------------------
@st.cache_resource
def initialize_clients(pinecone_key, openai_key, index_name):
    pc = Pinecone(api_key=pinecone_key)
    index = pc.Index(index_name)
    openai_client = OpenAI(api_key=openai_key)
    return pc, index, openai_client

# --------------------------------------------------
# Helpers
# --------------------------------------------------
def extract_text(file):
    try:
        if file.type == "text/plain":
            return file.read().decode("utf-8")

        if file.type == "application/pdf":
            reader = PdfReader(file)
            return "\n".join(
                page.extract_text() or "" for page in reader.pages
            )

    except Exception as e:
        st.error(f"Text extraction failed: {e}")

    return ""

def chunk_text(text, size=800, overlap=100):
    words = text.split()
    chunks = []

    for i in range(0, len(words), size - overlap):
        chunk = " ".join(words[i : i + size])
        if chunk.strip():
            chunks.append(chunk)

    return chunks

def embed_text(text):
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def format_bytes(size):
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return "TB+"

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")

    pinecone_key = st.text_input("Pinecone API Key", type="password")
    openai_key = st.text_input("OpenAI API Key", type="password")
    index_name = st.text_input("Index Name", value="quickstart")

    if pinecone_key and openai_key:
        try:
            pc, index, openai_client = initialize_clients(
                pinecone_key, openai_key, index_name
            )
            st.success("Clients initialized")
        except Exception as e:
            st.error(e)
    else:
        st.warning("Enter both API keys")

    st.divider()
    chunk_size = st.slider("Chunk Size", 200, 2000, 800, 100)
    overlap = st.slider("Overlap", 0, 500, 100, 50)

# --------------------------------------------------
# Main UI
# --------------------------------------------------
st.title("📚 Pinecone Document Manager")

st.markdown("""
Upload **TXT or PDF** files → chunk → embed → store in Pinecone  
Perfect for **RAG, chatbots, and semantic search**
""")

# --------------------------------------------------
# Upload Section
# --------------------------------------------------
files = st.file_uploader(
    "Upload Documents",
    type=["txt", "pdf"],
    accept_multiple_files=True
)

if files and st.button("🚀 Process & Upload", use_container_width=True):

    if not index or not openai_client:
        st.error("Clients not initialized")
        st.stop()

    progress = st.progress(0)
    status = st.empty()

    vectors = []
    total_chunks = 0

    for i, file in enumerate(files):
        status.text(f"Processing {file.name}")
        progress.progress(i / len(files))

        text = extract_text(file)
        chunks = chunk_text(text, chunk_size, overlap)
        total_chunks += len(chunks)

        for idx, chunk in enumerate(chunks):
            embedding = embed_text(chunk)

            vectors.append({
                "id": str(uuid.uuid4()),
                "values": embedding,
                "metadata": {
                    "source": file.name,
                    "chunk": idx,
                    "text": chunk[:1000],
                    "created": datetime.utcnow().isoformat()
                }
            })

    # Upload in batches
    status.text("Uploading to Pinecone...")
    for i in range(0, len(vectors), 100):
        index.upsert(vectors=vectors[i:i + 100])

    progress.progress(1.0)
    status.empty()

    st.success(f"✅ Uploaded {len(vectors)} vectors from {len(files)} files")

# --------------------------------------------------
# Index Stats
# --------------------------------------------------
st.header("📊 Index Stats")

if st.button("Refresh Stats"):
    if index:
        stats = index.describe_index_stats()
        st.json(stats)
    else:
        st.warning("Index not initialized")

# --------------------------------------------------
# Semantic Search
# --------------------------------------------------
st.header("🔍 Semantic Search")

query = st.text_input("Ask a question")
top_k = st.slider("Top K", 1, 20, 5)

if query and st.button("Search"):
    if not index:
        st.error("Index not initialized")
        st.stop()

    with st.spinner("Searching..."):
        query_embedding = embed_text(query)
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )

        if results["matches"]:
            for i, match in enumerate(results["matches"], 1):
                st.subheader(f"Result {i} — Score {match['score']:.4f}")
                st.info(match["metadata"].get("text", ""))
        else:
            st.warning("No results found")

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.divider()
st.caption("Streamlit • Pinecone • OpenAI — Upload → Embed → Search")
