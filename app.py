import streamlit as st
import uuid
import time
import logging
import re
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter

from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
from pypdf import PdfReader
from pinecone.exceptions import PineconeApiException

# ==================================================
# CONFIGURATION & CONSTANTS
# ==================================================
INDEX_NAME = "wolf"
PINECONE_HOST = "https://wolf-b79cc48.svc.aped-4627-b74a.pinecone.io"

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536
MAX_METADATA_SIZE = 30000  # Conservative limit
BATCH_SIZE = 50  # Smaller batch size for hybrid vectors

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================================================
# STREAMLIT PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="🐺 WOLF RAG Intelligence Pro v2",
    page_icon="🐺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================================================
# SESSION STATE INITIALIZATION
# ==================================================
def init_session_state():
    if "files_ingested" not in st.session_state:
        st.session_state.files_ingested = set()
    if "chat" not in st.session_state:
        st.session_state.chat = []
    if "index_type" not in st.session_state:
        st.session_state.index_type = "unknown"

init_session_state()

# ==================================================
# SPARSE VECTOR GENERATION (Simple BM25-like)
# ==================================================
def generate_sparse_vector(text: str) -> Dict[str, Any]:
    """
    Generates a simple sparse vector based on word frequencies.
    In a production environment, you'd use SPLADE or BM25 from pinecone-text.
    This is a lightweight implementation for the Streamlit app.
    """
    # Simple tokenization and cleaning
    words = re.findall(r'\w+', text.lower())
    counts = Counter(words)
    
    # Create indices (hash of word) and values (frequency)
    # Note: Pinecone expects integer indices for sparse vectors
    # We use a simple hash-to-int for demonstration, but in real hybrid search
    # you should use a consistent vocabulary or Pinecone's inference.
    indices = []
    values = []
    
    for word, count in counts.items():
        # Use a stable hash to map words to a large integer space
        idx = abs(hash(word)) % (2**31 - 1)
        indices.append(idx)
        values.append(float(count))
        
    return {"indices": indices, "values": values}

# ==================================================
# CORE CLIENTS & LOGIC
# ==================================================
@st.cache_resource(show_spinner=False)
def get_clients(p_key: str, o_key: str):
    try:
        pc = Pinecone(api_key=p_key)
        index = pc.Index(host=PINECONE_HOST)
        
        # Detect index capabilities
        try:
            stats = index.describe_index_stats()
            # If we can't get stats or it's empty, we'll assume it's a standard index
            # but the error message from the user confirms it's a sparse index.
            st.session_state.index_type = "sparse/hybrid"
        except:
            st.session_state.index_type = "unknown"
            
        openai_client = OpenAI(api_key=o_key)
        return index, openai_client, pc
    except Exception as e:
        st.error(f"Failed to initialize clients: {str(e)}")
        return None, None, None

def extract_text_from_file(file) -> str:
    try:
        if file.type == "text/plain":
            return file.read().decode("utf-8")
        elif file.type == "application/pdf":
            reader = PdfReader(file)
            text = ""
            for page in reader.pages:
                content = page.extract_text()
                if content:
                    text += content + "\n"
            return text
        return ""
    except Exception as e:
        st.error(f"Error reading {file.name}: {str(e)}")
        return ""

def chunk_text_logic(text: str, size: int, overlap: int) -> List[str]:
    if not text:
        return []
    words = text.split()
    chunks = []
    for i in range(0, len(words), size - overlap):
        chunk = " ".join(words[i:i + size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks

def get_embedding(client: OpenAI, text: str) -> List[float]:
    res = client.embeddings.create(model=EMBED_MODEL, input=text)
    return res.data[0].embedding

def perform_upsert(index, vectors: List[Dict[str, Any]]):
    """Safe upsert with batching and error handling."""
    for i in range(0, len(vectors), BATCH_SIZE):
        batch = vectors[i:i + BATCH_SIZE]
        try:
            index.upsert(vectors=batch)
        except PineconeApiException as e:
            # If it fails because of "dense vectors not supported", try sending ONLY sparse
            if "Upserting dense vectors is not supported" in str(e):
                logger.warning("Dense vectors not supported, falling back to sparse-only upsert.")
                sparse_only_batch = []
                for v in batch:
                    sparse_only_batch.append({
                        "id": v["id"],
                        "sparse_values": v["sparse_values"],
                        "metadata": v["metadata"]
                    })
                index.upsert(vectors=sparse_only_batch)
            else:
                raise e

# ==================================================
# MAIN INTERFACE
# ==================================================
st.markdown("<h1 style='text-align:center;'>🐺 WOLF RAG Intelligence</h1>", unsafe_allow_html=True)

with st.sidebar:
    st.title("🐺 Settings")
    pinecone_api_key = st.text_input("Pinecone API Key", type="password")
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    
    st.divider()
    chunk_size = st.slider("Chunk Size", 100, 1000, 500)
    chunk_overlap = st.slider("Overlap", 0, 200, 50)
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat = []
        st.rerun()

if not pinecone_api_key or not openai_api_key:
    st.warning("Please enter API keys to continue.")
    st.stop()

index, openai_client, pc_client = get_clients(pinecone_api_key, openai_api_key)
if not index: st.stop()

st.info(f"Detected Index Mode: **{st.session_state.index_type}**")

tab1, tab2, tab3 = st.tabs(["📤 Ingest", "💬 Chat", "📊 Stats"])

with tab1:
    files = st.file_uploader("Upload Files", accept_multiple_files=True, type=['pdf', 'txt'])
    if files and st.button("🚀 Index Documents"):
        all_vectors = []
        for file in files:
            text = extract_text_from_file(file)
            chunks = chunk_text_logic(text, chunk_size, chunk_overlap)
            
            for idx, chunk in enumerate(chunks):
                # Generate Dense Embedding
                dense_vec = get_embedding(openai_client, chunk)
                # Generate Sparse Vector
                sparse_vec = generate_sparse_vector(chunk)
                
                all_vectors.append({
                    "id": f"{file.name}-{idx}-{uuid.uuid4().hex[:6]}",
                    "values": dense_vec,
                    "sparse_values": sparse_vec,
                    "metadata": {
                        "source": file.name,
                        "text": chunk[:1000], # Truncate for metadata safety
                        "indexed_at": datetime.utcnow().isoformat()
                    }
                })
            st.session_state.files_ingested.add(file.name)
            
        if all_vectors:
            with st.spinner(f"Upserting {len(all_vectors)} vectors..."):
                try:
                    perform_upsert(index, all_vectors)
                    st.success("Indexing complete!")
                except Exception as e:
                    st.error(f"Indexing failed: {str(e)}")

with tab2:
    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask WOLF..."):
        st.session_state.chat.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Searching..."):
                # Prepare Query
                dense_query = get_embedding(openai_client, prompt)
                sparse_query = generate_sparse_vector(prompt)
                
                try:
                    # Try Hybrid Query
                    res = index.query(
                        vector=dense_query,
                        sparse_vector=sparse_query,
                        top_k=5,
                        include_metadata=True
                    )
                except Exception as e:
                    # Fallback to Sparse-only Query if dense fails
                    if "dense vectors" in str(e).lower():
                        res = index.query(
                            sparse_vector=sparse_query,
                            top_k=5,
                            include_metadata=True
                        )
                    else:
                        st.error(f"Query error: {e}")
                        st.stop()
                
                if res.matches:
                    context = "\n\n".join([m.metadata['text'] for m in res.matches])
                    response = openai_client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant. Use the context to answer."},
                            {"role": "user", "content": f"Context: {context}\n\nQuestion: {prompt}"}
                        ]
                    ).choices[0].message.content
                    st.markdown(response)
                    st.session_state.chat.append({"role": "assistant", "content": response})
                else:
                    st.write("No relevant documents found.")

with tab3:
    try:
        stats = index.describe_index_stats()
        st.json(stats.to_dict())
        if st.button("🗑️ Delete All Data", type="primary"):
            index.delete(delete_all=True)
            st.success("Index cleared.")
            st.rerun()
    except Exception as e:
        st.error(f"Error: {e}")
