import streamlit as st
import uuid
import time
import logging
import re
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter

from pinecone import Pinecone
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
BATCH_SIZE = 40  # Conservative batch size for hybrid search

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================================================
# STREAMLIT PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="🐺 WOLF RAG Intelligence Ultimate",
    page_icon="🐺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a professional look
st.markdown("""
    <style>
    .stAlert { border-radius: 10px; }
    .stButton>button { border-radius: 8px; font-weight: 600; }
    .file-card {
        padding: 15px;
        border-radius: 10px;
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        margin-bottom: 10px;
    }
    .status-badge {
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8em;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# ==================================================
# SESSION STATE INITIALIZATION
# ==================================================
def init_session_state():
    if "files_metadata" not in st.session_state:
        st.session_state.files_metadata = {} # {filename: {chunks: N, date: str, status: str}}
    if "chat" not in st.session_state:
        st.session_state.chat = []
    if "auth_status" not in st.session_state:
        st.session_state.auth_status = False

init_session_state()

# ==================================================
# UTILITIES
# ==================================================
def generate_sparse_vector(text: str) -> Dict[str, Any]:
    words = re.findall(r'\w+', text.lower())
    counts = Counter(words)
    indices = []
    values = []
    for word, count in counts.items():
        idx = abs(hash(word)) % (2**31 - 1)
        indices.append(idx)
        values.append(float(count))
    return {"indices": indices, "values": values}

@st.cache_resource(show_spinner=False)
def get_clients(p_key: str, o_key: str):
    try:
        pc = Pinecone(api_key=p_key)
        index = pc.Index(host=PINECONE_HOST)
        # Test connection
        index.describe_index_stats()
        openai_client = OpenAI(api_key=o_key)
        st.session_state.auth_status = True
        return index, openai_client
    except Exception as e:
        st.session_state.auth_status = False
        st.error(f"Authentication Failed: {str(e)}")
        return None, None

def extract_text(file) -> str:
    try:
        if file.type == "text/plain":
            return file.read().decode("utf-8")
        elif file.type == "application/pdf":
            reader = PdfReader(file)
            return "\n".join([p.extract_text() or "" for p in reader.pages])
        return ""
    except Exception as e:
        logger.error(f"Extraction error: {e}")
        return ""

def chunk_text(text: str, size: int = 500, overlap: int = 50) -> List[str]:
    words = text.split()
    return [" ".join(words[i:i + size]) for i in range(0, len(words), size - overlap) if words[i:i + size]]

# ==================================================
# SIDEBAR
# ==================================================
with st.sidebar:
    st.title("🐺 WOLF Control")
    
    with st.expander("🔑 Credentials", expanded=not st.session_state.auth_status):
        p_key = st.text_input("Pinecone API Key", type="password", value="")
        o_key = st.text_input("OpenAI API Key", type="password", value="")
        if st.button("Verify Keys", use_container_width=True):
            st.cache_resource.clear()
            st.rerun()

    st.divider()
    st.subheader("⚙️ Parameters")
    c_size = st.slider("Chunk Size", 200, 1500, 600)
    c_overlap = st.slider("Overlap", 0, 300, 100)
    top_k = st.slider("Top K Results", 1, 10, 5)
    
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat = []
        st.rerun()

# ==================================================
# MAIN APP LOGIC
# ==================================================
if not p_key or not o_key:
    st.info("👋 Please enter your API keys in the sidebar to activate WOLF.")
    st.stop()

index, openai_client = get_clients(p_key, o_key)
if not index: st.stop()

st.markdown("<h1 style='text-align:center;'>🐺 WOLF RAG Intelligence</h1>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📤 Smart Ingest", "💬 Chat Interface", "📂 File Manager"])

# --- TAB 1: SMART INGEST ---
with tab1:
    st.subheader("Bulk Upload & One-at-a-Time Ingestion")
    uploaded_files = st.file_uploader("Drop files here", accept_multiple_files=True, type=['pdf', 'txt'])
    
    if uploaded_files:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"Selected {len(uploaded_files)} files.")
        with col2:
            start_ingest = st.button("🚀 Start Ingestion", use_container_width=True, type="primary")
            
        if start_ingest:
            overall_progress = st.progress(0)
            status_area = st.empty()
            
            for i, file in enumerate(uploaded_files):
                try:
                    status_area.markdown(f"**Processing ({i+1}/{len(uploaded_files)}):** `{file.name}`")
                    text = extract_text(file)
                    chunks = chunk_text(text, c_size, c_overlap)
                    
                    vectors = []
                    for idx, chunk in enumerate(chunks):
                        dense = openai_client.embeddings.create(model=EMBED_MODEL, input=chunk).data[0].embedding
                        sparse = generate_sparse_vector(chunk)
                        
                        vectors.append({
                            "id": f"{file.name}-{idx}-{uuid.uuid4().hex[:4]}",
                            "values": dense,
                            "sparse_values": sparse,
                            "metadata": {"source": file.name, "text": chunk[:1000], "indexed_at": str(datetime.now())}
                        })
                    
                    # Upsert in small batches for this specific file
                    for j in range(0, len(vectors), BATCH_SIZE):
                        batch = vectors[j:j + BATCH_SIZE]
                        try:
                            index.upsert(vectors=batch)
                        except Exception as e:
                            if "dense vectors" in str(e).lower():
                                # Fallback for sparse-only indexes
                                index.upsert(vectors=[{"id": v["id"], "sparse_values": v["sparse_values"], "metadata": v["metadata"]} for v in batch])
                    
                    st.session_state.files_metadata[file.name] = {
                        "chunks": len(chunks),
                        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "status": "✅ Indexed"
                    }
                    overall_progress.progress((i + 1) / len(uploaded_files))
                    
                except Exception as e:
                    st.error(f"Error indexing {file.name}: {str(e)}")
                    st.session_state.files_metadata[file.name] = {"chunks": 0, "date": "-", "status": "❌ Failed"}
            
            status_area.success("🎉 All files processed!")

# --- TAB 2: CHAT INTERFACE ---
with tab2:
    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg:
                with st.expander("View Sources"):
                    for s in msg["sources"]:
                        st.caption(f"**{s['source']}** (Score: {s['score']:.2f})")
                        st.write(s['text'])

    if prompt := st.chat_input("Ask WOLF anything..."):
        st.session_state.chat.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Analyzing knowledge base..."):
                try:
                    q_dense = openai_client.embeddings.create(model=EMBED_MODEL, input=prompt).data[0].embedding
                    q_sparse = generate_sparse_vector(prompt)
                    
                    try:
                        res = index.query(vector=q_dense, sparse_vector=q_sparse, top_k=top_k, include_metadata=True)
                    except:
                        res = index.query(sparse_vector=q_sparse, top_k=top_k, include_metadata=True)
                    
                    if res.matches:
                        context = "\n\n".join([f"Source: {m.metadata['source']}\nContent: {m.metadata['text']}" for m in res.matches])
                        sources = [{"source": m.metadata['source'], "text": m.metadata['text'], "score": m.score} for m in res.matches]
                        
                        response = openai_client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[
                                {"role": "system", "content": "You are WOLF, a professional RAG assistant. Answer based on context. Cite sources."},
                                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {prompt}"}
                            ]
                        ).choices[0].message.content
                        
                        st.markdown(response)
                        with st.expander("View Sources"):
                            for s in sources:
                                st.caption(f"**{s['source']}** (Score: {s['score']:.2f})")
                                st.write(s['text'])
                        
                        st.session_state.chat.append({"role": "assistant", "content": response, "sources": sources})
                    else:
                        st.warning("No relevant information found in the index.")
                except Exception as e:
                    st.error(f"Chat Error: {e}")

# --- TAB 3: FILE MANAGER ---
with tab3:
    st.subheader("Knowledge Base Management")
    
    if not st.session_state.files_metadata:
        st.info("No files indexed yet. Go to 'Smart Ingest' to add documents.")
    else:
        # Convert metadata to DataFrame for better viewing
        df_data = []
        for name, meta in st.session_state.files_metadata.items():
            df_data.append({
                "File Name": name,
                "Chunks": meta["chunks"],
                "Date Added": meta["date"],
                "Status": meta["status"]
            })
        
        df = pd.DataFrame(df_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        st.divider()
        col_a, col_b = st.columns(2)
        
        with col_a:
            file_to_delete = st.selectbox("Select file to remove", options=[""] + list(st.session_state.files_metadata.keys()))
            if file_to_delete and st.button(f"🗑️ Delete {file_to_delete}", use_container_width=True):
                try:
                    index.delete(filter={"source": {"$eq": file_to_delete}})
                    del st.session_state.files_metadata[file_to_delete]
                    st.success(f"Removed {file_to_delete}")
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"Delete failed: {e}")
                    
        with col_b:
            st.write("### Index Stats")
            try:
                stats = index.describe_index_stats()
                st.json(stats.to_dict())
                if st.button("⚠️ Wipe Entire Index", type="primary", use_container_width=True):
                    index.delete(delete_all=True)
                    st.session_state.files_metadata = {}
                    st.success("Index wiped clean.")
                    time.sleep(1)
                    st.rerun()
            except Exception as e:
                st.error(f"Stats error: {e}")

st.divider()
st.caption("🐺 WOLF RAG Intelligence • Ultimate Edition • v3.0")
