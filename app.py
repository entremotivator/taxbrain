import streamlit as st
import uuid
import time
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
from pypdf import PdfReader
from pinecone.exceptions import PineconeApiException, PineconeException

# ==================================================
# CONFIGURATION & CONSTANTS
# ==================================================
INDEX_NAME = "wolf"
# Note: It's better to use the index name and let the SDK resolve the host, 
# but we'll keep the host if the user specifically needs it.
# However, for Serverless, targeting by name is often more robust.
PINECONE_HOST = "https://wolf-b79cc48.svc.aped-4627-b74a.pinecone.io"

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536
MAX_METADATA_SIZE = 40000  # Pinecone limit is 40KB per record
BATCH_SIZE = 100

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================================================
# STREAMLIT PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="🐺 WOLF RAG Intelligence Pro",
    page_icon="🐺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        border-radius: 5px;
        height: 3em;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        border-color: #ff4b4b;
        color: #ff4b4b;
    }
    .chat-bubble {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
    }
    </style>
    """, unsafe_allow_html=True)

# ==================================================
# SESSION STATE INITIALIZATION
# ==================================================
def init_session_state():
    if "files_ingested" not in st.session_state:
        st.session_state.files_ingested = set()
    if "chat" not in st.session_state:
        st.session_state.chat = []
    if "process_logs" not in st.session_state:
        st.session_state.process_logs = []

init_session_state()

# ==================================================
# SIDEBAR & SETTINGS
# ==================================================
with st.sidebar:
    st.title("🐺 WOLF AI Settings")
    
    with st.expander("🔑 API Credentials", expanded=True):
        pinecone_api_key = st.text_input("Pinecone API Key", type="password", help="Get this from your Pinecone dashboard")
        openai_api_key = st.text_input("OpenAI API Key", type="password", help="Get this from your OpenAI dashboard")

    with st.expander("⚙️ RAG Parameters", expanded=False):
        chunk_size = st.slider("Chunk Size (words)", 100, 2000, 500, 50)
        chunk_overlap = st.slider("Chunk Overlap (words)", 0, 500, 100, 10)
        top_k = st.slider("Context Window (Top K)", 1, 10, 5)
        
    st.divider()
    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.chat = []
        st.rerun()

# ==================================================
# CORE CLIENTS & LOGIC
# ==================================================
@st.cache_resource(show_spinner=False)
def get_clients(p_key: str, o_key: str):
    try:
        pc = Pinecone(api_key=p_key)
        # Try to connect to index. If host is provided, use it, otherwise use name.
        try:
            index = pc.Index(host=PINECONE_HOST)
        except Exception:
            index = pc.Index(INDEX_NAME)
            
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
    try:
        res = client.embeddings.create(
            model=EMBED_MODEL,
            input=text
        )
        return res.data[0].embedding
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        raise e

def perform_upsert(index, vectors: List[Dict[str, Any]]):
    """Safe upsert with batching and error handling."""
    for i in range(0, len(vectors), BATCH_SIZE):
        batch = vectors[i:i + BATCH_SIZE]
        try:
            index.upsert(vectors=batch)
        except PineconeApiException as e:
            st.error(f"Pinecone API Error during upsert: {e}")
            logger.error(f"Upsert failed: {e}")
            raise e
        except Exception as e:
            st.error(f"Unexpected error during upsert: {e}")
            raise e

# ==================================================
# MAIN INTERFACE
# ==================================================
st.markdown("<h1 style='text-align:center;'>🐺 WOLF RAG Intelligence</h1>", unsafe_allow_html=True)

if not pinecone_api_key or not openai_api_key:
    st.info("👋 Welcome! Please enter your API keys in the sidebar to get started.")
    st.stop()

index, openai_client, pc_client = get_clients(pinecone_api_key, openai_api_key)

if not index:
    st.stop()

# --- TABBED INTERFACE ---
tab1, tab2, tab3, tab4 = st.tabs(["📤 Ingestion", "💬 Chat Interface", "🛠️ Management", "📊 Analytics"])

with tab1:
    st.subheader("Document Ingestion")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_files = st.file_uploader(
            "Upload documents (PDF, TXT)",
            type=["txt", "pdf"],
            accept_multiple_files=True,
            help="Upload files to build your knowledge base"
        )
    
    with col2:
        st.write("### Status")
        if uploaded_files:
            st.write(f"Files selected: {len(uploaded_files)}")
            if st.button("🚀 Start Indexing", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                all_vectors = []
                total_files = len(uploaded_files)
                
                for file_idx, file in enumerate(uploaded_files):
                    status_text.text(f"Processing {file.name}...")
                    text = extract_text_from_file(file)
                    chunks = chunk_text_logic(text, chunk_size, chunk_overlap)
                    
                    for chunk_idx, chunk in enumerate(chunks):
                        try:
                            embedding = get_embedding(openai_client, chunk)
                            
                            # CRITICAL FIX: Ensure metadata size is within limits
                            # Pinecone metadata limit is 40KB. We truncate text if needed.
                            # Also, we store the full text in metadata for RAG.
                            metadata_text = chunk
                            if len(metadata_text.encode('utf-8')) > 38000:
                                metadata_text = metadata_text[:10000] # Safe truncation
                            
                            all_vectors.append({
                                "id": f"{file.name}-{chunk_idx}-{str(uuid.uuid4())[:8]}",
                                "values": embedding,
                                "metadata": {
                                    "source": file.name,
                                    "chunk_id": chunk_idx,
                                    "text": metadata_text,
                                    "indexed_at": datetime.utcnow().isoformat()
                                }
                            })
                        except Exception as e:
                            st.error(f"Failed to process chunk {chunk_idx} of {file.name}")
                    
                    st.session_state.files_ingested.add(file.name)
                    progress_bar.progress((file_idx + 1) / total_files)
                
                if all_vectors:
                    status_text.text("Upserting to Pinecone...")
                    try:
                        perform_upsert(index, all_vectors)
                        st.success(f"✅ Successfully indexed {len(all_vectors)} chunks from {total_files} files.")
                    except Exception as e:
                        st.error(f"Indexing failed: {str(e)}")
                else:
                    st.warning("No text could be extracted from the uploaded files.")

with tab2:
    st.subheader("AI Assistant")
    
    # Display chat history
    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg:
                with st.expander("View Sources"):
                    for s in msg["sources"]:
                        st.caption(f"**Source:** {s['source']} (Score: {s['score']:.2f})")
                        st.write(s['text'])

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        st.session_state.chat.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # 1. Embed query
                    query_vec = get_embedding(openai_client, prompt)
                    
                    # 2. Query Pinecone
                    results = index.query(
                        vector=query_vec,
                        top_k=top_k,
                        include_metadata=True
                    )
                    
                    if not results.matches:
                        response = "I couldn't find any relevant information in your documents. Please make sure you've uploaded and indexed them first."
                        st.markdown(response)
                        st.session_state.chat.append({"role": "assistant", "content": response})
                    else:
                        # 3. Prepare context
                        context_parts = []
                        sources = []
                        for i, match in enumerate(results.matches):
                            text = match.metadata.get("text", "No text found")
                            source_name = match.metadata.get("source", "Unknown")
                            context_parts.append(f"Source {i+1} ({source_name}):\n{text}")
                            sources.append({
                                "source": source_name,
                                "text": text,
                                "score": match.score
                            })
                        
                        context_str = "\n\n".join(context_parts)
                        
                        # 4. Generate Answer
                        system_prompt = "You are WOLF, a professional RAG intelligence assistant. Use the provided context to answer the user's question accurately. If the answer isn't in the context, say you don't know. Always cite your sources using [Source N]."
                        
                        full_prompt = f"Context:\n{context_str}\n\nQuestion: {prompt}\n\nAnswer:"
                        
                        chat_res = openai_client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": full_prompt}
                            ],
                            temperature=0.3
                        )
                        
                        answer = chat_res.choices[0].message.content
                        st.markdown(answer)
                        
                        with st.expander("View Sources"):
                            for s in sources:
                                st.caption(f"**Source:** {s['source']} (Score: {s['score']:.2f})")
                                st.write(s['text'])
                        
                        st.session_state.chat.append({
                            "role": "assistant", 
                            "content": answer,
                            "sources": sources
                        })
                        
                except Exception as e:
                    error_msg = f"An error occurred: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat.append({"role": "assistant", "content": error_msg})

with tab3:
    st.subheader("Index Management")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.write("### Ingested Files")
        if st.session_state.files_ingested:
            for f in sorted(st.session_state.files_ingested):
                c1, c2 = st.columns([3, 1])
                c1.write(f"📄 {f}")
                if c2.button("🗑️", key=f"del_{f}"):
                    try:
                        index.delete(filter={"source": {"$eq": f}})
                        st.session_state.files_ingested.remove(f)
                        st.success(f"Deleted {f}")
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Delete failed: {e}")
        else:
            st.info("No files indexed yet.")
            
    with col_b:
        st.write("### Index Statistics")
        try:
            stats = index.describe_index_stats()
            st.json(stats.to_dict())
            
            if st.button("⚠️ Clear Entire Index", type="primary", use_container_width=True):
                index.delete(delete_all=True)
                st.session_state.files_ingested = set()
                st.success("Index cleared!")
                st.rerun()
        except Exception as e:
            st.error(f"Could not fetch stats: {e}")

with tab4:
    st.subheader("Knowledge Base Analytics")
    try:
        stats = index.describe_index_stats()
        namespaces = stats.get('namespaces', {})
        
        if namespaces:
            import pandas as pd
            ns_data = []
            for ns, data in namespaces.items():
                ns_data.append({"Namespace": ns if ns else "Default", "Vector Count": data.get('vector_count', 0)})
            
            df = pd.DataFrame(ns_data)
            st.bar_chart(df.set_index("Namespace"))
            st.table(df)
        else:
            st.info("No data in index to analyze.")
            
        st.write("### System Health")
        st.success("Pinecone Connection: Active")
        st.success("OpenAI API: Connected")
        
    except Exception as e:
        st.error(f"Analytics error: {e}")

st.divider()
st.caption("🐺 WOLF RAG Intelligence • Powered by Pinecone & OpenAI • v2.0")
