import streamlit as st
import uuid
from datetime import datetime
from pinecone import Pinecone
from pypdf import PdfReader
from openai import OpenAI
import time

# --------------------------------------------------
# Load Configuration from Streamlit Secrets
# --------------------------------------------------
# API keys will be entered in the sidebar
PINECONE_API_KEY = None
OPENAI_API_KEY = None
INDEX_NAME = "quickstart"

# --------------------------------------------------
# Initialize Clients with Error Handling
# --------------------------------------------------
@st.cache_resource
def initialize_clients(_pinecone_key, _openai_key, _index_name):
    """Initialize and cache Pinecone and OpenAI clients"""
    try:
        if not _pinecone_key or not _openai_key:
            return None, None, None
        
        pc = Pinecone(api_key=_pinecone_key)
        index = pc.Index(_index_name)
        openai_client = OpenAI(api_key=_openai_key)
        return pc, index, openai_client
    except Exception as e:
        st.error(f"❌ Failed to initialize clients: {str(e)}")
        return None, None, None

# --------------------------------------------------
# Helper Functions
# --------------------------------------------------
def extract_text(file):
    """Extract text from uploaded files with error handling"""
    try:
        if file.type == "text/plain":
            return file.read().decode("utf-8")
        elif file.type == "application/pdf":
            reader = PdfReader(file)
            text = ""
            for page_num, page in enumerate(reader.pages, 1):
                page_text = page.extract_text()
                if page_text:
                    text += f"\n--- Page {page_num} ---\n{page_text}"
            return text
        else:
            st.warning(f"⚠️ Unsupported file type: {file.type}")
            return ""
    except Exception as e:
        st.error(f"❌ Error extracting text from {file.name}: {str(e)}")
        return ""

def chunk_text(text, chunk_size=800, overlap=100):
    """Split text into overlapping chunks for better context preservation"""
    if not text.strip():
        return []
    
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    
    return chunks

def embed_text(text):
    """Generate embeddings using OpenAI with retry logic"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(1)

def format_file_size(size_bytes):
    """Format file size in human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"

# --------------------------------------------------
# Streamlit UI Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Pinecone Document Manager",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Sidebar Configuration
# --------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Keys Input
    st.subheader("🔑 API Keys")
    
    PINECONE_API_KEY = st.text_input(
        "Pinecone API Key",
        type="password",
        help="Enter your Pinecone API key",
        key="pinecone_key"
    )
    
    OPENAI_API_KEY = st.text_input(
        "OpenAI API Key",
        type="password",
        help="Enter your OpenAI API key",
        key="openai_key"
    )
    
    INDEX_NAME = st.text_input(
        "Pinecone Index Name",
        value="quickstart",
        help="Enter your Pinecone index name",
        key="index_name"
    )
    
    # Validation
    if not PINECONE_API_KEY or not OPENAI_API_KEY:
        st.warning("⚠️ Please enter both API keys to continue")
    else:
        st.success("✅ API keys configured")
    
    # Pinecone Version Info
    st.subheader("🌲 Pinecone Info")
    try:
        import pinecone
        pinecone_version = pinecone.__version__
        st.info(f"**Version:** `{pinecone_version}`")
    except:
        st.info("**Version:** Unable to detect")
    
    st.divider()
    
    # Initialize clients with sidebar inputs
    if PINECONE_API_KEY and OPENAI_API_KEY:
        pc, index, openai_client = initialize_clients(PINECONE_API_KEY, OPENAI_API_KEY, INDEX_NAME)
    else:
        pc, index, openai_client = None, None, None
    
    st.subheader("Processing Options")
    chunk_size = st.slider("Chunk Size (words)", 200, 2000, 800, 100)
    overlap = st.slider("Chunk Overlap (words)", 0, 500, 100, 50)
    
    st.subheader("Upload Statistics")
    if "upload_history" not in st.session_state:
        st.session_state.upload_history = []
    
    if st.session_state.upload_history:
        total_uploads = len(st.session_state.upload_history)
        total_vectors = sum(h["vectors"] for h in st.session_state.upload_history)
        st.metric("Total Uploads", total_uploads)
        st.metric("Total Vectors", total_vectors)
    else:
        st.write("No uploads yet")
    
    if st.button("🗑️ Clear History"):
        st.session_state.upload_history = []
        st.rerun()

# --------------------------------------------------
# Main Content
# --------------------------------------------------
st.markdown("<h1 class='main-header'>📚 Pinecone Document Manager</h1>", unsafe_allow_html=True)

st.markdown("""
Transform your documents into searchable vector embeddings with this powerful tool. Upload TXT or PDF files, 
and they'll be automatically processed, chunked, embedded, and stored in your Pinecone index for AI-powered 
retrieval, RAG systems, chatbots, and semantic search applications.
""")

# --------------------------------------------------
# Upload Section
# --------------------------------------------------
st.header("📤 Upload & Process Documents")

col1, col2 = st.columns([2, 1])

with col1:
    uploaded_files = st.file_uploader(
        "Choose TXT or PDF files to upload",
        type=["txt", "pdf"],
        accept_multiple_files=True,
        help="Upload one or more documents to process and index"
    )

with col2:
    if uploaded_files:
        st.metric("Files Selected", len(uploaded_files))
        total_size = sum(f.size for f in uploaded_files)
        st.metric("Total Size", format_file_size(total_size))

if uploaded_files:
    # Check if API keys are configured
    if not PINECONE_API_KEY or not OPENAI_API_KEY:
        st.error("⚠️ Please enter your API keys in the sidebar to continue")
        st.stop()
    
    if not pc or not index or not openai_client:
        st.error("❌ Failed to initialize clients. Please check your API keys.")
        st.stop()
    
    # Display file preview
    with st.expander("📋 View Selected Files", expanded=False):
        for file in uploaded_files:
            col_a, col_b, col_c = st.columns([3, 2, 1])
            with col_a:
                st.write(f"📄 {file.name}")
            with col_b:
                st.write(f"Type: {file.type}")
            with col_c:
                st.write(format_file_size(file.size))
    
    # Process button
    if st.button("🚀 Process & Upload to Pinecone", type="primary", use_container_width=True):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            vectors = []
            file_stats = {}
            
            for idx, file in enumerate(uploaded_files):
                status_text.text(f"Processing {file.name}...")
                progress_bar.progress((idx) / len(uploaded_files))
                
                # Extract text
                raw_text = extract_text(file)
                if not raw_text:
                    continue
                
                # Create chunks
                chunks = chunk_text(raw_text, chunk_size=chunk_size, overlap=overlap)
                file_stats[file.name] = {
                    "chunks": len(chunks),
                    "characters": len(raw_text)
                }
                
                # Generate embeddings and prepare vectors
                for chunk_idx, chunk in enumerate(chunks):
                    try:
                        embedding = embed_text(chunk)
                        vector_id = str(uuid.uuid4())
                        
                        vectors.append({
                            "id": vector_id,
                            "values": embedding,
                            "metadata": {
                                "source": file.name,
                                "chunk_index": chunk_idx,
                                "total_chunks": len(chunks),
                                "text": chunk[:1000],  # Store first 1000 chars
                                "timestamp": datetime.now().isoformat(),
                                "file_type": file.type
                            }
                        })
                    except Exception as e:
                        st.warning(f"⚠️ Failed to process chunk {chunk_idx} from {file.name}: {str(e)}")
            
            # Upload to Pinecone
            if vectors:
                status_text.text("Uploading vectors to Pinecone...")
                progress_bar.progress(0.95)
                
                # Batch upsert (Pinecone handles up to 100 vectors per batch)
                batch_size = 100
                for i in range(0, len(vectors), batch_size):
                    batch = vectors[i:i + batch_size]
                    index.upsert(vectors=batch)
                
                progress_bar.progress(1.0)
                status_text.empty()
                
                # Record upload history
                st.session_state.upload_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "files": len(uploaded_files),
                    "vectors": len(vectors)
                })
                
                # Success message with details
                st.markdown(f"""
                <div class='success-box'>
                    <h3>✅ Upload Complete!</h3>
                    <p><strong>{len(vectors)}</strong> vectors successfully uploaded to Pinecone</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Show detailed statistics
                with st.expander("📊 Processing Details", expanded=True):
                    for filename, stats in file_stats.items():
                        st.write(f"**{filename}**")
                        col_x, col_y = st.columns(2)
                        with col_x:
                            st.metric("Chunks Created", stats["chunks"])
                        with col_y:
                            st.metric("Characters", f"{stats['characters']:,}")
                        st.divider()
            else:
                st.warning("⚠️ No vectors were created. Please check your files.")
                
        except Exception as e:
            st.error(f"❌ An error occurred during processing: {str(e)}")
        finally:
            progress_bar.empty()
            status_text.empty()

# --------------------------------------------------
# Index Viewer Section
# --------------------------------------------------
st.header("📊 Index Information")

col1, col2 = st.columns([1, 1])

with col1:
    if st.button("🔄 Refresh Index Stats", use_container_width=True):
    if not pc or not index or not openai_client:
        st.error("⚠️ Please configure API keys in the sidebar first")
    else:
        try:
            with st.spinner("Fetching index statistics..."):
                stats = index.describe_index_stats()
                
                # Display metrics
                metric_cols = st.columns(3)
                with metric_cols[0]:
                    st.metric("Total Vectors", f"{stats.get('total_vector_count', 0):,}")
                with metric_cols[1]:
                    st.metric("Dimension", stats.get('dimension', 'N/A'))
                with metric_cols[2]:
                    namespaces = stats.get('namespaces', {})
                    st.metric("Namespaces", len(namespaces))
                
                # Show detailed stats
                with st.expander("🔍 Detailed Statistics", expanded=False):
                    st.json(stats)
        except Exception as e:
            st.error(f"❌ Failed to fetch index stats: {str(e)}")

with col2:
    if st.button("🗑️ Clear Namespace (Careful!)", use_container_width=True):
        if not pc or not index or not openai_client:
            st.error("⚠️ Please configure API keys in the sidebar first")
        elif st.checkbox("I understand this will delete all vectors"):
            try:
                index.delete(delete_all=True)
                st.success("✅ Namespace cleared successfully")
            except Exception as e:
                st.error(f"❌ Failed to clear namespace: {str(e)}")

# --------------------------------------------------
# Query Testing Section
# --------------------------------------------------
st.header("🔍 Test Semantic Search")

query_text = st.text_input(
    "Enter your search query",
    placeholder="e.g., What are the main findings about climate change?",
    help="Search your indexed documents using natural language"
)

col1, col2 = st.columns([3, 1])
with col1:
    top_k = st.slider("Number of results", 1, 20, 5)
with col2:
    include_scores = st.checkbox("Show similarity scores", value=True)

if query_text:
    if st.button("🔍 Search", type="primary", use_container_width=True):
        if not pc or not index or not openai_client:
            st.error("⚠️ Please configure API keys in the sidebar first")
        else:
            try:
            with st.spinner("Searching..."):
                query_embedding = embed_text(query_text)
                results = index.query(
                    vector=query_embedding,
                    top_k=top_k,
                    include_metadata=True
                )
                
                if results["matches"]:
                    st.success(f"Found {len(results['matches'])} results")
                    
                    for idx, match in enumerate(results["matches"], 1):
                        score = match['score']
                        metadata = match.get('metadata', {})
                        
                        # Color code by relevance
                        if score > 0.8:
                            relevance_color = "🟢"
                        elif score > 0.6:
                            relevance_color = "🟡"
                        else:
                            relevance_color = "🔴"
                        
                        with st.container():
                            col_a, col_b = st.columns([4, 1])
                            with col_a:
                                st.subheader(f"{relevance_color} Result #{idx}")
                            with col_b:
                                if include_scores:
                                    st.metric("Score", f"{score:.4f}")
                            
                            st.markdown(f"**Source:** `{metadata.get('source', 'Unknown')}`")
                            
                            if metadata.get('chunk_index') is not None:
                                st.caption(f"Chunk {metadata.get('chunk_index', 0) + 1} of {metadata.get('total_chunks', 'N/A')}")
                            
                            if metadata.get('timestamp'):
                                st.caption(f"Indexed: {metadata.get('timestamp')}")
                            
                            st.markdown("**Content:**")
                            st.info(metadata.get('text', 'No text available'))
                            
                            st.divider()
                else:
                    st.warning("No results found. Try a different query or upload more documents.")
                    
        except Exception as e:
            st.error(f"❌ Search failed: {str(e)}")

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem 0;'>
    <p>Built with Streamlit, Pinecone, and OpenAI • Upload → Chunk → Embed → Search</p>
</div>
""", unsafe_allow_html=True)
