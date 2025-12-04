import os
import re
import time
from pathlib import Path
from typing import List, Optional, Tuple, Set

import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import LlamaCpp
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document

# ------------------------
# CONFIGURATION
# ------------------------
class Config:
    DB_PATH = "faiss_index"
    GGUF_PATH = "model/tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf"
    NOTES_DIR = "Obsidian"
    MAX_FILE_CHARS = 16000
    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".svg"}
    TEXT_EXTS = {".md", ".markdown", ".txt", ".rst", ".html"}
    PDF_EXTS = {".pdf"}
    
    PROMPT_TEMPLATE = """
You are a helpful assistant answering user questions based on their personal study notes.

Use only the information provided in the context below. Do not generate or assume any information that is not explicitly present in the context.

If the answer is not found in the notes, say:
"The information is not available in the current notes."

Format your response in a clear and organized manner using markdown when appropriate.

--------------------
User's Notes:
{context}
--------------------

User's Question: {question}

Answer:"""

config = Config()

# ------------------------
# UTILITIES
# ------------------------
@st.cache_data(show_spinner=False, max_entries=100)
def read_text_file(path: str, limit_chars: int = config.MAX_FILE_CHARS) -> str:
    """Read text file with character limit and error handling."""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()[:limit_chars]
    except Exception as e:  
        st.error(f"Error reading file {path}: {str(e)}")
        return ""

@st.cache_data(show_spinner=False, max_entries=50)
def extract_pdf_text(path: str, limit_chars: int = config.MAX_FILE_CHARS) -> Optional[str]:
    """Extract text from PDF with character limit."""
    try:
        import PyPDF2
    except ImportError:
        st.warning("PyPDF2 is not installed. Install it with: pip install PyPDF2")
        return None

    try:
        text_parts = []
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                t = page.extract_text() or ""
                text_parts.append(t)
                if sum(len(s) for s in text_parts) > limit_chars:
                    break
        return "".join(text_parts)[:limit_chars]
    except Exception as e:
        st.error(f"Error reading PDF {path}: {str(e)}")
        return None

@st.cache_data(show_spinner=False, max_entries=500, ttl=60)  # Cache for 60 seconds
def find_file_in_notes(fname: str) -> Optional[str]:
    """Find a file in the notes directory with flexible matching."""
    target = fname.lower()
    best_match = None
    best_contains = None

    # Add a cache invalidation mechanism
    cache_key = f"file_search_{target}_{time.time() // 300}"  # Invalidate every 5 minutes
    
    for root, _, files in os.walk(config.NOTES_DIR):
        for f in files:
            lower_f = f.lower()
            full_path = os.path.join(root, f)
            if lower_f == target:
                return full_path
            if target in lower_f and best_contains is None:
                best_contains = full_path
            if Path(f).stem.lower() == Path(target).stem.lower() and best_match is None:
                best_match = full_path
    return best_contains or best_match

def extract_filename(query: str) -> Optional[str]:
    """Extract a filename from a query string using multiple patterns."""
    # Pattern 1: Quoted filenames
    quoted_patterns = [r"['\"]([^'\"]+\.(?:md|markdown|txt|pdf|rst|html|png|jpg|jpeg|gif|bmp|webp|tiff|svg))['\"]"]
    for pattern in quoted_patterns:
        matches = re.findall(pattern, query, flags=re.IGNORECASE)
        if matches:
            return matches[0]

    # Pattern 2: Sentence patterns
    sentence_patterns = [
        r"(?:explain\s+what\s+is\s+in|summarize|about)\s+([a-zA-Z0-9_\-\s]+\.(?:md|markdown|txt|pdf|rst|html|png|jpg|jpeg|gif|bmp|webp|tiff|svg))",
        r"what\s+is\s+in\s+(?:the\s+)?([a-zA-Z0-9_\-\s]+\.(?:md|markdown|txt|pdf|rst|html|png|jpg|jpeg|gif|bmp|webp|tiff|svg))",
        r"(?:analyze|read|open|show|display)\s+(?:the\s+)?([a-zA-Z0-9_\-\s]+\.(?:md|markdown|txt|pdf|rst|html|png|jpg|jpeg|gif|bmp|webp|tiff|svg))",
        r"tell\s+me\s+about\s+([a-zA-Z0-9_\-\s]+\.(?:md|markdown|txt|pdf|rst|html|png|jpg|jpeg|gif|bmp|webp|tiff|svg))",
    ]
    
    for pattern in sentence_patterns:
        matches = re.findall(pattern, query, flags=re.IGNORECASE)
        if matches:
            candidate = matches[0]
            filename = candidate if isinstance(candidate, str) else candidate[0] if candidate else ""
            if filename:
                return filename.strip('.,!? ')

    # Pattern 3: Standalone filename pattern
    standalone_pattern = r"\b([a-zA-Z0-9_\-\s\.]+\.(?:md|markdown|txt|pdf|rst|html|png|jpg|jpeg|gif|bmp|webp|tiff|svg))\b"
    matches = re.findall(standalone_pattern, query, flags=re.IGNORECASE)
    if matches:
        longest_filename = max(matches, key=len)
        return longest_filename.strip('.,!? ')
    
    return None

def create_file_display(file_path: str) -> str:
    """Create a clickable file display for Streamlit."""
    abs_path = os.path.abspath(file_path)
    file_name = os.path.basename(file_path)
    
    # Create a clickable link that opens the file in default application
    if os.path.exists(abs_path):
        return f'📁 {file_name}\n   `{abs_path}`'
    else:
        return f'📁 {file_name}\n   `{abs_path}` (file not found)'

def parse_count_query(query: str) -> Tuple[Optional[List[str]], Optional[str]]:
    """Parse count queries to determine what file types to count."""
    lq = query.lower()
    if not any(keyword in lq for keyword in ["how many", "count of", "number of"]):
        return None, None
        
    type_mappings = {
        "image": (config.IMAGE_EXTS, "image files"),
        "images": (config.IMAGE_EXTS, "image files"),
        "pics": (config.IMAGE_EXTS, "image files"),
        "pictures": (config.IMAGE_EXTS, "image files"),
        "markdown": ({".md", ".markdown"}, "Markdown files"),
        "md files": ({".md"}, "Markdown (.md) files"),
        "pdf": (config.PDF_EXTS, "PDF files"),
        "text": (config.TEXT_EXTS, "text files"),
        "txt": ({".txt"}, "text (.txt) files"),
    }
    
    for keyword, (exts, label) in type_mappings.items():
        if keyword in lq:
            return list(exts), label
            
    # Look for file extensions in the query
    exts = {f".{m.lower()}" for m in re.findall(r"\.(\w+)", lq)}
    if exts:
        label = ", ".join(sorted(exts))
        return sorted(exts), f"{label} files"
        
    # Check for common file type keywords
    for token in ["pdf", "png", "jpg", "jpeg", "txt", "md", "html", "svg"]:
        if token in lq:
            return [f".{token}"], f"{token.upper()} files"
            
    return None, None

def count_files(extension_list: List[str]) -> int:
    """Count files without caching to get real-time results"""
    exts = set(e.lower() for e in extension_list)
    total = 0
    for root, _, files in os.walk(config.NOTES_DIR):
        for f in files:
            if Path(f).suffix.lower() in exts:
                total += 1
    return total

# Remove the @st.cache_data decorator from this function
    """Count files with given extensions in the notes directory."""
    exts = set(e.lower() for e in extension_list)
    total = 0
    for root, _, files in os.walk(config.NOTES_DIR):
        for f in files:
            if Path(f).suffix.lower() in exts:
                total += 1
    return total

# ------------------------
# VECTOR STORE & LLM SETUP
# ------------------------
@st.cache_resource
def load_faiss_db():
    """Load the FAISS vector database."""
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        return FAISS.load_local(config.DB_PATH, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"Failed to load FAISS database: {str(e)}")
        st.info("Make sure the FAISS index exists at the specified path.")
        return None

@st.cache_resource
def get_llm_model():
    """Load the LLM model with appropriate configuration."""
    try:
        return LlamaCpp(
            model_path=config.GGUF_PATH,
            n_ctx=1024,          # Slightly reduced context
            max_tokens=256,      # Reduced output length
            temperature=0.1,
            top_k=20,            # Reduced sampling pool
            top_p=0.9,
            n_threads=max(4, os.cpu_count() or 4),  # Use all available cores
            n_batch=512,
            use_mlock=True,
            use_mmap=True,
            verbose=False,
            # CPU-specific optimizations
            flash_attn=False,    # Disable flash attention (CPU doesn't benefit)
            main_gpu=0,          # Explicitly use CPU
        )
        # llm = LlamaCpp(
#     model_path=GGUF_PATH,
#     n_ctx=1024,
#     temperature=0.1,
#     top_k=40,
#     top_p=0.95,
#     max_tokens=128,
#     verbose=False,
#     streaming=True,
#     callbacks=[StreamingStdOutCallbackHandler()]
# )
    except Exception as e:
        st.error(f"Failed to load LLM model: {str(e)}")
        return None

# ------------------------
# SIMPLE CUSTOM RETRIEVER
# ------------------------
class SimpleCustomRetriever:
    """Simple retriever that doesn't inherit from BaseRetriever"""
    
    def __init__(self, docs: List[Document]):
        self.docs = docs
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        return self.docs

# ------------------------
# STREAM HANDLER
# ------------------------
class StreamHandler(BaseCallbackHandler):
    """Callback handler for streaming LLM responses."""
    
    def __init__(self, container):
        self.container = container
        self.text = ""
        self.start_time = time.time()
        
    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text + "▌")
        
    def on_llm_end(self, *args, **kwargs):
        self.container.markdown(self.text)
        elapsed = time.time() - self.start_time
        st.sidebar.metric("Response time", f"{elapsed:.2f}s")

def run_llm_streamed_prompt(formatted_prompt: str, container) -> str:
    """Run LLM with streaming response."""
    handler = StreamHandler(container)
    llm = get_llm_model()
    if llm is None:
        container.markdown("**Error**: Could not load the language model.")
        return ""
    llm.callbacks = [handler]
    llm.streaming = True
    try:
        llm.invoke(formatted_prompt)
        return handler.text
    except Exception as e:
        container.markdown(f"**Error during generation**: {str(e)}")
        return ""

def get_enhanced_documents(db, query: str, k: int = 3) -> List[Document]:  # Reduced from 6 to 3
    """Get enhanced documents with markdown prioritization."""
    if db is None:
        return []
    
    retriever = db.as_retriever(search_kwargs={"k": k * 2})  # Reduced from k*2
    docs = retriever.get_relevant_documents(query)
    
    # Check if it's a programming-related query
    programming_keywords = ["code", "python", "function", "program", "script", "algorithm", "date", "datetime", "import", "def", "class"]
    is_programming_query = any(kw in query.lower() for kw in programming_keywords)
    
    if is_programming_query:
        # Prioritize markdown files for programming queries
        md_docs = [d for d in docs if d.metadata.get("source", "").endswith((".md", ".markdown"))]
        other_docs = [d for d in docs if not d.metadata.get("source", "").endswith((".md", ".markdown"))]
        
        # Return a mix favoring markdown files
        if md_docs:
            return md_docs[:k//2 + 1] + other_docs[:k//2]
        else:
            return docs[:k]
    
    return docs[:k]

# ------------------------
# STREAMLIT APP LOGIC
# ------------------------
def setup_sidebar():
    """Setup the sidebar content."""
    with st.sidebar:
        st.title("🔧 Settings")
        st.subheader("Model Information")
        if os.path.exists(config.GGUF_PATH):
            size_gb = os.path.getsize(config.GGUF_PATH) / (1024**3)
            st.write(f"Model: {Path(config.GGUF_PATH).name}")
            st.write(f"Size: {size_gb:.2f} GB")
        else:
            st.error("Model file not found!")

        st.subheader("Database Status")
        db = load_faiss_db()
        if db:
            st.success("FAISS database loaded")
            try:
                st.write(f"Documents: {db.index.ntotal}")
            except:
                st.write("Database loaded successfully")
        else:
            st.error("Database not loaded")

        st.subheader("Notes Statistics")
        if os.path.exists(config.NOTES_DIR):
            # Use a unique key for the refresh button
            if st.button("🔄 Refresh File Counts", key="refresh_file_counts"):
                # Clear cache for count functions
                st.cache_data.clear()
                
            total_files = sum(len(files) for _, _, files in os.walk(config.NOTES_DIR))
            st.write(f"Total files: {total_files}")
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Markdown", count_files([".md", ".markdown"]))
                st.metric("PDFs", count_files([".pdf"]))
            with c2:
                st.metric("Images", count_files(list(config.IMAGE_EXTS)))
                st.metric("Text", count_files([".txt"]))
                
            # Show last update time
            st.caption(f"Last updated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            st.error("Notes directory not found!")

        if st.button("🧹 Clear Chat", use_container_width=True):
            st.session_state["messages"] = []
            st.rerun()
    
    return db

def handle_count_query(user_input: str) -> bool:
    """Handle count queries and return True if handled."""
    exts, label = parse_count_query(user_input)
    if not exts:
        return False
        
    with st.chat_message("assistant"):
        with st.spinner(f"Counting {label}..."):
            # Count all files with the specified extensions
            total = count_files(exts)
            
            # Get ALL files, not just a sample
            all_files = []
            for root, _, files in os.walk(config.NOTES_DIR):
                for f in files:
                    file_ext = Path(f).suffix.lower()
                    if file_ext in exts:
                        all_files.append(os.path.join(root, f))
            
            md = f"**Answer:** There are **{total}** {label} in your notes.\n\n"
            
            # Display all files if there are fewer than 15, otherwise show a sample
            if len(all_files) <= 15:
                md += "**All files:**\n\n"
                for file_path in sorted(all_files):
                    md += f"{create_file_display(file_path)}\n\n"
            else:
                md += "**Sample files:**\n\n"
                for i, file_path in enumerate(sorted(all_files)):
                    if i >= 8:
                        break
                    md += f"{create_file_display(file_path)}\n\n"
                md += f"... and {len(all_files) - 8} more files.\n\n"
            
            # Add debug information
            md += f"*Looking for extensions: {', '.join(exts)}*\n"
            md += f"*Verified at: {time.strftime('%H:%M:%S')}*"
            
            # st.markdown(md)
            st.markdown(md, unsafe_allow_html=True)
            st.session_state["messages"].append({"role": "assistant", "content": md})
    
    return True

def handle_file_query(user_input: str) -> bool:
    """Handle file-specific queries and return True if handled."""
    fname = extract_filename(user_input)
    if not fname:
        return False
        
    with st.chat_message("assistant"):
        with st.spinner(f"Looking for '{fname}'..."):
            fpath = find_file_in_notes(fname)
        
        if not fpath or not os.path.isfile(fpath):
            ans = f"I couldn't find '{fname}' in your notes. Try checking the exact filename and extension."
            st.markdown(ans)
            st.session_state["messages"].append({"role": "assistant", "content": ans})
            return True
            
        ext = Path(fpath).suffix.lower()
        context_text = ""
        
        if ext in config.TEXT_EXTS:
            context_text = read_text_file(fpath)
        elif ext in config.PDF_EXTS:
            context_text = extract_pdf_text(fpath) or ""
            if not context_text:
                md = "I found the PDF, but couldn't extract text. Install PyPDF2 or convert it to text/markdown.\n\n"
                md += create_file_display(fpath)
                # st.markdown(md)
                st.markdown(md, unsafe_allow_html=True)
                st.session_state["messages"].append({"role": "assistant", "content": md})
                return True
        else:
            md = f"This file is not a plaintext note.\n\n{create_file_display(fpath)}"
            # st.markdown(md)
            st.markdown(md, unsafe_allow_html=True)
            st.session_state["messages"].append({"role": "assistant", "content": md})
            return True

        if context_text and context_text.strip():
            prompt_template = PromptTemplate(
                template=config.PROMPT_TEMPLATE,
                input_variables=["context", "question"]
            )
            formatted = prompt_template.format(context=context_text, question=user_input)
            container = st.empty()
            resp = run_llm_streamed_prompt(formatted, container)
            result_md = resp + "\n\n" + create_file_display(fpath)
            container.markdown(result_md)
            st.session_state["messages"].append({"role": "assistant", "content": result_md})
        else:
            md = f"The file is empty or I couldn't read any text from it.\n\n{create_file_display(fpath)}"
            # st.markdown(md)
            st.markdown(md, unsafe_allow_html=True)
            st.session_state["messages"].append({"role": "assistant", "content": md})
    
    return True

def handle_general_query(user_input: str, db):
    """Handle general queries using the vector database."""
    if not db:
        with st.chat_message("assistant"):
            err = "Database not available. Please check if the FAISS index exists."
            st.markdown(err)
            st.session_state["messages"].append({"role": "assistant", "content": err})
        return
        
    with st.chat_message("assistant"):
        container = st.empty()
        handler = StreamHandler(container)
        llm_stream = get_llm_model()
        
        if not llm_stream:
            err = "Language model not available. Please check the model path."
            container.markdown(err)
            st.session_state["messages"].append({"role": "assistant", "content": err})
            return
            
        llm_stream.callbacks = [handler]
        llm_stream.streaming = True
        
        # Get enhanced documents with markdown prioritization
        enhanced_docs = get_enhanced_documents(db, user_input)
        
        # Create prompt template
        prompt_template = PromptTemplate(
            template=config.PROMPT_TEMPLATE,
            input_variables=["context", "question"]
        )
        
        # Use simple custom retriever
        custom_retriever = SimpleCustomRetriever(docs=enhanced_docs)
        
        # Create the chain manually
        from langchain.chains import LLMChain
        from langchain.chains.combine_documents.stuff import StuffDocumentsChain
        
        # Create the LLM chain
        llm_chain = LLMChain(llm=llm_stream, prompt=prompt_template)
        
        # Create the document chain
        document_chain = StuffDocumentsChain(
            llm_chain=llm_chain,
            document_variable_name="context"
        )
        
        try:
            # Get documents from our custom retriever
            docs = custom_retriever.get_relevant_documents(user_input)
            
            # Format the input for the document chain
            inputs = {
                "input_documents": docs,
                "question": user_input
            }
            
            # Run the chain
            result = document_chain.run(inputs)
            
            # Extract sources
            sources_md = "\n\n📂 **Sources:**\n\n"
            seen = set()
            
            for doc in docs:
                src = doc.metadata.get("source", "")
                if src and src not in seen:
                    seen.add(src)
                    sources_md += f"{create_file_display(src)}\n\n"
            
            final_response = result + sources_md
            container.markdown(final_response)
            st.session_state["messages"].append({"role": "assistant", "content": final_response})
        except Exception as e:
            err = f"Error during retrieval: {str(e)}"
            container.markdown(err)
            st.session_state["messages"].append({"role": "assistant", "content": err})

def main():
    """Main application function."""
    st.set_page_config(
        page_title="Obsidian Notes Chatbot",
        page_icon="💬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Setup sidebar and get database
    db = setup_sidebar()
    
    st.title("💬 Obsidian Notes Chatbot")
    st.caption("Ask questions about your notes or request file summaries")

    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "Hi! I can help you explore your Obsidian notes. Ask me anything about your content or request a summary of a specific file."}
        ]

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            if msg.get("is_html", False):
                st.markdown(msg["content"], unsafe_allow_html=True)
            else:
                st.markdown(msg["content"])

    user_input = st.chat_input("Ask anything about your notes")

    if user_input:
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Handle different types of queries
        if handle_count_query(user_input):
            return
        if handle_file_query(user_input):
            return
        handle_general_query(user_input, db)

if __name__ == "__main__":
    main()