
import os
from PIL import Image
import pytesseract

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_unstructured import UnstructuredLoader
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredFileLoader,
    UnstructuredPowerPointLoader,
)
from langchain.schema import Document  # For image OCR docs

# Paths
VAULT_PATH = "Obsidian"
DB_SAVE_PATH = "faiss_index"

# Supported extensions & loaders
# loaders_map = {
#     ".md": lambda path: TextLoader(path, encoding="utf-8", errors="ignore"),
#     ".txt": lambda path: TextLoader(path, encoding="utf-8", errors="ignore"),
#     ".pdf": PyPDFLoader,
#     ".docx": UnstructuredFileLoader,
#     ".pptx": UnstructuredPowerPointLoader,
# }
loaders_map = {
    ".md": UnstructuredLoader,
    ".txt": UnstructuredLoader,
    ".docx": UnstructuredLoader,
    ".pdf": PyPDFLoader,
    ".pptx": UnstructuredPowerPointLoader,
}
image_extensions = {".jpg", ".jpeg", ".png"}  # OCR

def ingest_documents():
    docs = []

    for root, _, files in os.walk(VAULT_PATH):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            path = os.path.join(root, file)
            normalized_path = path.replace("\\", "/")

            try:
                if ext in loaders_map:
                    loader = loaders_map[ext](normalized_path)
                    loaded_docs = loader.load()

                elif ext in image_extensions:
                    # OCR for images
                    text = pytesseract.image_to_string(Image.open(normalized_path))
                    loaded_docs = [
                        Document(page_content=text, metadata={"source": normalized_path})
                    ]

                else:
                    continue  # skip unsupported files

                # Ensure all docs are Document objects
                for i, doc in enumerate(loaded_docs):
                    if isinstance(doc, dict):
                        loaded_docs[i] = Document(
                            page_content=doc.get("page_content", ""),
                            metadata=doc.get("metadata", {"source": normalized_path})
                        )

                docs.extend(loaded_docs)
                print(f"[OK] Loaded: {file}")

            except UnicodeDecodeError:
                print(f"[WARN] Skipped (encoding issue): {file}")
            except Exception as e:
                print(f"[FAIL] Could not load {file} — {e}")

    if not docs:
        print("[INFO] No documents found to process.")
        return

    # Chunk docs
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    print(f"[INFO] Split into {len(chunks)} chunks.")

    # Embeddings
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    new_vector_db = FAISS.from_documents(chunks, embedding_model)

    # Merge or save FAISS index
    if os.path.exists(DB_SAVE_PATH):
        try:
            existing_vector_db = FAISS.load_local(
                DB_SAVE_PATH, embedding_model, allow_dangerous_deserialization=True
            )
            print("[INFO] Existing FAISS index found. Merging...")
            existing_vector_db.merge_from(new_vector_db)
            existing_vector_db.save_local(DB_SAVE_PATH)
            print(f"[DONE] FAISS index updated at {DB_SAVE_PATH}")
        except Exception as e:
            print(f"[WARN] Failed to load existing index — {e}")
            new_vector_db.save_local(DB_SAVE_PATH)
            print(f"[DONE] New FAISS index saved at {DB_SAVE_PATH}")
    else:
        new_vector_db.save_local(DB_SAVE_PATH)
        print(f"[DONE] FAISS index created at {DB_SAVE_PATH}")

if __name__ == "__main__":
    ingest_documents()
