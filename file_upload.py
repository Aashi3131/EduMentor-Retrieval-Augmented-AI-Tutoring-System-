"""file_upload.py handles PDF ingestion from users. 
1. It splits each PDF into manageable chunks, 
2. embeds them using SentenceTransformer, 
3. builds a FAISS index, 
4. persists chunks and index to disk for RAG retrieval, 
5. and provides a UI to upload, view, and delete course materials."""

import streamlit as st #for all ui interactions
import os #for file system interactions
import shutil #for file system operations
import fitz #for opening pdfs
import faiss #for building the faiss index
import pickle #for pickling/unpickling the faiss index, save py objects
import numpy as np #for numpy array operations, save embeddings
from sentence_transformers import SentenceTransformer

def check_and_upload_files(api_key: str):
    if "kb_docs" not in st.session_state:
        st.session_state.kb_docs = []
    if "all_chunks" not in st.session_state:
        st.session_state.all_chunks = []

    docs = st.session_state.kb_docs
    if not docs:
        st.warning("No course materials yet. Upload PDFs so answers can use your documents.")
    else:
        st.info(f"{len(docs)} document(s) loaded in this session. You can add more below.")

    uploaded_files = st.file_uploader(
        "Choose PDF files", type="pdf", accept_multiple_files=True
    )
    if st.button("Upload PDF"):
        if not uploaded_files:
            st.warning("Select at least one PDF.")
        else:
            try:
                added = 0
                new_chunks = []
                with st.spinner("Processing..."):
                    model = SentenceTransformer("all-MiniLM-L6-v2")
                    dimension = 384
                    for uf in uploaded_files:
                        if uf is None:
                            continue
                        # Open PDF using PyMuPDF
                        doc = fitz.open(stream=uf.getvalue(), filetype="pdf")
                        full_text_parts = []
                        file_chunks = []
                        chunk_id_start = len(st.session_state.all_chunks) + len(new_chunks)
                        for page_num, page in enumerate(doc):
                            text = page.get_text()
                            full_text_parts.append(text)
                            
                            # Detect images per page
                            has_image = len(page.get_images(full=True)) > 0
                            image_description = f"Image detected on page {page_num + 1}" if has_image else ""
                            
                            # Split each page's text into word-level chunks
                            words = text.split()
                            CHUNK_SIZE = 400
                            CHUNK_OVERLAP = 80
                            
                            i = 0
                            while i < len(words):
                                chunk_words = words[i : i + CHUNK_SIZE]
                                chunk_text = " ".join(chunk_words)
                                
                                if chunk_text.strip():
                                    file_chunks.append({
                                        "text": chunk_text,
                                        "source": uf.name,
                                        "page": page_num + 1,
                                        "chunk_id": chunk_id_start + len(file_chunks),
                                        "has_image": has_image,
                                        "image_description": image_description
                                    })
                                
                                if i + CHUNK_SIZE >= len(words):
                                    break
                                i += CHUNK_SIZE - CHUNK_OVERLAP
                                
                        new_chunks.extend(file_chunks)
                        st.session_state.kb_docs.append({
                            "name": uf.name,
                            "text": "\n".join(full_text_parts).strip()
                        })
                        added += 1
                        
                    if new_chunks:
                        st.session_state.all_chunks.extend(new_chunks)
                        
                        # Generate embeddings for all chunks in the session
                        texts = [c["text"] for c in st.session_state.all_chunks]
                        embeddings = model.encode(texts, normalize_embeddings=True)#Normalize to make vector length 1, cosine similarity
                        embeddings = np.array(embeddings, dtype=np.float32)
                        
                        # Store in FAISS index (IP for Cosine similarity on normalized vectors)
                        index = faiss.IndexFlatIP(dimension)
                        index.add(embeddings)
                        
                        # Persist index and metadata
                        os.makedirs("vector_store", exist_ok=True)
                        faiss.write_index(index, "vector_store/index.faiss")
                        pickle.dump(st.session_state.all_chunks, open("vector_store/chunks.pkl", "wb"))
                        
                if added:
                    st.success(f"Added {added} PDF(s) to FAISS index.")
                    st.rerun()
            except Exception as e:
                st.error(f"Upload failed: {e}")

    return list(st.session_state.kb_docs)


def delete_all_knowledge_files():
    n = len(st.session_state.get("kb_docs", []))
    st.session_state.kb_docs = []
    st.session_state.all_chunks = []
    
    # Clear vector store directory
    if os.path.exists("vector_store"):
        try:
            shutil.rmtree("vector_store")
        except Exception:
            pass
            
    return n
