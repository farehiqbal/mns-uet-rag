import os
import warnings
import json
import numpy as np
import nltk
import fitz  # PyMuPDF
import pdfplumber  # For table extraction
import faiss
import streamlit as st
import hashlib
import re
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
import google.generativeai as genai
from dotenv import load_dotenv

# Suppress PyTorch warnings
warnings.filterwarnings("ignore", message="Examining the path of torch.classes")

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="PDF RAG Chatbot", 
    layout="wide",
    page_icon="📄"
)

# Set GEMINI API key
api_key = os.getenv("GEMINI_API_KEY")

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# 1. Extract text with improved coverage and fallback
def extract_text_from_pdfs(pdf_paths: list) -> list:
    chunks = []
    
    for pdf_path in pdf_paths:
        try:
            if not os.path.exists(pdf_path):
                st.error(f"File not found: {pdf_path}")
                continue
                
            doc = fitz.open(pdf_path)
            doc_name = os.path.basename(pdf_path)
            st.info(f"Processing {doc_name}: {len(doc)} pages")
            
            with pdfplumber.open(pdf_path) as pdf:
                total_lines_processed = 0
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    text = page.get_text("text")
                    
                    if not text.strip():
                        st.warning(f"Skipping empty page {page_num + 1} in {doc_name}")
                        continue
                    
                    # Extract tables with pdfplumber
                    plumber_page = pdf.pages[page_num]
                    tables = plumber_page.extract_tables()
                    table_idx = 0
                    if tables:
                        for table in tables:
                            if table and any(row for row in table if any(cell.strip() for cell in row)):
                                table_text = []
                                for row in table:
                                    cleaned_row = [cell.strip() if cell else "" for cell in row]
                                    if any(cleaned_row):
                                        table_text.append(" | ".join(cleaned_row))
                                if table_text:
                                    table_content = "\n".join([f"• {row}" for row in table_text])
                                    chunks.append({
                                        "text": table_content,
                                        "doc_name": doc_name,
                                        "page_num": page_num + 1,
                                        "paragraph": 0,
                                        "chunk_id": f"{doc_name}_p{page_num + 1}_table{table_idx}"
                                    })
                                    table_idx += 1
                    
                    # Extract text lines with section detection and fallback
                    lines = text.split('\n')
                    current_section = None
                    section_content = []
                    lines_processed = 0
                    
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                        lines_processed += 1
                        total_lines_processed += 1
                        # Detect section headers
                        if re.match(r'^(Faculty|Department)\s+of\s+[A-Za-z\s]+$', line):
                            if section_content:
                                chunks.append({
                                    "text": f"{current_section}\n\n" + "\n".join(section_content) if current_section else "\n".join(section_content),
                                    "doc_name": doc_name,
                                    "page_num": page_num + 1,
                                    "paragraph": len(chunks) + 1,
                                    "chunk_id": f"{doc_name}_p{page_num + 1}_par{len(chunks) + 1}"
                                })
                                section_content = []
                            current_section = line
                        elif line and (line.startswith(('Engr.', 'Dr.', 'Prof.', 'Lecturer', 'MSc.', 'Area of Interest')) or re.search(r'\b[A-Za-z]+\s+[A-Za-z]+\b', line)):
                            section_content.append(line)
                        elif section_content:
                            section_content.append(line)
                    
                    if section_content:
                        chunks.append({
                            "text": f"{current_section}\n\n" + "\n".join(section_content) if current_section else "\n".join(section_content),
                            "doc_name": doc_name,
                            "page_num": page_num + 1,
                            "paragraph": len(chunks) + 1,
                            "chunk_id": f"{doc_name}_p{page_num + 1}_par{len(chunks) + 1}"
                        })
                    # Fallback: Extract all remaining lines as individual chunks
                    if not chunks or chunks[-1]["page_num"] != page_num + 1:
                        for line in lines:
                            line = line.strip()
                            if line and len(line) > 10:  # Avoid empty or short lines
                                chunks.append({
                                    "text": line,
                                    "doc_name": doc_name,
                                    "page_num": page_num + 1,
                                    "paragraph": len(chunks) + 1,
                                    "chunk_id": f"{doc_name}_p{page_num + 1}_par{len(chunks) + 1}"
                                })
                    st.write(f"Processed {lines_processed} lines on page {page_num + 1}")
            
            st.success(f"Extracted {len(chunks)} chunks from {doc_name} (Total lines processed: {total_lines_processed})")
            doc.close()
        except Exception as e:
            st.error(f"Error processing {pdf_path}: {str(e)}")
    
    st.info(f"Total chunks extracted: {len(chunks)}")
    return chunks

# 2. Create FAISS index
def create_vector_store(chunks: list) -> tuple:
    if not chunks:
        st.error("No chunks to embed. Check PDF extraction.")
        return None, None, []
    
    model = SentenceTransformer('BAAI/bge-small-en-v1.5')
    texts = [chunk["text"] for chunk in chunks]
    
    progress_text = st.empty()
    progress_bar = st.progress(0)
    batch_size = 32
    embeddings = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    for i in range(0, len(texts), batch_size):
        batch_number = i // batch_size + 1
        progress_text.text(f"Embedding batch {batch_number}/{total_batches}...")
        progress_bar.progress(batch_number / total_batches)
        batch_texts = texts[i:i+batch_size]
        batch_embeddings = model.encode(batch_texts, show_progress_bar=False)
        embeddings.append(batch_embeddings)
    
    progress_text.text("Creating FAISS index...")
    progress_bar.progress(0.9)
    
    embeddings = np.vstack(embeddings)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    
    progress_text.empty()
    progress_bar.empty()
    st.success(f"Successfully created vector store with {len(chunks)} chunks")
    return index, model, chunks

# 3. Retrieve chunks with chunk_id-based indexing
def retrieve_chunks(query: str, index, model, chunks: list, k: int = 15) -> list:
    query_emb = model.encode([query])[0]
    faiss.normalize_L2(np.array([query_emb]))
    distances, indices = index.search(np.array([query_emb]), k)
    
    initial_results = []
    for i, idx in enumerate(indices[0]):
        if 0 <= idx < len(chunks):
            chunk = chunks[idx].copy()
            chunk["similarity"] = float(distances[0][i])
            initial_results.append(chunk)
    
    expanded_results = []
    seen_ids = set()
    for result in initial_results:
        if result["chunk_id"] not in seen_ids:
            expanded_results.append(result)
            seen_ids.add(result["chunk_id"])
            # Find the original index using chunk_id
            for idx, chunk in enumerate(chunks):
                if chunk["chunk_id"] == result["chunk_id"]:
                    current_idx = idx
                    break
            page_num = result["page_num"]
            for offset in range(-2, 3):  # Look 2 chunks before and after
                new_idx = current_idx + offset
                if 0 <= new_idx < len(chunks):
                    nearby_chunk = chunks[new_idx]
                    if nearby_chunk["chunk_id"] not in seen_ids and abs(nearby_chunk["page_num"] - page_num) <= 1:
                        nearby_chunk = nearby_chunk.copy()
                        nearby_chunk["similarity"] = result["similarity"]  # Inherit similarity
                        expanded_results.append(nearby_chunk)
                        seen_ids.add(nearby_chunk["chunk_id"])
    
    return expanded_results

# 4. Generate answer with detailed synthesis
def generate_answer(query: str, retrieved_chunks: list, api_key: str) -> dict:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    doc_pages = {}
    for chunk in retrieved_chunks:
        doc_key = chunk['doc_name']
        page_key = chunk['page_num']
        if doc_key not in doc_pages:
            doc_pages[doc_key] = {}
        if page_key not in doc_pages[doc_key]:
            doc_pages[doc_key][page_key] = []
        doc_pages[doc_key][page_key].append(chunk)
    
    context_items = []
    for doc_name, pages in doc_pages.items():
        doc_content = f"# Document: {doc_name}\n\n"
        for page_num, page_chunks in sorted(pages.items()):
            page_chunks.sort(key=lambda x: (x['paragraph'], x['chunk_id']))
            page_text = "\n".join([chunk['text'] for chunk in page_chunks])
            doc_content += f"## Page {page_num}:\n{page_text}\n\n"
        context_items.append(doc_content)
    
    context = "\n".join(context_items)
    
    prompt = f"""You are a helpful AI assistant that answers questions based on the provided documents.

Document Context:
{context}

User Question: {query}

Instructions:
1. Start with a clear, concise conclusion that directly answers the question using information from ALL relevant parts of the documents.
2. For queries about a group of people (e.g., "faculty of electrical"), list all individuals mentioned in the relevant section, including their names, titles, qualifications, and areas of interest if available.
3. For queries about a specific person (e.g., "who is Engr. Dr. Muhammad Shahzad"), provide all available details about them, including their role, faculty/department affiliation, qualifications, and areas of interest.
4. After the conclusion, provide additional details or context if necessary to support the answer.
5. At the end, list the sources used in a section called 'Sources', citing the document name and page number (e.g., "Sources: [prospectus.pdf, Page 5]").
6. Do NOT include inline citations in the main answer (e.g., avoid "[doc, Page 5]" within the text).
7. If the documents don't contain the answer, say: "I don't have enough information to answer this question."
8. Ensure all relevant individuals or items are included, even if spread across multiple pages or sections.
9. Keep the answer organized, readable, and focused on the user's query.

Your Answer:"""

    try:
        response = model.generate_content(prompt)
        referenced_sources = []
        response_text = response.text
        sources_section = ""
        if "Sources:" in response_text:
            sources_section = response_text.split("Sources:")[1].strip()
            response_text = response_text.split("Sources:")[0].strip()
        
        for chunk in retrieved_chunks:
            doc = chunk['doc_name']
            page = chunk['page_num']
            source_key = f"{doc}-p{page}"
            source_pattern = f"[{doc}, Page {page}]"
            alt_pattern = f"[{doc}, page {page}]"
            if source_pattern in sources_section or alt_pattern in sources_section:
                if not any(s["document"] == doc and s["page"] == page for s in referenced_sources):
                    referenced_sources.append({
                        "document": doc,
                        "page": page,
                        "text": chunk['text'],
                        "similarity": chunk.get('similarity', 0)
                    })
        
        return {
            "answer": response_text,
            "sources": referenced_sources
        }
    except Exception as e:
        error_message = f"Error: Failed to generate answer ({str(e)})."
        return {
            "answer": error_message,
            "sources": []
        }

# Render source references
def render_source_references(sources):
    if not sources:
        st.info("No specific sources were referenced in this answer.")
        return
    
    st.markdown("### Source References")
    doc_sources = {}
    for source in sources:
        doc = source["document"]
        if doc not in doc_sources:
            doc_sources[doc] = []
        doc_sources[doc].append(source)
    
    if len(doc_sources) > 0:
        doc_tabs = st.tabs(list(doc_sources.keys()))
        for i, (doc, tab) in enumerate(zip(doc_sources.keys(), doc_tabs)):
            with tab:
                pages = sorted(doc_sources[doc], key=lambda x: x["page"])
                for j, page_info in enumerate(pages):
                    st.markdown(f"**Page {page_info['page']} (Relevance: {page_info['similarity']:.2f})**")
                    st.text(page_info['text'])
                    unique_key = hashlib.md5(f"{doc}_{page_info['page']}_{j}_{page_info['text'][:20]}".encode()).hexdigest()
                    if st.button(f"Highlight this reference", key=unique_key):
                        st.session_state.highlight_source = f"{doc}-p{page_num}"
                        st.experimental_rerun()
                    st.divider()

# Streamlit app
def main():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "highlight_source" not in st.session_state:
        st.session_state.highlight_source = None
    if "show_sources" not in st.session_state:
        st.session_state.show_sources = {}
    
    with st.sidebar:
        st.title("📄 PDF RAG Chatbot")
        st.write("This chatbot answers questions based on the content of these documents:")
        
        pdf_paths = ["prospectus.pdf", "rules.pdf"]
        missing_files = [f for f in pdf_paths if not os.path.exists(f)]
        if missing_files:
            st.error(f"Missing PDF files: {', '.join(missing_files)}")
            st.error("Please place these PDF files in the root directory.")
        else:
            file_sizes = [f"{os.path.basename(f)}: {os.path.getsize(f) // 1024} KB" 
                        for f in pdf_paths]
            st.success("PDF files found:")
            for file_info in file_sizes:
                st.write(f"- {file_info}")
        
        st.divider()
        st.subheader("Parameters")
        k_value = st.slider("Number of chunks to retrieve", min_value=3, max_value=20, value=15)
        
        if st.button("Reprocess PDFs", type="primary"):
            st.session_state.pop("index", None)
            st.session_state.pop("model", None)
            st.session_state.pop("chunks", None)
            st.success("PDFs will be reprocessed!")
            st.experimental_rerun()
        
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.session_state.highlight_source = None
            st.session_state.show_sources = {}
            st.success("Chat history cleared!")
        
        if "chunks" in st.session_state:
            with st.expander("Dataset Stats"):
                chunks = st.session_state.chunks
                st.write(f"Total chunks: {len(chunks)}")
                docs = {}
                for chunk in chunks:
                    doc = chunk["doc_name"]
                    if doc not in docs:
                        docs[doc] = set()
                    docs[doc].add(chunk["page_num"])
                for doc, pages in docs.items():
                    st.write(f"• {doc}: {len(pages)} pages")
    
    st.title("Ask me anything about the documents 🤖")
    
    pdf_paths = ["prospectus.pdf", "rules.pdf"]  

    if "processing_done" not in st.session_state:
        progress_bar = st.progress(0)
        status_text = st.empty()
        st.session_state.processing_done = False

    if "index" not in st.session_state:
        if os.path.exists("index.faiss") and os.path.exists("chunks.json"):
            status_text.text("Loading existing index...")
            progress_bar.progress(30)
            index = faiss.read_index("index.faiss")
            progress_bar.progress(60)
            with open("chunks.json", "r") as f:
                chunks = json.load(f)
            progress_bar.progress(80)
            model = SentenceTransformer('BAAI/bge-small-en-v1.5')
            progress_bar.progress(100)
            st.success(f"Loaded existing index with {len(chunks)} chunks")
        else:
            status_text.text("Processing PDF documents...")
            progress_bar.progress(10)
            chunks = extract_text_from_pdfs(pdf_paths)
            progress_bar.progress(40)
            status_text.text("Creating vector embeddings...")
            index, model, chunks = create_vector_store(chunks)
            progress_bar.progress(70)
            status_text.text("Saving index...")
            faiss.write_index(index, "index.faiss")
            progress_bar.progress(90)
            with open("chunks.json", "w") as f:
                json.dump(chunks, f)
            progress_bar.progress(100)
        
        st.session_state.index = index
        st.session_state.model = model
        st.session_state.chunks = chunks
        st.session_state.processing_done = True
        
        status_text.empty()
        progress_bar.empty()
    else:
        index = st.session_state.index
        model = st.session_state.model
        chunks = st.session_state.chunks

    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and "answer" in message:
                answer_text = message["answer"]
                if st.session_state.highlight_source and "sources" in message:
                    highlight_key = st.session_state.highlight_source
                    doc_name, page_str = highlight_key.split("-p")
                    page_num = int(page_str)
                    patterns = [
                        f"[{doc_name}, Page {page_num}]",
                        f"[{doc_name}, page {page_num}]"
                    ]
                    highlighted_text = answer_text
                    for pattern in patterns:
                        if pattern in highlighted_text:
                            highlighted_text = highlighted_text.replace(
                                pattern, 
                                f"**:red[{pattern}]**"
                            )
                    st.markdown(highlighted_text)
                    st.session_state.highlight_source = None
                else:
                    if len(answer_text) > 1500:
                        st.markdown(answer_text[:1500] + "...")
                        with st.expander("Show full answer"):
                            st.markdown(answer_text)
                    else:
                        st.markdown(answer_text)
                
                if "sources" in message and message["sources"]:
                    source_key = f"sources_{i}"
                    if source_key not in st.session_state.show_sources:
                        st.session_state.show_sources[source_key] = False
                    if st.button("View Source References", key=f"btn_{source_key}"):
                        st.session_state.show_sources[source_key] = not st.session_state.show_sources[source_key]
                    if st.session_state.show_sources.get(source_key, False):
                        render_source_references(message["sources"])
            else:
                st.markdown(message["content"])

    if query := st.chat_input("Ask a question about the documents"):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
            
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                retrieved_chunks = retrieve_chunks(query, index, model, chunks, k=k_value)
                api_key = os.getenv("GEMINI_API_KEY")
                if not api_key:
                    st.error("Please set GOOGLE_API_KEY in the environment variables.")
                    answer = "Error: API key not found."
                    sources = []
                else:
                    response = generate_answer(query, retrieved_chunks, api_key)
                    answer = response["answer"]
                    sources = response["sources"]
                
                if len(answer) > 1500:
                    st.markdown(answer[:1500] + "...")
                    with st.expander("Show full answer"):
                        st.markdown(answer)
                else:
                    st.markdown(answer)
                
                if sources:
                    st.markdown("---")
                    render_source_references(sources)
        
        st.session_state.messages.append({
            "role": "assistant", 
            "answer": answer,
            "sources": sources
        })

if __name__ == "__main__":
    main()