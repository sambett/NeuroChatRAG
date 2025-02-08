import os
import re
from tqdm import tqdm  # For progress bars
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Force CPU-only mode
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def clean_abstract(text):
    """Clean a single abstract."""
    # Remove markdown/URLs/references
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\[\d+\]', '', text)  # Remove [1], [2]
    text = re.sub(r'\s+', ' ', text)      # Collapse whitespace
    return text.strip()

def process_single_file(file_path):
    try:
        with open(file_path, 'r') as f:
            text = f.read()
        
        cleaned = clean_abstract(text)
        if len(cleaned) < 100:
            return []
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", ". "]
        )
        chunks = text_splitter.split_text(cleaned)
        
        return [{
            "text": chunk,
            "source": os.path.basename(file_path),
            "is_abstract": True
        } for chunk in chunks]
    
    except Exception as e:
        print(f"\nCRASHED ON {file_path}: {str(e)}\n")  # Explicit error logging
        return []

def process_all_abstracts(abstracts_dir="Parkinson_Resources/PubMed_Articles/Abstracts"):
    if not os.path.exists(abstracts_dir):
        raise FileNotFoundError(f"Directory {abstracts_dir} does not exist!")
    
    files = [os.path.join(abstracts_dir, f) for f in os.listdir(abstracts_dir) if f.endswith('.txt')]
    # files = files[:10]  # Already commented out to process all files.
    print(f"Found {len(files)} files. First 5: {files[:5]}")
    
    all_chunks = []
    # Loop through files with tqdm for a progress bar.
    for idx, file in enumerate(tqdm(files)):
        if not os.path.isfile(file):
            print(f"Warning: {file} is not a file!")
            continue
        
        chunks = process_single_file(file)
        all_chunks.extend(chunks)
        
        # Log every 100 files processed to ensure progress is visible.
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1} files out of {len(files)}")
    
    if all_chunks:
        print(f"Sample chunk: {all_chunks[0]}")
    return all_chunks, len(files)

# Execute the processing.
all_chunks, num_files = process_all_abstracts()
print(f"Processed {len(all_chunks)} chunks from {num_files} abstracts")

# ---------------------------------------------------------------
# Convert all_chunks to LangChain Documents and build the vector DB

from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

documents = [
    Document(
        page_content=chunk["text"],
        metadata={
            "source": chunk["source"],
            "is_abstract": chunk["is_abstract"]
        }
    )
    for chunk in all_chunks
]

# Build the vector DB.
embeddings = HuggingFaceEmbeddings(model_name="pritamdeka/S-PubMedBert-MS-MARCO")
db = FAISS.from_documents(documents, embeddings)
db.save_local("parkinsons_vector_db")
print("Vector database saved locally as 'parkinsons_vector_db'.")
